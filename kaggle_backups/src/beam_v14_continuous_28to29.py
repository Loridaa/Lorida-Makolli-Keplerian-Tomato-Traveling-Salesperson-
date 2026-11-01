
import json
import hashlib
import math
import time
from pathlib import Path
from scipy.optimize import differential_evolution
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_v13_continue_level_28.json"
CHECKPOINT = RESULTS / "beam_v14_continuous_28to29_checkpoint.json"
SNAPSHOT = RESULTS / "beam_v14_continuous_level_29.json"

P = TomatoProblem(str(ROOT / "data/easy.kttsp"))

START = time.monotonic()
LIMIT = 125

WAITS = [
    0, .1, .25, .5, 1, 2, 3,
    5, 8, 12, 18, 26, 36
]

TOFS = [
    .2, .25, .5, .75, 1, 1.5,
    2, 3, 5, 8, 13, 18, 24, 30, 36
]

N_OPT = 4


class Pause(Exception):
    pass


def budget():
    if time.monotonic() - START >= LIMIT:
        raise Pause()


def identity(obj):
    content = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":")
    )

    return hashlib.sha256(
        content.encode()
    ).hexdigest()[:24]


def atomic(path, obj):
    temp = path.with_suffix(".tmp")

    temp.write_text(
        json.dumps(obj, separators=(",", ":")),
        encoding="utf-8"
    )

    temp.replace(path)


def dv(origin, target, depart, tof):
    try:
        value = float(P.compute_transfer(
            origin, target, depart, tof
        ))

        return (
            value
            if math.isfinite(value)
            else float("inf")
        )

    except Exception:
        return float("inf")


def verify(state):
    route = state["route"]
    transfers = state["transfers"]

    if (
        not route
        or route[0] != 0
        or len(route) != len(set(route))
        or len(transfers) != len(route) - 1
    ):
        return False

    used = 0
    day = 0.0

    for i, tr in enumerate(transfers):
        depart = float(tr["start_time"])
        tof = float(tr["tof"])

        if (
            tr["from"] != route[i]
            or tr["to"] != route[i + 1]
            or depart < day - 1e-6
            or tof <= 0
            or depart + tof > P.max_time + 1e-6
        ):
            return False

        measured = dv(
            route[i], route[i + 1],
            depart, tof
        )

        if abs(measured - float(tr["dv"])) > .1:
            return False

        status = (
            "NORMAL"
            if measured < P.dv_threshold
            else "SPECIAL"
            if measured < P.exception_threshold
            else "INVALID"
        )

        if status != tr["status"]:
            return False

        used += status == "SPECIAL"
        day = depart + tof

    return (
        used == state["special_used"]
        and used <= P.n_exceptions
        and abs(
            day - float(state["current_time"])
        ) < 1e-5
    )


def make_child(state, target, best):
    value = best["dv"]

    status = (
        "NORMAL"
        if value < P.dv_threshold
        else "SPECIAL"
        if (
            value < P.exception_threshold
            and state["special_used"] < P.n_exceptions
        )
        else "INVALID"
    )

    if status == "INVALID":
        return None

    depart = (
        float(state["current_time"])
        + best["wait"]
    )

    transfer = {
        "from": state["route"][-1],
        "to": target,
        "start_time": depart,
        "tof": best["tof"],
        "dv": value,
        "status": status
    }

    candidate = {
        "route": state["route"] + [target],
        "transfers": (
            state["transfers"] + [transfer]
        ),
        "current_time": depart + best["tof"],
        "special_used": (
            state["special_used"]
            + int(status == "SPECIAL")
        )
    }

    return (
        candidate
        if verify(candidate)
        else None
    )


raw = SOURCE.read_bytes()

source_hash = hashlib.sha256(
    raw
).hexdigest()

states = sorted(
    [
        s
        for s in json.loads(raw)["frontier"]
        if len(s["route"]) == 28
        and s["special_used"] == 3
    ],
    key=lambda s: s["current_time"]
)

assert states, "Nuk u gjeten rruge me 3 SPECIAL."

assert all(
    verify(s) for s in states
), "Rruga fillestare deshtoi ne verifikim."

config = {
    "source_hash": source_hash,
    "ids": [identity(s) for s in states],
    "waits": WAITS,
    "tofs": TOFS,
    "n_opt": N_OPT,
    "version": 1
}

if CHECKPOINT.exists():
    report = json.loads(
        CHECKPOINT.read_text(encoding="utf-8")
    )

    assert report["config"] == config, (
        "Checkpoint me konfigurim tjeter."
    )

else:
    report = {
        "config": config,
        "parents": {},
        "found": {}
    }

    atomic(CHECKPOINT, report)


def save():
    atomic(CHECKPOINT, report)


def sample_grid(state, target):
    day = float(state["current_time"])
    remaining = P.max_time - day
    origin = state["route"][-1]

    best = None

    for wait in WAITS:
        budget()

        for tof in TOFS:
            if wait + tof > remaining:
                continue

            value = dv(
                origin,
                target,
                day + wait,
                tof
            )

            if best is None or value < best["dv"]:
                best = {
                    "dv": value,
                    "wait": wait,
                    "tof": tof
                }

    return best


def optimize(state, target, guess):
    day = float(state["current_time"])
    span = P.max_time - day

    if span <= .21:
        return guess

    origin = state["route"][-1]

    def objective(x):
        budget()

        wait = (span - .2) * float(x[0])

        tof = (
            .2
            + (span - .2 - wait) * float(x[1])
        )

        return dv(
            origin, target,
            day + wait, tof
        )

    seed = (
        int(identity(state)[:8], 16) + target
    ) % (2**32)

    try:
        fitted = differential_evolution(
            objective,
            [(0, 1), (0, 1)],
            seed=seed,
            maxiter=30,
            popsize=7,
            tol=.02,
            polish=False
        )

    except Pause:
        raise

    except Exception:
        return guess

    wait = (
        (span - .2) * float(fitted.x[0])
    )

    tof = (
        .2
        + (span - .2 - wait) * float(fitted.x[1])
    )

    value = dv(
        origin, target,
        day + wait, tof
    )

    if (
        math.isfinite(value)
        and (
            guess is None
            or value < guess["dv"]
        )
    ):
        return {
            "dv": value,
            "wait": wait,
            "tof": tof
        }

    return guess


print("FAZA 1 - KONTROLLI I RRUGEVE")
print("=" * 65, flush=True)

print(
    "Rruge me 28 domate dhe 3/5 SPECIAL:",
    len(states),
    flush=True
)

print("\nFAZA 2 - OPTIMIZIMI DELTA-V")
print("=" * 65, flush=True)

paused = False

try:
    for i, state in enumerate(states, 1):
        budget()

        sid = identity(state)

        record = report["parents"].setdefault(
            sid,
            {
                "grid": {},
                "optimized": {},
                "complete": False
            }
        )

        targets = [
            target
            for target in range(P.n_tomatoes)
            if target not in state["route"]
        ]

        # Kontrolli fillestar ne grid.
        for target in targets:
            budget()

            key = str(target)

            if key in record["grid"]:
                continue

            sample = sample_grid(
                state, target
            )

            record["grid"][key] = sample

            if (
                sample
                and sample["dv"] < P.exception_threshold
            ):
                candidate = make_child(
                    state, target, sample
                )

                if candidate is not None:
                    report["found"][
                        identity(candidate)
                    ] = candidate

            save()

        # Optimizimi global per 4
        # destinacionet me premtuese.
        ranked = sorted(
            [
                (int(target), value)
                for target, value
                in record["grid"].items()
                if value is not None
                and math.isfinite(value["dv"])
            ],
            key=lambda item: item[1]["dv"]
        )[:N_OPT]

        for target, guess in ranked:
            budget()

            key = str(target)

            if key in record["optimized"]:
                continue

            best = optimize(
                state, target, guess
            )

            record["optimized"][key] = best

            if (
                best
                and best["dv"] < P.exception_threshold
            ):
                candidate = make_child(
                    state, target, best
                )

                if candidate is not None:
                    report["found"][
                        identity(candidate)
                    ] = candidate

            save()

        record["complete"] = True
        save()

        values = [
            v["dv"]
            for v in (
                list(record["grid"].values())
                + list(record["optimized"].values())
            )
            if v is not None
        ]

        minimum = min(
            values,
            default=float("inf")
        )

        print(
            f"[{i}/{len(states)}] "
            f"Fund {state['route'][-1]} | "
            f"Dita {state['current_time']:.2f} | "
            f"DV minimale {minimum:.2f} | "
            f"Kufiri 600 | "
            f"Rruge29 {len(report['found'])}",
            flush=True
        )

except Pause:
    paused = True


if report["found"]:
    atomic(
        SNAPSHOT,
        {
            "level": 29,
            "verified": True,
            "frontier": list(
                report["found"].values()
            )
        }
    )


print("\nPERMBLEDHJA E HAPIT 134")
print("=" * 65, flush=True)

completed = sum(
    p["complete"]
    for p in report["parents"].values()
)

grid_total = sum(
    len(p["grid"])
    for p in report["parents"].values()
)

optim_total = sum(
    len(p["optimized"])
    for p in report["parents"].values()
)

print(
    f"Rruge te perfunduara: "
    f"{completed}/{len(states)}",
    flush=True
)

print(
    "Transferime destinacionesh ne grid:",
    grid_total,
    flush=True
)

print(
    "Optimizime diferenciale:",
    optim_total,
    flush=True
)

print(
    "Rruge me 29 domate "
    "dhe maksimum 4 SPECIAL:",
    len(report["found"]),
    flush=True
)

near_gaps = []

for state in states:
    sid = identity(state)

    entry = report["parents"].get(sid)

    if not entry:
        continue

    for key, grid_value in entry["grid"].items():
        best = entry["optimized"].get(
            key, grid_value
        )

        if (
            best
            and math.isfinite(best["dv"])
        ):
            near_gaps.append((
                best["dv"],
                state["route"][-1],
                int(key),
                state["current_time"],
                best
            ))

near_gaps.sort(
    key=lambda item: item[0]
)

print(
    "\n5 TRANSFERIMET ME DELTA-V ME TE ULET:",
    flush=True
)

for value, origin, target, day, best in near_gaps[:5]:
    print(
        f"{origin} -> {target} | "
        f"Dita {day:.2f} | "
        f"DV {value:.2f} | "
        f"Diferenca nga 600: {value-600:+.2f} | "
        f"Pritje {best['wait']:.2f} | "
        f"TOF {best['tof']:.2f}",
        flush=True
    )

print(
    "Kerkimi i pjesshem:",
    paused or completed < len(states),
    flush=True
)

if report["found"]:
    print(
        "U GJETEN KANDIDATE TE RINJ "
        "ME SPECIAL TE LIRA!",
        flush=True
    )

print(
    "Rekordi ekzistues mbetet 31/49 "
    "deri ne verifikimin e nivelit 32.",
    flush=True
)

print("\nHAPI 134 PERFUNDOI!", flush=True)
