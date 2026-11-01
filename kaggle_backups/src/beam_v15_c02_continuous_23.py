
import hashlib
import json
import math
import time
from pathlib import Path
from scipy.optimize import differential_evolution
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

OUT = RESULTS / "beam_v15_c02_continuous_23_checkpoint.json"
SNAP = RESULTS / "beam_v15_c02_continuous_level_24.json"

P = TomatoProblem(str(ROOT / "data/easy.kttsp"))

START = time.monotonic()
LIMIT = 125.0

WAITS = [
    0, .25, .5, 1, 2, 4, 8, 12,
    18, 26, 36, 48, 64, 80, 100
]

TOFS = [
    .25, .5, .75, 1, 1.5, 2, 3, 5,
    8, 12, 18, 26, 36, 48, 64, 80
]

FILES = [
    "beam_v11_c02_to32_checkpoint.json",
    "beam_v11_c02_wide_checkpoint.json",
    "beam_v11_C02_level_23.json"
]


class Pause(Exception):
    pass


def budget():
    if time.monotonic() - START >= LIMIT:
        raise Pause()


def identity(obj):
    data = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":")
    ).encode()

    return hashlib.sha256(
        data
    ).hexdigest()[:24]


def save(path, data):
    temp = path.with_suffix(".tmp")

    temp.write_text(
        json.dumps(data, separators=(",", ":")),
        encoding="utf-8"
    )

    temp.replace(path)


def dv(a, b, departure, tof):
    try:
        value = float(
            P.compute_transfer(
                a, b, departure, tof
            )
        )

        return (
            value
            if math.isfinite(value)
            else float("inf")
        )

    except Exception:
        return float("inf")


def verify(s):
    try:
        route = s["route"]
        transfers = s["transfers"]

        if (
            not route
            or route[0] != 0
            or len(set(route)) != len(route)
            or len(transfers) != len(route) - 1
        ):
            return False

        day = 0.0
        specials = 0

        for k, tr in enumerate(transfers):
            depart = float(tr["start_time"])
            tof = float(tr["tof"])

            if (
                tr["from"] != route[k]
                or tr["to"] != route[k + 1]
                or depart < day - 1e-6
                or tof <= 0
                or depart + tof > P.max_time + 1e-6
            ):
                return False

            measured = dv(
                route[k],
                route[k + 1],
                depart,
                tof
            )

            if (
                not math.isfinite(measured)
                or abs(
                    measured - float(tr["dv"])
                ) > .1
            ):
                return False

            status = (
                "NORMAL"
                if measured < P.dv_threshold
                else "SPECIAL"
                if measured < P.exception_threshold
                else "INVALID"
            )

            if (
                status != tr["status"]
                or status == "INVALID"
            ):
                return False

            specials += status == "SPECIAL"
            day = depart + tof

        return (
            specials == s["special_used"]
            and specials <= P.n_exceptions
            and abs(
                day - float(s["current_time"])
            ) < 1e-5
        )

    except Exception:
        return False


# Nxjerrim itineraret e nivelit 23
# nga checkpoint-et e vjetra.
def collect(obj, found):
    if isinstance(obj, dict):

        needed = {
            "route",
            "transfers",
            "special_used",
            "current_time"
        }

        if needed <= set(obj):
            route = obj["route"]

            if (
                isinstance(route, list)
                and len(route) == 23
            ):
                found[identity(obj)] = obj

        for value in obj.values():
            if isinstance(value, (dict, list)):
                collect(value, found)

    elif isinstance(obj, list):
        for value in obj:
            if isinstance(value, (dict, list)):
                collect(value, found)


sources = {}
extracted = {}

for filename in FILES:
    path = RESULTS / filename

    if not path.exists():
        continue

    raw = path.read_bytes()

    sources[filename] = hashlib.sha256(
        raw
    ).hexdigest()

    collect(
        json.loads(raw),
        extracted
    )


print("FAZA 1 - RRUGET E DEGËS C02")
print("=" * 65, flush=True)

print(
    "Skedare burimore:",
    list(sources),
    flush=True
)

print(
    "Kandidate nivel23:",
    len(extracted),
    flush=True
)

if not extracted:
    raise RuntimeError(
        "Nuk u gjet rruge C02 me 23 domate. "
        "Dergo kete gabim per kontroll."
    )


verified = [
    s for s in extracted.values()
    if verify(s)
]

print(
    "Itinerare nivel23 te verifikuara:",
    len(verified),
    flush=True
)

assert verified, (
    "Asnje rruge e nxjerre "
    "nuk kaloi verifikimin PyKEP."
)


# Zgjedhim rrugë që kanë ende
# SPECIAL të papërdorura.
selected = []

for used, quota in [
    (3, 5),
    (4, 4),
    (5, 2)
]:
    group = sorted(
        (
            s for s in verified
            if s["special_used"] == used
        ),
        key=lambda s: s["current_time"]
    )

    ends = {}
    count = 0

    for s in group:
        end = s["route"][-1]

        if ends.get(end, 0) >= 3:
            continue

        selected.append(s)

        ends[end] = ends.get(end, 0) + 1
        count += 1

        if count >= quota:
            break


print(
    "Rruge te zgjedhura:",
    len(selected),
    flush=True
)

for used in [3, 4, 5]:
    print(
        f"SPECIAL {used}/5:",
        sum(
            s["special_used"] == used
            for s in selected
        ),
        flush=True
    )

assert selected, "Nuk ka pika fillestare."


CONFIG = {
    "hashes": sources,
    "ids": [
        identity(s) for s in selected
    ],
    "waits": WAITS,
    "tofs": TOFS,
    "version": 1
}

if OUT.exists():
    report = json.loads(
        OUT.read_text(encoding="utf-8")
    )

    assert report["config"] == CONFIG, (
        "Checkpoint ka konfigurim tjeter."
    )

else:
    report = {
        "config": CONFIG,
        "parents": {},
        "found24": {}
    }

    save(OUT, report)


def child(state, target, best):
    value = best["dv"]
    day = float(state["current_time"])

    status = (
        "NORMAL"
        if value < P.dv_threshold
        else "SPECIAL"
        if (
            value < P.exception_threshold
            and state["special_used"]
            < P.n_exceptions
        )
        else None
    )

    if status is None:
        return None

    depart = day + best["wait"]
    tof = best["tof"]

    candidate = {
        "route": state["route"] + [target],
        "transfers": state["transfers"] + [{
            "from": state["route"][-1],
            "to": target,
            "start_time": depart,
            "tof": tof,
            "dv": value,
            "status": status
        }],
        "current_time": depart + tof,
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


def grid(state, target):
    day = float(state["current_time"])
    maximum = P.max_time - day
    origin = state["route"][-1]

    best = None

    for wait in WAITS:
        budget()

        for tof in TOFS:
            if wait + tof > maximum:
                continue

            value = dv(
                origin,
                target,
                day + wait,
                tof
            )

            if (
                math.isfinite(value)
                and (
                    best is None
                    or value < best["dv"]
                )
            ):
                best = {
                    "dv": value,
                    "wait": wait,
                    "tof": tof
                }

    return best


def optimize(state, target, initial, trial):
    day = float(state["current_time"])
    span = P.max_time - day
    origin = state["route"][-1]

    if span <= .21:
        return initial

    def objective(x):
        budget()

        wait = (
            (span - .2) * float(x[0])
        )

        tof = (
            .2
            + (span - .2 - wait)
            * float(x[1])
        )

        value = dv(
            origin,
            target,
            day + wait,
            tof
        )

        return (
            value
            if math.isfinite(value)
            else 1e9
        )

    seed = (
        int(identity(state)[:8], 16)
        + target * 131
        + trial * 101
    ) % (2**32)

    result = differential_evolution(
        objective,
        [(0, 1), (0, 1)],
        seed=seed,
        popsize=7,
        maxiter=30,
        polish=False,
        tol=.01
    )

    wait = (
        (span - .2) * float(result.x[0])
    )

    tof = (
        .2
        + (span - .2 - wait)
        * float(result.x[1])
    )

    value = dv(
        origin,
        target,
        day + wait,
        tof
    )

    if (
        math.isfinite(value)
        and (
            initial is None
            or value < initial["dv"]
        )
    ):
        return {
            "dv": value,
            "wait": wait,
            "tof": tof
        }

    return initial


print(
    "\nFAZA 2 - KERKIMI C02, NIVELI 23 -> 24",
    flush=True
)

paused = False

try:
    for i, state in enumerate(selected, 1):
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
            x for x in range(P.n_tomatoes)
            if x not in state["route"]
        ]

        # Kontrollojme te gjitha destinacionet.
        for target in targets:
            key = str(target)

            if key in record["grid"]:
                continue

            best = grid(state, target)

            record["grid"][key] = best

            if best is not None:
                candidate = child(
                    state, target, best
                )

                if candidate:
                    report["found24"][
                        identity(candidate)
                    ] = candidate

            save(OUT, report)

        # Optimizojme 6 destinacionet
        # me te mira nga gridi.
        ranked = sorted(
            (
                (int(target), best)
                for target, best
                in record["grid"].items()
                if best is not None
            ),
            key=lambda item: item[1]["dv"]
        )[:6]

        for j, (target, best) in enumerate(ranked):
            trials = 2 if j < 2 else 1

            for attempt in range(trials):
                budget()

                key = f"{target}:{attempt}"

                if key in record["optimized"]:
                    continue

                result = optimize(
                    state,
                    target,
                    best,
                    attempt
                )

                record["optimized"][key] = result

                candidate = (
                    child(state, target, result)
                    if result is not None
                    else None
                )

                if candidate:
                    report["found24"][
                        identity(candidate)
                    ] = candidate

                save(OUT, report)

        record["complete"] = True
        save(OUT, report)

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
            f"[{i}/{len(selected)}] "
            f"Fund {state['route'][-1]} | "
            f"Dita {state['current_time']:.2f} | "
            f"SPECIAL {state['special_used']}/5 | "
            f"DV min {minimum:.2f} | "
            f"Rruge24 {len(report['found24'])}",
            flush=True
        )

except Pause:
    paused = True


if report["found24"]:
    save(
        SNAP,
        {
            "level": 24,
            "verified": True,
            "frontier": list(
                report["found24"].values()
            )
        }
    )


print("\nPERMBLEDHJA E HAPIT 136")
print("=" * 65, flush=True)

complete = sum(
    row["complete"]
    for row in report["parents"].values()
)

print(
    "Rruge fillestare te perfunduara:",
    complete, "/", len(selected),
    flush=True
)

print(
    "Destinacione ne grid:",
    sum(
        len(row["grid"])
        for row in report["parents"].values()
    ),
    flush=True
)

print(
    "Optimizime diferenciale:",
    sum(
        len(row["optimized"])
        for row in report["parents"].values()
    ),
    flush=True
)

print(
    "Rruge te verifikuara nivel24:",
    len(report["found24"]),
    flush=True
)


near = []
by_id = {
    identity(s): s for s in selected
}

for sid, row in report["parents"].items():
    state = by_id[sid]

    threshold = (
        P.exception_threshold
        if state["special_used"] < P.n_exceptions
        else P.dv_threshold
    )

    per_target = {}

    for target, value in row["grid"].items():
        if value is not None:
            per_target[target] = value

    for key, value in row["optimized"].items():
        target = key.split(":")[0]

        if (
            value is not None
            and (
                target not in per_target
                or value["dv"]
                < per_target[target]["dv"]
            )
        ):
            per_target[target] = value

    for target, value in per_target.items():
        near.append((
            value["dv"] - threshold,
            state["route"][-1],
            int(target),
            state["current_time"],
            state["special_used"],
            value["dv"],
            value["wait"],
            value["tof"]
        ))


print(
    "\n5 TRANSFERIMET ME AFER KUFIRIT",
    flush=True
)

for (
    gap, origin, target, day,
    used, value, wait, tof
) in sorted(near)[:5]:

    print(
        f"{origin}->{target} | "
        f"Dita {day:.2f} | "
        f"SPECIAL {used}/5 | "
        f"DV {value:.2f} | "
        f"Diferenca {gap:+.2f} | "
        f"Mberritja {day+wait+tof:.2f}",
        flush=True
    )


print(
    "Kerkimi i pjesshem:",
    paused or complete < len(selected),
    flush=True
)

print(
    "Rekordi aktual i projektit: 31/49",
    flush=True
)

print("HAPI 136 PERFUNDOI!", flush=True)
