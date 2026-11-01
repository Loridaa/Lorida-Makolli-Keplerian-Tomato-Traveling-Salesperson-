
import hashlib
import json
import math
import time
from pathlib import Path
from collections import defaultdict
from scipy.optimize import minimize
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

CHECK = RESULTS / "beam_v12_dv32_checkpoint.json"
SNAP32 = RESULTS / "beam_v12_dv32_verified.json"

P = TomatoProblem(str(ROOT / "data/easy.kttsp"))

START = time.monotonic()
SOFT_LIMIT = 125.0

WAITS = [
    0, .1, .25, .5, 1, 2, 3,
    5, 8, 12, 18, 26, 36, 48
]

TOFS = [
    .2, .25, .5, .75, 1, 1.5,
    2, 3, 5, 8, 13, 18, 24, 32, 48
]

FILES = [
    "beam_v11_A14_level_31.json",
    "beam_v11_B12_level_31.json",
    "beam_v12_retimed_level_31.json",
    "beam_v12_reordered_level_31.json",
    "beam_v12mix_A14_level_31.json",
    "beam_v12mix_B12_level_31.json"
]

class Pause(Exception):
    pass

def budget():
    if time.monotonic() - START >= SOFT_LIMIT:
        raise Pause()

def digest(obj):
    encoded = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":")
    ).encode()

    return hashlib.sha256(
        encoded
    ).hexdigest()[:24]

def atomic(path, payload):
    temp = path.with_suffix(".tmp")

    temp.write_text(
        json.dumps(payload, separators=(",", ":")),
        encoding="utf-8"
    )

    temp.replace(path)

def verify(state):
    try:
        route = state["route"]
        trs = state["transfers"]

        if (
            not route
            or route[0] != 0
            or len(route) != len(set(route))
            or len(trs) != len(route) - 1
        ):
            return False

        day = 0.0
        specials = 0

        for i, tr in enumerate(trs):
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

            dv = float(P.compute_transfer(
                route[i],
                route[i + 1],
                depart,
                tof
            ))

            if (
                not math.isfinite(dv)
                or abs(dv - tr["dv"]) > 0.1
            ):
                return False

            st = (
                "NORMAL"
                if dv < P.dv_threshold
                else "SPECIAL"
                if dv < P.exception_threshold
                else "INVALID"
            )

            if st != tr["status"]:
                return False

            specials += st == "SPECIAL"
            day = depart + tof

        return (
            specials == state["special_used"]
            and specials <= P.n_exceptions
            and abs(
                day - state["current_time"]
            ) < 1e-5
        )

    except Exception:
        return False

def child32(state, target, result):
    dv = result["dv"]
    wait = result["wait"]
    tof = result["tof"]
    used = state["special_used"]

    st = (
        "NORMAL"
        if dv < P.dv_threshold
        else "SPECIAL"
        if (
            dv < P.exception_threshold
            and used < P.n_exceptions
        )
        else None
    )

    if st is None:
        return None

    depart = state["current_time"] + wait

    transfer = {
        "from": state["route"][-1],
        "to": target,
        "start_time": depart,
        "tof": tof,
        "dv": dv,
        "status": st
    }

    candidate = {
        "route": state["route"] + [target],
        "transfers": (
            state["transfers"] + [transfer]
        ),
        "current_time": depart + tof,
        "special_used": (
            used + (st == "SPECIAL")
        )
    }

    return (
        candidate
        if verify(candidate)
        else None
    )

# FAZA 1 - Lexojme itineraret e ruajtura.
all_states = {}
hashes = {}

for filename in FILES:
    path = RESULTS / filename

    if not path.exists():
        continue

    raw = path.read_bytes()

    hashes[filename] = hashlib.sha256(
        raw
    ).hexdigest()

    records = json.loads(raw)["frontier"]

    for state in records:
        if (
            len(state["route"]) != 31
            or state["special_used"] not in (4, 5)
        ):
            continue

        key = (
            tuple(sorted(state["route"])),
            state["route"][-1],
            round(state["current_time"], 5),
            state["special_used"]
        )

        if key not in all_states:
            all_states[key] = state

assert all_states, (
    "Nuk u gjet asnje itinerar "
    "me 31 domate."
)

by_type = defaultdict(list)

for state in all_states.values():
    by_type[(
        state["special_used"],
        state["route"][-1]
    )].append(state)

selected = []

# Perzgjedhim rruge me destinacione
# perfundimtare te ndryshme.
for used, quota in [(4, 6), (5, 2)]:
    ends = sorted([
        end
        for special, end in by_type
        if special == used
    ])

    for end in ends:
        group = sorted(
            by_type[(used, end)],
            key=lambda s: s["current_time"]
        )

        selected.extend(group[:quota])

selected = sorted(
    selected,
    key=lambda s: (
        s["special_used"],
        s["current_time"]
    )
)[:32]

for state in selected:
    assert verify(state), (
        "U gjet nje rruge e pavlefshme "
        "ne skedaret e nivelit 31."
    )

CONFIG = {
    "hashes": hashes,
    "selected_ids": [
        digest(s) for s in selected
    ],
    "waits": WAITS,
    "tofs": TOFS,
    "version": 1
}

if CHECK.exists():
    report = json.loads(CHECK.read_text())

    assert report["config"] == CONFIG, (
        "Checkpoint ka konfigurim tjeter."
    )

else:
    report = {
        "config": CONFIG,
        "analyses": {},
        "found32": {}
    }

    atomic(CHECK, report)

def save():
    atomic(CHECK, report)

# FAZA 2 - Kerkojme Delta-V minimale
# per secilen domate te pavizituar.
def test_target(state, target):
    remainder = (
        P.max_time - state["current_time"]
    )

    start = state["current_time"]
    end = state["route"][-1]

    best = None
    tested = 0

    for wait in WAITS:
        for tof in TOFS:
            budget()

            if wait + tof > remainder + 1e-8:
                continue

            try:
                dv = float(P.compute_transfer(
                    end,
                    target,
                    start + wait,
                    tof
                ))

                tested += 1

            except Exception:
                continue

            if not math.isfinite(dv):
                continue

            if (
                best is None
                or dv < best["dv"]
            ):
                best = {
                    "dv": dv,
                    "wait": wait,
                    "tof": tof
                }

    return {
        "best": best,
        "tested": tested
    }

# FAZA 3 - Perpiqemi ta ulim Delta-V
# me optimizim lokal.
def improve(state, target, guess):
    remaining = (
        P.max_time - state["current_time"]
    )

    end = state["route"][-1]

    def objective(x):
        budget()

        wait, tof = map(float, x)

        if (
            wait < 0
            or tof < .2
            or wait + tof > remaining
        ):
            return 1e8

        try:
            dv = float(P.compute_transfer(
                end,
                target,
                state["current_time"] + wait,
                tof
            ))

        except Exception:
            return 1e8

        return (
            dv
            if math.isfinite(dv)
            else 1e8
        )

    result = minimize(
        objective,
        [guess["wait"], guess["tof"]],
        method="Nelder-Mead",
        options={
            "maxfev": 65,
            "maxiter": 50,
            "xatol": 1e-3,
            "fatol": .05
        }
    )

    wait, tof = map(float, result.x)
    dv = objective([wait, tof])

    if dv < guess["dv"]:
        return {
            "dv": dv,
            "wait": wait,
            "tof": tof
        }

    return guess

print("FAZA 1 - DIAGNOSTIKIMI I NIVELIT 31")
print("=" * 65, flush=True)

print(
    "Skedare burimore:", len(hashes),
    "| Itinerare unike:", len(all_states),
    "| Te zgjedhura:", len(selected),
    flush=True
)

print(
    "Me 4 SPECIAL:",
    sum(s["special_used"] == 4 for s in selected),
    "| Me 5 SPECIAL:",
    sum(s["special_used"] == 5 for s in selected),
    flush=True
)

paused = False

try:
    for index, state in enumerate(selected, 1):
        budget()

        sid = digest(state)

        record = report["analyses"].setdefault(
            sid,
            {
                "targets": {},
                "refined": {},
                "completed": False
            }
        )

        remaining_targets = [
            x for x in range(P.n_tomatoes)
            if x not in state["route"]
        ]

        for target in remaining_targets:
            budget()

            key = str(target)

            if key in record["targets"]:
                continue

            value = test_target(
                state, target
            )

            record["targets"][key] = value

            if value["best"] is not None:
                candidate = child32(
                    state, target, value["best"]
                )

                if candidate:
                    report["found32"][
                        digest(candidate)
                    ] = candidate

            save()

        ranked = sorted(
            (
                (int(key), row["best"])
                for key, row
                in record["targets"].items()
                if row["best"] is not None
            ),
            key=lambda item: item[1]["dv"]
        )

        for target, guess in ranked[:3]:
            key = str(target)

            if key in record["refined"]:
                continue

            improved = improve(
                state, target, guess
            )

            record["refined"][key] = improved

            candidate = child32(
                state, target, improved
            )

            if candidate:
                report["found32"][
                    digest(candidate)
                ] = candidate

            save()

        record["completed"] = True
        save()

        minimum = (
            ranked[0][1]["dv"]
            if ranked
            else float("inf")
        )

        limit = (
            P.exception_threshold
            if state["special_used"] < P.n_exceptions
            else P.dv_threshold
        )

        print(
            f"[{index}/{len(selected)}] "
            f"Fund {state['route'][-1]} | "
            f"Dita {state['current_time']:.2f} | "
            f"SPECIAL {state['special_used']}/5 | "
            f"DV min {minimum:.2f} | "
            f"Kufiri {limit:.0f} | "
            f"Rruge32 {len(report['found32'])}",
            flush=True
        )

except Pause:
    paused = True

# FAZA 4 - Permbledhja.
print("\nPERMBLEDHJA E HAPIT 129")
print("=" * 65, flush=True)

complete = sum(
    record["completed"]
    for record in report["analyses"].values()
)

checks = sum(
    len(record["targets"])
    for record in report["analyses"].values()
)

print(
    f"Itinerare te analizuara: "
    f"{complete}/{len(selected)}",
    flush=True
)

print(
    "Transferime destinacioni te kontrolluara:",
    checks,
    flush=True
)

print(
    "Itinerare te verifikuara me 32 domate:",
    len(report["found32"]),
    flush=True
)

ranked_gaps = []

source_by_id = {
    digest(s): s for s in selected
}

for sid, entry in report["analyses"].items():
    state = source_by_id[sid]

    threshold = (
        P.exception_threshold
        if state["special_used"] < P.n_exceptions
        else P.dv_threshold
    )

    for target, row in entry["targets"].items():
        best = (
            entry["refined"].get(target)
            or row["best"]
        )

        if best is not None:
            ranked_gaps.append((
                best["dv"] - threshold,
                state,
                int(target),
                best,
                threshold
            ))

ranked_gaps.sort(
    key=lambda item: item[0]
)

print("\n5 TRANSFERIMET ME AFER KUFIRIT")
print("=" * 65, flush=True)

for gap, state, target, best, threshold in ranked_gaps[:5]:
    print(
        f"Fund {state['route'][-1]} -> {target} | "
        f"Dita {state['current_time']:.2f} | "
        f"SPECIAL {state['special_used']}/5 | "
        f"DV {best['dv']:.2f} | "
        f"Kufiri {threshold:.0f} | "
        f"Diferenca {gap:+.2f}",
        flush=True
    )

if report["found32"]:
    atomic(
        SNAP32,
        {
            "level": 32,
            "verified": True,
            "frontier": list(
                report["found32"].values()
            )
        }
    )

    print(
        "\nREKORD I RI I VERIFIKUAR: 32/49!",
        flush=True
    )

else:
    print(
        "\nRekordi mbetet 31/49.",
        flush=True
    )

print(
    "Kerkimi i pjesshem:",
    paused or complete < len(selected),
    flush=True
)

print("\nHAPI 129 PERFUNDOI!", flush=True)
