
import json
import math
import time
import hashlib
from pathlib import Path
from scipy.optimize import differential_evolution
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "beam_v10_bypass_special5.json"

P = TomatoProblem(str(ROOT / "data/easy.kttsp"))

SOURCES = [
    ("C19", 26, 3),
    ("C19", 25, 3),
    ("C19", 24, 4),
    ("V9", 26, 1)
]

WAITS = [0, 0.25, 0.5, 1, 2, 3, 5, 8, 13, 20, 28]
TOFS = [0.5, 1, 1.5, 2, 3, 5, 8, 13, 18, 24, 32, 40]

MAX_WINDOW = 45.0
WORK_LIMIT = 140.0
START = time.monotonic()

def signature(state):
    content = [
        state["route"],
        round(float(state["current_time"]), 7),
        int(state["special_used"])
    ]
    return hashlib.sha256(
        json.dumps(content).encode()
    ).hexdigest()[:20]

def verify(state):
    route = state["route"]
    transfers = state["transfers"]

    if (
        route[0] != 0
        or len(route) != len(set(route))
        or len(transfers) != len(route) - 1
        or state["special_used"] != 4
    ):
        return False

    day = 0.0
    specials = 0

    try:
        for i, tr in enumerate(transfers):
            start = float(tr["start_time"])
            tof = float(tr["tof"])

            if (
                tr["from"] != route[i]
                or tr["to"] != route[i + 1]
                or start < day - 1e-6
                or tof <= 0
                or start + tof > P.max_time
            ):
                return False

            dv = float(P.compute_transfer(
                route[i], route[i + 1], start, tof
            ))

            if (
                not math.isfinite(dv)
                or abs(dv - float(tr["dv"])) > 0.1
            ):
                return False

            if dv < P.dv_threshold:
                status = "NORMAL"
            elif dv < P.exception_threshold:
                status = "SPECIAL"
                specials += 1
            else:
                return False

            if status != tr["status"]:
                return False

            day = start + tof

    except Exception:
        return False

    return (
        abs(day - state["current_time"]) < 1e-5
        and specials == 4
        and day <= P.max_time
    )

def save():
    tmp = OUTPUT.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(report, separators=(",", ":"))
    )
    tmp.replace(OUTPUT)

def better(old, new):
    if old is None:
        return new

    if (new["arrival"], new["dv"]) < (
        old["arrival"], old["dv"]
    ):
        return new

    return old

# FAZA 1 - Rikuperojme rruget.
chosen = []
seen = set()
source_hashes = {}

print("FAZA 1 - RIKUPERIMI I RRUGEVE")
print("=" * 60, flush=True)

for name, level, quota in SOURCES:

    if name == "C19":
        path = RESULTS / f"beam_v10_saved_C19_level_{level}.json"
    else:
        path = RESULTS / f"beam_v9_auto_level_{level}.json"

    raw = path.read_bytes()
    source_hashes[path.name] = hashlib.sha256(raw).hexdigest()

    states = json.loads(raw)["frontier"]

    group = sorted(
        [s for s in states if s["special_used"] == 4],
        key=lambda s: s["current_time"]
    )

    picks = []
    picked = set()

    # Preferojme funde te ndryshme dhe
    # vecanerisht domatet 36 dhe 31.
    for end in [36, 31, 47, 18, 23, 34, 37]:
        for state in group:
            sid = signature(state)

            if (
                state["route"][-1] == end
                and sid not in picked
                and sid not in seen
            ):
                picks.append(state)
                picked.add(sid)
                break

        if len(picks) >= quota:
            break

    for state in group:
        if len(picks) >= quota:
            break

        sid = signature(state)

        if sid not in picked and sid not in seen:
            picks.append(state)
            picked.add(sid)

    for state in picks:
        sid = signature(state)
        seen.add(sid)

        chosen.append({
            "source": path.name,
            "state": state,
            "id": sid
        })

    print(
        f"{path.name}: {len(picks)} rruge te zgjedhura",
        flush=True
    )

CONFIG = {
    "source_hashes": source_hashes,
    "selected_ids": [s["id"] for s in chosen],
    "waits": WAITS,
    "tofs": TOFS,
    "max_window": MAX_WINDOW,
    "optimization_top": 3,
    "maxiter": 16,
    "popsize": 6,
    "version": 1
}

if OUTPUT.exists():
    report = json.loads(OUTPUT.read_text())

    assert report["config"] == CONFIG, (
        "Checkpoint ekzistues ka konfigurim tjeter."
    )
else:
    report = {
        "config": CONFIG,
        "branches": {},
        "normal_children": []
    }

# FAZA 2 - Kerkimi NORMAL pa shpenzuar SPECIAL.
print("\nFAZA 2 - KERKIMI I ALTERNATIVAVE")
print("=" * 60, flush=True)

for index, item in enumerate(chosen, 1):

    if time.monotonic() - START > WORK_LIMIT:
        print("Kufiri i punes u arrit.", flush=True)
        break

    state = item["state"]
    sid = item["id"]

    if not verify(state):
        print("Rruge e pavlefshme:", sid, flush=True)
        continue

    if sid not in report["branches"]:
        report["branches"][sid] = {
            "source": item["source"],
            "state": state,
            "targets": {},
            "complete": False
        }
        save()

    branch = report["branches"][sid]

    if branch["complete"]:
        print(
            f"[{index}/{len(chosen)}] Nga checkpoint",
            flush=True
        )
        continue

    route = state["route"]
    current = route[-1]
    day = float(state["current_time"])

    remaining = [
        t for t in range(P.n_tomatoes)
        if t not in route
    ]

    window = min(MAX_WINDOW, P.max_time - day)

    print(
        f"\n[{index}/{len(chosen)}] "
        f"Niveli {len(route)}/49 | "
        f"Fund {current} | "
        f"Dita {day:.2f} | "
        f"SPECIAL 4/5",
        flush=True
    )

    # Grid: kontrollojme te gjitha domatet
    # e pavizituara me kombinime kohore.
    for target in remaining:

        if str(target) in branch["targets"]:
            continue

        if time.monotonic() - START > WORK_LIMIT:
            break

        minimum = float("inf")
        best_normal = None

        for wait in WAITS:
            for tof in TOFS:

                if (
                    wait + tof > window
                    or day + wait + tof > P.max_time
                ):
                    continue

                try:
                    dv = float(P.compute_transfer(
                        current, target, day + wait, tof
                    ))
                except Exception:
                    continue

                if not math.isfinite(dv):
                    continue

                minimum = min(minimum, dv)

                if dv < P.dv_threshold:
                    move = {
                        "from": current,
                        "to": target,
                        "start_time": day + wait,
                        "tof": tof,
                        "arrival": day + wait + tof,
                        "dv": dv,
                        "status": "NORMAL",
                        "method": "grid"
                    }

                    best_normal = better(
                        best_normal, move
                    )

        branch["targets"][str(target)] = {
            "grid_min": (
                minimum if math.isfinite(minimum) else None
            ),
            "best_normal": best_normal,
            "opt_done": False,
            "opt_min": None
        }

        save()

    if len(branch["targets"]) < len(remaining):
        print("Grid i pjesshem. Rifillo qelizen.", flush=True)
        break

    grid_normals = sum(
        entry["best_normal"] is not None
        for entry in branch["targets"].values()
    )

    print(
        f"Grid: {len(remaining)} destinacione | "
        f"NORMAL {grid_normals}",
        flush=True
    )

    ranked = sorted(
        (
            (entry["grid_min"], int(target))
            for target, entry in branch["targets"].items()
            if entry["grid_min"] is not None
        )
    )

    optimize_targets = [
        target for _, target in ranked[:3]
    ]

    print(
        "Destinacionet per optimizim:",
        optimize_targets,
        flush=True
    )

    # Differential Evolution:
    # provojme kohe edhe mes pikave te grid-it.
    for target in optimize_targets:

        if time.monotonic() - START > WORK_LIMIT:
            break

        entry = branch["targets"][str(target)]

        if entry["opt_done"]:
            continue

        best_dv = float("inf")
        max_wait = min(20.0, window - 0.25)

        if max_wait <= 0:
            entry["opt_done"] = True
            save()
            continue

        def decode(x):
            wait = max_wait * float(x[0])
            tof = (
                0.25
                + (window - wait - 0.25) * float(x[1])
            )
            return wait, tof

        def objective(x):
            wait, tof = decode(x)

            try:
                dv = float(P.compute_transfer(
                    current, target, day + wait, tof
                ))
            except Exception:
                return 1e8

            return dv if math.isfinite(dv) else 1e8

        for seed in [2026, 2047]:

            try:
                result = differential_evolution(
                    objective,
                    bounds=[(0, 1), (0, 1)],
                    seed=seed + target + len(route) * 100,
                    maxiter=16,
                    popsize=6,
                    polish=False,
                    workers=1
                )

                wait, tof = decode(result.x)

                dv = float(P.compute_transfer(
                    current, target, day + wait, tof
                ))

            except Exception:
                continue

            if not math.isfinite(dv):
                continue

            best_dv = min(best_dv, dv)

            if dv < P.dv_threshold:
                move = {
                    "from": current,
                    "to": target,
                    "start_time": day + wait,
                    "tof": tof,
                    "arrival": day + wait + tof,
                    "dv": dv,
                    "status": "NORMAL",
                    "method": "differential_evolution"
                }

                entry["best_normal"] = better(
                    entry["best_normal"], move
                )

        entry["opt_done"] = True

        entry["opt_min"] = (
            best_dv if math.isfinite(best_dv) else None
        )

        save()

        print(
            f"  {current} -> {target} | "
            f"Delta-V minimale: "
            f"{best_dv:.2f}",
            flush=True
        )

    if not all(
        branch["targets"][str(t)]["opt_done"]
        for t in optimize_targets
    ):
        print(
            "Optimizim i pjesshem. Rifillo qelizen.",
            flush=True
        )
        break

    branch["complete"] = True
    save()

    normals = [
        entry["best_normal"]
        for entry in branch["targets"].values()
        if entry["best_normal"] is not None
    ]

    print(
        f"REZULTATI: {len(normals)} "
        f"destinacione NORMAL",
        flush=True
    )

    for move in sorted(
        normals,
        key=lambda m: (m["arrival"], m["dv"])
    )[:5]:
        print(
            f"  {move['from']} -> {move['to']} "
            f"| Dita {move['arrival']:.2f} "
            f"| DV {move['dv']:.2f}",
            flush=True
        )

# FAZA 3 - Ndertojme rruget e reja me
# 4 SPECIAL, pa e shpenzuar te pesten.
children = {}
completed = 0

for branch in report["branches"].values():

    if branch["complete"]:
        completed += 1

    state = branch["state"]

    for entry in branch["targets"].values():
        move = entry["best_normal"]

        if move is None:
            continue

        transfer = {
            key: move[key]
            for key in [
                "from", "to", "start_time",
                "tof", "dv", "status"
            ]
        }

        child = {
            "route": state["route"] + [move["to"]],
            "current_time": move["arrival"],
            "special_used": 4,
            "transfers": state["transfers"] + [transfer]
        }

        key = signature(child)
        children[key] = child

report["normal_children"] = list(children.values())
save()

print("\nREZULTATI I HAPIT 118")
print("=" * 60, flush=True)

print(
    "Rruge te perfunduara:",
    completed, "/", len(chosen),
    flush=True
)

print(
    "Vazhdime NORMAL te ruajtura:",
    len(children),
    flush=True
)

by_level = {}

for child in children.values():
    level = len(child["route"])
    by_level.setdefault(level, []).append(child)

for level in sorted(by_level):

    group = by_level[level]

    print(
        f"Niveli {level}/49 | "
        f"{len(group)} rruge me 4 SPECIAL",
        flush=True
    )

    for state in sorted(
        group,
        key=lambda s: s["current_time"]
    )[:3]:

        print(
            f"  Fund {state['route'][-1]} | "
            f"Dita {state['current_time']:.2f} | "
            f"SPECIAL {state['special_used']}/5",
            flush=True
        )

print("\nHAPI 118 - KERKIMI PERFUNDOI!", flush=True)
