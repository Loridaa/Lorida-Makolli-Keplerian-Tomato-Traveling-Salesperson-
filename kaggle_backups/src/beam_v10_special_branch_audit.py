
import json
import math
import time
import hashlib
from pathlib import Path
from scipy.optimize import differential_evolution
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_v10_saved_C19_level_29.json"
OUTPUT = RESULTS / "beam_v10_special_branch_audit.json"

P = TomatoProblem(str(ROOT / "data/easy.kttsp"))

WAITS = [0, 0.5, 1, 2, 3, 5, 8, 12, 18]
TOFS = [0.5, 1, 1.5, 2, 3, 5, 8, 13, 18, 25, 32]

MAX_WINDOW = 40.0
WORK_LIMIT = 135.0
START = time.monotonic()

source_bytes = SOURCE.read_bytes()
states = json.loads(source_bytes)["frontier"]

# Zgjedhim itinerarin me 29 domate me mberritjen me te hershme.
state = min(
    [s for s in states if len(s["route"]) == 29],
    key=lambda s: s["current_time"]
)

route = state["route"]
transfers = state["transfers"]

def verify_full():
    assert len(route) == 29
    assert len(set(route)) == 29
    assert len(transfers) == 28
    assert route[0] == 0

    day = 0.0
    specials = 0

    for i, tr in enumerate(transfers):
        depart = float(tr["start_time"])
        tof = float(tr["tof"])

        assert tr["from"] == route[i]
        assert tr["to"] == route[i + 1]
        assert depart >= day - 1e-6
        assert tof > 0
        assert depart + tof <= P.max_time

        dv = float(P.compute_transfer(
            route[i], route[i + 1], depart, tof
        ))

        assert math.isfinite(dv)
        assert abs(dv - float(tr["dv"])) < 0.1

        if dv < P.dv_threshold:
            status = "NORMAL"
        elif dv < P.exception_threshold:
            status = "SPECIAL"
            specials += 1
        else:
            raise ValueError(f"Transferim i pavlefshem: {i+1}")

        assert tr["status"] == status
        day = depart + tof

    assert specials == state["special_used"]
    assert specials == 5
    assert abs(day - state["current_time"]) < 1e-5

verify_full()

special_steps = [
    i + 1
    for i, tr in enumerate(transfers)
    if tr["status"] == "SPECIAL"
]

# Te fundit tashme e kemi analizuar ne Hapat 114 dhe 118.
steps_to_check = sorted(special_steps[:-1], reverse=True)

CONFIG = {
    "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
    "steps": steps_to_check,
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
        "Checkpoint nuk perputhet me konfigurimin."
    )
else:
    report = {
        "config": CONFIG,
        "original_route": route,
        "original_day": state["current_time"],
        "original_special_steps": special_steps,
        "branches": {},
        "candidate_prefixes": []
    }

def save():
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(report, separators=(",", ":"))
    )
    temporary.replace(OUTPUT)

def better(old, new):
    if old is None:
        return new

    if (new["arrival"], new["dv"]) < (
        old["arrival"], old["dv"]
    ):
        return new

    return old

print("FAZA 1 - ANALIZA E ITINERARIT")
print("=" * 65, flush=True)
print("Domate:", len(route), flush=True)
print("Dita:", round(state["current_time"], 3), flush=True)
print("SPECIAL:", state["special_used"], "/5", flush=True)
print("Transferimet SPECIAL:", special_steps, flush=True)
print("Do te analizohen:", steps_to_check, flush=True)

for step in steps_to_check:

    if time.monotonic() - START >= WORK_LIMIT:
        print("U arrit kufiri i punes.", flush=True)
        break

    key = str(step)

    # Prefix = rruga PERPARA manovres SPECIAL.
    prefix_route = route[:step]
    prefix_transfers = transfers[:step - 1]

    day = (
        float(prefix_transfers[-1]["start_time"])
        + float(prefix_transfers[-1]["tof"])
        if prefix_transfers else 0.0
    )

    used = sum(
        tr["status"] == "SPECIAL"
        for tr in prefix_transfers
    )

    current = prefix_route[-1]
    original_target = route[step]

    remaining = [
        target
        for target in range(P.n_tomatoes)
        if target not in prefix_route
    ]

    window = min(MAX_WINDOW, P.max_time - day)

    if key not in report["branches"]:
        report["branches"][key] = {
            "step": step,
            "prefix_route": prefix_route,
            "prefix_transfers": prefix_transfers,
            "day": day,
            "special_used": used,
            "original_target": original_target,
            "targets": {},
            "optimization_targets": None,
            "complete": False
        }
        save()

    branch = report["branches"][key]

    if branch["complete"]:
        print(
            f"\nTransferimi {step}: nga checkpoint",
            flush=True
        )
        continue

    print(
        f"\nFAZA 2 - PARA SPECIAL NR. {step}",
        flush=True
    )
    print("=" * 65, flush=True)

    print(
        f"Niveli {len(prefix_route)}/49 | "
        f"Fund {current} | "
        f"Dita {day:.2f} | "
        f"SPECIAL {used}/5",
        flush=True
    )

    print(
        f"Transferimi origjinal SPECIAL: "
        f"{current} -> {original_target}",
        flush=True
    )

    # Kontrollojme te gjitha domatet e pavizituara.
    for target in remaining:

        if str(target) in branch["targets"]:
            continue

        if time.monotonic() - START >= WORK_LIMIT:
            break

        minimum = float("inf")
        best_normal = None

        for wait in WAITS:
            for tof in TOFS:

                if wait + tof > window:
                    continue

                depart = day + wait
                arrival = depart + tof

                if arrival > P.max_time:
                    continue

                try:
                    dv = float(P.compute_transfer(
                        current, target, depart, tof
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
                        "start_time": depart,
                        "tof": tof,
                        "arrival": arrival,
                        "dv": dv,
                        "status": "NORMAL",
                        "method": "grid"
                    }

                    best_normal = better(best_normal, move)

        branch["targets"][str(target)] = {
            "grid_min_dv": (
                minimum if math.isfinite(minimum) else None
            ),
            "best_normal": best_normal,
            "opt_done": False,
            "optimized_min_dv": None
        }

        save()

    if len(branch["targets"]) != len(remaining):
        print(
            "Grid i pjesshem. Ekzekuto perseri qelizen.",
            flush=True
        )
        break

    grid_normals = sum(
        entry["best_normal"] is not None
        for entry in branch["targets"].values()
    )

    print(
        f"Destinacione te kontrolluara: "
        f"{len(remaining)}/{len(remaining)}",
        flush=True
    )

    print(
        f"Destinacione NORMAL nga grid: {grid_normals}",
        flush=True
    )

    # Optimizojme vetem destinacionet qe nuk kane
    # ende NORMAL, por kane Delta-V me te vogel.
    if branch["optimization_targets"] is None:
        ranked = sorted(
            (
                entry["grid_min_dv"],
                int(target)
            )
            for target, entry in branch["targets"].items()
            if (
                entry["grid_min_dv"] is not None
                and entry["best_normal"] is None
            )
        )

        branch["optimization_targets"] = [
            target for _, target in ranked[:3]
        ]
        save()

    to_optimize = branch["optimization_targets"]

    print(
        "Kandidatet per optimizim:",
        to_optimize,
        flush=True
    )

    for target in to_optimize:

        if time.monotonic() - START >= WORK_LIMIT:
            break

        entry = branch["targets"][str(target)]

        if entry["opt_done"]:
            continue

        best_dv = float("inf")
        max_wait = min(18.0, window - 0.25)

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
                    current,
                    target,
                    day + wait,
                    tof
                ))
            except Exception:
                return 1e8

            return dv if math.isfinite(dv) else 1e8

        for seed in [2026, 2047]:
            try:
                result = differential_evolution(
                    objective,
                    bounds=[(0, 1), (0, 1)],
                    seed=seed + step * 100 + target,
                    maxiter=16,
                    popsize=6,
                    polish=False,
                    workers=1
                )

                wait, tof = decode(result.x)
                dv = float(P.compute_transfer(
                    current,
                    target,
                    day + wait,
                    tof
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
        entry["optimized_min_dv"] = (
            best_dv if math.isfinite(best_dv) else None
        )
        save()

        print(
            f"  {current} -> {target} | "
            f"DV minimale e optimizuar: {best_dv:.2f}",
            flush=True
        )

    if not all(
        branch["targets"][str(t)]["opt_done"]
        for t in to_optimize
    ):
        print(
            "Optimizimi eshte i pjesshem. "
            "Ekzekuto perseri.",
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
        f"destinacione alternative NORMAL",
        flush=True
    )

    for move in sorted(
        normals,
        key=lambda m: (m["arrival"], m["dv"])
    )[:6]:
        print(
            f"  {move['from']} -> {move['to']} | "
            f"Dita {move['arrival']:.2f} | "
            f"DV {move['dv']:.2f}",
            flush=True
        )

# FAZA 3 - Ndertojme prefikse te reja, te vlefshme.
candidates = {}

for branch in report["branches"].values():

    for entry in branch["targets"].values():
        move = entry["best_normal"]

        if move is None:
            continue

        transfer = {
            field: move[field]
            for field in [
                "from", "to", "start_time",
                "tof", "dv", "status"
            ]
        }

        new_state = {
            "route": branch["prefix_route"] + [move["to"]],
            "transfers": (
                branch["prefix_transfers"] + [transfer]
            ),
            "current_time": move["arrival"],
            "special_used": branch["special_used"],
            "replaced_special_step": branch["step"]
        }

        assert (
            len(new_state["transfers"])
            == len(new_state["route"]) - 1
        )
        assert (
            len(set(new_state["route"]))
            == len(new_state["route"])
        )

        identity = hashlib.sha256(
            json.dumps([
                new_state["route"],
                round(new_state["current_time"], 6)
            ]).encode()
        ).hexdigest()[:20]

        candidates[identity] = new_state

report["candidate_prefixes"] = sorted(
    candidates.values(),
    key=lambda s: (
        -len(s["route"]),
        s["special_used"],
        s["current_time"]
    )
)

save()

print("\nREZULTATI FINAL I HAPIT 119")
print("=" * 65, flush=True)

print(
    "Manovra SPECIAL te analizuara:",
    len(report["branches"]),
    "/",
    len(steps_to_check),
    flush=True
)

print(
    "Prefikse alternative NORMAL:",
    len(report["candidate_prefixes"]),
    flush=True
)

for step in steps_to_check:
    branch = report["branches"].get(str(step))

    if not branch:
        continue

    normals = [
        entry["best_normal"]
        for entry in branch["targets"].values()
        if entry["best_normal"] is not None
    ]

    print(
        f"Para SPECIAL {step} | "
        f"Niveli fillestar {len(branch['prefix_route'])} | "
        f"SPECIAL {branch['special_used']}/5 | "
        f"NORMAL {len(normals)} | "
        f"Perfunduar {branch['complete']}",
        flush=True
    )

    if normals:
        earliest = min(
            normals,
            key=lambda m: (m["arrival"], m["dv"])
        )

        print(
            f"  Alternativa: "
            f"{earliest['from']} -> {earliest['to']} | "
            f"Dita {earliest['arrival']:.2f} | "
            f"DV {earliest['dv']:.2f}",
            flush=True
        )

print("\nHAPI 119 PERFUNDOI!", flush=True)
