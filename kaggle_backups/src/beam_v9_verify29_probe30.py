
import json
import math
import hashlib
from pathlib import Path

from scipy.optimize import differential_evolution
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"
SOURCE = RESULTS / "beam_v9_auto_level_29.json"
OUTPUT = RESULTS / "beam_v9_probe_level_30.json"

P = TomatoProblem(str(ROOT / "data/easy.kttsp"))
source_bytes = SOURCE.read_bytes()
states = json.loads(source_bytes)["frontier"]

WAITS = [0, 0.25, 0.5, 1, 2, 3, 5, 8]
TOFS = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 13]
MAX_EXTRA = 16.0
OPTIMIZE_TOP = 4

config = {
    "source_hash": hashlib.sha256(source_bytes).hexdigest(),
    "waits": WAITS,
    "tofs": TOFS,
    "max_extra": MAX_EXTRA,
    "optimize_top": OPTIMIZE_TOP,
    "seeds": [2026, 2041],
    "maxiter": 20,
    "popsize": 7
}

if OUTPUT.exists():
    report = json.loads(OUTPUT.read_text())
    assert report["config"] == config, (
        "Checkpoint ka konfigurim tjeter."
    )
else:
    report = {"config": config, "routes": {}}

def save():
    temp = OUTPUT.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2))
    temp.replace(OUTPUT)

def verify(state):
    route = state["route"]
    transfers = state["transfers"]

    if (
        len(route) != 29
        or route[0] != 0
        or len(set(route)) != 29
        or len(transfers) != 28
        or any(t < 0 or t >= P.n_tomatoes for t in route)
    ):
        return False, "Strukture e pavlefshme"

    day = 0.0
    used = 0

    for i, tr in enumerate(transfers):
        try:
            start = float(tr["start_time"])
            tof = float(tr["tof"])
            old_dv = float(tr["dv"])

            if (
                tr["from"] != route[i]
                or tr["to"] != route[i + 1]
                or start < day - 1e-6
                or tof <= 0
                or start + tof > P.max_time + 1e-6
            ):
                return False, f"Koha ose lidhja: hapi {i+1}"

            dv = float(P.compute_transfer(
                route[i], route[i+1], start, tof
            ))

            if (
                not math.isfinite(dv)
                or abs(dv - old_dv) > 0.1
            ):
                return False, f"Delta-V: hapi {i+1}"

            if dv < P.dv_threshold:
                status = "NORMAL"
            elif dv < P.exception_threshold:
                status = "SPECIAL"
                used += 1
            else:
                return False, f"Jashte kufirit: hapi {i+1}"

            if status != tr["status"]:
                return False, f"Status i pasakte: hapi {i+1}"

            day = start + tof

        except Exception as exc:
            return False, f"Hapi {i+1}: {exc}"

    if (
        abs(day - state["current_time"]) > 1e-5
        or used != state["special_used"]
        or used > P.n_exceptions
    ):
        return False, "Gjendja perfundimtare nuk perputhet"

    return True, "28/28 transferime te verifikuara"

def update_best(old, new):
    if old is None:
        return new
    if (new["arrival"], new["dv"]) < (
        old["arrival"], old["dv"]
    ):
        return new
    return old

def categorize(dv, used):
    if dv < P.dv_threshold:
        return "NORMAL"
    if dv < P.exception_threshold and used < P.n_exceptions:
        return "SPECIAL"
    return "ASNJE"

def decode(x):
    wait = 8.0 * float(x[0])
    tof = 0.25 + (MAX_EXTRA - wait - 0.25) * float(x[1])
    return wait, tof

print("FAZA 1 - VERIFIKIMI I RRUGEVE", flush=True)
print("=" * 60, flush=True)

verified = []

for index, state in enumerate(states, 1):
    ok, reason = verify(state)

    print(
        f"Rruga {index}: "
        f"{'E VLEFSHME' if ok else 'GABIM'} | {reason}",
        flush=True
    )

    if ok:
        verified.append((index, state))

if not verified:
    raise RuntimeError(
        "Asnje rruge me 29 domate nuk kaloi verifikimin."
    )

print("\nFAZA 2 - KERKIMI I DOMATES SE 30", flush=True)
print("=" * 60, flush=True)

for index, state in verified:
    key = str(index)
    route = state["route"]
    current = route[-1]
    day = float(state["current_time"])
    used = int(state["special_used"])

    remaining = [
        t for t in range(P.n_tomatoes)
        if t not in route
    ]

    metadata = {
        "route": route,
        "day": day,
        "special_used": used,
        "verified": True
    }

    if key not in report["routes"]:
        report["routes"][key] = {
            "metadata": metadata,
            "targets": {}
        }

    branch = report["routes"][key]
    assert branch["metadata"] == metadata

    print(
        f"\nRRUGA {index} | Fund {current} | "
        f"Dita {day:.3f} | "
        f"SPECIAL {used}/5",
        flush=True
    )

    for target in remaining:
        target_key = str(target)

        if target_key in branch["targets"]:
            continue

        minimum = float("inf")
        best_normal = None

        for wait in WAITS:
            for tof in TOFS:
                if (
                    wait + tof > MAX_EXTRA
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
                        "wait": wait,
                        "tof": tof,
                        "start_time": day + wait,
                        "arrival": day + wait + tof,
                        "dv": dv,
                        "status": "NORMAL",
                        "method": "grid"
                    }
                    best_normal = update_best(best_normal, move)

        branch["targets"][target_key] = {
            "grid_min_dv": (
                minimum if math.isfinite(minimum) else None
            ),
            "best_normal": best_normal,
            "optimization_complete": False,
            "optimized_min_dv": None
        }
        save()

    normals_grid = sum(
        x["best_normal"] is not None
        for x in branch["targets"].values()
    )

    print(
        f"Grid: {len(branch['targets'])}/{len(remaining)} "
        f"domate | NORMAL {normals_grid}",
        flush=True
    )

    # Zgjedhim kandidatet me Delta-V me te vogel
    ranked = sorted(
        (
            (v["grid_min_dv"], int(t))
            for t, v in branch["targets"].items()
            if v["grid_min_dv"] is not None
        )
    )

    selected = [t for _, t in ranked[:OPTIMIZE_TOP]]

    print(
        "Kandidatet per optimizim:", selected,
        flush=True
    )

    for target in selected:
        entry = branch["targets"][str(target)]

        if entry["optimization_complete"]:
            continue

        best_dv = float("inf")
        best_move = None

        for seed in config["seeds"]:

            def objective(x):
                wait, tof = decode(x)

                if day + wait + tof > P.max_time:
                    return 1000000.0

                try:
                    dv = float(P.compute_transfer(
                        current, target, day + wait, tof
                    ))
                except Exception:
                    return 1000000.0

                return dv if math.isfinite(dv) else 1000000.0

            result = differential_evolution(
                objective,
                bounds=[(0, 1), (0, 1)],
                seed=seed + 49 * index + target,
                maxiter=config["maxiter"],
                popsize=config["popsize"],
                polish=False,
                workers=1
            )

            wait, tof = decode(result.x)

            try:
                dv = float(P.compute_transfer(
                    current, target, day + wait, tof
                ))
            except Exception:
                continue

            if not math.isfinite(dv):
                continue

            if dv < best_dv:
                best_dv = dv
                best_move = {
                    "from": current,
                    "to": target,
                    "wait": wait,
                    "tof": tof,
                    "start_time": day + wait,
                    "arrival": day + wait + tof,
                    "dv": dv,
                    "status": categorize(dv, used),
                    "method": "differential_evolution"
                }

        if best_move is not None:
            entry["optimized_min_dv"] = best_dv

            if best_move["status"] == "NORMAL":
                entry["best_normal"] = update_best(
                    entry["best_normal"], best_move
                )

        entry["optimization_complete"] = True
        save()

        if best_move is not None:
            print(
                f"25+ | {current} -> {target} | "
                f"DV {best_dv:.3f} | "
                f"{best_move['status']} | "
                f"Dita {best_move['arrival']:.3f}",
                flush=True
            )

    all_normals = [
        v["best_normal"]
        for v in branch["targets"].values()
        if v["best_normal"] is not None
    ]

    print(
        f"RRUGA {index}: "
        f"{len(all_normals)} destinacione NORMAL",
        flush=True
    )

    if all_normals:
        earliest = min(
            all_normals,
            key=lambda m: (m["arrival"], m["dv"])
        )
        print(
            f"DOMATJA E 30: {current} -> {earliest['to']} "
            f"| Dita {earliest['arrival']:.3f} "
            f"| DV {earliest['dv']:.3f}",
            flush=True
        )

print("\nHAPI 113 - ANALIZA PERFUNDOI!", flush=True)
