
import json
import math
from pathlib import Path
from scipy.optimize import differential_evolution
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_v824_lookahead_24.json"
OUTPUT = RESULTS / "beam_v825_optimize_from25.json"

problem = TomatoProblem(str(ROOT / "data/easy.kttsp"))
source = json.loads(SOURCE.read_text())

TARGETS = [22, 13, 20]
MAX_EXTRA = 16.0
MAX_WAIT = 13.0
MIN_TOF = 0.25
SEEDS = [2026, 2037]

settings = {
    "targets": TARGETS,
    "max_extra": MAX_EXTRA,
    "max_wait": MAX_WAIT,
    "min_tof": MIN_TOF,
    "seeds": SEEDS
}

if OUTPUT.exists():
    report = json.loads(OUTPUT.read_text())
    assert report["settings"] == settings
else:
    report = {"settings": settings, "branches": {}}

def save():
    temp = OUTPUT.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2))
    temp.replace(OUTPUT)

# Koordinatat e optimizimit sigurojne qe:
# wait + tof <= 16 dite
def decode(x):
    wait = MAX_WAIT * float(x[0])
    available = MAX_EXTRA - wait
    tof = MIN_TOF + float(x[1]) * (available - MIN_TOF)
    return wait, tof

for name in ["A_NORMAL_NORMAL", "B_SPECIAL_NORMAL"]:

    meta = source["branches"][name]["metadata"]
    route = meta["route"]
    day = float(meta["arrival"])
    used = int(meta["special_used"])
    current = int(route[-1])

    assert current == 25
    assert len(route) == 23
    assert len(route) == len(set(route))

    branch_meta = {
        "route": route,
        "day": day,
        "special_used": used
    }

    if name not in report["branches"]:
        report["branches"][name] = {
            "metadata": branch_meta,
            "targets": {}
        }

    branch = report["branches"][name]
    assert branch["metadata"] == branch_meta

    print("\nRRUGA:", name, flush=True)
    print("Dita fillestare:", day, flush=True)
    print("Speciale te perdorura:", used, "/5", flush=True)

    for target in TARGETS:

        assert target not in route
        key = str(target)

        if key in branch["targets"]:
            print("Nga cache: domatja", target, flush=True)
            continue

        best = None
        total_evaluations = 0

        for seed in SEEDS:

            evaluations = [0]

            def objective(x):
                wait, tof = decode(x)
                depart = day + wait
                arrival = depart + tof

                if arrival > problem.max_time:
                    return 1e6

                try:
                    dv = float(problem.compute_transfer(
                        current, target, depart, tof
                    ))
                    evaluations[0] += 1
                except Exception:
                    return 1e6

                return dv if math.isfinite(dv) else 1e6

            res = differential_evolution(
                objective,
                bounds=[(0, 1), (0, 1)],
                maxiter=30,
                popsize=8,
                seed=seed + target,
                polish=True,
                workers=1,
                tol=0.01
            )

            total_evaluations += evaluations[0]

            wait, tof = decode(res.x)
            arrival = day + wait + tof

            try:
                dv = float(problem.compute_transfer(
                    current, target, day + wait, tof
                ))
            except Exception:
                dv = float("inf")

            if math.isfinite(dv):
                candidate = {
                    "from": current,
                    "to": target,
                    "wait": wait,
                    "tof": tof,
                    "arrival": arrival,
                    "dv": dv
                }

                if best is None or dv < best["dv"]:
                    best = candidate

        if best is None:
            result = {
                "status": "ASNJE",
                "dv": None,
                "evaluations": total_evaluations
            }
        else:
            if best["dv"] < problem.dv_threshold:
                status = "NORMAL"
            elif (
                best["dv"] < problem.exception_threshold
                and used < problem.n_exceptions
            ):
                status = "SPECIAL"
            else:
                status = "ASNJE"

            result = {
                **best,
                "status": status,
                "evaluations": total_evaluations
            }

        branch["targets"][key] = result
        save()

        if result["dv"] is not None:
            print(
                f"25 -> {target} | "
                f"DV {result['dv']:.3f} | "
                f"Dita {result['arrival']:.3f} | "
                f"{result['status']}",
                flush=True
            )
        else:
            print(f"25 -> {target}: ASNJE", flush=True)

print("\nOPTIMIZIMI PERFUNDOI!", flush=True)
