
import json
import math
from pathlib import Path
from scipy.optimize import differential_evolution
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_v820_lookahead_25.json"
OUTPUT = RESULTS / "beam_v821_continuous.json"

MAX_EXTRA_DAYS = 12.0
TARGETS_PER_BRANCH = 6

problem = TomatoProblem(str(ROOT / "data/easy.kttsp"))
source = json.loads(SOURCE.read_text())

settings = {
    "max_extra_days": MAX_EXTRA_DAYS,
    "targets_per_branch": TARGETS_PER_BRANCH,
    "maxiter": 18,
    "popsize": 7
}

if OUTPUT.exists():
    report = json.loads(OUTPUT.read_text())
    assert report["settings"] == settings
else:
    report = {
        "settings": settings,
        "branches": {}
    }

def save():
    temp = OUTPUT.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2))
    temp.replace(OUTPUT)

for name, branch in source["branches"].items():

    state = branch["metadata"]
    route = state["route"]
    current = int(route[-1])
    day = float(state["arrival"])
    used = int(state["special_used"])

    assert len(route) == 24
    assert len(route) == len(set(route))
    assert used == 4

    ranked = sorted(
        (
            (float(info["min_dv"]), int(target))
            for target, info in branch["targets"].items()
            if info.get("min_dv") is not None
            and math.isfinite(info["min_dv"])
        )
    )

    selected = [target for _, target in ranked[:6]]

    metadata = {
        "route": route,
        "day": day,
        "special_used": used,
        "selected": selected
    }

    if name not in report["branches"]:
        report["branches"][name] = {
            "metadata": metadata,
            "results": {}
        }

    result_branch = report["branches"][name]
    assert result_branch["metadata"] == metadata

    print("\nRRUGA:", name, flush=True)
    print("Domatja aktuale:", current, flush=True)
    print("Dita:", day, flush=True)
    print("Kandidate:", selected, flush=True)

    for target in selected:

        key = str(target)

        if key in result_branch["results"]:
            print("Nga cache:", target, flush=True)
            continue

        evaluations = [0]

        def objective(x):
            wait = float(x[0])
            tof = float(x[1])

            if wait + tof > MAX_EXTRA_DAYS:
                return 100000.0 + wait + tof

            arrival = day + wait + tof

            if arrival > problem.max_time:
                return 100000.0 + arrival

            try:
                dv = float(problem.compute_transfer(
                    current,
                    target,
                    day + wait,
                    tof
                ))
                evaluations[0] += 1
            except Exception:
                return 100000.0

            if not math.isfinite(dv):
                return 100000.0

            return dv

        res = differential_evolution(
            objective,
            bounds=[
                (0.0, 8.0),
                (0.25, 12.0)
            ],
            maxiter=18,
            popsize=7,
            seed=2026 + target + current,
            polish=True,
            workers=1
        )

        wait = float(res.x[0])
        tof = float(res.x[1])
        arrival = day + wait + tof

        if wait + tof <= MAX_EXTRA_DAYS + 1e-8:
            try:
                dv = float(problem.compute_transfer(
                    current,
                    target,
                    day + wait,
                    tof
                ))
            except Exception:
                dv = float("inf")
        else:
            dv = float("inf")

        if math.isfinite(dv) and dv < problem.dv_threshold:
            status = "NORMAL"
        elif (
            math.isfinite(dv)
            and dv < problem.exception_threshold
            and used < problem.n_exceptions
        ):
            status = "SPECIAL"
        else:
            status = "ASNJE"

        result_branch["results"][key] = {
            "from": current,
            "to": target,
            "wait": wait,
            "tof": tof,
            "arrival": arrival,
            "dv": dv if math.isfinite(dv) else None,
            "status": status,
            "evaluations": evaluations[0]
        }

        save()

        print(
            f"Domatja {target} | "
            f"DV {dv:.2f} | "
            f"Dita {arrival:.2f} | "
            f"{status}",
            flush=True
        )

print("\nOPTIMIZIMI PERFUNDOI!", flush=True)
