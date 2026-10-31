
import json
import math
from pathlib import Path
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_v5_level_12.json"
OUTPUT = RESULTS / "v6_finer_grid_analysis.json"

problem = TomatoProblem(str(ROOT / "data/easy.kttsp"))

# Kohet qe perdorte V5
OLD_WAITS = [0, 1, 2, 3, 5, 8, 13]
OLD_FLIGHTS = [0.5, 1, 2, 3, 5, 8, 13, 21, 34]

# Kohe shtese per eksperimentin V6
WAITS = sorted(set(OLD_WAITS + [
    0.5, 1.5, 2.5, 4, 6, 10, 17, 26
]))

FLIGHTS = sorted(set(OLD_FLIGHTS + [
    0.75, 1.25, 1.5, 2.5, 3.5,
    4, 6, 10, 17, 27
]))

data = json.loads(SOURCE.read_text())

# Zgjedhim 3 rruge, duke preferuar me pak
# speciale dhe pastaj me pak kohe
states = sorted(
    data["beam"],
    key=lambda s: (
        s["special_used"],
        s["current_time"]
    )
)[:3]

if OUTPUT.exists():
    report = json.loads(OUTPUT.read_text())
else:
    report = {"cases": [], "completed": False}

done = {x["route_number"] for x in report["cases"]}

print("BEAM SEARCH V6 - ANALIZA E KOHES")
print("=" * 50, flush=True)

for i, state in enumerate(states, 1):

    if i in done:
        print(f"Rruga {i}: analiza ekziston", flush=True)
        continue

    route = state["route"]
    now = float(state["current_time"])
    current = route[-1]
    visited = set(route)

    normal_old = {}
    normal_new = {}
    special_new = {}
    evaluations = 0

    print(f"\nPo analizohet rruga {i}", flush=True)
    print("Rruga:", route, flush=True)
    print("Speciale:", state["special_used"], flush=True)

    for target in range(problem.n_tomatoes):

        if target in visited:
            continue

        for wait in WAITS:
            for tof in FLIGHTS:

                departure = now + wait
                arrival = departure + tof

                if tof < problem.min_tof:
                    continue

                if arrival > problem.max_time:
                    continue

                try:
                    dv = float(problem.compute_transfer(
                        current, target, departure, tof
                    ))
                    evaluations += 1
                except Exception:
                    continue

                if not math.isfinite(dv):
                    continue

                candidate = {
                    "target": target,
                    "wait": wait,
                    "tof": tof,
                    "arrival": arrival,
                    "dv": dv
                }

                if dv < problem.dv_threshold:

                    old = normal_new.get(target)

                    if old is None or (
                        arrival, dv
                    ) < (
                        old["arrival"], old["dv"]
                    ):
                        normal_new[target] = candidate

                    # Krahasim me rrjetin e vjeter
                    if (
                        wait in OLD_WAITS
                        and tof in OLD_FLIGHTS
                    ):
                        old = normal_old.get(target)

                        if old is None or (
                            arrival, dv
                        ) < (
                            old["arrival"], old["dv"]
                        ):
                            normal_old[target] = candidate

                elif dv < problem.exception_threshold:

                    old = special_new.get(target)

                    if old is None or (
                        arrival, dv
                    ) < (
                        old["arrival"], old["dv"]
                    ):
                        special_new[target] = candidate

    new_targets = (
        set(normal_new) - set(normal_old)
    )

    print("Normale me V5:", len(normal_old))
    print("Normale me V6:", len(normal_new))
    print("Domate te reja normale:", len(new_targets))
    print("Llogaritje:", evaluations)

    print("Alternativat e reja normale:")

    for target in sorted(new_targets)[:5]:
        c = normal_new[target]
        print(
            f"  {current} -> {target} | "
            f"pritje {c['wait']} | "
            f"fluturim {c['tof']} | "
            f"DV {c['dv']:.2f}"
        )

    report["cases"].append({
        "route_number": i,
        "route": route,
        "current_time": now,
        "special_used": state["special_used"],
        "normal_old": len(normal_old),
        "normal_new": len(normal_new),
        "new_targets": sorted(new_targets),
        "normal_candidates": list(normal_new.values()),
        "special_count": len(special_new),
        "evaluations": evaluations
    })

    # Ruajtje pas cdo rruge
    temp = OUTPUT.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2))
    temp.replace(OUTPUT)

report["completed"] = True
OUTPUT.write_text(json.dumps(report, indent=2))

print("\nPERMBLEDHJA E V6")
print("=" * 50)

for case in sorted(
    report["cases"],
    key=lambda c: c["route_number"]
):
    print(
        f"Rruga {case['route_number']}: "
        f"V5={case['normal_old']} normale | "
        f"V6={case['normal_new']} normale | "
        f"Shtese={len(case['new_targets'])}"
    )

print("\nAnaliza u ruajt:", OUTPUT.name)
print("Checkpoint V5 nuk u ndryshua.")
