
import json
import math
from pathlib import Path
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_v7_level_12.json"
OUTPUT = RESULTS / "v7_next_moves_analysis.json"

problem = TomatoProblem(str(ROOT / "data/easy.kttsp"))

WAITS = [0, 0.5, 1, 2, 3, 5, 8, 13]
FLIGHTS = [0.5, 1, 1.5, 2, 3, 5, 8, 13, 21, 34]

beam = json.loads(SOURCE.read_text())["beam"]

# Zgjedhim 8 rruget me me pak speciale
states = sorted(
    beam,
    key=lambda s: (
        s["special_used"],
        s["current_time"]
    )
)[:8]

if OUTPUT.exists():
    report = json.loads(OUTPUT.read_text())
else:
    report = {"cases": []}

def save():
    temp = OUTPUT.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2))
    temp.replace(OUTPUT)

print("V7 - KONTROLLI I VAZHDIMEVE")
print("=" * 50, flush=True)

for i, state in enumerate(states, 1):
    route = state["route"]
    now = float(state["current_time"])
    used = int(state["special_used"])
    current = route[-1]
    visited = set(route)

    key = json.dumps([route, now, used])

    previous = next(
        (c for c in report["cases"] if c["key"] == key),
        None
    )

    if previous is not None:
        print(f"Rruga {i}: rezultat i ruajtur", flush=True)
        continue

    normal = {}
    special = {}
    evaluations = 0

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
                    store = normal
                elif (
                    dv < problem.exception_threshold
                    and used < problem.n_exceptions
                ):
                    store = special
                else:
                    continue

                old = store.get(target)

                if old is None or (
                    arrival, dv
                ) < (
                    old["arrival"], old["dv"]
                ):
                    store[target] = candidate

    options = sorted(
        normal.values(),
        key=lambda x: (x["arrival"], x["dv"])
    )

    item = {
        "key": key,
        "route": route,
        "time": now,
        "special_used": used,
        "normal_count": len(normal),
        "special_count": len(special),
        "normal_options": options,
        "evaluations": evaluations
    }

    report["cases"].append(item)
    save()

    print(f"\nRRUGA {i}", flush=True)
    print("Domatja e fundit:", current)
    print("Koha:", now)
    print("Speciale te perdorura:", used)
    print("Domate normale te arritshme:", len(normal))
    print("Domate speciale te arritshme:", len(special))

    for c in options[:3]:
        print(
            f"  {current} -> {c['target']} | "
            f"pritje {c['wait']} | "
            f"fluturim {c['tof']} | "
            f"DV {c['dv']:.2f}"
        )

print("\nPERMBLEDHJA E V7")
print("=" * 50)

for i, state in enumerate(states, 1):
    key = json.dumps([
        state["route"],
        float(state["current_time"]),
        int(state["special_used"])
    ])

    case = next(
        c for c in report["cases"] if c["key"] == key
    )

    print(
        f"Rruga {i}: "
        f"{case['special_used']} speciale | "
        f"{case['normal_count']} normale | "
        f"{case['special_count']} speciale te arritshme"
    )

print("\nAnaliza u ruajt:", OUTPUT.name)
print("Checkpoint V7 nuk u ndryshua.")
