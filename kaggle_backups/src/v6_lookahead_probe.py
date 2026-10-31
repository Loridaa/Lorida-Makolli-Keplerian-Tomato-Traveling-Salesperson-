
import json
import math
from pathlib import Path
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"
SOURCE = RESULTS / "beam_v5_level_12.json"
OUTPUT = RESULTS / "v6_lookahead_probe.json"

problem = TomatoProblem(str(ROOT / "data/easy.kttsp"))

WAITS = [0, 1, 2, 3, 5, 8, 13]
FLIGHTS = [0.5, 1, 2, 3, 5, 8, 13, 21, 34]

# Rrjet me i vogel per kontrollin e hapit pasues
FUTURE_WAITS = [0, 1, 3, 8]
FUTURE_FLIGHTS = [0.5, 1, 2, 3, 5, 8, 13, 21]

beam = json.loads(SOURCE.read_text())["beam"]

# Marrim rruge me kohe dhe speciale te ndryshme
ids = []

groups = [
    sorted(range(len(beam)),
           key=lambda i: (
               beam[i]["special_used"],
               beam[i]["current_time"]
           )),
    sorted(range(len(beam)),
           key=lambda i: beam[i]["current_time"]),
    sorted((i for i in range(len(beam))
            if beam[i]["special_used"] == 3),
           key=lambda i: beam[i]["current_time"]),
    sorted((i for i in range(len(beam))
            if beam[i]["special_used"] == 4),
           key=lambda i: beam[i]["current_time"])
]

for group in groups:
    for i in group:
        if i not in ids:
            ids.append(i)
            break

ids = ids[:4]


def find_moves(state):
    now = float(state["current_time"])
    current = state["route"][-1]
    visited = set(state["route"])
    used = int(state["special_used"])

    normal = {}
    special = {}

    for target in range(problem.n_tomatoes):
        if target in visited:
            continue

        for wait in WAITS:
            for tof in FLIGHTS:
                arrival = now + wait + tof

                if (tof < problem.min_tof
                    or arrival > problem.max_time):
                    continue

                try:
                    dv = float(problem.compute_transfer(
                        current, target, now + wait, tof
                    ))
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
                    status = "NORMAL"

                elif (dv < problem.exception_threshold
                      and used < problem.n_exceptions):
                    store = special
                    status = "SPECIAL"

                else:
                    continue

                old = store.get(target)

                if old is None or arrival < old["arrival"]:
                    store[target] = candidate

    normals = sorted(
        normal.values(),
        key=lambda c: (c["arrival"], c["dv"])
    )

    specials = sorted(
        special.values(),
        key=lambda c: (c["arrival"], c["dv"])
    )

    candidates = [
        ("NORMAL", c) for c in normals[:2]
    ] + [
        ("SPECIAL", c) for c in specials[:1]
    ]

    return candidates


def count_future_normals(route, current, now):
    visited = set(route)
    count = 0

    for target in range(problem.n_tomatoes):
        if target in visited:
            continue

        found = False

        for wait in FUTURE_WAITS:
            if found:
                break

            for tof in FUTURE_FLIGHTS:
                if (tof < problem.min_tof
                    or now + wait + tof > problem.max_time):
                    continue

                try:
                    dv = float(problem.compute_transfer(
                        current, target, now + wait, tof
                    ))
                except Exception:
                    continue

                if math.isfinite(dv) and dv < problem.dv_threshold:
                    found = True
                    break

        if found:
            count += 1

    return count


print("V6 - ANALIZA LOOKAHEAD")
print("=" * 48, flush=True)

report = []

for i in ids:
    state = beam[i]
    route = state["route"]
    moves = find_moves(state)

    print(f"\nRRUGA {i + 1}", flush=True)
    print("Domatja aktuale:", route[-1], flush=True)
    print("Speciale:", state["special_used"], flush=True)

    options = []

    for status, c in moves:
        future = count_future_normals(
            route + [c["target"]],
            c["target"],
            c["arrival"]
        )

        option = {
            **c,
            "status": status,
            "future_normals": future,
            "special_used_after": (
                state["special_used"] +
                int(status == "SPECIAL")
            )
        }

        options.append(option)

        print(
            f"{route[-1]} -> {c['target']} | "
            f"{status} | "
            f"Mberritja {c['arrival']} | "
            f"Normale ne hapin tjeter: {future}",
            flush=True
        )

    report.append({
        "route_number": i + 1,
        "route": route,
        "options": options
    })

    # Ruajme pas cdo rruge
    OUTPUT.write_text(json.dumps(report, indent=2))

print("\nANALIZA LOOKAHEAD PERFUNDOI!")
print("Rruge te analizuara:", len(report))
print("Skedari:", OUTPUT.name)
print("Checkpoint V5 nuk u ndryshua.")
