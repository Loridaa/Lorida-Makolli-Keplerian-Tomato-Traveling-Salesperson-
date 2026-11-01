
from pathlib import Path
from utils_users import TomatoProblem
import json
import math

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

RESCUE = RESULTS / "beam_v817_rescue_level_21.json"
COMP = RESULTS / "beam_v818_comparison.json"
PREVIOUS = RESULTS / "beam_v818_other_branches.json"
OUTPUT = RESULTS / "beam_v819_lookahead_24.json"

problem = TomatoProblem(str(ROOT / "data/easy.kttsp"))

states = json.loads(RESCUE.read_text())["beam"]
comparison = json.loads(COMP.read_text())
previous = json.loads(PREVIOUS.read_text())

matches = [
    s for s in states
    if abs(s["current_time"] - 107.0) < 1e-6
    and s["route"][-1] == 12
    and s["special_used"] == 3
]
assert len(matches) == 1

state = matches[0]

first = next(
    m for m in comparison["runs"]["C"]["special_moves"]
    if m["target"] == 34
)

first_start = 107.0 + first["wait"]
first_arrival = first_start + first["tof"]

first_dv = float(problem.compute_transfer(
    12, 34, first_start, first["tof"]
))

assert 100 <= first_dv < 600
assert abs(first_arrival - 107.5) < 1e-6

branch = previous["branches"]["34"]

WAITS = [0, 0.5, 1, 2, 3, 5, 8, 13]
TOFS = [0.5, 1, 1.5, 2, 3, 5, 8, 13, 21, 34]

if OUTPUT.exists():
    report = json.loads(OUTPUT.read_text())
else:
    report = {"branches": {}}

for second in [18, 23]:

    key = f"12-34-{second}"

    move = branch["targets"][str(second)]["best_normal"]
    assert move is not None, "Mungon transferimi NORMAL."

    depart = first_arrival + move["wait"]
    arrival = depart + move["tof"]

    dv = float(problem.compute_transfer(
        34, second, depart, move["tof"]
    ))

    assert math.isfinite(dv) and dv < problem.dv_threshold
    assert abs(arrival - move["arrival"]) < 1e-6

    route = state["route"] + [34, second]
    assert len(route) == len(set(route))

    metadata = {
        "route": route,
        "arrival": arrival,
        "special_used": 4,
        "second_dv": dv
    }

    if key not in report["branches"]:
        report["branches"][key] = {
            "metadata": metadata,
            "targets": {}
        }

    current_branch = report["branches"][key]
    assert current_branch["metadata"] == metadata

    remaining = [
        t for t in range(problem.n_tomatoes)
        if t not in route
    ]

    print(f"\nRRUGA {key}", flush=True)
    print("Domate:", len(route), "/49", flush=True)
    print("Dita:", arrival, flush=True)
    print("Speciale:", 4, "/5", flush=True)

    for target in remaining:

        if str(target) in current_branch["targets"]:
            continue

        min_dv = float("inf")
        best_normal = None
        best_special = None

        for wait in WAITS:
            for tof in TOFS:

                start = arrival + wait
                finish = start + tof

                if finish > problem.max_time:
                    continue

                try:
                    value = float(problem.compute_transfer(
                        second, target, start, tof
                    ))
                except Exception:
                    continue

                if not math.isfinite(value):
                    continue

                min_dv = min(min_dv, value)

                candidate = {
                    "from": second,
                    "to": target,
                    "start_time": start,
                    "wait": wait,
                    "tof": tof,
                    "arrival": finish,
                    "dv": value
                }

                if value < problem.dv_threshold:
                    if (
                        best_normal is None
                        or (finish, value) <
                           (best_normal["arrival"],
                            best_normal["dv"])
                    ):
                        best_normal = candidate

                elif value < problem.exception_threshold:
                    if (
                        best_special is None
                        or (finish, value) <
                           (best_special["arrival"],
                            best_special["dv"])
                    ):
                        best_special = candidate

        current_branch["targets"][str(target)] = {
            "min_dv": (
                min_dv if math.isfinite(min_dv) else None
            ),
            "best_normal": best_normal,
            "best_special": best_special
        }

        temp = OUTPUT.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2))
        temp.replace(OUTPUT)

    normals = [
        x["best_normal"]
        for x in current_branch["targets"].values()
        if x["best_normal"] is not None
    ]

    specials = [
        x["best_special"]
        for x in current_branch["targets"].values()
        if x["best_special"] is not None
    ]

    print("Kontrolluar:",
          len(current_branch["targets"]), "/26", flush=True)
    print("NORMAL:", len(normals), flush=True)
    print("SPECIAL:", len(specials), flush=True)

    for m in sorted(normals, key=lambda x: x["arrival"])[:5]:
        print(
            f"NORMAL {second} -> {m['to']} | "
            f"Dita {m['arrival']:.2f} | "
            f"DV {m['dv']:.2f}",
            flush=True
        )

print("\nHAPI 100 - ANALIZA PERFUNDOI!", flush=True)
