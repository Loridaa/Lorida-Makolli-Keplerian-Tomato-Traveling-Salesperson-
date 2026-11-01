
from pathlib import Path
from utils_users import TomatoProblem
import json
import math

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

RESCUE = RESULTS / "beam_v817_rescue_level_21.json"
COMP = RESULTS / "beam_v818_comparison.json"
OUTPUT = RESULTS / "beam_v818_other_branches.json"

problem = TomatoProblem(str(ROOT / "data/easy.kttsp"))
states = json.loads(RESCUE.read_text())["beam"]
comparison = json.loads(COMP.read_text())

matches = [
    s for s in states
    if abs(s["current_time"] - 107.0) < 1e-6
    and s["route"][-1] == 12
    and s["special_used"] == 3
]

assert len(matches) == 1, "Rruga C nuk u gjet."
state = matches[0]

# Zgjedhim dy alternativa te reja, jo 13 dhe 20
choices = [
    m for m in comparison["runs"]["C"]["special_moves"]
    if m["target"] not in [13, 20]
]

choices.sort(key=lambda m: (m["arrival"], m["dv"]))
choices = choices[:2]

assert len(choices) == 2, "Mungojne dy alternativa."

WAITS = [0, 0.5, 1, 2, 3, 5, 8, 13]
TOFS = [0.5, 1, 1.5, 2, 3, 5, 8, 13, 21, 34]

if OUTPUT.exists():
    report = json.loads(OUTPUT.read_text())
else:
    report = {"branches": {}}

for first in choices:

    target_first = int(first["target"])
    key = str(target_first)

    start = 107.0 + first["wait"]
    arrival = start + first["tof"]

    dv = float(problem.compute_transfer(
        12, target_first, start, first["tof"]
    ))

    assert 100 <= dv < 600
    assert target_first not in state["route"]
    assert arrival <= problem.max_time

    route = state["route"] + [target_first]
    remaining = [
        t for t in range(problem.n_tomatoes)
        if t not in route
    ]

    metadata = {
        "route": route,
        "arrival": arrival,
        "dv": dv,
        "special_used": 4
    }

    if key not in report["branches"]:
        report["branches"][key] = {
            "metadata": metadata,
            "targets": {}
        }

    branch = report["branches"][key]

    assert branch["metadata"] == metadata, (
        "Cache nuk perputhet me rrugen."
    )

    print(f"\nRRUGA 12 -> {target_first}", flush=True)
    print("Mberritja:", arrival, flush=True)
    print("Delta-V:", round(dv, 2), flush=True)
    print("Speciale:", 4, "/5", flush=True)

    for target in remaining:

        if str(target) in branch["targets"]:
            continue

        min_dv = float("inf")
        best_normal = None

        for wait in WAITS:
            for tof in TOFS:

                depart = arrival + wait
                end_time = depart + tof

                if end_time > problem.max_time:
                    continue

                try:
                    value = float(problem.compute_transfer(
                        target_first, target, depart, tof
                    ))
                except Exception:
                    continue

                if not math.isfinite(value):
                    continue

                min_dv = min(min_dv, value)

                if value < problem.dv_threshold:
                    move = {
                        "to": target,
                        "arrival": end_time,
                        "wait": wait,
                        "tof": tof,
                        "dv": value
                    }

                    if (
                        best_normal is None
                        or (end_time, value) <
                           (best_normal["arrival"],
                            best_normal["dv"])
                    ):
                        best_normal = move

        branch["targets"][str(target)] = {
            "min_dv": (
                min_dv if math.isfinite(min_dv) else None
            ),
            "best_normal": best_normal
        }

        temp = OUTPUT.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2))
        temp.replace(OUTPUT)

    normals = [
        x["best_normal"]
        for x in branch["targets"].values()
        if x["best_normal"] is not None
    ]

    print("Domate te kontrolluara:",
          len(branch["targets"]), "/27", flush=True)
    print("Transferime NORMAL:",
          len(normals), flush=True)

    for move in sorted(
        normals, key=lambda x: x["arrival"]
    )[:5]:
        print(
            f"{target_first} -> {move['to']} | "
            f"Dita {move['arrival']:.2f} | "
            f"DV {move['dv']:.2f}",
            flush=True
        )

print("\nHAPI 99 - KERKIMI PERFUNDOI!", flush=True)
