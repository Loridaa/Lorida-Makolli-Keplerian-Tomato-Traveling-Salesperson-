
import json
import math
from pathlib import Path
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

RESCUE = RESULTS / "beam_v817_rescue_level_21.json"
COMPARISON = RESULTS / "beam_v818_comparison.json"
OUTPUT = RESULTS / "beam_v818_direct20.json"

problem = TomatoProblem(str(ROOT / "data/easy.kttsp"))

states = json.loads(RESCUE.read_text())["beam"]
comparison = json.loads(COMPARISON.read_text())

# Gjejme rrugen C: dita 107, fund 12, 3 speciale
matches = [
    s for s in states
    if abs(s["current_time"] - 107.0) < 1e-6
    and s["route"][-1] == 12
    and s["special_used"] == 3
]

assert len(matches) == 1, "Rruga C nuk u gjet."

state = matches[0]

# Gjejme transferimin SPECIAL direkt 12 -> 20
first = next(
    (
        m for m in comparison["runs"]["C"]["special_moves"]
        if m["target"] == 20
    ),
    None
)

assert first is not None, "Transferimi 12 -> 20 mungon."
assert 20 not in state["route"]

start = state["current_time"] + first["wait"]
arrival = start + first["tof"]

dv = float(problem.compute_transfer(
    12, 20, start, first["tof"]
))

assert 100 <= dv < 600, "SPECIAL nuk u konfirmua."
assert arrival <= problem.max_time

route = state["route"] + [20]
used = state["special_used"] + 1

WAITS = [0, 0.5, 1, 2, 3, 5, 8, 13]
TOFS = [0.5, 1, 1.5, 2, 3, 5, 8, 13, 21, 34]

remaining = [
    t for t in range(problem.n_tomatoes)
    if t not in route
]

metadata = {
    "route": route,
    "arrival": arrival,
    "dv": dv,
    "special_used": used
}

if OUTPUT.exists():
    report = json.loads(OUTPUT.read_text())
    assert report["metadata"] == metadata, (
        "Cache nuk i perket kesaj rruge."
    )
else:
    report = {
        "metadata": metadata,
        "targets": {}
    }

print("HAPI 98 - RRUGA DIREKTE 12 -> 20")
print("=" * 60)
print("Koha e mberritjes:", round(arrival, 3))
print("Delta-V:", round(dv, 3))
print("Domate:", len(route), "/49")
print("Speciale te perdorura:", used, "/5")
print("Domate per kontroll:", len(remaining))
print("=" * 60, flush=True)

for target in remaining:

    if str(target) in report["targets"]:
        continue

    min_dv = float("inf")
    best_normal = None
    best_special = None

    for wait in WAITS:
        for tof in TOFS:

            t_start = arrival + wait
            t_end = t_start + tof

            if t_end > problem.max_time:
                continue

            try:
                value = float(problem.compute_transfer(
                    20, target, t_start, tof
                ))
            except Exception:
                continue

            if not math.isfinite(value):
                continue

            min_dv = min(min_dv, value)

            move = {
                "from": 20,
                "to": target,
                "start_time": t_start,
                "wait": wait,
                "tof": tof,
                "arrival": t_end,
                "dv": value
            }

            if value < 100:
                if (
                    best_normal is None
                    or t_end < best_normal["arrival"]
                ):
                    best_normal = move

            elif value < 600 and used < 5:
                if (
                    best_special is None
                    or t_end < best_special["arrival"]
                ):
                    best_special = move

    report["targets"][str(target)] = {
        "min_dv": (
            min_dv if math.isfinite(min_dv) else None
        ),
        "best_normal": best_normal,
        "best_special": best_special
    }

    temp = OUTPUT.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2))
    temp.replace(OUTPUT)

    status = (
        "NORMAL" if best_normal is not None
        else "SPECIAL" if best_special is not None
        else "ASNJE"
    )

    print(
        f"[{len(report['targets'])}/{len(remaining)}] "
        f"Domatja {target}: {status}",
        flush=True
    )

print("\nKONTROLLI PERFUNDOI!", flush=True)
