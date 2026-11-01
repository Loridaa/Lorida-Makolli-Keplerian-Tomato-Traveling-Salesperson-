
from pathlib import Path
import json
import math
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

RESCUE = RESULTS / "beam_v817_rescue_level_21.json"
COMPARISON = RESULTS / "beam_v818_comparison.json"
OUTPUT = RESULTS / "beam_v818_c13_lookahead.json"

states = json.loads(RESCUE.read_text())["beam"]
comparison = json.loads(COMPARISON.read_text())

matches = [
    s for s in states
    if abs(s["current_time"] - 107.0) < 1e-6
    and s["route"][-1] == 12
    and s["special_used"] == 3
]

assert len(matches) == 1, "Rruga C nuk u gjet sakte."

state = matches[0]

first = next(
    (
        m for m in comparison["runs"]["C"]["special_moves"]
        if m["target"] == 13
    ),
    None
)

assert first is not None, "Transferimi 12 -> 13 mungon."

problem = TomatoProblem(str(ROOT / "data/easy.kttsp"))

start = 107.0 + first["wait"]
arrival = start + first["tof"]

dv = float(problem.compute_transfer(
    12, 13, start, first["tof"]
))

assert 100 <= dv < 600, "SPECIAL nuk u konfirmua."
assert arrival <= problem.max_time
assert 13 not in state["route"]

route = state["route"] + [13]
special_used = state["special_used"] + 1

WAITS = [0, 0.5, 1, 2, 3, 5, 8, 13]
TOFS = [0.5, 1, 1.5, 2, 3, 5, 8, 13, 21, 34]

remaining = [
    t for t in range(problem.n_tomatoes)
    if t not in route
]

metadata = {
    "route": route,
    "from": 12,
    "to": 13,
    "arrival": arrival,
    "first_dv": dv,
    "special_used": special_used
}

if OUTPUT.exists():
    report = json.loads(OUTPUT.read_text())
    assert report["metadata"] == metadata, (
        "Cache ekzistues i perket nje gjendjeje tjeter."
    )
else:
    report = {"metadata": metadata, "targets": {}}

print("RRUGA C -> DOMATJA 13")
print("Domate ne rruge:", len(route))
print("Koha e mberritjes:", round(arrival, 3))
print("Delta-V:", round(dv, 3))
print("Speciale:", special_used, "/5")
print("Domate per kontroll:", len(remaining))
print("Kombinime per domate:", len(WAITS) * len(TOFS))
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
                    13, target, t_start, tof
                ))
            except Exception:
                continue

            if not math.isfinite(value):
                continue

            min_dv = min(min_dv, value)

            move = {
                "from": 13,
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

            elif value < 600 and special_used < 5:
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

print("\nANALIZA PERFUNDOI!", flush=True)
