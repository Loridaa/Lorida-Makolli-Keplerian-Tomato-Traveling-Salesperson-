
import json
import math
from pathlib import Path
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

problem = TomatoProblem(str(ROOT / "data/easy.kttsp"))

rescue = json.loads(
    (RESULTS / "beam_v817_rescue_level_21.json").read_text()
)
comparison = json.loads(
    (RESULTS / "beam_v818_comparison.json").read_text()
)
step99 = json.loads(
    (RESULTS / "beam_v818_other_branches.json").read_text()
)
step100 = json.loads(
    (RESULTS / "beam_v819_lookahead_24.json").read_text()
)

OUTPUT = RESULTS / "beam_v820_lookahead_25.json"

matches = [
    s for s in rescue["beam"]
    if abs(s["current_time"] - 107.0) < 1e-6
    and s["route"][-1] == 12
    and s["special_used"] == 3
]

assert len(matches) == 1, "Rruga C nuk u gjet."
base = matches[0]

# Verifikojme secilin transferim me PyKEP.
def extend(state, target, move, status):
    current = state["route"][-1]
    assert target not in state["route"]

    start = state["current_time"] + move["wait"]
    tof = move["tof"]
    arrival = start + tof

    dv = float(problem.compute_transfer(
        current, target, start, tof
    ))

    assert math.isfinite(dv)
    assert abs(arrival - move["arrival"]) < 1e-6

    if status == "NORMAL":
        assert dv < problem.dv_threshold
    else:
        assert problem.dv_threshold <= dv < problem.exception_threshold
        assert state["special_used"] < problem.n_exceptions

    return {
        "route": state["route"] + [target],
        "current_time": arrival,
        "special_used": (
            state["special_used"] + (status == "SPECIAL")
        )
    }

first = next(
    m for m in comparison["runs"]["C"]["special_moves"]
    if m["target"] == 34
)

state34 = extend(base, 34, first, "SPECIAL")

WAITS = [0, 0.5, 1, 2, 3, 5, 8, 13]
TOFS = [0.5, 1, 1.5, 2, 3, 5, 8, 13, 21, 34]

report = (
    json.loads(OUTPUT.read_text())
    if OUTPUT.exists()
    else {"branches": {}}
)

for second, third in [(23, 18), (18, 23)]:

    second_move = step99["branches"]["34"][
        "targets"][str(second)]["best_normal"]

    state23 = extend(
        state34, second, second_move, "NORMAL"
    )

    key100 = f"12-34-{second}"

    third_move = step100["branches"][key100][
        "targets"][str(third)]["best_normal"]

    state24 = extend(
        state23, third, third_move, "NORMAL"
    )

    key = f"12-34-{second}-{third}"

    metadata = {
        "route": state24["route"],
        "arrival": state24["current_time"],
        "special_used": state24["special_used"],
        "waits": WAITS,
        "tofs": TOFS
    }

    if key not in report["branches"]:
        report["branches"][key] = {
            "metadata": metadata,
            "targets": {}
        }

    branch = report["branches"][key]
    assert branch["metadata"] == metadata, (
        "Cache nuk perputhet me parametrat."
    )

    remaining = [
        t for t in range(problem.n_tomatoes)
        if t not in state24["route"]
    ]

    print(f"\nRRUGA: {key}", flush=True)
    print(
        "Dita:", state24["current_time"],
        "| Domate:", len(state24["route"]),
        "| Speciale:", state24["special_used"],
        flush=True
    )

    for target in remaining:

        if str(target) in branch["targets"]:
            continue

        best_normal = None
        best_special = None
        min_dv = float("inf")

        for wait in WAITS:
            for tof in TOFS:

                start = state24["current_time"] + wait
                arrival = start + tof

                if arrival > problem.max_time:
                    continue

                try:
                    dv = float(problem.compute_transfer(
                        third, target, start, tof
                    ))
                except Exception:
                    continue

                if not math.isfinite(dv):
                    continue

                min_dv = min(min_dv, dv)

                move = {
                    "from": third,
                    "to": target,
                    "wait": wait,
                    "tof": tof,
                    "arrival": arrival,
                    "dv": dv
                }

                if dv < problem.dv_threshold:
                    if (
                        best_normal is None
                        or (arrival, dv) <
                           (best_normal["arrival"],
                            best_normal["dv"])
                    ):
                        best_normal = move

                elif (
                    dv < problem.exception_threshold
                    and state24["special_used"] < 5
                ):
                    if (
                        best_special is None
                        or (arrival, dv) <
                           (best_special["arrival"],
                            best_special["dv"])
                    ):
                        best_special = move

        branch["targets"][str(target)] = {
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
            "NORMAL" if best_normal
            else "SPECIAL" if best_special
            else "ASNJE"
        )

        print(
            f"[{len(branch['targets'])}/{len(remaining)}] "
            f"Domatja {target}: {status}",
            flush=True
        )

print("\nKERKIMI PERFUNDOI!", flush=True)
