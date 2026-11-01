
import json
import math
from pathlib import Path
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

RESCUE = RESULTS / "beam_v817_rescue_level_21.json"
STEP103 = RESULTS / "beam_v822_rescue_scan.json"
STEP104 = RESULTS / "beam_v823_lookahead.json"
OUTPUT = RESULTS / "beam_v824_lookahead_24.json"

problem = TomatoProblem(str(ROOT / "data/easy.kttsp"))

states = json.loads(RESCUE.read_text())["beam"]
step103 = json.loads(STEP103.read_text())
step104 = json.loads(STEP104.read_text())

matches = [
    s for s in states
    if abs(s["current_time"] - 114.5) < 1e-6
    and s["route"][-1] == 41
    and s["special_used"] == 3
]

assert len(matches) == 1, "Rruga D nuk u gjet."
base = matches[0]

WAITS = [0, 0.5, 1, 2, 3, 5, 8, 13]
TOFS = [0.5, 1, 1.5, 2, 3, 5, 8, 13, 21, 34]
MAX_EXTRA_DAYS = 16.0

settings = {
    "waits": WAITS,
    "tofs": TOFS,
    "max_extra_days": MAX_EXTRA_DAYS
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

experiments = [
    ("A_NORMAL_NORMAL", "NORMAL", "D_NORMAL_2"),
    ("B_SPECIAL_NORMAL", "SPECIAL", "D_SPECIAL_2")
]

for name, first_status, previous_key in experiments:

    first_move = step103["branches"]["D"][
        "targets"]["2"]["best_" + first_status.lower()]

    state22 = extend(
        base, 2, first_move, first_status
    )

    second_move = step104["branches"][previous_key][
        "targets"]["25"]["best_normal"]

    assert second_move is not None

    state23 = extend(
        state22, 25, second_move, "NORMAL"
    )

    assert len(state23["route"]) == 23
    assert state23["current_time"] <= problem.max_time

    metadata = {
        "route": state23["route"],
        "arrival": state23["current_time"],
        "special_used": state23["special_used"]
    }

    if name not in report["branches"]:
        report["branches"][name] = {
            "metadata": metadata,
            "targets": {}
        }

    branch = report["branches"][name]
    assert branch["metadata"] == metadata

    remaining = [
        t for t in range(problem.n_tomatoes)
        if t not in state23["route"]
    ]

    print("\nRRUGA:", name, flush=True)
    print("Domate:", len(state23["route"]), "/49", flush=True)
    print("Dita:", state23["current_time"], flush=True)
    print("Speciale:", state23["special_used"], "/5", flush=True)

    for target in remaining:

        key = str(target)

        if key in branch["targets"]:
            continue

        best_normal = None
        best_special = None
        min_dv = float("inf")

        for wait in WAITS:
            for tof in TOFS:

                if wait + tof > MAX_EXTRA_DAYS:
                    continue

                start = state23["current_time"] + wait
                arrival = start + tof

                if arrival > problem.max_time:
                    continue

                try:
                    dv = float(problem.compute_transfer(
                        25, target, start, tof
                    ))
                except Exception:
                    continue

                if not math.isfinite(dv):
                    continue

                min_dv = min(min_dv, dv)

                move = {
                    "from": 25,
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
                        (best_normal["arrival"], best_normal["dv"])
                    ):
                        best_normal = move

                elif (
                    dv < problem.exception_threshold
                    and state23["special_used"] <
                        problem.n_exceptions
                ):
                    if (
                        best_special is None
                        or (arrival, dv) <
                        (best_special["arrival"], best_special["dv"])
                    ):
                        best_special = move

        branch["targets"][key] = {
            "min_dv": (
                min_dv if math.isfinite(min_dv) else None
            ),
            "best_normal": best_normal,
            "best_special": best_special
        }

        save()

        checked = len(branch["targets"])
        if checked % 5 == 0 or checked == len(remaining):
            print(
                f"Kontrolluar: {checked}/{len(remaining)}",
                flush=True
            )

    print("Rruga u analizua.", flush=True)

print("\nHAPI 105 - ANALIZA PERFUNDOI!", flush=True)
