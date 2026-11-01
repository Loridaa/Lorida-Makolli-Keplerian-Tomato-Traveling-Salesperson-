
from pathlib import Path
from utils_users import TomatoProblem
import json
import math

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

RESCUE = RESULTS / "beam_v817_rescue_level_21.json"
PREVIOUS = RESULTS / "beam_v822_rescue_scan.json"
OUTPUT = RESULTS / "beam_v823_lookahead.json"

problem = TomatoProblem(str(ROOT / "data/easy.kttsp"))

states = json.loads(RESCUE.read_text())["beam"]
previous = json.loads(PREVIOUS.read_text())

matches = [
    s for s in states
    if abs(s["current_time"] - 114.5) < 1e-6
    and s["route"][-1] == 41
    and s["special_used"] == 3
]

assert len(matches) == 1, "Rruga D nuk u gjet."
base = matches[0]

targets = previous["branches"]["D"]["targets"]

normal2 = targets["2"]["best_normal"]
special2 = targets["2"]["best_special"]

assert normal2 is not None
assert special2 is not None

# Gjejme nje destinacion tjeter SPECIAL, jo domaten 2
others = [
    (m["arrival"], m["dv"], int(t), m)
    for t, entry in targets.items()
    if int(t) != 2
    for m in [entry["best_special"]]
    if m is not None
]

others.sort()
assert others, "Nuk u gjet alternativa e trete."

other_target = others[0][2]
other_move = others[0][3]

experiments = [
    ("D_NORMAL_2", 2, normal2, "NORMAL"),
    ("D_SPECIAL_2", 2, special2, "SPECIAL"),
    ("D_SPECIAL_OTHER", other_target, other_move, "SPECIAL")
]

WAITS = [0, 1, 3, 8]
TOFS = [0.5, 1, 2, 5, 8, 13]

settings = {
    "waits": WAITS,
    "tofs": TOFS
}

if OUTPUT.exists():
    report = json.loads(OUTPUT.read_text())
    assert report["settings"] == settings
else:
    report = {"settings": settings, "branches": {}}

def save():
    temp = OUTPUT.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2))
    temp.replace(OUTPUT)

for name, target_first, first, status in experiments:

    start = 114.5 + first["wait"]
    arrival = start + first["tof"]

    dv = float(problem.compute_transfer(
        41, target_first, start, first["tof"]
    ))

    assert math.isfinite(dv)
    assert abs(arrival - first["arrival"]) < 1e-6
    assert target_first not in base["route"]

    if status == "NORMAL":
        assert dv < problem.dv_threshold
    else:
        assert problem.dv_threshold <= dv < problem.exception_threshold

    used = 3 + (status == "SPECIAL")
    route = base["route"] + [target_first]

    assert len(route) == 22
    assert len(route) == len(set(route))

    metadata = {
        "route": route,
        "arrival": arrival,
        "special_used": used,
        "first_status": status,
        "first_dv": dv
    }

    if name not in report["branches"]:
        report["branches"][name] = {
            "metadata": metadata,
            "targets": {}
        }

    branch = report["branches"][name]
    assert branch["metadata"] == metadata, "Cache nuk perputhet."

    remaining = [
        t for t in range(problem.n_tomatoes)
        if t not in route
    ]

    print(f"\nRRUGA: {name}", flush=True)
    print(f"41 -> {target_first} | {status}", flush=True)
    print(f"Dita: {arrival:.2f}", flush=True)
    print(f"Delta-V: {dv:.2f}", flush=True)
    print(f"Speciale: {used}/5", flush=True)

    for target in remaining:

        key = str(target)
        if key in branch["targets"]:
            continue

        best_normal = None
        best_special = None
        minimum = float("inf")

        for wait in WAITS:
            for tof in TOFS:

                depart = arrival + wait
                finish = depart + tof

                if finish > problem.max_time:
                    continue

                try:
                    value = float(problem.compute_transfer(
                        target_first, target, depart, tof
                    ))
                except Exception:
                    continue

                if not math.isfinite(value):
                    continue

                minimum = min(minimum, value)

                move = {
                    "from": target_first,
                    "to": target,
                    "wait": wait,
                    "tof": tof,
                    "arrival": finish,
                    "dv": value
                }

                if value < problem.dv_threshold:
                    if (
                        best_normal is None
                        or (finish, value) <
                        (best_normal["arrival"], best_normal["dv"])
                    ):
                        best_normal = move

                elif (
                    value < problem.exception_threshold
                    and used < problem.n_exceptions
                ):
                    if (
                        best_special is None
                        or (finish, value) <
                        (best_special["arrival"], best_special["dv"])
                    ):
                        best_special = move

        branch["targets"][key] = {
            "min_dv": (
                minimum if math.isfinite(minimum) else None
            ),
            "best_normal": best_normal,
            "best_special": best_special
        }

        save()

        checked = len(branch["targets"])
        if checked % 9 == 0 or checked == len(remaining):
            print(
                f"Kontrolluar: {checked}/{len(remaining)}",
                flush=True
            )

    normals = [
        x["best_normal"]
        for x in branch["targets"].values()
        if x["best_normal"] is not None
    ]

    specials = [
        x["best_special"]
        for x in branch["targets"].values()
        if x["best_special"] is not None
    ]

    print(f"\nREZULTATI {name}", flush=True)
    print("NORMAL:", len(normals), flush=True)
    print("SPECIAL:", len(specials), flush=True)

    if normals:
        best = min(normals, key=lambda x: x["arrival"])
        print("NORMAL me i hershem:", best, flush=True)

    if specials:
        best = min(specials, key=lambda x: x["arrival"])
        print("SPECIAL me i hershem:", best, flush=True)

print("\nHAPI 104 - ANALIZA PERFUNDOI!", flush=True)
