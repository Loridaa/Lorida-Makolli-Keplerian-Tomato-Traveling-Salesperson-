
import json
import math
from pathlib import Path
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_v817_rescue_level_21.json"
OUTPUT = RESULTS / "beam_v822_rescue_scan.json"

problem = TomatoProblem(str(ROOT / "data/easy.kttsp"))
states = json.loads(SOURCE.read_text())["beam"]

CANDIDATES = [
    ("D", 114.5, 41),
    ("E", 115.0, 10),
    ("F", 118.0, 2)
]

WAITS = [0, 1, 3, 8]
TOFS = [0.5, 1, 2, 5, 8, 13]

settings = {
    "waits": WAITS,
    "tofs": TOFS,
    "candidates": CANDIDATES
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

for name, day, end in CANDIDATES:

    matches = [
        s for s in states
        if abs(s["current_time"] - day) < 1e-6
        and s["route"][-1] == end
        and s["special_used"] == 3
    ]

    assert len(matches) == 1, (
        f"Rruga {name} nuk u gjet sakte."
    )

    state = matches[0]
    route = state["route"]

    assert len(route) == 21
    assert len(set(route)) == 21

    remaining = [
        t for t in range(problem.n_tomatoes)
        if t not in route
    ]

    metadata = {
        "route": route,
        "day": day,
        "end": end,
        "special_used": 3
    }

    if name not in report["branches"]:
        report["branches"][name] = {
            "metadata": metadata,
            "targets": {}
        }

    branch = report["branches"][name]
    assert branch["metadata"] == metadata

    print(f"\nRRUGA {name}", flush=True)
    print("Dita:", day, flush=True)
    print("Domatja fundore:", end, flush=True)
    print("Speciale:", 3, "/5", flush=True)

    for target in remaining:

        key = str(target)

        if key in branch["targets"]:
            continue

        min_dv = float("inf")
        best_normal = None
        best_special = None

        for wait in WAITS:
            for tof in TOFS:

                start = day + wait
                arrival = start + tof

                if arrival > problem.max_time:
                    continue

                try:
                    dv = float(problem.compute_transfer(
                        end, target, start, tof
                    ))
                except Exception:
                    continue

                if not math.isfinite(dv):
                    continue

                min_dv = min(min_dv, dv)

                move = {
                    "from": end,
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
                    and state["special_used"] <
                        problem.n_exceptions
                ):
                    if (
                        best_special is None
                        or (arrival, dv) <
                           (best_special["arrival"],
                            best_special["dv"])
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
        if checked % 7 == 0 or checked == len(remaining):
            print(
                f"Progresi {name}: "
                f"{checked}/{len(remaining)}",
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

    print(f"\nREZULTATI - RRUGA {name}", flush=True)
    print("NORMAL:", len(normals), flush=True)
    print("SPECIAL:", len(specials), flush=True)

    for status, moves in [
        ("NORMAL", normals),
        ("SPECIAL", specials)
    ]:
        if moves:
            best = min(
                moves,
                key=lambda x: (x["arrival"], x["dv"])
            )

            print(
                f"{status} me i hershem: "
                f"{end} -> {best['to']} | "
                f"Dita {best['arrival']:.2f} | "
                f"DV {best['dv']:.2f}",
                flush=True
            )

print("\nHAPI 103 - ANALIZA PERFUNDOI!", flush=True)
