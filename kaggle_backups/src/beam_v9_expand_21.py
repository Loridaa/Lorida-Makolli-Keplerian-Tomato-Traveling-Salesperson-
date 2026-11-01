
import json
import math
import hashlib
from pathlib import Path
from collections import defaultdict
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_v9_frontier.json"
OUTPUT = RESULTS / "beam_v9_level_21.json"

problem = TomatoProblem(str(ROOT / "data/easy.kttsp"))

frontier = json.loads(SOURCE.read_text())["frontier"]

WAITS = [0, 0.5, 1, 2, 4, 8]
TOFS = [0.5, 1, 2, 3, 5, 8]

MAX_PARENTS = 12
BEAM_WIDTH = 30

config = {
    "waits": WAITS,
    "tofs": TOFS,
    "max_parents": MAX_PARENTS,
    "beam_width": BEAM_WIDTH
}

assert len(frontier) > 0

for state in frontier:
    route = state["route"]
    assert len(route) == 20
    assert len(set(route)) == 20
    assert len(state["transfers"]) == 19
    assert state["special_used"] <= 5
    assert state["current_time"] <= 200

# Zgjedhim rruge me buxhete te ndryshme SPECIAL.
quotas = {3: 4, 4: 6, 5: 2}
selected = []
seen = set()

def identity(state):
    raw = json.dumps([
        state["route"],
        round(state["current_time"], 6),
        state["special_used"]
    ])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

for used, quota in quotas.items():

    group = sorted(
        [
            s for s in frontier
            if s["special_used"] == used
        ],
        key=lambda s: s["current_time"]
    )

    chosen = []
    ends = set()

    # Fillimisht zgjedhim funde te ndryshme.
    for state in group:
        if state["route"][-1] not in ends:
            chosen.append(state)
            ends.add(state["route"][-1])

        if len(chosen) >= quota:
            break

    # Plotesojme vendet e tjera.
    for state in group:
        if len(chosen) >= quota:
            break

        if identity(state) not in {
            identity(s) for s in chosen
        }:
            chosen.append(state)

    for state in chosen:
        pid = identity(state)
        if pid not in seen:
            selected.append(state)
            seen.add(pid)

# Nese ka me pak se 12, plotesojme me rruge te tjera.
remaining = sorted(
    frontier,
    key=lambda s: (
        s["current_time"] + 18 * s["special_used"],
        s["current_time"]
    )
)

for state in remaining:
    if len(selected) >= MAX_PARENTS:
        break

    pid = identity(state)

    if pid not in seen:
        selected.append(state)
        seen.add(pid)

seed_ids = [identity(s) for s in selected]

if OUTPUT.exists():
    report = json.loads(OUTPUT.read_text())

    assert report["config"] == config
    assert report["seed_ids"] == seed_ids
else:
    report = {
        "method": "Beam Search V9 - level 21",
        "config": config,
        "seed_ids": seed_ids,
        "parents": {},
        "frontier": []
    }

def save():
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(OUTPUT)

print("Niveli fillestar: 20/49", flush=True)
print("Rruge ne frontier:", len(frontier), flush=True)
print("Prinder te zgjedhur:", len(selected), flush=True)

for used in sorted({s["special_used"] for s in selected}):
    count = sum(s["special_used"] == used for s in selected)
    print(
        f"Me {used} SPECIAL te perdorura: {count}",
        flush=True
    )

# Zgjerojme secilin prind.
for index, parent in enumerate(selected, 1):

    pid = identity(parent)

    if pid in report["parents"]:
        print(
            f"[{index}/{len(selected)}] Nga checkpoint",
            flush=True
        )
        continue

    route = parent["route"]
    current = route[-1]
    day = float(parent["current_time"])
    used = int(parent["special_used"])
    visited = set(route)

    remaining_count = problem.n_tomatoes - len(route)
    average_time = (problem.max_time - day) / remaining_count

    max_step = min(18.0, max(4.0, average_time * 3))

    normal_moves = []
    special_moves = []
    evaluations = 0

    for target in range(problem.n_tomatoes):

        if target in visited:
            continue

        best_normal = None
        best_special = None

        for wait in WAITS:
            for tof in TOFS:

                if wait + tof > max_step:
                    continue

                depart = day + wait
                arrival = depart + tof

                if arrival > problem.max_time:
                    continue

                try:
                    dv = float(problem.compute_transfer(
                        current, target, depart, tof
                    ))
                    evaluations += 1
                except Exception:
                    continue

                if not math.isfinite(dv):
                    continue

                move = {
                    "from": current,
                    "to": target,
                    "start_time": depart,
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
                    and used < problem.n_exceptions
                ):

                    if (
                        best_special is None
                        or (arrival, dv) <
                        (best_special["arrival"], best_special["dv"])
                    ):
                        best_special = move

        if best_normal is not None:
            normal_moves.append(best_normal)

        if best_special is not None:
            special_moves.append(best_special)

    normal_moves.sort(
        key=lambda m: (m["arrival"], m["dv"])
    )
    special_moves.sort(
        key=lambda m: (m["arrival"], m["dv"])
    )

    children = []

    for status, moves in [
        ("NORMAL", normal_moves[:5]),
        ("SPECIAL", special_moves[:3])
    ]:

        for move in moves:

            new_used = used + (status == "SPECIAL")

            transfer = {
                "from": move["from"],
                "to": move["to"],
                "start_time": move["start_time"],
                "tof": move["tof"],
                "dv": move["dv"],
                "status": status
            }

            child = {
                "route": route + [move["to"]],
                "current_time": move["arrival"],
                "special_used": new_used,
                "transfers": parent["transfers"] + [transfer]
            }

            children.append(child)

    report["parents"][pid] = {
        "end": current,
        "day": day,
        "special_used": used,
        "normal_destinations": len(normal_moves),
        "special_destinations": len(special_moves),
        "evaluations": evaluations,
        "children": children
    }

    save()

    print(
        f"[{index}/{len(selected)}] "
        f"Fund {current} | "
        f"Dita {day:.2f} | "
        f"SPECIAL {used}/5 | "
        f"NORMAL {len(normal_moves)} | "
        f"SPECIAL {len(special_moves)}",
        flush=True
    )

# Bashkojme te gjithe femijet.
all_children = [
    child
    for parent in report["parents"].values()
    for child in parent["children"]
]

# Nese kemi te njejtat domate te vizituara,
# fundin dhe numrin e SPECIAL, mbajme kohen me te hershme.
unique = {}

for state in all_children:

    key = (
        tuple(sorted(state["route"])),
        state["route"][-1],
        state["special_used"]
    )

    if (
        key not in unique
        or state["current_time"] < unique[key]["current_time"]
    ):
        unique[key] = state

ranked = sorted(
    unique.values(),
    key=lambda s: (
        s["current_time"] + 18 * s["special_used"],
        s["current_time"]
    )
)

# Ruajme larmishmeri ne numrin e manovrave SPECIAL.
frontier_new = []
selected_keys = set()
end_counts = defaultdict(int)

level_quotas = {
    0: 4, 1: 6, 2: 8,
    3: 10, 4: 14, 5: 6
}

for used in sorted(level_quotas):

    count = 0

    for state in ranked:

        if len(frontier_new) >= BEAM_WIDTH:
            break

        if state["special_used"] != used:
            continue

        if count >= level_quotas[used]:
            break

        end = state["route"][-1]

        if end_counts[end] >= 3:
            continue

        key = (
            tuple(state["route"]),
            state["current_time"],
            used
        )

        if key in selected_keys:
            continue

        frontier_new.append(state)
        selected_keys.add(key)
        end_counts[end] += 1
        count += 1

for state in ranked:

    if len(frontier_new) >= BEAM_WIDTH:
        break

    key = (
        tuple(state["route"]),
        state["current_time"],
        state["special_used"]
    )

    if key not in selected_keys:
        frontier_new.append(state)
        selected_keys.add(key)

report["frontier"] = frontier_new
save()

print("\nREZULTATI I BEAM SEARCH V9")
print("=" * 60)
print("Prinder te analizuar:", len(report["parents"]))
print("Rruge te reja unike:", len(unique))
print("Frontier i ri:", len(frontier_new))

if frontier_new:
    print("Niveli i arritur:", len(frontier_new[0]["route"]))

    for used in sorted({
        s["special_used"] for s in frontier_new
    }):
        group = [
            s for s in frontier_new
            if s["special_used"] == used
        ]

        best = min(group, key=lambda s: s["current_time"])

        print(
            f"SPECIAL {used}/5 | "
            f"Rruge {len(group)} | "
            f"Dita me e hershme {best['current_time']:.2f}"
        )

print("\nHAPI 109 PERFUNDOI!", flush=True)
