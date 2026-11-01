
import json
import math
import hashlib
from pathlib import Path
from collections import defaultdict
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"
SOURCE = RESULTS / "beam_v9_level_22.json"
OUTPUT = RESULTS / "beam_v9_level_23.json"

problem = TomatoProblem(str(ROOT / "data/easy.kttsp"))
source = json.loads(SOURCE.read_text())
frontier = source["frontier"]

WAITS = [0, 0.5, 1, 2, 3, 5, 8, 12]
TOFS = [0.5, 1, 1.5, 2, 3, 5, 8, 13]

MAX_STEP = 18.0
BEAM_WIDTH = 48
MAX_FIVE_SPECIAL_PARENTS = 10

config = {
    "waits": WAITS,
    "tofs": TOFS,
    "max_step": MAX_STEP,
    "beam_width": BEAM_WIDTH,
    "max_five_special_parents": MAX_FIVE_SPECIAL_PARENTS
}

def identity(state):
    payload = [
        state["route"],
        round(float(state["current_time"]), 6),
        int(state["special_used"])
    ]
    raw = json.dumps(payload)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def validate_last(state):
    route = state["route"]
    transfers = state["transfers"]

    if (
        len(route) != 22
        or len(set(route)) != 22
        or len(transfers) != 21
        or state["special_used"] > 5
    ):
        return False

    transfer = transfers[-1]

    if (
        transfer["from"] != route[-2]
        or transfer["to"] != route[-1]
    ):
        return False

    start = float(transfer["start_time"])
    tof = float(transfer["tof"])

    if tof <= 0:
        return False

    arrival = start + tof

    if (
        abs(arrival - state["current_time"]) > 1e-5
        or arrival > problem.max_time
    ):
        return False

    try:
        dv = float(problem.compute_transfer(
            route[-2], route[-1], start, tof
        ))
    except Exception:
        return False

    if not math.isfinite(dv):
        return False

    if abs(dv - float(transfer["dv"])) > 0.1:
        return False

    status = (
        "NORMAL" if dv < problem.dv_threshold
        else "SPECIAL" if dv < problem.exception_threshold
        else "INVALID"
    )

    return status == transfer["status"]

valid = [s for s in frontier if validate_last(s)]

four = sorted(
    [s for s in valid if s["special_used"] == 4],
    key=lambda s: s["current_time"]
)

five = sorted(
    [s for s in valid if s["special_used"] == 5],
    key=lambda s: s["current_time"]
)

selected_five = []
per_end = defaultdict(int)

# Diversitet: jo te gjitha rruget te perfundojne
# ne te njejten domate.
for state in five:
    end = state["route"][-1]

    if per_end[end] < 2:
        selected_five.append(state)
        per_end[end] += 1

    if len(selected_five) >= MAX_FIVE_SPECIAL_PARENTS:
        break

selected_ids = {identity(s) for s in selected_five}

for state in five:
    if len(selected_five) >= MAX_FIVE_SPECIAL_PARENTS:
        break

    if identity(state) not in selected_ids:
        selected_five.append(state)
        selected_ids.add(identity(state))

selected = four + selected_five
seed_ids = [identity(s) for s in selected]

print("Rruge ne nivelin 22:", len(frontier), flush=True)
print("Rruge qe kaluan kontrollin:", len(valid), flush=True)
print("Prinder me 4 SPECIAL:", len(four), flush=True)
print("Prinder me 5 SPECIAL:", len(selected_five), flush=True)
print("Gjithsej per zgjerim:", len(selected), flush=True)

assert selected, "Nuk ka rruge per zgjerim."

if OUTPUT.exists():
    report = json.loads(OUTPUT.read_text())

    assert report["config"] == config
    assert report["seed_ids"] == seed_ids
else:
    report = {
        "method": "Beam Search V9 - level 23",
        "config": config,
        "seed_ids": seed_ids,
        "parents": {},
        "frontier": []
    }

def save():
    tmp = OUTPUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2))
    tmp.replace(OUTPUT)

# KERKIMI I TRANSFERIMEVE
for index, parent in enumerate(selected, 1):

    pid = identity(parent)

    if pid in report["parents"]:
        print(
            f"[{index}/{len(selected)}] Nga checkpoint",
            flush=True
        )
        continue

    route = parent["route"]
    visited = set(route)
    current = int(route[-1])
    day = float(parent["current_time"])
    used = int(parent["special_used"])

    normal_moves = []
    special_moves = []
    near_normal = []
    evaluations = 0

    for target in range(problem.n_tomatoes):

        if target in visited:
            continue

        normals = []
        specials = []
        min_dv = float("inf")

        for wait in WAITS:
            for tof in TOFS:

                if wait + tof > MAX_STEP:
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

                min_dv = min(min_dv, dv)

                move = {
                    "from": current,
                    "to": target,
                    "start_time": depart,
                    "tof": tof,
                    "arrival": arrival,
                    "dv": dv
                }

                if dv < problem.dv_threshold:
                    normals.append(move)

                elif (
                    dv < problem.exception_threshold
                    and used < problem.n_exceptions
                ):
                    specials.append(move)

        if normals:
            # Me i shpejti drejt kesaj domateje.
            fastest = min(
                normals,
                key=lambda m: (m["arrival"], m["dv"])
            )

            normal_moves.append(fastest)

            # Ruajme edhe nje variant me Delta-V me
            # te ulet, nese ka kohe tjeter mberritjeje.
            lowest_dv = min(
                normals,
                key=lambda m: (m["dv"], m["arrival"])
            )

            if abs(
                fastest["arrival"] - lowest_dv["arrival"]
            ) >= 0.5:
                normal_moves.append(lowest_dv)

        if specials:
            special_moves.append(
                min(
                    specials,
                    key=lambda m: (m["arrival"], m["dv"])
                )
            )

        if (
            math.isfinite(min_dv)
            and problem.dv_threshold <= min_dv < 150
        ):
            near_normal.append({
                "to": target,
                "min_dv": min_dv
            })

    normal_moves.sort(
        key=lambda m: (m["arrival"], m["dv"])
    )

    special_moves.sort(
        key=lambda m: (m["arrival"], m["dv"])
    )

    near_normal.sort(key=lambda m: m["min_dv"])

    children = []
    seen_moves = set()

    for status, moves, limit in [
        ("NORMAL", normal_moves, 10),
        ("SPECIAL", special_moves, 4)
    ]:
        selected_moves = []

        for move in moves:
            key = (
                move["to"],
                round(move["arrival"], 6),
                status
            )

            if key in seen_moves:
                continue

            seen_moves.add(key)
            selected_moves.append(move)

            if len(selected_moves) >= limit:
                break

        for move in selected_moves:
            transfer = {
                "from": move["from"],
                "to": move["to"],
                "start_time": move["start_time"],
                "tof": move["tof"],
                "dv": move["dv"],
                "status": status
            }

            children.append({
                "route": route + [move["to"]],
                "current_time": move["arrival"],
                "special_used": (
                    used + (status == "SPECIAL")
                ),
                "transfers": (
                    parent["transfers"] + [transfer]
                )
            })

    report["parents"][pid] = {
        "end": current,
        "day": day,
        "special_used": used,
        "normal_options": len(normal_moves),
        "special_destinations": len(special_moves),
        "near_normal": near_normal[:5],
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
        f"SPECIAL {len(special_moves)} | "
        f"Femije {len(children)}",
        flush=True
    )

# FORMIMI I FRONTIER-IT
all_children = [
    child
    for parent in report["parents"].values()
    for child in parent["children"]
]

unique = {}

for state in all_children:
    key = (
        tuple(sorted(state["route"])),
        state["route"][-1],
        state["special_used"],
        round(state["current_time"] * 2) / 2
    )

    if (
        key not in unique
        or state["current_time"] < unique[key]["current_time"]
    ):
        unique[key] = state

ranked = sorted(
    unique.values(),
    key=lambda s: (
        s["current_time"] + 16 * s["special_used"],
        s["current_time"]
    )
)

new_frontier = []
seen = set()
end_counts = defaultdict(int)

# Synojme me shume rruge qe kane ende SPECIAL.
for used, quota in [(4, 36), (5, 12)]:
    count = 0

    for state in ranked:
        if len(new_frontier) >= BEAM_WIDTH:
            break

        if count >= quota:
            break

        if state["special_used"] != used:
            continue

        end = state["route"][-1]

        if end_counts[end] >= 4:
            continue

        key = identity(state)

        if key in seen:
            continue

        new_frontier.append(state)
        seen.add(key)
        end_counts[end] += 1
        count += 1

# Plotesojme vendet qe kane mbetur.
for state in ranked:
    if len(new_frontier) >= BEAM_WIDTH:
        break

    key = identity(state)

    if key not in seen:
        new_frontier.append(state)
        seen.add(key)

report["frontier"] = new_frontier
save()

print("\nREZULTATI V9 - NIVELI 23")
print("=" * 60, flush=True)

print("Prinder te analizuar:", len(report["parents"]), flush=True)
print("Rruge te reja unike:", len(unique), flush=True)
print("Frontier i ri:", len(new_frontier), flush=True)

for used in range(6):
    group = [
        s for s in new_frontier
        if s["special_used"] == used
    ]

    if group:
        best = min(
            group,
            key=lambda s: s["current_time"]
        )

        print(
            f"SPECIAL {used}/5 | "
            f"Rruge {len(group)} | "
            f"Dita me e hershme "
            f"{best['current_time']:.2f}",
            flush=True
        )

print("\nHAPI 111 - KERKIMI PERFUNDOI!", flush=True)
