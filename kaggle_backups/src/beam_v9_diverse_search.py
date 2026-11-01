
import json
import math
import hashlib
from pathlib import Path
from collections import defaultdict
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "beam_v9_frontier.json"

problem = TomatoProblem(str(ROOT / "data/easy.kttsp"))

WAITS = [0, 0.5, 2, 5]
TOFS = [0.5, 1, 1.5, 2, 3, 5, 8]

MAX_PARENTS = 12
BEAM_WIDTH = 30

config = {
    "waits": WAITS,
    "tofs": TOFS,
    "max_parents": MAX_PARENTS,
    "beam_width": BEAM_WIDTH
}

# 1. Lexojme rruget e vjetra.
found = {}

def scan(obj, filename, depth=0):

    if depth > 8:
        return

    if isinstance(obj, dict):
        route = obj.get("route")
        day = obj.get("current_time")
        used = obj.get("special_used")
        transfers = obj.get("transfers")

        if (
            isinstance(route, list)
            and 18 <= len(route) <= 21
            and all(type(x) is int for x in route)
            and len(route) == len(set(route))
            and all(0 <= x < problem.n_tomatoes for x in route)
            and isinstance(day, (int, float))
            and isinstance(used, int)
            and math.isfinite(day)
            and 0 <= day <= problem.max_time
            and 0 <= used <= problem.n_exceptions
            and isinstance(transfers, list)
            and len(transfers) == len(route) - 1
        ):
            key = (
                tuple(route),
                round(float(day), 6),
                used
            )

            if key not in found:
                found[key] = {
                    "route": route,
                    "current_time": float(day),
                    "special_used": used,
                    "transfers": transfers,
                    "source_file": filename
                }

        for key, value in obj.items():
            if key in [
                "route", "transfers", "targets",
                "evaluations", "history", "cache"
            ]:
                continue

            if isinstance(value, (list, dict)):
                scan(value, filename, depth + 1)

    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (list, dict)):
                scan(item, filename, depth + 1)

for path in sorted(RESULTS.glob("*.json")):

    if path == OUTPUT or path.stat().st_size > 60_000_000:
        continue

    try:
        data = json.loads(path.read_text())
        scan(data, path.name)
    except Exception:
        continue

states = list(found.values())

print("Rruge te rikuperuara:", len(states), flush=True)

# 2. Zgjedhim nje nivel me alternativa te mjaftueshme.
preferred_levels = [19, 18, 20, 21]

seed_level = next(
    (
        level for level in preferred_levels
        if sum(len(s["route"]) == level for s in states) >= 8
    ),
    None
)

if seed_level is None:
    seed_level = next(
        (
            level for level in preferred_levels
            if any(len(s["route"]) == level for s in states)
        ),
        None
    )

if seed_level is None:
    raise RuntimeError(
        "Nuk u gjeten rruge me histori transferimesh te plote."
    )

candidates = [
    s for s in states
    if len(s["route"]) == seed_level
    and s["current_time"] < 160
]

if not candidates:
    raise RuntimeError("Nuk u gjeten kandidate me kohe te mjaftueshme.")

# Kosto me e ulet = me pak kohe dhe me pak SPECIAL.
candidates.sort(
    key=lambda s: (
        s["current_time"] + 18 * s["special_used"],
        s["current_time"]
    )
)

seeds = []
last_counts = defaultdict(int)
special_counts = defaultdict(int)

for state in candidates:

    end = state["route"][-1]
    used = state["special_used"]

    if last_counts[end] >= 2:
        continue

    if special_counts[used] >= 6:
        continue

    seeds.append(state)
    last_counts[end] += 1
    special_counts[used] += 1

    if len(seeds) >= MAX_PARENTS:
        break

# Plotesojme vendet nese kufizimet e diversitetit
# kane lene me pak se 12 rruge.
selected_keys = {
    (tuple(s["route"]), s["current_time"], s["special_used"])
    for s in seeds
}

for state in candidates:

    if len(seeds) >= MAX_PARENTS:
        break

    key = (
        tuple(state["route"]),
        state["current_time"],
        state["special_used"]
    )

    if key not in selected_keys:
        seeds.append(state)
        selected_keys.add(key)

def state_id(state):
    raw = json.dumps([
        state["route"],
        state["current_time"],
        state["special_used"]
    ])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

seed_ids = [state_id(s) for s in seeds]

if OUTPUT.exists():
    report = json.loads(OUTPUT.read_text())

    assert report["config"] == config
    assert report["seed_ids"] == seed_ids, (
        "Checkpoint ekzistues ka parametra te tjere."
    )
else:
    report = {
        "method": "Diverse Beam Search V9",
        "config": config,
        "seed_level": seed_level,
        "seed_ids": seed_ids,
        "parents": {},
        "frontier": []
    }

def save():
    tmp = OUTPUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2))
    tmp.replace(OUTPUT)

print("Niveli fillestar:", seed_level, "/49", flush=True)
print("Rruge fillestare:", len(seeds), flush=True)
print("Rruge maksimale per frontier:", BEAM_WIDTH, flush=True)

# 3. Zgjerojme rruget e zgjedhura.
for index, parent in enumerate(seeds, 1):

    pid = state_id(parent)

    if pid in report["parents"]:
        print(
            f"[{index}/{len(seeds)}] Nga checkpoint",
            flush=True
        )
        continue

    route = parent["route"]
    visited = set(route)
    current = route[-1]
    day = parent["current_time"]
    used = parent["special_used"]

    remaining_count = problem.n_tomatoes - len(route)
    average_time = (problem.max_time - day) / remaining_count

    # Nuk lejojme nje hap te konsumojne
    # teper kohe krahasuar me kohen e mbetur.
    max_step = min(18.0, max(3.0, average_time * 3.0))

    normals = []
    specials = []
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

                start = day + wait
                arrival = start + tof

                if arrival > problem.max_time:
                    continue

                try:
                    dv = float(problem.compute_transfer(
                        current, target, start, tof
                    ))
                    evaluations += 1
                except Exception:
                    continue

                if not math.isfinite(dv):
                    continue

                move = {
                    "from": current,
                    "to": target,
                    "start_time": start,
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
                    and used < problem.n_exceptions
                ):
                    if (
                        best_special is None
                        or (arrival, dv) <
                           (best_special["arrival"], best_special["dv"])
                    ):
                        best_special = move

        if best_normal:
            normals.append(best_normal)

        if best_special:
            specials.append(best_special)

    normals.sort(key=lambda m: (m["arrival"], m["dv"]))
    specials.sort(key=lambda m: (m["arrival"], m["dv"]))

    # Ruajme disa zgjedhje te ndryshme per secilin prind.
    children = []

    for status, moves in [
        ("NORMAL", normals[:4]),
        ("SPECIAL", specials[:3])
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

            children.append({
                "route": route + [move["to"]],
                "current_time": move["arrival"],
                "special_used": new_used,
                "transfers": parent["transfers"] + [transfer],
                "parent_id": pid
            })

    report["parents"][pid] = {
        "source": parent["source_file"],
        "start_day": day,
        "start_specials": used,
        "normal_destinations": len(normals),
        "special_destinations": len(specials),
        "evaluations": evaluations,
        "children": children
    }

    save()

    print(
        f"[{index}/{len(seeds)}] "
        f"Fund {current}, dita {day:.1f}, "
        f"speciale {used}/5 | "
        f"NORMAL {len(normals)}, "
        f"SPECIAL {len(specials)}",
        flush=True
    )

# 4. Bashkojme dhe diversifikojme rruget e reja.
all_children = [
    child
    for parent in report["parents"].values()
    for child in parent["children"]
]

unique = {}

for child in all_children:
    key = (
        tuple(child["route"]),
        round(child["current_time"], 6),
        child["special_used"]
    )
    unique[key] = child

ranked = sorted(
    unique.values(),
    key=lambda s: (
        s["current_time"] + 18 * s["special_used"],
        s["current_time"]
    )
)

frontier = []
end_counts = defaultdict(int)
used_counts = defaultdict(int)
frontier_keys = set()

for state in ranked:

    end = state["route"][-1]
    used = state["special_used"]

    if end_counts[end] >= 3 or used_counts[used] >= 12:
        continue

    frontier.append(state)
    end_counts[end] += 1
    used_counts[used] += 1

    frontier_keys.add((
        tuple(state["route"]),
        state["current_time"],
        used
    ))

    if len(frontier) >= BEAM_WIDTH:
        break

for state in ranked:

    if len(frontier) >= BEAM_WIDTH:
        break

    key = (
        tuple(state["route"]),
        state["current_time"],
        state["special_used"]
    )

    if key not in frontier_keys:
        frontier.append(state)
        frontier_keys.add(key)

report["frontier"] = frontier
save()

print("\nREZULTATI V9")
print("=" * 60)
print("Prinder te analizuar:", len(report["parents"]))
print("Rruge te reja unike:", len(unique))
print("Rruge ne frontier:", len(frontier))

if frontier:
    print("Niveli i ri:", len(frontier[0]["route"]), "/49")

    for state in frontier[:10]:
        print(
            f"Fund {state['route'][-1]} | "
            f"Dita {state['current_time']:.2f} | "
            f"SPECIAL {state['special_used']}/5",
            flush=True
        )

print("\nBEAM SEARCH V9 PERFUNDOI!", flush=True)
