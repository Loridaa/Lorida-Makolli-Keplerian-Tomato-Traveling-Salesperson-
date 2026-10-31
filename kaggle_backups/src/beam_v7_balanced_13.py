
import json
import math
from pathlib import Path
from collections import defaultdict, Counter
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_v7_level_12.json"
OUTPUT = RESULTS / "beam_v7_balanced_level_13.json"
CACHE = RESULTS / "beam_v7_balanced_cache_12.json"

problem = TomatoProblem(str(ROOT / "data/easy.kttsp"))

WAITS = [0, 0.5, 1, 2, 3, 5, 8, 13]
FLIGHTS = [0.5, 1, 1.5, 2, 3, 5, 8, 13, 21, 34]

# Vendet e rezervuara per kategori
QUOTAS = {1: 8, 2: 16, 3: 16, 4: 8, 5: 2}
WIDTH = 50


def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


if OUTPUT.exists():
    old = json.loads(OUTPUT.read_text())
    print("Checkpoint V7.1 ekziston.")
    print("Rruge:", len(old["beam"]))
    raise SystemExit(0)


beam = json.loads(SOURCE.read_text())["beam"]
assert len(beam) == 40, "Priten 40 rruge nga V7."
assert all(len(s["route"]) == 12 for s in beam)

cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}


def scan(state):
    current = state["route"][-1]
    now = float(state["current_time"])
    used = int(state["special_used"])
    visited = set(state["route"])

    grouped = defaultdict(list)
    evaluations = 0

    for target in range(problem.n_tomatoes):
        if target in visited:
            continue

        for wait in WAITS:
            for tof in FLIGHTS:
                arrival = now + wait + tof

                if tof < problem.min_tof:
                    continue

                if arrival > problem.max_time:
                    continue

                try:
                    dv = float(problem.compute_transfer(
                        current, target, now + wait, tof
                    ))
                    evaluations += 1
                except Exception:
                    continue

                if not math.isfinite(dv):
                    continue

                if dv < problem.dv_threshold:
                    status = "NORMAL"
                elif (
                    dv < problem.exception_threshold
                    and used < problem.n_exceptions
                ):
                    status = "SPECIAL"
                else:
                    continue

                grouped[(target, status)].append({
                    "target": target,
                    "wait": wait,
                    "tof": tof,
                    "arrival": arrival,
                    "dv": dv,
                    "status": status
                })

    primary = {"NORMAL": [], "SPECIAL": []}
    alternative = {"NORMAL": [], "SPECIAL": []}

    for (target, status), options in grouped.items():
        options.sort(
            key=lambda c: (c["arrival"], c["dv"])
        )

        first = options[0]
        primary[status].append(first)

        # Mbajme edhe nje kohe tjeter mberritjeje
        for c in options[1:]:
            gap = c["arrival"] - first["arrival"]

            if 2 <= gap <= 20:
                alternative[status].append(c)
                break

    for status in primary:
        primary[status].sort(
            key=lambda c: (c["arrival"], c["dv"])
        )
        alternative[status].sort(
            key=lambda c: (c["arrival"], c["dv"])
        )

    moves = (
        primary["NORMAL"][:8]
        + alternative["NORMAL"][:4]
        + primary["SPECIAL"][:6]
        + alternative["SPECIAL"][:3]
    )

    return moves, evaluations


def make_child(parent, c):
    current = parent["route"][-1]
    now = float(parent["current_time"])
    status = c["status"]
    target = int(c["target"])

    transfer = {
        "from": current,
        "to": target,
        "start_time": now + c["wait"],
        "wait": c["wait"],
        "tof": c["tof"],
        "dv": c["dv"],
        "status": status
    }

    return {
        "route": parent["route"] + [target],
        "transfers": parent["transfers"] + [transfer],
        "current_time": c["arrival"],
        "special_used": (
            parent["special_used"]
            + int(status == "SPECIAL")
        )
    }


print("V7.1 - KERKIM I BALANCUAR")
print("=" * 50, flush=True)
print("Rruge fillestare:", len(beam))
print("Kapaciteti:", WIDTH, flush=True)

all_candidates = []
total = 0

for i, state in enumerate(beam):
    key = json.dumps([
        state["route"],
        state["current_time"],
        state["special_used"]
    ])

    if key in cache:
        moves = cache[key]
        source = "CACHE"
    else:
        moves, count = scan(state)
        cache[key] = moves
        save(CACHE, cache)
        total += count
        source = "KERKIM"

    for c in moves:
        all_candidates.append((i, make_child(state, c)))

    print(
        f"Prindi {i+1}/40: "
        f"{len(moves)} kandidate | {source}",
        flush=True
    )


if not all_candidates:
    raise RuntimeError("Nuk u gjet asnje kandidat.")


# Ndarja sipas specialeve te shpenzuara
groups = defaultdict(list)

for parent_id, state in all_candidates:
    groups[state["special_used"]].append(
        (parent_id, state)
    )

for used in groups:
    groups[used].sort(
        key=lambda x: x[1]["current_time"]
    )


selected = []
seen = set()
per_parent = Counter()


def add(parent_id, state, limit):
    key = (
        tuple(state["route"]),
        round(state["current_time"], 6),
        state["special_used"]
    )

    if key in seen:
        return False

    if per_parent[parent_id] >= limit:
        return False

    selected.append(state)
    seen.add(key)
    per_parent[parent_id] += 1
    return True


# Ruajme kategori te ndryshme te specialeve
for used, quota in QUOTAS.items():
    options = groups.get(used, [])
    before = len(selected)
    represented = set()

    # Fillimisht nga nje kandidat per prind
    for pid, state in options:
        if len(selected) - before >= quota:
            break

        if pid not in represented:
            if add(pid, state, 5):
                represented.add(pid)

    # Plotesojme kuoten
    for pid, state in options:
        if len(selected) - before >= quota:
            break
        add(pid, state, 5)


# Plotesojme vendet e lira me kandidatet tjere
remaining = sorted(
    all_candidates,
    key=lambda x: (
        x[1]["current_time"]
        + 12 * x[1]["special_used"]
    )
)

for pid, state in remaining:
    if len(selected) >= WIDTH:
        break
    add(pid, state, 6)


# Kontrolli i rezultateve
for state in selected:
    route = state["route"]
    transfers = state["transfers"]

    assert len(route) == 13
    assert len(set(route)) == 13
    assert len(transfers) == 12
    assert state["special_used"] <= problem.n_exceptions
    assert state["current_time"] <= problem.max_time

    assert state["special_used"] == sum(
        t["status"] == "SPECIAL"
        for t in transfers
    )

    last = transfers[-1]
    dv = float(problem.compute_transfer(
        last["from"],
        last["to"],
        last["start_time"],
        last["tof"]
    ))

    assert math.isfinite(dv)
    assert abs(dv - last["dv"]) < 0.1


result = {
    "method": "Beam Search V7.1 Balanced",
    "source": SOURCE.name,
    "beam_width": WIDTH,
    "beam": selected,
    "evaluations_new": total,
    "completed": False
}

save(OUTPUT, result)

print("\nPERMBLEDHJA E V7.1")
print("=" * 50)

print("Domate: 13/49")
print("Kandidate gjithsej:", len(all_candidates))
print("Rruge te ruajtura:", len(selected))

counts = Counter(
    s["special_used"] for s in selected
)

for used, count in sorted(counts.items()):
    print(f"Rruge me {used} speciale: {count}")

print("Koha minimale:", min(
    s["current_time"] for s in selected
))

print("\nKOHA ME E SHKURTER SIPAS SPECIALEVE")
for used in sorted(counts):
    best = min(
        s["current_time"]
        for s in selected
        if s["special_used"] == used
    )
    print(f"{used} speciale: {best:.1f} dite")

print("\nCheckpoint:", OUTPUT.name)
print("V7.1 PERFUNDOI!")
