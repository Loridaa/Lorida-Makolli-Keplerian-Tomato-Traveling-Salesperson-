
import sys
import json
import math
from pathlib import Path
from collections import defaultdict, Counter
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

TARGET = int(sys.argv[1])
SOURCE = RESULTS / f"beam_v7_balanced_level_{TARGET-1}.json"
OUTPUT = RESULTS / f"beam_v7_balanced_level_{TARGET}.json"
CACHE = RESULTS / f"beam_v7_balanced_cache_{TARGET-1}.json"

WAITS = [0, 0.5, 1, 2, 3, 5, 8, 13]
FLIGHTS = [0.5, 1, 1.5, 2, 3, 5, 8, 13, 21, 34]

WIDTH = 50
QUOTAS = {1: 6, 2: 14, 3: 18, 4: 10, 5: 2}

problem = TomatoProblem(str(ROOT / "data/easy.kttsp"))


def save(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)


if not SOURCE.exists():
    raise FileNotFoundError(f"Mungon: {SOURCE.name}")

if OUTPUT.exists():
    print(f"Checkpoint {OUTPUT.name} ekziston tashme.")
    print("Nuk u mbishkrua.")
    sys.exit(0)

beam = json.loads(SOURCE.read_text())["beam"]

if not beam or any(len(s["route"]) != TARGET-1 for s in beam):
    raise ValueError("Checkpoint fillestar nuk eshte i sakte.")

cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}


def scan(parent):
    current = parent["route"][-1]
    now = float(parent["current_time"])
    used = int(parent["special_used"])
    visited = set(parent["route"])

    groups = defaultdict(list)
    evaluations = 0

    for target in range(problem.n_tomatoes):
        if target in visited:
            continue

        for wait in WAITS:
            for tof in FLIGHTS:
                start = now + wait
                arrival = start + tof

                if tof < problem.min_tof:
                    continue
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

                if dv < problem.dv_threshold:
                    status = "NORMAL"
                elif (
                    dv < problem.exception_threshold
                    and used < problem.n_exceptions
                ):
                    status = "SPECIAL"
                else:
                    continue

                groups[(target, status)].append({
                    "target": target,
                    "wait": wait,
                    "tof": tof,
                    "arrival": arrival,
                    "dv": dv,
                    "status": status
                })

    primary = {"NORMAL": [], "SPECIAL": []}
    alternative = {"NORMAL": [], "SPECIAL": []}

    for (target, status), options in groups.items():
        options.sort(
            key=lambda c: (c["arrival"], c["dv"])
        )

        first = options[0]
        primary[status].append(first)

        # Mbajme edhe nje kohe alternative
        for c in options[1:]:
            difference = c["arrival"] - first["arrival"]
            if 2 <= difference <= 20:
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
        primary["NORMAL"][:12]
        + alternative["NORMAL"][:4]
        + primary["SPECIAL"][:8]
        + alternative["SPECIAL"][:2]
    )

    return moves, evaluations


def make_child(parent, move):
    current = parent["route"][-1]
    now = float(parent["current_time"])
    target = int(move["target"])
    status = move["status"]

    transfer = {
        "from": current,
        "to": target,
        "start_time": now + move["wait"],
        "wait": move["wait"],
        "tof": move["tof"],
        "dv": move["dv"],
        "status": status
    }

    return {
        "route": parent["route"] + [target],
        "transfers": parent["transfers"] + [transfer],
        "current_time": move["arrival"],
        "special_used": (
            parent["special_used"]
            + int(status == "SPECIAL")
        )
    }


print("BEAM SEARCH V7.3 - VAZHDIM")
print("=" * 50)
print(f"Niveli: {TARGET-1} -> {TARGET}")
print("Rruge fillestare:", len(beam), flush=True)

candidates = []
total = 0

for i, parent in enumerate(beam):
    key = json.dumps([
        parent["route"],
        parent["current_time"],
        parent["special_used"]
    ])

    if key in cache:
        moves = cache[key]
        origin = "CACHE"
    else:
        moves, count = scan(parent)
        total += count
        cache[key] = moves
        save(CACHE, cache)
        origin = "KERKIM"

    for move in moves:
        candidates.append(
            (i, make_child(parent, move))
        )

    normals = sum(
        m["status"] == "NORMAL" for m in moves
    )

    print(
        f"Prindi {i+1}/{len(beam)}: "
        f"{normals} normale, "
        f"{len(moves)-normals} speciale | {origin}",
        flush=True
    )


if not candidates:
    raise RuntimeError("Nuk u gjet asnje vazhdim.")


groups = defaultdict(list)

for pid, state in candidates:
    groups[state["special_used"]].append((pid, state))

for used in groups:
    groups[used].sort(
        key=lambda x: x[1]["current_time"]
    )


selected = []
seen = set()
parent_counts = Counter()


def add(pid, state, limit=5):
    if len(selected) >= WIDTH:
        return False

    key = (
        tuple(state["route"]),
        round(state["current_time"], 6),
        state["special_used"]
    )

    if key in seen or parent_counts[pid] >= limit:
        return False

    selected.append(state)
    seen.add(key)
    parent_counts[pid] += 1
    return True


# Ruajme kategori me numer te ndryshem specialesh
for used, quota in QUOTAS.items():
    options = groups.get(used, [])
    before = len(selected)
    represented = set()

    for pid, state in options:
        if len(selected) - before >= quota:
            break

        if pid not in represented and add(pid, state):
            represented.add(pid)

    for pid, state in options:
        if len(selected) - before >= quota:
            break
        add(pid, state)


# Plotesojme vendet e lira
remaining = sorted(
    candidates,
    key=lambda x: (
        x[1]["current_time"]
        + 12 * x[1]["special_used"]
    )
)

for pid, state in remaining:
    if len(selected) >= WIDTH:
        break
    add(pid, state, limit=6)


# Validim i transferimeve me PyKEP
for state in selected:
    route = state["route"]
    transfers = state["transfers"]

    assert len(route) == TARGET
    assert len(set(route)) == TARGET
    assert len(transfers) == TARGET - 1

    special_count = 0
    now = 0.0

    for j, t in enumerate(transfers):
        start = float(t["start_time"])
        tof = float(t["tof"])

        assert t["from"] == route[j]
        assert t["to"] == route[j+1]
        assert start >= now - 1e-7
        assert tof >= problem.min_tof

        dv = float(problem.compute_transfer(
            t["from"], t["to"], start, tof
        ))

        assert math.isfinite(dv)
        assert abs(dv - t["dv"]) < 0.1

        if dv < problem.dv_threshold:
            assert t["status"] == "NORMAL"
        elif dv < problem.exception_threshold:
            assert t["status"] == "SPECIAL"
            special_count += 1
        else:
            raise ValueError("DV mbi kufirin")

        now = start + tof

    assert special_count == state["special_used"]
    assert special_count <= problem.n_exceptions
    assert now <= problem.max_time
    assert abs(now - state["current_time"]) < 1e-6


result = {
    "method": "Beam Search V7.3 Balanced",
    "source": SOURCE.name,
    "beam_width": WIDTH,
    "beam": selected,
    "evaluations_new": total,
    "completed": any(
        len(s["route"]) == problem.n_tomatoes
        for s in selected
    )
}

save(OUTPUT, result)


print("\nPERMBLEDHJA E V7.3")
print("=" * 50)
print(f"Domate: {TARGET}/{problem.n_tomatoes}")
print("Kandidate:", len(candidates))
print("Rruge aktive:", len(selected))
print("Llogaritje:", total)

counts = Counter(
    s["special_used"] for s in selected
)

print("\nKOHA MINIMALE SIPAS SPECIALEVE")

for used in sorted(counts):
    shortest = min(
        s["current_time"]
        for s in selected
        if s["special_used"] == used
    )

    print(
        f"{used} speciale: {counts[used]} rruge | "
        f"koha minimale {shortest:.1f} dite"
    )

print("\nCheckpoint:", OUTPUT.name)
print("V7.3 PERFUNDOI!")
