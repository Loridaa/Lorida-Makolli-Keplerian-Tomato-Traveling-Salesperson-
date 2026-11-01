
import json
import math
import sys
from pathlib import Path
from collections import defaultdict, Counter
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 18

SOURCE = RESULTS / f"beam_v84_diverse_level_{TARGET-1}.json"
OUTPUT = RESULTS / f"beam_v84_preselect_level_{TARGET}.json"
CACHE = RESULTS / f"beam_v84_diverse_cache_{TARGET-1}.json"

WIDTH = 50

QUOTAS = {
    2: 2,
    3: 22,
    4: 22,
    5: 4
}

WAITS = [0, 0.5, 1, 2, 3, 5, 8, 13]
FLIGHTS = [0.5, 1, 1.5, 2, 3, 5, 8, 13, 21, 34]

problem = TomatoProblem(
    str(ROOT / "data/easy.kttsp")
)


def save(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)


def summary(beam):
    counts = Counter(s["special_used"] for s in beam)

    print("\nPERMBLEDHJA E V8.3 - NIVELI", TARGET)
    print("=" * 50)
    print(f"Domate: {TARGET}/{problem.n_tomatoes}")
    print("Rruge aktive:", len(beam))

    print("Bashkesi te ndryshme domatesh:", len({
        frozenset(s["route"]) for s in beam
    }))

    print("Domate fundore te ndryshme:", len({
        s["route"][-1] for s in beam
    }))

    print("\nKOHA MINIMALE SIPAS SPECIALEVE")

    for used in sorted(counts):
        best = min(
            s["current_time"]
            for s in beam
            if s["special_used"] == used
        )

        print(
            f"{used} speciale: {counts[used]} rruge | "
            f"koha {best:.1f} dite | "
            f"mbeten {problem.max_time-best:.1f} dite"
        )

    print("\nCheckpoint:", OUTPUT.name)
    print("V8 NIVELI", TARGET, "PERFUNDOI!")


if not SOURCE.exists():
    raise FileNotFoundError(SOURCE.name)

if OUTPUT.exists():
    print("Checkpoint ekziston. Nuk u mbishkrua.")
    summary(json.loads(OUTPUT.read_text())["beam"])
    sys.exit(0)


parents = json.loads(SOURCE.read_text())["beam"]

if not parents:
    raise ValueError("Nuk ka rruge fillestare.")

if any(len(s["route"]) != TARGET-1 for s in parents):
    raise ValueError("Gjatesia e rrugeve nuk perputhet.")

cache = (
    json.loads(CACHE.read_text())
    if CACHE.exists() else {}
)


def scan(parent):
    current = parent["route"][-1]
    now = float(parent["current_time"])
    used = int(parent["special_used"])
    visited = set(parent["route"])

    groups = defaultdict(list)
    calculations = 0

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
                    calculations += 1
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
    alternatives = {"NORMAL": [], "SPECIAL": []}

    for (target, status), options in groups.items():
        options.sort(
            key=lambda m: (m["arrival"], m["dv"])
        )

        first = options[0]
        primary[status].append(first)

        # Ruajme edhe nje kohe alternative
        for option in options[1:]:
            gap = option["arrival"] - first["arrival"]

            if 2 <= gap <= 20:
                alternatives[status].append(option)
                break

    for status in primary:
        primary[status].sort(
            key=lambda m: (m["arrival"], m["dv"])
        )

        alternatives[status].sort(
            key=lambda m: (m["arrival"], m["dv"])
        )

    moves = (
        primary["NORMAL"][:12]
        + alternatives["NORMAL"][:4]
        + primary["SPECIAL"][:8]
        + alternatives["SPECIAL"][:2]
    )

    return moves, calculations


def create_child(parent, move):
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


print("BEAM SEARCH V8.3 - HYBRID")
print("=" * 55)
print(f"Niveli: {TARGET-1} -> {TARGET}")
print("Rruge fillestare:", len(parents), flush=True)

candidates = []
calculations = 0

for i, parent in enumerate(parents):
    key = json.dumps([
        parent["route"],
        parent["current_time"],
        parent["special_used"]
    ])

    if key in cache:
        moves = cache[key]
        source = "CACHE"
    else:
        moves, count = scan(parent)
        calculations += count

        cache[key] = moves
        save(CACHE, cache)
        source = "KERKIM"

    for move in moves:
        candidates.append(create_child(parent, move))

    normals = sum(
        m["status"] == "NORMAL" for m in moves
    )

    print(
        f"Prindi {i+1}/{len(parents)}: "
        f"{normals} normale, "
        f"{len(moves)-normals} speciale | {source}",
        flush=True
    )


if not candidates:
    raise RuntimeError("Nuk u gjet asnje vazhdim.")


# Heqim kandidatet identike
unique = {}

for state in candidates:
    key = (
        tuple(state["route"]),
        round(state["current_time"], 6),
        state["special_used"]
    )
    unique[key] = state

candidates = list(unique.values())

def score(s):
    now = float(s["current_time"])
    used = int(s["special_used"])
    remaining = problem.n_tomatoes - len(s["route"])

    avg = (
        (problem.max_time - now) / remaining
        if remaining > 0 else 0
    )

    time_penalty = 20 * max(0, 2.5 - avg)

    return (
        now + 10 * used + time_penalty,
        now
    )

candidates.sort(key=score)

groups = defaultdict(list)

for state in candidates:
    groups[state["special_used"]].append(state)

for used in groups:
    groups[used].sort(key=score)


selected = []
seen = set()
set_counts = Counter()
endpoint_counts = Counter()
category_counts = Counter()


def add(state, max_set, max_endpoint):
    if len(selected) >= WIDTH:
        return False

    signature = (
        tuple(state["route"]),
        round(state["current_time"], 6),
        state["special_used"]
    )

    tomato_set = frozenset(state["route"])
    endpoint = state["route"][-1]
    used = state["special_used"]

    if signature in seen:
        return False

    if set_counts[tomato_set] >= max_set:
        return False

    if endpoint_counts[endpoint] >= max_endpoint:
        return False

    if used == 2 and category_counts[2] >= 2:
        return False

    if used == 5 and category_counts[5] >= 4:
        return False

    selected.append(state)
    seen.add(signature)

    set_counts[tomato_set] += 1
    endpoint_counts[endpoint] += 1
    category_counts[used] += 1

    return True


def select(max_set, max_endpoint):
    # Ruajme kategori te ndryshme specialesh
    for used, quota in QUOTAS.items():
        for state in groups.get(used, []):
            if len(selected) >= WIDTH:
                return

            if category_counts[used] >= quota:
                break

            add(state, max_set, max_endpoint)

    # Plotesojme me kandidatet tjere
    for state in candidates:
        if len(selected) >= WIDTH:
            return

        add(state, max_set, max_endpoint)


# Faza 1: maksimumi 1 rruge per bashkesi
select(max_set=1, max_endpoint=3)

# Faza 2: lejojme 2 variante
if len(selected) < WIDTH:
    select(max_set=2, max_endpoint=6)

# Faza 3: plotesojme vendet e lira
if len(selected) < WIDTH:
    select(max_set=3, max_endpoint=10)

if len(selected) < WIDTH:
    select(max_set=WIDTH, max_endpoint=WIDTH)


if not selected:
    raise RuntimeError("Nuk u zgjodh asnje rruge.")


# Validim i gjendjeve te reja me PyKEP
for state in selected:
    route = state["route"]
    transfers = state["transfers"]

    assert len(route) == TARGET
    assert len(set(route)) == TARGET
    assert len(transfers) == TARGET-1
    assert state["current_time"] <= problem.max_time

    special_count = sum(
        t["status"] == "SPECIAL"
        for t in transfers
    )

    assert special_count == state["special_used"]
    assert special_count <= problem.n_exceptions

    last = transfers[-1]

    dv = float(problem.compute_transfer(
        last["from"],
        last["to"],
        last["start_time"],
        last["tof"]
    ))

    assert math.isfinite(dv)
    assert abs(dv-last["dv"]) < 0.1

    if last["status"] == "NORMAL":
        assert dv < problem.dv_threshold
    else:
        assert (
            problem.dv_threshold
            <= dv < problem.exception_threshold
        )


result = {
    "method": "Beam Search V8.3 - Hybrid Continuation",
    "source": SOURCE.name,
    "beam_width": WIDTH,
    "beam": selected,
    "evaluations_new": calculations,
    "completed": any(
        len(s["route"]) == problem.n_tomatoes
        for s in selected
    )
}

save(OUTPUT, result)

print("\nKandidate unike:", len(candidates))
print("Llogaritje te reja:", calculations)

summary(selected)
