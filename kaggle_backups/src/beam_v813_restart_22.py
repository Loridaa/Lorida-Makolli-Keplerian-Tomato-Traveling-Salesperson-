
import json
import math
from pathlib import Path
from collections import Counter

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_v8_diverse_level_21.json"
CACHE = RESULTS / "beam_v8_diverse_cache_21.json"

OLD = RESULTS / "beam_v8_diverse_level_22.json"
OUTPUT = RESULTS / "beam_v813_wide_level_22.json"

WIDTH = 100
LEVEL = 22

# Kufij eksperimentale per kategorite
CATEGORY_CAPS = {2: 10, 3: 40, 4: 55, 5: 15}

def load(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))

def save(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8"
    )
    temp.replace(path)

def signature(state):
    return (
        tuple(state["route"]),
        round(float(state["current_time"]), 6),
        int(state["special_used"])
    )

def tomato_set(state):
    return frozenset(state["route"])

def diversity(states):
    return (
        len({tomato_set(s) for s in states}),
        len({s["route"][-1] for s in states})
    )

parents = load(SOURCE)["beam"]
cache = load(CACHE)
old_beam = load(OLD)["beam"]

print("HAPI 58 - V8.13 WIDE BEAM SEARCH")
print("=" * 60)
print("Niveli: 21 -> 22")
print("Beam Width:", WIDTH)
print("Prinder:", len(parents))
print("Llogaritje te reja PyKEP: 0")
print("=" * 60, flush=True)

# 1. Rikrijojme kandidatet nga cache
pool = {}
missing = []

for parent in parents:

    assert len(parent["route"]) == 21

    key = json.dumps([
        parent["route"],
        parent["current_time"],
        parent["special_used"]
    ])

    if key not in cache:
        missing.append(key)
        continue

    current = parent["route"][-1]
    now = float(parent["current_time"])

    for move in cache[key]:

        target = int(move["target"])
        wait = float(move["wait"])
        tof = float(move["tof"])
        arrival = float(move["arrival"])
        dv = float(move["dv"])
        status = move["status"]

        if target in parent["route"]:
            continue

        if wait < 0 or tof <= 0:
            continue

        if not math.isfinite(dv) or dv < 0:
            continue

        if not (now < arrival <= 200):
            continue

        if abs(now + wait + tof - arrival) > 1e-6:
            continue

        if status == "NORMAL" and dv < 100:
            extra = 0

        elif status == "SPECIAL" and 100 <= dv < 600:
            extra = 1

        else:
            continue

        used = int(parent["special_used"]) + extra

        if used > 5:
            continue

        transfer = {
            "from": current,
            "to": target,
            "start_time": now + wait,
            "wait": wait,
            "tof": tof,
            "dv": dv,
            "status": status
        }

        child = {
            "route": parent["route"] + [target],
            "transfers": parent["transfers"] + [transfer],
            "current_time": arrival,
            "special_used": used
        }

        sig = signature(child)
        previous = pool.get(sig)

        # Ne rast dublikimi, ruajme
        # rrugen me DeltaV kumulative me te ulet.
        if previous is None:
            pool[sig] = child
        else:
            new_dv = sum(t["dv"] for t in child["transfers"])
            old_dv = sum(t["dv"] for t in previous["transfers"])

            if new_dv < old_dv:
                pool[sig] = child

if missing:
    raise RuntimeError(
        f"Mungon cache per {len(missing)} prinder."
    )

candidates = list(pool.values())

print("\nKANDIDATET E DISPONUESHEM")
print("=" * 60)

available_sets, available_ends = diversity(candidates)

print("Kandidate unike:", len(candidates))
print("Bashkesi te ndryshme:", available_sets)
print("Domate fundore te ndryshme:", available_ends)

if len(candidates) < WIDTH:
    print(
        "Verejtje: ka me pak se 100 "
        "kandidate te vlefshem."
    )

# 2. Perzgjedhja V8.13
selected = []
seen = set()

set_counts = Counter()
endpoint_counts = Counter()
category_counts = Counter()

def add(state):

    sig = signature(state)

    if sig in seen:
        return False

    selected.append(state)
    seen.add(sig)

    set_counts[tomato_set(state)] += 1
    endpoint_counts[state["route"][-1]] += 1
    category_counts[state["special_used"]] += 1

    return True

def base_score(state):

    now = float(state["current_time"])
    used = int(state["special_used"])

    remaining = 49 - len(state["route"])
    days_left = 200 - now
    avg = days_left / remaining

    # Penalitet per kohe te pamjaftueshme
    # dhe konsumim te specialeve.
    time_penalty = 14 * max(0, 3.5 - avg) ** 2

    return (
        now
        + 16 * used
        + time_penalty
    )

def selection_score(state):

    endpoint = state["route"][-1]
    used = state["special_used"]

    return (
        base_score(state)
        + 5 * endpoint_counts[endpoint]
        + 2 * category_counts[used]
    )

# 3. Ruajme fillimisht perfaqesues
# nga kategori te ndryshme speciale.
seed_targets = {2: 2, 3: 10, 4: 15, 5: 5}

for used in [3, 4, 5, 2]:

    for _ in range(seed_targets[used]):

        options = [
            s for s in candidates
            if s["special_used"] == used
            and signature(s) not in seen
            and set_counts[tomato_set(s)] == 0
        ]

        if not options:
            break

        best = min(options, key=selection_score)
        add(best)

# 4. Plotesojme deri ne 100 rruge
# duke i dhene perparesi diversitetit.
phases = [
    (1, 5, True),
    (1, 100, True),
    (1, 100, False),
    (2, 100, True),
    (100, 100, True),
    (100, 100, False)
]

for max_set, max_endpoint, use_caps in phases:

    while len(selected) < WIDTH:

        eligible = []

        for s in candidates:

            if signature(s) in seen:
                continue

            if set_counts[tomato_set(s)] >= max_set:
                continue

            endpoint = s["route"][-1]

            if endpoint_counts[endpoint] >= max_endpoint:
                continue

            used = s["special_used"]

            if use_caps:
                if category_counts[used] >= CATEGORY_CAPS[used]:
                    continue

            eligible.append(s)

        if not eligible:
            break

        best = min(eligible, key=selection_score)
        add(best)

# 5. Validimi
for state in selected:

    assert len(state["route"]) == LEVEL
    assert len(set(state["route"])) == LEVEL
    assert len(state["transfers"]) == LEVEL - 1
    assert state["current_time"] <= 200

    actual_specials = sum(
        t["status"] == "SPECIAL"
        for t in state["transfers"]
    )

    assert actual_specials == state["special_used"]
    assert actual_specials <= 5

save(OUTPUT, {
    "method": "V8.13 Wide Beam Restart",
    "source": SOURCE.name,
    "cache": CACHE.name,
    "beam_width": WIDTH,
    "available_candidates": len(candidates),
    "beam": selected,
    "completed": False
})

# 6. Krahasimi me V8
print("\nPERMBLEDHJA V8.13 - NIVELI 22")
print("=" * 60)

old_sets, old_ends = diversity(old_beam)
new_sets, new_ends = diversity(selected)

print("Domate:", LEVEL, "/49")
print("Rruge aktive:", len(selected))

print("\nKRAHASIMI I DIVERSITETIT")
print("V8 bashkesi:", old_sets)
print("V8.13 bashkesi:", new_sets)
print("V8 domate fundore:", old_ends)
print("V8.13 domate fundore:", new_ends)

print("\nKOHA MINIMALE SIPAS SPECIALEVE")

counts = Counter(
    s["special_used"] for s in selected
)

for used in sorted(counts):

    best = min(
        s["current_time"]
        for s in selected
        if s["special_used"] == used
    )

    remaining_days = 200 - best
    avg = remaining_days / (49 - LEVEL)

    print(
        f"{used} speciale: "
        f"{counts[used]} rruge | "
        f"koha {best:.2f} | "
        f"mbeten {remaining_days:.2f} dite | "
        f"{avg:.2f} dite/domate"
    )

print("\nLLOJI I TRANSFERIMIT TE FUNDIT")

last_moves = Counter(
    s["transfers"][-1]["status"]
    for s in selected
)

print("NORMAL:", last_moves["NORMAL"])
print("SPECIAL:", last_moves["SPECIAL"])

print("\nRRUGET ME TE MIRA ME SPECIALE REZERVE")

reserve = sorted(
    [s for s in selected if s["special_used"] <= 4],
    key=lambda s: (
        s["current_time"],
        s["special_used"]
    )
)

for state in reserve[:5]:

    print(
        f"Koha {state['current_time']:.2f} | "
        f"Speciale {state['special_used']} | "
        f"Fund {state['route'][-1]} | "
        f"Mbeten {200-state['current_time']:.2f} dite"
    )

print("\nCheckpoint:", OUTPUT.name)
print("V8.13 PERFUNDOI!")
