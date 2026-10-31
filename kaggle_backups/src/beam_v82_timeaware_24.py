
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_v8_diverse_level_23.json"
CACHE = RESULTS / "beam_v8_diverse_cache_23.json"
OLD = RESULTS / "beam_v8_diverse_level_24.json"
OUTPUT = RESULTS / "beam_v82_timeaware_level_24.json"

WIDTH = 50
TARGET = 24

# Synimi i ri: me pak rruge te vonshme
QUOTAS = {2: 2, 3: 22, 4: 22, 5: 4}
MAX_CATEGORY = {2: 2, 5: 5}

problem = TomatoProblem(
    str(ROOT / "data/easy.kttsp")
)

def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)

def diversity(beam):
    return (
        len({frozenset(s["route"]) for s in beam}),
        len({s["route"][-1] for s in beam})
    )

def show_summary(beam):
    sets, ends = diversity(beam)
    counts = Counter(s["special_used"] for s in beam)

    print("\nPERMBLEDHJA E V8.2")
    print("=" * 50)
    print("Domate:", TARGET, "/49")
    print("Rruge aktive:", len(beam))
    print("Bashkesi te ndryshme:", sets)
    print("Domate fundore te ndryshme:", ends)

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
            f"mbeten {200-best:.1f} dite"
        )

if not SOURCE.exists() or not CACHE.exists():
    raise FileNotFoundError(
        "Mungon niveli 23 ose cache-i 23."
    )

if OUTPUT.exists():
    beam = json.loads(OUTPUT.read_text())["beam"]
    print("V8.2 ekziston tashme.")
    show_summary(beam)
    raise SystemExit(0)

parents = json.loads(SOURCE.read_text())["beam"]
cache = json.loads(CACHE.read_text())
old_beam = json.loads(OLD.read_text())["beam"]

candidates = []
missing = []

print("BEAM SEARCH V8.2 - TIME AWARE")
print("=" * 50)
print("Nisja: 23 -> 24")
print("Prinder:", len(parents))

# Rikrijojme kandidatet pa bere skanim te ri
for parent in parents:
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
        status = move["status"]
        arrival = float(move["arrival"])
        tof = float(move["tof"])
        wait = float(move["wait"])

        if target in parent["route"]:
            continue

        used = parent["special_used"] + (
            status == "SPECIAL"
        )

        if used > problem.n_exceptions:
            continue

        if arrival > problem.max_time:
            continue

        if tof < problem.min_tof:
            continue

        transfer = {
            "from": current,
            "to": target,
            "start_time": now + wait,
            "wait": wait,
            "tof": tof,
            "dv": float(move["dv"]),
            "status": status
        }

        candidates.append({
            "route": parent["route"] + [target],
            "transfers": (
                parent["transfers"] + [transfer]
            ),
            "current_time": arrival,
            "special_used": int(used)
        })

if missing:
    raise ValueError(
        f"Mungon cache per {len(missing)} prinder."
    )

# Heqim dublikatat identike
unique = {}

for s in candidates:
    key = (
        tuple(s["route"]),
        round(s["current_time"], 6),
        s["special_used"]
    )
    unique[key] = s

candidates = list(unique.values())

if not candidates:
    raise RuntimeError("Nuk ka kandidate.")

# Koha ka perparesi, por edhe specialet kushtojne.
# Penaliteti 10 eshte heuristike eksperimentale.
def score(s):
    used = s["special_used"]
    now = s["current_time"]
    remaining = problem.n_tomatoes - len(s["route"])

    avg = (problem.max_time - now) / remaining

    # Penalizim per buxhet shume te vogel kohe
    time_penalty = 20 * max(0, 2.5 - avg)

    return (
        now + 10 * used + time_penalty,
        now
    )

candidates.sort(key=score)

groups = defaultdict(list)

for s in candidates:
    groups[s["special_used"]].append(s)

selected = []
seen = set()
set_counts = Counter()
endpoint_counts = Counter()
category_counts = Counter()

def add(s, max_set, max_endpoint):
    if len(selected) >= WIDTH:
        return False

    used = s["special_used"]
    signature = (
        tuple(s["route"]),
        round(s["current_time"], 6),
        used
    )

    visited = frozenset(s["route"])
    endpoint = s["route"][-1]

    if signature in seen:
        return False

    if (
        used in MAX_CATEGORY
        and category_counts[used] >= MAX_CATEGORY[used]
    ):
        return False

    if set_counts[visited] >= max_set:
        return False

    if endpoint_counts[endpoint] >= max_endpoint:
        return False

    selected.append(s)
    seen.add(signature)
    set_counts[visited] += 1
    endpoint_counts[endpoint] += 1
    category_counts[used] += 1

    return True


# Fillimisht tentojme te arrijme kuotat
# duke mbajtur sa me shume bashkesi te ndryshme
for max_set, max_endpoint in [
    (1, 4),
    (2, 7),
    (3, 12)
]:
    for used in [3, 4, 5, 2]:
        quota = QUOTAS[used]

        for s in groups.get(used, []):
            if category_counts[used] >= quota:
                break

            if len(selected) >= WIDTH:
                break

            add(s, max_set, max_endpoint)


# Plotesojme vendet e mbetura sipas score
for max_set, max_endpoint in [
    (1, 5),
    (2, 8),
    (3, 15),
    (WIDTH, WIDTH)
]:
    for s in candidates:
        if len(selected) >= WIDTH:
            break
        add(s, max_set, max_endpoint)

if not selected:
    raise RuntimeError("V8.2 nuk zgjodhi asnje rruge.")

# Verifikim me PyKEP i transferimit te fundit
for s in selected:
    assert len(s["route"]) == TARGET
    assert len(set(s["route"])) == TARGET
    assert len(s["transfers"]) == TARGET - 1
    assert s["special_used"] <= 5
    assert s["current_time"] <= 200

    specials = sum(
        t["status"] == "SPECIAL"
        for t in s["transfers"]
    )
    assert specials == s["special_used"]

    last = s["transfers"][-1]

    assert abs(
        last["start_time"]
        + last["tof"]
        - s["current_time"]
    ) < 1e-6

    dv = float(problem.compute_transfer(
        last["from"],
        last["to"],
        last["start_time"],
        last["tof"]
    ))

    assert math.isfinite(dv)
    assert abs(dv - last["dv"]) < 0.1

    if last["status"] == "NORMAL":
        assert dv < problem.dv_threshold
    else:
        assert (
            problem.dv_threshold
            <= dv < problem.exception_threshold
        )

result = {
    "method": "Beam Search V8.2 - Time Aware",
    "source": SOURCE.name,
    "cache": CACHE.name,
    "beam_width": WIDTH,
    "beam": selected,
    "completed": False
}

save(OUTPUT, result)

print("Kandidate unike:", len(candidates))
print("Llogaritje te reja skanimi: 0")

show_summary(selected)

old_sets, old_ends = diversity(old_beam)
new_sets, new_ends = diversity(selected)

print("\nKRAHASIMI V8 / V8.2")
print("=" * 50)
print("V8 bashkesi:", old_sets)
print("V8.2 bashkesi:", new_sets)
print("V8 domate fundore:", old_ends)
print("V8.2 domate fundore:", new_ends)

old_keys = {
    (tuple(s["route"]),
     round(s["current_time"], 6),
     s["special_used"])
    for s in old_beam
}

new_only = sum(
    (
        tuple(s["route"]),
        round(s["current_time"], 6),
        s["special_used"]
    ) not in old_keys
    for s in selected
)

print("Rruge te reja kundrejt V8:", new_only)
print("\nCheckpoint:", OUTPUT.name)
print("V8.2 PERFUNDOI!")
