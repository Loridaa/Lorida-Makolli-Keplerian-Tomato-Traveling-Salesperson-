
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_v7_balanced_level_16.json"
CACHE = RESULTS / "beam_v7_balanced_cache_16.json"
OLD = RESULTS / "beam_v7_balanced_level_17.json"
OUTPUT = RESULTS / "beam_v8_diverse_level_17.json"

WIDTH = 50
DIVERSE_SLOTS = 40

problem = TomatoProblem(str(ROOT / "data/easy.kttsp"))

def save(data):
    temp = OUTPUT.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(OUTPUT)

if OUTPUT.exists():
    print("V8 ekziston tashme:", OUTPUT.name)
    raise SystemExit(0)

parents = json.loads(SOURCE.read_text())["beam"]
cache = json.loads(CACHE.read_text())
old_beam = json.loads(OLD.read_text())["beam"]

candidates = []

# Rikrijojme kandidatet nga cache-i i V7
for parent in parents:
    key = json.dumps([
        parent["route"],
        parent["current_time"],
        parent["special_used"]
    ])

    if key not in cache:
        raise ValueError(
            "Mungon cache per rrugen: "
            + str(parent["route"])
        )

    for move in cache[key]:
        target = int(move["target"])
        status = move["status"]
        used = parent["special_used"] + (
            status == "SPECIAL"
        )

        if target in parent["route"]:
            continue

        if used > problem.n_exceptions:
            continue

        transfer = {
            "from": parent["route"][-1],
            "to": target,
            "start_time": (
                parent["current_time"] + move["wait"]
            ),
            "wait": move["wait"],
            "tof": move["tof"],
            "dv": move["dv"],
            "status": status
        }

        child = {
            "route": parent["route"] + [target],
            "transfers": (
                parent["transfers"] + [transfer]
            ),
            "current_time": move["arrival"],
            "special_used": int(used)
        }

        candidates.append(child)

# Heqim kandidatet plotesisht identike
unique = {}

for s in candidates:
    key = (
        tuple(s["route"]),
        round(s["current_time"], 6),
        s["special_used"]
    )
    unique[key] = s

candidates = list(unique.values())

# Renditja merr parasysh kohen dhe specialet
candidates.sort(
    key=lambda s: (
        s["current_time"]
        + 12 * s["special_used"],
        s["current_time"]
    )
)

groups = defaultdict(list)

for s in candidates:
    groups[s["special_used"]].append(s)

selected = []
seen = set()
set_counts = Counter()
endpoint_counts = Counter()
context_counts = Counter()
category_counts = Counter()


def tomato_set(s):
    return tuple(sorted(s["route"]))


def try_add(s, max_set, max_endpoint, max_context):
    if len(selected) >= WIDTH:
        return False

    signature = (
        tuple(s["route"]),
        round(s["current_time"], 6),
        s["special_used"]
    )

    visited = tomato_set(s)
    endpoint = s["route"][-1]

    context = (
        visited,
        endpoint,
        s["special_used"]
    )

    if signature in seen:
        return False

    if set_counts[visited] >= max_set:
        return False

    if endpoint_counts[endpoint] >= max_endpoint:
        return False

    if context_counts[context] >= max_context:
        return False

    # Jo me shume se 2 rruge me 5 speciale
    if s["special_used"] == 5:
        if category_counts[5] >= 2:
            return False

    selected.append(s)
    seen.add(signature)

    set_counts[visited] += 1
    endpoint_counts[endpoint] += 1
    context_counts[context] += 1
    category_counts[s["special_used"]] += 1

    return True


def fill(limit, quotas, max_set,
         max_endpoint, max_context):

    # Fillimisht ruajme kategori te ndryshme
    for used, quota in quotas.items():
        for s in groups.get(used, []):
            if len(selected) >= limit:
                return

            if category_counts[used] >= quota:
                break

            try_add(
                s, max_set,
                max_endpoint, max_context
            )

    # Pastaj plotesojme vendet e lira
    for s in candidates:
        if len(selected) >= limit:
            break

        try_add(
            s, max_set,
            max_endpoint, max_context
        )


print("BEAM SEARCH V8 - DIVERSITY")
print("=" * 50)

print("Prinder:", len(parents))
print("Kandidate unike:", len(candidates))

# Faza 1: maksimumi 1 rruge
# per secilen bashkesi domatesh
fill(
    limit=DIVERSE_SLOTS,
    quotas={1: 2, 2: 11, 3: 16, 4: 9, 5: 2},
    max_set=1,
    max_endpoint=3,
    max_context=1
)

print("Faza 1 - rruge:", len(selected))

# Faza 2: mbajme edhe kohe alternative
fill(
    limit=WIDTH,
    quotas={1: 3, 2: 14, 3: 20, 4: 11, 5: 2},
    max_set=2,
    max_endpoint=6,
    max_context=2
)

# Nese mungojne vende, lejojme me shume variante
fill(
    limit=WIDTH,
    quotas={1: 3, 2: 14, 3: 20, 4: 11, 5: 2},
    max_set=3,
    max_endpoint=10,
    max_context=2
)

if not selected:
    raise RuntimeError("V8 nuk zgjodhi asnje rruge.")

# Verifikojme kandidatet e zgjedhur
for s in selected:
    assert len(s["route"]) == 17
    assert len(set(s["route"])) == 17
    assert len(s["transfers"]) == 16
    assert s["current_time"] <= problem.max_time

    specials = sum(
        t["status"] == "SPECIAL"
        for t in s["transfers"]
    )

    assert specials == s["special_used"]
    assert specials <= problem.n_exceptions

    last = s["transfers"][-1]

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
            <= dv
            < problem.exception_threshold
        )

result = {
    "method": "Beam Search V8 - Diversity",
    "source": SOURCE.name,
    "beam_width": WIDTH,
    "beam": selected,
    "completed": False
}

save(result)


def diversity(beam):
    return (
        len({frozenset(s["route"]) for s in beam}),
        len({s["route"][-1] for s in beam})
    )


old_sets, old_ends = diversity(old_beam)
new_sets, new_ends = diversity(selected)

print("\nPERMBLEDHJA E V8")
print("=" * 50)
print("Domate: 17/49")
print("Rruge aktive:", len(selected))

print("\nKRAHASIMI I DIVERSITETIT")
print("V7 - Bashkesi domatesh:", old_sets)
print("V8 - Bashkesi domatesh:", new_sets)
print("V7 - Domate fundore:", old_ends)
print("V8 - Domate fundore:", new_ends)

print("\nKOHA SIPAS SPECIALEVE")

for used in sorted(category_counts):
    times = [
        s["current_time"]
        for s in selected
        if s["special_used"] == used
    ]

    print(
        f"{used} speciale: "
        f"{len(times)} rruge | "
        f"koha minimale {min(times):.1f} dite"
    )

print("\nCheckpoint:", OUTPUT.name)
print("V8 PERFUNDOI!")
