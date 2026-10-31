
import json
import math
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

V8 = RESULTS / "beam_v8_diverse_level_25.json"
V82 = RESULTS / "beam_v82_timeaware_level_25.json"
OUTPUT = RESULTS / "beam_v83_hybrid_level_25.json"

WIDTH = 50
TARGET = 25

# Shperndarja e synuar e 50 rrugeve
QUOTAS = {
    2: 2,
    3: 22,
    4: 22,
    5: 4
}

for path in [V8, V82]:
    if not path.exists():
        raise FileNotFoundError(path.name)


def signature(s):
    return (
        tuple(s["route"]),
        round(float(s["current_time"]), 6),
        int(s["special_used"])
    )


def diversity(beam):
    return (
        len({frozenset(s["route"]) for s in beam}),
        len({s["route"][-1] for s in beam})
    )


# 1. Bashkojme dy versionet
beam8 = json.loads(V8.read_text())["beam"]
beam82 = json.loads(V82.read_text())["beam"]

unique = {}

for state in beam8 + beam82:
    unique.setdefault(signature(state), state)

candidates = list(unique.values())

groups = defaultdict(list)

for state in candidates:
    groups[state["special_used"]].append(state)

print("BEAM SEARCH V8.3 - HYBRID")
print("=" * 55)
print("V8:", len(beam8), "rruge")
print("V8.2:", len(beam82), "rruge")
print("Kandidate unike:", len(candidates))

print("Bashkesi ne bashkimin e dy versioneve:",
      diversity(candidates)[0])

for used, quota in QUOTAS.items():
    if len(groups[used]) < quota:
        raise ValueError(
            f"Nuk ka mjaft rruge me {used} speciale."
        )


# 2. Zgjedhja e rrugeve
selected = []
seen = set()

set_counts = Counter()
endpoint_counts = Counter()
category_counts = Counter()

def add(state):
    key = signature(state)

    if key in seen:
        return False

    used = state["special_used"]

    if category_counts[used] >= QUOTAS[used]:
        return False

    selected.append(state)
    seen.add(key)

    category_counts[used] += 1
    set_counts[frozenset(state["route"])] += 1
    endpoint_counts[state["route"][-1]] += 1

    return True


def score(state):
    now = float(state["current_time"])
    endpoint = state["route"][-1]

    remaining = 49 - len(state["route"])
    average = (200 - now) / remaining

    # Penalizim i lehte per kohe te vogel
    time_penalty = 10 * max(0, 3 - average)

    # Favorizojme domate fundore te ndryshme
    endpoint_penalty = 5 * endpoint_counts[endpoint]

    return now + time_penalty + endpoint_penalty


if OUTPUT.exists():
    print("\nCheckpoint V8.3 ekziston. Po lexohet.")
    selected = json.loads(OUTPUT.read_text())["beam"]

else:
    # Ruajme rrugen me te shpejte nga cdo
    # kategori manovrash speciale
    for used in sorted(QUOTAS):
        fastest = min(
            groups[used],
            key=lambda s: s["current_time"]
        )
        add(fastest)

    # Fillimisht kerkojme bashkesi unike,
    # pastaj lejojme variante kur eshte nevoja
    phases = [
        (1, 3),
        (1, 5),
        (1, 50),
        (2, 5),
        (2, 50),
        (50, 50)
    ]

    for max_set, max_endpoint in phases:

        for used in [3, 4, 5, 2]:

            while category_counts[used] < QUOTAS[used]:

                eligible = [
                    s for s in groups[used]
                    if signature(s) not in seen
                    and set_counts[frozenset(s["route"])]
                        < max_set
                    and endpoint_counts[s["route"][-1]]
                        < max_endpoint
                ]

                if not eligible:
                    break

                best = min(eligible, key=score)
                add(best)

    if len(selected) != WIDTH:
        raise RuntimeError(
            f"U zgjodhen {len(selected)} nga {WIDTH} rruge."
        )

    # 3. Validimi i te dhenave
    for s in selected:
        assert len(s["route"]) == TARGET
        assert len(set(s["route"])) == TARGET
        assert len(s["transfers"]) == TARGET - 1
        assert math.isfinite(s["current_time"])
        assert s["current_time"] <= 200
        assert s["special_used"] <= 5

        specials = sum(
            t["status"] == "SPECIAL"
            for t in s["transfers"]
        )

        assert specials == s["special_used"]

    result = {
        "method": "Beam Search V8.3 Hybrid",
        "sources": [V8.name, V82.name],
        "beam_width": WIDTH,
        "beam": selected,
        "completed": False
    }

    temp = OUTPUT.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2))
    temp.replace(OUTPUT)


# 4. Krahasimi
print("\nPERMBLEDHJA E V8.3")
print("=" * 55)

print("Domate: 25/49")
print("Rruge aktive:", len(selected))

sets8, ends8 = diversity(beam8)
sets82, ends82 = diversity(beam82)
sets83, ends83 = diversity(selected)

print("\nKRAHASIMI I DIVERSITETIT")
print("V8 bashkesi:", sets8)
print("V8.2 bashkesi:", sets82)
print("V8.3 bashkesi:", sets83)

print("V8 domate fundore:", ends8)
print("V8.2 domate fundore:", ends82)
print("V8.3 domate fundore:", ends83)

print("\nKOHA MINIMALE SIPAS SPECIALEVE")

counts = Counter(s["special_used"] for s in selected)

for used in sorted(counts):
    minimum = min(
        s["current_time"]
        for s in selected
        if s["special_used"] == used
    )

    print(
        f"{used} speciale: {counts[used]} rruge | "
        f"koha {minimum:.1f} dite | "
        f"mbeten {200-minimum:.1f} dite"
    )

keys8 = {signature(s) for s in beam8}
keys82 = {signature(s) for s in beam82}
keys83 = {signature(s) for s in selected}

print("\nORIGJINA E RRUGEVE")
print("Nga V8:", len(keys83 & keys8))
print("Nga V8.2:", len(keys83 & keys82))
print(
    "Te perbashketa me te dyja:",
    len(keys83 & keys8 & keys82)
)

print("\nRruga me e shpejte:")

fastest = min(
    selected,
    key=lambda s: s["current_time"]
)

print("Koha:", fastest["current_time"])
print("Speciale:", fastest["special_used"])
print("Rruga:", fastest["route"])

print("\nCheckpoint:", OUTPUT.name)
print("V8.3 HYBRID PERFUNDOI!")
