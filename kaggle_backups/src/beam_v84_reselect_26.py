
import json
import math
from pathlib import Path
from collections import Counter

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_v83_hybrid_level_25.json"
CACHE = RESULTS / "beam_v83_hybrid_cache_25.json"
OLD = RESULTS / "beam_v83_hybrid_level_26.json"
OUTPUT = RESULTS / "beam_v84_diverse_level_26.json"

WIDTH = 50
TARGET = 26

# Kufij fleksibel per secilen kategori
CAPS = {2: 3, 3: 22, 4: 24, 5: 8}


def read(path):
    if not path.exists():
        raise FileNotFoundError(path.name)
    return json.loads(path.read_text())


def save(data):
    temp = OUTPUT.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(OUTPUT)


def signature(s):
    return (
        tuple(s["route"]),
        round(float(s["current_time"]), 6),
        int(s["special_used"])
    )


def tomato_set(s):
    return frozenset(s["route"])


def diversity(states):
    return (
        len({tomato_set(s) for s in states}),
        len({s["route"][-1] for s in states})
    )


parents = read(SOURCE)["beam"]
cache = read(CACHE)
old_beam = read(OLD)["beam"]

print("V8.4 - DIVERSITY AND TIME RESELECTION")
print("=" * 55)
print("Niveli: 25 -> 26")
print("Prinder:", len(parents))

if OUTPUT.exists():
    selected = read(OUTPUT)["beam"]
    print("Checkpoint V8.4 ekziston. Po lexohet.")

else:
    # 1. Rikrijojme kandidatet nga cache
    unique = {}
    missing = []

    for parent in parents:

        assert len(parent["route"]) == TARGET - 1

        key = json.dumps([
            parent["route"],
            parent["current_time"],
            parent["special_used"]
        ])

        if key not in cache:
            missing.append(key)
            continue

        now = float(parent["current_time"])
        current = parent["route"][-1]

        for move in cache[key]:

            target = int(move["target"])
            status = move["status"]
            wait = float(move["wait"])
            tof = float(move["tof"])
            dv = float(move["dv"])
            arrival = float(move["arrival"])

            used = int(parent["special_used"])
            used += int(status == "SPECIAL")

            if target in parent["route"]:
                continue

            if status not in ("NORMAL", "SPECIAL"):
                continue

            if used > 5 or arrival > 200:
                continue

            if wait < 0 or tof <= 0:
                continue

            if not math.isfinite(dv):
                continue

            if status == "NORMAL" and not (dv < 100):
                continue

            if status == "SPECIAL" and not (100 <= dv < 600):
                continue

            start = now + wait

            if abs(start + tof - arrival) > 1e-6:
                continue

            transfer = {
                "from": current,
                "to": target,
                "start_time": start,
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

            unique[signature(child)] = child

    if missing:
        raise RuntimeError(
            f"Mungon cache per {len(missing)} prinder. "
            "Nuk u krijua checkpoint i ri."
        )

    candidates = list(unique.values())

    if not candidates:
        raise RuntimeError("Nuk ka kandidate ne cache.")

    available_sets, available_ends = diversity(candidates)

    print("Kandidate unike:", len(candidates))
    print("Bashkesi te disponueshme:", available_sets)
    print("Domate fundore te disponueshme:", available_ends)
    print("Llogaritje te reja PyKEP: 0")

    # 2. Vleresimi sipas kohes dhe specialeve
    def base_score(s):
        now = float(s["current_time"])
        used = int(s["special_used"])
        remaining = 49 - len(s["route"])

        days_per_tomato = (200 - now) / remaining
        time_penalty = 15 * max(0, 3 - days_per_tomato)

        return now + 8 * used + time_penalty

    selected = []
    seen = set()
    set_counts = Counter()
    endpoint_counts = Counter()
    category_counts = Counter()

    def add(s):
        key = signature(s)

        if key in seen:
            return False

        selected.append(s)
        seen.add(key)

        set_counts[tomato_set(s)] += 1
        endpoint_counts[s["route"][-1]] += 1
        category_counts[s["special_used"]] += 1

        return True

    # Ruajme te pakten nje kandidat nga
    # secila kategori e disponueshme
    for used in [2, 3, 4, 5]:

        options = sorted(
            [s for s in candidates
             if s["special_used"] == used],
            key=base_score
        )

        for s in options:
            if set_counts[tomato_set(s)] == 0:
                add(s)
                break

    # 3. Perzgjedhje me diversitet
    phases = [
        (1, 3, True),
        (1, 6, True),
        (1, 50, False),
        (2, 8, False),
        (50, 50, False)
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

                if use_caps and category_counts[used] >= CAPS.get(used, 50):
                    continue

                eligible.append(s)

            if not eligible:
                break

            def selection_score(s):
                endpoint = s["route"][-1]
                used = s["special_used"]

                return (
                    base_score(s)
                    + 6 * endpoint_counts[endpoint]
                    + 2 * category_counts[used]
                )

            best = min(eligible, key=selection_score)
            add(best)

    # 4. Validimi strukturor
    for s in selected:
        assert len(s["route"]) == TARGET
        assert len(set(s["route"])) == TARGET
        assert len(s["transfers"]) == TARGET - 1
        assert s["current_time"] <= 200

        specials = sum(
            t["status"] == "SPECIAL"
            for t in s["transfers"]
        )

        assert specials == s["special_used"]
        assert specials <= 5

    result = {
        "method": "Beam Search V8.4 - Cache Reselection",
        "source": SOURCE.name,
        "cache": CACHE.name,
        "beam_width": WIDTH,
        "available_candidates": len(candidates),
        "available_tomato_sets": available_sets,
        "beam": selected,
        "completed": False
    }

    save(result)


# 5. Perfundimi dhe krahasimi
print("\nPERMBLEDHJA E V8.4 - NIVELI 26")
print("=" * 55)

old_sets, old_ends = diversity(old_beam)
new_sets, new_ends = diversity(selected)

print("Domate: 26/49")
print("Rruge aktive:", len(selected))
print("Domate te mbetura: 23")

print("\nKRAHASIMI I DIVERSITETIT")
print("V8.3 bashkesi:", old_sets)
print("V8.4 bashkesi:", new_sets)
print("V8.3 domate fundore:", old_ends)
print("V8.4 domate fundore:", new_ends)

print("\nKOHA MINIMALE SIPAS SPECIALEVE")

counts = Counter(s["special_used"] for s in selected)

for used in sorted(counts):
    best = min(
        s["current_time"]
        for s in selected
        if s["special_used"] == used
    )

    print(
        f"{used} speciale: {counts[used]} rruge | "
        f"koha {best:.1f} dite | "
        f"mbeten {200-best:.1f} dite"
    )

old_keys = {signature(s) for s in old_beam}
new_keys = {signature(s) for s in selected}

print("\nRRUGET")
print("Te perbashketa:", len(old_keys & new_keys))
print("Te reja ne V8.4:", len(new_keys - old_keys))

print("\nCheckpoint:", OUTPUT.name)
print("V8.4 PERFUNDOI!")
