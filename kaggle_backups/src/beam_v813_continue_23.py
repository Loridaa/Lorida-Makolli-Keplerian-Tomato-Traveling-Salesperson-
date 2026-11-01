
import json
import math
from pathlib import Path
from collections import Counter
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_v813_wide_level_22.json"
OLD_CACHE = RESULTS / "beam_v8_diverse_cache_22.json"

CACHE = RESULTS / "beam_v813_wide_cache_22.json"
OUTPUT = RESULTS / "beam_v813_wide_level_23.json"

TARGET = 23
WIDTH = 100

WAITS = [0, 0.5, 1, 2, 3, 5, 8, 13]
TOFS = [0.5, 1, 1.5, 2, 3, 5, 8, 13, 21, 34]

def read(path, default=None):
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))

def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8"
    )
    temporary.replace(path)

def cache_key(s):
    return json.dumps([
        s["route"],
        s["current_time"],
        s["special_used"]
    ])

def signature(s):
    return (
        tuple(s["route"]),
        round(float(s["current_time"]), 6),
        int(s["special_used"])
    )

problem = TomatoProblem(
    str(ROOT / "data/easy.kttsp")
)

parents = read(SOURCE)["beam"]

assert len(parents) == 100

old_cache = read(OLD_CACHE)
cache = read(CACHE)

print("HAPI 60 - V8.13 WIDE CONTINUATION")
print("=" * 60)
print("Niveli: 22 -> 23")
print("Prinder:", len(parents))
print("Beam Width:", WIDTH)
print("=" * 60, flush=True)

# 1. SKANIMI I PRINDERVE
new_scans = 0
reused = 0

if not OUTPUT.exists():

    for i, parent in enumerate(parents, 1):
        key = cache_key(parent)

        if key in cache:
            source = "CACHE V8.13"

        elif key in old_cache:
            cache[key] = old_cache[key]
            save(CACHE, cache)
            reused += 1
            source = "CACHE V8"

        else:
            route = parent["route"]
            current = route[-1]
            now = float(parent["current_time"])
            used = int(parent["special_used"])

            moves = []
            evaluations = 0
            errors = 0

            for target in range(problem.n_tomatoes):

                if target in route:
                    continue

                for wait in WAITS:

                    start = now + wait

                    for tof in TOFS:

                        arrival = start + tof

                        if arrival > problem.max_time:
                            continue

                        try:
                            dv = float(
                                problem.compute_transfer(
                                    current,
                                    target,
                                    start,
                                    tof
                                )
                            )
                            evaluations += 1

                        except Exception:
                            errors += 1
                            continue

                        if not math.isfinite(dv) or dv < 0:
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

                        moves.append({
                            "target": target,
                            "wait": wait,
                            "tof": tof,
                            "arrival": arrival,
                            "dv": dv,
                            "status": status
                        })

            cache[key] = moves
            save(CACHE, cache)

            new_scans += 1
            source = "PYKEP"

            print(
                f"  PyKEP: {evaluations} llogaritje | "
                f"gabime numerike: {errors}",
                flush=True
            )

        moves = cache[key]

        normal_targets = len({
            m["target"] for m in moves
            if m["status"] == "NORMAL"
        })

        special_targets = len({
            m["target"] for m in moves
            if m["status"] == "SPECIAL"
        })

        print(
            f"Prindi {i}/100 | "
            f"NORMAL {normal_targets} | "
            f"SPECIAL {special_targets} | "
            f"{source}",
            flush=True
        )

    # 2. RIKRIJIMI I KANDIDATEVE
    pool = {}

    for parent in parents:
        route = parent["route"]
        now = float(parent["current_time"])
        current = route[-1]

        for m in cache[cache_key(parent)]:

            target = int(m["target"])
            arrival = float(m["arrival"])
            wait = float(m["wait"])
            tof = float(m["tof"])
            dv = float(m["dv"])
            status = m["status"]

            if target in route or arrival > 200:
                continue

            if not math.isfinite(dv) or dv < 0:
                continue

            if wait < 0 or tof <= 0:
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
                "route": route + [target],
                "transfers": parent["transfers"] + [transfer],
                "current_time": arrival,
                "special_used": used
            }

            sig = signature(child)
            previous = pool.get(sig)

            if previous is None:
                pool[sig] = child
            else:
                new_dv = sum(
                    t["dv"] for t in child["transfers"]
                )
                old_dv = sum(
                    t["dv"] for t in previous["transfers"]
                )

                if new_dv < old_dv:
                    pool[sig] = child

    candidates = list(pool.values())

    print("\nKANDIDATET E NIVELIT 23")
    print("=" * 60)
    print("Kandidate unike:", len(candidates))
    print(
        "Bashkesi te disponueshme:",
        len({
            frozenset(s["route"])
            for s in candidates
        })
    )

    if not candidates:
        raise RuntimeError(
            "Nuk u gjet asnje kandidat i vlefshem."
        )

    # 3. PERZGJEDHJA ME DIVERSITET,
    # KOHE DHE SPECIALE REZERVE
    selected = []
    seen = set()

    set_counts = Counter()
    endpoint_counts = Counter()
    category_counts = Counter()

    CAPS = {0: 5, 1: 5, 2: 8, 3: 30, 4: 50, 5: 15}

    def tomato_set(s):
        return frozenset(s["route"])

    def add(s):
        sig = signature(s)

        if sig in seen:
            return False

        selected.append(s)
        seen.add(sig)

        set_counts[tomato_set(s)] += 1
        endpoint_counts[s["route"][-1]] += 1
        category_counts[s["special_used"]] += 1
        return True

    def score(s):
        now = float(s["current_time"])
        used = int(s["special_used"])

        remaining = 49 - TARGET
        avg = (200 - now) / remaining

        time_penalty = 20 * max(0, 3.0 - avg) ** 2
        exhaustion_penalty = 12 if used == 5 else 0

        return (
            now
            + 12 * used
            + time_penalty
            + exhaustion_penalty
            + 6 * endpoint_counts[s["route"][-1]]
            + 2 * category_counts[used]
        )

    # Ruajme nje kandidat te shpejte
    # nga secila kategori e disponueshme.
    for used in sorted({
        s["special_used"] for s in candidates
    }):
        group = [
            s for s in candidates
            if s["special_used"] == used
        ]

        if group:
            add(min(group, key=lambda s: s["current_time"]))

    phases = [
        (1, 5, True),
        (1, 100, True),
        (1, 100, False),
        (2, 100, False),
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

                used = int(s["special_used"])

                if (
                    use_caps
                    and category_counts[used] >= CAPS[used]
                ):
                    continue

                eligible.append(s)

            if not eligible:
                break

            add(min(eligible, key=score))

    # 4. VALIDIMI
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

    save(OUTPUT, {
        "method": "V8.13 Wide Beam Continuation",
        "source": SOURCE.name,
        "cache": CACHE.name,
        "beam_width": WIDTH,
        "candidate_count": len(candidates),
        "beam": selected,
        "completed": False
    })

else:
    print("Checkpoint i nivelit 23 ekziston.")

# 5. RAPORTI
beam = read(OUTPUT)["beam"]

print("\nPERMBLEDHJA V8.13 - NIVELI 23")
print("=" * 60)
print("Domate: 23/49")
print("Rruge aktive:", len(beam))

print(
    "Bashkesi te ndryshme:",
    len({frozenset(s["route"]) for s in beam})
)

print(
    "Domate fundore te ndryshme:",
    len({s["route"][-1] for s in beam})
)

print("\nKOHA MINIMALE SIPAS SPECIALEVE")

counts = Counter(s["special_used"] for s in beam)

for used in sorted(counts):

    best = min(
        s["current_time"]
        for s in beam
        if s["special_used"] == used
    )

    remaining = 200 - best

    print(
        f"{used} speciale: {counts[used]} rruge | "
        f"koha {best:.2f} | "
        f"mbeten {remaining:.2f} dite | "
        f"{remaining/26:.2f} dite/domate"
    )

print("\nTRANSFERIMI I FUNDIT")

last = Counter(
    s["transfers"][-1]["status"]
    for s in beam
)

print("NORMAL:", last["NORMAL"])
print("SPECIAL:", last["SPECIAL"])

print("\nCache:", CACHE.name)
print("Checkpoint:", OUTPUT.name)
print("V8.13 NIVELI 23 PERFUNDOI!")
