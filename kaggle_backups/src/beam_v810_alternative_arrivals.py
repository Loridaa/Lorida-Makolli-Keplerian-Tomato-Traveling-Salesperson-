
import json
import math
from pathlib import Path
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

REPORT26 = RESULTS / "beam_v88_candidate_report_26.json"
CACHE26_A = RESULTS / "beam_v88_candidate_cache_26.json"
CACHE26_B = RESULTS / "beam_v84_diverse_cache_26.json"

CACHE27_OLD = RESULTS / "beam_v89_two_step_cache_27.json"
CACHE27_V85 = RESULTS / "beam_v85_lookahead_cache_27.json"

CACHE = RESULTS / "beam_v810_alternative_cache_27.json"
OUTPUT = RESULTS / "beam_v810_alternative_report_27_28.json"

WAITS = [0, 0.5, 1, 2, 3, 5, 8, 13]
TOFS = [0.5, 1, 1.5, 2, 3, 5, 8, 13, 21, 34]

MAX_PER_PARENT = 8
MAX_ARRIVAL = 155.0

def read(path, default=None):
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))

def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8"
    )
    tmp.replace(path)

def make_key(route, time, used):
    return json.dumps([
        route, float(time), int(used)
    ])

problem = TomatoProblem(
    str(ROOT / "data/easy.kttsp")
)

report = read(REPORT26)["results"]
cache26a = read(CACHE26_A)
cache26b = read(CACHE26_B)

cache27old = read(CACHE27_OLD, {})
cache27v85 = read(CACHE27_V85, {})
cache27new = read(CACHE, {})

# 1. Zgjedhim prinderit me arritje
# NORMALE deri ne diten 145.
parents = sorted(
    [
        r for r in report
        if r["best_normal"] is not None
        and float(r["best_normal"]["arrival"]) <= 145
    ],
    key=lambda r: float(r["best_normal"]["arrival"])
)

print("HAPI 53 - V8.10 ALTERNATIVE ARRIVALS")
print("=" * 60)
print("Kandidate fillestare:", len(report))
print("Prinder me arritje deri ne diten 145:", len(parents))
print("Maksimum variante per prind:", MAX_PER_PARENT)
print("Synimi: 26 -> 27 -> 28")
print("=" * 60, flush=True)

choices = []

for parent_index, p in enumerate(parents, 1):
    route = p["route"]
    now = float(p["time"])
    used = int(p["special_used"])

    key26 = make_key(route, now, used)

    if key26 in cache26a:
        moves = cache26a[key26]
    elif key26 in cache26b:
        moves = cache26b[key26]
    else:
        raise RuntimeError(
            "Mungon cache-i i nje prindi."
        )

    # Mbajme transferimin me DeltaV me
    # te vogel per secilen domate + kohe.
    variants = {}

    for m in moves:
        target = int(m["target"])
        arrival = float(m["arrival"])
        dv = float(m["dv"])

        if target in route:
            continue
        if m["status"] != "NORMAL":
            continue
        if not (0 <= dv < problem.dv_threshold):
            continue
        if arrival > MAX_ARRIVAL or arrival > 200:
            continue

        sig = (target, round(arrival, 6))
        old = variants.get(sig)

        if old is None or dv < float(old["dv"]):
            variants[sig] = m

    ordered = sorted(
        variants.values(),
        key=lambda m: (
            float(m["arrival"]),
            float(m["dv"])
        )
    )

    if not ordered:
        continue

    earliest = float(ordered[0]["arrival"])
    picked = [ordered[0]]

    # Provojme mbërritje me vone,
    # jo vetem mbërritjen me te hershme.
    for offset in [1, 2, 3, 4, 6, 8, 10, 12, 16, 20]:
        if len(picked) >= MAX_PER_PARENT:
            break

        eligible = [
            m for m in ordered
            if float(m["arrival"]) >= earliest + offset
            and all(
                abs(float(m["arrival"]) - float(x["arrival"]))
                >= 1.0
                or int(m["target"]) != int(x["target"])
                for x in picked
            )
        ]

        if eligible:
            chosen = min(
                eligible,
                key=lambda m: (
                    float(m["arrival"]),
                    float(m["dv"])
                )
            )
            picked.append(chosen)

    for option_index, move in enumerate(picked, 1):
        choices.append({
            "parent_index": parent_index,
            "parent": p,
            "move": move,
            "is_earliest": option_index == 1
        })

    print(
        f"Prindi {parent_index}: "
        f"koha {now:.2f} | "
        f"normalet e ndryshme {len(ordered)} | "
        f"variante te zgjedhura {len(picked)}",
        flush=True
    )

print("\nGjithsej variante:", len(choices), flush=True)

# 2. Kontrollojme hapin e dyte
results = []
new_scans = 0
reused = 0

for i, entry in enumerate(choices, 1):
    parent = entry["parent"]
    move = entry["move"]

    route27 = parent["route"] + [int(move["target"])]
    now27 = float(move["arrival"])
    used = int(parent["special_used"])

    assert len(route27) == 27
    assert len(set(route27)) == 27

    current = route27[-1]
    state_key = make_key(route27, now27, used)

    if state_key in cache27new:
        moves28 = cache27new[state_key]
        source = "CACHE V8.10"
        reused += 1

    elif state_key in cache27old:
        moves28 = cache27old[state_key]
        source = "CACHE V8.9"
        reused += 1

    elif state_key in cache27v85:
        moves28 = cache27v85[state_key]
        source = "CACHE V8.5"
        reused += 1

    else:
        moves28 = []
        evaluations = 0

        for target in range(problem.n_tomatoes):
            if target in route27:
                continue

            for wait in WAITS:
                start = now27 + wait

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

                    moves28.append({
                        "target": target,
                        "wait": wait,
                        "tof": tof,
                        "arrival": arrival,
                        "dv": dv,
                        "status": status
                    })

        cache27new[state_key] = moves28
        save(CACHE, cache27new)

        source = "PYKEP"
        new_scans += 1

    normals = [
        m for m in moves28
        if m["status"] == "NORMAL"
        and int(m["target"]) not in route27
        and float(m["arrival"]) <= 200
        and 0 <= float(m["dv"]) < problem.dv_threshold
    ]

    specials = [
        m for m in moves28
        if m["status"] == "SPECIAL"
        and used < problem.n_exceptions
        and int(m["target"]) not in route27
        and float(m["arrival"]) <= 200
        and problem.dv_threshold <= float(m["dv"])
        < problem.exception_threshold
    ]

    normal_targets = len({
        int(m["target"]) for m in normals
    })

    special_targets = len({
        int(m["target"]) for m in specials
    })

    best_normal = min(
        normals,
        key=lambda m: (
            float(m["arrival"]),
            float(m["dv"])
        ),
        default=None
    )

    row = {
        "parent_index": entry["parent_index"],
        "was_selected": parent["was_selected"],
        "first_target": current,
        "time27": now27,
        "special_used": used,
        "is_earliest": entry["is_earliest"],
        "normal_targets_28": normal_targets,
        "special_targets_28": special_targets,
        "best_normal_28": best_normal
    }

    results.append(row)

    print(
        f"[{i}/{len(choices)}] "
        f"Arritja 27: {now27:.2f} | "
        f"NORMAL per 28: {normal_targets} | "
        f"SPECIAL: {special_targets} | {source}",
        flush=True
    )

# 3. Raporti perfundimtar
print("\nPERMBLEDHJA V8.10 - KOHE ALTERNATIVE")
print("=" * 60)

normal_viable = [
    r for r in results
    if r["normal_targets_28"] > 0
]

any_viable = [
    r for r in results
    if r["normal_targets_28"] > 0
    or r["special_targets_28"] > 0
]

print("Variante te testuara:", len(results))
print("Skanime te reja PyKEP:", new_scans)
print("Skanime nga cache:", reused)
print("Me NORMAL per domaten 28:", len(normal_viable))
print("Me transferim te lejueshem:", len(any_viable))
print("Te bllokuara:", len(results) - len(any_viable))

print("\nKRAHASIMI I KOHEVE")

earliest = [r for r in results if r["is_earliest"]]
alternatives = [r for r in results if not r["is_earliest"]]

for label, group in [
    ("Mberritja me e hershme", earliest),
    ("Mberritje alternative", alternatives)
]:
    count = sum(
        r["normal_targets_28"] > 0 for r in group
    )
    print(
        f"{label}: {len(group)} variante, "
        f"{count} me NORMAL per 28"
    )

print("\nRRUGET QE MUND TE VAZHDOJNE")

ranked = sorted(
    normal_viable,
    key=lambda r: float(r["best_normal_28"]["arrival"])
)

if not ranked:
    print("Nuk u gjet transferim NORMAL ne kete rrjet.")

for r in ranked[:10]:
    move = r["best_normal_28"]

    print(
        f"Prindi {r['parent_index']} | "
        f"27: domatja {r['first_target']} "
        f"ne diten {r['time27']:.2f} | "
        f"28: domatja {move['target']} "
        f"ne diten {move['arrival']:.2f} | "
        f"Speciale {r['special_used']} | "
        f"Mbeten {200-float(move['arrival']):.2f} dite"
    )

save(OUTPUT, {
    "method": "V8.10 Alternative Arrival Lookahead",
    "source_level": 26,
    "target_level": 28,
    "variants_tested": len(results),
    "normal_continuations": len(normal_viable),
    "results": results
})

print("\nRaporti:", OUTPUT.name)
print("Cache:", CACHE.name)
print("HAPI 53 PERFUNDOI!")
