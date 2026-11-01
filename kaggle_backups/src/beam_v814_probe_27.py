
import json
import math
from pathlib import Path
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_v813_wide_level_26.json"
CACHE = RESULTS / "beam_v814_probe_cache_26.json"
REPORT = RESULTS / "beam_v814_probe_report_26.json"

OLD_CACHES = [
    RESULTS / "beam_v84_diverse_cache_26.json",
    RESULTS / "beam_v88_candidate_cache_26.json"
]

WAITS = [0, 0.5, 1, 2, 3, 5, 8, 13]
TOFS = [0.5, 1, 1.5, 2, 3, 5, 8, 13, 21, 34]

def load(path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))

def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8"
    )
    tmp.replace(path)

def key(s):
    return json.dumps([
        s["route"],
        s["current_time"],
        s["special_used"]
    ])

problem = TomatoProblem(
    str(ROOT / "data/easy.kttsp")
)

beam = load(SOURCE)["beam"]

cache = load(CACHE, {})
old_caches = [
    load(path, {})
    for path in OLD_CACHES
]

# Rruget me nje speciale rezerve
four = sorted(
    [
        s for s in beam
        if s["special_used"] == 4
        and s["current_time"] <= 145
    ],
    key=lambda s: s["current_time"]
)[:24]

# Disa rruge te shpejta pa speciale rezerve
five = sorted(
    [
        s for s in beam
        if s["special_used"] == 5
        and s["current_time"] <= 140
    ],
    key=lambda s: s["current_time"]
)[:10]

selected = four + five

print("HAPI 65 - V8.14 TARGETED PROBE")
print("=" * 60)
print("Burimi: V8.13 niveli 26")
print("Rruge aktive ne burim:", len(beam))
print("Rruge me 4 speciale qe testohen:", len(four))
print("Rruge me 5 speciale qe testohen:", len(five))
print("Gjithsej per testim:", len(selected))
print("Synimi: 26 -> 27")
print("=" * 60, flush=True)

if not selected:
    raise RuntimeError(
        "Nuk ka rruge brenda kufijve kohore."
    )

results = []
new_scans = 0
reused = 0

for i, state in enumerate(selected, 1):

    route = state["route"]
    now = float(state["current_time"])
    used = int(state["special_used"])
    current = route[-1]
    state_key = key(state)

    source = None

    if state_key in cache:
        moves = cache[state_key]
        source = "CACHE V8.14"
        reused += 1

    else:
        for old in old_caches:
            if state_key in old:
                moves = old[state_key]
                cache[state_key] = moves
                save(CACHE, cache)
                source = "CACHE I VJETER"
                reused += 1
                break

    if source is None:

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

        cache[state_key] = moves
        save(CACHE, cache)

        new_scans += 1
        source = "PYKEP"

        print(
            f"  PyKEP: {evaluations} llogaritje | "
            f"{errors} gabime numerike",
            flush=True
        )

    normals = [
        m for m in moves
        if (
            m["status"] == "NORMAL"
            and int(m["target"]) not in route
            and float(m["arrival"]) <= 200
            and 0 <= float(m["dv"])
                < problem.dv_threshold
        )
    ]

    specials = [
        m for m in moves
        if (
            m["status"] == "SPECIAL"
            and used < problem.n_exceptions
            and int(m["target"]) not in route
            and float(m["arrival"]) <= 200
            and problem.dv_threshold
                <= float(m["dv"])
                < problem.exception_threshold
        )
    ]

    best_normal = min(
        normals,
        key=lambda m: (
            float(m["arrival"]),
            float(m["dv"])
        ),
        default=None
    )

    best_special = min(
        specials,
        key=lambda m: (
            float(m["arrival"]),
            float(m["dv"])
        ),
        default=None
    )

    row = {
        "route": route,
        "time": now,
        "endpoint": current,
        "special_used": used,
        "normal_targets": len({
            int(m["target"]) for m in normals
        }),
        "special_targets": len({
            int(m["target"]) for m in specials
        }),
        "best_normal": best_normal,
        "best_special": best_special
    }

    results.append(row)

    print(
        f"[{i}/{len(selected)}] "
        f"Koha {now:.2f} | "
        f"Fund {current} | "
        f"Speciale {used} | "
        f"NORMAL {row['normal_targets']} | "
        f"SPECIAL {row['special_targets']} | "
        f"{source}",
        flush=True
    )

save(REPORT, {
    "method": "V8.14 Targeted Level 27 Probe",
    "source_level": 26,
    "target_level": 27,
    "new_scans": new_scans,
    "reused_scans": reused,
    "results": results
})

print("\nPERMBLEDHJA V8.14 - NIVELI 26 -> 27")
print("=" * 60)
print("Rruge te testuara:", len(results))
print("Nga cache:", reused)
print("Skanime te reja PyKEP:", new_scans)

for used in [4, 5]:

    group = [
        r for r in results
        if r["special_used"] == used
    ]

    normal = sum(
        r["normal_targets"] > 0
        for r in group
    )

    possible = sum(
        r["normal_targets"] > 0
        or r["special_targets"] > 0
        for r in group
    )

    print(f"\n{used} SPECIALE")
    print("Rruge:", len(group))
    print("Me NORMAL:", normal)
    print("Me transferim te lejueshem:", possible)
    print("Te bllokuara:", len(group) - possible)

print("\nRRUGET ME TE MIRA ME TRANSFERIM NORMAL")
print("=" * 60)

viable = sorted(
    [
        r for r in results
        if r["best_normal"] is not None
    ],
    key=lambda r: (
        float(r["best_normal"]["arrival"]),
        r["special_used"]
    )
)

if not viable:
    print("Nuk u gjet transferim NORMAL.")

for r in viable[:12]:

    move = r["best_normal"]
    arrival = float(move["arrival"])
    left = 200 - arrival

    print(
        f"Fund {r['endpoint']} -> "
        f"Domatja {move['target']} | "
        f"Koha {r['time']:.2f} -> "
        f"{arrival:.2f} | "
        f"Speciale {r['special_used']} | "
        f"Mbeten {left:.2f} dite | "
        f"{left / 22:.2f} dite/domate"
    )

print("\nRaporti:", REPORT.name)
print("Cache:", CACHE.name)
print("V8.14 PERFUNDOI!")
