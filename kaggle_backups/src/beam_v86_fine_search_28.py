
import json
import math
from pathlib import Path
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_v85_lookahead_summary_27.json"
OLD_CACHE = RESULTS / "beam_v85_lookahead_cache_27.json"

CACHE = RESULTS / "beam_v86_fine_cache_27.json"
SUMMARY = RESULTS / "beam_v86_fine_summary_27.json"

# Rrjeti i vjeter
OLD_WAITS = {0, 0.5, 1, 2, 3, 5, 8, 13}
OLD_TOFS = {0.5, 1, 1.5, 2, 3, 5, 8, 13, 21, 34}

# Rrjet me i imet per kerkim
WAITS = sorted(set(
    [i / 4 for i in range(0, 17)] +
    [i / 2 for i in range(9, 17)] +
    [10.0, 13.0]
))

TOFS = sorted(set(
    [i / 4 for i in range(1, 21)] +
    [i / 2 for i in range(11, 21)] +
    [13.0, 21.0, 34.0]
))

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8"
    )
    tmp.replace(path)

report = load(SOURCE)["results"]
old_cache = load(OLD_CACHE)

# Zgjedhim dy rruget me te shpejta
# qe kane perdorur 5 speciale
selected = sorted(
    [r for r in report if r["special_used"] == 5],
    key=lambda r: r["current_time"]
)[:2]

assert len(selected) == 2

if CACHE.exists():
    progress = load(CACHE)
    assert progress.get("version") == "V8.6-fine-grid-1"
else:
    progress = {
        "version": "V8.6-fine-grid-1",
        "routes": {}
    }

problem = TomatoProblem(
    str(ROOT / "data/easy.kttsp")
)

print("V8.6 - FINE SEARCH")
print("=" * 55)
print("Rruge qe testohen:", len(selected))
print("Kohe pritjeje:", len(WAITS))
print("Kohe fluturimi:", len(TOFS))
print("Domate te vizituara: 27/49")
print("Niveli i ardhshem: 28")
print("=" * 55, flush=True)

results = []

for row in selected:

    route = row["route"]
    now = float(row["current_time"])
    used = int(row["special_used"])
    current = route[-1]

    key = json.dumps([route, now, used])

    if key not in old_cache:
        raise RuntimeError(
            "Mungon cache-i i rruges: " +
            str(row["route_index"])
        )

    if key not in progress["routes"]:
        progress["routes"][key] = {
            "route_index": row["route_index"],
            "targets": {}
        }

    state = progress["routes"][key]
    completed = state["targets"]

    print(
        f"\nRRUGA {row['route_index']} | "
        f"Koha {now:.1f} | "
        f"Speciale {used}/5",
        flush=True
    )

    old_moves = old_cache[key]
    old_by_target = {}

    for move in old_moves:
        target = int(move["target"])

        if move["status"] == "NORMAL":
            old_by_target.setdefault(target, []).append(move)

    remaining_targets = [
        t for t in range(problem.n_tomatoes)
        if t not in route
    ]

    for target in remaining_targets:

        target_key = str(target)

        if target_key in completed:
            continue

        # Marrim transferimin me te mire
        # normal nga kerkimi i vjeter
        existing = old_by_target.get(target, [])

        best = min(
            existing,
            key=lambda m: (m["arrival"], m["dv"]),
            default=None
        )

        min_dv = min(
            (float(m["dv"]) for m in existing),
            default=None
        )

        evaluations = 0
        errors = 0

        for wait in WAITS:

            start = now + wait

            for tof in TOFS:

                # Kjo dyshe eshte testuar
                # ne Hapin 43
                if (
                    wait in OLD_WAITS
                    and tof in OLD_TOFS
                ):
                    continue

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

                if min_dv is None or dv < min_dv:
                    min_dv = dv

                # Te dy rruget i kane perdorur
                # te 5 specialet.
                # Pranojme vetem NORMAL.
                if dv >= problem.dv_threshold:
                    continue

                move = {
                    "target": target,
                    "wait": wait,
                    "tof": tof,
                    "arrival": arrival,
                    "dv": dv,
                    "status": "NORMAL"
                }

                if best is None or (
                    move["arrival"], move["dv"]
                ) < (
                    best["arrival"], best["dv"]
                ):
                    best = move

        completed[target_key] = {
            "best_normal": best,
            "min_dv": min_dv,
            "new_evaluations": evaluations,
            "errors": errors
        }

        # Ruajtja pas cdo domateje
        save(CACHE, progress)

        if best is not None:
            print(
                f"Domatja {target}: NORMAL | "
                f"Arritja {best['arrival']:.2f} | "
                f"DeltaV {best['dv']:.2f}",
                flush=True
            )

    normal_options = []

    for target in remaining_targets:
        item = completed[str(target)]
        move = item["best_normal"]

        if move is not None:
            normal_options.append(move)

    normal_options.sort(
        key=lambda m: (m["arrival"], m["dv"])
    )

    best_next = (
        normal_options[0]
        if normal_options
        else None
    )

    result = {
        "route_index": row["route_index"],
        "route": route,
        "current_time": now,
        "special_used": used,
        "normal_target_count": len(normal_options),
        "best_next": best_next,
        "normal_options": normal_options
    }

    results.append(result)

    print(
        f"PERFUNDIMI I RRUGES {row['route_index']}: "
        f"{len(normal_options)} domate NORMAL",
        flush=True
    )

save(SUMMARY, {
    "method": "V8.6 Fine Time Grid",
    "level": 27,
    "next_level": 28,
    "results": results
})

print("\nPERMBLEDHJA V8.6 - KERKIMI I DETAJUAR")
print("=" * 55)

for result in results:

    print(
        f"\nRruga {result['route_index']} | "
        f"Koha {result['current_time']:.1f} dite"
    )

    print(
        "Domate NORMAL te arritshme:",
        result["normal_target_count"]
    )

    move = result["best_next"]

    if move is None:
        print("Nuk u gjet transferim NORMAL.")
        continue

    after = 49 - 28
    remaining_days = 200 - move["arrival"]

    print("Domatja e ardhshme:", move["target"])
    print("Wait:", move["wait"])
    print("TOF:", move["tof"])
    print("DeltaV:", round(move["dv"], 3))
    print("Arritja:", round(move["arrival"], 3))
    print("Dite te mbetura:", round(remaining_days, 3))
    print(
        "Mesatarja per 21 domatet tjera:",
        round(remaining_days / after, 3)
    )

print("\nCache:", CACHE.name)
print("Raporti:", SUMMARY.name)
print("V8.6 PERFUNDOI!")
