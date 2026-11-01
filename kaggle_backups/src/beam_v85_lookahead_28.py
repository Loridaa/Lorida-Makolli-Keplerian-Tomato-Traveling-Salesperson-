
import json
import math
from pathlib import Path

from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_v84_diverse_level_27.json"
CACHE = RESULTS / "beam_v85_lookahead_cache_27.json"
SUMMARY = RESULTS / "beam_v85_lookahead_summary_27.json"

WAITS = [0, 0.5, 1, 2, 3, 5, 8, 13]
TOFS = [0.5, 1, 1.5, 2, 3, 5, 8, 13, 21, 34]

print("V8.5 - TARGETED LOOKAHEAD")
print("=" * 55, flush=True)

problem = TomatoProblem(
    str(ROOT / "data" / "easy.kttsp")
)

beam = json.loads(
    SOURCE.read_text(encoding="utf-8")
)["beam"]

# 6 me te shpejtat me 5 speciale
five = sorted(
    [s for s in beam if s["special_used"] == 5],
    key=lambda s: s["current_time"]
)[:6]

# 4 me te shpejtat me 4 speciale
four = sorted(
    [s for s in beam if s["special_used"] == 4],
    key=lambda s: s["current_time"]
)[:4]

selected = five + four

assert len(selected) == 10

if CACHE.exists():
    cache = json.loads(CACHE.read_text())
else:
    cache = {}


def save_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8"
    )
    temporary.replace(path)


report = []

print("Rruge qe testohen:", len(selected))
print("Domate te vizituara: 27/49")
print("Domate te mbetura: 22")
print("Niveli qe kontrollojme: 28")
print("=" * 55, flush=True)

for index, state in enumerate(selected, 1):

    route = state["route"]
    current = route[-1]

    now = float(state["current_time"])
    used = int(state["special_used"])

    key = json.dumps([route, now, used])

    print(
        f"\nRruga {index}/10 | "
        f"koha {now:.1f} | "
        f"speciale {used}",
        flush=True
    )

    if key in cache:
        moves = cache[key]
        print("U perdor cache ekzistues.", flush=True)

    else:
        moves = []
        errors = 0
        evaluations = 0

        for target in range(problem.n_tomatoes):

            if target in route:
                continue

            for wait in WAITS:

                start = now + wait

                for tof in TOFS:

                    arrival = start + tof

                    if arrival > 200:
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

        # Ruajme menjehere pas seciles rruge
        cache[key] = moves
        save_json(CACHE, cache)

        print("Transferime te llogaritura:", evaluations)
        print("Gabime numerike te kapura:", errors)

    normal_targets = sorted({
        m["target"]
        for m in moves
        if m["status"] == "NORMAL"
    })

    special_targets = sorted({
        m["target"]
        for m in moves
        if m["status"] == "SPECIAL"
    })

    earliest = min(
        moves,
        key=lambda m: m["arrival"],
        default=None
    )

    record = {
        "route_index": index,
        "route": route,
        "current_time": now,
        "special_used": used,
        "normal_targets": normal_targets,
        "special_targets": special_targets,
        "normal_count": len(normal_targets),
        "special_count": len(special_targets),
        "earliest_move": earliest
    }

    report.append(record)

    print("Domate NORMAL te arritshme:", len(normal_targets))
    print("Domate SPECIAL te arritshme:", len(special_targets))

    if earliest:
        print(
            "Arritja me e hershme:",
            f"{earliest['arrival']:.1f} dite"
        )
        print(
            "Domatja e ardhshme:",
            earliest["target"]
        )
        print(
            "Lloji:",
            earliest["status"]
        )
    else:
        print("Nuk u gjet transferim me kete rrjet kohor.")

save_json(SUMMARY, {
    "method": "V8.5 Targeted Lookahead",
    "level": 27,
    "next_level": 28,
    "tested_routes": len(selected),
    "results": report
})

print("\nPERMBLEDHJA V8.5 - LOOKAHEAD NIVELI 28")
print("=" * 55)

with_normal = sum(
    r["normal_count"] > 0
    for r in report
)

with_any = sum(
    r["normal_count"] + r["special_count"] > 0
    for r in report
)

print("Rruge te testuara:", len(report))
print("Me te pakten nje transferim NORMAL:", with_normal)
print("Me te pakten nje transferim te lejueshem:", with_any)
print("Pa transferime te lejueshme:", len(report) - with_any)

print("\nRRUGET ME 5 SPECIALE")
for r in report:
    if r["special_used"] == 5:
        print(
            f"Rruga {r['route_index']}: "
            f"koha {r['current_time']:.1f} | "
            f"normale {r['normal_count']}"
        )

print("\nRRUGET ME 4 SPECIALE")
for r in report:
    if r["special_used"] == 4:
        print(
            f"Rruga {r['route_index']}: "
            f"koha {r['current_time']:.1f} | "
            f"normale {r['normal_count']} | "
            f"speciale {r['special_count']}"
        )

print("\nCache:", CACHE.name)
print("Raporti:", SUMMARY.name)
print("V8.5 LOOKAHEAD PERFUNDOI!")
