
import json
import math
from pathlib import Path
from collections import Counter

from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

POOL = RESULTS / "beam_v87_candidate_pool_level_26.json"
SELECTED = RESULTS / "beam_v84_diverse_level_26.json"
OLD_CACHE = RESULTS / "beam_v84_diverse_cache_26.json"

NEW_CACHE = RESULTS / "beam_v88_candidate_cache_26.json"
REPORT = RESULTS / "beam_v88_candidate_report_26.json"

WAITS = [0, 0.5, 1, 2, 3, 5, 8, 13]
TOFS = [0.5, 1, 1.5, 2, 3, 5, 8, 13, 21, 34]


def load(path):
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


def cache_key(state):
    return json.dumps([
        state["route"],
        state["current_time"],
        state["special_used"]
    ])


problem = TomatoProblem(
    str(ROOT / "data/easy.kttsp")
)

all_candidates = load(POOL)["beam"]
old_selected = load(SELECTED)["beam"]
previous_cache = load(OLD_CACHE)

selected_keys = {
    signature(s) for s in old_selected
}

# Te gjithe kandidatet me 4 ose 5 speciale
# qe nuk kane kaluar diten 145.
promising = [
    s for s in all_candidates
    if s["special_used"] in (4, 5)
    and s["current_time"] <= 145
]

promising.sort(
    key=lambda s: (
        s["current_time"],
        s["special_used"],
        s["route"][-1]
    )
)

if NEW_CACHE.exists():
    new_cache = load(NEW_CACHE)
else:
    new_cache = {}

print("HAPI 50 - V8.8 LOOKAHEAD")
print("=" * 60)
print("Kandidate premtues:", len(promising))

kept = sum(
    signature(s) in selected_keys
    for s in promising
)

print("Te ruajtur ne V8.4:", kept)
print("Te lene jashte:", len(promising) - kept)
print("Niveli aktual: 26/49")
print("Niveli qe kontrollojme: 27")
print("=" * 60, flush=True)

report = []
reused = 0
newly_scanned = 0

for index, state in enumerate(promising, 1):

    key = cache_key(state)
    route = state["route"]
    now = float(state["current_time"])
    used = int(state["special_used"])
    current = route[-1]

    was_selected = signature(state) in selected_keys

    if key in new_cache:
        moves = new_cache[key]
        source = "CACHE V8.8"
        reused += 1

    elif key in previous_cache:
        moves = previous_cache[key]
        source = "CACHE V8.4"
        reused += 1

    else:
        # Kandidat i lene jashte:
        # kontrollojme transferimet me PyKEP.
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

        new_cache[key] = moves

        # Ruajtje pas secilit kandidat,
        # edhe kur nuk gjendet asnje levizje.
        save(NEW_CACHE, new_cache)

        source = "PYKEP I RI"
        newly_scanned += 1

        print(
            f"  Llogaritje: {evaluations} | "
            f"Gabime te kapura: {errors}",
            flush=True
        )

    normal_moves = [
        m for m in moves
        if m["status"] == "NORMAL"
        and m["target"] not in route
        and m["arrival"] <= 200
        and m["dv"] < problem.dv_threshold
    ]

    special_moves = [
        m for m in moves
        if m["status"] == "SPECIAL"
        and used < problem.n_exceptions
        and m["target"] not in route
        and m["arrival"] <= 200
        and problem.dv_threshold
            <= m["dv"] < problem.exception_threshold
    ]

    normal_targets = len({
        m["target"] for m in normal_moves
    })

    special_targets = len({
        m["target"] for m in special_moves
    })

    best_normal = min(
        normal_moves,
        key=lambda m: (m["arrival"], m["dv"]),
        default=None
    )

    best_special = min(
        special_moves,
        key=lambda m: (m["arrival"], m["dv"]),
        default=None
    )

    row = {
        "route": route,
        "time": now,
        "special_used": used,
        "endpoint": current,
        "was_selected": was_selected,
        "normal_targets": normal_targets,
        "special_targets": special_targets,
        "best_normal": best_normal,
        "best_special": best_special
    }

    report.append(row)

    status = (
        "E RUAJTUR"
        if was_selected
        else "E LENE JASHTE"
    )

    print(
        f"[{index}/{len(promising)}] "
        f"Koha {now:.2f} | "
        f"Fund {current} | "
        f"Speciale {used} | "
        f"NORMAL {normal_targets} | "
        f"SPECIAL {special_targets} | "
        f"{status} | {source}",
        flush=True
    )


# PERMBLEDHJA
print("\nPERMBLEDHJA V8.8 - NIVELI 26 -> 27")
print("=" * 60)

print("Kandidate te testuar:", len(report))
print("Nga cache:", reused)
print("Te skanuar me PyKEP:", newly_scanned)

for label, flag in [
    ("TE RUAJTUR NE V8.4", True),
    ("TE LENE JASHTE", False)
]:
    group = [
        r for r in report
        if r["was_selected"] == flag
    ]

    with_normal = [
        r for r in group
        if r["normal_targets"] > 0
    ]

    with_any = [
        r for r in group
        if (
            r["normal_targets"] > 0
            or r["special_targets"] > 0
        )
    ]

    blocked = len(group) - len(with_any)

    print(f"\n{label}")
    print("Rruge:", len(group))
    print("Me transferim NORMAL:", len(with_normal))
    print("Me transferim te lejueshem:", len(with_any))
    print("Te bllokuara ne rrjetin aktual:", blocked)


# RENDITJA SIPAS ARRITJES NORMALE
normal_viable = [
    r for r in report
    if r["best_normal"] is not None
]

normal_viable.sort(
    key=lambda r: (
        r["best_normal"]["arrival"],
        r["special_used"],
        -r["normal_targets"]
    )
)

print("\n10 RRUGET ME ARRITJEN ME TE HERSHME NORMAL")
print("=" * 60)

for r in normal_viable[:10]:

    move = r["best_normal"]
    remaining_days = 200 - move["arrival"]
    avg = remaining_days / (49 - 27)

    status = (
        "E RUAJTUR"
        if r["was_selected"]
        else "E LENE JASHTE"
    )

    print(
        f"Fund {r['endpoint']:2d} | "
        f"Koha {r['time']:6.2f} -> "
        f"{move['arrival']:6.2f} | "
        f"Domatja {move['target']:2d} | "
        f"Normale {r['normal_targets']:2d} | "
        f"Speciale {r['special_used']} | "
        f"Mbeten {remaining_days:6.2f} dite | "
        f"{status}"
    )

save(REPORT, {
    "method": "V8.8 Candidate Lookahead",
    "source_level": 26,
    "target_level": 27,
    "tested_count": len(report),
    "reused_scans": reused,
    "new_scans": newly_scanned,
    "results": report
})

print("\nRaporti:", REPORT.name)
print("Cache i ri:", NEW_CACHE.name)
print("V8.8 PERFUNDOI!")
