
import json
import math
from pathlib import Path
from collections import Counter

from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

LEVEL26 = RESULTS / "beam_v813_wide_level_26.json"
REPORT26 = RESULTS / "beam_v814_probe_report_26.json"
CACHE26 = RESULTS / "beam_v814_probe_cache_26.json"

CACHE27 = RESULTS / "beam_v815_two_step_cache_27.json"
REPORT = RESULTS / "beam_v815_two_step_report_27_28.json"

OLD_CACHE_NAMES = [
    "beam_v89_two_step_cache_27.json",
    "beam_v85_lookahead_cache_27.json",
    "beam_v810_alternative_cache_27.json"
]

WAITS = [0, 0.5, 1, 2, 3, 5, 8, 13]
TOFS = [0.5, 1, 1.5, 2, 3, 5, 8, 13, 21, 34]

MAX_CHILDREN = 24

def load(path, default=None):
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)

    return json.loads(path.read_text(encoding="utf-8"))

def save(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8"
    )
    temp.replace(path)

def state_key(route, time, used):
    return json.dumps([
        route, float(time), int(used)
    ])

problem = TomatoProblem(
    str(ROOT / "data/easy.kttsp")
)

beam = load(LEVEL26)["beam"]
rows = load(REPORT26)["results"]

first_cache = load(CACHE26)
new_cache = load(CACHE27, {})

old_caches = [
    load(RESULTS / name, {})
    for name in OLD_CACHE_NAMES
]

# Rruget e plota te nivelit 26
parents = {
    state_key(
        s["route"],
        s["current_time"],
        s["special_used"]
    ): s
    for s in beam
}

# 1. Zgjedhim variante te nivelit 27
choices_by_parent = []

for row in sorted(
    rows,
    key=lambda r: (
        r["special_used"],
        r["time"]
    )
):

    key = state_key(
        row["route"],
        row["time"],
        row["special_used"]
    )

    if key not in parents or key not in first_cache:
        raise RuntimeError(
            "Mungon prindi ose cache-i i V8.14."
        )

    parent = parents[key]
    now = float(parent["current_time"])
    used = int(parent["special_used"])

    valid = []

    for m in first_cache[key]:

        target = int(m["target"])
        arrival = float(m["arrival"])
        dv = float(m["dv"])
        status = m["status"]

        if target in parent["route"]:
            continue

        if arrival > 200:
            continue

        if status == "NORMAL" and 0 <= dv < 100:
            pass
        elif (
            status == "SPECIAL"
            and used < 5
            and 100 <= dv < 600
        ):
            pass
        else:
            continue

        if abs(
            now + float(m["wait"])
            + float(m["tof"]) - arrival
        ) > 1e-6:
            continue

        valid.append(m)

    # Transferimi me i hershem per çdo
    # domate dhe lloj transferimi.
    best_by_target = {}

    for m in valid:
        identity = (
            int(m["target"]),
            m["status"]
        )

        previous = best_by_target.get(identity)

        if previous is None or (
            m["arrival"], m["dv"]
        ) < (
            previous["arrival"],
            previous["dv"]
        ):
            best_by_target[identity] = m

    normals = sorted(
        [
            m for m in best_by_target.values()
            if m["status"] == "NORMAL"
        ],
        key=lambda m: (m["arrival"], m["dv"])
    )

    specials = sorted(
        [
            m for m in best_by_target.values()
            if m["status"] == "SPECIAL"
        ],
        key=lambda m: (m["arrival"], m["dv"])
    )

    options = []

    if normals:
        options.append(normals[0])

    if specials:
        options.append(specials[0])

    if len(normals) > 1:
        options.append(normals[1])

    if len(specials) > 1:
        options.append(specials[1])

    choices_by_parent.append((parent, options))

# Perzgjedhje me radhe, qe te mos
# testojme vetem nje prind.
choices = []
seen_children = set()

for round_number in range(4):

    for parent, options in choices_by_parent:

        if len(choices) >= MAX_CHILDREN:
            break

        if round_number >= len(options):
            continue

        move = options[round_number]

        route27 = (
            parent["route"] +
            [int(move["target"])]
        )

        time27 = float(move["arrival"])

        used27 = (
            int(parent["special_used"])
            + int(move["status"] == "SPECIAL")
        )

        key = state_key(route27, time27, used27)

        if key in seen_children:
            continue

        seen_children.add(key)

        choices.append({
            "parent": parent,
            "first_move": move,
            "route27": route27,
            "time27": time27,
            "used27": used27,
            "key": key
        })

    if len(choices) >= MAX_CHILDREN:
        break

print("HAPI 67 - V8.15 TWO-STEP SEARCH")
print("=" * 60)
print("Niveli: 26 -> 27 -> 28")
print("Prinder te kontrolluar:", len(rows))
print("Variante te nivelit 27:", len(choices))
print("Maksimumi:", MAX_CHILDREN)
print("=" * 60, flush=True)

results = []
new_scans = 0
reused = 0

# 2. Kontrollojme transferimin e dyte
for index, entry in enumerate(choices, 1):

    route27 = entry["route27"]
    now = entry["time27"]
    used = entry["used27"]
    current = route27[-1]
    key = entry["key"]

    assert len(route27) == 27
    assert len(set(route27)) == 27
    assert used <= 5

    if key in new_cache:
        moves = new_cache[key]
        source = "CACHE V8.15"
        reused += 1

    else:
        moves = None
        source = None

        for old in old_caches:
            if key in old:
                moves = old[key]
                source = "CACHE I VJETER"
                reused += 1
                break

        if moves is None:
            moves = []
            evaluations = 0
            errors = 0

            for target in range(problem.n_tomatoes):

                if target in route27:
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

            new_scans += 1
            source = "PYKEP"

            print(
                f"  {evaluations} llogaritje | "
                f"{errors} gabime numerike",
                flush=True
            )

        new_cache[key] = moves
        save(CACHE27, new_cache)

    # Kontrollojme levizjet e nivelit 28
    normal = []
    special = []

    for m in moves:
        target = int(m["target"])
        dv = float(m["dv"])
        arrival = float(m["arrival"])

        if target in route27 or arrival > 200:
            continue

        if (
            m["status"] == "NORMAL"
            and 0 <= dv < 100
        ):
            normal.append(m)

        elif (
            m["status"] == "SPECIAL"
            and used < 5
            and 100 <= dv < 600
        ):
            special.append(m)

    all_legal = normal + special

    best = min(
        all_legal,
        key=lambda m: (m["arrival"], m["dv"]),
        default=None
    )

    best_normal = min(
        normal,
        key=lambda m: (m["arrival"], m["dv"]),
        default=None
    )

    row = {
        "parent_time": entry["parent"]["current_time"],
        "parent_specials": entry["parent"]["special_used"],
        "first_target": current,
        "first_status": entry["first_move"]["status"],
        "time27": now,
        "specials27": used,
        "normal_targets28": len({
            m["target"] for m in normal
        }),
        "special_targets28": len({
            m["target"] for m in special
        }),
        "best28": best,
        "best_normal28": best_normal,
        "route27": route27,
        "first_move": entry["first_move"]
    }

    results.append(row)

    print(
        f"[{index}/{len(choices)}] "
        f"27: {current} | "
        f"koha {now:.2f} | "
        f"speciale {used} | "
        f"NORMAL per 28: {row['normal_targets28']} | "
        f"SPECIAL per 28: {row['special_targets28']} | "
        f"{source}",
        flush=True
    )

# 3. Raporti perfundimtar
normal_viable = [
    r for r in results
    if r["normal_targets28"] > 0
]

any_viable = [
    r for r in results
    if (
        r["normal_targets28"] > 0
        or r["special_targets28"] > 0
    )
]

print("\nPERMBLEDHJA V8.15 - DY HAPA PERPARA")
print("=" * 60)

print("Variante te testuara:", len(results))
print("Skanime nga cache:", reused)
print("Skanime te reja PyKEP:", new_scans)
print("Me NORMAL per domaten 28:", len(normal_viable))
print("Me transferim te lejueshem per 28:", len(any_viable))
print("Te bllokuara:", len(results) - len(any_viable))

print("\nSIPAS SPECIALEVE DHE HAPIT TE PARE")

groups = {}

for r in results:
    group_key = (
        r["parent_specials"],
        r["first_status"]
    )
    groups.setdefault(group_key, []).append(r)

for group_key, group in sorted(groups.items()):
    used, status = group_key

    continuing = sum(
        r["best28"] is not None
        for r in group
    )

    print(
        f"Prindi me {used} speciale | "
        f"Hapi i pare {status}: "
        f"{len(group)} variante | "
        f"{continuing} mund te vazhdojne"
    )

print("\nRRUGET QE MUND TE VAZHDOJNE NE NIVELIN 28")
print("=" * 60)

ranked = sorted(
    any_viable,
    key=lambda r: (
        r["best28"]["arrival"],
        r["specials27"]
    )
)

if not ranked:
    print("Nuk u gjet vazhdim me rrjetin aktual.")

for r in ranked[:12]:
    move = r["best28"]
    left = 200 - float(move["arrival"])

    print(
        f"26->27: {r['first_target']} "
        f"({r['first_status']}) | "
        f"27->28: {move['target']} "
        f"({move['status']}) | "
        f"Arritja {move['arrival']:.2f} | "
        f"Speciale pas hapit 27: {r['specials27']} | "
        f"Mbeten {left:.2f} dite"
    )

save(REPORT, {
    "method": "V8.15 Two-Step Lookahead",
    "source_level": 26,
    "target_level": 28,
    "tested": len(results),
    "normal_continuations": len(normal_viable),
    "legal_continuations": len(any_viable),
    "results": results
})

print("\nRaporti:", REPORT.name)
print("Cache:", CACHE27.name)
print("V8.15 PERFUNDOI!")
