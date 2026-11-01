
import json
import math
from pathlib import Path
from collections import Counter
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

REPORT26 = RESULTS / "beam_v88_candidate_report_26.json"

CACHE26_NEW = RESULTS / "beam_v88_candidate_cache_26.json"
CACHE26_OLD = RESULTS / "beam_v84_diverse_cache_26.json"

CACHE27_OLD = RESULTS / "beam_v85_lookahead_cache_27.json"
CACHE27_NEW = RESULTS / "beam_v89_two_step_cache_27.json"

OUTPUT = RESULTS / "beam_v89_two_step_report_26_28.json"

WAITS = [0, 0.5, 1, 2, 3, 5, 8, 13]
TOFS = [0.5, 1, 1.5, 2, 3, 5, 8, 13, 21, 34]

MAX_CHILDREN = 20


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


def key(route, now, special):
    return json.dumps([
        route, float(now), int(special)
    ])


problem = TomatoProblem(
    str(ROOT / "data/easy.kttsp")
)

report = load(REPORT26)["results"]
cache26new = load(CACHE26_NEW)
cache26old = load(CACHE26_OLD)

cache27old = load(CACHE27_OLD, {})
cache27new = load(CACHE27_NEW, {})

# 1. Gjejme alternativat normale per secilin prind
first_choices = []
second_choices = []

for parent in report:

    route = parent["route"]
    now = float(parent["time"])
    used = int(parent["special_used"])

    parent_key = key(route, now, used)

    moves = cache26new.get(
        parent_key,
        cache26old.get(parent_key)
    )

    if moves is None:
        raise RuntimeError(
            "Mungon cache per nje kandidat ne nivelin 26."
        )

    # Mbajme transferimin me te hershem
    # per secilen domate te ardhshme.
    best_by_target = {}

    for move in moves:

        if move["status"] != "NORMAL":
            continue

        target = int(move["target"])

        if target in route:
            continue

        if not (0 <= float(move["dv"]) <
                problem.dv_threshold):
            continue

        if float(move["arrival"]) > problem.max_time:
            continue

        previous = best_by_target.get(target)

        if previous is None or (
            float(move["arrival"]), float(move["dv"])
        ) < (
            float(previous["arrival"]),
            float(previous["dv"])
        ):
            best_by_target[target] = move

    ranked = sorted(
        best_by_target.values(),
        key=lambda m: (
            float(m["arrival"]),
            float(m["dv"])
        )
    )

    if not ranked:
        continue

    entry = {
        "parent": parent,
        "move": ranked[0]
    }

    first_choices.append(entry)

    if len(ranked) > 1:
        second_choices.append({
            "parent": parent,
            "move": ranked[1]
        })


# Per secilin prind provojme fillimisht
# levizjen me te mire. Pastaj alternativat.
first_choices.sort(
    key=lambda x: float(x["move"]["arrival"])
)

second_choices.sort(
    key=lambda x: float(x["move"]["arrival"])
)

choices = (
    first_choices + second_choices
)[:MAX_CHILDREN]

print("HAPI 52 - V8.9 TWO-STEP LOOKAHEAD")
print("=" * 60)
print("Kandidate te nivelit 26:", len(report))
print("Prinder me transferim NORMAL:", len(first_choices))
print("Rruge te nivelit 27 qe kontrollohen:", len(choices))
print("Synimi: 26 -> 27 -> 28")
print("=" * 60, flush=True)

results = []

for index, entry in enumerate(choices, 1):

    parent = entry["parent"]
    move = entry["move"]

    route27 = (
        parent["route"] + [int(move["target"])]
    )

    now27 = float(move["arrival"])
    used = int(parent["special_used"])
    current = route27[-1]

    assert len(route27) == 27
    assert len(set(route27)) == 27
    assert now27 <= 200

    state_key = key(route27, now27, used)

    if state_key in cache27new:
        moves28 = cache27new[state_key]
        source = "CACHE V8.9"

    elif state_key in cache27old:
        moves28 = cache27old[state_key]
        source = "CACHE V8.5"

    else:
        moves28 = []
        evaluations = 0
        errors = 0

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

                    moves28.append({
                        "target": target,
                        "wait": wait,
                        "tof": tof,
                        "arrival": arrival,
                        "dv": dv,
                        "status": status
                    })

        cache27new[state_key] = moves28

        # Ruajme pas seciles rruge te testuar
        save(CACHE27_NEW, cache27new)

        source = "PYKEP"

        print(
            f"Rruga {index}: "
            f"{evaluations} llogaritje | "
            f"{errors} gabime numerike",
            flush=True
        )

    normal_moves = [
        m for m in moves28
        if (
            m["status"] == "NORMAL"
            and m["target"] not in route27
            and m["arrival"] <= 200
            and m["dv"] < problem.dv_threshold
        )
    ]

    special_moves = [
        m for m in moves28
        if (
            m["status"] == "SPECIAL"
            and used < problem.n_exceptions
            and m["target"] not in route27
            and m["arrival"] <= 200
            and problem.dv_threshold <=
                m["dv"] < problem.exception_threshold
        )
    ]

    normal_targets = len({
        m["target"] for m in normal_moves
    })

    special_targets = len({
        m["target"] for m in special_moves
    })

    best_normal = min(
        normal_moves,
        key=lambda m: (
            m["arrival"], m["dv"]
        ),
        default=None
    )

    best_special = min(
        special_moves,
        key=lambda m: (
            m["arrival"], m["dv"]
        ),
        default=None
    )

    row = {
        "route27": route27,
        "parent_time": parent["time"],
        "parent_endpoint": parent["endpoint"],
        "parent_was_selected": parent["was_selected"],
        "first_target": move["target"],
        "time27": now27,
        "special_used": used,
        "normal_targets_28": normal_targets,
        "special_targets_28": special_targets,
        "best_normal_28": best_normal,
        "best_special_28": best_special
    }

    results.append(row)

    print(
        f"[{index}/{len(choices)}] "
        f"26->27: {current} ne {now27:.2f} | "
        f"27->28 NORMAL: {normal_targets} | "
        f"SPECIAL: {special_targets} | "
        f"{source}",
        flush=True
    )

# 2. Permbledhja
with_normal = [
    r for r in results
    if r["normal_targets_28"] > 0
]

with_any = [
    r for r in results
    if (
        r["normal_targets_28"] > 0
        or r["special_targets_28"] > 0
    )
]

print("\nPERMBLEDHJA V8.9 - DY HAPA PERPARA")
print("=" * 60)

print("Rruge te testuara:", len(results))
print("Me NORMAL edhe per domaten 28:", len(with_normal))
print("Me transferim te lejueshem per 28:", len(with_any))
print("Te bllokuara ne rrjetin aktual:",
      len(results) - len(with_any))

print("\nSIPAS SPECIALEVE")

for used in sorted({
    r["special_used"] for r in results
}):

    group = [
        r for r in results
        if r["special_used"] == used
    ]

    normal_count = sum(
        r["normal_targets_28"] > 0
        for r in group
    )

    print(
        f"{used} speciale: "
        f"{len(group)} rruge | "
        f"{normal_count} me NORMAL per 28"
    )

print("\nRRUGET QE VAZHDOJNE ME DY TRANSFERIME NORMAL")

ranked = sorted(
    with_normal,
    key=lambda r: (
        r["best_normal_28"]["arrival"],
        r["special_used"],
        -r["normal_targets_28"]
    )
)

if not ranked:
    print("Nuk u gjet asnje ne rrjetin aktual.")

for r in ranked[:15]:

    move = r["best_normal_28"]
    remaining = 200 - move["arrival"]
    avg = remaining / (49 - 28)

    origin = (
        "E RUAJTUR"
        if r["parent_was_selected"]
        else "E LENE JASHTE"
    )

    print(
        f"26->27: domatja {r['first_target']} | "
        f"27->28: domatja {move['target']} | "
        f"Koha {move['arrival']:.2f} | "
        f"Speciale {r['special_used']} | "
        f"Normale te mundshme {r['normal_targets_28']} | "
        f"Mbeten {remaining:.2f} dite | "
        f"Mes. {avg:.2f} dite/domate | "
        f"{origin}"
    )

save(OUTPUT, {
    "method": "V8.9 Two-Step Lookahead",
    "source_level": 26,
    "tested_children": len(results),
    "normal_continuations": len(with_normal),
    "results": results
})

print("\nRaporti:", OUTPUT.name)
print("Cache:", CACHE27_NEW.name)
print("V8.9 PERFUNDOI!")
