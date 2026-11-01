
import json
import math
from pathlib import Path

from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

PARENTS = RESULTS / "beam_v88_candidate_report_26.json"
VARIANTS = RESULTS / "beam_v810_alternative_report_27_28.json"

CACHE = RESULTS / "beam_v811_numerical_cache.json"
OUTPUT = RESULTS / "beam_v811_numerical_report.json"

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def save(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8"
    )
    temp.replace(path)

# Pika te shperndara ne intervale reale.
def vdc(n, base):
    result = 0.0
    fraction = 1.0 / base

    while n:
        n, remainder = divmod(n, base)
        result += remainder * fraction
        fraction /= base

    return result

problem = TomatoProblem(
    str(ROOT / "data/easy.kttsp")
)

parent_report = read(PARENTS)["results"]
variant_report = read(VARIANTS)["results"]

parents = sorted(
    [
        r for r in parent_report
        if r["best_normal"] is not None
        and float(r["best_normal"]["arrival"]) <= 145
    ],
    key=lambda r: float(r["best_normal"]["arrival"])
)

# 4 mberritjet me te hershme +
# 2 variante alternative.
earliest = sorted(
    [r for r in variant_report if r["is_earliest"]],
    key=lambda r: r["time27"]
)[:4]

alternatives = sorted(
    [r for r in variant_report if not r["is_earliest"]],
    key=lambda r: r["time27"]
)[:2]

selected = earliest + alternatives

if CACHE.exists():
    progress = read(CACHE)
    assert progress["version"] == "V8.11-1"
else:
    progress = {
        "version": "V8.11-1",
        "states": {}
    }

print("HAPI 55 - V8.11 NUMERICAL SEARCH")
print("=" * 60)
print("Variante te zgjedhura:", len(selected))
print("Synimi: Niveli 27 -> 28")
print("Wait: 0 deri 13 dite")
print("TOF: 0.25 deri 40 dite")
print("Kerkim numerik + permiresim lokal")
print("=" * 60, flush=True)

summary = []

for index, variant in enumerate(selected, 1):

    parent = parents[variant["parent_index"] - 1]

    route = (
        parent["route"] +
        [int(variant["first_target"])]
    )

    now = float(variant["time27"])
    used = int(variant["special_used"])
    current = route[-1]

    assert len(route) == 27
    assert len(set(route)) == 27
    assert used == int(parent["special_used"])

    state_key = json.dumps([
        route, now, used
    ])

    if state_key not in progress["states"]:
        progress["states"][state_key] = {
            "time": now,
            "special_used": used,
            "endpoint": current,
            "targets": {}
        }

    state = progress["states"][state_key]
    targets = state["targets"]

    remaining = [
        t for t in range(problem.n_tomatoes)
        if t not in route
    ]

    print(
        f"\nVARIANTI {index}/{len(selected)} | "
        f"Domatja {current} | "
        f"Koha {now:.2f} | "
        f"Speciale {used}",
        flush=True
    )

    for target in remaining:

        target_key = str(target)

        if target_key in targets:
            continue

        stats = {
            "min_dv": None,
            "best_normal": None,
            "best_special": None,
            "evaluations": 0,
            "errors": 0,
            "points": []
        }

        seen = set()

        def evaluate(wait, tof):
            wait = max(0.0, min(13.0, float(wait)))
            tof = max(0.25, min(40.0, float(tof)))

            arrival = now + wait + tof

            if arrival > problem.max_time:
                return None

            pair = (round(wait, 8), round(tof, 8))

            if pair in seen:
                return None

            seen.add(pair)

            try:
                dv = float(
                    problem.compute_transfer(
                        current,
                        target,
                        now + wait,
                        tof
                    )
                )
                stats["evaluations"] += 1

            except Exception:
                stats["errors"] += 1
                return None

            if not math.isfinite(dv) or dv < 0:
                return None

            stats["points"].append((dv, wait, tof))

            if (
                stats["min_dv"] is None
                or dv < stats["min_dv"]
            ):
                stats["min_dv"] = dv

            move = {
                "target": target,
                "wait": wait,
                "tof": tof,
                "arrival": arrival,
                "dv": dv
            }

            if dv < problem.dv_threshold:

                move["status"] = "NORMAL"
                old = stats["best_normal"]

                if old is None or (
                    arrival, dv
                ) < (
                    old["arrival"], old["dv"]
                ):
                    stats["best_normal"] = move

            elif (
                dv < problem.exception_threshold
                and used < problem.n_exceptions
            ):

                move["status"] = "SPECIAL"
                old = stats["best_special"]

                if old is None or (
                    arrival, dv
                ) < (
                    old["arrival"], old["dv"]
                ):
                    stats["best_special"] = move

            return dv

        # FAZA A:
        # Shperndarje numerike ne hapesiren
        # e kohes se pritjes dhe fluturimit.
        for j in range(1, 181):

            wait = 13.0 * vdc(j, 2)
            tof = 0.25 + 39.75 * vdc(j, 3)

            evaluate(wait, tof)

        # FAZA B:
        # Permiresim rreth tre pikave
        # me DeltaV me te ulet.
        seeds = sorted(
            stats["points"],
            key=lambda p: p[0]
        )[:3]

        steps = [
            (2.0, 3.0),
            (0.8, 1.2),
            (0.3, 0.45),
            (0.1, 0.15),
            (0.025, 0.05)
        ]

        for seed in seeds:

            best_dv, best_wait, best_tof = seed

            for dw, dt in steps:

                for aw in (-1, 0, 1):
                    for at in (-1, 0, 1):

                        w = best_wait + aw * dw
                        t = best_tof + at * dt

                        dv = evaluate(w, t)

                        if dv is not None and dv < best_dv:
                            best_dv = dv
                            best_wait = max(
                                0.0, min(13.0, w)
                            )
                            best_tof = max(
                                0.25, min(40.0, t)
                            )

        targets[target_key] = {
            "min_dv": stats["min_dv"],
            "best_normal": stats["best_normal"],
            "best_special": stats["best_special"],
            "evaluations": stats["evaluations"],
            "errors": stats["errors"]
        }

        # Ruajme pas cdo domateje
        save(CACHE, progress)

        if stats["best_normal"] is not None:
            m = stats["best_normal"]

            print(
                f"Domatja {target}: NORMAL | "
                f"DeltaV {m['dv']:.2f} | "
                f"Arritja {m['arrival']:.2f}",
                flush=True
            )

    # PERMBLEDHJA PER VARIANTIN
    normal_options = [
        x["best_normal"]
        for x in targets.values()
        if x["best_normal"] is not None
    ]

    special_options = [
        x["best_special"]
        for x in targets.values()
        if x["best_special"] is not None
    ]

    dvs = [
        x["min_dv"]
        for x in targets.values()
        if x["min_dv"] is not None
    ]

    best_normal = min(
        normal_options,
        key=lambda m: (m["arrival"], m["dv"]),
        default=None
    )

    result = {
        "variant": index,
        "time27": now,
        "endpoint": current,
        "special_used": used,
        "normal_targets": len(normal_options),
        "special_targets": len(special_options),
        "minimum_dv": min(dvs) if dvs else None,
        "best_normal": best_normal
    }

    summary.append(result)

    print(
        f"PERFUNDIMI {index}: "
        f"NORMAL {len(normal_options)} | "
        f"SPECIAL {len(special_options)} | "
        f"DeltaV min "
        f"{result['minimum_dv']}",
        flush=True
    )

save(OUTPUT, {
    "method": "V8.11 Numerical Search",
    "results": summary
})

print("\nREZULTATI PERFUNDIMTAR - V8.11")
print("=" * 60)

for r in summary:

    dv = r["minimum_dv"]
    dv_text = (
        f"{dv:.2f}"
        if dv is not None
        else "N/A"
    )

    print(
        f"Varianti {r['variant']} | "
        f"Koha {r['time27']:.2f} | "
        f"Fund {r['endpoint']} | "
        f"Min DeltaV {dv_text} | "
        f"NORMAL {r['normal_targets']} | "
        f"SPECIAL {r['special_targets']}"
    )

print("\nRRUGET QE MUND TE VAZHDOJNE")

viable = [
    r for r in summary
    if r["best_normal"] is not None
]

if not viable:
    print(
        "Nuk u gjet NORMAL ne kerkimin numerik."
    )

for r in sorted(
    viable,
    key=lambda x: x["best_normal"]["arrival"]
):
    move = r["best_normal"]

    print(
        f"Varianti {r['variant']} | "
        f"Domatja {move['target']} | "
        f"Arritja {move['arrival']:.2f} | "
        f"DeltaV {move['dv']:.2f}"
    )

print("\nCache:", CACHE.name)
print("Raporti:", OUTPUT.name)
print("HAPI 55 PERFUNDOI!")
