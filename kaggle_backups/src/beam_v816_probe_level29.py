
import json
import math
from pathlib import Path
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE26 = RESULTS / "beam_v813_wide_level_26.json"
REPORT28 = RESULTS / "beam_v815_two_step_report_27_28.json"

LEVEL28 = RESULTS / "beam_v816_viable_level_28.json"
CACHE29 = RESULTS / "beam_v816_probe_cache_28.json"
REPORT29 = RESULTS / "beam_v816_probe_report_28_29.json"

WAITS = [0, 0.5, 1, 2, 3, 5, 8, 13]
TOFS = [0.5, 1, 1.5, 2, 3, 5, 8, 13, 21, 34]


def load(path, default=None):
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)

    return json.loads(path.read_text(encoding="utf-8"))


def save(path, data):
    temporary = path.with_suffix(".tmp")

    temporary.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8"
    )

    temporary.replace(path)


def key(s):
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


def make_transfer(start_tomato, now, move):
    return {
        "from": int(start_tomato),
        "to": int(move["target"]),
        "start_time": float(now) + float(move["wait"]),
        "wait": float(move["wait"]),
        "tof": float(move["tof"]),
        "dv": float(move["dv"]),
        "status": move["status"]
    }


def validate_move(move, used, route, now):
    target = int(move["target"])
    dv = float(move["dv"])
    arrival = float(move["arrival"])

    assert target not in route
    assert arrival <= 200
    assert float(move["wait"]) >= 0
    assert float(move["tof"]) > 0

    assert abs(
        now + float(move["wait"])
        + float(move["tof"]) - arrival
    ) < 1e-6

    if move["status"] == "NORMAL":
        assert 0 <= dv < 100
        return used

    if move["status"] == "SPECIAL":
        assert 100 <= dv < 600
        assert used < 5
        return used + 1

    raise ValueError("Status transferimi i panjohur.")


problem = TomatoProblem(
    str(ROOT / "data/easy.kttsp")
)

parents26 = load(SOURCE26)["beam"]
rows = load(REPORT28)["results"]

# 1. Rikrijojme 7 rruget e nivelit 28
states = {}

for row in rows:
    second_move = row["best28"]

    if second_move is None:
        continue

    route27 = row["route27"]
    first_move = row["first_move"]

    matching = [
        s for s in parents26
        if s["route"] == route27[:-1]
        and s["special_used"] == row["parent_specials"]
        and abs(
            float(s["current_time"])
            - float(row["parent_time"])
        ) < 1e-6
    ]

    if not matching:
        raise RuntimeError(
            "Nuk u gjet prindi ne nivelin 26."
        )

    parent = matching[0]

    now26 = float(parent["current_time"])

    used27 = validate_move(
        first_move,
        int(parent["special_used"]),
        parent["route"],
        now26
    )

    assert used27 == row["specials27"]
    assert int(first_move["target"]) == route27[-1]

    now27 = float(first_move["arrival"])

    used28 = validate_move(
        second_move,
        used27,
        route27,
        now27
    )

    transfer27 = make_transfer(
        parent["route"][-1],
        now26,
        first_move
    )

    transfer28 = make_transfer(
        route27[-1],
        now27,
        second_move
    )

    child = {
        "route": route27 + [int(second_move["target"])],
        "transfers": (
            parent["transfers"]
            + [transfer27, transfer28]
        ),
        "current_time": float(second_move["arrival"]),
        "special_used": used28
    }

    assert len(child["route"]) == 28
    assert len(set(child["route"])) == 28
    assert len(child["transfers"]) == 27

    assert sum(
        t["status"] == "SPECIAL"
        for t in child["transfers"]
    ) == used28

    states[signature(child)] = child

beam28 = sorted(
    states.values(),
    key=lambda s: (
        s["current_time"],
        s["special_used"]
    )
)

if not beam28:
    raise RuntimeError(
        "Nuk u gjet asnje rruge per nivelin 28."
    )

if LEVEL28.exists():
    previous = load(LEVEL28)["beam"]

    if {
        signature(s) for s in previous
    } != {
        signature(s) for s in beam28
    }:
        raise RuntimeError(
            "Checkpoint-i ekzistues nuk perputhet."
        )
else:
    save(LEVEL28, {
        "method": "V8.16 Viable Level 28",
        "source": REPORT28.name,
        "beam": beam28,
        "completed": False
    })

print("HAPI 68 - V8.16 PROBE LEVEL 29")
print("=" * 60)
print("Domate te vizituara: 28/49")
print("Rruge te rikrijuara:", len(beam28))
print("Synimi: 28 -> 29")
print("=" * 60, flush=True)

for i, s in enumerate(beam28, 1):
    print(
        f"Rruga {i}: "
        f"koha {s['current_time']:.2f} | "
        f"speciale {s['special_used']} | "
        f"fund {s['route'][-1]}",
        flush=True
    )

# 2. Skanimi i transferimeve drejt nivelit 29
cache = load(CACHE29, {})

results = []
new_evaluations = 0

for i, state in enumerate(beam28, 1):

    route = state["route"]
    now = float(state["current_time"])
    used = int(state["special_used"])
    current = route[-1]

    state_key = key(state)

    if state_key not in cache:
        cache[state_key] = {
            "targets": {}
        }

    saved_targets = cache[state_key]["targets"]

    remaining_targets = [
        target for target in range(problem.n_tomatoes)
        if target not in route
    ]

    for target in remaining_targets:

        target_key = str(target)

        if target_key in saved_targets:
            continue

        best_normal = None
        best_special = None
        min_dv = None
        evaluations = 0
        errors = 0

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

                if min_dv is None or dv < min_dv:
                    min_dv = dv

                move = {
                    "target": target,
                    "wait": wait,
                    "tof": tof,
                    "arrival": arrival,
                    "dv": dv
                }

                if dv < problem.dv_threshold:

                    move["status"] = "NORMAL"

                    if best_normal is None or (
                        arrival, dv
                    ) < (
                        best_normal["arrival"],
                        best_normal["dv"]
                    ):
                        best_normal = move

                elif (
                    dv < problem.exception_threshold
                    and used < problem.n_exceptions
                ):

                    move["status"] = "SPECIAL"

                    if best_special is None or (
                        arrival, dv
                    ) < (
                        best_special["arrival"],
                        best_special["dv"]
                    ):
                        best_special = move

        new_evaluations += evaluations

        saved_targets[target_key] = {
            "min_dv": min_dv,
            "best_normal": best_normal,
            "best_special": best_special,
            "evaluations": evaluations,
            "errors": errors
        }

        # Ruajme progresin pas cdo domateje
        save(CACHE29, cache)

    normal_options = [
        item["best_normal"]
        for item in saved_targets.values()
        if item["best_normal"] is not None
    ]

    special_options = [
        item["best_special"]
        for item in saved_targets.values()
        if item["best_special"] is not None
    ]

    all_options = normal_options + special_options

    best_next = min(
        all_options,
        key=lambda m: (m["arrival"], m["dv"]),
        default=None
    )

    valid_dvs = [
        item["min_dv"]
        for item in saved_targets.values()
        if item["min_dv"] is not None
    ]

    result = {
        "route": route,
        "time28": now,
        "special_used": used,
        "endpoint": current,
        "normal_targets": len(normal_options),
        "special_targets": len(special_options),
        "minimum_dv": min(valid_dvs) if valid_dvs else None,
        "best_next": best_next
    }

    results.append(result)

    print(
        f"\n[{i}/{len(beam28)}] "
        f"Koha {now:.2f} | "
        f"Fund {current} | "
        f"Speciale {used} | "
        f"NORMAL {len(normal_options)} | "
        f"SPECIAL {len(special_options)}",
        flush=True
    )

    if best_next:
        print(
            f"  Domatja {best_next['target']} | "
            f"Arritja {best_next['arrival']:.2f} | "
            f"{best_next['status']}",
            flush=True
        )

    # Ruajme raportin edhe gjate progresit
    save(REPORT29, {
        "method": "V8.16 Probe Level 29",
        "level": 28,
        "results": results
    })

# 3. Permbledhja
with_normal = [
    r for r in results
    if r["normal_targets"] > 0
]

with_any = [
    r for r in results
    if r["best_next"] is not None
]

print("\nPERMBLEDHJA V8.16 - NIVELI 28 -> 29")
print("=" * 60)
print("Rruge te testuara:", len(results))
print("Me transferim NORMAL:", len(with_normal))
print("Me transferim te lejueshem:", len(with_any))
print("Te bllokuara:", len(results) - len(with_any))
print("Llogaritje te reja PyKEP:", new_evaluations)

print("\nRRUGET QE MUND TE VAZHDOJNE")

for r in sorted(
    with_any,
    key=lambda x: x["best_next"]["arrival"]
):

    m = r["best_next"]
    days_left = 200 - m["arrival"]

    print(
        f"28: fund {r['endpoint']} | "
        f"29: domatja {m['target']} | "
        f"Arritja {m['arrival']:.2f} | "
        f"DeltaV {m['dv']:.2f} | "
        f"Speciale {r['special_used']} | "
        f"Mbeten {days_left:.2f} dite"
    )

if not with_any:
    print(
        "Nuk u gjet vazhdim ne rrjetin aktual."
    )

print("\nCheckpoint:", LEVEL28.name)
print("Cache:", CACHE29.name)
print("Raporti:", REPORT29.name)
print("HAPI 68 PERFUNDOI!")
