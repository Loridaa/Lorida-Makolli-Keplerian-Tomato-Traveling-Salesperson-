
import json
import math
import time
import hashlib
from pathlib import Path

from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_v16_diverse_level_22.json"
CHECKPOINT = RESULTS / "beam_v19_phase_checkpoint.json"
OUTPUT = RESULTS / "beam_v19_phase_level_24.json"
NEAR = RESULTS / "beam_v19_phase_nearmisses.json"

P = TomatoProblem(str(ROOT / "data/easy.kttsp"))

START = time.monotonic()
TIME_LIMIT = 110.0

MAX_FIRST = 32
FIRST_ARRIVAL_LIMIT = 130.0
FINAL_ARRIVAL_LIMIT = 145.0

FIRST_WAITS = [0, .25, .5, 1, 2, 3, 5, 8, 12, 16]
FIRST_TOFS = [
    .25, .5, 1, 1.5, 2, 3, 5,
    8, 12, 20, 30, 40
]

SECOND_WAITS = [0, .5, 2, 5, 10, 20]
SECOND_TOFS = [
    .5, 1, 2, 3, 5, 8,
    12, 20, 30, 40, 55
]


def atomic(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(data, separators=(",", ":")),
        encoding="utf-8"
    )
    temp.replace(path)


def dv(a, b, depart, tof):
    try:
        value = float(P.compute_transfer(
            int(a), int(b),
            float(depart), float(tof)
        ))

        if math.isfinite(value) and value >= 0:
            return value
    except Exception:
        pass

    return float("inf")


def verify(s):
    try:
        route = s["route"]
        transfers = s["transfers"]

        if (
            not route
            or route[0] != 0
            or len(route) != len(set(route))
            or len(transfers) != len(route) - 1
        ):
            return False

        day = 0.0
        special = 0

        for i, tr in enumerate(transfers):
            depart = float(tr["start_time"])
            tof = float(tr["tof"])

            if (
                tr["from"] != route[i]
                or tr["to"] != route[i + 1]
                or depart < day - 1e-6
                or tof <= 0
                or depart + tof > P.max_time + 1e-6
            ):
                return False

            calculated = dv(
                tr["from"], tr["to"],
                depart, tof
            )

            if (
                not math.isfinite(calculated)
                or abs(calculated - float(tr["dv"])) > .1
            ):
                return False

            if calculated < P.dv_threshold:
                status = "NORMAL"
            elif calculated < P.exception_threshold:
                status = "SPECIAL"
                special += 1
            else:
                return False

            if status != tr["status"]:
                return False

            day = depart + tof

        return (
            special == s["special_used"]
            and special <= P.n_exceptions
            and abs(day - s["current_time"]) < 1e-5
        )

    except Exception:
        return False


raw = SOURCE.read_bytes()
source_hash = hashlib.sha256(raw).hexdigest()

frontier = json.loads(raw)["frontier"]

parents = [
    (i, s)
    for i, s in enumerate(frontier)
    if s["special_used"] == 3
]

parents.sort(key=lambda pair: (
    pair[1]["current_time"],
    pair[1]["route"][-1],
    pair[0]
))

assert parents, "Nuk ka rruge me 3 SPECIAL."

for _, state in parents:
    assert len(state["route"]) == 22
    assert verify(state)

config = {
    "source_hash": source_hash,
    "max_first": MAX_FIRST,
    "first_arrival_limit": FIRST_ARRIVAL_LIMIT,
    "final_arrival_limit": FINAL_ARRIVAL_LIMIT,
    "first_waits": FIRST_WAITS,
    "first_tofs": FIRST_TOFS,
    "second_waits": SECOND_WAITS,
    "second_tofs": SECOND_TOFS,
    "version": 1
}

if CHECKPOINT.exists():
    checkpoint = json.loads(
        CHECKPOINT.read_text(encoding="utf-8")
    )
    assert checkpoint["config"] == config
else:
    checkpoint = {
        "config": config,
        "parents": {}
    }
    atomic(CHECKPOINT, checkpoint)


def generate_first(parent):
    origin = parent["route"][-1]
    day = float(parent["current_time"])
    visited = set(parent["route"])

    # Ruajme disa faza kohore per te
    # njejten domate, jo vetem me te hershmen.
    buckets = {}

    for target in range(P.n_tomatoes):
        if target in visited:
            continue

        for wait in FIRST_WAITS:
            depart = day + wait

            for tof in FIRST_TOFS:
                arrival = depart + tof

                if arrival > FIRST_ARRIVAL_LIMIT:
                    continue

                value = dv(
                    origin, target, depart, tof
                )

                # Transferimi i pare duhet NORMAL.
                if value >= P.dv_threshold:
                    continue

                phase = int(
                    (arrival - day) // 3.0
                )

                key = (target, phase)

                candidate = {
                    "target": target,
                    "depart": depart,
                    "tof": tof,
                    "dv": value,
                    "arrival": arrival
                }

                previous = buckets.get(key)

                if (
                    previous is None
                    or (
                        candidate["dv"]
                        + .2 * candidate["arrival"]
                    ) < (
                        previous["dv"]
                        + .2 * previous["arrival"]
                    )
                ):
                    buckets[key] = candidate

    ranked = sorted(
        buckets.values(),
        key=lambda x: (
            x["arrival"]
            + .03 * x["dv"],
            x["dv"]
        )
    )

    selected = []
    per_target = {}

    for candidate in ranked:
        target = candidate["target"]
        count = per_target.get(target, 0)

        if count >= 4:
            continue

        selected.append(candidate)
        per_target[target] = count + 1

        if len(selected) >= MAX_FIRST:
            break

    return selected


def check_second(parent, first):
    origin = first["target"]
    day = first["arrival"]

    visited = set(parent["route"])
    visited.add(origin)

    best = None
    solutions = []
    evaluations = 0

    # Provojme te gjitha domatet ende te pavizituara.
    for target in range(P.n_tomatoes):
        if target in visited:
            continue

        best_valid = None

        for wait in SECOND_WAITS:
            depart = day + wait

            for tof in SECOND_TOFS:
                arrival = depart + tof

                if arrival > FINAL_ARRIVAL_LIMIT:
                    continue

                value = dv(
                    origin, target, depart, tof
                )
                evaluations += 1

                if not math.isfinite(value):
                    continue

                item = {
                    "from": origin,
                    "to": target,
                    "depart": depart,
                    "tof": tof,
                    "dv": value,
                    "arrival": arrival
                }

                if (
                    best is None
                    or value < best["dv"]
                ):
                    best = dict(item)

                if value >= P.exception_threshold:
                    continue

                if (
                    best_valid is None
                    or (
                        arrival,
                        value
                    ) < (
                        best_valid["arrival"],
                        best_valid["dv"]
                    )
                ):
                    best_valid = item

        if best_valid is None:
            continue

        value = best_valid["dv"]

        status = (
            "NORMAL"
            if value < P.dv_threshold
            else "SPECIAL"
        )

        first_transfer = {
            "from": parent["route"][-1],
            "to": origin,
            "start_time": first["depart"],
            "tof": first["tof"],
            "dv": first["dv"],
            "status": "NORMAL"
        }

        second_transfer = {
            "from": origin,
            "to": target,
            "start_time": best_valid["depart"],
            "tof": best_valid["tof"],
            "dv": value,
            "status": status
        }

        child = {
            "route": (
                parent["route"]
                + [origin, target]
            ),
            "transfers": (
                parent["transfers"]
                + [first_transfer, second_transfer]
            ),
            "current_time": best_valid["arrival"],
            "special_used": (
                parent["special_used"]
                + int(status == "SPECIAL")
            )
        }

        if verify(child):
            solutions.append(child)

    return {
        "best": best,
        "solutions": solutions,
        "evaluations": evaluations
    }


print("FAZA 1 - BURIMI")
print("=" * 65, flush=True)
print(
    "Rruge fillestare me 3/5 SPECIAL:",
    len(parents),
    flush=True
)

print("\nFAZA 2 - KERKIMI DY-HAPESH")
print("=" * 65, flush=True)

for index, parent in parents:
    if time.monotonic() - START >= TIME_LIMIT:
        break

    parent_key = str(index)

    if parent_key not in checkpoint["parents"]:
        firsts = generate_first(parent)

        checkpoint["parents"][parent_key] = {
            "firsts": firsts,
            "done": {}
        }
        atomic(CHECKPOINT, checkpoint)

        print(
            f"\nPrindi {index} | "
            f"Fund {parent['route'][-1]} | "
            f"Dita {parent['current_time']:.2f} | "
            f"Alternativa NORMAL: {len(firsts)}",
            flush=True
        )

    record = checkpoint["parents"][parent_key]
    firsts = record["firsts"]

    for j, first in enumerate(firsts):
        if time.monotonic() - START >= TIME_LIMIT:
            break

        task_key = str(j)

        if task_key in record["done"]:
            continue

        outcome = check_second(parent, first)

        record["done"][task_key] = outcome
        atomic(CHECKPOINT, checkpoint)

        best = outcome["best"]
        found = len(outcome["solutions"])

        if best is not None:
            print(
                f"Prindi {index} "
                f"[{j + 1}/{len(firsts)}] "
                f"{parent['route'][-1]}"
                f"->{first['target']}->... | "
                f"DV2 min {best['dv']:.2f} | "
                f"Rruge24 {found}",
                flush=True
            )

        if found:
            print(
                "U GJET RRUGE E VLEFSHME "
                "ME SPECIAL TE LIRA!",
                flush=True
            )


# FAZA 3 - Rezultatet e verifikuara.
solutions = {}
near = []

tasks_done = 0
tasks_total = 0
evaluation_count = 0

for index, parent in parents:
    record = checkpoint["parents"].get(
        str(index)
    )

    if record is None:
        tasks_total += MAX_FIRST
        continue

    tasks_total += len(record["firsts"])

    for outcome in record["done"].values():
        tasks_done += 1
        evaluation_count += outcome["evaluations"]

        if outcome["best"] is not None:
            near.append(outcome["best"])

        for state in outcome["solutions"]:
            if not verify(state):
                continue

            identity = (
                tuple(sorted(state["route"])),
                state["route"][-1],
                state["special_used"]
            )

            if (
                identity not in solutions
                or state["current_time"]
                < solutions[identity]["current_time"]
            ):
                solutions[identity] = state

verified = sorted(
    solutions.values(),
    key=lambda s: (
        s["special_used"],
        s["current_time"]
    )
)

if verified:
    atomic(OUTPUT, {
        "level": 24,
        "verified": True,
        "frontier": verified
    })

near.sort(key=lambda x: x["dv"])

if near:
    atomic(NEAR, {
        "candidates": near[:100]
    })


print("\nPERMBLEDHJA E HAPIT 143")
print("=" * 65, flush=True)

print(
    "Rruge fillestare:",
    len(parents),
    flush=True
)
print(
    "Alternativa dy-hapeshe te perfunduara:",
    tasks_done,
    "/",
    tasks_total,
    flush=True
)
print(
    "Llogaritje te transferimit te dyte:",
    evaluation_count,
    flush=True
)
print(
    "Rruge te reja me 24 domate:",
    len(verified),
    flush=True
)

for used in range(5):
    group = [
        s for s in verified
        if s["special_used"] == used
    ]

    if group:
        print(
            f"SPECIAL {used}/5 | "
            f"Rruge {len(group)} | "
            f"Dita minimale "
            f"{min(s['current_time'] for s in group):.2f}",
            flush=True
        )

if near:
    best = near[0]

    print(
        "Delta-V minimale e transferimit te dyte:",
        round(best["dv"], 4),
        flush=True
    )
    print(
        "Diferenca nga kufiri SPECIAL:",
        round(
            best["dv"] - P.exception_threshold,
            4
        ),
        flush=True
    )
    print(
        "Transferimi me premtues:",
        f"{best['from']}->{best['to']}",
        flush=True
    )

partial = any(
    str(index) not in checkpoint["parents"]
    or len(
        checkpoint["parents"][str(index)]["done"]
    ) < len(
        checkpoint["parents"][str(index)]["firsts"]
    )
    for index, _ in parents
)

print(
    "Kerkimi i pjesshem:",
    partial,
    flush=True
)
print(
    "Rekordi ekzistues: 31/49",
    flush=True
)
print("HAPI 143 PERFUNDOI!", flush=True)
