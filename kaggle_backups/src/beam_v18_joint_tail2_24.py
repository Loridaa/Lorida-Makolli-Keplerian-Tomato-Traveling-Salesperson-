
import json
import math
import time
import hashlib
from pathlib import Path
from collections import defaultdict

from scipy.optimize import differential_evolution
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_v16_diverse_level_23.json"
PREVIOUS = RESULTS / "beam_v17_unlock_special_24_checkpoint.json"

CHECKPOINT = RESULTS / "beam_v18_joint_tail2_24_checkpoint.json"
OUTPUT = RESULTS / "beam_v18_joint_tail2_level_24.json"

P = TomatoProblem(str(ROOT / "data/easy.kttsp"))

START = time.monotonic()
TIME_LIMIT = 105.0

ARRIVAL_LIMIT = 145.0
CANDIDATE_LIMIT = 12


def atomic(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(data, separators=(",", ":")),
        encoding="utf-8"
    )
    temp.replace(path)


def dv(origin, target, depart, tof):
    try:
        value = float(P.compute_transfer(
            int(origin),
            int(target),
            float(depart),
            float(tof)
        ))
        if math.isfinite(value) and value >= 0:
            return value
    except Exception:
        pass

    return float("inf")


def verify(state):
    try:
        route = state["route"]
        transfers = state["transfers"]

        if (
            route[0] != 0
            or len(route) != len(set(route))
            or len(transfers) != len(route) - 1
        ):
            return False

        current_day = 0.0
        specials = 0

        for i, tr in enumerate(transfers):
            depart = float(tr["start_time"])
            tof = float(tr["tof"])

            if (
                tr["from"] != route[i]
                or tr["to"] != route[i + 1]
                or depart < current_day - 1e-6
                or tof <= 0
                or depart + tof > P.max_time + 1e-6
            ):
                return False

            value = dv(
                tr["from"], tr["to"],
                depart, tof
            )

            if (
                not math.isfinite(value)
                or abs(value - float(tr["dv"])) > 0.1
            ):
                return False

            if value < P.dv_threshold:
                status = "NORMAL"
            elif value < P.exception_threshold:
                status = "SPECIAL"
                specials += 1
            else:
                return False

            if tr["status"] != status:
                return False

            current_day = depart + tof

        return (
            specials == state["special_used"]
            and specials <= P.n_exceptions
            and abs(
                current_day - state["current_time"]
            ) < 1e-5
        )

    except Exception:
        return False


# 1. Lexojme rezultatet e Hapit 140 dhe 141.
source_bytes = SOURCE.read_bytes()
source_hash = hashlib.sha256(source_bytes).hexdigest()

parents = json.loads(source_bytes)["frontier"]
previous = json.loads(PREVIOUS.read_text())

assert parents
assert all(
    len(s["route"]) == 23 and verify(s)
    for s in parents
)

# 2. Zgjedhim kandidatet me DV me te afert
# me kufirin SPECIAL.
ranked = []

for index, parent in enumerate(parents):
    if parent["special_used"] > 4:
        continue

    prefix_transfers = parent["transfers"][:-1]
    prefix_used = sum(
        tr["status"] == "SPECIAL"
        for tr in prefix_transfers
    )

    # Me 4 SPECIAL ne prefiks, dy transferimet
    # e reja do te duhej te ishin te dyja NORMAL.
    # Fokusohemi tek prefikset me maksimum 3.
    if prefix_used > 3:
        continue

    for target in range(P.n_tomatoes):
        if target in parent["route"]:
            continue

        key = f"{index}:{target}"
        grid = previous["grid"].get(key)

        if not grid or not grid["best"]:
            continue

        moves = [grid["best"]]

        refined = previous["refined"].get(key)
        if refined is not None:
            moves.append(refined)

        best_move = min(
            moves, key=lambda item: item["dv"]
        )

        if best_move["dv"] > 1050:
            continue

        # Favorizojme prefikset qe kane ruajtur
        # me shume manovra SPECIAL.
        penalty = 0.0

        if prefix_used == 3:
            penalty += 25.0

        if (
            prefix_used == 3
            and parent["transfers"][-1]["status"] == "SPECIAL"
        ):
            penalty += 45.0

        score = best_move["dv"] + penalty

        ranked.append({
            "key": key,
            "index": index,
            "target": target,
            "score": score,
            "reference_dv": best_move["dv"],
            "reference_move": best_move,
            "prefix_used": prefix_used
        })

ranked.sort(
    key=lambda item: (
        item["score"],
        item["reference_dv"]
    )
)

selected = []
per_parent = defaultdict(int)

for candidate in ranked:
    if per_parent[candidate["index"]] >= 3:
        continue

    selected.append(candidate)
    per_parent[candidate["index"]] += 1

    if len(selected) >= CANDIDATE_LIMIT:
        break

assert selected, "Nuk u gjeten kandidate te pershtatshem."

config = {
    "source_hash": source_hash,
    "selected_keys": [c["key"] for c in selected],
    "arrival_limit": ARRIVAL_LIMIT,
    "candidate_limit": CANDIDATE_LIMIT,
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
        "completed": {}
    }
    atomic(CHECKPOINT, checkpoint)

print("FAZA 1 - PERZGJEDHJA E KANDIDATEVE")
print("=" * 65, flush=True)

print(
    "Kandidate per optimizim:",
    len(selected),
    flush=True
)

for i, c in enumerate(selected, 1):
    s = parents[c["index"]]

    print(
        f"[{i}] {s['route'][-2]} -> "
        f"{s['route'][-1]} -> {c['target']} | "
        f"Prefiksi SPECIAL {c['prefix_used']}/5 | "
        f"DV e fundit {c['reference_dv']:.2f}",
        flush=True
    )


def optimize(candidate):
    parent = parents[candidate["index"]]
    target = candidate["target"]

    a = parent["route"][-2]
    b = parent["route"][-1]
    c = target

    old_first = parent["transfers"][-1]
    prefix_transfers = parent["transfers"][:-1]

    last_prefix_transfer = prefix_transfers[-1]

    prefix_day = (
        float(last_prefix_transfer["start_time"])
        + float(last_prefix_transfer["tof"])
    )

    prefix_used = candidate["prefix_used"]

    max_wait = min(
        30.0, ARRIVAL_LIMIT - prefix_day - 0.5
    )

    max_tof1 = min(
        65.0, ARRIVAL_LIMIT - prefix_day - 0.25
    )

    if max_wait <= 0 or max_tof1 <= 0.25:
        return {
            "error": "Nuk ka kohe te mjaftueshme.",
            "solution": None
        }

    # x = pritja1, koha_fluturimit1,
    #     pritja2, koha_fluturimit2.
    bounds = [
        (0.0, max_wait),
        (0.25, max_tof1),
        (0.0, 35.0),
        (0.25, 75.0)
    ]

    def evaluate(x):
        wait1, tof1, wait2, tof2 = map(float, x)

        depart1 = prefix_day + wait1
        arrival1 = depart1 + tof1

        depart2 = arrival1 + wait2
        arrival2 = depart2 + tof2

        if arrival2 > ARRIVAL_LIMIT:
            return None

        dv1 = dv(a, b, depart1, tof1)
        dv2 = dv(b, c, depart2, tof2)

        if not (
            math.isfinite(dv1)
            and math.isfinite(dv2)
        ):
            return None

        specials1 = int(dv1 >= P.dv_threshold)
        specials2 = int(dv2 >= P.dv_threshold)

        used = prefix_used + specials1 + specials2

        return {
            "depart1": depart1,
            "tof1": tof1,
            "dv1": dv1,
            "depart2": depart2,
            "tof2": tof2,
            "dv2": dv2,
            "arrival": arrival2,
            "used": used
        }

    # Penalizojme DV mbi kufirin 600
    # dhe perdorimin e SPECIAL mbi 4.
    def objective(x):
        result = evaluate(x)

        if result is None:
            wait1, tof1, wait2, tof2 = map(float, x)
            arrival = (
                prefix_day + wait1 + tof1 + wait2 + tof2
            )

            return (
                100000.0
                + max(0, arrival - ARRIVAL_LIMIT) * 1000
            )

        d1 = result["dv1"]
        d2 = result["dv2"]

        threshold_gap = (
            max(0, d1 - 599.8)
            + max(0, d2 - 599.8)
        )

        remaining_special = 4 - prefix_used

        if remaining_special >= 2:
            budget_gap = 0.0

        elif remaining_special == 1:
            # Te pakten nje transferim duhet NORMAL.
            budget_gap = max(
                0, min(d1, d2) - 99.8
            )

        else:
            budget_gap = (
                max(0, d1 - 99.8)
                + max(0, d2 - 99.8)
            )

        return (
            5.0 * threshold_gap
            + 3.0 * budget_gap
            + 0.008 * (d1 + d2)
            + 0.015 * result["arrival"]
        )

    reference = candidate["reference_move"]

    x0 = [
        float(old_first["start_time"]) - prefix_day,
        float(old_first["tof"]),
        float(reference["wait"]),
        float(reference["tof"])
    ]

    # Sigurohemi qe x0 te jete brenda bounds.
    x0 = [
        max(low, min(high, value))
        for value, (low, high)
        in zip(x0, bounds)
    ]

    seed = int(
        hashlib.sha256(
            candidate["key"].encode()
        ).hexdigest()[:8],
        16
    )

    result = differential_evolution(
        objective,
        bounds=bounds,
        strategy="best1bin",
        maxiter=35,
        popsize=8,
        tol=0.005,
        polish=True,
        seed=seed,
        x0=x0,
        workers=1
    )

    best = evaluate(result.x)

    output = {
        "origin": a,
        "middle": b,
        "target": c,
        "prefix_special": prefix_used,
        "reference_dv": candidate["reference_dv"],
        "best": best,
        "solution": None
    }

    if best is None:
        return output

    d1 = best["dv1"]
    d2 = best["dv2"]

    if not (
        d1 < P.exception_threshold
        and d2 < P.exception_threshold
        and best["used"] <= 4
    ):
        return output

    status1 = (
        "NORMAL" if d1 < P.dv_threshold
        else "SPECIAL"
    )

    status2 = (
        "NORMAL" if d2 < P.dv_threshold
        else "SPECIAL"
    )

    transfers = prefix_transfers + [
        {
            "from": a,
            "to": b,
            "start_time": best["depart1"],
            "tof": best["tof1"],
            "dv": d1,
            "status": status1
        },
        {
            "from": b,
            "to": c,
            "start_time": best["depart2"],
            "tof": best["tof2"],
            "dv": d2,
            "status": status2
        }
    ]

    solution = {
        "route": parent["route"] + [c],
        "transfers": transfers,
        "current_time": best["arrival"],
        "special_used": best["used"]
    }

    if verify(solution):
        output["solution"] = solution

    return output


# 3. Optimizimi me checkpoint.
print("\nFAZA 2 - OPTIMIZIMI I PERBASHKET")
print("=" * 65, flush=True)

for position, candidate in enumerate(selected, 1):
    if time.monotonic() - START >= TIME_LIMIT:
        break

    key = candidate["key"]

    if key in checkpoint["completed"]:
        continue

    outcome = optimize(candidate)

    checkpoint["completed"][key] = outcome
    atomic(CHECKPOINT, checkpoint)

    best = outcome.get("best")

    if best is None:
        print(
            f"[{position}/{len(selected)}] "
            "Nuk u gjet zgjidhje.",
            flush=True
        )
        continue

    print(
        f"[{position}/{len(selected)}] "
        f"{outcome['origin']} -> "
        f"{outcome['middle']} -> "
        f"{outcome['target']} | "
        f"DV1={best['dv1']:.2f} | "
        f"DV2={best['dv2']:.2f} | "
        f"SPECIAL={best['used']}/5 | "
        f"Dita={best['arrival']:.2f}",
        flush=True
    )

    if outcome["solution"] is not None:
        print(
            "RRUGE E RE E VLEFSHME ME "
            "SPECIAL TE LIRA!",
            flush=True
        )


# 4. Verifikimi final.
solutions = {}

for outcome in checkpoint["completed"].values():
    state = outcome.get("solution")

    if state is None or not verify(state):
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


print("\nPERMBLEDHJA E HAPIT 142")
print("=" * 65, flush=True)

print(
    "Kandidate te optimizuar:",
    len(checkpoint["completed"]),
    "/",
    len(selected),
    flush=True
)

print(
    "Rruge te reja te vlefshme:",
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

gaps = []

for outcome in checkpoint["completed"].values():
    best = outcome.get("best")

    if best is not None:
        gaps.append(
            best["dv2"] - P.exception_threshold
        )

if gaps:
    print(
        "Gap minimale e transferimit te fundit:",
        round(min(gaps), 4),
        flush=True
    )

partial = (
    len(checkpoint["completed"]) < len(selected)
)

print(
    "Kerkimi i pjesshem:",
    partial,
    flush=True
)

print(
    "Rekordi ekzistues i projektit: 31/49",
    flush=True
)

print("HAPI 142 PERFUNDOI!", flush=True)
