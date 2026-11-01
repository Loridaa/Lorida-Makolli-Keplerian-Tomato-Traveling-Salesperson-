
from pathlib import Path
from collections import defaultdict
from utils_users import TomatoProblem
import json
import math
import hashlib

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_v9_level_21.json"
OUTPUT = RESULTS / "beam_v9_level_22.json"

problem = TomatoProblem(str(ROOT / "data/easy.kttsp"))
parents_raw = json.loads(SOURCE.read_text())["frontier"]

WAITS = [0, 0.5, 1, 2, 3, 5, 8]
TOFS = [0.5, 1, 1.5, 2, 3, 5, 8, 13]

BEAM_WIDTH = 48

config = {
    "waits": WAITS,
    "tofs": TOFS,
    "beam_width": BEAM_WIDTH
}

def identity(state):
    raw = json.dumps([
        state["route"],
        round(float(state["current_time"]), 6),
        state["special_used"]
    ])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def validate(state):
    route = state["route"]
    transfers = state["transfers"]

    if (
        len(route) != 21
        or len(set(route)) != 21
        or len(transfers) != 20
    ):
        return False, "Strukture jo e sakte"

    time_now = 0.0
    special_count = 0

    for i, transfer in enumerate(transfers):
        try:
            departure = int(transfer["from"])
            destination = int(transfer["to"])
            start = float(transfer["start_time"])
            tof = float(transfer["tof"])
            old_dv = float(transfer["dv"])

            if (
                departure != route[i]
                or destination != route[i + 1]
                or start < time_now - 1e-6
                or tof <= 0
            ):
                return False, f"Gabim ne transferimin {i + 1}"

            dv = float(problem.compute_transfer(
                departure, destination, start, tof
            ))

            if not math.isfinite(dv):
                return False, f"Delta-V jo finite ne hapin {i + 1}"

            if abs(dv - old_dv) > 0.1:
                return False, f"Delta-V nuk perputhet ne hapin {i + 1}"

            if dv < problem.dv_threshold:
                status = "NORMAL"
            elif dv < problem.exception_threshold:
                status = "SPECIAL"
                special_count += 1
            else:
                return False, f"Transferim jashte kufirit ne hapin {i + 1}"

            if transfer.get("status") != status:
                return False, f"Statusi nuk perputhet ne hapin {i + 1}"

            time_now = start + tof

        except Exception as exc:
            return False, f"Gabim leximi: {exc}"

    if abs(time_now - state["current_time"]) > 1e-5:
        return False, "Koha perfundimtare nuk perputhet"

    if special_count != state["special_used"]:
        return False, "Numri i SPECIAL nuk perputhet"

    if time_now > problem.max_time or special_count > 5:
        return False, "Tejkalim i kufijve"

    return True, "OK"

print("KONTROLLI I CHECKPOINT-IT")
print("=" * 60, flush=True)

valid = []
invalid = []

for state in parents_raw:
    ok, reason = validate(state)

    if ok:
        valid.append(state)
    else:
        invalid.append({
            "end": state["route"][-1],
            "reason": reason
        })

print("Rruge te kontrolluara:", len(parents_raw), flush=True)
print("Rruge te vlefshme:", len(valid), flush=True)
print("Rruge me probleme:", len(invalid), flush=True)

for item in invalid[:8]:
    print("KUJDES:", item, flush=True)

if not valid:
    raise RuntimeError("Nuk ka rruge te vlefshme per zgjerim.")

# Japim perparesi rrugeve me speciale te mbetura.
valid.sort(key=lambda s: (
    s["special_used"],
    s["current_time"]
))

seed_ids = [identity(s) for s in valid]

if OUTPUT.exists():
    report = json.loads(OUTPUT.read_text())
    assert report["config"] == config
    assert report["seed_ids"] == seed_ids, (
        "Checkpoint nuk perputhet me rruget."
    )
else:
    report = {
        "method": "Beam Search V9 - level 22",
        "config": config,
        "seed_ids": seed_ids,
        "invalid": invalid,
        "parents": {},
        "frontier": []
    }

def save():
    temp = OUTPUT.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2))
    temp.replace(OUTPUT)

print("\nZGJERIMI NGA 21 NE 22")
print("=" * 60, flush=True)

for index, parent in enumerate(valid, 1):
    pid = identity(parent)

    if pid in report["parents"]:
        print(f"[{index}/{len(valid)}] Nga checkpoint", flush=True)
        continue

    route = parent["route"]
    visited = set(route)
    current = route[-1]
    day = float(parent["current_time"])
    used = int(parent["special_used"])

    remaining_count = problem.n_tomatoes - len(route)
    average_time = (problem.max_time - day) / remaining_count

    max_step = min(16.0, max(6.0, average_time * 3.5))

    normal_moves = []
    special_moves = []
    evaluations = 0

    for target in range(problem.n_tomatoes):
        if target in visited:
            continue

        best_normal = None
        best_special = None

        for wait in WAITS:
            for tof in TOFS:
                if wait + tof > max_step:
                    continue

                start = day + wait
                arrival = start + tof

                if arrival > problem.max_time:
                    continue

                try:
                    dv = float(problem.compute_transfer(
                        current, target, start, tof
                    ))
                    evaluations += 1
                except Exception:
                    continue

                if not math.isfinite(dv):
                    continue

                move = {
                    "from": current,
                    "to": target,
                    "start_time": start,
                    "tof": tof,
                    "arrival": arrival,
                    "dv": dv
                }

                if dv < problem.dv_threshold:
                    if (
                        best_normal is None
                        or (arrival, dv) <
                           (best_normal["arrival"], best_normal["dv"])
                    ):
                        best_normal = move

                elif (
                    dv < problem.exception_threshold
                    and used < problem.n_exceptions
                ):
                    if (
                        best_special is None
                        or (arrival, dv) <
                           (best_special["arrival"], best_special["dv"])
                    ):
                        best_special = move

        if best_normal is not None:
            normal_moves.append(best_normal)

        if best_special is not None:
            special_moves.append(best_special)

    normal_moves.sort(key=lambda m: (m["arrival"], m["dv"]))
    special_moves.sort(key=lambda m: (m["arrival"], m["dv"]))

    children = []

    for status, moves in [
        ("NORMAL", normal_moves[:6]),
        ("SPECIAL", special_moves[:4])
    ]:
        for move in moves:
            transfer = {
                "from": move["from"],
                "to": move["to"],
                "start_time": move["start_time"],
                "tof": move["tof"],
                "dv": move["dv"],
                "status": status
            }

            children.append({
                "route": route + [move["to"]],
                "current_time": move["arrival"],
                "special_used": used + (status == "SPECIAL"),
                "transfers": parent["transfers"] + [transfer]
            })

    report["parents"][pid] = {
        "end": current,
        "day": day,
        "special_used": used,
        "normal_destinations": len(normal_moves),
        "special_destinations": len(special_moves),
        "evaluations": evaluations,
        "children": children
    }

    save()

    print(
        f"[{index}/{len(valid)}] "
        f"Fund {current} | Dita {day:.2f} | "
        f"SPECIAL {used}/5 | "
        f"NORMAL {len(normal_moves)} | "
        f"SPECIAL {len(special_moves)}",
        flush=True
    )

# Bashkojme rruget e reja.
all_children = [
    child
    for parent in report["parents"].values()
    for child in parent["children"]
]

# Ruajme kohe te ndryshme per te njejten gjendje,
# sepse koha e mberritjes ndikon transferimet e ardhshme.
unique = {}

for state in all_children:
    key = (
        tuple(sorted(state["route"])),
        state["route"][-1],
        state["special_used"],
        round(state["current_time"] * 2) / 2
    )

    if (
        key not in unique
        or state["current_time"] < unique[key]["current_time"]
    ):
        unique[key] = state

ranked = sorted(unique.values(), key=lambda s: (
    s["current_time"] + 15 * s["special_used"],
    s["current_time"]
))

frontier = []
seen = set()
end_counts = defaultdict(int)

# Kufij per te ruajtur larmishmerine.
quotas = {
    0: 10,
    1: 10,
    2: 14,
    3: 20,
    4: 20,
    5: 8
}

for used in [3, 4, 2, 1, 0, 5]:
    count = 0

    for state in ranked:
        if count >= quotas[used] or len(frontier) >= BEAM_WIDTH:
            break

        if state["special_used"] != used:
            continue

        end = state["route"][-1]

        if end_counts[end] >= 4:
            continue

        key = identity(state)

        if key in seen:
            continue

        frontier.append(state)
        seen.add(key)
        end_counts[end] += 1
        count += 1

# Plotesojme vendet e lira me rruge te tjera.
for state in ranked:
    if len(frontier) >= BEAM_WIDTH:
        break

    key = identity(state)

    if key not in seen:
        frontier.append(state)
        seen.add(key)

report["frontier"] = frontier
save()

print("\nREZULTATI I V9 - NIVELI 22")
print("=" * 60, flush=True)

print("Prinder te analizuar:", len(report["parents"]), flush=True)
print("Rruge te reja unike:", len(unique), flush=True)
print("Frontier i ri:", len(frontier), flush=True)

for used in range(6):
    group = [s for s in frontier if s["special_used"] == used]

    if group:
        earliest = min(s["current_time"] for s in group)
        print(
            f"SPECIAL {used}/5 | "
            f"Rruge {len(group)} | "
            f"Dita me e hershme {earliest:.2f}",
            flush=True
        )

print("\nHAPI 110 - KERKIMI PERFUNDOI!", flush=True)
