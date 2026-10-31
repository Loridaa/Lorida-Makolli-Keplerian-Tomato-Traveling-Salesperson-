import json
import math
import shutil
from pathlib import Path
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

INITIAL = RESULTS / "initial_moves_scan.json"
SEED = RESULTS / "beam_search_v4.json"
PROGRESS = RESULTS / "beam_search_v4_progress.json"

TIMES = [0.5, 1, 2, 3, 5, 7, 10, 15]
WIDTH = 10
TARGET_DEPTH = 8

problem = TomatoProblem(str(ROOT / "data/easy.kttsp"))


def save_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)


def build_seed():
    data = json.loads(INITIAL.read_text())

    def unique(moves, limit, excluded=None):
        chosen = []
        seen = set(excluded or [])

        for target, tof, dv in moves:
            target = int(target)

            if target not in seen:
                chosen.append(
                    (target, float(tof), float(dv))
                )
                seen.add(target)

            if len(chosen) >= limit:
                break

        return chosen

    normal = unique(data["normal"], 2)

    special = unique(
        data["special"],
        6,
        {x[0] for x in normal}
    )

    states = []

    for status, moves in [
        ("NORMAL", normal),
        ("SPECIAL", special)
    ]:
        for target, tof, dv in moves:
            states.append({
                "route": [0, target],
                "transfers": [{
                    "from": 0,
                    "to": target,
                    "start_time": 0.0,
                    "tof": tof,
                    "dv": dv,
                    "status": status
                }],
                "current_time": tof,
                "special_used": int(status == "SPECIAL")
            })

    return states


def expand(state):
    current = state["route"][-1]
    now = float(state["current_time"])
    used = int(state["special_used"])
    visited = set(state["route"])

    normal = {}
    special = {}
    evaluations = 0

    for target in range(problem.n_tomatoes):
        if target in visited:
            continue

        for tof in TIMES:
            if now + tof > problem.max_time:
                continue

            try:
                dv = float(problem.compute_transfer(
                    current, target, now, tof
                ))
                evaluations += 1
            except Exception:
                continue

            if not math.isfinite(dv):
                continue

            if dv < problem.dv_threshold:
                if target not in normal:
                    normal[target] = (
                        target, tof, dv, "NORMAL"
                    )

            elif (
                dv < problem.exception_threshold
                and used < problem.n_exceptions
            ):
                if target not in special:
                    special[target] = (
                        target, tof, dv, "SPECIAL"
                    )

    normals = sorted(
        normal.values(),
        key=lambda x: (x[1], x[2])
    )[:3]

    specials = sorted(
        special.values(),
        key=lambda x: (x[1], x[2])
    )[:3]

    children = []

    for target, tof, dv, status in normals + specials:
        transfer = {
            "from": current,
            "to": target,
            "start_time": now,
            "tof": tof,
            "dv": dv,
            "status": status
        }

        children.append({
            "route": state["route"] + [target],
            "transfers": state["transfers"] + [transfer],
            "current_time": now + tof,
            "special_used": (
                used + int(status == "SPECIAL")
            )
        })

    return children, evaluations


def choose(beam, first_stage=False):
    expanded = []
    evaluations = 0

    for parent_id, state in enumerate(beam):
        children, count = expand(state)
        evaluations += count

        for child in children:
            expanded.append((parent_id, child))

    expanded.sort(
        key=lambda item: (
            item[1]["special_used"],
            item[1]["current_time"]
        )
    )

    selected = []
    seen = set()
    covered = set()

    if first_stage:
        # Ruajme nje vazhdim per cdo nisje fillestare
        for parent_id in range(len(beam)):
            for pid, state in expanded:
                key = tuple(state["route"])

                if pid == parent_id and key not in seen:
                    selected.append(state)
                    seen.add(key)
                    break

            if len(selected) >= WIDTH:
                break

    else:
        # Ruajme nje vazhdim per prinder te ndryshem
        for pid, state in expanded:
            key = tuple(state["route"])

            if pid not in covered and key not in seen:
                selected.append(state)
                seen.add(key)
                covered.add(pid)

            if len(selected) >= WIDTH:
                break

    # Plotesojme vendet e mbetura
    for pid, state in expanded:
        if len(selected) >= WIDTH:
            break

        key = tuple(state["route"])

        if key not in seen:
            selected.append(state)
            seen.add(key)

    return selected, evaluations


def validate(beam, depth):
    if not beam:
        raise RuntimeError("Nuk ka rruge aktive.")

    for state in beam:
        route = state["route"]
        transfers = state["transfers"]

        if len(route) != depth:
            raise RuntimeError("Numer i gabuar domatesh.")

        if len(set(route)) != len(route):
            raise RuntimeError("Domate e perseritur.")

        if len(transfers) != depth - 1:
            raise RuntimeError("Transferime te pavlefshme.")

        if state["special_used"] > problem.n_exceptions:
            raise RuntimeError("Shume manovra speciale.")

        if state["current_time"] > problem.max_time:
            raise RuntimeError("Koha maksimale u tejkalua.")

        if state["special_used"] != sum(
            t["status"] == "SPECIAL"
            for t in transfers
        ):
            raise RuntimeError("Specialet nuk perputhen.")


def record(beam, evaluations=0):
    return {
        "method": "Beam Search V4",
        "beam_width": WIDTH,
        "beam": beam,
        "evaluations_last_level": evaluations,
        "completed": any(
            len(s["route"]) == problem.n_tomatoes
            for s in beam
        )
    }


print("RIKTHIMI AUTOMATIK I BEAM SEARCH V4")
print("=" * 50, flush=True)

# Vazhdojme nga checkpoint nese ekziston
if PROGRESS.exists():
    data = json.loads(PROGRESS.read_text())
    beam = data["beam"]
    depth = len(beam[0]["route"])

    print("Checkpoint ekzistues:", depth, flush=True)

else:
    if SEED.exists():
        data = json.loads(SEED.read_text())
        beam = data["beam"]
        depth = len(beam[0]["route"])

    else:
        initial = build_seed()

        beam, count = choose(
            initial,
            first_stage=True
        )

        depth = 3
        validate(beam, depth)
        save_json(SEED, record(beam, count))

        print("V4 fillestar: 3 domate", flush=True)

    save_json(PROGRESS, record(beam))

validate(beam, depth)

# Vazhdojme deri ne 8 domate
while depth < TARGET_DEPTH:

    snapshot = RESULTS / f"beam_v4_snapshot_{depth}.json"

    if not snapshot.exists():
        save_json(snapshot, record(beam))

    print(
        f"\nNiveli {depth} -> {depth + 1}",
        flush=True
    )

    new_beam, count = choose(beam)

    validate(new_beam, depth + 1)

    depth += 1
    beam = new_beam

    save_json(PROGRESS, record(beam, count))

    snapshot = RESULTS / f"beam_v4_snapshot_{depth}.json"
    save_json(snapshot, record(beam, count))

    print("Domate:", depth, "/49", flush=True)
    print("Rruge:", len(beam), flush=True)
    print(
        "Speciale minimale:",
        min(s["special_used"] for s in beam),
        flush=True
    )
    print("Llogaritje:", count, flush=True)


print("\nREZULTATET PERFUNDIMTARE V4")

for i, state in enumerate(beam, 1):
    print(
        f"Rruga {i}: {state['route']} | "
        f"Koha: {state['current_time']} | "
        f"Speciale: {state['special_used']}"
    )

print("\nDomate:", depth)
print("Rruge aktive:", len(beam))
print("Checkpoint:", PROGRESS.name)

# Krijojme edhe nje ZIP backup
backup = shutil.make_archive(
    "/kaggle/working/esa_spoc4_backup_v4",
    "zip",
    ROOT
)

print("BACKUP ZIP:", backup)
print("RIKTHIMI PERFUNDOI!")
