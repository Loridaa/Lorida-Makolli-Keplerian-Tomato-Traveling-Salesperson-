
import json
import math
import time
import hashlib
from pathlib import Path
from scipy.optimize import minimize
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

OUTPUT = RESULTS / "beam_v11_two_step_32_checkpoint.json"
SNAPSHOT = RESULTS / "beam_v11_lookahead_level_32.json"

P = TomatoProblem(str(ROOT / "data/easy.kttsp"))

WAIT1 = [0, 0.25, 0.5, 1, 2, 3, 5, 8, 13]
TOF1 = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 13, 18]

WAIT2 = [0, 0.25, 0.5, 1, 2, 3, 5, 8]
TOF2 = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 13]

MAX_FIRST = 8
WORK_LIMIT = 135.0
START = time.monotonic()

class PauseSearch(Exception):
    pass

def check_time():
    if time.monotonic() - START >= WORK_LIMIT:
        raise PauseSearch()

def signature(state):
    payload = [
        state["route"],
        round(float(state["current_time"]), 7),
        int(state["special_used"])
    ]

    return hashlib.sha256(
        json.dumps(payload).encode()
    ).hexdigest()[:20]

def verify(state):
    try:
        route = state["route"]
        transfers = state["transfers"]

        if (
            len(route) != len(set(route))
            or not route
            or route[0] != 0
            or len(route) > P.n_tomatoes
            or len(transfers) != len(route) - 1
        ):
            return False

        day = 0.0
        used = 0

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

            dv = float(P.compute_transfer(
                route[i], route[i + 1], depart, tof
            ))

            if (
                not math.isfinite(dv)
                or abs(dv - float(tr["dv"])) > 0.1
            ):
                return False

            if dv < P.dv_threshold:
                status = "NORMAL"
            elif dv < P.exception_threshold:
                status = "SPECIAL"
                used += 1
            else:
                return False

            if tr["status"] != status:
                return False

            day = depart + tof

        return (
            abs(day - float(state["current_time"])) < 1e-5
            and used == state["special_used"]
            and used <= P.n_exceptions
        )

    except Exception:
        return False

def make_child(parent, move):
    transfer = {
        key: move[key]
        for key in [
            "from", "to", "start_time",
            "tof", "dv", "status"
        ]
    }

    return {
        "route": parent["route"] + [move["to"]],
        "transfers": parent["transfers"] + [transfer],
        "current_time": move["arrival"],
        "special_used": (
            parent["special_used"]
            + int(move["status"] == "SPECIAL")
        )
    }

def save():
    temp = OUTPUT.with_suffix(".tmp")

    temp.write_text(
        json.dumps(report, separators=(",", ":"))
    )

    temp.replace(OUTPUT)

def evaluate_move(current, target, depart, tof, used):
    if (
        tof <= 0
        or depart + tof > P.max_time
    ):
        return None

    try:
        dv = float(P.compute_transfer(
            current, target, depart, tof
        ))
    except Exception:
        return None

    if not math.isfinite(dv):
        return None

    if dv < P.dv_threshold:
        status = "NORMAL"

    elif (
        dv < P.exception_threshold
        and used < P.n_exceptions
    ):
        status = "SPECIAL"

    else:
        return None

    return {
        "from": current,
        "to": target,
        "start_time": depart,
        "tof": tof,
        "arrival": depart + tof,
        "dv": dv,
        "status": status
    }

# 1. Lexojme 50 rruget e nivelit 30.
sources = {}
source_hashes = {}

for name in ["A14", "B12"]:
    path = RESULTS / f"beam_v11_{name}_level_30.json"
    raw = path.read_bytes()

    source_hashes[name] = hashlib.sha256(raw).hexdigest()

    states = json.loads(raw)["frontier"]

    assert all(len(s["route"]) == 30 for s in states)

    valid = [s for s in states if verify(s)]
    assert len(valid) == len(states)

    sources[name] = sorted(
        valid,
        key=lambda s: (
            s["special_used"],
            s["current_time"]
        )
    )

CONFIG = {
    "source_hashes": source_hashes,
    "wait1": WAIT1,
    "tof1": TOF1,
    "wait2": WAIT2,
    "tof2": TOF2,
    "max_first": MAX_FIRST,
    "version": 1
}

if OUTPUT.exists():
    report = json.loads(OUTPUT.read_text())
    assert report["config"] == CONFIG
else:
    report = {
        "config": CONFIG,
        "lanes": {
            name: {}
            for name in sources
        }
    }
    save()

def first_moves(state):
    current = state["route"][-1]
    day = float(state["current_time"])
    used = int(state["special_used"])
    visited = set(state["route"])

    choices = []

    for target in range(P.n_tomatoes):
        if target in visited:
            continue

        normal = []
        special = []

        for wait in WAIT1:
            for tof in TOF1:
                if day + wait + tof > P.max_time:
                    continue

                move = evaluate_move(
                    current, target,
                    day + wait, tof, used
                )

                if move is None:
                    continue

                if move["status"] == "NORMAL":
                    normal.append(move)
                else:
                    special.append(move)

        # Nuk marrim vetem kohen me te hershme.
        # Ruajme edhe nje variant me Delta-V
        # te ulet, qe mund te krijoje rruge tjeter.
        for group in [normal, special]:
            if not group:
                continue

            earliest = min(
                group,
                key=lambda m: (m["arrival"], m["dv"])
            )

            choices.append(earliest)

            lowest = min(
                group,
                key=lambda m: (m["dv"], m["arrival"])
            )

            if abs(
                lowest["arrival"] - earliest["arrival"]
            ) >= 0.5:
                choices.append(lowest)

    # Fillimisht nje kandidat per destinacion,
    # qe te shmangim te njejtat funde.
    choices.sort(
        key=lambda m: (
            m["status"] == "SPECIAL",
            m["arrival"],
            m["dv"]
        )
    )

    picked = []
    seen = set()
    destinations = set()

    for move in choices:
        if len(picked) >= MAX_FIRST:
            break

        if move["to"] in destinations:
            continue

        picked.append(move)
        destinations.add(move["to"])

        seen.add((
            move["to"],
            move["status"],
            round(move["arrival"], 5)
        ))

    for move in choices:
        if len(picked) >= MAX_FIRST:
            break

        key = (
            move["to"],
            move["status"],
            round(move["arrival"], 5)
        )

        if key not in seen:
            picked.append(move)
            seen.add(key)

    return picked

def find_second(child31):
    current = child31["route"][-1]
    day = float(child31["current_time"])
    used = int(child31["special_used"])
    visited = set(child31["route"])

    found = []
    near = []
    minimum_dv = float("inf")
    tested = 0

    for target in range(P.n_tomatoes):
        if target in visited:
            continue

        best_normal = None
        best_special = None
        target_minimum = None

        for wait in WAIT2:
            for tof in TOF2:
                check_time()

                if day + wait + tof > P.max_time:
                    continue

                depart = day + wait

                try:
                    dv = float(P.compute_transfer(
                        current, target, depart, tof
                    ))
                    tested += 1
                except Exception:
                    continue

                if not math.isfinite(dv):
                    continue

                minimum_dv = min(minimum_dv, dv)

                if (
                    target_minimum is None
                    or dv < target_minimum["dv"]
                ):
                    target_minimum = {
                        "dv": dv,
                        "target": target,
                        "wait": wait,
                        "tof": tof
                    }

                move = evaluate_move(
                    current, target, depart, tof, used
                )

                if move is None:
                    continue

                if move["status"] == "NORMAL":
                    if (
                        best_normal is None
                        or move["arrival"]
                        < best_normal["arrival"]
                    ):
                        best_normal = move

                elif (
                    best_special is None
                    or move["arrival"]
                    < best_special["arrival"]
                ):
                    best_special = move

        if best_normal is not None:
            found.append(best_normal)

        if best_special is not None:
            found.append(best_special)

        if (
            best_normal is None
            and target_minimum is not None
            and 100 <= target_minimum["dv"] < 160
        ):
            near.append(target_minimum)

    # Optimizim lokal vetem per rastet
    # qe ishin afer kufirit NORMAL.
    for entry in sorted(
        near,
        key=lambda x: x["dv"]
    )[:1]:
        check_time()

        target = entry["target"]
        max_window = P.max_time - day

        def objective(x):
            wait = float(x[0])
            tof = float(x[1])

            if (
                wait < 0
                or tof < 0.2
                or wait + tof > max_window
            ):
                return 1e8

            try:
                dv = float(P.compute_transfer(
                    current, target, day + wait, tof
                ))
            except Exception:
                return 1e8

            return dv if math.isfinite(dv) else 1e8

        try:
            result = minimize(
                objective,
                [entry["wait"], entry["tof"]],
                method="Nelder-Mead",
                options={
                    "maxfev": 40,
                    "maxiter": 30
                }
            )

            wait = float(result.x[0])
            tof = float(result.x[1])

            if (
                wait >= 0
                and tof >= 0.2
                and wait + tof <= max_window
            ):
                move = evaluate_move(
                    current, target,
                    day + wait, tof, used
                )

                if (
                    move is not None
                    and move["status"] == "NORMAL"
                ):
                    found.append(move)

        except Exception:
            pass

    found.sort(
        key=lambda m: (
            m["status"] == "SPECIAL",
            m["arrival"],
            m["dv"]
        )
    )

    return found[:4], {
        "tested": tested,
        "minimum_dv": (
            minimum_dv
            if math.isfinite(minimum_dv)
            else None
        )
    }

print("FAZA 1 - RRUGET FILLESTARE")
print("=" * 65, flush=True)

for name, states in sources.items():
    print(
        f"{name}: {len(states)} rruge me 30 domate",
        flush=True
    )

print("\nFAZA 2 - KERKIMI ME DY TRANSFERIME")
print("=" * 65, flush=True)

paused = False

try:
    for name in ["A14", "B12"]:
        states = sources[name]
        lane = report["lanes"][name]

        print(
            f"\nSTRATEGJIA {name}",
            flush=True
        )

        for index, state in enumerate(states, 1):
            check_time()

            sid = signature(state)

            if sid not in lane:
                moves = first_moves(state)

                lane[sid] = {
                    "parent_day": state["current_time"],
                    "parent_special": state["special_used"],
                    "first_moves": moves,
                    "checked": {},
                    "complete": False
                }

                save()

            parent_report = lane[sid]

            if parent_report["complete"]:
                continue

            moves = parent_report["first_moves"]

            for move_index, move in enumerate(moves):
                check_time()

                key = str(move_index)

                if key in parent_report["checked"]:
                    continue

                child31 = make_child(state, move)

                second_moves, info = find_second(child31)

                children32 = []

                for next_move in second_moves:
                    child32 = make_child(
                        child31, next_move
                    )

                    if verify(child32):
                        children32.append(child32)

                parent_report["checked"][key] = {
                    "first_move": move,
                    "second_options": len(second_moves),
                    "children32": children32,
                    "info": info
                }

                save()

                print(
                    f"[{index}/{len(states)}] "
                    f"30 -> {move['to']} "
                    f"-> 32 | "
                    f"Status {move['status']} | "
                    f"Dita31 {move['arrival']:.2f} | "
                    f"Femije32 {len(children32)}",
                    flush=True
                )

            parent_report["complete"] = (
                len(parent_report["checked"])
                == len(moves)
            )

            save()

except PauseSearch:
    paused = True
    print(
        "\nKufiri i punes u arrit. "
        "Checkpoint u ruajt.",
        flush=True
    )

# FAZA 3 - Bashkojme te gjitha rruget
# reale me 32 domate qe u gjeten.
all_children = {}

for lane in report["lanes"].values():
    for parent in lane.values():
        for checked in parent["checked"].values():
            for child in checked["children32"]:
                key = signature(child)
                all_children[key] = child

frontier = sorted(
    all_children.values(),
    key=lambda s: (
        s["special_used"],
        s["current_time"]
    )
)

if frontier:
    temp = SNAPSHOT.with_suffix(".tmp")

    temp.write_text(
        json.dumps({
            "level": 32,
            "verified": True,
            "frontier": frontier
        }, separators=(",", ":"))
    )

    temp.replace(SNAPSHOT)

print("\nREZULTATI I HAPIT 123")
print("=" * 65, flush=True)

for name, states in sources.items():
    lane = report["lanes"][name]

    completed = sum(
        parent["complete"]
        for parent in lane.values()
    )

    print(
        f"{name}: "
        f"{completed}/{len(states)} "
        f"prinder te perfunduar",
        flush=True
    )

print(
    "Rruge te verifikuara me 32 domate:",
    len(frontier),
    flush=True
)

for used in range(6):
    group = [
        s for s in frontier
        if s["special_used"] == used
    ]

    if group:
        print(
            f"SPECIAL {used}/5 | "
            f"{len(group)} rruge | "
            f"Dita minimale "
            f"{min(s['current_time'] for s in group):.3f}",
            flush=True
        )

if frontier:
    print("KANDIDAT PER REKORD TE RI: 32/49")
else:
    print(
        "Nuk eshte gjetur ende rruge "
        "me 32 domate nga alternativat e testuara."
    )

print("Kerkim i pjesshem:", paused, flush=True)
print("\nHAPI 123 PERFUNDOI!", flush=True)
