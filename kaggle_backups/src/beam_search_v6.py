import json
import math
from pathlib import Path
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_v5_level_12.json"
OUTPUT = RESULTS / "beam_v6_level_13.json"
CACHE = RESULTS / "beam_v6_cache_12.json"

WIDTH = 14

WAITS = [0, 1, 2, 3, 5, 8, 13]
FLIGHTS = [0.5, 1, 2, 3, 5, 8, 13, 21, 34]

FUTURE_WAITS = [0, 1, 3, 8]
FUTURE_FLIGHTS = [0.5, 1, 2, 3, 5, 8, 13]

problem = TomatoProblem(str(ROOT / "data/easy.kttsp"))


def save_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


if OUTPUT.exists():
    print("V6 me 13 domate ekziston tashme.")
    print("Nuk po mbishkruhet.")
    raise SystemExit(0)


beam = json.loads(SOURCE.read_text())["beam"]

if len(beam) != 14:
    raise ValueError("Priten 14 rruge nga V5.")

if any(len(s["route"]) != 12 for s in beam):
    raise ValueError("Priten 12 domate per rruge.")

cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}


def scan_moves(state):
    current = state["route"][-1]
    now = float(state["current_time"])
    used = int(state["special_used"])
    visited = set(state["route"])

    normal = {}
    special = {}

    for target in range(problem.n_tomatoes):
        if target in visited:
            continue

        for wait in WAITS:
            for tof in FLIGHTS:
                arrival = now + wait + tof

                if tof < problem.min_tof:
                    continue

                if arrival > problem.max_time:
                    continue

                try:
                    dv = float(problem.compute_transfer(
                        current, target, now + wait, tof
                    ))
                except Exception:
                    continue

                if not math.isfinite(dv):
                    continue

                candidate = {
                    "target": target,
                    "wait": wait,
                    "tof": tof,
                    "arrival": arrival,
                    "dv": dv
                }

                if dv < problem.dv_threshold:
                    store = normal

                elif (
                    dv < problem.exception_threshold
                    and used < problem.n_exceptions
                ):
                    store = special

                else:
                    continue

                old = store.get(target)

                if old is None or (
                    arrival, dv
                ) < (
                    old["arrival"], old["dv"]
                ):
                    store[target] = candidate

    normals = sorted(
        normal.values(),
        key=lambda c: (c["arrival"], c["dv"])
    )[:3]

    specials = sorted(
        special.values(),
        key=lambda c: (c["arrival"], c["dv"])
    )[:2]

    return (
        [("NORMAL", c) for c in normals]
        + [("SPECIAL", c) for c in specials]
    )


def future_normals(route, current, now):
    visited = set(route)
    count = 0

    for target in range(problem.n_tomatoes):
        if target in visited:
            continue

        found = False

        for wait in FUTURE_WAITS:
            if found:
                break

            for tof in FUTURE_FLIGHTS:
                if tof < problem.min_tof:
                    continue

                if now + wait + tof > problem.max_time:
                    continue

                try:
                    dv = float(problem.compute_transfer(
                        current, target, now + wait, tof
                    ))
                except Exception:
                    continue

                if math.isfinite(dv) and dv < problem.dv_threshold:
                    found = True
                    break

        if found:
            count += 1

    return count


def make_child(state, candidate, status):
    now = float(state["current_time"])
    target = int(candidate["target"])
    route = state["route"] + [target]

    used = (
        int(state["special_used"])
        + int(status == "SPECIAL")
    )

    transfer = {
        "from": state["route"][-1],
        "to": target,
        "start_time": now + candidate["wait"],
        "wait": candidate["wait"],
        "tof": candidate["tof"],
        "dv": candidate["dv"],
        "status": status
    }

    future = future_normals(
        route,
        target,
        candidate["arrival"]
    )

    return {
        "route": route,
        "transfers": state["transfers"] + [transfer],
        "current_time": candidate["arrival"],
        "special_used": used,
        "future_normals": future
    }


def rank(state):
    used = state["special_used"]
    now = state["current_time"]
    future = state["future_normals"]

    # Kandidatet pa speciale dhe pa vazhdim
    # normal marrin prioritet me te ulet.
    likely_stuck = (used == 5 and future == 0)

    # Kriter eksperimental, jo rezultat zyrtar ESA
    score = now + 12 * used - 3 * min(future, 3)

    return (likely_stuck, score, used, now)


print("BEAM SEARCH V6 - LOOKAHEAD")
print("=" * 45)
print("Nisja: 12 domate")
print("Rruge fillestare:", len(beam))
print("Rruge qe do te ruhen:", WIDTH, flush=True)

expanded = []

for i, state in enumerate(beam):
    key = json.dumps([
        state["route"],
        state["current_time"],
        state["special_used"]
    ])

    if key in cache:
        children = cache[key]
        origin = "CACHE"

    else:
        moves = scan_moves(state)

        children = [
            make_child(state, c, status)
            for status, c in moves
        ]

        cache[key] = children
        save_json(CACHE, cache)

        origin = "KERKIM"

    for child in children:
        expanded.append((i, child))

    print(
        f"Rruga {i + 1}/14: "
        f"{len(children)} kandidate | {origin}",
        flush=True
    )


if not expanded:
    raise RuntimeError("Nuk u gjet asnje vazhdim.")


expanded.sort(key=lambda x: rank(x[1]))

selected = []
seen = set()
represented = set()


def add(parent_id, state):
    key = (
        tuple(state["route"]),
        round(state["current_time"], 6),
        state["special_used"]
    )

    if key in seen:
        return False

    selected.append(state)
    seen.add(key)
    represented.add(parent_id)
    return True


# Ruajme larmi nga 8 rruge prind
for parent_id, state in expanded:
    if parent_id not in represented:
        add(parent_id, state)

    if len(represented) >= 8:
        break


# Plotesojme deri ne 14 rruge
for parent_id, state in expanded:
    if len(selected) >= WIDTH:
        break

    add(parent_id, state)


# Validim paraprak dhe rikontroll i
# transferimit te fundit me PyKEP
for state in selected:
    route = state["route"]
    transfers = state["transfers"]
    last = transfers[-1]

    assert len(route) == 13
    assert len(route) == len(set(route))
    assert len(transfers) == 12
    assert state["special_used"] <= problem.n_exceptions
    assert state["current_time"] <= problem.max_time

    assert state["special_used"] == sum(
        t["status"] == "SPECIAL"
        for t in transfers
    )

    dv_check = float(problem.compute_transfer(
        last["from"],
        last["to"],
        last["start_time"],
        last["tof"]
    ))

    assert math.isfinite(dv_check)
    assert abs(dv_check - last["dv"]) < 0.1


result = {
    "method": "Beam Search V6 Lookahead",
    "beam_width": WIDTH,
    "beam": selected,
    "completed": any(
        len(s["route"]) == problem.n_tomatoes
        for s in selected
    )
}

save_json(OUTPUT, result)

print("\nREZULTATET E V6")
print("=" * 45)

for i, state in enumerate(selected, 1):
    print(
        f"Rruga {i}: "
        f"Koha={state['current_time']} | "
        f"Speciale={state['special_used']} | "
        f"Normale ne hapin tjeter={state['future_normals']}"
    )

print("\nPERMBLEDHJA")
print("Domate: 13/49")
print("Rruge aktive:", len(selected))
print("Speciale minimale:", min(
    s["special_used"] for s in selected
))
print("Koha minimale:", min(
    s["current_time"] for s in selected
))
print("Rruge me vazhdime normale:", sum(
    s["future_normals"] > 0 for s in selected
))
print("Checkpoint:", OUTPUT.name)
print("V6 PERFUNDOI!")
