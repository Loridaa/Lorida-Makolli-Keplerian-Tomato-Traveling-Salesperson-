import json
import math
import shutil
from pathlib import Path
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_search_v4_progress.json"
OUTPUT = RESULTS / "beam_search_v5_progress.json"
CACHE = RESULTS / "beam_v5_rebuild_cache.json"

WAITS = [0, 1, 2, 3, 5, 8, 13, 21, 34]
FLIGHTS = [0.5, 1, 2, 3, 5, 8, 13, 21, 34]

WIDTH = 14

problem = TomatoProblem(str(ROOT / "data/easy.kttsp"))


def save_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def create_backup():
    return shutil.make_archive(
        "/kaggle/working/esa_spoc4_backup_v5",
        "zip",
        ROOT
    )


if OUTPUT.exists():
    saved = json.loads(OUTPUT.read_text())
    print("V5 ekziston tashme.")
    print("Rruge:", len(saved["beam"]))
    print("Domate:", len(saved["beam"][0]["route"]))
    print("Backup:", create_backup())
    raise SystemExit(0)


source_data = json.loads(SOURCE.read_text())
beam = source_data["beam"]

if len(beam) != 10:
    raise ValueError("Priten 10 rruge nga V4.")

if any(len(s["route"]) != 8 for s in beam):
    raise ValueError("Priten rruge me 8 domate.")


# Ruajme llogaritjet pas seciles rruge
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}


def scan(state):
    route = state["route"]
    current = route[-1]
    now = float(state["current_time"])
    used = int(state["special_used"])

    normal = {}
    special = {}
    evaluations = 0

    for target in range(problem.n_tomatoes):
        if target in route:
            continue

        for wait in WAITS:
            for tof in FLIGHTS:

                if tof < problem.min_tof:
                    continue

                departure = now + wait
                arrival = departure + tof

                if arrival > problem.max_time:
                    continue

                try:
                    dv = float(problem.compute_transfer(
                        current, target, departure, tof
                    ))
                    evaluations += 1
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

    return (
        sorted(
            normal.values(),
            key=lambda c: (c["arrival"], c["dv"])
        ),
        sorted(
            special.values(),
            key=lambda c: (c["arrival"], c["dv"])
        ),
        evaluations
    )


def make_child(state, candidate, status):
    now = float(state["current_time"])
    target = int(candidate["target"])
    wait = float(candidate["wait"])
    tof = float(candidate["tof"])

    transfer = {
        "from": state["route"][-1],
        "to": target,
        "start_time": now + wait,
        "wait": wait,
        "tof": tof,
        "dv": float(candidate["dv"]),
        "status": status
    }

    return {
        "route": state["route"] + [target],
        "transfers": state["transfers"] + [transfer],
        "current_time": float(candidate["arrival"]),
        "special_used": (
            int(state["special_used"])
            + int(status == "SPECIAL")
        )
    }


def rank(state):
    return (
        state["current_time"] + 12 * state["special_used"],
        state["special_used"],
        state["current_time"]
    )


print("RIKTHIMI I BEAM SEARCH V5")
print("=" * 45)
print("Rruge fillestare:", len(beam))
print("Domate fillestare: 8")
print("Beam width:", WIDTH, flush=True)

expanded = []
total_evaluations = 0

for i, state in enumerate(beam):

    key = json.dumps([
        state["route"],
        state["current_time"],
        state["special_used"]
    ])

    if key in cache:
        item = cache[key]
        normal = item["normal"]
        special = item["special"]
        count = 0
        origin = "CACHE"
    else:
        normal, special, count = scan(state)

        cache[key] = {
            "normal": normal,
            "special": special
        }

        save_json(CACHE, cache)
        origin = "KERKIM"

    total_evaluations += count

    candidates = (
        [("NORMAL", c) for c in normal[:3]]
        + [("SPECIAL", c) for c in special[:2]]
    )

    for status, candidate in candidates:
        expanded.append((
            i,
            make_child(state, candidate, status)
        ))

    print(
        f"Rruga {i + 1}/10: "
        f"{len(normal)} normale, "
        f"{len(special)} speciale | "
        f"{origin}",
        flush=True
    )


if not expanded:
    raise RuntimeError("Nuk u gjet vazhdim.")


expanded.sort(key=lambda item: rank(item[1]))

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


# Fillimisht nje rruge per secilin prind
for parent_id, state in expanded:
    if parent_id not in represented:
        add(parent_id, state)

    if len(selected) >= WIDTH:
        break


# Pastaj plotesojme deri ne 14 rruge
for parent_id, state in expanded:
    if len(selected) >= WIDTH:
        break

    add(parent_id, state)


# Kontrollet e rezultateve
for state in selected:
    route = state["route"]
    transfers = state["transfers"]

    if len(route) != 9:
        raise ValueError("Numer i gabuar domatesh.")

    if len(route) != len(set(route)):
        raise ValueError("Domate e perseritur.")

    if len(transfers) != len(route) - 1:
        raise ValueError("Transferime te pavlefshme.")

    if state["special_used"] > problem.n_exceptions:
        raise ValueError("Teper manovra speciale.")

    if state["current_time"] > problem.max_time:
        raise ValueError("Koha e tejkaluar.")

    if state["special_used"] != sum(
        t["status"] == "SPECIAL" for t in transfers
    ):
        raise ValueError("Numerim i gabuar i specialeve.")


result = {
    "method": "Beam Search V5",
    "beam_width": WIDTH,
    "beam": selected,
    "evaluations_last_level": total_evaluations,
    "completed": any(
        len(s["route"]) == problem.n_tomatoes
        for s in selected
    )
}

save_json(OUTPUT, result)

print("\nREZULTATET E V5")
print("=" * 45)

for i, state in enumerate(selected, 1):
    print(
        f"Rruga {i}: {state['route']} | "
        f"Koha: {state['current_time']} | "
        f"Speciale: {state['special_used']}"
    )

print("\nPERMBLEDHJA")
print("Domate: 9/49")
print("Rruge aktive:", len(selected))
print("Speciale minimale:", min(
    s["special_used"] for s in selected
))
print("Koha minimale:", min(
    s["current_time"] for s in selected
))
print("Checkpoint:", OUTPUT.name)

backup = create_backup()
print("BACKUP ZIP:", backup)

print("\nRIKTHIMI I V5 PERFUNDOI!")
