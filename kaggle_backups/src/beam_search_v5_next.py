
import json
import math
from pathlib import Path
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_search_v5_progress.json"
OUTPUT = RESULTS / "beam_search_v5_10.json"
CACHE = RESULTS / "beam_v5_10_cache.json"

WAITS = [0, 1, 2, 3, 5, 8, 13]
FLIGHTS = [0.5, 1, 2, 3, 5, 8, 13, 21, 34]
WIDTH = 14

problem = TomatoProblem(str(ROOT / "data/easy.kttsp"))

def save(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)

if OUTPUT.exists():
    data = json.loads(OUTPUT.read_text())
    print("Rezultati me 10 domate ekziston tashme.")
    print("Rruge:", len(data["beam"]))
    raise SystemExit(0)

data = json.loads(SOURCE.read_text())
beam = data["beam"]

if len(beam) != 14:
    raise ValueError("Priten 14 rruge nga V5.")

if any(len(s["route"]) != 9 for s in beam):
    raise ValueError("Priten 9 domate per rruge.")

cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

def scan(state):
    current = state["route"][-1]
    now = float(state["current_time"])
    used = int(state["special_used"])
    visited = set(state["route"])

    normal = {}
    special = {}
    count = 0

    for target in range(problem.n_tomatoes):
        if target in visited:
            continue

        for wait in WAITS:
            for tof in FLIGHTS:
                departure = now + wait
                arrival = departure + tof

                if tof < problem.min_tof:
                    continue

                if arrival > problem.max_time:
                    continue

                try:
                    dv = float(problem.compute_transfer(
                        current, target, departure, tof
                    ))
                    count += 1
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
                elif (dv < problem.exception_threshold
                      and used < problem.n_exceptions):
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
        sorted(normal.values(),
               key=lambda c: (c["arrival"], c["dv"])),
        sorted(special.values(),
               key=lambda c: (c["arrival"], c["dv"])),
        count
    )

def child(state, candidate, status):
    now = float(state["current_time"])
    target = int(candidate["target"])

    transfer = {
        "from": state["route"][-1],
        "to": target,
        "start_time": now + candidate["wait"],
        "wait": candidate["wait"],
        "tof": candidate["tof"],
        "dv": candidate["dv"],
        "status": status
    }

    return {
        "route": state["route"] + [target],
        "transfers": state["transfers"] + [transfer],
        "current_time": candidate["arrival"],
        "special_used": (
            state["special_used"] + int(status == "SPECIAL")
        )
    }

def rank(state):
    return (
        state["current_time"] + 12 * state["special_used"],
        state["special_used"],
        state["current_time"]
    )

print("BEAM SEARCH V5 - 9 NE 10 DOMATE")
print("=" * 45, flush=True)

expanded = []
total = 0

for i, state in enumerate(beam):
    key = json.dumps([
        state["route"],
        state["current_time"],
        state["special_used"]
    ])

    if key in cache:
        n = cache[key]["normal"]
        s = cache[key]["special"]
        count = 0
    else:
        n, s, count = scan(state)
        cache[key] = {"normal": n, "special": s}
        save(CACHE, cache)

    total += count

    for c in n[:4]:
        expanded.append((i, child(state, c, "NORMAL")))

    for c in s[:3]:
        expanded.append((i, child(state, c, "SPECIAL")))

    print(
        f"Rruga {i+1}/14: "
        f"{len(n)} normale, {len(s)} speciale",
        flush=True
    )

expanded.sort(key=lambda x: rank(x[1]))

selected = []
seen = set()
represented = set()

def add(pid, state):
    key = (
        tuple(state["route"]),
        round(state["current_time"], 6),
        state["special_used"]
    )

    if key in seen:
        return

    selected.append(state)
    seen.add(key)
    represented.add(pid)

# Ruajme larmi nga rruget prind
for pid, state in expanded:
    if pid not in represented:
        add(pid, state)
    if len(selected) >= WIDTH:
        break

# Plotesojme deri ne 14 rruge
for pid, state in expanded:
    if len(selected) >= WIDTH:
        break
    add(pid, state)

if not selected:
    raise RuntimeError("Nuk u gjet asnje vazhdim.")

for state in selected:
    route = state["route"]
    transfers = state["transfers"]

    assert len(route) == 10
    assert len(route) == len(set(route))
    assert len(transfers) == 9
    assert state["special_used"] <= problem.n_exceptions
    assert state["current_time"] <= problem.max_time
    assert state["special_used"] == sum(
        t["status"] == "SPECIAL" for t in transfers
    )

result = {
    "method": "Beam Search V5",
    "beam_width": WIDTH,
    "beam": selected,
    "evaluations_last_level": total,
    "completed": False
}

save(OUTPUT, result)

print("\nPERMBLEDHJA")
print("=" * 45)
print("Domate: 10/49")
print("Rruge aktive:", len(selected))
print("Speciale minimale:",
      min(s["special_used"] for s in selected))
print("Koha minimale:",
      min(s["current_time"] for s in selected))
print("Llogaritje:", total)
print("Checkpoint:", OUTPUT.name)
print("HAPI 10 PERFUNDOI!")
