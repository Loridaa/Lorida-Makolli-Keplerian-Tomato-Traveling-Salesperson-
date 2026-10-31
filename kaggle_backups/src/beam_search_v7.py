
import json
import math
from collections import defaultdict
from pathlib import Path
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_v5_level_11.json"
OUTPUT = RESULTS / "beam_v7_level_12.json"
CACHE = RESULTS / "beam_v7_cache_11.json"

WIDTH = 40

WAITS = [0, 0.5, 1, 2, 3, 5, 8, 13]
FLIGHTS = [0.5, 1, 1.5, 2, 3, 5, 8, 13, 21, 34]

problem = TomatoProblem(str(ROOT / "data/easy.kttsp"))


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


if OUTPUT.exists():
    old = json.loads(OUTPUT.read_text())
    print("V7 ekziston tashme.")
    print("Rruge:", len(old["beam"]))
    print("Checkpoint:", OUTPUT.name)
    raise SystemExit(0)


beam = json.loads(SOURCE.read_text())["beam"]

if not beam or any(len(s["route"]) != 11 for s in beam):
    raise ValueError("Pritet checkpoint me 11 domate.")

cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}


def scan(state):
    current = state["route"][-1]
    now = float(state["current_time"])
    used = int(state["special_used"])
    visited = set(state["route"])

    groups = defaultdict(list)
    evaluations = 0

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
                    evaluations += 1
                except Exception:
                    continue

                if not math.isfinite(dv):
                    continue

                if dv < problem.dv_threshold:
                    status = "NORMAL"

                elif (
                    dv < problem.exception_threshold
                    and used < problem.n_exceptions
                ):
                    status = "SPECIAL"

                else:
                    continue

                groups[(target, status)].append({
                    "target": target,
                    "wait": wait,
                    "tof": tof,
                    "arrival": arrival,
                    "dv": dv,
                    "status": status
                })

    primary = {"NORMAL": [], "SPECIAL": []}
    secondary = {"NORMAL": [], "SPECIAL": []}

    for (target, status), options in groups.items():
        options.sort(
            key=lambda c: (c["arrival"], c["dv"])
        )

        first = options[0]
        primary[status].append(first)

        # Mbajme edhe nje mberritje me vone,
        # sepse mund te kete vazhdime te tjera.
        for candidate in options[1:]:
            difference = (
                candidate["arrival"] - first["arrival"]
            )

            if 2 <= difference <= 20:
                secondary[status].append(candidate)
                break

    for status in primary:
        primary[status].sort(
            key=lambda c: (c["arrival"], c["dv"])
        )
        secondary[status].sort(
            key=lambda c: (c["arrival"], c["dv"])
        )

    # Se pari domate te ndryshme,
    # pastaj disa kohe alternative.
    moves = (
        primary["NORMAL"][:12]
        + secondary["NORMAL"][:5]
        + primary["SPECIAL"][:8]
        + secondary["SPECIAL"][:3]
    )

    return moves, evaluations, {
        "normal_targets": len(primary["NORMAL"]),
        "special_targets": len(primary["SPECIAL"])
    }


def make_child(state, c):
    status = c["status"]
    target = int(c["target"])
    now = float(state["current_time"])

    transfer = {
        "from": state["route"][-1],
        "to": target,
        "start_time": now + c["wait"],
        "wait": c["wait"],
        "tof": c["tof"],
        "dv": c["dv"],
        "status": status
    }

    return {
        "route": state["route"] + [target],
        "transfers": state["transfers"] + [transfer],
        "current_time": c["arrival"],
        "special_used": (
            state["special_used"]
            + int(status == "SPECIAL")
        )
    }


def rank(state):
    return (
        state["special_used"],
        state["current_time"]
    )


print("BEAM SEARCH V7 - DEGEZIM ALTERNATIV")
print("=" * 50, flush=True)
print("Nisja: 11 domate")
print("Beam width:", WIDTH)
print("Rruge fillestare:", len(beam), flush=True)

expanded = []
total = 0

for i, state in enumerate(beam):
    key = json.dumps([
        state["route"],
        state["current_time"],
        state["special_used"]
    ])

    if key in cache:
        item = cache[key]
        moves = item["moves"]
        stats = item["stats"]
        origin = "CACHE"
    else:
        moves, count, stats = scan(state)
        total += count

        cache[key] = {
            "moves": moves,
            "stats": stats
        }
        save(CACHE, cache)

        origin = "KERKIM"

    for c in moves:
        expanded.append((i, make_child(state, c)))

    print(
        f"Prindi {i+1}/{len(beam)}: "
        f"{stats['normal_targets']} normale, "
        f"{stats['special_targets']} speciale | "
        f"{len(moves)} kandidate | {origin}",
        flush=True
    )


if not expanded:
    raise RuntimeError("Nuk u gjet asnje vazhdim.")


expanded.sort(key=lambda x: rank(x[1]))

selected = []
seen = set()
parent_counts = defaultdict(int)


def add(pid, state):
    key = (
        tuple(state["route"]),
        round(state["current_time"], 6),
        state["special_used"]
    )

    if key in seen:
        return False

    seen.add(key)
    selected.append(state)
    parent_counts[pid] += 1
    return True


# Fillimisht nga nje rruge per secilin prind
for pid, state in expanded:
    if parent_counts[pid] == 0:
        add(pid, state)


# Pastaj ruajme alternativa, maksimumi 4 per prind
for pid, state in expanded:
    if len(selected) >= WIDTH:
        break

    if parent_counts[pid] < 4:
        add(pid, state)


# Kontrollojme rezultatet
for s in selected:
    assert len(s["route"]) == 12
    assert len(set(s["route"])) == 12
    assert len(s["transfers"]) == 11
    assert s["special_used"] <= problem.n_exceptions
    assert s["current_time"] <= problem.max_time

    assert s["special_used"] == sum(
        t["status"] == "SPECIAL"
        for t in s["transfers"]
    )


result = {
    "method": "Beam Search V7 Alternative Branching",
    "source": SOURCE.name,
    "beam_width": WIDTH,
    "beam": selected,
    "evaluations": total,
    "completed": False
}

save(OUTPUT, result)

print("\nPERMBLEDHJA E V7")
print("=" * 50)
print("Domate: 12/49")
print("Rruge te ruajtura:", len(selected))

for n in range(6):
    count = sum(
        s["special_used"] == n for s in selected
    )
    if count:
        print(f"Rruge me {n} speciale: {count}")

print("Speciale minimale:", min(
    s["special_used"] for s in selected
))
print("Koha minimale:", min(
    s["current_time"] for s in selected
))
print("Llogaritje:", total)
print("Checkpoint:", OUTPUT.name)
print("V7 PERFUNDOI!")
