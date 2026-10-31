
import json
import math
from pathlib import Path
from collections import defaultdict
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_v8_diverse_level_22.json"
OUTPUT = RESULTS / "v8_lookahead_level_22.json"

problem = TomatoProblem(str(ROOT / "data/easy.kttsp"))

WAITS = [0, 0.5, 1, 2, 3, 5, 8, 13]
FLIGHTS = [0.5, 1, 1.5, 2, 3, 5, 8, 13, 21, 34]

# Sa rruge do te provojme nga secila kategori
QUOTAS = {2: 1, 3: 6, 4: 7, 5: 2}

def save(data):
    temp = OUTPUT.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(OUTPUT)

beam = json.loads(SOURCE.read_text())["beam"]
groups = defaultdict(list)

for state in beam:
    groups[state["special_used"]].append(state)

# Zgjedhim kohe te ndryshme, jo vetem
# rrugen me te shpejte ne secilin grup
selected = []

for used, quota in QUOTAS.items():
    states = sorted(
        groups.get(used, []),
        key=lambda s: s["current_time"]
    )

    n = min(quota, len(states))

    if n == 0:
        continue

    if n == 1:
        indices = [0]
    else:
        indices = [
            round(i * (len(states)-1) / (n-1))
            for i in range(n)
        ]

    for idx in indices:
        selected.append(states[idx])

def state_key(state):
    return json.dumps([
        state["route"],
        state["current_time"],
        state["special_used"]
    ])

if OUTPUT.exists():
    data = json.loads(OUTPUT.read_text())
else:
    data = {
        "method": "V8 Lookahead Level 22",
        "source": SOURCE.name,
        "results": {},
        "completed": False
    }

records = data["results"]

print("V8 - LOOKAHEAD NIVELI 22")
print("=" * 55)
print("Rruge per testim:", len(selected))
print("Rruge te analizuara me pare:", len(records))
print(flush=True)

for i, state in enumerate(selected, 1):

    key = state_key(state)

    if key in records:
        record = records[key]
        print(
            f"Rruga {i}: CACHE | "
            f"Normale {record['normal_count']}",
            flush=True
        )
        continue

    current = state["route"][-1]
    now = float(state["current_time"])
    used = int(state["special_used"])
    visited = set(state["route"])

    normal = {}
    special = {}
    calculations = 0

    for target in range(problem.n_tomatoes):

        if target in visited:
            continue

        for wait in WAITS:
            for tof in FLIGHTS:

                start = now + wait
                arrival = start + tof

                if tof < getattr(problem, "min_tof", 0):
                    continue

                if arrival > problem.max_time:
                    continue

                try:
                    dv = float(problem.compute_transfer(
                        current, target, start, tof
                    ))
                    calculations += 1
                except Exception:
                    continue

                if not math.isfinite(dv):
                    continue

                if dv < problem.dv_threshold:
                    destination = normal

                elif (
                    dv < problem.exception_threshold
                    and used < problem.n_exceptions
                ):
                    destination = special

                else:
                    continue

                candidate = {
                    "target": target,
                    "wait": wait,
                    "tof": tof,
                    "arrival": arrival,
                    "dv": dv
                }

                # Per secilen domate ruajme
                # transferimin me mberritjen me te hershme
                if (
                    target not in destination
                    or arrival < destination[target]["arrival"]
                ):
                    destination[target] = candidate

    best_normal = sorted(
        normal.values(),
        key=lambda x: (x["arrival"], x["dv"])
    )

    best_special = sorted(
        special.values(),
        key=lambda x: (x["arrival"], x["dv"])
    )

    record = {
        "route": state["route"],
        "current_time": now,
        "special_used": used,
        "last_tomato": current,
        "normal_count": len(normal),
        "special_count": len(special),
        "best_normal": best_normal[:3],
        "best_special": best_special[:3],
        "calculations": calculations
    }

    records[key] = record

    # Ruajtje pas seciles rruge
    save(data)

    print(
        f"Rruga {i}/{len(selected)} | "
        f"Speciale {used} | "
        f"Koha {now:.1f} | "
        f"Normale {len(normal)} | "
        f"Speciale {len(special)}",
        flush=True
    )

data["completed"] = True
save(data)

all_records = list(records.values())

print("\nPERMBLEDHJA E V8 LOOKAHEAD")
print("=" * 55)
print("Domate: 22/49")
print("Rruge te testuara:", len(all_records))

print("\nMUNDESITE SIPAS SPECIALEVE")

for used in sorted({r["special_used"] for r in all_records}):
    subset = [
        r for r in all_records
        if r["special_used"] == used
    ]

    with_normals = sum(
        r["normal_count"] > 0 for r in subset
    )

    max_normals = max(
        r["normal_count"] for r in subset
    )

    print(
        f"{used} speciale: "
        f"{len(subset)} rruge | "
        f"{with_normals} me vazhdim normal | "
        f"maksimumi {max_normals} domate normale"
    )

print("\n5 RRUGET ME ME SHUME VAZHDIME NORMALE")

ranked = sorted(
    all_records,
    key=lambda r: (
        -r["normal_count"],
        r["special_used"],
        r["current_time"]
    )
)

for i, r in enumerate(ranked[:5], 1):
    print(f"\nRruga {i}")
    print("Koha:", r["current_time"])
    print("Speciale:", r["special_used"])
    print("Domatja e fundit:", r["last_tomato"])
    print("Domate normale te arritshme:",
          r["normal_count"])
    print("Domate speciale te arritshme:",
          r["special_count"])

    if r["best_normal"]:
        move = r["best_normal"][0]
        print(
            "Vazhdimi normal me i hershem:",
            r["last_tomato"], "->", move["target"],
            "| Mberritja:", move["arrival"],
            "| DV:", round(move["dv"], 2)
        )

print("\nCheckpoint:", OUTPUT.name)
print("V8 LOOKAHEAD PERFUNDOI!")
