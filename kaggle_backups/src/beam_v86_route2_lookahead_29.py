
import json
import math
from pathlib import Path
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

BEAM27 = RESULTS / "beam_v84_diverse_level_27.json"
FINE27 = RESULTS / "beam_v86_fine_summary_27.json"

LEVEL28 = RESULTS / "beam_v86_route2_level_28.json"
CACHE29 = RESULTS / "beam_v86_route2_probe_29.json"

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8"
    )
    temporary.replace(path)

problem = TomatoProblem(
    str(ROOT / "data/easy.kttsp")
)

# 1. Gjejme rrugen 2
report = load(FINE27)["results"]

row = next(
    r for r in report
    if r["route_index"] == 2
)

move = row["best_next"]

assert move is not None
assert move["status"] == "NORMAL"
assert move["target"] == 40
assert move["dv"] < problem.dv_threshold

parents = load(BEAM27)["beam"]

matching = [
    s for s in parents
    if s["route"] == row["route"]
    and s["special_used"] == 5
    and abs(
        s["current_time"] - row["current_time"]
    ) < 1e-6
]

assert matching, "Nuk u gjet rruga 2 ne nivelin 27."

parent = matching[0]

# 2. Krijojme rrugen per nivelin 28
transfer = {
    "from": parent["route"][-1],
    "to": move["target"],
    "start_time": (
        parent["current_time"] + move["wait"]
    ),
    "wait": move["wait"],
    "tof": move["tof"],
    "dv": move["dv"],
    "status": "NORMAL"
}

assert abs(
    transfer["start_time"]
    + transfer["tof"]
    - move["arrival"]
) < 1e-6

child = dict(parent)

child["route"] = parent["route"] + [40]
child["transfers"] = parent["transfers"] + [transfer]
child["current_time"] = move["arrival"]
child["special_used"] = 5

assert len(child["route"]) == 28
assert len(set(child["route"])) == 28
assert len(child["transfers"]) == 27

if LEVEL28.exists():
    old = load(LEVEL28)["beam"][0]
    assert old["route"] == child["route"]
    assert abs(old["current_time"] - child["current_time"]) < 1e-6
else:
    save(LEVEL28, {
        "method": "V8.6 Route 2",
        "beam": [child],
        "completed": False
    })

print("HAPI 47 - KONTROLLI I NIVELIT 29")
print("=" * 55)
print("Domate te vizituara:", len(child["route"]))
print("Koha:", child["current_time"])
print("Speciale:", child["special_used"])
print("Domatja aktuale:", child["route"][-1])
print("Dite te mbetura:", 200-child["current_time"])
print("=" * 55, flush=True)

# 3. Kerkojme transferime NORMALE
# drejt domates se 29-te

WAITS = sorted(set(
    [i / 4 for i in range(17)] +
    [i / 2 for i in range(9, 17)] +
    [10.0, 13.0]
))

TOFS = sorted(set(
    [i / 4 for i in range(1, 21)] +
    [i / 2 for i in range(11, 21)] +
    [13.0, 21.0, 34.0]
))

if CACHE29.exists():
    cache = load(CACHE29)
    assert cache["route"] == child["route"]
else:
    cache = {
        "route": child["route"],
        "current_time": child["current_time"],
        "targets": {}
    }

now = child["current_time"]
current = child["route"][-1]

remaining_targets = [
    target for target in range(problem.n_tomatoes)
    if target not in child["route"]
]

for target in remaining_targets:

    key = str(target)

    if key in cache["targets"]:
        continue

    best = None
    min_dv = None
    evaluations = 0
    errors = 0

    for wait in WAITS:
        start = now + wait

        for tof in TOFS:

            arrival = start + tof

            if arrival > problem.max_time:
                continue

            try:
                dv = float(
                    problem.compute_transfer(
                        current,
                        target,
                        start,
                        tof
                    )
                )
                evaluations += 1

            except Exception:
                errors += 1
                continue

            if not math.isfinite(dv) or dv < 0:
                continue

            if min_dv is None or dv < min_dv:
                min_dv = dv

            # 5/5 speciale jane perdorur.
            # Lejohen vetem transferime NORMAL.
            if dv >= problem.dv_threshold:
                continue

            candidate = {
                "target": target,
                "wait": wait,
                "tof": tof,
                "arrival": arrival,
                "dv": dv,
                "status": "NORMAL"
            }

            if (
                best is None
                or (arrival, dv)
                < (best["arrival"], best["dv"])
            ):
                best = candidate

    cache["targets"][key] = {
        "best_normal": best,
        "min_dv": min_dv,
        "evaluations": evaluations,
        "errors": errors
    }

    # Ruajme progresin pas cdo domateje
    save(CACHE29, cache)

    if best is not None:
        print(
            f"Domatja {target}: NORMAL | "
            f"Arritja {best['arrival']:.2f} | "
            f"DeltaV {best['dv']:.2f}",
            flush=True
        )

# 4. Analiza
options = [
    item["best_normal"]
    for item in cache["targets"].values()
    if item["best_normal"] is not None
]

options.sort(
    key=lambda m: (m["arrival"], m["dv"])
)

print("\nREZULTATI - RRUGA 2 DREJT NIVELIT 29")
print("=" * 55)

print("Domate aktualisht:", 28)
print("Domate te mbetura:", 21)
print("Koha aktuale:", now)
print("Domate NORMAL te arritshme:", len(options))

if options:

    print("\nALTERNATIVAT ME TE SHPEJTA")

    for m in options[:10]:

        remaining_days = 200 - m["arrival"]

        print(
            f"Domatja {m['target']} | "
            f"Arritja {m['arrival']:.2f} | "
            f"DeltaV {m['dv']:.2f} | "
            f"Mbeten {remaining_days:.2f} dite"
        )

    best = options[0]

    print("\nRRUGA KANDIDATE PER NIVELIN 29")
    print("Domatja:", best["target"])
    print("Koha:", best["arrival"])
    print("Speciale:", 5)
    print("Dite per 20 domatet tjera:",
          round((200-best["arrival"])/20, 3))

else:
    dvs = [
        v["min_dv"]
        for v in cache["targets"].values()
        if v["min_dv"] is not None
    ]

    print("Nuk u gjet transferim NORMAL.")

    if dvs:
        print(
            "DeltaV minimale e testuar:",
            round(min(dvs), 3)
        )

print("\nCheckpoint:", LEVEL28.name)
print("Raporti:", CACHE29.name)
print("HAPI 47 PERFUNDOI!")
