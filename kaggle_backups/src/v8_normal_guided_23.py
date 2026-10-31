
import json
import math
from pathlib import Path
from collections import Counter
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_v8_diverse_level_22.json"
LOOKAHEAD = RESULTS / "v8_lookahead_level_22.json"
OUTPUT = RESULTS / "beam_v8_normal_guided_level_23.json"

problem = TomatoProblem(str(ROOT / "data/easy.kttsp"))

def save(data):
    temp = OUTPUT.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(OUTPUT)

if not SOURCE.exists() or not LOOKAHEAD.exists():
    raise FileNotFoundError("Mungon checkpoint-i ose analiza.")

parents = json.loads(SOURCE.read_text())["beam"]
records = json.loads(LOOKAHEAD.read_text())["results"]

lookup = {
    (
        tuple(s["route"]),
        round(float(s["current_time"]), 6),
        int(s["special_used"])
    ): s
    for s in parents
}

print("V8.1 - VAZHDIME NORMALE")
print("=" * 50)
print("Nisja: 22 domate")
print("Rruge te analizuara:", len(records))

if OUTPUT.exists():
    selected = json.loads(OUTPUT.read_text())["beam"]
    print("Checkpoint ekziston. Nuk u mbishkrua.")
else:
    candidates = []

    for i, r in enumerate(records.values(), 1):
        key = (
            tuple(r["route"]),
            round(float(r["current_time"]), 6),
            int(r["special_used"])
        )

        parent = lookup.get(key)

        if parent is None:
            raise ValueError(
                f"Prindi {i} nuk u gjet ne nivelin 22."
            )

        current = parent["route"][-1]
        now = float(parent["current_time"])
        used = int(parent["special_used"])

        added = 0

        for move in r["best_normal"]:
            target = int(move["target"])
            wait = float(move["wait"])
            tof = float(move["tof"])

            start = now + wait
            arrival = start + tof

            if target in parent["route"]:
                continue

            if tof < getattr(problem, "min_tof", 0):
                continue

            if arrival > problem.max_time:
                continue

            # Konfirmim i ri me PyKEP
            dv = float(problem.compute_transfer(
                current, target, start, tof
            ))

            if not math.isfinite(dv):
                continue

            if dv >= problem.dv_threshold:
                continue

            transfer = {
                "from": current,
                "to": target,
                "start_time": start,
                "wait": wait,
                "tof": tof,
                "dv": dv,
                "status": "NORMAL"
            }

            child = {
                "route": parent["route"] + [target],
                "transfers": (
                    parent["transfers"] + [transfer]
                ),
                "current_time": arrival,
                "special_used": used
            }

            assert len(child["route"]) == 23
            assert len(set(child["route"])) == 23
            assert len(child["transfers"]) == 22

            candidates.append(child)
            added += 1

        print(
            f"Rruga {i}/{len(records)} | "
            f"{used} speciale | "
            f"{added} vazhdime normale",
            flush=True
        )

    # Heqim dublikatat identike
    unique = {}

    for s in candidates:
        key = (
            tuple(s["route"]),
            round(s["current_time"], 6),
            s["special_used"]
        )
        unique[key] = s

    selected = sorted(
        unique.values(),
        key=lambda s: (
            s["special_used"],
            s["current_time"]
        )
    )

    result = {
        "method": "V8.1 Normal Guided",
        "source": SOURCE.name,
        "lookahead": LOOKAHEAD.name,
        "beam": selected,
        "completed": False
    }

    save(result)

print("\nPERMBLEDHJA E V8.1")
print("=" * 50)

print("Domate: 23/49")
print("Rruge te krijuara:", len(selected))
print("Speciale te reja te shpenzuara: 0")

if selected:
    counts = Counter(
        s["special_used"] for s in selected
    )

    print("Bashkesi te ndryshme:", len({
        frozenset(s["route"]) for s in selected
    }))

    print("\nKOHA SIPAS SPECIALEVE")

    for used in sorted(counts):
        best = min(
            s["current_time"]
            for s in selected
            if s["special_used"] == used
        )

        print(
            f"{used} speciale: {counts[used]} rruge | "
            f"koha {best:.1f} dite | "
            f"mbeten {200-best:.1f} dite"
        )

    print("\n3 RRUGET ME TE SHPEJTA")

    for s in sorted(
        selected,
        key=lambda x: x["current_time"]
    )[:3]:
        t = s["transfers"][-1]

        print(
            f"{t['from']} -> {t['to']} | "
            f"koha {s['current_time']:.1f} | "
            f"speciale {s['special_used']} | "
            f"DV {t['dv']:.2f}"
        )
else:
    print("Nuk u konfirmua vazhdim normal.")

print("\nCheckpoint:", OUTPUT.name)
print("V8.1 PERFUNDOI!")
