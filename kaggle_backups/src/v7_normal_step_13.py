
import json
import math
from pathlib import Path
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_v7_level_12.json"
ANALYSIS = RESULTS / "v7_next_moves_analysis.json"
OUTPUT = RESULTS / "beam_v7_normal_level_13.json"

problem = TomatoProblem(str(ROOT / "data/easy.kttsp"))

beam = json.loads(SOURCE.read_text())["beam"]
cases = json.loads(ANALYSIS.read_text())["cases"]

# Lidhim analizen me rrugen origjinale
lookup = {
    (
        tuple(s["route"]),
        float(s["current_time"]),
        int(s["special_used"])
    ): s
    for s in beam
}

new_states = []

print("V7 - NGA 12 NE 13 DOMATE")
print("=" * 50)

for i, case in enumerate(cases, 1):

    key = (
        tuple(case["route"]),
        float(case["time"]),
        int(case["special_used"])
    )

    parent = lookup.get(key)

    if parent is None:
        print(f"Rruga {i}: nuk u gjet prindi")
        continue

    current = parent["route"][-1]
    now = float(parent["current_time"])
    used = int(parent["special_used"])

    print(f"\nRRUGA {i}")
    print("Domatja aktuale:", current)
    print("Speciale:", used)

    added = 0

    for c in case["normal_options"]:

        target = int(c["target"])
        wait = float(c["wait"])
        tof = float(c["tof"])

        start = now + wait
        arrival = start + tof

        if target in parent["route"]:
            continue

        if tof < problem.min_tof:
            continue

        if arrival > problem.max_time:
            continue

        # Rillogarisim DV me PyKEP
        try:
            dv = float(problem.compute_transfer(
                current, target, start, tof
            ))
        except Exception:
            continue

        if not math.isfinite(dv):
            continue

        # Lejohen vetem transferime NORMALE
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
            "transfers": parent["transfers"] + [transfer],
            "current_time": arrival,
            "special_used": used
        }

        assert len(child["route"]) == 13
        assert len(set(child["route"])) == 13
        assert len(child["transfers"]) == 12

        new_states.append(child)
        added += 1

        print(
            f"  {current} -> {target} | "
            f"Koha totale: {arrival:.1f} | "
            f"DV: {dv:.2f} | NORMAL"
        )

    print("Vazhdime normale te konfirmuara:", added)


# Heqim dublikatat identike
unique = {}

for state in new_states:
    key = (
        tuple(state["route"]),
        round(state["current_time"], 6),
        state["special_used"]
    )
    unique[key] = state

selected = sorted(
    unique.values(),
    key=lambda s: (
        s["special_used"],
        s["current_time"]
    )
)

if selected:
    result = {
        "method": "V7 Normal-only continuation",
        "source": SOURCE.name,
        "beam": selected,
        "completed": False
    }

    if not OUTPUT.exists():
        temp = OUTPUT.with_suffix(".tmp")
        temp.write_text(json.dumps(result, indent=2))
        temp.replace(OUTPUT)

    print("\nPERMBLEDHJA E V7 - 13 DOMATE")
    print("=" * 50)
    print("Domate: 13/49")
    print("Rruge aktive:", len(selected))
    print("Speciale minimale:", min(
        s["special_used"] for s in selected
    ))
    print("Koha minimale:", min(
        s["current_time"] for s in selected
    ))
    print("Checkpoint:", OUTPUT.name)
    print("V7 NIVELI 13 PERFUNDOI!")
else:
    print("\nNuk u konfirmua asnje vazhdim normal.")
    print("Checkpoint i ri nuk u krijua.")

print("Checkpoint-at e vjeter nuk u ndryshuan.")
