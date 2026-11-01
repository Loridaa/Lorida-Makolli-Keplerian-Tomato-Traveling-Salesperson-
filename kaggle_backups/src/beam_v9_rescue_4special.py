
import json
import math
import hashlib
import time
from pathlib import Path
from scipy.optimize import differential_evolution
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "beam_v9_rescue_4special.json"

P = TomatoProblem(str(ROOT / "data/easy.kttsp"))

SOURCE_LEVELS = {24: 5, 25: 3, 26: 1}

WAITS = [0, 0.5, 1, 2, 3, 5, 8, 12]
TOFS = [0.5, 1, 1.5, 2, 3, 5, 8, 13, 18]

MAX_EXTRA = 24.0
SEEDS = [2026, 2043]

START = time.monotonic()
LIMIT = 140.0

config = {
    "source_hashes": {},
    "levels": {
        str(k): v for k, v in SOURCE_LEVELS.items()
    },
    "waits": WAITS,
    "tofs": TOFS,
    "max_extra": MAX_EXTRA,
    "seeds": SEEDS,
    "maxiter": 12,
    "popsize": 6,
    "optimize_top": 3
}

def verify(state):
    route = state["route"]
    transfers = state["transfers"]

    if (
        len(route) < 2
        or len(transfers) != len(route) - 1
        or route[0] != 0
        or len(set(route)) != len(route)
    ):
        return False

    current_day = 0.0
    specials = 0

    for i, tr in enumerate(transfers):
        try:
            start = float(tr["start_time"])
            tof = float(tr["tof"])

            if (
                tr["from"] != route[i]
                or tr["to"] != route[i + 1]
                or start < current_day - 1e-6
                or tof <= 0
            ):
                return False

            dv = float(P.compute_transfer(
                route[i], route[i + 1], start, tof
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
            else:
                status = "INVALID"

            if tr["status"] != status or status == "INVALID":
                return False

            specials += (status == "SPECIAL")
            current_day = start + tof

        except Exception:
            return False

    return (
        abs(current_day - state["current_time"]) <= 1e-5
        and specials == state["special_used"]
        and specials <= P.n_exceptions
        and current_day <= P.max_time
    )

def best(old, move):
    if old is None:
        return move

    if (move["arrival"], move["dv"]) < (
        old["arrival"], old["dv"]
    ):
        return move

    return old

def save(report):
    tmp = OUTPUT.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(report, separators=(",", ":"))
    )
    tmp.replace(OUTPUT)

# Zgjedhim rrugët nga 3 nivele.
chosen = []

for level, number in SOURCE_LEVELS.items():

    source = RESULTS / f"beam_v9_auto_level_{level}.json"
    raw = source.read_bytes()

    config["source_hashes"][str(level)] = (
        hashlib.sha256(raw).hexdigest()
    )

    states = json.loads(raw)["frontier"]

    group = sorted(
        [
            s for s in states
            if s["special_used"] == 4
        ],
        key=lambda s: s["current_time"]
    )

    picks = []
    ends = set()
    ids = set()

    # Së pari zgjedhim domate fundore të ndryshme.
    for state in group:
        end = state["route"][-1]

        if end not in ends:
            picks.append(state)
            ends.add(end)
            ids.add(id(state))

        if len(picks) == number:
            break

    # Plotësojmë vendet e mbetura.
    for state in group:
        if len(picks) == number:
            break

        if id(state) not in ids:
            picks.append(state)
            ids.add(id(state))

    print(
        f"Niveli {level}: {len(group)} rruge me 4 SPECIAL, "
        f"zgjedhim {len(picks)}",
        flush=True
    )

    for state in picks:
        chosen.append((level, state))

if OUTPUT.exists():
    report = json.loads(OUTPUT.read_text())

    assert report["config"] == config, (
        "Checkpoint nuk perputhet me konfigurimin."
    )
else:
    report = {
        "config": config,
        "branches": {}
    }

# Analizojmë rrugët një nga një.
for index, (level, state) in enumerate(chosen, 1):

    if time.monotonic() - START > LIMIT:
        print(
            "U arrit kufiri i punes. "
            "Vazhdo me te njejten qelize.",
            flush=True
        )
        break

    uid = hashlib.sha256(
        json.dumps([
            state["route"],
            state["current_time"],
            state["special_used"]
        ]).encode()
    ).hexdigest()[:20]

    key = f"{level}_{uid}"

    if not verify(state):
        print(
            "KUJDES: Rruge e pavlefshme:",
            key,
            flush=True
        )
        continue

    if key not in report["branches"]:
        report["branches"][key] = {
            "level": level,
            "state": state,
            "targets": {},
            "complete": False
        }
        save(report)

    branch = report["branches"][key]

    if branch["complete"]:
        print(
            f"[{index}/{len(chosen)}] Nga checkpoint",
            flush=True
        )
        continue

    route = state["route"]
    current = route[-1]
    day = float(state["current_time"])

    remaining = [
        t for t in range(P.n_tomatoes)
        if t not in route
    ]

    print(
        f"\n[{index}/{len(chosen)}] "
        f"Niveli {level} | "
        f"Fund {current} | "
        f"Dita {day:.2f} | "
        f"SPECIAL 4/5",
        flush=True
    )

    # FAZA A: Kërkimi me shumë kombinime kohore.
    for target in remaining:

        if time.monotonic() - START > LIMIT:
            break

        if str(target) in branch["targets"]:
            continue

        min_dv = float("inf")
        earliest_normal = None
        earliest_special = None

        for wait in WAITS:
            for tof in TOFS:

                if (
                    wait + tof > MAX_EXTRA
                    or day + wait + tof > P.max_time
                ):
                    continue

                try:
                    dv = float(P.compute_transfer(
                        current,
                        target,
                        day + wait,
                        tof
                    ))
                except Exception:
                    continue

                if not math.isfinite(dv):
                    continue

                min_dv = min(min_dv, dv)

                move = {
                    "from": current,
                    "to": target,
                    "start_time": day + wait,
                    "tof": tof,
                    "arrival": day + wait + tof,
                    "dv": dv
                }

                if dv < P.dv_threshold:
                    earliest_normal = best(
                        earliest_normal, move
                    )

                elif dv < P.exception_threshold:
                    earliest_special = best(
                        earliest_special, move
                    )

        branch["targets"][str(target)] = {
            "grid_min_dv": (
                min_dv if math.isfinite(min_dv) else None
            ),
            "best_normal": earliest_normal,
            "best_special": earliest_special,
            "optimized": False,
            "optimized_min_dv": None
        }

        save(report)

    if len(branch["targets"]) != len(remaining):
        print(
            "Grid i pjesshem. Rifillo qelizen.",
            flush=True
        )
        break

    # FAZA B: Optimizimi i 3 kandidatëve më të mirë.
    ranked = sorted(
        (
            v["grid_min_dv"],
            int(t)
        )
        for t, v in branch["targets"].items()
        if v["grid_min_dv"] is not None
    )

    to_optimize = [
        target for _, target in ranked[:3]
    ]

    print(
        "Kandidatet per optimizim:",
        to_optimize,
        flush=True
    )

    for target in to_optimize:

        if time.monotonic() - START > LIMIT:
            break

        entry = branch["targets"][str(target)]

        if entry["optimized"]:
            continue

        best_dv = float("inf")

        def decode(x):
            wait = 12.0 * float(x[0])
            tof = (
                0.25
                + (MAX_EXTRA - wait - 0.25) * float(x[1])
            )
            return wait, tof

        def objective(x):
            wait, tof = decode(x)

            try:
                dv = float(P.compute_transfer(
                    current,
                    target,
                    day + wait,
                    tof
                ))
            except Exception:
                return 1e8

            return dv if math.isfinite(dv) else 1e8

        for seed in SEEDS:

            try:
                result = differential_evolution(
                    objective,
                    bounds=[(0, 1), (0, 1)],
                    seed=seed + target + level * 100,
                    maxiter=12,
                    popsize=6,
                    polish=False,
                    workers=1
                )

                wait, tof = decode(result.x)

                dv = float(P.compute_transfer(
                    current,
                    target,
                    day + wait,
                    tof
                ))

            except Exception:
                continue

            if math.isfinite(dv):

                best_dv = min(best_dv, dv)

                move = {
                    "from": current,
                    "to": target,
                    "start_time": day + wait,
                    "tof": tof,
                    "arrival": day + wait + tof,
                    "dv": dv
                }

                if dv < P.dv_threshold:
                    entry["best_normal"] = best(
                        entry["best_normal"], move
                    )

                elif dv < P.exception_threshold:
                    entry["best_special"] = best(
                        entry["best_special"], move
                    )

        entry["optimized"] = True

        entry["optimized_min_dv"] = (
            best_dv if math.isfinite(best_dv) else None
        )

        save(report)

        print(
            f"  {current} -> {target} | "
            f"Delta-V minimale e gjetur: {best_dv:.2f}",
            flush=True
        )

    if not all(
        branch["targets"][str(t)]["optimized"]
        for t in to_optimize
    ):
        print(
            "Optimizim i pjesshem. Rifillo qelizen.",
            flush=True
        )
        break

    branch["complete"] = True
    save(report)

    normals = [
        v["best_normal"]
        for v in branch["targets"].values()
        if v["best_normal"] is not None
    ]

    specials = [
        v["best_special"]
        for v in branch["targets"].values()
        if v["best_special"] is not None
    ]

    print(
        f"NORMAL: {len(normals)} destinacione | "
        f"SPECIAL: {len(specials)} destinacione",
        flush=True
    )

    if normals:
        first = min(
            normals,
            key=lambda m: (m["arrival"], m["dv"])
        )

        print(
            "NORMAL me i hershem:",
            first,
            flush=True
        )

print("\nPERMBLEDHJA E KERKIMIT")
print("=" * 60)

for key, branch in report["branches"].items():

    state = branch["state"]

    normals = [
        v["best_normal"]
        for v in branch["targets"].values()
        if v["best_normal"] is not None
    ]

    print(
        f"\nNiveli {branch['level']}/49 | "
        f"Dita {state['current_time']:.2f} | "
        f"Fund {state['route'][-1]} | "
        f"NORMAL {len(normals)} | "
        f"Kontrolluar {len(branch['targets'])} | "
        f"Perfunduar {branch['complete']}",
        flush=True
    )

    if normals:
        move = min(
            normals,
            key=lambda m: (m["arrival"], m["dv"])
        )

        print(
            f"VAZHDIM NORMAL: "
            f"{move['from']} -> {move['to']} | "
            f"Dita {move['arrival']:.2f} | "
            f"DV {move['dv']:.2f}",
            flush=True
        )

    else:
        minimums = sorted(
            (
                v["grid_min_dv"],
                int(t)
            )
            for t, v in branch["targets"].items()
            if v["grid_min_dv"] is not None
        )

        print(
            "3 minimumet nga grid:",
            [
                (round(dv, 2), target)
                for dv, target in minimums[:3]
            ],
            flush=True
        )

print("\nHAPI 114 - ANALIZA PERFUNDOI!")
