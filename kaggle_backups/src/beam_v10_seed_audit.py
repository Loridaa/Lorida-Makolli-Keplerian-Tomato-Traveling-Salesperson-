
import json
import math
import time
import hashlib
from pathlib import Path
from collections import defaultdict
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "beam_v10_seed_audit.json"

P = TomatoProblem(str(ROOT / "data/easy.kttsp"))

LEVELS = [16, 17, 18, 19, 20, 21]
MAX_PER_LEVEL = 5

WAITS = [0, 0.5, 1, 2, 3, 5]
TOFS = [0.5, 1, 1.5, 2, 3, 5, 8]
MAX_EXTRA = 12.0

START = time.monotonic()
WORK_LIMIT = 140.0

config = {
    "levels": LEVELS,
    "max_per_level": MAX_PER_LEVEL,
    "waits": WAITS,
    "tofs": TOFS,
    "max_extra": MAX_EXTRA,
    "version": 1
}

found = {}
scanned = 0

def state_key(state):
    payload = [
        state["route"],
        round(float(state["current_time"]), 6),
        int(state["special_used"])
    ]
    return hashlib.sha256(
        json.dumps(payload).encode()
    ).hexdigest()[:20]

def collect(obj, filename, depth=0):
    if depth > 9:
        return

    if isinstance(obj, dict):

        route = obj.get("route")
        day = obj.get("current_time")
        used = obj.get("special_used")
        transfers = obj.get("transfers")

        if (
            isinstance(route, list)
            and len(route) in LEVELS
            and all(type(t) is int and 0 <= t < 49 for t in route)
            and len(route) == len(set(route))
            and route[0] == 0
            and isinstance(day, (int, float))
            and math.isfinite(day)
            and 0 <= day <= 140
            and type(used) is int
            and 0 <= used <= 4
            and isinstance(transfers, list)
            and len(transfers) == len(route) - 1
        ):
            state = {
                "route": route,
                "current_time": float(day),
                "special_used": used,
                "transfers": transfers,
                "source_file": filename
            }

            key = state_key(state)

            if key not in found:
                found[key] = state

        for key, value in obj.items():

            if key in (
                "route", "transfers", "targets",
                "evaluations", "history", "cache"
            ):
                continue

            if isinstance(value, dict):
                if len(value) <= 3000:
                    collect(value, filename, depth + 1)

            elif isinstance(value, list):
                if len(value) <= 3000:
                    collect(value, filename, depth + 1)

    elif isinstance(obj, list):
        for value in obj:
            if isinstance(value, (dict, list)):
                collect(value, filename, depth + 1)

print("FAZA 1 - LEXIMI I CHECKPOINT-EVE", flush=True)

for path in sorted(RESULTS.glob("*.json")):

    if path == OUTPUT or path.stat().st_size > 30_000_000:
        continue

    try:
        obj = json.loads(path.read_text())
        collect(obj, path.name)
        scanned += 1
    except Exception:
        continue

states = list(found.values())

print("JSON te lexuar:", scanned, flush=True)
print("Rruge kandidate:", len(states), flush=True)

def select_level(group):
    chosen = []
    seen = set()
    endpoints = defaultdict(int)

    # Krahasojme ruajtjen e SPECIAL me shpejtesine.
    for budget, quota in [
        ("low", 1),
        (3, 2),
        (4, 2)
    ]:
        if budget == "low":
            subset = [
                s for s in group
                if s["special_used"] <= 2
            ]
        else:
            subset = [
                s for s in group
                if s["special_used"] == budget
            ]

        subset.sort(key=lambda s: s["current_time"])
        count = 0

        for state in subset:
            sid = state_key(state)
            end = state["route"][-1]

            if sid in seen or endpoints[end] >= 2:
                continue

            chosen.append(state)
            seen.add(sid)
            endpoints[end] += 1
            count += 1

            if count >= quota:
                break

    # Plotesojme nivelin me rruget me te mira
    # qe ende nuk jane perzgjedhur.
    remaining = sorted(
        group,
        key=lambda s: (
            s["current_time"] + 12 * s["special_used"],
            s["current_time"]
        )
    )

    for state in remaining:
        if len(chosen) >= MAX_PER_LEVEL:
            break

        sid = state_key(state)

        if sid not in seen:
            chosen.append(state)
            seen.add(sid)

    return chosen[:MAX_PER_LEVEL]

selected = []

print("\nFAZA 2 - PERZGJEDHJA E RRUGEVE", flush=True)

for level in LEVELS:
    group = [
        s for s in states
        if len(s["route"]) == level
    ]

    chosen = select_level(group)
    selected.extend(chosen)

    print(
        f"Niveli {level}: "
        f"{len(group)} kandidate, "
        f"{len(chosen)} te zgjedhura",
        flush=True
    )

seed_ids = [state_key(s) for s in selected]

if OUTPUT.exists():
    report = json.loads(OUTPUT.read_text())

    assert report["config"] == config
    assert report["seed_ids"] == seed_ids, (
        "Checkpoint nuk perputhet me perzgjedhjen."
    )

else:
    report = {
        "config": config,
        "seed_ids": seed_ids,
        "scanned_files": scanned,
        "candidate_count": len(states),
        "audit": {}
    }

def save():
    temp = OUTPUT.with_suffix(".tmp")
    temp.write_text(
        json.dumps(report, separators=(",", ":"))
    )
    temp.replace(OUTPUT)

def verify(state):
    route = state["route"]
    transfers = state["transfers"]

    day = 0.0
    specials = 0

    for i, tr in enumerate(transfers):
        try:
            start = float(tr["start_time"])
            tof = float(tr["tof"])

            if (
                tr["from"] != route[i]
                or tr["to"] != route[i + 1]
                or start < day - 1e-6
                or tof <= 0
            ):
                return False

            dv = float(P.compute_transfer(
                route[i],
                route[i + 1],
                start,
                tof
            ))

            if not math.isfinite(dv):
                return False

            if abs(dv - float(tr["dv"])) > 0.1:
                return False

            if dv < P.dv_threshold:
                status = "NORMAL"
            elif dv < P.exception_threshold:
                status = "SPECIAL"
                specials += 1
            else:
                return False

            if tr["status"] != status:
                return False

            day = start + tof

        except Exception:
            return False

    return (
        abs(day - state["current_time"]) < 1e-5
        and specials == state["special_used"]
        and day <= P.max_time
    )

def best_move(old, new):
    if old is None:
        return new

    if (new["arrival"], new["dv"]) < (
        old["arrival"], old["dv"]
    ):
        return new

    return old

print("\nFAZA 3 - VERIFIKIMI DHE MUNDESITE NORMALE")
print("=" * 60, flush=True)

for i, state in enumerate(selected, 1):

    if time.monotonic() - START > WORK_LIMIT:
        print(
            "Kufiri i punes u arrit; "
            "vazhdo me te njejten qelize.",
            flush=True
        )
        break

    sid = state_key(state)

    if sid in report["audit"]:
        continue

    level = len(state["route"])
    day = state["current_time"]
    used = state["special_used"]
    current = state["route"][-1]

    print(
        f"[{i}/{len(selected)}] "
        f"Niveli {level} | "
        f"Dita {day:.2f} | "
        f"SPECIAL {used}/5 | "
        f"Fund {current}",
        flush=True
    )

    valid = verify(state)

    if not valid:
        report["audit"][sid] = {
            "state": state,
            "verified": False,
            "normal_count": 0,
            "special_count": 0
        }
        save()
        print("  Rruga nuk kaloi verifikimin.", flush=True)
        continue

    visited = set(state["route"])
    normals = []
    specials = []
    evaluations = 0

    for target in range(P.n_tomatoes):

        if target in visited:
            continue

        best_normal = None
        best_special = None

        for wait in WAITS:
            for tof in TOFS:

                if (
                    wait + tof > MAX_EXTRA
                    or day + wait + tof > P.max_time
                ):
                    continue

                depart = day + wait
                arrival = depart + tof

                try:
                    dv = float(P.compute_transfer(
                        current,
                        target,
                        depart,
                        tof
                    ))
                    evaluations += 1
                except Exception:
                    continue

                if not math.isfinite(dv):
                    continue

                move = {
                    "from": current,
                    "to": target,
                    "start_time": depart,
                    "tof": tof,
                    "arrival": arrival,
                    "dv": dv
                }

                if dv < P.dv_threshold:
                    best_normal = best_move(
                        best_normal, move
                    )

                elif (
                    dv < P.exception_threshold
                    and used < P.n_exceptions
                ):
                    best_special = best_move(
                        best_special, move
                    )

        if best_normal is not None:
            normals.append(best_normal)

        if best_special is not None:
            specials.append(best_special)

    normals.sort(
        key=lambda m: (m["arrival"], m["dv"])
    )

    specials.sort(
        key=lambda m: (m["arrival"], m["dv"])
    )

    report["audit"][sid] = {
        "state": state,
        "verified": True,
        "normal_count": len(normals),
        "special_count": len(specials),
        "normal_moves": normals[:8],
        "special_moves": specials[:4],
        "evaluations": evaluations
    }

    save()

    print(
        f"  NORMAL: {len(normals)} | "
        f"SPECIAL: {len(specials)}",
        flush=True
    )

    if normals:
        move = normals[0]
        print(
            f"  NORMAL me i hershem: "
            f"{current} -> {move['to']} | "
            f"Dita {move['arrival']:.2f} | "
            f"DV {move['dv']:.2f}",
            flush=True
        )

print("\nREZULTATI I AUDITIMIT V10")
print("=" * 60, flush=True)

valid_audit = [
    a for a in report["audit"].values()
    if a["verified"]
]

ranked = sorted(
    valid_audit,
    key=lambda a: (
        -a["normal_count"],
        a["state"]["special_used"],
        a["state"]["current_time"]
    )
)

print("Rruge te perzgjedhura:", len(selected), flush=True)
print("Rruge te analizuara:", len(report["audit"]), flush=True)
print("Rruge te verifikuara:", len(valid_audit), flush=True)
print(
    "Rruge me te pakten 1 NORMAL:",
    sum(a["normal_count"] > 0 for a in valid_audit),
    flush=True
)

print("\nKANDIDATET ME PREMTUES")
print("-" * 60, flush=True)

for a in ranked[:12]:
    state = a["state"]

    print(
        f"Niveli {len(state['route'])} | "
        f"Dita {state['current_time']:.2f} | "
        f"SPECIAL {state['special_used']}/5 | "
        f"Fund {state['route'][-1]} | "
        f"NORMAL {a['normal_count']} | "
        f"SPECIAL {a['special_count']}",
        flush=True
    )

print("\nHAPI 115 - ANALIZA PERFUNDOI!", flush=True)
