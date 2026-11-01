
import json
import math
import time
import hashlib
from pathlib import Path
from scipy.optimize import minimize
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

OUTPUT = RESULTS / "beam_v11_expand_31_checkpoint.json"

P = TomatoProblem(str(ROOT / "data/easy.kttsp"))

WAITS = [0, 0.25, 0.5, 1, 2, 3, 5, 8, 13, 20]
TOFS = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 13, 18, 24]

WORK_LIMIT = 140.0
START = time.monotonic()

BEAM_WIDTH = {
    "A14": 60,
    "B12": 24
}

def state_id(state):
    data = [
        state["route"],
        round(float(state["current_time"]), 7),
        state["special_used"]
    ]

    return hashlib.sha256(
        json.dumps(data).encode()
    ).hexdigest()[:20]

def verify(state):
    try:
        route = state["route"]
        transfers = state["transfers"]

        if (
            len(route) != len(set(route))
            or route[0] != 0
            or len(transfers) != len(route) - 1
            or len(route) > P.n_tomatoes
        ):
            return False

        day = 0.0
        used = 0

        for i, tr in enumerate(transfers):
            depart = float(tr["start_time"])
            tof = float(tr["tof"])

            if (
                tr["from"] != route[i]
                or tr["to"] != route[i + 1]
                or depart < day - 1e-6
                or tof <= 0
                or depart + tof > P.max_time + 1e-6
            ):
                return False

            dv = float(P.compute_transfer(
                route[i],
                route[i + 1],
                depart,
                tof
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
                used += 1

            else:
                return False

            if status != tr["status"]:
                return False

            day = depart + tof

        return (
            abs(day - state["current_time"]) < 1e-5
            and used == state["special_used"]
            and used <= P.n_exceptions
        )

    except Exception:
        return False

def save(report):
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(report, separators=(",", ":"))
    )
    temporary.replace(OUTPUT)

def expand(state):
    route = state["route"]
    visited = set(route)

    current = route[-1]
    day = float(state["current_time"])
    used = int(state["special_used"])

    window = min(36.0, P.max_time - day)

    normals = []
    specials = []
    near = []

    minimum_dv = float("inf")
    evaluations = 0

    for target in range(P.n_tomatoes):

        if target in visited:
            continue

        normal_choices = []
        special_choices = []
        closest = None

        for wait in WAITS:
            for tof in TOFS:

                if wait + tof > window:
                    continue

                depart = day + wait
                arrival = depart + tof

                try:
                    dv = float(P.compute_transfer(
                        current, target, depart, tof
                    ))
                    evaluations += 1
                except Exception:
                    continue

                if not math.isfinite(dv):
                    continue

                minimum_dv = min(minimum_dv, dv)

                move = {
                    "from": current,
                    "to": target,
                    "start_time": depart,
                    "tof": tof,
                    "arrival": arrival,
                    "dv": dv
                }

                if closest is None or dv < closest["dv"]:
                    closest = move

                if dv < P.dv_threshold:
                    normal_choices.append(move)

                elif (
                    dv < P.exception_threshold
                    and used < P.n_exceptions
                ):
                    special_choices.append(move)

        if normal_choices:
            earliest = min(
                normal_choices,
                key=lambda m: (m["arrival"], m["dv"])
            )

            normals.append(earliest)

            low_dv = min(
                normal_choices,
                key=lambda m: (m["dv"], m["arrival"])
            )

            if abs(
                low_dv["arrival"] - earliest["arrival"]
            ) >= 1:
                normals.append(low_dv)

        if special_choices:
            specials.append(
                min(
                    special_choices,
                    key=lambda m: (m["arrival"], m["dv"])
                )
            )

        if (
            not normal_choices
            and closest is not None
            and 100 <= closest["dv"] < 220
        ):
            near.append(closest)

    # Optimizim lokal per rastet me DV
    # afer kufirit 100.
    for initial in sorted(
        near,
        key=lambda m: m["dv"]
    )[:3]:

        target = initial["to"]

        def objective(x):
            wait = float(x[0])
            tof = float(x[1])

            if (
                wait < 0
                or tof < 0.2
                or wait + tof > window
            ):
                return 1e8

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

        initial_guess = [
            max(
                0.001,
                initial["start_time"] - day
            ),
            initial["tof"]
        ]

        try:
            result = minimize(
                objective,
                initial_guess,
                method="Nelder-Mead",
                options={
                    "maxiter": 55,
                    "maxfev": 75,
                    "xatol": 0.01,
                    "fatol": 0.1
                }
            )

            wait = float(result.x[0])
            tof = float(result.x[1])

            if (
                wait >= 0
                and tof >= 0.2
                and wait + tof <= window
            ):
                dv = objective([wait, tof])

                if dv < P.dv_threshold:
                    normals.append({
                        "from": current,
                        "to": target,
                        "start_time": day + wait,
                        "tof": tof,
                        "arrival": day + wait + tof,
                        "dv": dv
                    })

        except Exception:
            pass

    normals.sort(
        key=lambda m: (m["arrival"], m["dv"])
    )

    specials.sort(
        key=lambda m: (m["arrival"], m["dv"])
    )

    children = []

    for status, moves, limit in [
        ("NORMAL", normals, 12),
        ("SPECIAL", specials, 4)
    ]:

        seen = set()

        for move in moves:

            if len(seen) >= limit:
                break

            key = (
                move["to"],
                round(move["arrival"] * 4) / 4
            )

            if key in seen:
                continue

            seen.add(key)

            transfer = {
                field: move[field]
                for field in [
                    "from",
                    "to",
                    "start_time",
                    "tof",
                    "dv"
                ]
            }

            transfer["status"] = status

            children.append({
                "route": route + [move["to"]],
                "transfers": (
                    state["transfers"] + [transfer]
                ),
                "current_time": move["arrival"],
                "special_used": (
                    used + int(status == "SPECIAL")
                )
            })

    return children, {
        "normal": len(normals),
        "special": len(specials),
        "minimum_dv": (
            minimum_dv
            if math.isfinite(minimum_dv)
            else None
        ),
        "evaluations": evaluations
    }

def make_frontier(parents, width):
    unique = {}

    for result in parents.values():
        for state in result["children"]:

            key = (
                tuple(sorted(state["route"])),
                state["route"][-1],
                state["special_used"],
                round(state["current_time"] * 4) / 4
            )

            if (
                key not in unique
                or state["current_time"] <
                unique[key]["current_time"]
            ):
                unique[key] = state

    ranked = sorted(
        unique.values(),
        key=lambda s: (
            s["special_used"],
            s["current_time"]
        )
    )

    selected = []
    seen = set()
    end_counts = {}

    # Ruajme alternativat me 4 SPECIAL.
    for used, quota in [(4, 45), (5, width)]:

        count = 0

        for state in ranked:
            if len(selected) >= width or count >= quota:
                break

            if state["special_used"] != used:
                continue

            sid = state_id(state)
            end = state["route"][-1]

            if sid in seen or end_counts.get(end, 0) >= 8:
                continue

            selected.append(state)
            seen.add(sid)
            end_counts[end] = end_counts.get(end, 0) + 1
            count += 1

    for state in ranked:
        if len(selected) >= width:
            break

        sid = state_id(state)

        if sid not in seen:
            selected.append(state)
            seen.add(sid)

    verified = [
        state
        for state in selected
        if verify(state)
    ]

    return verified, len(unique)

# FAZA 1 - Lexojme burimet.
sources = {}
hashes = {}

for name in ["A14", "B12"]:

    path = RESULTS / f"beam_v11_{name}_level_30.json"
    raw = path.read_bytes()

    hashes[name] = hashlib.sha256(raw).hexdigest()

    data = json.loads(raw)
    states = data["frontier"]

    assert states
    assert all(len(s["route"]) == 30 for s in states)

    valid = [s for s in states if verify(s)]

    if len(valid) != len(states):
        raise ValueError(
            f"{name}: disa rruge nuk jane te vlefshme."
        )

    sources[name] = sorted(
        valid,
        key=lambda s: (
            s["special_used"],
            s["current_time"]
        )
    )

CONFIG = {
    "sources": hashes,
    "waits": WAITS,
    "tofs": TOFS,
    "beam_width": BEAM_WIDTH,
    "version": 1
}

if OUTPUT.exists():
    report = json.loads(OUTPUT.read_text())

    assert report["config"] == CONFIG
else:
    report = {
        "config": CONFIG,
        "lanes": {
            name: {
                "seed_ids": [
                    state_id(s)
                    for s in states
                ],
                "parents": {},
                "frontier": [],
                "complete": False
            }
            for name, states in sources.items()
        }
    }

    save(report)

print("FAZA 1 - RRUGET E VERIFIKUARA")
print("=" * 65, flush=True)

for name, states in sources.items():
    print(
        f"{name}: {len(states)} rruge me 30 domate",
        flush=True
    )

print("\nFAZA 2 - KERKIMI 30 -> 31")
print("=" * 65, flush=True)

for name in ["A14", "B12"]:

    lane = report["lanes"][name]
    states = sources[name]

    assert lane["seed_ids"] == [
        state_id(s) for s in states
    ]

    if lane["complete"]:
        print(
            f"{name}: niveli 31 i analizuar me pare",
            flush=True
        )
        continue

    print(
        f"\nSTRATEGJIA {name} | "
        f"Prinder {len(states)}",
        flush=True
    )

    for index, state in enumerate(states, 1):

        if time.monotonic() - START >= WORK_LIMIT:
            print(
                "U arrit kufiri i punes.",
                flush=True
            )
            break

        sid = state_id(state)

        if sid in lane["parents"]:
            continue

        children, info = expand(state)

        lane["parents"][sid] = {
            "children": children,
            "info": info
        }

        save(report)

        print(
            f"[{index}/{len(states)}] "
            f"Fund {state['route'][-1]} | "
            f"Dita {state['current_time']:.2f} | "
            f"SPECIAL {state['special_used']}/5 | "
            f"NORMAL {info['normal']} | "
            f"SPECIAL {info['special']} | "
            f"Femije {len(children)}",
            flush=True
        )

    if len(lane["parents"]) < len(states):
        print(
            f"{name}: checkpoint i pjesshem.",
            flush=True
        )
        break

    frontier, unique_count = make_frontier(
        lane["parents"],
        BEAM_WIDTH[name]
    )

    lane["frontier"] = frontier
    lane["unique_children"] = unique_count
    lane["complete"] = True

    save(report)

    if frontier:
        snapshot = RESULTS / (
            f"beam_v11_{name}_level_31.json"
        )

        snapshot.write_text(
            json.dumps({
                "strategy": name,
                "level": 31,
                "frontier": frontier
            }, separators=(",", ":"))
        )

    print(
        f"\n{name} - REZULTATI NIVELI 31",
        flush=True
    )

    print(
        "Rruge unike:", unique_count,
        flush=True
    )

    print(
        "Rruge te verifikuara:",
        len(frontier),
        flush=True
    )

    for used in [4, 5]:
        group = [
            s for s in frontier
            if s["special_used"] == used
        ]

        if group:
            print(
                f"SPECIAL {used}/5 | "
                f"Rruge {len(group)} | "
                f"Dita minimale "
                f"{min(s['current_time'] for s in group):.2f}",
                flush=True
            )

print("\nREZULTATI I HAPIT 121")
print("=" * 65, flush=True)

for name, lane in report["lanes"].items():

    frontier = lane["frontier"]

    print(
        f"{name} | "
        f"Prinder {len(lane['parents'])}/"
        f"{len(lane['seed_ids'])} | "
        f"Perfunduar {lane['complete']} | "
        f"Rruge nivel 31: {len(frontier)}",
        flush=True
    )

    if frontier:
        best = min(
            frontier,
            key=lambda s: (
                s["special_used"],
                s["current_time"]
            )
        )

        print(
            f"  Dita {best['current_time']:.2f} | "
            f"SPECIAL {best['special_used']}/5 | "
            f"Fund {best['route'][-1]}",
            flush=True
        )

print("\nHAPI 121 - KERKIMI PERFUNDOI!", flush=True)
