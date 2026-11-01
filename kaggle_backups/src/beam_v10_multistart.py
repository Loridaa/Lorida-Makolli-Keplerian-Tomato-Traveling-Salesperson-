
import json
import math
import time
import hashlib
from pathlib import Path
from collections import defaultdict

from scipy.optimize import minimize
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_v10_seed_audit.json"
OUTPUT = RESULTS / "beam_v10_multistart.json"

P = TomatoProblem(str(ROOT / "data/easy.kttsp"))

SOURCE_HASH = hashlib.sha256(
    SOURCE.read_bytes()
).hexdigest()

AUDIT = json.loads(SOURCE.read_text())["audit"]

# Tri pikënisjet nga Hapi 115.
BASES = [
    ("A16", 16, 42, 58.0, 3),
    ("B17", 17, 33, 59.5, 3),
    ("C19", 19, 10, 102.0, 3)
]

WAITS = [0, 0.5, 1, 2, 3, 5, 8]
TOFS = [0.5, 1, 1.5, 2, 3, 5, 8, 13]

WIDTH = 42
MAX_PARENTS = 22
GOAL = 30
BUDGET_SECONDS = 135

CONFIG = {
    "source_sha256": SOURCE_HASH,
    "bases": BASES,
    "waits": WAITS,
    "tofs": TOFS,
    "width": WIDTH,
    "parents": MAX_PARENTS,
    "goal": GOAL,
    "version": 1
}

def uid(state):
    payload = [
        state["route"],
        round(float(state["current_time"]), 7),
        int(state["special_used"])
    ]

    raw = json.dumps(payload)

    return hashlib.sha256(
        raw.encode()
    ).hexdigest()[:20]

def save(report):
    temporary = OUTPUT.with_suffix(".tmp")

    temporary.write_text(
        json.dumps(report, separators=(",", ":"))
    )

    temporary.replace(OUTPUT)

def verify_last(state):
    try:
        route = state["route"]
        transfers = state["transfers"]

        if (
            len(route) != len(set(route))
            or len(transfers) != len(route) - 1
        ):
            return False

        if (
            state["current_time"] > P.max_time
            or state["special_used"] > P.n_exceptions
        ):
            return False

        last = transfers[-1]

        if (
            last["from"] != route[-2]
            or last["to"] != route[-1]
        ):
            return False

        start = float(last["start_time"])
        tof = float(last["tof"])

        if (
            start < 0
            or tof <= 0
            or abs(
                start + tof - state["current_time"]
            ) > 1e-5
        ):
            return False

        dv = float(P.compute_transfer(
            route[-2],
            route[-1],
            start,
            tof
        ))

        if dv < P.dv_threshold:
            status = "NORMAL"
        elif dv < P.exception_threshold:
            status = "SPECIAL"
        else:
            status = "INVALID"

        return (
            math.isfinite(dv)
            and abs(dv - last["dv"]) < 0.1
            and status == last["status"]
        )

    except Exception:
        return False

def pick(states, maximum):
    ranked = sorted(
        states,
        key=lambda s: (
            s["current_time"]
            + 15 * s["special_used"],
            s["current_time"]
        )
    )

    selected = []
    seen = set()
    endings = defaultdict(int)

    # Preferojme rruget qe ruajne SPECIAL.
    for used, quota in [
        (2, 8),
        (3, 12),
        (4, 16),
        (5, 6)
    ]:

        count = 0

        for state in ranked:
            if (
                len(selected) >= maximum
                or count >= quota
            ):
                break

            key = uid(state)
            ending = state["route"][-1]

            if (
                state["special_used"] != used
                or key in seen
                or endings[ending] >= 4
            ):
                continue

            selected.append(state)
            seen.add(key)
            endings[ending] += 1
            count += 1

    # Plotesojme vendet e mbetura.
    for state in ranked:
        if len(selected) >= maximum:
            break

        key = uid(state)

        if key not in seen:
            selected.append(state)
            seen.add(key)

    return selected

def expand(state):
    route = state["route"]
    day = float(state["current_time"])
    used = int(state["special_used"])

    current = route[-1]
    visited = set(route)

    remaining = P.n_tomatoes - len(route)

    average_time = (
        P.max_time - day
    ) / max(1, remaining)

    max_extra = min(
        18.0,
        max(7.0, 4 * average_time)
    )

    normals = []
    specials = []
    near_normal = []
    evaluations = 0

    for target in range(P.n_tomatoes):
        if target in visited:
            continue

        target_normals = []
        target_specials = []
        closest = None

        for wait in WAITS:
            for tof in TOFS:

                if (
                    wait + tof > max_extra
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

                    evaluations += 1

                except Exception:
                    continue

                if not math.isfinite(dv):
                    continue

                move = {
                    "from": current,
                    "to": target,
                    "start_time": day + wait,
                    "tof": tof,
                    "arrival": day + wait + tof,
                    "dv": dv
                }

                if (
                    closest is None
                    or dv < closest["dv"]
                ):
                    closest = move

                if dv < P.dv_threshold:
                    target_normals.append(move)

                elif (
                    dv < P.exception_threshold
                    and used < P.n_exceptions
                ):
                    target_specials.append(move)

        if target_normals:

            earliest = min(
                target_normals,
                key=lambda m: (
                    m["arrival"], m["dv"]
                )
            )

            normals.append(earliest)

            lowest_dv = min(
                target_normals,
                key=lambda m: (
                    m["dv"], m["arrival"]
                )
            )

            # Variant alternativ me kohe tjeter.
            if abs(
                lowest_dv["arrival"]
                - earliest["arrival"]
            ) >= 2:
                normals.append(lowest_dv)

        if target_specials:
            specials.append(
                min(
                    target_specials,
                    key=lambda m: (
                        m["arrival"], m["dv"]
                    )
                )
            )

        if (
            not target_normals
            and closest is not None
            and 100 <= closest["dv"] < 155
        ):
            near_normal.append(closest)

    # Optimizim lokal per rastet afer NORMAL.
    for initial in sorted(
        near_normal,
        key=lambda m: m["dv"]
    )[:2]:

        target = initial["to"]

        def objective(x):
            wait = float(x[0])
            tof = float(x[1])

            if (
                wait < 0
                or tof < 0.25
                or wait + tof > max_extra
            ):
                return 1e6 + abs(wait) + abs(tof)

            try:
                value = float(P.compute_transfer(
                    current,
                    target,
                    day + wait,
                    tof
                ))

                return (
                    value
                    if math.isfinite(value)
                    else 1e6
                )
            except Exception:
                return 1e6

        initial_guess = [
            max(
                0.02,
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
                    "maxfev": 65,
                    "maxiter": 50,
                    "xatol": 0.02,
                    "fatol": 0.1
                }
            )

            wait = float(result.x[0])
            tof = float(result.x[1])

            if (
                wait >= 0
                and tof >= 0.25
                and wait + tof <= max_extra
            ):

                dv = objective((wait, tof))

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
        key=lambda m: (
            m["arrival"], m["dv"]
        )
    )

    specials.sort(
        key=lambda m: (
            m["arrival"], m["dv"]
        )
    )

    children = []

    for status, moves, limit in [
        ("NORMAL", normals, 7),
        ("SPECIAL", specials, 4)
    ]:

        seen_moves = set()

        for move in moves:

            if len(seen_moves) >= limit:
                break

            key = (
                move["to"],
                round(move["arrival"] * 2) / 2
            )

            if key in seen_moves:
                continue

            seen_moves.add(key)

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
                "route": (
                    route + [move["to"]]
                ),
                "transfers": (
                    state["transfers"] + [transfer]
                ),
                "current_time": move["arrival"],
                "special_used": (
                    used + int(status == "SPECIAL")
                )
            })

    info = {
        "normal_moves": len(normals),
        "special_moves": len(specials),
        "near_normal": len(near_normal),
        "evaluations": evaluations
    }

    return children, info

def choose_frontier(parent_data):
    unique = {}

    for parent in parent_data.values():
        for state in parent["children"]:

            key = (
                tuple(sorted(state["route"])),
                state["route"][-1],
                state["special_used"],
                round(
                    state["current_time"] * 2
                ) / 2
            )

            if (
                key not in unique
                or state["current_time"]
                < unique[key]["current_time"]
            ):
                unique[key] = state

    return (
        pick(list(unique.values()), WIDTH),
        len(unique)
    )

# Marrim 3 rruget e verifikuara nga Hapi 115.
base_states = {}

for name, level, end, day, used in BASES:

    matches = [
        entry["state"]
        for entry in AUDIT.values()
        if (
            entry["verified"]
            and len(entry["state"]["route"]) == level
            and entry["state"]["route"][-1] == end
            and abs(
                entry["state"]["current_time"] - day
            ) < 0.001
            and entry["state"]["special_used"] == used
        )
    ]

    if not matches:
        raise RuntimeError(
            "Mungon pika fillestare: " + name
        )

    base_states[name] = matches[0]

# Checkpoint-i i perbashket.
if OUTPUT.exists():

    report = json.loads(OUTPUT.read_text())

    assert report["config"] == CONFIG, (
        "Checkpoint ka konfigurim tjeter."
    )

else:

    report = {
        "config": CONFIG,
        "lanes": {
            name: {
                "level": len(state["route"]),
                "frontier": [state],
                "pending": None,
                "history": [],
                "stopped": False
            }
            for name, state in base_states.items()
        }
    }

    save(report)

print("BEAM SEARCH V10 - TRI STRATEGJI")
print("=" * 60, flush=True)

for name, lane in report["lanes"].items():
    print(
        f"{name} | "
        f"Niveli {lane['level']}/49 | "
        f"Rruge {len(lane['frontier'])}",
        flush=True
    )

started = time.monotonic()

# Rrotullimi lejon secilen strategji
# te perparoje, pa u fokusuar vetem te njera.
while time.monotonic() - started < BUDGET_SECONDS:

    advanced_any = False

    for name in ["A16", "B17", "C19"]:

        lane = report["lanes"][name]

        if (
            lane["stopped"]
            or lane["level"] >= GOAL
        ):
            continue

        if (
            time.monotonic() - started
            >= BUDGET_SECONDS
        ):
            break

        valid = [
            state
            for state in lane["frontier"]
            if verify_last(state)
        ]

        if not valid:
            lane["stopped"] = True
            save(report)

            print(
                name,
                "- nuk ka rruge te vlefshme",
                flush=True
            )
            continue

        selected = pick(
            valid,
            MAX_PARENTS
        )

        seed_ids = [
            uid(state)
            for state in selected
        ]

        if lane["pending"] is None:
            lane["pending"] = {
                "target": lane["level"] + 1,
                "seed_ids": seed_ids,
                "parents": {}
            }

            save(report)

        pending = lane["pending"]

        assert pending["target"] == lane["level"] + 1
        assert pending["seed_ids"] == seed_ids

        print(
            f"\n{name}: "
            f"{lane['level']} -> "
            f"{lane['level'] + 1} | "
            f"Prinder {len(selected)}",
            flush=True
        )

        for index, state in enumerate(
            selected, 1
        ):

            sid = uid(state)

            if sid in pending["parents"]:
                continue

            if (
                time.monotonic() - started
                >= BUDGET_SECONDS
            ):
                break

            children, info = expand(state)

            pending["parents"][sid] = {
                "children": children,
                "info": info
            }

            save(report)

            print(
                f"[{index}/{len(selected)}] "
                f"Fund {state['route'][-1]} | "
                f"Dita {state['current_time']:.2f} | "
                f"SPECIAL {state['special_used']}/5 | "
                f"NORMAL {info['normal_moves']} | "
                f"SPECIAL {info['special_moves']}",
                flush=True
            )

        if len(pending["parents"]) < len(selected):

            print(
                name,
                "- checkpoint i pjesshem; "
                "rifillo qelizen",
                flush=True
            )

            break

        frontier_new, unique_count = (
            choose_frontier(
                pending["parents"]
            )
        )

        lane["level"] += 1
        lane["frontier"] = frontier_new

        lane["history"].append({
            "level": lane["level"],
            "unique": unique_count,
            "frontier": len(frontier_new)
        })

        lane["pending"] = None
        lane["stopped"] = not bool(frontier_new)

        save(report)

        advanced_any = True

        print(
            f"\n{name} - NIVELI "
            f"{lane['level']}/49",
            flush=True
        )

        print(
            "Rruge unike:",
            unique_count,
            flush=True
        )

        print(
            "Frontier:",
            len(frontier_new),
            flush=True
        )

        for used in range(6):

            group = [
                state
                for state in frontier_new
                if state["special_used"] == used
            ]

            if group:
                earliest = min(
                    s["current_time"]
                    for s in group
                )

                print(
                    f"  SPECIAL {used}/5 | "
                    f"Rruge {len(group)} | "
                    f"Dita {earliest:.2f}",
                    flush=True
                )

    if not advanced_any:
        break

print("\nREZULTATI V10")
print("=" * 60, flush=True)

for name, lane in report["lanes"].items():
    print(
        f"{name} | "
        f"Niveli {lane['level']}/49 | "
        f"Frontier {len(lane['frontier'])} | "
        f"Ne proces {lane['pending'] is not None} | "
        f"Ndalur {lane['stopped']}",
        flush=True
    )

print("\nHAPI 116 - EKZEKUTIMI PERFUNDOI!", flush=True)
