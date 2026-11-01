
import json
import math
import hashlib
from pathlib import Path
from collections import defaultdict
from scipy.optimize import minimize
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"
SOURCE = RESULTS / "beam_v9_level_23.json"

P = TomatoProblem(str(ROOT / "data/easy.kttsp"))

WAITS = [0, 0.5, 1, 2, 3, 5, 8]
TOFS = [0.5, 1, 1.5, 2, 3, 5, 8, 13]

WIDTH = 60
PARENTS = 36
GOAL = 29

CONFIG = {
    "waits": WAITS,
    "tofs": TOFS,
    "beam_width": WIDTH,
    "max_parents": PARENTS,
    "goal": GOAL,
    "local_refinements": 2
}

def sig(state):
    payload = [
        state["route"],
        round(float(state["current_time"]), 7),
        state["special_used"]
    ]
    text = json.dumps(payload)
    return hashlib.sha256(
        text.encode()
    ).hexdigest()[:20]

def save(path, report):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(report, separators=(",", ":")))
    temp.replace(path)

def valid_last(state, level):
    try:
        route = state["route"]
        transfers = state["transfers"]

        if (
            len(route) != level
            or len(set(route)) != level
            or len(transfers) != level - 1
        ):
            return False

        if not (
            0 <= state["special_used"] <= P.n_exceptions
            and 0 <= state["current_time"] <= P.max_time
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
            or abs(start + tof - state["current_time"]) > 1e-5
        ):
            return False

        dv = float(P.compute_transfer(
            route[-2], route[-1], start, tof
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

def choose_parents(states):
    selected = []
    seen = set()
    end_counts = defaultdict(int)

    # Ruajme me shume rruge qe kane SPECIAL te lira.
    for used, quota in [(3, 12), (4, 27), (5, 9)]:
        group = sorted(
            [
                s for s in states
                if s["special_used"] == used
            ],
            key=lambda s: s["current_time"]
        )

        count = 0

        for state in group:
            if count >= quota or len(selected) >= PARENTS:
                break

            key = sig(state)
            end = state["route"][-1]

            if (
                key in seen
                or end_counts[(used, end)] >= 3
            ):
                continue

            selected.append(state)
            seen.add(key)
            end_counts[(used, end)] += 1
            count += 1

        for state in group:
            if count >= quota or len(selected) >= PARENTS:
                break

            key = sig(state)

            if key not in seen:
                selected.append(state)
                seen.add(key)
                count += 1

    # Plotesojme vendet e lira.
    for state in sorted(
        states,
        key=lambda s: (
            s["special_used"],
            s["current_time"]
        )
    ):
        if len(selected) >= PARENTS:
            break

        key = sig(state)

        if key not in seen:
            selected.append(state)
            seen.add(key)

    return selected

def expand(parent):
    route = parent["route"]
    visited = set(route)
    current = route[-1]

    day = float(parent["current_time"])
    used = int(parent["special_used"])

    remaining = P.n_tomatoes - len(route)
    average = (P.max_time - day) / remaining
    max_step = min(18.0, max(9.0, 4.5 * average))

    normals = []
    specials = []
    near = []
    evaluations = 0

    for target in range(P.n_tomatoes):
        if target in visited:
            continue

        best_normal = None
        best_special = None
        closest = None

        for wait in WAITS:
            for tof in TOFS:

                if (
                    wait + tof > max_step
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
                    if (
                        best_normal is None
                        or (move["arrival"], dv) <
                        (
                            best_normal["arrival"],
                            best_normal["dv"]
                        )
                    ):
                        best_normal = move

                elif (
                    dv < P.exception_threshold
                    and used < P.n_exceptions
                ):
                    if (
                        best_special is None
                        or (move["arrival"], dv) <
                        (
                            best_special["arrival"],
                            best_special["dv"]
                        )
                    ):
                        best_special = move

        if (
            best_normal is None
            and closest is not None
            and 100 <= closest["dv"] < 145
        ):
            near.append((
                closest["dv"],
                target,
                closest
            ))

        if best_normal is not None:
            normals.append(best_normal)

        if best_special is not None:
            specials.append(best_special)

    # Provojme optimizim lokal per 2 raste
    # qe jane afer kufirit NORMAL.
    for _, target, closest in sorted(near)[:2]:

        def cost(x):
            wait = float(x[0])
            tof = float(x[1])

            if (
                wait < 0
                or tof < 0.25
                or wait + tof > max_step
                or day + wait + tof > P.max_time
            ):
                return 100000.0 + abs(wait) + abs(tof)

            try:
                dv = float(P.compute_transfer(
                    current, target, day + wait, tof
                ))
            except Exception:
                return 100000.0

            return dv if math.isfinite(dv) else 100000.0

        initial_wait = closest["start_time"] - day
        initial_tof = closest["tof"]

        try:
            result = minimize(
                cost,
                [initial_wait, initial_tof],
                method="Nelder-Mead",
                options={
                    "maxfev": 70,
                    "maxiter": 50,
                    "xatol": 0.02,
                    "fatol": 0.05
                }
            )

            wait = float(result.x[0])
            tof = float(result.x[1])

            if (
                wait >= 0
                and tof >= 0.25
                and wait + tof <= max_step
                and day + wait + tof <= P.max_time
            ):
                dv = float(P.compute_transfer(
                    current, target, day + wait, tof
                ))

                if math.isfinite(dv) and dv < P.dv_threshold:
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
    included = set()

    for status, moves, limit in [
        ("NORMAL", normals, 8),
        ("SPECIAL", specials, 3)
    ]:
        count = 0

        for move in moves:
            key = (move["to"], status)

            if key in included:
                continue

            included.add(key)

            transfer = {
                k: move[k]
                for k in [
                    "from", "to", "start_time",
                    "tof", "dv"
                ]
            }
            transfer["status"] = status

            children.append({
                "route": route + [move["to"]],
                "current_time": move["arrival"],
                "special_used": (
                    used + (status == "SPECIAL")
                ),
                "transfers": (
                    parent["transfers"] + [transfer]
                )
            })

            count += 1

            if count >= limit:
                break

    info = {
        "end": current,
        "day": day,
        "special_used": used,
        "normal_options": len(normals),
        "special_options": len(specials),
        "near_normal": len(near),
        "evaluations": evaluations
    }

    return children, info

def form_frontier(parents):
    all_children = [
        child
        for parent in parents.values()
        for child in parent["children"]
    ]

    unique = {}

    for state in all_children:
        key = (
            tuple(sorted(state["route"])),
            state["route"][-1],
            state["special_used"],
            round(state["current_time"] * 2) / 2
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
            s["current_time"] + 13 * s["special_used"],
            s["current_time"]
        )
    )

    frontier = []
    seen = set()
    ends = defaultdict(int)

    for used, quota in [(3, 12), (4, 42), (5, 18)]:
        count = 0

        for state in ranked:
            if count >= quota or len(frontier) >= WIDTH:
                break

            if state["special_used"] != used:
                continue

            key = sig(state)
            end = state["route"][-1]

            if key in seen or ends[end] >= 5:
                continue

            frontier.append(state)
            seen.add(key)
            ends[end] += 1
            count += 1

    for state in ranked:
        if len(frontier) >= WIDTH:
            break

        key = sig(state)

        if key not in seen:
            frontier.append(state)
            seen.add(key)

    return frontier, len(unique)

print("KERKIMI AUTOMATIK: 23 DERI NE 29")
print("=" * 60, flush=True)

for target_level in range(24, GOAL + 1):

    if target_level == 24:
        previous_path = SOURCE
    else:
        previous_path = (
            RESULTS /
            f"beam_v9_auto_level_{target_level - 1}.json"
        )

    if not previous_path.exists():
        print("Mungon niveli paraprak.", flush=True)
        break

    previous = json.loads(previous_path.read_text())

    if (
        target_level > 24
        and not previous.get("complete")
    ):
        print("Niveli paraprak nuk ka perfunduar.")
        break

    states = previous["frontier"]

    if not states:
        print(
            "Nuk ka rruge per nivelin",
            target_level,
            flush=True
        )
        break

    output = RESULTS / (
        f"beam_v9_auto_level_{target_level}.json"
    )

    if output.exists():
        report = json.loads(output.read_text())

        if report.get("complete"):
            print(
                f"Niveli {target_level}: "
                f"nga checkpoint "
                f"({len(report['frontier'])} rruge)",
                flush=True
            )
            continue

    valid = [
        state
        for state in states
        if valid_last(state, target_level - 1)
    ]

    print(
        f"\nNiveli {target_level}: "
        f"{len(valid)}/{len(states)} "
        f"rruge kaluan kontrollin",
        flush=True
    )

    if not valid:
        print("Nuk ka rruge te vlefshme.", flush=True)
        break

    selected = choose_parents(valid)
    seed_ids = [sig(state) for state in selected]

    if output.exists():
        report = json.loads(output.read_text())

        assert report["config"] == CONFIG
        assert report["seed_ids"] == seed_ids
    else:
        report = {
            "config": CONFIG,
            "seed_ids": seed_ids,
            "parents": {},
            "frontier": [],
            "complete": False
        }
        save(output, report)

    print(
        f"Prinder per zgjerim: {len(selected)}",
        flush=True
    )

    for index, state in enumerate(selected, 1):
        sid = sig(state)

        if sid in report["parents"]:
            continue

        children, info = expand(state)

        report["parents"][sid] = {
            **info,
            "children": children
        }

        save(output, report)

        print(
            f"[{index}/{len(selected)}] "
            f"Fund {info['end']} | "
            f"Dita {info['day']:.2f} | "
            f"SPECIAL {info['special_used']}/5 | "
            f"NORMAL {info['normal_options']} | "
            f"SPECIAL {info['special_options']} | "
            f"Femije {len(children)}",
            flush=True
        )

    frontier, unique_count = form_frontier(
        report["parents"]
    )

    report["frontier"] = frontier
    report["unique_children"] = unique_count
    report["complete"] = True

    save(output, report)

    print(
        f"\nNIVELI {target_level}/49 PERFUNDOI",
        flush=True
    )
    print("Rruge unike:", unique_count, flush=True)
    print("Frontier:", len(frontier), flush=True)

    for used in range(6):
        group = [
            state
            for state in frontier
            if state["special_used"] == used
        ]

        if group:
            earliest = min(
                state["current_time"]
                for state in group
            )

            print(
                f"SPECIAL {used}/5: "
                f"{len(group)} rruge | "
                f"Dita minimale {earliest:.2f}",
                flush=True
            )

    if not frontier:
        break

print("\nKERKIMI I DISPONUESHEM PERFUNDOI!", flush=True)
