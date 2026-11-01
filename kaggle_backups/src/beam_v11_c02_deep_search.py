
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

SOURCE = RESULTS / "beam_v10_special_branch_audit.json"
OUTPUT = RESULTS / "beam_v11_c02_to32_checkpoint.json"

P = TomatoProblem(str(ROOT / "data/easy.kttsp"))

WAITS = [0, 0.5, 1, 2, 4, 8, 12]
TOFS = [0.5, 1, 1.5, 2, 3, 5, 8, 13, 18, 24]

BEAM_WIDTH = 42
MAX_PARENTS = 18
GOAL = 32
WORK_LIMIT = 140.0

START = time.monotonic()

SOURCE_HASH = hashlib.sha256(
    SOURCE.read_bytes()
).hexdigest()

data = json.loads(SOURCE.read_text())
candidate_prefixes = data["candidate_prefixes"]

BASE_NAMES = {
    14: "A14",
    12: "B12",
    2: "C02"
}

CONFIG = {
    "source_hash": SOURCE_HASH,
    "waits": WAITS,
    "tofs": TOFS,
    "beam_width": BEAM_WIDTH,
    "max_parents": MAX_PARENTS,
    "goal": GOAL,
    "version": 1
}

def identity(state):
    payload = [
        state["route"],
        round(float(state["current_time"]), 7),
        int(state["special_used"])
    ]
    return hashlib.sha256(
        json.dumps(payload).encode()
    ).hexdigest()[:20]

def verify_full(state):
    try:
        route = state["route"]
        transfers = state["transfers"]

        if (
            len(route) < 2
            or len(route) > P.n_tomatoes
            or route[0] != 0
            or len(route) != len(set(route))
            or len(transfers) != len(route) - 1
            or any(
                type(t) is not int
                or t < 0 or t >= P.n_tomatoes
                for t in route
            )
        ):
            return False

        day = 0.0
        specials = 0

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
                route[i], route[i + 1], depart, tof
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
                specials += 1
            else:
                return False

            if status != tr["status"]:
                return False

            day = depart + tof

        return (
            abs(day - state["current_time"]) < 1e-5
            and specials == state["special_used"]
            and specials <= P.n_exceptions
        )

    except Exception:
        return False

def verify_last(state):
    try:
        route = state["route"]
        transfers = state["transfers"]

        if (
            len(route) != len(set(route))
            or len(transfers) != len(route) - 1
            or state["special_used"] > P.n_exceptions
            or state["current_time"] > P.max_time
        ):
            return False

        tr = transfers[-1]
        depart = float(tr["start_time"])
        tof = float(tr["tof"])

        if (
            tr["from"] != route[-2]
            or tr["to"] != route[-1]
            or tof <= 0
            or abs(
                depart + tof - state["current_time"]
            ) > 1e-5
        ):
            return False

        dv = float(P.compute_transfer(
            route[-2], route[-1], depart, tof
        ))

        status = (
            "NORMAL" if dv < P.dv_threshold
            else "SPECIAL" if dv < P.exception_threshold
            else "INVALID"
        )

        return (
            math.isfinite(dv)
            and abs(dv - float(tr["dv"])) < 0.1
            and status == tr["status"]
        )

    except Exception:
        return False

def save(report):
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(report, separators=(",", ":"))
    )
    temporary.replace(OUTPUT)

def select_states(states, maximum):
    ranked = sorted(
        states,
        key=lambda s: (
            s["current_time"] + 12 * s["special_used"],
            s["current_time"]
        )
    )

    quotas = {
        0: 7,
        1: 9,
        2: 12,
        3: 14,
        4: 12,
        5: 5
    }

    selected = []
    seen = set()
    ends = defaultdict(int)

    for used in range(6):
        count = 0

        for state in ranked:
            if (
                len(selected) >= maximum
                or count >= quotas[used]
            ):
                break

            sid = identity(state)
            end = state["route"][-1]

            if (
                state["special_used"] != used
                or sid in seen
                or ends[end] >= 4
            ):
                continue

            selected.append(state)
            seen.add(sid)
            ends[end] += 1
            count += 1

    for state in ranked:
        if len(selected) >= maximum:
            break

        sid = identity(state)

        if sid not in seen:
            selected.append(state)
            seen.add(sid)

    return selected

def expand(parent):
    route = parent["route"]
    visited = set(route)

    current = route[-1]
    day = float(parent["current_time"])
    used = int(parent["special_used"])

    remaining = P.n_tomatoes - len(route)
    remaining_days = P.max_time - day

    if remaining <= 0 or remaining_days <= 0:
        return [], {
            "normal": 0,
            "special": 0,
            "evaluations": 0
        }

    average = remaining_days / remaining

    max_extra = min(
        remaining_days,
        28.0,
        max(10.0, average * 6.5)
    )

    normals = []
    specials = []
    near_normals = []
    evaluations = 0

    for target in range(P.n_tomatoes):
        if target in visited:
            continue

        best_normal = None
        best_special = None
        closest = None

        for wait in WAITS:
            for tof in TOFS:
                if wait + tof > max_extra:
                    continue

                depart = day + wait
                arrival = depart + tof

                if arrival > P.max_time:
                    continue

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

                if closest is None or dv < closest["dv"]:
                    closest = move

                if dv < P.dv_threshold:
                    if (
                        best_normal is None
                        or (arrival, dv) <
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
                        or (arrival, dv) <
                        (
                            best_special["arrival"],
                            best_special["dv"]
                        )
                    ):
                        best_special = move

        if best_normal is not None:
            normals.append(best_normal)

        if best_special is not None:
            specials.append(best_special)

        if (
            best_normal is None
            and closest is not None
            and 100 <= closest["dv"] < 180
        ):
            near_normals.append(closest)

    # Optimizim lokal per transferime
    # qe jane afer kufirit NORMAL.
    for initial in sorted(
        near_normals,
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
                return 1e6

            try:
                dv = float(P.compute_transfer(
                    current,
                    target,
                    day + wait,
                    tof
                ))
            except Exception:
                return 1e6

            return dv if math.isfinite(dv) else 1e6

        initial_guess = [
            max(0.001, initial["start_time"] - day),
            initial["tof"]
        ]

        try:
            result = minimize(
                objective,
                initial_guess,
                method="Nelder-Mead",
                options={
                    "maxiter": 45,
                    "maxfev": 65,
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
        ("NORMAL", normals, 7),
        ("SPECIAL", specials, 4)
    ]:
        chosen_targets = set()

        for move in moves:
            if len(chosen_targets) >= limit:
                break

            target = move["to"]

            if target in chosen_targets:
                continue

            chosen_targets.add(target)

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
                "route": route + [target],
                "transfers": (
                    parent["transfers"] + [transfer]
                ),
                "current_time": move["arrival"],
                "special_used": (
                    used + int(status == "SPECIAL")
                )
            })

    return children, {
        "normal": len(normals),
        "special": len(specials),
        "evaluations": evaluations
    }

def form_frontier(parent_results):
    unique = {}

    for result in parent_results.values():
        for state in result["children"]:

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

    frontier = select_states(
        list(unique.values()),
        BEAM_WIDTH
    )

    return frontier, len(unique)

# FAZA 1 - Lexojme 3 prefikset e Hapit 119.
bases = {}

for state in candidate_prefixes:
    step = state.get("replaced_special_step")

    if step not in BASE_NAMES:
        continue

    name = BASE_NAMES[step]

    if not verify_full(state):
        raise ValueError(
            f"Prefiksi {name} nuk kaloi verifikimin."
        )

    bases[name] = {
        "route": state["route"],
        "transfers": state["transfers"],
        "current_time": state["current_time"],
        "special_used": state["special_used"]
    }

assert len(bases) == 3, (
    "Nuk u gjeten te 3 prefikset e Hapit 119."
)

if OUTPUT.exists():
    report = json.loads(OUTPUT.read_text())

    assert report["config"] == CONFIG, (
        "Checkpoint ekzistues ka konfigurim tjeter."
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
            for name, state in bases.items()
        }
    }

    save(report)

print("FAZA 1 - TRI PIKE-NISJET")
print("=" * 65, flush=True)

for name, lane in report["lanes"].items():
    best = min(
        lane["frontier"],
        key=lambda s: s["current_time"]
    )

    print(
        f"{name} | "
        f"Niveli {lane['level']}/49 | "
        f"Dita {best['current_time']:.2f} | "
        f"SPECIAL {best['special_used']}/5 | "
        f"Rruge {len(lane['frontier'])}",
        flush=True
    )

print("\nFAZA 2 - BEAM SEARCH V11")
print("=" * 65, flush=True)

# Secila strategji avancon nga nje nivel
# ne cdo rrotullim.
while time.monotonic() - START < WORK_LIMIT:

    progress = False

    for name in ["A14", "B12", "C02"]:

        if time.monotonic() - START >= WORK_LIMIT:
            break

        lane = report["lanes"][name]

        if lane["stopped"] or lane["level"] >= GOAL:
            continue

        selected = select_states(
            [
                state
                for state in lane["frontier"]
                if verify_last(state)
            ],
            MAX_PARENTS
        )

        if not selected:
            lane["stopped"] = True
            save(report)
            continue

        seed_ids = [identity(s) for s in selected]

        if lane["pending"] is None:
            lane["pending"] = {
                "target_level": lane["level"] + 1,
                "seed_ids": seed_ids,
                "parents": {}
            }
            save(report)

        pending = lane["pending"]

        assert pending["target_level"] == lane["level"] + 1
        assert pending["seed_ids"] == seed_ids

        print(
            f"\n{name}: {lane['level']} -> "
            f"{lane['level'] + 1} | "
            f"Prinder {len(selected)}",
            flush=True
        )

        for index, state in enumerate(selected, 1):

            sid = identity(state)

            if sid in pending["parents"]:
                continue

            if time.monotonic() - START >= WORK_LIMIT:
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
                f"NORMAL {info['normal']} | "
                f"SPECIAL {info['special']} | "
                f"Femije {len(children)}",
                flush=True
            )

        if len(pending["parents"]) < len(selected):
            print(
                name,
                "- checkpoint i pjesshem.",
                flush=True
            )
            break

        new_frontier, unique_count = form_frontier(
            pending["parents"]
        )

        attempted_level = lane["level"] + 1

        if not new_frontier:
            lane["stopped"] = True
            lane["pending"] = None

            lane["history"].append({
                "attempted_level": attempted_level,
                "success": False,
                "routes": 0
            })

            save(report)

            print(
                f"{name}: Nuk u gjet rruge "
                f"per nivelin {attempted_level}.",
                flush=True
            )
            continue

        # Ruajme nivelin e ri vetem nese
        # ka te pakten nje rruge reale.
        lane["level"] = attempted_level
        lane["frontier"] = new_frontier
        lane["pending"] = None

        lane["history"].append({
            "level": attempted_level,
            "success": True,
            "unique": unique_count,
            "routes": len(new_frontier)
        })

        save(report)

        # Snapshot i vecante per rikuperim.
        snapshot = RESULTS / (
            f"beam_v11_{name}_level_{attempted_level}.json"
        )

        snapshot.write_text(
            json.dumps({
                "strategy": name,
                "level": attempted_level,
                "frontier": new_frontier
            }, separators=(",", ":"))
        )

        progress = True

        print(
            f"\n{name} - NIVELI "
            f"{attempted_level}/49",
            flush=True
        )

        print(
            "Rruge unike:", unique_count,
            flush=True
        )

        print(
            "Frontier:", len(new_frontier),
            flush=True
        )

        for used in range(6):
            group = [
                s for s in new_frontier
                if s["special_used"] == used
            ]

            if group:
                print(
                    f"SPECIAL {used}/5 | "
                    f"{len(group)} rruge | "
                    f"Dita minimale "
                    f"{min(s['current_time'] for s in group):.2f}",
                    flush=True
                )

    if not progress:
        break

print("\nFAZA 3 - KONTROLLI FINAL")
print("=" * 65, flush=True)

for name, lane in report["lanes"].items():
    frontier = lane["frontier"]

    verified = [
        s for s in frontier
        if verify_full(s)
    ]

    print(
        f"{name} | "
        f"Niveli {lane['level']}/49 | "
        f"Rruge {len(frontier)} | "
        f"Te verifikuara {len(verified)} | "
        f"Ndalur {lane['stopped']} | "
        f"Checkpoint {lane['pending'] is not None}",
        flush=True
    )

    if verified:
        best = min(
            verified,
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

print("\nHAPI 120 - KERKIMI PERFUNDOI!", flush=True)
