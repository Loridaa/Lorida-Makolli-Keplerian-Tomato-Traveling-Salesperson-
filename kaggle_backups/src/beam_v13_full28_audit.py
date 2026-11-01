
import hashlib
import json
import math
import time
from pathlib import Path
from collections import defaultdict
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_v13_full28_seed.json"
OUTPUT = RESULTS / "beam_v13_full28_checkpoint.json"

P = TomatoProblem(str(ROOT / "data/easy.kttsp"))

GOAL = 29
BEAM_WIDTH = 180
MAX_PARENTS = 60
WORK_LIMIT = 130.0

WAITS = [
    0, .25, .5, 1, 2, 3, 5, 8, 12, 18, 26, 36
]

TOFS = [
    .25, .5, .75, 1, 1.5, 2, 3,
    5, 8, 13, 18, 24, 30, 36
]

START = time.monotonic()


def fingerprint(state):
    raw = json.dumps(
        state,
        sort_keys=True,
        separators=(",", ":")
    )

    return hashlib.sha256(
        raw.encode()
    ).hexdigest()[:24]


def save(data):
    temp = OUTPUT.with_suffix(".tmp")

    temp.write_text(
        json.dumps(data, separators=(",", ":")),
        encoding="utf-8"
    )

    temp.replace(OUTPUT)


def verify(state):
    try:
        route = state["route"]
        transfers = state["transfers"]

        if (
            not route
            or route[0] != 0
            or len(route) != len(set(route))
            or len(route) > P.n_tomatoes
            or len(transfers) != len(route) - 1
        ):
            return False

        day = 0.0
        used = 0

        for index, tr in enumerate(transfers):
            depart = float(tr["start_time"])
            tof = float(tr["tof"])

            if (
                tr["from"] != route[index]
                or tr["to"] != route[index + 1]
                or depart < day - 1e-6
                or tof <= 0
                or depart + tof > P.max_time + 1e-6
            ):
                return False

            dv = float(P.compute_transfer(
                route[index],
                route[index + 1],
                depart,
                tof
            ))

            if (
                not math.isfinite(dv)
                or abs(dv - float(tr["dv"])) > .1
            ):
                return False

            if dv < P.dv_threshold:
                status = "NORMAL"

            elif dv < P.exception_threshold:
                status = "SPECIAL"
                used += 1

            else:
                return False

            if tr["status"] != status:
                return False

            day = depart + tof

        return (
            used == state["special_used"]
            and used <= P.n_exceptions
            and abs(
                day - state["current_time"]
            ) < 1e-5
        )

    except Exception:
        return False


def make_child(state, move):
    transfer = {
        "from": state["route"][-1],
        "to": move["to"],
        "start_time": move["start"],
        "tof": move["tof"],
        "dv": move["dv"],
        "status": move["status"]
    }

    return {
        "route": state["route"] + [move["to"]],
        "transfers": (
            state["transfers"] + [transfer]
        ),
        "current_time": (
            move["start"] + move["tof"]
        ),
        "special_used": (
            state["special_used"]
            + int(move["status"] == "SPECIAL")
        )
    }


def select_diverse(states, limit, quotas, max_per_end):
    ranked = sorted(
        states,
        key=lambda s: (
            s["current_time"]
            + 10 * s["special_used"],
            s["special_used"],
            s["current_time"]
        )
    )

    selected = []
    seen = set()
    ends = defaultdict(int)

    for used, quota in quotas.items():
        taken = 0

        for state in ranked:
            if (
                len(selected) >= limit
                or taken >= quota
            ):
                break

            identity = fingerprint(state)
            end = state["route"][-1]

            if (
                state["special_used"] != used
                or identity in seen
                or ends[end] >= max_per_end
            ):
                continue

            selected.append(state)
            seen.add(identity)
            ends[end] += 1
            taken += 1

    for state in ranked:
        if len(selected) >= limit:
            break

        identity = fingerprint(state)

        if identity in seen:
            continue

        if ends[state["route"][-1]] >= max_per_end:
            continue

        selected.append(state)
        seen.add(identity)
        ends[state["route"][-1]] += 1

    # Plotesojme vendet e mbetura
    # vetem nese eshte e nevojshme.
    for state in ranked:
        if len(selected) >= limit:
            break

        identity = fingerprint(state)

        if identity not in seen:
            selected.append(state)
            seen.add(identity)

    return selected


def expand(state):
    current = state["route"][-1]
    day = float(state["current_time"])
    used = int(state["special_used"])
    visited = set(state["route"])

    normal_moves = []
    special_moves = []

    tested = 0

    for target in range(P.n_tomatoes):
        if target in visited:
            continue

        options = {
            "NORMAL": [],
            "SPECIAL": []
        }

        for wait in WAITS:
            for tof in TOFS:
                depart = day + wait

                if depart + tof > P.max_time:
                    continue

                try:
                    dv = float(P.compute_transfer(
                        current,
                        target,
                        depart,
                        tof
                    ))
                    tested += 1
                except Exception:
                    continue

                if not math.isfinite(dv):
                    continue

                if dv < P.dv_threshold:
                    status = "NORMAL"

                elif (
                    dv < P.exception_threshold
                    and used < P.n_exceptions
                ):
                    status = "SPECIAL"

                else:
                    continue

                options[status].append({
                    "to": target,
                    "start": depart,
                    "tof": tof,
                    "dv": dv,
                    "status": status
                })

        for status in ["NORMAL", "SPECIAL"]:
            group = options[status]

            if not group:
                continue

            group.sort(
                key=lambda m: (
                    m["start"] + m["tof"],
                    m["dv"]
                )
            )

            earliest = group[0]
            chosen = [earliest]

            # Ruajme nje alternative me Delta-V
            # me te ulet brenda 8 diteve.
            nearby = [
                m for m in group
                if m["start"] + m["tof"]
                <= earliest["start"] + earliest["tof"] + 8
            ]

            low_dv = min(
                nearby,
                key=lambda m: (
                    m["dv"],
                    m["start"] + m["tof"]
                )
            )

            if (
                abs(
                    low_dv["start"] + low_dv["tof"]
                    - earliest["start"] - earliest["tof"]
                ) >= .25
            ):
                chosen.append(low_dv)

            if status == "NORMAL":
                normal_moves.extend(chosen)
            else:
                special_moves.extend(chosen)

    normal_moves.sort(
        key=lambda m: (
            m["start"] + m["tof"],
            m["dv"]
        )
    )

    special_moves.sort(
        key=lambda m: (
            m["start"] + m["tof"],
            m["dv"]
        )
    )

    children = [
        make_child(state, move)
        for move in (
            normal_moves[:20]
            + special_moves[:10]
        )
    ]

    return children, tested


def form_frontier(children):
    unique = {}

    for state in children:
        key = (
            tuple(sorted(state["route"])),
            state["route"][-1],
            state["special_used"],
            round(state["current_time"] * 4)
        )

        if (
            key not in unique
            or state["current_time"]
            < unique[key]["current_time"]
        ):
            unique[key] = state

    selected = select_diverse(
        list(unique.values()),
        BEAM_WIDTH,
        {2: 8, 3: 35, 4: 36, 5: 17},
        max_per_end=9
    )

    verified = [
        state
        for state in selected
        if verify(state)
    ]

    return verified, len(unique)


raw = SOURCE.read_bytes()

source_hash = hashlib.sha256(
    raw
).hexdigest()

initial = json.loads(raw)["frontier"]

assert initial, "Nuk ka pika fillestare."
assert all(
    len(s["route"]) == 28
    for s in initial
)

print("FAZA 1 - KONTROLLI I BURIMIT")
print("=" * 65, flush=True)

print(
    "Rruge fillestare:",
    len(initial),
    flush=True
)

verified_initial = [
    s for s in initial if verify(s)
]

print(
    "Rruge te verifikuara:",
    len(verified_initial),
    flush=True
)

assert len(verified_initial) == len(initial), (
    "Disa rruge fillestare nuk kaluan verifikimin."
)

CONFIG = {
    "source_sha256": source_hash,
    "goal": GOAL,
    "beam_width": BEAM_WIDTH,
    "max_parents": MAX_PARENTS,
    "waits": WAITS,
    "tofs": TOFS,
    "version": 1
}

if OUTPUT.exists():
    report = json.loads(
        OUTPUT.read_text(encoding="utf-8")
    )

    assert report["config"] == CONFIG, (
        "Checkpoint ka konfigurim tjeter."
    )

else:
    report = {
        "config": CONFIG,
        "level": 28,
        "frontier": verified_initial,
        "pending": None,
        "stopped": False,
        "history": []
    }

    save(report)


print("\nFAZA 2 - KERKIMI DREJT NIVELIT 32")
print("=" * 65, flush=True)

while (
    time.monotonic() - START < WORK_LIMIT
    and not report["stopped"]
    and report["level"] < GOAL
):

    level = report["level"]
    frontier = report["frontier"]

    selected = select_diverse(
        frontier,
        MAX_PARENTS,
        {2: 4, 3: 20, 4: 14, 5: 6},
        max_per_end=5
    )

    selected_ids = [
        fingerprint(s)
        for s in selected
    ]

    if report["pending"] is None:
        report["pending"] = {
            "target_level": level + 1,
            "selected_ids": selected_ids,
            "parents": {}
        }

        save(report)

    pending = report["pending"]

    assert pending["target_level"] == level + 1
    assert pending["selected_ids"] == selected_ids

    print(
        f"\nNIVELI {level} -> {level + 1} | "
        f"Prinder {len(selected)}",
        flush=True
    )

    for index, state in enumerate(selected, 1):

        if time.monotonic() - START >= WORK_LIMIT:
            break

        sid = fingerprint(state)

        if sid in pending["parents"]:
            continue

        children, tested = expand(state)

        pending["parents"][sid] = {
            "children": children,
            "tested": tested
        }

        save(report)

        print(
            f"[{index}/{len(selected)}] "
            f"Fund {state['route'][-1]} | "
            f"Dita {state['current_time']:.2f} | "
            f"SPECIAL {state['special_used']}/5 | "
            f"Femije {len(children)}",
            flush=True
        )

    if len(pending["parents"]) < len(selected):
        print(
            "Checkpoint i pjesshem:",
            len(pending["parents"]),
            "/",
            len(selected),
            flush=True
        )
        break

    children = [
        child
        for result in pending["parents"].values()
        for child in result["children"]
    ]

    next_frontier, unique_count = form_frontier(
        children
    )

    if not next_frontier:
        report["stopped"] = True
        report["pending"] = None

        report["history"].append({
            "attempted_level": level + 1,
            "unique": unique_count,
            "verified": 0
        })

        save(report)

        print(
            f"Nuk u gjet vazhdim ne nivelin {level + 1}.",
            flush=True
        )
        break

    report["level"] = level + 1
    report["frontier"] = next_frontier
    report["pending"] = None

    report["history"].append({
        "level": report["level"],
        "unique": unique_count,
        "verified": len(next_frontier)
    })

    save(report)

    snapshot = RESULTS / (
        f"beam_v13_full28_level_{report['level']}.json"
    )

    temp = snapshot.with_suffix(".tmp")

    temp.write_text(
        json.dumps({
            "level": report["level"],
            "verified": True,
            "frontier": next_frontier
        }, separators=(",", ":"))
    )

    temp.replace(snapshot)

    print(
        f"\nARRITEM NIVELIN {report['level']}/49",
        flush=True
    )

    print(
        "Rruge unike:", unique_count,
        flush=True
    )

    print(
        "Rruge te verifikuara:",
        len(next_frontier),
        flush=True
    )

    for used in range(6):
        group = [
            s for s in next_frontier
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


print("\nPERMBLEDHJA E HAPIT 131")
print("=" * 65, flush=True)

frontier = report["frontier"]

verified_final = sum(
    verify(s) for s in frontier
)

print(
    "Niveli real:", report["level"], "/49",
    flush=True
)

print(
    "Rruge ne frontier:", len(frontier),
    flush=True
)

print(
    "Te verifikuara:",
    verified_final, "/", len(frontier),
    flush=True
)

print(
    "Ndalur:", report["stopped"],
    flush=True
)

print(
    "Checkpoint i pjesshem:",
    report["pending"] is not None,
    flush=True
)

if report["pending"] is not None:
    pending = report["pending"]

    print(
        "Prinder te analizuar:",
        len(pending["parents"]),
        "/",
        len(pending["selected_ids"]),
        flush=True
    )

for used in range(6):
    group = [
        s for s in frontier
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

print("\nREZULTATI PERFUNDIMTAR", flush=True)

if (
    report["level"] >= 32
    and frontier
    and verified_final == len(frontier)
):
    print(
        "REKORD I RI I VERIFIKUAR: 32/49!",
        flush=True
    )
else:
    print(
        "Rekordi ekzistues: 31/49",
        flush=True
    )

if (
    not report["stopped"]
    and report["level"] < GOAL
):
    print(
        "Kerkimi vazhdon. "
        "Ekzekuto perseri te njejten qelize.",
        flush=True
    )

print("\nHAPI 132 PERFUNDOI!", flush=True)
