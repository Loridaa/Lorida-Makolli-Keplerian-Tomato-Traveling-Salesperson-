
import json
import hashlib
import math
import time
from pathlib import Path
from collections import defaultdict
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_v13_early_level_20.json"
OUT = RESULTS / "beam_v16_diverse_20to24_checkpoint.json"

P = TomatoProblem(str(ROOT / "data/easy.kttsp"))

START = time.monotonic()
LIMIT = 115.0

GOAL = 24
BEAM = 100
MAX_PARENTS = 55

WAITS = [
    0, .25, .5, 1, 2, 3,
    5, 8, 12, 20, 32
]

TOFS = [
    .25, .5, .75, 1, 1.5, 2,
    3, 5, 8, 13, 20, 30
]


def uid(state):
    raw = json.dumps(
        state,
        sort_keys=True,
        separators=(",", ":")
    )
    return hashlib.sha256(
        raw.encode()
    ).hexdigest()[:24]


def atomic(path, obj):
    temp = path.with_suffix(".tmp")

    temp.write_text(
        json.dumps(obj, separators=(",", ":")),
        encoding="utf-8"
    )

    temp.replace(path)


def verify(s):
    try:
        route = s["route"]
        transfers = s["transfers"]

        if (
            not route
            or route[0] != 0
            or len(route) != len(set(route))
            or len(transfers) != len(route) - 1
            or len(route) > P.n_tomatoes
        ):
            return False

        day = 0.0
        used = 0

        for k, tr in enumerate(transfers):
            depart = float(tr["start_time"])
            tof = float(tr["tof"])

            if (
                tr["from"] != route[k]
                or tr["to"] != route[k + 1]
                or depart < day - 1e-6
                or tof <= 0
                or depart + tof > P.max_time + 1e-6
            ):
                return False

            dv = float(P.compute_transfer(
                route[k],
                route[k + 1],
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
            used == s["special_used"]
            and used <= P.n_exceptions
            and abs(day - s["current_time"]) < 1e-5
        )

    except Exception:
        return False


def expand(parent):
    origin = parent["route"][-1]
    day = float(parent["current_time"])
    visited = set(parent["route"])
    used = parent["special_used"]

    children = []
    tested = 0

    for target in range(P.n_tomatoes):
        if target in visited:
            continue

        best = {
            "NORMAL": None,
            "SPECIAL": None
        }

        for wait in WAITS:
            for tof in TOFS:
                depart = day + wait
                arrival = depart + tof

                if arrival > P.max_time:
                    continue

                try:
                    dv = float(P.compute_transfer(
                        origin,
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

                previous = best[status]

                if (
                    previous is None
                    or (
                        arrival,
                        dv
                    ) < (
                        previous["arrival"],
                        previous["dv"]
                    )
                ):
                    best[status] = {
                        "to": target,
                        "start_time": depart,
                        "tof": tof,
                        "dv": dv,
                        "arrival": arrival,
                        "status": status
                    }

        for move in best.values():
            if move is None:
                continue

            transfer = {
                "from": origin,
                "to": move["to"],
                "start_time": move["start_time"],
                "tof": move["tof"],
                "dv": move["dv"],
                "status": move["status"]
            }

            children.append({
                "route": (
                    parent["route"] + [move["to"]]
                ),
                "transfers": (
                    parent["transfers"] + [transfer]
                ),
                "current_time": move["arrival"],
                "special_used": (
                    used
                    + int(move["status"] == "SPECIAL")
                )
            })

    return children, tested


def select(states, maximum, quotas):
    # Ruajme rrugen me mberritjen me te hershme
    # kur bashkesia e domateve, fundi dhe
    # numri SPECIAL jane te njejte.
    unique = {}

    for s in states:
        key = (
            tuple(sorted(s["route"])),
            s["route"][-1],
            s["special_used"]
        )

        if (
            key not in unique
            or s["current_time"]
            < unique[key]["current_time"]
        ):
            unique[key] = s

    ranked = sorted(
        unique.values(),
        key=lambda s: (
            s["current_time"]
            + 8 * s["special_used"],
            s["special_used"],
            s["current_time"]
        )
    )

    selected = []
    seen = set()
    endings = defaultdict(int)
    tails = defaultdict(int)

    def add(s, relax=False):
        if len(selected) >= maximum:
            return False

        identity = uid(s)
        end = s["route"][-1]
        tail = tuple(s["route"][-3:])

        if identity in seen:
            return False

        if (
            not relax
            and (
                endings[end] >= 10
                or tails[tail] >= 3
            )
        ):
            return False

        selected.append(s)
        seen.add(identity)

        endings[end] += 1
        tails[tail] += 1

        return True

    # U japim vende te vecanta rrugeve
    # qe kane kursyer SPECIAL.
    for used, quota in quotas.items():
        count = 0

        for s in ranked:
            if s["special_used"] != used:
                continue

            if add(s):
                count += 1

            if count >= quota:
                break

    for s in ranked:
        add(s)

    # Plotesojme vendet e mbetura,
    # nese ka ende rruge unike.
    for s in ranked:
        add(s, relax=True)

    return selected, len(unique)


# FAZA 1 - Burimi i verifikuar.
raw = SOURCE.read_bytes()

source_hash = hashlib.sha256(
    raw
).hexdigest()

initial = json.loads(raw)["frontier"]

assert initial, "Nuk ka rruge fillestare."

assert all(
    len(s["route"]) == 20
    and verify(s)
    for s in initial
), "Disa rruge nuk kaluan verifikimin."

CONFIG = {
    "source_hash": source_hash,
    "goal": GOAL,
    "beam": BEAM,
    "max_parents": MAX_PARENTS,
    "waits": WAITS,
    "tofs": TOFS,
    "version": 1
}

if OUT.exists():
    report = json.loads(
        OUT.read_text(encoding="utf-8")
    )

    assert report["config"] == CONFIG, (
        "Checkpoint ka parametra te tjere."
    )

else:
    report = {
        "config": CONFIG,
        "level": 20,
        "frontier": initial,
        "pending": None,
        "stopped": False,
        "history": []
    }

    atomic(OUT, report)


print("FAZA 1 - KONTROLLI I BURIMIT")
print("=" * 65, flush=True)

print(
    "Rruge fillestare te verifikuara:",
    len(initial),
    flush=True
)


# FAZA 2 - Kerkimi me checkpoint.
print("\nFAZA 2 - KERKIMI 20 -> 24")
print("=" * 65, flush=True)

while (
    time.monotonic() - START < LIMIT
    and not report["stopped"]
    and report["level"] < GOAL
):
    frontier = report["frontier"]

    parents, _ = select(
        frontier,
        MAX_PARENTS,
        {
            2: 16,
            3: 26,
            4: 15,
            5: 6
        }
    )

    parent_ids = [
        uid(s) for s in parents
    ]

    if report["pending"] is None:
        report["pending"] = {
            "target": report["level"] + 1,
            "parent_ids": parent_ids,
            "done": {}
        }

        atomic(OUT, report)

    pending = report["pending"]

    assert (
        pending["target"]
        == report["level"] + 1
    )

    assert pending["parent_ids"] == parent_ids

    print(
        f"\nNIVELI {report['level']} "
        f"-> {report['level'] + 1} | "
        f"Prinder {len(parents)}",
        flush=True
    )

    for index, parent in enumerate(parents, 1):
        if time.monotonic() - START >= LIMIT:
            break

        pid = uid(parent)

        if pid in pending["done"]:
            continue

        children, tested = expand(parent)

        pending["done"][pid] = {
            "children": children,
            "tested": tested
        }

        atomic(OUT, report)

        print(
            f"[{index}/{len(parents)}] "
            f"Fund {parent['route'][-1]} | "
            f"Dita {parent['current_time']:.2f} | "
            f"SPECIAL {parent['special_used']}/5 | "
            f"Femije {len(children)}",
            flush=True
        )

    if len(pending["done"]) < len(parents):
        print(
            "Checkpoint i pjesshem:",
            len(pending["done"]),
            "/",
            len(parents),
            flush=True
        )
        break

    children = [
        child
        for entry in pending["done"].values()
        for child in entry["children"]
    ]

    next_states, unique_count = select(
        children,
        BEAM,
        {
            2: 28,
            3: 42,
            4: 22,
            5: 8
        }
    )

    next_states = [
        s for s in next_states
        if verify(s)
    ]

    if not next_states:
        report["stopped"] = True
        report["pending"] = None

        report["history"].append({
            "attempted_level": report["level"] + 1,
            "unique": unique_count,
            "verified": 0
        })

        atomic(OUT, report)

        print(
            "Nuk u gjet vazhdim me "
            "parametrat e ketij hapi.",
            flush=True
        )
        break

    report["level"] += 1
    report["frontier"] = next_states
    report["pending"] = None

    report["history"].append({
        "level": report["level"],
        "unique": unique_count,
        "verified": len(next_states)
    })

    atomic(OUT, report)

    snapshot = RESULTS / (
        f"beam_v16_diverse_level_{report['level']}.json"
    )

    atomic(snapshot, {
        "level": report["level"],
        "verified": True,
        "frontier": next_states
    })

    print(
        f"\nARRITEM NIVELIN "
        f"{report['level']}/49 | "
        f"Rruge unike {unique_count} | "
        f"Rruge te verifikuara {len(next_states)}",
        flush=True
    )

    for used in range(6):
        group = [
            s for s in next_states
            if s["special_used"] == used
        ]

        if group:
            earliest = min(
                s["current_time"]
                for s in group
            )

            print(
                f"SPECIAL {used}/5 | "
                f"Rruge {len(group)} | "
                f"Dita minimale {earliest:.2f}",
                flush=True
            )


# FAZA 3 - Permbledhja.
print("\nPERMBLEDHJA E HAPIT 140")
print("=" * 65, flush=True)

print(
    "Niveli i arritur:",
    report["level"], "/49",
    flush=True
)

print(
    "Rruge ne frontier:",
    len(report["frontier"]),
    flush=True
)

print(
    "Ndalur:",
    report["stopped"],
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
        "Prinder te perfunduar:",
        len(pending["done"]),
        "/",
        len(pending["parent_ids"]),
        flush=True
    )

for used in range(6):
    group = [
        s for s in report["frontier"]
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

print(
    "Rekordi i verifikuar i projektit: 31/49",
    flush=True
)

if (
    not report["stopped"]
    and report["level"] < GOAL
):
    print(
        "Kerkimi i pjesshem: True",
        flush=True
    )
    print(
        "Ekzekuto perseri te njejten qelize.",
        flush=True
    )
else:
    print(
        "Kerkimi i pjesshem: False",
        flush=True
    )

print("HAPI 140 PERFUNDOI!", flush=True)
