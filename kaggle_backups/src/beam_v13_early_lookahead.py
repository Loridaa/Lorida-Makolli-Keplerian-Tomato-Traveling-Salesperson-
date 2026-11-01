
import json
import math
import time
import hashlib
from pathlib import Path
from collections import defaultdict
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

OUT = RESULTS / "beam_v13_early_checkpoint.json"

P = TomatoProblem(str(ROOT / "data/easy.kttsp"))

START = time.monotonic()
LIMIT = 125.0

WAITS = [0, .5, 1, 2, 4, 8, 16, 32]
TOFS = [.5, 1, 2, 3, 5, 8, 13, 20, 30]

SOURCES = [
    ("C02", 18),
    ("C02", 20),
    ("C02", 22),
    ("A14", 22),
    ("A14", 24),
    ("B12", 22),
    ("B12", 24)
]

class Pause(Exception):
    pass

def check_time():
    if time.monotonic() - START >= LIMIT:
        raise Pause()

def uid(state):
    return hashlib.sha256(
        json.dumps(
            state,
            sort_keys=True,
            separators=(",", ":")
        ).encode()
    ).hexdigest()[:24]

def atomic(path, obj):
    temp = path.with_suffix(".tmp")

    temp.write_text(
        json.dumps(obj, separators=(",", ":"))
    )

    temp.replace(path)

def verify(state):
    try:
        route = state["route"]
        transfers = state["transfers"]

        if (
            not route
            or route[0] != 0
            or len(route) != len(set(route))
            or len(transfers) != len(route) - 1
            or len(route) > P.n_tomatoes
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
                route[i],
                route[i + 1],
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
                specials += 1

            else:
                return False

            if status != tr["status"]:
                return False

            day = depart + tof

        return (
            specials == state["special_used"]
            and specials <= P.n_exceptions
            and abs(day - state["current_time"]) < 1e-5
        )

    except Exception:
        return False

def child(parent, move):
    transfer = {
        "from": parent["route"][-1],
        "to": move["target"],
        "start_time": move["start"],
        "tof": move["tof"],
        "dv": move["dv"],
        "status": move["status"]
    }

    return {
        "route": parent["route"] + [move["target"]],
        "transfers": (
            parent["transfers"] + [transfer]
        ),
        "current_time": (
            move["start"] + move["tof"]
        ),
        "special_used": (
            parent["special_used"]
            + int(move["status"] == "SPECIAL")
        )
    }

def expand(state):
    current = state["route"][-1]
    day = float(state["current_time"])
    used = int(state["special_used"])
    visited = set(state["route"])

    moves = []
    evaluations = 0

    for target in range(P.n_tomatoes):
        if target in visited:
            continue

        best = {
            "NORMAL": None,
            "SPECIAL": None
        }

        for wait in WAITS:
            for tof in TOFS:
                check_time()

                depart = day + wait

                if depart + tof > P.max_time:
                    continue

                try:
                    dv = float(P.compute_transfer(
                        current, target, depart, tof
                    ))
                    evaluations += 1
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

                candidate = {
                    "target": target,
                    "start": depart,
                    "tof": tof,
                    "dv": dv,
                    "status": status
                }

                previous = best[status]

                if (
                    previous is None
                    or (
                        depart + tof,
                        dv
                    ) < (
                        previous["start"] + previous["tof"],
                        previous["dv"]
                    )
                ):
                    best[status] = candidate

        for value in best.values():
            if value is not None:
                moves.append(value)

    return moves, evaluations

def select(states, maximum, max_per_end=4):
    # Ruajme edhe rruge me me shume SPECIAL,
    # nese ato arrijne dukshëm me heret.
    ranked = sorted(
        states,
        key=lambda s: (
            s["current_time"] + 12 * s["special_used"],
            s["special_used"],
            s["current_time"]
        )
    )

    result = []
    seen = set()
    counts = defaultdict(int)

    for used, quota in [
        (0, 6),
        (1, 8),
        (2, 10),
        (3, 12),
        (4, 12),
        (5, 8)
    ]:
        taken = 0

        for s in ranked:
            if (
                len(result) >= maximum
                or taken >= quota
            ):
                break

            identity = uid(s)
            end = s["route"][-1]

            if (
                s["special_used"] != used
                or identity in seen
                or counts[end] >= max_per_end
            ):
                continue

            result.append(s)
            seen.add(identity)
            counts[end] += 1
            taken += 1

    for s in ranked:
        if len(result) >= maximum:
            break

        identity = uid(s)

        if identity not in seen:
            result.append(s)
            seen.add(identity)

    return result

# FAZA 1 - Perzgjedhja e pikave te hershme.
source_hashes = {}
seed_records = []
seen_seeds = set()

for name, level in SOURCES:
    path = RESULTS / (
        f"beam_v11_{name}_level_{level}.json"
    )

    if not path.exists():
        print(
            f"SKEDAR I MUNGUAR: {path.name}",
            flush=True
        )
        continue

    raw = path.read_bytes()

    source_hashes[path.name] = hashlib.sha256(
        raw
    ).hexdigest()

    records = json.loads(raw)["frontier"]

    valid = [
        state
        for state in records
        if len(state["route"]) == level
        and verify(state)
    ]

    ranked = sorted(
        valid,
        key=lambda s: (
            s["special_used"],
            s["current_time"]
        )
    )

    chosen = []
    ends = defaultdict(int)

    for state in ranked:
        if len(chosen) >= 3:
            break

        end = state["route"][-1]
        identity = uid(state)

        if (
            identity in seen_seeds
            or ends[end] >= 2
        ):
            continue

        chosen.append(state)
        seen_seeds.add(identity)
        ends[end] += 1

    for state in chosen:
        seed_records.append({
            "source": name,
            "source_level": level,
            "state": state
        })

    print(
        f"{name} nivel {level}: "
        f"{len(chosen)} pika fillestare",
        flush=True
    )

assert seed_records, (
    "Nuk u gjet asnje rruge fillestare."
)

CONFIG = {
    "source_hashes": source_hashes,
    "seed_ids": [
        uid(r["state"])
        for r in seed_records
    ],
    "waits": WAITS,
    "tofs": TOFS,
    "version": 1
}

if OUT.exists():
    report = json.loads(OUT.read_text())

    assert report["config"] == CONFIG, (
        "Checkpoint me konfigurim tjeter."
    )

else:
    report = {
        "config": CONFIG,
        "analyses": {}
    }

    atomic(OUT, report)

def save():
    atomic(OUT, report)

print("\nFAZA 2 - KERKIMI ME DY TRANSFERIME")
print("=" * 65, flush=True)

try:
    for i, item in enumerate(seed_records, 1):
        check_time()

        state = item["state"]
        sid = uid(state)

        if sid in report["analyses"]:
            continue

        first_moves, ev1 = expand(state)

        first_children = [
            child(state, m)
            for m in first_moves
        ]

        # Maksimum 10 variante pas hapit te pare.
        first_selected = select(
            first_children, 10, max_per_end=2
        )

        second_children = []
        ev2 = 0

        for first in first_selected:
            check_time()

            second_moves, count = expand(first)
            ev2 += count

            for move in second_moves:
                candidate = child(first, move)

                if verify(candidate):
                    second_children.append(candidate)

        # Largojme alternativat ekuivalente.
        unique = {}

        for candidate in second_children:
            key = (
                tuple(sorted(candidate["route"])),
                candidate["route"][-1],
                candidate["special_used"],
                round(candidate["current_time"] * 2) / 2
            )

            if (
                key not in unique
                or candidate["current_time"]
                < unique[key]["current_time"]
            ):
                unique[key] = candidate

        best = select(
            list(unique.values()),
            16,
            max_per_end=4
        )

        report["analyses"][sid] = {
            "source": item["source"],
            "source_level": item["source_level"],
            "first_moves": len(first_moves),
            "first_selected": len(first_selected),
            "second_unique": len(unique),
            "evaluations": ev1 + ev2,
            "states": best
        }

        save()

        if best:
            winner = min(
                best,
                key=lambda s: (
                    s["special_used"],
                    s["current_time"]
                )
            )

            print(
                f"[{i}/{len(seed_records)}] "
                f"{item['source']} "
                f"{item['source_level']} -> "
                f"{len(winner['route'])} | "
                f"Rruge {len(best)} | "
                f"Dita {winner['current_time']:.2f} | "
                f"SPECIAL {winner['special_used']}/5",
                flush=True
            )

        else:
            print(
                f"[{i}/{len(seed_records)}] "
                f"{item['source']} "
                f"{item['source_level']} | "
                "Pa vazhdim ne dy hapa",
                flush=True
            )

except Pause:
    print(
        "\nCheckpoint u ruajt. "
        "Kerkimi mund te vazhdoje.",
        flush=True
    )

# FAZA 3 - Ruajme kandidatet e verifikuar.
by_level = defaultdict(dict)

for record in report["analyses"].values():
    for state in record["states"]:
        by_level[len(state["route"])][
            uid(state)
        ] = state

for level, records in sorted(by_level.items()):
    path = RESULTS / (
        f"beam_v13_early_level_{level}.json"
    )

    states = sorted(
        records.values(),
        key=lambda s: (
            s["special_used"],
            s["current_time"]
        )
    )

    atomic(path, {
        "level": level,
        "verified": True,
        "frontier": states
    })

print("\nPERMBLEDHJA E HAPIT 130")
print("=" * 65, flush=True)

print(
    "Pika fillestare te analizuara:",
    len(report["analyses"]),
    "/",
    len(seed_records),
    flush=True
)

for level, records in sorted(by_level.items()):
    states = list(records.values())

    print(
        f"\nNIVELI {level}/49 | "
        f"Rruge te verifikuara {len(states)}",
        flush=True
    )

    for used in range(6):
        group = [
            s for s in states
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

print(
    "\nRekordi ekzistues i projektit: 31/49",
    flush=True
)

if len(report["analyses"]) < len(seed_records):
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
    print(
        "Kandidatet e hershem jane ruajtur "
        "per kerkimin e ardhshem.",
        flush=True
    )

print("\nHAPI 130 PERFUNDOI!", flush=True)
