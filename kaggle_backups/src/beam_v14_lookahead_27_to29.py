
import json
import math
import time
import hashlib
from pathlib import Path
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_v13_continue_level_27.json"
OUTPUT = RESULTS / "beam_v14_lookahead_27_to29_checkpoint.json"
SNAPSHOT = RESULTS / "beam_v14_lookahead_level_29.json"

P = TomatoProblem(str(ROOT / "data/easy.kttsp"))

T0 = time.monotonic()
LIMIT = 125.0
MAX_FIRST = 20

W1 = [0, .25, .5, 1, 2, 4, 7, 12, 18, 26, 36]
T1 = [.25, .5, .75, 1, 1.5, 2, 3, 5, 8, 13, 20, 30, 36]

W2 = [0, .25, .5, 1, 2, 3, 5, 8, 12, 18, 26, 36]
T2 = [.25, .5, .75, 1, 1.5, 2, 3, 5, 8, 13, 20, 30, 36]


class Pause(Exception):
    pass


def time_check():
    if time.monotonic() - T0 >= LIMIT:
        raise Pause()


def hash_obj(obj):
    content = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":")
    )

    return hashlib.sha256(
        content.encode()
    ).hexdigest()[:24]


def atomic(path, data):
    temp = path.with_suffix(".tmp")

    temp.write_text(
        json.dumps(data, separators=(",", ":")),
        encoding="utf-8"
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
            or len(route) > P.n_tomatoes
            or len(transfers) != len(route) - 1
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
                or abs(dv - float(tr["dv"])) > .1
            ):
                return False

            if dv < P.dv_threshold:
                status = "NORMAL"

            elif dv < P.exception_threshold:
                status = "SPECIAL"

            else:
                return False

            if status != tr["status"]:
                return False

            used += status == "SPECIAL"
            day = depart + tof

        return (
            abs(day - float(state["current_time"])) < 1e-5
            and used == state["special_used"]
            and used <= 4
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


def find_moves(state, waits, tofs):
    current = state["route"][-1]
    day = float(state["current_time"])
    visited = set(state["route"])
    used = int(state["special_used"])

    options = []
    least_dv = float("inf")
    evaluated = 0

    for target in range(P.n_tomatoes):
        if target in visited:
            continue

        hits = {
            "NORMAL": [],
            "SPECIAL": []
        }

        for wait in waits:
            time_check()

            for tof in tofs:
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

                    evaluated += 1

                except Exception:
                    continue

                if not math.isfinite(dv):
                    continue

                least_dv = min(least_dv, dv)

                if dv < P.dv_threshold:
                    status = "NORMAL"

                elif (
                    dv < P.exception_threshold
                    and used < 4
                ):
                    status = "SPECIAL"

                else:
                    continue

                hits[status].append({
                    "to": target,
                    "start": depart,
                    "tof": tof,
                    "dv": dv,
                    "status": status
                })

        for status, group in hits.items():
            if not group:
                continue

            group.sort(
                key=lambda m: (
                    m["start"] + m["tof"],
                    m["dv"]
                )
            )

            earliest = group[0]
            options.append(earliest)

            # Ruajme edhe nje alternative
            # me Delta-V me te ulet.
            close = [
                m for m in group
                if m["start"] + m["tof"]
                <= earliest["start"] + earliest["tof"] + 8
            ]

            lowest = min(
                close,
                key=lambda m: (
                    m["dv"],
                    m["start"] + m["tof"]
                )
            )

            difference = abs(
                lowest["start"] + lowest["tof"]
                - earliest["start"] - earliest["tof"]
            )

            if difference >= .25:
                options.append(lowest)

    minimum = (
        least_dv
        if math.isfinite(least_dv)
        else None
    )

    return options, evaluated, minimum


# FAZA 1 - Marrim rruget me 3 SPECIAL
# nga niveli 27.
raw = SOURCE.read_bytes()

all_states = json.loads(raw)["frontier"]

parents = sorted(
    [
        s for s in all_states
        if len(s["route"]) == 27
        and s["special_used"] == 3
    ],
    key=lambda s: s["current_time"]
)

assert parents, (
    "Nuk u gjeten rruge me 3 SPECIAL."
)

assert all(verify(s) for s in parents), (
    "Ndonje rruge fillestare nuk "
    "kaloi verifikimin PyKEP."
)

CONFIG = {
    "source_sha256": hashlib.sha256(
        raw
    ).hexdigest(),
    "parent_ids": [
        hash_obj(s) for s in parents
    ],
    "w1": W1,
    "t1": T1,
    "w2": W2,
    "t2": T2,
    "max_first": MAX_FIRST,
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
        "parents": {}
    }

    atomic(OUTPUT, report)


def save():
    atomic(OUTPUT, report)


print("FAZA 1 - KONTROLLI I RRUGEVE")
print("=" * 65, flush=True)

print(
    "Rruge gjithsej ne nivelin 27:",
    len(all_states),
    flush=True
)

print(
    "Prinder me 3/5 SPECIAL:",
    len(parents),
    flush=True
)

print("\nFAZA 2 - TWO-STEP LOOKAHEAD 27 -> 29")
print("=" * 65, flush=True)

try:
    for index, parent in enumerate(parents, 1):
        time_check()

        sid = hash_obj(parent)

        if sid not in report["parents"]:
            moves, evaluations, _ = find_moves(
                parent, W1, T1
            )

            normals = sorted(
                [
                    m for m in moves
                    if m["status"] == "NORMAL"
                ],
                key=lambda m: (
                    m["start"] + m["tof"],
                    m["dv"]
                )
            )[:12]

            specials = sorted(
                [
                    m for m in moves
                    if m["status"] == "SPECIAL"
                ],
                key=lambda m: (
                    m["start"] + m["tof"],
                    m["dv"]
                )
            )[:8]

            selected = (
                normals + specials
            )[:MAX_FIRST]

            report["parents"][sid] = {
                "first_moves": selected,
                "first_evaluations": evaluations,
                "checked": {},
                "complete": False
            }

            save()

        entry = report["parents"][sid]

        for i, move in enumerate(entry["first_moves"]):
            time_check()

            key = str(i)

            if key in entry["checked"]:
                continue

            child28 = make_child(parent, move)

            assert (
                len(child28["route"]) == 28
                and verify(child28)
            )

            possible, tested, minimum_dv = find_moves(
                child28, W2, T2
            )

            children29 = []

            for next_move in possible:
                child29 = make_child(
                    child28,
                    next_move
                )

                if (
                    len(child29["route"]) == 29
                    and verify(child29)
                ):
                    children29.append(child29)

            children29.sort(
                key=lambda s: (
                    s["special_used"],
                    s["current_time"]
                )
            )

            entry["checked"][key] = {
                "children29": children29[:20],
                "evaluations": tested,
                "minimum_dv": minimum_dv,
                "via": move["to"],
                "arrived28": child28["current_time"],
                "special28": child28["special_used"]
            }

            save()

            print(
                f"[{index}/{len(parents)}] "
                f"27 -> {move['to']} -> 29 | "
                f"SPECIAL28 "
                f"{child28['special_used']}/5 | "
                f"Rruge29 {len(children29)}",
                flush=True
            )

        entry["complete"] = (
            len(entry["checked"])
            == len(entry["first_moves"])
        )

        save()

except Pause:
    print(
        "\nKufiri kohor u arrit. "
        "Checkpoint u ruajt.",
        flush=True
    )


# FAZA 3 - Bashkojme rruget e verifikuara.
found = {}

for entry in report["parents"].values():
    for item in entry["checked"].values():
        for state in item["children29"]:
            found[hash_obj(state)] = state

if found:
    atomic(SNAPSHOT, {
        "level": 29,
        "verified": True,
        "frontier": list(found.values())
    })


print("\nPERMBLEDHJA E HAPIT 133")
print("=" * 65, flush=True)

completed = sum(
    item["complete"]
    for item in report["parents"].values()
)

checked = sum(
    len(item["checked"])
    for item in report["parents"].values()
)

print(
    f"Prinder te perfunduar: "
    f"{completed}/{len(parents)}",
    flush=True
)

print(
    "Alternativa te nivelit 28 te kontrolluara:",
    checked,
    flush=True
)

print(
    "Rruge te verifikuara me 29 domate "
    "dhe maksimum 4 SPECIAL:",
    len(found),
    flush=True
)

for used in [3, 4]:
    group = [
        s for s in found.values()
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

if found:
    print(
        "\nU GJETEN DEGE TE REJA "
        "ME SPECIAL TE LIRA!",
        flush=True
    )
else:
    print(
        "\nEnde nuk u gjet dege me "
        "SPECIAL te lira ne nivelin 29.",
        flush=True
    )

print(
    "Rekordi i projektit mbetet 31/49.",
    flush=True
)

if completed < len(parents):
    print(
        "Kerkimi eshte i pjesshem. "
        "Ekzekuto perseri te njejten qelize.",
        flush=True
    )

print("\nHAPI 133 PERFUNDOI!", flush=True)
