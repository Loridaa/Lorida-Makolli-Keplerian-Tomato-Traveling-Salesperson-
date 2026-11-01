
import hashlib
import json
import math
import time
from pathlib import Path
from scipy.optimize import differential_evolution
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_v13_continue_level_27.json"
LOOKAHEAD = RESULTS / "beam_v14_lookahead_27_to29_checkpoint.json"
OUT = RESULTS / "beam_v14_refine_second_leg_checkpoint.json"
SNAP = RESULTS / "beam_v14_refined_level_29.json"

P = TomatoProblem(str(ROOT / "data/easy.kttsp"))

START = time.monotonic()
LIMIT = 125.0
MAX_TASKS = 48

W = [0, .25, .5, 1, 2, 3, 5, 8, 12, 18, 25, 32]
T = [.25, .5, .75, 1, 1.5, 2, 3, 5, 8, 12, 18, 25, 32]


class Pause(Exception):
    pass


def budget():
    if time.monotonic() - START >= LIMIT:
        raise Pause()


def digest(obj):
    raw = json.dumps(
        obj, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(raw).hexdigest()[:24]


def save(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(data, separators=(",", ":")),
        encoding="utf-8"
    )
    temp.replace(path)


def calc(i, j, depart, tof):
    try:
        value = float(P.compute_transfer(i, j, depart, tof))
        return value if math.isfinite(value) else float("inf")
    except Exception:
        return float("inf")


def valid(s):
    try:
        route = s["route"]
        transfers = s["transfers"]

        if (
            not route
            or route[0] != 0
            or len(set(route)) != len(route)
            or len(transfers) != len(route) - 1
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

            dv = calc(route[k], route[k + 1], depart, tof)

            if abs(dv - float(tr["dv"])) > .1:
                return False

            status = (
                "NORMAL" if dv < P.dv_threshold
                else "SPECIAL" if dv < P.exception_threshold
                else "INVALID"
            )

            if status != tr["status"] or status == "INVALID":
                return False

            used += status == "SPECIAL"
            day = depart + tof

        return (
            used == s["special_used"]
            and used <= P.n_exceptions
            and abs(day - s["current_time"]) < 1e-5
        )
    except Exception:
        return False


def extend(parent, target, wait, tof, dv):
    status = (
        "NORMAL" if dv < P.dv_threshold
        else "SPECIAL"
        if (
            dv < P.exception_threshold
            and parent["special_used"] < P.n_exceptions
        )
        else None
    )

    if status is None:
        return None

    depart = parent["current_time"] + wait

    candidate = {
        "route": parent["route"] + [target],
        "transfers": parent["transfers"] + [{
            "from": parent["route"][-1],
            "to": target,
            "start_time": depart,
            "tof": tof,
            "dv": dv,
            "status": status
        }],
        "current_time": depart + tof,
        "special_used": (
            parent["special_used"] + int(status == "SPECIAL")
        )
    }

    return candidate if valid(candidate) else None


# Lexojme rruget e nivelit 27 dhe Hapin 133.
raw_source = SOURCE.read_bytes()
raw_old = LOOKAHEAD.read_bytes()

parents = {
    digest(s): s
    for s in json.loads(raw_source)["frontier"]
    if len(s["route"]) == 27 and s["special_used"] == 3
}

old = json.loads(raw_old)

assert all(
    pid in parents for pid in old["parents"]
), "Prinderit nuk perputhen me Hapin 133."

tasks = []

for pid, row in old["parents"].items():
    assert row["complete"], "Hapi 133 nuk eshte perfunduar."

    for index, move in enumerate(row["first_moves"]):
        if move["status"] != "NORMAL":
            continue

        details = row["checked"].get(str(index))

        if details is None:
            continue

        arrival = float(move["start"]) + float(move["tof"])

        if arrival >= P.max_time - .2:
            continue

        if float(move["dv"]) >= P.dv_threshold:
            continue

        tasks.append({
            "id": digest({"pid": pid, "move": move}),
            "pid": pid,
            "move": move,
            "previous_min": details.get("minimum_dv"),
            "arrival": arrival
        })

tasks.sort(
    key=lambda x: (
        float(x["previous_min"])
        if x["previous_min"] is not None
        else 1e12,
        x["arrival"]
    )
)

chosen = []
usage = {}
seen_ids = set()

for task in tasks:
    pid = task["pid"]

    if (
        usage.get(pid, 0) >= 3
        or task["id"] in seen_ids
    ):
        continue

    chosen.append(task)
    seen_ids.add(task["id"])
    usage[pid] = usage.get(pid, 0) + 1

    if len(chosen) >= MAX_TASKS:
        break

assert chosen, "Nuk ka transferime te para NORMAL."

config = {
    "source_hash": hashlib.sha256(raw_source).hexdigest(),
    "lookahead_hash": hashlib.sha256(raw_old).hexdigest(),
    "ids": [x["id"] for x in chosen],
    "W": W,
    "T": T,
    "version": 1
}

if OUT.exists():
    report = json.loads(OUT.read_text(encoding="utf-8"))
    assert report["config"] == config, (
        "Checkpoint ka konfigurim tjeter."
    )
else:
    report = {
        "config": config,
        "done": {},
        "found": {}
    }
    save(OUT, report)


print("HAPI 135 - OPTIMIZIMI I TRANSFERIMIT TE DYTE")
print("=" * 65, flush=True)
print(
    "Prinder me 3 SPECIAL:", len(parents),
    "| Alternativa NORMAL:", len(tasks),
    "| Te zgjedhura:", len(chosen),
    flush=True
)


def scan(state):
    day = state["current_time"]
    remaining = P.max_time - day
    origin = state["route"][-1]

    missing = [
        j for j in range(P.n_tomatoes)
        if j not in state["route"]
    ]

    best = []

    for target in missing:
        closest = None

        for wait in W:
            for tof in T:
                budget()

                if wait + tof > remaining:
                    continue

                value = calc(
                    origin, target, day + wait, tof
                )

                if not math.isfinite(value):
                    continue

                if closest is None or value < closest["dv"]:
                    closest = {
                        "dv": value,
                        "wait": wait,
                        "tof": tof
                    }

        if closest is not None:
            best.append((target, closest))

    best.sort(key=lambda row: row[1]["dv"])
    return best


def improve(state, target, guess, seed):
    remaining = P.max_time - state["current_time"]

    if remaining <= .21:
        return guess

    origin = state["route"][-1]
    day = state["current_time"]

    def objective(x):
        budget()

        wait = (remaining - .2) * float(x[0])
        tof = (
            .2 + (remaining - .2 - wait) * float(x[1])
        )

        return calc(origin, target, day + wait, tof)

    try:
        result = differential_evolution(
            objective,
            [(0, 1), (0, 1)],
            popsize=7,
            maxiter=32,
            seed=seed,
            polish=False,
            tol=.02
        )

    except Pause:
        raise

    except Exception:
        return guess

    wait = (remaining - .2) * float(result.x[0])
    tof = (
        .2 + (remaining - .2 - wait) * float(result.x[1])
    )

    value = calc(origin, target, day + wait, tof)

    if (
        math.isfinite(value)
        and value < guess["dv"]
    ):
        return {
            "dv": value,
            "wait": wait,
            "tof": tof
        }

    return guess


paused = False

try:
    for i, task in enumerate(chosen, 1):
        budget()

        if task["id"] in report["done"]:
            continue

        parent = parents[task["pid"]]
        move = task["move"]

        first = extend(
            parent,
            int(move["to"]),
            float(move["start"]) - parent["current_time"],
            float(move["tof"]),
            float(move["dv"])
        )

        assert (
            first is not None
            and len(first["route"]) == 28
            and first["special_used"] == 3
        ), "Transferimi i pare nuk u verifikua."

        ranked = scan(first)

        # Tre destinacionet me te mira.
        requested = ranked[:3]

        # Provojme edhe domaten 40, nese mundet.
        target40 = next(
            (r for r in ranked if r[0] == 40),
            None
        )

        if (
            target40
            and all(r[0] != 40 for r in requested)
        ):
            requested.append(target40)

        outcomes = []
        found = []

        for target, guess in requested:
            budget()

            seed = int(
                hashlib.sha256(
                    f"{task['id']}:{target}".encode()
                ).hexdigest()[:8],
                16
            )

            fitted = improve(
                first, target, guess, seed
            )

            candidate = extend(
                first,
                target,
                fitted["wait"],
                fitted["tof"],
                fitted["dv"]
            )

            if (
                candidate is not None
                and candidate["special_used"] <= 4
            ):
                found.append(candidate)

            outcomes.append({
                "target": target,
                "grid_dv": round(guess["dv"], 3),
                "best_dv": round(fitted["dv"], 3),
                "arrival": round(
                    first["current_time"]
                    + fitted["wait"]
                    + fitted["tof"],
                    3
                )
            })

        for state in found:
            report["found"][digest(state)] = state

        report["done"][task["id"]] = {
            "parent": task["pid"],
            "first_end": first["route"][-1],
            "arrival28": first["current_time"],
            "tests": outcomes,
            "found": len(found)
        }

        save(OUT, report)

        best_dv = min(
            (r["best_dv"] for r in outcomes),
            default=float("inf")
        )

        print(
            f"[{i}/{len(chosen)}] "
            f"27->{first['route'][-1]}->29 | "
            f"Dita28 {first['current_time']:.2f} | "
            f"DV min {best_dv:.2f} | "
            f"Rruge29 {len(found)}",
            flush=True
        )

except Pause:
    paused = True


if report["found"]:
    save(SNAP, {
        "level": 29,
        "verified": True,
        "frontier": list(report["found"].values())
    })


print("\nPERMBLEDHJA E HAPIT 135")
print("=" * 65, flush=True)

print(
    "Alternativa te perfunduara:",
    len(report["done"]), "/", len(chosen),
    flush=True
)

print(
    "Rruge te verifikuara nivel29 "
    "me maksimum 4 SPECIAL:",
    len(report["found"]),
    flush=True
)

near = []

for row in report["done"].values():
    for info in row["tests"]:
        near.append((
            info["best_dv"],
            row["first_end"],
            info["target"],
            row["arrival28"],
            info["arrival"]
        ))

near.sort()

print("\nTRANSFERIMET ME PREMTUESE")
print("=" * 65, flush=True)

for dv, origin, target, day28, day29 in near[:6]:
    print(
        f"{origin}->{target} | "
        f"Dita28 {day28:.2f} | "
        f"DV {dv:.2f} | "
        f"Diferenca nga 600: {dv-600:+.2f} | "
        f"Dita29 {day29:.2f}",
        flush=True
    )

print(
    "Kerkimi i pjesshem:",
    paused or len(report["done"]) < len(chosen),
    flush=True
)

print(
    "Rekordi i konfirmuar mbetet 31/49, "
    "derisa te verifikohet niveli 32.",
    flush=True
)

print("\nHAPI 135 PERFUNDOI!", flush=True)
