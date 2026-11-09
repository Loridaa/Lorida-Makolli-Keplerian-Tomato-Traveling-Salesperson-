
import json
import math
import time
import hashlib
from pathlib import Path

from scipy.optimize import differential_evolution
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

SOURCE = RESULTS / "beam_v16_diverse_level_23.json"

CHECKPOINT = (
    RESULTS / "beam_v17_unlock_special_24_checkpoint.json"
)

OUTPUT = (
    RESULTS / "beam_v17_unlock_special_level_24.json"
)

P = TomatoProblem(str(ROOT / "data/easy.kttsp"))

START = time.monotonic()
TIME_LIMIT = 110.0

# Synojme kohe te mjaftueshme per vazhdim.
ARRIVAL_LIMIT = 145.0

WAITS = [
    0, .25, .5, 1, 2, 3,
    5, 8, 12, 20, 30
]

TOFS = [
    .25, .5, .75, 1, 1.5,
    2, 3, 4, 5, 6,
    8, 10, 12, 16, 20,
    25, 30, 40, 50
]

MAX_REFINEMENTS = 35


def atomic(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(
            obj,
            separators=(",", ":")
        ),
        encoding="utf-8"
    )
    temp.replace(path)


def elapsed():
    return time.monotonic() - START


def transfer_dv(origin, target, depart, tof):
    try:
        dv = float(P.compute_transfer(
            origin,
            target,
            float(depart),
            float(tof)
        ))
        if math.isfinite(dv) and dv >= 0:
            return dv
    except Exception:
        pass
    return float("inf")


def verify(state):
    try:
        route = state["route"]
        transfers = state["transfers"]

        if (
            len(route) != len(set(route))
            or route[0] != 0
            or len(transfers) != len(route) - 1
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

            dv = transfer_dv(
                tr["from"],
                tr["to"],
                depart,
                tof
            )

            if (
                not math.isfinite(dv)
                or abs(dv - float(tr["dv"])) >= .1
            ):
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

            day = depart + tof

        return (
            specials == state["special_used"]
            and specials <= P.n_exceptions
            and abs(
                day - state["current_time"]
            ) < 1e-5
        )

    except Exception:
        return False


def make_child(parent, target, move):
    dv = float(move["dv"])

    if dv < P.dv_threshold:
        status = "NORMAL"

    elif (
        dv < P.exception_threshold
        and parent["special_used"] < P.n_exceptions
    ):
        status = "SPECIAL"

    else:
        return None

    transfer = {
        "from": parent["route"][-1],
        "to": target,
        "start_time": float(move["depart"]),
        "tof": float(move["tof"]),
        "dv": dv,
        "status": status
    }

    child = {
        "route": parent["route"] + [target],
        "transfers": parent["transfers"] + [transfer],
        "current_time": float(move["arrival"]),
        "special_used": (
            parent["special_used"]
            + int(status == "SPECIAL")
        )
    }

    if verify(child):
        return child

    return None


# 1. Lexojme burimin.
raw = SOURCE.read_bytes()
source_hash = hashlib.sha256(raw).hexdigest()

all_states = json.loads(raw)["frontier"]

parents = [
    (i, s)
    for i, s in enumerate(all_states)
    if s["special_used"] <= 4
]

assert parents, "Nuk ka rruge me SPECIAL te lira."

for _, state in parents:
    assert len(state["route"]) == 23
    assert verify(state), (
        "Nje itinerar fillestar nuk eshte i vlefshem."
    )

tasks = []

for index, state in parents:
    visited = set(state["route"])

    for target in range(P.n_tomatoes):
        if target not in visited:
            tasks.append((index, target))

tasks.sort(key=lambda item: (
    all_states[item[0]]["special_used"],
    all_states[item[0]]["current_time"],
    item[1]
))

config = {
    "source_hash": source_hash,
    "arrival_limit": ARRIVAL_LIMIT,
    "waits": WAITS,
    "tofs": TOFS,
    "max_refinements": MAX_REFINEMENTS,
    "version": 1
}

if CHECKPOINT.exists():
    checkpoint = json.loads(
        CHECKPOINT.read_text(encoding="utf-8")
    )

    assert checkpoint["config"] == config, (
        "Checkpoint nuk perputhet."
    )

else:
    checkpoint = {
        "config": config,
        "grid": {},
        "refined": {}
    }
    atomic(CHECKPOINT, checkpoint)


def key_for(index, target):
    return f"{index}:{target}"


def grid_search(state, target):
    origin = state["route"][-1]
    day = float(state["current_time"])

    best = None
    feasible = None
    tested = 0

    for wait in WAITS:
        depart = day + wait

        for tof in TOFS:
            arrival = depart + tof

            if arrival > ARRIVAL_LIMIT:
                continue

            dv = transfer_dv(
                origin,
                target,
                depart,
                tof
            )

            tested += 1

            if not math.isfinite(dv):
                continue

            move = {
                "wait": wait,
                "depart": depart,
                "tof": tof,
                "dv": dv,
                "arrival": arrival
            }

            if (
                best is None
                or dv < best["dv"]
            ):
                best = move

            if dv < P.exception_threshold:
                if (
                    feasible is None
                    or (
                        arrival,
                        dv
                    ) < (
                        feasible["arrival"],
                        feasible["dv"]
                    )
                ):
                    feasible = move

    return {
        "best": best,
        "feasible": feasible,
        "tested": tested
    }


print("FAZA 1 - BURIMI")
print("=" * 65, flush=True)
print(
    "Rruge te nivelit 23 me SPECIAL te lira:",
    len(parents),
    flush=True
)
print(
    "Kombinime prind-target:",
    len(tasks),
    flush=True
)

# 2. Kerkimi me rrjete kohe.
print("\nFAZA 2 - KERKIMI I TRANSFERIMEVE")
print("=" * 65, flush=True)

for index, target in tasks:
    if elapsed() >= TIME_LIMIT:
        break

    key = key_for(index, target)

    if key in checkpoint["grid"]:
        continue

    parent = all_states[index]
    result = grid_search(parent, target)

    checkpoint["grid"][key] = result
    atomic(CHECKPOINT, checkpoint)

    completed = len(checkpoint["grid"])

    if (
        completed % 15 == 0
        or result["feasible"] is not None
    ):
        best = result["best"]

        print(
            f"Kontrolluar {completed}/{len(tasks)} | "
            f"Fund {parent['route'][-1]} "
            f"-> {target} | "
            f"SPECIAL {parent['special_used']}/5 | "
            f"Min DV "
            f"{best['dv']:.2f}"
            if best else
            f"Kontrolluar {completed}/{len(tasks)} | "
            "Pa DV te vlefshme",
            flush=True
        )

        if result["feasible"] is not None:
            move = result["feasible"]

            print(
                "TRANSFERIM I MUNDSHEM! "
                f"DV={move['dv']:.2f}, "
                f"Arritja={move['arrival']:.2f}",
                flush=True
            )


# 3. Optimizimi i vazhdueshem, vetem pasi
# eshte perfunduar kerkimi ne rrjete.
grid_done = (
    len(checkpoint["grid"]) == len(tasks)
)

if grid_done:
    print("\nFAZA 3 - OPTIMIZIMI I VAZHDUESHEM")
    print("=" * 65, flush=True)

    candidates = []

    for index, target in tasks:
        key = key_for(index, target)
        record = checkpoint["grid"][key]
        best = record["best"]

        if best is None:
            continue

        # Renditje sipas afersise me DV 600.
        score = (
            best["dv"]
            + max(
                0.0,
                best["arrival"] - 115.0
            ) * 1.5
        )

        candidates.append((
            score,
            index,
            target
        ))

    candidates.sort()

    finalists = candidates[:MAX_REFINEMENTS]

    print(
        "Kandidate per optimizim:",
        len(finalists),
        flush=True
    )

    for rank, (_, index, target) in enumerate(
        finalists, 1
    ):
        if elapsed() >= TIME_LIMIT:
            break

        key = key_for(index, target)

        if key in checkpoint["refined"]:
            continue

        parent = all_states[index]
        origin = parent["route"][-1]
        day = float(parent["current_time"])

        grid_best = checkpoint["grid"][key]["best"]

        max_wait = min(
            35.0,
            ARRIVAL_LIMIT - day - .25
        )

        max_tof = min(
            75.0,
            ARRIVAL_LIMIT - day
        )

        if max_wait <= 0 or max_tof <= .25:
            checkpoint["refined"][key] = None
            atomic(CHECKPOINT, checkpoint)
            continue

        def objective(x):
            wait = float(x[0])
            tof = float(x[1])

            depart = day + wait
            arrival = depart + tof

            if arrival > ARRIVAL_LIMIT:
                return (
                    100000.0
                    + 1000.0 * (
                        arrival - ARRIVAL_LIMIT
                    )
                )

            dv = transfer_dv(
                origin,
                target,
                depart,
                tof
            )

            if not math.isfinite(dv):
                return 1000000.0

            return (
                dv
                + max(
                    0.0,
                    arrival - 115.0
                ) * 1.5
            )

        seed = int(
            hashlib.sha256(
                key.encode()
            ).hexdigest()[:8],
            16
        )

        result = differential_evolution(
            objective,
            bounds=[
                (0, max_wait),
                (.25, max_tof)
            ],
            strategy="best1bin",
            maxiter=20,
            popsize=7,
            tol=.02,
            polish=False,
            seed=seed,
            x0=[
                grid_best["wait"],
                grid_best["tof"]
            ],
            workers=1
        )

        wait = float(result.x[0])
        tof = float(result.x[1])
        depart = day + wait
        arrival = depart + tof

        dv = transfer_dv(
            origin,
            target,
            depart,
            tof
        )

        refined = None

        if (
            math.isfinite(dv)
            and arrival <= ARRIVAL_LIMIT
        ):
            refined = {
                "wait": wait,
                "depart": depart,
                "tof": tof,
                "dv": dv,
                "arrival": arrival
            }

        checkpoint["refined"][key] = refined
        atomic(CHECKPOINT, checkpoint)

        if refined is not None:
            print(
                f"[{rank}/{len(finalists)}] "
                f"{origin}->{target} | "
                f"DV={dv:.2f} | "
                f"Dita={arrival:.2f} | "
                f"Gap={dv - P.exception_threshold:+.2f}",
                flush=True
            )


# 4. Mbledhim rezultatet e vlefshme.
print("\nFAZA 4 - VERIFIKIMI I REZULTATEVE")
print("=" * 65, flush=True)

solutions = {}

for index, target in tasks:
    key = key_for(index, target)
    parent = all_states[index]

    moves = []

    if key in checkpoint["grid"]:
        grid_result = checkpoint["grid"][key]

        if grid_result["feasible"]:
            moves.append(
                grid_result["feasible"]
            )

    if key in checkpoint["refined"]:
        refined = checkpoint["refined"][key]

        if (
            refined is not None
            and refined["dv"] < P.exception_threshold
        ):
            moves.append(refined)

    for move in moves:
        child = make_child(
            parent,
            target,
            move
        )

        if child is None:
            continue

        identity = (
            tuple(sorted(child["route"])),
            child["route"][-1],
            child["special_used"]
        )

        if (
            identity not in solutions
            or child["current_time"]
            < solutions[identity]["current_time"]
        ):
            solutions[identity] = child

verified = sorted(
    solutions.values(),
    key=lambda s: (
        s["special_used"],
        s["current_time"]
    )
)

if verified:
    atomic(OUTPUT, {
        "level": 24,
        "verified": True,
        "frontier": verified
    })


# 5. Permbledhja.
all_moves = []

for record in checkpoint["grid"].values():
    if record["best"] is not None:
        all_moves.append(record["best"])

for record in checkpoint["refined"].values():
    if record is not None:
        all_moves.append(record)

print("\nPERMBLEDHJA E HAPIT 141")
print("=" * 65, flush=True)

print(
    "Kombinime te kontrolluara:",
    len(checkpoint["grid"]),
    "/",
    len(tasks),
    flush=True
)

print(
    "Optimizime te kryera:",
    len(checkpoint["refined"]),
    "/",
    min(MAX_REFINEMENTS, len(tasks)),
    flush=True
)

print(
    "Rruge te vlefshme ne nivelin 24:",
    len(verified),
    flush=True
)

if all_moves:
    minimum = min(
        all_moves,
        key=lambda m: m["dv"]
    )

    print(
        "Delta-V minimale:",
        round(minimum["dv"], 4),
        flush=True
    )

    print(
        "Diferenca nga kufiri SPECIAL 600:",
        round(
            minimum["dv"] - P.exception_threshold,
            4
        ),
        flush=True
    )

for used in range(5):
    group = [
        state for state in verified
        if state["special_used"] == used
    ]

    if group:
        print(
            f"SPECIAL {used}/5 | "
            f"Rruge {len(group)} | "
            f"Dita minimale "
            f"{min(s['current_time'] for s in group):.2f}",
            flush=True
        )

finished = (
    grid_done
    and len(checkpoint["refined"])
    >= min(
        MAX_REFINEMENTS,
        len([
            record
            for record in checkpoint["grid"].values()
            if record["best"] is not None
        ])
    )
)

print(
    "Kerkimi i pjesshem:",
    not finished,
    flush=True
)

print(
    "Rekordi ekzistues: 31/49",
    flush=True
)

print("HAPI 141 PERFUNDOI!", flush=True)
