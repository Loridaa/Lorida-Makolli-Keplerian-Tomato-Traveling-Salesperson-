
import json
import math
import time
import hashlib
from pathlib import Path
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

CHECKPOINT = (
    RESULTS / "beam_v21_insert_into31_checkpoint.json"
)
OUTPUT = (
    RESULTS / "beam_v21_insert_into31_level_32.json"
)

P = TomatoProblem(str(ROOT / "data/easy.kttsp"))

START = time.monotonic()
LIMIT = 112.0
SEED_LIMIT = 24
GAPS_PER_SEED = 10

FIRST_TOFS = [
    .25, .5, .75, 1, 1.5, 2, 3,
    4, 5, 8, 12, 16, 22
]
SECOND_TOFS = FIRST_TOFS
SECOND_WAITS = [0, .25, .5, 1, 2, 4, 8]


def atomic(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(value, separators=(",", ":")),
        encoding="utf-8"
    )
    temp.replace(path)


def identity(state):
    value = [
        state["route"],
        state["special_used"],
        [
            [
                round(float(t["start_time"]), 5),
                round(float(t["tof"]), 5)
            ]
            for t in state["transfers"]
        ]
    ]
    raw = json.dumps(value, sort_keys=True)
    return hashlib.sha256(
        raw.encode()
    ).hexdigest()[:22]


def dv(a, b, depart, tof):
    if (
        tof <= 0
        or depart < 0
        or depart + tof > P.max_time + 1e-8
    ):
        return float("inf")

    try:
        value = float(P.compute_transfer(
            int(a), int(b),
            float(depart), float(tof)
        ))
        if math.isfinite(value) and value >= 0:
            return value
    except Exception:
        pass

    return float("inf")


def verify(state):
    try:
        route = state["route"]
        transfers = state["transfers"]

        if (
            not route
            or route[0] != 0
            or len(route) != len(set(route))
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

            measured = dv(
                route[i], route[i + 1],
                depart, tof
            )

            if (
                not math.isfinite(measured)
                or abs(measured - float(tr["dv"])) > .1
            ):
                return False

            if measured < P.dv_threshold:
                status = "NORMAL"
            elif measured < P.exception_threshold:
                status = "SPECIAL"
            else:
                return False

            if status != tr["status"]:
                return False

            specials += int(status == "SPECIAL")
            day = depart + tof

        return (
            specials == state["special_used"]
            and specials <= P.n_exceptions
            and abs(day - state["current_time"]) < 1e-5
        )

    except Exception:
        return False


def extract(obj):
    found = []
    stack = [obj]

    while stack and len(found) < 600:
        item = stack.pop()

        if isinstance(item, dict):
            keys = (
                "route", "transfers",
                "special_used", "current_time"
            )
            if all(k in item for k in keys):
                if (
                    isinstance(item["route"], list)
                    and len(item["route"]) == 31
                ):
                    found.append(item)
            else:
                stack.extend(item.values())

        elif isinstance(item, list):
            stack.extend(item)

    return found


print("FAZA 1 - NGARKIMI I RRUGEVE ME 31 DOMATE")
print("=" * 65, flush=True)

paths = set(RESULTS.glob("*level_31.json"))

for pattern in [
    "*retime*checkpoint.json",
    "*reorder*checkpoint.json",
    "*mix*checkpoint.json",
    "*continue_checkpoint.json"
]:
    paths.update(RESULTS.glob(pattern))

seeds = {}
sources = []

for path in sorted(paths):
    if path.stat().st_size > 35_000_000:
        continue

    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )
        states = extract(data)
        accepted = 0

        for state in states:
            if state["special_used"] not in (4, 5):
                continue

            if not verify(state):
                continue

            sid = identity(state)

            if sid not in seeds:
                seeds[sid] = {
                    "id": sid,
                    "state": state,
                    "source": path.name
                }
                accepted += 1

        if accepted:
            sources.append((path.name, accepted))

    except Exception as error:
        print(
            "Skedar i anashkaluar:",
            path.name,
            str(error)[:100],
            flush=True
        )

assert seeds, (
    "Nuk u gjet asnje rruge e verifikuar "
    "me 31 domate."
)


def windows(state):
    transfers = state["transfers"]
    available = []

    for k in range(len(transfers)):
        earliest = (
            0.0 if k == 0
            else (
                float(transfers[k - 1]["start_time"])
                + float(transfers[k - 1]["tof"])
            )
        )

        deadline = (
            float(transfers[k + 1]["start_time"])
            if k + 1 < len(transfers)
            else float(P.max_time)
        )

        span = deadline - earliest

        if span < .5:
            continue

        was_special = (
            transfers[k]["status"] == "SPECIAL"
        )

        priority = (
            int(was_special) * 40
            + min(span, 60) * 1.5
            + int(state["special_used"] == 4) * 5
        )

        available.append({
            "gap": k,
            "earliest": earliest,
            "deadline": deadline,
            "span": span,
            "priority": priority
        })

    return sorted(
        available,
        key=lambda x: (-x["priority"], x["gap"])
    )[:GAPS_PER_SEED]


all_seeds = list(seeds.values())

for record in all_seeds:
    state = record["state"]
    record["windows"] = windows(state)
    record["score"] = (
        max(
            (
                x["priority"]
                for x in record["windows"]
            ),
            default=0
        ),
        -state["current_time"]
    )

groups = {4: [], 5: []}

for record in all_seeds:
    groups[record["state"]["special_used"]].append(
        record
    )

for group in groups.values():
    group.sort(
        key=lambda x: x["score"],
        reverse=True
    )

selected = []
seen_suffix = set()

# I japim perparesi rrugeve me nje SPECIAL te lire.
for used, quota in [(4, 12), (5, 12)]:
    count = 0

    for record in groups[used]:
        suffix = tuple(
            record["state"]["route"][-5:]
        )

        if suffix in seen_suffix:
            continue

        selected.append(record)
        seen_suffix.add(suffix)
        count += 1

        if count >= quota:
            break

selected_ids = {r["id"] for r in selected}

for record in sorted(
    all_seeds,
    key=lambda x: x["score"],
    reverse=True
):
    if len(selected) >= SEED_LIMIT:
        break

    if record["id"] not in selected_ids:
        selected.append(record)
        selected_ids.add(record["id"])

selected = selected[:SEED_LIMIT]

# Nderthurim provat e rrugeve te ndryshme.
tasks = []

for j in range(GAPS_PER_SEED):
    for i, record in enumerate(selected):
        if j < len(record["windows"]):
            tasks.append((
                i,
                record["windows"][j]
            ))

config = {
    "seed_ids": [r["id"] for r in selected],
    "gaps": [
        [x["gap"] for x in r["windows"]]
        for r in selected
    ],
    "version": 1
}

if CHECKPOINT.exists():
    checkpoint = json.loads(
        CHECKPOINT.read_text(encoding="utf-8")
    )
    assert checkpoint["config"] == config, (
        "Checkpoint nuk perputhet me burimet."
    )
else:
    checkpoint = {
        "config": config,
        "done": {},
        "solutions": {}
    }
    atomic(CHECKPOINT, checkpoint)

print("Burime me rruge 31:", len(sources), flush=True)
print(
    "Rruge unike te verifikuara:",
    len(seeds),
    flush=True
)
print(
    "Rruge te zgjedhura:",
    len(selected),
    flush=True
)
print(
    "Me 4/5 SPECIAL:",
    sum(
        r["state"]["special_used"] == 4
        for r in selected
    ),
    flush=True
)
print(
    "Me 5/5 SPECIAL:",
    sum(
        r["state"]["special_used"] == 5
        for r in selected
    ),
    flush=True
)
print(
    "Dritare per futje:",
    len(tasks),
    flush=True
)


def first_starts(earliest, deadline, original):
    span = deadline - earliest

    values = [
        earliest,
        original,
        original - 1,
        original + .5
    ]

    values += [
        earliest + offset
        for offset in [
            0, .25, .5, 1, 2,
            3, 5, 8, 12, 20, 30
        ]
    ]

    values += [
        earliest + span * fraction
        for fraction in [.2, .45, .7]
    ]

    return sorted({
        round(float(t), 6)
        for t in values
        if earliest - 1e-6 <= t <= deadline - .5
    })


def attempt_insertion(record, window):
    state = record["state"]
    route = state["route"]
    transfers = state["transfers"]

    k = window["gap"]
    origin = route[k]
    destination = route[k + 1]

    old = transfers[k]
    earliest = window["earliest"]
    deadline = window["deadline"]

    # Heqim SPECIAL-in e transferimit qe zevendesojme.
    existing_specials = (
        state["special_used"]
        - int(old["status"] == "SPECIAL")
    )

    missing = [
        tomato
        for tomato in range(P.n_tomatoes)
        if tomato not in route
    ]

    starts = first_starts(
        earliest,
        deadline,
        float(old["start_time"])
    )

    calls = 0
    legal_first = 0
    legal_second = 0
    solutions = []
    best_near = None

    for tomato in missing:
        if time.monotonic() - START >= LIMIT:
            return None

        phases = {}

        # Transferimi A -> domate e re.
        for dep1 in starts:
            for tof1 in FIRST_TOFS:
                arrival1 = dep1 + tof1

                if arrival1 + .25 > deadline:
                    continue

                value1 = dv(
                    origin, tomato, dep1, tof1
                )
                calls += 1

                if value1 >= P.exception_threshold:
                    continue

                kind1 = (
                    "NORMAL"
                    if value1 < P.dv_threshold
                    else "SPECIAL"
                )

                used1 = (
                    existing_specials
                    + int(kind1 == "SPECIAL")
                )

                if used1 > P.n_exceptions:
                    continue

                legal_first += 1

                phase = int(
                    (arrival1 - earliest) // 2
                )

                phase_key = (phase, kind1)

                item = (
                    dep1, tof1, arrival1,
                    value1, kind1, used1
                )

                previous = phases.get(phase_key)

                if (
                    previous is None
                    or arrival1 + .005 * value1
                    < previous[2] + .005 * previous[3]
                ):
                    phases[phase_key] = item

        first_moves = sorted(
            phases.values(),
            key=lambda x: (
                x[2] + .005 * x[3],
                x[3]
            )
        )[:12]

        # Transferimi domate e re -> B.
        for (
            dep1, tof1, arrival1,
            value1, kind1, used1
        ) in first_moves:

            limit2 = (
                P.dv_threshold
                if used1 >= P.n_exceptions
                else P.exception_threshold
            )

            for wait in SECOND_WAITS:
                dep2 = arrival1 + wait

                if dep2 + .25 > deadline:
                    continue

                for tof2 in SECOND_TOFS:
                    arrival2 = dep2 + tof2

                    if arrival2 > deadline + 1e-8:
                        continue

                    value2 = dv(
                        tomato,
                        destination,
                        dep2,
                        tof2
                    )
                    calls += 1

                    if not math.isfinite(value2):
                        continue

                    gap = max(
                        0.0,
                        value2 - limit2
                    )

                    if (
                        best_near is None
                        or gap < best_near["gap"]
                    ):
                        best_near = {
                            "gap": round(gap, 4),
                            "dv2": round(value2, 4),
                            "gap_index": k,
                            "new_tomato": tomato,
                            "end": destination
                        }

                    if value2 >= limit2:
                        continue

                    legal_second += 1

                    kind2 = (
                        "NORMAL"
                        if value2 < P.dv_threshold
                        else "SPECIAL"
                    )

                    tr1 = {
                        "from": origin,
                        "to": tomato,
                        "start_time": dep1,
                        "tof": tof1,
                        "dv": value1,
                        "status": kind1
                    }

                    tr2 = {
                        "from": tomato,
                        "to": destination,
                        "start_time": dep2,
                        "tof": tof2,
                        "dv": value2,
                        "status": kind2
                    }

                    new_transfers = (
                        transfers[:k]
                        + [tr1, tr2]
                        + transfers[k + 1:]
                    )

                    final_day = (
                        arrival2
                        if k == len(transfers) - 1
                        else state["current_time"]
                    )

                    child = {
                        "route": (
                            route[:k + 1]
                            + [tomato]
                            + route[k + 1:]
                        ),
                        "transfers": new_transfers,
                        "special_used": (
                            used1
                            + int(kind2 == "SPECIAL")
                        ),
                        "current_time": final_day
                    }

                    if verify(child):
                        solutions.append(child)

                        if len(solutions) >= 5:
                            break

                if len(solutions) >= 5:
                    break

            if len(solutions) >= 5:
                break

        if len(solutions) >= 5:
            break

    return {
        "calls": calls,
        "first_legal": legal_first,
        "second_legal": legal_second,
        "best_near": best_near,
        "solutions": solutions
    }


print("\nFAZA 2 - KERKIMI I DOMATES SE 32-TE")
print("=" * 65, flush=True)

for number, (index, window) in enumerate(tasks, 1):
    if time.monotonic() - START >= LIMIT:
        break

    record = selected[index]
    key = record["id"] + ":" + str(window["gap"])

    if key in checkpoint["done"]:
        continue

    outcome = attempt_insertion(
        record, window
    )

    if outcome is None:
        break

    for solution in outcome["solutions"]:
        checkpoint["solutions"][
            identity(solution)
        ] = solution

    checkpoint["done"][key] = {
        k: v
        for k, v in outcome.items()
        if k != "solutions"
    }

    atomic(CHECKPOINT, checkpoint)

    near = outcome["best_near"]

    line = (
        f"[{number}/{len(tasks)}] "
        f"SPECIAL {record['state']['special_used']}/5 | "
        f"Dritarja {window['gap'] + 1} | "
        f"Koha {window['span']:.2f} dite | "
        f"Teste {outcome['calls']} | "
        f"Rruge32 {len(outcome['solutions'])}"
    )

    if near:
        line += f" | Gap {near['gap']:.2f}"

    print(line, flush=True)

    if outcome["solutions"]:
        print(
            "\nU GJET ITINERAR ME 32 DOMATE!",
            flush=True
        )
        break


# Riverifikojme cdo zgjidhje te ruajtur.
valid = [
    state
    for state in checkpoint["solutions"].values()
    if len(state["route"]) == 32
    and verify(state)
]

if valid:
    valid.sort(
        key=lambda state: (
            state["special_used"],
            state["current_time"]
        )
    )

    atomic(OUTPUT, {
        "level": 32,
        "verified": True,
        "frontier": valid
    })


print("\nPERMBLEDHJA E HAPIT 145")
print("=" * 65, flush=True)

print(
    "Dritare te kontrolluara:",
    len(checkpoint["done"]),
    "/",
    len(tasks),
    flush=True
)

print(
    "Rruge te verifikuara 32/49:",
    len(valid),
    flush=True
)

near_values = [
    entry["best_near"]
    for entry in checkpoint["done"].values()
    if entry["best_near"] is not None
]

if near_values:
    best = min(
        near_values,
        key=lambda item: item["gap"]
    )

    print(
        "Transferimi i dyte me afer kufirit:",
        best,
        flush=True
    )

partial = (
    len(checkpoint["done"]) < len(tasks)
    and not valid
)

print(
    "Kerkimi i pjesshem:",
    partial,
    flush=True
)

if valid:
    best = valid[0]

    print("\nREKORD I RI: 32/49!", flush=True)
    print(
        "SPECIAL:",
        best["special_used"], "/5",
        flush=True
    )
    print(
        "Dita perfundimtare:",
        round(best["current_time"], 4),
        flush=True
    )
    print(
        "Rruga:",
        best["route"],
        flush=True
    )
else:
    print(
        "Rekordi aktual mbetet: 31/49",
        flush=True
    )

print("HAPI 145 PERFUNDOI!", flush=True)
