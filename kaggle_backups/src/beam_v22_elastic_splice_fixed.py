import json, math, time, hashlib
from pathlib import Path
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RES = ROOT / "results"
CP = RES / "beam_v22_elastic_splice_fixed_checkpoint.json"
OUT = RES / "beam_v22_elastic_splice_fixed_level_32.json"

P = TomatoProblem(str(ROOT / "data/easy.kttsp"))
START = time.monotonic()

WAITS = (0, .25, .5, 1, 2, 4, 8)
TOFS = (.25, .5, .75, 1, 1.5, 2, 3, 5, 8, 12, 18)

def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":")))
    tmp.replace(path)

def dv(a, b, depart, tof):
    try:
        value = float(P.compute_transfer(a, b, float(depart), float(tof)))
        return value if math.isfinite(value) and value >= 0 else math.inf
    except Exception:
        return math.inf

def verify(s):
    try:
        route, transfers = s["route"], s["transfers"]
        day, specials = 0.0, 0

        if not route or route[0] != 0:
            return False
        if len(set(route)) != len(route):
            return False
        if len(transfers) != len(route) - 1:
            return False

        for i, tr in enumerate(transfers):
            depart = float(tr["start_time"])
            tof = float(tr["tof"])

            if tr["from"] != route[i] or tr["to"] != route[i + 1]:
                return False
            if depart < day - 1e-6 or tof <= 0:
                return False
            if depart + tof > P.max_time + 1e-6:
                return False

            value = dv(route[i], route[i + 1], depart, tof)

            if not math.isfinite(value):
                return False
            if abs(value - float(tr["dv"])) > .1:
                return False
            if value >= P.exception_threshold:
                return False

            status = "NORMAL" if value < P.dv_threshold else "SPECIAL"

            if tr["status"] != status:
                return False

            specials += status == "SPECIAL"
            day = depart + tof

        return (
            specials == s["special_used"]
            and specials <= P.n_exceptions
            and abs(day - s["current_time"]) < 1e-5
        )
    except Exception:
        return False

def collect(obj, found):
    if isinstance(obj, dict):
        keys = ("route", "transfers", "current_time", "special_used")
        if all(key in obj for key in keys):
            if isinstance(obj["route"], list) and len(obj["route"]) == 31:
                found.append(obj)
        else:
            for value in obj.values():
                collect(value, found)
    elif isinstance(obj, list):
        for value in obj:
            collect(value, found)

def make_campaign():
    paths = set(RES.glob("*level_31.json"))

    for pattern in (
        "*retime*checkpoint.json",
        "*reorder*checkpoint.json",
        "*continue_checkpoint.json"
    ):
        paths.update(RES.glob(pattern))

    unique = {}

    for path in sorted(paths):
        if path.stat().st_size > 35_000_000:
            continue

        try:
            found = []
            collect(json.loads(path.read_text()), found)

            for state in found:
                if state["special_used"] not in (4, 5):
                    continue
                if not verify(state):
                    continue

                key = (
                    tuple(state["route"]),
                    state["special_used"],
                    round(state["current_time"], 2)
                )
                unique.setdefault(key, state)
        except (OSError, ValueError, TypeError, KeyError):
            continue

    states = sorted(
        unique.values(),
        key=lambda s: (s["special_used"], s["current_time"])
    )

    selected = []
    tails = set()

    for state in states:
        tail = tuple(state["route"][-7:])
        if tail in tails:
            continue

        selected.append(state)
        tails.add(tail)

        if len(selected) == 10:
            break

    for state in states:
        if len(selected) >= 10:
            break
        if state not in selected:
            selected.append(state)

    if not selected:
        raise RuntimeError("Nuk ka rruge 31 te verifikuara.")

    bundles = []

    for si, state in enumerate(selected):
        transfers = state["transfers"]
        windows = []

        for k in range(len(transfers) - 1):
            earliest = (
                0.0 if k == 0
                else transfers[k - 1]["start_time"]
                + transfers[k - 1]["tof"]
            )

            deadline = (
                transfers[k + 2]["start_time"]
                if k + 2 < len(transfers)
                else P.max_time
            )

            span = deadline - earliest

            if span < 2:
                continue

            freed = sum(
                tr["status"] == "SPECIAL"
                for tr in transfers[k:k + 2]
            )

            windows.append((
                span + 8 * freed, k, earliest, deadline
            ))

        windows.sort(reverse=True)

        missing = [
            tomato for tomato in range(P.n_tomatoes)
            if tomato not in state["route"]
        ]

        for _, k, earliest, deadline in windows[:4]:
            bundle = [
                (si, k, earliest, deadline, tomato, pos)
                for tomato in missing
                for pos in (0, 1)
            ]

            bundle.sort(
                key=lambda x: (
                    (x[4] * 17 + si * 7 + k * 11) % P.n_tomatoes,
                    x[5]
                )
            )

            bundles.append(bundle)

    tasks = []

    for j in range(36):
        for bundle in bundles:
            if j < len(bundle):
                tasks.append(bundle[j])
            if len(tasks) >= 120:
                break
        if len(tasks) >= 120:
            break

    return {
        "version": 2,
        "seeds": selected,
        "tasks": tasks,
        "done": {},
        "solutions": {}
    }

def prune(states, earliest, cap):
    grouped = {}

    for state in states:
        key = (
            state["used"],
            int((state["day"] - earliest) / 1.5)
        )

        score = state["day"] + .004 * state["sum"]

        if key not in grouped or score < grouped[key][0]:
            grouped[key] = (score, state)

    result = [item[1] for item in grouped.values()]

    result.sort(
        key=lambda s: (
            s["used"] * 80 + s["day"] + .006 * s["sum"]
        )
    )

    return result[:cap]

def attempt(seed, task):
    _, k, earliest, deadline, tomato, position = task

    route = seed["route"]
    transfers = seed["transfers"]
    a, b, c = route[k:k + 3]

    nodes = (
        [a, tomato, b, c]
        if position == 0
        else [a, b, tomato, c]
    )

    freed = sum(
        tr["status"] == "SPECIAL"
        for tr in transfers[k:k + 2]
    )

    fixed = seed["special_used"] - freed

    frontier = [{
        "day": earliest,
        "used": 0,
        "sum": 0.0,
        "tr": []
    }]

    calls = 0

    for step in range(3):
        expanded = []

        for state in frontier:
            for wait in WAITS:
                depart = state["day"] + wait

                for tof in TOFS:
                    arrival = depart + tof

                    if arrival + (2 - step) * .25 > deadline + 1e-8:
                        continue

                    value = dv(
                        nodes[step],
                        nodes[step + 1],
                        depart,
                        tof
                    )
                    calls += 1

                    if value >= P.exception_threshold:
                        continue

                    status = (
                        "NORMAL" if value < P.dv_threshold
                        else "SPECIAL"
                    )

                    used = state["used"] + (status == "SPECIAL")

                    if fixed + used > P.n_exceptions:
                        continue

                    transfer = {
                        "from": nodes[step],
                        "to": nodes[step + 1],
                        "start_time": depart,
                        "tof": tof,
                        "dv": value,
                        "status": status
                    }

                    expanded.append({
                        "day": arrival,
                        "used": used,
                        "sum": state["sum"] + value,
                        "tr": state["tr"] + [transfer]
                    })

        if not expanded:
            return calls, None

        frontier = prune(expanded, earliest, 20)

    for state in sorted(
        frontier,
        key=lambda s: (s["used"], s["day"])
    ):
        child = {
            "route": (
                route[:k + 1]
                + nodes[1:3]
                + route[k + 2:]
            ),
            "transfers": (
                transfers[:k]
                + state["tr"]
                + transfers[k + 2:]
            ),
            "special_used": fixed + state["used"],
            "current_time": (
                state["day"]
                if k + 2 == len(route) - 1
                else seed["current_time"]
            )
        }

        if len(child["route"]) == 32 and verify(child):
            return calls, child

    return calls, None

checkpoint = (
    json.loads(CP.read_text())
    if CP.exists()
    else make_campaign()
)

assert checkpoint["version"] == 2
save(CP, checkpoint)

print(
    "FAZA 1 | Rruge 31:",
    len(checkpoint["seeds"]),
    "| Prova:", len(checkpoint["tasks"]),
    flush=True
)

for i, task in enumerate(checkpoint["tasks"]):
    if time.monotonic() - START >= 108:
        break

    if str(i) in checkpoint["done"]:
        continue

    calls, solution = attempt(
        checkpoint["seeds"][task[0]],
        task
    )

    checkpoint["done"][str(i)] = {
        "calls": calls,
        "found": solution is not None
    }

    if solution:
        encoded = json.dumps(solution, sort_keys=True)
        uid = hashlib.sha256(encoded.encode()).hexdigest()[:20]
        checkpoint["solutions"][uid] = solution

    save(CP, checkpoint)

    if len(checkpoint["done"]) % 5 == 0 or solution:
        print(
            "Prova", len(checkpoint["done"]),
            "/", len(checkpoint["tasks"]),
            "| DV teste", calls,
            "| Rruge 32", len(checkpoint["solutions"]),
            flush=True
        )

    if solution:
        break

valid = [
    state for state in checkpoint["solutions"].values()
    if len(state["route"]) == 32 and verify(state)
]

if valid:
    save(OUT, {
        "level": 32,
        "verified": True,
        "frontier": valid
    })

print("\nPERMBLEDHJA E HAPIT 146")
print("Prova:", len(checkpoint["done"]), "/", len(checkpoint["tasks"]))
print("Rruge 32 te verifikuara:", len(valid))
print(
    "Kerkimi i pjesshem:",
    len(checkpoint["done"]) < len(checkpoint["tasks"]) and not valid
)
print(
    "REKORD I RI: 32/49!"
    if valid else
    "Rekordi mbetet: 31/49"
)
