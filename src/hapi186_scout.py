
import json, math, os, sys, time
from pathlib import Path
from collections import defaultdict

ROOT = Path("/kaggle/working/esa-spoc4")
R = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))

from utils_users import TomatoProblem

header = next(
    line.split()
    for line in (ROOT / "data/easy.kttsp").read_text().splitlines()
    if line.startswith("p kttsp ")
)

T0, MIN_TOF, TMAX, NORMAL, SPECIAL, MAX_SP = map(
    float, header[2:8]
)
MAX_SP = int(MAX_SP)

P = TomatoProblem(str(ROOT / "data/easy.kttsp"))
HARD = {4, 6, 11, 16, 17, 24, 26, 27, 32, 35, 38}

deadline = time.monotonic() + 105

def dv(a, b, dep, tof):
    try:
        val = float(P.compute_transfer(
            int(a), int(b), float(dep), float(tof)
        ))
        return val if math.isfinite(val) and val >= 0 else math.inf
    except (ValueError, RuntimeError, OverflowError, ZeroDivisionError):
        return math.inf

def state(obj):
    if not isinstance(obj, dict):
        return None

    if isinstance(obj.get("frontier"), list):
        if not obj["frontier"]:
            return None
        obj = obj["frontier"][0]

    route = obj.get("route")
    legs = obj.get("transfers")

    if not isinstance(route, list) or not isinstance(legs, list):
        return None

    if (not route or route[0] != 0 or
        len(route) != len(legs) + 1 or
        len(set(route)) != len(route)):
        return None

    if any(not isinstance(x, int) or x < 0 or x >= 49
           for x in route):
        return None

    try:
        t = (
            float(legs[-1]["start_time"]) + float(legs[-1]["tof"])
            if legs else T0
        )
        sp = sum(
            float(x["dv"]) > NORMAL + 1e-6 for x in legs
        )
    except (ValueError, TypeError, KeyError):
        return None

    if sp > MAX_SP or t > TMAX + 1e-6:
        return None

    return {
        "route": route,
        "transfers": legs,
        "t": t,
        "sp": sp
    }

def verify(s):
    t = T0
    sp = 0

    for i, tr in enumerate(s["transfers"]):
        try:
            dep = float(tr["start_time"])
            tof = float(tr["tof"])
            old = float(tr["dv"])

            if dep < t - 1e-5 or tof < MIN_TOF - 1e-6:
                return False

            a, b = s["route"][i:i+2]

            if (int(tr["from"]), int(tr["to"])) != (a, b):
                return False

            actual = dv(a, b, dep, tof)

            if actual > SPECIAL + 1e-6:
                return False

            if abs(actual - old) > max(.003, 2e-5 * abs(old)):
                return False

            sp += int(actual > NORMAL + 1e-6)
            t = dep + tof

        except (ValueError, KeyError, TypeError):
            return False

    return (
        sp == s["sp"]
        and t <= TMAX + 1e-6
        and abs(t - s["t"]) < 1e-4
    )

# Mbledhim itineraret alternative nga backup-et
archive = defaultdict(list)

for file in sorted(R.glob("*.json")):
    if file.stat().st_size > 2_000_000:
        continue

    try:
        obj = json.loads(file.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError, OSError):
        continue

    records = (
        obj["frontier"]
        if isinstance(obj, dict) and
           isinstance(obj.get("frontier"), list)
        else [obj]
    )

    for item in records[:60]:
        s = state(item)
        if s and 24 <= len(s["route"]) <= 38:
            archive[len(s["route"])].append(s)

selected = []
seen = set()

for level in [25, 28, 31, 34, 37, 38]:
    candidates = archive.get(level, [])

    candidates.sort(
        key=lambda s: (
            6 * s["sp"] + s["t"]
            - 5 * len(set(s["route"]) & HARD)
        )
    )

    added = 0

    for s in candidates:
        key = (tuple(s["route"]), round(s["t"], 2))

        if key in seen:
            continue

        seen.add(key)

        if not verify(s):
            continue

        selected.append(s)
        added += 1

        if added >= 10:
            break

assert selected, "Nuk u gjeten itinerare te verifikuara"

print("=" * 60)
print("HAPI 186 - ANALIZA E DOMATEVE TE VESHTIRA")
print("=" * 60)
print("Itinerare te verifikuara:", len(selected))
print("Nivele:", sorted({len(s["route"]) for s in selected}))
print("Kufiri:", TMAX, "dite")
print("SPECIAL:", MAX_SP)

waits = [0, .25, .75, 1.5, 3, 5, 8, 12, 18]
flights = sorted(set(
    x for x in [
        MIN_TOF, .25, .5, .75, 1, 1.5,
        2, 3, 4, 6, 8, 11, 15
    ] if x >= MIN_TOF
))

report = {
    j: {"normal": 0, "special": 0}
    for j in sorted(HARD)
}

best_candidates = {}
calls = 0

for s in selected:
    if time.monotonic() > deadline:
        break

    a = s["route"][-1]

    for j in sorted(HARD - set(s["route"])):
        if time.monotonic() > deadline:
            break

        found_normal = None
        found_special = None

        for wait in waits:
            dep = round(s["t"] + wait, 6)

            for tof in flights:
                if dep + tof > TMAX:
                    continue

                val = dv(a, j, dep, tof)
                calls += 1

                option = (dep + tof, dep, tof, val)

                if val <= NORMAL + 1e-6:
                    if (found_normal is None or
                        option[0] < found_normal[0]):
                        found_normal = option

                elif (s["sp"] < MAX_SP and
                      val <= SPECIAL + 1e-6):
                    if (found_special is None or
                        option[0] < found_special[0]):
                        found_special = option

        for option, kind in [
            (found_normal, "normal"),
            (found_special, "special")
        ]:
            if option is None:
                continue

            report[j][kind] += 1

            arrival, dep, tof, val = option
            extra_special = int(kind == "special")

            leg = {
                "from": a,
                "to": j,
                "start_time": dep,
                "tof": tof,
                "dv": val,
                "status": kind.upper()
            }

            candidate = {
                "route": s["route"] + [j],
                "transfers": s["transfers"] + [leg],
                "current_time": arrival,
                "special_used": s["sp"] + extra_special,
                "source_level": len(s["route"])
            }

            old = best_candidates.get(j)

            rank = (
                -len(candidate["route"]),
                candidate["special_used"],
                arrival
            )

            if old is None or rank < old[0]:
                best_candidates[j] = (rank, candidate)

output = {
    "method": "hapi186_scout",
    "sampled_only": True,
    "tested_states": len(selected),
    "dv_calculations": calls,
    "reachability": report,
    "candidate_seeds": {
        str(j): v[1]
        for j, v in best_candidates.items()
    }
}

target = R / "hapi186_scout.json"
tmp = target.with_suffix(".tmp")

tmp.write_text(
    json.dumps(output, indent=2, ensure_ascii=False),
    encoding="utf-8"
)
os.replace(tmp, target)

print("\n--- REZULTATET SIPAS DOMATEVE ---")

for j, info in sorted(
    report.items(),
    key=lambda x: (
        x[1]["normal"] + x[1]["special"],
        x[0]
    )
):
    print(
        "Domatja", j,
        "| NORMAL:", info["normal"],
        "| SPECIAL:", info["special"]
    )

print("\nPERMBLEDHJE HAPI 186")
print("Itinerare te kontrolluara:", len(selected))
print("Llogaritje DV:", calls)
print("Kandidate te ruajtur:", len(best_candidates))
print("Skedari:", target)
print("Rekordi 38/49 MBETET I PAPREKUR.")
print("Shenim: 0 mundesi do te thote 0 ne mostren e testuar.")
