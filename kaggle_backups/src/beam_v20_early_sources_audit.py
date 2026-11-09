
import json
import re
import math
from pathlib import Path
from collections import defaultdict
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

OUTPUT = RESULTS / "beam_v20_early_sources_audit.json"
SEEDS = RESULTS / "beam_v20_verified_early_seeds.json"

P = TomatoProblem(str(ROOT / "data/easy.kttsp"))

PATTERN = re.compile(
    r"(?:^|_)level[_-]?(\d+)(?:$|[_-])",
    re.IGNORECASE
)


def atomic(path, content):
    temp = path.with_suffix(".tmp")

    temp.write_text(
        json.dumps(content, separators=(",", ":")),
        encoding="utf-8"
    )

    temp.replace(path)


def verify(s):
    try:
        route = s["route"]
        transfers = s["transfers"]

        if (
            not isinstance(route, list)
            or not route
            or route[0] != 0
            or len(route) != len(set(route))
            or len(transfers) != len(route) - 1
            or len(route) > P.n_tomatoes
        ):
            return False

        day = 0.0
        specials = 0

        for i, tr in enumerate(transfers):
            departure = float(tr["start_time"])
            tof = float(tr["tof"])

            if (
                tr["from"] != route[i]
                or tr["to"] != route[i + 1]
                or departure < day - 1e-6
                or tof <= 0
                or departure + tof > P.max_time + 1e-6
            ):
                return False

            dv = float(P.compute_transfer(
                tr["from"],
                tr["to"],
                departure,
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

            if tr["status"] != status:
                return False

            day = departure + tof

        return (
            specials == s["special_used"]
            and specials <= P.n_exceptions
            and abs(day - s["current_time"]) < 1e-5
        )

    except Exception:
        return False


def extract_states(data):
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("frontier", "states", "routes"):
            value = data.get(key)

            if isinstance(value, list):
                return value

    return []


print("FAZA 1 - KERKIMI I SKEDAREVE")
print("=" * 65, flush=True)

files = []

for path in RESULTS.glob("*.json"):
    match = PATTERN.search(path.stem)

    if match is None:
        continue

    level = int(match.group(1))

    if 14 <= level <= 22:
        files.append((level, path))

files.sort(key=lambda item: (
    item[0],
    item[1].name
))

print(
    "Skedare te niveleve 14-22:",
    len(files),
    flush=True
)

assert files, (
    "Nuk u gjeten skedare te hershem."
)

# Lexojme rrugët e arkivuara.
candidates = []
sources = []
skipped = []

for expected_level, path in files:
    try:
        if path.stat().st_size > 80_000_000:
            skipped.append(path.name)
            continue

        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        states = extract_states(data)
        accepted = 0

        for state in states:
            if not isinstance(state, dict):
                continue

            needed = (
                "route",
                "transfers",
                "current_time",
                "special_used"
            )

            if not all(k in state for k in needed):
                continue

            if len(state["route"]) != expected_level:
                continue

            special = int(state["special_used"])

            if not 0 <= special <= 5:
                continue

            candidates.append({
                "source": path.name,
                "level": expected_level,
                "state": state
            })

            accepted += 1

        if accepted:
            sources.append({
                "file": path.name,
                "level": expected_level,
                "states": accepted
            })

    except Exception as error:
        skipped.append(
            f"{path.name}: {type(error).__name__}"
        )


print("\nFAZA 2 - ANALIZA E RRUGEVE")
print("=" * 65, flush=True)

print(
    "Skedare me rruge:",
    len(sources),
    flush=True
)

print(
    "Rruge te gjetura:",
    len(candidates),
    flush=True
)

# Heqim kopjet e njejta, por ruajme
# faza kohore te ndryshme.
unique = {}

for item in candidates:
    s = item["state"]
    route = s["route"]

    key = (
        tuple(route),
        int(s["special_used"]),
        round(float(s["current_time"]), 2)
    )

    if key not in unique:
        unique[key] = item

distinct = list(unique.values())

print(
    "Rruge te ndryshme:",
    len(distinct),
    flush=True
)

groups = defaultdict(list)

for item in distinct:
    groups[item["level"]].append(item)

for level in range(14, 23):
    states = groups[level]

    if not states:
        continue

    counts = {
        k: sum(
            int(item["state"]["special_used"]) == k
            for item in states
        )
        for k in range(6)
    }

    earliest = min(
        float(item["state"]["current_time"])
        for item in states
    )

    print(
        f"Niveli {level:2d} | "
        f"Rruge {len(states):4d} | "
        f"SPECIAL 0-2: "
        f"{sum(counts[k] for k in range(3))} | "
        f"3/5: {counts[3]} | "
        f"4/5: {counts[4]} | "
        f"5/5: {counts[5]} | "
        f"Dita min {earliest:.2f}",
        flush=True
    )


# Krijojme nje grup te larmishem kandidatësh
# per verifikim fizik me PyKEP.
shortlist = []

for level in range(14, 23):
    for used in range(4):
        group = sorted(
            [
                item for item in groups[level]
                if item["state"]["special_used"] == used
            ],
            key=lambda item: (
                float(item["state"]["current_time"]),
                item["state"]["route"][-1]
            )
        )

        endings = defaultdict(int)
        selected = 0

        for item in group:
            end = item["state"]["route"][-1]

            if endings[end] >= 3:
                continue

            shortlist.append(item)
            endings[end] += 1
            selected += 1

            if selected >= 12:
                break


print("\nFAZA 3 - VERIFIKIMI ME PYKEP")
print("=" * 65, flush=True)

verified = []
rejected = 0

for item in shortlist:
    if verify(item["state"]):
        verified.append(item)
    else:
        rejected += 1

print(
    "Kandidate te testuar:",
    len(shortlist),
    flush=True
)

print(
    "Rruge te verifikuara:",
    len(verified),
    flush=True
)

print(
    "Rruge te refuzuara:",
    rejected,
    flush=True
)

# Ruajme vetem gjendje qe kalojne verifikimin.
atomic(SEEDS, {
    "description": (
        "Verified archived early-stage seeds "
        "for a new resource-aware search"
    ),
    "frontier": verified
})

# Raport me numra per zgjedhjen e strategjise.
report = {
    "files_scanned": len(files),
    "sources": sources,
    "skipped": skipped,
    "candidates": len(candidates),
    "distinct": len(distinct),
    "tested": len(shortlist),
    "verified": len(verified),
    "levels": {}
}

for level in range(14, 23):
    group = groups[level]

    report["levels"][str(level)] = {
        "routes": len(group),
        "by_special": {
            str(used): sum(
                item["state"]["special_used"] == used
                for item in group
            )
            for used in range(6)
        }
    }

atomic(OUTPUT, report)


print("\nPERMBLEDHJA E HAPIT 144")
print("=" * 65, flush=True)

print(
    "Rruge unike te arkivuara:",
    len(distinct),
    flush=True
)

print(
    "Pikënisje te riverifikuara:",
    len(verified),
    flush=True
)

print("\nKANDIDATET ME PREMTUES")
print("=" * 65, flush=True)

promising = sorted(
    [
        item for item in verified
        if 16 <= item["level"] <= 21
        and item["state"]["special_used"] <= 2
    ],
    key=lambda item: (
        -item["level"],
        item["state"]["special_used"],
        item["state"]["current_time"]
    )
)

for item in promising[:15]:
    state = item["state"]

    print(
        f"Niveli {item['level']} | "
        f"Fund {state['route'][-1]} | "
        f"Dita {state['current_time']:.2f} | "
        f"SPECIAL {state['special_used']}/5 | "
        f"Burimi {item['source']}",
        flush=True
    )

if not promising:
    print(
        "Nuk u gjeten kandidate me maksimum "
        "2 SPECIAL ne nivelet 16-21.",
        flush=True
    )

print(
    "\nRekordi i projektit mbetet: 31/49",
    flush=True
)

print("HAPI 144 PERFUNDOI!", flush=True)
