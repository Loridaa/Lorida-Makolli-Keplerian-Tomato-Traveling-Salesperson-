
import hashlib
import json
import math
import time
from pathlib import Path
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"
OUT = RESULTS / "beam_v12_retime_checkpoint.json"
SNAP31 = RESULTS / "beam_v12_retimed_level_31.json"
SNAP32 = RESULTS / "beam_v12_retimed_level_32.json"

P = TomatoProblem(str(ROOT / "data/easy.kttsp"))

START = time.monotonic()
LIMIT = 125
TAIL = 7
WIDTH = 12

WAITS = [0, .25, .5, 1, 2, 3, 5, 8]
TOFS = [.25, .5, .75, 1, 1.5, 2, 3, 5, 8, 13, 18]

PROBE_WAITS = [0, .25, .5, 1, 2, 3, 5, 8, 13]
PROBE_TOFS = [.25, .5, .75, 1, 1.5, 2, 3, 5, 8, 13, 18, 24]

def fingerprint(s):
    payload = [
        s["route"],
        round(s["current_time"], 7),
        s["special_used"]
    ]
    return hashlib.sha256(
        json.dumps(payload).encode()
    ).hexdigest()[:20]

def status(dv, used):
    if not math.isfinite(dv):
        return None
    if dv < P.dv_threshold:
        return "NORMAL"
    if dv < P.exception_threshold and used < P.n_exceptions:
        return "SPECIAL"
    return None

def validate(s):
    try:
        route = s["route"]
        transfers = s["transfers"]

        if (
            not route
            or route[0] != 0
            or len(set(route)) != len(route)
            or len(transfers) != len(route) - 1
            or len(route) > P.n_tomatoes
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
                route[i], route[i + 1], depart, tof
            ))

            if (
                not math.isfinite(dv)
                or abs(dv - float(tr["dv"])) > 0.1
            ):
                return False

            actual = (
                "NORMAL" if dv < P.dv_threshold
                else "SPECIAL" if dv < P.exception_threshold
                else "INVALID"
            )

            if actual != tr["status"]:
                return False

            used += actual == "SPECIAL"
            day = depart + tof

        return (
            used == s["special_used"]
            and used <= P.n_exceptions
            and abs(day - s["current_time"]) < 1e-5
        )

    except Exception:
        return False

def atomic(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(data, separators=(",", ":"))
    )
    temp.replace(path)

def save():
    atomic(OUT, report)

def make_child(s, target, depart, tof, dv, st):
    transfer = {
        "from": s["route"][-1],
        "to": target,
        "start_time": depart,
        "tof": tof,
        "dv": dv,
        "status": st
    }

    return {
        "route": s["route"] + [target],
        "transfers": s["transfers"] + [transfer],
        "current_time": depart + tof,
        "special_used": (
            s["special_used"] + int(st == "SPECIAL")
        )
    }

# Zgjedhim rruget per eksperiment.
source_hashes = {}
seeds = []

for lane, groups in [
    ("A14", [(4, 6), (5, 2)]),
    ("B12", [(5, 2)])
]:
    path = RESULTS / f"beam_v11_{lane}_level_31.json"
    raw = path.read_bytes()

    source_hashes[path.name] = hashlib.sha256(
        raw
    ).hexdigest()

    states = json.loads(raw)["frontier"]

    for used, number in groups:
        options = sorted(
            (
                s for s in states
                if s["special_used"] == used
            ),
            key=lambda s: s["current_time"]
        )

        for state in options[:number]:
            if not validate(state):
                raise ValueError(
                    f"Rruge e pavlefshme ne {lane}"
                )

            seeds.append((lane, state))

CONFIG = {
    "source_hashes": source_hashes,
    "seed_ids": [fingerprint(s) for _, s in seeds],
    "tail": TAIL,
    "width": WIDTH,
    "waits": WAITS,
    "tofs": TOFS,
    "probe_waits": PROBE_WAITS,
    "probe_tofs": PROBE_TOFS,
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
        "seeds": {},
        "probes": {},
        "retimed": [],
        "level32": []
    }
    save()

def advance(states, target, anchor):
    children = []

    for s in states:
        day = s["current_time"]
        used = s["special_used"]
        origin = s["route"][-1]

        waits = sorted(set(
            WAITS + [
                max(0.0, float(anchor["start_time"]) - day)
            ]
        ))

        tofs = sorted(set(
            TOFS + [float(anchor["tof"])]
        ))

        choices = []

        for wait in waits:
            for tof in tofs:
                depart = day + wait

                if depart + tof > P.max_time:
                    continue

                try:
                    dv = float(P.compute_transfer(
                        origin, target, depart, tof
                    ))
                except Exception:
                    continue

                st = status(dv, used)

                if st:
                    choices.append((
                        st, depart + tof, dv, depart, tof
                    ))

        for st in ["NORMAL", "SPECIAL"]:
            group = [
                c for c in choices if c[0] == st
            ]

            if not group:
                continue

            chosen = sorted(
                group,
                key=lambda c: (c[1], c[2])
            )[:3]

            lowest_dv = min(
                group,
                key=lambda c: (c[2], c[1])
            )

            if lowest_dv not in chosen:
                chosen.append(lowest_dv)

            for _, _, dv, depart, tof in chosen:
                children.append(
                    make_child(
                        s, target, depart, tof, dv, st
                    )
                )

    unique = {}

    for s in children:
        key = (
            s["special_used"],
            round(s["current_time"] * 4) / 4
        )

        if (
            key not in unique
            or s["current_time"] <
            unique[key]["current_time"]
        ):
            unique[key] = s

    ranked = sorted(
        unique.values(),
        key=lambda s: (
            s["special_used"],
            s["current_time"]
        )
    )

    picked = []
    counts = {}

    for s in ranked:
        used = s["special_used"]

        if counts.get(used, 0) >= 8:
            continue

        picked.append(s)
        counts[used] = counts.get(used, 0) + 1

        if len(picked) >= WIDTH:
            break

    return picked

print("FAZA 1 - RITIMIMI I 7 TRANSFERIMEVE")
print("=" * 65, flush=True)

for index, (name, original) in enumerate(seeds, 1):
    if time.monotonic() - START >= LIMIT:
        break

    sid = fingerprint(original)

    if sid not in report["seeds"]:
        prefix_len = len(original["route"]) - TAIL

        prefix_transfers = original["transfers"][
            :prefix_len - 1
        ]

        prefix_day = (
            prefix_transfers[-1]["start_time"]
            + prefix_transfers[-1]["tof"]
            if prefix_transfers else 0.0
        )

        prefix = {
            "route": original["route"][:prefix_len],
            "transfers": prefix_transfers,
            "current_time": prefix_day,
            "special_used": sum(
                t["status"] == "SPECIAL"
                for t in prefix_transfers
            )
        }

        report["seeds"][sid] = {
            "lane": name,
            "step": 0,
            "frontier": [prefix],
            "complete": False,
            "baseline_day": original["current_time"],
            "baseline_used": original["special_used"]
        }
        save()

    entry = report["seeds"][sid]

    if entry["complete"]:
        continue

    for step in range(entry["step"], TAIL):
        if time.monotonic() - START >= LIMIT:
            break

        target = original["route"][
            len(original["route"]) - TAIL + step
        ]

        anchor = original["transfers"][
            len(original["transfers"]) - TAIL + step
        ]

        states = advance(
            entry["frontier"], target, anchor
        )

        entry["frontier"] = states
        entry["step"] = step + 1
        save()

        print(
            f"{name} [{index}/{len(seeds)}] | "
            f"Transferimi {step + 1}/{TAIL} | "
            f"Rruge {len(states)}",
            flush=True
        )

        if not states:
            break

    if entry["step"] == TAIL:
        verified = [
            s for s in entry["frontier"]
            if validate(s)
        ]

        entry["frontier"] = verified
        entry["complete"] = True
        save()

        if verified:
            best = min(
                verified,
                key=lambda s: (
                    s["special_used"],
                    s["current_time"]
                )
            )

            print(
                f"  PERFUNDOI | "
                f"Dita {best['current_time']:.2f} | "
                f"SPECIAL {best['special_used']}/5 | "
                f"Origjinali {original['current_time']:.2f}",
                flush=True
            )

# Ruajme edhe rruget origjinale.
retimed = {}

for name, s in seeds:
    retimed[fingerprint(s)] = s

for entry in report["seeds"].values():
    if entry["complete"]:
        for s in entry["frontier"]:
            retimed[fingerprint(s)] = s

report["retimed"] = list(retimed.values())
save()

atomic(SNAP31, {
    "level": 31,
    "verified": True,
    "frontier": report["retimed"]
})

all_complete = (
    len(report["seeds"]) == len(seeds)
    and all(
        entry["complete"]
        for entry in report["seeds"].values()
    )
)

# FAZA 2 - Provojme transferimin e 32-te.
if all_complete:
    print(
        "\nFAZA 2 - KERKIMI I DOMATES SE 32-TE",
        flush=True
    )
    print("=" * 65, flush=True)

    ranked = sorted(
        report["retimed"],
        key=lambda s: (
            s["special_used"],
            s["current_time"]
        )
    )

    selected = []
    end_counts = {}

    for s in ranked:
        key = (
            s["route"][-1],
            s["special_used"]
        )

        if end_counts.get(key, 0) >= 6:
            continue

        selected.append(s)
        end_counts[key] = end_counts.get(key, 0) + 1

        if len(selected) >= 18:
            break

    for index, s in enumerate(selected, 1):
        if time.monotonic() - START >= LIMIT:
            break

        sid = fingerprint(s)

        if sid in report["probes"]:
            continue

        tested = 0
        found = []
        minimum_dv = float("inf")

        for target in range(P.n_tomatoes):
            if target in s["route"]:
                continue

            best_normal = None
            best_special = None

            for wait in PROBE_WAITS:
                for tof in PROBE_TOFS:
                    depart = s["current_time"] + wait

                    if depart + tof > P.max_time:
                        continue

                    try:
                        dv = float(P.compute_transfer(
                            s["route"][-1],
                            target,
                            depart,
                            tof
                        ))
                        tested += 1
                    except Exception:
                        continue

                    if math.isfinite(dv):
                        minimum_dv = min(minimum_dv, dv)

                    st = status(dv, s["special_used"])

                    if not st:
                        continue

                    child = make_child(
                        s, target, depart, tof, dv, st
                    )

                    if st == "NORMAL":
                        if (
                            best_normal is None
                            or child["current_time"] <
                            best_normal["current_time"]
                        ):
                            best_normal = child

                    if st == "SPECIAL":
                        if (
                            best_special is None
                            or child["current_time"] <
                            best_special["current_time"]
                        ):
                            best_special = child

            for candidate in [best_normal, best_special]:
                if (
                    candidate is not None
                    and validate(candidate)
                ):
                    found.append(candidate)

        report["probes"][sid] = {
            "tested": tested,
            "minimum_dv": (
                minimum_dv
                if math.isfinite(minimum_dv)
                else None
            ),
            "children": found
        }

        save()

        print(
            f"PROBE [{index}/{len(selected)}] | "
            f"Dita {s['current_time']:.2f} | "
            f"SPECIAL {s['special_used']}/5 | "
            f"Femije32 {len(found)}",
            flush=True
        )

else:
    print(
        "\nRitimimi eshte i pjesshem. "
        "Ekzekuto perseri te njejten qelize.",
        flush=True
    )

found32 = {}

for probe in report["probes"].values():
    for s in probe["children"]:
        found32[fingerprint(s)] = s

report["level32"] = list(found32.values())
save()

if found32:
    atomic(SNAP32, {
        "level": 32,
        "verified": True,
        "frontier": list(found32.values())
    })

print("\nPERMBLEDHJA E HAPIT 126")
print("=" * 65, flush=True)

completed = sum(
    entry["complete"]
    for entry in report["seeds"].values()
)

print(
    f"Rruge fillestare te perfunduara: "
    f"{completed}/{len(seeds)}",
    flush=True
)

print(
    "Itinerare te vlefshme me 31 domate:",
    len(report["retimed"]),
    flush=True
)

print(
    "Kontrolle te domates se 32-te:",
    len(report["probes"]),
    flush=True
)

print(
    "Itinerare te verifikuara me 32 domate:",
    len(report["level32"]),
    flush=True
)

for used in [3, 4, 5]:
    group = [
        s for s in report["retimed"]
        if s["special_used"] == used
    ]

    if group:
        earliest = min(
            s["current_time"] for s in group
        )

        print(
            f"Niveli 31 | SPECIAL {used}/5 | "
            f"Rruge {len(group)} | "
            f"Dita minimale {earliest:.2f}",
            flush=True
        )

if found32:
    best = min(
        found32.values(),
        key=lambda s: (
            s["special_used"],
            s["current_time"]
        )
    )

    print(
        "REKORD I RI I VERIFIKUAR: 32/49 | "
        f"Dita {best['current_time']:.2f} | "
        f"SPECIAL {best['special_used']}/5",
        flush=True
    )

else:
    print(
        "Rekordi i verifikuar mbetet 31/49.",
        flush=True
    )

print(
    "Nese analiza eshte e pjesshme, "
    "ekzekuto perseri te njejten qelize.",
    flush=True
)

print("\nHAPI 126 PERFUNDOI!", flush=True)
