
import hashlib
import json
import math
import time
from pathlib import Path
from collections import defaultdict
from scipy.optimize import minimize
from utils_users import TomatoProblem

ROOT = Path("/kaggle/working/esa-spoc4")
RESULTS = ROOT / "results"

OUT = RESULTS / "beam_v12_reorder_tail7_checkpoint.json"
SNAP31 = RESULTS / "beam_v12_reordered_level_31.json"
SNAP32 = RESULTS / "beam_v12_reordered_level_32.json"

P = TomatoProblem(str(ROOT / "data/easy.kttsp"))

START = time.monotonic()
LIMIT = 125.0
TAIL = 7
WIDTH = 48
MAX_PROBES = 14

WAITS = [0, .25, .5, 1, 2, 3, 5, 8, 12, 18, 26, 36]
TOFS = [.25, .5, .75, 1, 1.5, 2, 3, 5, 8, 13, 18, 24, 32]

PROBE_WAITS = [0, .25, .5, 1, 2, 3, 5, 8, 12, 18]
PROBE_TOFS = [.25, .5, .75, 1, 1.5, 2, 3, 5, 8, 13, 18, 24, 32]


def uid(s):
    blob = [
        s["route"],
        round(float(s["current_time"]), 7),
        int(s["special_used"])
    ]
    return hashlib.sha256(
        json.dumps(blob).encode()
    ).hexdigest()[:20]


def atomic(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(obj, separators=(",", ":"))
    )
    tmp.replace(path)


def verify(s):
    try:
        r = s["route"]
        ts = s["transfers"]

        if (
            not r
            or r[0] != 0
            or len(r) != len(set(r))
            or len(ts) != len(r) - 1
        ):
            return False

        if any(
            type(t) is not int
            or t < 0
            or t >= P.n_tomatoes
            for t in r
        ):
            return False

        day = 0.0
        used = 0

        for k, tr in enumerate(ts):
            start = float(tr["start_time"])
            tof = float(tr["tof"])

            if (
                tr["from"] != r[k]
                or tr["to"] != r[k + 1]
                or start < day - 1e-6
                or tof <= 0
                or start + tof > P.max_time + 1e-6
            ):
                return False

            dv = float(P.compute_transfer(
                r[k], r[k + 1], start, tof
            ))

            if (
                not math.isfinite(dv)
                or abs(dv - float(tr["dv"])) > 0.1
            ):
                return False

            st = (
                "NORMAL" if dv < P.dv_threshold
                else "SPECIAL"
                if dv < P.exception_threshold
                else "INVALID"
            )

            if st != tr["status"]:
                return False

            used += st == "SPECIAL"
            day = start + tof

        return (
            used == s["special_used"]
            and used <= P.n_exceptions
            and abs(day - float(s["current_time"])) < 1e-5
        )

    except Exception:
        return False


def child(s, move, st):
    tr = {
        k: move[k]
        for k in (
            "from", "to", "start_time", "tof", "dv"
        )
    }

    tr["status"] = st

    return {
        "route": s["route"] + [move["to"]],
        "transfers": s["transfers"] + [tr],
        "current_time": (
            move["start_time"] + move["tof"]
        ),
        "special_used": (
            s["special_used"] + int(st == "SPECIAL")
        )
    }


def moves(s, targets, waits, tofs, do_local=False):
    day = float(s["current_time"])
    used = s["special_used"]
    origin = s["route"][-1]
    window = P.max_time - day

    answer = []
    evaluations = 0
    near = []

    for target in targets:
        options = {
            "NORMAL": None,
            "SPECIAL": None
        }
        closest = None

        for wait in waits:
            for tof in tofs:
                if wait + tof > window:
                    continue

                depart = day + wait

                try:
                    dv = float(P.compute_transfer(
                        origin, target, depart, tof
                    ))
                    evaluations += 1
                except Exception:
                    continue

                if not math.isfinite(dv):
                    continue

                if closest is None or dv < closest[0]:
                    closest = (dv, wait, tof, target)

                st = (
                    "NORMAL"
                    if dv < P.dv_threshold
                    else "SPECIAL"
                    if (
                        dv < P.exception_threshold
                        and used < P.n_exceptions
                    )
                    else None
                )

                if st is None:
                    continue

                move = {
                    "from": origin,
                    "to": target,
                    "start_time": depart,
                    "tof": tof,
                    "dv": dv
                }

                old = options[st]

                if (
                    old is None
                    or (
                        depart + tof, dv
                    ) < (
                        old["start_time"] + old["tof"],
                        old["dv"]
                    )
                ):
                    options[st] = move

        for st, move in options.items():
            if move is not None:
                answer.append((st, move))

        if (
            do_local
            and options["NORMAL"] is None
            and closest is not None
            and 100 <= closest[0] < 160
        ):
            near.append(closest)

    # Optimizim lokal per nje transferim
    # qe eshte afer kufirit NORMAL.
    if do_local:
        for _, wait0, tof0, target in sorted(near)[:1]:

            def objective(x):
                wait, tof = map(float, x)

                if (
                    wait < 0
                    or tof < .2
                    or wait + tof > window
                ):
                    return 1e8

                try:
                    return float(P.compute_transfer(
                        origin, target,
                        day + wait, tof
                    ))
                except Exception:
                    return 1e8

            try:
                result = minimize(
                    objective,
                    [wait0, tof0],
                    method="Nelder-Mead",
                    options={
                        "maxiter": 35,
                        "maxfev": 55
                    }
                )

                wait, tof = map(float, result.x)
                dv = objective([wait, tof])

                if (
                    math.isfinite(dv)
                    and dv < P.dv_threshold
                ):
                    answer.append((
                        "NORMAL",
                        {
                            "from": origin,
                            "to": target,
                            "start_time": day + wait,
                            "tof": tof,
                            "dv": dv
                        }
                    ))

            except Exception:
                pass

    return answer, evaluations


def prune(children, suffix, width):
    # Per domatet e njejta te vizituara,
    # fundin e njejte dhe numrin e njejte SPECIAL,
    # mbajme mberritjen me te hershme.
    unique = {}

    for s in children:
        remaining = tuple(sorted(
            set(suffix) - set(s["route"])
        ))

        key = (
            remaining,
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
            s["current_time"] + 9 * s["special_used"],
            s["current_time"]
        )
    )

    selected = []
    seen = set()
    ends = defaultdict(int)

    for used, quota in [
        (2, 12),
        (3, 14),
        (4, 30),
        (5, 18)
    ]:
        n = 0

        for s in ranked:
            if len(selected) >= width or n >= quota:
                break

            ident = uid(s)
            end = s["route"][-1]

            if (
                s["special_used"] != used
                or ident in seen
                or ends[end] >= 8
            ):
                continue

            selected.append(s)
            seen.add(ident)
            ends[end] += 1
            n += 1

    for s in ranked:
        if len(selected) >= width:
            break

        ident = uid(s)

        if ident not in seen:
            selected.append(s)
            seen.add(ident)

    return selected, len(unique)


# FAZA 1 - Zgjedhim 6 itinerare.
source_hashes = {}
seed_list = []
ids = set()

for name, used, quota in [
    ("A14", 4, 4),
    ("A14", 5, 1),
    ("B12", 5, 1)
]:
    path = RESULTS / (
        f"beam_v11_{name}_level_31.json"
    )

    raw = path.read_bytes()
    source_hashes[path.name] = hashlib.sha256(
        raw
    ).hexdigest()

    states = json.loads(raw)["frontier"]

    options = sorted(
        (
            s for s in states
            if s["special_used"] == used
        ),
        key=lambda s: s["current_time"]
    )

    count = 0

    for s in options:
        identity = uid(s)

        if identity in ids:
            continue

        assert len(s["route"]) == 31 and verify(s), (
            "Rruga fillestare eshte e pavlefshme."
        )

        ids.add(identity)
        seed_list.append((name, s))
        count += 1

        if count >= quota:
            break

    assert count == quota, (
        f"Nuk ka mjaft rruge {name} "
        f"me SPECIAL {used}/5."
    )

CONFIG = {
    "source_hashes": source_hashes,
    "seed_ids": [
        uid(s) for _, s in seed_list
    ],
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
        "Checkpoint ekzistues ka "
        "konfigurim tjeter."
    )
else:
    report = {
        "config": CONFIG,
        "seeds": {},
        "probes": {}
    }
    atomic(OUT, report)


def save():
    atomic(OUT, report)


# Per secilin itinerar mbajme 24 domatet e para.
for name, s in seed_list:
    sid = uid(s)

    if sid in report["seeds"]:
        continue

    prefix_size = len(s["route"]) - TAIL
    trs = s["transfers"][:prefix_size - 1]

    prefix = {
        "route": s["route"][:prefix_size],
        "transfers": trs,
        "current_time": (
            float(trs[-1]["start_time"])
            + float(trs[-1]["tof"])
            if trs else 0.0
        ),
        "special_used": sum(
            t["status"] == "SPECIAL"
            for t in trs
        )
    }

    assert verify(prefix)

    report["seeds"][sid] = {
        "name": name,
        "suffix": s["route"][prefix_size:],
        "level": prefix_size,
        "frontier": [prefix],
        "pending": None,
        "history": [],
        "complete": False,
        "stopped": False
    }

    save()


print(
    "FAZA 1 - RENDITJE TE NDRYSHME "
    "TE 7 DOMATEVE",
    flush=True
)
print("=" * 65, flush=True)

for entry in report["seeds"].values():
    print(
        f"{entry['name']} | "
        f"Fillimi {entry['level']}/49 | "
        f"7 domatet: {entry['suffix']}",
        flush=True
    )


# FAZA 2 - Kerkojme permutime te ndryshme.
while time.monotonic() - START < LIMIT:
    progress = False

    for sid, entry in report["seeds"].items():
        if time.monotonic() - START >= LIMIT:
            break

        if entry["complete"] or entry["stopped"]:
            continue

        frontier = entry["frontier"]

        if not frontier:
            entry["stopped"] = True
            save()
            continue

        pending = entry["pending"]

        if pending is None:
            pending = {
                "input_ids": [
                    uid(s) for s in frontier
                ],
                "done": {}
            }

            entry["pending"] = pending
            save()

        assert pending["input_ids"] == [
            uid(s) for s in frontier
        ]

        for s in frontier:
            if time.monotonic() - START >= LIMIT:
                break

            key = uid(s)

            if key in pending["done"]:
                continue

            remaining = [
                t for t in entry["suffix"]
                if t not in s["route"]
            ]

            opts, evaluations = moves(
                s, remaining, WAITS, TOFS,
                do_local=True
            )

            children = [
                child(s, move, st)
                for st, move in opts
            ]

            pending["done"][key] = {
                "children": children,
                "evaluations": evaluations
            }

            save()

        if len(pending["done"]) < len(frontier):
            print(
                f"{entry['name']}: "
                f"checkpoint i pjesshem "
                f"{len(pending['done'])}/"
                f"{len(frontier)}",
                flush=True
            )
            break

        all_children = [
            c
            for result in pending["done"].values()
            for c in result["children"]
        ]

        new, unique_count = prune(
            all_children,
            entry["suffix"],
            WIDTH
        )

        entry["pending"] = None

        if not new:
            entry["stopped"] = True
            entry["history"].append({
                "attempted_level": entry["level"] + 1,
                "routes": 0
            })

            save()

            print(
                f"{entry['name']}: "
                f"u bllokua te niveli "
                f"{entry['level']}",
                flush=True
            )
            continue

        entry["level"] += 1
        entry["frontier"] = new

        entry["history"].append({
            "level": entry["level"],
            "unique": unique_count,
            "routes": len(new)
        })

        if entry["level"] == 31:
            entry["frontier"] = [
                s for s in new
                if verify(s)
            ]
            entry["complete"] = True

        save()
        progress = True

        print(
            f"{entry['name']} | "
            f"Niveli {entry['level']}/49 | "
            f"Rruge {len(entry['frontier'])} | "
            f"SPECIAL 4/5: "
            f"{sum(s['special_used'] == 4 for s in entry['frontier'])}",
            flush=True
        )

    if not progress:
        break


# FAZA 3 - Itineraret e verifikuara me 31 domate.
level31 = {}

for entry in report["seeds"].values():
    if entry["complete"]:
        for s in entry["frontier"]:
            level31[uid(s)] = s

if level31:
    atomic(
        SNAP31,
        {
            "level": 31,
            "verified": True,
            "frontier": list(level31.values())
        }
    )


# FAZA 4 - Kerkimi i domates 32.
print(
    "\nFAZA 4 - KERKIMI I DOMATES SE 32-TE",
    flush=True
)
print("=" * 65, flush=True)

ranked = sorted(
    level31.values(),
    key=lambda s: (
        s["special_used"],
        s["current_time"]
    )
)

selected = []
ends = defaultdict(int)

for s in ranked:
    key = (
        s["route"][-1],
        s["special_used"]
    )

    if ends[key] >= 4:
        continue

    selected.append(s)
    ends[key] += 1

    if len(selected) >= MAX_PROBES:
        break

for i, s in enumerate(selected, 1):
    if time.monotonic() - START >= LIMIT:
        break

    key = uid(s)

    if key in report["probes"]:
        continue

    targets = [
        t for t in range(P.n_tomatoes)
        if t not in s["route"]
    ]

    opts, evaluations = moves(
        s,
        targets,
        PROBE_WAITS,
        PROBE_TOFS,
        do_local=True
    )

    children = [
        child(s, move, st)
        for st, move in opts
    ]

    verified = [
        c for c in children
        if verify(c)
    ]

    report["probes"][key] = {
        "evaluations": evaluations,
        "children": verified
    }

    save()

    print(
        f"PROBE [{i}/{len(selected)}] | "
        f"Fund {s['route'][-1]} | "
        f"Dita {s['current_time']:.2f} | "
        f"SPECIAL {s['special_used']}/5 | "
        f"Rruge32 {len(verified)}",
        flush=True
    )


found = {}

for probe in report["probes"].values():
    for s in probe["children"]:
        found[uid(s)] = s

if found:
    atomic(
        SNAP32,
        {
            "level": 32,
            "verified": True,
            "frontier": list(found.values())
        }
    )


print("\nPERMBLEDHJA E HAPIT 127")
print("=" * 65, flush=True)

for entry in report["seeds"].values():
    print(
        f"{entry['name']} | "
        f"Niveli {entry['level']}/31 | "
        f"Rruge {len(entry['frontier'])} | "
        f"Perfunduar {entry['complete']} | "
        f"Ndalur {entry['stopped']} | "
        f"Checkpoint {entry['pending'] is not None}",
        flush=True
    )

print(
    "Rruge te verifikuara nivel31:",
    len(level31),
    flush=True
)

print(
    "Kontrolle nivel32:",
    len(report["probes"]),
    "/",
    len(selected),
    flush=True
)

print(
    "Rruge te verifikuara nivel32:",
    len(found),
    flush=True
)

if found:
    print(
        "REKORD I RI I VERIFIKUAR: 32/49!",
        flush=True
    )
else:
    print(
        "Rekordi aktual mbetet 31/49.",
        flush=True
    )

if (
    any(
        not e["complete"] and not e["stopped"]
        for e in report["seeds"].values()
    )
    or any(
        uid(s) not in report["probes"]
        for s in selected
    )
):
    print(
        "Kerkimi eshte i pjesshem. "
        "Ekzekuto perseri te njejten qelize.",
        flush=True
    )

print("HAPI 127 PERFUNDOI!", flush=True)
