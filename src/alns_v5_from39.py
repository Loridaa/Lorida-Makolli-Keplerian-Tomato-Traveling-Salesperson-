"""ESA SpOC4 – V5: targeted large-neighborhood repair from verified 39/49.

Runs with Kaggle's separate PyKEP interpreter; imports the proven V4 orbital
validator and dynamic-programming repair implementation without modifying V4.

This is an experimental search, NOT a guarantee of 40/49. Never overwrites
best_verified_38.json, best_verified_39.json, or existing experiment results.

Commands:
  /kaggle/working/esa-pykep-env/bin/python -u src/alns_v5_from39.py --check-only
  /kaggle/working/esa-pykep-env/bin/python -u src/alns_v5_from39.py
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import alns_v4_dp_repair as v4  # previous, preserved algorithm

BASE38 = ROOT / "results/best_verified_38.json"
BACKUP38 = ROOT / "kaggle_backups/results/best_verified_38.json"
BASE39 = ROOT / "results/best_verified_39.json"
REPORT = ROOT / "results/alns_v5_from39_report.json"
CANDIDATE = ROOT / "results/alns_v5_verified_40_candidate.json"

SEED = 17340
MAX_ATTEMPTS = 210
MAX_EVALUATIONS = 160_000
MIN_TOF = 0.5
rng = random.Random(SEED)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_baseline():
    for path in (BASE38, BACKUP38, BASE39):
        if not path.is_file():
            raise RuntimeError("Mungon nje rekord i nevojshem: " + path.name)
    if sha(BASE38) != sha(BACKUP38):
        raise RuntimeError("NDAL: rekordi 38 nuk perputhet me backup-in.")
    p = v4.load_problem()
    d38 = json.loads(BASE38.read_text(encoding="utf-8"))
    d39 = json.loads(BASE39.read_text(encoding="utf-8"))
    for data, n in ((d38, 38), (d39, 39)):
        if (data.get("level") != n or data.get("verified") is not True
                or len(data.get("frontier", [])) != 1):
            raise RuntimeError(f"Formati i rekordit {n} eshte i pasakte.")
        s = data["frontier"][0]
        if (len(s["route"]) != n or len(s["transfers"]) != n-1
                or not v4.verify(p, s)):
            raise RuntimeError(f"Rekordi {n} nuk u verifikua me PyKEP.")
    old = d38["frontier"][0]
    new = d39["frontier"][0]
    if (not set(old["route"]).issubset(set(new["route"]))
            or set(new["route"]) - set(old["route"]) != {26}):
        raise RuntimeError("Rekordi 39 nuk permban shtimin e pritur te domates 26.")
    return p, new


def specs_for_route(p, route, legs):
    """Diverse windows, including the newly inserted tomato 26 and SPECIALs."""
    n = len(route)
    new_index = route.index(26)
    result = []
    for i in range(1, n+1):  # also consider append at the very end
        for width in (1, 2, 3, 5, 8, 12, 19, 26):
            end = min(n-1, i+width-1)
            if i == n and width != 1:
                continue
            prefix = legs[:i-1]
            suffix = legs[end:] if end < n-1 else []
            start = float(prefix[-1]["start_time"] + prefix[-1]["tof"]) if prefix else 0.0
            deadline = float(suffix[0]["start_time"]) if suffix else 200.0
            count = max(0, end - i + 1) + 1
            slack = deadline - start - count*MIN_TOF
            if slack < -1e-5:
                continue
            prior = legs[i-1:end]
            specials = sum(t["status"] == "SPECIAL" for t in prior)
            around_26 = i <= new_index <= end or abs(i-new_index) <= 2
            score = (8.0*min(3, specials)
                     + 3.5*min(8.0, slack / max(1, count))
                     + (10.0 if around_26 else 0.0)
                     - 1.5 * count)
            result.append({"i": i, "end": end, "start": start,
                           "deadline": deadline, "specials": specials,
                           "score": score, "slack": slack})
    return result


def make_plan(specs, route, unused):
    """Stratify by insertion site; mix missing tomatoes, window and edits."""
    by_cut = {}
    for s in specs:
        by_cut.setdefault(s["i"], []).append(s)
    for group in by_cut.values():
        rng.shuffle(group)
        group.sort(key=lambda s: -s["score"])
    indices = list(by_cut)
    insertion_near_26 = route.index(26)
    # Favor near the new tomato, but cover all remaining sites.
    indices.sort(key=lambda i: (abs(i-insertion_near_26), i))
    plans = []
    seen = set()
    ops = ("keep", "keep", "swap", "reverse3", "rotate3")
    for round_num in range(9):
        rotated = indices[round_num % len(indices):] + indices[:round_num % len(indices)]
        for i in rotated:
            group = by_cut[i]
            s = group[min(round_num % len(group), len(group)-1)]
            size = max(0, s["end"] - i + 1)
            tomato = unused[(i*7 + round_num*3 + s["end"]) % len(unused)]
            offsets = (0, size//2, size)
            offset = offsets[round_num % len(offsets)]
            op = ops[(round_num + i//3) % len(ops)]
            if size < 2:
                op = "keep"
            key = (i, s["end"], tomato, offset, op)
            if key not in seen:
                seen.add(key)
                plans.append((s, tomato, offset, op))
            if len(plans) >= MAX_ATTEMPTS:
                return plans
    return plans


def mutated_targets(route, i, end, tomato, offset, operator):
    targets = list(route[i:end+1]) if i <= end else []
    offset = max(0, min(len(targets), offset))
    if operator != "keep" and len(targets) >= 2:
        j = min(max(0, offset), len(targets)-2)
        if operator == "swap":
            targets[j], targets[j+1] = targets[j+1], targets[j]
        elif operator == "reverse3":
            k = min(len(targets), j+3)
            targets[j:k] = list(reversed(targets[j:k]))
        elif operator == "rotate3":
            k = min(len(targets), j+3)
            sub = targets[j:k]
            targets[j:k] = sub[1:] + sub[:1]
    targets.insert(offset, tomato)
    return targets


def write_exclusive(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("U ruajt:", path.name, flush=True)


def main():
    fingerprint38 = sha(BASE38) if BASE38.is_file() else None
    fingerprint39 = sha(BASE39) if BASE39.is_file() else None
    p, base = checked_baseline()
    route, oldlegs = base["route"], base["transfers"]
    unused = sorted(set(range(p.n_tomatoes)) - set(route))
    if len(unused) != 10:
        raise RuntimeError("Pritej te mungonin 10 domate.")
    print("BAZA OK: 39/49, dita", round(base["current_time"], 6),
          "SPECIAL", base["special_used"], flush=True)
    print("Domatet qe mungojne:", unused, flush=True)
    if "--check-only" in sys.argv:
        print("CHECK-ONLY OK. Asnje skedar nuk u ndryshua.", flush=True)
        return
    if REPORT.exists() or CANDIDATE.exists():
        raise RuntimeError("Ekziston raport ose kandidat V5; nuk mbishkruhet.")

    specs = specs_for_route(p, route, oldlegs)
    plans = make_plan(specs, route, unused)
    if not plans:
        raise RuntimeError("Nuk u gjeneruan plane te vlefshme.")
    # Reuse verified V4 evaluator & time/SPECIAL DP, with new search budget.
    v4.MAX_EVALUATIONS = MAX_EVALUATIONS
    evaluator = v4.Evaluator(p)
    failures = Counter()
    operators = Counter()
    full_repairs = 0
    tried = 0
    closest = {"legs_repaired": 0, "legs_needed": 0}
    found = None
    found_context = None
    wall = time.monotonic()
    hit_limit = False

    try:
        for spec, tomato, offset, operator in plans:
            i, end = spec["i"], spec["end"]
            prefix = oldlegs[:i-1]
            suffix = oldlegs[end:] if end < len(route)-1 else []
            reserved = sum(t["status"] == "SPECIAL" for t in prefix + suffix)
            cap = p.n_exceptions - reserved
            if cap < 0:
                failures["reserved_specials_over_budget"] += 1
                continue
            targets = mutated_targets(route, i, end, tomato, offset, operator)
            left = route[i-1]
            original_window = oldlegs[i-1:end]
            tried += 1
            operators[operator] += 1
            repaired, reached, reason, edge = v4.repair(
                p, evaluator, left, targets, spec["start"],
                spec["deadline"], cap, original_window)
            if (reached > closest["legs_repaired"]
                    or (reached == closest["legs_repaired"]
                        and len(targets) < closest["legs_needed"])):
                closest = {"attempt": tried, "cut": i, "end_index": end,
                           "missing_tomato": tomato, "operator": operator,
                           "legs_repaired": reached, "legs_needed": len(targets),
                           "failed_edge": edge}
            if repaired is None:
                failures[reason or "unknown"] += 1
            else:
                full_repairs += 1
                proposal = {
                    "route": route[:i] + targets + route[end+1:],
                    "transfers": prefix + repaired["legs"] + suffix,
                }
                proposal["special_used"] = sum(
                    t["status"] == "SPECIAL" for t in proposal["transfers"])
                endleg = proposal["transfers"][-1]
                proposal["current_time"] = float(endleg["start_time"] + endleg["tof"])
                if (len(proposal["route"]) == 40
                        and len(proposal["transfers"]) == 39
                        and v4.verify(p, proposal)):
                    found = proposal
                    found_context = {"attempt": tried, "cut": i,
                                     "end_index": end, "tomato": tomato,
                                     "operator": operator, "offset": offset}
                    print("*** KANDIDAT 40/49 I VERIFIKUAR!",
                          "dita", round(proposal["current_time"], 6), flush=True)
                    break
                failures["full_route_verification"] += 1
            if tried % 10 == 0:
                print(f"Prova {tried}/{len(plans)} | eval {evaluator.count} "
                      f"| riparime te plota {full_repairs} | "
                      f"progres maks {closest['legs_repaired']}/{closest['legs_needed']}",
                      flush=True)
    except v4.BudgetReached:
        hit_limit = True
        print("U arrit kufiri i vleresimeve:", MAX_EVALUATIONS, flush=True)

    if (sha(BASE38) != fingerprint38 or sha(BACKUP38) != fingerprint38
            or sha(BASE39) != fingerprint39):
        raise RuntimeError("NDAL: nje nga rekordet baze ka ndryshuar.")

    if found is not None:
        write_exclusive(CANDIDATE, {
            "level": 40, "verified": True, "source": "alns_v5_from39.py",
            "frontier": [found],
        })
    report = {
        "algorithm": "V5: diverse targeted insertion/mutation with V4 time/SPECIAL DP",
        "base_level": 39,
        "best_level": 40 if found is not None else 39,
        "candidate_verified": found is not None,
        "instance": "easy.kttsp (49 tomatoes)",
        "seed": SEED,
        "attempts": tried,
        "max_attempts": MAX_ATTEMPTS,
        "full_temporal_repairs": full_repairs,
        "transfer_evaluations": evaluator.count,
        "evaluation_limit": MAX_EVALUATIONS,
        "evaluation_limit_reached": hit_limit,
        "best_partial_repair": closest,
        "failure_reasons": dict(failures),
        "operators_tried": dict(operators),
        "candidate_context": found_context,
        "elapsed_seconds": round(time.monotonic() - wall, 2),
        "protected_38_sha256": fingerprint38,
        "protected_39_sha256": fingerprint39,
    }
    write_exclusive(REPORT, report)
    print("="*60, flush=True)
    print("V5:", report["best_level"], "/49 | provat", tried,
          "| vleresime", evaluator.count,
          "| riparime te plota", full_repairs, flush=True)
    print("Rekordet 38 dhe 39: TE PAPREKURA", flush=True)


if __name__ == "__main__":
    main()
