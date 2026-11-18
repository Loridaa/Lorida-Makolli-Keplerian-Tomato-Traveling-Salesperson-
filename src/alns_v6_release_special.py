"""ESA SpOC4 - ALNS V6: free a SPECIAL transfer, then attempt tomato 40.

This is a two-stage *experimental* neighborhood search from the independently
verified 39/49 record. Stage A seeks a verified alternative 39/49 route using
at most 4 SPECIAL transfers by retiming / permuting windows around existing
SPECIAL legs. Stage B attempts 40/49 insertion from those verified alternatives.

It never overwrites any record, existing candidate, source, or result.
It relies on the proven V4 full-route orbital verification and transfer DP.

Kaggle:
  /kaggle/working/esa-pykep-env/bin/python -u src/alns_v6_release_special.py --check-only
  /kaggle/working/esa-pykep-env/bin/python -u src/alns_v6_release_special.py
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
import alns_v4_dp_repair as v4

BASE38 = ROOT / "results/best_verified_38.json"
BACKUP38 = ROOT / "kaggle_backups/results/best_verified_38.json"
BASE39 = ROOT / "results/best_verified_39.json"
REPORT = ROOT / "results/alns_v6_release_special_report.json"
ALTERNATIVE39 = ROOT / "results/alns_v6_verified_39_alternative.json"
CANDIDATE40 = ROOT / "results/alns_v6_verified_40_candidate.json"

SEED = 17406
MAX_RELEASE_ATTEMPTS = 140
MAX_INSERT_ATTEMPTS = 260
MAX_EVALUATIONS = 160_000
MAX_ALTERNATIVES = 6
MIN_TOF = 0.5
RNG = random.Random(SEED)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_baseline():
    for path in (BASE38, BACKUP38, BASE39):
        if not path.is_file():
            raise RuntimeError("Mungon nje rekord: " + str(path))
    fingerprints = {p.name: sha(p) for p in (BASE38, BASE39)}
    if sha(BASE38) != sha(BACKUP38):
        raise RuntimeError("Kopjet e rekordit 38 nuk perputhen.")
    p = v4.load_problem()
    assert p.n_tomatoes == 49
    for path, level in ((BASE38, 38), (BASE39, 39)):
        data = json.loads(path.read_text(encoding="utf-8"))
        if (data.get("level") != level or data.get("verified") is not True
                or len(data.get("frontier", [])) != 1):
            raise RuntimeError("Formati i rekordit eshte i gabuar: " + path.name)
        state = data["frontier"][0]
        if (len(state.get("route", [])) != level
                or len(state.get("transfers", [])) != level - 1
                or not v4.verify(p, state)):
            raise RuntimeError("Rekordi nuk u verifikua: " + path.name)
        if level == 39:
            baseline = state
    if baseline["special_used"] != 5:
        raise RuntimeError("Ky version eshte ndertuar per bazen me 5 SPECIAL.")
    return p, baseline, fingerprints


def mutate(tomatoes, operator, pivot):
    """Permute inside a window but keep the exact same tomato set."""
    items = list(tomatoes)
    if len(items) < 2 or operator == "keep":
        return items
    j = min(max(0, pivot), len(items) - 2)
    if operator == "swap":
        items[j], items[j + 1] = items[j + 1], items[j]
    elif operator == "reverse":
        k = min(len(items), j + 3)
        items[j:k] = list(reversed(items[j:k]))
    elif operator == "rotate":
        k = min(len(items), j + 4)
        a = items[j:k]
        items[j:k] = a[1:] + a[:1]
    elif operator == "relocate":
        moved = items.pop(j)
        items.insert(min(len(items), j + 3), moved)
    else:
        raise ValueError("Operator i panjohur")
    return items


def stitch(route, oldlegs, i, end, replacement, repaired):
    prefix = oldlegs[:i - 1]
    suffix = oldlegs[end:] if end < len(route) - 1 else []
    newroute = route[:i] + replacement + route[end + 1:]
    newlegs = prefix + repaired["legs"] + suffix
    if not newlegs:
        raise RuntimeError("Rruge pa transferime")
    state = {
        "route": newroute,
        "transfers": newlegs,
        "special_used": sum(t["status"] == "SPECIAL" for t in newlegs),
        "current_time": float(newlegs[-1]["start_time"] + newlegs[-1]["tof"]),
    }
    return state


def frame(state, i, end):
    route, legs = state["route"], state["transfers"]
    if not (1 <= i <= len(route) and i - 1 <= end < len(route)):
        raise ValueError("Indeks i pavlefshem i dritares")
    prefix = legs[:i - 1]
    suffix = legs[end:] if end < len(route) - 1 else []
    start = float(prefix[-1]["start_time"] + prefix[-1]["tof"]) if prefix else 0.0
    deadline = float(suffix[0]["start_time"]) if suffix else 200.0
    window_legs = legs[i - 1:end]
    reserved = sum(t["status"] == "SPECIAL" for t in prefix + suffix)
    specials = sum(t["status"] == "SPECIAL" for t in window_legs)
    return start, deadline, window_legs, reserved, specials


def build_release_plans(baseline):
    route, legs = baseline["route"], baseline["transfers"]
    n = len(route)
    special_positions = [i + 1 for i, t in enumerate(legs) if t["status"] == "SPECIAL"]
    plans = {}
    for anchor in special_positions:
        for before, after in ((0, 1), (1, 2), (2, 3), (3, 5),
                              (5, 7), (8, 10), (12, 12), (18, 18)):
            i = max(1, anchor - before)
            end = min(n - 1, anchor + after)
            start, deadline, oldwindow, _, specials = frame(baseline, i, end)
            if specials == 0 or deadline - start < MIN_TOF * len(oldwindow) - 1e-5:
                continue
            for op in ("keep", "swap", "reverse", "rotate", "relocate"):
                for pivot in (0, max(0, (end - i) // 2), max(0, end - i - 1)):
                    change = mutate(route[i:end + 1], op, pivot)
                    key = (i, end, tuple(change))
                    if key in plans:
                        continue
                    # Favor windows near a SPECIAL with enough free time,
                    # but cover both small and larger neighborhoods.
                    slack = deadline - start - MIN_TOF * len(oldwindow)
                    score = (24.0 * specials + 4.0 * min(8.0, slack)
                             - 1.4 * (end - i + 1)
                             + (4.0 if op == "keep" else 0.0)
                             + (3.0 if end == n - 1 else 0.0))
                    plans[key] = (score, i, end, change, op, anchor)
    result = list(plans.values())
    RNG.shuffle(result)
    result.sort(key=lambda x: -x[0])
    # First pass covers distinct anchors rather than 140 similar tests.
    selected, done = [], set()
    for _ in range(4):
        for item in result:
            marker = (item[5], item[1], item[2], tuple(item[3]))
            if marker in done:
                continue
            if item[5] in {x[5] for x in selected[-len(special_positions):]}:
                continue
            selected.append(item)
            done.add(marker)
            if len(selected) >= MAX_RELEASE_ATTEMPTS:
                return selected
    for item in result:
        marker = (item[5], item[1], item[2], tuple(item[3]))
        if marker not in done:
            selected.append(item)
            done.add(marker)
        if len(selected) >= MAX_RELEASE_ATTEMPTS:
            break
    return selected


def release_specials(p, base, evaluator):
    route, oldlegs = base["route"], base["transfers"]
    plans = build_release_plans(base)
    failures = Counter()
    near = {"legs_repaired": 0, "legs_needed": 0}
    variants, seen = [], set()
    attempts = 0
    for _, i, end, targets, operator, anchor in plans:
        start, deadline, oldwin, reserved, specials = frame(base, i, end)
        # Enforce a genuine saving of at least one SPECIAL in this window.
        cap = min(p.n_exceptions - reserved, specials - 1)
        if cap < 0:
            continue
        attempts += 1
        repaired, reached, reason, edge = v4.repair(
            p, evaluator, route[i - 1], targets, start, deadline, cap, oldwin
        )
        if reached > near["legs_repaired"]:
            near = {"attempt": attempts, "cut": i, "end": end,
                    "operator": operator, "anchor": anchor,
                    "legs_repaired": reached, "legs_needed": len(targets),
                    "failed_edge": edge}
        if repaired is None:
            failures[reason or "unknown"] += 1
            continue
        candidate = stitch(route, oldlegs, i, end, targets, repaired)
        if (len(candidate["route"]) == 39
                and candidate["special_used"] < base["special_used"]
                and v4.verify(p, candidate)):
            key = (tuple(candidate["route"]), candidate["special_used"],
                   round(candidate["current_time"], 4))
            if key not in seen:
                seen.add(key)
                variants.append((candidate, {"cut": i, "end": end,
                                             "operator": operator,
                                             "anchor": anchor}))
                print("Rruge alternative 39/49: SPECIAL",
                      candidate["special_used"], "/5 | dita",
                      round(candidate["current_time"], 4), flush=True)
                if len(variants) >= MAX_ALTERNATIVES:
                    break
        else:
            failures["full_verification_or_no_special_saving"] += 1
        if attempts % 12 == 0:
            print("Lirimi SPECIAL:", attempts, "/", len(plans),
                  "| variante", len(variants), "| eval", evaluator.count,
                  flush=True)
    variants.sort(key=lambda x: (x[0]["special_used"], x[0]["current_time"]))
    return variants, attempts, failures, near


def insertion_plans(variant, context, unused):
    route, oldlegs = variant["route"], variant["transfers"]
    n = len(route)
    anchor = int(context["cut"])
    candidates = []
    for i in range(1, n + 1):
        for width in (1, 3, 6, 10, 16):
            end = min(n - 1, i + width - 1)
            if i == n and width != 1:
                continue
            start, deadline, oldwin, _, special = frame(variant, i, end)
            length = len(oldwin) + 1
            slack = deadline - start - MIN_TOF * length
            if slack < -1e-5:
                continue
            targets = list(route[i:end + 1])
            offsets = {0, len(targets) // 2, len(targets)}
            distance = min(abs(i - anchor), abs(end - context["end"]))
            for tomato in unused:
                for offset in offsets:
                    score = (10.0 * max(0, 6 - distance)
                             + 3.0 * min(10.0, max(0.0, slack))
                             + 3.0 * special - .6 * len(targets))
                    candidates.append((score, i, end, tomato, offset))
    RNG.shuffle(candidates)
    candidates.sort(key=lambda x: -x[0])
    # Diverse missing tomatoes: avoid using all budget on one target.
    chosen, seen = [], set()
    for pass_num in range(5):
        used_tomatoes = set()
        for item in candidates:
            _, i, end, tomato, offset = item
            key = (i, end, tomato, offset)
            if key in seen or tomato in used_tomatoes:
                continue
            chosen.append(item)
            seen.add(key)
            used_tomatoes.add(tomato)
            if len(chosen) >= 90:
                return chosen
    return chosen[:90]


def insert_tomato(p, alternatives, evaluator, original_route):
    unused = sorted(set(range(p.n_tomatoes)) - set(original_route))
    failures = Counter()
    attempts = 0
    full_repairs = 0
    partial = {"legs_repaired": 0, "legs_needed": 0}
    for state, context in alternatives:
        route, oldlegs = state["route"], state["transfers"]
        for _, i, end, tomato, offset in insertion_plans(state, context, unused):
            if attempts >= MAX_INSERT_ATTEMPTS:
                return None, None, attempts, full_repairs, failures, partial
            start, deadline, oldwin, reserved, _ = frame(state, i, end)
            cap = p.n_exceptions - reserved
            if cap < 0:
                continue
            targets = list(route[i:end + 1])
            targets.insert(offset, tomato)
            attempts += 1
            repaired, reached, reason, edge = v4.repair(
                p, evaluator, route[i - 1], targets, start, deadline, cap, oldwin
            )
            if reached > partial["legs_repaired"]:
                partial = {"attempt": attempts, "cut": i, "end": end,
                           "missing_tomato": tomato, "legs_repaired": reached,
                           "legs_needed": len(targets), "failed_edge": edge}
            if repaired is None:
                failures[reason or "unknown"] += 1
                continue
            full_repairs += 1
            proposal = stitch(route, oldlegs, i, end, targets, repaired)
            if (len(proposal["route"]) == 40
                    and len(set(proposal["route"])) == 40
                    and v4.verify(p, proposal)):
                print("*** KANDIDAT 40/49 I VERIFIKUAR!",
                      "SPECIAL", proposal["special_used"],
                      "dita", round(proposal["current_time"], 5), flush=True)
                return proposal, {"source_alt_specials": state["special_used"],
                                  "inserted_tomato": tomato, "cut": i,
                                  "end": end, "offset": offset}, attempts, full_repairs, failures, partial
            else:
                failures["full_verification"] += 1
            if attempts % 20 == 0:
                print("Futja e 40-tes:", attempts, "/", MAX_INSERT_ATTEMPTS,
                      "| full", full_repairs,
                      "| eval", evaluator.count, flush=True)
    return None, None, attempts, full_repairs, failures, partial


def write_exclusive(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("U ruajt:", path.name, flush=True)


def main():
    p, base, fingerprints = read_baseline()
    print("BAZA: 39/49 | dita", round(base["current_time"], 6),
          "| SPECIAL", base["special_used"], "/5", flush=True)
    print("Domatet qe mungojne:",
          sorted(set(range(p.n_tomatoes)) - set(base["route"])), flush=True)
    if "--check-only" in sys.argv:
        print("CHECK-ONLY OK - asnje skedar nuk u ndryshua.", flush=True)
        return
    if REPORT.exists() or ALTERNATIVE39.exists() or CANDIDATE40.exists():
        raise RuntimeError("V6 ekziston - nuk mbishkruajme asnje rezultat.")
    v4.MAX_EVALUATIONS = MAX_EVALUATIONS
    evaluator = v4.Evaluator(p)
    t0 = time.monotonic()
    hit_limit = False
    alternatives = []
    release_attempts = insertion_attempts = 0
    full_repairs = 0
    release_failures = Counter()
    insert_failures = Counter()
    release_near = {"legs_repaired": 0, "legs_needed": 0}
    insert_near = {"legs_repaired": 0, "legs_needed": 0}
    found40 = context40 = None
    try:
        alternatives, release_attempts, release_failures, release_near = release_specials(
            p, base, evaluator
        )
        if alternatives:
            print("U lirua buxheti SPECIAL. Tash provojme 40/49.", flush=True)
            (found40, context40, insertion_attempts, full_repairs,
             insert_failures, insert_near) = insert_tomato(
                p, alternatives, evaluator, base["route"]
            )
        else:
            print("Nuk u gjet rruge 39/49 me me pak SPECIAL. ",
                  "Nuk po perserisim V5.", flush=True)
    except v4.BudgetReached:
        hit_limit = True
        print("Kufiri orbital u arrit:", MAX_EVALUATIONS, flush=True)

    if (sha(BASE38) != fingerprints[BASE38.name]
            or sha(BACKUP38) != fingerprints[BASE38.name]
            or sha(BASE39) != fingerprints[BASE39.name]):
        raise RuntimeError("NDAL: rekordet baze kane ndryshuar!")
    if alternatives:
        best, move = alternatives[0]
        if not v4.verify(p, best) or best["special_used"] >= 5:
            raise RuntimeError("Rruga 39 alternative nuk u verifikua!")
        write_exclusive(ALTERNATIVE39, {
            "level": 39, "verified": True,
            "source": "alns_v6_release_special.py", "frontier": [best]
        })
    if found40 is not None:
        if not v4.verify(p, found40):
            raise RuntimeError("Kandidati 40 nuk u riverifikua!")
        write_exclusive(CANDIDATE40, {
            "level": 40, "verified": True,
            "source": "alns_v6_release_special.py", "frontier": [found40]
        })

    report = {
        "algorithm": "ALNS V6: verified SPECIAL-release and targeted insertion",
        "base_level": 39,
        "best_level": 40 if found40 is not None else 39,
        "candidate_verified": found40 is not None,
        "alternative_39_verified": bool(alternatives),
        "best_39_special_used": alternatives[0][0]["special_used"] if alternatives else 5,
        "release_attempts": release_attempts,
        "release_variants": len(alternatives),
        "insertion_attempts": insertion_attempts,
        "full_temporal_repairs": full_repairs,
        "transfer_evaluations": evaluator.count,
        "evaluation_limit": MAX_EVALUATIONS,
        "evaluation_limit_reached": hit_limit,
        "release_failure_reasons": dict(release_failures),
        "insert_failure_reasons": dict(insert_failures),
        "best_release_partial": release_near,
        "best_insert_partial": insert_near,
        "candidate_context": context40,
        "protected_38_sha256": fingerprints[BASE38.name],
        "protected_39_sha256": fingerprints[BASE39.name],
        "seed": SEED,
        "elapsed_seconds": round(time.monotonic() - t0, 2),
    }
    write_exclusive(REPORT, report)
    print("=" * 55, flush=True)
    print("V6:", report["best_level"], "/49",
          "| alternative SPECIAL:", report["best_39_special_used"],
          "| vleresime:", evaluator.count, flush=True)
    print("Rekordet 38 dhe 39: TE PAPREKURA", flush=True)


if __name__ == "__main__":
    main()
