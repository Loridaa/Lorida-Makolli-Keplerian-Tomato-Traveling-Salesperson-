"""ESA SpOC4 -- ALNS V9 multi-start route reconstruction (39 -> 40).

An independent, bounded experiment. It protects the official verified 38/49 and
39/49 records, V7's 39-neutral candidate and earlier algorithms/results.

Stage 1: diversify *verified* 39-tomato routes by neutral route/temporal edits
around difficult orbital edges and SPECIAL transitions (not restricted to 4
SPECIAL). Stage 2: try one-tomato insertions from multiple starting routes,
using longer retimed windows and feedback-guided operator selection. Every
candidate is independently checked with V4's PyKEP transfer validator.

Run ONLY in the Kaggle PyKEP environment, from the Git checkout:
  /kaggle/working/esa-pykep-env/bin/python -u src/alns_v9_multistart.py --check-only
  /kaggle/working/esa-pykep-env/bin/python -u src/alns_v9_multistart.py
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import alns_v4_dp_repair as v4  # noqa: E402

BASE38 = ROOT / "results/best_verified_38.json"
BACKUP38 = ROOT / "kaggle_backups/results/best_verified_38.json"
BASE39 = ROOT / "results/best_verified_39.json"
ALT39 = ROOT / "results/alns_v7_verified_39_neutral.json"
V8_REPORT = ROOT / "results/alns_v8_edge_escape_report.json"
REPORT = ROOT / "results/alns_v9_multistart_report.json"
ARCHIVE = ROOT / "results/alns_v9_multistart_verified_39_archive.json"
CAND40 = ROOT / "results/alns_v9_verified_40_candidate.json"

SEED = 17809
RNG = random.Random(SEED)
MAX_NEUTRAL_ATTEMPTS = 80
MAX_INSERT_ATTEMPTS = 420
MAX_NEW_NEUTRAL = 12
MAX_EVALUATIONS = 165_000
MAX_RUNTIME_SECONDS = 900
MIN_TOF = 0.5
EPS = 1e-6
OPERATORS = ("direct", "swap_left", "swap_right", "reverse3", "relocate", "rotate4")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_paths():
    paths = {"38": BASE38, "backup38": BACKUP38,
             "39": BASE39, "v7_alt39": ALT39}
    # V8's report is optional for reusing the code from an older checkout.
    if V8_REPORT.is_file():
        paths["v8_report"] = V8_REPORT
    for label, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"Mungon burimi {label}: {path}")
    fp = {label: sha(path) for label, path in paths.items()}
    if fp["38"] != fp["backup38"]:
        raise RuntimeError("NDAL: 38/49 nuk perputhet me backup-in!")
    return paths, fp


def read_verified(path: Path, level: int, problem):
    obj = json.loads(path.read_text(encoding="utf-8"))
    frontier = obj.get("frontier")
    if (obj.get("level") != level or obj.get("verified") is not True
            or not isinstance(frontier, list) or len(frontier) != 1):
        raise RuntimeError(f"Format i pavlefshem: {path.name}")
    state = frontier[0]
    if (len(state.get("route", [])) != level
            or len(state.get("transfers", [])) != level-1
            or not v4.verify(problem, state)):
        raise RuntimeError(f"Verifikimi orbital deshtoi: {path.name}")
    return state


def load_inputs():
    paths, fp = protected_paths()
    p = v4.load_problem()
    if p.n_tomatoes != 49:
        raise RuntimeError("Kerkohen 49 domate nga instanca easy.kttsp")
    old = read_verified(BASE38, 38, p)
    base = read_verified(BASE39, 39, p)
    alt = read_verified(ALT39, 39, p)
    if not set(old["route"]).issubset(set(base["route"])):
        raise RuntimeError("Baza 39 nuk perfshin te gjitha 38 domatet")
    if set(alt["route"]) != set(base["route"]):
        raise RuntimeError("Alternativa V7 nuk permban te njejtat 39 domate")
    missing = sorted(set(range(49)) - set(base["route"]))
    if len(missing) != 10:
        raise RuntimeError("Duhet te kene mbetur 10 domate")
    seeds = [("record39", base)]
    if fingerprint(alt) != fingerprint(base):
        seeds.append(("v7_alternative", alt))
    problematic = Counter({(20, 37): 13, (20, 27): 10,
                           (37, 27): 9, (37, 17): 9,
                           (23, 38): 5})
    if V8_REPORT.is_file():
        obj = json.loads(V8_REPORT.read_text(encoding="utf-8"))
        for edge, hits in obj.get("most_failed_edges", []):
            try:
                a, b = (int(x) for x in edge.split("->"))
                problematic[(a, b)] = max(problematic[(a, b)], int(hits))
            except (ValueError, TypeError):
                continue
    return p, seeds, missing, problematic, paths, fp


def fingerprint(state):
    return (tuple(state["route"]), tuple(
        (round(float(t["start_time"]), 4), round(float(t["tof"]), 4), t["status"])
        for t in state["transfers"]))


def edge_pairs(left, targets):
    seq = [left] + list(targets)
    return list(zip(seq[:-1], seq[1:]))


def window(state, i, end):
    route, legs = state["route"], state["transfers"]
    n = len(route)
    if not (1 <= i <= n and i - 1 <= end < n):
        raise ValueError("Dritarja e riparimit eshte jashte kufijve")
    prefix = legs[:i-1]
    suffix = legs[end:] if end < n-1 else []
    start = (float(prefix[-1]["start_time"] + prefix[-1]["tof"])
             if prefix else 0.0)
    deadline = float(suffix[0]["start_time"]) if suffix else 200.0
    oldlegs = legs[i-1:end]
    reserved = sum(leg["status"] == "SPECIAL" for leg in prefix + suffix)
    return start, deadline, oldlegs, reserved, prefix, suffix


def stitch(state, i, end, newtargets, repaired, prefix, suffix):
    new_route = list(state["route"][:i]) + list(newtargets) + list(state["route"][end+1:])
    new_legs = list(prefix) + list(repaired["legs"]) + list(suffix)
    last = new_legs[-1]
    return {"route": new_route, "transfers": new_legs,
            "special_used": sum(t["status"] == "SPECIAL" for t in new_legs),
            "current_time": float(last["start_time"] + last["tof"])}


def edit(nodes, op):
    seq = list(nodes)
    n = len(seq)
    if op == "direct":
        return seq
    if n < 2:
        return None
    mid = n // 2
    if op == "swap_left":
        j = max(0, mid-1)
        seq[j], seq[j+1] = seq[j+1], seq[j]
    elif op == "swap_right":
        j = min(mid, n-2)
        seq[j], seq[j+1] = seq[j+1], seq[j]
    elif op == "reverse3":
        j = min(max(mid-1, 0), max(0, n-3))
        seq[j:j+3] = list(reversed(seq[j:j+3]))
    elif op == "relocate":
        j = max(0, mid-2)
        moved = seq.pop(j)
        seq.insert(min(j+3, len(seq)), moved)
    elif op == "rotate4":
        j = min(max(mid-2, 0), max(0, n-4))
        piece = seq[j:j+4]
        seq[j:j+4] = piece[1:] + piece[:1]
    else:
        return None
    return seq


def insert_edit(nodes, tomato, offset, op):
    seq = list(nodes)
    seq.insert(offset, tomato)
    if op == "direct":
        return seq
    at = seq.index(tomato)
    if op == "swap_left":
        if at == 0: return None
        seq[at], seq[at-1] = seq[at-1], seq[at]
    elif op == "swap_right":
        if at == len(seq)-1: return None
        seq[at], seq[at+1] = seq[at+1], seq[at]
    elif op == "reverse3":
        if len(seq) < 3: return None
        lo = min(max(at-1, 0), len(seq)-3)
        seq[lo:lo+3] = reversed(seq[lo:lo+3])
    elif op == "relocate":
        item = seq.pop(at)
        seq.insert(min(len(seq), at+2), item)
    elif op == "rotate4":
        if len(seq) < 4: return None
        lo = min(max(at-2, 0), len(seq)-4)
        part = seq[lo:lo+4]
        seq[lo:lo+4] = part[-1:] + part[:-1]
    else:
        return None
    return seq


def window_score(state, i, end, inserted, troublesome):
    start, deadline, old, _, _, _ = window(state, i, end)
    cnt = len(old) + inserted
    if cnt < 1:
        return None
    slack = deadline - start - MIN_TOF * cnt
    if slack < -EPS:
        return None
    hits = sum(min(3, troublesome[(a, b)]) for a, b in
               edge_pairs(state["route"][i-1], state["route"][i:end+1]))
    special = sum(t["status"] == "SPECIAL" for t in old)
    return 4.2*min(3, special) + 1.5*hits + 1.2*min(10, slack/max(1, cnt)) - .75*cnt


def neutral_plans(state, troublesome):
    """Select diverse 39->39 changes; avoid V7's identical central-only attempts."""
    n = len(state["route"])
    collected = []
    seen = set()
    for i in range(1, n-1):
        for width in (2, 3, 4, 6, 9, 14, 22):
            end = min(n-1, i + width-1)
            if end == i:
                continue
            score = window_score(state, i, end, 0, troublesome)
            if score is None:
                continue
            for op in ("swap_left", "swap_right", "reverse3", "relocate", "rotate4"):
                target = edit(state["route"][i:end+1], op)
                if target is None or target == state["route"][i:end+1]:
                    continue
                key = (i, end, tuple(target))
                if key in seen: continue
                seen.add(key)
                collected.append((score + RNG.random()*1.4, i, end, target, op))
    # Balanced selection across cut positions; not all trials at one SPECIAL.
    bycut = defaultdict(list)
    for plan in collected: bycut[plan[1]].append(plan)
    for group in bycut.values(): group.sort(reverse=True, key=lambda x: x[0])
    cuts = sorted(bycut, key=lambda i: (-bycut[i][0][0], i))
    ordered = []
    for rank in range(8):
        for cut in cuts:
            if rank < len(bycut[cut]): ordered.append(bycut[cut][rank])
    return ordered


def make_archive(p, seeds, evaluator, troublesome, started):
    pool = [(name, state) for name, state in seeds]
    seen = {fingerprint(s) for _, s in pool}
    attempts = 0
    full = 0
    failures = Counter()
    variants = []
    # Include neighborhoods around two independently verified 39 states.
    plans = []
    for seed_idx, (name, state) in enumerate(seeds):
        for plan in neutral_plans(state, troublesome):
            plans.append((seed_idx, plan))
    # Round-robin by seed, retain priority within each seed.
    grouped = defaultdict(list)
    for item in plans: grouped[item[0]].append(item)
    for group in grouped.values(): group.sort(key=lambda v: -v[1][0])
    ordered = []
    for rank in range(max((len(group) for group in grouped.values()), default=0)):
        for idx in range(len(seeds)):
            if rank < len(grouped[idx]): ordered.append(grouped[idx][rank])
            if len(ordered) >= MAX_NEUTRAL_ATTEMPTS: break
        if len(ordered) >= MAX_NEUTRAL_ATTEMPTS: break
    for seed_idx, (_, i, end, targets, op) in ordered:
        if time.monotonic() - started >= MAX_RUNTIME_SECONDS:
            break
        name, base = seeds[seed_idx]
        start, deadline, old, reserved, prefix, suffix = window(base, i, end)
        cap = p.n_exceptions - reserved
        if cap < 0: continue
        attempts += 1
        state, reached, why, edge = v4.repair(
            p, evaluator, base["route"][i-1], targets, start, deadline, cap, old)
        if attempts % 20 == 0:
            print(f"Faza 39: {attempts} prova | {evaluator.count} eval",
                  flush=True)
        if state is None:
            failures[why or "unknown"] += 1
            continue
        full += 1
        proposal = stitch(base, i, end, targets, state, prefix, suffix)
        if (len(proposal["route"]) != 39 or
                set(proposal["route"]) != set(base["route"]) or
                not v4.verify(p, proposal)):
            failures["failed_full_verification"] += 1
            continue
        fp = fingerprint(proposal)
        if fp in seen:
            failures["duplicate"] += 1
            continue
        seen.add(fp)
        variants.append((f"v9_neutral_{len(variants)+1}", proposal))
        print(f"Alternativa V9 {len(variants)}: SPECIAL {proposal['special_used']} "
              f"dita {proposal['current_time']:.3f} | burimi {name}", flush=True)
        if len(variants) >= MAX_NEW_NEUTRAL:
            break
    pool.extend(variants)
    return pool, variants, attempts, full, failures


def insertion_plans(source_name, state, missing, troublesome, source_idx):
    """Bounded & cut-diverse plan generator: 3 top plans/tomato/cut.

    This avoids materializing hundreds of thousands of candidate permutations.
    """
    n = len(state["route"])
    groups = defaultdict(list)
    seen = set()
    for i in range(1, n+1):
        local = defaultdict(list)
        widths = (1,) if i == n else (1, 2, 3, 5, 8, 13, 20, n-i)
        for width in sorted(set(widths)):
            end = min(n-1, i+width-1)
            base_score = window_score(state, i, end, 1, troublesome)
            if base_score is None:
                continue
            targets = list(state["route"][i:end+1])
            offsets = sorted({0, 1, len(targets)//2, len(targets)})
            for tomato in missing:
                for offset in offsets:
                    for op in OPERATORS:
                        seq = insert_edit(targets, tomato, offset, op)
                        if seq is None: continue
                        k = (i, end, tuple(seq))
                        if k in seen: continue
                        seen.add(k)
                        trouble_count = sum(min(5, troublesome[pair])
                                            for pair in edge_pairs(state["route"][i-1], seq))
                        # Favor difficult neighborhoods while avoiding repeated
                        # impossible pairs in the proposed route.
                        score = base_score + 2.3 * min(3, trouble_count)
                        score -= .6*len(seq) + .5*sum(
                            troublesome[pair] for pair in edge_pairs(state["route"][i-1], seq))
                        if i >= 24 and tomato in (27, 38): score += 3.0
                        if op != "direct": score += .3
                        score += RNG.random()*1.5
                        local[tomato].append((score, source_idx, source_name,
                                              i, end, tomato, tuple(seq), op))
        # Keep three *different* choices per cut and missing tomato.
        for tomato, candidates in local.items():
            candidates.sort(key=lambda x: -x[0])
            groups[tomato].extend(candidates[:3])
    for group in groups.values(): group.sort(key=lambda x: -x[0])
    return groups


def candidate_search(p, pool, missing, troublesome, evaluator, started):
    # Multi-start: all known/produced 39 states get at least an opportunity.
    # Fairness: cycle across sources and missing tomatoes rather than top score only.
    grouped = defaultdict(list)
    for idx, (name, state) in enumerate(pool):
        for tomato, group in insertion_plans(name, state, missing, troublesome, idx).items():
            grouped[(idx, tomato)] = group
    keys = list(grouped)
    RNG.shuffle(keys)
    positions = {key: 0 for key in keys}
    tried = 0
    full = 0
    failures = Counter()
    failed_edges = Counter()
    progress = {"legs_repaired": 0, "legs_needed": 0}
    operator_stats = Counter()
    found = ctx = None
    seen_routes = set()
    while tried < MAX_INSERT_ATTEMPTS:
        if time.monotonic()-started >= MAX_RUNTIME_SECONDS: break
        did = False
        for key in keys:
            if tried >= MAX_INSERT_ATTEMPTS or time.monotonic()-started >= MAX_RUNTIME_SECONDS:
                break
            group = grouped[key]
            pos = positions[key]
            while pos < len(group):
                _, srcidx, name, i, end, tomato, seq, op = group[pos]
                pos += 1
                fp = (fingerprint(pool[srcidx][1]), i, end, seq)
                if fp not in seen_routes:
                    seen_routes.add(fp)
                    break
            else:
                positions[key] = pos
                continue
            positions[key] = pos
            did = True
            state = pool[srcidx][1]
            start, deadline, old, reserved, prefix, suffix = window(state, i, end)
            cap = p.n_exceptions - reserved
            if cap < 0: continue
            tried += 1
            operator_stats[op] += 1
            repaired, reached, why, edge = v4.repair(
                p, evaluator, state["route"][i-1], list(seq),
                start, deadline, cap, old)
            if tried % 25 == 0:
                print(f"Faza 40: {tried}/{MAX_INSERT_ATTEMPTS} prova "
                      f"| {evaluator.count} eval | {full} riparime te plota",
                      flush=True)
            if (reached > progress["legs_repaired"] or
                (reached == progress["legs_repaired"] and
                 len(seq) < progress.get("legs_needed", 10**9))):
                progress = {"attempt": tried, "source": name, "tomato": tomato,
                            "cut": i, "end": end, "operator": op,
                            "legs_repaired": reached, "legs_needed": len(seq),
                            "failed_edge": edge}
            if repaired is None:
                failures[why or "unknown"] += 1
                if edge:
                    failed_edges[f"{edge['from']}->{edge['to']}"] += 1
                continue
            full += 1
            proposal = stitch(state, i, end, list(seq), repaired, prefix, suffix)
            if (len(proposal["route"]) == 40 and
                len(set(proposal["route"])) == 40 and
                v4.verify(p, proposal)):
                found = proposal
                ctx = {"attempt": tried, "source": name, "tomato": tomato,
                       "cut": i, "end": end, "operator": op,
                       "special_used": proposal["special_used"]}
                print("*** 40/49 I VERIFIKUAR!", ctx, flush=True)
                break
            failures["failed_full_verification"] += 1
        if found is not None or not did:
            break
    return found, ctx, tried, full, failures, failed_edges, progress, operator_stats


def write_exclusive(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write temp first, then create target atomically; never overwrite.
    payload = (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("xb") as f:
        f.write(payload)
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    print("U ruajt:", path.name, flush=True)


def main():
    p, seeds, missing, troublesome, paths, fingerprints = load_inputs()
    print("BAZA: 39/49 | SPECIAL", seeds[0][1]["special_used"],
          "| dita", round(seeds[0][1]["current_time"], 6), flush=True)
    print("Burimet:", [n for n, _ in seeds], "| domatet qe mungojne:", missing, flush=True)
    if "--check-only" in sys.argv:
        print("CHECK-ONLY OK. Asnje skedar nuk u ndryshua.", flush=True)
        return
    if any(x.exists() for x in (REPORT, ARCHIVE, CAND40)):
        raise RuntimeError("Rezultati V9 ekziston. Nuk mbishkruhet!")
    v4.MAX_EVALUATIONS = MAX_EVALUATIONS
    evaluator = v4.Evaluator(p)
    start = time.monotonic()
    reached_budget = False
    pool, variants = list(seeds), []
    neutral_attempts = neutral_full = insert_attempts = insert_full = 0
    neutral_failures = Counter()
    insert_failures = Counter()
    failed_edges = Counter()
    operator_stats = Counter()
    progress = {"legs_repaired": 0, "legs_needed": 0}
    found = ctx = None
    try:
        pool, variants, neutral_attempts, neutral_full, neutral_failures = make_archive(
            p, seeds, evaluator, troublesome, start)
        print(f"Multi-start: {len(pool)} rrugë burimore (perfshire {len(variants)} "
              "alternativa V9 te verifikuara)", flush=True)
        (found, ctx, insert_attempts, insert_full, insert_failures,
         failed_edges, progress, operator_stats) = candidate_search(
             p, pool, missing, troublesome, evaluator, start)
    except v4.BudgetReached:
        reached_budget = True
        print("U arrit limiti i vleresimeve orbitale.", flush=True)
    for label, path in paths.items():
        if sha(path) != fingerprints[label]:
            raise RuntimeError(f"NDAL: burimi i mbrojtur {label} ndryshoi!")
    # Only fully verified paths are eligible for promotion/saving.
    for _, state in variants:
        if not v4.verify(p, state):
            raise RuntimeError("Alternativa 39 nuk kaloi riverifikimin!")
    if found is not None and not v4.verify(p, found):
        raise RuntimeError("Kandidati 40 nuk kaloi riverifikimin!")
    if variants:
        write_exclusive(ARCHIVE, {"level": 39, "verified": True,
                                  "source": "alns_v9_multistart.py",
                                  "frontier": [state for _, state in variants]})
    if found is not None:
        write_exclusive(CAND40, {"level": 40, "verified": True,
                                 "source": "alns_v9_multistart.py", "frontier": [found]})
    report = {
        "algorithm": "ALNS V9 multi-start 39-neutral reconstruction + guided 40 insertion",
        "base_level": 39, "best_level": 40 if found is not None else 39,
        "candidate_verified": found is not None,
        "neutral_attempts": neutral_attempts,
        "neutral_full_repairs": neutral_full,
        "new_neutral_variants": len(variants),
        "total_start_states": len(pool),
        "insertion_attempts": insert_attempts,
        "insertion_full_repairs": insert_full,
        "transfer_evaluations": evaluator.count,
        "evaluation_limit": MAX_EVALUATIONS,
        "evaluation_limit_reached": reached_budget,
        "runtime_limit_seconds": MAX_RUNTIME_SECONDS,
        "runtime_limit_reached": time.monotonic()-start >= MAX_RUNTIME_SECONDS,
        "neutral_failures": dict(neutral_failures),
        "insertion_failures": dict(insert_failures),
        "most_failed_edges": failed_edges.most_common(12),
        "closest_repair": progress,
        "operator_attempts": dict(operator_stats),
        "candidate_context": ctx,
        "elapsed_seconds": round(time.monotonic()-start, 2),
        "protected_38_sha256": fingerprints["38"],
        "protected_39_sha256": fingerprints["39"],
        "protected_alt39_sha256": fingerprints["v7_alt39"],
        "seed": SEED,
    }
    write_exclusive(REPORT, report)
    print("="*55)
    print("ALNS V9:", report["best_level"], "/49 |", evaluator.count, "vleresime")
    print("39 te reja:", len(variants), "| Prova per 40:", insert_attempts)
    print("Riparime te plota per 40:", insert_full)
    print("Kandidat i verifikuar:", found is not None)
    print("Rekordet 38/39: TE PAPREKURA")
    print("="*55)


if __name__ == "__main__":
    main()
