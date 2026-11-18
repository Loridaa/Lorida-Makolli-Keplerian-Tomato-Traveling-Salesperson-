"""ESA SpOC4 - ALNS V8: 39->40 edge-escape neighborhood repair.

Independent experiment. It never changes best_verified_38.json,
best_verified_39.json, prior algorithms, or any existing output.

Motivation: V7 found 12 verified neutral 39 paths, but did not reach 40.
Its nearest repair failed on 13 -> 27 after 11/13 legs. V8 explicitly
explores insertion offsets and neighboring permutations, including orders
that avoid that edge, from both the official 39 record and V7's saved
neutral alternative.

Real transfer evaluation & full verification are delegated to ALNS V4's
verified PyKEP-based helpers. Tests must run in the Kaggle PyKEP env.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import alns_v4_dp_repair as v4

BASE38 = ROOT / "results/best_verified_38.json"
BACKUP38 = ROOT / "kaggle_backups/results/best_verified_38.json"
BASE39 = ROOT / "results/best_verified_39.json"
ALT39 = ROOT / "results/alns_v7_verified_39_neutral.json"
REPORT = ROOT / "results/alns_v8_edge_escape_report.json"
CANDIDATE = ROOT / "results/alns_v8_verified_40_candidate.json"

SEED = 17708
RNG = random.Random(SEED)
MAX_ATTEMPTS = 320
MAX_EVALUATIONS = 160_000
MIN_TOF = .5
EPS = 1e-6


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_state(path, level, problem):
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("level") != level or obj.get("verified") is not True or len(obj.get("frontier", [])) != 1:
        raise RuntimeError(f"Gabim ne formatin e rekordit: {path.name}")
    state = obj["frontier"][0]
    if (len(state.get("route", [])) != level or
            len(state.get("transfers", [])) != level - 1 or not v4.verify(problem, state)):
        raise RuntimeError(f"Rekordi nuk u verifikua me PyKEP: {path.name}")
    return state


def load_sources():
    for path in (BASE38, BACKUP38, BASE39, ALT39):
        if not path.is_file():
            raise RuntimeError("Mungon: " + str(path))
    fingerprints = {"38": sha(BASE38), "39": sha(BASE39), "alt39": sha(ALT39)}
    if sha(BACKUP38) != fingerprints["38"]:
        raise RuntimeError("Rekordi 38 nuk perputhet me kopjen rezervë")
    problem = v4.load_problem()
    if problem.n_tomatoes != 49:
        raise RuntimeError("Pritet instanca easy me 49 domate")
    s38 = record_state(BASE38, 38, problem)
    s39 = record_state(BASE39, 39, problem)
    alt = record_state(ALT39, 39, problem)
    if not set(s38["route"]).issubset(set(s39["route"])):
        raise RuntimeError("Rekordi 39 nuk permban 38 domatet e meparshme")
    if s39["special_used"] > problem.n_exceptions or alt["special_used"] > problem.n_exceptions:
        raise RuntimeError("SPECIAL budget i tejkaluar")
    sources = [("verified39", s39)]
    # V7 saved one of the 12 neutral alternatives, not all 12.
    if (alt["route"] != s39["route"] or alt["transfers"] != s39["transfers"]):
        sources.append(("v7_neutral", alt))
    return problem, sources, fingerprints


def frame(state, i, end):
    route, legs = state["route"], state["transfers"]
    n = len(route)
    if not (1 <= i <= n and i - 1 <= end < n):
        raise ValueError("Dritare e pavlefshme")
    prefix = legs[:i-1]
    suffix = legs[end:] if end < n-1 else []
    start = float(prefix[-1]["start_time"] + prefix[-1]["tof"]) if prefix else 0.0
    deadline = float(suffix[0]["start_time"]) if suffix else 200.0
    old = legs[i-1:end]
    reserved = sum(l["status"] == "SPECIAL" for l in prefix + suffix)
    return start, deadline, old, reserved, prefix, suffix


def permutation(nodes, tomato, offset, op):
    seq = list(nodes)
    seq.insert(offset, tomato)
    t = seq.index(tomato)
    if op == "direct":
        pass
    elif op == "swap_left" and t >= 1:
        seq[t-1], seq[t] = seq[t], seq[t-1]
    elif op == "swap_right" and t + 1 < len(seq):
        seq[t+1], seq[t] = seq[t], seq[t+1]
    elif op == "reverse_neighbors" and len(seq) >= 3:
        lo = min(max(t-1, 0), len(seq)-3)
        seq[lo:lo+3] = reversed(seq[lo:lo+3])
    else:
        return None
    return seq


def generate_plans(problem, sources, missing):
    grouped = defaultdict(list)
    seen = set()
    for source_name, state in sources:
        route, legs = state["route"], state["transfers"]
        n = len(route)
        pos13 = route.index(13) if 13 in route else n
        for i in range(1, n+1):
            for width in (1, 2, 3, 5, 8, 12, 18, 25):
                if i == n and width != 1:
                    continue
                end = min(n-1, i+width-1)
                start, deadline, old, reserved, _, _ = frame(state, i, end)
                targets = list(route[i:end+1])
                # The new leg count increases by exactly one.
                count = len(targets) + 1
                slack = deadline - start - MIN_TOF*count
                if slack < -EPS or problem.n_exceptions-reserved < 0:
                    continue
                specials = sum(l["status"] == "SPECIAL" for l in old)
                offsets = sorted({0, 1, len(targets)//2, len(targets)})
                for tomato in missing:
                    for offset in offsets:
                        for operator in ("direct", "swap_left", "swap_right", "reverse_neighbors"):
                            replacement = permutation(targets, tomato, offset, operator)
                            if replacement is None:
                                continue
                            # V7's closest candidate failed on 13->27. Explicitly
                            # favor alternative orders that bypass that same edge.
                            if tomato == 27 and any(a == 13 and b == 27
                                                    for a, b in zip([route[i-1]] + replacement,
                                                                    replacement)):
                                continue
                            key = (source_name, i, end, tuple(replacement))
                            if key in seen:
                                continue
                            seen.add(key)
                            dist13 = abs(i - pos13)
                            score = (14*min(2, specials) + 4*min(7, max(0.0,slack)/count)
                                     - .55*count - .16*dist13 + (6 if tomato == 27 else 0)
                                     + (1 if source_name == "v7_neutral" else 0))
                            if operator != "direct":
                                score += 1.3  # Unlike V7, favor actual route reordering.
                            # Small deterministic tiebreaker for diversity.
                            score += RNG.random() * .6
                            grouped[(source_name, tomato, 0 if count <= 3 else (1 if count <= 8 else 2))].append(
                                (score, source_name, i, end, tomato, replacement, operator))
    for group in grouped.values():
        group.sort(key=lambda x: -x[0])
    groups = sorted(grouped)
    RNG.shuffle(groups)
    selected, per_window = [], Counter()
    indexes = {g: 0 for g in groups}
    while len(selected) < MAX_ATTEMPTS:
        added = 0
        for g in groups:
            plans = grouped[g]
            while indexes[g] < len(plans):
                plan = plans[indexes[g]]
                indexes[g] += 1
                _, source_name, i, end, tomato, replacement, op = plan
                key = (source_name, i, end, tomato)
                if per_window[key] >= 2:
                    continue
                per_window[key] += 1
                selected.append(plan)
                added += 1
                break
            if len(selected) == MAX_ATTEMPTS:
                break
        if added == 0:
            break
    return selected


def stitch(state, i, end, target_nodes, repaired, prefix, suffix):
    route = list(state["route"][:i]) + list(target_nodes) + list(state["route"][end+1:])
    legs = list(prefix) + repaired["legs"] + list(suffix)
    final = legs[-1]
    return {
        "route": route,
        "transfers": legs,
        "special_used": sum(l["status"] == "SPECIAL" for l in legs),
        "current_time": float(final["start_time"] + final["tof"]),
    }


def exclusive_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as out:
        json.dump(obj, out, indent=2, ensure_ascii=False)
        out.write("\n")
    print("U ruajt:", path.name, flush=True)


def main():
    p, sources, fingers = load_sources()
    base = sources[0][1]
    unused = sorted(set(range(p.n_tomatoes)) - set(base["route"]))
    if len(unused) != 10:
        raise RuntimeError("Rekordi 39 duhet te kete 10 domate te pambedhura")
    print("BAZA 39/49:", round(float(base["current_time"]), 6),
          "dite, SPECIAL:", base["special_used"], flush=True)
    print("Burime:", [n for n, _ in sources], flush=True)
    print("Domatet qe mungojne:", unused, flush=True)
    if "--check-only" in sys.argv:
        print("CHECK-ONLY OK. Asnje skedar nuk u ndryshua.", flush=True)
        return
    if REPORT.exists() or CANDIDATE.exists():
        raise RuntimeError("Eksperimenti V8 ekziston. Nuk mbishkruhet!")
    v4.MAX_EVALUATIONS = MAX_EVALUATIONS
    evaluator = v4.Evaluator(p)
    plans = generate_plans(p, sources, unused)
    print("Plani: ", len(plans), "prova te balancuara, kufiri:", MAX_EVALUATIONS, flush=True)
    source_map = dict(sources)
    start_time = time.monotonic()
    attempts = 0
    full = 0
    failures = Counter()
    failure_edges = Counter()
    closest = {"legs_repaired": 0, "legs_needed": 0}
    found, context = None, None
    limit_hit = False
    try:
        for _, source_name, i, end, tomato, target, op in plans:
            state = source_map[source_name]
            start, deadline, old, reserved, prefix, suffix = frame(state, i, end)
            cap = p.n_exceptions - reserved
            attempts += 1
            repaired, reached, reason, edge = v4.repair(
                p, evaluator, state["route"][i-1], target,
                start, deadline, cap, old)
            if (reached > closest["legs_repaired"]
                or (reached == closest["legs_repaired"] and
                    len(target) < closest.get("legs_needed", 10**9))):
                closest = {"attempt": attempts, "source": source_name,
                           "cut": i, "end": end, "tomato": tomato, "operator": op,
                           "legs_repaired": reached, "legs_needed": len(target),
                           "failed_edge": edge}
            if repaired is None:
                failures[reason or "unknown"] += 1
                if edge:
                    failure_edges[f"{edge['from']}->{edge['to']}"] += 1
                continue
            full += 1
            candidate = stitch(state, i, end, target, repaired, prefix, suffix)
            if (len(candidate["route"]) == 40 and
                    len(set(candidate["route"])) == 40 and
                    v4.verify(p, candidate)):
                found = candidate
                context = {"attempt": attempts, "source": source_name,
                           "cut": i, "end": end, "tomato": tomato,
                           "operator": op}
                print("*** KANDIDAT 40/49 I VERIFIKUAR!", flush=True)
                break
            failures["full_verification"] += 1
    except v4.BudgetReached:
        limit_hit = True
        print("U arrit kufiri i vleresimeve.", flush=True)
    if (sha(BASE38) != fingers["38"] or sha(BACKUP38) != fingers["38"] or
            sha(BASE39) != fingers["39"] or sha(ALT39) != fingers["alt39"]):
        raise RuntimeError("NDAL: nje rekord ose alternative u ndryshua!")
    if found is not None:
        exclusive_json(CANDIDATE, {"level": 40, "verified": True,
                                   "source": "alns_v8_edge_escape.py",
                                   "frontier": [found]})
    report = {
        "algorithm": "ALNS V8 edge-escape multi-offset + local permutation DP repair",
        "base_level": 39,
        "best_level": 40 if found is not None else 39,
        "candidate_verified": found is not None,
        "attempts": attempts,
        "max_attempts": MAX_ATTEMPTS,
        "full_temporal_repairs": full,
        "transfer_evaluations": evaluator.count,
        "evaluation_limit": MAX_EVALUATIONS,
        "evaluation_limit_reached": limit_hit,
        "failure_reasons": dict(failures),
        "most_failed_edges": failure_edges.most_common(10),
        "closest_repair": closest,
        "candidate_context": context,
        "sources": [name for name,_ in sources],
        "missing_tomatoes": unused,
        "protected_38_sha256": fingers["38"],
        "protected_39_sha256": fingers["39"],
        "protected_alt39_sha256": fingers["alt39"],
        "elapsed_seconds": round(time.monotonic()-start_time, 2),
        "seed": SEED,
    }
    exclusive_json(REPORT, report)
    print("ALNS V8:", report["best_level"], "/49 | prova:", attempts,
          "| eval:", evaluator.count,
          "| riparime te plota:", full, flush=True)
    print("Bllokimet kryesore:", report["most_failed_edges"][:5], flush=True)
    print("Rekordet 38/39: TE PAPREKURA", flush=True)


if __name__ == "__main__":
    main()
