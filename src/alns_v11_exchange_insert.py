"""ESA SpOC4 | ALNS V11: 1-for-1 tomato exchange, then insertion 39->40.

Unlike V9/V10 which optimized order and transfer times for a FIXED set of
39 tomatoes, V11 first explores genuinely different 39-tomato subsets by
replacing one visited tomato (other than start 0) with one missing tomato.
Each alternative and every 40-tomato candidate is verified with the official
PyKEP-backed V4.verify(). Existing algorithms and records are never edited.

Run only with the Kaggle PyKEP Python:
  .../esa-pykep-env/bin/python -u src/alns_v11_exchange_insert.py --check-only
  .../esa-pykep-env/bin/python -u src/alns_v11_exchange_insert.py
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import math
import os
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import alns_v4_dp_repair as v4  # noqa: E402
import alns_v9_multistart as v9  # noqa: E402
import alns_v10_temporal_labels as v10  # noqa: E402

BASE38 = ROOT / "results/best_verified_38.json"
BACKUP38 = ROOT / "kaggle_backups/results/best_verified_38.json"
BASE39 = ROOT / "results/best_verified_39.json"
ALT39 = ROOT / "results/alns_v7_verified_39_neutral.json"
V9_ARCHIVE = ROOT / "results/alns_v9_multistart_verified_39_archive.json"
V10_REPORT = ROOT / "results/alns_v10_temporal_labels_report.json"
REPORT = ROOT / "results/alns_v11_exchange_insert_report.json"
ARCHIVE = ROOT / "results/alns_v11_verified_39_exchange_archive.json"
CAND40 = ROOT / "results/alns_v11_verified_40_candidate.json"

SEED = 17911
RNG = random.Random(SEED)
MAX_EXCHANGES = 160
MAX_INSERTIONS = 380
MAX_EXCHANGE_VARIANTS = 8
MAX_EVALUATIONS = 150_000
MAX_SECONDS = 720
MIN_TOF = 0.5
PROGRESS_EVERY = 20


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs():
    # V10 already verifies baseline, V7 and each V9 archive member using PyKEP.
    p, seeds, missing, problematic, previous = v10.load_seeds()
    if not V10_REPORT.is_file():
        raise RuntimeError("Mungon raporti i V10, kontrollo GitHub checkout")
    record = json.loads(V10_REPORT.read_text(encoding="utf-8"))
    if record.get("base_level") != 39 or record.get("best_level") not in (39, 40):
        raise RuntimeError("Raporti i V10 nuk i perket problemit easy/49")
    for raw, n in record.get("most_failed_edges", []):
        try:
            a, b = (int(x) for x in raw.split("->"))
            problematic[(a, b)] += int(n)
        except (ValueError, TypeError, AttributeError):
            continue
    # Fingerprint every prerequisite, including the previous report.
    protected = {str(Path(k)): v for k, v in previous.items()}
    protected[str(V10_REPORT)] = sha(V10_REPORT)
    for path in (BASE38, BACKUP38, BASE39, ALT39, V9_ARCHIVE):
        if not path.is_file():
            raise RuntimeError(f"Mungon burimi i mbrojtur: {path}")
        protected[str(path)] = sha(path)
    if BASE38.read_bytes() != BACKUP38.read_bytes():
        raise RuntimeError("Backup 38 nuk perputhet!")
    if len(missing) != 10 or len(seeds) < 1:
        raise RuntimeError("Instanca/burimet nuk jane te sakta")
    return p, seeds, missing, problematic, protected


def verify_protected(protected):
    for file, digest in protected.items():
        if sha(Path(file)) != digest:
            raise RuntimeError(f"U ndryshua nje burim i mbrojtur: {file}")


def atomic_new(path: Path, obj):
    """Write an immutable experiment artifact; refuse to overwrite even on retry."""
    content = (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("xb") as f:
        f.write(content)
    try:
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    print("U ruajt:", path.name, flush=True)


def cost_score(base, index, end, tomato, troublesome):
    """Favor feasible short windows and swaps near repeated failed orbital edges."""
    start, deadline, old, reserved, _, _ = v9.window(base, index, end)
    cap = 5 - reserved
    count = end-index+1
    slack = deadline-start-MIN_TOF*count
    if cap < 0 or slack < -1e-6:
        return None
    before = base["route"][index-1]
    after = base["route"][index+1] if index+1 < len(base["route"]) else None
    failures = troublesome[(before, tomato)]
    if after is not None:
        failures += troublesome[(tomato, after)]
    # A small amount of SPECIAL budget in a window can make repair possible.
    local_special = sum(leg["status"] == "SPECIAL" for leg in old)
    # Include edge signal for location coverage, but penalize known bad *new* edges.
    return (4.0 * min(local_special, 2) + 1.1 * min(8, slack/max(count, 1))
            - .75*count - 1.7*failures + .008*index + RNG.random()*3.0)


def exchange_plans(seeds, missing, troublesome):
    """Balance old tomato positions, new tomato choices, and seed sources."""
    grouped = defaultdict(list)
    seen = set()
    for source_index, (name, state) in enumerate(seeds):
        route = state["route"]
        for index in range(1, len(route)):
            for horizon in (2, 3, 5, 9):
                end = min(len(route)-1, index+horizon-1)
                # Keep right boundary unchanged, unless there is no suffix.
                if end == index and end < len(route)-1:
                    continue
                orig = route[index:end+1]
                for tomato in missing:
                    new_nodes = [tomato] + orig[1:]
                    choices = [("replace", new_nodes)]
                    if len(new_nodes) >= 3:
                        swapped = list(new_nodes)
                        swapped[0], swapped[1] = swapped[1], swapped[0]
                        choices.append(("replace_swap_next", swapped))
                    for op, targets in choices:
                        if end < len(route)-1 and targets[-1] != route[end]:
                            continue
                        key = (source_index, index, end, tomato, tuple(targets))
                        if key in seen: continue
                        seen.add(key)
                        score = cost_score(state, index, end, tomato, troublesome)
                        if score is None: continue
                        grouped[(source_index, tomato)].append(
                            (score, source_index, name, index, end, tomato, tuple(targets), op))
    for plans in grouped.values():
        plans.sort(key=lambda item:-item[0])
    keys = sorted(grouped)
    RNG.shuffle(keys)
    # Round robin across sources and tomatoes; no one missing tomato dominates.
    chosen = []
    seen_index = {key: 0 for key in keys}
    while len(chosen) < MAX_EXCHANGES:
        progressed = False
        for key in keys:
            idx = seen_index[key]
            if idx >= len(grouped[key]): continue
            chosen.append(grouped[key][idx]); seen_index[key] += 1
            progressed = True
            if len(chosen) >= MAX_EXCHANGES: break
        if not progressed: break
    return chosen


def exchange_stage(p, seeds, missing, troublesome, evaluator, started):
    candidates = exchange_plans(seeds, missing, troublesome)
    archive = []
    seen_sets = {frozenset(s["route"]) for _, s in seeds}
    fails = Counter(); edges = Counter()
    attempts = 0; full = 0
    best_partial = {"legs_repaired":0, "legs_needed":0}
    for plan in candidates:
        if time.monotonic()-started >= MAX_SECONDS: break
        _, source_idx, name, i, end, tomato, targets, op = plan
        source = seeds[source_idx][1]
        start, deadline, old, reserved, prefix, suffix = v9.window(source, i, end)
        cap = p.n_exceptions-reserved
        attempts += 1
        try:
            repaired, reached, why, failed_edge = v10.repair_phase(
                p, evaluator, source["route"][i-1], list(targets),
                start, deadline, cap, old)
        except v4.BudgetReached:
            return archive, attempts, full, fails, edges, best_partial, True
        if (reached > best_partial["legs_repaired"] or
                (reached == best_partial["legs_repaired"] and
                 len(targets)<best_partial["legs_needed"])):
            best_partial = {"source":name, "tomato":tomato, "removed":source["route"][i],
                "cut":i, "end":end, "operator":op, "legs_repaired":reached,
                "legs_needed":len(targets), "failed_edge":failed_edge}
        if repaired is None:
            fails[why or "unknown"] += 1
            if failed_edge: edges[f"{failed_edge['from']}->{failed_edge['to']}"] += 1
            continue
        full += 1
        candidate = v9.stitch(source, i, end, list(targets), repaired, prefix, suffix)
        if (len(candidate["route"]) != 39 or
                len(set(candidate["route"])) != 39 or
                not v4.verify(p, candidate)):
            fails["full_verification_failed"] += 1
            continue
        key = frozenset(candidate["route"])
        if key in seen_sets:
            fails["duplicate_set"] += 1
            continue
        seen_sets.add(key)
        archive.append(candidate)
        print(f"V11 exchange: {len(archive)}/"f"{MAX_EXCHANGE_VARIANTS} variante 39 "
              f"| {source['route'][i]} -> {tomato} | "
              f"SPECIAL {candidate['special_used']}", flush=True)
        if len(archive) >= MAX_EXCHANGE_VARIANTS: break
        if attempts % PROGRESS_EVERY == 0:
            print(f"Exchange {attempts} | {evaluator.count} eval", flush=True)
    return archive, attempts, full, fails, edges, best_partial, False


def insert_stage(p, seeds, new_states, troublesome, evaluator, started):
    """Round-robin insertion using new 39 tomato sets (previously unexplored)."""
    # New sets are explored first; previously explored original seeds are fallback.
    pool = [(f"exchanged39_{i}", state) for i, state in enumerate(new_states,1)]
    pool.extend(seeds)
    by_key = defaultdict(list)
    seen = set()
    for idx, (source, state) in enumerate(pool):
        missing = sorted(set(range(p.n_tomatoes)) - set(state["route"]))
        for tomato, plans in v9.insertion_plans(source, state, missing,
                                                  troublesome, idx).items():
            by_key[(idx, tomato)] = plans
    keys = sorted(by_key)
    # Favor new sets but preserve coverage of all sources and missing tomatoes.
    new_keys = [key for key in keys if key[0] < len(new_states)]
    old_keys = [key for key in keys if key[0] >= len(new_states)]
    RNG.shuffle(new_keys); RNG.shuffle(old_keys)
    if new_keys:
        keys = new_keys + old_keys
    else:
        keys = old_keys
    pos = {key:0 for key in keys}
    attempts = 0; full = 0; fails = Counter(); edges = Counter()
    best_partial = {"legs_repaired":0, "legs_needed":0}
    found = None; found_info = None; budget_reached = False
    # Alternate between sources after first round; never indefinitely revisit a seed.
    while attempts < MAX_INSERTIONS and time.monotonic()-started < MAX_SECONDS:
        progressed = False
        for key in keys:
            if attempts >= MAX_INSERTIONS or time.monotonic()-started >= MAX_SECONDS:
                break
            plans = by_key[key]
            while pos[key] < len(plans):
                plan = plans[pos[key]]
                pos[key] += 1
                _, source_idx, source_name, i, end, tomato, targets, op = plan
                fp = (v9.fingerprint(pool[source_idx][1]), i, end, tuple(targets))
                if fp not in seen:
                    seen.add(fp); break
            else:
                continue
            progressed = True
            source = pool[source_idx][1]
            start, deadline, old, reserved, prefix, suffix = v9.window(source, i, end)
            cap = p.n_exceptions-reserved
            attempts += 1
            try:
                repaired, reached, why, failed_edge = v10.repair_phase(
                    p, evaluator, source["route"][i-1], list(targets),
                    start, deadline, cap, old)
            except v4.BudgetReached:
                budget_reached = True; break
            if (reached>best_partial["legs_repaired"] or
                (reached==best_partial["legs_repaired"] and
                 len(targets)<best_partial["legs_needed"])):
                best_partial = {"source":source_name,"tomato":tomato,"cut":i,"end":end,
                    "operator":op,"legs_repaired":reached,"legs_needed":len(targets),
                    "failed_edge":failed_edge}
            if repaired is None:
                fails[why or "unknown"] += 1
                if failed_edge: edges[f"{failed_edge['from']}->{failed_edge['to']}"] += 1
                continue
            full += 1
            candidate = v9.stitch(source, i, end, list(targets), repaired, prefix, suffix)
            if (len(candidate["route"]) == 40 and
                    len(set(candidate["route"])) == 40 and v4.verify(p, candidate)):
                found = candidate
                found_info = {"source":source_name,"tomato":tomato,"attempt":attempts,
                    "cut":i,"end":end,"operator":op}
                return found, found_info, attempts, full, fails, edges, best_partial, budget_reached
            fails["full_verification_failed"] += 1
            if attempts % PROGRESS_EVERY == 0:
                print(f"V11 insertion {attempts}/{MAX_INSERTIONS} "
                      f"| {evaluator.count} eval | {full} full",flush=True)
        if budget_reached or not progressed or found is not None:
            break
    return found, found_info, attempts, full, fails, edges, best_partial, budget_reached


def run():
    p, seeds, missing, trouble, protected = load_inputs()
    print("INSTANCA:",p.n_tomatoes, "domate | bazat:",[n for n,_ in seeds],
          "| mungojne:",missing,flush=True)
    if "--check-only" in sys.argv:
        print("CHECK-ONLY OK; nuk u ndryshua asnje skedar",flush=True)
        return
    if any(path.exists() for path in (REPORT, ARCHIVE, CAND40)):
        raise RuntimeError("Rezultate V11 ekzistuese; nuk mbishkruhen")
    v4.MAX_EVALUATIONS = MAX_EVALUATIONS
    evaluator = v4.Evaluator(p)
    started = time.monotonic()
    variants, ex_tries, ex_full, ex_fails, ex_edges, ex_partial, budget1 = exchange_stage(
        p, seeds, missing, trouble, evaluator, started)
    print("EXCHANGE: variante 39 me perberje tjeter:",len(variants),
          "| prova:",ex_tries, "| eval:",evaluator.count,flush=True)
    candidate = context = None
    ins_tries = ins_full = 0
    ins_fails = Counter(); ins_edges = Counter()
    ins_partial={"legs_repaired":0,"legs_needed":0}
    budget2=False
    # If no new 39-set was found, older 39 seeds have already been searched
    # in V9/V10. Avoid simply repeating those insertion experiments.
    if variants and not budget1 and time.monotonic()-started < MAX_SECONDS:
        (candidate,context,ins_tries,ins_full,ins_fails,ins_edges,
         ins_partial,budget2)=insert_stage(p,seeds,variants,trouble,evaluator,started)
    # Mandatory final verification before writing anything permanent.
    for state in variants:
        if not v4.verify(p,state) or len(state["route"])!=39:
            raise RuntimeError("Alternativa e shkembimit 39 eshte e pavlefshme")
    if candidate is not None and (len(candidate["route"])!=40 or not v4.verify(p,candidate)):
        raise RuntimeError("Kandidati 40 deshtoi ne verifikimin perfundimtar")
    verify_protected(protected)
    if variants:
        atomic_new(ARCHIVE,{"level":39,"verified":True,
            "source":"alns_v11_exchange_insert.py","frontier":variants})
    if candidate is not None:
        atomic_new(CAND40,{"level":40,"verified":True,
            "source":"alns_v11_exchange_insert.py","frontier":[candidate]})
    report = {
        "algorithm":"ALNS V11 - verified 1-for-1 subset exchange + temporal insertion",
        "base_level":39,"best_level":40 if candidate is not None else 39,
        "candidate_verified":candidate is not None,
        "seed":SEED,"source_seed_names":[n for n,_ in seeds],
        "exchange_attempts":ex_tries,"exchange_full_repairs":ex_full,
        "new_39_subsets":len(variants),"exchange_failure_reasons":dict(ex_fails),
        "exchange_closest_repair":ex_partial,
        "insertion_attempts":ins_tries,"insertion_full_repairs":ins_full,
        "insertion_failure_reasons":dict(ins_fails),
        "insertion_closest_repair":ins_partial,
        "most_failed_edges":(ex_edges+ins_edges).most_common(15),
        "transfer_evaluations":evaluator.count,
        "evaluation_limit_reached":bool(budget1 or budget2),
        "elapsed_seconds":round(time.monotonic()-started,3),
        "runtime_limit_reached":time.monotonic()-started>=MAX_SECONDS,
        "candidate_context":context,
        "protected_38_sha256":protected[str(BASE38)],
        "protected_39_sha256":protected[str(BASE39)],
        "protected_v9_archive_sha256":protected[str(V9_ARCHIVE)],
        "protected_v10_report_sha256":protected[str(V10_REPORT)],
    }
    atomic_new(REPORT,report)
    print("V11:",report["best_level"],"/49 | variante 39:",len(variants),
          "| exchange:",ex_tries,"| insert:",ins_tries,
          "| eval:",evaluator.count,flush=True)
    print("Rekordet 38/39: TE PAPREKURA",flush=True)


if __name__=="__main__":
    run()
