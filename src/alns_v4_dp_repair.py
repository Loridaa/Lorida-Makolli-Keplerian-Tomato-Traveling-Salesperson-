"""ESA SpOC4 - ALNS V4: gap insertion with Pareto time/SPECIAL DP repair.

Motivation: V3 repaired at most 19/31 legs and zero complete routes. V4 focuses
on smaller, targeted segments around existing SPECIAL transfers and uses a
resource-constrained dynamic program to keep the earliest achievable arrival
for each number of SPECIAL transfers. Early arrival dominates later arrival
when they have identical SPECIAL usage, because waiting is allowed.

No heuristic fitness is accepted as proof: every 39-tomato candidate is fully
recomputed using the official TomatoProblem.compute_transfer() API. The
verified 38-tomato record, backed-up copy, and prior algorithms are read-only.

Usage:
  /kaggle/working/esa-pykep-env/bin/python -u src/alns_v4_dp_repair.py --check-only
  /kaggle/working/esa-pykep-env/bin/python -u src/alns_v4_dp_repair.py
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
from utils_users import TomatoProblem  # noqa: E402

BASE = ROOT / "results/best_verified_38.json"
BACKUP = ROOT / "kaggle_backups/results/best_verified_38.json"
REPORT = ROOT / "results/alns_v4_dp_repair_report.json"
CANDIDATE = ROOT / "results/alns_v4_dp_verified_39_candidate.json"
SEED = 172
MAX_ATTEMPTS = 110
MAX_EVALUATIONS = 120_000
MIN_TOF = 0.5
EPS = 1e-6
rng = random.Random(SEED)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_problem():
    paths = []
    for folder in (ROOT / "data", ROOT / "kaggle_backups/data"):
        if folder.is_dir():
            paths.extend(folder.rglob("*.kttsp"))
    paths = sorted({p.resolve() for p in paths},
                   key=lambda p: ("easy" not in p.name.lower(), str(p)))
    for path in paths:
        try:
            p = TomatoProblem(str(path))
        except Exception:
            continue
        if getattr(p, "n_tomatoes", None) == 49:
            print("Instanca:", path.name, "| domate:", p.n_tomatoes, flush=True)
            return p
    raise RuntimeError("Nuk u gjet instanca e sakte me 49 domate.")


def verify(p, state):
    """Independently re-evaluate ALL orbital transfers and constraints."""
    try:
        route, transfers = state["route"], state["transfers"]
        if (not isinstance(route, list) or not route or route[0] != 0
                or len(set(route)) != len(route) or len(route) > p.n_tomatoes
                or len(transfers) != len(route) - 1
                or any(type(x) is not int or not 0 <= x < p.n_tomatoes
                       for x in route)):
            return False
        last_time = 0.0
        specials = 0
        for i, tr in enumerate(transfers):
            if (tr["from"], tr["to"]) != (route[i], route[i + 1]):
                return False
            d, tof, saved_dv = [float(tr[key]) for key in ("start_time", "tof", "dv")]
            if (not all(map(math.isfinite, (d, tof, saved_dv)))
                    or d < last_time - 1e-5 or tof <= 0
                    or d + tof > 200 + EPS):
                return False
            actual = float(p.compute_transfer(route[i], route[i + 1], d, tof))
            if (not math.isfinite(actual)
                    or abs(actual - saved_dv) > max(1e-4, abs(actual) * 2e-6)):
                return False
            if tr["status"] == "NORMAL":
                if actual >= p.dv_threshold - EPS:
                    return False
            elif tr["status"] == "SPECIAL":
                if actual >= p.exception_threshold - EPS:
                    return False
                specials += 1
            else:
                return False
            last_time = d + tof
        return (specials == state["special_used"] and specials <= p.n_exceptions
                and abs(last_time - state["current_time"]) < 1e-4)
    except Exception:
        return False


class BudgetReached(Exception):
    pass


class Evaluator:
    def __init__(self, p):
        self.p, self.count, self.cache = p, 0, {}

    def get(self, a, b, start, tof):
        d, t = round(float(start), 6), round(float(tof), 6)
        key = (int(a), int(b), d, t)
        if key not in self.cache:
            if self.count >= MAX_EVALUATIONS:
                raise BudgetReached
            self.count += 1
            try:
                value = float(self.p.compute_transfer(a, b, d, t))
            except Exception:
                value = math.inf
            self.cache[key] = value if math.isfinite(value) and value >= 0 else math.inf
        return self.cache[key]


def classify(p, dv):
    if dv < p.dv_threshold - .001:
        return "NORMAL", 0
    if dv < p.exception_threshold - .001:
        return "SPECIAL", 1
    return None


def add_state(table, new):
    """Earliest arrival dominates later arrivals with same SPECIAL count."""
    key = new["used"]
    old = table.get(key)
    if old is None or (new["t"], new["dv"]) < (old["t"], old["dv"]):
        table[key] = new


def repair(p, evaluator, left, targets, start, deadline, cap, original_legs):
    """Return (state, legs_reached, failure_reason, failure_edge).

    This is resource-constrained earliest-arrival dynamic programming, not a
    truncated beam search. Up to cap+1 non-dominated labels survive each leg.
    """
    warm = defaultdict(list)
    for leg in original_legs:
        warm[(leg["from"], leg["to"])].append(leg)
    states = [{"t": float(start), "used": 0, "dv": 0.0, "legs": []}]
    reached = 0
    for index, b in enumerate(targets):
        a = left if index == 0 else targets[index - 1]
        incoming = warm.get((a, b), [])
        ref = original_legs[min(index, len(original_legs)-1)] if original_legs else None
        pool = {}
        edge_count, time_count, dv_count, special_count = 0, 0, 0, 0
        for s in states:
            # Reuse known-valid transfers verbatim whenever timelines still fit.
            for old in incoming:
                dep, tof = float(old["start_time"]), float(old["tof"])
                if dep + EPS >= s["t"] and dep + tof <= deadline + EPS:
                    extra = int(old["status"] == "SPECIAL")
                    if s["used"] + extra <= cap:
                        add_state(pool, {
                            "t": round(dep + tof, 6),
                            "used": s["used"] + extra,
                            "dv": s["dv"] + float(old["dv"]),
                            "legs": s["legs"] + [dict(old)]})
            # Combine relative departure grid and absolute baseline timings.
            departures = {round(s["t"] + wait, 6) for wait in
                          (0., .25, .5, 1., 2., 3., 5., 8.)}
            durations = {.5, .75, 1., 1.5, 2., 3., 4., 5., 7., 10.}
            anchors = list(incoming)
            if ref is not None:
                anchors.append(ref)
            for tr in anchors:
                old_d = float(tr["start_time"])
                departures.update(round(old_d + off, 6) for off in
                                  (-1., -.5, -.25, 0., .25, .5, 1., 2.))
                old_t = float(tr["tof"])
                durations.update(round(old_t + off, 6) for off in
                                 (-1., -.5, 0., .5, 1.))
            durations = sorted(x for x in durations if x >= MIN_TOF)
            for dep in sorted(departures):
                if dep < s["t"] - EPS or dep + MIN_TOF > deadline + EPS:
                    continue
                for tof in durations:
                    if dep + tof > deadline + EPS:
                        continue
                    time_count += 1
                    dv = evaluator.get(a, b, dep, tof)
                    category = classify(p, dv)
                    if category is None:
                        continue
                    dv_count += 1
                    status, extra = category
                    used = s["used"] + extra
                    if used > cap:
                        special_count += 1
                        continue
                    leg = {"from": a, "to": b, "start_time": dep,
                           "tof": float(tof), "dv": dv, "status": status}
                    add_state(pool, {"t": round(dep + tof, 6),
                                     "used": used, "dv": s["dv"] + dv,
                                     "legs": s["legs"] + [leg]})
                    edge_count += 1
        if not pool:
            reason = ("no_time_options" if time_count == 0 else
                      "special_budget" if dv_count and special_count else
                      "no_transfer_under_threshold")
            return None, reached, reason, {"from": a, "to": b,
                                            "repair_leg_index": index + 1}
        states = list(pool.values())
        reached += 1
    # Lower used first preserves room for future opportunities.
    best = min(states, key=lambda s: (s["used"], s["t"], s["dv"]))
    return best, reached, None, None


def build_neighborhoods(p, route, legs, unused):
    """Prioritize short, time-feasible neighborhoods spanning existing SPECIALs."""
    n = len(route)
    specs = []
    for i in range(1, n):  # insert before old route[i], avoiding duplicate start 0
        prefix = legs[:i-1]
        start = float(prefix[-1]["start_time"] + prefix[-1]["tof"]) if prefix else 0.0
        for horizon in (2, 4, 7, 11, 18):
            end = min(n-1, i + horizon)
            suffix = legs[end:] if end < n-1 else []
            deadline = float(suffix[0]["start_time"]) if suffix else 200.0
            oldlegs = legs[i-1:end]
            specials = sum(t["status"] == "SPECIAL" for t in oldlegs)
            available = deadline - start
            length = end - i + 2  # original i..end + inserted tomato
            slack = available - MIN_TOF * length
            if slack < -EPS:
                continue
            # Prefer some SPECIAL flexibility, high slack/leg, and shorter repairs.
            score = (22 * min(2, specials) + 5 * min(6, max(0.0, slack)/length)
                     - 2.2 * length + .12 * i)
            specs.append({"i": i, "end": end, "start": start,
                          "deadline": deadline, "specials_in_window": specials,
                          "slack": slack, "score": score})
    # Variety across insertion sites and missing tomato choices.
    rng.shuffle(specs)
    specs.sort(key=lambda x: -x["score"])
    selected = []
    seen = set()
    # Round-robin: first cover different cut positions, then revisit cuts.
    for pass_num in range(3):
        used_cuts = set()
        for spec in specs:
            i = spec["i"]
            if i in used_cuts:
                continue
            used_cuts.add(i)
            for variant in range(2):
                x = unused[(i * 7 + spec["end"] * 3 + pass_num * 5 + variant * 3) % len(unused)]
                offset = min(variant, spec["end"] - i + 1)
                key = (i, spec["end"], x, offset)
                if key not in seen:
                    seen.add(key)
                    selected.append((spec, x, offset))
            if len(selected) >= MAX_ATTEMPTS:
                return selected[:MAX_ATTEMPTS]
    return selected[:MAX_ATTEMPTS]


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    # exclusive create: never overwrite a previous experiment or proven result
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("U ruajt:", path.name, flush=True)


def main():
    for path in (BASE, BACKUP):
        if not path.is_file():
            raise RuntimeError("Mungon rekordi i verifikuar ose backup-i.")
    fingerprint = sha(BASE)
    if sha(BACKUP) != fingerprint:
        raise RuntimeError("NDAL: backup-i i 38/49 nuk perputhet.")
    p = load_problem()
    record = json.loads(BASE.read_text(encoding="utf-8"))
    if (record.get("level") != 38 or record.get("verified") is not True
            or len(record.get("frontier", [])) != 1):
        raise RuntimeError("Formati i rekordit 38/49 nuk eshte i sakte.")
    base = record["frontier"][0]
    if not verify(p, base):
        raise RuntimeError("Rekordi 38/49 nuk u verifikua me PyKEP.")
    route, oldlegs = base["route"], base["transfers"]
    if len(route) != 38 or len(oldlegs) != 37:
        raise RuntimeError("Pritej saktesisht nje rruge 38/49.")
    unused = sorted(set(range(p.n_tomatoes)) - set(route))
    if len(unused) != 11:
        raise RuntimeError("Numri i domateve te mbetura nuk eshte 11.")
    print("BAZA OK: 38/49, SPECIAL", base["special_used"],
          "dita", round(base["current_time"], 6), flush=True)
    print("Domatet qe mungojne:", unused, flush=True)
    if "--check-only" in sys.argv:
        print("CHECK-ONLY OK. Asnje skedar nuk u ndryshua.", flush=True)
        return
    if REPORT.exists() or CANDIDATE.exists():
        raise RuntimeError("Eksperimenti V4 ekziston; nuk mbishkruhet.")

    choices = build_neighborhoods(p, route, oldlegs, unused)
    evaluator = Evaluator(p)
    failures = Counter()
    tried = 0
    full_repairs = 0
    progress = {"legs_repaired": 0, "legs_needed": 0}
    found = None
    found_context = None
    wallstart = time.monotonic()

    try:
        for spec, tomato, offset in choices:
            i, end = spec["i"], spec["end"]
            prefix = oldlegs[:i-1]
            suffix = oldlegs[end:] if end < len(route)-1 else []
            reserved = sum(leg["status"] == "SPECIAL" for leg in prefix + suffix)
            cap = p.n_exceptions - reserved
            if cap < 0:
                failures["negative_special_budget"] += 1
                continue
            targets = list(route[i:end+1])
            targets.insert(offset, tomato)
            left = route[i-1]
            original_window = oldlegs[i-1:end]
            tried += 1
            state, reached, reason, failed_edge = repair(
                p, evaluator, left, targets, spec["start"],
                spec["deadline"], cap, original_window)
            if reached > progress["legs_repaired"]:
                progress = {"attempt": tried, "cut": i, "end_index": end,
                            "missing_tomato": tomato, "insertion_offset": offset,
                            "legs_repaired": reached, "legs_needed": len(targets),
                            "failed_edge": failed_edge,
                            "specials_in_window": spec["specials_in_window"]}
            if state is None:
                failures[reason] += 1
            else:
                full_repairs += 1
                proposed = {
                    "route": route[:i] + targets + route[end+1:],
                    "transfers": prefix + state["legs"] + suffix,
                }
                proposed["special_used"] = sum(
                    leg["status"] == "SPECIAL" for leg in proposed["transfers"])
                last = proposed["transfers"][-1]
                proposed["current_time"] = float(last["start_time"] + last["tof"])
                if (len(proposed["route"]) == 39
                        and len(proposed["transfers"]) == 38
                        and verify(p, proposed)):
                    found = proposed
                    found_context = {"attempt": tried, "cut": i, "end_index": end,
                                     "tomato": tomato, "offset": offset,
                                     "special_used": proposed["special_used"]}
                    print("*** 39/49 I VERIFIKUAR! Dita:",
                          round(proposed["current_time"], 6), flush=True)
                    break
                failures["full_route_verification"] += 1
            if tried % 5 == 0:
                print(f"Prova {tried}/{len(choices)} | riparime te plota {full_repairs} "
                      f"| progres maks {progress['legs_repaired']}/"
                      f"{progress['legs_needed']} | eval {evaluator.count}", flush=True)
    except BudgetReached:
        print("U arrit limiti i vleresimeve:", MAX_EVALUATIONS, flush=True)

    if sha(BASE) != fingerprint or sha(BACKUP) != fingerprint:
        raise RuntimeError("NDAL: rekordi origjinal u ndryshua.")

    if found is not None:
        write_new(CANDIDATE, {"level": 39, "verified": True,
                              "source": "alns_v4_dp_repair.py",
                              "frontier": [found]})
    report = {
        "algorithm": "ALNS V4 targeted insertion + resource-constrained earliest-arrival DP",
        "base_level": 38,
        "best_level": 39 if found is not None else 38,
        "candidate_verified": found is not None,
        "instance": "easy.kttsp (49 tomatoes)",
        "seed": SEED,
        "attempts": tried,
        "max_attempts": MAX_ATTEMPTS,
        "full_temporal_repairs": full_repairs,
        "transfer_evaluations": evaluator.count,
        "evaluation_limit": MAX_EVALUATIONS,
        "evaluation_limit_reached": evaluator.count >= MAX_EVALUATIONS,
        "best_partial_repair": progress,
        "failure_reasons": dict(failures),
        "candidate_context": found_context,
        "elapsed_seconds": round(time.monotonic() - wallstart, 2),
        "protected_baseline_sha256": fingerprint,
    }
    write_new(REPORT, report)
    print("=" * 55, flush=True)
    print("ALNS V4:", report["best_level"], "/49",
          "| provat:", tried,
          "| eval:", evaluator.count,
          "| riparime te plota:", full_repairs, flush=True)
    print("Arsyet e deshtimit:", dict(failures), flush=True)
    print("Rekordi 38/49: I PAPREKUR", flush=True)


if __name__ == "__main__":
    main()
