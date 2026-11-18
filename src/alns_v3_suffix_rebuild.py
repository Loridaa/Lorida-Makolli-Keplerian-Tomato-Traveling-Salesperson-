"""ESA SpOC4 – ALNS V3: adaptive destroy/repair + flexible suffix timing.

Baseline: independently verifies the official 38/49 easy-instance route.
Search: inserts one missing tomato, reorders a neighborhood of 3–9 existing
nodes, and permits time/SPECIAL re-optimization across a longer suffix.

This differs from V2: V2 fixed departures immediately after a 2–5-node window;
V3 may repair up to the route end, allowing downstream SPECIAL redistribution.
A resource-constrained label-setting repair is used inside an adaptive
neighborhood search (rather than Beam Search as the primary optimizer).

Safety: never overwrites baseline, backup, original algorithms, or prior V3
reports; only creates new V3 report and verified candidate if found.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils_users import TomatoProblem

BASE = ROOT / "results/best_verified_38.json"
BACKUP = ROOT / "kaggle_backups/results/best_verified_38.json"
REPORT = ROOT / "results/alns_v3_suffix_rebuild_report.json"
CANDIDATE = ROOT / "results/alns_v3_suffix_verified_39_candidate.json"
SEED = 171
MAX_ATTEMPTS = 75
MAX_EVALUATIONS = 180_000
MAX_LABELS = 18
MIN_TOF = .5
rng = random.Random(SEED)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_problem():
    files = []
    for root in (ROOT / "data", ROOT / "kaggle_backups/data"):
        if root.is_dir():
            files.extend(root.rglob("*.kttsp"))
    files = sorted(set(p.resolve() for p in files),
                   key=lambda p: ("easy" not in p.name.lower(), str(p)))
    errors = []
    for file in files:
        try:
            p = TomatoProblem(str(file))
            if getattr(p, "n_tomatoes", None) == 49:
                print("Instanca:", file.name, "| 49 domate", flush=True)
                return p
            errors.append(f"{file.name}: {getattr(p, 'n_tomatoes', '?')}")
        except Exception as exc:
            errors.append(f"{file.name}: {type(exc).__name__}: {str(exc)[:100]}")
    raise RuntimeError("Instanca 49-domateshe nuk u gjet: " + "; ".join(errors))


def verify(p, s):
    """Recompute each dV using the official physical instance."""
    try:
        r, legs = s["route"], s["transfers"]
        if (not isinstance(r, list) or not r or r[0] != 0
                or len(r) > p.n_tomatoes or len(set(r)) != len(r)
                or any(type(x) is not int or not 0 <= x < p.n_tomatoes for x in r)
                or len(legs) != len(r) - 1):
            return False
        now, used = 0., 0
        for idx, leg in enumerate(legs):
            if leg["from"] != r[idx] or leg["to"] != r[idx+1]:
                return False
            d, tof, dv = map(float, (leg["start_time"], leg["tof"], leg["dv"]))
            if (not all(math.isfinite(x) for x in (d, tof, dv))
                    or d < now - 1e-5 or tof <= 0 or d+tof > 200 + 1e-6):
                return False
            actual = float(p.compute_transfer(r[idx], r[idx+1], d, tof))
            if not math.isfinite(actual) or abs(actual - dv) > max(1e-4, abs(actual)*2e-6):
                return False
            if leg["status"] == "NORMAL":
                if actual >= p.dv_threshold - 1e-6:
                    return False
            elif leg["status"] == "SPECIAL":
                if actual >= p.exception_threshold - 1e-6:
                    return False
                used += 1
            else:
                return False
            now = d + tof
        return (used <= p.n_exceptions and used == s["special_used"]
                and abs(now - s["current_time"]) < 1e-4)
    except Exception:
        return False


class BudgetReached(Exception):
    pass


class Evaluator:
    def __init__(self, p):
        self.p = p
        self.count = 0
        self.cache = {}

    def get(self, a, b, d, tof):
        d, tof = round(float(d), 6), round(float(tof), 6)
        key = (int(a), int(b), d, tof)
        if key not in self.cache:
            if self.count >= MAX_EVALUATIONS:
                raise BudgetReached
            self.count += 1
            try:
                value = float(self.p.compute_transfer(a, b, d, tof))
            except Exception:
                value = math.inf
            self.cache[key] = value if math.isfinite(value) and value >= 0 else math.inf
        return self.cache[key]


def classify(p, dv):
    if dv < p.dv_threshold - 0.001:
        return "NORMAL", 0
    if dv < p.exception_threshold - 0.001:
        return "SPECIAL", 1
    return None


def choose_labels(pool):
    """Bounded resource-constrained label setting; preserve time and SPECIAL diversity."""
    bins = {}
    for s in pool:
        # Quarter-day buckets: don't discard an earlier feasible state.
        key = (s["used"], int(math.floor(s["t"] * 4 + 1e-7)))
        old = bins.get(key)
        if old is None or (s["dv"], s["t"]) < (old["dv"], old["t"]):
            bins[key] = s
    groups = defaultdict(list)
    for state in bins.values():
        groups[state["used"]].append(state)
    if not groups:
        return []
    chosen = []
    keys = set()
    # Each SPECIAL count gets representation; both earliest and lowest-DV.
    per = max(1, MAX_LABELS // (2 * len(groups)))
    for u in sorted(groups):
        group = groups[u]
        for order in (sorted(group, key=lambda s: (s["t"], s["dv"])),
                      sorted(group, key=lambda s: (s["dv"], s["t"]))):
            for s in order[:per]:
                key = (s["used"], s["t"], round(s["dv"], 4))
                if key not in keys:
                    chosen.append(s)
                    keys.add(key)
    if len(chosen) < MAX_LABELS:
        for s in sorted(bins.values(), key=lambda s: (s["used"], s["t"], s["dv"])):
            key = (s["used"], s["t"], round(s["dv"], 4))
            if key not in keys:
                chosen.append(s)
                keys.add(key)
            if len(chosen) >= MAX_LABELS:
                break
    return chosen[:MAX_LABELS]


def repair(p, evaluator, left, targets, start, deadline, cap, original_legs):
    """Reschedule every leg to the selected anchor (possibly route end).

    Exact baseline legs are re-used when temporal feasibility permits; other
    edges are evaluated afresh. Returns (best_state, number_of_legs_reached).
    """
    warm = defaultdict(list)
    for tr in original_legs:
        warm[(tr["from"], tr["to"])].append(tr)
    states = [{"t": start, "used": 0, "dv": 0., "legs": []}]
    progress = 0
    for a, b in zip([left] + list(targets[:-1]), targets):
        pool = []
        old_edges = warm.get((a, b), [])
        for s in states:
            # Reuse verified old transfer exactly, whenever it still fits.
            for tr in old_edges:
                d, tof = float(tr["start_time"]), float(tr["tof"])
                if d + 1e-6 >= s["t"] and d+tof <= deadline + 1e-6:
                    special = int(tr["status"] == "SPECIAL")
                    if s["used"] + special <= cap:
                        pool.append({"t": round(d+tof, 6),
                                     "used": s["used"] + special,
                                     "dv": s["dv"] + float(tr["dv"]),
                                     "legs": s["legs"] + [dict(tr)]})
            departures = {round(s["t"] + w, 6) for w in (0, .25, .5, 1., 2., 4.)}
            tofs = {.5, .75, 1., 1.5, 2., 3., 5., 8.}
            for tr in old_edges:
                old_d = float(tr["start_time"])
                if old_d >= s["t"] - 1e-6:
                    departures.add(round(old_d, 6))
                tofs.add(round(float(tr["tof"]), 6))
            for d in sorted(departures):
                if d < s["t"] - 1e-6 or d + MIN_TOF > deadline + 1e-6:
                    continue
                for tof in sorted(tofs):
                    arr = round(d + tof, 6)
                    if arr > deadline + 1e-6:
                        continue
                    dv = evaluator.get(a, b, d, tof)
                    category = classify(p, dv)
                    if category is None:
                        continue
                    status, extra = category
                    used = s["used"] + extra
                    if used > cap:
                        continue
                    leg = {"from": a, "to": b, "start_time": d,
                           "tof": float(tof), "dv": dv, "status": status}
                    pool.append({"t": arr, "used": used,
                                 "dv": s["dv"] + dv,
                                 "legs": s["legs"] + [leg]})
        if not pool:
            return None, progress
        states = choose_labels(pool)
        if not states:
            return None, progress
        progress += 1
    best = min(states, key=lambda s: (s["used"] * 9 + s["t"], s["dv"]))
    return best, progress


def write_new(path, content):
    if path.exists():
        raise RuntimeError(f"NDAL: {path.name} ekziston; nuk e mbishkruajme.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("U ruajt:", path.name, flush=True)


def rearrange(order, method, local_rng):
    order = list(order)
    if method == "reverse" and len(order) > 1:
        order.reverse()
    elif method == "swap" and len(order) > 1:
        a, b = local_rng.sample(range(len(order)), 2)
        order[a], order[b] = order[b], order[a]
    elif method == "rotate" and len(order) > 1:
        offset = local_rng.randrange(1, len(order))
        order = order[offset:] + order[:offset]
    elif method == "shuffle":
        local_rng.shuffle(order)
    return order


def main():
    if not BASE.is_file() or not BACKUP.is_file():
        raise RuntimeError("Mungon rekordi 38/49 ose backup-i i tij.")
    h = digest(BASE)
    if digest(BACKUP) != h:
        raise RuntimeError("NDAL: backup-i dhe rekordi nuk jane identike.")
    p = load_problem()
    data = json.loads(BASE.read_text(encoding="utf-8"))
    if not (data.get("verified") is True and data.get("level") == 38
            and isinstance(data.get("frontier"), list)
            and len(data["frontier"]) == 1):
        raise RuntimeError("Formati i rekordit eshte i papritur.")
    base = data["frontier"][0]
    if not verify(p, base):
        raise RuntimeError("Rekordi nuk kaloi verifikimin me PyKEP.")
    route, legs = base["route"], base["transfers"]
    n = len(route)
    if n != 38 or len(legs) != 37:
        raise RuntimeError("Pritej rruga 38/49.")
    unused = sorted(set(range(p.n_tomatoes)) - set(route))
    print("BAZA OK:", n, "/49 | dita", round(base["current_time"], 6),
          "| SPECIAL", base["special_used"], flush=True)
    print("Mungojne:", unused, flush=True)
    specials = [idx+1 for idx, tr in enumerate(legs) if tr["status"] == "SPECIAL"]
    print("Pozicionet ku mberrijne transferimet SPECIAL:", specials, flush=True)
    if "--check-only" in sys.argv:
        print("CHECK-ONLY OK. Asnje optimizim apo shkrim skedaresh.", flush=True)
        return
    if REPORT.exists() or CANDIDATE.exists():
        raise RuntimeError("Rezultat V3 ekziston. Ruaje/perdor nje emer versioni tjeter; jo overwrite.")

    evaluator = Evaluator(p)
    operators = ("keep", "swap", "reverse", "rotate", "shuffle")
    weights = {name: 1.0 for name in operators}
    counts = defaultdict(int)
    progress_max = 0
    progress_info = None
    attempts = 0
    attempts_with_full_repair = 0
    found = None
    best_details = None
    t0 = time.monotonic()

    # Include late positions, around every SPECIAL, and diversified earlier cuts.
    starts = []
    for idx in range(max(1, n-12), n):
        starts.append(idx)
    for idx in specials:
        for offset in (-5, -3, -1, 0, 2, 4):
            starts.append(max(1, min(n-1, idx + offset)))
    early = list(range(1, n))
    rng.shuffle(early)
    starts.extend(early)
    starts = list(dict.fromkeys(starts))
    if not starts:
        raise RuntimeError("Asnje pozicion kandidat.")

    try:
        for attempt in range(MAX_ATTEMPTS):
            i = starts[attempt % len(starts)]
            k = (3, 5, 7, 9)[attempt % 4]
            j = min(n, i + k)
            old = route[i:j]
            if not old:
                continue
            op = rng.choices(operators, weights=[weights[x] for x in operators])[0]
            ordered = rearrange(old, op, rng)
            target = unused[(attempt * 7 + i * 3) % len(unused)]
            insert_at = (attempt * 3 + i) % (len(ordered) + 1)
            ordered.insert(insert_at, target)
            # V2 froze the suffix at j; here shift timing 6/12 edges further,
            # or rebuild all the way to the final tomato.
            horizon = (6, 12, n)[attempt % 3]
            end_idx = min(n-1, j + horizon) if j < n else n-1
            extra = list(route[j:end_idx+1]) if j < n else []
            targets = ordered + extra
            fixed_prefix = legs[:i-1]
            fixed_suffix = legs[end_idx:] if end_idx < n-1 else []
            left = route[i-1]
            start = float(fixed_prefix[-1]["start_time"] + fixed_prefix[-1]["tof"]) if fixed_prefix else 0.
            deadline = float(fixed_suffix[0]["start_time"]) if fixed_suffix else 200.
            reserved = sum(tr["status"] == "SPECIAL" for tr in fixed_prefix + fixed_suffix)
            cap = int(p.n_exceptions - reserved)
            if cap < 0 or start + len(targets) * MIN_TOF > deadline + 1e-6:
                counts["skipped_time_or_special"] += 1
                continue
            attempts += 1
            original_window = legs[i-1:end_idx]
            s, reached = repair(p, evaluator, left, targets, start, deadline, cap, original_window)
            rel = reached / max(1, len(targets))
            # Adapt which destroy/reorder operator gets selected; reward near-feasible sequences.
            weights[op] = min(5., max(.35, .96 * weights[op] + .6 * rel))
            counts[op] += 1
            if reached > progress_max:
                progress_max = reached
                progress_info = {"attempt": attempt + 1, "cut": i, "k": len(old),
                                 "repair_end_original_index": end_idx,
                                 "missing_tomato": target, "legs_repaired": reached,
                                 "legs_needed": len(targets)}
            if s is None:
                if (attempt + 1) % 5 == 0:
                    print("Prova", attempt + 1, "| progres", reached, "/", len(targets),
                          "| eval", evaluator.count, flush=True)
                continue
            attempts_with_full_repair += 1
            new_route = route[:i] + ordered + route[j:]
            new_legs = fixed_prefix + s["legs"] + fixed_suffix
            if len(new_route) != 39 or len(new_legs) != 38:
                raise RuntimeError("Gabim ne strukturen e rruges.")
            candidate = {"route": new_route, "transfers": new_legs,
                         "special_used": sum(tr["status"] == "SPECIAL" for tr in new_legs),
                         "current_time": float(new_legs[-1]["start_time"] + new_legs[-1]["tof"])}
            if verify(p, candidate):
                found = candidate
                best_details = {"attempt": attempt+1, "cut": i, "k": len(old),
                                "horizon": horizon, "added_tomato": target,
                                "operator": op}
                print("*** KANDIDAT 39/49 I VERIFIKUAR | dita",
                      round(candidate["current_time"], 6), flush=True)
                break
            else:
                counts["rejected_full_route_verify"] += 1
            if (attempt + 1) % 5 == 0:
                print("Prova", attempt + 1, "| eval", evaluator.count, flush=True)
    except BudgetReached:
        print("U arrit kufiri i vleresimeve orbitale:", MAX_EVALUATIONS, flush=True)

    if digest(BASE) != h or digest(BACKUP) != h:
        raise RuntimeError("NDAL: rekordi baze ose backup-i ka ndryshuar.")
    if found is not None:
        write_new(CANDIDATE, {"level": 39, "verified": True,
                              "source": "alns_v3_suffix_rebuild.py", "frontier": [found]})
    report = {
        "method": "ALNS V3 adaptive neighborhood + downstream resource-constrained suffix retiming",
        "base_level": 38, "best_level": 39 if found else 38,
        "candidate_verified": found is not None,
        "instance": "easy.kttsp (49 tomatoes)", "seed": SEED,
        "attempts": attempts, "max_attempts": MAX_ATTEMPTS,
        "full_temporal_repairs": attempts_with_full_repair,
        "transfer_evaluations": evaluator.count,
        "evaluation_limit": MAX_EVALUATIONS,
        "reached_evaluation_limit": evaluator.count >= MAX_EVALUATIONS,
        "max_legs_repaired": progress_max, "closest_repair": progress_info,
        "operators_tried": dict(counts),
        "adaptive_weights": {key: round(value, 4) for key, value in weights.items()},
        "elapsed_seconds": round(time.monotonic()-t0, 2),
        "candidate_details": best_details,
    }
    write_new(REPORT, report)
    print("=" * 56, flush=True)
    print("ALNS V3:", report["best_level"], "/49 | Kandidat i verifikuar:",
          report["candidate_verified"], flush=True)
    print("Prova:", attempts, "| Transferime te vleresuara:", evaluator.count,
          "| Repair te plota:", attempts_with_full_repair, flush=True)
    print("Rekordi 38/49: I PAPREKUR", flush=True)
    print("Raporti:", REPORT, flush=True)


if __name__ == "__main__":
    main()
