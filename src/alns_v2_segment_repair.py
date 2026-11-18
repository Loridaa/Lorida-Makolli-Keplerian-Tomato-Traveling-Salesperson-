"""ESA SpOC4 - 170E: targeted ALNS destroy/repair operator.

Reorders a short section of a verified 38-tomato route, inserts one unvisited
node, and re-optimizes all transfer times in that section. All transfer dV
values come from the real PyKEP TomatoProblem; the rest of the route is fixed.

Runs with a bounded evaluation budget; protects the official 38/49 solution.
Creates a diagnostic report and (only when independently verified) a new
candidate file. Does not modify original algorithms or perform Git operations.
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
REPORT = ROOT / "results/alns_v2_segment_repair_report.json"
CANDIDATE = ROOT / "results/alns_v2_segment_verified_39_candidate.json"
EVAL_LIMIT = 80000
ATTEMPT_LIMIT = 90
WINDOW_LIMIT = 28
BEAM_WIDTH = 20
SEED = 170
MIN_TOF = .5
rng = random.Random(SEED)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path, obj):
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != obj:
            raise RuntimeError(f"NDAL: {path.name} ekziston me permbajtje tjeter.")
        print("Ekziston dhe eshte identik:", path.name, flush=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("U ruajt:", path, flush=True)


def load_problem():
    folder = ROOT / "data"
    if not folder.is_dir():
        raise RuntimeError("Mungon data/ ne projekt.")
    paths = sorted(folder.rglob("*.kttsp"),
                   key=lambda p: ("easy" not in p.name.lower(), str(p)))
    errors = []
    for path in paths:
        try:
            problem = TomatoProblem(str(path))
            if getattr(problem, "n_tomatoes", None) == 49:
                print("Instanca:", path.name, "| Domate: 49", flush=True)
                return problem
            errors.append(f"{path.name}: {getattr(problem, 'n_tomatoes', None)}")
        except Exception as exc:
            errors.append(f"{path.name}: {str(exc)[:100]}")
    raise RuntimeError("Nuk u gjet instanca 49-domateshe. " + "; ".join(errors))


def verify(problem, s):
    try:
        route, legs = s["route"], s["transfers"]
        if (not isinstance(route, list) or not route or route[0] != 0
                or len(route) > problem.n_tomatoes or len(set(route)) != len(route)
                or len(legs) != len(route) - 1
                or any(type(x) is not int or x < 0 or x >= problem.n_tomatoes for x in route)):
            return False
        arrived, special = 0., 0
        for i, leg in enumerate(legs):
            a, b = route[i], route[i + 1]
            if leg.get("from") != a or leg.get("to") != b:
                return False
            d, tof, dv = (float(leg[k]) for k in ("start_time", "tof", "dv"))
            if (not all(math.isfinite(v) for v in (d, tof, dv))
                    or d < arrived - 1e-5 or tof <= 0
                    or d + tof > 200 + 1e-6):
                return False
            actual = float(problem.compute_transfer(a, b, d, tof))
            if not math.isfinite(actual) or abs(actual - dv) > max(1e-4, abs(actual) * 2e-6):
                return False
            if leg["status"] == "NORMAL":
                if actual >= problem.dv_threshold - 1e-6:
                    return False
            elif leg["status"] == "SPECIAL":
                if actual >= problem.exception_threshold - 1e-6:
                    return False
                special += 1
            else:
                return False
            arrived = d + tof
        return (special <= problem.n_exceptions
                and special == s["special_used"]
                and abs(arrived - s["current_time"]) < 1e-4)
    except Exception:
        return False


class BudgetExceeded(Exception):
    pass


def main():
    if not BASE.is_file() or not BACKUP.is_file():
        raise RuntimeError("Rekordi 38/49 ose backup-i mungon!")
    base_hash = sha(BASE)
    if sha(BACKUP) != base_hash:
        raise RuntimeError("NDAL: kopjet e rekordit 38/49 nuk jane identike.")

    record = json.loads(BASE.read_text(encoding="utf-8"))
    if record.get("level") != 38 or record.get("verified") is not True or len(record.get("frontier", [])) != 1:
        raise RuntimeError("Formati i rekordit 38/49 eshte i papritur.")
    base = record["frontier"][0]
    problem = load_problem()
    if not verify(problem, base):
        raise RuntimeError("NDAL: Rruga baze 38/49 nuk u verifikua!")

    route, legs = base["route"], base["transfers"]
    assert len(route) == 38 and len(legs) == 37
    remaining = sorted(set(range(problem.n_tomatoes)) - set(route))
    assert len(remaining) == 11
    print("BAZA: 38/49 | SPECIAL:", base["special_used"],
          "| Dita:", round(base["current_time"], 6), flush=True)
    print("Mungojne:", remaining, flush=True)

    # Unlike fixed two-leg insertion, this modifies a WHOLE local window.
    # [prefix] left -> (reordered old tomatoes + one new) -> anchor [suffix].
    windows = []
    n = len(route)
    for k in (2, 3, 4, 5):
        for i in range(1, n - k + 1):
            j = i + k
            prefix = legs[:i - 1]
            suffix = legs[j:] if j < n else []
            start = (float(prefix[-1]["start_time"]) + float(prefix[-1]["tof"])) if prefix else 0.0
            deadline = float(suffix[0]["start_time"]) if suffix else 200.0
            count_new_legs = k + 1 + int(j < n)
            slack = deadline - start - count_new_legs * MIN_TOF
            if slack < -1e-6:
                continue
            old_segment = legs[i - 1:j]  # all legs to be reoptimized
            special_count = sum(t["status"] == "SPECIAL" for t in old_segment)
            # More credit for windows containing SPECIAL, then available time.
            priority = special_count * 80 + min(45, max(0, slack)) + .02 * start
            windows.append((priority, i, j, start, deadline, special_count))
    windows.sort(key=lambda w: (-w[0], w[1], w[2]))
    # Keep geographic/temporal diversity; a pure SPECIAL-first ranking can
    # spend the entire budget around just one section of the route.
    chosen = []
    selected_keys = set()
    def pick(pool, count):
        if count <= 0:
            return
        added = 0
        for item in pool:
            key = (item[1], item[2])
            if key not in selected_keys:
                chosen.append(item)
                selected_keys.add(key)
                added += 1
                if added >= count:
                    break
    pick([w for w in windows if w[5] > 0], 12)
    pick(sorted(windows, key=lambda w: (-max(0., w[4]-w[3]), w[1])), 8)
    pick(sorted(windows, key=lambda w: (-w[3], -w[0])), 8)
    pick(windows, WINDOW_LIMIT - len(chosen))
    if not chosen:
        raise RuntimeError("Nuk u gjet asnje segment me kohe te mjaftueshme.")
    print("Segmente kandidate:", len(windows),
          "| te perzgjedhura:", len(chosen), flush=True)

    dv_cache = {}
    evaluations = 0
    attempts = 0
    valid_candidates = 0
    best = None
    best_info = None
    start_clock = time.monotonic()

    def evaluate(a, b, dep, tof):
        nonlocal evaluations
        dep, tof = round(float(dep), 6), round(float(tof), 6)
        key = (a, b, dep, tof)
        if key not in dv_cache:
            if evaluations >= EVAL_LIMIT:
                raise BudgetExceeded()
            evaluations += 1
            try:
                value = float(problem.compute_transfer(a, b, dep, tof))
            except Exception:
                value = math.inf
            dv_cache[key] = value if math.isfinite(value) and value >= 0 else math.inf
        return dv_cache[key]

    def repair(left, targets, start, deadline, cap, originals):
        # Beam search is used only as a bounded INNER temporal-repair operator,
        # not as the global algorithm. Keep diversity in time and SPECIAL usage.
        warm = defaultdict(list)
        for leg in originals:
            warm[(leg["from"], leg["to"])].append(leg)
        states = [{"t": start, "used": 0, "dv": 0., "legs": []}]
        for a, b in zip([left] + list(targets[:-1]), targets):
            bucket = {}
            warm_legs = warm.get((a, b), [])
            for s in states:
                departures = {round(s["t"] + dt, 6) for dt in (0., .25, .5, 1., 2., 4.)}
                for old_leg in warm_legs:
                    original_depart = float(old_leg["start_time"])
                    if original_depart >= s["t"] - 1e-6:
                        departures.add(round(original_depart, 6))
                durations = {.5, .75, 1., 1.5, 2., 3., 4., 6., 8.}
                for old_leg in warm_legs:
                    durations.add(round(float(old_leg["tof"]), 6))
                for d in sorted(departures):
                    if d + MIN_TOF > deadline + 1e-6:
                        continue
                    for tof in sorted(durations):
                        arrival = round(d + tof, 6)
                        if arrival > deadline + 1e-6:
                            continue
                        value = evaluate(a, b, d, tof)
                        if value < problem.dv_threshold - .001:
                            status, used = "NORMAL", 0
                        elif value < problem.exception_threshold - .001:
                            status, used = "SPECIAL", 1
                        else:
                            continue
                        new_used = s["used"] + used
                        if new_used > cap:
                            continue
                        candidate = {
                            "t": arrival, "used": new_used,
                            "dv": s["dv"] + value,
                            "legs": s["legs"] + [{"from": a, "to": b,
                                                    "start_time": d, "tof": float(tof),
                                                    "dv": value, "status": status}]
                        }
                        # Retain a fast and a low-cost state for each time/SPECIAL bin.
                        slot = (new_used, int(round(arrival * 4)))
                        old = bucket.get(slot)
                        if old is None or candidate["dv"] < old["dv"]:
                            bucket[slot] = candidate
            if not bucket:
                return None
            by_used = defaultdict(list)
            for state in bucket.values():
                by_used[state["used"]].append(state)
            diverse = []
            per_group = max(2, BEAM_WIDTH // len(by_used))
            for used in sorted(by_used):
                group = by_used[used]
                early = sorted(group, key=lambda s: (s["t"], s["dv"]))[:per_group]
                cheap = sorted(group, key=lambda s: (s["dv"], s["t"]))[:per_group]
                for item in early + cheap:
                    if item not in diverse:
                        diverse.append(item)
            states = sorted(diverse, key=lambda s: (s["used"] * 2 + s["t"], s["dv"]))[:BEAM_WIDTH]
        return min(states, key=lambda s: (s["used"] * 5 + s["t"], s["dv"]))

    # Spread attempts across windows before spending the whole budget in one.
    # Mutate with one extra tomato, and reorder 2-5 existing tomatoes.
    try:
        for round_no in range(4):
            for rank, (_, i, j, start, deadline, special_old) in enumerate(chosen):
                if attempts >= ATTEMPT_LIMIT or best is not None:
                    break
                order = list(route[i:j])
                target = remaining[(rank * 3 + round_no * 5) % len(remaining)]
                if round_no % 2 == 0:
                    rng.shuffle(order)
                else:
                    a, b = rng.sample(range(len(order)), 2)
                    order[a], order[b] = order[b], order[a]
                # Insert missing tomato at a different location each round.
                ins = (rank + round_no * 2) % (len(order) + 1)
                order.insert(ins, target)
                anchor = route[j] if j < n else None
                targets = order + ([anchor] if anchor is not None else [])
                prefix = legs[:i - 1]
                suffix = legs[j:] if anchor is not None else []
                reserved = sum(t["status"] == "SPECIAL" for t in prefix + suffix)
                cap = int(problem.n_exceptions - reserved)
                attempts += 1
                if cap < 0:
                    continue
                segment = repair(route[i - 1], targets, start, deadline, cap, legs[i - 1:j])
                if segment is None:
                    if attempts % 5 == 0:
                        print("Prova", attempts, "| eval", evaluations, "/", EVAL_LIMIT, flush=True)
                    continue
                new_route = route[:i] + order + route[j:]
                new_legs = prefix + segment["legs"] + suffix
                if len(new_legs) != len(new_route) - 1:
                    raise RuntimeError("Gabim ne bashkimin e rruges.")
                end = new_legs[-1]["start_time"] + new_legs[-1]["tof"]
                candidate = {
                    "route": new_route,
                    "transfers": new_legs,
                    "special_used": sum(t["status"] == "SPECIAL" for t in new_legs),
                    "current_time": end,
                }
                if verify(problem, candidate):
                    valid_candidates += 1
                    best = candidate
                    best_info = {"window": [i, j], "added_tomato": target,
                                 "specials_replaced": special_old,
                                 "specials_now": candidate["special_used"]}
                    print("*** U GJET 39/49! | Dita:", round(end, 6),
                          "| Domatja:", target, flush=True)
                    break
                if attempts % 5 == 0:
                    print("Prova", attempts, "| eval", evaluations, "/", EVAL_LIMIT, flush=True)
            if attempts >= ATTEMPT_LIMIT or best is not None:
                break
    except BudgetExceeded:
        print("U arrit kufiri i vleresimeve orbitale.", flush=True)

    if sha(BASE) != base_hash or sha(BACKUP) != base_hash:
        raise RuntimeError("NDAL: rekordi 38/49 ose backup-i ka ndryshuar.")
    if best is not None:
        write_once(CANDIDATE, {"level": 39, "verified": True,
                               "source": "alns_v2_segment_repair.py",
                               "frontier": [best]})
    report = {
        "method": "ALNS V2 targeted multi-node destroy-repair with temporal search",
        "base_level": 38,
        "best_level": 39 if best is not None else 38,
        "candidate_verified": best is not None,
        "instance": "easy.kttsp (49 tomatoes)",
        "seed": SEED,
        "selected_windows": len(chosen),
        "attempts": attempts,
        "transfer_evaluations": evaluations,
        "evaluation_limit": EVAL_LIMIT,
        "stopped_at_limit": evaluations >= EVAL_LIMIT,
        "elapsed_seconds": round(time.monotonic() - start_clock, 2),
        "candidate_details": best_info,
    }
    write_once(REPORT, report)
    print("=" * 52, flush=True)
    print("REZULTATI:", report["best_level"], "/49", flush=True)
    print("Prova:", attempts, "| Vleresime:", evaluations,
          "| Kandidat 39 i verifikuar:", report["candidate_verified"], flush=True)
    print("Rekordi origjinal 38/49: I PAPREKUR", flush=True)
    print("Raporti:", REPORT, flush=True)


if __name__ == "__main__":
    main()
