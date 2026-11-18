"""ESA SpOC4 - Hapi 170C: deterministic two-leg insertion diagnostic.

Tries to insert one missing tomato between each pair of consecutive tomatoes
of the verified 38/49 route. Keeps all other transfers and their timestamps
unchanged; computes real delta-v for each new leg using TomatoProblem.
Never overwrites the protected baseline or an existing conflicting result.
"""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils_users import TomatoProblem

BASE = ROOT / "results/best_verified_38.json"
BACKUP = ROOT / "kaggle_backups/results/best_verified_38.json"
REPORT = ROOT / "results/alns_v2_internal_insert_report.json"
OUT = ROOT / "results/alns_v2_internal_verified_39_candidate.json"
EVAL_LIMIT = 60000
MIN_TOF = 0.5


def load_problem():
    paths = sorted((ROOT / "data").rglob("*.kttsp"),
                   key=lambda p: ("easy" not in p.name.lower(), str(p)))
    for path in paths:
        try:
            p = TomatoProblem(str(path))
            if getattr(p, "n_tomatoes", None) == 49:
                print(f"Instanca: {path.name}, 49 domate", flush=True)
                return p
        except Exception:
            pass
    raise RuntimeError("Instanca easy me 49 domate nuk u gjet.")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_json(path, content):
    encoded = json.dumps(content, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != content:
            raise RuntimeError(f"NDAL: skedari ekziston me permbajtje tjeter: {path}")
        print(f"Skedari ekziston dhe eshte identik: {path.name}", flush=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding="utf-8")
        print(f"U ruajt: {path}", flush=True)


P = load_problem()
if not BASE.is_file() or not BACKUP.is_file():
    raise RuntimeError("Mungon rekordi 38/49 ose backup-i i tij")
original_hash = sha(BASE)
if original_hash != sha(BACKUP):
    raise RuntimeError("NDAL: rekordi dhe backup-i nuk jane identike")
record = json.loads(BASE.read_text(encoding="utf-8"))
if record.get("level") != 38 or record.get("verified") is not True:
    raise RuntimeError("Rekordi nuk eshte 38/49 i verifikuar")
if len(record.get("frontier", [])) != 1:
    raise RuntimeError("Frontier i papritur")
s = record["frontier"][0]
route, legs = s["route"], s["transfers"]
if len(route) != 38 or len(legs) != 37 or len(set(route)) != 38 or route[0] != 0:
    raise RuntimeError("Rruga baze eshte e pasakte")


def verify(state):
    try:
        r, tt = state["route"], state["transfers"]
        if not (r[0] == 0 and len(set(r)) == len(r)
                and len(tt) == len(r) - 1 and len(r) <= P.n_tomatoes):
            return False
        if any(type(x) is not int or not 0 <= x < P.n_tomatoes for x in r):
            return False
        previous, specials = 0.0, 0
        for i, tr in enumerate(tt):
            if (tr["from"], tr["to"]) != (r[i], r[i + 1]):
                return False
            d = float(tr["start_time"])
            tof = float(tr["tof"])
            saved_dv = float(tr["dv"])
            if (not all(math.isfinite(v) for v in (d, tof, saved_dv))
                    or d < previous - 1e-5 or tof <= 0
                    or d + tof > 200 + 1e-6):
                return False
            calculated_dv = float(P.compute_transfer(r[i], r[i + 1], d, tof))
            if (not math.isfinite(calculated_dv)
                    or abs(calculated_dv - saved_dv) > max(1e-4, abs(calculated_dv) * 2e-6)):
                return False
            if tr["status"] == "NORMAL":
                if calculated_dv >= P.dv_threshold - 1e-6:
                    return False
            elif tr["status"] == "SPECIAL":
                if calculated_dv >= P.exception_threshold - 1e-6:
                    return False
                specials += 1
            else:
                return False
            previous = d + tof
        return (specials == state["special_used"]
                and specials <= P.n_exceptions
                and abs(previous - state["current_time"]) < 1e-4)
    except Exception:
        return False


if not verify(s):
    raise RuntimeError("NDAL: rekordi 38/49 deshtoi verifikimin e pavarur")
missing = sorted(set(range(P.n_tomatoes)) - set(route))
if len(missing) != 11:
    raise RuntimeError("Numri i domateve te mbetura nuk eshte 11")
print("Baza: 38/49 | Dita:", round(s["current_time"], 6),
      "| SPECIAL:", s["special_used"], flush=True)
print("Mungojne:", missing, flush=True)
print("Metoda: futje midis dy domateve; pjesa tjeter e rruges nuk ndryshon", flush=True)

cached_dv = {}
evaluations = 0
pairs_checked = 0
first_feasible = 0
second_feasible = 0
best_near_second = math.inf
best_near_pair = None
found = None
found_details = None


class BudgetReached(Exception):
    pass


def eval_dv(a, b, dep, tof):
    global evaluations
    key = (a, b, round(float(dep), 6), round(float(tof), 6))
    if key in cached_dv:
        return cached_dv[key]
    if evaluations >= EVAL_LIMIT:
        raise BudgetReached
    evaluations += 1
    try:
        val = float(P.compute_transfer(a, b, key[2], key[3]))
        if not math.isfinite(val) or val < 0:
            val = math.inf
    except Exception:
        val = math.inf
    cached_dv[key] = val
    return val


def classify(value):
    if value < P.dv_threshold - 0.001:
        return "NORMAL", 0
    if value < P.exception_threshold - 0.001:
        return "SPECIAL", 1
    return None, 99


# Largest available windows first. Both orders are deterministic.
positions = []
for i, old in enumerate(legs):
    t0 = 0.0 if i == 0 else float(legs[i-1]["start_time"]) + float(legs[i-1]["tof"])
    limit = float(legs[i+1]["start_time"]) if i+1 < len(legs) else 200.0
    slack = limit - t0
    if slack >= 2 * MIN_TOF - 1e-6:
        positions.append((i, slack, t0, limit))
positions.sort(key=lambda x: (-x[1], x[0]))
print("Pozicione me kohe per dy transferime:", len(positions), "/37", flush=True)

try:
    for ix, (i, slack, t0, limit) in enumerate(positions):
        a, b = route[i], route[i + 1]
        old_leg = legs[i]
        preserved = legs[:i] + legs[i+1:]
        reserved_special = sum(t["status"] == "SPECIAL" for t in preserved)
        local_budget = int(P.n_exceptions - reserved_special)
        if local_budget < 0:
            continue

        dep_options = sorted({round(v, 6) for v in (
            t0, t0 + .25, t0 + .5, t0 + 1., t0 + 2.,
            float(old_leg["start_time"])
        ) if t0 - 1e-6 <= v <= limit - 1.0 + 1e-6})
        tof_options = sorted({.5, .75, 1., 1.25, 1.5, 2., 2.5, 3., 4., 5., 6., 8.,
                              round(float(old_leg["tof"]) / 2, 6)})
        tof_options = [v for v in tof_options if v >= MIN_TOF]

        for x in missing:
            pairs_checked += 1
            first_by_used = defaultdict(list)
            for d1 in dep_options:
                for tof1 in tof_options:
                    arr1 = round(d1 + tof1, 6)
                    if arr1 + MIN_TOF > limit + 1e-6:
                        continue
                    value1 = eval_dv(a, x, d1, tof1)
                    status1, inc1 = classify(value1)
                    if inc1 > local_budget:
                        continue
                    first_feasible += 1
                    first_by_used[inc1].append((arr1, value1, d1, tof1, status1, inc1))

            ranked_first = []
            for used in sorted(first_by_used):
                group = first_by_used[used]
                earliest = sorted(group, key=lambda q: (q[0], q[1]))[:3]
                cheapest = sorted(group, key=lambda q: (q[1], q[0]))[:2]
                seen = set()
                for item in earliest + cheapest:
                    key = (item[2], item[3])
                    if key not in seen:
                        ranked_first.append(item)
                        seen.add(key)

            for arr1, value1, d1, tof1, status1, inc1 in ranked_first:
                d2_options = sorted({round(v, 6) for v in (
                    arr1, arr1 + .25, arr1 + .5, arr1 + 1., arr1 + 2.
                ) if v <= limit - MIN_TOF + 1e-6})
                for d2 in d2_options:
                    for tof2 in tof_options:
                        if d2 + tof2 > limit + 1e-6:
                            continue
                        value2 = eval_dv(x, b, d2, tof2)
                        if value2 < best_near_second:
                            best_near_second = value2
                            best_near_pair = (i, x, a, b)
                        status2, inc2 = classify(value2)
                        if inc1 + inc2 > local_budget:
                            continue
                        second_feasible += 1
                        tr1 = {"from": a, "to": x, "start_time": d1, "tof": tof1,
                               "dv": value1, "status": status1}
                        tr2 = {"from": x, "to": b, "start_time": d2, "tof": tof2,
                               "dv": value2, "status": status2}
                        new_route = route[:i+1] + [x] + route[i+1:]
                        new_legs = legs[:i] + [tr1, tr2] + legs[i+1:]
                        end = (float(new_legs[-1]["start_time"])
                               + float(new_legs[-1]["tof"]))
                        candidate = {
                            "route": new_route,
                            "transfers": new_legs,
                            "special_used": sum(t["status"] == "SPECIAL" for t in new_legs),
                            "current_time": end,
                        }
                        if verify(candidate):
                            found = candidate
                            found_details = {"edge_index": i, "from": a, "insert": x, "to": b,
                                             "old_status": old_leg["status"],
                                             "new_statuses": [status1, status2]}
                            print(f"*** U GJET 39/49! {a} -> {x} -> {b}, "
                                  f"dita {end:.4f}", flush=True)
                            break
                    if found is not None:
                        break
                if found is not None:
                    break
            if found is not None:
                break
        if found is not None:
            break
        if (ix + 1) % 5 == 0 or ix + 1 == len(positions):
            print(f"Pozicione: {ix+1}/{len(positions)} | kombinime: {pairs_checked} "
                  f"| vleresime: {evaluations}/{EVAL_LIMIT}", flush=True)
except BudgetReached:
    print("U arrit kufiri i vleresimeve; testi perfundoi pa ndryshuar bazen.", flush=True)

if sha(BASE) != original_hash or sha(BACKUP) != original_hash:
    raise RuntimeError("NDAL: rekordi baze ose backup-i ka ndryshuar")

report = {
    "method": "ALNS V2 - deterministic one-tomato internal insertion (fixed suffix)",
    "base_level": 38,
    "best_level": 39 if found is not None else 38,
    "candidate_verified": found is not None,
    "instance": "easy.kttsp (49 tomatoes)",
    "special_limit": int(P.n_exceptions),
    "missing_tomatoes": missing,
    "possible_positions": len(positions),
    "pairs_checked": pairs_checked,
    "first_leg_feasible_options": first_feasible,
    "second_leg_feasible_options": second_feasible,
    "transfer_evaluations": evaluations,
    "evaluation_limit": EVAL_LIMIT,
    "stopped_at_limit": evaluations >= EVAL_LIMIT,
    "minimum_second_leg_dv_found": (
        round(best_near_second, 4) if math.isfinite(best_near_second) else None
    ),
    "min_second_leg_context": list(best_near_pair) if best_near_pair else None,
    "candidate_details": found_details,
}

if found is not None:
    payload = {"level": 39, "verified": True,
               "source": "alns_v2_internal_insert.py", "frontier": [found]}
    safe_json(OUT, payload)
safe_json(REPORT, report)
print("=" * 54, flush=True)
print("REZULTATI:", report["best_level"], "/49", flush=True)
print("Kombinime:", pairs_checked, "| Vleresime:", evaluations, flush=True)
print("Kandidate 39/49 te verifikuar:", 1 if found else 0, flush=True)
print("Rekordi 38/49: I PAPREKUR", flush=True)
print("Raporti:", REPORT, flush=True)
