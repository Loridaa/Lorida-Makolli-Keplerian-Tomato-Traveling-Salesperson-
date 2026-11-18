"""ALNS V1: protected 38/49 baseline, small time-aware destroy-and-repair search.
Run from repo root using ESA pykep Python interpreter.
"""
from pathlib import Path
from collections import OrderedDict
import copy
import itertools
import json
import math
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils_users import TomatoProblem

BASE = ROOT / "results" / "best_verified_38.json"
OUT = ROOT / "results" / "alns_v1_verified_39_candidate.json"
REPORT = ROOT / "results" / "alns_v1_summary.json"
SEED = 169
TRIALS = 18
EVAL_LIMIT = 12000
WIDTH = 8
rng = random.Random(SEED)


def problem():
    """Load the 49-tomato easy instance."""
    folders = [
        ROOT / "data",
        ROOT / "kaggle_backups" / "data"
    ]

    paths = []
    for folder in folders:
        if folder.is_dir():
            paths.extend(
                p.resolve()
                for p in folder.rglob("*.kttsp")
                if p.is_file()
            )

    paths = list(dict.fromkeys(paths))
    paths.sort(
        key=lambda p: (
            "easy" not in p.name.lower(),
            str(p)
        )
    )

    if not paths:
        raise RuntimeError(
            "Nuk u gjet asnje skedar .kttsp!"
        )

    errors = []

    for path in paths:
        try:
            p = TomatoProblem(str(path))
            n = getattr(p, "n_tomatoes", None)

            if n == 49:
                print(
                    f"Instanca e sakte: {path.name} "
                    f"({n} domate)",
                    flush=True
                )
                return p

            errors.append(f"{path.name}: {n} domate")

        except Exception as exc:
            errors.append(
                f"{path.name}: "
                f"{type(exc).__name__}: {str(exc)[:120]}"
            )

    raise RuntimeError(
        "Instanca 49-domateshe mungon. "
        + "; ".join(errors)
    )

P = problem()


def verify(s):
    try:
        r, tt = s["route"], s["transfers"]
        if (not isinstance(r, list) or len(set(r)) != len(r) or r[0] != 0
                or len(tt) != len(r)-1 or len(r) > P.n_tomatoes):
            return False
        if any(type(x) is not int or x < 0 or x >= P.n_tomatoes for x in r):
            return False
        prev, used = 0.0, 0
        for i, tr in enumerate(tt):
            if int(tr["from"]) != r[i] or int(tr["to"]) != r[i+1]:
                return False
            d, tof, dv = map(float, (tr["start_time"], tr["tof"], tr["dv"]))
            if not all(map(math.isfinite, (d, tof, dv))):
                return False
            if d < prev-1e-5 or tof <= 0 or d+tof > 200+1e-6:
                return False
            actual = float(P.compute_transfer(r[i], r[i+1], d, tof))
            if not math.isfinite(actual) or abs(actual-dv) > max(1e-4, abs(actual)*2e-6):
                return False
            if tr["status"] == "NORMAL":
                if actual >= P.dv_threshold-1e-6:
                    return False
            elif tr["status"] == "SPECIAL":
                if actual >= P.exception_threshold-1e-6:
                    return False
                used += 1
            else:
                return False
            prev = d+tof
        return (used <= P.n_exceptions and used == s["special_used"]
                and abs(prev-s["current_time"]) < 1e-4)
    except Exception:
        return False


record = json.loads(BASE.read_text(encoding="utf-8"))
assert record.get("verified") is True and record.get("level") == 38
assert len(record.get("frontier", [])) == 1
baseline = copy.deepcopy(record["frontier"][0])
if not verify(baseline):
    raise RuntimeError("NDAL: rekordi 38/49 nuk kaloi verifikimin e pavarur!")
print("BAZA OK:", len(baseline["route"]), "domate,", baseline["special_used"],
      "SPECIAL, dite", baseline["current_time"], flush=True)

if "--check-only" in sys.argv:
    print("CHECK-ONLY OK: 38/49 u verifikua.", flush=True)
    sys.exit(0)

# Shared bounded cache, to avoid repeated orbital computations.
cache = OrderedDict()
evaluations = 0

def dv(a, b, d, tof):
    global evaluations
    key = (a, b, round(d, 6), round(tof, 6))
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    if evaluations >= EVAL_LIMIT:
        raise StopIteration
    evaluations += 1
    try:
        v = float(P.compute_transfer(a, b, d, tof))
    except Exception:
        v = float("inf")
    if not math.isfinite(v) or v < 0:
        v = float("inf")
    cache[key] = v
    if len(cache) > 20000:
        cache.popitem(last=False)
    return v


def repair(left, ordered, anchor, time0, cap, deadline, warm_transfers):
    """Beam repair in small ALNS window; untouched suffix remains at fixed dates."""
    targets = list(ordered) + ([anchor] if anchor is not None else [])
    states = [{"t": time0, "used": 0, "cost": 0.0, "legs": []}]
    warm = {(t["from"], t["to"]): t for t in warm_transfers}
    for a, b in zip([left] + targets[:-1], targets):
        buckets = {}
        wt = warm.get((a, b))
        for state in states:
            departures = {round(state["t"]+w, 6) for w in (0,.25,.5,1,2,3)}
            if wt is not None and wt["start_time"] >= state["t"]-1e-6:
                departures.add(round(float(wt["start_time"]),6))
            tofs = [.5,.75,1,1.5,2,3,4,5,8]
            if wt is not None:
                tofs.append(float(wt["tof"]))
            for d in departures:
                if d < state["t"]-1e-6 or d+.5 > deadline+1e-6:
                    continue
                for tof in tofs:
                    arr = round(d+tof,6)
                    if arr > deadline+1e-6:
                        continue
                    value = dv(a,b,d,tof)
                    if value < P.dv_threshold-.001:
                        status, inc = "NORMAL", 0
                    elif value < P.exception_threshold-.001:
                        status, inc = "SPECIAL", 1
                    else:
                        continue
                    if state["used"]+inc > cap:
                        continue
                    leg = {"from":a,"to":b,"start_time":d,"tof":tof,
                           "dv":value,"status":status}
                    nxt = {"t":arr,"used":state["used"]+inc,
                           "cost":state["cost"]+value,
                           "legs":state["legs"]+[leg]}
                    key = (nxt["used"],round(arr*4))
                    old = buckets.get(key)
                    if old is None or nxt["cost"] < old["cost"]:
                        buckets[key] = nxt
        if not buckets:
            return None
        pool = list(buckets.values())
        # Preserve both the earliest and the lowest-dV states in each SPECIAL group.
        groups = {}
        for s in pool:
            groups.setdefault(s["used"], []).append(s)
        chosen = []
        per = max(1,WIDTH//len(groups))
        for group in groups.values():
            by_time = sorted(group,key=lambda s:(s["t"],s["cost"]))
            by_cost = sorted(group,key=lambda s:(s["cost"],s["t"]))
            seen=set()
            for s in itertools.chain(by_time[:per],by_cost[:per]):
                k = (s["used"],s["t"])
                if k not in seen:
                    chosen.append(s); seen.add(k)
        states = sorted(chosen,key=lambda s:(s["t"]+s["cost"]*.015, s["used"]))[:WIDTH]
    return min(states,key=lambda s:(s["cost"]+s["t"]*.5))


def propose(s, k, add):
    route, tt = s["route"], s["transfers"]
    # Bias to suffix, where insertion may also exploit up to 200 days.
    i = rng.randrange(max(1,len(route)-12),len(route)) if rng.random()<.65 else rng.randrange(1,len(route))
    j = min(len(route)-1, i+k-1)
    removed = route[i:j+1]
    ordered = rng.sample(removed, len(removed))
    missing = sorted(set(range(P.n_tomatoes))-set(route))
    if add and missing:
        ordered.insert(rng.randrange(len(ordered)+1),rng.choice(missing))
    anchor = route[j+1] if j+1 < len(route) else None
    suffix = tt[j+1:] if anchor is not None else []
    deadline = float(suffix[0]["start_time"]) if suffix else 200.0
    prefix = tt[:i-1]
    start_t = prefix[-1]["start_time"]+prefix[-1]["tof"] if prefix else 0.0
    reserved = sum(tr["status"]=="SPECIAL" for tr in prefix+suffix)
    remaining = P.n_exceptions-reserved
    if remaining < 0:
        return None
    segment = repair(route[i-1],ordered,anchor,start_t,remaining,deadline,tt[i-1:j+1])
    if segment is None:
        return None
    new_route = route[:i]+ordered+route[j+1:]
    new_legs = prefix+segment["legs"]+suffix
    if len(new_legs) != len(new_route)-1:
        return None
    end_t = new_legs[-1]["start_time"]+new_legs[-1]["tof"]
    candidate = {
        "route":new_route,
        "transfers":new_legs,
        "special_used":sum(tr["status"]=="SPECIAL" for tr in new_legs),
        "current_time":end_t,
    }
    return candidate if verify(candidate) else None


def score(s):
    return sum(tr["dv"] for tr in s["transfers"]) + .5*s["current_time"]


current = copy.deepcopy(baseline)
best = copy.deepcopy(baseline)
weights = {1:1.0,2:1.0,3:1.0}
accepted = 0
feasible = 0
start = time.monotonic()
for iteration in range(TRIALS):
    op = rng.choices(list(weights),weights=list(weights.values()))[0]
    add = rng.random() < .75
    temp = max(5.0, 40*(.94**iteration))
    try:
        cand = propose(current,op,add)
    except StopIteration:
        print("Kufiri i vleresimeve u arrit.", flush=True)
        break
    if cand is None:
        weights[op] = max(.3, weights[op]*.98)
        print(f"Prova {iteration+1}/{TRIALS}: pa rruge te vlefshme; eval={evaluations}",flush=True)
        continue
    feasible += 1
    diff = score(cand)-score(current)
    move = len(cand["route"]) > len(current["route"]) or (
        len(cand["route"]) == len(current["route"]) and
        (diff <= 0 or rng.random()<math.exp(-min(700, max(0,diff))/temp)))
    if move:
        current = cand
        accepted += 1
        weights[op] += 1.5 if len(cand["route"])>len(best["route"]) else .2
    if len(cand["route"]) > len(best["route"]) or (
        len(cand["route"]) == len(best["route"]) and score(cand)<score(best)):
        best = copy.deepcopy(cand)
        print(f"*** Kandidat: {len(best['route'])}/49, dita {best['current_time']:.3f}",flush=True)
    print(f"Prova {iteration+1}/{TRIALS}: {len(cand['route'])}/49, "
          f"pranuar={move}, eval={evaluations}",flush=True)
    if len(best["route"]) == P.n_tomatoes:
        break

summary = {"algorithm":"ALNS destroy-repair + SA V1", "seed":SEED,
           "trials":TRIALS,"feasible_proposals":feasible,"accepted":accepted,
           "transfer_evaluations":evaluations,"base_level":38,
           "best_level":len(best["route"]),"best_time":best["current_time"],
           "best_verified":verify(best),"elapsed_seconds":round(time.monotonic()-start,2)}
REPORT.parent.mkdir(parents=True,exist_ok=True)
# Save only diagnostic metadata: never alter official record or original algorithms.
REPORT.write_text(json.dumps(summary,indent=2),encoding="utf-8")
if len(best["route"])>=39 and verify(best):
    if OUT.exists():
        raise RuntimeError("Kandidati ekziston; nuk lejohet mbishkrimi!")
    payload = {"level":len(best["route"]),"verified":True,
               "source":"alns_keplerian_v1.py", "frontier":[best]}
    OUT.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print("KANDIDAT I RI I VERIFIKUAR:",OUT,flush=True)
else:
    print("Nuk u arrit ende 39/49. Rekordi 38/49 mbetet i pandryshuar.",flush=True)
print("PERMBLEDHJE:",json.dumps(summary,indent=2),flush=True)
