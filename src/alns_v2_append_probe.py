"""ESA SpOC4: deterministic time-aware append search from verified 38/49.
Diagnostics for ALNS V2; never overwrites the official baseline.
"""
from pathlib import Path
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils_users import TomatoProblem

BASE = ROOT / "results/best_verified_38.json"
REPORT = ROOT / "results/alns_v2_append_probe.json"
CANDIDATE = ROOT / "results/alns_v2_verified_39_candidate.json"

# Select only the 49-tomato instance.
instances = sorted((ROOT / "data").rglob("*.kttsp"), key=lambda x: ("easy" not in x.name.lower(), str(x)))
P = None
for path in instances:
    try:
        obj = TomatoProblem(str(path))
        if getattr(obj, "n_tomatoes", None) == 49:
            P = obj
            print("Instanca:", path.name, flush=True)
            break
    except Exception:
        continue
if P is None:
    raise RuntimeError("Instanca me 49 domate nuk u gjet.")

before = hashlib.sha256(BASE.read_bytes()).hexdigest()
record = json.loads(BASE.read_text(encoding="utf-8"))
assert record["verified"] is True and record["level"] == 38
s = record["frontier"][0]
route, transfers = s["route"], s["transfers"]
assert len(route) == 38 and len(transfers) == 37 and len(set(route)) == 38
assert route[0] == 0 and s["special_used"] == P.n_exceptions == 5

def verify(candidate):
    try:
        r, legs = candidate["route"], candidate["transfers"]
        if r[0] != 0 or len(set(r)) != len(r) or len(legs) != len(r)-1:
            return False
        t, specials = 0.0, 0
        for i, tr in enumerate(legs):
            if (tr["from"], tr["to"]) != (r[i], r[i+1]):
                return False
            d, tof, value = float(tr["start_time"]), float(tr["tof"]), float(tr["dv"])
            if (not all(map(math.isfinite, [d,tof,value])) or d < t-1e-5
                    or tof <= 0 or d+tof > 200+1e-6):
                return False
            actual = float(P.compute_transfer(r[i], r[i+1], d, tof))
            if not math.isfinite(actual) or abs(actual-value) > max(1e-4,abs(actual)*2e-6):
                return False
            if tr["status"] == "NORMAL":
                if actual >= P.dv_threshold-1e-6:
                    return False
            elif tr["status"] == "SPECIAL":
                if actual >= P.exception_threshold-1e-6:
                    return False
                specials += 1
            else:
                return False
            t = d+tof
        return (specials == candidate["special_used"]
                and specials <= P.n_exceptions
                and abs(t-candidate["current_time"]) < 1e-4)
    except Exception:
        return False

assert verify(s), "Rekordi 38/49 nuk kaloi verifikimin!"
missing = sorted(set(range(P.n_tomatoes))-set(route))
print("Baza: 38/49, dita", round(s["current_time"], 6), "SPECIAL", s["special_used"], flush=True)
print("Domatet qe mungojne:", missing, flush=True)

waits = [0,.25,.5,.75,1,1.5,2,3,4,5,6,8,10,12,14,16]
tofs = [.5,.75,1,1.25,1.5,2,2.5,3,4,5,6,8,10,12,15]
start = float(s["current_time"])
last = route[-1]
count = 0
best = None
stats = []
for tomato in missing:
    min_dv = math.inf
    best_time = None
    valid = 0
    for wait in waits:
        departure = round(start+wait, 6)
        for tof in tofs:
            arrival = round(departure+tof, 6)
            if arrival > 200+1e-6:
                continue
            try:
                v = float(P.compute_transfer(last, tomato, departure, tof))
            except Exception:
                continue
            count += 1
            if not math.isfinite(v) or v < 0:
                continue
            if v < min_dv:
                min_dv = v
                best_time = arrival
            # Already all 5 exceptions used: only NORMAL transfers qualify.
            if v >= P.dv_threshold-0.001:
                continue
            valid += 1
            leg = {"from":last, "to":tomato, "start_time":departure,
                   "tof":float(tof), "dv":v, "status":"NORMAL"}
            cand = {"route": route+[tomato],
                    "transfers":transfers+[leg],
                    "special_used":s["special_used"],
                    "current_time":arrival}
            if best is None or arrival < best["current_time"]:
                best = cand
    stats.append({"tomato":tomato, "min_dv":round(min_dv, 4) if math.isfinite(min_dv) else None,
                  "earliest_best_dv_arrival":best_time, "normal_options":valid})

if best is not None:
    if not verify(best):
        raise RuntimeError("Kandidati i ri nuk kaloi verifikimin!")
    payload = {"level":39, "verified":True, "source":"alns_v2_append_probe.py", "frontier":[best]}
    if CANDIDATE.exists():
        assert json.loads(CANDIDATE.read_text(encoding="utf-8")) == payload, "Skedari kandidat ekziston me permbajtje tjeter!"
    else:
        CANDIDATE.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print("*** U GJET 39/49, dita", best["current_time"], flush=True)
else:
    print("Pa rruge te vlefshme me shtim ne fund. Duhet riorganizuar rruga.", flush=True)

report = {"method":"ALNS V2 diagnostic - end insertion", "base_level":38,
          "best_level":39 if best is not None else 38,
          "evaluations":count, "normal_threshold":float(P.dv_threshold),
          "missing":stats, "candidate_verified":best is not None}
if REPORT.exists():
    assert json.loads(REPORT.read_text(encoding="utf-8")) == report, "Raporti ekzistues eshte ndryshe; nuk mbishkruhet!"
else:
    REPORT.write_text(json.dumps(report,indent=2),encoding="utf-8")

assert hashlib.sha256(BASE.read_bytes()).hexdigest() == before, "Baza ka ndryshuar!"
print("Vleresime:", count, "| Kufiri NORMAL:", P.dv_threshold, flush=True)
for row in sorted(stats,key=lambda x:float('inf') if x['min_dv'] is None else x['min_dv'])[:5]:
    print("Domatja",row["tomato"],"min DV",row["min_dv"], "opsione NORMAL",row["normal_options"],flush=True)
print("Raporti:", REPORT, flush=True)
print("Rekordi 38/49: I PAPREKUR", flush=True)
