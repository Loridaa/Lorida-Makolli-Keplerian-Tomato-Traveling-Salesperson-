"""ESA SpOC4: exact-on-sampled-edges subset DP for route reorder + insertion.
Safeguard: verified 38 baseline is never overwritten.
Run with Python 3.11 / PyKEP via Kaggle notebook launcher.
"""
from pathlib import Path
import json, math, hashlib, time, bisect, os
from utils_users import TomatoProblem

ROOT = Path(os.environ.get('ESA_SPOC4_ROOT', '/kaggle/working/esa-spoc4'))
RES = ROOT / 'results'
SAFE = RES / 'best_verified_38.json'
ALT = RES / 'beam_v43_alternatives_38_level_38.json'
OUT = RES / 'best_verified_39.json'
FREE = RES / 'hapi175b_38_four_specials.json'
CP = RES / 'hapi175b_subset_dp_checkpoint.json'
P = TomatoProblem(str(ROOT / 'data/easy.kttsp'))
params = next(s.split() for s in (ROOT/'data/easy.kttsp').read_text().splitlines()
              if s.lstrip().startswith('p '))
T0, MIN_TOF = float(params[2]), max(float(params[3]), 1e-5)
assert P.n_tomatoes == 49 and abs(P.max_time - 200) < 0.01
START = time.monotonic()
SECONDS = int(os.environ.get('ESA_RUN_SECONDS', '105'))
STEP = 0.50
WIN_START, WIN_END = 21, 26  # 5 -> 47 -> 36 -> 31 -> 37 -> 20


def save(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, separators=(',', ':')))
    temp.replace(path)


def dv(a, b, dep, tof):
    try:
        value = float(P.compute_transfer(int(a), int(b), float(dep), float(tof)))
        return value if math.isfinite(value) and value >= 0 else math.inf
    except Exception:
        return math.inf


def kind(value):
    return ('NORMAL' if value < P.dv_threshold else
            'SPECIAL' if value < P.exception_threshold else 'INVALID')


def verify(s, wanted):
    try:
        route, legs = s['route'], s['transfers']
        if (len(route) != wanted or route[0] != 0 or len(set(route)) != wanted
                or len(legs) != wanted-1 or any(not 0 <= int(x) < 49 for x in route)):
            return False
        day, special = T0, 0
        for ix, item in enumerate(legs):
            dep, tof = float(item['start_time']), float(item['tof'])
            if (item['from'] != route[ix] or item['to'] != route[ix+1]
                    or dep < day-1e-5 or tof < MIN_TOF-1e-6
                    or dep+tof > P.max_time+1e-5):
                return False
            value = dv(route[ix], route[ix+1], dep, tof)
            cat = kind(value)
            if (cat == 'INVALID' or cat != item['status']
                    or abs(value-float(item['dv'])) > 0.1):
                return False
            special += cat == 'SPECIAL'
            day = dep+tof
        return (special == s['special_used'] and special <= P.n_exceptions
                and abs(day-s['current_time']) < 1e-3)
    except Exception:
        return False


base = json.loads(SAFE.read_text())['frontier'][0]
assert verify(base, 38), 'Rekordi 38 nuk kaloi riverifikimin.'
L, R = WIN_START, WIN_END
route, legs = base['route'], base['transfers']
assert route[L:R+1] == [5,47,36,31,37,20], 'Segmenti nuk perputhet!'
left = T0 if L == 0 else float(legs[L-1]['start_time'])+float(legs[L-1]['tof'])
right = float(legs[R]['start_time']) if R < len(legs) else P.max_time
budget = sum(t['status'] == 'SPECIAL' for t in legs[L:R])
assert budget == 1 and right > left
middle = list(route[L+1:R])
missing = sorted(set(range(49))-set(route))
# Orient candidates using which were promising in recent experiments; search every missing target.
priority = [35, 24, 26, 17, 11, 16, 27, 32, 4, 6, 38]
targets = [x for x in priority if x in missing] + [x for x in missing if x not in priority]

# Event grid is anchored to the actual preceding-leg arrival, plus the exact
# original departures so we never exclude the known feasible route just by gridding.
days = [left+k*STEP for k in range(int((right-left)/STEP)+1)]
for tr in legs[L:R]:
    d = float(tr['start_time'])
    days += [d, d-.25, d+.25]
deps = sorted(set(round(d, 7) for d in days if left-1e-7 <= d <= right-MIN_TOF))
tofs = sorted(set(round(t, 7) for t in
    ([MIN_TOF, 0.4, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3,
      3.5, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 22, 26]
     + [float(t['tof']) for t in legs[L:R]])
    if MIN_TOF-1e-7 <= t <= right-left))

# Each edge table records the earliest arrival for normal/special flights
# after any given minimum departure time. Earlier arrivals dominate later
# arrivals for the same visited set, endpoint and SPECIAL count (waiting is allowed).
edge_cache = {}
dv_calls = 0


def build_edge(a, b):
    global dv_calls
    key = (a,b)
    if key in edge_cache:
        return edge_cache[key]
    best_normal = [None]*len(deps)
    best_special = [None]*len(deps)
    for di, departure in enumerate(deps):
        for tof in tofs:
            arrival = departure+tof
            if arrival > right+1e-7:
                break
            value = dv(a,b,departure,tof)
            dv_calls += 1
            k = kind(value)
            if k == 'INVALID':
                continue
            ev = (arrival, a, b, departure, tof, value, k)
            arr = best_normal if k == 'NORMAL' else best_special
            if arr[di] is None or arrival < arr[di][0]:
                arr[di] = ev
    for arr in [best_normal, best_special]:
        best = None
        for i in range(len(deps)-1,-1,-1):
            if arr[i] is not None and (best is None or arr[i][0]<best[0]):
                best = arr[i]
            arr[i] = best
    edge_cache[key] = (best_normal, best_special)
    return edge_cache[key]


def earliest_edge(a,b,arrival_time,k):
    idx = bisect.bisect_left(deps, arrival_time-1e-7)
    if idx >= len(deps):
        return None
    return build_edge(a,b)[k][idx]


def find_path(internal, limit):
    n = len(internal)
    first, last = route[L], route[R]
    origin = (0, -1, 0)
    dp = {origin: left}
    parents = {}
    sizes = [0]*(n+1)
    solution = None
    for level in range(n+1):
        active = [(state,arr) for state,arr in dp.items()
                  if state[0].bit_count() == level]
        sizes[level] = len(active)
        for state, ready in active:
            mask, loc, used = state
            src = first if loc == -1 else internal[loc]
            if level == n:
                for k in (0,1):
                    if used+k > limit:
                        continue
                    e = earliest_edge(src,last,ready,k)
                    if e is not None and (solution is None or e[0] < solution[0]):
                        solution = (e[0], state, e)
                continue
            for next_i, dest in enumerate(internal):
                if mask & (1<<next_i):
                    continue
                for k in (0,1):
                    if used+k > limit:
                        continue
                    e = earliest_edge(src,dest,ready,k)
                    if e is None:
                        continue
                    nxt = (mask|(1<<next_i), next_i, used+k)
                    if e[0] < dp.get(nxt, math.inf)-1e-8:
                        dp[nxt] = e[0]
                        parents[nxt] = (state,e)
    if solution is None:
        return None, sizes
    _,state,last_event = solution
    events = [last_event]
    while state != origin:
        prior,e = parents[state]
        events.append(e)
        state = prior
    events.reverse()
    return events, sizes


def make_record(events):
    new_legs = [{'from':int(e[1]),'to':int(e[2]),'start_time':float(e[3]),
                 'tof':float(e[4]),'dv':float(e[5]),'status':e[6]}
                for e in events]
    segment_nodes = [events[0][1]]+[e[2] for e in events]
    full_route = route[:L]+segment_nodes+route[R+1:]
    full_legs = legs[:L]+new_legs+legs[R:]
    day = base['current_time'] if R < len(legs) else events[-1][0]
    return {'route':full_route,'transfers':full_legs,
            'special_used':sum(t['status']=='SPECIAL' for t in full_legs),
            'current_time':day}


sig = hashlib.sha256(SAFE.read_bytes()+ALT.read_bytes()
                    +str((L,R,STEP,tofs)).encode()).hexdigest()
if CP.exists():
    cp = json.loads(CP.read_text())
    assert cp['fingerprint'] == sig, 'Checkpoint nga konfigurim tjeter.'
else:
    cp = {'fingerprint':sig,'next_target':0,'completed':False,
          'max_states':0,'attempts':[],'special_recovered':False,'free_checked':False}

print('HAPI 175B - TIME-EXPANDED SUBSET DP / REORDER + 39',flush=True)
print(f'Segmenti: {route[L:R+1]} | {right-left:.2f} dite',flush=True)
print(f'Grid: {STEP} dite | {len(deps)} nisje | {len(tofs)} kohe fluturimi',flush=True)
print(f'Domate te munguar: {targets} | Te kryera: {cp["next_target"]}/{len(targets)}',flush=True)
print('Rekordi 38/49 nuk mbishkruhet.',flush=True)

# A zero-SPECIAL restructuring of this window can release a SPECIAL transfer
# for a future insertion anywhere else in the route.
if not cp.get('free_checked') and not FREE.exists():
    events, levels = find_path(middle, 0)
    if events:
        candidate = make_record(events)
        if verify(candidate,38) and candidate['special_used'] < 5:
            save(FREE, {'frontier':[candidate],
                        'source':'hapi175b_subset_dp_freed_special'})
            cp['special_recovered'] = True
            print('SUCCES: 38/49 me',candidate['special_used'],
                  'SPECIAL! U lirua buxheti per nje transferim.',flush=True)
    cp['free_checked'] = True
    save(CP,cp)
    print('Kontrolli i lirimit SPECIAL u krye.',flush=True)

while (cp['next_target'] < len(targets) and not cp.get('solved')
       and time.monotonic()-START < SECONDS):
    target = targets[cp['next_target']]
    events,levels = find_path(middle+[target],budget)
    report = {'tomato':target,'layers':levels,
              'reached':max(i for i,x in enumerate(levels) if x>0)}
    if events:
        candidate = make_record(events)
        if verify(candidate,39):
            if OUT.exists():
                old = json.loads(OUT.read_text())['frontier'][0]
                assert verify(old,39), 'Skedari ekzistues 39 nuk eshte valid!'
            else:
                old = None
            if old is None or candidate['current_time'] < old['current_time']:
                save(OUT,{'frontier':[candidate], 'source':'hapi175b_subset_dp',
                          'verification':'full_pykep_37plus1'})
            cp['completed'] = True
            cp['solved'] = True
            print(f'!!! SUKSES 39/49, DOMATJA {target}, '
                  f'SPECIAL {candidate["special_used"]}/5, '
                  f'DITA {candidate["current_time"]:.5f} !!!',flush=True)
    cp['attempts'].append(report)
    cp['max_states'] = max(cp['max_states'],max(levels))
    cp['next_target'] += 1
    save(CP,cp)
    print(f'Kandidati {cp["next_target"]}/{len(targets)} | domatja {target} | '
          f'nivelet {levels} | DV llogaritje {dv_calls}',flush=True)
    if cp.get('solved'):
        break
cp['completed'] = cp.get('solved',False) or cp['next_target'] == len(targets)
save(CP,cp)
print('\nPERMBLEDHJA E HAPIT 175B',flush=True)
print(f'Prova te kryera: {cp["next_target"]}/{len(targets)}',flush=True)
print(f'39/49: {"PO" if cp.get("solved") or OUT.exists() else "JO"}',flush=True)
print(f'38/49 me SPECIAL te liruar: {"PO" if FREE.exists() else "JO"}',flush=True)
print(f'DV ne kete ekzekutim: {dv_calls}',flush=True)
print(f'Kerkimi i pjesshem: {not cp["completed"]}',flush=True)
