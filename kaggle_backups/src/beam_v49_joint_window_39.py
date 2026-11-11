"""Hapi 174: target-first multi-edge joint retiming (no repeated old campaigns)."""
from pathlib import Path
import json, math, time, hashlib, os
import numpy as np
from scipy.optimize import differential_evolution
from utils_users import TomatoProblem

ROOT = Path(os.environ.get('ESA_SPOC4_ROOT', '/kaggle/working/esa-spoc4'))
RES = ROOT / 'results'
SAFE = RES / 'best_verified_38.json'
ALTS = RES / 'beam_v43_alternatives_38_level_38.json'
PREV = RES / 'beam_v48_target_first_checkpoint.json'
CP = RES / 'beam_v49_joint_window_checkpoint.json'
OUT = RES / 'best_verified_39.json'
START = time.monotonic()
RUN_SECONDS = int(os.environ.get('ESA_RUN_SECONDS', '105'))
MAX_TASKS = 440
MAXITER = 11
POPSIZE = 4
P = TomatoProblem(str(ROOT / 'data/easy.kttsp'))
p_line = next(row.split() for row in (ROOT / 'data/easy.kttsp').read_text().splitlines() if row.lstrip().startswith('p '))
T0, MIN_TOF = float(p_line[2]), max(float(p_line[3]), 0.00001)
assert P.n_tomatoes == 49 and abs(P.max_time - 200) < 1e-4


def save(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, separators=(',', ':')))
    temp.replace(path)


def dv(a, b, start, tof):
    try:
        v = float(P.compute_transfer(int(a), int(b), float(start), float(tof)))
        return v if math.isfinite(v) and v >= 0 else math.inf
    except Exception:
        return math.inf


def status(v):
    return 'NORMAL' if v < P.dv_threshold else 'SPECIAL' if v < P.exception_threshold else 'INVALID'


def verify(s):
    try:
        route, legs = s['route'], s['transfers']
        if (len(route) != 39 and len(route) != 38) or len(legs) != len(route)-1:
            return False
        if route[0] != 0 or len(set(route)) != len(route) or any(not 0 <= int(x) < 49 for x in route):
            return False
        day, special = T0, 0
        for i, tr in enumerate(legs):
            dep, tof = float(tr['start_time']), float(tr['tof'])
            if (int(tr['from']) != route[i] or int(tr['to']) != route[i+1]
                    or dep < day-1e-6 or tof < MIN_TOF-1e-7 or dep+tof > P.max_time+1e-6):
                return False
            value = dv(route[i], route[i+1], dep, tof)
            label = status(value)
            if label == 'INVALID' or label != tr['status'] or abs(value-float(tr['dv'])) > 0.1:
                return False
            special += label == 'SPECIAL'
            day = dep+tof
        return special <= P.n_exceptions and special == s['special_used'] and abs(day-s['current_time']) < 0.001
    except Exception:
        return False


original = json.loads(SAFE.read_text())['frontier'][0]
alternatives = json.loads(ALTS.read_text())['frontier']
bases, seen = [], set()
for state in [original] + alternatives:
    sig = tuple(state['route'])
    if sig not in seen and verify(state):
        bases.append(state)
        seen.add(sig)
assert bases and len(bases[0]['route']) == 38

priority_target, priority_edge = None, 7
if PREV.exists():
    old = json.loads(PREV.read_text()).get('best_task') or {}
    priority_target, priority_edge = old.get('tomato'), old.get('edge', 7)

fingerprint = hashlib.sha256(SAFE.read_bytes()+ALTS.read_bytes()).hexdigest()
if CP.exists():
    cp = json.loads(CP.read_text())
    assert cp['fingerprint'] == fingerprint, 'Rekordi ndryshoi: checkpoint-i nuk perputhet.'
else:
    cp = {'fingerprint': fingerprint, 'next': 0, 'best_ratio': 999., 'best_task': None, 'solved': False}

# Choose windows, anchored by preceding arrival and following departure.
windows = []
for bi, base in enumerate(bases):
    legs = base['transfers']
    for width in (3, 4):
        for i in range(len(legs)-width+1):
            left = T0 if i == 0 else float(legs[i-1]['start_time'])+float(legs[i-1]['tof'])
            right = (float(legs[i+width]['start_time']) if i+width < len(legs) else float(P.max_time))
            room = right-left-(width+1)*MIN_TOF
            if room <= 0.1:
                continue
            specials = sum(t['status']=='SPECIAL' for t in legs[i:i+width])
            center = i+width/2
            dist = abs(center-(priority_edge+.5))
            score = -0.85*dist + 0.08*min(right-left, 90) + .45*specials + .4*(width==4)
            windows.append((score,bi,i,width,specials,right-left))
windows.sort(reverse=True)
# Ensure both bases have opportunities, while favoring windows near Hapi 172's closest insertion.
selected, per_base = [], {i:0 for i in range(len(bases))}
for entry in windows:
    _,bi,i,width,sp,span = entry
    if per_base[bi] >= 16:
        continue
    selected.append((bi,i,width,sp,span))
    per_base[bi] += 1
    if len(selected) >= 32:
        break
assert selected

# Round-robin over target tomatoes (not 32 identical edge trials in sequence).
targets = []
for base in bases:
    targets.append(sorted(set(range(49))-set(base['route']),
                          key=lambda x:(x != priority_target, x not in (16,24,35,11), x)))

tasks = []
for t_rank in range(max(map(len,targets))):
    for bi,i,width,sp,span in selected:
        if t_rank >= len(targets[bi]):
            continue
        target = targets[bi][t_rank]
        # First evaluate two distinct insertion slots; other slots follow on a later cycle.
        positions = sorted(range(1,width+1),key=lambda pos:abs((i+pos-1)-priority_edge))
        for pos in positions:
            tasks.append((bi,i,width,pos,target,False))
            if len(tasks) >= MAX_TASKS:
                break
        if len(tasks) >= MAX_TASKS:
            break
    if len(tasks) >= MAX_TASKS:
        break


def optimize(task, serial):
    bi,i,width,pos,target,_swap = task
    base = bases[bi]
    nodes = base['route'][i:i+width+1]
    nodes = nodes[:pos] + [target] + nodes[pos:]
    n = width+1
    originals = base['transfers'][i:i+width]
    left = T0 if i == 0 else float(base['transfers'][i-1]['start_time'])+float(base['transfers'][i-1]['tof'])
    right = float(base['transfers'][i+width]['start_time']) if i+width < len(base['transfers']) else float(P.max_time)
    available_sp = sum(t['status']=='SPECIAL' for t in originals)
    room = right-left-n*MIN_TOF
    if room <= 0:
        return math.inf, None
    dimension = 2*n+1
    best = {'ratio': math.inf, 'schedule': None, 'vals': None}

    # Encode original nearby time schedule, splitting one flight to insert the target.
    timings = []
    for k, tr in enumerate(originals):
        dep, tof = float(tr['start_time']), float(tr['tof'])
        if k == pos-1:
            f1 = tof/2
            timings.extend([(dep,f1),(dep+f1,tof-f1)])
        else:
            timings.append((dep,tof))
    weights, cursor = [], left
    for d,f in timings:
        weights.extend([max(1e-4,d-cursor),max(1e-4,f-MIN_TOF)])
        cursor = d+f
    weights.append(max(1e-4,right-cursor))
    seeds = np.log(np.array(weights))
    seeds -= np.mean(seeds)
    seeds = np.clip(seeds,-4.5,4.5)
    pop = POPSIZE*dimension
    rng = np.random.default_rng(174000+serial)
    init = rng.uniform(-4.5,4.5,size=(pop,dimension))
    init[0] = seeds
    for k in range(1,min(9,pop)):
        init[k] = np.clip(seeds+rng.normal(0,1.05,size=dimension),-5,5)

    def decode(x):
        z = np.exp(np.asarray(x,dtype=float)-max(x))
        slices = room*z/sum(z)
        schedule = []
        clock = left
        for j in range(n):
            dep = clock+float(slices[2*j])
            tof = MIN_TOF+float(slices[2*j+1])
            schedule.append((dep,tof))
            clock = dep+tof
        return schedule

    def objective(x):
        sched = decode(x)
        vals = [dv(nodes[j],nodes[j+1],*sched[j]) for j in range(n)]
        ordered = sorted(vals, reverse=True)
        ratio = max(([v/P.exception_threshold for v in ordered[:available_sp]] +
                     [v/P.dv_threshold for v in ordered[available_sp:]]),default=math.inf)
        if ratio < best['ratio']:
            best['ratio'],best['schedule'],best['vals'] = ratio,sched,vals
        return ratio if math.isfinite(ratio) else 1e7

    def early_stop(_x, convergence=None):
        return best['ratio'] < .99999

    differential_evolution(objective,[(-5,5)]*dimension,
        init=init,popsize=POPSIZE,maxiter=MAXITER,seed=174000+serial,
        polish=False,workers=1,tol=.01,callback=early_stop)
    if 1 <= best['ratio'] < 1.35:
        differential_evolution(objective,[(-5,5)]*dimension,
            maxiter=18,popsize=5,seed=274000+serial,
            polish=False,workers=1,tol=.005,callback=early_stop)
    if best['ratio'] >= 1 or best['schedule'] is None:
        return best['ratio'],None
    inserts=[]
    for j,((dep,tof),v) in enumerate(zip(best['schedule'],best['vals'])):
        inserts.append({'from':nodes[j],'to':nodes[j+1],
                        'start_time':dep,'tof':tof,'dv':v,'status':status(v)})
    full_route = base['route'][:i]+nodes+base['route'][i+width+1:]
    full_legs = base['transfers'][:i]+inserts+base['transfers'][i+width:]
    end = (base['current_time'] if i+width < len(base['transfers'])
           else inserts[-1]['start_time']+inserts[-1]['tof'])
    candidate = {'route':full_route,'transfers':full_legs,
                 'special_used':sum(t['status']=='SPECIAL' for t in full_legs),
                 'current_time':end}
    return best['ratio'],candidate if len(full_route)==39 and verify(candidate) else None


print('HAPI 174 - JOINT WINDOW RETIMING (3/4 LIDHJE)',flush=True)
print(f'Baza {len(bases)} | Detyra {len(tasks)} | Te perfunduara {cp["next"]}',flush=True)
print(f'Fokus nga Hapi 172: edge {priority_edge}, tomato {priority_target}',flush=True)
print('Rekordi 38 i mbrojtur. Metoda 171/172 nuk perseritet.',flush=True)
while cp['next'] < len(tasks) and not cp['solved'] and time.monotonic()-START < RUN_SECONDS:
    idx = cp['next']
    task = tasks[idx]
    r, solution = optimize(task,idx)
    if r < cp['best_ratio']:
        cp['best_ratio'] = round(float(r),9)
        cp['best_task'] = {'base':task[0], 'edge':task[1], 'window':task[2],
                           'insert_pos':task[3], 'tomato':task[4]}
    cp['next'] += 1
    if solution:
        old = json.loads(OUT.read_text())['frontier'][0] if OUT.exists() else None
        assert old is None or verify(old), 'Ekzistuesi 39 nuk eshte valid.'
        if old is None or solution['current_time'] < old['current_time']:
            save(OUT,{'frontier':[solution],'source':'beam_v49_joint_window',
                      'verification':'pykep_full_transfer_check'})
        cp['solved'] = True
        save(CP,cp)
        print(f'SUKSES 39/49 | DITA {solution["current_time"]:.6f} | SPECIAL {solution["special_used"]}/5',flush=True)
        break
    if cp['next'] % 3 == 0:
        save(CP,cp)
        print(f'Prova {cp["next"]}/{len(tasks)} | baza {task[0]} | '
              f'window {task[1]}:{task[1]+task[2]} | domatja {task[4]} | '
              f'raporti {r:.3f} | best {cp["best_ratio"]:.3f}',flush=True)
save(CP,cp)
print('\nPERMBLEDHJA E HAPIT 174',flush=True)
print(f'Prova: {cp["next"]}/{len(tasks)}',flush=True)
print(f'Raporti me i mire: {cp["best_ratio"]}',flush=True)
print(f'39/49: {"PO" if cp["solved"] else "JO"}',flush=True)
print(f'Kerkimi i pjesshem: {cp["next"]<len(tasks) and not cp["solved"]}',flush=True)
