from pathlib import Path
import sys, os, json, time, math, hashlib, random, itertools

ROOT = Path('/kaggle/working/esa-spoc4')
SRC, RES = ROOT / 'src', ROOT / 'results'
sys.path.insert(0, str(SRC))
from utils_users import TomatoProblem

START = time.monotonic()
RUN_SECONDS = 108
CALL_CAP = 17000
BEAM = 52
MAX_TASKS = 360
CP = RES / 'beam_v46_one_for_two_checkpoint.json'
OUTPUT = RES / 'beam_v46_one_for_two_level_39.json'
LOCK = RES / 'best_verified_39.json'
OFF = RES / 'best_verified_38.json'
ALT = RES / 'beam_v43_alternatives_38_level_38.json'
P = TomatoProblem(str(ROOT / 'data/easy.kttsp'))


def atomic(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, separators=(',', ':'), allow_nan=False), encoding='utf8')
    temp.replace(path)


def verify(s, size):
    try:
        nodes, ts = s['route'], s['transfers']
        if len(nodes) != size or len(ts) != size-1 or len(set(nodes)) != size or nodes[0] != 0:
            return False
        prev, special = 0., 0
        for i, t in enumerate(ts):
            a, b = int(t['from']), int(t['to'])
            d, f = float(t['start_time']), float(t['tof'])
            if (a, b) != (nodes[i], nodes[i+1]) or d < prev - 1e-5 or f <= 0:
                return False
            actual = float(P.compute_transfer(a, b, d, f))
            if not math.isfinite(actual) or abs(actual - float(t['dv'])) > 0.05:
                return False
            if t['status'] == 'NORMAL':
                if actual >= P.dv_threshold: return False
            elif t['status'] == 'SPECIAL':
                if actual >= P.exception_threshold: return False
                special += 1
            else:
                return False
            prev = d + f
        return (special == s['special_used'] and special <= P.n_exceptions and
                prev <= P.max_time+1e-5 and abs(prev - s['current_time']) < 0.05)
    except Exception:
        return False


def get_bases():
    official = json.loads(OFF.read_text())['frontier'][0]
    alternatives = json.loads(ALT.read_text())['frontier']
    out, seen = [], set()
    for s in [official] + alternatives:
        key = tuple(s['route'])
        if key not in seen and verify(s,38):
            seen.add(key)
            out.append(s)
    assert len(out) >= 2, 'Mungojne dy itineraret me 38.'
    return out


def make_tasks(bases):
    all_tasks = []
    for bi, b in enumerate(bases):
        ts, nodes = b['transfers'], b['route']
        missing = sorted(set(range(P.n_tomatoes))-set(nodes))
        durations = []
        prev = 0.
        for t in ts:
            arr = float(t['start_time'])+float(t['tof'])
            durations.append(arr-prev)
            prev = arr
        priorities = []
        for j in range(1, len(nodes)-1):
            local = sum(durations[max(0,j-2):min(len(ts),j+2)])
            special_bonus = 18 if any(t['status']=='SPECIAL' for t in ts[max(0,j-3):min(len(ts),j+3)]) else 0
            # Different from repeating one-tomato insertion: remove an existing node.
            priorities.append((-(local+special_bonus), j))
        ranked = [j for _, j in sorted(priorities)[:20]]
        if 3 not in ranked: ranked.insert(0,3)
        # Prioritize 14/38 since both are known independently feasible.
        favored = 38 if 38 in missing else (14 if 14 in missing else missing[0])
        pairs = list(itertools.combinations(missing, 2))
        pairs.sort(key=lambda p:(favored not in p, 6 not in p, 24 not in p, 26 not in p, p))
        # Distribute all missing pairs across different windows and dropped nodes.
        local_tasks = []
        for rank, (x,y) in enumerate(pairs):
            repeats = 7 if favored in (x,y) else 2
            for r in range(repeats):
                j = ranked[(rank*7+r*3+bi*5) % len(ranked)]
                L = (5,7)[r%2]
                k = max(0, min(j-L//2, len(nodes)-L-1))
                if not k < j < k+L: continue
                order = ((x,y),(y,x))[(rank+r+bi)%2]
                local_tasks.append({'bi':bi,'j':j,'k':k,'L':L,'xy':list(order)})
        all_tasks.append(local_tasks)
    # Interleave both source routes, avoid premature local saturation.
    merged = list(itertools.chain.from_iterable(itertools.zip_longest(*all_tasks)))
    merged = [t for t in merged if t is not None]
    return merged[:MAX_TASKS]


def keep_diverse(candidates, cap):
    # Retain alternate arrival phases and different SPECIAL counts.
    candidates.sort(key=lambda s:(s['used'], s['day'], s['dv']))
    chosen, bins = [], {}
    for s in candidates:
        key = (s['used'], int(s['day']*2.0))
        count = bins.get(key,0)
        if count >= 2: continue
        bins[key] = count+1
        chosen.append(s)
        if len(chosen) >= cap: break
    return chosen


def run_task(base, task, task_index):
    nodes, ts = base['route'], base['transfers']
    k, L, j = task['k'],task['L'],task['j']
    x, y = task['xy']
    inside = nodes[k+1:k+L].copy()
    inside[j-k-1:j-k] = [x,y]
    segment = [nodes[k]]+inside+[nodes[k+L]]
    prefix, suffix = ts[:k], ts[k+L:]
    outside = sum(t['status']=='SPECIAL' for t in prefix+suffix)
    cap = P.n_exceptions-outside
    if cap < 0: return None,0,[]
    t_start = float(prefix[-1]['start_time'])+float(prefix[-1]['tof']) if prefix else 0.
    deadline = float(suffix[0]['start_time']) if suffix else P.max_time
    nlegs = len(segment)-1
    if deadline-t_start <= nlegs*0.5: return None,0,[]
    original_map = {(int(t['from']),int(t['to'])):t for t in ts[k:k+L]}
    frontier = [{'day':t_start,'used':0,'dv':0.,'new':[]}]
    calls, layers = 0, []
    dv_cache = {}
    rng = random.Random(170000+task_index*31)
    for edge in range(nlegs):
        a,b = segment[edge:edge+2]
        remain = nlegs-edge
        expanded = []
        for s in frontier:
            if calls >= CALL_CAP: break
            available = deadline-s['day']
            if available < 0.5*remain-1e-6: continue
            old = original_map.get((a,b))
            waits = [0.,0.25,0.5,1.,2.,3.,5.,8.]
            tofs = [0.5,0.75,1.,1.5,2.,3.,4.,5.,7.,10.,13.]
            tofs.append(max(.5, available/remain))
            if old:
                waits.append(max(0.,float(old['start_time'])-s['day']))
                tofs.append(float(old['tof']))
            rng.shuffle(waits)
            for w in waits:
                dep=s['day']+w
                if dep+0.5*remain>deadline+1e-6:continue
                for tof in tofs:
                    arriving=dep+tof
                    if arriving+0.5*(remain-1)>deadline+1e-6: continue
                    key=(a,b,round(dep,5),round(tof,5))
                    if key in dv_cache:
                        val=dv_cache[key]
                    else:
                        try: val=float(P.compute_transfer(a,b,dep,tof))
                        except Exception: val=float('inf')
                        dv_cache[key]=val
                        calls+=1
                    if not math.isfinite(val):continue
                    if val<P.dv_threshold: kind='NORMAL'; used=s['used']
                    elif val<P.exception_threshold and s['used']<cap:
                        kind='SPECIAL'; used=s['used']+1
                    else:continue
                    t={'from':a,'to':b,'start_time':dep,'tof':tof,'dv':val,'status':kind}
                    expanded.append({'day':arriving,'used':used,'dv':s['dv']+val,'new':s['new']+[t]})
                    if len(expanded)>=8500:break
                if len(expanded)>=8500:break
            if len(expanded)>=8500:break
        layers.append(len(expanded))
        if not expanded: return None,calls,layers
        frontier=keep_diverse(expanded, BEAM)
        if calls >= CALL_CAP:return None,calls,layers
    for s in sorted(frontier,key=lambda z:(z['used'],z['day'],z['dv'])):
        new_route = nodes[:k+1]+inside+nodes[k+L:]
        new_ts = prefix+s['new']+suffix
        child = {'route':new_route, 'transfers':new_ts,
                 'current_time':(float(base['current_time']) if suffix else s['day']),
                 'special_used':outside+s['used']}
        if verify(child,39):return child,calls,layers
    return None,calls,layers

bases=get_bases()
raws=[OFF.read_bytes(),ALT.read_bytes()]
tasks=make_tasks(bases)
config={'official_sha':hashlib.sha256(raws[0]).hexdigest(),
        'alts_sha':hashlib.sha256(raws[1]).hexdigest(),
        'version':1,'call_cap':CALL_CAP,'beam':BEAM,'task_count':len(tasks)}
if CP.exists():
    ck=json.loads(CP.read_text())
    assert ck['config']==config,'Checkpoint me parametra te ndryshem.'
else:
    ck={'config':config,'done':{},'solutions':{}}
    atomic(CP,ck)

print('HAPI 170B - ONE FOR TWO / NEIGHBOR RE-TIMING',flush=True)
print('Baza te verifikuara:',len(bases),'| Prova:',len(tasks),'| Te perfunduara:',len(ck['done']),flush=True)
print('Rekordi 38 ruhet i paprekur.',flush=True)
for i,task in enumerate(tasks):
    if ck['solutions'] or time.monotonic()-START>=RUN_SECONDS:break
    if str(i) in ck['done']:continue
    result,calls,layers=run_task(bases[task['bi']],task,i)
    ck['done'][str(i)]={'calls':calls,'layers':layers,'found':bool(result)}
    if result:ck['solutions'][str(i)]=result
    atomic(CP,ck)
    if len(ck['done'])%5==0 or result:
        print('Prova',len(ck['done']),'/',len(tasks),'| baza',task['bi'],
              '| heq',bases[task['bi']]['route'][task['j']],
              '| shton',task['xy'],'| segment',task['k'],'+',task['L'],
              '| DV',calls,'| layers',layers,'| 39=',len(ck['solutions']),flush=True)
    if result:
        print('U GJET RRUGE 39/49 E VLEFSHME!',flush=True)
        break

valid=[s for s in ck['solutions'].values() if verify(s,39)]
if valid:
    atomic(OUTPUT,{'level':39,'verified':True,'source':'beam_v46_one_for_two','frontier':valid})
    if LOCK.exists():
        existing=json.loads(LOCK.read_text())['frontier'][0]
        assert verify(existing,39),'Rekordi 39 ekzistues nuk verifikohet.'
    else:
        atomic(LOCK,{'level':39,'verified':True,'source':OUTPUT.name,'frontier':[valid[0]]})
    print('REKORD I RI I RIVERIFIKUAR: 39/49! | dita',valid[0]['current_time'],flush=True)

print('\nPERMBLEDHJA E HAPIT 170B',flush=True)
print('Prova:',len(ck['done']),'/',len(tasks),flush=True)
print('Rruge 39 te verifikuara:',len(valid),flush=True)
print('Kerkimi i pjesshem:',not valid and len(ck['done'])<len(tasks),flush=True)
print('Rekordi:', '39/49' if valid else '38/49',flush=True)
