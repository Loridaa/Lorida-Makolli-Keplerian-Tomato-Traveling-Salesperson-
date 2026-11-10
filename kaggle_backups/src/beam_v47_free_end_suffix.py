from pathlib import Path
import sys, os, json, hashlib, time, math, random

ROOT = Path('/kaggle/working/esa-spoc4')
SRC, RES = ROOT / 'src', ROOT / 'results'
sys.path.insert(0, str(SRC))
from utils_users import TomatoProblem

P = TomatoProblem(str(ROOT / 'data/easy.kttsp'))
START = time.monotonic()
RUN_SECONDS = 105
CALL_CAP = 26000
BEAM = 72
MAX_TASKS = 360
CP = RES / 'beam_v47_free_end_suffix_checkpoint.json'
OUT = RES / 'beam_v47_free_end_suffix_level_39.json'
LOCK = RES / 'best_verified_39.json'
OFF = RES / 'best_verified_38.json'
ALTS = RES / 'beam_v43_alternatives_38_level_38.json'


def atomic(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, separators=(',', ':'), allow_nan=False), encoding='utf8')
    tmp.replace(path)


def verify(r, size):
    try:
        nodes, moves = r['route'], r['transfers']
        if len(nodes) != size or len(set(nodes)) != size or nodes[0] != 0 or len(moves) != size-1:
            return False
        last, special = 0., 0
        for i,t in enumerate(moves):
            d, tof = float(t['start_time']), float(t['tof'])
            if (int(t['from']),int(t['to'])) != (nodes[i],nodes[i+1]): return False
            if d < last-1e-7 or tof < 0.001-1e-9: return False
            dv = float(P.compute_transfer(nodes[i],nodes[i+1],d,tof))
            if not math.isfinite(dv) or abs(dv-float(t['dv'])) > 0.05: return False
            if t['status'] == 'NORMAL':
                if dv >= P.dv_threshold: return False
            elif t['status'] == 'SPECIAL':
                special += 1
                if dv >= P.exception_threshold: return False
            else: return False
            last = d + tof
        return (special <= P.n_exceptions and special == r['special_used']
                and last <= P.max_time+1e-8
                and abs(last-float(r['current_time'])) < 1e-4)
    except Exception:
        return False


def get_bases():
    a = json.loads(OFF.read_text())['frontier'][0]
    others = json.loads(ALTS.read_text())['frontier']
    res, seen = [], set()
    for r in [a] + others:
        key = tuple(r['route'])
        if key not in seen and verify(r,38):
            res.append(r)
            seen.add(key)
    assert len(res) >= 2, 'Mungojne dy itinerare 38/49 te verifikuara.'
    return res


def get_tasks(bases):
    tasks = []
    # Free-end suffix is new: unlike earlier LNS, there is no fixed arrival at day 181.
    # The end can use the full 200-day budget; the tail is rebuilt with alternate phases.
    ends = [35,34,32,30,28,26,24]
    targets_order = [26,24,35,6,16,27,4,11,17,32,38,14]
    for bi,base in enumerate(bases):
        missing = set(range(P.n_tomatoes)) - set(base['route'])
        for k in ends:
            if k >= len(base['route'])-1: continue
            spots = [len(base['route']),len(base['route'])-1,
                     len(base['route'])-3,k+2,k+1]
            spots = list(dict.fromkeys(j for j in spots if k+1 <= j <= len(base['route'])))
            for t in targets_order:
                if t not in missing: continue
                for j in spots:
                    tasks.append({'bi':bi,'k':k,'j':j,'target':t})
    # Interleave short and medium suffixes, and the two bases.
    tasks.sort(key=lambda t: (
        [34,32,35,30,28,26,24].index(t['k']),
        targets_order.index(t['target']),
        t['bi'],
        0 if t['j']==38 else 1 if t['j']>=35 else 2,
        t['j']))
    return tasks[:MAX_TASKS]


def keep_diverse(values, cap, anchor):
    # Preserve different orbital phases and SPECIAL budgets.
    # Not just fastest arrival: sometimes waiting gives lower Delta-V later.
    values.sort(key=lambda s:(s['used'], abs(s['day']-anchor)*0.06+s['day']*0.001+s['dv']*0.0001))
    picked, phase_count = [], {}
    for s in values:
        slot = (s['used'], int(s['day']*3))
        if phase_count.get(slot,0)>=2: continue
        phase_count[slot]=phase_count.get(slot,0)+1
        picked.append(s)
        if len(picked)>=cap:break
    return picked


def task_attempt(base, task, seed):
    nodes, moves = base['route'], base['transfers']
    k,j,target = task['k'],task['j'],task['target']
    # Insert a tomato at position j; rebuild every transfer from position k onward.
    route = nodes[:j] + [target] + nodes[j:]
    prefix = moves[:k]
    prev = float(prefix[-1]['start_time'])+float(prefix[-1]['tof']) if prefix else 0.
    cap = int(P.n_exceptions) - sum(t['status']=='SPECIAL' for t in prefix)
    if cap<0: return None,0,[]
    section = route[k:]
    n_edges = len(section)-1
    original = {(int(t['from']),int(t['to'])): t for t in moves[k:]}
    frontier=[{'day':prev,'used':0,'dv':0.,'new':[]}]
    calls, layer_counts = 0, []
    cache = {}
    rng = random.Random(171000 + seed*7)
    for pos in range(n_edges):
        a,b=section[pos:pos+2]
        left=n_edges-pos
        expanded=[]
        old = original.get((a,b))
        for state in frontier:
            if calls>=CALL_CAP or len(expanded)>=8000: break
            earliest=state['day']
            if earliest + .1*left > float(P.max_time):continue
            # Relevant time samples, not the old fixed end time.
            deps=[earliest+v for v in [0,.125,.25,.5,1,2,3,5,8]]
            if old:
                od=float(old['start_time'])
                deps += [od+v for v in [-.5,-.25,0,.25,.5,1.0]]
            # extra late-phase samples as the new final end can reach day 200
            if left<=3:
                deps += [earliest+v for v in [4,7,10,13,16]]
            deps=sorted({round(x,6) for x in deps if x>=earliest-1e-7 and x+.1*left<=P.max_time})
            # Preserve known valid transitions as a way to rejoin the old schedule.
            tof_opts=[.1,.25,.5,.75,1,1.5,2,3,4,5,7,10,13,18]
            if old:
                f=float(old['tof'])
                tof_opts += [f,max(.1,f-.5),f+.5]
            tof_opts += [max(.1,(P.max_time-earliest)/(left+1))]
            tof_opts=sorted(set(round(x,6) for x in tof_opts if x>=.1))
            # Alternate scan direction between tasks to discover different phases.
            if seed%2:deps.reverse()
            for d in deps:
                if calls>=CALL_CAP or len(expanded)>=8000:break
                for tof in tof_opts:
                    arriving=d+tof
                    if arriving+.1*(left-1)>P.max_time+1e-8:continue
                    key=(a,b,d,tof)
                    if key in cache:
                        dv=cache[key]
                    else:
                        try: dv=float(P.compute_transfer(a,b,d,tof))
                        except Exception: dv=float('inf')
                        cache[key]=dv
                        calls+=1
                    if not math.isfinite(dv):continue
                    if dv < P.dv_threshold:
                        status='NORMAL'; used=state['used']
                    elif dv < P.exception_threshold and state['used']<cap:
                        status='SPECIAL';used=state['used']+1
                    else:continue
                    tr={'from':a,'to':b,'start_time':d,'tof':tof,'dv':dv,'status':status}
                    expanded.append({'day':arriving,'used':used,'dv':state['dv']+dv,
                                     'new':state['new']+[tr]})
                    if len(expanded)>=8000:break
        layer_counts.append(len(expanded))
        if not expanded:return None,calls,layer_counts
        ref = float(old['start_time'])+float(old['tof']) if old else min(P.max_time,prev+(pos+1)*3)
        frontier=keep_diverse(expanded,BEAM,ref)
        if calls>=CALL_CAP:return None,calls,layer_counts
    for s in sorted(frontier,key=lambda t:(t['used'],t['day'])):
        child={'route':route,'transfers':prefix+s['new'],
               'special_used':int(P.n_exceptions)-cap+s['used'],
               'current_time':s['day']}
        if verify(child,39):return child,calls,layer_counts
    return None,calls,layer_counts

bases=get_bases()
tasks=get_tasks(bases)
sha=[hashlib.sha256(x.read_bytes()).hexdigest() for x in (OFF,ALTS)]
config={'sources':sha,'v':1,'tasks':len(tasks),'call_cap':CALL_CAP,'beam':BEAM}
if CP.exists():
    ck=json.loads(CP.read_text())
    assert ck['config']==config, 'Checkpoint i nje konfigurimi tjeter.'
else:
    ck={'config':config,'done':{},'solutions':{}}
    atomic(CP,ck)
print('HAPI 171 - FREE-END SUFFIX / RITIMING DERI NE DITEN 200',flush=True)
print('Baza:',len(bases),'| Prova:',len(tasks),'| Te kryera:',len(ck['done']),flush=True)
print('Kohe e lire ne fund:',round(P.max_time-bases[0]['current_time'],3),'dite',flush=True)
for idx,task in enumerate(tasks):
    if ck['solutions'] or time.monotonic()-START >= RUN_SECONDS:break
    if str(idx) in ck['done']:continue
    child,calls,layers=task_attempt(bases[task['bi']],task,idx)
    ck['done'][str(idx)]={'calls':calls,'layers':layers,'found':bool(child)}
    if child:ck['solutions'][str(idx)]=child
    atomic(CP,ck)
    if len(ck['done'])%5==0 or child:
        print('Prova',len(ck['done']),'/',len(tasks),'| baza',task['bi'],
              '| suffix',task['k'],'| pozita',task['j'],'| domatja',task['target'],
              '| DV',calls,'| shtresa',layers[-4:], '| 39=',len(ck['solutions']),flush=True)
    if child:
        print('U GJET ITINERAR 39/49 I VERIFIKUAR!',flush=True)
        break
valid=[s for s in ck['solutions'].values() if verify(s,39)]
if valid:
    atomic(OUT,{'level':39,'verified':True,'source':'beam_v47_free_end_suffix','frontier':valid})
    if LOCK.exists():
        old=json.loads(LOCK.read_text())['frontier'][0]
        assert verify(old,39),'Rekordi 39 ekzistues nuk verifikohet.'
    else:
        atomic(LOCK,{'level':39,'verified':True,'source':OUT.name,'frontier':[valid[0]]})
    print('REKORD 39/49 | Dita:',round(valid[0]['current_time'],5),flush=True)
print('\nPERMBLEDHJA E HAPIT 171',flush=True)
print('Prova:',len(ck['done']),'/',len(tasks),flush=True)
print('Rruge 39 te riverifikuara:',len(valid),flush=True)
print('Kerkimi i pjesshem:',not valid and len(ck['done'])<len(tasks),flush=True)
print('Rekordi:', '39/49' if valid else '38/49',flush=True)
