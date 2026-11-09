import json, math, time, hashlib
from pathlib import Path
from functools import lru_cache
from utils_users import TomatoProblem

ROOT = Path('/kaggle/working/esa-spoc4')
R = ROOT / 'results'
CHECK = R / 'beam_v24_lns5_checkpoint.json'
OUT = R / 'beam_v24_lns5_level_32.json'
P = TomatoProblem(str(ROOT / 'data/easy.kttsp'))
START = time.monotonic()
LIMIT = 110.0
WAITS = (0, .25, .5, 1, 2, 4, 8)
TOFS = (.05, .1, .25, .5, 1, 2, 3, 5, 8, 13, 21)

def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, separators=(',', ':')))
    temp.replace(path)

@lru_cache(maxsize=150000)
def dv(a, b, start, tof):
    try:
        x = float(P.compute_transfer(a, b, start, tof))
        return x if math.isfinite(x) and x >= 0 else math.inf
    except Exception:
        return math.inf

def verify(s):
    try:
        route = s['route']; ts = s['transfers']
        if not route or route[0] != 0 or len(route) != len(set(route)) or len(ts) != len(route)-1: return False
        day = 0.0; used = 0
        for i, tr in enumerate(ts):
            d = float(tr['start_time']); f = float(tr['tof'])
            if tr['from'] != route[i] or tr['to'] != route[i+1] or d < day-1e-6 or f <= 0 or d+f > P.max_time+1e-6: return False
            v = dv(route[i], route[i+1], d, f)
            if not math.isfinite(v) or abs(v-float(tr['dv'])) > .1 or v >= P.exception_threshold: return False
            status = 'NORMAL' if v < P.dv_threshold else 'SPECIAL'
            if status != tr['status']: return False
            used += status == 'SPECIAL'; day = d + f
        return used == s['special_used'] and used <= P.n_exceptions and abs(day-s['current_time']) < 1e-5
    except Exception:
        return False

def campaign():
    found = {}
    paths = sorted(set(R.glob('*level_31.json')) | set(R.glob('*retime*checkpoint.json')) | set(R.glob('*reorder*checkpoint.json')))
    def extract(o):
        if isinstance(o, dict):
            if all(k in o for k in ('route','transfers','special_used','current_time')):
                if isinstance(o['route'], list) and len(o['route']) == 31: yield o
            else:
                for v in o.values(): yield from extract(v)
        elif isinstance(o, list):
            for v in o: yield from extract(v)
    for path in paths:
        if path.stat().st_size > 35000000: continue
        try:
            for s in extract(json.loads(path.read_text())):
                if s['special_used'] not in (4,5) or not verify(s): continue
                key = (tuple(s['route']), s['special_used'], round(s['current_time'],2))
                found.setdefault(key, s)
        except (ValueError, TypeError, KeyError, OSError): pass
    if not found: raise RuntimeError('Nuk ka rruge te vlefshme te nivelit 31.')
    states = sorted(found.values(), key=lambda s:(s['special_used'],s['current_time']))
    selected=[]; seen=set()
    for used in (4,5):
        for s in states:
            if s['special_used'] != used: continue
            tail=tuple(s['route'][-8:])
            if tail in seen: continue
            selected.append(s); seen.add(tail)
            if sum(x['special_used']==used for x in selected)>=7: break
    for s in states:
        if len(selected)>=14: break
        if s not in selected: selected.append(s)
    bundles=[]
    for i,s in enumerate(selected):
        t=s['transfers']; opts=[]
        for k in range(len(t)-3):
            early = 0.0 if k==0 else float(t[k-1]['start_time'])+float(t[k-1]['tof'])
            late = float(t[k+4]['start_time']) if k+4<len(t) else float(P.max_time)
            span = late-early
            if span<2.0: continue
            freed=sum(x['status']=='SPECIAL' for x in t[k:k+4])
            opts.append((span+14*freed, k, early, late))
        opts.sort(reverse=True)
        taken=[]
        for score,k,early,late in opts:
            if any(abs(k-z)<4 for z in taken): continue
            taken.append(k)
            missing=[x for x in range(P.n_tomatoes) if x not in s['route']]
            bucket=[(i,k,early,late,new,pos) for new in missing for pos in (1,2,3,4)]
            bucket.sort(key=lambda q:((q[4]*19+i*7+k*13+q[5]*5)%P.n_tomatoes,q[5]))
            bundles.append((score,bucket))
            if len(taken)>=3: break
    bundles.sort(reverse=True,key=lambda x:x[0])
    tasks=[]
    for j in range(80):
        for _, b in bundles:
            if j<len(b): tasks.append(b[j])
            if len(tasks)>=144: break
        if len(tasks)>=144: break
    return dict(version=1,seeds=selected,tasks=tasks,done={},solutions={})

def prune(states, earliest, cap):
    groups={}
    for x in states:
        key=(x['used'], int((x['day']-earliest)/1.25))
        score=x['day']+.003*x['cost']
        if key not in groups or score<groups[key][0]: groups[key]=(score,x)
    arr=[v[1] for v in groups.values()]
    arr.sort(key=lambda x:(x['used']*27+x['day']+.005*x['cost'],x['day']))
    return arr[:cap]

def attempt(seed, task):
    _,k,early,late,new,pos=task
    route=seed['route']; ts=seed['transfers']
    base=route[k:k+5]
    nodes=base[:pos]+[new]+base[pos:]
    fixed_special=seed['special_used']-sum(x['status']=='SPECIAL' for x in ts[k:k+4])
    states=[dict(day=early,used=0,cost=0.0,legs=[])]
    counts=[]; calls=0
    for step in range(5):
        children=[]
        for x in states:
            for wait in WAITS:
                depart=round(x['day']+wait,8)
                for tof in TOFS:
                    arrive=depart+tof
                    if arrive+(4-step)*min(TOFS)>late+1e-8: continue
                    value=dv(nodes[step],nodes[step+1],depart,tof); calls+=1
                    if value>=P.exception_threshold: continue
                    kind='NORMAL' if value<P.dv_threshold else 'SPECIAL'
                    use=x['used']+int(kind=='SPECIAL')
                    if fixed_special+use>P.n_exceptions: continue
                    leg=dict(**{'from':nodes[step],'to':nodes[step+1], 'start_time':depart, 'tof':tof,'dv':value,'status':kind})
                    children.append(dict(day=arrive,used=use,cost=x['cost']+value,legs=x['legs']+[leg]))
        counts.append(len(children))
        if not children: return calls,counts,None
        states=prune(children,early,cap=36 if step<=2 else 24)
    for x in sorted(states,key=lambda q:(q['used'],q['day'])):
        child=dict(route=route[:k+1]+nodes[1:-1]+route[k+4:], transfers=ts[:k]+x['legs']+ts[k+4:],
                   current_time=(x['day'] if k+4==len(route)-1 else seed['current_time']), special_used=fixed_special+x['used'])
        if len(child['route'])==32 and verify(child): return calls,counts,child
    return calls,counts,None

cp=json.loads(CHECK.read_text()) if CHECK.exists() else campaign()
assert cp['version']==1
save(CHECK,cp)
print('HAPI 148 - LARGE NEIGHBORHOOD SEARCH', flush=True)
print('Rruge31:',len(cp['seeds']),'Prova:',len(cp['tasks']),flush=True)
for index,task in enumerate(cp['tasks']):
    if time.monotonic()-START>LIMIT: break
    if str(index) in cp['done']: continue
    calls,counts,solution=attempt(cp['seeds'][task[0]],task)
    cp['done'][str(index)]=dict(calls=calls,stages=counts,success=solution is not None)
    if solution:
        digest=hashlib.sha256(json.dumps(solution,sort_keys=True).encode()).hexdigest()[:20]
        cp['solutions'][digest]=solution
    save(CHECK,cp)
    n=len(cp['done'])
    if n%4==0 or solution:
        print(f'Prova {n}/{len(cp["tasks"])} | DV={calls} | Shtresa te vlefshme={counts} | 32={len(cp["solutions"])}',flush=True)
    if solution: break
valid=[s for s in cp['solutions'].values() if len(s['route'])==32 and verify(s)]
if valid: save(OUT,dict(level=32,verified=True,frontier=valid))
print('\nPERMBLEDHJA E HAPIT 148',flush=True)
print('Prova:',len(cp['done']),'/',len(cp['tasks']),flush=True)
print('Rruge te verifikuara 32:',len(valid),flush=True)
print('Kerkimi i pjesshem:',len(cp['done'])<len(cp['tasks']) and not valid,flush=True)
if valid:
    v=valid[0]
    print('REKORD I RI: 32/49! | SPECIAL:',v['special_used'],'| Dita:',v['current_time'],flush=True)
    print('Rruga:',v['route'],flush=True)
else:
    print('Rekordi mbetet 31/49',flush=True)
