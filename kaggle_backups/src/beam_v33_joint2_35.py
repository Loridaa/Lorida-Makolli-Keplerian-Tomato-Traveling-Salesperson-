"""ESA SpOC4 Step 156: 33 -> 35 joint two-tomato segment rebuilding.
Reuses independently verified level 33, preserves verified level 34.
Does not overwrite existing best states. Checkpoint after every completed task.
"""
import json, math, time, hashlib
from pathlib import Path
from functools import lru_cache
from collections import defaultdict
from utils_users import TomatoProblem

ROOT = Path('/kaggle/working/esa-spoc4')
R = ROOT / 'results'
SOURCE = R / 'best_verified_33.json'
BASE34 = R / 'best_verified_34.json'
CP = R / 'beam_v33_joint2_35_checkpoint.json'
OUT = R / 'beam_v33_joint2_35_level_35.json'
LOCK = R / 'best_verified_35.json'
P = TomatoProblem(str(ROOT / 'data/easy.kttsp'))
START = time.monotonic()
RUN_SECONDS = 108
MAX_TASKS = 360
TASK_CALL_CAP = 150000
BEAM_WIDTH = 72
WAITS = (0, .25, .5, 1, 2, 3, 5, 8, 12, 18)
TOFS = (.25, .5, .75, 1, 1.5, 2, 3, 4, 5, 7, 10, 14, 20, 28, 38)


def atomic(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, separators=(',', ':')), encoding='utf-8')
    tmp.replace(path)


@lru_cache(maxsize=400000)
def dv(a, b, dep, tof):
    try:
        value = float(P.compute_transfer(int(a), int(b), float(dep), float(tof)))
        return value if math.isfinite(value) and value >= 0 else math.inf
    except Exception:
        return math.inf


def verify(s, expected):
    try:
        route, trs = s['route'], s['transfers']
        if (len(route) != expected or route[0] != 0 or
            len(set(route)) != expected or len(trs) != expected - 1 or
            any(not 0 <= x < P.n_tomatoes for x in route)):
            return False
        day = 0.
        special = 0
        for i, tr in enumerate(trs):
            dep, tof = float(tr['start_time']), float(tr['tof'])
            a, b = route[i], route[i+1]
            if (tr['from'] != a or tr['to'] != b or
                dep < day - 1e-6 or tof <= 0 or dep+tof > P.max_time + 1e-6):
                return False
            value = dv(a,b,dep,tof)
            if not math.isfinite(value) or abs(value-float(tr['dv'])) > .1:
                return False
            if value < P.dv_threshold:
                kind = 'NORMAL'
            elif value < P.exception_threshold:
                kind = 'SPECIAL'
                special += 1
            else:
                return False
            if tr['status'] != kind:
                return False
            day = dep+tof
        return (special == s['special_used'] <= P.n_exceptions and
                abs(day-float(s['current_time'])) < 1e-5)
    except Exception:
        return False


def windows_and_tasks(base):
    trs = base['transfers']
    missing = [t for t in range(P.n_tomatoes) if t not in base['route']]
    wlist = []
    for L in (7,8,9,10,11):
        for k in range(len(trs)-L+1):
            earliest = 0. if k==0 else float(trs[k-1]['start_time'])+float(trs[k-1]['tof'])
            deadline = float(trs[k+L]['start_time']) if k+L<len(trs) else P.max_time
            freed = sum(t['status']=='SPECIAL' for t in trs[k:k+L])
            span = deadline-earliest
            if freed < 1 or span < (L+2)*.25+.5:
                continue
            score = 22*freed + .9*min(85,span) + 1.4*L
            if k<=5: score += 30-3*k
            if k==2: score += 22
            wlist.append({'k':k,'L':L,'earliest':earliest,
                          'deadline':deadline,'freed':freed,'score':score})
    wlist.sort(key=lambda w:(-w['score'],w['k']))
    chosen=[]
    # Force coverage around successful 33->34 insertion (0,39,8,7,21,...).
    for k,L in ((2,9),(2,10),(1,10),(0,11),(3,9),(2,11)):
        w=next((w for w in wlist if w['k']==k and w['L']==L),None)
        if w is not None and w not in chosen: chosen.append(w)
    for w in wlist:
        if any(abs(w['k']-v['k'])<3 and abs(w['L']-v['L'])<2 for v in chosen):
            continue
        chosen.append(w)
        if len(chosen)>=20: break
    for w in wlist:
        if len(chosen)>=20: break
        if w not in chosen: chosen.append(w)

    anchor = 7 if 7 in missing else missing[0]
    others = [t for t in missing if t!=anchor]
    pairs=[(anchor,t) for t in others]
    # Alternative pairs allow a completely different 35-route without tomato 7.
    otherpairs = [(a,b) for ia,a in enumerate(others) for b in others[ia+1:]]
    otherpairs.sort(key=lambda ab: ((ab[0]*19+ab[1]*7)%P.n_tomatoes,ab))
    pairs += otherpairs
    tasks=[]
    # Every wave explores different tomato pairs and placements across windows.
    for wave in range(30):
        for wi,w in enumerate(chosen):
            L = w['L']
            pair = pairs[(wave*7+wi*11) % len(pairs)]
            # Multiple two-insertion layouts, including both close to original tomato 21.
            placements=((0,0),(0,1),(0,L//2),(0,L-1),
                        (1,0),(L//2,0),(L//2,L//2),
                        (L//2,L-1),(L-1,L-1),(1,L-1))
            p1,p2=placements[(wave+wi)%len(placements)]
            tasks.append({'w':wi,'pair':list(pair),'p1':p1,'p2':p2,
                          'swap':(wave//4+wi)%2})
            if len(tasks)>=MAX_TASKS:
                return chosen,tasks
    return chosen,tasks


def construct_nodes(base,w,task):
    k,L=w['k'],w['L']
    interior=list(base['route'][k+1:k+L])
    if task['swap'] and len(interior)>=2:
        pos=min(max(task['p2']-1,0),len(interior)-2)
        interior[pos],interior[pos+1]=interior[pos+1],interior[pos]
    # Position denotes the gap before an ORIGINAL interior node.
    insertions=defaultdict(list)
    insertions[task['p1']].append(task['pair'][0])
    insertions[task['p2']].append(task['pair'][1])
    new=[]
    for i in range(len(interior)+1):
        new.extend(insertions[i])
        if i<len(interior):new.append(interior[i])
    return [base['route'][k]]+new+[base['route'][k+L]]


def diversify(candidates, start_day, cap=BEAM_WIDTH):
    bins=defaultdict(list)
    for s in candidates:
        key=(s['used'],int((s['day']-start_day)/1.0))
        bins[key].append(s)
    for vals in bins.values():
        vals.sort(key=lambda s:s['day']*.09+s['total_dv']*.004)
    by_used=defaultdict(list)
    for (used,_),vals in bins.items():
        by_used[used].extend(vals[:2])
    chosen=[]
    # At least 12 different arrival phases per SPECIAL budget.
    for used,vals in sorted(by_used.items()):
        vals.sort(key=lambda s:s['day'])
        sample=sorted({round((len(vals)-1)*i/11) for i in range(12)})
        chosen.extend(vals[i] for i in sample)
    ids={id(s) for s in chosen}
    rest=[s for vals in by_used.values() for s in vals if id(s) not in ids]
    rest.sort(key=lambda s:6*s['used']+.03*s['day']+.002*s['total_dv'])
    chosen.extend(rest[:max(0,cap-len(chosen))])
    return sorted(chosen[:cap],key=lambda s:(s['used'],s['day']))


def attempt(base,w,task):
    nodes=construct_nodes(base,w,task)
    k,L=w['k'],w['L']
    prefix=base['transfers'][:k]
    suffix=base['transfers'][k+L:]
    outside=sum(t['status']=='SPECIAL' for t in prefix+suffix)
    budget=P.n_exceptions-outside
    earliest, deadline=w['earliest'],w['deadline']
    original=base['transfers'][k:k+L]
    frontier=[{'day':earliest,'used':0,'total_dv':0.,'new':[]}]
    calls=0; layers=[]; gap=math.inf
    for step,(a,b) in enumerate(zip(nodes,nodes[1:])):
        remaining=len(nodes)-step-2
        expanded=[]
        anchor=float(original[min(step,L-1)]['start_time'])
        for s in frontier:
            dep_values={round(s['day']+wait,6) for wait in WAITS}
            dep_values.update(round(x,6) for x in
                (anchor-1,anchor-.25,anchor,anchor+.5,anchor+1.5)
                if x >= s['day']-1e-7)
            for dep in sorted(dep_values):
                if dep<s['day']-1e-7 or dep+(remaining+1)*.25>deadline+1e-7:
                    continue
                for tof in TOFS:
                    arrival=dep+tof
                    if arrival+remaining*.25>deadline+1e-7:
                        continue
                    value=dv(a,b,dep,tof)
                    calls+=1
                    if math.isfinite(value):gap=min(gap,max(0,value-P.exception_threshold))
                    if value>=P.exception_threshold:continue
                    kind='NORMAL' if value<P.dv_threshold else 'SPECIAL'
                    used=s['used']+(kind=='SPECIAL')
                    if used>budget:continue
                    tr={'from':a,'to':b,'start_time':dep,'tof':tof,
                        'dv':value,'status':kind}
                    expanded.append({'day':arrival,'used':used,
                         'total_dv':s['total_dv']+value,
                         'new':s['new']+[tr]})
                    if len(expanded)>=10000:break
                if len(expanded)>=10000 or calls>=TASK_CALL_CAP:break
            if len(expanded)>=10000 or calls>=TASK_CALL_CAP:break
        layers.append(len(expanded))
        if not expanded:return None,calls,layers,gap
        frontier=diversify(expanded,earliest)
        if calls>=TASK_CALL_CAP:return None,calls,layers,gap
    newroute=base['route'][:k+1]+nodes[1:-1]+base['route'][k+L:]
    for s in sorted(frontier,key=lambda s:(s['used'],s['day'])):
        child={'route':newroute,
               'transfers':prefix+s['new']+suffix,
               'special_used':outside+s['used'],
               'current_time':base['current_time'] if suffix else s['day']}
        if verify(child,35):return child,calls,layers,gap
    return None,calls,layers,gap


source_bytes=SOURCE.read_bytes()
base=json.loads(source_bytes.decode('utf-8'))['frontier'][0]
assert verify(base,33),'Rekordi 33 nuk u verifikua.'
assert BASE34.exists() and verify(json.loads(BASE34.read_text())['frontier'][0],34), 'Rekordi 34 nuk u verifikua.'
windows,tasks=windows_and_tasks(base)
assert windows and tasks,'Nuk u gjeten segmente.'
config={'source_sha256':hashlib.sha256(source_bytes).hexdigest(),
        'version':1,'tasks':len(tasks),'calls':TASK_CALL_CAP,'beam':BEAM_WIDTH}
if CP.exists():
    ck=json.loads(CP.read_text())
    assert ck['config']==config,'Checkpoint me parametra te tjere!'
else:
    ck={'config':config,'done':{},'solutions':{}}
    atomic(CP,ck)

print('HAPI 156 - DY DOMATE NE NJE SEGMENT (33 -> 35)',flush=True)
print('Baza e verifikuar: 33/49 | Rekordi i sigurt: 34/49',flush=True)
print('Segmente:',len(windows),'| Prova:',len(tasks),flush=True)
print('Checkpoint:',CP.name,flush=True)
for i,task in enumerate(tasks):
    if ck['solutions'] or time.monotonic()-START>=RUN_SECONDS:break
    if str(i) in ck['done']:continue
    state,calls,layers,gap=attempt(base,windows[task['w']],task)
    ck['done'][str(i)]={'calls':calls,'layers':layers,
         'gap':round(gap,3) if math.isfinite(gap) else None,
         'found':bool(state)}
    if state:ck['solutions'][str(i)]=state
    atomic(CP,ck)
    if len(ck['done'])%5==0 or state:
        w=windows[task['w']]
        print('Prova',len(ck['done']),'/',len(tasks),
              '| Segment',w['k'],'+',w['L'],
              '| pair',task['pair'],'| DV',calls,
              '| layers',layers,'| 35=',len(ck['solutions']),flush=True)
    if state:print('U GJET RRUGE E VLEFSHME 35!',flush=True)

valid=[s for s in ck['solutions'].values() if verify(s,35)]
if valid:
    valid.sort(key=lambda s:(s['special_used'],s['current_time']))
    if OUT.exists():
        old=json.loads(OUT.read_text())
        assert old['frontier']==valid,'Ekziston nje rezultat tjeter; nuk u mbishkrua.'
    else:
        atomic(OUT,{'level':35,'verified':True,'frontier':valid})
    locked={'level':35,'verified':True,'source':OUT.name,'frontier':[valid[0]]}
    if LOCK.exists():
        prev=json.loads(LOCK.read_text())
        assert prev['frontier'][0]==valid[0], 'Ekziston rekord tjeter 35; nuk u mbishkrua.'
    else:
        atomic(LOCK,locked)
    print('REKORDI 35 U RIVERIFIKUA DHE U RUAJT:',LOCK.name,flush=True)
print('\nPERMBLEDHJA E HAPIT 156',flush=True)
print('Prova:',len(ck['done']),'/',len(tasks),flush=True)
print('Rruge 35 te verifikuara:',len(valid),flush=True)
print('Kerkimi i pjesshem:',not valid and len(ck['done'])<len(tasks),flush=True)
if valid:
    s=valid[0]
    print('REKORD I RI: 35/49! | SPECIAL:',s['special_used'],
          '| Dita:',round(s['current_time'],6),flush=True)
    print('Rruga:',s['route'],flush=True)
else:
    print('Rekordi i ruajtur mbetet: 34/49',flush=True)
