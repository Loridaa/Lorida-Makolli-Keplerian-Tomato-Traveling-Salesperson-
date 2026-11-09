import json, math, time, re, hashlib
from pathlib import Path
from collections import defaultdict
from scipy.optimize import differential_evolution
from utils_users import TomatoProblem

ROOT = Path('/kaggle/working/esa-spoc4')
RES = ROOT / 'results'
CP = RES / 'beam_v23_recombine_checkpoint.json'
OUT = RES / 'beam_v23_recombine_32plus.json'
P = TomatoProblem(str(ROOT / 'data/easy.kttsp'))
START = time.monotonic()
LIMIT = 110


def atomic(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, separators=(',', ':')))
    temp.replace(path)


def dv(a, b, depart, tof):
    try:
        value = float(P.compute_transfer(a, b, float(depart), float(tof)))
        if math.isfinite(value) and value >= 0:
            return value
    except Exception:
        pass
    return math.inf


def verify(state):
    try:
        route, transfers = state['route'], state['transfers']
        if not route or route[0] != 0 or len(route) != len(set(route)):
            return False
        if len(transfers) != len(route) - 1 or len(route) > P.n_tomatoes:
            return False
        day, used = 0., 0
        for i, tr in enumerate(transfers):
            start, tof = float(tr['start_time']), float(tr['tof'])
            if tr['from'] != route[i] or tr['to'] != route[i + 1]:
                return False
            if start < day - 1e-6 or tof <= 0 or start + tof > P.max_time + 1e-6:
                return False
            value = dv(route[i], route[i+1], start, tof)
            if not math.isfinite(value) or abs(value-float(tr['dv'])) > .1:
                return False
            status = 'NORMAL' if value < P.dv_threshold else 'SPECIAL' if value < P.exception_threshold else 'INVALID'
            if tr['status'] != status or status == 'INVALID':
                return False
            used += status == 'SPECIAL'
            day = start + tof
        return used == state['special_used'] and used <= P.n_exceptions and abs(day - state['current_time']) < 1e-5
    except Exception:
        return False


def mask(route):
    m = 0
    for x in route:
        m |= 1 << x
    return m


def collect(obj, found, cap=8000):
    stack = [obj]
    keys = ('route','transfers','current_time','special_used')
    while stack and len(found) < cap:
        item = stack.pop()
        if isinstance(item, dict):
            if all(k in item for k in keys):
                found.append(item)
            else:
                stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)


def source_pool():
    prefixes, endings = defaultdict(list), []
    paths = sorted(RES.glob('*level_*.json'))
    paths.append(RES / 'beam_v20_verified_early_seeds.json')
    for path in paths:
        if not path.exists() or path.stat().st_size > 45_000_000:
            continue
        try:
            data = json.loads(path.read_text())
            states = data.get('frontier', []) if isinstance(data,dict) else []
            for item in states:
                s = item.get('state',item) if isinstance(item, dict) else None
                if not isinstance(s,dict) or not all(k in s for k in ('route','transfers','current_time','special_used')):
                    continue
                n = len(s['route'])
                if 20 <= n <= 30:
                    prefixes[n].append(s)
                elif n == 31:
                    endings.append(s)
        except (OSError,ValueError,TypeError,KeyError):
            continue
    for path in sorted(set(RES.glob('beam_v12*checkpoint.json')) | set(RES.glob('beam_v13_continue_checkpoint.json'))):
        if path.stat().st_size > 45_000_000:
            continue
        try:
            found=[]
            collect(json.loads(path.read_text()),found)
            endings.extend(s for s in found if isinstance(s.get('route'),list) and len(s['route'])==31)
        except (OSError,ValueError,TypeError,KeyError):
            continue
    chosen = []
    for level in range(20,31):
        group=sorted(prefixes[level],key=lambda s:(float(s['current_time'])+.0, s['special_used']))
        byend=defaultdict(int)
        seen=set()
        for s in group:
            sig=(tuple(s['route']),int(s['special_used']),round(float(s['current_time']),2))
            if sig in seen or byend[s['route'][-1]]>=8 or not verify(s):
                continue
            chosen.append(s)
            seen.add(sig)
            byend[s['route'][-1]]+=1
            if len(seen)>=30:
                break
    unique_end={}
    for s in endings:
        try:
            sig=(tuple(s['route']),s['special_used'],round(float(s['current_time']),2))
            if sig not in unique_end and verify(s):
                unique_end[sig]=s
        except Exception:
            continue
    tail_states=sorted(unique_end.values(),key=lambda s:(s['special_used'],s['current_time']))[:220]
    return chosen,tail_states


def merge(a,b,j,move=None):
    if move is None:
        route=a['route']+b['route'][j+1:]
        transfers=a['transfers']+b['transfers'][j:]
    else:
        route=a['route']+b['route'][j:]
        bridge={
            'from':a['route'][-1],'to':b['route'][j],
            'start_time':move['depart'],'tof':move['tof'],
            'dv':move['dv'],'status':move['status']
        }
        transfers=a['transfers']+[bridge]+b['transfers'][j:]
    count=sum(t['status']=='SPECIAL' for t in transfers)
    state={'route':route,'transfers':transfers,'special_used':count,'current_time':b['current_time']}
    return state if len(route)>31 and verify(state) else None


def build_campaign():
    prefixes, tails=source_pool()
    print('Prefikse te riverifikuara:',len(prefixes),flush=True)
    print('Rruge me 31 te riverifikuara:',len(tails),flush=True)
    assert prefixes and tails, 'Nuk u gjeten burime te mjaftueshme'
    candidates=[]
    direct=[]
    for ai,a in enumerate(prefixes):
        amask=mask(a['route'])
        alen=len(a['route'])
        for bi,b in enumerate(tails):
            suffixmask=0
            suffix_special=0
            for j in range(30,-1,-1):
                suffixmask |= 1<<b['route'][j]
                if j < 30:
                    suffix_special+=b['transfers'][j]['status']=='SPECIAL'
                if j<8 or j>=alen or alen+31-j<=31 or j>=len(b['transfers']):
                    continue
                departure=float(b['transfers'][j]['start_time'])
                span=departure-float(a['current_time'])
                if span < .25 or a['special_used']+suffix_special>5:
                    continue
                overlap=amask & suffixmask
                same=a['route'][-1]==b['route'][j]
                if same:
                    if alen+30-j>=32 and overlap==(1<<b['route'][j]):
                        result=merge(a,b,j)
                        if result:
                            direct.append(result)
                elif overlap==0 and span >= .5:
                    available=5-a['special_used']-suffix_special
                    length=alen+31-j
                    score=(-length,-available,-min(span,70),float(b['current_time']))
                    candidates.append((score,ai,bi,j))
    candidates.sort()
    tasks=[]
    seen=set()
    for _,ai,bi,j in candidates:
        a,b=prefixes[ai],tails[bi]
        sig=(tuple(a['route'][-7:]),tuple(b['route'][j:j+9]),
             round(a['current_time'],1),round(b['transfers'][j]['start_time'],1))
        if sig in seen:
            continue
        seen.add(sig)
        tasks.append([ai,bi,j])
        if len(tasks)>=240:
            break
    solutions={}
    for s in direct:
        key=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()[:20]
        solutions[key]=s
    return {'version':1,'prefixes':prefixes,'tails':tails,'tasks':tasks,
            'candidates':len(candidates),'direct':len(direct),'done':{},'solutions':solutions}


def try_bridge(a,b,j):
    deadline=float(b['transfers'][j]['start_time'])
    day=float(a['current_time'])
    span=deadline-day
    available=5-a['special_used']-sum(t['status']=='SPECIAL' for t in b['transfers'][j:])
    threshold=P.exception_threshold if available>0 else P.dv_threshold
    waits=[0,.25,.5,1,2,4,8,12,20,35,span*.2,span*.4,span*.6]
    tofs=[.25,.5,1,2,3,5,8,12,18,25,35,50,span*.2,span*.4,span*.65]
    best=None
    checks=0
    for wait in waits:
        depart=day+wait
        if depart+.25>deadline:continue
        for tof in tofs:
            if tof<=0 or depart+tof>deadline:continue
            value=dv(a['route'][-1],b['route'][j],depart,tof)
            checks+=1
            if not math.isfinite(value):continue
            if best is None or value<best['dv']:
                best={'depart':depart,'tof':tof,'dv':value}
            if value < threshold:
                move={'depart':depart,'tof':tof,'dv':value,
                      'status':'NORMAL' if value<P.dv_threshold else 'SPECIAL'}
                result=merge(a,b,j,move)
                if result:return result,value,checks,False
    refined=False
    if best and best['dv'] < threshold+300 and span>1:
        def objective(x):
            wait,tof=x
            if wait+tof>span:
                return 1e5+1000*(wait+tof-span)
            return dv(a['route'][-1],b['route'][j],day+wait,tof)
        seed=(sum(a['route'])+sum(b['route'])+j*12345)%4294967295
        try:
            fit=differential_evolution(objective,[(0,span-.25),(.25,span)],
                  seed=seed,maxiter=22,popsize=6,polish=False,workers=1)
            wait,tof=map(float,fit.x)
            if wait+tof<=span:
                value=dv(a['route'][-1],b['route'][j],day+wait,tof)
                checks+=1
                if math.isfinite(value):
                    if best is None or value<best['dv']:
                        best={'depart':day+wait,'tof':tof,'dv':value}
                    if value<threshold:
                        move={'depart':day+wait,'tof':tof,'dv':value,
                              'status':'NORMAL' if value<P.dv_threshold else 'SPECIAL'}
                        result=merge(a,b,j,move)
                        if result:return result,value,checks,True
            refined=True
        except Exception:
            pass
    return None,(best['dv'] if best else math.inf),checks,refined


cp=json.loads(CP.read_text()) if CP.exists() else build_campaign()
assert cp['version']==1
atomic(CP,cp)
print('\nHAPI 147 - CROSSOVER I ITINERAREVE',flush=True)
print('Prefikse:',len(cp['prefixes']),'| Rruge31:',len(cp['tails']),flush=True)
print('Bashkime me transferim kandidat:',cp['candidates'],flush=True)
print('Bashkime direkte te vlefshme:',cp['direct'],flush=True)
print('Prova per bridge:',len(cp['tasks']),flush=True)
for i,task in enumerate(cp['tasks']):
    if cp['solutions'] or time.monotonic()-START>LIMIT:break
    if str(i) in cp['done']:continue
    ai,bi,j=task
    solution,bestdv,checks,refined=try_bridge(cp['prefixes'][ai],cp['tails'][bi],j)
    cp['done'][str(i)]={'dv':bestdv if math.isfinite(bestdv) else None,
       'calls':checks,'refined':refined,'success':solution is not None}
    if solution:
        key=hashlib.sha256(json.dumps(solution,sort_keys=True).encode()).hexdigest()[:20]
        cp['solutions'][key]=solution
    atomic(CP,cp)
    if len(cp['done'])%10==0 or solution:
        print(f"Prova {len(cp['done'])}/{len(cp['tasks'])} | DV min {bestdv:.2f} | 32+ {len(cp['solutions'])}",flush=True)

valid=[s for s in cp['solutions'].values() if len(s['route'])>=32 and verify(s)]
valid.sort(key=lambda s:(-len(s['route']),s['special_used'],s['current_time']))
if valid:
    atomic(OUT,{'best_level':len(valid[0]['route']),'verified':True,'frontier':valid})
print('\nPERMBLEDHJA E HAPIT 147',flush=True)
print('Prova bridge:',len(cp['done']),'/',len(cp['tasks']),flush=True)
print('Bashkime direkte:',cp['direct'],flush=True)
print('Rruge te verifikuara 32+:',len(valid),flush=True)
print('Kerkimi i pjesshem:',len(cp['done'])<len(cp['tasks']) and not valid,flush=True)
if valid:
    s=valid[0]
    print('REKORD I RI:',len(s['route']),'/49',flush=True)
    print('SPECIAL:',s['special_used'],'/5 | Dita:',round(s['current_time'],3),flush=True)
    print('Rruga:',s['route'],flush=True)
else:
    print('Rekordi mbetet: 31/49',flush=True)
