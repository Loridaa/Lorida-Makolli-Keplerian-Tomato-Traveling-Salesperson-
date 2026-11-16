from pathlib import Path
import os, sys, json, math, time, random
from functools import lru_cache
from collections import defaultdict
ROOT=Path('/kaggle/working/esa-spoc4'); R=ROOT/'results'
os.environ['MPLBACKEND']='Agg'; os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(ROOT/'src'))
from utils_users import TomatoProblem
p=next(s.split() for s in (ROOT/'data/easy.kttsp').read_text().splitlines() if s.startswith('p kttsp '))
T0, MIN_F, TMAX, NORMAL, SPECIAL, MAX_SP=map(float,p[2:8]); MAX_SP=int(MAX_SP)
P=TomatoProblem(str(ROOT/'data/easy.kttsp'))
CUTOFF=time.monotonic()+240
RNG=random.Random(1842026)
HARD={4,6,11,16,17,24,26,27,32,35,38}
BEST=R/'hapi184_best.json'; STATUS=R/'hapi184_status.json'

def atomic(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)

@lru_cache(maxsize=350000)
def dv(a,b,dep,tof):
    try:
        x=float(P.compute_transfer(int(a),int(b),float(dep),float(tof)))
        return x if math.isfinite(x) and x>=0 else math.inf
    except (ValueError,OverflowError,RuntimeError,ZeroDivisionError): return math.inf

def state(obj):
    if not isinstance(obj,dict) or not isinstance(obj.get('route'),list) or not isinstance(obj.get('transfers'),list): return None
    route=[int(x) for x in obj['route']]; legs=obj['transfers']
    if not route or route[0]!=0 or len(route)>49 or len(route)!=len(set(route)) or len(legs)!=len(route)-1: return None
    if any(x<0 or x>=49 for x in route): return None
    t=float(legs[-1]['start_time'])+float(legs[-1]['tof']) if legs else T0
    sp=sum(float(x['dv'])>NORMAL+1e-6 for x in legs)
    if t>TMAX+1e-6 or sp>MAX_SP: return None
    return {'route':route,'transfers':legs,'t':t,'sp':sp,'mask':sum(1<<x for x in route)}

def verify(s):
    t=T0; sp=0; r=s['route']
    if len(s['transfers'])!=len(r)-1 or len(set(r))!=len(r):return False
    for i,x in enumerate(s['transfers']):
        a,b=int(x['from']),int(x['to'])
        dep,tof=float(x['start_time']),float(x['tof'])
        if a!=r[i] or b!=r[i+1] or dep<t-1e-6 or tof<MIN_F-1e-6:return False
        v=dv(a,b,round(dep,6),round(tof,6)); saved=float(x['dv'])
        if v>SPECIAL+1e-6 or abs(v-saved)>max(.003,2e-5*abs(saved)):return False
        sp+=int(v>NORMAL+1e-6);t=dep+tof
    return sp==s['sp'] and sp<=MAX_SP and abs(t-s['t'])<1e-4 and t<=TMAX+1e-6

def pack(s):
    return {'route':s['route'],'transfers':s['transfers'],'current_time':s['t'],'special_used':s['sp']}

archives=defaultdict(list)
seen=set()
for f in sorted(R.glob('*.json')):
    if f.stat().st_size>2_000_000:continue
    try: d=json.loads(f.read_text(encoding='utf-8'))
    except (ValueError,UnicodeError):continue
    if isinstance(d,dict):
        entries=(d.get('frontier') if isinstance(d.get('frontier'),list) else [d])
        for v in entries[:100]:
            try:s=state(v)
            except (KeyError,TypeError,ValueError,OverflowError):continue
            if s is None or len(s['route'])<24:continue
            k=(tuple(s['route']),round(s['t'],3),s['sp'])
            if k not in seen: seen.add(k);archives[len(s['route'])].append(s)

# Originalen e ruajme, edhe po te deshtoje kerkimi.
original=json.loads((R/'best_verified_38.json').read_text(encoding='utf-8'))
base=state(original['frontier'][0]); assert base is not None and verify(base)
archives[38].append(base)
best=base
for f in [R/'hapi179_suffix_best.json',R/'hapi180_multi_best.json',BEST]:
    if f.exists():
        try:
            s=state(json.loads(f.read_text(encoding='utf-8')))
            if s is not None and verify(s):
                archives[len(s['route'])].append(s)
                if (len(s['route']),-s['t'])>(len(best['route']),-best['t']):best=s
        except (ValueError,KeyError,TypeError): pass

def score(s):
    # Preferojme kohe me te shkurter, me pak SPECIAL dhe domatet e veshtira.
    return s['t']+8*s['sp']-4*len(HARD.intersection(s['route']))

def choose(items,n):
    items=sorted(items,key=lambda s:score(s)+RNG.random()*1.5)
    out=[]; bins=set()
    for s in items:
        k=(s['route'][-1],s['sp'],int(s['t']/2),len(HARD.intersection(s['route'])))
        if k in bins:continue
        bins.add(k);out.append(s)
        if len(out)>=n:break
    return out

def promote(s):
    global best
    if (len(s['route']),-s['t'])<=(len(best['route']),-best['t']):return
    if not verify(s):return
    best=s
    atomic(BEST,{'verified':True,'method':'hapi184_archived_frontiers',**pack(s),
                 'remaining':sorted(set(range(49))-set(s['route']))})
    print('*** PROGRES',len(s['route']),'/49 | SPECIAL',s['sp'],'| DITA',round(s['t'],5),'***',flush=True)

print('HAPI 184 - ARCHIVED FRONTIERS / HARD TOMATO EARLY',flush=True)
print('Baza',len(best['route']),'/49 | SPECIAL',best['sp'],'| Dita',round(best['t'],5),flush=True)
print('Rruge alternative nga backup (niveli: numri):',
      {k:len(v) for k,v in sorted(archives.items()) if k>=28},flush=True)
# Ndryshimi kryesor: nisemi edhe nga variante ne arkiv me me pak SPECIAL.
starts=[37,35,32,29]
startclock=time.monotonic()
for cut in starts:
    if time.monotonic()>CUTOFF-4 or len(best['route'])==49:break
    stage_limit=min(CUTOFF-3,time.monotonic()+56)
    pool=archives.get(cut,[])[:]
    for s in (best,base):
        if len(s['route'])>=cut:
            p=state({'route':s['route'][:cut],'transfers':s['transfers'][:cut-1]})
            if p:pool.append(p)
    pool=choose(pool,28)
    pool=[s for s in pool if verify(s)]
    if not pool:continue
    depth=cut
    cp_path=R/f'hapi184_checkpoint_{cut}.json'
    if cp_path.exists():
        try:
            ck=json.loads(cp_path.read_text(encoding='utf-8'))
            if ck.get('version')==184 and ck.get('cut')==cut:
                recovered=[state(v) for v in ck.get('frontier',[])]
                recovered=[v for v in recovered if v and verify(v)]
                if recovered and len(recovered[0]['route'])>cut:
                    pool=choose(recovered,36); depth=len(pool[0]['route'])
                    print('CHECKPOINT',cut,'rikthyer ne nivelin',depth,flush=True)
        except (KeyError,ValueError,TypeError):pass
    print('\nSTART',cut,'| niveli',depth,'| rruge',len(pool),'| speciale:',sorted(set(s['sp'] for s in pool)),flush=True)
    while pool and depth<49 and time.monotonic()<stage_limit:
        proposals=[]; timedout=False
        for s in pool:
            if time.monotonic()>stage_limit-1:timedout=True;break
            remaining=[j for j in range(49) if not(s['mask']>>j)&1]
            # Prioritet per 11 domatet e veshtira qe jane lene per fund ne baseline.
            remaining.sort(key=lambda j:(j not in HARD,RNG.random()))
            for j in remaining:
                feasible=[]
                waits=(0,.15,.4,.8,1.5,2.5,4,7,11,17)
                flights=(.1,.25,.5,.75,1,1.5,2,3,4,5,7,9)
                for wait in waits:
                    for tof in flights:
                        if tof<MIN_F-1e-7:continue
                        dep=round(s['t']+wait,6)
                        if dep+tof>TMAX:continue
                        v=dv(s['route'][-1],j,dep,tof)
                        if v<=NORMAL+1e-6 or (s['sp']<MAX_SP and v<=SPECIAL+1e-6):
                            feasible.append((dep+tof,dep,tof,v))
                # Provojme edhe faza jo-diskrete.
                for _ in range(26):
                    room=TMAX-s['t']-MIN_F
                    if room<=0:break
                    wait=RNG.uniform(0,min(room,16))
                    flight=round(RNG.uniform(MIN_F,min(10,TMAX-s['t']-wait)),6) if TMAX-s['t']-wait>=MIN_F else 0
                    if flight<MIN_F:continue
                    dep=round(s['t']+wait,6)
                    v=dv(s['route'][-1],j,dep,flight)
                    if v<=NORMAL+1e-6 or (s['sp']<MAX_SP and v<=SPECIAL+1e-6):
                        feasible.append((dep+flight,dep,flight,v))
                feasible.sort(key=lambda x:(x[3]>NORMAL,x[0]))
                chosen=[]
                for x in feasible:
                    if all(abs(x[0]-y[0])>1.0 for y in chosen): chosen.append(x)
                    if len(chosen)==2:break
                for arrive,dep,tof,v in chosen:
                    flag=int(v>NORMAL+1e-6)
                    leg={'from':s['route'][-1],'to':j,'start_time':dep,
                         'tof':tof,'dv':v,'status':'SPECIAL' if flag else 'NORMAL'}
                    proposals.append({'route':s['route']+[j], 'transfers':s['transfers']+[leg],
                                      't':arrive,'sp':s['sp']+flag,'mask':s['mask']|(1<<j)})
        if timedout:break
        if not proposals:
            print('Niveli',depth,': asnje kalim, provojme start tjeter.',flush=True)
            break
        pool=choose(proposals,36)
        depth+=1
        for s in sorted(pool,key=lambda z:(-len(z['route']),z['t']))[:6]:promote(s)
        atomic(R/f'hapi184_checkpoint_{cut}.json',{'version':184,'cut':cut,
                    'level':depth,'frontier':[pack(s) for s in pool]})
        print('Niveli',depth,'| beam',len(pool),'| best',len(best['route']),
              '| sek',round(time.monotonic()-startclock,1),flush=True)
    if depth>cut:archives[depth].extend(pool)
atomic(STATUS,{'best':len(best['route']),'day':best['t'],'special':best['sp'],
             'seconds':round(time.monotonic()-startclock,2),
             'cache':dv.cache_info()._asdict()})
print('\nPERMBLEDHJE HAPI 184:',len(best['route']),'/49 | SPECIAL',best['sp'],
      '| DITA',round(best['t'],6),flush=True)
print('Origjinali 38/49 nuk u prek. Gjithcka ruhet ne results/hapi184_*.json',flush=True)
