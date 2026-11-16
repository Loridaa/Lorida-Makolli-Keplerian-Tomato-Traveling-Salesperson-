# HAPI 185 - gap bridge scan. Intended for existing Kaggle notebook.
from pathlib import Path
from collections import OrderedDict
import os,sys,json,math,time,random,hashlib

ROOT=Path('/kaggle/working/esa-spoc4'); R=ROOT/'results'
sys.path.insert(0,str(ROOT/'src'))
from utils_users import TomatoProblem

header=next(x.split() for x in (ROOT/'data/easy.kttsp').read_text().splitlines() if x.startswith('p kttsp '))
T0,MIN_TOF,END,NORMAL,SPECIAL,MAX_SP=map(float,header[2:8]); MAX_SP=int(MAX_SP)
P=TomatoProblem(str(ROOT/'data/easy.kttsp'))
OUT=R/'hapi185_bridge_best.json'; CHECK=R/'hapi185_bridge_checkpoint.json'
LIMIT=time.monotonic()+215
cache=OrderedDict(); calls=0

def dv(a,b,dep,tof):
    global calls
    key=(int(a),int(b),round(float(dep),6),round(float(tof),6))
    if key in cache:
        cache.move_to_end(key);return cache[key]
    try:
        val=float(P.compute_transfer(*key))
        if not math.isfinite(val) or val<0: val=math.inf
    except (ValueError,RuntimeError,OverflowError,ZeroDivisionError):val=math.inf
    calls+=1
    cache[key]=val
    if len(cache)>180000:cache.popitem(last=False)
    return val

def atomic(path,obj):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
    os.replace(tmp,path)

def getroute(obj):
    if not isinstance(obj,dict):return None
    if isinstance(obj.get('frontier'),list):
        if not obj['frontier']:return None
        obj=obj['frontier'][0]
    if not isinstance(obj.get('route'),list) or not isinstance(obj.get('transfers'),list):return None
    route=obj['route'];legs=obj['transfers']
    if not route or len(set(route))!=len(route) or route[0]!=0 or len(route)!=len(legs)+1:return None
    if any(not isinstance(x,int) or x<0 or x>=49 for x in route):return None
    end=float(legs[-1]['start_time'])+float(legs[-1]['tof']) if legs else T0
    return {'route':list(route),'transfers':legs,'time':end}

def verify(x):
    route,legs=x['route'],x['transfers']
    if len(legs)!=len(route)-1 or len(route)!=len(set(route)):return False
    time_=T0; special_count=0
    for i,tr in enumerate(legs):
        a,b=int(tr['from']),int(tr['to']); dep=float(tr['start_time']);tof=float(tr['tof']); saved=float(tr['dv'])
        if (a,b)!=(route[i],route[i+1]) or dep<time_-1e-5 or tof<MIN_TOF-1e-6:return False
        actual=dv(a,b,dep,tof)
        if actual>SPECIAL+1e-6 or abs(actual-saved)>max(.003,2e-5*abs(saved)):return False
        special_count+=int(actual>NORMAL+1e-6);time_=dep+tof
    return special_count<=MAX_SP and time_<=END+1e-6

def record_for(s):
    sp=sum(float(tr['dv'])>NORMAL+1e-6 for tr in s['transfers'])
    return {'verified':True,'route':s['route'],'transfers':s['transfers'],
            'special_used':sp,'current_time':s['time'],
            'missing':sorted(set(range(49))-set(s['route'])),
            'method':'hapi185_gap_bridge'}

# Find distinct 38-target paths; keep one previous best and diverse archival routes.
proposed=[]
for name in ['hapi180_multi_best.json','hapi179_suffix_best.json',
             'hapi184_best.json','best_verified_38.json','hapi185_bridge_best.json']:
    f=R/name
    if f.exists():
        try:proposed.append(getroute(json.loads(f.read_text(encoding='utf8'))))
        except (OSError,ValueError,TypeError):pass
for f in sorted(R.glob('*.json')):
    if f.stat().st_size>400000:continue
    try:o=json.loads(f.read_text(encoding='utf8'))
    except (OSError,ValueError,TypeError):continue
    if isinstance(o,dict) and isinstance(o.get('frontier'),list):
        for item in o['frontier'][:20]:proposed.append(getroute(item))
    else:proposed.append(getroute(o))

unique={}
for s in proposed:
    if s and len(s['route'])==38:
        key=(tuple(s['route']),round(s['time'],3))
        unique[key]=s
proposed=sorted(unique.values(),key=lambda s:s['time'])
seeds=[]; endings=set()
for s in proposed:
    ending=tuple(s['route'][-4:])
    if ending not in endings and verify(s):
        endings.add(ending);seeds.append(s)
        if len(seeds)>=8:break
assert seeds,'Nuk u gjet itinerar 38/49 i verifikueshem'

best=None
if OUT.exists():
    try:
        prior=getroute(json.loads(OUT.read_text(encoding='utf8')))
        if prior and len(prior['route'])>=39 and verify(prior):best=prior
    except (OSError,ValueError,TypeError):pass

print('HAPI 185 - INSERTION BRIDGE + TIMING REPAIR',flush=True)
print('Itinerare burimore te verifikuara:',len(seeds),'| Rekordi origjinal: I MBROJTUR',flush=True)
print('SPECIAL maksimale:',MAX_SP,'| Afati i misionit:',END,flush=True)

# A task is one gap and one missing target on a particular source route.
tasks=[]
for n,s in enumerate(seeds):
    legs=s['transfers'];route=s['route'];missing=set(range(49))-set(route)
    sc=[int(float(t['dv'])>NORMAL+1e-6) for t in legs]
    # First search gaps where a SPECIAL can be replaced; then gaps with more free time.
    gaps=sorted(range(len(legs)),key=lambda i:(-sc[i],-
        ((float(legs[i+1]['start_time']) if i+1<len(legs) else END)-
         (float(legs[i-1]['start_time'])+float(legs[i-1]['tof']) if i else T0))))
    for i in gaps:
        for m in sorted(missing):tasks.append((n,i,m))

seed_signatures=[hashlib.sha256(json.dumps(s['transfers'],sort_keys=True).encode()).hexdigest() for s in seeds]
completed=set()
if CHECK.exists():
    try:
        saved=json.loads(CHECK.read_text(encoding='utf8'))
        if saved.get('version')==185 and saved.get('seed_signatures')==seed_signatures:
            completed=set(saved.get('completed',[]))
            print('Rikthehen',len(completed),'prova nga checkpoint.',flush=True)
    except (OSError,ValueError,TypeError):pass

def try_bridge(s,i,m,rng):
    global best
    legs=s['transfers'];route=s['route'];old=legs[i]
    a,b=route[i],route[i+1]
    prev=float(legs[i-1]['start_time'])+float(legs[i-1]['tof']) if i else T0
    nextdep=float(legs[i+1]['start_time']) if i+1<len(legs) else END
    old_end=s['time']
    shift_limit=max(0.,min(8.,END-old_end)) if i+1<len(legs) else 0.
    cap=min(END,nextdep+shift_limit)
    max_for_first=cap-2*MIN_TOF
    if max_for_first<=prev:return
    oldsp=int(float(old['dv'])>NORMAL+1e-6)
    totalsp=sum(float(z['dv'])>NORMAL+1e-6 for z in legs)
    available=MAX_SP-totalsp+oldsp
    if available<0:return

    flights=[MIN_TOF,.25,.5,.75,1.,1.5,2.,3.,4.,5.,6.,8.,10.,12.]
    flights=sorted(set(f for f in flights if f>=MIN_TOF and f<=cap-prev-MIN_TOF))
    first=[]
    # Focus on phase diversity, not just the original departure instant.
    for k in range(78):
        if time.monotonic()>LIMIT-3:return
        dep=prev+(max_for_first-prev)*(rng.random()**2)
        tof=(rng.choice(flights) if flights and k%3==0 else
             MIN_TOF+(cap-MIN_TOF-dep)*rng.random()**1.6)
        tof=round(tof,6);dep=round(dep,6)
        if tof<MIN_TOF or dep+tof>cap-MIN_TOF:continue
        val=dv(a,m,dep,tof)
        xsp=int(val>NORMAL+1e-6)
        if val<=SPECIAL+1e-6 and xsp<=available:
            first.append((dep+tof,dep,tof,val,xsp))
    # Prioritize varied arrival phases and NORMAL first transfers.
    first.sort(key=lambda v:(v[4],v[0]))
    selected=[]
    for x in first:
        if all(abs(x[0]-y[0])>.6 for y in selected):selected.append(x)
        if len(selected)>=10:break
    for t1,dep1,tof1,dv1,sp1 in selected:
        if time.monotonic()>LIMIT-3:return
        left=cap-t1-MIN_TOF
        if left<=0:continue
        for k in range(96):
            dep2=round(t1+left*rng.random()**2,6)
            remaining=cap-dep2
            if remaining<MIN_TOF:continue
            tof2=round((rng.choice(flights) if flights and k%4==0 else
                        MIN_TOF+(remaining-MIN_TOF)*rng.random()**1.5),6)
            if tof2>remaining or tof2<MIN_TOF:continue
            dv2=dv(m,b,dep2,tof2)
            sp2=int(dv2>NORMAL+1e-6)
            if dv2>SPECIAL+1e-6 or sp1+sp2>available:continue
            arrive2=dep2+tof2
            leg1={'from':a,'to':m,'start_time':dep1,'tof':tof1,'dv':dv1,'status':'SPECIAL' if sp1 else 'NORMAL'}
            leg2={'from':m,'to':b,'start_time':dep2,'tof':tof2,'dv':dv2,'status':'SPECIAL' if sp2 else 'NORMAL'}
            middle=legs[:i]+[leg1,leg2]
            if i+1==len(legs):suffixes=[[]]
            else:
                delay=max(0.,arrive2-nextdep)
                suffixes=[]
                for shift in [delay,delay+.25,delay+1.,delay+2.]:
                    if old_end+shift>END+1e-6:continue
                    prev_arr=arrive2
                    remaining_sp=MAX_SP-sum(float(z['dv'])>NORMAL+1e-6 for z in middle)
                    ss=[]
                    for tr in legs[i+1:]:
                        departure=round(float(tr['start_time'])+shift,6)
                        tof=float(tr['tof'])
                        if departure<prev_arr-1e-5:break
                        value=dv(tr['from'],tr['to'],departure,tof)
                        flag=int(value>NORMAL+1e-6)
                        if value>SPECIAL+1e-6 or remaining_sp-flag<0:break
                        remaining_sp-=flag
                        ss.append({'from':int(tr['from']),'to':int(tr['to']),
                                   'start_time':departure,'tof':tof,'dv':value,
                                   'status':'SPECIAL' if flag else 'NORMAL'})
                        prev_arr=departure+tof
                    if len(ss)==len(legs)-i-1:suffixes.append(ss)
            for suffix in suffixes:
                newroute=route[:i+1]+[m]+route[i+1:]
                newlegs=middle+suffix
                arrival=float(newlegs[-1]['start_time'])+float(newlegs[-1]['tof'])
                candidate={'route':newroute,'transfers':newlegs,'time':arrival}
                if len(newroute)!=39 or (best is not None and best['time']<=arrival):continue
                if verify(candidate):
                    best=candidate;atomic(OUT,record_for(candidate))
                    print('*** PROGRES 39/49 | DITA',round(arrival,6),'| +',m,
                          '| SPECIAL i zevendesuar:',oldsp,'***',flush=True)
                    return

t_start=time.monotonic()
try:
    for k,(n,i,m) in enumerate(tasks,1):
        if time.monotonic()>LIMIT-3:break
        taskid=f'{n}:{i}:{m}'
        if taskid in completed:continue
        rng=random.Random(185000+n*20000+i*100+m)
        try_bridge(seeds[n],i,m,rng)
        if time.monotonic()>LIMIT-3:break
        completed.add(taskid)
        if k%20==0:
            atomic(CHECK,{'version':185,'seed_signatures':seed_signatures,
                          'completed':sorted(completed)})
            print('Prova',k,'/',len(tasks),'| Kryer',len(completed),
                  '| best',('39/49' if best else '38/49'), '| DV',calls,
                  '| sek',round(time.monotonic()-t_start,1),flush=True)
finally:
    atomic(CHECK,{'version':185,'seed_signatures':seed_signatures,
                  'completed':sorted(completed)})
    atomic(R/'hapi185_status.json',{'tasks_completed':len(completed),'tasks_total':len(tasks),
               'found_39':best is not None,'best_day':best['time'] if best else None,
               'dv_calculations':calls,'seconds':round(time.monotonic()-t_start,2)})
    print('\nPERMBLEDHJE HAPI 185:',('39/49' if best else '38/49'),
          '| Me kohe',round(best['time'],6) if best else '173.466876',
          '| Prova',len(completed),'/',len(tasks),flush=True)
    print('Statusi:',R/'hapi185_status.json',flush=True)
    print('Rekordi origjinal mbetet i paprekur.',flush=True)
