import json, math, time, hashlib
from pathlib import Path
from scipy.optimize import differential_evolution
from utils_users import TomatoProblem

ROOT=Path('/kaggle/working/esa-spoc4')
RES=ROOT/'results'
SOURCE=RES/'best_verified_34.json'
CP=RES/'beam_v34_gap_splice_checkpoint.json'
OUT=RES/'beam_v34_gap_splice_level_35.json'
LOCK=RES/'best_verified_35.json'
P=TomatoProblem(str(ROOT/'data/easy.kttsp'))
raw=SOURCE.read_bytes()
base=json.loads(raw)['frontier'][0]


def verify(s, size):
    try:
        r=s['route']; ts=s['transfers']
        if len(r)!=size or r[0]!=0 or len(set(r))!=size or len(ts)!=size-1: return False
        day=0.; specials=0
        for j,tr in enumerate(ts):
            a,b=int(tr['from']),int(tr['to'])
            d=float(tr['start_time']); f=float(tr['tof'])
            if (a,b)!=(r[j],r[j+1]) or d<day-1e-7 or f<=0 or d+f>P.max_time+1e-6: return False
            dv=float(P.compute_transfer(a,b,d,f))
            if not math.isfinite(dv) or abs(dv-float(tr['dv']))>0.12: return False
            status='NORMAL' if dv<P.dv_threshold else 'SPECIAL' if dv<P.exception_threshold else 'INVALID'
            if status!=tr['status']: return False
            specials+=(status=='SPECIAL'); day=d+f
        return specials==s['special_used'] and specials<=P.n_exceptions and abs(day-s['current_time'])<1e-5
    except Exception as err:
        return False

assert verify(base,34), 'Rekordi 34 deshtoi riverifikimin'
missing=sorted(set(range(P.n_tomatoes))-set(base['route']))
windows=[]
for i,old in enumerate(base['transfers']):
    prev=0. if i==0 else float(base['transfers'][i-1]['start_time'])+float(base['transfers'][i-1]['tof'])
    upper=float(base['transfers'][i+1]['start_time']) if i+1<len(base['transfers']) else float(P.max_time)
    width=upper-prev
    if width>=1.5:
        windows.append((i,prev,upper,width))
ranked=sorted(windows,key=lambda v:-v[3])
wide=ranked[:14]
specials=[x for x in ranked if base['transfers'][x[0]]['status']=='SPECIAL']
windows=wide+[x for x in specials if x[0] not in {y[0] for y in wide}]
tasks=[(i,t) for i,_,_,_ in windows for t in missing]
config={'source_sha256':hashlib.sha256(raw).hexdigest(),'edges':'top14_plus_special','maxiter':14,'popsize':7,'seed':157}
if CP.exists():
    ck=json.loads(CP.read_text(encoding='utf-8'))
    assert ck['config']==config, 'Checkpoint i nje konfigurimi tjeter'
else:
    ck={'config':config,'done':{},'solutions':[]}

def atomic(path,obj):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj,indent=2),encoding='utf-8')
    temp.replace(path)

win={i:(lo,hi,w) for i,lo,hi,w in windows}
print('HAPI 157B - GAP-SPLICE ME OPTIMIZIM DE',flush=True)
print('Baza 34/49: OK | Domate te mbetura:',len(missing),flush=True)
print('Dritaret me te gjera:',[(i+1,round(w,2)) for i,_,_,w in windows[:7]],flush=True)
print('Prova te planifikuara:',len(tasks),'| te perfunduara:',len(ck['done']),flush=True)
start=time.monotonic()
for k,(i,target) in enumerate(tasks,1):
    key=f'{i}:{target}'
    if key in ck['done']: continue
    prev,upper,width=win[i]
    old=base['transfers'][i]
    a,b=base['route'][i:i+2]
    cap=int(old['status']=='SPECIAL')
    calls=[0]; best=[float('inf'),None]
    def decode(x):
        z=[float(v) for v in x]; total=1.+sum(z); m=width-1.0
        depart=prev+m*z[0]/total
        tof1=0.5+m*z[1]/total
        wait=m*z[2]/total
        tof2=0.5+m*z[3]/total
        return depart,tof1,wait,tof2
    def objective(x):
        d,t1,w,t2=decode(x)
        calls[0]+=1
        try:
            dv1=float(P.compute_transfer(a,target,d,t1))
            dv2=float(P.compute_transfer(target,b,d+t1+w,t2))
        except Exception:
            return 1e7
        if not math.isfinite(dv1) or not math.isfinite(dv2):return 1e7
        if cap:
            penalty=min(max(0,dv1-99.999)+max(0,dv2-599.999),max(0,dv1-599.999)+max(0,dv2-99.999))
        else:
            penalty=max(0,dv1-99.999)+max(0,dv2-99.999)
        score=penalty+0.0001*(dv1+dv2)
        if score<best[0]:best[:]=[score,(d,t1,w,t2,dv1,dv2)]
        return score
    result=differential_evolution(objective,[(0,8)]*4,maxiter=14,popsize=7,polish=False,seed=157000+k,tol=0.001)
    if best[1] is None:
        ck['done'][key]={'score':None,'calls':calls[0]}
    else:
        d,t1,w,t2,dv1,dv2=best[1]
        st1='NORMAL' if dv1<P.dv_threshold else 'SPECIAL' if dv1<P.exception_threshold else 'INVALID'
        st2='NORMAL' if dv2<P.dv_threshold else 'SPECIAL' if dv2<P.exception_threshold else 'INVALID'
        valid=(st1!='INVALID' and st2!='INVALID' and int(st1=='SPECIAL')+int(st2=='SPECIAL')<=cap)
        ck['done'][key]={'score':round(best[0],4),'dv1':round(dv1,3),'dv2':round(dv2,3),'calls':calls[0],'valid':valid}
        if valid:
            tr1={'from':a,'to':target,'start_time':d,'tof':t1,'dv':dv1,'status':st1}
            tr2={'from':target,'to':b,'start_time':d+t1+w,'tof':t2,'dv':dv2,'status':st2}
            ts=base['transfers'][:i]+[tr1,tr2]+base['transfers'][i+1:]
            end=ts[-1]['start_time']+ts[-1]['tof']
            child={'route':base['route'][:i+1]+[target]+base['route'][i+1:],'transfers':ts,'current_time':end,'special_used':base['special_used']-cap+int(st1=='SPECIAL')+int(st2=='SPECIAL')}
            if verify(child,35):
                ck['solutions'].append(child)
                atomic(OUT,{'level':35,'verified':True,'frontier':ck['solutions']})
                if not LOCK.exists():
                    atomic(LOCK,{'level':35,'verified':True,'source':OUT.name,'frontier':[child]})
                print('U GJET 35/49! | Dita:',round(end,5),'| domatja:',target,'| pozita:',i+1,flush=True)
    atomic(CP,ck)
    if k%10==0:
        print(f'Prova {k}/{len(tasks)} | DV={calls[0]} | permiresimi me i mire={best[0]:.3f} | 35={len(ck["solutions"])}',flush=True)
    if ck['solutions']:break
    if time.monotonic()-start>125:
        print('Kufiri i kohes per qelize: checkpoint u ruajt.',flush=True)
        break

print('\nPERMBLEDHJA E HAPIT 157B',flush=True)
print('Prova:',len(ck['done']),'/',len(tasks),flush=True)
print('Rruge 35:',len(ck['solutions']),flush=True)
print('Kerkimi i pjesshem:',len(ck['done'])<len(tasks) and not bool(ck['solutions']),flush=True)
print('Rekordi i sigurt: 34/49' if not ck['solutions'] else 'REKORD I RI I RIVERIFIKUAR: 35/49',flush=True)
if not ck['solutions']:
    top=sorted([(v['score'],key,v) for key,v in ck['done'].items() if v['score'] is not None])[:6]
    print('6 kandidatet me afer kufijve:',[(key,round(score,2)) for score,key,v in top],flush=True)
