"""ESA SpOC4 Hapi 160: targeted, widened gap optimizer (35 -> 36).
Concentrates on the target tomato 33 near the verified route prefix.
Never overwrites the verified 35-route; checkpoints after each completed job.
"""
from pathlib import Path
import json, math, time, hashlib
import numpy as np
from scipy.optimize import differential_evolution
from utils_users import TomatoProblem

ROOT = Path('/kaggle/working/esa-spoc4')
RES = ROOT / 'results'
SOURCE = RES / 'best_verified_35.json'
CP = RES / 'beam_v37_wide_prefix_33_checkpoint.json'
OUTPUT = RES / 'beam_v37_wide_prefix_33_level_36.json'
LOCK = RES / 'best_verified_36.json'
P = TomatoProblem(str(ROOT / 'data/easy.kttsp'))
TARGET = 33


def atomic(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, separators=(',', ':')), encoding='utf-8')
    tmp.replace(path)


def dv(a, b, depart, tof):
    try:
        val = float(P.compute_transfer(int(a), int(b), float(depart), float(tof)))
        return val if math.isfinite(val) and val >= 0 else 1e7
    except Exception:
        return 1e7


def verify(s, n):
    try:
        route, trans = s['route'], s['transfers']
        if (len(route) != n or route[0] != 0 or len(set(route)) != n
                or len(trans) != n-1):
            return False
        if any(not (0 <= int(node) < P.n_tomatoes) for node in route):
            return False
        day, special = 0.0, 0
        for i, tr in enumerate(trans):
            a, b = int(tr['from']), int(tr['to'])
            dep, tof = float(tr['start_time']), float(tr['tof'])
            if a != route[i] or b != route[i+1] or dep < day-1e-6 or tof < 0.5-1e-6:
                return False
            if dep+tof > P.max_time+1e-6:
                return False
            v = dv(a,b,dep,tof)
            if abs(v-float(tr['dv'])) > 0.1:
                return False
            status = ('NORMAL' if v < P.dv_threshold
                      else 'SPECIAL' if v < P.exception_threshold else 'INVALID')
            if status != tr['status']:
                return False
            special += status == 'SPECIAL'
            day = dep+tof
        return (special == s['special_used'] <= P.n_exceptions
                and abs(day-float(s['current_time'])) < 1e-5)
    except Exception:
        return False


def jobs_for(base):
    route = base['route']
    assert TARGET not in route, 'Domatja 33 eshte tashme ne itinerar.'
    tasks=[]
    # Previous Hapi 159B covered widths 3 and 4. These are NEW, wider windows.
    for width in (5,6,7,8,9):
        for left in (0,1,2):
            old_nodes=route[left:left+width+1]
            if len(old_nodes)!=width+1:
                continue
            # Insert near node 8/3/7; keep original nodes, but retime the whole window.
            anchor_nodes=set(old_nodes)
            if not {8,3}.issubset(anchor_nodes):
                continue
            for p in range(1,width+1):
                if p > 5-left or p < 2-left:
                    continue
                seq=old_nodes[:p]+[TARGET]+old_nodes[p:]
                # preserve all original edges outside the enlarged window
                tasks.append({'left':left,'width':width,'insert':p,'nodes':seq})
    return tasks


def optimize(base, job, seed):
    old, route=base['transfers'],base['route']
    left,width,insert=job['left'],job['width'],job['insert']
    nodes=job['nodes']; n=len(nodes)-1
    previous=(0.0 if left==0 else float(old[left-1]['start_time'])+float(old[left-1]['tof']))
    deadline=(float(old[left+width]['start_time'])
              if left+width<len(old) else float(P.max_time))
    free=deadline-previous-0.5*n-0.000001
    if free <= 0:
        return None, {'score':1e7,'dv_tests':0,'reason':'no_time'}
    allowed=sum(t['status']=='SPECIAL' for t in old[left:left+width])
    best=[float('inf'),None]
    found=[None]
    calls=[0]

    # Warm-start from verified original timing. Split the original transfer
    # around the new node and let DE redistribute waits and durations.
    orig=old[left:left+width]
    base_tofs=[]; base_waits=[]
    t=previous
    for i,tr in enumerate(orig):
        dep=float(tr['start_time'])
        duration=float(tr['tof'])
        w=max(0.,dep-t)
        if i==insert-1:
            base_tofs.extend([max(0.5,duration/2),max(0.5,duration/2)])
            base_waits.extend([w,0.])
        else:
            base_tofs.append(max(.5,duration))
            base_waits.append(w)
        t=dep+duration
    assert len(base_tofs)==n and len(base_waits)==n
    extras=[max(0.,x-0.5) for x in base_tofs]+base_waits
    extra_sum=sum(extras)
    desired=min(0.98, max(0.15, extra_sum/free))
    a=np.asarray([max(0.002,v) for v in extras],dtype=float)
    a=np.maximum(.002,a/(max(a.max(),.002))*0.85)
    baseline=np.r_[desired,a]
    rng=np.random.default_rng(seed)
    bounds=[(.05,1.)]+[(.002,1.)]*(2*n)
    pop_size=max(28,4*(2*n+1))
    population=rng.uniform(.005,1.,size=(pop_size,2*n+1))
    population[:,0]=rng.uniform(.5,1.,size=pop_size)
    population[0]=baseline
    for j in range(1,min(12,pop_size)):
        population[j]=np.clip(baseline*rng.lognormal(0,.25,size=2*n+1),
                              [lo for lo,_ in bounds],[hi for _,hi in bounds])

    def evaluate(x):
        calls[0]+=1
        use=free*float(x[0]); weights=np.maximum(np.asarray(x[1:],dtype=float),.001)
        weights=weights/weights.sum()
        tofs=.5+use*weights[:n]
        waits=use*weights[n:]
        t=previous; timeline=[]; vs=[]
        for j in range(n):
            dep=t+float(waits[j]); tof=float(tofs[j]); t=dep+tof
            timeline.append((dep,tof))
            vs.append(dv(nodes[j],nodes[j+1],dep,tof))
        ordinary=[max(0.,v-(P.dv_threshold-0.001)) for v in vs]
        special=[max(0.,v-(P.exception_threshold-0.001)) for v in vs]
        savings=sorted((a-b for a,b in zip(ordinary,special)), reverse=True)
        penalty=max(0.,sum(ordinary)-sum(savings[:allowed]))
        score=penalty+0.000001*sum(vs)
        if score<best[0]:
            best[:]=[score,np.asarray(x).tolist()]
        if penalty<1e-9 and found[0] is None:
            local=[]
            for j,(dep,tof) in enumerate(timeline):
                v=vs[j]
                if v>=P.exception_threshold:
                    return score
                local.append({'from':nodes[j],'to':nodes[j+1],
                              'start_time':dep,'tof':tof,'dv':v,
                              'status':'NORMAL' if v<P.dv_threshold else 'SPECIAL'})
            child={'route':route[:left+1]+nodes[1:-1]+route[left+width:],
                   'transfers':old[:left]+local+old[left+width:],
                   'special_used':sum(t['status']=='SPECIAL' for t in old[:left])
                        +sum(t['status']=='SPECIAL' for t in local)
                        +sum(t['status']=='SPECIAL' for t in old[left+width:]),
                   'current_time':base['current_time']}
            if verify(child,36):
                found[0]=child
        return score

    try:
        differential_evolution(evaluate,bounds,maxiter=16,popsize=4,
                               init=population,polish=False,tol=.001,
                               updating='immediate',workers=1,seed=seed)
    except Exception as e:
        print('DE warning:',repr(e),flush=True)
    return found[0], {'score':round(best[0],6),'dv_tests':calls[0],
                      'best_x':best[1]}


def main():
    raw=SOURCE.read_bytes()
    base=json.loads(raw)['frontier'][0]
    assert verify(base,35), 'Rekordi 35 nuk u riverifikua.'
    jobs=jobs_for(base)
    config={'base_sha256':hashlib.sha256(raw).hexdigest(),
            'job_keys':[f"{x['left']}:{x['width']}:{x['insert']}" for x in jobs],
            'algorithm':'wide_prefix_warmstart_v1','target':TARGET}
    if CP.exists():
        ck=json.loads(CP.read_text(encoding='utf-8'))
        assert ck['config']==config, 'Checkpoint-i ka konfigurim tjeter.'
    else:
        ck={'config':config,'done':{},'solutions':[]}
        atomic(CP,ck)
    print('HAPI 160 - WIDE PREFIX + WARM START (35 -> 36)',flush=True)
    print('Rekordi 35/49 OK | Target i perzgjedhur: 33',flush=True)
    print('Dritare te reja 5-9 kalime | Prova:',len(jobs),flush=True)
    print('Prova te kryera:',len(ck['done']),flush=True)
    began=time.monotonic()
    for i,job in enumerate(jobs):
        key=f"{job['left']}:{job['width']}:{job['insert']}"
        if key in ck['done']:
            continue
        child,report=optimize(base,job,seed=160000+i)
        ck['done'][key]=report
        if child is not None:
            ck['solutions'].append(child)
        atomic(CP,ck)
        print('Prova',len(ck['done']),'/',len(jobs), '| window',key,
              '| score',report['score'], '| 36=',len(ck['solutions']),flush=True)
        if child is not None or time.monotonic()-began>110:
            break
    valid=[s for s in ck['solutions'] if verify(s,36)]
    if valid:
        choice=min(valid,key=lambda s:(s['special_used'],s['current_time']))
        atomic(OUTPUT,{'level':36,'verified':True,'frontier':valid})
        if LOCK.exists():
            old_lock=json.loads(LOCK.read_text(encoding='utf-8'))
            assert verify(old_lock['frontier'][0],36),'Existing lock invalid.'
        else:
            atomic(LOCK,{'level':36,'verified':True,
                         'source':OUTPUT.name,'frontier':[choice]})
        print('REKORD I RI I RIVERIFIKUAR: 36/49! | Dita:',
              round(choice['current_time'],5),flush=True)
        print('Rruga:',choice['route'],flush=True)
    else:
        print('Rekordi i sigurt mbetet 35/49.',flush=True)
    print('\nPERMBLEDHJA E HAPIT 160',flush=True)
    print('Prova:',len(ck['done']),'/',len(jobs),flush=True)
    print('Rruge 36 te riverifikuara:',len(valid),flush=True)
    print('Kerkimi i pjesshem:',len(ck['done'])<len(jobs) and not valid,flush=True)
    if not valid and ck['done']:
        near=sorted(ck['done'].items(),key=lambda x:x[1]['score'])[:5]
        print('5 me te afert:',[(k,v['score']) for k,v in near],flush=True)


if __name__=='__main__':
    main()
