"""ESA SpOC4 Hapi 159B: neighboring-leg optimization, 35 -> 36.
Keeps original best_verified_35.json untouched. Runs in PyKEP 2.6.4 environment.
"""
from pathlib import Path
import json, math, hashlib, itertools, time
import numpy as np
from scipy.optimize import differential_evolution
from utils_users import TomatoProblem

ROOT = Path('/kaggle/working/esa-spoc4')
RESULTS = ROOT / 'results'
SOURCE = RESULTS / 'best_verified_35.json'
CP = RESULTS / 'beam_v36_neighbor_opt_checkpoint.json'
OUTPUT = RESULTS / 'beam_v36_neighbor_opt_level_36.json'
LOCK = RESULTS / 'best_verified_36.json'
P = TomatoProblem(str(ROOT / 'data/easy.kttsp'))


def atomic(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    tmp.replace(path)


def transfer_dv(a, b, depart, tof):
    try:
        v = float(P.compute_transfer(int(a), int(b), float(depart), float(tof)))
        return v if math.isfinite(v) else 1e7
    except Exception:
        return 1e7


def verify(s, expected):
    try:
        route = s['route']
        tr = s['transfers']
        if len(route) != expected or len(tr) != expected-1:
            return False
        if route[0] != 0 or len(set(route)) != expected:
            return False
        if any(not 0 <= int(x) < P.n_tomatoes for x in route):
            return False
        day = 0.0
        specials = 0
        for i, t in enumerate(tr):
            a, b = int(t['from']), int(t['to'])
            depart, tof = float(t['start_time']), float(t['tof'])
            if a != route[i] or b != route[i+1] or tof < 0.499999:
                return False
            if depart < day-1e-6 or depart+tof > P.max_time+1e-6:
                return False
            dv = transfer_dv(a, b, depart, tof)
            if abs(dv-float(t['dv'])) > 0.1:
                return False
            if dv < P.dv_threshold:
                status = 'NORMAL'
            elif dv < P.exception_threshold:
                status = 'SPECIAL'
                specials += 1
            else:
                return False
            if status != t['status']:
                return False
            day = depart+tof
        return (specials == s['special_used']
                and specials <= P.n_exceptions
                and abs(day-s['current_time']) < 1e-5)
    except Exception:
        return False


def make_tasks(route, transfers):
    # Most promising near-misses reported by Hapi 158, best first.
    pairs = [(2,26), (2,24), (2,35), (5,6), (2,33), (2,6)]
    missing = set(range(P.n_tomatoes))-set(route)
    jobs = []
    seen = set()
    for edge, target in pairs:
        if target not in missing:
            continue
        # 4 old legs -> 5 new: allows retiming neighbors, even two specials.
        # 3 old legs -> 4 new: narrower and cheaper search.
        for width in (4,3):
            for left in range(max(0,edge-width+1), edge+1):
                if left+width > len(transfers):
                    continue
                old_middle = route[left+1:left+width]
                orders = []
                for pos in range(len(old_middle)+1):
                    o = old_middle[:pos]+[target]+old_middle[pos:]
                    orders.append(o)
                if width == 3 and target in (26,24):
                    orders += [list(o) for o in itertools.permutations(old_middle+[target])]
                for middle in orders:
                    key = (left,width,target,tuple(middle))
                    if key not in seen:
                        seen.add(key)
                        jobs.append({'left':left,'width':width,
                                     'target':target,'middle':middle})
    return jobs


def optimize_job(base, job, seed):
    route = base['route']
    old = base['transfers']
    left, width = job['left'], job['width']
    middle = job['middle']
    nodes = [route[left]] + middle + [route[left+width]]
    n = len(nodes)-1
    previous = (0.0 if left == 0 else
                float(old[left-1]['start_time'])+float(old[left-1]['tof']))
    deadline = (float(old[left+width]['start_time'])
                if left+width < len(old) else float(P.max_time))
    window = deadline-previous
    if window <= n*0.5+1e-5:
        return None, 1e7, 0

    # SPECIAL capacity is taken from removed legs, so the full route
    # can never use more than 5 SPECIAL transfers.
    allowed = sum(t['status']=='SPECIAL' for t in old[left:left+width])
    free = window-n*0.5-1e-5
    calls = [0]
    best_score = [1e20]
    found = [None]

    def decode(x):
        x = np.asarray(x, dtype=float)
        used = free*float(x[0])
        weights = np.maximum(x[1:], 0.001)
        weights = weights/weights.sum()
        tofs = 0.5+used*weights[:n]
        waits = used*weights[n:]
        t = previous
        timeline = []
        for j in range(n):
            departure = t+float(waits[j])
            duration = float(tofs[j])
            timeline.append((departure,duration))
            t = departure+duration
        return timeline, t

    def build_candidate(timeline, dvs, arrival):
        local = []
        for j, (dep,tof) in enumerate(timeline):
            v = dvs[j]
            status = ('NORMAL' if v < P.dv_threshold else 'SPECIAL')
            local.append({'from':nodes[j], 'to':nodes[j+1],
                          'start_time':dep, 'tof':tof,
                          'dv':float(v), 'status':status})
        child = {'route':route[:left+1]+middle+route[left+width:],
                 'transfers':old[:left]+local+old[left+width:],
                 'special_used':sum(t['status']=='SPECIAL' for t in old[:left])
                                +sum(t['status']=='SPECIAL' for t in local)
                                +sum(t['status']=='SPECIAL' for t in old[left+width:]),
                 'current_time':(base['current_time']
                                 if left+width < len(old) else arrival)}
        return child

    def objective(x):
        timeline, arrival = decode(x)
        calls[0] += 1
        dvs = [transfer_dv(nodes[j],nodes[j+1],dep,tof)
               for j,(dep,tof) in enumerate(timeline)]
        ordinary = [max(0.0,v-(P.dv_threshold-0.001)) for v in dvs]
        special = [max(0.0,v-(P.exception_threshold-0.001)) for v in dvs]
        savings = sorted([a-b for a,b in zip(ordinary,special)],reverse=True)
        penalty = max(0.0, sum(ordinary)-sum(savings[:allowed]))
        score = penalty + 0.00001*sum(dvs)
        if score < best_score[0]:
            best_score[0] = score
        if penalty < 1e-8 and found[0] is None:
            child = build_candidate(timeline,dvs,arrival)
            if verify(child,36):
                found[0] = child
        return score

    dim = 2*n+1
    try:
        differential_evolution(
            objective,
            [(0.0,1.0)]+[(0.001,1.0)]*(dim-1),
            maxiter=12, popsize=4, polish=False,
            tol=0.001, seed=seed, workers=1
        )
    except Exception as ex:
        print('Optimizimi u nderpre:',repr(ex),flush=True)
    return found[0], best_score[0], calls[0]


def main():
    raw = SOURCE.read_bytes()
    data = json.loads(raw)
    base = data['frontier'][0]
    assert verify(base,35), 'Rekordi burimor 35 nuk u riverifikua.'
    jobs = make_tasks(base['route'], base['transfers'])
    config = {'source_sha256':hashlib.sha256(raw).hexdigest(),
              'jobs':len(jobs),'maxiter':12,'popsize':4,'version':1}
    if CP.exists():
        ck = json.loads(CP.read_text(encoding='utf-8'))
        assert ck['config']==config, 'Checkpoint me parametra te tjere.'
    else:
        ck = {'config':config,'done':{},'solutions':[]}
        atomic(CP,ck)

    print('HAPI 159B - OPTIMIZIMI I TRANSFERIMEVE FQINJE',flush=True)
    print('Baza 35/49: OK | SPECIAL:',base['special_used'],flush=True)
    print('Kandidate nga Hapi 158: 2:26, 2:24, 2:35, 5:6, 2:33, 2:6',flush=True)
    print('Prova:',len(jobs),'| Te perfunduara:',len(ck['done']),flush=True)
    started = time.monotonic()
    # Checkpoint after EVERY complete job; stop cleanly before outer timeout.
    for index,job in enumerate(jobs):
        key = (str(job['left'])+':'+str(job['width'])+':'+
               str(job['target'])+':'+','.join(map(str,job['middle'])))
        if key in ck['done']:
            continue
        seed = 159000+index
        child,score,calls = optimize_job(base,job,seed)
        ck['done'][key] = {'score':round(score,6),'dv_tests':calls}
        if child is not None:
            ck['solutions'].append(child)
        atomic(CP,ck)
        if len(ck['done'])%5==0 or child is not None:
            print('Prova',len(ck['done']),'/',len(jobs),
                  '| segment',job['left'],'+',job['width'],
                  '| target',job['target'],
                  '| score',round(score,3),'| 36=',len(ck['solutions']),flush=True)
        if child is not None:
            break
        if time.monotonic()-started > 120:
            break

    valid = [s for s in ck['solutions'] if verify(s,36)]
    if valid:
        best = min(valid,key=lambda s:(s['special_used'],s['current_time']))
        atomic(OUTPUT, {'level':36,'verified':True,'frontier':valid})
        if not LOCK.exists():
            atomic(LOCK, {'level':36,'verified':True,
                          'source':OUTPUT.name,'frontier':[best]})
        print('REKORD I RI I RIVERIFIKUAR: 36/49!',flush=True)
        print('SPECIAL:',best['special_used'],
              '| Dita:',round(best['current_time'],6),flush=True)
        print('Rruga:',best['route'],flush=True)
    else:
        print('Rekordi i sigurt mbetet: 35/49',flush=True)

    print('\nPERMBLEDHJA E HAPIT 159B',flush=True)
    print('Prova:',len(ck['done']),'/',len(jobs),flush=True)
    print('Rruge te riverifikuara 36:',len(valid),flush=True)
    print('Kerkimi i pjesshem:',
          bool(len(ck['done']) < len(jobs) and not valid),flush=True)
    if not valid and ck['done']:
        closest = sorted(ck['done'].items(),key=lambda kv:kv[1]['score'])[:5]
        print('5 kandidatet me te afert:',
              [(k,round(v['score'],3)) for k,v in closest],flush=True)


if __name__ == '__main__':
    main()
