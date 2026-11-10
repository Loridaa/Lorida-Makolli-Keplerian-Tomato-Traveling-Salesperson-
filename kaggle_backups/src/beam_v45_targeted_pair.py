from pathlib import Path
import json, hashlib, math, time, os, sys

ROOT = Path('/kaggle/working/esa-spoc4')
SRC = ROOT / 'src'
RES = ROOT / 'results'
sys.path.insert(0, str(SRC))
from utils_users import TomatoProblem

P = TomatoProblem(str(ROOT / 'data/easy.kttsp'))
OFFICIAL = RES / 'best_verified_38.json'
ALTERNATIVES = RES / 'beam_v43_alternatives_38_level_38.json'
CP = RES / 'beam_v45_targeted_pair_checkpoint.json'
OUTPUT = RES / 'beam_v45_targeted_pair_level_39.json'
LOCK = RES / 'best_verified_39.json'
END = time.monotonic() + 122
MAX_STATES = 105


def atomic(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, separators=(',', ':'), allow_nan=False), encoding='utf8')
    os.replace(tmp, path)


def verify(s, n):
    try:
        r = s['route']; tt = s['transfers']
        if len(r) != n or len(set(r)) != n or r[0] != 0 or len(tt) != n-1:
            return False
        if any(not isinstance(x, int) or x < 0 or x >= P.n_tomatoes for x in r):
            return False
        prev = 0.; used = 0
        for i, tr in enumerate(tt):
            if int(tr['from']) != r[i] or int(tr['to']) != r[i+1]:
                return False
            d, tof, dv = float(tr['start_time']), float(tr['tof']), float(tr['dv'])
            if not all(math.isfinite(x) for x in (d, tof, dv)):
                return False
            if d < prev - 1e-5 or tof <= 0 or d+tof > 200+1e-6:
                return False
            actual = float(P.compute_transfer(r[i], r[i+1], d, tof))
            if not math.isfinite(actual) or abs(actual-dv) > max(1e-4, abs(actual)*2e-6):
                return False
            if tr['status'] == 'NORMAL':
                if actual >= P.dv_threshold-1e-6:
                    return False
            elif tr['status'] == 'SPECIAL':
                if actual >= P.exception_threshold-1e-6:
                    return False
                used += 1
            else:
                return False
            prev = d+tof
        return used <= P.n_exceptions and used == s['special_used'] and abs(prev-s['current_time']) < 1e-4
    except Exception:
        return False


src_data = [json.loads(OFFICIAL.read_text())['frontier'][0]]
src_data.extend(json.loads(ALTERNATIVES.read_text())['frontier'])
sources = []
seen = set()
for b in src_data:
    if not verify(b, 38):
        raise ValueError('Nje nga rruget 38 nuk e kaloi riverifikimin PyKEP.')
    if b['route'][:3] != [0, 39, 8]:
        continue
    if b['route'][3] not in (14, 38) or b['route'][4] != 42:
        continue
    key = (tuple(b['route']), tuple((round(x['start_time'], 5), round(x['tof'], 5)) for x in b['transfers'][:9]))
    if key not in seen:
        seen.add(key)
        sources.append(b)
assert sources, 'Nuk u gjeten rruget 38 me 14 ose 38 ne poziten e katert.'

# Provojme te dy renditjet: 8->14->38->42 dhe 8->38->14->42.
tasks = []
for bi, b in enumerate(sources):
    missing = 38 if b['route'][3] == 14 else 14
    for anchor in (7, 8, 9, 10):
        for pos in (3, 4):
            for phase in (0., 0.125):
                tasks.append(dict(bi=bi, target=missing, anchor=anchor, pos=pos, phase=phase))

cfg = dict(version=168, base_sha256=hashlib.sha256(OFFICIAL.read_bytes()).hexdigest(),
           alt_sha256=hashlib.sha256(ALTERNATIVES.read_bytes()).hexdigest(),
           tasks=tasks, max_states=MAX_STATES)
if CP.exists():
    ck = json.loads(CP.read_text(encoding='utf8'))
    assert ck['config'] == cfg, 'Checkpoint i nje konfigurimi tjeter; nuk u ndryshua.'
else:
    ck = dict(config=cfg, done={}, active=None, solutions=[])
    atomic(CP, ck)

print('HAPI 168 - TARGETED PAIR 14+38 / PHASE GRID', flush=True)
print('Burime te verifikuara:',len(sources),'| Prova:',len(tasks), '| PERFUNDIME:',len(ck['done']),flush=True)
print('Rekordi zyrtar 38 ruhet i paprekur.',flush=True)


def advance(states, a, b, deadline, used_cap, phase, warm):
    waits = [0., .25, .5, .75, 1., 1.5, 2., 3., 4., 6., 8.]
    tofs = [.5, .75, 1., 1.25, 1.5, 2., 2.5, 3., 4., 5., 6., 8.]
    if phase:
        waits += [phase, .5+phase, 1.+phase, 2.+phase]
        tofs += [.5+phase, 1.+phase, 2.+phase, 3.+phase]
    if warm:
        tofs.append(float(warm['tof']))
    # Cache lokal per nje kalim. Vleresime identike nuk perseriten.
    cache = {}
    buckets = {}
    for s in states:
        departures = {round(s['t']+w, 6) for w in waits}
        if warm and float(warm['start_time']) >= s['t']-1e-6:
            departures.add(round(float(warm['start_time']), 6))
        for d in departures:
            if d < s['t']-1e-6 or d+.5 > deadline+1e-6:
                continue
            for tof in tofs:
                arr = round(d+tof, 6)
                if arr > deadline+1e-6:
                    continue
                key = (d,round(tof,6))
                if key not in cache:
                    try:
                        cache[key] = float(P.compute_transfer(a,b,d,tof))
                    except Exception:
                        cache[key] = float('inf')
                dv = cache[key]
                if not math.isfinite(dv) or dv < 0:
                    continue
                if dv < P.dv_threshold - 0.001:
                    status, inc = 'NORMAL', 0
                elif dv < P.exception_threshold - 0.001:
                    status, inc = 'SPECIAL', 1
                else:
                    continue
                if s['used'] + inc > used_cap:
                    continue
                tr = {'from':a,'to':b,'start_time':d,'tof':float(tof), 'dv':dv,'status':status}
                nxt = {'t':arr,'used':s['used']+inc,
                       'cost':s['cost']+dv,'legs':s['legs']+[tr]}
                k = (nxt['used'], round(arr*8))
                old = buckets.get(k)
                if old is None or nxt['cost'] < old['cost']:
                    buckets[k] = nxt
    result = list(buckets.values())
    if len(result) > MAX_STATES:
        # Ruajme shperndarjen ne kohe DHE buxhetin SPECIAL.
        groups = {}
        for s in result:
            groups.setdefault(s['used'], []).append(s)
        chosen = []
        quotas = {u:MAX_STATES//len(groups) for u in groups}
        for u in sorted(groups):
            quotas[u] += (MAX_STATES % len(groups)) if u == min(groups) else 0
            group = sorted(groups[u],key=lambda s:s['t'])
            count = min(quotas[u],len(group))
            if count >= len(group):
                chosen.extend(group)
            elif count == 1:
                chosen.append(min(group,key=lambda s:s['cost']))
            else:
                ids = sorted({round(i*(len(group)-1)/(count-1)) for i in range(count)})
                chosen.extend(group[i] for i in ids)
        if len(chosen) < MAX_STATES:
            have = {(s['used'],round(s['t'],6)) for s in chosen}
            remaining = sorted((s for s in result if (s['used'],round(s['t'],6)) not in have),
                               key=lambda s:(s['cost'],s['t']))
            chosen.extend(remaining[:MAX_STATES-len(chosen)])
        result = chosen[:MAX_STATES]
    return result, len(cache)


success = False
paused = False
for ti, task in enumerate(tasks):
    if str(ti) in ck['done']:
        continue
    b = sources[task['bi']]
    new_route = b['route'][:task['pos']] + [task['target']] + b['route'][task['pos']:]
    anchor = task['anchor']
    path = new_route[:anchor+2]
    deadline = float(b['transfers'][anchor]['start_time'])
    suffix = b['transfers'][anchor:]
    prefix_cap = P.n_exceptions - sum(tr['status']=='SPECIAL' for tr in suffix)
    warm = {(tr['from'],tr['to']):tr for tr in b['transfers']}
    if ck['active'] and ck['active']['id']==ti:
        j = ck['active']['layer']
        states = ck['active']['states']
    else:
        j = 0
        states = [dict(t=0., used=0, cost=0., legs=[])]
    print('Prova',ti+1,'/',len(tasks),'| baza',task['bi'],'| rendi',new_route[2:6],
          '| ankorimi',b['route'][anchor],'| kohe',round(deadline,3),
          '| SPECIAL prefix',prefix_cap,'| nis layer',j,flush=True)
    calls_total = 0
    while j < len(path)-1:
        if time.monotonic() >= END:
            paused = True
            break
        states, calls = advance(states, path[j], path[j+1], deadline,
                                prefix_cap, task['phase'],warm.get((path[j],path[j+1])))
        calls_total += calls
        j+=1
        ck['active'] = dict(id=ti,layer=j,states=states)
        atomic(CP,ck)
        print('  shtresa',j,'/',len(path)-1,'| gjendje',len(states),'| DV teste',calls, flush=True)
        if not states:
            break
    if paused:
        print('Kerkimi u ndal ne nje checkpoint te sigurt.',flush=True)
        break
    found = None
    if states and j == len(path)-1:
        for s in sorted(states,key=lambda q:(q['used'],q['cost'],q['t'])):
            candidate={'route':new_route,'transfers':s['legs']+suffix,
                       'special_used':s['used']+sum(tr['status']=='SPECIAL' for tr in suffix),
                       'current_time':float(b['current_time'])}
            if verify(candidate,39):
                found=candidate
                break
    ck['done'][str(ti)] = {'found':bool(found),'calls':calls_total,
                            'states':len(states)}
    ck['active'] = None
    if found:
        ck['solutions'].append(found)
    atomic(CP,ck)
    if found:
        atomic(OUTPUT, {'level':39,'verified':True,'source':'Hapi 168','frontier':[found]})
        if not LOCK.exists():
            atomic(LOCK,{'level':39,'verified':True,'source':OUTPUT.name,'frontier':[found]})
        else:
            assert verify(json.loads(LOCK.read_text())['frontier'][0],39)
        print('U GJET DHE U RIVERIFIKUA 39/49!',flush=True)
        print('Rruga:',found['route'],flush=True)
        success=True
        break

print('\nPERMBLEDHJA E HAPIT 168',flush=True)
print('Prova te perfunduara:',len(ck['done']),'/',len(tasks),flush=True)
print('Rruge te verifikuara 39:',len(ck['solutions']),flush=True)
print('Kerkimi i pjesshem:',not success and (len(ck['done'])<len(tasks)),flush=True)
print('Rekordi:', '39/49' if success or LOCK.exists() else '38/49',flush=True)
