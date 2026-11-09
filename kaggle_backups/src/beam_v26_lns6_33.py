import json, math, time, hashlib
from pathlib import Path
from collections import defaultdict
from functools import lru_cache
from utils_users import TomatoProblem

ROOT = Path('/kaggle/working/esa-spoc4')
RES = ROOT/'results'
SOURCE = RES/'best_verified_32.json'
CP = RES/'beam_v26_lns6_33_checkpoint.json'
OUT = RES/'beam_v26_lns6_level_33.json'
P = TomatoProblem(str(ROOT/'data/easy.kttsp'))
START = time.monotonic()
LIMIT = 110.0
WAITS = (0, .25, .5, 1, 2, 3, 5, 8)
TOFS = (.25, .5, .75, 1, 1.5, 2, 3, 4, 6, 8, 12, 18)


def save(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, separators=(',', ':')), encoding='utf-8')
    temp.replace(path)


@lru_cache(maxsize=180000)
def dv(a, b, dep, tof):
    try:
        value = float(P.compute_transfer(a, b, dep, tof))
        if math.isfinite(value) and value >= 0:
            return value
    except Exception:
        pass
    return math.inf


def verify(s, expected):
    try:
        route, moves = s['route'], s['transfers']
        if (len(route) != expected or route[0] != 0
                or len(set(route)) != expected or len(moves) != expected-1):
            return False
        day, specials = 0.0, 0
        for i, tr in enumerate(moves):
            dep, tof = float(tr['start_time']), float(tr['tof'])
            if (tr['from'] != route[i] or tr['to'] != route[i+1]
                    or dep < day - 1e-6 or tof <= 0
                    or dep + tof > P.max_time + 1e-6):
                return False
            val = dv(route[i], route[i+1], dep, tof)
            if not math.isfinite(val) or abs(val-float(tr['dv'])) > .1:
                return False
            kind = ('NORMAL' if val < P.dv_threshold else
                    'SPECIAL' if val < P.exception_threshold else 'INVALID')
            if kind == 'INVALID' or tr['status'] != kind:
                return False
            specials += (kind == 'SPECIAL')
            day = dep + tof
        return (specials == s['special_used'] and specials <= P.n_exceptions
                and abs(day-s['current_time']) < 1e-5)
    except Exception:
        return False


def prune(items, first, cap):
    buckets = {}
    for s in items:
        key = (round((s['day']-first)*2), s['used'])
        priority = s['dvsum']*.025 + s['day']
        if key not in buckets or priority < buckets[key][0]:
            buckets[key] = (priority, s)
    chosen = [v[1] for v in buckets.values()]
    chosen.sort(key=lambda s: (
        s['used']*42 + (s['day']-first)*.8 + s['dvsum']*.007,
        s['day']
    ))
    return chosen[:cap]


source_bytes = SOURCE.read_bytes()
source_hash = hashlib.sha256(source_bytes).hexdigest()
original = json.loads(source_bytes)['frontier'][0]
assert verify(original, 32), 'Rekordi 32 nuk kaloi verifikimin!'
missing = [i for i in range(P.n_tomatoes) if i not in original['route']]

# Dritaret jane segmente prej 4-6 transferimesh.
windows = []
transfers = original['transfers']
for length in (5, 4, 6):
    choices = []
    for k in range(len(transfers)-length+1):
        first = (0.0 if k == 0 else
                 transfers[k-1]['start_time'] + transfers[k-1]['tof'])
        last = (transfers[k+length]['start_time']
                if k+length < len(transfers) else P.max_time)
        span = last-first
        freed = sum(tr['status'] == 'SPECIAL'
                    for tr in transfers[k:k+length])
        if span < (length+1)*.25 or freed == 0:
            continue
        score = freed*18 + min(span, 55)*1.6 - .35*k
        choices.append((score, k, length, first, last, freed))
    choices.sort(reverse=True)
    chosen = []
    for w in choices:
        if all(abs(w[1]-old[1]) >= 3 for old in chosen):
            chosen.append(w)
        if len(chosen) == 6:
            break
    windows.extend(chosen)

assert windows, 'Nuk u gjet asnje segment per rindertim.'
# Provimet shperndahen ne te gjitha dritaret, jo ne nje dritare te vetme.
bundles = []
for _, k, length, first, last, freed in windows:
    choices = []
    for tomato in missing:
        for position in range(1, length+1):
            choices.append([k, length, tomato, position, first, last, freed])
    choices.sort(key=lambda t: (
        (t[2]*13 + k*7 + length*11) % P.n_tomatoes,
        (t[3]*7 + k) % (length+1)
    ))
    bundles.append(choices)

tasks = []
for i in range(max(len(b) for b in bundles)):
    for bundle in bundles:
        if i < len(bundle):
            tasks.append(bundle[i])
        if len(tasks) >= 280:
            break
    if len(tasks) >= 280:
        break

config = dict(source_sha256=source_hash, tasks=tasks, version=1)
if CP.exists():
    record = json.loads(CP.read_text(encoding='utf-8'))
    assert record['config'] == config, 'Checkpoint nuk perputhet.'
else:
    record = dict(config=config, done={}, solutions={})
    save(CP, record)


def attempt(task):
    k, length, tomato, position, first, last, freed = task
    old_nodes = original['route'][k:k+length+1]
    new_nodes = old_nodes[:position] + [tomato] + old_nodes[position:]
    fixed = original['special_used'] - freed
    states = [dict(day=first, used=0, dvsum=0., legs=[])]
    calls = 0
    sizes = []
    for step in range(length+1):
        expanded = []
        remaining = length-step
        for s in states:
            for wait in WAITS:
                dep = s['day'] + wait
                for tof in TOFS:
                    arrival = dep + tof
                    if arrival + remaining*.25 > last + 1e-8:
                        continue
                    val = dv(new_nodes[step], new_nodes[step+1], dep, tof)
                    calls += 1
                    if val >= P.exception_threshold:
                        continue
                    kind = 'NORMAL' if val < P.dv_threshold else 'SPECIAL'
                    used = s['used'] + int(kind == 'SPECIAL')
                    if fixed + used > P.n_exceptions:
                        continue
                    leg = dict(
                        **{'from': new_nodes[step], 'to': new_nodes[step+1]},
                        start_time=dep, tof=tof, dv=val, status=kind
                    )
                    expanded.append(dict(day=arrival, used=used,
                                         dvsum=s['dvsum']+val,
                                         legs=s['legs']+[leg]))
            if time.monotonic()-START >= LIMIT:
                return None
        sizes.append(len(expanded))
        if not expanded:
            return dict(calls=calls, layers=sizes, solution=None)
        states = prune(expanded, first, 34 if step < length else 48)
    for s in sorted(states, key=lambda x: (x['used'], x['day'])):
        route = (original['route'][:k] + new_nodes
                 + original['route'][k+length+1:])
        moves = (transfers[:k] + s['legs'] + transfers[k+length:])
        end = s['day'] if k+length == len(transfers) else original['current_time']
        child = dict(route=route, transfers=moves,
                     current_time=end, special_used=fixed+s['used'])
        if verify(child, 33):
            return dict(calls=calls, layers=sizes, solution=child)
    return dict(calls=calls, layers=sizes, solution=None)


print('HAPI 150 - LNS 32 -> 33', flush=True)
print('Rekordi 32: U riverifikua', flush=True)
print('Domate te pavizituara:', len(missing), flush=True)
print('Segmente te perzgjedhura:', len(windows), flush=True)
print('Prova te planifikuara:', len(tasks), flush=True)
print('Checkpoint:', CP.name, flush=True)

for i, task in enumerate(tasks):
    if time.monotonic()-START >= LIMIT or record['solutions']:
        break
    key = str(i)
    if key in record['done']:
        continue
    result = attempt(task)
    if result is None:  # Prove e nderprere: nuk shenohet e perfunduar.
        break
    if result['solution'] is not None:
        record['solutions'][key] = result['solution']
    record['done'][key] = dict(calls=result['calls'], layers=result['layers'],
                               found=result['solution'] is not None)
    save(CP, record)
    if len(record['done']) % 5 == 0 or result['solution']:
        print('Prova {}/{} | segmenti {} me {} kalime | DV teste {} | '
              'shtresa {} | 33={}'.format(
                  len(record['done']), len(tasks), task[0]+1, task[1],
                  result['calls'], result['layers'], len(record['solutions'])
              ), flush=True)

verified = [s for s in record['solutions'].values() if verify(s, 33)]
if verified:
    save(OUT, dict(level=33, verified=True, frontier=verified))

print('\nPERMBLEDHJA E HAPIT 150', flush=True)
print('Prova:', len(record['done']), '/', len(tasks), flush=True)
print('Rruge te verifikuara 33:', len(verified), flush=True)
print('Kerkimi i pjesshem:', bool(not verified and len(record['done']) < len(tasks)), flush=True)
if verified:
    best = min(verified, key=lambda x: (x['special_used'], x['current_time']))
    print('REKORD I RI: 33/49!', flush=True)
    print('SPECIAL:', best['special_used'], '/5', flush=True)
    print('Dita:', best['current_time'], flush=True)
    print('Rruga:', best['route'], flush=True)
else:
    print('Rekordi i ruajtur mbetet: 32/49', flush=True)
