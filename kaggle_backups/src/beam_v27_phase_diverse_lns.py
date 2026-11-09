"""ESA SpOC4 Step 151: phase-diverse adaptive LNS, 32 -> 33.
Does not overwrite the verified 32 route. Checkpoints after each complete task.
"""
import json
import math
import hashlib
import time
from pathlib import Path
from functools import lru_cache
from collections import defaultdict
from utils_users import TomatoProblem

ROOT = Path('/kaggle/working/esa-spoc4')
RESULTS = ROOT / 'results'
SOURCE = RESULTS / 'best_verified_32.json'
CP = RESULTS / 'beam_v27_phase_diverse_lns_checkpoint.json'
OUTPUT = RESULTS / 'beam_v27_phase_diverse_lns_level_33.json'
P = TomatoProblem(str(ROOT / 'data/easy.kttsp'))
START = time.monotonic()
RUN_SECONDS = 108
MAX_TASKS = 260
TASK_CALL_CAP = 48000
BEAM_WIDTH = 52
WAITS = (0, .25, .5, 1, 2, 3, 5, 8, 12, 18)
TOFS = (.25, .5, .75, 1, 1.5, 2, 3, 4, 5, 7, 10, 14, 20, 28, 38)


def atomic(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, separators=(',', ':')), encoding='utf-8')
    tmp.replace(path)


@lru_cache(maxsize=240000)
def dv(a, b, dep, tof):
    try:
        value = float(P.compute_transfer(int(a), int(b), float(dep), float(tof)))
        return value if math.isfinite(value) and value >= 0 else math.inf
    except Exception:
        return math.inf


def verify(s, expected=None):
    try:
        route, trs = s['route'], s['transfers']
        if (not route or route[0] != 0 or len(set(route)) != len(route)
                or len(trs) != len(route) - 1
                or (expected is not None and len(route) != expected)):
            return False
        day, special = 0., 0
        for i, tr in enumerate(trs):
            dep, tof = float(tr['start_time']), float(tr['tof'])
            if (tr['from'] != route[i] or tr['to'] != route[i + 1]
                    or dep < day - 1e-6 or tof <= 0 or dep + tof > P.max_time + 1e-6):
                return False
            val = dv(route[i], route[i+1], dep, tof)
            if not math.isfinite(val) or abs(val - float(tr['dv'])) > 0.1:
                return False
            if val < P.dv_threshold:
                status = 'NORMAL'
            elif val < P.exception_threshold:
                status = 'SPECIAL'
                special += 1
            else:
                return False
            if status != tr['status']:
                return False
            day = dep + tof
        return (special == s['special_used'] <= P.n_exceptions
                and abs(day - s['current_time']) < 1e-5)
    except Exception:
        return False


def make_tasks(base):
    trs = base['transfers']
    windows = []
    # Different segment lengths and locations, not repeated identical searches.
    for length in (5, 6, 7, 8, 9):
        for k in range(len(trs) - length + 1):
            earliest = 0. if k == 0 else float(trs[k-1]['start_time']) + float(trs[k-1]['tof'])
            deadline = (float(trs[k+length]['start_time'])
                        if k+length < len(trs) else float(P.max_time))
            span = deadline-earliest
            freed = sum(t['status'] == 'SPECIAL' for t in trs[k:k+length])
            if freed == 0 or span < (length+1)*.25+1:
                continue
            # Prefer time and SPECIAL flexibility; ensure global coverage.
            score = 25*freed + min(span, 75)*.8 + length*1.2
            windows.append({'k': k, 'length': length, 'earliest': earliest,
                            'deadline': deadline, 'span': span, 'freed': freed,
                            'score': score})
    windows.sort(key=lambda w: (-w['score'], w['k']))
    selected = []
    for w in windows:
        if any(abs(w['k']-v['k']) < 3 and abs(w['length']-v['length']) < 2 for v in selected):
            continue
        selected.append(w)
        if len(selected) >= 18:
            break
    # Add top windows even if clustered so all times have worthwhile candidates.
    for w in windows:
        if len(selected) >= 18:
            break
        if w not in selected:
            selected.append(w)
    missing = [x for x in range(P.n_tomatoes) if x not in base['route']]
    packs = []
    for wi, w in enumerate(selected):
        L, k = w['length'], w['k']
        special_positions = [j+1 for j, tr in enumerate(trs[k:k+L])
                             if tr['status'] == 'SPECIAL']
        positions = sorted(set(special_positions + [1, L//2, L]))
        # Near special maneuvers first; each window tests many tomatoes.
        positions.sort(key=lambda p: (0 if p in special_positions else 1, p))
        pack = []
        missing_order = sorted(missing, key=lambda t: ((t*13 + wi*7) % P.n_tomatoes, t))
        # Each first wave covers every missing tomato; vary insertion and swaps.
        for wave in range(max(1, len(positions))*2):
            for idx, t in enumerate(missing_order):
                pos = positions[(idx + wave) % len(positions)]
                mode = (idx + wave) % 2
                pack.append({'w': wi, 'tomato': t, 'position': pos, 'mode': mode})
        packs.append(pack)
    tasks = []
    # Round-robin prevents repeatedly spending all runtime on one segment.
    for round_index in range(max(map(len, packs), default=0)):
        for pack in packs:
            if round_index < len(pack):
                tasks.append(pack[round_index])
            if len(tasks) >= MAX_TASKS:
                return selected, tasks
    return selected, tasks


def ordered_nodes(base, window, task):
    k, L = window['k'], window['length']
    interior = list(base['route'][k+1:k+L])
    j = int(task['position'])
    # Optional single adjacent swap of original interior visits near insertion.
    if task['mode'] and len(interior) >= 2:
        swap = min(max(j-2, 0), len(interior)-2)
        interior[swap], interior[swap+1] = interior[swap+1], interior[swap]
    insert_at = min(max(j-1, 0), len(interior))
    interior.insert(insert_at, int(task['tomato']))
    return [base['route'][k]] + interior + [base['route'][k+L]]


def keep_diverse(candidates, start_day, cap=BEAM_WIDTH):
    # Bin by arrival time AND special use, keep multiple feasible phases.
    bins = defaultdict(list)
    for s in candidates:
        b = (s['used'], int((s['day']-start_day)/1.25))
        score = 0.006*s['total_dv'] + 0.12*s['day']
        bins[b].append((score, s))
    for values in bins.values():
        values.sort(key=lambda x: x[0])
    groups = defaultdict(list)
    for (used, _), values in sorted(bins.items()):
        groups[used].extend([x[1] for x in values[:2]])
    for used in groups:
        groups[used].sort(key=lambda s: (s['day'] + 0.006*s['total_dv']))
    chosen = []
    for used in sorted(groups):
        group = groups[used]
        if group:
            # Spread the retained phases across the whole feasible time window.
            sample = sorted({round((len(group)-1)*j/7) for j in range(8)})
            chosen.extend(group[j] for j in sample)
    chosen_ids = {id(s) for s in chosen}
    remaining = [s for group in groups.values() for s in group if id(s) not in chosen_ids]
    remaining.sort(key=lambda s: (s['used']*6 + .025*s['day'] + .004*s['total_dv']))
    chosen.extend(remaining[:max(0, cap-len(chosen))])
    chosen.sort(key=lambda s: (s['used'], s['day'], s['total_dv']))
    return chosen[:cap]


def probe_append(base):
    # Fast check: can one NORMAL transfer extend the verified route?
    origin = base['route'][-1]
    arrival = float(base['current_time'])
    missing = [x for x in range(P.n_tomatoes) if x not in base['route']]
    best = None
    tested = 0
    for target in missing:
        for wait in (0, .25, .5, 1, 2, 3, 4, 6, 8, 12, 16):
            dep = arrival+wait
            for tof in (.25, .5, .75, 1, 1.5, 2, 3, 4, 5, 7, 9, 12, 15, 18):
                if dep+tof > P.max_time:
                    continue
                value = dv(origin, target, dep, tof)
                tested += 1
                if not math.isfinite(value):
                    continue
                if best is None or value < best['dv']:
                    best = {'from':origin, 'to':target, 'dv':value,
                            'depart':dep, 'tof':tof}
                if value < P.dv_threshold:
                    tr = {'from':origin, 'to':target, 'start_time':dep,
                          'tof':tof, 'dv':value, 'status':'NORMAL'}
                    s = {'route':base['route']+[target],
                         'transfers':base['transfers']+[tr],
                         'current_time':dep+tof,
                         'special_used':base['special_used']}
                    if verify(s, 33):
                        return s, best, tested
    return None, best, tested


def attempt(base, window, task):
    nodes = ordered_nodes(base, window, task)
    k, L = window['k'], window['length']
    prefix = base['transfers'][:k]
    suffix = base['transfers'][k+L:]
    outside = sum(t['status']=='SPECIAL' for t in prefix+suffix)
    max_special = P.n_exceptions-outside
    deadline = window['deadline']
    earliest = window['earliest']
    frontier = [{'day':earliest, 'used':0, 'total_dv':0., 'new':[]}]
    calls, layers, best_gap = 0, [], float('inf')
    original = base['transfers'][k:k+L]
    for step, (a,b) in enumerate(zip(nodes, nodes[1:])):
        remaining = len(nodes)-step-2
        expanded = []
        # Have a few original departure anchors if the order changed.
        anchor = float(original[min(step, L-1)]['start_time'])
        for s in frontier:
            dep_values = {round(s['day']+w, 6) for w in WAITS}
            dep_values.update(round(x,6) for x in (anchor-.5, anchor, anchor+.5, anchor+1)
                              if x >= s['day']-1e-7)
            for dep in sorted(dep_values):
                if dep < s['day']-1e-6 or dep+(remaining+1)*.25 > deadline+1e-7:
                    continue
                for tof in TOFS:
                    arriving = dep+tof
                    if arriving+remaining*.25 > deadline+1e-7:
                        continue
                    value = dv(a,b,dep,tof)
                    calls += 1
                    if math.isfinite(value):
                        best_gap = min(best_gap, max(0., value-P.exception_threshold))
                    if value >= P.exception_threshold:
                        continue
                    kind = 'NORMAL' if value < P.dv_threshold else 'SPECIAL'
                    used = s['used']+(kind=='SPECIAL')
                    if used > max_special:
                        continue
                    tr = {'from':a, 'to':b, 'start_time':dep, 'tof':tof,
                          'dv':value, 'status':kind}
                    expanded.append({'day':arriving, 'used':used,
                                     'total_dv':s['total_dv']+value,
                                     'new':s['new']+[tr]})
                    if len(expanded) >= 7000:
                        break
                if len(expanded) >= 7000:
                    break
            if len(expanded) >= 7000:
                break
            if calls >= TASK_CALL_CAP:
                break
        layers.append(len(expanded))
        if not expanded:
            return None, calls, layers, best_gap
        frontier = keep_diverse(expanded, earliest)
        if calls >= TASK_CALL_CAP:
            return None, calls, layers, best_gap
    # All original visited plus one new tomato, then reattach unchanged suffix.
    for state in sorted(frontier, key=lambda s: (s['used'], s['day'])):
        new_route = base['route'][:k+1] + nodes[1:-1] + base['route'][k+L:]
        new_transfers = prefix + state['new'] + suffix
        child = {'route':new_route, 'transfers':new_transfers,
                 'special_used':outside+state['used'],
                 'current_time':(float(base['current_time']) if suffix else state['day'])}
        if verify(child, 33):
            return child, calls, layers, best_gap
    return None, calls, layers, best_gap


raw = SOURCE.read_bytes()
base = json.loads(raw)['frontier'][0]
assert verify(base, 32), 'Rekordi i ruajtur 32 nuk u verifikua.'
windows, tasks = make_tasks(base)
assert windows and tasks, 'Nuk u gjeten segmente te pershtatshme.'
config = {'source_sha256': hashlib.sha256(raw).hexdigest(),
          'task_count': len(tasks), 'version': 1,
          'task_call_cap': TASK_CALL_CAP, 'beam': BEAM_WIDTH}
if CP.exists():
    ck = json.loads(CP.read_text(encoding='utf-8'))
    assert ck['config'] == config, 'Checkpoint me parametra te tjere.'
else:
    ck = {'config':config, 'done':{}, 'direct':None,
          'solutions':{}}
    atomic(CP, ck)

print('HAPI 151 - PHASE-DIVERSE ADAPTIVE LNS', flush=True)
print('Rekordi burimor: 32/49 | SPECIAL: 5/5', flush=True)
print('Segmente:', len(windows), '| Prova te planifikuara:', len(tasks), flush=True)

if ck['direct'] is None:
    direct, best, tested = probe_append(base)
    ck['direct'] = {'tested':tested, 'best':best,
                    'found': bool(direct)}
    if direct:
        ck['solutions']['direct'] = direct
    atomic(CP, ck)
    print('Kontroll i fundit 32->33:', tested,
          '| DV minimale:', round(best['dv'],2) if best else 'asnje', flush=True)

for i, task in enumerate(tasks):
    if ck['solutions'] or time.monotonic()-START >= RUN_SECONDS:
        break
    if str(i) in ck['done']:
        continue
    result, calls, layers, gap = attempt(base, windows[task['w']], task)
    ck['done'][str(i)] = {'calls':calls, 'layers':layers,
                          'gap':round(gap,3) if math.isfinite(gap) else None,
                          'found':bool(result)}
    if result:
        ck['solutions'][str(i)] = result
    atomic(CP, ck)
    if len(ck['done']) % 5 == 0 or result:
        print('Prova',len(ck['done']),'/',len(tasks),
              '| Segment',windows[task['w']]['k'],'+',windows[task['w']]['length'],
              '| DV teste',calls,'| shtresa',layers,
              '| 33=',len(ck['solutions']),flush=True)
    if result:
        print('U GJET RRUGE E VLEFSHME ME 33 DOMATE!',flush=True)
        break

valid = [s for s in ck['solutions'].values() if verify(s,33)]
if valid:
    valid.sort(key=lambda s: (s['special_used'],s['current_time']))
    atomic(OUTPUT, {'level':33, 'verified':True, 'frontier':valid})

print('\nPERMBLEDHJA E HAPIT 151',flush=True)
print('Prova:',len(ck['done']),'/',len(tasks),flush=True)
print('Rruge 33 te verifikuara:',len(valid),flush=True)
print('Kerkimi i pjesshem:',not valid and len(ck['done'])<len(tasks),flush=True)
if valid:
    s = valid[0]
    print('REKORD I RI: 33/49! | SPECIAL:',s['special_used'],
          '| Dita:',round(s['current_time'],4),flush=True)
    print('Rruga:',s['route'],flush=True)
else:
    print('Rekordi i ruajtur mbetet: 32/49',flush=True)
