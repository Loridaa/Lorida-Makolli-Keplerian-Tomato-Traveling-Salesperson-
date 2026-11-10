"""ESA SpOC4 - Target-first single-edge insertion, PyKEP 2.x.
Run via Kaggle's Python 3.11/PyKEP environment.
"""
from pathlib import Path
import json, math, time, hashlib
from scipy.optimize import differential_evolution
from utils_users import TomatoProblem

ROOT = Path('/kaggle/working/esa-spoc4')
RES = ROOT / 'results'
SAFE = RES / 'best_verified_38.json'
ALT = RES / 'beam_v43_alternatives_38_level_38.json'
CP = RES / 'beam_v48_target_first_checkpoint.json'
OUT = RES / 'best_verified_39.json'
P = TomatoProblem(str(ROOT / 'data/easy.kttsp'))
RUN_SECONDS = 105
MAXITER = 9
POP = 5
START = time.monotonic()

params = next(line.split() for line in (ROOT / 'data/easy.kttsp').read_text().splitlines()
              if line.lstrip().startswith('p '))
T0 = float(params[2])
MIN_TOF = max(float(params[3]), 0.00001)
assert P.n_tomatoes == 49 and abs(P.max_time - 200) < 0.01


def atomic(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, separators=(',', ':')))
    tmp.replace(path)


def calc(a, b, d, f):
    try:
        v = float(P.compute_transfer(int(a), int(b), float(d), float(f)))
        return v if math.isfinite(v) and v >= 0 else math.inf
    except Exception:
        return math.inf


def label(v):
    if v < P.dv_threshold:
        return 'NORMAL'
    if v < P.exception_threshold:
        return 'SPECIAL'
    return 'INVALID'


def verify(state):
    route, legs = state['route'], state['transfers']
    if (not route or route[0] != 0 or len(set(route)) != len(route)
            or len(legs) != len(route) - 1 or len(route) > P.n_tomatoes):
        return False
    day, special = T0, 0
    for i, tr in enumerate(legs):
        d, f = float(tr['start_time']), float(tr['tof'])
        if (int(tr['from']) != route[i] or int(tr['to']) != route[i + 1]
                or d < day - 1e-6 or f < MIN_TOF - 1e-7
                or d + f > P.max_time + 1e-6):
            return False
        v = calc(route[i], route[i + 1], d, f)
        status = label(v)
        if (status == 'INVALID' or status != tr['status']
                or abs(v - float(tr['dv'])) > 0.1):
            return False
        special += status == 'SPECIAL'
        day = d + f
    return (special <= P.n_exceptions
            and special == state['special_used']
            and abs(day - state['current_time']) < 0.001)


raw = json.loads(SAFE.read_text())['frontier'][0]
alts = json.loads(ALT.read_text())['frontier']
bases = []
seen = set()
for item in [raw] + alts:
    sig = tuple(item['route'])
    if sig not in seen and len(sig) == 38 and verify(item):
        seen.add(sig)
        bases.append(item)
assert len(bases) >= 1, 'Nuk ka baze 38 te verifikuar.'

fingerprint = hashlib.sha256(SAFE.read_bytes() + ALT.read_bytes()).hexdigest()
if CP.exists():
    cp = json.loads(CP.read_text())
    assert cp['fingerprint'] == fingerprint, 'Burimet kane ndryshuar; mos perdor checkpoint-in e vjeter.'
else:
    cp = {'fingerprint': fingerprint, 'next': 0, 'best_ratio': 999.0,
          'best_task': None, 'solved': False}

tasks = []
for bi, base in enumerate(bases):
    nodes, legs = base['route'], base['transfers']
    missing = sorted(set(range(P.n_tomatoes)) - set(nodes))
    ranked = []
    for i in range(len(legs)):
        left = T0 if i == 0 else float(legs[i - 1]['start_time']) + float(legs[i - 1]['tof'])
        right = (float(legs[i + 1]['start_time']) if i + 1 < len(legs)
                 else float(P.max_time))
        span = right - left
        if span >= 2 * MIN_TOF + .01:
            ranked.append((span, i))
    ranked.sort(reverse=True)
    tasks.append((bi, missing, [i for _, i in ranked]))

# Interleave tomatoes and bases: avoid spending an entire session on one target.
ordered = []
max_rank = max(len(row[2]) for row in tasks)
for rank in range(max_rank):
    for j in range(12):
        for bi, missing, ranked in tasks:
            if rank < len(ranked) and j < len(missing):
                ordered.append((bi, ranked[rank], missing[j]))


def one_trial(bi, edge, target, serial):
    base = bases[bi]
    nodes, legs = base['route'], base['transfers']
    a, b = nodes[edge], nodes[edge + 1]
    left = (T0 if edge == 0 else
            float(legs[edge - 1]['start_time']) + float(legs[edge - 1]['tof']))
    right = (float(legs[edge + 1]['start_time']) if edge + 1 < len(legs)
             else float(P.max_time))
    slack = right - left - 2 * MIN_TOF
    old_special = legs[edge]['status'] == 'SPECIAL'
    best = {'ratio': math.inf}

    def decode(x):
        # Five nonnegative intervals: wait, fly1, wait, fly2, wait.
        expv = [math.exp(float(z)) for z in x]
        total = sum(expv)
        w = [slack * z / total for z in expv]
        d1 = left + w[0]
        f1 = MIN_TOF + w[1]
        d2 = d1 + f1 + w[2]
        f2 = MIN_TOF + w[3]
        return d1, f1, d2, f2

    def objective(x):
        d1, f1, d2, f2 = decode(x)
        v1 = calc(a, target, d1, f1)
        v2 = calc(target, b, d2, f2)
        if old_special:
            ratio = min(max(v1 / P.dv_threshold, v2 / P.exception_threshold),
                        max(v1 / P.exception_threshold, v2 / P.dv_threshold))
        else:
            ratio = max(v1 / P.dv_threshold, v2 / P.dv_threshold)
        if ratio < best['ratio']:
            best.update(ratio=ratio, d1=d1, f1=f1, d2=d2, f2=f2,
                        v1=v1, v2=v2)
        return ratio if math.isfinite(ratio) else 1e8

    def early_stop(_x, convergence=None):
        return best['ratio'] < 0.999999

    differential_evolution(objective, [(-5, 5)] * 5,
                           maxiter=MAXITER, popsize=POP, polish=False,
                           seed=172000 + serial, workers=1,
                           callback=early_stop, tol=0.01)
    if 1 <= best['ratio'] < 1.3:
        # Refine only promising insertion candidates.
        differential_evolution(objective, [(-5, 5)] * 5,
                               maxiter=22, popsize=7, polish=False,
                               seed=272000 + serial, workers=1,
                               callback=early_stop, tol=0.005)
    if best['ratio'] >= 1:
        return best['ratio'], None

    t1 = {'from': a, 'to': target, 'start_time': best['d1'],
          'tof': best['f1'], 'dv': best['v1'], 'status': label(best['v1'])}
    t2 = {'from': target, 'to': b, 'start_time': best['d2'],
          'tof': best['f2'], 'dv': best['v2'], 'status': label(best['v2'])}
    candidate = {
        'route': nodes[:edge + 1] + [target] + nodes[edge + 1:],
        'transfers': legs[:edge] + [t1, t2] + legs[edge + 1:],
        'special_used': (base['special_used'] - int(old_special)
                         + int(t1['status'] == 'SPECIAL')
                         + int(t2['status'] == 'SPECIAL')),
        'current_time': (base['current_time'] if edge < len(legs) - 1
                         else best['d2'] + best['f2'])
    }
    return best['ratio'], candidate if verify(candidate) and len(candidate['route']) == 39 else None


print('HAPI 172 - TARGET-FIRST DIRECT INSERTION', flush=True)
print(f'Baza: {len(bases)} | Prova: {len(ordered)} | Te kryera: {cp["next"]}', flush=True)
print(f'Maxiter: {MAXITER} | Pop: {POP} | Rekordi 38 i paprekur', flush=True)

while (cp['next'] < len(ordered) and not cp['solved']
       and time.monotonic() - START < RUN_SECONDS):
    ix = cp['next']
    bi, edge, tomato = ordered[ix]
    ratio, solution = one_trial(bi, edge, tomato, ix)
    if ratio < cp['best_ratio']:
        cp['best_ratio'] = round(float(ratio), 9)
        cp['best_task'] = {'base': bi, 'edge': edge, 'tomato': tomato}
    cp['next'] += 1
    if solution is not None:
        if OUT.exists():
            existing = json.loads(OUT.read_text())['frontier'][0]
            assert verify(existing), 'Skedari ekzistues 39 nuk kalon verifikimin.'
        else:
            existing = None
        if existing is None or solution['current_time'] < existing['current_time']:
            atomic(OUT, {'frontier': [solution], 'source': 'beam_v48_target_first',
                         'verification': 'pykep_full_transfer_check'})
        cp['solved'] = True
        atomic(CP, cp)
        print(f'SUKSES! 39/49 | Dita {solution["current_time"]:.6f} | '
              f'SPECIAL {solution["special_used"]}/5', flush=True)
        break
    if cp['next'] % 4 == 0:
        atomic(CP, cp)
        print(f'Prova {cp["next"]}/{len(ordered)} | baza {bi} | '
              f'{bases[bi]["route"][edge]} -> {tomato} -> '
              f'{bases[bi]["route"][edge + 1]} | '
              f'raporti {ratio:.3f} | best {cp["best_ratio"]:.3f}', flush=True)
atomic(CP, cp)
print('\nPERMBLEDHJA E HAPIT 172', flush=True)
print(f'Prova: {cp["next"]}/{len(ordered)}', flush=True)
print(f'Raporti me i mire i pragut DV: {cp["best_ratio"]}', flush=True)
print(f'39/49: {"PO" if cp["solved"] else "JO"}', flush=True)
print(f'Kerkimi i pjesshem: {cp["next"] < len(ordered) and not cp["solved"]}', flush=True)
