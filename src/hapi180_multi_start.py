
import json, os, math, time, random, sys
from functools import lru_cache
from pathlib import Path

ROOT = Path('/kaggle/working/esa-spoc4')
R = ROOT/'results'
sys.path.insert(0, str(ROOT/'src'))
from utils_users import TomatoProblem

param = next(
    line.split()
    for line in (ROOT/'data/easy.kttsp').read_text().splitlines()
    if line.startswith('p kttsp ')
)

t0, min_tof, end, normal, special, max_spec = map(
    float, param[2:8]
)
max_spec = int(max_spec)
P = TomatoProblem(str(ROOT/'data/easy.kttsp'))

@lru_cache(maxsize=300000)
def calc(a, b, dep, tof):
    try:
        v = float(P.compute_transfer(
            int(a), int(b), float(dep), float(tof)
        ))
        return v if math.isfinite(v) and v >= 0 else math.inf
    except (RuntimeError, ValueError, OverflowError, ZeroDivisionError):
        return math.inf

def atomic(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False),
        encoding='utf-8'
    )
    os.replace(temp, path)

def as_state(record):
    if 'frontier' in record:
        record = record['frontier'][0]

    rt = list(map(int, record['route']))
    legs = list(record['transfers'])

    t = (
        float(legs[-1]['start_time']) + float(legs[-1]['tof'])
        if legs else t0
    )
    sp = sum(float(l['dv']) > normal + 1e-6 for l in legs)

    return {
        'route': rt,
        'transfers': legs,
        't': t,
        'sp': int(sp),
        'mask': sum(1 << x for x in rt)
    }

def valid(s):
    r = s['route']
    legs = s['transfers']

    if (
        not r or r[0] != 0
        or len(set(r)) != len(r)
        or len(legs) != len(r)-1
        or max(r) > 48 or min(r) < 0
    ):
        return False

    if s['sp'] > max_spec or s['t'] > end + 1e-6:
        return False

    clock = t0
    nsp = 0

    for i, l in enumerate(legs):
        a, b = int(l['from']), int(l['to'])
        dep = float(l['start_time'])
        tof = float(l['tof'])

        if (
            (a, b) != (r[i], r[i+1])
            or dep < clock - 1e-6
            or tof < min_tof - 1e-6
        ):
            return False

        d = calc(a, b, round(dep, 6), round(tof, 6))

        if not math.isfinite(d) or d > special + 1e-6:
            return False

        if abs(d - float(l['dv'])) > max(.003, 2e-5*abs(d)):
            return False

        nsp += int(d > normal + 1e-6)
        clock = dep + tof

    return (
        nsp == s['sp']
        and abs(clock - s['t']) < 1e-5
        and clock <= end + 1e-6
    )

seeds = []

for p in [
    R/'best_verified_38.json',
    R/'hapi179_suffix_best.json',
    R/'hapi180_multi_best.json'
]:
    if p.is_file():
        try:
            s = as_state(json.loads(p.read_text()))
            if valid(s):
                seeds.append(s)
        except (ValueError, KeyError, IndexError, TypeError):
            pass

assert seeds, 'Nuk u gjet baze e verifikuar'

best = max(
    seeds,
    key=lambda s: (len(s['route']), -s['t'])
)

print('HAPI 180 - MULTI-START / PHASE DIVERSITY', flush=True)
print(
    'Baza:', len(best['route']), '/49',
    '| SPECIAL', best['sp'],
    '| Dita', round(best['t'], 5),
    flush=True
)
print('Provojme nga nivelet 38, 35 dhe 32.', flush=True)

waits = [0, .1, .25, .5, 1, 1.5, 2, 3, 4, 5, 7, 10, 14, 18]
tofs = [
    x for x in
    [.1, .2, .35, .5, .75, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 16]
    if x >= min_tof
]

fixed = [(w, f) for w in waits for f in tofs]

T0 = time.monotonic()
deadline = T0 + 210
rng = random.Random(180260)
W = 65

progress_path = R/'hapi180_multi_best.json'

def extend(s, j, dep, tof, d):
    sp = int(d > normal + 1e-6)

    leg = {
        'from': s['route'][-1],
        'to': j,
        'start_time': dep,
        'tof': tof,
        'dv': d,
        'status': 'SPECIAL' if sp else 'NORMAL'
    }

    return {
        'route': s['route'] + [j],
        'transfers': s['transfers'] + [leg],
        't': dep + tof,
        'sp': s['sp'] + sp,
        'mask': s['mask'] | (1 << j)
    }

def promote(s):
    global best

    if (
        len(s['route']), -s['t']
    ) <= (
        len(best['route']), -best['t']
    ):
        return

    if not valid(s):
        return

    best = s

    atomic(progress_path, {
        'verified': True,
        'method': 'hapi180_multi_start',
        'route': s['route'],
        'transfers': s['transfers'],
        'special_used': s['sp'],
        'current_time': s['t'],
        'remaining': sorted(
            set(range(49)) - set(s['route'])
        )
    })

    print(
        '*** PROGRES:',
        len(s['route']), '/49',
        '| SPECIAL', s['sp'],
        '| DITA', round(s['t'], 5),
        '***',
        flush=True
    )

for cut in [38, 35, 32]:

    if time.monotonic() >= deadline - 4:
        break

    if len(best['route']) == 49:
        break

    pool = []

    for z in seeds:
        if len(z['route']) >= cut:
            pool.append(as_state({
                'route': z['route'][:cut],
                'transfers': z['transfers'][:cut-1]
            }))

    if len(best['route']) >= cut:
        pool.append(as_state({
            'route': best['route'][:cut],
            'transfers': best['transfers'][:cut-1]
        }))

    pool = list({
        tuple(p['route']): p for p in pool
    }.values())

    cp = R/f'hapi180_checkpoint_{cut}.json'

    if cp.is_file():
        try:
            old = json.loads(cp.read_text())

            if (
                old.get('version') == 180
                and old.get('cut') == cut
                and old.get('width') == W
            ):
                restored = old['frontier']

                if restored and all(valid(s) for s in restored):
                    pool = restored
                    print(
                        'Checkpoint rikthyer nga niveli',
                        len(pool[0]['route']),
                        flush=True
                    )
        except (ValueError, KeyError, TypeError):
            pass

    print(
        '\nNIS NGA NIVELI', cut,
        '| fillim', len(pool), 'rruge',
        flush=True
    )

    while pool and time.monotonic() < deadline - 4:

        depth = len(pool[0]['route'])

        if depth >= 49:
            break

        proposals = []
        expired = False

        for s in pool:

            if time.monotonic() > deadline - 4:
                expired = True
                break

            rem = [
                j for j in range(49)
                if not ((s['mask'] >> j) & 1)
            ]

            for j in rem:

                room = max(0, end - s['t'] - min_tof)

                options = fixed + [
                    (
                        rng.random() * room,
                        min_tof + rng.random() * room
                    )
                    for _ in range(32)
                ]

                good = []

                for wait, tof in options:

                    dep = round(s['t'] + wait, 6)
                    tof = round(tof, 6)

                    if tof < min_tof or dep + tof > end + 1e-6:
                        continue

                    d = calc(s['route'][-1], j, dep, tof)

                    if (
                        d <= normal + 1e-6
                        or (
                            d <= special + 1e-6
                            and s['sp'] < max_spec
                        )
                    ):
                        good.append((dep + tof, dep, tof, d))

                good.sort(key=lambda x: x[0])
                chosen = []

                for x in good:
                    if all(
                        abs(x[0] - y[0]) > .75
                        for y in chosen
                    ):
                        chosen.append(x)

                    if len(chosen) == 2:
                        break

                for arr, dep, tof, d in chosen:
                    proposals.append(
                        extend(s, j, dep, tof, d)
                    )

        if expired:
            print(
                'Kufiri kohor; checkpoint i fundit ruhet.',
                flush=True
            )
            break

        if not proposals:
            print(
                'Niveli', depth,
                ': nuk pati kalim. Provojme prefiks tjeter.',
                flush=True
            )
            break

        proposals.sort(
            key=lambda s: (s['t'] + 4.0*s['sp'], s['t'])
        )

        next_pool = []
        bins = set()

        for s in proposals:
            key = (
                s['route'][-1],
                s['sp'],
                int(s['t'] / 1.25)
            )

            if key in bins:
                continue

            bins.add(key)
            next_pool.append(s)

            if len(next_pool) >= W:
                break

        pool = next_pool

        for s in pool:
            promote(s)

        atomic(cp, {
            'version': 180,
            'width': W,
            'cut': cut,
            'frontier': pool
        })

        print(
            'Niveli', depth, '->', depth+1,
            '| beam', len(pool),
            '| best', len(best['route']), '/49',
            '| sek', round(time.monotonic()-T0, 1),
            flush=True
        )

        if len(best['route']) == 49:
            break

atomic(R/'hapi180_multi_status.json', {
    'best': len(best['route']),
    'day': best['t'],
    'special': best['sp'],
    'duration': round(time.monotonic()-T0, 1)
})

print(
    '\nPERMBLEDHJE HAPI 180:',
    len(best['route']), '/49',
    '| SPECIAL', best['sp'],
    '| DITA', round(best['t'], 6),
    flush=True
)

print(
    'Rekordi origjinal 38 nuk eshte ndryshuar.',
    flush=True
)
