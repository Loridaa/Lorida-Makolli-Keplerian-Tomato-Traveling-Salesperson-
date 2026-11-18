"""ESA SpOC4 -- ALNS V7: walk across verified 39/49 plateaus, then try 40/49.

Stage A: construct *different*, fully verified routes with the same 39 tomatoes
using zero-loss local reorder/retiming. Unlike V6, DO NOT demand <=4 SPECIAL.
Stage B: attempt adding a missing tomato to these alternatives with time/SPECIAL DP.

Safety: preserve all prior files; write new outputs exclusively. Validate every
candidate against official PyKEP physics through the unchanged V4 verifier.

Use Kaggle's PyKEP environment:
  /kaggle/working/esa-pykep-env/bin/python -u src/alns_v7_neutral_walk.py --check-only
  /kaggle/working/esa-pykep-env/bin/python -u src/alns_v7_neutral_walk.py
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import alns_v4_dp_repair as v4

BASE38 = ROOT / 'results/best_verified_38.json'
BACKUP38 = ROOT / 'kaggle_backups/results/best_verified_38.json'
BASE39 = ROOT / 'results/best_verified_39.json'
REPORT = ROOT / 'results/alns_v7_neutral_walk_report.json'
ALT39 = ROOT / 'results/alns_v7_verified_39_neutral.json'
CAND40 = ROOT / 'results/alns_v7_verified_40_candidate.json'

SEED = 17640
MAX_NEUTRAL_ATTEMPTS = 125
MAX_INSERT_ATTEMPTS = 260
MAX_VARIANTS = 12
MAX_EVALUATIONS = 175_000
MIN_TOF = 0.5
RNG = random.Random(SEED)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_baseline():
    for path in (BASE38, BACKUP38, BASE39):
        if not path.is_file():
            raise RuntimeError(f'Mungon rekordi: {path}')
    sha38, sha39 = sha(BASE38), sha(BASE39)
    if sha(BACKUP38) != sha38:
        raise RuntimeError('NDAL: rekordi 38 nuk perputhet me kopjen rezervë')
    p = v4.load_problem()
    if p.n_tomatoes != 49:
        raise RuntimeError('Instanca nuk ka 49 domate')
    states = []
    for path, level in ((BASE38, 38), (BASE39, 39)):
        d = json.loads(path.read_text(encoding='utf-8'))
        if d.get('level') != level or d.get('verified') is not True or len(d.get('frontier', [])) != 1:
            raise RuntimeError(f'Formati i gabuar i rekordit {level}')
        s = d['frontier'][0]
        if len(s.get('route', [])) != level or not v4.verify(p, s):
            raise RuntimeError(f'Rekordi {level} nuk e kaloi testin orbital')
        states.append(s)
    s38, s39 = states
    if not set(s38['route']).issubset(set(s39['route'])):
        raise RuntimeError('Rruga 39 nuk perfshin 38 domatet e meparshme')
    if int(s39['special_used']) != 5:
        raise RuntimeError('V7 eshte pergatitur per 5 SPECIAL')
    return p, s39, {'38': sha38, '39': sha39}


def frame(state, i, end):
    route, legs = state['route'], state['transfers']
    n = len(route)
    if not (1 <= i <= n and i-1 <= end <= n-1):
        raise ValueError('Dritare e pavlefshme')
    prefix = legs[:i-1]
    suffix = legs[end:] if end < n-1 else []
    t0 = float(prefix[-1]['start_time'] + prefix[-1]['tof']) if prefix else 0.0
    deadline = float(suffix[0]['start_time']) if suffix else 200.0
    oldwindow = legs[i-1:end]
    reserved = sum(leg['status'] == 'SPECIAL' for leg in prefix + suffix)
    return t0, deadline, oldwindow, reserved, prefix, suffix


def stitched(route, i, end, targets, repaired, prefix, suffix):
    newroute = list(route[:i]) + list(targets) + list(route[end+1:])
    transfers = list(prefix) + list(repaired['legs']) + list(suffix)
    if len(transfers) != len(newroute)-1:
        raise RuntimeError('Gjatesi e gabuar e transferimeve')
    final = transfers[-1]
    return {'route': newroute,
            'transfers': transfers,
            'special_used': sum(leg['status'] == 'SPECIAL' for leg in transfers),
            'current_time': float(final['start_time'] + final['tof'])}


def mutate(items, operator):
    result = list(items)
    if operator == 'keep' or len(result) < 2:
        return result
    center = len(result)//2
    if operator == 'swap':
        j = min(center, len(result)-2)
        result[j], result[j+1] = result[j+1], result[j]
    elif operator == 'reverse3':
        j = max(0, center-1)
        result[j:j+3] = reversed(result[j:j+3])
    elif operator == 'rotate4':
        j = max(0, center-2)
        block = result[j:j+4]
        result[j:j+4] = block[1:] + block[:1]
    elif operator == 'relocate':
        j = max(0, center-1)
        item = result.pop(j)
        result.insert(min(len(result), j+3), item)
    else:
        raise ValueError('Operator i panjohur')
    return result


def neutral_plans(base):
    route, legs = base['route'], base['transfers']
    n = len(route)
    specials = [i+1 for i, leg in enumerate(legs) if leg['status'] == 'SPECIAL']
    plans = defaultdict(list)
    seen = set()
    for i in range(1, n):
        for width in (2, 3, 5, 7, 11, 17, 25):
            end = min(n-1, i+width-1)
            start, deadline, old, reserved, _, _ = frame(base, i, end)
            if deadline - start < MIN_TOF*len(old)-1e-6:
                continue
            inwindow = sum(x['status'] == 'SPECIAL' for x in old)
            nearest = min(abs(i-anchor) for anchor in specials)
            slack = max(0, deadline-start-MIN_TOF*len(old))
            for op in ('keep', 'swap', 'reverse3', 'rotate4', 'relocate'):
                targets = mutate(route[i:end+1], op)
                key = (i, end, tuple(targets))
                if key in seen:
                    continue
                seen.add(key)
                # Favor windows spanning SPECIAL but cover distant segments too.
                score = (18*min(2, inwindow) + 3*min(12, slack/len(old))
                         - .75*len(old) - .5*nearest + (1.5 if op == 'keep' else 0))
                plans[i].append((score, i, end, targets, op))
    for group in plans.values():
        RNG.shuffle(group)
        group.sort(key=lambda x: -x[0])
    # Diversify cuts instead of spending all evaluations around one SPECIAL.
    chosen = []
    cuts = list(plans)
    RNG.shuffle(cuts)
    for pass_id in range(8):
        if pass_id % 2 == 0:
            cuts.sort(key=lambda i: min(abs(i-s) for s in specials))
        else:
            cuts.sort(key=lambda i: -i)
        for i in cuts:
            group = plans[i]
            if len(group) <= pass_id:
                continue
            chosen.append(group[pass_id])
            if len(chosen) == MAX_NEUTRAL_ATTEMPTS:
                return chosen
    return chosen


def fingerprint(state):
    return (tuple(state['route']),
            tuple((round(float(t['start_time']), 3), round(float(t['tof']), 3),
                   t['status']) for t in state['transfers']))


def neutral_search(p, base, evaluator):
    attempts = 0
    repairs = 0
    failures = Counter()
    closest = {'legs_repaired': 0, 'legs_needed': 0}
    variants = []
    seen = {fingerprint(base)}
    for _, i, end, targets, op in neutral_plans(base):
        start, deadline, old, reserved, prefix, suffix = frame(base, i, end)
        cap = p.n_exceptions - reserved
        if cap < 0:
            continue
        attempts += 1
        repaired, reached, reason, edge = v4.repair(p, evaluator, base['route'][i-1],
                                                      targets, start, deadline, cap, old)
        if reached > closest['legs_repaired']:
            closest = {'attempt': attempts, 'cut': i, 'end': end, 'operator': op,
                       'legs_repaired': reached, 'legs_needed': len(targets),
                       'failed_edge': edge}
        if repaired is None:
            failures[reason or 'unknown'] += 1
            continue
        repairs += 1
        new = stitched(base['route'], i, end, targets, repaired, prefix, suffix)
        if (len(new['route']) != 39 or set(new['route']) != set(base['route'])
                or not v4.verify(p, new)):
            failures['full_verification'] += 1
            continue
        fp = fingerprint(new)
        if fp in seen:
            failures['identical_solution'] += 1
            continue
        seen.add(fp)
        variants.append((new, {'cut': i, 'end': end, 'op': op}))
        print('Alternativa 39 e verifikuar:', len(variants),
              '| SPECIAL', new['special_used'], '| dita',
              round(new['current_time'], 3), flush=True)
        if len(variants) >= MAX_VARIANTS:
            break
    return variants, attempts, repairs, failures, closest


def insertion_plans(state, context, unused, limit):
    route = state['route']
    n = len(route)
    plans = []
    seen = set()
    for i in range(1, n+1):
        for width in (1, 2, 4, 7, 12, 19, 27):
            if i == n and width != 1:
                continue
            end = min(n-1, i+width-1)
            start, deadline, old, reserved, _, _ = frame(state, i, end)
            count = len(old)+1
            slack = deadline-start-count*MIN_TOF
            if slack < -1e-6:
                continue
            distance = abs(i-context['cut'])
            inwindow = sum(t['status'] == 'SPECIAL' for t in old)
            for tomato in unused:
                # Across rounds try beginning, middle and end of the window.
                offset = (tomato*3 + i + end) % (len(old)+1)
                key = (i, end, tomato, offset)
                if key in seen:
                    continue
                seen.add(key)
                score = (13*min(2, inwindow) + 5*min(8, slack/max(1, count))
                         - .6*count - .7*distance)
                plans.append((score, i, end, tomato, offset))
    RNG.shuffle(plans)
    plans.sort(key=lambda x: -x[0])
    # One try per tomato in each pass, to ensure coverage.
    selected = []
    already = set()
    for _ in range(5):
        used_tomatoes = set()
        for candidate in plans:
            _, i, end, tomato, offset = candidate
            key = (i, end, tomato, offset)
            if key in already or tomato in used_tomatoes:
                continue
            already.add(key)
            used_tomatoes.add(tomato)
            selected.append(candidate)
            if len(selected) >= limit:
                return selected
    return selected


def insert_search(p, variants, evaluator, base):
    unused = sorted(set(range(p.n_tomatoes)) - set(base['route']))
    attempts = 0
    full_repairs = 0
    failures = Counter()
    closest = {'legs_repaired': 0, 'legs_needed': 0}
    for idx, (state, context) in enumerate(variants):
        remaining_variants = len(variants) - idx
        allotted = max(10, (MAX_INSERT_ATTEMPTS-attempts)//remaining_variants)
        for _, i, end, tomato, offset in insertion_plans(state, context, unused, allotted):
            if attempts >= MAX_INSERT_ATTEMPTS:
                break
            start, deadline, old, reserved, prefix, suffix = frame(state, i, end)
            cap = p.n_exceptions-reserved
            if cap < 0:
                continue
            targets = state['route'][i:end+1]
            targets = list(targets)
            targets.insert(offset, tomato)
            attempts += 1
            repaired, reached, reason, edge = v4.repair(
                p, evaluator, state['route'][i-1], targets, start, deadline, cap, old)
            if reached > closest['legs_repaired']:
                closest = {'variant': idx+1, 'attempt': attempts,
                           'cut': i, 'end': end, 'missing_tomato': tomato,
                           'legs_repaired': reached, 'legs_needed': len(targets),
                           'failed_edge': edge}
            if repaired is None:
                failures[reason or 'unknown'] += 1
                continue
            full_repairs += 1
            proposal = stitched(state['route'], i, end, targets, repaired, prefix, suffix)
            if (len(proposal['route']) == 40 and len(set(proposal['route'])) == 40
                    and v4.verify(p, proposal)):
                return proposal, {'variant': idx+1, 'tomato': tomato,
                                  'cut': i, 'end': end, 'offset': offset}, attempts, full_repairs, failures, closest
            failures['full_verification'] += 1
        if attempts >= MAX_INSERT_ATTEMPTS:
            break
    return None, None, attempts, full_repairs, failures, closest


def save_exclusive(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print('U ruajt:', path.name, flush=True)


def main():
    p, base, fingerprints = load_baseline()
    print('BAZA:', len(base['route']), '/49 | dita:',
          round(base['current_time'], 6), '| SPECIAL:', base['special_used'], flush=True)
    print('Domatet qe mungojne:', sorted(set(range(p.n_tomatoes))-set(base['route'])), flush=True)
    if '--check-only' in sys.argv:
        print('CHECK-ONLY OK. Asnje skedar nuk u ndryshua.', flush=True)
        return
    if REPORT.exists() or ALT39.exists() or CAND40.exists():
        raise RuntimeError('Rezultat V7 ekziston. Nuk mbishkruajme.')
    v4.MAX_EVALUATIONS = MAX_EVALUATIONS
    evaluator = v4.Evaluator(p)
    wallstart = time.monotonic()
    hit_limit = False
    alternatives = []
    neutral_attempts = neutral_repairs = 0
    insertion_attempts = insertion_repairs = 0
    neutral_failure = Counter()
    insertion_failure = Counter()
    near_neutral = {'legs_repaired': 0, 'legs_needed': 0}
    near_insertion = {'legs_repaired': 0, 'legs_needed': 0}
    found40 = context40 = None
    try:
        (alternatives, neutral_attempts, neutral_repairs,
         neutral_failure, near_neutral) = neutral_search(p, base, evaluator)
        if alternatives:
            print('Gjetem', len(alternatives), 'rruge alternative 39; provojme 40...', flush=True)
            (found40, context40, insertion_attempts, insertion_repairs,
             insertion_failure, near_insertion) = insert_search(p, alternatives, evaluator, base)
        else:
            print('Asnje rruge alternative 39; nuk po perserisim V5.', flush=True)
    except v4.BudgetReached:
        hit_limit = True
        print('U arrit kufiri orbital', MAX_EVALUATIONS, flush=True)
    if (sha(BASE38) != fingerprints['38'] or sha(BACKUP38) != fingerprints['38']
            or sha(BASE39) != fingerprints['39']):
        raise RuntimeError('NDAL: nje rekord baze ka ndryshuar!')
    if alternatives:
        best, ctx = min(alternatives, key=lambda a: (a[0]['special_used'], a[0]['current_time']))
        if not v4.verify(p, best):
            raise RuntimeError('Alternativa 39 nuk u verifikua')
        save_exclusive(ALT39, {'level': 39, 'verified': True,
                                'source': 'alns_v7_neutral_walk.py', 'frontier': [best]})
    if found40 is not None:
        if not v4.verify(p, found40):
            raise RuntimeError('Kandidati 40 nuk u verifikua')
        save_exclusive(CAND40, {'level': 40, 'verified': True,
                                 'source': 'alns_v7_neutral_walk.py', 'frontier': [found40]})
    report = {
        'algorithm': 'ALNS V7 plateau walk, neutral 39 -> 40 insertion',
        'base_level': 39,
        'best_level': 40 if found40 is not None else 39,
        'candidate_verified': found40 is not None,
        'alternative_39_verified': bool(alternatives),
        'neutral_variants': len(alternatives),
        'neutral_attempts': neutral_attempts,
        'neutral_repairs': neutral_repairs,
        'neutral_failure_reasons': dict(neutral_failure),
        'nearest_neutral': near_neutral,
        'insertion_attempts': insertion_attempts,
        'full_insertion_repairs': insertion_repairs,
        'insertion_failure_reasons': dict(insertion_failure),
        'nearest_insertion': near_insertion,
        'transfer_evaluations': evaluator.count,
        'evaluation_limit': MAX_EVALUATIONS,
        'evaluation_limit_reached': hit_limit,
        'candidate_context': context40,
        'protected_38_sha256': fingerprints['38'],
        'protected_39_sha256': fingerprints['39'],
        'seed': SEED,
        'elapsed_seconds': round(time.monotonic()-wallstart, 2)
    }
    save_exclusive(REPORT, report)
    print('ALNS V7:', report['best_level'], '/49 | variante 39:', len(alternatives),
          '| eval:', evaluator.count, '| kandidat 40:', found40 is not None, flush=True)
    print('Rekordet 38/39: TE PAPREKURA', flush=True)


if __name__ == '__main__':
    main()
