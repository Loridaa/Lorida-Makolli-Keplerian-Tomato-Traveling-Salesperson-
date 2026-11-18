"""ESA SpOC4 | V12 direct 1-for-2 exchange (39 -> 40).

V11 could not find a feasible intermediate 39-tomato exchange and therefore
skipped its 40-tomato insertion stage. V12 removes that prerequisite: replace
ONE currently visited tomato by TWO missing tomatoes in the same retimed
window, so the number visited increases immediately from 39 to 40.

Uses existing real PyKEP-backed loader, bounded evaluator, multi-label temporal
DP and final whole-route verification. Never edits existing records/results.

Run in Kaggle:
  /kaggle/working/esa-pykep-env/bin/python -u src/alns_v12_direct_1for2.py --check-only
  /kaggle/working/esa-pykep-env/bin/python -u src/alns_v12_direct_1for2.py
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import hashlib
import json
import math
import os
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import alns_v4_dp_repair as v4  # noqa: E402
import alns_v9_multistart as v9  # noqa: E402
import alns_v10_temporal_labels as v10  # noqa: E402

BASE38 = ROOT / 'results/best_verified_38.json'
BACKUP38 = ROOT / 'kaggle_backups/results/best_verified_38.json'
BASE39 = ROOT / 'results/best_verified_39.json'
ALT39 = ROOT / 'results/alns_v7_verified_39_neutral.json'
ARCHIVE39 = ROOT / 'results/alns_v9_multistart_verified_39_archive.json'
V11_REPORT = ROOT / 'results/alns_v11_exchange_insert_report.json'
REPORT = ROOT / 'results/alns_v12_direct_1for2_report.json'
CAND40 = ROOT / 'results/alns_v12_verified_40_candidate.json'

SEED = 18012
RNG = random.Random(SEED)
MAX_ATTEMPTS = 320
MAX_EVALUATIONS = 165_000
MAX_SECONDS = 650
MIN_TOF = 0.5
PROGRESS_EVERY = 20


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs():
    p, seeds, missing, trouble, protected = v10.load_seeds()
    for path in (BASE38, BACKUP38, BASE39, ALT39, ARCHIVE39, V11_REPORT):
        if not path.is_file():
            raise RuntimeError(f'Mungon burimi: {path}')
        protected[str(path)] = sha(path)
    if BASE38.read_bytes() != BACKUP38.read_bytes():
        raise RuntimeError('Backup-i i 38 ndryshon nga origjinali')
    report11 = json.loads(V11_REPORT.read_text(encoding='utf-8'))
    if report11.get('base_level') != 39 or report11.get('best_level') not in (39, 40):
        raise RuntimeError('Raporti i V11 nuk i takon bazes 39')
    if len(missing) != 10 or len(seeds) < 1 or p.n_tomatoes != 49:
        raise RuntimeError('Burimet nuk i takojne easy.kttsp (49 domate)')
    for pair, qty in report11.get('most_failed_edges', []):
        try:
            a, b = map(int, pair.split('->'))
            trouble[(a, b)] += int(qty)
        except (TypeError, ValueError, AttributeError):
            continue
    return p, seeds, missing, trouble, protected


def check_protected(protected):
    for raw, digest in protected.items():
        path = Path(raw)
        if not path.is_file() or sha(path) != digest:
            raise RuntimeError(f'Burimi i mbrojtur ndryshoi: {path}')


def new_targets(original, first, second, mode):
    """Replace original[0] with 2 absent tomato IDs, keep window's last anchor.

    With no original suffix, the final anchor still remains in this variant.
    """
    tail = list(original[1:])
    if not tail:  # a one-node window does not preserve its right anchor
        return None
    if mode == 'adjacent_front':
        return [first, second] + tail
    if mode == 'separated':
        return [first] + tail[:-1] + [second, tail[-1]]
    if mode == 'adjacent_back':
        return tail[:-1] + [first, second, tail[-1]]
    if mode == 'swap_separated':
        return [second] + tail[:-1] + [first, tail[-1]]
    return None


def generate_plans(seeds, missing, trouble):
    """Score and interleave plans across sources, cuts, and missing tomato IDs."""
    grouped = defaultdict(list)
    seen = set()
    for src_idx, (name, state) in enumerate(seeds):
        route = state['route']
        for i in range(1, len(route)-1):  # preserve starting tomato and right anchor
            for horizon in (3, 5, 8, 12, 18, 28):
                end = min(len(route)-1, i+horizon-1)
                if end <= i:
                    continue
                start, deadline, old, reserved, _, _ = v9.window(state, i, end)
                cap = 5 - reserved
                needed = end-i+2  # one tomato replaced by two -> +1 leg
                slack = deadline - start - MIN_TOF*needed
                if slack < -1e-6 or cap < 0:
                    continue
                local_special = sum(t['status']=='SPECIAL' for t in old)
                before = route[i-1]
                anchor = route[end]
                for x, y in combinations(missing, 2):
                    for a, b in ((x, y), (y, x)):
                        for mode in ('adjacent_front', 'separated', 'adjacent_back', 'swap_separated'):
                            targets = new_targets(route[i:end+1], a, b, mode)
                            if targets is None or targets[-1] != anchor:
                                continue
                            key = (src_idx, i, end, tuple(targets))
                            if key in seen:
                                continue
                            seen.add(key)
                            # More slack and more existing SPECIAL in the window
                            # allow retiming, but longer windows are more expensive.
                            bad = trouble[(before, targets[0])]
                            bad += trouble[(targets[-2], anchor)] if len(targets)>1 else 0
                            score = (3.0*min(local_special, 3)
                                     + 2.2*min(slack/max(1, needed), 8.)
                                     - .6*needed - .12*bad
                                     + .03*i + RNG.random()*4.)
                            grouped[(src_idx, x, y)].append((
                                score, src_idx, name, i, end, a, b, mode, tuple(targets)))
    for plans in grouped.values():
        plans.sort(reverse=True, key=lambda item: item[0])
    keys = list(grouped)
    RNG.shuffle(keys)
    chosen = []
    positions = {key: 0 for key in keys}
    while len(chosen) < MAX_ATTEMPTS:
        progress = False
        for key in keys:
            idx = positions[key]
            if idx >= len(grouped[key]):
                continue
            chosen.append(grouped[key][idx])
            positions[key] += 1
            progress = True
            if len(chosen) >= MAX_ATTEMPTS:
                break
        if not progress:
            break
    return chosen


def write_new(path, obj):
    """Atomic no-overwrite, so interruption cannot corrupt proven artifacts."""
    payload = (json.dumps(obj, ensure_ascii=False, indent=2)+'\n').encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with tmp.open('xb') as f:
        f.write(payload)
    try:
        os.link(tmp, path)  # fails if destination already exists
    finally:
        tmp.unlink(missing_ok=True)
    print('U ruajt:', path.name, flush=True)


def run():
    p, seeds, missing, trouble, protected = load_inputs()
    print('Instanca: easy.kttsp | 49 domate | burimet:',
          [name for name, _ in seeds], flush=True)
    print('Mungojne:', missing, '| 1 per 2: direkt 39 -> 40', flush=True)
    if '--check-only' in sys.argv:
        print('CHECK-ONLY OK. Asnje skedar nuk u ndryshua.', flush=True)
        return
    if REPORT.exists() or CAND40.exists():
        raise RuntimeError('Rezultate V12 ekzistuese; nuk mbishkruhen.')
    v4.MAX_EVALUATIONS = MAX_EVALUATIONS
    evaluator = v4.Evaluator(p)
    plans = generate_plans(seeds, missing, trouble)
    print('Planet e zgjedhura:', len(plans), flush=True)
    attempts = 0
    full_repairs = 0
    failures = Counter()
    failed_edges = Counter()
    best_partial = {'legs_repaired': 0, 'legs_needed': 0}
    found = None
    found_context = None
    started = time.monotonic()
    budget_hit = False
    time_hit = False
    for score, src_idx, name, i, end, x, y, mode, targets in plans:
        if time.monotonic()-started >= MAX_SECONDS:
            time_hit = True
            break
        base = seeds[src_idx][1]
        start, deadline, old, reserved, prefix, suffix = v9.window(base, i, end)
        cap = p.n_exceptions-reserved
        attempts += 1
        try:
            repaired, reached, reason, failed_edge = v10.repair_phase(
                p, evaluator, base['route'][i-1], list(targets),
                start, deadline, cap, old)
        except v4.BudgetReached:
            budget_hit = True
            break
        if (reached/max(len(targets), 1) >
                best_partial['legs_repaired']/max(best_partial['legs_needed'], 1)):
            best_partial = {
                'attempt': attempts, 'source': name,
                'removed': base['route'][i], 'added': [x, y],
                'cut': i, 'end': end, 'operator': mode,
                'legs_repaired': reached, 'legs_needed': len(targets),
                'failed_edge': failed_edge,
            }
        if repaired is None:
            failures[reason or 'unknown'] += 1
            if failed_edge:
                failed_edges[f"{failed_edge['from']}->{failed_edge['to']}"] += 1
        else:
            full_repairs += 1
            state = v9.stitch(base, i, end, list(targets), repaired, prefix, suffix)
            if len(state['route']) == 40 and len(set(state['route'])) == 40 and v4.verify(p, state):
                found = state
                found_context = {'source': name, 'removed': base['route'][i],
                    'added': [x, y], 'cut': i, 'end': end,
                    'operator': mode, 'attempt': attempts}
                print('GJETEM KANDIDAT 40/49 TE VERIFIKUAR!', found_context, flush=True)
                break
            failures['full_verification_failed'] += 1
        if attempts % PROGRESS_EVERY == 0:
            print(f'V12: {attempts}/{MAX_ATTEMPTS} prova '
                  f'| {evaluator.count} eval | {full_repairs} full', flush=True)
    if found is not None and not v4.verify(p, found):
        raise RuntimeError('Riverifikimi i 40 deshtoi!')
    check_protected(protected)
    if found is not None:
        write_new(CAND40, {'level':40,'verified':True,
            'source':'alns_v12_direct_1for2.py', 'frontier':[found]})
    report = {
        'algorithm':'ALNS V12 direct 1-for-2 exchange + temporal DP',
        'base_level':39,
        'best_level':40 if found is not None else 39,
        'candidate_verified':found is not None,
        'attempts':attempts,
        'full_temporal_repairs':full_repairs,
        'transfer_evaluations':evaluator.count,
        'closest_repair':best_partial,
        'failure_reasons':dict(failures),
        'most_failed_edges':failed_edges.most_common(15),
        'evaluation_limit_reached':budget_hit,
        'runtime_limit_reached':time_hit,
        'elapsed_seconds':round(time.monotonic()-started, 3),
        'candidate_context':found_context,
        'source_seed_names':[name for name,_ in seeds],
        'protected_38_sha256':protected[str(BASE38)],
        'protected_39_sha256':protected[str(BASE39)],
        'protected_v11_report_sha256':protected[str(V11_REPORT)],
        'protected_v9_archive_sha256':protected[str(ARCHIVE39)],
        'protected_alt39_sha256':protected[str(ALT39)],
        'script_sha256':sha(Path(__file__)),
        'seed':SEED,
    }
    write_new(REPORT, report)
    print('V12:', report['best_level'], '/49 | prova:', attempts,
          '| eval:', evaluator.count, '| full:',full_repairs,flush=True)
    print('Rekordet 38/39: TE PAPREKURA',flush=True)


if __name__ == '__main__':
    run()
