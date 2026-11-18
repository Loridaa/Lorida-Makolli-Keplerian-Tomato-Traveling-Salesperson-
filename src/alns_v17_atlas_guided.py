"""ESA SpOC4 — Hapi 187 / ALNS V17: atlas-guided 39 -> 40 insertion.

Read-only sources: verified 38/49 + backup, verified 39/49, verified
neutral alternatives from V7/V9, V16 pairwise orbital feasibility atlas.

IMPORTANT: Atlas evidence ranks candidate pairs but NEVER certifies a full
route; candidate is accepted only after independent full-route PyKEP checks.
Prior algorithms/records are never overwritten. Creates new report and optional
40/49 candidate exclusively after a complete run.

Run in Kaggle via /kaggle/working/esa-pykep-env/bin/python:
   python -u src/alns_v17_atlas_guided.py --check-only
   python -u src/alns_v17_atlas_guided.py
"""
from __future__ import annotations

from collections import Counter, defaultdict
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
import alns_v4_dp_repair as v4
import alns_v9_multistart as v9
import alns_v10_temporal_labels as v10

BASE38 = ROOT / 'results/best_verified_38.json'
BACKUP38 = ROOT / 'kaggle_backups/results/best_verified_38.json'
BASE39 = ROOT / 'results/best_verified_39.json'
V7 = ROOT / 'results/alns_v7_verified_39_neutral.json'
V9 = ROOT / 'results/alns_v9_multistart_verified_39_archive.json'
ATLAS = ROOT / 'results/v16_orbital_feasibility_atlas_report.json'
REPORT = ROOT / 'results/alns_v17_atlas_guided_report.json'
CAND40 = ROOT / 'results/alns_v17_verified_40_candidate.json'

SEED = 18717
RNG = random.Random(SEED)
MAX_ATTEMPTS = 170
MAX_EVALUATIONS = 140_000
MAX_SECONDS = 540
PROGRESS_EVERY = 15


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_once(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(obj, ensure_ascii=False, indent=2) + '\n').encode()
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temp.open('xb') as f:
        f.write(content)
    try:
        os.link(temp, path)  # fails if target already exists
    finally:
        temp.unlink(missing_ok=True)
    print('U shkrua:', path.name, flush=True)


def checked_state(p, state, level):
    return (isinstance(state, dict)
            and len(state.get('route', [])) == level
            and len(state.get('transfers', [])) == level-1
            and v4.verify(p, state))


def load_inputs():
    files = (BASE38, BACKUP38, BASE39, V7, V9, ATLAS)
    for path in files:
        if not path.is_file():
            raise RuntimeError(f'Mungon burimi: {path}')
    hashes = {str(path): sha(path) for path in files}
    if hashes[str(BASE38)] != hashes[str(BACKUP38)]:
        raise RuntimeError('Rekordi 38/49 ndryshon nga backup-i')
    p = v4.load_problem()
    if p.n_tomatoes != 49:
        raise RuntimeError('Pritet easy.kttsp me 49 domate')
    old = json.loads(BASE38.read_text())
    base = json.loads(BASE39.read_text())
    if (old.get('level') != 38 or old.get('verified') is not True
            or base.get('level') != 39 or base.get('verified') is not True
            or len(old.get('frontier', [])) != 1
            or len(base.get('frontier', [])) != 1
            or not checked_state(p, old['frontier'][0], 38)
            or not checked_state(p, base['frontier'][0], 39)):
        raise RuntimeError('Rekordet baze nuk kaluan verifikimin orbital')
    if not set(old['frontier'][0]['route']).issubset(base['frontier'][0]['route']):
        raise RuntimeError('Baza 39 nuk permban bazen 38')
    seeds = [('record39', base['frontier'][0])]
    seen = {v9.fingerprint(seeds[0][1])}
    for path, label in [(V7, 'v7_neutral'), (V9, 'v9_archive')]:
        obj = json.loads(path.read_text())
        if obj.get('level') != 39 or obj.get('verified') is not True:
            raise RuntimeError('Alternativa 39 ka metadata te pasakta: ' + path.name)
        frontier = obj.get('frontier')
        if not isinstance(frontier, list) or not frontier:
            raise RuntimeError('Arkivi eshte bosh: ' + path.name)
        for i, state in enumerate(frontier, 1):
            if (not checked_state(p, state, 39)
                    or set(state['route']) != set(seeds[0][1]['route'])):
                raise RuntimeError(f'Rruga alternative {label} nr.{i} eshte e pavlefshme')
            key = v9.fingerprint(state)
            if key not in seen:
                seeds.append((f'{label}_{i}', state))
                seen.add(key)
    missing = sorted(set(range(49)) - set(seeds[0][1]['route']))
    atlas = json.loads(ATLAS.read_text())
    if (atlas.get('diagnostic_only') is not True or atlas.get('base_level') != 39
            or sorted(atlas.get('missing_tomatoes', [])) != missing
            or atlas.get('protected_39_sha256') != hashes[str(BASE39)]
            or atlas.get('protected_38_sha256') != hashes[str(BASE38)]):
        raise RuntimeError('Atlasi V16 nuk i perket ketyre rekordeve')
    rows = {int(row['tomato']): row for row in atlas['by_missing_tomato']}
    if set(rows) != set(missing):
        raise RuntimeError('Atlasi nuk i mbulon saktesisht 10 domatet')
    return p, seeds, missing, atlas, rows, hashes


def samples(rows, tomato, neighbor, direction):
    row = rows[tomato]
    key = 'incoming_from_covered' if direction == 'incoming' else 'outgoing_to_covered'
    # Atlas top_feasible_edges lists at most 5 neighbors; lack of an edge in
    # this list is NOT evidence that the pair is infeasible.
    for entry in row[key]['top_feasible_edges']:
        other = entry['from'] if direction == 'incoming' else entry['to']
        if other == neighbor:
            return int(entry['normal_samples']), int(entry['special_samples']), True
    return 0, 0, False


def priority_for_pair(rows, tomato, incoming, outgoing, rank):
    in_n, in_s, in_known = samples(rows, tomato, incoming, 'incoming')
    out_n, out_s, out_known = samples(rows, tomato, outgoing, 'outgoing')
    score = 8.0 - 1.1*rank
    score += 1.3*math.log1p(in_n) + 1.3*math.log1p(out_n)
    score += .22*math.log1p(in_s) + .22*math.log1p(out_s)
    if in_n and out_n:
        score += 9.0
    if in_known and out_known:
        score += 4.0
    if tomato == 4 and (incoming == 17 or outgoing == 17):
        score += 18.0
    return score


def plan_choices(seeds, missing, atlas, rows):
    # Prioritize V16-ranked candidates, but do not exclude remaining tomatoes.
    ranking = [int(row['tomato']) for row in atlas['comparative_ranking']]
    if set(ranking) != set(missing):
        raise RuntimeError('Renditja e atlasit eshte e pasakte')
    gathered = {}
    for sidx, (source, state) in enumerate(seeds):
        route = state['route']
        # last slot 38 is allowed; suffix can be empty.
        for tomato in ranking:
            rank = ranking.index(tomato)
            for cut in range(1, len(route)):
                for end in sorted({min(len(route)-1, cut+d) for d in (0, 2, 5, 10, 20, 100)}):
                    start, deadline, old, reserved, prefix, suffix = v9.window(state, cut, end)
                    cap = 5 - reserved
                    if cap < 0:
                        continue
                    body = list(route[cut:end+1])
                    if deadline - start < .5*(len(body)+1) - 1e-5:
                        continue
                    positions = {0, 1, min(len(body), 2), len(body)}
                    # For 4, also place it immediately before or after 17.
                    if tomato == 4 and 17 in body:
                        j = body.index(17)
                        positions.update((j, j+1))
                    # Try around the route segment's SPECIAL edges.
                    for j, leg in enumerate(old):
                        if leg['status'] == 'SPECIAL':
                            positions.update((max(0, j), min(len(body), j+1)))
                    for offset in positions:
                        if not 0 <= offset <= len(body):
                            continue
                        targets = body.copy()
                        targets.insert(offset, tomato)
                        left = route[cut-1]
                        a = left if offset == 0 else targets[offset-1]
                        b = targets[offset+1] if offset+1 < len(targets) else None
                        # For last-target insert, we do not know the next
                        # neighbor inside the repaired window; do not invent it.
                        pscore = priority_for_pair(rows, tomato, a, b, rank) if b is not None else (8.0-1.1*rank)
                        slack = deadline - start - .5*len(targets)
                        specials_in_window = sum(t['status']=='SPECIAL' for t in old)
                        # Rewards a feasible local retiming window, not just
                        # global pairwise transfer feasibility.
                        score = (pscore + 1.7*min(3, specials_in_window)
                                 + 1.5*min(8, max(0., slack)/max(1,len(targets)))
                                 - .4*len(targets) - .2*sidx
                                 + RNG.random()*1.5)
                        if tomato == 4 and 17 not in (a, b):
                            score -= 12
                        key = (sidx, cut, end, tuple(targets))
                        gathered[key] = (score, sidx, source, cut, end, tomato, tuple(targets))
    groups = defaultdict(list)
    for item in gathered.values():
        # Round-robin by tomato and source to preserve diversity.
        groups[(item[5], item[1])].append(item)
    for lst in groups.values():
        lst.sort(reverse=True, key=lambda x: x[0])
    order = sorted(groups, key=lambda k: (
        ranking.index(k[0]), k[1]))
    # Give scarce tomatoes some attempts too: interleave all group keys.
    chosen = []
    while order and len(chosen) < MAX_ATTEMPTS:
        remaining = []
        for key in order:
            lst = groups[key]
            if lst and len(chosen) < MAX_ATTEMPTS:
                chosen.append(lst.pop(0))
            if lst:
                remaining.append(key)
        order = remaining
    return chosen


def main():
    p, seeds, missing, atlas, rows, protected = load_inputs()
    print('V17: easy.kttsp | baza 39/49 | burime:', [x[0] for x in seeds], flush=True)
    print('Atlasi: ', [i['tomato'] for i in atlas['comparative_ranking']], flush=True)
    print('Domatet qe mungojne:', missing, flush=True)
    if '--check-only' in sys.argv:
        print('CHECK-ONLY OK. Nuk u shkrua asnje skedar.', flush=True)
        return
    if REPORT.exists() or CAND40.exists():
        raise RuntimeError('Dalja V17 ekziston. Nuk mbishkruhet.')
    options = plan_choices(seeds, missing, atlas, rows)
    if not options:
        raise RuntimeError('Nuk ka plane te zbatueshme sipas kufijve kohore')
    print('Plane te perzgjedhura:', len(options), flush=True)
    v4.MAX_EVALUATIONS = MAX_EVALUATIONS
    ev = v4.Evaluator(p)
    failures, failed_edges = Counter(), Counter()
    attempts = full = 0
    best_partial = {'legs_repaired': 0, 'legs_needed': 0}
    discovered = None
    context = None
    budget_hit = time_hit = False
    start_clock = time.monotonic()
    per_tomato = Counter()

    for score, sidx, source, cut, end, tomato, targets in options:
        if time.monotonic()-start_clock > MAX_SECONDS:
            time_hit = True
            break
        state = seeds[sidx][1]
        start, deadline, old, reserved, prefix, suffix = v9.window(state, cut, end)
        cap = p.n_exceptions-reserved
        if cap < 0:
            failures['special_budget_window'] += 1
            continue
        attempts += 1
        per_tomato[tomato] += 1
        solver = v10.repair_phase if attempts % 3 else v4.repair
        try:
            repaired, reached, reason, failed = solver(
                p, ev, state['route'][cut-1], list(targets),
                start, deadline, cap, old)
        except v4.BudgetReached:
            budget_hit = True
            break
        if (reached > best_partial['legs_repaired'] or
            (reached == best_partial['legs_repaired'] and len(targets) < best_partial['legs_needed'])):
            best_partial = {'attempt': attempts, 'source': source,
                            'cut': cut, 'end': end, 'inserted': tomato,
                            'legs_repaired': reached, 'legs_needed': len(targets),
                            'failed_edge': failed}
        if repaired is None:
            failures[reason or 'unknown'] += 1
            if failed:
                failed_edges[f"{failed['from']}->{failed['to']}"] += 1
        else:
            full += 1
            candidate = v9.stitch(state, cut, end, list(targets), repaired, prefix, suffix)
            if (len(candidate['route']) == 40
                and len(set(candidate['route'])) == 40
                and len(candidate['transfers']) == 39
                and set(candidate['route']) - set(state['route']) == {tomato}
                and v4.verify(p, candidate)):
                discovered = candidate
                context = {'attempt': attempts, 'source': source,
                           'cut': cut, 'end': end, 'inserted': tomato}
                print('40/49 RIVERIFIKUAR ME PYKEP!', context, flush=True)
                break
            failures['full_verification_failed'] += 1
        if attempts % PROGRESS_EVERY == 0:
            print(f'V17: {attempts}/{len(options)} | eval={ev.count} | '
                  f'full={full} | prane={best_partial["legs_repaired"]}/'
                  f'{best_partial["legs_needed"]}', flush=True)
    if any(not Path(k).is_file() or sha(k) != digest
           for k, digest in protected.items()):
        raise RuntimeError('Nje burim i mbrojtur ndryshoi')
    if discovered is not None:
        write_once(CAND40, {'level':40, 'verified':True,
                            'source':'alns_v17_atlas_guided.py',
                            'frontier':[discovered]})
    report = {
        'algorithm':'V17 atlas-guided targeted insertion with temporal DP',
        'base_level':39, 'best_level':40 if discovered else 39,
        'candidate_verified': bool(discovered), 'attempts':attempts,
        'full_temporal_repairs':full, 'transfer_evaluations':ev.count,
        'best_partial_repair':best_partial,
        'failure_reasons':dict(failures),
        'most_failed_edges':failed_edges.most_common(12),
        'attempts_per_tomato':{str(k):v for k,v in sorted(per_tomato.items())},
        'ranking_used':[i['tomato'] for i in atlas['comparative_ranking']],
        'seed_sources':[x[0] for x in seeds],
        'protected_38_sha256':protected[str(BASE38)],
        'protected_39_sha256':protected[str(BASE39)],
        'protected_atlas_sha256':protected[str(ATLAS)],
        'protected_sources_sha256':protected,
        'script_sha256':sha(Path(__file__)),
        'candidate_context':context,
        'evaluation_limit_reached':budget_hit,
        'time_limit_reached':time_hit,
        'elapsed_seconds':round(time.monotonic()-start_clock,2),
        'note':'Atlas pair feasibility does not prove route feasibility.'}
    write_once(REPORT, report)
    print('V17 PERFUNDOI:', report['best_level'], '/49',
          '| prova:', attempts, '| evaluime:', ev.count, flush=True)
    print('Rekordet 38 dhe 39: TE PAPREKURA', flush=True)


if __name__ == '__main__':
    main()
