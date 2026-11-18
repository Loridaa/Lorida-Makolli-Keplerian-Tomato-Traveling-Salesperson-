"""ESA SpOC4 | ALNS V10 | phase-diverse temporal dynamic programming (39 -> 40).

This is a NEW algorithm file. It does not change V4-V9, existing verified
records, or submission files. Real PyKEP verification is mandatory.

V9's V4.repair kept only the earliest arrival per SPECIAL count. That
pruning is mathematically safe for exhaustive time choices with waiting,
but can discard useful time phases when departure/TOF grids are finite.
V10 retains several arrival-time labels per SPECIAL count and refines the
local grid near the lowest delta-v transfers when a leg fails.

Run in Kaggle's PyKEP interpreter (not the notebook Python):
  /kaggle/working/esa-pykep-env/bin/python -u src/alns_v10_temporal_labels.py --check-only
  /kaggle/working/esa-pykep-env/bin/python -u src/alns_v10_temporal_labels.py
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
import alns_v4_dp_repair as v4  # noqa: E402
import alns_v9_multistart as v9  # noqa: E402

BASE38 = ROOT / 'results/best_verified_38.json'
BACKUP38 = ROOT / 'kaggle_backups/results/best_verified_38.json'
BASE39 = ROOT / 'results/best_verified_39.json'
ALT39 = ROOT / 'results/alns_v7_verified_39_neutral.json'
ARCHIVE39 = ROOT / 'results/alns_v9_multistart_verified_39_archive.json'
V9_REPORT = ROOT / 'results/alns_v9_multistart_report.json'
REPORT = ROOT / 'results/alns_v10_temporal_labels_report.json'
CAND40 = ROOT / 'results/alns_v10_verified_40_candidate.json'

SEED = 17910
RNG = random.Random(SEED)
MAX_ATTEMPTS = 180
MAX_EVALUATIONS = 165_000
MAX_SECONDS = 540
MIN_TOF = 0.5
EPS = 1e-6
# Three diverse time labels per used SPECIAL count rather than one.
LABELS_PER_SPECIAL = 3
PROGRESS_EVERY = 10


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protect_sources():
    protected = [BASE38, BACKUP38, BASE39, ALT39, ARCHIVE39, V9_REPORT]
    for path in protected:
        if not path.is_file():
            raise RuntimeError(f'Mungon burimi: {path}')
    fingerprints = {str(p): sha(p) for p in protected}
    if BASE38.read_bytes() != BACKUP38.read_bytes():
        raise RuntimeError('38/49 nuk perputhet me kopjen rezerve')
    return fingerprints


def read_one(path: Path, count: int, problem):
    obj = json.loads(path.read_text(encoding='utf-8'))
    if obj.get('level') != count or obj.get('verified') is not True:
        raise RuntimeError(f'Format i pavlefshem: {path.name}')
    frontier = obj.get('frontier', [])
    if len(frontier) != 1:
        raise RuntimeError(f'Frontier jo unike: {path.name}')
    state = frontier[0]
    if len(state['route']) != count or not v4.verify(problem, state):
        raise RuntimeError(f'Verifikimi orbital deshtoi: {path.name}')
    return state


def load_seeds():
    fingerprints = protect_sources()
    p = v4.load_problem()
    if p.n_tomatoes != 49:
        raise RuntimeError('Kerkohen 49 domate nga easy.kttsp')
    base38 = read_one(BASE38, 38, p)
    base = read_one(BASE39, 39, p)
    alt = read_one(ALT39, 39, p)
    if not set(base38['route']).issubset(base['route']):
        raise RuntimeError('Rekordi 39 nuk permban 38 domatet e meparshme')
    if set(alt['route']) != set(base['route']):
        raise RuntimeError('Alternativa V7 ka domate te ndryshme')
    seeds = [('record39', base)]
    seen = {v9.fingerprint(base)}
    for name, state in [('v7_alternative', alt)]:
        if v9.fingerprint(state) not in seen:
            seeds.append((name, state)); seen.add(v9.fingerprint(state))
    archive = json.loads(ARCHIVE39.read_text(encoding='utf-8'))
    if archive.get('level') != 39 or archive.get('verified') is not True:
        raise RuntimeError('Arkivi i V9 eshte i pavlefshem')
    for i, state in enumerate(archive.get('frontier', []), 1):
        if (len(state.get('route', [])) != 39
                or set(state['route']) != set(base['route'])
                or not v4.verify(p, state)):
            raise RuntimeError(f'Alternativa V9 nr.{i} eshte e pavlefshme')
        fp = v9.fingerprint(state)
        if fp not in seen:
            seeds.append((f'v9_archive_{i}', state)); seen.add(fp)
    missing = sorted(set(range(49)) - set(base['route']))
    if len(missing) != 10:
        raise RuntimeError('Nuk u gjeten saktesisht 10 domate te pambuluara')
    v9data = json.loads(V9_REPORT.read_text(encoding='utf-8'))
    if v9data.get('base_level') != 39:
        raise RuntimeError('Raporti V9 nuk i takon rekordit 39')
    troublesome = Counter()
    for pair, hits in v9data.get('most_failed_edges', []):
        try:
            a, b = map(int, pair.split('->'))
            troublesome[(a, b)] += int(hits)
        except (ValueError, TypeError):
            continue
    return p, seeds, missing, troublesome, fingerprints


def select_labels(items, cap):
    """Diversify across time buckets, keep early/late/low-DV labels.

    This is an intentionally approximate *multi-label* DP for discretized
    transfer times; no state here counts as valid until PyKEP verification.
    """
    by_used = defaultdict(dict)
    for item in items:
        if item['used'] > cap:
            continue
        bucket = int(math.floor(float(item['t']) / 1.0))
        key = (item['used'], bucket)
        old = by_used[item['used']].get(key)
        if old is None or (item['dv'], item['t']) < (old['dv'], old['t']):
            by_used[item['used']][key] = item
    chosen = []
    for used in sorted(by_used):
        vals = list(by_used[used].values())
        if not vals:
            continue
        earliest = min(vals, key=lambda x: (x['t'], x['dv']))
        cheapest = min(vals, key=lambda x: (x['dv'], x['t']))
        latest = max(vals, key=lambda x: (x['t'], -x['dv']))
        picks = []
        for option in (earliest, cheapest, latest):
            if not any(option is p for p in picks):
                picks.append(option)
            if len(picks) >= LABELS_PER_SPECIAL:
                break
        if len(picks) < LABELS_PER_SPECIAL:
            for option in sorted(vals, key=lambda x: (x['t'], x['dv'])):
                if not any(option is p for p in picks): picks.append(option)
                if len(picks) == LABELS_PER_SPECIAL: break
        chosen.extend(picks)
    return chosen


def repair_phase(p, evaluator, left, targets, start, deadline, cap, original_legs):
    """Multi-label repair preserving a few alternative arrival-time phases."""
    warm = defaultdict(list)
    for leg in original_legs:
        warm[(leg['from'], leg['to'])].append(leg)
    states = [{'t': float(start), 'used': 0, 'dv': 0., 'legs': []}]
    reached = 0
    for index, b in enumerate(targets):
        a = left if index == 0 else targets[index-1]
        incoming = warm.get((a, b), [])
        anchor = original_legs[min(index, len(original_legs)-1)] if original_legs else None
        next_items = []
        near = []
        options_checked = 0
        threshold_pass = 0
        budget_excluded = 0
        for state in states:
            # Known-valid transfer can be reused exactly at its previous time.
            for leg in incoming:
                depart, tof = float(leg['start_time']), float(leg['tof'])
                if depart >= state['t']-EPS and depart+tof <= deadline+EPS:
                    extra = int(leg['status']=='SPECIAL')
                    if state['used']+extra <= cap:
                        next_items.append({'t':depart+tof,'used':state['used']+extra,
                            'dv':state['dv']+float(leg['dv']),
                            'legs':state['legs']+[dict(leg)]})
            departures = {round(state['t']+w, 6) for w in
                          (0., .25, .5, 1., 2., 4.)}
            durations = {.5, .75, 1., 1.5, 2., 3., 4., 6., 8.}
            for leg in incoming + ([anchor] if anchor else []):
                old_d, old_t = float(leg['start_time']), float(leg['tof'])
                departures.update(round(old_d+d, 6) for d in (-.5, 0., .5, 1.))
                durations.update(round(old_t+t, 6) for t in (-.5, 0., .5))
            for departure in sorted(departures):
                if departure < state['t']-EPS or departure+MIN_TOF>deadline+EPS:
                    continue
                for tof in sorted(d for d in durations if d>=MIN_TOF):
                    if departure+tof > deadline+EPS:
                        continue
                    options_checked += 1
                    dv = evaluator.get(a,b,departure,tof)
                    if math.isfinite(dv):
                        near.append((dv,departure,tof,state))
                    cat = v4.classify(p,dv)
                    if cat is None: continue
                    threshold_pass += 1
                    status, extra = cat
                    if state['used']+extra>cap:
                        budget_excluded += 1
                        continue
                    leg = {'from':a,'to':b,'start_time':departure,
                           'tof':float(tof),'dv':float(dv),'status':status}
                    next_items.append({'t':departure+tof,
                       'used':state['used']+extra,
                       'dv':state['dv']+dv, 'legs':state['legs']+[leg]})
        if not next_items and near:
            # Unlike V4, use a fine *local* time grid near promising costs.
            # This is only triggered when the coarse grid found no feasible label.
            tried = set()
            for _, d0, t0, state in sorted(near,key=lambda z:z[0])[:4]:
                for dshift in (-.5,-.25,0.,.25,.5):
                    for tshift in (-.5,-.25,0.,.25,.5):
                        dep, tof = round(d0+dshift,6), round(t0+tshift,6)
                        if (dep,tof,state['used']) in tried: continue
                        tried.add((dep,tof,state['used']))
                        if dep < state['t']-EPS or tof<MIN_TOF or dep+tof>deadline+EPS:
                            continue
                        options_checked += 1
                        dv = evaluator.get(a,b,dep,tof)
                        cat = v4.classify(p,dv)
                        if cat is None: continue
                        threshold_pass += 1
                        status, extra = cat
                        if state['used']+extra>cap:
                            budget_excluded+=1; continue
                        leg = {'from':a,'to':b,'start_time':dep,
                               'tof':tof,'dv':float(dv),'status':status}
                        next_items.append({'t':dep+tof,'used':state['used']+extra,
                            'dv':state['dv']+dv,'legs':state['legs']+[leg]})
        if not next_items:
            why = ('no_time_options' if options_checked==0 else
                   'special_budget' if threshold_pass and budget_excluded else
                   'no_transfer_under_threshold')
            return None, reached, why, {'from':a,'to':b,'repair_leg_index':index+1}
        states = select_labels(next_items,cap)
        if not states:
            return None, reached, 'empty_labels', {'from':a,'to':b,'repair_leg_index':index+1}
        reached += 1
    best = min(states,key=lambda s:(s['used'],s['t'],s['dv']))
    return best,reached,None,None


def write_exclusive(path: Path, data):
    """Write a finished result without ever overwriting an existing result."""
    payload=(json.dumps(data,indent=2,ensure_ascii=False)+'\n').encode('utf-8')
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with tmp.open('xb') as f: f.write(payload)
    try: os.link(tmp,path)
    finally: tmp.unlink(missing_ok=True)
    print('U ruajt:',path.name,flush=True)


def run():
    p,seeds,missing,trouble,protected=load_seeds()
    print('Burimet:',[name for name,_ in seeds],flush=True)
    print('Mungojne:',missing,'| maksimum SPECIAL:',p.n_exceptions,flush=True)
    if '--check-only' in sys.argv:
        print('CHECK-ONLY OK: te gjitha burimet u verifikuan. Nuk u shkrua asgje.',flush=True)
        return
    if REPORT.exists() or CAND40.exists():
        raise RuntimeError('Rezultat V10 ekzistues; nuk mbishkruhet!')
    started=time.monotonic()
    v4.MAX_EVALUATIONS=MAX_EVALUATIONS
    evaluator=v4.Evaluator(p)
    # Reuse V9's deterministic neighborhood generation, with *different*
    # repair logic: phase-aware multistate DP + local fine-grid repair.
    plans=defaultdict(list)
    seen=set()
    for idx,(source,state) in enumerate(seeds):
        for tomato, candidates in v9.insertion_plans(source,state,missing,trouble,idx).items():
            for item in candidates:
                _,srcidx,source_name,i,end,tomato,seq,op=item
                key=(v9.fingerprint(state),i,end,tuple(seq))
                if key in seen: continue
                seen.add(key)
                plans[(idx,tomato)].append(item)
    keys=list(plans)
    RNG.shuffle(keys)
    attempts=0; full=0; failures=Counter(); edges=Counter()
    best_partial={'legs_repaired':0,'legs_needed':0}
    position={key:0 for key in keys}
    found=None; context=None
    time_cap_hit=False; budget_hit=False
    while attempts<MAX_ATTEMPTS:
        if time.monotonic()-started > MAX_SECONDS:
            time_cap_hit=True; break
        progressed=False
        for key in keys:
            if attempts>=MAX_ATTEMPTS or time.monotonic()-started>MAX_SECONDS:
                break
            group=plans[key]
            offset=position[key]
            if offset>=len(group): continue
            plan=group[offset]
            position[key]+=1
            progressed=True
            _,idx,source,i,end,tomato,seq,op=plan
            state=seeds[idx][1]
            start,deadline,old,reserved,prefix,suffix=v9.window(state,i,end)
            cap=p.n_exceptions-reserved
            if cap<0: continue
            attempts+=1
            try:
                repaired,reached,why,edge=repair_phase(
                    p,evaluator,state['route'][i-1],list(seq),
                    start,deadline,cap,old)
            except v4.BudgetReached:
                budget_hit=True; break
            if attempts%PROGRESS_EVERY==0:
                print(f'V10: {attempts}/{MAX_ATTEMPTS} prova | '
                      f'{evaluator.count} vleresime | '
                      f'{full} riparime te plota',flush=True)
            if (reached>best_partial['legs_repaired'] or
                (reached==best_partial['legs_repaired'] and
                 len(seq)<best_partial.get('legs_needed',10**9))):
                best_partial={'source':source,'tomato':tomato,'cut':i,
                    'end':end,'operator':op,'legs_repaired':reached,
                    'legs_needed':len(seq),'failed_edge':edge}
            if repaired is None:
                failures[why or 'unknown']+=1
                if edge: edges[f"{edge['from']}->{edge['to']}"]+=1
                continue
            full+=1
            candidate=v9.stitch(state,i,end,list(seq),repaired,prefix,suffix)
            if (len(candidate['route'])==40 and
                    len(set(candidate['route']))==40 and
                    v4.verify(p,candidate)):
                found=candidate
                context={'source':source,'tomato':tomato,'cut':i,
                         'end':end,'operator':op,'attempt':attempts}
                print('KANDIDATI 40/49 I VERIFIKUAR!',context,flush=True)
                break
            failures['failed_full_verification']+=1
        if found is not None or budget_hit or not progressed: break
    # Safety: independently verify the full 40 route after search.
    if found is not None and not v4.verify(p,found):
        raise RuntimeError('Kandidati nuk kaloi riverifikimin!')
    for path,saved_sha in protected.items():
        if sha(Path(path))!=saved_sha:
            raise RuntimeError('Burimi i mbrojtur ndryshoi: '+path)
    if found is not None:
        write_exclusive(CAND40,{'level':40,'verified':True,
            'source':'alns_v10_temporal_labels.py','frontier':[found]})
    report={
        'algorithm':'ALNS V10 - multistart temporal diversity DP with local refinement',
        'base_level':39,'best_level':40 if found is not None else 39,
        'candidate_verified':found is not None,
        'seed':SEED,'attempts':attempts,'full_temporal_repairs':full,
        'transfer_evaluations':evaluator.count,'evaluation_limit':MAX_EVALUATIONS,
        'evaluation_limit_reached':budget_hit,'runtime_limit_reached':time_cap_hit,
        'maximum_labels_per_special':LABELS_PER_SPECIAL,
        'failure_reasons':dict(failures),
        'most_failed_edges':edges.most_common(12),
        'closest_repair':best_partial,'candidate_context':context,
        'seed_sources':[name for name,_ in seeds],
        'protected_38_sha256':protected[str(BASE38)],
        'protected_39_sha256':protected[str(BASE39)],
        'protected_v9_archive_sha256':protected[str(ARCHIVE39)],
        'elapsed_seconds':round(time.monotonic()-started,3),
    }
    write_exclusive(REPORT,report)
    print('ALNS V10:',report['best_level'],'/49 |',attempts,
          'prova |',evaluator.count,'vleresime',flush=True)
    print('Rekordet 38/39: TE PAPREKURA',flush=True)

if __name__=='__main__':
    run()
