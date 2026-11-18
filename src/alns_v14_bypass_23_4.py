"""ESA SpOC4 — Hapi 184 / ALNS V14: targeted bypass of a failed edge.

The V13 diagnostic found that the tested 23 -> 4 transfers were above
SPECIAL threshold in their *tested* time window. This algorithm generates
new 39->40 (1-for-2) route neighborhoods that avoid that adjacency, retimes
all affected legs using the proven V10/V4 DP solvers, and independently
validates any complete 40/49 candidate with real PyKEP.

No original routes, reports, or other algorithms are changed. Results are
written once into new V14 paths. This is heuristic search: failure does not
prove global infeasibility.

Execute via Kaggle's isolated /kaggle/working/esa-pykep-env/bin/python.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import hashlib
import json
import os
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import alns_v4_dp_repair as v4
import alns_v9_multistart as v9
import alns_v10_temporal_labels as v10
import alns_v13_edge_rescue as v13

BASE38 = ROOT / 'results/best_verified_38.json'
BACKUP38 = ROOT / 'kaggle_backups/results/best_verified_38.json'
BASE39 = ROOT / 'results/best_verified_39.json'
V13REPORT = ROOT / 'results/alns_v13_edge_rescue_report.json'
PROBE = ROOT / 'results/alns_v13_failure_probe_report.json'
REPORT = ROOT / 'results/alns_v14_bypass_23_4_report.json'
CAND40 = ROOT / 'results/alns_v14_verified_40_candidate.json'

SEED = 18414
RNG = random.Random(SEED)
MAX_ATTEMPTS = 220
MAX_EVALUATIONS = 120_000
MAX_RUNTIME_SECONDS = 700
PROGRESS_EVERY = 20


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    b = (json.dumps(obj, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    with path.open('xb') as f:
        f.write(b)
    print('U ruajt:', path.name, flush=True)


def get_inputs():
    p, seeds, missing, trouble, protected, old12 = v13.load_inputs()
    for path in (BASE38, BACKUP38, BASE39, V13REPORT, PROBE):
        if not path.is_file():
            raise RuntimeError('Mungon burimi: ' + str(path))
        protected[str(path)] = sha(path)
    if BASE38.read_bytes() != BACKUP38.read_bytes():
        raise RuntimeError('Rekordi 38/49 ndryshon nga backup-i!')
    prior = json.loads(V13REPORT.read_text(encoding='utf-8'))
    probe = json.loads(PROBE.read_text(encoding='utf-8'))
    near = prior.get('closest_repair') or {}
    edge = probe.get('target_edge') or {}
    failed = near.get('failed_edge') or {}
    if prior.get('base_level') != 39 or probe.get('base_level') != 39:
        raise RuntimeError('Raportet nuk i perkasin rekordit 39/49')
    if (not probe.get('diagnostic_only')
        or (edge.get('from'),edge.get('to')) != (failed.get('from'), failed.get('to'))
        or probe.get('v13_source_sha256') != sha(V13REPORT)):
        raise RuntimeError('Diagnostika nuk perputhet me V13')
    if not isinstance(near.get('added'), list) or len(near['added']) != 2:
        raise RuntimeError('Mungon plani i proves me te afert te V13')
    return p, seeds, missing, trouble, protected, near, edge, probe


def has_banned_edge(left, targets, forbidden):
    seq = [left] + list(targets)
    return any((a, b) == forbidden for a, b in zip(seq, seq[1:]))


def insert_pair(body, x, y, j, k):
    """Insert two distinct missing nodes with positions relative to current list."""
    seq = list(body)
    j = max(0, min(int(j), len(seq)))
    seq.insert(j, x)
    k = max(0, min(int(k), len(seq)))
    seq.insert(k, y)
    return tuple(seq)


def plan_score(state, cut, end, targets, trouble, near, forbidden):
    left = state['route'][cut-1]
    if has_banned_edge(left, targets, forbidden):
        return None
    start, deadline, old, reserved, _, _ = v9.window(state, cut, end)
    if (deadline - start) < .5 * len(targets) - 1e-5:
        return None
    # Prefer reusing known-valid legs, and avoiding repeated failed edges.
    old_pairs = {(tr['from'],tr['to']) for tr in old}
    seq = [left] + list(targets)
    pairs = list(zip(seq, seq[1:]))
    inherited = sum((a,b) in old_pairs for a,b in pairs)
    penalty = sum(min(20, trouble[(a,b)]) for a,b in pairs)
    focus = 6 if near.get('added') and set(near['added']).issubset(set(targets)) else 0
    slack = deadline - start - .5 * len(targets)
    return 2.0*inherited + 0.5*min(slack, 20) + focus - .8*penalty - .3*len(targets)


def plans(seeds, missing, trouble, near, forbidden):
    # The V13 near miss determines our first 1-for-2 exchange.
    drop = near.get('removed')
    preferred_pair = tuple(near['added'])
    preferred_source = near.get('source')
    collected = {}
    for sidx,(source,state) in enumerate(seeds):
        route = state['route']
        if drop not in route or route.index(drop) == 0:
            continue
        old_position = route.index(drop)
        if any(x in route for x in preferred_pair):
            continue
        # Repair can begin earlier than the removed tomato; later fixed segments
        # are *not* forced to retain the bad 23->4 adjacency.
        starts = sorted({max(1, old_position-d) for d in (0,2,4,7)})
        pair_choices = [preferred_pair, preferred_pair[::-1]]
        # Small amount of extra subset diversity, while retaining main focus.
        other = [z for z in missing if z not in preferred_pair]
        for x in preferred_pair:
            for y in other[:4]:
                pair_choices.append((x,y))
        for cut in starts:
            end = len(route)-1  # to the end: avoid a frozen suffix edge
            body = [z for z in route[cut:end+1] if z != drop]
            if len(body) != len(route[cut:end+1])-1:
                continue
            n = len(body)
            local_index = max(0, old_position-cut)
            focus_positions = {0, n, n//2, local_index}
            for node in (forbidden[0], forbidden[1], 17, 20, 37):
                if node in body:
                    at=body.index(node)
                    focus_positions.update((max(0,at-2), at, at+1, min(n,at+3)))
            positions = sorted(z for z in focus_positions if 0<=z<=n)
            for x,y in pair_choices:
                if x == y or x in route or y in route:
                    continue
                # Neighbor and distant placements; both x/y orderings tested.
                for j in positions:
                    for k in (j, min(n+1,j+1), max(0,j-2), n//2, n+1):
                        targets = insert_pair(body,x,y,j,k)
                        if len(targets) != end-cut+2 or len(set(targets)) != len(targets):
                            continue
                        if len(set(targets).intersection(set(route[:cut]))) != 0:
                            continue
                        score = plan_score(state,cut,end,targets,trouble,near,forbidden)
                        if score is None:
                            continue
                        # Give V13's near-miss source/pair extra priority.
                        if source == preferred_source and {x,y} == set(preferred_pair):
                            score += 7
                        score += RNG.random() * 4
                        key=(sidx,cut,end,targets)
                        candidate=(score,sidx,source,cut,end,x,y,targets)
                        if key not in collected or score > collected[key][0]:
                            collected[key]=candidate
    # Round-robin across source/removed pair to avoid all trials on one area.
    groups={}
    for entry in collected.values():
        key=(entry[1],tuple(sorted((entry[5],entry[6]))))
        groups.setdefault(key,[]).append(entry)
    for group in groups.values():
        group.sort(reverse=True,key=lambda item:item[0])
    keys=sorted(groups,key=lambda k:groups[k][0][0],reverse=True)
    selected=[];seen=set(); rounds=0
    while len(selected)<MAX_ATTEMPTS and keys:
        more=[]
        for key in keys:
            group=groups[key]
            if rounds<len(group):
                entry=group[rounds]
                k=(entry[1],entry[3],entry[4],entry[7])
                if k not in seen:
                    selected.append(entry)
                    seen.add(k)
                if rounds+1<len(group):
                    more.append(key)
            if len(selected)>=MAX_ATTEMPTS:
                break
        rounds+=1;keys=more
    # Restrict to a meaningful neighborhood instead of repeated 23->4 scans.
    return selected


def main():
    p,seeds,missing,trouble,protected,near,edge,probe = get_inputs()
    banned=(int(edge['from']),int(edge['to']))
    print('Instanca: easy.kttsp, 49 domate',flush=True)
    print('Rekordi baze: 39/49 | burimet:',[s for s,_ in seeds],flush=True)
    print('Lidhja qe po shmangim:',banned,'| min DV e testuar:',probe.get('extra_grid_min_dv'),flush=True)
    if '--check-only' in sys.argv:
        print('CHECK-ONLY OK. Asnje rezultat nuk u ndryshua.',flush=True)
        return
    if REPORT.exists() or CAND40.exists():
        raise RuntimeError('Rezultatet V14 ekzistojne; nuk mbishkruhen')
    options=plans(seeds,missing,trouble,near,banned)
    print('Plane per testim:',len(options),flush=True)
    if not options:
        raise RuntimeError('Nuk u gjenerua asnje plan i ri; kontrollo burimet')
    v4.MAX_EVALUATIONS=MAX_EVALUATIONS
    ev=v4.Evaluator(p)
    failures=Counter();failed_edges=Counter();repairs=0;attempts=0
    closest={'legs_repaired':0,'legs_needed':0}
    found=None;origin=None;budget_hit=False;time_hit=False
    t0=time.monotonic()
    for _,sidx,source,cut,end,x,y,targets in options:
        if time.monotonic()-t0>=MAX_RUNTIME_SECONDS:
            time_hit=True
            break
        state=seeds[sidx][1]
        start,deadline,old,reserved,prefix,suffix=v9.window(state,cut,end)
        cap=p.n_exceptions-reserved
        if cap<0:
            continue
        attempts+=1
        solver=v10.repair_phase if attempts%3!=0 else v4.repair
        try:
            repaired,reached,reason,failed=solver(
                p,ev,state['route'][cut-1],list(targets),start,deadline,cap,old)
        except v4.BudgetReached:
            budget_hit=True;break
        if reached/max(len(targets),1)>closest['legs_repaired']/max(closest['legs_needed'],1):
            closest={'attempt':attempts,'source':source,'cut':cut,'end':end,
                     'removed':near.get('removed'),'added':[x,y],
                     'legs_repaired':reached,'legs_needed':len(targets),'failed_edge':failed}
        if repaired is None:
            failures[reason or 'unknown']+=1
            if failed:
                failed_edges[f"{failed['from']}->{failed['to']}"]+=1
        else:
            repairs+=1
            candidate=v9.stitch(state,cut,end,list(targets),repaired,prefix,suffix)
            if (len(candidate['route'])==40 and len(set(candidate['route']))==40
                and len(candidate['transfers'])==39
                and len(set(candidate['route'])-set(state['route']))==2
                and len(set(state['route'])-set(candidate['route']))==1
                and v4.verify(p,candidate)):
                found=candidate
                origin={'source':source,'removed':near.get('removed'),
                        'added':[x,y],'cut':cut,'attempt':attempts}
                print('KANDIDATI 40/49 U GJET DHE U VERIFIKUA!',origin,flush=True)
                break
            failures['full_verification_failed']+=1
        if attempts%PROGRESS_EVERY==0:
            print(f'V14: {attempts}/{MAX_ATTEMPTS} | eval={ev.count} | complete={repairs}',flush=True)
    if any(not Path(k).is_file() or sha(Path(k))!=v for k,v in protected.items()):
        raise RuntimeError('Nje burim i mbrojtur ndryshoi!')
    if found is not None:
        if not v4.verify(p,found):
            raise RuntimeError('Riverifikimi i 40/49 deshtoi')
        write_once(CAND40,{'level':40,'verified':True,'source':'alns_v14_bypass_23_4.py','frontier':[found]})
    report={
        'algorithm':'ALNS V14 bypass diagnostic edge using alternate route orders',
        'base_level':39,'best_level':40 if found is not None else 39,
        'candidate_verified':found is not None,'attempts':attempts,
        'full_temporal_repairs':repairs,'transfer_evaluations':ev.count,
        'closest_repair':closest,'failure_reasons':dict(failures),
        'most_failed_edges':failed_edges.most_common(12),
        'diagnostic_banned_edge':list(banned),
        'diagnostic_tested_min_dv':probe.get('extra_grid_min_dv'),
        'protected_38_sha256':protected[str(BASE38)],
        'protected_39_sha256':protected[str(BASE39)],
        'protected_v13_report_sha256':protected[str(V13REPORT)],
        'protected_probe_sha256':protected[str(PROBE)],
        'script_sha256':sha(Path(__file__)),
        'candidate_context':origin,'time_limit_reached':time_hit,
        'evaluation_limit_reached':budget_hit,
        'elapsed_seconds':round(time.monotonic()-t0,3),
        'seed':SEED}
    write_once(REPORT,report)
    print('V14 PERFUNDOI:',report['best_level'],'/49 |',attempts,'prova |',ev.count,'eval',flush=True)
    print('Rekordet 38/39: TE PAPREKURA',flush=True)

if __name__ == '__main__':
    main()
