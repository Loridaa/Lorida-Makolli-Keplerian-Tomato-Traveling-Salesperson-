"""ESA SpOC4 / Keplerian Tomato - ALNS V13 Edge Rescue.

Starts from the validated 39/49 states. Replaces one visited tomato by two
missing ones (39 -> 40), with new placements on either side of troublesome
edges, *adaptive right window extensions*, and two complementary temporal DP
repairs (V10 multi-label + V4 early-arrival). Only fully PyKEP-verified routes
are candidates. No existing project file is overwritten.

Kaggle usage (in the isolated PyKEP environment):
  python -u src/alns_v13_edge_rescue.py --check-only
  python -u src/alns_v13_edge_rescue.py
"""
from __future__ import annotations

from collections import Counter
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
V9ARCHIVE = ROOT / 'results/alns_v9_multistart_verified_39_archive.json'
V12REPORT = ROOT / 'results/alns_v12_direct_1for2_report.json'
REPORT = ROOT / 'results/alns_v13_edge_rescue_report.json'
CAND40 = ROOT / 'results/alns_v13_verified_40_candidate.json'

SEED = 18113
MAX_ATTEMPTS = 260
MAX_EVALUATIONS = 145_000
MAX_SECONDS = 720
MIN_TOF = .5
PROGRESS_EVERY = 20
RNG = random.Random(SEED)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs():
    p, seeds, missing, trouble, source_hashes = v10.load_seeds()
    protected = {str(Path(path)): digest for path, digest in source_hashes.items()}
    for path in (BASE38, BACKUP38, BASE39, ALT39, V9ARCHIVE, V12REPORT):
        if not path.is_file():
            raise RuntimeError('Mungon burimi: '+str(path))
        protected[str(path)] = sha(path)
    if BASE38.read_bytes() != BACKUP38.read_bytes():
        raise RuntimeError('Rekordi 38/49 ndryshon nga backup-i')
    prev = json.loads(V12REPORT.read_text(encoding='utf-8'))
    if (prev.get('base_level') != 39 or prev.get('best_level') not in (39, 40)
            or prev.get('script_sha256') is None):
        raise RuntimeError('Raporti V12 eshte i papershtatshem')
    if p.n_tomatoes != 49 or len(missing) != 10 or not seeds:
        raise RuntimeError('Nuk eshte instanca easy.kttsp me 49 domate')
    for edge, count in prev.get('most_failed_edges', []):
        try:
            a, b = (int(x) for x in edge.split('->'))
            trouble[(a, b)] += int(count)
        except (TypeError, ValueError, AttributeError):
            pass
    return p, seeds, missing, trouble, protected, prev


def make_targets(old, x, y, mode, pivot=20):
    """Return a replacement window with exactly one extra tomato.

    The first old node is removed, the last is an unchanged right anchor.
    All other old nodes are retained. Operators can move a new node across
    a problematic anchor-like internal tomato (e.g. 20).
    """
    if len(old) < 2 or x == y or x in old or y in old:
        return None
    body = list(old[1:-1]); anchor = old[-1]
    if mode == 'front':
        seq = [x, y] + body
    elif mode == 'front_rev':
        seq = [y, x] + body
    elif mode == 'back':
        seq = body + [x, y]
    elif mode == 'back_rev':
        seq = body + [y, x]
    elif mode == 'split':
        seq = [x] + body + [y]
    elif mode == 'split_rev':
        seq = [y] + body + [x]
    elif mode in ('after_pivot_xy', 'after_pivot_yx', 'straddle_pivot', 'straddle_pivot_rev'):
        if pivot not in body:
            return None
        j = body.index(pivot)
        if mode == 'after_pivot_xy':
            seq = body[:j+1] + [x, y] + body[j+1:]
        elif mode == 'after_pivot_yx':
            seq = body[:j+1] + [y, x] + body[j+1:]
        elif mode == 'straddle_pivot':
            seq = body[:j] + [x, pivot, y] + body[j+1:]
        else:
            seq = body[:j] + [y, pivot, x] + body[j+1:]
    elif mode in ('middle_xy','middle_yx'):
        j = len(body)//2
        seq = body[:j] + ([x,y] if mode=='middle_xy' else [y,x]) + body[j:]
    else:
        return None
    result = seq + [anchor]
    if len(result) != len(old)+1 or len(set(result)) != len(result):
        return None
    return tuple(result)


MODES = ('front','front_rev','back','back_rev','split','split_rev',
         'after_pivot_xy','after_pivot_yx','straddle_pivot','straddle_pivot_rev',
         'middle_xy','middle_yx')


def plans_from_previous(seeds, prev):
    """Rescue and extend V12's 8/9-leg near miss before broad sampling."""
    near = prev.get('closest_repair') or {}
    cut = near.get('cut')
    pair = near.get('added') or []
    name = near.get('source')
    if not isinstance(cut, int) or len(pair) != 2:
        return []
    result = []
    for sidx,(source, state) in enumerate(seeds):
        if name and source != name:
            continue
        route = state['route']
        if not (1 <= cut < len(route)-1):
            continue
        if route[cut] != near.get('removed'):
            continue
        for end in sorted({min(len(route)-1, v) for v in (
                near.get('end', cut+6), cut+9, cut+13, cut+18, len(route)-1)}):
            if end <= cut:
                continue
            for x,y in (tuple(pair),tuple(reversed(pair))):
                for mode in MODES:
                    targets = make_targets(route[cut:end+1], x,y,mode)
                    if targets:
                        result.append((sidx, source, cut, end, x, y, mode, targets, 'v12_rescue'))
    return result


def generate_plans(seeds, missing, trouble, previous):
    """Prioritize difficult-edge-crossing operators and preserve source diversity."""
    priority = plans_from_previous(seeds, previous)
    seen = {(p[0],p[2],p[3],p[7]) for p in priority}
    broad = []
    for sidx,(name,state) in enumerate(seeds):
        route = state['route']
        likely = {int(previous.get('closest_repair',{}).get('cut',20))}
        for pivot in (20,31,37,40):
            if pivot in route:
                j=route.index(pivot)
                likely.update(range(max(1,j-2),min(len(route)-1,j+3)))
        likely.update((5,12,20,27,33))
        positions = sorted(i for i in likely if 1 <= i < len(route)-1)
        if not positions:
            continue
        pairs = [(x,y) for x in missing for y in missing if x!=y]
        RNG.shuffle(pairs)
        # Limit broad enumeration while explicitly retaining V12's near-miss pair.
        near_pair = tuple((previous.get('closest_repair') or {}).get('added') or ())
        forced = ([near_pair, near_pair[::-1]] if len(near_pair)==2 else [])
        pairs = list(dict.fromkeys(forced + pairs[:26]))
        for i in positions:
            for horizon in (5,9,15,25,len(route)):
                end=min(len(route)-1,i+horizon)
                if end<=i:
                    continue
                start,deadline,old,reserved,_,_=v9.window(state,i,end)
                needed=end-i+2  # two additions minus one existing
                slack=deadline-start-MIN_TOF*needed
                if slack<-1e-6:
                    continue
                specials=sum(t['status']=='SPECIAL' for t in old)
                for x,y in pairs:
                    if x in route or y in route:
                        continue
                    for mode in MODES:
                        targets=make_targets(route[i:end+1],x,y,mode)
                        if targets is None:
                            continue
                        key=(sidx,i,end,targets)
                        if key in seen:
                            continue
                        seen.add(key)
                        # More slack and SPECIAL legs give retiming flexibility.
                        # Penalize known recurring failed edges in the proposed route.
                        seq=(route[i-1],)+targets
                        failed=sum(min(trouble[(a,b)],30) for a,b in zip(seq[:-1],seq[1:]))
                        focus=(4 if 20 in targets else 0) + (2 if 37 in targets else 0)
                        score=(6*min(specials,3)+3*min(slack/max(needed,1),7)
                               -0.55*needed -0.2*failed +focus +RNG.uniform(0,7))
                        broad.append((score,(sidx,name,i,end,x,y,mode,targets,'broad')))
    broad.sort(reverse=True,key=lambda item:item[0])
    # Rescue portion first, diversified across horizon / operator; then broad plans.
    # Deliberately try extended windows crossing the failed edge first.
    prev_end = (previous.get('closest_repair') or {}).get('end', -1)
    priority.sort(key=lambda a: (a[3]>prev_end, 'pivot' in a[6], a[3]), reverse=True)
    # Important near-miss plans are examined first without exhausting attempts.
    result=priority[:min(70,MAX_ATTEMPTS//3)]
    used={(p[0],p[2],p[3],p[7]) for p in result}
    # Interleave broad candidates by source and pair to avoid one-source monopoly.
    groups={}
    for _, plan in broad:
        k=(plan[0],tuple(sorted((plan[4],plan[5]))))
        groups.setdefault(k,[]).append(plan)
    keys=list(groups)
    RNG.shuffle(keys)
    indices={key:0 for key in keys}
    while len(result)<MAX_ATTEMPTS and keys:
        next_keys=[]
        for key in keys:
            group=groups[key]
            j=indices[key]
            while j<len(group) and (group[j][0],group[j][2],group[j][3],group[j][7]) in used:
                j+=1
            indices[key]=j+1
            if j>=len(group):
                continue
            plan=group[j]
            used.add((plan[0],plan[2],plan[3],plan[7]))
            result.append(plan)
            if indices[key]<len(group):
                next_keys.append(key)
            if len(result)>=MAX_ATTEMPTS:
                break
        keys=next_keys
    return result


def write_once(path, payload):
    content=(json.dumps(payload,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with tmp.open('xb') as file:
        file.write(content)
    try:
        os.link(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)
    print('U ruajt:',path.name,flush=True)


def main():
    p,seeds,missing,trouble,protected,previous=load_inputs()
    print('Instanca: easy.kttsp | domate: 49',flush=True)
    print('Burimet:', [name for name,_ in seeds],flush=True)
    print('V12 near miss:',previous.get('closest_repair'),flush=True)
    if '--check-only' in sys.argv:
        print('CHECK-ONLY OK; rekordet nuk u ndryshuan.',flush=True)
        return
    if REPORT.exists() or CAND40.exists():
        raise RuntimeError('Raporti ose kandidati i V13 ekziston; nuk mbishkruhet.')
    plans=generate_plans(seeds,missing,trouble,previous)
    print('Plane unike:',len(plans),'| kufiri:',MAX_ATTEMPTS,flush=True)
    v4.MAX_EVALUATIONS=MAX_EVALUATIONS
    evaluator=v4.Evaluator(p)
    attempts=0; full=0; failures=Counter(); edges=Counter(); solver_stats=Counter()
    best_partial={'legs_repaired':0,'legs_needed':0}
    found=None; context=None; budget_hit=False; time_hit=False
    start_clock=time.monotonic()
    for sidx, source, i, end, x,y,mode,targets,origin in plans:
        if time.monotonic()-start_clock>MAX_SECONDS:
            time_hit=True
            break
        state=seeds[sidx][1]
        start,deadline,old,reserved,prefix,suffix=v9.window(state,i,end)
        cap=p.n_exceptions-reserved
        if cap<0:
            continue
        attempts+=1
        # Alternate repair formulations rather than using V10's narrow grid alone.
        solver=v10.repair_phase if attempts%3!=0 else v4.repair
        method='multi_label' if attempts%3!=0 else 'early_arrival'
        solver_stats[method]+=1
        try:
            repaired,reached,reason,edge=solver(
                p,evaluator,state['route'][i-1],list(targets),start,deadline,cap,old)
        except v4.BudgetReached:
            budget_hit=True
            break
        if (reached/max(1,len(targets)) >
            best_partial['legs_repaired']/max(1,best_partial['legs_needed'])):
            best_partial={'attempt': attempts,'origin':origin,'source':source,
                          'removed':state['route'][i],'added':[x,y],
                          'cut':i,'end':end,'operator':mode,'solver':method,
                          'legs_repaired':reached,'legs_needed':len(targets),
                          'failed_edge':edge}
        if repaired is None:
            failures[reason or 'unknown']+=1
            if edge:
                edges[f"{edge['from']}->{edge['to']}"]+=1
        else:
            full+=1
            cand=v9.stitch(state,i,end,list(targets),repaired,prefix,suffix)
            if (len(cand['route'])==40 and len(set(cand['route']))==40
                and len(cand['transfers'])==39 and v4.verify(p,cand)):
                # Check true 1-for-2 set difference, not just route length.
                if (len(set(cand['route'])-set(state['route']))==2 and
                    len(set(state['route'])-set(cand['route']))==1):
                    found=cand
                    context={'attempt':attempts,'source':source,
                             'removed':state['route'][i],'added':[x,y],
                             'cut':i,'end':end,'operator':mode,'solver':method}
                    print('GJETEM KANDIDAT 40/49!',context,flush=True)
                    break
            failures['full_verification_failed']+=1
        if attempts%PROGRESS_EVERY==0:
            print(f'V13: {attempts}/{MAX_ATTEMPTS} prova '
                  f'| {evaluator.count} eval | {full} riparime',flush=True)
    if found is not None and not v4.verify(p,found):
        raise RuntimeError('Kandidati 40 deshtoi ne riverifikim')
    for filename,digest in protected.items():
        if not Path(filename).is_file() or sha(Path(filename))!=digest:
            raise RuntimeError('Burimi i mbrojtur ka ndryshuar: '+filename)
    if found is not None:
        write_once(CAND40,{'level':40,'verified':True,
                           'source':'alns_v13_edge_rescue.py','frontier':[found]})
    report={'algorithm':'ALNS V13 edge rescue + extended-window dual DP',
            'base_level':39,'best_level':40 if found is not None else 39,
            'candidate_verified':found is not None,
            'attempts':attempts,'full_temporal_repairs':full,
            'transfer_evaluations':evaluator.count,'closest_repair':best_partial,
            'failure_reasons':dict(failures),'most_failed_edges':edges.most_common(15),
            'solvers':dict(solver_stats),'time_limit_reached':time_hit,
            'evaluation_limit_reached':budget_hit,'elapsed_seconds':round(time.monotonic()-start_clock,3),
            'seed':SEED,'candidate_context':context,
            'protected_38_sha256':protected[str(BASE38)],
            'protected_39_sha256':protected[str(BASE39)],
            'protected_v12_report_sha256':protected[str(V12REPORT)],
            'script_sha256':sha(Path(__file__))}
    write_once(REPORT,report)
    print('V13:',report['best_level'],'/49 | prova:',attempts,
          '| eval:',evaluator.count,'| riparime:',full,flush=True)
    print('Rekordet 38/39: TE PAPREKURA',flush=True)


if __name__=='__main__':
    main()
