"""ESA SpOC4 / Keplerian Tomato TSP - Step 185, ALNS V15.

Adaptive, failure-guided 1-for-2 exchange for a verified 39/49 route.

Unlike V14, which selected a static list of plans, V15 starts from its
best partial repair (25/27 legs in the Kaggle experiment) and reacts to
actual failed links: it relocates either endpoint, changes neighboring order,
and retries with complementary V4/V10 time-repair solvers. Every complete
40/49 candidate is independently checked with official PyKEP transfers.

The V14/V13 reports and verified records remain read-only. Outputs are new
V15 files only; no existing results are overwritten.

A failed heuristic search does NOT prove that 40/49 is impossible.

Execute in Kaggle's PyKEP environment:
  python -u src/alns_v15_adaptive_edge_repair.py --check-only
  python -u src/alns_v15_adaptive_edge_repair.py
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import heapq
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
import alns_v14_bypass_23_4 as v14

BASE38 = ROOT / 'results/best_verified_38.json'
BACKUP38 = ROOT / 'kaggle_backups/results/best_verified_38.json'
BASE39 = ROOT / 'results/best_verified_39.json'
V14REPORT = ROOT / 'results/alns_v14_bypass_23_4_report.json'
REPORT = ROOT / 'results/alns_v15_adaptive_edge_repair_report.json'
CAND40 = ROOT / 'results/alns_v15_verified_40_candidate.json'

SEED = 18515
MAX_ATTEMPTS = 210
MAX_EVALUATIONS = 130_000
MAX_RUNTIME_SECONDS = 600
MAX_QUEUE = 4000
MAX_BRANCHES_PER_FAILURE = 24
PROGRESS_EVERY = 20
rng = random.Random(SEED)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_exclusive(path, data):
    """Write only new files; never overwrite a previous result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(data, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
    tmp = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.tmp')
    with tmp.open('xb') as f:
        f.write(payload)
    try:
        os.link(tmp, path)  # raises if target already exists
    finally:
        tmp.unlink(missing_ok=True)
    print('U ruajt:', path.name, flush=True)


def edge_list(left, nodes):
    r = [left] + list(nodes)
    return list(zip(r[:-1], r[1:]))


def mutations(targets, failed_idx, forbidden):
    """Generate *new* local orderings around the actual failure, no duplicates.

    failed_idx is zero-based in targets, so the failed edge is either
    left -> targets[0] or targets[failed_idx-1] -> targets[failed_idx].
    """
    a = list(targets)
    n = len(a)
    j = min(max(0, int(failed_idx)), n - 1)
    picks = []

    def include(seq, label):
        t = tuple(seq)
        if len(t) == n and len(set(t)) == n and t != tuple(a):
            picks.append((t, label))

    # Relocate the failing destination to *before or after* a different neighbor.
    for old in (j, max(0, j-1)):
        for displacement in (-5, -3, -2, -1, 1, 2, 3, 5):
            move = a.copy()
            item = move.pop(old)
            new = min(max(0, old + displacement), len(move))
            move.insert(new, item)
            include(move, 'relocate')

    # A 2-opt and a 3-4 node rotation can break multiple problematic edges.
    for center in range(max(0, j-3), min(n, j+3)):
        if center + 1 < n:
            b = a.copy()
            b[center], b[center+1] = b[center+1], b[center]
            include(b, 'swap_adjacent')
        for width in (3, 4, 5):
            if center+width > n:
                continue
            b = a.copy()
            b[center:center+width] = reversed(b[center:center+width])
            include(b, 'reverse_block')
            c = a.copy()
            chunk = c[center:center+width]
            c[center:center+width] = chunk[1:] + chunk[:1]
            include(c, 'rotate_block')

    # Target insertion of specific new tomatoes near the missed edge.
    for tomato in (4, 17, 27, 35, 38):
        if tomato not in a:
            continue
        old = a.index(tomato)
        if abs(old-j) > 9:
            continue
        for displacement in (-5, -3, -1, 1, 3, 5):
            b = a.copy()
            x = b.pop(old)
            b.insert(min(max(0, old+displacement), len(b)), x)
            include(b, 'shift_target')

    # Avoid reproducing the V14 plan; diversify by resulting route, not operator.
    uniq = {}
    for targets2, label in picks:
        uniq.setdefault(targets2, label)
    entries = []
    for seq, label in uniq.items():
        # Prefer avoiding tested troublesome edges, but never treat them as
        # mathematically impossible in other time windows.
        new_edges = list(zip(seq[:-1], seq[1:]))
        banned_count = sum(pair in forbidden for pair in new_edges)
        entries.append((banned_count, seq, label))
    entries.sort(key=lambda x: (x[0], rng.random()))
    return [(seq, label) for _, seq, label in entries]


def load_inputs():
    p, seeds, missing, trouble, protected, prior_near, edge, probe = v14.get_inputs()
    if not V14REPORT.is_file():
        raise RuntimeError('Mungon raporti i V14')
    previous = json.loads(V14REPORT.read_text(encoding='utf-8'))
    close = previous.get('closest_repair') or {}
    if (previous.get('base_level') != 39 or previous.get('best_level') != 39
            or previous.get('candidate_verified') is not False
            or not isinstance(close.get('failed_edge'), dict)):
        raise RuntimeError('Raporti V14 nuk permban prove te afert te besueshme')
    if not isinstance(close.get('added'), list) or len(close['added']) != 2:
        raise RuntimeError('Mungojne domatet e proves me te afert te V14')
    for path in (BASE38, BACKUP38, BASE39, V14REPORT):
        if not path.is_file():
            raise RuntimeError('Mungon burimi: ' + str(path))
        protected[str(path)] = sha(path)
    if BASE38.read_bytes() != BACKUP38.read_bytes():
        raise RuntimeError('Baza 38 nuk perputhet me backup-in')
    if previous.get('protected_39_sha256') != sha(BASE39):
        raise RuntimeError('Baza 39 nuk perputhet me V14')
    if p.n_tomatoes != 49 or len(missing) != 10:
        raise RuntimeError('Nuk eshte instanca easy.kttsp me 49 domate')

    forbidden = {(int(edge['from']), int(edge['to']))}
    fail = close['failed_edge']
    forbidden.add((int(fail['from']), int(fail['to'])))
    for pair, hits in previous.get('most_failed_edges', []):
        if int(hits) >= 10:
            try:
                u, v = map(int, pair.split('->'))
            except (TypeError, ValueError):
                continue
            trouble[(u,v)] += int(hits)
            forbidden.add((u,v))

    # Recreate V14's original planned routes deterministically; do not
    # re-run its 220 old repairs. Only newly mutated sequences are evaluated.
    v14.RNG.seed(v14.SEED)
    old_plans = v14.plans(seeds, missing, trouble, prior_near,
                          (int(edge['from']), int(edge['to'])))
    if not old_plans:
        raise RuntimeError('Nuk u rikrijuan planet e V14')
    return p, seeds, missing, trouble, protected, previous, old_plans, forbidden


def initial_plans(seeds, previous, old_plans, forbidden):
    near = previous['closest_repair']
    match = []
    other = []
    for original_index, entry in enumerate(old_plans):
        score, sidx, source, cut, end, x, y, targets = entry
        near_match = (
            source == near['source'] and cut == near['cut']
            and end == near['end'] and set((x,y)) == set(near['added'])
        )
        if near_match:
            match.append((original_index, entry))
        elif len(other) < 70:
            other.append((original_index, entry))
    # Keep a few distinct neighborhoods from the proven V14 failure.
    # The original V14 report does not store the complete attempted target
    # order. Reconstruct several local orders directly from its verified
    # source/cut/end/pair, even if V14's plan selection changes slightly.
    preferred_index = int(near['attempt'])-1
    match.sort(key=lambda item: (item[0] != preferred_index, abs(item[0]-preferred_index)))
    result = []
    for sidx, (source, state) in enumerate(seeds):
        if source != near['source']:
            continue
        cut, end, drop = near['cut'], near['end'], near['removed']
        if not (1 <= cut <= end < len(state['route'])):
            continue
        body = list(state['route'][cut:end+1])
        if body.count(drop) != 1:
            continue
        body.remove(drop)
        x, y = near['added']
        if x == y or x in state['route'] or y in state['route']:
            continue
        focus_at = set((0, len(body), len(body)//2))
        for node in (18, 17, 23, 4, 9, 34):
            if node in body:
                j = body.index(node)
                focus_at.update((max(0,j-2),j,j+1,min(len(body),j+2)))
        for j in sorted(focus_at):
            for k in (j, min(len(body)+1,j+1), max(0,j-2), len(body)//2):
                targets = v14.insert_pair(body,x,y,j,k)
                if len(targets) != end-cut+2:
                    continue
                left = state['route'][cut-1]
                bad_edge = (near['failed_edge']['from'],near['failed_edge']['to'])
                hits = [pos for pos,pair in enumerate(edge_list(left,targets))
                        if pair == bad_edge]
                if not hits:
                    hits = [max(0,len(targets)-2)]
                for at in hits[:1]:
                    for mutated,label in mutations(targets,at,forbidden)[:2]:
                        result.append({'sidx':sidx,'source':source,'cut':cut,
                                       'end':end,'x':x,'y':y,'targets':mutated,
                                       'kind':'reconstructed_'+label,
                                       'depth':0,'focus':True})
        break
    for original_index,entry in match[:12] + other[:20]:
        _,sidx,source,cut,end,x,y,targets = entry
        state = seeds[sidx][1]
        left = state['route'][cut-1]
        failed_edge = near['failed_edge'] if source == near['source'] else None
        if failed_edge is not None:
            indexes = [i for i, pair in enumerate(edge_list(left,targets))
                       if pair == (failed_edge['from'],failed_edge['to'])]
        else:
            indexes = []
        if not indexes:
            indexes = [max(0,len(targets)-2), max(0,len(targets)//2)]
        for j in indexes[:2]:
            for mutated,label in mutations(targets,j,forbidden)[:9]:
                result.append({'sidx':sidx,'source':source,'cut':cut,'end':end,
                               'x':x,'y':y,'targets':mutated,'kind':'v14_' + label,
                               'depth':0,'focus':bool(failed_edge)})
    return result


def main():
    p,seeds,missing,trouble,protected,previous,old_plans,forbidden=load_inputs()
    near=previous['closest_repair']
    print('Instanca: easy.kttsp (49 domate)',flush=True)
    print('Rekordi: 39/49 | SPECIAL:',seeds[0][1]['special_used'],flush=True)
    print('V14 prova me e afert:',near['legs_repaired'],'/',near['legs_needed'],
          '| failed:',near['failed_edge'],flush=True)
    initial=initial_plans(seeds,previous,old_plans,forbidden)
    if not initial:
        raise RuntimeError('Nuk u gjeneruan variante te reja V15')
    print('Plane fillestare te reja:',len(initial),flush=True)
    if '--check-only' in sys.argv:
        print('CHECK-ONLY OK. Nuk u shkrua asnje raport.',flush=True)
        return
    if REPORT.exists() or CAND40.exists():
        raise RuntimeError('Rezultatet V15 ekzistojne. Nuk mbishkruhen!')

    # Extend the adaptive priority queue with new plans after each real failure.
    queue = []
    seen = set()
    # Never re-evaluate an unchanged V14 plan, including a mutation that
    # happens to coincide with a different original V14 neighborhood.
    old_keys = {(entry[1], entry[3], entry[4], entry[7]) for entry in old_plans}
    pushed = 0
    def push(item, bonus=0):
        nonlocal pushed
        key=(item['sidx'],item['cut'],item['end'],item['targets'])
        if key in seen or key in old_keys or len(queue)>=MAX_QUEUE:
            return False
        state=seeds[item['sidx']][1]
        if (len(item['targets']) != item['end']-item['cut']+2
                or len(set(item['targets'])) != len(item['targets'])):
            return False
        if len(set(item['targets']).intersection(state['route'][:item['cut']])):
            return False
        # Exactly 1 removed and 2 added relative to the full seed route.
        route=list(state['route'][:item['cut']])+list(item['targets'])+list(state['route'][item['end']+1:])
        oldset,newset=set(state['route']),set(route)
        if (len(route)!=40 or len(newset)!=40 or len(oldset-newset)!=1
                or len(newset-oldset)!=2 or route[0]!=0):
            return False
        # Do not admit the unmodified V14 plans; only the novel orders.
        score=20 if item.get('focus') else 0
        pairs=edge_list(state['route'][item['cut']-1],item['targets'])
        score-=sum(min(10,trouble[(a,b)])*0.55 for a,b in pairs)
        score-=14*sum((a,b) in forbidden for a,b in pairs)
        score-=2*item['depth']
        score+=bonus+0.05*rng.random()
        seen.add(key)
        pushed+=1
        heapq.heappush(queue,(-score,pushed,item))
        return True

    for plan in initial:
        push(plan)
    if not queue:
        raise RuntimeError('Nuk ka plan pas filtrimit')

    v4.MAX_EVALUATIONS=MAX_EVALUATIONS
    evaluator=v4.Evaluator(p)
    failures=Counter(); failed_edges=Counter(); solvers=Counter()
    closest={'legs_repaired':0,'legs_needed':0}
    attempted=0; repaired_count=0; found=None; found_context=None
    reached_time=False; reached_budget=False
    t0=time.monotonic()
    while queue and attempted<MAX_ATTEMPTS:
        if time.monotonic()-t0>MAX_RUNTIME_SECONDS:
            reached_time=True;break
        _,_,plan=heapq.heappop(queue)
        state=seeds[plan['sidx']][1]
        cut,end=plan['cut'],plan['end']
        start,deadline,old,reserved,prefix,suffix=v9.window(state,cut,end)
        cap=p.n_exceptions-reserved
        if cap<0 or deadline-start<.5*len(plan['targets'])-1e-6:
            failures['insufficient_window']+=1
            continue
        attempted+=1
        use_v10=(attempted%3!=0)
        solver=v10.repair_phase if use_v10 else v4.repair
        solvers['multi_label' if use_v10 else 'early_arrival']+=1
        try:
            new, reached, reason, bad=solver(p,evaluator,state['route'][cut-1],
                                             list(plan['targets']),start,deadline,cap,old)
        except v4.BudgetReached:
            reached_budget=True;break
        if reached / max(1,len(plan['targets'])) > closest['legs_repaired']/max(1,closest['legs_needed']):
            closest={'attempt':attempted,'source':plan['source'],
                     'cut':cut,'end':end,'removed':list(set(state['route']) - set(state['route'][:cut]) - set(plan['targets']) - set(state['route'][end+1:])),
                     'added':[plan['x'],plan['y']],'kind':plan['kind'],
                     'depth':plan['depth'],'legs_repaired':reached,
                     'legs_needed':len(plan['targets']),'failed_edge':bad}
        if new is None:
            failures[reason or 'unknown']+=1
            if bad is not None:
                failed_edges[f"{bad['from']}->{bad['to']}"]+=1
                j=int(bad['repair_leg_index'])-1
                # Branch on the ACTUAL failing edge (rather than rerunning
                # a fixed sequence of random route mutations).
                changes=mutations(plan['targets'],j,forbidden)
                for targets2,label in changes[:MAX_BRANCHES_PER_FAILURE]:
                    child=dict(plan,targets=targets2,kind='rescue_'+label,
                               depth=plan['depth']+1,focus=True)
                    push(child,bonus=max(0,12-plan['depth']*3))
        else:
            repaired_count+=1
            candidate=v9.stitch(state,cut,end,list(plan['targets']),new,prefix,suffix)
            if (len(candidate['route'])==40 and len(set(candidate['route']))==40
                    and len(candidate['transfers'])==39
                    and len(set(candidate['route'])-set(state['route']))==2
                    and len(set(state['route'])-set(candidate['route']))==1
                    and v4.verify(p,candidate)):
                found=candidate
                found_context={'attempt':attempted,'source':plan['source'],
                               'kind':plan['kind'],'depth':plan['depth'],
                               'cut':cut,'end':end,'added':[plan['x'],plan['y']]}
                print('40/49 U GJET DHE U VERIFIKUA!',found_context,flush=True)
                break
            failures['candidate_verification_failed']+=1
        if attempted%PROGRESS_EVERY==0:
            print(f'V15: {attempted}/{MAX_ATTEMPTS} | eval={evaluator.count}'
                  f' | repaired={repaired_count} | queue={len(queue)}',flush=True)

    if any(not Path(path).exists() or sha(path)!=saved
           for path,saved in protected.items()):
        raise RuntimeError('Nje burim i mbrojtur ndryshoi')
    if found is not None:
        if not v4.verify(p,found):
            raise RuntimeError('Riverifikimi i 40/49 deshtoi')
        write_exclusive(CAND40,{'level':40,'verified':True,
                                'source':'alns_v15_adaptive_edge_repair.py',
                                'frontier':[found]})
    report={
        'algorithm':'ALNS V15 failure-guided edge relocation and temporal repair',
        'base_level':39,'best_level':40 if found is not None else 39,
        'candidate_verified':found is not None,
        'attempts':attempted,'transfer_evaluations':evaluator.count,
        'full_temporal_repairs':repaired_count,
        'initial_plans':len(initial),'unique_plans_queued':pushed,
        'solver_usage':dict(solvers),'failure_reasons':dict(failures),
        'most_failed_edges':failed_edges.most_common(12),
        'closest_repair':closest,'candidate_context':found_context,
        'protected_38_sha256':protected[str(BASE38)],
        'protected_39_sha256':protected[str(BASE39)],
        'protected_v14_report_sha256':protected[str(V14REPORT)],
        'script_sha256':sha(Path(__file__)),
        'time_limit_reached':reached_time,
        'evaluation_limit_reached':reached_budget,
        'elapsed_seconds':round(time.monotonic()-t0,3),'seed':SEED,
    }
    write_exclusive(REPORT,report)
    print('V15 PERFUNDOI:',report['best_level'],'/49 |',attempted,
          'prova |',evaluator.count,'vleresime',flush=True)
    print('Rekordet 38/39: TE PAPREKURA',flush=True)

if __name__=='__main__':
    main()
