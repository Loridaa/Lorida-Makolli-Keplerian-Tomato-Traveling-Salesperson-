"""ESA SpOC4 -- Hapi 183: forensic probe of the closest V13 failed transfer.

This is a diagnostic, not a new optimization record. It replays one exact V13
plan with the original DP, traces states just before the failed orbital edge,
and explores feasible departure/flight-time alternatives under the same window.
No baseline or pre-existing result is modified; output is written once.

Run ONLY with isolated PyKEP Python, from repository root:
    /kaggle/working/esa-pykep-env/bin/python -u src/alns_v13_failure_probe.py
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import alns_v4_dp_repair as v4
import alns_v9_multistart as v9
import alns_v10_temporal_labels as v10
import alns_v13_edge_rescue as v13

REPORT_V13 = ROOT / 'results/alns_v13_edge_rescue_report.json'
REPORT = ROOT / 'results/alns_v13_failure_probe_report.json'
BASE38 = ROOT / 'results/best_verified_38.json'
BACKUP38 = ROOT / 'kaggle_backups/results/best_verified_38.json'
BASE39 = ROOT / 'results/best_verified_39.json'
MAX_EXTRA_EVAL = 5000


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_once(path, obj):
    data = (json.dumps(obj, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(data)


def main():
    if REPORT.exists():
        print('Raporti diagnostik ekziston; nuk po e mbishkruaj.')
        print('Raporti:', REPORT)
        return
    for p in (REPORT_V13, BASE38, BACKUP38, BASE39):
        if not p.is_file():
            raise RuntimeError('Mungon: ' + str(p))
    if BASE38.read_bytes() != BACKUP38.read_bytes():
        raise RuntimeError('Rekordi 38 ndryshon nga backup-i')
    protected = {str(p): sha(p) for p in (REPORT_V13, BASE38, BACKUP38, BASE39)}
    prev13 = json.loads(REPORT_V13.read_text(encoding='utf-8'))
    near = prev13.get('closest_repair') or {}
    expected_edge = near.get('failed_edge') or {}
    for key in ('attempt', 'source', 'cut', 'end', 'added', 'operator', 'legs_repaired', 'legs_needed'):
        if key not in near:
            raise RuntimeError(f'Raporti V13 nuk ka fushen {key}')
    if not all(key in expected_edge for key in ('from', 'to')):
        raise RuntimeError('Raporti V13 nuk ka lidhjen e deshtuar')

    p, seeds, missing, trouble, protected_v13, previous = v13.load_inputs()
    for filename, digest in protected_v13.items():
        protected[str(filename)] = digest
    v13.RNG.seed(v13.SEED)
    plans = v13.generate_plans(seeds, missing, trouble, previous)

    def matching(plan):
        idx, src, cut, end, x, y, mode, targets, origin = plan
        return (src == near['source'] and cut == near['cut'] and end == near['end']
                and mode == near['operator'] and [x, y] == near['added']
                and seeds[idx][1]['route'][cut] == near['removed'])

    matching_plans = [(i+1, plan) for i, plan in enumerate(plans) if matching(plan)]
    if not matching_plans:
        raise RuntimeError('Plani i proves me te afert nuk u rindertua; nuk bejme diagnoze te hamendesuar')
    # Prefer the original attempt number, otherwise log the matched plan explicitly.
    position, plan = next((item for item in matching_plans if item[0] == near['attempt']), matching_plans[0])
    idx, source, cut, end, x, y, mode, targets, origin = plan
    original = seeds[idx][1]
    start, deadline, oldlegs, reserved, prefix, suffix = v9.window(original, cut, end)
    special_cap = p.n_exceptions - reserved
    if special_cap < 0:
        raise RuntimeError('Buxhet i pavlefshem SPECIAL')

    failure_a, failure_b = int(expected_edge['from']), int(expected_edge['to'])
    reached_target = int(near['legs_repaired'])
    seen_labels = {}
    original_add_state = v4.add_state

    def trace_add_state(table, state):
        original_add_state(table, state)
        if len(state.get('legs', [])) == reached_target:
            used = int(state['used'])
            compact = {'arrival': float(state['t']), 'used_in_window': used,
                       'delta_v': float(state['dv'])}
            old = seen_labels.get(used)
            if old is None or (compact['arrival'], compact['delta_v']) < (old['arrival'], old['delta_v']):
                seen_labels[used] = compact

    class TrackingEvaluator(v4.Evaluator):
        def __init__(self, problem):
            super().__init__(problem)
            self.target_count = 0
            self.target_min_dv = math.inf
            self.target_pass = Counter()
        def get(self, a, b, depart, tof):
            dv = super().get(a, b, depart, tof)
            if a == failure_a and b == failure_b:
                self.target_count += 1
                if math.isfinite(dv):
                    self.target_min_dv = min(self.target_min_dv, dv)
                    category = v4.classify(p, dv)
                    if category:
                        self.target_pass[category[0]] += 1
            return dv

    evaluator = TrackingEvaluator(p)
    # Same solver the original attempt used (V13: every third uses V4 earliest arrival).
    solver = v4.repair if near['attempt'] % 3 == 0 else v10.repair_phase
    if solver is v4.repair:
        v4.add_state = trace_add_state
    try:
        repaired, reached, reason, edge = solver(
            p, evaluator, original['route'][cut-1], list(targets),
            start, deadline, special_cap, oldlegs)
    finally:
        v4.add_state = original_add_state

    print('=== HAPI 183 - DIAGNOSTIKA E V13 ===', flush=True)
    print('Burimi:', source, '| prova V13:', near['attempt'], '| plani:', position, flush=True)
    print('Dritarja:', cut, 'deri', end, '| afati:', round(deadline, 4), flush=True)
    print('Transferimi i deshtuar:', f'{failure_a}->{failure_b}', flush=True)
    print('Rindertim:', reached, '/', len(targets), '| shkaku:', reason, flush=True)
    if (repaired is not None or reached != near['legs_repaired']
            or (edge or {}).get('from') != failure_a
            or (edge or {}).get('to') != failure_b):
        raise RuntimeError('Riprodhimi nuk perputhet me raportin V13. Nuk ruajme analize te pasakte.')

    # For V4 earliest-arrival, the trace captures the earliest surviving state
    # per SPECIAL count just before the failing edge. No imagined arrival times.
    print('Gjendje para lidhjes:', len(seen_labels), flush=True)
    for label in sorted(seen_labels.values(), key=lambda q:(q['used_in_window'], q['arrival'])):
        print('  Arrival:', round(label['arrival'], 4), '| SPECIAL brenda dritares:', label['used_in_window'], flush=True)

    extra = 0
    by_kind = Counter()
    best = []
    min_dv = math.inf
    for label in sorted(seen_labels.values(), key=lambda q:q['arrival']):
        arrival = label['arrival']
        used = label['used_in_window']
        if arrival + .5 > deadline:
            continue
        departures = {round(arrival + d, 6) for d in
                      (0, .25, .5, 1, 1.5, 2, 3, 4, 6, 8, 10, 14)}
        # Coarser absolute-phase scan to detect missed windows.
        steps = int(max(0, math.floor((deadline - arrival - .5)/1.0)))
        departures.update(round(arrival + j*1.0, 6) for j in range(steps+1))
        durations = (.5,.75,1,1.25,1.5,2,2.5,3,4,5,6,8,10,12)
        for dep in sorted(departures):
            if dep < arrival - 1e-6 or dep + .5 > deadline + 1e-6:
                continue
            for tof in durations:
                if dep + tof > deadline + 1e-6:
                    continue
                if extra >= MAX_EXTRA_EVAL:
                    break
                dv = evaluator.get(failure_a, failure_b, dep, tof)
                extra += 1
                if not math.isfinite(dv):
                    by_kind['invalid_dv'] += 1
                    continue
                min_dv = min(min_dv, dv)
                category = v4.classify(p, dv)
                if category is None:
                    by_kind['above_exception_threshold'] += 1
                    continue
                status, special_inc = category
                if used + special_inc > special_cap:
                    by_kind['would_exceed_special_budget'] += 1
                else:
                    by_kind[f'feasible_{status.lower()}'] += 1
                    best.append({'depart':dep,'tof':tof,'dv':round(dv,6),
                                 'status':status,'arrival_before':arrival,
                                 'special_used_window':used})
            if extra >= MAX_EXTRA_EVAL:
                break
        if extra >= MAX_EXTRA_EVAL:
            break

    best.sort(key=lambda q:(q['status']!='NORMAL', q['depart']+q['tof'], q['dv']))
    result = {
        'diagnostic_only': True, 'base_level':39, 'v13_source_sha256':protected[str(REPORT_V13)],
        'target_edge':{'from':failure_a,'to':failure_b}, 'attempt':near['attempt'],
        'replayed_plan_position':position, 'replayed_legs_repaired':reached,
        'legs_needed':len(targets), 'replayed_failure_reason':reason,
        'window_start':start, 'window_deadline':deadline,
        'max_special_in_window':special_cap, 'arrival_labels':list(seen_labels.values()),
        'original_solver_edge_evaluations':evaluator.target_count-extra,
        'original_solver_min_dv':None if math.isinf(evaluator.target_min_dv) else evaluator.target_min_dv,
        'original_solver_threshold_pass_counts':dict(evaluator.target_pass),
        'extra_grid_evaluations':extra, 'extra_grid_result_counts':dict(by_kind),
        'extra_grid_min_dv':None if math.isinf(min_dv) else round(min_dv,6),
        'feasible_examples':best[:8],
        'dv_normal_limit':p.dv_threshold, 'dv_special_limit':p.exception_threshold,
        'protected_38_sha256':protected[str(BASE38)],
        'protected_39_sha256':protected[str(BASE39)]
    }
    for filename, digest in protected.items():
        if sha(Path(filename)) != digest:
            raise RuntimeError('Burim i mbrojtur ndryshoi: ' + filename)
    write_once(REPORT, result)
    print('\n--- DIAGNOSTIKA ---', flush=True)
    print('DV minimale ne skanim:', result['extra_grid_min_dv'], flush=True)
    print('Kufijt NORMAL/SPECIAL:', p.dv_threshold, '/', p.exception_threshold, flush=True)
    print('Opsione ne skanim:', dict(by_kind), flush=True)
    print('Shembuj te vlefshem:', best[:3], flush=True)
    print('RAPORTI:', REPORT, flush=True)
    print('REKORDET 38/39: TE PAPREKURA', flush=True)

if __name__ == '__main__':
    main()
