"""ESA SpOC4 / Step 186 — diagnostic orbital transfer atlas (no optimization).

For each of the 10 tomatoes absent from the verified 39/49 route, this script
samples PyKEP transfer delta-v on a fixed grid of departure epochs and times of
flight. Incoming/outgoing pairwise feasibility is NOT full-route feasibility.

Inputs (read-only): verified 38/49, its identical backup, verified 39/49.
Outputs (create-once): results/v16_orbital_feasibility_atlas_report.json.

Run with Kaggle's PyKEP interpreter:
  python -u src/v16_orbital_feasibility_atlas.py --check-only
  python -u src/v16_orbital_feasibility_atlas.py
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import alns_v4_dp_repair as v4

BASE38 = ROOT / 'results/best_verified_38.json'
BACKUP38 = ROOT / 'kaggle_backups/results/best_verified_38.json'
BASE39 = ROOT / 'results/best_verified_39.json'
REPORT = ROOT / 'results/v16_orbital_feasibility_atlas_report.json'

# Deliberately sparse survey across the entire 200-day horizon; NOT exhaustive.
DEPARTURES = (0.5, 2.5, 5.0, 10.0, 20.0, 35.0, 50.0,
              65.0, 80.0, 100.0, 120.0, 145.0, 170.0, 190.0)
TOFS = (0.5, 1.0, 1.5, 2.0, 4.0, 8.0)
FOCUS = ((12, 4), (23, 4), (41, 4), (46, 4), (2, 4),
         (25, 4), (17, 4), (4, 17), (18, 17), (20, 4))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs():
    for path in (BASE38, BACKUP38, BASE39):
        if not path.is_file():
            raise RuntimeError('Mungon: ' + str(path))
    if BASE38.read_bytes() != BACKUP38.read_bytes():
        raise RuntimeError('Backup-i 38/49 nuk perputhet me rekordin!')
    p = v4.load_problem()
    if getattr(p, 'n_tomatoes', None) != 49:
        raise RuntimeError('Pritet instanca easy.kttsp me 49 domate')
    records = [json.loads(path.read_text(encoding='utf-8')) for path in (BASE38, BASE39)]
    for record, count in zip(records, (38, 39)):
        if record.get('level') != count or record.get('verified') is not True:
            raise RuntimeError(f'Rekordi {count} ka metadata te pasakta')
        if len(record.get('frontier', [])) != 1:
            raise RuntimeError('Frontier duhet te kete nje rruge')
        if not v4.verify(p, record['frontier'][0]):
            raise RuntimeError(f'Rekordi {count} deshtoi riverifikimin me PyKEP')
    covered = records[1]['frontier'][0]['route']
    missing = sorted(set(range(p.n_tomatoes)) - set(covered))
    if len(covered) != 39 or len(set(covered)) != 39 or len(missing) != 10:
        raise RuntimeError('Nuk u konfirmuan 39 te vizituara dhe 10 qe mungojne')
    sources = {str(path): sha(path) for path in (BASE38, BACKUP38, BASE39)}
    return p, covered, missing, sources


def category(p, dv):
    if math.isfinite(dv) and dv >= 0:
        if dv < p.dv_threshold - 0.001:
            return 'NORMAL'
        if dv < p.exception_threshold - 0.001:
            return 'SPECIAL'
    return None


def sample_edge(p, a, b):
    best = None
    best_normal = None
    best_special = None
    n_normal = n_special = n_eval = n_errors = 0
    for departure in DEPARTURES:
        for tof in TOFS:
            if departure + tof > 200 + 1e-9:
                continue
            n_eval += 1
            try:
                dv = float(p.compute_transfer(int(a), int(b), float(departure), float(tof)))
            except Exception:
                n_errors += 1
                continue
            if not math.isfinite(dv) or dv < 0:
                continue
            example = {'departure': departure, 'tof': tof, 'dv': dv}
            if best is None or dv < best['dv']:
                best = example
            cat = category(p, dv)
            if cat == 'NORMAL':
                n_normal += 1
                if best_normal is None or dv < best_normal['dv']:
                    best_normal = example
            elif cat == 'SPECIAL':
                n_special += 1
                if best_special is None or dv < best_special['dv']:
                    best_special = example
    return {
        'from': a, 'to': b, 'samples': n_eval, 'exceptions': n_errors,
        'min_dv': None if best is None else best['dv'],
        'best_sample': best,
        'normal_samples': n_normal, 'special_samples': n_special,
        'best_normal_sample': best_normal,
        'best_special_sample': best_special,
    }


def edge_summary(edges, neighbors):
    # One neighbor counts once regardless of how many sample times work.
    relevant = [e for e in edges if (e['from'] in neighbors or e['to'] in neighbors)]
    normal = [e for e in relevant if e['normal_samples'] > 0]
    special = [e for e in relevant if e['special_samples'] > 0]
    possible = [e for e in relevant if e['normal_samples'] or e['special_samples']]
    possible.sort(key=lambda e: (
        0 if e['normal_samples'] else 1,
        e['min_dv'] if e['min_dv'] is not None else float('inf')
    ))
    return {
        'normal_neighbor_count': len(normal),
        'special_neighbor_count': len(special),
        'feasible_neighbor_count': len(possible),
        'normal_sample_count': sum(e['normal_samples'] for e in relevant),
        'special_sample_count': sum(e['special_samples'] for e in relevant),
        'top_feasible_edges': possible[:5],
    }


def write_once(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(obj, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    # Create exclusively and atomically; interrupted writes never replace records.
    temp = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.tmp')
    with temp.open('xb') as out:
        out.write(payload)
    try:
        os.link(temp, path)  # fails if the destination already exists
    finally:
        temp.unlink(missing_ok=True)
    print('Raporti u krijua:', path, flush=True)


def main():
    p, covered, missing, protected = load_inputs()
    print('HAPI 186 — HARTA DIAGNOSTIKUESE ORBITALE', flush=True)
    print('Domatet qe mungojne:', missing, flush=True)
    print('NORMAL:', p.dv_threshold, '| SPECIAL:', p.exception_threshold,
          '| buxheti:', p.n_exceptions, flush=True)
    if '--check-only' in sys.argv:
        print('CHECK-ONLY OK: asnje skedar nuk ndryshoi', flush=True)
        return
    if REPORT.exists():
        raise RuntimeError('Raporti ekziston; nuk do te mbishkruhet')

    # Cache pairwise evaluations; pairs between missing tomatoes may recur.
    cache = {}
    covered_set = set(covered)
    counts = {'orbital_evaluations': 0, 'exceptions': 0}
    started = time.monotonic()

    def edge(a, b):
        key = (a, b)
        if key not in cache:
            result = sample_edge(p, a, b)
            cache[key] = result
            counts['orbital_evaluations'] += result['samples']
            counts['exceptions'] += result['exceptions']
        return cache[key]

    rows = []
    for m in missing:
        inbound = [edge(a, m) for a in range(49) if a != m]
        outbound = [edge(m, b) for b in range(49) if b != m]
        covered_in = edge_summary(inbound, covered_set)
        covered_out = edge_summary(outbound, covered_set)
        missing_neighbors = set(missing) - {m}
        missing_in = edge_summary(inbound, missing_neighbors)
        missing_out = edge_summary(outbound, missing_neighbors)
        row = {
            'tomato': m,
            'incoming_from_covered': covered_in,
            'outgoing_to_covered': covered_out,
            'incoming_from_other_missing': missing_in,
            'outgoing_to_other_missing': missing_out,
            'conservative_pair_score': min(
                covered_in['feasible_neighbor_count'], covered_out['feasible_neighbor_count']
            ),
        }
        rows.append(row)
        print('Domatja', m,
              '| hyrje nga te vizituarat:', covered_in['feasible_neighbor_count'],
              '| dalje tek te vizituarat:', covered_out['feasible_neighbor_count'],
              '| NORMAL hyrje/dalje:',
              covered_in['normal_neighbor_count'], '/',
              covered_out['normal_neighbor_count'],
              '| eval:', counts['orbital_evaluations'], flush=True)

    ranking = sorted(rows, key=lambda x: (
        -x['conservative_pair_score'],
        -(x['incoming_from_covered']['normal_neighbor_count'] +
          x['outgoing_to_covered']['normal_neighbor_count']),
        x['tomato'],
    ))
    selected = [{'tomato': x['tomato'],
                 'conservative_pair_score': x['conservative_pair_score'],
                 'incoming_covered': x['incoming_from_covered']['feasible_neighbor_count'],
                 'outgoing_covered': x['outgoing_to_covered']['feasible_neighbor_count']}
                for x in ranking]
    focus_edges = [edge(a, b) for a, b in FOCUS]

    if any(not Path(path).is_file() or sha(path) != digest
           for path, digest in protected.items()):
        raise RuntimeError('Nje nga rekordet e mbrojtura ndryshoi')
    obj = {
        'algorithm': 'V16 diagnostic pairwise orbital feasibility atlas',
        'diagnostic_only': True,
        'base_level': 39,
        'target_level': 40,
        'instance': 'easy.kttsp',
        'normal_limit': float(p.dv_threshold),
        'special_limit': float(p.exception_threshold),
        'departure_grid': DEPARTURES,
        'tof_grid': TOFS,
        'not_a_route_proof': (
            'Pairwise samples do not enforce full route timing or the global SPECIAL budget; '
            'lack of samples under threshold is not proof of impossibility.'
        ),
        'missing_tomatoes': missing,
        'comparative_ranking': selected,
        'by_missing_tomato': rows,
        'focused_edges': focus_edges,
        'pair_count': len(cache),
        'orbital_evaluations': counts['orbital_evaluations'],
        'exceptions': counts['exceptions'],
        'elapsed_seconds': round(time.monotonic() - started, 3),
        'protected_38_sha256': protected[str(BASE38)],
        'protected_39_sha256': protected[str(BASE39)],
        'source_sha256': sha(Path(__file__)),
    }
    write_once(REPORT, obj)
    print('RANKING ORIENTUES:', selected[:5], flush=True)
    print('V16 PERFUNDOI - nuk u krijua kandidati 40 dhe nuk u ndryshuan rekordet', flush=True)


if __name__ == '__main__':
    main()
