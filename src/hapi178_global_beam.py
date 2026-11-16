#!/usr/bin/env python3
"""SpOC4 Small: phase-aware global/suffix beam from verified prefixes."""
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path('/kaggle/working/esa-spoc4')
RESULTS = ROOT / 'results'
DATA = ROOT / 'data/easy.kttsp'
BASE = RESULTS / 'best_verified_38.json'
BEST = RESULTS / 'hapi178_global_best.json'
STATUS = RESULTS / 'hapi178_global_status.json'
TIME_BUDGET = int(os.environ.get('ESA_SECONDS', '150'))
WIDTH = int(os.environ.get('ESA_BEAM', '65'))
TARGET_LIMIT = int(os.environ.get('ESA_TARGETS', '23'))
EPS = 1e-6

# Sample waits and flight durations (days). Explore different orbital phases.
WAITS = (0.0, .5, 1., 2., 4., 8., 13., 20.)
TOFS = (.25, .5, 1., 1.5, 2., 3., 4., 6., 9.)
OPTIONS = sorted([(w, f) for w in WAITS for f in TOFS if w+f <= 25], key=lambda x: x[0] + x[1])


def atomic_json(path, content):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def read_instance():
    desc, elements = None, []
    for line in DATA.read_text(encoding='utf-8').splitlines():
        parts = line.split()
        if not parts or parts[0] == 'c':
            continue
        if parts[0] == 'p':
            desc = list(map(float, parts[2:8]))
        else:
            elements.append(list(map(float, parts)))
    assert desc and len(elements) == 49, 'Instanca nuk ka 49 domate'
    return desc, elements


@dataclass(frozen=True)
class State:
    route: tuple
    legs: tuple
    mask: int
    t: float
    specials: int


def state_from(route, legs, t0, normal):
    assert len(legs) == len(route)-1
    mask = 0
    for x in route:
        assert not (mask & (1 << x)), 'Domate e perseritur'
        mask |= 1 << x
    end = float(legs[-1]['start_time']) + float(legs[-1]['tof']) if legs else t0
    specials = sum(float(z['dv']) > normal + EPS for z in legs)
    return State(tuple(route), tuple(legs), mask, end, specials)


def selected(states, width, rarity, rng):
    # Diversity over last vertex, special budget and arrival phase.
    weighted = []
    for s in states:
        r = sum(rarity[x] for x in s.route)
        score = s.t + 7.0*s.specials - 1.5*r + rng.random()*1.0
        weighted.append((score, rng.random(), s))
    weighted.sort(key=lambda z: (z[0], z[1]))
    output, used = [], {}
    for _, _, s in weighted:
        key = (s.route[-1], s.specials, int(s.t // 4))
        if used.get(key, 0) >= 2:
            continue
        output.append(s)
        used[key] = used.get(key, 0) + 1
        if len(output) == width:
            return output
    for _, _, s in weighted:
        if len(output) == width:
            break
        if s not in output:
            output.append(s)
    return output


def main():
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    sys.path.insert(0, str(ROOT / 'src'))
    from utils_users import TomatoProblem
    assert BASE.is_file() and DATA.is_file()
    m, elements = read_instance()
    t0, min_tof, max_t, normal, special, max_spec = m
    max_spec = int(max_spec)
    source = json.loads(BASE.read_text(encoding='utf-8'))
    assert source.get('verified') and source.get('level') == 38
    old = source['frontier'][0]
    baseline = state_from(old['route'], old['transfers'], t0, normal)
    assert len(baseline.route) == 38 and baseline.specials == 5
    problem = TomatoProblem(str(DATA))
    rng = random.Random(1782026)
    started = time.monotonic()
    until = started + TIME_BUDGET

    @lru_cache(maxsize=400000)
    def delta(a, b, depart, tof):
        try:
            value = float(problem.compute_transfer(a, b, depart, tof))
            return value if math.isfinite(value) and value >= 0 else math.inf
        except (ValueError, OverflowError, RuntimeError, ZeroDivisionError):
            return math.inf

    # Orbital similarity and a static scarcity indicator; no expensive training.
    a_values = [x[0] for x in elements]
    incl = [x[2] for x in elements]
    def distance(i, j):
        return 1.7*abs(math.log(a_values[i]/a_values[j])) + abs(incl[i]-incl[j])
    rarity = [2.0 / (1.0 + sum(distance(i, j) < .35 for j in range(49) if i != j)) for i in range(49)]

    def verify(s):
        if len(s.route) != len(set(s.route)) or s.route[0] != 0 or len(s.legs) != len(s.route)-1:
            return False
        now = t0
        specials = 0
        for index, leg in enumerate(s.legs):
            dep, tof = float(leg['start_time']), float(leg['tof'])
            if (leg['from'], leg['to']) != (s.route[index], s.route[index+1]):
                return False
            if dep < now - EPS or tof < min_tof - EPS:
                return False
            v = delta(int(leg['from']), int(leg['to']), round(dep, 6), round(tof, 6))
            if v > special + EPS:
                return False
            specials += v > normal + EPS
            now = dep + tof
        return specials <= max_spec and now <= max_t + EPS

    best = baseline
    if BEST.is_file():
        try:
            saved = json.loads(BEST.read_text(encoding='utf-8'))
            restored = state_from(saved['route'], saved['transfers'], t0, normal)
            if verify(restored) and (len(restored.route), -restored.t) > (len(best.route), -best.t):
                best = restored
        except (KeyError, ValueError, TypeError, AssertionError):
            pass
    print('HAPI 178 - GLOBAL PHASE DIVERSIFIED BEAM', flush=True)
    print('Baza: 38/49; SPECIAL: 5/5; dita: %.3f' % baseline.t, flush=True)
    print('Koha e kerkimit:', TIME_BUDGET, 'sekonda; Beam:', WIDTH, flush=True)
    print('Skedari baze nuk ndryshohet.', flush=True)

    def persist(s):
        nonlocal best
        if (len(s.route), -s.t) <= (len(best.route), -best.t):
            return
        if not verify(s):
            return
        best = s
        atomic_json(BEST, {
            'method': 'hapi178_global_phase_beam', 'verified': True,
            'route': list(s.route), 'transfers': list(s.legs),
            'level': len(s.route), 'special_used': s.specials,
            'current_time': s.t, 'remaining': sorted(set(range(49)) - set(s.route))
        })
        print('*** PROGRES: %d/49 | SPECIAL=%d | DITA=%.4f ***' %
              (len(s.route), s.specials, s.t), flush=True)
        if len(s.route) == 49:
            x = ([float(z['start_time']) for z in s.legs] +
                 [float(z['tof']) for z in s.legs] + list(s.route))
            assert len(x) == 145
            atomic_json(RESULTS/'hapi178_chromosome_145.json', x)
            print('Zgjidhje 49/49! Vektori 145 u ruajt.', flush=True)

    # Build verified baseline prefixes, and optionally best partial prefixes.
    seeds = {n: [] for n in range(1, 50)}
    for source_state in (baseline, best):
        for n in range(1, len(source_state.route)+1):
            s = state_from(source_state.route[:n], source_state.legs[:n-1], t0, normal)
            seeds[n].append(s)

    frontier = seeds[1]
    full = False
    expansions = 0
    depth = 1
    try:
        for depth in range(1, 49):
            if time.monotonic() >= until:
                break
            frontier = selected(frontier + seeds[depth], WIDTH, rarity, rng)
            generated = []
            for s in frontier:
                if time.monotonic() >= until:
                    break
                remaining = [i for i in range(49) if not (s.mask >> i) & 1]
                if not remaining:
                    persist(s)
                    full = True
                    break
                # Mixed targets: nearby orbits, rare ones, and random exploration.
                nearby = sorted(remaining, key=lambda x: distance(s.route[-1], x))[:15]
                hard = sorted(remaining, key=lambda x: -rarity[x])[:5]
                pool = list(dict.fromkeys(nearby + hard))
                others = [x for x in remaining if x not in pool]
                rng.shuffle(others)
                targets = (pool + others[:max(0, TARGET_LIMIT-len(pool))])[:TARGET_LIMIT]
                for j in targets:
                    normal_options, special_options = [], []
                    for wait, tof in OPTIONS:
                        dep = round(s.t + wait, 6)
                        arr = dep + tof
                        if arr > max_t + EPS or tof < min_tof:
                            continue
                        dv = delta(s.route[-1], j, dep, tof)
                        if dv <= normal + EPS:
                            normal_options.append((arr, dep, tof, dv))
                        elif dv <= special + EPS and s.specials < max_spec:
                            special_options.append((arr, dep, tof, dv))
                    # Two different arrival phases preserve flexibility.
                    offers = normal_options[:1]
                    if normal_options:
                        later = next((z for z in normal_options[1:]
                                      if z[0] >= normal_options[0][0] + 1.0), None)
                        if later:
                            offers.append(later)
                    if special_options:
                        offers += special_options[:1]
                    for arr, dep, tof, dv in offers:
                        is_special = int(dv > normal + EPS)
                        tr = {
                            'from': int(s.route[-1]), 'to': int(j),
                            'start_time': float(dep), 'tof': float(tof),
                            'dv': float(dv),
                            'status': 'SPECIAL' if is_special else 'NORMAL'
                        }
                        next_state = State(s.route+(j,), s.legs+(tr,),
                                           s.mask | (1 << j), float(arr),
                                           s.specials+is_special)
                        generated.append(next_state)
                        expansions += 1
            if full:
                break
            if not generated:
                frontier = []
                print('Thellesia %d: nuk u gjet kalim i ri.' % depth, flush=True)
                continue
            frontier = selected(generated, WIDTH, rarity, rng)
            for s in frontier:
                if len(s.route) >= len(best.route):
                    persist(s)
            if depth % 3 == 0 or depth == 48:
                print('Niveli %02d -> %02d | beam %d | best %d/49 | koha %.1fs | DV cache %s' %
                      (depth, depth+1, len(frontier), len(best.route),
                       time.monotonic()-started, delta.cache_info().currsize), flush=True)
            if len(best.route) == 49:
                full = True
                break
    finally:
        atomic_json(STATUS, {
            'version': 178, 'last_depth': depth, 'best_level': len(best.route),
            'best_day': best.t, 'specials': best.specials,
            'elapsed_sec': round(time.monotonic()-started, 2),
            'candidate_expansions': expansions,
            'cached_dv': delta.cache_info().currsize,
            'finished_49': len(best.route) == 49,
            'baseline_38_untouched': True
        })
        print('\nPERMBLEDHJE - HAPI 178', flush=True)
        print('Me i miri: %d/49, SPECIAL=%d, dita=%.4f' %
              (len(best.route), best.specials, best.t), flush=True)
        print('Rezultati:', BEST if BEST.exists() else 'rekordi origjinal 38/49', flush=True)
        print('Statusi:', STATUS, flush=True)

if __name__ == '__main__':
    main()
