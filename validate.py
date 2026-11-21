"""
Validate a stored solution.

    python validate.py easy            # own evaluator (numpy only)
    python validate.py easy --pykep    # ALSO check with pykep 2.x, exactly like the
                                       # official utils_users.compute_transfer

The pykep check is the same computation the ESA evaluator does, so if it says
"FEASIBLE" the submission will be accepted.
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import argparse
import json
from pathlib import Path

import numpy as np

from kttsp.problem import Problem

ROOT = Path(__file__).resolve().parent


def check_with_pykep(P, x):
    import pykep as pk  # needs pykep 2.x (NOT 3.x)
    MU_MOON = 4.904869500000000e12
    t0 = pk.epoch(P.p.t0)
    tomatoes = [pk.planet.keplerian(t0, list(row), MU_MOON, 0.0, 0.0, 0.0) for row in P.el]

    def compute_transfer(i_from, i_to, t_start, tof, max_revs=20):
        r_i, v_i = tomatoes[i_from].eph(float(t_start))
        r_j, v_j = tomatoes[i_to].eph(float(t_start + tof))
        best = float("inf")
        for cw in [False, True]:
            try:
                sol = pk.lambert_problem(r_i, r_j, tof * pk.DAY2SEC, MU_MOON, cw, max_revs)
            except Exception:
                continue
            for v1, v2 in zip(sol.get_v1(), sol.get_v2()):
                dv = np.linalg.norm(np.array(v1) - np.array(v_i)) + np.linalg.norm(np.array(v2) - np.array(v_j))
                best = min(best, dv)
        return best

    n = P.n
    t = x[: n - 1]
    tof = x[n - 1: 2 * n - 2]
    pi = [int(v) for v in x[2 * n - 2:]]
    dvs = np.array([compute_transfer(pi[k], pi[k + 1], t[k], tof[k]) for k in range(n - 1)])
    n_exc = int((dvs > P.p.dv_max).sum())
    ok = n_exc <= P.p.n_exc and dvs.max() <= P.p.dv_exc
    return ok, dvs, n_exc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("level", choices=["easy", "medium", "hard"])
    ap.add_argument("--file", default=None, help="json with {'x': [...]} or a submission file")
    ap.add_argument("--pykep", action="store_true")
    args = ap.parse_args()

    P = Problem(ROOT / "problems" / f"{args.level}.kttsp")
    fn = Path(args.file) if args.file else ROOT / "solutions" / f"{args.level}_best.json"
    d = json.loads(fn.read_text())
    x = d[0]["decisionVector"] if isinstance(d, list) else d["x"]
    x = np.array(x, dtype=float)

    ok, f, info = P.evaluate(x)
    print(f"[own evaluator]  feasible={ok}  objective f={f:.6f} days  "
          f"max dv={info['max_dv']:.3f} m/s  exceptions used={info['n_exc']}/{P.p.n_exc}")
    if args.pykep:
        okp, dvs, n_exc = check_with_pykep(P, x)
        diff = np.abs(dvs - info["dvs"]).max()
        print(f"[pykep check]    feasible={okp}  max dv={dvs.max():.3f}  exceptions={n_exc}  "
              f"max |dv_pykep - dv_own| = {diff:.2e} m/s")
        print("FEASIBLE" if okp else "NOT FEASIBLE")


if __name__ == "__main__":
    main()
