"""
Usage:
    python solve.py --level easy   --time 300
    python solve.py --level medium --time 1200
    python solve.py --level hard   --time 7200

The SAME code solves all three levels; only the input file changes.
Writes solutions/<level>_best.json (decision vector) if it beats the stored one,
and submissions/<level>_submission.json ready to upload on optimise.esa.int.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from kttsp.problem import Problem
from kttsp.solver import Solver
from submission_helper import create_submission

CHALLENGE = "spoc-4-keplerian-tomato-traveling-salesperson"
# problem ids as used on optimise.esa.int (check the "Problems" tab if upload fails)
PROBLEM_ID = {"easy": "small", "medium": "medium", "hard": "large"}

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["easy", "medium", "hard"], required=True)
    ap.add_argument("--time", type=float, default=600, help="time budget in seconds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=int, default=160)
    ap.add_argument("--no-ls", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="do not start from the stored best solution")
    args = ap.parse_args()

    P = Problem(ROOT / "problems" / f"{args.level}.kttsp")
    print(f"{args.level}: {P.n} tomatoes, params {P.p}")
    t0 = time.time()
    sol_dir = ROOT / "solutions"
    sol_dir.mkdir(exist_ok=True)
    sub_dir = ROOT / "submissions"
    sub_dir.mkdir(exist_ok=True)
    best_file = sol_dir / f"{args.level}_best.json"

    def save_if_better(x):
        """Checkpoint: validate x and store it if it beats the stored solution."""
        ok, f, info = P.evaluate(x)
        if not ok:
            print(f"   (candidate rejected by evaluator: {info if isinstance(info, str) else 'dv'})")
            return False
        old = json.loads(best_file.read_text()) if best_file.exists() else None
        if old is not None and f >= old["f"] - 1e-9:
            return False
        best_file.write_text(json.dumps({"f": f, "x": x}))
        create_submission(CHALLENGE, PROBLEM_ID[args.level], x,
                          str(sub_dir / f"{args.level}_submission.json"),
                          f"kttsp-{args.level}", f"f={f:.4f}")
        print(f"   ** new best f={f:.4f} saved -> {best_file.name}")
        return True

    init_order = None
    if best_file.exists() and not args.fresh:
        xs = json.loads(best_file.read_text())["x"]
        init_order = [int(v) for v in xs[2 * (P.n - 1):]]
        print(f"starting from stored best (f={json.loads(best_file.read_text())['f']:.4f}); use --fresh to ignore it")
    S = Solver(P, seed=args.seed, window=args.window)
    x, f = S.solve(time_limit=args.time, ls=not args.no_ls, on_best=save_if_better, init_order=init_order)
    ok, f_chk, info = P.evaluate(x)
    if ok:
        print(f"done in {time.time()-t0:.0f}s  f={f_chk:.4f}  feasible=True  "
              f"max_dv={info['max_dv']:.2f} exceptions={info['n_exc']}")
        save_if_better(x)
    else:
        print(f"done in {time.time()-t0:.0f}s  final solution rejected ({info}); best stored one kept")
    d = json.loads(best_file.read_text())
    print(f"==> BEST solution for {args.level}: f={d['f']:.4f} days  (solutions/{args.level}_best.json, "
          f"submissions/{args.level}_submission.json)")


if __name__ == "__main__":
    main()
