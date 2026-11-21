"""
Plot a stored solution: inclination of the visited tomato vs. time
(colour = shell). Useful for the report.

    python plot_solution.py hard      -> plots/hard.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kttsp.problem import Problem

ROOT = Path(__file__).resolve().parent
level = sys.argv[1]
P = Problem(ROOT / "problems" / f"{level}.kttsp")
d = json.loads((ROOT / "solutions" / f"{level}_best.json").read_text())
x = np.array(d["x"], float)
n = P.n
t, tof, pi = x[: n - 1], x[n - 1: 2 * n - 2], x[2 * n - 2:].astype(int)
ok, f, info = P.evaluate(x)
arr = np.r_[P.p.t0, t + tof]
inc = np.degrees(P.inc[pi])
low = P.a[pi] < 1e7

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(arr, inc, color="0.75", lw=0.8, zorder=1)
ax.scatter(arr[~low], inc[~low], s=10, c="#d62728", label="high orbit (a≈15 000 km)", zorder=2)
ax.scatter(arr[low], inc[low], s=10, c="#1f77b4", label="low orbit (a≈3 740 km)", zorder=2)
exc = np.where(info["dvs"] > P.p.dv_max)[0]
for k in exc:
    ax.axvline(arr[k + 1], color="orange", lw=1, ls="--")
ax.set_xlabel("arrival time [days]")
ax.set_ylabel("inclination of tomato [deg]")
ax.set_title(f"KTTSP {level}: {n} tomatoes, mission time f = {f:.3f} days "
             f"(orange = {len(exc)} high-Δv exceptions)")
ax.legend(loc="best")
fig.tight_layout()
out = ROOT / "plots"
out.mkdir(exist_ok=True)
fig.savefig(out / f"{level}.png", dpi=130)
print("saved", out / f"{level}.png")
