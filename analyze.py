"""Print a per-region summary of a stored solution: python analyze.py hard"""
import json, sys
import numpy as np
from kttsp.problem import Problem

level = sys.argv[1]
P = Problem(f"problems/{level}.kttsp")
d = json.load(open(f"solutions/{level}_best.json"))
x = np.array(d["x"]); n = P.n
t = x[:n-1]; tof = x[n-1:2*n-2]; pi = x[2*n-2:].astype(int)
ok, f, info = P.evaluate(x)
print("feasible", ok, "f", f, "exceptions", info["n_exc"], "max dv", round(info["max_dv"], 2))
inc = np.degrees(P.inc)
def region(q):
    s = "L" if P.a[q] < 1e7 else "H"
    i = inc[q]
    if abs(i) < 3: r = "eq0"
    elif abs(i - 90) < 3: r = "pol"
    elif abs(i - 180) < 3: r = "eq180"
    else: r = "ladder"
    return s + "-" + r
prev = P.p.t0
tot = {}
cnt = {}
for k in range(n - 1):
    leg = t[k] + tof[k] - prev
    key = region(pi[k]) + " -> " + region(pi[k + 1])
    tot[key] = tot.get(key, 0) + leg
    cnt[key] = cnt.get(key, 0) + 1
    prev = t[k] + tof[k]
for k in sorted(tot, key=lambda z: -tot[z]):
    print(f"{k:28s} legs={cnt[k]:4d} total={tot[k]:8.2f} d  avg={tot[k]/cnt[k]:.3f} d")
