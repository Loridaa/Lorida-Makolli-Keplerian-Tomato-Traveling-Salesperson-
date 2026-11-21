"""
Static (time-independent) guide order.

We estimate for every pair of orbits a lower-bound-like delta-v that ignores
phasing: plane-change cost between the two orbital planes, plus the Hohmann
cost when the semi-major axes differ.  Then we solve a TSP *path* on this
metric (nearest neighbour + 2-opt + or-opt) where each edge that needs an
exception (> dv_max) is heavily penalised and edges above the exception limit
are forbidden.  The resulting order tells the time-dependent search which
"region" of orbits to work on next (clusters, inclination ladder, ...).
"""
import numpy as np

from .problem import MU_MOON


def static_cost(P):
    a = P.a
    v = np.sqrt(MU_MOON / a)
    h = P.h_hat
    cosang = np.clip(h @ h.T, -1, 1)
    ang = np.arccos(cosang)
    ai, aj = a[:, None], a[None, :]
    vi, vj = v[:, None], v[None, :]
    # Hohmann between circular orbits ai -> aj
    at = (ai + aj) / 2
    dv1 = np.abs(np.sqrt(MU_MOON * (2 / ai - 1 / at)) - vi)
    dv2 = np.abs(vj - np.sqrt(MU_MOON * (2 / aj - 1 / at)))
    hoh = dv1 + dv2
    vmin = np.minimum(vi, vj)  # plane change done at the slowest point
    plane = 2 * vmin * np.sin(ang / 2)
    same = np.abs(ai - aj) / np.maximum(ai, aj) < 0.05
    est = np.where(same, plane, np.sqrt(hoh ** 2 + plane ** 2))
    np.fill_diagonal(est, 0.0)
    return est


def edge_weight(est, dv_max, dv_exc):
    w = est.copy()
    w = np.where(est > 0.85 * dv_max, w + 2000.0, w)
    w = np.where(est > 0.92 * dv_exc, w + 1e6, w)
    return w


def path_cost(order, W):
    return W[order[:-1], order[1:]].sum()


def nearest_neighbour(W, start):
    n = W.shape[0]
    seen = np.zeros(n, bool)
    order = [start]
    seen[start] = True
    for _ in range(n - 1):
        d = W[order[-1]].copy()
        d[seen] = np.inf
        j = int(np.argmin(d))
        order.append(j)
        seen[j] = True
    return np.array(order)


def two_opt(order, W, max_pass=50):
    """2-opt for an open path (endpoints free)."""
    order = order.copy()
    n = len(order)
    for _ in range(max_pass):
        improved = False
        for i in range(0, n - 1):
            a = order[i - 1] if i > 0 else -1
            b = order[i]
            js = np.arange(i + 1, n)
            c = order[js]
            d = np.where(js + 1 < n, order[np.minimum(js + 1, n - 1)], -1)
            old = (W[a, b] if a >= 0 else 0.0) + np.where(d >= 0, W[c, np.maximum(d, 0)], 0.0)
            new = (W[a, c] if a >= 0 else 0.0) + np.where(d >= 0, W[b, np.maximum(d, 0)], 0.0)
            gain = old - new
            k = int(np.argmax(gain))
            if gain[k] > 1e-9:
                j = js[k]
                order[i: j + 1] = order[i: j + 1][::-1]
                improved = True
        if not improved:
            break
    return order


def or_opt(order, W, seg_lens=(1, 2, 3), max_pass=20):
    order = list(order)
    n = len(order)
    for _ in range(max_pass):
        improved = False
        for L in seg_lens:
            i = 0
            while i + L <= n:
                seg = order[i: i + L]
                prev = order[i - 1] if i > 0 else None
                nxt = order[i + L] if i + L < n else None
                rem_gain = (W[prev, seg[0]] if prev is not None else 0) + (W[seg[-1], nxt] if nxt is not None else 0) - (
                    W[prev, nxt] if (prev is not None and nxt is not None) else 0)
                rest = order[:i] + order[i + L:]
                best, best_pos, best_rev = 1e-9, None, False
                m = len(rest)
                for rev in (False, True):
                    s0, s1 = (seg[-1], seg[0]) if rev else (seg[0], seg[-1])
                    for p in range(m + 1):
                        a = rest[p - 1] if p > 0 else None
                        b = rest[p] if p < m else None
                        add = (W[a, s0] if a is not None else 0) + (W[s1, b] if b is not None else 0) - (
                            W[a, b] if (a is not None and b is not None) else 0)
                        g = rem_gain - add
                        if g > best:
                            best, best_pos, best_rev = g, p, rev
                if best_pos is not None:
                    s = seg[::-1] if best_rev else seg
                    order = rest[:best_pos] + s + rest[best_pos:]
                    improved = True
                i += 1
        if not improved:
            break
    return np.array(order)


def guide_order(P, rng=None, restarts=4):
    est = static_cost(P)
    W = edge_weight(est, P.p.dv_max, P.p.dv_exc)
    rng = np.random.default_rng(rng)
    best, best_c = None, np.inf
    starts = rng.choice(P.n, size=min(restarts, P.n), replace=False)
    for s in starts:
        o = nearest_neighbour(W, int(s))
        o = two_opt(o, W)
        if P.n <= 300:
            o = or_opt(o, W)
            o = two_opt(o, W)
        c = path_cost(o, W)
        if c < best_c:
            best, best_c = o, c
    return best, est
