"""
Main solver (identical for easy / medium / hard).

Pipeline
--------
1. Static guide order (TSP path on a plane-change / Hohmann metric).
2. Time-dependent greedy construction: from the current tomato at the current
   time, look at the next W unvisited tomatoes of the guide order, screen them
   with a vectorised coarse Lambert grid and pick the one we can reach the
   earliest (exact earliest-arrival search on the best few).
3. Exception placement by dynamic programming on the fixed order
   (FIFO property: arriving earlier is never worse, so the earliest arrival per
   (position, exceptions used) is optimal).
4. Local search on the order (or-opt / swaps in a sliding window). Thanks to the
   FIFO property a move is accepted iff the arrival time at the first unchanged
   tomato after the window gets earlier; the change is then propagated forward
   only as long as arrival times keep changing.
"""
import time
import numpy as np

from .transfer import TransferSearch
from .static_order import guide_order


class Solver:
    def __init__(self, P, seed=0, window=160, safety=0.5, verbose=True, log_every=50):
        self.P = P
        self.rng = np.random.default_rng(seed)
        self.window = window
        self.S = TransferSearch(P, safety=safety)
        self.verbose = verbose
        self.log_every = log_every
        self.cache = {}
        self.guide, self.est = guide_order(P, seed)
        self.need_exc_thr = 0.85 * P.p.dv_max

    # ------------------------------------------------------------------ utils
    def log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def needs_exc(self, i, j):
        return self.est[i, j] > self.need_exc_thr

    def ea(self, i, j, t, exc=False, t_cap=np.inf):
        """Earliest arrival (arr, dep, tof, dv) for i->j leaving at >= t. Cached."""
        key = (int(i), int(j), float(t), bool(exc))
        if key in self.cache:
            r = self.cache[key]
            if r is None or r[0] <= t_cap:
                return r
            return None
        lim = self.P.p.dv_exc if exc else self.P.p.dv_max
        if not exc and self.needs_exc(i, j):
            r = None
        elif self.est[i, j] > self.P.p.dv_exc:
            r = None
        else:
            r = self.S.earliest(i, j, t, lim, t_cap=t_cap)
        if t_cap == np.inf or r is not None:
            self.cache[key] = r
        return r

    # ---------------------------------------------------------- screening
    def screen(self, i, cands, t, lims, div=20):
        """Coarse vectorised estimate of the earliest arrival to each candidate."""
        P = self.P
        cands = np.asarray(cands)
        per_i = P.period_days[i]
        per = np.minimum(per_i, P.period_days[cands])
        perM = np.maximum(per_i, P.period_days[cands])
        step = per.min() / div
        out = np.full(len(cands), np.inf)
        H = 0.5 * perM.min()
        lo = 0.0
        while H <= 6 * perM.max():
            arr = np.arange(lo + step, H + 1e-12, step)
            dep = np.arange(0.0, H, step)
            A, D = np.meshgrid(arr, dep, indexing="ij")
            F = A - D
            m = F >= P.p.min_tof
            A, D, F = A[m], D[m], F[m]
            todo = np.where(np.isinf(out))[0]
            if todo.size == 0:
                break
            g = A.size
            J = np.repeat(cands[todo], g)
            dv = P.dv(np.full(J.size, i), J, t + np.tile(D, todo.size), np.tile(F, todo.size)).reshape(todo.size, g)
            L = lims[todo][:, None]
            ok = dv <= L * 1.15
            arrs = np.where(ok, A[None, :], np.inf).min(axis=1)
            out[todo] = np.where(np.isfinite(arrs), t + arrs, np.inf)
            if np.isfinite(out).any():
                break
            lo = H
            H *= 2
        return out

    # -------------------------------------------------------- construction
    def construct(self, guide, n_exact=3, rand=0.0):
        P = self.P
        n = P.n
        visited = np.zeros(n, bool)
        cur = int(guide[0])
        visited[cur] = True
        t = P.p.t0
        order, deps, tofs, excs = [cur], [], [], []
        self._trace = (order, excs)
        exc_left = P.p.n_exc
        pos = {int(g): k for k, g in enumerate(guide)}
        seg = np.zeros(n, int)
        for k in range(1, n):
            seg[k] = seg[k - 1] + (1 if self.needs_exc(guide[k - 1], guide[k]) else 0)
        seg_exit = {int(seg[k]): int(guide[k]) for k in range(n - 1) if seg[k + 1] != seg[k]}
        t_start = time.time()
        while len(order) < n:
            # candidate window: next unvisited tomatoes of the current guide segment
            unv = [int(g) for g in guide if not visited[g]]
            p_min = pos[unv[0]]
            lag = pos[cur] - p_min
            sid = seg[p_min]
            same = [j for j in unv if seg[pos[j]] == sid]
            # candidates keep the invariant "remaining tomatoes of the segment can be
            # visited in guide order with cheap transfers" -> nothing gets stranded
            R = same
            ex_node = seg_exit.get(sid)
            cheap = []
            for q, j in enumerate(R):
                if len(cheap) >= self.window:
                    break
                if self.needs_exc(cur, j):
                    continue
                if len(R) > 1 and j == ex_node:
                    continue
                if len(R) > 1:
                    first = R[1] if q == 0 else R[0]
                    if self.needs_exc(j, first):
                        continue
                    if 0 < q < len(R) - 1 and self.needs_exc(R[q - 1], R[q + 1]):
                        continue
                cheap.append(j)
            if not cheap:
                # segment finished: jump to the next segment with an exception
                cands = same[:1] if same else unv[:1]
            else:
                cands = cheap
            cands = np.array(cands)
            use_exc = np.array([self.needs_exc(cur, j) for j in cands])
            need_rest = seg[-1] - sid
            if use_exc.any() and exc_left <= 0:
                raise RuntimeError(f"no exceptions left at node {cur}, visited {len(order)}")
            lims = np.where(use_exc, P.p.dv_exc, P.p.dv_max)
            est = self.screen(cur, cands, t, lims)
            # penalise skipping too far ahead in the guide (stranding)
            rank = np.array([unv.index(j) for j in cands])
            score = est + 0.0 * rank
            ordr = np.argsort(score)
            best = None
            for k in ordr[: n_exact]:
                if not np.isfinite(est[k]) and best is not None:
                    break
                j = int(cands[k])
                cap = best[0][0] if best is not None else np.inf
                r = self.ea(cur, j, t, exc=bool(use_exc[k]), t_cap=cap)
                if r is not None and (best is None or r[0] < best[0][0]):
                    best = (r, j, bool(use_exc[k]))
            if best is not None and rand > 0 and self.rng.random() < rand:
                # GRASP: sometimes take the 2nd best candidate (if not much worse)
                alt = None
                for k in ordr[: n_exact]:
                    j = int(cands[k])
                    if j == best[1]:
                        continue
                    r = self.ea(cur, j, t, exc=bool(use_exc[k]))
                    if r is not None and r[0] - t < 1.5 * (best[0][0] - t) + 0.05:
                        alt = (r, j, bool(use_exc[k]))
                        break
                if alt is not None:
                    best = alt
            if best is None:
                # fallback: try every candidate exactly, then with exception
                for k in range(len(cands)):
                    j = int(cands[k])
                    r = self.ea(cur, j, t, exc=bool(use_exc[k]))
                    if r is None and not use_exc[k] and exc_left > need_rest:
                        r = self.ea(cur, j, t, exc=True)
                        ex = True
                    else:
                        ex = bool(use_exc[k])
                    if r is not None and (best is None or r[0] < best[0][0]):
                        best = (r, j, ex)
            if best is None:
                raise RuntimeError(f"stuck at node {cur}, t={t:.3f}, visited {len(order)} exc_left={exc_left} cands={cands.tolist()} unv={unv}")
            (arr, dep, tof, dv), j, ex = best
            order.append(j)
            deps.append(dep)
            tofs.append(tof)
            excs.append(ex)
            if ex:
                exc_left -= 1
            visited[j] = True
            cur, t = j, arr
            if len(order) % self.log_every == 0:
                self.log(f"  construct {len(order)}/{n}  t={t:.3f}  exc_left={exc_left}  {time.time()-t_start:.0f}s")
        return np.array(order), t

    # ------------------------------------------------------------- chain
    def chain(self, order, exc_flags=None, t0=None):
        """Earliest-arrival chain along a fixed order. exc_flags[k] for leg k."""
        P = self.P
        t = P.p.t0 if t0 is None else t0
        n = len(order)
        arr = np.empty(n)
        arr[0] = t
        legs = []
        for k in range(n - 1):
            ex = bool(exc_flags[k]) if exc_flags is not None else self.needs_exc(order[k], order[k + 1])
            r = self.ea(order[k], order[k + 1], t, exc=ex)
            if r is None:
                return None
            legs.append(r)
            t = r[0]
            arr[k + 1] = t
        return arr, legs

    def dp_exceptions(self, order):
        """Optimal placement of the spare exceptions for a fixed order (FIFO DP).
        Legs that need an exception (shell change / big plane change) always use
        one; the remaining 'spare' ones are placed where they save most time.
        State: (position, spare exceptions used)."""
        P = self.P
        n = len(order)
        need = np.array([self.needs_exc(order[k], order[k + 1]) for k in range(n - 1)])
        spare = P.p.n_exc - int(need.sum())
        if spare < 0:
            return None
        INF = np.inf
        T = np.full((n, spare + 1), INF)
        back = np.zeros((n, spare + 1), dtype=np.int8)
        T[0, 0] = P.p.t0
        for k in range(n - 1):
            i, j = order[k], order[k + 1]
            for e in range(spare + 1):
                if not np.isfinite(T[k, e]):
                    continue
                r = self.ea(i, j, T[k, e], exc=bool(need[k]))
                if r is not None and r[0] < T[k + 1, e]:
                    T[k + 1, e], back[k + 1, e] = r[0], 0
                if not need[k] and e < spare:
                    cap = T[k + 1, e + 1]
                    r = self.ea(i, j, T[k, e], exc=True, t_cap=cap)
                    if r is not None and r[0] < T[k + 1, e + 1]:
                        T[k + 1, e + 1], back[k + 1, e + 1] = r[0], 1
            for e in range(1, spare + 1):  # dominance pruning
                if T[k + 1, e] >= T[k + 1, :e].min() - 1e-12:
                    T[k + 1, e] = INF
        e = int(np.argmin(T[-1]))
        if not np.isfinite(T[-1, e]):
            return None
        flags = need.copy()
        for k in range(n - 1, 0, -1):
            if back[k, e]:
                flags[k - 1] = True
                e -= 1
        return flags, T[-1].min()

    # -------------------------------------------------------- local search
    def _legs_from(self, order, flags, k0, t, stop_pos=None, stop_time=None):
        """Recompute the chain from position k0 (arrival time t at order[k0]).
        Stops early when the arrival at some position equals the stored one
        (FIFO: the rest of the chain is then unchanged)."""
        out = []
        n = len(order)
        for m in range(k0, n - 1):
            r = self.ea(order[m], order[m + 1], t, exc=bool(flags[m]))
            if r is None:
                return None
            out.append(r)
            t = r[0]
            if stop_time is not None and m + 1 >= stop_pos and abs(t - stop_time[m + 1]) < 1e-9:
                break
        return out

    def local_search(self, order, flags, time_limit=60.0, max_w=5):
        """Or-opt moves (take one tomato, reinsert it up to max_w positions away).
        A move is accepted iff the arrival time at the first unchanged position
        after the window gets earlier (FIFO property => global improvement)."""
        P = self.P
        order = [int(o) for o in order]
        flags = [bool(f) for f in flags]
        n = len(order)
        arr, legs = self.chain(order, flags)
        arr = list(arr)
        legs = list(legs)
        t_end = time.time() + time_limit
        n_acc = 0
        improved = True
        sweep = 0
        while improved and time.time() < t_end:
            improved = False
            sweep += 1
            for k in range(1, n - 1):
                if time.time() > t_end:
                    break
                best = None
                for p in range(k, min(n, k + max_w)):
                    for q in range(k, min(n, k + max_w)):
                        if p == q:
                            continue
                        new = order[:]
                        node = new.pop(p)
                        new.insert(q, node)
                        end = max(p, q) + 1
                        nf = flags[:]
                        for m in range(k - 1, min(end, n - 1)):
                            nf[m] = self.needs_exc(new[m], new[m + 1])
                        if sum(nf) > P.p.n_exc:
                            continue
                        t = arr[k - 1]
                        bound = arr[end] if end < n else arr[-1]
                        ok = True
                        last = min(end, n - 1)
                        for m in range(k - 1, last):
                            cap = bound if (m == last - 1 and end < n) else np.inf
                            r = self.ea(new[m], new[m + 1], t, exc=nf[m], t_cap=cap)
                            if r is None:
                                ok = False
                                break
                            t = r[0]
                        if not ok:
                            continue
                        if t < bound - 1e-6 and (best is None or t < best[0]):
                            best = (t, new, nf)
                if best is None:
                    continue
                _, new, nf = best
                tail = self._legs_from(new, nf, k - 1, arr[k - 1], stop_pos=max(k + max_w, k), stop_time=arr)
                if tail is None:
                    continue
                order, flags = new, nf
                for idx, r in enumerate(tail):
                    m = k - 1 + idx
                    legs[m] = r
                    arr[m + 1] = r[0]
                n_acc += 1
                improved = True
            self.log(f"    LS sweep {sweep}: f={arr[-1]:.4f} accepted={n_acc}")
        return np.array(order), np.array(flags)

    # -------------------------------------------------------------- solve
    def to_x(self, order, legs):
        dep = [r[1] for r in legs]
        tof = [r[2] for r in legs]
        return list(map(float, dep)) + list(map(float, tof)) + [int(o) for o in order]

    def evaluate_order(self, order):
        dp = self.dp_exceptions(order)
        if dp is None:
            return None
        return dp

    def solve(self, time_limit=600.0, ls=True, ls_share=0.35, on_best=None):
        """Multi-start: (guide variant, window, randomisation) -> greedy -> DP,
        then local search on the best order with the remaining time."""
        from .static_order import static_cost, edge_weight, nearest_neighbour, two_opt, or_opt
        P = self.P
        t0 = time.time()
        deadline = t0 + time_limit
        best = None  # (f, order, flags)
        it = 0
        est = self.est
        W0 = edge_weight(est, P.p.dv_max, P.p.dv_exc)
        while True:
            if it >= 2 and time.time() > t0 + time_limit * (1 - ls_share):
                break
            if it < 2:
                g = self.guide[:: (1 if it == 0 else -1)]
                rand, win = 0.0, self.window
            else:
                noise = 1 + 0.3 * self.rng.random(W0.shape)
                Wn = W0 * (noise + noise.T) / 2
                g = nearest_neighbour(Wn, int(self.rng.integers(P.n)))
                g = two_opt(g, Wn)
                if P.n <= 300:
                    g = or_opt(g, Wn)
                if self.rng.random() < 0.5:
                    g = g[::-1]
                rand = float(self.rng.choice([0.0, 0.03, 0.08, 0.15]))
                win = int(self.rng.choice([24, 48, 96, 160, 250]))
            it += 1
            self.window = win
            try:
                order, f_g = self.construct(g, rand=rand, n_exact=4)
            except RuntimeError as e:
                self.log(f"[start {it}] failed: {str(e)[:80]}")
                continue
            dp = self.dp_exceptions(order)
            if dp is None:
                continue
            flags, f = dp
            self.log(f"[start {it}] window={win} greedy={f_g:.4f} dp={f:.4f}  best={best[0] if best else None}  ({time.time()-t0:.0f}s)")
            if best is None or f < best[0]:
                best = (f, order, flags)
                if on_best is not None:
                    arr, legs = self.chain(order, flags)
                    on_best(self.to_x(order, legs))
        f, order, flags = best
        if ls and deadline - time.time() > 5:
            order, flags = self.local_search(order, flags, time_limit=(deadline - time.time()) * 0.9)
            dp = self.dp_exceptions(order)
            if dp is not None and dp[1] <= f + 1e-9:
                flags, f = dp
            self.log(f"[after LS + dp] f = {f:.4f}")
        arr, legs = self.chain(order, flags)
        x = self.to_x(order, legs)
        return x, arr[-1]
