"""
Earliest-arrival search for a single transfer i -> j.

Given that we are at tomato i at time t, find (t_dep >= t, tof >= min_tof)
minimising the arrival time t_dep + tof subject to dv(i,j,t_dep,tof) <= limit.

Method: coarse 2-D grid over (t_dep, tof) in growing time windows,
followed by local grid refinement around the most promising cells.
Grid steps scale with the orbital periods, so the same code works for the
low (0.24 d) and high (1.9 d) shells and for all three difficulty levels.
"""
import numpy as np


class TransferSearch:
    def __init__(self, prob, safety=0.5, coarse_div=48, max_window_periods=6.0):
        self.P = prob
        self.safety = safety  # m/s kept below the official limit
        self.coarse_div = coarse_div
        self.max_window_periods = max_window_periods
        self.n_evals = 0

    def _dv(self, i, j, td, tf):
        self.n_evals += np.size(td)
        return self.P.dv(i, j, td, tf)

    def earliest(self, i, j, t, limit, t_cap=np.inf, n_refine=4):
        """Return (arrival, t_dep, tof, dv) or None."""
        P = self.P
        lim = limit - self.safety
        per = min(P.period_days[i], P.period_days[j])
        perM = max(P.period_days[i], P.period_days[j])
        step = per / self.coarse_div
        min_tof = P.p.min_tof
        # windows of arrival time (relative to t)
        H = perM * 0.5
        lo_arr = 0.0
        best = None
        H_max = max(perM * self.max_window_periods, 3.0)
        while H <= H_max + 1e-9:
            # grid of arrival offsets in (lo_arr, H] and departure offsets
            arr = np.arange(lo_arr + step, H + 1e-12, step)
            dep = np.arange(0.0, H, step)
            A, D = np.meshgrid(arr, dep, indexing="ij")
            F = A - D
            m = F >= min_tof
            A, D, F = A[m], D[m], F[m]
            if t + A.min() > t_cap:
                break
            dv = self._dv(i, j, t + D, F)
            # 1) coarse-feasible cells (earliest first), 2) near-feasible earlier cells
            feas = np.where(dv <= lim)[0]
            near = np.where((dv > lim) & (dv <= lim * 1.12))[0]
            a_feas = A[feas].min() if feas.size else np.inf
            near = near[A[near] < a_feas]
            groups = []
            if feas.size:
                groups.append(feas[np.argsort(A[feas])])
            if near.size:
                groups.append(near[np.lexsort((dv[near], A[near]))])
            for order in groups:
                tried = 0
                seen = []
                for k in order:
                    if best is not None and A[k] - 2 * step > best[0] - t:
                        break
                    if any(abs(A[k] - a0) < 2 * step and abs(D[k] - d0) < 2 * step for a0, d0 in seen):
                        continue
                    seen.append((A[k], D[k]))
                    r = self._refine(i, j, t, D[k], F[k], step, lim, min_tof)
                    if r is not None and (best is None or r[0] < best[0]):
                        best = r
                    tried += 1
                    if tried >= n_refine:
                        break
            if best is not None:
                return best
            lo_arr = H
            H *= 2.0
        return best

    def _refine(self, i, j, t, d0, f0, step, lim, min_tof, levels=3):
        """Local refinement: minimise arrival t_dep+tof with dv <= lim near (d0,f0)."""
        best = None
        d_c, f_c, h = d0, f0, step
        for _ in range(levels):
            g = np.linspace(-1.5, 1.5, 9) * h
            DD, FF = np.meshgrid(d_c + g, f_c + g, indexing="ij")
            DD, FF = DD.ravel(), FF.ravel()
            m = (DD >= 0) & (FF >= min_tof)
            DD, FF = DD[m], FF[m]
            if DD.size == 0:
                break
            dv = self._dv(i, j, t + DD, FF)
            ok = dv <= lim
            if ok.any():
                arr = np.where(ok, DD + FF, np.inf)
                k = int(np.argmin(arr))
                cand = (t + DD[k] + FF[k], t + DD[k], FF[k], dv[k])
                if best is None or cand[0] < best[0]:
                    best = cand
                d_c, f_c = DD[k], FF[k]
            else:
                k = int(np.argmin(dv))
                d_c, f_c = DD[k], FF[k]
            h /= 3.0
        if best is not None:
            # final push: try to shave arrival by shrinking tof / dep along a line search
            best = self._shave(i, j, best, lim, min_tof, t)
        return best

    def _shave(self, i, j, best, lim, min_tof, t):
        arr, td, tf, dv = best
        for _ in range(2):
            deltas = np.linspace(0, 1, 21)[1:] * 0.02 * tf
            # reduce tof only
            F = tf - deltas
            D = np.full_like(F, td)
            # reduce departure only
            D2 = np.maximum(td - deltas, t)
            F2 = np.full_like(D2, tf)
            DD = np.r_[D, D2]
            FF = np.r_[F, F2]
            m = FF >= min_tof
            DD, FF = DD[m], FF[m]
            if DD.size == 0:
                break
            v = self._dv(i, j, DD, FF)
            ok = v <= lim
            if not ok.any():
                break
            a = np.where(ok, DD + FF, np.inf)
            k = int(np.argmin(a))
            if a[k] < arr:
                arr, td, tf, dv = a[k], DD[k], FF[k], v[k]
            else:
                break
        return (arr, td, tf, dv)
