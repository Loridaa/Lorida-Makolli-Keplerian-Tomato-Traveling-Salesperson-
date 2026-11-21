"""
Problem loading + Keplerian ephemerides (compatible with pykep 2.x
``pk.planet.keplerian(epoch(0), [a,e,i,W,w,M], MU_MOON, ...)``).

Note: pykep 2 interprets the 6th orbital element as the MEAN anomaly at t0.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .lambert import lambert_min_dv

MU_MOON = 4.904869500000000e12
DAY2SEC = 86400.0


@dataclass
class Params:
    t0: float
    min_tof: float
    t_max: float
    dv_max: float
    dv_exc: float
    n_exc: int


class Problem:
    def __init__(self, path, max_revs=20):
        path = Path(path)
        self.name = path.stem
        rows, header = [], None
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("c"):
                continue
            if line.startswith("p "):
                header = line.split()
                continue
            rows.append([float(v) for v in line.split()])
        h = [float(v) for v in header[2:]]
        self.p = Params(h[0], h[1], h[2], h[3], h[4], int(h[5]))
        self.el = np.asarray(rows, float)
        self.n = len(rows)
        self.max_revs = max_revs
        a, e, inc, W, w, M = self.el.T
        self.a, self.e, self.M0 = a, e, M
        self.nmot = np.sqrt(MU_MOON / a ** 3)  # rad/s
        self.period_days = 2 * np.pi / self.nmot / DAY2SEC
        # rotation matrix perifocal -> inertial for every body (n,3,3)
        cW, sW, cw, sw, ci, si = np.cos(W), np.sin(W), np.cos(w), np.sin(w), np.cos(inc), np.sin(inc)
        R = np.empty((self.n, 3, 3))
        R[:, 0, 0] = cW * cw - sW * sw * ci
        R[:, 0, 1] = -cW * sw - sW * cw * ci
        R[:, 0, 2] = sW * si
        R[:, 1, 0] = sW * cw + cW * sw * ci
        R[:, 1, 1] = -sW * sw + cW * cw * ci
        R[:, 1, 2] = -cW * si
        R[:, 2, 0] = sw * si
        R[:, 2, 1] = cw * si
        R[:, 2, 2] = ci
        self.R = R
        # orbit normal (for plane-change estimates)
        self.h_hat = R[:, :, 2]
        self.inc = inc

    # ------------------------------------------------------------------
    def eph(self, idx, t):
        """Positions/velocities of bodies ``idx`` at times ``t`` (days).
        idx and t are broadcast to the same shape (flattened)."""
        idx, t = np.broadcast_arrays(np.asarray(idx), np.asarray(t, float))
        idx = idx.ravel().astype(int)
        t = t.ravel()
        a, e = self.a[idx], self.e[idx]
        M = self.M0[idx] + self.nmot[idx] * (t - self.p.t0) * DAY2SEC
        M = np.mod(M, 2 * np.pi)
        E = M + e * np.sin(M)
        for _ in range(8):
            E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        cE, sE = np.cos(E), np.sin(E)
        b = a * np.sqrt(1 - e * e)
        xp = a * (cE - e)
        yp = b * sE
        rr = a * (1 - e * cE)
        k = np.sqrt(MU_MOON * a) / rr
        vxp = -k * sE
        vyp = k * np.sqrt(1 - e * e) * cE
        Rm = self.R[idx]
        r = Rm[:, :, 0] * xp[:, None] + Rm[:, :, 1] * yp[:, None]
        v = Rm[:, :, 0] * vxp[:, None] + Rm[:, :, 1] * vyp[:, None]
        return r, v

    def dv(self, i, j, t_dep, tof):
        """Vectorized delta-v of transfers i->j departing t_dep (days) lasting tof (days)."""
        i, j, t_dep, tof = np.broadcast_arrays(np.asarray(i), np.asarray(j), np.asarray(t_dep, float), np.asarray(tof, float))
        shape = t_dep.shape
        r1, v1 = self.eph(i, t_dep)
        r2, v2 = self.eph(j, t_dep + tof)
        out = lambert_min_dv(r1, v1, r2, v2, tof.ravel() * DAY2SEC, MU_MOON, self.max_revs)
        return out.reshape(shape)

    # ------------------------------------------------------------------
    def evaluate(self, x, margin=0.0, verbose=False):
        """Independent check of a chromosome. Returns (feasible, objective, info)."""
        n = self.n
        x = np.asarray(x, float)
        if x.size != 3 * n - 2:
            return False, np.inf, "wrong length"
        t = x[: n - 1]
        tof = x[n - 1: 2 * n - 2]
        pi = x[2 * n - 2:].astype(int)
        if sorted(pi.tolist()) != list(range(n)):
            return False, np.inf, "not a permutation"
        if np.any(tof < self.p.min_tof) or t[0] < self.p.t0:
            return False, np.inf, "tof<min or t1<t0"
        arr = t + tof
        if np.any(arr[:-1] > t[1:] + 1e-12):
            return False, np.inf, "chronology"
        f = arr[-1]
        if f > self.p.t_max:
            return False, f, "t_max exceeded"
        dvs = self.dv(pi[:-1], pi[1:], t, tof)
        n_exc = int(np.sum(dvs > self.p.dv_max - margin))
        bad = int(np.sum(dvs > self.p.dv_exc - margin))
        ok = n_exc <= self.p.n_exc and bad == 0
        info = dict(dvs=dvs, n_exc=n_exc, max_dv=float(dvs.max()))
        return ok, f, info
