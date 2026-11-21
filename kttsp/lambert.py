"""
Vectorized Lambert solver (Izzo 2015), a NumPy port of the algorithm used in
pykep 2.x ``pk.lambert_problem``.  It returns, for every input problem, the
minimum total delta-v (departure + arrival) over

    * both directions (cw = False / True), and
    * every Lambert solution (0-rev + 2 per revolution, up to ``max_revs``),

exactly like ``TomatoProblem.compute_transfer`` of the official utilities.

Everything is vectorized: r1, v1, r2, v2 are (n, 3) arrays and tof is (n,).
"""
import numpy as np

PI = np.pi


def _x2tof2(x, N, lam, lam2):
    a = 1.0 / (1.0 - x * x)
    out = np.empty_like(x)
    ell = a > 0
    with np.errstate(all="ignore"):
        # ellipse
        alfa = 2.0 * np.arccos(np.clip(x, -1, 1))
        beta = 2.0 * np.arcsin(np.sqrt(np.clip(lam2 / a, 0, 1)))
        beta = np.where(lam < 0, -beta, beta)
        t_e = a * np.sqrt(np.abs(a)) * ((alfa - np.sin(alfa)) - (beta - np.sin(beta)) + 2.0 * PI * N) / 2.0
        # hyperbola
        alfa_h = 2.0 * np.arccosh(np.maximum(x, 1.0))
        beta_h = 2.0 * np.arcsinh(np.sqrt(np.abs(-lam2 / a)))
        beta_h = np.where(lam < 0, -beta_h, beta_h)
        t_h = -a * np.sqrt(np.abs(a)) * ((beta_h - np.sinh(beta_h)) - (alfa_h - np.sinh(alfa_h))) / 2.0
    out = np.where(ell, t_e, t_h)
    return out


def _hypergeo(z, tol=1e-11):
    Sj = np.ones_like(z)
    Cj = np.ones_like(z)
    for j in range(60):
        Cj = Cj * (3.0 + j) * (1.0 + j) / (2.5 + j) * z / (j + 1)
        Sj = Sj + Cj
        if np.all(np.abs(Cj) < tol):
            break
    return Sj


def _x2tof(x, N, lam, lam2):
    battin, lagrange = 0.01, 0.2
    dist = np.abs(x - 1.0)
    with np.errstate(all="ignore"):
        K = lam2
        E = x * x - 1.0
        rho = np.abs(E)
        z = np.sqrt(1.0 + K * E)
        # Battin series
        eta = z - lam * x
        S1 = 0.5 * (1.0 - lam - x * eta)
        Q = _hypergeo(np.where(dist < battin, S1, 0.0)) * 4.0 / 3.0
        t_b = (eta ** 3 * Q + 4.0 * lam * eta) / 2.0 + N * PI / rho ** 1.5
        # Lancaster
        y = np.sqrt(rho)
        g = x * z - lam * E
        l_ell = np.arccos(np.clip(g, -1, 1))
        d_ell = N * PI + l_ell
        f = y * (z - lam * x)
        d_hyp = np.log(np.abs(f + g))
        d = np.where(E < 0, d_ell, d_hyp)
        t_l = (x - lam * z - d / y) / E
        t_lag = _x2tof2(x, N, lam, lam2)
    out = np.where(dist < battin, t_b, t_l)
    out = np.where((dist < lagrange) & (dist > battin), t_lag, out)
    return out


def _dTdx(x, T, lam, lam2):
    l3 = lam2 * lam
    umx2 = 1.0 - x * x
    with np.errstate(all="ignore"):
        y = np.sqrt(1.0 - lam2 * umx2)
        y2 = y * y
        y3 = y2 * y
        DT = 1.0 / umx2 * (3.0 * T * x - 2.0 + 2.0 * l3 * x / y)
        DDT = 1.0 / umx2 * (3.0 * T + 5.0 * x * DT + 2.0 * (1.0 - lam2) * l3 / y3)
        DDDT = 1.0 / umx2 * (7.0 * x * DDT + 8.0 * DT - 6.0 * (1.0 - lam2) * lam2 * l3 * x / y3 / y2)
    return DT, DDT, DDDT


def _householder(T, x0, N, lam, lam2, eps, iter_max):
    x = x0.copy()
    active = np.ones_like(x, dtype=bool)
    for _ in range(iter_max):
        if not active.any():
            break
        tof = _x2tof(x, N, lam, lam2)
        DT, DDT, DDDT = _dTdx(x, tof, lam, lam2)
        delta = tof - T
        DT2 = DT * DT
        with np.errstate(all="ignore"):
            xnew = x - delta * (DT2 - delta * DDT / 2.0) / (DT * (DT2 - delta * DDT) + DDDT * delta * delta / 6.0)
        err = np.abs(x - xnew)
        upd = active & np.isfinite(xnew)
        x = np.where(upd, xnew, x)
        active = upd & (err > eps)
    return x


def _norm(v):
    return np.sqrt(np.einsum("ij,ij->i", v, v))


def lambert_min_dv(r1, v1, r2, v2, tof, mu, max_revs=20, _debug=None):
    """Minimum |dv1|+|dv2| over both directions and all revolutions.

    r1, v1: departure body state (n,3)  r2, v2: arrival body state (n,3)
    tof: time of flight in seconds (n,)
    Returns array (n,) (np.inf where no solution exists).
    """
    r1 = np.atleast_2d(np.asarray(r1, float))
    r2 = np.atleast_2d(np.asarray(r2, float))
    v1 = np.atleast_2d(np.asarray(v1, float))
    v2 = np.atleast_2d(np.asarray(v2, float))
    tof = np.atleast_1d(np.asarray(tof, float))
    n = tof.shape[0]

    c = r2 - r1
    cn = _norm(c)
    R1 = _norm(r1)
    R2 = _norm(r2)
    s = (cn + R1 + R2) / 2.0
    ir1 = r1 / R1[:, None]
    ir2 = r2 / R2[:, None]
    ih = np.cross(ir1, ir2)
    ihn = _norm(ih)
    with np.errstate(all="ignore"):
        ih = ih / ihn[:, None]
    lam2_ = 1.0 - cn / s
    lam_base = np.sqrt(np.maximum(lam2_, 0.0))
    neg = ih[:, 2] < 0
    lam0 = np.where(neg, -lam_base, lam_base)
    it1_0 = np.where(neg[:, None], np.cross(ir1, ih), np.cross(ih, ir1))
    it2_0 = np.where(neg[:, None], np.cross(ir2, ih), np.cross(ih, ir2))
    it1_0 /= _norm(it1_0)[:, None]
    it2_0 /= _norm(it2_0)[:, None]

    T = np.sqrt(2.0 * mu / s ** 3) * tof
    gamma = np.sqrt(mu * s / 2.0)
    with np.errstate(all="ignore"):
        rho = (R1 - R2) / cn
        sigma = np.sqrt(np.maximum(1.0 - rho * rho, 0.0))

    best = np.full(n, np.inf)
    for cw in (False, True):
        lam = -lam0 if cw else lam0
        sgn = -1.0 if cw else 1.0
        it1 = sgn * it1_0
        it2 = sgn * it2_0
        lam2 = lam * lam
        lam3 = lam * lam2

        Nmax = np.floor(T / PI).astype(int)
        T00 = np.arccos(np.clip(lam, -1, 1)) + lam * np.sqrt(np.maximum(1.0 - lam2, 0))
        T0 = T00 + Nmax * PI
        T1 = 2.0 / 3.0 * (1.0 - lam3)
        # check existence of the Nmax-rev solution
        chk = (Nmax > 0) & (T < T0)
        if chk.any():
            idx = np.where(chk)[0]
            x_old = np.zeros(idx.size)
            T_min = T0[idx].copy()
            Nm = Nmax[idx]
            la, la2 = lam[idx], lam2[idx]
            for _ in range(13):
                dT, ddT, dddT = _dTdx(x_old, T_min, la, la2)
                with np.errstate(all="ignore"):
                    x_new = np.where(dT != 0, x_old - dT * ddT / (ddT * ddT - dT * dddT / 2.0), x_old)
                x_new = np.where(np.isfinite(x_new), x_new, x_old)
                done = np.abs(x_old - x_new) < 1e-13
                T_min = _x2tof(x_new, Nm, la, la2)
                x_old = x_new
                if done.all():
                    break
            dec = T_min > T[idx]
            Nmax[idx[dec]] -= 1
        Nmax = np.minimum(Nmax, max_revs)

        def velocities_dv(x):
            with np.errstate(all="ignore"):
                y = np.sqrt(1.0 - lam2 + lam2 * x * x)
                vr1 = gamma * ((lam * y - x) - rho * (lam * y + x)) / R1
                vr2 = -gamma * ((lam * y - x) + rho * (lam * y + x)) / R2
                vt = gamma * sigma * (y + lam * x)
                vt1 = vt / R1
                vt2 = vt / R2
                V1 = vr1[:, None] * ir1 + vt1[:, None] * it1
                V2 = vr2[:, None] * ir2 + vt2[:, None] * it2
                if _debug is not None:
                    _debug.append((0, cw, V1.copy(), V2.copy()))
                dv = _norm(V1 - v1) + _norm(V2 - v2)
            return np.where(np.isfinite(dv), dv, np.inf)

        # 0 rev
        with np.errstate(all="ignore"):
            x0 = np.where(
                T >= T00,
                -(T - T00) / (T - T00 + 4.0),
                np.where(
                    T <= T1,
                    T1 * (T1 - T) / (2.0 / 5.0 * (1.0 - lam2 * lam3) * T) + 1.0,
                    (T / T00) ** (0.69314718055994529 / np.log(T1 / T00)) - 1.0,
                ),
            )
        x = _householder(T, x0, 0, lam, lam2, 1e-5, 15)
        best = np.minimum(best, velocities_dv(x))

        nm = int(Nmax.max()) if n else 0
        for i in range(1, nm + 1):
            m = Nmax >= i
            if not m.any():
                break
            idx = np.where(m)[0]
            Ti, la, la2 = T[idx], lam[idx], lam2[idx]
            for left in (True, False):
                if left:
                    tmp = ((i * PI + PI) / (8.0 * Ti)) ** (2.0 / 3.0)
                else:
                    tmp = ((8.0 * Ti) / (i * PI)) ** (2.0 / 3.0)
                xg = (tmp - 1.0) / (tmp + 1.0)
                xs = _householder(Ti, xg, i, la, la2, 1e-8, 15)
                # compute dv only on the subset
                with np.errstate(all="ignore"):
                    y = np.sqrt(1.0 - la2 + la2 * xs * xs)
                    g, rh, sg = gamma[idx], rho[idx], sigma[idx]
                    vr1 = g * ((la * y - xs) - rh * (la * y + xs)) / R1[idx]
                    vr2 = -g * ((la * y - xs) + rh * (la * y + xs)) / R2[idx]
                    vt = g * sg * (y + la * xs)
                    V1 = vr1[:, None] * ir1[idx] + (vt / R1[idx])[:, None] * it1[idx]
                    V2 = vr2[:, None] * ir2[idx] + (vt / R2[idx])[:, None] * it2[idx]
                    dv = _norm(V1 - v1[idx]) + _norm(V2 - v2[idx])
                if _debug is not None:
                    _debug.append((i, cw, idx, V1.copy(), V2.copy()))
                dv = np.where(np.isfinite(dv), dv, np.inf)
                best[idx] = np.minimum(best[idx], dv)
    return best
