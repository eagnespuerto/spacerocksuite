"""Heavy numerical kernels.

Uses the Rust extension `srs_core` when it is installed, else pure NumPy
versions below. Both give the same answers (tested in tests/test_core.py),
so results do not depend on which one ran.
"""
from __future__ import annotations

import math
import os

import numpy as np

GM = 2.959122082855911e-4            # AU^3 / day^2
C_AUD = 173.1446326742403            # speed of light, AU/day
EPS = math.radians(84381.448 / 3600)

try:
    if os.environ.get("SRS_NO_RUST"):
        raise ImportError
    import srs_core as _rs  # type: ignore
    HAVE_RUST = True
except ImportError:  # pragma: no cover - depends on install
    _rs = None
    HAVE_RUST = False


def backend() -> str:
    return f"rust {_rs.version()}" if HAVE_RUST else "numpy"


# ---------------------------------------------------------------- orbits
def _stumpff(z):
    z = np.asarray(z, float)
    c2 = np.empty_like(z)
    c3 = np.empty_like(z)
    p = z > 1e-8
    n = z < -1e-8
    s = ~(p | n)
    sp = np.sqrt(z[p])
    c2[p] = (1 - np.cos(sp)) / z[p]
    c3[p] = (sp - np.sin(sp)) / sp ** 3
    sn = np.sqrt(-z[n])
    c2[n] = (np.cosh(sn) - 1) / -z[n]
    c3[n] = (np.sinh(sn) - sn) / sn ** 3
    c2[s] = 0.5 - z[s] / 24
    c3[s] = 1 / 6 - z[s] / 120
    return c2, c3


def _helio_xyz(el: np.ndarray, t) -> np.ndarray:
    """Heliocentric equatorial J2000 positions (N, M, 3) from q,e,i,node,peri,tp.

    t is (M,) or (N, M) epochs (JD). Universal-variable two-body solution,
    valid for elliptic, parabolic and hyperbolic orbits.
    """
    q, e, i, node, peri, tp = (el[:, k][:, None] for k in range(6))
    i, node, peri = np.radians(i), np.radians(node), np.radians(peri)
    dt = np.broadcast_to(np.asarray(t, float), (el.shape[0], np.shape(t)[-1])) - tp
    alpha = (1 - e) / q                                     # 1/a (negative if hyperbolic)
    sqmu = math.sqrt(GM)
    v0 = np.sqrt(GM * (1 + e) / q)                          # perihelion speed
    with np.errstate(all="ignore"):                         # Vallado's starting guesses
        a_ = np.where(alpha != 0, 1 / np.where(alpha != 0, alpha, 1), np.inf)
        sg = np.where(dt >= 0, 1.0, -1.0)
        hyp = sg * np.sqrt(-a_) * np.log(np.abs(-2 * GM * alpha * dt
                                                 / (sg * np.sqrt(np.abs(GM * a_)) * (1 - q * alpha))) + 1e-300)
        chi = np.where(alpha > 1e-9, sqmu * alpha * dt,
                       np.where(alpha < -1e-9, hyp, np.cbrt(6 * sqmu * dt) * 0 + sqmu * dt / q))
        chi = np.where(np.isfinite(chi), chi, sqmu * dt / q)
    for _ in range(100):
        z = alpha * chi ** 2
        c2, c3 = _stumpff(z)
        f = q * chi * (1 - z * c3) + chi ** 3 * c3 - sqmu * dt
        r = chi ** 2 * c2 + q * (1 - z * c2)                # df/dchi
        step = f / r
        step = np.clip(step, -np.abs(chi) - 1, np.abs(chi) + 1)
        chi = chi - step
        if np.nanmax(np.abs(step)) < 1e-13 * max(1.0, float(np.nanmax(np.abs(chi)))):
            break
    z = alpha * chi ** 2
    c2, c3 = _stumpff(z)
    fx = 1 - chi ** 2 / q * c2
    gx = dt - chi ** 3 / sqmu * c3
    xp, yp = fx * q, gx * v0                                # perifocal position
    cO, sO, ci, si, cw, sw = np.cos(node), np.sin(node), np.cos(i), np.sin(i), np.cos(peri), np.sin(peri)
    xe = (cO * cw - sO * sw * ci) * xp + (-cO * sw - sO * cw * ci) * yp
    ye = (sO * cw + cO * sw * ci) * xp + (-sO * sw + cO * cw * ci) * yp
    ze = (sw * si) * xp + (cw * si) * yp
    ce, se = math.cos(EPS), math.sin(EPS)
    return np.stack([xe, ye * ce - ze * se, ye * se + ze * ce], axis=-1)


def kepler_radec(elements, jd, earth_xyz):
    """Two-body RA/Dec/Delta of N orbits at M epochs seen from observer positions.

    elements: (N, 6) q[au], e, i, node, peri [deg, ecliptic J2000], tp [JD TDB]
    jd: (M,) epochs; earth_xyz: (M, 3) heliocentric equatorial observer [au].
    Returns ra_deg, dec_deg, delta_au, each (N, M). Light time applied.
    """
    el = np.ascontiguousarray(elements, float).reshape(-1, 6)
    jd = np.ascontiguousarray(jd, float).ravel()
    E = np.ascontiguousarray(earth_xyz, float).reshape(-1, 3)
    if HAVE_RUST:
        return tuple(np.asarray(a) for a in _rs.kepler_radec(el, jd, E))
    tau = np.zeros((el.shape[0], jd.size))
    for _ in range(3):
        d = _helio_xyz(el, jd[None, :] - tau) - E[None, :, :]
        delta = np.linalg.norm(d, axis=-1)
        tau = delta / C_AUD
    ra = np.degrees(np.arctan2(d[..., 1], d[..., 0])) % 360
    dec = np.degrees(np.arcsin(np.clip(d[..., 2] / delta, -1, 1)))
    return ra, dec, delta


# ---------------------------------------------------------------- images
def _reflect_pad(a, r):
    return np.pad(a, r, mode="reflect") if r > 0 else a


def gaussian_blur(img, sigma: float) -> np.ndarray:
    """Separable, NaN-aware Gaussian blur (normalised convolution)."""
    img = np.ascontiguousarray(img, np.float32)
    if sigma <= 0:
        return img.copy()
    if HAVE_RUST:
        return np.asarray(_rs.gaussian_blur(img, float(sigma)))
    r = int(math.ceil(4 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    good = np.isfinite(img)
    v = np.where(good, img, 0).astype(np.float64)
    w = good.astype(np.float64)

    def conv(a, axis):
        a = np.moveaxis(_reflect_pad(np.moveaxis(a, axis, 0), ((r, r),) + ((0, 0),) * (a.ndim - 1)), 0, axis)
        out = np.zeros_like(np.take(a, np.arange(r, a.shape[axis] - r), axis=axis))
        n = out.shape[axis]
        for j, kj in enumerate(k):
            out += kj * np.take(a, np.arange(j, j + n), axis=axis)
        return out

    num = conv(conv(v, 0), 1)
    den = conv(conv(w, 0), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den < 1e-6] = np.nan
    return out.astype(np.float32)


def median_filter(img, size: int) -> np.ndarray:
    """Square NaN-ignoring median filter (background map).

    For size >= 15 the median is computed on a grid every size//4 pixels and
    bilinearly upsampled; the background varies slowly, so this matches the
    exact filter to well under the noise and is ~20x faster (also on the Pi).
    """
    img = np.ascontiguousarray(img, np.float32)
    size = max(1, int(size) | 1)
    if HAVE_RUST:
        return np.asarray(_rs.median_filter(img, size))
    r = size // 2
    pad = _reflect_pad(img, r)
    H, W = img.shape
    if size < 15:
        from numpy.lib.stride_tricks import sliding_window_view
        win = sliding_window_view(pad, (size, size))
        return np.nanmedian(win.reshape(H, W, -1), axis=-1).astype(np.float32)
    step = max(1, size // 4)
    ys = np.arange(0, H, step)
    xs = np.arange(0, W, step)
    if ys[-1] != H - 1:
        ys = np.append(ys, H - 1)
    if xs[-1] != W - 1:
        xs = np.append(xs, W - 1)
    grid = np.empty((ys.size, xs.size), np.float32)
    for a, y in enumerate(ys):
        for b, x in enumerate(xs):
            grid[a, b] = np.nanmedian(pad[y:y + size, x:x + size])
    # bilinear upsample
    yi = np.interp(np.arange(H), ys, np.arange(ys.size))
    xi = np.interp(np.arange(W), xs, np.arange(xs.size))
    y0 = np.floor(yi).astype(int).clip(0, ys.size - 1)
    x0 = np.floor(xi).astype(int).clip(0, xs.size - 1)
    y1 = (y0 + 1).clip(max=ys.size - 1)
    x1 = (x0 + 1).clip(max=xs.size - 1)
    fy = (yi - y0)[:, None]
    fx = (xi - x0)[None, :]
    g = grid
    out = (g[y0][:, x0] * (1 - fy) * (1 - fx) + g[y1][:, x0] * fy * (1 - fx)
           + g[y0][:, x1] * (1 - fy) * fx + g[y1][:, x1] * fy * fx)
    return out.astype(np.float32)


def _shift_bilinear(f, dx, dy):
    H, W = f.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    sx, sy = xx - dx, yy - dy                     # sample source at (x - dx, y - dy)
    x0 = np.floor(sx).astype(int)
    y0 = np.floor(sy).astype(int)
    fx, fy = sx - x0, sy - y0
    ok = (x0 >= 0) & (y0 >= 0) & (x0 + 1 < W) & (y0 + 1 < H)
    x0c, y0c = x0.clip(0, W - 2), y0.clip(0, H - 2)
    v = (f[y0c, x0c] * (1 - fx) * (1 - fy) + f[y0c, x0c + 1] * fx * (1 - fy)
         + f[y0c + 1, x0c] * (1 - fx) * fy + f[y0c + 1, x0c + 1] * fx * fy)
    # exact integer shifts keep the edge column/row
    v = np.where(ok, v, np.nan)
    ex = (fx == 0) & (fy == 0) & (x0 >= 0) & (y0 >= 0) & (x0 < W) & (y0 < H)
    v[ex] = f[y0[ex], x0[ex]]
    return v.astype(np.float32)


def shift_stack(frames, dx, dy, method: str = "clipmean", clip_sigma: float = 3.0, iters: int = 3):
    """Shift each frame by (dx, dy) pixels and combine.

    Returns (stack, coverage). NaNs are ignored. method: mean | median | clipmean.
    """
    F = np.ascontiguousarray(frames, np.float32)
    dx = np.ascontiguousarray(dx, float).ravel()
    dy = np.ascontiguousarray(dy, float).ravel()
    if HAVE_RUST:
        s, c = _rs.shift_stack(F, dx, dy, method, float(clip_sigma), int(iters))
        return np.asarray(s), np.asarray(c)
    S = np.stack([_shift_bilinear(F[k], dx[k], dy[k]) for k in range(F.shape[0])])
    return combine(S, method, clip_sigma, iters)


def combine(S, method="clipmean", clip_sigma=3.0, iters=3):
    S = np.asarray(S, np.float32)
    cov = np.isfinite(S).sum(0).astype(np.uint16)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if method == "mean":
                out = np.nanmean(S, 0)
            elif method == "median":
                out = np.nanmedian(S, 0)
            else:
                A = S.copy()
                for _ in range(iters):
                    med = np.nanmedian(A, 0)
                    mad = 1.4826 * np.nanmedian(np.abs(A - med), 0)
                    bad = np.abs(A - med) > clip_sigma * np.where(mad > 0, mad, np.inf)
                    if not bad.any():
                        break
                    A[bad] = np.nan
                out = np.nanmean(A, 0)
                cov = np.isfinite(A).sum(0).astype(np.uint16)
    return out.astype(np.float32), cov


def points_in_ellipse(dx, dy, a, b, theta_deg):
    """True where (dx east, dy north) lies inside ellipse a x b, PA theta (N through E)."""
    dx = np.asarray(dx, float)
    dy = np.asarray(dy, float)
    t = math.radians(theta_deg)
    u = dx * math.sin(t) + dy * math.cos(t)      # along major axis
    v = dx * math.cos(t) - dy * math.sin(t)
    return (u / a) ** 2 + (v / b) ** 2 <= 1.0
