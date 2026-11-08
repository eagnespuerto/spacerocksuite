"""Image cleanups used in the 2026 ED138 precovery work.

- background:    image minus a 41 px median-filtered background (plates/check.py)
- matched:       Gaussian matched filter at the seeing FWHM, with the noise of
                 the smoothed image measured, not assumed (sm_check.py, check.py)
- noise:         robust (MAD) sigma, 2x2 block-correlation factor for scanned
                 plates (snr_fwhm_check.py), and empirical noise from random
                 apertures (seeded, so repeatable)
- template:      science minus a median of zero-point-scaled templates (VVV method),
                 or a least-squares scale + offset fit when zero points are unknown
- matrix deblend: one linear least-squares solve for star fluxes, a background
                 plane and the target at once (deblend_1991.py); returns the image
                 with the fitted stars removed
- gaussian fit:  round or elliptical 2-D Gaussian centroids (round_gauss.py)

NumPy only; heavy filters use the Rust core when present.
"""
from __future__ import annotations

import math

import numpy as np

from .. import core


# ------------------------------------------------------------------ noise
def robust_sigma(a) -> float:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(a - np.median(a))))


def block_noise_factor(img, block: int = 2) -> float:
    """How much more noise block sums have than independent pixels would give.

    SuperCOSMOS scans are correlated on ~2x2 pixels, so naive per-pixel S/N is
    too high by about this factor (1.0 = uncorrelated).
    """
    a = np.asarray(img, float)
    h, w = (a.shape[0] // block) * block, (a.shape[1] // block) * block
    a = a[:h, :w]
    s1 = robust_sigma(a)
    blocks = a.reshape(h // block, block, w // block, block).sum((1, 3))
    return float(robust_sigma(blocks) / (s1 * block)) if s1 > 0 else 1.0


def aperture_sum(img, x, y, r):
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
    m = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
    v = img[m]
    return float(np.nansum(v)), int(np.isfinite(v).sum())


def empirical_noise(img, x, y, r, n=300, rmin=None, rmax=None, seed=0):
    """Robust scatter (1.4826 * MAD) of aperture sums at n random spots around (x, y).

    Includes correlated noise, unlike sqrt(N)*sigma; MAD keeps the few apertures
    that land on stars from inflating it. Seeded, so repeatable.
    """
    rng = np.random.default_rng(seed)
    H, W = img.shape
    rmin = rmin if rmin is not None else 3 * r
    rmax = rmax if rmax is not None else min(H, W) / 2 - r - 1
    sums = []
    tries = 0
    while len(sums) < n and tries < 20 * n:
        tries += 1
        ang = rng.uniform(0, 2 * math.pi)
        rad = rng.uniform(rmin, max(rmin + 1, rmax))
        cx, cy = x + rad * math.cos(ang), y + rad * math.sin(ang)
        if r <= cx < W - r and r <= cy < H - r:
            s, npx = aperture_sum(img, cx, cy, r)
            if npx > 0.8 * math.pi * r * r:
                sums.append(s)
    sums = np.array(sums)
    return (robust_sigma(sums) if len(sums) > 5 else float("nan")), sums


# ------------------------------------------------------------------ filters
def background(img, size: int = 41):
    """Image minus its median-filtered background."""
    img = np.asarray(img, np.float32)
    return img - core.median_filter(img, size)


def matched(img, fwhm_px: float):
    """Gaussian matched filter at the PSF FWHM (sigma = FWHM / 2.355)."""
    return core.gaussian_blur(np.asarray(img, np.float32), fwhm_px / 2.3548)


def clean(img, bkg_size: int = 41, fwhm_px: float | None = None):
    """Background removal, then optional matched filter. Returns (image, sigma)."""
    out = background(img, bkg_size)
    if fwhm_px:
        out = matched(out, fwhm_px)
    return out, robust_sigma(out)


# ------------------------------------------------------------------ templates
def template_subtract(sci, templates, zp_sci=None, zp_tpl=None, exptime_sci=1.0, exptime_tpl=None):
    """Science minus the median of templates scaled to the science zero point.

    With zero points (VVV MAGZPT/EXPTIME) the scaling is exact; without them each
    template is fitted as a*T + b by least squares on pixels away from the centre.
    Returns (difference, model).
    """
    sci = np.asarray(sci, np.float64)
    T = np.asarray(templates, np.float64)
    if T.ndim == 2:
        T = T[None]
    scaled = []
    H, W = sci.shape
    yy, xx = np.mgrid[:H, :W]
    away = np.hypot(xx - W / 2, yy - H / 2) > min(H, W) / 6
    for k, t in enumerate(T):
        if zp_sci is not None and zp_tpl is not None:
            et = exptime_tpl[k] if exptime_tpl is not None else exptime_sci
            f = 10 ** (-0.4 * (zp_tpl[k] - zp_sci)) * exptime_sci / et
            scaled.append(t * f)
        else:
            m = away & np.isfinite(t) & np.isfinite(sci)
            A = np.column_stack([t[m], np.ones(m.sum())])
            (a, b), *_ = np.linalg.lstsq(A, sci[m], rcond=None)
            scaled.append(a * t + b)
    model = np.nanmedian(np.stack(scaled), 0)
    return (sci - model).astype(np.float32), model.astype(np.float32)


# ------------------------------------------------------------------ PSFs
def gaussian_psf_fn(size: int, fwhm: float, trail_px: float = 0.0, pa_deg: float = 0.0):
    """Callable PSF for matrix_deblend: exact sub-pixel placement."""
    return lambda dx, dy: gaussian_psf(size, fwhm, trail_px, pa_deg, dx, dy)


def gaussian_psf(size: int, fwhm: float, trail_px: float = 0.0, pa_deg: float = 0.0, dx=0.0, dy=0.0):
    """Unit-sum Gaussian PSF, optionally trailed (motion during the exposure).

    pa_deg is the trail direction measured from +y through -x (north through
    east on a north-up, east-left image).
    """
    s = fwhm / 2.3548
    c = (size - 1) / 2
    yy, xx = np.mgrid[:size, :size].astype(float)
    n = max(1, int(math.ceil(trail_px * 2)) + 1)
    out = np.zeros((size, size))
    t = math.radians(pa_deg)
    ux, uy = -math.sin(t), math.cos(t)
    for f in np.linspace(-0.5, 0.5, n) if trail_px > 0 else [0.0]:
        cx, cy = c + dx + f * trail_px * ux, c + dy + f * trail_px * uy
        out += np.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / s ** 2)
    return out / out.sum()


def empirical_psf(img, stars_xy, size: int = 15):
    """Median of unit-normalised star stamps (bright, isolated stars work best)."""
    h = size // 2
    stamps = []
    for x, y in stars_xy:
        xi, yi = int(round(x)), int(round(y))
        if h <= xi < img.shape[1] - h and h <= yi < img.shape[0] - h:
            s = img[yi - h:yi + h + 1, xi - h:xi + h + 1].astype(float)
            s = s - np.nanmedian(s)
            tot = np.nansum(s)
            if tot > 0:
                stamps.append(s / tot)
    if not stamps:
        raise ValueError("no usable stars for the PSF")
    p = np.nanmedian(np.stack(stamps), 0)
    p[p < 0] = 0
    return p / p.sum()


def _place(psf, shape, x, y):
    """PSF placed at sub-pixel (x, y) on a canvas of `shape`.

    psf is an array (placed with bilinear weights) or a callable f(dx, dy) that
    returns the PSF already shifted by the sub-pixel offset (exact, no smoothing).
    """
    if callable(psf):
        xi, yi = int(round(x)), int(round(y))
        arr = psf(x - xi, y - yi)
        canvas = np.zeros(shape)
        h = arr.shape[0] // 2
        y0, x0 = yi - h, xi - h
        ys, xs = slice(max(0, y0), min(shape[0], y0 + arr.shape[0])), slice(max(0, x0), min(shape[1], x0 + arr.shape[1]))
        if ys.stop > ys.start and xs.stop > xs.start:
            canvas[ys, xs] = arr[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
        return canvas
    canvas = np.zeros(shape)
    h = psf.shape[0] // 2
    xi, yi = int(math.floor(x)), int(math.floor(y))
    fx, fy = x - xi, y - yi
    for oy, ox, wgt in ((0, 0, (1 - fx) * (1 - fy)), (0, 1, fx * (1 - fy)), (1, 0, (1 - fx) * fy), (1, 1, fx * fy)):
        y0, x0 = yi + oy - h, xi + ox - h
        ys, xs = slice(max(0, y0), min(shape[0], y0 + psf.shape[0])), slice(max(0, x0), min(shape[1], x0 + psf.shape[1]))
        py, px = slice(ys.start - y0, ys.stop - y0), slice(xs.start - x0, xs.stop - x0)
        if ys.stop > ys.start and xs.stop > xs.start:
            canvas[ys, xs] += wgt * psf[py, px]
    return canvas


def matrix_deblend(img, stars_xy, psf, target_xy=None, target_psf=None, plane=True, weights=None):
    """Simultaneous linear fit: sum_k f_k PSF(star_k) [+ f_t PSF_t(target)] + plane.

    psf / target_psf: array (e.g. empirical_psf) or callable (gaussian_psf_fn).
    Use a trailed target PSF for long exposures (motion during the plate).

    One least-squares solve with a design matrix (one column per star, the
    target and the background plane terms). Returns dict with fluxes, their
    1-sigma errors, the residual image (stars and plane removed, target kept),
    and the target flux/S/N if a target was given.
    """
    img = np.asarray(img, float)
    H, W = img.shape
    cols, names = [], []
    for k, (x, y) in enumerate(stars_xy):
        cols.append(_place(psf, img.shape, x, y).ravel())
        names.append(f"star{k}")
    if target_xy is not None:
        cols.append(_place(target_psf if target_psf is not None else psf, img.shape, *target_xy).ravel())
        names.append("target")
    yy, xx = np.mgrid[:H, :W]
    cols.append(np.ones(H * W))
    names.append("bkg")
    if plane:
        cols += [(xx.ravel() - W / 2) / W, (yy.ravel() - H / 2) / H]
        names += ["bx", "by"]
    A = np.column_stack(cols)
    b = img.ravel()
    good = np.isfinite(b)
    w = np.ones_like(b) if weights is None else np.asarray(weights, float).ravel()
    Aw, bw = A[good] * w[good, None], b[good] * w[good]
    coef, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
    resid_all = b - A @ coef
    dof = max(1, good.sum() - len(coef))
    s2 = float(np.nansum((resid_all[good] * w[good]) ** 2) / dof)
    try:
        cov = np.linalg.inv(Aw.T @ Aw) * s2
        err = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        err = np.full(len(coef), np.nan)
    remove = np.array([n != "target" for n in names])
    cleaned = (b - A[:, remove] @ coef[remove]).reshape(H, W)
    out = {"names": names, "flux": coef, "err": err, "cleaned": cleaned.astype(np.float32),
           "residual": resid_all.reshape(H, W).astype(np.float32), "rms": math.sqrt(s2)}
    if target_xy is not None:
        i = names.index("target")
        out["target_flux"], out["target_err"] = float(coef[i]), float(err[i])
        out["target_snr"] = float(coef[i] / err[i]) if err[i] > 0 else float("nan")
    return out


# ------------------------------------------------------------------ Gaussian fits
def _lm(fun, p0, iters=60):
    """Small Levenberg-Marquardt (numeric Jacobian). Returns params, covariance, residual."""
    p = np.asarray(p0, float)
    lam = 1e-3
    r = fun(p)
    cost = r @ r
    for _ in range(iters):
        J = np.empty((r.size, p.size))
        for j in range(p.size):
            h = 1e-6 * max(1.0, abs(p[j]))
            q = p.copy()
            q[j] += h
            J[:, j] = (fun(q) - r) / h
        g = J.T @ r
        Hm = J.T @ J
        step = np.linalg.solve(Hm + lam * np.diag(np.diag(Hm) + 1e-12), -g)
        q = p + step
        rq = fun(q)
        cq = rq @ rq
        if cq < cost:
            p, r, cost, lam = q, rq, cq, lam / 3
            if np.max(np.abs(step)) < 1e-8:
                break
        else:
            lam *= 4
    dof = max(1, r.size - p.size)
    try:
        cov = np.linalg.inv(J.T @ J) * cost / dof
    except np.linalg.LinAlgError:
        cov = np.full((p.size, p.size), np.nan)
    return p, cov, r


def fit_gaussian(img, x0, y0, box: int = 6, fwhm0: float = 3.0, elliptical: bool = False):
    """2-D Gaussian + constant around (x0, y0). Round by default (Micheli's advice
    for ED138: round fits treat stars and target the same way).

    Returns dict: x, y, sigma (or sx, sy, theta), amp, bkg, x_err, y_err, flux, snr.
    """
    xi, yi = int(round(x0)), int(round(y0))
    ys, xs = np.mgrid[yi - box:yi + box + 1, xi - box:xi + box + 1]
    if ys.min() < 0 or xs.min() < 0 or ys.max() >= img.shape[0] or xs.max() >= img.shape[1]:
        return None
    z = np.asarray(img, float)[ys, xs]
    m = np.isfinite(z)
    s0 = fwhm0 / 2.3548
    bkg0 = float(np.nanmedian(z))
    amp0 = float(np.nanmax(z) - bkg0)
    if elliptical:
        def model(p):
            a, x, y, sx, sy, t, b = p
            ct, st = math.cos(t), math.sin(t)
            u = (xs - x) * ct + (ys - y) * st
            v = -(xs - x) * st + (ys - y) * ct
            return a * np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2)) + b
        p0 = [amp0, x0, y0, s0, s0, 0.0, bkg0]
    else:
        def model(p):
            a, x, y, s, b = p
            return a * np.exp(-0.5 * ((xs - x) ** 2 + (ys - y) ** 2) / s ** 2) + b
        p0 = [amp0, x0, y0, s0, bkg0]
    p, cov, r = _lm(lambda q: (model(q) - z)[m], p0)
    e = np.sqrt(np.clip(np.diag(cov), 0, None))
    out = {"x": p[1], "y": p[2], "amp": p[0], "bkg": p[-1], "x_err": e[1], "y_err": e[2]}
    if elliptical:
        out.update(sx=abs(p[3]), sy=abs(p[4]), theta=math.degrees(p[5]) % 180)
        out["flux"] = 2 * math.pi * p[0] * abs(p[3] * p[4])
    else:
        out.update(sigma=abs(p[3]), fwhm=abs(p[3]) * 2.3548)
        out["flux"] = 2 * math.pi * p[0] * p[3] ** 2
    out["snr"] = float(p[0] / e[0]) if e[0] > 0 else float("nan")
    return out
