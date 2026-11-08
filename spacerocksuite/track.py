"""An object's predicted track with its positional uncertainty.

Everything downstream (archive search, stacking, stamps) works from a Track,
whatever service produced it.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

from . import core


@dataclass
class Track:
    name: str
    source: str                       # "horizons" | "scout" | "findorb" | "ephemeral" | ...
    jd: np.ndarray                    # UTC JD, sorted
    ra: np.ndarray                    # deg, J2000/ICRF
    dec: np.ndarray                   # deg
    vmag: np.ndarray | None = None
    delta: np.ndarray | None = None   # au
    # 1-sigma error ellipse in arcsec, PA of major axis (deg, N through E)
    smaa: np.ndarray | None = None
    smia: np.ndarray | None = None
    theta: np.ndarray | None = None
    site: str = "500"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        o = np.argsort(self.jd)
        for k in ("jd", "ra", "dec", "vmag", "delta", "smaa", "smia", "theta"):
            v = getattr(self, k)
            if v is not None:
                setattr(self, k, np.asarray(v, float)[o])

    def __len__(self):
        return len(self.jd)

    # ------------------------------------------------------------ queries
    def at(self, jd):
        """Interpolated (ra, dec, vmag, smaa, smia, theta) at jd (scalar or array).

        Positions are interpolated on the unit sphere, so RA wrap is safe.
        """
        jd = np.asarray(jd, float)
        r, d = np.radians(self.ra), np.radians(self.dec)
        x = np.interp(jd, self.jd, np.cos(d) * np.cos(r))
        y = np.interp(jd, self.jd, np.cos(d) * np.sin(r))
        z = np.interp(jd, self.jd, np.sin(d))
        ra = np.degrees(np.arctan2(y, x)) % 360
        dec = np.degrees(np.arctan2(z, np.hypot(x, y)))

        def f(a):
            return None if a is None else np.interp(jd, self.jd, a)
        th = None
        if self.theta is not None:  # axial angle: interpolate doubled angle
            c = np.interp(jd, self.jd, np.cos(np.radians(2 * self.theta)))
            s = np.interp(jd, self.jd, np.sin(np.radians(2 * self.theta)))
            th = np.degrees(np.arctan2(s, c)) / 2 % 180
        return ra, dec, f(self.vmag), f(self.smaa), f(self.smia), th

    def ellipse(self, jd, nsigma: float):
        """(a, b, theta) of the n-sigma ellipse in arcsec at jd; floor of 1".

        For Monte Carlo tracks (Scout) the size follows the sample cloud's own
        containment, so heavy tails widen the 3-6 sigma ellipses honestly.
        """
        _, _, _, a, b, th = self.at(jd)
        if a is None:
            one = np.ones(np.shape(jd))
            return one, one.copy(), np.zeros(np.shape(jd))
        k = nsigma
        sc = self.meta.get("sigma_scale")
        if sc is not None:
            sc = np.asarray(sc)
            per_n = np.array([np.interp(jd, self.jd, row) for row in sc])   # (6, ...)
            lo = int(np.clip(np.floor(nsigma), 1, 6))
            hi = min(lo + 1, 6)
            f = np.clip(nsigma - lo, 0, 1)
            k = nsigma * (per_n[lo - 1] * (1 - f) + per_n[hi - 1] * f)
        return np.maximum(a * k, 1.0), np.maximum(b * k, 1.0), th

    def query_points(self, step_deg: float = 0.5, jd0=None, jd1=None):
        """Indices of track rows spaced at least step_deg apart (for archive queries)."""
        m = np.ones(len(self), bool)
        if jd0 is not None:
            m &= self.jd >= jd0
        if jd1 is not None:
            m &= self.jd <= jd1
        idx = np.flatnonzero(m)
        if idx.size == 0:
            return idx
        keep = [idx[0]]
        for i in idx[1:]:
            if sep_deg(self.ra[i], self.dec[i], self.ra[keep[-1]], self.dec[keep[-1]]) >= step_deg:
                keep.append(i)
        if keep[-1] != idx[-1]:
            keep.append(idx[-1])
        return np.array(keep)

    def rate(self, jd):
        """Sky motion (dRA*cos(dec), dDec) in arcsec/hour at jd."""
        h = 1 / 24
        r0, d0, *_ = self.at(np.asarray(jd) - h)
        r1, d1, *_ = self.at(np.asarray(jd) + h)
        dra = ((r1 - r0 + 540) % 360 - 180) * np.cos(np.radians((d0 + d1) / 2))
        return dra * 3600 / 2, (d1 - d0) * 3600 / 2

    # ------------------------------------------------------------ io
    def to_json(self) -> str:
        d = {k: (getattr(self, k).tolist() if isinstance(getattr(self, k), np.ndarray) else getattr(self, k))
             for k in ("name", "source", "jd", "ra", "dec", "vmag", "delta", "smaa", "smia", "theta", "site", "meta")}
        return json.dumps(d)

    @classmethod
    def from_json(cls, s: str) -> "Track":
        d = json.loads(s)
        for k in ("jd", "ra", "dec", "vmag", "delta", "smaa", "smia", "theta"):
            if d.get(k) is not None:
                d[k] = np.array(d[k], float)
        return cls(**d)

    @classmethod
    def from_samples(cls, name, source, jd, ra_s, dec_s, vmag=None, delta=None, nominal=None, **kw):
        """Build a track from Monte Carlo samples (S, M): nominal row + covariance ellipse."""
        ra_s, dec_s = np.atleast_2d(ra_s), np.atleast_2d(dec_s)
        if nominal is None:  # per-epoch median of the cloud (on the sphere, so RA wrap is safe)
            r, d = np.radians(ra_s), np.radians(dec_s)
            x, y, z = (np.median(np.cos(d) * np.cos(r), 0), np.median(np.cos(d) * np.sin(r), 0),
                       np.median(np.sin(d), 0))
            ra0 = np.degrees(np.arctan2(y, x)) % 360
            dec0 = np.degrees(np.arctan2(z, np.hypot(x, y)))
            k = ra_s.shape[1] // 2
            nominal = int(np.argmin(sep_deg(ra_s[:, k], dec_s[:, k], ra0[k], dec0[k])))
        else:
            ra0, dec0 = ra_s[nominal], dec_s[nominal]
        dx, dy = offset_arcsec(ra_s, dec_s, ra0, dec0)
        a, b, th, scales = cov_ellipse(dx, dy)
        if scales is not None:
            kw.setdefault("meta", {})["sigma_scale"] = scales.tolist()
        kw.setdefault("meta", {})["nominal_index"] = nominal
        return cls(name, source, jd, ra0, dec0, vmag,
                   None if delta is None else np.asarray(delta)[nominal] if np.ndim(delta) == 2 else delta,
                   a, b, th, **kw)


def cov_ellipse(dx, dy, clip: float = 3.03, iters: int = 10):
    """1-sigma ellipse (a, b, PA) from samples (S, M) of east/north offsets (arcsec).

    Short-arc clouds (Scout) have heavy tails, so samples beyond `clip`
    Mahalanobis units (99% for 2 dof) are dropped and the covariance redone.
    Deterministic: no random numbers.
    """
    dx = np.asarray(dx, float)
    dy = np.asarray(dy, float)
    keep = np.ones(dx.shape, bool)
    for _ in range(iters):
        n = keep.sum(0)
        mx = np.where(keep, dx, 0).sum(0) / n
        my = np.where(keep, dy, 0).sum(0) / n
        ux, uy = np.where(keep, dx - mx, 0), np.where(keep, dy - my, 0)
        sxx, syy, sxy = (ux * ux).sum(0) / n, (uy * uy).sum(0) / n, (ux * uy).sum(0) / n
        det = np.maximum(sxx * syy - sxy ** 2, 1e-30)
        m2 = ((dx - mx) ** 2 * syy - 2 * (dx - mx) * (dy - my) * sxy + (dy - my) ** 2 * sxx) / det
        new = m2 <= clip ** 2
        if np.array_equal(new, keep):
            break
        keep = new
    tr, det = sxx + syy, sxx * syy - sxy ** 2
    disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
    l1, l2 = tr / 2 + disc, np.maximum(tr / 2 - disc, 0)
    # clipping at 3.03 sigma trims a Gaussian's variance by ~5%; undo that
    corr = 1.0 / (1 - np.exp(-clip ** 2 / 2) * (1 + clip ** 2 / 2)) if clip < 10 else 1.0
    l1, l2 = l1 * corr, l2 * corr
    pa = np.degrees(0.5 * np.arctan2(2 * sxy, syy - sxx))
    scales = None
    if dx.shape[0] >= 50:
        # Per-sigma scale: the n-sigma ellipse is stretched so it holds the same
        # fraction of samples as a 2-D Gaussian would (39.3%, 86.5%, 98.9%, ...).
        sxx2, syy2, sxy2 = sxx * corr, syy * corr, sxy * corr
        det2 = np.maximum(sxx2 * syy2 - sxy2 ** 2, 1e-30)
        m2 = ((dx - mx) ** 2 * syy2 - 2 * (dx - mx) * (dy - my) * sxy2 + (dy - my) ** 2 * sxx2) / det2
        n = np.arange(1, 7)[:, None]
        frac = 1 - np.exp(-n[:, 0] ** 2 / 2)
        q = np.quantile(m2, np.minimum(frac, 1 - 0.5 / dx.shape[0]), axis=0)
        scales = np.sqrt(np.maximum(q, 1e-12)) / n          # (6, M), 1.0 for a Gaussian cloud
    return np.sqrt(l1), np.sqrt(l2), pa % 180, scales


def sep_deg(ra1, dec1, ra2, dec2):
    r1, d1, r2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    s = np.sin((d2 - d1) / 2) ** 2 + np.cos(d1) * np.cos(d2) * np.sin((r2 - r1) / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(s, 0, 1))))


def offset_arcsec(ra, dec, ra0, dec0):
    """Tangent-plane (east, north) offsets of (ra, dec) from (ra0, dec0) in arcsec.

    NaN for points more than 90 deg from (ra0, dec0).
    """
    r, d, r0, d0 = map(np.radians, (ra, dec, ra0, dec0))
    cosc = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(r - r0)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = np.cos(d) * np.sin(r - r0) / cosc
        eta = (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(r - r0)) / cosc
    # the projection is only defined on the near hemisphere; beyond 90 deg the
    # formula folds far-side points back near the centre, so mark them undefined
    far = cosc <= 0
    xi = np.where(far, np.nan, xi)
    eta = np.where(far, np.nan, eta)
    return np.degrees(xi) * 3600, np.degrees(eta) * 3600


def earth_helio(jd_utc):
    """Heliocentric equatorial J2000 position of the geocentre (au), via ERFA."""
    from astropy.time import Time
    import erfa
    t = Time(np.atleast_1d(jd_utc), format="jd", scale="utc").tdb
    pvh, _ = erfa.epv00(t.jd1, t.jd2)
    return np.asarray(pvh["p"])


def track_from_elements(name, source, elements, jd, nominal=None, H=None, G=0.15):
    """Propagate element samples (S, 6) with the two-body core; returns a Track.

    Two-body is fine for sigma clouds over a few years; for long arcs the
    nominal track should come from Horizons or Find_Orb (n-body).
    """
    from astropy.time import Time
    jd = np.asarray(jd, float)
    tdb = Time(jd, format="jd", scale="utc").tdb.jd
    E = earth_helio(jd)
    ra, dec, delta = core.kepler_radec(np.asarray(elements, float), tdb, E)
    t = Track.from_samples(name, source, jd, ra, dec, None, delta, nominal=nominal)
    if H is not None:
        k = t.meta["nominal_index"]
        el = np.asarray(elements, float)[k]
        from .core import _helio_xyz
        P = _helio_xyz(el[None, :], tdb)[0]
        r = np.linalg.norm(P, axis=1)
        d = delta[k]
        cosph = np.clip((r ** 2 + d ** 2 - np.sum(E ** 2, 1)) / (2 * r * d), -1, 1)
        ph = np.arccos(cosph)
        phi1 = np.exp(-3.33 * np.tan(ph / 2) ** 0.63)
        phi2 = np.exp(-1.87 * np.tan(ph / 2) ** 1.22)
        t.vmag = H + 5 * np.log10(r * d) - 2.5 * np.log10((1 - G) * phi1 + G * phi2)
    return t


def ellipse_overlaps_box(dx, dy, a, b, th, half_w, half_h):
    """Does the error ellipse centred (dx, dy) arcsec from a field centre touch the box?

    Conservative: tests the ellipse's bounding box against the field box.
    """
    t = np.radians(th)
    ex = np.sqrt((a * np.sin(t)) ** 2 + (b * np.cos(t)) ** 2)
    ey = np.sqrt((a * np.cos(t)) ** 2 + (b * np.sin(t)) ** 2)
    return (np.abs(dx) <= half_w + ex) & (np.abs(dy) <= half_h + ey)


def inside_fraction(dx, dy, a, b, th, half_w, half_h, n=400, seed=0):
    """Fraction of the n-sigma ellipse area that lies inside the field box (deterministic)."""
    if not (np.isfinite(dx) and np.isfinite(dy)):
        return 0.0
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1, 1, (n, 2))
    u = u[(u ** 2).sum(1) <= 1]
    t = math.radians(float(th))
    ex = dx + u[:, 0] * b * math.cos(t) + u[:, 1] * a * math.sin(t)
    ey = dy - u[:, 0] * b * math.sin(t) + u[:, 1] * a * math.cos(t)
    return float(np.mean((np.abs(ex) <= half_w) & (np.abs(ey) <= half_h)))
