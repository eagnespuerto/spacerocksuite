"""Stacking and shift-stacking of archive frames on a predicted track.

Each frame is resampled onto a north-up, east-left grid of fixed pixel scale,
centred on:
  - "static": one sky position (ordinary stack; stars add up)
  - "track":  the object's predicted position at each frame's mid-time
              (shift-stack; the mover adds up, stars smear)
  - "rate":   a fixed rate (dRA*cos(dec), dDec in "/h) from a reference time,
              for synthetic tracking when the motion is uncertain

Noise comes from control tracks offset from the real one (as in the EE138
TESS/PS1 and ED138 Swift work), and null tests repeat the stack with reversed,
zero, rotated and doubled motion. Everything is seeded and deterministic.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import numpy as np

from .. import core
from ..device import PROFILE
from . import cleanup


@dataclass
class Frame:
    data: np.ndarray
    wcs: object                  # astropy.wcs.WCS
    jd_mid: float
    exptime: float = 0.0
    zp: float | None = None      # magnitude zero point (flux scale = 10**(-0.4*(zp-25)))
    name: str = ""
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_fits(cls, src, jd_mid=None, ext=None, name=""):
        """Read a FITS file/bytes. jd_mid from MJD-OBS/DATE-OBS (+EXPTIME/2) if not given."""
        from astropy.io import fits
        from astropy.wcs import WCS
        from astropy.time import Time
        hdul = fits.open(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src)
        hdu = hdul[ext] if ext is not None else next(h for h in hdul if getattr(h, "data", None) is not None
                                                     and np.ndim(h.data) == 2)
        h = hdu.header
        exp = float(h.get("EXPTIME", h.get("EXPOSURE", 0)) or 0)
        if jd_mid is None:
            if "MJD-MID" in h:
                jd_mid = float(h["MJD-MID"]) + 2400000.5
            elif "MJD-OBS" in h:
                jd_mid = float(h["MJD-OBS"]) + 2400000.5 + exp / 2 / 86400
            elif "DATE-OBS" in h:
                jd_mid = Time(h["DATE-OBS"]).jd + exp / 2 / 86400
            else:
                raise ValueError("no time in header; pass jd_mid")
        zp = h.get("MAGZPT", h.get("PHOTZP"))
        return cls(np.asarray(hdu.data, np.float32), clean_wcs(h), float(jd_mid), exp,
                   None if zp is None else float(zp), name or h.get("FILENAME", ""), {"header": h})


DSS_KEYS = ("OBJCTRA", "OBJCTDEC", "OBJCTX", "OBJCTY", "XPIXELSZ", "YPIXELSZ", "PPO1", "PPO2", "PPO3",
            "PPO4", "PPO5", "PPO6", "PLTRAH", "PLTRAM", "PLTRAS", "PLTDECSN", "PLTDECD", "PLTDECM",
            "PLTDECS", "PLTSCALE", "CNPIX1", "CNPIX2") + tuple(f"AMDX{i}" for i in range(1, 21))            + tuple(f"AMDY{i}" for i in range(1, 21))


def clean_wcs(h):
    """WCS from a header; drops DSS plate-model keywords when a TAN solution is present
    (SuperCOSMOS cutouts carry both, and astropy chokes on the mix)."""
    import warnings
    from astropy.wcs import WCS, FITSFixedWarning
    h = h.copy()
    if "CTYPE1" in h:
        for k in DSS_KEYS:
            h.remove(k, ignore_missing=True, remove_all=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        return WCS(h)


def _bilinear(img, x, y):
    H, W = img.shape
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    ok = (x0 >= 0) & (y0 >= 0) & (x0 < W - 1) & (y0 < H - 1)
    x0c, y0c = np.clip(x0, 0, W - 2), np.clip(y0, 0, H - 2)
    v = (img[y0c, x0c] * (1 - fx) * (1 - fy) + img[y0c, x0c + 1] * fx * (1 - fy)
         + img[y0c + 1, x0c] * (1 - fx) * fy + img[y0c + 1, x0c + 1] * fx * fy)
    return np.where(ok, v, np.nan)


def resample(frame: Frame, ra0, dec0, size_px: int, scale_arcsec: float):
    """North-up, east-left stamp of `frame` centred on (ra0, dec0) (gnomonic grid)."""
    c = (size_px - 1) / 2
    yy, xx = np.mgrid[:size_px, :size_px].astype(float)
    xi = -(xx - c) * scale_arcsec / 3600          # east to the left
    eta = (yy - c) * scale_arcsec / 3600
    xi, eta = np.radians(xi), np.radians(eta)
    r0, d0 = math.radians(ra0), math.radians(dec0)
    den = math.cos(d0) - eta * math.sin(d0)
    ra = np.degrees(r0 + np.arctan2(xi, den)) % 360
    dec = np.degrees(np.arctan2(math.sin(d0) + eta * math.cos(d0), np.hypot(xi, den)))
    px, py = frame.wcs.all_world2pix(ra, dec, 0)
    return _bilinear(frame.data.astype(np.float64), px, py).astype(np.float32)


def flux_scale(frame: Frame) -> float:
    """Scale to a common zero point of 25 (counts -> 'ZP25' units)."""
    s = 1.0
    if frame.zp is not None:
        s = 10 ** (-0.4 * (frame.zp - 25))
    return s


def _centres(frames, track, mode, rate=None, ref_jd=None, offset=(0.0, 0.0)):
    """Stamp centres per frame. offset (east, north) arcsec shifts the whole track (controls)."""
    jd = np.array([f.jd_mid for f in frames])
    if mode == "static":
        ra, dec, *_ = track.at(np.full(jd.size, ref_jd if ref_jd is not None else np.median(jd)))
    elif mode == "track":
        ra, dec, *_ = track.at(jd)
    elif mode == "rate":
        ref = ref_jd if ref_jd is not None else np.median(jd)
        r0, d0, *_ = track.at(ref)
        dt_h = (jd - ref) * 24
        dec = d0 + rate[1] * dt_h / 3600
        ra = r0 + rate[0] * dt_h / 3600 / np.cos(np.radians(dec))
    else:
        raise ValueError(mode)
    ra = np.asarray(ra, float)
    dec = np.asarray(dec, float)
    if offset != (0.0, 0.0):
        dec = dec + offset[1] / 3600
        ra = ra + offset[0] / 3600 / np.cos(np.radians(dec))
    return ra, dec


class NoCoverage(ValueError):
    """No frame contains the predicted position."""


def check_frames(frames, track, mode="track", nsigma=3.0):
    """Where the predicted position falls on each frame.

    Returns one dict per frame: name, jd, x, y (pixels), on (prediction inside
    the frame), off_arcmin (distance from the frame edge when it is outside),
    in_span (frame time inside the track's time range), ell_arcmin (n-sigma
    semi-major axis) and vmag.
    """
    ra, dec = _centres(frames, track, mode)
    jd = np.array([f.jd_mid for f in frames])
    _, _, vmag, *_ = track.at(jd)
    a, _, _ = track.ellipse(jd, nsigma)
    out = []
    for k, f in enumerate(frames):
        H, W = f.data.shape[-2:]
        x, y = (float(v) for v in f.wcs.all_world2pix(ra[k], dec[k], 0))
        ok = np.isfinite(x) and np.isfinite(y)
        on = ok and -0.5 <= x <= W - 0.5 and -0.5 <= y <= H - 0.5
        off = 0.0
        if ok and not on:  # sky distance to the nearest point on the frame
            ex, ey = min(max(x, 0), W - 1), min(max(y, 0), H - 1)
            r2, d2 = f.wcs.all_pix2world(ex, ey, 0)
            from ..track import sep_deg
            off = float(sep_deg(ra[k], dec[k], float(r2), float(d2))) * 60
        out.append({"name": f.name or f"frame {k + 1}", "jd": float(jd[k]), "x": x, "y": y, "on": bool(on),
                    "off_arcmin": off if ok else float("inf"),
                    "in_span": bool(track.jd[0] - 1e-6 <= jd[k] <= track.jd[-1] + 1e-6),
                    "ell_arcmin": float(np.atleast_1d(a)[k]) / 60,
                    "vmag": None if vmag is None else float(np.atleast_1d(vmag)[k])})
    return out


def usable_frames(frames, track, mode="track", nsigma=3.0, log=print):
    """Frames whose predicted position is on the image; logs why others are dropped.

    Raises NoCoverage, with a plain explanation, when none are left.
    """
    rep = check_frames(frames, track, mode, nsigma)
    keep = []
    for f, r in zip(frames, rep):
        if not r["in_span"]:
            log(f"  {r['name']}: frame time is outside the track ({track.jd[0]:.2f}-{track.jd[-1]:.2f} JD); "
                "get a track that covers it")
        elif not r["on"]:
            log(f"  {r['name']}: prediction is {r['off_arcmin']:.1f}' outside this frame, skipped")
        else:
            keep.append(f)
    v = [r["vmag"] for r in rep if r["vmag"] is not None and np.isfinite(r["vmag"])]
    ell = max(r["ell_arcmin"] for r in rep)
    log(f"  prediction: {nsigma:g}-sigma ellipse up to {ell:.1f}'" + (f", predicted V {min(v):.1f}" if v else ""))
    if not keep:
        near = min(rep, key=lambda r: r["off_arcmin"])
        msg = (f"The predicted position is not on any of the {len(frames)} frames "
               f"(nearest: {near['off_arcmin']:.1f}' outside {near['name']}).")
        if not all(r["in_span"] for r in rep):
            msg += " Some frame times are outside the track; get a track that covers them."
        elif ell > near["off_arcmin"]:
            msg += (f" The {nsigma:g}-sigma ellipse ({ell:.1f}') reaches these frames, but a stack is centred on "
                    "the prediction, so they cannot be stacked on it.")
        raise NoCoverage(msg)
    return keep


def stack(frames, track, mode="track", size_px=None, scale_arcsec=1.0, method="clipmean",
          rate=None, ref_jd=None, bkg_size=None, templates=None, offset=(0.0, 0.0)):
    """Resample every frame onto the stamp grid and combine.

    templates: optional list of (frame) used to build a static template that is
    subtracted from each science stamp first (difference shift-stack).
    Returns dict(stack, coverage, stamps, centres).
    """
    size_px = size_px or PROFILE.stamp_px
    ra, dec = _centres(frames, track, mode, rate, ref_jd, offset)
    stamps = []
    for f, r, d in zip(frames, ra, dec):
        s = resample(f, r, d, size_px, scale_arcsec) * flux_scale(f)
        if bkg_size:
            s = cleanup.background(np.nan_to_num(s, nan=np.nanmedian(s)), bkg_size)
        stamps.append(s)
    S = np.stack(stamps)
    if templates:
        T = np.stack([np.stack([resample(t, r, d, size_px, scale_arcsec) * flux_scale(t) for t in templates])
                      for r, d in zip(ra, dec)])                  # (K, nT, H, W)
        S = S - np.nanmedian(T, 1)
    st, cov = core.combine(S, method)
    return {"stack": st, "coverage": cov, "stamps": S, "centres": np.column_stack([ra, dec]),
            "scale": scale_arcsec, "mode": mode}


def photometry(img, r_px, ann=(2.0, 3.5)):
    """Aperture sum at the centre minus the annulus median * area."""
    H, W = img.shape
    c = ((W - 1) / 2, (H - 1) / 2)
    yy, xx = np.ogrid[:H, :W]
    d = np.hypot(xx - c[0], yy - c[1])
    ap = d <= r_px
    an = (d >= ann[0] * r_px) & (d <= ann[1] * r_px)
    n = int(np.isfinite(img[ap]).sum())
    if n == 0 or not np.isfinite(img[an]).any():   # no data at the prediction: not a zero flux
        return float("nan"), n
    sky = np.nanmedian(img[an])
    return float(np.nansum(img[ap] - sky)), n


def control_noise(frames, track, mode, n_controls=8, sep_arcsec=None, r_px=3.0, **kw):
    """Repeat the stack on n control tracks offset by sep_arcsec around the real one;
    the scatter of their aperture fluxes is the noise (includes star residuals)."""
    sz = kw.get("size_px") or PROFILE.stamp_px
    sep = sep_arcsec or 12 * kw.get("scale_arcsec", 1.0)
    fl = []
    for k in range(n_controls):
        a = 2 * math.pi * k / n_controls
        res = stack(frames, track, mode, offset=(sep * math.cos(a), sep * math.sin(a)), **kw)
        fl.append(photometry(res["stack"], r_px)[0])
    fl = np.array(fl)
    ok = fl[np.isfinite(fl)]                       # controls off the frame edge carry no data
    return float(np.std(ok, ddof=1)) if len(ok) > 2 else float("nan"), fl


def null_tests(frames, track, r_px=3.0, **kw):
    """Stacks with the predicted, reversed, zero, 90-degree-rotated and doubled motion.

    A real mover is strongest for 'predicted' (Swift/UVOT ED138 test pattern).
    Returns {name: flux}.
    """
    jd = np.array([f.jd_mid for f in frames])
    ref = float(np.median(jd))
    rr, rd = track.rate(ref)
    rr, rd = float(rr), float(rd)
    tests = {"predicted": (rr, rd), "reversed": (-rr, -rd), "stationary": (0.0, 0.0),
             "rotated 90": (-rd, rr), "double rate": (2 * rr, 2 * rd)}
    out = {}
    for name, rate in tests.items():
        res = stack(frames, track, "rate", rate=rate, ref_jd=ref, **kw)
        out[name] = photometry(res["stack"], r_px)[0]
    return out


def rate_grid(frames, track, span=0.5, n=5, r_px=3.0, **kw):
    """Synthetic tracking over a grid of rates around the predicted one.

    span: +/- fraction of the predicted rate. Returns (rates (n*n, 2), fluxes).
    """
    ref = float(np.median([f.jd_mid for f in frames]))
    rr, rd = (float(v) for v in track.rate(ref))
    sp = span * max(math.hypot(rr, rd), 0.1)
    rates, fl = [], []
    for a in np.linspace(-sp, sp, n):
        for b in np.linspace(-sp, sp, n):
            r = (rr + a, rd + b)
            res = stack(frames, track, "rate", rate=r, ref_jd=ref, **kw)
            rates.append(r)
            fl.append(photometry(res["stack"], r_px)[0])
    return np.array(rates), np.array(fl)
