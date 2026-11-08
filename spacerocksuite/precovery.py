"""The precovery pipeline: track -> plate search -> ranking -> stamps -> grid.

Each step is a plain function so the GUI, the CLI and scripts share it.
"""
from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass

import numpy as np

from .track import Track

warnings.filterwarnings("ignore", message=".*dubious year.*")


def jd(s: str) -> float:
    from astropy.time import Time
    return float(Time(s.replace("/", "-"), scale="utc").jd)


def iso(j: float) -> str:
    from astropy.time import Time
    return Time(j, format="jd", scale="utc").iso[:16]


# ------------------------------------------------------------------ tracks
def get_track(target: str, start: str, stop: str, source: str = "horizons", step_days: float = 1.0,
              site: str = "500", obs80: str | None = None, ephem_text: str | None = None,
              findorb_binary: str | None = None, progress=None) -> Track:
    """source: horizons | scout | findorb | ephemeral."""
    j0, j1 = jd(start), jd(stop)
    if source == "horizons":
        from .services import horizons
        return horizons.ephemeris_long(target, j0, j1, step_days, site, progress)
    if source == "scout":
        from .services import scout, mpc
        t = scout.track(target, np.arange(j0, j1 + step_days / 2, step_days))
        try:  # keep the NEOCP astrometry for arc-based searches (SSOIS) and Find_Orb
            t.meta["obs80"] = mpc.neocp_observations(target)
        except Exception:
            pass
        return t
    if source == "findorb":
        from .services import findorb, mpc
        obs = obs80 or mpc.observations(target)
        return findorb.ephemeris(obs, j0, j1, step_days, site, target, findorb_binary, progress)
    if source == "ephemeral":
        from .services import ephemeral
        if not ephem_text:
            raise ValueError("paste an ephemeris first")
        return ephemeral.track_from_text(ephem_text, target or None)
    raise ValueError(f"unknown source {source!r}")


# ------------------------------------------------------------------ plates
def search_plates(track: Track, nsigma: float, archives: list[str], jd0=None, jd1=None,
                  obs80=None, date_fix_days=1.0, progress=None, stop=None, log=print):
    from . import archives as A
    hits = []
    for k, name in enumerate(archives):
        if stop is not None and stop.is_set():
            break
        if name == "SSOIS":
            undesignated = track.source in ("scout", "findorb", "ephemeral")
            arc = A.SSOIS(obs80=(obs80 or track.meta.get("obs80")) if undesignated else None,
                          elements=track.meta.get("elements") if track.source == "scout" else None)
        elif name == "SuperCOSMOS":
            arc = A.SuperCOSMOS(date_fix_days=date_fix_days)
        else:
            arc = A.ALL[name]()
        lo, hi = arc.years
        a = max(jd0 or track.jd[0], track.jd[0], jd(f"{lo}-01-01"))
        b = min(jd1 or track.jd[-1], track.jd[-1], jd(f"{min(hi, 2100)}-01-01"))
        if b <= a:
            log(f"{arc.name}: track outside {lo}-{hi}, skipped")
            continue
        log(f"{arc.name}: searching {iso(a)[:10]} to {iso(b)[:10]} at {nsigma:g} sigma ...")

        def prog(f, k=k):
            if progress:
                progress((k + f) / len(archives))
        try:
            h = arc.search(track, nsigma, a, b, prog, stop)
        except Exception as e:      # one archive failing should not stop the rest
            log(f"{arc.name}: failed ({e})")
            continue
        for m in getattr(arc, "messages", []):
            log(f"{arc.name}: {m}")
        log(f"{arc.name}: {len(h)} frames/plates")
        hits += h
    return rank(hits)


def rank(hits):
    """Deepest-reaching first: depth margin (limit - V), then coverage, then time."""
    def key(h):
        dm = h.depth_margin
        return (-(dm if dm is not None else -5), -h.covered_fraction, h.mid_jd)
    return sorted(hits, key=key)


# ------------------------------------------------------------------ stamps
def fetch_cutout(hit, size_arcmin=None):
    """FITS bytes of a stamp around the predicted position, where the archive serves one."""
    size = size_arcmin or max(2.0, min(30.0, 3 * hit.sigma_a / 60 + 1.0))
    if hit.archive.startswith("SuperCOSMOS"):
        from .archives.supercosmos import cutout
        return cutout(hit.ra_obj, hit.dec_obj, hit.extra["survey_id"], size)
    if hit.archive.startswith("DSS"):
        from .archives.dss import cutout
        return cutout(hit.ra_obj, hit.dec_obj, size, hit.extra.get("survey", "poss1_red"))
    raise NotImplementedError(f"no cutout service for {hit.archive}; open {hit.url}")


def check_plate_header(fits_bytes, hit) -> str:
    """Warn when the cutout server picked a different plate than the hit."""
    from astropy.io import fits
    import io
    h = fits.getheader(io.BytesIO(fits_bytes))
    got = str(h.get("PLATENUM", h.get("PLATEID", ""))).strip()
    want = hit.plate_id.split()[-1]
    return "" if not got or got == want else f"cutout is from plate {got}, not {want}"


@dataclass
class StampResult:
    hit: object
    image: np.ndarray        # cleaned, north-up stamp centred on the prediction
    scale: float             # arcsec / px
    hw: float                # half-width arcsec
    r_ap: float              # aperture radius arcsec
    flux: float
    noise: float
    snr: float
    note: str


def measure_stamp(fits_bytes, hit, fwhm_arcsec=3.5, size_arcsec=None, bkg_px=41, matched=False):
    """Resample the cutout north-up on the prediction, clean it and do forced photometry."""
    from .imaging import cleanup
    from .imaging.stack import Frame, resample
    fr = Frame.from_fits(fits_bytes, jd_mid=hit.mid_jd)
    from astropy.wcs.utils import proj_plane_pixel_scales
    scale = float(np.mean(np.asarray(proj_plane_pixel_scales(fr.wcs.celestial), float)) * 3600)
    size_arcsec = size_arcsec or max(30.0, 2.5 * hit.sigma_a + 20)
    n = int(round(size_arcsec / scale)) | 1
    img = resample(fr, hit.ra_obj, hit.dec_obj, n, scale)
    img = np.nan_to_num(img, nan=float(np.nanmedian(img)))
    img = cleanup.background(img, min(bkg_px, (n // 2) * 2 - 1))
    if matched:
        img = cleanup.matched(img, fwhm_arcsec / scale)
    r_px = max(1.5, 0.75 * fwhm_arcsec / scale)
    c = (n - 1) / 2
    flux, _ = cleanup.aperture_sum(img, c, c, r_px)
    noise, _ = cleanup.empirical_noise(img, c, c, r_px, n=200, seed=0)
    snr = flux / noise if noise and noise > 0 else float("nan")
    return StampResult(hit, img, scale, n * scale / 2, r_px * scale, flux, noise, snr,
                       check_plate_header(fits_bytes, hit))


def stampgrid_name(title):
    from .imaging.stampgrid import default_name
    return default_name(title)


def grid_spec(object_name, results, folder, subtitle=None):
    """Write stamps (.npz) + stamp_grid.json for the house-style grid; returns spec path."""
    os.makedirs(folder, exist_ok=True)
    arrays = {f"p{i}": r.image for i, r in enumerate(results)}
    np.savez_compressed(os.path.join(folder, "stamps.npz"), **arrays)
    panels = []
    for i, r in enumerate(results):
        h = r.hit
        line2 = (f"S/N {r.snr:.1f} at prediction" if np.isfinite(r.snr) else "no measurement")
        if h.vmag_pred is not None and h.limit_mag is not None:
            line2 += f", V {h.vmag_pred:.1f} vs limit {h.limit_mag:.1f}"
        panels.append({"file": "stamps.npz", "key": f"p{i}", "hw": round(r.hw, 2), "r": round(r.r_ap, 2),
                       "title": f"{h.plate_id}, {iso(h.mid_jd)[:10]}", "subtitle": line2})
    spec = {"title": object_name,
            "subtitle": subtitle or "Archival plates and frames (centred on predicted position)",
            "out": stampgrid_name(object_name), "ncols": 3, "panels": panels}
    p = os.path.join(folder, "stamp_grid.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1, ensure_ascii=False)
    return p


def measure_with_date_probe(fits_bytes, hit, track, offsets=(-1.0, 0.0, 1.0), **kw):
    """Measure a SuperCOSMOS stamp at the hit's date and at nearby whole days.

    SSA plate dates can be wrong by one or two days (2026 ED138: +1 d; Eris on
    POSS-I E 1194: +2 d). The probe re-centres the stamp on the track at each
    trial date. It switches only when one date is clearly best (S/N >= 5 and at
    least twice every other trial); the note always lists all trials.
    """
    import copy
    trials = []
    for d in offsets:
        h = copy.copy(hit)
        h.extra = dict(hit.extra)
        h.mid_jd = hit.mid_jd + d
        ra, dec, v, *_ = track.at(h.mid_jd)
        h.ra_obj, h.dec_obj = float(ra), float(dec)
        h.vmag_pred = None if v is None else float(v)
        trials.append((d, measure_stamp(fits_bytes, h, **kw)))
    snr = {d: (r.snr if np.isfinite(r.snr) else -np.inf) for d, r in trials}
    best = max(snr, key=snr.get)
    clear = snr[best] >= 5 and all(snr[best] >= 2 * max(s, 0.5) for d, s in snr.items() if d != best)
    pick = best if clear else 0.0
    r = dict(trials)[pick]
    listing = ", ".join(f"{d:+.0f} d: " + (f"{s:.1f}" if np.isfinite(s) else "n/a") for d, s in snr.items())
    r.note = (r.note + "; " if r.note else "") + f"date probe S/N ({listing})" + \
        (f"; plate date moved {pick:+.0f} d" if pick else "")
    return r
