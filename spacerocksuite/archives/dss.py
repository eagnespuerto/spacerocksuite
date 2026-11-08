"""STScI Digitized Sky Survey cutouts (POSS-I/II, SERC, AAO) for checking a position.

DSS serves one plate per survey at a position, with the plate's date in the
header. It is used to fetch stamps of candidate plates and to list which DSS
plate covers each track point."""
from __future__ import annotations

import io

import numpy as np

from .. import net
from ..device import PROFILE
from ..track import inside_fraction, offset_arcsec
from .base import Archive, PlateHit, pmap

URL = "https://archive.stsci.edu/cgi-bin/dss_search"
SURVEYS = {"poss1_red": "POSS-I E", "poss1_blue": "POSS-I O", "poss2ukstu_red": "POSS-II/UKST R",
           "poss2ukstu_blue": "POSS-II/UKST B", "poss2ukstu_ir": "POSS-II/UKST I"}


def cutout(ra, dec, size_arcmin=10.0, survey="poss1_red"):
    """FITS bytes of a DSS cutout."""
    return net.fetch(URL, {"v": survey, "r": f"{ra:.6f}", "d": f"{dec:.6f}", "e": "J2000",
                           "h": size_arcmin, "w": size_arcmin, "f": "fits"}, timeout=300, cache_hours=24 * 30)


class DSS(Archive):
    name = "DSS (STScI)"
    description = "Digitized Sky Survey plates: POSS-I (1949-58), POSS-II, SERC/AAO"
    years = (1949, 2000)

    # POSS-I blue is the one DSS survey SuperCOSMOS does not have, so it is the default
    def __init__(self, surveys=("poss1_blue",)):
        self.surveys = surveys

    def plate_info(self, survey, ra, dec):
        from astropy.io import fits
        from astropy.time import Time
        h = fits.getheader(io.BytesIO(cutout(ra, dec, 1.0, survey)))
        date = h.get("DATE-OBS")
        if not date:
            return None
        exp = float(h.get("EXPOSURE", 0) or 0)          # minutes
        # DSS headers can carry minutes > 59 (e.g. 05:82:00): add the parts by hand
        d, _, t = date.replace("/", "-").partition("T")
        hh, mm, ss = ([float(x) for x in t.split(":")] + [0, 0, 0])[:3] if t else (0, 0, 0)
        jd = Time(d, scale="utc").jd + (hh * 3600 + mm * 60 + ss) / 86400 + exp / 2 / 1440
        return dict(survey=survey, plate=str(h.get("PLATEID", "")).strip(), jd=jd, exp_min=exp,
                    ra_c=float(h.get("PLTRAH", 0)) * 15 + float(h.get("PLTRAM", 0)) / 4 + float(h.get("PLTRAS", 0)) / 240,
                    dec_c=(-1 if str(h.get("PLTDECSN", "+")).strip() == "-" else 1)
                    * (float(h.get("PLTDECD", 0)) + float(h.get("PLTDECM", 0)) / 60 + float(h.get("PLTDECS", 0)) / 3600),
                    emulsion=str(h.get("EMULSION", "")).strip(), filt=str(h.get("FILTER", "")).strip())

    def search(self, track, nsigma, jd0=None, jd1=None, progress=None, stop=None):
        idx = track.query_points(PROFILE.track_step_deg * 4, jd0, jd1)
        jobs = [(s, float(track.ra[i]), float(track.dec[i])) for i in idx for s in self.surveys]
        infos = pmap(lambda j: self.plate_info(*j), jobs, workers=min(3, PROFILE.net_workers),
                     progress=progress, stop=stop)
        seen = {}
        for inf in infos:
            if isinstance(inf, dict):
                seen[(inf["survey"], inf["plate"], round(inf["jd"], 3))] = inf
        hits = []
        for inf in seen.values():
            jd = inf["jd"]
            if not (track.jd[0] <= jd <= track.jd[-1]):
                continue
            ra, dec, v, *_ = track.at(jd)
            a, b, th = track.ellipse(jd, nsigma)
            dx, dy = offset_arcsec(ra, dec, inf["ra_c"], inf["dec_c"])
            half = 3.2 * 3600
            frac = inside_fraction(float(dx), float(dy), float(a), float(b), float(th), half, half)
            if frac <= 0:
                continue
            rr, rd = track.rate(jd)
            exp_s = inf["exp_min"] * 60
            name = SURVEYS.get(inf["survey"], inf["survey"])
            hits.append(PlateHit(
                self.name, f"{name} {inf['plate']}", name, f"{inf['emulsion']} {inf['filt']}".strip(), jd, exp_s,
                float(ra), float(dec), None if v is None else float(v), float(a), float(b), float(th),
                limit_mag={"poss1_red": 20.0, "poss1_blue": 21.0}.get(inf["survey"], 21.5),
                margin_arcsec=half - max(abs(float(dx)), abs(float(dy))), covered_fraction=frac,
                trail_arcsec=float(np.hypot(rr, rd) * exp_s / 3600), url=URL,
                note="DATE-OBS from DSS header (exposure start); mid-time added",
                extra={"survey": inf["survey"]}))
        hits.sort(key=lambda h: h.mid_jd)
        return hits
