"""DASCH DR7: digitised Harvard plates, 1885-1992.

Method from the 2026 ED138 DASCH search: query exposures at points every
~0.5 deg along the track, then keep a plate when the object's position at the
plate's own epoch is on the plate (edge distance minus offset, widened by the
n-sigma ellipse)."""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np

from .. import net
from ..device import PROFILE
from ..track import sep_deg
from .base import Archive, PlateHit, pmap

URL = "https://api.starglass.cfa.harvard.edu/public/dasch/dr7/queryexps"


def _jd(iso: str) -> float:
    """ISO time -> JD; tolerant of DASCH values such as '...:60.0Z'."""
    d, t = iso.rstrip("Z").split("T")
    day = datetime.strptime(d, "%Y-%m-%d")
    h, m, s = (t.split(":") + ["0", "0"])[:3]
    secs = int(h) * 3600 + int(m) * 60 + float(s)
    return (day - datetime(2000, 1, 1, 12)).total_seconds() / 86400 + 2451545.0 + secs / 86400


class DASCH(Archive):
    name = "DASCH (Harvard)"
    description = "Harvard College Observatory plates 1885-1992 (B ~ 12-18)"
    years = (1885, 1993)

    def query(self, ra, dec):
        body = json.dumps({"ra_deg": float(ra), "dec_deg": float(dec)}).encode()
        rows = json.loads(net.fetch(URL, data=body, headers={"Content-Type": "application/json"},
                                    timeout=300, cache_hours=24 * 30))
        hdr = rows[0].split(",")
        return [dict(zip(hdr, r.split(","))) for r in rows[1:]]

    def search(self, track, nsigma, jd0=None, jd1=None, progress=None, stop=None):
        jd0 = max(jd0 or track.jd[0], _jd("1885-01-01T00:00:00"))
        jd1 = min(jd1 or track.jd[-1], _jd("1993-01-01T00:00:00"))
        if jd1 <= jd0:
            return []
        idx = track.query_points(PROFILE.track_step_deg, jd0, jd1)
        pts = [(float(track.ra[i]), float(track.dec[i])) for i in idx]
        res = pmap(lambda p: self.query(*p), pts, progress=progress, stop=stop)
        plates = {}
        for (qra, qdec), rows in zip(pts, res):
            if isinstance(rows, Exception) or rows is None:
                continue
            for d in rows:
                if not d.get("expdate") or not d.get("edgedist"):
                    continue
                key = (d["series"], d["platenum"], d["expnum"])
                plates.setdefault(key, []).append((qra, qdec, d))
        hits = []
        for key, lst in plates.items():
            d = lst[0][2]
            jd = _jd(d["expdate"])          # DASCH expdate: exposure mid-point
            if not (jd0 <= jd <= jd1):
                continue
            ra, dec, v, *_ = track.at(jd)
            a, b, th = track.ellipse(jd, nsigma)
            # distance from object to plate edge = edgedist(query) - |object - query|, best over queries
            margin = max(float(x[2]["edgedist"]) - float(sep_deg(ra, dec, x[0], x[1])) for x in lst) * 3600
            if margin + a < 0:            # whole ellipse off the plate
                continue
            frac = 1.0 if margin >= a else max(0.0, min(1.0, 0.5 + margin / (2 * a)))
            lim = d.get("limMagAtlas") or d.get("limMagApass")
            exp_s = float(d.get("exptime") or 0) * 60
            rr, rd = track.rate(jd)
            plate = f"{d['series']}{d['platenum']}" + (f"_{d['expnum']}" if d["expnum"] not in ("0", "") else "")
            hits.append(PlateHit(
                self.name, plate, f"HCO {d['series']}", "B (pg)", jd, exp_s, float(ra), float(dec),
                None if v is None else float(v), float(a), float(b), float(th),
                limit_mag=float(lim) if lim else None, margin_arcsec=margin, covered_fraction=frac,
                trail_arcsec=float(np.hypot(rr, rd) * exp_s / 3600),
                url=f"https://starglass.cfa.harvard.edu/plate/{d['series']}{int(d['platenum']):05d}",
                note="" if d.get("wcssource") else "no WCS", extra={"wcs": d.get("wcssource", "")}))
        hits.sort(key=lambda h: h.mid_jd)
        return hits
