"""SuperCOSMOS: scanned UKST (SERC), ESO Schmidt, POSS-I and POSS-II plates, 1949-2003.

Searched offline against the SSA Plate table (6,646 plates, bundled; refresh
from ssa.roe.ac.uk in Settings). For each plate the object's position at the
plate's own epoch is tested against the plate area, widened by the n-sigma
ellipse. Cutouts come from the SSS pixel server.

Date caveat: in the 2026 ED138 work the SSA dates of ESO, UKST and POSS-I
plates were one day early (checked with known asteroids). `date_fix_days`
(default +1) applies that; check it with SkyBoT before you submit. The SSA
`mjd` is already mid-exposure.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import re
import time
from importlib import resources

import numpy as np

from .. import net
from ..track import offset_arcsec, inside_fraction
from .base import Archive, PlateHit

PIXEL = "http://www-wfau.roe.ac.uk/sss/cgi-bin/pixel.cgi"
SQL = "http://ssa.roe.ac.uk:8080/ssa/SSASQL"
# surveyID: name, half field (deg), limit, SSS waveband letter, MPC station
SURVEYS = {
    1: ("SERC-J/EJ", 3.0, 23.0, "J", "260"), 2: ("SERC-R/AAO-R", 3.0, 22.0, "R", "260"),
    3: ("SERC-I", 3.0, 19.0, "I", "260"), 4: ("ESO-R", 2.5, 22.0, "E", "809"),
    5: ("POSS-I E (N)", 3.0, 20.0, "P", "675"), 6: ("POSS-II B", 3.0, 22.5, "J", "675"),
    7: ("POSS-II R", 3.0, 20.8, "R", "675"), 8: ("POSS-II I", 3.0, 19.5, "I", "675"),
    9: ("POSS-I E (S)", 3.0, 20.0, "P", "675"),
}


def load_plates():
    """Plate list: cached refresh if present, else the bundled copy."""
    p = net.app_dir() / "ssa_plates.json"
    if p.exists():
        return json.loads(p.read_text())
    return json.loads(resources.files("spacerocksuite.data").joinpath("ssa_plates.json").read_text())


def refresh_plates() -> int:
    """Download the SSA Plate table again (a few seconds)."""
    from astropy.coordinates import SkyCoord, FK4
    import astropy.units as u
    q = ("SELECT surveyID,plateNum,emulsion,filterID,raPnt,decPnt,mjd,expLength FROM Plate "
         "WHERE surveyID > 0")
    page = net.fetch(SQL, data={"action": "freeform", "server": "amachine", "format": "CSV",
                                "compress": "NONE", "rows": "1", "sqlstmt": q}, timeout=600).decode("latin1")
    m = re.search(r'href="([^"]+\.csv)"', page, re.I)
    if not m:
        raise ConnectionError("SSA did not return a CSV link")
    rows = list(csv.DictReader(io.StringIO(net.fetch(m.group(1), timeout=600).decode("latin1"))))
    c = SkyCoord([float(r["raPnt"]) for r in rows] * u.deg, [float(r["decPnt"]) for r in rows] * u.deg,
                 frame=FK4(equinox="B1950")).icrs
    out = [[int(r["surveyID"]), r["plateNum"].strip(), r["emulsion"].strip(), r["filterID"].strip(),
            round(c.ra.deg[i], 4), round(c.dec.deg[i], 4), float(r["mjd"]), float(r["expLength"])]
           for i, r in enumerate(rows)]
    d = {"source": f"SSA Plate table, fetched {time.strftime('%Y-%m-%d')}",
         "columns": ["survey", "plate", "emulsion", "filter", "ra_icrs", "dec_icrs", "mjd_ssa", "exp_min"],
         "rows": out}
    (net.app_dir() / "ssa_plates.json").write_text(json.dumps(d, separators=(",", ":")))
    return len(out)


def cutout(ra, dec, survey_id: int, size_arcmin=10.0):
    """FITS bytes of an SSS cutout in the waveband of `survey_id`.

    The pixel server picks its own plate per band; check PLATENUM in the header.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    coords = SkyCoord(ra * u.deg, dec * u.deg).to_string("hmsdms", sep=" ", precision=2)
    data = {"coords": coords, "equinox": 2, "size": size_arcmin, "waveband": SURVEYS[survey_id][3], "gif": 0}
    page = net.fetch(PIXEL, data=data, timeout=300, cache_hours=24 * 30).decode("latin1")
    m = re.search(r"href=(http\S+?\.fits\.gz)", page)
    if not m:
        raise ConnectionError("SSS pixel server returned no FITS")
    return gzip.decompress(net.fetch(m.group(1), timeout=300, cache_hours=24 * 30))


class SuperCOSMOS(Archive):
    name = "SuperCOSMOS (SSA)"
    description = "UKST, ESO Schmidt, POSS-I and POSS-II plates (offline plate list)"
    years = (1949, 2003)

    def __init__(self, surveys=tuple(SURVEYS), date_fix_days: float = 1.0):
        self.surveys = set(surveys)
        self.date_fix = date_fix_days

    def search(self, track, nsigma, jd0=None, jd1=None, progress=None, stop=None):
        P = load_plates()["rows"]
        sid = np.array([r[0] for r in P])
        ra_c = np.array([r[4] for r in P])
        dec_c = np.array([r[5] for r in P])
        jd = np.array([r[6] for r in P]) + 2400000.5 + self.date_fix
        lo = max(jd0 or track.jd[0], track.jd[0])
        hi = min(jd1 or track.jd[-1], track.jd[-1])
        m = np.isin(sid, list(self.surveys)) & (jd >= lo) & (jd <= hi)
        idx = np.flatnonzero(m)
        hits = []
        if idx.size == 0:
            return hits
        ra, dec, v, *_ = track.at(jd[idx])
        a, b, th = track.ellipse(jd[idx], nsigma)
        dx, dy = offset_arcsec(ra, dec, ra_c[idx], dec_c[idx])
        for k, i in enumerate(idx):
            name, half_deg, lim, _, site = SURVEYS[sid[i]]
            half = half_deg * 3600
            if not (abs(dx[k]) <= half + a[k] and abs(dy[k]) <= half + a[k]):
                continue          # also drops NaN (plate on the far side of the sky)
            frac = inside_fraction(dx[k], dy[k], a[k], b[k], th[k], half, half)
            if frac <= 0:
                continue
            r = P[i]
            rr, rd = track.rate(jd[i])
            exp_s = r[7] * 60
            hits.append(PlateHit(
                self.name, f"{name} {r[1]}", name, f"{r[2]} {r[3]}", float(jd[i]), exp_s,
                float(ra[k]), float(dec[k]), None if v is None else float(v[k]),
                float(a[k]), float(b[k]), float(th[k]), limit_mag=lim,
                margin_arcsec=float(half - max(abs(dx[k]), abs(dy[k]))), covered_fraction=frac,
                trail_arcsec=float(np.hypot(rr, rd) * exp_s / 3600), url="http://www-wfau.roe.ac.uk/sss/pixel.html",
                note=f"SSA date +{self.date_fix:g} d applied; check with SkyBoT" if self.date_fix else "",
                extra={"survey_id": int(sid[i]), "site": site, "centre": (float(ra_c[i]), float(dec_c[i]))}))
            if progress:
                progress((k + 1) / idx.size)
        hits.sort(key=lambda h: h.mid_jd)
        return hits
