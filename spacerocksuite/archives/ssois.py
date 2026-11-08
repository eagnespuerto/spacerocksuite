"""CADC SSOIS: images from many telescopes (ESO, CFHT, NOIRLab, Subaru, HST, ...)
that contain a moving object. Searched by name (Horizons orbit) or by an arc of
observations; the search box is the track's n-sigma ellipse."""
from __future__ import annotations

import html

import numpy as np

from .. import net
from ..track import sep_deg
from .base import Archive, PlateHit, tsv_rows

URL = "https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/cadcbin/ssos/ssosclf.pl"
MAX_BOX = 5 * 3600.0          # arcsec; beyond this the search says nothing useful


def _date(jd):
    from astropy.time import Time
    return Time(jd, format="jd").strftime("%Y %m %d")


class SSOIS(Archive):
    name = "SSOIS (CADC)"
    description = "Images from ~40 telescopes, 1990-now, via CADC's moving-object search"
    years = (1990, 2100)

    def __init__(self, obs80: str | None = None, telinst: list[str] | None = None, elements=None):
        self.obs80 = obs80          # arc search (MPC fitter) when there is no designation
        self.elements = elements    # (q, e, i, node, peri, tp_jd): orbital-element search
        self.telinst = telinst or []
        self.messages = []

    def _query(self, track, q):
        """Pick the SSOIS search type. GET only: the server rejects these as POST."""
        import math
        if self.elements is not None:
            qq, e, i, om, w, tp = (float(x) for x in self.elements)
            if e >= 0.99:
                raise ValueError("SSOIS element search needs an elliptic orbit (e < 0.99); "
                                 "use the plate archives for this object")
            a = qq / (1 - e)
            ep = float(np.round(track.jd[len(track) // 2], 1))
            n = math.degrees(math.sqrt(2.959122082855911e-4 / a ** 3))
            q.update(search="orbel", epoch=f"{ep:.1f}", a=f"{a:.8f}", e=f"{e:.8f}", i=f"{i:.6f}",
                     longi=f"{om:.6f}", peri=f"{w:.6f}", anom=f"{(n * (ep - tp)) % 360:.6f}")
        elif self.obs80:
            lines = [l for l in self.obs80.splitlines() if l.strip()]
            idx = np.unique(np.linspace(0, len(lines) - 1, min(25, len(lines))).round().astype(int))
            q.update(search="mpc", obs="\n".join(lines[k] for k in idx))
        else:
            q.update(search="bynameHorizons", object=track.name)
        text = net.fetch(URL, q, timeout=600, cache_hours=24).decode("utf-8", "replace")
        if text.lstrip().startswith(("There was an error", "An error")):
            raise ConnectionError(text.strip().splitlines()[0])
        return text

    def search(self, track, nsigma, jd0=None, jd1=None, progress=None, stop=None):
        jd0 = max(jd0 or track.jd[0], track.jd[0])
        jd1 = min(jd1 or track.jd[-1], track.jd[-1])
        hits = []
        # SSOIS wants one box size; query year by year so the box follows the ellipse
        edges = np.arange(jd0, jd1 + 365.25, 365.25)
        for k, (a, b) in enumerate(zip(edges[:-1], np.minimum(edges[1:], jd1))):
            if stop is not None and stop.is_set():
                break
            if b <= a:
                continue
            m = (track.jd >= a) & (track.jd <= b)
            jj = track.jd[m] if m.any() else np.array([a, b])
            ea, eb, _ = track.ellipse(jj, nsigma)
            box = float(np.nanmax(ea))
            if box > MAX_BOX:
                self.messages.append(f"{_date(a)[:4]}: {nsigma:g}-sigma ellipse is {box / 3600:.0f} deg; "
                                     f"too uncertain to search")
                continue
            q = dict(lang="en", epoch1=_date(a), epoch2=_date(b), eellipse=f"{box:.1f}",
                     eunits="arcseconds", extres="no", xyres="no", format="tsv")
            if self.telinst:
                q["telinst"] = self.telinst
            text = self._query(track, q)
            for r in tsv_rows(text):
                try:
                    mjd = float(r["MJD"])
                except (KeyError, ValueError):
                    continue
                jd = mjd + 2400000.5
                ra, dec, v, *_ = track.at(jd)
                a_, b_, th = track.ellipse(jd, nsigma)
                off = float(sep_deg(ra, dec, float(r["Object_RA"]), float(r["Object_Dec"]))) * 3600
                exp = float(r.get("Exptime") or 0)
                rr, rd = track.rate(jd)
                hits.append(PlateHit(
                    self.name, r["Image"], r.get("Telescope/Instrument", ""), r.get("Filter", ""), jd, exp,
                    float(ra), float(dec), None if v is None else float(v), float(a_), float(b_), float(th),
                    trail_arcsec=float(np.hypot(rr, rd) * exp / 3600),
                    url=html.unescape(r.get("Datalink", "")),
                    note=f"SSOIS prediction {off:.1f}\" from ours" if off > 2 else "",
                    extra={"target": r.get("Image_target", "")}))
            if progress:
                progress((k + 1) / max(1, len(edges) - 1))
        # SSOIS lists one exposure several times (raw + reduced products, extensions):
        # keep one row per telescope and mid-time, preferring the raw frame
        hits.sort(key=lambda h: ("_ori" not in h.plate_id and "raw" not in h.plate_id.lower(), h.plate_id))
        seen, out = set(), []
        for h in hits:
            key = (h.telescope, round(h.mid_jd, 5))
            if key not in seen:
                seen.add(key)
                out.append(h)
        out.sort(key=lambda h: h.mid_jd)
        return out
