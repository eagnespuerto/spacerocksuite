"""JPL Scout (NEOCP objects): orbit samples -> track with an uncertainty ellipse."""
from __future__ import annotations

import numpy as np

from .. import net
from ..device import PROFILE
from ..track import track_from_elements

API = "https://ssd-api.jpl.nasa.gov/scout.api"


def listing() -> list[dict]:
    """Current Scout objects (one dict per NEOCP object)."""
    d = net.fetch_json(API, cache_hours=0.25)
    return d.get("data", [])


def orbits(tdes: str) -> dict:
    return net.fetch_json(API, {"tdes": tdes, "orbits": "1"}, cache_hours=0.25)


def samples(tdes: str) -> tuple[np.ndarray, np.ndarray]:
    """(elements (S, 6) as q,e,i,node,peri,tp_jd ; H (S,))"""
    d = orbits(tdes)
    o = d.get("orbits") or {}
    f = o.get("fields") or []
    rows = o.get("data") or []
    if not rows:
        raise ValueError(f"Scout has no orbit samples for {tdes!r} (is it still on the NEOCP?)")
    ix = {k: f.index(k) for k in f}

    def col(*names):
        for n in names:
            if n in ix:
                return np.array([float(r[ix[n]]) for r in rows])
        raise KeyError(names)
    el = np.column_stack([col("qr", "q"), col("ec", "e"), col("inc", "in", "i"), col("om", "node"),
                          col("w", "peri"), col("tp")])
    H = col("H") if "H" in ix else np.full(len(rows), np.nan)
    return el, H


def track(tdes: str, jd, max_samples: int | None = None):
    el, H = samples(tdes)
    n = max_samples or PROFILE.mc_samples
    if len(el) > n:  # deterministic thinning
        el = el[np.linspace(0, len(el) - 1, n).round().astype(int)]
    t = track_from_elements(tdes, "scout", el, jd, H=float(np.nanmedian(H)))
    t.meta["n_samples"] = len(el)
    t.meta["elements"] = [float(x) for x in np.median(el, 0)]   # for SSOIS element search
    return t
