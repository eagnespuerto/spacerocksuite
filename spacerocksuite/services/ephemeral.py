"""Ephemeral (https://eagnespuerto.github.io/ephemeral/) integration.

- read any ephemeris Ephemeral reads (Horizons, MPC, Find_Orb, 80-column, tables)
- open a Track in Ephemeral through its share link (#s=..., deflate-raw + base64url)
"""
from __future__ import annotations

import base64
import json
import webbrowser
import zlib

import numpy as np

from ..ephemeral_parse import parse
from ..track import Track

SITE = "https://eagnespuerto.github.io/ephemeral/"


def track_from_text(text: str, name: str | None = None) -> Track:
    d = parse(text)
    rows = d["rows"]
    if not rows:
        raise ValueError(" ".join(d.get("notes", [])) or "No positions found")

    def col(k):
        v = np.array([np.nan if r.get(k) is None else r[k] for r in rows], float)
        return None if np.all(np.isnan(v)) else v
    sra, sdec = col("sig_ra_arcsec"), col("sig_dec_arcsec")
    a = b = th = None
    if sra is not None and sdec is not None:  # axis-aligned ellipse from RA/Dec sigmas
        a = np.maximum(sra, sdec)
        b = np.minimum(sra, sdec)
        th = np.where(sra >= sdec, 90.0, 0.0)
    return Track(name or d.get("name") or "object", "ephemeral", col("jd"), col("ra_deg"), col("dec_deg"),
                 col("mag"), col("delta_au"), a, b, th, meta={"format": d.get("format")})


def _pack(obj) -> str:
    c = zlib.compressobj(9, zlib.DEFLATED, -15)          # raw deflate, like CompressionStream('deflate-raw')
    raw = c.compress(json.dumps(obj, separators=(",", ":")).encode()) + c.flush()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def unpack(s: str):
    pad = "=" * (-len(s) % 4)
    return json.loads(zlib.decompress(base64.urlsafe_b64decode(s + pad), -15))


def share_link(track: Track, max_rows: int = 400, color: str = "#ffb347") -> str:
    """A link that opens Ephemeral with this track as a simple time/RA/Dec/delta table."""
    idx = np.linspace(0, len(track) - 1, min(len(track), max_rows)).round().astype(int)
    lines = [f"# Object: {track.name}"]
    for i in np.unique(idx):
        jd = track.jd[i]
        d = "" if track.delta is None else f", {track.delta[i]:.6f}"
        lines.append(f"{jd:.5f}, {track.ra[i]:.6f}, {track.dec[i]:.6f}{d}")
    t_ms = round((track.jd[len(track) // 2] - 2440587.5) * 86400000)
    data = {"v": 1, "t": t_ms, "o": [{"n": track.name, "c": color, "x": "\n".join(lines),
                                      "d": None, "dt": None, "h": 0}]}
    return SITE + "#s=" + _pack(data)


def open_in_browser(track: Track) -> str:
    url = share_link(track)
    webbrowser.open(url)
    return url
