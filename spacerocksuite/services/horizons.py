"""JPL Horizons and SBDB.

Observer ephemerides with the 3-sigma error ellipse (quantities 36 and 37),
converted to 1-sigma for the Track.
"""
from __future__ import annotations

import re

import numpy as np

from .. import net
from ..track import Track

API = "https://ssd.jpl.nasa.gov/api/horizons.api"
SBDB = "https://ssd-api.jpl.nasa.gov/sbdb.api"
MAX_ROWS = 90000


def _command(target: str) -> str:
    t = target.strip()
    if re.fullmatch(r"\d+", t):            # numbered asteroid
        return f"'{t};'"
    if re.fullmatch(r"(\d+)?[PCDXI](/.*)?", t, re.I) or re.match(r"^[PCDXI]/", t):
        return f"'DES={t};CAP;NOFRAG;'"     # comets: current apparition
    return f"'DES={t};'" if re.match(r"^\d{4} ", t) or re.match(r"^[A-Z]\d{2}", t) else f"'{t};'"


def ephemeris(target: str, start: str, stop: str, step: str = "1 d", site: str = "500",
              cache_hours: float = 24) -> Track:
    """Observer ephemeris. start/stop: 'YYYY-MM-DD[ HH:MM]'; step: '1 d', '6 h', '30 m'.

    site is an MPC observatory code (500 = geocentre).
    """
    params = dict(format="text", COMMAND=_command(target), EPHEM_TYPE="OBSERVER",
                  CENTER=f"'{site}@399'", START_TIME=f"'{start}'", STOP_TIME=f"'{stop}'",
                  STEP_SIZE=f"'{step}'", QUANTITIES="'1,9,20,36,37'", CSV_FORMAT="YES",
                  ANG_FORMAT="DEG", CAL_FORMAT="JD", MAKE_EPHEM="YES", OBJ_DATA="YES",
                  EXTRA_PREC="YES")
    text = net.fetch(API, params, timeout=300, cache_hours=cache_hours).decode("utf-8", "replace")
    return parse_observer_csv(text, target, site)


def parse_observer_csv(text: str, target: str = "", site: str = "500") -> Track:
    if "$$SOE" not in text:
        msg = text.strip().splitlines()
        hint = next((l for l in msg if "No matches" in l or "Multiple" in l or "ERROR" in l.upper()), "")
        raise ValueError(f"Horizons returned no ephemeris for {target!r}. {hint}".strip())
    lines = text.splitlines()
    soe = next(i for i, l in enumerate(lines) if l.startswith("$$SOE"))
    hdr = next(lines[i] for i in range(soe - 1, -1, -1) if "Date" in lines[i] and "," in lines[i])
    cols = [c.strip() for c in hdr.split(",")]

    def ix(pat):
        for i, c in enumerate(cols):
            if re.search(pat, c):
                return i
        return -1
    i_ra, i_dec = ix(r"^R\.A\._*\("), ix(r"^DEC_*\(")
    i_mag = ix(r"^(APmag|T-mag|N-mag)$")
    i_delta = ix(r"^delta$")
    i_a, i_b, i_t = ix("SMAA_3sig"), ix("SMIA_3sig"), ix("^Theta$")
    rows = []
    for l in lines[soe + 1:]:
        if l.startswith("$$EOE"):
            break
        p = [x.strip() for x in l.split(",")]
        try:
            rows.append([float(p[0]), float(p[i_ra]), float(p[i_dec]),
                         _f(p, i_mag), _f(p, i_delta), _f(p, i_a), _f(p, i_b), _f(p, i_t)])
        except (ValueError, IndexError):
            continue
    if not rows:
        raise ValueError("Horizons table was empty")
    a = np.array(rows, float)
    name = re.search(r"Target body name:\s*(.+?)\s{2,}", text)
    nm = name.group(1).strip() if name else target
    if nm.startswith("(") and nm.endswith(")"):   # provisional designations come as "(2026 ED138)"
        nm = nm[1:-1]

    def col(k):
        v = a[:, k]
        return None if np.all(np.isnan(v)) else v
    sa, sb = col(5), col(6)
    return Track(nm, "horizons", a[:, 0], a[:, 1], a[:, 2], col(3), col(4),
                 None if sa is None else sa / 3, None if sb is None else sb / 3, col(7), site=site,
                 meta={"orbit": _orbit_line(text)})


def _f(p, i):
    if i < 0 or i >= len(p):
        return np.nan
    try:
        return float(p[i])
    except ValueError:
        return np.nan


def _orbit_line(text):
    m = re.search(r"(soln ref\.=.*?)(?:\n|,)", text)
    return m.group(1).strip() if m else ""


def ephemeris_long(target, start_jd, stop_jd, step_days=1.0, site="500", progress=None) -> Track:
    """Split long spans into Horizons-sized chunks and join them."""
    parts = []
    span = MAX_ROWS * step_days
    t = start_jd
    while t < stop_jd:
        t1 = min(stop_jd, t + span)
        parts.append(ephemeris(target, f"JD{t:.5f}", f"JD{t1:.5f}", f"{max(1, round(step_days * 1440))} m"
                               if step_days < 1 else f"{int(round(step_days))} d", site))
        if progress:
            progress((t1 - start_jd) / (stop_jd - start_jd))
        t = t1
    if len(parts) == 1:
        return parts[0]
    cat = {k: np.concatenate([getattr(p, k) for p in parts]) if getattr(parts[0], k) is not None else None
           for k in ("jd", "ra", "dec", "vmag", "delta", "smaa", "smia", "theta")}
    _, u = np.unique(cat["jd"], return_index=True)
    cat = {k: (v[u] if v is not None else None) for k, v in cat.items()}
    return Track(parts[0].name, "horizons", site=site, meta=parts[0].meta, **cat)


def sbdb(target: str) -> dict:
    """Orbit, H and covariance from the JPL Small-Body Database."""
    return net.fetch_json(SBDB, {"sstr": target, "cov": "mat", "phys-par": "1", "full-prec": "1"},
                          cache_hours=24)


def elements_from_sbdb(d: dict) -> tuple[np.ndarray, float | None]:
    """(q, e, i, node, peri, tp_jd) and H from an SBDB response."""
    el = {e["name"]: float(e["value"]) for e in d["orbit"]["elements"] if e.get("value") is not None}
    H = None
    for p in d.get("phys_par", []):
        if p["name"] == "H":
            H = float(p["value"])
    return np.array([el["q"], el["e"], el["i"], el["om"], el["w"], el["tp"]]), H
