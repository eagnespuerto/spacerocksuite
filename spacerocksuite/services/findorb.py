"""Find_Orb (Bill Gray, projectpluto.com): orbit fit + ephemeris from astrometry.

Uses a local console Find_Orb (`fo`, `fo64.exe`) when one is set in Settings
or found on PATH, else the public web service. Both print the same table:

    Date (UTC)   RA             Dec           delta  r     elong  mag " sig PA
    1982 03 18  13 44 21.325   -36 39 34.59  41.268 41.964 133.9 19.4 .176 120

The sigma is 1-sigma along the major axis. Find_Orb does not print the minor
axis, so the track uses a circle of that radius (conservative).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime, timedelta

import numpy as np

from .. import net
from ..ephemeral_parse import parse_time, parse_ra, parse_dec
from ..track import Track

WEB = "https://www.projectpluto.com/cgi-bin/fo/fo_serve.cgi"
MAX_STEPS = 1000

_ROW = re.compile(
    r"^\s*(\d{4} \d\d \d\d(?:\.\d+)?(?: \d\d:\d\d(?::\d\d)?)?)\s+"
    r"(\d\d \d\d \d\d\.\d+)\s+([-+]?\d\d \d\d \d\d\.\d+)\s+(.*)$")


def local_binary(configured: str | None = None) -> str | None:
    for c in filter(None, [configured, os.environ.get("SRS_FINDORB")]):
        if os.path.isfile(c):
            return c
    for name in ("fo64", "fo", "fo32"):
        p = shutil.which(name)
        if p:
            return p
    return None


def parse_table(text: str) -> dict:
    """Rows from a Find_Orb ephemeris table -> dict of arrays."""
    hdr = next((l.strip() for l in text.splitlines() if l.lstrip().startswith("Date") and "RA" in l), "")
    has_sig = "sig" in hdr.split()
    has_mag = " mag" in hdr
    out = {k: [] for k in ("jd", "ra", "dec", "delta", "mag", "sig", "pa")}
    for line in text.splitlines():
        m = _ROW.match(line)
        if not m:
            continue
        date = m.group(1)
        if ":" in date:  # "1982 03 18 06:00" -> fractional day for parse_time
            d, hm = date.rsplit(" ", 1)
            p = [float(x) for x in hm.split(":")] + [0, 0]
            y, mo, dd = d.split()
            date = f"{y} {mo} {float(dd) + (p[0] + p[1] / 60 + p[2] / 3600) / 24:.6f}"
        rest = m.group(4).split()
        out["jd"].append(parse_time(date))
        out["ra"].append(parse_ra(m.group(2)))
        out["dec"].append(parse_dec(m.group(3)))
        out["delta"].append(_num(rest[0]) if rest else np.nan)
        if has_sig and len(rest) >= 2:
            out["sig"].append(_sig(rest[-2]))
            out["pa"].append(_num(rest[-1]))
            out["mag"].append(_num(rest[-3]) if has_mag and len(rest) >= 3 else np.nan)
        else:
            out["sig"].append(np.nan)
            out["pa"].append(np.nan)
            out["mag"].append(_num(rest[-1]) if has_mag and rest else np.nan)
    if not out["jd"]:
        raise ValueError("Find_Orb returned no ephemeris rows")
    return {k: np.array(v, float) for k, v in out.items()}


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def _sig(s):
    """Find_Orb sigma tokens: '.176', '12.3', '1.2'' (arcmin), '3.4d', '15"'."""
    s = s.strip()
    mult = 1.0
    if s.endswith("'"):
        mult, s = 60.0, s[:-1]
    elif s.endswith("d") or s.endswith("°"):
        mult, s = 3600.0, s[:-1]
    elif s.endswith('"'):
        s = s[:-1]
    return _num(s) * mult


def _step_str(step_days: float) -> str:
    if step_days >= 1:
        return f"{step_days:g}"
    if step_days >= 1 / 24:
        return f"{step_days * 24:g}h"
    return f"{step_days * 1440:g}m"


def _web(obs: str, start: str, n: int, step: str, site: str) -> str:
    boundary = uuid.uuid4().hex
    fields = dict(year=start, n_steps=str(n), stepsize=step, mpc_code=site, faint_limit="99",
                  ephem_type="0", sigmas="on", element_center="-2", epoch="default", resids="0",
                  language="e", file_no="0")
    parts = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n')
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="upfile"; filename="obs.txt"\r\n'
                 f"Content-Type: text/plain\r\n\r\n{obs}\r\n--{boundary}--\r\n")
    body = "".join(parts).encode("utf-8")
    html = net.fetch(WEB, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                     timeout=600, cache_hours=6).decode("utf-8", "replace")
    import html as h
    return h.unescape(re.sub(r"<[^>]+>", "", html))


def _local(binary: str, obs: str, start: str, n: int, step: str, site: str) -> str:
    with tempfile.TemporaryDirectory() as d:
        of = os.path.join(d, "obs.txt")
        ef = os.path.join(d, "eph.txt")
        with open(of, "w", encoding="utf-8") as f:
            f.write(obs)
        cmd = [binary, of, "-e", ef, "-C", site, f"EPHEM_START={start}", f"EPHEM_STEPS={n}",
               f"EPHEM_STEP_SIZE={step}", "-E", "16"]
        subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=1800, check=False)
        if not os.path.exists(ef):
            raise RuntimeError("local Find_Orb wrote no ephemeris; check that console Find_Orb is set up")
        with open(ef, encoding="utf-8", errors="replace") as f:
            return f.read()


def ephemeris(obs: str, start_jd: float, stop_jd: float, step_days: float = 1.0, site: str = "500",
              name: str = "object", binary: str | None = None, progress=None) -> Track:
    """Fit `obs` (MPC 80-col or ADES) with Find_Orb and return its ephemeris as a Track."""
    exe = local_binary(binary)
    total = int(np.floor((stop_jd - start_jd) / step_days)) + 1
    chunks = []
    done = 0
    t = start_jd
    while done < total:
        n = min(MAX_STEPS, total - done)
        start = _jd_to_str(t)
        text = _local(exe, obs, start, n, _step_str(step_days), site) if exe else \
            _web(obs, start, n, _step_str(step_days), site)
        chunks.append(parse_table(text))
        done += n
        t += n * step_days
        if progress:
            progress(done / total)
    cat = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    sig = cat["sig"]
    sig = None if np.all(np.isnan(sig)) else sig
    return Track(name, "findorb", cat["jd"], cat["ra"], cat["dec"],
                 None if np.all(np.isnan(cat["mag"])) else cat["mag"],
                 None if np.all(np.isnan(cat["delta"])) else cat["delta"],
                 sig, sig, cat["pa"] if sig is not None else None, site=site,
                 meta={"engine": "local" if exe else "web"})


def _jd_to_str(jd: float) -> str:
    dt = datetime(2000, 1, 1, 12) + timedelta(days=jd - 2451545.0)
    return dt.strftime("%Y-%m-%d %H:%M")
