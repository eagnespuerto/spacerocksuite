"""
Ephemeris parser – a pure‑Python port of Ephemeral's JavaScript parser.

Supported formats
-----------------
* JPL Horizons observer tables (text or CSV, with optional $$SOE/$$EOE markers)
* Minor Planet Center (MPC) Ephemeris Service output
* Find_Orb ephemeris output
* MPC 80‑column astrometry records
* Simple whitespace / CSV tables containing at least time, RA and Dec
  (optional delta, magnitude, uncertainties)

Public API
---------
* parse(text) → dict
* parse_time(s) → jd (float)
* parse_ra(s) → deg (float)
* parse_dec(s) → deg (float)

The returned dictionary contains:
    name            – target name (or None)
    site            – site description (or None)
    topocentric     – True if a site is known
    disc_code       – discovery site code (or None)
    disc_time_jd    – discovery time JD (or None)
    rows            – list of dicts, each with keys:
        jd                – Julian Date (float)
        ra_deg            – Right‑ascension in degrees (float)
        dec_deg           – Declination in degrees (float)
        delta_au          – Geocentric distance in AU (float|None)
        mag               – Magnitude (float|None)
        sig_ra_arcsec     – 1‑σ RA uncertainty in arcseconds (float|None)
        sig_dec_arcsec    – 1‑σ Dec uncertainty in arcseconds (float|None)
    notes           – list of informational strings
"""

import re
import math
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

# ----------------------------------------------------------------------
# Helper conversion utilities
# ----------------------------------------------------------------------
def _jd_from_timestamp(ts: float) -> float:
    """Convert POSIX timestamp (seconds since 1970‑01‑01 UTC) to JD."""
    return ts / 86400.0 + 2440587.5

def _timestamp_from_jd(jd: float) -> float:
    """Convert JD to POSIX timestamp (seconds since 1970‑01‑01 UTC)."""
    return (jd - 2440587.5) * 86400.0

def _utc(y: int, mo: int, d: int, h: int = 0, mi: int = 0, s: float = 0.0) -> float:
    """Return JD for a UTC date/time."""
    dt = datetime(y, mo + 1, d, h, mi, int(s), int(round((s - int(s)) * 1e6)), tzinfo=timezone.utc)
    return _jd_from_timestamp(dt.timestamp())

# ----------------------------------------------------------------------
# Core tokenizers (mirrors the JS takeDate / takeAngle)
# ----------------------------------------------------------------------
_MONTHS = {
    'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5,
    'jul': 6, 'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11,
}

def _take_date(s: str) -> Optional[Tuple[float, str]]:
    """Parse a date/time at the start of *s*.
    Returns (jd, rest_of_string) or None.
    """
    s = s.lstrip()
    # 1. Horizons style: 2024-Dec-20 00:00[:00[.000]]
    m = re.match(r'^(\d{4})-([A-Za-z]{3})-(\d{1,2})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}(?:\.\d+)?))?)?', s)
    if m:
        mo = _MONTHS.get(m.group(2).lower())
        if mo is None:
            return None
        jd = _utc(int(m.group(1)), mo, int(m.group(3)),
                  int(m.group(4) or 0), int(m.group(5) or 0), float(m.group(6) or 0))
        return jd, s[len(m.group(0)) :]

    # 2. ISO 8601
    m = re.match(r'^(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}(?:\.\d+)?))?)?Z?', s)
    if m:
        jd = _utc(int(m.group(1)), int(m.group(2)) - 1, int(m.group(3)),
                 int(m.group(4) or 0), int(m.group(5) or 0), float(m.group(6) or 0))
        return jd, s[len(m.group(0)) :]

    # 3. Julian Date (plain number)
    m = re.match(r'^(2\d{6}(?:\.\d+)?)(?=[\s,]|$)', s)
    if m:
        jd = float(m.group(1))
        return jd, s[len(m.group(0)) :]

    # 4. MPC style: 2024 12 20.12345  or 2024/12/20 00:00
    m = re.match(r'^(\d{4})[ /](\d{1,2})[ /](\d{1,2}(?:\.\d+)?)/?', s)
    if m:
        y, mo, d_str = int(m.group(1)), int(m.group(2)) - 1, m.group(3)
        rest = s[len(m.group(0)) :]
        if '.' in d_str:
            d = float(d_str)
            day_int = int(math.floor(d))
            frac = d - day_int
            jd = _utc(y, mo, day_int) + frac
            return jd, rest
        # No fractional day – look for optional time components
        h = mi = sec = 0
        # HH:MM[:SS]
        m2 = re.match(r'^\s+(\d{1,2}):(\d{2})(?::(\d{2}(?:\.\d+)?))?', rest)
        if m2:
            h, mi, sec = int(m2.group(1)), int(m2.group(2)), float(m2.group(3) or 0)
            rest = rest[len(m2.group(0)) :]
        else:
            # HHMMSS
            m2 = re.match(r'^\s+(\d{2})(\d{2})(\d{2})(?=\s)', rest)
            if m2:
                h, mi, sec = map(int, m2.groups())
                rest = rest[len(m2.group(0)) :]
            else:
                # HHMM
                m2 = re.match(r'^\s+(\d{2})(\d{2})(?=\s)', rest)
                if m2:
                    h, mi = map(int, m2.groups())
                    rest = rest[len(m2.group(0)) :]
                else:
                    # HH (stand‑alone hour)
                    m2 = re.match(r'^\s+(\d{2})(?=\s+\d{1,2}[ :]\d{2}[ :])', rest)
                    if m2:
                        h = int(m2.group(1))
                        rest = rest[len(m2.group(0)) :]
        jd = _utc(y, mo, int(d_str), h, mi, sec)
        return jd, rest

    return None

def _take_angle(s: str, is_ra: bool) -> Optional[Tuple[float, str]]:
    """Parse an angle from the start of *s*.
    Returns (degrees, rest_of_string) or None.
    """
    s = s.lstrip(' ,')
    # 1. Sexagesimal with delimiters (h m s or d m s)
    m = re.match(r'^([+-]?)(\d{1,3})[ :hd°](\s?\d{1,2})[ :m\'](\s?\d{1,2}(?:\.\d*)?)[s"]?', s)
    if m:
        sign = -1 if m.group(1) == '-' else 1
        v = int(m.group(2)) + int(m.group(3)) / 60.0 + float(m.group(4)) / 3600.0
        if is_ra:
            v *= 15.0
        return sign * v, s[len(m.group(0)) :]

    # 2. Degrees + minutes (dd mm.m)
    m = re.match(r'^([+-]?)(\d{1,3})[ :](\d{1,2}(?:\.\d+)?)(?=[\s,])', s)
    if m:
        sign = -1 if m.group(1) == '-' else 1
        v = int(m.group(2)) + float(m.group(3)) / 60.0
        if is_ra:
            v *= 15.0
        return sign * v, s[len(m.group(0)) :]

    # 3. Simple decimal number
    m = re.match(r'^([+-]?\d+(?:\.\d+)?)(?=[\s,]|$)', s)
    if m:
        return float(m.group(1)), s[len(m.group(0)) :]

    return None

def _numbers(s: str) -> List[float]:
    """Extract all signed/unsigned numbers from *s*."""
    return [float(x) for x in re.findall(r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?', s)]

def _clean_name(n: Optional[str]) -> str:
    if not n:
        return ''
    n = ' '.join(n.strip().split())
    m = re.match(r'^\((.+)\)$', n)
    return m.group(1) if m else n

def _num(x: str) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None

def _token_ending_near(line: str, end: int) -> Optional[float]:
    """Return the numeric token whose *right* edge is closest to *end*."""
    if end < 0:
        return None
    best = None
    best_dist = 1e9
    for m in re.finditer(r'\S+', line):
        token_end = m.start() + len(m.group(0))
        dist = abs(token_end - end)
        if dist < best_dist:
            best_dist = dist
            best = m.group(0)
    return _num(best) if best_dist <= 6 else None

# ----------------------------------------------------------------------
# Format‑specific parsers
# ----------------------------------------------------------------------
def _parse_horizons(text: str, out: Dict[str, Any]) -> bool:
    lines = text.splitlines()
    try:
        soe = next(i for i, l in enumerate(lines) if l.startswith('$$SOE'))
        eoe = next((i for i, l in enumerate(lines) if l.startswith('$$EOE')), -1)
    except StopIteration:
        return False

    out['format'] = 'JPL Horizons'
    # Target name
    m = re.search(r'Target body name:\s*(.+?)(?:\s{2,}|\{|$)', text, re.M)
    if m:
        out['name'] = _clean_name(m.group(1))

    # Site information (geodetic)
    m = re.search(r'Center geodetic\s*:\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)', text)
    if m:
        lon, lat, hkm = map(float, m.groups())
        if hkm > -6000:
            sn = re.search(r'Center-site name:\s*(.+)', text)
            site_name = sn.group(1).strip() if sn else 'Ephemeris site'
            # Store a simple string representation; the public API expects a string.
            out['site'] = f"{site_name} (lon={lon:.3f}, lat={lat:.3f}, h={hkm*1000:.0f}m)"

    # Determine CSV vs fixed‑width
    body = [l for l in lines[soe + 1 : eoe if eoe >= 0 else None] if l.strip()]
    if not body:
        return False
    is_csv = len(body[0].split(',')) > 3

    # ------------------------------------------------------------------
    # CSV version
    # ------------------------------------------------------------------
    if is_csv:
        hdr = next((lines[i] for i in range(soe - 1, -1, -1)
                    if re.search(r'Date', lines[i]) and ',' in lines[i]), '')
        cols = [c.strip() for c in hdr.split(',')]
        def col_index(regex):
            for i, c in enumerate(cols):
                if re.search(regex, c, re.I):
                    return i
            return -1

        i_ra = col_index(r'^R\.A\.')
        i_dec = col_index(r'^DEC(?!_3)')
        i_delta = col_index(r'^delta$')
        i_mag = col_index(r'^(APmag|T[- ]?mag|N[- ]?mag)$')
        i_ra3 = col_index(r'RA_3sigma')
        i_dec3 = col_index(r'DEC_3sigma')

        for line in body:
            parts = [p.strip() for p in line.split(',')]
            d = _take_date(parts[0])
            if not d:
                continue
            jd, _ = d
            ra = _take_angle(parts[i_ra], True) if i_ra >= 0 else None
            dec = _take_angle(parts[i_dec], False) if i_dec >= 0 else None
            if not ra or not dec:
                continue
            row = {
                'jd': jd,
                'ra_deg': ra[0],
                'dec_deg': dec[0],
                'delta_au': _num(parts[i_delta]) if i_delta >= 0 else None,
                'mag': _num(parts[i_mag]) if i_mag >= 0 else None,
                'sig_ra_arcsec': _num(parts[i_ra3]) / 3.0 if i_ra3 >= 0 and parts[i_ra3] else None,
                'sig_dec_arcsec': _num(parts[i_dec3]) / 3.0 if i_dec3 >= 0 and parts[i_dec3] else None,
            }
            out['rows'].append(row)
        return True

    # ------------------------------------------------------------------
    # Fixed‑width version
    # ------------------------------------------------------------------
    # Find the header line (the last line before $$SOE that contains the word "Date")
    header_line = ''
    for i in range(soe - 1, -1, -1):
        if re.search(r'Date', lines[i], re.I):
            header_line = lines[i]
            break

    def end_of(label: str) -> int:
        idx = header_line.find(label)
        return idx + len(label) if idx >= 0 else -1

    delta_end = end_of(' delta ') if end_of(' delta ') > 0 else end_of(' delta')
    mag_end = max(end_of('APmag'), end_of('T-mag'))
    ra3_end = end_of('RA_3sigma')
    dec3_end = end_of('DEC_3sigma')

    for line in body:
        d = _take_date(line)
        if not d:
            continue
        jd, rest = d
        # Strip possible flag characters that appear between date and RA
        rest = re.sub(r'^\s*[*CNAm]{0,3}\s', ' ', rest)
        ra = _take_angle(rest, True)
        if not ra:
            continue
        dec = _take_angle(ra[1], False)
        if not dec:
            continue
        row = {
            'jd': jd,
            'ra_deg': ra[0],
            'dec_deg': dec[0],
            'delta_au': _token_ending_near(line, delta_end),
            'mag': _token_ending_near(line, mag_end),
            'sig_ra_arcsec': (_token_ending_near(line, ra3_end) / 3.0) if ra3_end >= 0 else None,
            'sig_dec_arcsec': (_token_ending_near(line, dec3_end) / 3.0) if dec3_end >= 0 else None,
        }
        out['rows'].append(row)

    return True

def _parse_obs80(text: str, out: Dict[str, Any]) -> bool:
    lines = [l for l in text.splitlines() if len(l) >= 80 and re.match(r'^.{14}.\d{4} \d\d \d\d\.\d+', l[:32])]
    if not lines:
        return False

    out['format'] = 'MPC 80-column astrometry'
    for l in lines:
        # Date field (positions 15‑32, 0‑based indexing)
        d = _take_date(l[15:32])
        if not d:
            continue
        jd, _ = d
        ra = _take_angle(l[32:44], True)
        dec = _take_angle(l[44:56], False)
        if not ra or not dec:
            continue
        site = l[77:80].strip()
        mag = _num(l[65:70])
        row = {
            'jd': jd,
            'ra_deg': ra[0],
            'dec_deg': dec[0],
            'delta_au': None,
            'mag': mag,
            'sig_ra_arcsec': None,
            'sig_dec_arcsec': None,
            'site': site,
        }
        if l[12] == '*' and out.get('disc_code') is None:
            out['disc_code'] = site
            out['disc_time_jd'] = jd
        out['rows'].append(row)

    # Object designation (first 12 columns)
    desig = lines[0][:12].strip()
    if desig:
        out['name'] = desig
    return True

def _parse_generic(text: str, out: Dict[str, Any]) -> bool:
    lines = text.splitlines()
    header: Optional[List[str]] = None
    is_csv = False

    # Optional name line
    m = re.search(r'^\s*(?:#\s*)?(?:Object|Target|Name)\s*[:=]\s*(.+)$', text, re.I | re.M)
    if m:
        out['name'] = _clean_name(m.group(1))

    for raw in lines:
        # Remove full‑line comments
        l = re.sub(r'^\s*#.*$', '', raw).strip()
        if not l:
            continue

        # Detect header line
        if not header and re.search(r'\b(ra|r\.a\.)\b', l, re.I) and re.search(r'\bdec\b', l, re.I):
            is_csv = ',' in l
            sep = ',' if is_csv else r'\s{2,}|\t'
            header = [c.strip().lower() for c in re.split(sep, l)]
            continue

        # CSV data rows (when a header exists)
        if header and is_csv:
            cols = [c.strip() for c in l.split(',')]
            idx = lambda pat: next((i for i, h in enumerate(header) if re.search(pat, h, re.I)), -1)

            i_time = max(idx(r'^(time|date|utc|jd|epoch|datetime)'), 0)
            i_ra = idx(r'^(ra|r\.a\.)')
            i_dec = idx(r'^dec')
            i_delta = idx(r'^(delta|dist|range|geo)')
            i_mag = idx(r'^(mag|v$|vmag)')
            i_ra3 = idx(r'ra_3sigma')
            i_dec3 = idx(r'dec_3sigma')

            d = _take_date(cols[i_time]) if i_time >= 0 else None
            if not d:
                continue
            jd, _ = d
            ra = _take_angle(cols[i_ra], not re.search(r'deg', header[i_ra])) if i_ra >= 0 else None
            dec = _take_angle(cols[i_dec], False) if i_dec >= 0 else None
            if not ra or not dec:
                continue
            row = {
                'jd': jd,
                'ra_deg': ra[0],
                'dec_deg': dec[0],
                'delta_au': _num(cols[i_delta]) if i_delta >= 0 else None,
                'mag': _num(cols[i_mag]) if i_mag >= 0 else None,
                'sig_ra_arcsec': _num(cols[i_ra3]) / 3.0 if i_ra3 >= 0 and cols[i_ra3] else None,
                'sig_dec_arcsec': _num(cols[i_dec3]) / 3.0 if i_dec3 >= 0 and cols[i_dec3] else None,
            }
            out['rows'].append(row)
            continue

        # Free‑form line (no header)
        d = _take_date(l)
        if not d:
            continue
        jd, rest = d
        ra = _take_angle(rest, True)
        if not ra:
            continue
        dec = _take_angle(ra[1], False)
        if not dec:
            continue

        # The first number after Dec is delta in MPC, Find_Orb and simple tables.
        # Other columns differ between services, so they are not guessed here.
        nums = _numbers(dec[1])
        delta = nums[0] if nums and 0 < nums[0] < 2000 else None
        sigma = None
        mag = None

        row = {
            'jd': jd,
            'ra_deg': ra[0],
            'dec_deg': dec[0],
            'delta_au': delta,
            'mag': mag,
            'sig_ra_arcsec': sigma,
            'sig_dec_arcsec': sigma,
        }
        out['rows'].append(row)

    if out['rows']:
        out['format'] = 'Table with header' if header else 'MPC / Find_Orb style table'
    return bool(out['rows'])

# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def parse_time(s: str) -> float:
    """Parse *s* as a date/time and return the Julian Date."""
    d = _take_date(s)
    if not d:
        raise ValueError(f"Unable to parse time from '{s}'")
    return d[0]

def parse_ra(s: str) -> float:
    """Parse *s* as a right‑ascension and return degrees."""
    a = _take_angle(s, True)
    if not a:
        raise ValueError(f"Unable to parse RA from '{s}'")
    return a[0] % 360.0

def parse_dec(s: str) -> float:
    """Parse *s* as a declination and return degrees."""
    a = _take_angle(s, False)
    if not a:
        raise ValueError(f"Unable to parse Dec from '{s}'")
    return a[0]

def parse(text: str) -> Dict[str, Any]:
    """
    Parse an ephemeris text block.

    Returns a dictionary with keys:
        name, site, topocentric, disc_code, disc_time_jd, rows, notes
    """
    out: Dict[str, Any] = {
        'name': None,
        'site': None,
        'topocentric': False,
        'disc_code': None,
        'disc_time_jd': None,
        'rows': [],
        'notes': [],
    }

    if not text or not text.strip():
        out['notes'].append('Empty input.')
        return out

    # Try each format in order of specificity
    if not _parse_horizons(text, out):
        if not _parse_obs80(text, out):
            _parse_generic(text, out)

    # Normalise rows
    cleaned_rows = []
    for r in out['rows']:
        if not (math.isfinite(r['jd']) and math.isfinite(r['ra_deg']) and math.isfinite(r['dec_deg'])):
            continue
        if not (-90.0 <= r['dec_deg'] <= 90.0):
            continue
        r['ra_deg'] = (r['ra_deg'] % 360.0 + 360.0) % 360.0
        cleaned_rows.append(r)

    out['rows'] = sorted(cleaned_rows, key=lambda x: x['jd'])
    out['topocentric'] = bool(out.get('site'))

    # Discovery information (if not already set by 80‑column parser)
    if out.get('disc_code') is None and out.get('discoverySite'):
        out['disc_code'] = out.pop('discoverySite')
        out['disc_time_jd'] = out.pop('discoveryTime')
    else:
        out.pop('discoverySite', None)
        out.pop('discoveryTime', None)

    if not out['rows']:
        out['notes'].append(
            'No positions found. Paste a Horizons observer table, an MPC/Find_Orb ephemeris, '
            '80‑column astrometry, or "time, RA, Dec, delta" lines.'
        )
    elif not any(r.get('delta_au') is not None for r in out['rows']):
        out['notes'].append('No distance (delta) column: Space View arc and parallax need it.')

    return out
