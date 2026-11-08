"""Minor Planet Center: observations (for Find_Orb) and the MPC orbit."""
from __future__ import annotations

import json

from .. import net

OBS_API = "https://data.minorplanetcenter.net/api/get-obs"


def observations(desig: str, fmt: str = "OBS80") -> str:
    """All published observations of an object, as MPC 80-column text (or ADES_DF)."""
    body = json.dumps({"desigs": [desig], "output_format": [fmt]}).encode()
    raw = net.fetch(OBS_API, data=body, headers={"Content-Type": "application/json"},
                    method="GET", timeout=180, cache_hours=6)
    d = json.loads(raw)
    if isinstance(d, list) and d:
        return d[0].get(fmt, "") if isinstance(d[0], dict) else ""
    if isinstance(d, dict) and fmt in d:
        return d[fmt]
    raise ValueError(f"MPC returned no observations for {desig!r}")


NEOCP_OBS = "https://minorplanetcenter.net/cgi-bin/showobsorbs.cgi"


def neocp_observations(tdes: str) -> str:
    """80-column astrometry of an object on the NEO Confirmation Page."""
    import re
    page = net.fetch(NEOCP_OBS, {"Obj": tdes, "obs": "y"}, timeout=60, cache_hours=0.25).decode("utf-8", "replace")
    lines = [l for l in re.sub(r"<[^>]+>", "", page).splitlines() if len(l.rstrip()) >= 70]
    if not lines:
        raise ValueError(f"No NEOCP observations for {tdes!r}")
    return "\n".join(lines) + "\n"
