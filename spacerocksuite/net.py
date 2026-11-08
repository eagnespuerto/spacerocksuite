"""HTTP helpers: retries, on-disk cache, and a generic polite User-Agent."""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import __version__

UA = f"SpaceRockSuite/{__version__} (+https://github.com/eagnespuerto/spacerocksuite)"


def app_dir() -> Path:
    base = os.environ.get("SRS_HOME")
    if base:
        p = Path(base)
    elif os.name == "nt":
        p = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "SpaceRockSuite"
    else:
        p = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "spacerocksuite"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir() -> Path:
    p = app_dir() / "cache"
    p.mkdir(exist_ok=True)
    return p


def fetch(url: str, params: dict | None = None, data: bytes | dict | None = None, *,
          headers: dict | None = None, timeout: float = 120, retries: int = 3,
          cache_hours: float = 0, method: str | None = None) -> bytes:
    """GET (or POST when data is given). Cached on disk when cache_hours > 0."""
    if params:
        url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params, doseq=True)
    if isinstance(data, dict):
        data = urllib.parse.urlencode(data, doseq=True).encode()
    key = hashlib.sha256((url + "|" + (data or b"").decode("latin1")).encode()).hexdigest()[:32]
    cfile = cache_dir() / key
    if cache_hours > 0 and cfile.exists() and time.time() - cfile.stat().st_mtime < cache_hours * 3600:
        return cfile.read_bytes()
    h = {"User-Agent": UA}
    h.update(headers or {})
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers=h, method=method)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
            if cache_hours > 0:
                cfile.write_bytes(body)
            return body
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (400, 401, 403, 404):
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        time.sleep(2 * (attempt + 1))
    raise ConnectionError(f"{url[:120]}: {last}")


def fetch_json(url: str, params: dict | None = None, data=None, **kw):
    return json.loads(fetch(url, params, data, **kw).decode("utf-8"))


def post_json(url: str, obj, **kw):
    return json.loads(fetch(url, data=json.dumps(obj).encode(),
                            headers={"Content-Type": "application/json"}, **kw).decode("utf-8"))
