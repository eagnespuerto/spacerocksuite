"""Check GitHub Releases for a newer SpaceRockSuite and install it.

Windows: download the installer (.exe) and run it (the user confirms first in
the GUI). Linux / Raspberry Pi OS: download the wheel for this platform and
pip-install it into the running environment; the app restarts afterwards.
"""
from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
import tempfile

from . import REPO, __version__, net

API = f"https://api.github.com/repos/{REPO}/releases/latest"


def _v(s: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", s)[:3])


def latest() -> dict | None:
    try:
        return net.fetch_json(API, headers={"Accept": "application/vnd.github+json"}, timeout=20, retries=1)
    except Exception:
        return None


def check() -> dict | None:
    """{'version', 'notes', 'asset': {...}} when a newer release exists, else None."""
    rel = latest()
    if not rel or rel.get("draft") or rel.get("prerelease"):
        return None
    tag = rel.get("tag_name", "")
    if _v(tag) <= _v(__version__):
        return None
    a = pick_asset(rel.get("assets", []))
    return {"version": tag.lstrip("v"), "notes": rel.get("body", ""), "url": rel.get("html_url"), "asset": a}


def platform_tag() -> str:
    m = platform.machine().lower()
    if os.name == "nt":
        return "windows"
    if m in ("aarch64", "arm64"):
        return "linux-arm64"
    if m.startswith("arm"):
        return "linux-armv7"
    return "linux-x86_64"


def pick_asset(assets: list[dict]) -> dict | None:
    tag = platform_tag()
    want = {"windows": r"setup.*\.exe$", "linux-x86_64": r"(x86_64|amd64).*\.(whl|tar\.gz)$",
            "linux-arm64": r"(aarch64|arm64).*\.(whl|tar\.gz)$", "linux-armv7": r"armv7.*\.(whl|tar\.gz)$"}[tag]
    for a in assets:
        if re.search(want, a["name"], re.I):
            return a
    return None


def download(asset: dict, progress=None) -> str:
    d = tempfile.mkdtemp(prefix="srs-update-")
    path = os.path.join(d, asset["name"])
    data = net.fetch(asset["browser_download_url"], timeout=600)
    with open(path, "wb") as f:
        f.write(data)
    if progress:
        progress(1.0)
    return path


def install(path: str) -> str:
    """Start installing; returns a message. The caller should then quit the app."""
    if path.endswith(".exe"):
        subprocess.Popen([path, "/SILENT", "/CLOSEAPPLICATIONS"], close_fds=True)
        return "The installer is running. SpaceRockSuite will close now."
    if path.endswith(".tar.gz"):     # Pi bundle: wheels + install script
        import tarfile
        d = os.path.dirname(path)
        with tarfile.open(path) as t:
            t.extractall(d, filter="data") if sys.version_info >= (3, 12) else t.extractall(d)
        wheels = [os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs if f.endswith(".whl")]
        path = " ".join(wheels)
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", *path.split()]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-800:])
    return "Updated. Please restart SpaceRockSuite."
