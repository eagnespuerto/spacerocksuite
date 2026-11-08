"""User settings, stored as JSON in the app folder. Defaults follow the device profile."""
from __future__ import annotations

import json

from .device import PROFILE
from .net import app_dir

DEFAULTS = {
    "nsigma": 3.0,
    "archives": ["SuperCOSMOS", "DASCH", "SSOIS", "DSS"],
    "source": "horizons",
    "site": "500",
    "step_days": 1.0,
    "ssa_date_fix_days": 1.0,
    "findorb_binary": "",
    "net_workers": PROFILE.net_workers,
    "mc_samples": PROFILE.mc_samples,
    "stamp_px": PROFILE.stamp_px,
    "output_dir": str(app_dir() / "runs"),
    "check_updates": True,
}


def path():
    return app_dir() / "settings.json"


def load() -> dict:
    s = dict(DEFAULTS)
    try:
        s.update(json.loads(path().read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    return s


def save(s: dict) -> None:
    path().write_text(json.dumps(s, indent=1), encoding="utf-8")


def apply(s: dict) -> None:
    """Push settings that change runtime limits into the device profile."""
    PROFILE.net_workers = int(s.get("net_workers", PROFILE.net_workers))
    PROFILE.mc_samples = int(s.get("mc_samples", PROFILE.mc_samples))
    PROFILE.stamp_px = int(s.get("stamp_px", PROFILE.stamp_px))
