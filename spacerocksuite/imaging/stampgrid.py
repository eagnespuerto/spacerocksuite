"""Deterministic stamp grids (port of the `stamp-grid` Claude skill).

Same spec file and house style as the skill, so existing stamp_grid.json files
work unchanged:

    {"title": "2026 EE138", "subtitle": "TESS and SPHEREx shift-stacks (...)",
     "out": "stamp_grid.png", "ncols": 3,
     "panels": [{"file": "x.npz", "key": "s45", "hw": 168, "r": 31.5,
                 "title": "TESS S45, 2021-11", "subtitle": "not detected", "n": 2655}, ...]}

file: .npy (2-D, or 3-D with "index"), .npz (with "key") or .fits (with "ext").
hw = stamp half-width, r = aperture radius (arcsec). Paths relative to the spec.

House style: header "<Object>: <subtitle>" (13 pt, not bold); two-line panel
titles (11 pt, not bold) with "  (N = n)" when n > 3; grayscale, -3..+5 sigma
(MAD) stretch, origin lower, east left / north up, axes in arcsec; red #d62728
lw 1.3 aperture circle at the centre; DejaVu Sans; 4.5 x ncols by 5.2 x nrows
inches at 110 dpi.

Determinism: Agg backend, fixed rcParams, no PNG metadata (no timestamp or
version string), sorted panel input exactly as given. The same spec and data
give the same bytes on any machine with the same matplotlib/FreeType.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import numpy as np

STYLE = {
    "font.family": "DejaVu Sans", "font.weight": "normal", "axes.titleweight": "normal",
    "figure.titleweight": "normal", "svg.hashsalt": "spacerocksuite", "path.simplify": False,
    "image.interpolation": "nearest", "image.resample": False, "text.hinting": "default",
    "agg.path.chunksize": 0,
}
RED = "#d62728"


def _mpl():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    matplotlib.rcdefaults()
    plt.rcParams.update(STYLE)
    return plt


def load_panel(p: dict, base: str):
    if "data" in p:                       # in-memory array (from the app)
        a = np.asarray(p["data"])
    else:
        path = os.path.join(base, p["file"])
        if path.lower().endswith((".fits", ".fit", ".fz", ".fits.gz")):
            from astropy.io import fits
            a = np.asarray(fits.getdata(path, p.get("ext", 0)), float)
        else:
            a = np.load(path)
            a = a[p["key"]] if "key" in p else a
    a = np.asarray(a, float)
    return a[p.get("index", 0)] if a.ndim == 3 else a


def stretch(im):
    """(vmin, vmax) = -3, +5 robust sigma about zero (MAD), as in the skill."""
    s = 1.4826 * np.nanmedian(np.abs(im - np.nanmedian(im)))
    s = float(s) if np.isfinite(s) and s > 0 else 1.0
    return -3 * s, 5 * s


def render(spec: dict, base: str = ".", out: str | None = None) -> str:
    plt = _mpl()
    from matplotlib.patches import Circle
    P = spec["panels"]
    nc = int(spec.get("ncols", 3))
    nr = max(1, math.ceil(len(P) / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(4.5 * nc, 5.2 * nr), squeeze=False)
    for a in ax.flat[len(P):]:
        a.axis("off")
    for a, p in zip(ax.flat, P):
        im = load_panel(p, base)
        hw = float(p["hw"])
        vmin, vmax = stretch(im)
        a.imshow(im, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, extent=[-hw, hw, -hw, hw])
        a.invert_xaxis()
        a.add_patch(Circle((0, 0), float(p["r"]), fill=False, ec=RED, lw=1.3))
        n = int(p.get("n", 0) or 0)
        a.set_title(f"{p['title']}\n{p['subtitle']}" + (f"  (N = {n})" if n > 3 else ""), fontsize=11)
        a.set_xlabel("arcsec (east left)", fontsize=9)
        a.set_ylabel("arcsec (north up)", fontsize=9)
        a.tick_params(labelsize=8)
    fig.suptitle(f"{spec['title']}: {spec['subtitle']}", fontsize=13, fontweight="normal")
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3.0)
    out = out or os.path.join(base, spec.get("out") or default_name(spec["title"]))
    fig.savefig(out, dpi=110, metadata={"Software": None})
    plt.close(fig)
    return out


def default_name(title: str) -> str:
    """stamp_grid_<OBJECT_NO_SPACES>.png, keeping only letters, digits, '-' and '_'."""
    import re
    return "stamp_grid_" + re.sub(r"[^A-Za-z0-9_-]", "", title) + ".png"


def render_file(spec_path: str) -> str:
    base = os.path.dirname(os.path.abspath(spec_path))
    with open(spec_path, encoding="utf-8") as f:
        return render(json.load(f), base)


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def panel_text(instrument: str, result: dict) -> tuple[str, str]:
    """Line-2 wording from a photometry result, in the skill's style.

    result: {"flux", "err"} (any unit) and optional "mag", "limit_mag", "band".
    """
    f, e = result.get("flux"), result.get("err")
    snr = f / e if (f is not None and e) else None
    band = result.get("band", "")
    if snr is not None and snr >= 3:
        line2 = f"{snr:.1f}σ" + (f", {band} = {result['mag']:.2f}" if result.get("mag") is not None else "")
    elif result.get("limit_mag") is not None:
        line2 = f"not detected, {band} > {result['limit_mag']:.1f} (3σ)".replace(" ,", ",")
    else:
        line2 = "not detected" + (f" ({snr:.1f}σ)" if snr is not None else "")
    return instrument, line2


if __name__ == "__main__":
    print("saved", render_file(sys.argv[1]))
