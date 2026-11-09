"""Command line: every GUI step can also run headless (handy over SSH on a Pi).

  srs info
  srs track "2026 ED138" 1950-01-01 2003-12-31 --source horizons -o ed138.json
  srs plates ed138.json --sigma 3 --archives SuperCOSMOS DASCH -o plates.csv
  srs stamps ed138.json plates.csv --top 9 -o grid/
  srs grid grid/stamp_grid.json
  srs stack ed138.json frames/*.fits --mode track -o stack.npz
  srs ephemeral ed138.json            (prints a link that opens the track in Ephemeral)
  srs update
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

from . import __version__, config


def _track(path):
    from .track import Track
    with open(path, encoding="utf-8") as f:
        return Track.from_json(f.read())


def cmd_info(a):
    from . import core
    from .device import PROFILE
    print(f"SpaceRockSuite {__version__}")
    print(f"backend: {core.backend()}")
    print(json.dumps(PROFILE.as_dict(), indent=1))


def cmd_track(a):
    from . import precovery as P
    obs = open(a.obs, encoding="utf-8").read() if a.obs else None
    eph = open(a.ephem, encoding="utf-8").read() if a.ephem else None
    t = P.get_track(a.target, a.start, a.stop, a.source, a.step, a.site, obs, eph)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(t.to_json())
    print(f"{t.name}: {len(t)} rows from {t.source} -> {a.out}")


def cmd_plates(a):
    from . import precovery as P
    from .archives import write_csv
    s = config.load()
    t = _track(a.track)
    obs = open(a.obs, encoding="utf-8").read() if a.obs else None
    hits = P.search_plates(t, a.sigma, a.archives or s["archives"], obs80=obs,
                           date_fix_days=s["ssa_date_fix_days"])
    write_csv(hits, a.out)
    for h in hits[:20]:
        dm = h.depth_margin
        print(f"{P.iso(h.mid_jd)}  {h.archive:<18} {h.plate_id:<28} V={h.vmag_pred or float('nan'):5.1f} "
              f"lim={h.limit_mag or float('nan'):5.1f} margin={'' if dm is None else f'{dm:+.1f}'}")
    print(f"{len(hits)} plates/frames -> {a.out}")


def cmd_stamps(a):
    from . import precovery as P
    from .archives.base import PlateHit
    from .imaging import stampgrid
    t = _track(a.track)
    rows = list(csv.DictReader(open(a.plates, encoding="utf-8")))
    hits = []
    for r in rows:
        if not (r["archive"].startswith("SuperCOSMOS") or r["archive"].startswith("DSS")):
            continue
        h = PlateHit(r["archive"], r["plate_id"], r["telescope"], r["band"], float(r["mid_jd"]),
                     float(r["exptime_s"]), float(r["ra_obj"]), float(r["dec_obj"]),
                     float(r["vmag_pred"]) if r["vmag_pred"] else None, float(r["sigma_a"]),
                     float(r["sigma_b"]), float(r["sigma_pa"]),
                     float(r["limit_mag"]) if r["limit_mag"] else None)
        if h.archive.startswith("SuperCOSMOS"):
            from .archives.supercosmos import SURVEYS
            h.extra["survey_id"] = next(k for k, v in SURVEYS.items() if h.plate_id.startswith(v[0]))
        else:
            from .archives.dss import SURVEYS
            h.extra["survey"] = next(k for k, v in SURVEYS.items() if h.plate_id.startswith(v))
        hits.append(h)
    res = []
    for h in hits[:a.top]:
        try:
            blob = P.fetch_cutout(h)
            r = (P.measure_with_date_probe(blob, h, t, matched=a.matched)
                 if h.archive.startswith("SuperCOSMOS") else P.measure_stamp(blob, h, matched=a.matched))
            res.append(r)
            print(f"{h.plate_id}: S/N {r.snr:.1f} {r.note}")
        except Exception as e:
            print(f"{h.plate_id}: skipped ({e})")
    spec = P.grid_spec(t.name, res, a.out)
    print("saved", stampgrid.render_file(spec))


def cmd_grid(a):
    from .imaging import stampgrid
    out = stampgrid.render_file(a.spec)
    print("saved", out, stampgrid.sha256(out)[:16])


def cmd_stack(a):
    import numpy as np
    from .imaging import stack as S
    t = _track(a.track)
    frames = [S.Frame.from_fits(p, name=os.path.basename(p)) for p in a.frames]
    try:
        frames = S.usable_frames(frames, t, a.mode)
    except S.NoCoverage as e:
        raise SystemExit(f"Nothing to stack. {e}")
    res = S.stack(frames, t, a.mode, a.size, a.scale, a.method, bkg_size=a.bkg or None)
    noise, _ = S.control_noise(frames, t, a.mode, size_px=a.size, scale_arcsec=a.scale, method=a.method,
                               r_px=a.r)
    flux, _ = S.photometry(res["stack"], a.r)
    np.savez_compressed(a.out, stack=res["stack"], coverage=res["coverage"])
    print(f"stack of {len(frames)} frames ({a.mode}): flux {flux:.4g}, control noise {noise:.4g}, "
          f"S/N {flux / noise if noise else float('nan'):.1f} -> {a.out}")
    if a.null:
        for k, v in S.null_tests(frames, t, r_px=a.r, size_px=a.size, scale_arcsec=a.scale).items():
            print(f"  {k:<12} {v / noise if noise else v:+.1f} sigma")


def cmd_ephemeral(a):
    from .services import ephemeral
    print(ephemeral.share_link(_track(a.track)))


def cmd_update(a):
    from . import updater
    u = updater.check()
    if not u:
        print(f"SpaceRockSuite {__version__} is up to date.")
        return
    print(f"Version {u['version']} is available: {u['url']}")
    if a.yes and u["asset"]:
        print(updater.install(updater.download(u["asset"])))


def main(argv=None):
    config.apply(config.load())
    p = argparse.ArgumentParser(prog="srs", description="SpaceRockSuite precovery toolkit")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("info").set_defaults(fn=cmd_info)
    s = sub.add_parser("track")
    s.add_argument("target"); s.add_argument("start"); s.add_argument("stop")
    s.add_argument("--source", default="horizons", choices=["horizons", "scout", "findorb", "ephemeral"])
    s.add_argument("--step", type=float, default=1.0, help="days")
    s.add_argument("--site", default="500")
    s.add_argument("--obs", help="MPC 80-col file for Find_Orb")
    s.add_argument("--ephem", help="ephemeris text file for the Ephemeral parser")
    s.add_argument("-o", "--out", default="track.json"); s.set_defaults(fn=cmd_track)
    s = sub.add_parser("plates")
    s.add_argument("track"); s.add_argument("--sigma", type=float, default=3.0)
    s.add_argument("--archives", nargs="*"); s.add_argument("--obs")
    s.add_argument("-o", "--out", default="plates.csv"); s.set_defaults(fn=cmd_plates)
    s = sub.add_parser("stamps")
    s.add_argument("track"); s.add_argument("plates"); s.add_argument("--top", type=int, default=9)
    s.add_argument("--matched", action="store_true"); s.add_argument("-o", "--out", default="stamps")
    s.set_defaults(fn=cmd_stamps)
    s = sub.add_parser("grid"); s.add_argument("spec"); s.set_defaults(fn=cmd_grid)
    s = sub.add_parser("stack")
    s.add_argument("track"); s.add_argument("frames", nargs="+")
    s.add_argument("--mode", default="track", choices=["track", "static"])
    s.add_argument("--size", type=int, default=101); s.add_argument("--scale", type=float, default=1.0)
    s.add_argument("--method", default="clipmean", choices=["mean", "median", "clipmean"])
    s.add_argument("--bkg", type=int, default=0); s.add_argument("--r", type=float, default=3.0)
    s.add_argument("--null", action="store_true"); s.add_argument("-o", "--out", default="stack.npz")
    s.set_defaults(fn=cmd_stack)
    s = sub.add_parser("ephemeral"); s.add_argument("track"); s.set_defaults(fn=cmd_ephemeral)
    s = sub.add_parser("update"); s.add_argument("-y", "--yes", action="store_true"); s.set_defaults(fn=cmd_update)
    a = p.parse_args(argv)
    if not getattr(a, "fn", None):
        from .gui.app import main as gui
        return gui()
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
