"""Offline tests: parsers, tracks, ellipses, plate matching, imaging, stamp grid."""
import json
import math
import os

import numpy as np
import pytest

from spacerocksuite.track import Track, cov_ellipse, offset_arcsec, sep_deg, inside_fraction

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")


# ------------------------------------------------------------------ parsers
def test_horizons_csv():
    from spacerocksuite.services.horizons import parse_observer_csv
    t = parse_observer_csv(open(os.path.join(DATA, "horizons_ed138_1982.txt")).read(), "2026 ED138")
    assert t.name == "2026 ED138" and len(t) == 4
    assert abs(t.ra[1] - 206.06888) < 1e-4 and abs(t.dec[1] + 36.66019) < 1e-4
    assert t.smaa is not None and 0 < t.smaa[0] < 1 and t.theta is not None


def test_findorb_table():
    from spacerocksuite.services.findorb import parse_table
    d = parse_table(open(os.path.join(DATA, "findorb_ed138_809.txt")).read())
    assert len(d["jd"]) == 3
    assert abs(d["sig"][0] - 0.176) < 1e-9 and d["pa"][0] == 120 and d["mag"][0] == 19.4
    ra = (13 + 44 / 60 + 21.325 / 3600) * 15
    assert abs(d["ra"][0] - ra) < 1e-6 and abs(d["delta"][0] - 41.268) < 1e-9


def test_ephemeral_parser_and_share_link():
    from spacerocksuite.services import ephemeral
    t = ephemeral.track_from_text(open(os.path.join(DATA, "horizons_ed138_1982.txt")).read())
    assert len(t) == 4 and t.vmag is not None
    url = ephemeral.share_link(t)
    d = ephemeral.unpack(url.split("#s=")[1])
    assert d["v"] == 1 and d["o"][0]["n"] == t.name
    assert ephemeral.track_from_text(d["o"][0]["x"]).ra[0] == pytest.approx(t.ra[0], abs=1e-6)


def test_obs80_parser():
    from spacerocksuite.ephemeral_parse import parse
    line = "     K26ED8D 0 1937 12 30.03599510 09 20.105+07 33 17.21               W     074"
    d = parse(line)
    assert len(d["rows"]) == 1
    assert d["rows"][0]["ra_deg"] == pytest.approx((10 + 9 / 60 + 20.105 / 3600) * 15)


# ------------------------------------------------------------------ geometry
def test_cov_ellipse_recovers_gaussian():
    rng = np.random.default_rng(0)
    a, b, pa = 30.0, 10.0, 35.0
    t = math.radians(pa)
    u, v = rng.normal(0, a, 20000), rng.normal(0, b, 20000)
    dx = u * math.sin(t) + v * math.cos(t)
    dy = u * math.cos(t) - v * math.sin(t)
    A, B, PA, sc = cov_ellipse(dx[:, None], dy[:, None])
    assert A[0] == pytest.approx(a, rel=0.03) and B[0] == pytest.approx(b, rel=0.03)
    assert PA[0] == pytest.approx(pa, abs=1.0)
    assert np.allclose(sc[:4, 0], 1.0, atol=0.06)       # Gaussian cloud: no extra tail scaling


def test_track_interp_wraps_ra():
    t = Track("x", "test", np.array([0.0, 1.0]), np.array([359.9, 0.1]), np.array([0.0, 0.0]))
    ra, dec, *_ = t.at(0.5)
    assert min(abs(ra - 0), abs(ra - 360)) < 1e-6


def test_offsets_and_fraction():
    dx, dy = offset_arcsec(10.001, 20.0, 10.0, 20.0)
    assert dx == pytest.approx(3.6 * math.cos(math.radians(20)), rel=1e-4) and abs(dy) < 1e-3
    assert inside_fraction(0, 0, 10, 5, 0, 100, 100) == 1.0
    assert inside_fraction(200, 0, 10, 5, 0, 100, 100) == 0.0
    assert 0.3 < inside_fraction(100, 0, 10, 10, 0, 100, 100) < 0.7


def test_far_side_points_are_not_near():
    dx, dy = offset_arcsec(213.29, -38.64, 33.27, 40.23)     # ED138 vs POSS-II R 1557 centre
    assert np.isnan(dx) and np.isnan(dy)
    assert inside_fraction(float(dx), float(dy), 10, 10, 0, 1e4, 1e4) == 0.0


def test_supercosmos_offline_finds_eso_4699():
    from spacerocksuite.services.horizons import parse_observer_csv
    from spacerocksuite.archives import SuperCOSMOS
    t = parse_observer_csv(open(os.path.join(DATA, "horizons_ed138_1982.txt")).read())
    hits = SuperCOSMOS().search(t, 3)
    ids = [h.plate_id for h in hits]
    assert "ESO-R 4699" in ids
    h = hits[ids.index("ESO-R 4699")]
    assert h.covered_fraction == 1.0 and h.extra["site"] == "809"
    assert all(not x.plate_id.startswith("POSS-II") for x in hits)  # ED138 is far south of POSS-II


# ------------------------------------------------------------------ imaging
def _wcs(ra0, dec0, scale, n):
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra0, dec0]
    w.wcs.crpix = [n / 2 + 0.5, n / 2 + 0.5]
    w.wcs.cdelt = [-scale / 3600, scale / 3600]
    return w


def _gauss(n, x, y, f, s=1.3):
    yy, xx = np.mgrid[:n, :n]
    return f * np.exp(-0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / s ** 2) / (2 * np.pi * s * s)


def test_shift_stack_finds_faint_mover_and_null_tests():
    from spacerocksuite.imaging import stack as S
    n, scale, K = 120, 1.0, 16
    rng = np.random.default_rng(5)
    jd = 2450000 + np.arange(K) * 0.02
    ra = 150.0 + np.arange(K) * 6e-4            # ~2.2"/frame east
    dec = 10.0 + np.arange(K) * 3e-4
    tr = Track("mover", "test", jd, ra, dec)
    w = _wcs(150.0045, 10.0022, scale, n)
    frames = []
    for k in range(K):
        x, y = w.all_world2pix(ra[k], dec[k], 0)
        img = rng.normal(0, 1, (n, n)) + _gauss(n, x, y, 9.0)      # S/N ~2 per frame
        img += _gauss(n, 30, 30, 400)                               # a static star
        frames.append(S.Frame(img.astype(np.float32), w, jd[k]))
    res = S.stack(frames, tr, "track", size_px=41, scale_arcsec=scale, method="mean")
    noise, _ = S.control_noise(frames, tr, "track", size_px=41, scale_arcsec=scale, method="mean", r_px=3)
    flux, _ = S.photometry(res["stack"], 3)
    assert flux / noise > 5
    nt = S.null_tests(frames, tr, r_px=3, size_px=41, scale_arcsec=scale, method="mean")
    assert nt["predicted"] == max(nt.values())
    assert nt["reversed"] < 0.5 * nt["predicted"]


def test_matrix_deblend_separates_blend():
    from spacerocksuite.imaging import cleanup
    n = 41
    psf = cleanup.gaussian_psf_fn(15, 3.0)
    img = _gauss(n, 20, 20, 1000, 3.0 / 2.3548) + _gauss(n, 24.5, 20, 300, 3.0 / 2.3548) + 5.0
    img += np.random.default_rng(1).normal(0, 0.5, (n, n))
    r = cleanup.matrix_deblend(img, [(20, 20)], psf, target_xy=(24.5, 20))
    assert r["flux"][0] == pytest.approx(1000, rel=0.02)
    assert r["target_flux"] == pytest.approx(300, rel=0.05) and r["target_snr"] > 20
    assert abs(np.median(r["cleaned"])) < 1.0


def test_round_gaussian_fit():
    from spacerocksuite.imaging import cleanup
    img = _gauss(31, 15.3, 14.6, 500, 1.5) + np.random.default_rng(2).normal(0, 0.2, (31, 31))
    f = cleanup.fit_gaussian(img, 15, 15, box=6)
    assert f["x"] == pytest.approx(15.3, abs=0.02) and f["y"] == pytest.approx(14.6, abs=0.02)
    assert f["sigma"] == pytest.approx(1.5, rel=0.03)


def test_background_and_template():
    from spacerocksuite.imaging import cleanup
    yy, xx = np.mgrid[:80, :80]
    sky = 100 + 0.5 * xx
    img = sky + _gauss(80, 40, 40, 200)
    b = cleanup.background(img.astype(np.float32), 21)
    assert abs(np.median(b)) < 0.5 and b[40, 40] > 10
    tpl = 2.0 * sky + 7
    diff, _ = cleanup.template_subtract(img, [tpl])
    assert diff[40, 40] > 10 and abs(np.median(diff)) < 0.5


def test_stamp_grid_is_deterministic(tmp_path):
    from spacerocksuite.imaging import stampgrid
    rng = np.random.default_rng(0)
    np.savez(tmp_path / "s.npz", a=rng.normal(size=(40, 40)), b=rng.normal(size=(40, 40)))
    spec = {"title": "Test 1", "subtitle": "stamps", "out": "g.png", "ncols": 3,
            "panels": [{"file": "s.npz", "key": "a", "hw": 20, "r": 3, "title": "A", "subtitle": "x", "n": 5},
                       {"file": "s.npz", "key": "b", "hw": 20, "r": 3, "title": "B", "subtitle": "y"}]}
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    p1 = stampgrid.render_file(str(tmp_path / "spec.json"))
    h1 = stampgrid.sha256(p1)
    p2 = stampgrid.render_file(str(tmp_path / "spec.json"))
    assert stampgrid.sha256(p2) == h1
    assert b"Software" not in open(p1, "rb").read()[:400]


def test_device_profile():
    from spacerocksuite.device import detect
    p = detect()
    assert p.tier in ("light", "medium", "full") and p.net_workers >= 1


# ------------------------------------------------------------------ network
@pytest.mark.network
def test_horizons_live():
    from spacerocksuite.services import horizons
    t = horizons.ephemeris("2026 ED138", "1982-03-18", "1982-03-20", "1 d")
    assert len(t) == 3 and t.smaa is not None
