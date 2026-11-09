"""Target tab: pasted Find_Orb ephemerides, the orbit sketch and NEOCP hints."""
import math
import os

import numpy as np
import pytest

DATA = os.path.join(os.path.dirname(__file__), "data")


def _read(name):
    return open(os.path.join(DATA, name), encoding="utf-8").read()


def test_generic_parser_does_not_eat_ra_hour():
    # "1982 03 18  13 44 21.325": 13 is the RA hour, not the time of day
    from spacerocksuite.ephemeral_parse import parse
    r = parse(_read("findorb_ed138_809.txt"))["rows"][0]
    assert r["jd"] == pytest.approx(2445046.5)
    assert r["ra_deg"] == pytest.approx((13 + 44 / 60 + 21.325 / 3600) * 15)


@pytest.mark.parametrize("name,ra0,sig", [("findorb_ed138_809.txt", 206.0888542, 0.176),
                                          ("findorb_web_paste.txt", 273.0100208, 0.020)])
def test_pasted_findorb_keeps_sigma_mag_delta(name, ra0, sig):
    from spacerocksuite.services import ephemeral
    t = ephemeral.track_from_text(_read(name))
    assert t.ra[0] == pytest.approx(ra0, abs=1e-6)
    assert t.smaa[0] == pytest.approx(sig)
    assert t.vmag is not None and t.delta is not None
    assert t.meta["format"] == "Find_Orb"


def test_findorb_name_from_header():
    from spacerocksuite.services import ephemeral
    assert ephemeral.track_from_text(_read("findorb_ed138_809.txt")).name == "2026 ED138"
    assert ephemeral.track_from_text(_read("findorb_ed138_809.txt"), "mine").name == "mine"


def test_paste_source_in_get_track():
    from spacerocksuite import precovery as P
    t = P.get_track("", "2000-01-01", "2000-01-02", source="ephemeral", ephem_text=_read("findorb_web_paste.txt"))
    assert len(t) == 5


@pytest.mark.parametrize("a,e,i", [(2.77, 0.08, 10.6), (43.0, 0.14, 24.4), (-1.27, 1.2, 122.7)])
def test_elements_round_trip(a, e, i):
    from spacerocksuite.orbit_view import GM, elements, orbit_curve
    # state at perihelion of an orbit inclined about the x axis
    q = a * (1 - e)
    vq = math.sqrt(GM * (1 + e) / q)
    ir = math.radians(i)
    r = np.array([q, 0, 0])
    v = np.array([0, vq * math.cos(ir), vq * math.sin(ir)])
    a2, e2, i2, q2, _, _ = elements(r, v)
    assert (a2, e2, i2, q2) == pytest.approx((a, e, i, q), rel=1e-9)
    pts = orbit_curve(r, v, r_max=10 * q)
    rr = np.linalg.norm(pts, axis=1)
    assert rr.min() == pytest.approx(q, rel=1e-3)
    if e < 1:
        assert rr.max() == pytest.approx(a * (1 + e), rel=1e-3)


def test_orbit_sketch_from_track():
    from spacerocksuite import orbit_view
    from spacerocksuite.services import ephemeral
    g = orbit_view.build(ephemeral.track_from_text(_read("findorb_web_paste.txt")))
    assert 46 < g["r_range"][0] < 48.5            # Find_Orb's r column says 47.227 au
    assert g["extent"] > 30 and "Neptune" in [n for n, _ in g["planets"]]


def test_orbit_sketch_without_delta():
    from spacerocksuite import orbit_view
    from spacerocksuite.services import ephemeral
    g = orbit_view.build(ephemeral.track_from_text(_read("horizons_ed138_1982.txt")))
    assert "error" in g


def test_neocp_hint():
    from spacerocksuite.precovery import looks_like_neocp
    assert looks_like_neocp("P12rWvN") and looks_like_neocp("ZTF0ABC")
    assert not looks_like_neocp("2026 ED138") and not looks_like_neocp("Ceres") and not looks_like_neocp("433")
