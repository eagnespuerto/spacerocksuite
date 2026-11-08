"""The Rust core and the NumPy fallback must give the same numbers."""
import importlib
import math
import os

import numpy as np
import pytest


def load(no_rust: bool):
    if no_rust:
        os.environ["SRS_NO_RUST"] = "1"
    else:
        os.environ.pop("SRS_NO_RUST", None)
    import spacerocksuite.core as c
    return importlib.reload(c)


@pytest.fixture(scope="module")
def py():
    return load(True)


def test_kepler_against_kepler_equation(py):
    GM = py.GM
    for q, e in [(1.0, 0.0), (2.1, 0.3), (0.5, 0.999), (1.5, 1.8)]:
        for dt in (-3000.0, 1.0, 900.0):
            P = py._helio_xyz(np.array([[q, e, 0, 0, 0, 0]]), np.array([dt]))[0, 0]
            ce, se = math.cos(py.EPS), math.sin(py.EPS)
            x, y = P[0], P[1] * ce + P[2] * se
            if e < 1:
                a = q / (1 - e); M = math.sqrt(GM / a ** 3) * dt; E = M
                for _ in range(60):
                    E -= (E - e * math.sin(E) - M) / (1 - e * math.cos(E))
                rx, ry = a * (math.cos(E) - e), a * math.sqrt(1 - e * e) * math.sin(E)
            else:
                a = q / (e - 1); M = math.sqrt(GM / a ** 3) * dt; H = math.asinh(M / e)
                for _ in range(100):
                    H -= (e * math.sinh(H) - H - M) / (e * math.cosh(H) - 1)
                rx, ry = a * (e - math.cosh(H)), a * math.sqrt(e * e - 1) * math.sinh(H)
            assert math.hypot(x - rx, y - ry) < 1e-9 * max(1, math.hypot(rx, ry))


def test_circular_orbit_distance(py):
    # observer at the Sun -> delta == q for a circular orbit
    ra, dec, d = py.kepler_radec([[2.0, 0.0, 10, 30, 40, 2451545.0]], [2451545.0, 2451645.0], np.zeros((2, 3)))
    assert np.allclose(d, 2.0, atol=1e-12)


rust = pytest.importorskip("srs_core")


def test_parity_kepler(py):
    rs = load(False)
    rng = np.random.default_rng(1)
    el = np.column_stack([rng.uniform(0.5, 40, 50), rng.uniform(0, 1.5, 50), rng.uniform(0, 170, 50),
                          rng.uniform(0, 360, 50), rng.uniform(0, 360, 50), rng.uniform(2.44e6, 2.47e6, 50)])
    jd = np.linspace(2.43e6, 2.47e6, 7)
    E = np.column_stack([np.cos(jd / 58), np.sin(jd / 58), np.zeros(7)])
    a = py.kepler_radec(el, jd, E)
    b = rs.kepler_radec(el, jd, E)
    dra = (a[0] - b[0] + 180) % 360 - 180
    assert np.nanmax(np.abs(dra) * 3600) < 1e-4 and np.nanmax(np.abs(a[1] - b[1]) * 3600) < 1e-4
    assert np.allclose(a[2], b[2], rtol=1e-10)


def test_parity_filters_and_stack(py):
    rs = load(False)
    rng = np.random.default_rng(2)
    img = rng.normal(0, 1, (60, 70)).astype(np.float32)
    img[5, 7] = np.nan
    for f, args in ((lambda c, i: c.gaussian_blur(i, 1.5), ()), (lambda c, i: c.median_filter(i, 5), ()),
                    (lambda c, i: c.median_filter(i, 21), ())):
        x, y = f(py, img), f(rs, img)
        assert np.allclose(x, y, atol=1e-5, equal_nan=True)
    frames = rng.normal(0, 1, (6, 40, 50)).astype(np.float32)
    dx = np.linspace(0, 5.5, 6)
    dy = np.linspace(0, -2.25, 6)
    for m in ("mean", "median", "clipmean"):
        s1, c1 = py.shift_stack(frames, dx, dy, m)
        s2, c2 = rs.shift_stack(frames, dx, dy, m)
        assert np.array_equal(c1, c2)
        assert np.allclose(s1, s2, atol=1e-5, equal_nan=True)


def test_stack_identity(py):
    f = np.random.default_rng(3).normal(size=(4, 20, 20)).astype(np.float32)
    s, c = py.shift_stack(np.repeat(f[:1], 4, 0), np.zeros(4), np.zeros(4), "mean")
    assert np.allclose(s, f[0]) and (c == 4).all()


def test_blur_conserves_flux(py):
    img = np.zeros((41, 41), np.float32)
    img[20, 20] = 1
    assert abs(py.gaussian_blur(img, 2.0).sum() - 1) < 1e-4
