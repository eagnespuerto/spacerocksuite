"""A simple top-down orbit sketch from a Track (no extra dependencies).

The ephemeris gives RA, Dec and distance from Earth. Adding Earth's position
gives the object's position around the Sun at each row; that path is the
"track". A two-body orbit fitted to the middle of the track is drawn dashed
around it so you can see where the arc sits on the whole orbit.

It is a sketch: light-time and the observer's offset from the geocentre are
ignored, which is far below what the diagram can show.
"""
from __future__ import annotations

import math

import numpy as np

GM = 0.01720209895 ** 2          # au^3 / day^2 (Gaussian constant)
EPS = math.radians(23.4392911)   # J2000 obliquity
PLANETS = [("Mercury", 0.387), ("Venus", 0.723), ("Earth", 1.0), ("Mars", 1.524), ("Jupiter", 5.203),
           ("Saturn", 9.537), ("Uranus", 19.19), ("Neptune", 30.07)]


def _eq_to_ecl(xyz):
    c, s = math.cos(EPS), math.sin(EPS)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    return np.stack([x, c * y + s * z, -s * y + c * z], -1)


def helio_positions(track, max_rows: int = 2000):
    """(jd, object xyz, earth xyz) in heliocentric ecliptic au; None if there is no distance column."""
    if track.delta is None or np.all(~np.isfinite(track.delta)):
        return None
    ok = np.isfinite(track.delta) & np.isfinite(track.ra) & np.isfinite(track.dec)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return None
    if idx.size > max_rows:
        idx = idx[np.linspace(0, idx.size - 1, max_rows).round().astype(int)]
    jd = track.jd[idx]
    r, d = np.radians(track.ra[idx]), np.radians(track.dec[idx])
    u = np.stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)], -1)
    from .track import earth_helio
    E = earth_helio(jd)
    P = E + track.delta[idx, None] * u
    return jd, _eq_to_ecl(P), _eq_to_ecl(E)


def elements(r, v):
    """Osculating (a, e, i_deg, q, h_vec, e_vec) from heliocentric r (au), v (au/day)."""
    r, v = np.asarray(r, float), np.asarray(v, float)
    rn = np.linalg.norm(r)
    h = np.cross(r, v)
    ev = np.cross(v, h) / GM - r / rn
    e = float(np.linalg.norm(ev))
    energy = v @ v / 2 - GM / rn
    a = -GM / (2 * energy) if abs(energy) > 1e-15 else math.inf
    hn = np.linalg.norm(h)
    i = math.degrees(math.acos(np.clip(h[2] / hn, -1, 1)))
    q = hn ** 2 / GM / (1 + e)
    return a, e, i, q, h, ev


def orbit_curve(r, v, n: int = 361, r_max: float | None = None):
    """Points (n, 3) of the two-body orbit through state (r, v). Open orbits stop at r_max."""
    a, e, i, q, h, ev = elements(r, v)
    hn = h / np.linalg.norm(h)
    p = q * (1 + e)
    if e > 1e-8:
        P = ev / np.linalg.norm(ev)
    else:  # circle: any in-plane direction will do
        P = np.asarray(r, float) / np.linalg.norm(r)
    Q = np.cross(hn, P)
    if e < 1:
        nu = np.linspace(-math.pi, math.pi, n)
    else:
        rm = r_max or 3 * float(np.linalg.norm(r))
        c = (p / rm - 1) / e                    # cos(nu) where the orbit reaches r_max
        nu_max = math.acos(np.clip(c, -1, 1)) if c > -1 else math.acos(-1 / e) * 0.999
        nu = np.linspace(-nu_max, nu_max, n)
    rr = p / (1 + e * np.cos(nu))
    return (rr * np.cos(nu))[:, None] * P + (rr * np.sin(nu))[:, None] * Q


def build(track):
    """Everything the GUI needs to draw the sketch, or a dict with 'error'."""
    hp = helio_positions(track)
    if hp is None:
        return {"error": "No distance (delta) column in this ephemeris, so there is no orbit to draw."}
    jd, P, E = hp
    out = {"name": track.name, "jd": jd, "track": P, "earth": E, "orbit": None, "elements": None}
    if len(jd) >= 2:
        k = len(jd) // 2
        k0, k1 = max(k - 1, 0), min(k + 1, len(jd) - 1)
        if k1 == k0:
            k0, k1 = 0, len(jd) - 1
        dt = jd[k1] - jd[k0]
        if dt > 0:
            r = P[k]
            v = (P[k1] - P[k0]) / dt
            try:
                a, e, i, q, _, _ = elements(r, v)
                rmax = 2.5 * float(np.max(np.linalg.norm(P, axis=1)))
                out["orbit"] = orbit_curve(r, v, r_max=rmax)
                out["elements"] = {"a": a, "e": e, "i": i, "q": q, "Q": a * (1 + e) if e < 1 else math.inf}
            except (FloatingPointError, ValueError, ZeroDivisionError):
                pass
    xy = [P[:, :2]]
    if out["orbit"] is not None:
        xy.append(out["orbit"][:, :2])
    ext = max(1.2, float(np.nanmax(np.abs(np.concatenate(xy)))) * 1.08)
    out["extent"] = ext
    # reference orbits: those inside the frame, plus Earth always
    out["planets"] = [(n, a) for n, a in PLANETS if a <= ext or n == "Earth"]
    r = np.linalg.norm(P, axis=1)
    out["r_range"] = (float(r.min()), float(r.max()))
    return out


def draw(canvas, g, dark: bool = False):
    """Draw build()'s result on a Tk Canvas, scaled to the canvas size."""
    canvas.delete("all")
    W = max(int(canvas.winfo_width()), 50)
    H = max(int(canvas.winfo_height()), 50)
    bg, fg, dim = ("#111418", "#e8e8e8", "#556070") if dark else ("#ffffff", "#222222", "#b8c0cc")
    canvas.configure(background=bg)
    if "error" in g:
        canvas.create_text(W / 2, H / 2, text=g["error"], fill=fg, width=W - 20, justify="center")
        return
    cx, cy = W / 2, H / 2 + 8
    s = (min(W, H - 40) / 2 - 10) / g["extent"]

    def pts(xy):
        out = []
        for x, y in xy:
            out += [cx + x * s, cy - y * s]
        return out
    last_box = None
    for name, a in reversed(g["planets"]):      # outermost first, so big rings keep their labels
        R = a * s
        if R < 4:
            continue
        canvas.create_oval(cx - R, cy - R, cx + R, cy + R, outline=dim)
        t = canvas.create_text(cx + R * 0.71 + 2, cy - R * 0.71 - 2, text=name, fill=dim, anchor="sw",
                               font=("TkDefaultFont", 8))
        box = canvas.bbox(t)
        if last_box and box[1] < last_box[3] + 2:   # would touch the label above: drop it
            canvas.delete(t)
        else:
            last_box = box
    if g["orbit"] is not None:
        canvas.create_line(*pts(g["orbit"][:, :2]), fill="#7a8899", dash=(4, 3))
    canvas.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, fill="#ffcc33", outline="")
    tr = g["track"][:, :2]
    if len(tr) >= 2:
        canvas.create_line(*pts(tr), fill="#e8590c", width=3)
    x0, y0 = pts(tr[:1])
    x1, y1 = pts(tr[-1:])
    canvas.create_oval(x0 - 4, y0 - 4, x0 + 4, y0 + 4, fill="#2f9e44", outline="")
    canvas.create_oval(x1 - 4, y1 - 4, x1 + 4, y1 + 4, fill="#c92a2a", outline="")
    ex, ey = pts(g["earth"][len(g["earth"]) // 2:len(g["earth"]) // 2 + 1, :2])
    canvas.create_oval(ex - 3, ey - 3, ex + 3, ey + 3, fill="#1c7ed6", outline="")
    lines = [f"{g['name']}  (looking down on the ecliptic, ♈ to the right)"]
    el = g["elements"]
    if el:
        a = "open orbit" if not math.isfinite(el["a"]) or el["a"] < 0 else f"a {el['a']:.2f} au"
        lines.append(f"{a}   e {el['e']:.3f}   i {el['i']:.1f}°   q {el['q']:.2f} au")
    lines.append(f"r {g['r_range'][0]:.2f} to {g['r_range'][1]:.2f} au over the track")
    canvas.create_text(8, 6, text="\n".join(lines), fill=fg, anchor="nw", font=("TkDefaultFont", 9))
    x, y = 10, H - 12
    for kind, col, label in (("dot", "#2f9e44", "start"), ("dot", "#c92a2a", "end"),
                             ("dot", "#1c7ed6", "Earth (mid-track)"), ("line", "#e8590c", "track"),
                             ("dash", "#7a8899", "orbit (two-body)")):
        if kind == "dot":
            canvas.create_oval(x, y - 4, x + 8, y + 4, fill=col, outline="")
            x += 12
        else:
            canvas.create_line(x, y, x + 18, y, fill=col, width=3 if kind == "line" else 1,
                               dash=(4, 3) if kind == "dash" else None)
            x += 22
        t = canvas.create_text(x, y, text=label, anchor="w", fill=fg, font=("TkDefaultFont", 8))
        x = canvas.bbox(t)[2] + 10
