//! Two-body propagation with universal variables (elliptic, parabolic, hyperbolic).
//! Mirrors `spacerocksuite.core._helio_xyz` / `kepler_radec`.

pub const GM: f64 = 2.959122082855911e-4; // au^3 / day^2
pub const C_AUD: f64 = 173.1446326742403; // au / day
pub fn eps() -> f64 {
    (84381.448f64 / 3600.0).to_radians()
}

fn stumpff(z: f64) -> (f64, f64) {
    if z > 1e-8 {
        let s = z.sqrt();
        ((1.0 - s.cos()) / z, (s - s.sin()) / (s * s * s))
    } else if z < -1e-8 {
        let s = (-z).sqrt();
        ((s.cosh() - 1.0) / -z, (s.sinh() - s) / (s * s * s))
    } else {
        (0.5 - z / 24.0, 1.0 / 6.0 - z / 120.0)
    }
}

/// Heliocentric equatorial J2000 position at JD `t` for elements
/// [q, e, i, node, peri, tp] (angles in degrees, ecliptic J2000).
pub fn helio_xyz(el: &[f64], t: f64) -> [f64; 3] {
    let (q, e) = (el[0], el[1]);
    let (i, node, peri, tp) = (el[2].to_radians(), el[3].to_radians(), el[4].to_radians(), el[5]);
    let dt = t - tp;
    let alpha = (1.0 - e) / q;
    let sqmu = GM.sqrt();
    let v0 = (GM * (1.0 + e) / q).sqrt();
    let mut chi = if alpha > 1e-9 {
        sqmu * alpha * dt
    } else if alpha < -1e-9 {
        let a = 1.0 / alpha;
        let sg = if dt >= 0.0 { 1.0 } else { -1.0 };
        let v = sg * (-a).sqrt()
            * ((-2.0 * GM * alpha * dt / (sg * (GM * a).abs().sqrt() * (1.0 - q * alpha))).abs() + 1e-300).ln();
        if v.is_finite() { v } else { sqmu * dt / q }
    } else {
        sqmu * dt / q
    };
    for _ in 0..100 {
        let z = alpha * chi * chi;
        let (c2, c3) = stumpff(z);
        let f = q * chi * (1.0 - z * c3) + chi * chi * chi * c3 - sqmu * dt;
        let r = chi * chi * c2 + q * (1.0 - z * c2);
        let mut step = f / r;
        let lim = chi.abs() + 1.0;
        step = step.clamp(-lim, lim);
        chi -= step;
        if step.abs() < 1e-13 * chi.abs().max(1.0) {
            break;
        }
    }
    let z = alpha * chi * chi;
    let (c2, c3) = stumpff(z);
    let fx = 1.0 - chi * chi / q * c2;
    let gx = dt - chi * chi * chi / sqmu * c3;
    let (xp, yp) = (fx * q, gx * v0);
    let (co, so, ci, si, cw, sw) = (node.cos(), node.sin(), i.cos(), i.sin(), peri.cos(), peri.sin());
    let xe = (co * cw - so * sw * ci) * xp + (-co * sw - so * cw * ci) * yp;
    let ye = (so * cw + co * sw * ci) * xp + (-so * sw + co * cw * ci) * yp;
    let ze = (sw * si) * xp + (cw * si) * yp;
    let (ce, se) = (eps().cos(), eps().sin());
    [xe, ye * ce - ze * se, ye * se + ze * ce]
}

/// RA, Dec (deg) and delta (au) of one orbit at one epoch, with light time.
pub fn radec_one(el: &[f64], jd: f64, obs: &[f64]) -> (f64, f64, f64) {
    let mut tau = 0.0;
    let mut d = [0.0; 3];
    let mut delta = 0.0;
    for _ in 0..3 {
        let p = helio_xyz(el, jd - tau);
        d = [p[0] - obs[0], p[1] - obs[1], p[2] - obs[2]];
        delta = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        tau = delta / C_AUD;
    }
    let ra = d[1].atan2(d[0]).to_degrees().rem_euclid(360.0);
    let dec = (d[2] / delta).clamp(-1.0, 1.0).asin().to_degrees();
    (ra, dec, delta)
}
