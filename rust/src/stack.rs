//! Shift-and-add stacking. Mirrors `spacerocksuite.core.shift_stack` / `combine`.

/// Bilinear sample of frame `f` at (x - dx, y - dy); NaN outside.
fn shifted(f: &[f32], h: usize, w: usize, dx: f64, dy: f64, y: usize, x: usize) -> f32 {
    let sx = x as f64 - dx;
    let sy = y as f64 - dy;
    let x0 = sx.floor();
    let y0 = sy.floor();
    let (fx, fy) = (sx - x0, sy - y0);
    let (x0, y0) = (x0 as isize, y0 as isize);
    if fx == 0.0 && fy == 0.0 {
        if x0 >= 0 && y0 >= 0 && (x0 as usize) < w && (y0 as usize) < h {
            return f[y0 as usize * w + x0 as usize];
        }
        return f32::NAN;
    }
    if x0 < 0 || y0 < 0 || x0 as usize + 1 >= w || y0 as usize + 1 >= h {
        return f32::NAN;
    }
    let (x0, y0) = (x0 as usize, y0 as usize);
    let p = |yy: usize, xx: usize| f[yy * w + xx] as f64;
    (p(y0, x0) * (1.0 - fx) * (1.0 - fy)
        + p(y0, x0 + 1) * fx * (1.0 - fy)
        + p(y0 + 1, x0) * (1.0 - fx) * fy
        + p(y0 + 1, x0 + 1) * fx * fy) as f32
}

fn combine_pixel(vals: &mut Vec<f32>, method: &str, clip: f64, iters: usize) -> (f32, u16) {
    vals.retain(|v| v.is_finite());
    if vals.is_empty() {
        return (f32::NAN, 0);
    }
    match method {
        "mean" => {
            let s: f64 = vals.iter().map(|&v| v as f64).sum();
            ((s / vals.len() as f64) as f32, vals.len() as u16)
        }
        "median" => {
            let n = vals.len() as u16;
            (crate::filters::nanmedian(vals), n)
        }
        _ => {
            for _ in 0..iters {
                let med = crate::filters::nanmedian(&mut vals.clone()) as f64;
                let mut dev: Vec<f32> = vals.iter().map(|&v| ((v as f64 - med).abs()) as f32).collect();
                let mad = 1.4826 * crate::filters::nanmedian(&mut dev) as f64;
                let thr = if mad > 0.0 { clip * mad } else { f64::INFINITY };
                let before = vals.len();
                vals.retain(|&v| (v as f64 - med).abs() <= thr);
                if vals.len() == before {
                    break;
                }
            }
            let s: f64 = vals.iter().map(|&v| v as f64).sum();
            ((s / vals.len() as f64) as f32, vals.len() as u16)
        }
    }
}

pub fn shift_stack(
    frames: &[f32], k: usize, h: usize, w: usize, dx: &[f64], dy: &[f64], method: &str, clip: f64, iters: usize,
) -> (Vec<f32>, Vec<u16>) {
    let row = |y: usize| -> (Vec<f32>, Vec<u16>) {
        let mut out = vec![0f32; w];
        let mut cov = vec![0u16; w];
        let mut vals = Vec::with_capacity(k);
        for x in 0..w {
            vals.clear();
            for i in 0..k {
                vals.push(shifted(&frames[i * h * w..(i + 1) * h * w], h, w, dx[i], dy[i], y, x));
            }
            let (v, c) = combine_pixel(&mut vals, method, clip, iters);
            out[x] = v;
            cov[x] = c;
        }
        (out, cov)
    };
    #[cfg(feature = "parallel")]
    let rows: Vec<(Vec<f32>, Vec<u16>)> = {
        use rayon::prelude::*;
        (0..h).into_par_iter().map(row).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let rows: Vec<(Vec<f32>, Vec<u16>)> = (0..h).map(row).collect();
    let mut s = Vec::with_capacity(h * w);
    let mut c = Vec::with_capacity(h * w);
    for (a, b) in rows {
        s.extend(a);
        c.extend(b);
    }
    (s, c)
}
