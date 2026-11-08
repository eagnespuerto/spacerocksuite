//! Image filters. Mirrors `spacerocksuite.core.gaussian_blur` / `median_filter`.

/// numpy 'reflect' padding index (edge pixel not repeated).
#[inline]
pub fn reflect(i: isize, n: usize) -> usize {
    let n = n as isize;
    if n == 1 {
        return 0;
    }
    let p = 2 * (n - 1);
    let mut k = i.rem_euclid(p);
    if k >= n {
        k = p - k;
    }
    k as usize
}

pub fn gaussian_blur(img: &[f32], h: usize, w: usize, sigma: f64) -> Vec<f32> {
    let r = (4.0 * sigma).ceil() as isize;
    let mut k: Vec<f64> = (-r..=r).map(|x| (-0.5 * (x as f64 / sigma).powi(2)).exp()).collect();
    let s: f64 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= s);
    let val: Vec<f64> = img.iter().map(|&v| if v.is_finite() { v as f64 } else { 0.0 }).collect();
    let wt: Vec<f64> = img.iter().map(|&v| if v.is_finite() { 1.0 } else { 0.0 }).collect();
    let pass = |a: &[f64], vertical: bool| -> Vec<f64> {
        let mut out = vec![0.0; h * w];
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0.0;
                for (j, kj) in k.iter().enumerate() {
                    let o = j as isize - r;
                    let (yy, xx) = if vertical {
                        (reflect(y as isize + o, h), x)
                    } else {
                        (y, reflect(x as isize + o, w))
                    };
                    acc += kj * a[yy * w + xx];
                }
                out[y * w + x] = acc;
            }
        }
        out
    };
    let num = pass(&pass(&val, true), false);
    let den = pass(&pass(&wt, true), false);
    num.iter()
        .zip(den.iter())
        .map(|(n, d)| if *d < 1e-6 { f32::NAN } else { (n / d) as f32 })
        .collect()
}

/// Median of finite values (numpy nanmedian semantics); NaN if none.
pub fn nanmedian(buf: &mut Vec<f32>) -> f32 {
    buf.retain(|v| v.is_finite());
    let n = buf.len();
    if n == 0 {
        return f32::NAN;
    }
    buf.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    if n % 2 == 1 {
        buf[n / 2]
    } else {
        ((buf[n / 2 - 1] as f64 + buf[n / 2] as f64) / 2.0) as f32
    }
}

fn window_median(img: &[f32], h: usize, w: usize, cy: usize, cx: usize, r: usize, buf: &mut Vec<f32>) -> f32 {
    buf.clear();
    for dy in 0..(2 * r + 1) {
        let y = reflect(cy as isize + dy as isize - r as isize, h);
        for dx in 0..(2 * r + 1) {
            let x = reflect(cx as isize + dx as isize - r as isize, w);
            buf.push(img[y * w + x]);
        }
    }
    nanmedian(buf)
}

fn interp_index(n: usize, grid: &[usize]) -> Vec<f64> {
    // np.interp(arange(n), grid, arange(len(grid)))
    (0..n)
        .map(|i| {
            let x = i as f64;
            let j = grid.partition_point(|&g| (g as f64) <= x).max(1) - 1;
            if j + 1 >= grid.len() {
                return (grid.len() - 1) as f64;
            }
            let (g0, g1) = (grid[j] as f64, grid[j + 1] as f64);
            j as f64 + (x - g0) / (g1 - g0)
        })
        .collect()
}

pub fn median_filter(img: &[f32], h: usize, w: usize, size: usize) -> Vec<f32> {
    let size = size.max(1) | 1;
    let r = size / 2;
    let mut buf = Vec::with_capacity(size * size);
    if size < 15 {
        let mut out = vec![0f32; h * w];
        for y in 0..h {
            for x in 0..w {
                out[y * w + x] = window_median(img, h, w, y, x, r, &mut buf);
            }
        }
        return out;
    }
    let step = (size / 4).max(1);
    let mut ys: Vec<usize> = (0..h).step_by(step).collect();
    if *ys.last().unwrap() != h - 1 {
        ys.push(h - 1);
    }
    let mut xs: Vec<usize> = (0..w).step_by(step).collect();
    if *xs.last().unwrap() != w - 1 {
        xs.push(w - 1);
    }
    // grid median of window whose top-left in padded coords is (y, x) == centred on (y, x)
    let mut grid = vec![0f32; ys.len() * xs.len()];
    for (a, &y) in ys.iter().enumerate() {
        for (b, &x) in xs.iter().enumerate() {
            grid[a * xs.len() + b] = window_median(img, h, w, y, x, r, &mut buf);
        }
    }
    let yi = interp_index(h, &ys);
    let xi = interp_index(w, &xs);
    let (ny, nx) = (ys.len(), xs.len());
    let mut out = vec![0f32; h * w];
    for y in 0..h {
        let y0 = (yi[y].floor() as usize).min(ny - 1);
        let y1 = (y0 + 1).min(ny - 1);
        let fy = yi[y] - y0 as f64;
        for x in 0..w {
            let x0 = (xi[x].floor() as usize).min(nx - 1);
            let x1 = (x0 + 1).min(nx - 1);
            let fx = xi[x] - x0 as f64;
            let g = |a: usize, b: usize| grid[a * nx + b] as f64;
            out[y * w + x] = (g(y0, x0) * (1.0 - fy) * (1.0 - fx)
                + g(y1, x0) * fy * (1.0 - fx)
                + g(y0, x1) * (1.0 - fy) * fx
                + g(y1, x1) * fy * fx) as f32;
        }
    }
    out
}
