//! srs_core: native kernels for SpaceRockSuite.
//! Every function has a NumPy twin in `spacerocksuite/core.py` that gives the
//! same result; the Python side falls back to it when this module is absent.

mod filters;
mod kepler;
mod stack;

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type Arr2<'py> = Bound<'py, PyArray2<f64>>;

#[pyfunction]
fn kepler_radec<'py>(
    py: Python<'py>,
    elements: PyReadonlyArray2<'py, f64>,
    jd: PyReadonlyArray1<'py, f64>,
    earth_xyz: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Arr2<'py>, Arr2<'py>, Arr2<'py>)> {
    let el = elements.as_array().to_owned();
    let t = jd.as_array().to_owned();
    let e = earth_xyz.as_array().to_owned();
    if el.ncols() != 6 || e.ncols() != 3 || e.nrows() != t.len() {
        return Err(PyValueError::new_err("elements must be (N,6), earth_xyz (M,3) with M == len(jd)"));
    }
    let (n, m) = (el.nrows(), t.len());
    let rows = py.allow_threads(|| {
        let one = |i: usize| -> Vec<(f64, f64, f64)> {
            let r = el.row(i).to_vec();
            (0..m).map(|k| kepler::radec_one(&r, t[k], &[e[[k, 0]], e[[k, 1]], e[[k, 2]]])).collect()
        };
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            (0..n).into_par_iter().map(one).collect::<Vec<_>>()
        }
        #[cfg(not(feature = "parallel"))]
        {
            (0..n).map(one).collect::<Vec<_>>()
        }
    });
    let mut ra = Array2::<f64>::zeros((n, m));
    let mut dec = Array2::<f64>::zeros((n, m));
    let mut dl = Array2::<f64>::zeros((n, m));
    for (i, r) in rows.iter().enumerate() {
        for (k, v) in r.iter().enumerate() {
            ra[[i, k]] = v.0;
            dec[[i, k]] = v.1;
            dl[[i, k]] = v.2;
        }
    }
    Ok((ra.into_pyarray_bound(py), dec.into_pyarray_bound(py), dl.into_pyarray_bound(py)))
}

#[pyfunction]
#[pyo3(signature = (frames, dx, dy, method="clipmean", clip_sigma=3.0, iters=3))]
fn shift_stack<'py>(
    py: Python<'py>,
    frames: PyReadonlyArray3<'py, f32>,
    dx: PyReadonlyArray1<'py, f64>,
    dy: PyReadonlyArray1<'py, f64>,
    method: &str,
    clip_sigma: f64,
    iters: usize,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<u16>>)> {
    let sh = frames.shape().to_vec();
    let (k, h, w) = (sh[0], sh[1], sh[2]);
    let f: Vec<f32> = frames.as_array().iter().copied().collect();
    let dx = dx.as_array().to_vec();
    let dy = dy.as_array().to_vec();
    if dx.len() != k || dy.len() != k {
        return Err(PyValueError::new_err("dx and dy need one value per frame"));
    }
    let m = method.to_string();
    let (s, c) = py.allow_threads(|| stack::shift_stack(&f, k, h, w, &dx, &dy, &m, clip_sigma, iters));
    let s = Array2::from_shape_vec((h, w), s).unwrap();
    let c = Array2::from_shape_vec((h, w), c).unwrap();
    Ok((s.into_pyarray_bound(py), c.into_pyarray_bound(py)))
}

fn img_vec(img: &PyReadonlyArray2<'_, f32>) -> (Vec<f32>, usize, usize) {
    let sh = img.shape();
    (img.as_array().iter().copied().collect(), sh[0], sh[1])
}

#[pyfunction]
fn gaussian_blur<'py>(py: Python<'py>, img: PyReadonlyArray2<'py, f32>, sigma: f64) -> Bound<'py, PyArray2<f32>> {
    let (v, h, w) = img_vec(&img);
    let out = py.allow_threads(|| filters::gaussian_blur(&v, h, w, sigma));
    Array2::from_shape_vec((h, w), out).unwrap().into_pyarray_bound(py)
}

#[pyfunction]
fn median_filter<'py>(py: Python<'py>, img: PyReadonlyArray2<'py, f32>, size: usize) -> Bound<'py, PyArray2<f32>> {
    let (v, h, w) = img_vec(&img);
    let out = py.allow_threads(|| filters::median_filter(&v, h, w, size));
    Array2::from_shape_vec((h, w), out).unwrap().into_pyarray_bound(py)
}

#[pyfunction]
fn points_in_ellipse<'py>(
    py: Python<'py>,
    dx: PyReadonlyArray1<'py, f64>,
    dy: PyReadonlyArray1<'py, f64>,
    a: f64,
    b: f64,
    theta_deg: f64,
) -> Bound<'py, PyArray1<bool>> {
    let t = theta_deg.to_radians();
    let (st, ct) = (t.sin(), t.cos());
    let out: Vec<bool> = dx
        .as_array()
        .iter()
        .zip(dy.as_array().iter())
        .map(|(&x, &y)| {
            let u = x * st + y * ct;
            let v = x * ct - y * st;
            (u / a).powi(2) + (v / b).powi(2) <= 1.0
        })
        .collect();
    out.into_pyarray_bound(py)
}

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn srs_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(kepler_radec, m)?)?;
    m.add_function(wrap_pyfunction!(shift_stack, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_blur, m)?)?;
    m.add_function(wrap_pyfunction!(median_filter, m)?)?;
    m.add_function(wrap_pyfunction!(points_in_ellipse, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
