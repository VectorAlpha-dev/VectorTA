// Integration tests for CUDA DTI kernels

use my_project::indicators::dti::{DtiBatchBuilder, DtiBatchRange, DtiParams};

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::oscillators::CudaDti;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() { return true; }
    (a - b).abs() <= tol
}

#[test]
fn cuda_feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    {
        assert!(true);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn dti_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[dti_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 12_345usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    for i in 1..len {
        let x = i as f64;
        // synthetic, but keep high >= low
        high[i] = (x * 0.00123).sin() + 0.00021 * x + 100.0;
        low[i]  = high[i] - (0.9 + (x * 0.00077).cos().abs());
    }
    let sweep = DtiBatchRange { r: (8, 16, 2), s: (6, 12, 2), u: (3, 9, 2) };

    let cpu = {
        // Build a simple f32 CPU baseline matching the GPU math (precomputed x/ax, f32 EMA)
        let mut rows_vals = Vec::new();
        let combos = {
            let rr = axis_usize(sweep.r);
            let ss = axis_usize(sweep.s);
            let uu = axis_usize(sweep.u);
            let mut c = Vec::new();
            for &r in &rr { for &s in &ss { for &u in &uu { c.push(DtiParams { r: Some(r), s: Some(s), u: Some(u) }); } } }
            c
        };
        let len = high.len();
        let first_valid = (0..len).find(|&i| !high[i].is_nan() && !low[i].is_nan()).unwrap();
        let start = first_valid + 1;
        let hf: Vec<f32> = high.iter().map(|&v| v as f32).collect();
        let lf: Vec<f32> = low.iter().map(|&v| v as f32).collect();
        let (x, ax) = precompute_x_ax(&hf, &lf, start);
        let rows = combos.len();
        let cols = len;
        rows_vals.resize(rows * cols, f64::NAN);
        for (row, prm) in combos.iter().enumerate() {
            let (r, s, u) = (prm.r.unwrap(), prm.s.unwrap(), prm.u.unwrap());
            let (ar, as_, au) = (2.0/(r as f32 + 1.0), 2.0/(s as f32 + 1.0), 2.0/(u as f32 + 1.0));
            let (br, bs, bu) = (1.0 - ar, 1.0 - as_, 1.0 - au);
            let mut e0r=0f32; let mut e0s=0f32; let mut e0u=0f32;
            let mut e1r=0f32; let mut e1s=0f32; let mut e1u=0f32;
            // prefix NaN
            for i in 0..start { rows_vals[row*cols + i] = f64::NAN; }
            for i in start..len {
                let xi = x[i]; let axi = ax[i];
                e0r = ar.mul_add(xi - e0r, e0r);
                e0s = as_.mul_add(e0r - e0s, e0s);
                e0u = au.mul_add(e0s - e0u, e0u);
                e1r = ar.mul_add(axi - e1r, e1r);
                e1s = as_.mul_add(e1r - e1s, e1s);
                e1u = au.mul_add(e1s - e1u, e1u);
                rows_vals[row*cols + i] = if e1u != 0.0 && !e1u.is_nan() { 100.0 * (e0u as f64) / (e1u as f64) } else { 0.0 };
            }
        }
        struct CpuOut { values: Vec<f64>, rows: usize, cols: usize, combos: Vec<DtiParams> }
        CpuOut { values: rows_vals, rows: rows, cols: cols, combos }
    };

    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let cuda = CudaDti::new(0).expect("CudaDti::new");
    let (dev, combos) = cuda
        .dti_batch_dev(&high_f32, &low_f32, &sweep)
        .expect("cuda dti_batch_dev");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);
    assert_eq!(cpu.combos.len(), combos.len());

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 1e-3; // f32 device vs f64 host
    for idx in 0..(cpu.rows * cpu.cols) {
        let c = cpu.values[idx];
        let g = host[idx] as f64;
        assert!(
            approx_eq(c, g, tol),
            "DTI mismatch at {}: cpu={} gpu={}",
            idx,
            c,
            g
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn dti_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[dti_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 16usize;
    let rows = 8192usize;
    let mut high_tm = vec![f64::NAN; cols * rows];
    let mut low_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in (s % 3)..rows { // stagger first_valid
            let x = (t as f64) + (s as f64) * 0.17;
            high_tm[t * cols + s] = (x * 0.0021).sin() + 0.00029 * x + 100.0 + s as f64 * 0.01;
            low_tm [t * cols + s] = high_tm[t * cols + s] - (1.0 + (x * 0.0011).cos().abs());
        }
    }

    let r = 14usize; let s = 10usize; let u = 5usize;

    // CPU f32 baseline (time-major), replicating GPU math.
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    let (ar, as_, au) = (2.0/(r as f32 + 1.0), 2.0/(s as f32 + 1.0), 2.0/(u as f32 + 1.0));
    let (br, bs, bu) = (1.0 - ar, 1.0 - as_, 1.0 - au);
    let hf: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let lf: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    for series in 0..cols {
        // find first_valid
        let mut fv = rows;
        for t in 0..rows {
            let h = hf[t * cols + series]; let l = lf[t * cols + series];
            if !h.is_nan() && !l.is_nan() { fv = t; break; }
        }
        let start = fv + 1;
        let mut e0r=0f32; let mut e0s=0f32; let mut e0u=0f32;
        let mut e1r=0f32; let mut e1s=0f32; let mut e1u=0f32;
        // prefix NaN
        for t in 0..start.min(rows) { cpu_tm[t*cols + series] = f64::NAN; }
        if start >= rows { continue; }
        let mut prev_h = hf[fv * cols + series];
        let mut prev_l = lf[fv * cols + series];
        for t in start..rows {
            let h = hf[t * cols + series];
            let l = lf[t * cols + series];
            let dh = h - prev_h; let dl = l - prev_l; prev_h = h; prev_l = l;
            let x_hmu = if dh > 0.0 { dh } else { 0.0 };
            let x_lmd = if dl < 0.0 { -dl } else { 0.0 };
            let xi = x_hmu - x_lmd; let axi = xi.abs();
            e0r = ar.mul_add(xi - e0r, e0r);
            e0s = as_.mul_add(e0r - e0s, e0s);
            e0u = au.mul_add(e0s - e0u, e0u);
            e1r = ar.mul_add(axi - e1r, e1r);
            e1s = as_.mul_add(e1r - e1s, e1s);
            e1u = au.mul_add(e1s - e1u, e1u);
            cpu_tm[t * cols + series] = if e1u != 0.0 && !e1u.is_nan() { 100.0 * (e0u as f64) / (e1u as f64) } else { 0.0 };
        }
    }

    let high_tm_f32: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let low_tm_f32: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaDti::new(0).expect("CudaDti::new");
    let dev_tm = cuda
        .dti_many_series_one_param_time_major_dev(&high_tm_f32, &low_tm_f32, cols, rows, &DtiParams { r: Some(r), s: Some(s), u: Some(u) })
        .expect("dti_many_series_one_param_time_major_dev");

    assert_eq!(dev_tm.rows, rows);
    assert_eq!(dev_tm.cols, cols);
    let mut g_tm = vec![0f32; dev_tm.len()];
    dev_tm.buf.copy_to(&mut g_tm)?;

    let tol = 1e-3;
    for idx in 0..g_tm.len() {
        assert!(approx_eq(cpu_tm[idx], g_tm[idx] as f64, tol), "many-series mismatch at {}", idx);
    }
    Ok(())
}

fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
    if step == 0 || start == end { return vec![start]; }
    (start..=end).step_by(step).collect()
}

fn precompute_x_ax(high: &[f32], low: &[f32], start: usize) -> (Vec<f32>, Vec<f32>) {
    let len = high.len();
    let mut x = vec![0f32; len];
    let mut ax = vec![0f32; len];
    if start == 0 || start >= len { return (x, ax); }
    for i in start..len {
        let dh = high[i] - high[i - 1];
        let dl = low[i] - low[i - 1];
        let x_hmu = if dh > 0.0 { dh } else { 0.0 };
        let x_lmd = if dl < 0.0 { -dl } else { 0.0 };
        let v = x_hmu - x_lmd;
        x[i] = v; ax[i] = v.abs();
    }
    (x, ax)
}
