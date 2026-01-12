#![cfg(feature = "cuda")]

pub fn axis_usize(axis: (usize, usize, usize)) -> Vec<usize> {
    let (start, end, step) = axis;
    if step == 0 || start == end {
        return vec![start];
    }
    if start > end {
        return Vec::new();
    }
    (start..=end).step_by(step).collect()
}

pub fn axis_f64(axis: (f64, f64, f64)) -> Vec<f64> {
    let (start, end, step) = axis;
    if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
        return vec![start];
    }
    if start > end {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut v = start;
    let lim = end + step.abs() * 1e-12;
    while v <= lim {
        out.push(v);
        v += step;
    }
    out
}

pub fn normalize_vec_f32(mut weights: Vec<f32>) -> Vec<f32> {
    let sum: f32 = weights.iter().copied().sum();
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for w in &mut weights {
            *w *= inv;
        }
    }
    weights
}

pub fn gen_series(len: usize) -> Vec<f32> {
    let mut out = vec![f32::NAN; len];
    for i in 3..len {
        let x = i as f32;
        out[i] = (x * 0.001).sin() + 0.0001 * x;
    }
    out
}

pub fn gen_volume(len: usize) -> Vec<f32> {
    let mut out = vec![f32::NAN; len];
    for i in 5..len {
        let x = i as f32;
        out[i] = ((x * 0.007).cos().abs() + 1.2) * 950.0;
    }
    out
}

pub fn gen_positive_series(len: usize) -> Vec<f32> {
    let mut out = vec![f32::NAN; len];
    for i in 10..len {
        let x = i as f32 * 0.0017;
        out[i] = 1.35 + 0.55 * x.sin() + 0.25 * x.cos();
    }
    out
}

pub fn gen_time_major_prices(num_series: usize, series_len: usize) -> Vec<f32> {
    let mut out = vec![f32::NAN; num_series * series_len];
    for j in 0..num_series {
        for t in j..series_len {
            let idx = t * num_series + j;
            let x = t as f32 + j as f32 * 0.1;
            out[idx] = (x * 0.003).cos() + 0.001 * x;
        }
    }
    out
}

pub fn gen_time_major_volumes(num_series: usize, series_len: usize) -> Vec<f32> {
    let mut out = vec![f32::NAN; num_series * series_len];
    for j in 0..num_series {
        for t in j..series_len {
            let idx = t * num_series + j;
            let base = t as f32 + j as f32 * 0.2;
            out[idx] = ((base * 0.008).sin().abs() + 0.9) * (300.0 + 20.0 * j as f32);
        }
    }
    out
}

pub fn gen_positive_time_major(num_series: usize, series_len: usize) -> Vec<f32> {
    let mut out = vec![f32::NAN; num_series * series_len];
    for j in 0..num_series {
        for t in (j + 6)..series_len {
            let idx = t * num_series + j;
            let x = t as f32 * 0.0024 + j as f32 * 0.11;
            out[idx] = 1.28 + 0.48 * x.cos() + 0.18 * x.sin();
        }
    }
    out
}
