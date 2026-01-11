

use vector_ta::indicators::rsi::{rsi, RsiInput, RsiParams};
use vector_ta::indicators::srsi::{SrsiBatchRange, SrsiParams};
use std::collections::BTreeMap;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::oscillators::CudaSrsi;

fn approx_eq_f32(a: f32, b: f32, tol: f32) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    (a - b).abs() <= tol
}

#[inline(always)]
fn ftz_f32(x: f32) -> f32 {
    
    if x.abs() < f32::MIN_POSITIVE {
        0.0f32
    } else {
        x
    }
}

fn srsi_from_rsi_f32(
    rsi: &[f32],
    first_valid: usize,
    rsi_period: usize,
    stoch_period: usize,
    k_period: usize,
    d_period: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n = rsi.len();
    let mut out_k = vec![f32::NAN; n];
    let mut out_d = vec![f32::NAN; n];
    if n == 0 || stoch_period == 0 || k_period == 0 || d_period == 0 {
        return (out_k, out_d);
    }

    let rsi_warmup = first_valid.saturating_add(rsi_period);
    if rsi_warmup >= n {
        return (out_k, out_d);
    }
    let stoch_warmup = rsi_warmup.saturating_add(stoch_period - 1);
    let k_warmup = stoch_warmup.saturating_add(k_period - 1);
    let d_warmup = k_warmup.saturating_add(d_period - 1);
    if stoch_warmup >= n {
        return (out_k, out_d);
    }

    let inv_kp = 1.0f32 / (k_period as f32);
    let inv_dp = 1.0f32 / (d_period as f32);

    let mut ring_k = vec![0f32; k_period];
    let mut ring_d = vec![0f32; d_period];
    let mut sum_k = 0.0f32;
    let mut sum_d = 0.0f32;
    let mut head_k = 0usize;
    let mut head_d = 0usize;
    let mut cnt_k = 0usize;
    let mut cnt_d = 0usize;

    for i in stoch_warmup..n {
        let rv = ftz_f32(rsi[i]);
        let start = i + 1 - stoch_period;
        let mut hi = -1e30f32;
        let mut lo = 1e30f32;
        for t in start..=i {
            let v = ftz_f32(rsi[t]);
            hi = hi.max(v);
            lo = lo.min(v);
        }
        let denom = hi - lo;
        
        let fk = if denom >= f32::MIN_POSITIVE {
            (rv - lo) * 100.0f32 / denom
        } else {
            50.0f32
        };

        if cnt_k < k_period {
            sum_k += fk;
            ring_k[head_k] = fk;
            cnt_k += 1;
            head_k += 1;
            if head_k == k_period {
                head_k = 0;
            }
        } else {
            sum_k += fk - ring_k[head_k];
            ring_k[head_k] = fk;
            head_k += 1;
            if head_k == k_period {
                head_k = 0;
            }
        }

        if i >= k_warmup {
            let slow_k = sum_k * inv_kp;
            out_k[i] = slow_k;

            if cnt_d < d_period {
                sum_d += slow_k;
                ring_d[head_d] = slow_k;
                cnt_d += 1;
                head_d += 1;
                if head_d == d_period {
                    head_d = 0;
                }
            } else {
                sum_d += slow_k - ring_d[head_d];
                ring_d[head_d] = slow_k;
                head_d += 1;
                if head_d == d_period {
                    head_d = 0;
                }
            }
            if i >= d_warmup {
                out_d[i] = sum_d * inv_dp;
            }
        }
    }

    (out_k, out_d)
}

fn rsi_wilder_f32(prices: &[f32], first_valid: usize, period: usize) -> Vec<f32> {
    let n = prices.len();
    let mut out = vec![f32::NAN; n];
    if n == 0 || period == 0 {
        return out;
    }

    let warm = first_valid.saturating_add(period);
    if warm >= n {
        return out;
    }

    
    let mut avg_gain = 0.0f32;
    let mut avg_loss = 0.0f32;
    let mut prev = prices[first_valid];
    let last = (first_valid + period).min(n - 1);
    for i in (first_valid + 1)..=last {
        let cur = prices[i];
        let ch = cur - prev;
        prev = cur;
        if ch > 0.0f32 {
            avg_gain += ch;
        } else {
            avg_loss += -ch;
        }
    }

    let inv_p = 1.0f32 / (period as f32);
    avg_gain *= inv_p;
    avg_loss *= inv_p;

    out[warm] = if avg_loss == 0.0f32 {
        100.0f32
    } else {
        100.0f32 - 100.0f32 / (1.0f32 + avg_gain / avg_loss)
    };

    
    let alpha = inv_p;
    let mut prev = prices[warm];
    for i in (warm + 1)..n {
        let cur = prices[i];
        let ch = cur - prev;
        prev = cur;
        let gain = if ch > 0.0f32 { ch } else { 0.0f32 };
        let loss = if ch < 0.0f32 { -ch } else { 0.0f32 };
        avg_gain = (gain - avg_gain).mul_add(alpha, avg_gain);
        avg_loss = (loss - avg_loss).mul_add(alpha, avg_loss);
        out[i] = if avg_loss == 0.0f32 {
            100.0f32
        } else {
            100.0f32 - 100.0f32 / (1.0f32 + avg_gain / avg_loss)
        };
    }

    out
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
fn srsi_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[srsi_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let len = 8192usize;
    let mut price = vec![f64::NAN; len];
    for i in 8..len {
        let x = i as f64;
        price[i] = (x * 0.00123).sin() + 0.00031 * x.cos();
    }
    let sweep = SrsiBatchRange {
        rsi_period: (4, 22, 3),
        stoch_period: (4, 20, 4),
        k: (3, 5, 1),
        d: (3, 5, 1),
    };

    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let first_valid = price_f32.iter().position(|v| !v.is_nan()).unwrap_or(len);
    let price_f64: Vec<f64> = price_f32.iter().map(|&v| v as f64).collect();

    let cuda = CudaSrsi::new(0).expect("CudaSrsi::new");
    let (dev_pair, combos) = cuda
        .srsi_batch_dev(&price_f32, &sweep)
        .expect("srsi_cuda_batch_dev");
    assert_eq!(dev_pair.k.rows, combos.len());
    assert_eq!(dev_pair.k.cols, len);
    assert_eq!(dev_pair.d.rows, combos.len());
    assert_eq!(dev_pair.d.cols, len);

    let mut gk = vec![0f32; dev_pair.k.len()];
    let mut gd = vec![0f32; dev_pair.d.len()];
    dev_pair.k.buf.copy_to(&mut gk)?;
    dev_pair.d.buf.copy_to(&mut gd)?;

    
    
    let mut rsi_cache: BTreeMap<usize, Vec<f32>> = BTreeMap::new();
    for prm in &combos {
        let rp = prm.rsi_period.unwrap();
        if rsi_cache.contains_key(&rp) {
            continue;
        }
        let rsi_out = rsi(&RsiInput::from_slice(&price_f64, RsiParams { period: Some(rp) }))?;
        if rp == 4 {
            eprintln!(
                "[srsi_cuda_batch][debug] rsi_f64 sample: idx12={} idx20={} idx1771={} idx6874={}",
                rsi_out.values[12],
                rsi_out.values[20],
                rsi_out.values[1771],
                rsi_out.values[6874]
            );
        }
        let rsi_f32: Vec<f32> = rsi_out.values.iter().map(|&v| v as f32).collect();
        rsi_cache.insert(rp, rsi_f32);
    }

    let mut cpu_k = vec![f32::NAN; combos.len() * len];
    let mut cpu_d = vec![f32::NAN; combos.len() * len];
    for (row, prm) in combos.iter().enumerate() {
        let rp = prm.rsi_period.unwrap();
        let sp = prm.stoch_period.unwrap();
        let kp = prm.k.unwrap();
        let dp = prm.d.unwrap();
        let rsi_vals = rsi_cache.get(&rp).expect("cached rsi");
        let (k_row, d_row) = srsi_from_rsi_f32(rsi_vals, first_valid, rp, sp, kp, dp);
        let off = row * len;
        cpu_k[off..(off + len)].copy_from_slice(&k_row);
        cpu_d[off..(off + len)].copy_from_slice(&d_row);
    }

    let tol = 1e-2f32;
    let mut max_k = 0.0f32;
    let mut max_d = 0.0f32;
    let mut max_ki = 0usize;
    let mut max_di = 0usize;
    for idx in 0..gk.len() {
        let dk = (cpu_k[idx] - gk[idx]).abs();
        if dk > max_k {
            max_k = dk;
            max_ki = idx;
        }
        let dd = (cpu_d[idx] - gd[idx]).abs();
        if dd > max_d {
            max_d = dd;
            max_di = idx;
        }
    }
    eprintln!("[srsi_cuda_batch] max |K| diff = {:.6e} @{}", max_k, max_ki);
    eprintln!("[srsi_cuda_batch] max |D| diff = {:.6e} @{}", max_d, max_di);
    for idx in 0..gk.len() {
        if !approx_eq_f32(cpu_k[idx], gk[idx], tol) {
            let row = idx / len;
            let t = idx % len;
            let base = row * len;
            for dt in [-2isize, -1, 0, 1, 2] {
                let ti = t as isize + dt;
                if ti >= 0 && (ti as usize) < len {
                    let j = base + (ti as usize);
                    eprintln!(
                        "[srsi_cuda_batch][debug] row={} t={} cpu_k={} gpu_k={}",
                        row,
                        ti,
                        cpu_k[j],
                        gk[j]
                    );
                }
            }
            {
                let prm = &combos[row];
                let rp = prm.rsi_period.unwrap();
                let sp = prm.stoch_period.unwrap();
                let rsi_vals = rsi_cache.get(&rp).expect("cached rsi");
                let w_start = t + 1 - sp;
                let win = &rsi_vals[w_start..(t + 1)];
                let mut hi = -1e30f32;
                let mut lo = 1e30f32;
                for &v in win {
                    hi = hi.max(v);
                    lo = lo.min(v);
                }
                eprintln!(
                    "[srsi_cuda_batch][debug] rsi_window={:?} hi={} lo={} rv={}",
                    win,
                    hi,
                    lo,
                    rsi_vals[t]
                );
            }
            panic!(
                "K mismatch at idx={} (row={}, t={}) params={:?} cpu={} gpu={}",
                idx,
                row,
                t,
                combos[row],
                cpu_k[idx],
                gk[idx]
            );
        }
        if !approx_eq_f32(cpu_d[idx], gd[idx], tol) {
            let row = idx / len;
            let t = idx % len;
            panic!(
                "D mismatch at idx={} (row={}, t={}) params={:?} cpu={} gpu={}",
                idx,
                row,
                t,
                combos[row],
                cpu_d[idx],
                gd[idx]
            );
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn srsi_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[srsi_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let cols = 8usize;
    let rows = 2048usize;
    let mut prices_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let idx = t * cols + s;
            let x = (t as f64) * 0.002 + (s as f64) * 0.01;
            prices_tm[idx] = (x.sin() * 0.7 + x * 0.0009).into();
        }
    }
    let rp = 14usize;
    let sp = 14usize;
    let kp = 3usize;
    let dp = 3usize;

    
    let prices_tm_f32: Vec<f32> = prices_tm.iter().map(|&v| v as f32).collect();
    let mut cpu_k = vec![f32::NAN; cols * rows];
    let mut cpu_d = vec![f32::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f32::NAN; rows];
        for t in 0..rows {
            series[t] = prices_tm_f32[t * cols + s];
        }
        let first = series.iter().position(|v| !v.is_nan()).unwrap_or(rows);
        let rsi_vals = rsi_wilder_f32(&series, first, rp);
        let (k_row, d_row) = srsi_from_rsi_f32(&rsi_vals, first, rp, sp, kp, dp);
        for t in 0..rows {
            cpu_k[t * cols + s] = k_row[t];
            cpu_d[t * cols + s] = d_row[t];
        }
    }

    let cuda = CudaSrsi::new(0).expect("CudaSrsi::new");
    let dev_pair = cuda
        .srsi_many_series_one_param_time_major_dev(
            &prices_tm_f32,
            cols,
            rows,
            &SrsiParams {
                rsi_period: Some(rp),
                stoch_period: Some(sp),
                k: Some(kp),
                d: Some(dp),
                source: None,
            },
        )
        .expect("srsi many-series");

    assert_eq!(dev_pair.k.rows, rows); assert_eq!(dev_pair.k.cols, cols);
    assert_eq!(dev_pair.d.rows, rows); assert_eq!(dev_pair.d.cols, cols);
    let mut gk = vec![0f32; dev_pair.k.len()]; let mut gd = vec![0f32; dev_pair.d.len()];
    dev_pair.k.buf.copy_to(&mut gk)?; dev_pair.d.buf.copy_to(&mut gd)?;
    let tol = 1e-2f32;
    let mut max_k = 0.0f32;
    let mut max_d = 0.0f32;
    let mut max_ki = 0usize;
    let mut max_di = 0usize;
    for i in 0..gk.len() {
        let dk = (cpu_k[i] - gk[i]).abs();
        if dk > max_k {
            max_k = dk;
            max_ki = i;
        }
        let dd = (cpu_d[i] - gd[i]).abs();
        if dd > max_d {
            max_d = dd;
            max_di = i;
        }
    }
    eprintln!("[srsi_cuda_many] max |K| diff = {:.6e} @{}", max_k, max_ki);
    eprintln!("[srsi_cuda_many] max |D| diff = {:.6e} @{}", max_d, max_di);
    for i in 0..gk.len() {
        if !approx_eq_f32(cpu_k[i], gk[i], tol) {
            let t = i / cols;
            let s = i % cols;
            panic!("K mismatch at i={} (t={}, s={})", i, t, s);
        }
        if !approx_eq_f32(cpu_d[i], gd[i], tol) {
            let t = i / cols;
            let s = i % cols;
            panic!("D mismatch at i={} (t={}, s={})", i, t, s);
        }
    }
    Ok(())
}
