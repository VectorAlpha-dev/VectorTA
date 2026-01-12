use vector_ta::indicators::stc::{StcBatchRange, StcParams};

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::oscillators::CudaStc;

fn approx_eq_f32(a: f32, b: f32, tol: f32) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    (a - b).abs() <= tol
}

#[derive(Clone, Copy)]
struct KahanF32 {
    s: f32,
    c: f32,
}

impl KahanF32 {
    #[inline(always)]
    fn new() -> Self {
        Self { s: 0.0, c: 0.0 }
    }

    #[inline(always)]
    fn add(&mut self, x: f32) {
        let t = self.s + x;
        if self.s.abs() >= x.abs() {
            self.c += (self.s - t) + x;
        } else {
            self.c += (x - t) + self.s;
        }
        self.s = t;
    }

    #[inline(always)]
    fn result(self) -> f32 {
        self.s + self.c
    }
}

#[inline(always)]
fn ema_update_f32(prev: f32, a: f32, x: f32) -> f32 {
    (x - prev).mul_add(a, prev)
}

fn stc_series_f32(
    prices: &[f32],
    first_valid: usize,
    fast: usize,
    slow: usize,
    k: usize,
    d: usize,
) -> Vec<f32> {
    const HUNDRED: f32 = 100.0;
    const STC_RANGE_EPS: f32 = 2.2204460492503131e-16;

    let n = prices.len();
    let mut out = vec![f32::NAN; n];
    if n == 0 || fast == 0 || slow == 0 || k == 0 || d == 0 || first_valid >= n {
        return out;
    }

    let max_needed = fast.max(slow).max(k.max(d));
    let warm = first_valid.saturating_add(max_needed).saturating_sub(1);
    if warm >= n {
        return out;
    }

    let fast_a = 2.0f32 / (fast as f32 + 1.0f32);
    let slow_a = 2.0f32 / (slow as f32 + 1.0f32);
    let d_a = 2.0f32 / (d as f32 + 1.0f32);

    let mut fast_acc = KahanF32::new();
    let mut slow_acc = KahanF32::new();
    let mut fast_seed_nan = false;
    let mut slow_seed_nan = false;
    let f_end = fast.min(n - first_valid);
    let s_end = slow.min(n - first_valid);
    for i in 0..f_end {
        let v = prices[first_valid + i];
        if !v.is_finite() {
            fast_seed_nan = true;
            break;
        }
        fast_acc.add(v);
    }
    for i in 0..s_end {
        let v = prices[first_valid + i];
        if !v.is_finite() {
            slow_seed_nan = true;
            break;
        }
        slow_acc.add(v);
    }
    let mut fast_ema = if f_end == fast && !fast_seed_nan {
        fast_acc.result() / (fast as f32)
    } else {
        f32::NAN
    };
    let mut slow_ema = if s_end == slow && !slow_seed_nan {
        slow_acc.result() / (slow as f32)
    } else {
        f32::NAN
    };

    let mut macd_ring = vec![f32::NAN; k];
    let mut d_ring = vec![f32::NAN; k];
    let mut macd_run = 0usize;
    let mut d_run = 0usize;

    let mut d_seed_acc = KahanF32::new();
    let mut d_seed_cnt = 0usize;
    let mut d_ema = f32::NAN;

    let mut final_seed_acc = KahanF32::new();
    let mut final_seed_cnt = 0usize;
    let mut final_ema = f32::NAN;

    let fast_thr = fast - 1;
    let slow_thr = slow - 1;

    for i in 0..n {
        let x = prices[i];

        if i >= first_valid {
            let rel = i - first_valid;
            if rel >= fast_thr && rel != fast_thr {
                if x.is_finite() && fast_ema.is_finite() {
                    fast_ema = ema_update_f32(fast_ema, fast_a, x);
                } else {
                    fast_ema = f32::NAN;
                }
            }
            if rel >= slow_thr && rel != slow_thr {
                if x.is_finite() && slow_ema.is_finite() {
                    slow_ema = ema_update_f32(slow_ema, slow_a, x);
                } else {
                    slow_ema = f32::NAN;
                }
            }
        }

        let (macd, macd_valid) =
            if i >= first_valid + slow_thr && fast_ema.is_finite() && slow_ema.is_finite() {
                (fast_ema - slow_ema, true)
            } else {
                (f32::NAN, false)
            };

        let stok = if macd_valid {
            macd_ring[i % k] = macd;
            macd_run += 1;
            if macd_run >= k {
                let start = i + 1 - k;
                let mut mn = macd_ring[start % k];
                let mut mx = mn;
                for t in start..=i {
                    let v = macd_ring[t % k];
                    mn = mn.min(v);
                    mx = mx.max(v);
                }
                let range = mx - mn;
                if range.abs() > STC_RANGE_EPS {
                    (macd - mn) * (HUNDRED / range)
                } else {
                    50.0f32
                }
            } else {
                50.0f32
            }
        } else {
            macd_run = 0;
            f32::NAN
        };

        let d_val = if stok.is_finite() {
            if d_seed_cnt < d {
                d_seed_acc.add(stok);
                d_seed_cnt += 1;
                let sum = d_seed_acc.result();
                if d_seed_cnt == d {
                    d_ema = sum / (d as f32);
                    d_ema
                } else {
                    sum / (d_seed_cnt as f32)
                }
            } else {
                d_ema = ema_update_f32(d_ema, d_a, stok);
                d_ema
            }
        } else if d_seed_cnt == 0 {
            f32::NAN
        } else if d_seed_cnt < d {
            d_seed_acc.result() / (d_seed_cnt as f32)
        } else {
            d_ema
        };

        let kd = if d_val.is_finite() {
            d_ring[i % k] = d_val;
            d_run += 1;
            if d_run >= k {
                let start = i + 1 - k;
                let mut mn = d_ring[start % k];
                let mut mx = mn;
                for t in start..=i {
                    let v = d_ring[t % k];
                    mn = mn.min(v);
                    mx = mx.max(v);
                }
                let range = mx - mn;
                if range.abs() > STC_RANGE_EPS {
                    (d_val - mn) * (HUNDRED / range)
                } else {
                    50.0f32
                }
            } else {
                50.0f32
            }
        } else {
            d_run = 0;
            f32::NAN
        };

        let out_i = if kd.is_finite() {
            if final_seed_cnt < d {
                final_seed_acc.add(kd);
                final_seed_cnt += 1;
                let sum = final_seed_acc.result();
                if final_seed_cnt == d {
                    final_ema = sum / (d as f32);
                    final_ema
                } else {
                    sum / (final_seed_cnt as f32)
                }
            } else {
                final_ema = ema_update_f32(final_ema, d_a, kd);
                final_ema
            }
        } else if final_seed_cnt == 0 {
            f32::NAN
        } else if final_seed_cnt < d {
            final_seed_acc.result() / (final_seed_cnt as f32)
        } else {
            final_ema
        };

        if i >= warm {
            out[i] = out_i;
        }
    }

    out
}

#[test]
fn cuda_feature_off_noop_stc() {
    #[cfg(not(feature = "cuda"))]
    {
        assert!(true);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn stc_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[stc_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut price = vec![f64::NAN; len];
    for i in 60..len {
        let x = i as f64;
        price[i] = (x * 0.00123).sin() + 0.00017 * x;
    }

    let sweep = StcBatchRange {
        fast_period: (10, 20, 5),
        slow_period: (30, 60, 10),
        k_period: (10, 10, 0),
        d_period: (3, 3, 0),
    };

    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let first_valid = price_f32.iter().position(|v| v.is_finite()).unwrap_or(len);
    let cuda = CudaStc::new(0).expect("CudaStc::new");
    let (dev, combos) = cuda
        .stc_batch_dev(&price_f32, &sweep)
        .expect("stc_batch_dev");

    assert_eq!(dev.rows, combos.len());
    assert_eq!(dev.cols, len);
    assert_eq!(combos.len(), 12);

    let mut g = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut g)?;

    let mut cpu = vec![f32::NAN; combos.len() * len];
    for (row, prm) in combos.iter().enumerate() {
        let fast = prm.fast_period.unwrap();
        let slow = prm.slow_period.unwrap();
        let k = prm.k_period.unwrap();
        let d = prm.d_period.unwrap();
        let row_out = stc_series_f32(&price_f32, first_valid, fast, slow, k, d);
        cpu[row * len..(row + 1) * len].copy_from_slice(&row_out);
    }

    let tol = 1.5e-3f32;
    for idx in 0..g.len() {
        assert!(
            approx_eq_f32(cpu[idx], g[idx], tol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            cpu[idx],
            g[idx]
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn stc_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[stc_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize;
    let rows = 2048usize;
    let mut tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in 40..rows {
            let x = (t as f64) + (s as f64) * 0.1;
            tm[t * cols + s] = (x * 0.0019).sin() + 0.00011 * x;
        }
    }

    let params = StcParams {
        fast_period: Some(23),
        slow_period: Some(50),
        k_period: Some(10),
        d_period: Some(3),
        fast_ma_type: Some("ema".to_string()),
        slow_ma_type: Some("ema".to_string()),
    };

    let tm_f32: Vec<f32> = tm.iter().map(|&v| v as f32).collect();

    let mut cpu_tm = vec![f32::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f32::NAN; rows];
        for t in 0..rows {
            series[t] = tm_f32[t * cols + s];
        }
        let first_valid = series.iter().position(|v| v.is_finite()).unwrap_or(rows);
        let out = stc_series_f32(
            &series,
            first_valid,
            params.fast_period.unwrap(),
            params.slow_period.unwrap(),
            params.k_period.unwrap(),
            params.d_period.unwrap(),
        );
        for t in 0..rows {
            cpu_tm[t * cols + s] = out[t];
        }
    }

    let cuda = CudaStc::new(0).expect("CudaStc::new");
    let dev_tm = cuda
        .stc_many_series_one_param_time_major_dev(&tm_f32, cols, rows, &params)
        .expect("stc many-series");

    assert_eq!(dev_tm.rows, rows);
    assert_eq!(dev_tm.cols, cols);

    let mut g_tm = vec![0f32; dev_tm.len()];
    dev_tm.buf.copy_to(&mut g_tm)?;

    let tol = 2e-3f32;
    for idx in 0..g_tm.len() {
        assert!(
            approx_eq_f32(cpu_tm[idx], g_tm[idx], tol),
            "mismatch at {}",
            idx
        );
    }

    Ok(())
}
