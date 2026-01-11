// Integration tests for CUDA HalfTrend kernels

use my_project::indicators::halftrend::HalfTrendBatchRange;
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::{cuda_available, CudaHalftrend};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
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
fn halftrend_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[halftrend_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut close = vec![f64::NAN; len];
    for i in 4..len {
        let x = i as f64;
        close[i] = (x * 0.0023).sin() + 0.0004 * x;
    }
    let mut high = close.clone();
    let mut low = close.clone();
    for i in 0..len {
        if !close[i].is_nan() {
            let x = i as f64 * 0.0025;
            let off = (0.002 * x.sin()).abs() + 0.15;
            high[i] = close[i] + off;
            low[i] = close[i] - off;
        }
    }

    let sweep = HalfTrendBatchRange {
        amplitude: (2, 16, 2),
        channel_deviation: (2.0, 2.0, 0.0),
        atr_period: (14, 14, 0),
    };

    // Quantize to f32 to match CUDA inputs
    let hq: Vec<f64> = high.iter().map(|&v| (v as f32) as f64).collect();
    let lq: Vec<f64> = low.iter().map(|&v| (v as f32) as f64).collect();
    let cq: Vec<f64> = close.iter().map(|&v| (v as f32) as f64).collect();
    let cpu =
        halftrend_batch_with_kernel_slices_internal(&hq, &lq, &cq, &sweep, Kernel::ScalarBatch)?;

    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let close_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();

    let cuda = CudaHalftrend::new(0).expect("CudaHalftrend::new");
    let dev = cuda
        .halftrend_batch_dev(&high_f32, &low_f32, &close_f32, &sweep)
        .expect("halftrend batch");

    assert_eq!(cpu.rows, dev.halftrend.rows);
    assert_eq!(cpu.cols, dev.halftrend.cols);

    let need = cpu.rows * cpu.cols;
    let mut g_ht = vec![0f32; need];
    let mut g_tr = vec![0f32; need];
    let mut g_ah = vec![0f32; need];
    let mut g_al = vec![0f32; need];
    let mut g_bs = vec![0f32; need];
    let mut g_ss = vec![0f32; need];
    dev.halftrend.buf.copy_to(&mut g_ht)?;
    dev.trend.buf.copy_to(&mut g_tr)?;
    dev.atr_high.buf.copy_to(&mut g_ah)?;
    dev.atr_low.buf.copy_to(&mut g_al)?;
    dev.buy.buf.copy_to(&mut g_bs)?;
    dev.sell.buf.copy_to(&mut g_ss)?;

    let tol = 1e-3; // FP32 tolerance
    for idx in 0..need {
        assert!(
            approx_eq(cpu.halftrend[idx], g_ht[idx] as f64, tol),
            "halftrend mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu.trend[idx], g_tr[idx] as f64, 1e-3),
            "trend mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu.atr_high[idx], g_ah[idx] as f64, tol),
            "atr_high mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu.atr_low[idx], g_al[idx] as f64, tol),
            "atr_low mismatch at {}",
            idx
        );
        // buy/sell are sparse; compare only where one is finite
        let cb = cpu.buy_signal[idx];
        let gb = g_bs[idx] as f64;
        if !(cb.is_nan() && gb.is_nan()) {
            assert!(approx_eq(cb, gb, 5e-2), "buy mismatch at {}", idx);
        }
        let cs = cpu.sell_signal[idx];
        let gs = g_ss[idx] as f64;
        if !(cs.is_nan() && gs.is_nan()) {
            assert!(approx_eq(cs, gs, 5e-2), "sell mismatch at {}", idx);
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn halftrend_cuda_batch_time_major_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[halftrend_cuda_batch_time_major_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    // Force the wrapper's time-major batch path: rows>=16 && len>=8192.
    let len = 8192usize;
    let mut close = vec![f64::NAN; len];
    for i in 4..len {
        let x = i as f64;
        close[i] = (x * 0.0023).sin() + 0.0004 * x;
    }
    let mut high = close.clone();
    let mut low = close.clone();
    for i in 0..len {
        if !close[i].is_nan() {
            let x = i as f64 * 0.0025;
            let off = (0.002 * x.sin()).abs() + 0.15;
            high[i] = close[i] + off;
            low[i] = close[i] - off;
        }
    }

    let sweep = HalfTrendBatchRange {
        amplitude: (2, 32, 2),
        channel_deviation: (2.0, 2.0, 0.0),
        atr_period: (14, 14, 0),
    };

    // Quantize to f32 to match CUDA inputs
    let hq: Vec<f64> = high.iter().map(|&v| (v as f32) as f64).collect();
    let lq: Vec<f64> = low.iter().map(|&v| (v as f32) as f64).collect();
    let cq: Vec<f64> = close.iter().map(|&v| (v as f32) as f64).collect();
    let cpu =
        halftrend_batch_with_kernel_slices_internal(&hq, &lq, &cq, &sweep, Kernel::ScalarBatch)?;

    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let close_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();

    let cuda = CudaHalftrend::new(0).expect("CudaHalftrend::new");

    let need = cpu.rows * cpu.cols;
    let mut g_ht = vec![0f32; need];
    let mut g_tr = vec![0f32; need];
    let mut g_ah = vec![0f32; need];
    let mut g_al = vec![0f32; need];
    let mut g_bs = vec![0f32; need];
    let mut g_ss = vec![0f32; need];
    let (rows, cols, _) = cuda.halftrend_batch_into_host_f32(
        &high_f32,
        &low_f32,
        &close_f32,
        &sweep,
        &mut g_ht,
        &mut g_tr,
        &mut g_ah,
        &mut g_al,
        &mut g_bs,
        &mut g_ss,
    )?;

    assert_eq!(rows, cpu.rows);
    assert_eq!(cols, cpu.cols);

    let tol = 1e-3; // FP32 tolerance
    for idx in 0..need {
        assert!(
            approx_eq(cpu.halftrend[idx], g_ht[idx] as f64, tol),
            "halftrend mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu.trend[idx], g_tr[idx] as f64, 1e-3),
            "trend mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu.atr_high[idx], g_ah[idx] as f64, tol),
            "atr_high mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu.atr_low[idx], g_al[idx] as f64, tol),
            "atr_low mismatch at {}",
            idx
        );
        // buy/sell are sparse; compare only where one is finite
        let cb = cpu.buy_signal[idx];
        let gb = g_bs[idx] as f64;
        if !(cb.is_nan() && gb.is_nan()) {
            assert!(approx_eq(cb, gb, 5e-2), "buy mismatch at {}", idx);
        }
        let cs = cpu.sell_signal[idx];
        let gs = g_ss[idx] as f64;
        if !(cs.is_nan() && gs.is_nan()) {
            assert!(approx_eq(cs, gs, 5e-2), "sell mismatch at {}", idx);
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn halftrend_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[halftrend_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let cols = 16usize;
    let rows = 2048usize;
    let mut close_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.2;
            close_tm[t * cols + s] = (x * 0.002).sin() + 0.0003 * x;
        }
    }
    let mut high_tm = close_tm.clone();
    let mut low_tm = close_tm.clone();
    for s in 0..cols {
        for t in 0..rows {
            let idx = t * cols + s;
            if !close_tm[idx].is_nan() {
                let x = (t as f64) * 0.0025;
                let off = (0.002 * x.sin()).abs() + 0.15;
                high_tm[idx] = close_tm[idx] + off;
                low_tm[idx] = close_tm[idx] - off;
            }
        }
    }

    let amplitude = 2usize;
    let atr_period = 14usize;
    let ch = 2.0f64;

    // CPU per series using single-row batch helper
    let mut cpu_ht_tm = vec![f64::NAN; cols * rows];
    let mut cpu_tr_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut h = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        let mut c = vec![f64::NAN; rows];
        for t in 0..rows {
            let idx = t * cols + s;
            h[t] = high_tm[idx];
            l[t] = low_tm[idx];
            c[t] = close_tm[idx];
        }
        let range = HalfTrendBatchRange {
            amplitude: (amplitude, amplitude, 0),
            channel_deviation: (ch, ch, 0.0),
            atr_period: (atr_period, atr_period, 0),
        };
        // Quantize baseline to f32 domain
        let hq: Vec<f64> = h.iter().map(|&v| (v as f32) as f64).collect();
        let lq: Vec<f64> = l.iter().map(|&v| (v as f32) as f64).collect();
        let cq: Vec<f64> = c.iter().map(|&v| (v as f32) as f64).collect();
        let out = halftrend_batch_with_kernel_slices_internal(
            &hq,
            &lq,
            &cq,
            &range,
            Kernel::ScalarBatch,
        )?;
        for t in 0..rows {
            let idx = t * cols + s;
            cpu_ht_tm[idx] = out.halftrend[t];
            cpu_tr_tm[idx] = out.trend[t];
        }
    }

    let high_tm_f32: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let low_tm_f32: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let close_tm_f32: Vec<f32> = close_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaHalftrend::new(0).expect("CudaHalftrend::new");
    let dev = cuda
        .halftrend_many_series_one_param_time_major_dev(
            &high_tm_f32,
            &low_tm_f32,
            &close_tm_f32,
            cols,
            rows,
            amplitude,
            ch,
            atr_period,
        )
        .expect("many-series");
    assert_eq!(dev.halftrend.rows, rows);
    assert_eq!(dev.halftrend.cols, cols);
    let mut g_ht = vec![0f32; cols * rows];
    let mut g_tr = vec![0f32; cols * rows];
    dev.halftrend.buf.copy_to(&mut g_ht)?;
    dev.trend.buf.copy_to(&mut g_tr)?;
    let tol = 1e-3;
    for i in 0..g_ht.len() {
        assert!(
            approx_eq(cpu_ht_tm[i], g_ht[i] as f64, tol),
            "ht mismatch at {}",
            i
        );
        assert!(
            approx_eq(cpu_tr_tm[i], g_tr[i] as f64, tol),
            "tr mismatch at {}",
            i
        );
    }
    Ok(())
}

// Helper: call private slices variant via inline (keeps test in sync if API changes)
#[allow(dead_code)]
fn halftrend_batch_with_kernel_slices_internal(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &HalfTrendBatchRange,
    kern: Kernel,
) -> Result<HalfTrendBatchOutputLite, Box<dyn std::error::Error>> {
    // Use the public builder as a simple forwarder
    let out = halftrend_batch_with_kernel_slices_public(high, low, close, sweep, kern)?;
    Ok(HalfTrendBatchOutputLite {
        halftrend: out.halftrend,
        trend: out.trend,
        atr_high: out.atr_high,
        atr_low: out.atr_low,
        buy_signal: out.buy_signal,
        sell_signal: out.sell_signal,
        rows: out.rows,
        cols: out.cols,
    })
}

#[derive(Clone)]
struct HalfTrendBatchOutputLite {
    halftrend: Vec<f64>,
    trend: Vec<f64>,
    atr_high: Vec<f64>,
    atr_low: Vec<f64>,
    buy_signal: Vec<f64>,
    sell_signal: Vec<f64>,
    rows: usize,
    cols: usize,
}

fn halftrend_batch_with_kernel_slices_public(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &HalfTrendBatchRange,
    kern: Kernel,
) -> Result<my_project::indicators::halftrend::HalfTrendBatchOutput, Box<dyn std::error::Error>> {
    // Expose the private function via builder API
    let combos_out = my_project::indicators::halftrend::HalfTrendBatchBuilder::new()
        .amplitude_range(sweep.amplitude.0, sweep.amplitude.1, sweep.amplitude.2)
        .channel_deviation_range(
            sweep.channel_deviation.0,
            sweep.channel_deviation.1,
            sweep.channel_deviation.2,
        )
        .atr_period_range(sweep.atr_period.0, sweep.atr_period.1, sweep.atr_period.2)
        .kernel(kern)
        .apply_slices(high, low, close)?;
    Ok(combos_out)
}
