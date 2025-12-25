// Integration tests for CUDA Correlation Cycle kernels

use my_project::indicators::correlation_cycle::{
    correlation_cycle_batch_with_kernel, correlation_cycle_with_kernel, CorrelationCycleBatchRange,
    CorrelationCycleInput, CorrelationCycleParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::moving_averages::CudaCorrelationCycle;

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
fn correlation_cycle_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[correlation_cycle_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let len = 16_384usize;
    let mut price = vec![f64::NAN; len];
    for i in 8..len {
        let x = i as f64;
        price[i] = (x * 0.00123).sin() + 0.00017 * x;
    }
    let sweep = CorrelationCycleBatchRange {
        period: (16, 64, 8),
        threshold: (9.0, 9.0, 0.0),
    };
    // Quantize CPU baseline inputs to match the CUDA FP32 input domain.
    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let price_q: Vec<f64> = price_f32.iter().map(|&v| v as f64).collect();
    let cpu = correlation_cycle_batch_with_kernel(&price_q, &sweep, Kernel::ScalarBatch)?;
    let mut cuda = CudaCorrelationCycle::new(0).expect("CudaCorrelationCycle::new");
    let quad = cuda
        .correlation_cycle_batch_dev(&price_f32, &sweep)
        .expect("cuda correlation_cycle_batch_dev");

    assert_eq!(cpu.rows, quad.rows());
    assert_eq!(cpu.cols, quad.cols());

    let mut g_real = vec![0f32; quad.real.len()];
    let mut g_imag = vec![0f32; quad.imag.len()];
    let mut g_ang = vec![0f32; quad.angle.len()];
    let mut g_st = vec![0f32; quad.state.len()];
    quad.real.buf.copy_to(&mut g_real)?;
    quad.imag.buf.copy_to(&mut g_imag)?;
    quad.angle.buf.copy_to(&mut g_ang)?;
    quad.state.buf.copy_to(&mut g_st)?;

    // Period sweep includes small periods (e.g., 16) which are more numerically sensitive in FP32.
    let tol = 1.25e-2;
    let tol_ang = 0.45f64;
    let state_eps = 0.1f64;
    for idx in 0..(cpu.rows * cpu.cols) {
        assert!(
            approx_eq(cpu.real[idx], g_real[idx] as f64, tol),
            "real mismatch at {}: cpu={} gpu={} diff={}",
            idx,
            cpu.real[idx],
            g_real[idx] as f64,
            (cpu.real[idx] - (g_real[idx] as f64)).abs()
        );
        assert!(
            approx_eq(cpu.imag[idx], g_imag[idx] as f64, tol),
            "imag mismatch at {}: cpu={} gpu={} diff={}",
            idx,
            cpu.imag[idx],
            g_imag[idx] as f64,
            (cpu.imag[idx] - (g_imag[idx] as f64)).abs()
        );
        if !approx_eq(cpu.angle[idx], g_ang[idx] as f64, tol_ang) {
            let row = idx / cpu.cols;
            let col = idx % cpu.cols;
            let period = sweep.period.0 + row * sweep.period.2;
            panic!(
                "angle mismatch at idx={} row={} col={} period={} cpu={} gpu={} diff={} tol={}",
                idx,
                row,
                col,
                period,
                cpu.angle[idx],
                g_ang[idx] as f64,
                (cpu.angle[idx] - (g_ang[idx] as f64)).abs(),
                tol_ang
            );
        }
        // state is discrete, tolerate exact or NaN warmup
        let cs = cpu.state[idx];
        let gs = g_st[idx] as f64;
        if cs.is_nan() && gs.is_nan() {
            continue;
        }
        if approx_eq(cs, gs, 1e-3) {
            continue;
        }
        let row = idx / cpu.cols;
        let col = idx % cpu.cols;
        if col == 0 {
            panic!("state mismatch at idx={} col=0 cpu_state={} gpu_state={}", idx, cs, gs);
        }
        let period = sweep.period.0 + row * sweep.period.2;
        let thr = sweep.threshold.0 + (row as f64) * sweep.threshold.2;
        let cpu_a = cpu.angle[idx];
        let gpu_a = g_ang[idx] as f64;
        let cpu_ap = cpu.angle[idx - 1];
        let gpu_ap = g_ang[idx - 1] as f64;
        let cpu_diff = (cpu_a - cpu_ap).abs();
        let gpu_diff = (gpu_a - gpu_ap).abs();
        if (cpu_diff - thr).abs() <= state_eps || (gpu_diff - thr).abs() <= state_eps {
            continue;
        }
        panic!(
            "state mismatch at idx={} row={} col={} period={} thr={} cpu_state={} gpu_state={} cpu_angle={} gpu_angle={} cpu_prev_angle={} gpu_prev_angle={} cpu_diff={} gpu_diff={}",
            idx,
            row,
            col,
            period,
            thr,
            cs,
            gs,
            cpu_a,
            gpu_a,
            cpu_ap,
            gpu_ap,
            cpu_diff,
            gpu_diff
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn correlation_cycle_cuda_many_series_one_param_matches_cpu(
) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!(
            "[correlation_cycle_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }
    let cols = 7usize;
    let rows = 2048usize;
    let mut price_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.2;
            price_tm[t * cols + s] = (x * 0.002).sin() + 0.0003 * x;
        }
    }
    let period = 32usize;
    let threshold = 9.0f64;

    // CPU baseline per series
    let mut cpu_real = vec![f64::NAN; cols * rows];
    let mut cpu_imag = vec![f64::NAN; cols * rows];
    let mut cpu_ang = vec![f64::NAN; cols * rows];
    let mut cpu_st = vec![f64::NAN; cols * rows];
    // Quantize CPU inputs to match CUDA FP32 domain.
    let price_tm_f32: Vec<f32> = price_tm.iter().map(|&v| v as f32).collect();
    let price_tm_q: Vec<f64> = price_tm_f32.iter().map(|&v| v as f64).collect();
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for t in 0..rows {
            series[t] = price_tm_q[t * cols + s];
        }
        let params = CorrelationCycleParams {
            period: Some(period),
            threshold: Some(threshold),
        };
        let input = CorrelationCycleInput::from_slice(&series, params);
        let out = correlation_cycle_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            let idx = t * cols + s;
            cpu_real[idx] = out.real[t];
            cpu_imag[idx] = out.imag[t];
            cpu_ang[idx] = out.angle[t];
            cpu_st[idx] = out.state[t];
        }
    }

    let params = CorrelationCycleParams {
        period: Some(period),
        threshold: Some(threshold),
    };
    let mut cuda = CudaCorrelationCycle::new(0).expect("cc cuda");
    let quad = cuda
        .correlation_cycle_many_series_one_param_time_major_dev(&price_tm_f32, cols, rows, &params)
        .expect("cc many series dev");

    assert_eq!(quad.rows(), rows);
    assert_eq!(quad.cols(), cols);
    let mut g_real = vec![0f32; quad.real.len()];
    let mut g_imag = vec![0f32; quad.imag.len()];
    let mut g_ang = vec![0f32; quad.angle.len()];
    let mut g_st = vec![0f32; quad.state.len()];
    quad.real.buf.copy_to(&mut g_real)?;
    quad.imag.buf.copy_to(&mut g_imag)?;
    quad.angle.buf.copy_to(&mut g_ang)?;
    quad.state.buf.copy_to(&mut g_st)?;

    let tol = 6e-3;
    let tol_ang = 0.15f64;
    let state_eps = 0.1f64;
    for idx in 0..(cols * rows) {
        assert!(
            approx_eq(cpu_real[idx], g_real[idx] as f64, tol),
            "real mismatch at {}: cpu={} gpu={} diff={}",
            idx,
            cpu_real[idx],
            g_real[idx] as f64,
            (cpu_real[idx] - (g_real[idx] as f64)).abs()
        );
        assert!(
            approx_eq(cpu_imag[idx], g_imag[idx] as f64, tol),
            "imag mismatch at {}: cpu={} gpu={} diff={}",
            idx,
            cpu_imag[idx],
            g_imag[idx] as f64,
            (cpu_imag[idx] - (g_imag[idx] as f64)).abs()
        );
        if !(cpu_ang[idx].is_nan() && (g_ang[idx] as f64).is_nan()) {
            if !approx_eq(cpu_ang[idx], g_ang[idx] as f64, tol_ang) {
                let row = idx / cols;
                let col = idx % cols;
                panic!(
                    "angle mismatch at idx={} row={} col={} period={} cpu={} gpu={} diff={} tol={}",
                    idx,
                    row,
                    col,
                    period,
                    cpu_ang[idx],
                    g_ang[idx] as f64,
                    (cpu_ang[idx] - (g_ang[idx] as f64)).abs(),
                    tol_ang
                );
            }
        }
        let cs = cpu_st[idx];
        let gs = g_st[idx] as f64;
        if cs.is_nan() && gs.is_nan() {
            continue;
        }
        if approx_eq(cs, gs, 1e-3) {
            continue;
        }
        let row = idx / cols;
        if row == 0 {
            panic!("state mismatch at idx={} row=0 cpu_state={} gpu_state={}", idx, cs, gs);
        }
        let col = idx % cols;
        let cpu_a = cpu_ang[idx];
        let gpu_a = g_ang[idx] as f64;
        let cpu_ap = cpu_ang[idx - cols];
        let gpu_ap = g_ang[idx - cols] as f64;
        let cpu_diff = (cpu_a - cpu_ap).abs();
        let gpu_diff = (gpu_a - gpu_ap).abs();
        if (cpu_diff - threshold).abs() <= state_eps || (gpu_diff - threshold).abs() <= state_eps {
            continue;
        }
        panic!(
            "state mismatch at idx={} row={} col={} period={} thr={} cpu_state={} gpu_state={} cpu_angle={} gpu_angle={} cpu_prev_angle={} gpu_prev_angle={} cpu_diff={} gpu_diff={}",
            idx,
            row,
            col,
            period,
            threshold,
            cs,
            gs,
            cpu_a,
            gpu_a,
            cpu_ap,
            gpu_ap,
            cpu_diff,
            gpu_diff
        );
    }
    Ok(())
}
