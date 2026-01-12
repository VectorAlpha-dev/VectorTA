use vector_ta::indicators::mod_god_mode::{
    mod_god_mode, mod_god_mode_batch_with_kernel, ModGodModeBatchRange, ModGodModeData,
    ModGodModeInput, ModGodModeMode, ModGodModeParams,
};
use vector_ta::utilities::data_loader::Candles;
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::mod_god_mode_wrapper::CudaModGodMode;

fn gen_candles(len: usize) -> Candles {
    let mut h = vec![f64::NAN; len];
    let mut l = vec![f64::NAN; len];
    let mut c = vec![f64::NAN; len];
    let mut v = vec![f64::NAN; len];
    for i in 4..len {
        let x = i as f64;
        c[i] = (x * 0.0023).sin() + 0.0007 * x;
        h[i] = c[i] + 0.5;
        l[i] = c[i] - 0.5;
        v[i] = (x * 0.0017).cos().abs() + 0.4;
    }
    let ts: Vec<i64> = (0..len as i64).collect();
    Candles::new(ts, vec![0.0; len], h, l, c, v)
}

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
fn mod_god_mode_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[mod_god_mode_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let candles = gen_candles(4096);
    let h: Vec<f32> = candles.high.iter().map(|&x| x as f32).collect();
    let l: Vec<f32> = candles.low.iter().map(|&x| x as f32).collect();
    let c: Vec<f32> = candles.close.iter().map(|&x| x as f32).collect();

    let sweep = ModGodModeBatchRange {
        n1: (10, 14, 2),
        n2: (6, 6, 0),
        n3: (4, 4, 0),
        mode: ModGodModeMode::TraditionMg,
    };

    let cpu = mod_god_mode_batch_with_kernel(
        &candles.high,
        &candles.low,
        &candles.close,
        None,
        &sweep,
        Kernel::ScalarBatch,
    )?;

    let cuda = CudaModGodMode::new(0).expect("CudaModGodMode");
    let res = cuda
        .mod_god_mode_batch_dev(&h, &l, &c, None, &sweep)
        .expect("cuda batch");
    assert_eq!(cpu.rows, res.outputs.rows());
    assert_eq!(cpu.cols, res.outputs.cols());

    let mut wt_host = vec![0f32; res.outputs.wt1.len()];
    let mut sig_host = vec![0f32; res.outputs.wt2.len()];
    let mut hist_host = vec![0f32; res.outputs.hist.len()];
    res.outputs.wt1.buf.copy_to(&mut wt_host)?;
    res.outputs.wt2.buf.copy_to(&mut sig_host)?;
    res.outputs.hist.buf.copy_to(&mut hist_host)?;

    let tol = 1.5e-1;
    for i in 0..(cpu.rows * cpu.cols) {
        let cpu_wt = cpu.wavetrend[i];
        let gpu_wt = wt_host[i] as f64;
        assert!(
            approx_eq(cpu_wt, gpu_wt, tol),
            "wavetrend mismatch @{} cpu={} gpu={} diff={}",
            i,
            cpu_wt,
            gpu_wt,
            (cpu_wt - gpu_wt).abs()
        );
        let cpu_sig = cpu.signal[i];
        let gpu_sig = sig_host[i] as f64;
        assert!(
            approx_eq(cpu_sig, gpu_sig, tol),
            "signal mismatch @{} cpu={} gpu={} diff={}",
            i,
            cpu_sig,
            gpu_sig,
            (cpu_sig - gpu_sig).abs()
        );
        let cpu_hist = cpu.histogram[i];
        let gpu_hist = hist_host[i] as f64;
        assert!(
            approx_eq(cpu_hist, gpu_hist, tol),
            "hist mismatch @{} cpu={} gpu={} diff={}",
            i,
            cpu_hist,
            gpu_hist,
            (cpu_hist - gpu_hist).abs()
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn mod_god_mode_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[mod_god_mode_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let cols = 4usize;
    let rows = 1024usize;
    let mut h_tm = vec![f64::NAN; cols * rows];
    let mut l_tm = vec![f64::NAN; cols * rows];
    let mut c_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in 4..rows {
            let x = (t as f64) + (s as f64) * 0.11;
            let c = (x * 0.0021).sin() + 0.0005 * x;
            h_tm[t * cols + s] = c + 0.5;
            l_tm[t * cols + s] = c - 0.5;
            c_tm[t * cols + s] = c;
        }
    }

    let mut wt_cpu = vec![f64::NAN; cols * rows];
    let mut sig_cpu = vec![f64::NAN; cols * rows];
    let mut hist_cpu = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut h = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        let mut c = vec![f64::NAN; rows];
        for t in 0..rows {
            h[t] = h_tm[t * cols + s];
            l[t] = l_tm[t * cols + s];
            c[t] = c_tm[t * cols + s];
        }
        let input = ModGodModeInput {
            data: ModGodModeData::Slices {
                high: &h,
                low: &l,
                close: &c,
                volume: None,
            },
            params: ModGodModeParams {
                n1: Some(17),
                n2: Some(6),
                n3: Some(4),
                mode: Some(ModGodModeMode::TraditionMg),
                use_volume: Some(false),
            },
        };
        let out = mod_god_mode(&input)?;
        for t in 0..rows {
            wt_cpu[t * cols + s] = out.wavetrend[t];
            sig_cpu[t * cols + s] = out.signal[t];
            hist_cpu[t * cols + s] = out.histogram[t];
        }
    }

    let h_tm_f32: Vec<f32> = h_tm.iter().map(|&v| v as f32).collect();
    let l_tm_f32: Vec<f32> = l_tm.iter().map(|&v| v as f32).collect();
    let c_tm_f32: Vec<f32> = c_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaModGodMode::new(0).expect("CudaModGodMode");
    let out = cuda
        .mod_god_mode_many_series_one_param_time_major_dev(
            &h_tm_f32,
            &l_tm_f32,
            &c_tm_f32,
            None,
            cols,
            rows,
            &ModGodModeParams::default(),
        )
        .expect("cuda many-series");
    assert_eq!(out.rows(), rows);
    assert_eq!(out.cols(), cols);
    let mut wt_g = vec![0f32; out.wt1.len()];
    let mut sig_g = vec![0f32; out.wt2.len()];
    let mut hist_g = vec![0f32; out.hist.len()];
    out.wt1.buf.copy_to(&mut wt_g)?;
    out.wt2.buf.copy_to(&mut sig_g)?;
    out.hist.buf.copy_to(&mut hist_g)?;

    let tol = 1.5e-1;
    for i in 0..wt_g.len() {
        let cpu_wt = wt_cpu[i];
        let gpu_wt = wt_g[i] as f64;
        assert!(
            approx_eq(cpu_wt, gpu_wt, tol),
            "wt mismatch @{} cpu={} gpu={} diff={}",
            i,
            cpu_wt,
            gpu_wt,
            (cpu_wt - gpu_wt).abs()
        );
        let cpu_sig = sig_cpu[i];
        let gpu_sig = sig_g[i] as f64;
        assert!(
            approx_eq(cpu_sig, gpu_sig, tol),
            "sig mismatch @{} cpu={} gpu={} diff={}",
            i,
            cpu_sig,
            gpu_sig,
            (cpu_sig - gpu_sig).abs()
        );
        let cpu_hist = hist_cpu[i];
        let gpu_hist = hist_g[i] as f64;
        assert!(
            approx_eq(cpu_hist, gpu_hist, tol),
            "hist mismatch @{} cpu={} gpu={} diff={}",
            i,
            cpu_hist,
            gpu_hist,
            (cpu_hist - gpu_hist).abs()
        );
    }
    Ok(())
}
