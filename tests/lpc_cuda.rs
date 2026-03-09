use vector_ta::indicators::lpc::{
    lpc_batch_with_kernel, lpc_with_kernel, LpcBatchRange, LpcInput, LpcParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::{CopyDestination, DeviceBuffer};
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::lpc_wrapper::CudaLpc;

fn approx(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        true
    } else {
        (a - b).abs() <= tol
    }
}

#[cfg(feature = "cuda")]
fn copy_triplet(
    triplet: &vector_ta::cuda::wto_wrapper::DeviceArrayF32Triplet,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
    let mut filter = vec![0f32; triplet.wt1.len()];
    triplet.wt1.buf.copy_to(&mut filter)?;
    let mut high = vec![0f32; triplet.wt2.len()];
    triplet.wt2.buf.copy_to(&mut high)?;
    let mut low = vec![0f32; triplet.hist.len()];
    triplet.hist.buf.copy_to(&mut low)?;
    Ok((filter, high, low))
}

#[test]
fn feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    {
        assert!(true);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn lpc_cuda_batch_matches_cpu_fixed() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[lpc_cuda_batch_matches_cpu_fixed] skipped - no CUDA device");
        return Ok(());
    }

    let n = 8192usize;
    let mut h = vec![f64::NAN; n];
    let mut l = vec![f64::NAN; n];
    let mut c = vec![f64::NAN; n];
    let mut s = vec![f64::NAN; n];
    for i in 4..n {
        let x = i as f64;
        let base = (x * 0.00123).sin() + 0.00017 * x;
        s[i] = base;
        c[i] = base;
        h[i] = base + 0.5;
        l[i] = base - 0.5;
    }
    let sweep = LpcBatchRange {
        fixed_period: (10, 40, 5),
        cycle_mult: (1.0, 1.0, 0.0),
        tr_mult: (0.5, 1.5, 0.5),
        cutoff_type: "fixed".to_string(),
        max_cycle_limit: 60,
    };
    let cpu = lpc_batch_with_kernel(&h, &l, &c, &s, &sweep, Kernel::ScalarBatch)?;
    let rows = cpu.rows;
    let cols = cpu.cols;

    let hf32: Vec<f32> = h.iter().map(|&v| v as f32).collect();
    let lf32: Vec<f32> = l.iter().map(|&v| v as f32).collect();
    let cf32: Vec<f32> = c.iter().map(|&v| v as f32).collect();
    let sf32: Vec<f32> = s.iter().map(|&v| v as f32).collect();
    let cuda = CudaLpc::new(0).expect("CudaLpc::new");
    let (triplet, combos) = cuda
        .lpc_batch_dev(&hf32, &lf32, &cf32, &sf32, &sweep)
        .expect("lpc_batch_dev");
    assert_eq!(combos.len() * 3, rows);
    assert_eq!(triplet.rows(), combos.len());
    assert_eq!(triplet.cols(), cols);

    let mut gf = vec![0f32; triplet.wt1.len()];
    triplet.wt1.buf.copy_to(&mut gf)?;
    let mut ghi = vec![0f32; triplet.wt2.len()];
    triplet.wt2.buf.copy_to(&mut ghi)?;
    let mut glo = vec![0f32; triplet.hist.len()];
    triplet.hist.buf.copy_to(&mut glo)?;

    let tol = 5e-4;
    for combo in 0..combos.len() {
        let cpu_f_row = (combo * 3 + 0) * cols;
        let cpu_hi_row = (combo * 3 + 1) * cols;
        let cpu_lo_row = (combo * 3 + 2) * cols;
        let gpu_f_row = combo * cols;
        for j in 0..cols {
            let cf = cpu.values[cpu_f_row + j];
            let ch = cpu.values[cpu_hi_row + j];
            let clv = cpu.values[cpu_lo_row + j];
            assert!(
                approx(cf, gf[gpu_f_row + j] as f64, tol),
                "filter mismatch at row {}, col {}",
                combo,
                j
            );
            assert!(
                approx(ch, ghi[gpu_f_row + j] as f64, tol),
                "high mismatch at row {}, col {}",
                combo,
                j
            );
            assert!(
                approx(clv, glo[gpu_f_row + j] as f64, tol),
                "low mismatch at row {}, col {}",
                combo,
                j
            );
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn lpc_cuda_many_series_one_param_matches_cpu_fixed() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[lpc_cuda_many_series_one_param_matches_cpu_fixed] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 16usize;
    let rows = 2048usize;
    let mut h_tm = vec![f64::NAN; cols * rows];
    let mut l_tm = vec![f64::NAN; cols * rows];
    let mut c_tm = vec![f64::NAN; cols * rows];
    let mut s_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in 4..rows {
            let x = (t as f64) + (s as f64) * 0.2;
            let base = (x * 0.002).sin() + 0.0003 * x;
            s_tm[t * cols + s] = base;
            c_tm[t * cols + s] = base;
            h_tm[t * cols + s] = base + 0.4;
            l_tm[t * cols + s] = base - 0.4;
        }
    }

    let period = 21usize;
    let tr_mult = 1.1;
    let mut cpu_f_tm = vec![f64::NAN; cols * rows];
    let mut cpu_hi_tm = vec![f64::NAN; cols * rows];
    let mut cpu_lo_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut hs = vec![f64::NAN; rows];
        let mut ls = vec![f64::NAN; rows];
        let mut cs = vec![f64::NAN; rows];
        let mut ss = vec![f64::NAN; rows];
        for t in 0..rows {
            let i = t * cols + s;
            hs[t] = h_tm[i];
            ls[t] = l_tm[i];
            cs[t] = c_tm[i];
            ss[t] = s_tm[i];
        }
        let params = LpcParams {
            cutoff_type: Some("fixed".to_string()),
            fixed_period: Some(period),
            max_cycle_limit: Some(60),
            cycle_mult: Some(1.0),
            tr_mult: Some(tr_mult),
        };
        let input = LpcInput::from_slices(&hs, &ls, &cs, &ss, params);
        let out = lpc_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            let i = t * cols + s;
            cpu_f_tm[i] = out.filter[t];
            cpu_hi_tm[i] = out.high_band[t];
            cpu_lo_tm[i] = out.low_band[t];
        }
    }

    let hf32: Vec<f32> = h_tm.iter().map(|&v| v as f32).collect();
    let lf32: Vec<f32> = l_tm.iter().map(|&v| v as f32).collect();
    let cf32: Vec<f32> = c_tm.iter().map(|&v| v as f32).collect();
    let sf32: Vec<f32> = s_tm.iter().map(|&v| v as f32).collect();
    let params = LpcParams {
        cutoff_type: Some("fixed".to_string()),
        fixed_period: Some(period),
        max_cycle_limit: Some(60),
        cycle_mult: Some(1.0),
        tr_mult: Some(tr_mult),
    };
    let cuda = CudaLpc::new(0).expect("CudaLpc::new");
    let triplet = cuda
        .lpc_many_series_one_param_time_major_dev(&hf32, &lf32, &cf32, &sf32, cols, rows, &params)
        .expect("lpc_many_series_one_param_time_major_dev");
    assert_eq!(triplet.rows(), rows);
    assert_eq!(triplet.cols(), cols);
    let mut gf = vec![0f32; triplet.wt1.len()];
    triplet.wt1.buf.copy_to(&mut gf)?;
    let mut ghi = vec![0f32; triplet.wt2.len()];
    triplet.wt2.buf.copy_to(&mut ghi)?;
    let mut glo = vec![0f32; triplet.hist.len()];
    triplet.hist.buf.copy_to(&mut glo)?;
    let tol = 1e-4;
    for i in 0..(cols * rows) {
        assert!(
            approx(cpu_f_tm[i], gf[i] as f64, tol),
            "filter mismatch at {}",
            i
        );
        assert!(
            approx(cpu_hi_tm[i], ghi[i] as f64, tol),
            "high mismatch at {}",
            i
        );
        assert!(
            approx(cpu_lo_tm[i], glo[i] as f64, tol),
            "low mismatch at {}",
            i
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn lpc_cuda_borrowed_device_path_matches_legacy_wrapper() -> Result<(), Box<dyn std::error::Error>>
{
    if !cuda_available() {
        eprintln!(
            "[lpc_cuda_borrowed_device_path_matches_legacy_wrapper] skipped - no CUDA device"
        );
        return Ok(());
    }

    let n = 4096usize;
    let first_valid = 6usize;
    let mut h = vec![f32::NAN; n];
    let mut l = vec![f32::NAN; n];
    let mut c = vec![f32::NAN; n];
    let mut s = vec![f32::NAN; n];
    for i in first_valid..n {
        let x = i as f32;
        let base = (x * 0.017).sin() + 0.0009 * x + (x * 0.0031).cos() * 0.2;
        s[i] = base;
        c[i] = base + (x * 0.011).sin() * 0.03;
        h[i] = c[i] + 0.35 + (x * 0.005).cos().abs() * 0.04;
        l[i] = c[i] - 0.35 - (x * 0.007).sin().abs() * 0.04;
    }

    let sweep = LpcBatchRange {
        fixed_period: (10, 22, 6),
        cycle_mult: (0.8, 1.2, 0.2),
        tr_mult: (0.75, 1.25, 0.5),
        cutoff_type: "adaptive".to_string(),
        max_cycle_limit: 48,
    };

    let cuda = CudaLpc::new(0).expect("CudaLpc::new");
    let (legacy_triplet, legacy_combos) = cuda
        .lpc_batch_dev(&h, &l, &c, &s, &sweep)
        .expect("legacy lpc_batch_dev");

    let d_h = DeviceBuffer::from_slice(&h)?;
    let d_l = DeviceBuffer::from_slice(&l)?;
    let d_c = DeviceBuffer::from_slice(&c)?;
    let d_s = DeviceBuffer::from_slice(&s)?;
    let (borrowed_triplet, borrowed_combos) = cuda
        .lpc_batch_dev_from_device_inputs(&d_h, &d_l, &d_c, &d_s, n, first_valid, &sweep)
        .expect("borrowed-device lpc_batch_dev_from_device_inputs");
    cuda.synchronize()?;

    assert_eq!(legacy_combos.len(), borrowed_combos.len());
    for (legacy, borrowed) in legacy_combos.iter().zip(borrowed_combos.iter()) {
        assert_eq!(
            legacy.cutoff_type.as_deref(),
            borrowed.cutoff_type.as_deref()
        );
        assert_eq!(legacy.fixed_period, borrowed.fixed_period);
        assert_eq!(legacy.max_cycle_limit, borrowed.max_cycle_limit);
        assert_eq!(legacy.cycle_mult, borrowed.cycle_mult);
        assert_eq!(legacy.tr_mult, borrowed.tr_mult);
    }

    assert_eq!(legacy_triplet.rows(), borrowed_triplet.rows());
    assert_eq!(legacy_triplet.cols(), borrowed_triplet.cols());

    let (legacy_f, legacy_hi, legacy_lo) = copy_triplet(&legacy_triplet)?;
    let (borrowed_f, borrowed_hi, borrowed_lo) = copy_triplet(&borrowed_triplet)?;
    let tol = 1e-4;
    for i in 0..legacy_f.len() {
        assert!(
            approx(legacy_f[i] as f64, borrowed_f[i] as f64, tol),
            "filter mismatch at {}",
            i
        );
        assert!(
            approx(legacy_hi[i] as f64, borrowed_hi[i] as f64, tol),
            "high mismatch at {}",
            i
        );
        assert!(
            approx(legacy_lo[i] as f64, borrowed_lo[i] as f64, tol),
            "low mismatch at {}",
            i
        );
    }

    Ok(())
}
