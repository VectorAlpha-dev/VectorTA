use vector_ta::indicators::keltner::{
    keltner_with_kernel, KeltnerBatchBuilder, KeltnerInput, KeltnerParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaKeltner};

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
fn keltner_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[keltner_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut close = vec![f64::NAN; len];
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    for i in 4..len {
        let x = i as f64;
        close[i] = (x * 0.00123).sin() + 0.00019 * x;
        let off = (0.004 * (x * 0.002).sin()).abs() + 0.12;
        high[i] = close[i] + off;
        low[i] = close[i] - off;
    }
    let src = close.clone();

    let sweep = vector_ta::indicators::keltner::KeltnerBatchRange {
        period: (10, 32, 3),
        multiplier: (1.0, 2.0, 0.5),
    };

    let cpu = KeltnerBatchBuilder::new()
        .period_range(10, 32, 3)
        .multiplier_range(1.0, 2.0, 0.5)
        .apply_slice(&high, &low, &close, &src)?;

    let (h_f32, l_f32, c_f32, s_f32): (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) = (
        high.iter().map(|&v| v as f32).collect(),
        low.iter().map(|&v| v as f32).collect(),
        close.iter().map(|&v| v as f32).collect(),
        src.iter().map(|&v| v as f32).collect(),
    );
    let cuda = CudaKeltner::new(0).expect("CudaKeltner::new");
    let gpu = cuda
        .keltner_batch_dev(&h_f32, &l_f32, &c_f32, &s_f32, &sweep, "ema")
        .expect("keltner_batch_dev");

    assert_eq!(cpu.rows, gpu.outputs.middle.rows);
    assert_eq!(cpu.cols, gpu.outputs.middle.cols);

    let mut up_g = vec![0f32; gpu.outputs.upper.len()];
    let mut mid_g = vec![0f32; gpu.outputs.middle.len()];
    let mut low_g = vec![0f32; gpu.outputs.lower.len()];
    gpu.outputs.upper.buf.copy_to(&mut up_g)?;
    gpu.outputs.middle.buf.copy_to(&mut mid_g)?;
    gpu.outputs.lower.buf.copy_to(&mut low_g)?;

    let tol = 2e-3;
    for idx in 0..(cpu.rows * cpu.cols) {
        if idx == 13 {
            eprintln!(
                "cpu up/mid/low @13 = {:.6} {:.6} {:.6}; gpu {:.6} {:.6} {:.6}",
                cpu.upper_band[idx],
                cpu.middle_band[idx],
                cpu.lower_band[idx],
                up_g[idx],
                mid_g[idx],
                low_g[idx]
            );
        }
        assert!(
            approx_eq(cpu.upper_band[idx], up_g[idx] as f64, tol),
            "upper mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu.middle_band[idx], mid_g[idx] as f64, tol),
            "middle mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu.lower_band[idx], low_g[idx] as f64, tol),
            "lower mismatch at {}",
            idx
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn keltner_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[keltner_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let (cols, rows) = (8usize, 2048usize);
    let mut close_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.3;
            close_tm[t * cols + s] = (x * 0.002).sin() + 0.00025 * x;
        }
    }
    let mut high_tm = close_tm.clone();
    let mut low_tm = close_tm.clone();
    for s in 0..cols {
        for t in 0..rows {
            let v = close_tm[t * cols + s];
            if v.is_nan() {
                continue;
            }
            let x = (t as f64) * 0.002;
            let off = (0.004 * x.cos()).abs() + 0.11;
            high_tm[t * cols + s] = v + off;
            low_tm[t * cols + s] = v - off;
        }
    }

    let period = 20usize;
    let mult = 2.0f64;

    let mut upper_tm = vec![f64::NAN; cols * rows];
    let mut middle_tm = vec![f64::NAN; cols * rows];
    let mut lower_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut h = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        let mut c = vec![f64::NAN; rows];
        let mut src = vec![f64::NAN; rows];
        for t in 0..rows {
            let idx = t * cols + s;
            h[t] = high_tm[idx];
            l[t] = low_tm[idx];
            c[t] = close_tm[idx];
            src[t] = close_tm[idx];
        }
        let params = KeltnerParams {
            period: Some(period),
            multiplier: Some(mult),
            ma_type: Some("ema".into()),
        };
        let out = keltner_with_kernel(
            &KeltnerInput::from_slice(&h, &l, &c, &src, params),
            Kernel::Scalar,
        )?;
        for t in 0..rows {
            let idx = t * cols + s;
            upper_tm[idx] = out.upper_band[t];
            middle_tm[idx] = out.middle_band[t];
            lower_tm[idx] = out.lower_band[t];
        }
    }

    let to_f32 = |v: &Vec<f64>| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
    let (hf, lf, cf, sf) = (
        to_f32(&high_tm),
        to_f32(&low_tm),
        to_f32(&close_tm),
        to_f32(&close_tm),
    );
    let cuda = CudaKeltner::new(0).expect("CudaKeltner::new");
    let trip = cuda
        .keltner_many_series_one_param_time_major_dev(
            &hf,
            &lf,
            &cf,
            &sf,
            cols,
            rows,
            period,
            mult as f32,
            "ema",
        )
        .expect("keltner_many_series_one_param_time_major_dev");

    let mut up_g = vec![0f32; trip.upper.len()];
    let mut mid_g = vec![0f32; trip.middle.len()];
    let mut low_g = vec![0f32; trip.lower.len()];
    trip.upper.buf.copy_to(&mut up_g)?;
    trip.middle.buf.copy_to(&mut mid_g)?;
    trip.lower.buf.copy_to(&mut low_g)?;

    let tol = 2e-3;
    for idx in 0..up_g.len() {
        assert!(
            approx_eq(upper_tm[idx], up_g[idx] as f64, tol),
            "upper mismatch at {}",
            idx
        );
        assert!(
            approx_eq(middle_tm[idx], mid_g[idx] as f64, tol),
            "middle mismatch at {}",
            idx
        );
        assert!(
            approx_eq(lower_tm[idx], low_g[idx] as f64, tol),
            "lower mismatch at {}",
            idx
        );
    }
    Ok(())
}
