use vector_ta::indicators::acosc::{acosc_with_kernel, AcoscData, AcoscInput, AcoscParams};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::oscillators::CudaAcosc;

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
fn acosc_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[acosc_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 8192usize;
    let mut high = vec![f64::NAN; series_len];
    let mut low = vec![f64::NAN; series_len];

    for i in 10..series_len {
        let x = i as f64;
        let base = (x * 0.002).sin() + 0.001 * x;
        high[i] = base + 0.8;
        low[i] = base - 0.6;
    }

    let input = AcoscInput {
        data: AcoscData::Slices {
            high: &high,
            low: &low,
        },
        params: AcoscParams::default(),
    };
    let cpu = acosc_with_kernel(&input, Kernel::Scalar)?;

    let cuda = CudaAcosc::new(0).expect("CudaAcosc::new");
    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let pair = cuda
        .acosc_batch_dev(&high_f32, &low_f32)
        .expect("acosc batch");

    assert_eq!(pair.rows(), 1);
    assert_eq!(pair.cols(), series_len);

    let mut osc_host = vec![0f32; pair.osc.len()];
    let mut chg_host = vec![0f32; pair.change.len()];
    pair.osc.buf.copy_to(&mut osc_host).unwrap();
    pair.change.buf.copy_to(&mut chg_host).unwrap();

    let tol = 5e-4;
    for i in 0..series_len {
        let cpu_o = cpu.osc[i];
        let cpu_c = cpu.change[i];
        let gpu_o = osc_host[i] as f64;
        let gpu_c = chg_host[i] as f64;
        assert!(
            approx_eq(cpu_o, gpu_o, tol),
            "osc mismatch at {}: {} vs {}",
            i,
            cpu_o,
            gpu_o
        );
        assert!(
            approx_eq(cpu_c, gpu_c, tol),
            "change mismatch at {}: {} vs {}",
            i,
            cpu_c,
            gpu_c
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn acosc_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[acosc_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let num_series = 64usize;
    let series_len = 2048usize;
    let mut high_tm = vec![f32::NAN; num_series * series_len];
    let mut low_tm = vec![f32::NAN; num_series * series_len];

    for s in 0..num_series {
        for t in 10..series_len {
            let x = (t as f64) + (s as f64) * 0.01;
            let base = (x * 0.002).sin() + 0.001 * x;
            let h = (base + 0.75) as f32;
            let l = (base - 0.55) as f32;
            high_tm[t * num_series + s] = h;
            low_tm[t * num_series + s] = l;
        }
    }

    let cuda = CudaAcosc::new(0).expect("CudaAcosc::new");
    let pair = cuda
        .acosc_many_series_one_param_time_major_dev(&high_tm, &low_tm, num_series, series_len)
        .expect("acosc many-series");
    assert_eq!(pair.rows(), num_series);
    assert_eq!(pair.cols(), series_len);

    let mut osc_host = vec![0f32; pair.osc.len()];
    let mut chg_host = vec![0f32; pair.change.len()];
    pair.osc.buf.copy_to(&mut osc_host).unwrap();
    pair.change.buf.copy_to(&mut chg_host).unwrap();

    let tol = 8e-4;
    for s in [0usize, 7, 13, 31, 63] {
        let mut high = vec![f64::NAN; series_len];
        let mut low = vec![f64::NAN; series_len];
        for t in 0..series_len {
            let idx = t * num_series + s;
            high[t] = high_tm[idx] as f64;
            low[t] = low_tm[idx] as f64;
        }
        let input = AcoscInput {
            data: AcoscData::Slices {
                high: &high,
                low: &low,
            },
            params: AcoscParams::default(),
        };
        let cpu = acosc_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..series_len {
            let idx = t * num_series + s;
            assert!(
                approx_eq(cpu.osc[t], osc_host[idx] as f64, tol),
                "series {} osc[{}]",
                s,
                t
            );
            assert!(
                approx_eq(cpu.change[t], chg_host[idx] as f64, tol),
                "series {} change[{}]",
                s,
                t
            );
        }
    }

    Ok(())
}
