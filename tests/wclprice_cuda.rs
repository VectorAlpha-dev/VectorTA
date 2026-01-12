use vector_ta::indicators::wclprice::{
    wclprice_with_kernel, WclpriceData, WclpriceInput, WclpriceParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaWclprice};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    (a - b).abs() <= tol
}

#[test]
fn cuda_feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    assert!(true);
}

#[cfg(feature = "cuda")]
#[test]
fn wclprice_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[wclprice_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let len = 4096usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    for i in 5..len {
        let x = i as f64;
        close[i] = (x * 0.0021).sin() + 0.00011 * x;
        let off = (0.003 * x.sin()).abs() + 0.1;
        high[i] = close[i] + off;
        low[i] = close[i] - off;
    }

    let input = WclpriceInput {
        data: WclpriceData::Slices {
            high: &high,
            low: &low,
            close: &close,
        },
        params: WclpriceParams,
    };
    let cpu = wclprice_with_kernel(&input, Kernel::Scalar)?;

    let hf: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let lf: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let cf: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let cuda = CudaWclprice::new(0).expect("CudaWclprice::new");
    let dev = cuda
        .wclprice_batch_dev(
            &hf,
            &lf,
            &cf,
            &vector_ta::indicators::wclprice::WclpriceBatchRange,
        )
        .expect("wclprice_batch_dev");
    assert_eq!(dev.rows, 1);
    assert_eq!(dev.cols, len);
    let mut out = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut out)?;

    let tol = 1e-4;
    for i in 0..len {
        assert!(
            approx_eq(cpu.values[i], out[i] as f64, tol),
            "mismatch at {}",
            i
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn wclprice_cuda_many_series_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[wclprice_cuda_many_series_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let cols = 8usize;
    let rows = 1024usize;
    let mut high_tm = vec![f64::NAN; cols * rows];
    let mut low_tm = vec![f64::NAN; cols * rows];
    let mut close_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let idx = t * cols + s;
            let x = (t as f64) + (s as f64) * 0.2;
            close_tm[idx] = (x * 0.002).sin() + 0.0003 * x;
            let off = (0.0031 * x.sin()).abs() + 0.12;
            high_tm[idx] = close_tm[idx] + off;
            low_tm[idx] = close_tm[idx] - off;
        }
    }

    let mut cpu_tm = vec![f64::NAN; cols * rows];
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
        let input = WclpriceInput {
            data: WclpriceData::Slices {
                high: &h,
                low: &l,
                close: &c,
            },
            params: WclpriceParams,
        };
        let out = wclprice_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out.values[t];
        }
    }

    let hf: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let lf: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let cf: Vec<f32> = close_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaWclprice::new(0).expect("CudaWclprice::new");
    let dev = cuda
        .wclprice_many_series_one_param_time_major_dev(&hf, &lf, &cf, cols, rows)
        .expect("wclprice many");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);
    let mut out_tm = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut out_tm)?;

    let tol = 1e-4;
    for i in 0..out_tm.len() {
        assert!(
            approx_eq(cpu_tm[i], out_tm[i] as f64, tol),
            "mismatch at {}",
            i
        );
    }
    Ok(())
}
