use cust::memory::CopyDestination;
use my_project::cuda::moving_averages::CudaVama;
use my_project::indicators::moving_averages::volatility_adjusted_ma::{
    vama_batch_with_kernel, VamaBatchRange, VamaInput, VamaParams,
};
use my_project::utilities::enums::Kernel;

fn main() {
    let series_len = 128usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 5..series_len {
        let x = i as f64;
        data[i] = (x * 0.00123).sin() + 0.00017 * x;
    }

    let sweep = VamaBatchRange {
        base_period: (9, 9, 0),
        vol_period: (5, 5, 0),
    };
    let cpu = vama_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch).unwrap();
    let cpu_row = &cpu.values[0..series_len];

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let cuda = CudaVama::new(0).unwrap();
    let gpu = cuda.vama_batch_dev(&data_f32, &sweep).unwrap();
    let mut gpu_row = vec![0f32; series_len];
    unsafe {
        gpu.buf.copy_to(&mut gpu_row).unwrap();
    }

    println!("first_valid=5 base=9 vol=5 warm={} ", 5 + 9 - 1);
    // Simple reference using pure high/low over vol window (should equal VAMA center)
    let mut ref_simple = vec![f64::NAN; series_len];
    let vol = 5usize;
    for i in (5 + 9 - 1)..series_len {
        let start = i + 1 - vol;
        let mut mx = f64::NEG_INFINITY;
        let mut mn = f64::INFINITY;
        for j in start..=i {
            let v = data[j];
            if v.is_finite() {
                if v > mx {
                    mx = v;
                }
                if v < mn {
                    mn = v;
                }
            }
        }
        if mx.is_finite() && mn.is_finite() {
            ref_simple[i] = 0.5 * (mx + mn);
        }
    }
    for i in 0..32 {
        println!(
            "i={:02} cpu={:.8} gpu={:.8} mid(hl/2)={:.8}",
            i, cpu_row[i], gpu_row[i] as f64, ref_simple[i]
        );
    }

    // Recompute EMA + band exactly as in CUDA kernel (double mean warmup + double recurrence)
    let first_valid = 5usize;
    let base = 9usize;
    let vol = 5usize;
    let alpha = 2.0f64 / (base as f64 + 1.0);
    let beta = 1.0f64 - alpha;
    let mut ema = vec![f64::NAN; series_len];
    let mut mean = data[first_valid];
    ema[first_valid] = mean;
    let mut count = 1f64;
    let warm_end = (first_valid + base).min(series_len);
    for i in (first_valid + 1)..warm_end {
        let p = data[i];
        if p.is_finite() {
            mean = (mean * count + p) / (count + 1.0);
            count += 1.0;
        }
        ema[i] = mean;
    }
    let mut prev = mean;
    for i in warm_end..series_len {
        let p = data[i];
        if p.is_finite() {
            prev = beta * prev + alpha * p;
        }
        ema[i] = prev;
    }
    let warm = first_valid + base.max(vol) - 1;
    let mut out_cuda_style = vec![f64::NAN; series_len];
    for i in warm..series_len {
        let mid = ema[i];
        let available = i + 1 - first_valid;
        let wlen = std::cmp::min(vol, available);
        let start = i + 1 - wlen;
        let mut up = f64::NEG_INFINITY;
        let mut dn = f64::INFINITY;
        for j in start..=i {
            let e = ema[j];
            let p = data[j];
            if e.is_finite() && p.is_finite() {
                let d = p - e;
                if d > up {
                    up = d;
                }
                if d < dn {
                    dn = d;
                }
            }
        }
        if up.is_finite() && dn.is_finite() {
            out_cuda_style[i] = mid + 0.5 * (up + dn);
        } else {
            out_cuda_style[i] = mid;
        }
    }
    println!("\nCUDA-style recompute vs CPU and GPU (first 20 valid):");
    for i in warm..(warm + 20).min(series_len) {
        println!(
            "i={:02} cpu={:.8} gpu={:.8} cuda_style={:.8}",
            i, cpu_row[i], gpu_row[i] as f64, out_cuda_style[i]
        );
    }
}
