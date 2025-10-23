use my_project::indicators::aroon::{aroon_batch_with_kernel, AroonBatchRange};
use my_project::utilities::enums::Kernel;
#[cfg(feature="cuda")]
use my_project::cuda::CudaAroon;
#[cfg(feature="cuda")]
use cust::memory::CopyDestination;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool { if a.is_nan()&&b.is_nan(){return true;} (a-b).abs()<=tol }

fn main(){
    let n = 50_500usize;
    let mut high = vec![f64::NAN; n];
    let mut low = vec![f64::NAN; n];
    for i in 7..n {
        let x = i as f64 * 0.0023;
        let base = x.sin() * 0.6 + 0.0002 * (i as f64);
        high[i] = base + 1.3 + 0.02 * (x * 1.9).cos();
        low[i] = base - 1.3 - 0.015 * (x * 1.2).sin();
    }
    let sweep = AroonBatchRange { length: (5, 60, 5) };
    let cpu = aroon_batch_with_kernel(&high, &low, &sweep, Kernel::ScalarBatch).unwrap();
    #[cfg(feature="cuda")]
    {
        let hf32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
        let lf32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
        let cuda = CudaAroon::new(0).unwrap();
        let out = cuda.aroon_batch_dev(&hf32, &lf32, &sweep).unwrap();
        let mut up_host = vec![0f32; out.outputs.first.len()];
        let mut dn_host = vec![0f32; out.outputs.second.len()];
        out.outputs.first.buf.copy_to(&mut up_host).unwrap();
        out.outputs.second.buf.copy_to(&mut dn_host).unwrap();
        let rows = cpu.rows; let cols = cpu.cols; let tol=1e-4;
        for r in 0..rows { for c in 0..cols { let idx=r*cols+c; let cu=cpu.up[idx]; let cd=cpu.down[idx]; let gu=up_host[idx] as f64; let gd=dn_host[idx] as f64; if !approx_eq(cu,gu,tol)||!approx_eq(cd,gd,tol){ println!("mismatch r={} c={} idx={} cu={:.6} gu={:.6} cd={:.6} gd={:.6}", r,c,idx, cu,gu,cd,gd); let length = 5 + r*5; let start = c.saturating_sub(length); println!("length={} start={} window=[{}..{}]", length, start, start, c); for k in start..=c { println!("t={} low={:.8} high={:.8}", k, lf32[k], hf32[k]); }
            // compute scalar baseline for this window
            let mut best_h = hf32[start] as f64; let mut off_h = 0usize; let mut best_l = lf32[start] as f64; let mut off_l = 0usize; let window = length+1; let mut valid=true; for off in 1..window { let h = hf32[start+off] as f64; let l = lf32[start+off] as f64; if !h.is_finite()||!l.is_finite(){valid=false;break;} if h>best_h{best_h=h;off_h=off;} if l<best_l{best_l=l;off_l=off;} } if !valid { println!("scalar window invalid"); } else { let dist_hi = length - off_h; let dist_lo = length - off_l; let sup = if dist_hi==0 {100.0} else if dist_hi>=length {0.0} else {(-(dist_hi as f64))*(100.0/(length as f64))+100.0}; let sdn = if dist_lo==0 {100.0} else if dist_lo>=length {0.0} else {(-(dist_lo as f64))*(100.0/(length as f64))+100.0}; println!("scalar_up={:.6} scalar_dn={:.6}", sup, sdn); }
            // compute using crate's aroon_row_scalar for the row
            let mut tmp_up = vec![f64::NAN; cols]; let mut tmp_dn = vec![f64::NAN; cols];
            unsafe { my_project::indicators::aroon::aroon_row_scalar(&high, &low, length, &mut tmp_up, &mut tmp_dn); }
            println!("row_scalar_up[c]={:.6} row_scalar_dn[c]={:.6}", tmp_up[c], tmp_dn[c]);
            return; } } }
        println!("all matched");
    }
}
