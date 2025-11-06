#[cfg(feature = "gpu")]
mod imp {
    use anyhow::{anyhow, Result};
    use cust::memory::DeviceBuffer;
    use cust::module::Module;
    use cust::prelude::*;
    use std::ffi::c_void;
    use crate::backends::types::{OptimizeRequest, OptimizeResponse, OptimizeResponseMeta, AxisMeta, expand_range};
    use my_project::cuda::moving_averages::CudaAlma;

    fn compute_weights(period: usize, offset: f64, sigma: f64) -> (Vec<f32>, f32) {
        let m = (offset * (period as f64 - 1.0)) as f32;
        let s = (period as f64 / sigma) as f32; let s2 = 2.0 * s * s;
        let mut w = vec![0f32; period]; let mut norm=0f32; for i in 0..period { let d=i as f32 - m; let wi = (-(d*d)/s2).exp(); w[i]=wi; norm+=wi; }
        (w, 1.0/norm)
    }

    fn expand_periods(r: (usize,usize,usize))->Vec<usize>{ if r.2==0||r.0==r.1{vec![r.0]} else {(r.0..=r.1).step_by(r.2).collect()} }

    pub fn run_gpu(req: OptimizeRequest) -> Result<OptimizeResponse> {
        let series = match (req.series, req.synthetic_len) {
            (Some(s), _) => s,
            (None, Some(n)) => {
                let mut v = vec![f64::NAN; n]; for i in 3..n { let x = i as f64; v[i] = (x*0.001).sin() + 0.0001*x; } v
            }
            _ => { let n=100_000; let mut v=vec![f64::NAN;n]; for i in 3..n { let x=i as f64; v[i]=(x*0.001).sin()+0.0001*x; } v }
        };

        let fast_periods = expand_periods(req.fast_period);
        let slow_periods = expand_periods(req.slow_period);
        let rows = fast_periods.len();
        let cols = slow_periods.len();
        let metrics = req.metrics.max(5).min(5);

        // Parse ALMA-specific extra parameters (offset/sigma) as ranges/enumerations
        fn values_or_range(p: &Option<serde_json::Value>, key: &str, default_val: f64) -> Vec<f64> {
            if let Some(j) = p {
                if let Some(v) = j.get(key) {
                    if let Some(arr) = v.as_array() {
                        let mut out = Vec::new();
                        for x in arr { if let Some(f) = x.as_f64() { out.push(f) } }
                        if !out.is_empty() { return out; }
                    }
                    if v.is_object() {
                        let s = v.get("start").and_then(|x| x.as_f64()).unwrap_or(default_val);
                        let e = v.get("end").and_then(|x| x.as_f64()).unwrap_or(s);
                        let st = v.get("step").and_then(|x| x.as_f64()).unwrap_or(0.0);
                        if st.abs() > 0.0 && e >= s {
                            let mut out = Vec::new();
                            let mut x = s; while x <= e + 1e-12 { out.push(x); x += st; }
                            if !out.is_empty() { return out; }
                        }
                    }
                    if let Some(f) = v.as_f64() { return vec![f]; }
                }
            }
            vec![default_val]
        }

        let f_offs = values_or_range(&req.fast_params, "offset", req.offset);
        let f_sigs = values_or_range(&req.fast_params, "sigma", req.sigma);
        let s_offs = values_or_range(&req.slow_params, "offset", req.offset);
        let s_sigs = values_or_range(&req.slow_params, "sigma", req.sigma);
        let f_ext = f_offs.len() * f_sigs.len();
        let s_ext = s_offs.len() * s_sigs.len();
        let layers = f_ext * s_ext;
        let mut out = vec![0f32; layers * rows * cols * metrics];

        // Setup CUDA context via CudaAlma (loads ALMA PTX)
        let alma = CudaAlma::new(0).map_err(|e| anyhow!(e.to_string()))?;

        // Load backtest kernel PTX compiled at build time
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/double_crossover.ptx"));
        let module = Module::from_ptx(ptx, &[])?;
        let kernel = module.get_function("double_cross_backtest_f32")?;
        let t_len = series.len();
        let first_valid = series.iter().position(|x| !x.is_nan()).unwrap_or(0) as i32;
        let d_prices = DeviceBuffer::from_slice(&series.iter().map(|&x| x as f32).collect::<Vec<_>>())?;
        let d_fast_periods = DeviceBuffer::from_slice(&fast_periods.iter().map(|&p| p as i32).collect::<Vec<_>>())?;
        let d_slow_periods = DeviceBuffer::from_slice(&slow_periods.iter().map(|&p| p as i32).collect::<Vec<_>>())?;

        // Tile planner: simple blocks to keep memory in check
        let max_pf = *fast_periods.iter().max().unwrap();
        let max_ps = *slow_periods.iter().max().unwrap();
        let pf_tile = (rows).min(512);
        let ps_tile = (cols).min(512);

        // Device tiles
        let mut d_fast_ma: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(pf_tile * t_len) }?;
        let mut d_slow_ma: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(ps_tile * t_len) }?;
        let mut d_fast_w: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(pf_tile * max_pf) }?;
        let mut d_slow_w: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(ps_tile * max_ps) }?;
        let mut d_fast_inv: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(pf_tile) }?;
        let mut d_slow_inv: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(ps_tile) }?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let block_x: u32 = 256;
        // Sweep extra ALMA params by layers: for each (fast offset,sigma) × (slow offset,sigma)
        let mut layer = 0usize;
        for &f_offv in &f_offs {
            for &f_sigv in &f_sigs {
                // Build fast MA tile once per fast extra
                let mut f_ofs = 0usize;
                while f_ofs < rows {
                    let pf = pf_tile.min(rows - f_ofs);
                    let mut fw = vec![0f32; pf * max_pf];
                    let mut fi = vec![0f32; pf];
                    for i in 0..pf {
                        let p = fast_periods[f_ofs + i]; let (w, inv) = compute_weights(p, f_offv, f_sigv);
                        fi[i] = inv; let base=i*max_pf; fw[base..base+p].copy_from_slice(&w);
                    }
                    d_fast_w.copy_from(&fw)?; d_fast_inv.copy_from(&fi)?;
                    let d_fp = DeviceBuffer::from_slice(&fast_periods[f_ofs..f_ofs+pf].iter().map(|&x| x as i32).collect::<Vec<_>>())?;
                    alma.alma_batch_device(&d_prices, &d_fast_w, &d_fp, &d_fast_inv, max_pf as i32, t_len as i32, pf as i32, first_valid, &mut d_fast_ma).map_err(|e| anyhow!(e.to_string()))?;

                    for &s_offv in &s_offs {
                        for &s_sigv in &s_sigs {
                            let mut s_ofs = 0usize;
                            while s_ofs < cols {
                                let ps = ps_tile.min(cols - s_ofs);
                                let mut sw = vec![0f32; ps * max_ps];
                                let mut si = vec![0f32; ps];
                                for j in 0..ps {
                                    let p = slow_periods[s_ofs + j]; let (w, inv) = compute_weights(p, s_offv, s_sigv);
                                    si[j] = inv; let base=j*max_ps; sw[base..base+p].copy_from_slice(&w);
                                }
                                d_slow_w.copy_from(&sw)?; d_slow_inv.copy_from(&si)?;
                                let d_sp = DeviceBuffer::from_slice(&slow_periods[s_ofs..s_ofs+ps].iter().map(|&x| x as i32).collect::<Vec<_>>())?;
                                alma.alma_batch_device(&d_prices, &d_slow_w, &d_sp, &d_slow_inv, max_ps as i32, t_len as i32, ps as i32, first_valid, &mut d_slow_ma).map_err(|e| anyhow!(e.to_string()))?;

                                // Launch backtest for this layer/tile
                                let pairs = pf * ps;
                                let grid_x = ((pairs as u32) + block_x - 1) / block_x;
                                let mut args: Vec<*mut c_void> = Vec::new();
                                unsafe {
                                    let mut f_ma = d_fast_ma.as_device_ptr().as_raw();
                                    let mut pf_i = pf as i32; let mut pf_tot = rows as i32; let mut f_of = f_ofs as i32;
                                    let mut s_ma = d_slow_ma.as_device_ptr().as_raw();
                                    let mut ps_i = ps as i32; let mut ps_tot = cols as i32; let mut s_of = s_ofs as i32;
                                    let mut fper = d_fast_periods.as_device_ptr().as_raw();
                                    let mut sper = d_slow_periods.as_device_ptr().as_raw();
                                    let mut pr = d_prices.as_device_ptr().as_raw();
                                    let mut T = t_len as i32; let mut fv = first_valid as i32;
                                    let mut comm = req.commission as f32; let mut M = metrics as i32;
                                    let mut d_tile: DeviceBuffer<f32> = DeviceBuffer::uninitialized(pairs * metrics)?;
                                    let mut out_p = d_tile.as_device_ptr().as_raw();
                                    args.extend_from_slice(&mut [
                                        &mut f_ma as *mut _ as *mut c_void,
                                        &mut pf_i as *mut _ as *mut c_void,
                                        &mut pf_tot as *mut _ as *mut c_void,
                                        &mut f_of as *mut _ as *mut c_void,
                                        &mut s_ma as *mut _ as *mut c_void,
                                        &mut ps_i as *mut _ as *mut c_void,
                                        &mut ps_tot as *mut _ as *mut c_void,
                                        &mut s_of as *mut _ as *mut c_void,
                                        &mut fper as *mut _ as *mut c_void,
                                        &mut sper as *mut _ as *mut c_void,
                                        &mut pr as *mut _ as *mut c_void,
                                        &mut T as *mut _ as *mut c_void,
                                        &mut fv as *mut _ as *mut c_void,
                                        &mut comm as *mut _ as *mut c_void,
                                        &mut M as *mut _ as *mut c_void,
                                        &mut out_p as *mut _ as *mut c_void,
                                    ]);
                                    stream.launch(&kernel, (grid_x,1,1), (block_x,1,1), 0, &mut args)?;
                                    stream.synchronize()?;
                                    let mut host_tile = vec![0f32; pairs * metrics];
                                    d_tile.copy_to(&mut host_tile)?;
                                    // Scatter into final with layer-major layout
                                    for i in 0..pf { for j in 0..ps {
                                        let f_idx = f_ofs + i; let s_idx = s_ofs + j;
                                        let base = (((layer * rows + f_idx) * cols + s_idx) * metrics) as usize;
                                        let src = (i * ps + j) * metrics;
                                        out[base..base+metrics].copy_from_slice(&host_tile[src..src+metrics]);
                                    }}
                                }
                                s_ofs += ps;
                            }
                            layer += 1;
                        }
                    }
                    f_ofs += pf;
                }
            }
        }

        let axes = vec![
            AxisMeta { name: "fast_period".to_string(), values: fast_periods.iter().map(|&x| x as f64).collect() },
            AxisMeta { name: "slow_period".to_string(), values: slow_periods.iter().map(|&x| x as f64).collect() },
            AxisMeta { name: "fast.offset".to_string(), values: f_offs.clone() },
            AxisMeta { name: "fast.sigma".to_string(), values: f_sigs.clone() },
            AxisMeta { name: "slow.offset".to_string(), values: s_offs.clone() },
            AxisMeta { name: "slow.sigma".to_string(), values: s_sigs.clone() },
        ];

        let meta = OptimizeResponseMeta {
            fast_periods, slow_periods,
            metrics: vec!["total_return","trades","max_dd","mean_ret","std_ret"],
            rows, cols,
            axes,
        };
        Ok(OptimizeResponse { meta, values: out, layers })
    }
}

#[cfg(feature = "gpu")]
pub use imp::run_gpu;
