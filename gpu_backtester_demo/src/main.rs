use anyhow::{anyhow, Result};
use clap::{Parser, ValueEnum};
use cust::memory::DeviceBuffer;
use cust::module::Module;
use cust::prelude::*;
use std::ffi::c_void;

// Import the ALMA CUDA wrapper from the parent library crate
use my_project::cuda::moving_averages::CudaAlma;

#[derive(Debug, Clone, Parser)]
#[command(name = "gpu_backtester_demo", version, about = "GPU-only double-crossover optimizer demo (ALMA only)")]
struct Cli {
    /// CSV file with price data (expects a header). If omitted, synthetic data is generated.
    #[arg(long)]
    csv: Option<String>,

    /// Column name to use from CSV (default: close)
    #[arg(long, default_value = "close")]
    column: String,

    /// Synthetic length (used when --csv is not provided)
    #[arg(long, default_value_t = 100_000)]
    synth_len: usize,

    /// Fast ALMA period range as start:end:step
    #[arg(long, default_value = "5:100:1")]
    fast_period: String,

    /// Slow ALMA period range as start:end:step
    #[arg(long, default_value = "20:200:2")]
    slow_period: String,

    /// ALMA offset (single value applied to both fast and slow)
    #[arg(long, default_value_t = 0.85)]
    offset: f64,

    /// ALMA sigma (single value applied to both fast and slow)
    #[arg(long, default_value_t = 6.0)]
    sigma: f64,

    /// Commission rate (fraction) applied on entries/exits (e.g., 0.0005)
    #[arg(long, default_value_t = 0.0, alias = "fee")]
    commission: f32,

    /// Neutral band as a fraction of |slow| to avoid churn (0 disables it)
    #[arg(long, default_value_t = 0.0)]
    eps_rel: f32,

    /// Enable long-only (restrict to {0,+1})
    #[arg(long, default_value_t = false)]
    long_only: bool,

    /// Do not flip directly; exit then wait
    #[arg(long, default_value_t = false)]
    no_flip: bool,

    /// Use t-1 signal and enter/exit on next bar
    #[arg(long, default_value_t = false)]
    trade_on_next: bool,

    /// Enforce fast_period < slow_period (skip invalid combos)
    #[arg(long, default_value_t = true)]
    enforce_fast_lt_slow: bool,

    /// Also compute signed exposure (net position avg) into metric[6]
    #[arg(long, default_value_t = false)]
    signed_exposure: bool,

    /// Max tile sizes (Pf_tile and Ps_tile). If 0, auto-choose a conservative value.
    #[arg(long, default_value_t = 0)]
    fast_tile: usize,
    #[arg(long, default_value_t = 0)]
    slow_tile: usize,

    /// Metrics count written by kernel (fixed to 5 for now)
    #[arg(long, default_value_t = 5)]
    metrics: usize,
}

fn parse_range(spec: &str) -> Result<(usize, usize, usize)> {
    let parts: Vec<_> = spec.split(':').collect();
    if parts.len() != 3 { return Err(anyhow!("range must be start:end:step")); }
    let s: usize = parts[0].parse()?;
    let e: usize = parts[1].parse()?;
    let st: usize = parts[2].parse()?;
    Ok((s, e, st))
}

fn load_prices_from_csv(path: &str, column: &str) -> Result<Vec<f64>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let headers = rdr.headers()?.clone();
    let idx = headers.iter().position(|h| h == column)
        .ok_or_else(|| anyhow!("column '{}' not found in CSV", column))?;
    let mut out = Vec::new();
    for rec in rdr.into_records() {
        let rec = rec?;
        let v: f64 = rec.get(idx)
            .ok_or_else(|| anyhow!("missing value in column"))?
            .parse()?;
        out.push(v);
    }
    Ok(out)
}

fn gen_synthetic_prices(n: usize) -> Vec<f64> {
    let mut v = Vec::with_capacity(n);
    let mut price = 100.0;
    for i in 0..n {
        // simple random walk-ish deterministic pattern
        let drift = 0.00002;
        let seasonal = (i as f64 * 0.0005).sin() * 0.001;
        price *= 1.0 + drift + seasonal;
        v.push(price);
    }
    v
}

#[inline]
fn compute_weights_f32(period: usize, offset: f64, sigma: f64) -> (Vec<f32>, f32) {
    let m = (offset * (period as f64 - 1.0)) as f32;
    let s = (period as f64 / sigma) as f32;
    let s2 = 2.0f32 * s * s;
    let mut w = vec![0.0f32; period];
    let mut norm = 0.0f32;
    for i in 0..period {
        let diff = i as f32 - m;
        let wi = (-(diff * diff) / s2).exp();
        w[i] = wi;
        norm += wi;
    }
    (w, 1.0f32 / norm)
}

fn expand_periods((s, e, st): (usize, usize, usize)) -> Vec<usize> {
    if st == 0 || s == e { return vec![s]; }
    (s..=e).step_by(st).collect()
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Load price series
    let mut prices: Vec<f64> = if let Some(p) = &cli.csv { load_prices_from_csv(p, &cli.column)? } else { gen_synthetic_prices(cli.synth_len) };
    if prices.is_empty() { return Err(anyhow!("no price data")); }
    let first_valid = prices.iter().position(|x| !x.is_nan()).unwrap_or(0);
    let t_len = prices.len();

    // Parameter grids
    let fast_periods = expand_periods(parse_range(&cli.fast_period)?);
    let slow_periods = expand_periods(parse_range(&cli.slow_period)?);
    if fast_periods.is_empty() || slow_periods.is_empty() { return Err(anyhow!("empty parameter ranges")); }
    let f_total = fast_periods.len();
    let s_total = slow_periods.len();
    let max_pf = *fast_periods.iter().max().unwrap();
    let max_ps = *slow_periods.iter().max().unwrap();

    // Create ALMA CUDA context and module
    let alma = CudaAlma::new(0).map_err(|e| anyhow!("{:?}", e))?;
    // The Context created in CudaAlma::new() is now current on this thread.

    // Device buffers that persist across tiles
    let prices_f32: Vec<f32> = prices.iter().map(|&x| x as f32).collect();
    let d_prices = DeviceBuffer::from_slice(&prices_f32)?;
    let d_fast_periods = DeviceBuffer::from_slice(&fast_periods.iter().map(|&p| p as i32).collect::<Vec<_>>())?;
    let d_slow_periods = DeviceBuffer::from_slice(&slow_periods.iter().map(|&p| p as i32).collect::<Vec<_>>())?;

    // Prepare backtest kernel module (compiled by this demo crate)
    let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/double_crossover.ptx"));
    let bt_module = Module::from_ptx(ptx, &[])?;

    // Precompute log returns once on device (lr[0]=0, lr[t]=log(p[t])-log(p[t-1]))
    let mut d_lr: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(t_len)? };
    {
        let kernel = bt_module.get_function("compute_log_returns_f32")?;
        let block_x: u32 = 256;
        let grid_x: u32 = ((t_len as u32) + block_x - 1) / block_x;
        unsafe {
            let mut pr = d_prices.as_device_ptr().as_raw();
            let mut t_i = t_len as i32;
            let mut lr = d_lr.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut pr as *mut _ as *mut c_void,
                &mut t_i as *mut _ as *mut c_void,
                &mut lr as *mut _ as *mut c_void,
            ];
            let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
            stream.launch(&kernel, (grid_x, 1, 1), (block_x, 1, 1), 0, args)?;
            stream.synchronize()?;
        }
    }

    // Simple tile planner
    let (pf_tile, ps_tile) = choose_tiles(cli.fast_tile, cli.slow_tile, t_len, max_pf, max_ps, cli.metrics)?;

    // Allocate output metrics on device if it fits; else collect on host per tile
    let total_pairs = f_total * s_total;
    let metrics = cli.metrics;
    let metrics_bytes = total_pairs * metrics * std::mem::size_of::<f32>();
    let headroom = 64usize * 1024 * 1024;
    let d_metrics_global = if will_fit(metrics_bytes, headroom) { Some(unsafe { DeviceBuffer::<f32>::uninitialized(total_pairs * metrics)? }) } else { None };
    let mut host_metrics: Vec<f32> = if d_metrics_global.is_some() { Vec::new() } else { vec![0.0; total_pairs * metrics] };

    // Temporary buffers reused per tile
    // These per-tile buffers will be allocated to the actual tile size inside the loops
    // to avoid partial copies.
    // Host-side workspace for weights (flattened per tile)
    // We'll resize these on each tile iteration as needed.
    let mut fast_w_flat: Vec<f32> = Vec::new();
    let mut slow_w_flat: Vec<f32> = Vec::new();
    let mut fast_inv: Vec<f32> = Vec::new();
    let mut slow_inv: Vec<f32> = Vec::new();

    // Device buffers declared here to be reused across tiles when size matches; otherwise reallocated
    let mut d_fast_ma: Option<DeviceBuffer<f32>> = None;      // [Pf, T] row-major
    let mut d_slow_ma: Option<DeviceBuffer<f32>> = None;      // [Ps, T] row-major
    let mut d_fast_ma_tm: Option<DeviceBuffer<f32>> = None;   // [T, Pf] time-major
    let mut d_slow_ma_tm: Option<DeviceBuffer<f32>> = None;   // [T, Ps] time-major
    let mut d_fast_w: Option<DeviceBuffer<f32>> = None;
    let mut d_slow_w: Option<DeviceBuffer<f32>> = None;
    let mut d_fast_inv: Option<DeviceBuffer<f32>> = None;
    let mut d_slow_inv: Option<DeviceBuffer<f32>> = None;
    let mut d_fast_p: Option<DeviceBuffer<i32>> = None;
    let mut d_slow_p: Option<DeviceBuffer<i32>> = None;

    // Stream for backtest launches
    let bt_stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Iterate tiles
    let mut f_start = 0;
    while f_start < f_total {
        let pf = pf_tile.min(f_total - f_start);
        // Build fast tile weights and periods
        fast_w_flat.resize(pf * max_pf, 0.0);
        fast_inv.resize(pf, 0.0);
        for i in 0..pf {
            let per = fast_periods[f_start + i];
            let (w, inv) = compute_weights_f32(per, cli.offset, cli.sigma);
            fast_inv[i] = inv;
            let base = i * max_pf;
            fast_w_flat[base..base + per].copy_from_slice(&w);
        }
        // Allocate/reuse device buffers sized to the current tile
        if d_fast_w.as_ref().map(|b| b.len()) != Some(pf * max_pf) {
            d_fast_w = Some(unsafe { DeviceBuffer::<f32>::uninitialized(pf * max_pf)? });
        }
        if d_fast_inv.as_ref().map(|b| b.len()) != Some(pf) {
            d_fast_inv = Some(unsafe { DeviceBuffer::<f32>::uninitialized(pf)? });
        }
        if d_fast_p.as_ref().map(|b| b.len()) != Some(pf) {
            d_fast_p = Some(unsafe { DeviceBuffer::<i32>::uninitialized(pf)? });
        }
        if d_fast_ma.as_ref().map(|b| b.len()) != Some(pf * t_len) {
            d_fast_ma = Some(unsafe { DeviceBuffer::<f32>::uninitialized(pf * t_len)? });
        }
        d_fast_w.as_mut().unwrap().copy_from(&fast_w_flat)?;
        d_fast_inv.as_mut().unwrap().copy_from(&fast_inv)?;
        let host_p: Vec<i32> = fast_periods[f_start..f_start + pf].iter().map(|&x| x as i32).collect();
        d_fast_p.as_mut().unwrap().copy_from(&host_p)?;
        // Compute fast ALMA tile
        alma.alma_batch_device(
            &d_prices,
            d_fast_w.as_ref().unwrap(),
            d_fast_p.as_ref().unwrap(),
            d_fast_inv.as_ref().unwrap(),
            max_pf as i32,
            t_len as i32,
            pf as i32,
            first_valid as i32,
            d_fast_ma.as_mut().unwrap(),
        ).map_err(|e| anyhow!("{:?}", e))?;

        let mut s_start = 0;
        while s_start < s_total {
            let ps = ps_tile.min(s_total - s_start);
            // Build slow tile weights and periods
            slow_w_flat.resize(ps * max_ps, 0.0);
            slow_inv.resize(ps, 0.0);
            for j in 0..ps {
                let per = slow_periods[s_start + j];
                let (w, inv) = compute_weights_f32(per, cli.offset, cli.sigma);
                slow_inv[j] = inv;
                let base = j * max_ps;
                slow_w_flat[base..base + per].copy_from_slice(&w);
            }
            if d_slow_w.as_ref().map(|b| b.len()) != Some(ps * max_ps) {
                d_slow_w = Some(unsafe { DeviceBuffer::<f32>::uninitialized(ps * max_ps)? });
            }
            if d_slow_inv.as_ref().map(|b| b.len()) != Some(ps) {
                d_slow_inv = Some(unsafe { DeviceBuffer::<f32>::uninitialized(ps)? });
            }
            if d_slow_p.as_ref().map(|b| b.len()) != Some(ps) {
                d_slow_p = Some(unsafe { DeviceBuffer::<i32>::uninitialized(ps)? });
            }
            if d_slow_ma.as_ref().map(|b| b.len()) != Some(ps * t_len) {
                d_slow_ma = Some(unsafe { DeviceBuffer::<f32>::uninitialized(ps * t_len)? });
            }
            d_slow_w.as_mut().unwrap().copy_from(&slow_w_flat)?;
            d_slow_inv.as_mut().unwrap().copy_from(&slow_inv)?;
            let host_p: Vec<i32> = slow_periods[s_start..s_start + ps].iter().map(|&x| x as i32).collect();
            d_slow_p.as_mut().unwrap().copy_from(&host_p)?;
            // Compute slow ALMA tile
            alma.alma_batch_device(
                &d_prices,
                d_slow_w.as_ref().unwrap(),
                d_slow_p.as_ref().unwrap(),
                d_slow_inv.as_ref().unwrap(),
                max_ps as i32,
                t_len as i32,
                ps as i32,
                first_valid as i32,
                d_slow_ma.as_mut().unwrap(),
            ).map_err(|e| anyhow!("{:?}", e))?;

            // Transpose MAs to time-major for coalesced reads: [Pf, T] -> [T, Pf]; [Ps, T] -> [T, Ps]
            let tr = bt_module.get_function("transpose_row_to_tm")?;
            // Fast
            if d_fast_ma_tm.as_ref().map(|b| b.len()) != Some(t_len * pf) {
                d_fast_ma_tm = Some(unsafe { DeviceBuffer::<f32>::uninitialized(t_len * pf)? });
            }
            unsafe {
                let mut in_ptr = d_fast_ma.as_ref().unwrap().as_device_ptr().as_raw();
                let mut rows = pf as i32;
                let mut cols = t_len as i32;
                let mut out_ptr = d_fast_ma_tm.as_ref().unwrap().as_device_ptr().as_raw();
                let block_x: u32 = 256;
                let grid_x: u32 = (((pf * t_len) as u32) + block_x - 1) / block_x;
                let args: &mut [*mut c_void] = &mut [
                    &mut in_ptr as *mut _ as *mut c_void,
                    &mut rows as *mut _ as *mut c_void,
                    &mut cols as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
                stream.launch(&tr, (grid_x, 1, 1), (block_x, 1, 1), 0, args)?;
                stream.synchronize()?;
            }
            // Slow
            if d_slow_ma_tm.as_ref().map(|b| b.len()) != Some(t_len * ps) {
                d_slow_ma_tm = Some(unsafe { DeviceBuffer::<f32>::uninitialized(t_len * ps)? });
            }
            unsafe {
                let mut in_ptr = d_slow_ma.as_ref().unwrap().as_device_ptr().as_raw();
                let mut rows = ps as i32;
                let mut cols = t_len as i32;
                let mut out_ptr = d_slow_ma_tm.as_ref().unwrap().as_device_ptr().as_raw();
                let block_x: u32 = 256;
                let grid_x: u32 = (((ps * t_len) as u32) + block_x - 1) / block_x;
                let args: &mut [*mut c_void] = &mut [
                    &mut in_ptr as *mut _ as *mut c_void,
                    &mut rows as *mut _ as *mut c_void,
                    &mut cols as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
                stream.launch(&tr, (grid_x, 1, 1), (block_x, 1, 1), 0, args)?;
                stream.synchronize()?;
            }

            // Launch backtest (grid-stride inside kernel)
            let kernel = bt_module.get_function("double_cross_backtest_tm_flex_f32")?;
            let pairs = pf * ps;
            let block_x: u32 = 256;
            let grid_x: u32 = ((pairs as u32) + block_x - 1) / block_x;
            let mut flags: u32 = 0;
            if cli.long_only { flags |= 1u32 << 0; }
            if cli.no_flip { flags |= 1u32 << 1; }
            if cli.trade_on_next { flags |= 1u32 << 2; }
            if cli.enforce_fast_lt_slow { flags |= 1u32 << 3; }
            if cli.signed_exposure { flags |= 1u32 << 4; }

            let mut tile_host_buf: Vec<f32> = Vec::new();
            if let Some(ref d_global) = d_metrics_global {
                // d_metrics_global exists; launch kernel to write directly there.
                unsafe {
                    let mut f_ma_tm = d_fast_ma_tm.as_ref().unwrap().as_device_ptr().as_raw();
                    let mut s_ma_tm = d_slow_ma_tm.as_ref().unwrap().as_device_ptr().as_raw();
                    let mut lr = d_lr.as_device_ptr().as_raw();
                    let mut T = t_len as i32;
                    let mut pf_tile_i = pf as i32;
                    let mut ps_tile_i = ps as i32;
                    let mut pf_total_i = f_total as i32;
                    let mut ps_total_i = s_total as i32;
                    let mut f_off = f_start as i32;
                    let mut s_off = s_start as i32;
                    let mut f_per = d_fast_periods.as_device_ptr().as_raw();
                    let mut s_per = d_slow_periods.as_device_ptr().as_raw();
                    let mut fv = first_valid as i32;
                    let mut commission = cli.commission as f32;
                    let mut eps_rel = cli.eps_rel as f32;
                    let mut flags_u = flags as u32;
                    let mut M = metrics as i32;
                    let mut out = d_global.as_device_ptr().as_raw();

                    let args: &mut [*mut c_void] = &mut [
                        &mut f_ma_tm as *mut _ as *mut c_void,
                        &mut s_ma_tm as *mut _ as *mut c_void,
                        &mut lr as *mut _ as *mut c_void,
                        &mut T as *mut _ as *mut c_void,
                        &mut pf_tile_i as *mut _ as *mut c_void,
                        &mut ps_tile_i as *mut _ as *mut c_void,
                        &mut pf_total_i as *mut _ as *mut c_void,
                        &mut ps_total_i as *mut _ as *mut c_void,
                        &mut f_off as *mut _ as *mut c_void,
                        &mut s_off as *mut _ as *mut c_void,
                        &mut f_per as *mut _ as *mut c_void,
                        &mut s_per as *mut _ as *mut c_void,
                        &mut fv as *mut _ as *mut c_void,
                        &mut commission as *mut _ as *mut c_void,
                        &mut eps_rel as *mut _ as *mut c_void,
                        &mut flags_u as *mut _ as *mut c_void,
                        &mut M as *mut _ as *mut c_void,
                        &mut out as *mut _ as *mut c_void,
                    ];
                    bt_stream.launch(&kernel, (grid_x, 1, 1), (block_x, 1, 1), 0, args)?;
                }
                bt_stream.synchronize()?;
            } else {
                tile_host_buf.resize(pairs * metrics, 0.0);
                // We'll allocate a scratch device buffer for the tile metrics
                let mut d_tile: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(pairs * metrics)? };
                unsafe {
                    let mut f_ma_tm = d_fast_ma_tm.as_ref().unwrap().as_device_ptr().as_raw();
                    let mut s_ma_tm = d_slow_ma_tm.as_ref().unwrap().as_device_ptr().as_raw();
                    let mut lr = d_lr.as_device_ptr().as_raw();
                    let mut T = t_len as i32;
                    let mut pf_tile_i = pf as i32;
                    let mut ps_tile_i = ps as i32;
                    let mut pf_total_i = pf as i32; // local dims
                    let mut ps_total_i = ps as i32; // local dims
                    let mut f_off = 0i32;
                    let mut s_off = 0i32;
                    let mut f_per = d_fast_p.as_ref().unwrap().as_device_ptr().as_raw();
                    let mut s_per = d_slow_p.as_ref().unwrap().as_device_ptr().as_raw();
                    let mut fv = first_valid as i32;
                    let mut commission = cli.commission as f32;
                    let mut eps_rel = cli.eps_rel as f32;
                    let mut flags_u = flags as u32;
                    let mut M = metrics as i32;
                    let mut out = d_tile.as_device_ptr().as_raw();

                    let args: &mut [*mut c_void] = &mut [
                        &mut f_ma_tm as *mut _ as *mut c_void,
                        &mut s_ma_tm as *mut _ as *mut c_void,
                        &mut lr as *mut _ as *mut c_void,
                        &mut T as *mut _ as *mut c_void,
                        &mut pf_tile_i as *mut _ as *mut c_void,
                        &mut ps_tile_i as *mut _ as *mut c_void,
                        &mut pf_total_i as *mut _ as *mut c_void,
                        &mut ps_total_i as *mut _ as *mut c_void,
                        &mut f_off as *mut _ as *mut c_void,
                        &mut s_off as *mut _ as *mut c_void,
                        &mut f_per as *mut _ as *mut c_void,
                        &mut s_per as *mut _ as *mut c_void,
                        &mut fv as *mut _ as *mut c_void,
                        &mut commission as *mut _ as *mut c_void,
                        &mut eps_rel as *mut _ as *mut c_void,
                        &mut flags_u as *mut _ as *mut c_void,
                        &mut M as *mut _ as *mut c_void,
                        &mut out as *mut _ as *mut c_void,
                    ];
                    bt_stream.launch(&kernel, (grid_x, 1, 1), (block_x, 1, 1), 0, args)?;
                }
                bt_stream.synchronize()?;
                d_tile.copy_to(&mut tile_host_buf)?;
                // Scatter tile into host_metrics
                for i in 0..pf {
                    for j in 0..ps {
                        let f_idx = f_start + i;
                        let s_idx = s_start + j;
                        let pair_global = f_idx * s_total + s_idx;
                        let src_base = (i * ps + j) * metrics;
                        let dst_base = pair_global * metrics;
                        host_metrics[dst_base..dst_base + metrics]
                            .copy_from_slice(&tile_host_buf[src_base..src_base + metrics]);
                    }
                }
            }

            s_start += ps;
        }

        f_start += pf;
    }

    // If using device buffer, copy back to host now
    let mut metrics_host: Vec<f32> = if let Some(d) = d_metrics_global {
        let mut out = vec![0.0f32; total_pairs * metrics];
        d.copy_to(&mut out)?;
        out
    } else {
        host_metrics
    };

    // Print a tiny summary
    println!("Computed {} pairs ({} x {}), metrics per pair = {}", total_pairs, f_total, s_total, metrics);
    println!("Example first 5 pairs (total_return, trades, max_dd, mean, std):");
    for k in 0..total_pairs.min(5) {
        let b = k * metrics;
        println!(
            "pair {} -> [{:.4}, {:.0}, {:.4}, {:.6}, {:.6}]",
            k,
            metrics_host[b + 0], metrics_host[b + 1], metrics_host[b + 2], metrics_host[b + 3], metrics_host[b + 4]
        );
    }

    Ok(())
}

fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
    unsafe {
        let mut free: usize = 0;
        let mut total: usize = 0;
        let res = cust::sys::cuMemGetInfo_v2(&mut free as *mut usize, &mut total as *mut usize);
        if res != cust::sys::CUresult::CUDA_SUCCESS { return true; }
        required_bytes.saturating_add(headroom_bytes) <= free
    }
}

fn choose_tiles(fast_tile: usize, slow_tile: usize, t_len: usize, max_pf: usize, max_ps: usize, metrics: usize) -> Result<(usize, usize)> {
    if fast_tile > 0 && slow_tile > 0 { return Ok((fast_tile, slow_tile)); }
    // Conservative auto-tile: aim for ~256MB budget for ALMA tiles + 128MB for other
    let budget = 256usize * 1024 * 1024;
    let bytes_per_fast = t_len * std::mem::size_of::<f32>() + max_pf * std::mem::size_of::<f32>();
    let bytes_per_slow = t_len * std::mem::size_of::<f32>() + max_ps * std::mem::size_of::<f32>();
    // Small base to avoid division by zero
    let mut pf = (budget / (bytes_per_fast.max(1))) / 2; // split budget roughly in half
    let mut ps = (budget / (bytes_per_slow.max(1))) / 2;
    pf = pf.clamp(1, 4096);
    ps = ps.clamp(1, 4096);
    Ok((pf, ps))
}
