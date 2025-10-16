#![cfg(feature = "cuda")]

//! CUDA wrapper for the DevStop (Deviation Stop) indicator.
//!
//! Scope (parity with ALMA wrapper conventions where sensible):
//! - PTX embedded via include_str!(concat!(env!("OUT_DIR"), "/devstop_kernel.ptx"))
//! - NON_BLOCKING stream; module created in the active device context
//! - Batch entry uses host-side precomputation of two-bar range prefixes and groups
//!   parameter combos by period to allow fixed shared-mem allocation per launch.
//! - Many-series × one-param entry mirrors the scalar fused path (SMA/stddev), per series.
//! - Warmup/NaN semantics match scalar: warm = first_valid + 2*period − 1; outputs before warm are NaN.
//!
//! Limitations (by design):
//! - devtype=0 (standard deviation) only; mean/median-abs deviations fallback to CPU paths.
//! - Range mean uses SMA (prefix sums). EMA variants are not provided in the batch kernel.

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::devstop::DevStopBatchRange;
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, CopyDestination, DeviceBuffer};
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaDevStopError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaDevStopError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaDevStopError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaDevStopError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaDevStopError {}

#[derive(Clone, Debug)]
pub struct DevStopCombo { pub period: usize, pub mult: f32 }

pub struct CudaDevStop {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaDevStop {
    pub fn new(device_id: usize) -> Result<Self, CudaDevStopError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaDevStopError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaDevStopError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaDevStopError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/devstop_kernel.ptx"));
        let module = Module::from_ptx(ptx, &[]).map_err(|e| CudaDevStopError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaDevStopError::Cuda(e.to_string()))?;

        Ok(Self { module, stream, _context: context })
    }

    fn expand_grid(range: &DevStopBatchRange) -> Vec<(usize, f32, usize)> {
        fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
            if step == 0 || start == end { return vec![start]; }
            (start..=end).step_by(step).collect()
        }
        fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
            if step.abs() < 1e-12 || (start - end).abs() < 1e-12 { return vec![start]; }
            let mut v = Vec::new();
            let mut x = start; while x <= end + 1e-12 { v.push(x); x += step; } v
        }
        let periods = axis_usize(range.period);
        let mults = axis_f64(range.mult);
        let devtypes = axis_usize(range.devtype);
        let mut out = Vec::with_capacity(periods.len() * mults.len() * devtypes.len());
        for &p in &periods { for &m in &mults { for &d in &devtypes { out.push((p, m as f32, d)); } } }
        out
    }

    fn first_valid_hl(high: &[f32], low: &[f32]) -> Option<usize> {
        let fh = high.iter().position(|v| !v.is_nan());
        let fl = low .iter().position(|v| !v.is_nan());
        match (fh, fl) { (Some(h), Some(l)) => Some(h.min(l)), _ => None }
    }

    fn build_range_prefixes(high: &[f32], low: &[f32]) -> (Vec<f64>, Vec<f64>, Vec<i32>, usize) {
        let len = high.len().min(low.len());
        let first = Self::first_valid_hl(high, low).unwrap_or(0);
        // r[i] is defined from (i-1,i); we accumulate prefixes directly skipping NaNs
        let mut p1 = vec![0.0f64; len + 1];
        let mut p2 = vec![0.0f64; len + 1];
        let mut pc = vec![0i32;  len + 1];
        let mut acc1 = 0.0f64; let mut acc2 = 0.0f64; let mut accc = 0i32;
        let mut prev_h = if first < len { high[first] } else { f32::NAN };
        let mut prev_l = if first < len { low[first]  } else { f32::NAN };
        for i in 0..len {
            if i >= first + 1 {
                let h = high[i]; let l = low[i];
                if !h.is_nan() && !l.is_nan() && !prev_h.is_nan() && !prev_l.is_nan() {
                    let hi2 = if h > prev_h { h } else { prev_h } as f64;
                    let lo2 = if l < prev_l { l } else { prev_l } as f64;
                    let r = hi2 - lo2;
                    acc1 += r; acc2 += r * r; accc += 1;
                }
                prev_h = h; prev_l = l;
            }
            p1[i + 1] = acc1; p2[i + 1] = acc2; pc[i + 1] = accc;
        }
        (p1, p2, pc, first)
    }

    pub fn devstop_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
        sweep: &DevStopBatchRange,
        is_long: bool,
    ) -> Result<(DeviceArrayF32, Vec<(usize, f32)>), CudaDevStopError> {
        let len = high.len().min(low.len());
        if len == 0 { return Err(CudaDevStopError::InvalidInput("empty inputs".into())); }
        let first = Self::first_valid_hl(high, low)
            .ok_or_else(|| CudaDevStopError::InvalidInput("all values are NaN".into()))?;

        let combos_raw = Self::expand_grid(sweep);
        if combos_raw.is_empty() { return Err(CudaDevStopError::InvalidInput("no parameter combos".into())); }
        // Validate supported subset: devtype==0 only
        for &(_, _, dt) in &combos_raw {
            if dt != 0 {
                return Err(CudaDevStopError::InvalidInput(
                    "unsupported devtype (only 0=stddev supported in CUDA batch)".into(),
                ));
            }
        }

        // Group combos by period to allow per-launch shared memory sizing
        let mut groups: BTreeMap<usize, Vec<f32>> = BTreeMap::new();
        let mut meta: Vec<(usize, f32)> = Vec::with_capacity(combos_raw.len());
        for (p, m, _dt) in combos_raw { groups.entry(p).or_default().push(m); meta.push((p, m)); }

        // Host prefixes over r (SMA-based)
        let (p1, p2, pc, first_valid) = Self::build_range_prefixes(high, low);

        // Upload invariant inputs
        let d_high = DeviceBuffer::from_slice(&high[..len]).map_err(|e| CudaDevStopError::Cuda(e.to_string()))?;
        let d_low  = DeviceBuffer::from_slice(&low[..len ]) .map_err(|e| CudaDevStopError::Cuda(e.to_string()))?;
        let d_p1   = DeviceBuffer::from_slice(&p1)        .map_err(|e| CudaDevStopError::Cuda(e.to_string()))?;
        let d_p2   = DeviceBuffer::from_slice(&p2)        .map_err(|e| CudaDevStopError::Cuda(e.to_string()))?;
        let d_pc   = DeviceBuffer::from_slice(&pc)        .map_err(|e| CudaDevStopError::Cuda(e.to_string()))?;

        // Final output buffer (rows = total combos, cols = len). VRAM check with headroom.
        let total_rows = meta.len();
        let bytes_inputs = (high.len() + low.len()) * std::mem::size_of::<f32>()
            + (p1.len() + p2.len()) * std::mem::size_of::<f64>()
            + pc.len() * std::mem::size_of::<i32>();
        let bytes_out = total_rows * len * std::mem::size_of::<f32>();
        if let Ok((free, _total)) = mem_get_info() {
            let needed = (bytes_inputs + bytes_out) as u64 + 64 * 1024 * 1024u64;
            if needed > (free as u64) {
                return Err(CudaDevStopError::InvalidInput(format!(
                    "insufficient VRAM: need ~{} MB, free {} MB",
                    needed / (1024 * 1024),
                    (free as u64) / (1024 * 1024)
                )));
            }
        }
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total_rows * len) }
            .map_err(|e| CudaDevStopError::Cuda(e.to_string()))?;

        let func = self.module
            .get_function("devstop_batch_grouped_f32")
            .map_err(|e| CudaDevStopError::Cuda(e.to_string()))?;

        let mut out_row_base = 0usize;
        for (period, mults_host) in groups.into_iter() {
            if period == 0 || period > len { return Err(CudaDevStopError::InvalidInput(format!("invalid period {}", period))); }
            let n = mults_host.len();
            let d_mults = DeviceBuffer::from_slice(&mults_host)
                .map_err(|e| CudaDevStopError::Cuda(e.to_string()))?;

            // grid.x = n combos, one block per combo; block.x parallelizes NaN init
            let block_x: u32 = 128;
            let grid_x: u32 = (n as u32).max(1);
            let grid: GridSize = (grid_x, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();

            // shared mem size: period*(sizeof(float) + sizeof(int))
            let shmem_bytes = (period * (std::mem::size_of::<f32>() + std::mem::size_of::<i32>())) as u32;

            unsafe {
                let mut high_ptr = d_high.as_device_ptr().as_raw();
                let mut low_ptr  = d_low.as_device_ptr().as_raw();
                let mut p1_ptr   = d_p1.as_device_ptr().as_raw();
                let mut p2_ptr   = d_p2.as_device_ptr().as_raw();
                let mut pc_ptr   = d_pc.as_device_ptr().as_raw();
                let mut len_i = len as i32;
                let mut first_i = (first_valid as i32).min(len_i);
                let mut period_i = period as i32;
                let mut mults_ptr = d_mults.as_device_ptr().as_raw();
                let mut n_i = n as i32;
                let mut long_i = if is_long { 1i32 } else { 0i32 };
                let mut base_i = out_row_base as i32;
                let mut out_ptr = d_out.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut high_ptr as *mut _ as *mut c_void,
                    &mut low_ptr  as *mut _ as *mut c_void,
                    &mut p1_ptr   as *mut _ as *mut c_void,
                    &mut p2_ptr   as *mut _ as *mut c_void,
                    &mut pc_ptr   as *mut _ as *mut c_void,
                    &mut len_i    as *mut _ as *mut c_void,
                    &mut first_i  as *mut _ as *mut c_void,
                    &mut period_i as *mut _ as *mut c_void,
                    &mut mults_ptr as *mut _ as *mut c_void,
                    &mut n_i      as *mut _ as *mut c_void,
                    &mut long_i   as *mut _ as *mut c_void,
                    &mut base_i   as *mut _ as *mut c_void,
                    &mut out_ptr  as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, shmem_bytes, args)
                    .map_err(|e| CudaDevStopError::Cuda(e.to_string()))?;
            }
            out_row_base += n;
        }

        self.stream.synchronize().map_err(|e| CudaDevStopError::Cuda(e.to_string()))?;

        Ok((DeviceArrayF32 { buf: d_out, rows: total_rows, cols: len }, meta))
    }

    pub fn devstop_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
        mult: f32,
        is_long: bool,
    ) -> Result<DeviceArrayF32, CudaDevStopError> {
        if cols == 0 || rows == 0 { return Err(CudaDevStopError::InvalidInput("cols/rows must be > 0".into())); }
        if high_tm.len() != low_tm.len() || high_tm.len() != cols * rows {
            return Err(CudaDevStopError::InvalidInput("time-major arrays must match cols*rows".into()));
        }
        if period == 0 || period > rows { return Err(CudaDevStopError::InvalidInput("invalid period".into())); }

        // Compute first_valid per series
        let mut firsts = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let h = high_tm[t * cols + s];
                let l = low_tm [t * cols + s];
                if !h.is_nan() && !l.is_nan() { fv = Some(t as i32); break; }
            }
            firsts[s] = fv.unwrap_or(0);
        }

        let d_high = DeviceBuffer::from_slice(high_tm).map_err(|e| CudaDevStopError::Cuda(e.to_string()))?;
        let d_low  = DeviceBuffer::from_slice(low_tm ) .map_err(|e| CudaDevStopError::Cuda(e.to_string()))?;
        let d_firsts = DeviceBuffer::from_slice(&firsts).map_err(|e| CudaDevStopError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaDevStopError::Cuda(e.to_string()))?;

        let func = self.module
            .get_function("devstop_many_series_one_param_f32")
            .map_err(|e| CudaDevStopError::Cuda(e.to_string()))?;

        let grid: GridSize = ((cols as u32).max(1), 1, 1).into();
        let block: BlockSize = (128, 1, 1).into();
        let shmem_bytes = (period * (2 * std::mem::size_of::<f32>() + std::mem::size_of::<i32>())) as u32;

        unsafe {
            let mut high_ptr = d_high.as_device_ptr().as_raw();
            let mut low_ptr  = d_low.as_device_ptr().as_raw();
            let mut firsts_ptr = d_firsts.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut period_i = period as i32;
            let mut mult_f = mult as f32;
            let mut is_long_i = if is_long { 1i32 } else { 0i32 };
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr  as *mut _ as *mut c_void,
                &mut firsts_ptr as *mut _ as *mut c_void,
                &mut cols_i    as *mut _ as *mut c_void,
                &mut rows_i    as *mut _ as *mut c_void,
                &mut period_i  as *mut _ as *mut c_void,
                &mut mult_f    as *mut _ as *mut c_void,
                &mut is_long_i as *mut _ as *mut c_void,
                &mut out_ptr   as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, shmem_bytes, args)
                .map_err(|e| CudaDevStopError::Cuda(e.to_string()))?;
        }

        self.stream.synchronize().map_err(|e| CudaDevStopError::Cuda(e.to_string()))?;
        Ok(DeviceArrayF32 { buf: d_out, rows, cols })
    }
}

// ---------- Bench profiles (lightweight) ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 200;

    struct DevStopBatchState {
        cuda: CudaDevStop,
        high: Vec<f32>,
        low: Vec<f32>,
        sweep: DevStopBatchRange,
    }
    impl CudaBenchState for DevStopBatchState { fn launch(&mut self) {
        let _ = self.cuda.devstop_batch_dev(&self.high, &self.low, &self.sweep, true).unwrap();
    }}

    fn synth_hl_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i];
            if v.is_nan() { continue; }
            let x = i as f32 * 0.0021;
            let off = 0.20 + 0.01 * (x.sin().abs());
            high[i] = v + off; low[i] = v - off;
        }
        (high, low)
    }

    fn prep_batch() -> Box<dyn CudaBenchState> {
        let cuda = CudaDevStop::new(0).expect("cuda devstop");
        let close = gen_series(ONE_SERIES_LEN);
        let (high, low) = synth_hl_from_close(&close);
        let sweep = DevStopBatchRange { period: (10, 10 + PARAM_SWEEP - 1, 1), mult: (0.0, 2.0, 0.01), devtype: (0,0,0) };
        Box::new(DevStopBatchState { cuda, high, low, sweep })
    }

    struct DevStopManySeriesState {
        cuda: CudaDevStop,
        high_tm: Vec<f32>,
        low_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        period: usize,
        mult: f32,
    }
    impl CudaBenchState for DevStopManySeriesState { fn launch(&mut self) {
        let _ = self.cuda.devstop_many_series_one_param_time_major_dev(
            &self.high_tm, &self.low_tm, self.cols, self.rows, self.period, self.mult, true
        ).unwrap();
    }}

    fn prep_many_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaDevStop::new(0).expect("cuda devstop");
        let cols = 128usize; let rows = 1_000_000usize / cols;
        let close = gen_series(cols * rows);
        // synthesize time-major high/low from close
        let mut high_tm = close.clone();
        let mut low_tm  = close.clone();
        for s in 0..cols { for t in 0..rows {
            let idx = t * cols + s; let v = close[idx]; if v.is_nan() { continue; }
            let x = (t as f32) * 0.002 + s as f32 * 0.01;
            let off = 0.18 + 0.01 * (x.cos().abs());
            high_tm[idx] = v + off; low_tm[idx] = v - off;
        }}
        Box::new(DevStopManySeriesState { cuda, high_tm, low_tm, cols, rows, period: 20, mult: 1.5 })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new("devstop", "batch_dev", "devstop_cuda_batch_dev", "1m_x_200", prep_batch)
                .with_inner_iters(3),
            CudaBenchScenario::new(
                "devstop",
                "many_series_one_param",
                "devstop_cuda_many_series_one_param_dev",
                "128x8k",
                prep_many_series,
            )
            .with_inner_iters(3),
        ]
    }
}
