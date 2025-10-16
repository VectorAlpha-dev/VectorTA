//! CUDA support for Parabolic SAR (Stop and Reverse).
//!
//! Parity with ALMA/CWMA wrappers:
//! - PTX load via include_str!(.../sar_kernel.ptx) with DetermineTargetFromContext + O2
//! - NON_BLOCKING stream
//! - Policy enums and simple selection
//! - VRAM checks with ~64MB headroom and grid.y chunking (<= 65_535)
//! - Public device entry points:
//!     - one-series × many-params (batch)
//!     - many-series × one-param (time‑major)
//! - Numerics & warmup/NaN semantics match src/indicators/sar.rs

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::sar::{SarBatchRange, SarParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaSarError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaSarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaSarError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaSarError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaSarError {}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
}
impl Default for BatchKernelPolicy { fn default() -> Self { BatchKernelPolicy::Auto } }

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32, block_y: u32 },
}
impl Default for ManySeriesKernelPolicy { fn default() -> Self { ManySeriesKernelPolicy::Auto } }

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaSarPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

pub struct CudaSar {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaSarPolicy,
    debug_logged: std::sync::atomic::AtomicBool,
}

impl CudaSar {
    pub fn new(device_id: usize) -> Result<Self, CudaSarError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaSarError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaSarError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaSarError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/sar_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => match Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                Ok(m) => m,
                Err(_) => Module::from_ptx(ptx, &[]).map_err(|e| CudaSarError::Cuda(e.to_string()))?,
            },
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaSarError::Cuda(e.to_string()))?;

        Ok(Self { module, stream, _context: context, policy: CudaSarPolicy::default(), debug_logged: std::sync::atomic::AtomicBool::new(false) })
    }

    pub fn set_policy(&mut self, p: CudaSarPolicy) { self.policy = p; }

    #[inline] fn headroom_bytes() -> usize {
        env::var("CUDA_MEM_HEADROOM").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(64 * 1024 * 1024)
    }
    #[inline] fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") { Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"), Err(_) => true }
    }
    #[inline] fn will_fit(bytes: usize, headroom: usize) -> bool {
        if !Self::mem_check_enabled() { return true; }
        if let Ok((free, _)) = mem_get_info() { bytes.saturating_add(headroom) <= free } else { true }
    }

    // ---------------- One-series × many-params (batch) ----------------
    pub fn sar_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
        sweep: &SarBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<SarParams>), CudaSarError> {
        if high.is_empty() || low.is_empty() || high.len() != low.len() {
            return Err(CudaSarError::InvalidInput("inputs must be non-empty and same length".into()));
        }
        let len = high.len();
        let first = first_valid_hl(high, low).ok_or_else(|| CudaSarError::InvalidInput("all values are NaN".into()))?;
        if len - first < 2 {
            return Err(CudaSarError::InvalidInput("not enough valid data (need >= 2 after first)".into()));
        }

        let combos = expand_grid(sweep);
        if combos.is_empty() { return Err(CudaSarError::InvalidInput("no parameter combinations".into())); }

        // Validate params & build f32 arrays
        let mut accs = Vec::with_capacity(combos.len());
        let mut maxs = Vec::with_capacity(combos.len());
        for p in &combos {
            let a = p.acceleration.unwrap_or(0.02);
            let m = p.maximum.unwrap_or(0.2);
            if !(a > 0.0) || !(m > 0.0) {
                return Err(CudaSarError::InvalidInput("invalid acceleration/maximum".into()));
            }
            accs.push(a as f32);
            maxs.push(m as f32);
        }

        // VRAM: inputs + params + output
        let in_bytes = 2 * len * std::mem::size_of::<f32>();
        let param_bytes = 2 * combos.len() * std::mem::size_of::<f32>();
        let out_bytes = combos.len() * len * std::mem::size_of::<f32>();
        let required = in_bytes + param_bytes + out_bytes;
        if !Self::will_fit(required, Self::headroom_bytes()) {
            return Err(CudaSarError::InvalidInput("insufficient device memory for sar batch".into()));
        }

        // H2D
        let d_high = DeviceBuffer::from_slice(high).map_err(|e| CudaSarError::Cuda(e.to_string()))?;
        let d_low = DeviceBuffer::from_slice(low).map_err(|e| CudaSarError::Cuda(e.to_string()))?;
        let d_accs = DeviceBuffer::from_slice(&accs).map_err(|e| CudaSarError::Cuda(e.to_string()))?;
        let d_maxs = DeviceBuffer::from_slice(&maxs).map_err(|e| CudaSarError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len * combos.len()) }
            .map_err(|e| CudaSarError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(&d_high, &d_low, len as i32, first as i32, &d_accs, &d_maxs, combos.len() as i32, &mut d_out)?;

        Ok((DeviceArrayF32 { buf: d_out, rows: combos.len(), cols: len }, combos))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        len: i32,
        first_valid: i32,
        d_accs: &DeviceBuffer<f32>,
        d_maxs: &DeviceBuffer<f32>,
        n_rows: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSarError> {
        if len <= 0 || n_rows <= 0 { return Ok(()); }
        let func = self.module.get_function("sar_batch_f32").map_err(|e| CudaSarError::Cuda(e.to_string()))?;
        let block_x = match self.policy.batch { BatchKernelPolicy::Plain { block_x } if block_x > 0 => block_x, _ => 256 };

        // y-chunk to grid.y <= 65_535
        let mut launched = 0usize;
        const MAX_GRID_Y: usize = 65_535;
        while launched < (n_rows as usize) {
            let count = ((n_rows as usize) - launched).min(MAX_GRID_Y);
            let grid: GridSize = (1u32, count as u32, 1u32).into();
            let block: BlockSize = (block_x, 1u32, 1u32).into();
            unsafe {
                let mut p_high = d_high.as_device_ptr().as_raw();
                let mut p_low = d_low.as_device_ptr().as_raw();
                let mut p_len = len;
                let mut p_first = first_valid;
                let mut p_accs = d_accs.as_device_ptr().add(launched).as_raw();
                let mut p_maxs = d_maxs.as_device_ptr().add(launched).as_raw();
                let mut p_n = count as i32;
                let mut p_out = d_out.as_device_ptr().add(launched * (len as usize)).as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut p_high as *mut _ as *mut c_void,
                    &mut p_low as *mut _ as *mut c_void,
                    &mut p_len as *mut _ as *mut c_void,
                    &mut p_first as *mut _ as *mut c_void,
                    &mut p_accs as *mut _ as *mut c_void,
                    &mut p_maxs as *mut _ as *mut c_void,
                    &mut p_n as *mut _ as *mut c_void,
                    &mut p_out as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, 0, args).map_err(|e| CudaSarError::Cuda(e.to_string()))?;
            }
            launched += count;
        }
        Ok(())
    }

    // ---------------- Many-series × one-param (time‑major) ----------------
    pub fn sar_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &SarParams,
    ) -> Result<DeviceArrayF32, CudaSarError> {
        if cols == 0 || rows == 0 { return Err(CudaSarError::InvalidInput("empty matrix".into())); }
        let elems = cols.checked_mul(rows).ok_or_else(|| CudaSarError::InvalidInput("size overflow".into()))?;
        if high_tm.len() != elems || low_tm.len() != elems {
            return Err(CudaSarError::InvalidInput("inputs must be time‑major and equal size".into()));
        }
        let acceleration = params.acceleration.unwrap_or(0.02);
        let maximum = params.maximum.unwrap_or(0.2);
        if !(acceleration > 0.0) || !(maximum > 0.0) {
            return Err(CudaSarError::InvalidInput("invalid acceleration/maximum".into()));
        }

        let first_valids = first_valids_time_major(high_tm, low_tm, cols, rows);

        // VRAM: inputs + first_valids + output
        let in_bytes = 2 * elems * std::mem::size_of::<f32>();
        let fv_bytes = cols * std::mem::size_of::<i32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        if !Self::will_fit(in_bytes + fv_bytes + out_bytes, Self::headroom_bytes()) {
            return Err(CudaSarError::InvalidInput("insufficient device memory for sar many-series".into()));
        }

        // H2D
        let d_high = DeviceBuffer::from_slice(high_tm).map_err(|e| CudaSarError::Cuda(e.to_string()))?;
        let d_low = DeviceBuffer::from_slice(low_tm).map_err(|e| CudaSarError::Cuda(e.to_string()))?;
        let d_fv = DeviceBuffer::from_slice(&first_valids).map_err(|e| CudaSarError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(|e| CudaSarError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_high,
            &d_low,
            &d_fv,
            cols as i32,
            rows as i32,
            acceleration as f32,
            maximum as f32,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 { buf: d_out, rows, cols })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_kernel(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_fv: &DeviceBuffer<i32>,
        cols: i32,
        rows: i32,
        acceleration: f32,
        maximum: f32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSarError> {
        if cols <= 0 || rows <= 0 { return Ok(()); }
        let func = self.module.get_function("sar_many_series_one_param_time_major_f32").map_err(|e| CudaSarError::Cuda(e.to_string()))?;
        let (block_x, block_y) = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x, block_y } if block_x > 0 && block_y > 0 => (block_x, block_y),
            _ => (128, 4),
        };
        let grid_y = ((cols as u32) + block_y - 1) / block_y;
        let grid: GridSize = (1u32, grid_y.max(1), 1u32).into();
        let block: BlockSize = (block_x, block_y, 1u32).into();
        unsafe {
            let mut p_high = d_high.as_device_ptr().as_raw();
            let mut p_low = d_low.as_device_ptr().as_raw();
            let mut p_fv = d_fv.as_device_ptr().as_raw();
            let mut p_cols = cols;
            let mut p_rows = rows;
            let mut p_acc = acceleration;
            let mut p_max = maximum;
            let mut p_out = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_high as *mut _ as *mut c_void,
                &mut p_low as *mut _ as *mut c_void,
                &mut p_fv as *mut _ as *mut c_void,
                &mut p_cols as *mut _ as *mut c_void,
                &mut p_rows as *mut _ as *mut c_void,
                &mut p_acc as *mut _ as *mut c_void,
                &mut p_max as *mut _ as *mut c_void,
                &mut p_out as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args).map_err(|e| CudaSarError::Cuda(e.to_string()))?;
        }
        Ok(())
    }
}

// -------- Helpers --------
fn first_valid_hl(high: &[f32], low: &[f32]) -> Option<usize> {
    high.iter().zip(low.iter()).position(|(&h, &l)| h.is_finite() && l.is_finite())
}

fn first_valids_time_major(high_tm: &[f32], low_tm: &[f32], cols: usize, rows: usize) -> Vec<i32> {
    let mut out = vec![-1i32; cols];
    for s in 0..cols {
        for t in s..rows { // mirror staggered warmup used in helpers
            let idx = t * cols + s;
            let h = high_tm[idx];
            let l = low_tm[idx];
            if h == h && l == l { out[s] = t as i32; break; }
        }
    }
    out
}

fn axis_f64(axis: (f64, f64, f64)) -> Vec<f64> {
    let (start, end, step) = axis;
    if step.abs() < 1e-12 || (start - end).abs() < 1e-12 { return vec![start]; }
    if start > end { return Vec::new(); }
    let mut v = Vec::new();
    let mut x = start;
    let lim = end + step.abs() * 1e-12;
    while x <= lim { v.push(x); x += step; }
    v
}

fn expand_grid(r: &SarBatchRange) -> Vec<SarParams> {
    let accs = axis_f64(r.acceleration);
    let maxs = axis_f64(r.maximum);
    let mut out = Vec::with_capacity(accs.len() * maxs.len());
    for &a in &accs {
        for &m in &maxs {
            out.push(SarParams { acceleration: Some(a), maximum: Some(m) });
        }
    }
    out
}

// ---------------- Benches ----------------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const MANY_SERIES_COLS: usize = 256;
    const MANY_SERIES_ROWS: usize = 1_000_000;

    fn bytes_one_series_many_params(n_rows: usize) -> usize {
        let in_bytes = 2 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let param_bytes = 2 * n_rows * std::mem::size_of::<f32>();
        let out_bytes = n_rows * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        in_bytes + param_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_ROWS;
        let in_bytes = 2 * elems * std::mem::size_of::<f32>();
        let fv_bytes = MANY_SERIES_COLS * std::mem::size_of::<i32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        in_bytes + fv_bytes + out_bytes + 64 * 1024 * 1024
    }

    fn synth_high_low_from_price(price: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = price.to_vec();
        let mut low = price.to_vec();
        for i in 0..price.len() {
            let p = price[i];
            if !p.is_finite() { continue; }
            let x = i as f32 * 0.0021;
            let off = (0.0087 * x.cos()).abs() + 0.1;
            high[i] = p + off;
            low[i] = p - off;
        }
        (high, low)
    }

    struct SarBatchState { cuda: CudaSar, high: Vec<f32>, low: Vec<f32>, sweep: SarBatchRange }
    impl CudaBenchState for SarBatchState { fn launch(&mut self) { let _ = self.cuda.sar_batch_dev(&self.high, &self.low, &self.sweep).expect("sar batch"); } }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaSar::new(0).expect("cuda sar");
        let price = gen_series(ONE_SERIES_LEN);
        let (high, low) = synth_high_low_from_price(&price);
        let sweep = SarBatchRange { acceleration: (0.01, 0.1, 0.01), maximum: (0.1, 0.3, 0.05) };
        Box::new(SarBatchState { cuda, high, low, sweep })
    }

    struct SarManyState { cuda: CudaSar, high_tm: Vec<f32>, low_tm: Vec<f32>, cols: usize, rows: usize, params: SarParams }
    impl CudaBenchState for SarManyState { fn launch(&mut self) { let _ = self.cuda.sar_many_series_one_param_time_major_dev(&self.high_tm, &self.low_tm, self.cols, self.rows, &self.params).expect("sar many-series"); } }
    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaSar::new(0).expect("cuda sar");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_ROWS;
        // build time‑major from single-series generator with per-series phase offset
        let mut high_tm = vec![f32::NAN; cols * rows];
        let mut low_tm = vec![f32::NAN; cols * rows];
        for s in 0..cols {
            let price = gen_series(rows);
            let (h, l) = synth_high_low_from_price(&price);
            for t in s..rows {
                let idx = t * cols + s;
                high_tm[idx] = h[t];
                low_tm[idx] = l[t];
            }
        }
        let params = SarParams { acceleration: Some(0.02), maximum: Some(0.2) };
        Box::new(SarManyState { cuda, high_tm, low_tm, cols, rows, params })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        let combos = (((0.1 - 0.01) / 0.01) as usize + 1) * (((0.3 - 0.1) / 0.05) as usize + 1);
        vec![
            CudaBenchScenario::new(
                "sar",
                "one_series_many_params",
                "sar_batch",
                "sar_batch/rowsxcols",
                prep_one_series_many_params,
            )
            .with_mem_required(bytes_one_series_many_params(combos)),
            CudaBenchScenario::new(
                "sar",
                "many_series_one_param",
                "sar_many_series",
                "sar_many/colsxrows",
                prep_many_series_one_param,
            )
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}

