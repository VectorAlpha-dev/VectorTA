//! CUDA wrapper for EHMA (Ehlers Hann Moving Average) kernels.
//!
//! Upgraded to ALMA "gold standard" parity:
//! - Policy selection with introspection (batch and many-series)
//! - VRAM checks and BENCH_DEBUG kernel selection logs
//! - Plain batch (on-device weights) and tiled batch 2x (precomputed weights)
//! - Many-series 1D and 2D tiled (time-major)

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::ehma::{expand_grid, EhmaBatchRange, EhmaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer, LockedBuffer};
use cust::memory::AsyncCopyDestination;
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
pub enum CudaEhmaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaEhmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaEhmaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaEhmaError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaEhmaError {}

#[derive(Clone, Copy, Debug)]
pub enum BatchThreadsPerOutput { One, Two }

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
    Tiled { tile: u32, per_thread: BatchThreadsPerOutput },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
    Tiled2D { tx: u32, ty: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaEhmaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaEhmaPolicy {
    fn default() -> Self {
        Self { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Auto }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
    Tiled2x { tile: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
    Tiled2D { tx: u32, ty: u32 },
}

pub struct CudaEhma {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaEhmaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaEhma {
    pub fn new(device_id: usize) -> Result<Self, CudaEhmaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/ehma_kernel.ptx"));
        // Prefer O2 and target-from-context for stability across drivers
        let jit_opts = &[
            cust::module::ModuleJitOption::DetermineTargetFromContext,
            cust::module::ModuleJitOption::OptLevel(cust::module::OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaEhmaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    /// Create with an explicit policy.
    pub fn new_with_policy(device_id: usize, policy: CudaEhmaPolicy) -> Result<Self, CudaEhmaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaEhmaPolicy) { self.policy = policy; }
    pub fn policy(&self) -> &CudaEhmaPolicy { &self.policy }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> { self.last_batch }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> { self.last_many }

    /// Explicit synchronize for deterministic timing in benches
    pub fn synchronize(&self) -> Result<(), CudaEhmaError> {
        self.stream.synchronize().map_err(|e| CudaEhmaError::Cuda(e.to_string()))
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_scenario = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] EHMA batch selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaEhma)).debug_batch_logged = true; }
            }
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per_scenario = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] EHMA many-series selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaEhma)).debug_many_logged = true; }
            }
        }
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }
    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> { mem_get_info().ok() }
    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() { return true; }
        if let Some((free, _)) = Self::device_mem_info() { required_bytes.saturating_add(headroom_bytes) <= free } else { true }
    }

    #[inline]
    fn pick_tiled_block(&self, series_len: usize) -> u32 {
        if let Ok(v) = std::env::var("EHMA_TILE") {
            if let Ok(tile) = v.parse::<u32>() {
                let name = match tile { 128 => Some("ehma_batch_tiled_f32_2x_tile128"), 256 => Some("ehma_batch_tiled_f32_2x_tile256"), 512 => Some("ehma_batch_tiled_f32_2x_tile512"), _ => None };
                if let Some(fname) = name { if self.module.get_function(fname).is_ok() { return tile; } }
            }
        }
        // Default: 256 unless very short series
        if series_len < 8192 {
            if self.module.get_function("ehma_batch_tiled_f32_2x_tile128").is_ok() { return 128; }
        }
        256
    }

    #[inline]
    fn grid_y_chunks(n: usize) -> impl Iterator<Item = (usize, usize)> {
        const MAX_Y: usize = 65_535;
        (0..n).step_by(MAX_Y).map(move |start| (start, (n - start).min(MAX_Y)))
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &EhmaBatchRange,
    ) -> Result<(Vec<EhmaParams>, usize, usize, usize), CudaEhmaError> {
        if data_f32.is_empty() {
            return Err(CudaEhmaError::InvalidInput("empty data".into()));
        }

        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaEhmaError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaEhmaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let len = data_f32.len();
        let mut max_period = 0usize;
        for prm in &combos {
            let period = prm.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaEhmaError::InvalidInput("period must be > 0".into()));
            }
            if period > len {
                return Err(CudaEhmaError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            if len - first_valid < period {
                return Err(CudaEhmaError::InvalidInput(format!(
                    "not enough valid data: needed {}, have {}",
                    period,
                    len - first_valid
                )));
            }
            max_period = max_period.max(period);
        }

        Ok((combos, first_valid, len, max_period))
    }

    fn compute_normalized_weights(period: usize) -> Vec<f32> {
        // Use exact Hann normalization: sum_{i=1..P} (1 - cos(2*pi*i/(P+1))) = P+1
        // Generate weights via 2*sin^2(pi*x) with f64 trig, then normalize by 1/(P+1) exactly.
        let mut weights = vec![0.0f32; period];
        if period == 0 {
            return weights;
        }
        let inv = 1.0f32 / (period as f32 + 1.0f32);
        for idx in 0..period {
            let i = (period - idx) as f64; // map oldest..newest to i=P..1
            let x = i / ((period as f64) + 1.0);
            let s = (std::f64::consts::PI * x).sin();
            let wt = 2.0 * s * s; // 1 - cos = 2*sin^2
            weights[idx] = (wt as f32) * inv;
        }
        weights
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &EhmaParams,
    ) -> Result<(Vec<i32>, usize, Vec<f32>), CudaEhmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaEhmaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaEhmaError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }

        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(CudaEhmaError::InvalidInput("period must be > 0".into()));
        }

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut fv = None;
            for row in 0..rows {
                let idx = row * cols + series;
                let v = data_tm_f32[idx];
                if !v.is_nan() {
                    fv = Some(row);
                    break;
                }
            }
            let fv_row = fv.ok_or_else(|| {
                CudaEhmaError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            if rows - fv_row < period {
                return Err(CudaEhmaError::InvalidInput(format!(
                    "series {} lacks enough valid data: needed {}, have {}",
                    series,
                    period,
                    rows - fv_row
                )));
            }
            first_valids[series] = fv_row as i32;
        }

        let weights = Self::compute_normalized_weights(period);
        Ok((first_valids, period, weights))
    }

    fn launch_batch_kernel_plain(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warms: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhmaError> {
        if series_len == 0 {
            return Err(CudaEhmaError::InvalidInput("series_len is zero".into()));
        }
        if n_combos == 0 {
            return Err(CudaEhmaError::InvalidInput("no parameter combos".into()));
        }
        if max_period == 0 {
            return Err(CudaEhmaError::InvalidInput("max_period is zero".into()));
        }
        if series_len > i32::MAX as usize
            || n_combos > i32::MAX as usize
            || max_period > i32::MAX as usize
        {
            return Err(CudaEhmaError::InvalidInput(
                "series_len, n_combos, or max_period exceed i32::MAX".into(),
            ));
        }

        let func = self.module.get_function("ehma_batch_f32").map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;

        const BLOCK_X: u32 = 256;
        let grid_x = ((series_len as u32) + BLOCK_X - 1) / BLOCK_X;
        let grid: GridSize = (grid_x.max(1), n_combos as u32, 1).into();
        let block: BlockSize = (BLOCK_X, 1, 1).into();
        let shared_bytes = (max_period * std::mem::size_of::<f32>()) as u32;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut warms_ptr = d_warms.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut n_combos_i = n_combos as i32;
            let mut max_period_i = max_period as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut warms_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut n_combos_i as *mut _ as *mut c_void,
                &mut max_period_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes, args)
                .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        }

        // Introspection/log
        unsafe { (*(self as *const _ as *mut CudaEhma)).last_batch = Some(BatchKernelSelected::Plain { block_x: BLOCK_X }); }
        self.maybe_log_batch_debug();
        Ok(())
    }

    fn launch_many_series_kernel_1d(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        period: usize,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhmaError> {
        if period == 0 {
            return Err(CudaEhmaError::InvalidInput("period is zero".into()));
        }
        if num_series == 0 || series_len == 0 {
            return Err(CudaEhmaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if period > i32::MAX as usize
            || num_series > i32::MAX as usize
            || series_len > i32::MAX as usize
        {
            return Err(CudaEhmaError::InvalidInput(
                "period, num_series, or series_len exceed i32::MAX".into(),
            ));
        }

        let func = self.module.get_function("ehma_multi_series_one_param_f32").map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;

        const BLOCK_X: u32 = 256;
        let grid_x = ((series_len as u32) + BLOCK_X - 1) / BLOCK_X;
        let grid: GridSize = (grid_x.max(1), num_series as u32, 1).into();
        let block: BlockSize = (BLOCK_X, 1, 1).into();
        let shared_bytes = (period * std::mem::size_of::<f32>()) as u32;

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut weights_ptr = d_weights.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut weights_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes, args)
                .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        }

        // Introspection/log
        unsafe { (*(self as *const _ as *mut CudaEhma)).last_many = Some(ManySeriesKernelSelected::OneD { block_x: BLOCK_X }); }
        self.maybe_log_many_debug();
        Ok(())
    }

    fn launch_many_series_kernel_2d(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        period: usize,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
        tx: u32,
        ty: u32,
    ) -> Result<(), CudaEhmaError> {
        let fname = match (tx, ty) {
            (128, 4) => "ehma_ms1p_tiled_f32_tx128_ty4",
            (128, 2) => "ehma_ms1p_tiled_f32_tx128_ty2",
            _ => return Err(CudaEhmaError::InvalidInput(format!("unsupported 2D tile tx={}, ty={}", tx, ty))),
        };
        let func = self.module.get_function(fname).map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let grid_x = ((series_len as u32) + tx - 1) / tx;
        let grid_y = ((num_series as u32) + ty - 1) / ty;
        let grid: GridSize = (grid_x.max(1), grid_y.max(1), 1).into();
        let block: BlockSize = (tx, ty, 1).into();
        // Shared layout: align16(period*4) + (TX+period-1)*TY*4
        let period_bytes = period * std::mem::size_of::<f32>();
        let period_aligned = (period_bytes + 15) & !15; // align up to 16
        let tile_elems = (tx as usize + period - 1) * (ty as usize);
        let shared_bytes = (period_aligned + tile_elems * std::mem::size_of::<f32>()) as u32;

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut weights_ptr = d_weights.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut inv_norm = 1.0f32; // ignored in kernels
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut weights_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut inv_norm as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, shared_bytes, args).map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        }
        unsafe { (*(self as *const _ as *mut CudaEhma)).last_many = Some(ManySeriesKernelSelected::Tiled2D { tx, ty }); }
        self.maybe_log_many_debug();
        Ok(())
    }

    pub fn ehma_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warms: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhmaError> {
        self.launch_batch_kernel_plain(
            d_prices, d_periods, d_warms, series_len, n_combos, max_period, d_out,
        )
    }

    pub fn ehma_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &EhmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaEhmaError> {
        let (combos, first_valid, series_len, max_period) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = combos.len();

        // VRAM preflight (prices + weights_flat + out)
        let prices_bytes = series_len * std::mem::size_of::<f32>();
        let weights_bytes = n_combos * max_period * std::mem::size_of::<f32>();
        let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + weights_bytes + out_bytes;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaEhmaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        // Upload prices once
        let d_prices = unsafe { DeviceBuffer::from_slice_async(data_f32, &self.stream) }.map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;

        // Build per-combo weights (pre-normalized) and periods; warms for plain fallback
        let mut periods_i32 = vec![0i32; n_combos];
        let mut warms_i32 = vec![0i32; n_combos];
        let mut weights_flat = vec![0f32; n_combos * max_period];
        let mut inv_norms = vec![1.0f32; n_combos]; // dummy
        for (i, prm) in combos.iter().enumerate() {
            let p = prm.period.unwrap();
            periods_i32[i] = p as i32;
            warms_i32[i] = (first_valid + p - 1) as i32;
            let w = Self::compute_normalized_weights(p);
            let base = i * max_period;
            weights_flat[base..base + p].copy_from_slice(&w);
        }
        let d_periods = DeviceBuffer::from_slice(&periods_i32).map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let d_warms = DeviceBuffer::from_slice(&warms_i32).map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let d_weights = DeviceBuffer::from_slice(&weights_flat).map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let d_inv_norms = DeviceBuffer::from_slice(&inv_norms).map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;

        // Output
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(n_combos * series_len, &self.stream) }.map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;

        // Choose kernel per policy
        let mut use_tiled = series_len > 8192;
        let mut force_tile: Option<u32> = None;
        match self.policy.batch {
            BatchKernelPolicy::Auto => {}
            BatchKernelPolicy::Plain { .. } => use_tiled = false,
            BatchKernelPolicy::Tiled { tile, .. } => { use_tiled = true; force_tile = Some(tile); }
        }

        if use_tiled {
            let tile = force_tile.unwrap_or_else(|| self.pick_tiled_block(series_len));
            let fname = match tile { 128 => "ehma_batch_tiled_f32_2x_tile128", 256 => "ehma_batch_tiled_f32_2x_tile256", 512 => "ehma_batch_tiled_f32_2x_tile512", _ => "ehma_batch_tiled_f32_2x_tile256" };
            if let Ok(func) = self.module.get_function(fname) {
                let grid_x = ((series_len as u32) + tile - 1) / tile;
                let block_x = (tile / 2) as u32; // 2 outputs per thread
                let block: BlockSize = (block_x, 1, 1).into();
                // Chunk over combos in grid.y
                for (start, len) in Self::grid_y_chunks(n_combos) {
                    let grid: GridSize = (grid_x.max(1), len as u32, 1).into();
                    let shared_bytes = ((max_period + (tile as usize) - 1 + max_period) * std::mem::size_of::<f32>()) as u32;
                    unsafe {
                        let out_ptr = d_out.as_device_ptr().add(start * series_len);
                        let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                        let mut wflat_ptr = d_weights.as_device_ptr().as_raw();
                        let mut periods_ptr = d_periods.as_device_ptr().as_raw();
                        let mut inv_ptr = d_inv_norms.as_device_ptr().as_raw();
                        let mut maxp_i = max_period as i32;
                        let mut len_i = series_len as i32;
                        let mut ncomb_i = len as i32;
                        let mut fv_i = first_valid as i32;
                        let mut out_raw = out_ptr.as_raw();
                        let args: &mut [*mut c_void] = &mut [
                            &mut prices_ptr as *mut _ as *mut c_void,
                            &mut wflat_ptr as *mut _ as *mut c_void,
                            &mut periods_ptr as *mut _ as *mut c_void,
                            &mut inv_ptr as *mut _ as *mut c_void,
                            &mut maxp_i as *mut _ as *mut c_void,
                            &mut len_i as *mut _ as *mut c_void,
                            &mut ncomb_i as *mut _ as *mut c_void,
                            &mut fv_i as *mut _ as *mut c_void,
                            &mut out_raw as *mut _ as *mut c_void,
                        ];
                        self.stream.launch(&func, grid, block, shared_bytes, args).map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
                    }
                }
                // Introspection
                unsafe { (*(self as *const _ as *mut CudaEhma)).last_batch = Some(BatchKernelSelected::Tiled2x { tile }); }
                self.maybe_log_batch_debug();
            } else {
                // Fallback to plain
                self.launch_batch_kernel_plain(&d_prices, &d_periods, &d_warms, series_len, n_combos, max_period, &mut d_out)?;
            }
        } else {
            self.launch_batch_kernel_plain(&d_prices, &d_periods, &d_warms, series_len, n_combos, max_period, &mut d_out)?;
        }

        self.stream.synchronize().map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        Ok(DeviceArrayF32 { buf: d_out, rows: n_combos, cols: series_len })
    }

    /// Device input → VRAM output, avoids host price copies. Caller supplies first_valid.
    pub fn ehma_batch_from_device_prices(
        &self,
        d_prices: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        sweep: &EhmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaEhmaError> {
        let (combos, _fv, _len, max_period) = Self::prepare_batch_inputs(&vec![0f32; series_len], sweep)?;
        let n_combos = combos.len();
        let mut periods_i32 = vec![0i32; n_combos];
        let mut warms_i32 = vec![0i32; n_combos];
        let mut weights_flat = vec![0f32; n_combos * max_period];
        let mut inv_norms = vec![1.0f32; n_combos];
        for (i, prm) in combos.iter().enumerate() {
            let p = prm.period.unwrap();
            periods_i32[i] = p as i32;
            warms_i32[i] = (first_valid + p - 1) as i32;
            let w = Self::compute_normalized_weights(p);
            let base = i * max_period;
            weights_flat[base..base + p].copy_from_slice(&w);
        }
        let d_periods = DeviceBuffer::from_slice(&periods_i32).map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let d_warms = DeviceBuffer::from_slice(&warms_i32).map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let d_weights = DeviceBuffer::from_slice(&weights_flat).map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let d_inv_norms = DeviceBuffer::from_slice(&inv_norms).map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(n_combos * series_len, &self.stream) }.map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;

        // Prefer tiled if available
        let tile = self.pick_tiled_block(series_len);
        let fname = match tile { 128 => "ehma_batch_tiled_f32_2x_tile128", 256 => "ehma_batch_tiled_f32_2x_tile256", 512 => "ehma_batch_tiled_f32_2x_tile512", _ => "ehma_batch_tiled_f32_2x_tile256" };
        if let Ok(func) = self.module.get_function(fname) {
            let grid_x = ((series_len as u32) + tile - 1) / tile;
            let block_x = (tile / 2) as u32;
            let block: BlockSize = (block_x, 1, 1).into();
            for (start, len) in Self::grid_y_chunks(n_combos) {
                let grid: GridSize = (grid_x.max(1), len as u32, 1).into();
                let shared_bytes = ((max_period + (tile as usize) - 1 + max_period) * std::mem::size_of::<f32>()) as u32;
                unsafe {
                    let out_ptr = d_out.as_device_ptr().add(start * series_len);
                    let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                    let mut wflat_ptr = d_weights.as_device_ptr().as_raw();
                    let mut periods_ptr = d_periods.as_device_ptr().as_raw();
                    let mut inv_ptr = d_inv_norms.as_device_ptr().as_raw();
                    let mut maxp_i = max_period as i32;
                    let mut len_i = series_len as i32;
                    let mut ncomb_i = len as i32;
                    let mut fv_i = first_valid as i32;
                    let mut out_raw = out_ptr.as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut prices_ptr as *mut _ as *mut c_void,
                        &mut wflat_ptr as *mut _ as *mut c_void,
                        &mut periods_ptr as *mut _ as *mut c_void,
                        &mut inv_ptr as *mut _ as *mut c_void,
                        &mut maxp_i as *mut _ as *mut c_void,
                        &mut len_i as *mut _ as *mut c_void,
                        &mut ncomb_i as *mut _ as *mut c_void,
                        &mut fv_i as *mut _ as *mut c_void,
                        &mut out_raw as *mut _ as *mut c_void,
                    ];
                    self.stream.launch(&func, grid, block, shared_bytes, args).map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
                }
            }
            unsafe { (*(self as *const _ as *mut CudaEhma)).last_batch = Some(BatchKernelSelected::Tiled2x { tile }); }
            self.maybe_log_batch_debug();
        } else {
            self.launch_batch_kernel_plain(d_prices, &d_periods, &d_warms, series_len, n_combos, max_period, &mut d_out)?;
        }
        self.stream.synchronize().map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        Ok(DeviceArrayF32 { buf: d_out, rows: n_combos, cols: series_len })
    }

    pub fn ehma_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &EhmaBatchRange,
        out: &mut [f32],
    ) -> Result<Vec<EhmaParams>, CudaEhmaError> {
        let (combos, first_valid, series_len, max_period) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = combos.len();
        if out.len() != n_combos * series_len {
            return Err(CudaEhmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                n_combos * series_len
            )));
        }

        let mut periods_i32 = Vec::with_capacity(n_combos);
        let mut warms_i32 = Vec::with_capacity(n_combos);
        for prm in &combos {
            let period = prm.period.unwrap();
            if period > i32::MAX as usize {
                return Err(CudaEhmaError::InvalidInput(
                    "period exceeds i32::MAX".into(),
                ));
            }
            let warm = first_valid + period - 1;
            if warm > i32::MAX as usize {
                return Err(CudaEhmaError::InvalidInput(
                    "warm index exceeds i32::MAX".into(),
                ));
            }
            periods_i32.push(period as i32);
            warms_i32.push(warm as i32);
        }

        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)
            .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let d_warms =
            DeviceBuffer::from_slice(&warms_i32).map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;

        self.launch_batch_kernel_plain(
            &d_prices, &d_periods, &d_warms, series_len, n_combos, max_period, &mut d_out,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;

        d_out
            .copy_to(out)
            .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;

        Ok(combos)
    }

    pub fn ehma_multi_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        period: i32,
        num_series: i32,
        series_len: i32,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhmaError> {
        if period <= 0 || num_series <= 0 || series_len <= 0 {
            return Err(CudaEhmaError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        self.launch_many_series_kernel_1d(
            d_prices_tm,
            d_weights,
            period as usize,
            num_series as usize,
            series_len as usize,
            d_first_valids,
            d_out_tm,
        )
    }

    pub fn ehma_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &EhmaParams,
    ) -> Result<DeviceArrayF32, CudaEhmaError> {
        let (first_valids, period, weights) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        // VRAM preflight
        let prices_bytes = cols * rows * std::mem::size_of::<f32>();
        let weights_bytes = period * std::mem::size_of::<f32>();
        let out_bytes = cols * rows * std::mem::size_of::<f32>();
        let required = prices_bytes + weights_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaEhmaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let d_weights =
            DeviceBuffer::from_slice(&weights).map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;

        // Select kernel
        match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => {
                // Prefer 2D tiled for larger problems
                if cols >= 16 && rows >= 8192 && self.module.get_function("ehma_ms1p_tiled_f32_tx128_ty4").is_ok() {
                    self.launch_many_series_kernel_2d(&d_prices_tm, &d_weights, period, cols, rows, &d_first_valids, &mut d_out_tm, 128, 4)?;
                } else if self.module.get_function("ehma_ms1p_tiled_f32_tx128_ty2").is_ok() && (rows >= 8192) {
                    self.launch_many_series_kernel_2d(&d_prices_tm, &d_weights, period, cols, rows, &d_first_valids, &mut d_out_tm, 128, 2)?;
                } else {
                    self.launch_many_series_kernel_1d(&d_prices_tm, &d_weights, period, cols, rows, &d_first_valids, &mut d_out_tm)?;
                }
            }
            ManySeriesKernelPolicy::OneD { .. } => {
                self.launch_many_series_kernel_1d(&d_prices_tm, &d_weights, period, cols, rows, &d_first_valids, &mut d_out_tm)?;
            }
            ManySeriesKernelPolicy::Tiled2D { tx, ty } => {
                self.launch_many_series_kernel_2d(&d_prices_tm, &d_weights, period, cols, rows, &d_first_valids, &mut d_out_tm, tx, ty)?;
            }
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows,
            cols,
        })
    }

    pub fn ehma_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &EhmaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaEhmaError> {
        if out_tm.len() != cols * rows {
            return Err(CudaEhmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out_tm.len(),
                cols * rows
            )));
        }
        let (first_valids, period, weights) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let d_weights =
            DeviceBuffer::from_slice(&weights).map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;

        match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => {
                if cols >= 16 && rows >= 8192 && self.module.get_function("ehma_ms1p_tiled_f32_tx128_ty4").is_ok() {
                    self.launch_many_series_kernel_2d(&d_prices_tm, &d_weights, period, cols, rows, &d_first_valids, &mut d_out_tm, 128, 4)?;
                } else if self.module.get_function("ehma_ms1p_tiled_f32_tx128_ty2").is_ok() && (rows >= 8192) {
                    self.launch_many_series_kernel_2d(&d_prices_tm, &d_weights, period, cols, rows, &d_first_valids, &mut d_out_tm, 128, 2)?;
                } else {
                    self.launch_many_series_kernel_1d(&d_prices_tm, &d_weights, period, cols, rows, &d_first_valids, &mut d_out_tm)?;
                }
            }
            ManySeriesKernelPolicy::OneD { .. } => {
                self.launch_many_series_kernel_1d(&d_prices_tm, &d_weights, period, cols, rows, &d_first_valids, &mut d_out_tm)?;
            }
            ManySeriesKernelPolicy::Tiled2D { tx, ty } => {
                self.launch_many_series_kernel_2d(&d_prices_tm, &d_weights, period, cols, rows, &d_first_valids, &mut d_out_tm, tx, ty)?;
            }
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;

        // Pinned D2H for determinism and throughput
        let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(out_tm.len()) }.map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        unsafe { d_out_tm.async_copy_to(pinned.as_mut_slice(), &self.stream).map_err(|e| CudaEhmaError::Cuda(e.to_string()))?; }
        self.stream.synchronize().map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        out_tm.copy_from_slice(pinned.as_slice());

        Ok(())
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        ehma_benches,
        CudaEhma,
        crate::indicators::moving_averages::ehma::EhmaBatchRange,
        crate::indicators::moving_averages::ehma::EhmaParams,
        ehma_batch_dev,
        ehma_multi_series_one_param_time_major_dev,
        crate::indicators::moving_averages::ehma::EhmaBatchRange { period: (10, 10 + PARAM_SWEEP - 1, 1) },
        crate::indicators::moving_averages::ehma::EhmaParams { period: Some(64) },
        "ehma",
        "ehma"
    );
    pub use ehma_benches::bench_profiles;
}
