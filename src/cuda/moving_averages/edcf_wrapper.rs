#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::edcf::{EdcfBatchRange, EdcfParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, CopyDestination, DeviceBuffer, LockedBuffer};
use cust::memory::AsyncCopyDestination;
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
pub enum CudaEdcfError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaEdcfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaEdcfError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaEdcfError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}

impl std::error::Error for CudaEdcfError {}

/// Policy for the batch (one-series × many-params) path.
#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
    Tiled { tile: u32 },
}
impl Default for BatchKernelPolicy { fn default() -> Self { BatchKernelPolicy::Auto } }

/// Policy for the many-series (time-major) path.
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
    Tiled2D { tx: u32, ty: u32 },
}
impl Default for ManySeriesKernelPolicy { fn default() -> Self { ManySeriesKernelPolicy::Auto } }

/// Global CUDA policy for EDCF wrapper.
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaEdcfPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

/// Introspection of last-selected batch kernel.
#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
    Tiled { tile: u32 },
}

/// Introspection of last-selected many-series kernel.
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
    Tiled2D { tx: u32, ty: u32 },
}

/// CUDA wrapper for EDCF kernels. Mirrors ALMA wrapper layout: includes stream,
/// context, policy selection, VRAM checks, and kernel introspection. All kernel
/// launches synchronize for deterministic timings when measuring.
pub struct CudaEdcf {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaEdcfPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaEdcf {
    pub fn new(device_id: usize) -> Result<Self, CudaEdcfError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;

        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/edcf_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                    m
                } else {
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaEdcfError::Cuda(e.to_string()))?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaEdcfPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(device_id: usize, policy: CudaEdcfPolicy) -> Result<Self, CudaEdcfError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }

    pub fn set_policy(&mut self, policy: CudaEdcfPolicy) { self.policy = policy; }
    pub fn policy(&self) -> &CudaEdcfPolicy { &self.policy }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> { self.last_batch }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> { self.last_many }

    pub fn synchronize(&self) -> Result<(), CudaEdcfError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaEdcfError::Cuda(e.to_string()))
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch { 
                let per_scenario = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] EDCF batch selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaEdcf)).debug_batch_logged = true; }
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
                    eprintln!("[DEBUG] EDCF many-series selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaEdcf)).debug_many_logged = true; }
            }
        }
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }

    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> { mem_get_info().ok() }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() { return true; }
        if let Some((free, _total)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else { true }
    }

    /// One-series × many-params. Host input → VRAM output.
    ///
    /// - Validates data and parameter sweep, computes `first_valid`.
    /// - Allocates device buffers and runs the compute+apply kernels per combo.
    /// - Synchronizes before returning for deterministic timing.
    pub fn edcf_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &EdcfBatchRange,
    ) -> Result<DeviceArrayF32, CudaEdcfError> {
        let (combos, first_valid, series_len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = combos.len();
        // VRAM check: input + output + scratch dist
        let in_bytes = series_len * std::mem::size_of::<f32>();
        let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
        let scratch_bytes = series_len * std::mem::size_of::<f32>();
        let required = in_bytes + out_bytes + scratch_bytes;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaEdcfError::Cuda(format!(
                "insufficient VRAM: need ~{} MiB", (required + (1<<20)-1) >> 20
            )));
        }
        // H2D async copy for prices (pinned path improves throughput)
        let d_prices = unsafe {
            DeviceBuffer::from_slice_async(data_f32, &self.stream)
                .map_err(|e| CudaEdcfError::Cuda(e.to_string()))?
        };
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;
        let mut d_dist: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(series_len) }
            .map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;

        self.edcf_batch_device_impl(
            &d_prices,
            &combos,
            first_valid,
            series_len,
            &mut d_dist,
            &mut d_out,
        )?;
        self.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    /// Convenience: host input → host output (FP32); synchronizes internally.
    pub fn edcf_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &EdcfBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<EdcfParams>), CudaEdcfError> {
        let (combos, first_valid, series_len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos.len() * series_len;
        if out.len() != expected {
            return Err(CudaEdcfError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                expected
            )));
        }
        let d_prices = unsafe {
            DeviceBuffer::from_slice_async(data_f32, &self.stream)
                .map_err(|e| CudaEdcfError::Cuda(e.to_string()))?
        };
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(expected) }
            .map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;
        let mut d_dist: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(series_len) }
            .map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;

        self.edcf_batch_device_impl(
            &d_prices,
            &combos,
            first_valid,
            series_len,
            &mut d_dist,
            &mut d_out,
        )?;
        self.synchronize()?;

        // Copy back via pinned buffer to reduce D2H overhead on large outputs
        let mut pinned: LockedBuffer<f32> = unsafe {
            LockedBuffer::uninitialized(expected).map_err(|e| CudaEdcfError::Cuda(e.to_string()))?
        };
        unsafe {
            d_out.async_copy_to(pinned.as_mut_slice(), &self.stream)
                .map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;
        }
        self.synchronize()?;
        out.copy_from_slice(pinned.as_slice());

        Ok((combos.len(), series_len, combos))
    }

    /// Device-friendly path: run batch using preallocated device buffers.
    ///
    /// - `d_prices`: device-resident prices (length `series_len`).
    /// - `combos`: parameter list (period each). Validates warmup constraints.
    /// - `d_dist`: scratch distance buffer (len ≥ `series_len`).
    /// - `d_out`: output matrix rows×cols = combos×series_len.
    ///
    /// Synchronizes before returning for deterministic timing.
    pub fn edcf_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        combos: &[EdcfParams],
        first_valid: usize,
        series_len: usize,
        d_dist: &mut DeviceBuffer<f32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEdcfError> {
        self.edcf_batch_device_impl(d_prices, combos, first_valid, series_len, d_dist, d_out)?;
        self.synchronize()
    }

    /// One-series × many-params using device-resident prices and a sweep.
    /// Builds `combos` on host, validates sizes, allocates `d_out` and `d_dist`.
    pub fn edcf_batch_from_device_prices(
        &self,
        d_prices: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        sweep: &EdcfBatchRange,
    ) -> Result<DeviceArrayF32, CudaEdcfError> {
        let combos = Self::expand_range(sweep);
        if combos.is_empty() { return Err(CudaEdcfError::InvalidInput("no parameter combinations".into())); }
        for (i, prm) in combos.iter().enumerate() {
            let p = prm.period.unwrap_or(0);
            if p == 0 { return Err(CudaEdcfError::InvalidInput(format!("invalid period at combo {}: 0", i))); }
            let need = first_valid + 2 * p;
            if series_len < need { return Err(CudaEdcfError::InvalidInput(format!(
                "not enough valid data (needed >= {}, series_len = {})", need, series_len))); }
        }
        let n_combos = combos.len();
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
            .map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;
        let mut d_dist: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(series_len) }
            .map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;
        self.edcf_batch_device_impl(d_prices, &combos, first_valid, series_len, &mut d_dist, &mut d_out)?;
        self.synchronize()?;
        Ok(DeviceArrayF32 { buf: d_out, rows: n_combos, cols: series_len })
    }

    fn edcf_batch_device_impl(
        &self,
        d_prices: &DeviceBuffer<f32>,
        combos: &[EdcfParams],
        first_valid: usize,
        series_len: usize,
        d_dist: &mut DeviceBuffer<f32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEdcfError> {
        if combos.is_empty() {
            return Err(CudaEdcfError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        if d_dist.len() < series_len {
            return Err(CudaEdcfError::InvalidInput(format!(
                "dist buffer too small: got {}, need {}",
                d_dist.len(),
                series_len
            )));
        }
        if d_out.len() != combos.len() * series_len {
            return Err(CudaEdcfError::InvalidInput(format!(
                "output buffer wrong length: got {}, expected {}",
                d_out.len(),
                combos.len() * series_len
            )));
        }
        if series_len == 0 {
            return Err(CudaEdcfError::InvalidInput("series_len is zero".into()));
        }
        if series_len > i32::MAX as usize {
            return Err(CudaEdcfError::InvalidInput(
                "series_len exceeds i32::MAX (unsupported)".into(),
            ));
        }

        let compute_fn = self
            .module
            .get_function("edcf_compute_dist_f32")
            .map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;

        for (row_idx, params) in combos.iter().enumerate() {
            let period = params.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaEdcfError::InvalidInput(format!(
                    "invalid period at combo {}: {}",
                    row_idx, period
                )));
            }
            if period > series_len {
                return Err(CudaEdcfError::InvalidInput(format!(
                    "period {} exceeds series length {}",
                    period, series_len
                )));
            }
            let needed = 2 * period;
            if series_len - first_valid < needed {
                return Err(CudaEdcfError::InvalidInput(format!(
                    "not enough valid data (needed >= {}, valid = {})",
                    needed,
                    series_len - first_valid
                )));
            }

            self.launch_compute_dist(
                &compute_fn,
                d_prices,
                series_len,
                period,
                first_valid,
                d_dist,
            )?;

            // Select apply path: tiled vs plain
            let offset_elems = row_idx * series_len;
            let row_ptr = unsafe { d_out.as_device_ptr().add(offset_elems) }.as_raw();

            // Prefer tiled when series is long; allow explicit policy overrides
            let use_tiled = match self.policy.batch {
                BatchKernelPolicy::Tiled { .. } => true,
                BatchKernelPolicy::Plain { .. } => false,
                BatchKernelPolicy::Auto => series_len >= 8192,
            };

            if use_tiled {
                // Choose tile size (128/256/512)
                let tile = match self.policy.batch {
                    BatchKernelPolicy::Tiled { tile } => tile,
                    _ => 256,
                };
                let func_name = match tile { 128 => "edcf_apply_weights_tiled_f32_tile128", 256 => "edcf_apply_weights_tiled_f32_tile256", 512 => "edcf_apply_weights_tiled_f32_tile512", _ => "edcf_apply_weights_tiled_f32_tile256" };
                let func = self
                    .module
                    .get_function(func_name)
                    .map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;

                // Introspection
                unsafe {
                    let this = self as *const _ as *mut CudaEdcf;
                    (*this).last_batch = Some(BatchKernelSelected::Tiled { tile });
                }
                self.maybe_log_batch_debug();

                let outputs_per_block = tile;
                let grid_x = ((series_len as u32) + outputs_per_block - 1) / outputs_per_block;
                let block_x = 128u32;
                let grid: GridSize = (grid_x, 1, 1).into();
                let block: BlockSize = (block_x, 1, 1).into();
                // shared: prices (tile+P-1) + dist (tile+P-1)
                let sh_elems = (tile as usize + period - 1) * 2;
                let shared_bytes = (sh_elems * std::mem::size_of::<f32>()) as u32;

                let stream = &self.stream;
                unsafe {
                    launch!(
                        func<<<grid, block, shared_bytes, stream>>>(
                            d_prices.as_device_ptr(),
                            d_dist.as_device_ptr(),
                            (series_len as i32),
                            (period as i32),
                            (first_valid as i32),
                            row_ptr
                        )
                    )
                    .map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;
                }
            } else {
                let apply_fn = self
                    .module
                    .get_function("edcf_apply_weights_f32")
                    .map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;

                // Introspection
                let block_x = match self.policy.batch { BatchKernelPolicy::Plain { block_x } => block_x, _ => 256 };
                unsafe {
                    let this = self as *const _ as *mut CudaEdcf;
                    (*this).last_batch = Some(BatchKernelSelected::Plain { block_x });
                }
                self.maybe_log_batch_debug();

                self.launch_apply_weights(&apply_fn, d_prices, d_dist, series_len, period, first_valid, row_ptr)?;
            }
        }

        Ok(())
    }

    fn launch_compute_dist(
        &self,
        func: &cust::function::Function,
        d_prices: &DeviceBuffer<f32>,
        len: usize,
        period: usize,
        first_valid: usize,
        d_dist: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEdcfError> {
        let block_x: u32 = match self.policy.batch { BatchKernelPolicy::Plain { block_x } => block_x, _ => 256 };
        let mut grid_x = ((len as u32) + block_x - 1) / block_x;
        if grid_x == 0 {
            grid_x = 1;
        }
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut period_i = period as i32;
            let mut first_i = first_valid as i32;
            let mut dist_ptr = d_dist.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut dist_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(func, grid, block, 0, args)
                .map_err(|e| CudaEdcfError::Cuda(e.to_string()))?
        }
        Ok(())
    }

    fn launch_apply_weights(
        &self,
        func: &cust::function::Function,
        d_prices: &DeviceBuffer<f32>,
        d_dist: &DeviceBuffer<f32>,
        len: usize,
        period: usize,
        first_valid: usize,
        out_row_ptr: u64,
    ) -> Result<(), CudaEdcfError> {
        let block_x: u32 = match self.policy.batch { BatchKernelPolicy::Plain { block_x } => block_x, _ => 256 };
        let mut grid_x = ((len as u32) + block_x - 1) / block_x;
        if grid_x == 0 {
            grid_x = 1;
        }
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut dist_ptr = d_dist.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut period_i = period as i32;
            let mut first_i = first_valid as i32;
            let mut out_ptr = out_row_ptr;
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut dist_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(func, grid, block, 0, args)
                .map_err(|e| CudaEdcfError::Cuda(e.to_string()))?
        }
        Ok(())
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &EdcfBatchRange,
    ) -> Result<(Vec<EdcfParams>, usize, usize), CudaEdcfError> {
        if data_f32.is_empty() {
            return Err(CudaEdcfError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaEdcfError::InvalidInput("all values are NaN".into()))?;
        let series_len = data_f32.len();

        let combos = Self::expand_range(sweep);
        if combos.is_empty() {
            return Err(CudaEdcfError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        for (idx, prm) in combos.iter().enumerate() {
            let period = prm.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaEdcfError::InvalidInput(format!(
                    "invalid period at combo {}: {}",
                    idx, period
                )));
            }
            if period > series_len {
                return Err(CudaEdcfError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, series_len
                )));
            }
            let needed = 2 * period;
            if series_len - first_valid < needed {
                return Err(CudaEdcfError::InvalidInput(format!(
                    "not enough valid data (needed >= {}, valid = {})",
                    needed,
                    series_len - first_valid
                )));
            }
        }

        Ok((combos, first_valid, series_len))
    }

    fn expand_range(sweep: &EdcfBatchRange) -> Vec<EdcfParams> {
        let (start, end, step) = sweep.period;
        if step == 0 || start == end {
            return vec![EdcfParams {
                period: Some(start),
            }];
        }
        let mut periods = Vec::new();
        let mut value = start;
        while value <= end {
            periods.push(EdcfParams {
                period: Some(value),
            });
            value = match value.checked_add(step) {
                Some(v) => v,
                None => break,
            };
        }
        periods
    }

    /// Many-series (time-major) one-parameter path. Host input → VRAM output.
    ///
    /// - Validates per-series warmup using `first_valids` computed here.
    /// - Prefers 2D tiled kernels when rows/cols are large; falls back to 1D.
    /// - Synchronizes before returning for deterministic timing.
    pub fn edcf_many_series_one_param_time_major_dev(
        &mut self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &EdcfParams,
    ) -> Result<DeviceArrayF32, CudaEdcfError> {
        if cols == 0 || rows == 0 { return Err(CudaEdcfError::InvalidInput("empty matrix".into())); }
        let period = params.period.unwrap_or(0);
        if period == 0 { return Err(CudaEdcfError::InvalidInput("period is zero".into())); }
        if prices_tm_f32.len() != cols * rows { return Err(CudaEdcfError::InvalidInput("prices size mismatch".into())); }

        // Compute first_valids per series
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = 0usize; let mut found = false;
            for t in 0..rows { let v = prices_tm_f32[t * cols + s]; if !v.is_nan() { fv = t; found = true; break; } }
            if !found { return Err(CudaEdcfError::InvalidInput(format!("column {} is all NaN", s))); }
            first_valids[s] = fv as i32;
            let warm = fv + 2 * period;
            if rows < warm { return Err(CudaEdcfError::InvalidInput(format!("not enough valid data in series {}: need >= {}, have {}", s, warm, rows - fv))); }
        }

        // VRAM check
        let bytes = prices_tm_f32.len() * 4 + prices_tm_f32.len() * 4 + cols * 4;
        if !Self::will_fit(bytes, 64 * 1024 * 1024) {
            return Err(CudaEdcfError::Cuda("insufficient VRAM for many-series".into()));
        }

        let d_prices_tm = unsafe { DeviceBuffer::from_slice_async(prices_tm_f32, &self.stream) }
            .map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }.map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;

        // Kernel selection
        let try_2d = |tx: u32, ty: u32| -> Result<bool, CudaEdcfError> {
            let fname = match (tx, ty) { (128, 4) => "edcf_ms1p_tiled_f32_tx128_ty4", (128, 2) => "edcf_ms1p_tiled_f32_tx128_ty2", _ => return Ok(false) };
            let func = match self.module.get_function(fname) { Ok(f) => f, Err(_) => return Ok(false) };
            // Shared size per block
            let prices_elems = (tx as usize) + 2 * (period - 1);
            let dist_elems   = (tx as usize) + (period - 1);
            let per_series = (prices_elems + dist_elems) * std::mem::size_of::<f32>();
            let shared_bytes = (per_series * (ty as usize)) as u32;
            let grid_x = ((rows as u32) + tx - 1) / tx;
            let grid_y = ((cols as u32) + ty - 1) / ty;
            let grid: GridSize = (grid_x, grid_y, 1).into();
            let block: BlockSize = (128, ty, 1).into();
            let stream = &self.stream;
            unsafe {
                launch!(
                    func<<<grid, block, shared_bytes, stream>>>(
                        d_prices_tm.as_device_ptr(),
                        d_first_valids.as_device_ptr(),
                        (period as i32),
                        (cols as i32),
                        (rows as i32),
                        d_out_tm.as_device_ptr()
                    )
                )
                .map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;
            }
            unsafe { let this = self as *const _ as *mut CudaEdcf; (*this).last_many = Some(ManySeriesKernelSelected::Tiled2D { tx, ty }); }
            self.maybe_log_many_debug();
            Ok(true)
        };

        let launched = match self.policy.many_series {
            ManySeriesKernelPolicy::Tiled2D { tx, ty } => try_2d(tx, ty)?,
            ManySeriesKernelPolicy::Auto => {
                // Prefer 2D when rows and cols are reasonably large
                if cols >= 16 && rows >= 2048 { if try_2d(128, 4)? { true } else { try_2d(128, 2)? } } else { false }
            }
            ManySeriesKernelPolicy::OneD { .. } => false,
        };

        if !launched {
            // 1D fallback
            let func = self.module.get_function("edcf_many_series_one_param_f32").map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;
            let block_x = match self.policy.many_series { ManySeriesKernelPolicy::OneD { block_x } => block_x, _ => 128 };
            let grid: GridSize = (cols as u32, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            let shared_bytes = (2 * period * std::mem::size_of::<f32>()) as u32;
            let stream = &self.stream;
            unsafe {
                launch!(
                    func<<<grid, block, shared_bytes, stream>>>(
                        d_prices_tm.as_device_ptr(),
                        d_first_valids.as_device_ptr(),
                        (period as i32),
                        (cols as i32),
                        (rows as i32),
                        d_out_tm.as_device_ptr()
                    )
                )
                .map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;
            }
            unsafe { let this = self as *const _ as *mut CudaEdcf; (*this).last_many = Some(ManySeriesKernelSelected::OneD { block_x }); }
            self.maybe_log_many_debug();
        }

        self.synchronize()?;
        Ok(DeviceArrayF32 { buf: d_out_tm, rows, cols })
    }

    /// Many-series (time-major) device-resident path with precomputed `first_valids`.
    /// Launches the same kernels as `_dev`, but accepts device inputs and output.
    pub fn edcf_many_series_one_param_time_major_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEdcfError> {
        if cols == 0 || rows == 0 || period == 0 { return Err(CudaEdcfError::InvalidInput("invalid dims/period".into())); }
        // Try 2D tiled first (ty=4 then ty=2)
        let try_2d = |tx: u32, ty: u32| -> Result<bool, CudaEdcfError> {
            let fname = match (tx, ty) { (128, 4) => "edcf_ms1p_tiled_f32_tx128_ty4", (128, 2) => "edcf_ms1p_tiled_f32_tx128_ty2", _ => return Ok(false) };
            let func = match self.module.get_function(fname) { Ok(f) => f, Err(_) => return Ok(false) };
            let prices_elems = (tx as usize) + 2 * (period - 1);
            let dist_elems   = (tx as usize) + (period - 1);
            let per_series = (prices_elems + dist_elems) * std::mem::size_of::<f32>();
            let shared_bytes = (per_series * (ty as usize)) as u32;
            let grid_x = ((rows as u32) + tx - 1) / tx; let grid_y = ((cols as u32) + ty - 1) / ty;
            let grid: GridSize = (grid_x, grid_y, 1).into(); let block: BlockSize = (128, ty, 1).into();
            let stream = &self.stream;
            unsafe {
                launch!(func<<<grid, block, shared_bytes, stream>>>(
                    d_prices_tm.as_device_ptr(), d_first_valids.as_device_ptr(), (period as i32), (cols as i32), (rows as i32), d_out_tm.as_device_ptr()
                ))
                .map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;
            }
            Ok(true)
        };

        let launched = if cols >= 16 && rows >= 2048 { if try_2d(128, 4)? { true } else { try_2d(128, 2)? } } else { false };
        if !launched {
            let func = self.module.get_function("edcf_many_series_one_param_f32").map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;
            let grid: GridSize = (cols as u32, 1, 1).into(); let block: BlockSize = (128, 1, 1).into();
            let shared_bytes = (2 * period * std::mem::size_of::<f32>()) as u32;
            let stream = &self.stream;
            unsafe {
                launch!(func<<<grid, block, shared_bytes, stream>>>(
                    d_prices_tm.as_device_ptr(), d_first_valids.as_device_ptr(), (period as i32), (cols as i32), (rows as i32), d_out_tm.as_device_ptr()
                ))
                .map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;
            }
        }
        self.synchronize()
    }
}

// ---------- Benches ----------

pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;
    const MANY_SERIES_COLS: usize = 250;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * 4;
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * 4;
        let scratch = ONE_SERIES_LEN * 4;
        in_bytes + out_bytes + scratch + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = elems * 4;
        let out_bytes = elems * 4;
        let aux = MANY_SERIES_COLS * 4;
        in_bytes + out_bytes + aux + 64 * 1024 * 1024
    }

    // Preallocated, device-resident batch state (avoids per-iter allocs and H2D copies)
    struct EdcfBatchDeviceState {
        cuda: CudaEdcf,
        d_prices: DeviceBuffer<f32>,
        d_out: DeviceBuffer<f32>,
        d_dist: DeviceBuffer<f32>,
        combos: Vec<EdcfParams>,
        first_valid: usize,
        series_len: usize,
        warmed: bool,
    }
    impl CudaBenchState for EdcfBatchDeviceState {
        fn launch(&mut self) {
            self.cuda
                .edcf_batch_device(&self.d_prices, &self.combos, self.first_valid, self.series_len, &mut self.d_dist, &mut self.d_out)
                .expect("edcf_batch_device");
            self.cuda.synchronize().expect("sync");
            if !self.warmed { self.warmed = true; }
        }
    }
    fn prep_edcf_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaEdcf::new(0).expect("cuda edcf");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = EdcfBatchRange { period: (8, 8 + PARAM_SWEEP - 1, 1) };
        // Compute first_valid
        let first_valid = price.iter().position(|&x| !x.is_nan()).unwrap_or(0);
        let series_len = price.len();
        let combos: Vec<EdcfParams> = (sweep.period.0..=sweep.period.1).step_by(sweep.period.2.max(1)).map(|p| EdcfParams { period: Some(p) }).collect();
        let d_prices = DeviceBuffer::from_slice(&price).expect("d_prices");
        let d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(combos.len() * series_len) }.expect("d_out");
        let d_dist: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(series_len) }.expect("d_dist");
        Box::new(EdcfBatchDeviceState { cuda, d_prices, d_out, d_dist, combos, first_valid, series_len, warmed: false })
    }

    struct EdcfManyDeviceState {
        cuda: CudaEdcf,
        d_prices_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        d_out_tm: DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        period: usize,
        warmed: bool,
    }
    impl CudaBenchState for EdcfManyDeviceState {
        fn launch(&mut self) {
            self.cuda
                .edcf_many_series_one_param_time_major_device(&self.d_prices_tm, &self.d_first_valids, self.cols, self.rows, self.period, &mut self.d_out_tm)
                .expect("edcf many device");
            self.cuda.synchronize().expect("sync");
            if !self.warmed { self.warmed = true; }
        }
    }
    fn prep_edcf_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaEdcf::new(0).expect("cuda edcf");
        let cols = MANY_SERIES_COLS; let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let period = 64usize;
        // Compute first_valids once on host
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = 0usize; for t in 0..rows { if !data_tm[t * cols + s].is_nan() { fv = t; break; } }
            first_valids[s] = fv as i32;
        }
        let d_prices_tm = DeviceBuffer::from_slice(&data_tm).expect("d_prices_tm");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out_tm");
        Box::new(EdcfManyDeviceState { cuda, d_prices_tm, d_first_valids, d_out_tm, cols, rows, period, warmed: false })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new("edcf", "one_series_many_params", "edcf_cuda_batch_dev", "1m_x_250", prep_edcf_one_series_many_params)
                .with_sample_size(10)
                .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new("edcf", "many_series_one_param", "edcf_cuda_many_series_one_param", "250x1m", prep_edcf_many_series_one_param)
                .with_sample_size(6)
                .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
