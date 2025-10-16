//! CUDA scaffolding for the QStick indicator (average of close-open).
//!
//! Parity goals with ALMA wrapper:
//! - Policy enums for batch and many-series selection with light introspection
//! - PTX load via OUT_DIR, JIT DetermineTargetFromContext + OptLevel O2, fallbacks
//! - Stream NON_BLOCKING
//! - VRAM checks + chunking grid.y to <= 65_535 for batch
//! - Host precompute for prefix-sum category; kernels consume prefixes
//
#![cfg(feature = "cuda")]

use crate::indicators::qstick::{QstickBatchRange, QstickParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;

// Reuse the common VRAM handle from ALMA
use crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32;

#[derive(Debug)]
pub enum CudaQstickError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaQstickError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaQstickError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaQstickError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl std::error::Error for CudaQstickError {}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
    Tiled { tile: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
    // Tiled2D could be added later if needed
}

#[derive(Clone, Copy, Debug)]
pub struct CudaQstickPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaQstickPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
    Tiled1x { tile: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaQstick {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaQstickPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaQstick {
    pub fn new(device_id: usize) -> Result<Self, CudaQstickError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaQstickError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaQstickError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaQstickError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/qstick_kernel.ptx"));
        // Follow ALMA JIT policy: determine target from context + O2, fallback
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(
                match env::var("QS_JIT_OPT").ok().as_deref() {
                    Some("O0") => OptLevel::O0,
                    Some("O1") => OptLevel::O1,
                    Some("O3") => OptLevel::O3,
                    Some("O4") => OptLevel::O4,
                    _ => OptLevel::O2,
                }
            ),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
                {
                    m
                } else {
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaQstickError::Cuda(e.to_string()))?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaQstickError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaQstickPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn set_policy(&mut self, p: CudaQstickPolicy) { self.policy = p; }
    pub fn policy(&self) -> &CudaQstickPolicy { &self.policy }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> { self.last_batch }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> { self.last_many }
    pub fn synchronize(&self) -> Result<(), CudaQstickError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaQstickError::Cuda(e.to_string()))
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        if self.debug_batch_logged { return; }
        if env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                eprintln!("[DEBUG] QStick batch selected kernel: {:?}", sel);
                unsafe { (*(self as *const _ as *mut CudaQstick)).debug_batch_logged = true; }
            }
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        if self.debug_many_logged { return; }
        if env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                eprintln!("[DEBUG] QStick many-series selected kernel: {:?}", sel);
                unsafe { (*(self as *const _ as *mut CudaQstick)).debug_many_logged = true; }
            }
        }
    }

    // ------------------- Utilities -------------------
    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }
    #[inline]
    fn will_fit(required_bytes: usize, headroom: usize) -> bool {
        if !Self::mem_check_enabled() { return true; }
        if let Ok((free, _total)) = mem_get_info() { required_bytes.saturating_add(headroom) <= free } else { true }
    }
    #[inline]
    fn pick_tiled_block(&self, len: usize) -> u32 {
        // Prefer 256 for long series; allow override via QS_TILE
        if let Ok(v) = env::var("QS_TILE") { if let Ok(b) = v.parse::<u32>() { return b; } }
        if len < 8192 { 128 } else { 256 }
    }

    // ------------------- Host precompute -------------------
    pub fn build_diff_prefix_f32(open: &[f32], close: &[f32]) -> (Vec<f32>, usize, usize) {
        let len = open.len().min(close.len());
        // first valid: both non-NaN
        let first = (0..len).find(|&i| !open[i].is_nan() && !close[i].is_nan()).unwrap_or(0);
        let mut prefix = vec![0.0f32; len + 1];
        let mut acc = 0.0f64;
        for i in 0..len {
            if i < first { prefix[i + 1] = acc as f32; continue; }
            let d = (close[i] as f64) - (open[i] as f64);
            acc += d;
            prefix[i + 1] = acc as f32;
        }
        (prefix, first, len)
    }

    fn prepare_batch_inputs(
        open: &[f32],
        close: &[f32],
        sweep: &QstickBatchRange,
    ) -> Result<(Vec<QstickParams>, usize, usize), CudaQstickError> {
        if open.is_empty() || close.is_empty() {
            return Err(CudaQstickError::InvalidInput("empty inputs".into()));
        }
        let len = open.len().min(close.len());
        let first = (0..len)
            .find(|&i| !open[i].is_nan() && !close[i].is_nan())
            .ok_or_else(|| CudaQstickError::InvalidInput("all values are NaN".into()))?;

        // Expand grid
        let (start, end, step) = sweep.period;
        let combos: Vec<QstickParams> = if step == 0 || start == end {
            vec![QstickParams { period: Some(start) }]
        } else {
            (start..=end).step_by(step).map(|p| QstickParams { period: Some(p) }).collect()
        };
        if combos.is_empty() {
            return Err(CudaQstickError::InvalidInput("no parameter combinations".into()));
        }
        for c in &combos {
            let p = c.period.unwrap_or(0);
            if p == 0 || p > len { return Err(CudaQstickError::InvalidInput(format!("invalid period {}", p))); }
            if len - first < p {
                return Err(CudaQstickError::InvalidInput(format!(
                    "not enough valid data: need {}, have {} after first_valid {}",
                    p, len - first, first
                )));
            }
        }
        Ok((combos, first, len))
    }

    fn launch_batch_kernel(
        &self,
        d_prefix: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaQstickError> {
        // Decide kernel
        let mut use_tiled = len > 8192;
        let mut block_x: u32 = 256;
        let mut tile_choice: Option<u32> = None;
        match self.policy.batch {
            BatchKernelPolicy::Auto => {}
            BatchKernelPolicy::Plain { block_x: bx } => { use_tiled = false; block_x = bx; }
            BatchKernelPolicy::Tiled { tile } => { use_tiled = true; tile_choice = Some(tile); }
        }

        const MAX_GRID_Y: usize = 65_535;
        if use_tiled {
            let tile = tile_choice.unwrap_or_else(|| self.pick_tiled_block(len));
            let func_name = match tile { 128 => "qstick_batch_prefix_tiled_f32_tile128", _ => "qstick_batch_prefix_tiled_f32_tile256" };
            let func = self.module.get_function(func_name)
                .or_else(|_| self.module.get_function("qstick_batch_prefix_f32"))
                .map_err(|e| CudaQstickError::Cuda(e.to_string()))?;
            unsafe { (*(self as *const _ as *mut CudaQstick)).last_batch = Some(BatchKernelSelected::Tiled1x { tile }); }
            self.maybe_log_batch_debug();

            let grid_x = ((len as u32) + tile - 1) / tile;
            let block: BlockSize = (tile, 1, 1).into();
            let mut start = 0usize;
            while start < n_combos {
                let chunk = (n_combos - start).min(MAX_GRID_Y);
                let grid: GridSize = (grid_x.max(1), chunk as u32, 1).into();
                unsafe {
                    let mut p_ptr = d_prefix.as_device_ptr().as_raw();
                    let mut len_i = len as i32;
                    let mut first_i = first_valid as i32;
                    let mut per_ptr = d_periods.as_device_ptr().as_raw().wrapping_add((start * std::mem::size_of::<i32>()) as u64);
                    let mut n_i = chunk as i32;
                    let mut out_ptr = d_out.as_device_ptr().as_raw().wrapping_add((start * len * std::mem::size_of::<f32>()) as u64);
                    let args: &mut [*mut c_void] = &mut [
                        &mut p_ptr as *mut _ as *mut c_void,
                        &mut len_i as *mut _ as *mut c_void,
                        &mut first_i as *mut _ as *mut c_void,
                        &mut per_ptr as *mut _ as *mut c_void,
                        &mut n_i as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];
                    self.stream.launch(&func, grid, block, 0, args)
                        .map_err(|e| CudaQstickError::Cuda(e.to_string()))?;
                }
                start += chunk;
            }
        } else {
            let func = self.module.get_function("qstick_batch_prefix_f32")
                .map_err(|e| CudaQstickError::Cuda(e.to_string()))?;
            unsafe { (*(self as *const _ as *mut CudaQstick)).last_batch = Some(BatchKernelSelected::Plain { block_x }); }
            self.maybe_log_batch_debug();

            let grid_x = ((len as u32) + block_x - 1) / block_x;
            let block: BlockSize = (block_x, 1, 1).into();
            let mut start = 0usize;
            while start < n_combos {
                let chunk = (n_combos - start).min(MAX_GRID_Y);
                let grid: GridSize = (grid_x.max(1), chunk as u32, 1).into();
                unsafe {
                    let mut p_ptr = d_prefix.as_device_ptr().as_raw();
                    let mut len_i = len as i32;
                    let mut first_i = first_valid as i32;
                    let mut per_ptr = d_periods.as_device_ptr().as_raw().wrapping_add((start * std::mem::size_of::<i32>()) as u64);
                    let mut n_i = chunk as i32;
                    let mut out_ptr = d_out.as_device_ptr().as_raw().wrapping_add((start * len * std::mem::size_of::<f32>()) as u64);
                    let args: &mut [*mut c_void] = &mut [
                        &mut p_ptr as *mut _ as *mut c_void,
                        &mut len_i as *mut _ as *mut c_void,
                        &mut first_i as *mut _ as *mut c_void,
                        &mut per_ptr as *mut _ as *mut c_void,
                        &mut n_i as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];
                    self.stream.launch(&func, grid, block, 0, args)
                        .map_err(|e| CudaQstickError::Cuda(e.to_string()))?;
                }
                start += chunk;
            }
        }
        Ok(())
    }

    fn launch_many_series_kernel(
        &self,
        d_prefix_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaQstickError> {
        // Only OneD for now; grid.y over series, grid.x over time
        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 256,
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
        };
        let func = self
            .module
            .get_function("qstick_many_series_one_param_f32")
            .map_err(|e| CudaQstickError::Cuda(e.to_string()))?;

        unsafe { (*(self as *const _ as *mut CudaQstick)).last_many = Some(ManySeriesKernelSelected::OneD { block_x }); }
        self.maybe_log_many_debug();

        let grid_x = ((rows as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), cols as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut p_ptr = d_prefix_tm.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut num_series_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut fv_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaQstickError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    // ------------------- Public device entry points -------------------
    pub fn qstick_batch_dev(
        &self,
        open_f32: &[f32],
        close_f32: &[f32],
        sweep: &QstickBatchRange,
    ) -> Result<DeviceArrayF32, CudaQstickError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(open_f32, close_f32, sweep)?;

        // VRAM estimate: prefix (len+1) + periods + outputs
        let bytes_required = (len + 1) * 4                    // prefix
            + combos.len() * 4                                // periods
            + combos.len() * len * 4;                         // outputs
        let headroom = env::var("CUDA_MEM_HEADROOM").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(64 * 1024 * 1024);
        if !Self::will_fit(bytes_required, headroom) {
            return Err(CudaQstickError::InvalidInput(format!(
                "insufficient VRAM: need ~{} MB (incl headroom)",
                (bytes_required + headroom) / (1024 * 1024)
            )));
        }

        // Build prefix on host
        let (prefix, _fv2, _l2) = Self::build_diff_prefix_f32(open_f32, close_f32);
        let d_prefix = DeviceBuffer::from_slice(&prefix)
            .map_err(|e| CudaQstickError::Cuda(e.to_string()))?;
        let periods_i32: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let d_periods = DeviceBuffer::from_slice(&periods_i32)
            .map_err(|e| CudaQstickError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(combos.len() * len, &self.stream)
                .map_err(|e| CudaQstickError::Cuda(e.to_string()))?
        };

        self.launch_batch_kernel(&d_prefix, &d_periods, len, first_valid, combos.len(), &mut d_out)?;
        self.stream
            .synchronize()
            .map_err(|e| CudaQstickError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 { buf: d_out, rows: combos.len(), cols: len })
    }

    pub fn prepare_many_series_inputs(
        open_tm_f32: &[f32],
        close_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<(Vec<f32>, Vec<i32>), CudaQstickError> {
        if cols == 0 || rows == 0 { return Err(CudaQstickError::InvalidInput("empty matrix".into())); }
        if open_tm_f32.len() != cols * rows || close_tm_f32.len() != cols * rows {
            return Err(CudaQstickError::InvalidInput("shape mismatch".into()));
        }
        if period == 0 || period > rows {
            return Err(CudaQstickError::InvalidInput("invalid period".into()));
        }
        // Build (rows+1) x cols prefix of diffs, time-major
        let mut prefix_tm = vec![0.0f32; (rows + 1) * cols];
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            // find first valid in this series
            let mut fv = 0usize;
            for t in 0..rows {
                let o = open_tm_f32[t * cols + s];
                let c = close_tm_f32[t * cols + s];
                if !o.is_nan() && !c.is_nan() { fv = t; break; }
                if t == rows - 1 { return Err(CudaQstickError::InvalidInput("all values NaN in a series".into())); }
            }
            first_valids[s] = fv as i32;
            let mut acc = 0.0f64;
            for t in 0..rows {
                if t < fv {
                    prefix_tm[(t + 1) * cols + s] = acc as f32;
                } else {
                    let d = (close_tm_f32[t * cols + s] as f64) - (open_tm_f32[t * cols + s] as f64);
                    acc += d;
                    prefix_tm[(t + 1) * cols + s] = acc as f32;
                }
            }
        }
        Ok((prefix_tm, first_valids))
    }

    pub fn qstick_many_series_one_param_time_major_dev(
        &self,
        open_tm_f32: &[f32],
        close_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaQstickError> {
        // Validate + build host prefixes
        let (prefix_tm, first_valids) = Self::prepare_many_series_inputs(open_tm_f32, close_tm_f32, cols, rows, period)?;

        // VRAM estimate: prefix (rows+1)*cols + first_valids + output rows*cols
        let bytes_required = (rows + 1) * cols * 4    // prefix
            + cols * 4                                // first_valids
            + rows * cols * 4;                        // output
        let headroom = env::var("CUDA_MEM_HEADROOM").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(64 * 1024 * 1024);
        if !Self::will_fit(bytes_required, headroom) {
            return Err(CudaQstickError::InvalidInput(format!(
                "insufficient VRAM: need ~{} MB (incl headroom)",
                (bytes_required + headroom) / (1024 * 1024)
            )));
        }

        let d_prefix_tm = DeviceBuffer::from_slice(&prefix_tm)
            .map_err(|e| CudaQstickError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaQstickError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(rows * cols, &self.stream)
                .map_err(|e| CudaQstickError::Cuda(e.to_string()))?
        };

        self.launch_many_series_kernel(&d_prefix_tm, &d_first_valids, cols, rows, period, &mut d_out_tm)?;
        self.stream
            .synchronize()
            .map_err(|e| CudaQstickError::Cuda(e.to_string()))?;
        Ok(DeviceArrayF32 { buf: d_out_tm, rows, cols })
    }
}

// ---------------- Benches registration (for benches/cuda_bench.rs) ----------------
pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::cuda::bench::helpers::gen_time_major_prices;

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "qstick",
                "batch_dev",
                "qstick_cuda_batch_dev",
                "60k_x_40periods",
                prep_qstick_batch_box,
            )
            .with_inner_iters(8),
            CudaBenchScenario::new(
                "qstick",
                "many_series_one_param",
                "qstick_cuda_many_series_one_param",
                "250x1m",
                prep_qstick_many_series_box,
            )
            .with_inner_iters(4),
        ]
    }

    struct QsBatchState {
        cuda: CudaQstick,
        d_prefix: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_out: DeviceBuffer<f32>,
        len: usize,
        n_combos: usize,
        first_valid: usize,
    }
    impl CudaBenchState for QsBatchState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_prefix,
                    &self.d_periods,
                    self.len,
                    self.first_valid,
                    self.n_combos,
                    &mut self.d_out,
                )
                .expect("qstick launch");
            self.cuda.synchronize().expect("sync");
        }
    }

    fn prep_qstick_batch() -> QsBatchState {
        let mut cuda = CudaQstick::new(0).expect("cuda qstick");
        cuda.set_policy(CudaQstickPolicy { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Auto });

        let len = 60_000usize;
        let mut open = vec![f32::NAN; len];
        let mut close = vec![f32::NAN; len];
        for i in 3..len {
            let x = i as f32;
            open[i] = (x * 0.0007).cos() + 0.01 * (x * 0.0001).sin();
            close[i] = open[i] + 0.05 * (x * 0.0017).sin();
        }
        let (prefix, first_valid, _len) = CudaQstick::build_diff_prefix_f32(&open, &close);
        let periods: Vec<i32> = (5..=200).step_by(5).map(|p| p as i32).collect();
        let d_prefix = DeviceBuffer::from_slice(&prefix).expect("d_prefix");
        let d_periods = DeviceBuffer::from_slice(&periods).expect("d_periods");
        let n_combos = periods.len();
        let d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n_combos * len) }.expect("d_out");

        QsBatchState { cuda, d_prefix, d_periods, d_out, len, n_combos, first_valid }
    }
    fn prep_qstick_batch_box() -> Box<dyn CudaBenchState> { Box::new(prep_qstick_batch()) }

    struct QsManyState {
        cuda: CudaQstick,
        d_prefix_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        d_out_tm: DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        period: usize,
    }
    impl CudaBenchState for QsManyState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series_kernel(
                    &self.d_prefix_tm,
                    &self.d_first_valids,
                    self.cols,
                    self.rows,
                    self.period,
                    &mut self.d_out_tm,
                )
                .expect("qstick many launch");
            self.cuda.synchronize().expect("sync");
        }
    }

    fn prep_qstick_many_series() -> QsManyState {
        let mut cuda = CudaQstick::new(0).expect("cuda qstick");
        cuda.set_policy(CudaQstickPolicy { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Auto });

        let cols = 250usize;
        let rows = 1_000_000usize;
        // Generate a synthetic open/close TM pair
        let p_tm = gen_time_major_prices(cols, rows);
        let mut o_tm = vec![0f32; cols * rows];
        let mut c_tm = vec![0f32; cols * rows];
        for t in 0..rows { for s in 0..cols { let idx = t * cols + s; o_tm[idx] = p_tm[idx] - 0.05; c_tm[idx] = p_tm[idx] + 0.05; } }

        let period = 21usize;
        let (prefix_tm, first_valids) = CudaQstick::prepare_many_series_inputs(&o_tm, &c_tm, cols, rows, period).expect("prep");
        let d_prefix_tm = DeviceBuffer::from_slice(&prefix_tm).expect("d_prefix_tm");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out_tm");
        QsManyState { cuda, d_prefix_tm, d_first_valids, d_out_tm, cols, rows, period }
    }
    fn prep_qstick_many_series_box() -> Box<dyn CudaBenchState> { Box::new(prep_qstick_many_series()) }
}

