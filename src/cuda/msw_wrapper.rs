//! CUDA wrapper for Mesa Sine Wave (MSW).
//!
//! Parity with ALMA/CWMA wrappers:
//! - PTX load via DetermineTargetFromContext + OptLevel O2 with simpler fallbacks.
//! - NON_BLOCKING stream.
//! - Policy enums for batch and many-series; selection recorded and optionally logged once
//!   when BENCH_DEBUG=1.
//! - VRAM checks with ~64MB headroom and grid.y chunking (≤ 65_535) for batch.
//! - Outputs stacked: batch rows = 2 * n_combos (row0=sine, row1=lead per combo),
//!   many-series cols = 2 * num_series (col0..N-1 sine, colN..2N-1 lead).

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32;
use crate::indicators::msw::{MswBatchRange, MswParams};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use thiserror::Error;

// Must match the kernel's MSW_CHUNK_PER_THREAD macro (default 8)
const MSW_CHUNK_PER_THREAD: u32 = 8;

#[derive(Debug, Error)]
pub enum CudaMswError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buf={buf}, current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaMsw {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy_batch: BatchKernelPolicy,
    policy_many: ManySeriesKernelPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaMsw {
    pub fn new(device_id: usize) -> Result<Self, CudaMswError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/msw_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy_batch: BatchKernelPolicy::Auto,
            policy_many: ManySeriesKernelPolicy::Auto,
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    // ----- Policy control & introspection -----
    pub fn set_batch_policy(&mut self, p: BatchKernelPolicy) {
        self.policy_batch = p;
    }
    pub fn set_many_series_policy(&mut self, p: ManySeriesKernelPolicy) {
        self.policy_many = p;
    }
    pub fn batch_policy(&self) -> BatchKernelPolicy {
        self.policy_batch
    }
    pub fn many_series_policy(&self) -> ManySeriesKernelPolicy {
        self.policy_many
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
    pub fn synchronize(&self) -> Result<(), CudaMswError> {
        self.stream.synchronize().map_err(Into::into)
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if self.debug_batch_logged {
            return;
        }
        if env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per = env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] MSW batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaMsw)).debug_batch_logged = true;
                }
            }
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged {
            return;
        }
        if self.debug_many_logged {
            return;
        }
        if env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per = env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] MSW many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaMsw)).debug_many_logged = true;
                }
            }
        }
    }

    // ----- Helpers -----
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }
    fn will_fit(required_bytes: usize, headroom: usize) -> Result<(), CudaMswError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            if required_bytes.saturating_add(headroom) > free {
                return Err(CudaMswError::OutOfMemory {
                    required: required_bytes,
                    free,
                    headroom,
                });
            }
            Ok(())
        } else {
            // If mem_get_info failed earlier, conservatively succeed here but callers
            // still see any underlying CUDA errors from allocations.
            Ok(())
        }
    }
    fn first_valid_f32(series: &[f32]) -> Result<usize, CudaMswError> {
        if series.is_empty() {
            return Err(CudaMswError::InvalidInput("empty series".into()));
        }
        series
            .iter()
            .position(|x| x.is_finite())
            .ok_or_else(|| CudaMswError::InvalidInput("all values are NaN".into()))
    }
    fn expand_grid(range: &MswBatchRange) -> Result<Vec<MswParams>, CudaMswError> {
        fn axis_usize(
            (start, end, step): (usize, usize, usize),
        ) -> Result<Vec<usize>, CudaMswError> {
            if step == 0 || start == end {
                return Ok(vec![start]);
            }
            if start < end {
                let mut out = Vec::new();
                let mut v = start;
                while v <= end {
                    out.push(v);
                    match v.checked_add(step) {
                        Some(next) => {
                            if next == v {
                                break;
                            }
                            v = next;
                        }
                        None => break,
                    }
                }
                if out.is_empty() {
                    return Err(CudaMswError::InvalidInput(format!(
                        "invalid range: start={}, end={}, step={}",
                        start, end, step
                    )));
                }
                return Ok(out);
            }
            // Reversed bounds: walk downwards, including both ends.
            let mut out = Vec::new();
            let mut v = start;
            loop {
                out.push(v);
                if v <= end {
                    break;
                }
                match v.checked_sub(step) {
                    Some(next) => {
                        v = next;
                        if v <= end {
                            break;
                        }
                    }
                    None => break,
                }
            }
            if out.is_empty() {
                return Err(CudaMswError::InvalidInput(format!(
                    "invalid range: start={}, end={}, step={}",
                    start, end, step
                )));
            }
            Ok(out)
        }

        let (s, e, step) = range.period;
        let periods = axis_usize((s, e, step))?;
        if periods.is_empty() {
            return Err(CudaMswError::InvalidInput(format!(
                "no parameter combinations for range start={}, end={}, step={}",
                s, e, step
            )));
        }
        Ok(periods
            .into_iter()
            .map(|p| MswParams { period: Some(p) })
            .collect())
    }

    // Dynamic shared memory for optimized kernel (use_lut = true):
    // floats = cos(P) + sin(P) + tile(T + P - 1), with T = block_x * MSW_CHUNK_PER_THREAD.
    #[inline]
    fn dyn_smem_floats(period: usize, block_x: u32) -> usize {
        let t = (block_x as usize) * (MSW_CHUNK_PER_THREAD as usize);
        t + 3usize.saturating_mul(period) - 1
    }

    fn try_pick_block_x(
        &self,
        func: &Function,
        period: usize,
        prefer: Option<u32>,
    ) -> Result<(u32, usize), CudaMswError> {
        let mut candidates = [512u32, 384, 256, 192, 128, 96, 64, 48, 32];
        if let Some(px) = prefer {
            if !candidates.contains(&px) {
                candidates[0] = px;
            } else {
                let mut v = vec![px];
                v.extend(candidates.into_iter().filter(|&b| b != px));
                candidates = v
                    .try_into()
                    .unwrap_or([px, 256, 192, 128, 96, 64, 48, 32, 32]);
            }
        }
        let dev = Device::get_device(self.device_id)?;
        let max_threads = dev
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;

        for &bx in &candidates {
            if bx > max_threads {
                continue;
            }
            let need_bytes = Self::dyn_smem_floats(period, bx) * std::mem::size_of::<f32>();
            let avail = func
                .available_dynamic_shared_memory_per_block(
                    GridSize::xy(1, 1),
                    BlockSize::xyz(bx, 1, 1),
                )
                .unwrap_or(48 * 1024);
            if need_bytes <= avail {
                return Ok((bx, need_bytes));
            }
        }
        // Fallback: try a small bx if nothing fit
        let bx = 64u32.min(max_threads);
        let need_bytes = Self::dyn_smem_floats(period, bx) * std::mem::size_of::<f32>();
        let avail = func
            .available_dynamic_shared_memory_per_block(GridSize::xy(1, 1), BlockSize::xyz(bx, 1, 1))
            .unwrap_or(48 * 1024);
        if need_bytes > avail {
            return Err(CudaMswError::InvalidInput(format!(
                "period {} requires too much shared memory (need {}B, avail {}B)",
                period, need_bytes, avail
            )));
        }
        Ok((bx, need_bytes))
    }

    // ----- Batch: one series × many params -----
    fn launch_batch_chunk(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods_all: &DeviceBuffer<i32>,
        series_len: usize,
        first_valid: usize,
        chunk_rows: usize,
        base_row: usize,
        d_out: &mut DeviceBuffer<f32>,
        block_x: u32,
        shared_bytes: usize,
        func: &Function,
    ) -> Result<(), CudaMswError> {
        // Batch kernel currently advances one output per thread (tile = block_x)
        let grid_x = ((series_len as u32) + block_x - 1) / block_x;
        let grid_y = chunk_rows as u32;
        let grid: GridSize = (grid_x.max(1), grid_y, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        // Basic launch config validation against device limits
        let dev = Device::get_device(self.device_id)?;
        let max_bx = dev
            .get_attribute(DeviceAttribute::MaxBlockDimX)
            .unwrap_or(1024) as u32;
        if block_x > max_bx || grid_y == 0 {
            return Err(CudaMswError::LaunchConfigTooLarge {
                gx: grid_x.max(1),
                gy: grid_y,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods_all.as_device_ptr().as_raw()
                + ((base_row * std::mem::size_of::<i32>()) as u64);
            let mut len_i = series_len as i32;
            let mut combos_i = chunk_rows as i32;
            let mut first_valid_i = first_valid as i32;
            // offset out by base_row (2 rows per combo)
            let mut out_ptr = d_out.as_device_ptr().as_raw()
                + ((2 * base_row * series_len) * std::mem::size_of::<f32>()) as u64;
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(func, grid, block, shared_bytes as u32, args)?;
        }
        unsafe {
            (*(self as *const _ as *mut CudaMsw)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();
        Ok(())
    }

    pub fn msw_batch_dev(
        &self,
        prices_f32: &[f32],
        sweep: &MswBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<MswParams>), CudaMswError> {
        if prices_f32.is_empty() {
            return Err(CudaMswError::InvalidInput("empty series".into()));
        }
        let first_valid = Self::first_valid_f32(prices_f32)?;
        let combos = Self::expand_grid(sweep)?;

        let len = prices_f32.len();
        let mut periods_i32 = Vec::with_capacity(combos.len());
        let mut max_p = 0usize;
        for prm in &combos {
            let p = prm.period.unwrap_or(0);
            if p == 0 {
                return Err(CudaMswError::InvalidInput("period must be >= 1".into()));
            }
            if p > len {
                return Err(CudaMswError::InvalidInput(format!(
                    "period {} exceeds length {}",
                    p, len
                )));
            }
            if len - first_valid < p {
                return Err(CudaMswError::InvalidInput(format!(
                    "not enough valid data: need {}, valid {}",
                    p,
                    len - first_valid
                )));
            }
            max_p = max_p.max(p);
            periods_i32.push(p as i32);
        }

        // VRAM estimate: input + periods + output(2*rows*len)
        let rows = combos.len();
        let prices_bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaMswError::InvalidInput("series length overflow".into()))?;
        let periods_bytes = periods_i32
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaMswError::InvalidInput("periods size overflow".into()))?;
        let out_elems = rows
            .checked_mul(len)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| CudaMswError::InvalidInput("rows*len overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaMswError::InvalidInput("output size overflow".into()))?;
        let req = prices_bytes
            .checked_add(periods_bytes)
            .and_then(|b| b.checked_add(out_bytes))
            .ok_or_else(|| CudaMswError::InvalidInput("total VRAM size overflow".into()))?;
        let headroom = 64 * 1024 * 1024usize;
        Self::will_fit(req, headroom)?;

        // Device buffers
        let d_prices = unsafe { DeviceBuffer::from_slice_async(prices_f32, &self.stream) }?;
        let d_periods = unsafe { DeviceBuffer::from_slice_async(&periods_i32, &self.stream) }?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_elems, &self.stream) }?;

        // Select block_x based on max_p for shared memory sizing once using the function handle
        // (we still pass it to launch method via try_pick inside, so warm up function cache here)
        let func = self
            .module
            .get_function("msw_batch_f32")
            .map_err(|_| CudaMswError::MissingKernelSymbol { name: "msw_batch_f32" })?;
        let (block_x, shared_bytes) = self.try_pick_block_x(
            &func,
            max_p,
            match self.policy_batch {
                BatchKernelPolicy::Plain { block_x } => Some(block_x),
                _ => None,
            },
        )?;

        // Chunk grid.y if needed
        let mut base = 0usize;
        const MAX_Y: usize = 65_535;
        while base < combos.len() {
            let take = (combos.len() - base).min(MAX_Y);
            self.launch_batch_chunk(&d_prices, &d_periods, len, first_valid, take, base, &mut d_out, block_x, shared_bytes, &func)?;
            base += take;
        }
        self.stream.synchronize()?;

        Ok((
            DeviceArrayF32 {
                buf: d_out,
                rows: 2 * combos.len(),
                cols: len,
            },
            combos,
        ))
    }

    // ----- Many-series: time‑major, one param -----
    pub fn msw_many_series_one_param_time_major_dev(
        &self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &MswParams,
    ) -> Result<DeviceArrayF32, CudaMswError> {
        if cols == 0 || rows == 0 {
            return Err(CudaMswError::InvalidInput("cols/rows must be > 0".into()));
        }
        let expected_elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaMswError::InvalidInput("cols*rows overflow".into()))?;
        if prices_tm_f32.len() != expected_elems {
            return Err(CudaMswError::InvalidInput(
                "data length != cols*rows".into(),
            ));
        }
        let period = params.period.unwrap_or(0);
        if period == 0 {
            return Err(CudaMswError::InvalidInput("period must be >= 1".into()));
        }

        // Per-series first_valid and validation
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let idx = t * cols + s;
                if prices_tm_f32[idx].is_finite() {
                    fv = Some(t as i32);
                    break;
                }
            }
            let found =
                fv.ok_or_else(|| CudaMswError::InvalidInput(format!("series {} all NaN", s)))?;
            if (rows as i32 - found) < period as i32 {
                return Err(CudaMswError::InvalidInput(format!(
                    "series {} lacks data: need {}, valid {}",
                    s,
                    period,
                    rows as i32 - found
                )));
            }
            first_valids[s] = found;
        }

        // VRAM: input + first_valids + output (rows * 2*cols)
        let prices_bytes = expected_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaMswError::InvalidInput("input size overflow".into()))?;
        let first_bytes = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaMswError::InvalidInput("first_valids size overflow".into()))?;
        let out_elems = rows
            .checked_mul(2)
            .and_then(|n| n.checked_mul(cols))
            .ok_or_else(|| CudaMswError::InvalidInput("output rows*cols overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaMswError::InvalidInput("output size overflow".into()))?;
        let req = prices_bytes
            .checked_add(first_bytes)
            .and_then(|b| b.checked_add(out_bytes))
            .ok_or_else(|| CudaMswError::InvalidInput("total VRAM size overflow".into()))?;
        Self::will_fit(req, 64 * 1024 * 1024)?;

        let d_prices = unsafe { DeviceBuffer::from_slice_async(prices_tm_f32, &self.stream) }?;
        let d_first = unsafe { DeviceBuffer::from_slice_async(&first_valids, &self.stream) }?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_elems, &self.stream) }?;

        let mut func = self
            .module
            .get_function("msw_many_series_one_param_time_major_f32")
            .map_err(|_| CudaMswError::MissingKernelSymbol {
                name: "msw_many_series_one_param_time_major_f32",
            })?;
        let (block_x, shared_bytes) = self.try_pick_block_x(&func, period, match self.policy_many { ManySeriesKernelPolicy::OneD { block_x } => Some(block_x), _ => None })?;
        // grid.x in tiles of T = block_x * MSW_CHUNK_PER_THREAD
        let t_per_block = block_x * MSW_CHUNK_PER_THREAD;
        let grid_x = ((rows as u32) + t_per_block - 1) / t_per_block;
        let grid: GridSize = (grid_x.max(1), cols as u32, 1).into();
        let dev = Device::get_device(self.device_id)?;
        let max_bx = dev
            .get_attribute(DeviceAttribute::MaxBlockDimX)
            .unwrap_or(1024) as u32;
        if block_x > max_bx {
            return Err(CudaMswError::LaunchConfigTooLarge {
                gx: grid_x.max(1),
                gy: cols as u32,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut first_ptr = d_first.as_device_ptr().as_raw();
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes as u32, args)?;
        }
        unsafe {
            (*(self as *const _ as *mut CudaMsw)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        unsafe {
            (*(self as *const _ as *mut CudaMsw)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();
        self.stream.synchronize()?;

        Ok(DeviceArrayF32 { buf: d_out, rows, cols: 2 * cols })
    }
}

// -------- Benches --------
pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 256;
    const MANY_COLS: usize = 256;
    const MANY_ROWS: usize = 16 * 1024;

    fn gen_series(n: usize) -> Vec<f32> {
        let mut v = vec![f32::NAN; n];
        for i in 64..n {
            let x = i as f32;
            v[i] = (x * 0.00123).sin() + 0.0001 * x;
        }
        for i in 64..n {
            let x = i as f32;
            v[i] = (x * 0.00123).sin() + 0.0001 * x;
        }
        v
    }

    struct BatchState {
        cuda: CudaMsw,
        prices: Vec<f32>,
        sweep: MswBatchRange,
    }
    impl CudaBenchState for BatchState {
        fn launch(&mut self) {
            let _ = self.cuda.msw_batch_dev(&self.prices, &self.sweep).unwrap();
        }
    }

    struct ManyState {
        cuda: CudaMsw,
        prices_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        params: MswParams,
    }
    impl CudaBenchState for ManyState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .msw_many_series_one_param_time_major_dev(
                    &self.prices_tm,
                    self.cols,
                    self.rows,
                    &self.params,
                )
                .unwrap();
        }
    }

    fn prep_batch() -> Box<dyn CudaBenchState> {
        let cuda = CudaMsw::new(0).expect("cuda msw");
        let prices = gen_series(ONE_SERIES_LEN);
        let sweep = MswBatchRange {
            period: (8, 8 + PARAM_SWEEP - 1, 1),
        };
        Box::new(BatchState {
            cuda,
            prices,
            sweep,
        })
    }
    fn prep_many() -> Box<dyn CudaBenchState> {
        let cuda = CudaMsw::new(0).expect("cuda msw");
        let cols = MANY_COLS;
        let rows = MANY_ROWS;
        let mut tm = vec![f32::NAN; cols * rows];
        for s in 0..cols {
            for t in s..rows {
                let x = (t as f32) + 0.1 * (s as f32);
                tm[t * cols + s] = (0.002 * x).sin() + 0.0003 * x;
            }
        }
        let params = MswParams { period: Some(32) };
        Box::new(ManyState {
            cuda,
            prices_tm: tm,
            cols,
            rows,
            params,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "msw",
                "one_series_many_params",
                "msw_cuda_batch_dev",
                "1m_x_256",
                prep_batch,
            )
            .with_sample_size(12)
            .with_mem_required(
                ONE_SERIES_LEN * 4 + ONE_SERIES_LEN * PARAM_SWEEP * 2 * 4 + 64 * 1024 * 1024,
            ),
            CudaBenchScenario::new(
                "msw",
                "many_series_one_param",
                "msw_cuda_many_series_one_param_dev",
                "256x16k",
                prep_many,
            )
            .with_sample_size(12)
            .with_mem_required(MANY_COLS * MANY_ROWS * 3 * 4 + 64 * 1024 * 1024),
        ]
    }
}
