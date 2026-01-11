//! CUDA support for Forecast Oscillator (FOSC).
//!
//! Math pattern classification: recurrence/IIR.
//! - Batch (one series × many params): each parameter row is processed by a
//!   single thread that scans time sequentially with O(1) sliding OLS updates.
//! - Many-series × one-param (time-major): each series/column is processed by
//!   one thread with the same scan. No global prefixes are required.
//!
//! Semantics match the scalar FOSC implementation:
//! - Warmup per row/series: warm = first_valid + period - 1
//! - Warmup prefix is filled with NaN
//! - If current value is NaN or 0.0, output is NaN
//! - Critical accumulations and OLS solve use f64 for parity; outputs are f32

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::fosc::{FoscBatchRange, FoscParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::cell::Cell;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaFoscError {
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
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    // In this recurrence kernel, a single thread processes each row.
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    // 1D launch with each series handled by one thread in X.
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaFoscPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaFoscPolicy {
    fn default() -> Self {
        Self { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Auto }
    }
}

pub struct CudaFosc {
    module: Module,
    stream: Stream,
    context: std::sync::Arc<Context>,
    device_id: u32,
    policy: CudaFoscPolicy,
    last_batch_block_x: Cell<Option<u32>>,
    last_many_block_x: Cell<Option<u32>>,
    // Debug: ensure we only print selection once per instance when BENCH_DEBUG=1
    debug_batch_logged: AtomicBool,
    debug_many_logged: AtomicBool,
}

impl CudaFosc {
    pub fn new(device_id: usize) -> Result<Self, CudaFoscError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = std::sync::Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/fosc_kernel.ptx"));
        let module = Module::from_ptx(
            ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O2),
            ],
        )
        .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
        .or_else(|_| Module::from_ptx(ptx, &[]))?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy: CudaFoscPolicy::default(),
            last_batch_block_x: Cell::new(None),
            last_many_block_x: Cell::new(None),
            debug_batch_logged: AtomicBool::new(false),
            debug_many_logged: AtomicBool::new(false),
        })
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        env::var("CUDA_MEM_CHECK")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(true)
    }

    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    pub fn set_policy(&mut self, policy: CudaFoscPolicy) { self.policy = policy; }
    pub fn policy(&self) -> &CudaFoscPolicy { &self.policy }
    pub fn selected_batch_block_x(&self) -> Option<u32> { self.last_batch_block_x.get() }
    pub fn selected_many_block_x(&self) -> Option<u32> { self.last_many_block_x.get() }
    pub fn device_id(&self) -> u32 { self.device_id }
    pub fn context_arc(&self) -> std::sync::Arc<Context> { self.context.clone() }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        if self.debug_batch_logged.load(Ordering::Relaxed) {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(bx) = self.last_batch_block_x.get() {
                eprintln!("[DEBUG] FOSC batch selected block_x={} (one thread per combo)", bx);
                self.debug_batch_logged.store(true, Ordering::Relaxed);
            }
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        if self.debug_many_logged.load(Ordering::Relaxed) {
            return;
        }
        if self.debug_many_logged.load(Ordering::Relaxed) {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(bx) = self.last_many_block_x.get() {
                eprintln!("[DEBUG] FOSC many-series selected block_x={} (one thread per series)", bx);
                self.debug_many_logged.store(true, Ordering::Relaxed);
            }
        }
    }

    // ---------- One-series × many-params ----------

    pub fn fosc_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &FoscBatchRange,
    ) -> Result<DeviceArrayF32, CudaFoscError> {
        let (periods, first_valid) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let len = data_f32.len();
        let n_combos = periods.len();
        let elems = len
            .checked_mul(n_combos)
            .ok_or_else(|| CudaFoscError::InvalidInput("rows*cols overflow".into()))?;

        // VRAM bytes: input + periods + output; +64MB headroom
        let headroom = 64 * 1024 * 1024usize;
        let required = len
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|a| n_combos.checked_mul(std::mem::size_of::<i32>()).map(|b| a + b))
            .and_then(|c| elems.checked_mul(std::mem::size_of::<f32>()).map(|d| c + d))
            .ok_or_else(|| CudaFoscError::InvalidInput("byte size overflow".into()))?;
        if !Self::will_fit(required, headroom) {
            if let Some((free, _)) = Self::device_mem_info() {
                return Err(CudaFoscError::OutOfMemory { required, free, headroom });
            } else {
                return Err(CudaFoscError::InvalidInput(
                    "insufficient device memory for fosc_batch_dev".into(),
                ));
            }
        }

        let d_data = DeviceBuffer::from_slice(data_f32)?;
        let d_periods = DeviceBuffer::from_slice(&periods)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems)? };

        self.launch_batch_kernel(
            &d_data,
            len as i32,
            first_valid as i32,
            &d_periods,
            n_combos as i32,
            &mut d_out,
        )?;

        // Ensure producer stream completes before exposing VRAM handle to higher layers.
        self.stream.synchronize()?;

        Ok(DeviceArrayF32 { buf: d_out, rows: n_combos, cols: len })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel(
        &self,
        d_data: &DeviceBuffer<f32>,
        len: i32,
        first_valid: i32,
        d_periods: &DeviceBuffer<i32>,
        n_combos: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaFoscError> {
        if len <= 0 || n_combos <= 0 {
            return Ok(());
        }
        let func = self
            .module
            .get_function("fosc_batch_f32")
            .map_err(|_| CudaFoscError::MissingKernelSymbol { name: "fosc_batch_f32" })?;

        let block_x = match self.policy.batch {
            BatchKernelPolicy::OneD { block_x } if block_x > 0 => block_x,
            _ => 256,
        };
        let grid_x = ((n_combos as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1u32, 1u32).into();
        let block: BlockSize = (block_x, 1u32, 1u32).into();

        self.validate_launch(grid_x.max(1), 1, 1, block_x, 1, 1)?;

        unsafe {
            let mut p_data = d_data.as_device_ptr().as_raw();
            let mut p_len = len;
            let mut p_first = first_valid;
            let mut p_periods = d_periods.as_device_ptr().as_raw();
            let mut p_n = n_combos;
            let mut p_out = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_data as *mut _ as *mut c_void,
                &mut p_len as *mut _ as *mut c_void,
                &mut p_first as *mut _ as *mut c_void,
                &mut p_periods as *mut _ as *mut c_void,
                &mut p_n as *mut _ as *mut c_void,
                &mut p_out as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        self.last_batch_block_x.set(Some(block_x));
        self.maybe_log_batch_debug();
        Ok(())
    }

    // ---------- Many-series × one-param (time-major) ----------

    pub fn fosc_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &FoscParams,
    ) -> Result<DeviceArrayF32, CudaFoscError> {
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaFoscError::InvalidInput("rows*cols overflow".into()))?;
        let headroom = 64 * 1024 * 1024usize;
        let required = elems
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|a| cols.checked_mul(std::mem::size_of::<i32>()).map(|b| a + b))
            .and_then(|c| elems.checked_mul(std::mem::size_of::<f32>()).map(|d| c + d))
            .ok_or_else(|| CudaFoscError::InvalidInput("byte size overflow".into()))?;
        if !Self::will_fit(required, headroom) {
            if let Some((free, _)) = Self::device_mem_info() {
                return Err(CudaFoscError::OutOfMemory { required, free, headroom });
            } else {
                return Err(CudaFoscError::InvalidInput(
                    "insufficient device memory for fosc_many_series_one_param".into(),
                ));
            }
        }

        let d_data = DeviceBuffer::from_slice(data_tm_f32)?;
        let d_fv = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems)? };

        self.launch_many_series_kernel(
            &d_data,
            &d_fv,
            cols as i32,
            rows as i32,
            period as i32,
            &mut d_out,
        )?;

        // Ensure producer stream completes before exposing VRAM handle.
        self.stream.synchronize()?;

        Ok(DeviceArrayF32 { buf: d_out, rows, cols })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_kernel(
        &self,
        d_data: &DeviceBuffer<f32>,
        d_fv: &DeviceBuffer<i32>,
        cols: i32,
        rows: i32,
        period: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaFoscError> {
        if cols <= 0 || rows <= 0 {
            return Ok(());
        }
        if cols <= 0 || rows <= 0 {
            return Ok(());
        }
        let func = self
            .module
            .get_function("fosc_many_series_one_param_time_major_f32")
            .map_err(|_| {
                CudaFoscError::MissingKernelSymbol { name: "fosc_many_series_one_param_time_major_f32" }
            })?;

        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } if block_x > 0 => block_x,
            _ => 256,
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1u32, 1u32).into();
        let block: BlockSize = (block_x, 1u32, 1u32).into();

        self.validate_launch(grid_x.max(1), 1, 1, block_x, 1, 1)?;

        unsafe {
            let mut p_data = d_data.as_device_ptr().as_raw();
            let mut p_fv = d_fv.as_device_ptr().as_raw();
            let mut p_cols = cols;
            let mut p_rows = rows;
            let mut p_period = period;
            let mut p_out = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_data as *mut _ as *mut c_void,
                &mut p_fv as *mut _ as *mut c_void,
                &mut p_cols as *mut _ as *mut c_void,
                &mut p_rows as *mut _ as *mut c_void,
                &mut p_period as *mut _ as *mut c_void,
                &mut p_out as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        self.last_many_block_x.set(Some(block_x));
        self.maybe_log_many_debug();
        Ok(())
    }

    // ---------- Input validation and helpers ----------

    fn expand_periods(r: &FoscBatchRange) -> Result<Vec<usize>, CudaFoscError> {
        let (start, end, step) = r.period;
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let v: Vec<usize> = (start..=end).step_by(step).collect();
            if v.is_empty() {
                return Err(CudaFoscError::InvalidInput(
                    "invalid period sweep: produced no values".into(),
                ));
            }
            return Ok(v);
        }
        let mut v = Vec::new();
        let mut cur = start;
        loop {
            v.push(cur);
            if cur <= end {
                break;
            }
            match cur.checked_sub(step) {
                Some(n) => cur = n,
                None => break,
            }
        }
        if v.is_empty() {
            return Err(CudaFoscError::InvalidInput(
                "invalid period sweep: produced no values".into(),
            ));
        }
        Ok(v)
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &FoscBatchRange,
    ) -> Result<(Vec<i32>, usize), CudaFoscError> {
        if data_f32.is_empty() {
            return Err(CudaFoscError::InvalidInput("empty data".into()));
        }
        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaFoscError::InvalidInput("all values are NaN".into()))?;
        let periods_usize = Self::expand_periods(sweep)?;
        if periods_usize.is_empty() {
            return Err(CudaFoscError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        for &p in &periods_usize {
            if p == 0 {
                return Err(CudaFoscError::InvalidInput("period must be > 0".into()));
            }
            if p > len {
                return Err(CudaFoscError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    p, len
                )));
            }
            if len - first_valid < p {
                return Err(CudaFoscError::InvalidInput(format!(
                    "not enough valid data for period {} (valid after first {}: {})",
                    p,
                    first_valid,
                    len - first_valid
                )));
            }
        }
        Ok((
            periods_usize.into_iter().map(|p| p as i32).collect(),
            first_valid,
        ))
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &FoscParams,
    ) -> Result<(Vec<i32>, usize), CudaFoscError> {
        if cols == 0 || rows == 0 {
            return Err(CudaFoscError::InvalidInput("empty matrix".into()));
        }
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaFoscError::InvalidInput("cols*rows overflow".into()))?;
        if data_tm_f32.len() != elems {
            return Err(CudaFoscError::InvalidInput(
                "data size does not match cols*rows".into(),
            ));
        }
        let period = params.period.unwrap_or(5);
        if period == 0 {
            return Err(CudaFoscError::InvalidInput("period must be > 0".into()));
        }
        if period == 0 {
            return Err(CudaFoscError::InvalidInput("period must be > 0".into()));
        }
        if period > rows {
            return Err(CudaFoscError::InvalidInput(format!(
                "period {} exceeds series length {}",
                period, rows
            )));
        }
        let mut first_valids = vec![-1i32; cols];
        for s in 0..cols {
            let mut fv = -1i32;
            for t in 0..rows {
                if !data_tm_f32[t * cols + s].is_nan() {
                    fv = t as i32;
                    break;
                }
                if !data_tm_f32[t * cols + s].is_nan() {
                    fv = t as i32;
                    break;
                }
            }
            if fv < 0 {
                return Err(CudaFoscError::InvalidInput(format!(
                    "series {} consists entirely of NaNs",
                    s
                )));
            }
            if (rows - fv as usize) < period {
                return Err(CudaFoscError::InvalidInput(format!(
                    "series {} does not have enough valid data for period {} (valid after {}: {})",
                    s,
                    period,
                    fv,
                    rows - fv as usize
                )));
            }
            first_valids[s] = fv;
        }
        Ok((first_valids, period))
    }
    #[inline]
    fn validate_launch(
        &self,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    ) -> Result<(), CudaFoscError> {
        let dev = Device::get_device(self.device_id)?;
        let max_threads = dev
            .get_attribute(cust::device::DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_bx = dev
            .get_attribute(cust::device::DeviceAttribute::MaxBlockDimX)
            .unwrap_or(1024) as u32;
        let max_by = dev
            .get_attribute(cust::device::DeviceAttribute::MaxBlockDimY)
            .unwrap_or(1024) as u32;
        let max_bz = dev
            .get_attribute(cust::device::DeviceAttribute::MaxBlockDimZ)
            .unwrap_or(64) as u32;
        let max_gx = dev
            .get_attribute(cust::device::DeviceAttribute::MaxGridDimX)
            .unwrap_or(2_147_483_647) as u32;
        let max_gy = dev
            .get_attribute(cust::device::DeviceAttribute::MaxGridDimY)
            .unwrap_or(65_535) as u32;
        let max_gz = dev
            .get_attribute(cust::device::DeviceAttribute::MaxGridDimZ)
            .unwrap_or(65_535) as u32;

        let threads = bx.saturating_mul(by).saturating_mul(bz);
        if threads > max_threads
            || bx > max_bx
            || by > max_by
            || bz > max_bz
            || gx > max_gx
            || gy > max_gy
            || gz > max_gz
        {
            return Err(CudaFoscError::LaunchConfigTooLarge { gx, gy, gz, bx, by, bz });
        }
        Ok(())
    }
}

#[inline]
fn grid_y_chunks(n: usize) -> impl Iterator<Item = (usize, usize)> {
    struct YChunks {
        n: usize,
        launched: usize,
    }

    impl Iterator for YChunks {
        type Item = (usize, usize);
        fn next(&mut self) -> Option<Self::Item> {
            const MAX: usize = 65_535;
            if self.launched >= self.n {
                return None;
            }
            let start = self.launched;
            let len = (self.n - self.launched).min(MAX);
            self.launched += len;
            Some((start, len))
        }
    }
    YChunks { n, launched: 0 }
}

// ---------- Benches ----------

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250; // vary periods only
    const MANY_SERIES_COLS: usize = 250;
    const MANY_SERIES_ROWS: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_ROWS;
        let in_bytes = elems * std::mem::size_of::<f32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        let fv_bytes = MANY_SERIES_COLS * std::mem::size_of::<i32>();
        in_bytes + out_bytes + fv_bytes + 64 * 1024 * 1024
    }

    struct FoscBatchState {
        cuda: CudaFosc,
        d_price: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_out: DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
    }
    impl CudaBenchState for FoscBatchState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_price,
                    self.len as i32,
                    self.first_valid as i32,
                    &self.d_periods,
                    self.n_combos as i32,
                    &mut self.d_out,
                )
                .expect("fosc launch_batch_kernel");
            let _ = self.cuda.stream.synchronize();
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaFosc::new(0).expect("cuda fosc");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = FoscBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
        };
        let (periods, first_valid) =
            CudaFosc::prepare_batch_inputs(&price, &sweep).expect("prepare_batch_inputs");
        let len = price.len();
        let n_combos = periods.len();
        let d_price = DeviceBuffer::from_slice(&price).expect("d_price");
        let d_periods = DeviceBuffer::from_slice(&periods).expect("d_periods");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * len) }.expect("d_out");
        Box::new(FoscBatchState {
            cuda,
            d_price,
            d_periods,
            d_out,
            len,
            first_valid,
            n_combos,
        })
    }

    struct FoscManyState {
        cuda: CudaFosc,
        d_tm: DeviceBuffer<f32>,
        d_first: DeviceBuffer<i32>,
        d_out: DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        period: usize,
    }
    impl CudaBenchState for FoscManyState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series_kernel(
                    &self.d_tm,
                    &self.d_first,
                    self.cols as i32,
                    self.rows as i32,
                    self.period as i32,
                    &mut self.d_out,
                )
                .expect("fosc launch_many_series_kernel");
            let _ = self.cuda.stream.synchronize();
        }
    }
    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaFosc::new(0).expect("cuda fosc");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_ROWS;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = FoscParams { period: Some(14) };
        let (first_valids, period) =
            CudaFosc::prepare_many_series_inputs(&data_tm, cols, rows, &params).expect("prepare_many_series_inputs");
        let d_tm = DeviceBuffer::from_slice(&data_tm).expect("d_tm");
        let d_first = DeviceBuffer::from_slice(&first_valids).expect("d_first");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out");
        Box::new(FoscManyState {
            cuda,
            d_tm,
            d_first,
            d_out,
            cols,
            rows,
            period,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "fosc",
                "one_series_many_params",
                "fosc_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "fosc",
                "many_series_one_param",
                "fosc_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
