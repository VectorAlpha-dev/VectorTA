//! CUDA scaffolding for the Square Weighted Moving Average (SQWMA).
//!
//! The GPU path mirrors the scalar implementation: squared weights are generated
//! once per parameter combination in shared memory and each thread processes a
//! slice of the time axis. Both the one-series×many-params and many-series×one-
//! param entry points are provided to match the ALMA CUDA API surface.

#![cfg(feature = "cuda")]

use super::DeviceArrayF32;
use crate::indicators::moving_averages::sqwma::{SqwmaBatchRange, SqwmaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::DeviceBuffer;
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::sys as cu;
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaSqwmaError {
    #[error("CUDA error: {0}")]
    Cuda(#[from] cust::error::CudaError),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Out of memory on device: required={required}B, free={free}B, headroom={headroom}B")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("Missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("Launch config too large (grid=({gx},{gy},{gz}), block=({bx},{by},{bz}))")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("Invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("Device mismatch: buffer device = {buf}, current context device = {current}")]
    DeviceMismatch { buf: i32, current: i32 },
    #[error("Not implemented")]
    NotImplemented,
}

pub struct CudaSqwma {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    // Cached device attributes for launch sizing
    sm_count: i32,
    max_grid_x: u32,
    _warp_size: i32,
}

impl CudaSqwma {
    pub fn new(device_id: usize) -> Result<Self, CudaSqwmaError> {
        cust::init(CudaFlags::empty())?;

        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        // Cache basic device attributes for launch heuristics
        let sm_count = device.get_attribute(cust::device::DeviceAttribute::MultiprocessorCount)?;
        let max_grid_x = device.get_attribute(cust::device::DeviceAttribute::MaxGridDimX)? as u32;
        let warp_size = device.get_attribute(cust::device::DeviceAttribute::WarpSize)?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/sqwma_kernel.ptx"));

        // Prefer higher JIT opt level first, then fallback for compatibility
        let module = match Module::from_ptx(
            ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O4),
            ],
        ) {
            Ok(m) => m,
            Err(_) => match Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                Ok(m) => m,
                Err(_) => Module::from_ptx(ptx, &[])?
            },
        };

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            sm_count,
            max_grid_x,
            _warp_size: warp_size,
        })
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> { self._context.clone() }

    #[inline]
    pub fn device_id(&self) -> u32 { self.device_id }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }

    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> { cust::memory::mem_get_info().ok() }

    #[inline]
    fn will_fit_checked(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaSqwmaError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        let (free, _total) = match Self::device_mem_info() { Some(v) => v, None => return Ok(()) };
        let need = required_bytes
            .checked_add(headroom_bytes)
            .ok_or(CudaSqwmaError::InvalidInput("required_bytes overflow".into()))?;
        if need <= free { Ok(()) } else { Err(CudaSqwmaError::OutOfMemory { required: required_bytes, free, headroom: headroom_bytes }) }
    }

    pub fn sqwma_batch_dev(
        &self,
        prices: &[f32],
        sweep: &SqwmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaSqwmaError> {
        let inputs = Self::prepare_batch_inputs(prices, sweep)?;
        self.run_batch_kernel(prices, &inputs)
    }

    pub fn sqwma_batch_into_host_f32(
        &self,
        prices: &[f32],
        sweep: &SqwmaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<SqwmaParams>), CudaSqwmaError> {
        let inputs = Self::prepare_batch_inputs(prices, sweep)?;
        let expected = inputs
            .series_len
            .checked_mul(inputs.combos.len())
            .ok_or_else(|| CudaSqwmaError::InvalidInput("rows * cols overflow".into()))?;
        if out.len() != expected {
            return Err(CudaSqwmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                expected
            )));
        }

        let arr = self.run_batch_kernel(prices, &inputs)?;
        arr.buf.copy_to(out).map_err(CudaSqwmaError::Cuda)?;
        Ok((arr.rows, arr.cols, inputs.combos))
    }

    pub fn sqwma_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSqwmaError> {
        if series_len == 0 || n_combos == 0 || max_period <= 1 {
            return Err(CudaSqwmaError::InvalidInput(
                "series_len, n_combos must be > 0 and max_period > 1".into(),
            ));
        }
        if series_len > i32::MAX as usize || n_combos > i32::MAX as usize {
            return Err(CudaSqwmaError::InvalidInput(
                "arguments exceed kernel limits".into(),
            ));
        }

        self.launch_batch_kernel(
            d_prices,
            d_periods,
            series_len,
            n_combos,
            first_valid,
            max_period,
            d_out,
        )
    }

    pub fn sqwma_many_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        period: usize,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSqwmaError> {
        if period <= 1 || num_series == 0 || series_len == 0 {
            return Err(CudaSqwmaError::InvalidInput(
                "period must be > 1 and dimensions > 0".into(),
            ));
        }
        if period > i32::MAX as usize
            || num_series > i32::MAX as usize
            || series_len > i32::MAX as usize
        {
            return Err(CudaSqwmaError::InvalidInput(
                "arguments exceed kernel limits".into(),
            ));
        }

        self.launch_many_series_kernel(
            d_prices_tm,
            period,
            num_series,
            series_len,
            d_first_valids,
            d_out_tm,
        )
    }

    pub fn sqwma_many_series_one_param_time_major_dev(
        &self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaSqwmaError> {
        let inputs = Self::prepare_many_series_inputs(prices_tm_f32, cols, rows, period)?;
        self.run_many_series_kernel(prices_tm_f32, cols, rows, period, &inputs)
    }

    pub fn sqwma_many_series_one_param_time_major_into_host_f32(
        &self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
        out_tm: &mut [f32],
    ) -> Result<(), CudaSqwmaError> {
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaSqwmaError::InvalidInput("rows * cols overflow".into()))?;
        if out_tm.len() != expected {
            return Err(CudaSqwmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out_tm.len(),
                expected
            )));
        }

        let inputs = Self::prepare_many_series_inputs(prices_tm_f32, cols, rows, period)?;
        let arr = self.run_many_series_kernel(prices_tm_f32, cols, rows, period, &inputs)?;
        arr.buf.copy_to(out_tm).map_err(CudaSqwmaError::Cuda)
    }

    fn run_batch_kernel(
        &self,
        prices: &[f32],
        inputs: &BatchInputs,
    ) -> Result<DeviceArrayF32, CudaSqwmaError> {
        let n_combos = inputs.combos.len();
        let series_len = inputs.series_len;

        let prices_bytes = series_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaSqwmaError::InvalidInput("series_len bytes overflow".into()))?;
        let periods_bytes = n_combos
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaSqwmaError::InvalidInput("n_combos bytes overflow".into()))?;
        let elems = n_combos
            .checked_mul(series_len)
            .ok_or_else(|| CudaSqwmaError::InvalidInput("n_combos * series_len overflow".into()))?;
        let out_bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaSqwmaError::InvalidInput("out bytes overflow".into()))?;
        let required = prices_bytes
            .checked_add(periods_bytes)
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaSqwmaError::InvalidInput("required bytes overflow".into()))?;
        let headroom = 64 * 1024 * 1024; // ~64MB safety margin

        Self::will_fit_checked(required, headroom)?;

        let d_prices = DeviceBuffer::from_slice(prices)?;
        let d_periods = DeviceBuffer::from_slice(&inputs.periods)?;
        let out_elems = series_len
            .checked_mul(n_combos)
            .ok_or_else(|| CudaSqwmaError::InvalidInput("series_len * n_combos overflow".into()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(out_elems) }?;

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            series_len,
            n_combos,
            inputs.first_valid,
            inputs.max_period,
            &mut d_out,
        )?;

        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    fn run_many_series_kernel(
        &self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
        prepared: &ManySeriesInputs,
    ) -> Result<DeviceArrayF32, CudaSqwmaError> {
        let prices_bytes = prices_tm_f32
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaSqwmaError::InvalidInput("prices bytes overflow".into()))?;
        let first_valid_bytes = prepared
            .first_valids
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaSqwmaError::InvalidInput("first_valids bytes overflow".into()))?;
        let out_bytes = prices_tm_f32
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaSqwmaError::InvalidInput("out bytes overflow".into()))?;
        let required = prices_bytes
            .checked_add(first_valid_bytes)
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaSqwmaError::InvalidInput("required bytes overflow".into()))?;
        let headroom = 64 * 1024 * 1024; // ~64MB safety margin

        Self::will_fit_checked(required, headroom)?;

        let d_prices_tm = DeviceBuffer::from_slice(prices_tm_f32)?;
        let d_first_valids = DeviceBuffer::from_slice(&prepared.first_valids)?;
        let mut d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(prices_tm_f32.len()) }?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            period,
            cols,
            rows,
            &d_first_valids,
            &mut d_out_tm,
        )?;

        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows,
            cols,
        })
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSqwmaError> {
        // keep API-compatible even though max_period no longer affects shared mem
        let _ = max_period;

        let func = self
            .module
            .get_function("sqwma_batch_f32")
            .map_err(|_| CudaSqwmaError::MissingKernelSymbol { name: "sqwma_batch_f32" })?;
        let block_x: u32 = Self::block_x();
        let grid_x: u32 = self.grid_x_for_series(series_len);
        let grid: GridSize = (grid_x, n_combos as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        let shared_bytes: u32 = 0; // optimized kernels do not use dynamic shared memory

        if grid_x > self.max_grid_x {
            return Err(CudaSqwmaError::LaunchConfigTooLarge {
                gx: grid_x,
                gy: n_combos as u32,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, shared_bytes, args)?
        }
        Ok(())
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        period: usize,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSqwmaError> {
        let func = self
            .module
            .get_function("sqwma_many_series_one_param_f32")
            .map_err(|_| CudaSqwmaError::MissingKernelSymbol { name: "sqwma_many_series_one_param_f32" })?;
        let block_x: u32 = Self::block_x();
        let grid_x: u32 = self.grid_x_for_series(series_len);
        let grid: GridSize = (grid_x, num_series as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        let shared_bytes: u32 = 0; // optimized kernels do not use dynamic shared memory

        if grid_x > self.max_grid_x {
            return Err(CudaSqwmaError::LaunchConfigTooLarge {
                gx: grid_x,
                gy: num_series as u32,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_valids_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, shared_bytes, args)?
        }
        Ok(())
    }

    // === Launch sizing helpers (kept simple; no gating; defaults match kernel) ===
    #[inline]
    fn out_per_thread() -> u32 {
        if let Ok(s) = std::env::var("SQWMA_OUT_PER_THREAD") {
            if let Ok(v) = s.parse::<u32>() {
                return v.max(1);
            }
        }
        8
    }

    #[inline]
    fn block_x() -> u32 {
        if let Ok(s) = std::env::var("SQWMA_BLOCK_X") {
            if let Ok(v) = s.parse::<u32>() {
                // clamp to multiples of 32, within [32, 1024]
                let v = (v / 32).max(1).min(32) * 32;
                return v as u32;
            }
        }
        256
    }

    #[inline]
    fn grid_x_for_series(&self, series_len: usize) -> u32 {
        let bx = Self::block_x() as u64;
        let opt = Self::out_per_thread() as u64;
        let tile = bx * opt;
        let need = if tile == 0 {
            1
        } else {
            ((series_len as u64) + tile - 1) / tile
        };
        // target many resident blocks per SM to hide latency
        let target = (self.sm_count.max(1) as u32) * 32;
        let gx = std::cmp::max(
            1,
            std::cmp::min(need.min(self.max_grid_x as u64) as u32, target),
        );
        gx
    }

    fn prepare_batch_inputs(
        prices: &[f32],
        sweep: &SqwmaBatchRange,
    ) -> Result<BatchInputs, CudaSqwmaError> {
        if prices.is_empty() {
            return Err(CudaSqwmaError::InvalidInput("empty prices".into()));
        }

        let combos = expand_grid_sqwma(sweep)?;
        if combos.is_empty() {
            return Err(CudaSqwmaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let first_valid = prices
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaSqwmaError::InvalidInput("all values are NaN".into()))?;

        let series_len = prices.len();
        let mut periods = Vec::with_capacity(combos.len());
        let mut max_period = 0usize;
        for params in &combos {
            let period = params.period.unwrap_or(0);
            if period <= 1 {
                return Err(CudaSqwmaError::InvalidInput(
                    "period must be greater than 1".into(),
                ));
            }
            if period > i32::MAX as usize {
                return Err(CudaSqwmaError::InvalidInput(
                    "period exceeds i32 kernel limit".into(),
                ));
            }
            periods.push(period as i32);
            max_period = max_period.max(period);
        }

        if series_len - first_valid < max_period {
            return Err(CudaSqwmaError::InvalidInput(format!(
                "not enough valid data (needed >= {}, valid = {})",
                max_period,
                series_len - first_valid
            )));
        }

        Ok(BatchInputs {
            combos,
            periods,
            first_valid,
            series_len,
            max_period,
        })
    }

    fn prepare_many_series_inputs(
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<ManySeriesInputs, CudaSqwmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaSqwmaError::InvalidInput(
                "matrix dimensions must be positive".into(),
            ));
        }
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaSqwmaError::InvalidInput("cols * rows overflow".into()))?;
        if prices_tm_f32.len() != elems {
            return Err(CudaSqwmaError::InvalidInput("matrix shape mismatch".into()));
        }
        if period <= 1 {
            return Err(CudaSqwmaError::InvalidInput(
                "period must be greater than 1".into(),
            ));
        }
        if period > i32::MAX as usize {
            return Err(CudaSqwmaError::InvalidInput(
                "period exceeds i32 kernel limit".into(),
            ));
        }

        let mut first_valids = vec![0i32; cols];
        for series_idx in 0..cols {
            let mut fv = None;
            for row in 0..rows {
                let idx = row * cols + series_idx;
                let price = prices_tm_f32[idx];
                if !price.is_nan() {
                    fv = Some(row);
                    break;
                }
            }
            let first = fv.ok_or_else(|| {
                CudaSqwmaError::InvalidInput(format!("series {} has all NaN values", series_idx))
            })?;
            if rows - first < period {
                return Err(CudaSqwmaError::InvalidInput(format!(
                    "series {} lacks data: needed >= {}, valid = {}",
                    series_idx,
                    period,
                    rows - first
                )));
            }
            first_valids[series_idx] = first as i32;
        }

        Ok(ManySeriesInputs { first_valids })
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;
    const MANY_SERIES_COLS: usize = 250;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = elems * std::mem::size_of::<f32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct BatchDevState {
        cuda: CudaSqwma,
        d_prices: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        max_period: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for BatchDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_prices,
                    &self.d_periods,
                    self.series_len,
                    self.n_combos,
                    self.first_valid,
                    self.max_period,
                    &mut self.d_out,
                )
                .expect("sqwma batch kernel");
            self.cuda.stream.synchronize().expect("sqwma sync");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaSqwma::new(0).expect("cuda sqwma");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = SqwmaBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
        };
        let inputs = CudaSqwma::prepare_batch_inputs(&price, &sweep).expect("sqwma prepare batch");
        let n_combos = inputs.periods.len();

        let d_prices = DeviceBuffer::from_slice(&price).expect("d_prices");
        let d_periods = DeviceBuffer::from_slice(&inputs.periods).expect("d_periods");
        let d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(inputs.series_len.checked_mul(n_combos).expect("out size"))
        }
        .expect("d_out");
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(BatchDevState {
            cuda,
            d_prices,
            d_periods,
            series_len: inputs.series_len,
            n_combos,
            first_valid: inputs.first_valid,
            max_period: inputs.max_period,
            d_out,
        })
    }

    struct ManyDevState {
        cuda: CudaSqwma,
        d_prices_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for ManyDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series_kernel(
                    &self.d_prices_tm,
                    self.period,
                    self.cols,
                    self.rows,
                    &self.d_first_valids,
                    &mut self.d_out_tm,
                )
                .expect("sqwma many-series kernel");
            self.cuda.stream.synchronize().expect("sqwma many-series sync");
        }
    }
    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaSqwma::new(0).expect("cuda sqwma");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let period = 64;
        let inputs =
            CudaSqwma::prepare_many_series_inputs(&data_tm, cols, rows, period).expect("sqwma prepare many");

        let d_prices_tm = DeviceBuffer::from_slice(&data_tm).expect("d_prices_tm");
        let d_first_valids = DeviceBuffer::from_slice(&inputs.first_valids).expect("d_first_valids");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols.checked_mul(rows).expect("out size")) }
                .expect("d_out_tm");
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(ManyDevState {
            cuda,
            d_prices_tm,
            d_first_valids,
            cols,
            rows,
            period,
            d_out_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "sqwma",
                "one_series_many_params",
                "sqwma_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "sqwma",
                "many_series_one_param",
                "sqwma_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}

struct BatchInputs {
    combos: Vec<SqwmaParams>,
    periods: Vec<i32>,
    first_valid: usize,
    series_len: usize,
    max_period: usize,
}

struct ManySeriesInputs {
    first_valids: Vec<i32>,
}

fn expand_grid_sqwma(range: &SqwmaBatchRange) -> Result<Vec<SqwmaParams>, CudaSqwmaError> {
    let (start, end, step) = range.period;
    if step == 0 || start == end {
        return Ok(vec![SqwmaParams { period: Some(start) }]);
    }
    if start < end {
        let v: Vec<SqwmaParams> = (start..=end)
            .step_by(step)
            .map(|p| SqwmaParams { period: Some(p) })
            .collect();
        if v.is_empty() {
            return Err(CudaSqwmaError::InvalidInput("invalid period range (empty)".into()));
        }
        return Ok(v);
    }
    // Descend
    let mut v = Vec::new();
    let mut cur = start;
    loop {
        v.push(SqwmaParams { period: Some(cur) });
        if cur == end { break; }
        cur = cur
            .checked_sub(step)
            .ok_or_else(|| CudaSqwmaError::InvalidInput("period underflow".into()))?;
        if cur < end { break; }
    }
    if v.is_empty() {
        return Err(CudaSqwmaError::InvalidInput("invalid period range (empty)".into()));
    }
    Ok(v)
}
