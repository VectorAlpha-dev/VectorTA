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

#[derive(Debug)]
pub enum CudaSqwmaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaSqwmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaSqwmaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaSqwmaError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}

impl std::error::Error for CudaSqwmaError {}

pub struct CudaSqwma {
    module: Module,
    stream: Stream,
    _context: Context,
    // Cached device attributes for launch sizing
    sm_count: i32,
    max_grid_x: u32,
    _warp_size: i32,
}

impl CudaSqwma {
    pub fn new(device_id: usize) -> Result<Self, CudaSqwmaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaSqwmaError::Cuda(e.to_string()))?;

        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaSqwmaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaSqwmaError::Cuda(e.to_string()))?;

        // Cache basic device attributes for launch heuristics
        let sm_count = device
            .get_attribute(cust::device::DeviceAttribute::MultiprocessorCount)
            .map_err(|e| CudaSqwmaError::Cuda(e.to_string()))?;
        let max_grid_x = device
            .get_attribute(cust::device::DeviceAttribute::MaxGridDimX)
            .map_err(|e| CudaSqwmaError::Cuda(e.to_string()))? as u32;
        let warp_size = device
            .get_attribute(cust::device::DeviceAttribute::WarpSize)
            .map_err(|e| CudaSqwmaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/sqwma_kernel.ptx"));

        // Prefer higher JIT opt level first, then fallback for compatibility
        let module = Module::from_ptx(
            ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O4),
            ],
        )
        .or_else(|_| {
            Module::from_ptx(
                ptx,
                &[
                    ModuleJitOption::DetermineTargetFromContext,
                    ModuleJitOption::OptLevel(OptLevel::O2),
                ],
            )
        })
        .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
        .or_else(|_| Module::from_ptx(ptx, &[]))
        .map_err(|e| CudaSqwmaError::Cuda(e.to_string()))?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaSqwmaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            sm_count,
            max_grid_x,
            _warp_size: warp_size,
        })
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }

    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> {
        unsafe {
            let mut free: usize = 0;
            let mut total: usize = 0;
            let res = cu::cuMemGetInfo_v2(&mut free as *mut usize, &mut total as *mut usize);
            if res == cu::CUresult::CUDA_SUCCESS {
                Some((free, total))
            } else {
                None
            }
        }
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
        let expected = inputs.series_len * inputs.combos.len();
        if out.len() != expected {
            return Err(CudaSqwmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                expected
            )));
        }

        let arr = self.run_batch_kernel(prices, &inputs)?;
        arr.buf
            .copy_to(out)
            .map_err(|e| CudaSqwmaError::Cuda(e.to_string()))?;
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
        if out_tm.len() != cols * rows {
            return Err(CudaSqwmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out_tm.len(),
                cols * rows
            )));
        }

        let inputs = Self::prepare_many_series_inputs(prices_tm_f32, cols, rows, period)?;
        let arr = self.run_many_series_kernel(prices_tm_f32, cols, rows, period, &inputs)?;
        arr.buf
            .copy_to(out_tm)
            .map_err(|e| CudaSqwmaError::Cuda(e.to_string()))
    }

    fn run_batch_kernel(
        &self,
        prices: &[f32],
        inputs: &BatchInputs,
    ) -> Result<DeviceArrayF32, CudaSqwmaError> {
        let n_combos = inputs.combos.len();
        let series_len = inputs.series_len;

        let prices_bytes = series_len * std::mem::size_of::<f32>();
        let periods_bytes = n_combos * std::mem::size_of::<i32>();
        let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + periods_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024; // ~64MB safety margin

        if !Self::will_fit(required, headroom) {
            return Err(CudaSqwmaError::InvalidInput(
                "insufficient device memory for SQWMA batch launch".into(),
            ));
        }

        let d_prices =
            DeviceBuffer::from_slice(prices).map_err(|e| CudaSqwmaError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&inputs.periods)
            .map_err(|e| CudaSqwmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(series_len * n_combos) }
                .map_err(|e| CudaSqwmaError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            series_len,
            n_combos,
            inputs.first_valid,
            inputs.max_period,
            &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaSqwmaError::Cuda(e.to_string()))?;

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
        let prices_bytes = prices_tm_f32.len() * std::mem::size_of::<f32>();
        let first_valid_bytes = prepared.first_valids.len() * std::mem::size_of::<i32>();
        let out_bytes = prices_tm_f32.len() * std::mem::size_of::<f32>();
        let required = prices_bytes + first_valid_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024; // ~64MB safety margin

        if !Self::will_fit(required, headroom) {
            return Err(CudaSqwmaError::InvalidInput(
                "insufficient device memory for SQWMA many-series launch".into(),
            ));
        }

        let d_prices_tm = DeviceBuffer::from_slice(prices_tm_f32)
            .map_err(|e| CudaSqwmaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&prepared.first_valids)
            .map_err(|e| CudaSqwmaError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(prices_tm_f32.len()) }
                .map_err(|e| CudaSqwmaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            period,
            cols,
            rows,
            &d_first_valids,
            &mut d_out_tm,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaSqwmaError::Cuda(e.to_string()))?;

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
            .map_err(|e| CudaSqwmaError::Cuda(e.to_string()))?;
        let block_x: u32 = Self::block_x();
        let grid_x: u32 = self.grid_x_for_series(series_len);
        let grid: GridSize = (grid_x, n_combos as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        let shared_bytes: u32 = 0; // optimized kernels do not use dynamic shared memory

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
            self.stream
                .launch(&func, grid, block, shared_bytes, args)
                .map_err(|e| CudaSqwmaError::Cuda(e.to_string()))?
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
            .map_err(|e| CudaSqwmaError::Cuda(e.to_string()))?;
        let block_x: u32 = Self::block_x();
        let grid_x: u32 = self.grid_x_for_series(series_len);
        let grid: GridSize = (grid_x, num_series as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        let shared_bytes: u32 = 0; // optimized kernels do not use dynamic shared memory

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
            self.stream
                .launch(&func, grid, block, shared_bytes, args)
                .map_err(|e| CudaSqwmaError::Cuda(e.to_string()))?
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
        let need = if tile == 0 { 1 } else { ((series_len as u64) + tile - 1) / tile };
        // target many resident blocks per SM to hide latency
        let target = (self.sm_count.max(1) as u32) * 32;
        let gx = std::cmp::max(1, std::cmp::min(need.min(self.max_grid_x as u64) as u32, target));
        gx
    }

    fn prepare_batch_inputs(
        prices: &[f32],
        sweep: &SqwmaBatchRange,
    ) -> Result<BatchInputs, CudaSqwmaError> {
        if prices.is_empty() {
            return Err(CudaSqwmaError::InvalidInput("empty prices".into()));
        }

        let combos = expand_grid_sqwma(sweep);
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
        if prices_tm_f32.len() != cols * rows {
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
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};

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

    struct BatchState {
        cuda: CudaSqwma,
        price: Vec<f32>,
        sweep: SqwmaBatchRange,
    }
    impl CudaBenchState for BatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .sqwma_batch_dev(&self.price, &self.sweep)
                .expect("sqwma batch");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaSqwma::new(0).expect("cuda sqwma");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = SqwmaBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
        };
        Box::new(BatchState { cuda, price, sweep })
    }

    struct ManyState {
        cuda: CudaSqwma,
        data_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        period: usize,
    }
    impl CudaBenchState for ManyState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .sqwma_many_series_one_param_time_major_dev(
                    &self.data_tm,
                    self.cols,
                    self.rows,
                    self.period,
                )
                .expect("sqwma many-series");
        }
    }
    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaSqwma::new(0).expect("cuda sqwma");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let period = 64;
        Box::new(ManyState {
            cuda,
            data_tm,
            cols,
            rows,
            period,
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

fn expand_grid_sqwma(range: &SqwmaBatchRange) -> Vec<SqwmaParams> {
    let (start, end, step) = range.period;
    if step == 0 || start == end {
        return vec![SqwmaParams {
            period: Some(start),
        }];
    }
    (start..=end)
        .step_by(step)
        .map(|p| SqwmaParams { period: Some(p) })
        .collect()
}
