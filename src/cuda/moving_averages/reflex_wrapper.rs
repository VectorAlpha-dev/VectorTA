//! CUDA scaffolding for the Reflex indicator.
//!
//! The GPU implementation mirrors the scalar Reflex algorithm: each parameter
//! combination is evaluated sequentially while its smoothed history (the last
//! `period` samples) lives in shared memory. This keeps the zero-copy contract
//! of the CPU paths while exposing the same device/host helpers provided by the
//! other CUDA-enabled moving averages (ALMA, SQWMA, etc.).

#![cfg(feature = "cuda")]

use super::DeviceArrayF32;
use crate::indicators::moving_averages::reflex::{ReflexBatchRange, ReflexParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::DeviceBuffer;
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::sys as cu;
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaReflexError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaReflexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaReflexError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaReflexError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}

impl std::error::Error for CudaReflexError {}

pub struct CudaReflex {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaReflex {
    pub fn new(device_id: usize) -> Result<Self, CudaReflexError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaReflexError::Cuda(e.to_string()))?;

        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaReflexError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaReflexError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/reflex_kernel.ptx"));
        let module =
            Module::from_ptx(ptx, &[]).map_err(|e| CudaReflexError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaReflexError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
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

    pub fn reflex_batch_dev(
        &self,
        prices: &[f32],
        sweep: &ReflexBatchRange,
    ) -> Result<DeviceArrayF32, CudaReflexError> {
        let inputs = Self::prepare_batch_inputs(prices, sweep)?;
        self.run_batch_kernel(prices, &inputs)
    }

    pub fn reflex_batch_into_host_f32(
        &self,
        prices: &[f32],
        sweep: &ReflexBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<ReflexParams>), CudaReflexError> {
        let inputs = Self::prepare_batch_inputs(prices, sweep)?;
        let expected = inputs.series_len * inputs.combos.len();
        if out.len() != expected {
            return Err(CudaReflexError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                expected
            )));
        }

        let arr = self.run_batch_kernel(prices, &inputs)?;
        arr.buf
            .copy_to(out)
            .map_err(|e| CudaReflexError::Cuda(e.to_string()))?;
        Ok((arr.rows, arr.cols, inputs.combos))
    }

    pub fn reflex_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaReflexError> {
        if series_len == 0 || n_combos == 0 {
            return Err(CudaReflexError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }
        if series_len > i32::MAX as usize || n_combos > i32::MAX as usize {
            return Err(CudaReflexError::InvalidInput(
                "arguments exceed kernel limits".into(),
            ));
        }
        if max_period < 2 {
            return Err(CudaReflexError::InvalidInput(
                "max_period must be >= 2".into(),
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

    pub fn reflex_many_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        period: usize,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaReflexError> {
        if period < 2 || num_series == 0 || series_len == 0 {
            return Err(CudaReflexError::InvalidInput(
                "period >= 2 and positive dimensions required".into(),
            ));
        }
        if period > i32::MAX as usize
            || num_series > i32::MAX as usize
            || series_len > i32::MAX as usize
        {
            return Err(CudaReflexError::InvalidInput(
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

    pub fn reflex_many_series_one_param_time_major_dev(
        &self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaReflexError> {
        let prepared = Self::prepare_many_series_inputs(prices_tm_f32, cols, rows, period)?;
        self.run_many_series_kernel(prices_tm_f32, cols, rows, period, &prepared)
    }

    pub fn reflex_many_series_one_param_time_major_into_host_f32(
        &self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
        out_tm: &mut [f32],
    ) -> Result<(), CudaReflexError> {
        if out_tm.len() != cols * rows {
            return Err(CudaReflexError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out_tm.len(),
                cols * rows
            )));
        }

        let prepared = Self::prepare_many_series_inputs(prices_tm_f32, cols, rows, period)?;
        let arr = self.run_many_series_kernel(prices_tm_f32, cols, rows, period, &prepared)?;
        arr.buf
            .copy_to(out_tm)
            .map_err(|e| CudaReflexError::Cuda(e.to_string()))?;
        Ok(())
    }

    fn run_batch_kernel(
        &self,
        prices: &[f32],
        inputs: &BatchInputs,
    ) -> Result<DeviceArrayF32, CudaReflexError> {
        let n_combos = inputs.combos.len();
        let series_len = inputs.series_len;

        let prices_bytes = series_len * std::mem::size_of::<f32>();
        let periods_bytes = n_combos * std::mem::size_of::<i32>();
        let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + periods_bytes + out_bytes;
        let headroom = 32 * 1024 * 1024; // 32MB safety margin

        if !Self::will_fit(required, headroom) {
            return Err(CudaReflexError::InvalidInput(
                "insufficient device memory for Reflex batch launch".into(),
            ));
        }

        let d_prices =
            DeviceBuffer::from_slice(prices).map_err(|e| CudaReflexError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&inputs.periods)
            .map_err(|e| CudaReflexError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(series_len * n_combos) }
                .map_err(|e| CudaReflexError::Cuda(e.to_string()))?;

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
            .map_err(|e| CudaReflexError::Cuda(e.to_string()))?;

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
    ) -> Result<DeviceArrayF32, CudaReflexError> {
        let prices_bytes = prices_tm_f32.len() * std::mem::size_of::<f32>();
        let first_valid_bytes = prepared.first_valids.len() * std::mem::size_of::<i32>();
        let out_bytes = prices_tm_f32.len() * std::mem::size_of::<f32>();
        let required = prices_bytes + first_valid_bytes + out_bytes;
        let headroom = 32 * 1024 * 1024; // 32MB safety margin

        if !Self::will_fit(required, headroom) {
            return Err(CudaReflexError::InvalidInput(
                "insufficient device memory for Reflex many-series launch".into(),
            ));
        }

        let d_prices_tm = DeviceBuffer::from_slice(prices_tm_f32)
            .map_err(|e| CudaReflexError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&prepared.first_valids)
            .map_err(|e| CudaReflexError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(prices_tm_f32.len()) }
                .map_err(|e| CudaReflexError::Cuda(e.to_string()))?;

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
            .map_err(|e| CudaReflexError::Cuda(e.to_string()))?;

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
    ) -> Result<(), CudaReflexError> {
        let func = self
            .module
            .get_function("reflex_batch_f32")
            .map_err(|e| CudaReflexError::Cuda(e.to_string()))?;

        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (1, 1, 1).into();
        let shared_bytes = (max_period * std::mem::size_of::<f32>()) as u32;

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
                .map_err(|e| CudaReflexError::Cuda(e.to_string()))?
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
    ) -> Result<(), CudaReflexError> {
        let func = self
            .module
            .get_function("reflex_many_series_one_param_f32")
            .map_err(|e| CudaReflexError::Cuda(e.to_string()))?;

        let grid: GridSize = (num_series as u32, 1, 1).into();
        let block: BlockSize = (1, 1, 1).into();
        let shared_bytes = (period * std::mem::size_of::<f32>()) as u32;

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
                .map_err(|e| CudaReflexError::Cuda(e.to_string()))?
        }
        Ok(())
    }

    fn prepare_batch_inputs(
        prices: &[f32],
        sweep: &ReflexBatchRange,
    ) -> Result<BatchInputs, CudaReflexError> {
        if prices.is_empty() {
            return Err(CudaReflexError::InvalidInput("empty prices".into()));
        }

        let combos = expand_grid_reflex(sweep);
        if combos.is_empty() {
            return Err(CudaReflexError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let first_valid = prices
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaReflexError::InvalidInput("all values are NaN".into()))?;

        let series_len = prices.len();
        let mut periods = Vec::with_capacity(combos.len());
        let mut max_period = 0usize;
        for params in &combos {
            let period = params.period.unwrap_or(0);
            if period < 2 {
                return Err(CudaReflexError::InvalidInput("period must be >= 2".into()));
            }
            if period > i32::MAX as usize {
                return Err(CudaReflexError::InvalidInput(
                    "period exceeds i32 kernel limit".into(),
                ));
            }
            periods.push(period as i32);
            max_period = max_period.max(period);
        }

        if series_len - first_valid < max_period {
            return Err(CudaReflexError::InvalidInput(format!(
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
    ) -> Result<ManySeriesInputs, CudaReflexError> {
        if cols == 0 || rows == 0 {
            return Err(CudaReflexError::InvalidInput(
                "matrix dimensions must be positive".into(),
            ));
        }
        if prices_tm_f32.len() != cols * rows {
            return Err(CudaReflexError::InvalidInput(
                "matrix shape mismatch".into(),
            ));
        }
        if period < 2 {
            return Err(CudaReflexError::InvalidInput("period must be >= 2".into()));
        }
        if period > i32::MAX as usize {
            return Err(CudaReflexError::InvalidInput(
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
                CudaReflexError::InvalidInput(format!("series {} has all NaN values", series_idx))
            })?;
            if rows - first < period {
                return Err(CudaReflexError::InvalidInput(format!(
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

fn expand_grid_reflex(range: &ReflexBatchRange) -> Vec<ReflexParams> {
    let (start, end, step) = range.period;
    if step == 0 || start == end {
        return vec![ReflexParams {
            period: Some(start),
        }];
    }
    (start..=end)
        .step_by(step)
        .map(|p| ReflexParams { period: Some(p) })
        .collect()
}

struct BatchInputs {
    combos: Vec<ReflexParams>,
    periods: Vec<i32>,
    first_valid: usize,
    series_len: usize,
    max_period: usize,
}

struct ManySeriesInputs {
    first_valids: Vec<i32>,
}
