//! CUDA support for the Williams' %R (WILLR) indicator.
//!
//! Mirrors the CPU batching API by accepting a single price series with many
//! period combinations. Kernels operate in FP32 and replicate the scalar
//! semantics (warm-up NaNs, NaN propagation, zero denominator handling).

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::willr::{
    build_willr_gpu_tables, WillrBatchRange, WillrGpuTables, WillrParams,
};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::DeviceBuffer;
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaWillrError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaWillrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaWillrError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaWillrError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}

impl std::error::Error for CudaWillrError {}

pub struct CudaWillr {
    module: Module,
    stream: Stream,
    _context: Context,
}

struct PreparedWillrBatch {
    combos: Vec<WillrParams>,
    first_valid: usize,
    series_len: usize,
    tables: WillrGpuTables,
}

impl CudaWillr {
    pub fn new(device_id: usize) -> Result<Self, CudaWillrError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaWillrError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaWillrError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaWillrError::Cuda(e.to_string()))?;

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/willr_kernel.ptx"));
        let module = Module::from_ptx(ptx, &[]).map_err(|e| CudaWillrError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaWillrError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    pub fn willr_batch_dev(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        close_f32: &[f32],
        sweep: &WillrBatchRange,
    ) -> Result<DeviceArrayF32, CudaWillrError> {
        let prepared = Self::prepare_batch_inputs(high_f32, low_f32, close_f32, sweep)?;
        let n_combos = prepared.combos.len();
        let periods: Vec<i32> = prepared
            .combos
            .iter()
            .map(|p| p.period.unwrap_or(0) as i32)
            .collect();

        let d_close =
            DeviceBuffer::from_slice(close_f32).map_err(|e| CudaWillrError::Cuda(e.to_string()))?;
        let d_periods =
            DeviceBuffer::from_slice(&periods).map_err(|e| CudaWillrError::Cuda(e.to_string()))?;
        let d_log2 = DeviceBuffer::from_slice(&prepared.tables.log2)
            .map_err(|e| CudaWillrError::Cuda(e.to_string()))?;
        let d_offsets = DeviceBuffer::from_slice(&prepared.tables.level_offsets)
            .map_err(|e| CudaWillrError::Cuda(e.to_string()))?;
        let d_st_max = DeviceBuffer::from_slice(&prepared.tables.st_max)
            .map_err(|e| CudaWillrError::Cuda(e.to_string()))?;
        let d_st_min = DeviceBuffer::from_slice(&prepared.tables.st_min)
            .map_err(|e| CudaWillrError::Cuda(e.to_string()))?;
        let d_nan_psum = DeviceBuffer::from_slice(&prepared.tables.nan_psum)
            .map_err(|e| CudaWillrError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(prepared.series_len * n_combos)
                .map_err(|e| CudaWillrError::Cuda(e.to_string()))?
        };

        self.launch_batch_kernel(
            &d_close,
            &d_periods,
            &d_log2,
            &d_offsets,
            &d_st_max,
            &d_st_min,
            &d_nan_psum,
            prepared.series_len,
            prepared.first_valid,
            prepared.tables.level_offsets.len() - 1,
            n_combos,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: prepared.series_len,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn willr_batch_device(
        &self,
        d_close: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_log2: &DeviceBuffer<i32>,
        d_offsets: &DeviceBuffer<i32>,
        d_st_max: &DeviceBuffer<f32>,
        d_st_min: &DeviceBuffer<f32>,
        d_nan_psum: &DeviceBuffer<i32>,
        series_len: i32,
        first_valid: i32,
        n_combos: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWillrError> {
        if series_len <= 0 || n_combos <= 0 {
            return Err(CudaWillrError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }
        if first_valid < 0 || first_valid >= series_len {
            return Err(CudaWillrError::InvalidInput(format!(
                "first_valid out of range: {} (len {})",
                first_valid, series_len
            )));
        }

        let level_count = d_offsets
            .len()
            .checked_sub(1)
            .ok_or_else(|| CudaWillrError::InvalidInput("level offsets is empty".into()))?;

        self.launch_batch_kernel(
            d_close,
            d_periods,
            d_log2,
            d_offsets,
            d_st_max,
            d_st_min,
            d_nan_psum,
            series_len as usize,
            first_valid as usize,
            level_count,
            n_combos as usize,
            d_out,
        )
    }

    fn prepare_batch_inputs(
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &WillrBatchRange,
    ) -> Result<PreparedWillrBatch, CudaWillrError> {
        let len = high.len();
        if len == 0 || low.len() != len || close.len() != len {
            return Err(CudaWillrError::InvalidInput(
                "input slices are empty or mismatched".into(),
            ));
        }

        let combos = expand_periods(sweep);
        if combos.is_empty() {
            return Err(CudaWillrError::InvalidInput(
                "no period combinations".into(),
            ));
        }

        let first_valid = (0..len)
            .find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
            .ok_or_else(|| CudaWillrError::InvalidInput("all values are NaN".into()))?;

        let max_period = combos
            .iter()
            .map(|p| p.period.unwrap_or(0))
            .max()
            .unwrap_or(0);
        if max_period == 0 {
            return Err(CudaWillrError::InvalidInput(
                "period must be positive".into(),
            ));
        }

        let valid = len - first_valid;
        if valid < max_period {
            return Err(CudaWillrError::InvalidInput(format!(
                "not enough valid data: needed >= {}, have {}",
                max_period, valid
            )));
        }

        let tables = build_willr_gpu_tables(high, low);

        Ok(PreparedWillrBatch {
            combos,
            first_valid,
            series_len: len,
            tables,
        })
    }

    fn launch_batch_kernel(
        &self,
        d_close: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_log2: &DeviceBuffer<i32>,
        d_offsets: &DeviceBuffer<i32>,
        d_st_max: &DeviceBuffer<f32>,
        d_st_min: &DeviceBuffer<f32>,
        d_nan_psum: &DeviceBuffer<i32>,
        series_len: usize,
        first_valid: usize,
        level_count: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWillrError> {
        if n_combos == 0 {
            return Ok(());
        }

        let func = self
            .module
            .get_function("willr_batch_f32")
            .map_err(|e| CudaWillrError::Cuda(e.to_string()))?;

        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (256, 1, 1).into();

        unsafe {
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut log2_ptr = d_log2.as_device_ptr().as_raw();
            let mut offsets_ptr = d_offsets.as_device_ptr().as_raw();
            let mut st_max_ptr = d_st_max.as_device_ptr().as_raw();
            let mut st_min_ptr = d_st_min.as_device_ptr().as_raw();
            let mut nan_psum_ptr = d_nan_psum.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut first_valid_i = first_valid as i32;
            let mut level_count_i = level_count as i32;
            let mut n_combos_i = n_combos as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut close_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut log2_ptr as *mut _ as *mut c_void,
                &mut offsets_ptr as *mut _ as *mut c_void,
                &mut st_max_ptr as *mut _ as *mut c_void,
                &mut st_min_ptr as *mut _ as *mut c_void,
                &mut nan_psum_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut level_count_i as *mut _ as *mut c_void,
                &mut n_combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaWillrError::Cuda(e.to_string()))?;
        }

        Ok(())
    }
}

fn expand_periods(range: &WillrBatchRange) -> Vec<WillrParams> {
    let (start, end, step) = range.period;
    if step == 0 || start == end {
        return vec![WillrParams {
            period: Some(start),
        }];
    }
    let mut periods = Vec::new();
    let mut value = start;
    while value <= end {
        periods.push(WillrParams {
            period: Some(value),
        });
        value = value.saturating_add(step);
    }
    periods
}
