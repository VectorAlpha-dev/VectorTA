//! CUDA support for the Wilder's Moving Average (Wilders) indicator.
//!
//! Mirrors the CPU batching API by accepting a single series alongside a sweep
//! of periods. Kernels run entirely in FP32 and reuse precomputed alpha values
//! and warm-up indices to keep the GPU work focused on the recurrence itself.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::wilders::{WildersBatchRange, WildersParams};
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
pub enum CudaWildersError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaWildersError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaWildersError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaWildersError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}

impl std::error::Error for CudaWildersError {}

pub struct CudaWilders {
    module: Module,
    stream: Stream,
    _context: Context,
}

struct PreparedWildersBatch {
    combos: Vec<WildersParams>,
    first_valid: usize,
    series_len: usize,
    periods_i32: Vec<i32>,
    alphas_f32: Vec<f32>,
    warm_indices: Vec<i32>,
}

impl CudaWilders {
    pub fn new(device_id: usize) -> Result<Self, CudaWildersError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaWildersError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaWildersError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaWildersError::Cuda(e.to_string()))?;

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/wilders_kernel.ptx"));
        let module =
            Module::from_ptx(ptx, &[]).map_err(|e| CudaWildersError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaWildersError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    pub fn wilders_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &WildersBatchRange,
    ) -> Result<DeviceArrayF32, CudaWildersError> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = prepared.combos.len();

        let d_prices = DeviceBuffer::from_slice(data_f32)
            .map_err(|e| CudaWildersError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&prepared.periods_i32)
            .map_err(|e| CudaWildersError::Cuda(e.to_string()))?;
        let d_alphas = DeviceBuffer::from_slice(&prepared.alphas_f32)
            .map_err(|e| CudaWildersError::Cuda(e.to_string()))?;
        let d_warm = DeviceBuffer::from_slice(&prepared.warm_indices)
            .map_err(|e| CudaWildersError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(prepared.series_len * n_combos)
                .map_err(|e| CudaWildersError::Cuda(e.to_string()))?
        };

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            &d_alphas,
            &d_warm,
            prepared.series_len,
            prepared.first_valid,
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
    pub fn wilders_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_alphas: &DeviceBuffer<f32>,
        d_warm: &DeviceBuffer<i32>,
        series_len: i32,
        first_valid: i32,
        n_combos: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWildersError> {
        if series_len <= 0 {
            return Err(CudaWildersError::InvalidInput(
                "series_len must be positive".into(),
            ));
        }
        if first_valid < 0 || first_valid >= series_len {
            return Err(CudaWildersError::InvalidInput(format!(
                "first_valid out of range: {} (len {})",
                first_valid, series_len
            )));
        }
        if n_combos <= 0 {
            return Err(CudaWildersError::InvalidInput(
                "n_combos must be positive".into(),
            ));
        }
        let expected = n_combos as usize;
        if d_periods.len() != expected || d_alphas.len() != expected || d_warm.len() != expected {
            return Err(CudaWildersError::InvalidInput(
                "device buffer length mismatch".into(),
            ));
        }

        self.launch_batch_kernel(
            d_prices,
            d_periods,
            d_alphas,
            d_warm,
            series_len as usize,
            first_valid as usize,
            expected,
            d_out,
        )
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &WildersBatchRange,
    ) -> Result<PreparedWildersBatch, CudaWildersError> {
        if data_f32.is_empty() {
            return Err(CudaWildersError::InvalidInput("input data is empty".into()));
        }
        let combos = expand_periods(sweep);
        if combos.is_empty() {
            return Err(CudaWildersError::InvalidInput(
                "no period combinations provided".into(),
            ));
        }

        let series_len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaWildersError::InvalidInput("all values are NaN".into()))?;

        let mut periods_i32 = Vec::with_capacity(combos.len());
        let mut alphas_f32 = Vec::with_capacity(combos.len());
        let mut warm_indices = Vec::with_capacity(combos.len());

        for params in &combos {
            let period = params.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaWildersError::InvalidInput(
                    "period must be positive".into(),
                ));
            }
            if series_len - first_valid < period {
                return Err(CudaWildersError::InvalidInput(format!(
                    "not enough valid data: needed >= {}, have {}",
                    period,
                    series_len - first_valid
                )));
            }
            for idx in 0..period {
                let sample = data_f32[first_valid + idx];
                if !sample.is_finite() {
                    return Err(CudaWildersError::InvalidInput(format!(
                        "non-finite value in warm window at offset {}",
                        idx
                    )));
                }
            }
            periods_i32.push(period as i32);
            alphas_f32.push(1.0f32 / (period as f32));
            warm_indices.push((first_valid + period - 1) as i32);
        }

        Ok(PreparedWildersBatch {
            combos,
            first_valid,
            series_len,
            periods_i32,
            alphas_f32,
            warm_indices,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_alphas: &DeviceBuffer<f32>,
        d_warm: &DeviceBuffer<i32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWildersError> {
        if n_combos == 0 {
            return Ok(());
        }

        let func = self
            .module
            .get_function("wilders_batch_f32")
            .map_err(|e| CudaWildersError::Cuda(e.to_string()))?;

        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (256, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut alphas_ptr = d_alphas.as_device_ptr().as_raw();
            let mut warm_ptr = d_warm.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut first_valid_i = first_valid as i32;
            let mut n_combos_i = n_combos as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut alphas_ptr as *mut _ as *mut c_void,
                &mut warm_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut n_combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaWildersError::Cuda(e.to_string()))?;
        }

        Ok(())
    }
}

fn expand_periods(range: &WildersBatchRange) -> Vec<WildersParams> {
    let (start, end, step) = range.period;
    if step == 0 || start == end {
        return vec![WildersParams {
            period: Some(start),
        }];
    }

    let mut out = Vec::new();
    let mut value = start;
    while value <= end {
        out.push(WildersParams {
            period: Some(value),
        });
        value = value.saturating_add(step);
    }
    out
}
