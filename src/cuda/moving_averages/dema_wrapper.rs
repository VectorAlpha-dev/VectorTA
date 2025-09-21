//! CUDA support for the DEMA (Double Exponential Moving Average) indicator.
//!
//! This module mirrors the host-side ergonomics of `CudaAlma` but with the
//! simplified parameter space of DEMA (period only). Kernels operate in FP32 and
//! respect the same warm-up semantics as the scalar Rust implementation.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::dema::{DemaBatchRange, DemaParams};
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
pub enum CudaDemaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaDemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaDemaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaDemaError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}

impl std::error::Error for CudaDemaError {}

pub struct CudaDema {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaDema {
    pub fn new(device_id: usize) -> Result<Self, CudaDemaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaDemaError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaDemaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaDemaError::Cuda(e.to_string()))?;

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/dema_kernel.ptx"));
        let module = Module::from_ptx(ptx, &[]).map_err(|e| CudaDemaError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaDemaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    pub fn dema_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &DemaBatchRange,
    ) -> Result<DeviceArrayF32, CudaDemaError> {
        let (combos, first_valid, series_len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let periods: Vec<i32> = combos
            .iter()
            .map(|p| p.period.unwrap_or(0) as i32)
            .collect();

        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaDemaError::Cuda(e.to_string()))?;
        let d_periods =
            DeviceBuffer::from_slice(&periods).map_err(|e| CudaDemaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(series_len * combos.len())
                .map_err(|e| CudaDemaError::Cuda(e.to_string()))?
        };

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            series_len,
            first_valid,
            combos.len(),
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: combos.len(),
            cols: series_len,
        })
    }

    pub fn dema_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: i32,
        first_valid: i32,
        n_combos: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaDemaError> {
        if series_len <= 0 || n_combos <= 0 {
            return Err(CudaDemaError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }
        self.launch_batch_kernel(
            d_prices,
            d_periods,
            series_len as usize,
            first_valid.max(0) as usize,
            n_combos as usize,
            d_out,
        )
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &DemaBatchRange,
    ) -> Result<(Vec<DemaParams>, usize, usize), CudaDemaError> {
        if data_f32.is_empty() {
            return Err(CudaDemaError::InvalidInput("input data is empty".into()));
        }

        let combos = expand_periods(sweep);
        if combos.is_empty() {
            return Err(CudaDemaError::InvalidInput(
                "no period combinations provided".into(),
            ));
        }

        let series_len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaDemaError::InvalidInput("all values are NaN".into()))?;

        let max_period = combos
            .iter()
            .map(|p| p.period.unwrap_or(0))
            .max()
            .unwrap_or(0);
        if max_period == 0 {
            return Err(CudaDemaError::InvalidInput(
                "period must be positive".into(),
            ));
        }
        let needed = 2 * (max_period - 1);
        if series_len < needed {
            return Err(CudaDemaError::InvalidInput(format!(
                "not enough data: needed >= {}, have {}",
                needed, series_len
            )));
        }
        let valid = series_len - first_valid;
        if valid < needed {
            return Err(CudaDemaError::InvalidInput(format!(
                "not enough valid data: needed >= {}, have {}",
                needed, valid
            )));
        }

        Ok((combos, first_valid, series_len))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaDemaError> {
        if n_combos == 0 {
            return Ok(());
        }

        let func = self
            .module
            .get_function("dema_batch_f32")
            .map_err(|e| CudaDemaError::Cuda(e.to_string()))?;

        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (256, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut first_valid_i = first_valid as i32;
            let mut n_combos_i = n_combos as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut n_combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaDemaError::Cuda(e.to_string()))?;
        }

        Ok(())
    }
}

fn expand_periods(range: &DemaBatchRange) -> Vec<DemaParams> {
    let (start, end, step) = range.period;
    if step == 0 || start == end {
        return vec![DemaParams {
            period: Some(start),
        }];
    }

    let mut out = Vec::new();
    let mut value = start;
    while value <= end {
        out.push(DemaParams {
            period: Some(value),
        });
        value = value.saturating_add(step);
    }
    out
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches_batch_only;

    // DEMA currently exposes only the batch path; register batch-only benches.
    define_ma_period_benches_batch_only!(
        dema_benches,
        CudaDema,
        crate::indicators::moving_averages::dema::DemaBatchRange,
        dema_batch_dev,
        crate::indicators::moving_averages::dema::DemaBatchRange { period: (10, 10 + PARAM_SWEEP - 1, 1) },
        "dema",
        "dema"
    );
    pub use dema_benches::bench_profiles;
}
