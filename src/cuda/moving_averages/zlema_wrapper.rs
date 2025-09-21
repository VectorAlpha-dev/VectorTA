//! CUDA scaffolding for ZLEMA (Zero Lag Exponential Moving Average).
//!
//! Mirrors the VRAM-first design used by the ALMA integration: inputs are
//! converted to `f32`, parameter sweeps are validated on the host, and each
//! parameter combination is evaluated on the device, returning a `DeviceArrayF32`.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::zlema::{expand_grid_zlema, ZlemaBatchRange, ZlemaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::DeviceBuffer;
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaZlemaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaZlemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaZlemaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaZlemaError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaZlemaError {}

pub struct CudaZlema {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaZlema {
    pub fn new(device_id: usize) -> Result<Self, CudaZlemaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaZlemaError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaZlemaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaZlemaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/zlema_kernel.ptx"));
        let module = Module::from_ptx(ptx, &[]).map_err(|e| CudaZlemaError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaZlemaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &ZlemaBatchRange,
    ) -> Result<(Vec<ZlemaParams>, usize, usize), CudaZlemaError> {
        if data_f32.is_empty() {
            return Err(CudaZlemaError::InvalidInput("empty data".into()));
        }

        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaZlemaError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid_zlema(sweep);
        if combos.is_empty() {
            return Err(CudaZlemaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let mut max_period = 0usize;
        for combo in &combos {
            let period = combo.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaZlemaError::InvalidInput(
                    "period must be at least 1 in CUDA path".into(),
                ));
            }
            if period > len {
                return Err(CudaZlemaError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            max_period = max_period.max(period);
        }

        if len - first_valid < max_period {
            return Err(CudaZlemaError::InvalidInput(format!(
                "not enough valid data (need >= {}, have {} after first valid)",
                max_period,
                len - first_valid
            )));
        }

        Ok((combos, first_valid, len))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_lags: &DeviceBuffer<i32>,
        d_alphas: &DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaZlemaError> {
        let func = self
            .module
            .get_function("zlema_batch_f32")
            .map_err(|e| CudaZlemaError::Cuda(e.to_string()))?;

        let block_x: u32 = 128;
        let grid_x = ((n_combos as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut lags_ptr = d_lags.as_device_ptr().as_raw();
            let mut alphas_ptr = d_alphas.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut lags_ptr as *mut _ as *mut c_void,
                &mut alphas_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaZlemaError::Cuda(e.to_string()))?;
        }

        Ok(())
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        combos: &[ZlemaParams],
        first_valid: usize,
        len: usize,
    ) -> Result<DeviceArrayF32, CudaZlemaError> {
        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaZlemaError::Cuda(e.to_string()))?;

        let periods: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let lags: Vec<i32> = combos
            .iter()
            .map(|c| ((c.period.unwrap() - 1) / 2) as i32)
            .collect();
        let alphas: Vec<f32> = combos
            .iter()
            .map(|c| 2.0f32 / (c.period.unwrap() as f32 + 1.0f32))
            .collect();

        let d_periods =
            DeviceBuffer::from_slice(&periods).map_err(|e| CudaZlemaError::Cuda(e.to_string()))?;
        let d_lags =
            DeviceBuffer::from_slice(&lags).map_err(|e| CudaZlemaError::Cuda(e.to_string()))?;
        let d_alphas =
            DeviceBuffer::from_slice(&alphas).map_err(|e| CudaZlemaError::Cuda(e.to_string()))?;

        let elems = combos.len() * len;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }
            .map_err(|e| CudaZlemaError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            &d_lags,
            &d_alphas,
            len,
            first_valid,
            combos.len(),
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: combos.len(),
            cols: len,
        })
    }

    pub fn zlema_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &ZlemaBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<ZlemaParams>), CudaZlemaError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let dev = self.run_batch_kernel(data_f32, &combos, first_valid, len)?;
        Ok((dev, combos))
    }

    pub fn zlema_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &ZlemaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<ZlemaParams>), CudaZlemaError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos.len() * len;
        if out.len() != expected {
            return Err(CudaZlemaError::InvalidInput(format!(
                "output slice length mismatch: expected {}, got {}",
                expected,
                out.len()
            )));
        }
        let dev = self.run_batch_kernel(data_f32, &combos, first_valid, len)?;
        dev.buf
            .copy_to(out)
            .map_err(|e| CudaZlemaError::Cuda(e.to_string()))?;
        Ok((combos.len(), len, combos))
    }
}

// ---------- Bench profiles (batch-only) ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches_batch_only;

    define_ma_period_benches_batch_only!(
        zlema_benches,
        CudaZlema,
        crate::indicators::moving_averages::zlema::ZlemaBatchRange,
        zlema_batch_dev,
        crate::indicators::moving_averages::zlema::ZlemaBatchRange { period: (10, 10 + PARAM_SWEEP - 1, 1) },
        "zlema",
        "zlema"
    );
    pub use zlema_benches::bench_profiles;
}
