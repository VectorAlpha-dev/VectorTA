//! CUDA scaffolding for the Ehlers Predictive Moving Average (PMA).
//!
//! The kernels operate purely in FP32 memory with FP64 intermediates to mirror
//! the scalar implementation. This wrapper mirrors the ALMA GPU surface: a
//! "one series × many params" batch launcher (where `combos` is a synthetic
//! sweep count) and a "many series × one param" launch that consumes
//! time-major data without extra host copies.

#![cfg(feature = "cuda")]

use super::DeviceArrayF32;
use crate::indicators::moving_averages::ehlers_pma::{
    expand_grid, EhlersPmaBatchRange, EhlersPmaParams,
};
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
pub enum CudaEhlersPmaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaEhlersPmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaEhlersPmaError::Cuda(e) => write!(f, "CUDA error: {e}"),
            CudaEhlersPmaError::InvalidInput(e) => write!(f, "Invalid input: {e}"),
        }
    }
}

impl std::error::Error for CudaEhlersPmaError {}

/// VRAM-backed pair of outputs (predict + trigger).
pub struct DeviceEhlersPmaPair {
    pub predict: DeviceArrayF32,
    pub trigger: DeviceArrayF32,
}

impl DeviceEhlersPmaPair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.predict.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.predict.cols
    }
}

pub struct CudaEhlersPma {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaEhlersPma {
    pub fn new(device_id: usize) -> Result<Self, CudaEhlersPmaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaEhlersPmaError::Cuda(e.to_string()))?;

        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaEhlersPmaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaEhlersPmaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/ehlers_pma_kernel.ptx"));
        let module =
            Module::from_ptx(ptx, &[]).map_err(|e| CudaEhlersPmaError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaEhlersPmaError::Cuda(e.to_string()))?;

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
            let mut free = 0usize;
            let mut total = 0usize;
            if cu::cuMemGetInfo_v2(&mut free as *mut usize, &mut total as *mut usize)
                == cu::CUresult::CUDA_SUCCESS
            {
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

    fn prepare_batch_inputs(
        prices: &[f32],
        sweep: &EhlersPmaBatchRange,
    ) -> Result<(Vec<EhlersPmaParams>, usize, usize), CudaEhlersPmaError> {
        if prices.is_empty() {
            return Err(CudaEhlersPmaError::InvalidInput(
                "empty price series".into(),
            ));
        }

        let first_valid = prices
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaEhlersPmaError::InvalidInput("all values are NaN".into()))?;

        const MIN_REQUIRED: usize = 14;
        if prices.len() - first_valid < MIN_REQUIRED {
            return Err(CudaEhlersPmaError::InvalidInput(format!(
                "not enough valid data (needed >= {MIN_REQUIRED}, valid = {})",
                prices.len() - first_valid
            )));
        }

        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaEhlersPmaError::InvalidInput(
                "no parameter combinations for Ehlers PMA".into(),
            ));
        }

        Ok((combos, first_valid, prices.len()))
    }

    fn run_batch_kernel(
        &self,
        prices: &[f32],
        combos: &[EhlersPmaParams],
        first_valid: usize,
        series_len: usize,
    ) -> Result<DeviceEhlersPmaPair, CudaEhlersPmaError> {
        let n_combos = combos.len();
        let prices_bytes = series_len * std::mem::size_of::<f32>();
        let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + 2 * out_bytes;
        let headroom = 32 * 1024 * 1024; // 32 MB safety margin
        if !Self::will_fit(required, headroom) {
            return Err(CudaEhlersPmaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices = DeviceBuffer::from_slice(prices)
            .map_err(|e| CudaEhlersPmaError::Cuda(e.to_string()))?;
        let mut d_predict: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(n_combos * series_len)
                .map_err(|e| CudaEhlersPmaError::Cuda(e.to_string()))?
        };
        let mut d_trigger: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(n_combos * series_len)
                .map_err(|e| CudaEhlersPmaError::Cuda(e.to_string()))?
        };

        self.launch_batch_kernel(
            &d_prices,
            series_len,
            n_combos,
            first_valid,
            &mut d_predict,
            &mut d_trigger,
        )?;

        Ok(DeviceEhlersPmaPair {
            predict: DeviceArrayF32 {
                buf: d_predict,
                rows: n_combos,
                cols: series_len,
            },
            trigger: DeviceArrayF32 {
                buf: d_trigger,
                rows: n_combos,
                cols: series_len,
            },
        })
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_predict: &mut DeviceBuffer<f32>,
        d_trigger: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhlersPmaError> {
        let func = self
            .module
            .get_function("ehlers_pma_batch_f32")
            .map_err(|e| CudaEhlersPmaError::Cuda(e.to_string()))?;

        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (1, 1, 1).into();
        let shared = 0u32;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut predict_ptr = d_predict.as_device_ptr().as_raw();
            let mut trigger_ptr = d_trigger.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 6] = [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut predict_ptr as *mut _ as *mut c_void,
                &mut trigger_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared, &mut args)
                .map_err(|e| CudaEhlersPmaError::Cuda(e.to_string()))?
        }
        Ok(())
    }

    fn prepare_many_series_inputs(
        prices_tm: &[f32],
        cols: usize,
        rows: usize,
    ) -> Result<Vec<i32>, CudaEhlersPmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaEhlersPmaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if prices_tm.len() != cols * rows {
            return Err(CudaEhlersPmaError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                prices_tm.len(),
                cols * rows
            )));
        }

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut fv = None;
            for row in 0..rows {
                let idx = row * cols + series;
                let val = prices_tm[idx];
                if !val.is_nan() {
                    fv = Some(row);
                    break;
                }
            }
            let fv_idx = fv.ok_or_else(|| {
                CudaEhlersPmaError::InvalidInput(format!("series {} is entirely NaN", series))
            })?;
            if rows - fv_idx < 14 {
                return Err(CudaEhlersPmaError::InvalidInput(format!(
                    "series {} lacks warmup samples (valid = {})",
                    series,
                    rows - fv_idx
                )));
            }
            first_valids[series] = fv_idx as i32;
        }
        Ok(first_valids)
    }

    fn run_many_series_kernel(
        &self,
        prices_tm: &[f32],
        cols: usize,
        rows: usize,
        first_valids: &[i32],
    ) -> Result<DeviceEhlersPmaPair, CudaEhlersPmaError> {
        let prices_bytes = cols * rows * std::mem::size_of::<f32>();
        let first_valid_bytes = cols * std::mem::size_of::<i32>();
        let out_bytes = cols * rows * std::mem::size_of::<f32>();
        let required = prices_bytes + first_valid_bytes + 2 * out_bytes;
        let headroom = 32 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaEhlersPmaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices_tm = DeviceBuffer::from_slice(prices_tm)
            .map_err(|e| CudaEhlersPmaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(first_valids)
            .map_err(|e| CudaEhlersPmaError::Cuda(e.to_string()))?;
        let mut d_predict_tm: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(cols * rows)
                .map_err(|e| CudaEhlersPmaError::Cuda(e.to_string()))?
        };
        let mut d_trigger_tm: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(cols * rows)
                .map_err(|e| CudaEhlersPmaError::Cuda(e.to_string()))?
        };

        self.launch_many_series_kernel(
            &d_prices_tm,
            cols,
            rows,
            &d_first_valids,
            &mut d_predict_tm,
            &mut d_trigger_tm,
        )?;

        Ok(DeviceEhlersPmaPair {
            predict: DeviceArrayF32 {
                buf: d_predict_tm,
                rows,
                cols,
            },
            trigger: DeviceArrayF32 {
                buf: d_trigger_tm,
                rows,
                cols,
            },
        })
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_predict_tm: &mut DeviceBuffer<f32>,
        d_trigger_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhlersPmaError> {
        let func = self
            .module
            .get_function("ehlers_pma_many_series_one_param_f32")
            .map_err(|e| CudaEhlersPmaError::Cuda(e.to_string()))?;

        let grid: GridSize = (cols as u32, 1, 1).into();
        let block: BlockSize = (1, 1, 1).into();
        let shared = 0u32;

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut predict_ptr = d_predict_tm.as_device_ptr().as_raw();
            let mut trigger_ptr = d_trigger_tm.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 6] = [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_valids_ptr as *mut _ as *mut c_void,
                &mut predict_ptr as *mut _ as *mut c_void,
                &mut trigger_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared, &mut args)
                .map_err(|e| CudaEhlersPmaError::Cuda(e.to_string()))?
        }
        Ok(())
    }

    /// Launch the batch (one series × many combos) path and return VRAM handles.
    pub fn ehlers_pma_batch_dev(
        &self,
        prices: &[f32],
        sweep: &EhlersPmaBatchRange,
    ) -> Result<DeviceEhlersPmaPair, CudaEhlersPmaError> {
        let (combos, first_valid, series_len) = Self::prepare_batch_inputs(prices, sweep)?;
        self.run_batch_kernel(prices, &combos, first_valid, series_len)
    }

    /// Copy the batch result back to host memory (FP32) while returning metadata.
    pub fn ehlers_pma_batch_into_host_f32(
        &self,
        prices: &[f32],
        sweep: &EhlersPmaBatchRange,
        out_predict: &mut [f32],
        out_trigger: &mut [f32],
    ) -> Result<(usize, usize, Vec<EhlersPmaParams>), CudaEhlersPmaError> {
        let (combos, first_valid, series_len) = Self::prepare_batch_inputs(prices, sweep)?;
        let expected = combos.len() * series_len;
        if out_predict.len() != expected || out_trigger.len() != expected {
            return Err(CudaEhlersPmaError::InvalidInput(format!(
                "output slice wrong length: got predict={}, trigger={}, expected={}",
                out_predict.len(),
                out_trigger.len(),
                expected
            )));
        }

        let pair = self.run_batch_kernel(prices, &combos, first_valid, series_len)?;
        pair.predict
            .buf
            .copy_to(out_predict)
            .map_err(|e| CudaEhlersPmaError::Cuda(e.to_string()))?;
        pair.trigger
            .buf
            .copy_to(out_trigger)
            .map_err(|e| CudaEhlersPmaError::Cuda(e.to_string()))?;
        Ok((pair.rows(), pair.cols(), combos))
    }

    /// Batch launch using pre-allocated device buffers (benchmark helper).
    #[allow(clippy::too_many_arguments)]
    pub fn ehlers_pma_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_predict: &mut DeviceBuffer<f32>,
        d_trigger: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhlersPmaError> {
        if series_len == 0 || n_combos == 0 {
            return Err(CudaEhlersPmaError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }
        if series_len > i32::MAX as usize || n_combos > i32::MAX as usize {
            return Err(CudaEhlersPmaError::InvalidInput(
                "arguments exceed kernel limits".into(),
            ));
        }
        self.launch_batch_kernel(
            d_prices,
            series_len,
            n_combos,
            first_valid,
            d_predict,
            d_trigger,
        )
    }

    /// Many-series launch returning VRAM handles (time-major input/output).
    pub fn ehlers_pma_many_series_one_param_time_major_dev(
        &self,
        prices_tm: &[f32],
        cols: usize,
        rows: usize,
    ) -> Result<DeviceEhlersPmaPair, CudaEhlersPmaError> {
        let first_valids = Self::prepare_many_series_inputs(prices_tm, cols, rows)?;
        self.run_many_series_kernel(prices_tm, cols, rows, &first_valids)
    }

    /// Copy many-series outputs back to host memory (time-major layout).
    pub fn ehlers_pma_many_series_one_param_time_major_into_host_f32(
        &self,
        prices_tm: &[f32],
        cols: usize,
        rows: usize,
        out_predict_tm: &mut [f32],
        out_trigger_tm: &mut [f32],
    ) -> Result<(), CudaEhlersPmaError> {
        if out_predict_tm.len() != cols * rows || out_trigger_tm.len() != cols * rows {
            return Err(CudaEhlersPmaError::InvalidInput(format!(
                "output slice wrong length: predict={}, trigger={}, expected={}",
                out_predict_tm.len(),
                out_trigger_tm.len(),
                cols * rows
            )));
        }
        let first_valids = Self::prepare_many_series_inputs(prices_tm, cols, rows)?;
        let pair = self.run_many_series_kernel(prices_tm, cols, rows, &first_valids)?;
        pair.predict
            .buf
            .copy_to(out_predict_tm)
            .map_err(|e| CudaEhlersPmaError::Cuda(e.to_string()))?;
        pair.trigger
            .buf
            .copy_to(out_trigger_tm)
            .map_err(|e| CudaEhlersPmaError::Cuda(e.to_string()))?;
        Ok(())
    }

    /// Many-series launch using pre-allocated device buffers (benchmark helper).
    #[allow(clippy::too_many_arguments)]
    pub fn ehlers_pma_many_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_predict_tm: &mut DeviceBuffer<f32>,
        d_trigger_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhlersPmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaEhlersPmaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if cols > i32::MAX as usize || rows > i32::MAX as usize {
            return Err(CudaEhlersPmaError::InvalidInput(
                "arguments exceed kernel limits".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_prices_tm,
            cols,
            rows,
            d_first_valids,
            d_predict_tm,
            d_trigger_tm,
        )
    }
}
