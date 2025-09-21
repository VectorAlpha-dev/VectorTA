//! CUDA wrapper for MAAQ (Moving Average Adaptive Q) kernels.
//!
//! Mirrors the ALMA/SWMA/PWMA scaffolding: validate host inputs, upload FP32
//! data once, then launch kernels that keep the adaptive smoothing recursion on
//! the device. Supports both the single-series × many-parameter sweep and the
//! many-series × one-parameter path using time-major inputs.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::maaq::{expand_grid, MaaqBatchRange, MaaqParams};
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
pub enum CudaMaaqError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaMaaqError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaMaaqError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaMaaqError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaMaaqError {}

pub struct CudaMaaq {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaMaaq {
    pub fn new(device_id: usize) -> Result<Self, CudaMaaqError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/maaq_kernel.ptx"));
        let module = Module::from_ptx(ptx, &[]).map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &MaaqBatchRange,
    ) -> Result<(Vec<MaaqParams>, usize, usize, usize), CudaMaaqError> {
        if data_f32.is_empty() {
            return Err(CudaMaaqError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaMaaqError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaMaaqError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let len = data_f32.len();
        let mut max_period = 0usize;
        for prm in &combos {
            let period = prm.period.unwrap_or(0);
            let fast = prm.fast_period.unwrap_or(0);
            let slow = prm.slow_period.unwrap_or(0);
            if period == 0 || fast == 0 || slow == 0 {
                return Err(CudaMaaqError::InvalidInput(
                    "period, fast_period, and slow_period must be > 0".into(),
                ));
            }
            if period > len {
                return Err(CudaMaaqError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            if len - first_valid < period {
                return Err(CudaMaaqError::InvalidInput(format!(
                    "not enough valid data: need {}, have {}",
                    period,
                    len - first_valid
                )));
            }
            if period > i32::MAX as usize {
                return Err(CudaMaaqError::InvalidInput(
                    "period exceeds i32::MAX".into(),
                ));
            }
            if max_period < period {
                max_period = period;
            }
        }

        Ok((combos, first_valid, len, max_period))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_fast_scs: &DeviceBuffer<f32>,
        d_slow_scs: &DeviceBuffer<f32>,
        first_valid: usize,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMaaqError> {
        if series_len == 0 || n_combos == 0 || max_period == 0 {
            return Err(CudaMaaqError::InvalidInput(
                "series_len, n_combos, and max_period must be positive".into(),
            ));
        }
        if series_len > i32::MAX as usize
            || n_combos > i32::MAX as usize
            || max_period > i32::MAX as usize
        {
            return Err(CudaMaaqError::InvalidInput(
                "series_len, n_combos, or max_period exceed i32::MAX".into(),
            ));
        }
        if first_valid > i32::MAX as usize {
            return Err(CudaMaaqError::InvalidInput(
                "first_valid exceeds i32::MAX".into(),
            ));
        }

        let func = self
            .module
            .get_function("maaq_batch_f32")
            .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;

        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (1, 1, 1).into();
        let shared_bytes = (max_period * std::mem::size_of::<f32>()) as u32;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut fast_ptr = d_fast_scs.as_device_ptr().as_raw();
            let mut slow_ptr = d_slow_scs.as_device_ptr().as_raw();
            let mut first_valid_i = first_valid as i32;
            let mut series_len_i = series_len as i32;
            let mut n_combos_i = n_combos as i32;
            let mut max_period_i = max_period as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut fast_ptr as *mut _ as *mut c_void,
                &mut slow_ptr as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut n_combos_i as *mut _ as *mut c_void,
                &mut max_period_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes, args)
                .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;
        }

        Ok(())
    }

    pub fn maaq_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_fast_scs: &DeviceBuffer<f32>,
        d_slow_scs: &DeviceBuffer<f32>,
        first_valid: usize,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMaaqError> {
        self.launch_batch_kernel(
            d_prices,
            d_periods,
            d_fast_scs,
            d_slow_scs,
            first_valid,
            series_len,
            n_combos,
            max_period,
            d_out,
        )
    }

    pub fn maaq_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &MaaqBatchRange,
    ) -> Result<DeviceArrayF32, CudaMaaqError> {
        let (combos, first_valid, len, max_period) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = combos.len();

        let mut periods_i32 = Vec::with_capacity(n_combos);
        let mut fast_scs = Vec::with_capacity(n_combos);
        let mut slow_scs = Vec::with_capacity(n_combos);
        for prm in &combos {
            let period = prm.period.unwrap();
            let fast = prm.fast_period.unwrap();
            let slow = prm.slow_period.unwrap();
            periods_i32.push(period as i32);
            fast_scs.push(2.0f32 / (fast as f32 + 1.0f32));
            slow_scs.push(2.0f32 / (slow as f32 + 1.0f32));
        }

        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)
            .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;
        let d_fast =
            DeviceBuffer::from_slice(&fast_scs).map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;
        let d_slow =
            DeviceBuffer::from_slice(&slow_scs).map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n_combos * len) }
            .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            &d_fast,
            &d_slow,
            first_valid,
            len,
            n_combos,
            max_period,
            &mut d_out,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: len,
        })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &MaaqParams,
    ) -> Result<(Vec<i32>, usize, f32, f32), CudaMaaqError> {
        if cols == 0 || rows == 0 {
            return Err(CudaMaaqError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaMaaqError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }

        let period = params.period.unwrap_or(0);
        let fast = params.fast_period.unwrap_or(0);
        let slow = params.slow_period.unwrap_or(0);
        if period == 0 || fast == 0 || slow == 0 {
            return Err(CudaMaaqError::InvalidInput(
                "period, fast_period, and slow_period must be > 0".into(),
            ));
        }
        if period > rows {
            return Err(CudaMaaqError::InvalidInput(format!(
                "period {} exceeds series length {}",
                period, rows
            )));
        }
        if period > i32::MAX as usize {
            return Err(CudaMaaqError::InvalidInput(
                "period exceeds i32::MAX".into(),
            ));
        }

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut found = None;
            for row in 0..rows {
                let idx = row * cols + series;
                let v = data_tm_f32[idx];
                if !v.is_nan() {
                    found = Some(row);
                    break;
                }
            }
            let fv = found.ok_or_else(|| {
                CudaMaaqError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            if rows - fv < period {
                return Err(CudaMaaqError::InvalidInput(format!(
                    "series {} lacks enough valid data: need {} have {}",
                    series,
                    period,
                    rows - fv
                )));
            }
            if fv > i32::MAX as usize {
                return Err(CudaMaaqError::InvalidInput(
                    "first_valid exceeds i32::MAX".into(),
                ));
            }
            first_valids[series] = fv as i32;
        }

        let fast_sc = 2.0f32 / (fast as f32 + 1.0f32);
        let slow_sc = 2.0f32 / (slow as f32 + 1.0f32);

        Ok((first_valids, period, fast_sc, slow_sc))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        period: usize,
        fast_sc: f32,
        slow_sc: f32,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMaaqError> {
        if period == 0 || num_series == 0 || series_len == 0 {
            return Err(CudaMaaqError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        if period > i32::MAX as usize
            || num_series > i32::MAX as usize
            || series_len > i32::MAX as usize
        {
            return Err(CudaMaaqError::InvalidInput(
                "period, num_series, or series_len exceed i32::MAX".into(),
            ));
        }

        let func = self
            .module
            .get_function("maaq_multi_series_one_param_f32")
            .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;

        let grid: GridSize = (num_series as u32, 1, 1).into();
        let block: BlockSize = (1, 1, 1).into();
        let shared_bytes = (period * std::mem::size_of::<f32>()) as u32;

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut fast = fast_sc;
            let mut slow = slow_sc;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut fast as *mut _ as *mut c_void,
                &mut slow as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes, args)
                .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;
        }

        Ok(())
    }

    pub fn maaq_multi_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        period: i32,
        fast_sc: f32,
        slow_sc: f32,
        num_series: i32,
        series_len: i32,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMaaqError> {
        if period <= 0 || num_series <= 0 || series_len <= 0 {
            return Err(CudaMaaqError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_prices_tm,
            period as usize,
            fast_sc,
            slow_sc,
            num_series as usize,
            series_len as usize,
            d_first_valids,
            d_out_tm,
        )
    }

    pub fn maaq_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &MaaqParams,
    ) -> Result<DeviceArrayF32, CudaMaaqError> {
        let (first_valids, period, fast_sc, slow_sc) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            period,
            fast_sc,
            slow_sc,
            cols,
            rows,
            &d_first_valids,
            &mut d_out_tm,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows,
            cols,
        })
    }

    pub fn maaq_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &MaaqParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaMaaqError> {
        if out_tm.len() != cols * rows {
            return Err(CudaMaaqError::InvalidInput(format!(
                "output slice wrong length: got {} expected {}",
                out_tm.len(),
                cols * rows
            )));
        }
        let (first_valids, period, fast_sc, slow_sc) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            period,
            fast_sc,
            slow_sc,
            cols,
            rows,
            &d_first_valids,
            &mut d_out_tm,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;

        d_out_tm
            .copy_to(out_tm)
            .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;
        Ok(())
    }
}
