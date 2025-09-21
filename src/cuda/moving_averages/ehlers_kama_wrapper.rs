//! CUDA wrapper for Ehlers KAMA kernels.
//!
//! Mirrors the ALMA/SWMA scaffolding: validate host inputs, upload FP32 data
//! once, and launch kernels that keep the sequential adaptive update entirely
//! on the device. Provides both the single-series × many-parameter sweep and
//! the many-series × one-parameter time-major path so the CUDA API matches the
//! existing moving-average feature set.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::ehlers_kama::{EhlersKamaBatchRange, EhlersKamaParams};
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
pub enum CudaEhlersKamaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaEhlersKamaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaEhlersKamaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaEhlersKamaError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}

impl std::error::Error for CudaEhlersKamaError {}

pub struct CudaEhlersKama {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaEhlersKama {
    pub fn new(device_id: usize) -> Result<Self, CudaEhlersKamaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/ehlers_kama_kernel.ptx"));
        let module =
            Module::from_ptx(ptx, &[]).map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    fn expand_grid(range: &EhlersKamaBatchRange) -> Vec<EhlersKamaParams> {
        let (start, end, step) = range.period;
        if step == 0 || start == end {
            return vec![EhlersKamaParams {
                period: Some(start),
            }];
        }
        let mut params = Vec::new();
        let step_sz = if step == 0 { 1 } else { step };
        let mut value = start;
        loop {
            if value > end {
                break;
            }
            params.push(EhlersKamaParams {
                period: Some(value),
            });
            match value.checked_add(step_sz) {
                Some(next) => {
                    if next > end {
                        break;
                    }
                    value = next;
                }
                None => break,
            }
        }
        if params.is_empty() {
            params.push(EhlersKamaParams {
                period: Some(start),
            });
        }
        params
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &EhlersKamaBatchRange,
    ) -> Result<(Vec<EhlersKamaParams>, usize, usize), CudaEhlersKamaError> {
        if data_f32.is_empty() {
            return Err(CudaEhlersKamaError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaEhlersKamaError::InvalidInput("all values are NaN".into()))?;

        let combos = Self::expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaEhlersKamaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let len = data_f32.len();
        for prm in &combos {
            let period = prm.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaEhlersKamaError::InvalidInput(
                    "period must be greater than zero".into(),
                ));
            }
            if period > len {
                return Err(CudaEhlersKamaError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            if len - first_valid < period {
                return Err(CudaEhlersKamaError::InvalidInput(format!(
                    "not enough valid data: needed {}, have {}",
                    period,
                    len - first_valid
                )));
            }
        }

        Ok((combos, first_valid, len))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        first_valid: usize,
        series_len: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhlersKamaError> {
        if series_len == 0 || n_combos == 0 {
            return Err(CudaEhlersKamaError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }
        if series_len > i32::MAX as usize || n_combos > i32::MAX as usize {
            return Err(CudaEhlersKamaError::InvalidInput(
                "series_len or n_combos exceed i32::MAX".into(),
            ));
        }
        if first_valid > i32::MAX as usize {
            return Err(CudaEhlersKamaError::InvalidInput(
                "first_valid exceeds i32::MAX".into(),
            ));
        }

        let func = self
            .module
            .get_function("ehlers_kama_batch_f32")
            .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;

        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (1, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut first_valid_i = first_valid as i32;
            let mut series_len_i = series_len as i32;
            let mut n_combos_i = n_combos as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut n_combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
        }

        Ok(())
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        combos: &[EhlersKamaParams],
        first_valid: usize,
        series_len: usize,
    ) -> Result<DeviceArrayF32, CudaEhlersKamaError> {
        let n_combos = combos.len();
        let mut periods_i32 = Vec::with_capacity(n_combos);
        for prm in combos {
            let period = prm.period.unwrap();
            if period > i32::MAX as usize {
                return Err(CudaEhlersKamaError::InvalidInput(
                    "period exceeds i32::MAX".into(),
                ));
            }
            periods_i32.push(period as i32);
        }

        let d_prices = DeviceBuffer::from_slice(data_f32)
            .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)
            .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            first_valid,
            series_len,
            n_combos,
            &mut d_out,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    pub fn ehlers_kama_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        first_valid: usize,
        series_len: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhlersKamaError> {
        self.launch_batch_kernel(
            d_prices,
            d_periods,
            first_valid,
            series_len,
            n_combos,
            d_out,
        )
    }

    pub fn ehlers_kama_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &EhlersKamaBatchRange,
    ) -> Result<DeviceArrayF32, CudaEhlersKamaError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        self.run_batch_kernel(data_f32, &combos, first_valid, len)
    }

    pub fn ehlers_kama_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &EhlersKamaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<EhlersKamaParams>), CudaEhlersKamaError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos.len() * len;
        if out.len() != expected {
            return Err(CudaEhlersKamaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                expected
            )));
        }
        let arr = self.run_batch_kernel(data_f32, &combos, first_valid, len)?;
        arr.buf
            .copy_to(out)
            .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
        Ok((arr.rows, arr.cols, combos))
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &EhlersKamaParams,
    ) -> Result<(Vec<i32>, usize), CudaEhlersKamaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaEhlersKamaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaEhlersKamaError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }

        let period = params.period.unwrap_or(0);
        if period == 0 {
            return Err(CudaEhlersKamaError::InvalidInput(
                "period must be greater than zero".into(),
            ));
        }

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + series];
                if !v.is_nan() {
                    fv = Some(t);
                    break;
                }
            }
            let fv = fv.ok_or_else(|| {
                CudaEhlersKamaError::InvalidInput(format!("series {} all NaN", series))
            })?;
            if rows - fv < period {
                return Err(CudaEhlersKamaError::InvalidInput(format!(
                    "series {} not enough valid data: needed {}, have {}",
                    series,
                    period,
                    rows - fv
                )));
            }
            if fv > i32::MAX as usize {
                return Err(CudaEhlersKamaError::InvalidInput(
                    "first_valid exceeds i32::MAX".into(),
                ));
            }
            first_valids[series] = fv as i32;
        }

        Ok((first_valids, period))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        period: usize,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhlersKamaError> {
        if period == 0 || num_series == 0 || series_len == 0 {
            return Err(CudaEhlersKamaError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        if period > i32::MAX as usize
            || num_series > i32::MAX as usize
            || series_len > i32::MAX as usize
        {
            return Err(CudaEhlersKamaError::InvalidInput(
                "period, num_series, or series_len exceed i32::MAX".into(),
            ));
        }

        let func = self
            .module
            .get_function("ehlers_kama_multi_series_one_param_f32")
            .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;

        let grid: GridSize = (num_series as u32, 1, 1).into();
        let block: BlockSize = (1, 1, 1).into();

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
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
        }

        Ok(())
    }

    pub fn ehlers_kama_multi_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        period: i32,
        num_series: i32,
        series_len: i32,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhlersKamaError> {
        if period <= 0 || num_series <= 0 || series_len <= 0 {
            return Err(CudaEhlersKamaError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_prices_tm,
            period as usize,
            num_series as usize,
            series_len as usize,
            d_first_valids,
            d_out_tm,
        )
    }

    pub fn ehlers_kama_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &EhlersKamaParams,
    ) -> Result<DeviceArrayF32, CudaEhlersKamaError> {
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }
                .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;

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
            .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows,
            cols,
        })
    }

    pub fn ehlers_kama_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &EhlersKamaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaEhlersKamaError> {
        if out_tm.len() != cols * rows {
            return Err(CudaEhlersKamaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out_tm.len(),
                cols * rows
            )));
        }
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }
                .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;

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
            .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;

        d_out_tm
            .copy_to(out_tm)
            .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))
    }
}
