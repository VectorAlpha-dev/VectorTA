//! CUDA wrapper for the New Adaptive Moving Average (NAMA) kernels.
//!
//! Mirrors the VRAM-first design of the other moving-average wrappers: inputs
//! are accepted as FP32 host slices, staged to device memory, and the public
//! API returns `DeviceArrayF32` handles so callers control any host copies.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::nama::{NamaBatchRange, NamaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{CopyDestination, DeviceBuffer};
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::sys as cu;
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaNamaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaNamaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaNamaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaNamaError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaNamaError {}

pub struct CudaNama {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaNama {
    pub fn new(device_id: usize) -> Result<Self, CudaNamaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaNamaError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaNamaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaNamaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/nama_kernel.ptx"));
        let module = Module::from_ptx(ptx, &[]).map_err(|e| CudaNamaError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaNamaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }

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

    fn expand_periods(range: &NamaBatchRange) -> Vec<NamaParams> {
        let (start, end, step) = range.period;
        let periods = if step == 0 || start == end {
            vec![start]
        } else {
            (start..=end).step_by(step.max(1)).collect::<Vec<_>>()
        };
        periods
            .into_iter()
            .map(|p| NamaParams { period: Some(p) })
            .collect()
    }

    fn prepare_batch_inputs(
        prices: &[f32],
        high: Option<&[f32]>,
        low: Option<&[f32]>,
        close: Option<&[f32]>,
        sweep: &NamaBatchRange,
    ) -> Result<(Vec<NamaParams>, usize, usize, usize, bool), CudaNamaError> {
        if prices.is_empty() {
            return Err(CudaNamaError::InvalidInput("price data is empty".into()));
        }
        let has_ohlc = high.is_some() || low.is_some() || close.is_some();
        if has_ohlc {
            if high.is_none() || low.is_none() || close.is_none() {
                return Err(CudaNamaError::InvalidInput(
                    "when providing OHLC data, high/low/close must all be present".into(),
                ));
            }
            let len = prices.len();
            if high.unwrap().len() != len
                || low.unwrap().len() != len
                || close.unwrap().len() != len
            {
                return Err(CudaNamaError::InvalidInput(
                    "price/high/low/close slices must have equal length".into(),
                ));
            }
        }

        let first_valid = prices
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaNamaError::InvalidInput("all price values are NaN".into()))?;

        let combos = Self::expand_periods(sweep);
        if combos.is_empty() {
            return Err(CudaNamaError::InvalidInput(
                "no parameter combinations generated".into(),
            ));
        }

        let series_len = prices.len();
        let mut max_period = 0usize;
        for prm in &combos {
            let period = prm.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaNamaError::InvalidInput("period must be >= 1".into()));
            }
            if period > series_len {
                return Err(CudaNamaError::InvalidInput(format!(
                    "period {} exceeds series length {}",
                    period, series_len
                )));
            }
            let valid = series_len - first_valid;
            if valid < period {
                return Err(CudaNamaError::InvalidInput(format!(
                    "not enough valid data: needed >= {}, valid = {}",
                    period, valid
                )));
            }
            max_period = max_period.max(period);
        }

        Ok((combos, first_valid, series_len, max_period, has_ohlc))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_high: Option<&DeviceBuffer<f32>>,
        d_low: Option<&DeviceBuffer<f32>>,
        d_close: Option<&DeviceBuffer<f32>>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        max_period: usize,
        has_ohlc: bool,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaNamaError> {
        if max_period == 0 {
            return Err(CudaNamaError::InvalidInput(
                "max_period must be positive".into(),
            ));
        }

        let shared_bytes = (max_period + 1)
            .checked_mul(2 * std::mem::size_of::<i32>())
            .ok_or_else(|| CudaNamaError::InvalidInput("shared memory size overflow".into()))?;
        if shared_bytes > 96 * 1024 {
            return Err(CudaNamaError::InvalidInput(format!(
                "period {} requires {} bytes shared memory (exceeds limit)",
                max_period, shared_bytes
            )));
        }

        const BLOCK_X: u32 = 128;
        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (BLOCK_X, 1, 1).into();

        let func = self
            .module
            .get_function("nama_batch_f32")
            .map_err(|e| CudaNamaError::Cuda(e.to_string()))?;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut high_ptr = d_high.map(|buf| buf.as_device_ptr().as_raw()).unwrap_or(0);
            let mut low_ptr = d_low.map(|buf| buf.as_device_ptr().as_raw()).unwrap_or(0);
            let mut close_ptr = d_close.map(|buf| buf.as_device_ptr().as_raw()).unwrap_or(0);
            let mut has_ohlc_i = if has_ohlc { 1i32 } else { 0i32 };
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut has_ohlc_i as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes as u32, args)
                .map_err(|e| CudaNamaError::Cuda(e.to_string()))?;
        }

        Ok(())
    }

    fn run_batch_kernel(
        &self,
        prices: &[f32],
        high: Option<&[f32]>,
        low: Option<&[f32]>,
        close: Option<&[f32]>,
        combos: &[NamaParams],
        first_valid: usize,
        series_len: usize,
        max_period: usize,
        has_ohlc: bool,
    ) -> Result<DeviceArrayF32, CudaNamaError> {
        let n_combos = combos.len();
        if n_combos == 0 {
            return Err(CudaNamaError::InvalidInput("no period combinations".into()));
        }

        let mut periods_i32 = Vec::with_capacity(n_combos);
        for prm in combos {
            let period = prm.period.unwrap();
            periods_i32.push(period as i32);
        }

        let prices_bytes = series_len * std::mem::size_of::<f32>();
        let ohlc_bytes = if has_ohlc {
            3 * series_len * std::mem::size_of::<f32>()
        } else {
            0
        };
        let periods_bytes = n_combos * std::mem::size_of::<i32>();
        let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + ohlc_bytes + periods_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024; // 64 MiB safety margin
        if !Self::will_fit(required, headroom) {
            return Err(CudaNamaError::InvalidInput(
                "not enough free device memory".into(),
            ));
        }

        let d_prices =
            DeviceBuffer::from_slice(prices).map_err(|e| CudaNamaError::Cuda(e.to_string()))?;
        let d_high = if has_ohlc {
            Some(
                DeviceBuffer::from_slice(high.unwrap())
                    .map_err(|e| CudaNamaError::Cuda(e.to_string()))?,
            )
        } else {
            None
        };
        let d_low = if has_ohlc {
            Some(
                DeviceBuffer::from_slice(low.unwrap())
                    .map_err(|e| CudaNamaError::Cuda(e.to_string()))?,
            )
        } else {
            None
        };
        let d_close = if has_ohlc {
            Some(
                DeviceBuffer::from_slice(close.unwrap())
                    .map_err(|e| CudaNamaError::Cuda(e.to_string()))?,
            )
        } else {
            None
        };

        let d_periods = DeviceBuffer::from_slice(&periods_i32)
            .map_err(|e| CudaNamaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaNamaError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_prices,
            d_high.as_ref(),
            d_low.as_ref(),
            d_close.as_ref(),
            &d_periods,
            series_len,
            n_combos,
            first_valid,
            max_period,
            has_ohlc,
            &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaNamaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    pub fn nama_batch_dev(
        &self,
        prices: &[f32],
        sweep: &NamaBatchRange,
    ) -> Result<DeviceArrayF32, CudaNamaError> {
        let (combos, first_valid, series_len, max_period, has_ohlc) =
            Self::prepare_batch_inputs(prices, None, None, None, sweep)?;
        debug_assert!(!has_ohlc);
        self.run_batch_kernel(
            prices,
            None,
            None,
            None,
            &combos,
            first_valid,
            series_len,
            max_period,
            false,
        )
    }

    pub fn nama_batch_with_ohlc_dev(
        &self,
        prices: &[f32],
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &NamaBatchRange,
    ) -> Result<DeviceArrayF32, CudaNamaError> {
        let (combos, first_valid, series_len, max_period, has_ohlc) =
            Self::prepare_batch_inputs(prices, Some(high), Some(low), Some(close), sweep)?;
        self.run_batch_kernel(
            prices,
            Some(high),
            Some(low),
            Some(close),
            &combos,
            first_valid,
            series_len,
            max_period,
            has_ohlc,
        )
    }

    pub fn nama_batch_into_host_f32(
        &self,
        prices: &[f32],
        sweep: &NamaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<NamaParams>), CudaNamaError> {
        let (combos, first_valid, series_len, max_period, has_ohlc) =
            Self::prepare_batch_inputs(prices, None, None, None, sweep)?;
        let expected = combos.len() * series_len;
        if out.len() != expected {
            return Err(CudaNamaError::InvalidInput(format!(
                "output slice len {} != expected {}",
                out.len(),
                expected
            )));
        }
        let arr = self.run_batch_kernel(
            prices,
            None,
            None,
            None,
            &combos,
            first_valid,
            series_len,
            max_period,
            has_ohlc,
        )?;
        arr.buf
            .copy_to(out)
            .map_err(|e| CudaNamaError::Cuda(e.to_string()))?;
        Ok((arr.rows, arr.cols, combos))
    }

    pub fn nama_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_high: Option<&DeviceBuffer<f32>>,
        d_low: Option<&DeviceBuffer<f32>>,
        d_close: Option<&DeviceBuffer<f32>>,
        d_periods: &DeviceBuffer<i32>,
        series_len: i32,
        n_combos: i32,
        first_valid: i32,
        max_period: i32,
        has_ohlc: bool,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaNamaError> {
        if series_len <= 0 || n_combos <= 0 || max_period <= 0 {
            return Err(CudaNamaError::InvalidInput(
                "series_len, n_combos, and max_period must be positive".into(),
            ));
        }
        self.launch_batch_kernel(
            d_prices,
            d_high,
            d_low,
            d_close,
            d_periods,
            series_len as usize,
            n_combos as usize,
            first_valid.max(0) as usize,
            max_period as usize,
            has_ohlc,
            d_out,
        )
    }

    fn run_many_series_kernel(
        &self,
        prices_tm: &[f32],
        high_tm: Option<&[f32]>,
        low_tm: Option<&[f32]>,
        close_tm: Option<&[f32]>,
        num_series: usize,
        series_len: usize,
        first_valids: &[i32],
        period: usize,
        has_ohlc: bool,
    ) -> Result<DeviceArrayF32, CudaNamaError> {
        let total = num_series * series_len;
        let prices_bytes = total * std::mem::size_of::<f32>();
        let ohlc_bytes = if has_ohlc {
            3 * total * std::mem::size_of::<f32>()
        } else {
            0
        };
        let fv_bytes = first_valids.len() * std::mem::size_of::<i32>();
        let out_bytes = total * std::mem::size_of::<f32>();
        let required = prices_bytes + ohlc_bytes + fv_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaNamaError::InvalidInput(
                "not enough free device memory".into(),
            ));
        }

        let d_prices =
            DeviceBuffer::from_slice(prices_tm).map_err(|e| CudaNamaError::Cuda(e.to_string()))?;
        let d_high = if has_ohlc {
            Some(
                DeviceBuffer::from_slice(high_tm.unwrap())
                    .map_err(|e| CudaNamaError::Cuda(e.to_string()))?,
            )
        } else {
            None
        };
        let d_low = if has_ohlc {
            Some(
                DeviceBuffer::from_slice(low_tm.unwrap())
                    .map_err(|e| CudaNamaError::Cuda(e.to_string()))?,
            )
        } else {
            None
        };
        let d_close = if has_ohlc {
            Some(
                DeviceBuffer::from_slice(close_tm.unwrap())
                    .map_err(|e| CudaNamaError::Cuda(e.to_string()))?,
            )
        } else {
            None
        };
        let d_first_valids = DeviceBuffer::from_slice(first_valids)
            .map_err(|e| CudaNamaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total) }
            .map_err(|e| CudaNamaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices,
            d_high.as_ref(),
            d_low.as_ref(),
            d_close.as_ref(),
            num_series,
            series_len,
            period,
            &d_first_valids,
            has_ohlc,
            &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaNamaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: series_len,
            cols: num_series,
        })
    }

    pub fn nama_many_series_one_param_time_major_dev(
        &self,
        prices_tm: &[f32],
        num_series: usize,
        series_len: usize,
        params: &NamaParams,
    ) -> Result<DeviceArrayF32, CudaNamaError> {
        let (first_valids, period, has_ohlc) = Self::prepare_many_series_inputs(
            prices_tm, None, None, None, num_series, series_len, params,
        )?;
        self.run_many_series_kernel(
            prices_tm,
            None,
            None,
            None,
            num_series,
            series_len,
            &first_valids,
            period,
            has_ohlc,
        )
    }

    pub fn nama_many_series_one_param_time_major_into_host_f32(
        &self,
        prices_tm: &[f32],
        num_series: usize,
        series_len: usize,
        params: &NamaParams,
        out: &mut [f32],
    ) -> Result<(), CudaNamaError> {
        if out.len() != num_series * series_len {
            return Err(CudaNamaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                num_series * series_len
            )));
        }
        let arr = self
            .nama_many_series_one_param_time_major_dev(prices_tm, num_series, series_len, params)?;
        arr.buf
            .copy_to(out)
            .map_err(|e| CudaNamaError::Cuda(e.to_string()))
    }

    pub fn nama_many_series_one_param_time_major_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_high: Option<&DeviceBuffer<f32>>,
        d_low: Option<&DeviceBuffer<f32>>,
        d_close: Option<&DeviceBuffer<f32>>,
        num_series: i32,
        series_len: i32,
        period: i32,
        d_first_valids: &DeviceBuffer<i32>,
        has_ohlc: bool,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaNamaError> {
        if num_series <= 0 || series_len <= 0 || period <= 0 {
            return Err(CudaNamaError::InvalidInput(
                "num_series, series_len, and period must be positive".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_prices,
            d_high,
            d_low,
            d_close,
            num_series as usize,
            series_len as usize,
            period as usize,
            d_first_valids,
            has_ohlc,
            d_out,
        )
    }

    fn prepare_many_series_inputs(
        prices_tm: &[f32],
        high_tm: Option<&[f32]>,
        low_tm: Option<&[f32]>,
        close_tm: Option<&[f32]>,
        num_series: usize,
        series_len: usize,
        params: &NamaParams,
    ) -> Result<(Vec<i32>, usize, bool), CudaNamaError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaNamaError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if prices_tm.len() != num_series * series_len {
            return Err(CudaNamaError::InvalidInput(format!(
                "price tensor length {} != num_series*series_len {}",
                prices_tm.len(),
                num_series * series_len
            )));
        }
        let has_ohlc = high_tm.is_some() || low_tm.is_some() || close_tm.is_some();
        if has_ohlc {
            if high_tm.is_none() || low_tm.is_none() || close_tm.is_none() {
                return Err(CudaNamaError::InvalidInput(
                    "when supplying OHLC tensors, high/low/close must all be provided".into(),
                ));
            }
            let expected = num_series * series_len;
            if high_tm.unwrap().len() != expected
                || low_tm.unwrap().len() != expected
                || close_tm.unwrap().len() != expected
            {
                return Err(CudaNamaError::InvalidInput(
                    "price/high/low/close tensors must share the same length".into(),
                ));
            }
        }

        let period = params
            .period
            .ok_or_else(|| CudaNamaError::InvalidInput("period must be specified".into()))?;
        if period == 0 || period > series_len {
            return Err(CudaNamaError::InvalidInput(format!(
                "invalid period {} for series_len {}",
                period, series_len
            )));
        }

        let mut first_valids = Vec::with_capacity(num_series);
        for series in 0..num_series {
            let mut fv = None;
            for t in 0..series_len {
                let idx = t * num_series + series;
                if !prices_tm[idx].is_nan() {
                    fv = Some(t as i32);
                    break;
                }
            }
            let fv_i = fv.ok_or_else(|| {
                CudaNamaError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            let valid = series_len - fv_i as usize;
            if valid < period {
                return Err(CudaNamaError::InvalidInput(format!(
                    "series {} lacks data: needed >= {}, valid = {}",
                    series, period, valid
                )));
            }
            first_valids.push(fv_i);
        }

        Ok((first_valids, period, has_ohlc))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_high: Option<&DeviceBuffer<f32>>,
        d_low: Option<&DeviceBuffer<f32>>,
        d_close: Option<&DeviceBuffer<f32>>,
        num_series: usize,
        series_len: usize,
        period: usize,
        d_first_valids: &DeviceBuffer<i32>,
        has_ohlc: bool,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaNamaError> {
        if period == 0 {
            return Err(CudaNamaError::InvalidInput(
                "period must be positive".into(),
            ));
        }
        let shared_bytes = (period + 1)
            .checked_mul(2 * std::mem::size_of::<i32>())
            .ok_or_else(|| CudaNamaError::InvalidInput("shared memory size overflow".into()))?;
        if shared_bytes > 96 * 1024 {
            return Err(CudaNamaError::InvalidInput(format!(
                "period {} requires {} bytes shared memory (exceeds limit)",
                period, shared_bytes
            )));
        }

        const BLOCK_X: u32 = 128;
        let grid: GridSize = (num_series as u32, 1, 1).into();
        let block: BlockSize = (BLOCK_X, 1, 1).into();

        let func = self
            .module
            .get_function("nama_many_series_one_param_time_major_f32")
            .map_err(|e| CudaNamaError::Cuda(e.to_string()))?;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut high_ptr = d_high.map(|buf| buf.as_device_ptr().as_raw()).unwrap_or(0);
            let mut low_ptr = d_low.map(|buf| buf.as_device_ptr().as_raw()).unwrap_or(0);
            let mut close_ptr = d_close.map(|buf| buf.as_device_ptr().as_raw()).unwrap_or(0);
            let mut has_ohlc_i = if has_ohlc { 1i32 } else { 0i32 };
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut period_i = period as i32;
            let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut has_ohlc_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut first_valids_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes as u32, args)
                .map_err(|e| CudaNamaError::Cuda(e.to_string()))?;
        }

        Ok(())
    }
}
