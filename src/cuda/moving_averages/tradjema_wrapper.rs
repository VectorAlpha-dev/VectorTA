//! CUDA wrapper for the TrAdjEMA (Trend Adjusted EMA) kernels.
//!
//! The implementation mirrors the VRAM-first approach used by ALMA/WMA: kernels
//! operate entirely in FP32, callers can provide pre-existing device buffers to
//! avoid host copies, and the public API returns `DeviceArrayF32` handles so
//! higher layers decide when to move results back to the host.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::tradjema::{TradjemaBatchRange, TradjemaParams};
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
pub enum CudaTradjemaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaTradjemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaTradjemaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaTradjemaError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaTradjemaError {}

pub struct CudaTradjema {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaTradjema {
    pub fn new(device_id: usize) -> Result<Self, CudaTradjemaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaTradjemaError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaTradjemaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaTradjemaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/tradjema_kernel.ptx"));
        let module =
            Module::from_ptx(ptx, &[]).map_err(|e| CudaTradjemaError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaTradjemaError::Cuda(e.to_string()))?;

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

    fn expand_range(range: &TradjemaBatchRange) -> Vec<TradjemaParams> {
        let (l_start, l_end, l_step) = range.length;
        let (m_start, m_end, m_step) = range.mult;

        let length_is_single = l_step == 0 || l_start >= l_end;
        let mult_is_single = m_step == 0.0 || m_start >= m_end;

        if length_is_single && mult_is_single {
            return vec![TradjemaParams {
                length: Some(l_start),
                mult: Some(m_start),
            }];
        }

        let mut combos = Vec::new();
        let mut length = l_start;
        loop {
            let mut mult = m_start;
            loop {
                combos.push(TradjemaParams {
                    length: Some(length),
                    mult: Some(mult),
                });

                if mult_is_single || mult >= m_end {
                    break;
                }
                mult += m_step;
                if mult > m_end {
                    break;
                }
            }

            if length_is_single || length >= l_end {
                break;
            }
            length += l_step;
            if length > l_end {
                break;
            }
        }

        combos
    }

    fn prepare_batch_inputs(
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &TradjemaBatchRange,
    ) -> Result<(Vec<TradjemaParams>, usize, usize, usize), CudaTradjemaError> {
        if high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaTradjemaError::InvalidInput("empty OHLC data".into()));
        }
        if high.len() != low.len() || low.len() != close.len() {
            return Err(CudaTradjemaError::InvalidInput(format!(
                "OHLC length mismatch: h={}, l={}, c={}",
                high.len(),
                low.len(),
                close.len()
            )));
        }

        let combos = Self::expand_range(sweep);
        if combos.is_empty() {
            return Err(CudaTradjemaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let series_len = close.len();
        let first_valid = close
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaTradjemaError::InvalidInput("all close values are NaN".into()))?;

        let mut max_length = 0usize;
        for prm in &combos {
            let length = prm.length.unwrap_or(0);
            let mult = prm.mult.unwrap_or(0.0) as f32;
            if length < 2 || length > series_len {
                return Err(CudaTradjemaError::InvalidInput(format!(
                    "invalid length {} (series len {})",
                    length, series_len
                )));
            }
            let valid = series_len - first_valid;
            if valid < length {
                return Err(CudaTradjemaError::InvalidInput(format!(
                    "not enough valid data: needed >= {}, valid = {}",
                    length, valid
                )));
            }
            if !mult.is_finite() || mult <= 0.0f32 {
                return Err(CudaTradjemaError::InvalidInput(format!(
                    "invalid mult {}",
                    prm.mult.unwrap_or(0.0)
                )));
            }
            max_length = max_length.max(length);
        }

        Ok((combos, first_valid, series_len, max_length))
    }

    fn launch_batch_kernel(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        d_lengths: &DeviceBuffer<i32>,
        d_mults: &DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        max_length: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTradjemaError> {
        let shared_bytes = max_length
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| CudaTradjemaError::InvalidInput("shared memory size overflow".into()))?;
        if shared_bytes > 96 * 1024 {
            return Err(CudaTradjemaError::InvalidInput(format!(
                "length {} requires {} bytes shared memory (exceeds 96 KiB limit)",
                max_length, shared_bytes
            )));
        }

        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (1, 1, 1).into();

        let func = self
            .module
            .get_function("tradjema_batch_f32")
            .map_err(|e| CudaTradjemaError::Cuda(e.to_string()))?;

        unsafe {
            let mut high_ptr = d_high.as_device_ptr().as_raw();
            let mut low_ptr = d_low.as_device_ptr().as_raw();
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut lengths_ptr = d_lengths.as_device_ptr().as_raw();
            let mut mults_ptr = d_mults.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut lengths_ptr as *mut _ as *mut c_void,
                &mut mults_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes as u32, args)
                .map_err(|e| CudaTradjemaError::Cuda(e.to_string()))?;
        }

        Ok(())
    }

    fn run_batch_kernel(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        combos: &[TradjemaParams],
        first_valid: usize,
        series_len: usize,
        max_length: usize,
    ) -> Result<DeviceArrayF32, CudaTradjemaError> {
        let n_combos = combos.len();
        let mut lengths_i32 = vec![0i32; n_combos];
        let mut mults_f32 = vec![0f32; n_combos];
        for (idx, prm) in combos.iter().enumerate() {
            lengths_i32[idx] = prm.length.unwrap() as i32;
            mults_f32[idx] = prm.mult.unwrap() as f32;
        }

        let bytes_ohlc = series_len * std::mem::size_of::<f32>() * 3;
        let lengths_bytes = n_combos * std::mem::size_of::<i32>();
        let mults_bytes = n_combos * std::mem::size_of::<f32>();
        let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
        let required = bytes_ohlc + lengths_bytes + mults_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaTradjemaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_high =
            DeviceBuffer::from_slice(high).map_err(|e| CudaTradjemaError::Cuda(e.to_string()))?;
        let d_low =
            DeviceBuffer::from_slice(low).map_err(|e| CudaTradjemaError::Cuda(e.to_string()))?;
        let d_close =
            DeviceBuffer::from_slice(close).map_err(|e| CudaTradjemaError::Cuda(e.to_string()))?;
        let d_lengths = DeviceBuffer::from_slice(&lengths_i32)
            .map_err(|e| CudaTradjemaError::Cuda(e.to_string()))?;
        let d_mults = DeviceBuffer::from_slice(&mults_f32)
            .map_err(|e| CudaTradjemaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaTradjemaError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_high,
            &d_low,
            &d_close,
            &d_lengths,
            &d_mults,
            series_len,
            n_combos,
            first_valid,
            max_length,
            &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaTradjemaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    pub fn tradjema_batch_device(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        d_lengths: &DeviceBuffer<i32>,
        d_mults: &DeviceBuffer<f32>,
        series_len: i32,
        n_combos: i32,
        first_valid: i32,
        max_length: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTradjemaError> {
        if series_len <= 0 || n_combos <= 0 || max_length <= 1 {
            return Err(CudaTradjemaError::InvalidInput(
                "series_len, n_combos must be positive and length > 1".into(),
            ));
        }
        self.launch_batch_kernel(
            d_high,
            d_low,
            d_close,
            d_lengths,
            d_mults,
            series_len as usize,
            n_combos as usize,
            first_valid.max(0) as usize,
            max_length as usize,
            d_out,
        )
    }

    pub fn tradjema_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &TradjemaBatchRange,
    ) -> Result<DeviceArrayF32, CudaTradjemaError> {
        let (combos, first_valid, series_len, max_length) =
            Self::prepare_batch_inputs(high, low, close, sweep)?;
        self.run_batch_kernel(
            high,
            low,
            close,
            &combos,
            first_valid,
            series_len,
            max_length,
        )
    }

    pub fn tradjema_batch_into_host_f32(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &TradjemaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<TradjemaParams>), CudaTradjemaError> {
        let (combos, first_valid, series_len, max_length) =
            Self::prepare_batch_inputs(high, low, close, sweep)?;
        let expected = combos.len() * series_len;
        if out.len() != expected {
            return Err(CudaTradjemaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                expected
            )));
        }
        let arr = self.run_batch_kernel(
            high,
            low,
            close,
            &combos,
            first_valid,
            series_len,
            max_length,
        )?;
        arr.buf
            .copy_to(out)
            .map_err(|e| CudaTradjemaError::Cuda(e.to_string()))?;
        Ok((arr.rows, arr.cols, combos))
    }

    fn prepare_many_series_inputs(
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &TradjemaParams,
    ) -> Result<(Vec<i32>, usize, f32), CudaTradjemaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaTradjemaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        let expected = cols * rows;
        if high_tm.len() != expected || low_tm.len() != expected || close_tm.len() != expected {
            return Err(CudaTradjemaError::InvalidInput(format!(
                "time-major length mismatch: high={}, low={}, close={}, expected={}",
                high_tm.len(),
                low_tm.len(),
                close_tm.len(),
                expected
            )));
        }

        let length = params.length.unwrap_or(0);
        let mult = params.mult.unwrap_or(0.0) as f32;
        if length < 2 || length > rows {
            return Err(CudaTradjemaError::InvalidInput(format!(
                "invalid length {} (series len {})",
                length, rows
            )));
        }
        if !mult.is_finite() || mult <= 0.0 {
            return Err(CudaTradjemaError::InvalidInput(format!(
                "invalid mult {}",
                params.mult.unwrap_or(0.0)
            )));
        }

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let close = close_tm[t * cols + series];
                if !close.is_nan() {
                    fv = Some(t);
                    break;
                }
            }
            let fv = fv.ok_or_else(|| {
                CudaTradjemaError::InvalidInput(format!("series {} all NaN", series))
            })?;
            if rows - fv < length {
                return Err(CudaTradjemaError::InvalidInput(format!(
                    "series {} not enough valid data: needed >= {}, valid = {}",
                    series,
                    length,
                    rows - fv
                )));
            }
            first_valids[series] = fv as i32;
        }

        Ok((first_valids, length, mult))
    }

    fn launch_many_series_kernel(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        length: usize,
        mult: f32,
        d_first_valids: &DeviceBuffer<i32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTradjemaError> {
        let shared_bytes = length
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| CudaTradjemaError::InvalidInput("shared memory size overflow".into()))?;
        if shared_bytes > 96 * 1024 {
            return Err(CudaTradjemaError::InvalidInput(format!(
                "length {} requires {} bytes shared memory (exceeds 96 KiB limit)",
                length, shared_bytes
            )));
        }

        let grid: GridSize = (cols as u32, 1, 1).into();
        let block: BlockSize = (1, 1, 1).into();

        let func = self
            .module
            .get_function("tradjema_many_series_one_param_time_major_f32")
            .map_err(|e| CudaTradjemaError::Cuda(e.to_string()))?;

        unsafe {
            let mut high_ptr = d_high.as_device_ptr().as_raw();
            let mut low_ptr = d_low.as_device_ptr().as_raw();
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut length_i = length as i32;
            let mut mult_f = mult;
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut length_i as *mut _ as *mut c_void,
                &mut mult_f as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes as u32, args)
                .map_err(|e| CudaTradjemaError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    fn run_many_series_kernel(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        first_valids: &[i32],
        length: usize,
        mult: f32,
    ) -> Result<DeviceArrayF32, CudaTradjemaError> {
        let bytes_ohlc = cols * rows * std::mem::size_of::<f32>() * 3;
        let first_valid_bytes = cols * std::mem::size_of::<i32>();
        let out_bytes = cols * rows * std::mem::size_of::<f32>();
        let required = bytes_ohlc + first_valid_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaTradjemaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_high = DeviceBuffer::from_slice(high_tm)
            .map_err(|e| CudaTradjemaError::Cuda(e.to_string()))?;
        let d_low =
            DeviceBuffer::from_slice(low_tm).map_err(|e| CudaTradjemaError::Cuda(e.to_string()))?;
        let d_close = DeviceBuffer::from_slice(close_tm)
            .map_err(|e| CudaTradjemaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(first_valids)
            .map_err(|e| CudaTradjemaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaTradjemaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_high,
            &d_low,
            &d_close,
            cols,
            rows,
            length,
            mult,
            &d_first_valids,
            &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaTradjemaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn tradjema_many_series_one_param_device(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        num_series: i32,
        series_len: i32,
        length: i32,
        mult: f32,
        d_first_valids: &DeviceBuffer<i32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTradjemaError> {
        if num_series <= 0 || series_len <= 0 || length <= 1 || !mult.is_finite() || mult <= 0.0 {
            return Err(CudaTradjemaError::InvalidInput(
                "invalid dimensions or parameters".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_high,
            d_low,
            d_close,
            num_series as usize,
            series_len as usize,
            length as usize,
            mult,
            d_first_valids,
            d_out,
        )
    }

    pub fn tradjema_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &TradjemaParams,
    ) -> Result<DeviceArrayF32, CudaTradjemaError> {
        let (first_valids, length, mult) =
            Self::prepare_many_series_inputs(high_tm, low_tm, close_tm, cols, rows, params)?;
        self.run_many_series_kernel(
            high_tm,
            low_tm,
            close_tm,
            cols,
            rows,
            &first_valids,
            length,
            mult,
        )
    }

    pub fn tradjema_many_series_one_param_time_major_into_host_f32(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &TradjemaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaTradjemaError> {
        if out_tm.len() != cols * rows {
            return Err(CudaTradjemaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out_tm.len(),
                cols * rows
            )));
        }
        let (first_valids, length, mult) =
            Self::prepare_many_series_inputs(high_tm, low_tm, close_tm, cols, rows, params)?;
        let arr = self.run_many_series_kernel(
            high_tm,
            low_tm,
            close_tm,
            cols,
            rows,
            &first_valids,
            length,
            mult,
        )?;
        arr.buf
            .copy_to(out_tm)
            .map_err(|e| CudaTradjemaError::Cuda(e.to_string()))
    }
}
