//! CUDA wrapper for the Ehlers Error Correcting Exponential Moving Average (ECEMA).
//!
//! Mirrors the VRAM-first design used across the moving-average CUDA wrappers:
//! inputs are staged once into device buffers, kernels operate entirely in FP32
//! memory, and intermediate arithmetic is promoted to FP64 inside the kernels to
//! stay aligned with the CPU reference implementation. Two entry points are
//! provided: a batch launcher for a single series across many `(length, gain)`
//! pairs, and a time-major variant that processes many series sharing a common
//! parameter set.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::ehlers_ecema::{EhlersEcemaBatchRange, EhlersEcemaParams};
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
pub enum CudaEhlersEcemaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaEhlersEcemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaEhlersEcemaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaEhlersEcemaError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaEhlersEcemaError {}

pub struct CudaEhlersEcema {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaEhlersEcema {
    pub fn new(device_id: usize) -> Result<Self, CudaEhlersEcemaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;
        let context =
            Context::new(device).map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/ehlers_ecema_kernel.ptx"));
        let module =
            Module::from_ptx(ptx, &[]).map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;

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

    fn axis_usize(axis: (usize, usize, usize)) -> Vec<usize> {
        let (start, end, step) = axis;
        if step == 0 || start == end {
            vec![start]
        } else if start <= end {
            (start..=end).step_by(step).collect()
        } else {
            vec![start]
        }
    }

    fn expand_range(
        range: &EhlersEcemaBatchRange,
        pine_mode: bool,
        confirmed: bool,
    ) -> Vec<EhlersEcemaParams> {
        let lengths = Self::axis_usize(range.length);
        let gain_limits = Self::axis_usize(range.gain_limit);
        let mut combos = Vec::with_capacity(lengths.len() * gain_limits.len());
        for &len in &lengths {
            for &gain in &gain_limits {
                combos.push(EhlersEcemaParams {
                    length: Some(len),
                    gain_limit: Some(gain),
                    pine_compatible: Some(pine_mode),
                    confirmed_only: Some(confirmed),
                });
            }
        }
        combos
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &EhlersEcemaBatchRange,
        pine_mode: bool,
        confirmed: bool,
    ) -> Result<(Vec<EhlersEcemaParams>, usize, usize), CudaEhlersEcemaError> {
        if data_f32.is_empty() {
            return Err(CudaEhlersEcemaError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaEhlersEcemaError::InvalidInput("all values are NaN".into()))?;

        let combos = Self::expand_range(sweep, pine_mode, confirmed);
        if combos.is_empty() {
            return Err(CudaEhlersEcemaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let series_len = data_f32.len();
        for prm in &combos {
            let length = prm.length.unwrap_or(0);
            let gain = prm.gain_limit.unwrap_or(0);
            if length == 0 {
                return Err(CudaEhlersEcemaError::InvalidInput(
                    "length must be >= 1".into(),
                ));
            }
            if gain == 0 {
                return Err(CudaEhlersEcemaError::InvalidInput(
                    "gain_limit must be >= 1".into(),
                ));
            }
            if length > series_len {
                return Err(CudaEhlersEcemaError::InvalidInput(format!(
                    "length {} exceeds data length {}",
                    length, series_len
                )));
            }
            let valid = series_len - first_valid;
            if !pine_mode && valid < length {
                return Err(CudaEhlersEcemaError::InvalidInput(format!(
                    "not enough valid data: need >= {}, valid = {}",
                    length, valid
                )));
            }
        }

        Ok((combos, first_valid, series_len))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_lengths: &DeviceBuffer<i32>,
        d_gain_limits: &DeviceBuffer<i32>,
        d_pine_flags: &DeviceBuffer<u8>,
        d_confirmed_flags: &DeviceBuffer<u8>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhlersEcemaError> {
        const BLOCK_X: u32 = 128;
        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (BLOCK_X, 1, 1).into();

        let func = self
            .module
            .get_function("ehlers_ecema_batch_f32")
            .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut lengths_ptr = d_lengths.as_device_ptr().as_raw();
            let mut gains_ptr = d_gain_limits.as_device_ptr().as_raw();
            let mut pine_ptr = d_pine_flags.as_device_ptr().as_raw();
            let mut confirmed_ptr = d_confirmed_flags.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut lengths_ptr as *mut _ as *mut c_void,
                &mut gains_ptr as *mut _ as *mut c_void,
                &mut pine_ptr as *mut _ as *mut c_void,
                &mut confirmed_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        combos: &[EhlersEcemaParams],
        first_valid: usize,
        series_len: usize,
    ) -> Result<DeviceArrayF32, CudaEhlersEcemaError> {
        let n_combos = combos.len();
        let prices_bytes = series_len * std::mem::size_of::<f32>();
        let lengths_bytes = n_combos * std::mem::size_of::<i32>();
        let gains_bytes = n_combos * std::mem::size_of::<i32>();
        let flags_bytes = n_combos * std::mem::size_of::<u8>() * 2;
        let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + lengths_bytes + gains_bytes + flags_bytes + out_bytes;
        let headroom = 32 * 1024 * 1024; // 32MB cushion
        if !Self::will_fit(required, headroom) {
            return Err(CudaEhlersEcemaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices = DeviceBuffer::from_slice(data_f32)
            .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;
        let lengths_i32: Vec<i32> = combos.iter().map(|p| p.length.unwrap() as i32).collect();
        let gain_i32: Vec<i32> = combos
            .iter()
            .map(|p| p.gain_limit.unwrap() as i32)
            .collect();
        let pine_flags: Vec<u8> = combos
            .iter()
            .map(|p| {
                if p.pine_compatible.unwrap_or(false) {
                    1
                } else {
                    0
                }
            })
            .collect();
        let confirmed_flags: Vec<u8> = combos
            .iter()
            .map(|p| {
                if p.confirmed_only.unwrap_or(false) {
                    1
                } else {
                    0
                }
            })
            .collect();

        let d_lengths = DeviceBuffer::from_slice(&lengths_i32)
            .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;
        let d_gains = DeviceBuffer::from_slice(&gain_i32)
            .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;
        let d_pine = DeviceBuffer::from_slice(&pine_flags)
            .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;
        let d_confirmed = DeviceBuffer::from_slice(&confirmed_flags)
            .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_prices,
            &d_lengths,
            &d_gains,
            &d_pine,
            &d_confirmed,
            series_len,
            n_combos,
            first_valid,
            &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    pub fn ehlers_ecema_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &EhlersEcemaBatchRange,
        params: &EhlersEcemaParams,
    ) -> Result<DeviceArrayF32, CudaEhlersEcemaError> {
        let pine_mode = params.pine_compatible.unwrap_or(false);
        let confirmed = params.confirmed_only.unwrap_or(false);
        let (combos, first_valid, series_len) =
            Self::prepare_batch_inputs(data_f32, sweep, pine_mode, confirmed)?;
        self.run_batch_kernel(data_f32, &combos, first_valid, series_len)
    }

    pub fn ehlers_ecema_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &EhlersEcemaBatchRange,
        params: &EhlersEcemaParams,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<EhlersEcemaParams>), CudaEhlersEcemaError> {
        let pine_mode = params.pine_compatible.unwrap_or(false);
        let confirmed = params.confirmed_only.unwrap_or(false);
        let (combos, first_valid, series_len) =
            Self::prepare_batch_inputs(data_f32, sweep, pine_mode, confirmed)?;
        let expected = combos.len() * series_len;
        if out.len() != expected {
            return Err(CudaEhlersEcemaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                expected
            )));
        }
        let arr = self.run_batch_kernel(data_f32, &combos, first_valid, series_len)?;
        arr.buf
            .copy_to(out)
            .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;
        Ok((arr.rows, arr.cols, combos))
    }

    pub fn ehlers_ecema_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_lengths: &DeviceBuffer<i32>,
        d_gain_limits: &DeviceBuffer<i32>,
        d_pine_flags: &DeviceBuffer<u8>,
        d_confirmed_flags: &DeviceBuffer<u8>,
        series_len: i32,
        n_combos: i32,
        first_valid: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhlersEcemaError> {
        if series_len <= 0 || n_combos <= 0 {
            return Err(CudaEhlersEcemaError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }
        self.launch_batch_kernel(
            d_prices,
            d_lengths,
            d_gain_limits,
            d_pine_flags,
            d_confirmed_flags,
            series_len as usize,
            n_combos as usize,
            first_valid.max(0) as usize,
            d_out,
        )
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &EhlersEcemaParams,
    ) -> Result<(Vec<i32>, usize, usize, bool, bool), CudaEhlersEcemaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaEhlersEcemaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaEhlersEcemaError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }

        let length = params.length.unwrap_or(20);
        let gain_limit = params.gain_limit.unwrap_or(50);
        if length == 0 {
            return Err(CudaEhlersEcemaError::InvalidInput(
                "length must be >= 1".into(),
            ));
        }
        if gain_limit == 0 {
            return Err(CudaEhlersEcemaError::InvalidInput(
                "gain_limit must be >= 1".into(),
            ));
        }

        let pine_mode = params.pine_compatible.unwrap_or(false);
        let confirmed = params.confirmed_only.unwrap_or(false);

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut found = None;
            for t in 0..rows {
                let idx = t * cols + series;
                if !data_tm_f32[idx].is_nan() {
                    found = Some(t as i32);
                    break;
                }
            }
            let fv = found.ok_or_else(|| {
                CudaEhlersEcemaError::InvalidInput(format!("series {} all NaN", series))
            })?;
            let valid = rows - fv as usize;
            if !pine_mode && valid < length {
                return Err(CudaEhlersEcemaError::InvalidInput(format!(
                    "series {} does not have enough valid data: need >= {}, valid = {}",
                    series, length, valid
                )));
            }
            first_valids[series] = fv;
        }

        Ok((first_valids, length, gain_limit, pine_mode, confirmed))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        length: usize,
        gain_limit: usize,
        pine_mode: bool,
        confirmed: bool,
        d_first_valids: &DeviceBuffer<i32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhlersEcemaError> {
        const BLOCK_X: u32 = 128;
        let grid: GridSize = (cols as u32, 1, 1).into();
        let block: BlockSize = (BLOCK_X, 1, 1).into();

        let func = self
            .module
            .get_function("ehlers_ecema_many_series_one_param_time_major_f32")
            .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut length_i = length as i32;
            let mut gain_limit_i = gain_limit as i32;
            let mut pine_flag = if pine_mode { 1u8 } else { 0u8 };
            let mut confirmed_flag = if confirmed { 1u8 } else { 0u8 };
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut length_i as *mut _ as *mut c_void,
                &mut gain_limit_i as *mut _ as *mut c_void,
                &mut pine_flag as *mut _ as *mut c_void,
                &mut confirmed_flag as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    fn run_many_series_kernel(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        length: usize,
        gain_limit: usize,
        pine_mode: bool,
        confirmed: bool,
        first_valids: &[i32],
    ) -> Result<DeviceArrayF32, CudaEhlersEcemaError> {
        let prices_bytes = cols * rows * std::mem::size_of::<f32>();
        let first_valid_bytes = cols * std::mem::size_of::<i32>();
        let out_bytes = cols * rows * std::mem::size_of::<f32>();
        let required = prices_bytes + first_valid_bytes + out_bytes;
        let headroom = 16 * 1024 * 1024; // 16MB cushion
        if !Self::will_fit(required, headroom) {
            return Err(CudaEhlersEcemaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(first_valids)
            .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices,
            cols,
            rows,
            length,
            gain_limit,
            pine_mode,
            confirmed,
            &d_first_valids,
            &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn ehlers_ecema_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &EhlersEcemaParams,
    ) -> Result<DeviceArrayF32, CudaEhlersEcemaError> {
        let (first_valids, length, gain_limit, pine_mode, confirmed) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        self.run_many_series_kernel(
            data_tm_f32,
            cols,
            rows,
            length,
            gain_limit,
            pine_mode,
            confirmed,
            &first_valids,
        )
    }

    pub fn ehlers_ecema_many_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &EhlersEcemaParams,
        out: &mut [f32],
    ) -> Result<(), CudaEhlersEcemaError> {
        if out.len() != cols * rows {
            return Err(CudaEhlersEcemaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                cols * rows
            )));
        }
        let arr = self.ehlers_ecema_many_series_one_param_time_major_dev(
            data_tm_f32,
            cols,
            rows,
            params,
        )?;
        arr.buf
            .copy_to(out)
            .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))
    }

    pub fn ehlers_ecema_many_series_one_param_time_major_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        cols: i32,
        rows: i32,
        length: i32,
        gain_limit: i32,
        pine_flag: u8,
        confirmed_flag: u8,
        d_first_valids: &DeviceBuffer<i32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhlersEcemaError> {
        if cols <= 0 || rows <= 0 || length <= 0 || gain_limit <= 0 {
            return Err(CudaEhlersEcemaError::InvalidInput(
                "cols, rows, length and gain_limit must be positive".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_prices,
            cols as usize,
            rows as usize,
            length as usize,
            gain_limit as usize,
            pine_flag != 0,
            confirmed_flag != 0,
            d_first_valids,
            d_out,
        )
    }

}
