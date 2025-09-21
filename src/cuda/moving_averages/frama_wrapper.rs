//! CUDA scaffolding for the Fractal Adaptive Moving Average (FRAMA).
//!
//! Follows the VRAM-first approach established by the ALMA integration: host
//! validation prepares parameter sweeps, kernels operate entirely in `f32`, and
//! zero-copy `DeviceArrayF32` handles are returned for both the single-series
//! batch path and the many-series × one-parameter path.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::frama::{FramaBatchRange, FramaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{CopyDestination, DeviceBuffer};
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;

const FRAMA_MAX_WINDOW: usize = 1024;

#[derive(Debug)]
pub enum CudaFramaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaFramaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaFramaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaFramaError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaFramaError {}

fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
    if step == 0 || start == end {
        return vec![start];
    }
    (start..=end).step_by(step).collect()
}

fn evenize(window: usize) -> usize {
    if window & 1 == 1 {
        window + 1
    } else {
        window
    }
}

fn expand_grid(range: &FramaBatchRange) -> Vec<FramaParams> {
    let windows = axis_usize(range.window);
    let scs = axis_usize(range.sc);
    let fcs = axis_usize(range.fc);
    let mut out = Vec::with_capacity(windows.len() * scs.len() * fcs.len());
    for &w in &windows {
        for &s in &scs {
            for &f in &fcs {
                out.push(FramaParams {
                    window: Some(w),
                    sc: Some(s),
                    fc: Some(f),
                });
            }
        }
    }
    out
}

fn first_valid_index(high: &[f32], low: &[f32], close: &[f32]) -> Option<usize> {
    for idx in 0..high.len() {
        if !high[idx].is_nan() && !low[idx].is_nan() && !close[idx].is_nan() {
            return Some(idx);
        }
    }
    None
}

pub struct CudaFrama {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaFrama {
    pub fn new(device_id: usize) -> Result<Self, CudaFramaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaFramaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/frama_kernel.ptx"));
        let module = Module::from_ptx(ptx, &[]).map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaFramaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    fn prepare_batch_inputs(
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &FramaBatchRange,
    ) -> Result<(Vec<FramaParams>, usize, usize), CudaFramaError> {
        if high.is_empty() {
            return Err(CudaFramaError::InvalidInput("empty input".into()));
        }
        if low.len() != high.len() || close.len() != high.len() {
            return Err(CudaFramaError::InvalidInput(format!(
                "mismatched slice lengths: high={}, low={}, close={}",
                high.len(),
                low.len(),
                close.len()
            )));
        }

        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaFramaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let len = high.len();
        let first_valid = first_valid_index(high, low, close)
            .ok_or_else(|| CudaFramaError::InvalidInput("all values are NaN".into()))?;

        let mut max_even = 0usize;
        for combo in &combos {
            let window = combo.window.unwrap_or(0);
            let sc = combo.sc.unwrap_or(0);
            let fc = combo.fc.unwrap_or(0);
            if window == 0 {
                return Err(CudaFramaError::InvalidInput(
                    "window must be greater than zero".into(),
                ));
            }
            if window > len {
                return Err(CudaFramaError::InvalidInput(format!(
                    "window {} exceeds data length {}",
                    window, len
                )));
            }
            if sc == 0 {
                return Err(CudaFramaError::InvalidInput(
                    "sc smoothing constant must be greater than zero".into(),
                ));
            }
            if fc == 0 {
                return Err(CudaFramaError::InvalidInput(
                    "fc smoothing constant must be greater than zero".into(),
                ));
            }
            let even = evenize(window);
            if even > FRAMA_MAX_WINDOW {
                return Err(CudaFramaError::InvalidInput(format!(
                    "evenized window {} exceeds CUDA limit {}",
                    even, FRAMA_MAX_WINDOW
                )));
            }
            if len - first_valid < even {
                return Err(CudaFramaError::InvalidInput(format!(
                    "not enough valid data: need >= {}, have {}",
                    even,
                    len - first_valid
                )));
            }
            max_even = max_even.max(even);
        }

        if max_even == 0 {
            return Err(CudaFramaError::InvalidInput(
                "invalid parameter grid (zero window)".into(),
            ));
        }

        Ok((combos, first_valid, len))
    }

    fn launch_batch_kernel(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        d_windows: &DeviceBuffer<i32>,
        d_scs: &DeviceBuffer<i32>,
        d_fcs: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaFramaError> {
        let func = self
            .module
            .get_function("frama_batch_f32")
            .map_err(|e| CudaFramaError::Cuda(e.to_string()))?;

        let block_x: u32 = 32;
        let grid_x = ((n_combos as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut high_ptr = d_high.as_device_ptr().as_raw();
            let mut low_ptr = d_low.as_device_ptr().as_raw();
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut win_ptr = d_windows.as_device_ptr().as_raw();
            let mut sc_ptr = d_scs.as_device_ptr().as_raw();
            let mut fc_ptr = d_fcs.as_device_ptr().as_raw();
            let mut len_i = series_len as i32;
            let mut combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut win_ptr as *mut _ as *mut c_void,
                &mut sc_ptr as *mut _ as *mut c_void,
                &mut fc_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
        }

        Ok(())
    }

    fn run_batch_kernel(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        combos: &[FramaParams],
        first_valid: usize,
        len: usize,
    ) -> Result<DeviceArrayF32, CudaFramaError> {
        let d_high =
            DeviceBuffer::from_slice(high).map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
        let d_low =
            DeviceBuffer::from_slice(low).map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
        let d_close =
            DeviceBuffer::from_slice(close).map_err(|e| CudaFramaError::Cuda(e.to_string()))?;

        let windows: Vec<i32> = combos.iter().map(|c| c.window.unwrap() as i32).collect();
        let scs: Vec<i32> = combos.iter().map(|c| c.sc.unwrap() as i32).collect();
        let fcs: Vec<i32> = combos.iter().map(|c| c.fc.unwrap() as i32).collect();

        let d_windows =
            DeviceBuffer::from_slice(&windows).map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
        let d_scs =
            DeviceBuffer::from_slice(&scs).map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
        let d_fcs =
            DeviceBuffer::from_slice(&fcs).map_err(|e| CudaFramaError::Cuda(e.to_string()))?;

        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(combos.len() * len) }
            .map_err(|e| CudaFramaError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_high,
            &d_low,
            &d_close,
            &d_windows,
            &d_scs,
            &d_fcs,
            len,
            combos.len(),
            first_valid,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: combos.len(),
            cols: len,
        })
    }

    pub fn frama_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &FramaBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<FramaParams>), CudaFramaError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(high, low, close, sweep)?;
        let dev = self.run_batch_kernel(high, low, close, &combos, first_valid, len)?;
        Ok((dev, combos))
    }

    pub fn frama_batch_into_host_f32(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &FramaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<FramaParams>), CudaFramaError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(high, low, close, sweep)?;
        let expected = combos.len() * len;
        if out.len() != expected {
            return Err(CudaFramaError::InvalidInput(format!(
                "output length mismatch: expected {}, got {}",
                expected,
                out.len()
            )));
        }
        let dev = self.run_batch_kernel(high, low, close, &combos, first_valid, len)?;
        dev.buf
            .copy_to(out)
            .map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
        Ok((dev.rows, dev.cols, combos))
    }

    fn prepare_many_series_inputs(
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &FramaParams,
    ) -> Result<(Vec<i32>, usize, i32, i32, i32), CudaFramaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaFramaError::InvalidInput(
                "series dimensions must be positive".into(),
            ));
        }
        let expected = cols * rows;
        if high_tm.len() != expected || low_tm.len() != expected || close_tm.len() != expected {
            return Err(CudaFramaError::InvalidInput(format!(
                "time-major buffer mismatch: expected {}, got high={}, low={}, close={}",
                expected,
                high_tm.len(),
                low_tm.len(),
                close_tm.len()
            )));
        }

        let window = params.window.ok_or_else(|| {
            CudaFramaError::InvalidInput("window parameter must be provided".into())
        })?;
        let sc = params
            .sc
            .ok_or_else(|| CudaFramaError::InvalidInput("sc parameter must be provided".into()))?;
        let fc = params
            .fc
            .ok_or_else(|| CudaFramaError::InvalidInput("fc parameter must be provided".into()))?;

        if window == 0 {
            return Err(CudaFramaError::InvalidInput(
                "window must be greater than zero".into(),
            ));
        }
        if sc == 0 {
            return Err(CudaFramaError::InvalidInput(
                "sc smoothing constant must be greater than zero".into(),
            ));
        }
        if fc == 0 {
            return Err(CudaFramaError::InvalidInput(
                "fc smoothing constant must be greater than zero".into(),
            ));
        }

        let even = evenize(window);
        if even > FRAMA_MAX_WINDOW {
            return Err(CudaFramaError::InvalidInput(format!(
                "evenized window {} exceeds CUDA limit {}",
                even, FRAMA_MAX_WINDOW
            )));
        }
        if even > rows {
            return Err(CudaFramaError::InvalidInput(format!(
                "window {} exceeds series length {}",
                even, rows
            )));
        }

        let stride = cols;
        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut first = None;
            for row in 0..rows {
                let idx = row * stride + series;
                if !high_tm[idx].is_nan() && !low_tm[idx].is_nan() && !close_tm[idx].is_nan() {
                    first = Some(row);
                    break;
                }
            }
            let fv = first.ok_or_else(|| {
                CudaFramaError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            if rows - fv < even {
                return Err(CudaFramaError::InvalidInput(format!(
                    "series {} lacks sufficient tail length: need >= {}, have {}",
                    series,
                    even,
                    rows - fv
                )));
            }
            first_valids[series] = fv as i32;
        }

        Ok((first_valids, even, window as i32, sc as i32, fc as i32))
    }

    fn launch_many_series_kernel(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        num_series: usize,
        series_len: usize,
        window: i32,
        sc: i32,
        fc: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaFramaError> {
        let func = self
            .module
            .get_function("frama_many_series_one_param_f32")
            .map_err(|e| CudaFramaError::Cuda(e.to_string()))?;

        let block_x: u32 = 64;
        let grid_x = ((num_series as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut high_ptr = d_high.as_device_ptr().as_raw();
            let mut low_ptr = d_low.as_device_ptr().as_raw();
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut cols_i = num_series as i32;
            let mut rows_i = series_len as i32;
            let mut window_i = window;
            let mut sc_i = sc;
            let mut fc_i = fc;
            let mut out_ptr = d_out.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut window_i as *mut _ as *mut c_void,
                &mut sc_i as *mut _ as *mut c_void,
                &mut fc_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
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
        window: i32,
        sc: i32,
        fc: i32,
    ) -> Result<DeviceArrayF32, CudaFramaError> {
        let d_high =
            DeviceBuffer::from_slice(high_tm).map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
        let d_low =
            DeviceBuffer::from_slice(low_tm).map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
        let d_close =
            DeviceBuffer::from_slice(close_tm).map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(first_valids)
            .map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(cols * rows) }
            .map_err(|e| CudaFramaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_high, &d_low, &d_close, &d_first, cols, rows, window, sc, fc, &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn frama_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &FramaParams,
    ) -> Result<DeviceArrayF32, CudaFramaError> {
        let (first_valids, _even_window, window_i, sc_i, fc_i) =
            Self::prepare_many_series_inputs(high_tm, low_tm, close_tm, cols, rows, params)?;
        self.run_many_series_kernel(
            high_tm,
            low_tm,
            close_tm,
            cols,
            rows,
            &first_valids,
            window_i,
            sc_i,
            fc_i,
        )
    }

    pub fn frama_many_series_one_param_time_major_into_host_f32(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &FramaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaFramaError> {
        let expected = cols * rows;
        if out_tm.len() != expected {
            return Err(CudaFramaError::InvalidInput(format!(
                "output length mismatch: expected {}, got {}",
                expected,
                out_tm.len()
            )));
        }
        let (first_valids, _even_window, window_i, sc_i, fc_i) =
            Self::prepare_many_series_inputs(high_tm, low_tm, close_tm, cols, rows, params)?;
        let dev = self.run_many_series_kernel(
            high_tm,
            low_tm,
            close_tm,
            cols,
            rows,
            &first_valids,
            window_i,
            sc_i,
            fc_i,
        )?;
        dev.buf
            .copy_to(out_tm)
            .map_err(|e| CudaFramaError::Cuda(e.to_string()))
    }
}
