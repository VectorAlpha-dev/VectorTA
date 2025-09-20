//! CUDA scaffolding for the ALMA (Arnaud Legoux Moving Average) kernels.
//!
//! The implementation now follows a VRAM-first design: kernels operate purely in
//! FP32, weights are generated on-device, and host copies are optional helpers
//! layered on top of device-returning entry points.

#![cfg(feature = "cuda")]

use crate::indicators::moving_averages::alma::{AlmaBatchRange, AlmaParams};
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
pub enum CudaAlmaError {
    Cuda(String),
    NotImplemented,
    InvalidInput(String),
}

impl fmt::Display for CudaAlmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaAlmaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaAlmaError::NotImplemented => write!(f, "CUDA ALMA not implemented yet"),
            CudaAlmaError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}

impl std::error::Error for CudaAlmaError {}

/// VRAM-backed array handle returned to higher layers (Python bindings, etc.).
pub struct DeviceArrayF32 {
    pub buf: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl DeviceArrayF32 {
    #[inline]
    pub fn device_ptr(&self) -> u64 {
        self.buf.as_device_ptr().as_raw() as u64
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaAlma {
    module: Module,
    stream: Stream,
    _context: Context, // keep context alive for the CUDA driver
}

impl CudaAlma {
    /// Create a new `CudaAlma` on `device_id` and load the ALMA PTX module.
    pub fn new(device_id: usize) -> Result<Self, CudaAlmaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;

        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/alma_kernel.ptx"));
        let module = Module::from_ptx(ptx, &[]).map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;

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
        data_f32: &[f32],
        sweep: &AlmaBatchRange,
    ) -> Result<(Vec<AlmaParams>, usize, usize, usize), CudaAlmaError> {
        if data_f32.is_empty() {
            return Err(CudaAlmaError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaAlmaError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaAlmaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let series_len = data_f32.len();
        let max_period = combos
            .iter()
            .map(|c| c.period.unwrap_or(0))
            .max()
            .unwrap_or(0);

        if max_period == 0 || series_len - first_valid < max_period {
            return Err(CudaAlmaError::InvalidInput(format!(
                "not enough valid data (needed >= {}, valid = {})",
                max_period,
                series_len - first_valid
            )));
        }

        for prm in &combos {
            let period = prm.period.unwrap_or(0);
            let offset = prm.offset.unwrap_or(0.85);
            let sigma = prm.sigma.unwrap_or(6.0);
            if period == 0 || sigma <= 0.0 || !(0.0..=1.0).contains(&offset) {
                return Err(CudaAlmaError::InvalidInput(format!(
                    "invalid params: period={}, offset={}, sigma={}",
                    period, offset, sigma
                )));
            }
        }

        Ok((combos, first_valid, series_len, max_period))
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        combos: &[AlmaParams],
        first_valid: usize,
        series_len: usize,
        max_period: usize,
    ) -> Result<DeviceArrayF32, CudaAlmaError> {
        let n_combos = combos.len();
        let mut periods_i32 = vec![0i32; n_combos];
        let mut weights_flat = vec![0f32; n_combos * max_period];
        let mut inv_norms = vec![0f32; n_combos];

        for (idx, prm) in combos.iter().enumerate() {
            let period = prm.period.unwrap() as usize;
            let offset = prm.offset.unwrap();
            let sigma = prm.sigma.unwrap();
            let (weights, inv_norm) = compute_weights_cpu_f32(period, offset, sigma);
            periods_i32[idx] = period as i32;
            inv_norms[idx] = inv_norm;
            let base = idx * max_period;
            weights_flat[base..base + period].copy_from_slice(&weights);
        }

        let prices_bytes = series_len * std::mem::size_of::<f32>();
        let periods_bytes = n_combos * std::mem::size_of::<i32>();
        let weights_bytes = n_combos * max_period * std::mem::size_of::<f32>();
        let inv_norm_bytes = n_combos * std::mem::size_of::<f32>();
        let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + periods_bytes + weights_bytes + inv_norm_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024; // 64MB safety margin
        if !Self::will_fit(required, headroom) {
            return Err(CudaAlmaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        let d_weights = DeviceBuffer::from_slice(&weights_flat)
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        let d_inv_norms =
            DeviceBuffer::from_slice(&inv_norms).map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_prices,
            &d_weights,
            &d_periods,
            &d_inv_norms,
            series_len,
            n_combos,
            first_valid,
            max_period,
            &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_inv_norms: &DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAlmaError> {
        let use_tiled = series_len > 8192;
        let mut block_x: u32 = 256;
        let (func, shared_bytes) = if use_tiled {
            block_x = Self::pick_tiled_block(max_period, series_len, n_combos);
            let tile = block_x as usize;
            let elems = max_period + (tile + max_period - 1);
            let func_name = match block_x {
                128 => "alma_batch_tiled_f32_tile128",
                _ => "alma_batch_tiled_f32_tile256",
            };
            (
                self.module
                    .get_function(func_name)
                    .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?,
                (elems * std::mem::size_of::<f32>()) as u32,
            )
        } else {
            block_x = 256;
            (
                self.module
                    .get_function("alma_batch_f32")
                    .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?,
                (max_period * std::mem::size_of::<f32>()) as u32,
            )
        };

        let grid_x = ((series_len as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x, n_combos as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut weights_ptr = d_weights.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut inv_ptr = d_inv_norms.as_device_ptr().as_raw();
            let mut max_period_i = max_period as i32;
            let mut series_len_i = series_len as i32;
            let mut n_combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut weights_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut inv_ptr as *mut _ as *mut c_void,
                &mut max_period_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut n_combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes, args)
                .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?
        }
        Ok(())
    }

    #[inline]
    fn pick_tiled_block(max_period: usize, series_len: usize, n_combos: usize) -> u32 {
        if max_period <= 128 && series_len >= 32_768 && n_combos >= 256 {
            128
        } else {
            256
        }
    }

    /// Launch using precomputed device buffers (legacy performance path).
    pub fn alma_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_inv_norms: &DeviceBuffer<f32>,
        max_period: i32,
        series_len: i32,
        n_combos: i32,
        first_valid: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAlmaError> {
        if series_len <= 0 || n_combos <= 0 || max_period <= 0 {
            return Err(CudaAlmaError::InvalidInput(
                "series_len, n_combos, and max_period must be positive".into(),
            ));
        }
        self.launch_batch_kernel(
            d_prices,
            d_weights,
            d_periods,
            d_inv_norms,
            series_len as usize,
            n_combos as usize,
            first_valid.max(0) as usize,
            max_period as usize,
            d_out,
        )
    }

    /// Launch one-series × many-params and return a VRAM-resident handle.
    pub fn alma_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &AlmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaAlmaError> {
        let (combos, first_valid, series_len, max_period) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        self.run_batch_kernel(data_f32, &combos, first_valid, series_len, max_period)
    }

    /// Host-copy helper that writes into `out` (FP32) while returning metadata.
    pub fn alma_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &AlmaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<AlmaParams>), CudaAlmaError> {
        let (combos, first_valid, series_len, max_period) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos.len() * series_len;
        if out.len() != expected {
            return Err(CudaAlmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                expected
            )));
        }

        let arr = self.run_batch_kernel(data_f32, &combos, first_valid, series_len, max_period)?;
        arr.buf
            .copy_to(out)
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        Ok((arr.rows, arr.cols, combos))
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &AlmaParams,
    ) -> Result<(Vec<i32>, usize, f32, f32), CudaAlmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaAlmaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaAlmaError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }

        let period = params.period.unwrap_or(0);
        let offset = params.offset.unwrap_or(0.85);
        let sigma = params.sigma.unwrap_or(6.0);
        if period == 0 || sigma <= 0.0 || !(0.0..=1.0).contains(&offset) {
            return Err(CudaAlmaError::InvalidInput(format!(
                "invalid params: period={}, offset={}, sigma={}",
                period, offset, sigma
            )));
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
            let fv = fv
                .ok_or_else(|| CudaAlmaError::InvalidInput(format!("series {} all NaN", series)))?;
            if rows - fv < period {
                return Err(CudaAlmaError::InvalidInput(format!(
                    "series {} not enough valid data (needed >= {}, valid = {})",
                    series,
                    period,
                    rows - fv
                )));
            }
            first_valids[series] = fv as i32;
        }

        Ok((first_valids, period, offset as f32, sigma as f32))
    }

    fn run_many_series_kernel(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        first_valids: &[i32],
        period: usize,
        offset: f32,
        sigma: f32,
    ) -> Result<DeviceArrayF32, CudaAlmaError> {
        let prices_bytes = cols * rows * std::mem::size_of::<f32>();
        let first_valid_bytes = cols * std::mem::size_of::<i32>();
        let weights_bytes = period * std::mem::size_of::<f32>();
        let out_bytes = cols * rows * std::mem::size_of::<f32>();
        let required = prices_bytes + first_valid_bytes + weights_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaAlmaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(first_valids)
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        let (weights_host, inv_norm) = compute_weights_cpu_f32(period, offset as f64, sigma as f64);
        let d_weights = DeviceBuffer::from_slice(&weights_host)
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices,
            &d_weights,
            period,
            inv_norm,
            cols,
            rows,
            &d_first_valids,
            &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    fn launch_many_series_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        period: usize,
        inv_norm: f32,
        cols: usize,
        rows: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAlmaError> {
        const BLOCK_X: u32 = 128;
        let grid_x = ((rows as u32) + BLOCK_X - 1) / BLOCK_X;
        let grid: GridSize = (grid_x, cols as u32, 1).into();
        let block: BlockSize = (BLOCK_X, 1, 1).into();
        let shared_bytes = (period * std::mem::size_of::<f32>()) as u32;

        let func = self
            .module
            .get_function("alma_multi_series_one_param_f32")
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut weights_ptr = d_weights.as_device_ptr().as_raw();
            let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut inv = inv_norm;
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut weights_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut inv as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_valids_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes, args)
                .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?
        }
        Ok(())
    }

    /// Precomputed-weight path for many-series × one param.
    pub fn alma_multi_series_one_param_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        period: i32,
        inv_norm: f32,
        num_series: i32,
        series_len: i32,
        d_first_valids: &DeviceBuffer<i32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAlmaError> {
        if period <= 0 || num_series <= 0 || series_len <= 0 {
            return Err(CudaAlmaError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_prices,
            d_weights,
            period as usize,
            inv_norm,
            num_series as usize,
            series_len as usize,
            d_first_valids,
            d_out,
        )
    }

    /// Many-series × one-parameter (time-major). Returns a VRAM handle.
    pub fn alma_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &AlmaParams,
    ) -> Result<DeviceArrayF32, CudaAlmaError> {
        let (first_valids, period, offset, sigma) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        self.run_many_series_kernel(
            data_tm_f32,
            cols,
            rows,
            &first_valids,
            period,
            offset,
            sigma,
        )
    }

    /// Host-copy helper for many-series × one-param (FP32 output).
    pub fn alma_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &AlmaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaAlmaError> {
        if out_tm.len() != cols * rows {
            return Err(CudaAlmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out_tm.len(),
                cols * rows
            )));
        }
        let (first_valids, period, offset, sigma) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let arr = self.run_many_series_kernel(
            data_tm_f32,
            cols,
            rows,
            &first_valids,
            period,
            offset,
            sigma,
        )?;
        arr.buf
            .copy_to(out_tm)
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))
    }
}

fn expand_grid(r: &AlmaBatchRange) -> Vec<AlmaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return vec![start];
        }
        let mut v = Vec::new();
        let mut x = start;
        while x <= end + 1e-12 {
            v.push(x);
            x += step;
        }
        v
    }

    let periods = axis_usize(r.period);
    let offsets = axis_f64(r.offset);
    let sigmas = axis_f64(r.sigma);

    let mut out = Vec::with_capacity(periods.len() * offsets.len() * sigmas.len());
    for &p in &periods {
        for &o in &offsets {
            for &s in &sigmas {
                out.push(AlmaParams {
                    period: Some(p),
                    offset: Some(o),
                    sigma: Some(s),
                });
            }
        }
    }
    out
}

fn compute_weights_cpu_f32(period: usize, offset: f64, sigma: f64) -> (Vec<f32>, f32) {
    let mut weights = vec![0f32; period];
    if period == 0 {
        return (weights, 0.0);
    }
    let m = offset * (period.saturating_sub(1)) as f64;
    let s = (period as f64) / sigma;
    let s2 = 2.0 * s * s;
    let mut norm = 0.0f64;
    for i in 0..period {
        let diff = i as f64 - m;
        let w = (-((diff * diff) / s2)).exp() as f32;
        weights[i] = w;
        norm += w as f64;
    }
    let inv = if norm == 0.0 {
        0.0
    } else {
        (1.0 / norm) as f32
    };
    (weights, inv)
}
