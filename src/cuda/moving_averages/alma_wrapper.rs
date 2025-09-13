//! CUDA ALMA (Arnaud Legoux Moving Average) scaffolding using `cust`.
//!
//! This provides host-side setup that loads PTX and exposes a `CudaAlma` type with
//! an initial batch computation over a single series and multiple parameter
//! combinations. This is a development-first implementation that prioritizes
//! correctness; performance tuning and kernel improvements will follow.

#![cfg(feature = "cuda")]

use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::DeviceBuffer;
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;

use std::fmt;
use std::env;
use cust::sys as cu;

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

pub struct CudaAlma {
    module: Module,
    stream: Stream,
    _context: Context, // keep context alive (drop last)
}

impl CudaAlma {
    /// Create a new `CudaAlma` on `device_id` and load the ALMA PTX module.
    pub fn new(device_id: usize) -> Result<Self, CudaAlmaError> {
        // Initialize the CUDA driver
        cust::init(CudaFlags::empty()).map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;

        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        // Create primary context and make current
        let context = Context::new(device).map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;

        // Embed the PTX generated at build time by `build.rs`
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/alma_kernel.ptx"));
        let module = Module::from_ptx(ptx, &[]).map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;

        Ok(Self { module, stream, _context: context })
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
            if res == cu::CUresult::CUDA_SUCCESS { Some((free, total)) } else { None }
        }
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() { return true; }
        if let Some((free, _total)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            // If we cannot query memory info, proceed (no-op check)
            true
        }
    }

    /// Compute ALMA for one series across many parameter combinations on GPU.
    ///
    /// Notes:
    /// - Uses CPU-computed Gaussian weights initially, then launches the GPU dot-product kernel.
    /// - Launches with one thread per block to avoid shared-memory hazards in the current kernel.
    ///   This favors correctness first; we will improve kernel/block mapping later.
    pub fn alma_batch(
        &self,
        data: &[f64],
        sweep: &crate::indicators::moving_averages::alma::AlmaBatchRange,
    ) -> Result<crate::indicators::moving_averages::alma::AlmaBatchOutput, CudaAlmaError> {
        use crate::indicators::moving_averages::alma::{AlmaBatchOutput, AlmaParams};

        if data.is_empty() {
            return Err(CudaAlmaError::InvalidInput("empty data".into()));
        }

        // Find first valid data point
        let first_valid = data
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaAlmaError::InvalidInput("all values are NaN".into()))?;

        // Expand parameter grid
        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaAlmaError::InvalidInput("no parameter combinations".into()));
        }

        // Validate feasibility and compute CPU weights per combo
        let series_len = data.len();
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

        let n_combos = combos.len();
        let mut weights_flat = vec![0f32; n_combos * max_period];
        let mut inv_norms = vec![0f32; n_combos];
        let mut periods_i32 = vec![0i32; n_combos];

        for (idx, prm) in combos.iter().enumerate() {
            let period = prm.period.unwrap_or(0);
            let offset = prm.offset.unwrap_or(0.85);
            let sigma = prm.sigma.unwrap_or(6.0);
            if period == 0 || sigma <= 0.0 || !(0.0..=1.0).contains(&offset) {
                return Err(CudaAlmaError::InvalidInput(format!(
                    "invalid params: period={}, offset={}, sigma={}",
                    period, offset, sigma
                )));
            }
            let (w, inv) = compute_weights_cpu_f32(period, offset, sigma);
            periods_i32[idx] = period as i32;
            inv_norms[idx] = inv;
            let base = idx * max_period;
            weights_flat[base..base + period].copy_from_slice(&w);
        }

        // Estimate device memory and enforce a light guardrail before allocating
        let prices_bytes = series_len * std::mem::size_of::<f32>();
        let weights_bytes = n_combos * max_period * std::mem::size_of::<f32>();
        let periods_bytes = n_combos * std::mem::size_of::<i32>();
        let inv_bytes = n_combos * std::mem::size_of::<f32>();
        let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + weights_bytes + periods_bytes + inv_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024; // 64MB safety headroom
        if !Self::will_fit(required, headroom) {
            return Err(CudaAlmaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        // Allocate device buffers and copy (convert host data to f32)
        let mut host_prices_f32 = vec![0f32; data.len()];
        for (i, &v) in data.iter().enumerate() { host_prices_f32[i] = v as f32; }
        let d_prices = DeviceBuffer::from_slice(&host_prices_f32)
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        let d_weights = DeviceBuffer::from_slice(&weights_flat)
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        let d_inv_norms = DeviceBuffer::from_slice(&inv_norms)
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(n_combos * series_len)
        }
        .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;

        // Configure a 2D launch: Y dimension is combo index; X tiles over time indices
        const BLOCK_X: u32 = 256;
        let grid_x = ((series_len as u32) + BLOCK_X - 1) / BLOCK_X;
        let grid: GridSize = (grid_x, n_combos as u32, 1u32).into();
        let block: BlockSize = (BLOCK_X, 1u32, 1u32).into();

        // Choose kernel variant: for short series, skip tiling overhead
        let use_tiled = series_len > 8192;
        let shared_bytes: u32 = if use_tiled {
            // weights (period) + price tile (BLOCK_X + period - 1)
            (((max_period as usize) + (BLOCK_X as usize + max_period as usize - 1)) * std::mem::size_of::<f32>()) as u32
        } else {
            (max_period * std::mem::size_of::<f32>()) as u32
        };

        // Launch via Stream::new + stream.launch
        let stream = &self.stream;
        let func = if use_tiled {
            self.module
                .get_function("alma_batch_tiled_f32")
                .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?
        } else {
            self.module
                .get_function("alma_batch_f32")
                .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?
        };
        unsafe {
            // Prepare argument pointers
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut weights_ptr = d_weights.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut inv_ptr = d_inv_norms.as_device_ptr().as_raw();
            let mut max_p = max_period as i32;
            let mut series_len_i = series_len as i32;
            let mut n_combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut weights_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut inv_ptr as *mut _ as *mut c_void,
                &mut max_p as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut n_combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            stream
                .launch(&func, grid, block, shared_bytes, args)
                .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        }

        // Synchronize and copy results back
        stream
            .synchronize()
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        let mut host_out_f32 = vec![0f32; n_combos * series_len];
        d_out
            .copy_to(&mut host_out_f32)
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        let mut host_out = vec![0f64; n_combos * series_len];
        for i in 0..host_out.len() { host_out[i] = host_out_f32[i] as f64; }

        Ok(AlmaBatchOutput {
            values: host_out,
            combos,
            rows: n_combos,
            cols: series_len,
        })
    }

    /// Same as `alma_batch`, but partitions the parameter grid across `num_streams`
    /// independent CUDA streams to improve overlap and scalability for very large
    /// parameter counts. Returns the same row-major layout as `alma_batch`.
        // removed: alma_batch_multi_stream
    /// Compute ALMA for multiple series (time-major layout) with a single parameter combination.
    /// Returns a time-major Vec<f64> of length `num_series * series_len`.
    pub fn alma_multi_series_one_param_time_major(
        &self,
        data_tm: &[f64],
        num_series: usize,
        series_len: usize,
        params: &crate::indicators::moving_averages::alma::AlmaParams,
    ) -> Result<Vec<f64>, CudaAlmaError> {
        use crate::indicators::moving_averages::alma::AlmaParams;
        if num_series == 0 || series_len == 0 {
            return Err(CudaAlmaError::InvalidInput("num_series or series_len is zero".into()));
        }
        if data_tm.len() != num_series * series_len {
            return Err(CudaAlmaError::InvalidInput(format!(
                "data length {} != num_series*series_len {}",
                data_tm.len(),
                num_series * series_len
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

        // First valid index per series
        let mut first_valids = vec![0i32; num_series];
        for j in 0..num_series {
            let mut fv = None;
            for t in 0..series_len {
                let v = data_tm[t * num_series + j];
                if !v.is_nan() {
                    fv = Some(t);
                    break;
                }
            }
            let fv = fv.ok_or_else(|| CudaAlmaError::InvalidInput(format!("series {} all NaN", j)))?;
            first_valids[j] = fv as i32;
        }

        // Check feasibility: all series must have enough valid data for `period`
        for (j, &fv) in first_valids.iter().enumerate() {
            if series_len - (fv as usize) < period {
                return Err(CudaAlmaError::InvalidInput(format!(
                    "series {} not enough valid data (needed >= {}, valid = {})",
                    j,
                    period,
                    series_len - (fv as usize)
                )));
            }
        }

        // Compute weights on CPU
        let (weights, inv_norm) = compute_weights_cpu_f32(period, offset, sigma);

        // Estimate VRAM for this op and guard
        let prices_bytes = num_series * series_len * std::mem::size_of::<f32>();
        let weights_bytes = period * std::mem::size_of::<f32>();
        let first_valids_bytes = num_series * std::mem::size_of::<i32>();
        let out_bytes = num_series * series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + weights_bytes + first_valids_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024; // 64MB
        if !Self::will_fit(required, headroom) {
            return Err(CudaAlmaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        // Device buffers
        let host_tm_f32: Vec<f32> = data_tm.iter().map(|&x| x as f32).collect();
        let d_prices = DeviceBuffer::from_slice(&host_tm_f32)
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        let d_weights = DeviceBuffer::from_slice(&weights)
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(num_series * series_len)
        }
        .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;

        const BLOCK_X: u32 = 128;
        let grid_x = ((series_len as u32) + BLOCK_X - 1) / BLOCK_X;
        let grid: GridSize = (grid_x, num_series as u32, 1u32).into();
        let block: BlockSize = (BLOCK_X, 1u32, 1u32).into();
        let shared_bytes: u32 = (period * std::mem::size_of::<f32>()) as u32;

        let stream = &self.stream;
        let func = self
            .module
            .get_function("alma_multi_series_one_param_f32")
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut weights_ptr = d_weights.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut inv = inv_norm as f32;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
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
            stream
                .launch(&func, grid, block, shared_bytes, args)
                .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        }
        stream
            .synchronize()
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;

        let mut host_out_f32 = vec![0f32; num_series * series_len];
        d_out
            .copy_to(&mut host_out_f32)
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        let host_out: Vec<f64> = host_out_f32.iter().map(|&x| x as f64).collect();
        Ok(host_out)
    }

    /// FP32-only variant: many-series × one-param, time-major input, write into provided host slice (f32).
    pub fn alma_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &crate::indicators::moving_averages::alma::AlmaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaAlmaError> {
        if num_series == 0 || series_len == 0 { return Err(CudaAlmaError::InvalidInput("num_series or series_len is zero".into())); }
        if data_tm_f32.len() != num_series * series_len { return Err(CudaAlmaError::InvalidInput("data length mismatch".into())); }
        if out_tm.len() != num_series * series_len { return Err(CudaAlmaError::InvalidInput("out length mismatch".into())); }
        let period = params.period.unwrap_or(0);
        let offset = params.offset.unwrap_or(0.85);
        let sigma = params.sigma.unwrap_or(6.0);
        if period == 0 || sigma <= 0.0 || !(0.0..=1.0).contains(&offset) {
            return Err(CudaAlmaError::InvalidInput("invalid params".into()));
        }
        let mut first_valids = vec![0i32; num_series];
        for j in 0..num_series {
            let mut fv = None;
            for t in 0..series_len {
                let v = data_tm_f32[t * num_series + j];
                if !v.is_nan() { fv = Some(t); break; }
            }
            let fv = fv.ok_or_else(|| CudaAlmaError::InvalidInput(format!("series {} all NaN", j)))?;
            first_valids[j] = fv as i32;
        }
        for (j, &fv) in first_valids.iter().enumerate() {
            if series_len - (fv as usize) < period {
                return Err(CudaAlmaError::InvalidInput(format!("series {} not enough valid data", j)));
            }
        }
        let (weights, inv_norm) = compute_weights_cpu_f32(period, offset, sigma);
        let d_prices = DeviceBuffer::from_slice(data_tm_f32).map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        let d_weights = DeviceBuffer::from_slice(&weights).map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(num_series * series_len) }.map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;

        self.alma_multi_series_one_param_device(
            &d_prices, &d_weights, period as i32, inv_norm, num_series as i32, series_len as i32, &d_first_valids, &mut d_out,
        )?;
        d_out.copy_to(out_tm).map_err(|e| CudaAlmaError::Cuda(e.to_string()))
    }
}

impl CudaAlma {
    /// Launch one-series × many-params using existing device buffers (no host copies).
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
        let func = if series_len as u32 > 8192 {
            self.module
                .get_function("alma_batch_tiled_f32")
                .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?
        } else {
            self.module
                .get_function("alma_batch_f32")
                .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?
        };
        // Grid/block
        const BLOCK_X: u32 = 256;
        let grid_x = ((series_len as u32) + BLOCK_X - 1) / BLOCK_X;
        let grid: GridSize = (grid_x, n_combos as u32, 1u32).into();
        let block: BlockSize = (BLOCK_X, 1u32, 1u32).into();
        // weights (period) + price tile (BLOCK_X + period - 1)
        let shared_bytes: u32 = (((max_period as usize) + (BLOCK_X as usize + max_period as usize - 1)) * std::mem::size_of::<f32>()) as u32;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut weights_ptr = d_weights.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut inv_ptr = d_inv_norms.as_device_ptr().as_raw();
            let mut max_p = max_period;
            let mut series_len_i = series_len;
            let mut n_combos_i = n_combos;
            let mut first_valid_i = first_valid;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut weights_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut inv_ptr as *mut _ as *mut c_void,
                &mut max_p as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut n_combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes, args)
                .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))
    }

    /// Launch many-series × one-param (time-major) using existing device buffers.
    pub fn alma_multi_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        period: i32,
        inv_norm: f32,
        num_series: i32,
        series_len: i32,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAlmaError> {
        let func = self
            .module
            .get_function("alma_multi_series_one_param_f32")
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        const BLOCK_X: u32 = 256;
        let grid_x = ((series_len as u32) + BLOCK_X - 1) / BLOCK_X;
        let grid: GridSize = (grid_x, num_series as u32, 1u32).into();
        let block: BlockSize = (BLOCK_X, 1u32, 1u32).into();
        let shared_bytes: u32 = (period as usize * std::mem::size_of::<f32>()) as u32;
        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut weights_ptr = d_weights.as_device_ptr().as_raw();
            let mut period_i = period;
            let mut inv = inv_norm;
            let mut num_series_i = num_series;
            let mut series_len_i = series_len;
            let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
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
                .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))
    }

    // removed: alma_batch_device_multi_stream
}

// Helper: compute Gaussian weights and 1/sum(weights) on CPU
fn compute_weights_cpu_f32(period: usize, offset: f64, sigma: f64) -> (Vec<f32>, f32) {
    let m = (offset * (period as f64 - 1.0)) as f32;
    let s = (period as f64 / sigma) as f32;
    let s2 = 2.0f32 * s * s;
    let mut w = vec![0.0f32; period];
    let mut norm = 0.0f32;
    for i in 0..period {
        let diff = i as f32 - m;
        let wi = (-(diff * diff) / s2).exp();
        w[i] = wi;
        norm += wi;
    }
    (w, 1.0f32 / norm)
}

impl CudaAlma {
    /// FP32-only variant: compute ALMA for one series across many parameter combinations
    /// and write results directly into the provided host slice `out` (row-major: combos x series_len).
    pub fn alma_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &crate::indicators::moving_averages::alma::AlmaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<crate::indicators::moving_averages::alma::AlmaParams>), CudaAlmaError> {
        use crate::indicators::moving_averages::alma::AlmaParams;
        if data_f32.is_empty() {
            return Err(CudaAlmaError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaAlmaError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaAlmaError::InvalidInput("no parameter combinations".into()));
        }
        let series_len = data_f32.len();
        let max_period = combos.iter().map(|c| c.period.unwrap_or(0)).max().unwrap_or(0);
        if max_period == 0 || series_len - first_valid < max_period {
            return Err(CudaAlmaError::InvalidInput(format!(
                "not enough valid data (needed >= {}, valid = {})",
                max_period,
                series_len - first_valid
            )));
        }
        let n_combos = combos.len();
        if out.len() != n_combos * series_len {
            return Err(CudaAlmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(), n_combos * series_len
            )));
        }

        let mut weights_flat = vec![0f32; n_combos * max_period];
        let mut inv_norms = vec![0f32; n_combos];
        let mut periods_i32 = vec![0i32; n_combos];
        for (idx, prm) in combos.iter().enumerate() {
            let period = prm.period.unwrap_or(0);
            let offset = prm.offset.unwrap_or(0.85);
            let sigma = prm.sigma.unwrap_or(6.0);
            if period == 0 || sigma <= 0.0 || !(0.0..=1.0).contains(&offset) {
                return Err(CudaAlmaError::InvalidInput(format!(
                    "invalid params: period={}, offset={}, sigma={}",
                    period, offset, sigma
                )));
            }
            let (w, inv) = compute_weights_cpu_f32(period, offset, sigma);
            periods_i32[idx] = period as i32;
            inv_norms[idx] = inv;
            let base = idx * max_period;
            weights_flat[base..base + period].copy_from_slice(&w);
        }

        // Estimate device memory and guard
        let prices_bytes = series_len * std::mem::size_of::<f32>();
        let weights_bytes = n_combos * max_period * std::mem::size_of::<f32>();
        let periods_bytes = n_combos * std::mem::size_of::<i32>();
        let inv_bytes = n_combos * std::mem::size_of::<f32>();
        let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + weights_bytes + periods_bytes + inv_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaAlmaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        // Device buffers
        let d_prices = DeviceBuffer::from_slice(data_f32)
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        let d_weights = DeviceBuffer::from_slice(&weights_flat)
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        let d_inv_norms = DeviceBuffer::from_slice(&inv_norms)
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;

        // Launch
        self.alma_batch_device(
            &d_prices,
            &d_weights,
            &d_periods,
            &d_inv_norms,
            max_period as i32,
            series_len as i32,
            n_combos as i32,
            first_valid as i32,
            &mut d_out,
        )?;

        // Copy into provided out
        d_out.copy_to(out).map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        Ok((n_combos, series_len, combos))
    }

    // removed: alma_batch_multi_stream_into_host_f32
}

// Local expansion of AlmaBatchRange into individual param combos (copied pattern)
fn expand_grid(
    r: &crate::indicators::moving_averages::alma::AlmaBatchRange,
) -> Vec<crate::indicators::moving_averages::alma::AlmaParams> {
    use crate::indicators::moving_averages::alma::AlmaParams;

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
