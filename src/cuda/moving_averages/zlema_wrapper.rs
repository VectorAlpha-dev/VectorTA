//! CUDA scaffolding for ZLEMA (Zero Lag Exponential Moving Average).
//!
//! Mirrors the VRAM-first design used by the ALMA integration: inputs are
//! converted to `f32`, parameter sweeps are validated on the host, and each
//! parameter combination is evaluated on the device, returning a `DeviceArrayF32`.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::zlema::{expand_grid_zlema, ZlemaBatchRange, ZlemaParams};
use cust::context::{CacheConfig, Context};
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
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
    #[inline]
    fn env_flag(name: &str, default: bool) -> bool {
        match env::var(name) {
            Ok(v) => {
                let v = v.to_ascii_lowercase();
                matches!(v.as_str(), "1" | "true" | "yes" | "on")
            }
            Err(_) => default,
        }
    }
    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }

    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> { cust::memory::mem_get_info().ok() }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() { return true; }
        if let Some((free, _total)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else { true }
    }
    pub fn new(device_id: usize) -> Result<Self, CudaZlemaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaZlemaError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaZlemaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaZlemaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/zlema_kernel.ptx"));
        // Keep JIT at highest optimization (O4) by default.
        // We rely on the default being O4; only set DetermineTargetFromContext here.
        let jit_opts = &[ModuleJitOption::DetermineTargetFromContext];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                    m
                } else {
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaZlemaError::Cuda(e.to_string()))?
                }
            }
        };
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
        let mut func = self
            .module
            .get_function("zlema_batch_f32")
            .map_err(|e| CudaZlemaError::Cuda(e.to_string()))?;

        // Prefer L1 for memory-bound kernels unless user turns it off
        if Self::env_flag("ZLEMA_PREFER_L1", true) {
            let _ = func.set_cache_config(CacheConfig::PreferL1);
        }

        // Optional manual override; otherwise use occupancy suggestion
        let block_x_override = env::var("ZLEMA_BATCH_BLOCK_X")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .filter(|&v| v > 0);

        let (grid, block): (GridSize, BlockSize) = if let Some(bx) = block_x_override {
            let grid_x = ((n_combos as u32) + bx - 1) / bx;
            ((grid_x.max(1), 1, 1).into(), (bx, 1, 1).into())
        } else {
            let (min_grid, block_size) = func
                .suggested_launch_configuration(0, (0, 0, 0).into())
                .map_err(|e| CudaZlemaError::Cuda(e.to_string()))?;
            let bx = block_size.clamp(64, 1024);
            let grid_x = ((n_combos as u32) + bx - 1) / bx;
            let gx = grid_x.max(min_grid);
            ((gx.max(1), 1, 1).into(), (bx, 1, 1).into())
        };

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

    // Tiled batch kernel launcher (uses dynamic shared memory based on tile + max_lag)
    fn launch_batch_kernel_tiled(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_lags: &DeviceBuffer<i32>,
        d_alphas: &DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        max_lag: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaZlemaError> {
        let mut func = self
            .module
            .get_function("zlema_batch_f32_tiled_f32")
            .map_err(|e| CudaZlemaError::Cuda(e.to_string()))?;

        if Self::env_flag("ZLEMA_PREFER_L1", true) {
            let _ = func.set_cache_config(CacheConfig::PreferL1);
        }

        // Must match kernel default unless overridden at compile time.
        let tile: usize = env::var("ZLEMA_BATCH_TILE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(1024);
        let shmem_bytes = (tile + (max_lag as usize)) * std::mem::size_of::<f32>();

        // Optional manual override; otherwise occupancy suggestion that accounts for shmem
        let block_x_override = env::var("ZLEMA_BATCH_BLOCK_X")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .filter(|&v| v > 0);
        let (grid, block): (GridSize, BlockSize) = if let Some(bx) = block_x_override {
            let grid_x = ((n_combos as u32) + bx - 1) / bx;
            ((grid_x.max(1), 1, 1).into(), (bx, 1, 1).into())
        } else {
            let (min_grid, block_size) = func
                .suggested_launch_configuration(shmem_bytes, (0, 0, 0).into())
                .map_err(|e| CudaZlemaError::Cuda(e.to_string()))?;
            let bx = block_size.clamp(64, 1024);
            let grid_x = ((n_combos as u32) + bx - 1) / bx;
            let gx = grid_x.max(min_grid);
            ((gx.max(1), 1, 1).into(), (bx, 1, 1).into())
        };

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut lags_ptr = d_lags.as_device_ptr().as_raw();
            let mut alphas_ptr = d_alphas.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut max_lag_i = max_lag as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut lags_ptr as *mut _ as *mut c_void,
                &mut alphas_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut max_lag_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream
                .launch(&func, grid, block, shmem_bytes as u32, args)
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
        // VRAM estimate and guard (host inputs + params + outputs)
        let rows = combos.len();
        let bytes_required = len * std::mem::size_of::<f32>()    // prices
            + rows * std::mem::size_of::<i32>()                  // periods
            + rows * std::mem::size_of::<i32>()                  // lags
            + rows * std::mem::size_of::<f32>()                  // alphas
            + rows * len * std::mem::size_of::<f32>();           // outputs
        let headroom = env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        if !Self::will_fit(bytes_required, headroom) {
            return Err(CudaZlemaError::InvalidInput(format!(
                "insufficient VRAM: need ~{} MB (incl headroom)",
                (bytes_required + headroom) / (1024 * 1024)
            )));
        }
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

        // Heuristic: use tiled kernel when sweeping many combos or long series.
        // Default thresholds chosen conservatively; refine with benches if needed.
        let n_combos = combos.len();
        let max_lag = *lags.iter().max().unwrap_or(&0);
        // Prefer tiled only when both the series is reasonably long and there
        // are many parameter rows to amortize the shared-memory stage.
        // This avoids overhead on small sweeps and keeps unit-test defaults intact.
        let use_tiled = (n_combos >= 64) && (len >= 4096);

        if use_tiled {
            self.launch_batch_kernel_tiled(
                &d_prices,
                &d_periods,
                &d_lags,
                &d_alphas,
                len,
                first_valid,
                n_combos,
                max_lag,
                &mut d_out,
            )?;
        } else {
            self.launch_batch_kernel(
                &d_prices,
                &d_periods,
                &d_lags,
                &d_alphas,
                len,
                first_valid,
                n_combos,
                &mut d_out,
            )?;
        }

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
        // Stage Device -> Host into pinned memory asynchronously, then memcpy to out
        let mut pinned = unsafe {
            LockedBuffer::<f32>::uninitialized(expected)
                .map_err(|e| CudaZlemaError::Cuda(e.to_string()))?
        };
        unsafe {
            dev.buf
                .async_copy_to(&mut pinned.as_mut_slice(), &self.stream)
                .map_err(|e| CudaZlemaError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaZlemaError::Cuda(e.to_string()))?;
        out.copy_from_slice(pinned.as_slice());
        Ok((combos.len(), len, combos))
    }

    // ---------- Many-series (time-major) one-param ----------

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &ZlemaParams,
    ) -> Result<(Vec<i32>, usize, f32), CudaZlemaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaZlemaError::InvalidInput(
                "series dimensions must be positive".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaZlemaError::InvalidInput(format!(
                "data length mismatch: expected {}, got {}",
                cols * rows,
                data_tm_f32.len()
            )));
        }

        let period = params.period.unwrap_or(0);
        if period == 0 {
            return Err(CudaZlemaError::InvalidInput(
                "period must be at least 1".into(),
            ));
        }
        if period > rows {
            return Err(CudaZlemaError::InvalidInput(format!(
                "period {} exceeds series length {}",
                period, rows
            )));
        }

        // First-valid per series (time-major layout)
        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut fv: Option<usize> = None;
            for row in 0..rows {
                let idx = row * cols + series;
                if !data_tm_f32[idx].is_nan() {
                    fv = Some(row);
                    break;
                }
            }
            let fvu = fv.ok_or_else(|| CudaZlemaError::InvalidInput(format!(
                "series {} all NaN",
                series
            )))?;
            if rows - fvu < period {
                return Err(CudaZlemaError::InvalidInput(format!(
                    "series {} insufficient data for period {} (tail = {})",
                    series,
                    period,
                    rows - fvu
                )));
            }
            first_valids[series] = fvu as i32;
        }

        let alpha = 2.0f32 / (period as f32 + 1.0f32);
        Ok((first_valids, period, alpha))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        alpha: f32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaZlemaError> {
        let mut func = self
            .module
            .get_function("zlema_many_series_one_param_f32")
            .map_err(|e| CudaZlemaError::Cuda(e.to_string()))?;

        // Prefer L1 for memory-bound kernels unless disabled
        if Self::env_flag("ZLEMA_PREFER_L1", true) {
            let _ = func.set_cache_config(CacheConfig::PreferL1);
        }

        // Optional override or occupancy-guided block size
        let block_x_override = env::var("ZLEMA_MS_BLOCK_X")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .filter(|&v| v > 0);
        let (grid, block): (GridSize, BlockSize) = if let Some(bx) = block_x_override {
            let grid_x = ((cols as u32) + bx - 1) / bx;
            ((grid_x.max(1), 1, 1).into(), (bx, 1, 1).into())
        } else {
            let (min_grid, block_size) = func
                .suggested_launch_configuration(0, (0, 0, 0).into())
                .map_err(|e| CudaZlemaError::Cuda(e.to_string()))?;
            let bx = block_size.clamp(64, 1024);
            let grid_x = ((cols as u32) + bx - 1) / bx;
            let gx = grid_x.max(min_grid);
            ((gx.max(1), 1, 1).into(), (bx, 1, 1).into())
        };

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut period_i = period as i32;
            let mut alpha_f = alpha as f32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut alpha_f as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaZlemaError::Cuda(e.to_string()))?;
        }

        Ok(())
    }

    fn run_many_series_kernel(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        first_valids: &[i32],
        period: usize,
        alpha: f32,
    ) -> Result<DeviceArrayF32, CudaZlemaError> {
        // Optional VRAM check similar to SMA wrapper
        let elems = cols * rows;
        let bytes_required = elems * std::mem::size_of::<f32>() // prices
            + cols * std::mem::size_of::<i32>()                 // first_valids
            + elems * std::mem::size_of::<f32>();               // outputs
        let headroom = env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        if !Self::will_fit(bytes_required, headroom) {
            return Err(CudaZlemaError::InvalidInput(format!(
                "insufficient VRAM: need ~{} MB (incl headroom)",
                (bytes_required + headroom) / (1024 * 1024)
            )));
        }
        let d_prices =
            DeviceBuffer::from_slice(data_tm_f32).map_err(|e| CudaZlemaError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(first_valids)
            .map_err(|e| CudaZlemaError::Cuda(e.to_string()))?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }
            .map_err(|e| CudaZlemaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(&d_prices, &d_first, cols, rows, period, alpha, &mut d_out)?;

        Ok(DeviceArrayF32 { buf: d_out, rows, cols })
    }

    pub fn zlema_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &ZlemaParams,
    ) -> Result<DeviceArrayF32, CudaZlemaError> {
        let (first_valids, p, alpha) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, p, alpha)
    }

    pub fn zlema_many_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &ZlemaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaZlemaError> {
        if out_tm.len() != cols * rows {
            return Err(CudaZlemaError::InvalidInput(format!(
                "output length mismatch: expected {}, got {}",
                cols * rows,
                out_tm.len()
            )));
        }
        let (first_valids, p, alpha) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let dev = self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, p, alpha)?;
        // Device -> Host via pinned buffer + async copy
        let mut pinned = unsafe {
            LockedBuffer::<f32>::uninitialized(cols * rows)
                .map_err(|e| CudaZlemaError::Cuda(e.to_string()))?
        };
        unsafe {
            dev.buf
                .async_copy_to(&mut pinned.as_mut_slice(), &self.stream)
                .map_err(|e| CudaZlemaError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaZlemaError::Cuda(e.to_string()))?;
        out_tm.copy_from_slice(pinned.as_slice());
        Ok(())
    }
}

// ---------- Bench profiles (batch-only) ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        zlema_benches,
        CudaZlema,
        crate::indicators::moving_averages::zlema::ZlemaBatchRange,
        crate::indicators::moving_averages::zlema::ZlemaParams,
        zlema_batch_dev,
        zlema_many_series_one_param_time_major_dev,
        crate::indicators::moving_averages::zlema::ZlemaBatchRange { period: (10, 10 + PARAM_SWEEP - 1, 1) },
        crate::indicators::moving_averages::zlema::ZlemaParams { period: Some(64) },
        "zlema",
        "zlema"
    );
    pub use zlema_benches::bench_profiles;
}
