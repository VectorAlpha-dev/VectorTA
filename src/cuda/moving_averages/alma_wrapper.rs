//! CUDA scaffolding for the ALMA (Arnaud Legoux Moving Average) kernels.
//!
//! VRAM-first design with optional on-device weight generation, async copies,
//! and chunked launches that respect grid limits. Falls back to the legacy
//! precomputed-weights kernels if on-device variants are not present.
//!
//! Kernel symbols expected (any missing ones are auto-fallbacks to the others):
//! - "alma_batch_f32_ondev"                          // one-series × many-params, weights computed on device
//! - "alma_batch_f32"                                // one-series × many-params, precomputed weights
//! - "alma_batch_tiled_f32_tile128" / "..._tile256"  // tiled variant for long series, precomputed weights
//! - "alma_multi_series_one_param_f32_ondev"         // many-series × one-param, weights on device
//! - "alma_multi_series_one_param_f32"               // many-series × one-param, precomputed weights
//!
//! Notes:
//! - Async allocations/copies use the stream-ordered allocator and pinned
//!   host buffers for throughput.
//! - Grid-Y is chunked to <= 65_535 to avoid launch limit.
//! - Uses occupancy-based suggestion for non-tiled batch kernel.
//!
//! This file is a drop-in replacement for the previous wrapper.

#![cfg(feature = "cuda")]

use crate::indicators::moving_averages::alma::{AlmaBatchRange, AlmaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, DeviceBuffer, LockedBuffer};
use cust::memory::AsyncCopyDestination; // for async_copy_to on DeviceBuffer
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
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
    _context: Context, // keep context alive
    device_id: u32,
}

impl CudaAlma {
    /// Create a new `CudaAlma` on `device_id` and load the ALMA PTX module.
    pub fn new(device_id: usize) -> Result<Self, CudaAlmaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/alma_kernel.ptx"));
        // High optimization, target from current context
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O4),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
        })
    }

    /// Expose a simple synchronize for callers (e.g., benches) that manage
    /// their own device buffers and want deterministic timing.
    pub fn synchronize(&self) -> Result<(), CudaAlmaError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))
    }

    // ---------- Utilities ----------

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }

    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
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

    #[inline]
    fn pick_tiled_block(&self, max_period: usize, series_len: usize, n_combos: usize) -> u32 {
        // Env override: ALMA_TILE=128|256|512
        if let Ok(v) = std::env::var("ALMA_TILE") {
            if let Ok(tile) = v.parse::<u32>() {
                let name = match tile {
                    128 => Some("alma_batch_tiled_f32_tile128"),
                    256 => Some("alma_batch_tiled_f32_tile256"),
                    512 => Some("alma_batch_tiled_f32_tile512"),
                    _ => None,
                };
                if let Some(fname) = name {
                    if self.module.get_function(fname).is_ok() {
                        return tile;
                    }
                }
            }
        }

        // Heuristic: choose tile to maintain enough CTAs relative to SM count
        let sm_count = Device::get_device(self.device_id)
            .ok()
            .and_then(|d| d.get_attribute(cust::device::DeviceAttribute::MultiprocessorCount).ok())
            .unwrap_or(64) as u32;

        let candidates: [u32; 3] = [256, 128, 512];
        let mut best = 256u32;
        for &tile in &candidates {
            // Only consider tiles that have a compiled kernel symbol
            let fname = match tile {
                128 => "alma_batch_tiled_f32_tile128",
                256 => "alma_batch_tiled_f32_tile256",
                512 => "alma_batch_tiled_f32_tile512",
                _ => continue,
            };
            if self.module.get_function(fname).is_err() {
                continue;
            }

            let grid_x = ((series_len as u32) + tile - 1) / tile;
            let grid_y = n_combos as u32;
            let ctas = grid_x.saturating_mul(grid_y.min(65_535));

            // If CTAs < 2*SMs, prefer smaller tile to increase grid density
            let enough_density = ctas >= 2 * sm_count;
            if !enough_density && tile > 128 {
                best = 128;
                continue;
            }

            // Favor larger tile when period small to reduce per-CTA overhead
            if max_period <= 64 && tile >= best {
                best = tile;
            } else if best == 0 {
                best = tile;
            }
        }
        if best == 0 { 256 } else { best }
    }

    #[inline]
    fn grid_y_chunks(n_combos: usize) -> impl Iterator<Item = (usize, usize)> {
        const MAX_GRID_Y: usize = 65_535;
        (0..n_combos)
            .step_by(MAX_GRID_Y)
            .map(move |start| {
                let len = (n_combos - start).min(MAX_GRID_Y);
                (start, len)
            })
    }

    // ---------- Host helpers ----------

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

    fn prepare_batch_inputs_device(
        series_len: usize,
        first_valid: usize,
        sweep: &AlmaBatchRange,
    ) -> Result<(Vec<AlmaParams>, usize), CudaAlmaError> {
        if series_len == 0 {
            return Err(CudaAlmaError::InvalidInput("series_len is zero".into()));
        }
        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaAlmaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
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
        Ok((combos, max_period))
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &AlmaParams,
    ) -> Result<(Vec<i32>, usize, f32, f32), CudaAlmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaAlmaError::InvalidInput(
                "cols or rows is zero".into(),
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

    // ---------- Public API: one-series × many-params ----------

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
        self.launch_batch_kernel_precomputed(
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

    /// Host input → VRAM output (on-device weights if available, else legacy).
    pub fn alma_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &AlmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaAlmaError> {
        let (combos, first_valid, series_len, max_period) =
            Self::prepare_batch_inputs(data_f32, sweep)?;

        let prices_bytes = series_len * std::mem::size_of::<f32>();
        // Upper bound for legacy path; on-device path avoids weight traffic
        let weights_bytes = combos.len() * max_period * std::mem::size_of::<f32>();
        let out_bytes = combos.len() * series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + weights_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024; // 64MB safety
        if !Self::will_fit(required, headroom) {
            return Err(CudaAlmaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        // H2D async copy for prices
        let d_prices = unsafe {
            DeviceBuffer::from_slice_async(data_f32, &self.stream)
                .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?
        };

        self.run_batch_with_prices_device(&d_prices, series_len, first_valid, &combos, max_period)
    }

    /// Device input → VRAM output, avoids host copies of prices.
    /// Caller supplies `first_valid` computed elsewhere.
    pub fn alma_batch_from_device_prices(
        &self,
        d_prices: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        sweep: &AlmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaAlmaError> {
        let (combos, max_period) =
            Self::prepare_batch_inputs_device(series_len, first_valid, sweep)?;
        self.run_batch_with_prices_device(d_prices, series_len, first_valid, &combos, max_period)
    }

    fn run_batch_with_prices_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        combos: &[AlmaParams],
        max_period: usize,
    ) -> Result<DeviceArrayF32, CudaAlmaError> {
        let n_combos = combos.len();

        // Output buffer
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(n_combos * series_len, &self.stream)
                .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?
        };

        // Prefer precomputed host weights for large sweeps; allow opt-in to on-device
        // weights via env ALMA_BATCH_ONDEV=1.
        let has_ondev = self
            .module
            .get_function("alma_batch_f32_ondev")
            .or_else(|_| self.module.get_function("alma_batch_f32_onthefly"))
            .is_ok();
        let has_pre = self
            .module
            .get_function("alma_batch_f32")
            .is_ok()
            || self
                .module
                .get_function("alma_batch_tiled_f32_tile128")
                .is_ok()
            || self
                .module
                .get_function("alma_batch_tiled_f32_tile256")
                .is_ok();

        let env_force_ondev = matches!(
            std::env::var("ALMA_BATCH_ONDEV"),
            Ok(ref v) if v == "1" || v.eq_ignore_ascii_case("true")
        );

        // Heuristic: on-device weight generation is only beneficial for very small
        // combo counts; otherwise precomputing once on host is faster.
        let prefer_ondev = env_force_ondev || (n_combos <= 16);

        if has_pre && (!has_ondev || !prefer_ondev) {
            // Legacy: build CPU weights + inv_norms per-combo
            let mut periods_i32 = vec![0i32; n_combos];
            let mut inv_norms = vec![0f32; n_combos];
            let mut weights_flat = vec![0f32; n_combos * max_period];

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

            let d_weights = DeviceBuffer::from_slice(&weights_flat)
                .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
            let d_periods = DeviceBuffer::from_slice(&periods_i32)
                .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
            let d_inv_norms = DeviceBuffer::from_slice(&inv_norms)
                .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;

            self.launch_batch_kernel_precomputed(
                d_prices,
                &d_weights,
                &d_periods,
                &d_inv_norms,
                series_len,
                n_combos,
                first_valid,
                max_period,
                &mut d_out,
            )?;
        } else if has_ondev {
            self.launch_batch_kernel_ondev(
                d_prices,
                combos,
                series_len,
                n_combos,
                first_valid,
                max_period,
                &mut d_out,
            )?;
        } else {
            return Err(CudaAlmaError::NotImplemented);
        }

        // Ensure completion for VRAM handle consistency.
        self.stream
            .synchronize()
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
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
        if out.len() != combos.len() * series_len {
            return Err(CudaAlmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                combos.len() * series_len
            )));
        }
        let arr = self.run_batch_with_prices_device(
            &unsafe {
                DeviceBuffer::from_slice_async(data_f32, &self.stream)
                    .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?
            },
            series_len,
            first_valid,
            &combos,
            max_period,
        )?;

        // Pinned buffer for faster D2H
        let mut pinned: LockedBuffer<f32> = unsafe {
            LockedBuffer::uninitialized(arr.len()).map_err(|e| CudaAlmaError::Cuda(e.to_string()))?
        };
        unsafe {
            arr.buf
                .async_copy_to(pinned.as_mut_slice(), &self.stream)
                .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        out.copy_from_slice(pinned.as_slice());
        Ok((arr.rows, arr.cols, combos))
    }

    // ---------- Public API: many-series × one-parameter ----------

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

        let prices_bytes = cols * rows * std::mem::size_of::<f32>();
        let weights_bytes = period * std::mem::size_of::<f32>(); // legacy upper bound
        let out_bytes = cols * rows * std::mem::size_of::<f32>();
        let required = prices_bytes + weights_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaAlmaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices = unsafe {
            DeviceBuffer::from_slice_async(data_tm_f32, &self.stream)
                .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?
        };
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;

        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(cols * rows, &self.stream) }
                .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;

        if self
            .module
            .get_function("alma_multi_series_one_param_f32_ondev")
            .is_ok()
        {
            self.launch_many_series_kernel_ondev(
                &d_prices,
                period,
                offset,
                sigma,
                cols,
                rows,
                &d_first_valids,
                &mut d_out,
            )?;
        } else {
            // Legacy: precompute weights on host
            let (weights_host, inv_norm) = compute_weights_cpu_f32(period, offset as f64, sigma as f64);
            let d_weights = DeviceBuffer::from_slice(&weights_host)
                .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
            self.launch_many_series_kernel_precomputed(
                &d_prices,
                &d_weights,
                period,
                inv_norm,
                cols,
                rows,
                &d_first_valids,
                &mut d_out,
            )?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 { buf: d_out, rows, cols })
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
        let arr = self.alma_multi_series_one_param_time_major_dev(data_tm_f32, cols, rows, params)?;
        let mut pinned: LockedBuffer<f32> = unsafe {
            LockedBuffer::uninitialized(arr.len()).map_err(|e| CudaAlmaError::Cuda(e.to_string()))?
        };
        unsafe {
            arr.buf
                .async_copy_to(pinned.as_mut_slice(), &self.stream)
                .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        out_tm.copy_from_slice(pinned.as_slice());
        Ok(())
    }

    // ---------- Kernel launches (internal) ----------

    fn launch_batch_kernel_ondev(
        &self,
        d_prices: &DeviceBuffer<f32>,
        combos: &[AlmaParams],
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAlmaError> {
        // Compact params
        let periods: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let offsets: Vec<f32> = combos.iter().map(|c| c.offset.unwrap() as f32).collect();
        let sigmas: Vec<f32> = combos.iter().map(|c| c.sigma.unwrap() as f32).collect();

        let d_periods = DeviceBuffer::from_slice(&periods)
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        let d_offsets = DeviceBuffer::from_slice(&offsets)
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        let d_sigmas =
            DeviceBuffer::from_slice(&sigmas).map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;

        let func = self
            .module
            .get_function("alma_batch_f32_ondev")
            .or_else(|_| self.module.get_function("alma_batch_f32_onthefly"))
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;

        // Shared memory needed for weights (dynamic) ~ max_period * sizeof(f32)
        let shared_bytes = (max_period * std::mem::size_of::<f32>()) as u32;

        // Ask occupancy for a good block size
        let (_, suggested_block) = func
            .suggested_launch_configuration(shared_bytes as usize, BlockSize::xyz(0, 0, 0))
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        let block_x = if suggested_block > 0 { suggested_block } else { 256 };

        // Chunk over grid.y if needed
        for (start, len) in Self::grid_y_chunks(n_combos) {
            let grid_x = ((series_len as u32) + block_x - 1) / block_x;
            let grid: GridSize = (grid_x, len as u32, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();

            // Offset output pointer by start * series_len elements
            let out_ptr = unsafe { d_out.as_device_ptr().add(start * series_len) };

            let stream = &self.stream;
            unsafe {
                launch!(
                    func<<<grid, block, shared_bytes, stream>>>(
                        d_prices.as_device_ptr(),
                        d_periods.as_device_ptr(),
                        d_offsets.as_device_ptr(),
                        d_sigmas.as_device_ptr(),
                        series_len as i32,
                        len as i32,
                        (first_valid as i32),
                        out_ptr
                    )
                )
                .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
            }
        }

        Ok(())
    }

    fn launch_batch_kernel_precomputed(
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
        // Decide between plain and tiled kernels
        let use_tiled = series_len > 8192;
        let mut block_x: u32 = 256;

        if use_tiled {
            block_x = self.pick_tiled_block(max_period, series_len, n_combos);
            let tile = block_x as usize; // outputs per block
            // Automatic 2x-per-thread selection for larger periods if kernel exists
            let two_x_name = match block_x {
                128 => "alma_batch_tiled_f32_2x_tile128",
                256 => "alma_batch_tiled_f32_2x_tile256",
                512 => "alma_batch_tiled_f32_2x_tile512",
                _ => "",
            };
            let two_x_available = !two_x_name.is_empty() && self.module.get_function(two_x_name).is_ok();
            // Heuristic: enable 2x when period is moderately large; adjust as needed.
            let use_two_per_thread = two_x_available && max_period >= 96;
            let threads_x = if use_two_per_thread { (block_x / 2).max(1) } else { block_x };
            // Dynamic shared: weights + tile_overlap buffer sized by outputs per block
            let elems = max_period + (tile + max_period - 1);
            let shared_bytes = (elems * std::mem::size_of::<f32>()) as u32;

            let func_name = if use_two_per_thread {
                if block_x == 128 {
                    "alma_batch_tiled_f32_2x_tile128"
                } else if block_x == 256 {
                    "alma_batch_tiled_f32_2x_tile256"
                } else if block_x == 512 {
                    "alma_batch_tiled_f32_2x_tile512"
                } else { "alma_batch_tiled_f32_2x_tile256" }
            } else {
                if block_x == 128 {
                    "alma_batch_tiled_f32_tile128"
                } else if block_x == 256 {
                    "alma_batch_tiled_f32_tile256"
                } else if block_x == 512 {
                    "alma_batch_tiled_f32_tile512"
                } else { "alma_batch_tiled_f32_tile256" }
            };
            let func = self
                .module
                .get_function(func_name)
                .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;

            for (start, len) in Self::grid_y_chunks(n_combos) {
                let grid_x = ((series_len as u32) + (block_x - 1)) / block_x; // divide by outputs per block
                let grid: GridSize = (grid_x, len as u32, 1).into();
                let block: BlockSize = (threads_x, 1, 1).into();
                let out_ptr = unsafe { d_out.as_device_ptr().add(start * series_len) };

                let stream = &self.stream;
                unsafe {
                    launch!(
                        func<<<grid, block, shared_bytes, stream>>>(
                            d_prices.as_device_ptr(),
                            d_weights.as_device_ptr(),
                            d_periods.as_device_ptr(),
                            d_inv_norms.as_device_ptr(),
                            (max_period as i32),
                            (series_len as i32),
                            (len as i32),
                            (first_valid as i32),
                            out_ptr
                        )
                    )
                    .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
                }
            }
        } else {
            // Plain kernel: dynamic shared = weights only
            let shared_bytes = (max_period * std::mem::size_of::<f32>()) as u32;
            let func = self
                .module
                .get_function("alma_batch_f32")
                .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;

            // Fixed block size is usually better for bandwidth-bound 1D loops; allow override.
            block_x = match std::env::var("ALMA_BLOCK_X").ok().and_then(|s| s.parse::<u32>().ok()) {
                Some(v) if v == 128 || v == 256 || v == 512 => v,
                _ => 256,
            };

            for (start, len) in Self::grid_y_chunks(n_combos) {
                let grid_x = ((series_len as u32) + block_x - 1) / block_x;
                let grid: GridSize = (grid_x, len as u32, 1).into();
                let block: BlockSize = (block_x, 1, 1).into();
                let out_ptr = unsafe { d_out.as_device_ptr().add(start * series_len) };

                let stream = &self.stream;
                unsafe {
                    launch!(
                        func<<<grid, block, shared_bytes, stream>>>(
                            d_prices.as_device_ptr(),
                            d_weights.as_device_ptr(),
                            d_periods.as_device_ptr(),
                            d_inv_norms.as_device_ptr(),
                            (max_period as i32),
                            (series_len as i32),
                            (len as i32),
                            (first_valid as i32),
                            out_ptr
                        )
                    )
                    .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
                }
            }
        }

        Ok(())
    }

    fn launch_many_series_kernel_ondev(
        &self,
        d_prices: &DeviceBuffer<f32>,
        period: usize,
        offset: f32,
        sigma: f32,
        cols: usize,
        rows: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAlmaError> {
        let func = self
            .module
            .get_function("alma_multi_series_one_param_f32_ondev")
            .or_else(|_| self.module.get_function("alma_multi_series_one_param_onthefly_f32"))
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;

        // Shared memory for per-block weights
        let shared_bytes = (period * std::mem::size_of::<f32>()) as u32;

        // Fixed mapping: block.x over time, grid.y over series
        const BLOCK_X: u32 = 128;
        let grid_x = ((rows as u32) + BLOCK_X - 1) / BLOCK_X;
        let grid: GridSize = (grid_x, cols as u32, 1).into();
        let block: BlockSize = (BLOCK_X, 1, 1).into();

        let stream = &self.stream;
        unsafe {
            launch!(
                func<<<grid, block, shared_bytes, stream>>>(
                    d_prices.as_device_ptr(),
                    (period as i32),
                    offset,
                    sigma,
                    (cols as i32),
                    (rows as i32),
                    d_first_valids.as_device_ptr(),
                    d_out.as_device_ptr()
                )
            )
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    fn launch_many_series_kernel_precomputed(
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
        let func = self
            .module
            .get_function("alma_multi_series_one_param_f32")
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;

        // Shared memory for weights
        let shared_bytes = (period * std::mem::size_of::<f32>()) as u32;

        const BLOCK_X: u32 = 128;
        let grid_x = ((rows as u32) + BLOCK_X - 1) / BLOCK_X;
        let grid: GridSize = (grid_x, cols as u32, 1).into();
        let block: BlockSize = (BLOCK_X, 1, 1).into();

        let stream = &self.stream;
        unsafe {
            launch!(
                func<<<grid, block, shared_bytes, stream>>>(
                    d_prices.as_device_ptr(),
                    d_weights.as_device_ptr(),
                    (period as i32),
                    inv_norm,
                    (cols as i32),
                    (rows as i32),
                    d_first_valids.as_device_ptr(),
                    d_out.as_device_ptr()
                )
            )
            .map_err(|e| CudaAlmaError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    /// Public: many-series × one-param with device-side inputs and precomputed weights.
    pub fn alma_many_series_one_param_time_major_device_precomputed(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        period: usize,
        inv_norm: f32,
        cols: usize,
        rows: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAlmaError> {
        if cols == 0 || rows == 0 || period == 0 {
            return Err(CudaAlmaError::InvalidInput(
                "cols, rows, and period must be positive".into(),
            ));
        }
        self.launch_many_series_kernel_precomputed(
            d_prices_tm,
            d_weights,
            period,
            inv_norm,
            cols,
            rows,
            d_first_valids,
            d_out_tm,
        )
    }
}

// ---------- Grid expansion + CPU weights (fallback) ----------

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
    let inv = if norm == 0.0 { 0.0 } else { (1.0 / norm) as f32 };
    (weights, inv)
}

// ---------- Bench profiles (wrapper-owned) ----------

pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;
    const MANY_SERIES_COLS: usize = 250;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        // small overhead for params + 64MB headroom
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = elems * std::mem::size_of::<f32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    // Preallocated, device-resident batch state to avoid per-iteration allocs/copies.
    struct AlmaBatchDeviceState {
        cuda: CudaAlma,
        d_prices: DeviceBuffer<f32>,
        d_weights: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_inv_norms: DeviceBuffer<f32>,
        d_out: DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        first_valid: usize,
    }

    impl CudaBenchState for AlmaBatchDeviceState {
        fn launch(&mut self) {
            self.cuda
                .alma_batch_device(
                    &self.d_prices,
                    &self.d_weights,
                    &self.d_periods,
                    &self.d_inv_norms,
                    self.max_period as i32,
                    self.series_len as i32,
                    self.n_combos as i32,
                    self.first_valid as i32,
                    &mut self.d_out,
                )
                .expect("launch alma_batch_device");
            self.cuda.synchronize().expect("sync");
        }
    }

    fn prep_alma_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaAlma::new(0).expect("cuda alma");
        let price = gen_series(ONE_SERIES_LEN);
        let start_period = 10usize;
        let end_period = start_period + PARAM_SWEEP - 1;
        let sweep = AlmaBatchRange {
            period: (start_period, end_period, 1),
            offset: (0.85, 0.85, 0.0),
            sigma: (6.0, 6.0, 0.0),
        };

        // Compute first_valid on host once
        let first_valid = price
            .iter()
            .position(|x| !x.is_nan())
            .unwrap_or(0);

        // Expand combos and host-precompute weights once
        let combos = super::expand_grid(&sweep);
        let n_combos = combos.len();
        let max_period = combos
            .iter()
            .map(|c| c.period.unwrap_or(0))
            .max()
            .unwrap_or(0);

        let mut periods_i32 = vec![0i32; n_combos];
        let mut inv_norms = vec![0f32; n_combos];
        let mut weights_flat = vec![0f32; n_combos * max_period];
        for (idx, prm) in combos.iter().enumerate() {
            let period = prm.period.unwrap() as usize;
            let offset = prm.offset.unwrap();
            let sigma = prm.sigma.unwrap();
            let (weights, inv_norm) = super::compute_weights_cpu_f32(period, offset, sigma);
            periods_i32[idx] = period as i32;
            inv_norms[idx] = inv_norm;
            let base = idx * max_period;
            weights_flat[base..base + period].copy_from_slice(&weights);
        }

        // Upload once
        let d_prices = unsafe { DeviceBuffer::from_slice_async(&price, &cuda.stream) }.expect("d_prices");
        let d_weights = DeviceBuffer::from_slice(&weights_flat).expect("d_weights");
        let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("d_periods");
        let d_inv_norms = DeviceBuffer::from_slice(&inv_norms).expect("d_inv_norms");
        let d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(n_combos * ONE_SERIES_LEN, &cuda.stream)
        }
        .expect("d_out");
        cuda.synchronize().expect("sync after prep");

        Box::new(AlmaBatchDeviceState {
            cuda,
            d_prices,
            d_weights,
            d_periods,
            d_inv_norms,
            d_out,
            series_len: ONE_SERIES_LEN,
            n_combos,
            max_period,
            first_valid,
        })
    }

    struct AlmaManySeriesDeviceState {
        cuda: CudaAlma,
        d_prices_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        d_weights: DeviceBuffer<f32>,
        d_out_tm: DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        period: usize,
        inv_norm: f32,
    }

    impl CudaBenchState for AlmaManySeriesDeviceState {
        fn launch(&mut self) {
            self.cuda
                .alma_many_series_one_param_time_major_device_precomputed(
                    &self.d_prices_tm,
                    &self.d_weights,
                    self.period,
                    self.inv_norm,
                    self.cols,
                    self.rows,
                    &self.d_first_valids,
                    &mut self.d_out_tm,
                )
                .expect("alma many-series precomputed");
            self.cuda.synchronize().expect("sync");
        }
    }

    fn prep_alma_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaAlma::new(0).expect("cuda alma");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let prices_tm = gen_time_major_prices(cols, rows);

        // Params fixed
        let period = 64usize;
        let offset = 0.85f64;
        let sigma = 6.0f64;
        let (weights, inv_norm) = super::compute_weights_cpu_f32(period, offset, sigma);

        // Compute first_valids
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = 0usize;
            for t in 0..rows {
                if !prices_tm[t * cols + s].is_nan() { fv = t; break; }
            }
            first_valids[s] = fv as i32;
        }

        // Upload once
        let d_prices_tm = unsafe { DeviceBuffer::from_slice_async(&prices_tm, &cuda.stream) }.expect("d_prices_tm");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let d_weights = DeviceBuffer::from_slice(&weights).expect("d_weights");
        let d_out_tm: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(cols * rows, &cuda.stream)
        }
        .expect("d_out_tm");
        cuda.synchronize().expect("sync after prep");

        Box::new(AlmaManySeriesDeviceState {
            cuda,
            d_prices_tm,
            d_first_valids,
            d_weights,
            d_out_tm,
            cols,
            rows,
            period,
            inv_norm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "alma",
                "one_series_many_params",
                "alma_cuda_batch_dev",
                "1m_x_250",
                prep_alma_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "alma",
                "many_series_one_param",
                "alma_cuda_many_series_one_param",
                "250x1m",
                prep_alma_many_series_one_param,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
