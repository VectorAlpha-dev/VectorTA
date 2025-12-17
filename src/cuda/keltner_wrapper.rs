//! CUDA wrapper for Keltner Channels (MA +/- multiplier * ATR).
//!
//! Pattern classification: Composite. We precompute ATR and the selected
//! moving average using their existing CUDA wrappers, then combine them on
//! device via a lightweight kernel into upper/middle/lower bands.
//!
//! - Batch (one series × many params): periods and multipliers sweep; middle
//!   band is chosen MA (EMA default), ATR uses Wilder RMA. We reuse shared
//!   precompute by launching MA/ATR once over a contiguous period range and
//!   mapping output rows.
//! - Many-series × one-param (time-major): one period/multiplier across many
//!   series (columns). We compute MA/ATR per series and combine in-place.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::{CudaEma, CudaSma, DeviceArrayF32};
use crate::indicators::keltner::{KeltnerBatchRange, KeltnerParams};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CudaKeltnerError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
    #[error("unsupported MA: {0}")]
    UnsupportedMa(String),
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaKeltnerPolicy {
    pub batch_block_x: Option<u32>,
    pub many_block_x: Option<u32>,
}

pub struct DeviceKeltnerTriplet {
    pub upper: DeviceArrayF32,
    pub middle: DeviceArrayF32,
    pub lower: DeviceArrayF32,
}

pub struct CudaKeltnerBatchResult {
    pub outputs: DeviceKeltnerTriplet,
    pub combos: Vec<KeltnerParams>,
}

pub struct CudaKeltner {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaKeltnerPolicy,
    max_grid_y: u32,
}

impl CudaKeltner {
    pub fn new(device_id: usize) -> Result<Self, CudaKeltnerError> {
        cust::init(CudaFlags::empty())?;
        let dev = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(dev)?);
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/keltner_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) =
                    Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
                {
                    m
                } else {
                    Module::from_ptx(ptx, &[])?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let max_grid_y =
            Device::get_device(device_id as u32)?.get_attribute(DeviceAttribute::MaxGridDimY)? as u32;
        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy: CudaKeltnerPolicy::default(),
            max_grid_y,
        })
    }

    #[inline]
    pub fn set_policy(&mut self, p: CudaKeltnerPolicy) {
        self.policy = p;
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> {
        self.context.clone()
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
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
        mem_get_info().ok()
    }

    #[inline]
    fn will_fit(bytes: usize, headroom: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            bytes.saturating_add(headroom) <= free
        } else {
            true
        }
    }

    #[inline]
    fn validate_launch(
        &self,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    ) -> Result<(), CudaKeltnerError> {
        let dev = Device::get_device(self.device_id)?;
        let max_threads = dev
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_bx = dev
            .get_attribute(DeviceAttribute::MaxBlockDimX)
            .unwrap_or(1024) as u32;
        let max_by = dev
            .get_attribute(DeviceAttribute::MaxBlockDimY)
            .unwrap_or(1024) as u32;
        let max_bz = dev
            .get_attribute(DeviceAttribute::MaxBlockDimZ)
            .unwrap_or(64) as u32;
        let max_gx = dev
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(2_147_483_647) as u32;
        let max_gy = dev
            .get_attribute(DeviceAttribute::MaxGridDimY)
            .unwrap_or(65_535) as u32;
        let max_gz = dev
            .get_attribute(DeviceAttribute::MaxGridDimZ)
            .unwrap_or(65_535) as u32;

        let threads = bx.saturating_mul(by).saturating_mul(bz);
        if threads > max_threads
            || bx > max_bx
            || by > max_by
            || bz > max_bz
            || gx > max_gx
            || gy > max_gy
            || gz > max_gz
        {
            return Err(CudaKeltnerError::LaunchConfigTooLarge { gx, gy, gz, bx, by, bz });
        }
        Ok(())
    }

    // ---- Batch: one series × many params ----
    pub fn keltner_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        source: &[f32],
        sweep: &KeltnerBatchRange,
        ma_type: &str, // "ema" (default) or "sma"
    ) -> Result<CudaKeltnerBatchResult, CudaKeltnerError> {
        let len = close.len();
        if !(high.len() == low.len() && low.len() == close.len() && close.len() == source.len()) {
            return Err(CudaKeltnerError::InvalidInput(
                "input length mismatch".into(),
            ));
        }
        if len == 0 {
            return Err(CudaKeltnerError::InvalidInput("empty series".into()));
        }

        let combos = expand_grid_local(sweep)?;
        if combos.is_empty() {
            return Err(CudaKeltnerError::InvalidInput("empty sweep".into()));
        }

        // Period coverage: compute a single contiguous [min..max] range for MA + ATR.
        let min_p = combos.iter().map(|c| c.period.unwrap()).min().unwrap();
        let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
        if min_p == 0 || max_p > len {
            return Err(CudaKeltnerError::InvalidInput(
                "invalid period limits".into(),
            ));
        }

        // Launch MA and ATR batch once for [min..max] with step=1
        let rows_p = (max_p - min_p + 1) as usize;
        let ma_rows = match ma_type.to_ascii_lowercase().as_str() {
            "ema" => {
                use crate::indicators::moving_averages::ema::EmaBatchRange;
                let cuda = CudaEma::new(0)
                    .map_err(|e| CudaKeltnerError::InvalidInput(e.to_string()))?;
                cuda.ema_batch_dev(
                    source,
                    &EmaBatchRange {
                        period: (min_p, max_p, 1),
                    },
                )
                .map_err(|e| CudaKeltnerError::InvalidInput(e.to_string()))?
            }
            "sma" => {
                use crate::indicators::moving_averages::sma::SmaBatchRange;
                let cuda = CudaSma::new(0)
                    .map_err(|e| CudaKeltnerError::InvalidInput(e.to_string()))?;
                let (dev, _combos) = cuda
                    .sma_batch_dev(
                        source,
                        &SmaBatchRange {
                            period: (min_p, max_p, 1),
                        },
                    )
                    .map_err(|e| CudaKeltnerError::InvalidInput(e.to_string()))?;
                dev
            }
            other => return Err(CudaKeltnerError::UnsupportedMa(other.to_string())),
        };

        // Build ATR rows on host to mirror scalar Keltner ATR semantics exactly
        let atr_rows = {
            let mut flat = vec![0f32; rows_p * len];
            for (r, p) in (min_p..=max_p).enumerate() {
                let period = p;
                let alpha = 1.0f64 / (period as f64);
                let mut sum_tr = 0.0f64;
                let mut rma = f64::NAN;
                for i in 0..len {
                    let tr = if i == 0 {
                        (high[0] as f64) - (low[0] as f64)
                    } else {
                        let hi = high[i] as f64;
                        let lo = low[i] as f64;
                        let pc = close[i - 1] as f64;
                        let hl = hi - lo;
                        let hc = (hi - pc).abs();
                        let lc = (lo - pc).abs();
                        hl.max(hc).max(lc)
                    };
                    if i < period {
                        sum_tr += tr;
                        if i == period - 1 {
                            rma = sum_tr / (period as f64);
                            flat[r * len + i] = rma as f32;
                        }
                    } else {
                        rma = (tr - rma).mul_add(alpha, rma);
                        flat[r * len + i] = rma as f32;
                    }
                }
            }
            let buf = unsafe { DeviceBuffer::from_slice_async(&flat, &self.stream) }?;
            DeviceArrayF32 { buf, rows: rows_p, cols: len }
        };

        // VRAM estimate: parameters + outputs. Inputs already allocated above.
        let out_elems = combos
            .len()
            .checked_mul(len)
            .and_then(|v| v.checked_mul(3))
            .ok_or_else(|| CudaKeltnerError::InvalidInput("output size overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaKeltnerError::InvalidInput("output bytes overflow".into()))?;
        let param_bytes = combos
            .len()
            .checked_mul(std::mem::size_of::<i32>() + std::mem::size_of::<f32>())
            .ok_or_else(|| CudaKeltnerError::InvalidInput("param bytes overflow".into()))?;
        let inputs_elems = rows_p
            .checked_mul(len)
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| CudaKeltnerError::InvalidInput("input size overflow".into()))?;
        let inputs_bytes = inputs_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaKeltnerError::InvalidInput("input bytes overflow".into()))?;
        let required = out_bytes
            .checked_add(param_bytes)
            .and_then(|v| v.checked_add(inputs_bytes))
            .ok_or_else(|| CudaKeltnerError::InvalidInput("total bytes overflow".into()))?;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            if let Some((free, _total)) = Self::device_mem_info() {
                return Err(CudaKeltnerError::OutOfMemory {
                    required,
                    free,
                    headroom: 64 * 1024 * 1024,
                });
            } else {
                return Err(CudaKeltnerError::InvalidInput(
                    "insufficient device memory".into(),
                ));
            }
        }

        // Map each combo row -> period-row index in [min..max]
        let row_period_idx: Vec<i32> = combos
            .iter()
            .map(|c| (c.period.unwrap() as i32) - (min_p as i32))
            .collect();
        let row_multipliers: Vec<f32> = combos
            .iter()
            .map(|c| c.multiplier.unwrap() as f32)
            .collect();

        // Warm indices per combo: Keltner warm = first_valid(close) + period - 1
        let first_valid_close = close
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaKeltnerError::InvalidInput("all close values are NaN".into()))?;
        let row_warms: Vec<i32> = combos
            .iter()
            .map(|c| (first_valid_close + c.period.unwrap() - 1) as i32)
            .collect();

        let d_row_period_idx =
            unsafe { DeviceBuffer::from_slice_async(&row_period_idx, &self.stream) }?;
        let d_row_multipliers =
            unsafe { DeviceBuffer::from_slice_async(&row_multipliers, &self.stream) }?;
        let d_row_warms =
            unsafe { DeviceBuffer::from_slice_async(&row_warms, &self.stream) }?;

        // Outputs
        let mut d_upper: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(combos.len() * len, &self.stream) }?;
        let mut d_middle: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(combos.len() * len, &self.stream) }?;
        let mut d_lower: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(combos.len() * len, &self.stream) }?;

        // Kernel
        let func = self
            .module
            .get_function("keltner_batch_f32")
            .map_err(|_| CudaKeltnerError::MissingKernelSymbol { name: "keltner_batch_f32" })?;

        // grid.y must be <= device max; chunk if needed
        let block_x = self.policy.batch_block_x.unwrap_or(256);
        let grid_x = ((len as u32) + block_x - 1) / block_x;
        let max_y = self.max_grid_y as usize;
        let mut launched = 0usize;
        while launched < combos.len() {
            let chunk = (combos.len() - launched).min(max_y);
            let grid: GridSize = (grid_x.max(1), chunk as u32, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            self.validate_launch(grid_x.max(1), chunk as u32, 1, block_x, 1, 1)?;
            unsafe {
                let mut ma_ptr = ma_rows.buf.as_device_ptr().as_raw();
                let mut atr_ptr = atr_rows.buf.as_device_ptr().as_raw();
                let mut len_i = len as i32;
                let mut rows_i = chunk as i32;
                // typed element offsets
                let mut idx_ptr = d_row_period_idx.as_device_ptr().offset(launched as isize).as_raw();
                let mut mul_ptr = d_row_multipliers.as_device_ptr().offset(launched as isize).as_raw();
                let mut warm_ptr = d_row_warms.as_device_ptr().offset(launched as isize).as_raw();
                let mut up_ptr = d_upper.as_device_ptr().offset((launched * len) as isize).as_raw();
                let mut mid_ptr = d_middle.as_device_ptr().offset((launched * len) as isize).as_raw();
                let mut low_ptr = d_lower.as_device_ptr().offset((launched * len) as isize).as_raw();

                let args: &mut [*mut c_void] = &mut [
                    &mut ma_ptr as *mut _ as *mut c_void,
                    &mut atr_ptr as *mut _ as *mut c_void,
                    &mut idx_ptr as *mut _ as *mut c_void,
                    &mut mul_ptr as *mut _ as *mut c_void,
                    &mut warm_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut up_ptr as *mut _ as *mut c_void,
                    &mut mid_ptr as *mut _ as *mut c_void,
                    &mut low_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, 0, args)?;
            }
            launched += chunk;
        }

        self.stream.synchronize()?;
        self.stream.synchronize()?;

        Ok(CudaKeltnerBatchResult {
            outputs: DeviceKeltnerTriplet {
                upper: DeviceArrayF32 { buf: d_upper, rows: combos.len(), cols: len },
                middle: DeviceArrayF32 { buf: d_middle, rows: combos.len(), cols: len },
                lower: DeviceArrayF32 { buf: d_lower, rows: combos.len(), cols: len },
            },
            combos,
        })
    }

    // ---- Many-series × one-param (time-major) ----
    pub fn keltner_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        source_tm: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
        multiplier: f32,
        ma_type: &str,
    ) -> Result<DeviceKeltnerTriplet, CudaKeltnerError> {
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaKeltnerError::InvalidInput("rows*cols overflow".into()))?;
        if high_tm.len() != expected
            || low_tm.len() != expected
            || close_tm.len() != expected
            || source_tm.len() != expected
        {
            return Err(CudaKeltnerError::InvalidInput(
                "time-major input length mismatch".into(),
            ));
        }
        if period == 0 || rows < period {
            return Err(CudaKeltnerError::InvalidInput("invalid period".into()));
        }

        // MA per series (time-major) depending on selection
        let ma_tm = match ma_type.to_ascii_lowercase().as_str() {
            "ema" => {
                use crate::indicators::moving_averages::ema::EmaParams;
                let cuda = CudaEma::new(0)
                    .map_err(|e| CudaKeltnerError::InvalidInput(e.to_string()))?;
                cuda.ema_many_series_one_param_time_major_dev(
                    source_tm,
                    cols,
                    rows,
                    &EmaParams {
                        period: Some(period),
                    },
                )
                .map_err(|e| CudaKeltnerError::InvalidInput(e.to_string()))?
            }
            "sma" => {
                use crate::indicators::moving_averages::sma::SmaParams;
                let cuda = CudaSma::new(0)
                    .map_err(|e| CudaKeltnerError::InvalidInput(e.to_string()))?;
                cuda.sma_multi_series_one_param_time_major_dev(
                    source_tm,
                    cols,
                    rows,
                    &SmaParams {
                        period: Some(period),
                    },
                )
                .map_err(|e| CudaKeltnerError::InvalidInput(e.to_string()))?
            }
            other => return Err(CudaKeltnerError::UnsupportedMa(other.to_string())),
        };

        // ATR per series (time-major)
        // Host ATR per series to mirror scalar behavior
        let atr_tm = {
            let mut out = vec![f32::NAN; cols * rows];
            let alpha = 1.0f64 / (period as f64);
            for s in 0..cols {
                let mut sum_tr = 0.0f64;
                let mut rma = f64::NAN;
                for t in 0..rows {
                    let idx = t * cols + s;
                    let tr = if t == 0 {
                        (high_tm[idx] as f64) - (low_tm[idx] as f64)
                    } else {
                        let hi = high_tm[idx] as f64;
                        let lo = low_tm[idx] as f64;
                        let pc = close_tm[(t - 1) * cols + s] as f64;
                        let hl = hi - lo;
                        let hc = (hi - pc).abs();
                        let lc = (lo - pc).abs();
                        hl.max(hc).max(lc)
                    };
                    if t < period {
                        sum_tr += tr;
                        if t == period - 1 {
                            rma = sum_tr / (period as f64);
                            out[idx] = rma as f32;
                        }
                        if t == period - 1 {
                            rma = sum_tr / (period as f64);
                            out[idx] = rma as f32;
                        }
                    } else {
                        rma = (tr - rma).mul_add(alpha, rma);
                        out[idx] = rma as f32;
                    }
                }
            }
            let buf = unsafe { DeviceBuffer::from_slice_async(&out, &self.stream) }?;
            DeviceArrayF32 { buf, rows, cols }
        };

        // VRAM + outputs
        let out_bytes = expected
            .checked_mul(3)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaKeltnerError::InvalidInput("output bytes overflow".into()))?;
        let inputs_bytes = expected
            .checked_mul(2)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaKeltnerError::InvalidInput("input bytes overflow".into()))?;
        let required = out_bytes
            .checked_add(inputs_bytes)
            .ok_or_else(|| CudaKeltnerError::InvalidInput("total bytes overflow".into()))?;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            if let Some((free, _total)) = Self::device_mem_info() {
                return Err(CudaKeltnerError::OutOfMemory {
                    required,
                    free,
                    headroom: 64 * 1024 * 1024,
                });
            } else {
                return Err(CudaKeltnerError::InvalidInput(
                    "insufficient device memory".into(),
                ));
            }
        }

        let mut d_upper: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(expected, &self.stream) }?;
        let mut d_middle: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(expected, &self.stream) }?;
        let mut d_lower: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(expected, &self.stream) }?;

        let func = self
            .module
            .get_function("keltner_many_series_one_param_f32")
            .map_err(|_| {
                CudaKeltnerError::MissingKernelSymbol {
                    name: "keltner_many_series_one_param_f32",
                }
            })?;
        let block_x = self.policy.many_block_x.unwrap_or(256);
        // Build first_valids per series from close (Keltner semantics)
        let mut first_valids: Vec<i32> = vec![-1; cols];
        for s in 0..cols {
            first_valids[s] = (0..rows)
                .find(|&t| {
                    let v = close_tm[t * cols + s];
                    !v.is_nan()
                })
                .unwrap_or(rows) as i32;
        }
        let d_first =
            unsafe { DeviceBuffer::from_slice_async(&first_valids, &self.stream) }?;

        // Prefer 2-D launch if rows fits in grid.y; else 1-D fallback
        let use_2d = (rows as u32) <= self.max_grid_y;
        if use_2d {
            let grid_x = (((cols as u32) + block_x - 1) / block_x).max(1);
            let grid: GridSize = (grid_x, rows as u32, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            self.validate_launch(grid_x, rows as u32, 1, block_x, 1, 1)?;
            unsafe {
                let mut ma_ptr = ma_tm.buf.as_device_ptr().as_raw();
                let mut atr_ptr = atr_tm.buf.as_device_ptr().as_raw();
                let mut fv_ptr = d_first.as_device_ptr().as_raw();
                let mut period_i = period as i32;
                let mut cols_i = cols as i32;
                let mut rows_i = rows as i32;
                let mut elems_i = expected as i32; // unused in 2D path
                let mut mult = multiplier as f32;
                let mut up_ptr = d_upper.as_device_ptr().as_raw();
                let mut mid_ptr = d_middle.as_device_ptr().as_raw();
                let mut low_ptr = d_lower.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut ma_ptr as *mut _ as *mut c_void,
                    &mut atr_ptr as *mut _ as *mut c_void,
                    &mut fv_ptr as *mut _ as *mut c_void,
                    &mut period_i as *mut _ as *mut c_void,
                    &mut cols_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut elems_i as *mut _ as *mut c_void,
                    &mut mult as *mut _ as *mut c_void,
                    &mut up_ptr as *mut _ as *mut c_void,
                    &mut mid_ptr as *mut _ as *mut c_void,
                    &mut low_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, 0, args)?;
            }
        } else {
            let grid_x = (((expected as u32) + block_x - 1) / block_x).max(1);
            let grid: GridSize = (grid_x, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            self.validate_launch(grid_x, 1, 1, block_x, 1, 1)?;
            unsafe {
                let mut ma_ptr = ma_tm.buf.as_device_ptr().as_raw();
                let mut atr_ptr = atr_tm.buf.as_device_ptr().as_raw();
                let mut fv_ptr = d_first.as_device_ptr().as_raw();
                let mut period_i = period as i32;
                let mut cols_i = cols as i32;
                let mut rows_i = rows as i32;
                let mut elems_i = expected as i32;
                let mut mult = multiplier as f32;
                let mut up_ptr = d_upper.as_device_ptr().as_raw();
                let mut mid_ptr = d_middle.as_device_ptr().as_raw();
                let mut low_ptr = d_lower.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut ma_ptr as *mut _ as *mut c_void,
                    &mut atr_ptr as *mut _ as *mut c_void,
                    &mut fv_ptr as *mut _ as *mut c_void,
                    &mut period_i as *mut _ as *mut c_void,
                    &mut cols_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut elems_i as *mut _ as *mut c_void,
                    &mut mult as *mut _ as *mut c_void,
                    &mut up_ptr as *mut _ as *mut c_void,
                    &mut mid_ptr as *mut _ as *mut c_void,
                    &mut low_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, 0, args)?;
            }
        }

        self.stream.synchronize()?;

        Ok(DeviceKeltnerTriplet {
            upper: DeviceArrayF32 { buf: d_upper, rows, cols },
            middle: DeviceArrayF32 { buf: d_middle, rows, cols },
            lower: DeviceArrayF32 { buf: d_lower, rows, cols },
        })
    }
}

// Local replica of expand_grid from the indicator (kept private there)
fn expand_grid_local(r: &KeltnerBatchRange) -> Result<Vec<KeltnerParams>, CudaKeltnerError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, CudaKeltnerError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            return Ok((start..=end).step_by(step.max(1)).collect());
        }
        let mut v = Vec::new();
        let mut x = start as isize;
        let end_i = end as isize;
        let st = (step as isize).max(1);
        while x >= end_i {
            v.push(x as usize);
            x -= st;
        }
        if v.is_empty() {
            return Err(CudaKeltnerError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        Ok(v)
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, CudaKeltnerError> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }
        if start < end {
            let mut v = Vec::new();
            let mut x = start;
            let st = step.abs();
            while x <= end + 1e-12 {
                v.push(x);
                x += st;
            }
            if v.is_empty() {
                return Err(CudaKeltnerError::InvalidInput(format!(
                    "invalid range: start={start}, end={end}, step={step}"
                )));
            }
            return Ok(v);
        }
        let mut v = Vec::new();
        let mut x = start;
        let st = step.abs();
        while x + 1e-12 >= end {
            v.push(x);
            x -= st;
        }
        if v.is_empty() {
            return Err(CudaKeltnerError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        Ok(v)
    }

    let periods = axis_usize(r.period)?;
    let mults = axis_f64(r.multiplier)?;

    let cap = periods
        .len()
        .checked_mul(mults.len())
        .ok_or_else(|| CudaKeltnerError::InvalidInput("rows*cols overflow".into()))?;

    let mut out = Vec::with_capacity(cap);
    for &p in &periods {
        for &m in &mults {
            out.push(KeltnerParams {
                period: Some(p),
                multiplier: Some(m),
                ma_type: None,
            });
        }
    }

    Ok(out)
}
// ---------------- Bench profiles ----------------
#[cfg(not(test))]
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::keltner::KeltnerBatchRange;

    const ONE_SERIES_LEN: usize = 1_000_000;

    struct BatchState {
        cuda: CudaKeltner,
        high: Vec<f32>,
        low: Vec<f32>,
        close: Vec<f32>,
        src: Vec<f32>,
        sweep: KeltnerBatchRange,
    }
    impl CudaBenchState for BatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .keltner_batch_dev(
                    &self.high,
                    &self.low,
                    &self.close,
                    &self.src,
                    &self.sweep,
                    "ema",
                )
                .unwrap();
        }
    }
    

    struct ManyState {
        cuda: CudaKeltner,
        high_tm: Vec<f32>,
        low_tm: Vec<f32>,
        close_tm: Vec<f32>,
        src_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        period: usize,
        mult: f32,
    }
    impl CudaBenchState for ManyState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .keltner_many_series_one_param_time_major_dev(
                    &self.high_tm,
                    &self.low_tm,
                    &self.close_tm,
                    &self.src_tm,
                    self.cols,
                    self.rows,
                    self.period,
                    self.mult,
                    "ema",
                )
                .unwrap();
        }
    }
    

    fn synth_hlc_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i];
            if v.is_nan() {
                continue;
            }
            if v.is_nan() {
                continue;
            }
            let x = i as f32 * 0.002f32;
            let off = (0.004 * x.sin()).abs() + 0.12;
            high[i] = v + off;
            low[i] = v - off;
        }
        (high, low)
    }

    fn prep_batch() -> Box<dyn CudaBenchState> {
        let len = ONE_SERIES_LEN;
        let close = gen_series(len);
        let (high, low) = synth_hlc_from_close(&close);
        let sweep = KeltnerBatchRange {
            period: (10, 73, 1),
            multiplier: (1.0, 2.0, 0.25),
        };
        Box::new(BatchState { cuda: CudaKeltner::new(0).unwrap(), high, low, close: close.clone(), src: close, sweep })
    }

    fn prep_many() -> Box<dyn CudaBenchState> {
        let (cols, rows, period, mult) = (256usize, 262_144usize, 20usize, 2.0f32);
        let close_tm = gen_time_major_prices(cols, rows);
        let mut high_tm = close_tm.clone();
        let mut low_tm = close_tm.clone();
        for s in 0..cols {
            for t in 0..rows {
                let v = close_tm[t * cols + s];
                if v.is_nan() {
                    continue;
                }
                let x = (t as f32) * 0.002;
                let off = (0.004 * x.cos()).abs() + 0.11;
                high_tm[t * cols + s] = v + off;
                low_tm[t * cols + s] = v - off;
            }
        }
        Box::new(ManyState { cuda: CudaKeltner::new(0).unwrap(), high_tm, low_tm, close_tm: close_tm.clone(), src_tm: close_tm, cols, rows, period, mult })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        let scen_batch = CudaBenchScenario::new(
            "keltner",
            "one_series_many_params",
            "keltner_cuda_batch_dev",
            "1m_len",
            prep_batch,
        )
        .with_mem_required(
            (3 * ONE_SERIES_LEN + 2 * ONE_SERIES_LEN) * std::mem::size_of::<f32>()
                + 64 * 1024 * 1024,
        );

        let (cols, rows) = (256usize, 262_144usize);
        let scen_many = CudaBenchScenario::new(
            "keltner",
            "many_series_one_param",
            "keltner_cuda_many_series_one_param_dev",
            "256x262k",
            prep_many,
        )
        .with_mem_required((5 * cols * rows) * std::mem::size_of::<f32>() + 64 * 1024 * 1024);

        vec![scen_batch, scen_many]
    }
}
