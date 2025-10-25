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

#[derive(Debug)]
pub enum CudaKeltnerError {
    Cuda(String),
    InvalidInput(String),
    UnsupportedMa(String),
}

impl fmt::Display for CudaKeltnerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaKeltnerError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaKeltnerError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
            CudaKeltnerError::UnsupportedMa(s) => write!(f, "Unsupported MA: {}", s),
        }
    }
}
impl std::error::Error for CudaKeltnerError {}

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
    _ctx: Context,
    policy: CudaKeltnerPolicy,
    max_grid_y: u32,
}

impl CudaKeltner {
    pub fn new(device_id: usize) -> Result<Self, CudaKeltnerError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaKeltnerError::Cuda(e.to_string()))?;
        let dev = Device::get_device(device_id as u32)
            .map_err(|e| CudaKeltnerError::Cuda(e.to_string()))?;
        let ctx = Context::new(dev).map_err(|e| CudaKeltnerError::Cuda(e.to_string()))?;
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/keltner_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaKeltnerError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaKeltnerError::Cuda(e.to_string()))?;
        let max_grid_y = dev
            .get_attribute(DeviceAttribute::MaxGridDimY)
            .map_err(|e| CudaKeltnerError::Cuda(e.to_string()))? as u32;
        Ok(Self { module, stream, _ctx: ctx, policy: CudaKeltnerPolicy::default(), max_grid_y })
    }

    #[inline]
    pub fn set_policy(&mut self, p: CudaKeltnerPolicy) {
        self.policy = p;
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }
    #[inline]
    fn will_fit(bytes: usize, headroom: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Ok((free, _)) = mem_get_info() {
            bytes.saturating_add(headroom) <= free
        } else {
            true
        }
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

        let combos = expand_grid_local(sweep);
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
                let cuda = CudaEma::new(0).map_err(|e| CudaKeltnerError::Cuda(e.to_string()))?;
                cuda.ema_batch_dev(
                    source,
                    &EmaBatchRange {
                        period: (min_p, max_p, 1),
                    },
                )
                .map_err(|e| CudaKeltnerError::Cuda(e.to_string()))?
            }
            "sma" => {
                use crate::indicators::moving_averages::sma::SmaBatchRange;
                let cuda = CudaSma::new(0).map_err(|e| CudaKeltnerError::Cuda(e.to_string()))?;
                let (dev, _combos) = cuda
                    .sma_batch_dev(
                        source,
                        &SmaBatchRange {
                            period: (min_p, max_p, 1),
                        },
                    )
                    .map_err(|e| CudaKeltnerError::Cuda(e.to_string()))?;
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
            let buf = unsafe { DeviceBuffer::from_slice_async(&flat, &self.stream) }
                .map_err(|e| CudaKeltnerError::Cuda(e.to_string()))?;
            DeviceArrayF32 { buf, rows: rows_p, cols: len }
        };

        // VRAM estimate: parameters + outputs. Inputs already allocated above.
        let out_bytes = 3 * combos.len() * len * std::mem::size_of::<f32>();
        let param_bytes = combos.len() * (std::mem::size_of::<i32>() + std::mem::size_of::<f32>());
        let inputs_bytes = 2 * rows_p * len * std::mem::size_of::<f32>();
        let required = out_bytes + param_bytes + inputs_bytes;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaKeltnerError::InvalidInput(
                "estimated device memory exceeds free VRAM".into(),
            ));
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

        let d_row_period_idx = unsafe { DeviceBuffer::from_slice_async(&row_period_idx, &self.stream) }
            .map_err(|e| CudaKeltnerError::Cuda(e.to_string()))?;
        let d_row_multipliers = unsafe { DeviceBuffer::from_slice_async(&row_multipliers, &self.stream) }
            .map_err(|e| CudaKeltnerError::Cuda(e.to_string()))?;
        let d_row_warms = unsafe { DeviceBuffer::from_slice_async(&row_warms, &self.stream) }
            .map_err(|e| CudaKeltnerError::Cuda(e.to_string()))?;

        // Outputs
        let mut d_upper: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(combos.len() * len, &self.stream) }
            .map_err(|e| CudaKeltnerError::Cuda(e.to_string()))?;
        let mut d_middle: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(combos.len() * len, &self.stream) }
            .map_err(|e| CudaKeltnerError::Cuda(e.to_string()))?;
        let mut d_lower: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(combos.len() * len, &self.stream) }
            .map_err(|e| CudaKeltnerError::Cuda(e.to_string()))?;

        // Kernel
        let func = self
            .module
            .get_function("keltner_batch_f32")
            .map_err(|e| CudaKeltnerError::Cuda(e.to_string()))?;

        // grid.y must be <= device max; chunk if needed
        let block_x = self.policy.batch_block_x.unwrap_or(256);
        let grid_x = ((len as u32) + block_x - 1) / block_x;
        let max_y = self.max_grid_y as usize;
        let mut launched = 0usize;
        while launched < combos.len() {
            let chunk = (combos.len() - launched).min(max_y);
            let grid: GridSize = (grid_x.max(1), chunk as u32, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
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
                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaKeltnerError::Cuda(e.to_string()))?;
            }
            launched += chunk;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaKeltnerError::Cuda(e.to_string()))?;
        self.stream
            .synchronize()
            .map_err(|e| CudaKeltnerError::Cuda(e.to_string()))?;

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
                let cuda = CudaEma::new(0).map_err(|e| CudaKeltnerError::Cuda(e.to_string()))?;
                cuda.ema_many_series_one_param_time_major_dev(
                    source_tm,
                    cols,
                    rows,
                    &EmaParams {
                        period: Some(period),
                    },
                )
                .map_err(|e| CudaKeltnerError::Cuda(e.to_string()))?
            }
            "sma" => {
                use crate::indicators::moving_averages::sma::SmaParams;
                let cuda = CudaSma::new(0).map_err(|e| CudaKeltnerError::Cuda(e.to_string()))?;
                cuda.sma_multi_series_one_param_time_major_dev(
                    source_tm,
                    cols,
                    rows,
                    &SmaParams {
                        period: Some(period),
                    },
                )
                .map_err(|e| CudaKeltnerError::Cuda(e.to_string()))?
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
            let buf = unsafe { DeviceBuffer::from_slice_async(&out, &self.stream) }
                .map_err(|e| CudaKeltnerError::Cuda(e.to_string()))?;
            DeviceArrayF32 { buf, rows, cols }
        };

        // VRAM + outputs
        let out_bytes = 3 * expected * std::mem::size_of::<f32>();
        let inputs_bytes = 2 * expected * std::mem::size_of::<f32>();
        if !Self::will_fit(out_bytes + inputs_bytes, 64 * 1024 * 1024) {
            return Err(CudaKeltnerError::InvalidInput(
                "estimated device memory exceeds free VRAM".into(),
            ));
            return Err(CudaKeltnerError::InvalidInput(
                "estimated device memory exceeds free VRAM".into(),
            ));
        }

        let mut d_upper: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(expected, &self.stream) }
            .map_err(|e| CudaKeltnerError::Cuda(e.to_string()))?;
        let mut d_middle: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(expected, &self.stream) }
            .map_err(|e| CudaKeltnerError::Cuda(e.to_string()))?;
        let mut d_lower: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(expected, &self.stream) }
            .map_err(|e| CudaKeltnerError::Cuda(e.to_string()))?;

        let func = self
            .module
            .get_function("keltner_many_series_one_param_f32")
            .map_err(|e| CudaKeltnerError::Cuda(e.to_string()))?;
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
        let d_first = unsafe { DeviceBuffer::from_slice_async(&first_valids, &self.stream) }
            .map_err(|e| CudaKeltnerError::Cuda(e.to_string()))?;

        // Prefer 2-D launch if rows fits in grid.y; else 1-D fallback
        let use_2d = (rows as u32) <= self.max_grid_y;
        if use_2d {
            let grid_x = (((cols as u32) + block_x - 1) / block_x).max(1);
            let grid: GridSize = (grid_x, rows as u32, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
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
                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaKeltnerError::Cuda(e.to_string()))?;
            }
        } else {
            let grid_x = (((expected as u32) + block_x - 1) / block_x).max(1);
            let grid: GridSize = (grid_x, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
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
                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaKeltnerError::Cuda(e.to_string()))?;
            }
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaKeltnerError::Cuda(e.to_string()))?;

        Ok(DeviceKeltnerTriplet {
            upper: DeviceArrayF32 { buf: d_upper, rows, cols },
            middle: DeviceArrayF32 { buf: d_middle, rows, cols },
            lower: DeviceArrayF32 { buf: d_lower, rows, cols },
        })
    }
}

// Local replica of expand_grid from the indicator (kept private there)
fn expand_grid_local(r: &KeltnerBatchRange) -> Vec<KeltnerParams> {
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
    let mults = axis_f64(r.multiplier);
    let mut out = Vec::with_capacity(periods.len() * mults.len());
    for &p in &periods {
        for &m in &mults {
            out.push(KeltnerParams {
                period: Some(p),
                multiplier: Some(m),
                ma_type: None,
            });
        }
    }
    
    out
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
