//! CUDA wrapper for Moving Average Bands (MAB).
//!
//! Mirrors ALMA/CWMA conventions:
//! - PTX loaded via include_str!(concat!(env!("OUT_DIR"), "/mab_kernel.ptx"))
//! - Stream NON_BLOCKING; conservative JIT options (DetermineTargetFromContext + O2)
//! - VRAM check with headroom; chunking not required for typical sizes
//! - Two public device entry points:
//!     - `mab_batch_dev(&[f32], &MabBatchRange)` -> (upper, middle, lower, combos)
//!     - `mab_many_series_one_param_time_major_dev(&[f32], cols, rows, &MabParams)`
//! - Reuses CUDA MA wrappers via `CudaMaSelector` (batch) and direct wrappers (many-series)

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::mab::{MabBatchRange, MabParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaMabError {
    Cuda(String),
    InvalidInput(String),
    Unsupported(String),
}

impl fmt::Display for CudaMabError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaMabError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaMabError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
            CudaMabError::Unsupported(s) => write!(f, "Unsupported: {}", s),
        }
    }
}
impl std::error::Error for CudaMabError {}

pub struct DeviceArrayF32Triplet {
    pub upper: DeviceArrayF32,
    pub middle: DeviceArrayF32,
    pub lower: DeviceArrayF32,
}

impl DeviceArrayF32Triplet {
    #[inline]
    pub fn rows(&self) -> usize {
        self.upper.rows
    }
    #[inline]
    pub fn cols(&self) -> usize {
        self.upper.cols
    }
}

pub struct CudaMab {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaMab {
    pub fn new(device_id: usize) -> Result<Self, CudaMabError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaMabError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaMabError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaMabError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/mab_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
                {
                    m
                } else {
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaMabError::Cuda(e.to_string()))?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaMabError::Cuda(e.to_string()))?;
        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    // Build an MA on host (f32) using the CPU reference implementations, then convert to f32.
    fn compute_ma_host(
        ma_type: &str,
        prices_f32: &[f32],
        period: usize,
    ) -> Result<Vec<f32>, CudaMabError> {
        use crate::indicators::moving_averages::ema::{ema, EmaInput, EmaParams};
        use crate::indicators::moving_averages::sma::{sma, SmaInput, SmaParams};
        let prices: Vec<f64> = prices_f32.iter().map(|&v| v as f64).collect();
        let n = prices.len();
        if period == 0 || period > n {
            return Err(CudaMabError::InvalidInput("invalid period".into()));
        }
        let out_f64 = match ma_type.to_ascii_lowercase().as_str() {
            "ema" => {
                ema(&EmaInput::from_slice(
                    &prices,
                    EmaParams {
                        period: Some(period),
                    },
                ))
                .map_err(|e| CudaMabError::InvalidInput(e.to_string()))?
                .values
            }
            _ => {
                sma(&SmaInput::from_slice(
                    &prices,
                    SmaParams {
                        period: Some(period),
                    },
                ))
                .map_err(|e| CudaMabError::InvalidInput(e.to_string()))?
                .values
            }
        };
        Ok(out_f64.into_iter().map(|v| v as f32).collect())
    }

    fn compute_ma_host_time_major(
        ma_type: &str,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<Vec<f32>, CudaMabError> {
        use crate::indicators::moving_averages::ema::{ema, EmaInput, EmaParams};
        use crate::indicators::moving_averages::sma::{sma, SmaInput, SmaParams};
        if data_tm_f32.len() != cols * rows {
            return Err(CudaMabError::InvalidInput(
                "time-major dims mismatch".into(),
            ));
        }
        let mut out_tm = vec![f32::NAN; cols * rows];
        for s in 0..cols {
            // extract column
            let mut col = vec![f64::NAN; rows];
            for r in 0..rows {
                col[r] = data_tm_f32[r * cols + s] as f64;
            }
            let vals = match ma_type.to_ascii_lowercase().as_str() {
                "ema" => {
                    ema(&EmaInput::from_slice(
                        &col,
                        EmaParams {
                            period: Some(period),
                        },
                    ))
                    .map_err(|e| CudaMabError::InvalidInput(e.to_string()))?
                    .values
                }
                _ => {
                    sma(&SmaInput::from_slice(
                        &col,
                        SmaParams {
                            period: Some(period),
                        },
                    ))
                    .map_err(|e| CudaMabError::InvalidInput(e.to_string()))?
                    .values
                }
            };
            for r in 0..rows {
                out_tm[r * cols + s] = vals[r] as f32;
            }
        }
        Ok(out_tm)
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Ok((free, _total)) = mem_get_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    // --------- Batch (one-series × many params) ----------
    pub fn mab_batch_dev(
        &self,
        prices_f32: &[f32],
        sweep: &MabBatchRange,
    ) -> Result<(DeviceArrayF32Triplet, Vec<MabParams>), CudaMabError> {
        if prices_f32.is_empty() {
            return Err(CudaMabError::InvalidInput("empty input".into()));
        }
        let len = prices_f32.len();
        let first_valid = prices_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaMabError::InvalidInput("all values are NaN".into()))?;

        // Expand parameter grid
        let combos = {
            use crate::indicators::mab::MabParams as P;
            let mut v = Vec::new();
            // reuse CPU expand_grid logic via a tiny mirror
            let mut fasts = Vec::new();
            if sweep.fast_period.2 == 0 {
                fasts.push(sweep.fast_period.0);
            } else {
                let mut p = sweep.fast_period.0;
                while p <= sweep.fast_period.1 {
                    fasts.push(p);
                    p += sweep.fast_period.2;
                }
            }
            let mut slows = Vec::new();
            if sweep.slow_period.2 == 0 {
                slows.push(sweep.slow_period.0);
            } else {
                let mut p = sweep.slow_period.0;
                while p <= sweep.slow_period.1 {
                    slows.push(p);
                    p += sweep.slow_period.2;
                }
            }
            let mut ups = Vec::new();
            if sweep.devup.2 == 0.0 {
                ups.push(sweep.devup.0);
            } else {
                let mut d = sweep.devup.0;
                while d <= sweep.devup.1 {
                    ups.push(d);
                    d += sweep.devup.2;
                }
            }
            let mut dns = Vec::new();
            if sweep.devdn.2 == 0.0 {
                dns.push(sweep.devdn.0);
            } else {
                let mut d = sweep.devdn.0;
                while d <= sweep.devdn.1 {
                    dns.push(d);
                    d += sweep.devdn.2;
                }
            }
            for &f in &fasts {
                for &s in &slows {
                    for &u in &ups {
                        for &d in &dns {
                            v.push(P {
                                fast_period: Some(f),
                                slow_period: Some(s),
                                devup: Some(u),
                                devdn: Some(d),
                                fast_ma_type: Some(sweep.fast_ma_type.0.clone()),
                                slow_ma_type: Some(sweep.slow_ma_type.0.clone()),
                            });
                        }
                    }
                }
            }
            v
        };
        if combos.is_empty() {
            return Err(CudaMabError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        // VRAM check for outputs
        let rows = combos.len();
        let out_bytes = 3usize * rows * len * std::mem::size_of::<f32>();
        let in_bytes = len * std::mem::size_of::<f32>();
        let required = out_bytes + in_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaMabError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                required as f64 / (1024.0 * 1024.0)
            )));
        }

        // Build per-row arrays for devup/devdn
        let devups: Vec<f32> = combos.iter().map(|p| p.devup.unwrap() as f32).collect();
        let devdns: Vec<f32> = combos.iter().map(|p| p.devdn.unwrap() as f32).collect();

        // Detect fast-path: identical MA setup across rows
        let p0 = &combos[0];
        let all_same_ma = combos.iter().all(|p| {
            p.fast_period == p0.fast_period
                && p.slow_period == p0.slow_period
                && p.fast_ma_type == p0.fast_ma_type
                && p.slow_ma_type == p0.slow_ma_type
        });

        // Allocate outputs
        let mut d_upper: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(rows * len) }
            .map_err(|e| CudaMabError::Cuda(e.to_string()))?;
        let mut d_middle: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(rows * len) }
            .map_err(|e| CudaMabError::Cuda(e.to_string()))?;
        let mut d_lower: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(rows * len) }
            .map_err(|e| CudaMabError::Cuda(e.to_string()))?;

        if all_same_ma {
            // Single fast/slow MA built on host for this context, then dev once, then apply per-row multipliers.
            let fast_ma_host = Self::compute_ma_host(
                p0.fast_ma_type.as_deref().unwrap_or("sma"),
                prices_f32,
                p0.fast_period.unwrap(),
            )?;
            let slow_ma_host = Self::compute_ma_host(
                p0.slow_ma_type.as_deref().unwrap_or("sma"),
                prices_f32,
                p0.slow_period.unwrap(),
            )?;
            let d_fast = DeviceBuffer::from_slice(&fast_ma_host)
                .map_err(|e| CudaMabError::Cuda(e.to_string()))?;
            let d_slow = DeviceBuffer::from_slice(&slow_ma_host)
                .map_err(|e| CudaMabError::Cuda(e.to_string()))?;

            // dev buffer
            let mut d_dev: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }
                .map_err(|e| CudaMabError::Cuda(e.to_string()))?;

            // Launch mab_dev_from_ma_f32
            let mut f_dev: Function = self
                .module
                .get_function("mab_dev_from_ma_f32")
                .map_err(|e| CudaMabError::Cuda(e.to_string()))?;

            unsafe {
                let mut fast_ptr = d_fast.as_device_ptr().as_raw();
                let mut slow_ptr = d_slow.as_device_ptr().as_raw();
                let mut fp_i = p0.fast_period.unwrap() as i32;
                let mut fv_i = first_valid as i32;
                let mut len_i = len as i32;
                let mut dev_ptr = d_dev.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut fast_ptr as *mut _ as *mut c_void,
                    &mut slow_ptr as *mut _ as *mut c_void,
                    &mut fp_i as *mut _ as *mut c_void,
                    &mut fv_i as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut dev_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(
                        &mut f_dev,
                        GridSize::xyz(1, 1, 1),
                        BlockSize::xyz(1, 1, 1),
                        0,
                        args,
                    )
                    .map_err(|e| CudaMabError::Cuda(e.to_string()))?;
            }

            // Upload per-row multipliers
            let h_ups =
                LockedBuffer::from_slice(&devups).map_err(|e| CudaMabError::Cuda(e.to_string()))?;
            let h_dns =
                LockedBuffer::from_slice(&devdns).map_err(|e| CudaMabError::Cuda(e.to_string()))?;
            let mut d_ups = unsafe { DeviceBuffer::<f32>::uninitialized_async(rows, &self.stream) }
                .map_err(|e| CudaMabError::Cuda(e.to_string()))?;
            let mut d_dns = unsafe { DeviceBuffer::<f32>::uninitialized_async(rows, &self.stream) }
                .map_err(|e| CudaMabError::Cuda(e.to_string()))?;
            unsafe {
                d_ups
                    .async_copy_from(&h_ups, &self.stream)
                    .map_err(|e| CudaMabError::Cuda(e.to_string()))?;
                d_dns
                    .async_copy_from(&h_dns, &self.stream)
                    .map_err(|e| CudaMabError::Cuda(e.to_string()))?;
            }

            // Apply shared dev to outputs for all rows
            let mut f_apply: Function = self
                .module
                .get_function("mab_apply_dev_shared_ma_batch_f32")
                .map_err(|e| CudaMabError::Cuda(e.to_string()))?;

            // Choose a reasonable 1D x grid for time; y=rows
            let block_x: u32 = 256;
            let grid_x = ((len as u32) + block_x - 1) / block_x;
            let grid = GridSize::xyz(grid_x.max(1), rows as u32, 1);
            let block = BlockSize::xyz(block_x, 1, 1);

            unsafe {
                let mut fast_ptr = d_fast.as_device_ptr().as_raw();
                let mut slow_ptr = d_slow.as_device_ptr().as_raw();
                let mut dev_ptr = d_dev.as_device_ptr().as_raw();
                let mut fp_i = p0.fast_period.unwrap() as i32;
                let mut sp_i = p0.slow_period.unwrap() as i32;
                let mut fv_i = first_valid as i32;
                let mut len_i = len as i32;
                let mut ups_ptr = d_ups.as_device_ptr().as_raw();
                let mut dns_ptr = d_dns.as_device_ptr().as_raw();
                let mut rows_i = rows as i32;
                let mut up_ptr = d_upper.as_device_ptr().as_raw();
                let mut mid_ptr = d_middle.as_device_ptr().as_raw();
                let mut lo_ptr = d_lower.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut fast_ptr as *mut _ as *mut c_void,
                    &mut slow_ptr as *mut _ as *mut c_void,
                    &mut dev_ptr as *mut _ as *mut c_void,
                    &mut fp_i as *mut _ as *mut c_void,
                    &mut sp_i as *mut _ as *mut c_void,
                    &mut fv_i as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut ups_ptr as *mut _ as *mut c_void,
                    &mut dns_ptr as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut up_ptr as *mut _ as *mut c_void,
                    &mut mid_ptr as *mut _ as *mut c_void,
                    &mut lo_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&mut f_apply, grid, block, 0, args)
                    .map_err(|e| CudaMabError::Cuda(e.to_string()))?;
            }
        } else {
            // Generic path: build per-row fast/slow on host for this context and compute row independently
            let mut f_row: Function = self
                .module
                .get_function("mab_single_row_from_ma_f32")
                .map_err(|e| CudaMabError::Cuda(e.to_string()))?;
            for (row, p) in combos.iter().enumerate() {
                let fast_ma_host = Self::compute_ma_host(
                    p.fast_ma_type.as_deref().unwrap_or("sma"),
                    prices_f32,
                    p.fast_period.unwrap(),
                )?;
                let slow_ma_host = Self::compute_ma_host(
                    p.slow_ma_type.as_deref().unwrap_or("sma"),
                    prices_f32,
                    p.slow_period.unwrap(),
                )?;
                let d_fast = DeviceBuffer::from_slice(&fast_ma_host)
                    .map_err(|e| CudaMabError::Cuda(e.to_string()))?;
                let d_slow = DeviceBuffer::from_slice(&slow_ma_host)
                    .map_err(|e| CudaMabError::Cuda(e.to_string()))?;

                // target row window
                let row_off = row * len;
                let mut up_row =
                    unsafe { d_upper.as_device_ptr().offset(row_off as isize).as_raw() };
                let mut mid_row =
                    unsafe { d_middle.as_device_ptr().offset(row_off as isize).as_raw() };
                let mut lo_row =
                    unsafe { d_lower.as_device_ptr().offset(row_off as isize).as_raw() };

                unsafe {
                    let mut fast_ptr = d_fast.as_device_ptr().as_raw();
                    let mut slow_ptr = d_slow.as_device_ptr().as_raw();
                    let mut fp_i = p.fast_period.unwrap() as i32;
                    let mut sp_i = p.slow_period.unwrap() as i32;
                    let mut fv_i = first_valid as i32;
                    let mut len_i = len as i32;
                    let mut upf = p.devup.unwrap() as f32;
                    let mut dnf = p.devdn.unwrap() as f32;
                    let args: &mut [*mut c_void] = &mut [
                        &mut fast_ptr as *mut _ as *mut c_void,
                        &mut slow_ptr as *mut _ as *mut c_void,
                        &mut fp_i as *mut _ as *mut c_void,
                        &mut sp_i as *mut _ as *mut c_void,
                        &mut fv_i as *mut _ as *mut c_void,
                        &mut len_i as *mut _ as *mut c_void,
                        &mut upf as *mut _ as *mut c_void,
                        &mut dnf as *mut _ as *mut c_void,
                        &mut up_row as *mut _ as *mut c_void,
                        &mut mid_row as *mut _ as *mut c_void,
                        &mut lo_row as *mut _ as *mut c_void,
                    ];
                    self.stream
                        .launch(
                            &mut f_row,
                            GridSize::xyz(1, 1, 1),
                            BlockSize::xyz(1, 1, 1),
                            0,
                            args,
                        )
                        .map_err(|e| CudaMabError::Cuda(e.to_string()))?;
                }
            }
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaMabError::Cuda(e.to_string()))?;

        let trip = DeviceArrayF32Triplet {
            upper: DeviceArrayF32 {
                buf: d_upper,
                rows,
                cols: len,
            },
            middle: DeviceArrayF32 {
                buf: d_middle,
                rows,
                cols: len,
            },
            lower: DeviceArrayF32 {
                buf: d_lower,
                rows,
                cols: len,
            },
        };
        Ok((trip, combos))
    }

    // --------- Many-series (time-major), one param ----------
    pub fn mab_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &MabParams,
    ) -> Result<DeviceArrayF32Triplet, CudaMabError> {
        if cols == 0 || rows == 0 {
            return Err(CudaMabError::InvalidInput("invalid series dims".into()));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaMabError::InvalidInput(
                "time-major length mismatch".into(),
            ));
        }
        let fast = params.fast_period.unwrap_or(0);
        let slow = params.slow_period.unwrap_or(0);
        if fast == 0 || slow == 0 {
            return Err(CudaMabError::InvalidInput("periods must be >=1".into()));
        }

        // Compute first_valid per series
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for r in 0..rows {
                if !data_tm_f32[r * cols + s].is_nan() {
                    fv = Some(r as i32);
                    break;
                }
            }
            let fv =
                fv.ok_or_else(|| CudaMabError::InvalidInput(format!("series {} all-NaN", s)))?;
            // require enough tail for both MAs and dev window
            let need_total = (fast.max(slow) + fast - 1) as i32;
            if (rows as i32) - fv < need_total {
                return Err(CudaMabError::InvalidInput(format!(
                    "series {} insufficient valid tail for fast={}, slow={}",
                    s, fast, slow
                )));
            }
            first_valids[s] = fv;
        }

        // Build MAs using specific wrappers (covering common types sma/ema; extend as needed)
        let fast_type = params.fast_ma_type.as_deref().unwrap_or("sma");
        let slow_type = params.slow_ma_type.as_deref().unwrap_or("sma");

        let fast_tm_host =
            Self::compute_ma_host_time_major(fast_type, data_tm_f32, cols, rows, fast)?;
        let slow_tm_host =
            Self::compute_ma_host_time_major(slow_type, data_tm_f32, cols, rows, slow)?;
        let fast_dev = DeviceArrayF32 {
            buf: DeviceBuffer::from_slice(&fast_tm_host)
                .map_err(|e| CudaMabError::Cuda(e.to_string()))?,
            rows,
            cols,
        };
        let slow_dev = DeviceArrayF32 {
            buf: DeviceBuffer::from_slice(&slow_tm_host)
                .map_err(|e| CudaMabError::Cuda(e.to_string()))?,
            rows,
            cols,
        };

        // Outputs
        let elems = cols * rows;
        let mut d_upper: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(|e| CudaMabError::Cuda(e.to_string()))?;
        let mut d_middle: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(|e| CudaMabError::Cuda(e.to_string()))?;
        let mut d_lower: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(|e| CudaMabError::Cuda(e.to_string()))?;

        let mut func: Function = self
            .module
            .get_function("mab_many_series_one_param_time_major_f32")
            .map_err(|e| CudaMabError::Cuda(e.to_string()))?;

        // We run 1 thread per series (sequential time), grid.y=cols
        let grid = GridSize::xyz(1, cols as u32, 1);
        let block = BlockSize::xyz(1, 1, 1);
        unsafe {
            let mut f_ptr = fast_dev.buf.as_device_ptr().as_raw();
            let mut s_ptr = slow_dev.buf.as_device_ptr().as_raw();
            let mut first_ptr = DeviceBuffer::from_slice(&first_valids)
                .map_err(|e| CudaMabError::Cuda(e.to_string()))?
                .as_device_ptr()
                .as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut fp_i = fast as i32;
            let mut sp_i = slow as i32;
            let mut upf = params.devup.unwrap_or(1.0) as f32;
            let mut dnf = params.devdn.unwrap_or(1.0) as f32;
            let mut up_ptr = d_upper.as_device_ptr().as_raw();
            let mut mid_ptr = d_middle.as_device_ptr().as_raw();
            let mut lo_ptr = d_lower.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut f_ptr as *mut _ as *mut c_void,
                &mut s_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut fp_i as *mut _ as *mut c_void,
                &mut sp_i as *mut _ as *mut c_void,
                &mut upf as *mut _ as *mut c_void,
                &mut dnf as *mut _ as *mut c_void,
                &mut up_ptr as *mut _ as *mut c_void,
                &mut mid_ptr as *mut _ as *mut c_void,
                &mut lo_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&mut func, grid, block, 0, args)
                .map_err(|e| CudaMabError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaMabError::Cuda(e.to_string()))?;
        Ok(DeviceArrayF32Triplet {
            upper: DeviceArrayF32 {
                buf: d_upper,
                rows,
                cols,
            },
            middle: DeviceArrayF32 {
                buf: d_middle,
                rows,
                cols,
            },
            lower: DeviceArrayF32 {
                buf: d_lower,
                rows,
                cols,
            },
        })
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "mab",
                "batch_dev",
                "mab_cuda_batch_dev",
                "60k_x_49combos",
                prep_mab_batch_box,
            )
            .with_inner_iters(4),
            CudaBenchScenario::new(
                "mab",
                "many_series_one_param",
                "mab_cuda_many_series_one_param",
                "128x1m",
                prep_mab_many_series_box,
            )
            .with_inner_iters(2),
        ]
    }

    struct MabBatchState {
        cuda: CudaMab,
        d_up: DeviceBuffer<f32>,
        d_mid: DeviceBuffer<f32>,
        d_lo: DeviceBuffer<f32>,
        // keep inputs
        price: Vec<f32>,
        combos: MabBatchRange,
        rows: usize,
        len: usize,
    }

    impl CudaBenchState for MabBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .mab_batch_dev(&self.price, &self.combos)
                .expect("mab batch");
            self.cuda.stream.synchronize().unwrap();
        }
    }

    fn prep_mab_batch() -> MabBatchState {
        let cuda = CudaMab::new(0).expect("cuda mab");
        let len = 60_000usize;
        let mut price = vec![f32::NAN; len];
        for i in 10..len {
            let x = i as f32;
            price[i] = (x * 0.001).sin() + 0.001 * x;
        }
        let combos = MabBatchRange {
            fast_period: (10, 22, 4),
            slow_period: (50, 74, 12),
            devup: (1.0, 1.0, 0.0),
            devdn: (1.0, 1.0, 0.0),
            fast_ma_type: ("sma".into(), "sma".into(), "".into()),
            slow_ma_type: ("sma".into(), "sma".into(), "".into()),
        };
        let rows = ((combos.fast_period.1 - combos.fast_period.0) / combos.fast_period.2 + 1)
            * ((combos.slow_period.1 - combos.slow_period.0) / combos.slow_period.2 + 1) as usize;
        let elems = rows * len;
        let d_up: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.unwrap();
        let d_mid: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.unwrap();
        let d_lo: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.unwrap();
        MabBatchState {
            cuda,
            d_up,
            d_mid,
            d_lo,
            price,
            combos,
            rows,
            len,
        }
    }
    fn prep_mab_batch_box() -> Box<dyn CudaBenchState> {
        Box::new(prep_mab_batch())
    }

    struct MabManySeriesState {
        cuda: CudaMab,
        tm: Vec<f32>,
        cols: usize,
        rows: usize,
        p: MabParams,
    }
    impl CudaBenchState for MabManySeriesState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .mab_many_series_one_param_time_major_dev(&self.tm, self.cols, self.rows, &self.p)
                .unwrap();
            self.cuda.stream.synchronize().unwrap();
        }
    }
    fn prep_mab_many_series() -> MabManySeriesState {
        let cuda = CudaMab::new(0).expect("cuda mab");
        let cols = 128usize;
        let rows = 1_000_000usize;
        let mut tm = vec![f32::NAN; cols * rows];
        for s in 0..cols {
            for r in s..rows {
                let x = (r as f32) + 0.1 * (s as f32);
                tm[r * cols + s] = (x * 0.002).sin() + 0.0005 * x;
            }
        }
        let p = MabParams {
            fast_period: Some(10),
            slow_period: Some(50),
            devup: Some(1.0),
            devdn: Some(1.0),
            fast_ma_type: Some("sma".into()),
            slow_ma_type: Some("sma".into()),
        };
        MabManySeriesState {
            cuda,
            tm,
            cols,
            rows,
            p,
        }
    }
    fn prep_mab_many_series_box() -> Box<dyn CudaBenchState> {
        Box::new(prep_mab_many_series())
    }
}
