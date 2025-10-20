//! CUDA support for the Stochastic Oscillator (Stoch).
//!
//! Goals (ALMA-parity):
//! - PTX load from OUT_DIR with DetermineTargetFromContext + OptLevel O2 (fallbacks on error)
//! - NON_BLOCKING stream
//! - Batch (one series × many params) and Many-series × one param (time-major)
//! - Warmup/NaN semantics identical to scalar implementation
//! - Where helpful, reuse existing CUDA MA wrappers via the thin `CudaMaSelector`
//!   for the slow %K and %D smoothing stages.
//!
//! Implementation notes:
//! - Batch path computes raw %K once per unique `fastk_period` by precomputing
//!   rolling highest-high/lowest-low on host (O(n)), then launches a light
//!   kernel to convert (close, hh, ll) -> raw %K in FP32. Smoothing is done via
//!   the MA selector per-row to match the scalar `ma` dispatch semantics.
//! - Many-series path operates on time-major OHLC inputs with a shared param
//!   set; raw %K uses an O(period) per-step loop which is sufficient for common
//!   periods (~14). Smoothing uses native SMA/EMA many-series wrappers when
//!   applicable, otherwise falls back to per-series MA selection.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::ma_selector::{CudaMaData, CudaMaSelector};
use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::stoch::{StochBatchRange, StochParams};
use crate::indicators::utility_functions::{max_rolling, min_rolling};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::mem_get_info;
use cust::memory::{AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaStochError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaStochError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaStochError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaStochError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaStochError {}

pub struct CudaStoch {
    module: Module,
    stream: Stream,
    _context: Context,
}

pub struct CudaStochBatch {
    pub k: DeviceArrayF32,
    pub d: DeviceArrayF32,
    pub combos: Vec<StochParams>,
}

impl CudaStoch {
    pub fn new(device_id: usize) -> Result<Self, CudaStochError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaStochError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaStochError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/stoch_kernel.ptx"));
        let module = Module::from_ptx(
            ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O2),
            ],
        )
        .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
        .or_else(|_| Module::from_ptx(ptx, &[]))
        .map_err(|e| CudaStochError::Cuda(e.to_string()))?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaStochError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    // ---------------------- Batch (one series × many params) ----------------------
    pub fn stoch_batch_dev(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        close_f32: &[f32],
        sweep: &StochBatchRange,
    ) -> Result<CudaStochBatch, CudaStochError> {
        let len = high_f32.len();
        if len == 0 || low_f32.len() != len || close_f32.len() != len {
            return Err(CudaStochError::InvalidInput(
                "inputs must be non-empty and same length".into(),
            ));
        }

        let first_valid = (0..len)
            .find(|&i| {
                high_f32[i].is_finite() && low_f32[i].is_finite() && close_f32[i].is_finite()
            })
            .ok_or_else(|| CudaStochError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid_stoch(sweep);
        if combos.is_empty() {
            return Err(CudaStochError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        let max_fkp = combos
            .iter()
            .map(|c| c.fastk_period.unwrap_or(14))
            .max()
            .unwrap_or(14);
        if len - first_valid < max_fkp {
            return Err(CudaStochError::InvalidInput(format!(
                "not enough valid data for fastk {} (tail = {})",
                max_fkp,
                len - first_valid
            )));
        }

        // VRAM check (rough): close + hh + ll + kraw + 2*outputs
        let rows = combos.len();
        let est_bytes =
            (len * 4) * (1 /*close*/ + 1 /*hh*/ + 1 /*ll*/ + 1/*kraw*/) + (rows * len * 4) * 2;
        if let Ok((free, _)) = mem_get_info() {
            let headroom = 64 * 1024 * 1024usize;
            if est_bytes.saturating_add(headroom) > free {
                return Err(CudaStochError::InvalidInput(format!(
                    "insufficient device memory: need ~{:.2} MB (incl. outputs)",
                    (est_bytes as f64) / (1024.0 * 1024.0)
                )));
            }
        }

        // Upload base inputs once
        let d_close =
            DeviceBuffer::from_slice(close_f32).map_err(|e| CudaStochError::Cuda(e.to_string()))?;

        // Group rows by fastk to reuse raw %K
        let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for (row, prm) in combos.iter().enumerate() {
            let fkp = prm.fastk_period.unwrap_or(14);
            groups.entry(fkp).or_default().push(row);
        }

        // Host accumulation buffers for final outputs
        let mut host_k = vec![f32::NAN; combos.len() * len];
        let mut host_d = vec![f32::NAN; combos.len() * len];
        let selector = CudaMaSelector::new(0);

        // Launch kernel handle
        let func = self
            .module
            .get_function("stoch_k_raw_from_hhll_f32")
            .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
        let block: BlockSize = (256, 1, 1).into();
        let grid: GridSize = (1, 1, 1).into();

        // Temporary device buffers (reused per group)
        let mut d_hh: Option<DeviceBuffer<f32>> = None;
        let mut d_ll: Option<DeviceBuffer<f32>> = None;
        let mut d_kraw: Option<DeviceBuffer<f32>> = None;

        for (fkp, rows_in_group) in groups {
            // Build hh/ll on host (use existing f64 helpers, then cast)
            let mut hh = vec![f32::NAN; len];
            let mut ll = vec![f32::NAN; len];
            let warm = first_valid + fkp - 1;
            let high_f64: Vec<f64> = high_f32.iter().map(|&v| v as f64).collect();
            let low_f64: Vec<f64> = low_f32.iter().map(|&v| v as f64).collect();
            let highs = max_rolling(&high_f64[first_valid..], fkp)
                .map_err(|e| CudaStochError::InvalidInput(e.to_string()))?;
            let lows = min_rolling(&low_f64[first_valid..], fkp)
                .map_err(|e| CudaStochError::InvalidInput(e.to_string()))?;
            for (i, &v) in highs.iter().enumerate() {
                if v.is_finite() {
                    hh[first_valid + i] = v as f32;
                }
            }
            for (i, &v) in lows.iter().enumerate() {
                if v.is_finite() {
                    ll[first_valid + i] = v as f32;
                }
            }

            // H2D hh/ll and allocate kraw
            if d_hh.as_ref().map(|b| b.len()).unwrap_or(0) != len {
                d_hh = Some(
                    unsafe { DeviceBuffer::uninitialized(len) }
                        .map_err(|e| CudaStochError::Cuda(e.to_string()))?,
                );
            }
            if d_ll.as_ref().map(|b| b.len()).unwrap_or(0) != len {
                d_ll = Some(
                    unsafe { DeviceBuffer::uninitialized(len) }
                        .map_err(|e| CudaStochError::Cuda(e.to_string()))?,
                );
            }
            if d_kraw.as_ref().map(|b| b.len()).unwrap_or(0) != len {
                d_kraw = Some(
                    unsafe { DeviceBuffer::uninitialized(len) }
                        .map_err(|e| CudaStochError::Cuda(e.to_string()))?,
                );
            }
            let d_hh_ref = d_hh.as_mut().unwrap();
            let d_ll_ref = d_ll.as_mut().unwrap();
            let d_kraw_ref = d_kraw.as_mut().unwrap();
            d_hh_ref
                .copy_from(&hh)
                .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
            d_ll_ref
                .copy_from(&ll)
                .map_err(|e| CudaStochError::Cuda(e.to_string()))?;

            // Launch raw %K kernel
            unsafe {
                let mut p_close = d_close.as_device_ptr().as_raw();
                let mut p_hh = d_hh_ref.as_device_ptr().as_raw();
                let mut p_ll = d_ll_ref.as_device_ptr().as_raw();
                let mut p_len = len as i32;
                let mut p_fv = first_valid as i32;
                let mut p_fastk = fkp as i32;
                let mut p_out = d_kraw_ref.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut p_close as *mut _ as *mut c_void,
                    &mut p_hh as *mut _ as *mut c_void,
                    &mut p_ll as *mut _ as *mut c_void,
                    &mut p_len as *mut _ as *mut c_void,
                    &mut p_fv as *mut _ as *mut c_void,
                    &mut p_fastk as *mut _ as *mut c_void,
                    &mut p_out as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
            }

            // Stage raw %K to pinned host once per group
            let mut kraw_host: LockedBuffer<f32> = unsafe {
                LockedBuffer::uninitialized(len).map_err(|e| CudaStochError::Cuda(e.to_string()))?
            };
            unsafe {
                d_kraw_ref
                    .async_copy_to(kraw_host.as_mut_slice(), &self.stream)
                    .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
            }
            self.stream
                .synchronize()
                .map_err(|e| CudaStochError::Cuda(e.to_string()))?;

            let kraw_slice = kraw_host.as_slice();

            // For each row in this fastk group: apply slow K and D smoothing via selector
            for &row in &rows_in_group {
                let prm = &combos[row];

                // slow %K
                let slowk_ma = prm
                    .slowk_ma_type
                    .as_ref()
                    .map(|s| s.as_str())
                    .unwrap_or("sma");
                let slowk_p = prm.slowk_period.unwrap_or(3);
                let k_dev = selector
                    .ma_to_device(slowk_ma, CudaMaData::SliceF32(kraw_slice), slowk_p)
                    .map_err(|e| CudaStochError::Cuda(format!("slowK: {}", e)))?;
                let mut k_row_host: LockedBuffer<f32> = unsafe {
                    LockedBuffer::uninitialized(len)
                        .map_err(|e| CudaStochError::Cuda(e.to_string()))?
                };
                unsafe {
                    k_dev
                        .buf
                        .async_copy_to(k_row_host.as_mut_slice(), &self.stream)
                        .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
                }
                self.stream
                    .synchronize()
                    .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
                // copy into host_k row-major
                host_k[row * len..(row + 1) * len].copy_from_slice(k_row_host.as_slice());

                // %D over slowK
                let slowd_ma = prm
                    .slowd_ma_type
                    .as_ref()
                    .map(|s| s.as_str())
                    .unwrap_or("sma");
                let slowd_p = prm.slowd_period.unwrap_or(3);
                let d_dev = selector
                    .ma_to_device(
                        slowd_ma,
                        CudaMaData::SliceF32(k_row_host.as_slice()),
                        slowd_p,
                    )
                    .map_err(|e| CudaStochError::Cuda(format!("slowD: {}", e)))?;
                let mut d_row_host: LockedBuffer<f32> = unsafe {
                    LockedBuffer::uninitialized(len)
                        .map_err(|e| CudaStochError::Cuda(e.to_string()))?
                };
                unsafe {
                    d_dev
                        .buf
                        .async_copy_to(d_row_host.as_mut_slice(), &self.stream)
                        .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
                }
                self.stream
                    .synchronize()
                    .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
                host_d[row * len..(row + 1) * len].copy_from_slice(d_row_host.as_slice());
            }
        }

        // Upload final K and D matrices
        let mut d_k: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(host_k.len()) }
            .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
        let mut d_d: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(host_d.len()) }
            .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
        d_k.copy_from(&host_k)
            .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
        d_d.copy_from(&host_d)
            .map_err(|e| CudaStochError::Cuda(e.to_string()))?;

        Ok(CudaStochBatch {
            k: DeviceArrayF32 {
                buf: d_k,
                rows: combos.len(),
                cols: len,
            },
            d: DeviceArrayF32 {
                buf: d_d,
                rows: combos.len(),
                cols: len,
            },
            combos,
        })
    }

    // ---------------- Many-series × one param (time-major inputs) ----------------
    pub fn stoch_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &StochParams,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32), CudaStochError> {
        if cols == 0 || rows == 0 {
            return Err(CudaStochError::InvalidInput(
                "series dims must be positive".into(),
            ));
        }
        let total = cols * rows;
        if high_tm.len() != total || low_tm.len() != total || close_tm.len() != total {
            return Err(CudaStochError::InvalidInput(
                "time-major inputs must all be rows*cols".into(),
            ));
        }
        // First-valid per series (column) in time-major
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for r in 0..rows {
                let idx = r * cols + s;
                if high_tm[idx].is_finite() && low_tm[idx].is_finite() && close_tm[idx].is_finite()
                {
                    fv = Some(r as i32);
                    break;
                }
            }
            first_valids[s] =
                fv.ok_or_else(|| CudaStochError::InvalidInput(format!("series {} all NaN", s)))?;
        }

        let fastk = params.fastk_period.unwrap_or(14);
        let slowk_p = params.slowk_period.unwrap_or(3);
        let slowd_p = params.slowd_period.unwrap_or(3);
        let slowk_ty = params
            .slowk_ma_type
            .as_ref()
            .map(|s| s.as_str())
            .unwrap_or("sma");
        let slowd_ty = params
            .slowd_ma_type
            .as_ref()
            .map(|s| s.as_str())
            .unwrap_or("sma");

        if fastk == 0 || fastk > rows {
            return Err(CudaStochError::InvalidInput("invalid fastk period".into()));
        }

        // H2D
        let d_high =
            DeviceBuffer::from_slice(high_tm).map_err(|e| CudaStochError::Cuda(e.to_string()))?;
        let d_low =
            DeviceBuffer::from_slice(low_tm).map_err(|e| CudaStochError::Cuda(e.to_string()))?;
        let d_close =
            DeviceBuffer::from_slice(close_tm).map_err(|e| CudaStochError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
        let mut d_k_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total) }
            .map_err(|e| CudaStochError::Cuda(e.to_string()))?;

        // Kernel: raw %K time-major
        let func = self
            .module
            .get_function("stoch_many_series_one_param_f32")
            .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
        let block_x: u32 = 256;
        let grid_x: u32 = ((cols as u32) + block_x - 1) / block_x;
        let block: BlockSize = (block_x, 1, 1).into();
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        unsafe {
            let mut p_h = d_high.as_device_ptr().as_raw();
            let mut p_l = d_low.as_device_ptr().as_raw();
            let mut p_c = d_close.as_device_ptr().as_raw();
            let mut p_first = d_first.as_device_ptr().as_raw();
            let mut p_cols = cols as i32;
            let mut p_rows = rows as i32;
            let mut p_fastk = fastk as i32;
            let mut p_out = d_k_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_h as *mut _ as *mut c_void,
                &mut p_l as *mut _ as *mut c_void,
                &mut p_c as *mut _ as *mut c_void,
                &mut p_first as *mut _ as *mut c_void,
                &mut p_cols as *mut _ as *mut c_void,
                &mut p_rows as *mut _ as *mut c_void,
                &mut p_fastk as *mut _ as *mut c_void,
                &mut p_out as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
        }

        // Smoothing (native SMA/EMA many-series path if available; otherwise per-series)
        // K smoothing
        let k_tm: DeviceBuffer<f32> = if slowk_ty.eq_ignore_ascii_case("sma") {
            use crate::cuda::moving_averages::sma_wrapper::CudaSma;
            use crate::indicators::moving_averages::sma::SmaParams as SParams;
            let sma = CudaSma::new(0).map_err(|e| CudaStochError::Cuda(e.to_string()))?;
            let params = SParams {
                period: Some(slowk_p),
            };
            let dev = sma
                .sma_multi_series_one_param_time_major_dev_from_device(
                    &d_k_tm, &d_first, cols, rows, slowk_p,
                )
                .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
            dev.buf
        } else if slowk_ty.eq_ignore_ascii_case("ema") {
            use crate::cuda::moving_averages::ema_wrapper::CudaEma;
            use crate::indicators::moving_averages::ema::EmaParams as EParams;
            let ema = CudaEma::new(0).map_err(|e| CudaStochError::Cuda(e.to_string()))?;
            let params = EParams {
                period: Some(slowk_p),
            };
            // Stage K_tm to host and use EMA's time-major host API
            let mut k_tm_host = vec![0f32; total];
            d_k_tm
                .copy_to(&mut k_tm_host)
                .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
            let dev = ema
                .ema_many_series_one_param_time_major_dev(&k_tm_host, cols, rows, &params)
                .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
            dev.buf
        } else {
            // Fallback: per-series via selector
            let selector = CudaMaSelector::new(0);
            // Stage d_k_tm to host
            let mut k_tm_host = vec![0f32; total];
            d_k_tm
                .copy_to(&mut k_tm_host)
                .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
            let mut out_tm = vec![f32::NAN; total];
            for s in 0..cols {
                let mut series = vec![f32::NAN; rows];
                for r in 0..rows {
                    series[r] = k_tm_host[r * cols + s];
                }
                let dev = selector
                    .ma_to_device(slowk_ty, CudaMaData::SliceF32(&series), slowk_p)
                    .map_err(|e| CudaStochError::Cuda(format!("slowK many-series: {}", e)))?;
                let mut host_row = vec![0f32; rows];
                dev.buf
                    .copy_to(&mut host_row)
                    .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
                for r in 0..rows {
                    out_tm[r * cols + s] = host_row[r];
                }
            }
            let mut tmp: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total) }
                .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
            tmp.copy_from(&out_tm)
                .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
            tmp
        };

        // D smoothing over K_slow
        let d_tm: DeviceBuffer<f32> = if slowd_ty.eq_ignore_ascii_case("sma") {
            use crate::cuda::moving_averages::sma_wrapper::CudaSma;
            let sma = CudaSma::new(0).map_err(|e| CudaStochError::Cuda(e.to_string()))?;
            let dev = sma
                .sma_multi_series_one_param_time_major_dev_from_device(
                    &k_tm, &d_first, cols, rows, slowd_p,
                )
                .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
            dev.buf
        } else if slowd_ty.eq_ignore_ascii_case("ema") {
            use crate::cuda::moving_averages::ema_wrapper::CudaEma;
            let ema = CudaEma::new(0).map_err(|e| CudaStochError::Cuda(e.to_string()))?;
            // Stage K_tm to host and use EMA's time-major host API
            let mut k_tm_host = vec![0f32; total];
            k_tm.copy_to(&mut k_tm_host)
                .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
            use crate::indicators::moving_averages::ema::EmaParams as EParams;
            let params = EParams {
                period: Some(slowd_p),
            };
            let dev = ema
                .ema_many_series_one_param_time_major_dev(&k_tm_host, cols, rows, &params)
                .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
            dev.buf
        } else {
            // Fallback: per-series via selector again
            let selector = CudaMaSelector::new(0);
            let mut k_tm_host = vec![0f32; total];
            k_tm.copy_to(&mut k_tm_host)
                .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
            let mut out_tm = vec![f32::NAN; total];
            for s in 0..cols {
                let mut series = vec![f32::NAN; rows];
                for r in 0..rows {
                    series[r] = k_tm_host[r * cols + s];
                }
                let dev = selector
                    .ma_to_device(slowd_ty, CudaMaData::SliceF32(&series), slowd_p)
                    .map_err(|e| CudaStochError::Cuda(format!("slowD many-series: {}", e)))?;
                let mut host_row = vec![0f32; rows];
                dev.buf
                    .copy_to(&mut host_row)
                    .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
                for r in 0..rows {
                    out_tm[r * cols + s] = host_row[r];
                }
            }
            let mut tmp: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total) }
                .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
            tmp.copy_from(&out_tm)
                .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
            tmp
        };

        Ok((
            DeviceArrayF32 {
                buf: k_tm,
                rows,
                cols,
            },
            DeviceArrayF32 {
                buf: d_tm,
                rows,
                cols,
            },
        ))
    }
}

// -------- helpers --------
fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
    if step == 0 || start == end {
        return vec![start];
    }
    let mut v = Vec::new();
    let mut x = start;
    while x <= end {
        v.push(x);
        x = x.saturating_add(step);
    }
    v
}

fn expand_grid_stoch(r: &StochBatchRange) -> Vec<StochParams> {
    let fastk = axis_usize(r.fastk_period);
    let slowk = axis_usize(r.slowk_period);
    let slowd = axis_usize(r.slowd_period);
    let mut out = Vec::new();
    for fk in &fastk {
        for sk in &slowk {
            for sd in &slowd {
                out.push(StochParams {
                    fastk_period: Some(*fk),
                    slowk_period: Some(*sk),
                    slowk_ma_type: Some(r.slowk_ma_type.0.clone()),
                    slowd_period: Some(*sd),
                    slowd_ma_type: Some(r.slowd_ma_type.0.clone()),
                });
            }
        }
    }
    out
}

// ---------- Benches ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_00; // 100k
    const PARAM_SWEEP: usize = 128;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = 3 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = 2 * ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>(); // K + D
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    fn synth_hlc_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i];
            if !v.is_finite() {
                continue;
            }
            let x = i as f32 * 0.0037;
            let off = (0.0041 * x.sin()).abs() + 0.15;
            high[i] = v + off;
            low[i] = v - off;
        }
        (high, low)
    }

    struct StochBatchState {
        cuda: CudaStoch,
        high: Vec<f32>,
        low: Vec<f32>,
        close: Vec<f32>,
        sweep: StochBatchRange,
    }
    impl CudaBenchState for StochBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .stoch_batch_dev(&self.high, &self.low, &self.close, &self.sweep)
                .expect("stoch batch");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaStoch::new(0).expect("cuda stoch");
        let close = gen_series(ONE_SERIES_LEN);
        let (high, low) = synth_hlc_from_close(&close);
        let sweep = StochBatchRange {
            fastk_period: (14, 14 + (PARAM_SWEEP - 1), 1),
            slowk_period: (3, 3, 0),
            slowk_ma_type: ("sma".into(), "sma".into(), 0.0),
            slowd_period: (3, 3, 0),
            slowd_ma_type: ("sma".into(), "sma".into(), 0.0),
        };
        Box::new(StochBatchState {
            cuda,
            high,
            low,
            close,
            sweep,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "stoch",
            "one_series_many_params",
            "stoch_cuda_batch_dev",
            "100k_x_128",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}
