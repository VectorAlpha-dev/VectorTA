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

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::cuda::moving_averages::ma_selector::{CudaMaData, CudaMaSelector};
use crate::indicators::stoch::{StochBatchRange, StochParams};
use crate::indicators::utility_functions::{max_rolling, min_rolling};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::memory::mem_get_info;
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
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaStochError::Cuda(e.to_string()))?;
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

        Ok(Self { module, stream, _context: context })
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
            .find(|&i| high_f32[i].is_finite() && low_f32[i].is_finite() && close_f32[i].is_finite())
            .ok_or_else(|| CudaStochError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid_stoch(sweep);
        if combos.is_empty() {
            return Err(CudaStochError::InvalidInput("no parameter combinations".into()));
        }
        let max_fkp = combos.iter().map(|c| c.fastk_period.unwrap_or(14)).max().unwrap_or(14);
        if len - first_valid < max_fkp {
            return Err(CudaStochError::InvalidInput(format!(
                "not enough valid data for fastk {} (tail = {})",
                max_fkp, len - first_valid
            )));
        }

        // Rough VRAM check: close + hh + ll + kraw + 2 * (rows*len outputs)
        let rows_total = combos.len();
        let est_bytes = (len * 4) * (1 + 1 + 1 + 1) + (rows_total * len * 4) * 2;
        if let Ok((free, _)) = mem_get_info() {
            let headroom = 64 * 1024 * 1024usize;
            if est_bytes.saturating_add(headroom) > free {
                return Err(CudaStochError::InvalidInput(format!(
                    "insufficient device memory: need ~{:.2} MB (incl. outputs)",
                    (est_bytes as f64) / (1024.0 * 1024.0)
                )));
            }
        }

        // Upload base CLOSE once
        let d_close = DeviceBuffer::from_slice(close_f32)
            .map_err(|e| CudaStochError::Cuda(e.to_string()))?;

        // Final outputs on device (row-major: [combos, len])
        let total_out = rows_total * len;
        let mut d_k: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total_out) }
            .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
        let mut d_d: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total_out) }
            .map_err(|e| CudaStochError::Cuda(e.to_string()))?;

        // Kernel handles
        let func_kraw = self
            .module
            .get_function("stoch_k_raw_from_hhll_f32")
            .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
        let func_pack = self
            .module
            .get_function("pack_row_broadcast_rowmajor_f32")
            .map_err(|e| CudaStochError::Cuda(e.to_string()))?;

        // Group parameter rows by fastk period so we reuse Kraw.
        use std::collections::HashMap;
        let mut by_fastk: HashMap<usize, Vec<usize>> = HashMap::new();
        for (row, prm) in combos.iter().enumerate() {
            by_fastk.entry(prm.fastk_period.unwrap_or(14)).or_default().push(row);
        }

        // Reusable device temporaries
        let mut d_hh: Option<DeviceBuffer<f32>> = None;
        let mut d_ll: Option<DeviceBuffer<f32>> = None;
        let mut d_kraw: Option<DeviceBuffer<f32>> = None;

        // Note: For smoothing stages we must pass the first index where the input is finite.
        // This depends on the upstream stage:
        //  - Kraw first valid: first_valid_ohlc + fastk - 1
        //  - slowK first valid: (first_valid_kraw) + slowk_period - 1

        // Helper: 1D launch config
        let launch_1d = |n: usize| -> (GridSize, BlockSize) {
            let block_x: u32 = 256;
            let grid_x: u32 = ((n as u32) + block_x - 1) / block_x;
            ((grid_x.max(1), 1, 1).into(), (block_x, 1, 1).into())
        };

        let norm = |s: &str| s.to_ascii_lowercase();

        for (fkp, rows_in_group) in by_fastk {
            // Build hh/ll on host (f64 helpers → cast to f32)
            let mut hh = vec![f32::NAN; len];
            let mut ll = vec![f32::NAN; len];
            let high_f64: Vec<f64> = high_f32.iter().map(|&v| v as f64).collect();
            let low_f64: Vec<f64> = low_f32.iter().map(|&v| v as f64).collect();
            let highs = max_rolling(&high_f64[first_valid..], fkp)
                .map_err(|e| CudaStochError::InvalidInput(e.to_string()))?;
            let lows = min_rolling(&low_f64[first_valid..], fkp)
                .map_err(|e| CudaStochError::InvalidInput(e.to_string()))?;
            for (i, &v) in highs.iter().enumerate() { if v.is_finite() { hh[first_valid + i] = v as f32; } }
            for (i, &v) in lows.iter().enumerate() { if v.is_finite() { ll[first_valid + i] = v as f32; } }

            // Ensure device buffers sized
            if d_hh.as_ref().map(|b| b.len()).unwrap_or(0) != len {
                d_hh = Some(unsafe { DeviceBuffer::uninitialized(len) }
                    .map_err(|e| CudaStochError::Cuda(e.to_string()))?);
            }
            if d_ll.as_ref().map(|b| b.len()).unwrap_or(0) != len {
                d_ll = Some(unsafe { DeviceBuffer::uninitialized(len) }
                    .map_err(|e| CudaStochError::Cuda(e.to_string()))?);
            }
            if d_kraw.as_ref().map(|b| b.len()).unwrap_or(0) != len {
                d_kraw = Some(unsafe { DeviceBuffer::uninitialized(len) }
                    .map_err(|e| CudaStochError::Cuda(e.to_string()))?);
            }
            let d_hh_ref = d_hh.as_mut().unwrap();
            let d_ll_ref = d_ll.as_mut().unwrap();
            let d_kraw_ref = d_kraw.as_mut().unwrap();

            // Async H2D hh/ll
            unsafe {
                d_hh_ref
                    .async_copy_from(&hh, &self.stream)
                    .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
                d_ll_ref
                    .async_copy_from(&ll, &self.stream)
                    .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
            }

            // Launch raw %K (grid‑stride parallel)
            {
                let (grid, block) = launch_1d(len);
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
                        .launch(&func_kraw, grid, block, 0, args)
                        .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
                }
            }

            // Partition rows by slowK settings
            #[derive(Hash, Eq, PartialEq, Clone)]
            struct SlowKKey { ty: String, p: usize }
            let mut by_slowk: HashMap<SlowKKey, Vec<usize>> = HashMap::new();
            for &row in &rows_in_group {
                let prm = &combos[row];
                let ty = norm(prm.slowk_ma_type.as_deref().unwrap_or("sma"));
                let p = prm.slowk_period.unwrap_or(3);
                by_slowk.entry(SlowKKey { ty, p }).or_default().push(row);
            }

            for (sk_key, rows_sk) in by_slowk {
                let first_kraw = first_valid + fkp - 1;
                let d_first_kraw = DeviceBuffer::from_slice(&[first_kraw as i32])
                    .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
                // Compute slowK once from device d_kraw_ref (1 column time-major)
                let slowk_dev_buf = if sk_key.ty == "sma" {
                    use crate::cuda::moving_averages::sma_wrapper::CudaSma;
                    let sma = CudaSma::new(0).map_err(|e| CudaStochError::Cuda(e.to_string()))?;
                    let dev = sma
                        .sma_multi_series_one_param_time_major_dev_from_device(
                            d_kraw_ref, &d_first_kraw, 1, len, sk_key.p,
                        )
                        .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
                    dev.buf
                } else if sk_key.ty == "ema" {
                    // Fallback to host path (EMA lacks D2D variant currently)
                    use crate::cuda::moving_averages::ema_wrapper::CudaEma;
                    use crate::indicators::moving_averages::ema::EmaParams as EParams;
                    let ema = CudaEma::new(0).map_err(|e| CudaStochError::Cuda(e.to_string()))?;
                    let mut kraw_host: LockedBuffer<f32> = unsafe {
                        LockedBuffer::uninitialized(len)
                            .map_err(|e| CudaStochError::Cuda(e.to_string()))?
                    };
                    unsafe {
                        d_kraw_ref
                            .async_copy_to(kraw_host.as_mut_slice(), &self.stream)
                            .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
                    }
                    self.stream
                        .synchronize()
                        .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
                    let params = EParams { period: Some(sk_key.p) };
                    let dev = ema
                        .ema_many_series_one_param_time_major_dev(
                            kraw_host.as_slice(), 1, len, &params,
                        )
                        .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
                    dev.buf
                } else {
                    // Generic MA via selector: copy once to host then run selector
                    let selector = CudaMaSelector::new(0);
                    let mut kraw_host: LockedBuffer<f32> = unsafe {
                        LockedBuffer::uninitialized(len)
                            .map_err(|e| CudaStochError::Cuda(e.to_string()))?
                    };
                    unsafe {
                        d_kraw_ref
                            .async_copy_to(kraw_host.as_mut_slice(), &self.stream)
                            .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
                    }
                    self.stream
                        .synchronize()
                        .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
                    let dev = selector
                        .ma_to_device(&sk_key.ty, CudaMaData::SliceF32(kraw_host.as_slice()), sk_key.p)
                        .map_err(|e| CudaStochError::Cuda(format!("slowK: {}", e)))?;
                    dev.buf
                };

                // Broadcast slowK to all K rows in this subgroup
                {
                    let idx_i32: Vec<i32> = rows_sk.iter().map(|&r| r as i32).collect();
                    let d_rows = DeviceBuffer::from_slice(&idx_i32)
                        .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
                    let (grid, block) = launch_1d(len);
                    unsafe {
                        let mut p_src = slowk_dev_buf.as_device_ptr().as_raw();
                        let mut p_len = len as i32;
                        let mut p_rows = d_rows.as_device_ptr().as_raw();
                        let mut p_nrows = rows_sk.len() as i32;
                        let mut p_dst = d_k.as_device_ptr().as_raw();
                        let mut p_stride = len as i32;
                        let args: &mut [*mut c_void] = &mut [
                            &mut p_src as *mut _ as *mut c_void,
                            &mut p_len as *mut _ as *mut c_void,
                            &mut p_rows as *mut _ as *mut c_void,
                            &mut p_nrows as *mut _ as *mut c_void,
                            &mut p_dst as *mut _ as *mut c_void,
                            &mut p_stride as *mut _ as *mut c_void,
                        ];
                        self.stream
                            .launch(&func_pack, grid, block, 0, args)
                            .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
                    }
                }

                // Partition by slowD settings
                #[derive(Hash, Eq, PartialEq, Clone)]
                struct SlowDKey { ty: String, p: usize }
                let mut by_slowd: HashMap<SlowDKey, Vec<usize>> = HashMap::new();
                for &row in &rows_sk {
                    let prm = &combos[row];
                    let ty = norm(prm.slowd_ma_type.as_deref().unwrap_or("sma"));
                    let p = prm.slowd_period.unwrap_or(3);
                    by_slowd.entry(SlowDKey { ty, p }).or_default().push(row);
                }

                for (sd_key, rows_sd) in by_slowd {
                    let first_slowk = first_valid + fkp - 1 + sk_key.p - 1;
                    let d_first_slowk = DeviceBuffer::from_slice(&[first_slowk as i32])
                        .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
                    // slowD once from device slowK
                    let slowd_dev_buf = if sd_key.ty == "sma" {
                        use crate::cuda::moving_averages::sma_wrapper::CudaSma;
                        let sma = CudaSma::new(0).map_err(|e| CudaStochError::Cuda(e.to_string()))?;
                        let dev = sma
                            .sma_multi_series_one_param_time_major_dev_from_device(
                                &slowk_dev_buf, &d_first_slowk, 1, len, sd_key.p,
                            )
                            .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
                        dev.buf
                    } else if sd_key.ty == "ema" {
                        use crate::cuda::moving_averages::ema_wrapper::CudaEma;
                        use crate::indicators::moving_averages::ema::EmaParams as EParams;
                        let ema = CudaEma::new(0).map_err(|e| CudaStochError::Cuda(e.to_string()))?;
                        let mut slowk_host: LockedBuffer<f32> = unsafe {
                            LockedBuffer::uninitialized(len)
                                .map_err(|e| CudaStochError::Cuda(e.to_string()))?
                        };
                        unsafe {
                            slowk_dev_buf
                                .async_copy_to(slowk_host.as_mut_slice(), &self.stream)
                                .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
                        }
                        self.stream
                            .synchronize()
                            .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
                        let params = EParams { period: Some(sd_key.p) };
                        let dev = ema
                            .ema_many_series_one_param_time_major_dev(
                                slowk_host.as_slice(), 1, len, &params,
                            )
                            .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
                        dev.buf
                    } else {
                        let selector = CudaMaSelector::new(0);
                        let mut slowk_host: LockedBuffer<f32> = unsafe {
                            LockedBuffer::uninitialized(len)
                                .map_err(|e| CudaStochError::Cuda(e.to_string()))?
                        };
                        unsafe {
                            slowk_dev_buf
                                .async_copy_to(slowk_host.as_mut_slice(), &self.stream)
                                .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
                        }
                        self.stream
                            .synchronize()
                            .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
                        let dev = selector
                            .ma_to_device(&sd_key.ty, CudaMaData::SliceF32(slowk_host.as_slice()), sd_key.p)
                            .map_err(|e| CudaStochError::Cuda(format!("slowD: {}", e)))?;
                        dev.buf
                    };

                    // Broadcast slowD into D matrix rows
                    let idx_i32: Vec<i32> = rows_sd.iter().map(|&r| r as i32).collect();
                    let d_rows = DeviceBuffer::from_slice(&idx_i32)
                        .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
                    let (grid, block) = launch_1d(len);
                    unsafe {
                        let mut p_src = slowd_dev_buf.as_device_ptr().as_raw();
                        let mut p_len = len as i32;
                        let mut p_rows = d_rows.as_device_ptr().as_raw();
                        let mut p_nrows = rows_sd.len() as i32;
                        let mut p_dst = d_d.as_device_ptr().as_raw();
                        let mut p_stride = len as i32;
                        let args: &mut [*mut c_void] = &mut [
                            &mut p_src as *mut _ as *mut c_void,
                            &mut p_len as *mut _ as *mut c_void,
                            &mut p_rows as *mut _ as *mut c_void,
                            &mut p_nrows as *mut _ as *mut c_void,
                            &mut p_dst as *mut _ as *mut c_void,
                            &mut p_stride as *mut _ as *mut c_void,
                        ];
                        self.stream
                            .launch(&func_pack, grid, block, 0, args)
                            .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
                    }
                }
            }
        }

        // Ensure completion
        self.stream
            .synchronize()
            .map_err(|e| CudaStochError::Cuda(e.to_string()))?;

        Ok(CudaStochBatch {
            k: DeviceArrayF32 { buf: d_k, rows: rows_total, cols: len },
            d: DeviceArrayF32 { buf: d_d, rows: rows_total, cols: len },
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
                if high_tm[idx].is_finite() && low_tm[idx].is_finite() && close_tm[idx].is_finite() {
                    fv = Some(r as i32);
                    break;
                }
            }
            first_valids[s] = fv.ok_or_else(|| CudaStochError::InvalidInput(format!(
                "series {} all NaN", s
            )))?;
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
        let d_high = DeviceBuffer::from_slice(high_tm).map_err(|e| CudaStochError::Cuda(e.to_string()))?;
        let d_low = DeviceBuffer::from_slice(low_tm).map_err(|e| CudaStochError::Cuda(e.to_string()))?;
        let d_close = DeviceBuffer::from_slice(close_tm).map_err(|e| CudaStochError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
        let mut d_k_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(total) }.map_err(|e| CudaStochError::Cuda(e.to_string()))?;

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
            let params = SParams { period: Some(slowk_p) };
            let dev = sma
                .sma_multi_series_one_param_time_major_dev_from_device(
                    &d_k_tm,
                    &d_first,
                    cols,
                    rows,
                    slowk_p,
                )
                .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
            dev.buf
        } else if slowk_ty.eq_ignore_ascii_case("ema") {
            use crate::cuda::moving_averages::ema_wrapper::CudaEma;
            use crate::indicators::moving_averages::ema::EmaParams as EParams;
            let ema = CudaEma::new(0).map_err(|e| CudaStochError::Cuda(e.to_string()))?;
            let params = EParams { period: Some(slowk_p) };
            // Stage K_tm to host and use EMA's time-major host API
            let mut k_tm_host = vec![0f32; total];
            d_k_tm
                .copy_to(&mut k_tm_host)
                .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
            let dev = ema
                .ema_many_series_one_param_time_major_dev(
                    &k_tm_host,
                    cols,
                    rows,
                    &params,
                )
                .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
            dev.buf
        } else {
            // Fallback: per-series via selector
            let selector = CudaMaSelector::new(0);
            // Stage d_k_tm to host
            let mut k_tm_host = vec![0f32; total];
            d_k_tm.copy_to(&mut k_tm_host).map_err(|e| CudaStochError::Cuda(e.to_string()))?;
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
                dev.buf.copy_to(&mut host_row).map_err(|e| CudaStochError::Cuda(e.to_string()))?;
                for r in 0..rows {
                    out_tm[r * cols + s] = host_row[r];
                }
            }
            let mut tmp: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total) }
                .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
            tmp.copy_from(&out_tm).map_err(|e| CudaStochError::Cuda(e.to_string()))?;
            tmp
        };

        // D smoothing over K_slow
        let d_tm: DeviceBuffer<f32> = if slowd_ty.eq_ignore_ascii_case("sma") {
            use crate::cuda::moving_averages::sma_wrapper::CudaSma;
            let sma = CudaSma::new(0).map_err(|e| CudaStochError::Cuda(e.to_string()))?;
            let dev = sma
                .sma_multi_series_one_param_time_major_dev_from_device(
                    &k_tm,
                    &d_first,
                    cols,
                    rows,
                    slowd_p,
                )
                .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
            dev.buf
        } else if slowd_ty.eq_ignore_ascii_case("ema") {
            use crate::cuda::moving_averages::ema_wrapper::CudaEma;
            let ema = CudaEma::new(0).map_err(|e| CudaStochError::Cuda(e.to_string()))?;
            // Stage K_tm to host and use EMA's time-major host API
            let mut k_tm_host = vec![0f32; total];
            k_tm
                .copy_to(&mut k_tm_host)
                .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
            use crate::indicators::moving_averages::ema::EmaParams as EParams;
            let params = EParams { period: Some(slowd_p) };
            let dev = ema
                .ema_many_series_one_param_time_major_dev(
                    &k_tm_host,
                    cols,
                    rows,
                    &params,
                )
                .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
            dev.buf
        } else {
            // Fallback: per-series via selector again
            let selector = CudaMaSelector::new(0);
            let mut k_tm_host = vec![0f32; total];
            k_tm.copy_to(&mut k_tm_host).map_err(|e| CudaStochError::Cuda(e.to_string()))?;
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
                dev.buf.copy_to(&mut host_row).map_err(|e| CudaStochError::Cuda(e.to_string()))?;
                for r in 0..rows {
                    out_tm[r * cols + s] = host_row[r];
                }
            }
            let mut tmp: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total) }
                .map_err(|e| CudaStochError::Cuda(e.to_string()))?;
            tmp.copy_from(&out_tm).map_err(|e| CudaStochError::Cuda(e.to_string()))?;
            tmp
        };

        Ok(
            (
                DeviceArrayF32 { buf: k_tm, rows, cols },
                DeviceArrayF32 { buf: d_tm, rows, cols },
            )
        )
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
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::cuda::bench::helpers::gen_series;

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
            if !v.is_finite() { continue; }
            let x = i as f32 * 0.0037;
            let off = (0.0041 * x.sin()).abs() + 0.15;
            high[i] = v + off;
            low[i]  = v - off;
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
        Box::new(StochBatchState { cuda, high, low, close, sweep })
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
