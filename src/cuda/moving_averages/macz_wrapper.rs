#![cfg(feature = "cuda")]

//! CUDA wrapper for MAC-Z (ZVWAP + MACD/Stddev + optional Laguerre)
//!
//! Parity targets aligned with ALMA/CWMA wrappers:
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/macz_kernel.ptx")) with
//!   DetermineTargetFromContext + OptLevel O2 fallback.
//! - Stream NON_BLOCKING.
//! - Policy enums for kernel selection; Auto by default.
//! - VRAM checks using mem_get_info() with ~64MB headroom.
//! - Warmup/NaN semantics match scalar macz.rs exactly.
//!
//! Implementation notes:
//! - Host computes prefix sums for close, close^2, and when volume is present
//!   also volume and price*volume, as well as NaN-prefix counters. Kernels use
//!   these to evaluate per-time windows in O(1) with exact NaN parity.
//! - Batch kernel: one thread per row (parameter combo) scans the series and
//!   computes histogram; MAC-Z temporary values are produced internally for the
//!   signal SMA.
//! - Many-series one-param kernel: one thread per column (time-major layout).

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::macz::{MaczBatchRange, MaczParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaMaczPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaMaczPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

#[derive(Debug)]
pub enum CudaMaczError {
    Cuda(String),
    InvalidInput(String),
}
impl fmt::Display for CudaMaczError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaMaczError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaMaczError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaMaczError {}

pub struct CudaMacz {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaMaczPolicy,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaMacz {
    pub fn new(device_id: usize) -> Result<Self, CudaMaczError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaMaczError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaMaczError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaMaczError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/macz_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaMaczError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaMaczError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaMaczPolicy::default(),
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn set_policy(&mut self, policy: CudaMaczPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaMaczPolicy {
        &self.policy
    }
    pub fn synchronize(&self) -> Result<(), CudaMaczError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaMaczError::Cuda(e.to_string()))
    }

    #[inline]
    fn device_mem_ok(bytes: usize) -> bool {
        match mem_get_info() {
            Ok((free, _)) => bytes.saturating_add(64 * 1024 * 1024) <= free,
            Err(_) => true,
        }
    }

    fn expand_grid(sweep: &MaczBatchRange) -> Vec<MaczParams> {
        fn axis_usize((s, e, step): (usize, usize, usize)) -> Vec<usize> {
            if step == 0 || s == e {
                return vec![s];
            }
            (s..=e).step_by(step).collect()
        }
        fn axis_f64((s, e, step): (f64, f64, f64)) -> Vec<f64> {
            if step.abs() < 1e-12 || (s - e).abs() < 1e-12 {
                return vec![s];
            }
            let mut v = Vec::new();
            let mut x = s;
            while x <= e + 1e-12 {
                v.push(x);
                x += step;
            }
            v
        }
        let fs = axis_usize(sweep.fast_length);
        let ss = axis_usize(sweep.slow_length);
        let gs = axis_usize(sweep.signal_length);
        let zs = axis_usize(sweep.lengthz);
        let ds = axis_usize(sweep.length_stdev);
        let as_ = axis_f64(sweep.a);
        let bs = axis_f64(sweep.b);
        let mut out = Vec::with_capacity(
            fs.len() * ss.len() * gs.len() * zs.len() * ds.len() * as_.len() * bs.len(),
        );
        for &f in &fs {
            for &s in &ss {
                for &g in &gs {
                    for &z in &zs {
                        for &d in &ds {
                            for &a in &as_ {
                                for &b in &bs {
                                    out.push(MaczParams {
                                        fast_length: Some(f),
                                        slow_length: Some(s),
                                        signal_length: Some(g),
                                        lengthz: Some(z),
                                        length_stdev: Some(d),
                                        a: Some(a),
                                        b: Some(b),
                                        use_lag: Some(false),
                                        gamma: Some(0.02),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }

    // ---------- Prefix helpers (single series) ----------
    fn build_prefixes_single(
        prices: &[f32],
        volume: Option<&[f32]>,
    ) -> (
        Vec<f64>,
        Vec<f64>,
        Vec<i32>,
        Option<(Vec<f64>, Vec<f64>, Vec<i32>)>,
    ) {
        let len = prices.len();
        let mut pcs = vec![0.0f64; len + 1];
        let mut pcsq = vec![0.0f64; len + 1];
        let mut pnan = vec![0i32; len + 1];
        let mut acc_s = 0.0f64;
        let mut acc_sq = 0.0f64;
        let mut acc_nan = 0i32;
        for i in 0..len {
            let x = prices[i] as f64;
            if x.is_nan() {
                acc_nan += 1;
            } else {
                acc_s += x;
                acc_sq += x * x;
            }
            pcs[i + 1] = acc_s;
            pcsq[i + 1] = acc_sq;
            pnan[i + 1] = acc_nan;
        }
        if let Some(vol) = volume {
            let mut pvs = vec![0.0f64; len + 1];
            let mut pps = vec![0.0f64; len + 1];
            let mut pvn = vec![0i32; len + 1];
            let mut acc_vs = 0.0f64;
            let mut acc_pv = 0.0f64;
            let mut acc_vn = 0i32;
            for i in 0..len {
                let v = vol[i] as f64;
                let c = prices[i] as f64;
                if v.is_nan() || c.is_nan() {
                    acc_vn += 1;
                } else {
                    acc_vs += v;
                    acc_pv += v * c;
                }
                pvs[i + 1] = acc_vs;
                pps[i + 1] = acc_pv;
                pvn[i + 1] = acc_vn;
            }
            (pcs, pcsq, pnan, Some((pvs, pps, pvn)))
        } else {
            (pcs, pcsq, pnan, None)
        }
    }

    // ---------- Prefix helpers (time-major) ----------
    fn build_prefixes_time_major(
        close_tm: &[f32],
        volume_tm: Option<&[f32]>,
        cols: usize,
        rows: usize,
    ) -> (
        Vec<f64>,
        Vec<f64>,
        Vec<i32>,
        Option<(Vec<f64>, Vec<f64>, Vec<i32>)>,
    ) {
        let mut pcs = vec![0.0f64; (rows + 1) * cols];
        let mut pcsq = vec![0.0f64; (rows + 1) * cols];
        let mut pcn = vec![0i32; (rows + 1) * cols];
        for s in 0..cols {
            let mut acc_s = 0.0f64;
            let mut acc_sq = 0.0f64;
            let mut acc_n = 0i32;
            for t in 0..rows {
                let idx = t * cols + s;
                let x = close_tm[idx] as f64;
                if x.is_nan() {
                    acc_n += 1;
                } else {
                    acc_s += x;
                    acc_sq += x * x;
                }
                let off = s * (rows + 1) + (t + 1);
                pcs[off] = acc_s;
                pcsq[off] = acc_sq;
                pcn[off] = acc_n;
            }
        }
        if let Some(vtm) = volume_tm {
            let mut pvs = vec![0.0f64; (rows + 1) * cols];
            let mut pps = vec![0.0f64; (rows + 1) * cols];
            let mut pvn = vec![0i32; (rows + 1) * cols];
            for s in 0..cols {
                let mut acc_vs = 0.0f64;
                let mut acc_pv = 0.0f64;
                let mut acc_vn = 0i32;
                for t in 0..rows {
                    let idx = t * cols + s;
                    let c = close_tm[idx] as f64;
                    let v = vtm[idx] as f64;
                    if c.is_nan() || v.is_nan() {
                        acc_vn += 1;
                    } else {
                        acc_vs += v;
                        acc_pv += v * c;
                    }
                    let off = s * (rows + 1) + (t + 1);
                    pvs[off] = acc_vs;
                    pps[off] = acc_pv;
                    pvn[off] = acc_vn;
                }
            }
            (pcs, pcsq, pcn, Some((pvs, pps, pvn)))
        } else {
            (pcs, pcsq, pcn, None)
        }
    }

    fn validate_first_valid(prices: &[f32]) -> Result<usize, CudaMaczError> {
        if prices.is_empty() {
            return Err(CudaMaczError::InvalidInput("empty input".into()));
        }
        prices
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaMaczError::InvalidInput("all values are NaN".into()))
    }

    // ---------- Batch: one series × many params ----------
    pub fn macz_batch_dev(
        &self,
        prices: &[f32],
        volume: Option<&[f32]>,
        sweep: &MaczBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<MaczParams>), CudaMaczError> {
        let combos = Self::expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaMaczError::InvalidInput("empty param grid".into()));
        }
        let len = prices.len();
        if let Some(v) = volume {
            if v.len() != len {
                return Err(CudaMaczError::InvalidInput(
                    "price/volume length mismatch".into(),
                ));
            }
        }
        let first_valid = Self::validate_first_valid(prices)?;
        // Validate warmups
        let mut max_need = 0usize;
        for p in &combos {
            let slow = p.slow_length.unwrap_or(25);
            let lz = p.lengthz.unwrap_or(20);
            let lsd = p.length_stdev.unwrap_or(25);
            let sig = p.signal_length.unwrap_or(9);
            let warm_hist = first_valid + slow.max(lz).max(lsd) + sig - 1;
            if warm_hist > max_need {
                max_need = warm_hist;
            }
        }
        if len <= max_need {
            return Err(CudaMaczError::InvalidInput("not enough valid data".into()));
        }

        // VRAM estimate: inputs + prefixes + params + outputs (two arrays)
        let rows = combos.len();
        let prefix_base = (len + 1) * (std::mem::size_of::<f64>() * 2 + std::mem::size_of::<i32>());
        let prefix_vol = if volume.is_some() {
            (len + 1) * (std::mem::size_of::<f64>() * 2 + std::mem::size_of::<i32>())
        } else {
            0
        };
        let req = prices.len() * std::mem::size_of::<f32>()
            + prefix_base
            + prefix_vol
            + rows
                * (5 * std::mem::size_of::<i32>()
                    + 3 * std::mem::size_of::<f32>()
                    + std::mem::size_of::<i32>())
            + 2 * rows * len * std::mem::size_of::<f32>();
        if !Self::device_mem_ok(req) {
            return Err(CudaMaczError::InvalidInput(
                "insufficient device memory".into(),
            ));
        }

        // Upload inputs/prefixes
        let d_close = unsafe { DeviceBuffer::from_slice_async(prices, &self.stream) }
            .map_err(|e| CudaMaczError::Cuda(e.to_string()))?;
        let (pcs, pcsq, pcn, vol_tuple) = Self::build_prefixes_single(prices, volume);
        let d_pcs =
            DeviceBuffer::from_slice(&pcs).map_err(|e| CudaMaczError::Cuda(e.to_string()))?;
        let d_pcsq =
            DeviceBuffer::from_slice(&pcsq).map_err(|e| CudaMaczError::Cuda(e.to_string()))?;
        let d_pcn =
            DeviceBuffer::from_slice(&pcn).map_err(|e| CudaMaczError::Cuda(e.to_string()))?;
        let (d_pvs, d_pps, d_pvn) = if let Some((pvs, pps, pvn)) = vol_tuple {
            (
                Some(
                    DeviceBuffer::from_slice(&pvs)
                        .map_err(|e| CudaMaczError::Cuda(e.to_string()))?,
                ),
                Some(
                    DeviceBuffer::from_slice(&pps)
                        .map_err(|e| CudaMaczError::Cuda(e.to_string()))?,
                ),
                Some(
                    DeviceBuffer::from_slice(&pvn)
                        .map_err(|e| CudaMaczError::Cuda(e.to_string()))?,
                ),
            )
        } else {
            (None, None, None)
        };

        // Pack params
        let fasts: Vec<i32> = combos
            .iter()
            .map(|p| p.fast_length.unwrap_or(12) as i32)
            .collect();
        let slows: Vec<i32> = combos
            .iter()
            .map(|p| p.slow_length.unwrap_or(25) as i32)
            .collect();
        let sigs: Vec<i32> = combos
            .iter()
            .map(|p| p.signal_length.unwrap_or(9) as i32)
            .collect();
        let lzs: Vec<i32> = combos
            .iter()
            .map(|p| p.lengthz.unwrap_or(20) as i32)
            .collect();
        let lsds: Vec<i32> = combos
            .iter()
            .map(|p| p.length_stdev.unwrap_or(25) as i32)
            .collect();
        let a_s: Vec<f32> = combos.iter().map(|p| p.a.unwrap_or(1.0) as f32).collect();
        let b_s: Vec<f32> = combos.iter().map(|p| p.b.unwrap_or(1.0) as f32).collect();
        // Enable Laguerre only if requested explicitly in params (defaults false)
        let use_lag: Vec<i32> = combos
            .iter()
            .map(|p| if p.use_lag.unwrap_or(false) { 1 } else { 0 })
            .collect();
        let gammas: Vec<f32> = combos
            .iter()
            .map(|p| p.gamma.unwrap_or(0.02) as f32)
            .collect();

        let d_fasts =
            DeviceBuffer::from_slice(&fasts).map_err(|e| CudaMaczError::Cuda(e.to_string()))?;
        let d_slows =
            DeviceBuffer::from_slice(&slows).map_err(|e| CudaMaczError::Cuda(e.to_string()))?;
        let d_sigs =
            DeviceBuffer::from_slice(&sigs).map_err(|e| CudaMaczError::Cuda(e.to_string()))?;
        let d_lzs =
            DeviceBuffer::from_slice(&lzs).map_err(|e| CudaMaczError::Cuda(e.to_string()))?;
        let d_lsds =
            DeviceBuffer::from_slice(&lsds).map_err(|e| CudaMaczError::Cuda(e.to_string()))?;
        let d_as =
            DeviceBuffer::from_slice(&a_s).map_err(|e| CudaMaczError::Cuda(e.to_string()))?;
        let d_bs =
            DeviceBuffer::from_slice(&b_s).map_err(|e| CudaMaczError::Cuda(e.to_string()))?;
        let d_ul =
            DeviceBuffer::from_slice(&use_lag).map_err(|e| CudaMaczError::Cuda(e.to_string()))?;
        let d_gam =
            DeviceBuffer::from_slice(&gammas).map_err(|e| CudaMaczError::Cuda(e.to_string()))?;

        // Outputs
        let mut d_macz: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(rows * len)
                .map_err(|e| CudaMaczError::Cuda(e.to_string()))?
        };
        let mut d_hist: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(rows * len)
                .map_err(|e| CudaMaczError::Cuda(e.to_string()))?
        };

        // Launch
        let func = self
            .module
            .get_function("macz_batch_f32")
            .map_err(|e| CudaMaczError::Cuda(e.to_string()))?;
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Auto => 256,
            BatchKernelPolicy::Plain { block_x } => block_x.max(32),
        };
        if cfg!(debug_assertions) || std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if !self.debug_batch_logged {
                eprintln!(
                    "[macz] batch kernel: block_x={} rows={} len={} vwap_fallback_sma={}",
                    block_x,
                    rows,
                    len,
                    if volume.is_some() { 0 } else { 1 }
                );
                unsafe {
                    (*(self as *const _ as *mut CudaMacz)).debug_batch_logged = true;
                }
            }
        }
        unsafe {
            let grid: GridSize = (((rows as u32 + block_x - 1) / block_x).max(1), 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            let mut close_p = d_close.as_device_ptr().as_raw();
            let mut vol_p = if let Some(ref b) = volume {
                close_p /* placeholder */
            } else {
                0u64
            };
            if volume.is_some() {
                // Upload volume separately (async) and get pointer
                // Note: copy above not done; do it now if provided
            }
            // Actually upload volume now (if any)
            let d_volume = if let Some(vol) = volume {
                Some(
                    DeviceBuffer::from_slice(vol)
                        .map_err(|e| CudaMaczError::Cuda(e.to_string()))?,
                )
            } else {
                None
            };
            if let Some(ref dv) = d_volume {
                vol_p = dv.as_device_ptr().as_raw();
            }

            let mut pcs_p = d_pcs.as_device_ptr().as_raw();
            let mut pcsq_p = d_pcsq.as_device_ptr().as_raw();
            let mut pcn_p = d_pcn.as_device_ptr().as_raw();
            let mut pvs_p = d_pvs
                .as_ref()
                .map(|b| b.as_device_ptr().as_raw())
                .unwrap_or(0);
            let mut pps_p = d_pps
                .as_ref()
                .map(|b| b.as_device_ptr().as_raw())
                .unwrap_or(0);
            let mut pvn_p = d_pvn
                .as_ref()
                .map(|b| b.as_device_ptr().as_raw())
                .unwrap_or(0);
            let mut f_p = d_fasts.as_device_ptr().as_raw();
            let mut s_p = d_slows.as_device_ptr().as_raw();
            let mut g_p = d_sigs.as_device_ptr().as_raw();
            let mut lz_p = d_lzs.as_device_ptr().as_raw();
            let mut lsd_p = d_lsds.as_device_ptr().as_raw();
            let mut a_p = d_as.as_device_ptr().as_raw();
            let mut b_p = d_bs.as_device_ptr().as_raw();
            let mut ul_p = d_ul.as_device_ptr().as_raw();
            let mut ga_p = d_gam.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut fv_i = first_valid as i32;
            let mut rows_i = rows as i32;
            let mut use_sma = if volume.is_some() { 0i32 } else { 1i32 };
            let mut macz_p = d_macz.as_device_ptr().as_raw();
            let mut hist_p = d_hist.as_device_ptr().as_raw();
            // exact arg list for macz_batch_f32 (23 params)
            let mut args: [*mut c_void; 23] = [
                &mut close_p as *mut _ as *mut c_void,
                &mut vol_p as *mut _ as *mut c_void,
                &mut pcs_p as *mut _ as *mut c_void,
                &mut pcsq_p as *mut _ as *mut c_void,
                &mut pcn_p as *mut _ as *mut c_void,
                &mut pvs_p as *mut _ as *mut c_void,
                &mut pps_p as *mut _ as *mut c_void,
                &mut pvn_p as *mut _ as *mut c_void,
                &mut f_p as *mut _ as *mut c_void,
                &mut s_p as *mut _ as *mut c_void,
                &mut g_p as *mut _ as *mut c_void,
                &mut lz_p as *mut _ as *mut c_void,
                &mut lsd_p as *mut _ as *mut c_void,
                &mut a_p as *mut _ as *mut c_void,
                &mut b_p as *mut _ as *mut c_void,
                &mut ul_p as *mut _ as *mut c_void,
                &mut ga_p as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut fv_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut use_sma as *mut _ as *mut c_void,
                &mut macz_p as *mut _ as *mut c_void,
                &mut hist_p as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(|e| CudaMaczError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaMaczError::Cuda(e.to_string()))?;

        Ok((
            DeviceArrayF32 {
                buf: d_hist,
                rows,
                cols: len,
            },
            combos,
        ))
    }

    // ---------- Many-series × one-param (time-major) ----------
    pub fn macz_many_series_one_param_time_major_dev(
        &self,
        close_tm: &[f32],
        volume_tm: Option<&[f32]>,
        cols: usize,
        rows: usize,
        params: &MaczParams,
    ) -> Result<DeviceArrayF32, CudaMaczError> {
        if cols == 0 || rows == 0 {
            return Err(CudaMaczError::InvalidInput("empty matrix".into()));
        }
        if close_tm.len() != cols * rows {
            return Err(CudaMaczError::InvalidInput("matrix shape mismatch".into()));
        }
        if let Some(vt) = volume_tm {
            if vt.len() != cols * rows {
                return Err(CudaMaczError::InvalidInput("volume shape mismatch".into()));
            }
        }

        // Per-series first_valid (based on close only, per scalar semantics)
        let mut first_valids = vec![rows as i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                if !close_tm[idx].is_nan() {
                    first_valids[s] = t as i32;
                    break;
                }
            }
        }
        // Warmup validation
        let f = params.fast_length.unwrap_or(12);
        let sl = params.slow_length.unwrap_or(25);
        let sg = params.signal_length.unwrap_or(9);
        let lz = params.lengthz.unwrap_or(20);
        let lsd = params.length_stdev.unwrap_or(25);
        for &fv in &first_valids {
            if (fv as usize) + sl.max(lz).max(lsd) + sg - 1 >= rows {
                return Err(CudaMaczError::InvalidInput(
                    "not enough valid data for at least one series".into(),
                ));
            }
        }

        // VRAM estimate
        let prefix_base =
            (rows + 1) * cols * (std::mem::size_of::<f64>() * 2 + std::mem::size_of::<i32>());
        let prefix_vol = if volume_tm.is_some() {
            (rows + 1) * cols * (std::mem::size_of::<f64>() * 2 + std::mem::size_of::<i32>())
        } else {
            0
        };
        let req = (close_tm.len() + volume_tm.as_ref().map(|v| v.len()).unwrap_or(0))
            * std::mem::size_of::<f32>()
            + prefix_base
            + prefix_vol
            + 2 * cols * rows * std::mem::size_of::<f32>()
            + cols * std::mem::size_of::<i32>();
        if !Self::device_mem_ok(req) {
            return Err(CudaMaczError::InvalidInput(
                "insufficient device memory".into(),
            ));
        }

        // Upload inputs and prefixes
        let d_close_tm =
            DeviceBuffer::from_slice(close_tm).map_err(|e| CudaMaczError::Cuda(e.to_string()))?;
        let d_volume_tm = if let Some(v) = volume_tm {
            Some(DeviceBuffer::from_slice(v).map_err(|e| CudaMaczError::Cuda(e.to_string()))?)
        } else {
            None
        };
        let (pcs, pcsq, pcn, vol_tuple) =
            Self::build_prefixes_time_major(close_tm, volume_tm, cols, rows);
        let d_pcs =
            DeviceBuffer::from_slice(&pcs).map_err(|e| CudaMaczError::Cuda(e.to_string()))?;
        let d_pcsq =
            DeviceBuffer::from_slice(&pcsq).map_err(|e| CudaMaczError::Cuda(e.to_string()))?;
        let d_pcn =
            DeviceBuffer::from_slice(&pcn).map_err(|e| CudaMaczError::Cuda(e.to_string()))?;
        let (d_pvs, d_pps, d_pvn) = if let Some((pvs, pps, pvn)) = vol_tuple {
            (
                Some(
                    DeviceBuffer::from_slice(&pvs)
                        .map_err(|e| CudaMaczError::Cuda(e.to_string()))?,
                ),
                Some(
                    DeviceBuffer::from_slice(&pps)
                        .map_err(|e| CudaMaczError::Cuda(e.to_string()))?,
                ),
                Some(
                    DeviceBuffer::from_slice(&pvn)
                        .map_err(|e| CudaMaczError::Cuda(e.to_string()))?,
                ),
            )
        } else {
            (None, None, None)
        };

        let mut d_macz_tm: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(cols * rows)
                .map_err(|e| CudaMaczError::Cuda(e.to_string()))?
        };
        let mut d_hist_tm: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(cols * rows)
                .map_err(|e| CudaMaczError::Cuda(e.to_string()))?
        };

        // Launch
        let func = self
            .module
            .get_function("macz_many_series_one_param_time_major_f32")
            .map_err(|e| CudaMaczError::Cuda(e.to_string()))?;
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 256,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32),
        };
        if cfg!(debug_assertions) || std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if !self.debug_many_logged {
                eprintln!(
                    "[macz] many-series kernel: block_x={} cols={} rows={} vwap_fallback_sma={}",
                    block_x,
                    cols,
                    rows,
                    if volume_tm.is_some() { 0 } else { 1 }
                );
                unsafe {
                    (*(self as *const _ as *mut CudaMacz)).debug_many_logged = true;
                }
            }
        }
        // Allocate first_valids on device outside the launch scope to keep it alive
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaMaczError::Cuda(e.to_string()))?;
        unsafe {
            let grid: GridSize = (((cols as u32 + block_x - 1) / block_x).max(1), 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            let mut c_p = d_close_tm.as_device_ptr().as_raw();
            let mut v_p = d_volume_tm
                .as_ref()
                .map(|b| b.as_device_ptr().as_raw())
                .unwrap_or(0);
            let mut pcs_p = d_pcs.as_device_ptr().as_raw();
            let mut pcsq_p = d_pcsq.as_device_ptr().as_raw();
            let mut pcn_p = d_pcn.as_device_ptr().as_raw();
            let mut pvs_p = d_pvs
                .as_ref()
                .map(|b| b.as_device_ptr().as_raw())
                .unwrap_or(0);
            let mut pps_p = d_pps
                .as_ref()
                .map(|b| b.as_device_ptr().as_raw())
                .unwrap_or(0);
            let mut pvn_p = d_pvn
                .as_ref()
                .map(|b| b.as_device_ptr().as_raw())
                .unwrap_or(0);
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut f_i = f as i32;
            let mut s_i = sl as i32;
            let mut g_i = sg as i32;
            let mut lz_i = lz as i32;
            let mut lsd_i = lsd as i32;
            let mut a_f = params.a.unwrap_or(1.0) as f32;
            let mut b_f = params.b.unwrap_or(1.0) as f32;
            let mut ul_i = if params.use_lag.unwrap_or(false) {
                1i32
            } else {
                0i32
            };
            let mut gam_f = params.gamma.unwrap_or(0.02) as f32;
            let mut fv_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut use_sma = if volume_tm.is_some() { 0i32 } else { 1i32 };
            let mut macz_p = d_macz_tm.as_device_ptr().as_raw();
            let mut hist_p = d_hist_tm.as_device_ptr().as_raw();
            // exact arg list for macz_many_series_one_param_time_major_f32 (23 params)
            let mut args: [*mut c_void; 23] = [
                &mut c_p as *mut _ as *mut c_void,
                &mut v_p as *mut _ as *mut c_void,
                &mut pcs_p as *mut _ as *mut c_void,
                &mut pcsq_p as *mut _ as *mut c_void,
                &mut pcn_p as *mut _ as *mut c_void,
                &mut pvs_p as *mut _ as *mut c_void,
                &mut pps_p as *mut _ as *mut c_void,
                &mut pvn_p as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut f_i as *mut _ as *mut c_void,
                &mut s_i as *mut _ as *mut c_void,
                &mut g_i as *mut _ as *mut c_void,
                &mut lz_i as *mut _ as *mut c_void,
                &mut lsd_i as *mut _ as *mut c_void,
                &mut a_f as *mut _ as *mut c_void,
                &mut b_f as *mut _ as *mut c_void,
                &mut ul_i as *mut _ as *mut c_void,
                &mut gam_f as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut use_sma as *mut _ as *mut c_void,
                &mut macz_p as *mut _ as *mut c_void,
                &mut hist_p as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(|e| CudaMaczError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaMaczError::Cuda(e.to_string()))?;
        Ok(DeviceArrayF32 {
            buf: d_hist_tm,
            rows,
            cols,
        })
    }
}

// ---------- Benches ----------
#[cfg(any(test, feature = "cuda"))]
pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    struct MaczBatchBenchState {
        cuda: CudaMacz,
        price: Vec<f32>,
        volume: Vec<f32>,
        sweep: MaczBatchRange,
    }
    impl CudaBenchState for MaczBatchBenchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .macz_batch_dev(&self.price, Some(&self.volume), &self.sweep);
        }
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        let mut v = Vec::new();
        // One-series × many-params (moderate grid)
        v.push(CudaBenchScenario::new(
            "macz",
            "one_series_many_params",
            "macz/one_series_many_params",
            "macz_hist",
            || {
                let cuda = CudaMacz::new(0).expect("cuda macz");
                let len = 100_000usize;
                let mut price = vec![f32::NAN; len];
                let mut volume = vec![f32::NAN; len];
                for i in 50..len {
                    let x = i as f32;
                    price[i] = (x * 0.001).sin() + 0.0002 * x;
                    volume[i] = (x * 0.0007).cos().abs() + 0.5;
                }
                let sweep = MaczBatchRange {
                    fast_length: (10, 10, 0),
                    slow_length: (26, 26, 0),
                    signal_length: (9, 9, 0),
                    lengthz: (20, 20, 0),
                    length_stdev: (25, 25, 0),
                    a: (1.0, 1.0, 0.0),
                    b: (1.0, 1.0, 0.0),
                };
                Box::new(MaczBatchBenchState {
                    cuda,
                    price,
                    volume,
                    sweep,
                }) as Box<dyn CudaBenchState>
            },
        ));
        v
    }
}
