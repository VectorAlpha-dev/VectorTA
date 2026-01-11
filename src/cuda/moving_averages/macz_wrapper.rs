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
//! - Batch kernel: two-pass tiled implementation (use_lag==false).
//!   - Pass 1 computes macz_tmp in parallel across time tiles × rows.
//!   - Pass 2 computes histogram from macz_tmp, preserving NaN parity.
//! - Many-series one-param kernel: one thread per column (time-major layout).

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::macz::{MaczBatchRange, MaczParams};
use cust::context::Context;
use cust::device::Device;
use cust::error::CudaError;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::sys as cu;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::sync::Arc;
use thiserror::Error;

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

#[derive(Debug, Error)]
pub enum CudaMaczError {
    #[error(transparent)]
    Cuda(#[from] CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

pub struct CudaMacz {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaMaczPolicy,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaMacz {
    pub fn new(device_id: usize) -> Result<Self, CudaMaczError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/macz_kernel.ptx"));
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

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
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
        self.stream.synchronize()?;
        Ok(())
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> {
        Arc::clone(&self.context)
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }

    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaMaczError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            if required_bytes.saturating_add(headroom_bytes) <= free {
                Ok(())
            } else {
                Err(CudaMaczError::OutOfMemory {
                    required: required_bytes,
                    free,
                    headroom: headroom_bytes,
                })
            }
        } else {
            Ok(())
        }
    }

    #[inline]
    fn validate_launch(&self, grid: GridSize, block: BlockSize) -> Result<(), CudaMaczError> {
        use cust::device::DeviceAttribute as DevAttr;
        let dev = Device::get_device(self.device_id)?;
        let max_threads = dev
            .get_attribute(DevAttr::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let bx = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if bx > max_threads {
            return Err(CudaMaczError::LaunchConfigTooLarge {
                gx: grid.x,
                gy: grid.y,
                gz: grid.z,
                bx: block.x,
                by: block.y,
                bz: block.z,
            });
        }
        let max_gx = dev
            .get_attribute(DevAttr::MaxGridDimX)
            .unwrap_or(2_147_483_647) as u32;
        let max_gy = dev
            .get_attribute(DevAttr::MaxGridDimY)
            .unwrap_or(65_535) as u32;
        let max_gz = dev
            .get_attribute(DevAttr::MaxGridDimZ)
            .unwrap_or(65_535) as u32;
        if grid.x > max_gx || grid.y > max_gy || grid.z > max_gz {
            return Err(CudaMaczError::LaunchConfigTooLarge {
                gx: grid.x,
                gy: grid.y,
                gz: grid.z,
                bx: block.x,
                by: block.y,
                bz: block.z,
            });
        }
        Ok(())
    }

    fn expand_grid(sweep: &MaczBatchRange) -> Result<Vec<MaczParams>, CudaMaczError> {
        fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, CudaMaczError> {
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
                return Err(CudaMaczError::InvalidInput(format!(
                    "Invalid range: start={}, end={}, step={}",
                    start, end, step
                )));
            }
            Ok(v)
        }
        fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, CudaMaczError> {
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
                    return Err(CudaMaczError::InvalidInput(format!(
                        "Invalid range: start={}, end={}, step={}",
                        start, end, step
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
                return Err(CudaMaczError::InvalidInput(format!(
                    "Invalid range: start={}, end={}, step={}",
                    start, end, step
                )));
            }
            Ok(v)
        }
        let fs = axis_usize(sweep.fast_length)?;
        let ss = axis_usize(sweep.slow_length)?;
        let gs = axis_usize(sweep.signal_length)?;
        let zs = axis_usize(sweep.lengthz)?;
        let ds = axis_usize(sweep.length_stdev)?;
        let as_ = axis_f64(sweep.a)?;
        let bs = axis_f64(sweep.b)?;
        let cap = fs
            .len()
            .checked_mul(ss.len())
            .and_then(|v| v.checked_mul(gs.len()))
            .and_then(|v| v.checked_mul(zs.len()))
            .and_then(|v| v.checked_mul(ds.len()))
            .and_then(|v| v.checked_mul(as_.len()))
            .and_then(|v| v.checked_mul(bs.len()))
            .ok_or_else(|| {
                CudaMaczError::InvalidInput("range size overflow in MACZ CUDA expand_grid".into())
            })?;
        let mut out = Vec::with_capacity(cap);
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
        Ok(out)
    }

    
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

    
    pub fn macz_batch_dev(
        &self,
        prices: &[f32],
        volume: Option<&[f32]>,
        sweep: &MaczBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<MaczParams>), CudaMaczError> {
        let combos = Self::expand_grid(sweep)?;
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

        
        let rows = combos.len();
        let sz_f32 = std::mem::size_of::<f32>();
        let sz_f64 = std::mem::size_of::<f64>();
        let sz_i32 = std::mem::size_of::<i32>();
        let prefix_slot = sz_f64
            .checked_mul(2)
            .and_then(|v| v.checked_add(sz_i32))
            .ok_or_else(|| CudaMaczError::InvalidInput("byte size overflow".into()))?;
        let prefix_base = (len + 1)
            .checked_mul(prefix_slot)
            .ok_or_else(|| CudaMaczError::InvalidInput("byte size overflow".into()))?;
        let prefix_vol = if volume.is_some() {
            (len + 1)
                .checked_mul(prefix_slot)
                .ok_or_else(|| CudaMaczError::InvalidInput("byte size overflow".into()))?
        } else {
            0
        };
        let prices_b = prices
            .len()
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaMaczError::InvalidInput("byte size overflow".into()))?;
        let params_slot = 5usize
            .checked_mul(sz_i32)
            .and_then(|v| v.checked_add(2usize.checked_mul(sz_f32)?))
            .ok_or_else(|| CudaMaczError::InvalidInput("byte size overflow".into()))?;
        let params_b = rows
            .checked_mul(params_slot)
            .ok_or_else(|| CudaMaczError::InvalidInput("byte size overflow".into()))?;
        let out_elems = rows
            .checked_mul(len)
            .ok_or_else(|| CudaMaczError::InvalidInput("rows*cols overflow".into()))?;
        let out_b = out_elems
            .checked_mul(sz_f32)
            .and_then(|v| v.checked_mul(2)) 
            .ok_or_else(|| CudaMaczError::InvalidInput("byte size overflow".into()))?;
        let req = prices_b
            .checked_add(prefix_base)
            .and_then(|v| v.checked_add(prefix_vol))
            .and_then(|v| v.checked_add(params_b))
            .and_then(|v| v.checked_add(out_b))
            .ok_or_else(|| CudaMaczError::InvalidInput("byte size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        Self::will_fit(req, headroom)?;

        
        let d_close = unsafe { DeviceBuffer::from_slice_async(prices, &self.stream) }?;
        let (pcs, pcsq, pcn, vol_tuple) = Self::build_prefixes_single(prices, volume);
        let d_pcs = DeviceBuffer::from_slice(&pcs)?;
        let d_pcsq = DeviceBuffer::from_slice(&pcsq)?;
        let d_pcn = DeviceBuffer::from_slice(&pcn)?;
        let (d_pvs, d_pps, d_pvn) = if let Some((pvs, pps, pvn)) = vol_tuple {
            (
                Some(DeviceBuffer::from_slice(&pvs)?),
                Some(DeviceBuffer::from_slice(&pps)?),
                Some(DeviceBuffer::from_slice(&pvn)?),
            )
        } else {
            (None, None, None)
        };

        
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

        let d_fasts = DeviceBuffer::from_slice(&fasts)?;
        let d_slows = DeviceBuffer::from_slice(&slows)?;
        let d_sigs = DeviceBuffer::from_slice(&sigs)?;
        let d_lzs = DeviceBuffer::from_slice(&lzs)?;
        let d_lsds = DeviceBuffer::from_slice(&lsds)?;
        let d_as = DeviceBuffer::from_slice(&a_s)?;
        let d_bs = DeviceBuffer::from_slice(&b_s)?;

        
        let elems = rows
            .checked_mul(len)
            .ok_or_else(|| CudaMaczError::InvalidInput("rows*cols overflow".into()))?;
        let mut d_macz: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems)? };
        let mut d_hist: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems)? };

        let d_volume = if let Some(vol) = volume {
            Some(unsafe { DeviceBuffer::from_slice_async(vol, &self.stream) }?)
        } else {
            None
        };

        
        let func_macz = self
            .module
            .get_function("macz_batch_macz_tmp_f32")
            .map_err(|_| CudaMaczError::MissingKernelSymbol { name: "macz_batch_macz_tmp_f32" })?;
        let func_hist = self
            .module
            .get_function("macz_batch_hist_from_macz_f32")
            .map_err(|_| CudaMaczError::MissingKernelSymbol { name: "macz_batch_hist_from_macz_f32" })?;
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Auto => 256,
            BatchKernelPolicy::Plain { block_x } => block_x.max(32),
        };
        if cfg!(debug_assertions) || std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if !self.debug_batch_logged {
                eprintln!(
                    "[macz] batch kernels: block_x={} rows={} len={} vwap_fallback_sma={}",
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
            let grid: GridSize =
                (((len as u32 + block_x - 1) / block_x).max(1), rows as u32, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            self.validate_launch(grid, block)?;
            let mut close_p = d_close.as_device_ptr().as_raw();
            let mut vol_p = d_volume
                .as_ref()
                .map(|dv| dv.as_device_ptr().as_raw())
                .unwrap_or(0u64);

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
            let mut len_i = len as i32;
            let mut fv_i = first_valid as i32;
            let mut rows_i = rows as i32;
            let mut use_sma = if volume.is_some() { 0i32 } else { 1i32 };
            let mut macz_p = d_macz.as_device_ptr().as_raw();
            let mut hist_p = d_hist.as_device_ptr().as_raw();
            
            let mut args_macz: [*mut c_void; 19] = [
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
                &mut lz_p as *mut _ as *mut c_void,
                &mut lsd_p as *mut _ as *mut c_void,
                &mut a_p as *mut _ as *mut c_void,
                &mut b_p as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut fv_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut use_sma as *mut _ as *mut c_void,
                &mut macz_p as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func_macz, grid, block, 0, &mut args_macz)?;

            
            let mut args_hist: [*mut c_void; 9] = [
                &mut macz_p as *mut _ as *mut c_void,
                &mut s_p as *mut _ as *mut c_void,
                &mut g_p as *mut _ as *mut c_void,
                &mut lz_p as *mut _ as *mut c_void,
                &mut lsd_p as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut fv_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut hist_p as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func_hist, grid, block, 0, &mut args_hist)?;
        }

        self.stream.synchronize()?;

        Ok((
            DeviceArrayF32 {
                buf: d_hist,
                rows,
                cols: len,
            },
            combos,
        ))
    }

    pub fn macz_batch_device(
        &self,
        d_close: &DeviceBuffer<f32>,
        d_volume: Option<&DeviceBuffer<f32>>,
        d_pcs: &DeviceBuffer<f64>,
        d_pcsq: &DeviceBuffer<f64>,
        d_pcn: &DeviceBuffer<i32>,
        d_pvs: Option<&DeviceBuffer<f64>>,
        d_pps: Option<&DeviceBuffer<f64>>,
        d_pvn: Option<&DeviceBuffer<i32>>,
        d_fasts: &DeviceBuffer<i32>,
        d_slows: &DeviceBuffer<i32>,
        d_sigs: &DeviceBuffer<i32>,
        d_lzs: &DeviceBuffer<i32>,
        d_lsds: &DeviceBuffer<i32>,
        d_as: &DeviceBuffer<f32>,
        d_bs: &DeviceBuffer<f32>,
        series_len: usize,
        n_rows: usize,
        first_valid: usize,
        d_macz_tmp: &mut DeviceBuffer<f32>,
        d_out_hist: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMaczError> {
        
        let cur_dev = unsafe {
            let mut dev: i32 = 0;
            let _ = cu::cuCtxGetDevice(&mut dev as *mut _);
            dev as u32
        };
        if cur_dev != self.device_id {
            return Err(CudaMaczError::DeviceMismatch {
                buf: self.device_id,
                current: cur_dev,
            });
        }

        if series_len == 0 || n_rows == 0 {
            return Err(CudaMaczError::InvalidInput(
                "series_len and n_rows must be positive".into(),
            ));
        }
        if series_len > i32::MAX as usize
            || n_rows > i32::MAX as usize
            || first_valid > i32::MAX as usize
        {
            return Err(CudaMaczError::InvalidInput(
                "arguments exceed kernel limits".into(),
            ));
        }

        let func_macz = self
            .module
            .get_function("macz_batch_macz_tmp_f32")
            .map_err(|_| CudaMaczError::MissingKernelSymbol {
                name: "macz_batch_macz_tmp_f32",
            })?;
        let func_hist = self
            .module
            .get_function("macz_batch_hist_from_macz_f32")
            .map_err(|_| CudaMaczError::MissingKernelSymbol {
                name: "macz_batch_hist_from_macz_f32",
            })?;

        let block_x = match self.policy.batch {
            BatchKernelPolicy::Auto => 256,
            BatchKernelPolicy::Plain { block_x } => block_x.max(32),
        };

        unsafe {
            let grid: GridSize = (
                ((series_len as u32 + block_x - 1) / block_x).max(1),
                n_rows as u32,
                1,
            )
                .into();
            let block: BlockSize = (block_x, 1, 1).into();
            self.validate_launch(grid, block)?;

            let mut close_p = d_close.as_device_ptr().as_raw();
            let mut vol_p = d_volume
                .as_ref()
                .map(|dv| dv.as_device_ptr().as_raw())
                .unwrap_or(0u64);

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

            let mut len_i = series_len as i32;
            let mut fv_i = first_valid as i32;
            let mut rows_i = n_rows as i32;
            let mut use_sma = if d_volume.is_some() { 0i32 } else { 1i32 };

            let mut macz_p = d_macz_tmp.as_device_ptr().as_raw();
            let mut hist_p = d_out_hist.as_device_ptr().as_raw();

            
            let mut args_macz: [*mut c_void; 19] = [
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
                &mut lz_p as *mut _ as *mut c_void,
                &mut lsd_p as *mut _ as *mut c_void,
                &mut a_p as *mut _ as *mut c_void,
                &mut b_p as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut fv_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut use_sma as *mut _ as *mut c_void,
                &mut macz_p as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func_macz, grid, block, 0, &mut args_macz)?;

            
            let mut args_hist: [*mut c_void; 9] = [
                &mut macz_p as *mut _ as *mut c_void,
                &mut s_p as *mut _ as *mut c_void,
                &mut g_p as *mut _ as *mut c_void,
                &mut lz_p as *mut _ as *mut c_void,
                &mut lsd_p as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut fv_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut hist_p as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func_hist, grid, block, 0, &mut args_hist)?;
        }

        Ok(())
    }

    
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

        
        let sz_f32 = std::mem::size_of::<f32>();
        let sz_f64 = std::mem::size_of::<f64>();
        let sz_i32 = std::mem::size_of::<i32>();
        let prefix_slot = sz_f64
            .checked_mul(2)
            .and_then(|v| v.checked_add(sz_i32))
            .ok_or_else(|| CudaMaczError::InvalidInput("byte size overflow".into()))?;
        let prefix_base = (rows + 1)
            .checked_mul(cols)
            .and_then(|v| v.checked_mul(prefix_slot))
            .ok_or_else(|| CudaMaczError::InvalidInput("byte size overflow".into()))?;
        let prefix_vol = if volume_tm.is_some() {
            (rows + 1)
                .checked_mul(cols)
                .and_then(|v| v.checked_mul(prefix_slot))
                .ok_or_else(|| CudaMaczError::InvalidInput("byte size overflow".into()))?
        } else {
            0
        };
        let mat_elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaMaczError::InvalidInput("cols*rows overflow".into()))?;
        let data_b = (close_tm.len() + volume_tm.as_ref().map(|v| v.len()).unwrap_or(0))
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaMaczError::InvalidInput("byte size overflow".into()))?;
        let out_b = mat_elems
            .checked_mul(sz_f32)
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| CudaMaczError::InvalidInput("byte size overflow".into()))?;
        let first_valids_b = cols
            .checked_mul(sz_i32)
            .ok_or_else(|| CudaMaczError::InvalidInput("byte size overflow".into()))?;
        let req = data_b
            .checked_add(prefix_base)
            .and_then(|v| v.checked_add(prefix_vol))
            .and_then(|v| v.checked_add(out_b))
            .and_then(|v| v.checked_add(first_valids_b))
            .ok_or_else(|| CudaMaczError::InvalidInput("byte size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        Self::will_fit(req, headroom)?;

        
        let d_close_tm = DeviceBuffer::from_slice(close_tm)?;
        let d_volume_tm = if let Some(v) = volume_tm {
            Some(DeviceBuffer::from_slice(v)?)
        } else {
            None
        };
        let (pcs, pcsq, pcn, vol_tuple) =
            Self::build_prefixes_time_major(close_tm, volume_tm, cols, rows);
        let d_pcs = DeviceBuffer::from_slice(&pcs)?;
        let d_pcsq = DeviceBuffer::from_slice(&pcsq)?;
        let d_pcn = DeviceBuffer::from_slice(&pcn)?;
        let (d_pvs, d_pps, d_pvn) = if let Some((pvs, pps, pvn)) = vol_tuple {
            (
                Some(DeviceBuffer::from_slice(&pvs)?),
                Some(DeviceBuffer::from_slice(&pps)?),
                Some(DeviceBuffer::from_slice(&pvn)?),
            )
        } else {
            (None, None, None)
        };

        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaMaczError::InvalidInput("cols*rows overflow".into()))?;
        let mut d_macz_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems)? };
        let mut d_hist_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems)? };

        
        let func = self
            .module
            .get_function("macz_many_series_one_param_time_major_f32")
            .map_err(|_| {
                CudaMaczError::MissingKernelSymbol {
                    name: "macz_many_series_one_param_time_major_f32",
                }
            })?;
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
        
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)?;
        unsafe {
            let grid: GridSize = (((cols as u32 + block_x - 1) / block_x).max(1), 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            self.validate_launch(grid, block)?;
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
            self.stream.launch(&func, grid, block, 0, &mut args)?;
        }
        self.stream.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_hist_tm,
            rows,
            cols,
        })
    }
}


#[cfg(any(test, feature = "cuda"))]
pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 100_000;
    const FIRST_VALID: usize = 50;
    const PARAM_COMBOS: usize = 256;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let vol_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let pref_slot = 2 * std::mem::size_of::<f64>() + std::mem::size_of::<i32>();
        let pref_close = (ONE_SERIES_LEN + 1) * pref_slot;
        let pref_vol = (ONE_SERIES_LEN + 1) * pref_slot;
        let params = PARAM_COMBOS * (5 * std::mem::size_of::<i32>() + 2 * std::mem::size_of::<f32>());
        let out = ONE_SERIES_LEN * PARAM_COMBOS * std::mem::size_of::<f32>();
        in_bytes + vol_bytes + pref_close + pref_vol + params + 2 * out + 64 * 1024 * 1024
    }

    struct MaczBatchDeviceState {
        cuda: CudaMacz,
        d_close: DeviceBuffer<f32>,
        d_volume: DeviceBuffer<f32>,
        d_pcs: DeviceBuffer<f64>,
        d_pcsq: DeviceBuffer<f64>,
        d_pcn: DeviceBuffer<i32>,
        d_pvs: DeviceBuffer<f64>,
        d_pps: DeviceBuffer<f64>,
        d_pvn: DeviceBuffer<i32>,
        d_fasts: DeviceBuffer<i32>,
        d_slows: DeviceBuffer<i32>,
        d_sigs: DeviceBuffer<i32>,
        d_lzs: DeviceBuffer<i32>,
        d_lsds: DeviceBuffer<i32>,
        d_as: DeviceBuffer<f32>,
        d_bs: DeviceBuffer<f32>,
        series_len: usize,
        n_rows: usize,
        first_valid: usize,
        d_macz_tmp: DeviceBuffer<f32>,
        d_hist: DeviceBuffer<f32>,
    }
    impl CudaBenchState for MaczBatchDeviceState {
        fn launch(&mut self) {
            self.cuda
                .macz_batch_device(
                    &self.d_close,
                    Some(&self.d_volume),
                    &self.d_pcs,
                    &self.d_pcsq,
                    &self.d_pcn,
                    Some(&self.d_pvs),
                    Some(&self.d_pps),
                    Some(&self.d_pvn),
                    &self.d_fasts,
                    &self.d_slows,
                    &self.d_sigs,
                    &self.d_lzs,
                    &self.d_lsds,
                    &self.d_as,
                    &self.d_bs,
                    self.series_len,
                    self.n_rows,
                    self.first_valid,
                    &mut self.d_macz_tmp,
                    &mut self.d_hist,
                )
                .expect("macz_batch_device");
            self.cuda.synchronize().expect("sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaMacz::new(0).expect("cuda macz");
        let len = ONE_SERIES_LEN;
        let mut price = vec![f32::NAN; len];
        let mut volume = vec![f32::NAN; len];
        for i in FIRST_VALID..len {
            let x = i as f32;
            price[i] = (x * 0.001).sin() + 0.0002 * x;
            volume[i] = (x * 0.0007).cos().abs() + 0.5;
        }
        let first_valid = FIRST_VALID;

        let sweep = MaczBatchRange {
            fast_length: (8, 14, 2),
            slow_length: (20, 35, 5),
            signal_length: (9, 9, 0),
            lengthz: (20, 20, 0),
            length_stdev: (25, 25, 0),
            a: (0.8, 1.1, 0.1),
            b: (0.8, 1.1, 0.1),
        };
        let combos = CudaMacz::expand_grid(&sweep).expect("expand_grid");
        assert_eq!(combos.len(), PARAM_COMBOS, "unexpected MACZ combo count");

        let (pcs, pcsq, pcn, vol_tuple) = CudaMacz::build_prefixes_single(&price, Some(&volume));
        let (pvs, pps, pvn) = vol_tuple.expect("volume prefixes");

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
        let lzs: Vec<i32> = combos.iter().map(|p| p.lengthz.unwrap_or(20) as i32).collect();
        let lsds: Vec<i32> = combos
            .iter()
            .map(|p| p.length_stdev.unwrap_or(25) as i32)
            .collect();
        let a_s: Vec<f32> = combos.iter().map(|p| p.a.unwrap_or(1.0) as f32).collect();
        let b_s: Vec<f32> = combos.iter().map(|p| p.b.unwrap_or(1.0) as f32).collect();

        let d_close = DeviceBuffer::from_slice(&price).expect("upload close");
        let d_volume = DeviceBuffer::from_slice(&volume).expect("upload volume");
        let d_pcs = DeviceBuffer::from_slice(&pcs).expect("upload pcs");
        let d_pcsq = DeviceBuffer::from_slice(&pcsq).expect("upload pcsq");
        let d_pcn = DeviceBuffer::from_slice(&pcn).expect("upload pcn");
        let d_pvs = DeviceBuffer::from_slice(&pvs).expect("upload pvs");
        let d_pps = DeviceBuffer::from_slice(&pps).expect("upload pps");
        let d_pvn = DeviceBuffer::from_slice(&pvn).expect("upload pvn");

        let d_fasts = DeviceBuffer::from_slice(&fasts).expect("upload fasts");
        let d_slows = DeviceBuffer::from_slice(&slows).expect("upload slows");
        let d_sigs = DeviceBuffer::from_slice(&sigs).expect("upload sigs");
        let d_lzs = DeviceBuffer::from_slice(&lzs).expect("upload lzs");
        let d_lsds = DeviceBuffer::from_slice(&lsds).expect("upload lsds");
        let d_as = DeviceBuffer::from_slice(&a_s).expect("upload a");
        let d_bs = DeviceBuffer::from_slice(&b_s).expect("upload b");

        let n_rows = PARAM_COMBOS;
        let out_elems = n_rows * len;
        let d_macz_tmp: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(out_elems) }.expect("alloc macz_tmp");
        let d_hist: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(out_elems) }.expect("alloc hist");

        Box::new(MaczBatchDeviceState {
            cuda,
            d_close,
            d_volume,
            d_pcs,
            d_pcsq,
            d_pcn,
            d_pvs,
            d_pps,
            d_pvn,
            d_fasts,
            d_slows,
            d_sigs,
            d_lzs,
            d_lsds,
            d_as,
            d_bs,
            series_len: len,
            n_rows,
            first_valid,
            d_macz_tmp,
            d_hist,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "macz",
            "one_series_many_params",
            "macz_cuda_batch_dev",
            "100k_x_256",
            prep_one_series_many_params,
        )
        .with_mem_required(bytes_one_series_many_params())]
    }
}
