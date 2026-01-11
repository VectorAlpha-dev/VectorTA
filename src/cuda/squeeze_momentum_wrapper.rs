//! CUDA wrapper for Squeeze Momentum Indicator (SMI).
//!
//! Parity with ALMA-style wrappers:
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/squeeze_momentum_kernel.ptx"))
//! - Stream NON_BLOCKING
//! - Minimal policies + introspection
//! - VRAM checks and grid chunking (grid.y <= 65_535)
//!
//! Exposes two device APIs:
//! - Batch (one series × many params): returns three DeviceArrayF32 handles
//! - Many-series × one param (time-major): returns three DeviceArrayF32 handles

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32;
use crate::indicators::squeeze_momentum::SqueezeMomentumBatchRange;
use cust::context::{CacheConfig, Context};
use cust::device::Device;
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;

#[derive(thiserror::Error, Debug)]
pub enum CudaSmiError {
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
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaSmiPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaSmiPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    OneD { block_x: u32 },
}
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}


#[derive(Clone, Debug)]
struct SmCombo {
    lbb: usize,
    mbb: f32,
    lkc: usize,
    mkc: f32,
}

pub struct CudaSqueezeMomentum {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaSmiPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
}

impl CudaSqueezeMomentum {
    pub fn new(device_id: usize) -> Result<Self, CudaSmiError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/squeeze_momentum_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                    m
                } else {
                    Module::from_ptx(ptx, &[])?
                }
            }
        };

        let _ = cust::context::CurrentContext::set_cache_config(CacheConfig::PreferL1);
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy: CudaSmiPolicy::default(),
            last_batch: None,
            last_many: None,
        })
    }

    pub fn set_policy(&mut self, policy: CudaSmiPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaSmiPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
    pub fn synchronize(&self) -> Result<(), CudaSmiError> {
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn context_arc(&self) -> Arc<Context> {
        self.context.clone()
    }

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
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Some((free, _)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    

    #[inline]
    fn sparse_k(n: usize) -> i32 {
        
        let mut k: i32 = 1;
        let mut span: usize = 1;
        while (span << 1) <= n {
            span <<= 1;
            k += 1;
        }
        k
    }

    #[inline]
    fn precompute_bytes(len: usize, k: usize) -> usize {
        
        let floats = (4 * len) + (2 * k * len);
        let ints = len + 1;
        floats * std::mem::size_of::<f32>() + ints * std::mem::size_of::<i32>()
    }

    fn expand_grid(sweep: &SqueezeMomentumBatchRange) -> Result<Vec<SmCombo>, CudaSmiError> {
        fn axis_usize((s, e, st): (usize, usize, usize)) -> Result<Vec<usize>, CudaSmiError> {
            if st == 0 {
                return Ok(vec![s]);
            }
            if s == e {
                return Ok(vec![s]);
            }
            let mut out = Vec::new();
            if s < e {
                let mut v = s;
                while v <= e {
                    out.push(v);
                    match v.checked_add(st) {
                        Some(next) => {
                            if next == v {
                                break;
                            }
                            v = next;
                        }
                        None => break,
                    }
                }
            } else {
                let mut v = s;
                while v >= e {
                    out.push(v);
                    if v == 0 {
                        break;
                    }
                    let next = v.saturating_sub(st);
                    if next == v {
                        break;
                    }
                    v = next;
                    if v < e {
                        break;
                    }
                }
            }
            if out.is_empty() {
                return Err(CudaSmiError::InvalidInput(
                    "invalid range for batch axis (usize)".into(),
                ));
            }
            Ok(out)
        }
        fn axis_f64((s, e, st): (f64, f64, f64)) -> Result<Vec<f64>, CudaSmiError> {
            if st.abs() < 1e-12 || (s - e).abs() < 1e-12 {
                return Ok(vec![s]);
            }
            let mut out = Vec::new();
            if s < e {
                let mut x = s;
                let step = st.abs();
                while x <= e + 1e-12 {
                    out.push(x);
                    x += step;
                }
            } else {
                let mut x = s;
                let step = st.abs();
                while x + 1e-12 >= e {
                    out.push(x);
                    x -= step;
                }
            }
            if out.is_empty() {
                return Err(CudaSmiError::InvalidInput(
                    "invalid range for batch axis (f64)".into(),
                ));
            }
            Ok(out)
        }
        let lbb = axis_usize(sweep.length_bb)?;
        let mbb = axis_f64(sweep.mult_bb)?;
        let lkc = axis_usize(sweep.length_kc)?;
        let mkc = axis_f64(sweep.mult_kc)?;
        let cap = lbb
            .len()
            .checked_mul(mbb.len())
            .and_then(|x| x.checked_mul(lkc.len()))
            .and_then(|x| x.checked_mul(mkc.len()))
            .ok_or_else(|| CudaSmiError::InvalidInput("rows*cols overflow".into()))?;
        let mut out = Vec::with_capacity(cap);
        for &a in &lbb {
            for &b in &mbb {
                for &c in &lkc {
                    for &d in &mkc {
                        out.push(SmCombo {
                            lbb: a,
                            mbb: b as f32,
                            lkc: c,
                            mkc: d as f32,
                        });
                    }
                }
            }
        }
        if out.is_empty() {
            return Err(CudaSmiError::InvalidInput(
                "no parameter combos after expansion".into(),
            ));
        }
        Ok(out)
    }

    fn prepare_batch_inputs(
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &SqueezeMomentumBatchRange,
    ) -> Result<(Vec<SmCombo>, usize, usize), CudaSmiError> {
        if high.len() != low.len() || low.len() != close.len() {
            return Err(CudaSmiError::InvalidInput(
                "inconsistent array lengths".into(),
            ));
        }
        if close.is_empty() {
            return Err(CudaSmiError::InvalidInput("empty data".into()));
        }
        let len = close.len();
        let first_valid = (0..len)
            .find(|&i| !(high[i].is_nan() || low[i].is_nan() || close[i].is_nan()))
            .ok_or_else(|| CudaSmiError::InvalidInput("all values are NaN".into()))?;
        let combos = Self::expand_grid(sweep)?;
        
        let mut need = 0usize;
        for c in &combos {
            need = need.max(c.lbb.max(c.lkc));
        }
        let tail = len - first_valid;
        if tail < need {
            return Err(CudaSmiError::InvalidInput(format!(
                "not enough valid data: needed {}, valid {}",
                need, tail
            )));
        }
        Ok((combos, first_valid, len))
    }

    fn launch_batch_kernel(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        d_lbb: &DeviceBuffer<i32>,
        d_mbb: &DeviceBuffer<f32>,
        d_lkc: &DeviceBuffer<i32>,
        d_mkc: &DeviceBuffer<f32>,
        len: usize,
        n_combos: usize,
        first_valid: usize,
        d_sq: &mut DeviceBuffer<f32>,
        d_mo: &mut DeviceBuffer<f32>,
        d_si: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSmiError> {
        let mut func: Function = self
            .module
            .get_function("squeeze_momentum_batch_f32")
            .map_err(|_| CudaSmiError::MissingKernelSymbol { name: "squeeze_momentum_batch_f32" })?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);

        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Auto => 256,
            BatchKernelPolicy::OneD { block_x } => block_x.max(32),
        };
        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut p_h = d_high.as_device_ptr().as_raw();
            let mut p_l = d_low.as_device_ptr().as_raw();
            let mut p_c = d_close.as_device_ptr().as_raw();
            let mut p_lbb = d_lbb.as_device_ptr().as_raw();
            let mut p_mbb = d_mbb.as_device_ptr().as_raw();
            let mut p_lkc = d_lkc.as_device_ptr().as_raw();
            let mut p_mkc = d_mkc.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut n_i = n_combos as i32;
            let mut fv_i = first_valid as i32;
            let mut p_sq = d_sq.as_device_ptr().as_raw();
            let mut p_mo = d_mo.as_device_ptr().as_raw();
            let mut p_si = d_si.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 13] = [
                &mut p_h as *mut _ as *mut c_void,
                &mut p_l as *mut _ as *mut c_void,
                &mut p_c as *mut _ as *mut c_void,
                &mut p_lbb as *mut _ as *mut c_void,
                &mut p_mbb as *mut _ as *mut c_void,
                &mut p_lkc as *mut _ as *mut c_void,
                &mut p_mkc as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut fv_i as *mut _ as *mut c_void,
                &mut p_sq as *mut _ as *mut c_void,
                &mut p_mo as *mut _ as *mut c_void,
                &mut p_si as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, &mut args)?;
        }
        unsafe {
            (*(self as *const _ as *mut CudaSqueezeMomentum)).last_batch =
                Some(BatchKernelSelected::OneD { block_x });
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_precompute(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        len: usize,
        d_tr: &mut DeviceBuffer<f32>,
        d_ps_close: &mut DeviceBuffer<f32>,
        d_ps_close2: &mut DeviceBuffer<f32>,
        d_ps_tr: &mut DeviceBuffer<f32>,
        d_log2: &mut DeviceBuffer<i32>,
        d_st_max: &mut DeviceBuffer<f32>,
        d_st_min: &mut DeviceBuffer<f32>,
        k_levels: i32,
    ) -> Result<(), CudaSmiError> {
        let mut func: Function = self
            .module
            .get_function("smi_precompute_shared_f32")
            .map_err(|_| CudaSmiError::MissingKernelSymbol { name: "smi_precompute_shared_f32" })?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);
        let grid: GridSize = (1, 1, 1).into();
        let block: BlockSize = (256, 1, 1).into();
        unsafe {
            let mut p_h = d_high.as_device_ptr().as_raw();
            let mut p_l = d_low.as_device_ptr().as_raw();
            let mut p_c = d_close.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut p_tr = d_tr.as_device_ptr().as_raw();
            let mut p_psc = d_ps_close.as_device_ptr().as_raw();
            let mut p_psc2 = d_ps_close2.as_device_ptr().as_raw();
            let mut p_pstr = d_ps_tr.as_device_ptr().as_raw();
            let mut p_log2 = d_log2.as_device_ptr().as_raw();
            let mut p_stmax = d_st_max.as_device_ptr().as_raw();
            let mut p_stmin = d_st_min.as_device_ptr().as_raw();
            let mut k_i = k_levels as i32;
            let mut args: [*mut c_void; 12] = [
                &mut p_h as *mut _ as *mut c_void,
                &mut p_l as *mut _ as *mut c_void,
                &mut p_c as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut p_tr as *mut _ as *mut c_void,
                &mut p_psc as *mut _ as *mut c_void,
                &mut p_psc2 as *mut _ as *mut c_void,
                &mut p_pstr as *mut _ as *mut c_void,
                &mut p_log2 as *mut _ as *mut c_void,
                &mut p_stmax as *mut _ as *mut c_void,
                &mut p_stmin as *mut _ as *mut c_void,
                &mut k_i as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, &mut args)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel_opt(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        d_lbb: &DeviceBuffer<i32>,
        d_mbb: &DeviceBuffer<f32>,
        d_lkc: &DeviceBuffer<i32>,
        d_mkc: &DeviceBuffer<f32>,
        len: usize,
        n_combos: usize,
        first_valid: usize,
        d_tr: &DeviceBuffer<f32>,
        d_ps_close: &DeviceBuffer<f32>,
        d_ps_close2: &DeviceBuffer<f32>,
        d_ps_tr: &DeviceBuffer<f32>,
        d_log2: &DeviceBuffer<i32>,
        d_st_max: &DeviceBuffer<f32>,
        d_st_min: &DeviceBuffer<f32>,
        k_levels: i32,
        max_lkc: usize,
        d_sq: &mut DeviceBuffer<f32>,
        d_mo: &mut DeviceBuffer<f32>,
        d_si: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSmiError> {
        let mut func: Function = self
            .module
            .get_function("squeeze_momentum_batch_f32_opt")
            .map_err(|_| CudaSmiError::MissingKernelSymbol { name: "squeeze_momentum_batch_f32_opt" })?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);

        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Auto => 128,
            BatchKernelPolicy::OneD { block_x } => block_x.max(32),
        };
        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        let shared_bytes: u32 = (max_lkc as u32) * std::mem::size_of::<f32>() as u32;

        unsafe {
            let mut p_h = d_high.as_device_ptr().as_raw();
            let mut p_l = d_low.as_device_ptr().as_raw();
            let mut p_c = d_close.as_device_ptr().as_raw();
            let mut p_lbb = d_lbb.as_device_ptr().as_raw();
            let mut p_mbb = d_mbb.as_device_ptr().as_raw();
            let mut p_lkc = d_lkc.as_device_ptr().as_raw();
            let mut p_mkc = d_mkc.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut n_i = n_combos as i32;
            let mut fv_i = first_valid as i32;
            let mut p_tr = d_tr.as_device_ptr().as_raw();
            let mut p_psc = d_ps_close.as_device_ptr().as_raw();
            let mut p_psc2 = d_ps_close2.as_device_ptr().as_raw();
            let mut p_pstr = d_ps_tr.as_device_ptr().as_raw();
            let mut p_log2 = d_log2.as_device_ptr().as_raw();
            let mut p_stmax = d_st_max.as_device_ptr().as_raw();
            let mut p_stmin = d_st_min.as_device_ptr().as_raw();
            let mut k_i = k_levels as i32;
            let mut p_sq = d_sq.as_device_ptr().as_raw();
            let mut p_mo = d_mo.as_device_ptr().as_raw();
            let mut p_si = d_si.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 21] = [
                &mut p_h as *mut _ as *mut c_void,
                &mut p_l as *mut _ as *mut c_void,
                &mut p_c as *mut _ as *mut c_void,
                &mut p_lbb as *mut _ as *mut c_void,
                &mut p_mbb as *mut _ as *mut c_void,
                &mut p_lkc as *mut _ as *mut c_void,
                &mut p_mkc as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut fv_i as *mut _ as *mut c_void,
                &mut p_tr as *mut _ as *mut c_void,
                &mut p_psc as *mut _ as *mut c_void,
                &mut p_psc2 as *mut _ as *mut c_void,
                &mut p_pstr as *mut _ as *mut c_void,
                &mut p_log2 as *mut _ as *mut c_void,
                &mut p_stmax as *mut _ as *mut c_void,
                &mut p_stmin as *mut _ as *mut c_void,
                &mut k_i as *mut _ as *mut c_void,
                &mut p_sq as *mut _ as *mut c_void,
                &mut p_mo as *mut _ as *mut c_void,
                &mut p_si as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes, &mut args)?;
        }
        unsafe {
            (*(self as *const _ as *mut CudaSqueezeMomentum)).last_batch =
                Some(BatchKernelSelected::OneD { block_x });
        }
        Ok(())
    }

    pub fn squeeze_momentum_batch_dev(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        close_f32: &[f32],
        sweep: &SqueezeMomentumBatchRange,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32, DeviceArrayF32), CudaSmiError> {
        let (combos, first_valid, len) =
            Self::prepare_batch_inputs(high_f32, low_f32, close_f32, sweep)?;

        
        let in_bytes = 3usize
            .saturating_mul(len)
            .saturating_mul(std::mem::size_of::<f32>());
        let params_bytes = combos
            .len()
            .saturating_mul(2 * std::mem::size_of::<i32>() + 2 * std::mem::size_of::<f32>());
        let out_bytes = 3usize
            .saturating_mul(combos.len())
            .saturating_mul(len)
            .saturating_mul(std::mem::size_of::<f32>());
        let k_levels_us = Self::sparse_k(len) as usize;
        let pre_bytes = Self::precompute_bytes(len, k_levels_us);
        let required = in_bytes
            .saturating_add(params_bytes)
            .saturating_add(out_bytes)
            .saturating_add(pre_bytes);
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            if let Some((free, _)) = Self::device_mem_info() {
                return Err(CudaSmiError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaSmiError::InvalidInput(
                    "insufficient device memory".into(),
                ));
            }
        }

        
        let d_h = DeviceBuffer::from_slice(high_f32)?;
        let d_l = DeviceBuffer::from_slice(low_f32)?;
        let d_c = DeviceBuffer::from_slice(close_f32)?;

        let v_lbb: Vec<i32> = combos.iter().map(|c| c.lbb as i32).collect();
        let v_mbb: Vec<f32> = combos.iter().map(|c| c.mbb).collect();
        let v_lkc: Vec<i32> = combos.iter().map(|c| c.lkc as i32).collect();
        let v_mkc: Vec<f32> = combos.iter().map(|c| c.mkc).collect();
        let d_lbb = DeviceBuffer::from_slice(&v_lbb)?;
        let d_mbb = DeviceBuffer::from_slice(&v_mbb)?;
        let d_lkc = DeviceBuffer::from_slice(&v_lkc)?;
        let d_mkc = DeviceBuffer::from_slice(&v_mkc)?;

        
        let elems = combos
            .len()
            .checked_mul(len)
            .ok_or_else(|| CudaSmiError::InvalidInput("rows*cols overflow".into()))?;
        let mut d_sq: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }?;
        let mut d_mo: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }?;
        let mut d_si: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }?;

        
        let n = len;
        let mut d_tr: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n) }?;
        let mut d_ps_close: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n) }?;
        let mut d_ps_close2: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n) }?;
        let mut d_ps_tr: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n) }?;
        let mut d_log2: DeviceBuffer<i32> = unsafe { DeviceBuffer::uninitialized(n + 1) }?;
        let st_size = k_levels_us * n;
        let mut d_st_max: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(st_size) }?;
        let mut d_st_min: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(st_size) }?;

        
        self.launch_precompute(
            &d_h,
            &d_l,
            &d_c,
            len,
            &mut d_tr,
            &mut d_ps_close,
            &mut d_ps_close2,
            &mut d_ps_tr,
            &mut d_log2,
            &mut d_st_max,
            &mut d_st_min,
            Self::sparse_k(len),
        )?;

        
        let max_lkc = combos.iter().map(|c| c.lkc).max().unwrap_or(1);
        self.launch_batch_kernel_opt(
            &d_h,
            &d_l,
            &d_c,
            &d_lbb,
            &d_mbb,
            &d_lkc,
            &d_mkc,
            len,
            combos.len(),
            first_valid,
            &d_tr,
            &d_ps_close,
            &d_ps_close2,
            &d_ps_tr,
            &d_log2,
            &d_st_max,
            &d_st_min,
            Self::sparse_k(len),
            max_lkc,
            &mut d_sq,
            &mut d_mo,
            &mut d_si,
        )?;
        self.stream.synchronize()?;

        Ok((
            DeviceArrayF32 {
                buf: d_sq,
                rows: combos.len(),
                cols: len,
            },
            DeviceArrayF32 {
                buf: d_mo,
                rows: combos.len(),
                cols: len,
            },
            DeviceArrayF32 {
                buf: d_si,
                rows: combos.len(),
                cols: len,
            },
        ))
    }

    
    pub fn squeeze_momentum_many_series_one_param_time_major_dev(
        &self,
        high_tm_f32: &[f32],
        low_tm_f32: &[f32],
        close_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        lbb: usize,
        mbb: f32,
        lkc: usize,
        mkc: f32,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32, DeviceArrayF32), CudaSmiError> {
        if cols == 0 || rows == 0 {
            return Err(CudaSmiError::InvalidInput("cols or rows is zero".into()));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaSmiError::InvalidInput("cols*rows overflow".into()))?;
        if high_tm_f32.len() != expected
            || low_tm_f32.len() != expected
            || close_tm_f32.len() != expected
        {
            return Err(CudaSmiError::InvalidInput(
                "time-major arrays length mismatch".into(),
            ));
        }
        if lbb == 0 || lkc == 0 || lbb > rows || lkc > rows {
            return Err(CudaSmiError::InvalidInput("invalid window lengths".into()));
        }
        
        let in_bytes = 3usize
            .saturating_mul(expected)
            .saturating_mul(std::mem::size_of::<f32>());
        let out_bytes = 3usize
            .saturating_mul(expected)
            .saturating_mul(std::mem::size_of::<f32>());
        let fv_bytes = cols.saturating_mul(std::mem::size_of::<i32>());
        let required = in_bytes
            .saturating_add(out_bytes)
            .saturating_add(fv_bytes);
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            if let Some((free, _)) = Self::device_mem_info() {
                return Err(CudaSmiError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaSmiError::InvalidInput(
                    "insufficient device memory".into(),
                ));
            }
        }

        
        let mut fv = vec![0i32; cols];
        for s in 0..cols {
            let mut found = None;
            for r in 0..rows {
                let idx = r * cols + s;
                let h = high_tm_f32[idx];
                let l = low_tm_f32[idx];
                let c = close_tm_f32[idx];
                if !(h.is_nan() || l.is_nan() || c.is_nan()) {
                    found = Some(r);
                    break;
                }
            }
            let fv_s =
                found.ok_or_else(|| CudaSmiError::InvalidInput(format!("series {} all NaN", s)))?;
            if rows - fv_s < lbb.max(lkc) {
                return Err(CudaSmiError::InvalidInput(format!(
                    "series {} not enough valid data (needed {}, valid {})",
                    s,
                    lbb.max(lkc),
                    rows - fv_s
                )));
            }
            fv[s] = fv_s as i32;
        }

        let d_h = DeviceBuffer::from_slice(high_tm_f32)?;
        let d_l = DeviceBuffer::from_slice(low_tm_f32)?;
        let d_c = DeviceBuffer::from_slice(close_tm_f32)?;
        let d_fv = DeviceBuffer::from_slice(&fv)?;
        let mut d_sq_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(expected) }?;
        let mut d_mo_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(expected) }?;
        let mut d_si_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(expected) }?;

        let func = self
            .module
            .get_function("squeeze_momentum_many_series_one_param_f32")
            .map_err(|_| CudaSmiError::MissingKernelSymbol {
                name: "squeeze_momentum_many_series_one_param_f32",
            })?;

        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 1,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(1),
        };
        let grid: GridSize = (1, cols as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut p_h = d_h.as_device_ptr().as_raw();
            let mut p_l = d_l.as_device_ptr().as_raw();
            let mut p_c = d_c.as_device_ptr().as_raw();
            let mut p_fv = d_fv.as_device_ptr().as_raw();
            let mut cols_i = cols as i32; 
            let mut rows_i = rows as i32; 
            let mut lbb_i = lbb as i32;
            let mut mbb_f = mbb as f32;
            let mut lkc_i = lkc as i32;
            let mut mkc_f = mkc as f32;
            let mut p_sq = d_sq_tm.as_device_ptr().as_raw();
            let mut p_mo = d_mo_tm.as_device_ptr().as_raw();
            let mut p_si = d_si_tm.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 13] = [
                &mut p_h as *mut _ as *mut c_void,
                &mut p_l as *mut _ as *mut c_void,
                &mut p_c as *mut _ as *mut c_void,
                &mut p_fv as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut lbb_i as *mut _ as *mut c_void,
                &mut mbb_f as *mut _ as *mut c_void,
                &mut lkc_i as *mut _ as *mut c_void,
                &mut mkc_f as *mut _ as *mut c_void,
                &mut p_sq as *mut _ as *mut c_void,
                &mut p_mo as *mut _ as *mut c_void,
                &mut p_si as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, &mut args)?;
        }
        self.stream.synchronize()?;
        unsafe {
            (*(self as *const _ as *mut CudaSqueezeMomentum)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }

        Ok((
            DeviceArrayF32 {
                buf: d_sq_tm,
                rows,
                cols,
            },
            DeviceArrayF32 {
                buf: d_mo_tm,
                rows,
                cols,
            },
            DeviceArrayF32 {
                buf: d_si_tm,
                rows,
                cols,
            },
        ))
    }
}


pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_00; 
    const PARAM_SWEEP: usize = 128;

    fn bytes_one_series_many_params() -> usize {
        let n = ONE_SERIES_LEN;
        let k = CudaSqueezeMomentum::sparse_k(n) as usize;
        let pre = CudaSqueezeMomentum::precompute_bytes(n, k);
        let in_bytes = 3 * n * std::mem::size_of::<f32>();
        let out_bytes = 3 * n * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + pre + 64 * 1024 * 1024
    }

    struct SmiBatchDeviceState {
        cuda: CudaSqueezeMomentum,
        d_h: DeviceBuffer<f32>,
        d_l: DeviceBuffer<f32>,
        d_c: DeviceBuffer<f32>,
        d_lbb: DeviceBuffer<i32>,
        d_mbb: DeviceBuffer<f32>,
        d_lkc: DeviceBuffer<i32>,
        d_mkc: DeviceBuffer<f32>,
        len: usize,
        n_combos: usize,
        first_valid: usize,
        k: i32,
        max_lkc: usize,
        
        d_tr: DeviceBuffer<f32>,
        d_ps_close: DeviceBuffer<f32>,
        d_ps_close2: DeviceBuffer<f32>,
        d_ps_tr: DeviceBuffer<f32>,
        d_log2: DeviceBuffer<i32>,
        d_st_max: DeviceBuffer<f32>,
        d_st_min: DeviceBuffer<f32>,
        
        d_sq: DeviceBuffer<f32>,
        d_mo: DeviceBuffer<f32>,
        d_si: DeviceBuffer<f32>,
    }
    impl CudaBenchState for SmiBatchDeviceState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel_opt(
                    &self.d_h,
                    &self.d_l,
                    &self.d_c,
                    &self.d_lbb,
                    &self.d_mbb,
                    &self.d_lkc,
                    &self.d_mkc,
                    self.len,
                    self.n_combos,
                    self.first_valid,
                    &self.d_tr,
                    &self.d_ps_close,
                    &self.d_ps_close2,
                    &self.d_ps_tr,
                    &self.d_log2,
                    &self.d_st_max,
                    &self.d_st_min,
                    self.k,
                    self.max_lkc,
                    &mut self.d_sq,
                    &mut self.d_mo,
                    &mut self.d_si,
                )
                .expect("smi launch");
            self.cuda.stream.synchronize().expect("smi sync");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaSqueezeMomentum::new(0).expect("cuda smi");
        let h = gen_series(ONE_SERIES_LEN);
        let mut l = h.clone();
        for v in &mut l {
            *v -= 0.5;
        }
        let mut c = h.clone();
        for v in &mut c {
            *v -= 0.25;
        }
        let sweep = SqueezeMomentumBatchRange {
            length_bb: (10, 10 + PARAM_SWEEP - 1, 1),
            mult_bb: (2.0, 2.0, 0.0),
            length_kc: (10, 10, 0),
            mult_kc: (1.5, 1.5, 0.0),
        };

        let (combos, first_valid, len) =
            CudaSqueezeMomentum::prepare_batch_inputs(&h, &l, &c, &sweep).expect("prep inputs");
        let n_combos = combos.len();
        let v_lbb: Vec<i32> = combos.iter().map(|c| c.lbb as i32).collect();
        let v_mbb: Vec<f32> = combos.iter().map(|c| c.mbb).collect();
        let v_lkc: Vec<i32> = combos.iter().map(|c| c.lkc as i32).collect();
        let v_mkc: Vec<f32> = combos.iter().map(|c| c.mkc).collect();
        let max_lkc = combos.iter().map(|c| c.lkc).max().unwrap_or(1);

        
        let d_h = DeviceBuffer::from_slice(&h).expect("d_h H2D");
        let d_l = DeviceBuffer::from_slice(&l).expect("d_l H2D");
        let d_c = DeviceBuffer::from_slice(&c).expect("d_c H2D");
        let d_lbb = DeviceBuffer::from_slice(&v_lbb).expect("d_lbb H2D");
        let d_mbb = DeviceBuffer::from_slice(&v_mbb).expect("d_mbb H2D");
        let d_lkc = DeviceBuffer::from_slice(&v_lkc).expect("d_lkc H2D");
        let d_mkc = DeviceBuffer::from_slice(&v_mkc).expect("d_mkc H2D");

        
        let n = len;
        let k = CudaSqueezeMomentum::sparse_k(len);
        let k_levels_us = k as usize;
        let mut d_tr: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n) }.expect("d_tr");
        let mut d_ps_close: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n) }.expect("d_ps_close");
        let mut d_ps_close2: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n) }.expect("d_ps_close2");
        let mut d_ps_tr: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n) }.expect("d_ps_tr");
        let mut d_log2: DeviceBuffer<i32> =
            unsafe { DeviceBuffer::uninitialized(n + 1) }.expect("d_log2");
        let st_size = k_levels_us * n;
        let mut d_st_max: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(st_size) }.expect("d_st_max");
        let mut d_st_min: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(st_size) }.expect("d_st_min");
        cuda.launch_precompute(
            &d_h,
            &d_l,
            &d_c,
            len,
            &mut d_tr,
            &mut d_ps_close,
            &mut d_ps_close2,
            &mut d_ps_tr,
            &mut d_log2,
            &mut d_st_max,
            &mut d_st_min,
            k,
        )
        .expect("precompute");

        
        let elems = n_combos * len;
        let d_sq: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_sq");
        let d_mo: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_mo");
        let d_si: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_si");
        cuda.stream.synchronize().expect("smi prep sync");

        Box::new(SmiBatchDeviceState {
            cuda,
            d_h,
            d_l,
            d_c,
            d_lbb,
            d_mbb,
            d_lkc,
            d_mkc,
            len,
            n_combos,
            first_valid,
            k,
            max_lkc,
            d_tr,
            d_ps_close,
            d_ps_close2,
            d_ps_tr,
            d_log2,
            d_st_max,
            d_st_min,
            d_sq,
            d_mo,
            d_si,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "squeeze_momentum",
            "one_series_many_params",
            "squeeze_momentum_cuda_batch_dev",
            "100k_x_128",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}
