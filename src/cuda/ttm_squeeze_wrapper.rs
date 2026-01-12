#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32;
use crate::indicators::ttm_squeeze::TtmSqueezeBatchRange;
use cust::context::{CacheConfig, Context, CurrentContext};
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CudaTtmSqueezeError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
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
pub struct CudaTtmSqueezePolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaTtmSqueezePolicy {
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
struct Combo {
    length: i32,
    bb_mult: f32,
    kc_high: f32,
    kc_mid: f32,
    kc_low: f32,
}

pub struct CudaTtmSqueeze {
    module: Module,
    stream: Stream,
    context: std::sync::Arc<Context>,
    device_id: u32,
    policy: CudaTtmSqueezePolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    smem_limit_optin: usize,
}

impl CudaTtmSqueeze {
    pub fn new(device_id: usize) -> Result<Self, CudaTtmSqueezeError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = std::sync::Arc::new(Context::new(device)?);

        let def = device.get_attribute(DeviceAttribute::MaxSharedMemoryPerBlock)? as usize;

        let smem_limit_optin = def;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/ttm_squeeze_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))?;

        let _ = CurrentContext::set_cache_config(CacheConfig::PreferShared);
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy: CudaTtmSqueezePolicy::default(),
            last_batch: None,
            last_many: None,
            smem_limit_optin,
        })
    }

    const SHMEM_ELEM_BYTES: usize = 2 * std::mem::size_of::<i32>()
        + 2 * std::mem::size_of::<f32>()
        + 2 * std::mem::size_of::<u8>();

    #[inline(always)]
    fn smem_bytes_for_len(len: usize) -> usize {
        len.saturating_mul(Self::SHMEM_ELEM_BYTES)
    }

    pub fn set_policy(&mut self, policy: CudaTtmSqueezePolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaTtmSqueezePolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
    pub fn synchronize(&self) -> Result<(), CudaTtmSqueezeError> {
        self.stream.synchronize().map_err(CudaTtmSqueezeError::from)
    }
    pub fn context_arc(&self) -> std::sync::Arc<Context> {
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

    fn expand_grid(range: &TtmSqueezeBatchRange) -> Result<Vec<Combo>, CudaTtmSqueezeError> {
        fn axis_usize((s, e, st): (usize, usize, usize)) -> Vec<usize> {
            if st == 0 || s == e {
                return vec![s];
            }
            let mut v = Vec::new();
            if s < e {
                let mut x = s;
                while x <= e {
                    v.push(x);
                    match x.checked_add(st) {
                        Some(next) => {
                            if next == x {
                                break;
                            }
                            x = next;
                        }
                        None => break,
                    }
                }
            } else {
                let mut x = s;
                loop {
                    if x < e {
                        break;
                    }
                    v.push(x);
                    match x.checked_sub(st) {
                        Some(next) => {
                            if next == x {
                                break;
                            }
                            x = next;
                        }
                        None => break,
                    }
                }
            }
            v
        }
        fn axis_f64((s, e, st): (f64, f64, f64)) -> Vec<f64> {
            let step = st.abs();
            if (step < 1e-12) || ((s - e).abs() < 1e-12) {
                return vec![s];
            }
            let mut v = Vec::new();
            let mut x = s;
            if s <= e {
                while x <= e + 1e-12 {
                    v.push(x);
                    x += step;
                }
            } else {
                while x >= e - 1e-12 {
                    v.push(x);
                    x -= step;
                }
            }
            v
        }
        let lengths = axis_usize(range.length);
        let bb = axis_f64(range.bb_mult);
        let kh = axis_f64(range.kc_high);
        let km = axis_f64(range.kc_mid);
        let kl = axis_f64(range.kc_low);
        if lengths.is_empty() || bb.is_empty() || kh.is_empty() || km.is_empty() || kl.is_empty() {
            return Err(CudaTtmSqueezeError::InvalidInput(
                "empty sweep axis for ttm_squeeze batch".into(),
            ));
        }
        let cap = lengths
            .len()
            .checked_mul(bb.len())
            .and_then(|v| v.checked_mul(kh.len()))
            .and_then(|v| v.checked_mul(km.len()))
            .and_then(|v| v.checked_mul(kl.len()))
            .ok_or_else(|| {
                CudaTtmSqueezeError::InvalidInput(
                    "size overflow when expanding ttm_squeeze sweep grid".into(),
                )
            })?;

        let mut out = Vec::with_capacity(cap);
        for &l in &lengths {
            for &b in &bb {
                for &h in &kh {
                    for &m in &km {
                        for &lo in &kl {
                            out.push(Combo {
                                length: l as i32,
                                bb_mult: b as f32,
                                kc_high: h as f32,
                                kc_mid: m as f32,
                                kc_low: lo as f32,
                            });
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    fn prepare_batch_inputs(
        high_f32: &[f32],
        low_f32: &[f32],
        close_f32: &[f32],
        sweep: &TtmSqueezeBatchRange,
    ) -> Result<(Vec<Combo>, usize, usize), CudaTtmSqueezeError> {
        if high_f32.len() != low_f32.len() || low_f32.len() != close_f32.len() {
            return Err(CudaTtmSqueezeError::InvalidInput(format!(
                "inconsistent lengths: high={}, low={}, close={}",
                high_f32.len(),
                low_f32.len(),
                close_f32.len()
            )));
        }
        if close_f32.is_empty() {
            return Err(CudaTtmSqueezeError::InvalidInput("empty series".into()));
        }
        let len = close_f32.len();
        let first_valid = close_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaTtmSqueezeError::InvalidInput("all values are NaN".into()))?;
        let combos = Self::expand_grid(sweep)?;
        if combos.is_empty() {
            return Err(CudaTtmSqueezeError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        for c in &combos {
            if c.length <= 0 || (len - first_valid) < (c.length as usize) {
                return Err(CudaTtmSqueezeError::InvalidInput(
                    "invalid length or insufficient data".into(),
                ));
            }
        }
        Ok((combos, first_valid, len))
    }

    fn launch_batch_kernel(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        d_len: &DeviceBuffer<i32>,
        d_bb: &DeviceBuffer<f32>,
        d_kh: &DeviceBuffer<f32>,
        d_km: &DeviceBuffer<f32>,
        d_kl: &DeviceBuffer<f32>,
        len: usize,
        n_combos: usize,
        first_valid: usize,
        l_max: usize,
        d_mo: &mut DeviceBuffer<f32>,
        d_sq: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTtmSqueezeError> {
        let mut func: Function =
            self.module
                .get_function("ttm_squeeze_batch_f32")
                .map_err(|_| CudaTtmSqueezeError::MissingKernelSymbol {
                    name: "ttm_squeeze_batch_f32",
                })?;

        let smem_bytes = Self::smem_bytes_for_len(l_max);
        if smem_bytes > self.smem_limit_optin {
            let max_L = self.smem_limit_optin / Self::SHMEM_ELEM_BYTES;
            return Err(CudaTtmSqueezeError::InvalidInput(format!(
                "window length {} requires {} bytes of dynamic shared memory, exceeding device limit {} bytes. Maximum supported length on this GPU is {}.",
                l_max, smem_bytes, self.smem_limit_optin, max_L
            )));
        }
        let _ = func.set_cache_config(CacheConfig::PreferShared);
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Auto => 256,
            BatchKernelPolicy::OneD { block_x } => block_x.max(32),
        };
        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut p_h = d_high.as_device_ptr().as_raw();
            let mut p_l = d_low.as_device_ptr().as_raw();
            let mut p_c = d_close.as_device_ptr().as_raw();
            let mut p_len = d_len.as_device_ptr().as_raw();
            let mut p_bb = d_bb.as_device_ptr().as_raw();
            let mut p_kh = d_kh.as_device_ptr().as_raw();
            let mut p_km = d_km.as_device_ptr().as_raw();
            let mut p_kl = d_kl.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut n_i = n_combos as i32;
            let mut fv_i = first_valid as i32;
            let mut p_mo = d_mo.as_device_ptr().as_raw();
            let mut p_sq = d_sq.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 14] = [
                &mut p_h as *mut _ as *mut c_void,
                &mut p_l as *mut _ as *mut c_void,
                &mut p_c as *mut _ as *mut c_void,
                &mut p_len as *mut _ as *mut c_void,
                &mut p_bb as *mut _ as *mut c_void,
                &mut p_kh as *mut _ as *mut c_void,
                &mut p_km as *mut _ as *mut c_void,
                &mut p_kl as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut fv_i as *mut _ as *mut c_void,
                &mut p_mo as *mut _ as *mut c_void,
                &mut p_sq as *mut _ as *mut c_void,
                std::ptr::null_mut(),
            ];
            self.stream
                .launch(&func, grid, block, smem_bytes as u32, &mut args)
                .map_err(CudaTtmSqueezeError::from)?;
        }
        unsafe {
            (*(self as *const _ as *mut CudaTtmSqueeze)).last_batch =
                Some(BatchKernelSelected::OneD { block_x });
        }
        Ok(())
    }

    pub fn ttm_squeeze_batch_dev(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        close_f32: &[f32],
        sweep: &TtmSqueezeBatchRange,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32), CudaTtmSqueezeError> {
        let (combos, first_valid, len) =
            Self::prepare_batch_inputs(high_f32, low_f32, close_f32, sweep)?;

        let elem = std::mem::size_of::<f32>();
        let in_bytes = len
            .checked_mul(3)
            .and_then(|v| v.checked_mul(elem))
            .ok_or_else(|| {
                CudaTtmSqueezeError::InvalidInput("size overflow in input bytes".into())
            })?;
        let per_combo_bytes = std::mem::size_of::<i32>() + 4 * elem;
        let params_bytes = combos.len().checked_mul(per_combo_bytes).ok_or_else(|| {
            CudaTtmSqueezeError::InvalidInput("size overflow in params bytes".into())
        })?;
        let out_elems = combos
            .len()
            .checked_mul(len)
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| {
                CudaTtmSqueezeError::InvalidInput("size overflow in output elements".into())
            })?;
        let out_bytes = out_elems.checked_mul(elem).ok_or_else(|| {
            CudaTtmSqueezeError::InvalidInput("size overflow in output bytes".into())
        })?;
        let required = in_bytes
            .checked_add(params_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| {
                CudaTtmSqueezeError::InvalidInput("size overflow in total bytes".into())
            })?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            if let Some((free, _)) = Self::device_mem_info() {
                return Err(CudaTtmSqueezeError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaTtmSqueezeError::InvalidInput(
                    "insufficient device memory".into(),
                ));
            }
        }

        let mut d_h: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(len) }.map_err(CudaTtmSqueezeError::from)?;
        let mut d_l: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(len) }.map_err(CudaTtmSqueezeError::from)?;
        let mut d_c: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(len) }.map_err(CudaTtmSqueezeError::from)?;
        unsafe {
            d_h.async_copy_from(high_f32, &self.stream)
                .map_err(CudaTtmSqueezeError::from)?;
            d_l.async_copy_from(low_f32, &self.stream)
                .map_err(CudaTtmSqueezeError::from)?;
            d_c.async_copy_from(close_f32, &self.stream)
                .map_err(CudaTtmSqueezeError::from)?;
        }

        let v_len: Vec<i32> = combos.iter().map(|c| c.length).collect();
        let v_bb: Vec<f32> = combos.iter().map(|c| c.bb_mult).collect();
        let v_kh: Vec<f32> = combos.iter().map(|c| c.kc_high).collect();
        let v_km: Vec<f32> = combos.iter().map(|c| c.kc_mid).collect();
        let v_kl: Vec<f32> = combos.iter().map(|c| c.kc_low).collect();

        let mut d_len: DeviceBuffer<i32> = unsafe { DeviceBuffer::uninitialized(v_len.len()) }
            .map_err(CudaTtmSqueezeError::from)?;
        let mut d_bb: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(v_bb.len()) }
            .map_err(CudaTtmSqueezeError::from)?;
        let mut d_kh: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(v_kh.len()) }
            .map_err(CudaTtmSqueezeError::from)?;
        let mut d_km: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(v_km.len()) }
            .map_err(CudaTtmSqueezeError::from)?;
        let mut d_kl: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(v_kl.len()) }
            .map_err(CudaTtmSqueezeError::from)?;
        unsafe {
            d_len
                .async_copy_from(&v_len, &self.stream)
                .map_err(CudaTtmSqueezeError::from)?;
            d_bb.async_copy_from(&v_bb, &self.stream)
                .map_err(CudaTtmSqueezeError::from)?;
            d_kh.async_copy_from(&v_kh, &self.stream)
                .map_err(CudaTtmSqueezeError::from)?;
            d_km.async_copy_from(&v_km, &self.stream)
                .map_err(CudaTtmSqueezeError::from)?;
            d_kl.async_copy_from(&v_kl, &self.stream)
                .map_err(CudaTtmSqueezeError::from)?;
        }

        let elems = combos.len().checked_mul(len).ok_or_else(|| {
            CudaTtmSqueezeError::InvalidInput("size overflow in output elems".into())
        })?;
        let mut d_mo: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.map_err(CudaTtmSqueezeError::from)?;
        let mut d_sq: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.map_err(CudaTtmSqueezeError::from)?;

        self.launch_batch_kernel(
            &d_h,
            &d_l,
            &d_c,
            &d_len,
            &d_bb,
            &d_kh,
            &d_km,
            &d_kl,
            len,
            combos.len(),
            first_valid,
            combos.iter().map(|c| c.length as usize).max().unwrap_or(0),
            &mut d_mo,
            &mut d_sq,
        )?;

        self.stream
            .synchronize()
            .map_err(CudaTtmSqueezeError::from)?;

        Ok((
            DeviceArrayF32 {
                buf: d_mo,
                rows: combos.len(),
                cols: len,
            },
            DeviceArrayF32 {
                buf: d_sq,
                rows: combos.len(),
                cols: len,
            },
        ))
    }

    fn launch_many_series_kernel(
        &self,
        d_h_tm: &DeviceBuffer<f32>,
        d_l_tm: &DeviceBuffer<f32>,
        d_c_tm: &DeviceBuffer<f32>,

        d_first: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        length: usize,
        bb_mult: f32,
        kh: f32,
        km: f32,
        kl: f32,
        d_mo_tm: &mut DeviceBuffer<f32>,
        d_sq_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTtmSqueezeError> {
        let mut func: Function = self
            .module
            .get_function("ttm_squeeze_many_series_one_param_f32")
            .map_err(|_| CudaTtmSqueezeError::MissingKernelSymbol {
                name: "ttm_squeeze_many_series_one_param_f32",
            })?;

        let smem_bytes = Self::smem_bytes_for_len(length);
        if smem_bytes > self.smem_limit_optin {
            let max_L = self.smem_limit_optin / Self::SHMEM_ELEM_BYTES;
            return Err(CudaTtmSqueezeError::InvalidInput(format!(
                "window length {} requires {} bytes of dynamic shared memory, exceeding device limit {} bytes. Maximum supported length on this GPU is {}.",
                length, smem_bytes, self.smem_limit_optin, max_L
            )));
        }
        let _ = func.set_cache_config(CacheConfig::PreferShared);
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 1,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(1),
        };

        let grid: GridSize = (1u32, cols as u32, 1u32).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut p_h = d_h_tm.as_device_ptr().as_raw();
            let mut p_l = d_l_tm.as_device_ptr().as_raw();
            let mut p_c = d_c_tm.as_device_ptr().as_raw();
            let mut p_fv = d_first.as_device_ptr().as_raw();
            let mut nser = cols as i32;
            let mut slen = rows as i32;
            let mut l_i = length as i32;
            let mut bb = bb_mult as f32;
            let mut khf = kh as f32;
            let mut kmf = km as f32;
            let mut klf = kl as f32;
            let mut p_mo = d_mo_tm.as_device_ptr().as_raw();
            let mut p_sq = d_sq_tm.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 14] = [
                &mut p_h as *mut _ as *mut c_void,
                &mut p_l as *mut _ as *mut c_void,
                &mut p_c as *mut _ as *mut c_void,
                &mut p_fv as *mut _ as *mut c_void,
                &mut nser as *mut _ as *mut c_void,
                &mut slen as *mut _ as *mut c_void,
                &mut l_i as *mut _ as *mut c_void,
                &mut bb as *mut _ as *mut c_void,
                &mut khf as *mut _ as *mut c_void,
                &mut kmf as *mut _ as *mut c_void,
                &mut klf as *mut _ as *mut c_void,
                &mut p_mo as *mut _ as *mut c_void,
                &mut p_sq as *mut _ as *mut c_void,
                std::ptr::null_mut(),
            ];
            self.stream
                .launch(&func, grid, block, smem_bytes as u32, &mut args)
                .map_err(CudaTtmSqueezeError::from)?;
        }
        unsafe {
            (*(self as *const _ as *mut CudaTtmSqueeze)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        Ok(())
    }

    pub fn ttm_squeeze_many_series_one_param_time_major_dev(
        &self,
        high_tm_f32: &[f32],
        low_tm_f32: &[f32],
        close_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        length: usize,
        bb_mult: f32,
        kc_high: f32,
        kc_mid: f32,
        kc_low: f32,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32), CudaTtmSqueezeError> {
        if high_tm_f32.len() != low_tm_f32.len() || low_tm_f32.len() != close_tm_f32.len() {
            return Err(CudaTtmSqueezeError::InvalidInput(
                "inconsistent time-major inputs".into(),
            ));
        }
        if cols == 0 || rows == 0 {
            return Err(CudaTtmSqueezeError::InvalidInput("zero dims".into()));
        }
        let expected_elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaTtmSqueezeError::InvalidInput("size overflow in dims".into()))?;
        if high_tm_f32.len() != expected_elems {
            return Err(CudaTtmSqueezeError::InvalidInput(
                "dims mismatch with buffer length".into(),
            ));
        }

        let mut first_valids: Vec<i32> = vec![0; cols];
        for s in 0..cols {
            let mut fv = -1i32;
            for t in 0..rows {
                let v = close_tm_f32[t * cols + s];
                if !v.is_nan() {
                    fv = t as i32;
                    break;
                }
            }
            if fv < 0 {
                return Err(CudaTtmSqueezeError::InvalidInput(format!(
                    "series {} all NaN",
                    s
                )));
            }
            if (rows as i32 - fv) < (length as i32) {
                return Err(CudaTtmSqueezeError::InvalidInput(
                    "insufficient valid data for length".into(),
                ));
            }
            first_valids[s] = fv;
        }

        let elems = expected_elems;
        let elem = std::mem::size_of::<f32>();
        let in_bytes = elems
            .checked_mul(3)
            .and_then(|v| v.checked_mul(elem))
            .ok_or_else(|| {
                CudaTtmSqueezeError::InvalidInput("size overflow in input bytes".into())
            })?;
        let out_bytes = elems
            .checked_mul(2)
            .and_then(|v| v.checked_mul(elem))
            .ok_or_else(|| {
                CudaTtmSqueezeError::InvalidInput("size overflow in output bytes".into())
            })?;
        let params_bytes = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| {
                CudaTtmSqueezeError::InvalidInput("size overflow in params bytes".into())
            })?;
        let required = in_bytes
            .checked_add(out_bytes)
            .and_then(|v| v.checked_add(params_bytes))
            .ok_or_else(|| {
                CudaTtmSqueezeError::InvalidInput("size overflow in total bytes".into())
            })?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            if let Some((free, _)) = Self::device_mem_info() {
                return Err(CudaTtmSqueezeError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaTtmSqueezeError::InvalidInput(
                    "insufficient device memory".into(),
                ));
            }
        }

        let mut d_h: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.map_err(CudaTtmSqueezeError::from)?;
        let mut d_l: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.map_err(CudaTtmSqueezeError::from)?;
        let mut d_c: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.map_err(CudaTtmSqueezeError::from)?;
        unsafe {
            d_h.async_copy_from(high_tm_f32, &self.stream)
                .map_err(CudaTtmSqueezeError::from)?;
            d_l.async_copy_from(low_tm_f32, &self.stream)
                .map_err(CudaTtmSqueezeError::from)?;
            d_c.async_copy_from(close_tm_f32, &self.stream)
                .map_err(CudaTtmSqueezeError::from)?;
        }
        let mut d_fv: DeviceBuffer<i32> =
            unsafe { DeviceBuffer::uninitialized(cols) }.map_err(CudaTtmSqueezeError::from)?;
        unsafe {
            d_fv.async_copy_from(&first_valids, &self.stream)
                .map_err(CudaTtmSqueezeError::from)?
        };

        let mut d_mo: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.map_err(CudaTtmSqueezeError::from)?;
        let mut d_sq: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.map_err(CudaTtmSqueezeError::from)?;

        self.launch_many_series_kernel(
            &d_h, &d_l, &d_c, &d_fv, cols, rows, length, bb_mult, kc_high, kc_mid, kc_low,
            &mut d_mo, &mut d_sq,
        )?;
        self.stream
            .synchronize()
            .map_err(CudaTtmSqueezeError::from)?;
        Ok((
            DeviceArrayF32 {
                buf: d_mo,
                rows,
                cols,
            },
            DeviceArrayF32 {
                buf: d_sq,
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

    const ONE_SERIES_LEN: usize = 100_000;
    const PARAM_SWEEP: usize = 128;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = 3 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = 2 * ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct TtmBatchDeviceState {
        cuda: CudaTtmSqueeze,
        d_h: DeviceBuffer<f32>,
        d_l: DeviceBuffer<f32>,
        d_c: DeviceBuffer<f32>,
        d_len: DeviceBuffer<i32>,
        d_bb: DeviceBuffer<f32>,
        d_kh: DeviceBuffer<f32>,
        d_km: DeviceBuffer<f32>,
        d_kl: DeviceBuffer<f32>,
        len: usize,
        n_combos: usize,
        first_valid: usize,
        max_len: usize,
        d_mo: DeviceBuffer<f32>,
        d_sq: DeviceBuffer<f32>,
    }
    impl CudaBenchState for TtmBatchDeviceState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_h,
                    &self.d_l,
                    &self.d_c,
                    &self.d_len,
                    &self.d_bb,
                    &self.d_kh,
                    &self.d_km,
                    &self.d_kl,
                    self.len,
                    self.n_combos,
                    self.first_valid,
                    self.max_len,
                    &mut self.d_mo,
                    &mut self.d_sq,
                )
                .expect("ttm_squeeze launch");
            self.cuda.stream.synchronize().expect("ttm_squeeze sync");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaTtmSqueeze::new(0).expect("cuda ttm squeeze");
        let h = gen_series(ONE_SERIES_LEN);
        let mut l = h.clone();
        for v in &mut l {
            *v -= 0.5;
        }
        let mut c = h.clone();
        for v in &mut c {
            *v -= 0.25;
        }
        let sweep = TtmSqueezeBatchRange {
            length: (10, 10 + PARAM_SWEEP - 1, 1),
            bb_mult: (2.0, 2.0, 0.0),
            kc_high: (1.0, 1.0, 0.0),
            kc_mid: (1.5, 1.5, 0.0),
            kc_low: (2.0, 2.0, 0.0),
        };

        let (combos, first_valid, len) =
            CudaTtmSqueeze::prepare_batch_inputs(&h, &l, &c, &sweep).expect("prep inputs");
        let n_combos = combos.len();
        let v_len: Vec<i32> = combos.iter().map(|c| c.length).collect();
        let v_bb: Vec<f32> = combos.iter().map(|c| c.bb_mult).collect();
        let v_kh: Vec<f32> = combos.iter().map(|c| c.kc_high).collect();
        let v_km: Vec<f32> = combos.iter().map(|c| c.kc_mid).collect();
        let v_kl: Vec<f32> = combos.iter().map(|c| c.kc_low).collect();
        let max_len = combos.iter().map(|c| c.length as usize).max().unwrap_or(0);

        let d_h = unsafe { DeviceBuffer::from_slice_async(&h, &cuda.stream) }.expect("d_h H2D");
        let d_l = unsafe { DeviceBuffer::from_slice_async(&l, &cuda.stream) }.expect("d_l H2D");
        let d_c = unsafe { DeviceBuffer::from_slice_async(&c, &cuda.stream) }.expect("d_c H2D");

        let d_len =
            unsafe { DeviceBuffer::from_slice_async(&v_len, &cuda.stream) }.expect("d_len H2D");
        let d_bb =
            unsafe { DeviceBuffer::from_slice_async(&v_bb, &cuda.stream) }.expect("d_bb H2D");
        let d_kh =
            unsafe { DeviceBuffer::from_slice_async(&v_kh, &cuda.stream) }.expect("d_kh H2D");
        let d_km =
            unsafe { DeviceBuffer::from_slice_async(&v_km, &cuda.stream) }.expect("d_km H2D");
        let d_kl =
            unsafe { DeviceBuffer::from_slice_async(&v_kl, &cuda.stream) }.expect("d_kl H2D");

        let elems = n_combos * len;
        let d_mo: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &cuda.stream) }.expect("d_mo");
        let d_sq: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &cuda.stream) }.expect("d_sq");
        cuda.stream.synchronize().expect("ttm_squeeze prep sync");

        Box::new(TtmBatchDeviceState {
            cuda,
            d_h,
            d_l,
            d_c,
            d_len,
            d_bb,
            d_kh,
            d_km,
            d_kl,
            len,
            n_combos,
            first_valid,
            max_len,
            d_mo,
            d_sq,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "ttm_squeeze",
            "one_series_many_params",
            "ttm_squeeze_cuda_batch_dev",
            "100k_x_128",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}
