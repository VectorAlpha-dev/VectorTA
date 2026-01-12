#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::emd::{EmdBatchRange, EmdParams};
use cust::context::{CacheConfig, Context, SharedMemoryConfig};
use cust::device::{Device, DeviceAttribute};
use cust::error::CudaError;
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::sys::{
    cuDeviceGetAttribute, cuFuncSetAttribute, CUdevice_attribute, CUfunction_attribute,
};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;

#[derive(Debug)]
pub enum CudaEmdError {
    Cuda(CudaError),
    InvalidInput(String),
    MissingKernelSymbol {
        name: &'static str,
    },
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
    InvalidPolicy(&'static str),
    DeviceMismatch {
        buf: u32,
        current: u32,
    },
    NotImplemented,
}
impl fmt::Display for CudaEmdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaEmdError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaEmdError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
            CudaEmdError::MissingKernelSymbol { name } => {
                write!(f, "Missing kernel symbol: {}", name)
            }
            CudaEmdError::OutOfMemory {
                required,
                free,
                headroom,
            } => write!(
                f,
                "Out of memory on device: required={}B, free={}B, headroom={}B",
                required, free, headroom
            ),
            CudaEmdError::LaunchConfigTooLarge {
                gx,
                gy,
                gz,
                bx,
                by,
                bz,
            } => write!(
                f,
                "Launch config too large (grid=({gx},{gy},{gz}), block=({bx},{by},{bz}))"
            ),
            CudaEmdError::InvalidPolicy(p) => write!(f, "Invalid policy: {}", p),
            CudaEmdError::DeviceMismatch { buf, current } => {
                write!(f, "Device mismatch: buffer on {}, current {}", buf, current)
            }
            CudaEmdError::NotImplemented => write!(f, "Not implemented"),
        }
    }
}
impl std::error::Error for CudaEmdError {}

impl From<CudaError> for CudaEmdError {
    #[inline]
    fn from(e: CudaError) -> Self {
        CudaEmdError::Cuda(e)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
}
impl Default for BatchKernelPolicy {
    fn default() -> Self {
        BatchKernelPolicy::Auto
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}
impl Default for ManySeriesKernelPolicy {
    fn default() -> Self {
        ManySeriesKernelPolicy::Auto
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaEmdPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

pub struct DeviceArrayF32Triple {
    pub upper: DeviceArrayF32,
    pub middle: DeviceArrayF32,
    pub lower: DeviceArrayF32,
}
impl DeviceArrayF32Triple {
    #[inline]
    pub fn rows(&self) -> usize {
        self.upper.rows
    }
    #[inline]
    pub fn cols(&self) -> usize {
        self.upper.cols
    }
}

pub struct CudaEmdBatchResult {
    pub outputs: DeviceArrayF32Triple,
    pub combos: Vec<EmdParams>,
}

pub struct CudaEmd {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaEmdPolicy,
}

fn opt_in_dynamic_smem(func: &Function, bytes: u32) -> Result<(), CudaEmdError> {
    let res = unsafe {
        cuFuncSetAttribute(
            func.to_raw(),
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            bytes as i32,
        )
    };
    if res != cust::sys::CUresult::CUDA_SUCCESS {
        return Err(CudaEmdError::InvalidInput(format!(
            "cuFuncSetAttribute(MAX_DYNAMIC_SHARED) failed: {:?}",
            res
        )));
    }
    Ok(())
}

fn prefer_shared(func: &mut Function) -> Result<(), CudaEmdError> {
    func.set_cache_config(CacheConfig::PreferShared)
        .map_err(CudaEmdError::Cuda)?;
    func.set_shared_memory_config(SharedMemoryConfig::FourByteBankSize)
        .map_err(CudaEmdError::Cuda)
}

const PER_UP_LOW: usize = 50;
const RINGS_PER_BLOCK: usize = 2;

#[inline]
fn smem_for(block_x: u32) -> usize {
    RINGS_PER_BLOCK * PER_UP_LOW * (block_x as usize) * core::mem::size_of::<f32>()
}

#[inline]
fn query_optin_smem_limit(device: Device) -> usize {
    let default = device
        .get_attribute(DeviceAttribute::MaxSharedMemoryPerBlock)
        .unwrap_or(48 * 1024) as usize;
    let mut optin = default as i32;
    unsafe {
        let _ = cuDeviceGetAttribute(
            &mut optin as *mut _,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            device.as_raw(),
        );
    }
    optin.max(default as i32) as usize
}

#[inline]
fn clamp_block_x_for_smem(device: Device, requested: u32) -> u32 {
    let limit = query_optin_smem_limit(device);
    let mut bx = requested.max(64).min(1024);
    while smem_for(bx) > limit && bx > 32 {
        bx -= 32;
    }
    let max_tpb = device
        .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
        .unwrap_or(1024) as u32;
    bx.min(max_tpb).max(32)
}

impl CudaEmd {
    pub fn new(device_id: usize) -> Result<Self, CudaEmdError> {
        cust::init(CudaFlags::empty()).map_err(CudaEmdError::Cuda)?;
        let device = Device::get_device(device_id as u32).map_err(CudaEmdError::Cuda)?;
        let context = Arc::new(Context::new(device).map_err(CudaEmdError::Cuda)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/emd_kernel.ptx"));

        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O4),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(CudaEmdError::Cuda)?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).map_err(CudaEmdError::Cuda)?;

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy: CudaEmdPolicy::default(),
        })
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
    pub fn stream_handle_usize(&self) -> usize {
        self.stream.as_inner() as usize
    }

    #[inline]
    pub fn set_policy(&mut self, policy: CudaEmdPolicy) {
        self.policy = policy;
    }
    #[inline]
    pub fn policy(&self) -> &CudaEmdPolicy {
        &self.policy
    }
    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaEmdError> {
        self.stream.synchronize().map_err(CudaEmdError::Cuda)
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
    fn ensure_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaEmdError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            if required_bytes.saturating_add(headroom_bytes) <= free {
                Ok(())
            } else {
                Err(CudaEmdError::OutOfMemory {
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
    fn validate_launch(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
    ) -> Result<(), CudaEmdError> {
        let dev = Device::get_device(self.device_id).map_err(CudaEmdError::Cuda)?;
        let max_bx = dev
            .get_attribute(DeviceAttribute::MaxBlockDimX)
            .map_err(CudaEmdError::Cuda)? as u32;
        let max_gx = dev
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .map_err(CudaEmdError::Cuda)? as u32;
        if block.0 == 0 || block.0 > max_bx || grid.0 == 0 || grid.0 > max_gx {
            return Err(CudaEmdError::LaunchConfigTooLarge {
                gx: grid.0,
                gy: grid.1,
                gz: grid.2,
                bx: block.0,
                by: block.1,
                bz: block.2,
            });
        }
        Ok(())
    }

    fn expand_combos(range: &EmdBatchRange) -> Result<Vec<EmdParams>, CudaEmdError> {
        fn axis_usize(t: (usize, usize, usize)) -> Vec<usize> {
            let (start, end, step) = t;
            if step == 0 || start == end {
                return vec![start];
            }
            let mut v = Vec::new();
            if start < end {
                let mut cur = start;
                while cur <= end {
                    v.push(cur);
                    match cur.checked_add(step) {
                        Some(n) => cur = n,
                        None => break,
                    }
                }
            } else {
                let mut cur = start;
                while cur >= end {
                    v.push(cur);
                    match cur.checked_sub(step) {
                        Some(n) => cur = n,
                        None => break,
                    }
                }
            }
            v
        }
        fn axis_f64(t: (f64, f64, f64)) -> Vec<f64> {
            let (start, end, step) = t;
            if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
                return vec![start];
            }
            let mut v = Vec::new();
            if start < end {
                let mut x = start;
                while x <= end + 1e-12 {
                    v.push(x);
                    x += step;
                    if !x.is_finite() {
                        break;
                    }
                }
            } else {
                let mut x = start;
                let st = step.abs();
                while x >= end - 1e-12 {
                    v.push(x);
                    x -= st;
                    if !x.is_finite() {
                        break;
                    }
                }
            }
            v
        }
        let periods = axis_usize(range.period);
        let deltas = axis_f64(range.delta);
        let fracs = axis_f64(range.fraction);
        if periods.is_empty() || deltas.is_empty() || fracs.is_empty() {
            return Err(CudaEmdError::InvalidInput(
                "empty parameter expansion".into(),
            ));
        }
        let cap = periods
            .len()
            .checked_mul(deltas.len())
            .and_then(|v| v.checked_mul(fracs.len()))
            .ok_or_else(|| CudaEmdError::InvalidInput("parameter grid size overflow".into()))?;
        let mut out = Vec::with_capacity(cap);
        for &p in &periods {
            for &d in &deltas {
                for &f in &fracs {
                    out.push(EmdParams {
                        period: Some(p),
                        delta: Some(d),
                        fraction: Some(f),
                    });
                }
            }
        }
        Ok(out)
    }

    pub fn emd_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
        sweep: &EmdBatchRange,
    ) -> Result<CudaEmdBatchResult, CudaEmdError> {
        if high.is_empty() || high.len() != low.len() {
            return Err(CudaEmdError::InvalidInput(
                "high/low must be non-empty and same length".into(),
            ));
        }
        let len = high.len();
        let first_valid = (0..len)
            .find(|&i| high[i].is_finite() && low[i].is_finite())
            .ok_or_else(|| CudaEmdError::InvalidInput("all values are NaN".into()))?;

        let combos = Self::expand_combos(sweep)?;
        if combos.is_empty() {
            return Err(CudaEmdError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        let max_p = combos.iter().map(|c| c.period.unwrap_or(20)).max().unwrap();

        if len - first_valid < (2 * max_p).max(50) {
            return Err(CudaEmdError::InvalidInput(
                "not enough valid data for warmup".into(),
            ));
        }

        let mut prices = vec![f32::NAN; len];
        for i in first_valid..len {
            prices[i] = 0.5f32 * (high[i] + low[i]);
        }

        let n = combos.len();
        let mut periods_i32 = Vec::with_capacity(n);
        let mut deltas_f32 = Vec::with_capacity(n);
        let mut fracs_f32 = Vec::with_capacity(n);
        for c in &combos {
            periods_i32.push(c.period.unwrap_or(20) as i32);
            deltas_f32.push(c.delta.unwrap_or(0.5) as f32);
            fracs_f32.push(c.fraction.unwrap_or(0.1) as f32);
        }

        let sz_f32 = std::mem::size_of::<f32>();
        let sz_i32 = std::mem::size_of::<i32>();
        let in_bytes = len
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaEmdError::InvalidInput("byte size overflow".into()))?;
        let params_each = sz_i32
            .checked_add(2 * sz_f32)
            .ok_or_else(|| CudaEmdError::InvalidInput("byte size overflow".into()))?;
        let params_bytes = n
            .checked_mul(params_each)
            .ok_or_else(|| CudaEmdError::InvalidInput("byte size overflow".into()))?;
        let plane_elems = n
            .checked_mul(len)
            .ok_or_else(|| CudaEmdError::InvalidInput("byte size overflow".into()))?;
        let out_elems = plane_elems
            .checked_mul(3)
            .ok_or_else(|| CudaEmdError::InvalidInput("byte size overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaEmdError::InvalidInput("byte size overflow".into()))?;
        let required = in_bytes
            .checked_add(params_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaEmdError::InvalidInput("byte size overflow".into()))?;
        Self::ensure_fit(required, 64 * 1024 * 1024)?;

        let h_prices = LockedBuffer::from_slice(&prices).map_err(CudaEmdError::Cuda)?;
        let h_p = LockedBuffer::from_slice(&periods_i32).map_err(CudaEmdError::Cuda)?;
        let h_d = LockedBuffer::from_slice(&deltas_f32).map_err(CudaEmdError::Cuda)?;
        let h_f = LockedBuffer::from_slice(&fracs_f32).map_err(CudaEmdError::Cuda)?;

        let mut d_prices: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(len) }.map_err(CudaEmdError::Cuda)?;
        let mut d_p: DeviceBuffer<i32> =
            unsafe { DeviceBuffer::uninitialized(n) }.map_err(CudaEmdError::Cuda)?;
        let mut d_d: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n) }.map_err(CudaEmdError::Cuda)?;
        let mut d_f: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n) }.map_err(CudaEmdError::Cuda)?;
        unsafe {
            d_prices
                .async_copy_from(h_prices.as_slice(), &self.stream)
                .map_err(CudaEmdError::Cuda)?;
            d_p.async_copy_from(h_p.as_slice(), &self.stream)
                .map_err(CudaEmdError::Cuda)?;
            d_d.async_copy_from(h_d.as_slice(), &self.stream)
                .map_err(CudaEmdError::Cuda)?;
            d_f.async_copy_from(h_f.as_slice(), &self.stream)
                .map_err(CudaEmdError::Cuda)?;
        }

        let elems = plane_elems;
        let mut d_ub: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.map_err(CudaEmdError::Cuda)?;
        let mut d_mb: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.map_err(CudaEmdError::Cuda)?;
        let mut d_lb: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.map_err(CudaEmdError::Cuda)?;

        let req_block = match self.policy.batch {
            BatchKernelPolicy::Auto => 128,
            BatchKernelPolicy::Plain { block_x } => block_x,
        } as u32;
        let device = Device::get_device(self.device_id).map_err(CudaEmdError::Cuda)?;
        let block_x = clamp_block_x_for_smem(device, req_block);
        let grid_x = ((n as u32) + block_x - 1) / block_x;

        unsafe {
            let mut func = self.module.get_function("emd_batch_f32").map_err(|_| {
                CudaEmdError::MissingKernelSymbol {
                    name: "emd_batch_f32",
                }
            })?;

            let dyn_smem_bytes: u32 = smem_for(block_x) as u32;
            opt_in_dynamic_smem(&func, dyn_smem_bytes)?;
            prefer_shared(&mut func)?;

            unsafe {
                let _ = cuFuncSetAttribute(
                    func.to_raw(),
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                    100,
                );
            }

            let mut p_prices = d_prices.as_device_ptr().as_raw();
            let mut p_p = d_p.as_device_ptr().as_raw();
            let mut p_d = d_d.as_device_ptr().as_raw();
            let mut p_f = d_f.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut n_i = n as i32;
            let mut fv_i = first_valid as i32;
            let mut p_ub = d_ub.as_device_ptr().as_raw();
            let mut p_mb = d_mb.as_device_ptr().as_raw();
            let mut p_lb = d_lb.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_prices as *mut _ as *mut c_void,
                &mut p_p as *mut _ as *mut c_void,
                &mut p_d as *mut _ as *mut c_void,
                &mut p_f as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut fv_i as *mut _ as *mut c_void,
                &mut p_ub as *mut _ as *mut c_void,
                &mut p_mb as *mut _ as *mut c_void,
                &mut p_lb as *mut _ as *mut c_void,
            ];
            let grid: GridSize = (grid_x, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            self.validate_launch((grid_x, 1, 1), (block_x, 1, 1))?;
            self.stream
                .launch(&func, grid, block, dyn_smem_bytes as u32, args)
                .map_err(CudaEmdError::Cuda)?;
        }

        self.stream.synchronize().map_err(CudaEmdError::Cuda)?;

        let outputs = DeviceArrayF32Triple {
            upper: DeviceArrayF32 {
                buf: d_ub,
                rows: n,
                cols: len,
            },
            middle: DeviceArrayF32 {
                buf: d_mb,
                rows: n,
                cols: len,
            },
            lower: DeviceArrayF32 {
                buf: d_lb,
                rows: n,
                cols: len,
            },
        };
        Ok(CudaEmdBatchResult { outputs, combos })
    }

    pub fn emd_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &EmdParams,
        first_valids: &[i32],
    ) -> Result<DeviceArrayF32Triple, CudaEmdError> {
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaEmdError::InvalidInput("time-major shape overflow".into()))?;
        if cols == 0 || rows == 0 || data_tm_f32.len() != expected {
            return Err(CudaEmdError::InvalidInput(
                "invalid time-major input shape".into(),
            ));
        }
        if first_valids.len() != cols {
            return Err(CudaEmdError::InvalidInput(
                "first_valids length must equal cols".into(),
            ));
        }
        let period = params.period.unwrap_or(20) as i32;
        let delta = params.delta.unwrap_or(0.5) as f32;
        let fraction = params.fraction.unwrap_or(0.1) as f32;

        let sz_f32 = std::mem::size_of::<f32>();
        let sz_i32 = std::mem::size_of::<i32>();
        let input_bytes = data_tm_f32
            .len()
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaEmdError::InvalidInput("byte size overflow".into()))?;
        let fv_bytes = first_valids
            .len()
            .checked_mul(sz_i32)
            .ok_or_else(|| CudaEmdError::InvalidInput("byte size overflow".into()))?;
        let out_bytes = data_tm_f32
            .len()
            .checked_mul(3)
            .and_then(|v| v.checked_mul(sz_f32))
            .ok_or_else(|| CudaEmdError::InvalidInput("byte size overflow".into()))?;
        let bytes = input_bytes
            .checked_add(fv_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaEmdError::InvalidInput("byte size overflow".into()))?;
        Self::ensure_fit(bytes, 64 * 1024 * 1024)?;

        let h_prices = LockedBuffer::from_slice(data_tm_f32).map_err(CudaEmdError::Cuda)?;
        let h_fv = LockedBuffer::from_slice(first_valids).map_err(CudaEmdError::Cuda)?;
        let mut d_prices: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(expected) }.map_err(CudaEmdError::Cuda)?;
        let mut d_fv: DeviceBuffer<i32> =
            unsafe { DeviceBuffer::uninitialized(cols) }.map_err(CudaEmdError::Cuda)?;
        unsafe {
            d_prices
                .async_copy_from(h_prices.as_slice(), &self.stream)
                .map_err(CudaEmdError::Cuda)?;
            d_fv.async_copy_from(h_fv.as_slice(), &self.stream)
                .map_err(CudaEmdError::Cuda)?;
        }
        let mut d_ub: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(expected) }.map_err(CudaEmdError::Cuda)?;
        let mut d_mb: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(expected) }.map_err(CudaEmdError::Cuda)?;
        let mut d_lb: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(expected) }.map_err(CudaEmdError::Cuda)?;

        let req_block = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 128,
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
        } as u32;
        let device = Device::get_device(self.device_id).map_err(CudaEmdError::Cuda)?;
        let block_x = clamp_block_x_for_smem(device, req_block);
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        unsafe {
            let mut func = self
                .module
                .get_function("emd_many_series_one_param_time_major_f32")
                .map_err(|_| CudaEmdError::MissingKernelSymbol {
                    name: "emd_many_series_one_param_time_major_f32",
                })?;
            let dyn_smem_bytes: u32 = smem_for(block_x) as u32;
            opt_in_dynamic_smem(&func, dyn_smem_bytes)?;
            prefer_shared(&mut func)?;
            unsafe {
                let _ = cuFuncSetAttribute(
                    func.to_raw(),
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                    100,
                );
            }
            let mut p_prices = d_prices.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut period_i = period as i32;
            let mut delta_f = delta;
            let mut frac_f = fraction;
            let mut p_fv = d_fv.as_device_ptr().as_raw();
            let mut p_ub = d_ub.as_device_ptr().as_raw();
            let mut p_mb = d_mb.as_device_ptr().as_raw();
            let mut p_lb = d_lb.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_prices as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut delta_f as *mut _ as *mut c_void,
                &mut frac_f as *mut _ as *mut c_void,
                &mut p_fv as *mut _ as *mut c_void,
                &mut p_ub as *mut _ as *mut c_void,
                &mut p_mb as *mut _ as *mut c_void,
                &mut p_lb as *mut _ as *mut c_void,
            ];
            let grid: GridSize = (grid_x, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            self.validate_launch((grid_x, 1, 1), (block_x, 1, 1))?;
            self.stream
                .launch(&func, grid, block, dyn_smem_bytes as u32, args)
                .map_err(CudaEmdError::Cuda)?;
        }
        self.stream.synchronize().map_err(CudaEmdError::Cuda)?;
        Ok(DeviceArrayF32Triple {
            upper: DeviceArrayF32 {
                buf: d_ub,
                rows,
                cols,
            },
            middle: DeviceArrayF32 {
                buf: d_mb,
                rows,
                cols,
            },
            lower: DeviceArrayF32 {
                buf: d_lb,
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
                "emd",
                "batch_dev",
                "emd_cuda_batch_dev",
                "60k_x_27combos",
                prep_emd_batch_box,
            )
            .with_inner_iters(8),
            CudaBenchScenario::new(
                "emd",
                "many_series_one_param",
                "emd_cuda_many_series_one_param_dev",
                "128x120k",
                prep_emd_many_series_box,
            )
            .with_inner_iters(4),
        ]
    }

    struct EmdBatchDeviceState {
        cuda: CudaEmd,
        d_prices: DeviceBuffer<f32>,
        d_p: DeviceBuffer<i32>,
        d_d: DeviceBuffer<f32>,
        d_f: DeviceBuffer<f32>,
        d_ub: DeviceBuffer<f32>,
        d_mb: DeviceBuffer<f32>,
        d_lb: DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        n: usize,
        grid_x: u32,
        block_x: u32,
        dyn_smem_bytes: u32,
    }
    impl CudaBenchState for EmdBatchDeviceState {
        fn launch(&mut self) {
            unsafe {
                let mut func = self
                    .cuda
                    .module
                    .get_function("emd_batch_f32")
                    .expect("emd_batch_f32");
                opt_in_dynamic_smem(&func, self.dyn_smem_bytes).expect("opt-in smem");
                prefer_shared(&mut func).expect("prefer_shared");
                let _ = cuFuncSetAttribute(
                    func.to_raw(),
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                    100,
                );

                let mut p_prices = self.d_prices.as_device_ptr().as_raw();
                let mut p_p = self.d_p.as_device_ptr().as_raw();
                let mut p_d = self.d_d.as_device_ptr().as_raw();
                let mut p_f = self.d_f.as_device_ptr().as_raw();
                let mut len_i = self.len as i32;
                let mut n_i = self.n as i32;
                let mut fv_i = self.first_valid as i32;
                let mut p_ub = self.d_ub.as_device_ptr().as_raw();
                let mut p_mb = self.d_mb.as_device_ptr().as_raw();
                let mut p_lb = self.d_lb.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut p_prices as *mut _ as *mut c_void,
                    &mut p_p as *mut _ as *mut c_void,
                    &mut p_d as *mut _ as *mut c_void,
                    &mut p_f as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut n_i as *mut _ as *mut c_void,
                    &mut fv_i as *mut _ as *mut c_void,
                    &mut p_ub as *mut _ as *mut c_void,
                    &mut p_mb as *mut _ as *mut c_void,
                    &mut p_lb as *mut _ as *mut c_void,
                ];
                self.cuda
                    .validate_launch((self.grid_x, 1, 1), (self.block_x, 1, 1))
                    .expect("launch dims");
                self.cuda
                    .stream
                    .launch(
                        &func,
                        (self.grid_x, 1, 1),
                        (self.block_x, 1, 1),
                        self.dyn_smem_bytes,
                        args,
                    )
                    .expect("emd batch launch");
            }
            self.cuda.stream.synchronize().expect("emd batch sync");
        }
    }
    fn prep_emd_batch_box() -> Box<dyn CudaBenchState> {
        let cuda = CudaEmd::new(0).expect("cuda emd");

        let len = 60_000usize;
        let first_valid = 2usize;
        let mut prices = vec![f32::NAN; len];
        for i in first_valid..len {
            let x = i as f32;
            prices[i] = (x * 0.001).sin() + 0.0002 * x;
        }
        let sweep = EmdBatchRange {
            period: (8, 20, 4),
            delta: (0.3, 0.7, 0.2),
            fraction: (0.05, 0.15, 0.05),
        };
        let combos = CudaEmd::expand_combos(&sweep).expect("expand_combos");
        let n = combos.len();
        let mut periods_i32 = Vec::with_capacity(n);
        let mut deltas_f32 = Vec::with_capacity(n);
        let mut fracs_f32 = Vec::with_capacity(n);
        for c in &combos {
            periods_i32.push(c.period.unwrap_or(20) as i32);
            deltas_f32.push(c.delta.unwrap_or(0.5) as f32);
            fracs_f32.push(c.fraction.unwrap_or(0.1) as f32);
        }

        let d_prices = DeviceBuffer::from_slice(&prices).expect("d_prices");
        let d_p = DeviceBuffer::from_slice(&periods_i32).expect("d_p");
        let d_d = DeviceBuffer::from_slice(&deltas_f32).expect("d_d");
        let d_f = DeviceBuffer::from_slice(&fracs_f32).expect("d_f");
        let elems = n * len;
        let d_ub: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_ub");
        let d_mb: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_mb");
        let d_lb: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_lb");

        let req_block = match cuda.policy.batch {
            BatchKernelPolicy::Auto => 128,
            BatchKernelPolicy::Plain { block_x } => block_x,
        } as u32;
        let device = Device::get_device(cuda.device_id).expect("device");
        let block_x = clamp_block_x_for_smem(device, req_block);
        let grid_x = ((n as u32) + block_x - 1) / block_x;
        let dyn_smem_bytes = smem_for(block_x) as u32;
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(EmdBatchDeviceState {
            cuda,
            d_prices,
            d_p,
            d_d,
            d_f,
            d_ub,
            d_mb,
            d_lb,
            len,
            first_valid,
            n,
            grid_x,
            block_x,
            dyn_smem_bytes,
        })
    }

    struct EmdManyDeviceState {
        cuda: CudaEmd,
        d_prices: DeviceBuffer<f32>,
        d_fv: DeviceBuffer<i32>,
        d_ub: DeviceBuffer<f32>,
        d_mb: DeviceBuffer<f32>,
        d_lb: DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        period: i32,
        delta: f32,
        fraction: f32,
        grid_x: u32,
        block_x: u32,
        dyn_smem_bytes: u32,
    }
    impl CudaBenchState for EmdManyDeviceState {
        fn launch(&mut self) {
            unsafe {
                let mut func = self
                    .cuda
                    .module
                    .get_function("emd_many_series_one_param_time_major_f32")
                    .expect("emd_many_series_one_param_time_major_f32");
                opt_in_dynamic_smem(&func, self.dyn_smem_bytes).expect("opt-in smem");
                prefer_shared(&mut func).expect("prefer_shared");
                let _ = cuFuncSetAttribute(
                    func.to_raw(),
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                    100,
                );

                let mut p_prices = self.d_prices.as_device_ptr().as_raw();
                let mut cols_i = self.cols as i32;
                let mut rows_i = self.rows as i32;
                let mut period_i = self.period;
                let mut delta_f = self.delta;
                let mut frac_f = self.fraction;
                let mut p_fv = self.d_fv.as_device_ptr().as_raw();
                let mut p_ub = self.d_ub.as_device_ptr().as_raw();
                let mut p_mb = self.d_mb.as_device_ptr().as_raw();
                let mut p_lb = self.d_lb.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut p_prices as *mut _ as *mut c_void,
                    &mut cols_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut period_i as *mut _ as *mut c_void,
                    &mut delta_f as *mut _ as *mut c_void,
                    &mut frac_f as *mut _ as *mut c_void,
                    &mut p_fv as *mut _ as *mut c_void,
                    &mut p_ub as *mut _ as *mut c_void,
                    &mut p_mb as *mut _ as *mut c_void,
                    &mut p_lb as *mut _ as *mut c_void,
                ];
                self.cuda
                    .validate_launch((self.grid_x, 1, 1), (self.block_x, 1, 1))
                    .expect("launch dims");
                self.cuda
                    .stream
                    .launch(
                        &func,
                        (self.grid_x, 1, 1),
                        (self.block_x, 1, 1),
                        self.dyn_smem_bytes,
                        args,
                    )
                    .expect("emd many launch");
            }
            self.cuda.stream.synchronize().expect("emd many sync");
        }
    }
    fn prep_emd_many_series_box() -> Box<dyn CudaBenchState> {
        let cuda = CudaEmd::new(0).expect("cuda emd");
        let cols = 128usize;
        let rows = 120_000usize;
        let mut data_tm = vec![f32::NAN; cols * rows];
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            first_valids[s] = 2;
            for t in 2..rows {
                let x = (t as f32) + 0.1 * (s as f32);
                data_tm[t * cols + s] = (x * 0.0008).sin() + 0.0001 * x;
            }
        }
        let params = EmdParams {
            period: Some(18),
            delta: Some(0.5),
            fraction: Some(0.1),
        };
        let period = params.period.unwrap_or(20) as i32;
        let delta = params.delta.unwrap_or(0.5) as f32;
        let fraction = params.fraction.unwrap_or(0.1) as f32;

        let d_prices = DeviceBuffer::from_slice(&data_tm).expect("d_prices_tm");
        let d_fv = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let expected = cols * rows;
        let d_ub: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(expected) }.expect("d_ub");
        let d_mb: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(expected) }.expect("d_mb");
        let d_lb: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(expected) }.expect("d_lb");

        let req_block = match cuda.policy.many_series {
            ManySeriesKernelPolicy::Auto => 128,
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
        } as u32;
        let device = Device::get_device(cuda.device_id).expect("device");
        let block_x = clamp_block_x_for_smem(device, req_block);
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let dyn_smem_bytes = smem_for(block_x) as u32;
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(EmdManyDeviceState {
            cuda,
            d_prices,
            d_fv,
            d_ub,
            d_mb,
            d_lb,
            cols,
            rows,
            period,
            delta,
            fraction,
            grid_x,
            block_x,
            dyn_smem_bytes,
        })
    }
}
