//! CUDA wrapper for the Empirical Mode Decomposition (EMD) indicator.
//!
//! Parity with ALMA wrapper conventions:
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/emd_kernel.ptx"))
//! - NON_BLOCKING stream
//! - Simple policy enums and introspection hooks
//! - VRAM estimation and grid.y chunking (<= 65_535 rows)
//!
//! Math category: Recurrence/IIR. We parallelize across parameter combinations
//! (batch) or across independent series (many-series). Warmup/NaN semantics
//! match the scalar path in `indicators::emd`:
//! - upper/lower warmup = first_valid + 50 - 1
//! - middle warmup      = first_valid + 2*period - 1

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::emd::{EmdBatchRange, EmdParams};
use cust::context::{CacheConfig, Context, SharedMemoryConfig};
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use cust::device::{Device, DeviceAttribute};
use cust::sys::{
    cuDeviceGetAttribute, cuFuncSetAttribute, CUdevice_attribute, CUfunction_attribute,
};

#[derive(Debug)]
pub enum CudaEmdError {
    Cuda(String),
    InvalidInput(String),
}
impl fmt::Display for CudaEmdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaEmdError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaEmdError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl std::error::Error for CudaEmdError {}

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
    _context: Context,
    device_id: u32,
    policy: CudaEmdPolicy,
}

// ----- Helpers: dynamic shared memory attribute and cache preference -----
fn opt_in_dynamic_smem(func: &Function, bytes: u32) -> Result<(), CudaEmdError> {
    let res = unsafe {
        cuFuncSetAttribute(
            func.to_raw(),
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            bytes as i32,
        )
    };
    if res != cust::sys::CUresult::CUDA_SUCCESS {
        return Err(CudaEmdError::Cuda(format!(
            "cuFuncSetAttribute(MAX_DYNAMIC_SHARED) failed: {:?}",
            res
        )));
    }
    Ok(())
}

fn prefer_shared(func: &mut Function) -> Result<(), CudaEmdError> {
    func.set_cache_config(CacheConfig::PreferShared)
        .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
    func.set_shared_memory_config(SharedMemoryConfig::FourByteBankSize)
        .map_err(|e| CudaEmdError::Cuda(e.to_string()))
}

// ---- Shared-memory sizing and device limits ----
const PER_UP_LOW: usize = 50;
const RINGS_PER_BLOCK: usize = 2; // ub + lb

#[inline]
fn smem_for(block_x: u32) -> usize {
    RINGS_PER_BLOCK * PER_UP_LOW * (block_x as usize) * core::mem::size_of::<f32>()
}

#[inline]
fn query_optin_smem_limit(device: Device) -> usize {
    // Fallback to legacy per-block limit first
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
        bx -= 32; // step down by a warp
    }
    let max_tpb = device
        .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
        .unwrap_or(1024) as u32;
    bx.min(max_tpb).max(32)
}

impl CudaEmd {
    pub fn new(device_id: usize) -> Result<Self, CudaEmdError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaEmdError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/emd_kernel.ptx"));
        // Prefer most aggressive JIT optimization level explicitly
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O4),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaEmdPolicy::default(),
        })
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
        self.stream
            .synchronize()
            .map_err(|e| CudaEmdError::Cuda(e.to_string()))
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
        if let Ok((free, _total)) = mem_get_info() {
            bytes.saturating_add(headroom) <= free
        } else {
            true
        }
    }

    // Expand the batch range into concrete parameter combos
    fn expand_combos(range: &EmdBatchRange) -> Vec<EmdParams> {
        fn axis_usize(t: (usize, usize, usize)) -> Vec<usize> {
            let (start, end, step) = t;
            if step == 0 || start == end {
                return vec![start];
            }
            (start..=end).step_by(step).collect()
        }
        fn axis_f64(t: (f64, f64, f64)) -> Vec<f64> {
            let (start, end, step) = t;
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
        let periods = axis_usize(range.period);
        let deltas = axis_f64(range.delta);
        let fracs = axis_f64(range.fraction);
        let mut out = Vec::with_capacity(periods.len() * deltas.len() * fracs.len());
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
        out
    }

    // -------- Batch: one series × many params --------
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

        let combos = Self::expand_combos(sweep);
        if combos.is_empty() {
            return Err(CudaEmdError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        let max_p = combos.iter().map(|c| c.period.unwrap_or(20)).max().unwrap();
        // Basic feasibility guard: ensure tail can cover the longest warmup
        if len - first_valid < (2 * max_p).max(50) {
            return Err(CudaEmdError::InvalidInput(
                "not enough valid data for warmup".into(),
            ));
        }

        // Host precompute: midpoint prices
        let mut prices = vec![f32::NAN; len];
        for i in first_valid..len {
            prices[i] = 0.5f32 * (high[i] + low[i]);
        }

        // Gather params
        let n = combos.len();
        let mut periods_i32 = Vec::with_capacity(n);
        let mut deltas_f32 = Vec::with_capacity(n);
        let mut fracs_f32 = Vec::with_capacity(n);
        for c in &combos {
            periods_i32.push(c.period.unwrap_or(20) as i32);
            deltas_f32.push(c.delta.unwrap_or(0.5) as f32);
            fracs_f32.push(c.fraction.unwrap_or(0.1) as f32);
        }

        // VRAM estimate (inputs + params + outputs(3 planes))
        let in_bytes = len * std::mem::size_of::<f32>();
        let params_bytes = n * (std::mem::size_of::<i32>() + 2 * std::mem::size_of::<f32>());
        let out_bytes = 3 * n * len * std::mem::size_of::<f32>();
        // Scratch rings: sp/sv (50 each) + bp (2*max_period)
        let mut ring_stride_mid = 2 * max_p;
        let ring_bytes = n * (50 + 50 + ring_stride_mid) * std::mem::size_of::<f32>();
        let required = in_bytes + params_bytes + out_bytes + ring_bytes;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaEmdError::InvalidInput(
                "insufficient device memory for emd batch".into(),
            ));
        }

        // H2D (pinned host + async copies on our NON_BLOCKING stream)
        let h_prices = LockedBuffer::from_slice(&prices)
            .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
        let h_p = LockedBuffer::from_slice(&periods_i32)
            .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
        let h_d = LockedBuffer::from_slice(&deltas_f32)
            .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
        let h_f = LockedBuffer::from_slice(&fracs_f32)
            .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;

        let mut d_prices: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }
            .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
        let mut d_p: DeviceBuffer<i32> = unsafe { DeviceBuffer::uninitialized(n) }
            .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
        let mut d_d: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n) }
            .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
        let mut d_f: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n) }
            .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
        unsafe {
            d_prices
                .async_copy_from(h_prices.as_slice(), &self.stream)
                .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
            d_p
                .async_copy_from(h_p.as_slice(), &self.stream)
                .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
            d_d
                .async_copy_from(h_d.as_slice(), &self.stream)
                .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
            d_f
                .async_copy_from(h_f.as_slice(), &self.stream)
                .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
        }

        let elems = n * len;
        let mut d_ub: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
        let mut d_mb: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
        let mut d_lb: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;

        // Allocate zero-initialized ring buffers on device (no host copy)
        let mut d_sp_ring: DeviceBuffer<f32> = DeviceBuffer::zeroed(n * 50)
            .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
        let mut d_sv_ring: DeviceBuffer<f32> = DeviceBuffer::zeroed(n * 50)
            .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
        let mut d_bp_ring: DeviceBuffer<f32> = DeviceBuffer::zeroed(n * ring_stride_mid)
            .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;

        // Chunk grid.y to <= 65_535 if needed (rows == combos). Use x dimension like other wrappers.
        // Kernel uses dynamic shared memory for 2*50 rings; choose safe block size
        let req_block = match self.policy.batch {
            BatchKernelPolicy::Auto => 128,
            BatchKernelPolicy::Plain { block_x } => block_x,
        } as u32;
        let device = Device::get_device(self.device_id)
            .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
        let block_x = clamp_block_x_for_smem(device, req_block);
        let grid_x = ((n as u32) + block_x - 1) / block_x;

        // Single launch is fine (x-dimension only) — kernels index combos along x.
        unsafe {
            let mut func = self
                .module
                .get_function("emd_batch_f32")
                .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;

            // Configure for dynamic shared memory and prefer shared cache
            let dyn_smem_bytes: u32 = smem_for(block_x) as u32;
            opt_in_dynamic_smem(&func, dyn_smem_bytes)?;
            prefer_shared(&mut func)?;
            // Additionally hint carve-out preference to shared
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
            // ring buffers for peaks/valleys/bridge
            let mut p_sp = d_sp_ring.as_device_ptr().as_raw();
            let mut p_sv = d_sv_ring.as_device_ptr().as_raw();
            let mut p_bp = d_bp_ring.as_device_ptr().as_raw();
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
                &mut ring_stride_mid as *mut _ as *mut c_void,
                &mut p_sp as *mut _ as *mut c_void,
                &mut p_sv as *mut _ as *mut c_void,
                &mut p_bp as *mut _ as *mut c_void,
            ];
            let grid: GridSize = (grid_x, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            self.stream
                .launch(&func, grid, block, dyn_smem_bytes as u32, args)
                .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;

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

    // -------- Many series × one param (time-major) --------
    pub fn emd_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &EmdParams,
        first_valids: &[i32],
    ) -> Result<DeviceArrayF32Triple, CudaEmdError> {
        if cols == 0 || rows == 0 || data_tm_f32.len() != cols * rows {
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

        // VRAM estimate (inputs + first_valids + 3 outputs + ring scratch)
        let per_mid = 2 * (period as usize);
        let ring_bytes = cols * (50 + 50 + per_mid) * std::mem::size_of::<f32>();
        let bytes = data_tm_f32.len() * std::mem::size_of::<f32>()
            + first_valids.len() * std::mem::size_of::<i32>()
            + 3 * data_tm_f32.len() * std::mem::size_of::<f32>()
            + ring_bytes;
        if !Self::will_fit(bytes, 64 * 1024 * 1024) {
            return Err(CudaEmdError::InvalidInput(
                "insufficient device memory for emd many-series".into(),
            ));
        }

        let h_prices = LockedBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
        let h_fv = LockedBuffer::from_slice(first_valids)
            .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
        let mut d_prices: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }
                .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
        let mut d_fv: DeviceBuffer<i32> =
            unsafe { DeviceBuffer::uninitialized(cols) }
                .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
        unsafe {
            d_prices
                .async_copy_from(h_prices.as_slice(), &self.stream)
                .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
            d_fv
                .async_copy_from(h_fv.as_slice(), &self.stream)
                .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
        }
        let mut d_ub: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
        let mut d_mb: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
        let mut d_lb: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;

        // Needs dynamic shared memory like batch path
        let req_block = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 128,
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
        } as u32;
        let device = Device::get_device(self.device_id)
            .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
        let block_x = clamp_block_x_for_smem(device, req_block);
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        unsafe {
            let mut func = self
                .module
                .get_function("emd_many_series_one_param_time_major_f32")
                .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
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
            // Allocate ring buffers for many-series scratch (zeroed on device)
            let mut ring_stride_mid: i32 = per_mid as i32;
            let mut p_ub = d_ub.as_device_ptr().as_raw();
            let mut p_mb = d_mb.as_device_ptr().as_raw();
            let mut p_lb = d_lb.as_device_ptr().as_raw();
            // ring buffers pointers again for many-series variant (allocate zeroed on device)
            let mut d_sp_ring: DeviceBuffer<f32> = DeviceBuffer::zeroed(cols * 50)
                .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
            let mut d_sv_ring: DeviceBuffer<f32> = DeviceBuffer::zeroed(cols * 50)
                .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
            let mut d_bp_ring: DeviceBuffer<f32> = DeviceBuffer::zeroed(cols * per_mid)
                .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
            let mut ring_stride_mid: i32 = per_mid as i32;
            let mut p_sp = d_sp_ring.as_device_ptr().as_raw();
            let mut p_sv = d_sv_ring.as_device_ptr().as_raw();
            let mut p_bp = d_bp_ring.as_device_ptr().as_raw();
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
                &mut ring_stride_mid as *mut _ as *mut c_void,
                &mut p_sp as *mut _ as *mut c_void,
                &mut p_sv as *mut _ as *mut c_void,
                &mut p_bp as *mut _ as *mut c_void,
            ];
            let grid: GridSize = (grid_x, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            self.stream
                .launch(&func, grid, block, dyn_smem_bytes as u32, args)
                .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaEmdError::Cuda(e.to_string()))?;
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

    struct EmdBatchState {
        cuda: CudaEmd,
        high: Vec<f32>,
        low: Vec<f32>,
        sweep: EmdBatchRange,
    }
    impl CudaBenchState for EmdBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .emd_batch_dev(&self.high, &self.low, &self.sweep)
                .expect("emd batch");
            let _ = self.cuda.synchronize();
        }
    }
    fn prep_emd_batch_box() -> Box<dyn CudaBenchState> {
        let mut high = vec![f32::NAN; 60_000];
        let mut low = vec![f32::NAN; 60_000];
        for i in 2..60_000 {
            let x = i as f32;
            high[i] = (x * 0.001).sin() + 0.0002 * x + 0.5;
            low[i] = (x * 0.001).sin() + 0.0002 * x - 0.5;
        }
        let sweep = EmdBatchRange {
            period: (8, 20, 4),
            delta: (0.3, 0.7, 0.2),
            fraction: (0.05, 0.15, 0.05),
        };
        Box::new(EmdBatchState {
            cuda: CudaEmd::new(0).expect("cuda emd"),
            high,
            low,
            sweep,
        })
    }

    struct EmdManySeriesState {
        cuda: CudaEmd,
        data_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        params: EmdParams,
        first_valids: Vec<i32>,
    }
    impl CudaBenchState for EmdManySeriesState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .emd_many_series_one_param_time_major_dev(
                    &self.data_tm,
                    self.cols,
                    self.rows,
                    &self.params,
                    &self.first_valids,
                )
                .expect("emd many series");
            let _ = self.cuda.synchronize();
        }
    }
    fn prep_emd_many_series_box() -> Box<dyn CudaBenchState> {
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
        Box::new(EmdManySeriesState {
            cuda: CudaEmd::new(0).expect("cuda emd"),
            data_tm,
            cols,
            rows,
            params,
            first_valids,
        })
    }
}
