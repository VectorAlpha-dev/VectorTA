//! CUDA wrapper for KDJ (Stochastic Oscillator with K, D, J lines).
//!
//! - Mirrors ALMA-style PTX loading (DetermineTargetFromContext + O2 fallback) and NON_BLOCKING stream.
//! - Batch: one series × many-params (rows = combinations, cols = len)
//! - Many-series: time-major, one param for all series
//! - Fused fast paths in the kernel for SMA→SMA and EMA→EMA smoothing with scalar-identical warmup/NaN rules.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::cuda::DeviceArrayF32Triplet;
use crate::indicators::kdj::{KdjBatchRange, KdjParams};
use crate::indicators::willr::build_willr_gpu_tables; // re-use sparse tables for HH/LL
use cust::context::{CacheConfig, Context};
use cust::device::Device;
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{
    mem_get_info, AsyncCopyDestination, CopyDestination, DeviceBuffer, LockedBuffer,
};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;

#[derive(Debug)]
pub enum CudaKdjError {
    Cuda(String),
    InvalidInput(String),
}

impl std::fmt::Display for CudaKdjError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaKdjError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaKdjError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaKdjError {}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
}
impl Default for BatchKernelPolicy {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}
impl Default for ManySeriesKernelPolicy {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CudaKdjPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaKdjPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

pub struct CudaKdj {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaKdjPolicy,
}

impl CudaKdj {
    pub fn new(device_id: usize) -> Result<Self, CudaKdjError> {
        Self::new_with_policy(device_id, CudaKdjPolicy::default())
    }

    pub fn new_with_policy(device_id: usize, policy: CudaKdjPolicy) -> Result<Self, CudaKdjError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaKdjError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/kdj_kernel.ptx"));
        let jit = [
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, &jit)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaKdjError::Cuda(e.to_string()))?;

        // Prefer L1 when available
        let _ = cust::context::CurrentContext::set_cache_config(CacheConfig::PreferL1);

        Ok(Self { module, stream, _context: context, policy })
    }

    fn device_mem_ok(required: usize, headroom: usize) -> bool {
        mem_get_info()
            .map(|(free, _)| required.saturating_add(headroom) <= free)
            .unwrap_or(true)
    }

    fn ma_to_code(s: &str) -> Result<i32, CudaKdjError> {
        if s.eq_ignore_ascii_case("sma") {
            return Ok(0);
        } else if s.eq_ignore_ascii_case("ema") {
            return Ok(1);
        } else {
            return Err(CudaKdjError::InvalidInput(format!(
                "unsupported MA type '{}'; supported: sma, ema",
                s
            )));
        }
    }

    fn expand_grid(range: &KdjBatchRange) -> Vec<KdjParams> {
        // Mirror indicators::kdj::expand_grid
        fn axis_usize(a: (usize, usize, usize)) -> Vec<usize> {
            let (start, end, step) = a;
            if step == 0 || start == end {
                vec![start]
            } else {
                (start..=end).step_by(step).collect()
            }
        }
        fn axis_str(a: (String, String, String)) -> Vec<String> {
            let (start, end, _step) = a;
            if start == end { vec![start] } else { vec![start, end] }
        }
        let fks = axis_usize(range.fast_k_period);
        let sks = axis_usize(range.slow_k_period);
        let kmas = axis_str(range.slow_k_ma_type.clone());
        let sds = axis_usize(range.slow_d_period);
        let dmas = axis_str(range.slow_d_ma_type.clone());
        let mut out = Vec::new();
        for &fk in &fks {
            for &sk in &sks {
                for kma in &kmas {
                    for &sd in &sds {
                        for dma in &dmas {
                            out.push(KdjParams {
                                fast_k_period: Some(fk),
                                slow_k_period: Some(sk),
                                slow_k_ma_type: Some(kma.clone()),
                                slow_d_period: Some(sd),
                                slow_d_ma_type: Some(dma.clone()),
                            });
                        }
                    }
                }
            }
        }
        out
    }

    // ---- Batch path ----
    pub fn kdj_batch_dev(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        close_f32: &[f32],
        sweep: &KdjBatchRange,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32, DeviceArrayF32), CudaKdjError> {
        let len = high_f32.len();
        if len == 0 || low_f32.len() != len || close_f32.len() != len {
            return Err(CudaKdjError::InvalidInput(
                "input slices are empty or mismatched".into(),
            ));
        }
        // first valid overall index
        let first_valid = (0..len)
            .find(|&i| high_f32[i].is_finite() && low_f32[i].is_finite() && close_f32[i].is_finite())
            .ok_or_else(|| CudaKdjError::InvalidInput("all values are NaN".into()))?
            as i32;

        let combos = Self::expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaKdjError::InvalidInput("no parameter combinations".into()));
        }

        // Validate and encode params; only SMA/EMA pairs are supported in the fused kernel
        let mut fk: Vec<i32> = Vec::with_capacity(combos.len());
        let mut sk: Vec<i32> = Vec::with_capacity(combos.len());
        let mut sd: Vec<i32> = Vec::with_capacity(combos.len());
        let mut kma: Vec<i32> = Vec::with_capacity(combos.len());
        let mut dma: Vec<i32> = Vec::with_capacity(combos.len());
        let mut max_fk = 0usize;
        for p in &combos {
            let fkv = p.fast_k_period.unwrap_or(0);
            let skv = p.slow_k_period.unwrap_or(0);
            let sdv = p.slow_d_period.unwrap_or(0);
            if fkv == 0 || skv == 0 || sdv == 0 {
                return Err(CudaKdjError::InvalidInput("periods must be positive".into()));
            }
            fk.push(fkv as i32);
            sk.push(skv as i32);
            sd.push(sdv as i32);
            kma.push(Self::ma_to_code(p.slow_k_ma_type.as_deref().unwrap_or("sma"))?);
            dma.push(Self::ma_to_code(p.slow_d_ma_type.as_deref().unwrap_or("sma"))?);
            fk.push(fkv as i32);
            sk.push(skv as i32);
            sd.push(sdv as i32);
            kma.push(Self::ma_to_code(
                p.slow_k_ma_type.as_deref().unwrap_or("sma"),
            )?);
            dma.push(Self::ma_to_code(
                p.slow_d_ma_type.as_deref().unwrap_or("sma"),
            )?);
            max_fk = max_fk.max(fkv);
        }
        let valid_tail = len as i32 - first_valid;
        if valid_tail < max_fk as i32 {
            return Err(CudaKdjError::InvalidInput(format!(
                "not enough valid data: need >= {}, have {}",
                max_fk, valid_tail
            )));
        }

        // Build sparse tables for HH/LL (shared across all rows)
        let tables = build_willr_gpu_tables(high_f32, low_f32);
        let level_count = tables.level_offsets.len() as i32; // full count; kernel checks k_log2 < level_count

        // Estimate VRAM and allow chunking by rows
        let nrows = combos.len();
        let bytes_inputs = (close_f32.len()) * std::mem::size_of::<f32>(); // batch kernel only reads close + tables
        let bytes_tables = (tables.log2.len() + tables.level_offsets.len() + tables.nan_psum.len()) * std::mem::size_of::<i32>()
            + (tables.st_max.len() + tables.st_min.len()) * std::mem::size_of::<f32>();
        let bytes_params =
            (fk.len() + sk.len() + sd.len() + kma.len() + dma.len()) * std::mem::size_of::<i32>();
        let bytes_params =
            (fk.len() + sk.len() + sd.len() + kma.len() + dma.len()) * std::mem::size_of::<i32>();
        let bytes_outputs = nrows * len * 3 * std::mem::size_of::<f32>();
        let required = bytes_inputs + bytes_tables + bytes_params + bytes_outputs;
        let headroom = 64 * 1024 * 1024; // 64MB
        // Modern GPUs allow gridDim.x up to 2^31-1; we do not chunk by 65,535.
        // Keep a single launch domain; rows cap is applied inside the loop.
        let combos_per_launch = nrows;

        // D2H pinned staging (async) when large
        let use_async = required > (64 * 1024 * 1024);

        // Upload common inputs (batch kernel does NOT dereference high/low)
        let (d_close, d_log2, d_offsets, d_st_max, d_st_min, d_nan_psum) = if use_async {
            let h_close = LockedBuffer::from_slice(close_f32).map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
            let h_log2 = LockedBuffer::from_slice(&tables.log2).map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
            let h_offs = LockedBuffer::from_slice(&tables.level_offsets).map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
            let h_max  = LockedBuffer::from_slice(&tables.st_max).map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
            let h_min  = LockedBuffer::from_slice(&tables.st_min).map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
            let h_nps  = LockedBuffer::from_slice(&tables.nan_psum).map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
            let mut dc = unsafe { DeviceBuffer::<f32>::uninitialized_async(len, &self.stream) }.map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
            let mut dl = unsafe { DeviceBuffer::<i32>::uninitialized_async(tables.log2.len(), &self.stream) }.map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
            let mut dof= unsafe { DeviceBuffer::<i32>::uninitialized_async(tables.level_offsets.len(), &self.stream) }.map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
            let mut dmx= unsafe { DeviceBuffer::<f32>::uninitialized_async(tables.st_max.len(), &self.stream) }.map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
            let mut dmn= unsafe { DeviceBuffer::<f32>::uninitialized_async(tables.st_min.len(), &self.stream) }.map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
            let mut dnp= unsafe { DeviceBuffer::<i32>::uninitialized_async(tables.nan_psum.len(), &self.stream) }.map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
            unsafe { dc.async_copy_from(&h_close, &self.stream) }.map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
            unsafe { dl.async_copy_from(&h_log2, &self.stream) }.map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
            unsafe { dof.async_copy_from(&h_offs, &self.stream) }.map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
            unsafe { dmx.async_copy_from(&h_max , &self.stream) }.map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
            unsafe { dmn.async_copy_from(&h_min , &self.stream) }.map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
            unsafe { dnp.async_copy_from(&h_nps , &self.stream) }.map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
            (dc, dl, dof, dmx, dmn, dnp)
        } else {
            (
                DeviceBuffer::from_slice(close_f32).map_err(|e| CudaKdjError::Cuda(e.to_string()))?,
                DeviceBuffer::from_slice(&tables.log2).map_err(|e| CudaKdjError::Cuda(e.to_string()))?,
                DeviceBuffer::from_slice(&tables.level_offsets).map_err(|e| CudaKdjError::Cuda(e.to_string()))?,
                DeviceBuffer::from_slice(&tables.st_max).map_err(|e| CudaKdjError::Cuda(e.to_string()))?,
                DeviceBuffer::from_slice(&tables.st_min).map_err(|e| CudaKdjError::Cuda(e.to_string()))?,
                DeviceBuffer::from_slice(&tables.nan_psum).map_err(|e| CudaKdjError::Cuda(e.to_string()))?,
            )
        };

        // Persist parameter arrays on device (avoid per-chunk allocations)
        let d_fk_all = DeviceBuffer::from_slice(&fk).map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
        let d_sk_all = DeviceBuffer::from_slice(&sk).map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
        let d_sd_all = DeviceBuffer::from_slice(&sd).map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
        let d_km_all = DeviceBuffer::from_slice(&kma).map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
        let d_dm_all = DeviceBuffer::from_slice(&dma).map_err(|e| CudaKdjError::Cuda(e.to_string()))?;

        // Allocate outputs
        let mut d_k = unsafe { DeviceBuffer::<f32>::uninitialized(nrows * len) }
            .map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
        let mut d_d = unsafe { DeviceBuffer::<f32>::uninitialized(nrows * len) }
            .map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
        let mut d_j = unsafe { DeviceBuffer::<f32>::uninitialized(nrows * len) }
            .map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
        let mut d_k = unsafe { DeviceBuffer::<f32>::uninitialized(nrows * len) }
            .map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
        let mut d_d = unsafe { DeviceBuffer::<f32>::uninitialized(nrows * len) }
            .map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
        let mut d_j = unsafe { DeviceBuffer::<f32>::uninitialized(nrows * len) }
            .map_err(|e| CudaKdjError::Cuda(e.to_string()))?;

        // Locate kernel
        let mut func: Function = self
            .module
            .get_function("kdj_batch_f32")
            .map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
        let mut func: Function = self
            .module
            .get_function("kdj_batch_f32")
            .map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);

        // Chunk rows if needed
        let mut row0 = 0usize;
        while row0 < nrows {
            let rows = (nrows - row0).min(combos_per_launch).min(2_147_483_647usize);
            let grid: GridSize = (rows as u32, 1, 1).into();
            let mut block_x = match self.policy.batch { BatchKernelPolicy::Plain { block_x } => block_x, _ => 256 };
            if block_x < 32 { block_x = 32; }
            let block: BlockSize = (block_x, 1, 1).into();

            unsafe {
                let mut close_ptr   = d_close.as_device_ptr().as_raw();
                let mut log2_ptr    = d_log2.as_device_ptr().as_raw();
                let mut offs_ptr    = d_offsets.as_device_ptr().as_raw();
                let mut stmax_ptr   = d_st_max.as_device_ptr().as_raw();
                let mut stmin_ptr   = d_st_min.as_device_ptr().as_raw();
                let mut nanp_ptr    = d_nan_psum.as_device_ptr().as_raw();
                // Reuse close pointer for the (unused) high/low kernel args
                let mut high_ptr    = close_ptr;
                let mut low_ptr     = close_ptr;
                // Param pointers with row offsets
                let mut fk_ptr      = d_fk_all.as_device_ptr().offset(row0 as isize).as_raw();
                let mut sk_ptr      = d_sk_all.as_device_ptr().offset(row0 as isize).as_raw();
                let mut sd_ptr      = d_sd_all.as_device_ptr().offset(row0 as isize).as_raw();
                let mut kma_ptr     = d_km_all.as_device_ptr().offset(row0 as isize).as_raw();
                let mut dma_ptr     = d_dm_all.as_device_ptr().offset(row0 as isize).as_raw();
                let mut series_len_i= len as i32;
                let mut first_i     = first_valid as i32;
                let mut level_cnt_i = level_count as i32;
                let mut nrows_i = rows as i32;
                let mut outk_ptr =
                    unsafe { d_k.as_device_ptr().offset((row0 * len) as isize).as_raw() };
                let mut outd_ptr =
                    unsafe { d_d.as_device_ptr().offset((row0 * len) as isize).as_raw() };
                let mut outj_ptr =
                    unsafe { d_j.as_device_ptr().offset((row0 * len) as isize).as_raw() };
                let mut nrows_i = rows as i32;
                let mut outk_ptr =
                    unsafe { d_k.as_device_ptr().offset((row0 * len) as isize).as_raw() };
                let mut outd_ptr =
                    unsafe { d_d.as_device_ptr().offset((row0 * len) as isize).as_raw() };
                let mut outj_ptr =
                    unsafe { d_j.as_device_ptr().offset((row0 * len) as isize).as_raw() };

                let args: &mut [*mut c_void] = &mut [
                    &mut high_ptr as *mut _ as *mut c_void,
                    &mut low_ptr as *mut _ as *mut c_void,
                    &mut close_ptr as *mut _ as *mut c_void,
                    &mut log2_ptr as *mut _ as *mut c_void,
                    &mut offs_ptr as *mut _ as *mut c_void,
                    &mut stmax_ptr as *mut _ as *mut c_void,
                    &mut stmin_ptr as *mut _ as *mut c_void,
                    &mut nanp_ptr as *mut _ as *mut c_void,
                    &mut fk_ptr as *mut _ as *mut c_void,
                    &mut sk_ptr as *mut _ as *mut c_void,
                    &mut sd_ptr as *mut _ as *mut c_void,
                    &mut kma_ptr as *mut _ as *mut c_void,
                    &mut dma_ptr as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut level_cnt_i as *mut _ as *mut c_void,
                    &mut nrows_i as *mut _ as *mut c_void,
                    &mut outk_ptr as *mut _ as *mut c_void,
                    &mut outd_ptr as *mut _ as *mut c_void,
                    &mut outj_ptr as *mut _ as *mut c_void,
                ];

                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
            }
            row0 += rows;
        }

        // Ensure kernels finished before dropping short‑lived device buffers
        self.stream.synchronize().map_err(|e| CudaKdjError::Cuda(e.to_string()))?;

        // Ensure kernels finished before dropping short‑lived device buffers
        self.stream.synchronize().map_err(|e| CudaKdjError::Cuda(e.to_string()))?;

        Ok((
            DeviceArrayF32 { buf: d_k, rows: nrows, cols: len },
            DeviceArrayF32 { buf: d_d, rows: nrows, cols: len },
            DeviceArrayF32 { buf: d_j, rows: nrows, cols: len },
        ))
    }

    // ---- Many-series (time-major) ----
    pub fn kdj_many_series_one_param_time_major_dev(
        &self,
        high_tm_f32: &[f32],
        low_tm_f32: &[f32],
        close_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &KdjParams,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32, DeviceArrayF32), CudaKdjError> {
        if cols == 0 || rows == 0 {
            return Err(CudaKdjError::InvalidInput(
                "series dims must be positive".into(),
            ));
        }
        if high_tm_f32.len() != cols * rows
            || low_tm_f32.len() != cols * rows
            || close_tm_f32.len() != cols * rows
        {
            return Err(CudaKdjError::InvalidInput(
                "time-major slices mismatch dims".into(),
            ));
        }
        let fk = params.fast_k_period.unwrap_or(9);
        let sk = params.slow_k_period.unwrap_or(3);
        let sd = params.slow_d_period.unwrap_or(3);
        let kma = Self::ma_to_code(params.slow_k_ma_type.as_deref().unwrap_or("sma"))?;
        let dma = Self::ma_to_code(params.slow_d_ma_type.as_deref().unwrap_or("sma"))?;

        // first_valid per series
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let idx = t * cols + s;
                if high_tm_f32[idx].is_finite()
                    && low_tm_f32[idx].is_finite()
                    && close_tm_f32[idx].is_finite()
                {
                    fv = Some(t as i32);
                    break;
                }
                if high_tm_f32[idx].is_finite()
                    && low_tm_f32[idx].is_finite()
                    && close_tm_f32[idx].is_finite()
                {
                    fv = Some(t as i32);
                    break;
                }
            }
            let f =
                fv.ok_or_else(|| CudaKdjError::InvalidInput(format!("series {} all NaN", s)))?;
            if rows - (f as usize) < fk {
                return Err(CudaKdjError::InvalidInput(format!(
                    "series {} insufficient data for fk {}",
                    s, fk
                )));
            }
            first_valids[s] = f;
        }

        // Upload inputs
        let d_h = DeviceBuffer::from_slice(high_tm_f32).map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
        let d_l = DeviceBuffer::from_slice(low_tm_f32).map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
        let d_c = DeviceBuffer::from_slice(close_tm_f32).map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
        let d_fv = DeviceBuffer::from_slice(&first_valids).map_err(|e| CudaKdjError::Cuda(e.to_string()))?;

        let mut d_k = unsafe { DeviceBuffer::<f32>::uninitialized(cols * rows) }
            .map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
        let mut d_d = unsafe { DeviceBuffer::<f32>::uninitialized(cols * rows) }
            .map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
        let mut d_j = unsafe { DeviceBuffer::<f32>::uninitialized(cols * rows) }
            .map_err(|e| CudaKdjError::Cuda(e.to_string()))?;

        let mut func: Function = self
            .module
            .get_function("kdj_many_series_one_param_f32")
            .map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);
        let mut block_x: u32 = match self.policy.many_series { ManySeriesKernelPolicy::OneD { block_x } => block_x, _ => 128 };
        if block_x < 32 { block_x = 32; }
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut h_ptr = d_h.as_device_ptr().as_raw();
            let mut l_ptr = d_l.as_device_ptr().as_raw();
            let mut c_ptr = d_c.as_device_ptr().as_raw();
            let mut fv_ptr = d_fv.as_device_ptr().as_raw();
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut fk_i = fk as i32;
            let mut sk_i = sk as i32;
            let mut sd_i = sd as i32;
            let mut kma_i = kma as i32;
            let mut dma_i = dma as i32;
            let mut ko_ptr = d_k.as_device_ptr().as_raw();
            let mut do_ptr = d_d.as_device_ptr().as_raw();
            let mut jo_ptr = d_j.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut h_ptr as *mut _ as *mut c_void,
                &mut l_ptr as *mut _ as *mut c_void,
                &mut c_ptr as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut fk_i as *mut _ as *mut c_void,
                &mut sk_i as *mut _ as *mut c_void,
                &mut sd_i as *mut _ as *mut c_void,
                &mut kma_i as *mut _ as *mut c_void,
                &mut dma_i as *mut _ as *mut c_void,
                &mut ko_ptr as *mut _ as *mut c_void,
                &mut do_ptr as *mut _ as *mut c_void,
                &mut jo_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaKdjError::Cuda(e.to_string()))?;
        }

        Ok((
            DeviceArrayF32 { buf: d_k, rows, cols },
            DeviceArrayF32 { buf: d_d, rows, cols },
            DeviceArrayF32 { buf: d_j, rows, cols },
        ))
    }
}

// ------------------- Benches -------------------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::kdj::KdjBatchRange;

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250; // (fast_k, slow_k, slow_d) sweep along fast_k

    fn synth_hlc_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i];
            if !v.is_finite() {
                continue;
            }
            let x = i as f32 * 0.0023;
            let off = (0.0029 * x.sin()).abs() + 0.1;
            high[i] = v + off;
            low[i] = v - off;
        }
        (high, low)
    }

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = 1 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = 3 * ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct KdjBatchState {
        cuda: CudaKdj,
        high: Vec<f32>,
        low: Vec<f32>,
        close: Vec<f32>,
        sweep: KdjBatchRange,
    }
    impl CudaBenchState for KdjBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .kdj_batch_dev(&self.high, &self.low, &self.close, &self.sweep)
                .expect("kdj batch");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaKdj::new(0).expect("cuda kdj");
        let close = gen_series(ONE_SERIES_LEN);
        let (high, low) = synth_hlc_from_close(&close);
        // Sweep fast_k only; keep smoothing fixed to SMA/SMA (fused path)
        let sweep = KdjBatchRange {
            fast_k_period: (9, 9 + PARAM_SWEEP - 1, 1),
            slow_k_period: (3, 3, 0),
            slow_k_ma_type: ("sma".into(), "sma".into(), "".into()),
            slow_d_period: (3, 3, 0),
            slow_d_ma_type: ("sma".into(), "sma".into(), "".into()),
        };
        Box::new(KdjBatchState { cuda, high, low, close, sweep })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "kdj",
            "one_series_many_params",
            "kdj_cuda_batch_dev",
            "1m_x_250",
            prep_one_series_many_params,
        )
        .with_mem_required(bytes_one_series_many_params())
        .with_sample_size(10)]
    }
}
