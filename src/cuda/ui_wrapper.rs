#![cfg(feature = "cuda")]

//! CUDA wrapper for Ulcer Index (UI)
//!
//! Parity goals with ALMA wrapper:
//! - PTX load with DetermineTargetFromContext + OptLevel O2 (graceful fallback)
//! - Non-blocking stream
//! - VRAM checks with headroom and grid.y chunking
//! - Public device entry points for batch and many-series (time-major)
//! - Bench profiles exposed via `benches::bench_profiles()`

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::ui::{UiBatchRange, UiParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize, Function};
use cust::launch;
use cust::error::CudaError;
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::context::CacheConfig;
use cust::sys; 
use std::env;
use std::ffi::c_void;
use std::sync::Arc;
use thiserror::Error;

#[derive(Clone, Copy, Debug)]
pub enum UiBatchKernelPolicy {
    Auto,
    
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum UiManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

impl Default for UiBatchKernelPolicy {
    fn default() -> Self {
        UiBatchKernelPolicy::Auto
    }
}
impl Default for UiManySeriesKernelPolicy {
    fn default() -> Self {
        UiManySeriesKernelPolicy::Auto
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaUiPolicy {
    pub batch: UiBatchKernelPolicy,
    pub many_series: UiManySeriesKernelPolicy,
}

#[derive(Error, Debug)]
pub enum CudaUiError {
    #[error(transparent)]
    Cuda(#[from] CudaError),
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

pub struct CudaUi {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaUiPolicy,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaUi {
    pub fn new(device_id: usize) -> Result<Self, CudaUiError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/ui_kernel.ptx"));
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
            policy: CudaUiPolicy::default(),
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn set_policy(&mut self, policy: CudaUiPolicy) { self.policy = policy; }
    pub fn policy(&self) -> &CudaUiPolicy { &self.policy }
    pub fn context_arc(&self) -> Arc<Context> { self.context.clone() }
    pub fn device_id(&self) -> u32 { self.device_id }
    pub fn synchronize(&self) -> Result<(), CudaUiError> {
        self.stream.synchronize()?;
        Ok(())
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }
    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> { mem_get_info().ok() }
    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    #[inline]
    fn align16(x: usize) -> usize { (x + 15) & !15 }

    fn set_kernel_dynamic_smem(func: &mut Function, smem_bytes: usize, carveout_percent: i32)
        -> Result<(), CudaUiError>
    {
        unsafe {
            let f = func.to_raw();
            let r1 = sys::cuFuncSetAttribute(
                f,
                sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                smem_bytes as i32,
            );
            if r1 != sys::CUresult::CUDA_SUCCESS {
                return Err(CudaUiError::Cuda(CudaError::UnknownError));
            }
            let _ = sys::cuFuncSetAttribute(
                f,
                sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                carveout_percent,
            );
        }
        Ok(())
    }

    fn device_optin_shared_mem(&self) -> usize {
        unsafe {
            let dev = Device::get_device(self.device_id).ok();
            if let Some(d) = dev {
                let mut v: i32 = 0;
                let rc = sys::cuDeviceGetAttribute(
                    &mut v as *mut i32,
                    sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                    d.as_raw(),
                );
                if rc == sys::CUresult::CUDA_SUCCESS && v > 0 { return v as usize; }
            }
            48 * 1024
        }
    }

    #[inline]
    fn validate_launch(&self, gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32) -> Result<(), CudaUiError> {
        let dev = Device::get_device(self.device_id)?;
        let max_threads = dev
            .get_attribute(cust::device::DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_bx = dev
            .get_attribute(cust::device::DeviceAttribute::MaxBlockDimX)
            .unwrap_or(1024) as u32;
        let max_by = dev
            .get_attribute(cust::device::DeviceAttribute::MaxBlockDimY)
            .unwrap_or(1024) as u32;
        let max_bz = dev
            .get_attribute(cust::device::DeviceAttribute::MaxBlockDimZ)
            .unwrap_or(64) as u32;
        let max_gx = dev
            .get_attribute(cust::device::DeviceAttribute::MaxGridDimX)
            .unwrap_or(2_147_483_647) as u32;
        let max_gy = dev
            .get_attribute(cust::device::DeviceAttribute::MaxGridDimY)
            .unwrap_or(65_535) as u32;
        let max_gz = dev
            .get_attribute(cust::device::DeviceAttribute::MaxGridDimZ)
            .unwrap_or(65_535) as u32;

        let threads = bx.saturating_mul(by).saturating_mul(bz);
        if threads > max_threads
            || bx > max_bx
            || by > max_by
            || bz > max_bz
            || gx > max_gx
            || gy > max_gy
            || gz > max_gz
        {
            return Err(CudaUiError::LaunchConfigTooLarge { gx, gy, gz, bx, by, bz });
        }
        Ok(())
    }

    

    fn expand_grid(sweep: &UiBatchRange) -> Result<(Vec<usize>, Vec<f32>), CudaUiError> {
        let (ps, pe, pst) = sweep.period;
        let (ss, se, sst) = sweep.scalar;
        let periods: Vec<usize> = if pst == 0 || ps == pe {
            vec![ps]
        } else if ps < pe {
            (ps..=pe).step_by(pst).collect()
        } else {
            
            let mut v: Vec<usize> = (pe..=ps).step_by(pst).collect();
            v.reverse();
            v
        };
        let scalars: Vec<f32> = if sst.abs() < 1e-12 || (ss - se).abs() < 1e-12 {
            vec![ss as f32]
        } else {
            if (ss < se && sst <= 0.0) || (ss > se && sst >= 0.0) {
                return Err(CudaUiError::InvalidInput(
                    "invalid scalar sweep: step sign must move from start toward end".into(),
                ));
            }
            let mut v = Vec::new();
            let mut x = ss;
            let mut iters: usize = 0;
            const MAX_ITERS: usize = 10_000;
            if ss < se {
                while x <= se + 1e-12 {
                    if iters >= MAX_ITERS {
                        return Err(CudaUiError::InvalidInput(
                            "scalar sweep produced too many steps".into(),
                        ));
                    }
                    v.push(x as f32);
                    x += sst;
                    iters += 1;
                }
            } else {
                while x >= se - 1e-12 {
                    if iters >= MAX_ITERS {
                        return Err(CudaUiError::InvalidInput(
                            "scalar sweep produced too many steps".into(),
                        ));
                    }
                    v.push(x as f32);
                    x += sst; 
                    iters += 1;
                }
            }
            if v.is_empty() {
                return Err(CudaUiError::InvalidInput("empty scalar sweep".into()));
            }
            v
        };
        if periods.is_empty() || scalars.is_empty() {
            return Err(CudaUiError::InvalidInput("empty sweep".into()));
        }
        Ok((periods, scalars))
    }

    pub fn ui_batch_dev(
        &self,
        prices: &[f32],
        sweep: &UiBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<UiParams>), CudaUiError> {
        if prices.is_empty() {
            return Err(CudaUiError::InvalidInput("empty input".into()));
        }
        let len = prices.len();
        let first_valid = (0..len)
            .find(|&i| prices[i].is_finite())
            .ok_or_else(|| CudaUiError::InvalidInput("all values are NaN".into()))?;

        let (periods, scalars) = Self::expand_grid(sweep)?;

        
        let combos_cap = periods
            .len()
            .checked_mul(scalars.len())
            .ok_or_else(|| CudaUiError::InvalidInput("rows * cols overflow in ui_batch_dev".into()))?;
        let mut combos: Vec<UiParams> = Vec::with_capacity(combos_cap);
        for &p in &periods {
            for &s in &scalars {
                combos.push(UiParams {
                    period: Some(p),
                    scalar: Some(s as f64),
                });
            }
        }
        let rows = combos.len();
        let max_p = *periods.iter().max().unwrap();
        let span = max_p
            .checked_mul(2)
            .and_then(|v| v.checked_sub(2))
            .ok_or_else(|| {
                CudaUiError::InvalidInput("period too large for warmup computation".into())
            })?;
        let max_warm = first_valid
            .checked_add(span)
            .ok_or_else(|| CudaUiError::InvalidInput("warmup index overflow".into()))?;
        if len <= max_warm {
            return Err(CudaUiError::InvalidInput("not enough valid data".into()));
        }

        let headroom = env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);

        let prices_bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                CudaUiError::InvalidInput("len overflow when computing prices_bytes".into())
            })?;
        let out_elems = rows
            .checked_mul(len)
            .ok_or_else(|| {
                CudaUiError::InvalidInput("rows * len overflow for output".into())
            })?;
        let out_bytes = out_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                CudaUiError::InvalidInput("output bytes overflow in ui_batch_dev".into())
            })?;

        
        let n_unique_periods = periods.len();
        let scalars_per_period = scalars.len();
        let use_multi_param = match self.policy.batch {
            UiBatchKernelPolicy::Auto => {
                n_unique_periods > 1 && (scalars_per_period <= 2 || n_unique_periods >= 8)
            }
            UiBatchKernelPolicy::Plain { .. } => false,
        };

        let (_required, d_prices, mut d_out) = if use_multi_param {
            let params_bytes_per_row =
                std::mem::size_of::<i32>() + std::mem::size_of::<f32>();
            let params_bytes = rows
                .checked_mul(params_bytes_per_row)
                .ok_or_else(|| {
                    CudaUiError::InvalidInput(
                        "rows overflow when computing params_bytes".into(),
                    )
                })?;
            let required = prices_bytes
                .checked_add(out_bytes)
                .and_then(|v| v.checked_add(params_bytes))
                .ok_or_else(|| {
                    CudaUiError::InvalidInput("required bytes overflow (multi-param)".into())
                })?;
            if !Self::will_fit(required, headroom) {
                if let Some((free, _total)) = Self::device_mem_info() {
                    return Err(CudaUiError::OutOfMemory {
                        required,
                        free,
                        headroom,
                    });
                } else {
                    return Err(CudaUiError::InvalidInput(
                        "insufficient device memory for UI batch (multi-param)".into(),
                    ));
                }
            }
            let d_prices = DeviceBuffer::from_slice(prices)?;
            let d_out: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized(out_elems) }?;
            (required, d_prices, d_out)
        } else {
            let base_bytes = len
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| {
                    CudaUiError::InvalidInput(
                        "len overflow when computing base_bytes".into(),
                    )
                })?;
            let scalars_bytes = scalars
                .len()
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| {
                    CudaUiError::InvalidInput(
                        "scalar count overflow when computing scalars_bytes".into(),
                    )
                })?;
            let required = prices_bytes
                .checked_add(out_bytes)
                .and_then(|v| v.checked_add(base_bytes))
                .and_then(|v| v.checked_add(scalars_bytes))
                .ok_or_else(|| {
                    CudaUiError::InvalidInput("required bytes overflow (base+scale)".into())
                })?;
            if !Self::will_fit(required, headroom) {
                if let Some((free, _total)) = Self::device_mem_info() {
                    return Err(CudaUiError::OutOfMemory {
                        required,
                        free,
                        headroom,
                    });
                } else {
                    return Err(CudaUiError::InvalidInput(
                        "insufficient device memory for UI batch (base+scale)".into(),
                    ));
                }
            }
            let d_prices = DeviceBuffer::from_slice(prices)?;
            let d_out: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized(out_elems) }?;
            (required, d_prices, d_out)
        };

        if use_multi_param {
            
            let mut periods_params: Vec<i32> = Vec::with_capacity(rows);
            let mut scalars_params: Vec<f32> = Vec::with_capacity(rows);
            for &p in &periods { for &s in &scalars { periods_params.push(p as i32); scalars_params.push(s); } }

            let d_periods = DeviceBuffer::from_slice(&periods_params)?;
            let d_scalars = DeviceBuffer::from_slice(&scalars_params)?;

            let mut func = self
                .module
                .get_function("ui_one_series_many_params_f32")
                .map_err(|_| CudaUiError::MissingKernelSymbol {
                    name: "ui_one_series_many_params_f32",
                })?;
            func.set_cache_config(CacheConfig::PreferShared).ok();

            let bytes_per_param = {
                let deq_idx = Self::align16(max_p * std::mem::size_of::<i32>());
                let deq_val = Self::align16(max_p * std::mem::size_of::<f32>());
                let sq_ring = Self::align16(max_p * std::mem::size_of::<f32>());
                deq_idx + deq_val + sq_ring + max_p * std::mem::size_of::<u8>()
            };
            let optin = self.device_optin_shared_mem();
            let mut warps_per_block = (optin / bytes_per_param).max(1) as u32;
            if warps_per_block > 8 { warps_per_block = 8; }
            let smem = (bytes_per_param as u64 * warps_per_block as u64) as usize;
            CudaUi::set_kernel_dynamic_smem(&mut func, smem, 100)?;

            let block_x = warps_per_block * 32;
            let grid_x = ((rows as u32) + warps_per_block - 1) / warps_per_block;
            let block = BlockSize::xyz(block_x, 1, 1);
            let grid = GridSize::xyz(grid_x, 1, 1);
            self.validate_launch(grid_x, 1, 1, block_x, 1, 1)?;

            unsafe {
                let mut a_prices = d_prices.as_device_ptr().as_raw();
                let mut a_len = len as i32;
                let mut a_periods = d_periods.as_device_ptr().as_raw();
                let mut a_scalars = d_scalars.as_device_ptr().as_raw();
                let mut a_nparams = rows as i32;
                let mut a_first = first_valid as i32;
                let mut a_maxp = max_p as i32;
                let mut a_out = d_out.as_device_ptr().as_raw();
                let mut args: [*mut c_void; 8] = [
                    &mut a_prices as *mut _ as *mut c_void,
                    &mut a_len as *mut _ as *mut c_void,
                    &mut a_periods as *mut _ as *mut c_void,
                    &mut a_scalars as *mut _ as *mut c_void,
                    &mut a_nparams as *mut _ as *mut c_void,
                    &mut a_first as *mut _ as *mut c_void,
                    &mut a_maxp as *mut _ as *mut c_void,
                    &mut a_out as *mut _ as *mut c_void,
                ];
                self.stream.launch(&mut func, grid, block, smem as u32, &mut args)?;
            }

            if (cfg!(debug_assertions) || std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1")) && !self.debug_batch_logged {
                eprintln!("[ui] batch: multi-param rows={} len={} warps/block={} smem={}B", rows, len, warps_per_block, smem);
                unsafe { (*(self as *const _ as *mut CudaUi)).debug_batch_logged = true; }
            }
        } else {
            
            let mut d_base: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }?;
            
            let d_scalars = DeviceBuffer::from_slice(&scalars)?;

            let mut fn_single = self
                .module
                .get_function("ui_single_series_f32")
                .map_err(|_| CudaUiError::MissingKernelSymbol {
                    name: "ui_single_series_f32",
                })?;
            fn_single.set_cache_config(CacheConfig::PreferShared).ok();
            let fn_scale = self
                .module
                .get_function("ui_scale_rows_from_base_f32")
                .map_err(|_| CudaUiError::MissingKernelSymbol {
                    name: "ui_scale_rows_from_base_f32",
                })?;

            let block_scale_x = match self.policy.batch { UiBatchKernelPolicy::Plain { block_x } => block_x, UiBatchKernelPolicy::Auto => 256 };
            let grid_scale_x = ((len as u32) + block_scale_x - 1) / block_scale_x;

            let mut start_row = 0usize;
            for &p in &periods {
                
                let shmem = {
                    let ints = p * std::mem::size_of::<i32>();
                    let align = std::mem::size_of::<f64>() - 1;
                    let ints_pad = (ints + align) & !align;
                    (ints_pad + p * std::mem::size_of::<f64>() + p * std::mem::size_of::<u8>()) as u32
                };
                CudaUi::set_kernel_dynamic_smem(&mut fn_single, shmem as usize, 100)?;

                unsafe {
                    let mut a_prices = d_prices.as_device_ptr().as_raw();
                    let mut a_len = len as i32;
                    let mut a_first = first_valid as i32;
                    let mut a_p = p as i32;
                    let mut a_base = d_base.as_device_ptr().as_raw();
                    let mut args: [*mut c_void; 5] = [
                        &mut a_prices as *mut _ as *mut c_void,
                        &mut a_len as *mut _ as *mut c_void,
                        &mut a_first as *mut _ as *mut c_void,
                        &mut a_p as *mut _ as *mut c_void,
                        &mut a_base as *mut _ as *mut c_void,
                    ];
                    self.validate_launch(1, 1, 1, 1, 1, 1)?;
                    self.stream.launch(&mut fn_single, GridSize::xyz(1,1,1), BlockSize::xyz(1,1,1), shmem, &mut args)?;
                }

                
                const MAX_GRID_Y: usize = 65_535;
                let mut remaining = scalars.len();
                let mut row_off = start_row;
                while remaining > 0 {
                    let chunk = remaining.min(MAX_GRID_Y);
                    let grid_x = grid_scale_x.max(1);
                    let grid_y = chunk as u32;
                    let grid = GridSize::xyz(grid_x, grid_y, 1);
                    let block = BlockSize::xyz(block_scale_x, 1, 1);
                    self.validate_launch(grid_x, grid_y, 1, block_scale_x, 1, 1)?;
                    unsafe {
                        let mut a_base = d_base.as_device_ptr().as_raw();
                        let mut a_scalars = d_scalars.as_device_ptr().as_raw();
                        let mut a_len = len as i32;
                        let mut a_rows = chunk as i32;
                        let mut a_out = d_out.as_device_ptr().as_raw().wrapping_add((row_off * len * std::mem::size_of::<f32>()) as u64);
                        let mut args: [*mut c_void; 5] = [
                            &mut a_base as *mut _ as *mut c_void,
                            &mut a_scalars as *mut _ as *mut c_void,
                            &mut a_len as *mut _ as *mut c_void,
                            &mut a_rows as *mut _ as *mut c_void,
                            &mut a_out as *mut _ as *mut c_void,
                        ];
                        self.stream.launch(&fn_scale, grid, block, 0, &mut args)?;
                    }
                    remaining -= chunk;
                    row_off += chunk;
                }

                if (cfg!(debug_assertions) || std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1")) && !self.debug_batch_logged {
                    eprintln!("[ui] batch: base+scale period={} rows={} len={} block_x={}", p, scalars.len(), len, block_scale_x);
                    unsafe { (*(self as *const _ as *mut CudaUi)).debug_batch_logged = true; }
                }

                start_row += scalars.len();
            }
        }

        self.stream.synchronize()?;
        Ok((DeviceArrayF32 { buf: d_out, rows, cols: len }, combos))
    }

    

    pub fn ui_many_series_one_param_time_major_dev(
        &self,
        prices_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &UiParams,
    ) -> Result<DeviceArrayF32, CudaUiError> {
        if cols == 0 || rows == 0 {
            return Err(CudaUiError::InvalidInput("empty dims".into()));
        }
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaUiError::InvalidInput("cols * rows overflow".into()))?;
        if prices_tm.len() != elems {
            return Err(CudaUiError::InvalidInput("matrix shape mismatch".into()));
        }
        let period = params.period.unwrap_or(14);
        let scalar_f32 = params.scalar.unwrap_or(100.0) as f32;

        
        let mut first_valids = vec![rows as i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                if prices_tm[idx].is_finite() {
                    first_valids[s] = t as i32;
                    break;
                }
                if prices_tm[idx].is_finite() {
                    first_valids[s] = t as i32;
                    break;
                }
            }
        }
        let warm_span = period
            .checked_mul(2)
            .and_then(|v| v.checked_sub(2))
            .ok_or_else(|| {
                CudaUiError::InvalidInput(
                    "period too large for warmup computation (many-series)".into(),
                )
            })?;
        for &fv in &first_valids {
            let warm = (fv as usize)
                .checked_add(warm_span)
                .ok_or_else(|| {
                    CudaUiError::InvalidInput("warmup index overflow (many-series)".into())
                })?;
            if warm >= rows {
                return Err(CudaUiError::InvalidInput(
                    "not enough valid data for at least one series".into(),
                ));
            }
        }

        
        let fp_bytes = elems
            .checked_mul(2)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                CudaUiError::InvalidInput(
                    "required bytes overflow when computing fp32 traffic (many-series)".into(),
                )
            })?;
        let first_bytes = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| {
                CudaUiError::InvalidInput(
                    "cols overflow when computing first_valids bytes".into(),
                )
            })?;
        let required = fp_bytes
            .checked_add(first_bytes)
            .ok_or_else(|| {
                CudaUiError::InvalidInput(
                    "required bytes overflow (many-series total)".into(),
                )
            })?;
        let headroom = env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        if !Self::will_fit(required, headroom) {
            if let Some((free, _total)) = Self::device_mem_info() {
                return Err(CudaUiError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaUiError::InvalidInput(
                    "insufficient VRAM for many-series".into(),
                ));
            }
        }

        let d_prices = DeviceBuffer::from_slice(prices_tm)?;
        let d_first = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }?;

        let mut func = self
            .module
            .get_function("ui_many_series_one_param_time_major_f32")
            .map_err(|_| CudaUiError::MissingKernelSymbol {
                name: "ui_many_series_one_param_time_major_f32",
            })?;
        func.set_cache_config(CacheConfig::PreferShared).ok();

        
        
        let block_x: u32 = 1;
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid_x_eff = grid_x.max(1);
        let grid: GridSize = (grid_x_eff, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        self.validate_launch(grid_x_eff, 1, 1, block_x, 1, 1)?;
        
        let shmem = {
            let deq_idx = Self::align16(period * std::mem::size_of::<i32>());
            let deq_val = Self::align16(period * std::mem::size_of::<f32>());
            let sq_ring = Self::align16(period * std::mem::size_of::<f32>());
            (deq_idx + deq_val + sq_ring + period * std::mem::size_of::<u8>()) as u32
        };
        
        CudaUi::set_kernel_dynamic_smem(&mut func, shmem as usize, 100)?;

        unsafe {
            let mut a_prices = d_prices.as_device_ptr().as_raw();
            let mut a_first = d_first.as_device_ptr().as_raw();
            let mut a_cols = cols as i32;
            let mut a_rows = rows as i32;
            let mut a_period = period as i32;
            let mut a_scalar = scalar_f32;
            let mut a_out = d_out.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 7] = [
                &mut a_prices as *mut _ as *mut c_void,
                &mut a_first as *mut _ as *mut c_void,
                &mut a_cols as *mut _ as *mut c_void,
                &mut a_rows as *mut _ as *mut c_void,
                &mut a_period as *mut _ as *mut c_void,
                &mut a_scalar as *mut _ as *mut c_void,
                &mut a_out as *mut _ as *mut c_void,
            ];
            self.stream.launch(&mut func, grid, block, shmem, &mut args)?;
        }

        self.stream.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    
    pub fn ui_batch_into_host_f32(
        &self,
        prices: &[f32],
        sweep: &UiBatchRange,
        out_host: &mut [f32],
    ) -> Result<(usize, usize, Vec<UiParams>), CudaUiError> {
        let (dev, combos) = self.ui_batch_dev(prices, sweep)?;
        let expected = dev.len();
        if out_host.len() != expected {
            return Err(CudaUiError::InvalidInput(format!(
                "output slice must be len {}",
                expected
            )));
        }
        dev.buf.copy_to(out_host)?;
        Ok((dev.rows, dev.cols, combos))
    }
}


pub mod benches {
    use super::*;
    use crate::cuda::{CudaBenchScenario, CudaBenchState};
    use std::ffi::c_void;

    const ONE_SERIES_LEN: usize = 1_000_000; 

    fn bytes_one_series() -> usize {
        
        let n_params = 11usize; 
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let params_bytes = n_params * (std::mem::size_of::<i32>() + std::mem::size_of::<f32>());
        let out_bytes = n_params * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        in_bytes + params_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct UiBatchState {
        cuda: CudaUi,
        d_prices: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_scalars: DeviceBuffer<f32>,
        d_out: DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        n_params: usize,
        max_p: usize,
        grid: GridSize,
        block: BlockSize,
        smem: u32,
    }
    impl CudaBenchState for UiBatchState {
        fn launch(&mut self) {
            let mut func = self
                .cuda
                .module
                .get_function("ui_one_series_many_params_f32")
                .expect("ui_one_series_many_params_f32");
            func.set_cache_config(CacheConfig::PreferShared).ok();
            unsafe {
                let mut a_prices = self.d_prices.as_device_ptr().as_raw();
                let mut a_len = self.len as i32;
                let mut a_periods = self.d_periods.as_device_ptr().as_raw();
                let mut a_scalars = self.d_scalars.as_device_ptr().as_raw();
                let mut a_nparams = self.n_params as i32;
                let mut a_first = self.first_valid as i32;
                let mut a_maxp = self.max_p as i32;
                let mut a_out = self.d_out.as_device_ptr().as_raw();
                let mut args: [*mut c_void; 8] = [
                    &mut a_prices as *mut _ as *mut c_void,
                    &mut a_len as *mut _ as *mut c_void,
                    &mut a_periods as *mut _ as *mut c_void,
                    &mut a_scalars as *mut _ as *mut c_void,
                    &mut a_nparams as *mut _ as *mut c_void,
                    &mut a_first as *mut _ as *mut c_void,
                    &mut a_maxp as *mut _ as *mut c_void,
                    &mut a_out as *mut _ as *mut c_void,
                ];
                self.cuda
                    .stream
                    .launch(&mut func, self.grid, self.block, self.smem, &mut args)
                    .expect("ui launch");
            }
            self.cuda.stream.synchronize().expect("ui sync");
        }
    }
    fn prep_one_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaUi::new(0).expect("cuda ui");
        let mut prices = vec![f32::NAN; ONE_SERIES_LEN];
        for i in 0..ONE_SERIES_LEN {
            let x = i as f32 * 0.00123;
            prices[i] = (x * 0.91).sin() + 0.0007 * x;
        }
        let first_valid = prices
            .iter()
            .position(|v| v.is_finite())
            .unwrap_or(ONE_SERIES_LEN);
        let sweep = UiBatchRange {
            period: (10, 60, 5),
            scalar: (100.0, 100.0, 0.0),
        };
        let (periods, scalars) = CudaUi::expand_grid(&sweep).expect("expand_grid");
        let max_p = *periods.iter().max().unwrap_or(&1);
        let n_params = periods.len() * scalars.len();
        let mut periods_params: Vec<i32> = Vec::with_capacity(n_params);
        let mut scalars_params: Vec<f32> = Vec::with_capacity(n_params);
        for &p in &periods {
            for &s in &scalars {
                periods_params.push(p as i32);
                scalars_params.push(s);
            }
        }

        
        let bytes_per_param = {
            let deq_idx = CudaUi::align16(max_p * std::mem::size_of::<i32>());
            let deq_val = CudaUi::align16(max_p * std::mem::size_of::<f32>());
            let sq_ring = CudaUi::align16(max_p * std::mem::size_of::<f32>());
            deq_idx + deq_val + sq_ring + max_p * std::mem::size_of::<u8>()
        };
        let optin = cuda.device_optin_shared_mem();
        let mut warps_per_block = (optin / bytes_per_param).max(1) as u32;
        if warps_per_block > 8 {
            warps_per_block = 8;
        }
        let smem = (bytes_per_param as u64 * warps_per_block as u64) as usize;
        let block_x = warps_per_block * 32;
        let grid_x = ((n_params as u32) + warps_per_block - 1) / warps_per_block;
        cuda.validate_launch(grid_x, 1, 1, block_x, 1, 1)
            .expect("ui validate_launch");

        
        let mut func = cuda
            .module
            .get_function("ui_one_series_many_params_f32")
            .expect("ui_one_series_many_params_f32");
        func.set_cache_config(CacheConfig::PreferShared).ok();
        CudaUi::set_kernel_dynamic_smem(&mut func, smem, 100).expect("set_kernel_dynamic_smem");

        let d_prices = DeviceBuffer::from_slice(&prices).expect("d_prices");
        let d_periods = DeviceBuffer::from_slice(&periods_params).expect("d_periods");
        let d_scalars = DeviceBuffer::from_slice(&scalars_params).expect("d_scalars");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_params * ONE_SERIES_LEN) }.expect("d_out");
        cuda.stream.synchronize().expect("ui prep sync");

        Box::new(UiBatchState {
            cuda,
            d_prices,
            d_periods,
            d_scalars,
            d_out,
            len: ONE_SERIES_LEN,
            first_valid,
            n_params,
            max_p,
            grid: GridSize::xyz(grid_x, 1, 1),
            block: BlockSize::xyz(block_x, 1, 1),
            smem: smem as u32,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "ui",
                "one_series_many_params",
                "ui_cuda_batch",
                "1m",
                prep_one_series,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series()),
        ]
    }
}
