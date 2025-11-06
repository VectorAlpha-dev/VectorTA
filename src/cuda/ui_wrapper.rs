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
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::context::CacheConfig;
use cust::sys; // low-level CUDA driver API access
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Clone, Copy, Debug)]
pub enum UiBatchKernelPolicy {
    Auto,
    // Parameterized plain policy for the scaling step
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

#[derive(Debug)]
pub enum CudaUiError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaUiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaUiError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaUiError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl std::error::Error for CudaUiError {}

pub struct CudaUi {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaUiPolicy,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaUi {
    pub fn new(device_id: usize) -> Result<Self, CudaUiError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaUiError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaUiError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaUiError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/ui_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
                {
                    m
                } else {
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaUiError::Cuda(e.to_string()))?
                }
            }
        };

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaUiError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaUiPolicy::default(),
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn set_policy(&mut self, policy: CudaUiPolicy) { self.policy = policy; }
    pub fn policy(&self) -> &CudaUiPolicy { &self.policy }
    pub fn synchronize(&self) -> Result<(), CudaUiError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaUiError::Cuda(e.to_string()))
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
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else { true }
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
                return Err(CudaUiError::Cuda(format!(
                    "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES={}) failed: {:?}",
                    smem_bytes, r1
                )));
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

    // ---- Batch: one-series × many-params ----

    fn expand_grid(sweep: &UiBatchRange) -> (Vec<usize>, Vec<f32>) {
        let (ps, pe, pst) = sweep.period;
        let (ss, se, sst) = sweep.scalar;
        let periods: Vec<usize> = if pst == 0 || ps == pe {
            vec![ps]
        } else {
            (ps..=pe).step_by(pst).collect()
        };
        let periods: Vec<usize> = if pst == 0 || ps == pe {
            vec![ps]
        } else {
            (ps..=pe).step_by(pst).collect()
        };
        let scalars: Vec<f32> = if sst.abs() < 1e-12 || (ss - se).abs() < 1e-12 {
            vec![ss as f32]
        } else {
            let mut v = Vec::new();
            let mut x = ss;
            let mut iters = 0;
            while x <= se + 1e-12 && iters < 10000 {
                v.push(x as f32);
                x += sst;
                iters += 1;
            }
            if v.is_empty() {
                v.push(ss as f32);
            }
            if v.is_empty() {
                v.push(ss as f32);
            }
            v
        };
        (periods, scalars)
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

        let (periods, scalars) = Self::expand_grid(sweep);
        if periods.is_empty() || scalars.is_empty() {
            return Err(CudaUiError::InvalidInput("empty sweep".into()));
        }

        // Build combos
        let mut combos: Vec<UiParams> = Vec::with_capacity(periods.len() * scalars.len());
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
        let max_warm = first_valid + (max_p * 2 - 2);
        if len <= max_warm {
            return Err(CudaUiError::InvalidInput("not enough valid data".into()));
        }
        
        let headroom = env::var("CUDA_MEM_HEADROOM").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(64 * 1024 * 1024);
        let d_prices = DeviceBuffer::from_slice(prices).map_err(|e| CudaUiError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(rows * len) }
            .map_err(|e| CudaUiError::Cuda(e.to_string()))?;

        // Heuristic for multi-param kernel
        let n_unique_periods = periods.len();
        let scalars_per_period = scalars.len();
        let use_multi_param = match self.policy.batch {
            UiBatchKernelPolicy::Auto => n_unique_periods > 1 && (scalars_per_period <= 2 || n_unique_periods >= 8),
            UiBatchKernelPolicy::Plain { .. } => false,
        };

        if use_multi_param {
            // ---------- Multi-param (warp-per-param) ----------
            let mut periods_params: Vec<i32> = Vec::with_capacity(rows);
            let mut scalars_params: Vec<f32> = Vec::with_capacity(rows);
            for &p in &periods { for &s in &scalars { periods_params.push(p as i32); scalars_params.push(s); } }

            let required = len * std::mem::size_of::<f32>()
                + rows * len * std::mem::size_of::<f32>()
                + rows * (std::mem::size_of::<i32>() + std::mem::size_of::<f32>());
            if !Self::will_fit(required, headroom) { return Err(CudaUiError::InvalidInput("insufficient VRAM for UI batch (multi-param)".into())); }

            let d_periods = DeviceBuffer::from_slice(&periods_params).map_err(|e| CudaUiError::Cuda(e.to_string()))?;
            let d_scalars = DeviceBuffer::from_slice(&scalars_params).map_err(|e| CudaUiError::Cuda(e.to_string()))?;

            let mut func = self.module.get_function("ui_one_series_many_params_f32").map_err(|e| CudaUiError::Cuda(e.to_string()))?;
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

            let block = BlockSize::xyz(warps_per_block * 32, 1, 1);
            let grid = GridSize::xyz(((rows as u32) + warps_per_block - 1) / warps_per_block, 1, 1);

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
                self.stream.launch(&mut func, grid, block, smem as u32, &mut args)
                    .map_err(|e| CudaUiError::Cuda(e.to_string()))?;
            }

            if (cfg!(debug_assertions) || std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1")) && !self.debug_batch_logged {
                eprintln!("[ui] batch: multi-param rows={} len={} warps/block={} smem={}B", rows, len, warps_per_block, smem);
                unsafe { (*(self as *const _ as *mut CudaUi)).debug_batch_logged = true; }
            }
        } else {
            // ---------- Base+scale (optimized) ----------
            let required = (len /*in*/ + len /*base*/ + rows * len /*out*/)
                * std::mem::size_of::<f32>()
                + scalars.len() * std::mem::size_of::<f32>();
            if !Self::will_fit(required, headroom) { return Err(CudaUiError::InvalidInput("insufficient VRAM for UI batch (base+scale)".into())); }

            let mut d_base: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }
                .map_err(|e| CudaUiError::Cuda(e.to_string()))?;
            // Upload scalars once
            let d_scalars = DeviceBuffer::from_slice(&scalars).map_err(|e| CudaUiError::Cuda(e.to_string()))?;

            let mut fn_single = self.module.get_function("ui_single_series_f32").map_err(|e| CudaUiError::Cuda(e.to_string()))?;
            fn_single.set_cache_config(CacheConfig::PreferShared).ok();
            let fn_scale = self.module.get_function("ui_scale_rows_from_base_f32").map_err(|e| CudaUiError::Cuda(e.to_string()))?;

            let block_scale_x = match self.policy.batch { UiBatchKernelPolicy::Plain { block_x } => block_x, UiBatchKernelPolicy::Auto => 256 };
            let grid_scale_x = ((len as u32) + block_scale_x - 1) / block_scale_x;

            let mut start_row = 0usize;
            for &p in &periods {
                // Shared mem sizing matches kernel’s single-series layout (double ring region)
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
                    self.stream.launch(&mut fn_single, GridSize::xyz(1,1,1), BlockSize::xyz(1,1,1), shmem, &mut args)
                        .map_err(|e| CudaUiError::Cuda(e.to_string()))?;
                }

                // Scale across all scalars into rows
                const MAX_GRID_Y: usize = 65_535;
                let mut remaining = scalars.len();
                let mut row_off = start_row;
                while remaining > 0 {
                    let chunk = remaining.min(MAX_GRID_Y);
                    let grid = GridSize::xyz(grid_scale_x.max(1), chunk as u32, 1);
                    let block = BlockSize::xyz(block_scale_x, 1, 1);
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
                        self.stream.launch(&fn_scale, grid, block, 0, &mut args)
                            .map_err(|e| CudaUiError::Cuda(e.to_string()))?;
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

        self.stream.synchronize().map_err(|e| CudaUiError::Cuda(e.to_string()))?;
        Ok((DeviceArrayF32 { buf: d_out, rows, cols: len }, combos))
    }

    // ---- Many-series × one-param (time-major) ----

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
        if prices_tm.len() != cols * rows {
            return Err(CudaUiError::InvalidInput("matrix shape mismatch".into()));
        }
        if cols == 0 || rows == 0 {
            return Err(CudaUiError::InvalidInput("empty dims".into()));
        }
        if prices_tm.len() != cols * rows {
            return Err(CudaUiError::InvalidInput("matrix shape mismatch".into()));
        }
        let period = params.period.unwrap_or(14);
        let scalar_f32 = params.scalar.unwrap_or(100.0) as f32;

        // Per-series first valid detection
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
        for &fv in &first_valids {
            if (fv as usize) + (2 * period).saturating_sub(2) >= rows {
                return Err(CudaUiError::InvalidInput(
                    "not enough valid data for at least one series".into(),
                ));
            }
        }
        for &fv in &first_valids {
            if (fv as usize) + (2 * period).saturating_sub(2) >= rows {
                return Err(CudaUiError::InvalidInput(
                    "not enough valid data for at least one series".into(),
                ));
            }
        }

        // VRAM: inputs + outputs + first_valids (int32)
        let required = (cols * rows * 2) * std::mem::size_of::<f32>()
            + cols * std::mem::size_of::<i32>();
        let headroom = env::var("CUDA_MEM_HEADROOM").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(64 * 1024 * 1024);
        if !Self::will_fit(required, headroom) { return Err(CudaUiError::InvalidInput("insufficient VRAM for many-series".into())); }

        let d_prices =
            DeviceBuffer::from_slice(prices_tm).map_err(|e| CudaUiError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaUiError::Cuda(e.to_string()))?;
        let d_prices =
            DeviceBuffer::from_slice(prices_tm).map_err(|e| CudaUiError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaUiError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaUiError::Cuda(e.to_string()))?;

        let mut func = self.module.get_function("ui_many_series_one_param_time_major_f32")
            .map_err(|e| CudaUiError::Cuda(e.to_string()))?;
        func.set_cache_config(CacheConfig::PreferShared).ok();

        // This kernel uses per-block shared memory for a single series;
        // launch with exactly one thread per block to avoid shared-memory aliasing.
        let block_x: u32 = 1;
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        // FP32 + uint8 SMEM layout matches kernel (index, value, ring, valid)
        let shmem = {
            let deq_idx = Self::align16(period * std::mem::size_of::<i32>());
            let deq_val = Self::align16(period * std::mem::size_of::<f32>());
            let sq_ring = Self::align16(period * std::mem::size_of::<f32>());
            (deq_idx + deq_val + sq_ring + period * std::mem::size_of::<u8>()) as u32
        };
        // Allow large dynamic SMEM if needed
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
            self.stream.launch(&mut func, grid, block, shmem, &mut args)
                .map_err(|e| CudaUiError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaUiError::Cuda(e.to_string()))?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    // Convenience: copy back to host (used by tests)
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
        dev.buf
            .copy_to(out_host)
            .map_err(|e| CudaUiError::Cuda(e.to_string()))?;
        Ok((dev.rows, dev.cols, combos))
    }
}

// -------------- Benches --------------
pub mod benches {
    use super::*;
    use crate::cuda::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000; // 1m
    const MANY_COLS: usize = 128;
    const MANY_ROWS: usize = 200_000; // 200k

    fn bytes_one_series() -> usize {
        (ONE_SERIES_LEN * 2) * std::mem::size_of::<f32>()
    }
    fn bytes_many_series() -> usize {
        (MANY_COLS * MANY_ROWS * 2 + MANY_COLS) * std::mem::size_of::<f32>()
    }

    struct BatchState {
        cuda: CudaUi,
        prices: Vec<f32>,
        sweep: UiBatchRange,
    }
    impl CudaBenchState for BatchState {
        fn launch(&mut self) {
            let _ = self.cuda.ui_batch_dev(&self.prices, &self.sweep).unwrap();
        }
    }
    fn prep_one_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaUi::new(0).expect("cuda ui");
        let mut prices = vec![f32::NAN; ONE_SERIES_LEN];
        for i in 0..ONE_SERIES_LEN {
            let x = i as f32 * 0.00123;
            prices[i] = (x * 0.91).sin() + 0.0007 * x;
        }
        let sweep = UiBatchRange {
            period: (10, 60, 5),
            scalar: (100.0, 100.0, 0.0),
        };
        Box::new(BatchState {
            cuda,
            prices,
            sweep,
        })
    }

    struct ManyState {
        cuda: CudaUi,
        prices_tm: Vec<f32>,
    }
    impl CudaBenchState for ManyState {
        fn launch(&mut self) {
            let params = UiParams {
                period: Some(14),
                scalar: Some(100.0),
            };
            let _ = self
                .cuda
                .ui_many_series_one_param_time_major_dev(
                    &self.prices_tm,
                    MANY_COLS,
                    MANY_ROWS,
                    &params,
                )
                .unwrap();
        }
    }
    fn prep_many_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaUi::new(0).expect("cuda ui");
        let mut prices_tm = vec![f32::NAN; MANY_COLS * MANY_ROWS];
        for s in 0..MANY_COLS {
            for t in 0..MANY_ROWS {
                let x = t as f32 * 0.002 + s as f32 * 0.01;
                prices_tm[t * MANY_COLS + s] = (x * 0.73).sin() + 0.0009 * x;
            }
        }
        Box::new(ManyState { cuda, prices_tm })
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
            CudaBenchScenario::new(
                "ui",
                "many_series_one_param",
                "ui_cuda_many_series_tm",
                "200k x 128",
                prep_many_series,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_many_series()),
        ]
    }
}
