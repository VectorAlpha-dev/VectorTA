#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::cuda::DeviceArrayF32Triplet; // shared 3-output handle
use crate::indicators::mod_god_mode::{ModGodModeBatchRange, ModGodModeMode, ModGodModeParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{CopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::sys as cu;
use std::env;
use std::ffi::c_void;
use std::fmt;

// Must match the kernel's MGM_RING_KCAP (power-of-two, default 64)
const MGM_RING_KCAP: i32 = 64;

#[derive(Debug)]
pub enum CudaModGodModeError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaModGodModeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaModGodModeError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaModGodModeError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl std::error::Error for CudaModGodModeError {}

pub struct CudaModGodModeBatchResult {
    pub outputs: DeviceArrayF32Triplet,
    pub combos: Vec<ModGodModeParams>,
}

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
pub struct CudaModGodModePolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaModGodModePolicy {
    fn default() -> Self {
        Self { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Auto }
    }
}

pub struct CudaModGodMode {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaModGodModePolicy,
    debug_logged: bool,
}

impl CudaModGodMode {
    pub fn new(device_id: usize) -> Result<Self, CudaModGodModeError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaModGodModeError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaModGodModeError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaModGodModeError::Cuda(e.to_string()))?;
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/mod_god_mode_kernel.ptx"));
        let jit_opts = &[ModuleJitOption::DetermineTargetFromContext, ModuleJitOption::OptLevel(OptLevel::O2)];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => Module::from_ptx(ptx, &[]).map_err(|e| CudaModGodModeError::Cuda(e.to_string()))?,
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaModGodModeError::Cuda(e.to_string()))?;
        Ok(Self { module, stream, _context: context, policy: CudaModGodModePolicy::default(), debug_logged: false })
    }

    fn device_mem_info() -> Option<(usize, usize)> {
        unsafe {
            let mut free: usize = 0;
            let mut total: usize = 0;
            let res = cu::cuMemGetInfo_v2(&mut free as *mut usize, &mut total as *mut usize);
            if res == cu::CUresult::CUDA_SUCCESS { Some((free, total)) } else { None }
        }
    }
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") { Ok(v) => v != "0" && v.to_lowercase() != "false", Err(_) => true }
    }
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() { return true; }
        if let Some((free, _)) = Self::device_mem_info() { required_bytes.saturating_add(headroom_bytes) <= free } else { true }
    }

    fn expand_range(r: &ModGodModeBatchRange) -> Vec<ModGodModeParams> {
        fn axis(a: (usize, usize, usize)) -> Vec<usize> {
            let (s, e, st) = a; if st == 0 || s == e { vec![s] } else { (s..=e).step_by(st).collect() }
        }
        let n1s = axis(r.n1);
        let n2s = axis(r.n2);
        let n3s = axis(r.n3);
        let mut v = Vec::with_capacity(n1s.len() * n2s.len() * n3s.len());
        for &a in &n1s { for &b in &n2s { for &c in &n3s {
            v.push(ModGodModeParams { n1: Some(a), n2: Some(b), n3: Some(c), mode: Some(r.mode), use_volume: Some(false) });
        }}}
        v
    }

    #[inline]
    fn fast_cap() -> i32 { MGM_RING_KCAP }

    #[inline]
    fn fast_block_x() -> u32 { 64 }

    #[inline]
    fn fast_shared_bytes(block_x: u32) -> usize {
        // Per-thread bytes: (2*f32 + 2*i32 + 1*i8) * KCAP = 17 * KCAP
        let cap = Self::fast_cap() as usize;
        let per_thread = (2 * std::mem::size_of::<f32>()
            + 2 * std::mem::size_of::<i32>()
            + std::mem::size_of::<i8>()) * cap; // = 17*cap
        per_thread * (block_x as usize)
    }

    pub fn mod_god_mode_batch_dev(
        &self,
        high: &[f32], low: &[f32], close: &[f32], volume: Option<&[f32]>,
        sweep: &ModGodModeBatchRange,
    ) -> Result<CudaModGodModeBatchResult, CudaModGodModeError> {
        let n = close.len();
        if n == 0 { return Err(CudaModGodModeError::InvalidInput("empty inputs".into())); }
        if high.len() != n || low.len() != n { return Err(CudaModGodModeError::InvalidInput("H/L/C length mismatch".into())); }
        if let Some(v) = volume { if v.len() != n { return Err(CudaModGodModeError::InvalidInput("volume length mismatch".into())); } }
        let combos = Self::expand_range(sweep);
        if combos.is_empty() { return Err(CudaModGodModeError::InvalidInput("no parameter combinations".into())); }
        let rows = combos.len();

        // First valid index
        let mut first_valid = None;
        for (i, &v) in close.iter().enumerate() { if v.is_finite() { first_valid = Some(i); break; } }
        let first_valid = first_valid.ok_or_else(|| CudaModGodModeError::InvalidInput("all values are NaN".into()))?;

        // Build param arrays
        let mut n1s: Vec<i32> = Vec::with_capacity(rows);
        let mut n2s: Vec<i32> = Vec::with_capacity(rows);
        let mut n3s: Vec<i32> = Vec::with_capacity(rows);
        let mut modes: Vec<i32> = Vec::with_capacity(rows);
        for p in &combos {
            n1s.push(p.n1.unwrap() as i32);
            n2s.push(p.n2.unwrap() as i32);
            n3s.push(p.n3.unwrap() as i32);
            let m = match p.mode.unwrap() { ModGodModeMode::Godmode => 0, ModGodModeMode::Tradition => 1, ModGodModeMode::GodmodeMg => 2, ModGodModeMode::TraditionMg => 3 };
            modes.push(m);
        }

        // Partition by fast-cap for heuristic selection
        let cap = Self::fast_cap();
        let mut large_idxs: Vec<usize> = Vec::new();
        for i in 0..rows {
            let b = n2s[i];
            let c = n3s[i];
            let m = modes[i];
            if b > cap || c > cap || m >= 2 { large_idxs.push(i); }
        }

        // VRAM estimation (skip H/L when volume is unused)
        let use_vol = volume.is_some();
        let in_bytes = (if use_vol { 3 * n } else { 1 * n }) * std::mem::size_of::<f32>()
            + volume.map(|_| n * std::mem::size_of::<f32>()).unwrap_or(0);
        let param_bytes = 4 * rows * std::mem::size_of::<i32>();
        let out_bytes = 3 * rows * n * std::mem::size_of::<f32>();
        let required = in_bytes + param_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaModGodModeError::InvalidInput(format!("estimated device memory {:.2} MB exceeds free VRAM", required as f64 / (1024.0 * 1024.0))));
        }

        // Upload inputs (skip high/low if volume is unused)
        let d_close = DeviceBuffer::from_slice(close).map_err(|e| CudaModGodModeError::Cuda(e.to_string()))?;
        let d_high = if use_vol { Some(DeviceBuffer::from_slice(high).map_err(|e| CudaModGodModeError::Cuda(e.to_string()))?) } else { None };
        let d_low  = if use_vol { Some(DeviceBuffer::from_slice(low).map_err(|e| CudaModGodModeError::Cuda(e.to_string()))?) } else { None };
        let d_volume = if let Some(v) = volume { Some(DeviceBuffer::from_slice(v).map_err(|e| CudaModGodModeError::Cuda(e.to_string()))?) } else { None };
        let d_n1s = DeviceBuffer::from_slice(&n1s).map_err(|e| CudaModGodModeError::Cuda(e.to_string()))?;
        let d_n2s = DeviceBuffer::from_slice(&n2s).map_err(|e| CudaModGodModeError::Cuda(e.to_string()))?;
        let d_n3s = DeviceBuffer::from_slice(&n3s).map_err(|e| CudaModGodModeError::Cuda(e.to_string()))?;
        let d_modes = DeviceBuffer::from_slice(&modes).map_err(|e| CudaModGodModeError::Cuda(e.to_string()))?;

        let mut d_wt: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(rows * n) }.map_err(|e| CudaModGodModeError::Cuda(e.to_string()))?;
        let mut d_sig: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(rows * n) }.map_err(|e| CudaModGodModeError::Cuda(e.to_string()))?;
        let mut d_hist: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(rows * n) }.map_err(|e| CudaModGodModeError::Cuda(e.to_string()))?;
        // A) Fast shared-memory kernel over ALL rows
        {
            let func_fast = self.module.get_function("mod_god_mode_batch_f32_shared_fast")
                .map_err(|e| CudaModGodModeError::Cuda(e.to_string()))?;
            let mut block_x = Self::fast_block_x();
            let mut shmem_bytes = Self::fast_shared_bytes(block_x);
            let max_dyn_default: usize = 48 * 1024; // conservative default if opt-in not set
            while shmem_bytes > max_dyn_default && block_x > 1 {
                block_x /= 2;
                shmem_bytes = Self::fast_shared_bytes(block_x);
            }
            if !self.debug_logged && std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
                let grid_x_total = ((rows as u32) + block_x - 1) / block_x;
                eprintln!("[mod_god_mode] fast kernel: block_x={} grid_x={} shmem={} bytes", block_x, grid_x_total, shmem_bytes);
                unsafe { (*(self as *const _ as *mut CudaModGodMode)).debug_logged = true; }
            }
            let max_blocks = 65_535usize;
            let rows_per_launch = max_blocks.saturating_mul(block_x as usize);
            let mut launched = 0usize;
            while launched < rows {
                let chunk = std::cmp::min(rows - launched, rows_per_launch);
                let mut high_ptr = d_high.as_ref().map(|b| b.as_device_ptr().as_raw()).unwrap_or(0);
                let mut low_ptr  = d_low.as_ref().map(|b| b.as_device_ptr().as_raw()).unwrap_or(0);
                let mut close_ptr= d_close.as_device_ptr().as_raw();
                let mut vol_ptr  = d_volume.as_ref().map(|b| b.as_device_ptr().as_raw()).unwrap_or(0);
                let mut len_i    = n as i32;
                let mut first_i  = first_valid as i32;
                let mut rows_i   = chunk as i32;
                let mut n1_ptr = d_n1s.as_device_ptr().as_raw() + (launched * std::mem::size_of::<i32>()) as u64;
                let mut n2_ptr = d_n2s.as_device_ptr().as_raw() + (launched * std::mem::size_of::<i32>()) as u64;
                let mut n3_ptr = d_n3s.as_device_ptr().as_raw() + (launched * std::mem::size_of::<i32>()) as u64;
                let mut modes_ptr = d_modes.as_device_ptr().as_raw() + (launched * std::mem::size_of::<i32>()) as u64;
                let mut use_vol_i = if use_vol { 1i32 } else { 0i32 };
                let mut wt_ptr   = d_wt.as_device_ptr().as_raw()   + (launched * n * std::mem::size_of::<f32>()) as u64;
                let mut sig_ptr  = d_sig.as_device_ptr().as_raw()  + (launched * n * std::mem::size_of::<f32>()) as u64;
                let mut hist_ptr = d_hist.as_device_ptr().as_raw() + (launched * n * std::mem::size_of::<f32>()) as u64;
                let args: &mut [*mut c_void] = &mut [
                    &mut high_ptr as *mut _ as *mut c_void,
                    &mut low_ptr as *mut _ as *mut c_void,
                    &mut close_ptr as *mut _ as *mut c_void,
                    &mut vol_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut n1_ptr as *mut _ as *mut c_void,
                    &mut n2_ptr as *mut _ as *mut c_void,
                    &mut n3_ptr as *mut _ as *mut c_void,
                    &mut modes_ptr as *mut _ as *mut c_void,
                    &mut use_vol_i as *mut _ as *mut c_void,
                    &mut wt_ptr as *mut _ as *mut c_void,
                    &mut sig_ptr as *mut _ as *mut c_void,
                    &mut hist_ptr as *mut _ as *mut c_void,
                ];
                let grid_x = ((chunk as u32) + block_x - 1) / block_x;
                let grid: GridSize  = (grid_x.max(1), 1, 1).into();
                let block: BlockSize = (block_x, 1, 1).into();
                unsafe {
                    self.stream.launch(&func_fast, grid, block, (shmem_bytes as u32), args)
                        .map_err(|e| CudaModGodModeError::Cuda(e.to_string()))?;
                }
                launched += chunk;
            }
        }

        // B) Fallback kernel for large rows only (per-row launch)
        if !large_idxs.is_empty() {
            let func_fallback = self.module.get_function("mod_god_mode_batch_f32")
                .map_err(|e| CudaModGodModeError::Cuda(e.to_string()))?;
            let block_x: u32 = match self.policy.batch { BatchKernelPolicy::Auto => 128, BatchKernelPolicy::Plain { block_x } => block_x.max(32) };
            let block: BlockSize = (block_x, 1, 1).into();
            let grid:  GridSize  = (1, 1, 1).into();
            for &row_idx in &large_idxs {
                let mut high_ptr = d_high.as_ref().map(|b| b.as_device_ptr().as_raw()).unwrap_or(0);
                let mut low_ptr  = d_low.as_ref().map(|b| b.as_device_ptr().as_raw()).unwrap_or(0);
                let mut close_ptr= d_close.as_device_ptr().as_raw();
                let mut vol_ptr  = d_volume.as_ref().map(|b| b.as_device_ptr().as_raw()).unwrap_or(0);
                let mut len_i    = n as i32;
                let mut first_i  = first_valid as i32;
                let mut rows_i   = 1i32;
                let mut n1_ptr = d_n1s.as_device_ptr().as_raw() + (row_idx * std::mem::size_of::<i32>()) as u64;
                let mut n2_ptr = d_n2s.as_device_ptr().as_raw() + (row_idx * std::mem::size_of::<i32>()) as u64;
                let mut n3_ptr = d_n3s.as_device_ptr().as_raw() + (row_idx * std::mem::size_of::<i32>()) as u64;
                let mut modes_ptr = d_modes.as_device_ptr().as_raw() + (row_idx * std::mem::size_of::<i32>()) as u64;
                let mut use_vol_i = if use_vol { 1i32 } else { 0i32 };
                let mut wt_ptr   = d_wt.as_device_ptr().as_raw()   + (row_idx * n * std::mem::size_of::<f32>()) as u64;
                let mut sig_ptr  = d_sig.as_device_ptr().as_raw()  + (row_idx * n * std::mem::size_of::<f32>()) as u64;
                let mut hist_ptr = d_hist.as_device_ptr().as_raw() + (row_idx * n * std::mem::size_of::<f32>()) as u64;
                let args: &mut [*mut c_void] = &mut [
                    &mut high_ptr as *mut _ as *mut c_void,
                    &mut low_ptr as *mut _ as *mut c_void,
                    &mut close_ptr as *mut _ as *mut c_void,
                    &mut vol_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut n1_ptr as *mut _ as *mut c_void,
                    &mut n2_ptr as *mut _ as *mut c_void,
                    &mut n3_ptr as *mut _ as *mut c_void,
                    &mut modes_ptr as *mut _ as *mut c_void,
                    &mut use_vol_i as *mut _ as *mut c_void,
                    &mut wt_ptr as *mut _ as *mut c_void,
                    &mut sig_ptr as *mut _ as *mut c_void,
                    &mut hist_ptr as *mut _ as *mut c_void,
                ];
                unsafe {
                    self.stream.launch(&func_fallback, grid, block, 0, args)
                        .map_err(|e| CudaModGodModeError::Cuda(e.to_string()))?;
                }
            }
        }
        self.stream.synchronize().map_err(|e| CudaModGodModeError::Cuda(e.to_string()))?;

        let outputs = DeviceArrayF32Triplet {
            wt1: DeviceArrayF32 { buf: d_wt, rows, cols: n },
            wt2: DeviceArrayF32 { buf: d_sig, rows, cols: n },
            hist: DeviceArrayF32 { buf: d_hist, rows, cols: n },
        };
        Ok(CudaModGodModeBatchResult { outputs, combos })
    }

    pub fn mod_god_mode_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32], low_tm: &[f32], close_tm: &[f32], volume_tm: Option<&[f32]>,
        cols: usize, rows: usize, params: &ModGodModeParams,
    ) -> Result<DeviceArrayF32Triplet, CudaModGodModeError> {
        if cols == 0 || rows == 0 { return Err(CudaModGodModeError::InvalidInput("cols/rows must be > 0".into())); }
        let elems = cols.checked_mul(rows).ok_or_else(|| CudaModGodModeError::InvalidInput("cols*rows overflow".into()))?;
        if high_tm.len() != elems || low_tm.len() != elems || close_tm.len() != elems { return Err(CudaModGodModeError::InvalidInput("time-major inputs must be cols*rows".into())); }
        if let Some(v) = volume_tm { if v.len() != elems { return Err(CudaModGodModeError::InvalidInput("volume_tm length mismatch".into())); } }
        let n1 = params.n1.unwrap_or(17); let n2 = params.n2.unwrap_or(6); let n3 = params.n3.unwrap_or(4);
        let mode_i = match params.mode.unwrap_or(ModGodModeMode::TraditionMg) { ModGodModeMode::Godmode => 0, ModGodModeMode::Tradition => 1, ModGodModeMode::GodmodeMg => 2, ModGodModeMode::TraditionMg => 3 };
        let use_vol = params.use_volume.unwrap_or(false) && volume_tm.is_some();

        // Skip high/low if volume is unused
        let d_close = DeviceBuffer::from_slice(close_tm).map_err(|e| CudaModGodModeError::Cuda(e.to_string()))?;
        let d_high  = if use_vol { Some(DeviceBuffer::from_slice(high_tm).map_err(|e| CudaModGodModeError::Cuda(e.to_string()))?) } else { None };
        let d_low   = if use_vol { Some(DeviceBuffer::from_slice(low_tm).map_err(|e| CudaModGodModeError::Cuda(e.to_string()))?) } else { None };
        let d_vol   = if let Some(v) = volume_tm { Some(DeviceBuffer::from_slice(v).map_err(|e| CudaModGodModeError::Cuda(e.to_string()))?) } else { None };
        let mut d_wt: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.map_err(|e| CudaModGodModeError::Cuda(e.to_string()))?;
        let mut d_sig: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.map_err(|e| CudaModGodModeError::Cuda(e.to_string()))?;
        let mut d_hist: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }.map_err(|e| CudaModGodModeError::Cuda(e.to_string()))?;

        let func = self.module.get_function("mod_god_mode_many_series_one_param_time_major_f32").map_err(|e| CudaModGodModeError::Cuda(e.to_string()))?;
        // Kernel uses threadIdx.x == 0 only
        let bx: u32 = 1;
        let grid: GridSize = (cols as u32, 1, 1).into();
        let block: BlockSize = (bx, 1, 1).into();
        if !self.debug_logged && std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            eprintln!("[mod_god_mode] many-series policy selected: block_x={} grid.x={}", bx, cols);
            unsafe { (*(self as *const _ as *mut CudaModGodMode)).debug_logged = true; }
        }
        unsafe {
            let mut high_ptr = d_high.as_ref().map(|b| b.as_device_ptr().as_raw()).unwrap_or(0);
            let mut low_ptr = d_low.as_ref().map(|b| b.as_device_ptr().as_raw()).unwrap_or(0);
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut vol_ptr = d_vol.as_ref().map(|b| b.as_device_ptr().as_raw()).unwrap_or(0);
            let mut cols_i = cols as i32; let mut rows_i = rows as i32;
            let mut n1_i = n1 as i32; let mut n2_i = n2 as i32; let mut n3_i = n3 as i32; let mut mode_i32 = mode_i as i32; let mut use_vol_i = if use_vol {1i32} else {0i32};
            let mut wt_ptr = d_wt.as_device_ptr().as_raw(); let mut sig_ptr = d_sig.as_device_ptr().as_raw(); let mut hist_ptr = d_hist.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut vol_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut n1_i as *mut _ as *mut c_void,
                &mut n2_i as *mut _ as *mut c_void,
                &mut n3_i as *mut _ as *mut c_void,
                &mut mode_i32 as *mut _ as *mut c_void,
                &mut use_vol_i as *mut _ as *mut c_void,
                &mut wt_ptr as *mut _ as *mut c_void,
                &mut sig_ptr as *mut _ as *mut c_void,
                &mut hist_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args).map_err(|e| CudaModGodModeError::Cuda(e.to_string()))?;
        }
        self.stream.synchronize().map_err(|e| CudaModGodModeError::Cuda(e.to_string()))?;
        Ok(DeviceArrayF32Triplet {
            wt1: DeviceArrayF32 { buf: d_wt, rows, cols },
            wt2: DeviceArrayF32 { buf: d_sig, rows, cols },
            hist: DeviceArrayF32 { buf: d_hist, rows, cols },
        })
    }
}

// ---------- Benches (batch only; conservative sizes to avoid OOM) ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 500_000;
    const PARAM_SWEEP: usize = 100; // keep outputs moderate (3 * 100 * 500k * 4B ~ 57 MB)

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = 3 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = 3 * ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct MgBatchState {
        cuda: CudaModGodMode,
        high: Vec<f32>,
        low: Vec<f32>,
        close: Vec<f32>,
        sweep: ModGodModeBatchRange,
    }
    impl CudaBenchState for MgBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .mod_god_mode_batch_dev(&self.high, &self.low, &self.close, None, &self.sweep)
                .expect("mod_god_mode batch");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaModGodMode::new(0).expect("cuda mgm");
        let close = gen_series(ONE_SERIES_LEN);
        let mut high = close.clone();
        let mut low = close.clone();
        for i in 0..ONE_SERIES_LEN { high[i] += 0.5; low[i] -= 0.5; }
        let sweep = ModGodModeBatchRange { n1: (10, 10 + PARAM_SWEEP - 1, 1), n2: (6, 6, 0), n3: (4, 4, 0), mode: ModGodModeMode::TraditionMg };
        Box::new(MgBatchState { cuda, high, low, close, sweep })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "mod_god_mode",
            "one_series_many_params",
            "mod_god_mode_cuda_batch_dev",
            "500k_x_100",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}
