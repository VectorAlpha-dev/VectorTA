#![cfg(feature = "cuda")]

//! CUDA wrapper for Vortex Indicator (VI)
//!
//! Mirrors ALMA/CWMA patterns:
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/vi_kernel.ptx")) with DetermineTargetFromContext + O2 and fallbacks
//! - NON_BLOCKING stream
//! - Batch (one series × many params) uses host-precomputed prefix sums (TR/VP/VM), shared across rows
//! - Many-series (time-major, one param) also uses host-precomputed prefix sums per series
//! - Warmup/NaN and first_valid semantics match scalar exactly

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::vi::{ViBatchRange, ViParams};
use cust::context::{CacheConfig, Context};
use cust::device::Device;
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer, LockedBuffer, DeviceCopy};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaViError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaViError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaViError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaViError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl std::error::Error for CudaViError {}

// Pair of VRAM-resident arrays (for VI+ and VI-)
pub struct DeviceArrayF32Pair {
    pub a: DeviceArrayF32,
    pub b: DeviceArrayF32,
}
impl DeviceArrayF32Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.a.rows
    }
    #[inline]
    pub fn cols(&self) -> usize {
        self.a.cols
    }
    #[inline]
    pub fn rows(&self) -> usize {
        self.a.rows
    }
    #[inline]
    pub fn cols(&self) -> usize {
        self.a.cols
    }
}

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
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}
impl Default for ManySeriesKernelPolicy {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaViPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

pub struct CudaVi {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaViPolicy,
}

impl CudaVi {
    pub fn new(device_id: usize) -> Result<Self, CudaViError> {
        Self::new_with_policy(device_id, CudaViPolicy::default())
    }

    pub fn new_with_policy(device_id: usize, policy: CudaViPolicy) -> Result<Self, CudaViError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaViError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaViError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaViError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaViError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/vi_kernel.ptx"));
        let jit = [
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let jit = [
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, &jit)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaViError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaViError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaViError::Cuda(e.to_string()))?;
        let _ = cust::context::CurrentContext::set_cache_config(CacheConfig::PreferL1);

        Ok(Self {
            module,
            stream,
            _context: context,
            policy,
        })
        Ok(Self {
            module,
            stream,
            _context: context,
            policy,
        })
    }

    #[inline]
    fn mem_ok(bytes: usize, headroom: usize) -> bool {
        if env::var("CUDA_MEM_CHECK")
            .ok()
            .filter(|v| v == "0" || v.eq_ignore_ascii_case("false"))
            .is_some()
        {
        if env::var("CUDA_MEM_CHECK")
            .ok()
            .filter(|v| v == "0" || v.eq_ignore_ascii_case("false"))
            .is_some()
        {
            return true;
        }
        mem_get_info().map(|(free, _)| bytes.saturating_add(headroom) <= free).unwrap_or(true)
    }

    /// Heuristic: prefer pinned host memory for >= 1 MiB transfers
    #[inline(always)]
    fn use_pinned(bytes: usize) -> bool { bytes >= (1 << 20) }

    /// Upload a host slice to device with a safe and fast path:
    /// - small: DeviceBuffer::from_slice (synchronous)
    /// - large: LockedBuffer (pinned) + DeviceBuffer::uninitialized + copy_from (synchronous)
    /// Keeps bandwidth benefits of pinned memory without async lifetime hazards.
    fn h2d_upload<T: DeviceCopy>(&self, src: &[T]) -> Result<DeviceBuffer<T>, CudaViError> {
        let bytes = src.len() * std::mem::size_of::<T>();
        if Self::use_pinned(bytes) {
            let h = LockedBuffer::from_slice(src).map_err(|e| CudaViError::Cuda(e.to_string()))?;
            let mut d = unsafe { DeviceBuffer::<T>::uninitialized(src.len()) }
                .map_err(|e| CudaViError::Cuda(e.to_string()))?;
            d.copy_from(&h).map_err(|e| CudaViError::Cuda(e.to_string()))?;
            Ok(d)
        } else {
            DeviceBuffer::from_slice(src).map_err(|e| CudaViError::Cuda(e.to_string()))
        }
    }

    /// Ask the driver for an occupancy-friendly (grid, block), then clamp to work size.
    #[inline]
    fn choose_launch_1d(&self, func: &Function, n_items: usize) -> (GridSize, BlockSize) {
        let (min_grid_suggest, block_suggest) = func
            .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
            .unwrap_or((0, 256));
        let block_x = block_suggest.clamp(64, 1024);
        let mut grid_x = ((n_items as u32) + block_x - 1) / block_x;
        if min_grid_suggest > 0 { grid_x = grid_x.max(min_grid_suggest); }
        ((grid_x.max(1), 1, 1).into(), (block_x, 1, 1).into())
    }

    // ---------------- Host prefix sums (single series) ----------------
    fn build_prefix_single(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
    ) -> Result<(usize, Vec<f32>, Vec<f32>, Vec<f32>), CudaViError> {
        if high.len() != low.len() || high.len() != close.len() {
            return Err(CudaViError::InvalidInput("length mismatch".into()));
        }
        let n = high.len();
        if n == 0 {
            return Err(CudaViError::InvalidInput("empty input".into()));
        }
        let first = (0..n)
            .find(|&i| high[i].is_finite() && low[i].is_finite() && close[i].is_finite())
        if n == 0 {
            return Err(CudaViError::InvalidInput("empty input".into()));
        }
        let first = (0..n)
            .find(|&i| high[i].is_finite() && low[i].is_finite() && close[i].is_finite())
            .ok_or_else(|| CudaViError::InvalidInput("all values NaN".into()))?;

        // Accumulate in f64 for numerical parity with scalar path; cast to f32 for device
        let mut pfx_tr64 = vec![0.0f64; n];
        let mut pfx_vp64 = vec![0.0f64; n];
        let mut pfx_vm64 = vec![0.0f64; n];
        // Seed at `first`
        pfx_tr64[first] = (high[first] - low[first]) as f64;
        pfx_vp64[first] = 0.0;
        pfx_vm64[first] = 0.0;
        let mut prev_h = high[first];
        let mut prev_l = low[first];
        let mut prev_c = close[first];
        for i in (first + 1)..n {
            let hi = high[i];
            let lo = low[i];
            let hl = hi - lo;
            let hc = (hi - prev_c).abs();
            let lc = (lo - prev_c).abs();
            let tr_i = hl.max(hc.max(lc)) as f64;
            let vp_i = (hi - prev_l).abs() as f64;
            let vm_i = (lo - prev_h).abs() as f64;
            pfx_tr64[i] = pfx_tr64[i - 1] + tr_i;
            pfx_vp64[i] = pfx_vp64[i - 1] + vp_i;
            pfx_vm64[i] = pfx_vm64[i - 1] + vm_i;
            prev_h = hi;
            prev_l = lo;
            prev_c = close[i];
        }
        let pfx_tr: Vec<f32> = pfx_tr64.into_iter().map(|v| v as f32).collect();
        let pfx_vp: Vec<f32> = pfx_vp64.into_iter().map(|v| v as f32).collect();
        let pfx_vm: Vec<f32> = pfx_vm64.into_iter().map(|v| v as f32).collect();
        Ok((first, pfx_tr, pfx_vp, pfx_vm))
    }

    // ---------------- Host prefix sums (time-major, many series) ----------------
    fn build_prefix_time_major(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
    ) -> Result<(Vec<i32>, Vec<f32>, Vec<f32>, Vec<f32>), CudaViError> {
        if high_tm.len() != low_tm.len() || high_tm.len() != close_tm.len() {
            return Err(CudaViError::InvalidInput("length mismatch".into()));
        }
        if cols == 0 || rows == 0 {
            return Err(CudaViError::InvalidInput("invalid dims".into()));
        }
        if high_tm.len() != cols * rows {
            return Err(CudaViError::InvalidInput(
                "dims do not match data length".into(),
            ));
        }
        if cols == 0 || rows == 0 {
            return Err(CudaViError::InvalidInput("invalid dims".into()));
        }
        if high_tm.len() != cols * rows {
            return Err(CudaViError::InvalidInput(
                "dims do not match data length".into(),
            ));
        }

        let mut first_valids = vec![-1i32; cols];
        let mut pfx_tr64 = vec![0.0f64; cols * rows];
        let mut pfx_vp64 = vec![0.0f64; cols * rows];
        let mut pfx_vm64 = vec![0.0f64; cols * rows];

        for s in 0..cols {
            // Find first valid row for this series
            let mut first = None;
            for r in 0..rows {
                let idx = r * cols + s;
                let h = high_tm[idx];
                let l = low_tm[idx];
                let c = close_tm[idx];
                if h.is_finite() && l.is_finite() && c.is_finite() {
                    first = Some(r);
                    break;
                }
                if h.is_finite() && l.is_finite() && c.is_finite() {
                    first = Some(r);
                    break;
                }
            }
            if let Some(fv) = first {
                first_valids[s] = fv as i32;
                // Seed at first
                let base = fv * cols + s;
                pfx_tr64[base] = (high_tm[base] - low_tm[base]) as f64;
                pfx_vp64[base] = 0.0;
                pfx_vm64[base] = 0.0;
                let mut prev_h = high_tm[base];
                let mut prev_l = low_tm[base];
                let mut prev_c = close_tm[base];
                for r in (fv + 1)..rows {
                    let idx = r * cols + s;
                    let hi = high_tm[idx];
                    let lo = low_tm[idx];
                    let hl = hi - lo;
                    let hc = (hi - prev_c).abs();
                    let lc = (lo - prev_c).abs();
                    let tr_i = hl.max(hc.max(lc));
                    let vp_i = (hi - prev_l).abs();
                    let vm_i = (lo - prev_h).abs();
                    pfx_tr64[idx] = pfx_tr64[idx - cols] + tr_i as f64;
                    pfx_vp64[idx] = pfx_vp64[idx - cols] + vp_i as f64;
                    pfx_vm64[idx] = pfx_vm64[idx - cols] + vm_i as f64;
                    prev_h = hi;
                    prev_l = lo;
                    prev_c = close_tm[idx];
                }
            } else {
                // all NaN → mark invalid, leave prefixes at 0
                first_valids[s] = -1;
            }
        }
        let pfx_tr: Vec<f32> = pfx_tr64.into_iter().map(|v| v as f32).collect();
        let pfx_vp: Vec<f32> = pfx_vp64.into_iter().map(|v| v as f32).collect();
        let pfx_vm: Vec<f32> = pfx_vm64.into_iter().map(|v| v as f32).collect();
        Ok((first_valids, pfx_tr, pfx_vp, pfx_vm))
    }

    // ---------------- Batch entry ----------------
    pub fn vi_batch_dev(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        close_f32: &[f32],
        sweep: &ViBatchRange,
    ) -> Result<(DeviceArrayF32Pair, Vec<ViParams>), CudaViError> {
        if high_f32.len() != low_f32.len() || high_f32.len() != close_f32.len() {
            return Err(CudaViError::InvalidInput("length mismatch".into()));
        }
        let len = high_f32.len();
        if len == 0 {
            return Err(CudaViError::InvalidInput("empty input".into()));
        }
        if len == 0 {
            return Err(CudaViError::InvalidInput("empty input".into()));
        }

        fn expand_grid_local(r: &ViBatchRange) -> Vec<ViParams> {
            let (start, end, step) = r.period;
            if step == 0 || start == end {
                return vec![ViParams {
                    period: Some(start),
                }];
                return vec![ViParams {
                    period: Some(start),
                }];
            }
            (start..=end)
                .step_by(step)
                .map(|p| ViParams { period: Some(p) })
                .collect()
            (start..=end)
                .step_by(step)
                .map(|p| ViParams { period: Some(p) })
                .collect()
        }
        let combos = expand_grid_local(sweep);
        if combos.is_empty() {
            return Err(CudaViError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        if combos.is_empty() {
            return Err(CudaViError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();

        let (first_valid, pfx_tr, pfx_vp, pfx_vm) =
            self.build_prefix_single(high_f32, low_f32, close_f32)?;
        let (first_valid, pfx_tr, pfx_vp, pfx_vm) =
            self.build_prefix_single(high_f32, low_f32, close_f32)?;
        if len - first_valid < max_p {
            return Err(CudaViError::InvalidInput(
                "insufficient valid data after first_valid".into(),
            ));
            return Err(CudaViError::InvalidInput(
                "insufficient valid data after first_valid".into(),
            ));
        }

        // VRAM estimate (inputs already host-only; device: 3*pfx + periods + 2*out)
        let rows = combos.len();
        let bytes = 3 * len * std::mem::size_of::<f32>()
            + rows * std::mem::size_of::<i32>()
            + 2 * rows * len * std::mem::size_of::<f32>();
        let headroom = env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        let headroom = env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        if !Self::mem_ok(bytes, headroom) {
            return Err(CudaViError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                bytes as f64 / (1024.0 * 1024.0)
            )));
            return Err(CudaViError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                bytes as f64 / (1024.0 * 1024.0)
            )));
        }

        // Upload inputs (safe pinned-or-pageable synchronous path)
        let d_tr = self.h2d_upload(&pfx_tr)?;
        let d_vp = self.h2d_upload(&pfx_vp)?;
        let d_vm = self.h2d_upload(&pfx_vm)?;

        let periods_host: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let d_periods = self.h2d_upload(&periods_host)?;

        // Allocate outputs
        let mut d_plus = unsafe { DeviceBuffer::<f32>::uninitialized(rows * len) }
            .map_err(|e| CudaViError::Cuda(e.to_string()))?;
        let mut d_minus = unsafe { DeviceBuffer::<f32>::uninitialized(rows * len) }
            .map_err(|e| CudaViError::Cuda(e.to_string()))?;
        let mut d_plus = unsafe { DeviceBuffer::<f32>::uninitialized(rows * len) }
            .map_err(|e| CudaViError::Cuda(e.to_string()))?;
        let mut d_minus = unsafe { DeviceBuffer::<f32>::uninitialized(rows * len) }
            .map_err(|e| CudaViError::Cuda(e.to_string()))?;

        // Launch kernel
        let mut func: Function = self
            .module
            .get_function("vi_batch_f32")
            .map_err(|e| CudaViError::Cuda(e.to_string()))?;
        let mut func: Function = self
            .module
            .get_function("vi_batch_f32")
            .map_err(|e| CudaViError::Cuda(e.to_string()))?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);
        let (grid, block) = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => {
                let bx = block_x.max(32).min(1024);
                let gx = ((rows as u32) + bx - 1) / bx;
                ((gx.max(1), 1, 1).into(), (bx, 1, 1).into())
            }
            _ => self.choose_launch_1d(&func, rows),
        };

        unsafe {
            let mut tr_ptr = d_tr.as_device_ptr().as_raw();
            let mut vp_ptr = d_vp.as_device_ptr().as_raw();
            let mut vm_ptr = d_vm.as_device_ptr().as_raw();
            let mut pr_ptr = d_periods.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut rows_i = rows as i32;
            let mut first_i = first_valid as i32;
            let mut plus_ptr = d_plus.as_device_ptr().as_raw();
            let mut plus_ptr = d_plus.as_device_ptr().as_raw();
            let mut minus_ptr = d_minus.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut tr_ptr as *mut _ as *mut c_void,
                &mut vp_ptr as *mut _ as *mut c_void,
                &mut vm_ptr as *mut _ as *mut c_void,
                &mut pr_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut plus_ptr as *mut _ as *mut c_void,
                &mut minus_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaViError::Cuda(e.to_string()))?;
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaViError::Cuda(e.to_string()))?;
        }

        Ok((
            DeviceArrayF32Pair {
                a: DeviceArrayF32 {
                    buf: d_plus,
                    rows,
                    cols: len,
                },
                b: DeviceArrayF32 {
                    buf: d_minus,
                    rows,
                    cols: len,
                },
            },
            combos,
        ))
        Ok((
            DeviceArrayF32Pair {
                a: DeviceArrayF32 {
                    buf: d_plus,
                    rows,
                    cols: len,
                },
                b: DeviceArrayF32 {
                    buf: d_minus,
                    rows,
                    cols: len,
                },
            },
            combos,
        ))
    }

    // ---------------- Many-series (time-major) entry ----------------
    pub fn vi_many_series_one_param_time_major_dev(
        &self,
        high_tm_f32: &[f32],
        low_tm_f32: &[f32],
        close_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &ViParams,
    ) -> Result<DeviceArrayF32Pair, CudaViError> {
        if cols == 0 || rows == 0 {
            return Err(CudaViError::InvalidInput("invalid dims".into()));
        }
        if high_tm_f32.len() != cols * rows
            || low_tm_f32.len() != cols * rows
            || close_tm_f32.len() != cols * rows
        {
        if cols == 0 || rows == 0 {
            return Err(CudaViError::InvalidInput("invalid dims".into()));
        }
        if high_tm_f32.len() != cols * rows
            || low_tm_f32.len() != cols * rows
            || close_tm_f32.len() != cols * rows
        {
            return Err(CudaViError::InvalidInput("dims do not match data".into()));
        }
        let period = params.period.unwrap_or(14);
        if period == 0 || period > rows {
            return Err(CudaViError::InvalidInput("invalid period".into()));
        }
        if period == 0 || period > rows {
            return Err(CudaViError::InvalidInput("invalid period".into()));
        }

        let (first_valids, pfx_tr, pfx_vp, pfx_vm) =
            self.build_prefix_time_major(high_tm_f32, low_tm_f32, close_tm_f32, cols, rows)?;
        let (first_valids, pfx_tr, pfx_vp, pfx_vm) =
            self.build_prefix_time_major(high_tm_f32, low_tm_f32, close_tm_f32, cols, rows)?;
        // Validate sufficient tail per series; if any series lacks enough tail, we still run and kernel will NaN-fill

        // VRAM estimate (3 * rows*cols + first_valids + 2*out)
        let n = rows * cols;
        let bytes = 3 * n * std::mem::size_of::<f32>()
            + cols * std::mem::size_of::<i32>()
            + 2 * n * std::mem::size_of::<f32>();
        let headroom = env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        let headroom = env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        if !Self::mem_ok(bytes, headroom) {
            return Err(CudaViError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                bytes as f64 / (1024.0 * 1024.0)
            )));
            return Err(CudaViError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                bytes as f64 / (1024.0 * 1024.0)
            )));
        }

        // Upload inputs (safe pinned-or-pageable synchronous path)
        let d_tr = self.h2d_upload(&pfx_tr)?;
        let d_vp = self.h2d_upload(&pfx_vp)?;
        let d_vm = self.h2d_upload(&pfx_vm)?;
        let d_first = self.h2d_upload(&first_valids)?;

        // Allocate outputs
        let mut d_plus = unsafe { DeviceBuffer::<f32>::uninitialized(n) }
            .map_err(|e| CudaViError::Cuda(e.to_string()))?;
        let mut d_minus = unsafe { DeviceBuffer::<f32>::uninitialized(n) }
            .map_err(|e| CudaViError::Cuda(e.to_string()))?;
        let mut d_plus = unsafe { DeviceBuffer::<f32>::uninitialized(n) }
            .map_err(|e| CudaViError::Cuda(e.to_string()))?;
        let mut d_minus = unsafe { DeviceBuffer::<f32>::uninitialized(n) }
            .map_err(|e| CudaViError::Cuda(e.to_string()))?;

        // Launch kernel
        let mut func: Function = self
            .module
            .get_function("vi_many_series_one_param_f32")
            .map_err(|e| CudaViError::Cuda(e.to_string()))?;
        let mut func: Function = self
            .module
            .get_function("vi_many_series_one_param_f32")
            .map_err(|e| CudaViError::Cuda(e.to_string()))?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);
        let (grid, block) = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => {
                let bx = block_x.max(32).min(1024);
                let gx = ((cols as u32) + bx - 1) / bx;
                ((gx.max(1), 1, 1).into(), (bx, 1, 1).into())
            }
            _ => self.choose_launch_1d(&func, cols),
        };

        unsafe {
            let mut tr_ptr = d_tr.as_device_ptr().as_raw();
            let mut vp_ptr = d_vp.as_device_ptr().as_raw();
            let mut vm_ptr = d_vm.as_device_ptr().as_raw();
            let mut fv_ptr = d_first.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut p_i = period as i32;
            let mut plus_ptr = d_plus.as_device_ptr().as_raw();
            let mut plus_ptr = d_plus.as_device_ptr().as_raw();
            let mut minus_ptr = d_minus.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut tr_ptr as *mut _ as *mut c_void,
                &mut vp_ptr as *mut _ as *mut c_void,
                &mut vm_ptr as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut p_i as *mut _ as *mut c_void,
                &mut plus_ptr as *mut _ as *mut c_void,
                &mut minus_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaViError::Cuda(e.to_string()))?;
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaViError::Cuda(e.to_string()))?;
        }

        Ok(DeviceArrayF32Pair {
            a: DeviceArrayF32 {
                buf: d_plus,
                rows,
                cols,
            },
            b: DeviceArrayF32 {
                buf: d_minus,
                rows,
                cols,
            },
            a: DeviceArrayF32 {
                buf: d_plus,
                rows,
                cols,
            },
            b: DeviceArrayF32 {
                buf: d_minus,
                rows,
                cols,
            },
        })
    }
}

// ---------------- Benches registration ----------------
pub mod benches {
    use super::*;
    use crate::cuda::{CudaBenchScenario, CudaBenchState};

    struct BatchState {
        cuda: CudaVi,
        d: DeviceArrayF32Pair,
        _combos: Vec<ViParams>,
    }
    impl CudaBenchState for BatchState {
        fn launch(&mut self) {
            let _ = self.cuda.stream.synchronize();
        }
    }
    struct BatchState {
        cuda: CudaVi,
        d: DeviceArrayF32Pair,
        _combos: Vec<ViParams>,
    }
    impl CudaBenchState for BatchState {
        fn launch(&mut self) {
            let _ = self.cuda.stream.synchronize();
        }
    }

    struct ManyState {
        cuda: CudaVi,
        d: DeviceArrayF32Pair,
    }
    impl CudaBenchState for ManyState {
        fn launch(&mut self) {
            let _ = self.cuda.stream.synchronize();
        }
    }
    struct ManyState {
        cuda: CudaVi,
        d: DeviceArrayF32Pair,
    }
    impl CudaBenchState for ManyState {
        fn launch(&mut self) {
            let _ = self.cuda.stream.synchronize();
        }
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        // Keep sizes reasonable to avoid OOM across varied GPUs
        let mut v = Vec::new();
        v.push(CudaBenchScenario::new(
            "vi",
            "one_series_many_params",
            "vi_batch",
            "vi_batch/len100k/periods[7..35]",
            || {
                let len = 100_000usize;
                let mut h = vec![f32::NAN; len];
                let mut l = vec![f32::NAN; len];
                let mut c = vec![f32::NAN; len];
                for i in 5..len {
                    let x = i as f32;
                    h[i] = x.sin() + 0.01 * x;
                    l[i] = h[i] - 0.5;
                    c[i] = (0.5 * x).cos() + 0.02 * x;
                }
                for i in 5..len {
                    let x = i as f32;
                    h[i] = x.sin() + 0.01 * x;
                    l[i] = h[i] - 0.5;
                    c[i] = (0.5 * x).cos() + 0.02 * x;
                }
                let sweep = ViBatchRange { period: (7, 35, 2) };
                let cuda = CudaVi::new(0).unwrap();
                let (d, combos) = cuda.vi_batch_dev(&h, &l, &c, &sweep).unwrap();
                Box::new(BatchState {
                    cuda,
                    d,
                    _combos: combos,
                }) as Box<dyn CudaBenchState>
                Box::new(BatchState {
                    cuda,
                    d,
                    _combos: combos,
                }) as Box<dyn CudaBenchState>
            },
        ));

        v.push(CudaBenchScenario::new(
            "vi",
            "many_series_one_param",
            "vi_many",
            "vi_many/rows65536xcols64/period14",
            || {
                let rows = 65_536usize;
                let cols = 64usize;
                let mut h = vec![f32::NAN; rows * cols];
                let mut l = vec![f32::NAN; rows * cols];
                let mut c = vec![f32::NAN; rows * cols];
                for s in 0..cols {
                    for r in s..rows {
                        let idx = r * cols + s;
                        let x = (r as f32) * 0.002 + (s as f32) * 0.01;
                        h[idx] = x.sin() + 0.01 * x;
                        l[idx] = h[idx] - 0.4;
                        c[idx] = 0.5 * x.cos() + 0.02 * x;
                    }
                }
                let rows = 65_536usize;
                let cols = 64usize;
                let mut h = vec![f32::NAN; rows * cols];
                let mut l = vec![f32::NAN; rows * cols];
                let mut c = vec![f32::NAN; rows * cols];
                for s in 0..cols {
                    for r in s..rows {
                        let idx = r * cols + s;
                        let x = (r as f32) * 0.002 + (s as f32) * 0.01;
                        h[idx] = x.sin() + 0.01 * x;
                        l[idx] = h[idx] - 0.4;
                        c[idx] = 0.5 * x.cos() + 0.02 * x;
                    }
                }
                let cuda = CudaVi::new(0).unwrap();
                let params = ViParams { period: Some(14) };
                let d = cuda
                    .vi_many_series_one_param_time_major_dev(&h, &l, &c, cols, rows, &params)
                    .unwrap();
                let d = cuda
                    .vi_many_series_one_param_time_major_dev(&h, &l, &c, cols, rows, &params)
                    .unwrap();
                Box::new(ManyState { cuda, d }) as Box<dyn CudaBenchState>
            },
        ));

        v
    }
}
