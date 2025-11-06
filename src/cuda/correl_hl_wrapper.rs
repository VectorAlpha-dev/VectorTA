//! CUDA wrapper for CORREL_HL (Pearson correlation of High vs Low).
//!
//! Parity goals (per Agents Guide):
//! - API and behavior match ALMA-style wrappers
//! - Non-blocking stream; PTX loaded with DetermineTargetFromContext and O2 fallback
//! - VRAM estimation + ~64MB headroom; grid-y chunking to <= 65_535
//! - Batch builds prefixes as DS (float2) in pinned memory and uploads once
//! - Many-series×one-param uses time-major scan with O(1) updates per series

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::correl_hl::{CorrelHlBatchRange, CorrelHlParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer, DeviceCopy};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaCorrelHlError {
    Cuda(String),
    InvalidInput(String),
}
impl fmt::Display for CudaCorrelHlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaCorrelHlError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaCorrelHlError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaCorrelHlError {}

// Host-side POD that matches CUDA `float2` layout for DS prefixes
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Float2 { pub hi: f32, pub lo: f32 }

// Safety: Float2 is POD and safe for device copies
unsafe impl DeviceCopy for Float2 {}

#[inline(always)]
fn pack_ds(v: f64) -> Float2 {
    // v ≈ hi + lo, with hi as nearest f32 and lo the residual
    let hi = v as f32;
    let lo = (v - (hi as f64)) as f32;
    Float2 { hi, lo }
}

#[derive(Clone, Copy, Debug, Default)]
pub enum BatchKernelPolicy {
    #[default]
    Auto,
    Plain {
        block_x: u32,
    },
}

#[derive(Clone, Copy, Debug, Default)]
pub enum ManySeriesKernelPolicy {
    #[default]
    Auto,
    OneD {
        block_x: u32,
    },
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaCorrelHlPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaCorrelHl {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaCorrelHlPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaCorrelHl {
    pub fn new(device_id: usize) -> Result<Self, CudaCorrelHlError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/correl_hl_kernel.ptx"));
        let module = Module::from_ptx(
            ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O2),
            ],
        )
        .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
        .or_else(|_| Module::from_ptx(ptx, &[]))
        .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaCorrelHlPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn policy(&self) -> &CudaCorrelHlPolicy { &self.policy }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> { self.last_batch }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> { self.last_many }
    pub fn set_policy(&mut self, policy: CudaCorrelHlPolicy) { self.policy = policy; }

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
    fn maybe_log_batch_debug(&self) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                if !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] correl_hl batch selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaCorrelHl)).debug_batch_logged = true; }
            }
        }
    }

    #[inline]
    fn maybe_log_many_debug(&self) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged {
            return;
        }
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                if !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] correl_hl many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaCorrelHl)).debug_many_logged = true;
                }
                unsafe {
                    (*(self as *const _ as *mut CudaCorrelHl)).debug_many_logged = true;
                }
            }
        }
    }

    fn expand_grid(range: &CorrelHlBatchRange) -> Vec<CorrelHlParams> {
        fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
            if step == 0 || start == end {
                return vec![start];
            }
            if step == 0 || start == end {
                return vec![start];
            }
            (start..=end).step_by(step).collect()
        }
        let periods = axis_usize(range.period);
        let mut v = Vec::with_capacity(periods.len());
        for &p in &periods {
            v.push(CorrelHlParams { period: Some(p) });
        }
        v
    }

    fn prepare_batch_inputs(
        high: &[f32],
        low: &[f32],
        sweep: &CorrelHlBatchRange,
    ) -> Result<(Vec<CorrelHlParams>, usize, usize), CudaCorrelHlError> {
        if high.len() != low.len() {
            return Err(CudaCorrelHlError::InvalidInput("length mismatch".into()));
        }
        if high.is_empty() {
            return Err(CudaCorrelHlError::InvalidInput("empty input".into()));
        }
        if high.len() != low.len() {
            return Err(CudaCorrelHlError::InvalidInput("length mismatch".into()));
        }
        if high.is_empty() {
            return Err(CudaCorrelHlError::InvalidInput("empty input".into()));
        }
        let len = high.len();
        let first_valid = high
            .iter()
            .zip(low.iter())
            .position(|(h, l)| !h.is_nan() && !l.is_nan())
            .ok_or_else(|| CudaCorrelHlError::InvalidInput("all values are NaN".into()))?;

        let combos = Self::expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaCorrelHlError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        for c in &combos {
            let p = c.period.unwrap_or(0);
            if p == 0 {
                return Err(CudaCorrelHlError::InvalidInput("period must be > 0".into()));
            }
            if p > len {
                return Err(CudaCorrelHlError::InvalidInput(
                    "period exceeds data length".into(),
                ));
            }
            if len - first_valid < p {
                return Err(CudaCorrelHlError::InvalidInput(
                    "not enough valid data".into(),
                ));
            }
        }
        Ok((combos, first_valid, len))
    }

    // Build DS (float2) prefixes directly into pinned memory to avoid pageable staging copies
    fn build_prefixes_ds_pinned(
        high: &[f32],
        low: &[f32],
    ) -> Result<
        (
            LockedBuffer<Float2>, // ps_h
            LockedBuffer<Float2>, // ps_h2
            LockedBuffer<Float2>, // ps_l
            LockedBuffer<Float2>, // ps_l2
            LockedBuffer<Float2>, // ps_hl
            LockedBuffer<i32>,    // ps_nan
        ),
        CudaCorrelHlError,
    > {
        let n = high.len();
        let mut ps_h   = unsafe { LockedBuffer::<Float2>::uninitialized(n + 1) }
            .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;
        let mut ps_h2  = unsafe { LockedBuffer::<Float2>::uninitialized(n + 1) }
            .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;
        let mut ps_l   = unsafe { LockedBuffer::<Float2>::uninitialized(n + 1) }
            .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;
        let mut ps_l2  = unsafe { LockedBuffer::<Float2>::uninitialized(n + 1) }
            .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;
        let mut ps_hl  = unsafe { LockedBuffer::<Float2>::uninitialized(n + 1) }
            .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;
        let mut ps_nan = unsafe { LockedBuffer::<i32>::uninitialized(n + 1) }
            .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;

        // prefix[0]
        ps_h.as_mut_slice()[0]  = Float2::default();
        ps_h2.as_mut_slice()[0] = Float2::default();
        ps_l.as_mut_slice()[0]  = Float2::default();
        ps_l2.as_mut_slice()[0] = Float2::default();
        ps_hl.as_mut_slice()[0] = Float2::default();
        ps_nan.as_mut_slice()[0] = 0;

        // running sums in f64 for accuracy; we store as DS per step
        let (mut sum_h, mut sum_l, mut sum_h2, mut sum_l2, mut sum_hl) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
        let mut nan = 0i32;
        for i in 0..n {
            let h = high[i];
            let l = low[i];
            if h.is_nan() || l.is_nan() {
                nan += 1;
                // carry previous DS values
                ps_h.as_mut_slice()[i + 1]  = ps_h.as_slice()[i];
                ps_h2.as_mut_slice()[i + 1] = ps_h2.as_slice()[i];
                ps_l.as_mut_slice()[i + 1]  = ps_l.as_slice()[i];
                ps_l2.as_mut_slice()[i + 1] = ps_l2.as_slice()[i];
                ps_hl.as_mut_slice()[i + 1] = ps_hl.as_slice()[i];
                ps_nan.as_mut_slice()[i + 1] = nan;
            } else {
                let hd = h as f64;
                let ld = l as f64;
                sum_h  += hd;
                sum_l  += ld;
                sum_h2 += hd * hd;
                sum_l2 += ld * ld;
                sum_hl += hd * ld;
                ps_h.as_mut_slice()[i + 1]  = pack_ds(sum_h);
                ps_h2.as_mut_slice()[i + 1] = pack_ds(sum_h2);
                ps_l.as_mut_slice()[i + 1]  = pack_ds(sum_l);
                ps_l2.as_mut_slice()[i + 1] = pack_ds(sum_l2);
                ps_hl.as_mut_slice()[i + 1] = pack_ds(sum_hl);
                ps_nan.as_mut_slice()[i + 1] = nan;
            }
        }

        Ok((ps_h, ps_h2, ps_l, ps_l2, ps_hl, ps_nan))
    }

    // Legacy f64 prefix builder retained for parity path
    fn build_prefixes_f64(high: &[f32], low: &[f32]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<i32>) {
        let n = high.len();
        let mut ps_h = vec![0.0f64; n + 1];
        let mut ps_h2 = vec![0.0f64; n + 1];
        let mut ps_l = vec![0.0f64; n + 1];
        let mut ps_l2 = vec![0.0f64; n + 1];
        let mut ps_hl = vec![0.0f64; n + 1];
        let mut ps_nan = vec![0i32; n + 1];
        let mut ps_h = vec![0.0f64; n + 1];
        let mut ps_h2 = vec![0.0f64; n + 1];
        let mut ps_l = vec![0.0f64; n + 1];
        let mut ps_l2 = vec![0.0f64; n + 1];
        let mut ps_hl = vec![0.0f64; n + 1];
        let mut ps_nan = vec![0i32; n + 1];
        for i in 0..n {
            let h = high[i];
            let l = low[i];
            let (ph, ph2, pl, pl2, phl) = (ps_h[i], ps_h2[i], ps_l[i], ps_l2[i], ps_hl[i]);
            if h.is_nan() || l.is_nan() {
                ps_h[i + 1] = ph;
                ps_h2[i + 1] = ph2;
                ps_l[i + 1] = pl;
                ps_l2[i + 1] = pl2;
                ps_hl[i + 1] = phl;
                ps_nan[i + 1] = ps_nan[i] + 1;
                ps_h[i + 1] = ph;
                ps_h2[i + 1] = ph2;
                ps_l[i + 1] = pl;
                ps_l2[i + 1] = pl2;
                ps_hl[i + 1] = phl;
                ps_nan[i + 1] = ps_nan[i] + 1;
            } else {
                let hd = h as f64;
                let ld = l as f64;
                ps_h[i + 1] = ph + hd;
                ps_h2[i + 1] = ph2 + hd * hd;
                ps_l[i + 1] = pl + ld;
                ps_l2[i + 1] = pl2 + ld * ld;
                ps_hl[i + 1] = phl + hd * ld;
                ps_nan[i + 1] = ps_nan[i];
                let hd = h as f64;
                let ld = l as f64;
                ps_h[i + 1] = ph + hd;
                ps_h2[i + 1] = ph2 + hd * hd;
                ps_l[i + 1] = pl + ld;
                ps_l2[i + 1] = pl2 + ld * ld;
                ps_hl[i + 1] = phl + hd * ld;
                ps_nan[i + 1] = ps_nan[i];
            }
        }
        (ps_h, ps_h2, ps_l, ps_l2, ps_hl, ps_nan)
    }

    fn launch_batch_ds(
        &self,
        d_ps_h: &DeviceBuffer<Float2>,
        d_ps_h2: &DeviceBuffer<Float2>,
        d_ps_l: &DeviceBuffer<Float2>,
        d_ps_l2: &DeviceBuffer<Float2>,
        d_ps_hl: &DeviceBuffer<Float2>,
        d_ps_nan: &DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        d_periods: &DeviceBuffer<i32>,
        combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaCorrelHlError> {
        let func = self
            .module
            .get_function("correl_hl_batch_f32ds")
            .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;

        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Auto => 256,
            BatchKernelPolicy::Plain { block_x } => block_x.max(64),
        };
        let grid_x: u32 = ((len as u32) + block_x - 1) / block_x;
        let grid_base: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe { (*(self as *const _ as *mut CudaCorrelHl)).last_batch = Some(BatchKernelSelected::Plain { block_x }); }

        // Chunk grid.y to <= 65_535
        let mut launched = 0usize;
        while launched < combos {
            let chunk = (combos - launched).min(65_535);
            let grid: GridSize = (grid_base.x, chunk as u32, 1).into();
            unsafe {
                let mut ps_h = d_ps_h.as_device_ptr().as_raw();
                let mut ps_h2 = d_ps_h2.as_device_ptr().as_raw();
                let mut ps_l = d_ps_l.as_device_ptr().as_raw();
                let mut ps_l2 = d_ps_l2.as_device_ptr().as_raw();
                let mut ps_hl = d_ps_hl.as_device_ptr().as_raw();
                let mut ps_nan = d_ps_nan.as_device_ptr().as_raw();
                let mut len_i = len as i32;
                let mut first_i = first_valid as i32;
                let mut periods = (d_periods.as_device_ptr().as_raw() + (launched as u64) * std::mem::size_of::<i32>() as u64) as u64;
                let mut n_chunk = chunk as i32;
                let mut out_ptr = (d_out.as_device_ptr().as_raw()
                    + (launched as u64) * (len as u64) * std::mem::size_of::<f32>() as u64) as u64;

                let args: &mut [*mut c_void] = &mut [
                    &mut ps_h as *mut _ as *mut c_void,
                    &mut ps_h2 as *mut _ as *mut c_void,
                    &mut ps_l as *mut _ as *mut c_void,
                    &mut ps_l2 as *mut _ as *mut c_void,
                    &mut ps_hl as *mut _ as *mut c_void,
                    &mut ps_nan as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut periods as *mut _ as *mut c_void,
                    &mut n_chunk as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;
            }
            launched += chunk;
        }
        self.maybe_log_batch_debug();
        Ok(())
    }

    fn launch_batch_dp(
        &self,
        d_ps_h: &DeviceBuffer<Float2>,
        d_ps_h2: &DeviceBuffer<Float2>,
        d_ps_l: &DeviceBuffer<Float2>,
        d_ps_l2: &DeviceBuffer<Float2>,
        d_ps_hl: &DeviceBuffer<Float2>,
        d_ps_nan: &DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        d_periods: &DeviceBuffer<i32>,
        combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaCorrelHlError> {
        let func = self
            .module
            .get_function("correl_hl_batch_f32ds")
            .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;

        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Auto => 256,
            BatchKernelPolicy::Plain { block_x } => block_x.max(64),
        };
        let grid_x: u32 = ((len as u32) + block_x - 1) / block_x;
        let grid_base: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe { (*(self as *const _ as *mut CudaCorrelHl)).last_batch = Some(BatchKernelSelected::Plain { block_x }); }

        // Chunk grid.y to <= 65_535
        let mut launched = 0usize;
        while launched < combos {
            let chunk = (combos - launched).min(65_535);
            let grid: GridSize = (grid_base.x, chunk as u32, 1).into();
            unsafe {
                let mut ps_h = d_ps_h.as_device_ptr().as_raw();
                let mut ps_h2 = d_ps_h2.as_device_ptr().as_raw();
                let mut ps_l = d_ps_l.as_device_ptr().as_raw();
                let mut ps_l2 = d_ps_l2.as_device_ptr().as_raw();
                let mut ps_hl = d_ps_hl.as_device_ptr().as_raw();
                let mut ps_nan = d_ps_nan.as_device_ptr().as_raw();
                let mut len_i = len as i32;
                let mut first_i = first_valid as i32;
                let mut periods = (d_periods.as_device_ptr().as_raw() + (launched as u64) * std::mem::size_of::<i32>() as u64) as u64;
                let mut n_chunk = chunk as i32;
                let mut out_ptr = (d_out.as_device_ptr().as_raw()
                    + (launched as u64) * (len as u64) * std::mem::size_of::<f32>() as u64) as u64;

                let args: &mut [*mut c_void] = &mut [
                    &mut ps_h as *mut _ as *mut c_void,
                    &mut ps_h2 as *mut _ as *mut c_void,
                    &mut ps_l as *mut _ as *mut c_void,
                    &mut ps_l2 as *mut _ as *mut c_void,
                    &mut ps_hl as *mut _ as *mut c_void,
                    &mut ps_nan as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut periods as *mut _ as *mut c_void,
                    &mut n_chunk as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;
            }
            launched += chunk;
        }
        self.maybe_log_batch_debug();
        Ok(())
    }

    #[inline]
    fn select_batch_impl(len: usize, combos: usize) -> bool {
        // Heuristic: use DS for larger problems; otherwise prefer legacy DP to match references.
        // Threshold chosen to keep tests stable while enabling DS for bench-sized inputs.
        let work = len.saturating_mul(combos);
        work >= 5_000_000
    }

    pub fn correl_hl_batch_dev(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        sweep: &CorrelHlBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<CorrelHlParams>), CudaCorrelHlError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(high_f32, low_f32, sweep)?;

        // VRAM sizing (rough upper bound):
        // - prefixes: 5 * (len+1) * f64 + (len+1) * i32
        // - periods: combos * i32
        // - outputs: combos * len * f32
        let bytes_prefix = 5 * (len + 1) * 8 + (len + 1) * 4;
        let bytes_periods = combos.len() * 4;
        let bytes_out = combos.len() * len * 4;
        let required = bytes_prefix + bytes_periods + bytes_out;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaCorrelHlError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        // Periods + output allocations
        let periods: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let d_periods = unsafe { DeviceBuffer::from_slice_async(&periods, &self.stream) }
            .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;

        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(combos.len() * len, &self.stream)
                .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?
        };

        // Build DS prefixes directly in pinned host buffers
        let use_ds = Self::select_batch_impl(len, combos.len());
        if use_ds {
            let (ps_h, ps_h2, ps_l, ps_l2, ps_hl, ps_nan) = Self::build_prefixes_ds_pinned(high_f32, low_f32)?;
            // Upload from pinned buffers to device asynchronously
            let d_ps_h: DeviceBuffer<Float2>  = unsafe { DeviceBuffer::from_slice_async(ps_h.as_slice(),  &self.stream) }
                .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;
            let d_ps_h2: DeviceBuffer<Float2> = unsafe { DeviceBuffer::from_slice_async(ps_h2.as_slice(), &self.stream) }
                .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;
            let d_ps_l: DeviceBuffer<Float2>  = unsafe { DeviceBuffer::from_slice_async(ps_l.as_slice(),  &self.stream) }
                .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;
            let d_ps_l2: DeviceBuffer<Float2> = unsafe { DeviceBuffer::from_slice_async(ps_l2.as_slice(), &self.stream) }
                .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;
            let d_ps_hl: DeviceBuffer<Float2> = unsafe { DeviceBuffer::from_slice_async(ps_hl.as_slice(), &self.stream) }
                .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;
            let d_ps_nan: DeviceBuffer<i32>   = unsafe { DeviceBuffer::from_slice_async(ps_nan.as_slice(), &self.stream) }
                .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;

            self.launch_batch_ds(
                &d_ps_h,
                &d_ps_h2,
                &d_ps_l,
                &d_ps_l2,
                &d_ps_hl,
                &d_ps_nan,
                len,
                first_valid,
                &d_periods,
                combos.len(),
                &mut d_out,
            )?;
        } else {
            // Legacy f64 prefixes for maximal parity with CPU baseline on smaller inputs
            let (ps_h, ps_h2, ps_l, ps_l2, ps_hl, ps_nan) = Self::build_prefixes_f64(high_f32, low_f32);
            // Convert to DS format required by kernels
            let ps_h_ds: Vec<Float2>  = ps_h.iter().map(|&x| pack_ds(x)).collect();
            let ps_h2_ds: Vec<Float2> = ps_h2.iter().map(|&x| pack_ds(x)).collect();
            let ps_l_ds: Vec<Float2>  = ps_l.iter().map(|&x| pack_ds(x)).collect();
            let ps_l2_ds: Vec<Float2> = ps_l2.iter().map(|&x| pack_ds(x)).collect();
            let ps_hl_ds: Vec<Float2> = ps_hl.iter().map(|&x| pack_ds(x)).collect();

            let d_ps_h: DeviceBuffer<Float2>  = unsafe { DeviceBuffer::from_slice_async(ps_h_ds.as_slice(),  &self.stream) }
                .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;
            let d_ps_h2: DeviceBuffer<Float2> = unsafe { DeviceBuffer::from_slice_async(ps_h2_ds.as_slice(), &self.stream) }
                .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;
            let d_ps_l: DeviceBuffer<Float2>  = unsafe { DeviceBuffer::from_slice_async(ps_l_ds.as_slice(),  &self.stream) }
                .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;
            let d_ps_l2: DeviceBuffer<Float2> = unsafe { DeviceBuffer::from_slice_async(ps_l2_ds.as_slice(), &self.stream) }
                .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;
            let d_ps_hl: DeviceBuffer<Float2> = unsafe { DeviceBuffer::from_slice_async(ps_hl_ds.as_slice(), &self.stream) }
                .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;
            let d_ps_nan: DeviceBuffer<i32>   = unsafe { DeviceBuffer::from_slice_async(&ps_nan, &self.stream) }
                .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;

            self.launch_batch_dp(
                &d_ps_h,
                &d_ps_h2,
                &d_ps_l,
                &d_ps_l2,
                &d_ps_hl,
                &d_ps_nan,
                len,
                first_valid,
                &d_periods,
                combos.len(),
                &mut d_out,
            )?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;

        Ok((
            DeviceArrayF32 {
                buf: d_out,
                rows: combos.len(),
                cols: len,
            },
            combos,
        ))
    }

    pub fn correl_hl_many_series_one_param_time_major_dev(
        &self,
        high_tm_f32: &[f32],
        low_tm_f32: &[f32],
        cols: usize, // num_series
        rows: usize, // series_len
        period: usize,
    ) -> Result<DeviceArrayF32, CudaCorrelHlError> {
        if high_tm_f32.len() != low_tm_f32.len() {
            return Err(CudaCorrelHlError::InvalidInput("length mismatch".into()));
        }
        if high_tm_f32.len() != rows * cols {
            return Err(CudaCorrelHlError::InvalidInput("shape mismatch".into()));
        }
        if period == 0 || period > rows {
            return Err(CudaCorrelHlError::InvalidInput("invalid period".into()));
        }
        if period == 0 || period > rows {
            return Err(CudaCorrelHlError::InvalidInput("invalid period".into()));
        }

        // Compute per-series first_valid index
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = -1i32;
            for t in 0..rows {
                let h = high_tm_f32[t * cols + s];
                let l = low_tm_f32[t * cols + s];
                if !h.is_nan() && !l.is_nan() {
                    fv = t as i32;
                    break;
                }
                if !h.is_nan() && !l.is_nan() {
                    fv = t as i32;
                    break;
                }
            }
            first_valids[s] = fv;
        }

        let d_high = unsafe { DeviceBuffer::from_slice_async(high_tm_f32, &self.stream) }
            .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(low_tm_f32, &self.stream) }
            .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;
        let d_first = unsafe { DeviceBuffer::from_slice_async(&first_valids, &self.stream) }
            .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;

        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(rows * cols, &self.stream)
                .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?
        };

        let func = self
            .module
            .get_function("correl_hl_many_series_one_param_f32")
            .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;

        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 128,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(64),
        };
        let grid_x: u32 = cols as u32;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            (*(self as *const _ as *mut CudaCorrelHl)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        unsafe {
            (*(self as *const _ as *mut CudaCorrelHl)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        unsafe {
            let mut high_ptr = d_high.as_device_ptr().as_raw();
            let mut low_ptr = d_low.as_device_ptr().as_raw();
            let mut first_ptr = d_first.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaCorrelHlError::Cuda(e.to_string()))?;
        self.maybe_log_many_debug();
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }
}

// ------------------------ Benches (optional helpers) ------------------------

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::correl_hl::CorrelHlBatchRange;

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = 2 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let prefix_bytes = 5 * (ONE_SERIES_LEN + 1) * std::mem::size_of::<f64>()
            + (ONE_SERIES_LEN + 1) * std::mem::size_of::<i32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + prefix_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct CorrelHlBatchState {
        cuda: CudaCorrelHl,
        high: Vec<f32>,
        low: Vec<f32>,
        sweep: CorrelHlBatchRange,
    }
    impl CudaBenchState for CorrelHlBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .correl_hl_batch_dev(&self.high, &self.low, &self.sweep)
                .expect("correl_hl batch");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaCorrelHl::new(0).expect("CudaCorrelHl");
        let mut high = gen_series(ONE_SERIES_LEN);
        let mut low = vec![0.0f32; ONE_SERIES_LEN];
        // Make low a shifted/scaled version with some NaNs early
        
        for i in 0..ONE_SERIES_LEN {
            low[i] = 0.6 * high[i] + 0.2 * (i as f32).sin();
        }
        for i in 0..16 {
            high[i] = f32::NAN;
            low[i] = f32::NAN;
        }
        let sweep = CorrelHlBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
        };
        Box::new(CorrelHlBatchState {
            cuda,
            high,
            low,
            sweep,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "correl_hl",
            "one_series_many_params",
            "correl_hl_cuda_batch_dev",
            "1m_x_250",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}
