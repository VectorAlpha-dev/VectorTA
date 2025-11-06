//! CUDA wrapper for Kurtosis (excess kurtosis) indicator.
//!
//! Parity goals per Agents Guide:
//! - API/behavior match ALMA-style wrappers (policy enums, NON_BLOCKING stream)
//! - PTX load with DetermineTargetFromContext and O2 fallback
//! - VRAM estimation with ~64MB headroom; grid-y chunking to <= 65_535
//! - Batch builds DS (float2) host prefix sums of x, x^2, x^3, x^4 and NaN counts (pinned memory)
//! - Many-series×one-param uses time-major scan with incremental raw sums (DS accumulators)

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::kurtosis::{KurtosisBatchRange, KurtosisParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;

// Host-side DS primitives for building float2 prefix sums (FP64-free)
#[repr(C, align(8))]
#[derive(Clone, Copy, Default)]
struct Float2 {
    x: f32,
    y: f32,
}
unsafe impl cust::memory::DeviceCopy for Float2 {}

#[inline(always)]
fn two_sum_f32(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    let bb = s - a;
    let e = (a - (s - bb)) + (b - bb);
    (s, e)
}

#[inline(always)]
fn ds_add((ahi, alo): (f32, f32), (bhi, blo): (f32, f32)) -> (f32, f32) {
    let (s, mut e) = two_sum_f32(ahi, bhi);
    e += alo + blo;
    let hi = s + e;
    let lo = e - (hi - s);
    (hi, lo)
}

#[derive(Debug)]
pub enum CudaKurtosisError {
    Cuda(String),
    InvalidInput(String),
}
impl fmt::Display for CudaKurtosisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaKurtosisError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaKurtosisError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaKurtosisError {}

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
pub struct CudaKurtosisPolicy {
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

pub struct CudaKurtosis {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaKurtosisPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaKurtosis {
    pub fn new(device_id: usize) -> Result<Self, CudaKurtosisError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaKurtosisError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaKurtosisError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaKurtosisError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/kurtosis_kernel.ptx"));
        let module = Module::from_ptx(
            ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O2),
            ],
        )
        .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
        .or_else(|_| Module::from_ptx(ptx, &[]))
        .map_err(|e| CudaKurtosisError::Cuda(e.to_string()))?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaKurtosisError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaKurtosisPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn set_policy(&mut self, policy: CudaKurtosisPolicy) { self.policy = policy; }
    pub fn policy(&self) -> &CudaKurtosisPolicy { &self.policy }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> { self.last_batch }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> { self.last_many }

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
        if !Self::mem_check_enabled() { return true; }
        if let Some((free, _)) = Self::device_mem_info() {
            return required_bytes.saturating_add(headroom_bytes) <= free;
        }
        true
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                if !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] kurtosis batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaKurtosis)).debug_batch_logged = true;
                }
                unsafe {
                    (*(self as *const _ as *mut CudaKurtosis)).debug_batch_logged = true;
                }
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
                    eprintln!("[DEBUG] kurtosis many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaKurtosis)).debug_many_logged = true;
                }
                unsafe {
                    (*(self as *const _ as *mut CudaKurtosis)).debug_many_logged = true;
                }
            }
        }
    }

    fn expand_grid(r: &KurtosisBatchRange) -> Vec<KurtosisParams> {
        let (start, end, step) = r.period;
        let periods = if step == 0 || start == end {
            vec![start]
        } else {
            (start..=end).step_by(step).collect::<Vec<_>>()
        };
        periods
            .into_iter()
            .map(|p| KurtosisParams { period: Some(p) })
            .collect()
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &KurtosisBatchRange,
    ) -> Result<(Vec<KurtosisParams>, usize, usize), CudaKurtosisError> {
        if data_f32.is_empty() {
            return Err(CudaKurtosisError::InvalidInput("empty data".into()));
        }
        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaKurtosisError::InvalidInput("all values are NaN".into()))?;
        let combos = Self::expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaKurtosisError::InvalidInput(
                "no parameter combinations".into(),
            ));
            return Err(CudaKurtosisError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        for c in &combos {
            let p = c.period.unwrap_or(0);
            if p == 0 {
                return Err(CudaKurtosisError::InvalidInput("period must be > 0".into()));
            }
            if p > len {
                return Err(CudaKurtosisError::InvalidInput(
                    "period exceeds data length".into(),
                ));
            }
            if p == 0 {
                return Err(CudaKurtosisError::InvalidInput("period must be > 0".into()));
            }
            if p > len {
                return Err(CudaKurtosisError::InvalidInput(
                    "period exceeds data length".into(),
                ));
            }
            if len - first_valid < p {
                return Err(CudaKurtosisError::InvalidInput(
                    "not enough valid data after first_valid".into(),
                ));
                return Err(CudaKurtosisError::InvalidInput(
                    "not enough valid data after first_valid".into(),
                ));
            }
        }
        Ok((combos, first_valid, len))
    }

    // Build DS prefixes directly into pinned host buffers for faster H2D transfers.
    fn build_prefixes_ds(
        &self,
        data: &[f32],
    ) -> Result<
        (
            LockedBuffer<Float2>,
            LockedBuffer<Float2>,
            LockedBuffer<Float2>,
            LockedBuffer<Float2>,
            LockedBuffer<i32>,
        ),
        CudaKurtosisError,
    > {
        let n = data.len();
        let mut ps1 = unsafe { LockedBuffer::<Float2>::uninitialized(n + 1) }
            .map_err(|e| CudaKurtosisError::Cuda(e.to_string()))?;
        let mut ps2 = unsafe { LockedBuffer::<Float2>::uninitialized(n + 1) }
            .map_err(|e| CudaKurtosisError::Cuda(e.to_string()))?;
        let mut ps3 = unsafe { LockedBuffer::<Float2>::uninitialized(n + 1) }
            .map_err(|e| CudaKurtosisError::Cuda(e.to_string()))?;
        let mut ps4 = unsafe { LockedBuffer::<Float2>::uninitialized(n + 1) }
            .map_err(|e| CudaKurtosisError::Cuda(e.to_string()))?;
        let mut ps_nan = unsafe { LockedBuffer::<i32>::uninitialized(n + 1) }
            .map_err(|e| CudaKurtosisError::Cuda(e.to_string()))?;

        // init first element
        ps1.as_mut_slice()[0] = Float2 { x: 0.0, y: 0.0 };
        ps2.as_mut_slice()[0] = Float2 { x: 0.0, y: 0.0 };
        ps3.as_mut_slice()[0] = Float2 { x: 0.0, y: 0.0 };
        ps4.as_mut_slice()[0] = Float2 { x: 0.0, y: 0.0 };
        ps_nan.as_mut_slice()[0] = 0;

        // running DS accumulators (hi, lo)
        let mut s1 = (0.0f32, 0.0f32);
        let mut s2 = (0.0f32, 0.0f32);
        let mut s3 = (0.0f32, 0.0f32);
        let mut s4 = (0.0f32, 0.0f32);
        let mut nan_count = 0i32;

        let ps1_slice = ps1.as_mut_slice();
        let ps2_slice = ps2.as_mut_slice();
        let ps3_slice = ps3.as_mut_slice();
        let ps4_slice = ps4.as_mut_slice();
        let psn_slice = ps_nan.as_mut_slice();

        for i in 0..n {
            let v = data[i];
            if v.is_nan() {
                nan_count += 1;
            } else {
                let d = v;
                let d2 = d.mul_add(d, 0.0);
                s1 = ds_add(s1, (d, 0.0));
                s2 = ds_add(s2, (d2, 0.0));
                s3 = ds_add(s3, (d2 * d, 0.0));
                s4 = ds_add(s4, (d2 * d2, 0.0));
            }

            ps1_slice[i + 1] = Float2 { x: s1.0, y: s1.1 };
            ps2_slice[i + 1] = Float2 { x: s2.0, y: s2.1 };
            ps3_slice[i + 1] = Float2 { x: s3.0, y: s3.1 };
            ps4_slice[i + 1] = Float2 { x: s4.0, y: s4.1 };
            psn_slice[i + 1] = nan_count;
        }
        Ok((ps1, ps2, ps3, ps4, ps_nan))
    }

    fn launch_batch(
        &self,
        d_ps1: &DeviceBuffer<Float2>,
        d_ps2: &DeviceBuffer<Float2>,
        d_ps3: &DeviceBuffer<Float2>,
        d_ps4: &DeviceBuffer<Float2>,
        d_ps_nan: &DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        d_periods: &DeviceBuffer<i32>,
        combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaKurtosisError> {
        let func = self
            .module
            .get_function("kurtosis_batch_f32")
            .map_err(|e| CudaKurtosisError::Cuda(e.to_string()))?;

        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Auto => 256,
            BatchKernelPolicy::Plain { block_x } => block_x.max(64),
        };
        let grid_x: u32 = ((len as u32) + block_x - 1) / block_x;
        let grid_base: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            (*(self as *const _ as *mut CudaKurtosis)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }

        // Chunk grid.y to <= 65_535
        let mut launched = 0usize;
        while launched < combos {
            let chunk = (combos - launched).min(65_535);
            let grid: GridSize = (grid_base.x, chunk as u32, 1).into();
            unsafe {
                let mut ps1 = d_ps1.as_device_ptr().as_raw();
                let mut ps2 = d_ps2.as_device_ptr().as_raw();
                let mut ps3 = d_ps3.as_device_ptr().as_raw();
                let mut ps4 = d_ps4.as_device_ptr().as_raw();
                let mut psn = d_ps_nan.as_device_ptr().as_raw();
                let mut len_i = len as i32;
                let mut first_valid_i = first_valid as i32;
                // Adjust raw device pointer by element offset (u64 arithmetic)
                let mut periods = d_periods
                    .as_device_ptr()
                    .as_raw()
                    .saturating_add((launched as u64) * (std::mem::size_of::<i32>() as u64));
                let mut combos_i = chunk as i32;
                let mut outp = d_out.as_device_ptr().as_raw().saturating_add(
                    ((launched * len) as u64) * (std::mem::size_of::<f32>() as u64),
                );
                let args: &mut [*mut c_void] = &mut [
                    &mut ps1 as *mut _ as *mut c_void,
                    &mut ps2 as *mut _ as *mut c_void,
                    &mut ps3 as *mut _ as *mut c_void,
                    &mut ps4 as *mut _ as *mut c_void,
                    &mut psn as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_valid_i as *mut _ as *mut c_void,
                    &mut periods as *mut _ as *mut c_void,
                    &mut combos_i as *mut _ as *mut c_void,
                    &mut outp as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaKurtosisError::Cuda(e.to_string()))?;
            }
            launched += chunk;
        }
        Ok(())
    }

    pub fn kurtosis_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &KurtosisBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<KurtosisParams>), CudaKurtosisError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let (h_ps1, h_ps2, h_ps3, h_ps4, h_ps_nan) = self.build_prefixes_ds(data_f32)?;
        let periods: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();

        // VRAM estimate
        let bytes_prefix = (h_ps1.len() + h_ps2.len() + h_ps3.len() + h_ps4.len())
            * std::mem::size_of::<Float2>()
            + h_ps_nan.len() * std::mem::size_of::<i32>();
        let bytes_periods = periods.len() * std::mem::size_of::<i32>();
        let bytes_out = combos.len() * len * std::mem::size_of::<f32>();
        let required = bytes_prefix + bytes_periods + bytes_out;
        let headroom = 64 * 1024 * 1024; // ~64MB headroom
        if !Self::will_fit(required, headroom) {
            return Err(CudaKurtosisError::Cuda(format!(
                "insufficient VRAM (need ~{} MiB incl. headroom)",
                (required + headroom + (1 << 20) - 1) / (1 << 20)
            )));
        }

        // Device allocations/copies
        let d_ps1 = unsafe { DeviceBuffer::from_slice_async(h_ps1.as_slice(), &self.stream) }
            .map_err(|e| CudaKurtosisError::Cuda(e.to_string()))?;
        let d_ps2 = unsafe { DeviceBuffer::from_slice_async(h_ps2.as_slice(), &self.stream) }
            .map_err(|e| CudaKurtosisError::Cuda(e.to_string()))?;
        let d_ps3 = unsafe { DeviceBuffer::from_slice_async(h_ps3.as_slice(), &self.stream) }
            .map_err(|e| CudaKurtosisError::Cuda(e.to_string()))?;
        let d_ps4 = unsafe { DeviceBuffer::from_slice_async(h_ps4.as_slice(), &self.stream) }
            .map_err(|e| CudaKurtosisError::Cuda(e.to_string()))?;
        let d_psn = unsafe { DeviceBuffer::from_slice_async(h_ps_nan.as_slice(), &self.stream) }
            .map_err(|e| CudaKurtosisError::Cuda(e.to_string()))?;
        let d_periods = unsafe { DeviceBuffer::from_slice_async(&periods, &self.stream) }
            .map_err(|e| CudaKurtosisError::Cuda(e.to_string()))?;
        let mut d_out =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(combos.len() * len, &self.stream) }
                .map_err(|e| CudaKurtosisError::Cuda(e.to_string()))?;

        self.launch_batch(
            &d_ps1,
            &d_ps2,
            &d_ps3,
            &d_ps4,
            &d_psn,
            len,
            first_valid,
            &d_periods,
            combos.len(),
            &mut d_out,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaKurtosisError::Cuda(e.to_string()))?;
        self.maybe_log_batch_debug();

        Ok((
            DeviceArrayF32 {
                buf: d_out,
                rows: combos.len(),
                cols: len,
            },
            combos,
        ))
    }

    pub fn kurtosis_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaKurtosisError> {
        if cols == 0 || rows == 0 {
            return Err(CudaKurtosisError::InvalidInput("empty matrix".into()));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaKurtosisError::InvalidInput(
                "time-major slice length mismatch".into(),
            ));
        }
        if period == 0 {
            return Err(CudaKurtosisError::InvalidInput("period must be > 0".into()));
        }

        // Compute per-series first_valid
        let mut first_valids = vec![-1i32; cols];
        for s in 0..cols {
            let mut fv = -1i32;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + s];
                if !v.is_nan() {
                    fv = t as i32;
                    break;
                }
            }
            first_valids[s] = fv;
        }

        // VRAM estimate
        let bytes_in = cols * rows * std::mem::size_of::<f32>();
        let bytes_out = bytes_in;
        let bytes_fv = cols * std::mem::size_of::<i32>();
        let required = bytes_in + bytes_out + bytes_fv;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaKurtosisError::Cuda("insufficient VRAM".into()));
        }

        let d_in = unsafe { DeviceBuffer::from_slice_async(data_tm_f32, &self.stream) }
            .map_err(|e| CudaKurtosisError::Cuda(e.to_string()))?;
        let d_fv = unsafe { DeviceBuffer::from_slice_async(&first_valids, &self.stream) }
            .map_err(|e| CudaKurtosisError::Cuda(e.to_string()))?;
        let mut d_out =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(cols * rows, &self.stream) }
                .map_err(|e| CudaKurtosisError::Cuda(e.to_string()))?;

        let func = self
            .module
            .get_function("kurtosis_many_series_one_param_f32")
            .map_err(|e| CudaKurtosisError::Cuda(e.to_string()))?;
        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 128,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32),
        };
        let grid: GridSize = (cols as u32, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            (*(self as *const _ as *mut CudaKurtosis)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        unsafe {
            (*(self as *const _ as *mut CudaKurtosis)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }

        unsafe {
            let mut in_ptr = d_in.as_device_ptr().as_raw();
            let mut fv_ptr = d_fv.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut in_ptr as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaKurtosisError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaKurtosisError::Cuda(e.to_string()))?;
        self.maybe_log_many_debug();

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }
}

// ---------- Bench profiles ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::kurtosis::KurtosisBatchRange;

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;

    fn bytes_one_series_many_params() -> usize {
        let ps = (ONE_SERIES_LEN + 1) * std::mem::size_of::<super::Float2>();
        let prefixes = 4 * ps + (ONE_SERIES_LEN + 1) * std::mem::size_of::<i32>();
        let periods = PARAM_SWEEP * std::mem::size_of::<i32>();
        let out = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        prefixes + periods + out + 64 * 1024 * 1024
    }

    struct KurtosisBatchState {
        cuda: CudaKurtosis,
        price: Vec<f32>,
        sweep: KurtosisBatchRange,
    }
    impl CudaBenchState for KurtosisBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .kurtosis_batch_dev(&self.price, &self.sweep)
                .expect("kurtosis batch");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaKurtosis::new(0).expect("cuda kurtosis");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = KurtosisBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
        };
        let sweep = KurtosisBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
        };
        Box::new(KurtosisBatchState { cuda, price, sweep })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "kurtosis",
            "one_series_many_params",
            "kurtosis_cuda_batch_dev",
            "1m_x_250",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}
