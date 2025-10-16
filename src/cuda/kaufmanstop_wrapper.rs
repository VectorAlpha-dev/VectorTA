//! CUDA support for the Kaufmanstop indicator (adaptive stop based on MA of range).
//!
//! Parity goals (mirrors ALMA wrapper patterns):
//! - PTX load via DetermineTargetFromContext + OptLevel O2 with fallback
//! - NON_BLOCKING stream
//! - VRAM checks and (lightweight) chunking where applicable
//! - Public entry points return VRAM-backed arrays (DeviceArrayF32)
//! - Warmup/NaN semantics identical to scalar implementation

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32;
use crate::cuda::moving_averages::ma_selector::{CudaMaData, CudaMaSelector, CudaMaSelectorError};
use crate::indicators::kaufmanstop::{expand_grid_wrapper, KaufmanstopBatchRange, KaufmanstopParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaKaufmanstopError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaKaufmanstopError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaKaufmanstopError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaKaufmanstopError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl std::error::Error for CudaKaufmanstopError {}

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
pub struct CudaKaufmanstopPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaKaufmanstopPolicy {
    fn default() -> Self {
        Self { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Auto }
    }
}

impl Default for BatchKernelPolicy {
    fn default() -> Self { BatchKernelPolicy::Auto }
}

impl Default for ManySeriesKernelPolicy {
    fn default() -> Self { ManySeriesKernelPolicy::Auto }
}

pub struct CudaKaufmanstop {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaKaufmanstopPolicy,
    device_id: u32,
}

impl CudaKaufmanstop {
    pub fn new(device_id: usize) -> Result<Self, CudaKaufmanstopError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaKaufmanstopError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaKaufmanstopError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaKaufmanstopError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/kaufmanstop_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => match Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                Ok(m) => m,
                Err(_) => Module::from_ptx(ptx, &[])
                    .map_err(|e| CudaKaufmanstopError::Cuda(e.to_string()))?,
            },
        };
        let stream =
            Stream::new(StreamFlags::NON_BLOCKING, None).map_err(|e| CudaKaufmanstopError::Cuda(e.to_string()))?;

        Ok(Self { module, stream, _context: context, policy: CudaKaufmanstopPolicy::default(), device_id: device_id as u32 })
    }

    pub fn new_with_policy(device_id: usize, policy: CudaKaufmanstopPolicy) -> Result<Self, CudaKaufmanstopError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }

    #[inline]
    fn headroom_bytes() -> usize {
        env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024)
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }

    #[inline]
    fn will_fit(bytes: usize, headroom: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Ok((free, _)) = mem_get_info() { bytes.saturating_add(headroom) <= free } else { true }
    }

    // -------------- Batch: one series × many params --------------
    pub fn kaufmanstop_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
        sweep: &KaufmanstopBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<KaufmanstopParams>), CudaKaufmanstopError> {
        if high.is_empty() || low.is_empty() || high.len() != low.len() {
            return Err(CudaKaufmanstopError::InvalidInput("high/low must be same non-zero length".into()));
        }
        let len = high.len();
        let first = high
            .iter()
            .zip(low.iter())
            .position(|(&h, &l)| !h.is_nan() && !l.is_nan())
            .ok_or_else(|| CudaKaufmanstopError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid_wrapper(sweep);
        if combos.is_empty() {
            return Err(CudaKaufmanstopError::InvalidInput("no parameter combinations".into()));
        }

        // Validate rows and warmup
        for prm in &combos {
            let p = prm.period.unwrap_or(0);
            if p == 0 || p > len {
                return Err(CudaKaufmanstopError::InvalidInput("invalid period".into()));
            }
            if len - first < p {
                return Err(CudaKaufmanstopError::InvalidInput(
                    "not enough valid data after first valid".into(),
                ));
            }
        }

        // VRAM estimate: inputs + outputs (MA rows are streamed per-row via selector)
        let in_bytes = 2 * len * std::mem::size_of::<f32>();
        let out_bytes = combos.len() * len * std::mem::size_of::<f32>();
        let head = Self::headroom_bytes();
        if !Self::will_fit(in_bytes + out_bytes, head) {
            return Err(CudaKaufmanstopError::InvalidInput(
                "insufficient device memory for kaufmanstop batch".into(),
            ));
        }

        // H2D for base series and output buffer
        let d_high = DeviceBuffer::from_slice(high).map_err(|e| CudaKaufmanstopError::Cuda(e.to_string()))?;
        let d_low = DeviceBuffer::from_slice(low).map_err(|e| CudaKaufmanstopError::Cuda(e.to_string()))?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(combos.len() * len) }
            .map_err(|e| CudaKaufmanstopError::Cuda(e.to_string()))?;

        // Build range (high - low) with NaN handling on host as in scalar path
        let mut range = vec![f32::NAN; len];
        for i in first..len {
            let h = high[i];
            let l = low[i];
            range[i] = if h.is_nan() || l.is_nan() { f32::NAN } else { h - l };
        }

        // Prepare kernel function once
        let mut axpy_fn: Function = self
            .module
            .get_function("kaufmanstop_axpy_row_f32")
            .map_err(|e| CudaKaufmanstopError::Cuda(e.to_string()))?;

        // Block sizing
        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Auto => {
                let (_min_grid, suggested) = axpy_fn
                    .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
                    .map_err(|e| CudaKaufmanstopError::Cuda(e.to_string()))?;
                suggested.max(128).min(512)
            }
            BatchKernelPolicy::Plain { block_x } => block_x.max(64).min(1024),
        };
        let grid_x = ((len as u32) + block_x - 1) / block_x;
        let block: BlockSize = (block_x, 1, 1).into();
        let grid_base: GridSize = (grid_x.max(1), 1, 1).into();

        let selector = CudaMaSelector::new(self.device_id as usize);
        for (row, prm) in combos.iter().enumerate() {
            let period = prm.period.unwrap();
            let mult = prm.mult.unwrap() as f32;
            let is_long = prm
                .direction
                .as_deref()
                .unwrap_or("long")
                .eq_ignore_ascii_case("long");
            let signed_mult = if is_long { -mult } else { mult };
            let base_is_low = if is_long { 1i32 } else { 0i32 };
            let warm = (first + period - 1) as i32;
            let ma_type = prm.ma_type.as_deref().unwrap_or("sma");

            // Compute MA(range) on device using the selector (single-period)
            let ma_dev = selector
                .ma_to_device(ma_type, CudaMaData::SliceF32(&range), period)
                .map_err(|e| CudaKaufmanstopError::Cuda(format!("ma_to_device: {}", e)))?;
            debug_assert_eq!(ma_dev.rows, 1);
            debug_assert_eq!(ma_dev.cols, len);

            // Launch per-row AXPY into row slice of d_out
            unsafe {
                let mut hp = d_high.as_device_ptr().as_raw();
                let mut lp = d_low.as_device_ptr().as_raw();
                let mut mp = ma_dev.buf.as_device_ptr().as_raw();
                let mut n = len as i32;
                let mut sm = signed_mult;
                let mut w = warm;
                let mut bil = base_is_low;
                let mut out_ptr = d_out
                    .as_device_ptr()
                    .add(row * len)
                    .as_raw();

                let args: &mut [*mut c_void] = &mut [
                    &mut hp as *mut _ as *mut c_void,
                    &mut lp as *mut _ as *mut c_void,
                    &mut mp as *mut _ as *mut c_void,
                    &mut n as *mut _ as *mut c_void,
                    &mut sm as *mut _ as *mut c_void,
                    &mut w as *mut _ as *mut c_void,
                    &mut bil as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];

                self.stream
                    .launch(&axpy_fn, grid_base, block, 0, args)
                    .map_err(|e| CudaKaufmanstopError::Cuda(e.to_string()))?;
            }
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaKaufmanstopError::Cuda(e.to_string()))?;

        Ok((DeviceArrayF32 { buf: d_out, rows: combos.len(), cols: len }, combos))
    }

    // -------------- Many-series: time-major, one param --------------
    pub fn kaufmanstop_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &KaufmanstopParams,
    ) -> Result<DeviceArrayF32, CudaKaufmanstopError> {
        if cols == 0 || rows == 0 || high_tm.len() != cols * rows || low_tm.len() != cols * rows {
            return Err(CudaKaufmanstopError::InvalidInput(
                "invalid dims for time-major inputs".into(),
            ));
        }
        let period = params.period.unwrap_or(0);
        if period == 0 || period > rows {
            return Err(CudaKaufmanstopError::InvalidInput("invalid period".into()));
        }
        let mult = params.mult.unwrap_or(2.0) as f32;
        let is_long = params
            .direction
            .as_deref()
            .unwrap_or("long")
            .eq_ignore_ascii_case("long");
        let signed_mult = if is_long { -mult } else { mult };
        let base_is_low = if is_long { 1i32 } else { 0i32 };
        let ma_type = params.ma_type.as_deref().unwrap_or("sma");

        // Per-series first valid (both high & low must be non-NaN at t)
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv: Option<usize> = None;
            for t in 0..rows {
                let idx = t * cols + s;
                if !high_tm[idx].is_nan() && !low_tm[idx].is_nan() {
                    fv = Some(t);
                    break;
                }
            }
            let fv = fv.ok_or_else(|| CudaKaufmanstopError::InvalidInput(format!("series {} all NaN", s)))?;
            if rows - fv < period {
                return Err(CudaKaufmanstopError::InvalidInput(format!(
                    "series {} insufficient data for period {} (tail = {})",
                    s,
                    period,
                    rows - fv
                )));
            }
            first_valids[s] = fv as i32;
        }

        // Build range (high - low) time-major with NaN propagation
        let mut range_tm = vec![f32::NAN; cols * rows];
        for idx in 0..(cols * rows) {
            let h = high_tm[idx];
            let l = low_tm[idx];
            range_tm[idx] = if h.is_nan() || l.is_nan() { f32::NAN } else { h - l };
        }

        // Compute MA(range) on device. For now, use the selector in single-parameter
        // time-major mode by flattening; only a subset of MAs expose time-major kernels
        // through their wrappers. To keep behavior broad, compute per-series via selector
        // but write directly to the final output with our AXPY kernel.
        // However, to reduce kernel launches, if ma_type == "sma", use the SMA wrapper's
        // native many-series path.
        let total = cols * rows;
        let mut d_high = DeviceBuffer::from_slice(high_tm).map_err(|e| CudaKaufmanstopError::Cuda(e.to_string()))?;
        let mut d_low = DeviceBuffer::from_slice(low_tm).map_err(|e| CudaKaufmanstopError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaKaufmanstopError::Cuda(e.to_string()))?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(total) }
            .map_err(|e| CudaKaufmanstopError::Cuda(e.to_string()))?;

        // Try native SMA many-series path; otherwise, fall back to per-series selector
        let ma_tm_dev: DeviceBuffer<f32> = if ma_type.eq_ignore_ascii_case("sma") {
            use crate::cuda::moving_averages::sma_wrapper::CudaSma;
            use crate::indicators::moving_averages::sma::SmaParams as SParams;
            let sma = CudaSma::new(self.device_id as usize)
                .map_err(|e| CudaKaufmanstopError::Cuda(e.to_string()))?;
            let sparams = SParams { period: Some(period) };
            let ma_dev = sma
                .sma_multi_series_one_param_time_major_dev(&range_tm, cols, rows, &sparams)
                .map_err(|e| CudaKaufmanstopError::Cuda(e.to_string()))?;
            ma_dev.buf
        } else {
            // Generic (slower) fallback: compute each series via selector and copy into a device buffer
            let selector = CudaMaSelector::new(0);
            let mut d_ma = unsafe { DeviceBuffer::<f32>::uninitialized(total) }
                .map_err(|e| CudaKaufmanstopError::Cuda(e.to_string()))?;
            // Prepare a host pinned buffer for staging each row
            for s in 0..cols {
                // Gather one series contiguous
                let mut series = vec![f32::NAN; rows];
                for t in 0..rows {
                    series[t] = range_tm[t * cols + s];
                }
                let ma_dev = selector
                    .ma_to_device(ma_type, CudaMaData::SliceF32(&series), period)
                    .map_err(|e| CudaKaufmanstopError::Cuda(format!("ma_to_device: {}", e)))?;
                debug_assert_eq!(ma_dev.rows, 1);
                debug_assert_eq!(ma_dev.cols, rows);
                // Copy into device buffer at column s (time-major)
                // We need to scatter to strided positions; use a small kernel-free host copy via staging
                // by pulling to pinned host and then writing into the strided layout.
                let mut pinned = unsafe {
                    LockedBuffer::<f32>::uninitialized(rows)
                        .map_err(|e| CudaKaufmanstopError::Cuda(e.to_string()))?
                };
                unsafe {
                    ma_dev
                        .buf
                        .async_copy_to(&mut pinned.as_mut_slice(), &self.stream)
                        .map_err(|e| CudaKaufmanstopError::Cuda(e.to_string()))?;
                }
                self.stream
                    .synchronize()
                    .map_err(|e| CudaKaufmanstopError::Cuda(e.to_string()))?;
                // Scatter into d_ma via host-side strided copy (one series at a time)
                // Note: For large matrices, a dedicated scatter kernel would be better.
                let mut host_scatter = vec![0f32; rows];
                host_scatter.copy_from_slice(pinned.as_slice());
                // Write into device at strided positions by building a temp time-major column
                // and copying as a contiguous slice per series by transposing logic.
                // Here we just copy to host and then to device slice with the same layout.
                for t in 0..rows {
                    let idx = t * cols + s;
                    // SAFETY: writing via staging host buffer; final upload after loop
                    // We'll accumulate into a full host buffer and upload once at the end.
                    range_tm[idx] = host_scatter[t];
                }
            }
            // Upload the filled time-major MA matrix
            d_ma
                .copy_from(&range_tm)
                .map_err(|e| CudaKaufmanstopError::Cuda(e.to_string()))?;
            d_ma
        };

        // Launch many-series AXPY kernel
        let mut func: Function = self
            .module
            .get_function("kaufmanstop_many_series_one_param_time_major_f32")
            .map_err(|e| CudaKaufmanstopError::Cuda(e.to_string()))?;
        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => {
                let (_min_grid, suggested) = func
                    .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
                    .map_err(|e| CudaKaufmanstopError::Cuda(e.to_string()))?;
                suggested.max(128).min(512)
            }
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(64).min(1024),
        };
        let total_i = (cols * rows) as u32;
        let grid_x = (total_i + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut hp = d_high.as_device_ptr().as_raw();
            let mut lp = d_low.as_device_ptr().as_raw();
            let mut mp = ma_tm_dev.as_device_ptr().as_raw();
            let mut fp = d_first.as_device_ptr().as_raw();
            let mut c = cols as i32;
            let mut r = rows as i32;
            let mut sm = signed_mult;
            let mut bil = base_is_low;
            let mut p = period as i32;
            let mut op = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut hp as *mut _ as *mut c_void,
                &mut lp as *mut _ as *mut c_void,
                &mut mp as *mut _ as *mut c_void,
                &mut fp as *mut _ as *mut c_void,
                &mut c as *mut _ as *mut c_void,
                &mut r as *mut _ as *mut c_void,
                &mut sm as *mut _ as *mut c_void,
                &mut bil as *mut _ as *mut c_void,
                &mut p as *mut _ as *mut c_void,
                &mut op as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaKaufmanstopError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaKaufmanstopError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 { buf: d_out, rows, cols })
    }
}

// ---------- Benches ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = 2 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct KaufmanstopBatchState {
        cuda: CudaKaufmanstop,
        high: Vec<f32>,
        low: Vec<f32>,
        sweep: KaufmanstopBatchRange,
    }
    impl CudaBenchState for KaufmanstopBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .kaufmanstop_batch_dev(&self.high, &self.low, &self.sweep)
                .expect("kaufmanstop batch");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaKaufmanstop::new(0).expect("cuda kaufmanstop");
        let p = gen_series(ONE_SERIES_LEN);
        let mut high = vec![0f32; ONE_SERIES_LEN];
        let mut low = vec![0f32; ONE_SERIES_LEN];
        for i in 0..ONE_SERIES_LEN {
            let r = 0.5f32 + ((i as f32) * 0.00037).cos().abs();
            high[i] = p[i] + 0.5 * r;
            low[i] = p[i] - 0.5 * r;
        }
        let sweep = KaufmanstopBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
            mult: (2.0, 2.0, 0.0),
            direction: ("long".to_string(), "long".to_string(), 0.0),
            ma_type: ("sma".to_string(), "sma".to_string(), 0.0),
        };
        Box::new(KaufmanstopBatchState { cuda, high, low, sweep })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "kaufmanstop",
            "one_series_many_params",
            "kaufmanstop_cuda_batch_dev",
            "1m_x_250",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}
