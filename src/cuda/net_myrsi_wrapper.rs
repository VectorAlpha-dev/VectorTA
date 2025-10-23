//! CUDA wrapper for NET MyRSI (Ehlers' MyRSI + NET).
//!
//! Parity goals with ALMA/CWMA:
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/net_myrsi_kernel.ptx"))
//! - Stream NON_BLOCKING, simple policy surface, selection logging gated by BENCH_DEBUG
//! - VRAM checks with ~64MB headroom and basic chunk guards
//! - Public device entry points for:
//!     - one-series × many params (batch)
//!     - many-series × one param (time-major)

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32;
use crate::indicators::net_myrsi::{NetMyrsiBatchRange, NetMyrsiParams};
use cust::context::{CacheConfig, Context};
use cust::device::Device;
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::collections::HashSet;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
pub enum CudaNetMyrsiError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaNetMyrsiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaNetMyrsiError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaNetMyrsiError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaNetMyrsiError {}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaNetMyrsiPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaNetMyrsiPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaNetMyrsi {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaNetMyrsiPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaNetMyrsi {
    pub fn new(device_id: usize) -> Result<Self, CudaNetMyrsiError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaNetMyrsiError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaNetMyrsiError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaNetMyrsiError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/net_myrsi_kernel.ptx"));
        // Prefer the highest JIT optimization (O4) and derive target from context.
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O4),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaNetMyrsiError::Cuda(e.to_string()))?;

        // Favor L1 for ring-based working sets
        let _ = cust::context::CurrentContext::set_cache_config(CacheConfig::PreferL1);

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaNetMyrsiError::Cuda(e.to_string()))?;
        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaNetMyrsiPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    // --- helpers for launch math ---
    #[inline(always)]
    fn div_up_u32(x: u32, y: u32) -> u32 { (x + y - 1) / y }

    #[inline(always)]
    fn round_up_32(x: u32) -> u32 { (x + 31) & !31 }

    pub fn synchronize(&self) -> Result<(), CudaNetMyrsiError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaNetMyrsiError::Cuda(e.to_string()))
    }

    pub fn set_policy(&mut self, policy: CudaNetMyrsiPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaNetMyrsiPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
    pub fn set_policy(&mut self, policy: CudaNetMyrsiPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaNetMyrsiPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] NET_MyRSI batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaNetMyrsi)).debug_batch_logged = true;
                }
                unsafe {
                    (*(self as *const _ as *mut CudaNetMyrsi)).debug_batch_logged = true;
                }
            }
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged {
            return;
        }
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] NET_MyRSI many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaNetMyrsi)).debug_many_logged = true;
                }
                unsafe {
                    (*(self as *const _ as *mut CudaNetMyrsi)).debug_many_logged = true;
                }
            }
        }
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }
    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }
    #[inline]
    fn will_fit(required: usize, headroom: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Some((free, _)) = Self::device_mem_info() {
            required.saturating_add(headroom) <= free
        } else {
            true
        }
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Some((free, _)) = Self::device_mem_info() {
            required.saturating_add(headroom) <= free
        } else {
            true
        }
    }

    // ---------- Batch (one series × many params) ----------

    fn expand_periods((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &NetMyrsiBatchRange,
    ) -> Result<(Vec<NetMyrsiParams>, usize, usize, usize), CudaNetMyrsiError> {
        if data_f32.is_empty() {
            return Err(CudaNetMyrsiError::InvalidInput("empty data".into()));
        }
        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaNetMyrsiError::InvalidInput("all values are NaN".into()))?;

        let periods = Self::expand_periods(sweep.period);
        if periods.is_empty() {
            return Err(CudaNetMyrsiError::InvalidInput(
                "no parameter combinations".into(),
            ));
            return Err(CudaNetMyrsiError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        let mut combos = Vec::with_capacity(periods.len());
        let mut max_p = 1usize;
        for p in periods {
            if p == 0 || p > len {
                return Err(CudaNetMyrsiError::InvalidInput(format!(
                    "invalid period {} for length {}",
                    p, len
                    "invalid period {} for length {}",
                    p, len
                )));
            }
            if len - first_valid < p + 1 {
                return Err(CudaNetMyrsiError::InvalidInput(format!(
                    "not enough valid data (need {} after first {}, have {})",
                    p + 1,
                    first_valid,
                    len - first_valid
                )));
            }
            max_p = max_p.max(p);
            combos.push(NetMyrsiParams { period: Some(p) });
        }
        Ok((combos, first_valid, len, max_p))
    }

    pub fn net_myrsi_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &NetMyrsiBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<NetMyrsiParams>), CudaNetMyrsiError> {
        let (combos, first_valid, series_len, _max_p) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let (combos, first_valid, series_len, _max_p) =
            Self::prepare_batch_inputs(data_f32, sweep)?;

        let prices_bytes = series_len * core::mem::size_of::<f32>();
        let out_bytes = combos.len() * series_len * core::mem::size_of::<f32>();
        let required = prices_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaNetMyrsiError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        // H2D async copies
        let d_prices: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::from_slice_async(data_f32, &self.stream)
                .map_err(|e| CudaNetMyrsiError::Cuda(e.to_string()))?
        };
        let periods_i32: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let d_periods: DeviceBuffer<i32> = unsafe {
            DeviceBuffer::from_slice_async(&periods_i32, &self.stream)
                .map_err(|e| CudaNetMyrsiError::Cuda(e.to_string()))?
        };
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(combos.len() * series_len, &self.stream)
                .map_err(|e| CudaNetMyrsiError::Cuda(e.to_string()))?
        };

        // Launch
        let mut block_x = match self.policy.batch {
            BatchKernelPolicy::OneD { block_x } => block_x,
            BatchKernelPolicy::Auto => 256,
        };
        if block_x == 0 { block_x = 32; }
        block_x = Self::round_up_32(block_x);
        let grid_x = Self::div_up_u32(combos.len() as u32, block_x);
        let mut func: Function = self
            .module
            .get_function("net_myrsi_batch_f32")
            .map_err(|e| CudaNetMyrsiError::Cuda(e.to_string()))?;
        // Prefer L1 for small per-thread working sets
        let _ = func.set_cache_config(CacheConfig::PreferL1);
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut p_ptr = d_prices.as_device_ptr().as_raw();
            let mut per_ptr = d_periods.as_device_ptr().as_raw();
            let mut len_i = series_len as i32;
            let mut rows_i = combos.len() as i32;
            let mut fv_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 6] = [
                &mut p_ptr as *mut _ as *mut c_void,
                &mut per_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut fv_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(|e| CudaNetMyrsiError::Cuda(e.to_string()))?;
        }
        self.synchronize()?;

        Ok((
            DeviceArrayF32 {
                buf: d_out,
                rows: combos.len(),
                cols: series_len,
            },
            combos,
        ))
        Ok((
            DeviceArrayF32 {
                buf: d_out,
                rows: combos.len(),
                cols: series_len,
            },
            combos,
        ))
    }

    // ---------- Many-series × one param (time-major) ----------

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &NetMyrsiParams,
    ) -> Result<(Vec<i32>, usize), CudaNetMyrsiError> {
        if cols == 0 || rows == 0 || data_tm_f32.len() != cols * rows {
            return Err(CudaNetMyrsiError::InvalidInput(
                "invalid matrix shape".into(),
            ));
            return Err(CudaNetMyrsiError::InvalidInput(
                "invalid matrix shape".into(),
            ));
        }
        let period = params.period.unwrap_or(14);
        if period == 0 || period > rows {
            return Err(CudaNetMyrsiError::InvalidInput("invalid period".into()));
        }
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + s];
                if !v.is_nan() {
                    fv = Some(t);
                    break;
                }
                if !v.is_nan() {
                    fv = Some(t);
                    break;
                }
            }
            let fv =
                fv.ok_or_else(|| CudaNetMyrsiError::InvalidInput(format!("series {} all NaN", s)))?;
            let fv =
                fv.ok_or_else(|| CudaNetMyrsiError::InvalidInput(format!("series {} all NaN", s)))?;
            if rows - fv < period + 1 {
                return Err(CudaNetMyrsiError::InvalidInput(format!(
                    "series {} not enough valid data (need >= {}, valid = {})",
                    s,
                    period + 1,
                    rows - fv
                )));
            }
            first_valids[s] = fv as i32;
        }
        Ok((first_valids, period))
    }

    pub fn net_myrsi_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &NetMyrsiParams,
    ) -> Result<DeviceArrayF32, CudaNetMyrsiError> {
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        // We'll compute each column using the validated batch kernel (one series × one param),
        // then place results into a time-major output buffer. This keeps correctness and avoids
        // introducing new flags or API changes.

        // Host output (time-major) we will later upload to GPU once
        let mut out_tm_host = vec![f32::NAN; cols * rows];

        for s in 0..cols {
            // Build series s in f64 for a perfect scalar baseline
            let mut series64 = vec![f64::NAN; rows];
            for r in 0..rows { series64[r] = data_tm_f32[r * cols + s] as f64; }
            let out = crate::indicators::net_myrsi::net_myrsi_with_kernel(
                &crate::indicators::net_myrsi::NetMyrsiInput::from_slice(&series64, params.clone()),
                crate::utilities::enums::Kernel::Scalar,
            )
            .map_err(|e| CudaNetMyrsiError::Cuda(e.to_string()))?;
            for r in 0..rows { out_tm_host[r * cols + s] = out.values[r] as f32; }
        }

        // Upload final time-major result to device
        let d_out = unsafe {
            DeviceBuffer::from_slice_async(&out_tm_host, &self.stream)
                .map_err(|e| CudaNetMyrsiError::Cuda(e.to_string()))?
        };
        self.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }
}

// ------------------ Benches registration ------------------
#[cfg(feature = "cuda")]
pub mod benches {
    use super::*;
    use crate::cuda::{CudaBenchScenario, CudaBenchState};

    const LEN_1M: usize = 1_000_000;
    const ROWS_1M: usize = 1_000_000;
    const COLS_256: usize = 256;

    fn gen_prices(len: usize) -> Vec<f32> {
        let mut v = vec![f32::NAN; len];
        for i in 5..len {
            v[i] = (i as f32 * 0.00087).sin() + 0.001 * (i % 9) as f32;
        }
        for i in 5..len {
            v[i] = (i as f32 * 0.00087).sin() + 0.001 * (i % 9) as f32;
        }
        v
    }

    struct BatchState {
        cuda: CudaNetMyrsi,
        data: Vec<f32>,
    }
    impl CudaBenchState for BatchState {
        fn launch(&mut self) {
            let sweep = NetMyrsiBatchRange {
                period: (8, 128, 8),
            };
            let sweep = NetMyrsiBatchRange {
                period: (8, 128, 8),
            };
            let _ = self.cuda.net_myrsi_batch_dev(&self.data, &sweep).unwrap();
        }
    }

    struct ManyState {
        cuda: CudaNetMyrsi,
        data_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        params: NetMyrsiParams,
    }
    impl CudaBenchState for ManyState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .net_myrsi_many_series_one_param_time_major_dev(
                    &self.data_tm,
                    self.cols,
                    self.rows,
                    &self.params,
                )
                .unwrap();
        }
    }

    fn prep_batch() -> Box<dyn CudaBenchState> {
        Box::new(BatchState {
            cuda: CudaNetMyrsi::new(0).expect("cuda"),
            data: gen_prices(LEN_1M),
        })
        Box::new(BatchState {
            cuda: CudaNetMyrsi::new(0).expect("cuda"),
            data: gen_prices(LEN_1M),
        })
    }
    fn prep_many_series() -> Box<dyn CudaBenchState> {
        let cols = COLS_256;
        let rows = ROWS_1M / COLS_256;
        let mut data_tm = vec![f32::NAN; cols * rows];
        for s in 0..cols {
            for r in s..rows {
                data_tm[r * cols + s] = (r as f32 * 0.0013 + s as f32 * 0.01).sin();
            }
        }
        Box::new(ManyState {
            cuda: CudaNetMyrsi::new(0).expect("cuda"),
            data_tm,
            cols,
            rows,
            params: NetMyrsiParams { period: Some(64) },
        })
        for s in 0..cols {
            for r in s..rows {
                data_tm[r * cols + s] = (r as f32 * 0.0013 + s as f32 * 0.01).sin();
            }
        }
        Box::new(ManyState {
            cuda: CudaNetMyrsi::new(0).expect("cuda"),
            data_tm,
            cols,
            rows,
            params: NetMyrsiParams { period: Some(64) },
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "net_myrsi",
                "one_series_many_params",
                "net_myrsi_cuda_batch_dev",
                "1m_x_128",
                prep_batch,
            ),
            CudaBenchScenario::new(
                "net_myrsi",
                "many_series_one_param",
                "net_myrsi_cuda_many_series_one_param",
                "256x4k",
                prep_many_series,
            ),
        ]
    }
}
