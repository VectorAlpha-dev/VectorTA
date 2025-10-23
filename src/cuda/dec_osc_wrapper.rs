//! CUDA wrapper for Decycler Oscillator (DEC_OSC).
//!
//! Parity goals with ALMA wrapper style:
//! - PTX load from OUT_DIR, DetermineTargetFromContext, OptLevel O2 fallback
//! - NON_BLOCKING stream
//! - Batch (one-series × many-params) and many-series × one-param (time-major)
//! - Warmup/NaN identical to scalar: write NaN for indices < first_valid+2
//! - VRAM estimation + 64MB headroom guard; chunk grid to <= 65_535 rows

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::dec_osc::{DecOscBatchRange, DecOscParams};
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
pub enum CudaDecOscError {
    Cuda(String),
    InvalidInput(String),
}
impl fmt::Display for CudaDecOscError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaDecOscError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaDecOscError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl std::error::Error for CudaDecOscError {}

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
impl Default for ManySeriesKernelPolicy {
    fn default() -> Self {
        ManySeriesKernelPolicy::Auto
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaDecOscPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaDecOsc {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaDecOscPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaDecOsc {
    pub fn new(device_id: usize) -> Result<Self, CudaDecOscError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/dec_osc_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;
        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaDecOscPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn set_policy(&mut self, policy: CudaDecOscPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaDecOscPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
    pub fn set_policy(&mut self, policy: CudaDecOscPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaDecOscPolicy {
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
        static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                if !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!("[DEBUG] dec_osc batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaDecOsc)).debug_batch_logged = true;
                }
                unsafe {
                    (*(self as *const _ as *mut CudaDecOsc)).debug_batch_logged = true;
                }
            }
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if self.debug_many_logged {
            return;
        }
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                if !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!("[DEBUG] dec_osc many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaDecOsc)).debug_many_logged = true;
                }
                unsafe {
                    (*(self as *const _ as *mut CudaDecOsc)).debug_many_logged = true;
                }
            }
        }
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }
    #[inline]
    fn will_fit(required_bytes: usize, headroom: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Ok((free, _total)) = mem_get_info() {
            required_bytes.saturating_add(headroom) <= free
        } else {
            true
        }
    }

    #[inline]
    fn ceil_div_u32(n: u32, d: u32) -> u32 { (n + d - 1) / d }
    #[inline]
    fn ceil_div_usize(n: usize, d: usize) -> usize { (n + d - 1) / d }

    fn expand_grid(range: &DecOscBatchRange) -> Vec<DecOscParams> {
        fn axis_usize(a: (usize, usize, usize)) -> Vec<usize> {
            let (s, e, st) = a;
            if st == 0 || s == e {
                return vec![s];
            }
            let mut v = Vec::new();
            let mut x = s;
            while x <= e {
                v.push(x);
                x = x.saturating_add(st);
            }
            while x <= e {
                v.push(x);
                x = x.saturating_add(st);
            }
            v
        }
        fn axis_f64(a: (f64, f64, f64)) -> Vec<f64> {
            let (s, e, st) = a;
            if st.abs() < 1e-12 || (s - e).abs() < 1e-12 {
                return vec![s];
            }
            let mut v = Vec::new();
            let mut x = s;
            while x <= e + 1e-12 {
                v.push(x);
                x += st;
            }
            while x <= e + 1e-12 {
                v.push(x);
                x += st;
            }
            v
        }
        let periods = axis_usize(range.hp_period);
        let ks = axis_f64(range.k);
        let mut out = Vec::with_capacity(periods.len() * ks.len());
        for &p in &periods {
            for &k in &ks {
                out.push(DecOscParams {
                    hp_period: Some(p),
                    k: Some(k),
                });
                out.push(DecOscParams {
                    hp_period: Some(p),
                    k: Some(k),
                });
            }
        }
        out
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &DecOscBatchRange,
    ) -> Result<(Vec<DecOscParams>, usize, usize), CudaDecOscError> {
        if data_f32.is_empty() {
            return Err(CudaDecOscError::InvalidInput("empty data".into()));
        }
        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaDecOscError::InvalidInput("all values are NaN".into()))?;

        let combos = Self::expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaDecOscError::InvalidInput(
                "no parameter combinations".into(),
            ));
            return Err(CudaDecOscError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        for prm in &combos {
            let p = prm.hp_period.unwrap_or(0);
            let k = prm.k.unwrap_or(0.0);
            if p < 2 || p > len {
                return Err(CudaDecOscError::InvalidInput(format!(
                    "invalid hp_period {} for len {}",
                    p, len
                )));
            }
            if k <= 0.0 || !k.is_finite() {
                return Err(CudaDecOscError::InvalidInput(format!("invalid k {}", k)));
            }
            if len - first_valid < 2 {
                return Err(CudaDecOscError::InvalidInput(
                    "not enough valid data".into(),
                ));
                return Err(CudaDecOscError::InvalidInput(
                    "not enough valid data".into(),
                ));
            }
        }
        Ok((combos, first_valid, len))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_ks: &DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
        periods_off: usize,
        out_off_elems: usize,
    ) -> Result<(), CudaDecOscError> {
        let mut func: Function = self
            .module
            .get_function("dec_osc_batch_f32")
            .map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;

        // Use occupancy suggestion for block size and minimum grid size
        let (suggested_block_x, min_grid) = func
            .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
            .map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Auto => suggested_block_x.max(128),
            BatchKernelPolicy::Plain { block_x } => block_x.max(64),
        };
        unsafe {
            (*(self as *const _ as *mut CudaDecOsc)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        unsafe {
            (*(self as *const _ as *mut CudaDecOsc)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();
        // Right-sized grid: one thread per combo → ceil_div(n_combos, block_x)
        let combos_u32 = n_combos as u32;
        let mut grid_x = Self::ceil_div_u32(combos_u32, block_x);
        grid_x = grid_x.max(min_grid);
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw()
                + (periods_off * std::mem::size_of::<i32>()) as u64;
            let mut ks_ptr =
                d_ks.as_device_ptr().as_raw() + (periods_off * std::mem::size_of::<f32>()) as u64;
            let mut ks_ptr =
                d_ks.as_device_ptr().as_raw() + (periods_off * std::mem::size_of::<f32>()) as u64;
            let mut len_i = series_len as i32;
            let mut combos_i = n_combos as i32;
            let mut first_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw()
                + (out_off_elems * std::mem::size_of::<f32>()) as u64;
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut ks_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    pub fn dec_osc_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &DecOscBatchRange,
    ) -> Result<DeviceArrayF32, CudaDecOscError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let rows = combos.len();

        let prices_bytes = len * std::mem::size_of::<f32>();
        let params_bytes = rows * (std::mem::size_of::<i32>() + std::mem::size_of::<f32>());
        let out_bytes = rows * len * std::mem::size_of::<f32>();
        let need = prices_bytes + params_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(need, headroom) {
            return Err(CudaDecOscError::InvalidInput(format!(
                "insufficient VRAM: need ~{:.2} MB",
                (need + headroom) as f64 / (1024.0 * 1024.0)
            )));
        }

        let periods: Vec<i32> = combos.iter().map(|c| c.hp_period.unwrap() as i32).collect();
        let ks: Vec<f32> = combos.iter().map(|c| c.k.unwrap() as f32).collect();

        // Prefer async/pinned path for bulk transfers
        let h_prices =
            LockedBuffer::from_slice(data_f32).map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;
        let h_periods =
            LockedBuffer::from_slice(&periods).map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;
        let h_ks =
            LockedBuffer::from_slice(&ks).map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;
        let h_prices =
            LockedBuffer::from_slice(data_f32).map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;
        let h_periods =
            LockedBuffer::from_slice(&periods).map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;
        let h_ks =
            LockedBuffer::from_slice(&ks).map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;

        let mut d_prices = unsafe { DeviceBuffer::<f32>::uninitialized_async(len, &self.stream) }
            .map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;
        let mut d_periods = unsafe { DeviceBuffer::<i32>::uninitialized_async(rows, &self.stream) }
            .map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;
        let mut d_ks = unsafe { DeviceBuffer::<f32>::uninitialized_async(rows, &self.stream) }
            .map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;
        let mut d_out =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(rows * len, &self.stream) }
                .map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;
        let mut d_out =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(rows * len, &self.stream) }
                .map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;
        unsafe {
            d_prices
                .async_copy_from(&h_prices, &self.stream)
                .map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;
            d_periods
                .async_copy_from(&h_periods, &self.stream)
                .map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;
            d_ks.async_copy_from(&h_ks, &self.stream)
            d_ks.async_copy_from(&h_ks, &self.stream)
                .map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;
        }

        // Compute block_x once to size chunks up to grid.x limit
        let func = self
            .module
            .get_function("dec_osc_batch_f32")
            .map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;
        let (suggested_block_x, _min_grid) = func
            .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
            .map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Auto => suggested_block_x.max(128),
            BatchKernelPolicy::Plain { block_x } => block_x.max(64),
        } as usize;

        const MAX_GRID_X: usize = 65_535; // keep legacy guard
        let max_combos_per_launch = MAX_GRID_X.saturating_mul(block_x);

        let mut launched = 0usize;
        while launched < rows {
            let n = (rows - launched).min(max_combos_per_launch);
            self.launch_batch_kernel(
                &d_prices,
                &d_periods,
                &d_ks,
                len,
                n,
                first_valid,
                &mut d_out,
                launched,
                launched * len,
            )?;
            launched += n;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols: len,
        })
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols: len,
        })
    }

    fn prepare_many_series(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &DecOscParams,
    ) -> Result<(Vec<i32>, usize, f32), CudaDecOscError> {
        if cols == 0 || rows == 0 {
            return Err(CudaDecOscError::InvalidInput("cols or rows is zero".into()));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaDecOscError::InvalidInput(
                "time-major shape mismatch".into(),
            ));
            return Err(CudaDecOscError::InvalidInput(
                "time-major shape mismatch".into(),
            ));
        }
        let p = params.hp_period.unwrap_or(0);
        let k = params.k.unwrap_or(0.0);
        if p < 2 || p > rows {
            return Err(CudaDecOscError::InvalidInput("invalid hp_period".into()));
        }
        if k <= 0.0 || !k.is_finite() {
            return Err(CudaDecOscError::InvalidInput("invalid k".into()));
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
                fv.ok_or_else(|| CudaDecOscError::InvalidInput(format!("series {} all NaN", s)))?;
            let fv =
                fv.ok_or_else(|| CudaDecOscError::InvalidInput(format!("series {} all NaN", s)))?;
            if rows - fv < 2 {
                return Err(CudaDecOscError::InvalidInput(format!(
                    "series {} not enough valid data (need >= 2, got {})",
                    s,
                    rows - fv
                )));
            }
            first_valids[s] = fv as i32;
        }
        Ok((first_valids, p, k as f32))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        k: f32,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaDecOscError> {
        let func: Function = self
            .module
            .get_function("dec_osc_many_series_one_param_time_major_f32")
            .map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;

        // Use occupancy hint for 1D grid across series
        let (suggested_block_x, _min_grid) = func
            .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
            .map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => suggested_block_x.max(128),
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(64),
        };
        unsafe {
            (*(self as *const _ as *mut CudaDecOsc)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        unsafe {
            (*(self as *const _ as *mut CudaDecOsc)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut period_i = period as i32;
            let mut k_f = k as f32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut k_f as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    pub fn dec_osc_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &DecOscParams,
    ) -> Result<DeviceArrayF32, CudaDecOscError> {
        let (first_valids, period, k_f32) =
            Self::prepare_many_series(data_tm_f32, cols, rows, params)?;

        // VRAM guard similar to batch path
        let prices_bytes = data_tm_f32.len() * std::mem::size_of::<f32>();
        let first_bytes  = first_valids.len() * std::mem::size_of::<i32>();
        let out_bytes    = cols * rows * std::mem::size_of::<f32>();
        let need         = prices_bytes + first_bytes + out_bytes;
        let headroom     = 64 * 1024 * 1024;
        if !Self::will_fit(need, headroom) {
            return Err(CudaDecOscError::InvalidInput(format!(
                "insufficient VRAM: need ~{:.2} MB",
                (need + headroom) as f64 / (1024.0 * 1024.0)
            )));
        }

        // Pinned + async transfers for truly asynchronous copies
        let h_prices = LockedBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;
        let h_first = LockedBuffer::from_slice(&first_valids)
            .map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;

        let mut d_prices = unsafe { DeviceBuffer::<f32>::uninitialized_async(cols * rows, &self.stream) }
            .map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;
        let mut d_first  = unsafe { DeviceBuffer::<i32>::uninitialized_async(cols, &self.stream) }
            .map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;
        let mut d_out    = unsafe { DeviceBuffer::<f32>::uninitialized_async(cols * rows, &self.stream) }
            .map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;

        unsafe {
            d_prices
                .async_copy_from(&h_prices, &self.stream)
                .map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;
            d_first
                .async_copy_from(&h_first, &self.stream)
                .map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;
        }

        self.launch_many_series_kernel(&d_prices, &d_first, cols, rows, period, k_f32, &mut d_out)?;

        self.stream
            .synchronize()
            .map_err(|e| CudaDecOscError::Cuda(e.to_string()))?;

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

// ---------- Benches ----------
pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        dec_osc_benches,
        crate::cuda::oscillators::CudaDecOsc,
        crate::indicators::dec_osc::DecOscBatchRange,
        crate::indicators::dec_osc::DecOscParams,
        dec_osc_batch_dev,
        dec_osc_many_series_one_param_time_major_dev,
        crate::indicators::dec_osc::DecOscBatchRange {
            hp_period: (50, 50 + PARAM_SWEEP - 1, 1),
            k: (1.0, 1.0, 0.0)
        },
        crate::indicators::dec_osc::DecOscParams {
            hp_period: Some(125),
            k: Some(1.0)
        },
        crate::indicators::dec_osc::DecOscParams {
            hp_period: Some(125),
            k: Some(1.0)
        },
        "dec_osc",
        "dec_osc"
    );
    pub use dec_osc_benches::bench_profiles;
}
