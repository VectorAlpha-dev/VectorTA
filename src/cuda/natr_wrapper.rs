//! CUDA support for the NATR (Normalized Average True Range) indicator.
//!
//! Parity goals (mirrors ALMA/CWMA wrappers):
//! - PTX load via DetermineTargetFromContext + OptLevel O2 with simple fallback
//! - NON_BLOCKING stream
//! - VRAM checks with optional headroom and grid chunking
//! - Public entry points:
//!     - one-series × many-params (batch)
//!     - many-series × one-param (time‑major)
//! - Numerics: warmup/NaN identical to scalar; warmup ATR by mean(TR) and
//!   sequential Wilder recurrence; outputs in f32.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::natr::{NatrBatchRange, NatrParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaNatrError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaNatrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaNatrError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaNatrError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl std::error::Error for CudaNatrError {}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
}

impl Default for BatchKernelPolicy {
    fn default() -> Self {
        BatchKernelPolicy::Auto
    }
    fn default() -> Self {
        BatchKernelPolicy::Auto
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 }, // expressed as threads (must be multiple of 32)
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
pub struct CudaNatrPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

pub struct CudaNatr {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaNatrPolicy,
    debug_logged: bool,
}

impl CudaNatr {
    pub fn new(device_id: usize) -> Result<Self, CudaNatrError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaNatrError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaNatrError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaNatrError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaNatrError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/natr_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
                {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
                {
                    m
                } else {
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaNatrError::Cuda(e.to_string()))?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaNatrError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaNatrPolicy::default(),
            debug_logged: false,
        })
        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaNatrPolicy::default(),
            debug_logged: false,
        })
    }

    pub fn set_policy(&mut self, policy: CudaNatrPolicy) {
        self.policy = policy;
    }
    pub fn set_policy(&mut self, policy: CudaNatrPolicy) {
        self.policy = policy;
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
        if let Ok((free, _)) = mem_get_info() {
            bytes.saturating_add(headroom) <= free
        } else {
            true
        }
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Ok((free, _)) = mem_get_info() {
            bytes.saturating_add(headroom) <= free
        } else {
            true
        }
    }

    // ----------------------- Host precompute (batch) -----------------------
    fn first_valid_hlc(high: &[f32], low: &[f32], close: &[f32]) -> Option<usize> {
        let n = high.len().min(low.len()).min(close.len());
        for i in 0..n {
            if high[i].is_finite() && low[i].is_finite() && close[i].is_finite() {
                return Some(i);
            }
            if high[i].is_finite() && low[i].is_finite() && close[i].is_finite() {
                return Some(i);
            }
        }
        None
    }

    fn build_tr_one_series(
        high: &[f32],
        low: &[f32],
        close: &[f32],
    ) -> Result<(Vec<f32>, usize), CudaNatrError> {
        if high.len() != low.len() || high.len() != close.len() || high.is_empty() {
            return Err(CudaNatrError::InvalidInput(
                "mismatched or empty inputs".into(),
            ));
            return Err(CudaNatrError::InvalidInput(
                "mismatched or empty inputs".into(),
            ));
        }
        let len = high.len();
        let first = Self::first_valid_hlc(high, low, close)
            .ok_or_else(|| CudaNatrError::InvalidInput("all values are NaN".into()))?;
        let mut tr = vec![0f32; len];
        if first < len {
            tr[first] = high[first] - low[first];
            for i in (first + 1)..len {
                let h = high[i];
                let l = low[i];
                let pc = close[i - 1];
                let hl = h - l;
                let hc = (h - pc).abs();
                let lc = (l - pc).abs();
                tr[i] = hl.max(hc.max(lc));
            }
        }
        Ok((tr, first))
    }

    fn first_valids_time_major(
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
    ) -> Result<Vec<i32>, CudaNatrError> {
        if cols == 0 || rows == 0 {
            return Err(CudaNatrError::InvalidInput("cols/rows zero".into()));
        }
        if high_tm.len() != cols * rows
            || low_tm.len() != cols * rows
            || close_tm.len() != cols * rows
        {
            return Err(CudaNatrError::InvalidInput(
                "time-major inputs wrong length".into(),
            ));
        if high_tm.len() != cols * rows
            || low_tm.len() != cols * rows
            || close_tm.len() != cols * rows
        {
            return Err(CudaNatrError::InvalidInput(
                "time-major inputs wrong length".into(),
            ));
        }
        let mut fv = vec![0i32; cols];
        for s in 0..cols {
            let mut first: i32 = rows as i32; // default: no valid
            for t in 0..rows {
                let idx = t * cols + s;
                if high_tm[idx].is_finite() && low_tm[idx].is_finite() && close_tm[idx].is_finite()
                {
                if high_tm[idx].is_finite() && low_tm[idx].is_finite() && close_tm[idx].is_finite()
                {
                    first = t as i32;
                    break;
                }
            }
            fv[s] = first;
        }
        Ok(fv)
    }

    // ----------------------- Public device entry points -----------------------
    pub fn natr_batch_dev(
        &mut self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &NatrBatchRange,
    ) -> Result<DeviceArrayF32, CudaNatrError> {
        let len = high.len();
        if len == 0 || low.len() != len || close.len() != len {
            return Err(CudaNatrError::InvalidInput(
                "mismatched or empty inputs".into(),
            ));
            return Err(CudaNatrError::InvalidInput(
                "mismatched or empty inputs".into(),
            ));
        }

        // Expand periods
        fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
            if step == 0 || start == end {
                return vec![start];
            }
            if step == 0 || start == end {
                return vec![start];
            }
            (start..=end).step_by(step).collect()
        }
        let periods_v = axis_usize(sweep.period);
        if periods_v.is_empty() {
            return Err(CudaNatrError::InvalidInput("no periods".into()));
        }
        if periods_v.is_empty() {
            return Err(CudaNatrError::InvalidInput("no periods".into()));
        }
        let (tr, first_valid) = Self::build_tr_one_series(high, low, close)?;

        let max_p = *periods_v.iter().max().unwrap();
        if len - first_valid < max_p {
            return Err(CudaNatrError::InvalidInput(format!(
                "not enough valid data (needed >= {}, valid = {})",
                max_p,
                len - first_valid
                max_p,
                len - first_valid
            )));
        }
        let periods_i32: Vec<i32> = periods_v.iter().map(|&p| p as i32).collect();
        let rows = periods_v.len();

        // Heuristic: enable shared inv_close100 if it will save real work
        let min_period = *periods_v.iter().min().unwrap();
        let warm_needed = first_valid + min_period - 1;
        let active_len = if len > warm_needed { len - warm_needed } else { 0 };
        let use_precompute = rows >= 4 || (rows * active_len >= 1_000_000);

        // VRAM checks (account for optional inv buffer)
        let out_bytes = rows * len * std::mem::size_of::<f32>();
        let in_bytes = tr.len() * std::mem::size_of::<f32>()
            + close.len() * std::mem::size_of::<f32>()
        let in_bytes = tr.len() * std::mem::size_of::<f32>()
            + close.len() * std::mem::size_of::<f32>()
            + periods_i32.len() * std::mem::size_of::<i32>();
        let head = Self::headroom_bytes();
        let extra_inv = if use_precompute { len * std::mem::size_of::<f32>() } else { 0 };
        let total = out_bytes + in_bytes + extra_inv + head;
        if !Self::will_fit(total, 0) {
            // try without the inv buffer; if still too big, bail
            let total_no_inv = out_bytes + in_bytes + head;
            if !Self::will_fit(total_no_inv, 0) {
                return Err(CudaNatrError::InvalidInput("insufficient VRAM for NATR batch".into()));
            }
        }

        // Upload inputs (async on NON_BLOCKING stream)
        let d_tr: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::from_slice_async(&tr, &self.stream)
                .map_err(|e| CudaNatrError::Cuda(e.to_string()))?
        };
        let d_close: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::from_slice_async(close, &self.stream)
                .map_err(|e| CudaNatrError::Cuda(e.to_string()))?
        };
        let d_periods: DeviceBuffer<i32> = unsafe {
            DeviceBuffer::from_slice_async(&periods_i32, &self.stream)
                .map_err(|e| CudaNatrError::Cuda(e.to_string()))?
        };
        let mut d_out: DeviceBuffer<f32> = match unsafe { DeviceBuffer::uninitialized_async(rows * len, &self.stream) } {
            Ok(buf) => buf,
            Err(_) => unsafe { DeviceBuffer::uninitialized(rows * len) }
                .map_err(|e| CudaNatrError::Cuda(e.to_string()))?,
        };

        // Optionally build inv_close100 on device
        let mut d_inv: Option<DeviceBuffer<f32>> = None;
        if use_precompute && Self::will_fit(out_bytes + in_bytes + (len * std::mem::size_of::<f32>()) + head, 0) {
            // Additional VRAM check for the extra vector
            let mut d = match unsafe { DeviceBuffer::<f32>::uninitialized_async(len, &self.stream) } {
                Ok(buf) => buf,
                Err(_) => unsafe { DeviceBuffer::<f32>::uninitialized(len) }
                    .map_err(|e| CudaNatrError::Cuda(e.to_string()))?,
            };
            let make_fn = self
                .module
                .get_function("natr_make_inv_close100")
                .map_err(|e| CudaNatrError::Cuda(e.to_string()))?;
            unsafe {
                let mut c_ptr = d_close.as_device_ptr().as_raw();
                let mut len_i = len as i32;
                let mut inv_ptr = d.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut c_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut inv_ptr as *mut _ as *mut c_void,
                ];
                let grid_x = ((len as u32) + 255) / 256;
                let grid: GridSize = (grid_x, 1, 1).into();
                let block: BlockSize = (256u32, 1, 1).into();
                self.stream
                    .launch(&make_fn, grid, block, 0, args)
                    .map_err(|e| CudaNatrError::Cuda(e.to_string()))?;
            }
            d_inv = Some(d);
        }

        // Choose kernel symbol based on availability of precompute buffer
        let func = if d_inv.is_some() {
            self.module
                .get_function("natr_batch_f32_with_inv")
                .map_err(|e| CudaNatrError::Cuda(e.to_string()))?
        } else {
            self.module
                .get_function("natr_batch_f32")
                .map_err(|e| CudaNatrError::Cuda(e.to_string()))?
        };

        let block_x = match self.policy.batch {
            BatchKernelPolicy::Auto => 256u32,
            BatchKernelPolicy::Plain { block_x } => block_x.max(32).min(1024),
        };
        let grid_x = rows as u32; // one block per period row
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            if let Some(mut d_inv) = d_inv {
                let mut tr_ptr = d_tr.as_device_ptr().as_raw();
                let mut inv_ptr = d_inv.as_device_ptr().as_raw();
                let mut periods_ptr = d_periods.as_device_ptr().as_raw();
                let mut len_i = len as i32;
                let mut first_i = first_valid as i32;
                let mut rows_i = rows as i32;
                let mut out_ptr = d_out.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut tr_ptr as *mut _ as *mut c_void,
                    &mut inv_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaNatrError::Cuda(e.to_string()))?;
            } else {
                let mut tr_ptr = d_tr.as_device_ptr().as_raw();
                let mut close_ptr = d_close.as_device_ptr().as_raw();
                let mut periods_ptr = d_periods.as_device_ptr().as_raw();
                let mut len_i = len as i32;
                let mut first_i = first_valid as i32;
                let mut rows_i = rows as i32;
                let mut out_ptr = d_out.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut tr_ptr as *mut _ as *mut c_void,
                    &mut close_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaNatrError::Cuda(e.to_string()))?;
            }
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaNatrError::Cuda(e.to_string()))?;

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

    pub fn natr_many_series_one_param_time_major_dev(
        &mut self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaNatrError> {
        if cols == 0 || rows == 0 {
            return Err(CudaNatrError::InvalidInput("cols/rows zero".into()));
        }
        if high_tm.len() != cols * rows
            || low_tm.len() != cols * rows
            || close_tm.len() != cols * rows
        {
            return Err(CudaNatrError::InvalidInput(
                "time-major inputs wrong length".into(),
            ));
        if cols == 0 || rows == 0 {
            return Err(CudaNatrError::InvalidInput("cols/rows zero".into()));
        }
        if high_tm.len() != cols * rows
            || low_tm.len() != cols * rows
            || close_tm.len() != cols * rows
        {
            return Err(CudaNatrError::InvalidInput(
                "time-major inputs wrong length".into(),
            ));
        }
        if period == 0 {
            return Err(CudaNatrError::InvalidInput("period must be > 0".into()));
        }
        if period == 0 {
            return Err(CudaNatrError::InvalidInput("period must be > 0".into()));
        }

        let first_valids = Self::first_valids_time_major(high_tm, low_tm, close_tm, cols, rows)?;

        // VRAM estimate and check
        let out_bytes = cols * rows * std::mem::size_of::<f32>();
        let in_bytes = (high_tm.len() + low_tm.len() + close_tm.len()) * std::mem::size_of::<f32>()
            + first_valids.len() * std::mem::size_of::<i32>();
        let total = out_bytes + in_bytes + Self::headroom_bytes();
        if !Self::will_fit(total, 0) {
            return Err(CudaNatrError::InvalidInput(
                "insufficient VRAM for NATR many-series".into(),
            ));
            return Err(CudaNatrError::InvalidInput(
                "insufficient VRAM for NATR many-series".into(),
            ));
        }

        // Upload inputs (async on our stream)
        let d_high: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::from_slice_async(high_tm, &self.stream)
                .map_err(|e| CudaNatrError::Cuda(e.to_string()))?
        };
        let d_low: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::from_slice_async(low_tm, &self.stream)
                .map_err(|e| CudaNatrError::Cuda(e.to_string()))?
        };
        let d_close: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::from_slice_async(close_tm, &self.stream)
                .map_err(|e| CudaNatrError::Cuda(e.to_string()))?
        };
        let d_fv: DeviceBuffer<i32> = unsafe {
            DeviceBuffer::from_slice_async(&first_valids, &self.stream)
                .map_err(|e| CudaNatrError::Cuda(e.to_string()))?
        };
        let mut d_out: DeviceBuffer<f32> = match unsafe { DeviceBuffer::uninitialized_async(cols * rows, &self.stream) } {
            Ok(buf) => buf,
            Err(_) => unsafe { DeviceBuffer::uninitialized(cols * rows) }
                .map_err(|e| CudaNatrError::Cuda(e.to_string()))?,
        };

        // Launch
        let func = self
            .module
            .get_function("natr_many_series_one_param_f32")
            .map_err(|e| CudaNatrError::Cuda(e.to_string()))?;

        // Choose 4 warps per block by default (128 threads), tuneable via policy
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 128u32,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32).min(1024),
        };
        let warps_per_block = (block_x / 32).max(1);
        let grid_x = ((cols as u32) + warps_per_block - 1) / warps_per_block;
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut h_ptr = d_high.as_device_ptr().as_raw();
            let mut l_ptr = d_low.as_device_ptr().as_raw();
            let mut c_ptr = d_close.as_device_ptr().as_raw();
            let mut per_i = period as i32;
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut fv_ptr = d_fv.as_device_ptr().as_raw();
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut h_ptr as *mut _ as *mut c_void,
                &mut l_ptr as *mut _ as *mut c_void,
                &mut c_ptr as *mut _ as *mut c_void,
                &mut per_i as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaNatrError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaNatrError::Cuda(e.to_string()))?;

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
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series(n_combos: usize) -> usize {
        // 3 inputs + TR + periods + output
        let in_bytes = 3 * ONE_SERIES_LEN * std::mem::size_of::<f32>()
            + ONE_SERIES_LEN * std::mem::size_of::<f32>()
            + n_combos * std::mem::size_of::<i32>();
        let out_bytes = n_combos * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    fn synth_hlc_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i];
            if !v.is_finite() {
                continue;
            }
            if !v.is_finite() {
                continue;
            }
            let x = i as f32 * 0.0031;
            let off = (0.002 * x.sin()).abs() + 0.5;
            high[i] = v + off;
            low[i] = v - off;
        }
        (high, low)
    }

    struct NatrBatchState {
        cuda: CudaNatr,
        high: Vec<f32>,
        low: Vec<f32>,
        close: Vec<f32>,
        sweep: NatrBatchRange,
    }
    impl CudaBenchState for NatrBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .natr_batch_dev(&self.high, &self.low, &self.close, &self.sweep)
                .expect("natr batch dev");
        }
    }

    fn prep_one_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaNatr::new(0).expect("cuda natr");
        let close = gen_series(ONE_SERIES_LEN);
        let (high, low) = synth_hlc_from_close(&close);
        let sweep = NatrBatchRange { period: (7, 64, 3) };
        Box::new(NatrBatchState {
            cuda,
            high,
            low,
            close,
            sweep,
        })
        Box::new(NatrBatchState {
            cuda,
            high,
            low,
            close,
            sweep,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "natr",
            "one_series_many_params",
            "natr_cuda_batch",
            "1m",
            prep_one_series,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series(20))]
    }
}
