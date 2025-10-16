#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::wavetrend::{WavetrendBatchRange, WavetrendParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, CopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
pub enum CudaWavetrendError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaWavetrendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaWavetrendError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaWavetrendError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaWavetrendError {}

// -------- Kernel selection policy (parity with ALMA style) --------

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
pub struct CudaWavetrendPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaWavetrendPolicy {
    fn default() -> Self {
        Self { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Auto }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaWavetrend {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaWavetrendPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

pub struct CudaWavetrendBatch {
    pub wt1: DeviceArrayF32,
    pub wt2: DeviceArrayF32,
    pub wt_diff: DeviceArrayF32,
    pub combos: Vec<WavetrendParams>,
}

struct PreparedBatch {
    combos: Vec<WavetrendParams>,
    first_valid: usize,
    series_len: usize,
}

impl CudaWavetrend {
    pub fn new(device_id: usize) -> Result<Self, CudaWavetrendError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;

        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/wavetrend_kernel.ptx"));
        // High optimization with context-derived target, then graceful fallbacks.
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
                    Module::from_ptx(ptx, &[])
                        .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaWavetrendPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(device_id: usize, policy: CudaWavetrendPolicy) -> Result<Self, CudaWavetrendError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaWavetrendPolicy) { self.policy = policy; }
    pub fn policy(&self) -> &CudaWavetrendPolicy { &self.policy }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged { return; }
        if env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_scenario = env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] Wavetrend batch selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaWavetrend)).debug_batch_logged = true; }
            }
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged { return; }
        if env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per_scenario = env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] Wavetrend many-series selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaWavetrend)).debug_many_logged = true; }
            }
        }
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") { Ok(v) => v != "0" && v.to_lowercase() != "false", Err(_) => true }
    }
    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() { return true; }
        if let Ok((free, _total)) = mem_get_info() { required_bytes.saturating_add(headroom_bytes) <= free } else { true }
    }

    pub fn wavetrend_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &WavetrendBatchRange,
    ) -> Result<CudaWavetrendBatch, CudaWavetrendError> {
        let PreparedBatch {
            combos,
            first_valid,
            series_len,
        } = Self::prepare_batch_inputs(data_f32, sweep)?;
        let rows = combos.len();

        // VRAM estimate and guard
        let prices_bytes = series_len * std::mem::size_of::<f32>();
        let params_bytes = 3 * rows * std::mem::size_of::<i32>() + rows * std::mem::size_of::<f32>();
        let out_bytes = 3 * rows * series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + params_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaWavetrendError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                required as f64 / (1024.0*1024.0)
            )));
        }

        let d_prices = DeviceBuffer::from_slice(data_f32)
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;

        let (channels, averages, mas, factors) = Self::build_param_arrays(&combos)?;
        let d_channels = DeviceBuffer::from_slice(&channels)
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;
        let d_averages = DeviceBuffer::from_slice(&averages)
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;
        let d_mas =
            DeviceBuffer::from_slice(&mas).map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;
        let d_factors = DeviceBuffer::from_slice(&factors)
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;

        let mut d_wt1: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(rows * series_len) }
                .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;
        let mut d_wt2: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(rows * series_len) }
                .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;
        let mut d_wt_diff: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(rows * series_len) }
                .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;

        self.launch_kernel(
            &d_prices,
            &d_channels,
            &d_averages,
            &d_mas,
            &d_factors,
            first_valid,
            series_len,
            rows,
            &mut d_wt1,
            &mut d_wt2,
            &mut d_wt_diff,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;

        Ok(CudaWavetrendBatch {
            wt1: DeviceArrayF32 {
                buf: d_wt1,
                rows,
                cols: series_len,
            },
            wt2: DeviceArrayF32 {
                buf: d_wt2,
                rows,
                cols: series_len,
            },
            wt_diff: DeviceArrayF32 {
                buf: d_wt_diff,
                rows,
                cols: series_len,
            },
            combos,
        })
    }

    pub fn wavetrend_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &WavetrendBatchRange,
        out_wt1: &mut [f32],
        out_wt2: &mut [f32],
        out_wt_diff: &mut [f32],
    ) -> Result<(usize, usize, Vec<WavetrendParams>), CudaWavetrendError> {
        let batch = self.wavetrend_batch_dev(data_f32, sweep)?;
        let rows = batch.wt1.rows;
        let cols = batch.wt1.cols;
        let expected = rows * cols;
        if out_wt1.len() != expected || out_wt2.len() != expected || out_wt_diff.len() != expected {
            return Err(CudaWavetrendError::InvalidInput(format!(
                "output slices have wrong length (expected {})",
                expected
            )));
        }

        batch
            .wt1
            .buf
            .copy_to(out_wt1)
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;
        batch
            .wt2
            .buf
            .copy_to(out_wt2)
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;
        batch
            .wt_diff
            .buf
            .copy_to(out_wt_diff)
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;

        Ok((rows, cols, batch.combos))
    }

    pub fn wavetrend_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        combos: &[WavetrendParams],
        first_valid: usize,
        series_len: usize,
        d_wt1: &mut DeviceBuffer<f32>,
        d_wt2: &mut DeviceBuffer<f32>,
        d_wt_diff: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWavetrendError> {
        if combos.is_empty() {
            return Err(CudaWavetrendError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        if series_len == 0 {
            return Err(CudaWavetrendError::InvalidInput(
                "series_len is zero".into(),
            ));
        }
        if d_prices.len() != series_len {
            return Err(CudaWavetrendError::InvalidInput(format!(
                "price buffer len {} != series_len {}",
                d_prices.len(),
                series_len
            )));
        }

        let (channels, averages, mas, factors) = Self::build_param_arrays(combos)?;
        let d_channels = DeviceBuffer::from_slice(&channels)
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;
        let d_averages = DeviceBuffer::from_slice(&averages)
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;
        let d_mas =
            DeviceBuffer::from_slice(&mas).map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;
        let d_factors = DeviceBuffer::from_slice(&factors)
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;

        self.launch_kernel(
            d_prices,
            &d_channels,
            &d_averages,
            &d_mas,
            &d_factors,
            first_valid,
            series_len,
            combos.len(),
            d_wt1,
            d_wt2,
            d_wt_diff,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_channels: &DeviceBuffer<i32>,
        d_averages: &DeviceBuffer<i32>,
        d_mas: &DeviceBuffer<i32>,
        d_factors: &DeviceBuffer<f32>,
        first_valid: usize,
        series_len: usize,
        rows: usize,
        d_wt1: &mut DeviceBuffer<f32>,
        d_wt2: &mut DeviceBuffer<f32>,
        d_wt_diff: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWavetrendError> {
        if series_len == 0 {
            return Err(CudaWavetrendError::InvalidInput(
                "series_len is zero".into(),
            ));
        }
        if d_prices.len() != series_len {
            return Err(CudaWavetrendError::InvalidInput(format!(
                "price buffer len {} != series_len {}",
                d_prices.len(),
                series_len
            )));
        }
        if d_channels.len() != rows
            || d_averages.len() != rows
            || d_mas.len() != rows
            || d_factors.len() != rows
        {
            return Err(CudaWavetrendError::InvalidInput(
                "parameter buffers must match number of combinations".into(),
            ));
        }
        let expected = rows * series_len;
        if d_wt1.len() != expected || d_wt2.len() != expected || d_wt_diff.len() != expected {
            return Err(CudaWavetrendError::InvalidInput(format!(
                "output buffer mismatch: expected {} entries per output",
                expected
            )));
        }
        if series_len > i32::MAX as usize {
            return Err(CudaWavetrendError::InvalidInput(
                "series length exceeds i32::MAX".into(),
            ));
        }
        if rows > i32::MAX as usize {
            return Err(CudaWavetrendError::InvalidInput(
                "row count exceeds i32::MAX".into(),
            ));
        }

        let func = self
            .module
            .get_function("wavetrend_batch_f32")
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;

        // Policy-based block size (default 128)
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x.max(32),
            BatchKernelPolicy::Auto => 128,
        };
        // Record selection for debug once
        unsafe { (*(self as *const _ as *mut CudaWavetrend)).last_batch = Some(BatchKernelSelected::Plain { block_x }); }
        self.maybe_log_batch_debug();

        // Chunk rows so each launch handles at most MAX_ROWS_PER_LAUNCH combos.
        const MAX_ROWS_PER_LAUNCH: usize = 65_535; // conservative cap
        let mut launched = 0usize;
        while launched < rows {
            let count = (rows - launched).min(MAX_ROWS_PER_LAUNCH);
            let grid_x = ((count as u32) + block_x - 1) / block_x;
            let grid: GridSize = (grid_x.max(1), 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();

            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                let mut len_i = series_len as i32;
                let mut first_i = first_valid as i32;
                let mut rows_i = count as i32;
                let mut ch_ptr = d_channels.as_device_ptr().add(launched).as_raw();
                let mut avg_ptr = d_averages.as_device_ptr().add(launched).as_raw();
                let mut ma_ptr = d_mas.as_device_ptr().add(launched).as_raw();
                let mut factor_ptr = d_factors.as_device_ptr().add(launched).as_raw();
                let base = launched * series_len;
                let mut wt1_ptr = d_wt1.as_device_ptr().add(base).as_raw();
                let mut wt2_ptr = d_wt2.as_device_ptr().add(base).as_raw();
                let mut diff_ptr = d_wt_diff.as_device_ptr().add(base).as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut ch_ptr as *mut _ as *mut c_void,
                    &mut avg_ptr as *mut _ as *mut c_void,
                    &mut ma_ptr as *mut _ as *mut c_void,
                    &mut factor_ptr as *mut _ as *mut c_void,
                    &mut wt1_ptr as *mut _ as *mut c_void,
                    &mut wt2_ptr as *mut _ as *mut c_void,
                    &mut diff_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;
            }
            launched += count;
        }

        Ok(())
    }

    fn prepare_batch_inputs(
        data: &[f32],
        sweep: &WavetrendBatchRange,
    ) -> Result<PreparedBatch, CudaWavetrendError> {
        if data.is_empty() {
            return Err(CudaWavetrendError::InvalidInput("empty data".into()));
        }
        let first_valid = data
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaWavetrendError::InvalidInput("all values are NaN".into()))?;
        let series_len = data.len();
        let combos = Self::expand_range(sweep);
        if combos.is_empty() {
            return Err(CudaWavetrendError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        if series_len > i32::MAX as usize {
            return Err(CudaWavetrendError::InvalidInput(
                "series length exceeds i32::MAX (unsupported)".into(),
            ));
        }
        if combos.len() > i32::MAX as usize {
            return Err(CudaWavetrendError::InvalidInput(
                "combination count exceeds i32::MAX (unsupported)".into(),
            ));
        }

        for (idx, combo) in combos.iter().enumerate() {
            let ch = combo.channel_length.unwrap_or(0);
            let avg = combo.average_length.unwrap_or(0);
            let ma = combo.ma_length.unwrap_or(0);
            if ch == 0 || avg == 0 || ma == 0 {
                return Err(CudaWavetrendError::InvalidInput(format!(
                    "invalid periods at combo {} (ch={}, avg={}, ma={})",
                    idx, ch, avg, ma
                )));
            }
            if ch > series_len || avg > series_len || ma > series_len {
                return Err(CudaWavetrendError::InvalidInput(format!(
                    "period exceeds series length at combo {}",
                    idx
                )));
            }
            let needed = ch.max(avg).max(ma);
            let valid = series_len - first_valid;
            if valid < needed {
                return Err(CudaWavetrendError::InvalidInput(format!(
                    "not enough valid data for combo {} (needed {}, valid {})",
                    idx, needed, valid
                )));
            }
        }

        Ok(PreparedBatch {
            combos,
            first_valid,
            series_len,
        })
    }

    fn build_param_arrays(
        combos: &[WavetrendParams],
    ) -> Result<(Vec<i32>, Vec<i32>, Vec<i32>, Vec<f32>), CudaWavetrendError> {
        let mut channels = Vec::with_capacity(combos.len());
        let mut averages = Vec::with_capacity(combos.len());
        let mut mas = Vec::with_capacity(combos.len());
        let mut factors = Vec::with_capacity(combos.len());
        for combo in combos {
            channels.push(combo.channel_length.unwrap_or(0) as i32);
            averages.push(combo.average_length.unwrap_or(0) as i32);
            mas.push(combo.ma_length.unwrap_or(0) as i32);
            factors.push(combo.factor.unwrap_or(0.015) as f32);
        }
        Ok((channels, averages, mas, factors))
    }

    fn expand_range(sweep: &WavetrendBatchRange) -> Vec<WavetrendParams> {
        fn axis_usize(axis: (usize, usize, usize)) -> Vec<usize> {
            let (start, end, step) = axis;
            if step == 0 || start == end {
                return vec![start];
            }
            (start..=end).step_by(step).collect()
        }
        fn axis_f64(axis: (f64, f64, f64)) -> Vec<f64> {
            let (start, end, step) = axis;
            if step.abs() < f64::EPSILON || (start - end).abs() < f64::EPSILON {
                return vec![start];
            }
            let mut out = Vec::new();
            let mut v = start;
            while v <= end + f64::EPSILON {
                out.push(v);
                v += step;
            }
            out
        }

        let channels = axis_usize(sweep.channel_length);
        let averages = axis_usize(sweep.average_length);
        let mas = axis_usize(sweep.ma_length);
        let factors = axis_f64(sweep.factor);

        let mut combos =
            Vec::with_capacity(channels.len() * averages.len() * mas.len() * factors.len());
        for &ch in &channels {
            for &avg in &averages {
                for &ma in &mas {
                    for &f in &factors {
                        combos.push(WavetrendParams {
                            channel_length: Some(ch),
                            average_length: Some(avg),
                            ma_length: Some(ma),
                            factor: Some(f),
                        });
                    }
                }
            }
        }
        combos
    }

    // ---------- Many-series × one-param (time-major) ----------
    pub fn wavetrend_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &WavetrendParams,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32, DeviceArrayF32), CudaWavetrendError> {
        let (first_valids, ch, avg, ma, factor) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        // VRAM estimate: inputs + first_valids + 3 outputs
        let elems = cols * rows;
        let in_bytes = elems * std::mem::size_of::<f32>();
        let fv_bytes = cols * std::mem::size_of::<i32>();
        let out_bytes = 3 * elems * std::mem::size_of::<f32>();
        let required = in_bytes + fv_bytes + out_bytes;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaWavetrendError::InvalidInput(
                "insufficient device memory for wavetrend many-series".into(),
            ));
        }

        let d_prices = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;
        let mut d_wt1: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }
                .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;
        let mut d_wt2: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }
                .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;
        let mut d_wt_diff: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }
                .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices,
            cols,
            rows,
            ch,
            avg,
            ma,
            factor,
            &d_first,
            &mut d_wt1,
            &mut d_wt2,
            &mut d_wt_diff,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;

        Ok((
            DeviceArrayF32 { buf: d_wt1, rows, cols },
            DeviceArrayF32 { buf: d_wt2, rows, cols },
            DeviceArrayF32 { buf: d_wt_diff, rows, cols },
        ))
    }

    pub fn wavetrend_many_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &WavetrendParams,
        wt1_tm: &mut [f32],
        wt2_tm: &mut [f32],
        wt_diff_tm: &mut [f32],
    ) -> Result<(), CudaWavetrendError> {
        let expected = cols.checked_mul(rows).ok_or_else(|| CudaWavetrendError::InvalidInput("size overflow".into()))?;
        if wt1_tm.len() != expected || wt2_tm.len() != expected || wt_diff_tm.len() != expected {
            return Err(CudaWavetrendError::InvalidInput("output slices must be cols*rows".into()));
        }
        let (wt1, wt2, wt_diff) =
            self.wavetrend_many_series_one_param_time_major_dev(data_tm_f32, cols, rows, params)?;
        wt1.buf.copy_to(wt1_tm).map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;
        wt2.buf.copy_to(wt2_tm).map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;
        wt_diff
            .buf
            .copy_to(wt_diff_tm)
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;
        Ok(())
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &WavetrendParams,
    ) -> Result<(Vec<i32>, i32, i32, i32, f32), CudaWavetrendError> {
        if cols == 0 || rows == 0 {
            return Err(CudaWavetrendError::InvalidInput("cols or rows is zero".into()));
        }
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaWavetrendError::InvalidInput("size overflow".into()))?;
        if data_tm_f32.len() != elems {
            return Err(CudaWavetrendError::InvalidInput(
                "data length must be time-major cols*rows".into(),
            ));
        }
        let ch = params.channel_length.unwrap_or(9) as i32;
        let avg = params.average_length.unwrap_or(12) as i32;
        let ma = params.ma_length.unwrap_or(3) as i32;
        let factor = params.factor.unwrap_or(0.015) as f32;
        if ch <= 0 || avg <= 0 || ma <= 0 {
            return Err(CudaWavetrendError::InvalidInput(
                "periods must be positive".into(),
            ));
        }
        let need = ch.max(avg).max(ma) as usize;
        // Per-series first-valid indices
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv: Option<i32> = None;
            for t in 0..rows {
                if !data_tm_f32[t * cols + s].is_nan() {
                    fv = Some(t as i32);
                    break;
                }
            }
            let fv = fv.ok_or_else(|| CudaWavetrendError::InvalidInput(format!("series {} all NaN", s)))?;
            if (rows as i32) - fv < (need as i32) {
                return Err(CudaWavetrendError::InvalidInput(format!(
                    "series {} not enough valid data (needed >= {}, valid = {})",
                    s,
                    need,
                    (rows as i32) - fv
                )));
            }
            first_valids[s] = fv;
        }
        Ok((first_valids, ch, avg, ma, factor))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        ch: i32,
        avg: i32,
        ma: i32,
        factor: f32,
        d_first_valids: &DeviceBuffer<i32>,
        d_wt1: &mut DeviceBuffer<f32>,
        d_wt2: &mut DeviceBuffer<f32>,
        d_wt_diff: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWavetrendError> {
        if cols == 0 || rows == 0 { return Ok(()); }
        let func = self
            .module
            .get_function("wavetrend_many_series_one_param_time_major_f32")
            .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;

        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32),
            ManySeriesKernelPolicy::Auto => 256,
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut p_prices = d_prices_tm.as_device_ptr().as_raw();
            let mut p_cols = cols as i32;
            let mut p_rows = rows as i32;
            let mut p_ch = ch;
            let mut p_avg = avg;
            let mut p_ma = ma;
            let mut p_factor = factor;
            let mut p_first = d_first_valids.as_device_ptr().as_raw();
            let mut p_wt1 = d_wt1.as_device_ptr().as_raw();
            let mut p_wt2 = d_wt2.as_device_ptr().as_raw();
            let mut p_diff = d_wt_diff.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_prices as *mut _ as *mut c_void,
                &mut p_cols as *mut _ as *mut c_void,
                &mut p_rows as *mut _ as *mut c_void,
                &mut p_ch as *mut _ as *mut c_void,
                &mut p_avg as *mut _ as *mut c_void,
                &mut p_ma as *mut _ as *mut c_void,
                &mut p_factor as *mut _ as *mut c_void,
                &mut p_first as *mut _ as *mut c_void,
                &mut p_wt1 as *mut _ as *mut c_void,
                &mut p_wt2 as *mut _ as *mut c_void,
                &mut p_diff as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaWavetrendError::Cuda(e.to_string()))?;
        }
        // Record selection and maybe log
        unsafe { (*(self as *const _ as *mut CudaWavetrend)).last_many = Some(ManySeriesKernelSelected::OneD { block_x }); }
        self.maybe_log_many_debug();
        Ok(())
    }
}

// ---------- Bench profiles (batch only) ----------

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;
    const MANY_SERIES_COLS: usize = 250;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        // 3 outputs
        let out_bytes = 3 * ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = elems * std::mem::size_of::<f32>();
        let out_bytes = 3 * elems * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct WtBatchState {
        cuda: CudaWavetrend,
        price: Vec<f32>,
        sweep: WavetrendBatchRange,
    }
    impl CudaBenchState for WtBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .wavetrend_batch_dev(&self.price, &self.sweep)
                .expect("wavetrend batch");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaWavetrend::new(0).expect("cuda wavetrend");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = WavetrendBatchRange {
            channel_length: (10, 10 + PARAM_SWEEP - 1, 1),
            average_length: (21, 21, 0),
            ma_length: (4, 4, 0),
            factor: (0.015, 0.015, 0.0),
        };
        Box::new(WtBatchState { cuda, price, sweep })
    }

    struct WtManySeriesState {
        cuda: CudaWavetrend,
        tm: Vec<f32>,
    }
    impl CudaBenchState for WtManySeriesState {
        fn launch(&mut self) {
            let params = WavetrendParams { channel_length: Some(10), average_length: Some(21), ma_length: Some(4), factor: Some(0.015) };
            let _ = self
                .cuda
                .wavetrend_many_series_one_param_time_major_dev(&self.tm, MANY_SERIES_COLS, MANY_SERIES_LEN, &params)
                .expect("wavetrend many-series");
        }
    }
    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaWavetrend::new(0).expect("cuda wavetrend");
        let tm = gen_time_major_prices(MANY_SERIES_COLS, MANY_SERIES_LEN);
        Box::new(WtManySeriesState { cuda, tm })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "wavetrend",
                "one_series_many_params",
                "wavetrend_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "wavetrend",
                "many_series_one_param",
                "wavetrend_cuda_many_series_one_param_dev",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
