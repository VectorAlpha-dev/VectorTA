//! CUDA wrapper for ATR (Average True Range) kernels.
//!
//! Parity goals:
//! - ALMA-style device buffer API returning `DeviceArrayF32`.
//! - Batch (one-series × many-params) and Many-series × one-param (time-major).
//! - NaN warmup identical to scalar: warm = first_valid + period - 1.
//! - VRAM check + simple chunking for large combo counts.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::atr::AtrBatchRange;
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
pub enum CudaAtrError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaAtrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaAtrError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaAtrError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl std::error::Error for CudaAtrError {}

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
pub struct CudaAtrPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaAtrPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

pub struct CudaAtr {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaAtrPolicy,
}

impl CudaAtr {
    pub fn new(device_id: usize) -> Result<Self, CudaAtrError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaAtrError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaAtrError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaAtrError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/atr_kernel.ptx"));
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
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaAtrError::Cuda(e.to_string()))?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaAtrError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaAtrPolicy::default(),
        })
    }

    pub fn set_policy(&mut self, policy: CudaAtrPolicy) {
        self.policy = policy;
    }
    pub fn synchronize(&self) -> Result<(), CudaAtrError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaAtrError::Cuda(e.to_string()))
    }

    fn first_valid_hlc(high: &[f32], low: &[f32], close: &[f32]) -> Result<usize, CudaAtrError> {
        if high.len() == 0 || low.len() == 0 || close.len() == 0 {
            return Err(CudaAtrError::InvalidInput("empty input".into()));
        }
        let len = high.len().min(low.len()).min(close.len());
        for i in 0..len {
            if !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan() {
                return Ok(i);
            }
        }
        Err(CudaAtrError::InvalidInput("all values are NaN".into()))
    }

    fn device_will_fit(bytes: usize, headroom: usize) -> bool {
        let check = match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        };
        if !check {
            return true;
        }
        if let Ok((free, _)) = mem_get_info() {
            bytes.saturating_add(headroom) <= free
        } else {
            true
        }
    }

    fn chunk_size_for_batch(n_combos: usize, len: usize) -> usize {
        // Inputs: 3×len f32; params per combo (periods i32, alphas f32, warms i32); outputs: combos×len f32.
        let input_bytes = 3 * len * std::mem::size_of::<f32>();
        let params_bytes = n_combos * (std::mem::size_of::<i32>() * 2 + std::mem::size_of::<f32>());
        let out_per_combo = len * std::mem::size_of::<f32>();
        let headroom = 64 * 1024 * 1024; // ~64MB
                                         // Start from all combos and shrink until it fits.
        let mut chunk = n_combos.max(1);
        while chunk > 1 {
            let need = input_bytes + params_bytes + chunk * out_per_combo + headroom;
            if Self::device_will_fit(need, 0) {
                break;
            }
            chunk = (chunk + 1) / 2;
        }
        chunk.max(1)
    }

    pub fn atr_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &AtrBatchRange,
    ) -> Result<DeviceArrayF32, CudaAtrError> {
        if high.len() != low.len() || low.len() != close.len() {
            return Err(CudaAtrError::InvalidInput("input length mismatch".into()));
        }
        let len = close.len();
        if len == 0 {
            return Err(CudaAtrError::InvalidInput("empty input".into()));
        }
        let first_valid = Self::first_valid_hlc(high, low, close)?;

        // Expand parameter combos (length axis only)
        let (start, end, step) = sweep.length;
        if start == 0 {
            return Err(CudaAtrError::InvalidInput("period must be > 0".into()));
        }
        let periods: Vec<usize> = if step == 0 || start == end {
            vec![start]
        } else {
            (start..=end).step_by(step).collect()
        };
        if periods.is_empty() {
            return Err(CudaAtrError::InvalidInput("no parameter combos".into()));
        }
        for &p in &periods {
            if p == 0 || p > len || (len - first_valid) < p {
                return Err(CudaAtrError::InvalidInput(format!(
                    "invalid period {} for data length {} (valid after {}: {})",
                    p,
                    len,
                    first_valid,
                    len - first_valid
                )));
            }
        }

        let n_combos = periods.len();
        // Precompute TR and prefix sums on host (shared across combos)
        let mut tr = vec![0f32; len];
        // compute TR starting from first_valid; earlier entries unused but keep zeros
        let mut prev_c = close[first_valid] as f32; // used only from first_valid+1
        for t in first_valid..len {
            let hi = high[t] as f32;
            let lo = low[t] as f32;
            if t == first_valid {
                tr[t] = hi - lo;
            } else {
                let mut tri = hi - lo;
                let hc = (hi - prev_c).abs();
                if hc > tri {
                    tri = hc;
                }
                let lc = (lo - prev_c).abs();
                if lc > tri {
                    tri = lc;
                }
                tr[t] = tri;
            }
            prev_c = close[t] as f32;
        }
        let mut prefix = vec![0f64; len + 1];
        for i in 0..len {
            prefix[i + 1] = prefix[i] + (tr[i] as f64);
        }

        let d_tr = DeviceBuffer::from_slice(&tr).map_err(|e| CudaAtrError::Cuda(e.to_string()))?;
        let d_prefix =
            DeviceBuffer::from_slice(&prefix).map_err(|e| CudaAtrError::Cuda(e.to_string()))?;

        // Prepare params on device
        let mut h_periods_i32: Vec<i32> = periods.iter().map(|&p| p as i32).collect();
        let mut h_alphas: Vec<f32> = periods.iter().map(|&p| (1.0f32 / (p as f32))).collect();
        let mut h_warms: Vec<i32> = periods
            .iter()
            .map(|&p| (first_valid + p - 1) as i32)
            .collect();
        let d_periods = DeviceBuffer::from_slice(&h_periods_i32)
            .map_err(|e| CudaAtrError::Cuda(e.to_string()))?;
        let d_alphas =
            DeviceBuffer::from_slice(&h_alphas).map_err(|e| CudaAtrError::Cuda(e.to_string()))?;
        let d_warms =
            DeviceBuffer::from_slice(&h_warms).map_err(|e| CudaAtrError::Cuda(e.to_string()))?;

        // Output buffer
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n_combos * len) }
            .map_err(|e| CudaAtrError::Cuda(e.to_string()))?;

        // Kernel
        // Prefer shared-precompute kernel; fallback to on-the-fly TR kernel if symbol missing
        let func = match self.module.get_function("atr_batch_from_tr_prefix_f32") {
            Ok(f) => f,
            Err(_) => self
                .module
                .get_function("atr_batch_f32")
                .map_err(|e| CudaAtrError::Cuda(e.to_string()))?,
        };

        // Chunk combos if needed
        let chunk = Self::chunk_size_for_batch(n_combos, len);
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            BatchKernelPolicy::Auto => 256,
        };

        let mut launched = 0usize;
        while launched < n_combos {
            let cur = (n_combos - launched).min(chunk);
            let grid: GridSize = (cur as u32, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();

            unsafe {
                let mut tr_ptr = d_tr.as_device_ptr().as_raw();
                let mut prefix_ptr = d_prefix.as_device_ptr().as_raw();
                let mut periods_ptr = d_periods
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * std::mem::size_of::<i32>()) as u64);
                let mut alphas_ptr = d_alphas
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * std::mem::size_of::<f32>()) as u64);
                let mut warms_ptr = d_warms
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * std::mem::size_of::<i32>()) as u64);
                let mut len_i = len as i32;
                let mut first_i = first_valid as i32;
                let mut cur_i = cur as i32;
                let mut out_ptr = d_out
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * len * std::mem::size_of::<f32>()) as u64);

                let args: &mut [*mut c_void] = &mut [
                    &mut tr_ptr as *mut _ as *mut c_void,
                    &mut prefix_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut alphas_ptr as *mut _ as *mut c_void,
                    &mut warms_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut cur_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaAtrError::Cuda(e.to_string()))?;
            }

            launched += cur;
        }

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: len,
        })
    }

    fn first_valids_time_major(
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
    ) -> Result<Vec<i32>, CudaAtrError> {
        let n = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaAtrError::InvalidInput("rows*cols overflow".into()))?;
        if high_tm.len() != n || low_tm.len() != n || close_tm.len() != n {
            return Err(CudaAtrError::InvalidInput(
                "time-major input length mismatch".into(),
            ));
        }
        let mut out = vec![-1i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                let h = high_tm[idx];
                let l = low_tm[idx];
                let c = close_tm[idx];
                if !h.is_nan() && !l.is_nan() && !c.is_nan() {
                    out[s] = t as i32;
                    break;
                }
            }
        }
        Ok(out)
    }

    pub fn atr_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaAtrError> {
        if period == 0 {
            return Err(CudaAtrError::InvalidInput("period must be > 0".into()));
        }
        let first_valids = Self::first_valids_time_major(high_tm, low_tm, close_tm, cols, rows)?;
        if rows < period {
            return Err(CudaAtrError::InvalidInput(
                "not enough rows for period".into(),
            ));
        }

        let mut d_high =
            DeviceBuffer::from_slice(high_tm).map_err(|e| CudaAtrError::Cuda(e.to_string()))?;
        let mut d_low =
            DeviceBuffer::from_slice(low_tm).map_err(|e| CudaAtrError::Cuda(e.to_string()))?;
        let mut d_close =
            DeviceBuffer::from_slice(close_tm).map_err(|e| CudaAtrError::Cuda(e.to_string()))?;
        let d_fv = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaAtrError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaAtrError::Cuda(e.to_string()))?;

        let func = self
            .module
            .get_function("atr_many_series_one_param_f32")
            .map_err(|e| CudaAtrError::Cuda(e.to_string()))?;

        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            ManySeriesKernelPolicy::Auto => 256,
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut high_ptr = d_high.as_device_ptr().as_raw();
            let mut low_ptr = d_low.as_device_ptr().as_raw();
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut fv_ptr = d_fv.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut alpha = 1.0f32 / (period as f32);
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut alpha as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaAtrError::Cuda(e.to_string()))?;
        }

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }
}

// ---------------- Bench profiles ----------------
// Exclude from test builds to avoid compiling heavy bench prep when running unit tests.
#[cfg(not(test))]
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series(n_combos: usize) -> usize {
        let in_bytes = 3 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = n_combos * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    fn synth_hlc_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i];
            if v.is_nan() {
                continue;
            }
            let x = i as f32 * 0.002f32;
            let off = (0.004 * x.sin()).abs() + 0.12;
            high[i] = v + off;
            low[i] = v - off;
        }
        (high, low)
    }

    struct AtrBatchState {
        cuda: CudaAtr,
        high: Vec<f32>,
        low: Vec<f32>,
        close: Vec<f32>,
        sweep: AtrBatchRange,
    }
    impl CudaBenchState for AtrBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .atr_batch_dev(&self.high, &self.low, &self.close, &self.sweep)
                .unwrap();
        }
    }

    struct AtrManyState {
        cuda: CudaAtr,
        high_tm: Vec<f32>,
        low_tm: Vec<f32>,
        close_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        period: usize,
    }
    impl CudaBenchState for AtrManyState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .atr_many_series_one_param_time_major_dev(
                    &self.high_tm,
                    &self.low_tm,
                    &self.close_tm,
                    self.cols,
                    self.rows,
                    self.period,
                )
                .unwrap();
        }
    }

    struct BatchPrepCfg;
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let len = ONE_SERIES_LEN;
        let pstart = 5usize;
        let pend = 64usize;
        let pstep = 5usize;
        let close = gen_series(len);
        let (high, low) = synth_hlc_from_close(&close);
        Box::new(AtrBatchState {
            cuda: CudaAtr::new(0).unwrap(),
            high,
            low,
            close,
            sweep: AtrBatchRange {
                length: (pstart, pend, pstep),
            },
        })
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let (cols, rows, period) = (256usize, 262_144usize, 14usize);
        let mut close_tm = vec![f32::NAN; cols * rows];
        for s in 0..cols {
            for t in s..rows {
                let x = (t as f32) + (s as f32) * 0.2;
                close_tm[t * cols + s] = (x * 0.0017).sin() + 0.00015 * x;
            }
        }
        let (mut high_tm, mut low_tm) = (close_tm.clone(), close_tm.clone());
        for s in 0..cols {
            for t in 0..rows {
                let v = close_tm[t * cols + s];
                if v.is_nan() {
                    continue;
                }
                let x = (t as f32) * 0.002;
                let off = (0.004 * x.cos()).abs() + 0.11;
                high_tm[t * cols + s] = v + off;
                low_tm[t * cols + s] = v - off;
            }
        }
        Box::new(AtrManyState {
            cuda: CudaAtr::new(0).unwrap(),
            high_tm,
            low_tm,
            close_tm,
            cols,
            rows,
            period,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        let pstart = 5usize;
        let pend = 64usize;
        let pstep = 5usize;
        let n_combos = ((pend - pstart) / pstep + 1).max(1);
        let scen_batch = CudaBenchScenario::new(
            "atr",
            "one_series_many_params",
            "atr_cuda_batch_dev",
            "1m_x_params",
            prep_one_series_many_params,
        )
        .with_mem_required(bytes_one_series(n_combos));

        let (cols, rows) = (256usize, 262_144usize);
        let scen_many = CudaBenchScenario::new(
            "atr",
            "many_series_one_param",
            "atr_cuda_many_series_one_param_dev",
            "256x262k",
            prep_many_series_one_param,
        )
        .with_mem_required(
            (3 * cols * rows + cols * rows) * std::mem::size_of::<f32>() + 64 * 1024 * 1024,
        );

        vec![scen_batch, scen_many]
    }
}
