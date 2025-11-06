//! CUDA wrapper for Chaikin's Volatility (CVI).
//!
//! Parity targets with ALMA-style wrappers:
//! - Device-returning API (`DeviceArrayF32`).
//! - Batch (one-series × many-params) and Many-series × one-param (time-major).
//! - Warmup/NaN identical to scalar (first_valid + 2*period - 1).
//! - VRAM check + combo chunking; auto-select shared-precompute kernel when available.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::cvi::CviBatchRange;
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
pub enum CudaCviError {
    Cuda(String),
    InvalidInput(String),
}
impl fmt::Display for CudaCviError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaCviError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaCviError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl std::error::Error for CudaCviError {}

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
pub struct CudaCviPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaCviPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

pub struct CudaCvi {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaCviPolicy,
}

impl CudaCvi {
    pub fn new(device_id: usize) -> Result<Self, CudaCviError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaCviError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaCviError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaCviError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/cvi_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O4),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
                {
                    m
                } else {
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaCviError::Cuda(e.to_string()))?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaCviError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaCviPolicy::default(),
        })
    }

    pub fn set_policy(&mut self, policy: CudaCviPolicy) {
        self.policy = policy;
    }
    pub fn synchronize(&self) -> Result<(), CudaCviError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaCviError::Cuda(e.to_string()))
    }

    fn first_valid_hl(high: &[f32], low: &[f32]) -> Result<usize, CudaCviError> {
        if high.is_empty() || low.is_empty() {
            return Err(CudaCviError::InvalidInput("empty input".into()));
        }
        let n = high.len().min(low.len());
        for i in 0..n {
            if !high[i].is_nan() && !low[i].is_nan() {
                return Ok(i);
            }
        }
        Err(CudaCviError::InvalidInput("all values are NaN".into()))
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

    #[inline(always)]
    fn grid_1d_for_elems(elems: usize, block_x: u32) -> GridSize {
        let gx = ((elems as u32) + block_x - 1) / block_x;
        (gx.max(1), 1, 1).into()
    }

    #[inline(always)]
    fn bytes_required_batch(n_combos: usize, len: usize) -> usize {
        let f32b = std::mem::size_of::<f32>();
        let i32b = std::mem::size_of::<i32>();
        let in_bytes = 2 * len * f32b; // high + low
        let params_bytes = n_combos * (2 * i32b + f32b); // period, warm, alpha
        let out_bytes = n_combos * len * f32b; // output
        in_bytes + params_bytes + out_bytes
    }

    fn chunk_size_for_batch(n_combos: usize, len: usize) -> usize {
        // Inputs: 2×len f32; params per combo (period i32, alpha f32, warm i32); outputs: combos×len f32.
        let in_bytes = 2 * len * std::mem::size_of::<f32>();
        let params_bytes = n_combos * (std::mem::size_of::<i32>() * 2 + std::mem::size_of::<f32>());
        let out_per_combo = len * std::mem::size_of::<f32>();
        let headroom = 64 * 1024 * 1024;
        let mut chunk = n_combos.max(1);
        while chunk > 1 {
            let need = in_bytes + params_bytes + chunk * out_per_combo + headroom;
            if Self::device_will_fit(need, 0) {
                break;
            }
            chunk = (chunk + 1) / 2;
        }
        chunk.max(1)
    }

    pub fn cvi_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
        sweep: &CviBatchRange,
    ) -> Result<DeviceArrayF32, CudaCviError> {
        if high.len() != low.len() {
            return Err(CudaCviError::InvalidInput("input length mismatch".into()));
        }
        let len = high.len();
        if len == 0 {
            return Err(CudaCviError::InvalidInput("empty input".into()));
        }
        let first_valid = Self::first_valid_hl(high, low)?;

        // Expand period combos
        let (ps, pe, pst) = sweep.period;
        if ps == 0 {
            return Err(CudaCviError::InvalidInput("period must be > 0".into()));
        }
        let periods: Vec<usize> = if pst == 0 || ps == pe {
            vec![ps]
        } else {
            (ps..=pe).step_by(pst).collect()
        };
        if periods.is_empty() {
            return Err(CudaCviError::InvalidInput("no parameter combos".into()));
        }
        for &p in &periods {
            if p == 0 || p > len || (len - first_valid) < (2 * p - 1) {
                return Err(CudaCviError::InvalidInput(format!(
                    "invalid period {} for data length {} (valid after {}: {}), need >= {}",
                    p,
                    len,
                    first_valid,
                    len - first_valid,
                    2 * p - 1
                )));
            }
        }
        let n_combos = periods.len();

        // Fail fast if the full output + inputs cannot fit (chunking won't help d_out size)
        let headroom = 64 * 1024 * 1024; // 64 MiB safety margin
        let need = Self::bytes_required_batch(n_combos, len);
        if !Self::device_will_fit(need, headroom) {
            return Err(CudaCviError::Cuda(format!(
                "insufficient device memory: need ~{:.2} MiB (+{:.0} MiB headroom)",
                (need as f64) / (1024.0 * 1024.0),
                (headroom as f64) / (1024.0 * 1024.0)
            )));
        }

        // Host params
        let h_periods: Vec<i32> = periods.iter().map(|&p| p as i32).collect();
        let h_alphas: Vec<f32> = periods
            .iter()
            .map(|&p| (2.0f32 / (p as f32 + 1.0f32)))
            .collect();
        let h_warms: Vec<i32> = periods
            .iter()
            .map(|&p| (first_valid + (2 * p - 1)) as i32)
            .collect();

        // Device buffers
        let mut d_high_opt = Some(
            DeviceBuffer::from_slice(high).map_err(|e| CudaCviError::Cuda(e.to_string()))?,
        );
        let mut d_low_opt = Some(
            DeviceBuffer::from_slice(low).map_err(|e| CudaCviError::Cuda(e.to_string()))?,
        );
        let d_periods =
            DeviceBuffer::from_slice(&h_periods).map_err(|e| CudaCviError::Cuda(e.to_string()))?;
        let d_alphas =
            DeviceBuffer::from_slice(&h_alphas).map_err(|e| CudaCviError::Cuda(e.to_string()))?;
        let d_warms =
            DeviceBuffer::from_slice(&h_warms).map_err(|e| CudaCviError::Cuda(e.to_string()))?;

        // Can we use the range-based kernel and build range on device?
        let has_cvi_from_range = self
            .module
            .get_function("cvi_batch_from_range_f32")
            .is_ok();
        let has_range_kernel = self
            .module
            .get_function("range_from_high_low_f32")
            .is_ok();
        let mut d_range_opt: Option<DeviceBuffer<f32>> = None;
        if has_cvi_from_range {
            if has_range_kernel {
                // Build range on device: range[t] = high[t] - low[t]
                let mut d_range: DeviceBuffer<f32> =
                    unsafe { DeviceBuffer::uninitialized(len) }
                        .map_err(|e| CudaCviError::Cuda(e.to_string()))?;
                let range_func = self
                    .module
                    .get_function("range_from_high_low_f32")
                    .unwrap();
                unsafe {
                    let mut len_i = len as i32;
                    let mut high_ptr = d_high_opt
                        .as_ref()
                        .unwrap()
                        .as_device_ptr()
                        .as_raw();
                    let mut low_ptr = d_low_opt
                        .as_ref()
                        .unwrap()
                        .as_device_ptr()
                        .as_raw();
                    let mut out_ptr = d_range.as_device_ptr().as_raw();
                    let block_x_range = 256u32;
                    let grid = Self::grid_1d_for_elems(len, block_x_range);
                    let block: BlockSize = (block_x_range, 1, 1).into();
                    let args: &mut [*mut c_void] = &mut [
                        &mut high_ptr as *mut _ as *mut c_void,
                        &mut low_ptr as *mut _ as *mut c_void,
                        &mut len_i as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];
                    self.stream
                        .launch(&range_func, grid, block, 0, args)
                        .map_err(|e| CudaCviError::Cuda(e.to_string()))?;
                }
                // Free high/low early once range exists
                d_high_opt = None;
                d_low_opt = None;
                d_range_opt = Some(d_range);
            } else {
                // Fallback: host precompute range and copy
                let mut r = vec![0f32; len];
                for i in 0..len {
                    r[i] = high[i] - low[i];
                }
                let dev = DeviceBuffer::from_slice(&r)
                    .map_err(|e| CudaCviError::Cuda(e.to_string()))?;
                d_range_opt = Some(dev);
                // Inputs no longer needed once range is available
                d_high_opt = None;
                d_low_opt = None;
            }
        }

        // Output buffer (full size; we already checked it fits)
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * len) }
                .map_err(|e| CudaCviError::Cuda(e.to_string()))?;

        // Choose kernel once
        let func = if has_cvi_from_range {
            self.module
                .get_function("cvi_batch_from_range_f32")
                .unwrap()
        } else {
            self.module
                .get_function("cvi_batch_f32")
                .map_err(|e| CudaCviError::Cuda(e.to_string()))?
        };

        // Keep chunking purely as launch heuristic (not memory workaround)
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            BatchKernelPolicy::Auto => 256,
        };
        let chunk = {
            let max_blocks: usize = 16_384; // cap grids to reasonable size
            (n_combos).min(max_blocks * (block_x as usize))
        };

        let mut launched = 0usize;
        while launched < n_combos {
            let cur = (n_combos - launched).min(chunk);
            let grid: GridSize = Self::grid_1d_for_elems(cur, block_x);
            let block: BlockSize = (block_x, 1, 1).into();
            unsafe {
                let mut len_i = len as i32;
                let mut first_i = first_valid as i32;
                let mut cur_i = cur as i32;
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
                let mut out_ptr = d_out
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * len * std::mem::size_of::<f32>()) as u64);
                if has_cvi_from_range {
                    let dr = d_range_opt.as_ref().expect("range device buffer missing");
                    let mut range_ptr = dr.as_device_ptr().as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut range_ptr as *mut _ as *mut c_void,
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
                        .map_err(|e| CudaCviError::Cuda(e.to_string()))?;
                } else {
                    let mut high_ptr = d_high_opt
                        .as_ref()
                        .expect("device high buffer missing")
                        .as_device_ptr()
                        .as_raw();
                    let mut low_ptr = d_low_opt
                        .as_ref()
                        .expect("device low buffer missing")
                        .as_device_ptr()
                        .as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut high_ptr as *mut _ as *mut c_void,
                        &mut low_ptr as *mut _ as *mut c_void,
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
                        .map_err(|e| CudaCviError::Cuda(e.to_string()))?;
                }
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
        cols: usize,
        rows: usize,
    ) -> Result<Vec<i32>, CudaCviError> {
        let n = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaCviError::InvalidInput("rows*cols overflow".into()))?;
        if high_tm.len() != n || low_tm.len() != n {
            return Err(CudaCviError::InvalidInput(
                "time-major input length mismatch".into(),
            ));
        }
        let mut out = vec![-1i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                let h = high_tm[idx];
                let l = low_tm[idx];
                if !h.is_nan() && !l.is_nan() {
                    out[s] = t as i32;
                    break;
                }
            }
        }
        Ok(out)
    }

    pub fn cvi_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaCviError> {
        if period == 0 {
            return Err(CudaCviError::InvalidInput("period must be > 0".into()));
        }
        let first_valids = Self::first_valids_time_major(high_tm, low_tm, cols, rows)?;
        let warm = first_valids
            .iter()
            .copied()
            .filter(|&fv| fv >= 0)
            .map(|fv| fv as usize + (2 * period - 1))
            .max()
            .unwrap_or(0);
        if warm >= rows {
            return Err(CudaCviError::InvalidInput(
                "not enough rows for period/warmup".into(),
            ));
        }

        let d_high =
            DeviceBuffer::from_slice(high_tm).map_err(|e| CudaCviError::Cuda(e.to_string()))?;
        let d_low =
            DeviceBuffer::from_slice(low_tm).map_err(|e| CudaCviError::Cuda(e.to_string()))?;
        let d_fv = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaCviError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaCviError::Cuda(e.to_string()))?;

        let func = self
            .module
            .get_function("cvi_many_series_one_param_f32")
            .map_err(|e| CudaCviError::Cuda(e.to_string()))?;

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
            let mut fv_ptr = d_fv.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut alpha_f = 2.0f32 / (period as f32 + 1.0f32);
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut alpha_f as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaCviError::Cuda(e.to_string()))?;
        }

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }
}

// ---------------- Bench profiles ----------------
#[cfg(not(test))]
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;

    fn synth_hl_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
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

    struct CviBatchState {
        cuda: CudaCvi,
        high: Vec<f32>,
        low: Vec<f32>,
        sweep: CviBatchRange,
    }
    impl CudaBenchState for CviBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .cvi_batch_dev(&self.high, &self.low, &self.sweep)
                .unwrap();
        }
    }

    struct CviManyState {
        cuda: CudaCvi,
        high_tm: Vec<f32>,
        low_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        period: usize,
    }
    impl CudaBenchState for CviManyState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .cvi_many_series_one_param_time_major_dev(
                    &self.high_tm,
                    &self.low_tm,
                    self.cols,
                    self.rows,
                    self.period,
                )
                .unwrap();
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let len = ONE_SERIES_LEN;
        let close = gen_series(len);
        let (high, low) = synth_hl_from_close(&close);
        let sweep = CviBatchRange { period: (5, 64, 5) };
        Box::new(CviBatchState {
            cuda: CudaCvi::new(0).unwrap(),
            high,
            low,
            sweep,
        })
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let (cols, rows, period) = (256usize, 262_144usize, 14usize);
        let mut base = vec![f32::NAN; cols * rows];
        for s in 0..cols {
            for t in s..rows {
                let x = (t as f32) + (s as f32) * 0.2;
                base[t * cols + s] = (x * 0.0017).sin() + 0.00015 * x;
            }
        }
        let mut high_tm = base.clone();
        let mut low_tm = base.clone();
        for s in 0..cols {
            for t in 0..rows {
                let v = base[t * cols + s];
                if v.is_nan() {
                    continue;
                }
                let x = (t as f32) * 0.002;
                let off = (0.004 * x.cos()).abs() + 0.11;
                high_tm[t * cols + s] = v + off;
                low_tm[t * cols + s] = v - off;
            }
        }
        Box::new(CviManyState {
            cuda: CudaCvi::new(0).unwrap(),
            high_tm,
            low_tm,
            cols,
            rows,
            period,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        let scen_batch = CudaBenchScenario::new(
            "cvi",
            "one_series_many_params",
            "cvi_cuda_batch_dev",
            "1m_x_params",
            prep_one_series_many_params,
        );
        let scen_many = CudaBenchScenario::new(
            "cvi",
            "many_series_one_param",
            "cvi_cuda_many_series_one_param_dev",
            "256x262k",
            prep_many_series_one_param,
        );
        vec![scen_batch, scen_many]
    }
}
