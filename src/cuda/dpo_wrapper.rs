//! CUDA support for Detrended Price Oscillator (DPO).
//!
//! Math pattern classification: prefix-sum average with lagged subtraction.
//! - Batch (one series × many params): host builds prefix sums P (sum_y)
//!   over the valid segment [first_valid..), and the kernel computes O(1) window outputs.
//! - Many-series × one-param (time-major): host builds time-major P per series with respect to
//!   the per-series first_valid; the kernel mirrors the same window logic.
//!
//! Semantics match the scalar DPO implementation:
//! - Warmup per row/series: warm = max(first_valid + period - 1, back) where back = period/2 + 1
//! - Warmup prefix is filled with NaN
//! - Critical accumulations use f64; outputs are f32

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::dpo::{DpoBatchRange, DpoParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaDpoError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaDpoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaDpoError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaDpoError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaDpoError {}

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
pub struct CudaDpoPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaDpoPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

pub struct CudaDpo {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaDpoPolicy,
    last_batch_block: Option<u32>,
    last_many_block: Option<u32>,
}

impl CudaDpo {
    pub fn new(device_id: usize) -> Result<Self, CudaDpoError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaDpoError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaDpoError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaDpoError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/dpo_kernel.ptx"));
        let module = Module::from_ptx(
            ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O2),
            ],
        )
        .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
        .or_else(|_| Module::from_ptx(ptx, &[]))
        .map_err(|e| CudaDpoError::Cuda(e.to_string()))?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaDpoError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaDpoPolicy::default(),
            last_batch_block: None,
            last_many_block: None,
        })
    }

    pub fn set_policy(&mut self, policy: CudaDpoPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaDpoPolicy {
        &self.policy
    }

    // ---------- One-series × many-params ----------

    pub fn dpo_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &DpoBatchRange,
    ) -> Result<DeviceArrayF32, CudaDpoError> {
        let (periods, first_valid) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let len = data_f32.len();
        let n_combos = periods.len();

        // Build f64 prefix over [first_valid..)
        let ps = build_prefixes_from_first(data_f32, first_valid);

        // VRAM estimate
        let bytes = len * 4 + (len + 1) * 8 + n_combos * 4 + len * n_combos * 4 + 64 * 1024 * 1024;
        if let Ok((free, _)) = mem_get_info() {
            if bytes > free {
                return Err(CudaDpoError::InvalidInput(format!(
                    "estimated device memory {:.2} MB exceeds free VRAM",
                    (bytes as f64) / (1024.0 * 1024.0)
                )));
            }
        }

        let d_data =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaDpoError::Cuda(e.to_string()))?;
        let d_ps = DeviceBuffer::from_slice(&ps).map_err(|e| CudaDpoError::Cuda(e.to_string()))?;
        let d_periods =
            DeviceBuffer::from_slice(&periods).map_err(|e| CudaDpoError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(len * n_combos)
                .map_err(|e| CudaDpoError::Cuda(e.to_string()))?
        };

        self.launch_batch_kernel(
            &d_data,
            &d_ps,
            len as i32,
            first_valid as i32,
            &d_periods,
            n_combos as i32,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: len,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel(
        &self,
        d_data: &DeviceBuffer<f32>,
        d_ps: &DeviceBuffer<f64>,
        len: i32,
        first_valid: i32,
        d_periods: &DeviceBuffer<i32>,
        n_combos: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaDpoError> {
        if len <= 0 || n_combos <= 0 {
            return Ok(());
        }
        let func = self
            .module
            .get_function("dpo_batch_f32")
            .map_err(|e| CudaDpoError::Cuda(e.to_string()))?;

        let block_x = match self.policy.batch {
            BatchKernelPolicy::OneD { block_x } if block_x > 0 => block_x,
            _ => 256,
        };
        let grid_x = ((len as u32) + block_x - 1) / block_x;
        // Launch in grid.y chunks to respect 65_535 limit
        for (start, count) in grid_y_chunks(n_combos as usize) {
            let grid: GridSize = (grid_x.max(1), count as u32, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            unsafe {
                let mut p_data = d_data.as_device_ptr().as_raw();
                let mut p_ps = d_ps.as_device_ptr().as_raw();
                let mut p_len = len;
                let mut p_first = first_valid;
                let mut p_periods = d_periods.as_device_ptr().add(start).as_raw();
                let mut p_n = count as i32;
                let mut p_out = d_out.as_device_ptr().add(start * (len as usize)).as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut p_data as *mut _ as *mut c_void,
                    &mut p_ps as *mut _ as *mut c_void,
                    &mut p_len as *mut _ as *mut c_void,
                    &mut p_first as *mut _ as *mut c_void,
                    &mut p_periods as *mut _ as *mut c_void,
                    &mut p_n as *mut _ as *mut c_void,
                    &mut p_out as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaDpoError::Cuda(e.to_string()))?;
            }
        }
        Ok(())
    }

    // ---------- Many-series × one-param (time-major) ----------

    pub fn dpo_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &DpoParams,
    ) -> Result<DeviceArrayF32, CudaDpoError> {
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        // Build time-major prefix P per series
        let ps_tm = build_prefixes_time_major(data_tm_f32, cols, rows, &first_valids);

        // VRAM estimate
        let elems = cols * rows;
        let bytes = elems * 4 + (elems + 1) * 8 + cols * 4 + rows * cols * 4 + 64 * 1024 * 1024;
        if let Ok((free, _)) = mem_get_info() {
            if bytes > free {
                return Err(CudaDpoError::InvalidInput(format!(
                    "estimated device memory {:.2} MB exceeds free VRAM",
                    (bytes as f64) / (1024.0 * 1024.0)
                )));
            }
        }

        let d_data =
            DeviceBuffer::from_slice(data_tm_f32).map_err(|e| CudaDpoError::Cuda(e.to_string()))?;
        let d_ps =
            DeviceBuffer::from_slice(&ps_tm).map_err(|e| CudaDpoError::Cuda(e.to_string()))?;
        let d_fv = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaDpoError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(elems).map_err(|e| CudaDpoError::Cuda(e.to_string()))?
        };

        self.launch_many_series_kernel(
            &d_data,
            &d_ps,
            &d_fv,
            cols as i32,
            rows as i32,
            period as i32,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_kernel(
        &self,
        d_data: &DeviceBuffer<f32>,
        d_ps: &DeviceBuffer<f64>,
        d_fv: &DeviceBuffer<i32>,
        cols: i32,
        rows: i32,
        period: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaDpoError> {
        if cols <= 0 || rows <= 0 {
            return Ok(());
        }
        let func = self
            .module
            .get_function("dpo_many_series_one_param_time_major_f32")
            .map_err(|e| CudaDpoError::Cuda(e.to_string()))?;

        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } if block_x > 0 => block_x,
            _ => 256,
        };
        // Use 1D over time and y-dim over series for modest sizes
        let grid_x = ((rows as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), cols as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut p_data = d_data.as_device_ptr().as_raw();
            let mut p_ps = d_ps.as_device_ptr().as_raw();
            let mut p_fv = d_fv.as_device_ptr().as_raw();
            let mut p_cols = cols;
            let mut p_rows = rows;
            let mut p_period = period;
            let mut p_out = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_data as *mut _ as *mut c_void,
                &mut p_ps as *mut _ as *mut c_void,
                &mut p_fv as *mut _ as *mut c_void,
                &mut p_cols as *mut _ as *mut c_void,
                &mut p_rows as *mut _ as *mut c_void,
                &mut p_period as *mut _ as *mut c_void,
                &mut p_out as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaDpoError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    // ----- Prep helpers -----

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &DpoBatchRange,
    ) -> Result<(Vec<i32>, usize), CudaDpoError> {
        if data_f32.is_empty() {
            return Err(CudaDpoError::InvalidInput("empty data".into()));
        }
        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaDpoError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaDpoError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let mut periods = Vec::with_capacity(combos.len());
        for c in combos {
            let p = c.period.unwrap_or(0);
            if p == 0 || p > len {
                return Err(CudaDpoError::InvalidInput(format!(
                    "invalid period {} for data length {}",
                    p, len
                )));
            }
            if len - first_valid < p {
                return Err(CudaDpoError::InvalidInput(format!(
                    "not enough valid data: needed {}, valid {}",
                    p,
                    len - first_valid
                )));
            }
            periods.push(p as i32);
        }
        Ok((periods, first_valid))
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &DpoParams,
    ) -> Result<(Vec<i32>, usize), CudaDpoError> {
        if cols == 0 || rows == 0 {
            return Err(CudaDpoError::InvalidInput("empty matrix".into()));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaDpoError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }
        let period = params.period.unwrap_or(5);
        if period == 0 || period > rows {
            return Err(CudaDpoError::InvalidInput(format!(
                "invalid period {} for series length {}",
                period, rows
            )));
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
            }
            let fv = fv.ok_or_else(|| {
                CudaDpoError::InvalidInput(format!("series {} consists entirely of NaNs", s))
            })?;
            if rows - fv < period {
                return Err(CudaDpoError::InvalidInput(format!(
                    "series {} lacks data: needed {}, valid {}",
                    s,
                    period,
                    rows - fv
                )));
            }
            first_valids[s] = fv as i32;
        }
        Ok((first_valids, period))
    }
}

// ---- Prefix builders ----

fn build_prefixes_from_first(data: &[f32], first_valid: usize) -> Vec<f64> {
    let len = data.len();
    // Store length+1 array aligned to absolute indices: prefix[i] holds sum over [first_valid..i-1]
    // For i <= first_valid, value remains 0.
    let mut ps = vec![0.0f64; len + 1];
    let mut acc_s = 0.0f64;
    for i in 0..len {
        if i >= first_valid {
            let v = data[i] as f64;
            acc_s += v;
        }
        let w = i + 1;
        ps[w] = acc_s;
    }
    ps
}

fn build_prefixes_time_major(
    data_tm: &[f32],
    cols: usize,
    rows: usize,
    first_valids: &[i32],
) -> Vec<f64> {
    let total = data_tm.len();
    let mut ps = vec![0.0f64; total + 1];
    for s in 0..cols {
        let fv = first_valids[s].max(0) as usize;
        let mut acc_s = 0.0f64;
        for t in 0..rows {
            if t >= fv {
                let v = data_tm[t * cols + s] as f64;
                acc_s += v;
            }
            let w = (t * cols + s) + 1;
            ps[w] = acc_s;
        }
    }
    ps
}

// ---- Grid expand (mirror indicator) ----

fn expand_grid(r: &DpoBatchRange) -> Vec<DpoParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(DpoParams { period: Some(p) });
    }
    out
}

#[inline(always)]
fn grid_y_chunks(n: usize) -> impl Iterator<Item = (usize, usize)> {
    struct YChunks {
        n: usize,
        launched: usize,
    }
    impl Iterator for YChunks {
        type Item = (usize, usize);
        fn next(&mut self) -> Option<Self::Item> {
            const MAX: usize = 65_535;
            if self.launched >= self.n {
                return None;
            }
            let start = self.launched;
            let len = (self.n - self.launched).min(MAX);
            self.launched += len;
            Some((start, len))
        }
    }
    YChunks { n, launched: 0 }
}

// ---------- Benches ----------

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250; // vary periods only
    const MANY_SERIES_COLS: usize = 250;
    const MANY_SERIES_ROWS: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        // prefixes dominate but amortized
        let prefix_bytes = (ONE_SERIES_LEN + 1) * std::mem::size_of::<f64>();
        in_bytes + out_bytes + prefix_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_ROWS;
        let in_bytes = elems * std::mem::size_of::<f32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        let prefix_bytes = (elems + 1) * std::mem::size_of::<f64>();
        let fv_bytes = MANY_SERIES_COLS * std::mem::size_of::<i32>();
        in_bytes + out_bytes + prefix_bytes + fv_bytes + 64 * 1024 * 1024
    }

    struct DpoBatchState {
        cuda: CudaDpo,
        price: Vec<f32>,
        sweep: DpoBatchRange,
    }
    impl CudaBenchState for DpoBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .dpo_batch_dev(&self.price, &self.sweep)
                .expect("dpo batch");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaDpo::new(0).expect("cuda dpo");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = DpoBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
        };
        Box::new(DpoBatchState { cuda, price, sweep })
    }

    struct DpoManyState {
        cuda: CudaDpo,
        data_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        params: DpoParams,
    }
    impl CudaBenchState for DpoManyState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .dpo_many_series_one_param_time_major_dev(
                    &self.data_tm,
                    self.cols,
                    self.rows,
                    &self.params,
                )
                .expect("dpo many-series");
        }
    }
    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaDpo::new(0).expect("cuda dpo");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_ROWS;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = DpoParams { period: Some(20) };
        Box::new(DpoManyState {
            cuda,
            data_tm,
            cols,
            rows,
            params,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "dpo",
                "one_series_many_params",
                "dpo_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "dpo",
                "many_series_one_param",
                "dpo_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
