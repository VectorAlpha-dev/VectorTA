//! CUDA wrapper for SuperTrend (ATR-based dynamic bands) kernels.
//!
//! Pattern: Composite/Recurrence. We precompute ATR rows on device using the
//! existing ATR CUDA wrapper and build a shared HL2(midpoint) vector on host.
//! Each row (param combo) or series is scanned sequentially along time on the
//! device mirroring scalar warmup/NaN semantics.

#![cfg(feature = "cuda")]

use crate::cuda::atr_wrapper::CudaAtr;
use crate::cuda::di_wrapper::DeviceArrayF32Pair;
use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::atr::AtrBatchRange;
use crate::indicators::supertrend::{SuperTrendBatchRange, SuperTrendParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CudaSupertrendError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error(
        "launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})"
    )]
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    /// 1-D launch across rows (recommended). Threads cover rows: r = blockIdx.x * blockDim.x + threadIdx.x
    OneD {
        block_x: u32,
    },
    /// Legacy slow-path that launches exactly one thread per row/block (kept for compatibility).
    OneThreadPerRow,
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 }, // threads across series (time-major)
}

#[derive(Clone, Copy, Debug)]
pub struct CudaSupertrendPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaSupertrendPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

pub struct CudaSupertrend {
    module: Module,
    stream: Stream,
    context: std::sync::Arc<Context>,
    device_id: u32,
    policy: CudaSupertrendPolicy,
}

impl CudaSupertrend {
    pub fn new(device_id: usize) -> Result<Self, CudaSupertrendError> {
        cust::init(CudaFlags::empty())?;
        let dev = Device::get_device(device_id as u32)?;
        let ctx = std::sync::Arc::new(Context::new(dev)?);
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/supertrend_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O4), // most optimized JIT level
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(Self {
            module,
            stream,
            context: ctx,
            device_id: device_id as u32,
            policy: CudaSupertrendPolicy::default(),
        })
    }

    pub fn set_policy(&mut self, p: CudaSupertrendPolicy) {
        self.policy = p;
    }
    pub fn synchronize(&self) -> Result<(), CudaSupertrendError> {
        self.stream.synchronize().map_err(Into::into)
    }
    pub fn context_arc(&self) -> std::sync::Arc<Context> { self.context.clone() }
    pub fn device_id(&self) -> u32 { self.device_id }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }
    #[inline]
    fn will_fit(bytes: usize, headroom: usize) -> Result<(), CudaSupertrendError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Ok((free, _)) = mem_get_info() {
            let free_u = free as usize;
            if bytes.saturating_add(headroom) <= free_u {
                Ok(())
            } else {
                Err(CudaSupertrendError::OutOfMemory {
                    required: bytes,
                    free: free_u,
                    headroom,
                })
            }
        } else {
            Ok(())
        }
    }

    #[inline]
    fn pick_block_x(rows_or_cols: usize) -> u32 {
        if rows_or_cols >= (1 << 14) {
            256
        } else {
            128
        }
    }

    fn first_valid_hlc(
        high: &[f32],
        low: &[f32],
        close: &[f32],
    ) -> Result<usize, CudaSupertrendError> {
        if high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaSupertrendError::InvalidInput("empty input".into()));
        }
        let len = high.len().min(low.len()).min(close.len());
        for i in 0..len {
            if !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan() {
                return Ok(i);
            }
        }
        Err(CudaSupertrendError::InvalidInput("all values are NaN".into()))
    }

    // ---- Batch: one series × many params ----
    pub fn supertrend_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &SuperTrendBatchRange,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32, Vec<SuperTrendParams>), CudaSupertrendError> {
        let len = close.len();
        if !(high.len() == low.len() && low.len() == close.len()) {
            return Err(CudaSupertrendError::InvalidInput(
                "input length mismatch".into(),
            ));
        }
        if len == 0 {
            return Err(CudaSupertrendError::InvalidInput("empty series".into()));
        }

        // Expand combos
        let combos = expand_grid_local(sweep)?;
        if combos.is_empty() {
            return Err(CudaSupertrendError::InvalidInput("empty sweep".into()));
        }

        // Period coverage for ATR rows
        let min_p = combos.iter().map(|c| c.period.unwrap()).min().unwrap();
        let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
        if min_p == 0 || max_p > len {
            return Err(CudaSupertrendError::InvalidInput(
                "invalid period limits".into(),
            ));
        }

        // First valid index for warm computations
        let first_valid = Self::first_valid_hlc(high, low, close)?;
        if len - first_valid < min_p {
            return Err(CudaSupertrendError::InvalidInput(
                "not enough valid data".into(),
            ));
        }

        // Precompute HL2 on host (shared across rows)
        let mut hl2 = vec![f32::NAN; len];
        for i in 0..len {
            let h = high[i];
            let l = low[i];
            hl2[i] = 0.5f32 * (h + l);
        }
        let d_hl2 = DeviceBuffer::from_slice(&hl2)?;
        let d_close = DeviceBuffer::from_slice(close)?;

        // ATR rows for continuous [min..max]
        let cuda_atr =
            CudaAtr::new(self.device_id as usize).map_err(|e| CudaSupertrendError::InvalidInput(e.to_string()))?;
        let atr_rows = cuda_atr
            .atr_batch_dev(
                high,
                low,
                close,
                &AtrBatchRange {
                    length: (min_p, max_p, 1),
                },
            )
            .map_err(|e| CudaSupertrendError::InvalidInput(format!("atr: {}", e.to_string())))?;

        // Params per row
        let row_period_idx: Vec<i32> = combos
            .iter()
            .map(|c| (c.period.unwrap() as i32) - (min_p as i32))
            .collect();
        let row_factors: Vec<f32> = combos.iter().map(|c| c.factor.unwrap() as f32).collect();
        let row_warms: Vec<i32> = combos
            .iter()
            .map(|c| (first_valid + c.period.unwrap() - 1) as i32)
            .collect();
        let d_row_idx = DeviceBuffer::from_slice(&row_period_idx)?;
        let d_row_fac = DeviceBuffer::from_slice(&row_factors)?;
        let d_row_warm = DeviceBuffer::from_slice(&row_warms)?;

        // VRAM check: only account for future allocations (outputs). ATR/inputs already on device.
        let combos_len = combos.len();
        let total_elems = combos_len
            .checked_mul(len)
            .ok_or_else(|| CudaSupertrendError::InvalidInput("rows*cols overflow".into()))?;
        let bytes = total_elems
            .checked_mul(2)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaSupertrendError::InvalidInput("byte size overflow".into()))?;
        Self::will_fit(bytes, 64 * 1024 * 1024)?;

        // Outputs
        let mut d_trend: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total_elems) }?;
        let mut d_changed: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(total_elems) }?;

        // Kernel and launch: single 1-D grid (no 65k chunking)
        let func = self
            .module
            .get_function("supertrend_batch_f32")
            .map_err(|_| CudaSupertrendError::MissingKernelSymbol { name: "supertrend_batch_f32" })?;

        match self.policy.batch {
            BatchKernelPolicy::OneThreadPerRow => {
                // Legacy: one thread per block, grid covers rows
                let grid: GridSize = ((combos_len as u32).max(1), 1, 1).into();
                let block: BlockSize = (1, 1, 1).into();
                unsafe {
                    use cust::prelude::launch;
                    let stream = &self.stream;
                    launch!(func<<<grid, block, 0, stream>>>(
                        d_hl2.as_device_ptr(),
                        d_close.as_device_ptr(),
                        atr_rows.buf.as_device_ptr(),
                        d_row_idx.as_device_ptr(),
                        d_row_fac.as_device_ptr(),
                        d_row_warm.as_device_ptr(),
                        len as i32,
                        combos_len as i32,
                        d_trend.as_device_ptr(),
                        d_changed.as_device_ptr()
                    ))
                    ?;
                }
            }
            _ => {
                let bx = if let BatchKernelPolicy::OneD { block_x } = self.policy.batch {
                    block_x
                } else {
                    Self::pick_block_x(combos_len)
                };
                let gx = ((combos_len as u32) + bx - 1) / bx;
                let grid: GridSize = (gx.max(1), 1, 1).into();
                let block: BlockSize = (bx, 1, 1).into();
                unsafe {
                    use cust::prelude::launch;
                    let stream = &self.stream;
                    launch!(func<<<grid, block, 0, stream>>>(
                        d_hl2.as_device_ptr(),
                        d_close.as_device_ptr(),
                        atr_rows.buf.as_device_ptr(),
                        d_row_idx.as_device_ptr(),
                        d_row_fac.as_device_ptr(),
                        d_row_warm.as_device_ptr(),
                        len as i32,
                        combos_len as i32,
                        d_trend.as_device_ptr(),
                        d_changed.as_device_ptr()
                    ))?;
                }
            }
        }

        self.stream.synchronize()?;

        Ok((
            DeviceArrayF32 {
                buf: d_trend,
                rows: combos_len,
                cols: len,
            },
            DeviceArrayF32 {
                buf: d_changed,
                rows: combos_len,
                cols: len,
            },
            combos,
        ))
    }

    // ---- Many-series × one-param (time-major) ----
    pub fn supertrend_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
        factor: f32,
    ) -> Result<DeviceArrayF32Pair, CudaSupertrendError> {
        let n = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaSupertrendError::InvalidInput("rows*cols overflow".into()))?;
        if high_tm.len() != n || low_tm.len() != n || close_tm.len() != n {
            return Err(CudaSupertrendError::InvalidInput(
                "time-major input length mismatch".into(),
            ));
        }
        if period == 0 || rows < period {
            return Err(CudaSupertrendError::InvalidInput("invalid period".into()));
        }

        // HL2 time-major
        let mut hl2_tm = vec![f32::NAN; n];
        for idx in 0..n {
            hl2_tm[idx] = 0.5f32 * (high_tm[idx] + low_tm[idx]);
        }
        let d_hl2 = DeviceBuffer::from_slice(&hl2_tm)?;
        let d_close = DeviceBuffer::from_slice(close_tm)?;

        // ATR time-major for given period
        let cuda_atr =
            CudaAtr::new(self.device_id as usize).map_err(|e| CudaSupertrendError::InvalidInput(e.to_string()))?;
        let atr_tm = cuda_atr
            .atr_many_series_one_param_time_major_dev(high_tm, low_tm, close_tm, cols, rows, period)
            .map_err(|e| CudaSupertrendError::InvalidInput(format!("atr: {}", e.to_string())))?;

        // first-valid per series (time-major)
        let mut first_valids = vec![-1i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                let (h, l, c) = (high_tm[idx], low_tm[idx], close_tm[idx]);
                if !h.is_nan() && !l.is_nan() && !c.is_nan() {
                    first_valids[s] = t as i32;
                    break;
                }
            }
            if first_valids[s] < 0 {
                return Err(CudaSupertrendError::InvalidInput("all-NaN series".into()));
            }
        }
        let d_fv = DeviceBuffer::from_slice(&first_valids)?;

        // Outputs
        let bytes = n
            .checked_mul(2)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaSupertrendError::InvalidInput("byte size overflow".into()))?;
        Self::will_fit(bytes, 64 * 1024 * 1024)?;
        let mut d_trend_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n) }?;
        let mut d_changed_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n) }?;

        // Kernel
        let func = self
            .module
            .get_function("supertrend_many_series_one_param_f32")
            .map_err(|_| CudaSupertrendError::MissingKernelSymbol { name: "supertrend_many_series_one_param_f32" })?;
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            _ => Self::pick_block_x(cols),
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut hl_ptr = d_hl2.as_device_ptr().as_raw();
            let mut cl_ptr = d_close.as_device_ptr().as_raw();
            let mut at_ptr = atr_tm.buf.as_device_ptr().as_raw();
            let mut fv_ptr = d_fv.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut factor_f = factor as f32;
            let mut tr_ptr = d_trend_tm.as_device_ptr().as_raw();
            let mut ch_ptr = d_changed_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut hl_ptr as *mut _ as *mut c_void,
                &mut cl_ptr as *mut _ as *mut c_void,
                &mut at_ptr as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut factor_f as *mut _ as *mut c_void,
                &mut tr_ptr as *mut _ as *mut c_void,
                &mut ch_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)?;
        }

        self.stream.synchronize()?;
        Ok(DeviceArrayF32Pair {
            plus: DeviceArrayF32 {
                buf: d_trend_tm,
                rows,
                cols,
            },
            minus: DeviceArrayF32 {
                buf: d_changed_tm,
                rows,
                cols,
            },
        })
    }
}

// Local expand_grid clone
fn expand_grid_local(r: &SuperTrendBatchRange) -> Result<Vec<SuperTrendParams>, CudaSupertrendError> {
    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, CudaSupertrendError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let v: Vec<usize> = (start..=end).step_by(step.max(1)).collect();
            if v.is_empty() {
                return Err(CudaSupertrendError::InvalidInput(format!(
                    "invalid range: start={}, end={}, step={}",
                    start, end, step
                )));
            }
            return Ok(v);
        }
        let mut v = Vec::new();
        let mut cur = start as isize;
        let end_i = end as isize;
        let st = (step as isize).max(1);
        while cur >= end_i {
            v.push(cur as usize);
            cur -= st;
        }
        if v.is_empty() {
            return Err(CudaSupertrendError::InvalidInput(format!(
                "invalid range: start={}, end={}, step={}",
                start, end, step
            )));
        }
        Ok(v)
    }
    fn axis_f64(
        (start, end, step): (f64, f64, f64),
    ) -> Result<Vec<f64>, CudaSupertrendError> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }
        let st = step.abs();
        if start < end {
            let mut v = Vec::new();
            let mut x = start;
            while x <= end + 1e-12 {
                v.push(x);
                x += st;
            }
            if v.is_empty() {
                return Err(CudaSupertrendError::InvalidInput(format!(
                    "invalid range: start={}, end={}, step={}",
                    start, end, step
                )));
            }
            return Ok(v);
        }
        let mut v = Vec::new();
        let mut x = start;
        while x + 1e-12 >= end {
            v.push(x);
            x -= st;
        }
        if v.is_empty() {
            return Err(CudaSupertrendError::InvalidInput(format!(
                "invalid range: start={}, end={}, step={}",
                start, end, step
            )));
        }
        Ok(v)
    }
    let periods = axis_usize(r.period)?;
    let factors = axis_f64(r.factor)?;
    let cap = periods
        .len()
        .checked_mul(factors.len())
        .ok_or_else(|| CudaSupertrendError::InvalidInput("range size overflow".into()))?;
    let mut out = Vec::with_capacity(cap);
    for &p in &periods {
        for &f in &factors {
            out.push(SuperTrendParams {
                period: Some(p),
                factor: Some(f),
            });
        }
    }
    Ok(out)
}

// ---------------- Bench profiles ----------------
#[cfg(not(test))]
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;

    struct StBatchState {
        cuda: CudaSupertrend,
        high: Vec<f32>,
        low: Vec<f32>,
        close: Vec<f32>,
        sweep: SuperTrendBatchRange,
    }
    impl CudaBenchState for StBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .supertrend_batch_dev(&self.high, &self.low, &self.close, &self.sweep)
                .unwrap();
        }
    }

    struct StManyState {
        cuda: CudaSupertrend,
        high_tm: Vec<f32>,
        low_tm: Vec<f32>,
        close_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        period: usize,
        factor: f32,
    }
    impl CudaBenchState for StManyState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .supertrend_many_series_one_param_time_major_dev(
                    &self.high_tm,
                    &self.low_tm,
                    &self.close_tm,
                    self.cols,
                    self.rows,
                    self.period,
                    self.factor,
                )
                .unwrap();
        }
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

    fn prep_batch() -> Box<dyn CudaBenchState> {
        let len = ONE_SERIES_LEN;
        let close = gen_series(len);
        let (high, low) = synth_hlc_from_close(&close);
        let sweep = SuperTrendBatchRange {
            period: (10, 64, 2),
            factor: (2.0, 4.0, 0.5),
        };
        Box::new(StBatchState {
            cuda: CudaSupertrend::new(0).unwrap(),
            high,
            low,
            close,
            sweep,
        })
    }

    fn prep_many() -> Box<dyn CudaBenchState> {
        let (cols, rows, period, factor) = (256usize, 262_144usize, 14usize, 3.0f32);
        let close_tm = gen_time_major_prices(cols, rows);
        let mut high_tm = close_tm.clone();
        let mut low_tm = close_tm.clone();
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
        Box::new(StManyState {
            cuda: CudaSupertrend::new(0).unwrap(),
            high_tm,
            low_tm,
            close_tm,
            cols,
            rows,
            period,
            factor,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        let scen_batch = CudaBenchScenario::new(
            "supertrend",
            "one_series_many_params",
            "supertrend_cuda_batch_dev",
            "1m_len",
            prep_batch,
        )
        .with_mem_required((2 * ONE_SERIES_LEN) * std::mem::size_of::<f32>() + 64 * 1024 * 1024);

        let (cols, rows) = (256usize, 262_144usize);
        let scen_many = CudaBenchScenario::new(
            "supertrend",
            "many_series_one_param",
            "supertrend_cuda_many_series_one_param_dev",
            "256x262k",
            prep_many,
        )
        .with_mem_required(
            (3 * cols * rows + 2 * cols * rows) * std::mem::size_of::<f32>() + 64 * 1024 * 1024,
        );

        vec![scen_batch, scen_many]
    }
}
