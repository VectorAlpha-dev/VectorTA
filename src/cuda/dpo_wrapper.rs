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
//! - Critical accumulations use dual‑fp32 (float2: hi, lo); outputs are f32

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::dpo::{DpoBatchRange, DpoParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, DeviceCopy, LockedBuffer};
use cust::module::{Module, ModuleJitOption};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;

#[derive(thiserror::Error, Debug)]
pub enum CudaDpoError {
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
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

// ---------------------- Host mirror for CUDA float2 ----------------------
// 8-byte aligned to match device-side float2 alignment.
#[repr(C, align(8))]
#[derive(Clone, Copy, Default)]
pub struct Float2 {
    pub x: f32, // hi
    pub y: f32, // lo
}
unsafe impl DeviceCopy for Float2 {}

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
        Self { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Auto }
    }
}

pub struct CudaDpo {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaDpoPolicy,
    last_batch_block: Option<u32>,
    last_many_block: Option<u32>,
}

impl CudaDpo {
    pub fn new(device_id: usize) -> Result<Self, CudaDpoError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/dpo_kernel.ptx"));
        // Determine target from current context; keep defaults otherwise
        let module = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy: CudaDpoPolicy::default(),
            last_batch_block: None,
            last_many_block: None,
        })
    }

    #[inline]
    pub fn context_arc_clone(&self) -> Arc<Context> { self.context.clone() }
    #[inline]
    pub fn device_id(&self) -> u32 { self.device_id }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Option<(usize, usize)> {
        if let Ok((free, total)) = mem_get_info() {
            let free = free.saturating_sub(headroom_bytes);
            return Some((free, total));
        }
        None
    }

    // Upload helper: page-locked (pinned) + async copy for large slices
    fn upload_slice<T: DeviceCopy + Clone>(&self, h: &[T]) -> Result<DeviceBuffer<T>, CudaDpoError> {
        use std::mem::size_of;
        const PIN_THRESHOLD_BYTES: usize = 1 << 20; // 1 MiB
        let bytes = h
            .len()
            .checked_mul(size_of::<T>())
            .ok_or_else(|| CudaDpoError::InvalidInput("size overflow computing upload bytes".into()))?;
        if bytes >= PIN_THRESHOLD_BYTES {
            let h_locked = LockedBuffer::from_slice(h).map_err(CudaDpoError::Cuda)?;
            unsafe {
                let mut d = DeviceBuffer::uninitialized_async(h.len(), &self.stream)
                    .map_err(CudaDpoError::Cuda)?;
                d.async_copy_from(&h_locked, &self.stream)
                    .map_err(CudaDpoError::Cuda)?;
                Ok(d)
            }
        } else {
            unsafe {
                DeviceBuffer::from_slice_async(h, &self.stream).map_err(CudaDpoError::Cuda)
            }
        }
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

        // Build dual-fp32 (float2) prefix over [first_valid..)
        let ps = build_prefixes_from_first(data_f32, first_valid);

        // VRAM estimate with checked arithmetic and headroom
        let headroom = 64 * 1024 * 1024; // 64 MiB safety margin
        let bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|b| b.checked_add((len + 1).checked_mul(std::mem::size_of::<Float2>())?))
            .and_then(|b| b.checked_add(n_combos.checked_mul(std::mem::size_of::<i32>())?))
            .and_then(|b| b.checked_add(len.checked_mul(n_combos)?.checked_mul(std::mem::size_of::<f32>())?))
            .and_then(|b| b.checked_add(headroom))
            .ok_or_else(|| CudaDpoError::InvalidInput("size overflow computing allocation".into()))?;
        if let Some((free, _total)) = Self::will_fit(bytes, headroom) {
            if bytes > free {
                return Err(CudaDpoError::OutOfMemory { required: bytes, free, headroom });
            }
        }

        let d_data = self.upload_slice(data_f32)?;
        let d_ps = self.upload_slice(&ps)?; // Float2
        let d_periods = self.upload_slice(&periods)?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(len * n_combos, &self.stream)
                .map_err(CudaDpoError::Cuda)?
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
        // Ensure completion so Python CAI can omit a stream key
        self.stream.synchronize()?;

        Ok(DeviceArrayF32 { buf: d_out, rows: n_combos, cols: len })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel(
        &self,
        d_data: &DeviceBuffer<f32>,
        d_ps: &DeviceBuffer<Float2>,
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
            .map_err(|_| CudaDpoError::MissingKernelSymbol { name: "dpo_batch_f32" })?;

        let block_x = match self.policy.batch {
            BatchKernelPolicy::OneD { block_x } if block_x > 0 => block_x,
            _ => 256,
        };
        let grid_x = ((len as u32) + block_x - 1) / block_x;
        // Launch in grid.y chunks to respect 65_535 limit
        for (start, count) in grid_y_chunks(n_combos as usize) {
            let grid: GridSize = (grid_x.max(1), count as u32, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            self.validate_launch(grid_x.max(1), count as u32, 1, block_x, 1, 1)?;
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
                self.stream.launch(&func, grid, block, 0, args)?;
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

        // Build time-major dual-fp32 prefix P per series
        let ps_tm = build_prefixes_time_major(data_tm_f32, cols, rows, &first_valids);

        // VRAM estimate (checked) + headroom
        let elems = cols.checked_mul(rows).ok_or_else(|| CudaDpoError::InvalidInput("cols*rows overflow".into()))?;
        let headroom = 64 * 1024 * 1024; // 64 MiB safety margin
        let bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|b| b.checked_add((elems + 1).checked_mul(std::mem::size_of::<Float2>())?))
            .and_then(|b| b.checked_add(cols.checked_mul(std::mem::size_of::<i32>())?))
            .and_then(|b| b.checked_add(elems.checked_mul(std::mem::size_of::<f32>())?))
            .and_then(|b| b.checked_add(headroom))
            .ok_or_else(|| CudaDpoError::InvalidInput("size overflow computing allocation".into()))?;
        if let Some((free, _total)) = Self::will_fit(bytes, headroom) {
            if bytes > free {
                return Err(CudaDpoError::OutOfMemory { required: bytes, free, headroom });
            }
        }

        let d_data = self.upload_slice(data_tm_f32)?;
        let d_ps = self.upload_slice(&ps_tm)?;
        let d_fv = self.upload_slice(&first_valids)?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(elems, &self.stream)
                .map_err(CudaDpoError::Cuda)?
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
        // Ensure completion so Python CAI can omit a stream key
        self.stream.synchronize()?;

        Ok(DeviceArrayF32 { buf: d_out, rows, cols })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_kernel(
        &self,
        d_data: &DeviceBuffer<f32>,
        d_ps: &DeviceBuffer<Float2>,
        d_fv: &DeviceBuffer<i32>,
        cols: i32,
        rows: i32,
        period: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaDpoError> {
        if cols <= 0 || rows <= 0 {
            return Ok(());
        }
        if cols <= 0 || rows <= 0 {
            return Ok(());
        }
        let func = self
            .module
            .get_function("dpo_many_series_one_param_time_major_f32")
            .map_err(|_| CudaDpoError::MissingKernelSymbol { name: "dpo_many_series_one_param_time_major_f32" })?;

        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } if block_x > 0 => block_x,
            _ => 256,
        };
        // Use 1D over time and y-dim over series for modest sizes
        let grid_x = ((rows as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), cols as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        self.validate_launch(grid_x.max(1), cols as u32, 1, block_x, 1, 1)?;

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
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    #[inline]
    fn validate_launch(&self, gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32) -> Result<(), CudaDpoError> {
        let dev = Device::get_device(self.device_id)?;
        let max_threads = dev
            .get_attribute(cust::device::DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_bx = dev
            .get_attribute(cust::device::DeviceAttribute::MaxBlockDimX)
            .unwrap_or(1024) as u32;
        let max_by = dev
            .get_attribute(cust::device::DeviceAttribute::MaxBlockDimY)
            .unwrap_or(1024) as u32;
        let max_bz = dev
            .get_attribute(cust::device::DeviceAttribute::MaxBlockDimZ)
            .unwrap_or(64) as u32;
        let max_gx = dev
            .get_attribute(cust::device::DeviceAttribute::MaxGridDimX)
            .unwrap_or(2_147_483_647) as u32;
        let max_gy = dev
            .get_attribute(cust::device::DeviceAttribute::MaxGridDimY)
            .unwrap_or(65_535) as u32;
        let max_gz = dev
            .get_attribute(cust::device::DeviceAttribute::MaxGridDimZ)
            .unwrap_or(65_535) as u32;

        let threads = bx.saturating_mul(by).saturating_mul(bz);
        if threads > max_threads || bx > max_bx || by > max_by || bz > max_bz || gx > max_gx || gy > max_gy || gz > max_gz {
            return Err(CudaDpoError::LaunchConfigTooLarge { gx, gy, gz, bx, by, bz });
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

        let combos = expand_grid(sweep)?;
        if combos.is_empty() {
            return Err(CudaDpoError::InvalidInput("no parameter combinations".into()));
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

#[inline(always)]
fn kahan_add(mut hi: f32, mut lo: f32, v: f32) -> (f32, f32) {
    let y = v - lo;
    let t = hi + y;
    lo = (t - hi) - y;
    hi = t;
    (hi, lo)
}

fn build_prefixes_from_first(data: &[f32], first_valid: usize) -> Vec<Float2> {
    let len = data.len();
    // length+1, 1-based prefix; ps[w] holds sum over [first_valid .. w-1]
    let mut ps = vec![Float2 { x: 0.0, y: 0.0 }; len + 1];
    let (mut hi, mut lo) = (0.0f32, 0.0f32);
    for i in 0..len {
        if i >= first_valid {
            (hi, lo) = kahan_add(hi, lo, data[i]);
        }
        let w = i + 1;
        ps[w] = Float2 { x: hi, y: lo };
    }
    ps
}

fn build_prefixes_time_major(
    data_tm: &[f32],
    cols: usize,
    rows: usize,
    first_valids: &[i32],
) -> Vec<Float2> {
    let total = data_tm.len();
    let mut ps = vec![Float2 { x: 0.0, y: 0.0 }; total + 1];
    for s in 0..cols {
        let fv = first_valids[s].max(0) as usize;
        let (mut hi, mut lo) = (0.0f32, 0.0f32);
        for t in 0..rows {
            if t >= fv {
                let v = data_tm[t * cols + s];
                (hi, lo) = kahan_add(hi, lo, v);
            }
            let w = (t * cols + s) + 1;
            ps[w] = Float2 { x: hi, y: lo };
        }
    }
    ps
}

// ---- Grid expand (mirror indicator) ----

fn expand_grid(r: &DpoBatchRange) -> Result<Vec<DpoParams>, CudaDpoError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, CudaDpoError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let mut v = Vec::new();
            let mut x = start;
            let st = step.max(1);
            while x <= end {
                v.push(x);
                match x.checked_add(st) {
                    Some(nx) => x = nx,
                    None => break,
                }
            }
            if v.is_empty() {
                return Err(CudaDpoError::InvalidInput(format!(
                    "invalid period range: start={} end={} step={} (empty expansion)",
                    start, end, step
                )));
            }
            return Ok(v);
        }
        // reversed bounds: walk downward in `step` units
        let mut v = Vec::new();
        let mut x = start as isize;
        let end_i = end as isize;
        let st = (step as isize).max(1);
        while x >= end_i {
            v.push(x as usize);
            x -= st;
        }
        if v.is_empty() {
            return Err(CudaDpoError::InvalidInput(format!(
                "invalid period range: start={} end={} step={} (empty expansion)",
                start, end, step
            )));
        }
        Ok(v)
    }
    let periods = axis_usize(r.period)?;
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(DpoParams { period: Some(p) });
    }
    Ok(out)
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
        let prefix_bytes = (ONE_SERIES_LEN + 1) * std::mem::size_of::<Float2>();
        in_bytes + out_bytes + prefix_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_ROWS;
        let in_bytes = elems * std::mem::size_of::<f32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        let prefix_bytes = (elems + 1) * std::mem::size_of::<Float2>();
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
        Box::new(DpoManyState { cuda, data_tm, cols, rows, params })
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
