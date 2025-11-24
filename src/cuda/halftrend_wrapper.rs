#![cfg(feature = "cuda")]

//! CUDA wrapper for the HalfTrend indicator.
//!
//! Category: recurrence/time-scan with shared precompute across rows.
//! - Batch (one-series × many-params): precompute ATR, SMA(high/low) and
//!   rolling window extrema once per unique period/amplitude on the host,
//!   duplicate into row-major matrices, and scan per row on device.
//! - Many-series × one-param (time-major): accept H/L/C matrices laid out as
//!   time-major (index = t*cols + s), precompute required helpers on host per
//!   series, and scan per series on device.
//!
//! Parity goals with ALMA wrapper:
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/halftrend_kernel.ptx"))
//!   with DetermineTargetFromContext and OptLevel O2 (fallbacks applied).
//! - Non-blocking stream, policy enums, once-per-instance debug logging.
//! - Warmup/NaN semantics identical to scalar halftrend.rs.

use crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32;
use crate::indicators::atr::{atr, AtrInput, AtrParams};
use crate::indicators::halftrend::{HalfTrendBatchRange, HalfTrendParams};
use crate::indicators::moving_averages::sma::{sma, SmaInput, SmaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

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
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
    TimeMajor { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

// Results returned by device batch/many-series APIs (module-scope structs).
pub struct CudaHalftrendBatch {
    pub halftrend: DeviceArrayF32,
    pub trend: DeviceArrayF32,
    pub atr_high: DeviceArrayF32,
    pub atr_low: DeviceArrayF32,
    pub buy: DeviceArrayF32,
    pub sell: DeviceArrayF32,
    pub combos: Vec<HalfTrendParams>,
}

pub struct CudaHalftrendMany {
    pub halftrend: DeviceArrayF32,
    pub trend: DeviceArrayF32,
    pub atr_high: DeviceArrayF32,
    pub atr_low: DeviceArrayF32,
    pub buy: DeviceArrayF32,
    pub sell: DeviceArrayF32,
}

#[derive(Error, Debug)]
pub enum CudaHalftrendError {
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

pub struct CudaHalftrend {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    batch_policy: BatchKernelPolicy,
    many_policy: ManySeriesKernelPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaHalftrend {
    pub fn new(device_id: usize) -> Result<Self, CudaHalftrendError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/halftrend_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                    m
                } else {
                    Module::from_ptx(ptx, &[])?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            batch_policy: BatchKernelPolicy::Auto,
            many_policy: ManySeriesKernelPolicy::Auto,
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn context_arc(&self) -> Arc<Context> { self.context.clone() }
    pub fn device_id(&self) -> u32 { self.device_id }

    pub fn set_batch_policy(&mut self, p: BatchKernelPolicy) { self.batch_policy = p; }
    pub fn set_many_series_policy(&mut self, p: ManySeriesKernelPolicy) { self.many_policy = p; }
    pub fn batch_policy(&self) -> BatchKernelPolicy { self.batch_policy }
    pub fn many_series_policy(&self) -> ManySeriesKernelPolicy { self.many_policy }

    #[inline]
    fn mem_ok(required_bytes: usize, headroom: usize) -> bool {
        match mem_get_info() {
            Ok((free, _)) => required_bytes.saturating_add(headroom) <= free,
            Err(_) => true,
        }
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if let Ok((free, _)) = mem_get_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    #[inline]
    fn first_valid_ohlc_f32(high: &[f32], low: &[f32], close: &[f32]) -> Option<usize> {
        let n = high.len().min(low.len()).min(close.len());
        for i in 0..n {
            if !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan() {
                return Some(i);
            }
        }
        None
    }

    // ---------------------- Batch (one-series × many-params) ----------------------

    fn expand_grid(range: &HalfTrendBatchRange) -> Result<Vec<HalfTrendParams>, CudaHalftrendError> {
        fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, CudaHalftrendError> {
            if step == 0 || start == end { return Ok(vec![start]); }
            if start < end { return Ok((start..=end).step_by(step.max(1)).collect()); }
            let mut v = Vec::new();
            let mut x = start as isize;
            let end_i = end as isize;
            let st = (step as isize).max(1);
            while x >= end_i { v.push(x as usize); x -= st; }
            if v.is_empty() { return Err(CudaHalftrendError::InvalidInput("empty usize axis".into())); }
            Ok(v)
        }
        fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, CudaHalftrendError> {
            if step.abs() < 1e-12 || (start - end).abs() < 1e-12 { return Ok(vec![start]); }
            if start < end {
                let mut v = Vec::new(); let mut x = start; let st = step.abs();
                while x <= end + 1e-12 { v.push(x); x += st; }
                if v.is_empty() { return Err(CudaHalftrendError::InvalidInput("empty f64 axis".into())); }
                return Ok(v);
            }
            let mut v = Vec::new(); let mut x = start; let st = step.abs();
            while x + 1e-12 >= end { v.push(x); x -= st; }
            if v.is_empty() { return Err(CudaHalftrendError::InvalidInput("empty f64 axis".into())); }
            Ok(v)
        }
        let amps = axis_usize(range.amplitude)?;
        let cds = axis_f64(range.channel_deviation)?;
        let atrs = axis_usize(range.atr_period)?;
        let cap = amps.len().checked_mul(cds.len()).and_then(|x| x.checked_mul(atrs.len()))
            .ok_or_else(|| CudaHalftrendError::InvalidInput("combination overflow".into()))?;
        let mut out = Vec::with_capacity(cap);
        for &a in &amps { for &c in &cds { for &p in &atrs {
            out.push(HalfTrendParams { amplitude: Some(a), channel_deviation: Some(c), atr_period: Some(p) });
        } } }
        Ok(out)
    }

    #[inline]
    fn validate_launch(&self, gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32) -> Result<(), CudaHalftrendError> {
        let dev = Device::get_device(self.device_id)?;
        let max_threads = dev
            .get_attribute(cust::device::DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_bx = dev.get_attribute(cust::device::DeviceAttribute::MaxBlockDimX).unwrap_or(1024) as u32;
        let max_by = dev.get_attribute(cust::device::DeviceAttribute::MaxBlockDimY).unwrap_or(1024) as u32;
        let max_bz = dev.get_attribute(cust::device::DeviceAttribute::MaxBlockDimZ).unwrap_or(64) as u32;
        let max_gx = dev.get_attribute(cust::device::DeviceAttribute::MaxGridDimX).unwrap_or(2_147_483_647) as u32;
        let max_gy = dev.get_attribute(cust::device::DeviceAttribute::MaxGridDimY).unwrap_or(65_535) as u32;
        let max_gz = dev.get_attribute(cust::device::DeviceAttribute::MaxGridDimZ).unwrap_or(65_535) as u32;
        let threads = bx.saturating_mul(by).saturating_mul(bz);
        if threads > max_threads || bx > max_bx || by > max_by || bz > max_bz || gx > max_gx || gy > max_gy || gz > max_gz {
            return Err(CudaHalftrendError::LaunchConfigTooLarge { gx, gy, gz, bx, by, bz });
        }
        Ok(())
    }

    fn rolling_max(src: &[f64], period: usize) -> Vec<f64> {
        let n = src.len();
        if n == 0 || period == 0 {
            return vec![f64::NAN; n];
        }
        if n == 0 || period == 0 {
            return vec![f64::NAN; n];
        }
        let cap = period;
        let mut idx = vec![0usize; cap];
        let mut val = vec![f64::NAN; cap];
        let mut head = 0usize;
        let mut tail = 0usize;
        let mut cnt = 0usize;
        let mut head = 0usize;
        let mut tail = 0usize;
        let mut cnt = 0usize;
        let mut out = vec![f64::NAN; n];
        let inc = |i: usize, c: usize| if i + 1 == c { 0 } else { i + 1 };
        let dec = |i: usize, c: usize| if i == 0 { c - 1 } else { i - 1 };
        for i in 0..n {
            let wstart = i + 1 - period.min(i + 1);
            while cnt > 0 && idx[head] < wstart {
                head = inc(head, cap);
                cnt -= 1;
            }
            while cnt > 0 && idx[head] < wstart {
                head = inc(head, cap);
                cnt -= 1;
            }
            let x = src[i];
            while cnt > 0 {
                let back = dec(tail, cap);
                if val[back] <= x {
                    tail = back;
                    cnt -= 1;
                } else {
                    break;
                }
            }
            val[tail] = x;
            idx[tail] = i;
            tail = inc(tail, cap);
            cnt += 1;
            out[i] = val[head];
            while cnt > 0 {
                let back = dec(tail, cap);
                if val[back] <= x {
                    tail = back;
                    cnt -= 1;
                } else {
                    break;
                }
            }
            val[tail] = x;
            idx[tail] = i;
            tail = inc(tail, cap);
            cnt += 1;
            out[i] = val[head];
        }
        out
    }
    fn rolling_min(src: &[f64], period: usize) -> Vec<f64> {
        let n = src.len();
        if n == 0 || period == 0 {
            return vec![f64::NAN; n];
        }
        if n == 0 || period == 0 {
            return vec![f64::NAN; n];
        }
        let cap = period;
        let mut idx = vec![0usize; cap];
        let mut val = vec![f64::NAN; cap];
        let mut head = 0usize;
        let mut tail = 0usize;
        let mut cnt = 0usize;
        let mut head = 0usize;
        let mut tail = 0usize;
        let mut cnt = 0usize;
        let mut out = vec![f64::NAN; n];
        let inc = |i: usize, c: usize| if i + 1 == c { 0 } else { i + 1 };
        let dec = |i: usize, c: usize| if i == 0 { c - 1 } else { i - 1 };
        for i in 0..n {
            let wstart = i + 1 - period.min(i + 1);
            while cnt > 0 && idx[head] < wstart {
                head = inc(head, cap);
                cnt -= 1;
            }
            while cnt > 0 && idx[head] < wstart {
                head = inc(head, cap);
                cnt -= 1;
            }
            let x = src[i];
            while cnt > 0 {
                let back = dec(tail, cap);
                if val[back] >= x {
                    tail = back;
                    cnt -= 1;
                } else {
                    break;
                }
            }
            val[tail] = x;
            idx[tail] = i;
            tail = inc(tail, cap);
            cnt += 1;
            out[i] = val[head];
            while cnt > 0 {
                let back = dec(tail, cap);
                if val[back] >= x {
                    tail = back;
                    cnt -= 1;
                } else {
                    break;
                }
            }
            val[tail] = x;
            idx[tail] = i;
            tail = inc(tail, cap);
            cnt += 1;
            out[i] = val[head];
        }
        out
    }

    pub fn halftrend_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &HalfTrendBatchRange,
    ) -> Result<CudaHalftrendBatch, CudaHalftrendError> {
        if high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaHalftrendError::InvalidInput("empty input".into()));
        }
        let n = high.len().min(low.len()).min(close.len());
        let first = Self::first_valid_ohlc_f32(high, low, close)
            .ok_or_else(|| CudaHalftrendError::InvalidInput("all values are NaN".into()))?;

        let combos = Self::expand_grid(sweep)?;
        if combos.is_empty() {
            return Err(CudaHalftrendError::InvalidInput("no parameter combinations".into()));
        }
        let rows = combos.len();
        // Rough VRAM requirement: inputs 3N + helpers 5*rows*N + warms + params + 6*rows*N
        let req_bytes = (3usize
            .checked_mul(n)
            .and_then(|x| x.checked_add(5usize.checked_mul(rows).and_then(|y| y.checked_mul(n))?))
            .and_then(|x| x.checked_add(rows))
            .and_then(|x| x.checked_add(6usize.checked_mul(rows).and_then(|y| y.checked_mul(n))?))
            .ok_or_else(|| CudaHalftrendError::InvalidInput("size overflow".into()))?)
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|x| x.checked_add(64 * 1024 * 1024))
            .ok_or_else(|| CudaHalftrendError::InvalidInput("size overflow".into()))?;
        if let Ok((free, _)) = mem_get_info() {
            if !Self::will_fit(req_bytes, 0) {
                return Err(CudaHalftrendError::OutOfMemory { required: req_bytes, free, headroom: 0 });
            }
        }

        // Shared precompute on host by unique periods/amplitudes
        use std::collections::{BTreeSet, HashMap};
        let amps: BTreeSet<usize> = combos.iter().map(|p| p.amplitude.unwrap()).collect();
        let atrs: BTreeSet<usize> = combos.iter().map(|p| p.atr_period.unwrap()).collect();

        let high_f64: Vec<f64> = high.iter().map(|&v| v as f64).collect();
        let low_f64: Vec<f64> = low.iter().map(|&v| v as f64).collect();
        let close_f64: Vec<f64> = close.iter().map(|&v| v as f64).collect();

        let mut hma_map: HashMap<usize, Vec<f64>> = HashMap::new();
        let mut lma_map: HashMap<usize, Vec<f64>> = HashMap::new();
        let mut rhi_map: HashMap<usize, Vec<f64>> = HashMap::new();
        let mut rlo_map: HashMap<usize, Vec<f64>> = HashMap::new();
        for &a in &amps {
            let SmaParams { .. } = SmaParams { period: Some(a) };
            hma_map.insert(
                a,
                sma(&SmaInput::from_slice(
                    &high_f64,
                    SmaParams { period: Some(a) },
                ))
                .map_err(|e| CudaHalftrendError::InvalidInput(e.to_string()))?
                .values,
            );
            lma_map.insert(
                a,
                sma(&SmaInput::from_slice(
                    &low_f64,
                    SmaParams { period: Some(a) },
                ))
                .map_err(|e| CudaHalftrendError::InvalidInput(e.to_string()))?
                .values,
            );
            rhi_map.insert(a, Self::rolling_max(&high_f64, a));
            rlo_map.insert(a, Self::rolling_min(&low_f64, a));
        }
        let mut atr_map: HashMap<usize, Vec<f64>> = HashMap::new();
        for &p in &atrs {
            atr_map.insert(
                p,
                atr(&AtrInput::from_slices(
                    &high_f64,
                    &low_f64,
                    &close_f64,
                    AtrParams { length: Some(p) },
                ))
                .map_err(|e| CudaHalftrendError::InvalidInput(e.to_string()))?
                .values,
            );
        }

        // Heuristic: prefer time-major path for larger combo counts and lengths.
        let use_time_major = match self.batch_policy {
            BatchKernelPolicy::Auto => (rows >= 16) && (n >= 8192),
            BatchKernelPolicy::Plain { .. } => false,
        };

        // Build helper matrices in either row-major (legacy) or time-major (optimized) layout
        use cust::memory::LockedBuffer;
        let mut warms = vec![0i32; rows];
        let mut chdevs = vec![0f32; rows];

        // Common per-row fields
        for (row, prm) in combos.iter().enumerate() {
            let a = prm.amplitude.unwrap();
            let p = prm.atr_period.unwrap();
            let ch = prm.channel_deviation.unwrap_or(2.0) as f32;
            chdevs[row] = ch;
            let warm = first + a.max(p) - 1;
            warms[row] = warm.min(n) as i32;
        }

        // Prepare device inputs and helper buffers
        let d_high = unsafe { DeviceBuffer::from_slice_async(&high[..n], &self.stream) }
            .map_err(CudaHalftrendError::from)?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(&low[..n], &self.stream) }
            .map_err(CudaHalftrendError::from)?;
        let d_close = unsafe { DeviceBuffer::from_slice_async(&close[..n], &self.stream) }
            .map_err(CudaHalftrendError::from)?;
        let d_warms = DeviceBuffer::from_slice(&warms)
            .map_err(CudaHalftrendError::from)?;
        let d_chdevs = DeviceBuffer::from_slice(&chdevs)
            .map_err(CudaHalftrendError::from)?;

        // Outputs (we keep external row-major metadata: rows x n)
        let elems = rows
            .checked_mul(n)
            .ok_or_else(|| CudaHalftrendError::InvalidInput("size overflow".into()))?;
        let mut d_ht: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }
                .map_err(CudaHalftrendError::from)?;
        let mut d_tr: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }
                .map_err(CudaHalftrendError::from)?;
        let mut d_ah: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }
                .map_err(CudaHalftrendError::from)?;
        let mut d_al: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }
                .map_err(CudaHalftrendError::from)?;
        let mut d_bs: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }
                .map_err(CudaHalftrendError::from)?;
        let mut d_ss: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }
                .map_err(CudaHalftrendError::from)?;

        if !use_time_major {
            // Legacy ROW-MAJOR [rows, n] helper matrices
            let mut atr_rows = vec![0f32; rows * n];
            let mut hma_rows = vec![0f32; rows * n];
            let mut lma_rows = vec![0f32; rows * n];
            let mut rhi_rows = vec![0f32; rows * n];
            let mut rlo_rows = vec![0f32; rows * n];
            for (row, prm) in combos.iter().enumerate() {
                let a = prm.amplitude.unwrap();
                let p = prm.atr_period.unwrap();
                let base = row * n;
                let atrv = &atr_map[&p];
                let hmv = &hma_map[&a];
                let lmv = &lma_map[&a];
                let rhv = &rhi_map[&a];
                let rlv = &rlo_map[&a];
                for i in 0..n {
                    atr_rows[base + i] = atrv[i] as f32;
                    hma_rows[base + i] = hmv[i] as f32;
                    lma_rows[base + i] = lmv[i] as f32;
                    rhi_rows[base + i] = rhv[i] as f32;
                    rlo_rows[base + i] = rlv[i] as f32;
                }
            }

            // Upload legacy helpers
            let d_atr = unsafe { DeviceBuffer::from_slice_async(&atr_rows, &self.stream) }
                .map_err(CudaHalftrendError::from)?;
            let d_hma = unsafe { DeviceBuffer::from_slice_async(&hma_rows, &self.stream) }
                .map_err(CudaHalftrendError::from)?;
            let d_lma = unsafe { DeviceBuffer::from_slice_async(&lma_rows, &self.stream) }
                .map_err(CudaHalftrendError::from)?;
            let d_rhi = unsafe { DeviceBuffer::from_slice_async(&rhi_rows, &self.stream) }
                .map_err(CudaHalftrendError::from)?;
            let d_rlo = unsafe { DeviceBuffer::from_slice_async(&rlo_rows, &self.stream) }
                .map_err(CudaHalftrendError::from)?;

            // Launch row-major kernel (keeps existing ABI)
            let func = self
                .module
                .get_function("halftrend_batch_f32")
                .map_err(|_| CudaHalftrendError::MissingKernelSymbol { name: "halftrend_batch_f32" })?;
            let block_x = match self.batch_policy { BatchKernelPolicy::Auto => 256, BatchKernelPolicy::Plain { block_x } => block_x.max(32) };
            if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") && !self.debug_batch_logged {
                eprintln!("[halftrend] batch kernel (row-major): block_x={} rows={} len={} first_valid={}",
                    block_x, rows, n, first);
                unsafe { (*(self as *const _ as *mut CudaHalftrend)).debug_batch_logged = true; }
            }
            unsafe { (*(self as *const _ as *mut CudaHalftrend)).last_batch = Some(BatchKernelSelected::Plain { block_x }); }

            unsafe {
                let grid_x = (((rows as u32) + block_x - 1) / block_x).max(1);
                self.validate_launch(grid_x, 1, 1, block_x, 1, 1)?;
                let grid: GridSize = (grid_x, 1, 1).into();
                let block: BlockSize = (block_x, 1, 1).into();
                let mut h = d_high.as_device_ptr().as_raw();
                let mut l = d_low.as_device_ptr().as_raw();
                let mut c = d_close.as_device_ptr().as_raw();
                let mut a = d_atr.as_device_ptr().as_raw();
                let mut hm = d_hma.as_device_ptr().as_raw();
                let mut lm = d_lma.as_device_ptr().as_raw();
                let mut rh = d_rhi.as_device_ptr().as_raw();
                let mut rl = d_rlo.as_device_ptr().as_raw();
                let mut w = d_warms.as_device_ptr().as_raw();
                let mut cd = d_chdevs.as_device_ptr().as_raw();
                let mut n_i = n as i32;
                let mut r_i = rows as i32;
                let mut oht = d_ht.as_device_ptr().as_raw();
                let mut otr = d_tr.as_device_ptr().as_raw();
                let mut oah = d_ah.as_device_ptr().as_raw();
                let mut oal = d_al.as_device_ptr().as_raw();
                let mut obs = d_bs.as_device_ptr().as_raw();
                let mut oss = d_ss.as_device_ptr().as_raw();
                let mut args: [*mut c_void; 18] = [
                    &mut h as *mut _ as *mut c_void,
                    &mut l as *mut _ as *mut c_void,
                    &mut c as *mut _ as *mut c_void,
                    &mut a as *mut _ as *mut c_void,
                    &mut hm as *mut _ as *mut c_void,
                    &mut lm as *mut _ as *mut c_void,
                    &mut rh as *mut _ as *mut c_void,
                    &mut rl as *mut _ as *mut c_void,
                    &mut w as *mut _ as *mut c_void,
                    &mut cd as *mut _ as *mut c_void,
                    &mut n_i as *mut _ as *mut c_void,
                    &mut r_i as *mut _ as *mut c_void,
                    &mut oht as *mut _ as *mut c_void,
                    &mut otr as *mut _ as *mut c_void,
                    &mut oah as *mut _ as *mut c_void,
                    &mut oal as *mut _ as *mut c_void,
                    &mut obs as *mut _ as *mut c_void,
                    &mut oss as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, 0, &mut args)
                    .map_err(CudaHalftrendError::from)?;
            }
        } else {
            // TIME-MAJOR [n, rows] helper matrices in pinned memory
            let len_tm = rows
                .checked_mul(n)
                .ok_or_else(|| {
                    CudaHalftrendError::InvalidInput("size overflow".into())
                })?;
            let mut atr_tm =
                unsafe { LockedBuffer::<f32>::uninitialized(len_tm) }
                    .map_err(CudaHalftrendError::from)?;
            let mut hma_tm =
                unsafe { LockedBuffer::<f32>::uninitialized(len_tm) }
                    .map_err(CudaHalftrendError::from)?;
            let mut lma_tm =
                unsafe { LockedBuffer::<f32>::uninitialized(len_tm) }
                    .map_err(CudaHalftrendError::from)?;
            let mut rhi_tm =
                unsafe { LockedBuffer::<f32>::uninitialized(len_tm) }
                    .map_err(CudaHalftrendError::from)?;
            let mut rlo_tm =
                unsafe { LockedBuffer::<f32>::uninitialized(len_tm) }
                    .map_err(CudaHalftrendError::from)?;

            for (row, prm) in combos.iter().enumerate() {
                let a = prm.amplitude.unwrap();
                let p = prm.atr_period.unwrap();
                let atrv = &atr_map[&p];
                let hmv = &hma_map[&a];
                let lmv = &lma_map[&a];
                let rhv = &rhi_map[&a];
                let rlv = &rlo_map[&a];
                for t in 0..n {
                    let idx = t * rows + row;
                    atr_tm[idx] = atrv[t] as f32;
                    hma_tm[idx] = hmv[t] as f32;
                    lma_tm[idx] = lmv[t] as f32;
                    rhi_tm[idx] = rhv[t] as f32;
                    rlo_tm[idx] = rlv[t] as f32;
                }
            }

            // Device helpers (uninitialized + async copy from pinned host)
            let mut d_atr: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized_async(len_tm, &self.stream) }
                    .map_err(CudaHalftrendError::from)?;
            let mut d_hma: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized_async(len_tm, &self.stream) }
                    .map_err(CudaHalftrendError::from)?;
            let mut d_lma: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized_async(len_tm, &self.stream) }
                    .map_err(CudaHalftrendError::from)?;
            let mut d_rhi: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized_async(len_tm, &self.stream) }
                    .map_err(CudaHalftrendError::from)?;
            let mut d_rlo: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized_async(len_tm, &self.stream) }
                    .map_err(CudaHalftrendError::from)?;

            unsafe {
                d_atr
                    .async_copy_from(&atr_tm, &self.stream)
                    .map_err(CudaHalftrendError::from)?;
                d_hma
                    .async_copy_from(&hma_tm, &self.stream)
                    .map_err(CudaHalftrendError::from)?;
                d_lma
                    .async_copy_from(&lma_tm, &self.stream)
                    .map_err(CudaHalftrendError::from)?;
                d_rhi
                    .async_copy_from(&rhi_tm, &self.stream)
                    .map_err(CudaHalftrendError::from)?;
                d_rlo
                    .async_copy_from(&rlo_tm, &self.stream)
                    .map_err(CudaHalftrendError::from)?;
            }

            // Launch TIME-MAJOR kernel
            let func = self
                .module
                .get_function("halftrend_batch_time_major_f32")
                .map_err(|_| CudaHalftrendError::MissingKernelSymbol { name: "halftrend_batch_time_major_f32" })?;
            let block_x = 256u32;
            if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") && !self.debug_batch_logged {
                eprintln!("[halftrend] batch kernel (time-major): block_x={} rows={} len={} first_valid={}",
                    block_x, rows, n, first);
                unsafe { (*(self as *const _ as *mut CudaHalftrend)).debug_batch_logged = true; }
            }
            unsafe { (*(self as *const _ as *mut CudaHalftrend)).last_batch = Some(BatchKernelSelected::TimeMajor { block_x }); }

            // Note: kernel outputs time-major [n, rows] into the same d_* buffers.
            unsafe {
                let grid_x = (((rows as u32) + block_x - 1) / block_x).max(1);
                self.validate_launch(grid_x, 1, 1, block_x, 1, 1)?;
                let grid: GridSize = (grid_x, 1, 1).into();
                let block: BlockSize = (block_x, 1, 1).into();
                let mut h = d_high.as_device_ptr().as_raw();
                let mut l = d_low.as_device_ptr().as_raw();
                let mut c = d_close.as_device_ptr().as_raw();
                let mut a = d_atr.as_device_ptr().as_raw();
                let mut hm = d_hma.as_device_ptr().as_raw();
                let mut lm = d_lma.as_device_ptr().as_raw();
                let mut rh = d_rhi.as_device_ptr().as_raw();
                let mut rl = d_rlo.as_device_ptr().as_raw();
                let mut w = d_warms.as_device_ptr().as_raw();
                let mut cd = d_chdevs.as_device_ptr().as_raw();
                let mut n_i = n as i32;
                let mut r_i = rows as i32;
                let mut oht = d_ht.as_device_ptr().as_raw();
                let mut otr = d_tr.as_device_ptr().as_raw();
                let mut oah = d_ah.as_device_ptr().as_raw();
                let mut oal = d_al.as_device_ptr().as_raw();
                let mut obs = d_bs.as_device_ptr().as_raw();
                let mut oss = d_ss.as_device_ptr().as_raw();
                let mut args: [*mut c_void; 18] = [
                    &mut h as *mut _ as *mut c_void,
                    &mut l as *mut _ as *mut c_void,
                    &mut c as *mut _ as *mut c_void,
                    &mut a as *mut _ as *mut c_void,
                    &mut hm as *mut _ as *mut c_void,
                    &mut lm as *mut _ as *mut c_void,
                    &mut rh as *mut _ as *mut c_void,
                    &mut rl as *mut _ as *mut c_void,
                    &mut w as *mut _ as *mut c_void,
                    &mut cd as *mut _ as *mut c_void,
                    &mut n_i as *mut _ as *mut c_void,
                    &mut r_i as *mut _ as *mut c_void,
                    &mut oht as *mut _ as *mut c_void,
                    &mut otr as *mut _ as *mut c_void,
                    &mut oah as *mut _ as *mut c_void,
                    &mut oal as *mut _ as *mut c_void,
                    &mut obs as *mut _ as *mut c_void,
                    &mut oss as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, 0, &mut args)
                    .map_err(CudaHalftrendError::from)?;
            }
        }

        self.stream
            .synchronize()
            .map_err(CudaHalftrendError::from)?;

        Ok(CudaHalftrendBatch {
            // Keep external shape as [rows, n] to preserve API/tests
            halftrend: DeviceArrayF32 { buf: d_ht, rows, cols: n },
            trend: DeviceArrayF32 { buf: d_tr, rows, cols: n },
            atr_high: DeviceArrayF32 { buf: d_ah, rows, cols: n },
            atr_low: DeviceArrayF32 { buf: d_al, rows, cols: n },
            buy: DeviceArrayF32 { buf: d_bs, rows, cols: n },
            sell: DeviceArrayF32 { buf: d_ss, rows, cols: n },
            combos,
        })
    }

    // ---------------------- Many-series × one-param (time-major) ----------------------

    pub fn halftrend_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        amplitude: usize,
        channel_deviation: f64,
        atr_period: usize,
    ) -> Result<CudaHalftrendMany, CudaHalftrendError> {
        if cols == 0 || rows == 0 {
            return Err(CudaHalftrendError::InvalidInput("empty matrix".into()));
        }
        if high_tm.len() != cols * rows
            || low_tm.len() != cols * rows
            || close_tm.len() != cols * rows
        {
            return Err(CudaHalftrendError::InvalidInput(
                "matrix shape mismatch".into(),
            ));
        }

        // Per-series first_valid and warm
        let mut firsts = vec![rows as i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                if !high_tm[idx].is_nan() && !low_tm[idx].is_nan() && !close_tm[idx].is_nan() {
                    firsts[s] = t as i32;
                    break;
                }
            }
            if firsts[s] as usize >= rows {
                return Err(CudaHalftrendError::InvalidInput(
                    "all values are NaN for a series".into(),
                ));
            }
        }
        let mut warms = vec![0i32; cols];
        for s in 0..cols {
            warms[s] = (firsts[s] as usize + amplitude.max(atr_period) - 1).min(rows) as i32;
        }
        for s in 0..cols {
            warms[s] = (firsts[s] as usize + amplitude.max(atr_period) - 1).min(rows) as i32;
        }

        // Host precompute per series (ATR + SMA + rolling extrema) and flatten back to TM
        // Use page-locked buffers to enable true async H->D copies.
        let len_tm = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaHalftrendError::InvalidInput("size overflow".into()))?;
        let mut atr_tm =
            unsafe { LockedBuffer::<f32>::uninitialized(len_tm) }
                .map_err(CudaHalftrendError::from)?;
        let mut hma_tm =
            unsafe { LockedBuffer::<f32>::uninitialized(len_tm) }
                .map_err(CudaHalftrendError::from)?;
        let mut lma_tm =
            unsafe { LockedBuffer::<f32>::uninitialized(len_tm) }
                .map_err(CudaHalftrendError::from)?;
        let mut rhi_tm =
            unsafe { LockedBuffer::<f32>::uninitialized(len_tm) }
                .map_err(CudaHalftrendError::from)?;
        let mut rlo_tm =
            unsafe { LockedBuffer::<f32>::uninitialized(len_tm) }
                .map_err(CudaHalftrendError::from)?;
        for s in 0..cols {
            let mut h = vec![f64::NAN; rows];
            let mut l = vec![f64::NAN; rows];
            let mut c = vec![f64::NAN; rows];
            for t in 0..rows {
                let idx = t * cols + s;
                h[t] = high_tm[idx] as f64;
                l[t] = low_tm[idx] as f64;
                c[t] = close_tm[idx] as f64;
            }
            let atr_v = atr(&AtrInput::from_slices(
                &h,
                &l,
                &c,
                AtrParams {
                    length: Some(atr_period),
                },
            ))
            .map_err(|e| CudaHalftrendError::InvalidInput(e.to_string()))?
            .values;
            let hma_v = sma(&SmaInput::from_slice(
                &h,
                SmaParams {
                    period: Some(amplitude),
                },
            ))
            .map_err(|e| CudaHalftrendError::InvalidInput(e.to_string()))?
            .values;
            let lma_v = sma(&SmaInput::from_slice(
                &l,
                SmaParams {
                    period: Some(amplitude),
                },
            ))
            .map_err(|e| CudaHalftrendError::InvalidInput(e.to_string()))?
            .values;
            let rhi_v = Self::rolling_max(&h, amplitude);
            let rlo_v = Self::rolling_min(&l, amplitude);
            for t in 0..rows {
                let idx = t * cols + s;
                atr_tm[idx] = atr_v[t] as f32;
                hma_tm[idx] = hma_v[t] as f32;
                lma_tm[idx] = lma_v[t] as f32;
                rhi_tm[idx] = rhi_v[t] as f32;
                rlo_tm[idx] = rlo_v[t] as f32;
            }
            for t in 0..rows {
                let idx = t * cols + s;
                atr_tm[idx] = atr_v[t] as f32;
                hma_tm[idx] = hma_v[t] as f32;
                lma_tm[idx] = lma_v[t] as f32;
                rhi_tm[idx] = rhi_v[t] as f32;
                rlo_tm[idx] = rlo_v[t] as f32;
            }
        }

        // VRAM estimate: inputs + 5 tm helpers + warms + 6 outputs
        let req = (3 * cols * rows + 5 * cols * rows + cols + 6 * cols * rows)
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|x| x.checked_add(64 * 1024 * 1024))
            .ok_or_else(|| CudaHalftrendError::InvalidInput("size overflow".into()))?;
        if let Ok((free, _)) = mem_get_info() {
            if !Self::will_fit(req, 0) {
                return Err(CudaHalftrendError::OutOfMemory { required: req, free, headroom: 0 });
            }
        }

        let d_high =
            unsafe { DeviceBuffer::from_slice_async(high_tm, &self.stream) }
                .map_err(CudaHalftrendError::from)?;
        let d_low =
            unsafe { DeviceBuffer::from_slice_async(low_tm, &self.stream) }
                .map_err(CudaHalftrendError::from)?;
        let d_close =
            unsafe { DeviceBuffer::from_slice_async(close_tm, &self.stream) }
                .map_err(CudaHalftrendError::from)?;
        let mut d_atr: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(len_tm, &self.stream) }
                .map_err(CudaHalftrendError::from)?;
        let mut d_hma: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(len_tm, &self.stream) }
                .map_err(CudaHalftrendError::from)?;
        let mut d_lma: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(len_tm, &self.stream) }
                .map_err(CudaHalftrendError::from)?;
        let mut d_rhi: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(len_tm, &self.stream) }
                .map_err(CudaHalftrendError::from)?;
        let mut d_rlo: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(len_tm, &self.stream) }
                .map_err(CudaHalftrendError::from)?;

        unsafe {
            d_atr
                .async_copy_from(&atr_tm, &self.stream)
                .map_err(CudaHalftrendError::from)?;
            d_hma
                .async_copy_from(&hma_tm, &self.stream)
                .map_err(CudaHalftrendError::from)?;
            d_lma
                .async_copy_from(&lma_tm, &self.stream)
                .map_err(CudaHalftrendError::from)?;
            d_rhi
                .async_copy_from(&rhi_tm, &self.stream)
                .map_err(CudaHalftrendError::from)?;
            d_rlo
                .async_copy_from(&rlo_tm, &self.stream)
                .map_err(CudaHalftrendError::from)?;
        }
        let d_warms =
            DeviceBuffer::from_slice(&warms).map_err(CudaHalftrendError::from)?;

        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaHalftrendError::InvalidInput("size overflow".into()))?;
        let mut d_ht: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }
                .map_err(CudaHalftrendError::from)?;
        let mut d_tr: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }
                .map_err(CudaHalftrendError::from)?;
        let mut d_ah: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }
                .map_err(CudaHalftrendError::from)?;
        let mut d_al: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }
                .map_err(CudaHalftrendError::from)?;
        let mut d_bs: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }
                .map_err(CudaHalftrendError::from)?;
        let mut d_ss: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }
                .map_err(CudaHalftrendError::from)?;

        // Launch
        let func = self
            .module
            .get_function("halftrend_many_series_one_param_time_major_f32")
            .map_err(|_| CudaHalftrendError::MissingKernelSymbol { name: "halftrend_many_series_one_param_time_major_f32" })?;
        let block_x = match self.many_policy {
            ManySeriesKernelPolicy::Auto => 256,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32),
        };
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") && !self.debug_many_logged {
            eprintln!(
                "[halftrend] many-series kernel: block_x={} cols={} rows={} amp={} atr={} ch={}",
                block_x, cols, rows, amplitude, atr_period, channel_deviation
            );
            unsafe {
                (*(self as *const _ as *mut CudaHalftrend)).debug_many_logged = true;
            }
            unsafe {
                (*(self as *const _ as *mut CudaHalftrend)).last_many =
                    Some(ManySeriesKernelSelected::OneD { block_x });
            }
            eprintln!(
                "[halftrend] many-series kernel: block_x={} cols={} rows={} amp={} atr={} ch={}",
                block_x, cols, rows, amplitude, atr_period, channel_deviation
            );
            unsafe {
                (*(self as *const _ as *mut CudaHalftrend)).debug_many_logged = true;
            }
            unsafe {
                (*(self as *const _ as *mut CudaHalftrend)).last_many =
                    Some(ManySeriesKernelSelected::OneD { block_x });
            }
        }
        unsafe {
            let grid_x = (((cols as u32) + block_x - 1) / block_x).max(1);
            self.validate_launch(grid_x, 1, 1, block_x, 1, 1)?;
            let grid: GridSize = (grid_x, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            let mut h = d_high.as_device_ptr().as_raw();
            let mut l = d_low.as_device_ptr().as_raw();
            let mut c = d_close.as_device_ptr().as_raw();
            let mut a = d_atr.as_device_ptr().as_raw();
            let mut hm = d_hma.as_device_ptr().as_raw();
            let mut lm = d_lma.as_device_ptr().as_raw();
            let mut rh = d_rhi.as_device_ptr().as_raw();
            let mut rl = d_rlo.as_device_ptr().as_raw();
            let mut w = d_warms.as_device_ptr().as_raw();
            let mut ch = channel_deviation as f32;
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut oht = d_ht.as_device_ptr().as_raw();
            let mut otr = d_tr.as_device_ptr().as_raw();
            let mut oah = d_ah.as_device_ptr().as_raw();
            let mut oal = d_al.as_device_ptr().as_raw();
            let mut obs = d_bs.as_device_ptr().as_raw();
            let mut oss = d_ss.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 18] = [
                &mut h as *mut _ as *mut c_void,
                &mut l as *mut _ as *mut c_void,
                &mut c as *mut _ as *mut c_void,
                &mut a as *mut _ as *mut c_void,
                &mut hm as *mut _ as *mut c_void,
                &mut lm as *mut _ as *mut c_void,
                &mut rh as *mut _ as *mut c_void,
                &mut rl as *mut _ as *mut c_void,
                &mut w as *mut _ as *mut c_void,
                &mut ch as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut oht as *mut _ as *mut c_void,
                &mut otr as *mut _ as *mut c_void,
                &mut oah as *mut _ as *mut c_void,
                &mut oal as *mut _ as *mut c_void,
                &mut obs as *mut _ as *mut c_void,
                &mut oss as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(CudaHalftrendError::from)?;
        }

        self.stream
            .synchronize()
            .map_err(CudaHalftrendError::from)?;
        Ok(CudaHalftrendMany {
            halftrend: DeviceArrayF32 {
                buf: d_ht,
                rows,
                cols,
            },
            trend: DeviceArrayF32 {
                buf: d_tr,
                rows,
                cols,
            },
            atr_high: DeviceArrayF32 {
                buf: d_ah,
                rows,
                cols,
            },
            atr_low: DeviceArrayF32 {
                buf: d_al,
                rows,
                cols,
            },
            buy: DeviceArrayF32 {
                buf: d_bs,
                rows,
                cols,
            },
            sell: DeviceArrayF32 {
                buf: d_ss,
                rows,
                cols,
            },
        })
    }

    // Host-copy helpers for tests/benches
    pub fn halftrend_batch_into_host_f32(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &HalfTrendBatchRange,
        out_ht: &mut [f32],
        out_tr: &mut [f32],
        out_ah: &mut [f32],
        out_al: &mut [f32],
        out_bs: &mut [f32],
        out_ss: &mut [f32],
    ) -> Result<(usize, usize, Vec<HalfTrendParams>), CudaHalftrendError> {
        let dev = self.halftrend_batch_dev(high, low, close, sweep)?;
        let rows = dev.halftrend.rows;
        let cols = dev.halftrend.cols;
        let need = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaHalftrendError::InvalidInput("size overflow".into()))?;
        if [
            out_ht.len(),
            out_tr.len(),
            out_ah.len(),
            out_al.len(),
            out_bs.len(),
            out_ss.len(),
        ]
        .iter()
        .any(|&m| m != need)
        {
            return Err(CudaHalftrendError::InvalidInput(
                "output slice wrong length".into(),
            ));
        }

        // Copy from device
        dev.halftrend.buf.copy_to(out_ht).map_err(CudaHalftrendError::from)?;
        dev.trend.buf.copy_to(out_tr).map_err(CudaHalftrendError::from)?;
        dev.atr_high.buf.copy_to(out_ah).map_err(CudaHalftrendError::from)?;
        dev.atr_low.buf.copy_to(out_al).map_err(CudaHalftrendError::from)?;
        dev.buy.buf.copy_to(out_bs).map_err(CudaHalftrendError::from)?;
        dev.sell.buf.copy_to(out_ss).map_err(CudaHalftrendError::from)?;

        // If the last batch used the time-major kernel, transpose back to row-major for host slices
        let used_time_major = matches!(self.last_batch, Some(BatchKernelSelected::TimeMajor { .. }));
        if used_time_major {
            // Input buffers are laid out time-major [n, rows] on device; we copied raw order.
            // Transpose to row-major [rows, n] for the host outputs.
            let (n, r) = (cols, rows);
            let mut tmp = vec![0f32; need];
            // halftrend
            tmp.copy_from_slice(out_ht);
            for row in 0..r { for t in 0..n { out_ht[row*n + t] = tmp[t*r + row]; } }
            // trend
            tmp.copy_from_slice(out_tr);
            for row in 0..r { for t in 0..n { out_tr[row*n + t] = tmp[t*r + row]; } }
            // atr_high
            tmp.copy_from_slice(out_ah);
            for row in 0..r { for t in 0..n { out_ah[row*n + t] = tmp[t*r + row]; } }
            // atr_low
            tmp.copy_from_slice(out_al);
            for row in 0..r { for t in 0..n { out_al[row*n + t] = tmp[t*r + row]; } }
            // buy
            tmp.copy_from_slice(out_bs);
            for row in 0..r { for t in 0..n { out_bs[row*n + t] = tmp[t*r + row]; } }
            // sell
            tmp.copy_from_slice(out_ss);
            for row in 0..r { for t in 0..n { out_ss[row*n + t] = tmp[t*r + row]; } }
        }
        Ok((rows, cols, dev.combos))
    }
}

// ---------- Benches ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const LEN_1M: usize = 1_000_000;
    const COLS_256: usize = 256;
    const ROWS_8K: usize = 8_192;

    fn synth_hlc_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i];
            if v.is_nan() {
                continue;
            }
            let x = i as f32 * 0.0025;
            let off = (0.002 * x.sin()).abs() + 0.15;
            high[i] = v + off;
            low[i] = v - off;
            let v = close[i];
            if v.is_nan() {
                continue;
            }
            let x = i as f32 * 0.0025;
            let off = (0.002 * x.sin()).abs() + 0.15;
            high[i] = v + off;
            low[i] = v - off;
        }
        (high, low)
    }

    struct BatchState {
        cuda: CudaHalftrend,
        high: Vec<f32>,
        low: Vec<f32>,
        close: Vec<f32>,
        sweep: HalfTrendBatchRange,
    }
    impl CudaBenchState for BatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .halftrend_batch_dev(&self.high, &self.low, &self.close, &self.sweep)
                .unwrap();
        }
    }

    struct ManyState {
        cuda: CudaHalftrend,
        high_tm: Vec<f32>,
        low_tm: Vec<f32>,
        close_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        amp: usize,
        ch: f64,
        atr: usize,
    }
    impl CudaBenchState for ManyState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .halftrend_many_series_one_param_time_major_dev(
                    &self.high_tm,
                    &self.low_tm,
                    &self.close_tm,
                    self.cols,
                    self.rows,
                    self.amp,
                    self.ch,
                    self.atr,
                )
                .unwrap();
        }
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        // Batch: 1M bars, 32 combos (amp 2..32 step 2, fixed ch=2.0, atr=14)
        let prep_batch = || -> Box<dyn CudaBenchState> {
            let cuda = CudaHalftrend::new(0).expect("cuda halftrend");
            let close = gen_series(LEN_1M);
            let (high, low) = synth_hlc_from_close(&close);
            let sweep = HalfTrendBatchRange {
                amplitude: (2, 32, 2),
                channel_deviation: (2.0, 2.0, 0.0),
                atr_period: (14, 14, 0),
            };
            Box::new(BatchState {
                cuda,
                high,
                low,
                close,
                sweep,
            })
        };
        let bytes_batch = || -> usize {
            // Rough: inputs 3N + helpers 5*32*N + outputs 6*32*N + headroom
            (3 * LEN_1M + 5 * 16 * LEN_1M + 6 * 16 * LEN_1M) * std::mem::size_of::<f32>()
                + 64 * 1024 * 1024
        }();

        // Many-series: 8k x 256
        let prep_many = || -> Box<dyn CudaBenchState> {
            let cuda = CudaHalftrend::new(0).expect("cuda halftrend");
            let cols = COLS_256;
            let rows = ROWS_8K;
            let close_tm = {
                let mut v = vec![f32::NAN; cols * rows];
                for s in 0..cols {
                    for t in s..rows {
                        let x = (t as f32) + (s as f32) * 0.2;
                        v[t * cols + s] = (x * 0.002).sin() + 0.0003 * x;
                    }
                }
                v
            };
            let (high_tm, low_tm) = synth_hlc_from_close(&close_tm);
            Box::new(ManyState {
                cuda,
                high_tm,
                low_tm,
                close_tm,
                cols,
                rows,
                amp: 2,
                ch: 2.0,
                atr: 14,
            })
        };
        let bytes_many =
            (3 * COLS_256 * ROWS_8K + 5 * COLS_256 * ROWS_8K + COLS_256 + 6 * COLS_256 * ROWS_8K)
                * std::mem::size_of::<f32>()
                + 64 * 1024 * 1024;

        vec![
            CudaBenchScenario::new(
                "halftrend",
                "many_series_one_param",
                "halftrend_cuda_many_series",
                "8k x 256",
                prep_many,
            )
            .with_mem_required(bytes_many),
            CudaBenchScenario::new(
                "halftrend",
                "batch",
                "halftrend_cuda_batch",
                "1m",
                prep_batch,
            )
            .with_mem_required(bytes_batch),
        ]
    }
}
