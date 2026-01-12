#![cfg(feature = "cuda")]

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
    FusedPlain { block_x: u32 },
    FusedTimeMajor { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

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
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
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

const HALF_TREND_FUSED_MAX_AMP: usize = 64;

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
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
                {
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

    pub fn context_arc(&self) -> Arc<Context> {
        self.context.clone()
    }
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    pub fn set_batch_policy(&mut self, p: BatchKernelPolicy) {
        self.batch_policy = p;
    }
    pub fn set_many_series_policy(&mut self, p: ManySeriesKernelPolicy) {
        self.many_policy = p;
    }
    pub fn batch_policy(&self) -> BatchKernelPolicy {
        self.batch_policy
    }
    pub fn many_series_policy(&self) -> ManySeriesKernelPolicy {
        self.many_policy
    }

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

    fn expand_grid(
        range: &HalfTrendBatchRange,
    ) -> Result<Vec<HalfTrendParams>, CudaHalftrendError> {
        fn axis_usize(
            (start, end, step): (usize, usize, usize),
        ) -> Result<Vec<usize>, CudaHalftrendError> {
            if step == 0 || start == end {
                return Ok(vec![start]);
            }
            if start < end {
                return Ok((start..=end).step_by(step.max(1)).collect());
            }
            let mut v = Vec::new();
            let mut x = start as isize;
            let end_i = end as isize;
            let st = (step as isize).max(1);
            while x >= end_i {
                v.push(x as usize);
                x -= st;
            }
            if v.is_empty() {
                return Err(CudaHalftrendError::InvalidInput("empty usize axis".into()));
            }
            Ok(v)
        }
        fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, CudaHalftrendError> {
            if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
                return Ok(vec![start]);
            }
            if start < end {
                let mut v = Vec::new();
                let mut x = start;
                let st = step.abs();
                while x <= end + 1e-12 {
                    v.push(x);
                    x += st;
                }
                if v.is_empty() {
                    return Err(CudaHalftrendError::InvalidInput("empty f64 axis".into()));
                }
                return Ok(v);
            }
            let mut v = Vec::new();
            let mut x = start;
            let st = step.abs();
            while x + 1e-12 >= end {
                v.push(x);
                x -= st;
            }
            if v.is_empty() {
                return Err(CudaHalftrendError::InvalidInput("empty f64 axis".into()));
            }
            Ok(v)
        }
        let amps = axis_usize(range.amplitude)?;
        let cds = axis_f64(range.channel_deviation)?;
        let atrs = axis_usize(range.atr_period)?;
        let cap = amps
            .len()
            .checked_mul(cds.len())
            .and_then(|x| x.checked_mul(atrs.len()))
            .ok_or_else(|| CudaHalftrendError::InvalidInput("combination overflow".into()))?;
        let mut out = Vec::with_capacity(cap);
        for &a in &amps {
            for &c in &cds {
                for &p in &atrs {
                    out.push(HalfTrendParams {
                        amplitude: Some(a),
                        channel_deviation: Some(c),
                        atr_period: Some(p),
                    });
                }
            }
        }
        Ok(out)
    }

    #[inline]
    fn validate_launch(
        &self,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    ) -> Result<(), CudaHalftrendError> {
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
        if threads > max_threads
            || bx > max_bx
            || by > max_by
            || bz > max_bz
            || gx > max_gx
            || gy > max_gy
            || gz > max_gz
        {
            return Err(CudaHalftrendError::LaunchConfigTooLarge {
                gx,
                gy,
                gz,
                bx,
                by,
                bz,
            });
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
            return Err(CudaHalftrendError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        let rows = combos.len();

        let max_amp = combos
            .iter()
            .map(|p| p.amplitude.unwrap_or(0))
            .max()
            .unwrap_or(0);
        let use_fused = matches!(self.batch_policy, BatchKernelPolicy::Auto)
            && max_amp <= HALF_TREND_FUSED_MAX_AMP;

        let use_time_major = match self.batch_policy {
            BatchKernelPolicy::Auto => (rows >= 16) && (n >= 8192),
            BatchKernelPolicy::Plain { .. } => false,
        };

        let helpers = if use_fused { 0usize } else { 5usize };
        let f32_elems = (3usize
            .checked_mul(n)
            .and_then(|x| x.checked_add(helpers.checked_mul(rows).and_then(|y| y.checked_mul(n))?))
            .and_then(|x| x.checked_add(6usize.checked_mul(rows).and_then(|y| y.checked_mul(n))?))
            .ok_or_else(|| CudaHalftrendError::InvalidInput("size overflow".into()))?)
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| CudaHalftrendError::InvalidInput("size overflow".into()))?;
        let param_bytes = if use_fused {
            rows.checked_mul(2 * std::mem::size_of::<i32>() + std::mem::size_of::<f32>())
        } else {
            rows.checked_mul(std::mem::size_of::<i32>() + std::mem::size_of::<f32>())
        }
        .ok_or_else(|| CudaHalftrendError::InvalidInput("size overflow".into()))?;
        let req_bytes = f32_elems
            .checked_add(param_bytes)
            .and_then(|x| x.checked_add(64 * 1024 * 1024))
            .ok_or_else(|| CudaHalftrendError::InvalidInput("size overflow".into()))?;
        if let Ok((free, _)) = mem_get_info() {
            if !Self::will_fit(req_bytes, 0) {
                return Err(CudaHalftrendError::OutOfMemory {
                    required: req_bytes,
                    free,
                    headroom: 0,
                });
            }
        }

        if use_fused {
            let mut amps = Vec::<i32>::with_capacity(rows);
            let mut atr_periods = Vec::<i32>::with_capacity(rows);
            let mut chdevs = Vec::<f32>::with_capacity(rows);
            for prm in &combos {
                amps.push(prm.amplitude.unwrap_or(2) as i32);
                atr_periods.push(prm.atr_period.unwrap_or(14) as i32);
                chdevs.push(prm.channel_deviation.unwrap_or(2.0) as f32);
            }

            let d_high = unsafe { DeviceBuffer::from_slice_async(&high[..n], &self.stream) }
                .map_err(CudaHalftrendError::from)?;
            let d_low = unsafe { DeviceBuffer::from_slice_async(&low[..n], &self.stream) }
                .map_err(CudaHalftrendError::from)?;
            let d_close = unsafe { DeviceBuffer::from_slice_async(&close[..n], &self.stream) }
                .map_err(CudaHalftrendError::from)?;
            let d_amps = unsafe { DeviceBuffer::from_slice_async(&amps, &self.stream) }
                .map_err(CudaHalftrendError::from)?;
            let d_atr_periods =
                unsafe { DeviceBuffer::from_slice_async(&atr_periods, &self.stream) }
                    .map_err(CudaHalftrendError::from)?;
            let d_chdevs = unsafe { DeviceBuffer::from_slice_async(&chdevs, &self.stream) }
                .map_err(CudaHalftrendError::from)?;

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

            let (kernel, selected) = if use_time_major {
                (
                    "halftrend_batch_fused_time_major_f32",
                    BatchKernelSelected::FusedTimeMajor { block_x: 256 },
                )
            } else {
                (
                    "halftrend_batch_fused_f32",
                    BatchKernelSelected::FusedPlain { block_x: 256 },
                )
            };
            if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") && !self.debug_batch_logged
            {
                eprintln!(
                    "[halftrend] batch kernel (fused): kernel={} block_x={} rows={} len={} first_valid={}",
                    kernel, 256u32, rows, n, first
                );
                unsafe {
                    (*(self as *const _ as *mut CudaHalftrend)).debug_batch_logged = true;
                }
            }
            unsafe {
                (*(self as *const _ as *mut CudaHalftrend)).last_batch = Some(selected);
            }

            let func = self
                .module
                .get_function(kernel)
                .map_err(|_| CudaHalftrendError::MissingKernelSymbol { name: kernel })?;
            unsafe {
                let block_x = 256u32;
                let grid_x = (((rows as u32) + block_x - 1) / block_x).max(1);
                self.validate_launch(grid_x, 1, 1, block_x, 1, 1)?;
                let grid: GridSize = (grid_x, 1, 1).into();
                let block: BlockSize = (block_x, 1, 1).into();

                let mut h = d_high.as_device_ptr().as_raw();
                let mut l = d_low.as_device_ptr().as_raw();
                let mut c = d_close.as_device_ptr().as_raw();
                let mut a = d_amps.as_device_ptr().as_raw();
                let mut ap = d_atr_periods.as_device_ptr().as_raw();
                let mut cd = d_chdevs.as_device_ptr().as_raw();
                let mut first_i = first as i32;
                let mut n_i = n as i32;
                let mut r_i = rows as i32;
                let mut oht = d_ht.as_device_ptr().as_raw();
                let mut otr = d_tr.as_device_ptr().as_raw();
                let mut oah = d_ah.as_device_ptr().as_raw();
                let mut oal = d_al.as_device_ptr().as_raw();
                let mut obs = d_bs.as_device_ptr().as_raw();
                let mut oss = d_ss.as_device_ptr().as_raw();
                let mut args: [*mut c_void; 15] = [
                    &mut h as *mut _ as *mut c_void,
                    &mut l as *mut _ as *mut c_void,
                    &mut c as *mut _ as *mut c_void,
                    &mut a as *mut _ as *mut c_void,
                    &mut ap as *mut _ as *mut c_void,
                    &mut cd as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut n_i as *mut _ as *mut c_void,
                    &mut r_i as *mut _ as *mut c_void,
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

            return Ok(CudaHalftrendBatch {
                halftrend: DeviceArrayF32 {
                    buf: d_ht,
                    rows,
                    cols: n,
                },
                trend: DeviceArrayF32 {
                    buf: d_tr,
                    rows,
                    cols: n,
                },
                atr_high: DeviceArrayF32 {
                    buf: d_ah,
                    rows,
                    cols: n,
                },
                atr_low: DeviceArrayF32 {
                    buf: d_al,
                    rows,
                    cols: n,
                },
                buy: DeviceArrayF32 {
                    buf: d_bs,
                    rows,
                    cols: n,
                },
                sell: DeviceArrayF32 {
                    buf: d_ss,
                    rows,
                    cols: n,
                },
                combos,
            });
        }

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

        use cust::memory::LockedBuffer;
        let mut warms = vec![0i32; rows];
        let mut chdevs = vec![0f32; rows];

        for (row, prm) in combos.iter().enumerate() {
            let a = prm.amplitude.unwrap();
            let p = prm.atr_period.unwrap();
            let ch = prm.channel_deviation.unwrap_or(2.0) as f32;
            chdevs[row] = ch;
            let warm = first + a.max(p) - 1;
            warms[row] = warm.min(n) as i32;
        }

        let d_high = unsafe { DeviceBuffer::from_slice_async(&high[..n], &self.stream) }
            .map_err(CudaHalftrendError::from)?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(&low[..n], &self.stream) }
            .map_err(CudaHalftrendError::from)?;
        let d_close = unsafe { DeviceBuffer::from_slice_async(&close[..n], &self.stream) }
            .map_err(CudaHalftrendError::from)?;
        let d_warms = DeviceBuffer::from_slice(&warms).map_err(CudaHalftrendError::from)?;
        let d_chdevs = DeviceBuffer::from_slice(&chdevs).map_err(CudaHalftrendError::from)?;

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

            let func = self
                .module
                .get_function("halftrend_batch_f32")
                .map_err(|_| CudaHalftrendError::MissingKernelSymbol {
                    name: "halftrend_batch_f32",
                })?;
            let block_x = match self.batch_policy {
                BatchKernelPolicy::Auto => 256,
                BatchKernelPolicy::Plain { block_x } => block_x.max(32),
            };
            if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") && !self.debug_batch_logged
            {
                eprintln!("[halftrend] batch kernel (row-major): block_x={} rows={} len={} first_valid={}",
                    block_x, rows, n, first);
                unsafe {
                    (*(self as *const _ as *mut CudaHalftrend)).debug_batch_logged = true;
                }
            }
            unsafe {
                (*(self as *const _ as *mut CudaHalftrend)).last_batch =
                    Some(BatchKernelSelected::Plain { block_x });
            }

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
                self.stream
                    .launch(&func, grid, block, 0, &mut args)
                    .map_err(CudaHalftrendError::from)?;
            }

            self.stream
                .synchronize()
                .map_err(CudaHalftrendError::from)?;
        } else {
            let len_tm = rows
                .checked_mul(n)
                .ok_or_else(|| CudaHalftrendError::InvalidInput("size overflow".into()))?;
            let mut atr_tm = unsafe { LockedBuffer::<f32>::uninitialized(len_tm) }
                .map_err(CudaHalftrendError::from)?;
            let mut hma_tm = unsafe { LockedBuffer::<f32>::uninitialized(len_tm) }
                .map_err(CudaHalftrendError::from)?;
            let mut lma_tm = unsafe { LockedBuffer::<f32>::uninitialized(len_tm) }
                .map_err(CudaHalftrendError::from)?;
            let mut rhi_tm = unsafe { LockedBuffer::<f32>::uninitialized(len_tm) }
                .map_err(CudaHalftrendError::from)?;
            let mut rlo_tm = unsafe { LockedBuffer::<f32>::uninitialized(len_tm) }
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

            let func = self
                .module
                .get_function("halftrend_batch_time_major_f32")
                .map_err(|_| CudaHalftrendError::MissingKernelSymbol {
                    name: "halftrend_batch_time_major_f32",
                })?;
            let block_x = 256u32;
            if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") && !self.debug_batch_logged
            {
                eprintln!("[halftrend] batch kernel (time-major): block_x={} rows={} len={} first_valid={}",
                    block_x, rows, n, first);
                unsafe {
                    (*(self as *const _ as *mut CudaHalftrend)).debug_batch_logged = true;
                }
            }
            unsafe {
                (*(self as *const _ as *mut CudaHalftrend)).last_batch =
                    Some(BatchKernelSelected::TimeMajor { block_x });
            }

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
                self.stream
                    .launch(&func, grid, block, 0, &mut args)
                    .map_err(CudaHalftrendError::from)?;
            }

            self.stream
                .synchronize()
                .map_err(CudaHalftrendError::from)?;
        }

        Ok(CudaHalftrendBatch {
            halftrend: DeviceArrayF32 {
                buf: d_ht,
                rows,
                cols: n,
            },
            trend: DeviceArrayF32 {
                buf: d_tr,
                rows,
                cols: n,
            },
            atr_high: DeviceArrayF32 {
                buf: d_ah,
                rows,
                cols: n,
            },
            atr_low: DeviceArrayF32 {
                buf: d_al,
                rows,
                cols: n,
            },
            buy: DeviceArrayF32 {
                buf: d_bs,
                rows,
                cols: n,
            },
            sell: DeviceArrayF32 {
                buf: d_ss,
                rows,
                cols: n,
            },
            combos,
        })
    }

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

        let len_tm = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaHalftrendError::InvalidInput("size overflow".into()))?;
        let mut atr_tm = unsafe { LockedBuffer::<f32>::uninitialized(len_tm) }
            .map_err(CudaHalftrendError::from)?;
        let mut hma_tm = unsafe { LockedBuffer::<f32>::uninitialized(len_tm) }
            .map_err(CudaHalftrendError::from)?;
        let mut lma_tm = unsafe { LockedBuffer::<f32>::uninitialized(len_tm) }
            .map_err(CudaHalftrendError::from)?;
        let mut rhi_tm = unsafe { LockedBuffer::<f32>::uninitialized(len_tm) }
            .map_err(CudaHalftrendError::from)?;
        let mut rlo_tm = unsafe { LockedBuffer::<f32>::uninitialized(len_tm) }
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

        let req = (3 * cols * rows + 5 * cols * rows + cols + 6 * cols * rows)
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|x| x.checked_add(64 * 1024 * 1024))
            .ok_or_else(|| CudaHalftrendError::InvalidInput("size overflow".into()))?;
        if let Ok((free, _)) = mem_get_info() {
            if !Self::will_fit(req, 0) {
                return Err(CudaHalftrendError::OutOfMemory {
                    required: req,
                    free,
                    headroom: 0,
                });
            }
        }

        let d_high = unsafe { DeviceBuffer::from_slice_async(high_tm, &self.stream) }
            .map_err(CudaHalftrendError::from)?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(low_tm, &self.stream) }
            .map_err(CudaHalftrendError::from)?;
        let d_close = unsafe { DeviceBuffer::from_slice_async(close_tm, &self.stream) }
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
        let d_warms = DeviceBuffer::from_slice(&warms).map_err(CudaHalftrendError::from)?;

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

        let func = self
            .module
            .get_function("halftrend_many_series_one_param_time_major_f32")
            .map_err(|_| CudaHalftrendError::MissingKernelSymbol {
                name: "halftrend_many_series_one_param_time_major_f32",
            })?;
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

        dev.halftrend
            .buf
            .copy_to(out_ht)
            .map_err(CudaHalftrendError::from)?;
        dev.trend
            .buf
            .copy_to(out_tr)
            .map_err(CudaHalftrendError::from)?;
        dev.atr_high
            .buf
            .copy_to(out_ah)
            .map_err(CudaHalftrendError::from)?;
        dev.atr_low
            .buf
            .copy_to(out_al)
            .map_err(CudaHalftrendError::from)?;
        dev.buy
            .buf
            .copy_to(out_bs)
            .map_err(CudaHalftrendError::from)?;
        dev.sell
            .buf
            .copy_to(out_ss)
            .map_err(CudaHalftrendError::from)?;

        let used_time_major = matches!(
            self.last_batch,
            Some(
                BatchKernelSelected::TimeMajor { .. } | BatchKernelSelected::FusedTimeMajor { .. }
            )
        );
        if used_time_major {
            let (n, r) = (cols, rows);
            let mut tmp = vec![0f32; need];

            tmp.copy_from_slice(out_ht);
            for row in 0..r {
                for t in 0..n {
                    out_ht[row * n + t] = tmp[t * r + row];
                }
            }

            tmp.copy_from_slice(out_tr);
            for row in 0..r {
                for t in 0..n {
                    out_tr[row * n + t] = tmp[t * r + row];
                }
            }

            tmp.copy_from_slice(out_ah);
            for row in 0..r {
                for t in 0..n {
                    out_ah[row * n + t] = tmp[t * r + row];
                }
            }

            tmp.copy_from_slice(out_al);
            for row in 0..r {
                for t in 0..n {
                    out_al[row * n + t] = tmp[t * r + row];
                }
            }

            tmp.copy_from_slice(out_bs);
            for row in 0..r {
                for t in 0..n {
                    out_bs[row * n + t] = tmp[t * r + row];
                }
            }

            tmp.copy_from_slice(out_ss);
            for row in 0..r {
                for t in 0..n {
                    out_ss[row * n + t] = tmp[t * r + row];
                }
            }
        }
        Ok((rows, cols, dev.combos))
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use cust::memory::DeviceBuffer;

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
        }
        (high, low)
    }

    struct BatchDevInplaceState {
        cuda: CudaHalftrend,
        d_high: DeviceBuffer<f32>,
        d_low: DeviceBuffer<f32>,
        d_close: DeviceBuffer<f32>,
        d_amp: DeviceBuffer<i32>,
        d_atr: DeviceBuffer<i32>,
        d_ch: DeviceBuffer<f32>,
        first: i32,
        len: usize,
        rows: usize,
        block_x: u32,
        kernel: &'static str,
        d_ht: DeviceBuffer<f32>,
        d_tr: DeviceBuffer<f32>,
        d_ah: DeviceBuffer<f32>,
        d_al: DeviceBuffer<f32>,
        d_bs: DeviceBuffer<f32>,
        d_ss: DeviceBuffer<f32>,
    }
    impl CudaBenchState for BatchDevInplaceState {
        fn launch(&mut self) {
            let func = self
                .cuda
                .module
                .get_function(self.kernel)
                .expect(self.kernel);

            let rows_u32 = self.rows as u32;
            let grid_x = ((rows_u32 + self.block_x - 1) / self.block_x).max(1);
            self.cuda
                .validate_launch(grid_x, 1, 1, self.block_x, 1, 1)
                .expect("halftrend validate launch");
            let grid: GridSize = (grid_x, 1, 1).into();
            let block: BlockSize = (self.block_x, 1, 1).into();

            unsafe {
                let mut h = self.d_high.as_device_ptr().as_raw();
                let mut l = self.d_low.as_device_ptr().as_raw();
                let mut c = self.d_close.as_device_ptr().as_raw();
                let mut a = self.d_amp.as_device_ptr().as_raw();
                let mut ap = self.d_atr.as_device_ptr().as_raw();
                let mut ch = self.d_ch.as_device_ptr().as_raw();
                let mut first_i = self.first as i32;
                let mut n_i = self.len as i32;
                let mut r_i = self.rows as i32;
                let mut oht = self.d_ht.as_device_ptr().as_raw();
                let mut otr = self.d_tr.as_device_ptr().as_raw();
                let mut oah = self.d_ah.as_device_ptr().as_raw();
                let mut oal = self.d_al.as_device_ptr().as_raw();
                let mut obs = self.d_bs.as_device_ptr().as_raw();
                let mut oss = self.d_ss.as_device_ptr().as_raw();
                let mut args: [*mut c_void; 15] = [
                    &mut h as *mut _ as *mut c_void,
                    &mut l as *mut _ as *mut c_void,
                    &mut c as *mut _ as *mut c_void,
                    &mut a as *mut _ as *mut c_void,
                    &mut ap as *mut _ as *mut c_void,
                    &mut ch as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut n_i as *mut _ as *mut c_void,
                    &mut r_i as *mut _ as *mut c_void,
                    &mut oht as *mut _ as *mut c_void,
                    &mut otr as *mut _ as *mut c_void,
                    &mut oah as *mut _ as *mut c_void,
                    &mut oal as *mut _ as *mut c_void,
                    &mut obs as *mut _ as *mut c_void,
                    &mut oss as *mut _ as *mut c_void,
                ];
                self.cuda
                    .stream
                    .launch(&func, grid, block, 0, &mut args)
                    .expect("halftrend launch");
            }
            self.cuda.stream.synchronize().expect("halftrend sync");
        }
    }

    struct ManyState {
        cuda: CudaHalftrend,
        d_high_tm: DeviceBuffer<f32>,
        d_low_tm: DeviceBuffer<f32>,
        d_close_tm: DeviceBuffer<f32>,
        d_atr_tm: DeviceBuffer<f32>,
        d_hma_tm: DeviceBuffer<f32>,
        d_lma_tm: DeviceBuffer<f32>,
        d_rhi_tm: DeviceBuffer<f32>,
        d_rlo_tm: DeviceBuffer<f32>,
        d_warms: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        ch: f32,
        block_x: u32,
        grid_x: u32,
        d_ht: DeviceBuffer<f32>,
        d_tr: DeviceBuffer<f32>,
        d_ah: DeviceBuffer<f32>,
        d_al: DeviceBuffer<f32>,
        d_bs: DeviceBuffer<f32>,
        d_ss: DeviceBuffer<f32>,
    }
    impl CudaBenchState for ManyState {
        fn launch(&mut self) {
            let func = self
                .cuda
                .module
                .get_function("halftrend_many_series_one_param_time_major_f32")
                .expect("halftrend_many_series_one_param_time_major_f32");
            let grid: GridSize = (self.grid_x, 1, 1).into();
            let block: BlockSize = (self.block_x, 1, 1).into();
            unsafe {
                let mut h = self.d_high_tm.as_device_ptr().as_raw();
                let mut l = self.d_low_tm.as_device_ptr().as_raw();
                let mut c = self.d_close_tm.as_device_ptr().as_raw();
                let mut a = self.d_atr_tm.as_device_ptr().as_raw();
                let mut hm = self.d_hma_tm.as_device_ptr().as_raw();
                let mut lm = self.d_lma_tm.as_device_ptr().as_raw();
                let mut rh = self.d_rhi_tm.as_device_ptr().as_raw();
                let mut rl = self.d_rlo_tm.as_device_ptr().as_raw();
                let mut w = self.d_warms.as_device_ptr().as_raw();
                let mut ch = self.ch;
                let mut cols_i = self.cols as i32;
                let mut rows_i = self.rows as i32;
                let mut oht = self.d_ht.as_device_ptr().as_raw();
                let mut otr = self.d_tr.as_device_ptr().as_raw();
                let mut oah = self.d_ah.as_device_ptr().as_raw();
                let mut oal = self.d_al.as_device_ptr().as_raw();
                let mut obs = self.d_bs.as_device_ptr().as_raw();
                let mut oss = self.d_ss.as_device_ptr().as_raw();
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
                self.cuda
                    .stream
                    .launch(&func, grid, block, 0, &mut args)
                    .expect("halftrend many-series launch");
            }
            self.cuda.stream.synchronize().expect("halftrend sync");
        }
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        let bytes_batch = || -> usize {
            (3 * LEN_1M + 6 * 16 * LEN_1M) * std::mem::size_of::<f32>() + 64 * 1024 * 1024
        }();

        let prep_batch_dev_inplace = || -> Box<dyn CudaBenchState> {
            let cuda = CudaHalftrend::new(0).expect("cuda halftrend");
            let close = gen_series(LEN_1M);
            let (high, low) = synth_hlc_from_close(&close);
            let sweep = HalfTrendBatchRange {
                amplitude: (2, 32, 2),
                channel_deviation: (2.0, 2.0, 0.0),
                atr_period: (14, 14, 0),
            };
            let combos = CudaHalftrend::expand_grid(&sweep).expect("halftrend expand grid");
            let rows = combos.len();
            let first = CudaHalftrend::first_valid_ohlc_f32(&high, &low, &close)
                .expect("halftrend first_valid") as i32;

            let mut amp = Vec::<i32>::with_capacity(rows);
            let mut atr = Vec::<i32>::with_capacity(rows);
            let mut ch = Vec::<f32>::with_capacity(rows);
            for p in &combos {
                amp.push(p.amplitude.unwrap_or(2) as i32);
                atr.push(p.atr_period.unwrap_or(14) as i32);
                ch.push(p.channel_deviation.unwrap_or(2.0) as f32);
            }

            let d_high =
                unsafe { DeviceBuffer::from_slice_async(&high, &cuda.stream) }.expect("d_high");
            let d_low =
                unsafe { DeviceBuffer::from_slice_async(&low, &cuda.stream) }.expect("d_low");
            let d_close =
                unsafe { DeviceBuffer::from_slice_async(&close, &cuda.stream) }.expect("d_close");
            let d_amp =
                unsafe { DeviceBuffer::from_slice_async(&amp, &cuda.stream) }.expect("d_amp");
            let d_atr =
                unsafe { DeviceBuffer::from_slice_async(&atr, &cuda.stream) }.expect("d_atr");
            let d_ch = unsafe { DeviceBuffer::from_slice_async(&ch, &cuda.stream) }.expect("d_ch");

            let elems = rows.checked_mul(LEN_1M).expect("halftrend rows*n overflow");
            let d_ht: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized_async(elems, &cuda.stream) }.expect("d_ht");
            let d_tr: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized_async(elems, &cuda.stream) }.expect("d_tr");
            let d_ah: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized_async(elems, &cuda.stream) }.expect("d_ah");
            let d_al: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized_async(elems, &cuda.stream) }.expect("d_al");
            let d_bs: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized_async(elems, &cuda.stream) }.expect("d_bs");
            let d_ss: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized_async(elems, &cuda.stream) }.expect("d_ss");
            cuda.stream.synchronize().expect("halftrend sync");

            let use_time_major = rows >= 16 && LEN_1M >= 8192;
            let kernel = if use_time_major {
                "halftrend_batch_fused_time_major_f32"
            } else {
                "halftrend_batch_fused_f32"
            };

            Box::new(BatchDevInplaceState {
                cuda,
                d_high,
                d_low,
                d_close,
                d_amp,
                d_atr,
                d_ch,
                first,
                len: LEN_1M,
                rows,
                block_x: 256,
                kernel,
                d_ht,
                d_tr,
                d_ah,
                d_al,
                d_bs,
                d_ss,
            })
        };

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

            let (amp, atr_period, ch) = (2usize, 14usize, 2.0f32);
            let mut firsts = vec![rows as i32; cols];
            for s in 0..cols {
                for t in 0..rows {
                    let idx = t * cols + s;
                    if high_tm[idx].is_finite()
                        && low_tm[idx].is_finite()
                        && close_tm[idx].is_finite()
                    {
                        firsts[s] = t as i32;
                        break;
                    }
                }
            }
            let warm_len = amp.max(atr_period);
            let mut warms: Vec<i32> = Vec::with_capacity(cols);
            for s in 0..cols {
                let fv = firsts[s].max(0) as usize;
                warms.push((fv + warm_len - 1).min(rows) as i32);
            }

            let elems = cols * rows;
            let mut atr_tm = vec![f32::NAN; elems];
            let mut hma_tm = vec![f32::NAN; elems];
            let mut lma_tm = vec![f32::NAN; elems];
            let mut rhi_tm = vec![f32::NAN; elems];
            let mut rlo_tm = vec![f32::NAN; elems];
            for idx in 0..elems {
                let h = high_tm[idx];
                let l = low_tm[idx];
                if h.is_finite() && l.is_finite() {
                    atr_tm[idx] = (h - l).abs();
                }
                hma_tm[idx] = h;
                lma_tm[idx] = l;
                rhi_tm[idx] = h;
                rlo_tm[idx] = l;
            }

            let d_high_tm = DeviceBuffer::from_slice(&high_tm).expect("d_high_tm");
            let d_low_tm = DeviceBuffer::from_slice(&low_tm).expect("d_low_tm");
            let d_close_tm = DeviceBuffer::from_slice(&close_tm).expect("d_close_tm");
            let d_atr_tm = DeviceBuffer::from_slice(&atr_tm).expect("d_atr_tm");
            let d_hma_tm = DeviceBuffer::from_slice(&hma_tm).expect("d_hma_tm");
            let d_lma_tm = DeviceBuffer::from_slice(&lma_tm).expect("d_lma_tm");
            let d_rhi_tm = DeviceBuffer::from_slice(&rhi_tm).expect("d_rhi_tm");
            let d_rlo_tm = DeviceBuffer::from_slice(&rlo_tm).expect("d_rlo_tm");
            let d_warms = DeviceBuffer::from_slice(&warms).expect("d_warms");

            let d_ht: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_ht");
            let d_tr: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_tr");
            let d_ah: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_ah");
            let d_al: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_al");
            let d_bs: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_bs");
            let d_ss: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_ss");

            let block_x = match cuda.many_policy {
                ManySeriesKernelPolicy::Auto => 256,
                ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32),
            };
            let grid_x = (((cols as u32) + block_x - 1) / block_x).max(1);
            cuda.stream
                .synchronize()
                .expect("halftrend sync after prep");

            Box::new(ManyState {
                cuda,
                d_high_tm,
                d_low_tm,
                d_close_tm,
                d_atr_tm,
                d_hma_tm,
                d_lma_tm,
                d_rhi_tm,
                d_rlo_tm,
                d_warms,
                cols,
                rows,
                ch,
                block_x,
                grid_x,
                d_ht,
                d_tr,
                d_ah,
                d_al,
                d_bs,
                d_ss,
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
                "halftrend_cuda_batch_dev_inplace",
                "1m",
                prep_batch_dev_inplace,
            )
            .with_mem_required(bytes_batch)
            .with_sample_size(10),
        ]
    }
}
