//! CUDA wrapper for Awesome Oscillator (AO): AO = SMA(short) - SMA(long) over HL2.
//!
//! Pattern classification: Prefix-sum/rational for batch (one-series × many-params)
//! and rolling-sum recurrence for many-series × one-param (time-major).
//!
//! Semantics:
//! - Warmup/writes: NaN until `warm = first_valid + long - 1`, identical to scalar AO.
//! - NaN inputs: we honor `first_valid` and do no mid-stream masking (matches CPU path).
//! - Accumulation: host builds a double-single (DS) prefix in f32x2; device stays FP64-free.

#![cfg(feature = "cuda")]

use crate::indicators::ao::{AoBatchRange, AoParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::mem_get_info;
use cust::memory::DeviceBuffer;
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaAoError {
    #[error("CUDA: {0}")]
    Cuda(#[from] cust::error::CudaError),
    #[error("Out of memory: required={required} bytes, free={free} bytes, headroom={headroom} bytes")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("Missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("Launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("Device mismatch: buffer device {buf}, current {current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("Not implemented")]
    NotImplemented,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaAoPolicy {
    pub batch_block_x: Option<u32>,
    pub many_block_x: Option<u32>,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaAo {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaAoPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

/// VRAM-backed array handle for AO with context guard and device id
pub struct DeviceArrayF32Ao {
    pub buf: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
    pub ctx: Arc<Context>,
    pub device_id: u32,
}
impl DeviceArrayF32Ao {
    #[inline]
    pub fn device_ptr(&self) -> u64 { self.buf.as_device_ptr().as_raw() as u64 }
    #[inline]
    pub fn len(&self) -> usize { self.rows * self.cols }
}

// Host-side POD matching CUDA float2 ABI (8-byte aligned)
#[repr(C, align(8))]
#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct Float2 { pub x: f32, pub y: f32 }
unsafe impl cust::memory::DeviceCopy for Float2 {}

#[inline(always)]
fn build_prefix_ds(hl2: &[f32], first_valid: usize) -> Vec<Float2> {
    // Exclusive prefix: out[0] = {0,0}
    let mut out = Vec::with_capacity(hl2.len() + 1);
    out.push(Float2 { x: 0.0, y: 0.0 });
    #[inline(always)]
    fn two_sum(a: f32, b: f32) -> (f32, f32) {
        let s = a + b;
        let bb = s - a;
        let e = (a - (s - bb)) + (b - bb);
        (s, e)
    }
    let (mut hi, mut lo) = (0.0f32, 0.0f32);
    for (i, &vv) in hl2.iter().enumerate() {
        let v = if i >= first_valid && !vv.is_nan() { vv } else { 0.0 };
        let (s, e1) = two_sum(hi, v);
        let e2 = e1 + lo;
        let (hi2, lo2) = two_sum(s, e2);
        hi = hi2; lo = lo2;
        out.push(Float2 { x: hi, y: lo });
    }
    out
}

impl CudaAo {
    #[inline]
    fn will_fit(&self, required: usize, headroom: usize) -> Result<(), CudaAoError> {
        match mem_get_info() {
            Ok((free, _total)) => {
                if required.saturating_add(headroom) > free {
                    return Err(CudaAoError::OutOfMemory { required, free, headroom });
                }
                Ok(())
            }
            Err(e) => Err(CudaAoError::Cuda(e)),
        }
    }
    pub fn new(device_id: usize) -> Result<Self, CudaAoError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/ao_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaAoPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    #[inline]
    pub fn set_policy(&mut self, p: CudaAoPolicy) {
        self.policy = p;
    }

    // ---------- Batch (one series × many params) ----------
    pub fn ao_batch_dev(
        &self,
        hl2: &[f32],
        sweep: &AoBatchRange,
    ) -> Result<DeviceArrayF32Ao, CudaAoError> {
        let len = hl2.len();
        if len == 0 { return Err(CudaAoError::InvalidInput("empty series".into())); }

        let first_valid = hl2
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaAoError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid_checked_cuda(sweep)?;

        // Validate params and warmup
        let mut shorts: Vec<i32> = Vec::with_capacity(combos.len());
        let mut longs: Vec<i32> = Vec::with_capacity(combos.len());
        for prm in &combos {
            let s = prm.short_period.unwrap_or(5) as i32;
            let l = prm.long_period.unwrap_or(34) as i32;
            if s <= 0 || l <= 0 || s >= l {
                return Err(CudaAoError::InvalidInput(format!(
                    "invalid params: short={} long={}",
                    s, l
                )));
            }
            if len - first_valid < (l as usize) {
                return Err(CudaAoError::InvalidInput(format!(
                    "not enough valid data for long={}, tail={} (first_valid={})",
                    l,
                    len - first_valid,
                    first_valid
                )));
            }
            shorts.push(s);
            longs.push(l);
        }

        // Build DS (float2) exclusive prefix on host (FP64-free on device)
        let prefix: Vec<Float2> = build_prefix_ds(hl2, first_valid);

        // VRAM estimate and headroom (checked arithmetic)
        let rows = combos.len();
        let len_plus_one = len
            .checked_add(1)
            .ok_or_else(|| CudaAoError::InvalidInput("len+1 overflow".into()))?;
        let bytes_prefix = len_plus_one
            .checked_mul(std::mem::size_of::<Float2>())
            .ok_or_else(|| CudaAoError::InvalidInput("prefix size overflow".into()))?;
        let bytes_periods = rows
            .checked_mul(2)
            .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
            .ok_or_else(|| CudaAoError::InvalidInput("periods size overflow".into()))?;
        let elems_out = rows
            .checked_mul(len)
            .ok_or_else(|| CudaAoError::InvalidInput("rows*len overflow".into()))?;
        let bytes_out_total = elems_out
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaAoError::InvalidInput("output size overflow".into()))?;
        let required = bytes_prefix
            .checked_add(bytes_periods)
            .and_then(|v| v.checked_add(bytes_out_total))
            .ok_or_else(|| CudaAoError::InvalidInput("total VRAM size overflow".into()))?;
        let headroom = 64usize * 1024 * 1024;
        self.will_fit(required, headroom)?;

        // Device buffers
        let d_prefix: DeviceBuffer<Float2> = DeviceBuffer::from_slice(&prefix)?;

        // Allocate one device output for all rows (avoid host bounce)
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems_out) }?;

        if rows <= 65_535 {
            let d_shorts_full = DeviceBuffer::from_slice(&shorts)?;
            let d_longs_full  = DeviceBuffer::from_slice(&longs)?;
            unsafe { (*(self as *const _ as *mut CudaAo)).last_batch = Some(BatchKernelSelected::Plain { block_x: 256 }); }
            self.launch_batch_into(&d_prefix, len, first_valid, &d_shorts_full, &d_longs_full, rows, &d_out, 0)?;
            self.maybe_log_batch_debug();
            return Ok(DeviceArrayF32Ao { buf: d_out, rows, cols: len, ctx: self._context.clone(), device_id: self.device_id });
        }

        // Chunk grid.y to ≤ 65_535, writing each chunk directly into d_out
        unsafe { (*(self as *const _ as *mut CudaAo)).last_batch = Some(BatchKernelSelected::Plain { block_x: 256 }); }
        self.maybe_log_batch_debug();
        let max_grid = 65_535usize;
        let mut start = 0usize;
        while start < rows {
            let chunk = (rows - start).min(max_grid);
            let d_shorts = DeviceBuffer::from_slice(&shorts[start..start + chunk])?;
            let d_longs = DeviceBuffer::from_slice(&longs[start..start + chunk])?;
            self.launch_batch_into(&d_prefix, len, first_valid, &d_shorts, &d_longs, chunk, &d_out, start)?;
            start += chunk;
        }
        Ok(DeviceArrayF32Ao { buf: d_out, rows, cols: len, ctx: self._context.clone(), device_id: self.device_id })
    }

    fn launch_batch_into(
        &self,
        d_prefix: &DeviceBuffer<Float2>,
        len: usize,
        first_valid: usize,
        d_shorts: &DeviceBuffer<i32>,
        d_longs: &DeviceBuffer<i32>,
        n_combos: usize,
        d_out: &DeviceBuffer<f32>,
        combo_offset: usize,
    ) -> Result<(), CudaAoError> {
        let func = self
            .module
            .get_function("ao_batch_f32")
            .map_err(|_| CudaAoError::MissingKernelSymbol { name: "ao_batch_f32" })?;
        let block_x = self.policy.batch_block_x.unwrap_or(256);
        let grid_x = ((len as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), n_combos as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        // Optional launch bounds validation
        let dev = Device::get_device(self.device_id).map_err(CudaAoError::Cuda)?;
        let max_bx = dev.get_attribute(cust::device::DeviceAttribute::MaxBlockDimX).map_err(CudaAoError::Cuda)? as u32;
        if block_x > max_bx || n_combos as u32 > 65_535 {
            return Err(CudaAoError::LaunchConfigTooLarge { gx: grid_x.max(1), gy: n_combos as u32, gz: 1, bx: block_x, by: 1, bz: 1 });
        }
        unsafe {
            let mut prefix_ptr = d_prefix.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut first_i = first_valid as i32;
            let mut shorts_ptr = d_shorts.as_device_ptr().as_raw();
            let mut longs_ptr = d_longs.as_device_ptr().as_raw();
            let mut combos_i = n_combos as i32;
            let elem_offset = combo_offset
                .checked_mul(len)
                .ok_or_else(|| CudaAoError::InvalidInput("combo_offset*len overflow".into()))?;
            let byte_off = elem_offset
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| CudaAoError::InvalidInput("byte offset overflow".into()))?
                as u64;
            let mut out_ptr = d_out.as_device_ptr().as_raw() + byte_off;
            let args: &mut [*mut c_void] = &mut [
                &mut prefix_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut shorts_ptr as *mut _ as *mut c_void,
                &mut longs_ptr as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        self.stream.synchronize()?;
        Ok(())
    }

    // ---------- Many-series × one-param (time-major) ----------
    pub fn ao_many_series_one_param_time_major_dev(
        &self,
        hl2_tm: &[f32],
        cols: usize,
        rows: usize,
        short: usize,
        long: usize,
    ) -> Result<DeviceArrayF32Ao, CudaAoError> {
        if cols == 0 || rows == 0 {
            return Err(CudaAoError::InvalidInput("invalid dims".into()));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaAoError::InvalidInput("rows*cols overflow".into()))?;
        if hl2_tm.len() != expected {
            return Err(CudaAoError::InvalidInput(format!(
                "time-major input length mismatch (expected {}, got {})",
                expected,
                hl2_tm.len()
            )));
        }
        if short == 0 || long == 0 || short >= long {
            return Err(CudaAoError::InvalidInput("invalid short/long".into()));
        }

        // Per-series first_valids
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let idx = t * cols + s;
                if !hl2_tm[idx].is_nan() {
                    fv = Some(t as i32);
                    break;
                }
            }
            let fv =
                fv.ok_or_else(|| CudaAoError::InvalidInput(format!("series {} all NaN", s)))?;
            if rows - (fv as usize) < long {
                return Err(CudaAoError::InvalidInput(format!(
                    "series {} insufficient data for long {} (tail={})",
                    s,
                    long,
                    rows - fv as usize
                )));
            }
            first_valids[s] = fv;
        }

        let prices_bytes = hl2_tm
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaAoError::InvalidInput("prices size overflow".into()))?;
        let first_bytes = first_valids
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaAoError::InvalidInput("first_valids size overflow".into()))?;
        let elems_out = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaAoError::InvalidInput("rows*cols overflow".into()))?;
        let out_bytes = elems_out
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaAoError::InvalidInput("output size overflow".into()))?;
        let required = prices_bytes
            .checked_add(first_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaAoError::InvalidInput("total VRAM size overflow".into()))?;
        self.will_fit(required, 64 * 1024 * 1024)?;

        let d_prices = DeviceBuffer::from_slice(hl2_tm)?;
        let d_first = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems_out) }?;
        self.launch_many_series(&d_prices, &d_first, cols, rows, short, long, &mut d_out)?;
        Ok(DeviceArrayF32Ao { buf: d_out, rows, cols, ctx: self._context.clone(), device_id: self.device_id })
    }

    fn launch_many_series(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        short: usize,
        long: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAoError> {
        let func = self.module.get_function("ao_many_series_one_param_f32").map_err(|_| CudaAoError::MissingKernelSymbol { name: "ao_many_series_one_param_f32" })?;
        let block_x = self.policy.many_block_x.unwrap_or(256);
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        let dev = Device::get_device(self.device_id).map_err(CudaAoError::Cuda)?;
        let max_bx = dev.get_attribute(cust::device::DeviceAttribute::MaxBlockDimX).map_err(CudaAoError::Cuda)? as u32;
        if block_x > max_bx { return Err(CudaAoError::LaunchConfigTooLarge { gx: grid_x.max(1), gy: 1, gz: 1, bx: block_x, by: 1, bz: 1 }); }
        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut nseries_i = cols as i32;
            let mut slen_i = rows as i32;
            let mut short_i = short as i32;
            let mut long_i = long as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut nseries_i as *mut _ as *mut c_void,
                &mut slen_i as *mut _ as *mut c_void,
                &mut short_i as *mut _ as *mut c_void,
                &mut long_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        self.stream.synchronize()?;
        Ok(())
    }

    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_scenario =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] AO batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaAo)).debug_batch_logged = true;
                }
            }
        }
    }
    fn maybe_log_many_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per_scenario =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] AO many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaAo)).debug_many_logged = true;
                }
            }
        }
    }
}

// ---- Local helpers ----
fn expand_grid_checked_cuda(r: &AoBatchRange) -> Result<Vec<AoParams>, CudaAoError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, CudaAoError> {
        if step == 0 || start == end { return Ok(vec![start]); }
        let mut out = Vec::new();
        if start < end {
            let mut v = start;
            while v <= end {
                out.push(v);
                match v.checked_add(step) {
                    Some(next) => { if next == v { break; } v = next; }
                    None => break,
                }
            }
        } else {
            let mut v = start;
            while v >= end {
                out.push(v);
                if v < end + step { break; }
                v -= step;
                if v == 0 { break; }
            }
        }
        if out.is_empty() {
            return Err(CudaAoError::InvalidInput(format!(
                "invalid range: start={}, end={}, step={}", start, end, step
            )));
        }
        Ok(out)
    }
    let shorts = axis_usize(r.short_period)?;
    let longs = axis_usize(r.long_period)?;
    let cap = shorts.len().checked_mul(longs.len()).ok_or_else(|| CudaAoError::InvalidInput("rows*cols overflow".into()))?;
    let mut out = Vec::with_capacity(cap);
    for &s in &shorts {
        for &l in &longs {
            if s > 0 && l > 0 && s < l {
                out.push(AoParams { short_period: Some(s), long_period: Some(l) });
            }
        }
    }
    if out.is_empty() {
        return Err(CudaAoError::InvalidInput("no parameter combos".into()));
    }
    Ok(out)
}

// ---------- Benches ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250; // ~250 (short,long) pairs

    fn bytes_one_series_many_params() -> usize {
        let prefix_bytes = (ONE_SERIES_LEN + 1) * std::mem::size_of::<super::Float2>();
        let periods_bytes = PARAM_SWEEP * 2 * std::mem::size_of::<i32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        prefix_bytes + periods_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct AoBatchDeviceState {
        cuda: CudaAo,
        d_prefix: DeviceBuffer<super::Float2>,
        d_shorts: DeviceBuffer<i32>,
        d_longs: DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for AoBatchDeviceState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_into(
                    &self.d_prefix,
                    self.len,
                    self.first_valid,
                    &self.d_shorts,
                    &self.d_longs,
                    self.n_combos,
                    &self.d_out,
                    0,
                )
                .expect("ao launch");
            self.cuda.stream.synchronize().expect("ao sync");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaAo::new(0).expect("cuda ao");
        let hl2 = gen_series(ONE_SERIES_LEN);
        let sweep = AoBatchRange {
            // Keep output surface O(PARAM_SWEEP * len). Sweeping both short+long
            // multiplies combinations and can exceed VRAM / overflow indices.
            short_period: (5, 5, 0),
            long_period: (20, 20 + PARAM_SWEEP - 1, 1),
        };

        let len = hl2.len();
        let first_valid = hl2
            .iter()
            .position(|v| !v.is_nan())
            .unwrap_or(len);
        let combos = expand_grid_checked_cuda(&sweep).expect("expand_grid");
        let mut shorts: Vec<i32> = Vec::with_capacity(combos.len());
        let mut longs: Vec<i32> = Vec::with_capacity(combos.len());
        for prm in &combos {
            shorts.push(prm.short_period.unwrap_or(0) as i32);
            longs.push(prm.long_period.unwrap_or(0) as i32);
        }
        let n_combos = combos.len();

        let prefix: Vec<super::Float2> = build_prefix_ds(&hl2, first_valid);
        let d_prefix: DeviceBuffer<super::Float2> =
            DeviceBuffer::from_slice(&prefix).expect("d_prefix H2D");
        let d_shorts = DeviceBuffer::from_slice(&shorts).expect("d_shorts H2D");
        let d_longs = DeviceBuffer::from_slice(&longs).expect("d_longs H2D");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(len * n_combos) }.expect("d_out alloc");
        cuda.stream.synchronize().expect("ao prep sync");

        Box::new(AoBatchDeviceState {
            cuda,
            d_prefix,
            d_shorts,
            d_longs,
            len,
            first_valid,
            n_combos,
            d_out,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "ao",
            "one_series_many_params",
            "ao_cuda_batch_dev",
            "1m_x_250",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}
