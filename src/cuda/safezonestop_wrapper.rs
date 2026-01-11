#![cfg(feature = "cuda")]

//! CUDA wrapper for SafeZoneStop (recurrence-based stop levels).
//!
//! Parity with ALMA wrapper conventions:
//! - PTX load via OUT_DIR, JIT opts: DetermineTargetFromContext + O2
//! - NON_BLOCKING stream
//! - VRAM checks + y-chunking (<= 65_535)
//! - Public device entry points for batch and many-series (time-major)
//! - Batch precomputes dm_raw on host (shared across rows)

use crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32;
use crate::indicators::safezonestop::{SafeZoneStopBatchRange, SafeZoneStopParams};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaSafeZoneStopError {
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

pub struct CudaSafeZoneStop {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaSafeZoneStop {
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }

    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
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
    ) -> Result<(), CudaSafeZoneStopError> {
        let dev = Device::get_device(self.device_id)?;
        let max_threads = dev
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_bx = dev
            .get_attribute(DeviceAttribute::MaxBlockDimX)
            .unwrap_or(1024) as u32;
        let max_by = dev
            .get_attribute(DeviceAttribute::MaxBlockDimY)
            .unwrap_or(1024) as u32;
        let max_bz = dev
            .get_attribute(DeviceAttribute::MaxBlockDimZ)
            .unwrap_or(64) as u32;
        let max_gx = dev
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(2_147_483_647) as u32;
        let max_gy = dev
            .get_attribute(DeviceAttribute::MaxGridDimY)
            .unwrap_or(65_535) as u32;
        let max_gz = dev
            .get_attribute(DeviceAttribute::MaxGridDimZ)
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
            return Err(CudaSafeZoneStopError::LaunchConfigTooLarge { gx, gy, gz, bx, by, bz });
        }
        Ok(())
    }

    pub fn new(device_id: usize) -> Result<Self, CudaSafeZoneStopError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/safezonestop_kernel.ptx"));
        let jit = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) =
                    Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
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
        })
    }

    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaSafeZoneStopError> {
        self.stream.synchronize()?;
        Ok(())
    }

    #[inline]
    fn upload_pinned_f32(
        &self,
        src: &[f32],
    ) -> Result<(DeviceBuffer<f32>, LockedBuffer<f32>), CudaSafeZoneStopError> {
        // NOTE: For async copies, the pinned host buffer must stay alive until the stream
        // completes. Callers are responsible for keeping the returned LockedBuffer alive.
        let h_pin = LockedBuffer::from_slice(src)?;
        let mut d = unsafe { DeviceBuffer::<f32>::uninitialized_async(src.len(), &self.stream) }?;
        unsafe {
            d.async_copy_from(&h_pin, &self.stream)?;
        }
        Ok((d, h_pin))
    }

    #[inline]
    fn find_first_valid_pair(high: &[f32], low: &[f32]) -> Option<usize> {
        let n = high.len().min(low.len());
        for i in 0..n {
            let h = high[i];
            let l = low[i];
            if h.is_finite() && l.is_finite() {
                return Some(i);
            }
        }
        None
    }

    fn expand_grid(r: &SafeZoneStopBatchRange) -> Result<Vec<SafeZoneStopParams>, CudaSafeZoneStopError> {
        fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, CudaSafeZoneStopError> {
            if step == 0 || start == end {
                return Ok(vec![start]);
            }
            let mut vals = Vec::new();
            if start < end {
                let mut x = start;
                while x <= end {
                    vals.push(x);
                    x = x.checked_add(step).ok_or_else(|| {
                        CudaSafeZoneStopError::InvalidInput(format!(
                            "invalid range: start={}, end={}, step={}",
                            start, end, step
                        ))
                    })?;
                }
            } else {
                let mut x = start;
                while x >= end {
                    vals.push(x);
                    if x == end {
                        break;
                    }
                    x = x.checked_sub(step).ok_or_else(|| {
                        CudaSafeZoneStopError::InvalidInput(format!(
                            "invalid range: start={}, end={}, step={}",
                            start, end, step
                        ))
                    })?;
                }
            }
            if vals.is_empty() {
                return Err(CudaSafeZoneStopError::InvalidInput(format!(
                    "invalid range: start={}, end={}, step={}",
                    start, end, step
                )));
            }
            Ok(vals)
        }
        fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, CudaSafeZoneStopError> {
            if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
                return Ok(vec![start]);
            }
            let mut v = Vec::new();
            let mut x = start;
            if step > 0.0 {
                while x <= end + 1e-12 {
                    v.push(x);
                    x += step;
                }
            } else {
                while x >= end - 1e-12 {
                    v.push(x);
                    x += step;
                }
            }
            if v.is_empty() {
                return Err(CudaSafeZoneStopError::InvalidInput(format!(
                    "invalid range: start={}, end={}, step={}",
                    start, end, step
                )));
            }
            Ok(v)
        }
        let periods = axis_usize(r.period)?;
        let mults = axis_f64(r.mult)?;
        let looks = axis_usize(r.max_lookback)?;
        let cap = periods
            .len()
            .checked_mul(mults.len())
            .and_then(|v| v.checked_mul(looks.len()))
            .ok_or_else(|| CudaSafeZoneStopError::InvalidInput("grid size overflow".into()))?;
        let mut out = Vec::with_capacity(cap);
        for &p in &periods {
            for &m in &mults {
                for &lb in &looks {
                    out.push(SafeZoneStopParams {
                        period: Some(p),
                        mult: Some(m),
                        max_lookback: Some(lb),
                    });
                }
            }
        }
        Ok(out)
    }

    fn compute_dm_raw_f32(high: &[f32], low: &[f32], first: usize, dir_long: bool) -> Vec<f32> {
        let len = high.len();
        let mut dm = vec![0.0f32; len];
        if len == 0 {
            return dm;
        }
        let mut prev_h = high[first];
        let mut prev_l = low[first];
        for i in (first + 1)..len {
            let h = high[i];
            let l = low[i];
            let up = h - prev_h;
            let dn = prev_l - l;
            let up_pos = if up > 0.0 { up } else { 0.0 };
            let dn_pos = if dn > 0.0 { dn } else { 0.0 };
            let v = if dir_long {
                if dn_pos > up_pos { dn_pos } else { 0.0 }
            } else {
                if up_pos > dn_pos { up_pos } else { 0.0 }
            };
            dm[i] = v;
            prev_h = h;
            prev_l = l;
        }
        dm
    }

    /// Batch (one series × many params). Returns VRAM-backed matrix and combos.
    pub fn safezonestop_batch_dev(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        direction: &str,
        sweep: &SafeZoneStopBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<SafeZoneStopParams>), CudaSafeZoneStopError> {
        let n = high_f32.len();
        if n == 0 || low_f32.len() != n {
            return Err(CudaSafeZoneStopError::InvalidInput(
                "empty or mismatched inputs".into(),
            ));
        }
        let dir_long = match direction.as_bytes().get(0) {
            Some(b'l') => true,
            Some(b's') => false,
            _ => true,
        };
        let first = Self::find_first_valid_pair(high_f32, low_f32)
            .ok_or_else(|| CudaSafeZoneStopError::InvalidInput("all values are NaN".into()))?;

        let combos = Self::expand_grid(sweep)?;
        // Validate and collect params per row
        let mut periods_i32 = Vec::with_capacity(combos.len());
        let mut mults_f32 = Vec::with_capacity(combos.len());
        let mut looks_i32 = Vec::with_capacity(combos.len());
        let mut max_look = 0usize;
        for c in &combos {
            let p = c.period.unwrap_or(22);
            let m = c.mult.unwrap_or(2.5) as f32;
            let lb = c.max_lookback.unwrap_or(3);
            if p == 0 || lb == 0 {
                return Err(CudaSafeZoneStopError::InvalidInput(
                    "period/lookback must be > 0".into(),
                ));
            }
            if p > n || lb > n {
                return Err(CudaSafeZoneStopError::InvalidInput(
                    "period/lookback exceed length".into(),
                ));
            }
            if n - first < (p + 1).max(lb) {
                return Err(CudaSafeZoneStopError::InvalidInput(format!(
                    "not enough valid data for period={}, lb={} (valid after first={} is {})",
                    p,
                    lb,
                    first,
                    n - first
                )));
            }
            periods_i32.push(p as i32);
            mults_f32.push(m);
            looks_i32.push(lb as i32);
            max_look = max_look.max(lb);
        }

        // Shared precompute
        let dm_raw = Self::compute_dm_raw_f32(high_f32, low_f32, first, dir_long);

        // Decide if we need deque workspace (lb <= 4 uses register-only path)
        let need_deque = max_look > 4;

        // VRAM accounting: inputs + params + outputs (+ optional deque), with checked arithmetic.
        let out_elems = combos
            .len()
            .checked_mul(n)
            .ok_or_else(|| CudaSafeZoneStopError::InvalidInput("rows*cols overflow".into()))?;
        let sz_f32 = std::mem::size_of::<f32>();
        let sz_i32 = std::mem::size_of::<i32>();
        let inputs_bytes = high_f32
            .len()
            .checked_add(low_f32.len())
            .and_then(|v| v.checked_add(dm_raw.len()))
            .and_then(|v| v.checked_mul(sz_f32))
            .ok_or_else(|| CudaSafeZoneStopError::InvalidInput("input bytes overflow".into()))?;
        let params_bytes = periods_i32
            .len()
            .checked_mul(sz_i32)
            .and_then(|v| mults_f32.len().checked_mul(sz_f32).and_then(|m| v.checked_add(m)))
            .and_then(|v| looks_i32.len().checked_mul(sz_i32).and_then(|l| v.checked_add(l)))
            .ok_or_else(|| CudaSafeZoneStopError::InvalidInput("param bytes overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaSafeZoneStopError::InvalidInput("output bytes overflow".into()))?;
        let mut required = inputs_bytes
            .checked_add(params_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaSafeZoneStopError::InvalidInput("total bytes overflow".into()))?;
        if need_deque {
            let deque_elems = combos
                .len()
                .checked_mul(max_look.checked_add(1).ok_or_else(|| {
                    CudaSafeZoneStopError::InvalidInput("deque lookback overflow".into())
                })?)
                .ok_or_else(|| CudaSafeZoneStopError::InvalidInput("deque elems overflow".into()))?;
            let deque_bytes = deque_elems
                .checked_mul(sz_f32 + sz_i32)
                .ok_or_else(|| CudaSafeZoneStopError::InvalidInput("deque bytes overflow".into()))?;
            required = required
                .checked_add(deque_bytes)
                .ok_or_else(|| CudaSafeZoneStopError::InvalidInput("total bytes overflow".into()))?;
        }
        let headroom = 64 * 1024 * 1024; // ~64MB
        if !Self::will_fit(required, headroom) {
            if let Some((free, _)) = Self::device_mem_info() {
                return Err(CudaSafeZoneStopError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaSafeZoneStopError::InvalidInput(
                    "insufficient device memory".into(),
                ));
            }
        }

        // Uploads: pinned for big arrays for true async H2D
        let (d_high, h_high_pin) = self.upload_pinned_f32(high_f32)?;
        let (d_low, h_low_pin) = self.upload_pinned_f32(low_f32)?;
        let (d_dm, h_dm_pin) = self.upload_pinned_f32(&dm_raw)?;

        // Small arrays sync-upload is fine
        let d_periods = DeviceBuffer::from_slice(&periods_i32)?;
        let d_mults = DeviceBuffer::from_slice(&mults_f32)?;
        let d_looks = DeviceBuffer::from_slice(&looks_i32)?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_elems, &self.stream) }?;

        // Optional deque workspace only when needed
        let (mut opt_q_idx, mut opt_q_val): (Option<DeviceBuffer<i32>>, Option<DeviceBuffer<f32>>) = (None, None);
        let mut lb_cap_i32 = 0i32;
        if need_deque {
            let lb_cap = (max_look + 1).max(2);
            let d_q_idx: DeviceBuffer<i32> =
                unsafe { DeviceBuffer::uninitialized_async(combos.len() * lb_cap, &self.stream) }?;
            let d_q_val: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized_async(combos.len() * lb_cap, &self.stream) }?;
            lb_cap_i32 = lb_cap as i32;
            opt_q_idx = Some(d_q_idx);
            opt_q_val = Some(d_q_val);
        }

        // Launch: 1D grid.x with many threads per block
        let func = self
            .module
            .get_function("safezonestop_batch_f32")
            .map_err(|_| CudaSafeZoneStopError::MissingKernelSymbol {
                name: "safezonestop_batch_f32",
            })?;

        const TB: u32 = 256;
        let block: BlockSize = (TB, 1, 1).into();
        let grid_x = ((combos.len() as u32) + TB - 1) / TB;
        let grid: GridSize = (grid_x, 1, 1).into();
        self.validate_launch(grid_x, 1, 1, TB, 1, 1)?;
        let dir_i32 = if dir_long { 1i32 } else { 0i32 };
        let stream = &self.stream;

        unsafe {
            if need_deque {
                let q_idx_ptr = opt_q_idx.as_ref().unwrap().as_device_ptr().as_raw();
                let q_val_ptr = opt_q_val.as_ref().unwrap().as_device_ptr().as_raw();
                launch!(
                    func<<<grid, block, 0, stream>>>(
                        d_high.as_device_ptr().as_raw(),
                        d_low.as_device_ptr().as_raw(),
                        d_dm.as_device_ptr().as_raw(),
                        n as i32,
                        first as i32,
                        d_periods.as_device_ptr().as_raw(),
                        d_mults.as_device_ptr().as_raw(),
                        d_looks.as_device_ptr().as_raw(),
                        combos.len() as i32,
                        dir_i32,
                        q_idx_ptr,
                        q_val_ptr,
                        lb_cap_i32,
                        d_out.as_device_ptr().as_raw()
                    )
                )?;
            } else {
                launch!(
                    func<<<grid, block, 0, stream>>>(
                        d_high.as_device_ptr().as_raw(),
                        d_low.as_device_ptr().as_raw(),
                        d_dm.as_device_ptr().as_raw(),
                        n as i32,
                        first as i32,
                        d_periods.as_device_ptr().as_raw(),
                        d_mults.as_device_ptr().as_raw(),
                        d_looks.as_device_ptr().as_raw(),
                        combos.len() as i32,
                        dir_i32,
                        0u64,
                        0u64,
                        0i32,
                        d_out.as_device_ptr().as_raw()
                    )
                )?;
            }
        }

        // Keep optional buffers alive across the launch
        drop(opt_q_idx);
        drop(opt_q_val);

        self.stream.synchronize()?;

        // Keep pinned host staging buffers alive until all async copies are complete.
        drop(h_high_pin);
        drop(h_low_pin);
        drop(h_dm_pin);

        Ok((
            DeviceArrayF32 {
                buf: d_out,
                rows: combos.len(),
                cols: n,
            },
            combos,
        ))
    }

    /// Many-series × one-param (time-major)
    pub fn safezonestop_many_series_one_param_time_major_dev(
        &self,
        high_tm_f32: &[f32],
        low_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
        mult: f32,
        max_lookback: usize,
        direction: &str,
    ) -> Result<DeviceArrayF32, CudaSafeZoneStopError> {
        if cols == 0 || rows == 0 {
            return Err(CudaSafeZoneStopError::InvalidInput("empty matrix".into()));
        }
        let n = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaSafeZoneStopError::InvalidInput("rows*cols overflow".into()))?;
        if high_tm_f32.len() != n || low_tm_f32.len() != n {
            return Err(CudaSafeZoneStopError::InvalidInput(
                "matrix inputs mismatch".into(),
            ));
        }
        if period == 0 || max_lookback == 0 {
            return Err(CudaSafeZoneStopError::InvalidInput(
                "period/lookback must be > 0".into(),
            ));
        }
        let dir_long = match direction.as_bytes().get(0) {
            Some(b'l') => true,
            Some(b's') => false,
            _ => true,
        };

        // first-valid per series
        let mut first_valids = vec![-1i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let h = high_tm_f32[t * cols + s];
                let l = low_tm_f32[t * cols + s];
                if h.is_finite() && l.is_finite() {
                    first_valids[s] = t as i32;
                    break;
                }
            }
            if first_valids[s] < 0 {
                return Err(CudaSafeZoneStopError::InvalidInput(format!(
                    "series {} all NaN",
                    s
                )));
            }
            let f = first_valids[s] as usize;
            if rows - f < (period + 1).max(max_lookback) {
                return Err(CudaSafeZoneStopError::InvalidInput(format!(
                    "series {} not enough valid data (need >= {}, have {})",
                    s,
                    (period + 1).max(max_lookback),
                    rows - f
                )));
            }
        }

        // VRAM accounting: inputs + first_valids + outputs (+ optional deque)
        let need_deque = max_lookback > 4;
        let sz_f32 = std::mem::size_of::<f32>();
        let sz_i32 = std::mem::size_of::<i32>();
        let inputs_bytes = n
            .checked_mul(2)
            .and_then(|v| v.checked_mul(sz_f32))
            .ok_or_else(|| CudaSafeZoneStopError::InvalidInput("input bytes overflow".into()))?;
        let first_bytes = cols
            .checked_mul(sz_i32)
            .ok_or_else(|| CudaSafeZoneStopError::InvalidInput("first_valid bytes overflow".into()))?;
        let out_bytes = n
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaSafeZoneStopError::InvalidInput("output bytes overflow".into()))?;
        let mut required = inputs_bytes
            .checked_add(first_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaSafeZoneStopError::InvalidInput("total bytes overflow".into()))?;
        if need_deque {
            let deque_elems = cols
                .checked_mul(max_lookback.checked_add(1).ok_or_else(|| {
                    CudaSafeZoneStopError::InvalidInput("deque lookback overflow".into())
                })?)
                .ok_or_else(|| CudaSafeZoneStopError::InvalidInput("deque elems overflow".into()))?;
            let deque_bytes = deque_elems
                .checked_mul(sz_f32 + sz_i32)
                .ok_or_else(|| CudaSafeZoneStopError::InvalidInput("deque bytes overflow".into()))?;
            required = required
                .checked_add(deque_bytes)
                .ok_or_else(|| CudaSafeZoneStopError::InvalidInput("total bytes overflow".into()))?;
        }
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            if let Some((free, _)) = Self::device_mem_info() {
                return Err(CudaSafeZoneStopError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaSafeZoneStopError::InvalidInput(
                    "insufficient device memory".into(),
                ));
            }
        }

        // Uploads (pinned for big matrices)
        let (d_high, h_high_pin) = self.upload_pinned_f32(high_tm_f32)?;
        let (d_low, h_low_pin) = self.upload_pinned_f32(low_tm_f32)?;
        let d_first = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(n, &self.stream) }?;

        // Optional deque workspace per series
        let (mut opt_q_idx, mut opt_q_val): (Option<DeviceBuffer<i32>>, Option<DeviceBuffer<f32>>) = (None, None);
        let mut lb_cap_i32 = 0i32;
        if need_deque {
            let lb_cap = (max_lookback + 1).max(2);
            let d_q_idx: DeviceBuffer<i32> =
                unsafe { DeviceBuffer::uninitialized_async(cols * lb_cap, &self.stream) }?;
            let d_q_val: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized_async(cols * lb_cap, &self.stream) }?;
            lb_cap_i32 = lb_cap as i32;
            opt_q_idx = Some(d_q_idx);
            opt_q_val = Some(d_q_val);
        }

        // Launch (1D grid.x with threads-per-block)
        let func = self
            .module
            .get_function("safezonestop_many_series_one_param_time_major_f32")
            .map_err(|_| CudaSafeZoneStopError::MissingKernelSymbol {
                name: "safezonestop_many_series_one_param_time_major_f32",
            })?;
        const TB: u32 = 256;
        let block: BlockSize = (TB, 1, 1).into();
        let grid_x = ((cols as u32) + TB - 1) / TB;
        let grid: GridSize = (grid_x, 1, 1).into();
        self.validate_launch(grid_x, 1, 1, TB, 1, 1)?;
        let dir_i32 = if dir_long { 1i32 } else { 0i32 };
        let stream = &self.stream;
        unsafe {
            if need_deque {
                let q_idx_ptr = opt_q_idx.as_ref().unwrap().as_device_ptr().as_raw();
                let q_val_ptr = opt_q_val.as_ref().unwrap().as_device_ptr().as_raw();
                launch!(
                    func<<<grid, block, 0, stream>>>(
                        d_high.as_device_ptr().as_raw(),
                        d_low.as_device_ptr().as_raw(),
                        cols as i32,
                        rows as i32,
                        period as i32,
                        mult as f32,
                        max_lookback as i32,
                        d_first.as_device_ptr().as_raw(),
                        dir_i32,
                        q_idx_ptr,
                        q_val_ptr,
                        lb_cap_i32,
                        d_out.as_device_ptr().as_raw()
                    )
                )?;
            } else {
                launch!(
                    func<<<grid, block, 0, stream>>>(
                        d_high.as_device_ptr().as_raw(),
                        d_low.as_device_ptr().as_raw(),
                        cols as i32,
                        rows as i32,
                        period as i32,
                        mult as f32,
                        max_lookback as i32,
                        d_first.as_device_ptr().as_raw(),
                        dir_i32,
                        0u64,
                        0u64,
                        0i32,
                        d_out.as_device_ptr().as_raw()
                    )
                )?;
            }
        }
        // Keep optional buffers alive across the launch
        drop(opt_q_idx);
        drop(opt_q_val);

        self.stream.synchronize()?;

        // Keep pinned host staging buffers alive until all async copies are complete.
        drop(h_high_pin);
        drop(h_low_pin);

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "safezonestop",
                "batch_dev",
                "safezonestop_cuda_batch_dev",
                "60k_x_27combos",
                prep_batch_box,
            )
            .with_inner_iters(4),
            CudaBenchScenario::new(
                "safezonestop",
                "many_series_one_param",
                "safezonestop_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_box,
            )
            .with_inner_iters(2),
        ]
    }

    struct BatchDeviceState {
        cuda: CudaSafeZoneStop,
        d_high: DeviceBuffer<f32>,
        d_low: DeviceBuffer<f32>,
        d_dm: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_mults: DeviceBuffer<f32>,
        d_looks: DeviceBuffer<i32>,
        d_q_idx: DeviceBuffer<i32>,
        d_q_val: DeviceBuffer<f32>,
        lb_cap_i32: i32,
        d_out: DeviceBuffer<f32>,
        n: usize,
        first: usize,
        combos: usize,
        dir_i32: i32,
        grid: GridSize,
        block: BlockSize,
    }
    impl CudaBenchState for BatchDeviceState {
        fn launch(&mut self) {
            let func = self
                .cuda
                .module
                .get_function("safezonestop_batch_f32")
                .expect("safezonestop_batch_f32");
            let stream = &self.cuda.stream;
            unsafe {
                launch!(
                    func<<<self.grid, self.block, 0, stream>>>(
                        self.d_high.as_device_ptr().as_raw(),
                        self.d_low.as_device_ptr().as_raw(),
                        self.d_dm.as_device_ptr().as_raw(),
                        self.n as i32,
                        self.first as i32,
                        self.d_periods.as_device_ptr().as_raw(),
                        self.d_mults.as_device_ptr().as_raw(),
                        self.d_looks.as_device_ptr().as_raw(),
                        self.combos as i32,
                        self.dir_i32,
                        self.d_q_idx.as_device_ptr().as_raw(),
                        self.d_q_val.as_device_ptr().as_raw(),
                        self.lb_cap_i32,
                        self.d_out.as_device_ptr().as_raw()
                    )
                )
                .expect("safezonestop batch launch");
            }
            self.cuda.stream.synchronize().expect("safezonestop batch sync");
        }
    }
    fn prep_batch() -> BatchDeviceState {
        let cuda = CudaSafeZoneStop::new(0).expect("cuda szz");
        let len = 60_000usize;
        let mut high = vec![f32::NAN; len];
        let mut low = vec![f32::NAN; len];
        for i in 3..len {
            let x = i as f32;
            let base = (x * 0.001).sin() + 0.0002 * x;
            high[i] = base + 0.5;
            low[i] = base - 0.5;
        }
        let sweep = SafeZoneStopBatchRange {
            period: (10, 22, 6),
            mult: (1.5, 3.0, 0.75),
            max_lookback: (3, 5, 1),
        };
        let dir_long = true;
        let dir_i32 = 1i32;
        let first = (0..len)
            .find(|&i| high[i].is_finite() && low[i].is_finite())
            .unwrap_or(0);
        let combos = CudaSafeZoneStop::expand_grid(&sweep).expect("expand_grid");
        let n_combos = combos.len();
        let mut periods_i32 = Vec::with_capacity(n_combos);
        let mut mults_f32 = Vec::with_capacity(n_combos);
        let mut looks_i32 = Vec::with_capacity(n_combos);
        let mut max_look = 0usize;
        for prm in &combos {
            let p = prm.period.unwrap_or(14);
            let m = prm.mult.unwrap_or(2.0) as f32;
            let lb = prm.max_lookback.unwrap_or(3);
            periods_i32.push(p as i32);
            mults_f32.push(m);
            looks_i32.push(lb as i32);
            max_look = max_look.max(lb);
        }
        let dm_raw = CudaSafeZoneStop::compute_dm_raw_f32(&high, &low, first, dir_long);
        let lb_cap = (max_look + 1).max(2);
        let out_elems = n_combos * len;

        let d_high = DeviceBuffer::from_slice(&high).expect("d_high");
        let d_low = DeviceBuffer::from_slice(&low).expect("d_low");
        let d_dm = DeviceBuffer::from_slice(&dm_raw).expect("d_dm");
        let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("d_periods");
        let d_mults = DeviceBuffer::from_slice(&mults_f32).expect("d_mults");
        let d_looks = DeviceBuffer::from_slice(&looks_i32).expect("d_looks");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(out_elems) }.expect("d_out");
        let d_q_idx: DeviceBuffer<i32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * lb_cap) }.expect("d_q_idx");
        let d_q_val: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * lb_cap) }.expect("d_q_val");

        const TB: u32 = 256;
        let block: BlockSize = (TB, 1, 1).into();
        let grid_x = ((n_combos as u32) + TB - 1) / TB;
        let grid: GridSize = (grid_x, 1, 1).into();
        cuda.validate_launch(grid_x, 1, 1, TB, 1, 1)
            .expect("validate launch");
        cuda.stream.synchronize().expect("sync after prep");

        BatchDeviceState {
            cuda,
            d_high,
            d_low,
            d_dm,
            d_periods,
            d_mults,
            d_looks,
            d_q_idx,
            d_q_val,
            lb_cap_i32: lb_cap as i32,
            d_out,
            n: len,
            first,
            combos: n_combos,
            dir_i32,
            grid,
            block,
        }
    }
    fn prep_batch_box() -> Box<dyn CudaBenchState> {
        Box::new(prep_batch())
    }

    struct ManySeriesDeviceState {
        cuda: CudaSafeZoneStop,
        d_high_tm: DeviceBuffer<f32>,
        d_low_tm: DeviceBuffer<f32>,
        d_first: DeviceBuffer<i32>,
        d_out: DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        period: usize,
        mult: f32,
        lb: usize,
    }
    impl CudaBenchState for ManySeriesDeviceState {
        fn launch(&mut self) {
            let func = self
                .cuda
                .module
                .get_function("safezonestop_many_series_one_param_time_major_f32")
                .expect("safezonestop_many_series_one_param_time_major_f32");
            const TB: u32 = 256;
            let block: BlockSize = (TB, 1, 1).into();
            let grid_x = ((self.cols as u32) + TB - 1) / TB;
            let grid: GridSize = (grid_x, 1, 1).into();
            let stream = &self.cuda.stream;
            unsafe {
                launch!(
                    func<<<grid, block, 0, stream>>>(
                        self.d_high_tm.as_device_ptr().as_raw(),
                        self.d_low_tm.as_device_ptr().as_raw(),
                        self.cols as i32,
                        self.rows as i32,
                        self.period as i32,
                        self.mult,
                        self.lb as i32,
                        self.d_first.as_device_ptr().as_raw(),
                        1i32,
                        0u64,
                        0u64,
                        0i32,
                        self.d_out.as_device_ptr().as_raw()
                    )
                )
                .expect("safezonestop many launch");
            }
            self.cuda.stream.synchronize().expect("safezonestop many sync");
        }
    }
    fn prep_many_series() -> ManySeriesDeviceState {
        let cuda = CudaSafeZoneStop::new(0).expect("cuda szz");
        let cols = 250usize;
        let rows = 1_000_000usize;
        let mut high_tm = vec![f32::NAN; cols * rows];
        let mut low_tm = vec![f32::NAN; cols * rows];
        for s in 0..cols {
            for t in s..rows {
                let x = (t as f32) + (s as f32) * 0.17;
                let base = (x * 0.001).sin() + 0.0002 * x;
                high_tm[t * cols + s] = base + 0.5;
                low_tm[t * cols + s] = base - 0.5;
            }
        }
        let mut first_valids = vec![rows as i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                if high_tm[idx].is_finite() && low_tm[idx].is_finite() {
                    first_valids[s] = t as i32;
                    break;
                }
            }
        }
        let d_high_tm = DeviceBuffer::from_slice(&high_tm).expect("d_high_tm");
        let d_low_tm = DeviceBuffer::from_slice(&low_tm).expect("d_low_tm");
        let d_first = DeviceBuffer::from_slice(&first_valids).expect("d_first");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out");
        cuda.stream.synchronize().expect("sync after prep");
        ManySeriesDeviceState {
            cuda,
            d_high_tm,
            d_low_tm,
            d_first,
            d_out,
            cols,
            rows,
            period: 22,
            mult: 2.5,
            lb: 3,
        }
    }
    fn prep_many_series_box() -> Box<dyn CudaBenchState> {
        Box::new(prep_many_series())
    }
}
