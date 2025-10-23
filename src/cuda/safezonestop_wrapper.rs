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
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaSafeZoneStopError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaSafeZoneStopError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaSafeZoneStopError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaSafeZoneStopError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl std::error::Error for CudaSafeZoneStopError {}

pub struct CudaSafeZoneStop {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaSafeZoneStop {
    pub fn new(device_id: usize) -> Result<Self, CudaSafeZoneStopError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaSafeZoneStopError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaSafeZoneStopError::Cuda(e.to_string()))?;
        let context =
            Context::new(device).map_err(|e| CudaSafeZoneStopError::Cuda(e.to_string()))?;
        let context =
            Context::new(device).map_err(|e| CudaSafeZoneStopError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/safezonestop_kernel.ptx"));
        let jit = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaSafeZoneStopError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaSafeZoneStopError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaSafeZoneStopError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaSafeZoneStopError::Cuda(e.to_string()))
    }

    #[inline]
    fn upload_pinned_f32(&self, src: &[f32]) -> Result<DeviceBuffer<f32>, CudaSafeZoneStopError> {
        // Stage in pinned (page-locked) host memory so async H2D truly overlaps compute.
        let h_pin = LockedBuffer::from_slice(src)
            .map_err(|e| CudaSafeZoneStopError::Cuda(e.to_string()))?;
        let mut d = unsafe { DeviceBuffer::<f32>::uninitialized_async(src.len(), &self.stream) }
            .map_err(|e| CudaSafeZoneStopError::Cuda(e.to_string()))?;
        unsafe {
            d.async_copy_from(&h_pin, &self.stream)
                .map_err(|e| CudaSafeZoneStopError::Cuda(e.to_string()))?;
        }
        Ok(d)
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
            if h.is_finite() && l.is_finite() {
                return Some(i);
            }
        }
        None
    }

    fn expand_grid(r: &SafeZoneStopBatchRange) -> Vec<SafeZoneStopParams> {
        fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
            if step == 0 || start == end {
                return vec![start];
            }
            if step == 0 || start == end {
                return vec![start];
            }
            (start..=end).step_by(step).collect()
        }
        fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
            if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
                return vec![start];
            }
            if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
                return vec![start];
            }
            let mut out = Vec::new();
            let mut x = start;
            while x <= end + 1e-12 {
                out.push(x);
                x += step;
            }
            while x <= end + 1e-12 {
                out.push(x);
                x += step;
            }
            out
        }
        let periods = axis_usize(r.period);
        let mults = axis_f64(r.mult);
        let looks = axis_usize(r.max_lookback);
        let mut out = Vec::with_capacity(periods.len() * mults.len() * looks.len());
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
        out
    }

    fn compute_dm_raw_f32(high: &[f32], low: &[f32], first: usize, dir_long: bool) -> Vec<f32> {
        let len = high.len();
        let mut dm = vec![0.0f32; len];
        if len == 0 {
            return dm;
        }
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
                if dn_pos > up_pos {
                    dn_pos
                } else {
                    0.0
                }
                if dn_pos > up_pos {
                    dn_pos
                } else {
                    0.0
                }
            } else {
                if up_pos > dn_pos {
                    up_pos
                } else {
                    0.0
                }
                if up_pos > dn_pos {
                    up_pos
                } else {
                    0.0
                }
            };
            dm[i] = v;
            prev_h = h;
            prev_l = l;
        }
        dm
    }

    #[inline]
    fn will_fit(bytes_needed: usize, headroom: usize) -> bool {
        if let Ok((free, _)) = mem_get_info() {
            let free = free.saturating_sub(headroom);
            (bytes_needed as u64) <= (free as u64)
        } else {
            true
        }
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
            return Err(CudaSafeZoneStopError::InvalidInput(
                "empty or mismatched inputs".into(),
            ));
        }
        let dir_long = match direction.as_bytes().get(0) {
            Some(b'l') => true,
            Some(b's') => false,
            _ => true,
        };
        let dir_long = match direction.as_bytes().get(0) {
            Some(b'l') => true,
            Some(b's') => false,
            _ => true,
        };
        let first = Self::find_first_valid_pair(high_f32, low_f32)
            .ok_or_else(|| CudaSafeZoneStopError::InvalidInput("all values are NaN".into()))?;

        let combos = Self::expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaSafeZoneStopError::InvalidInput(
                "no parameter combinations".into(),
            ));
            return Err(CudaSafeZoneStopError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
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

        // VRAM accounting: inputs + params + outputs (+ optional deque)
        let out_elems = combos.len() * n;
        let headroom = 64 * 1024 * 1024; // ~64MB
        let mut bytes = (high_f32.len() + low_f32.len() + dm_raw.len()) * 4
            + (periods_i32.len() * 4 + mults_f32.len() * 4 + looks_i32.len() * 4)
            + (out_elems * 4);
        if need_deque {
            bytes += combos.len() * (max_look + 1) * (4 + 4); // q_idx + q_val
        }
        if !Self::will_fit(bytes, headroom) {
            return Err(CudaSafeZoneStopError::InvalidInput(
                "insufficient device memory".into(),
            ));
            return Err(CudaSafeZoneStopError::InvalidInput(
                "insufficient device memory".into(),
            ));
        }

        // Uploads: pinned for big arrays for true async H2D
        let d_high = self.upload_pinned_f32(high_f32)?;
        let d_low  = self.upload_pinned_f32(low_f32)?;
        let d_dm   = self.upload_pinned_f32(&dm_raw)?;

        // Small arrays sync-upload is fine
        let d_periods = DeviceBuffer::from_slice(&periods_i32)
            .map_err(|e| CudaSafeZoneStopError::Cuda(e.to_string()))?;
        let d_mults = DeviceBuffer::from_slice(&mults_f32)
            .map_err(|e| CudaSafeZoneStopError::Cuda(e.to_string()))?;
        let d_looks = DeviceBuffer::from_slice(&looks_i32)
            .map_err(|e| CudaSafeZoneStopError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_elems, &self.stream) }
                .map_err(|e| CudaSafeZoneStopError::Cuda(e.to_string()))?;

        // Optional deque workspace only when needed
        let (mut opt_q_idx, mut opt_q_val): (Option<DeviceBuffer<i32>>, Option<DeviceBuffer<f32>>) = (None, None);
        let mut lb_cap_i32 = 0i32;
        if need_deque {
            let lb_cap = (max_look + 1).max(2);
            let d_q_idx: DeviceBuffer<i32> =
                unsafe { DeviceBuffer::uninitialized_async(combos.len() * lb_cap, &self.stream) }
                    .map_err(|e| CudaSafeZoneStopError::Cuda(e.to_string()))?;
            let d_q_val: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized_async(combos.len() * lb_cap, &self.stream) }
                    .map_err(|e| CudaSafeZoneStopError::Cuda(e.to_string()))?;
            lb_cap_i32 = lb_cap as i32;
            opt_q_idx = Some(d_q_idx);
            opt_q_val = Some(d_q_val);
        }

        // Launch: 1D grid.x with many threads per block
        let func = self
            .module
            .get_function("safezonestop_batch_f32")
            .map_err(|e| CudaSafeZoneStopError::Cuda(e.to_string()))?;

        const TB: u32 = 256;
        let block: BlockSize = (TB, 1, 1).into();
        let grid_x = ((combos.len() as u32) + TB - 1) / TB;
        let grid: GridSize = (grid_x, 1, 1).into();
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
                )
                .map_err(|e| CudaSafeZoneStopError::Cuda(e.to_string()))?;
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
                )
                .map_err(|e| CudaSafeZoneStopError::Cuda(e.to_string()))?;
            }
        }

        // Keep optional buffers alive across the launch
        drop(opt_q_idx);
        drop(opt_q_val);

        self.stream.synchronize().map_err(|e| CudaSafeZoneStopError::Cuda(e.to_string()))?;

        Ok((
            DeviceArrayF32 {
                buf: d_out,
                rows: combos.len(),
                cols: n,
            },
            combos,
        ))
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
        if cols == 0 || rows == 0 {
            return Err(CudaSafeZoneStopError::InvalidInput("empty matrix".into()));
        }
        let n = cols * rows;
        if high_tm_f32.len() != n || low_tm_f32.len() != n {
            return Err(CudaSafeZoneStopError::InvalidInput(
                "matrix inputs mismatch".into(),
            ));
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
                    s,
                    (period + 1).max(max_lookback),
                    rows - f
                )));
            }
        }

        // VRAM accounting: inputs + first_valids + outputs (+ optional deque)
        let need_deque = max_lookback > 4;
        let headroom = 64 * 1024 * 1024;
        let mut bytes = n * 4 * 2 + cols * 4 + n * 4;
        if need_deque { bytes += cols * (max_lookback + 1) * (4 + 4); }
        if !Self::will_fit(bytes, headroom) {
            return Err(CudaSafeZoneStopError::InvalidInput(
                "insufficient device memory".into(),
            ));
            return Err(CudaSafeZoneStopError::InvalidInput(
                "insufficient device memory".into(),
            ));
        }

        // Uploads (pinned for big matrices)
        let d_high = self.upload_pinned_f32(high_tm_f32)?;
        let d_low  = self.upload_pinned_f32(low_tm_f32)?;
        let d_first = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaSafeZoneStopError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(n, &self.stream) }
                .map_err(|e| CudaSafeZoneStopError::Cuda(e.to_string()))?;

        // Optional deque workspace per series
        let (mut opt_q_idx, mut opt_q_val): (Option<DeviceBuffer<i32>>, Option<DeviceBuffer<f32>>) = (None, None);
        let mut lb_cap_i32 = 0i32;
        if need_deque {
            let lb_cap = (max_lookback + 1).max(2);
            let d_q_idx: DeviceBuffer<i32> =
                unsafe { DeviceBuffer::uninitialized_async(cols * lb_cap, &self.stream) }
                    .map_err(|e| CudaSafeZoneStopError::Cuda(e.to_string()))?;
            let d_q_val: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized_async(cols * lb_cap, &self.stream) }
                    .map_err(|e| CudaSafeZoneStopError::Cuda(e.to_string()))?;
            lb_cap_i32 = lb_cap as i32;
            opt_q_idx = Some(d_q_idx);
            opt_q_val = Some(d_q_val);
        }

        // Launch (1D grid.x with threads-per-block)
        let func = self
            .module
            .get_function("safezonestop_many_series_one_param_time_major_f32")
            .map_err(|e| CudaSafeZoneStopError::Cuda(e.to_string()))?;
        const TB: u32 = 256;
        let block: BlockSize = (TB, 1, 1).into();
        let grid_x = ((cols as u32) + TB - 1) / TB;
        let grid: GridSize = (grid_x, 1, 1).into();
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
                )
                .map_err(|e| CudaSafeZoneStopError::Cuda(e.to_string()))?;
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
                )
                .map_err(|e| CudaSafeZoneStopError::Cuda(e.to_string()))?;
            }
        }
        // Keep optional buffers alive across the launch
        drop(opt_q_idx);
        drop(opt_q_val);

        self.stream.synchronize().map_err(|e| CudaSafeZoneStopError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
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

    struct BatchState {
        cuda: CudaSafeZoneStop,
        high: Vec<f32>,
        low: Vec<f32>,
        sweep: SafeZoneStopBatchRange,
        dir: &'static str,
    }
    impl CudaBenchState for BatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .safezonestop_batch_dev(&self.high, &self.low, self.dir, &self.sweep);
        }
    }
    fn prep_batch() -> BatchState {
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
        BatchState {
            cuda,
            high,
            low,
            sweep,
            dir: "long",
        }
        let sweep = SafeZoneStopBatchRange {
            period: (10, 22, 6),
            mult: (1.5, 3.0, 0.75),
            max_lookback: (3, 5, 1),
        };
        BatchState {
            cuda,
            high,
            low,
            sweep,
            dir: "long",
        }
    }
    fn prep_batch_box() -> Box<dyn CudaBenchState> {
        Box::new(prep_batch())
    }
    fn prep_batch_box() -> Box<dyn CudaBenchState> {
        Box::new(prep_batch())
    }

    struct ManySeriesState {
        cuda: CudaSafeZoneStop,
        high_tm: Vec<f32>,
        low_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        period: usize,
        mult: f32,
        lb: usize,
    }
    impl CudaBenchState for ManySeriesState {
        fn launch(&mut self) {
            let _ = self.cuda.safezonestop_many_series_one_param_time_major_dev(
                &self.high_tm,
                &self.low_tm,
                self.cols,
                self.rows,
                self.period,
                self.mult,
                self.lb,
                "long",
            );
        }
    }
    fn prep_many_series() -> ManySeriesState {
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
        ManySeriesState {
            cuda,
            high_tm,
            low_tm,
            cols,
            rows,
            period: 22,
            mult: 2.5,
            lb: 3,
        }
        for s in 0..cols {
            for t in s..rows {
                let x = (t as f32) + (s as f32) * 0.17;
                let base = (x * 0.001).sin() + 0.0002 * x;
                high_tm[t * cols + s] = base + 0.5;
                low_tm[t * cols + s] = base - 0.5;
            }
        }
        ManySeriesState {
            cuda,
            high_tm,
            low_tm,
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
    fn prep_many_series_box() -> Box<dyn CudaBenchState> {
        Box::new(prep_many_series())
    }
}
