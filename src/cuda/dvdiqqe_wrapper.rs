#![cfg(feature = "cuda")]

//! CUDA wrapper for the DVDIQQE indicator.
//!
//! Parity goals with ALMA wrapper:
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/dvdiqqe_kernel.ptx"))
//! - NON_BLOCKING stream
//! - Simple explicit policies for 1D launches
//! - VRAM guard + chunking; grid.y <= 65_535
//! - Public device entry points mirror other wrappers

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::dvdiqqe::DvdiqqeBatchRange;
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
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
pub struct CudaDvdiqqePolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaDvdiqqePolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

#[derive(Debug, Error)]
pub enum CudaDvdiqqeError {
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

/// Four-plane output for DVDIQQE (dvdi, fast, slow, center)
pub struct DeviceDvdiqqeQuad {
    pub dvdi: DeviceArrayF32,
    pub fast: DeviceArrayF32,
    pub slow: DeviceArrayF32,
    pub center: DeviceArrayF32,
    pub ctx: Arc<Context>,
    pub device_id: u32,
}

pub struct CudaDvdiqqe {
    module: Module,
    stream: Stream,
    ctx: Arc<Context>,
    device_id: u32,
    policy: CudaDvdiqqePolicy,
}

impl CudaDvdiqqe {
    pub fn new(device_id: usize) -> Result<Self, CudaDvdiqqeError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let ctx = Arc::new(Context::new(device)?);
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/dvdiqqe_kernel.ptx"));
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
            ctx,
            device_id: device_id as u32,
            policy: CudaDvdiqqePolicy::default(),
        })
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> { self.ctx.clone() }
    #[inline]
    pub fn device_id(&self) -> u32 { self.device_id }

    #[inline]
    fn align_to_warp(x: u32) -> u32 {
        let w = 32u32;
        ((x + (w - 1)) / w).max(1) * w
    }

    #[inline]
    fn warps_per_block(block_x: u32) -> u32 {
        (block_x / 32).max(1)
    }

    #[inline]
    fn chunk_size_for_batch(n_rows: usize, len: usize) -> usize {
        // Allow large grid.x (limit ~2.147e9). Keep a modest VRAM guard
        // and cap by memory-per-combo (4 planes: dvdi, fast, slow, center).
        let max_grid_x = 2_147_000_000usize; // below u32::MAX for safety
        let per_combo_bytes = 4usize * len * std::mem::size_of::<f32>();

        let mut max_by_mem = n_rows;
        if let Ok((free, _)) = mem_get_info() {
            let headroom = 64usize << 20; // ~64MB headroom
            if free > (per_combo_bytes + headroom) {
                let budget = free - headroom;
                max_by_mem = (budget / per_combo_bytes).max(1);
            } else {
                max_by_mem = 1;
            }
        }
        max_by_mem.min(max_grid_x).max(1)
    }

    #[inline]
    fn will_fit(&self, required: usize, headroom: usize) -> Result<(), CudaDvdiqqeError> {
        match mem_get_info() {
            Ok((free, _)) => {
                if required.saturating_add(headroom) > free {
                    return Err(CudaDvdiqqeError::OutOfMemory { required, free, headroom });
                }
                Ok(())
            }
            Err(e) => Err(CudaDvdiqqeError::Cuda(e)),
        }
    }

    pub fn dvdiqqe_batch_dev(
        &self,
        open: &[f32],
        close: &[f32],
        volume: Option<&[f32]>,
        sweep: &DvdiqqeBatchRange,
        volume_type: &str,
        center_type: &str,
        tick_size: f32,
    ) -> Result<DeviceDvdiqqeQuad, CudaDvdiqqeError> {
        if open.len() != close.len() {
            return Err(CudaDvdiqqeError::InvalidInput("open/close length mismatch".into()));
        }
        if let Some(v) = volume {
            if v.len() != close.len() {
                return Err(CudaDvdiqqeError::InvalidInput("volume length mismatch".into()));
            }
        }
        let len = close.len();
        if len == 0 { return Err(CudaDvdiqqeError::InvalidInput("empty series".into())); }

        // Find first finite close
        let first_valid = match close.iter().position(|x| x.is_finite()) {
            Some(i) => i,
            None => return Err(CudaDvdiqqeError::InvalidInput("all NaN close".into())),
        };

        // Build parameter grid like CPU path
        let (p_start, p_end, p_step) = sweep.period;
        let (s_start, s_end, s_step) = sweep.smoothing_period;
        let (f_start, f_end, f_step) = sweep.fast_multiplier;
        let (sl_start, sl_end, sl_step) = sweep.slow_multiplier;
        let mut periods = Vec::<i32>::new();
        let mut smoothings = Vec::<i32>::new();
        let mut fasts = Vec::<f32>::new();
        let mut slows = Vec::<f32>::new();
        let mut n_combos = 0usize;
        // robust range expansion: zero step => static; reversed bounds supported
        let mut push_axis_usize = |start: usize, end: usize, step: usize, dst: &mut Vec<i32>| {
            if step == 0 || start == end { dst.push(start as i32); return; }
            if start < end {
                let mut cur = start; while cur <= end { dst.push(cur as i32); cur = cur.saturating_add(step); }
            } else {
                let mut cur = start; while cur >= end { dst.push(cur as i32); if cur < step { break; } cur -= step; if cur == usize::MAX { break; } }
            }
        };
        let mut push_axis_f64 = |start: f64, end: f64, step: f64, dst: &mut Vec<f32>| {
            if step == 0.0 || start == end { dst.push(start as f32); return; }
            if start < end { let mut v = start; while v <= end + 1e-12 { dst.push(v as f32); v += step; } }
            else { let mut v = start; let d = step.abs(); while v >= end - 1e-12 { dst.push(v as f32); v -= d; } }
        };
        let (p_start, p_end, p_step) = sweep.period; let (s_start, s_end, s_step) = sweep.smoothing_period;
        let (f_start, f_end, f_step) = sweep.fast_multiplier; let (sl_start, sl_end, sl_step) = sweep.slow_multiplier;
        let mut pvec = Vec::<i32>::new(); push_axis_usize(p_start, p_end, p_step, &mut pvec);
        let mut svec = Vec::<i32>::new(); push_axis_usize(s_start, s_end, s_step, &mut svec);
        let mut fvec = Vec::<f32>::new(); push_axis_f64(f_start, f_end, f_step, &mut fvec);
        let mut slvec = Vec::<f32>::new(); push_axis_f64(sl_start, sl_end, sl_step, &mut slvec);
        for &p in &pvec { for &s in &svec { for &f in &fvec { for &sl in &slvec {
            periods.push(p); smoothings.push(s); fasts.push(f); slows.push(sl); n_combos += 1;
        }}}}
        if n_combos == 0 {
            return Err(CudaDvdiqqeError::InvalidInput("empty sweep".into()));
        }
        // checked sizes
        let plane = n_combos
            .checked_mul(len)
            .ok_or_else(|| CudaDvdiqqeError::InvalidInput("n_combos*len overflow".into()))?;
        // VRAM budget check (4 planes + params)
        let bytes_out = plane
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaDvdiqqeError::InvalidInput("bytes overflow".into()))?;
        let required = 4usize
            .checked_mul(bytes_out)
            .ok_or_else(|| CudaDvdiqqeError::InvalidInput("bytes overflow".into()))?
            .saturating_add(periods.len() * std::mem::size_of::<i32>())
            .saturating_add(smoothings.len() * std::mem::size_of::<i32>())
            .saturating_add(fasts.len() * std::mem::size_of::<f32>())
            .saturating_add(slows.len() * std::mem::size_of::<f32>());
        self.will_fit(required, 64usize << 20)?;

        // Device inputs
        let d_open: DeviceBuffer<f32> = DeviceBuffer::from_slice(open)?;
        let d_close: DeviceBuffer<f32> = DeviceBuffer::from_slice(close)?;
        let d_vol: Option<DeviceBuffer<f32>> = if let Some(v) = volume { Some(DeviceBuffer::from_slice(v)?) } else { None };
        let has_volume = d_vol.is_some() as i32;
        let d_periods: DeviceBuffer<i32> = DeviceBuffer::from_slice(&periods)?;
        let d_smooths: DeviceBuffer<i32> = DeviceBuffer::from_slice(&smoothings)?;
        let d_fasts: DeviceBuffer<f32> = DeviceBuffer::from_slice(&fasts)?;
        let d_slows: DeviceBuffer<f32> = DeviceBuffer::from_slice(&slows)?;

        // Outputs
        let mut d_dvdi: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(plane) }?;
        let mut d_fast: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(plane) }?;
        let mut d_slow: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(plane) }?;
        let mut d_cent: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(plane) }?;

        // Kernel
        let func = self
            .module
            .get_function("dvdiqqe_batch_f32")
            .map_err(|_| CudaDvdiqqeError::MissingKernelSymbol { name: "dvdiqqe_batch_f32" })?;
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => Self::align_to_warp(block_x).max(32),
            BatchKernelPolicy::Auto => 32, // one warp per param row; lane 0 scans, others help NaN warmup
        };
        let chunk = Self::chunk_size_for_batch(n_combos, len);
        let mut launched = 0usize;
        while launched < n_combos {
            let cur = (n_combos - launched).min(chunk);
            let gx = cur as u32; let gy = 1u32; let gz = 1u32;
            if gx == 0 || block_x == 0 { return Err(CudaDvdiqqeError::LaunchConfigTooLarge { gx, gy, gz, bx: block_x, by: 1, bz: 1 }); }
            let grid: GridSize = (gx, gy, gz).into();
            let block: BlockSize = (block_x, 1, 1).into();
            unsafe {
                let mut open_ptr = d_open.as_device_ptr().as_raw();
                let mut close_ptr = d_close.as_device_ptr().as_raw();
                // Honor tick-only volume mode
                let use_tick_only = volume_type.eq_ignore_ascii_case("tick-only")
                    || volume_type.eq_ignore_ascii_case("tick_only")
                    || volume_type.eq_ignore_ascii_case("tick");
                let mut vol_ptr: u64 = if use_tick_only {
                    0u64
                } else if let Some(ref dv) = d_vol {
                    dv.as_device_ptr().as_raw()
                } else {
                    0u64
                };
                let mut has_vol_i = if use_tick_only {
                    0i32
                } else {
                    has_volume as i32
                };
                let mut per_ptr = d_periods
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * std::mem::size_of::<i32>()) as u64);
                let mut sm_ptr = d_smooths
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * std::mem::size_of::<i32>()) as u64);
                let mut fa_ptr = d_fasts
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * std::mem::size_of::<f32>()) as u64);
                let mut sl_ptr = d_slows
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * std::mem::size_of::<f32>()) as u64);
                let mut ncomb_i = cur as i32;
                let mut len_i = len as i32;
                let mut fv_i = first_valid as i32;
                let mut tick_f = tick_size as f32;
                // Note: current CUDA kernel assumes center_type is constant across rows; 1=dynamic, 0=static
                let mut center_dyn = if center_type.eq_ignore_ascii_case("dynamic") {
                    1i32
                } else {
                    0i32
                };
                let mut dvdi_ptr = d_dvdi
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * len * std::mem::size_of::<f32>()) as u64);
                let mut fast_ptr = d_fast
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * len * std::mem::size_of::<f32>()) as u64);
                let mut slow_ptr = d_slow
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * len * std::mem::size_of::<f32>()) as u64);
                let mut cent_ptr = d_cent
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * len * std::mem::size_of::<f32>()) as u64);
                let mut center_dyn = if center_type.eq_ignore_ascii_case("dynamic") {
                    1i32
                } else {
                    0i32
                };
                let mut dvdi_ptr = d_dvdi
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * len * std::mem::size_of::<f32>()) as u64);
                let mut fast_ptr = d_fast
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * len * std::mem::size_of::<f32>()) as u64);
                let mut slow_ptr = d_slow
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * len * std::mem::size_of::<f32>()) as u64);
                let mut cent_ptr = d_cent
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * len * std::mem::size_of::<f32>()) as u64);
                let args: &mut [*mut c_void] = &mut [
                    &mut open_ptr as *mut _ as *mut c_void,
                    &mut close_ptr as *mut _ as *mut c_void,
                    &mut vol_ptr as *mut _ as *mut c_void,
                    &mut has_vol_i as *mut _ as *mut c_void,
                    &mut per_ptr as *mut _ as *mut c_void,
                    &mut sm_ptr as *mut _ as *mut c_void,
                    &mut fa_ptr as *mut _ as *mut c_void,
                    &mut sl_ptr as *mut _ as *mut c_void,
                    &mut ncomb_i as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut fv_i as *mut _ as *mut c_void,
                    &mut tick_f as *mut _ as *mut c_void,
                    &mut center_dyn as *mut _ as *mut c_void,
                    &mut dvdi_ptr as *mut _ as *mut c_void,
                    &mut fast_ptr as *mut _ as *mut c_void,
                    &mut slow_ptr as *mut _ as *mut c_void,
                    &mut cent_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(CudaDvdiqqeError::Cuda)?;
            }
            launched += cur;
        }
        self.stream
            .synchronize()
            .map_err(CudaDvdiqqeError::Cuda)?;

        Ok(DeviceDvdiqqeQuad {
            dvdi: DeviceArrayF32 {
                buf: d_dvdi,
                rows: n_combos,
                cols: len,
            },
            fast: DeviceArrayF32 {
                buf: d_fast,
                rows: n_combos,
                cols: len,
            },
            slow: DeviceArrayF32 {
                buf: d_slow,
                rows: n_combos,
                cols: len,
            },
            center: DeviceArrayF32 {
                buf: d_cent,
                rows: n_combos,
                cols: len,
            },
            ctx: self.ctx.clone(),
            device_id: self.device_id,
        })
    }

    pub fn dvdiqqe_many_series_one_param_time_major_dev(
        &self,
        open_tm: &[f32],
        close_tm: &[f32],
        volume_tm: Option<&[f32]>,
        cols: usize,
        rows: usize,
        period: usize,
        smoothing: usize,
        fast_mult: f32,
        slow_mult: f32,
        volume_type: &str,
        center_type: &str,
        tick_size: f32,
    ) -> Result<DeviceDvdiqqeQuad, CudaDvdiqqeError> {
        let n = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaDvdiqqeError::InvalidInput("rows*cols overflow".into()))?;
        if open_tm.len() != n || close_tm.len() != n {
            return Err(CudaDvdiqqeError::InvalidInput(
                "time-major open/close mismatch".into(),
            ));
        }
        if let Some(v) = volume_tm {
            if v.len() != n {
                return Err(CudaDvdiqqeError::InvalidInput(
                    "time-major volume mismatch".into(),
                ));
            }
        }
        if period == 0 || smoothing == 0 {
            return Err(CudaDvdiqqeError::InvalidInput(
                "period/smoothing must be > 0".into(),
            ));
        }
        let n = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaDvdiqqeError::InvalidInput("rows*cols overflow".into()))?;
        if open_tm.len() != n || close_tm.len() != n {
            return Err(CudaDvdiqqeError::InvalidInput(
                "time-major open/close mismatch".into(),
            ));
        }
        if let Some(v) = volume_tm {
            if v.len() != n {
                return Err(CudaDvdiqqeError::InvalidInput(
                    "time-major volume mismatch".into(),
                ));
            }
        }
        if period == 0 || smoothing == 0 {
            return Err(CudaDvdiqqeError::InvalidInput(
                "period/smoothing must be > 0".into(),
            ));
        }

        // First-valid per series based on close
        let mut first_valids = vec![-1i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                if close_tm[idx].is_finite() && open_tm[idx].is_finite() {
                    first_valids[s] = t as i32;
                    break;
                }
            }
        }
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                if close_tm[idx].is_finite() && open_tm[idx].is_finite() {
                    first_valids[s] = t as i32;
                    break;
                }
            }
        }

        // VRAM fit for outputs
        let bytes = n
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaDvdiqqeError::InvalidInput("bytes overflow".into()))?;
        let required = 4usize
            .checked_mul(bytes)
            .ok_or_else(|| CudaDvdiqqeError::InvalidInput("bytes overflow".into()))?;
        self.will_fit(required, 64usize << 20)?;

        let d_open: DeviceBuffer<f32> = DeviceBuffer::from_slice(open_tm)?;
        let d_close: DeviceBuffer<f32> = DeviceBuffer::from_slice(close_tm)?;
        let d_vol: Option<DeviceBuffer<f32>> = if let Some(v) = volume_tm { Some(DeviceBuffer::from_slice(v)?) } else { None };
        let has_volume = d_vol.is_some() as i32;
        let d_fv: DeviceBuffer<i32> = DeviceBuffer::from_slice(&first_valids)?;

        let mut d_dvdi: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n) }?;
        let mut d_fast: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n) }?;
        let mut d_slow: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n) }?;
        let mut d_cent: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n) }?;

        let func = self
            .module
            .get_function("dvdiqqe_many_series_one_param_f32")
            .map_err(|_| CudaDvdiqqeError::MissingKernelSymbol { name: "dvdiqqe_many_series_one_param_f32" })?;
        let mut block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => Self::align_to_warp(block_x),
            ManySeriesKernelPolicy::Auto => 128, // 4 warps per block
        };
        block_x = block_x.max(32);
        let wpb = Self::warps_per_block(block_x);
        let grid_x = ((cols as u32) + wpb - 1) / wpb; // 1 warp -> 1 series
        if grid_x == 0 || block_x == 0 { return Err(CudaDvdiqqeError::LaunchConfigTooLarge { gx: grid_x, gy: 1, gz: 1, bx: block_x, by: 1, bz: 1 }); }
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut open_ptr = d_open.as_device_ptr().as_raw();
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            // Honor tick-only volume mode
            let use_tick_only = volume_type.eq_ignore_ascii_case("tick-only")
                || volume_type.eq_ignore_ascii_case("tick_only")
                || volume_type.eq_ignore_ascii_case("tick");
            let mut vol_ptr: u64 = if use_tick_only {
                0u64
            } else if let Some(ref dv) = d_vol {
                dv.as_device_ptr().as_raw()
            } else {
                0u64
            };
            let mut has_vol_i = if use_tick_only {
                0i32
            } else {
                has_volume as i32
            };
            let mut fv_ptr = d_fv.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut smooth_i = smoothing as i32;
            let mut fast_f = fast_mult as f32;
            let mut slow_f = slow_mult as f32;
            let mut tick_f = tick_size as f32;
            let mut center_dyn = if center_type.eq_ignore_ascii_case("dynamic") {
                1i32
            } else {
                0i32
            };
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut dvdi_ptr = d_dvdi.as_device_ptr().as_raw();
            let mut fast_ptr = d_fast.as_device_ptr().as_raw();
            let mut slow_ptr = d_slow.as_device_ptr().as_raw();
            let mut cent_ptr = d_cent.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut open_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut vol_ptr as *mut _ as *mut c_void,
                &mut has_vol_i as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut smooth_i as *mut _ as *mut c_void,
                &mut fast_f as *mut _ as *mut c_void,
                &mut slow_f as *mut _ as *mut c_void,
                &mut tick_f as *mut _ as *mut c_void,
                &mut center_dyn as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut dvdi_ptr as *mut _ as *mut c_void,
                &mut fast_ptr as *mut _ as *mut c_void,
                &mut slow_ptr as *mut _ as *mut c_void,
                &mut cent_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(CudaDvdiqqeError::Cuda)?;
        }
        self.stream
            .synchronize()
            .map_err(CudaDvdiqqeError::Cuda)?;
        self.stream
            .synchronize()
            .map_err(CudaDvdiqqeError::Cuda)?;

        Ok(DeviceDvdiqqeQuad {
            dvdi: DeviceArrayF32 { buf: d_dvdi, rows, cols },
            fast: DeviceArrayF32 { buf: d_fast, rows, cols },
            slow: DeviceArrayF32 { buf: d_slow, rows, cols },
            center: DeviceArrayF32 { buf: d_cent, rows, cols },
            ctx: self.ctx.clone(),
            device_id: self.device_id,
        })
    }
}

// ---------------- Benches ----------------
#[cfg(not(test))]
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    struct DvdiqqeBatchState {
        cuda: CudaDvdiqqe,
        open: Vec<f32>,
        close: Vec<f32>,
        volume: Vec<f32>,
        sweep: DvdiqqeBatchRange,
        vt: String,
        ct: String,
        tick: f32,
    }
    impl CudaBenchState for DvdiqqeBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .dvdiqqe_batch_dev(
                    &self.open,
                    &self.close,
                    Some(&self.volume),
                    &self.sweep,
                    &self.vt,
                    &self.ct,
                    self.tick,
                )
                .unwrap();
            let _ = self
                .cuda
                .dvdiqqe_batch_dev(
                    &self.open,
                    &self.close,
                    Some(&self.volume),
                    &self.sweep,
                    &self.vt,
                    &self.ct,
                    self.tick,
                )
                .unwrap();
        }
    }

    struct DvdiqqeManyState {
        cuda: CudaDvdiqqe,
        open_tm: Vec<f32>,
        close_tm: Vec<f32>,
        volume_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        period: usize,
        smoothing: usize,
        fast: f32,
        slow: f32,
        vt: String,
        ct: String,
        tick: f32,
    }
    impl CudaBenchState for DvdiqqeManyState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .dvdiqqe_many_series_one_param_time_major_dev(
                    &self.open_tm,
                    &self.close_tm,
                    Some(&self.volume_tm),
                    self.cols,
                    self.rows,
                    self.period,
                    self.smoothing,
                    self.fast,
                    self.slow,
                    &self.vt,
                    &self.ct,
                    self.tick,
                )
                .unwrap();
            let _ = self
                .cuda
                .dvdiqqe_many_series_one_param_time_major_dev(
                    &self.open_tm,
                    &self.close_tm,
                    Some(&self.volume_tm),
                    self.cols,
                    self.rows,
                    self.period,
                    self.smoothing,
                    self.fast,
                    self.slow,
                    &self.vt,
                    &self.ct,
                    self.tick,
                )
                .unwrap();
        }
    }

    fn synth_ohlcv(len: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let close = gen_series(len);
        let mut open = close.clone();
        let mut vol = vec![0f32; len];
        for i in 0..len {
            let x = i as f32 * 0.0023;
            open[i] = close[i] - 0.15 + (0.03 * x).sin();
            vol[i] = (0.5 + (x * 0.77).cos().abs()).max(0.0);
        }
        (open, close, vol)
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let len = 512_000usize;
        let (open, close, volume) = synth_ohlcv(len);
        let sweep = DvdiqqeBatchRange {
            period: (10, 28, 3),
            smoothing_period: (3, 9, 3),
            fast_multiplier: (1.5, 3.0, 0.5),
            slow_multiplier: (3.0, 6.0, 1.0),
        };
        Box::new(DvdiqqeBatchState {
            cuda: CudaDvdiqqe::new(0).unwrap(),
            open,
            close,
            volume,
            sweep,
            vt: "default".into(),
            ct: "dynamic".into(),
            tick: 0.01,
        })
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let (cols, rows) = (128usize, 262_144usize);
        let mut close_tm = vec![f32::NAN; cols * rows];
        let mut open_tm = vec![f32::NAN; cols * rows];
        let mut vol_tm = vec![0f32; cols * rows];
        for s in 0..cols {
            for t in s..rows {
                let x = (t as f32) + (s as f32) * 0.2;
                let c = (x * 0.0017).sin() + 0.00012 * x;
                close_tm[t * cols + s] = c;
                open_tm[t * cols + s] = c - 0.12 + (0.03 * x).cos();
                vol_tm[t * cols + s] = (0.4 + (x * 0.77).cos().abs()).max(0.0);
            }
        }
        Box::new(DvdiqqeManyState {
            cuda: CudaDvdiqqe::new(0).unwrap(),
            open_tm,
            close_tm,
            volume_tm: vol_tm,
            cols,
            rows,
            period: 13,
            smoothing: 6,
            fast: 2.618,
            slow: 4.236,
            vt: "default".into(),
            ct: "dynamic".into(),
            tick: 0.01,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        let scen_batch = CudaBenchScenario::new(
            "dvdiqqe",
            "one_series_many_params",
            "dvdiqqe_cuda_batch_dev",
            "512k_x_params",
            prep_one_series_many_params,
        );
        let scen_many = CudaBenchScenario::new(
            "dvdiqqe",
            "many_series_one_param",
            "dvdiqqe_cuda_many_series_one_param_dev",
            "128x262k",
            prep_many_series_one_param,
        );
        vec![scen_batch, scen_many]
    }
}
