//! CUDA support for Percentage Price Oscillator (PPO).
//!
//! Parity with scalar PPO (see `src/indicators/ppo.rs`):
//! - Warmup index per row/series: `first_valid + slow - 1` is the first output
//! - Before warmup, outputs are NaN
//! - Division-by-zero yields NaN (matches scalar checks on slow MA == 0)
//! - SMA path uses host prefix sums (double) to do O(1) window ops
//! - EMA path uses per-row/per-series sequential recurrence with classic seeding
//!
//! ALMA-style wrapper behavior:
//! - PTX is embedded via OUT_DIR, JIT opts: DetermineTargetFromContext + OptLevel O2 with fallbacks
//! - Stream NON_BLOCKING
//! - VRAM estimates with ~64MB headroom and grid.y chunking (<= 65,535)

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::{CudaEma, CudaSma};
use crate::cuda::moving_averages::CudaEmaError;
use crate::indicators::moving_averages::ema::EmaParams;
use crate::cuda::moving_averages::ema_wrapper::{
    BatchKernelPolicy as EmaBatchKernelPolicy, CudaEmaPolicy as EmaCudaPolicy,
    ManySeriesKernelPolicy as EmaManySeriesKernelPolicy,
};

use crate::cuda::moving_averages::CudaSmaError;
use crate::indicators::moving_averages::sma::{SmaBatchRange, SmaParams};
use crate::indicators::ppo::{PpoBatchRange, PpoParams};

use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CudaPpoError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error(transparent)]
    Ema(#[from] CudaEmaError),
    #[error(transparent)]
    Sma(#[from] CudaSmaError),
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

/// VRAM-backed array for PPO with context guard and device id.
pub struct DeviceArrayF32Ppo {
    pub buf: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
    pub ctx: Arc<Context>,
    pub device_id: u32,
}

struct EmaSurfacesF32 {
    out: DeviceBuffer<f32>,
    
    _periods: DeviceBuffer<i32>,
}

impl DeviceArrayF32Ppo {
    #[inline]
    pub fn device_ptr(&self) -> u64 {
        self.buf.as_device_ptr().as_raw() as u64
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.rows.saturating_mul(self.cols)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    /// One-dimensional time parallel for SMA; EMA uses one block per row and thread 0 runs the scan.
    OneD {
        block_x: u32,
    },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    /// 2D grid (x=time, y=series) for SMA; EMA uses 1 thread along x per series
    Tiled2D {
        tx: u32,
        ty: u32,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaPpoPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaPpoPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

pub struct CudaPpo {
    module: Module,
    ema_module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaPpoPolicy,
    
    ema: CudaEma,
    sma: CudaSma,
}

impl CudaPpo {
    pub fn new(device_id: usize) -> Result<Self, CudaPpoError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/ppo_kernel.ptx"));
        let module = Module::from_ptx(
            ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O2),
            ],
        )
        .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
        .or_else(|_| Module::from_ptx(ptx, &[]))?;

        let ema_ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/ema_kernel.ptx"));
        let ema_module = Module::from_ptx(
            ema_ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O2),
            ],
        )
        .or_else(|_| Module::from_ptx(ema_ptx, &[ModuleJitOption::DetermineTargetFromContext]))
        .or_else(|_| Module::from_ptx(ema_ptx, &[]))?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        
        
        let mut ema_policy = EmaCudaPolicy::default();
        ema_policy.batch = EmaBatchKernelPolicy::Plain { block_x: 16 };
        ema_policy.many_series = EmaManySeriesKernelPolicy::Auto;
        let ema = CudaEma::new_with_policy(device_id, ema_policy)?;
        let sma = CudaSma::new(device_id)?;

        Ok(Self {
            module,
            ema_module,
            stream,
            context,
            device_id: device_id as u32,
            policy: CudaPpoPolicy::default(),
            ema,
            sma,
        })
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> {
        self.context.clone()
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaPpoError> {
        self.stream.synchronize().map_err(Into::into)
    }

    #[inline]
    fn will_fit(&self, required: usize, headroom: usize) -> Result<(), CudaPpoError> {
        match mem_get_info() {
            Ok((free, _total)) => {
                if required.saturating_add(headroom) > free {
                    return Err(CudaPpoError::OutOfMemory {
                        required,
                        free,
                        headroom,
                    });
                }
                Ok(())
            }
            Err(e) => Err(CudaPpoError::Cuda(e)),
        }
    }

    fn validate_launch(
        &self,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    ) -> Result<(), CudaPpoError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)?
            .max(1) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)?
            .max(1) as u32;
        let max_grid_y = device
            .get_attribute(DeviceAttribute::MaxGridDimY)?
            .max(1) as u32;
        let max_grid_z = device
            .get_attribute(DeviceAttribute::MaxGridDimZ)?
            .max(1) as u32;

        let threads_per_block = bx
            .saturating_mul(by)
            .saturating_mul(bz);
        if threads_per_block > max_threads
            || gx > max_grid_x
            || gy > max_grid_y
            || gz > max_grid_z
        {
            return Err(CudaPpoError::LaunchConfigTooLarge {
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

    fn ema_surfaces_one_series_f32(
        &self,
        d_prices: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: i32,
        periods: &[i32],
    ) -> Result<EmaSurfacesF32, CudaPpoError> {
        if series_len == 0 {
            return Err(CudaPpoError::InvalidInput("empty series".into()));
        }
        if periods.is_empty() {
            return Err(CudaPpoError::InvalidInput("empty period list".into()));
        }
        let fv = first_valid.max(0) as usize;
        if fv >= series_len {
            return Err(CudaPpoError::InvalidInput(
                "first_valid out of range".into(),
            ));
        }
        let remaining = series_len - fv;

        for &p in periods {
            if p <= 0 {
                return Err(CudaPpoError::InvalidInput(
                    "period must be positive".into(),
                ));
            }
            let up = p as usize;
            if remaining < up {
                return Err(CudaPpoError::InvalidInput(format!(
                    "not enough valid data: need {} valid samples, have {}",
                    up, remaining
                )));
            }
        }

        let d_periods: DeviceBuffer<i32> = DeviceBuffer::from_slice(periods)?;
        let out_elems = periods
            .len()
            .checked_mul(series_len)
            .ok_or_else(|| CudaPpoError::InvalidInput("surfaces bytes overflow".into()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(out_elems) }?;

        let func = self
            .ema_module
            .get_function("ema_batch_f64_to_f32")
            .map_err(|_| CudaPpoError::MissingKernelSymbol { name: "ema_batch_f64_to_f32" })?;

        
        
        let block_x = 32u32;

        
        let device = Device::get_device(self.device_id)?;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)?
            .max(1) as usize;
        let cap = max_grid_x.max(1);
        let stream = &self.stream;

        let mut start = 0usize;
        while start < periods.len() {
            let count = (periods.len() - start).min(cap);
            let grid: GridSize = (count as u32, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            self.validate_launch(count as u32, 1, 1, block_x, 1, 1)?;

            let out_ptr = unsafe { d_out.as_device_ptr().add(start * series_len) };
            let periods_ptr = unsafe { d_periods.as_device_ptr().add(start) };

            unsafe {
                launch!(
                    func<<<grid, block, 0, stream>>>(
                        d_prices.as_device_ptr(),
                        periods_ptr,
                        series_len as i32,
                        first_valid,
                        count as i32,
                        out_ptr
                    )
                )?;
            }
            start = start.saturating_add(count);
        }

        Ok(EmaSurfacesF32 {
            out: d_out,
            _periods: d_periods,
        })
    }

    pub fn set_policy(&mut self, p: CudaPpoPolicy) {
        self.policy = p;
    }
    pub fn policy(&self) -> &CudaPpoPolicy {
        &self.policy
    }

    
    pub fn ppo_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &PpoBatchRange,
    ) -> Result<(DeviceArrayF32Ppo, Vec<PpoParams>), CudaPpoError> {
        let len = data_f32.len();
        if len == 0 {
            return Err(CudaPpoError::InvalidInput("empty data".into()));
        }

        let (fs, fe, fstep) = sweep.fast_period;
        let (ss, se, sstep) = sweep.slow_period;
        let nf = axis_len(fs, fe, fstep);
        let ns = axis_len(ss, se, sstep);
        if nf == 0 || ns == 0 {
            return Err(CudaPpoError::InvalidInput("empty fast/slow sweep".into()));
        }

        
        let combos: Vec<PpoParams> = expand_grid(sweep);
        let rows = combos.len();
        if rows == 0 {
            return Err(CudaPpoError::InvalidInput("no parameter combos".into()));
        }

        let ma_mode = ma_mode_from(&sweep.ma_type)?;

        
        let elem_f32 = std::mem::size_of::<f32>();
        let elem_i32 = std::mem::size_of::<i32>();
        let elem_f64 = std::mem::size_of::<f64>();
        let prices_bytes = len
            .checked_mul(elem_f32)
            .ok_or_else(|| CudaPpoError::InvalidInput("price bytes overflow".into()))?;
        let params_bytes = rows
            .checked_mul(2usize)
            .and_then(|v| v.checked_mul(elem_i32))
            .ok_or_else(|| CudaPpoError::InvalidInput("params bytes overflow".into()))?;
        
        let prefix_bytes = (len + 1)
            .checked_mul(elem_f64)
            .ok_or_else(|| CudaPpoError::InvalidInput("prefix bytes overflow".into()))?;
        let out_elems = rows
            .checked_mul(len)
            .ok_or_else(|| CudaPpoError::InvalidInput("rows*len overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(elem_f32)
            .ok_or_else(|| CudaPpoError::InvalidInput("output bytes overflow".into()))?;
        
        let surfaces_elems = (nf + ns)
            .checked_mul(len)
            .ok_or_else(|| CudaPpoError::InvalidInput("surfaces elems overflow".into()))?;
        let surfaces_bytes = surfaces_elems
            .checked_mul(elem_f32)
            .ok_or_else(|| CudaPpoError::InvalidInput("surfaces bytes overflow".into()))?;

        let base_required = prices_bytes
            .checked_add(params_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaPpoError::InvalidInput("base bytes overflow".into()))?;
        let required = match ma_mode {
            0 => base_required
                .checked_add(prefix_bytes)
                .ok_or_else(|| CudaPpoError::InvalidInput("total bytes overflow".into()))?,
            1 => base_required
                .checked_add(surfaces_bytes)
                .ok_or_else(|| CudaPpoError::InvalidInput("total bytes overflow".into()))?,
            _ => unreachable!(),
        };
        let headroom = 64usize * 1024 * 1024;
        self.will_fit(required, headroom)?;

        
        let mut fasts_i32 = Vec::with_capacity(rows);
        let mut slows_i32 = Vec::with_capacity(rows);
        for p in &combos {
            fasts_i32.push(p.fast_period.unwrap() as i32);
            slows_i32.push(p.slow_period.unwrap() as i32);
        }

        
        let d_prices: DeviceBuffer<f32> = DeviceBuffer::from_slice(data_f32)?;
        let d_fasts: DeviceBuffer<i32> = DeviceBuffer::from_slice(&fasts_i32)?;
        let d_slows: DeviceBuffer<i32> = DeviceBuffer::from_slice(&slows_i32)?;

        
        let out_elems = rows
            .checked_mul(len)
            .ok_or_else(|| CudaPpoError::InvalidInput("rows*len overflow for d_out".into()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(out_elems) }?;

        
        let first_valid = data_f32
            .iter()
            .position(|v| v.is_finite())
            .unwrap_or(0) as i32;

        match ma_mode {
            
            0 => {
                let prefix = prefix_sum_one_series_f64(data_f32, first_valid);
                let d_prefix: DeviceBuffer<f64> = DeviceBuffer::from_slice(&prefix)?;

                self.launch_batch_kernel(
                    &d_prices,
                    &d_prefix,
                    len as i32,
                    first_valid,
                    &d_fasts,
                    &d_slows,
                    0,
                    rows as i32,
                    &mut d_out,
                )?;
            }
            
            1 => {
                
                
                let warp_coop_enabled = match std::env::var("PPO_EMA_WARP_COOP") {
                    Ok(v) => v == "1" || v.eq_ignore_ascii_case("true"),
                    Err(_) => false,
                };

                let mut used_warp_coop = false;
                if warp_coop_enabled {
                    used_warp_coop = self
                        .launch_batch_ema_manyparams(
                            &d_prices,
                            len as i32,
                            first_valid,
                            &d_fasts,
                            &d_slows,
                            rows as i32,
                            &mut d_out,
                        )
                        .is_ok();
                }

                
                if !used_warp_coop {
                    let fast_periods: Vec<i32> = axis_vals(fs, fe, fstep)
                        .into_iter()
                        .map(|v| v as i32)
                        .collect();
                    let slow_periods: Vec<i32> = axis_vals(ss, se, sstep)
                        .into_iter()
                        .map(|v| v as i32)
                        .collect();

                    let fast_dev = self.ema_surfaces_one_series_f32(
                        &d_prices,
                        len,
                        first_valid,
                        &fast_periods,
                    )?;
                    let slow_dev = self.ema_surfaces_one_series_f32(
                        &d_prices,
                        len,
                        first_valid,
                        &slow_periods,
                    )?;

                    let func = self
                        .module
                        .get_function("ppo_from_ma_batch_f32")
                        .map_err(|_| CudaPpoError::MissingKernelSymbol { name: "ppo_from_ma_batch_f32" })?;
                    let block: BlockSize = (256, 1, 1).into();
                    let grid_x = ((len as u32) + 255) / 256;
                    let d_slow = DeviceBuffer::from_slice(&slow_periods)?;
                    for (start, count) in grid_y_chunks(rows) {
                        let grid: GridSize = (grid_x.max(1), count as u32, 1).into();
                        unsafe {
                            let mut p_fast = fast_dev.out.as_device_ptr().as_raw();
                            let mut p_slow = slow_dev.out.as_device_ptr().as_raw();
                            let mut p_len = len as i32;
                            let mut p_nf = nf as i32;
                            let mut p_ns = ns as i32;
                            let mut p_first = first_valid;
                            let mut p_slow_arr = d_slow.as_device_ptr().as_raw();
                            let mut p_row_start = start as i32;
                            let mut p_out = d_out.as_device_ptr().add(start * len).as_raw();
                            let args: &mut [*mut c_void] = &mut [
                                &mut p_fast as *mut _ as *mut c_void,
                                &mut p_slow as *mut _ as *mut c_void,
                                &mut p_len as *mut _ as *mut c_void,
                                &mut p_nf as *mut _ as *mut c_void,
                                &mut p_ns as *mut _ as *mut c_void,
                                &mut p_first as *mut _ as *mut c_void,
                                &mut p_slow_arr as *mut _ as *mut c_void,
                                &mut p_row_start as *mut _ as *mut c_void,
                                &mut p_out as *mut _ as *mut c_void,
                            ];
                            self.validate_launch(grid_x.max(1), count as u32, 1, 256, 1, 1)?;
                            self.stream.launch(&func, grid, block, 0, args)?;
                        }
                    }
                }
            }
            _ => unreachable!(),
        }

        
        self.synchronize()?;

        Ok((
            DeviceArrayF32Ppo {
                buf: d_out,
                rows,
                cols: len,
                ctx: self.context_arc(),
                device_id: self.device_id,
            },
            combos,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_prefix: &DeviceBuffer<f64>,
        len: i32,
        first_valid: i32,
        d_fasts: &DeviceBuffer<i32>,
        d_slows: &DeviceBuffer<i32>,
        ma_mode: i32,
        n_combos: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaPpoError> {
        if len <= 0 || n_combos <= 0 {
            return Ok(());
        }
        let func = self
            .module
            .get_function("ppo_batch_f32")
            .map_err(|_| CudaPpoError::MissingKernelSymbol { name: "ppo_batch_f32" })?;

        let mut block_x = match self.policy.batch {
            BatchKernelPolicy::OneD { block_x } if block_x > 0 => block_x,
            _ => 256u32,
        };
        if block_x == 0 {
            return Err(CudaPpoError::InvalidPolicy("block_x must be > 0"));
        }

        let grid_x = if ma_mode == 0 {
            ((len as u32) + block_x - 1) / block_x
        } else {
            1
        };
        let gx = grid_x.max(1);

        
        for (start, count) in grid_y_chunks(n_combos as usize) {
            let gy = count as u32;
            self.validate_launch(gx, gy, 1, block_x, 1, 1)?;
            let grid_launch: GridSize = (gx, gy, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            unsafe {
                let mut p_prices = d_prices.as_device_ptr().as_raw();
                let mut p_prefix = d_prefix.as_device_ptr().as_raw();
                let mut p_len = len;
                let mut p_first = first_valid;
                let mut p_fasts = d_fasts.as_device_ptr().add(start).as_raw();
                let mut p_slows = d_slows.as_device_ptr().add(start).as_raw();
                let mut p_mode = ma_mode;
                let mut p_n = count as i32;
                let mut p_out = d_out.as_device_ptr().add(start * (len as usize)).as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut p_prices as *mut _ as *mut c_void,
                    &mut p_prefix as *mut _ as *mut c_void,
                    &mut p_len as *mut _ as *mut c_void,
                    &mut p_first as *mut _ as *mut c_void,
                    &mut p_fasts as *mut _ as *mut c_void,
                    &mut p_slows as *mut _ as *mut c_void,
                    &mut p_mode as *mut _ as *mut c_void,
                    &mut p_n as *mut _ as *mut c_void,
                    &mut p_out as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid_launch, block, 0, args)?;
            }
        }

        Ok(())
    }

    
    pub fn ppo_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &PpoParams,
    ) -> Result<DeviceArrayF32Ppo, CudaPpoError> {
        if cols == 0 || rows == 0 {
            return Err(CudaPpoError::InvalidInput("empty dims".into()));
        }
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaPpoError::InvalidInput("cols*rows overflow".into()))?;
        if data_tm_f32.len() != elems {
            return Err(CudaPpoError::InvalidInput(
                "length mismatch for time-major input".into(),
            ));
        }

        let fast = params.fast_period.unwrap_or(12) as i32;
        let slow = params.slow_period.unwrap_or(26) as i32;
        let ma_mode = ma_mode_from(params.ma_type.as_deref().unwrap_or("sma"))?;

        
        let elem_f32 = std::mem::size_of::<f32>();
        let elem_i32 = std::mem::size_of::<i32>();
        let elem_f64 = std::mem::size_of::<f64>();
        let price_bytes = elems
            .checked_mul(elem_f32)
            .ok_or_else(|| CudaPpoError::InvalidInput("price_tm bytes overflow".into()))?;
        let first_bytes = cols
            .checked_mul(elem_i32)
            .ok_or_else(|| CudaPpoError::InvalidInput("first_valids bytes overflow".into()))?;
        let prefix_bytes = if ma_mode == 0 {
            elems
                .checked_add(1)
                .and_then(|v| v.checked_mul(elem_f64))
                .ok_or_else(|| CudaPpoError::InvalidInput("prefix_tm bytes overflow".into()))?
        } else {
            elem_f64
        };
        let out_bytes = elems
            .checked_mul(elem_f32)
            .ok_or_else(|| CudaPpoError::InvalidInput("out_tm bytes overflow".into()))?;
        let required = price_bytes
            .checked_add(first_bytes)
            .and_then(|v| v.checked_add(prefix_bytes))
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaPpoError::InvalidInput("total bytes overflow".into()))?;
        let headroom = 64usize * 1024 * 1024;
        self.will_fit(required, headroom)?;

        
        let d_prices_tm: DeviceBuffer<f32> = DeviceBuffer::from_slice(data_tm_f32)?;

        
        let mut first_valids = vec![rows as i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let v = data_tm_f32[t * cols + s];
                if v.is_finite() {
                    first_valids[s] = t as i32;
                    break;
                }
            }
        }
        let d_first = DeviceBuffer::from_slice(&first_valids)?;

        
        let d_prefix_tm: DeviceBuffer<f64> = if ma_mode == 0 {
            let prefix = prefix_sum_time_major_f64(data_tm_f32, cols, rows, &first_valids)?;
            DeviceBuffer::from_slice(&prefix)?
        } else {
            DeviceBuffer::from_slice(&[0.0f64])?
        };

        
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }?;

        
        
        let attempted = self.launch_many_series_kernel(
            &d_prices_tm,
            &d_prefix_tm,
            &d_first,
            cols as i32,
            rows as i32,
            fast,
            slow,
            ma_mode,
            &mut d_out,
        );
        if let Err(err) = attempted {
            eprintln!(
                "[ppo] direct many-series kernel failed ({}); falling back to MA surfaces",
                err
            );
            
            match ma_mode {
                0 => {
                    let sma = &self.sma;
                    let pfast = SmaParams {
                        period: Some(fast as usize),
                    };
                    let pslow = SmaParams {
                        period: Some(slow as usize),
                    };
                    let fast_dev = sma
                        .sma_multi_series_one_param_time_major_dev(
                            data_tm_f32, cols, rows, &pfast,
                        )?;
                    let slow_dev = sma
                        .sma_multi_series_one_param_time_major_dev(
                            data_tm_f32, cols, rows, &pslow,
                        )?;
                    
                    let func = self
                        .module
                        .get_function(
                            "ppo_from_ma_many_series_one_param_time_major_f32",
                        )
                        .map_err(|_| CudaPpoError::MissingKernelSymbol {
                            name: "ppo_from_ma_many_series_one_param_time_major_f32",
                        })?;
                    let tx = 256u32;
                    let ty = 1u32;
                    let grid_x = ((rows as u32) + tx - 1) / tx;
                    let grid_y = ((cols as u32) + ty - 1) / ty;
                    let gx = grid_x.max(1);
                    let gy = grid_y.max(1);
                    self.validate_launch(gx, gy, 1, tx, ty, 1)?;
                    let grid: GridSize = (gx, gy, 1).into();
                    let block: BlockSize = (tx, ty, 1).into();
                    unsafe {
                        let mut p_fast = fast_dev.buf.as_device_ptr().as_raw();
                        let mut p_slow = slow_dev.buf.as_device_ptr().as_raw();
                        let mut p_cols = cols as i32;
                        let mut p_rows = rows as i32;
                        let mut p_first = d_first.as_device_ptr().as_raw();
                        let mut p_slowp = slow as i32;
                        let mut p_out = d_out.as_device_ptr().as_raw();
                        let args: &mut [*mut c_void] = &mut [
                            &mut p_fast as *mut _ as *mut c_void,
                            &mut p_slow as *mut _ as *mut c_void,
                            &mut p_cols as *mut _ as *mut c_void,
                            &mut p_rows as *mut _ as *mut c_void,
                            &mut p_first as *mut _ as *mut c_void,
                            &mut p_slowp as *mut _ as *mut c_void,
                            &mut p_out as *mut _ as *mut c_void,
                        ];
                        self.stream.launch(&func, grid, block, 0, args)?;
                    }
                }
                1 => {
                    let ema = &self.ema;
                    let pfast = EmaParams {
                        period: Some(fast as usize),
                    };
                    let pslow = EmaParams {
                        period: Some(slow as usize),
                    };
                    let fast_dev = ema
                        .ema_many_series_one_param_time_major_dev(
                            data_tm_f32, cols, rows, &pfast,
                        )?;
                    let slow_dev = ema
                        .ema_many_series_one_param_time_major_dev(
                            data_tm_f32, cols, rows, &pslow,
                        )?;
                    
                    let func = self
                        .module
                        .get_function(
                            "ppo_from_ma_many_series_one_param_time_major_f32",
                        )
                        .map_err(|_| CudaPpoError::MissingKernelSymbol {
                            name: "ppo_from_ma_many_series_one_param_time_major_f32",
                        })?;
                    let tx = 256u32;
                    let ty = 1u32;
                    let grid_x = ((rows as u32) + tx - 1) / tx;
                    let grid_y = ((cols as u32) + ty - 1) / ty;
                    let gx = grid_x.max(1);
                    let gy = grid_y.max(1);
                    self.validate_launch(gx, gy, 1, tx, ty, 1)?;
                    let grid: GridSize = (gx, gy, 1).into();
                    let block: BlockSize = (tx, ty, 1).into();
                    unsafe {
                        let mut p_fast = fast_dev.buf.as_device_ptr().as_raw();
                        let mut p_slow = slow_dev.buf.as_device_ptr().as_raw();
                        let mut p_cols = cols as i32;
                        let mut p_rows = rows as i32;
                        let mut p_first = d_first.as_device_ptr().as_raw();
                        let mut p_slowp = slow as i32;
                        let mut p_out = d_out.as_device_ptr().as_raw();
                        let args: &mut [*mut c_void] = &mut [
                            &mut p_fast as *mut _ as *mut c_void,
                            &mut p_slow as *mut _ as *mut c_void,
                            &mut p_cols as *mut _ as *mut c_void,
                            &mut p_rows as *mut _ as *mut c_void,
                            &mut p_first as *mut _ as *mut c_void,
                            &mut p_slowp as *mut _ as *mut c_void,
                            &mut p_out as *mut _ as *mut c_void,
                        ];
                        self.stream.launch(&func, grid, block, 0, args)?;
                    }
                }
                _ => unreachable!(),
            }
        }

        self.synchronize()?;

        Ok(DeviceArrayF32Ppo {
            buf: d_out,
            rows,
            cols,
            ctx: self.context_arc(),
            device_id: self.device_id,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_prefix_tm: &DeviceBuffer<f64>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: i32,
        rows: i32,
        fast: i32,
        slow: i32,
        ma_mode: i32,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaPpoError> {
        if cols <= 0 || rows <= 0 {
            return Ok(());
        }
        let func = self
            .module
            .get_function("ppo_many_series_one_param_time_major_f32")
            .map_err(|_| CudaPpoError::MissingKernelSymbol {
                name: "ppo_many_series_one_param_time_major_f32",
            })?;

        let (tx, ty) = match self.policy.many_series {
            ManySeriesKernelPolicy::Tiled2D { tx, ty } if tx > 0 && ty > 0 => (tx, ty),
            _ => (128u32, 1u32), 
        };
        if tx == 0 || ty == 0 {
            return Err(CudaPpoError::InvalidPolicy("tx, ty must be > 0"));
        }
        let grid_x = ((rows as u32) + tx - 1) / tx;
        let grid_y = ((cols as u32) + ty - 1) / ty;
        let gx = grid_x.max(1);
        let gy = grid_y.max(1);
        self.validate_launch(gx, gy, 1, tx, ty, 1)?;
        let grid: GridSize = (gx, gy, 1).into();
        let block: BlockSize = (tx, ty, 1).into();

        unsafe {
            let mut p_prices = d_prices_tm.as_device_ptr().as_raw();
            let mut p_prefix = d_prefix_tm.as_device_ptr().as_raw();
            let mut p_first = d_first_valids.as_device_ptr().as_raw();
            let mut p_cols = cols;
            let mut p_rows = rows;
            let mut p_fast = fast;
            let mut p_slow = slow;
            let mut p_mode = ma_mode;
            let mut p_out = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_prices as *mut _ as *mut c_void,
                &mut p_prefix as *mut _ as *mut c_void,
                &mut p_first as *mut _ as *mut c_void,
                &mut p_cols as *mut _ as *mut c_void,
                &mut p_rows as *mut _ as *mut c_void,
                &mut p_fast as *mut _ as *mut c_void,
                &mut p_slow as *mut _ as *mut c_void,
                &mut p_mode as *mut _ as *mut c_void,
                &mut p_out as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_ema_manyparams(
        &self,
        d_prices: &DeviceBuffer<f32>,
        len: i32,
        first_valid: i32,
        d_fasts: &DeviceBuffer<i32>,
        d_slows: &DeviceBuffer<i32>,
        n_combos: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaPpoError> {
        if len <= 0 || n_combos <= 0 {
            return Ok(());
        }
        let func = self
            .module
            .get_function("ppo_batch_ema_manyparams_f32")
            .map_err(|_| CudaPpoError::MissingKernelSymbol {
                name: "ppo_batch_ema_manyparams_f32",
            })?;

        
        let mut block_x = match self.policy.batch {
            BatchKernelPolicy::OneD { block_x } if block_x >= 32 => block_x,
            _ => 256u32,
        };
        block_x -= block_x % 32;
        if block_x == 0 {
            block_x = 32;
        }

        let warps_per_block = (block_x / 32) as usize;
        let combos_per_block = warps_per_block * 32;

        
        let total = n_combos as usize;
        let max_rows_per_launch = combos_per_block * 65_535usize;
        let mut start = 0usize;
        while start < total {
            let count = (total - start).min(max_rows_per_launch);
            let gy = div_ceil_u32(count as u32, combos_per_block as u32);
            self.validate_launch(1, gy, 1, block_x, 1, 1)?;
            let grid: GridSize = (1, gy, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();

            unsafe {
                let mut p_prices = d_prices.as_device_ptr().as_raw();
                let mut p_len = len;
                let mut p_first = first_valid;
                let mut p_fasts = d_fasts.as_device_ptr().add(start).as_raw();
                let mut p_slows = d_slows.as_device_ptr().add(start).as_raw();
                let mut p_n = count as i32;
                let mut p_out = d_out.as_device_ptr().add(start * (len as usize)).as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut p_prices as *mut _ as *mut c_void,
                    &mut p_len as *mut _ as *mut c_void,
                    &mut p_first as *mut _ as *mut c_void,
                    &mut p_fasts as *mut _ as *mut c_void,
                    &mut p_slows as *mut _ as *mut c_void,
                    &mut p_n as *mut _ as *mut c_void,
                    &mut p_out as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, 0, args)?;
            }

            start += count;
        }
        Ok(())
    }
}



fn expand_grid(range: &PpoBatchRange) -> Vec<PpoParams> {
    fn axis_u((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        if start <= end {
            (start..=end).step_by(step).collect()
        } else {
            let mut v = Vec::new();
            let mut cur = start;
            loop {
                v.push(cur);
                if cur <= end {
                    break;
                }
                let next = match cur.checked_sub(step) {
                    Some(n) => n,
                    None => break,
                };
                if next < end {
                    break;
                }
                cur = next;
            }
            v
        }
    }

    let fasts = axis_u(range.fast_period);
    let slows = axis_u(range.slow_period);
    let mut out = Vec::with_capacity(fasts.len().saturating_mul(slows.len()));
    for &f in &fasts {
        for &s in &slows {
            out.push(PpoParams {
                fast_period: Some(f),
                slow_period: Some(s),
                ma_type: Some(range.ma_type.clone()),
            });
        }
    }
    out
}

#[inline]
fn div_ceil_u32(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}


#[inline]
fn prefix_sum_one_series_f64(data: &[f32], first_valid: i32) -> Vec<f64> {
    let len = data.len();
    let mut ps = vec![0.0f64; len + 1];
    let mut acc = 0.0f64;
    for i in 0..len {
        if (i as i32) >= first_valid {
            acc += data[i] as f64;
            ps[i + 1] = acc;
        } else {
            ps[i + 1] = 0.0;
        }
    }
    ps
}


#[inline]
fn prefix_sum_time_major_f64(
    data_tm: &[f32],
    cols: usize,
    rows: usize,
    first_valids: &[i32],
) -> Result<Vec<f64>, CudaPpoError> {
    let elems = rows
        .checked_mul(cols)
        .ok_or_else(|| CudaPpoError::InvalidInput("rows*cols overflow in prefix_sum_time_major_f64".into()))?;
    let mut ps = vec![0.0f64; elems + 1];
    for s in 0..cols {
        let fv = first_valids[s] as usize;
        let mut acc = 0.0f64;
        for t in 0..rows {
            let i = t * cols + s;
            if t >= fv {
                acc += data_tm[i] as f64;
            }
            ps[i + 1] = acc;
        }
    }
    Ok(ps)
}

#[inline]
fn ma_mode_from(s: &str) -> Result<i32, CudaPpoError> {
    let sl = s.to_ascii_lowercase();
    match sl.as_str() {
        "sma" => Ok(0),
        "ema" => Ok(1),
        other => Err(CudaPpoError::InvalidInput(format!(
            "unsupported ma_type for CUDA PPO: {}",
            other
        ))),
    }
}

#[inline]
fn grid_y_chunks(total: usize) -> impl Iterator<Item = (usize, usize)> {
    const MAX_Y: usize = 65_535;
    (0..total).step_by(MAX_Y).map(move |start| {
        let len = (total - start).min(MAX_Y);
        (start, len)
    })
}

#[inline]
fn axis_vals(start: usize, end: usize, step: usize) -> Vec<usize> {
    if step == 0 || start == end {
        return vec![start];
    }
    if start <= end {
        (start..=end).step_by(step).collect()
    } else {
        let mut v = Vec::new();
        let mut cur = start;
        loop {
            v.push(cur);
            if cur <= end {
                break;
            }
            let next = match cur.checked_sub(step) {
                Some(n) => n,
                None => break,
            };
            if next < end {
                break;
            }
            cur = next;
        }
        v
    }
}
#[inline]
fn axis_len(start: usize, end: usize, step: usize) -> usize {
    axis_vals(start, end, step).len()
}


pub mod benches {
    use super::*;
    use crate::cuda::{CudaBenchScenario, CudaBenchState};

    const SERIES_LEN: usize = 1_000_000; 
    const FAST_SWEEP: usize = 25;
    const SLOW_SWEEP: usize = 10;
    const MANY_COLS: usize = 250;
    const MANY_ROWS: usize = 1_000_000; 

    fn gen_prices(n: usize) -> Vec<f32> {
        let mut v = vec![f32::NAN; n];
        for i in 10..n {
            let x = i as f32;
            v[i] = (x * 0.00123).sin() + 0.00011 * x;
        }
        v
    }
    fn gen_tm(cols: usize, rows: usize) -> Vec<f32> {
        let mut v = vec![f32::NAN; cols * rows];
        for s in 0..cols {
            for t in (s % 11)..rows {
                let x = (t as f32) + (s as f32) * 0.37;
                v[t * cols + s] = (x * 0.0019).sin() + 0.00021 * x;
            }
        }
        v
    }

    fn bytes_one_series_many() -> usize {
        let len = SERIES_LEN;
        let combos = FAST_SWEEP * SLOW_SWEEP; 
        len * 4 + (len + 1) * 8 + combos * 2 * 4 + combos * len * 4 + 64 * 1024 * 1024
    }
    fn bytes_one_series_many_ema() -> usize {
        let len = SERIES_LEN;
        let nf = FAST_SWEEP;
        let ns = SLOW_SWEEP;
        let combos = nf * ns; 
        let out_bytes = combos * len * 4;
        let surfaces_bytes = (nf + ns) * len * 4;
        len * 4 + combos * 2 * 4 + out_bytes + surfaces_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one() -> usize {
        let elems = MANY_COLS * MANY_ROWS;
        elems * 4 + (elems + 1) * 8 + MANY_COLS * 4 + elems * 4 + 64 * 1024 * 1024
    }

    struct PpoBatchSmaDeviceState {
        cuda: CudaPpo,
        d_prices: DeviceBuffer<f32>,
        d_prefix: DeviceBuffer<f64>,
        d_fasts: DeviceBuffer<i32>,
        d_slows: DeviceBuffer<i32>,
        d_out: DeviceBuffer<f32>,
        len: i32,
        first_valid: i32,
        n_combos: i32,
    }
    impl CudaBenchState for PpoBatchSmaDeviceState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_prices,
                    &self.d_prefix,
                    self.len,
                    self.first_valid,
                    &self.d_fasts,
                    &self.d_slows,
                    0, 
                    self.n_combos,
                    &mut self.d_out,
                )
                .expect("ppo sma batch launch");
            self.cuda.synchronize().expect("ppo sma sync");
        }
    }

    fn prep_batch_sma() -> Box<dyn CudaBenchState> {
        let cuda = CudaPpo::new(0).expect("cuda ppo");
        let data = gen_prices(SERIES_LEN);
        let first_valid = data.iter().position(|v| v.is_finite()).unwrap_or(0) as i32;
        let prefix = prefix_sum_one_series_f64(&data, first_valid);

        let sweep = PpoBatchRange {
            fast_period: (10, 10 + FAST_SWEEP - 1, 1),
            slow_period: (35, 35 + SLOW_SWEEP - 1, 1),
            ma_type: "sma".into(),
        };
        let combos = expand_grid(&sweep);
        let n_combos = combos.len();
        let fasts_i32: Vec<i32> = combos
            .iter()
            .map(|c| c.fast_period.unwrap_or(0) as i32)
            .collect();
        let slows_i32: Vec<i32> = combos
            .iter()
            .map(|c| c.slow_period.unwrap_or(0) as i32)
            .collect();

        let d_prices: DeviceBuffer<f32> = DeviceBuffer::from_slice(&data).expect("d_prices");
        let d_prefix = DeviceBuffer::from_slice(&prefix).expect("d_prefix");
        let d_fasts = DeviceBuffer::from_slice(&fasts_i32).expect("d_fasts");
        let d_slows = DeviceBuffer::from_slice(&slows_i32).expect("d_slows");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * SERIES_LEN) }.expect("d_out");

        cuda.synchronize().expect("sync after prep");
        Box::new(PpoBatchSmaDeviceState {
            cuda,
            d_prices,
            d_prefix,
            d_fasts,
            d_slows,
            d_out,
            len: SERIES_LEN as i32,
            first_valid,
            n_combos: n_combos as i32,
        })
    }

    struct PpoBatchEmaDeviceState {
        cuda: CudaPpo,
        d_prices: DeviceBuffer<f32>,
        d_fasts: DeviceBuffer<i32>,
        d_slows: DeviceBuffer<i32>,
        d_out: DeviceBuffer<f32>,
        len: i32,
        first_valid: i32,
        n_combos: i32,
    }
    impl CudaBenchState for PpoBatchEmaDeviceState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_ema_manyparams(
                    &self.d_prices,
                    self.len,
                    self.first_valid,
                    &self.d_fasts,
                    &self.d_slows,
                    self.n_combos,
                    &mut self.d_out,
                )
                .expect("ppo ema batch launch");
            self.cuda.synchronize().expect("ppo ema sync");
        }
    }

    fn prep_batch_ema() -> Box<dyn CudaBenchState> {
        let cuda = CudaPpo::new(0).expect("cuda ppo");
        let data = gen_prices(SERIES_LEN);
        let first_valid = data.iter().position(|v| v.is_finite()).unwrap_or(0) as i32;

        let sweep = PpoBatchRange {
            fast_period: (10, 10 + FAST_SWEEP - 1, 1),
            slow_period: (35, 35 + SLOW_SWEEP - 1, 1),
            ma_type: "ema".into(),
        };
        let combos = expand_grid(&sweep);
        let n_combos = combos.len();
        let fasts_i32: Vec<i32> = combos
            .iter()
            .map(|c| c.fast_period.unwrap_or(0) as i32)
            .collect();
        let slows_i32: Vec<i32> = combos
            .iter()
            .map(|c| c.slow_period.unwrap_or(0) as i32)
            .collect();

        let d_prices: DeviceBuffer<f32> = DeviceBuffer::from_slice(&data).expect("d_prices");
        let d_fasts = DeviceBuffer::from_slice(&fasts_i32).expect("d_fasts");
        let d_slows = DeviceBuffer::from_slice(&slows_i32).expect("d_slows");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * SERIES_LEN) }.expect("d_out");

        cuda.synchronize().expect("sync after prep");
        Box::new(PpoBatchEmaDeviceState {
            cuda,
            d_prices,
            d_fasts,
            d_slows,
            d_out,
            len: SERIES_LEN as i32,
            first_valid,
            n_combos: n_combos as i32,
        })
    }

    struct PpoManySeriesEmaDeviceState {
        cuda: CudaPpo,
        d_prices_tm: DeviceBuffer<f32>,
        d_prefix_tm: DeviceBuffer<f64>,
        d_first: DeviceBuffer<i32>,
        d_out_tm: DeviceBuffer<f32>,
        cols: i32,
        rows: i32,
        fast: i32,
        slow: i32,
    }
    impl CudaBenchState for PpoManySeriesEmaDeviceState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series_kernel(
                    &self.d_prices_tm,
                    &self.d_prefix_tm,
                    &self.d_first,
                    self.cols,
                    self.rows,
                    self.fast,
                    self.slow,
                    1, 
                    &mut self.d_out_tm,
                )
                .expect("ppo many-series ema launch");
            self.cuda.synchronize().expect("ppo many-series sync");
        }
    }

    fn prep_many_series_ema() -> Box<dyn CudaBenchState> {
        let mut cuda = CudaPpo::new(0).expect("cuda ppo");
        cuda.set_policy(CudaPpoPolicy {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Tiled2D { tx: 128, ty: 4 },
        });

        let cols = MANY_COLS;
        let rows = MANY_ROWS;
        let data_tm = gen_tm(cols, rows);
        let first_valids: Vec<i32> = (0..cols).map(|s| (s % 11) as i32).collect();

        let d_prices_tm: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::from_slice_async(&data_tm, &cuda.stream)
        }
        .expect("d_prices_tm");
        let d_prefix_tm: DeviceBuffer<f64> =
            DeviceBuffer::from_slice(&[0.0f64]).expect("d_prefix_tm");
        let d_first: DeviceBuffer<i32> = DeviceBuffer::from_slice(&first_valids).expect("d_first");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(cols * rows, &cuda.stream) }
                .expect("d_out_tm");

        cuda.synchronize().expect("sync after prep");
        Box::new(PpoManySeriesEmaDeviceState {
            cuda,
            d_prices_tm,
            d_prefix_tm,
            d_first,
            d_out_tm,
            cols: cols as i32,
            rows: rows as i32,
            fast: 12,
            slow: 26,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "ppo",
                "one_series_many_params",
                "ppo_cuda_batch_dev",
                "1m_x_250",
                prep_batch_sma,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many()),
            CudaBenchScenario::new(
                "ppo",
                "one_series_many_params",
                "ppo_cuda_batch_dev",
                "1m_x_250_ema",
                prep_batch_ema,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_ema()),
            CudaBenchScenario::new(
                "ppo",
                "many_series_one_param",
                "ppo_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_ema,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one()),
        ]
    }
}
