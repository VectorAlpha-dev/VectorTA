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

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::cuda::moving_averages::{CudaEma, CudaSma};
use crate::indicators::moving_averages::ema::{EmaBatchRange, EmaParams};
use crate::cuda::moving_averages::{CudaEma, CudaSma};
use crate::indicators::moving_averages::ema::{EmaBatchRange, EmaParams};
use crate::indicators::moving_averages::sma::{SmaBatchRange, SmaParams};
use crate::indicators::ppo::{PpoBatchRange, PpoParams};
use crate::indicators::ppo::{PpoBatchRange, PpoParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaPpoError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaPpoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaPpoError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaPpoError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaPpoError {}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    /// One-dimensional time parallel for SMA; EMA uses one block per row and thread 0 runs the scan.
    OneD {
        block_x: u32,
    },
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
    stream: Stream,
    _context: Context,
    policy: CudaPpoPolicy,
}

impl CudaPpo {
    pub fn new(device_id: usize) -> Result<Self, CudaPpoError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaPpoError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaPpoError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaPpoError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/ppo_kernel.ptx"));
        let module = Module::from_ptx(
            ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O2),
            ],
        )
        .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
        .or_else(|_| Module::from_ptx(ptx, &[]))
        .map_err(|e| CudaPpoError::Cuda(e.to_string()))?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaPpoError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaPpoPolicy::default(),
        })
        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaPpoPolicy::default(),
        })
    }

    pub fn set_policy(&mut self, p: CudaPpoPolicy) {
        self.policy = p;
    }
    pub fn policy(&self) -> &CudaPpoPolicy {
        &self.policy
    }
    pub fn set_policy(&mut self, p: CudaPpoPolicy) {
        self.policy = p;
    }
    pub fn policy(&self) -> &CudaPpoPolicy {
        &self.policy
    }

    // --------------- Batch (one series × many params) ---------------
    pub fn ppo_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &PpoBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<PpoParams>), CudaPpoError> {
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
        if nf == 0 || ns == 0 {
            return Err(CudaPpoError::InvalidInput("empty fast/slow sweep".into()));
        }

        // Materialize combos in canonical order (fast outer × slow inner)
        let combos: Vec<PpoParams> = expand_grid(sweep);
        let rows = combos.len();

        // Period arrays aligned with 'combos'
        let mut fasts_i32 = Vec::with_capacity(rows);
        let mut slows_i32 = Vec::with_capacity(rows);
        for p in &combos {
            fasts_i32.push(p.fast_period.unwrap() as i32);
            slows_i32.push(p.slow_period.unwrap() as i32);
        }

        // Copy inputs once
        let d_prices: DeviceBuffer<f32> =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaPpoError::Cuda(e.to_string()))?;
        let d_fasts: DeviceBuffer<i32> =
            DeviceBuffer::from_slice(&fasts_i32).map_err(|e| CudaPpoError::Cuda(e.to_string()))?;
        let d_slows: DeviceBuffer<i32> =
            DeviceBuffer::from_slice(&slows_i32).map_err(|e| CudaPpoError::Cuda(e.to_string()))?;

        // Output buffer: [rows × len], row‑major
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(rows * len)
        }
        .map_err(|e| CudaPpoError::Cuda(e.to_string()))?;

        // First finite sample index for warmup/prefix
        let first_valid = data_f32
            .iter()
            .position(|v| v.is_finite())
            .unwrap_or(0) as i32;

        let ma_mode = ma_mode_from(&sweep.ma_type)?;
        match ma_mode {
            // SMA path: single FP64 prefix sum + direct kernel (no MA surfaces)
            0 => {
                let prefix = prefix_sum_one_series_f64(data_f32, first_valid);
                let d_prefix: DeviceBuffer<f64> = DeviceBuffer::from_slice(&prefix)
                    .map_err(|e| CudaPpoError::Cuda(e.to_string()))?;

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
            // EMA path: warp‑cooperative kernel (no MA surfaces)
            1 => {
                // Prefer the new warp‑cooperative EMA path; fall back to MA surfaces on failure.
                let attempted = self.launch_batch_ema_manyparams(
                    &d_prices,
                    len as i32,
                    first_valid,
                    &d_fasts,
                    &d_slows,
                    rows as i32,
                    &mut d_out,
                );
                if let Err(err) = attempted {
                    eprintln!(
                        "[ppo] warp-coop EMA launch failed ({}); falling back to MA surfaces",
                        err
                    );
                    // Fallback: build EMA surfaces and use elementwise PPO
                    let (fs, fe, fstep) = sweep.fast_period;
                    let (ss, se, sstep) = sweep.slow_period;
                    let ema = CudaEma::new(0)
                        .map_err(|e| CudaPpoError::Cuda(e.to_string()))?;
                    let fast_dev = ema
                        .ema_batch_dev(
                            data_f32,
                            &EmaBatchRange {
                                period: (fs, fe, fstep),
                            },
                        )
                        .map_err(|e| CudaPpoError::Cuda(e.to_string()))?;
                    let slow_dev = ema
                        .ema_batch_dev(
                            data_f32,
                            &EmaBatchRange {
                                period: (ss, se, sstep),
                            },
                        )
                        .map_err(|e| CudaPpoError::Cuda(e.to_string()))?;

                    let func = self
                        .module
                        .get_function("ppo_from_ma_batch_f32")
                        .map_err(|e| CudaPpoError::Cuda(e.to_string()))?;
                    let block: BlockSize = (256, 1, 1).into();
                    let grid_x = ((len as u32) + 255) / 256;
                    // slow periods array and row chunking
                    let slow_periods: Vec<i32> = axis_vals(ss, se, sstep)
                        .into_iter()
                        .map(|v| v as i32)
                        .collect();
                    let d_slow = DeviceBuffer::from_slice(&slow_periods)
                        .map_err(|e| CudaPpoError::Cuda(e.to_string()))?;
                    for (start, count) in grid_y_chunks(rows) {
                        let grid: GridSize = (grid_x.max(1), count as u32, 1).into();
                        unsafe {
                            let mut p_fast = fast_dev.buf.as_device_ptr().as_raw();
                            let mut p_slow = slow_dev.buf.as_device_ptr().as_raw();
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
                            self.stream
                                .launch(&func, grid, block, 0, args)
                                .map_err(|e| CudaPpoError::Cuda(e.to_string()))?;
                        }
                    }
                }
            }
            _ => unreachable!(),
        }

        Ok((
            DeviceArrayF32 {
                buf: d_out,
                rows,
                cols: len,
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
        if len <= 0 || n_combos <= 0 {
            return Ok(());
        }
        let func = self
            .module
            .get_function("ppo_batch_f32")
            .map_err(|e| CudaPpoError::Cuda(e.to_string()))?;

        // Choose geometry
        let (grid, block): (GridSize, BlockSize) = match self.policy.batch {
            BatchKernelPolicy::OneD { block_x } if block_x > 0 => {
                if ma_mode == 0 {
                    // SMA: time-parallel
                if ma_mode == 0 {
                    // SMA: time-parallel
                    let bx = block_x;
                    let gx = ((len as u32) + bx - 1) / bx;
                    ((gx.max(1), 1, 1).into(), (bx, 1, 1).into())
                } else {
                    // EMA: one block per combo; let thread 0 run
                } else {
                    // EMA: one block per combo; let thread 0 run
                    ((1, 1, 1).into(), (block_x, 1, 1).into())
                }
            }
            _ => {
                if ma_mode == 0 {
                    let bx = 256u32;
                    let gx = ((len as u32) + bx - 1) / bx;
                    ((gx.max(1), 1, 1).into(), (bx, 1, 1).into())
                } else {
                    ((1, 1, 1).into(), (256, 1, 1).into())
                }
            }
        };

        // Chunk grid.y across combos
        for (start, count) in grid_y_chunks(n_combos as usize) {
            let gy = count as u32;
            let grid_launch: GridSize = (grid.x, gy, 1).into();
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
                self.stream
                    .launch(&func, grid_launch, block, 0, args)
                    .map_err(|e| CudaPpoError::Cuda(e.to_string()))?;
            }
        }

        Ok(())
    }

    // --------------- Many-series × one-param (time-major) ---------------
    pub fn ppo_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &PpoParams,
    ) -> Result<DeviceArrayF32, CudaPpoError> {
        if cols == 0 || rows == 0 {
            return Err(CudaPpoError::InvalidInput("empty dims".into()));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaPpoError::InvalidInput(
                "length mismatch for time-major input".into(),
            ));
        }

        let fast = params.fast_period.unwrap_or(12) as i32;
        let slow = params.slow_period.unwrap_or(26) as i32;
        let ma_mode = ma_mode_from(params.ma_type.as_deref().unwrap_or("sma"))?;

        // Copy input once
        let d_prices_tm: DeviceBuffer<f32> = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaPpoError::Cuda(e.to_string()))?;

        // first_valids per series (time-major layout)
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
        let d_first = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaPpoError::Cuda(e.to_string()))?;

        // SMA needs time-major per-series prefix sums; EMA ignores this buffer.
        let d_prefix_tm: DeviceBuffer<f64> = if ma_mode == 0 {
            let prefix = prefix_sum_time_major_f64(data_tm_f32, cols, rows, &first_valids);
            DeviceBuffer::from_slice(&prefix).map_err(|e| CudaPpoError::Cuda(e.to_string()))?
        } else {
            DeviceBuffer::from_slice(&[0.0f64]).map_err(|e| CudaPpoError::Cuda(e.to_string()))?
        };

        // Output
        let elems = cols * rows;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(|e| CudaPpoError::Cuda(e.to_string()))?;

        // Direct kernel (handles both SMA and EMA paths). If it fails (e.g., run-time
        // JIT constraint), fall back to building MA surfaces + elementwise PPO.
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
            // Build MA surfaces
            match ma_mode {
                0 => {
                    let sma = CudaSma::new(0)
                        .map_err(|e| CudaPpoError::Cuda(e.to_string()))?;
                    let pfast = SmaParams {
                        period: Some(fast as usize),
                    };
                    let pslow = SmaParams {
                        period: Some(slow as usize),
                    };
                    let fast_dev = sma
                        .sma_multi_series_one_param_time_major_dev(
                            data_tm_f32, cols, rows, &pfast,
                        )
                        .map_err(|e| CudaPpoError::Cuda(e.to_string()))?;
                    let slow_dev = sma
                        .sma_multi_series_one_param_time_major_dev(
                            data_tm_f32, cols, rows, &pslow,
                        )
                        .map_err(|e| CudaPpoError::Cuda(e.to_string()))?;
                    // Launch elementwise PPO kernel
                    let func = self
                        .module
                        .get_function(
                            "ppo_from_ma_many_series_one_param_time_major_f32",
                        )
                        .map_err(|e| CudaPpoError::Cuda(e.to_string()))?;
                    let tx = 256u32;
                    let ty = 1u32;
                    let grid: GridSize = (
                        ((rows as u32) + tx - 1) / tx,
                        ((cols as u32) + ty - 1) / ty,
                        1,
                    )
                        .into();
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
                        self.stream
                            .launch(&func, grid, block, 0, args)
                            .map_err(|e| CudaPpoError::Cuda(e.to_string()))?;
                    }
                }
                1 => {
                    let ema = CudaEma::new(0)
                        .map_err(|e| CudaPpoError::Cuda(e.to_string()))?;
                    let pfast = EmaParams {
                        period: Some(fast as usize),
                    };
                    let pslow = EmaParams {
                        period: Some(slow as usize),
                    };
                    let fast_dev = ema
                        .ema_many_series_one_param_time_major_dev(
                            data_tm_f32, cols, rows, &pfast,
                        )
                        .map_err(|e| CudaPpoError::Cuda(e.to_string()))?;
                    let slow_dev = ema
                        .ema_many_series_one_param_time_major_dev(
                            data_tm_f32, cols, rows, &pslow,
                        )
                        .map_err(|e| CudaPpoError::Cuda(e.to_string()))?;
                    // Elementwise PPO kernel
                    let func = self
                        .module
                        .get_function(
                            "ppo_from_ma_many_series_one_param_time_major_f32",
                        )
                        .map_err(|e| CudaPpoError::Cuda(e.to_string()))?;
                    let tx = 256u32;
                    let ty = 1u32;
                    let grid: GridSize = (
                        ((rows as u32) + tx - 1) / tx,
                        ((cols as u32) + ty - 1) / ty,
                        1,
                    )
                        .into();
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
                        self.stream
                            .launch(&func, grid, block, 0, args)
                            .map_err(|e| CudaPpoError::Cuda(e.to_string()))?;
                    }
                }
                _ => unreachable!(),
            }
        }

        Ok(DeviceArrayF32 { buf: d_out, rows, cols })
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
        if cols <= 0 || rows <= 0 {
            return Ok(());
        }
        let func = self
            .module
            .get_function("ppo_many_series_one_param_time_major_f32")
            .map_err(|e| CudaPpoError::Cuda(e.to_string()))?;

        let (tx, ty) = match self.policy.many_series {
            ManySeriesKernelPolicy::Tiled2D { tx, ty } if tx > 0 && ty > 0 => (tx, ty),
            _ => (128u32, 1u32), // wide in time; let y expand as needed below
        };
        let grid_x = ((rows as u32) + tx - 1) / tx;
        let grid_y = ((cols as u32) + ty - 1) / ty;
        let grid: GridSize = (grid_x.max(1), grid_y.max(1), 1).into();
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
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaPpoError::Cuda(e.to_string()))?;
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
            .map_err(|e| CudaPpoError::Cuda(e.to_string()))?;

        // Choose a warp-friendly block size (>=32 and multiple of 32)
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

        // chunk across grid.y limit (≤ 65,535)
        let total = n_combos as usize;
        let max_rows_per_launch = combos_per_block * 65_535usize;
        let mut start = 0usize;
        while start < total {
            let count = (total - start).min(max_rows_per_launch);
            let gy = div_ceil_u32(count as u32, combos_per_block as u32);
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
                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaPpoError::Cuda(e.to_string()))?;
            }

            start += count;
        }
        Ok(())
    }
}

// ---------------- Helpers ----------------

fn expand_grid(range: &PpoBatchRange) -> Vec<PpoParams> {
    fn axis_u((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let mut out = Vec::new();
    let fasts = axis_u(range.fast_period);
    let slows = axis_u(range.slow_period);
    for &f in &fasts {
        for &s in &slows {
            out.push(PpoParams {
                fast_period: Some(f),
                slow_period: Some(s),
                ma_type: Some(range.ma_type.clone()),
            });
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

// Host prefix sums in f64 for robust SMA windows (single series)
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

// Time‑major per‑series prefix sums in f64 (rows×cols)
#[inline]
fn prefix_sum_time_major_f64(
    data_tm: &[f32],
    cols: usize,
    rows: usize,
    first_valids: &[i32],
) -> Vec<f64> {
    let mut ps = vec![0.0f64; rows * cols + 1];
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
    ps
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
    if step == 0 || start == end {
        return vec![start];
    }
    (start..=end).step_by(step).collect()
}
#[inline]
fn axis_len(start: usize, end: usize, step: usize) -> usize {
    axis_vals(start, end, step).len()
}
fn axis_len(start: usize, end: usize, step: usize) -> usize {
    axis_vals(start, end, step).len()
}

// ---------------- Bench profiles ----------------
pub mod benches {
    use super::*;
    use crate::cuda::{CudaBenchScenario, CudaBenchState};

    const SERIES_LEN: usize = 1_000_000; // 1m
    const PARAM_SWEEP: usize = 250;
    const MANY_COLS: usize = 250;
    const MANY_ROWS: usize = 1_000_000; // 1m per series (large)

    fn gen_prices(n: usize) -> Vec<f32> {
        let mut v = vec![f32::NAN; n];
        for i in 10..n {
            let x = i as f32;
            v[i] = (x * 0.00123).sin() + 0.00011 * x;
        }
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
            for t in (s % 11)..rows {
                let x = (t as f32) + (s as f32) * 0.37;
                v[t * cols + s] = (x * 0.0019).sin() + 0.00021 * x;
            }
        }
        v
    }

    fn bytes_one_series_many() -> usize {
        let len = SERIES_LEN;
        let combos = PARAM_SWEEP * PARAM_SWEEP; // fast×slow grid
        len * 4 + (len + 1) * 8 + combos * 2 * 4 + combos * len * 4 + 64 * 1024 * 1024
        let len = SERIES_LEN;
        let combos = PARAM_SWEEP * PARAM_SWEEP; // fast×slow grid
        len * 4 + (len + 1) * 8 + combos * 2 * 4 + combos * len * 4 + 64 * 1024 * 1024
    }
    fn bytes_many_series_one() -> usize {
        let elems = MANY_COLS * MANY_ROWS;
        elems * 4 + (elems + 1) * 8 + MANY_COLS * 4 + elems * 4 + 64 * 1024 * 1024
        let elems = MANY_COLS * MANY_ROWS;
        elems * 4 + (elems + 1) * 8 + MANY_COLS * 4 + elems * 4 + 64 * 1024 * 1024
    }

    struct BatchState {
        cuda: CudaPpo,
        data: Vec<f32>,
        sweep: PpoBatchRange,
    }
    impl CudaBenchState for BatchState {
        fn launch(&mut self) {
            let _ = self.cuda.ppo_batch_dev(&self.data, &self.sweep).unwrap();
        }
    }
    struct BatchState {
        cuda: CudaPpo,
        data: Vec<f32>,
        sweep: PpoBatchRange,
    }
    impl CudaBenchState for BatchState {
        fn launch(&mut self) {
            let _ = self.cuda.ppo_batch_dev(&self.data, &self.sweep).unwrap();
        }
    }
    fn prep_batch_sma() -> Box<dyn CudaBenchState> {
        Box::new(BatchState {
            cuda: CudaPpo::new(0).unwrap(),
            data: gen_prices(SERIES_LEN),
            sweep: PpoBatchRange {
                fast_period: (10, 10 + PARAM_SWEEP - 1, 1),
                slow_period: (20, 20 + PARAM_SWEEP - 1, 1),
                ma_type: "sma".into(),
            },
        })
        Box::new(BatchState {
            cuda: CudaPpo::new(0).unwrap(),
            data: gen_prices(SERIES_LEN),
            sweep: PpoBatchRange {
                fast_period: (10, 10 + PARAM_SWEEP - 1, 1),
                slow_period: (20, 20 + PARAM_SWEEP - 1, 1),
                ma_type: "sma".into(),
            },
        })
    }
    fn prep_many_series_ema() -> Box<dyn CudaBenchState> {
        struct St {
            cuda: CudaPpo,
            data_tm: Vec<f32>,
            cols: usize,
            rows: usize,
            params: PpoParams,
        }
        impl CudaBenchState for St {
            fn launch(&mut self) {
                let _ = self
                    .cuda
                    .ppo_many_series_one_param_time_major_dev(
                        &self.data_tm,
                        self.cols,
                        self.rows,
                        &self.params,
                    )
                    .unwrap();
            }
        }
        struct St {
            cuda: CudaPpo,
            data_tm: Vec<f32>,
            cols: usize,
            rows: usize,
            params: PpoParams,
        }
        impl CudaBenchState for St {
            fn launch(&mut self) {
                let _ = self
                    .cuda
                    .ppo_many_series_one_param_time_major_dev(
                        &self.data_tm,
                        self.cols,
                        self.rows,
                        &self.params,
                    )
                    .unwrap();
            }
        }
        let cuda = CudaPpo::new(0).unwrap();
        let cols = MANY_COLS;
        let rows = MANY_ROWS;
        let data_tm = gen_tm(cols, rows);
        let params = PpoParams {
            fast_period: Some(12),
            slow_period: Some(26),
            ma_type: Some("ema".into()),
        };
        Box::new(St {
            cuda,
            data_tm,
            cols,
            rows,
            params,
        })
        let cols = MANY_COLS;
        let rows = MANY_ROWS;
        let data_tm = gen_tm(cols, rows);
        let params = PpoParams {
            fast_period: Some(12),
            slow_period: Some(26),
            ma_type: Some("ema".into()),
        };
        Box::new(St {
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
                "many_series_one_param",
                "ppo_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_ema,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one()),
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
