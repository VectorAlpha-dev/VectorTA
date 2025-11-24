//! CUDA wrapper for Rate of Change (ROC)
//!
//! Semantics (parity with scalar/batch):
//! - Warmup region: NaN until `first_valid + period - 1`
//! - If previous value is 0.0 or NaN => output 0.0 (scalar policy)
//! - FP32 compute, NON_BLOCKING stream
//! - VRAM guard with ~64MB headroom; chunk grid.x over combos to <= 65_535

#![cfg(feature = "cuda")]

use crate::indicators::roc::{RocBatchRange, RocParams};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaRocError {
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

/// VRAM-backed array handle for ROC with context guard and device id.
pub struct DeviceArrayF32Roc {
    pub buf: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
    pub ctx: Arc<Context>,
    pub device_id: u32,
}
impl DeviceArrayF32Roc {
    #[inline]
    pub fn device_ptr(&self) -> u64 {
        self.buf.as_device_ptr().as_raw() as u64
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CudaRocPolicy {
    pub batch_block_x: Option<u32>,
    pub many_block_x: Option<u32>,
    /// If true (default), the stream is synchronized before returning.
    /// Set to false to keep launches fully async and overlap with other work.
    pub sync_after_launch: bool,
}

impl Default for CudaRocPolicy {
    fn default() -> Self {
        Self {
            batch_block_x: None,
            many_block_x: None,
            sync_after_launch: true,
        }
    }
}

pub struct CudaRoc {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaRocPolicy,
    // device limits used for chunking & clamping
    max_grid_x: u32,
    max_threads_per_block: u32,
}

impl CudaRoc {
    pub fn new(device_id: usize) -> Result<Self, CudaRocError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;

        // Query device limits before moving Device into Context
        let max_grid_x =
            device.get_attribute(DeviceAttribute::MaxGridDimX)? as u32;
        let max_threads_per_block =
            device.get_attribute(DeviceAttribute::MaxThreadsPerBlock)? as u32;

        let context = Arc::new(Context::new(device)?);

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/roc_kernel.ptx"));
        // Use explicit O4 JIT (most optimized) and let the driver pick SASS for current device.
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O4),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy: CudaRocPolicy::default(),
            max_grid_x,
            max_threads_per_block,
        })
    }

    #[inline]
    pub fn set_policy(&mut self, p: CudaRocPolicy) {
        self.policy = p;
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaRocError> {
        match mem_get_info() {
            Ok((free, _total)) => {
                let need = required_bytes.saturating_add(headroom_bytes);
                if need <= free {
                    Ok(())
                } else {
                    Err(CudaRocError::OutOfMemory {
                        required: required_bytes,
                        free,
                        headroom: headroom_bytes,
                    })
                }
            }
            Err(e) => Err(CudaRocError::from(e)),
        }
    }

    // ---------- Batch (one series × many params) ----------
    pub fn roc_batch_dev(
        &self,
        prices_f32: &[f32],
        sweep: &RocBatchRange,
    ) -> Result<DeviceArrayF32Roc, CudaRocError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(prices_f32, sweep)?;
        let n_combos = combos.len();
        let periods_i32: Vec<i32> = combos
            .iter()
            .map(|p| p.period.unwrap_or(0) as i32)
            .collect();

        // VRAM estimate (best-effort) with overflow-checked sizes
        let item_f32 = std::mem::size_of::<f32>();
        let item_i32 = std::mem::size_of::<i32>();
        let in_bytes = prices_f32
            .len()
            .checked_mul(item_f32)
            .ok_or_else(|| CudaRocError::InvalidInput("byte size overflow".into()))?;
        let params_bytes = periods_i32
            .len()
            .checked_mul(item_i32)
            .ok_or_else(|| CudaRocError::InvalidInput("byte size overflow".into()))?;
        let out_elems = n_combos
            .checked_mul(len)
            .ok_or_else(|| CudaRocError::InvalidInput("rows*cols overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(item_f32)
            .ok_or_else(|| CudaRocError::InvalidInput("byte size overflow".into()))?;
        let headroom = 64usize * 1024 * 1024;
        let total_bytes = in_bytes
            .checked_add(params_bytes)
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaRocError::InvalidInput("byte size overflow".into()))?;
        Self::will_fit(total_bytes, headroom)?;

        // Stream-ordered async copies/allocations (enable overlap)
        let d_prices = unsafe {
            DeviceBuffer::from_slice_async(prices_f32, &self.stream)?
        };
        let d_periods = unsafe {
            DeviceBuffer::from_slice_async(&periods_i32, &self.stream)?
        };
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(out_elems, &self.stream)?
        };

        self.launch_batch(
            &d_prices,
            &d_periods,
            len,
            first_valid,
            n_combos,
            &mut d_out,
        )?;
        Ok(DeviceArrayF32Roc {
            buf: d_out,
            rows: n_combos,
            cols: len,
            ctx: self.context.clone(),
            device_id: self.device_id,
        })
    }

    fn launch_batch(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaRocError> {
        if n_combos == 0 {
            return Ok(());
        }

        // Block size heuristic for memory-bound kernels; clamp to device limit
        let block_x = self
            .policy
            .batch_block_x
            .unwrap_or(256)
            .min(self.max_threads_per_block);

        let mut launched = 0usize;
        while launched < n_combos {
            // Respect hardware grid.x limit (typically >> 65k on cc >= 3.0)
            let this_chunk = (n_combos - launched).min(self.max_grid_x as usize);
            let grid_x = this_chunk as u32;
            if grid_x == 0 || grid_x > self.max_grid_x {
                return Err(CudaRocError::LaunchConfigTooLarge {
                    gx: grid_x,
                    gy: 1,
                    gz: 1,
                    bx: block_x,
                    by: 1,
                    bz: 1,
                });
            }

            let grid: GridSize = (grid_x.max(1), 1, 1).into();
            let block: BlockSize = (block_x.max(1), 1, 1).into();
            let func = self
                .module
                .get_function("roc_batch_f32")
                .map_err(|_| CudaRocError::MissingKernelSymbol { name: "roc_batch_f32" })?;
            unsafe {
                // Typed device-pointer arithmetic (in elements)
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                let mut periods_ptr = d_periods.as_device_ptr().add(launched).as_raw();
                let mut series_len_i = len as i32;
                let mut first_i = first_valid as i32;
                let mut combos_i = this_chunk as i32;
                let offset = launched
                    .checked_mul(len)
                    .ok_or_else(|| CudaRocError::InvalidInput("rows*cols overflow".into()))?;
                let mut out_ptr = d_out.as_device_ptr().add(offset).as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut combos_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, 0, args)?;
            }
            launched += this_chunk;
        }

        if self.policy.sync_after_launch {
            self.stream.synchronize().map_err(Into::into)
        } else {
            Ok(())
        }
    }

    // ---------- Many-series × one-param (time-major) ----------
    pub fn roc_many_series_one_param_time_major_dev(
        &self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32Roc, CudaRocError> {
        if cols == 0 || rows == 0 {
            return Err(CudaRocError::InvalidInput("invalid dims".into()));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaRocError::InvalidInput("rows*cols overflow".into()))?;
        if prices_tm_f32.len() != expected {
            return Err(CudaRocError::InvalidInput(
                "time-major length mismatch".into(),
            ));
        }
        if period == 0 {
            return Err(CudaRocError::InvalidInput("period must be > 0".into()));
        }

        // Per-series first_valid
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = -1i32;
            for t in 0..rows {
                let v = prices_tm_f32[t * cols + s];
                if !v.is_nan() {
                    fv = t as i32;
                    break;
                }
            }
            if fv < 0 {
                return Err(CudaRocError::InvalidInput(format!(
                    "series {} all NaN",
                    s
                )));
            }
            first_valids[s] = fv;
        }

        // VRAM estimate with checked arithmetic
        let item_f32 = std::mem::size_of::<f32>();
        let item_i32 = std::mem::size_of::<i32>();
        let in_bytes = expected
            .checked_mul(item_f32)
            .and_then(|x| x.checked_mul(2))
            .ok_or_else(|| CudaRocError::InvalidInput("byte size overflow".into()))?;
        let first_bytes = cols
            .checked_mul(item_i32)
            .ok_or_else(|| CudaRocError::InvalidInput("byte size overflow".into()))?;
        let out_bytes = expected
            .checked_mul(item_f32)
            .ok_or_else(|| CudaRocError::InvalidInput("byte size overflow".into()))?;
        let headroom = 64usize * 1024 * 1024;
        let total_bytes = in_bytes
            .checked_add(first_bytes)
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaRocError::InvalidInput("byte size overflow".into()))?;
        Self::will_fit(total_bytes, headroom)?;

        let d_prices = unsafe {
            DeviceBuffer::from_slice_async(prices_tm_f32, &self.stream)?
        };
        let d_first = unsafe {
            DeviceBuffer::from_slice_async(&first_valids, &self.stream)?
        };
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(expected, &self.stream)?
        };

        self.launch_many_series(&d_prices, &d_first, cols, rows, period, &mut d_out)?;
        Ok(DeviceArrayF32Roc {
            buf: d_out,
            rows,
            cols,
            ctx: self.context.clone(),
            device_id: self.device_id,
        })
    }

    fn launch_many_series(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaRocError> {
        let block_x_default = 256u32.min(self.max_threads_per_block);
        let block_x = self.policy.many_block_x.unwrap_or(block_x_default);
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        if grid_x == 0 || grid_x > self.max_grid_x {
            return Err(CudaRocError::LaunchConfigTooLarge {
                gx: grid_x,
                gy: 1,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        let func = self
            .module
            .get_function("roc_many_series_one_param_f32")
            .map_err(|_| {
                CudaRocError::MissingKernelSymbol {
                    name: "roc_many_series_one_param_f32",
                }
            })?;
        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut period_i = period as i32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        if self.policy.sync_after_launch {
            self.stream.synchronize().map_err(Into::into)
        } else {
            Ok(())
        }
    }

    // ---------- Helpers ----------
    fn prepare_batch_inputs(
        prices: &[f32],
        sweep: &RocBatchRange,
    ) -> Result<(Vec<RocParams>, usize, usize), CudaRocError> {
        let len = prices.len();
        if len == 0 {
            return Err(CudaRocError::InvalidInput("empty prices".into()));
        }
        // expand grid (mirror CPU semantics; map errors to InvalidInput)
        let combos = crate::indicators::roc::expand_grid(sweep)
            .map_err(|e| CudaRocError::InvalidInput(e.to_string()))?;

        let first_valid = (0..len)
            .find(|&i| !prices[i].is_nan())
            .ok_or_else(|| CudaRocError::InvalidInput("all values NaN".into()))?;
        let max_p = combos
            .iter()
            .map(|c| c.period.unwrap_or(0))
            .max()
            .unwrap_or(0);
        if max_p == 0 {
            return Err(CudaRocError::InvalidInput("period must be > 0".into()));
        }
        let valid = len - first_valid;
        if valid < max_p {
            return Err(CudaRocError::InvalidInput(format!(
                "not enough valid data: need >= {}, have {}",
                max_p, valid
            )));
        }
        Ok((combos, first_valid, len))
    }
}

// ---------- Benches ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const MANY_COLS: usize = 1024;
    const MANY_ROWS: usize = 8192;
    const PARAM_SWEEP: usize = 250; // 2..=251

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series() -> usize {
        let n = MANY_COLS * MANY_ROWS;
        let in_bytes = n * std::mem::size_of::<f32>();
        let out_bytes = n * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct RocBatchState {
        cuda: CudaRoc,
        prices: Vec<f32>,
        sweep: RocBatchRange,
    }
    impl CudaBenchState for RocBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .roc_batch_dev(&self.prices, &self.sweep)
                .expect("roc batch");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaRoc::new(0).expect("cuda roc");
        let mut prices = gen_series(ONE_SERIES_LEN);
        for i in 0..8 {
            prices[i] = f32::NAN;
        }
        for i in 8..ONE_SERIES_LEN {
            let x = i as f32 * 0.0019;
            prices[i] += 0.0005 * x.sin();
        }
        let sweep = RocBatchRange {
            period: (2, 1 + PARAM_SWEEP, 1),
        };
        Box::new(RocBatchState {
            cuda,
            prices,
            sweep,
        })
    }

    struct RocManyState {
        cuda: CudaRoc,
        prices_tm: Vec<f32>,
    }
    impl CudaBenchState for RocManyState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .roc_many_series_one_param_time_major_dev(&self.prices_tm, MANY_COLS, MANY_ROWS, 14)
                .expect("roc many");
        }
    }
    fn prep_many_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaRoc::new(0).expect("cuda roc");
        let n = MANY_COLS * MANY_ROWS;
        let mut base = gen_series(n);
        let mut prices = vec![f32::NAN; n];
        for s in 0..MANY_COLS {
            for t in s..MANY_ROWS {
                let idx = t * MANY_COLS + s;
                let x = (t as f32) * 0.002 + (s as f32) * 0.01;
                prices[idx] = base[idx] + 0.05 * x.sin();
            }
        }
        Box::new(RocManyState {
            cuda,
            prices_tm: prices,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "roc",
                "one_series_many_params",
                "roc_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "roc",
                "many_series_one_param",
                "roc_cuda_many_series_one_param_dev",
                "1024x8192",
                prep_many_series,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_many_series()),
        ]
    }
}
