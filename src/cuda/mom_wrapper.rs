//! CUDA wrapper for Momentum (MOM): value[t] = price[t] - price[t - period]
//!
//! Parity with scalar semantics:
//! - Warmup prefix: NaN up to first_valid + period - 1
//! - Mid-stream NaNs propagate naturally (no masking)
//! - FP32 compute and NON_BLOCKING stream
//! - VRAM guard with ~64MB headroom and simple chunking for large combo counts

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::mom::{MomBatchRange, MomParams};
use cust::context::{Context, ContextFlags};
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
pub enum CudaMomError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaMomPolicy {
    pub batch_block_x: Option<u32>,
    pub many_block_x: Option<u32>,
}

pub struct CudaMom {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaMomPolicy,
    sm_count: u32,
    max_grid_x: u32,
}

fn expand_grid_checked_cuda(r: &MomBatchRange) -> Result<Vec<MomParams>, CudaMomError> {
    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, CudaMomError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let mut out = Vec::new();
        if start < end {
            let mut v = start;
            while v <= end {
                out.push(v);
                match v.checked_add(step) {
                    Some(next) if next > v => v = next,
                    _ => break,
                }
            }
        } else {
            let mut v = start;
            while v >= end {
                out.push(v);
                if v < end + step {
                    break;
                }
                v = v.saturating_sub(step);
                if v == 0 {
                    break;
                }
            }
        }
        if out.is_empty() {
            return Err(CudaMomError::InvalidInput(format!(
                "invalid range: start={} end={} step={}",
                start, end, step
            )));
        }
        Ok(out)
    }

    let periods = axis_usize(r.period)?;
    let mut out = Vec::with_capacity(periods.len());
    for p in periods {
        out.push(MomParams { period: Some(p) });
    }
    Ok(out)
}

impl CudaMom {
    pub fn new(device_id: usize) -> Result<Self, CudaMomError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        // Enable recommended primary-context flags to support mapped/pinned host memory
        // and let the driver choose appropriate scheduling.
        // SCHED_AUTO is accepted after context creation; MAP_HOST must be set at
        // creation time for legacy contexts, so we do not set it here.
        context.set_flags(ContextFlags::SCHED_AUTO)?;

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/mom_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O4),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        // Device properties used for better launch sizing & chunking.
        let sm_count = device
            .get_attribute(DeviceAttribute::MultiprocessorCount)? as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)? as u32;

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy: CudaMomPolicy::default(),
            sm_count,
            max_grid_x,
        })
    }

    #[inline]
    pub fn set_policy(&mut self, p: CudaMomPolicy) {
        self.policy = p;
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
    fn will_fit(&self, required: usize, headroom: usize) -> Result<(), CudaMomError> {
        match mem_get_info() {
            Ok((free, _total)) => {
                if required.saturating_add(headroom) > free {
                    return Err(CudaMomError::OutOfMemory { required, free, headroom });
                }
                Ok(())
            }
            Err(e) => Err(CudaMomError::Cuda(e)),
        }
    }

    #[inline]
    fn validate_launch(grid: GridSize, block: BlockSize) -> Result<(), CudaMomError> {
        let (gx, gy, gz) = (grid.x, grid.y, grid.z);
        let (bx, by, bz) = (block.x, block.y, block.z);
        let threads = (bx as u64) * (by as u64) * (bz as u64);
        if threads > 1024 {
            return Err(CudaMomError::LaunchConfigTooLarge { gx, gy, gz, bx, by, bz });
        }
        Ok(())
    }

    // ---------- Batch (one series × many params) ----------
    pub fn mom_batch_dev(
        &self,
        prices_f32: &[f32],
        sweep: &MomBatchRange,
    ) -> Result<DeviceArrayF32, CudaMomError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(prices_f32, sweep)?;
        let n_combos = combos.len();
        let periods_i32: Vec<i32> = combos
            .iter()
            .map(|p| p.period.unwrap_or(0) as i32)
            .collect();

        // VRAM estimate (best-effort, checked arithmetic)
        let in_bytes = prices_f32
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaMomError::InvalidInput("prices bytes overflow".into()))?;
        let params_bytes = periods_i32
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaMomError::InvalidInput("params bytes overflow".into()))?;
        let out_elems = n_combos
            .checked_mul(len)
            .ok_or_else(|| CudaMomError::InvalidInput("rows*cols overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaMomError::InvalidInput("output bytes overflow".into()))?;
        let required = in_bytes
            .checked_add(params_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaMomError::InvalidInput("total VRAM size overflow".into()))?;
        self.will_fit(required, 64 * 1024 * 1024)?;

        let d_prices = DeviceBuffer::from_slice(prices_f32)?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(out_elems)?
        };

        self.launch_batch(
            &d_prices,
            &d_periods,
            len,
            first_valid,
            n_combos,
            &mut d_out,
        )?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: len,
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
    ) -> Result<(), CudaMomError> {
        if n_combos == 0 {
            return Ok(());
        }
        let func = self
            .module
            .get_function("mom_batch_f32")
            .map_err(|_| CudaMomError::MissingKernelSymbol { name: "mom_batch_f32" })?;

        // Chunk across combos if needed to respect grid limits
        let block_x = self.policy.batch_block_x.unwrap_or(256);
        // Use actual device limit (Ada: ~2.1B), not legacy 65,535.
        let max_blocks: u32 = self.max_grid_x.max(1);
        let mut launched = 0usize;
        while launched < n_combos {
            let this_chunk = (n_combos - launched).min(max_blocks as usize);
            let grid_x = this_chunk as u32;
            let grid: GridSize = (grid_x.max(1), 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            CudaMom::validate_launch(grid, block)?;

            unsafe {
                // Offset param/output pointers for chunk
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                let mut periods_ptr = d_periods
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * std::mem::size_of::<i32>()) as u64);
                let mut series_len_i = len as i32;
                let mut first_i = first_valid as i32;
                let mut combos_i = this_chunk as i32;
                let mut out_ptr = d_out
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add(((launched * len) * std::mem::size_of::<f32>()) as u64);

                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut combos_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(CudaMomError::Cuda)?;
            }
            launched += this_chunk;
        }
        self.stream.synchronize().map_err(CudaMomError::Cuda)
    }

    // ---------- Many-series × one-param (time-major) ----------
    pub fn mom_many_series_one_param_time_major_dev(
        &self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaMomError> {
        if cols == 0 || rows == 0 {
            return Err(CudaMomError::InvalidInput("invalid dims".into()));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaMomError::InvalidInput("rows*cols overflow".into()))?;
        if prices_tm_f32.len() != expected {
            return Err(CudaMomError::InvalidInput(
                "time-major length mismatch".into(),
            ));
        }
        if period == 0 {
            return Err(CudaMomError::InvalidInput("period must be > 0".into()));
        }

        // Per-series first_valid (row-major scan, early exit per series)
        let mut first_valids = vec![i32::MAX; cols];
        let mut remaining = cols;
        for t in 0..rows {
            let row = &prices_tm_f32[t * cols..(t + 1) * cols];
            for s in 0..cols {
                if first_valids[s] == i32::MAX && !row[s].is_nan() {
                    first_valids[s] = t as i32;
                    remaining -= 1;
                }
            }
            if remaining == 0 {
                break;
            }
        }
        if let Some(s_bad) = first_valids.iter().position(|&fv| fv == i32::MAX) {
            return Err(CudaMomError::InvalidInput(format!("series {} all NaN", s_bad)));
        }

        // VRAM estimate
        let elems = expected;
        let prices_bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| CudaMomError::InvalidInput("prices bytes overflow".into()))?;
        let first_bytes = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaMomError::InvalidInput("first_valids bytes overflow".into()))?;
        let out_bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaMomError::InvalidInput("output bytes overflow".into()))?;
        let required = prices_bytes
            .checked_add(first_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaMomError::InvalidInput("total VRAM size overflow".into()))?;
        self.will_fit(required, 64 * 1024 * 1024)?;

        let d_prices = DeviceBuffer::from_slice(prices_tm_f32)?;
        let d_first = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(expected)?
        };

        self.launch_many_series(&d_prices, &d_first, cols, rows, period, &mut d_out)?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
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
    ) -> Result<(), CudaMomError> {
        let func = self
            .module
            .get_function("mom_many_series_one_param_f32")
            .map_err(|_| CudaMomError::MissingKernelSymbol { name: "mom_many_series_one_param_f32" })?;
        // Kernel is grid-stride across series; we don't need to launch one block per column.
        let block_x = self.policy.many_block_x.unwrap_or(256);
        let needed = ((cols as u32) + block_x - 1) / block_x;
        // Heuristic: ~8 blocks per SM keeps the GPU busy without overlaunch.
        let cap = self.sm_count.saturating_mul(8).max(1);
        let grid_x = needed.min(cap).max(1);
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        CudaMom::validate_launch(grid, block)?;
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
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(CudaMomError::Cuda)?;
        }
        self.stream.synchronize().map_err(CudaMomError::Cuda)
    }

    // ---------- Helpers ----------
    fn prepare_batch_inputs(
        prices: &[f32],
        sweep: &MomBatchRange,
    ) -> Result<(Vec<MomParams>, usize, usize), CudaMomError> {
        let len = prices.len();
        if len == 0 {
            return Err(CudaMomError::InvalidInput("empty prices".into()));
        }
        let combos = expand_grid_checked_cuda(sweep)?;

        let first_valid = (0..len)
            .find(|&i| !prices[i].is_nan())
            .ok_or_else(|| CudaMomError::InvalidInput("all values NaN".into()))?;
        let max_p = combos
            .iter()
            .map(|c| c.period.unwrap_or(0))
            .max()
            .unwrap_or(0);
        if max_p == 0 {
            return Err(CudaMomError::InvalidInput("period must be > 0".into()));
        }
        // If some periods exceed available valid tail, we let the kernel mark those rows as NaN
        // (scalar parity). Other rows still compute.
        Ok((combos, first_valid, len))
    }

    /// Copy a host series once and get its first_valid (host-scan).
    pub fn copy_series_to_device_with_first_valid(
        &self,
        prices_f32: &[f32],
    ) -> Result<(DeviceBuffer<f32>, usize), CudaMomError> {
        let len = prices_f32.len();
        if len == 0 {
            return Err(CudaMomError::InvalidInput("empty prices".into()));
        }
        let first_valid = (0..len)
            .find(|&i| !prices_f32[i].is_nan())
            .ok_or_else(|| CudaMomError::InvalidInput("all values NaN".into()))?;
        let bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaMomError::InvalidInput("prices bytes overflow".into()))?;
        self.will_fit(bytes, 64 * 1024 * 1024)?;
        let d_prices = DeviceBuffer::from_slice(prices_f32)?;
        Ok((d_prices, first_valid))
    }

    /// Fast path: reuse a device-resident series across many parameter sweeps (no H→D for prices).
    pub fn mom_batch_dev_with_device_prices(
        &self,
        d_prices: &DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        sweep: &MomBatchRange,
    ) -> Result<DeviceArrayF32, CudaMomError> {
        let combos = expand_grid_checked_cuda(sweep)?;
        let n_combos = combos.len();
        let periods_i32: Vec<i32> = combos.iter().map(|p| p.period.unwrap_or(0) as i32).collect();

        // VRAM estimate (no input duplication)
        let params_bytes = periods_i32
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaMomError::InvalidInput("params bytes overflow".into()))?;
        let out_elems = n_combos
            .checked_mul(len)
            .ok_or_else(|| CudaMomError::InvalidInput("rows*cols overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaMomError::InvalidInput("output bytes overflow".into()))?;
        let required = params_bytes
            .checked_add(out_bytes)
            .ok_or_else(|| CudaMomError::InvalidInput("total VRAM size overflow".into()))?;
        self.will_fit(required, 64 * 1024 * 1024)?;

        // Ensure current device matches wrapper device when reusing a device-resident buffer
        unsafe {
            let mut cur: i32 = 0;
            let _ = cust::sys::cuCtxGetDevice(&mut cur);
            if cur as u32 != self.device_id {
                return Err(CudaMomError::DeviceMismatch {
                    buf: self.device_id,
                    current: cur as u32,
                });
            }
        }

        let d_periods = DeviceBuffer::from_slice(&periods_i32)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(out_elems)? };
        self.launch_batch(d_prices, &d_periods, len, first_valid, n_combos, &mut d_out)?;
        Ok(DeviceArrayF32 { buf: d_out, rows: n_combos, cols: len })
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

    struct MomBatchDeviceState {
        cuda: CudaMom,
        d_prices: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for MomBatchDeviceState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch(
                    &self.d_prices,
                    &self.d_periods,
                    self.len,
                    self.first_valid,
                    self.n_combos,
                    &mut self.d_out,
                )
                .expect("mom launch");
            self.cuda.stream.synchronize().expect("mom sync");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaMom::new(0).expect("cuda mom");
        let mut prices = gen_series(ONE_SERIES_LEN);
        // introduce a small NaN warmup and trend
        for i in 0..8 {
            prices[i] = f32::NAN;
        }
        for i in 8..ONE_SERIES_LEN {
            let x = i as f32 * 0.0019;
            prices[i] += 0.0005 * x.sin();
        }
        let sweep = MomBatchRange {
            period: (2, 1 + PARAM_SWEEP, 1),
        };

        let (combos, first_valid, len) =
            CudaMom::prepare_batch_inputs(&prices, &sweep).expect("prepare_batch_inputs");
        let periods_i32: Vec<i32> = combos
            .iter()
            .map(|p| p.period.unwrap_or(0) as i32)
            .collect();
        let n_combos = periods_i32.len();

        let d_prices = unsafe {
            DeviceBuffer::from_slice_async(&prices, &cuda.stream)
        }
        .expect("d_prices H2D");
        let d_periods = unsafe {
            DeviceBuffer::from_slice_async(&periods_i32, &cuda.stream)
        }
        .expect("d_periods H2D");
        let d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(len * n_combos, &cuda.stream)
        }
        .expect("d_out alloc");
        cuda.stream.synchronize().expect("mom prep sync");

        Box::new(MomBatchDeviceState {
            cuda,
            d_prices,
            d_periods,
            len,
            first_valid,
            n_combos,
            d_out,
        })
    }

    struct MomManyDeviceState {
        cuda: CudaMom,
        d_prices_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for MomManyDeviceState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series(
                    &self.d_prices_tm,
                    &self.d_first_valids,
                    MANY_COLS,
                    MANY_ROWS,
                    14,
                    &mut self.d_out_tm,
                )
                .expect("mom many launch");
            self.cuda.stream.synchronize().expect("mom many sync");
        }
    }
    fn prep_many_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaMom::new(0).expect("cuda mom");
        let n = MANY_COLS * MANY_ROWS;
        let mut base = gen_series(n);
        let mut prices = vec![f32::NAN; n];
        for s in 0..MANY_COLS {
            for t in s..MANY_ROWS {
                // stagger first_valids
                let idx = t * MANY_COLS + s;
                let x = (t as f32) * 0.002 + (s as f32) * 0.01;
                prices[idx] = base[idx] + 0.05 * x.sin();
            }
        }

        let mut first_valids = vec![0i32; MANY_COLS];
        for s in 0..MANY_COLS {
            let mut fv = -1i32;
            for t in 0..MANY_ROWS {
                let v = prices[t * MANY_COLS + s];
                if !v.is_nan() {
                    fv = t as i32;
                    break;
                }
            }
            first_valids[s] = fv.max(0);
        }

        let d_prices_tm = unsafe {
            DeviceBuffer::from_slice_async(&prices, &cuda.stream)
        }
        .expect("d_prices_tm H2D");
        let d_first_valids = unsafe {
            DeviceBuffer::from_slice_async(&first_valids, &cuda.stream)
        }
        .expect("d_first_valids H2D");
        let d_out_tm: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(n, &cuda.stream)
        }
        .expect("d_out_tm alloc");
        cuda.stream.synchronize().expect("mom many prep sync");

        Box::new(MomManyDeviceState {
            cuda,
            d_prices_tm,
            d_first_valids,
            d_out_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "mom",
                "one_series_many_params",
                "mom_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "mom",
                "many_series_one_param",
                "mom_cuda_many_series_one_param_dev",
                "1024x8192",
                prep_many_series,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_many_series()),
        ]
    }
}
