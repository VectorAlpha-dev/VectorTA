#![cfg(feature = "cuda")]

//! CUDA wrapper for ADX (Average Directional Index)
//!
//! Parity goals
//! - Stream NON_BLOCKING
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/adx_kernel.ptx"))
//! - API mirrors ALMA-style device+host helpers
//! - Warmup/NaN semantics match scalar: before first + 2*period - 1 -> NaN
//! - VRAM sanity check + simple chunking guardrails

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::adx::{AdxBatchRange, AdxParams};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::error::CudaError;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum CudaAdxError {
    #[error(transparent)]
    Cuda(#[from] CudaError),
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
    #[error("device mismatch: buf on {buf}, current {current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

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
pub struct CudaAdxPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaAdxPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

pub struct CudaAdx {
    module: Module,
    stream: Stream,
    ctx: Arc<Context>,
    device_id: u32,
    policy: CudaAdxPolicy,
}

impl CudaAdx {
    #[inline(always)]
    fn div_up(n: u32, d: u32) -> u32 { (n + d - 1) / d }

    pub fn new(device_id: usize) -> Result<Self, CudaAdxError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Context::new(device)?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/adx_kernel.ptx"));
        let module = Module::from_ptx(
            ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O4),
            ],
        )
        .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
        .or_else(|_| Module::from_ptx(ptx, &[]))?;
        
        let pr = cust::context::CurrentContext::get_stream_priority_range()?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, Some(pr.greatest))?;

        Ok(Self { module, stream, ctx: Arc::new(context), device_id: device_id as u32, policy: CudaAdxPolicy::default() })
    }

    pub fn set_policy(&mut self, policy: CudaAdxPolicy) {
        self.policy = policy;
    }

    #[inline]
    pub fn ctx(&self) -> std::sync::Arc<Context> { std::sync::Arc::clone(&self.ctx) }

    #[inline]
    pub fn device_id(&self) -> u32 { self.device_id }

    #[inline]
    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaAdxError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaAdxError::OutOfMemory { required, free, headroom });
            }
        }
        Ok(())
    }

    fn prepare_batch(
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &AdxBatchRange,
    ) -> Result<(Vec<AdxParams>, usize, usize, usize), CudaAdxError> {
        if high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaAdxError::InvalidInput("empty input".into()));
        }
        let len = high.len().min(low.len()).min(close.len());
        let first_valid = (0..len)
            .find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
            .ok_or_else(|| CudaAdxError::InvalidInput("all values are NaN".into()))?;
        
        let (start, end, step) = sweep.period;
        let periods: Vec<usize> = if start == end || step == 0 {
            vec![start]
        } else if start < end {
            (start..=end).step_by(step.max(1)).collect()
        } else {
            let mut v = Vec::new();
            let mut cur = start;
            let s = step.max(1);
            while cur >= end {
                v.push(cur);
                if cur < s { break; }
                cur -= s;
                if cur == usize::MAX { break; }
            }
            v
        };
        if periods.is_empty() {
            return Err(CudaAdxError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        let combos: Vec<AdxParams> = periods
            .iter()
            .map(|&p| AdxParams { period: Some(p) })
            .collect();
        let max_p = *periods.iter().max().unwrap();
        if len - first_valid < max_p + 1 {
            return Err(CudaAdxError::InvalidInput(format!(
                "not enough valid data (needed >= {}, valid = {})",
                max_p + 1,
                len - first_valid
            )));
        }
        Ok((combos, first_valid, len, max_p))
    }

    pub fn adx_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &AdxBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<AdxParams>), CudaAdxError> {
        let (combos, first_valid, len, _max_p) = Self::prepare_batch(high, low, close, sweep)?;
        let rows = combos.len();

        
        let el = std::mem::size_of::<f32>();
        let req = len
            .checked_mul(3)
            .and_then(|x| x.checked_add(rows))
            .and_then(|x| x.checked_add(rows.checked_mul(len)?))
            .and_then(|x| x.checked_mul(el))
            .ok_or_else(|| CudaAdxError::InvalidInput("size overflow".into()))?;
        Self::will_fit(req, 64 * 1024 * 1024)?;

        let out_len = rows
            .checked_mul(len)
            .ok_or_else(|| CudaAdxError::InvalidInput("rows*len overflow".into()))?;

        
        let d_high = unsafe { DeviceBuffer::from_slice_async(&high[..len], &self.stream) }?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(&low[..len], &self.stream) }?;
        let d_close = unsafe { DeviceBuffer::from_slice_async(&close[..len], &self.stream) }?;

        let periods_host: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let d_periods = unsafe { DeviceBuffer::from_slice_async(&periods_host, &self.stream) }?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_len, &self.stream) }?;

        self.launch_batch(
            &d_high,
            &d_low,
            &d_close,
            &d_periods,
            len,
            rows,
            first_valid,
            &mut d_out,
        )?;
        self.stream.synchronize()?;

        Ok((
            DeviceArrayF32 {
                buf: d_out,
                rows,
                cols: len,
            },
            combos,
        ))
    }

    pub fn adx_batch_into_host_f32(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &AdxBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<AdxParams>), CudaAdxError> {
        let (arr, combos) = self.adx_batch_dev(high, low, close, sweep)?;
        let expected = arr.rows * arr.cols;
        if out.len() != expected {
            return Err(CudaAdxError::InvalidInput(format!(
                "output slice wrong length: got {}, need {}",
                out.len(),
                expected
            )));
        }
        
        let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(expected) }?;
        unsafe { arr.buf.async_copy_to(pinned.as_mut_slice(), &self.stream) }?;
        self.stream.synchronize()?;
        out.copy_from_slice(pinned.as_slice());
        Ok((arr.rows, arr.cols, combos))
    }

    fn launch_batch(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAdxError> {
        let func = self
            .module
            .get_function("adx_batch_f32")
            .map_err(|_| CudaAdxError::MissingKernelSymbol { name: "adx_batch_f32" })?;

        let block_x = match self.policy.batch {
            BatchKernelPolicy::Auto => 32,
            BatchKernelPolicy::Plain { block_x } => block_x.max(32),
        };
        let grid_x = Self::div_up(n_combos as u32, block_x);
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        
        let dev = Device::get_device(self.device_id)?;
        let max_threads = dev.get_attribute(DeviceAttribute::MaxThreadsPerBlock)? as u32;
        let max_grid_x = dev.get_attribute(DeviceAttribute::MaxGridDimX)? as u32;
        if block_x > max_threads || grid_x > max_grid_x {
            return Err(CudaAdxError::LaunchConfigTooLarge { gx: grid_x, gy: 1, gz: 1, bx: block_x, by: 1, bz: 1 });
        }
        unsafe {
            let mut h = d_high.as_device_ptr().as_raw();
            let mut l = d_low.as_device_ptr().as_raw();
            let mut c = d_close.as_device_ptr().as_raw();
            let mut p = d_periods.as_device_ptr().as_raw();
            let mut n = series_len as i32;
            let mut r = n_combos as i32;
            let mut f = first_valid as i32;
            let mut o = d_out.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 8] = [
                &mut h as *mut _ as *mut c_void,
                &mut l as *mut _ as *mut c_void,
                &mut c as *mut _ as *mut c_void,
                &mut p as *mut _ as *mut c_void,
                &mut n as *mut _ as *mut c_void,
                &mut r as *mut _ as *mut c_void,
                &mut f as *mut _ as *mut c_void,
                &mut o as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, &mut args)?;
        }
        Ok(())
    }

    pub fn adx_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaAdxError> {
        if cols == 0 || rows == 0 {
            return Err(CudaAdxError::InvalidInput("empty matrix".into()));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaAdxError::InvalidInput("rows*cols overflow".into()))?;
        if high_tm.len() != expected || low_tm.len() != expected || close_tm.len() != expected {
            return Err(CudaAdxError::InvalidInput("matrix shape mismatch".into()));
        }
        
        let mut first_valids = vec![rows as i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                let ok = !high_tm[idx].is_nan() && !low_tm[idx].is_nan() && !close_tm[idx].is_nan();
                if ok {
                    first_valids[s] = t as i32;
                    break;
                }
            }
        }
        
        for &fv in &first_valids {
            if fv as usize + period >= rows {
                return Err(CudaAdxError::InvalidInput(
                    "not enough valid data for at least one series".into(),
                ));
            }
        }

        let el = std::mem::size_of::<f32>();
        let bytes_inputs = 3usize
            .checked_mul(cols)
            .and_then(|x| x.checked_mul(rows))
            .and_then(|x| x.checked_mul(el))
            .ok_or_else(|| CudaAdxError::InvalidInput("size overflow".into()))?;
        let bytes_first = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaAdxError::InvalidInput("size overflow".into()))?;
        let bytes_out = cols
            .checked_mul(rows)
            .and_then(|x| x.checked_mul(el))
            .ok_or_else(|| CudaAdxError::InvalidInput("size overflow".into()))?;
        let req = bytes_inputs
            .checked_add(bytes_first)
            .and_then(|x| x.checked_add(bytes_out))
            .ok_or_else(|| CudaAdxError::InvalidInput("size overflow".into()))?;
        Self::will_fit(req, 64 * 1024 * 1024)?;

        let d_high = unsafe { DeviceBuffer::from_slice_async(high_tm, &self.stream) }?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(low_tm, &self.stream) }?;
        let d_close = unsafe { DeviceBuffer::from_slice_async(close_tm, &self.stream) }?;
        let d_first = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(expected, &self.stream) }?;

        self.launch_many_series(
            &d_high, &d_low, &d_close, cols, rows, period, &d_first, &mut d_out,
        )?;
        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn adx_many_series_one_param_time_major_into_host_f32(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
        out_tm: &mut [f32],
    ) -> Result<(), CudaAdxError> {
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaAdxError::InvalidInput("rows*cols overflow".into()))?;
        if out_tm.len() != expected {
            return Err(CudaAdxError::InvalidInput("out slice wrong length".into()));
        }
        let arr = self.adx_many_series_one_param_time_major_dev(high_tm, low_tm, close_tm, cols, rows, period)?;
        let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(expected) }?;
        unsafe { arr.buf.async_copy_to(pinned.as_mut_slice(), &self.stream) }?;
        self.stream.synchronize()?;
        out_tm.copy_from_slice(pinned.as_slice());
        Ok(())
    }

    fn launch_many_series(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        period: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAdxError> {
        let func = self
            .module
            .get_function("adx_many_series_one_param_time_major_f32")
            .map_err(|_| CudaAdxError::MissingKernelSymbol { name: "adx_many_series_one_param_time_major_f32" })?;
        let block_x = match self.policy.many_series { ManySeriesKernelPolicy::Auto => 256, ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32) };
        let grid_x = Self::div_up(cols as u32, block_x);
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        let dev = Device::get_device(self.device_id)?;
        let max_threads = dev.get_attribute(DeviceAttribute::MaxThreadsPerBlock)? as u32;
        let max_grid_x = dev.get_attribute(DeviceAttribute::MaxGridDimX)? as u32;
        if block_x > max_threads || grid_x > max_grid_x {
            return Err(CudaAdxError::LaunchConfigTooLarge { gx: grid_x, gy: 1, gz: 1, bx: block_x, by: 1, bz: 1 });
        }
        unsafe {
            let mut h = d_high.as_device_ptr().as_raw();
            let mut l = d_low.as_device_ptr().as_raw();
            let mut c = d_close.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut p = period as i32;
            let mut fv = d_first_valids.as_device_ptr().as_raw();
            let mut o = d_out.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 8] = [
                &mut h as *mut _ as *mut c_void,
                &mut l as *mut _ as *mut c_void,
                &mut c as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut p as *mut _ as *mut c_void,
                &mut fv as *mut _ as *mut c_void,
                &mut o as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, &mut args)?;
        }
        Ok(())
    }
}


pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use cust::memory::DeviceBuffer;

    const LEN_1M: usize = 1_000_000;
    const COLS_512: usize = 512;
    const ROWS_16K: usize = 16_384;
    const PARAM_SWEEP_250: usize = 250;

    fn synth_hlc_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i];
            if v.is_nan() {
                continue;
            }
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

    struct ManySeriesState {
        cuda: CudaAdx,
        d_high_tm: DeviceBuffer<f32>,
        d_low_tm: DeviceBuffer<f32>,
        d_close_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for ManySeriesState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series(
                    &self.d_high_tm,
                    &self.d_low_tm,
                    &self.d_close_tm,
                    self.cols,
                    self.rows,
                    self.period,
                    &self.d_first_valids,
                    &mut self.d_out_tm,
                )
                .expect("adx many-series kernel");
            self.cuda.stream.synchronize().expect("adx many-series sync");
        }
    }

    struct BatchDevState {
        cuda: CudaAdx,
        d_high: DeviceBuffer<f32>,
        d_low: DeviceBuffer<f32>,
        d_close: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for BatchDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch(
                    &self.d_high,
                    &self.d_low,
                    &self.d_close,
                    &self.d_periods,
                    self.len,
                    self.n_combos,
                    self.first_valid,
                    &mut self.d_out,
                )
                .expect("adx launch_batch");
            self.cuda.stream.synchronize().expect("cuda sync");
        }
    }

    fn prep_batch_dev_1m_x_250() -> Box<dyn CudaBenchState> {
        let cuda = CudaAdx::new(0).expect("cuda adx");
        let close = gen_series(LEN_1M);
        let (high, low) = synth_hlc_from_close(&close);

        let first_valid = (0..LEN_1M)
            .find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
            .unwrap_or(LEN_1M);

        
        let periods_host: Vec<i32> = (0..PARAM_SWEEP_250)
            .map(|i| (8 + 8 * i) as i32)
            .collect();
        let n_combos = periods_host.len();

        
        
        let d_high = DeviceBuffer::from_slice(&high).unwrap();
        let d_low = DeviceBuffer::from_slice(&low).unwrap();
        let d_close = DeviceBuffer::from_slice(&close).unwrap();
        let d_periods = DeviceBuffer::from_slice(&periods_host).unwrap();

        let out_len = n_combos.checked_mul(LEN_1M).unwrap();
        let d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(out_len) }.unwrap();

        cuda.stream.synchronize().unwrap();
        Box::new(BatchDevState {
            cuda,
            d_high,
            d_low,
            d_close,
            d_periods,
            len: LEN_1M,
            n_combos,
            first_valid,
            d_out,
        })
    }

    fn prep_many() -> Box<dyn CudaBenchState> {
        let cuda = CudaAdx::new(0).expect("cuda adx");
        
        let cols = COLS_512;
        let rows = ROWS_16K;
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
        let mut first_valids: Vec<i32> = vec![rows as i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                if !high_tm[idx].is_nan() && !low_tm[idx].is_nan() && !close_tm[idx].is_nan() {
                    first_valids[s] = t as i32;
                    break;
                }
            }
        }
        let period = 14usize;
        let elems = cols * rows;
        let d_high_tm = DeviceBuffer::from_slice(&high_tm).expect("d_high_tm");
        let d_low_tm = DeviceBuffer::from_slice(&low_tm).expect("d_low_tm");
        let d_close_tm = DeviceBuffer::from_slice(&close_tm).expect("d_close_tm");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_out_tm");
        cuda.stream.synchronize().expect("sync after prep");
        Box::new(ManySeriesState {
            cuda,
            d_high_tm,
            d_low_tm,
            d_close_tm,
            d_first_valids,
            cols,
            rows,
            period,
            d_out_tm,
        })
    }

    fn bytes_batch_1m_x_250() -> usize {
        let in_bytes = 3 * LEN_1M * std::mem::size_of::<f32>();
        let out_bytes = PARAM_SWEEP_250 * LEN_1M * std::mem::size_of::<f32>();
        let periods_bytes = PARAM_SWEEP_250 * std::mem::size_of::<i32>();
        in_bytes + out_bytes + periods_bytes + 64 * 1024 * 1024
    }
    fn bytes_many() -> usize {
        (3 * COLS_512 * ROWS_16K + COLS_512 * ROWS_16K) * std::mem::size_of::<f32>()
            + 64 * 1024 * 1024
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "adx",
                "batch",
                "adx_cuda_batch_dev",
                "1m_x_250",
                prep_batch_dev_1m_x_250,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_batch_1m_x_250()),
            CudaBenchScenario::new(
                "adx",
                "many_series_one_param",
                "adx_cuda_many_series",
                "16k x 512",
                prep_many,
            )
            .with_mem_required(bytes_many()),
        ]
    }
}
