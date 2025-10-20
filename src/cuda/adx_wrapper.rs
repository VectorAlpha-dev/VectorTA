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
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaAdxError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaAdxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaAdxError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaAdxError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaAdxError {}

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
    _context: Context,
    policy: CudaAdxPolicy,
}

impl CudaAdx {
    #[inline(always)]
    fn div_up(n: u32, d: u32) -> u32 { (n + d - 1) / d }

    pub fn new(device_id: usize) -> Result<Self, CudaAdxError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaAdxError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaAdxError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaAdxError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/adx_kernel.ptx"));
        let module = Module::from_ptx(
            ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O4),
            ],
        )
        .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
        .or_else(|_| Module::from_ptx(ptx, &[]))
        .map_err(|e| CudaAdxError::Cuda(e.to_string()))?;
        // Prefer a high-priority NON_BLOCKING stream; if priorities unsupported, this is 0.
        let pr = cust::context::CurrentContext
            ::get_stream_priority_range()
            .map_err(|e| CudaAdxError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, Some(pr.greatest))
            .map_err(|e| CudaAdxError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaAdxPolicy::default(),
        })
    }

    pub fn set_policy(&mut self, policy: CudaAdxPolicy) { self.policy = policy; }

    #[inline]
    fn device_mem_ok(required: usize) -> bool {
        match mem_get_info() {
            Ok((free, _)) => required.saturating_add(64 * 1024 * 1024) <= free,
            Err(_) => true,
        }
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
        // Expand grid (mirror of scalar expand_grid)
        let (start, end, step) = sweep.period;
        let periods: Vec<usize> = if step == 0 || start == end {
            vec![start]
        } else {
            (start..=end).step_by(step).collect()
        };
        if periods.is_empty() {
            return Err(CudaAdxError::InvalidInput("no parameter combinations".into()));
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

        // VRAM estimate: 3 inputs + periods + output
        let req = (3 * len + rows + rows * len) * std::mem::size_of::<f32>();
        if !Self::device_mem_ok(req) {
            return Err(CudaAdxError::InvalidInput("insufficient device memory".into()));
        }

        // Upload inputs
        let d_high = unsafe { DeviceBuffer::from_slice_async(&high[..len], &self.stream) }
            .map_err(|e| CudaAdxError::Cuda(e.to_string()))?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(&low[..len], &self.stream) }
            .map_err(|e| CudaAdxError::Cuda(e.to_string()))?;
        let d_close = unsafe { DeviceBuffer::from_slice_async(&close[..len], &self.stream) }
            .map_err(|e| CudaAdxError::Cuda(e.to_string()))?;

        let periods_host: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let d_periods = unsafe { DeviceBuffer::from_slice_async(&periods_host, &self.stream) }
            .map_err(|e| CudaAdxError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(rows * len, &self.stream) }
            .map_err(|e| CudaAdxError::Cuda(e.to_string()))?;

        self.launch_batch(&d_high, &d_low, &d_close, &d_periods, len, rows, first_valid, &mut d_out)?;
        self.stream
            .synchronize()
            .map_err(|e| CudaAdxError::Cuda(e.to_string()))?;

        Ok((DeviceArrayF32 { buf: d_out, rows, cols: len }, combos))
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
                out.len(), expected
            )));
        }
        // Pinned host + async D→H ensures true async transfer; then memcpy into user slice.
        let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(expected) }
            .map_err(|e| CudaAdxError::Cuda(e.to_string()))?;
        unsafe { arr.buf.async_copy_to(pinned.as_mut_slice(), &self.stream) }
            .map_err(|e| CudaAdxError::Cuda(e.to_string()))?;
        self.stream
            .synchronize()
            .map_err(|e| CudaAdxError::Cuda(e.to_string()))?;
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
            .map_err(|e| CudaAdxError::Cuda(e.to_string()))?;

        let block_x = match self.policy.batch {
            BatchKernelPolicy::Auto => 32,
            BatchKernelPolicy::Plain { block_x } => block_x.max(32),
        };
        let grid_x = Self::div_up(n_combos as u32, block_x);
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
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
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(|e| CudaAdxError::Cuda(e.to_string()))?;
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
        if high_tm.len() != cols * rows
            || low_tm.len() != cols * rows
            || close_tm.len() != cols * rows
        {
            return Err(CudaAdxError::InvalidInput("matrix shape mismatch".into()));
        }
        // Per-series first_valid
        let mut first_valids = vec![rows as i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                let ok = !high_tm[idx].is_nan() && !low_tm[idx].is_nan() && !close_tm[idx].is_nan();
                if ok { first_valids[s] = t as i32; break; }
            }
        }
        // Ensure each series has enough data
        for &fv in &first_valids {
            if fv as usize + period >= rows {
                return Err(CudaAdxError::InvalidInput("not enough valid data for at least one series".into()));
            }
        }

        let bytes_inputs = 3usize * cols * rows * std::mem::size_of::<f32>();
        let bytes_first  = cols * std::mem::size_of::<i32>();
        let bytes_out    = cols * rows * std::mem::size_of::<f32>();
        let req = bytes_inputs + bytes_first + bytes_out;
        if !Self::device_mem_ok(req) {
            return Err(CudaAdxError::InvalidInput("insufficient device memory".into()));
        }

        let d_high = unsafe { DeviceBuffer::from_slice_async(high_tm, &self.stream) }
            .map_err(|e| CudaAdxError::Cuda(e.to_string()))?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(low_tm, &self.stream) }
            .map_err(|e| CudaAdxError::Cuda(e.to_string()))?;
        let d_close = unsafe { DeviceBuffer::from_slice_async(close_tm, &self.stream) }
            .map_err(|e| CudaAdxError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaAdxError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(cols * rows, &self.stream) }
            .map_err(|e| CudaAdxError::Cuda(e.to_string()))?;

        self.launch_many_series(&d_high, &d_low, &d_close, cols, rows, period, &d_first, &mut d_out)?;
        self.stream
            .synchronize()
            .map_err(|e| CudaAdxError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 { buf: d_out, rows, cols })
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
        if out_tm.len() != cols * rows {
            return Err(CudaAdxError::InvalidInput("out slice wrong length".into()));
        }
        let arr = self.adx_many_series_one_param_time_major_dev(high_tm, low_tm, close_tm, cols, rows, period)?;
        let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaAdxError::Cuda(e.to_string()))?;
        unsafe { arr.buf.async_copy_to(pinned.as_mut_slice(), &self.stream) }
            .map_err(|e| CudaAdxError::Cuda(e.to_string()))?;
        self.stream.synchronize().map_err(|e| CudaAdxError::Cuda(e.to_string()))?;
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
            .map_err(|e| CudaAdxError::Cuda(e.to_string()))?;
        let block_x = match self.policy.many_series { ManySeriesKernelPolicy::Auto => 256, ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32) };
        let grid_x = Self::div_up(cols as u32, block_x);
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
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
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(|e| CudaAdxError::Cuda(e.to_string()))?;
        }
        Ok(())
    }
}

// ---------- Benches ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const LEN_1M: usize = 1_000_000;
    const COLS_512: usize = 512;
    const ROWS_16K: usize = 16_384;

    fn synth_hlc_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i];
            if v.is_nan() { continue; }
            let x = i as f32 * 0.0025;
            let off = (0.002 * x.sin()).abs() + 0.15;
            high[i] = v + off;
            low[i] = v - off;
        }
        (high, low)
    }

    struct BatchState { cuda: CudaAdx, high: Vec<f32>, low: Vec<f32>, close: Vec<f32>, sweep: AdxBatchRange }
    impl CudaBenchState for BatchState { fn launch(&mut self) { let _ = self.cuda.adx_batch_dev(&self.high, &self.low, &self.close, &self.sweep).unwrap(); } }

    struct ManySeriesState { cuda: CudaAdx, high_tm: Vec<f32>, low_tm: Vec<f32>, close_tm: Vec<f32>, cols: usize, rows: usize, period: usize }
    impl CudaBenchState for ManySeriesState { fn launch(&mut self) { let _ = self.cuda.adx_many_series_one_param_time_major_dev(&self.high_tm, &self.low_tm, &self.close_tm, self.cols, self.rows, self.period).unwrap(); } }

    fn prep_batch() -> Box<dyn CudaBenchState> {
        let cuda = CudaAdx::new(0).expect("cuda adx");
        let close = gen_series(LEN_1M);
        let (high, low) = synth_hlc_from_close(&close);
        let sweep = AdxBatchRange { period: (8, 64, 8) };
        Box::new(BatchState { cuda, high, low, close, sweep })
    }

    fn prep_many() -> Box<dyn CudaBenchState> {
        let cuda = CudaAdx::new(0).expect("cuda adx");
        // Build time-major matrices
        let cols = COLS_512; let rows = ROWS_16K;
        let close_tm = {
            let mut v = vec![f32::NAN; cols * rows];
            for s in 0..cols { for t in s..rows { let x = (t as f32) + (s as f32) * 0.2; v[t*cols + s] = (x * 0.002).sin() + 0.0003 * x; } }
            v
        };
        let (high_tm, low_tm) = synth_hlc_from_close(&close_tm);
        let period = 14usize;
        Box::new(ManySeriesState { cuda, high_tm, low_tm, close_tm, cols, rows, period })
    }

    fn bytes_batch() -> usize {
        (3 * LEN_1M + (LEN_1M / 8) + (LEN_1M * ((64 - 8) / 8 + 1))) * std::mem::size_of::<f32>() + 64 * 1024 * 1024
    }
    fn bytes_many() -> usize {
        (3 * COLS_512 * ROWS_16K + COLS_512 * ROWS_16K) * std::mem::size_of::<f32>() + 64 * 1024 * 1024
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new("adx", "batch", "adx_cuda_batch", "1m", prep_batch)
                .with_mem_required(bytes_batch()),
            CudaBenchScenario::new("adx", "many_series_one_param", "adx_cuda_many_series", "16k x 512", prep_many)
                .with_mem_required(bytes_many()),
        ]
    }
}
