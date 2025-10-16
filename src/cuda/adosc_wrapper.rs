//! CUDA wrapper for Chaikin Accumulation/Distribution Oscillator (ADOSC).
//!
//! Category: Recurrence/IIR. We precompute the ADL once on device, then run
//! one EMA-pair (short/long) per block over that ADL for batch sweeps.
//! For many-series × one-param (time-major), each block handles one series.
//!
//! Semantics:
//! - No warmup NaNs — ADOSC starts from index 0 (out[0] = 0.0).
//! - Division by zero in MFM (high==low) yields 0.0 contribution.
//! - NaN propagation is natural via arithmetic; mirrors scalar path.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::adosc::{AdoscBatchRange, AdoscParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::DeviceBuffer;
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::memory::mem_get_info;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
pub enum CudaAdoscError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaAdoscError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaAdoscError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaAdoscError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaAdoscError {}

pub struct CudaAdosc {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaAdoscPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaAdosc {
    pub fn new(device_id: usize) -> Result<Self, CudaAdoscError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaAdoscError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaAdoscError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaAdoscError::Cuda(e.to_string()))?;

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/adosc_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts).or_else(|_| {
            Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
                .or_else(|_| Module::from_ptx(ptx, &[]))
        })
        .map_err(|e| CudaAdoscError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaAdoscError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaAdoscPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    /// Create using an explicit policy (for testing/tuning).
    pub fn new_with_policy(device_id: usize, policy: CudaAdoscPolicy) -> Result<Self, CudaAdoscError> {
        let mut s = Self::new(device_id)?;
        unsafe { (*( &s as *const _ as *mut CudaAdosc)).policy = policy; }
        Ok(s)
    }
    #[inline]
    pub fn set_policy(&mut self, policy: CudaAdoscPolicy) { self.policy = policy; }
    #[inline]
    pub fn policy(&self) -> &CudaAdoscPolicy { &self.policy }

    /// One-series × many-params. Returns a row-major device matrix of size
    /// (rows = #combos, cols = series_len).
    pub fn adosc_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        volume: &[f32],
        sweep: &AdoscBatchRange,
    ) -> Result<DeviceArrayF32, CudaAdoscError> {
        let len = high.len();
        if len == 0 || low.len() != len || close.len() != len || volume.len() != len {
            return Err(CudaAdoscError::InvalidInput(
                "input slices are empty or mismatched".into(),
            ));
        }
        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaAdoscError::InvalidInput("no parameter combos".into()));
        }

        // Copy inputs
        let d_high = DeviceBuffer::from_slice(high)
            .map_err(|e| CudaAdoscError::Cuda(e.to_string()))?;
        let d_low = DeviceBuffer::from_slice(low)
            .map_err(|e| CudaAdoscError::Cuda(e.to_string()))?;
        let d_close = DeviceBuffer::from_slice(close)
            .map_err(|e| CudaAdoscError::Cuda(e.to_string()))?;
        let d_volume = DeviceBuffer::from_slice(volume)
            .map_err(|e| CudaAdoscError::Cuda(e.to_string()))?;

        // Precompute ADL on device
        let mut d_adl: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(len)
                .map_err(|e| CudaAdoscError::Cuda(e.to_string()))?
        };
        self.launch_adl(&d_high, &d_low, &d_close, &d_volume, len, &mut d_adl)?;

        // Period arrays
        let mut shorts: Vec<i32> = Vec::with_capacity(combos.len());
        let mut longs: Vec<i32> = Vec::with_capacity(combos.len());
        for prm in &combos {
            let sp = prm.short_period.unwrap_or(3) as i32;
            let lp = prm.long_period.unwrap_or(10) as i32;
            if sp <= 0 || lp <= 0 || sp >= lp {
                return Err(CudaAdoscError::InvalidInput(format!(
                    "invalid params: short={} long={}",
                    sp, lp
                )));
            }
            shorts.push(sp);
            longs.push(lp);
        }
        let d_shorts = DeviceBuffer::from_slice(&shorts)
            .map_err(|e| CudaAdoscError::Cuda(e.to_string()))?;
        let d_longs = DeviceBuffer::from_slice(&longs)
            .map_err(|e| CudaAdoscError::Cuda(e.to_string()))?;

        // Estimate VRAM and chunk launches if needed
        let (rows, cols) = (combos.len(), len);
        let bytes_inputs = 4 * cols * std::mem::size_of::<f32>();
        let bytes_adl = cols * std::mem::size_of::<f32>();
        let bytes_periods = 2 * rows * std::mem::size_of::<i32>();
        let bytes_out_total = rows * cols * std::mem::size_of::<f32>();
        let mut required = bytes_inputs + bytes_adl + bytes_periods + bytes_out_total;
        // Leave ~64MB headroom by default
        let headroom = 64usize * 1024 * 1024;
        let fits = match mem_get_info() {
            Ok((free, _)) => required.saturating_add(headroom) <= free,
            Err(_) => true,
        };

        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(rows * cols)
                .map_err(|e| CudaAdoscError::Cuda(e.to_string()))?
        };

        if fits {
            unsafe {
                (*(self as *const _ as *mut CudaAdosc)).last_batch =
                    Some(BatchKernelSelected::Plain { block_x: 1 });
            }
            self.launch_batch_from_adl(&d_adl, &d_shorts, &d_longs, cols, rows, &mut d_out)?;
            self.maybe_log_batch_debug();
        } else {
            // Chunk by combos to fit VRAM and grid limits (<= 65_535)
            unsafe {
                (*(self as *const _ as *mut CudaAdosc)).last_batch =
                    Some(BatchKernelSelected::Plain { block_x: 1 });
            }
            self.maybe_log_batch_debug();
            let max_grid = 65_535usize;
            let mut start = 0usize;
            let mut host_out = vec![0f32; rows * cols];
            while start < rows {
                let remain = rows - start;
                let chunk = remain.min(max_grid);
                let mut d_out_chunk: DeviceBuffer<f32> = unsafe {
                    DeviceBuffer::uninitialized(chunk * cols)
                        .map_err(|e| CudaAdoscError::Cuda(e.to_string()))?
                };
                let shorts_off = DeviceBuffer::from_slice(&shorts[start..start + chunk])
                    .map_err(|e| CudaAdoscError::Cuda(e.to_string()))?;
                let longs_off = DeviceBuffer::from_slice(&longs[start..start + chunk])
                    .map_err(|e| CudaAdoscError::Cuda(e.to_string()))?;
                self.launch_batch_from_adl(
                    &d_adl,
                    &shorts_off,
                    &longs_off,
                    cols,
                    chunk,
                    &mut d_out_chunk,
                )?;
                let base = start * cols;
                d_out_chunk
                    .copy_to(&mut host_out[base..base + chunk * cols])
                    .map_err(|e| CudaAdoscError::Cuda(e.to_string()))?;
                start += chunk;
            }
            let d_out = DeviceBuffer::from_slice(&host_out)
                .map_err(|e| CudaAdoscError::Cuda(e.to_string()))?;
            return Ok(DeviceArrayF32 { buf: d_out, rows, cols });
        }

        // Unreachable; both branches return
        // but keep a fallback to satisfy type checker in some editors
        // (Rust compiler sees returns above).
        #[allow(unreachable_code)]
        {
            let d_out = DeviceBuffer::from_slice(&vec![0f32; rows * cols])
                .map_err(|e| CudaAdoscError::Cuda(e.to_string()))?;
            Ok(DeviceArrayF32 { buf: d_out, rows, cols })
        }
    }

    /// Many-series × one-param (time-major). Returns a (rows x cols) device matrix
    /// in time-major layout.
    pub fn adosc_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        volume_tm: &[f32],
        cols: usize,
        rows: usize,
        short: usize,
        long: usize,
    ) -> Result<DeviceArrayF32, CudaAdoscError> {
        let len = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaAdoscError::InvalidInput("rows*cols overflow".into()))?;
        if high_tm.len() != len
            || low_tm.len() != len
            || close_tm.len() != len
            || volume_tm.len() != len
        {
            return Err(CudaAdoscError::InvalidInput(
                "time-major inputs are mismatched".into(),
            ));
        }
        if short == 0 || long == 0 || short >= long {
            return Err(CudaAdoscError::InvalidInput(
                "invalid short/long".into(),
            ));
        }

        let d_high = DeviceBuffer::from_slice(high_tm)
            .map_err(|e| CudaAdoscError::Cuda(e.to_string()))?;
        let d_low = DeviceBuffer::from_slice(low_tm)
            .map_err(|e| CudaAdoscError::Cuda(e.to_string()))?;
        let d_close = DeviceBuffer::from_slice(close_tm)
            .map_err(|e| CudaAdoscError::Cuda(e.to_string()))?;
        let d_volume = DeviceBuffer::from_slice(volume_tm)
            .map_err(|e| CudaAdoscError::Cuda(e.to_string()))?;

        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(len)
                .map_err(|e| CudaAdoscError::Cuda(e.to_string()))?
        };
        self.launch_many_series_one_param(
            &d_high, &d_low, &d_close, &d_volume, cols, rows, short, long, &mut d_out,
        )?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    fn launch_adl(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        d_volume: &DeviceBuffer<f32>,
        series_len: usize,
        d_adl_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAdoscError> {
        let func = self
            .module
            .get_function("adosc_adl_f32")
            .map_err(|e| CudaAdoscError::Cuda(e.to_string()))?;
        let grid: GridSize = (1, 1, 1).into();
        let block: BlockSize = (1, 1, 1).into();
        unsafe {
            let mut h = d_high.as_device_ptr().as_raw();
            let mut l = d_low.as_device_ptr().as_raw();
            let mut c = d_close.as_device_ptr().as_raw();
            let mut v = d_volume.as_device_ptr().as_raw();
            let mut n_i = series_len as i32;
            let mut out = d_adl_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut h as *mut _ as *mut c_void,
                &mut l as *mut _ as *mut c_void,
                &mut c as *mut _ as *mut c_void,
                &mut v as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut out as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaAdoscError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaAdoscError::Cuda(e.to_string()))
    }

    fn launch_batch_from_adl(
        &self,
        d_adl: &DeviceBuffer<f32>,
        d_shorts: &DeviceBuffer<i32>,
        d_longs: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAdoscError> {
        if n_combos == 0 || series_len == 0 {
            return Ok(());
        }
        let func = self
            .module
            .get_function("adosc_batch_from_adl_f32")
            .map_err(|e| CudaAdoscError::Cuda(e.to_string()))?;
        // One combo per block; single thread inside performs the scan
        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (1, 1, 1).into();
        unsafe {
            let mut adl = d_adl.as_device_ptr().as_raw();
            let mut sp = d_shorts.as_device_ptr().as_raw();
            let mut lp = d_longs.as_device_ptr().as_raw();
            let mut n = series_len as i32;
            let mut combos = n_combos as i32;
            let mut out = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut adl as *mut _ as *mut c_void,
                &mut sp as *mut _ as *mut c_void,
                &mut lp as *mut _ as *mut c_void,
                &mut n as *mut _ as *mut c_void,
                &mut combos as *mut _ as *mut c_void,
                &mut out as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaAdoscError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaAdoscError::Cuda(e.to_string()))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_one_param(
        &self,
        d_high_tm: &DeviceBuffer<f32>,
        d_low_tm: &DeviceBuffer<f32>,
        d_close_tm: &DeviceBuffer<f32>,
        d_volume_tm: &DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        short: usize,
        long: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAdoscError> {
        if cols == 0 || rows == 0 {
            return Err(CudaAdoscError::InvalidInput(
                "cols/rows must be positive".into(),
            ));
        }
        let func = self
            .module
            .get_function("adosc_many_series_one_param_f32")
            .map_err(|e| CudaAdoscError::Cuda(e.to_string()))?;
        let grid: GridSize = (cols as u32, 1, 1).into();
        let block: BlockSize = (1, 1, 1).into();
        unsafe {
            let mut h = d_high_tm.as_device_ptr().as_raw();
            let mut l = d_low_tm.as_device_ptr().as_raw();
            let mut c = d_close_tm.as_device_ptr().as_raw();
            let mut v = d_volume_tm.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut sp = short as i32;
            let mut lp = long as i32;
            let mut out = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut h as *mut _ as *mut c_void,
                &mut l as *mut _ as *mut c_void,
                &mut c as *mut _ as *mut c_void,
                &mut v as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut sp as *mut _ as *mut c_void,
                &mut lp as *mut _ as *mut c_void,
                &mut out as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaAdoscError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaAdoscError::Cuda(e.to_string()))
    }
}

// -------- Policy and introspection (parity with ALMA style) --------

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
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
pub struct CudaAdoscPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaAdoscPolicy {
    fn default() -> Self {
        Self { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Auto }
    }
}

impl CudaAdosc {
    #[inline]
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
                    eprintln!("[DEBUG] ADOSC batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaAdosc)).debug_batch_logged = true;
                }
            }
        }
    }

    #[inline]
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
                    eprintln!("[DEBUG] ADOSC many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaAdosc)).debug_many_logged = true;
                }
            }
        }
    }
}

// ---------- Bench profiles ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250; // ~250 (short,long) combos

    fn bytes_one_series_many_params() -> usize {
        // 4 inputs + 1 ADL + outputs
        let in_bytes = 4 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let adl_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + adl_bytes + out_bytes + 64 * 1024 * 1024
    }

    fn synth_hlc_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i];
            if v.is_nan() {
                continue;
            }
            let x = i as f32 * 0.0019;
            let off = (0.0031 * x.sin()).abs() + 0.08;
            high[i] = v + off;
            low[i] = v - off;
        }
        (high, low)
    }

    struct AdoscBatchState {
        cuda: CudaAdosc,
        high: Vec<f32>,
        low: Vec<f32>,
        close: Vec<f32>,
        volume: Vec<f32>,
        sweep: AdoscBatchRange,
    }
    impl CudaBenchState for AdoscBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .adosc_batch_dev(&self.high, &self.low, &self.close, &self.volume, &self.sweep)
                .expect("adosc batch");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaAdosc::new(0).expect("cuda adosc");
        let close = gen_series(ONE_SERIES_LEN);
        let (high, low) = synth_hlc_from_close(&close);
        let mut volume = vec![0.0f32; ONE_SERIES_LEN];
        for i in 0..ONE_SERIES_LEN {
            let x = i as f32 * 0.0027;
            volume[i] = (x.cos().abs() + 0.4) * 1000.0;
        }
        let sweep = AdoscBatchRange {
            short_period: (3, 3 + PARAM_SWEEP / 5, 1),
            long_period: (10, 10 + PARAM_SWEEP, 2),
        };
        Box::new(AdoscBatchState {
            cuda,
            high,
            low,
            close,
            volume,
            sweep,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "adosc",
            "one_series_many_params",
            "adosc_cuda_batch_dev",
            "1m_x_250",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}

fn expand_grid(r: &AdoscBatchRange) -> Vec<AdoscParams> {
    fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let shorts = axis(r.short_period);
    let longs = axis(r.long_period);
    let mut out = Vec::new();
    for &s in &shorts {
        for &l in &longs {
            if s == 0 || l == 0 || s >= l {
                continue;
            }
            out.push(AdoscParams {
                short_period: Some(s),
                long_period: Some(l),
            });
        }
    }
    out
}
