//! CUDA wrapper for Awesome Oscillator (AO): AO = SMA(short) - SMA(long) over HL2.
//!
//! Pattern classification: Prefix-sum/rational for batch (one-series × many-params)
//! and rolling-sum recurrence for many-series × one-param (time-major).
//!
//! Semantics:
//! - Warmup/writes: NaN until `warm = first_valid + long - 1`, identical to scalar AO.
//! - NaN inputs: we honor `first_valid` and do no mid-stream masking (matches CPU path).
//! - Accumulation: host prefix uses f64; device uses f64-derived values → f32 outputs.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::ao::{AoBatchRange, AoParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::DeviceBuffer;
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::memory::mem_get_info;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
pub enum CudaAoError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaAoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaAoError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaAoError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}

impl std::error::Error for CudaAoError {}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaAoPolicy {
    pub batch_block_x: Option<u32>,
    pub many_block_x: Option<u32>,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected { Plain { block_x: u32 } }
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected { OneD { block_x: u32 } }

pub struct CudaAo {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaAoPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaAo {
    pub fn new(device_id: usize) -> Result<Self, CudaAoError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaAoError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaAoError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaAoError::Cuda(e.to_string()))?;

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/ao_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaAoError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaAoError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaAoPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    #[inline]
    pub fn set_policy(&mut self, p: CudaAoPolicy) { self.policy = p; }

    // ---------- Batch (one series × many params) ----------
    pub fn ao_batch_dev(
        &self,
        hl2: &[f32],
        sweep: &AoBatchRange,
    ) -> Result<DeviceArrayF32, CudaAoError> {
        let len = hl2.len();
        if len == 0 { return Err(CudaAoError::InvalidInput("empty series".into())); }

        let first_valid = hl2.iter().position(|v| !v.is_nan())
            .ok_or_else(|| CudaAoError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid(sweep);
        if combos.is_empty() { return Err(CudaAoError::InvalidInput("no parameter combos".into())); }

        // Validate params and warmup
        let mut shorts: Vec<i32> = Vec::with_capacity(combos.len());
        let mut longs: Vec<i32> = Vec::with_capacity(combos.len());
        for prm in &combos {
            let s = prm.short_period.unwrap_or(5) as i32;
            let l = prm.long_period.unwrap_or(34) as i32;
            if s <= 0 || l <= 0 || s >= l {
                return Err(CudaAoError::InvalidInput(format!(
                    "invalid params: short={} long={}", s, l
                )));
            }
            if len - first_valid < (l as usize) {
                return Err(CudaAoError::InvalidInput(format!(
                    "not enough valid data for long={}, tail={} (first_valid={})",
                    l, len - first_valid, first_valid
                )));
            }
            shorts.push(s);
            longs.push(l);
        }

        // Build f64 prefix sums on host (exclusive, length=len+1)
        let mut prefix: Vec<f64> = vec![0.0; len + 1];
        let mut acc = 0.0f64;
        for i in 0..len {
            let vv = hl2[i];
            let v = if vv.is_nan() { 0.0 } else { vv } as f64;
            acc += v;
            prefix[i + 1] = acc;
        }

        // VRAM estimate and headroom
        let rows = combos.len();
        let bytes_prefix = (len + 1) * std::mem::size_of::<f64>();
        let bytes_periods = 2 * rows * std::mem::size_of::<i32>();
        let bytes_out_total = rows * len * std::mem::size_of::<f32>();
        let mut required = bytes_prefix + bytes_periods + bytes_out_total;
        let headroom = 64usize * 1024 * 1024;
        let fits = match mem_get_info() { Ok((free, _)) => required.saturating_add(headroom) <= free, Err(_) => true };

        // Device buffers
        let d_prefix = DeviceBuffer::from_slice(&prefix).map_err(|e| CudaAoError::Cuda(e.to_string()))?;

        // If it fits, do one go; else chunk by combos
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(rows * len) }
            .map_err(|e| CudaAoError::Cuda(e.to_string()))?;

        if fits && rows <= 65_535 {
            let d_shorts = DeviceBuffer::from_slice(&shorts).map_err(|e| CudaAoError::Cuda(e.to_string()))?;
            let d_longs = DeviceBuffer::from_slice(&longs).map_err(|e| CudaAoError::Cuda(e.to_string()))?;
            unsafe { (*(self as *const _ as *mut CudaAo)).last_batch = Some(BatchKernelSelected::Plain { block_x: 256 }); }
            self.launch_batch(&d_prefix, len, first_valid, &d_shorts, &d_longs, rows, &mut d_out)?;
            self.maybe_log_batch_debug();
            return Ok(DeviceArrayF32 { buf: d_out, rows, cols: len });
        }

        // Chunking path
        unsafe { (*(self as *const _ as *mut CudaAo)).last_batch = Some(BatchKernelSelected::Plain { block_x: 256 }); }
        self.maybe_log_batch_debug();
        let max_grid = 65_535usize;
        let mut start = 0usize;
        let mut host_out = vec![0f32; rows * len];
        while start < rows {
            let remain = rows - start;
            let chunk = remain.min(max_grid);
            let d_shorts = DeviceBuffer::from_slice(&shorts[start..start + chunk])
                .map_err(|e| CudaAoError::Cuda(e.to_string()))?;
            let d_longs = DeviceBuffer::from_slice(&longs[start..start + chunk])
                .map_err(|e| CudaAoError::Cuda(e.to_string()))?;
            let mut d_out_chunk: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(chunk * len) }
                .map_err(|e| CudaAoError::Cuda(e.to_string()))?;
            self.launch_batch(&d_prefix, len, first_valid, &d_shorts, &d_longs, chunk, &mut d_out_chunk)?;
            d_out_chunk
                .copy_to(&mut host_out[start * len..start * len + chunk * len])
                .map_err(|e| CudaAoError::Cuda(e.to_string()))?;
            start += chunk;
        }
        let d_out = DeviceBuffer::from_slice(&host_out).map_err(|e| CudaAoError::Cuda(e.to_string()))?;
        Ok(DeviceArrayF32 { buf: d_out, rows, cols: len })
    }

    fn launch_batch(
        &self,
        d_prefix: &DeviceBuffer<f64>,
        len: usize,
        first_valid: usize,
        d_shorts: &DeviceBuffer<i32>,
        d_longs: &DeviceBuffer<i32>,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAoError> {
        let func = self.module.get_function("ao_batch_f32").map_err(|e| CudaAoError::Cuda(e.to_string()))?;
        let block_x = self.policy.batch_block_x.unwrap_or(256);
        let grid_x = ((len as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), n_combos as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut prefix_ptr = d_prefix.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut first_i = first_valid as i32;
            let mut shorts_ptr = d_shorts.as_device_ptr().as_raw();
            let mut longs_ptr = d_longs.as_device_ptr().as_raw();
            let mut combos_i = n_combos as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prefix_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut shorts_ptr as *mut _ as *mut c_void,
                &mut longs_ptr as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args).map_err(|e| CudaAoError::Cuda(e.to_string()))?;
        }
        self.stream.synchronize().map_err(|e| CudaAoError::Cuda(e.to_string()))
    }

    // ---------- Many-series × one-param (time-major) ----------
    pub fn ao_many_series_one_param_time_major_dev(
        &self,
        hl2_tm: &[f32],
        cols: usize,
        rows: usize,
        short: usize,
        long: usize,
    ) -> Result<DeviceArrayF32, CudaAoError> {
        if cols == 0 || rows == 0 { return Err(CudaAoError::InvalidInput("invalid dims".into())); }
        if hl2_tm.len() != cols * rows {
            return Err(CudaAoError::InvalidInput(format!(
                "time-major input length mismatch (expected {}, got {})", cols * rows, hl2_tm.len()
            )));
        }
        if short == 0 || long == 0 || short >= long {
            return Err(CudaAoError::InvalidInput("invalid short/long".into()));
        }

        // Per-series first_valids
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let idx = t * cols + s;
                if !hl2_tm[idx].is_nan() { fv = Some(t as i32); break; }
            }
            let fv = fv.ok_or_else(|| CudaAoError::InvalidInput(format!("series {} all NaN", s)))?;
            if rows - (fv as usize) < long {
                return Err(CudaAoError::InvalidInput(format!(
                    "series {} insufficient data for long {} (tail={})", s, long, rows - fv as usize
                )));
            }
            first_valids[s] = fv;
        }

        let d_prices = DeviceBuffer::from_slice(hl2_tm).map_err(|e| CudaAoError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&first_valids).map_err(|e| CudaAoError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaAoError::Cuda(e.to_string()))?;
        self.launch_many_series(&d_prices, &d_first, cols, rows, short, long, &mut d_out)?;
        Ok(DeviceArrayF32 { buf: d_out, rows, cols })
    }

    fn launch_many_series(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        short: usize,
        long: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAoError> {
        let func = self.module.get_function("ao_many_series_one_param_f32").map_err(|e| CudaAoError::Cuda(e.to_string()))?;
        let block_x = self.policy.many_block_x.unwrap_or(128);
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut nseries_i = cols as i32;
            let mut slen_i = rows as i32;
            let mut short_i = short as i32;
            let mut long_i = long as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut nseries_i as *mut _ as *mut c_void,
                &mut slen_i as *mut _ as *mut c_void,
                &mut short_i as *mut _ as *mut c_void,
                &mut long_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args).map_err(|e| CudaAoError::Cuda(e.to_string()))?;
        }
        self.stream.synchronize().map_err(|e| CudaAoError::Cuda(e.to_string()))
    }

    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_scenario = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] AO batch selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaAo)).debug_batch_logged = true; }
            }
        }
    }
    fn maybe_log_many_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per_scenario = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] AO many-series selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaAo)).debug_many_logged = true; }
            }
        }
    }
}

// ---- Local helpers ----
fn expand_grid(r: &AoBatchRange) -> Vec<AoParams> {
    fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end { return vec![start]; }
        (start..=end).step_by(step).collect()
    }
    let shorts = axis(r.short_period);
    let longs = axis(r.long_period);
    let mut out = Vec::new();
    for &s in &shorts { for &l in &longs { if s > 0 && l > 0 && s < l { out.push(AoParams { short_period: Some(s), long_period: Some(l) }); } } }
    out
}

// ---------- Benches ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250; // ~250 (short,long) pairs

    fn bytes_one_series_many_params() -> usize {
        let prefix_bytes = (ONE_SERIES_LEN + 1) * std::mem::size_of::<f64>();
        let periods_bytes = PARAM_SWEEP * 2 * std::mem::size_of::<i32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        prefix_bytes + periods_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct AoBatchState { cuda: CudaAo, hl2: Vec<f32>, sweep: AoBatchRange }
    impl CudaBenchState for AoBatchState { fn launch(&mut self) { let _ = self.cuda.ao_batch_dev(&self.hl2, &self.sweep).expect("ao batch"); } }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaAo::new(0).expect("cuda ao");
        let hl2 = gen_series(ONE_SERIES_LEN);
        let sweep = AoBatchRange { short_period: (4, 4 + PARAM_SWEEP / 5, 1), long_period: (20, 20 + PARAM_SWEEP, 2) };
        Box::new(AoBatchState { cuda, hl2, sweep })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "ao",
            "one_series_many_params",
            "ao_cuda_batch_dev",
            "1m_x_250",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}
