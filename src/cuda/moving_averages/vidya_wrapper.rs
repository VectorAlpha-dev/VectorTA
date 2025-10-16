//! CUDA support for VIDYA (Variable Index Dynamic Average).
//!
//! Mirrors the ALMA/EMA wrapper patterns:
//! - PTX load via OUT_DIR with DetermineTargetFromContext and OptLevel::O2, with fallbacks
//! - NON_BLOCKING stream
//! - VRAM estimation with ~64MB headroom and grid chunking
//! - Policy enums for batch and many-series variants
//!
//! Math pattern: recurrence/IIR with adaptive factor k = alpha * (short_std / long_std).
//! Warmup/writes match the scalar path in src/indicators/vidya.rs:
//! - warmup index = first_valid + long_period - 2
//! - out[warm-2] = price[warm-2]; out[warm-1] uses initial k

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::vidya::{VidyaBatchRange, VidyaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::error::Error;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaVidyaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaVidyaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaVidyaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaVidyaError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl Error for CudaVidyaError {}

// -------- Policy (kept simple; one block per combo/series) --------

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
}
impl Default for BatchKernelPolicy {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}
impl Default for ManySeriesKernelPolicy {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaVidyaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaVidya {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaVidyaPolicy,
    // Introspection
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
    // Device caps
    max_grid_x: usize,
}

impl CudaVidya {
    pub fn new(device_id: usize) -> Result<Self, CudaVidyaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaVidyaError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaVidyaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaVidyaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/vidya_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => match Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                Ok(m) => m,
                Err(_) => Module::from_ptx(ptx, &[])
                    .map_err(|e| CudaVidyaError::Cuda(e.to_string()))?,
            },
        };

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaVidyaError::Cuda(e.to_string()))?;

        let max_grid_x = device
            .get_attribute(cust::device::DeviceAttribute::MaxGridDimX)
            .map_err(|e| CudaVidyaError::Cuda(e.to_string()))? as usize;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaVidyaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
            max_grid_x,
        })
    }

    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaVidyaError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaVidyaError::Cuda(e.to_string()))
    }

    // ---------- Batch (one-series × many-params) ----------

    pub fn vidya_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &VidyaBatchRange,
    ) -> Result<DeviceArrayF32, CudaVidyaError> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = prepared.combos.len();

        let prices_bytes = prepared.series_len * std::mem::size_of::<f32>();
        let params_bytes = (prepared.short_i32.len() + prepared.long_i32.len()) * 4
            + prepared.alpha_f32.len() * 4;
        let out_bytes = n_combos * prepared.series_len * 4;
        let required = prices_bytes + params_bytes + out_bytes;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaVidyaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices = unsafe {
            DeviceBuffer::from_slice_async(data_f32, &self.stream)
                .map_err(|e| CudaVidyaError::Cuda(e.to_string()))?
        };
        let d_short = unsafe {
            DeviceBuffer::from_slice_async(&prepared.short_i32, &self.stream)
                .map_err(|e| CudaVidyaError::Cuda(e.to_string()))?
        };
        let d_long = unsafe {
            DeviceBuffer::from_slice_async(&prepared.long_i32, &self.stream)
                .map_err(|e| CudaVidyaError::Cuda(e.to_string()))?
        };
        let d_alpha = unsafe {
            DeviceBuffer::from_slice_async(&prepared.alpha_f32, &self.stream)
                .map_err(|e| CudaVidyaError::Cuda(e.to_string()))?
        };
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(n_combos * prepared.series_len, &self.stream)
                .map_err(|e| CudaVidyaError::Cuda(e.to_string()))?
        };

        self.launch_batch_kernel(
            &d_prices,
            &d_short,
            &d_long,
            &d_alpha,
            prepared.series_len,
            prepared.first_valid,
            n_combos,
            &mut d_out,
        )?;

        self.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: prepared.series_len,
        })
    }

    // ---------- Many-series (time-major, one param) ----------

    pub fn vidya_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &VidyaParams,
    ) -> Result<DeviceArrayF32, CudaVidyaError> {
        let prepared = Self::prepare_many_series_inputs(data_tm_f32, num_series, series_len, params)?;

        let prices_bytes = num_series * series_len * 4;
        let params_bytes = prepared.first_valids.len() * 4;
        let out_bytes = num_series * series_len * 4;
        let required = prices_bytes + params_bytes + out_bytes;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaVidyaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices = unsafe {
            DeviceBuffer::from_slice_async(data_tm_f32, &self.stream)
                .map_err(|e| CudaVidyaError::Cuda(e.to_string()))?
        };
        let d_first = unsafe {
            DeviceBuffer::from_slice_async(&prepared.first_valids, &self.stream)
                .map_err(|e| CudaVidyaError::Cuda(e.to_string()))?
        };
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(num_series * series_len, &self.stream)
                .map_err(|e| CudaVidyaError::Cuda(e.to_string()))?
        };

        self.launch_many_series_kernel(
            &d_prices,
            &d_first,
            prepared.short as i32,
            prepared.long as i32,
            prepared.alpha as f32,
            num_series,
            series_len,
            &mut d_out,
        )?;

        self.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: series_len,
            cols: num_series,
        })
    }

    // ---------- Internal launches ----------

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_short: &DeviceBuffer<i32>,
        d_long: &DeviceBuffer<i32>,
        d_alpha: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVidyaError> {
        if n_combos == 0 {
            return Ok(());
        }
        let func = self
            .module
            .get_function("vidya_batch_f32")
            .map_err(|e| CudaVidyaError::Cuda(e.to_string()))?;

        let mut block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            BatchKernelPolicy::Auto => env::var("VIDYA_BLOCK_X")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(256),
        };
        if block_x == 0 {
            block_x = 256;
        }
        unsafe { (*(self as *const _ as *mut CudaVidya)).last_batch = Some(BatchKernelSelected::Plain { block_x }); }
        self.maybe_log_batch_debug();

        // Chunk by device grid.x cap
        let cap = self.max_grid_x.max(1).min(usize::MAX / 2);
        for (start, len) in Self::grid_chunks(n_combos, cap) {
            let grid: GridSize = (len as u32, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            let out_ptr = unsafe { d_out.as_device_ptr().add(start * series_len) };
            let short_ptr = unsafe { d_short.as_device_ptr().add(start) };
            let long_ptr = unsafe { d_long.as_device_ptr().add(start) };
            let alpha_ptr = unsafe { d_alpha.as_device_ptr().add(start) };
            let series_len_i = series_len as i32;
            let first_valid_i = first_valid as i32;
            let n_combos_i = len as i32;
            let stream = &self.stream;
            unsafe {
                launch!(
                    func<<<grid, block, 0, stream>>>(
                        d_prices.as_device_ptr(),
                        short_ptr,
                        long_ptr,
                        alpha_ptr,
                        series_len_i,
                        first_valid_i,
                        n_combos_i,
                        out_ptr
                    )
                )
                .map_err(|e| CudaVidyaError::Cuda(e.to_string()))?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        short_period: i32,
        long_period: i32,
        alpha: f32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVidyaError> {
        if num_series == 0 {
            return Ok(());
        }
        let func = self
            .module
            .get_function("vidya_many_series_one_param_f32")
            .map_err(|e| CudaVidyaError::Cuda(e.to_string()))?;

        let mut block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            ManySeriesKernelPolicy::Auto => env::var("VIDYA_MS_BLOCK_X")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(128),
        };
        if block_x == 0 {
            block_x = 128;
        }
        unsafe { (*(self as *const _ as *mut CudaVidya)).last_many = Some(ManySeriesKernelSelected::OneD { block_x }); }
        self.maybe_log_many_debug();

        // One block per series (compat 1D launch)
        let grid: GridSize = (num_series as u32, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        let stream = &self.stream;
        unsafe {
            launch!(
                func<<<grid, block, 0, stream>>>(
                    d_prices_tm.as_device_ptr(),
                    d_first_valids.as_device_ptr(),
                    short_period,
                    long_period,
                    alpha,
                    num_series as i32,
                    series_len as i32,
                    d_out_tm.as_device_ptr()
                )
            )
            .map_err(|e| CudaVidyaError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    // ---------- Prep helpers ----------

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &VidyaBatchRange,
    ) -> Result<PreparedVidyaBatch, CudaVidyaError> {
        if data_f32.is_empty() {
            return Err(CudaVidyaError::InvalidInput("input data is empty".into()));
        }
        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaVidyaError::InvalidInput(
                "no parameter combinations provided".into(),
            ));
        }
        let series_len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| v.is_finite())
            .ok_or_else(|| CudaVidyaError::InvalidInput("all values are NaN".into()))?;
        let mut short_i32 = Vec::with_capacity(combos.len());
        let mut long_i32 = Vec::with_capacity(combos.len());
        let mut alpha_f32 = Vec::with_capacity(combos.len());
        for p in &combos {
            let sp = p.short_period.unwrap_or(0);
            let lp = p.long_period.unwrap_or(0);
            let a = p.alpha.unwrap_or(-1.0);
            if sp < 2 || lp < sp || lp < 2 || !(0.0..=1.0).contains(&a) {
                return Err(CudaVidyaError::InvalidInput(
                    format!("invalid params: short={}, long={}, alpha={}", sp, lp, a),
                ));
            }
            if series_len - first_valid < lp {
                return Err(CudaVidyaError::InvalidInput(format!(
                    "not enough valid data: need {} valid samples, have {}",
                    lp,
                    series_len - first_valid
                )));
            }
            short_i32.push(sp as i32);
            long_i32.push(lp as i32);
            alpha_f32.push(a as f32);
        }
        Ok(PreparedVidyaBatch {
            combos,
            first_valid,
            series_len,
            short_i32,
            long_i32,
            alpha_f32,
        })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &VidyaParams,
    ) -> Result<PreparedVidyaManySeries, CudaVidyaError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaVidyaError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if data_tm_f32.len() != num_series * series_len {
            return Err(CudaVidyaError::InvalidInput(
                "time-major slice length mismatch".into(),
            ));
        }
        let sp = params.short_period.unwrap_or(0);
        let lp = params.long_period.unwrap_or(0);
        let a = params.alpha.unwrap_or(-1.0);
        if sp < 2 || lp < sp || lp < 2 || !(0.0..=1.0).contains(&a) {
            return Err(CudaVidyaError::InvalidInput(
                format!("invalid params: short={}, long={}, alpha={}", sp, lp, a),
            ));
        }
        let mut first_valids = Vec::with_capacity(num_series);
        for s in 0..num_series {
            let mut fv: Option<usize> = None;
            for t in 0..series_len {
                let v = data_tm_f32[t * num_series + s];
                if v.is_finite() {
                    fv = Some(t);
                    break;
                }
            }
            let fv = fv.ok_or_else(|| {
                CudaVidyaError::InvalidInput(format!("series {} contains only NaNs", s))
            })?;
            let remain = series_len - fv;
            if remain < lp {
                return Err(CudaVidyaError::InvalidInput(format!(
                    "series {} does not have enough valid data: need {} valid samples, have {}",
                    s, lp, remain
                )));
            }
            first_valids.push(fv as i32);
        }
        Ok(PreparedVidyaManySeries {
            first_valids,
            short: sp,
            long: lp,
            alpha: a,
        })
    }

    // ---------- Utilities ----------

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }
    #[inline]
    fn will_fit(required_bytes: usize, headroom: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Ok((free, _)) = mem_get_info() {
            return required_bytes + headroom <= free as usize;
        }
        true
    }
    #[inline]
    fn grid_chunks(total: usize, cap_x: usize) -> impl Iterator<Item = (usize, usize)> {
        struct It { total: usize, cap: usize, start: usize }
        impl Iterator for It {
            type Item = (usize, usize);
            fn next(&mut self) -> Option<Self::Item> {
                if self.start >= self.total { return None; }
                let remain = self.total - self.start;
                let len = remain.min(self.cap);
                let s = self.start;
                self.start += len;
                Some((s, len))
            }
        }
        It { total, cap: cap_x.max(1), start: 0 }
    }
    fn maybe_log_batch_debug(&self) {
        if self.debug_batch_logged { return; }
        if env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(BatchKernelSelected::Plain { block_x }) = self.last_batch {
                eprintln!("[VIDYA] batch kernel: Plain block_x={}", block_x);
            }
        }
        unsafe { (*(self as *const _ as *mut CudaVidya)).debug_batch_logged = true; }
    }
    fn maybe_log_many_debug(&self) {
        if self.debug_many_logged { return; }
        if env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(ManySeriesKernelSelected::OneD { block_x }) = self.last_many {
                eprintln!("[VIDYA] many-series kernel: OneD block_x={}", block_x);
            }
        }
        unsafe { (*(self as *const _ as *mut CudaVidya)).debug_many_logged = true; }
    }
}

// ---------- Prep structs ----------

struct PreparedVidyaBatch {
    combos: Vec<VidyaParams>,
    first_valid: usize,
    series_len: usize,
    short_i32: Vec<i32>,
    long_i32: Vec<i32>,
    alpha_f32: Vec<f32>,
}
struct PreparedVidyaManySeries {
    first_valids: Vec<i32>,
    short: usize,
    long: usize,
    alpha: f64,
}

fn expand_grid(r: &VidyaBatchRange) -> Vec<VidyaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end { return vec![start]; }
        (start..=end).step_by(step).collect()
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 { return vec![start]; }
        let mut v = Vec::new();
        let mut x = start;
        while x <= end + 1e-12 { v.push(x); x += step; }
        v
    }
    let sp = axis_usize(r.short_period);
    let lp = axis_usize(r.long_period);
    let al = axis_f64(r.alpha);
    let mut out = Vec::with_capacity(sp.len() * lp.len() * al.len());
    for &s in &sp { for &l in &lp { for &a in &al {
        out.push(VidyaParams { short_period: Some(s), long_period: Some(l), alpha: Some(a) });
    }}}
    out
}

// ---------- Benches ----------

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;
    const MANY_SERIES_COLS: usize = 256;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * 4;
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * 4;
        in_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = elems * 4;
        let out_bytes = elems * 4;
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct VidyaBatchState {
        cuda: CudaVidya,
        d_prices: DeviceBuffer<f32>,
        d_out: DeviceBuffer<f32>,
        d_short: DeviceBuffer<i32>,
        d_long: DeviceBuffer<i32>,
        d_alpha: DeviceBuffer<f32>,
        first_valid: usize,
        len: usize,
        combos: usize,
        warmed: bool,
    }
    impl CudaBenchState for VidyaBatchState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_prices,
                    &self.d_short,
                    &self.d_long,
                    &self.d_alpha,
                    self.len,
                    self.first_valid,
                    self.combos,
                    &mut self.d_out,
                )
                .expect("vidya batch launch");
            self.cuda.synchronize().expect("sync");
            if !self.warmed { self.warmed = true; }
        }
    }
    fn prep_vidya_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaVidya::new(0).expect("cuda vidya");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = VidyaBatchRange { short_period: (2, 2, 0), long_period: (10, 10 + PARAM_SWEEP - 1, 1), alpha: (0.2, 0.2, 0.0) };
        let combos = super::expand_grid(&sweep);
        let first_valid = price.iter().position(|&x| !x.is_nan()).unwrap_or(0);
        let d_prices = DeviceBuffer::from_slice(&price).expect("d_prices");
        let short_i32: Vec<i32> = combos.iter().map(|p| p.short_period.unwrap() as i32).collect();
        let long_i32: Vec<i32> = combos.iter().map(|p| p.long_period.unwrap() as i32).collect();
        let alpha_f32: Vec<f32> = combos.iter().map(|p| p.alpha.unwrap() as f32).collect();
        let d_short = DeviceBuffer::from_slice(&short_i32).expect("d_short");
        let d_long = DeviceBuffer::from_slice(&long_i32).expect("d_long");
        let d_alpha = DeviceBuffer::from_slice(&alpha_f32).expect("d_alpha");
        let d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(ONE_SERIES_LEN * combos.len()) }.expect("d_out");
        Box::new(VidyaBatchState { cuda, d_prices, d_out, d_short, d_long, d_alpha, first_valid, len: ONE_SERIES_LEN, combos: combos.len(), warmed: false })
    }

    struct VidyaManyState {
        cuda: CudaVidya,
        d_prices_tm: DeviceBuffer<f32>,
        d_first: DeviceBuffer<i32>,
        d_out_tm: DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        sp: i32,
        lp: i32,
        alpha: f32,
        warmed: bool,
    }
    impl CudaBenchState for VidyaManyState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series_kernel(
                    &self.d_prices_tm,
                    &self.d_first,
                    self.sp,
                    self.lp,
                    self.alpha,
                    self.cols,
                    self.rows,
                    &mut self.d_out_tm,
                )
                .expect("vidya many launch");
            self.cuda.synchronize().expect("sync");
            if !self.warmed { self.warmed = true; }
        }
    }
    fn prep_vidya_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaVidya::new(0).expect("cuda vidya");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let prices_tm = gen_time_major_prices(cols, rows);
        let sp = 2;
        let lp = 64;
        let alpha = 0.2f32;
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols { for t in 0..rows { if prices_tm[t*cols+s].is_finite() { first_valids[s] = t as i32; break; } } }
        let d_prices_tm = DeviceBuffer::from_slice(&prices_tm).expect("d_prices_tm");
        let d_first = DeviceBuffer::from_slice(&first_valids).expect("d_first");
        let d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols*rows) }.expect("d_out_tm");
        Box::new(VidyaManyState { cuda, d_prices_tm, d_first, d_out_tm, cols, rows, sp, lp, alpha, warmed: false })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "vidya",
                "one_series_many_params",
                "vidya_cuda_batch_dev",
                "1m_x_250",
                prep_vidya_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "vidya",
                "many_series_one_param",
                "vidya_cuda_many_series_one_param_dev",
                "256x1m",
                prep_vidya_many_series_one_param,
            )
            .with_sample_size(6)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}

