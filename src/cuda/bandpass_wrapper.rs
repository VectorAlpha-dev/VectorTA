//! CUDA wrapper for the Band-Pass indicator.
//!
//! Design mirrors ALMA and WTO wrappers:
//! - VRAM-first API returning device-resident arrays
//! - Non-blocking stream
//! - JIT PTX load with DetermineTargetFromContext and O2 fallback
//! - Simple policy/introspection hooks kept minimal (no tiling variants yet)
//! - VRAM estimation and conservative headroom; grid chunking where needed
//!
//! Math pattern: Recurrence/IIR. We reuse GPU High-Pass precompute across
//! parameter rows in batch by launching the existing `highpass_batch_f32`
//! kernel once for the unique `hp_period`s, then applying the band-pass
//! recurrence per row. For many-series×one-param we reuse the time-major
//! high-pass kernel and run a per-series recurrence.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::{CudaHighpass, DeviceArrayF32};
use crate::indicators::bandpass::{BandPassBatchRange, BandPassParams};
use cust::context::{CacheConfig, Context};
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::collections::HashMap;
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaBandpassError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaBandpassError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaBandpassError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaBandpassError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaBandpassError {}

pub struct DeviceArrayF32Quad {
    pub first: DeviceArrayF32,  // bp
    pub second: DeviceArrayF32, // bp_normalized
    pub third: DeviceArrayF32,  // signal
    pub fourth: DeviceArrayF32, // trigger
}

impl DeviceArrayF32Quad {
    #[inline]
    pub fn rows(&self) -> usize {
        self.first.rows
    }
    #[inline]
    pub fn cols(&self) -> usize {
        self.first.cols
    }
}

pub struct CudaBandpassBatchResult {
    pub outputs: DeviceArrayF32Quad,
    pub combos: Vec<BandPassParams>,
}

#[derive(Clone, Copy, Debug, Default)]
pub enum BatchKernelPolicy {
    #[default]
    Auto,
    Plain {
        block_x: u32,
    },
}

#[derive(Clone, Copy, Debug, Default)]
pub enum ManySeriesKernelPolicy {
    #[default]
    Auto,
    OneD {
        block_x: u32,
    },
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaBandpassPolicy {
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

pub struct CudaBandpass {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaBandpassPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaBandpass {
    pub fn new(device_id: usize) -> Result<Self, CudaBandpassError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaBandpassError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaBandpassError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaBandpassError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/bandpass_kernel.ptx"));
        let module = Module::from_ptx(
            ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O2),
            ],
        )
        .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
        .or_else(|_| Module::from_ptx(ptx, &[]))
        .map_err(|e| CudaBandpassError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaBandpassError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaBandpassPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    #[inline]
    pub fn stream(&self) -> &Stream { &self.stream }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                if !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] bandpass batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaBandpass)).debug_batch_logged = true;
                }
            }
        }
    }

    #[inline]
    fn maybe_log_many_debug(&self) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                if !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] bandpass many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaBandpass)).debug_many_logged = true;
                }
            }
        }
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }
    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }
    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Some((free, _)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    fn expand_grid(range: &BandPassBatchRange) -> Vec<BandPassParams> {
        fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
            if step == 0 || start == end {
                return vec![start];
            }
            (start..=end).step_by(step).collect()
        }
        fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
            if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
                return vec![start];
            }
            let mut out = Vec::new();
            let mut x = start;
            while x <= end + 1e-12 {
                out.push(x);
                x += step;
            }
            out
        }
        let periods = axis_usize(range.period);
        let bands = axis_f64(range.bandwidth);
        let mut v = Vec::with_capacity(periods.len() * bands.len());
        for &p in &periods {
            for &b in &bands {
                v.push(BandPassParams {
                    period: Some(p),
                    bandwidth: Some(b),
                });
            }
        }
        v
    }

    fn prepare_batch(
        data_f32: &[f32],
        sweep: &BandPassBatchRange,
    ) -> Result<(Vec<BandPassParams>, usize, usize), CudaBandpassError> {
        if data_f32.is_empty() {
            return Err(CudaBandpassError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaBandpassError::InvalidInput("all values are NaN".into()))?;
        let len = data_f32.len();
        let combos = Self::expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaBandpassError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        for p in &combos {
            let period = p.period.unwrap_or(0);
            let bw = p.bandwidth.unwrap_or(0.0);
            if period < 2 || period > len {
                return Err(CudaBandpassError::InvalidInput(format!(
                    "invalid period {} for len {}",
                    period, len
                )));
            }
            if !(0.0..=1.0).contains(&bw) || !bw.is_finite() || bw == 0.0 {
                return Err(CudaBandpassError::InvalidInput(format!(
                    "invalid bandwidth {}",
                    bw
                )));
            }
            // Ensure enough valid after first for HP stage
            let hp_period = (4.0 * period as f64 / bw).round() as usize;
            if len - first_valid < hp_period {
                return Err(CudaBandpassError::InvalidInput(format!(
                    "not enough valid data: need >= {}, have {}",
                    hp_period,
                    len - first_valid
                )));
            }
        }
        Ok((combos, first_valid, len))
    }

    fn host_coeffs(period: usize, bw: f64) -> (f32, f32, i32, usize) {
        use std::f64::consts::PI;
        let beta = (2.0 * PI / period as f64).cos();
        let gamma = (2.0 * PI * bw / period as f64).cos();
        // alpha = 1/g - sqrt(1/g^2 - 1)
        let alpha = 1.0 / gamma - ((1.0 / (gamma * gamma)) - 1.0).sqrt();
        let trig = ((period as f64 / bw) / 1.5).round() as i32;
        let hp_period = (4.0 * period as f64 / bw).round() as usize;
        (alpha as f32, beta as f32, trig, hp_period)
    }

    pub fn bandpass_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &BandPassBatchRange,
    ) -> Result<CudaBandpassBatchResult, CudaBandpassError> {
        let (combos, _first_valid, len) = Self::prepare_batch(data_f32, sweep)?;
        let n = len;
        let rows = combos.len();

        // Build per-combo coeffs + hp-periods, dedupe hp rows
        let mut alphas = vec![0f32; rows];
        let mut betas = vec![0f32; rows];
        let mut trig = vec![0i32; rows];
        let mut hp_row_idx = vec![0i32; rows];
        let mut hp_map: HashMap<usize, usize> = HashMap::new();
        let mut hp_unique: Vec<i32> = Vec::new();
        for (i, p) in combos.iter().enumerate() {
            let period = p.period.unwrap();
            let bw = p.bandwidth.unwrap();
            let (a, b, t, hp_p) = Self::host_coeffs(period, bw);
            alphas[i] = a;
            betas[i] = b;
            trig[i] = t;
            let idx = *hp_map.entry(hp_p).or_insert_with(|| {
                hp_unique.push(hp_p as i32);
                hp_unique.len() - 1
            });
            hp_row_idx[i] = idx as i32;
        }

        // VRAM estimate: prices + hp unique rows + params + 4 outputs
        let prices_bytes = n * std::mem::size_of::<f32>();
        let hp_bytes = hp_unique.len() * n * std::mem::size_of::<f32>();
        let params_bytes = rows * (2 * std::mem::size_of::<f32>() + 2 * std::mem::size_of::<i32>());
        let outs_bytes = 4 * rows * n * std::mem::size_of::<f32>();
        let required = prices_bytes + hp_bytes + params_bytes + outs_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaBandpassError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        // Stage 1: push prices + compute HP unique rows on device using CudaHighpass
        // Use synchronous HtoD for prices to avoid cross-stream hazards with CudaHighpass
        // (its kernels run on a different stream).
        let d_prices = DeviceBuffer::from_slice(data_f32)
            .map_err(|e| CudaBandpassError::Cuda(e.to_string()))?;
        // NOTE: This buffer is consumed by CudaHighpass on its own stream.
        // Keep this copy synchronous to avoid cross-stream hazards without events.
        let d_hp_periods = DeviceBuffer::from_slice(&hp_unique)
            .map_err(|e| CudaBandpassError::Cuda(e.to_string()))?;
        let mut d_hp: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(hp_unique.len() * n, &self.stream)
                .map_err(|e| CudaBandpassError::Cuda(e.to_string()))?
        };
        let cuda_hp = CudaHighpass::new(0).map_err(|e| CudaBandpassError::Cuda(e.to_string()))?;
        cuda_hp
            .highpass_batch_device(
                &d_prices,
                &d_hp_periods,
                n as i32,
                hp_unique.len() as i32,
                &mut d_hp,
            )
            .map_err(|e| CudaBandpassError::Cuda(e.to_string()))?;

        // Params to device
        let d_hp_idx = DeviceBuffer::from_slice(&hp_row_idx)
            .map_err(|e| CudaBandpassError::Cuda(e.to_string()))?;
        let d_alpha = DeviceBuffer::from_slice(&alphas)
            .map_err(|e| CudaBandpassError::Cuda(e.to_string()))?;
        let d_beta =
            DeviceBuffer::from_slice(&betas).map_err(|e| CudaBandpassError::Cuda(e.to_string()))?;
        let d_trig =
            DeviceBuffer::from_slice(&trig).map_err(|e| CudaBandpassError::Cuda(e.to_string()))?;

        // Outputs
        let mut d_bp: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(rows * n, &self.stream) }
            .map_err(|e| CudaBandpassError::Cuda(e.to_string()))?;
        let mut d_bpn: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(rows * n, &self.stream) }
            .map_err(|e| CudaBandpassError::Cuda(e.to_string()))?;
        let mut d_sig: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(rows * n, &self.stream) }
            .map_err(|e| CudaBandpassError::Cuda(e.to_string()))?;
        let mut d_trg: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(rows * n, &self.stream) }
            .map_err(|e| CudaBandpassError::Cuda(e.to_string()))?;

        // Launch bandpass kernel
        let mut func = self
            .module
            .get_function("bandpass_batch_from_hp_f32")
            .map_err(|e| CudaBandpassError::Cuda(e.to_string()))?;
        // Prefer L1 for read-heavy streaming kernel; ignore errors if unsupported.
        let _ = func.set_cache_config(CacheConfig::PreferL1);
        // occupancy suggestion
        let (suggested, _min_grid) = func
            .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
            .map_err(|e| CudaBandpassError::Cuda(e.to_string()))?;
        let bx = match self.policy.batch {
            BatchKernelPolicy::Auto => suggested.max(128),
            BatchKernelPolicy::Plain { block_x } => block_x.max(32),
        };
        unsafe {
            (*(self as *const _ as *mut CudaBandpass)).last_batch =
                Some(BatchKernelSelected::Plain { block_x: bx });
        }
        let block: BlockSize = (bx, 1, 1).into();
        let grid_x = ((rows as u32) + block.x - 1) / block.x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();

        unsafe {
            let mut hp_ptr = d_hp.as_device_ptr().as_raw();
            let mut hp_rows_i = hp_unique.len() as i32;
            let mut len_i = n as i32;
            let mut hp_idx_ptr = d_hp_idx.as_device_ptr().as_raw();
            let mut alpha_ptr = d_alpha.as_device_ptr().as_raw();
            let mut beta_ptr = d_beta.as_device_ptr().as_raw();
            let mut trig_ptr = d_trig.as_device_ptr().as_raw();
            let mut combos_i = rows as i32;
            let mut bp_ptr = d_bp.as_device_ptr().as_raw();
            let mut bpn_ptr = d_bpn.as_device_ptr().as_raw();
            let mut sig_ptr = d_sig.as_device_ptr().as_raw();
            let mut trg_ptr = d_trg.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut hp_ptr as *mut _ as *mut c_void,
                &mut hp_rows_i as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut hp_idx_ptr as *mut _ as *mut c_void,
                &mut alpha_ptr as *mut _ as *mut c_void,
                &mut beta_ptr as *mut _ as *mut c_void,
                &mut trig_ptr as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut bp_ptr as *mut _ as *mut c_void,
                &mut bpn_ptr as *mut _ as *mut c_void,
                &mut sig_ptr as *mut _ as *mut c_void,
                &mut trg_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaBandpassError::Cuda(e.to_string()))?;
        }

        self.maybe_log_batch_debug();

        Ok(CudaBandpassBatchResult {
            outputs: DeviceArrayF32Quad {
                first: DeviceArrayF32 {
                    buf: d_bp,
                    rows,
                    cols: n,
                },
                second: DeviceArrayF32 {
                    buf: d_bpn,
                    rows,
                    cols: n,
                },
                third: DeviceArrayF32 {
                    buf: d_sig,
                    rows,
                    cols: n,
                },
                fourth: DeviceArrayF32 {
                    buf: d_trg,
                    rows,
                    cols: n,
                },
            },
            combos,
        })
    }

    pub fn bandpass_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &BandPassParams,
    ) -> Result<DeviceArrayF32Quad, CudaBandpassError> {
        if cols == 0 || rows == 0 || data_tm_f32.len() != cols * rows {
            return Err(CudaBandpassError::InvalidInput("invalid cols/rows".into()));
        }
        let period = params.period.unwrap_or(0);
        let bw = params.bandwidth.unwrap_or(0.0);
        if period < 2 || !(0.0..=1.0).contains(&bw) || bw == 0.0 {
            return Err(CudaBandpassError::InvalidInput("invalid params".into()));
        }
        let (_a, _b, _trig, hp_period) = Self::host_coeffs(period, bw);

        // Compute HP time-major for this param
        let cuda_hp = CudaHighpass::new(0).map_err(|e| CudaBandpassError::Cuda(e.to_string()))?;
        let hp_dev = cuda_hp
            .highpass_many_series_one_param_time_major_dev(
                data_tm_f32,
                cols,
                rows,
                &crate::indicators::moving_averages::highpass::HighPassParams {
                    period: Some(hp_period),
                },
            )
            .map_err(|e| CudaBandpassError::Cuda(e.to_string()))?;

        // Prepare params
        let (alpha, beta, trig, _hp) = Self::host_coeffs(period, bw);
        // No device copies of scalars; they are passed by value to the kernel.

        // Outputs (time-major)
        let total = cols * rows;
        let mut d_bp: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(total, &self.stream) }
            .map_err(|e| CudaBandpassError::Cuda(e.to_string()))?;
        let mut d_bpn: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(total, &self.stream) }
            .map_err(|e| CudaBandpassError::Cuda(e.to_string()))?;
        let mut d_sig: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(total, &self.stream) }
            .map_err(|e| CudaBandpassError::Cuda(e.to_string()))?;
        let mut d_trg_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(total, &self.stream) }
            .map_err(|e| CudaBandpassError::Cuda(e.to_string()))?;

        // Launch many-series kernel
        let mut func = self
            .module
            .get_function("bandpass_many_series_one_param_time_major_from_hp_f32")
            .map_err(|e| CudaBandpassError::Cuda(e.to_string()))?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);
        let (suggested, _mg) = func
            .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
            .map_err(|e| CudaBandpassError::Cuda(e.to_string()))?;
        let bx = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => suggested.max(128),
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32),
        };
        unsafe {
            (*(self as *const _ as *mut CudaBandpass)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x: bx });
        }
        let grid_x = ((cols as u32) + bx - 1) / bx;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (bx, 1, 1).into();

        unsafe {
            let mut hp_ptr = hp_dev.buf.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut alpha_f = alpha;
            let mut beta_f = beta;
            let mut trig_i = trig;
            let mut bp_ptr = d_bp.as_device_ptr().as_raw();
            let mut bpn_ptr = d_bpn.as_device_ptr().as_raw();
            let mut sig_ptr = d_sig.as_device_ptr().as_raw();
            let mut trg_ptr = d_trg_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut hp_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut alpha_f as *mut _ as *mut c_void,
                &mut beta_f as *mut _ as *mut c_void,
                &mut trig_i as *mut _ as *mut c_void,
                &mut bp_ptr as *mut _ as *mut c_void,
                &mut bpn_ptr as *mut _ as *mut c_void,
                &mut sig_ptr as *mut _ as *mut c_void,
                &mut trg_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaBandpassError::Cuda(e.to_string()))?;
        }

        self.maybe_log_many_debug();

        Ok(DeviceArrayF32Quad {
            first: DeviceArrayF32 {
                buf: d_bp,
                rows,
                cols,
            },
            second: DeviceArrayF32 {
                buf: d_bpn,
                rows,
                cols,
            },
            third: DeviceArrayF32 {
                buf: d_sig,
                rows,
                cols,
            },
            fourth: DeviceArrayF32 {
                buf: d_trg_out,
                rows,
                cols,
            },
        })
    }
}

// ---------- Benches ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const MANY_SERIES_COLS: usize = 512;
    const MANY_SERIES_ROWS: usize = 8_192;

    fn bytes_one_series(rows: usize, len: usize) -> usize {
        // prices + hp(unique ~ 1) + 4 outputs
        let in_b = len * std::mem::size_of::<f32>();
        let hp_b = len * std::mem::size_of::<f32>();
        let out_b = 4 * rows * len * std::mem::size_of::<f32>();
        in_b + hp_b + out_b + 32 * 1024 * 1024
    }

    struct BPBatchState {
        cuda: CudaBandpass,
        prices: Vec<f32>,
        sweep: BandPassBatchRange,
    }
    impl CudaBenchState for BPBatchState {
        fn launch(&mut self) {
            let _ = self.cuda.bandpass_batch_dev(&self.prices, &self.sweep);
        }
    }

    fn prep_one_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaBandpass::new(0).expect("cuda bandpass");
        let prices = gen_series(ONE_SERIES_LEN);
        // Small sweep to avoid huge VRAM usage
        let sweep = BandPassBatchRange {
            period: (16, 22, 2),
            bandwidth: (0.2, 0.4, 0.1),
        };
        Box::new(BPBatchState {
            cuda,
            prices,
            sweep,
        })
    }

    struct BPManyState {
        cuda: CudaBandpass,
        data_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        params: BandPassParams,
    }
    impl CudaBenchState for BPManyState {
        fn launch(&mut self) {
            let _ = self.cuda.bandpass_many_series_one_param_time_major_dev(
                &self.data_tm,
                self.cols,
                self.rows,
                &self.params,
            );
        }
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaBandpass::new(0).expect("cuda bandpass");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_ROWS;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = BandPassParams {
            period: Some(20),
            bandwidth: Some(0.3),
        };
        Box::new(BPManyState {
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
                "bandpass",
                "batch_one_series",
                "bandpass_cuda_batch",
                "1m",
                prep_one_series,
            )
            .with_mem_required(bytes_one_series(6, ONE_SERIES_LEN)),
            CudaBenchScenario::new(
                "bandpass",
                "many_series_one_param",
                "bandpass_cuda_many_series_one_param",
                "tm",
                prep_many_series_one_param,
            ),
        ]
    }
}
