//! CUDA wrapper for Variable Length Moving Average (VLMA)
//!
//! Scope (matching CPU fast path and ALMA-style API):
//! - One-series × many-params using SMA reference + StdDev bands via host prefix sums.
//! - Many-series × one-param (time-major) using per-series rolling sums on device.
//!
//! Parity guarantees:
//! - Warmup/NaN semantics identical to scalar VLMA: preserve the value at `first_valid`,
//!   write NaN elsewhere before warmup; pass through NaNs after warmup without state updates.
//! - API shape mirrors ALMA: device-backed array handle, NON_BLOCKING stream, JIT options
//!   with DetermineTargetFromContext + OptLevel O2, VRAM checks, and optional debug logging
//!   of selected kernels when BENCH_DEBUG=1.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::vlma::{expand_grid_vlma, VlmaBatchRange, VlmaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CudaVlmaError {
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
    #[error(
        "launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})"
    )]
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

// -------- Kernel selection policy (kept for ALMA parity/debug; single impl under the hood) --------

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
pub struct CudaVlmaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaVlmaPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    SmaStdPrefix { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    TimeMajor { block_x: u32 },
}

pub struct CudaVlma {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaVlmaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaVlma {
    pub fn new(device_id: usize) -> Result<Self, CudaVlmaError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Context::new(device)?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/vlma_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) =
                    Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
                {
                    m
                } else {
                    Module::from_ptx(ptx, &[])?
                }
            }
        };

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaVlmaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(
        device_id: usize,
        policy: CudaVlmaPolicy,
    ) -> Result<Self, CudaVlmaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaVlmaPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaVlmaPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
    pub fn synchronize(&self) -> Result<(), CudaVlmaError> {
        self.stream.synchronize().map_err(Into::into)
    }

    #[inline]
    fn maybe_log_batch(&self) {
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_scen = env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scen || !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] VLMA batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaVlma)).debug_batch_logged = true;
                }
            }
        }
    }
    #[inline]
    fn maybe_log_many(&self) {
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged {
            return;
        }
        if env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per_scen = env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scen || !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] VLMA many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaVlma)).debug_many_logged = true;
                }
            }
        }
    }

    // ----- Utilities -----
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
        if let Some((free, _total)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    #[inline]
    fn validate_launch(
        &self,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    ) -> Result<(), CudaVlmaError> {
        let dev = Device::get_device(self.device_id)?;
        let max_threads = dev
            .get_attribute(cust::device::DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_bx = dev
            .get_attribute(cust::device::DeviceAttribute::MaxBlockDimX)
            .unwrap_or(1024) as u32;
        let max_by = dev
            .get_attribute(cust::device::DeviceAttribute::MaxBlockDimY)
            .unwrap_or(1024) as u32;
        let max_bz = dev
            .get_attribute(cust::device::DeviceAttribute::MaxBlockDimZ)
            .unwrap_or(64) as u32;
        let max_gx = dev
            .get_attribute(cust::device::DeviceAttribute::MaxGridDimX)
            .unwrap_or(2_147_483_647) as u32;
        let max_gy = dev
            .get_attribute(cust::device::DeviceAttribute::MaxGridDimY)
            .unwrap_or(65_535) as u32;
        let max_gz = dev
            .get_attribute(cust::device::DeviceAttribute::MaxGridDimZ)
            .unwrap_or(65_535) as u32;

        let threads = bx.saturating_mul(by).saturating_mul(bz);
        if threads > max_threads
            || bx > max_bx
            || by > max_by
            || bz > max_bz
            || gx > max_gx
            || gy > max_gy
            || gz > max_gz
        {
            return Err(CudaVlmaError::LaunchConfigTooLarge {
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

    // ----- Host-side helpers -----
    fn build_prefixes(data: &[f32]) -> (Vec<f64>, Vec<f64>, Vec<i32>) {
        let n = data.len();
        let mut ps = vec![0.0f64; n + 1];
        let mut pss = vec![0.0f64; n + 1];
        let mut pn = vec![0i32; n + 1];
        let mut acc = 0.0f64;
        let mut acc2 = 0.0f64;
        let mut nan = 0i32;
        for i in 0..n {
            let v = data[i];
            if v.is_nan() {
                nan += 1;
            } else {
                let d = v as f64;
                acc += d;
                acc2 += d * d;
            }
            ps[i + 1] = acc;
            pss[i + 1] = acc2;
            pn[i + 1] = nan;
        }
        (ps, pss, pn)
    }

    fn expand_supported_combos(sweep: &VlmaBatchRange) -> Result<Vec<VlmaParams>, CudaVlmaError> {
        let combos = expand_grid_vlma(sweep)
            .map_err(|e| CudaVlmaError::InvalidInput(e.to_string()))?;
        Ok(combos
            .into_iter()
            .filter(|p| p.matype.as_deref() == Some("sma") && p.devtype == Some(0))
            .collect())
    }

    // ----- Kernel launches -----
    fn launch_batch(
        &mut self,
        d_prices: &DeviceBuffer<f32>,
        d_ps: &DeviceBuffer<f64>,
        d_pss: &DeviceBuffer<f64>,
        d_pn: &DeviceBuffer<i32>,
        d_min: &DeviceBuffer<i32>,
        d_max: &DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVlmaError> {
        let func = self
            .module
            .get_function("vlma_batch_sma_std_prefix_f32")
            .map_err(|_| CudaVlmaError::MissingKernelSymbol {
                name: "vlma_batch_sma_std_prefix_f32",
            })?;

        // Chunk combos to respect grid limits (y/x either way). We use grid.x here.
        let bx = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x.max(32),
            BatchKernelPolicy::Auto => 128,
        };

        let mut launched = 0usize;
        while launched < n_combos {
            let this = (n_combos - launched).min(65_535usize);
            let grid: GridSize = (this as u32, 1, 1).into();
            let block: BlockSize = (bx, 1, 1).into();

            self.validate_launch(grid.x, grid.y, grid.z, block.x, block.y, block.z)?;

            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                let mut ps_ptr = d_ps.as_device_ptr().as_raw();
                let mut pss_ptr = d_pss.as_device_ptr().as_raw();
                let mut pn_ptr = d_pn.as_device_ptr().as_raw();
                let mut min_ptr = d_min
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched) as u64 * 4);
                let mut max_ptr = d_max
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched) as u64 * 4);
                let mut len_i = len as i32;
                let mut first_i = first_valid as i32;
                let mut combos_i = this as i32;
                let mut out_ptr = d_out
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * len) as u64 * 4);

                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut ps_ptr as *mut _ as *mut c_void,
                    &mut pss_ptr as *mut _ as *mut c_void,
                    &mut pn_ptr as *mut _ as *mut c_void,
                    &mut min_ptr as *mut _ as *mut c_void,
                    &mut max_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut combos_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];

                self.stream.launch(&func, grid, block, 0, args)?;
            }

            launched += this;
        }

        self.last_batch = Some(BatchKernelSelected::SmaStdPrefix { block_x: bx });
        self.maybe_log_batch();
        Ok(())
    }

    fn launch_many_series(
        &mut self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        min_p: usize,
        max_p: usize,
        cols: usize,
        rows: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVlmaError> {
        let func = self
            .module
            .get_function("vlma_many_series_one_param_f32")
            .map_err(|_| CudaVlmaError::MissingKernelSymbol {
                name: "vlma_many_series_one_param_f32",
            })?;
        let bx = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32),
            ManySeriesKernelPolicy::Auto => 128,
        };
        let grid: GridSize = (cols as u32, 1, 1).into();
        let block: BlockSize = (bx, 1, 1).into();

        self.validate_launch(grid.x, grid.y, grid.z, block.x, block.y, block.z)?;

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut min_i = min_p as i32;
            let mut max_i = max_p as i32;
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut min_i as *mut _ as *mut c_void,
                &mut max_i as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        self.last_many = Some(ManySeriesKernelSelected::TimeMajor { block_x: bx });
        self.maybe_log_many();
        Ok(())
    }

    // ----- Public API -----
    pub fn vlma_batch_dev(
        &mut self,
        data_f32: &[f32],
        sweep: &VlmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaVlmaError> {
        if data_f32.is_empty() {
            return Err(CudaVlmaError::InvalidInput("empty data".into()));
        }
        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaVlmaError::InvalidInput("all values are NaN".into()))?;

        // Supported combos only: SMA + stddev (devtype=0)
        let combos = Self::expand_supported_combos(sweep)?;
        if combos.is_empty() {
            return Err(CudaVlmaError::InvalidInput(
                "no supported parameter combinations (require matype='sma', devtype=0)".into(),
            ));
        }
        // Validate periods and valid window
        for c in &combos {
            let max_p = c.max_period.unwrap_or(0);
            if max_p == 0 || max_p > len {
                return Err(CudaVlmaError::InvalidInput(format!(
                    "invalid max_period {} for length {}",
                    max_p, len
                )));
            }
            if len - first_valid < max_p {
                return Err(CudaVlmaError::InvalidInput(format!(
                    "not enough valid data for max_period {} (valid after first {}: {})",
                    max_p,
                    first_valid,
                    len - first_valid
                )));
            }
        }

        // Estimate VRAM: prices + 3 prefixes + periods + output
        let (ps, pss, pn) = Self::build_prefixes(data_f32);
        let n = len;
        let m = combos.len();
        let prices_b = n
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaVlmaError::InvalidInput("prices size overflow".into()))?;
        let prefixes_b = (n + 1)
            .checked_mul(std::mem::size_of::<f64>() * 2 + std::mem::size_of::<i32>())
            .ok_or_else(|| CudaVlmaError::InvalidInput("prefix size overflow".into()))?;
        let periods_b = m
            .checked_mul(std::mem::size_of::<i32>() * 2)
            .ok_or_else(|| CudaVlmaError::InvalidInput("periods size overflow".into()))?;
        let out_b = m
            .checked_mul(n)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaVlmaError::InvalidInput("output size overflow".into()))?;
        let bytes = prices_b
            .checked_add(prefixes_b)
            .and_then(|x| x.checked_add(periods_b))
            .and_then(|x| x.checked_add(out_b))
            .ok_or_else(|| CudaVlmaError::InvalidInput("total VRAM size overflow".into()))?;
        let headroom = 64 * 1024 * 1024; // 64MB
        if !Self::will_fit(bytes, headroom) {
            if let Some((free, _total)) = Self::device_mem_info() {
                return Err(CudaVlmaError::OutOfMemory {
                    required: bytes,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaVlmaError::InvalidInput("insufficient device memory".into()));
            }
        }

        // Upload inputs
        let d_prices = DeviceBuffer::from_slice(data_f32)?;
        let d_ps = DeviceBuffer::from_slice(&ps)?;
        let d_pss = DeviceBuffer::from_slice(&pss)?;
        let d_pn = DeviceBuffer::from_slice(&pn)?;
        let min_periods: Vec<i32> = combos
            .iter()
            .map(|c| c.min_period.unwrap_or(1) as i32)
            .collect();
        let max_periods: Vec<i32> = combos
            .iter()
            .map(|c| c.max_period.unwrap() as i32)
            .collect();
        let d_min = DeviceBuffer::from_slice(&min_periods)?;
        let d_max = DeviceBuffer::from_slice(&max_periods)?;

        // Output buffer
        let total_elems = m
            .checked_mul(n)
            .ok_or_else(|| CudaVlmaError::InvalidInput("m * n overflowed".into()))?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(total_elems) }?;

        // Launch
        self.launch_batch(
            &d_prices,
            &d_ps,
            &d_pss,
            &d_pn,
            &d_min,
            &d_max,
            n,
            first_valid,
            m,
            &mut d_out,
        )?;
        self.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: m,
            cols: n,
        })
    }

    pub fn vlma_many_series_one_param_time_major_dev(
        &mut self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &VlmaParams,
    ) -> Result<DeviceArrayF32, CudaVlmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaVlmaError::InvalidInput("empty matrix".into()));
        }
        if cols
            .checked_mul(rows)
            .map(|n| n != data_tm_f32.len())
            .unwrap_or(true)
        {
            return Err(CudaVlmaError::InvalidInput(
                "flat input length mismatch".into(),
            ));
        }
        if params.matype.as_deref() != Some("sma") || params.devtype.unwrap_or(0) != 0 {
            return Err(CudaVlmaError::InvalidInput(
                "only matype='sma' and devtype=0 supported in CUDA".into(),
            ));
        }
        let min_p = params.min_period.unwrap_or(5);
        let max_p = params.max_period.unwrap_or(50);
        if min_p == 0 || max_p == 0 || min_p > max_p {
            return Err(CudaVlmaError::InvalidInput("invalid periods".into()));
        }

        // Compute first_valid per series and validate
        let mut first_valids = Vec::with_capacity(cols);
        for s in 0..cols {
            let mut fv: Option<usize> = None;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + s];
                if !v.is_nan() {
                    fv = Some(t);
                    break;
                }
            }
            let fv =
                fv.ok_or_else(|| CudaVlmaError::InvalidInput(format!("series {} all NaN", s)))?;
            if rows - fv < max_p {
                return Err(CudaVlmaError::InvalidInput(format!(
                    "series {} not enough valid data (need {}, have {})",
                    s,
                    max_p,
                    rows - fv
                )));
            }
            first_valids.push(fv as i32);
        }

        // VRAM estimate: input + first_valids + output
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaVlmaError::InvalidInput("cols * rows overflowed".into()))?;
        let bytes_in_out = elems
            .checked_mul(std::mem::size_of::<f32>() * 2)
            .ok_or_else(|| CudaVlmaError::InvalidInput("VRAM size overflow".into()))?;
        let bytes_first = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaVlmaError::InvalidInput("first_valid size overflow".into()))?;
        let bytes = bytes_in_out
            .checked_add(bytes_first)
            .ok_or_else(|| CudaVlmaError::InvalidInput("total VRAM size overflow".into()))?;
        let headroom = 32 * 1024 * 1024;
        if !Self::will_fit(bytes, headroom) {
            if let Some((free, _total)) = Self::device_mem_info() {
                return Err(CudaVlmaError::OutOfMemory {
                    required: bytes,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaVlmaError::InvalidInput("insufficient device memory".into()));
            }
        }

        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32)?;
        let d_first = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out_tm = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }?;

        self.launch_many_series(
            &d_prices_tm,
            &d_first,
            min_p,
            max_p,
            cols,
            rows,
            &mut d_out_tm,
        )?;
        self.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows,
            cols,
        })
    }
}

// ---------- Bench profiles (simple smoke; align with ALMA style names) ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::vlma::VlmaBatchRange;

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 200;

    fn bytes_one_series_many_params() -> usize {
        let in_b = ONE_SERIES_LEN * 4;
        let out_b = ONE_SERIES_LEN * PARAM_SWEEP * 4;
        // add rough prefixes and headroom
        in_b + out_b + (ONE_SERIES_LEN + 1) * (8 + 8 + 4) + 64 * 1024 * 1024
    }

    struct VlmaBatchState {
        cuda: CudaVlma,
        d_prices: DeviceBuffer<f32>,
        d_ps: DeviceBuffer<f64>,
        d_pss: DeviceBuffer<f64>,
        d_pn: DeviceBuffer<i32>,
        d_min: DeviceBuffer<i32>,
        d_max: DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for VlmaBatchState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch(
                    &self.d_prices,
                    &self.d_ps,
                    &self.d_pss,
                    &self.d_pn,
                    &self.d_min,
                    &self.d_max,
                    self.len,
                    self.first_valid,
                    self.n_combos,
                    &mut self.d_out,
                )
                .expect("vlma batch kernel");
            self.cuda.synchronize().expect("vlma sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let mut cuda = CudaVlma::new(0).expect("cuda vlma");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = VlmaBatchRange {
            min_period: (5, 5, 0),
            max_period: (20, 20 + PARAM_SWEEP - 1, 1),
            matype: ("sma".to_string(), "sma".to_string(), "".to_string()),
            devtype: (0, 0, 0),
        };
        let first_valid = price
            .iter()
            .position(|v| !v.is_nan())
            .unwrap_or(0);
        let combos = CudaVlma::expand_supported_combos(&sweep).expect("vlma expand combos");
        let n_combos = combos.len();
        let (ps, pss, pn) = CudaVlma::build_prefixes(&price);
        let min_periods: Vec<i32> = combos.iter().map(|c| c.min_period.unwrap_or(1) as i32).collect();
        let max_periods: Vec<i32> = combos.iter().map(|c| c.max_period.unwrap_or(1) as i32).collect();

        let d_prices = DeviceBuffer::from_slice(&price).expect("d_prices");
        let d_ps = DeviceBuffer::from_slice(&ps).expect("d_ps");
        let d_pss = DeviceBuffer::from_slice(&pss).expect("d_pss");
        let d_pn = DeviceBuffer::from_slice(&pn).expect("d_pn");
        let d_min = DeviceBuffer::from_slice(&min_periods).expect("d_min");
        let d_max = DeviceBuffer::from_slice(&max_periods).expect("d_max");
        let d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(ONE_SERIES_LEN.checked_mul(n_combos).expect("out size"))
        }
        .expect("d_out");
        cuda.synchronize().expect("sync after prep");

        Box::new(VlmaBatchState {
            cuda,
            d_prices,
            d_ps,
            d_pss,
            d_pn,
            d_min,
            d_max,
            len: ONE_SERIES_LEN,
            first_valid,
            n_combos,
            d_out,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "vlma",
            "one_series_many_params",
            "vlma_cuda_batch_dev",
            "1m_x_200",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}
