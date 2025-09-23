//! CUDA wrapper for the Ehlers Error Correcting Exponential Moving Average (ECEMA).
//!
//! Mirrors the VRAM-first design used across the moving-average CUDA wrappers:
//! inputs are staged once into device buffers, kernels operate entirely in FP32
//! memory, and intermediate arithmetic is promoted to FP64 inside the kernels to
//! stay aligned with the CPU reference implementation. Two entry points are
//! provided: a batch launcher for a single series across many `(length, gain)`
//! pairs, and a time-major variant that processes many series sharing a common
//! parameter set.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::ehlers_ecema::{EhlersEcemaBatchRange, EhlersEcemaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{CopyDestination, DeviceBuffer};
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::sys as cu;
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaEhlersEcemaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaEhlersEcemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaEhlersEcemaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaEhlersEcemaError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaEhlersEcemaError {}

/// Kernel selection policy, mirroring the ALMA interface for consistency.
#[derive(Clone, Copy, Debug)]
pub enum BatchThreadsPerOutput { One, Two }

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
    Tiled { tile: u32, per_thread: BatchThreadsPerOutput },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
    Tiled2D { tx: u32, ty: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaEhlersEcemaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaEhlersEcemaPolicy {
    fn default() -> Self {
        Self { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Auto }
    }
}

/// Introspection: record last selected kernel for debugging/bench logging.
#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    PlainOneBlockPerCombo { block_x: u32 },
    ThreadPerCombo { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
    Tiled2D { tx: u32, ty: u32 },
}

pub struct CudaEhlersEcema {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaEhlersEcemaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaEhlersEcema {
    pub fn new(device_id: usize) -> Result<Self, CudaEhlersEcemaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;
        let context =
            Context::new(device).map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/ehlers_ecema_kernel.ptx"));
        let module =
            Module::from_ptx(ptx, &[]).map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaEhlersEcemaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    /// Create using an explicit kernel selection policy.
    pub fn new_with_policy(device_id: usize, policy: CudaEhlersEcemaPolicy) -> Result<Self, CudaEhlersEcemaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    /// Change current policy.
    pub fn set_policy(&mut self, policy: CudaEhlersEcemaPolicy) { self.policy = policy; }
    /// Retrieve current policy.
    pub fn policy(&self) -> &CudaEhlersEcemaPolicy { &self.policy }
    /// Last selected batch kernel (if any).
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> { self.last_batch }
    /// Last selected many-series kernel (if any).
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> { self.last_many }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        if self.debug_batch_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                eprintln!("[DEBUG] ECEMA batch selected kernel: {:?}", sel);
                unsafe { (*(self as *const _ as *mut CudaEhlersEcema)).debug_batch_logged = true; }
            }
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        if self.debug_many_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                eprintln!("[DEBUG] ECEMA many-series selected kernel: {:?}", sel);
                unsafe { (*(self as *const _ as *mut CudaEhlersEcema)).debug_many_logged = true; }
            }
        }
    }

    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }

    #[inline]
    fn env_bool(key: &str) -> Option<bool> {
        env::var(key).ok().and_then(|v| {
            let s = v.trim().to_ascii_lowercase();
            match s.as_str() {
                "1" | "true" | "yes" | "on" => Some(true),
                "0" | "false" | "no" | "off" => Some(false),
                _ => None,
            }
        })
    }

    #[inline]
    fn env_u32(key: &str) -> Option<u32> {
        env::var(key).ok().and_then(|v| v.trim().parse::<u32>().ok())
    }

    fn device_mem_info() -> Option<(usize, usize)> {
        unsafe {
            let mut free: usize = 0;
            let mut total: usize = 0;
            let res = cu::cuMemGetInfo_v2(&mut free as *mut usize, &mut total as *mut usize);
            if res == cu::CUresult::CUDA_SUCCESS {
                Some((free, total))
            } else {
                None
            }
        }
    }

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

    fn axis_usize(axis: (usize, usize, usize)) -> Vec<usize> {
        let (start, end, step) = axis;
        if step == 0 || start == end {
            vec![start]
        } else if start <= end {
            (start..=end).step_by(step).collect()
        } else {
            vec![start]
        }
    }

    fn expand_range(
        range: &EhlersEcemaBatchRange,
        pine_mode: bool,
        confirmed: bool,
    ) -> Vec<EhlersEcemaParams> {
        let lengths = Self::axis_usize(range.length);
        let gain_limits = Self::axis_usize(range.gain_limit);
        let mut combos = Vec::with_capacity(lengths.len() * gain_limits.len());
        for &len in &lengths {
            for &gain in &gain_limits {
                combos.push(EhlersEcemaParams {
                    length: Some(len),
                    gain_limit: Some(gain),
                    pine_compatible: Some(pine_mode),
                    confirmed_only: Some(confirmed),
                });
            }
        }
        combos
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &EhlersEcemaBatchRange,
        pine_mode: bool,
        confirmed: bool,
    ) -> Result<(Vec<EhlersEcemaParams>, usize, usize), CudaEhlersEcemaError> {
        if data_f32.is_empty() {
            return Err(CudaEhlersEcemaError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaEhlersEcemaError::InvalidInput("all values are NaN".into()))?;

        let combos = Self::expand_range(sweep, pine_mode, confirmed);
        if combos.is_empty() {
            return Err(CudaEhlersEcemaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let series_len = data_f32.len();
        for prm in &combos {
            let length = prm.length.unwrap_or(0);
            let gain = prm.gain_limit.unwrap_or(0);
            if length == 0 {
                return Err(CudaEhlersEcemaError::InvalidInput(
                    "length must be >= 1".into(),
                ));
            }
            if gain == 0 {
                return Err(CudaEhlersEcemaError::InvalidInput(
                    "gain_limit must be >= 1".into(),
                ));
            }
            if length > series_len {
                return Err(CudaEhlersEcemaError::InvalidInput(format!(
                    "length {} exceeds data length {}",
                    length, series_len
                )));
            }
            let valid = series_len - first_valid;
            if !pine_mode && valid < length {
                return Err(CudaEhlersEcemaError::InvalidInput(format!(
                    "not enough valid data: need >= {}, valid = {}",
                    length, valid
                )));
            }
        }

        Ok((combos, first_valid, series_len))
    }

    fn launch_batch_plain(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_lengths: &DeviceBuffer<i32>,
        d_gain_limits: &DeviceBuffer<i32>,
        d_pine_flags: &DeviceBuffer<u8>,
        d_confirmed_flags: &DeviceBuffer<u8>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhlersEcemaError> {
        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x.max(1),
            _ => 1,
        };
        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        let func = self
            .module
            .get_function("ehlers_ecema_batch_f32")
            .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut lengths_ptr = d_lengths.as_device_ptr().as_raw();
            let mut gains_ptr = d_gain_limits.as_device_ptr().as_raw();
            let mut pine_ptr = d_pine_flags.as_device_ptr().as_raw();
            let mut confirmed_ptr = d_confirmed_flags.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut lengths_ptr as *mut _ as *mut c_void,
                &mut gains_ptr as *mut _ as *mut c_void,
                &mut pine_ptr as *mut _ as *mut c_void,
                &mut confirmed_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;
        }
        unsafe {
            let this = self as *const _ as *mut CudaEhlersEcema;
            (*this).last_batch = Some(BatchKernelSelected::PlainOneBlockPerCombo { block_x });
        }
        self.maybe_log_batch_debug();
        Ok(())
    }

    fn launch_batch_thread_per_combo(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_lengths: &DeviceBuffer<i32>,
        d_gain_limits: &DeviceBuffer<i32>,
        d_pine_flags: &DeviceBuffer<u8>,
        d_confirmed_flags: &DeviceBuffer<u8>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhlersEcemaError> {
        // Batch block size heuristic with env override.
        let block_x_env = Self::env_u32("ECEMA_BLOCK_X");
        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Tiled { tile, .. } => tile.max(1),
            BatchKernelPolicy::Plain { block_x } => block_x.max(1),
            BatchKernelPolicy::Auto => block_x_env.unwrap_or(128).max(1),
        };
        let grid_x = ((n_combos as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        let func = self
            .module
            .get_function("ehlers_ecema_batch_thread_per_combo_f32")
            .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut lengths_ptr = d_lengths.as_device_ptr().as_raw();
            let mut gains_ptr = d_gain_limits.as_device_ptr().as_raw();
            let mut pine_ptr = d_pine_flags.as_device_ptr().as_raw();
            let mut confirmed_ptr = d_confirmed_flags.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut lengths_ptr as *mut _ as *mut c_void,
                &mut gains_ptr as *mut _ as *mut c_void,
                &mut pine_ptr as *mut _ as *mut c_void,
                &mut confirmed_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;
        }
        unsafe {
            let this = self as *const _ as *mut CudaEhlersEcema;
            (*this).last_batch = Some(BatchKernelSelected::ThreadPerCombo { block_x });
        }
        self.maybe_log_batch_debug();
        Ok(())
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        combos: &[EhlersEcemaParams],
        first_valid: usize,
        series_len: usize,
    ) -> Result<DeviceArrayF32, CudaEhlersEcemaError> {
        let n_combos = combos.len();
        let prices_bytes = series_len * std::mem::size_of::<f32>();
        let lengths_bytes = n_combos * std::mem::size_of::<i32>();
        let gains_bytes = n_combos * std::mem::size_of::<i32>();
        let flags_bytes = n_combos * std::mem::size_of::<u8>() * 2;
        let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + lengths_bytes + gains_bytes + flags_bytes + out_bytes;
        let headroom = 32 * 1024 * 1024; // 32MB cushion
        if !Self::will_fit(required, headroom) {
            return Err(CudaEhlersEcemaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices = DeviceBuffer::from_slice(data_f32)
            .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;
        let lengths_i32: Vec<i32> = combos.iter().map(|p| p.length.unwrap() as i32).collect();
        let gain_i32: Vec<i32> = combos
            .iter()
            .map(|p| p.gain_limit.unwrap() as i32)
            .collect();
        let pine_flags: Vec<u8> = combos
            .iter()
            .map(|p| {
                if p.pine_compatible.unwrap_or(false) {
                    1
                } else {
                    0
                }
            })
            .collect();
        let confirmed_flags: Vec<u8> = combos
            .iter()
            .map(|p| {
                if p.confirmed_only.unwrap_or(false) {
                    1
                } else {
                    0
                }
            })
            .collect();

        let d_lengths = DeviceBuffer::from_slice(&lengths_i32)
            .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;
        let d_gains = DeviceBuffer::from_slice(&gain_i32)
            .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;
        let d_pine = DeviceBuffer::from_slice(&pine_flags)
            .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;
        let d_confirmed = DeviceBuffer::from_slice(&confirmed_flags)
            .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;

        // Select kernel according to policy. Default to thread-per-combo when available.
        let have_thread_per_combo = self.module.get_function("ehlers_ecema_batch_thread_per_combo_f32").is_ok();
        let force_plain = Self::env_bool("ECEMA_FORCE_PLAIN").unwrap_or(false);
        let force_tiled = Self::env_bool("ECEMA_FORCE_TILED").unwrap_or(false);
        let use_thread_per_combo = match self.policy.batch {
            BatchKernelPolicy::Auto => {
                if force_plain { false } else if force_tiled { true } else { have_thread_per_combo }
            }
            BatchKernelPolicy::Plain { .. } => false,
            BatchKernelPolicy::Tiled { .. } => true,
        };

        if use_thread_per_combo {
            self.launch_batch_thread_per_combo(
                &d_prices,
                &d_lengths,
                &d_gains,
                &d_pine,
                &d_confirmed,
                series_len,
                n_combos,
                first_valid,
                &mut d_out,
            )?
        } else {
            self.launch_batch_plain(
                &d_prices,
                &d_lengths,
                &d_gains,
                &d_pine,
                &d_confirmed,
                series_len,
                n_combos,
                first_valid,
                &mut d_out,
            )?
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    pub fn ehlers_ecema_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &EhlersEcemaBatchRange,
        params: &EhlersEcemaParams,
    ) -> Result<DeviceArrayF32, CudaEhlersEcemaError> {
        let pine_mode = params.pine_compatible.unwrap_or(false);
        let confirmed = params.confirmed_only.unwrap_or(false);
        let (combos, first_valid, series_len) =
            Self::prepare_batch_inputs(data_f32, sweep, pine_mode, confirmed)?;
        self.run_batch_kernel(data_f32, &combos, first_valid, series_len)
    }

    pub fn ehlers_ecema_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &EhlersEcemaBatchRange,
        params: &EhlersEcemaParams,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<EhlersEcemaParams>), CudaEhlersEcemaError> {
        let pine_mode = params.pine_compatible.unwrap_or(false);
        let confirmed = params.confirmed_only.unwrap_or(false);
        let (combos, first_valid, series_len) =
            Self::prepare_batch_inputs(data_f32, sweep, pine_mode, confirmed)?;
        let expected = combos.len() * series_len;
        if out.len() != expected {
            return Err(CudaEhlersEcemaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                expected
            )));
        }
        let arr = self.run_batch_kernel(data_f32, &combos, first_valid, series_len)?;
        arr.buf
            .copy_to(out)
            .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;
        Ok((arr.rows, arr.cols, combos))
    }

    pub fn ehlers_ecema_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_lengths: &DeviceBuffer<i32>,
        d_gain_limits: &DeviceBuffer<i32>,
        d_pine_flags: &DeviceBuffer<u8>,
        d_confirmed_flags: &DeviceBuffer<u8>,
        series_len: i32,
        n_combos: i32,
        first_valid: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhlersEcemaError> {
        if series_len <= 0 || n_combos <= 0 {
            return Err(CudaEhlersEcemaError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }
        let have_thread_per_combo = self.module.get_function("ehlers_ecema_batch_thread_per_combo_f32").is_ok();
        let use_thread_per_combo = match self.policy.batch {
            BatchKernelPolicy::Auto => have_thread_per_combo,
            BatchKernelPolicy::Plain { .. } => false,
            BatchKernelPolicy::Tiled { .. } => true,
        };
        if use_thread_per_combo {
            self.launch_batch_thread_per_combo(
                d_prices,
                d_lengths,
                d_gain_limits,
                d_pine_flags,
                d_confirmed_flags,
                series_len as usize,
                n_combos as usize,
                first_valid.max(0) as usize,
                d_out,
            )
        } else {
            self.launch_batch_plain(
                d_prices,
                d_lengths,
                d_gain_limits,
                d_pine_flags,
                d_confirmed_flags,
                series_len as usize,
                n_combos as usize,
                first_valid.max(0) as usize,
                d_out,
            )
        }
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &EhlersEcemaParams,
    ) -> Result<(Vec<i32>, usize, usize, bool, bool), CudaEhlersEcemaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaEhlersEcemaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaEhlersEcemaError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }

        let length = params.length.unwrap_or(20);
        let gain_limit = params.gain_limit.unwrap_or(50);
        if length == 0 {
            return Err(CudaEhlersEcemaError::InvalidInput(
                "length must be >= 1".into(),
            ));
        }
        if gain_limit == 0 {
            return Err(CudaEhlersEcemaError::InvalidInput(
                "gain_limit must be >= 1".into(),
            ));
        }

        let pine_mode = params.pine_compatible.unwrap_or(false);
        let confirmed = params.confirmed_only.unwrap_or(false);

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut found = None;
            for t in 0..rows {
                let idx = t * cols + series;
                if !data_tm_f32[idx].is_nan() {
                    found = Some(t as i32);
                    break;
                }
            }
            let fv = found.ok_or_else(|| {
                CudaEhlersEcemaError::InvalidInput(format!("series {} all NaN", series))
            })?;
            let valid = rows - fv as usize;
            if !pine_mode && valid < length {
                return Err(CudaEhlersEcemaError::InvalidInput(format!(
                    "series {} does not have enough valid data: need >= {}, valid = {}",
                    series, length, valid
                )));
            }
            first_valids[series] = fv;
        }

        Ok((first_valids, length, gain_limit, pine_mode, confirmed))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        length: usize,
        gain_limit: usize,
        pine_mode: bool,
        confirmed: bool,
        d_first_valids: &DeviceBuffer<i32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhlersEcemaError> {
        match self.policy.many_series {
            ManySeriesKernelPolicy::Auto | ManySeriesKernelPolicy::OneD { .. } => {
                let force_2d = Self::env_bool("ECEMA_FORCE_2D").unwrap_or(false);
                let force_1d = Self::env_bool("ECEMA_FORCE_1D").unwrap_or(false);
                let series_2d_min = Self::env_u32("ECEMA_2D_MIN_SERIES").unwrap_or(2048) as usize;

                // Auto switch to 2D for large number of series, unless forced.
                if matches!(self.policy.many_series, ManySeriesKernelPolicy::Auto)
                    && !force_1d
                    && (force_2d || cols >= series_2d_min)
                {
                    // Choose tx/ty with env overrides, defaults tuned for occupancy.
                    let tx = Self::env_u32("ECEMA_2D_TX").unwrap_or(128).max(1);
                    let ty = Self::env_u32("ECEMA_2D_TY").unwrap_or(2).max(1);
                    let series_per_block = (tx * ty) as usize;
                    let total_blocks = ((cols + series_per_block - 1) / series_per_block) as u32;
                    let grid_x = ((cols as u32) + tx - 1) / tx;
                    let grid_y = ((total_blocks + grid_x - 1) / grid_x).max(1);
                    let grid: GridSize = (grid_x, grid_y, 1).into();
                    let block: BlockSize = (tx, ty, 1).into();
                    let func_name = if self.module.get_function("ehlers_ecema_many_series_one_param_2d_f32").is_ok() {
                        "ehlers_ecema_many_series_one_param_2d_f32"
                    } else if self.module.get_function("ehlers_ecema_many_series_one_param_1d_f32").is_ok() {
                        "ehlers_ecema_many_series_one_param_1d_f32"
                    } else {
                        "ehlers_ecema_many_series_one_param_time_major_f32"
                    };
                    let func = self.module.get_function(func_name)
                        .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;
                    unsafe {
                        let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                        let mut cols_i = cols as i32;
                        let mut rows_i = rows as i32;
                        let mut length_i = length as i32;
                        let mut gain_limit_i = gain_limit as i32;
                        let mut pine_flag = if pine_mode { 1u8 } else { 0u8 };
                        let mut confirmed_flag = if confirmed { 1u8 } else { 0u8 };
                        let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
                        let mut out_ptr = d_out.as_device_ptr().as_raw();
                        let args: &mut [*mut c_void] = &mut [
                            &mut prices_ptr as *mut _ as *mut c_void,
                            &mut cols_i as *mut _ as *mut c_void,
                            &mut rows_i as *mut _ as *mut c_void,
                            &mut length_i as *mut _ as *mut c_void,
                            &mut gain_limit_i as *mut _ as *mut c_void,
                            &mut pine_flag as *mut _ as *mut c_void,
                            &mut confirmed_flag as *mut _ as *mut c_void,
                            &mut first_ptr as *mut _ as *mut c_void,
                            &mut out_ptr as *mut _ as *mut c_void,
                        ];
                        self.stream.launch(&func, grid, block, 0, args)
                            .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;
                    }
                    unsafe {
                        let this = self as *const _ as *mut CudaEhlersEcema;
                        (*this).last_many = Some(ManySeriesKernelSelected::Tiled2D { tx, ty });
                    }
                    self.maybe_log_many_debug();
                    return Ok(());
                }

                let block_x = match self.policy.many_series {
                    ManySeriesKernelPolicy::OneD { block_x } => block_x.max(1),
                    _ => Self::env_u32("ECEMA_ONE_D_BLOCK_X").unwrap_or(128).max(1),
                };
                let grid_x = ((cols as u32) + block_x - 1) / block_x;
                let grid: GridSize = (grid_x, 1, 1).into();
                let block: BlockSize = (block_x, 1, 1).into();
                let func_name = if self.module.get_function("ehlers_ecema_many_series_one_param_1d_f32").is_ok() {
                    "ehlers_ecema_many_series_one_param_1d_f32"
                } else {
                    "ehlers_ecema_many_series_one_param_time_major_f32"
                };
                let func = self.module.get_function(func_name)
                    .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;
                unsafe {
                    let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                    let mut cols_i = cols as i32;
                    let mut rows_i = rows as i32;
                    let mut length_i = length as i32;
                    let mut gain_limit_i = gain_limit as i32;
                    let mut pine_flag = if pine_mode { 1u8 } else { 0u8 };
                    let mut confirmed_flag = if confirmed { 1u8 } else { 0u8 };
                    let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
                    let mut out_ptr = d_out.as_device_ptr().as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut prices_ptr as *mut _ as *mut c_void,
                        &mut cols_i as *mut _ as *mut c_void,
                        &mut rows_i as *mut _ as *mut c_void,
                        &mut length_i as *mut _ as *mut c_void,
                        &mut gain_limit_i as *mut _ as *mut c_void,
                        &mut pine_flag as *mut _ as *mut c_void,
                        &mut confirmed_flag as *mut _ as *mut c_void,
                        &mut first_ptr as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];
                    self.stream.launch(&func, grid, block, 0, args)
                        .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;
                }
                unsafe {
                    let this = self as *const _ as *mut CudaEhlersEcema;
                    (*this).last_many = Some(ManySeriesKernelSelected::OneD { block_x });
                }
                self.maybe_log_many_debug();
                Ok(())
            }
            ManySeriesKernelPolicy::Tiled2D { tx, ty } => {
                let tx = tx.max(1);
                let ty = ty.max(1);
                let series_per_block = (tx * ty) as usize;
                let total_blocks = ((cols + series_per_block - 1) / series_per_block) as u32;
                // Split across grid.x with width tx and grid.y slices covering the remainder
                let grid_x = ((cols as u32) + tx - 1) / tx; // number of tiles along x
                let grid_y = ((total_blocks + grid_x - 1) / grid_x).max(1);
                let grid: GridSize = (grid_x, grid_y, 1).into();
                let block: BlockSize = (tx, ty, 1).into();
                let func_name = if self.module.get_function("ehlers_ecema_many_series_one_param_2d_f32").is_ok() {
                    "ehlers_ecema_many_series_one_param_2d_f32"
                } else if self.module.get_function("ehlers_ecema_many_series_one_param_1d_f32").is_ok() {
                    "ehlers_ecema_many_series_one_param_1d_f32"
                } else {
                    "ehlers_ecema_many_series_one_param_time_major_f32"
                };
                let func = self.module.get_function(func_name)
                    .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;
                unsafe {
                    let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                    let mut cols_i = cols as i32;
                    let mut rows_i = rows as i32;
                    let mut length_i = length as i32;
                    let mut gain_limit_i = gain_limit as i32;
                    let mut pine_flag = if pine_mode { 1u8 } else { 0u8 };
                    let mut confirmed_flag = if confirmed { 1u8 } else { 0u8 };
                    let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
                    let mut out_ptr = d_out.as_device_ptr().as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut prices_ptr as *mut _ as *mut c_void,
                        &mut cols_i as *mut _ as *mut c_void,
                        &mut rows_i as *mut _ as *mut c_void,
                        &mut length_i as *mut _ as *mut c_void,
                        &mut gain_limit_i as *mut _ as *mut c_void,
                        &mut pine_flag as *mut _ as *mut c_void,
                        &mut confirmed_flag as *mut _ as *mut c_void,
                        &mut first_ptr as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];
                    self.stream.launch(&func, grid, block, 0, args)
                        .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;
                }
                unsafe {
                    let this = self as *const _ as *mut CudaEhlersEcema;
                    (*this).last_many = Some(ManySeriesKernelSelected::Tiled2D { tx, ty });
                }
                self.maybe_log_many_debug();
                Ok(())
            }
        }
    }

    fn run_many_series_kernel(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        length: usize,
        gain_limit: usize,
        pine_mode: bool,
        confirmed: bool,
        first_valids: &[i32],
    ) -> Result<DeviceArrayF32, CudaEhlersEcemaError> {
        let prices_bytes = cols * rows * std::mem::size_of::<f32>();
        let first_valid_bytes = cols * std::mem::size_of::<i32>();
        let out_bytes = cols * rows * std::mem::size_of::<f32>();
        let required = prices_bytes + first_valid_bytes + out_bytes;
        let headroom = 16 * 1024 * 1024; // 16MB cushion
        if !Self::will_fit(required, headroom) {
            return Err(CudaEhlersEcemaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(first_valids)
            .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices,
            cols,
            rows,
            length,
            gain_limit,
            pine_mode,
            confirmed,
            &d_first_valids,
            &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn ehlers_ecema_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &EhlersEcemaParams,
    ) -> Result<DeviceArrayF32, CudaEhlersEcemaError> {
        let (first_valids, length, gain_limit, pine_mode, confirmed) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        self.run_many_series_kernel(
            data_tm_f32,
            cols,
            rows,
            length,
            gain_limit,
            pine_mode,
            confirmed,
            &first_valids,
        )
    }

    pub fn ehlers_ecema_many_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &EhlersEcemaParams,
        out: &mut [f32],
    ) -> Result<(), CudaEhlersEcemaError> {
        if out.len() != cols * rows {
            return Err(CudaEhlersEcemaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                cols * rows
            )));
        }
        let arr = self.ehlers_ecema_many_series_one_param_time_major_dev(
            data_tm_f32,
            cols,
            rows,
            params,
        )?;
        arr.buf
            .copy_to(out)
            .map_err(|e| CudaEhlersEcemaError::Cuda(e.to_string()))
    }

    pub fn ehlers_ecema_many_series_one_param_time_major_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        cols: i32,
        rows: i32,
        length: i32,
        gain_limit: i32,
        pine_flag: u8,
        confirmed_flag: u8,
        d_first_valids: &DeviceBuffer<i32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhlersEcemaError> {
        if cols <= 0 || rows <= 0 || length <= 0 || gain_limit <= 0 {
            return Err(CudaEhlersEcemaError::InvalidInput(
                "cols, rows, length and gain_limit must be positive".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_prices,
            cols as usize,
            rows as usize,
            length as usize,
            gain_limit as usize,
            pine_flag != 0,
            confirmed_flag != 0,
            d_first_valids,
            d_out,
        )
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;
    const MANY_SERIES_COLS: usize = 250;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = elems * std::mem::size_of::<f32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    fn default_params() -> EhlersEcemaParams {
        EhlersEcemaParams {
            length: Some(20),
            gain_limit: Some(50),
            pine_compatible: Some(false),
            confirmed_only: Some(false),
        }
    }

    struct EcemaBatchState {
        cuda: CudaEhlersEcema,
        price: Vec<f32>,
        sweep: EhlersEcemaBatchRange,
        params: EhlersEcemaParams,
    }
    impl CudaBenchState for EcemaBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .ehlers_ecema_batch_dev(&self.price, &self.sweep, &self.params)
                .expect("ecema batch launch");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaEhlersEcema::new(0).expect("cuda ecema");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = EhlersEcemaBatchRange {
            length: (10, 10 + PARAM_SWEEP - 1, 1),
            gain_limit: (50, 50, 0),
        };
        Box::new(EcemaBatchState {
            cuda,
            price,
            sweep,
            params: default_params(),
        })
    }

    struct EcemaManyState {
        cuda: CudaEhlersEcema,
        data_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        params: EhlersEcemaParams,
    }
    impl CudaBenchState for EcemaManyState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .ehlers_ecema_many_series_one_param_time_major_dev(
                    &self.data_tm,
                    self.cols,
                    self.rows,
                    &self.params,
                )
                .expect("ecema many-series launch");
        }
    }
    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaEhlersEcema::new(0).expect("cuda ecema");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        Box::new(EcemaManyState {
            cuda,
            data_tm,
            cols,
            rows,
            params: default_params(),
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "ehlers_ecema",
                "one_series_many_params",
                "ecema_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "ehlers_ecema",
                "many_series_one_param",
                "ecema_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
