//! CUDA wrapper for Holt-Winters Moving Average (HWMA) kernels.
//!
//! Aligned with the ALMA wrapper conventions:
//! - JIT options with DetermineTargetFromContext and O2 fallback.
//! - Policy enums + introspection for selected kernels.
//! - VRAM estimation (with ~64MB headroom) and early failure when insufficient.
//! - Chunked launches for large parameter sweeps.
//! - NON_BLOCKING stream and zero-copy DeviceArrayF32 return type.
//!
//! HWMA is a recurrence (time‑marching) indicator: each thread scans time for a
//! single parameter combo (batch) or a single series (many‑series).

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::hwma::{HwmaBatchRange, HwmaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaHwmaError {
    #[error("CUDA error: {0}")]
    Cuda(#[from] cust::error::CudaError),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("Out of memory on device (required={required} bytes, free={free} bytes, headroom={headroom} bytes)")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("Missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("Launch configuration too large (grid=({gx},{gy},{gz}), block=({bx},{by},{bz}))")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("Device buffer context/device mismatch (buf device={buf}, current={current})")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("Not implemented")]
    NotImplemented,
}

// -------- Kernel policy + introspection (subset for recurrence) --------

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
pub struct CudaHwmaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaHwmaPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaHwma {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaHwmaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaHwma {
    pub fn new(device_id: usize) -> Result<Self, CudaHwmaError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Context::new(device)?;
        let context = Arc::new(context);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/hwma_kernel.ptx"));
        // Prefer DetermineTargetFromContext + O2 for stability; fall back to simpler modes.
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => match Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                Ok(m) => m,
                Err(_) => { Module::from_ptx(ptx, &[])? }
            },
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaHwmaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(
        device_id: usize,
        policy: CudaHwmaPolicy,
    ) -> Result<Self, CudaHwmaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaHwmaPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaHwmaPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> { self._context.clone() }
    pub fn synchronize(&self) -> Result<(), CudaHwmaError> {
        self.stream.synchronize().map_err(|e| CudaHwmaError::Cuda(e))
    }

    // -------- Host↔Device helpers (pinned+async by default) --------
    #[inline]
    fn env_flag(name: &str, default: bool) -> bool {
        match env::var(name) {
            Ok(v) => !(v == "0" || v.eq_ignore_ascii_case("false")),
            Err(_) => default,
        }
    }

    fn h2d_f32(&self, src: &[f32]) -> Result<DeviceBuffer<f32>, CudaHwmaError> {
        if Self::env_flag("HWMA_PINNED", true) {
            let host = LockedBuffer::from_slice(src).map_err(|e| CudaHwmaError::Cuda(e))?;
            let dev = unsafe { DeviceBuffer::from_slice_async(&host, &self.stream) }
                .map_err(|e| CudaHwmaError::Cuda(e))?;
            Ok(dev)
        } else {
            DeviceBuffer::from_slice(src).map_err(|e| CudaHwmaError::Cuda(e))
        }
    }

    fn h2d_i32(&self, src: &[i32]) -> Result<DeviceBuffer<i32>, CudaHwmaError> {
        if Self::env_flag("HWMA_PINNED", true) {
            let host = LockedBuffer::from_slice(src).map_err(|e| CudaHwmaError::Cuda(e))?;
            let dev = unsafe { DeviceBuffer::from_slice_async(&host, &self.stream) }
                .map_err(|e| CudaHwmaError::Cuda(e))?;
            Ok(dev)
        } else {
            DeviceBuffer::from_slice(src).map_err(|e| CudaHwmaError::Cuda(e))
        }
    }

    fn h2d_f64(&self, src: &[f64]) -> Result<DeviceBuffer<f64>, CudaHwmaError> {
        if Self::env_flag("HWMA_PINNED", true) {
            let host = LockedBuffer::from_slice(src).map_err(|e| CudaHwmaError::Cuda(e))?;
            let dev = unsafe { DeviceBuffer::from_slice_async(&host, &self.stream) }
                .map_err(|e| CudaHwmaError::Cuda(e))?;
            Ok(dev)
        } else {
            DeviceBuffer::from_slice(src).map_err(|e| CudaHwmaError::Cuda(e))
        }
    }

    // VRAM helpers
    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }
    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }
    #[inline]
    fn will_fit_checked(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaHwmaError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _)) = Self::device_mem_info() {
            if required_bytes.saturating_add(headroom_bytes) <= free {
                Ok(())
            } else {
                Err(CudaHwmaError::OutOfMemory { required: required_bytes, free, headroom: headroom_bytes })
            }
        } else {
            Ok(())
        }
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] HWMA batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaHwma)).debug_batch_logged = true;
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
                let per = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] HWMA many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaHwma)).debug_many_logged = true;
                }
            }
        }
    }

    fn axis_f64_cuda_checked(t: (f64, f64, f64)) -> Result<Vec<f64>, CudaHwmaError> {
        let (start, end, step) = t;
        let eps = 1e-12;
        if step.abs() < eps || (start - end).abs() < eps {
            return Ok(vec![start]);
        }
        let mut v = Vec::new();
        if step > 0.0 {
            if start > end + eps {
                return Err(CudaHwmaError::InvalidInput(format!(
                    "invalid range: start={} end={} step={}", start, end, step
                )));
            }
            let mut x = start;
            while x <= end + eps { v.push(x); x += step; }
        } else {
            if start < end - eps {
                return Err(CudaHwmaError::InvalidInput(format!(
                    "invalid range: start={} end={} step={}", start, end, step
                )));
            }
            let mut x = start;
            while x >= end - eps { v.push(x); x += step; }
        }
        Ok(v)
    }

    fn expand_grid_cuda_checked(r: &HwmaBatchRange) -> Result<Vec<HwmaParams>, CudaHwmaError> {
        let nas = Self::axis_f64_cuda_checked(r.na)?;
        let nbs = Self::axis_f64_cuda_checked(r.nb)?;
        let ncs = Self::axis_f64_cuda_checked(r.nc)?;
        let cap = nas
            .len()
            .checked_mul(nbs.len())
            .and_then(|x| x.checked_mul(ncs.len()))
            .ok_or_else(|| CudaHwmaError::InvalidInput("expand_grid capacity overflow".into()))?;
        let mut out = Vec::with_capacity(cap);
        for &a in &nas { for &b in &nbs { for &c in &ncs { out.push(HwmaParams { na: Some(a), nb: Some(b), nc: Some(c) }); } } }
        Ok(out)
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &HwmaBatchRange,
    ) -> Result<(Vec<HwmaParams>, usize, usize), CudaHwmaError> {
        if data_f32.is_empty() {
            return Err(CudaHwmaError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaHwmaError::InvalidInput("all values are NaN".into()))?;
        let len = data_f32.len();

        // Checked expansion mirrors CPU rules; return typed error instead of fallback
        let combos = Self::expand_grid_cuda_checked(sweep)?;

        for (idx, prm) in combos.iter().enumerate() {
            let na = prm.na.unwrap_or(0.2);
            let nb = prm.nb.unwrap_or(0.1);
            let nc = prm.nc.unwrap_or(0.1);
            if !na.is_finite() || !nb.is_finite() || !nc.is_finite() {
                return Err(CudaHwmaError::InvalidInput(format!(
                    "params[{}] contain non-finite values",
                    idx
                )));
            }
            if !(na > 0.0 && na < 1.0 && nb > 0.0 && nb < 1.0 && nc > 0.0 && nc < 1.0) {
                return Err(CudaHwmaError::InvalidInput(format!(
                    "params[{}] must lie in (0,1): na={}, nb={}, nc={}",
                    idx, na, nb, nc
                )));
            }
        }

        Ok((combos, first_valid, len))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f64>,
        d_nas: &DeviceBuffer<f64>,
        d_nbs: &DeviceBuffer<f64>,
        d_ncs: &DeviceBuffer<f64>,
        first_valid: usize,
        series_len: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaHwmaError> {
        if series_len == 0 || n_combos == 0 {
            return Err(CudaHwmaError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }
        if series_len > i32::MAX as usize || n_combos > i32::MAX as usize {
            return Err(CudaHwmaError::InvalidInput(
                "series_len or n_combos exceed i32::MAX".into(),
            ));
        }
        if first_valid > i32::MAX as usize {
            return Err(CudaHwmaError::InvalidInput(
                "first_valid exceeds i32::MAX".into(),
            ));
        }

        let func = self
            .module
            .get_function("hwma_batch_f32")
            .map_err(|_| CudaHwmaError::MissingKernelSymbol { name: "hwma_batch_f32" })?;

        // Occupancy-guided block size with optional env override
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            BatchKernelPolicy::Auto => std::env::var("HWMA_BLOCK_X")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .filter(|&v| matches!(v, 64 | 128 | 256 | 512))
                .unwrap_or(128),
        };
        let grid_x = ((n_combos as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        // Optional sanity: ensure we do not exceed zero/absurd launch sizes
        if block_x == 0 || grid_x == 0 {
            return Err(CudaHwmaError::LaunchConfigTooLarge { gx: grid_x, gy: 1, gz: 1, bx: block_x, by: 1, bz: 1 });
        }

        // Introspection
        unsafe {
            (*(self as *const _ as *mut CudaHwma)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        unsafe {
            (*(self as *const _ as *mut CudaHwma)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut nas_ptr = d_nas.as_device_ptr().as_raw();
            let mut nbs_ptr = d_nbs.as_device_ptr().as_raw();
            let mut ncs_ptr = d_ncs.as_device_ptr().as_raw();
            let mut first_valid_i = first_valid as i32;
            let mut series_len_i = series_len as i32;
            let mut n_combos_i = n_combos as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut std::ffi::c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut std::ffi::c_void,
                &mut nas_ptr as *mut _ as *mut std::ffi::c_void,
                &mut nbs_ptr as *mut _ as *mut std::ffi::c_void,
                &mut ncs_ptr as *mut _ as *mut std::ffi::c_void,
                &mut first_valid_i as *mut _ as *mut std::ffi::c_void,
                &mut series_len_i as *mut _ as *mut std::ffi::c_void,
                &mut n_combos_i as *mut _ as *mut std::ffi::c_void,
                &mut out_ptr as *mut _ as *mut std::ffi::c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    pub fn hwma_batch_device(
        &self,
        d_prices: &DeviceBuffer<f64>,
        d_nas: &DeviceBuffer<f64>,
        d_nbs: &DeviceBuffer<f64>,
        d_ncs: &DeviceBuffer<f64>,
        first_valid: usize,
        series_len: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaHwmaError> {
        self.launch_batch_kernel(
            d_prices,
            d_nas,
            d_nbs,
            d_ncs,
            first_valid,
            series_len,
            n_combos,
            d_out,
        )
    }

    pub fn hwma_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &HwmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaHwmaError> {
        let (combos, first_valid, series_len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = combos.len();

        // VRAM estimate: prices + 3 param arrays + output
        let sz_f32 = std::mem::size_of::<f32>();
        let prices_bytes = series_len
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaHwmaError::InvalidInput("series_len bytes overflow".into()))?;
        let params_bytes = 3usize
            .checked_mul(n_combos)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f64>()))
            .ok_or_else(|| CudaHwmaError::InvalidInput("params bytes overflow".into()))?;
        let out_elems = n_combos
            .checked_mul(series_len)
            .ok_or_else(|| CudaHwmaError::InvalidInput("rows*cols overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaHwmaError::InvalidInput("output bytes overflow".into()))?;
        let required = prices_bytes
            .checked_add(params_bytes)
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaHwmaError::InvalidInput("total bytes overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        Self::will_fit_checked(required, headroom)?;

        // Promote prices to f64 to reduce accumulation error vs CPU f64 baseline
        let prices_f64: Vec<f64> = data_f32.iter().map(|&x| x as f64).collect();
        let d_prices = self.h2d_f64(&prices_f64)?;

        let mut nas: Vec<f64> = Vec::with_capacity(n_combos);
        let mut nbs: Vec<f64> = Vec::with_capacity(n_combos);
        let mut ncs: Vec<f64> = Vec::with_capacity(n_combos);
        for prm in &combos {
            nas.push(prm.na.unwrap_or(0.2));
            nbs.push(prm.nb.unwrap_or(0.1));
            ncs.push(prm.nc.unwrap_or(0.1));
        }
        let d_nas = if Self::env_flag("HWMA_PINNED", true) {
            let host = LockedBuffer::from_slice(&nas)?;
            unsafe { DeviceBuffer::from_slice_async(&host, &self.stream) }?
        } else {
            DeviceBuffer::from_slice(&nas)?
        };
        let d_nbs = if Self::env_flag("HWMA_PINNED", true) {
            let host = LockedBuffer::from_slice(&nbs)?;
            unsafe { DeviceBuffer::from_slice_async(&host, &self.stream) }?
        } else {
            DeviceBuffer::from_slice(&nbs)?
        };
        let d_ncs = if Self::env_flag("HWMA_PINNED", true) {
            let host = LockedBuffer::from_slice(&ncs)?;
            unsafe { DeviceBuffer::from_slice_async(&host, &self.stream) }?
        } else {
            DeviceBuffer::from_slice(&ncs)?
        };

        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }?;

        // Single launch: grid-stride kernel covers all combos
        self.launch_batch_kernel(
            &d_prices,
            &d_nas,
            &d_nbs,
            &d_ncs,
            first_valid,
            series_len,
            n_combos,
            &mut d_out,
        )?;
        // Producer stream is synchronized here; CAI v3 consumers may omit 'stream' key.
        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &HwmaParams,
    ) -> Result<(Vec<i32>, f32, f32, f32), CudaHwmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaHwmaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaHwmaError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }

        let na = params.na.unwrap_or(0.2);
        let nb = params.nb.unwrap_or(0.1);
        let nc = params.nc.unwrap_or(0.1);
        if !na.is_finite() || !nb.is_finite() || !nc.is_finite() {
            return Err(CudaHwmaError::InvalidInput(
                "parameters must be finite".into(),
            ));
        }
        if !(na > 0.0 && na < 1.0 && nb > 0.0 && nb < 1.0 && nc > 0.0 && nc < 1.0) {
            return Err(CudaHwmaError::InvalidInput(format!(
                "parameters must lie in (0,1): na={}, nb={}, nc={}",
                na, nb, nc
            )));
        }

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut found = None;
            for row in 0..rows {
                let idx = row * cols + series;
                if !data_tm_f32[idx].is_nan() {
                    found = Some(row);
                    break;
                }
            }
            let fv = found.ok_or_else(|| {
                CudaHwmaError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            if fv > i32::MAX as usize {
                return Err(CudaHwmaError::InvalidInput(
                    "first_valid exceeds i32::MAX".into(),
                ));
            }
            first_valids[series] = fv as i32;
        }

        Ok((first_valids, na as f32, nb as f32, nc as f32))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        na: f32,
        nb: f32,
        nc: f32,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaHwmaError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaHwmaError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if num_series > i32::MAX as usize || series_len > i32::MAX as usize {
            return Err(CudaHwmaError::InvalidInput(
                "num_series or series_len exceed i32::MAX".into(),
            ));
        }

        let func = self
            .module
            .get_function("hwma_multi_series_one_param_f32")
            .map_err(|_| CudaHwmaError::MissingKernelSymbol { name: "hwma_multi_series_one_param_f32" })?;

        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            ManySeriesKernelPolicy::Auto => {
                if let Ok(v) = std::env::var("HWMA_MS_BLOCK_X") {
                    v.parse::<u32>()
                        .ok()
                        .filter(|&v| matches!(v, 64 | 128 | 256 | 512))
                        .unwrap_or(128)
                } else {
                    match func.suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0)) {
                        Ok((_min_grid, suggested_block)) => suggested_block.max(128),
                        Err(_) => 256,
                    }
                }
            }
        };
        let grid_x = ((num_series as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            (*(self as *const _ as *mut CudaHwma)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut na_f = na;
            let mut nb_f = nb;
            let mut nc_f = nc;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut std::ffi::c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut std::ffi::c_void,
                &mut na_f as *mut _ as *mut std::ffi::c_void,
                &mut nb_f as *mut _ as *mut std::ffi::c_void,
                &mut nc_f as *mut _ as *mut std::ffi::c_void,
                &mut num_series_i as *mut _ as *mut std::ffi::c_void,
                &mut series_len_i as *mut _ as *mut std::ffi::c_void,
                &mut first_valids_ptr as *mut _ as *mut std::ffi::c_void,
                &mut out_ptr as *mut _ as *mut std::ffi::c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    pub fn hwma_multi_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        na: f32,
        nb: f32,
        nc: f32,
        num_series: i32,
        series_len: i32,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaHwmaError> {
        if num_series <= 0 || series_len <= 0 {
            return Err(CudaHwmaError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_prices_tm,
            na,
            nb,
            nc,
            num_series as usize,
            series_len as usize,
            d_first_valids,
            d_out_tm,
        )
    }

    pub fn hwma_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &HwmaParams,
    ) -> Result<DeviceArrayF32, CudaHwmaError> {
        let (first_valids, na, nb, nc) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        // VRAM: prices + first_valids + output
        let sz_f32 = std::mem::size_of::<f32>();
        let prices_elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaHwmaError::InvalidInput("rows*cols overflow".into()))?;
        let prices_bytes = prices_elems
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaHwmaError::InvalidInput("prices bytes overflow".into()))?;
        let first_bytes = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaHwmaError::InvalidInput("first_valids bytes overflow".into()))?;
        let out_bytes = prices_elems
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaHwmaError::InvalidInput("output bytes overflow".into()))?;
        let required = prices_bytes
            .checked_add(first_bytes)
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaHwmaError::InvalidInput("total bytes overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        Self::will_fit_checked(required, headroom)?;

        let d_prices_tm = self.h2d_f32(data_tm_f32)?;
        let d_first_valids = self.h2d_i32(&first_valids)?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            na,
            nb,
            nc,
            cols,
            rows,
            &d_first_valids,
            &mut d_out_tm,
        )?;
        // Producer stream synchronized; CAI v3 'stream' key may be omitted.
        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows,
            cols,
        })
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        hwma_benches,
        CudaHwma,
        crate::indicators::moving_averages::hwma::HwmaBatchRange,
        crate::indicators::moving_averages::hwma::HwmaParams,
        hwma_batch_dev,
        hwma_multi_series_one_param_time_major_dev,
        crate::indicators::moving_averages::hwma::HwmaBatchRange {
            na: (0.05, 0.05 + (PARAM_SWEEP as f64 - 1.0) * 0.001, 0.001),
            nb: (0.1, 0.1, 0.0),
            nc: (0.1, 0.1, 0.0)
        },
        crate::indicators::moving_averages::hwma::HwmaParams {
            na: Some(0.2),
            nb: Some(0.1),
            nc: Some(0.1)
        },
        "hwma",
        "hwma"
    );
    pub use hwma_benches::bench_profiles;
}
