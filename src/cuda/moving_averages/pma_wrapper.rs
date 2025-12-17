//! CUDA scaffolding for the Predictive Moving Average (PMA).
//!
//! Mirrors ALMA-style GPU surface: one-series × many-params (synthetic combos)
//! and many-series × one-param (time-major). This PMA variant matches
//! src/indicators/pma.rs semantics (no 1-bar lag; warmups at first+6 and first+9).

#![cfg(feature = "cuda")]

use super::DeviceArrayF32;
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use thiserror::Error;

use crate::indicators::pma::PmaBatchRange;

// Kernel policy (parity with ALMA/CWMA wrappers)
#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
    Tiled { tile: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
    Tiled2D { tx: u32, ty: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaPmaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaPmaPolicy {
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
    Tiled { tile: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
    Tiled2D { tx: u32, ty: u32 },
}

#[derive(Error, Debug)]
pub enum CudaPmaError {
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
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

/// VRAM-backed pair of PMA outputs (predict + trigger)
pub struct DevicePmaPair {
    pub predict: DeviceArrayF32,
    pub trigger: DeviceArrayF32,
}

impl DevicePmaPair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.predict.rows
    }
    #[inline]
    pub fn cols(&self) -> usize {
        self.predict.cols
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.predict.len()
    }
}

pub struct CudaPma {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaPmaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

// Helper container for batch preparation
struct BatchInputs {
    combos: usize,
    first_valid: usize,
    series_len: usize,
}

impl CudaPma {
    pub fn new(device_id: usize) -> Result<Self, CudaPmaError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Context::new(device)?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/pma_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaPmaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(device_id: usize, policy: CudaPmaPolicy) -> Result<Self, CudaPmaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaPmaPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaPmaPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
    pub fn synchronize(&self) -> Result<(), CudaPmaError> {
        self.stream.synchronize()?;
        Ok(())
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
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaPmaError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _t)) = Self::device_mem_info() {
            let required = required_bytes.saturating_add(headroom_bytes);
            if required <= free {
                Ok(())
            } else {
                Err(CudaPmaError::OutOfMemory {
                    required,
                    free,
                    headroom: headroom_bytes,
                })
            }
        } else {
            Ok(())
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
    ) -> Result<(), CudaPmaError> {
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
            return Err(CudaPmaError::LaunchConfigTooLarge {
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

    fn maybe_log_batch_debug(&self) {
        static GLOBAL: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_scen =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scen || !GLOBAL.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] PMA batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaPma)).debug_batch_logged = true;
                }
            }
        }
    }
    fn maybe_log_many_debug(&self) {
        static GLOBAL: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per_scen =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scen || !GLOBAL.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] PMA many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaPma)).debug_many_logged = true;
                }
            }
        }
    }

    // ---------------- Batch (one series × many params) ----------------
    fn prepare_batch_inputs(
        prices: &[f32],
        _sweep: &PmaBatchRange,
    ) -> Result<BatchInputs, CudaPmaError> {
        if prices.is_empty() {
            return Err(CudaPmaError::InvalidInput("empty price series".into()));
        }
        let first_valid = prices
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaPmaError::InvalidInput("all values are NaN".into()))?;
        const MIN_REQUIRED: usize = 7; // 7 samples for first predict
        if prices.len() - first_valid < MIN_REQUIRED {
            return Err(CudaPmaError::InvalidInput(format!(
                "not enough valid data (needed >= {MIN_REQUIRED}, valid = {})",
                prices.len() - first_valid
            )));
        }
        // PMA has no tunable params; keep one synthetic combo for API parity.
        Ok(BatchInputs {
            combos: 1,
            first_valid,
            series_len: prices.len(),
        })
    }

    fn run_batch_kernel(
        &self,
        prices: &[f32],
        inputs: &BatchInputs,
    ) -> Result<DevicePmaPair, CudaPmaError> {
        let elem = core::mem::size_of::<f32>();
        let prices_bytes = inputs
            .series_len
            .checked_mul(elem)
            .ok_or_else(|| CudaPmaError::InvalidInput("series_len * sizeof(f32) overflow".into()))?;
        let out_elems = inputs
            .combos
            .checked_mul(inputs.series_len)
            .ok_or_else(|| {
                CudaPmaError::InvalidInput("combos * series_len overflow".into())
            })?;
        let out_bytes = out_elems
            .checked_mul(elem)
            .ok_or_else(|| CudaPmaError::InvalidInput("out_elems * sizeof(f32) overflow".into()))?;
        let two_out = out_bytes
            .checked_mul(2)
            .ok_or_else(|| CudaPmaError::InvalidInput("2 * out_bytes overflow".into()))?;
        let required = prices_bytes
            .checked_add(two_out)
            .ok_or_else(|| {
                CudaPmaError::InvalidInput("prices_bytes + 2*out_bytes overflow".into())
            })?;
        let headroom = 64 * 1024 * 1024; // 64MB safety
        Self::will_fit(required, headroom)?;

        let mut d_prices: DeviceBuffer<f32> = DeviceBuffer::from_slice(prices)?;
        let mut d_predict: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(inputs.combos * inputs.series_len) }?;
        let mut d_trigger: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(inputs.combos * inputs.series_len) }?;

        self.launch_batch_kernel_select(
            &d_prices,
            inputs.series_len,
            inputs.combos,
            inputs.first_valid,
            &mut d_predict,
            &mut d_trigger,
        )?;

        Ok(DevicePmaPair {
            predict: DeviceArrayF32 {
                buf: d_predict,
                rows: inputs.combos,
                cols: inputs.series_len,
            },
            trigger: DeviceArrayF32 {
                buf: d_trigger,
                rows: inputs.combos,
                cols: inputs.series_len,
            },
        })
    }

    fn launch_batch_kernel_select(
        &self,
        d_prices: &DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_predict: &mut DeviceBuffer<f32>,
        d_trigger: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaPmaError> {
        let (fname, block, grid, sel, gx, gy, gz, bx, by, bz) = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => {
                let bx = block_x.max(1);
                let gx = n_combos as u32;
                (
                    "pma_batch_f32",
                    BlockSize::xy(bx, 1),
                    GridSize::xyz(gx, 1, 1),
                    Some(BatchKernelSelected::Plain { block_x }),
                    gx,
                    1,
                    1,
                    bx,
                    1,
                    1,
                )
            }
            BatchKernelPolicy::Tiled { tile } => {
                let sym = if tile >= 256 {
                    "pma_batch_tiled_f32_tile256"
                } else {
                    "pma_batch_tiled_f32_tile128"
                };
                let name = if self.module.get_function(sym).is_ok() {
                    sym
                } else {
                    "pma_batch_f32"
                };
                let gx = n_combos as u32;
                (
                    name,
                    BlockSize::xy(1, 1),
                    GridSize::xyz(gx, 1, 1),
                    Some(BatchKernelSelected::Tiled { tile }),
                    gx,
                    1,
                    1,
                    1,
                    1,
                    1,
                )
            }
            BatchKernelPolicy::Auto => {
                // Default to simple 1D launch; computation is strictly sequential per combo.
                let gx = n_combos as u32;
                (
                    "pma_batch_f32",
                    BlockSize::xy(1, 1),
                    GridSize::xyz(gx, 1, 1),
                    Some(BatchKernelSelected::Plain { block_x: 1 }),
                    gx,
                    1,
                    1,
                    1,
                    1,
                    1,
                )
            }
        };

        self.validate_launch(gx, gy, gz, bx, by, bz)?;

        if let Some(s) = sel {
            unsafe {
                (*(self as *const _ as *mut CudaPma)).last_batch = Some(s);
            }
        }
        self.maybe_log_batch_debug();

        let func = self
            .module
            .get_function(fname)
            .map_err(|_| CudaPmaError::MissingKernelSymbol { name: fname })?;
        let mut args: [*mut c_void; 6] = [
            &mut d_prices.as_device_ptr().as_raw() as *mut _ as *mut c_void,
            &mut (series_len as i32) as *mut _ as *mut c_void,
            &mut (n_combos as i32) as *mut _ as *mut c_void,
            &mut (first_valid as i32) as *mut _ as *mut c_void,
            &mut d_predict.as_device_ptr().as_raw() as *mut _ as *mut c_void,
            &mut d_trigger.as_device_ptr().as_raw() as *mut _ as *mut c_void,
        ];
        unsafe { self.stream.launch(&func, grid, block, 0, &mut args) }?;
        Ok(())
    }

    pub fn pma_batch_dev(
        &self,
        prices: &[f32],
        sweep: &PmaBatchRange,
    ) -> Result<DevicePmaPair, CudaPmaError> {
        let inputs = Self::prepare_batch_inputs(prices, sweep)?;
        self.run_batch_kernel(prices, &inputs)
    }

    pub fn pma_batch_into_host_f32(
        &self,
        prices: &[f32],
        sweep: &PmaBatchRange,
        out_predict: &mut [f32],
        out_trigger: &mut [f32],
    ) -> Result<(usize, usize), CudaPmaError> {
        let inputs = Self::prepare_batch_inputs(prices, sweep)?;
        let expected = inputs
            .series_len
            .checked_mul(inputs.combos)
            .ok_or_else(|| {
                CudaPmaError::InvalidInput(format!(
                    "series_len * combos overflow: series_len={} combos={}",
                    inputs.series_len, inputs.combos
                ))
            })?;
        if out_predict.len() != expected || out_trigger.len() != expected {
            return Err(CudaPmaError::InvalidInput(format!(
                "output slice wrong length: got p={}, t={}, expected={}",
                out_predict.len(),
                out_trigger.len(),
                expected
            )));
        }
        let pair = self.run_batch_kernel(prices, &inputs)?;
        pair.predict.buf.copy_to(out_predict)?;
        pair.trigger.buf.copy_to(out_trigger)?;
        Ok((pair.rows(), pair.cols()))
    }

    // ---------------- Many series × one param (time-major) ----------------
    fn prepare_many_series_inputs(
        prices_tm: &[f32],
        cols: usize,
        rows: usize,
    ) -> Result<Vec<i32>, CudaPmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaPmaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| {
                CudaPmaError::InvalidInput(format!(
                    "cols * rows overflow: cols={} rows={}",
                    cols, rows
                ))
            })?;
        if prices_tm.len() != expected {
            return Err(CudaPmaError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                prices_tm.len(),
                expected
            )));
        }
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for r in 0..rows {
                let v = prices_tm[r * cols + s];
                if !v.is_nan() {
                    fv = Some(r);
                    break;
                }
            }
            let idx = fv.ok_or_else(|| {
                CudaPmaError::InvalidInput(format!("series {} is entirely NaN", s))
            })?;
            if rows - idx < 7 {
                return Err(CudaPmaError::InvalidInput(format!(
                    "series {} lacks warmup samples (valid = {})",
                    s,
                    rows - idx
                )));
            }
            first_valids[s] = idx as i32;
        }
        Ok(first_valids)
    }

    fn launch_many_series_kernel_select(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_predict_tm: &mut DeviceBuffer<f32>,
        d_trigger_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaPmaError> {
        let (fname, block, grid, sel, gx, gy, gz, bx, by, bz) = match self.policy.many_series {
            ManySeriesKernelPolicy::Tiled2D { tx, ty } => {
                let (fname, gx) = {
                    let base = if (tx, ty) == (1, 4) {
                        "pma_ms1p_tiled_f32_tx1_ty4"
                    } else if (tx, ty) == (1, 2) {
                        "pma_ms1p_tiled_f32_tx1_ty2"
                    } else {
                        "pma_many_series_one_param_f32"
                    };
                    let gx_val = if base == "pma_many_series_one_param_f32" {
                        cols as u32
                    } else {
                        ((cols as u32) + ty - 1) / ty
                    };
                    (base, gx_val)
                };
                let bx = tx.max(1);
                let by = ty.max(1);
                (
                    fname,
                    BlockSize::xyz(bx, by, 1),
                    GridSize::xyz(gx, 1, 1),
                    Some(ManySeriesKernelSelected::Tiled2D { tx, ty }),
                    gx,
                    1,
                    1,
                    bx,
                    by,
                    1,
                )
            }
            ManySeriesKernelPolicy::OneD { block_x } => {
                let bx = block_x.max(1);
                let gx = cols as u32;
                (
                    "pma_many_series_one_param_f32",
                    BlockSize::xy(bx, 1),
                    GridSize::xyz(gx, 1, 1),
                    Some(ManySeriesKernelSelected::OneD { block_x }),
                    gx,
                    1,
                    1,
                    bx,
                    1,
                    1,
                )
            }
            ManySeriesKernelPolicy::Auto => {
                if cols >= 16
                    && self
                        .module
                        .get_function("pma_ms1p_tiled_f32_tx1_ty4")
                        .is_ok()
                {
                    let gx = ((cols as u32) + 4 - 1) / 4;
                    (
                        "pma_ms1p_tiled_f32_tx1_ty4",
                        BlockSize::xyz(1, 4, 1),
                        GridSize::xyz(gx, 1, 1),
                        Some(ManySeriesKernelSelected::Tiled2D { tx: 1, ty: 4 }),
                        gx,
                        1,
                        1,
                        1,
                        4,
                        1,
                    )
                } else if cols >= 8
                    && self
                        .module
                        .get_function("pma_ms1p_tiled_f32_tx1_ty2")
                        .is_ok()
                {
                    let gx = ((cols as u32) + 2 - 1) / 2;
                    (
                        "pma_ms1p_tiled_f32_tx1_ty2",
                        BlockSize::xyz(1, 2, 1),
                        GridSize::xyz(gx, 1, 1),
                        Some(ManySeriesKernelSelected::Tiled2D { tx: 1, ty: 2 }),
                        gx,
                        1,
                        1,
                        1,
                        2,
                        1,
                    )
                } else {
                    let gx = cols as u32;
                    (
                        "pma_many_series_one_param_f32",
                        BlockSize::xy(1, 1),
                        GridSize::xyz(gx, 1, 1),
                        Some(ManySeriesKernelSelected::OneD { block_x: 1 }),
                        gx,
                        1,
                        1,
                        1,
                        1,
                        1,
                    )
                }
            }
        };
        if let Some(s) = sel {
            unsafe {
                (*(self as *const _ as *mut CudaPma)).last_many = Some(s);
            }
        }
        self.maybe_log_many_debug();

        self.validate_launch(gx, gy, gz, bx, by, bz)?;

        let func = self
            .module
            .get_function(fname)
            .map_err(|_| CudaPmaError::MissingKernelSymbol { name: fname })?;
        let mut args: [*mut c_void; 6] = [
            &mut d_prices_tm.as_device_ptr().as_raw() as *mut _ as *mut c_void,
            &mut (cols as i32) as *mut _ as *mut c_void,
            &mut (rows as i32) as *mut _ as *mut c_void,
            &mut d_first_valids.as_device_ptr().as_raw() as *mut _ as *mut c_void,
            &mut d_predict_tm.as_device_ptr().as_raw() as *mut _ as *mut c_void,
            &mut d_trigger_tm.as_device_ptr().as_raw() as *mut _ as *mut c_void,
        ];
        unsafe { self.stream.launch(&func, grid, block, 0, &mut args) }?;
        Ok(())
    }

    pub fn pma_many_series_one_param_time_major_dev(
        &self,
        prices_tm: &[f32],
        cols: usize,
        rows: usize,
    ) -> Result<DevicePmaPair, CudaPmaError> {
        let first_valids = Self::prepare_many_series_inputs(prices_tm, cols, rows)?;
        let elem_f32 = core::mem::size_of::<f32>();
        let elem_i32 = core::mem::size_of::<i32>();
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| {
                CudaPmaError::InvalidInput(format!(
                    "cols * rows overflow when sizing device buffers: cols={} rows={}",
                    cols, rows
                ))
            })?;
        let prices_bytes = elems
            .checked_mul(elem_f32)
            .ok_or_else(|| {
                CudaPmaError::InvalidInput("cols*rows*sizeof(f32) overflow".into())
            })?;
        let first_bytes = cols
            .checked_mul(elem_i32)
            .ok_or_else(|| {
                CudaPmaError::InvalidInput("cols*sizeof(i32) overflow".into())
            })?;
        let out_bytes = elems
            .checked_mul(elem_f32)
            .ok_or_else(|| {
                CudaPmaError::InvalidInput("output bytes overflow".into())
            })?;
        let two_out = out_bytes
            .checked_mul(2)
            .ok_or_else(|| CudaPmaError::InvalidInput("2*out_bytes overflow".into()))?;
        let tmp = prices_bytes
            .checked_add(first_bytes)
            .ok_or_else(|| {
                CudaPmaError::InvalidInput("prices_bytes + first_bytes overflow".into())
            })?;
        let required = tmp
            .checked_add(two_out)
            .ok_or_else(|| {
                CudaPmaError::InvalidInput("total device bytes overflow".into())
            })?;
        let headroom = 64 * 1024 * 1024;
        Self::will_fit(required, headroom)?;

        let mut d_prices_tm: DeviceBuffer<f32> = DeviceBuffer::from_slice(prices_tm)?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_predict_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }?;
        let mut d_trigger_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }?;

        self.launch_many_series_kernel_select(
            &d_prices_tm,
            cols,
            rows,
            &d_first_valids,
            &mut d_predict_tm,
            &mut d_trigger_tm,
        )?;

        Ok(DevicePmaPair {
            predict: DeviceArrayF32 {
                buf: d_predict_tm,
                rows,
                cols,
            },
            trigger: DeviceArrayF32 {
                buf: d_trigger_tm,
                rows,
                cols,
            },
        })
    }
}

// ---------- Bench profiles ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const MANY_SERIES_COLS: usize = 256;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * core::mem::size_of::<f32>();
        // combos fixed = 1; two outputs
        let out_bytes = 2 * ONE_SERIES_LEN * core::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = elems * core::mem::size_of::<f32>();
        let out_bytes = 2 * elems * core::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct PmaBatchState {
        cuda: CudaPma,
        price: Vec<f32>,
        sweep: PmaBatchRange,
    }
    impl CudaBenchState for PmaBatchState {
        fn launch(&mut self) {
            let _ = self.cuda.pma_batch_dev(&self.price, &self.sweep).unwrap();
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaPma::new(0).expect("cuda pma");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = PmaBatchRange::default();
        Box::new(PmaBatchState { cuda, price, sweep })
    }

    struct PmaManyState {
        cuda: CudaPma,
        data_tm: Vec<f32>,
        cols: usize,
        rows: usize,
    }
    impl CudaBenchState for PmaManyState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .pma_many_series_one_param_time_major_dev(&self.data_tm, self.cols, self.rows)
                .unwrap();
        }
    }
    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaPma::new(0).expect("cuda pma");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        Box::new(PmaManyState {
            cuda,
            data_tm,
            cols,
            rows,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "pma",
                "one_series_many_params",
                "pma_cuda_batch_dev",
                "1m_x1",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "pma",
                "many_series_one_param",
                "pma_cuda_many_series_one_param",
                "256x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
