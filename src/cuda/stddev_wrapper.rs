//! CUDA wrapper for rolling Standard Deviation (population) indicator.
//!
//! Parity goals per Agents Guide:
//! - API/behavior match ALMA-style wrappers (policy enums, NON_BLOCKING stream)
//! - PTX load with DetermineTargetFromContext and O2 fallback
//! - VRAM estimation + ~64MB headroom; grid-y chunking to <= 65_535 for batch
//! - Batch uses host prefix sums of x and x^2 and NaN counts (f64 accum)
//! - Many-series×one-param uses time-major scan with incremental raw sums

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::stddev::{StdDevBatchRange, StdDevParams};
use cust::context::Context;
use cust::device::Device;
use cust::error::CudaError;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer, DeviceCopy};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaStddevError {
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
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
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
pub struct CudaStddevPolicy {
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

pub struct CudaStddev {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaStddevPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

// CUDA vector equivalent for float2 (hi, lo) compensated prefix sums
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Float2 { pub x: f32, pub y: f32 }
unsafe impl DeviceCopy for Float2 {}

impl CudaStddev {
    pub fn new(device_id: usize) -> Result<Self, CudaStddevError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/stddev_kernel.ptx"));
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
            context,
            device_id: device_id as u32,
            policy: CudaStddevPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> {
        self.context.clone()
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    pub fn set_policy(&mut self, policy: CudaStddevPolicy) { self.policy = policy; }
    pub fn policy(&self) -> &CudaStddevPolicy { &self.policy }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> { self.last_batch }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> { self.last_many }

    #[inline]
    fn mem_check_enabled() -> bool {
        env::var("CUDA_MEM_CHECK")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(true)
    }
    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }
    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaStddevError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _)) = Self::device_mem_info() {
            if required_bytes.saturating_add(headroom_bytes) <= free {
                Ok(())
            } else {
                Err(CudaStddevError::OutOfMemory {
                    required: required_bytes,
                    free,
                    headroom: headroom_bytes,
                })
            }
        } else {
            Ok(())
        }
    }

    fn validate_launch(grid: GridSize, block: BlockSize) -> Result<(), CudaStddevError> {
        let (gx, gy, gz) = (grid.x, grid.y, grid.z);
        let (bx, by, bz) = (block.x, block.y, block.z);
        let max_threads_per_block = 1024u32;
        if bx
            .saturating_mul(by)
            .saturating_mul(bz)
            > max_threads_per_block
        {
            return Err(CudaStddevError::LaunchConfigTooLarge {
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
        use std::sync::atomic::{AtomicBool, Ordering};
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                if !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] stddev batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaStddev)).debug_batch_logged = true;
                }
            }
        }
    }
    fn maybe_log_many_debug(&self) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                if !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] stddev many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaStddev)).debug_many_logged = true;
                }
            }
        }
    }

    fn expand_grid_checked(r: &StdDevBatchRange) -> Result<Vec<StdDevParams>, CudaStddevError> {
        fn axis_usize(
            (s, e, st): (usize, usize, usize),
        ) -> Result<Vec<usize>, CudaStddevError> {
            if st == 0 || s == e {
                return Ok(vec![s]);
            }
            let mut v = Vec::new();
            if s < e {
                let mut cur = s;
                while cur <= e {
                    v.push(cur);
                    let next = cur.saturating_add(st);
                    if next == cur {
                        break;
                    }
                    cur = next;
                }
            } else {
                let mut cur = s;
                while cur >= e {
                    v.push(cur);
                    let next = cur.saturating_sub(st);
                    if next == cur {
                        break;
                    }
                    cur = next;
                    if cur == 0 && e > 0 {
                        break;
                    }
                }
            }
            if v.is_empty() {
                return Err(CudaStddevError::InvalidInput(format!(
                    "invalid usize range: start={s} end={e} step={st}"
                )));
            }
            Ok(v)
        }
        fn axis_f64(
            (s, e, st): (f64, f64, f64),
        ) -> Result<Vec<f64>, CudaStddevError> {
            if st.abs() < 1e-12 || (s - e).abs() < 1e-12 {
                return Ok(vec![s]);
            }
            let mut out = Vec::new();
            if s < e {
                let step = if st > 0.0 { st } else { -st };
                let mut x = s;
                while x <= e + 1e-12 {
                    out.push(x);
                    x += step;
                }
            } else {
                let step = if st > 0.0 { -st } else { st };
                if step.abs() < 1e-12 {
                    return Ok(vec![s]);
                }
                let mut x = s;
                while x >= e - 1e-12 {
                    out.push(x);
                    x += step;
                }
            }
            if out.is_empty() {
                return Err(CudaStddevError::InvalidInput(format!(
                    "invalid f64 range: start={s} end={e} step={st}"
                )));
            }
            Ok(out)
        }

        let periods = axis_usize(r.period)?;
        let nbdevs = axis_f64(r.nbdev)?;
        let cap = periods
            .len()
            .checked_mul(nbdevs.len())
            .ok_or_else(|| {
                CudaStddevError::InvalidInput(
                    "stddev CUDA: parameter grid size overflow".into(),
                )
            })?;

        let mut out = Vec::with_capacity(cap);
        for &p in &periods {
            for &n in &nbdevs {
                out.push(StdDevParams {
                    period: Some(p),
                    nbdev: Some(n),
                });
            }
        }
        if out.is_empty() {
            return Err(CudaStddevError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        Ok(out)
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &StdDevBatchRange,
    ) -> Result<(Vec<(usize, f32)>, usize, usize), CudaStddevError> {
        if data_f32.is_empty() {
            return Err(CudaStddevError::InvalidInput("empty data".into()));
        }
        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaStddevError::InvalidInput("all values are NaN".into()))?;
        let combos = Self::expand_grid_checked(sweep)?;

        let mut out = Vec::with_capacity(combos.len());
        for c in combos {
            let p = c.period.unwrap_or(0);
            if p == 0 {
                return Err(CudaStddevError::InvalidInput("period must be > 0".into()));
            }
            if p > len {
                return Err(CudaStddevError::InvalidInput(
                    "period exceeds data length".into(),
                ));
            }
            if len - first_valid < p {
                return Err(CudaStddevError::InvalidInput(
                    "not enough valid data after first_valid".into(),
                ));
            }
            let nb = c.nbdev.unwrap_or(1.0) as f32;
            if !nb.is_finite() || nb < 0.0 {
                return Err(CudaStddevError::InvalidInput(
                    "nbdev must be non-negative and finite".into(),
                ));
            }
            out.push((p, nb));
        }
        Ok((out, first_valid, len))
    }

    #[inline(always)]
    fn f64_to_float2(v: f64) -> Float2 {
        let hi = v as f32;
        let lo = (v - hi as f64) as f32;
        Float2 { x: hi, y: lo }
    }

    // Build DS (hi, lo) prefix sums directly into page-locked host buffers.
    fn build_prefixes_ds_locked(
        data: &[f32],
    ) -> cust::error::CudaResult<(LockedBuffer<Float2>, LockedBuffer<Float2>, LockedBuffer<i32>)> {
        let n = data.len();
        let cap = n.checked_add(1).ok_or_else(|| cust::error::CudaError::InvalidValue)?;
        let mut ps1: LockedBuffer<Float2> = unsafe { LockedBuffer::uninitialized(cap)? };
        let mut ps2: LockedBuffer<Float2> = unsafe { LockedBuffer::uninitialized(cap)? };
        let mut psn: LockedBuffer<i32> = unsafe { LockedBuffer::uninitialized(cap)? };

        ps1.as_mut_slice()[0] = Float2 { x: 0.0, y: 0.0 };
        ps2.as_mut_slice()[0] = Float2 { x: 0.0, y: 0.0 };
        psn.as_mut_slice()[0] = 0;

        let (mut s1, mut s2) = (0.0f64, 0.0f64);
        let mut nan = 0i32;
        for i in 0..n {
            let v = data[i];
            if v.is_nan() {
                nan += 1;
            } else {
                let d = v as f64;
                s1 += d;
                s2 += d * d;
            }
            ps1.as_mut_slice()[i + 1] = Self::f64_to_float2(s1);
            ps2.as_mut_slice()[i + 1] = Self::f64_to_float2(s2);
            psn.as_mut_slice()[i + 1] = nan;
        }

        Ok((ps1, ps2, psn))
    }

    fn launch_batch(
        &self,
        d_ps1: &DeviceBuffer<Float2>,
        d_ps2: &DeviceBuffer<Float2>,
        d_psn: &DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        d_periods: &DeviceBuffer<i32>,
        d_nbdevs: &DeviceBuffer<f32>,
        combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaStddevError> {
        let func = self
            .module
            .get_function("stddev_batch_f32")
            .map_err(|_| CudaStddevError::MissingKernelSymbol { name: "stddev_batch_f32" })?;

        #[inline(always)]
        fn pick_block_x(len: usize) -> u32 {
            if len >= (1usize << 20) { 512 } else if len >= (1usize << 14) { 256 } else { 128 }
        }
        let block_x: u32 = match self.policy.batch { BatchKernelPolicy::Auto => pick_block_x(len), BatchKernelPolicy::Plain { block_x } => block_x.max(64) };
        let grid_x: u32 = ((len as u32) + block_x - 1) / block_x;
        let grid_base: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        Self::validate_launch(grid_base, block)?;
        unsafe {
            (*(self as *const _ as *mut CudaStddev)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        unsafe {
            (*(self as *const _ as *mut CudaStddev)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }

        let mut launched = 0usize;
        while launched < combos {
            let chunk = (combos - launched).min(65_535);
            let grid: GridSize = (grid_base.x, chunk as u32, 1).into();
            unsafe {
                let mut ps1 = d_ps1.as_device_ptr().as_raw();
                let mut ps2 = d_ps2.as_device_ptr().as_raw();
                let mut psn = d_psn.as_device_ptr().as_raw();
                let mut len_i = len as i32;
                let mut first_valid_i = first_valid as i32;
                let mut periods = d_periods
                    .as_device_ptr()
                    .as_raw()
                    .saturating_add((launched as u64) * (std::mem::size_of::<i32>() as u64));
                let mut nbdevs = d_nbdevs
                    .as_device_ptr()
                    .as_raw()
                    .saturating_add((launched as u64) * (std::mem::size_of::<f32>() as u64));
                let mut combos_i = chunk as i32;
                let mut outp = d_out.as_device_ptr().as_raw().saturating_add(
                    ((launched * len) as u64) * (std::mem::size_of::<f32>() as u64),
                );
                let mut outp = d_out.as_device_ptr().as_raw().saturating_add(
                    ((launched * len) as u64) * (std::mem::size_of::<f32>() as u64),
                );
                let args: &mut [*mut c_void] = &mut [
                    &mut ps1 as *mut _ as *mut c_void,
                    &mut ps2 as *mut _ as *mut c_void,
                    &mut psn as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_valid_i as *mut _ as *mut c_void,
                    &mut periods as *mut _ as *mut c_void,
                    &mut nbdevs as *mut _ as *mut c_void,
                    &mut combos_i as *mut _ as *mut c_void,
                    &mut outp as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, 0, args)?;
            }
            launched += chunk;
        }
        Ok(())
    }

    pub fn stddev_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &StdDevBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<StdDevParams>), CudaStddevError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let periods: Vec<i32> = combos.iter().map(|c| c.0 as i32).collect();
        let nbdevs: Vec<f32> = combos.iter().map(|c| c.1).collect();
        // Build DS prefixes in pinned memory
        let (h_ps1, h_ps2, h_psn) = Self::build_prefixes_ds_locked(data_f32)
            .map_err(CudaStddevError::Cuda)?;

        // VRAM estimate with checked arithmetic
        let item_f2 = std::mem::size_of::<Float2>();
        let item_i32 = std::mem::size_of::<i32>();
        let item_f32 = std::mem::size_of::<f32>();
        let prefix_len = h_ps1
            .len()
            .checked_add(h_ps2.len())
            .ok_or_else(|| CudaStddevError::InvalidInput("prefix length overflow".into()))?;
        let bytes_prefix_f2 = prefix_len
            .checked_mul(item_f2)
            .ok_or_else(|| CudaStddevError::InvalidInput("prefix bytes overflow".into()))?;
        let bytes_prefix = bytes_prefix_f2
            .checked_add(
                h_psn
                    .len()
                    .checked_mul(item_i32)
                    .ok_or_else(|| {
                        CudaStddevError::InvalidInput("nan-count bytes overflow".into())
                    })?,
            )
            .ok_or_else(|| CudaStddevError::InvalidInput("prefix total bytes overflow".into()))?;
        let bytes_params = periods
            .len()
            .checked_mul(item_i32)
            .and_then(|v| v.checked_add(nbdevs.len().checked_mul(item_f32)?))
            .ok_or_else(|| CudaStddevError::InvalidInput("param bytes overflow".into()))?;
        let bytes_out = combos
            .len()
            .checked_mul(len)
            .and_then(|v| v.checked_mul(item_f32))
            .ok_or_else(|| CudaStddevError::InvalidInput("output bytes overflow".into()))?;
        let required = bytes_prefix
            .checked_add(bytes_params)
            .and_then(|v| v.checked_add(bytes_out))
            .ok_or_else(|| CudaStddevError::InvalidInput("total bytes overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        Self::will_fit(required, headroom)?;

        // Device allocations + async H->D from pinned
        let mut d_ps1: DeviceBuffer<Float2> =
            unsafe { DeviceBuffer::uninitialized_async(h_ps1.len(), &self.stream) }?;
        let mut d_ps2: DeviceBuffer<Float2> =
            unsafe { DeviceBuffer::uninitialized_async(h_ps2.len(), &self.stream) }?;
        let mut d_psn: DeviceBuffer<i32> =
            unsafe { DeviceBuffer::uninitialized_async(h_psn.len(), &self.stream) }?;
        unsafe {
            d_ps1.async_copy_from(&h_ps1, &self.stream)?;
            d_ps2.async_copy_from(&h_ps2, &self.stream)?;
            d_psn.async_copy_from(&h_psn, &self.stream)?;
        }
        let d_periods = DeviceBuffer::from_slice(&periods)?;
        let d_nbdevs = DeviceBuffer::from_slice(&nbdevs)?;
        let out_len = combos
            .len()
            .checked_mul(len)
            .ok_or_else(|| CudaStddevError::InvalidInput("output length overflow".into()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_len, &self.stream) }?;

        self.launch_batch(
            &d_ps1,
            &d_ps2,
            &d_psn,
            len,
            first_valid,
            &d_periods,
            &d_nbdevs,
            combos.len(),
            &mut d_out,
        )?;
        self.stream.synchronize()?;
        self.maybe_log_batch_debug();

        let params: Vec<StdDevParams> = combos
            .iter()
            .map(|(p, nb)| StdDevParams {
                period: Some(*p),
                nbdev: Some(*nb as f64),
            })
            .collect();
        Ok((
            DeviceArrayF32 {
                buf: d_out,
                rows: params.len(),
                cols: len,
            },
            params,
        ))
    }

    pub fn stddev_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &StdDevBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<StdDevParams>), CudaStddevError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos
            .len()
            .checked_mul(len)
            .ok_or_else(|| {
                CudaStddevError::InvalidInput("output length overflow in stddev_batch_into_host_f32".into())
            })?;
        if out.len() != expected {
            return Err(CudaStddevError::InvalidInput(format!(
                "output slice length mismatch (expected {}, got {})",
                expected,
                out.len()
            )));
        }
        let (dev, params) = self.stddev_batch_dev(data_f32, sweep)?;
        dev.buf.copy_to(out)?;
        Ok((combos.len(), len, params))
    }

    pub fn stddev_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
        nbdev: f32,
    ) -> Result<DeviceArrayF32, CudaStddevError> {
        if cols == 0 || rows == 0 {
            return Err(CudaStddevError::InvalidInput("empty matrix".into()));
        }
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaStddevError::InvalidInput("cols*rows overflow".into()))?;
        if data_tm_f32.len() != elems {
            return Err(CudaStddevError::InvalidInput(
                "time-major slice length mismatch".into(),
            ));
        }
        if period == 0 {
            return Err(CudaStddevError::InvalidInput("period must be > 0".into()));
        }
        if !nbdev.is_finite() || nbdev < 0.0 {
            return Err(CudaStddevError::InvalidInput(
                "nbdev must be non-negative and finite".into(),
            ));
        }
        // Per-series first_valid
        let mut first_valids = vec![-1i32; cols];
        for s in 0..cols {
            let mut fv = -1;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + s];
                if !v.is_nan() {
                    fv = t as i32;
                    break;
                }
            }
            first_valids[s] = fv;
        }
        for s in 0..cols {
            let mut fv = -1;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + s];
                if !v.is_nan() {
                    fv = t as i32;
                    break;
                }
            }
            first_valids[s] = fv;
        }

        // VRAM estimate
        let item_f32 = std::mem::size_of::<f32>();
        let item_i32 = std::mem::size_of::<i32>();
        let bytes_in = elems
            .checked_mul(item_f32)
            .ok_or_else(|| CudaStddevError::InvalidInput("input bytes overflow".into()))?;
        let bytes_fv = cols
            .checked_mul(item_i32)
            .ok_or_else(|| CudaStddevError::InvalidInput("first_valid bytes overflow".into()))?;
        let bytes_out = elems
            .checked_mul(item_f32)
            .ok_or_else(|| CudaStddevError::InvalidInput("output bytes overflow".into()))?;
        let required = bytes_in
            .checked_add(bytes_fv)
            .and_then(|v| v.checked_add(bytes_out))
            .ok_or_else(|| CudaStddevError::InvalidInput("total bytes overflow".into()))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        let d_in = DeviceBuffer::from_slice(data_tm_f32)?;
        let d_fv = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out =
            unsafe { DeviceBuffer::<f32>::uninitialized(elems) }?;

        let func = self
            .module
            .get_function("stddev_many_series_one_param_f32")
            .map_err(|_| {
                CudaStddevError::MissingKernelSymbol {
                    name: "stddev_many_series_one_param_f32",
                }
            })?;
        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 128,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32),
        };
        let grid: GridSize = (cols as u32, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        Self::validate_launch(grid, block)?;
        unsafe {
            (*(self as *const _ as *mut CudaStddev)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        
        unsafe {
            let mut in_ptr = d_in.as_device_ptr().as_raw();
            let mut fv_ptr = d_fv.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut nbdev_f = nbdev as f32;
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut in_ptr as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut nbdev_f as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        self.stream.synchronize()?;
        self.maybe_log_many_debug();

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }
}

// ---------- Bench profiles ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;

    fn bytes_one_series_many_params() -> usize {
        let prefixes = 2 * (ONE_SERIES_LEN + 1) * std::mem::size_of::<Float2>()
            + (ONE_SERIES_LEN + 1) * std::mem::size_of::<i32>();
        let params = PARAM_SWEEP * (std::mem::size_of::<i32>() + std::mem::size_of::<f32>());
        let out = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        prefixes + params + out + 64 * 1024 * 1024
    }

    struct StddevBatchState {
        cuda: CudaStddev,
        price: Vec<f32>,
        sweep: StdDevBatchRange,
    }
    impl CudaBenchState for StddevBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .stddev_batch_dev(&self.price, &self.sweep)
                .expect("stddev batch");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaStddev::new(0).expect("cuda stddev");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = StdDevBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
            nbdev: (2.0, 2.0, 0.0),
        };
        Box::new(StddevBatchState { cuda, price, sweep })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "stddev",
            "one_series_many_params",
            "stddev_cuda_batch_dev",
            "1m_x_250",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}
