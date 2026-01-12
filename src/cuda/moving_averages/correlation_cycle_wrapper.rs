#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::correlation_cycle::{CorrelationCycleBatchRange, CorrelationCycleParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{
    mem_get_info, AsyncCopyDestination, CopyDestination, DeviceBuffer, LockedBuffer,
};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::sys as cu;
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CudaCorrelationCycleError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
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

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,

    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,

    Tiled2D { tx: u32, ty: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaCorrelationCyclePolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaCorrelationCyclePolicy {
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
    Tiled2D { tx: u32, ty: u32 },
}

pub struct DeviceCorrelationCycleQuad {
    pub real: DeviceArrayF32,
    pub imag: DeviceArrayF32,
    pub angle: DeviceArrayF32,
    pub state: DeviceArrayF32,
}
impl DeviceCorrelationCycleQuad {
    #[inline]
    pub fn rows(&self) -> usize {
        self.real.rows
    }
    #[inline]
    pub fn cols(&self) -> usize {
        self.real.cols
    }
}

pub struct CudaCorrelationCycle {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaCorrelationCyclePolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaCorrelationCycle {
    pub fn new(device_id: usize) -> Result<Self, CudaCorrelationCycleError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/correlation_cycle_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
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
            policy: CudaCorrelationCyclePolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn set_policy(&mut self, p: CudaCorrelationCyclePolicy) {
        self.policy = p;
    }
    pub fn policy(&self) -> &CudaCorrelationCyclePolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
    pub fn synchronize(&self) -> Result<(), CudaCorrelationCycleError> {
        self.stream.synchronize().map_err(Into::into)
    }

    pub fn ctx(&self) -> Arc<Context> {
        self.context.clone()
    }
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] CC batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut Self)).debug_batch_logged = true;
                }
            }
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] CC many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut Self)).debug_many_logged = true;
                }
            }
        }
    }

    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }
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

    #[inline]
    fn validate_launch(
        &self,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    ) -> Result<(), CudaCorrelationCycleError> {
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
        let threads = bx.checked_mul(by).and_then(|v| v.checked_mul(bz)).ok_or(
            CudaCorrelationCycleError::LaunchConfigTooLarge {
                gx,
                gy,
                gz,
                bx,
                by,
                bz,
            },
        )?;
        if threads > max_threads
            || bx > max_bx
            || by > max_by
            || bz > max_bz
            || gx > max_gx
            || gy > max_gy
            || gz > max_gz
        {
            return Err(CudaCorrelationCycleError::LaunchConfigTooLarge {
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

    #[inline]
    fn pick_launch_1d(&self, n: usize, bx_opt: Option<u32>) -> (GridSize, BlockSize, u32) {
        let bx = bx_opt.unwrap_or(256).clamp(64, 1024);
        let blocks = ((n + bx as usize - 1) / bx as usize).min(65_535) as u32;
        ((blocks, 1, 1).into(), (bx, 1, 1).into(), bx)
    }

    #[inline]
    fn pick_launch_2d(
        &self,
        cols: usize,
        rows: usize,
        tx_ty: Option<(u32, u32)>,
    ) -> (GridSize, BlockSize, (u32, u32)) {
        let (tx, ty) = tx_ty.unwrap_or((128, 4));
        let gx = ((rows + tx as usize - 1) / tx as usize).min(65_535) as u32;
        let gy = ((cols + ty as usize - 1) / ty as usize).min(65_535) as u32;
        ((gx, gy, 1).into(), (tx, ty, 1).into(), (tx, ty))
    }

    fn compute_trig_weights_and_consts(period: usize) -> (Vec<f32>, Vec<f32>, f32, f32, f32, f32) {
        let mut wcos = vec![0f32; period];
        let mut wsin = vec![0f32; period];
        if period == 0 {
            return (wcos, wsin, 0.0, 0.0, 0.0, 0.0);
        }
        let two_pi = 2.0_f64 * std::f64::consts::PI;
        let n = period as f64;
        let w = two_pi / n;
        let mut sum_c = 0.0f64;
        let mut sum_s = 0.0f64;
        let mut sum_c2 = 0.0f64;
        let mut sum_s2 = 0.0f64;
        for j in 0..period {
            let a = w * ((j as f64) + 1.0);
            let (s, c) = a.sin_cos();
            let ys = -s;
            wcos[j] = c as f32;
            wsin[j] = ys as f32;
            sum_c += c;
            sum_s += ys;
            sum_c2 += c * c;
            sum_s2 += ys * ys;
        }
        let t2 = n.mul_add(sum_c2, -(sum_c * sum_c));
        let t4 = n.mul_add(sum_s2, -(sum_s * sum_s));
        let sqrt_t2 = if t2 > 0.0 { t2.sqrt() as f32 } else { 0.0 };
        let sqrt_t4 = if t4 > 0.0 { t4.sqrt() as f32 } else { 0.0 };
        (wcos, wsin, sum_c as f32, sum_s as f32, sqrt_t2, sqrt_t4)
    }

    fn expand_grid(r: &CorrelationCycleBatchRange) -> Vec<CorrelationCycleParams> {
        fn axis_usize(a: (usize, usize, usize)) -> Vec<usize> {
            let (start, end, step) = a;
            if step == 0 || start == end {
                return vec![start];
            }
            let mut vals = Vec::new();
            if start < end {
                let mut v = start;
                loop {
                    vals.push(v);
                    if v >= end {
                        break;
                    }
                    let next = match v.checked_add(step) {
                        Some(n) => n,
                        None => break,
                    };
                    if next == v {
                        break;
                    }
                    v = next;
                }
            } else {
                let mut v = start;
                loop {
                    vals.push(v);
                    if v <= end {
                        break;
                    }
                    let next = v.saturating_sub(step);
                    if next == v {
                        break;
                    }
                    v = next;
                }
            }
            vals
        }
        fn axis_f64(a: (f64, f64, f64)) -> Vec<f64> {
            let (start, end, step) = a;
            if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
                return vec![start];
            }
            let mut vals = Vec::new();
            if start <= end {
                let mut x = start;
                loop {
                    vals.push(x);
                    if x >= end {
                        break;
                    }
                    let next = x + step;
                    if !next.is_finite() || next == x {
                        break;
                    }
                    x = next;
                    if x > end + 1e-12 {
                        break;
                    }
                }
            } else {
                let mut x = start;
                loop {
                    vals.push(x);
                    if x <= end {
                        break;
                    }
                    let next = x - step.abs();
                    if !next.is_finite() || next == x {
                        break;
                    }
                    x = next;
                    if x < end - 1e-12 {
                        break;
                    }
                }
            }
            vals
        }
        let periods = axis_usize(r.period);
        let thresholds = axis_f64(r.threshold);
        let mut out = Vec::with_capacity(periods.len() * thresholds.len());
        for &p in &periods {
            for &t in &thresholds {
                out.push(CorrelationCycleParams {
                    period: Some(p),
                    threshold: Some(t),
                });
            }
        }
        out
    }

    pub fn correlation_cycle_batch_dev(
        &mut self,
        data_f32: &[f32],
        sweep: &CorrelationCycleBatchRange,
    ) -> Result<DeviceCorrelationCycleQuad, CudaCorrelationCycleError> {
        let series_len = data_f32.len();
        if series_len == 0 {
            return Err(CudaCorrelationCycleError::InvalidInput(
                "empty input".into(),
            ));
        }
        let first_valid = data_f32
            .iter()
            .position(|x| x.is_finite())
            .unwrap_or(series_len);
        if first_valid >= series_len {
            return Err(CudaCorrelationCycleError::InvalidInput("all NaN".into()));
        }

        let combos = Self::expand_grid(sweep);
        let n_combos = combos.len();
        if n_combos == 0 {
            return Err(CudaCorrelationCycleError::InvalidInput(
                "empty sweep".into(),
            ));
        }
        let max_period = combos
            .iter()
            .map(|c| c.period.unwrap_or(0))
            .max()
            .unwrap_or(0);
        if max_period == 0 {
            return Err(CudaCorrelationCycleError::InvalidInput("period=0".into()));
        }

        let mut periods_i32 = vec![0i32; n_combos];
        let mut thresholds_f32 = vec![0f32; n_combos];
        let mut sum_cos = vec![0f32; n_combos];
        let mut sum_sin = vec![0f32; n_combos];
        let mut sqrt_t2 = vec![0f32; n_combos];
        let mut sqrt_t4 = vec![0f32; n_combos];
        let mut cos_flat = vec![0f32; n_combos * max_period];
        let mut sin_flat = vec![0f32; n_combos * max_period];
        for (i, prm) in combos.iter().enumerate() {
            let p = prm.period.unwrap();
            let t = prm.threshold.unwrap() as f32;
            periods_i32[i] = p as i32;
            thresholds_f32[i] = t;
            let (wc, ws, sc, ss, st2, st4) = Self::compute_trig_weights_and_consts(p);
            let base = i * max_period;
            cos_flat[base..base + p].copy_from_slice(&wc);
            sin_flat[base..base + p].copy_from_slice(&ws);
            sum_cos[i] = sc;
            sum_sin[i] = ss;
            sqrt_t2[i] = st2;
            sqrt_t4[i] = st4;
        }

        let total_out_elems = n_combos
            .checked_mul(series_len)
            .ok_or_else(|| CudaCorrelationCycleError::InvalidInput("rows*cols overflow".into()))?;
        let f32_bytes = std::mem::size_of::<f32>();
        let i32_bytes = std::mem::size_of::<i32>();
        let prices_bytes = series_len.checked_mul(f32_bytes).ok_or_else(|| {
            CudaCorrelationCycleError::InvalidInput("series_len bytes overflow".into())
        })?;

        let per_combo_meta = i32_bytes
            .checked_add(5usize.checked_mul(f32_bytes).ok_or_else(|| {
                CudaCorrelationCycleError::InvalidInput("meta bytes overflow".into())
            })?)
            .ok_or_else(|| CudaCorrelationCycleError::InvalidInput("meta bytes overflow".into()))?;
        let meta_bytes = n_combos.checked_mul(per_combo_meta).ok_or_else(|| {
            CudaCorrelationCycleError::InvalidInput("params bytes overflow".into())
        })?;

        let weight_elems = n_combos.checked_mul(max_period).ok_or_else(|| {
            CudaCorrelationCycleError::InvalidInput("n_combos*max_period overflow".into())
        })?;
        let weight_bytes = weight_elems
            .checked_mul(2usize.checked_mul(f32_bytes).ok_or_else(|| {
                CudaCorrelationCycleError::InvalidInput("weight bytes overflow".into())
            })?)
            .ok_or_else(|| {
                CudaCorrelationCycleError::InvalidInput("weight bytes overflow".into())
            })?;
        let params_bytes = meta_bytes.checked_add(weight_bytes).ok_or_else(|| {
            CudaCorrelationCycleError::InvalidInput("params bytes overflow".into())
        })?;

        let outputs_bytes = total_out_elems
            .checked_mul(4)
            .and_then(|v| v.checked_mul(f32_bytes))
            .ok_or_else(|| {
                CudaCorrelationCycleError::InvalidInput("output bytes overflow".into())
            })?;
        let required = prices_bytes
            .checked_add(params_bytes)
            .and_then(|v| v.checked_add(outputs_bytes))
            .ok_or_else(|| CudaCorrelationCycleError::InvalidInput("VRAM size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            let free = Self::device_mem_info().map(|(f, _)| f).unwrap_or(0);
            return Err(CudaCorrelationCycleError::OutOfMemory {
                required,
                free,
                headroom,
            });
        }

        let d_prices = unsafe { DeviceBuffer::from_slice_async(data_f32, &self.stream) }?;
        let d_cos = DeviceBuffer::from_slice(&cos_flat)?;
        let d_sin = DeviceBuffer::from_slice(&sin_flat)?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)?;
        let d_thresholds = DeviceBuffer::from_slice(&thresholds_f32)?;
        let d_sum_cos = DeviceBuffer::from_slice(&sum_cos)?;
        let d_sum_sin = DeviceBuffer::from_slice(&sum_sin)?;
        let d_sqrt_t2 = DeviceBuffer::from_slice(&sqrt_t2)?;
        let d_sqrt_t4 = DeviceBuffer::from_slice(&sqrt_t4)?;
        let total = total_out_elems;
        let mut d_real: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total) }?;
        let mut d_imag: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total) }?;
        let mut d_angle: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total) }?;
        let mut d_state: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total) }?;

        let (grid, block, bx) = match self.policy.batch {
            BatchKernelPolicy::Auto | BatchKernelPolicy::Plain { .. } => {
                self.pick_launch_1d(series_len, Some(256))
            }
        };
        self.last_batch = Some(BatchKernelSelected::Plain { block_x: bx });
        self.maybe_log_batch_debug();

        let func_ria = self
            .module
            .get_function("correlation_cycle_batch_f32_ria")
            .map_err(|_| CudaCorrelationCycleError::MissingKernelSymbol {
                name: "correlation_cycle_batch_f32_ria",
            })?;
        let smem = (max_period * 2 * std::mem::size_of::<f32>()) as u32;
        let stream = &self.stream;

        let mut processed = 0usize;
        while processed < n_combos {
            let chunk = (n_combos - processed).min(65_535);
            let grid_chunk: GridSize = (grid.x, chunk as u32, 1).into();
            self.validate_launch(
                grid_chunk.x,
                grid_chunk.y,
                grid_chunk.z,
                block.x,
                block.y,
                block.z,
            )?;
            let out_off = processed * series_len;
            let out_real_ptr = unsafe { d_real.as_device_ptr().add(out_off) };
            let out_imag_ptr = unsafe { d_imag.as_device_ptr().add(out_off) };
            let out_angle_ptr = unsafe { d_angle.as_device_ptr().add(out_off) };
            unsafe {
                launch!(func_ria<<<grid_chunk, block, smem, stream>>>(
                    d_prices.as_device_ptr(),
                    d_cos.as_device_ptr(),
                    d_sin.as_device_ptr(),
                    d_periods.as_device_ptr(),
                    d_sum_cos.as_device_ptr(),
                    d_sum_sin.as_device_ptr(),
                    d_sqrt_t2.as_device_ptr(),
                    d_sqrt_t4.as_device_ptr(),
                    max_period as i32,
                    series_len as i32,
                    n_combos as i32,
                    first_valid as i32,
                    processed as i32,
                    out_real_ptr,
                    out_imag_ptr,
                    out_angle_ptr
                ))?;
            }

            let func_state = self
                .module
                .get_function("correlation_cycle_state_batch_f32")
                .map_err(|_| CudaCorrelationCycleError::MissingKernelSymbol {
                    name: "correlation_cycle_state_batch_f32",
                })?;
            let out_state_ptr = unsafe { d_state.as_device_ptr().add(out_off) };
            unsafe {
                launch!(func_state<<<grid_chunk, block, 0, stream>>>(
                    d_angle.as_device_ptr(),
                    d_thresholds.as_device_ptr(),
                    d_periods.as_device_ptr(),
                    series_len as i32,
                    n_combos as i32,
                    first_valid as i32,
                    processed as i32,
                    out_state_ptr
                ))?;
            }
            processed += chunk;
        }

        self.synchronize()?;

        Ok(DeviceCorrelationCycleQuad {
            real: DeviceArrayF32 {
                buf: d_real,
                rows: n_combos,
                cols: series_len,
            },
            imag: DeviceArrayF32 {
                buf: d_imag,
                rows: n_combos,
                cols: series_len,
            },
            angle: DeviceArrayF32 {
                buf: d_angle,
                rows: n_combos,
                cols: series_len,
            },
            state: DeviceArrayF32 {
                buf: d_state,
                rows: n_combos,
                cols: series_len,
            },
        })
    }

    pub fn correlation_cycle_many_series_one_param_time_major_dev(
        &mut self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &CorrelationCycleParams,
    ) -> Result<DeviceCorrelationCycleQuad, CudaCorrelationCycleError> {
        if cols == 0 || rows == 0 {
            return Err(CudaCorrelationCycleError::InvalidInput("empty dims".into()));
        }
        let period = params.period.unwrap_or(20);
        let threshold = params.threshold.unwrap_or(9.0) as f32;
        if period == 0 {
            return Err(CudaCorrelationCycleError::InvalidInput("period=0".into()));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaCorrelationCycleError::InvalidInput("rows*cols overflow".into()))?;
        if prices_tm_f32.len() != expected {
            return Err(CudaCorrelationCycleError::InvalidInput(
                "flat data size mismatch".into(),
            ));
        }

        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = 0usize;
            let mut found = false;
            for t in 0..rows {
                let v = prices_tm_f32[t * cols + s];
                if v.is_finite() {
                    fv = t;
                    found = true;
                    break;
                }
            }
            first_valids[s] = if found { fv as i32 } else { rows as i32 };
        }
        let (wcos, wsin, sum_c, sum_s, st2, st4) = Self::compute_trig_weights_and_consts(period);

        let elems = expected;
        let f32_bytes = std::mem::size_of::<f32>();
        let prices_bytes = elems.checked_mul(f32_bytes).ok_or_else(|| {
            CudaCorrelationCycleError::InvalidInput("prices bytes overflow".into())
        })?;

        let weights_bytes = period
            .checked_mul(2)
            .and_then(|v| v.checked_mul(f32_bytes))
            .ok_or_else(|| {
                CudaCorrelationCycleError::InvalidInput("weights bytes overflow".into())
            })?;

        let first_valid_bytes = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| {
                CudaCorrelationCycleError::InvalidInput("first_valid bytes overflow".into())
            })?;

        let outputs_bytes = elems
            .checked_mul(4)
            .and_then(|v| v.checked_mul(f32_bytes))
            .ok_or_else(|| {
                CudaCorrelationCycleError::InvalidInput("output bytes overflow".into())
            })?;
        let required = prices_bytes
            .checked_add(weights_bytes)
            .and_then(|v| v.checked_add(first_valid_bytes))
            .and_then(|v| v.checked_add(outputs_bytes))
            .ok_or_else(|| CudaCorrelationCycleError::InvalidInput("VRAM size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            let free = Self::device_mem_info().map(|(f, _)| f).unwrap_or(0);
            return Err(CudaCorrelationCycleError::OutOfMemory {
                required,
                free,
                headroom,
            });
        }

        let d_prices_tm = unsafe { DeviceBuffer::from_slice_async(prices_tm_f32, &self.stream) }?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)?;
        let d_wcos = DeviceBuffer::from_slice(&wcos)?;
        let d_wsin = DeviceBuffer::from_slice(&wsin)?;
        let mut d_real: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }?;
        let mut d_imag: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }?;
        let mut d_angle: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }?;
        let mut d_state: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }?;

        let (grid, block, (tx, ty)) = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto | ManySeriesKernelPolicy::Tiled2D { .. } => {
                self.pick_launch_2d(cols, rows, Some((128, 4)))
            }
        };
        self.last_many = Some(ManySeriesKernelSelected::Tiled2D { tx, ty });
        self.maybe_log_many_debug();

        let func_ria = self
            .module
            .get_function("correlation_cycle_many_series_one_param_f32_ria")
            .map_err(|_| CudaCorrelationCycleError::MissingKernelSymbol {
                name: "correlation_cycle_many_series_one_param_f32_ria",
            })?;
        let smem = (period * 2 * std::mem::size_of::<f32>()) as u32;
        let stream = &self.stream;
        self.validate_launch(grid.x, grid.y, grid.z, block.x, block.y, block.z)?;
        unsafe {
            launch!(func_ria<<<grid, block, smem, stream>>>(
                d_prices_tm.as_device_ptr(),
                d_wcos.as_device_ptr(),
                d_wsin.as_device_ptr(),
                sum_c, sum_s, st2, st4,
                cols as i32, rows as i32, period as i32,
                d_first_valids.as_device_ptr(),
                d_real.as_device_ptr(), d_imag.as_device_ptr(), d_angle.as_device_ptr()
            ))
            .map_err(CudaCorrelationCycleError::from)?;
        }

        let func_state = self
            .module
            .get_function("correlation_cycle_state_many_series_one_param_f32")
            .map_err(|_| CudaCorrelationCycleError::MissingKernelSymbol {
                name: "correlation_cycle_state_many_series_one_param_f32",
            })?;
        let stream = &self.stream;
        unsafe {
            launch!(func_state<<<grid, block, 0, stream>>>(
                d_angle.as_device_ptr(),
                threshold,
                d_first_valids.as_device_ptr(),
                cols as i32, rows as i32, period as i32,
                d_state.as_device_ptr()
            ))
            .map_err(CudaCorrelationCycleError::from)?;
        }

        self.synchronize()?;

        Ok(DeviceCorrelationCycleQuad {
            real: DeviceArrayF32 {
                buf: d_real,
                rows,
                cols,
            },
            imag: DeviceArrayF32 {
                buf: d_imag,
                rows,
                cols,
            },
            angle: DeviceArrayF32 {
                buf: d_angle,
                rows,
                cols,
            },
            state: DeviceArrayF32 {
                buf: d_state,
                rows,
                cols,
            },
        })
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 250_000;
    const PARAM_SWEEP: usize = 64;
    const MANY_SERIES_COLS: usize = 128;
    const MANY_SERIES_LEN: usize = 500_000;

    struct CcBatchState {
        cuda: CudaCorrelationCycle,
        d_prices: DeviceBuffer<f32>,
        d_cos: DeviceBuffer<f32>,
        d_sin: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_th: DeviceBuffer<f32>,
        d_sumc: DeviceBuffer<f32>,
        d_sums: DeviceBuffer<f32>,
        d_st2: DeviceBuffer<f32>,
        d_st4: DeviceBuffer<f32>,
        d_real: DeviceBuffer<f32>,
        d_imag: DeviceBuffer<f32>,
        d_ang: DeviceBuffer<f32>,
        d_st: DeviceBuffer<f32>,
        first_valid: usize,
        max_period: usize,
        series_len: usize,
        n_combos: usize,
    }
    impl CudaBenchState for CcBatchState {
        fn launch(&mut self) {
            let (grid, block, _bx) = self.cuda.pick_launch_1d(self.series_len, Some(256));
            let func_ria = self
                .cuda
                .module
                .get_function("correlation_cycle_batch_f32_ria")
                .unwrap();
            let smem = (self.max_period * 2 * std::mem::size_of::<f32>()) as u32;
            let stream = &self.cuda.stream;
            let mut processed = 0usize;
            while processed < self.n_combos {
                let chunk = (self.n_combos - processed).min(65_535);
                let grid_chunk: GridSize = (grid.x, chunk as u32, 1).into();
                let out_off = processed * self.series_len;
                let out_real_ptr = unsafe { self.d_real.as_device_ptr().add(out_off) };
                let out_imag_ptr = unsafe { self.d_imag.as_device_ptr().add(out_off) };
                let out_ang_ptr = unsafe { self.d_ang.as_device_ptr().add(out_off) };
                unsafe {
                    launch!(func_ria<<<grid_chunk, block, smem, stream>>>(
                        self.d_prices.as_device_ptr(),
                        self.d_cos.as_device_ptr(), self.d_sin.as_device_ptr(),
                        self.d_periods.as_device_ptr(),
                        self.d_sumc.as_device_ptr(), self.d_sums.as_device_ptr(),
                        self.d_st2.as_device_ptr(), self.d_st4.as_device_ptr(),
                        self.max_period as i32,
                        self.series_len as i32,
                        self.n_combos as i32,
                        self.first_valid as i32,
                        processed as i32,
                        out_real_ptr, out_imag_ptr, out_ang_ptr
                    ))
                    .unwrap();
                    let func_state = self
                        .cuda
                        .module
                        .get_function("correlation_cycle_state_batch_f32")
                        .unwrap();
                    let out_st_ptr = unsafe { self.d_st.as_device_ptr().add(out_off) };
                    launch!(func_state<<<grid_chunk, block, 0, stream>>>(
                        self.d_ang.as_device_ptr(), self.d_th.as_device_ptr(), self.d_periods.as_device_ptr(),
                        self.series_len as i32, self.n_combos as i32, self.first_valid as i32,
                        processed as i32,
                        out_st_ptr
                    )).unwrap();
                }
                processed += chunk;
            }
            self.cuda.synchronize().unwrap();
        }
    }

    fn prep_batch() -> Box<dyn CudaBenchState> {
        let mut cuda = CudaCorrelationCycle::new(0).expect("cc cuda");
        let price = gen_series(ONE_SERIES_LEN);
        let first_valid = price.iter().position(|x| x.is_finite()).unwrap_or(0);
        let start_p = 16usize;
        let end_p = start_p + PARAM_SWEEP - 1;
        let sweep = CorrelationCycleBatchRange {
            period: (start_p, end_p, 1),
            threshold: (9.0, 9.0, 0.0),
        };
        let combos = CudaCorrelationCycle::expand_grid(&sweep);
        let n_combos = combos.len();
        let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
        let mut periods = vec![0i32; n_combos];
        let mut th = vec![0f32; n_combos];
        let mut sumc = vec![0f32; n_combos];
        let mut sums = vec![0f32; n_combos];
        let mut st2 = vec![0f32; n_combos];
        let mut st4 = vec![0f32; n_combos];
        let mut cos_flat = vec![0f32; n_combos * max_p];
        let mut sin_flat = vec![0f32; n_combos * max_p];
        for (i, prm) in combos.iter().enumerate() {
            let p = prm.period.unwrap();
            let (wc, ws, sc, ss, t2, t4) = CudaCorrelationCycle::compute_trig_weights_and_consts(p);
            periods[i] = p as i32;
            th[i] = prm.threshold.unwrap() as f32;
            let base = i * max_p;
            cos_flat[base..base + p].copy_from_slice(&wc);
            sin_flat[base..base + p].copy_from_slice(&ws);
            sumc[i] = sc;
            sums[i] = ss;
            st2[i] = t2;
            st4[i] = t4;
        }
        let d_prices = unsafe { DeviceBuffer::from_slice_async(&price, &cuda.stream) }.unwrap();
        let d_cos = DeviceBuffer::from_slice(&cos_flat).unwrap();
        let d_sin = DeviceBuffer::from_slice(&sin_flat).unwrap();
        let d_periods = DeviceBuffer::from_slice(&periods).unwrap();
        let d_th = DeviceBuffer::from_slice(&th).unwrap();
        let d_sumc = DeviceBuffer::from_slice(&sumc).unwrap();
        let d_sums = DeviceBuffer::from_slice(&sums).unwrap();
        let d_st2 = DeviceBuffer::from_slice(&st2).unwrap();
        let d_st4 = DeviceBuffer::from_slice(&st4).unwrap();
        let d_real = unsafe { DeviceBuffer::uninitialized(n_combos * ONE_SERIES_LEN) }.unwrap();
        let d_imag = unsafe { DeviceBuffer::uninitialized(n_combos * ONE_SERIES_LEN) }.unwrap();
        let d_ang = unsafe { DeviceBuffer::uninitialized(n_combos * ONE_SERIES_LEN) }.unwrap();
        let d_state = unsafe { DeviceBuffer::uninitialized(n_combos * ONE_SERIES_LEN) }.unwrap();
        Box::new(CcBatchState {
            cuda,
            d_prices,
            d_cos,
            d_sin,
            d_periods,
            d_th,
            d_sumc,
            d_sums,
            d_st2,
            d_st4,
            d_real,
            d_imag,
            d_ang,
            d_st: d_state,
            first_valid,
            max_period: max_p,
            series_len: ONE_SERIES_LEN,
            n_combos,
        })
    }

    struct CcManyState {
        cuda: CudaCorrelationCycle,
        d_prices_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        d_wcos: DeviceBuffer<f32>,
        d_wsin: DeviceBuffer<f32>,
        d_real: DeviceBuffer<f32>,
        d_imag: DeviceBuffer<f32>,
        d_ang: DeviceBuffer<f32>,
        d_st: DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        period: usize,
        threshold: f32,
        sumc: f32,
        sums: f32,
        st2: f32,
        st4: f32,
    }
    impl CudaBenchState for CcManyState {
        fn launch(&mut self) {
            let (grid, block, _cfg) =
                self.cuda
                    .pick_launch_2d(self.cols, self.rows, Some((128, 4)));
            let func_ria = self
                .cuda
                .module
                .get_function("correlation_cycle_many_series_one_param_f32_ria")
                .unwrap();
            let smem = (self.period * 2 * std::mem::size_of::<f32>()) as u32;
            let stream = &self.cuda.stream;
            unsafe {
                launch!(func_ria<<<grid, block, smem, stream>>>(
                    self.d_prices_tm.as_device_ptr(),
                    self.d_wcos.as_device_ptr(), self.d_wsin.as_device_ptr(),
                    self.sumc, self.sums, self.st2, self.st4,
                    self.cols as i32, self.rows as i32, self.period as i32,
                    self.d_first_valids.as_device_ptr(),
                    self.d_real.as_device_ptr(), self.d_imag.as_device_ptr(), self.d_ang.as_device_ptr()
                )).unwrap();
                let func_st = self
                    .cuda
                    .module
                    .get_function("correlation_cycle_state_many_series_one_param_f32")
                    .unwrap();
                launch!(func_st<<<grid, block, 0, stream>>>(
                    self.d_ang.as_device_ptr(), self.threshold,
                    self.d_first_valids.as_device_ptr(),
                    self.cols as i32, self.rows as i32, self.period as i32,
                    self.d_st.as_device_ptr()
                ))
                .unwrap();
            }
            self.cuda.synchronize().unwrap();
        }
    }

    fn prep_many() -> Box<dyn CudaBenchState> {
        let mut cuda = CudaCorrelationCycle::new(0).expect("cc cuda");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let prices_tm = gen_time_major_prices(cols, rows);
        let mut fvs = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = 0usize;
            for t in 0..rows {
                if prices_tm[t * cols + s].is_finite() {
                    fv = t;
                    break;
                }
            }
            fvs[s] = fv as i32;
        }
        let period = 32usize;
        let threshold = 9.0f32;
        let (wcos, wsin, sumc, sums, st2, st4) =
            CudaCorrelationCycle::compute_trig_weights_and_consts(period);
        let d_prices_tm =
            unsafe { DeviceBuffer::from_slice_async(&prices_tm, &cuda.stream) }.unwrap();
        let d_first_valids = DeviceBuffer::from_slice(&fvs).unwrap();
        let d_wcos = DeviceBuffer::from_slice(&wcos).unwrap();
        let d_wsin = DeviceBuffer::from_slice(&wsin).unwrap();
        let d_real = unsafe { DeviceBuffer::uninitialized(cols * rows) }.unwrap();
        let d_imag = unsafe { DeviceBuffer::uninitialized(cols * rows) }.unwrap();
        let d_ang = unsafe { DeviceBuffer::uninitialized(cols * rows) }.unwrap();
        let d_st = unsafe { DeviceBuffer::uninitialized(cols * rows) }.unwrap();
        Box::new(CcManyState {
            cuda,
            d_prices_tm,
            d_first_valids,
            d_wcos,
            d_wsin,
            d_real,
            d_imag,
            d_ang,
            d_st,
            cols,
            rows,
            period,
            threshold,
            sumc,
            sums,
            st2,
            st4,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "correlation_cycle",
                "one_series_many_params",
                "correlation_cycle_cuda_batch_dev",
                "250k_x_64",
                prep_batch,
            ),
            CudaBenchScenario::new(
                "correlation_cycle",
                "many_series_one_param",
                "correlation_cycle_cuda_many_series_one_param_dev",
                "128x500k",
                prep_many,
            ),
        ]
    }
}
