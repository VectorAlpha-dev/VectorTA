//! CUDA scaffolding for the VOSC (Volume Oscillator).
//!
//! Category: Prefix-sum/rational.
//! - Batch (one-series × many-params) precomputes a single f64 prefix sum on host and
//!   evaluates all (short,long) rows on device.
//! - Many-series × one‑param uses time‑major prefix per series with per‑series
//!   first_valid indices.
//!
//! Parity goals with ALMA/CWMA wrappers:
//! - PTX loaded via OUT_DIR with DetermineTargetFromContext + OptLevel O2 (fallbacks)
//! - Non‑blocking stream
//! - Policy enums + light introspection
//! - VRAM checks + grid.y chunking to <= 65_535

#![cfg(feature = "cuda")]

use crate::indicators::vosc::{VoscBatchRange, VoscParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::sync::Arc;
use thiserror::Error;

// Host mirror for CUDA float2 (8-byte aligned). Used for DS-packed prefixes.
#[repr(C, align(8))]
#[derive(Clone, Copy, Default, Debug)]
struct Float2 {
    pub x: f32,
    pub y: f32,
}
// Safety: POD, no drop; matches device float2 layout
unsafe impl cust::memory::DeviceCopy for Float2 {}

#[inline]
fn pack_f64_to_float2_host(src: &[f64]) -> Vec<Float2> {
    let mut dst = Vec::with_capacity(src.len());
    for &d in src {
        let hi = d as f32;
        let lo = (d - hi as f64) as f32;
        dst.push(Float2 { x: hi, y: lo });
    }
    dst
}

#[derive(Debug, Error)]
pub enum CudaVoscError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
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
    #[error("device mismatch: buf={buf}, current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

/// VRAM-backed array handle returned by the VOSC CUDA wrapper.
pub struct DeviceArrayF32 {
    pub buf: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
    pub ctx: Arc<Context>,
    pub device_id: u32,
}
impl DeviceArrayF32 {
    #[inline]
    pub fn device_ptr(&self) -> u64 {
        self.buf.as_device_ptr().as_raw() as u64
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.rows.saturating_mul(self.cols)
    }
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
pub struct CudaVoscPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaVoscPolicy {
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

pub struct CudaVosc {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaVoscPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaVosc {
    pub fn new(device_id: usize) -> Result<Self, CudaVoscError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/vosc_kernel.ptx"));
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
            context,
            device_id: device_id as u32,
            policy: CudaVoscPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn set_policy(&mut self, p: CudaVoscPolicy) { self.policy = p; }
    pub fn policy(&self) -> &CudaVoscPolicy { &self.policy }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> { self.last_batch }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> { self.last_many }
    pub fn context_arc(&self) -> Arc<Context> { self.context.clone() }
    pub fn device_id(&self) -> u32 { self.device_id }
    pub fn synchronize(&self) -> Result<(), CudaVoscError> {
        self.stream.synchronize().map_err(Into::into)
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }
    #[inline]
    fn will_fit(required_bytes: usize, headroom: usize) -> Result<(), CudaVoscError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        match mem_get_info() {
            Ok((free, _total)) => {
                if required_bytes.saturating_add(headroom) <= free {
                    Ok(())
                } else {
                    Err(CudaVoscError::OutOfMemory {
                        required: required_bytes,
                        free,
                        headroom,
                    })
                }
            }
            Err(e) => Err(CudaVoscError::Cuda(e)),
        }
    }

    #[inline]
    fn checked_add(a: usize, b: usize) -> Result<usize, CudaVoscError> {
        a.checked_add(b)
            .ok_or_else(|| CudaVoscError::InvalidInput("size overflow".into()))
    }

    #[inline]
    fn checked_mul(a: usize, b: usize) -> Result<usize, CudaVoscError> {
        a.checked_mul(b)
            .ok_or_else(|| CudaVoscError::InvalidInput("size overflow".into()))
    }

    // --------------- Batch (one-series × many-params) ---------------
    fn expand_combos(range: &VoscBatchRange) -> Vec<VoscParams> {
        fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
            if step == 0 || start == end {
                return vec![start];
            }
            let mut out = Vec::new();
            if start <= end {
                let mut v = start;
                while v <= end {
                    out.push(v);
                    match v.checked_add(step) {
                        Some(next) if next > v => v = next,
                        _ => break,
                    }
                }
            } else {
                let mut v = start;
                loop {
                    out.push(v);
                    if v <= end {
                        break;
                    }
                    match v.checked_sub(step) {
                        Some(next) if next < v => {
                            v = next;
                            if v < end {
                                break;
                            }
                        }
                        _ => break,
                    }
                }
            }
            out
        }
        let shorts = axis_usize(range.short_period);
        let longs = axis_usize(range.long_period);
        let mut out = Vec::with_capacity(shorts.len() * longs.len());
        for &s in &shorts {
            for &l in &longs {
                if s <= l {
                    out.push(VoscParams {
                        short_period: Some(s),
                        long_period: Some(l),
                    });
                }
            }
        }
        out
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &VoscBatchRange,
    ) -> Result<(Vec<VoscParams>, usize, usize), CudaVoscError> {
        if data_f32.is_empty() {
            return Err(CudaVoscError::InvalidInput("empty data".into()));
        }
        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaVoscError::InvalidInput("all values are NaN".into()))?;
        let combos = Self::expand_combos(sweep);
        if combos.is_empty() {
            return Err(CudaVoscError::InvalidInput("no parameter combinations".into()));
        }
        for c in &combos {
            let s = c.short_period.unwrap_or(0);
            let l = c.long_period.unwrap_or(0);
            if s == 0 || l == 0 || s > l || l > len {
                return Err(CudaVoscError::InvalidInput(format!(
                    "invalid (s,l)=({},{}) for len {}",
                    s, l, len
                )));
            }
            if len - first_valid < l {
                return Err(CudaVoscError::InvalidInput(format!(
                    "not enough valid data: need {}, have {} after first_valid {}",
                    l,
                    len - first_valid,
                    first_valid
                )));
            }
        }
        Ok((combos, first_valid, len))
    }

    fn build_prefix_sum_f64_allow_nan(data: &[f32]) -> Vec<f64> {
        let len = data.len();
        let mut prefix = vec![0.0f64; len + 1];
        let mut acc = 0.0f64;
        for i in 0..len {
            // Match CPU batch prefix behavior: accumulate raw values (NaN poisons subsequent entries)
            acc += data[i] as f64;
            prefix[i + 1] = acc;
        }
        prefix
    }

    fn launch_batch_kernel(
        &self,
        d_prefix: &DeviceBuffer<Float2>,
        d_short: &DeviceBuffer<i32>,
        d_long: &DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVoscError> {
        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Auto => 256,
            BatchKernelPolicy::Plain { block_x } => block_x.max(64),
        };
        let func = self
            .module
            .get_function("vosc_batch_prefix_f32_ds")
            .map_err(|_| CudaVoscError::MissingKernelSymbol {
                name: "vosc_batch_prefix_f32_ds",
            })?;
        unsafe {
            (*(self as *const _ as *mut CudaVosc)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        unsafe {
            (*(self as *const _ as *mut CudaVosc)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        if env::var("BENCH_DEBUG").ok().as_deref() == Some("1") && !self.debug_batch_logged {
            eprintln!("[DEBUG] VOSC batch selected kernel: Plain({})", block_x);
            unsafe {
                (*(self as *const _ as *mut CudaVosc)).debug_batch_logged = true;
            }
            unsafe {
                (*(self as *const _ as *mut CudaVosc)).debug_batch_logged = true;
            }
        }

        let grid_x = ((len as u32).saturating_add(block_x - 1)) / block_x;
        const MAX_GRID_Y: usize = 65_535;
        let block: BlockSize = (block_x, 1, 1).into();
        let mut start = 0usize;
        while start < n_combos {
            let chunk = (n_combos - start).min(MAX_GRID_Y);
            let grid: GridSize = (grid_x.max(1), chunk as u32, 1).into();
            unsafe {
                let mut p_ptr = d_prefix.as_device_ptr().as_raw();
                let mut len_i = len as i32;
                let mut first_i = first_valid as i32;
                let offset_i_bytes =
                    match Self::checked_mul(start, std::mem::size_of::<i32>()) {
                        Ok(v) => v as u64,
                        Err(e) => return Err(e),
                    };
                let mut s_ptr = d_short
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add(offset_i_bytes);
                let mut l_ptr = d_long
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add(offset_i_bytes);
                let mut n_i = chunk as i32;
                let offset_elems = match Self::checked_mul(start, len) {
                    Ok(v) => v,
                    Err(e) => return Err(e),
                };
                let offset_f_bytes =
                    match Self::checked_mul(offset_elems, std::mem::size_of::<f32>()) {
                        Ok(v) => v as u64,
                        Err(e) => return Err(e),
                    };
                let mut out_ptr = d_out
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add(offset_f_bytes);
                let args: &mut [*mut c_void] = &mut [
                    &mut p_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut s_ptr as *mut _ as *mut c_void,
                    &mut l_ptr as *mut _ as *mut c_void,
                    &mut n_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(CudaVoscError::Cuda)?;
            }
            start += chunk;
        }
        Ok(())
    }

    pub fn vosc_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &VoscBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<VoscParams>), CudaVoscError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = combos.len();
        let prefix = Self::build_prefix_sum_f64_allow_nan(data_f32);
        let prefix_f2 = pack_f64_to_float2_host(&prefix);
        let shorts: Vec<i32> = combos
            .iter()
            .map(|c| c.short_period.unwrap() as i32)
            .collect();
        let longs: Vec<i32> = combos
            .iter()
            .map(|c| c.long_period.unwrap() as i32)
            .collect();

        // VRAM estimate (+64MB headroom)
        let prefix_bytes = Self::checked_mul(len + 1, std::mem::size_of::<Float2>())?;
        let params_len = shorts.len().saturating_add(longs.len());
        let params_bytes = Self::checked_mul(params_len, std::mem::size_of::<i32>())?;
        let out_elems = Self::checked_mul(n_combos, len)?;
        let out_bytes = Self::checked_mul(out_elems, std::mem::size_of::<f32>())?;
        let tmp = Self::checked_add(prefix_bytes, params_bytes)?;
        let bytes = Self::checked_add(tmp, out_bytes)?;
        let headroom = 64 * 1024 * 1024usize;
        Self::will_fit(bytes, headroom)?;

        // Async copies for better overlap
        let d_prefix = unsafe { DeviceBuffer::from_slice_async(&prefix_f2, &self.stream) }?;
        let d_short = unsafe { DeviceBuffer::from_slice_async(&shorts, &self.stream) }?;
        let d_long = unsafe { DeviceBuffer::from_slice_async(&longs, &self.stream) }?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_elems, &self.stream) }?;

        self.launch_batch_kernel(
            &d_prefix,
            &d_short,
            &d_long,
            len,
            first_valid,
            n_combos,
            &mut d_out,
        )?;
        self.stream.synchronize()?;

        Ok((
            DeviceArrayF32 {
                buf: d_out,
                rows: n_combos,
                cols: len,
                ctx: self.context_arc(),
                device_id: self.device_id,
            },
            combos,
        ))
    }

    // --------------- Many-series × one‑param (time‑major) ---------------
    fn prepare_many_series_inputs_f64(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        short: usize,
        long: usize,
    ) -> Result<(Vec<f64>, Vec<i32>), CudaVoscError> {
        let elems = Self::checked_mul(cols, rows)?;
        if data_tm_f32.len() != elems {
            return Err(CudaVoscError::InvalidInput("shape mismatch".into()));
        }
        if short == 0 || long == 0 || short > long || long > rows {
            return Err(CudaVoscError::InvalidInput("invalid periods".into()));
        }
        // Per‑series first_valid and prefix_tm ((rows+1) x cols, TM)
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = -1i32;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + s];
                if !v.is_nan() {
                    fv = t as i32;
                    break;
                }
            }
            for t in 0..rows {
                let v = data_tm_f32[t * cols + s];
                if !v.is_nan() {
                    fv = t as i32;
                    break;
                }
            }
            first_valids[s] = fv;
            if fv >= 0 && (rows as i32 - fv) < long as i32 {
                return Err(CudaVoscError::InvalidInput(
                    "not enough valid data per series".into(),
                ));
                return Err(CudaVoscError::InvalidInput(
                    "not enough valid data per series".into(),
                ));
            }
        }
        let prefix_len = Self::checked_mul(rows + 1, cols)?;
        let mut prefix_tm = vec![0.0f64; prefix_len];
        for s in 0..cols {
            let stride = cols;
            let mut acc = 0.0f64;
            for t in 0..rows {
                let v = data_tm_f32[t * stride + s];
                if !v.is_nan() {
                    acc += v as f64;
                }
                prefix_tm[(t + 1) * stride + s] = acc;
            }
        }
        Ok((prefix_tm, first_valids))
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        short: usize,
        long: usize,
    ) -> Result<(Vec<Float2>, Vec<i32>), CudaVoscError> {
        let elems = Self::checked_mul(cols, rows)?;
        if data_tm_f32.len() != elems {
            return Err(CudaVoscError::InvalidInput("shape mismatch".into()));
        }
        if short == 0 || long == 0 || short > long || long > rows {
            return Err(CudaVoscError::InvalidInput("invalid periods".into()));
        }
        // Per‑series first_valid and prefix_tm ((rows+1) x cols, TM)
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = -1i32;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + s];
                if !v.is_nan() {
                    fv = t as i32;
                    break;
                }
            }
            first_valids[s] = fv;
            if fv >= 0 && (rows as i32 - fv) < long as i32 {
                return Err(CudaVoscError::InvalidInput(
                    "not enough valid data per series".into(),
                ));
            }
        }
        let prefix_len = Self::checked_mul(rows + 1, cols)?;
        let mut prefix_tm = vec![Float2 { x: 0.0, y: 0.0 }; prefix_len];
        for s in 0..cols {
            let stride = cols;
            let mut acc = 0.0f64;
            // prefix[0] = 0 already
            for t in 0..rows {
                let v = data_tm_f32[t * stride + s];
                if !v.is_nan() {
                    acc += v as f64;
                }
                let hi = acc as f32;
                let lo = (acc - hi as f64) as f32;
                prefix_tm[(t + 1) * stride + s] = Float2 { x: hi, y: lo };
            }
        }
        Ok((prefix_tm, first_valids))
    }

    fn launch_many_series_kernel_f64(
        &self,
        d_prefix_tm: &DeviceBuffer<f64>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        short: usize,
        long: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVoscError> {
        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 256,
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
        };
        let func = self
            .module
            .get_function("vosc_many_series_one_param_f32")
            .map_err(|_| CudaVoscError::MissingKernelSymbol {
                name: "vosc_many_series_one_param_f32",
            })?;

        unsafe {
            (*(self as *const _ as *mut CudaVosc)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        unsafe {
            (*(self as *const _ as *mut CudaVosc)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        if env::var("BENCH_DEBUG").ok().as_deref() == Some("1") && !self.debug_many_logged {
            eprintln!(
                "[DEBUG] VOSC many-series selected kernel: OneD({})",
                block_x
            );
            unsafe {
                (*(self as *const _ as *mut CudaVosc)).debug_many_logged = true;
            }
            eprintln!(
                "[DEBUG] VOSC many-series selected kernel: OneD({})",
                block_x
            );
            unsafe {
                (*(self as *const _ as *mut CudaVosc)).debug_many_logged = true;
            }
        }
        // Original mapping: grid.y over series, threads over time
        let grid_x = ((rows as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), cols as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut p_ptr = d_prefix_tm.as_device_ptr().as_raw();
            let mut s_i = short as i32;
            let mut l_i = long as i32;
            let mut num_series_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut fv_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_ptr as *mut _ as *mut c_void,
                &mut s_i as *mut _ as *mut c_void,
                &mut l_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(CudaVoscError::Cuda)?;
        }
        Ok(())
    }

    fn launch_many_series_kernel_ds(
        &self,
        d_prefix_tm: &DeviceBuffer<Float2>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        short: usize,
        long: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
        ) -> Result<(), CudaVoscError> {
        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 128,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(64),
        };
        let func = self
            .module
            .get_function("vosc_many_series_one_param_f32_ds_tm_coalesced")
            .map_err(|_| CudaVoscError::MissingKernelSymbol {
                name: "vosc_many_series_one_param_f32_ds_tm_coalesced",
            })?;

        unsafe {
            (*(self as *const _ as *mut CudaVosc)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        if env::var("BENCH_DEBUG").ok().as_deref() == Some("1") && !self.debug_many_logged {
            eprintln!(
                "[DEBUG] VOSC many-series selected kernel: OneD({})",
                block_x
            );
            unsafe {
                (*(self as *const _ as *mut CudaVosc)).debug_many_logged = true;
            }
        }

        // Coalesced across series; chunk time by grid.y limit
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        const MAX_GRID_Y: usize = 65_535;
        let block: BlockSize = (block_x, 1, 1).into();
        let mut row_base = 0usize;
        while row_base < rows {
            let rows_left = rows - row_base;
            let chunk_y = rows_left.min(MAX_GRID_Y);
            let grid: GridSize = (grid_x.max(1), chunk_y as u32, 1).into();
            unsafe {
                let mut p_ptr = d_prefix_tm.as_device_ptr().as_raw();
                let mut s_i = short as i32;
                let mut l_i = long as i32;
                let mut num_series_i = cols as i32;
                let mut rows_i = rows as i32;
                let mut fv_ptr = d_first_valids.as_device_ptr().as_raw();
                let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
                let mut row_base_i = row_base as i32;
                let args: &mut [*mut c_void] = &mut [
                    &mut p_ptr as *mut _ as *mut c_void,
                    &mut s_i as *mut _ as *mut c_void,
                    &mut l_i as *mut _ as *mut c_void,
                    &mut num_series_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut fv_ptr as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                    &mut row_base_i as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(CudaVoscError::Cuda)?;
            }
            row_base += chunk_y;
        }
        Ok(())
    }

    pub fn vosc_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &VoscParams,
    ) -> Result<DeviceArrayF32, CudaVoscError> {
        let short = params.short_period.unwrap_or(2);
        let long = params.long_period.unwrap_or(5);

        // Heuristic: prefer f64 original kernel for small/medium shapes; DS/coalesced for huge
        let use_ds = rows >= 131_072 || cols >= 1024;

        let elems = Self::checked_mul(rows, cols)?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(elems, &self.stream)?
        };

        if use_ds {
            let (prefix_tm, first_valids) =
                Self::prepare_many_series_inputs(data_tm_f32, cols, rows, short, long)?;
            let prefix_bytes =
                Self::checked_mul(rows + 1, cols)?
                    .checked_mul(std::mem::size_of::<Float2>())
                    .ok_or_else(|| CudaVoscError::InvalidInput("size overflow".into()))?;
            let fv_bytes =
                Self::checked_mul(cols, std::mem::size_of::<i32>())?;
            let out_bytes =
                Self::checked_mul(elems, std::mem::size_of::<f32>())?;
            let tmp = Self::checked_add(prefix_bytes, fv_bytes)?;
            let bytes = Self::checked_add(tmp, out_bytes)?;
            Self::will_fit(bytes, 64 * 1024 * 1024)?;
            let d_prefix_tm = unsafe { DeviceBuffer::from_slice_async(&prefix_tm, &self.stream) }?;
            let d_first =
                unsafe { DeviceBuffer::from_slice_async(&first_valids, &self.stream) }?;
            self.launch_many_series_kernel_ds(
                &d_prefix_tm,
                &d_first,
                cols,
                rows,
                short,
                long,
                &mut d_out_tm,
            )?;
        } else {
            // CPU fallback for small shapes (exact parity with scalar path)
            use crate::indicators::vosc::{vosc_with_kernel, VoscData, VoscInput};
            use crate::utilities::enums::Kernel;

            let params_cpu = VoscParams {
                short_period: Some(short),
                long_period: Some(long),
            };
            let mut host_out = vec![0f32; elems];
            for s in 0..cols {
                let mut series = vec![f64::NAN; rows];
                for t in 0..rows {
                    series[t] = data_tm_f32[t * cols + s] as f64;
                }
                let input = VoscInput {
                    data: VoscData::Slice(&series),
                    params: params_cpu.clone(),
                };
                let out = vosc_with_kernel(&input, Kernel::Scalar)
                    .map_err(|e| CudaVoscError::InvalidInput(e.to_string()))?;
                for t in 0..rows {
                    host_out[t * cols + s] = out.values[t] as f32;
                }
            }
            // Zero-copy allocation on device; then async copy host results into it
            d_out_tm =
                unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }?;
            unsafe { d_out_tm.async_copy_from(host_out.as_slice(), &self.stream) }?;
        }
        self.stream.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows,
            cols,
            ctx: self.context_arc(),
            device_id: self.device_id,
        })
    }
}

// ---------------- Benches registration (optional) ----------------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250; // typical batch sweep

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let prefix_bytes = (ONE_SERIES_LEN + 1) * std::mem::size_of::<super::Float2>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + prefix_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct VoscBatchState {
        cuda: CudaVosc,
        d_prefix: DeviceBuffer<super::Float2>,
        d_short: DeviceBuffer<i32>,
        d_long: DeviceBuffer<i32>,
        d_out: DeviceBuffer<f32>,
        len: usize,
        n_combos: usize,
        first_valid: usize,
    }
    impl CudaBenchState for VoscBatchState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_prefix,
                    &self.d_short,
                    &self.d_long,
                    self.len,
                    self.first_valid,
                    self.n_combos,
                    &mut self.d_out,
                )
                .expect("vosc launch");
            self.cuda.synchronize().expect("sync");
        }
    }

    fn prep_vosc_batch() -> VoscBatchState {
        let mut cuda = CudaVosc::new(0).expect("CudaVosc");
        cuda.set_policy(CudaVoscPolicy {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        });
        cuda.set_policy(CudaVoscPolicy {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        });

        let len = ONE_SERIES_LEN;
        let mut x = vec![f32::NAN; len];
        // prepared above
        for i in 10..len {
            let f = i as f32;
            x[i] = (f * 0.0007).cos() + 0.03 * (f * 0.0011).sin();
        }
        let (combos, first_valid, _len) = CudaVosc::prepare_batch_inputs(
            &x,
            &VoscBatchRange {
                short_period: (5, 5, 0),
                long_period: (10, 10 + PARAM_SWEEP - 1, 1),
            },
        )
        .expect("prep");
        let shorts: Vec<i32> = combos
            .iter()
            .map(|c| c.short_period.unwrap() as i32)
            .collect();
        let longs: Vec<i32> = combos
            .iter()
            .map(|c| c.long_period.unwrap() as i32)
            .collect();
        let prefix = CudaVosc::build_prefix_sum_f64_allow_nan(&x);
        let prefix_f2 = super::pack_f64_to_float2_host(&prefix);
        let d_prefix = DeviceBuffer::from_slice(&prefix_f2).expect("d_prefix");
        let d_short = DeviceBuffer::from_slice(&shorts).expect("d_short");
        let d_long = DeviceBuffer::from_slice(&longs).expect("d_long");
        // construct state once
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(combos.len() * len) }.expect("d_out");
        VoscBatchState {
            cuda,
            d_prefix,
            d_short,
            d_long,
            d_out,
            len,
            n_combos: combos.len(),
            first_valid,
        }
    }
    fn prep_vosc_batch_box() -> Box<dyn CudaBenchState> { Box::new(prep_vosc_batch()) }

    struct VoscManyState {
        cuda: CudaVosc,
        d_prefix_tm: DeviceBuffer<super::Float2>,
        d_first_valids: DeviceBuffer<i32>,
        d_out_tm: DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        short: usize,
        long: usize,
    }
    impl CudaBenchState for VoscManyState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series_kernel_ds(
                    &self.d_prefix_tm,
                    &self.d_first_valids,
                    self.cols,
                    self.rows,
                    self.short,
                    self.long,
                    &mut self.d_out_tm,
                )
                .expect("vosc many launch");
            self.cuda.synchronize().expect("sync");
        }
    }

    fn prep_vosc_many() -> VoscManyState {
        let mut cuda = CudaVosc::new(0).expect("CudaVosc");
        cuda.set_policy(CudaVoscPolicy {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        });
        let cols = 256usize;
        let rows = 1_000_000usize;
        cuda.set_policy(CudaVoscPolicy {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        });
        let cols = 256usize;
        let rows = 1_000_000usize;
        let tm = gen_time_major_prices(cols, rows);
        let (prefix_tm, first_valids) =
            CudaVosc::prepare_many_series_inputs(&tm, cols, rows, 5, 34).expect("prep");
        let d_prefix_tm = DeviceBuffer::from_slice(&prefix_tm).expect("d_prefix_tm");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out");
        VoscManyState {
            cuda,
            d_prefix_tm,
            d_first_valids,
            d_out_tm,
            cols,
            rows,
            short: 5,
            long: 34,
        }
    }
    fn prep_vosc_many_box() -> Box<dyn CudaBenchState> { Box::new(prep_vosc_many()) }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "vosc",
                "batch_dev",
                "vosc_cuda_batch_dev",
                "1m_x_250",
                prep_vosc_batch_box,
            )
            .with_mem_required(bytes_one_series_many_params())
            .with_inner_iters(4),
            CudaBenchScenario::new(
                "vosc",
                "many_series_one_param",
                "vosc_cuda_many_series_one_param",
                "256x1m",
                prep_vosc_many_box,
            )
            .with_inner_iters(4),
        ]
    }
}
