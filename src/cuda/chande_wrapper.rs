//! CUDA wrapper for Chande (Chandelier Exit) kernels.
//!
//! Parity targets:
//! - ALMA-like API returning `DeviceArrayF32` VRAM handles.
//! - Batch (one-series × many-params) and Many-series × one-param (time-major).
//! - Warmup/NaN identical to scalar: warm = first_valid + period - 1.
//! - Simple VRAM estimate + chunked launches (grid.x) to respect limits.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::chande::ChandeBatchRange;
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
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CudaChandeError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: isize, end: isize, step: isize },
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
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
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaChandePolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaChandePolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

pub struct CudaChande {
    module: Module,
    stream: Stream,
    context: std::sync::Arc<Context>,
    device_id: u32,
    policy: CudaChandePolicy,

    // Persistent monotone-deque workspaces for one-series × many-params path
    dq_idx: Option<DeviceBuffer<i32>>,
    dq_val: Option<DeviceBuffer<f32>>,
    dq_combo_cap: usize,
    dq_cap: usize,
}

impl CudaChande {
    pub fn new(device_id: usize) -> Result<Self, CudaChandeError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = std::sync::Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/chande_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
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
            policy: CudaChandePolicy::default(),
            dq_idx: None,
            dq_val: None,
            dq_combo_cap: 0,
            dq_cap: 0,
        })
    }

    pub fn set_policy(&mut self, policy: CudaChandePolicy) {
        self.policy = policy;
    }
    pub fn synchronize(&self) -> Result<(), CudaChandeError> { Ok(self.stream.synchronize()?) }

    #[inline]
    pub fn context_arc(&self) -> std::sync::Arc<Context> { self.context.clone() }
    #[inline]
    pub fn device_id(&self) -> u32 { self.device_id }

    fn first_valid_hlc(high: &[f32], low: &[f32], close: &[f32]) -> Result<usize, CudaChandeError> {
        if high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaChandeError::InvalidInput("empty input".into()));
        }
        let n = high.len().min(low.len()).min(close.len());
        for i in 0..n {
            if !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan() {
                return Ok(i);
            }
            if !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan() {
                return Ok(i);
            }
        }
        Err(CudaChandeError::InvalidInput("all values are NaN".into()))
    }

    fn axis_usize_range(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, CudaChandeError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let vals: Vec<usize> = (start..=end).step_by(step).collect();
            if vals.is_empty() {
                return Err(CudaChandeError::InvalidRange {
                    start: start as isize,
                    end: end as isize,
                    step: step as isize,
                });
            }
            return Ok(vals);
        }
        let mut vals = Vec::new();
        let s = step.max(1);
        let mut cur = start;
        while cur >= end {
            vals.push(cur);
            if cur < s {
                break;
            }
            cur -= s;
            if cur == usize::MAX {
                break;
            }
        }
        if vals.is_empty() {
            return Err(CudaChandeError::InvalidRange {
                start: start as isize,
                end: end as isize,
                step: step as isize,
            });
        }
        Ok(vals)
    }

    fn axis_f64_range(
        (start, end, step): (f64, f64, f64),
    ) -> Result<Vec<f32>, CudaChandeError> {
        if step.abs() < f64::EPSILON || (start - end).abs() < f64::EPSILON {
            return Ok(vec![start as f32]);
        }
        let mut v = Vec::new();
        if start < end {
            let mut x = start;
            while x <= end + 1e-12 {
                v.push(x as f32);
                x += step;
            }
        } else {
            let mut x = start;
            let st = -step.abs();
            while x >= end - 1e-12 {
                v.push(x as f32);
                x += st;
            }
        }
        if v.is_empty() {
            return Err(CudaChandeError::InvalidRange {
                start: start as isize,
                end: end as isize,
                step: step as isize,
            });
        }
        Ok(v)
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaChandeError> {
        let check = match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        };
        if !check {
            return Ok(());
        }
        if let Ok((free, _)) = mem_get_info() {
            if required_bytes.saturating_add(headroom_bytes) <= free {
                Ok(())
            } else {
                Err(CudaChandeError::OutOfMemory { required: required_bytes, free, headroom: headroom_bytes })
            }
        } else {
            Ok(())
        }
    }

    #[inline]
    fn next_pow2_usize(x: usize) -> usize { (x.max(1)).next_power_of_two() }

    fn ensure_workspace(&mut self, combos: usize, queue_cap: usize) -> Result<(), CudaChandeError> {
        if self.dq_idx.is_some() && self.dq_combo_cap >= combos && self.dq_cap >= queue_cap {
            return Ok(());
        }
        let need = combos
            .checked_mul(queue_cap)
            .ok_or_else(|| CudaChandeError::InvalidInput("dq size overflow".into()))?;
        self.dq_idx = Some(DeviceBuffer::<i32>::zeroed(need)?);
        self.dq_val = Some(DeviceBuffer::<f32>::zeroed(need)?);
        self.dq_combo_cap = combos;
        self.dq_cap = queue_cap;
        Ok(())
    }

    fn will_fit_full_output_one_series(
        &self,
        n_combos: usize,
        len: usize,
        queue_cap: usize,
    ) -> Result<(), CudaChandeError> {
        let elt_f32 = std::mem::size_of::<f32>();
        let elt_i32 = std::mem::size_of::<i32>();

        let in_bytes = 3usize
            .checked_mul(len)
            .and_then(|n| n.checked_mul(elt_f32))
            .ok_or_else(|| CudaChandeError::InvalidInput("size overflow (inputs)".into()))?;
        let params_bytes = n_combos
            .checked_mul(3 * elt_i32 + 2 * elt_f32)
            .ok_or_else(|| CudaChandeError::InvalidInput("size overflow (params)".into()))?;
        let out_bytes = n_combos
            .checked_mul(len)
            .and_then(|n| n.checked_mul(elt_f32))
            .ok_or_else(|| CudaChandeError::InvalidInput("size overflow (outputs)".into()))?;
        let dq_bytes = n_combos
            .checked_mul(queue_cap)
            .and_then(|n| n.checked_mul(elt_i32 + elt_f32))
            .ok_or_else(|| CudaChandeError::InvalidInput("size overflow (workspace)".into()))?;

        let headroom = 64usize * 1024 * 1024;
        let need = in_bytes
            .checked_add(params_bytes)
            .and_then(|n| n.checked_add(out_bytes))
            .and_then(|n| n.checked_add(dq_bytes))
            .ok_or_else(|| CudaChandeError::InvalidInput("size overflow (total)".into()))?;
        Self::will_fit(need, headroom)
    }

    fn chunk_size_for_batch(n_combos: usize, len: usize) -> usize {
        // Inputs: 3×len f32; params: periods/mults/dirs/alphas/warm per combo; outputs: combos×len f32.
        let in_bytes = 3usize
            .checked_mul(len)
            .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
            .unwrap_or(usize::MAX);
        let params_bytes = n_combos
            .checked_mul(std::mem::size_of::<i32>() * 3 + std::mem::size_of::<f32>() * 2)
            .unwrap_or(usize::MAX);
        let out_per_combo = len
            .checked_mul(std::mem::size_of::<f32>())
            .unwrap_or(usize::MAX);
        let headroom = 64 * 1024 * 1024;
        let mut chunk = n_combos.max(1);
        while chunk > 1 {
            let need = in_bytes
                .saturating_add(params_bytes)
                .saturating_add(chunk.saturating_mul(out_per_combo))
                .saturating_add(headroom);
            if Self::will_fit(need, 0).is_ok() {
                break;
            }
            chunk = (chunk + 1) / 2;
        }
        chunk.max(1)
    }

    pub fn chande_batch_dev(
        &mut self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &ChandeBatchRange,
        direction: &str,
    ) -> Result<DeviceArrayF32, CudaChandeError> {
        if high.len() != low.len() || low.len() != close.len() {
            return Err(CudaChandeError::InvalidInput("input length mismatch".into()));
        }
        let len = high.len();
        let first_valid = Self::first_valid_hlc(high, low, close)?;
        // Expand params on host (robust range rules: zero step => static, reversed bounds supported)
        let (ps, pe, pst) = sweep.period;
        let (ms, me, mst) = sweep.mult;
        if !(direction.eq_ignore_ascii_case("long") || direction.eq_ignore_ascii_case("short")) {
            return Err(CudaChandeError::InvalidInput(
                "direction must be 'long' or 'short'".into(),
            ));
        }
        let dir_flag = if direction.eq_ignore_ascii_case("long") { 1i32 } else { 0i32 };
        let periods = Self::axis_usize_range((ps, pe, pst))?;
        let mults_host = Self::axis_f64_range((ms, me, mst))?;
        let mut h_periods = Vec::<i32>::new();
        let mut h_alphas = Vec::<f32>::new();
        let mut h_warms = Vec::<i32>::new();
        let mut h_mults = Vec::<f32>::new();
        let mut h_dirs  = Vec::<i32>::new();
        let mut max_p = 0usize;
        for &p in &periods {
            if p == 0 || p > len || (len - first_valid) < p {
                return Err(CudaChandeError::InvalidInput(format!(
                    "invalid period {} for data length {} (valid after {}: {})",
                    p,
                    len,
                    first_valid,
                    len - first_valid
                )));
            }
            if p > max_p { max_p = p; }
            for &m in &mults_host {
                h_periods.push(p as i32);
                h_alphas.push(1.0f32 / (p as f32));
                h_warms.push((first_valid + p - 1) as i32);
                h_mults.push(m);
                h_dirs.push(dir_flag);
            }
        }
        let n_combos = h_periods.len();

        // Device buffers (inputs & params)
        let d_high  = DeviceBuffer::from_slice(high)?;
        let d_low   = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_periods = DeviceBuffer::from_slice(&h_periods)?;
        let d_mults   = DeviceBuffer::from_slice(&h_mults)?;
        let d_dirs    = DeviceBuffer::from_slice(&h_dirs)?;
        let d_alphas  = DeviceBuffer::from_slice(&h_alphas)?;

        // Check for new kernels and compute queue_cap for deque buffers
        let have_oneseries = self.module.get_function("chande_one_series_many_params_f32").is_ok();
        let have_oneseries_tr = self.module.get_function("chande_one_series_many_params_from_tr_f32").is_ok();
        let queue_cap = Self::next_pow2_usize(max_p + 1);

        // VRAM checks and workspace ensure
        if have_oneseries || have_oneseries_tr {
            self.will_fit_full_output_one_series(n_combos, len, queue_cap)?;
            self.ensure_workspace(n_combos, queue_cap)?;
        } else {
            let in_bytes = 3usize
                .checked_mul(len).and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
                .ok_or_else(|| CudaChandeError::InvalidInput("size overflow (inputs)".into()))?;
            let params_bytes = n_combos
                .checked_mul(3 * std::mem::size_of::<i32>() + 2 * std::mem::size_of::<f32>())
                .ok_or_else(|| CudaChandeError::InvalidInput("size overflow (params)".into()))?;
            let out_bytes = n_combos
                .checked_mul(len)
                .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
                .ok_or_else(|| CudaChandeError::InvalidInput("size overflow (outputs)".into()))?;
            let headroom = 64 * 1024 * 1024;
            let need = in_bytes
                .checked_add(params_bytes)
                .and_then(|n| n.checked_add(out_bytes))
                .ok_or_else(|| CudaChandeError::InvalidInput("size overflow (total)".into()))?;
            Self::will_fit(need, headroom)?;
        }

        let total = n_combos
            .checked_mul(len)
            .ok_or_else(|| CudaChandeError::InvalidInput("n_combos*len overflow".into()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total) }?;

        if have_oneseries {
            // Fast path: on-the-fly TR, no extra copy
            let func = self
                .module
                .get_function("chande_one_series_many_params_f32")
                .map_err(|_| CudaChandeError::MissingKernelSymbol { name: "chande_one_series_many_params_f32" })?;
            let warps_needed = ((n_combos + 31) / 32) as u32;
            let warps_per_block = match self.policy.batch {
                BatchKernelPolicy::Plain { block_x } => (block_x.max(32) / 32),
                BatchKernelPolicy::Auto => 4,
            }
            .max(1);
            let block_x = warps_per_block * 32;
            let grid_x = ((warps_needed + warps_per_block - 1) / warps_per_block).max(1);
            if block_x > 1024 { return Err(CudaChandeError::LaunchConfigTooLarge { gx: grid_x, gy: 1, gz: 1, bx: block_x, by: 1, bz: 1 }); }
            let grid: GridSize = (grid_x, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();

            unsafe {
                let mut high_ptr   = d_high.as_device_ptr().as_raw();
                let mut low_ptr    = d_low.as_device_ptr().as_raw();
                let mut close_ptr  = d_close.as_device_ptr().as_raw();
                let mut periods_ptr= d_periods.as_device_ptr().as_raw();
                let mut mults_ptr  = d_mults.as_device_ptr().as_raw();
                let mut dirs_ptr   = d_dirs.as_device_ptr().as_raw();
                let mut alphas_ptr = d_alphas.as_device_ptr().as_raw();
                let mut first_i    = first_valid as i32;
                let mut len_i      = len as i32;
                let mut combos_i   = n_combos as i32;
                let mut qcap_i     = queue_cap as i32;
                let dq_idx_ref = self.dq_idx.as_ref().unwrap();
                let dq_val_ref = self.dq_val.as_ref().unwrap();
                let mut dq_idx_ptr = dq_idx_ref.as_device_ptr().as_raw();
                let mut dq_val_ptr = dq_val_ref.as_device_ptr().as_raw();
                let mut out_ptr   = d_out.as_device_ptr().as_raw();

                let args: &mut [*mut c_void] = &mut [
                    &mut high_ptr as *mut _ as *mut c_void,
                    &mut low_ptr as *mut _ as *mut c_void,
                    &mut close_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut mults_ptr as *mut _ as *mut c_void,
                    &mut dirs_ptr as *mut _ as *mut c_void,
                    &mut alphas_ptr as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut combos_i as *mut _ as *mut c_void,
                    &mut qcap_i as *mut _ as *mut c_void,
                    &mut dq_idx_ptr as *mut _ as *mut c_void,
                    &mut dq_val_ptr as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, 0, args)?;
            }
        } else {
            // Fallback to legacy batch implementation
            let use_tr = self.module.get_function("chande_batch_from_tr_f32").is_ok();
            let func = if use_tr {
                self.module.get_function("chande_batch_from_tr_f32").unwrap()
            } else {
                self.module
                    .get_function("chande_batch_f32")
                    .map_err(|_| CudaChandeError::MissingKernelSymbol { name: "chande_batch_f32" })?
            };

            let d_tr: Option<DeviceBuffer<f32>> = if use_tr {
                let mut tr = vec![0f32; len];
                let mut prev_c = close[first_valid];
                for t in first_valid..len {
                    let hi = high[t];
                    let lo = low[t];
                    if t == first_valid {
                        tr[t] = hi - lo;
                    } else {
                        let mut tri = hi - lo;
                        let hc = (hi - prev_c).abs();
                        if hc > tri { tri = hc; }
                        let lc = (lo - prev_c).abs();
                        if lc > tri { tri = lc; }
                        tr[t] = tri;
                    }
                    prev_c = close[t];
                }
                Some(DeviceBuffer::from_slice(&tr)?)
            } else { None };

            // No VRAM-driven chunking anymore; we validated full output.
            let block_x = match self.policy.batch { BatchKernelPolicy::Plain { block_x } => block_x, BatchKernelPolicy::Auto => 256 };
            let chunk = n_combos.max(1);
            let mut launched = 0usize;
            // Upload warms per chunk to simplify pointer arithmetic
            while launched < n_combos {
                let cur = (n_combos - launched).min(chunk);
                let grid: GridSize = (cur as u32, 1, 1).into();
                let block: BlockSize = (block_x, 1, 1).into();
                unsafe {
                    let mut len_i   = len as i32;
                    let mut first_i = first_valid as i32;
                    let mut cur_i   = cur as i32;
                    let mut out_ptr = d_out.as_device_ptr().as_raw()
                        .wrapping_add(((launched * len) * std::mem::size_of::<f32>()) as u64);

                    if use_tr {
                        let d_tr_ref = d_tr.as_ref().unwrap();
                        let mut high_ptr = d_high.as_device_ptr().as_raw();
                        let mut low_ptr  = d_low.as_device_ptr().as_raw();
                        let mut tr_ptr   = d_tr_ref.as_device_ptr().as_raw();
                        let mut periods_ptr = d_periods.as_device_ptr().as_raw()
                            .wrapping_add((launched * std::mem::size_of::<i32>()) as u64);
                        let mut mults_ptr   = d_mults.as_device_ptr().as_raw()
                            .wrapping_add((launched * std::mem::size_of::<f32>()) as u64);
                        let mut dirs_ptr    = d_dirs.as_device_ptr().as_raw()
                            .wrapping_add((launched * std::mem::size_of::<i32>()) as u64);
                        let mut alphas_ptr  = d_alphas.as_device_ptr().as_raw()
                            .wrapping_add((launched * std::mem::size_of::<f32>()) as u64);
                        let warms_slice = &h_warms[launched..(launched + cur)];
                        let d_warms   = DeviceBuffer::from_slice(warms_slice)?;
                        let mut warms_ptr = d_warms.as_device_ptr().as_raw();

                        let args: &mut [*mut c_void] = &mut [
                            &mut high_ptr as *mut _ as *mut c_void,
                            &mut low_ptr as *mut _ as *mut c_void,
                            &mut tr_ptr as *mut _ as *mut c_void,
                            &mut periods_ptr as *mut _ as *mut c_void,
                            &mut mults_ptr as *mut _ as *mut c_void,
                            &mut dirs_ptr as *mut _ as *mut c_void,
                            &mut alphas_ptr as *mut _ as *mut c_void,
                            &mut warms_ptr as *mut _ as *mut c_void,
                            &mut len_i as *mut _ as *mut c_void,
                            &mut first_i as *mut _ as *mut c_void,
                            &mut cur_i as *mut _ as *mut c_void,
                            &mut out_ptr as *mut _ as *mut c_void,
                        ];
                        self.stream.launch(&func, grid, block, 0, args)?;
                    } else {
                        let mut high_ptr = d_high.as_device_ptr().as_raw();
                        let mut low_ptr  = d_low.as_device_ptr().as_raw();
                        let mut close_ptr= d_close.as_device_ptr().as_raw();
                        let mut periods_ptr = d_periods.as_device_ptr().as_raw()
                            .wrapping_add((launched * std::mem::size_of::<i32>()) as u64);
                        let mut mults_ptr   = d_mults.as_device_ptr().as_raw()
                            .wrapping_add((launched * std::mem::size_of::<f32>()) as u64);
                        let mut dirs_ptr    = d_dirs.as_device_ptr().as_raw()
                            .wrapping_add((launched * std::mem::size_of::<i32>()) as u64);
                        let mut alphas_ptr  = d_alphas.as_device_ptr().as_raw()
                            .wrapping_add((launched * std::mem::size_of::<f32>()) as u64);
                        let warms_slice = &h_warms[launched..(launched + cur)];
                        let d_warms   = DeviceBuffer::from_slice(warms_slice)?;
                        let mut warms_ptr = d_warms.as_device_ptr().as_raw();

                        let args: &mut [*mut c_void] = &mut [
                            &mut high_ptr as *mut _ as *mut c_void,
                            &mut low_ptr as *mut _ as *mut c_void,
                            &mut close_ptr as *mut _ as *mut c_void,
                            &mut periods_ptr as *mut _ as *mut c_void,
                            &mut mults_ptr as *mut _ as *mut c_void,
                            &mut dirs_ptr as *mut _ as *mut c_void,
                            &mut alphas_ptr as *mut _ as *mut c_void,
                            &mut warms_ptr as *mut _ as *mut c_void,
                            &mut len_i as *mut _ as *mut c_void,
                            &mut first_i as *mut _ as *mut c_void,
                            &mut cur_i as *mut _ as *mut c_void,
                            &mut out_ptr as *mut _ as *mut c_void,
                        ];
                        self.stream.launch(&func, grid, block, 0, args)?;
                    }
                }
                launched += cur;
            }
        }

        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: len,
        })
    }

    fn first_valids_time_major(
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
    ) -> Result<Vec<i32>, CudaChandeError> {
        let n = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaChandeError::InvalidInput("rows*cols overflow".into()))?;
        if high_tm.len() != n || low_tm.len() != n || close_tm.len() != n {
            return Err(CudaChandeError::InvalidInput(
                "time-major input length mismatch".into(),
            ));
        }
        let mut out = vec![-1i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                let h = high_tm[idx];
                let l = low_tm[idx];
                let c = close_tm[idx];
                if !h.is_nan() && !l.is_nan() && !c.is_nan() {
                    out[s] = t as i32;
                    break;
                }
            }
        }
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                let h = high_tm[idx];
                let l = low_tm[idx];
                let c = close_tm[idx];
                if !h.is_nan() && !l.is_nan() && !c.is_nan() {
                    out[s] = t as i32;
                    break;
                }
            }
        }
        Ok(out)
    }

    pub fn chande_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
        mult: f32,
        direction: &str,
    ) -> Result<DeviceArrayF32, CudaChandeError> {
        if period == 0 {
            return Err(CudaChandeError::InvalidInput("period must be > 0".into()));
        }
        if !(direction.eq_ignore_ascii_case("long") || direction.eq_ignore_ascii_case("short")) {
            return Err(CudaChandeError::InvalidInput(
                "direction must be 'long' or 'short'".into(),
            ));
        }
        let first_valids = Self::first_valids_time_major(high_tm, low_tm, close_tm, cols, rows)?;
        if rows < period {
            return Err(CudaChandeError::InvalidInput(
                "not enough rows for period".into(),
            ));
        }
        if rows < period {
            return Err(CudaChandeError::InvalidInput(
                "not enough rows for period".into(),
            ));
        }

        let d_high = DeviceBuffer::from_slice(high_tm)?;
        let d_low  = DeviceBuffer::from_slice(low_tm)?;
        let d_close= DeviceBuffer::from_slice(close_tm)?;
        let d_fv   = DeviceBuffer::from_slice(&first_valids)?;
        let total = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaChandeError::InvalidInput("cols*rows overflow".into()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total) }?;

        let func = self
            .module
            .get_function("chande_many_series_one_param_f32")
            .map_err(|_| CudaChandeError::MissingKernelSymbol { name: "chande_many_series_one_param_f32" })?;
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            ManySeriesKernelPolicy::Auto => 256,
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        if block_x > 1024 {
            return Err(CudaChandeError::LaunchConfigTooLarge { gx: grid_x, gy: 1, gz: 1, bx: block_x, by: 1, bz: 1 });
        }
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        let dir_flag: i32 = if direction.eq_ignore_ascii_case("long") { 1 } else { 0 };
        let alpha = 1.0f32 / (period as f32);
        unsafe {
            let mut high_ptr = d_high.as_device_ptr().as_raw();
            let mut low_ptr  = d_low.as_device_ptr().as_raw();
            let mut close_ptr= d_close.as_device_ptr().as_raw();
            let mut fv_ptr   = d_fv.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut mult_f   = mult;
            let mut dir_i    = dir_flag;
            let mut alpha_f  = alpha;
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut mult_f as *mut _ as *mut c_void,
                &mut dir_i as *mut _ as *mut c_void,
                &mut alpha_f as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }
}

// ---------------- Bench profiles ----------------
#[cfg(not(test))]
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 512_000; // moderate, naive window scan is O(period)

    fn synth_hlc_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i];
            if v.is_nan() {
                continue;
            }
            let v = close[i];
            if v.is_nan() {
                continue;
            }
            let x = i as f32 * 0.002f32;
            let off = (0.004 * x.sin()).abs() + 0.12;
            high[i] = v + off;
            low[i] = v - off;
        }
        (high, low)
    }

    struct ChandeBatchDeviceState {
        cuda: CudaChande,
        d_high: DeviceBuffer<f32>,
        d_low: DeviceBuffer<f32>,
        d_close: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_mults: DeviceBuffer<f32>,
        d_dirs: DeviceBuffer<i32>,
        d_alphas: DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        queue_cap: usize,
        grid: GridSize,
        block: BlockSize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for ChandeBatchDeviceState {
        fn launch(&mut self) {
            let func = self
                .cuda
                .module
                .get_function("chande_one_series_many_params_f32")
                .expect("chande_one_series_many_params_f32");

            unsafe {
                let mut high_ptr = self.d_high.as_device_ptr().as_raw();
                let mut low_ptr = self.d_low.as_device_ptr().as_raw();
                let mut close_ptr = self.d_close.as_device_ptr().as_raw();
                let mut periods_ptr = self.d_periods.as_device_ptr().as_raw();
                let mut mults_ptr = self.d_mults.as_device_ptr().as_raw();
                let mut dirs_ptr = self.d_dirs.as_device_ptr().as_raw();
                let mut alphas_ptr = self.d_alphas.as_device_ptr().as_raw();
                let mut first_i = self.first_valid as i32;
                let mut len_i = self.len as i32;
                let mut combos_i = self.n_combos as i32;
                let mut qcap_i = self.queue_cap as i32;
                let dq_idx_ref = self.cuda.dq_idx.as_ref().expect("dq_idx");
                let dq_val_ref = self.cuda.dq_val.as_ref().expect("dq_val");
                let mut dq_idx_ptr = dq_idx_ref.as_device_ptr().as_raw();
                let mut dq_val_ptr = dq_val_ref.as_device_ptr().as_raw();
                let mut out_ptr = self.d_out.as_device_ptr().as_raw();

                let args: &mut [*mut c_void] = &mut [
                    &mut high_ptr as *mut _ as *mut c_void,
                    &mut low_ptr as *mut _ as *mut c_void,
                    &mut close_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut mults_ptr as *mut _ as *mut c_void,
                    &mut dirs_ptr as *mut _ as *mut c_void,
                    &mut alphas_ptr as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut combos_i as *mut _ as *mut c_void,
                    &mut qcap_i as *mut _ as *mut c_void,
                    &mut dq_idx_ptr as *mut _ as *mut c_void,
                    &mut dq_val_ptr as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.cuda
                    .stream
                    .launch(&func, self.grid, self.block, 0, args)
                    .expect("chande oneseries launch");
            }
            self.cuda.stream.synchronize().expect("chande sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let len = ONE_SERIES_LEN;
        let close = gen_series(len);
        let (high, low) = synth_hlc_from_close(&close);
        let sweep = ChandeBatchRange {
            period: (10, 40, 5),
            mult: (2.0, 4.0, 1.0),
        };

        let mut cuda = CudaChande::new(0).expect("cuda chande");

        let first_valid = CudaChande::first_valid_hlc(&high, &low, &close).expect("first_valid");
        let dir_flag = 1i32; // long

        let periods = CudaChande::axis_usize_range(sweep.period).expect("period axis");
        let mults_host = CudaChande::axis_f64_range(sweep.mult).expect("mult axis");

        let mut h_periods = Vec::<i32>::new();
        let mut h_alphas = Vec::<f32>::new();
        let mut h_mults = Vec::<f32>::new();
        let mut h_dirs = Vec::<i32>::new();
        let mut max_p = 0usize;
        for &p in &periods {
            max_p = max_p.max(p);
            for &m in &mults_host {
                h_periods.push(p as i32);
                h_alphas.push(1.0f32 / (p as f32));
                h_mults.push(m as f32);
                h_dirs.push(dir_flag);
            }
        }
        let n_combos = h_periods.len();
        let queue_cap = CudaChande::next_pow2_usize(max_p + 1);
        cuda.ensure_workspace(n_combos, queue_cap).expect("workspace");

        // Upload once
        let d_high = DeviceBuffer::from_slice(&high).expect("d_high");
        let d_low = DeviceBuffer::from_slice(&low).expect("d_low");
        let d_close = DeviceBuffer::from_slice(&close).expect("d_close");
        let d_periods = DeviceBuffer::from_slice(&h_periods).expect("d_periods");
        let d_mults = DeviceBuffer::from_slice(&h_mults).expect("d_mults");
        let d_dirs = DeviceBuffer::from_slice(&h_dirs).expect("d_dirs");
        let d_alphas = DeviceBuffer::from_slice(&h_alphas).expect("d_alphas");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * len) }.expect("d_out");

        // Launch config (same as chande_batch_dev oneseries path)
        let warps_needed = ((n_combos + 31) / 32) as u32;
        let warps_per_block = match cuda.policy.batch {
            BatchKernelPolicy::Plain { block_x } => (block_x.max(32) / 32),
            BatchKernelPolicy::Auto => 4,
        }
        .max(1);
        let block_x = warps_per_block * 32;
        let grid_x = ((warps_needed + warps_per_block - 1) / warps_per_block).max(1);
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        Box::new(ChandeBatchDeviceState {
            cuda,
            d_high,
            d_low,
            d_close,
            d_periods,
            d_mults,
            d_dirs,
            d_alphas,
            len,
            first_valid,
            n_combos,
            queue_cap,
            grid,
            block,
            d_out,
        })
    }

    struct ChandeManyDeviceState {
        cuda: CudaChande,
        d_high_tm: DeviceBuffer<f32>,
        d_low_tm: DeviceBuffer<f32>,
        d_close_tm: DeviceBuffer<f32>,
        d_first: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        mult: f32,
        dir_flag: i32,
        alpha: f32,
        grid: GridSize,
        block: BlockSize,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for ChandeManyDeviceState {
        fn launch(&mut self) {
            let func = self
                .cuda
                .module
                .get_function("chande_many_series_one_param_f32")
                .expect("chande_many_series_one_param_f32");
            unsafe {
                let mut high_ptr = self.d_high_tm.as_device_ptr().as_raw();
                let mut low_ptr = self.d_low_tm.as_device_ptr().as_raw();
                let mut close_ptr = self.d_close_tm.as_device_ptr().as_raw();
                let mut fv_ptr = self.d_first.as_device_ptr().as_raw();
                let mut period_i = self.period as i32;
                let mut mult_f = self.mult;
                let mut dir_i = self.dir_flag;
                let mut alpha_f = self.alpha;
                let mut num_series_i = self.cols as i32;
                let mut series_len_i = self.rows as i32;
                let mut out_ptr = self.d_out_tm.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut high_ptr as *mut _ as *mut c_void,
                    &mut low_ptr as *mut _ as *mut c_void,
                    &mut close_ptr as *mut _ as *mut c_void,
                    &mut fv_ptr as *mut _ as *mut c_void,
                    &mut period_i as *mut _ as *mut c_void,
                    &mut mult_f as *mut _ as *mut c_void,
                    &mut dir_i as *mut _ as *mut c_void,
                    &mut alpha_f as *mut _ as *mut c_void,
                    &mut num_series_i as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.cuda
                    .stream
                    .launch(&func, self.grid, self.block, 0, args)
                    .expect("chande many launch");
            }
            self.cuda.stream.synchronize().expect("chande many sync");
        }
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let (cols, rows, period, mult) = (128usize, 262_144usize, 22usize, 3.0f32);
        let mut close_tm = vec![f32::NAN; cols * rows];
        for s in 0..cols {
            for t in s..rows {
                let x = (t as f32) + (s as f32) * 0.2;
                close_tm[t * cols + s] = (x * 0.0017).sin() + 0.00015 * x;
            }
        }
        let (mut high_tm, mut low_tm) = (close_tm.clone(), close_tm.clone());
        for s in 0..cols {
            for t in 0..rows {
                let v = close_tm[t * cols + s];
                if v.is_nan() {
                    continue;
                }
                let x = (t as f32) * 0.002;
                let off = (0.004 * x.cos()).abs() + 0.11;
                high_tm[t * cols + s] = v + off;
                low_tm[t * cols + s] = v - off;
            }
        }

        let cuda = CudaChande::new(0).expect("cuda chande");
        let first_valids = CudaChande::first_valids_time_major(&high_tm, &low_tm, &close_tm, cols, rows)
            .expect("first_valids");
        let d_high_tm = DeviceBuffer::from_slice(&high_tm).expect("d_high_tm");
        let d_low_tm = DeviceBuffer::from_slice(&low_tm).expect("d_low_tm");
        let d_close_tm = DeviceBuffer::from_slice(&close_tm).expect("d_close_tm");
        let d_first = DeviceBuffer::from_slice(&first_valids).expect("d_first");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out_tm");

        let dir_flag: i32 = 1;
        let alpha = 1.0f32 / (period as f32);
        let block_x = match cuda.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            ManySeriesKernelPolicy::Auto => 256,
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        Box::new(ChandeManyDeviceState {
            cuda,
            d_high_tm,
            d_low_tm,
            d_close_tm,
            d_first,
            cols,
            rows,
            period,
            mult,
            dir_flag,
            alpha,
            grid,
            block,
            d_out_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        let scen_batch = CudaBenchScenario::new(
            "chande",
            "one_series_many_params",
            "chande_cuda_batch_dev",
            "512k_x_params",
            prep_one_series_many_params,
        )
        .with_sample_size(10);
        let scen_many = CudaBenchScenario::new(
            "chande",
            "many_series_one_param",
            "chande_cuda_many_series_one_param_dev",
            "128x262k",
            prep_many_series_one_param,
        )
        .with_sample_size(10);
        vec![scen_batch, scen_many]
    }
}
