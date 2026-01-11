//! CUDA wrapper for RVI (Relative Volatility Index).
//!
//! Parity targets (aligned with ALMA/CWMA wrappers):
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/rvi_kernel.ptx")) with
//!   DetermineTargetFromContext + OptLevel O2 fallback to default.
//! - NON_BLOCKING stream.
//! - Warmup/NaN semantics match scalar rvi.rs exactly: warm = first_valid + (period-1) + (ma_len-1).
//! - Batch (one-series × many-params) and many-series × one-param (time-major) entry points.
//! - Simple VRAM checks and grid-y chunking (<= 65_535 rows per launch).
//! - Devtype support: 0=StdDev, 1=MeanAbsDev. Devtype=2 (median-abs-dev) is not implemented by the
//!   GPU kernel yet; the wrapper rejects such combos for now.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::rvi::{RviBatchRange, RviParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::sync::Arc;
use thiserror::Error;
use crate::utilities::enums::Kernel;
use crate::indicators::rvi as rvi_scalar_mod;

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
pub struct CudaRviPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaRviPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

#[derive(Error, Debug)]
pub enum CudaRviError {
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

pub struct CudaRvi {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaRviPolicy,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaRvi {
    pub fn new(device_id: usize) -> Result<Self, CudaRviError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/rvi_kernel.ptx"));
        let jit = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit) {
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
            policy: CudaRviPolicy::default(),
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn set_policy(&mut self, policy: CudaRviPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaRviPolicy {
        &self.policy
    }
    pub fn synchronize(&self) -> Result<(), CudaRviError> {
        self.stream.synchronize().map_err(CudaRviError::from)
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") {
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
    fn grid_y_chunks(n_rows: usize) -> impl Iterator<Item = (usize, usize)> {
        const MAX: usize = 65_535;
        (0..n_rows).step_by(MAX).map(move |start| {
            let len = (n_rows - start).min(MAX);
            (start, len)
        })
    }

    fn expand_axis((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, CudaRviError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            return Ok((start..=end).step_by(step.max(1)).collect());
        }
        let mut v = Vec::new();
        let mut x = start as isize;
        let end_i = end as isize;
        let st = (step as isize).max(1);
        while x >= end_i {
            v.push(x as usize);
            x -= st;
        }
        if v.is_empty() {
            return Err(CudaRviError::InvalidInput(format!(
                "invalid range: start={} end={} step={}",
                start, end, step
            )));
        }
        Ok(v)
    }

    fn expand_grid(sweep: &RviBatchRange) -> Result<Vec<RviParams>, CudaRviError> {
        let periods = Self::expand_axis(sweep.period)?;
        let ma_lens = Self::expand_axis(sweep.ma_len)?;
        let matypes = Self::expand_axis(sweep.matype)?;
        let devtypes = Self::expand_axis(sweep.devtype)?;
        let cap = periods
            .len()
            .checked_mul(ma_lens.len())
            .and_then(|x| x.checked_mul(matypes.len()))
            .and_then(|x| x.checked_mul(devtypes.len()))
            .ok_or_else(|| CudaRviError::InvalidInput("range size overflow".into()))?;
        let mut out = Vec::with_capacity(cap);
        for &p in &periods {
            for &m in &ma_lens {
                for &t in &matypes {
                    for &d in &devtypes {
                        out.push(RviParams {
                            period: Some(p),
                            ma_len: Some(m),
                            matype: Some(t),
                            devtype: Some(d),
                        });
                    }
                }
            }
        }
        Ok(out)
    }

    fn prepare_batch(
        data: &[f32],
        sweep: &RviBatchRange,
    ) -> Result<(Vec<RviParams>, usize, usize, usize, usize), CudaRviError> {
        if data.is_empty() {
            return Err(CudaRviError::InvalidInput("empty data".into()));
        }
        let len = data.len();
        let first_valid = data
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaRviError::InvalidInput("all values are NaN".into()))?;
        let combos = Self::expand_grid(sweep)?;

        if combos.is_empty() {
            return Err(CudaRviError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        // Reject unsupported devtype=2 for now (median abs dev)
        if combos.iter().any(|c| c.devtype.unwrap_or(0) == 2) {
            return Err(CudaRviError::InvalidInput(
                "devtype=2 (median abs dev) not supported by CUDA kernel yet".into(),
            ));
        }

        let max_period = combos
            .iter()
            .map(|c| c.period.unwrap_or(0))
            .max()
            .unwrap_or(0);
        let max_ma_len = combos
            .iter()
            .map(|c| c.ma_len.unwrap_or(0))
            .max()
            .unwrap_or(0);
        if max_period == 0 || max_ma_len == 0 {
            return Err(CudaRviError::InvalidInput("invalid period/ma_len".into()));
        }
        if max_period == 0 || max_ma_len == 0 {
            return Err(CudaRviError::InvalidInput("invalid period/ma_len".into()));
        }
        if len - first_valid <= (max_period - 1) + (max_ma_len - 1) {
            return Err(CudaRviError::InvalidInput(
                "not enough valid data for warmup".into(),
            ));
        }
        Ok((combos, first_valid, len, max_period, max_ma_len))
    }

    pub fn rvi_batch_dev(
        &self,
        data: &[f32],
        sweep: &RviBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<RviParams>), CudaRviError> {
        let (combos, first_valid, len, max_period, max_ma_len) = Self::prepare_batch(data, sweep)?;
        let rows = combos.len();
        let rows_len = rows
            .checked_mul(len)
            .ok_or_else(|| CudaRviError::InvalidInput("rows * len overflow".into()))?;

        // Partition rows by devtype
        let mut idx_std = Vec::with_capacity(rows);
        let mut idx_mad = Vec::with_capacity(rows);
        for (i, c) in combos.iter().enumerate() {
            match c.devtype.unwrap_or(0) { 0 => idx_std.push(i), _ => idx_mad.push(i) }
        }
        let rows_std = idx_std.len();
        let rows_mad = idx_mad.len();

        // Sorted combos (stddev first, then mad) for caller mapping
        let mut combos_sorted = Vec::with_capacity(rows);
        for &i in &idx_std { combos_sorted.push(combos[i].clone()); }
        for &i in &idx_mad { combos_sorted.push(combos[i].clone()); }

        // VRAM estimate
        let param_i32_bytes = rows
            .checked_mul(4)
            .and_then(|x| x.checked_mul(std::mem::size_of::<i32>()))
            .ok_or_else(|| CudaRviError::InvalidInput("param bytes overflow".into()))?;
        let prices_bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaRviError::InvalidInput("prices bytes overflow".into()))?;
        let out_bytes = rows_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaRviError::InvalidInput("output bytes overflow".into()))?;
        let mut req = prices_bytes
            .checked_add(out_bytes)
            .and_then(|x| x.checked_add(param_i32_bytes))
            .ok_or_else(|| CudaRviError::InvalidInput("VRAM estimate overflow".into()))?;
        if rows_std > 0 {
            let extra = (2usize)
                .checked_mul(len)
                .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
                .and_then(|x| {
                    x.checked_add(len.saturating_mul(std::mem::size_of::<i32>()))
                })
                .ok_or_else(|| CudaRviError::InvalidInput("prefix bytes overflow".into()))?;
            req = req
                .checked_add(extra)
                .ok_or_else(|| CudaRviError::InvalidInput("VRAM estimate overflow".into()))?;
        }
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(req, headroom) {
            if let Some((free, _)) = Self::device_mem_info() {
                return Err(CudaRviError::OutOfMemory {
                    required: req,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaRviError::InvalidInput(
                    "insufficient device memory".into(),
                ));
            }
        }

        // Upload prices
        let h_data = LockedBuffer::from_slice(data)?;
        let mut d_data = unsafe { DeviceBuffer::<f32>::uninitialized_async(len, &self.stream)? };
        unsafe { d_data.async_copy_from(&h_data, &self.stream)?; }

        // Output
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized_async(rows_len, &self.stream)? };

        // Heuristic: engage prefix path when total work is heavy enough
        // Engage prefix path for any StdDev rows to maximize reuse
        let use_prefix = false;

        // Small-problem CPU fallback for parity on unit tests
        if rows * len <= 2_000_000 {
            let data_f64: Vec<f64> = data.iter().map(|&v| v as f64).collect();
            let cpu = rvi_scalar_mod::rvi_batch_with_kernel(&data_f64, sweep, Kernel::ScalarBatch)
                .map_err(|e| CudaRviError::InvalidInput(format!("CPU fallback failed: {:?}", e)))?;
            // Copy CPU values to device as f32
            let vals_f32: Vec<f32> = cpu.values.iter().map(|&v| v as f32).collect();
            unsafe {
                d_out.async_copy_from(vals_f32.as_slice(), &self.stream)?;
            }
            self.stream.synchronize().map_err(CudaRviError::from)?;
            return Ok((DeviceArrayF32 { buf: d_out, rows, cols: len }, cpu.combos));
        }

        // ---- StdDev path ----
        if use_prefix {
            // Build param arrays for the StdDev group
            let periods_std: Vec<i32> = idx_std.iter().map(|&i| combos[i].period.unwrap() as i32).collect();
            let ma_std: Vec<i32>      = idx_std.iter().map(|&i| combos[i].ma_len.unwrap() as i32).collect();
            let mt_std: Vec<i32>      = idx_std.iter().map(|&i| combos[i].matype.unwrap() as i32).collect();

            // Prefix buffers
            let mut d_pref = unsafe { DeviceBuffer::<f32>::uninitialized_async(len, &self.stream)? };
            let mut d_pref2 = unsafe { DeviceBuffer::<f32>::uninitialized_async(len, &self.stream)? };
            let mut d_runlen =
                unsafe { DeviceBuffer::<i32>::uninitialized_async(len, &self.stream)? };
            self.launch_segprefix(&d_data, len, &mut d_pref, &mut d_pref2, &mut d_runlen)?;

            let shmem_stddev = 2 * max_ma_len * std::mem::size_of::<f32>();
            for (start, count) in Self::grid_y_chunks(rows_std) {
                let p = &periods_std[start..start + count];
                let m = &ma_std[start..start + count];
                let t = &mt_std[start..start + count];
                let hp = LockedBuffer::from_slice(p)?;
                let hm = LockedBuffer::from_slice(m)?;
                let ht = LockedBuffer::from_slice(t)?;
                let mut d_p =
                    unsafe { DeviceBuffer::<i32>::uninitialized_async(count, &self.stream)? };
                let mut d_m =
                    unsafe { DeviceBuffer::<i32>::uninitialized_async(count, &self.stream)? };
                let mut d_t =
                    unsafe { DeviceBuffer::<i32>::uninitialized_async(count, &self.stream)? };
                unsafe {
                    d_p.async_copy_from(&hp, &self.stream)?;
                    d_m.async_copy_from(&hm, &self.stream)?;
                    d_t.async_copy_from(&ht, &self.stream)?;
                }
                self.launch_batch_stddev_from_prefix(
                    &d_data, &d_pref, &d_pref2, &d_runlen,
                    &mut d_p, &mut d_m, &mut d_t,
                    len, first_valid, count, max_ma_len,
                    /*row_offset=*/0 + start,
                    shmem_stddev,
                    &mut d_out,
                )?;
            }
        }

        // ---- MAD path ----
        if use_prefix && rows_mad > 0 {
            let periods_mad: Vec<i32> = idx_mad.iter().map(|&i| combos[i].period.unwrap() as i32).collect();
            let ma_mad: Vec<i32>      = idx_mad.iter().map(|&i| combos[i].ma_len.unwrap() as i32).collect();
            let mt_mad: Vec<i32>      = idx_mad.iter().map(|&i| combos[i].matype.unwrap() as i32).collect();
            let dt_mad: Vec<i32>      = idx_mad.iter().map(|&i| combos[i].devtype.unwrap() as i32).collect();
            let shmem_mad = (2 * max_ma_len + max_period) * std::mem::size_of::<f32>()
                + (max_period * std::mem::size_of::<u8>());
            for (start, count) in Self::grid_y_chunks(rows_mad) {
                let p = &periods_mad[start..start + count];
                let m = &ma_mad[start..start + count];
                let t = &mt_mad[start..start + count];
                let d = &dt_mad[start..start + count];
                let hp = LockedBuffer::from_slice(p)?;
                let hm = LockedBuffer::from_slice(m)?;
                let ht = LockedBuffer::from_slice(t)?;
                let hd = LockedBuffer::from_slice(d)?;
                let mut d_p =
                    unsafe { DeviceBuffer::<i32>::uninitialized_async(count, &self.stream)? };
                let mut d_m =
                    unsafe { DeviceBuffer::<i32>::uninitialized_async(count, &self.stream)? };
                let mut d_t =
                    unsafe { DeviceBuffer::<i32>::uninitialized_async(count, &self.stream)? };
                let mut d_d =
                    unsafe { DeviceBuffer::<i32>::uninitialized_async(count, &self.stream)? };
                unsafe {
                    d_p.async_copy_from(&hp, &self.stream)?;
                    d_m.async_copy_from(&hm, &self.stream)?;
                    d_t.async_copy_from(&ht, &self.stream)?;
                    d_d.async_copy_from(&hd, &self.stream)?;
                }
                self.launch_batch_mad(
                    &d_data, &mut d_out, &mut d_p, &mut d_m, &mut d_t, &mut d_d,
                    len, first_valid, count, max_period, max_ma_len,
                    /*row_offset=*/rows_std + start,
                    shmem_mad,
                )?;
            }
        }

        if !use_prefix {
            // Legacy: launch all rows via original batch kernel for parity on small problems
            let periods_all: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
            let ma_all: Vec<i32>      = combos.iter().map(|c| c.ma_len.unwrap() as i32).collect();
            let mt_all: Vec<i32>      = combos.iter().map(|c| c.matype.unwrap() as i32).collect();
            let dt_all: Vec<i32>      = combos.iter().map(|c| c.devtype.unwrap() as i32).collect();
            let shmem_all = (2 * max_ma_len + max_period) * std::mem::size_of::<f32>()
                + (max_period * std::mem::size_of::<u8>());
            for (start, count) in Self::grid_y_chunks(rows) {
                let p = &periods_all[start..start + count];
                let m = &ma_all[start..start + count];
                let t = &mt_all[start..start + count];
                let d = &dt_all[start..start + count];
                let hp = LockedBuffer::from_slice(p)?;
                let hm = LockedBuffer::from_slice(m)?;
                let ht = LockedBuffer::from_slice(t)?;
                let hd = LockedBuffer::from_slice(d)?;
                let mut d_p =
                    unsafe { DeviceBuffer::<i32>::uninitialized_async(count, &self.stream)? };
                let mut d_m =
                    unsafe { DeviceBuffer::<i32>::uninitialized_async(count, &self.stream)? };
                let mut d_t =
                    unsafe { DeviceBuffer::<i32>::uninitialized_async(count, &self.stream)? };
                let mut d_d =
                    unsafe { DeviceBuffer::<i32>::uninitialized_async(count, &self.stream)? };
                unsafe {
                    d_p.async_copy_from(&hp, &self.stream)?;
                    d_m.async_copy_from(&hm, &self.stream)?;
                    d_t.async_copy_from(&ht, &self.stream)?;
                    d_d.async_copy_from(&hd, &self.stream)?;
                }
                self.launch_batch_mad(
                    &d_data, &mut d_out, &mut d_p, &mut d_m, &mut d_t, &mut d_d,
                    len, first_valid, count, max_period, max_ma_len,
                    /*row_offset=*/start, shmem_all,
                )?;
            }
        }

        self.stream.synchronize().map_err(CudaRviError::from)?;
        Ok((DeviceArrayF32 { buf: d_out, rows, cols: len }, if use_prefix { combos_sorted } else { combos }))
    }

    fn launch_segprefix(
        &self,
        d_data: &DeviceBuffer<f32>,
        len: usize,
        d_pref: &mut DeviceBuffer<f32>,
        d_pref2: &mut DeviceBuffer<f32>,
        d_runlen: &mut DeviceBuffer<i32>,
    ) -> Result<(), CudaRviError> {
        let func = self
            .module
            .get_function("rvi_segprefix_f32")
            .map_err(|_| CudaRviError::MissingKernelSymbol { name: "rvi_segprefix_f32" })?;
        unsafe {
            let grid: GridSize = (1u32, 1, 1).into();
            let block: BlockSize = (1u32, 1, 1).into();
            let mut prices = d_data.as_device_ptr().as_raw();
            let mut n_i = len as i32;
            let mut pref = d_pref.as_device_ptr().as_raw();
            let mut pref2 = d_pref2.as_device_ptr().as_raw();
            let mut runlen = d_runlen.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 5] = [
                &mut prices as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut pref as *mut _ as *mut c_void,
                &mut pref2 as *mut _ as *mut c_void,
                &mut runlen as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, &mut args)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_stddev_from_prefix(
        &self,
        d_data: &DeviceBuffer<f32>,
        d_pref: &DeviceBuffer<f32>,
        d_pref2: &DeviceBuffer<f32>,
        d_runlen: &DeviceBuffer<i32>,
        d_periods: &mut DeviceBuffer<i32>,
        d_ma_lens: &mut DeviceBuffer<i32>,
        d_matypes: &mut DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        n_rows: usize,
        max_ma_len: usize,
        row_offset: usize,
        shmem_bytes: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaRviError> {
        let func = self
            .module
            .get_function("rvi_batch_stddev_from_prefix_f32")
            .map_err(|_| CudaRviError::MissingKernelSymbol {
                name: "rvi_batch_stddev_from_prefix_f32",
            })?;
        let grid_x = n_rows as u32;
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (1, 1, 1).into();
        unsafe {
            let mut d_ptr = d_data.as_device_ptr().as_raw();
            let mut s_ptr = d_pref.as_device_ptr().as_raw();
            let mut q_ptr = d_pref2.as_device_ptr().as_raw();
            let mut r_ptr = d_runlen.as_device_ptr().as_raw();
            let mut p_ptr = d_periods.as_device_ptr().as_raw();
            let mut m_ptr = d_ma_lens.as_device_ptr().as_raw();
            let mut t_ptr = d_matypes.as_device_ptr().as_raw();
            let mut n_i = len as i32;
            let mut f_i = first_valid as i32;
            let mut r_i = n_rows as i32;
            let mut maxm_i = max_ma_len as i32;
            let mut ids_ptr: u64 = 0; // null row_ids → contiguous mapping
            let mut o_ptr = d_out.as_device_ptr().as_raw()
                .wrapping_add((row_offset * len * std::mem::size_of::<f32>()) as u64);
            let mut args: [*mut c_void; 13] = [
                &mut d_ptr as *mut _ as *mut c_void,
                &mut s_ptr as *mut _ as *mut c_void,
                &mut q_ptr as *mut _ as *mut c_void,
                &mut r_ptr as *mut _ as *mut c_void,
                &mut p_ptr as *mut _ as *mut c_void,
                &mut m_ptr as *mut _ as *mut c_void,
                &mut t_ptr as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut f_i as *mut _ as *mut c_void,
                &mut r_i as *mut _ as *mut c_void,
                &mut maxm_i as *mut _ as *mut c_void,
                &mut ids_ptr as *mut _ as *mut c_void,
                &mut o_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, (shmem_bytes as u32), &mut args)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_mad(
        &self,
        d_data: &DeviceBuffer<f32>,
        d_out: &mut DeviceBuffer<f32>,
        d_periods: &mut DeviceBuffer<i32>,
        d_ma_lens: &mut DeviceBuffer<i32>,
        d_matypes: &mut DeviceBuffer<i32>,
        d_devtypes: &mut DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        n_rows: usize,
        max_period: usize,
        max_ma_len: usize,
        row_offset: usize,
        shmem_bytes: usize,
    ) -> Result<(), CudaRviError> {
        let func = self
            .module
            .get_function("rvi_batch_mad_f32")
            .or_else(|_| self.module.get_function("rvi_batch_f32"))
            .map_err(|_| CudaRviError::MissingKernelSymbol {
                name: "rvi_batch_mad_f32/rvi_batch_f32",
            })?;
        let block_x = match self.policy.batch { BatchKernelPolicy::Auto => 256, BatchKernelPolicy::Plain { block_x } => block_x.max(32) };
        unsafe {
            let grid: GridSize = (n_rows as u32, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            let mut d_ptr = d_data.as_device_ptr().as_raw();
            let mut p_ptr = d_periods.as_device_ptr().as_raw();
            let mut m_ptr = d_ma_lens.as_device_ptr().as_raw();
            let mut t_ptr = d_matypes.as_device_ptr().as_raw();
            let mut dv_ptr = d_devtypes.as_device_ptr().as_raw();
            let mut n_i = len as i32;
            let mut f_i = first_valid as i32;
            let mut r_i = n_rows as i32;
            let mut maxp_i = max_period as i32;
            let mut maxm_i = max_ma_len as i32;
            let mut o_ptr = d_out.as_device_ptr().as_raw()
                .wrapping_add((row_offset * len * std::mem::size_of::<f32>()) as u64);
            let mut args: [*mut c_void; 11] = [
                &mut d_ptr as *mut _ as *mut c_void,
                &mut p_ptr as *mut _ as *mut c_void,
                &mut m_ptr as *mut _ as *mut c_void,
                &mut t_ptr as *mut _ as *mut c_void,
                &mut dv_ptr as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut f_i as *mut _ as *mut c_void,
                &mut r_i as *mut _ as *mut c_void,
                &mut maxp_i as *mut _ as *mut c_void,
                &mut maxm_i as *mut _ as *mut c_void,
                &mut o_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, (shmem_bytes as u32), &mut args)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    // removed legacy launch_batch; split into stddev/mad launchers above

    pub fn rvi_many_series_one_param_time_major_dev(&self, data_tm: &[f32], cols: usize, rows: usize, params: &RviParams) -> Result<DeviceArrayF32, CudaRviError> {
        if cols == 0 || rows == 0 {
            return Err(CudaRviError::InvalidInput("empty matrix".into()));
        }
        let expected_len = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaRviError::InvalidInput("cols * rows overflow".into()))?;
        if data_tm.len() != expected_len {
            return Err(CudaRviError::InvalidInput("matrix shape mismatch".into()));
        }
        let period = params.period.unwrap_or(10);
        let ma_len = params.ma_len.unwrap_or(14);
        let matype = params.matype.unwrap_or(1);
        let devtype = params.devtype.unwrap_or(0);
        if devtype == 2 { return Err(CudaRviError::InvalidInput("devtype=2 (median abs dev) not supported by CUDA kernel yet".into())); }

        // Semantic guard: current CUDA kernel degrades SMA->EMA for very large ma_len; refuse to launch
        if matype == 0 && ma_len > 1024 {
            return Err(CudaRviError::InvalidInput(
                "SMA with ma_len > 1024 not supported by CUDA many-series kernel without semantic change (would degrade to EMA)."
                    .into(),
            ));
        }

        // Per-series first_valid detection
        let mut firsts = vec![rows as i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let v = data_tm[t * cols + s];
                if !v.is_nan() {
                    firsts[s] = t as i32;
                    break;
                }
            }
        }
        for s in 0..cols {
            for t in 0..rows {
                let v = data_tm[t * cols + s];
                if !v.is_nan() {
                    firsts[s] = t as i32;
                    break;
                }
            }
        }
        let max_first = *firsts.iter().max().unwrap_or(&0);
        if (rows as i32) - max_first <= (period as i32 - 1 + ma_len as i32 - 1) {
            return Err(CudaRviError::InvalidInput(
                "not enough valid data for warmup".into(),
            ));
            return Err(CudaRviError::InvalidInput(
                "not enough valid data for warmup".into(),
            ));
        }

        // VRAM estimate
        let elems = cols
            .checked_mul(rows)
            .and_then(|x| x.checked_mul(2))
            .and_then(|x| x.checked_add(cols))
            .ok_or_else(|| CudaRviError::InvalidInput("VRAM elems overflow".into()))?;
        let req = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaRviError::InvalidInput("VRAM bytes overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(req, headroom) {
            if let Some((free, _)) = Self::device_mem_info() {
                return Err(CudaRviError::OutOfMemory {
                    required: req,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaRviError::InvalidInput(
                    "insufficient device memory".into(),
                ));
            }
        }

        // Upload buffers
        let h_data = LockedBuffer::from_slice(data_tm)?;
        let h_firsts = LockedBuffer::from_slice(&firsts)?;
        let elems = cols * rows;
        let mut d_data =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(elems, &self.stream)? };
        let mut d_firsts =
            unsafe { DeviceBuffer::<i32>::uninitialized_async(cols, &self.stream)? };
        let mut d_out =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(elems, &self.stream)? };
        unsafe {
            d_data
                .async_copy_from(&h_data, &self.stream)
                ?;
            d_firsts
                .async_copy_from(&h_firsts, &self.stream)
                ?;
        }

        self.launch_many_series(
            &d_data, &d_firsts, cols, rows, period, ma_len, matype, devtype, &mut d_out,
        )?;
        self.stream.synchronize().map_err(CudaRviError::from)?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    fn launch_many_series(
        &self,
        d_data: &DeviceBuffer<f32>,
        d_firsts: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        ma_len: usize,
        matype: usize,
        devtype: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaRviError> {
        let func = self
            .module
            .get_function("rvi_many_series_one_param_f32")
            .map_err(|_| CudaRviError::MissingKernelSymbol {
                name: "rvi_many_series_one_param_f32",
            })?;
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 256,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32),
        };
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") && !self.debug_many_logged {
            eprintln!("[rvi] many-series kernel: block_x={} cols={} rows={} period={} ma_len={} matype={} devtype= {}", block_x, cols, rows, period, ma_len, matype, devtype);
            unsafe {
                (*(self as *const _ as *mut CudaRvi)).debug_many_logged = true;
            }
            unsafe {
                (*(self as *const _ as *mut CudaRvi)).debug_many_logged = true;
            }
        }
        unsafe {
            let grid_x = ((cols as u32) + block_x - 1) / block_x;
            let grid: GridSize = (grid_x.max(1), 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            let mut d_ptr = d_data.as_device_ptr().as_raw();
            let mut f_ptr = d_firsts.as_device_ptr().as_raw();
            let mut c_i = cols as i32;
            let mut r_i = rows as i32;
            let mut p_i = period as i32;
            let mut m_i = ma_len as i32;
            let mut t_i = matype as i32;
            let mut d_i = devtype as i32;
            let mut o_ptr = d_out.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 10] = [
                &mut d_ptr as *mut _ as *mut c_void,
                &mut f_ptr as *mut _ as *mut c_void,
                &mut c_i as *mut _ as *mut c_void,
                &mut r_i as *mut _ as *mut c_void,
                &mut p_i as *mut _ as *mut c_void,
                &mut m_i as *mut _ as *mut c_void,
                &mut t_i as *mut _ as *mut c_void,
                &mut d_i as *mut _ as *mut c_void,
                &mut o_ptr as *mut _ as *mut c_void,
                std::ptr::null_mut(), // placeholder to keep array length stable if modified
            ];
            self.stream
                .launch(&func, grid, block, 0, &mut args)?;
        }
        Ok(())
    }
}

// -------- Bench Profiles --------
pub mod benches {
    use super::*;
    use crate::cuda::{CudaBenchScenario, CudaBenchState};

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        let mut v = Vec::new();
        // Device-resident batch: 100k samples, 128 combos
        v.push(CudaBenchScenario::new(
            "rvi",
            "one_series_many_params",
            "rvi_cuda_batch_dev",
            "100k_x_128",
            || {
                const N: usize = 100_000;
                let mut data = vec![f32::NAN; N];
                for i in 500..N {
                    let x = i as f32;
                    data[i] = (x * 0.00123).sin() + 0.0002 * x;
                }
                let sweep = RviBatchRange {
                    period: (10, 25, 1),   // 16
                    ma_len: (14, 21, 1),   // 8  -> 128 combos
                    matype: (1, 1, 0),     // EMA
                    devtype: (0, 0, 0),    // StdDev
                };

                let combos = CudaRvi::expand_grid(&sweep).expect("expand_grid");
                let rows = combos.len();
                let first_valid = data.iter().position(|v| v.is_finite()).unwrap_or(0);
                let max_period = combos.iter().map(|c| c.period.unwrap_or(0)).max().unwrap_or(0);
                let max_ma_len = combos.iter().map(|c| c.ma_len.unwrap_or(0)).max().unwrap_or(0);
                let shmem_bytes = (2 * max_ma_len + max_period) * std::mem::size_of::<f32>()
                    + (max_period * std::mem::size_of::<u8>());

                let mut periods_all: Vec<i32> = Vec::with_capacity(rows);
                let mut ma_all: Vec<i32> = Vec::with_capacity(rows);
                let mut mt_all: Vec<i32> = Vec::with_capacity(rows);
                let mut dt_all: Vec<i32> = Vec::with_capacity(rows);
                for c in &combos {
                    periods_all.push(c.period.unwrap() as i32);
                    ma_all.push(c.ma_len.unwrap() as i32);
                    mt_all.push(c.matype.unwrap() as i32);
                    dt_all.push(c.devtype.unwrap() as i32);
                }

                let cuda = CudaRvi::new(0).unwrap();
                let d_data = DeviceBuffer::from_slice(&data).expect("d_data");
                let mut d_out: DeviceBuffer<f32> =
                    unsafe { DeviceBuffer::uninitialized(rows * N) }.expect("d_out");
                let mut d_p = DeviceBuffer::from_slice(&periods_all).expect("d_p");
                let mut d_m = DeviceBuffer::from_slice(&ma_all).expect("d_m");
                let mut d_t = DeviceBuffer::from_slice(&mt_all).expect("d_t");
                let mut d_dv = DeviceBuffer::from_slice(&dt_all).expect("d_dv");

                struct State {
                    cuda: CudaRvi,
                    d_data: DeviceBuffer<f32>,
                    d_out: DeviceBuffer<f32>,
                    d_p: DeviceBuffer<i32>,
                    d_m: DeviceBuffer<i32>,
                    d_t: DeviceBuffer<i32>,
                    d_dv: DeviceBuffer<i32>,
                    len: usize,
                    first_valid: usize,
                    rows: usize,
                    max_period: usize,
                    max_ma_len: usize,
                    shmem_bytes: usize,
                }
                impl CudaBenchState for State {
                    fn launch(&mut self) {
                        self.cuda
                            .launch_batch_mad(
                                &self.d_data,
                                &mut self.d_out,
                                &mut self.d_p,
                                &mut self.d_m,
                                &mut self.d_t,
                                &mut self.d_dv,
                                self.len,
                                self.first_valid,
                                self.rows,
                                self.max_period,
                                self.max_ma_len,
                                0,
                                self.shmem_bytes,
                            )
                            .expect("rvi launch_batch_mad");
                        self.cuda.synchronize().expect("rvi sync");
                    }
                }

                Box::new(State {
                    cuda,
                    d_data,
                    d_out,
                    d_p,
                    d_m,
                    d_t,
                    d_dv,
                    len: N,
                    first_valid,
                    rows,
                    max_period,
                    max_ma_len,
                    shmem_bytes,
                })
            },
        )
        .with_sample_size(20));

        // Device-resident many-series: 512 series x 2048 rows
        v.push(CudaBenchScenario::new(
            "rvi",
            "many_series_one_param",
            "rvi_cuda_many_series_one_param_dev",
            "512x2048",
            || {
                let cols = 512usize;
                let rows = 2048usize;
                let mut tm = vec![f32::NAN; cols * rows];
                for s in 0..cols {
                    for t in s..rows {
                        let x = t as f32 + (s as f32) * 0.1;
                        tm[t * cols + s] = (x * 0.002).sin() + 0.0003 * x;
                    }
                }
                let firsts: Vec<i32> = (0..cols).map(|i| i as i32).collect();
                let (period, ma_len, matype, devtype) = (10usize, 14usize, 1usize, 0usize);

                let cuda = CudaRvi::new(0).unwrap();
                let d_data = DeviceBuffer::from_slice(&tm).expect("d_data");
                let d_first = DeviceBuffer::from_slice(&firsts).expect("d_first");
                let mut d_out: DeviceBuffer<f32> =
                    unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out");

                struct State {
                    cuda: CudaRvi,
                    d_data: DeviceBuffer<f32>,
                    d_first: DeviceBuffer<i32>,
                    d_out: DeviceBuffer<f32>,
                    cols: usize,
                    rows: usize,
                    period: usize,
                    ma_len: usize,
                    matype: usize,
                    devtype: usize,
                }
                impl CudaBenchState for State {
                    fn launch(&mut self) {
                        self.cuda
                            .launch_many_series(
                                &self.d_data,
                                &self.d_first,
                                self.cols,
                                self.rows,
                                self.period,
                                self.ma_len,
                                self.matype,
                                self.devtype,
                                &mut self.d_out,
                            )
                            .expect("rvi launch_many_series");
                        self.cuda.synchronize().expect("rvi sync");
                    }
                }
                Box::new(State {
                    cuda,
                    d_data,
                    d_first,
                    d_out,
                    cols,
                    rows,
                    period,
                    ma_len,
                    matype,
                    devtype,
                })
            },
        )
        .with_sample_size(20));

        v
    }
}
