#![cfg(feature = "cuda")]

use crate::indicators::adosc::{AdoscBatchRange, AdoscParams};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{
    mem_get_info, AsyncCopyDestination, CopyDestination, DeviceBuffer, LockedBuffer,
};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaAdoscError {
    #[error("CUDA: {0}")]
    Cuda(#[from] cust::error::CudaError),
    #[error(
        "Out of memory: required={required} bytes, free={free} bytes, headroom={headroom} bytes"
    )]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("Missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("Launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
    #[error("Device mismatch: buffer device {buf}, current {current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("Not implemented")]
    NotImplemented,
}

pub struct DeviceArrayF32Adosc {
    pub buf: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
    pub ctx: Arc<Context>,
    pub device_id: u32,
}
impl DeviceArrayF32Adosc {
    #[inline]
    pub fn device_ptr(&self) -> u64 {
        self.buf.as_device_ptr().as_raw() as u64
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaAdosc {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaAdoscPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaAdosc {
    #[inline]
    fn div_up(&self, n: usize, d: usize) -> u32 {
        ((n + d - 1) / d) as u32
    }
    #[inline]
    fn default_block_x_for_batch(&self) -> u32 {
        match self.policy.batch {
            BatchKernelPolicy::Auto => 256,
            BatchKernelPolicy::Plain { block_x } => block_x.max(1),
        }
    }
    #[inline]
    fn default_block_x_for_many(&self) -> u32 {
        match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 256,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(1),
        }
    }
    #[inline]
    fn device_max_grid_x(&self) -> Result<u32, CudaAdoscError> {
        let dev = Device::get_device(self.device_id).map_err(CudaAdoscError::Cuda)?;
        Ok(dev
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .map_err(CudaAdoscError::Cuda)? as u32)
    }
    pub fn new(device_id: usize) -> Result<Self, CudaAdoscError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/adosc_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O4),
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
            policy: CudaAdoscPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(
        device_id: usize,
        policy: CudaAdoscPolicy,
    ) -> Result<Self, CudaAdoscError> {
        let mut s = Self::new(device_id)?;
        unsafe {
            (*(&s as *const _ as *mut CudaAdosc)).policy = policy;
        }
        Ok(s)
    }
    #[inline]
    pub fn set_policy(&mut self, policy: CudaAdoscPolicy) {
        self.policy = policy;
    }
    #[inline]
    pub fn policy(&self) -> &CudaAdoscPolicy {
        &self.policy
    }

    pub fn adosc_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        volume: &[f32],
        sweep: &AdoscBatchRange,
    ) -> Result<DeviceArrayF32Adosc, CudaAdoscError> {
        let len = high.len();
        if len == 0 || low.len() != len || close.len() != len || volume.len() != len {
            return Err(CudaAdoscError::InvalidInput(
                "input slices are empty or mismatched".into(),
            ));
        }
        let combos = expand_grid_checked_cuda(sweep)?;

        let d_high = unsafe { DeviceBuffer::from_slice_async(high, &self.stream) }?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(low, &self.stream) }?;
        let d_close = unsafe { DeviceBuffer::from_slice_async(close, &self.stream) }?;
        let d_volume = unsafe { DeviceBuffer::from_slice_async(volume, &self.stream) }?;

        let mut d_adl: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(len, &self.stream)? };
        self.launch_adl(&d_high, &d_low, &d_close, &d_volume, len, &mut d_adl)?;

        let mut shorts: Vec<i32> = Vec::with_capacity(combos.len());
        let mut longs: Vec<i32> = Vec::with_capacity(combos.len());
        for prm in &combos {
            let sp = prm.short_period.unwrap_or(3) as i32;
            let lp = prm.long_period.unwrap_or(10) as i32;
            if sp <= 0 || lp <= 0 || sp >= lp {
                return Err(CudaAdoscError::InvalidInput(format!(
                    "invalid params: short={} long={}",
                    sp, lp
                )));
            }
            shorts.push(sp);
            longs.push(lp);
        }
        let d_shorts = unsafe { DeviceBuffer::from_slice_async(&shorts, &self.stream) }?;
        let d_longs = unsafe { DeviceBuffer::from_slice_async(&longs, &self.stream) }?;

        let (rows, cols) = (combos.len(), len);
        let bytes_inputs = 4usize
            .checked_mul(cols)
            .ok_or_else(|| CudaAdoscError::InvalidInput("size overflow".into()))?
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaAdoscError::InvalidInput("size overflow".into()))?;
        let bytes_adl = cols
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaAdoscError::InvalidInput("size overflow".into()))?;
        let bytes_periods = 2usize
            .checked_mul(rows)
            .ok_or_else(|| CudaAdoscError::InvalidInput("size overflow".into()))?
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaAdoscError::InvalidInput("size overflow".into()))?;
        let bytes_out_total = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaAdoscError::InvalidInput("rows*cols overflow".into()))?
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaAdoscError::InvalidInput("size overflow".into()))?;
        let headroom = 64usize * 1024 * 1024;

        let required = bytes_inputs
            .checked_add(bytes_adl)
            .ok_or_else(|| CudaAdoscError::InvalidInput("size overflow".into()))?
            .checked_add(bytes_periods)
            .ok_or_else(|| CudaAdoscError::InvalidInput("size overflow".into()))?
            .checked_add(bytes_out_total)
            .ok_or_else(|| CudaAdoscError::InvalidInput("size overflow".into()))?;
        let fits_all = match Self::will_fit(required, headroom) {
            Ok(true) => true,
            Ok(false) => false,
            Err(CudaAdoscError::OutOfMemory { .. }) => false,
            Err(e) => return Err(e),
        };

        if fits_all {
            let total = rows
                .checked_mul(cols)
                .ok_or_else(|| CudaAdoscError::InvalidInput("rows*cols overflow".into()))?;
            let mut d_out: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized_async(total, &self.stream)? };
            unsafe {
                (*(self as *const _ as *mut CudaAdosc)).last_batch =
                    Some(BatchKernelSelected::Plain {
                        block_x: self.default_block_x_for_batch(),
                    });
            }
            self.launch_batch_from_adl(&d_adl, &d_shorts, &d_longs, cols, rows, &mut d_out)?;
            self.maybe_log_batch_debug();

            self.stream.synchronize()?;
            return Ok(DeviceArrayF32Adosc {
                buf: d_out,
                rows,
                cols,
                ctx: Arc::clone(&self._context),
                device_id: self.device_id,
            });
        }

        let can_hold_whole_output = match mem_get_info() {
            Ok((free, _)) => {
                let static_now = bytes_inputs + bytes_adl + headroom;
                static_now.saturating_add(bytes_out_total) <= free
            }
            Err(_) => false,
        };
        let max_grid = self.device_max_grid_x().unwrap_or(65_535) as usize;

        if can_hold_whole_output {
            let total = rows
                .checked_mul(cols)
                .ok_or_else(|| CudaAdoscError::InvalidInput("rows*cols overflow".into()))?;
            let mut d_out_full: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized_async(total, &self.stream)? };
            let mut start = 0usize;
            while start < rows {
                let remain = rows - start;
                let chunk = remain.min(max_grid);
                let d_shorts_off = unsafe {
                    DeviceBuffer::from_slice_async(&shorts[start..start + chunk], &self.stream)?
                };
                let d_longs_off = unsafe {
                    DeviceBuffer::from_slice_async(&longs[start..start + chunk], &self.stream)?
                };
                let mut d_out_chunk: DeviceBuffer<f32> =
                    unsafe { DeviceBuffer::uninitialized_async(chunk * cols, &self.stream)? };
                self.launch_batch_from_adl(
                    &d_adl,
                    &d_shorts_off,
                    &d_longs_off,
                    cols,
                    chunk,
                    &mut d_out_chunk,
                )?;
                let base = start * cols;
                let mut dst_slice = d_out_full.index(base..base + chunk * cols);
                unsafe { dst_slice.async_copy_from(&d_out_chunk, &self.stream) }?;
                start += chunk;
            }

            self.stream.synchronize()?;
            return Ok(DeviceArrayF32Adosc {
                buf: d_out_full,
                rows,
                cols,
                ctx: Arc::clone(&self._context),
                device_id: self.device_id,
            });
        }

        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaAdoscError::InvalidInput("rows*cols overflow".into()))?;
        let mut d_out_full: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(total, &self.stream)? };
        let mut start = 0usize;
        while start < rows {
            let remain = rows - start;
            let chunk = remain.min(max_grid);
            let d_shorts_off = unsafe {
                DeviceBuffer::from_slice_async(&shorts[start..start + chunk], &self.stream)?
            };
            let d_longs_off = unsafe {
                DeviceBuffer::from_slice_async(&longs[start..start + chunk], &self.stream)?
            };

            let mut d_out_chunk: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized_async(chunk * cols, &self.stream)? };
            self.launch_batch_from_adl(
                &d_adl,
                &d_shorts_off,
                &d_longs_off,
                cols,
                chunk,
                &mut d_out_chunk,
            )?;
            let base = start * cols;
            let mut dst_slice = d_out_full.index(base..base + chunk * cols);
            unsafe { dst_slice.async_copy_from(&d_out_chunk, &self.stream) }?;
            start += chunk;
        }

        self.stream.synchronize()?;
        Ok(DeviceArrayF32Adosc {
            buf: d_out_full,
            rows,
            cols,
            ctx: Arc::clone(&self._context),
            device_id: self.device_id,
        })
    }

    pub fn adosc_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        volume_tm: &[f32],
        cols: usize,
        rows: usize,
        short: usize,
        long: usize,
    ) -> Result<DeviceArrayF32Adosc, CudaAdoscError> {
        let len = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaAdoscError::InvalidInput("rows*cols overflow".into()))?;
        if high_tm.len() != len
            || low_tm.len() != len
            || close_tm.len() != len
            || volume_tm.len() != len
        {
            return Err(CudaAdoscError::InvalidInput(
                "time-major inputs are mismatched".into(),
            ));
        }
        if short == 0 || long == 0 || short >= long {
            return Err(CudaAdoscError::InvalidInput("invalid short/long".into()));
        }

        let bytes_inputs = 4usize
            .checked_mul(len)
            .ok_or_else(|| CudaAdoscError::InvalidInput("size overflow".into()))?
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaAdoscError::InvalidInput("size overflow".into()))?;
        let bytes_output = len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaAdoscError::InvalidInput("size overflow".into()))?;
        let required = bytes_inputs
            .checked_add(bytes_output)
            .ok_or_else(|| CudaAdoscError::InvalidInput("size overflow".into()))?;
        let headroom = 64usize * 1024 * 1024;
        let _ = Self::will_fit(required, headroom)?;

        let d_high = unsafe { DeviceBuffer::from_slice_async(high_tm, &self.stream) }?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(low_tm, &self.stream) }?;
        let d_close = unsafe { DeviceBuffer::from_slice_async(close_tm, &self.stream) }?;
        let d_volume = unsafe { DeviceBuffer::from_slice_async(volume_tm, &self.stream) }?;

        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(len, &self.stream)? };
        self.launch_many_series_one_param(
            &d_high, &d_low, &d_close, &d_volume, cols, rows, short, long, &mut d_out,
        )?;

        self.stream.synchronize()?;
        Ok(DeviceArrayF32Adosc {
            buf: d_out,
            rows,
            cols,
            ctx: Arc::clone(&self._context),
            device_id: self.device_id,
        })
    }

    fn launch_adl(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        d_volume: &DeviceBuffer<f32>,
        series_len: usize,
        d_adl_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAdoscError> {
        let func = self.module.get_function("adosc_adl_f32").map_err(|_| {
            CudaAdoscError::MissingKernelSymbol {
                name: "adosc_adl_f32",
            }
        })?;
        let grid: GridSize = (1, 1, 1).into();
        let block: BlockSize = (1, 1, 1).into();
        unsafe {
            let mut h = d_high.as_device_ptr().as_raw();
            let mut l = d_low.as_device_ptr().as_raw();
            let mut c = d_close.as_device_ptr().as_raw();
            let mut v = d_volume.as_device_ptr().as_raw();
            let mut n_i = series_len as i32;
            let mut out = d_adl_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut h as *mut _ as *mut c_void,
                &mut l as *mut _ as *mut c_void,
                &mut c as *mut _ as *mut c_void,
                &mut v as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut out as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    fn launch_batch_from_adl(
        &self,
        d_adl: &DeviceBuffer<f32>,
        d_shorts: &DeviceBuffer<i32>,
        d_longs: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAdoscError> {
        if n_combos == 0 || series_len == 0 {
            return Ok(());
        }
        let func = self
            .module
            .get_function("adosc_batch_from_adl_f32")
            .map_err(|_| CudaAdoscError::MissingKernelSymbol {
                name: "adosc_batch_from_adl_f32",
            })?;

        let mut block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x.max(32).min(1024),
            BatchKernelPolicy::Auto => 256,
        };
        block_x = block_x.max(32).min(1024);
        block_x -= block_x % 32;
        let warps_per_block = (block_x / 32).max(1);
        let max_grid = self.device_max_grid_x().unwrap_or(65_535).max(1);
        let combos_per_launch: usize = (warps_per_block as usize) * (max_grid as usize);
        unsafe {
            (*(self as *const _ as *mut CudaAdosc)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }

        let mut launched = 0usize;
        while launched < n_combos {
            let this_chunk = (n_combos - launched).min(combos_per_launch);
            let grid_x = ((this_chunk as u32) + warps_per_block - 1) / warps_per_block;
            let grid: GridSize = (grid_x.max(1), 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            unsafe {
                let mut adl = d_adl.as_device_ptr().as_raw();
                let mut sp = d_shorts
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * std::mem::size_of::<i32>()) as u64);
                let mut lp = d_longs
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * std::mem::size_of::<i32>()) as u64);
                let mut n = series_len as i32;
                let mut combos = this_chunk as i32;
                let mut out = d_out
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add(((launched * series_len) * std::mem::size_of::<f32>()) as u64);
                let args: &mut [*mut c_void] = &mut [
                    &mut adl as *mut _ as *mut c_void,
                    &mut sp as *mut _ as *mut c_void,
                    &mut lp as *mut _ as *mut c_void,
                    &mut n as *mut _ as *mut c_void,
                    &mut combos as *mut _ as *mut c_void,
                    &mut out as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, 0, args)?;
            }
            launched += this_chunk;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_one_param(
        &self,
        d_high_tm: &DeviceBuffer<f32>,
        d_low_tm: &DeviceBuffer<f32>,
        d_close_tm: &DeviceBuffer<f32>,
        d_volume_tm: &DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        short: usize,
        long: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAdoscError> {
        if cols == 0 || rows == 0 {
            return Err(CudaAdoscError::InvalidInput(
                "cols/rows must be positive".into(),
            ));
        }
        let func = self
            .module
            .get_function("adosc_many_series_one_param_f32")
            .map_err(|_| CudaAdoscError::MissingKernelSymbol {
                name: "adosc_many_series_one_param_f32",
            })?;

        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32).min(1024),
            ManySeriesKernelPolicy::Auto => 256,
        };
        let max_grid = self.device_max_grid_x().unwrap_or(65_535);
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1).min(max_grid), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            (*(self as *const _ as *mut CudaAdosc)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        unsafe {
            let mut h = d_high_tm.as_device_ptr().as_raw();
            let mut l = d_low_tm.as_device_ptr().as_raw();
            let mut c = d_close_tm.as_device_ptr().as_raw();
            let mut v = d_volume_tm.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut sp = short as i32;
            let mut lp = long as i32;
            let mut out = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut h as *mut _ as *mut c_void,
                &mut l as *mut _ as *mut c_void,
                &mut c as *mut _ as *mut c_void,
                &mut v as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut sp as *mut _ as *mut c_void,
                &mut lp as *mut _ as *mut c_void,
                &mut out as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
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
pub struct CudaAdoscPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaAdoscPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

impl CudaAdosc {
    #[inline]
    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_scenario =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] ADOSC batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaAdosc)).debug_batch_logged = true;
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
                let per_scenario =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] ADOSC many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaAdosc)).debug_many_logged = true;
                }
            }
        }
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = 4 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let adl_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + adl_bytes + out_bytes + 64 * 1024 * 1024
    }

    fn synth_hlc_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i];
            if v.is_nan() {
                continue;
            }
            let x = i as f32 * 0.0019;
            let off = (0.0031 * x.sin()).abs() + 0.08;
            high[i] = v + off;
            low[i] = v - off;
        }
        (high, low)
    }

    struct AdoscBatchDeviceState {
        cuda: CudaAdosc,
        d_adl: DeviceBuffer<f32>,
        d_shorts: DeviceBuffer<i32>,
        d_longs: DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for AdoscBatchDeviceState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_from_adl(
                    &self.d_adl,
                    &self.d_shorts,
                    &self.d_longs,
                    self.series_len,
                    self.n_combos,
                    &mut self.d_out,
                )
                .expect("adosc launch");
            self.cuda.stream.synchronize().expect("adosc sync");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaAdosc::new(0).expect("cuda adosc");
        let close = gen_series(ONE_SERIES_LEN);
        let (high, low) = synth_hlc_from_close(&close);
        let mut volume = vec![0.0f32; ONE_SERIES_LEN];
        for i in 0..ONE_SERIES_LEN {
            let x = i as f32 * 0.0027;
            volume[i] = (x.cos().abs() + 0.4) * 1000.0;
        }
        let sweep = AdoscBatchRange {
            short_period: (3, 3, 0),
            long_period: (10, 10 + PARAM_SWEEP - 1, 1),
        };

        let combos = expand_grid_checked_cuda(&sweep).expect("expand_grid");
        let series_len = close.len();
        let n_combos = combos.len();
        let mut shorts: Vec<i32> = Vec::with_capacity(n_combos);
        let mut longs: Vec<i32> = Vec::with_capacity(n_combos);
        for prm in &combos {
            shorts.push(prm.short_period.unwrap_or(0) as i32);
            longs.push(prm.long_period.unwrap_or(0) as i32);
        }

        let d_high = DeviceBuffer::from_slice(&high).expect("H2D");
        let d_low = DeviceBuffer::from_slice(&low).expect("H2D");
        let d_close = DeviceBuffer::from_slice(&close).expect("H2D");
        let d_volume = DeviceBuffer::from_slice(&volume).expect("H2D");

        let mut d_adl: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(series_len) }.expect("adl alloc");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(series_len * n_combos) }.expect("out alloc");
        let d_shorts = DeviceBuffer::from_slice(&shorts).expect("shorts H2D");
        let d_longs = DeviceBuffer::from_slice(&longs).expect("longs H2D");

        cuda.launch_adl(&d_high, &d_low, &d_close, &d_volume, series_len, &mut d_adl)
            .expect("adl kernel");
        cuda.stream.synchronize().expect("adosc prep sync");

        Box::new(AdoscBatchDeviceState {
            cuda,
            d_adl,
            d_shorts,
            d_longs,
            series_len,
            n_combos,
            d_out,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "adosc",
            "one_series_many_params",
            "adosc_cuda_batch_dev",
            "1m_x_250",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}

fn expand_grid(r: &AdoscBatchRange) -> Vec<AdoscParams> {
    match expand_grid_checked_cuda(r) {
        Ok(v) => v,
        Err(_) => Vec::new(),
    }
}

fn expand_grid_checked_cuda(r: &AdoscBatchRange) -> Result<Vec<AdoscParams>, CudaAdoscError> {
    fn axis((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, CudaAdoscError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let v: Vec<_> = (start..=end).step_by(step).collect();
            if v.is_empty() {
                return Err(CudaAdoscError::InvalidInput("empty range".into()));
            }
            Ok(v)
        } else {
            let mut v = Vec::new();
            let mut cur = start;
            while cur >= end {
                v.push(cur);
                if cur - end < step {
                    break;
                }
                cur -= step;
            }
            if v.is_empty() {
                return Err(CudaAdoscError::InvalidInput("empty range".into()));
            }
            Ok(v)
        }
    }
    let shorts = axis(r.short_period)?;
    let longs = axis(r.long_period)?;
    let mut out = Vec::new();
    for &s in &shorts {
        for &l in &longs {
            if s == 0 || l == 0 || s >= l {
                continue;
            }
            out.push(AdoscParams {
                short_period: Some(s),
                long_period: Some(l),
            });
        }
    }
    if out.is_empty() {
        return Err(CudaAdoscError::InvalidInput("no parameter combos".into()));
    }
    Ok(out)
}

impl CudaAdosc {
    #[inline]
    fn will_fit(required: usize, headroom: usize) -> Result<bool, CudaAdoscError> {
        match mem_get_info() {
            Ok((free, _)) => {
                let need = required.saturating_add(headroom);
                if need <= free {
                    Ok(true)
                } else {
                    Err(CudaAdoscError::OutOfMemory {
                        required,
                        free,
                        headroom,
                    })
                }
            }
            Err(_) => Ok(true),
        }
    }
}
