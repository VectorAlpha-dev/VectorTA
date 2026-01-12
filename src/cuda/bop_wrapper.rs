#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(thiserror::Error, Debug)]
pub enum CudaBopError {
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

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaBopPolicy {
    pub batch_block_x: Option<u32>,
    pub many_block_x: Option<u32>,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaBop {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaBopPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
    sm_count: u32,
}

impl CudaBop {
    #[inline]
    fn copy_h2d_maybe_async_f32(
        &self,
        src: &[f32],
    ) -> Result<(DeviceBuffer<f32>, Option<LockedBuffer<f32>>), CudaBopError> {
        const ASYNC_PIN_THRESHOLD_BYTES: usize = 1 << 20;
        let bytes = src
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaBopError::InvalidInput("size overflow".into()))?;
        if bytes >= ASYNC_PIN_THRESHOLD_BYTES {
            let h_locked = LockedBuffer::from_slice(src).map_err(CudaBopError::Cuda)?;
            let mut d = unsafe {
                DeviceBuffer::uninitialized_async(src.len(), &self.stream)
                    .map_err(CudaBopError::Cuda)?
            };
            unsafe {
                d.async_copy_from(&h_locked, &self.stream)
                    .map_err(CudaBopError::Cuda)?;
            }
            Ok((d, Some(h_locked)))
        } else {
            DeviceBuffer::from_slice(src)
                .map(|d| (d, None))
                .map_err(CudaBopError::Cuda)
        }
    }
    pub fn new(device_id: usize) -> Result<Self, CudaBopError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let sm_count = device.get_attribute(DeviceAttribute::MultiprocessorCount)? as u32;

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/bop_kernel.ptx"));

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
            policy: CudaBopPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
            sm_count,
        })
    }

    #[inline]
    fn will_fit(required: usize, headroom: usize) -> Result<bool, CudaBopError> {
        let total = required
            .checked_add(headroom)
            .ok_or_else(|| CudaBopError::InvalidInput("size overflow".into()))?;
        if let Ok((free, _)) = mem_get_info() {
            Ok(total <= free)
        } else {
            Ok(true)
        }
    }

    pub fn context_arc(&self) -> Arc<Context> {
        self.context.clone()
    }
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    #[inline]
    pub fn set_policy(&mut self, p: CudaBopPolicy) {
        self.policy = p;
    }

    pub fn bop_batch_dev(
        &self,
        open: &[f32],
        high: &[f32],
        low: &[f32],
        close: &[f32],
    ) -> Result<DeviceArrayF32, CudaBopError> {
        let (first_valid, len) = Self::validate_ohlc_slices(open, high, low, close)?;

        let elems = 5usize
            .checked_mul(len)
            .ok_or_else(|| CudaBopError::InvalidInput("size overflow".into()))?;
        let bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaBopError::InvalidInput("size overflow".into()))?;
        let headroom = 64usize * 1024 * 1024;
        if !Self::will_fit(bytes, headroom)? {
            if let Ok((free, _)) = mem_get_info() {
                return Err(CudaBopError::OutOfMemory {
                    required: bytes,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaBopError::InvalidInput(
                    "insufficient device memory".into(),
                ));
            }
        }

        let mut _pinned_guards: Vec<LockedBuffer<f32>> = Vec::new();
        let (d_open, p0) = self.copy_h2d_maybe_async_f32(open)?;
        if let Some(h) = p0 {
            _pinned_guards.push(h);
        }
        let (d_high, p1) = self.copy_h2d_maybe_async_f32(high)?;
        if let Some(h) = p1 {
            _pinned_guards.push(h);
        }
        let (d_low, p2) = self.copy_h2d_maybe_async_f32(low)?;
        if let Some(h) = p2 {
            _pinned_guards.push(h);
        }
        let (d_close, p3) = self.copy_h2d_maybe_async_f32(close)?;
        if let Some(h) = p3 {
            _pinned_guards.push(h);
        }
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(len, &self.stream) }
                .map_err(CudaBopError::Cuda)?;

        self.launch_batch(
            &d_open,
            &d_high,
            &d_low,
            &d_close,
            len,
            first_valid,
            &mut d_out,
        )?;
        self.launch_batch(
            &d_open,
            &d_high,
            &d_low,
            &d_close,
            len,
            first_valid,
            &mut d_out,
        )?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: 1,
            cols: len,
        })
    }

    fn launch_batch(
        &self,
        d_open: &DeviceBuffer<f32>,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaBopError> {
        let func = self.module.get_function("bop_batch_f32").map_err(|_| {
            CudaBopError::MissingKernelSymbol {
                name: "bop_batch_f32",
            }
        })?;

        let block_x = self.policy.batch_block_x.unwrap_or(256);

        const ILP: u32 = 4;
        let work = ((len as u32) + block_x * ILP - 1) / (block_x * ILP);

        let max_grid = (self.sm_count.max(1)) * 32;
        let grid_x = work.min(max_grid).max(1);
        let max_threads_per_block = Device::get_device(self.device_id)?
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)?
            as u32;
        if block_x > max_threads_per_block {
            return Err(CudaBopError::LaunchConfigTooLarge {
                gx: grid_x,
                gy: 1,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut open_ptr = d_open.as_device_ptr().as_raw();
            let mut high_ptr = d_high.as_device_ptr().as_raw();
            let mut low_ptr = d_low.as_device_ptr().as_raw();
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut first_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut open_ptr as *mut _ as *mut c_void,
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(CudaBopError::Cuda)?;
        }
        unsafe {
            (*(self as *const _ as *mut CudaBop)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();
        self.stream.synchronize().map_err(CudaBopError::Cuda)
    }

    pub fn bop_many_series_one_param_time_major_dev(
        &self,
        open_tm: &[f32],
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
    ) -> Result<DeviceArrayF32, CudaBopError> {
        if cols == 0 || rows == 0 {
            return Err(CudaBopError::InvalidInput("invalid dims".into()));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaBopError::InvalidInput("rows*cols overflow".into()))?;
        if open_tm.len() != expected
            || high_tm.len() != expected
            || low_tm.len() != expected
            || close_tm.len() != expected
        {
            return Err(CudaBopError::InvalidInput(
                "time-major inputs length mismatch".into(),
            ));
        }

        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = -1i32;
            for t in 0..rows {
                let idx = t * cols + s;
                let o = open_tm[idx];
                let h = high_tm[idx];
                let l = low_tm[idx];
                let c = close_tm[idx];
                if !o.is_nan() && !h.is_nan() && !l.is_nan() && !c.is_nan() {
                    fv = t as i32;
                    break;
                }
            }
            first_valids[s] = fv;
        }

        let n = expected;
        let in_elems = 4usize
            .checked_mul(n)
            .ok_or_else(|| CudaBopError::InvalidInput("size overflow".into()))?;
        let in_bytes = in_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaBopError::InvalidInput("size overflow".into()))?;
        let out_bytes = n
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaBopError::InvalidInput("size overflow".into()))?;
        let first_bytes = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaBopError::InvalidInput("size overflow".into()))?;
        let bytes = in_bytes
            .checked_add(out_bytes)
            .and_then(|b| b.checked_add(first_bytes))
            .ok_or_else(|| CudaBopError::InvalidInput("size overflow".into()))?;
        let headroom = 64usize * 1024 * 1024;
        if !Self::will_fit(bytes, headroom)? {
            if let Ok((free, _)) = mem_get_info() {
                return Err(CudaBopError::OutOfMemory {
                    required: bytes,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaBopError::InvalidInput(
                    "insufficient device memory".into(),
                ));
            }
        }

        let mut _pinned_guards: Vec<LockedBuffer<f32>> = Vec::new();
        let (d_open, p0) = self.copy_h2d_maybe_async_f32(open_tm)?;
        if let Some(h) = p0 {
            _pinned_guards.push(h);
        }
        let (d_high, p1) = self.copy_h2d_maybe_async_f32(high_tm)?;
        if let Some(h) = p1 {
            _pinned_guards.push(h);
        }
        let (d_low, p2) = self.copy_h2d_maybe_async_f32(low_tm)?;
        if let Some(h) = p2 {
            _pinned_guards.push(h);
        }
        let (d_close, p3) = self.copy_h2d_maybe_async_f32(close_tm)?;
        if let Some(h) = p3 {
            _pinned_guards.push(h);
        }
        let d_first = DeviceBuffer::from_slice(&first_valids).map_err(CudaBopError::Cuda)?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(expected, &self.stream) }
                .map_err(CudaBopError::Cuda)?;

        self.launch_many_series(
            &d_open, &d_high, &d_low, &d_close, &d_first, cols, rows, &mut d_out,
        )?;
        self.launch_many_series(
            &d_open, &d_high, &d_low, &d_close, &d_first, cols, rows, &mut d_out,
        )?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    fn launch_many_series(
        &self,
        d_open_tm: &DeviceBuffer<f32>,
        d_high_tm: &DeviceBuffer<f32>,
        d_low_tm: &DeviceBuffer<f32>,
        d_close_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaBopError> {
        let func = self
            .module
            .get_function("bop_many_series_one_param_f32")
            .map_err(|_| CudaBopError::MissingKernelSymbol {
                name: "bop_many_series_one_param_f32",
            })?;
        let block_x = self.policy.many_block_x.unwrap_or(256);
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let max_threads_per_block = Device::get_device(self.device_id)?
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)?
            as u32;
        if block_x > max_threads_per_block {
            return Err(CudaBopError::LaunchConfigTooLarge {
                gx: grid_x,
                gy: 1,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut open_ptr = d_open_tm.as_device_ptr().as_raw();
            let mut high_ptr = d_high_tm.as_device_ptr().as_raw();
            let mut low_ptr = d_low_tm.as_device_ptr().as_raw();
            let mut close_ptr = d_close_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut nseries_i = cols as i32;
            let mut slen_i = rows as i32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut open_ptr as *mut _ as *mut c_void,
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut nseries_i as *mut _ as *mut c_void,
                &mut slen_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(CudaBopError::Cuda)?;
        }
        unsafe {
            (*(self as *const _ as *mut CudaBop)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();
        self.stream.synchronize().map_err(CudaBopError::Cuda)
    }

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
                    eprintln!("[DEBUG] BOP batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaBop)).debug_batch_logged = true;
                }
                unsafe {
                    (*(self as *const _ as *mut CudaBop)).debug_batch_logged = true;
                }
            }
        }
    }
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
                    eprintln!("[DEBUG] BOP many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaBop)).debug_many_logged = true;
                }
                unsafe {
                    (*(self as *const _ as *mut CudaBop)).debug_many_logged = true;
                }
            }
        }
    }

    fn validate_ohlc_slices(
        open: &[f32],
        high: &[f32],
        low: &[f32],
        close: &[f32],
    ) -> Result<(usize, usize), CudaBopError> {
        let len = open.len();
        if len == 0 || high.len() != len || low.len() != len || close.len() != len {
            return Err(CudaBopError::InvalidInput(
                "input slices are empty or mismatched".into(),
            ));
        }
        let first_valid = (0..len)
            .find(|&i| {
                !open[i].is_nan() && !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan()
            })
            .ok_or_else(|| CudaBopError::InvalidInput("all values are NaN".into()))?;
        Ok((first_valid, len))
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const MANY_COLS: usize = 1024;
    const MANY_ROWS: usize = 8192;

    fn bytes_one_series() -> usize {
        let in_bytes = 4 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series() -> usize {
        let n = MANY_COLS * MANY_ROWS;
        let in_bytes = 4 * n * std::mem::size_of::<f32>();
        let out_bytes = n * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct BopBatchDeviceState {
        cuda: CudaBop,
        d_open: DeviceBuffer<f32>,
        d_high: DeviceBuffer<f32>,
        d_low: DeviceBuffer<f32>,
        d_close: DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for BopBatchDeviceState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch(
                    &self.d_open,
                    &self.d_high,
                    &self.d_low,
                    &self.d_close,
                    self.len,
                    self.first_valid,
                    &mut self.d_out,
                )
                .expect("bop launch");
        }
    }
    fn prep_one_series_batch() -> Box<dyn CudaBenchState> {
        let cuda = CudaBop::new(0).expect("cuda bop");
        let mut open = gen_series(ONE_SERIES_LEN);
        let mut high = open.clone();
        let mut low = open.clone();
        let mut close = open.clone();
        for i in 4..ONE_SERIES_LEN {
            let x = i as f32 * 0.0023;
            open[i] = open[i] + 0.001 * x.sin();
            high[i] = open[i] + (0.5 + 0.05 * x.cos()).abs();
            low[i] = open[i] - (0.5 + 0.05 * x.sin()).abs();
            close[i] = open[i] + 0.2 * (x).sin();
        }

        let (first_valid, len) =
            CudaBop::validate_ohlc_slices(&open, &high, &low, &close).expect("validate");
        let d_open =
            unsafe { DeviceBuffer::from_slice_async(&open, &cuda.stream) }.expect("d_open H2D");
        let d_high =
            unsafe { DeviceBuffer::from_slice_async(&high, &cuda.stream) }.expect("d_high H2D");
        let d_low =
            unsafe { DeviceBuffer::from_slice_async(&low, &cuda.stream) }.expect("d_low H2D");
        let d_close =
            unsafe { DeviceBuffer::from_slice_async(&close, &cuda.stream) }.expect("d_close H2D");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(len, &cuda.stream) }.expect("d_out alloc");
        cuda.stream.synchronize().expect("bop prep sync");

        Box::new(BopBatchDeviceState {
            cuda,
            d_open,
            d_high,
            d_low,
            d_close,
            len,
            first_valid,
            d_out,
        })
    }

    struct BopManyDeviceState {
        cuda: CudaBop,
        d_open_tm: DeviceBuffer<f32>,
        d_high_tm: DeviceBuffer<f32>,
        d_low_tm: DeviceBuffer<f32>,
        d_close_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for BopManyDeviceState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series(
                    &self.d_open_tm,
                    &self.d_high_tm,
                    &self.d_low_tm,
                    &self.d_close_tm,
                    &self.d_first_valids,
                    MANY_COLS,
                    MANY_ROWS,
                    &mut self.d_out_tm,
                )
                .expect("bop many launch");
        }
    }
    fn prep_many_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaBop::new(0).expect("cuda bop");
        let n = MANY_COLS * MANY_ROWS;
        let mut base = gen_series(n);
        let mut open = vec![f32::NAN; n];
        let mut high = vec![f32::NAN; n];
        let mut low = vec![f32::NAN; n];
        let mut close = vec![f32::NAN; n];
        for s in 0..MANY_COLS {
            for t in s..MANY_ROWS {
                let idx = t * MANY_COLS + s;
                let x = (t as f32) * 0.002 + (s as f32) * 0.01;
                let b = base[idx];
                open[idx] = b + 0.001 * x.cos();
                high[idx] = b + 0.3 + 0.02 * x.sin();
                low[idx] = b - 0.3 - 0.02 * x.cos();
                close[idx] = b + 0.05 * x.sin();
            }
        }

        let mut first_valids = vec![0i32; MANY_COLS];
        for s in 0..MANY_COLS {
            let mut fv = -1i32;
            for t in 0..MANY_ROWS {
                let idx = t * MANY_COLS + s;
                let o = open[idx];
                let h = high[idx];
                let l = low[idx];
                let c = close[idx];
                if !o.is_nan() && !h.is_nan() && !l.is_nan() && !c.is_nan() {
                    fv = t as i32;
                    break;
                }
            }
            first_valids[s] = fv;
        }

        let d_open_tm =
            unsafe { DeviceBuffer::from_slice_async(&open, &cuda.stream) }.expect("d_open_tm H2D");
        let d_high_tm =
            unsafe { DeviceBuffer::from_slice_async(&high, &cuda.stream) }.expect("d_high_tm H2D");
        let d_low_tm =
            unsafe { DeviceBuffer::from_slice_async(&low, &cuda.stream) }.expect("d_low_tm H2D");
        let d_close_tm = unsafe { DeviceBuffer::from_slice_async(&close, &cuda.stream) }
            .expect("d_close_tm H2D");
        let d_first_valids = unsafe { DeviceBuffer::from_slice_async(&first_valids, &cuda.stream) }
            .expect("d_first_valids H2D");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(n, &cuda.stream) }.expect("d_out_tm alloc");
        cuda.stream.synchronize().expect("bop many prep sync");

        Box::new(BopManyDeviceState {
            cuda,
            d_open_tm,
            d_high_tm,
            d_low_tm,
            d_close_tm,
            d_first_valids,
            d_out_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "bop",
                "one_series_many_params",
                "bop_cuda_batch_dev",
                "1m_x_1",
                prep_one_series_batch,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series()),
            CudaBenchScenario::new(
                "bop",
                "many_series_one_param",
                "bop_cuda_many_series_one_param_dev",
                "1024x8192",
                prep_many_series,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_many_series()),
        ]
    }
}
