#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32;
use crate::indicators::qqe::{QqeBatchRange, QqeParams};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::error::CudaError;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaQqeError {
    #[error(transparent)]
    Cuda(#[from] CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
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
    #[error("device mismatch: buf={buf}, current={current}")]
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
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaQqe {
    module: Module,
    stream: Stream,
    ctx: Arc<Context>,
    device_id: u32,
    policy_batch: BatchKernelPolicy,
    policy_many: ManySeriesKernelPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaQqe {
    #[inline]
    fn warp_align(x: u32) -> u32 {
        let clamped = x.clamp(32, 1024);
        ((clamped + 31) / 32) * 32
    }
    pub fn new(device_id: usize) -> Result<Self, CudaQqeError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/qqe_kernel.ptx"));
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
            ctx: context,
            device_id: device_id as u32,
            policy_batch: BatchKernelPolicy::Auto,
            policy_many: ManySeriesKernelPolicy::Auto,
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    #[inline]
    pub fn set_batch_policy(&mut self, p: BatchKernelPolicy) {
        self.policy_batch = p;
    }
    #[inline]
    pub fn set_many_series_policy(&mut self, p: ManySeriesKernelPolicy) {
        self.policy_many = p;
    }
    #[inline]
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    #[inline]
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaQqeError> {
        self.stream.synchronize()?;
        Ok(())
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per = env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] QQE batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaQqe)).debug_batch_logged = true;
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
        if env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per = env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] QQE many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaQqe)).debug_many_logged = true;
                }
            }
        }
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaQqeError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Ok((free, _)) = mem_get_info() {
            let need = required_bytes.checked_add(headroom_bytes).ok_or_else(|| {
                CudaQqeError::InvalidInput(
                    "size overflow when adding headroom to required bytes".into(),
                )
            })?;
            if need > free {
                return Err(CudaQqeError::OutOfMemory {
                    required: required_bytes,
                    free,
                    headroom: headroom_bytes,
                });
            }
        }
        Ok(())
    }

    #[inline]
    fn validate_launch(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
    ) -> Result<(), CudaQqeError> {
        let dev = Device::get_device(self.device_id)?;
        let max_bx = dev.get_attribute(DeviceAttribute::MaxBlockDimX)? as u32;
        let max_by = dev.get_attribute(DeviceAttribute::MaxBlockDimY)? as u32;
        let max_bz = dev.get_attribute(DeviceAttribute::MaxBlockDimZ)? as u32;
        let max_gx = dev.get_attribute(DeviceAttribute::MaxGridDimX)? as u32;
        let max_gy = dev.get_attribute(DeviceAttribute::MaxGridDimY)? as u32;
        let max_gz = dev.get_attribute(DeviceAttribute::MaxGridDimZ)? as u32;

        let (gx, gy, gz) = grid;
        let (bx, by, bz) = block;
        if bx > max_bx || by > max_by || bz > max_bz || gx > max_gx || gy > max_gy || gz > max_gz {
            return Err(CudaQqeError::LaunchConfigTooLarge {
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

    fn first_valid_f32(series: &[f32]) -> Result<usize, CudaQqeError> {
        if series.is_empty() {
            return Err(CudaQqeError::InvalidInput("empty series".into()));
        }
        series
            .iter()
            .position(|x| x.is_finite())
            .ok_or_else(|| CudaQqeError::InvalidInput("all values are NaN".into()))
    }
    fn expand_grid(range: &QqeBatchRange) -> Vec<QqeParams> {
        fn axis_usize(t: (usize, usize, usize)) -> Vec<usize> {
            let (s, e, st) = t;
            if st == 0 || s == e {
                return vec![s];
            }
            if s < e {
                return (s..=e).step_by(st.max(1)).collect();
            }
            let mut v = Vec::new();
            let step = st.max(1);
            let mut cur = s;
            while cur >= e {
                v.push(cur);
                if cur < step {
                    break;
                }
                cur -= step;
                if cur == usize::MAX {
                    break;
                }
            }
            v
        }
        fn axis_f64(t: (f64, f64, f64)) -> Vec<f64> {
            let (s, e, st) = t;
            let step = if st.is_sign_negative() { -st } else { st };
            if step.abs() < 1e-12 || (s - e).abs() < 1e-12 {
                return vec![s];
            }
            let mut v = Vec::new();
            if s <= e {
                let mut x = s;
                while x <= e + 1e-12 {
                    v.push(x);
                    x += step;
                }
            } else {
                let mut x = s;
                while x + 1e-12 >= e {
                    v.push(x);
                    x -= step;
                }
            }
            v
        }
        let rs = axis_usize(range.rsi_period);
        let sm = axis_usize(range.smoothing_factor);
        let ff = axis_f64(range.fast_factor);
        let mut out = Vec::with_capacity(rs.len() * sm.len() * ff.len());
        for &r in &rs {
            for &s in &sm {
                for &k in &ff {
                    out.push(QqeParams {
                        rsi_period: Some(r),
                        smoothing_factor: Some(s),
                        fast_factor: Some(k),
                    });
                }
            }
        }
        out
    }

    pub fn qqe_batch_dev(
        &self,
        prices_f32: &[f32],
        sweep: &QqeBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<QqeParams>), CudaQqeError> {
        if prices_f32.is_empty() {
            return Err(CudaQqeError::InvalidInput("empty price input".into()));
        }
        let first_valid = Self::first_valid_f32(prices_f32)?;
        let combos = Self::expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaQqeError::InvalidInput("empty parameter sweep".into()));
        }
        let len = prices_f32.len();

        let mut worst_needed = 0usize;
        for c in &combos {
            let need = c.rsi_period.unwrap() + c.smoothing_factor.unwrap();
            worst_needed = worst_needed.max(need);
        }
        if len - first_valid < worst_needed {
            return Err(CudaQqeError::InvalidInput(
                "not enough valid data for warmup".into(),
            ));
        }
        for c in &combos {
            let need = c.rsi_period.unwrap() + c.smoothing_factor.unwrap();
            worst_needed = worst_needed.max(need);
        }
        if len - first_valid < worst_needed {
            return Err(CudaQqeError::InvalidInput(
                "not enough valid data for warmup".into(),
            ));
        }

        let rows = combos.len();
        let bytes_prices = len
            .checked_mul(4)
            .ok_or_else(|| CudaQqeError::InvalidInput("size overflow in prices bytes".into()))?;
        let bytes_params = rows
            .checked_mul(12)
            .ok_or_else(|| CudaQqeError::InvalidInput("size overflow in params bytes".into()))?;
        let elems_out = rows
            .checked_mul(len)
            .and_then(|x| x.checked_mul(2))
            .ok_or_else(|| CudaQqeError::InvalidInput("size overflow in output elements".into()))?;
        let bytes_out = elems_out
            .checked_mul(4)
            .ok_or_else(|| CudaQqeError::InvalidInput("size overflow in output bytes".into()))?;
        let required = bytes_prices
            .checked_add(bytes_params)
            .and_then(|x| x.checked_add(bytes_out))
            .ok_or_else(|| CudaQqeError::InvalidInput("size overflow in VRAM estimate".into()))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        let d_prices: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::from_slice_async(prices_f32, &self.stream) }?;
        let rsi_i32: Vec<i32> = combos
            .iter()
            .map(|c| c.rsi_period.unwrap() as i32)
            .collect();
        let ema_i32: Vec<i32> = combos
            .iter()
            .map(|c| c.smoothing_factor.unwrap() as i32)
            .collect();
        let fast_f32: Vec<f32> = combos
            .iter()
            .map(|c| c.fast_factor.unwrap() as f32)
            .collect();

        let d_rsi: DeviceBuffer<i32> =
            unsafe { DeviceBuffer::from_slice_async(&rsi_i32, &self.stream) }?;
        let d_ema: DeviceBuffer<i32> =
            unsafe { DeviceBuffer::from_slice_async(&ema_i32, &self.stream) }?;
        let d_fast: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::from_slice_async(&fast_f32, &self.stream) }?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems_out, &self.stream) }?;

        let mut block_x = match self.policy_batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            BatchKernelPolicy::Auto => 256,
        };
        block_x = Self::warp_align(block_x);

        let func = self.module.get_function("qqe_batch_f32").map_err(|_| {
            CudaQqeError::MissingKernelSymbol {
                name: "qqe_batch_f32",
            }
        })?;
        const MAX_Y: usize = 65_535;
        let mut base = 0usize;
        while base < rows {
            let take = (rows - base).min(MAX_Y);
            unsafe {
                let mut f_prices = d_prices.as_device_ptr().as_raw();
                let mut f_rsi = d_rsi.as_device_ptr().add(base).as_raw();
                let mut f_ema = d_ema.as_device_ptr().add(base).as_raw();
                let mut f_fast = d_fast.as_device_ptr().add(base).as_raw();
                let mut series_len_i = len as i32;
                let mut n_combos_i = take as i32;
                let mut first_i = first_valid as i32;
                let row_offset_elems = 2 * base * len;
                let mut f_out = d_out.as_device_ptr().add(row_offset_elems).as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut f_prices as *mut _ as *mut c_void,
                    &mut f_rsi as *mut _ as *mut c_void,
                    &mut f_ema as *mut _ as *mut c_void,
                    &mut f_fast as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut n_combos_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut f_out as *mut _ as *mut c_void,
                ];
                let grid_dims = (1u32, take as u32, 1u32);
                let block_dims = (block_x, 1u32, 1u32);
                Self::validate_launch(self, grid_dims, block_dims)?;
                let grid: GridSize = grid_dims.into();
                let block: BlockSize = block_dims.into();
                self.stream.launch(&func, grid, block, 0, args)?;
            }
            unsafe {
                (*(self as *const _ as *mut CudaQqe)).last_batch =
                    Some(BatchKernelSelected::Plain { block_x });
            }
            self.maybe_log_batch_debug();
            base += take;
        }
        self.stream.synchronize()?;
        Ok((
            DeviceArrayF32 {
                buf: d_out,
                rows: 2 * rows,
                cols: len,
            },
            combos,
        ))
    }

    pub fn qqe_many_series_one_param_time_major_dev(
        &self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &QqeParams,
    ) -> Result<DeviceArrayF32, CudaQqeError> {
        if cols == 0 || rows == 0 {
            return Err(CudaQqeError::InvalidInput("cols/rows must be > 0".into()));
        }
        if prices_tm_f32.len() != cols * rows {
            return Err(CudaQqeError::InvalidInput(
                "data length != cols*rows".into(),
            ));
        }
        let rsi_p = params.rsi_period.unwrap_or(0);
        let ema_p = params.smoothing_factor.unwrap_or(0);
        let fast_k = params.fast_factor.unwrap_or(4.236) as f32;
        if rsi_p == 0 || ema_p == 0 {
            return Err(CudaQqeError::InvalidInput("invalid rsi/ema period".into()));
        }

        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let v = prices_tm_f32[t * cols + s];

                if v.is_finite() {
                    fv = Some(t);
                    break;
                }
            }
            let fv =
                fv.ok_or_else(|| CudaQqeError::InvalidInput(format!("series {} all NaN", s)))?;
            if rows - fv < rsi_p + ema_p {
                return Err(CudaQqeError::InvalidInput(
                    "not enough valid data per series".into(),
                ));
            }
            first_valids[s] = fv as i32;
        }

        let bytes_series = cols
            .checked_mul(rows)
            .and_then(|x| x.checked_mul(4))
            .ok_or_else(|| CudaQqeError::InvalidInput("size overflow in series bytes".into()))?;
        let bytes_first = cols.checked_mul(4).ok_or_else(|| {
            CudaQqeError::InvalidInput("size overflow in first-valid bytes".into())
        })?;
        let elems_out = rows
            .checked_mul(2)
            .and_then(|x| x.checked_mul(cols))
            .ok_or_else(|| CudaQqeError::InvalidInput("size overflow in output elements".into()))?;
        let bytes_out = elems_out
            .checked_mul(4)
            .ok_or_else(|| CudaQqeError::InvalidInput("size overflow in output bytes".into()))?;
        let required = bytes_series
            .checked_add(bytes_first)
            .and_then(|x| x.checked_add(bytes_out))
            .ok_or_else(|| CudaQqeError::InvalidInput("size overflow in VRAM estimate".into()))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;
        let d_prices: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::from_slice_async(prices_tm_f32, &self.stream) }?;
        let d_first: DeviceBuffer<i32> =
            unsafe { DeviceBuffer::from_slice_async(&first_valids, &self.stream) }?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems_out, &self.stream) }?;

        let warm_max = (0..cols)
            .map(|s| {
                (first_valids[s] as usize)
                    .saturating_add(rsi_p + ema_p)
                    .saturating_sub(2)
            })
            .max()
            .unwrap_or(0);
        let mut block_x = match self.policy_many {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            ManySeriesKernelPolicy::Auto => {
                if warm_max < 128 {
                    128
                } else if warm_max < 512 {
                    256
                } else {
                    512
                }
            }
        };
        block_x = Self::warp_align(block_x);

        let func = self
            .module
            .get_function("qqe_many_series_one_param_time_major_f32")
            .map_err(|_| CudaQqeError::MissingKernelSymbol {
                name: "qqe_many_series_one_param_time_major_f32",
            })?;
        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut rsi_i = rsi_p as i32;
            let mut ema_i = ema_p as i32;
            let mut fast_k_f = fast_k as f32;
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut first_ptr = d_first.as_device_ptr().as_raw();
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut rsi_i as *mut _ as *mut c_void,
                &mut ema_i as *mut _ as *mut c_void,
                &mut fast_k_f as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            let grid_dims = (1u32, cols as u32, 1u32);
            let block_dims = (block_x, 1u32, 1u32);
            Self::validate_launch(self, grid_dims, block_dims)?;
            let grid: GridSize = grid_dims.into();
            let block: BlockSize = block_dims.into();
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        unsafe {
            (*(self as *const _ as *mut CudaQqe)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        unsafe {
            (*(self as *const _ as *mut CudaQqe)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();
        self.stream.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols: 2 * cols,
        })
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 192;
    const MANY_COLS: usize = 256;
    const MANY_ROWS: usize = 16 * 1024;

    fn gen_series(n: usize) -> Vec<f32> {
        let mut v = vec![f32::NAN; n];
        for i in 64..n {
            let x = i as f32;
            v[i] = (x * 0.00123).sin() + 0.00025 * x;
        }
        v
    }

    struct BatchState {
        cuda: CudaQqe,
        d_prices: DeviceBuffer<f32>,
        d_rsi: DeviceBuffer<i32>,
        d_ema: DeviceBuffer<i32>,
        d_fast: DeviceBuffer<f32>,
        d_out: DeviceBuffer<f32>,
        len: usize,
        rows: usize,
        first_valid: usize,
        block_x: u32,
    }
    impl CudaBenchState for BatchState {
        fn launch(&mut self) {
            let func = self
                .cuda
                .module
                .get_function("qqe_batch_f32")
                .expect("qqe_batch_f32");
            const MAX_Y: usize = 65_535;
            let mut base = 0usize;
            while base < self.rows {
                let take = (self.rows - base).min(MAX_Y);
                unsafe {
                    let mut f_prices = self.d_prices.as_device_ptr().as_raw();
                    let mut f_rsi = self.d_rsi.as_device_ptr().add(base).as_raw();
                    let mut f_ema = self.d_ema.as_device_ptr().add(base).as_raw();
                    let mut f_fast = self.d_fast.as_device_ptr().add(base).as_raw();
                    let mut series_len_i = self.len as i32;
                    let mut n_combos_i = take as i32;
                    let mut first_i = self.first_valid as i32;
                    let row_offset_elems = 2 * base * self.len;
                    let mut f_out = self.d_out.as_device_ptr().add(row_offset_elems).as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut f_prices as *mut _ as *mut c_void,
                        &mut f_rsi as *mut _ as *mut c_void,
                        &mut f_ema as *mut _ as *mut c_void,
                        &mut f_fast as *mut _ as *mut c_void,
                        &mut series_len_i as *mut _ as *mut c_void,
                        &mut n_combos_i as *mut _ as *mut c_void,
                        &mut first_i as *mut _ as *mut c_void,
                        &mut f_out as *mut _ as *mut c_void,
                    ];
                    let grid_dims = (1u32, take as u32, 1u32);
                    let block_dims = (self.block_x, 1u32, 1u32);
                    self.cuda
                        .validate_launch(grid_dims, block_dims)
                        .expect("launch dims");
                    let grid: GridSize = grid_dims.into();
                    let block: BlockSize = block_dims.into();
                    self.cuda
                        .stream
                        .launch(&func, grid, block, 0, args)
                        .expect("launch qqe_batch_f32");
                }
                base += take;
            }
            let _ = self.cuda.stream.synchronize();
        }
    }

    struct ManyState {
        cuda: CudaQqe,
        d_prices: DeviceBuffer<f32>,
        d_first: DeviceBuffer<i32>,
        d_out: DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        rsi_p: usize,
        ema_p: usize,
        fast_k: f32,
        block_x: u32,
    }
    impl CudaBenchState for ManyState {
        fn launch(&mut self) {
            let func = self
                .cuda
                .module
                .get_function("qqe_many_series_one_param_time_major_f32")
                .expect("qqe_many_series_one_param_time_major_f32");
            unsafe {
                let mut prices_ptr = self.d_prices.as_device_ptr().as_raw();
                let mut rsi_i = self.rsi_p as i32;
                let mut ema_i = self.ema_p as i32;
                let mut fast_k_f = self.fast_k;
                let mut num_series_i = self.cols as i32;
                let mut series_len_i = self.rows as i32;
                let mut first_ptr = self.d_first.as_device_ptr().as_raw();
                let mut out_ptr = self.d_out.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut rsi_i as *mut _ as *mut c_void,
                    &mut ema_i as *mut _ as *mut c_void,
                    &mut fast_k_f as *mut _ as *mut c_void,
                    &mut num_series_i as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut first_ptr as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                let grid_dims = (1u32, self.cols as u32, 1u32);
                let block_dims = (self.block_x, 1u32, 1u32);
                self.cuda
                    .validate_launch(grid_dims, block_dims)
                    .expect("launch dims");
                let grid: GridSize = grid_dims.into();
                let block: BlockSize = block_dims.into();
                self.cuda
                    .stream
                    .launch(&func, grid, block, 0, args)
                    .expect("launch qqe_many_series_one_param_time_major_f32");
            }
            let _ = self.cuda.stream.synchronize();
        }
    }

    fn prep_batch() -> Box<dyn CudaBenchState> {
        let cuda = CudaQqe::new(0).expect("cuda qqe");
        let prices = gen_series(ONE_SERIES_LEN);
        let sweep = QqeBatchRange {
            rsi_period: (8, 8 + PARAM_SWEEP - 1, 1),
            smoothing_factor: (5, 5, 0),
            fast_factor: (4.236, 4.236, 0.0),
        };
        let first_valid = CudaQqe::first_valid_f32(&prices).expect("first_valid_f32");
        let combos = CudaQqe::expand_grid(&sweep);
        let rsi_i32: Vec<i32> = combos
            .iter()
            .map(|c| c.rsi_period.unwrap() as i32)
            .collect();
        let ema_i32: Vec<i32> = combos
            .iter()
            .map(|c| c.smoothing_factor.unwrap() as i32)
            .collect();
        let fast_f32: Vec<f32> = combos
            .iter()
            .map(|c| c.fast_factor.unwrap() as f32)
            .collect();
        let d_prices = DeviceBuffer::from_slice(&prices).expect("d_prices");
        let d_rsi = DeviceBuffer::from_slice(&rsi_i32).expect("d_rsi");
        let d_ema = DeviceBuffer::from_slice(&ema_i32).expect("d_ema");
        let d_fast = DeviceBuffer::from_slice(&fast_f32).expect("d_fast");
        let rows = combos.len();
        let out_elems = 2 * rows * ONE_SERIES_LEN;
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(out_elems) }.expect("d_out");
        let mut block_x = match cuda.policy_batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            BatchKernelPolicy::Auto => 256,
        };
        block_x = CudaQqe::warp_align(block_x);
        cuda.stream.synchronize().expect("sync after prep");
        Box::new(BatchState {
            cuda,
            d_prices,
            d_rsi,
            d_ema,
            d_fast,
            d_out,
            len: ONE_SERIES_LEN,
            rows,
            first_valid,
            block_x,
        })
    }
    fn prep_many() -> Box<dyn CudaBenchState> {
        let cuda = CudaQqe::new(0).expect("cuda qqe");
        let cols = MANY_COLS;
        let rows = MANY_ROWS;
        let mut tm = vec![f32::NAN; cols * rows];
        for s in 0..cols {
            for t in s..rows {
                let x = (t as f32) + 0.1 * (s as f32);
                tm[t * cols + s] = (0.002 * x).sin() + 0.0003 * x;
            }
        }
        let params = QqeParams {
            rsi_period: Some(14),
            smoothing_factor: Some(5),
            fast_factor: Some(4.236),
        };
        let (rsi_p, ema_p, fast_k) = (
            params.rsi_period.unwrap(),
            params.smoothing_factor.unwrap(),
            params.fast_factor.unwrap() as f32,
        );
        let first_valids: Vec<i32> = (0..cols).map(|s| s as i32).collect();
        let d_prices = DeviceBuffer::from_slice(&tm).expect("d_prices");
        let d_first = DeviceBuffer::from_slice(&first_valids).expect("d_first");
        let out_elems = rows * 2 * cols;
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(out_elems) }.expect("d_out");
        let warm_max = (0..cols)
            .map(|s| {
                (first_valids[s] as usize)
                    .saturating_add(rsi_p + ema_p)
                    .saturating_sub(2)
            })
            .max()
            .unwrap_or(0);
        let mut block_x = match cuda.policy_many {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            ManySeriesKernelPolicy::Auto => {
                if warm_max < 128 {
                    128
                } else if warm_max < 512 {
                    256
                } else {
                    512
                }
            }
        };
        block_x = CudaQqe::warp_align(block_x);
        cuda.stream.synchronize().expect("sync after prep");
        Box::new(ManyState {
            cuda,
            d_prices,
            d_first,
            d_out,
            cols,
            rows,
            rsi_p,
            ema_p,
            fast_k,
            block_x,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "qqe",
                "one_series_many_params",
                "qqe_cuda_batch_dev",
                "1m_x_192",
                prep_batch,
            )
            .with_sample_size(12)
            .with_mem_required(
                ONE_SERIES_LEN * 4 + (2 * PARAM_SWEEP * ONE_SERIES_LEN) * 4 + 64 * 1024 * 1024,
            ),
            CudaBenchScenario::new(
                "qqe",
                "many_series_one_param",
                "qqe_cuda_many_series_one_param_dev",
                "256x16k",
                prep_many,
            )
            .with_sample_size(12)
            .with_mem_required(MANY_COLS * MANY_ROWS * 3 * 4 + 64 * 1024 * 1024),
        ]
    }
}
