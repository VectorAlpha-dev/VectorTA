#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32;
use cust::context::{CacheConfig, Context};
use cust::device::Device;
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use thiserror::Error;

const H2D_PINNED_THRESHOLD_BYTES: usize = 128 * 1024 * 1024;

#[inline]
fn round_block_x_to_warp(x: u32) -> u32 {
    const WARP: u32 = 32;
    let y = (x / WARP) * WARP;
    if y == 0 {
        WARP
    } else {
        y.min(1024)
    }
}

#[inline]
fn is_triplet_valid(h: f32, l: f32, v: f32) -> bool {
    !(h.is_nan() || l.is_nan() || v.is_nan())
}

#[derive(Error, Debug)]
pub enum CudaEmvError {
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
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaEmvPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaEmvPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaEmv {
    module: Module,
    stream: Stream,
    context: std::sync::Arc<Context>,
    device_id: u32,
    policy: CudaEmvPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaEmv {
    pub fn new(device_id: usize) -> Result<Self, CudaEmvError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = std::sync::Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/emv_kernel.ptx"));
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

        let _ = cust::context::CurrentContext::set_cache_config(CacheConfig::PreferL1);

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy: CudaEmvPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn set_policy(&mut self, policy: CudaEmvPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaEmvPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
    pub fn synchronize(&self) -> Result<(), CudaEmvError> {
        Ok(self.stream.synchronize()?)
    }
    pub fn context_arc(&self) -> std::sync::Arc<Context> {
        self.context.clone()
    }
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

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
                    eprintln!("[DEBUG] EMV batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaEmv)).debug_batch_logged = true;
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
                    eprintln!("[DEBUG] EMV many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaEmv)).debug_many_logged = true;
                }
            }
        }
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
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
        if let Some((free, _)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    #[inline]
    fn copy_to_device_adaptive_f32(&self, host: &[f32]) -> Result<DeviceBuffer<f32>, CudaEmvError> {
        let elem = std::mem::size_of::<f32>();
        let bytes = host
            .len()
            .checked_mul(elem)
            .ok_or_else(|| CudaEmvError::InvalidInput("byte size overflow".into()))?;
        if bytes >= H2D_PINNED_THRESHOLD_BYTES {
            Ok(DeviceBuffer::from_slice(host)?)
        } else {
            let pinned = LockedBuffer::from_slice(host)?;
            let mut dev =
                unsafe { DeviceBuffer::<f32>::uninitialized_async(host.len(), &self.stream) }?;
            unsafe {
                dev.async_copy_from(&pinned, &self.stream)?;
            }
            Ok(dev)
        }
    }

    #[inline]
    fn assert_current_device(&self) -> Result<(), CudaEmvError> {
        unsafe {
            let mut dev: i32 = -1;
            cust::sys::cuCtxGetDevice(&mut dev);
            if dev < 0 {
                return Ok(());
            }
            let cur = dev as u32;
            if cur != self.device_id {
                return Err(CudaEmvError::DeviceMismatch {
                    buf: self.device_id,
                    current: cur,
                });
            }
        }
        Ok(())
    }

    fn validate_batch_inputs(
        high: &[f32],
        low: &[f32],
        volume: &[f32],
    ) -> Result<(usize, usize), CudaEmvError> {
        if high.is_empty() || low.is_empty() || volume.is_empty() {
            return Err(CudaEmvError::InvalidInput("empty input slices".into()));
        }
        let len = high.len();
        if low.len() != len || volume.len() != len {
            return Err(CudaEmvError::InvalidInput(
                "input slice length mismatch".into(),
            ));
        }
        let first = (0..len)
            .find(|&i| is_triplet_valid(high[i], low[i], volume[i]))
            .ok_or_else(|| CudaEmvError::InvalidInput("all values are NaN".into()))?;
        let has_second = (first + 1..len).any(|i| is_triplet_valid(high[i], low[i], volume[i]));
        if !has_second {
            return Err(CudaEmvError::InvalidInput(
                "not enough valid data: need at least 2".into(),
            ));
        }
        Ok((first, len))
    }

    fn launch_batch_kernel(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_volume: &DeviceBuffer<f32>,
        series_len: usize,
        n_rows: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEmvError> {
        let mut func: Function = self.module.get_function("emv_batch_f32").map_err(|_| {
            CudaEmvError::MissingKernelSymbol {
                name: "emv_batch_f32",
            }
        })?;

        let _ = func.set_cache_config(CacheConfig::PreferL1);

        let mut block_x: u32 = match std::env::var("EMV_BLOCK_X").ok().as_deref() {
            Some("auto") | None => {
                let (_min_grid, suggested) = func
                    .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
                    .map_err(CudaEmvError::Cuda)?;
                suggested
            }
            Some(s) => s.parse::<u32>().ok().filter(|&v| v > 0).unwrap_or(128),
        };
        block_x = round_block_x_to_warp(block_x);
        let grid_x = ((n_rows as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        if block_x > 1024 || grid_x == 0 {
            return Err(CudaEmvError::LaunchConfigTooLarge {
                gx: grid_x,
                gy: 1,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }

        unsafe {
            let mut high_ptr = d_high.as_device_ptr().as_raw();
            let mut low_ptr = d_low.as_device_ptr().as_raw();
            let mut vol_ptr = d_volume.as_device_ptr().as_raw();
            let mut len_i = series_len as i32;
            let mut combos_i = n_rows as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut vol_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        unsafe {
            let this = self as *const _ as *mut CudaEmv;
            (*this).last_batch = Some(BatchKernelSelected::OneD { block_x });
        }
        self.maybe_log_batch_debug();

        Ok(())
    }

    pub fn emv_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
        volume: &[f32],
    ) -> Result<DeviceArrayF32, CudaEmvError> {
        let (first, len) = Self::validate_batch_inputs(high, low, volume)?;
        let _ = self.assert_current_device();

        let elem = std::mem::size_of::<f32>();
        let in_elems = len
            .checked_mul(3)
            .and_then(|v| v.checked_add(len))
            .ok_or_else(|| CudaEmvError::InvalidInput("element count overflow".into()))?;
        let bytes = in_elems
            .checked_mul(elem)
            .ok_or_else(|| CudaEmvError::InvalidInput("byte size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(bytes, headroom) {
            if let Some((free, _)) = Self::device_mem_info() {
                return Err(CudaEmvError::OutOfMemory {
                    required: bytes,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaEmvError::InvalidInput(
                    "insufficient VRAM for EMV batch".into(),
                ));
            }
        }

        let d_high = self.copy_to_device_adaptive_f32(high)?;
        let d_low = self.copy_to_device_adaptive_f32(low)?;
        let d_vol = self.copy_to_device_adaptive_f32(volume)?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized_async(len, &self.stream) }?;

        self.launch_batch_kernel(&d_high, &d_low, &d_vol, len, 1, first, &mut d_out)?;

        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: 1,
            cols: len,
        })
    }

    fn prepare_first_valids_hlv_tm(
        high_tm: &[f32],
        low_tm: &[f32],
        vol_tm: &[f32],
        cols: usize,
        rows: usize,
    ) -> Result<Vec<i32>, CudaEmvError> {
        if cols == 0 || rows == 0 {
            return Err(CudaEmvError::InvalidInput(
                "matrix dimensions must be positive".into(),
            ));
        }
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaEmvError::InvalidInput("rows*cols overflow".into()))?;
        if high_tm.len() != elems || low_tm.len() != elems || vol_tm.len() != elems {
            return Err(CudaEmvError::InvalidInput("matrix shape mismatch".into()));
        }

        let mut first = vec![-1i32; cols];
        let mut have_second = vec![false; cols];
        let mut remaining_first = cols;
        let mut remaining_second = cols;

        'rowsweep: for r in 0..rows {
            let base = r * cols;
            for s in 0..cols {
                if first[s] >= 0 && have_second[s] {
                    continue;
                }
                let idx = base + s;
                if is_triplet_valid(high_tm[idx], low_tm[idx], vol_tm[idx]) {
                    if first[s] < 0 {
                        first[s] = r as i32;
                        remaining_first -= 1;
                    } else if !have_second[s] {
                        have_second[s] = true;
                        remaining_second -= 1;
                    }
                }
            }
            if remaining_first == 0 && remaining_second == 0 {
                break 'rowsweep;
            }
        }
        for s in 0..cols {
            if first[s] < 0 {
                return Err(CudaEmvError::InvalidInput(format!(
                    "all NaN in series {}",
                    s
                )));
            }
            if !have_second[s] {
                return Err(CudaEmvError::InvalidInput(format!(
                    "not enough valid data in series {}: need >=2",
                    s
                )));
            }
        }
        Ok(first)
    }

    fn launch_many_series_kernel(
        &self,
        d_high_tm: &DeviceBuffer<f32>,
        d_low_tm: &DeviceBuffer<f32>,
        d_vol_tm: &DeviceBuffer<f32>,
        d_first: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEmvError> {
        let mut func: Function = self
            .module
            .get_function("emv_many_series_one_param_f32")
            .map_err(|_| CudaEmvError::MissingKernelSymbol {
                name: "emv_many_series_one_param_f32",
            })?;

        let _ = func.set_cache_config(CacheConfig::PreferL1);

        let mut block_x: u32 = match std::env::var("EMV_BLOCK_X").ok().as_deref() {
            Some("auto") | None => {
                let (_min_grid, suggested) = func
                    .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
                    .map_err(CudaEmvError::Cuda)?;
                suggested
            }
            Some(s) => s.parse::<u32>().ok().filter(|&v| v > 0).unwrap_or(256),
        };
        block_x = round_block_x_to_warp(block_x);
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        if block_x > 1024 || grid_x == 0 {
            return Err(CudaEmvError::LaunchConfigTooLarge {
                gx: grid_x,
                gy: 1,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }

        unsafe {
            let mut high_ptr = d_high_tm.as_device_ptr().as_raw();
            let mut low_ptr = d_low_tm.as_device_ptr().as_raw();
            let mut vol_ptr = d_vol_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut vol_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        unsafe {
            let this = self as *const _ as *mut CudaEmv;
            (*this).last_many = Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();
        Ok(())
    }

    pub fn emv_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        vol_tm: &[f32],
        cols: usize,
        rows: usize,
    ) -> Result<DeviceArrayF32, CudaEmvError> {
        let first_valids = Self::prepare_first_valids_hlv_tm(high_tm, low_tm, vol_tm, cols, rows)?;

        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaEmvError::InvalidInput("rows*cols overflow".into()))?;
        let elem = std::mem::size_of::<f32>();
        let total_f32_elems = elems
            .checked_mul(3)
            .and_then(|v| v.checked_add(elems))
            .ok_or_else(|| CudaEmvError::InvalidInput("element count overflow".into()))?;
        let bytes_f32 = total_f32_elems
            .checked_mul(elem)
            .ok_or_else(|| CudaEmvError::InvalidInput("byte size overflow".into()))?;
        let bytes_i32 = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaEmvError::InvalidInput("byte size overflow".into()))?;
        let bytes = bytes_f32
            .checked_add(bytes_i32)
            .ok_or_else(|| CudaEmvError::InvalidInput("byte size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(bytes, headroom) {
            if let Some((free, _)) = Self::device_mem_info() {
                return Err(CudaEmvError::OutOfMemory {
                    required: bytes,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaEmvError::InvalidInput(
                    "insufficient VRAM for EMV many-series".into(),
                ));
            }
        }

        let d_high_tm = self.copy_to_device_adaptive_f32(high_tm)?;
        let d_low_tm = self.copy_to_device_adaptive_f32(low_tm)?;
        let d_vol_tm = self.copy_to_device_adaptive_f32(vol_tm)?;
        let d_first = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }?;

        self.launch_many_series_kernel(
            &d_high_tm,
            &d_low_tm,
            &d_vol_tm,
            &d_first,
            cols,
            rows,
            &mut d_out_tm,
        )?;
        self.launch_many_series_kernel(
            &d_high_tm,
            &d_low_tm,
            &d_vol_tm,
            &d_first,
            cols,
            rows,
            &mut d_out_tm,
        )?;
        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows,
            cols,
        })
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices, gen_time_major_volumes};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const MANY_SERIES_COLS: usize = 256;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series() -> usize {
        let in_bytes = 3 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = 3 * elems * std::mem::size_of::<f32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    fn synth_hl_from_price(price: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut h = price.to_vec();
        let mut l = price.to_vec();
        for i in 0..price.len() {
            let v = price[i];
            if v.is_nan() {
                continue;
            }
            if v.is_nan() {
                continue;
            }
            let x = i as f32 * 0.0019;
            let off = (0.0027 * x.cos()).abs() + 0.07;
            h[i] = v + off;
            l[i] = v - off;
        }
        (h, l)
    }
    fn synth_volume(len: usize) -> Vec<f32> {
        let mut v = vec![f32::NAN; len];
        for i in 7..len {
            let x = i as f32 * 0.0063;
            v[i] = ((x.sin().abs() + 0.9) * 400.0) + 50.0;
        }
        v
    }

    struct BatchDeviceState {
        cuda: CudaEmv,
        d_high: DeviceBuffer<f32>,
        d_low: DeviceBuffer<f32>,
        d_vol: DeviceBuffer<f32>,
        d_out: DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
    }
    impl CudaBenchState for BatchDeviceState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_high,
                    &self.d_low,
                    &self.d_vol,
                    self.len,
                    1,
                    self.first_valid,
                    &mut self.d_out,
                )
                .expect("emv launch_batch_kernel");
            self.cuda.synchronize().expect("emv sync");
        }
    }
    fn prep_one_series() -> Box<dyn CudaBenchState> {
        let price = gen_series(ONE_SERIES_LEN);
        let (high, low) = synth_hl_from_price(&price);
        let vol = synth_volume(ONE_SERIES_LEN);
        let (first_valid, len) =
            CudaEmv::validate_batch_inputs(&high, &low, &vol).expect("emv validate_batch_inputs");

        let cuda = CudaEmv::new(0).expect("cuda");
        let d_high =
            unsafe { DeviceBuffer::from_slice_async(&high, &cuda.stream) }.expect("d_high");
        let d_low = unsafe { DeviceBuffer::from_slice_async(&low, &cuda.stream) }.expect("d_low");
        let d_vol = unsafe { DeviceBuffer::from_slice_async(&vol, &cuda.stream) }.expect("d_vol");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(len, &cuda.stream) }.expect("d_out");
        cuda.synchronize().expect("sync after prep");
        Box::new(BatchDeviceState {
            cuda,
            d_high,
            d_low,
            d_vol,
            d_out,
            len,
            first_valid,
        })
    }

    struct ManySeriesDeviceState {
        cuda: CudaEmv,
        d_high_tm: DeviceBuffer<f32>,
        d_low_tm: DeviceBuffer<f32>,
        d_vol_tm: DeviceBuffer<f32>,
        d_first: DeviceBuffer<i32>,
        d_out_tm: DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
    }
    impl CudaBenchState for ManySeriesDeviceState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series_kernel(
                    &self.d_high_tm,
                    &self.d_low_tm,
                    &self.d_vol_tm,
                    &self.d_first,
                    self.cols,
                    self.rows,
                    &mut self.d_out_tm,
                )
                .expect("emv launch_many_series_kernel");
            self.cuda.synchronize().expect("emv sync");
        }
    }
    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let price_tm = gen_time_major_prices(cols, rows);
        let (high_tm, low_tm) = synth_hl_from_price(&price_tm);
        let vol_tm = gen_time_major_volumes(cols, rows);

        let first_valids: Vec<i32> = (0..cols).map(|i| i as i32).collect();

        let cuda = CudaEmv::new(0).expect("cuda");
        let d_high_tm =
            unsafe { DeviceBuffer::from_slice_async(&high_tm, &cuda.stream) }.expect("d_high_tm");
        let d_low_tm =
            unsafe { DeviceBuffer::from_slice_async(&low_tm, &cuda.stream) }.expect("d_low_tm");
        let d_vol_tm =
            unsafe { DeviceBuffer::from_slice_async(&vol_tm, &cuda.stream) }.expect("d_vol_tm");
        let d_first = DeviceBuffer::from_slice(&first_valids).expect("d_first");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out_tm");
        cuda.synchronize().expect("sync after prep");
        Box::new(ManySeriesDeviceState {
            cuda,
            d_high_tm,
            d_low_tm,
            d_vol_tm,
            d_first,
            d_out_tm,
            cols,
            rows,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "emv",
                "one_series",
                "emv_cuda_batch_dev",
                "1m",
                prep_one_series,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series()),
            CudaBenchScenario::new(
                "emv",
                "many_series_one_param",
                "emv_cuda_many_series_one_param",
                "256x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
