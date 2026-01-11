//! CUDA wrapper for Fisher Transform (oscillator).
//!
//! Parity with ALMA-style wrappers:
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/fisher_kernel.ptx"))
//! - Non-blocking stream; DetermineTargetFromContext + O2 with fallbacks
//! - VRAM checks with ~64MB headroom; simple grid policies
//! - Public device entry points:
//!     - `fisher_batch_dev(&[f32],[f32], &FisherBatchRange)` -> (DeviceFisherPair, Vec<FisherParams>)
//!     - `fisher_many_series_one_param_time_major_dev(&[f32],[f32], cols, rows, period)` -> DeviceFisherPair
//!
//! Numeric rules mirror scalar implementation in src/indicators/fisher.rs.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::fisher::{FisherBatchRange, FisherParams};
use cust::context::{CacheConfig, Context};
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::sys;
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CudaFisherError {
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
pub struct CudaFisherPolicy {
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

/// Pair of VRAM-backed arrays produced by the Fisher kernels (fisher + signal).
pub struct DeviceFisherPair {
    pub fisher: DeviceArrayF32,
    pub signal: DeviceArrayF32,
}

impl DeviceFisherPair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.fisher.rows
    }
    #[inline]
    pub fn cols(&self) -> usize {
        self.fisher.cols
    }
}

pub struct CudaFisher {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaFisherPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaFisher {
    pub fn new(device_id: usize) -> Result<Self, CudaFisherError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/fisher_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => match Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                Ok(m) => m,
                Err(_) => Module::from_ptx(ptx, &[])?
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy: CudaFisherPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn context_arc(&self) -> Arc<Context> { self.context.clone() }
    pub fn device_id(&self) -> u32 { self.device_id }

    pub fn set_policy(&mut self, policy: CudaFisherPolicy) {
        self.policy = policy;
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }
    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> { mem_get_info().ok() }
    #[inline]
    fn ensure_will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaFisherError> {
        if !Self::mem_check_enabled() { return Ok(()); }
        if let Some((free, _total)) = Self::device_mem_info() {
            if required_bytes.saturating_add(headroom_bytes) <= free { Ok(()) } else {
                Err(CudaFisherError::OutOfMemory { required: required_bytes, free, headroom: headroom_bytes })
            }
        } else { Ok(()) }
    }
    #[inline]
    fn maybe_log_batch_debug(&self) {
        static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                if !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!("[DEBUG] fisher batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaFisher)).debug_batch_logged = true;
                }
            }
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                if !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!("[DEBUG] fisher many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaFisher)).debug_many_logged = true;
                }
            }
        }
    }

    fn expand_grid(range: &FisherBatchRange) -> Vec<FisherParams> {
        let (s, e, st) = range.period;
        let arr: Vec<usize> = if st == 0 || s == e {
            vec![s]
        } else if s < e {
            let mut v = s; let mut out = Vec::new();
            while v <= e { out.push(v); if let Some(n) = v.checked_add(st) { v = n; } else { break; } }
            out
        } else {
            let mut v = s; let mut out = Vec::new();
            while v >= e { out.push(v); if v <= e + st { break; } v -= st; }
            out
        };
        arr.into_iter().map(|p| FisherParams { period: Some(p) }).collect()
    }

    #[inline]
    fn validate_launch_dims(grid: GridSize, block: BlockSize) -> Result<(), CudaFisherError> {
        
        let (gx, gy, gz) = (grid.x, grid.y, grid.z);
        let (bx, by, bz) = (block.x, block.y, block.z);
        if gx == 0 || gy == 0 || gz == 0 || bx == 0 || by == 0 || bz == 0 {
            return Err(CudaFisherError::InvalidInput("zero-sized launch dims".into()));
        }
        
        if bx > 1024 || by > 1024 || bz > 64 { return Err(CudaFisherError::LaunchConfigTooLarge { gx, gy, gz, bx, by, bz }); }
        if gx > 2_147_483_647 || gy > 65_535 || gz > 65_535 { return Err(CudaFisherError::LaunchConfigTooLarge { gx, gy, gz, bx, by, bz }); }
        Ok(())
    }

    fn prepare_batch_inputs(
        high_f32: &[f32],
        low_f32: &[f32],
        sweep: &FisherBatchRange,
    ) -> Result<
        (
            Vec<FisherParams>,
            usize,             
            usize,             
            LockedBuffer<f32>, 
            usize,             
        ),
        CudaFisherError,
    > {
        if high_f32.len() != low_f32.len() {
            return Err(CudaFisherError::InvalidInput("length mismatch".into()));
        }
        let len = high_f32.len();
        if len == 0 {
            return Err(CudaFisherError::InvalidInput("empty input".into()));
        }
        
        let mut first_valid: Option<usize> = None;
        for i in 0..len {
            let h = high_f32[i];
            let l = low_f32[i];
            if h == h && l == l {
                first_valid = Some(i);
                break;
            }
        }
        let first_valid = first_valid
            .ok_or_else(|| CudaFisherError::InvalidInput("all values are NaN".into()))?;

        let combos = Self::expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaFisherError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        let max_p = combos
            .iter()
            .map(|c| c.period.unwrap_or(0))
            .max()
            .unwrap_or(0);
        if max_p == 0 || max_p > len {
            return Err(CudaFisherError::InvalidInput("invalid period".into()));
        }
        if len - first_valid < max_p {
            return Err(CudaFisherError::InvalidInput(
                "not enough valid data".into(),
            ));
        }

        
        let mut hl2 = unsafe { LockedBuffer::uninitialized(len) }?;
        {
            let dst = hl2.as_mut_slice();
            for i in 0..len {
                dst[i] = 0.5f32 * (high_f32[i] + low_f32[i]);
            }
        }
        Ok((combos, first_valid, len, hl2, max_p))
    }

    pub fn fisher_batch_dev(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        sweep: &FisherBatchRange,
    ) -> Result<(DeviceFisherPair, Vec<FisherParams>), CudaFisherError> {
        let (combos, first_valid, len, hl2_locked, max_p) =
            Self::prepare_batch_inputs(high_f32, low_f32, sweep)?;

        
        let bytes_in = len.checked_mul(std::mem::size_of::<f32>()).ok_or_else(|| CudaFisherError::InvalidInput("size overflow".into()))?;
        let bytes_periods = combos.len().checked_mul(std::mem::size_of::<i32>()).ok_or_else(|| CudaFisherError::InvalidInput("size overflow".into()))?;
        let two = 2usize;
        let bytes_out = two
            .checked_mul(combos.len()).and_then(|v| v.checked_mul(len)).and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaFisherError::InvalidInput("size overflow".into()))?;
        let required = bytes_in.checked_add(bytes_periods).and_then(|v| v.checked_add(bytes_out)).ok_or_else(|| CudaFisherError::InvalidInput("size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        Self::ensure_will_fit(required, headroom)?;

        
        let mut periods_locked = LockedBuffer::new(&0i32, combos.len())?;
        {
            let p = periods_locked.as_mut_slice();
            for (i, c) in combos.iter().enumerate() {
                p[i] = c.period.unwrap_or(0) as i32;
            }
        }
        
        let mut d_hl: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(len, &self.stream) }?;
        unsafe { d_hl.async_copy_from(&hl2_locked, &self.stream) }?;
        let mut d_periods: DeviceBuffer<i32> = unsafe { DeviceBuffer::uninitialized_async(combos.len(), &self.stream) }?;
        unsafe { d_periods.async_copy_from(&periods_locked, &self.stream) }?;

        let total = combos.len().checked_mul(len).ok_or_else(|| CudaFisherError::InvalidInput("size overflow".into()))?;
        let mut d_fish: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(total, &self.stream) }?;
        let mut d_sig: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(total, &self.stream) }?;

        let mut func = self
            .module
            .get_function("fisher_batch_f32")
            .map_err(|_| CudaFisherError::MissingKernelSymbol { name: "fisher_batch_f32" })?;
        
        let shmem_bytes = (2 * max_p * std::mem::size_of::<i32>()) as usize;
        if shmem_bytes >= 32 * 1024 {
            func.set_cache_config(CacheConfig::PreferShared)?;
        }
        if shmem_bytes > 48 * 1024 {
            let res = unsafe {
                sys::cuFuncSetAttribute(
                    func.to_raw(),
                    sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    shmem_bytes as i32,
                )
            };
            if res != sys::CUresult::CUDA_SUCCESS {
                return Err(CudaFisherError::InvalidPolicy("dynamic shared memory attribute"));
            }
            let _ = unsafe {
                sys::cuFuncSetAttribute(
                    func.to_raw(),
                    sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                    100,
                )
            };
        }
        
        let (auto_block, _) = func.suggested_launch_configuration(shmem_bytes, BlockSize::x(256)).unwrap_or((128, 0));
        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Auto => auto_block.clamp(64, 256),
            BatchKernelPolicy::Plain { block_x } => block_x.max(32),
        };
        let grid_x: u32 = combos.len() as u32;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        Self::validate_launch_dims(grid, block)?;
        let shared_bytes: u32 = shmem_bytes as u32;
        unsafe {
            (*(self as *const _ as *mut CudaFisher)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        unsafe {
            let mut hl_ptr = d_hl.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut series_len_i = len as i32;
            let mut n_combos_i = combos.len() as i32;
            let mut first_i = first_valid as i32;
            let mut fish_ptr = d_fish.as_device_ptr().as_raw();
            let mut sig_ptr = d_sig.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut hl_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut n_combos_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut fish_ptr as *mut _ as *mut c_void,
                &mut sig_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, shared_bytes, args)?;
        }
        self.stream.synchronize()?;
        self.maybe_log_batch_debug();

        Ok((
            DeviceFisherPair {
                fisher: DeviceArrayF32 {
                    buf: d_fish,
                    rows: combos.len(),
                    cols: len,
                },
                signal: DeviceArrayF32 {
                    buf: d_sig,
                    rows: combos.len(),
                    cols: len,
                },
            },
            combos,
        ))
    }

    pub fn fisher_many_series_one_param_time_major_dev(
        &self,
        high_tm_f32: &[f32],
        low_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceFisherPair, CudaFisherError> {
        if high_tm_f32.len() != low_tm_f32.len() {
            return Err(CudaFisherError::InvalidInput("length mismatch".into()));
        }
        if cols == 0 || rows == 0 {
            return Err(CudaFisherError::InvalidInput("empty matrix".into()));
        }
        if high_tm_f32.len() != cols * rows {
            return Err(CudaFisherError::InvalidInput("bad shape".into()));
        }
        if period == 0 || period > rows {
            return Err(CudaFisherError::InvalidInput("invalid period".into()));
        }

        
        let n = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaFisherError::InvalidInput("size overflow".into()))?;
        let mut hl2_tm = unsafe { LockedBuffer::uninitialized(n) }?;
        {
            let dst = hl2_tm.as_mut_slice();
            for r in 0..rows {
                for c in 0..cols {
                    let idx = r * cols + c;
                    dst[idx] = 0.5f32 * (high_tm_f32[idx] + low_tm_f32[idx]);
                }
            }
        }
        let mut first_valids = LockedBuffer::new(&-1i32, cols)?;
        {
            let fv = first_valids.as_mut_slice();
            for s in 0..cols {
                let mut found = -1i32;
                for r in 0..rows {
                    let h = high_tm_f32[r * cols + s];
                    let l = low_tm_f32[r * cols + s];
                    if h == h && l == l {
                        found = r as i32;
                        break;
                    }
                }
                fv[s] = found;
            }
        }

        let bytes_in = n
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaFisherError::InvalidInput("size overflow".into()))?;
        let bytes_first = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaFisherError::InvalidInput("size overflow".into()))?;
        let two = 2usize;
        let bytes_out = two
            .checked_mul(cols)
            .and_then(|v| v.checked_mul(rows))
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaFisherError::InvalidInput("size overflow".into()))?;
        let required = bytes_in
            .checked_add(bytes_first)
            .and_then(|v| v.checked_add(bytes_out))
            .ok_or_else(|| CudaFisherError::InvalidInput("size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        Self::ensure_will_fit(required, headroom)?;

        let mut d_hl: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(n, &self.stream) }?;
        unsafe { d_hl.async_copy_from(&hl2_tm, &self.stream) }?;
        let mut d_first: DeviceBuffer<i32> = unsafe { DeviceBuffer::uninitialized_async(cols, &self.stream) }?;
        unsafe { d_first.async_copy_from(&first_valids, &self.stream) }?;
        let total = cols.checked_mul(rows).ok_or_else(|| CudaFisherError::InvalidInput("size overflow".into()))?;
        let mut d_fish: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(total, &self.stream) }?;
        let mut d_sig: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(total, &self.stream) }?;

        let func = self
            .module
            .get_function("fisher_many_series_one_param_f32")
            .map_err(|_| CudaFisherError::MissingKernelSymbol { name: "fisher_many_series_one_param_f32" })?;
        
        let (auto_block, _) = func
            .suggested_launch_configuration(0, BlockSize::x(256))
            .unwrap_or((128, 0));
        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => auto_block.clamp(64, 256),
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(64),
        };
        let grid_x: u32 = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        Self::validate_launch_dims(grid, block)?;
        unsafe {
            (*(self as *const _ as *mut CudaFisher)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        unsafe {
            let mut hl_ptr = d_hl.as_device_ptr().as_raw();
            let mut first_ptr = d_first.as_device_ptr().as_raw();
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut period_i = period as i32;
            let mut fish_ptr = d_fish.as_device_ptr().as_raw();
            let mut sig_ptr = d_sig.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut hl_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut fish_ptr as *mut _ as *mut c_void,
                &mut sig_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        self.stream.synchronize()?;
        self.maybe_log_many_debug();
        Ok(DeviceFisherPair {
            fisher: DeviceArrayF32 {
                buf: d_fish,
                rows,
                cols,
            },
            signal: DeviceArrayF32 {
                buf: d_sig,
                rows,
                cols,
            },
        })
    }
}


pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::fisher::FisherBatchRange;

    const ONE_SERIES_LEN: usize = 200_000; 
    const PARAM_SWEEP: usize = 64;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>(); 
        let out_bytes = 2 * ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct FisherBatchDeviceState {
        cuda: CudaFisher,
        func: Function<'static>,
        d_hl: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_fish: DeviceBuffer<f32>,
        d_sig: DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        shared_bytes: u32,
        block_x: u32,
    }
    impl CudaBenchState for FisherBatchDeviceState {
        fn launch(&mut self) {
            unsafe {
                let grid: GridSize = (self.n_combos as u32, 1, 1).into();
                let block: BlockSize = (self.block_x, 1, 1).into();
                let mut hl_ptr = self.d_hl.as_device_ptr().as_raw();
                let mut periods_ptr = self.d_periods.as_device_ptr().as_raw();
                let mut series_len_i = self.len as i32;
                let mut n_combos_i = self.n_combos as i32;
                let mut first_i = self.first_valid as i32;
                let mut fish_ptr = self.d_fish.as_device_ptr().as_raw();
                let mut sig_ptr = self.d_sig.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut hl_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut n_combos_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut fish_ptr as *mut _ as *mut c_void,
                    &mut sig_ptr as *mut _ as *mut c_void,
                ];
                self.cuda
                    .stream
                    .launch(&self.func, grid, block, self.shared_bytes, args)
                    .expect("fisher launch");
            }
            self.cuda.stream.synchronize().expect("fisher sync");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let mut cuda = CudaFisher::new(0).expect("CudaFisher");
        let mut high = gen_series(ONE_SERIES_LEN);
        let mut low = vec![0.0f32; ONE_SERIES_LEN];
        for i in 0..ONE_SERIES_LEN {
            low[i] = 0.7 * high[i] + 0.1 * (i as f32).sin();
        }
        // NaNs at start for warmup semantics
        for i in 0..16 {
            high[i] = f32::NAN;
            low[i] = f32::NAN;
        }
        let sweep = FisherBatchRange {
            period: (9, 9 + PARAM_SWEEP - 1, 1),
        };

        let (combos, first_valid, len, hl2_locked, max_p) =
            CudaFisher::prepare_batch_inputs(&high, &low, &sweep).expect("prepare_batch_inputs");
        let n_combos = combos.len();
        let periods_i32: Vec<i32> = combos
            .iter()
            .map(|c| c.period.unwrap_or(0) as i32)
            .collect();

        let d_hl: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::from_slice_async(hl2_locked.as_slice(), &cuda.stream) }
                .expect("d_hl");
        let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("d_periods");
        let total = n_combos * len;
        let d_fish: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(total, &cuda.stream) }.expect("d_fish");
        let d_sig: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(total, &cuda.stream) }.expect("d_sig");

        let func = cuda
            .module
            .get_function("fisher_batch_f32")
            .expect("fisher_batch_f32");
        let mut func: Function<'static> = unsafe { std::mem::transmute(func) };
        let shmem_bytes = (2 * max_p * std::mem::size_of::<i32>()) as usize;
        if shmem_bytes >= 32 * 1024 {
            func.set_cache_config(CacheConfig::PreferShared).expect("cache_config");
        }
        if shmem_bytes > 48 * 1024 {
            let res = unsafe {
                sys::cuFuncSetAttribute(
                    func.to_raw(),
                    sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    shmem_bytes as i32,
                )
            };
            if res != sys::CUresult::CUDA_SUCCESS {
                panic!("failed to set dynamic shared memory attribute");
            }
            let _ = unsafe {
                sys::cuFuncSetAttribute(
                    func.to_raw(),
                    sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                    100,
                )
            };
        }
        let (auto_block, _) =
            func.suggested_launch_configuration(shmem_bytes, BlockSize::x(256)).unwrap_or((128, 0));
        let block_x: u32 = match cuda.policy.batch {
            BatchKernelPolicy::Auto => auto_block.clamp(64, 256),
            BatchKernelPolicy::Plain { block_x } => block_x.max(32),
        };
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(FisherBatchDeviceState {
            cuda,
            func,
            d_hl,
            d_periods,
            d_fish,
            d_sig,
            len,
            first_valid,
            n_combos,
            shared_bytes: shmem_bytes as u32,
            block_x,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "fisher",
            "one_series_many_params",
            "fisher_cuda_batch_dev",
            "200k_x_64",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}
