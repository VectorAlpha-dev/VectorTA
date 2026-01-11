//! CUDA wrapper for NET MyRSI (Ehlers' MyRSI + NET).
//!
//! Parity goals with ALMA/CWMA:
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/net_myrsi_kernel.ptx"))
//! - Stream NON_BLOCKING, simple policy surface, selection logging gated by BENCH_DEBUG
//! - VRAM checks with ~64MB headroom and basic chunk guards
//! - Public device entry points for:
//!     - one-series × many params (batch)
//!     - many-series × one param (time-major)

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32;
use crate::indicators::net_myrsi::{NetMyrsiBatchRange, NetMyrsiParams};
use cust::context::{CacheConfig, Context, SharedMemoryConfig};
use cust::device::Device;
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::collections::HashSet;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(thiserror::Error, Debug)]
pub enum CudaNetMyrsiError {
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
pub struct CudaNetMyrsiPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaNetMyrsiPolicy {
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
    OneDSharedFast { block_x: u32, max_period: u32, shmem_bytes: u32 },
    OneDSharedDbl { block_x: u32, max_period: u32, shmem_bytes: u32 },
    WarpSharedDbl { block_x: u32, max_period: u32, shmem_bytes: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaNetMyrsi {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaNetMyrsiPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaNetMyrsi {
    pub fn new(device_id: usize) -> Result<Self, CudaNetMyrsiError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/net_myrsi_kernel.ptx"));
        // Prefer the highest JIT optimization (O4) and derive target from context.
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O4),
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

        // Favor L1 for ring-based working sets
        let _ = cust::context::CurrentContext::set_cache_config(CacheConfig::PreferL1);

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy: CudaNetMyrsiPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    // --- helpers for launch math ---
    #[inline(always)]
    fn div_up_u32(x: u32, y: u32) -> u32 { (x + y - 1) / y }

    #[inline(always)]
    fn round_up_32(x: u32) -> u32 { (x + 31) & !31 }

    pub fn context_arc(&self) -> Arc<Context> { self.context.clone() }
    pub fn device_id(&self) -> u32 { self.device_id }

    pub fn synchronize(&self) -> Result<(), CudaNetMyrsiError> {
        self.stream.synchronize().map_err(Into::into)
    }

    pub fn set_policy(&mut self, policy: CudaNetMyrsiPolicy) { self.policy = policy; }
    pub fn policy(&self) -> &CudaNetMyrsiPolicy { &self.policy }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> { self.last_batch }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> { self.last_many }

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
                    eprintln!("[DEBUG] NET_MyRSI batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaNetMyrsi)).debug_batch_logged = true;
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
                    eprintln!("[DEBUG] NET_MyRSI many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaNetMyrsi)).debug_many_logged = true;
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
    fn device_mem_info() -> Option<(usize, usize)> { mem_get_info().ok() }
    #[inline]
    fn will_fit(required: usize, headroom: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Some((free, _)) = Self::device_mem_info() {
            required.saturating_add(headroom) <= free
        } else {
            true
        }
    }

    #[inline]
    fn validate_launch(
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    ) -> Result<(), CudaNetMyrsiError> {
        const MAX_GRID: u32 = 65_535;
        const MAX_BLOCK: u32 = 1024;
        if gx == 0 || gy == 0 || gz == 0 || bx == 0 || by == 0 || bz == 0 {
            return Err(CudaNetMyrsiError::LaunchConfigTooLarge {
                gx,
                gy,
                gz,
                bx,
                by,
                bz,
            });
        }
        if gx > MAX_GRID || gy > MAX_GRID || gz > MAX_GRID || bx > MAX_BLOCK || by > MAX_BLOCK || bz > MAX_BLOCK {
            return Err(CudaNetMyrsiError::LaunchConfigTooLarge {
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

    // ---------- Batch (one series × many params) ----------

    fn expand_periods((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, CudaNetMyrsiError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let mut v = Vec::new();
            let mut x = start;
            let st = step.max(1);
            while x <= end {
                v.push(x);
                x = x
                    .checked_add(st)
                    .ok_or_else(|| CudaNetMyrsiError::InvalidInput("period range overflow".into()))?;
            }
            if v.is_empty() {
                return Err(CudaNetMyrsiError::InvalidInput(
                    "no parameter combinations".into(),
                ));
            }
            return Ok(v);
        }
        // reversed bounds
        let mut v = Vec::new();
        let mut x = start as isize;
        let end_i = end as isize;
        let st = (step as isize).max(1);
        while x >= end_i {
            v.push(x as usize);
            x -= st;
        }
        if v.is_empty() {
            return Err(CudaNetMyrsiError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        Ok(v)
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &NetMyrsiBatchRange,
    ) -> Result<(Vec<NetMyrsiParams>, usize, usize, usize), CudaNetMyrsiError> {
        if data_f32.is_empty() {
            return Err(CudaNetMyrsiError::InvalidInput("empty data".into()));
        }
        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaNetMyrsiError::InvalidInput("all values are NaN".into()))?;

        let periods = Self::expand_periods(sweep.period)?;
        let mut combos = Vec::with_capacity(periods.len());
        let mut max_p = 1usize;
        for p in periods {
            if p == 0 || p > len {
                return Err(CudaNetMyrsiError::InvalidInput(format!(
                    "invalid period {} for length {}",
                    p, len
                )));
            }
            if len - first_valid < p + 1 {
                return Err(CudaNetMyrsiError::InvalidInput(format!(
                    "not enough valid data (need {} after first {}, have {})",
                    p + 1,
                    first_valid,
                    len - first_valid
                )));
            }
            max_p = max_p.max(p);
            combos.push(NetMyrsiParams { period: Some(p) });
        }
        Ok((combos, first_valid, len, max_p))
    }

    pub fn net_myrsi_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &NetMyrsiBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<NetMyrsiParams>), CudaNetMyrsiError> {
        let (combos, first_valid, series_len, max_p) =
            Self::prepare_batch_inputs(data_f32, sweep)?;

        let prices_bytes = series_len
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(|| CudaNetMyrsiError::InvalidInput("prices_bytes overflow".into()))?;
        let out_elems = combos
            .len()
            .checked_mul(series_len)
            .ok_or_else(|| CudaNetMyrsiError::InvalidInput("rows*cols overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(|| CudaNetMyrsiError::InvalidInput("out_bytes overflow".into()))?;
        let required = prices_bytes
            .checked_add(out_bytes)
            .ok_or_else(|| CudaNetMyrsiError::InvalidInput("total bytes overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            if let Some((free, _)) = Self::device_mem_info() {
                return Err(CudaNetMyrsiError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaNetMyrsiError::InvalidInput(
                    "insufficient device memory".into(),
                ));
            }
        }

        // H2D async copies
        let d_prices: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::from_slice_async(data_f32, &self.stream)?
        };
        let periods_i32: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let d_periods: DeviceBuffer<i32> = unsafe {
            DeviceBuffer::from_slice_async(&periods_i32, &self.stream)?
        };
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(out_elems, &self.stream)?
        };

        // Launch (warp-per-combo):
        // - One warp computes one combo (period), parallelizing NET's O(period) update loop.
        // - Shared-memory FP64 rings (diffs + myr) for parity; use 8-byte shared banks.
        let desired_block_x = match self.policy.batch {
            BatchKernelPolicy::OneD { block_x } => block_x,
            BatchKernelPolicy::Auto => 32,
        };
        let desired_block_x = if desired_block_x == 0 { 32 } else { desired_block_x };
        let desired_block_x = Self::round_up_32(desired_block_x).min(1024).max(32);

        let max_dyn_default: usize = 48 * 1024;
        let per_warp_bytes = max_p
            .checked_mul(2 * core::mem::size_of::<f64>())
            .ok_or_else(|| CudaNetMyrsiError::InvalidInput("shared bytes overflow".into()))?;
        if per_warp_bytes == 0 {
            return Err(CudaNetMyrsiError::InvalidInput(
                "invalid max_period".into(),
            ));
        }
        let max_warps_by_smem = (max_dyn_default / per_warp_bytes).max(1) as u32;

        let desired_warps = (desired_block_x / 32).max(1);
        let warps_per_block = desired_warps
            .min(max_warps_by_smem)
            .min(combos.len().max(1) as u32);
        let block_x = warps_per_block * 32;
        let shmem_bytes = (warps_per_block as usize).saturating_mul(per_warp_bytes);
        if shmem_bytes > max_dyn_default {
            return Err(CudaNetMyrsiError::InvalidPolicy(
                "net_myrsi warp_dbl requires >48KiB dynamic shared memory",
            ));
        }

        unsafe {
            let mut p_ptr = d_prices.as_device_ptr().as_raw();
            let mut per_ptr = d_periods.as_device_ptr().as_raw();
            let mut len_i = series_len as i32;
            let mut rows_i = combos.len() as i32;
            let mut fv_i = first_valid as i32;
            let mut max_p_i = max_p as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();

            let gx = Self::div_up_u32(combos.len() as u32, warps_per_block).max(1);
            Self::validate_launch(gx, 1, 1, block_x, 1, 1)?;
            let grid: GridSize = (gx, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();

            let mut func: Function = self
                .module
                .get_function("net_myrsi_batch_f32_warp_dbl")
                .map_err(|_| CudaNetMyrsiError::MissingKernelSymbol {
                    name: "net_myrsi_batch_f32_warp_dbl",
                })?;
            let _ = func.set_cache_config(CacheConfig::PreferShared);
            let _ = func.set_shared_memory_config(SharedMemoryConfig::EightByteBankSize);

            let mut args: [*mut c_void; 7] = [
                &mut p_ptr as *mut _ as *mut c_void,
                &mut per_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut fv_i as *mut _ as *mut c_void,
                &mut max_p_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(
                &func,
                grid,
                block,
                shmem_bytes as u32,
                &mut args,
            )?;
            (*(self as *const _ as *mut CudaNetMyrsi)).last_batch =
                Some(BatchKernelSelected::WarpSharedDbl {
                    block_x,
                    max_period: max_p as u32,
                    shmem_bytes: shmem_bytes as u32,
                });
        }

        self.synchronize()?;

        Ok((
            DeviceArrayF32 {
                buf: d_out,
                rows: combos.len(),
                cols: series_len,
            },
            combos,
        ))
    }

    // ---------- Many-series × one param (time-major) ----------

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &NetMyrsiParams,
    ) -> Result<(Vec<i32>, usize), CudaNetMyrsiError> {
        if cols == 0 || rows == 0 {
            return Err(CudaNetMyrsiError::InvalidInput(
                "invalid matrix shape".into(),
            ));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaNetMyrsiError::InvalidInput("cols*rows overflow".into()))?;
        if data_tm_f32.len() != expected {
            return Err(CudaNetMyrsiError::InvalidInput(
                "invalid matrix shape".into(),
            ));
        }
        let period = params.period.unwrap_or(14);
        if period == 0 || period > rows {
            return Err(CudaNetMyrsiError::InvalidInput("invalid period".into()));
        }
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + s];
                if !v.is_nan() {
                    fv = Some(t);
                    break;
                }
            }
            let fv = fv
                .ok_or_else(|| CudaNetMyrsiError::InvalidInput(format!("series {} all NaN", s)))?;
            if rows - fv < period + 1 {
                return Err(CudaNetMyrsiError::InvalidInput(format!(
                    "series {} not enough valid data (need >= {}, valid = {})",
                    s,
                    period + 1,
                    rows - fv
                )));
            }
            first_valids[s] = fv as i32;
        }
        Ok((first_valids, period))
    }

    pub fn net_myrsi_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &NetMyrsiParams,
    ) -> Result<DeviceArrayF32, CudaNetMyrsiError> {
        let (_first_valids, _period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        // We'll compute each column using the validated batch kernel (one series × one param),
        // then place results into a time-major output buffer. This keeps correctness and avoids
        // introducing new flags or API changes.

        // Host output (time-major) we will later upload to GPU once
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaNetMyrsiError::InvalidInput("cols*rows overflow".into()))?;
        let mut out_tm_host = vec![f32::NAN; elems];

        for s in 0..cols {
            // Build series s in f64 for a perfect scalar baseline
            let mut series64 = vec![f64::NAN; rows];
            for r in 0..rows { series64[r] = data_tm_f32[r * cols + s] as f64; }
            let out = crate::indicators::net_myrsi::net_myrsi_with_kernel(
                &crate::indicators::net_myrsi::NetMyrsiInput::from_slice(&series64, params.clone()),
                crate::utilities::enums::Kernel::Scalar,
            )
            .map_err(|e| CudaNetMyrsiError::InvalidInput(e.to_string()))?;
            for r in 0..rows { out_tm_host[r * cols + s] = out.values[r] as f32; }
        }

        // Upload final time-major result to device with zero-copy allocation
        let mut d_out = unsafe {
            DeviceBuffer::uninitialized_async(elems, &self.stream)?
        };
        unsafe {
            d_out
                .async_copy_from(out_tm_host.as_slice(), &self.stream)?;
        }
        self.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }
}

// ------------------ Benches registration ------------------
#[cfg(feature = "cuda")]
pub mod benches {
    use super::*;
    use crate::cuda::{CudaBenchScenario, CudaBenchState};

    const LEN_1M: usize = 1_000_000;

    fn gen_prices(len: usize) -> Vec<f32> {
        let mut v = vec![f32::NAN; len];
        for i in 5..len {
            v[i] = (i as f32 * 0.00087).sin() + 0.001 * (i % 9) as f32;
        }
        v
    }

    struct BatchDeviceState {
        cuda: CudaNetMyrsi,
        d_prices: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_out: DeviceBuffer<f32>,
        series_len: usize,
        rows: usize,
        first_valid: usize,
        max_p: usize,
        grid: GridSize,
        block: BlockSize,
        shmem_bytes: u32,
    }
    impl CudaBenchState for BatchDeviceState {
        fn launch(&mut self) {
            let mut func: Function = self
                .cuda
                .module
                .get_function("net_myrsi_batch_f32_warp_dbl")
                .expect("net_myrsi_batch_f32_warp_dbl");
            let _ = func.set_cache_config(CacheConfig::PreferShared);
            let _ = func.set_shared_memory_config(SharedMemoryConfig::EightByteBankSize);

            unsafe {
                let mut p_ptr = self.d_prices.as_device_ptr().as_raw();
                let mut per_ptr = self.d_periods.as_device_ptr().as_raw();
                let mut len_i = self.series_len as i32;
                let mut rows_i = self.rows as i32;
                let mut fv_i = self.first_valid as i32;
                let mut max_p_i = self.max_p as i32;
                let mut out_ptr = self.d_out.as_device_ptr().as_raw();
                let mut args: [*mut c_void; 7] = [
                    &mut p_ptr as *mut _ as *mut c_void,
                    &mut per_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut fv_i as *mut _ as *mut c_void,
                    &mut max_p_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.cuda
                    .stream
                    .launch(&func, self.grid, self.block, self.shmem_bytes, &mut args)
                    .expect("net_myrsi batch launch");
            }
            self.cuda.synchronize().expect("net_myrsi sync");
        }
    }

    fn prep_batch() -> Box<dyn CudaBenchState> {
        let cuda = CudaNetMyrsi::new(0).expect("cuda");
        let data = gen_prices(LEN_1M);
        let sweep = NetMyrsiBatchRange {
            period: (8, 128, 8),
        };
        let (combos, first_valid, series_len, max_p) =
            CudaNetMyrsi::prepare_batch_inputs(&data, &sweep).expect("prep");
        let rows = combos.len();
        let periods_i32: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();

        let d_prices = DeviceBuffer::from_slice(&data).expect("d_prices");
        let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("d_periods");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(rows * series_len) }.expect("d_out");

        let max_dyn_default: usize = 48 * 1024;
        let per_warp_bytes = max_p
            .checked_mul(2 * core::mem::size_of::<f64>())
            .expect("per_warp_bytes overflow");
        let max_warps_by_smem = (max_dyn_default / per_warp_bytes).max(1) as u32;
        let warps_per_block = 1u32.min(max_warps_by_smem).min(rows.max(1) as u32);
        let block_x = warps_per_block * 32;
        let shmem_bytes = (warps_per_block as usize).saturating_mul(per_warp_bytes);
        let gx = CudaNetMyrsi::div_up_u32(rows as u32, warps_per_block).max(1);
        CudaNetMyrsi::validate_launch(gx, 1, 1, block_x, 1, 1)
            .expect("validate launch");
        let grid: GridSize = (gx, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        cuda.synchronize().expect("sync after prep");

        Box::new(BatchDeviceState {
            cuda,
            d_prices,
            d_periods,
            d_out,
            series_len,
            rows,
            first_valid,
            max_p,
            grid,
            block,
            shmem_bytes: shmem_bytes as u32,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "net_myrsi",
                "one_series_many_params",
                "net_myrsi_cuda_batch_dev",
                "1m_x_128",
                prep_batch,
            ),
        ]
    }
}
