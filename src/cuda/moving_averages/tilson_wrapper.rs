#![cfg(feature = "cuda")]

use crate::indicators::moving_averages::tilson::{TilsonBatchRange, TilsonParams};
use cust::context::Context;
use cust::device::Device;
use cust::error::CudaError;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::cell::Cell;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaTilsonError {
    Cuda(#[from] CudaError),

    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    MissingKernelSymbol {
        name: &'static str,
    },
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
    InvalidPolicy(&'static str),
    DeviceMismatch {
        buf: u32,
        current: u32,
    },
    NotImplemented,

    InvalidInput(String),
}

impl fmt::Display for CudaTilsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaTilsonError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaTilsonError::OutOfMemory {
                required,
                free,
                headroom,
            } => write!(
                f,
                "Out of memory: required={}B, free={}B, headroom={}B",
                required, free, headroom
            ),
            CudaTilsonError::MissingKernelSymbol { name } => {
                write!(f, "Missing CUDA kernel symbol: {}", name)
            }
            CudaTilsonError::LaunchConfigTooLarge {
                gx,
                gy,
                gz,
                bx,
                by,
                bz,
            } => write!(
                f,
                "Launch config too large: grid=({}, {}, {}), block=({}, {}, {})",
                gx, gy, gz, bx, by, bz
            ),
            CudaTilsonError::InvalidPolicy(s) => write!(f, "Invalid policy: {}", s),
            CudaTilsonError::DeviceMismatch { buf, current } => write!(
                f,
                "Device/context mismatch: buffer on device {}, current device {}",
                buf, current
            ),
            CudaTilsonError::NotImplemented => write!(f, "Not implemented"),
            CudaTilsonError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
    WarpScan { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaTilsonPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaTilsonPolicy {
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
    WarpScan { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaTilson {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaTilsonPolicy,
    last_batch: Cell<Option<BatchKernelSelected>>,
    last_many: Cell<Option<ManySeriesKernelSelected>>,
    debug_batch_logged: Cell<bool>,
    debug_many_logged: Cell<bool>,
}

pub struct DeviceArrayF32Tilson {
    pub buf: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
    pub ctx: Arc<Context>,
    pub device_id: u32,
}
impl DeviceArrayF32Tilson {
    #[inline]
    pub fn device_ptr(&self) -> u64 {
        self.buf.as_device_ptr().as_raw() as u64
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

struct PreparedTilsonBatch {
    combos: Vec<TilsonParams>,
    first_valid: usize,
    series_len: usize,
    periods_i32: Vec<i32>,
    ks_f32: Vec<f32>,
    c1_f32: Vec<f32>,
    c2_f32: Vec<f32>,
    c3_f32: Vec<f32>,
    c4_f32: Vec<f32>,
    lookbacks_i32: Vec<i32>,
}

struct PreparedTilsonManySeries {
    first_valids: Vec<i32>,
    period: usize,
    k_f32: f32,
    c1_f32: f32,
    c2_f32: f32,
    c3_f32: f32,
    c4_f32: f32,
    lookback_i32: i32,
}

impl CudaTilson {
    pub fn new(device_id: usize) -> Result<Self, CudaTilsonError> {
        cust::init(CudaFlags::empty()).map_err(CudaTilsonError::from)?;
        let device = Device::get_device(device_id as u32).map_err(CudaTilsonError::from)?;
        let context = Context::new(device).map_err(CudaTilsonError::from)?;
        let context = Arc::new(context);

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/tilson_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => match Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                Ok(m) => m,
                Err(_) => Module::from_ptx(ptx, &[]).map_err(CudaTilsonError::from)?,
            },
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).map_err(CudaTilsonError::from)?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaTilsonPolicy::default(),
            last_batch: Cell::new(None),
            last_many: Cell::new(None),
            debug_batch_logged: Cell::new(false),
            debug_many_logged: Cell::new(false),
        })
    }

    pub fn new_with_policy(
        device_id: usize,
        policy: CudaTilsonPolicy,
    ) -> Result<Self, CudaTilsonError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaTilsonPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaTilsonPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch.get()
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many.get()
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged.get() {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch.get() {
                let per_scenario =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] Tilson batch selected kernel: {:?}", sel);
                }
                self.debug_batch_logged.set(true);
            }
        }
    }

    #[inline]
    fn maybe_log_many_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged.get() {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many.get() {
                let per_scenario =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] Tilson many-series selected kernel: {:?}", sel);
                }
                self.debug_many_logged.set(true);
            }
        }
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
    fn headroom_bytes() -> usize {
        const DEFAULT_MB: usize = 64;
        if let Ok(v) = std::env::var("CUDA_MEM_HEADROOM_MB") {
            if let Ok(mb) = v.parse::<usize>() {
                return mb.saturating_mul(1024 * 1024);
            }
        }
        DEFAULT_MB * 1024 * 1024
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

    pub fn tilson_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &TilsonBatchRange,
    ) -> Result<DeviceArrayF32Tilson, CudaTilsonError> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = prepared.combos.len();

        let headroom = Self::headroom_bytes();
        let item_f32 = std::mem::size_of::<f32>();
        let input_bytes = data_f32
            .len()
            .checked_mul(item_f32)
            .ok_or_else(|| CudaTilsonError::InvalidInput("input_bytes overflow".into()))?;
        let per_combo_bytes = std::mem::size_of::<i32>()
            .checked_add(
                5usize
                    .checked_mul(item_f32)
                    .ok_or_else(|| CudaTilsonError::InvalidInput("params_bytes overflow".into()))?,
            )
            .and_then(|v| v.checked_add(std::mem::size_of::<i32>()))
            .ok_or_else(|| CudaTilsonError::InvalidInput("params_bytes overflow".into()))?;
        let params_bytes = prepared
            .combos
            .len()
            .checked_mul(per_combo_bytes)
            .ok_or_else(|| CudaTilsonError::InvalidInput("params_bytes overflow".into()))?;
        let out_elems = prepared
            .series_len
            .checked_mul(n_combos)
            .ok_or_else(|| CudaTilsonError::InvalidInput("out elements overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(item_f32)
            .ok_or_else(|| CudaTilsonError::InvalidInput("out bytes overflow".into()))?;
        let required = input_bytes
            .checked_add(params_bytes)
            .ok_or_else(|| CudaTilsonError::InvalidInput("bytes overflow".into()))?
            .checked_add(out_bytes)
            .ok_or_else(|| CudaTilsonError::InvalidInput("bytes overflow".into()))?;
        if let Some((free, _)) = Self::device_mem_info() {
            if !Self::will_fit(required, headroom) {
                return Err(CudaTilsonError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            }
        }

        let d_prices = DeviceBuffer::from_slice(data_f32).map_err(CudaTilsonError::from)?;
        let d_periods =
            DeviceBuffer::from_slice(&prepared.periods_i32).map_err(CudaTilsonError::from)?;
        let d_ks = DeviceBuffer::from_slice(&prepared.ks_f32).map_err(CudaTilsonError::from)?;
        let d_c1 = DeviceBuffer::from_slice(&prepared.c1_f32).map_err(CudaTilsonError::from)?;
        let d_c2 = DeviceBuffer::from_slice(&prepared.c2_f32).map_err(CudaTilsonError::from)?;
        let d_c3 = DeviceBuffer::from_slice(&prepared.c3_f32).map_err(CudaTilsonError::from)?;
        let d_c4 = DeviceBuffer::from_slice(&prepared.c4_f32).map_err(CudaTilsonError::from)?;
        let d_lookbacks =
            DeviceBuffer::from_slice(&prepared.lookbacks_i32).map_err(CudaTilsonError::from)?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(prepared.series_len * n_combos)
                .map_err(CudaTilsonError::from)?
        };

        let combos_per_launch = self.suggest_combos_per_launch(prepared.series_len, n_combos);
        let mut launched = 0usize;
        while launched < n_combos {
            let this = (n_combos - launched).min(combos_per_launch);
            self.launch_batch_kernel_chunk(
                &d_prices,
                &d_periods,
                &d_ks,
                &d_c1,
                &d_c2,
                &d_c3,
                &d_c4,
                &d_lookbacks,
                prepared.series_len,
                prepared.first_valid,
                launched,
                this,
                &mut d_out,
            )?;
            launched += this;
        }

        self.stream.synchronize().map_err(CudaTilsonError::from)?;

        Ok(DeviceArrayF32Tilson {
            buf: d_out,
            rows: n_combos,
            cols: prepared.series_len,
            ctx: self._context.clone(),
            device_id: self.device_id,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn tilson_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_ks: &DeviceBuffer<f32>,
        d_c1: &DeviceBuffer<f32>,
        d_c2: &DeviceBuffer<f32>,
        d_c3: &DeviceBuffer<f32>,
        d_c4: &DeviceBuffer<f32>,
        d_lookbacks: &DeviceBuffer<i32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTilsonError> {
        if series_len == 0 {
            return Err(CudaTilsonError::InvalidInput(
                "series_len must be positive".into(),
            ));
        }
        if first_valid >= series_len {
            return Err(CudaTilsonError::InvalidInput(format!(
                "first_valid {} out of range for len {}",
                first_valid, series_len
            )));
        }
        if n_combos == 0 {
            return Err(CudaTilsonError::InvalidInput(
                "n_combos must be positive".into(),
            ));
        }
        if d_periods.len() != n_combos
            || d_ks.len() != n_combos
            || d_c1.len() != n_combos
            || d_c2.len() != n_combos
            || d_c3.len() != n_combos
            || d_c4.len() != n_combos
            || d_lookbacks.len() != n_combos
        {
            return Err(CudaTilsonError::InvalidInput(
                "device buffer length mismatch".into(),
            ));
        }
        if d_out.len() != n_combos * series_len {
            return Err(CudaTilsonError::InvalidInput(
                "output buffer has incorrect length".into(),
            ));
        }

        let combos_per_launch = self.suggest_combos_per_launch(series_len, n_combos);
        let mut launched = 0usize;
        while launched < n_combos {
            let this = (n_combos - launched).min(combos_per_launch);
            self.launch_batch_kernel_chunk(
                d_prices,
                d_periods,
                d_ks,
                d_c1,
                d_c2,
                d_c3,
                d_c4,
                d_lookbacks,
                series_len,
                first_valid,
                launched,
                this,
                d_out,
            )?;
            launched += this;
        }
        self.stream.synchronize().map_err(CudaTilsonError::from)
    }

    pub fn tilson_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &TilsonParams,
    ) -> Result<DeviceArrayF32Tilson, CudaTilsonError> {
        let prepared =
            Self::prepare_many_series_inputs(data_tm_f32, num_series, series_len, params)?;

        let headroom = Self::headroom_bytes();
        let item_f32 = std::mem::size_of::<f32>();
        let input_bytes = data_tm_f32
            .len()
            .checked_mul(item_f32)
            .ok_or_else(|| CudaTilsonError::InvalidInput("input_bytes overflow".into()))?;
        let first_valids_bytes = prepared
            .first_valids
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaTilsonError::InvalidInput("first_valids_bytes overflow".into()))?;
        let out_elems = num_series
            .checked_mul(series_len)
            .ok_or_else(|| CudaTilsonError::InvalidInput("out elements overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(item_f32)
            .ok_or_else(|| CudaTilsonError::InvalidInput("out bytes overflow".into()))?;
        let required = input_bytes
            .checked_add(first_valids_bytes)
            .ok_or_else(|| CudaTilsonError::InvalidInput("bytes overflow".into()))?
            .checked_add(out_bytes)
            .ok_or_else(|| CudaTilsonError::InvalidInput("bytes overflow".into()))?;
        if let Some((free, _)) = Self::device_mem_info() {
            if !Self::will_fit(required, headroom) {
                return Err(CudaTilsonError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            }
        }

        let d_prices = DeviceBuffer::from_slice(data_tm_f32).map_err(CudaTilsonError::from)?;
        let d_first_valids =
            DeviceBuffer::from_slice(&prepared.first_valids).map_err(CudaTilsonError::from)?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(num_series * series_len).map_err(CudaTilsonError::from)?
        };

        self.launch_many_series_kernel(
            &d_prices,
            &d_first_valids,
            prepared.period,
            prepared.k_f32,
            prepared.c1_f32,
            prepared.c2_f32,
            prepared.c3_f32,
            prepared.c4_f32,
            prepared.lookback_i32,
            num_series,
            series_len,
            &mut d_out,
        )?;

        self.stream.synchronize().map_err(CudaTilsonError::from)?;

        Ok(DeviceArrayF32Tilson {
            buf: d_out,
            rows: series_len,
            cols: num_series,
            ctx: self._context.clone(),
            device_id: self.device_id,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn tilson_many_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: usize,
        k: f32,
        c1: f32,
        c2: f32,
        c3: f32,
        c4: f32,
        lookback: i32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTilsonError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaTilsonError::InvalidInput(
                "series dimensions must be positive".into(),
            ));
        }
        if d_first_valids.len() != num_series {
            return Err(CudaTilsonError::InvalidInput(
                "first_valids length mismatch".into(),
            ));
        }
        if d_out_tm.len() != num_series * series_len {
            return Err(CudaTilsonError::InvalidInput(
                "output buffer length mismatch".into(),
            ));
        }

        self.launch_many_series_kernel(
            d_prices_tm,
            d_first_valids,
            period,
            k,
            c1,
            c2,
            c3,
            c4,
            lookback,
            num_series,
            series_len,
            d_out_tm,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel_chunk(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_ks: &DeviceBuffer<f32>,
        d_c1: &DeviceBuffer<f32>,
        d_c2: &DeviceBuffer<f32>,
        d_c3: &DeviceBuffer<f32>,
        d_c4: &DeviceBuffer<f32>,
        d_lookbacks: &DeviceBuffer<i32>,
        series_len: usize,
        first_valid: usize,
        combos_offset: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTilsonError> {
        if n_combos == 0 {
            return Ok(());
        }

        let (func, grid, block, selected) = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => {
                let block_x = block_x.max(32).min(1024);
                let blocks_x = ((n_combos as u32) + block_x - 1) / block_x;
                if blocks_x == 0 {
                    return Err(CudaTilsonError::LaunchConfigTooLarge {
                        gx: blocks_x,
                        gy: 1,
                        gz: 1,
                        bx: block_x,
                        by: 1,
                        bz: 1,
                    });
                }
                let func = self.module.get_function("tilson_batch_f32").map_err(|_| {
                    CudaTilsonError::MissingKernelSymbol {
                        name: "tilson_batch_f32",
                    }
                })?;
                let grid: GridSize = (blocks_x, 1, 1).into();
                let block: BlockSize = (block_x, 1, 1).into();
                (func, grid, block, BatchKernelSelected::Plain { block_x })
            }
            BatchKernelPolicy::WarpScan { block_x } => {
                let mut block_x = block_x.max(32).min(1024);
                block_x = ((block_x + 31) / 32) * 32;
                let warps_per_block = (block_x / 32).max(1);
                let blocks_x = ((n_combos as u32) + warps_per_block - 1) / warps_per_block;
                if blocks_x == 0 {
                    return Err(CudaTilsonError::LaunchConfigTooLarge {
                        gx: blocks_x,
                        gy: 1,
                        gz: 1,
                        bx: block_x,
                        by: 1,
                        bz: 1,
                    });
                }
                let func = self
                    .module
                    .get_function("tilson_batch_warp_scan_f32")
                    .map_err(|_| CudaTilsonError::MissingKernelSymbol {
                        name: "tilson_batch_warp_scan_f32",
                    })?;
                let grid: GridSize = (blocks_x, 1, 1).into();
                let block: BlockSize = (block_x, 1, 1).into();
                (func, grid, block, BatchKernelSelected::WarpScan { block_x })
            }
            BatchKernelPolicy::Auto => {
                let block_x = 256u32;
                if let Ok(func) = self.module.get_function("tilson_batch_warp_scan_f32") {
                    let warps_per_block = (block_x / 32).max(1);
                    let blocks_x = ((n_combos as u32) + warps_per_block - 1) / warps_per_block;
                    let grid: GridSize = (blocks_x, 1, 1).into();
                    let block: BlockSize = (block_x, 1, 1).into();
                    (func, grid, block, BatchKernelSelected::WarpScan { block_x })
                } else {
                    let blocks_x = ((n_combos as u32) + block_x - 1) / block_x;
                    let func = self.module.get_function("tilson_batch_f32").map_err(|_| {
                        CudaTilsonError::MissingKernelSymbol {
                            name: "tilson_batch_f32",
                        }
                    })?;
                    let grid: GridSize = (blocks_x, 1, 1).into();
                    let block: BlockSize = (block_x, 1, 1).into();
                    (func, grid, block, BatchKernelSelected::Plain { block_x })
                }
            }
        };

        self.last_batch.set(Some(selected));
        self.maybe_log_batch_debug();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();

            let combos_off = combos_offset as u64;
            let off_i32 = combos_off * std::mem::size_of::<i32>() as u64;
            let off_f32 = combos_off * std::mem::size_of::<f32>() as u64;

            let mut periods_ptr = d_periods.as_device_ptr().as_raw() + off_i32;
            let mut ks_ptr = d_ks.as_device_ptr().as_raw() + off_f32;
            let mut c1_ptr = d_c1.as_device_ptr().as_raw() + off_f32;
            let mut c2_ptr = d_c2.as_device_ptr().as_raw() + off_f32;
            let mut c3_ptr = d_c3.as_device_ptr().as_raw() + off_f32;
            let mut c4_ptr = d_c4.as_device_ptr().as_raw() + off_f32;
            let mut lookbacks_ptr = d_lookbacks.as_device_ptr().as_raw() + off_i32;

            let mut series_len_i = series_len as i32;
            let mut first_valid_i = first_valid as i32;
            let mut combos_i = n_combos as i32;

            let out_elem_off = (combos_offset * series_len) as u64;
            let mut out_ptr =
                d_out.as_device_ptr().as_raw() + out_elem_off * std::mem::size_of::<f32>() as u64;

            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut ks_ptr as *mut _ as *mut c_void,
                &mut c1_ptr as *mut _ as *mut c_void,
                &mut c2_ptr as *mut _ as *mut c_void,
                &mut c3_ptr as *mut _ as *mut c_void,
                &mut c4_ptr as *mut _ as *mut c_void,
                &mut lookbacks_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(CudaTilsonError::from)?;
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: usize,
        k: f32,
        c1: f32,
        c2: f32,
        c3: f32,
        c4: f32,
        lookback: i32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTilsonError> {
        if num_series == 0 || series_len == 0 {
            return Ok(());
        }

        let func = self
            .module
            .get_function("tilson_many_series_one_param_f32")
            .map_err(|_| CudaTilsonError::MissingKernelSymbol {
                name: "tilson_many_series_one_param_f32",
            })?;

        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 256u32,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32),
        };

        let blocks_x = ((num_series as u32) + block_x - 1) / block_x;
        if block_x > 1024 || blocks_x == 0 {
            return Err(CudaTilsonError::LaunchConfigTooLarge {
                gx: blocks_x,
                gy: 1,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }
        let grid: GridSize = (blocks_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        self.last_many
            .set(Some(ManySeriesKernelSelected::OneD { block_x }));
        self.maybe_log_many_debug();

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut k_f = k;
            let mut c1_f = c1;
            let mut c2_f = c2;
            let mut c3_f = c3;
            let mut c4_f = c4;
            let mut lookback_i = lookback;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_valids_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut k_f as *mut _ as *mut c_void,
                &mut c1_f as *mut _ as *mut c_void,
                &mut c2_f as *mut _ as *mut c_void,
                &mut c3_f as *mut _ as *mut c_void,
                &mut c4_f as *mut _ as *mut c_void,
                &mut lookback_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(CudaTilsonError::from)?;
        }

        Ok(())
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &TilsonBatchRange,
    ) -> Result<PreparedTilsonBatch, CudaTilsonError> {
        if data_f32.is_empty() {
            return Err(CudaTilsonError::InvalidInput("input data is empty".into()));
        }

        let combos = expand_combos(sweep)?;
        if combos.is_empty() {
            return Err(CudaTilsonError::InvalidInput(
                "no parameter combinations provided".into(),
            ));
        }

        let series_len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaTilsonError::InvalidInput("all values are NaN".into()))?;

        let mut periods_i32 = Vec::with_capacity(combos.len());
        let mut ks_f32 = Vec::with_capacity(combos.len());
        let mut c1_f32 = Vec::with_capacity(combos.len());
        let mut c2_f32 = Vec::with_capacity(combos.len());
        let mut c3_f32 = Vec::with_capacity(combos.len());
        let mut c4_f32 = Vec::with_capacity(combos.len());
        let mut lookbacks_i32 = Vec::with_capacity(combos.len());

        for params in &combos {
            let period = params.period.unwrap_or(0);
            let volume_factor = params.volume_factor.unwrap_or(0.0);
            if period == 0 {
                return Err(CudaTilsonError::InvalidInput(
                    "period must be positive".into(),
                ));
            }
            if !volume_factor.is_finite() {
                return Err(CudaTilsonError::InvalidInput(
                    "volume_factor must be finite".into(),
                ));
            }
            if period > i32::MAX as usize {
                return Err(CudaTilsonError::InvalidInput(
                    "period exceeds CUDA i32 range".into(),
                ));
            }
            let lookback = 6usize
                .checked_mul(period.saturating_sub(1))
                .ok_or_else(|| CudaTilsonError::InvalidInput("lookback overflow".into()))?;
            if lookback > i32::MAX as usize {
                return Err(CudaTilsonError::InvalidInput(
                    "lookback exceeds CUDA i32 range".into(),
                ));
            }

            if first_valid + lookback >= series_len {
                return Err(CudaTilsonError::InvalidInput(format!(
                    "not enough valid data: need >= {}, have {}",
                    lookback + 1,
                    series_len - first_valid
                )));
            }
            if first_valid + period > series_len {
                return Err(CudaTilsonError::InvalidInput(
                    "period exceeds remaining data".into(),
                ));
            }

            let k = 2.0f32 / (period as f32 + 1.0f32);
            let vf = volume_factor as f32;
            let temp = vf * vf;
            let c1 = -(temp * vf);
            let c2 = 3.0f32 * (temp - c1);
            let c3 = -6.0f32 * temp - 3.0f32 * (vf - c1);
            let c4 = 1.0f32 + 3.0f32 * vf - c1 + 3.0f32 * temp;

            periods_i32.push(period as i32);
            ks_f32.push(k);
            c1_f32.push(c1);
            c2_f32.push(c2);
            c3_f32.push(c3);
            c4_f32.push(c4);
            lookbacks_i32.push(lookback as i32);
            for coeff in [c1, c2, c3, c4] {
                if !coeff.is_finite() {
                    return Err(CudaTilsonError::InvalidInput(
                        "computed coefficient is not finite".into(),
                    ));
                }
            }
        }

        Ok(PreparedTilsonBatch {
            combos,
            first_valid,
            series_len,
            periods_i32,
            ks_f32,
            c1_f32,
            c2_f32,
            c3_f32,
            c4_f32,
            lookbacks_i32,
        })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &TilsonParams,
    ) -> Result<PreparedTilsonManySeries, CudaTilsonError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaTilsonError::InvalidInput(
                "series dimensions must be positive".into(),
            ));
        }
        if data_tm_f32.len() != num_series * series_len {
            return Err(CudaTilsonError::InvalidInput(
                "time-major buffer length mismatch".into(),
            ));
        }

        let period = params.period.unwrap_or(5);
        let v_factor = params.volume_factor.unwrap_or(0.0);
        if period == 0 {
            return Err(CudaTilsonError::InvalidInput(
                "period must be positive".into(),
            ));
        }
        if !v_factor.is_finite() {
            return Err(CudaTilsonError::InvalidInput(
                "volume_factor must be finite".into(),
            ));
        }
        if period > i32::MAX as usize {
            return Err(CudaTilsonError::InvalidInput(
                "period exceeds CUDA i32 range".into(),
            ));
        }
        let lookback = 6usize
            .checked_mul(period.saturating_sub(1))
            .ok_or_else(|| CudaTilsonError::InvalidInput("lookback overflow".into()))?;
        if lookback > i32::MAX as usize {
            return Err(CudaTilsonError::InvalidInput(
                "lookback exceeds CUDA i32 range".into(),
            ));
        }

        let stride = num_series;
        let mut first_valids = Vec::with_capacity(num_series);
        for series in 0..num_series {
            let mut fv_opt = None;
            for t in 0..series_len {
                let value = data_tm_f32[t * stride + series];
                if !value.is_nan() {
                    fv_opt = Some(t);
                    break;
                }
            }
            let fv = fv_opt.ok_or_else(|| {
                CudaTilsonError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            if series_len - fv <= lookback {
                return Err(CudaTilsonError::InvalidInput(format!(
                    "series {} not enough valid data (need >= {}, have {})",
                    series,
                    lookback + 1,
                    series_len - fv
                )));
            }
            first_valids.push(fv as i32);
        }

        let k = 2.0f32 / (period as f32 + 1.0f32);
        let vf = v_factor as f32;
        let temp = vf * vf;
        let c1 = -(temp * vf);
        let c2 = 3.0f32 * (temp - c1);
        let c3 = -6.0f32 * temp - 3.0f32 * (vf - c1);
        let c4 = 1.0f32 + 3.0f32 * vf - c1 + 3.0f32 * temp;

        for coeff in [c1, c2, c3, c4] {
            if !coeff.is_finite() {
                return Err(CudaTilsonError::InvalidInput(
                    "computed coefficient is not finite".into(),
                ));
            }
        }

        Ok(PreparedTilsonManySeries {
            first_valids,
            period,
            k_f32: k,
            c1_f32: c1,
            c2_f32: c2,
            c3_f32: c3,
            c4_f32: c4,
            lookback_i32: lookback as i32,
        })
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::moving_averages::tilson::{TilsonBatchRange, TilsonParams};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;
    const MANY_SERIES_COLS: usize = 250;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = elems * std::mem::size_of::<f32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct TilsonBatchDevState {
        cuda: CudaTilson,
        d_prices: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_ks: DeviceBuffer<f32>,
        d_c1: DeviceBuffer<f32>,
        d_c2: DeviceBuffer<f32>,
        d_c3: DeviceBuffer<f32>,
        d_c4: DeviceBuffer<f32>,
        d_lookbacks: DeviceBuffer<i32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for TilsonBatchDevState {
        fn launch(&mut self) {
            self.cuda
                .tilson_batch_device(
                    &self.d_prices,
                    &self.d_periods,
                    &self.d_ks,
                    &self.d_c1,
                    &self.d_c2,
                    &self.d_c3,
                    &self.d_c4,
                    &self.d_lookbacks,
                    self.series_len,
                    self.first_valid,
                    self.n_combos,
                    &mut self.d_out,
                )
                .expect("tilson batch kernel");
            self.cuda.stream.synchronize().expect("tilson sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaTilson::new(0).expect("cuda tilson");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = TilsonBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
            volume_factor: (0.0, 0.0, 0.0),
        };

        let prepared =
            CudaTilson::prepare_batch_inputs(&price, &sweep).expect("tilson prepare batch");
        let n_combos = prepared.combos.len();

        let d_prices = DeviceBuffer::from_slice(&price).expect("d_prices");
        let d_periods = DeviceBuffer::from_slice(&prepared.periods_i32).expect("d_periods");
        let d_ks = DeviceBuffer::from_slice(&prepared.ks_f32).expect("d_ks");
        let d_c1 = DeviceBuffer::from_slice(&prepared.c1_f32).expect("d_c1");
        let d_c2 = DeviceBuffer::from_slice(&prepared.c2_f32).expect("d_c2");
        let d_c3 = DeviceBuffer::from_slice(&prepared.c3_f32).expect("d_c3");
        let d_c4 = DeviceBuffer::from_slice(&prepared.c4_f32).expect("d_c4");
        let d_lookbacks = DeviceBuffer::from_slice(&prepared.lookbacks_i32).expect("d_lookbacks");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(prepared.series_len * n_combos) }.expect("d_out");

        cuda.stream.synchronize().expect("sync after prep");
        Box::new(TilsonBatchDevState {
            cuda,
            d_prices,
            d_periods,
            d_ks,
            d_c1,
            d_c2,
            d_c3,
            d_c4,
            d_lookbacks,
            series_len: prepared.series_len,
            first_valid: prepared.first_valid,
            n_combos,
            d_out,
        })
    }

    struct TilsonManyDevState {
        cuda: CudaTilson,
        d_prices_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        period: usize,
        k: f32,
        c1: f32,
        c2: f32,
        c3: f32,
        c4: f32,
        lookback: i32,
        cols: usize,
        rows: usize,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for TilsonManyDevState {
        fn launch(&mut self) {
            self.cuda
                .tilson_many_series_one_param_device(
                    &self.d_prices_tm,
                    &self.d_first_valids,
                    self.period,
                    self.k,
                    self.c1,
                    self.c2,
                    self.c3,
                    self.c4,
                    self.lookback,
                    self.cols,
                    self.rows,
                    &mut self.d_out_tm,
                )
                .expect("tilson many-series kernel");
            self.cuda.stream.synchronize().expect("tilson sync");
        }
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaTilson::new(0).expect("cuda tilson");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = TilsonParams {
            period: Some(64),
            volume_factor: Some(0.0),
        };
        let prepared = CudaTilson::prepare_many_series_inputs(&data_tm, cols, rows, &params)
            .expect("tilson prep");

        let d_prices_tm = DeviceBuffer::from_slice(&data_tm).expect("d_prices_tm");
        let d_first_valids =
            DeviceBuffer::from_slice(&prepared.first_valids).expect("d_first_valids");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out_tm");

        cuda.stream.synchronize().expect("sync after prep");
        Box::new(TilsonManyDevState {
            cuda,
            d_prices_tm,
            d_first_valids,
            period: prepared.period,
            k: prepared.k_f32,
            c1: prepared.c1_f32,
            c2: prepared.c2_f32,
            c3: prepared.c3_f32,
            c4: prepared.c4_f32,
            lookback: prepared.lookback_i32,
            cols,
            rows,
            d_out_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "tilson",
                "one_series_many_params",
                "tilson_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "tilson",
                "many_series_one_param",
                "tilson_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}

fn expand_combos(range: &TilsonBatchRange) -> Result<Vec<TilsonParams>, CudaTilsonError> {
    fn axis_usize(axis: (usize, usize, usize)) -> Result<Vec<usize>, CudaTilsonError> {
        let (start, end, step) = axis;
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let mut v = Vec::new();
            let mut cur = start;
            while cur <= end {
                v.push(cur);
                cur = cur
                    .checked_add(step)
                    .ok_or_else(|| CudaTilsonError::InvalidInput("usize range overflow".into()))?;
            }
            return Ok(v);
        }

        let mut v = Vec::new();
        let mut cur = start;
        loop {
            v.push(cur);
            match cur.checked_sub(step) {
                Some(next) if next >= end => cur = next,
                _ => break,
            }
        }
        if v.is_empty() {
            Err(CudaTilsonError::InvalidInput("empty usize range".into()))
        } else {
            Ok(v)
        }
    }

    fn axis_f64(axis: (f64, f64, f64)) -> Result<Vec<f64>, CudaTilsonError> {
        let (start, end, step) = axis;
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }
        let mut v = Vec::new();
        let mut x = start;
        if step > 0.0 {
            while x <= end + 1e-12 {
                v.push(x);
                x = x + step;
            }
        } else {
            while x >= end - 1e-12 {
                v.push(x);
                x = x + step;
            }
        }
        if v.is_empty() {
            Err(CudaTilsonError::InvalidInput("empty f64 range".into()))
        } else {
            Ok(v)
        }
    }

    let periods = axis_usize(range.period)?;
    let volume_factors = axis_f64(range.volume_factor)?;
    let mut combos = Vec::with_capacity(periods.len().saturating_mul(volume_factors.len()));
    for &period in &periods {
        for &vf in &volume_factors {
            combos.push(TilsonParams {
                period: Some(period),
                volume_factor: Some(vf),
            });
        }
    }
    if combos.is_empty() {
        return Err(CudaTilsonError::InvalidInput(
            "no parameter combinations".into(),
        ));
    }
    Ok(combos)
}

impl CudaTilson {
    #[inline]
    fn suggest_combos_per_launch(&self, series_len: usize, total_combos: usize) -> usize {
        const MAX_GRID_DIM: usize = 2_147_483_647;
        let by_dim = total_combos.min(MAX_GRID_DIM).max(1);
        if !Self::mem_check_enabled() {
            return by_dim;
        }

        let input_bytes = series_len.saturating_mul(std::mem::size_of::<f32>());
        let params_per_combo = std::mem::size_of::<i32>()
            + std::mem::size_of::<f32>() * 5
            + std::mem::size_of::<i32>();
        let out_per_combo = series_len.saturating_mul(std::mem::size_of::<f32>());

        let free = match Self::device_mem_info() {
            Some((f, _)) => f,
            None => return by_dim,
        };
        let headroom = Self::headroom_bytes();
        let usable = free.saturating_sub(headroom);
        if usable <= input_bytes {
            return 1;
        }
        let rem = usable - input_bytes;

        let denom = out_per_combo.saturating_add(params_per_combo.max(1));
        let vram_cap = (rem / denom).max(1);
        by_dim.min(vram_cap).max(1)
    }
}
