#![cfg(feature = "cuda")]

use crate::indicators::moving_averages::trima::{TrimaBatchRange, TrimaParams};
use cust::context::Context;
use cust::context::{CacheConfig, SharedMemoryConfig};
use cust::device::Device;
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use thiserror::Error;

const TRIMA_TS: u32 = 128;
const TRIMA_TT: u32 = 64;

#[derive(Debug, Error)]
pub enum CudaTrimaError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("out of memory: required={required}B, free={free}B, headroom={headroom}B")]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("launch config too large: grid=({gx},{gy},{gz}), block=({bx},{by},{bz})")]
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("device mismatch: buf={buf}, current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

pub struct DeviceArrayF32Trima {
    pub buf: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
    pub ctx: Arc<Context>,
    pub device_id: u32,
}
impl DeviceArrayF32Trima {
    #[inline]
    pub fn device_ptr(&self) -> u64 {
        self.buf.as_device_ptr().as_raw() as u64
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
    Tiled { tile: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
    Tiled { tile_s: u32, tile_t: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaTrimaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaTrimaPolicy {
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
    Tiled { tile: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
    Tiled { tile_s: u32, tile_t: u32 },
}

pub struct CudaTrima {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaTrimaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaTrima {
    pub fn new(device_id: usize) -> Result<Self, CudaTrimaError> {
        cust::init(CudaFlags::empty())?;

        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/trima_kernel.ptx"));

        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O4),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => match Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                Ok(m) => m,
                Err(_) => Module::from_ptx(ptx, &[])?,
            },
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaTrimaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn synchronize(&self) -> Result<(), CudaTrimaError> {
        self.stream.synchronize().map_err(Into::into)
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
                    eprintln!("[DEBUG] TRIMA batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaTrima)).debug_batch_logged = true;
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
                    eprintln!("[DEBUG] TRIMA many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaTrima)).debug_many_logged = true;
                }
            }
        }
    }

    pub fn set_policy(&mut self, policy: CudaTrimaPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaTrimaPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }

    fn expand_periods(range: &TrimaBatchRange) -> Vec<usize> {
        let (start, end, step) = range.period;
        if step == 0 || start == end {
            return vec![start];
        }
        let (lo, hi) = if start <= end {
            (start, end)
        } else {
            (end, start)
        };
        let mut v = Vec::new();
        let mut cur = lo;
        while cur <= hi {
            v.push(cur);
            match cur.checked_add(step) {
                Some(n) => cur = n,
                None => break,
            }
            if Some(&cur) == v.last() {
                break;
            }
        }
        v
    }

    #[inline]
    fn will_fit(&self, required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaTrimaError> {
        if let Ok((free, _total)) = mem_get_info() {
            if required_bytes.saturating_add(headroom_bytes) > free {
                return Err(CudaTrimaError::OutOfMemory {
                    required: required_bytes,
                    free,
                    headroom: headroom_bytes,
                });
            }
        }
        Ok(())
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &TrimaBatchRange,
    ) -> Result<(Vec<usize>, usize), CudaTrimaError> {
        if data_f32.is_empty() {
            return Err(CudaTrimaError::InvalidInput("empty data".into()));
        }

        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaTrimaError::InvalidInput("all values are NaN".into()))?;

        let periods = Self::expand_periods(sweep);
        if periods.is_empty() {
            return Err(CudaTrimaError::InvalidInput("no periods in sweep".into()));
        }

        let len = data_f32.len();
        for &period in &periods {
            if period <= 3 {
                return Err(CudaTrimaError::InvalidInput(format!(
                    "period {} too small (must be > 3)",
                    period
                )));
            }
            if period > len {
                return Err(CudaTrimaError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            if len - first_valid < period {
                return Err(CudaTrimaError::InvalidInput(format!(
                    "not enough valid data: needed {}, have {}",
                    period,
                    len - first_valid
                )));
            }
        }

        Ok((periods, first_valid))
    }

    fn compute_weights(period: usize) -> Vec<f32> {
        let mut weights = vec![0.0f32; period];
        let m1 = (period + 1) / 2;
        let m2 = period - m1 + 1;
        let norm = (m1 * m2) as f32;
        for i in 0..period {
            let w = if i < m1 {
                (i + 1) as f32
            } else if i < m2 {
                m1 as f32
            } else {
                (m1 + m2 - 1 - i) as f32
            };
            weights[i] = w / norm;
        }
        weights
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &TrimaParams,
    ) -> Result<(Vec<i32>, usize), CudaTrimaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaTrimaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaTrimaError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }

        let period = params.period.unwrap_or(30);
        if period <= 3 {
            return Err(CudaTrimaError::InvalidInput(format!(
                "period {} too small (must be > 3)",
                period
            )));
        }

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut fv = None;
            for row in 0..rows {
                let idx = row * cols + series;
                let v = data_tm_f32[idx];
                if !v.is_nan() {
                    fv = Some(row);
                    break;
                }
            }
            let fv_row = fv.ok_or_else(|| {
                CudaTrimaError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            if rows - fv_row < period {
                return Err(CudaTrimaError::InvalidInput(format!(
                    "series {} lacks enough valid data: needed {}, have {}",
                    series,
                    period,
                    rows - fv_row
                )));
            }
            first_valids[series] = fv_row as i32;
        }

        Ok((first_valids, period))
    }

    fn launch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warms: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTrimaError> {
        if series_len == 0 {
            return Err(CudaTrimaError::InvalidInput("series_len is zero".into()));
        }
        if n_combos == 0 {
            return Err(CudaTrimaError::InvalidInput("no parameter combos".into()));
        }
        if max_period == 0 {
            return Err(CudaTrimaError::InvalidInput("max_period is zero".into()));
        }
        if series_len > i32::MAX as usize
            || n_combos > i32::MAX as usize
            || max_period > i32::MAX as usize
        {
            return Err(CudaTrimaError::InvalidInput(
                "series_len, n_combos, or max_period exceed i32::MAX".into(),
            ));
        }

        let sizeof_f32 = std::mem::size_of::<f32>();
        let mut func: Function;
        let grid_x: u32;
        let block: BlockSize;
        let shared_bytes: u32;
        let selected: BatchKernelSelected;
        if let Ok(mut f_tiled) = self.module.get_function("trima_batch_f32_tiled") {
            let tile_x = match self.policy.batch {
                BatchKernelPolicy::Tiled { tile } if tile > 0 => tile,
                _ => 256,
            };
            let smem_bytes = (max_period + (tile_x as usize + max_period - 1)) * sizeof_f32;
            self.prefer_shared_and_optin_smem(&mut f_tiled, smem_bytes);
            let tiles_t = ((series_len as u32) + tile_x - 1) / tile_x;
            let block_cfg: BlockSize = (tile_x, 1, 1).into();
            let grid_cfg: GridSize = (tiles_t.max(1), 1, 1).into();
            let mut use_tiled = true;
            if let Ok(avail) =
                f_tiled.available_dynamic_shared_memory_per_block(grid_cfg, block_cfg)
            {
                if smem_bytes > avail {
                    use_tiled = false;
                }
            }
            if use_tiled {
                func = f_tiled;
                grid_x = tiles_t;
                block = block_cfg;
                shared_bytes = smem_bytes as u32;
                selected = BatchKernelSelected::Tiled { tile: tile_x };
            } else {
                func = self.module.get_function("trima_batch_f32").map_err(|_| {
                    CudaTrimaError::MissingKernelSymbol {
                        name: "trima_batch_f32",
                    }
                })?;
                let block_x = match self.policy.batch {
                    BatchKernelPolicy::Plain { block_x } if block_x > 0 => block_x,
                    _ => 256,
                };
                grid_x = ((series_len as u32) + block_x - 1) / block_x;
                block = (block_x, 1, 1).into();
                shared_bytes = (max_period * sizeof_f32) as u32;
                selected = BatchKernelSelected::OneD { block_x };
            }
        } else {
            func = self.module.get_function("trima_batch_f32").map_err(|_| {
                CudaTrimaError::MissingKernelSymbol {
                    name: "trima_batch_f32",
                }
            })?;
            let block_x = match self.policy.batch {
                BatchKernelPolicy::Plain { block_x } if block_x > 0 => block_x,
                _ => 256,
            };
            grid_x = ((series_len as u32) + block_x - 1) / block_x;
            block = (block_x, 1, 1).into();
            shared_bytes = (max_period * sizeof_f32) as u32;
            selected = BatchKernelSelected::OneD { block_x };
        }

        unsafe {
            (*(self as *const _ as *mut CudaTrima)).last_batch = Some(selected);
        }
        self.maybe_log_batch_debug();

        const MAX_GRID_Y: usize = 65_535;
        let mut launched = 0usize;
        while launched < n_combos {
            let len = (n_combos - launched).min(MAX_GRID_Y);
            let grid: GridSize = (grid_x.max(1), len as u32, 1).into();

            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();

                let periods_ptr = d_periods.as_device_ptr().add(launched);
                let mut periods_ptr = periods_ptr.as_raw();
                let warms_ptr = d_warms.as_device_ptr().add(launched);
                let mut warms_ptr = warms_ptr.as_raw();
                let mut series_len_i = series_len as i32;
                let mut n_combos_i = len as i32;
                let mut max_period_i = max_period as i32;
                let out_ptr = d_out.as_device_ptr().add(launched * series_len);
                let mut out_ptr = out_ptr.as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut warms_ptr as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut n_combos_i as *mut _ as *mut c_void,
                    &mut max_period_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, shared_bytes, args)?;
            }

            launched += len;
        }

        Ok(())
    }
    #[inline]
    fn prefer_shared_and_optin_smem(&self, func: &mut Function, requested_dynamic_smem: usize) {
        let _ = func.set_cache_config(CacheConfig::PreferShared);
        let _ = func.set_shared_memory_config(SharedMemoryConfig::FourByteBankSize);
        unsafe {
            use cust::sys::{cuFuncSetAttribute, CUfunction_attribute_enum as Attr};
            let raw = func.to_raw();
            let _ = cuFuncSetAttribute(
                raw,
                Attr::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                requested_dynamic_smem as i32,
            );
            let _ = cuFuncSetAttribute(
                raw,
                Attr::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                100,
            );
        }
    }

    #[inline]
    fn upload_pinned_async(&self, data: &[f32]) -> Result<DeviceBuffer<f32>, CudaTrimaError> {
        if data.is_empty() {
            return Err(CudaTrimaError::InvalidInput("empty input slice".into()));
        }
        unsafe {
            use cust::sys as cu;
            let ptr = data.as_ptr() as *mut std::ffi::c_void;
            let bytes = data.len() * std::mem::size_of::<f32>();
            let r = cu::cuMemHostRegister_v2(ptr, bytes, 0);
            if r != cu::CUresult::CUDA_SUCCESS {
                return Err(CudaTrimaError::InvalidInput(format!(
                    "cuMemHostRegister failed: {:?}",
                    r
                )));
            }
            let mut dev: DeviceBuffer<f32> =
                DeviceBuffer::uninitialized_async(data.len(), &self.stream)?;
            dev.async_copy_from(data, &self.stream)?;

            self.stream.synchronize()?;
            let r2 = cu::cuMemHostUnregister(ptr);
            if r2 != cu::CUresult::CUDA_SUCCESS {
                return Err(CudaTrimaError::InvalidInput(format!(
                    "cuMemHostUnregister failed: {:?}",
                    r2
                )));
            }
            Ok(dev)
        }
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: usize,
        cols: usize,
        rows: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTrimaError> {
        if period == 0 || cols == 0 || rows == 0 {
            return Err(CudaTrimaError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        if period > i32::MAX as usize || cols > i32::MAX as usize || rows > i32::MAX as usize {
            return Err(CudaTrimaError::InvalidInput(
                "period, num_series, or series_len exceed i32::MAX".into(),
            ));
        }

        let mut use_tiled = matches!(
            self.policy.many_series,
            ManySeriesKernelPolicy::Auto | ManySeriesKernelPolicy::Tiled { .. }
        );
        if matches!(self.policy.many_series, ManySeriesKernelPolicy::OneD { .. }) {
            use_tiled = false;
        }
        if let ManySeriesKernelPolicy::Tiled { tile_s, tile_t } = self.policy.many_series {
            if tile_s != TRIMA_TS || tile_t != TRIMA_TT {
                use_tiled = false;
            }
        }

        let tile_s: u32 = TRIMA_TS;
        let tile_t: u32 = TRIMA_TT;

        if cols < tile_s as usize || rows < tile_t as usize {
            use_tiled = false;
        }

        let sizeof_f32 = std::mem::size_of::<f32>();
        let shared_bytes_tiled =
            ((period + (tile_s as usize * (tile_t as usize + period - 1))) * sizeof_f32) as u32;

        if use_tiled {
            let mut func = self
                .module
                .get_function("trima_multi_series_one_param_f32_tm_tiled")
                .map_err(|_| CudaTrimaError::MissingKernelSymbol {
                    name: "trima_multi_series_one_param_f32_tm_tiled",
                })?;
            self.prefer_shared_and_optin_smem(&mut func, shared_bytes_tiled as usize);
            let grid_x = ((cols as u32) + tile_s - 1) / tile_s;
            let grid_y = ((rows as u32) + tile_t - 1) / tile_t;
            if grid_y > 65_535 {
                return Err(CudaTrimaError::LaunchConfigTooLarge {
                    gx: grid_x,
                    gy: grid_y,
                    gz: 1,
                    bx: tile_s,
                    by: 1,
                    bz: 1,
                });
            }
            let grid: GridSize = (grid_x.max(1), grid_y.max(1), 1).into();
            let block: BlockSize = (tile_s, 1, 1).into();

            let launch_res = unsafe {
                let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
                let mut weights_ptr = d_weights.as_device_ptr().as_raw();
                let mut period_i = period as i32;
                let mut num_series_i = cols as i32;
                let mut series_len_i = rows as i32;
                let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
                let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut weights_ptr as *mut _ as *mut c_void,
                    &mut period_i as *mut _ as *mut c_void,
                    &mut num_series_i as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut first_valids_ptr as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, shared_bytes_tiled, args)
            };
            match launch_res {
                Ok(()) => {
                    unsafe {
                        (*(self as *const _ as *mut CudaTrima)).last_many =
                            Some(ManySeriesKernelSelected::Tiled { tile_s, tile_t });
                    }
                    self.maybe_log_many_debug();
                    return Ok(());
                }
                Err(_e) => {}
            }
        }

        {
            let func = self
                .module
                .get_function("trima_multi_series_one_param_f32")
                .map_err(|_| CudaTrimaError::MissingKernelSymbol {
                    name: "trima_multi_series_one_param_f32",
                })?;
            let block_x = match self.policy.many_series {
                ManySeriesKernelPolicy::OneD { block_x } if block_x > 0 => block_x,
                _ => 128,
            };
            unsafe {
                (*(self as *const _ as *mut CudaTrima)).last_many =
                    Some(ManySeriesKernelSelected::OneD { block_x });
            }
            self.maybe_log_many_debug();
            let grid_x = ((rows as u32) + block_x - 1) / block_x;
            if (cols as u32) > 65_535 {
                return Err(CudaTrimaError::LaunchConfigTooLarge {
                    gx: grid_x,
                    gy: cols as u32,
                    gz: 1,
                    bx: block_x,
                    by: 1,
                    bz: 1,
                });
            }
            let grid: GridSize = (grid_x.max(1), cols as u32, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            let shared_bytes = (period * sizeof_f32) as u32;

            unsafe {
                let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
                let mut weights_ptr = d_weights.as_device_ptr().as_raw();
                let mut period_i = period as i32;
                let mut num_series_i = cols as i32;
                let mut series_len_i = rows as i32;
                let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
                let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut weights_ptr as *mut _ as *mut c_void,
                    &mut period_i as *mut _ as *mut c_void,
                    &mut num_series_i as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut first_valids_ptr as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, shared_bytes, args)?;
            }
        }

        Ok(())
    }

    pub fn trima_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warms: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTrimaError> {
        self.launch_kernel(
            d_prices, d_periods, d_warms, series_len, n_combos, max_period, d_out,
        )
    }

    pub fn trima_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &TrimaBatchRange,
    ) -> Result<DeviceArrayF32Trima, CudaTrimaError> {
        let (periods, first_valid) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let series_len = data_f32.len();
        let n_combos = periods.len();
        let max_period = periods.iter().copied().max().unwrap_or(0);

        let prices_bytes = series_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaTrimaError::InvalidInput("size overflow in VRAM estimate".into()))?;
        let periods_bytes = n_combos
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaTrimaError::InvalidInput("size overflow in VRAM estimate".into()))?;
        let warms_bytes = n_combos
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaTrimaError::InvalidInput("size overflow in VRAM estimate".into()))?;
        let out_bytes = n_combos
            .checked_mul(series_len)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaTrimaError::InvalidInput("size overflow in VRAM estimate".into()))?;
        let required = prices_bytes
            .checked_add(periods_bytes)
            .and_then(|x| x.checked_add(warms_bytes))
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaTrimaError::InvalidInput("size overflow in VRAM estimate".into()))?;
        self.will_fit(required, 64usize * 1024 * 1024)?;

        let periods_i32: Vec<i32> = periods.iter().map(|&p| p as i32).collect();
        let warms_i32: Vec<i32> = periods
            .iter()
            .map(|&p| (first_valid + p - 1) as i32)
            .collect();

        let d_prices = self.upload_pinned_async(data_f32)?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)?;
        let d_warms = DeviceBuffer::from_slice(&warms_i32)?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }?;

        self.launch_kernel(
            &d_prices, &d_periods, &d_warms, series_len, n_combos, max_period, &mut d_out,
        )?;

        self.stream.synchronize()?;

        Ok(DeviceArrayF32Trima {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
            ctx: self._context.clone(),
            device_id: self.device_id,
        })
    }

    fn run_many_series_kernel(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        first_valids: &[i32],
        period: usize,
    ) -> Result<DeviceArrayF32Trima, CudaTrimaError> {
        let prices_bytes = cols * rows * std::mem::size_of::<f32>();
        let weights_bytes = period * std::mem::size_of::<f32>();
        let first_valids_bytes = cols * std::mem::size_of::<i32>();
        let out_bytes = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaTrimaError::InvalidInput("cols*rows overflow".into()))?
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaTrimaError::InvalidInput("byte size overflow".into()))?;
        let required = prices_bytes
            .checked_add(weights_bytes)
            .and_then(|v| v.checked_add(first_valids_bytes))
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaTrimaError::InvalidInput("accumulated byte size overflow".into()))?;
        self.will_fit(required, 64usize * 1024 * 1024)?;

        let weights = Self::compute_weights(period);
        let d_prices = self.upload_pinned_async(data_tm_f32)?;
        let d_weights = DeviceBuffer::from_slice(&weights)?;
        let d_first_valids = DeviceBuffer::from_slice(first_valids)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }?;

        self.launch_many_series_kernel(
            &d_prices,
            &d_weights,
            &d_first_valids,
            period,
            cols,
            rows,
            &mut d_out,
        )?;

        self.stream.synchronize()?;

        Ok(DeviceArrayF32Trima {
            buf: d_out,
            rows,
            cols,
            ctx: self._context.clone(),
            device_id: self.device_id,
        })
    }

    pub fn trima_multi_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        period: i32,
        num_series: i32,
        series_len: i32,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTrimaError> {
        if period <= 0 || num_series <= 0 || series_len <= 0 {
            return Err(CudaTrimaError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_prices_tm,
            d_weights,
            d_first_valids,
            period as usize,
            num_series as usize,
            series_len as usize,
            d_out_tm,
        )
    }

    pub fn trima_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &TrimaParams,
    ) -> Result<DeviceArrayF32Trima, CudaTrimaError> {
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, period)
    }

    pub fn trima_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &TrimaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaTrimaError> {
        if out_tm.len() != cols * rows {
            return Err(CudaTrimaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out_tm.len(),
                cols * rows
            )));
        }
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let arr = self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, period)?;
        arr.buf.copy_to(out_tm)?;
        Ok(())
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::moving_averages::trima::TrimaParams;

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

    struct TrimaBatchDevState {
        cuda: CudaTrima,
        d_prices: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_warms: DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for TrimaBatchDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_kernel(
                    &self.d_prices,
                    &self.d_periods,
                    &self.d_warms,
                    self.series_len,
                    self.n_combos,
                    self.max_period,
                    &mut self.d_out,
                )
                .expect("trima batch kernel");
            self.cuda.stream.synchronize().expect("trima sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaTrima::new(0).expect("cuda trima");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = TrimaBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
        };
        let (periods, first_valid) =
            CudaTrima::prepare_batch_inputs(&price, &sweep).expect("trima prepare batch inputs");
        let series_len = price.len();
        let n_combos = periods.len();
        let max_period = periods.iter().copied().max().unwrap_or(0);

        let periods_i32: Vec<i32> = periods.iter().map(|&p| p as i32).collect();
        let warms_i32: Vec<i32> = periods
            .iter()
            .map(|&p| (first_valid + p - 1) as i32)
            .collect();

        let d_prices = DeviceBuffer::from_slice(&price).expect("d_prices");
        let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("d_periods");
        let d_warms = DeviceBuffer::from_slice(&warms_i32).expect("d_warms");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(series_len * n_combos) }.expect("d_out");
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(TrimaBatchDevState {
            cuda,
            d_prices,
            d_periods,
            d_warms,
            series_len,
            n_combos,
            max_period,
            d_out,
        })
    }

    struct TrimaManyDevState {
        cuda: CudaTrima,
        d_prices_tm: DeviceBuffer<f32>,
        d_weights: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for TrimaManyDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series_kernel(
                    &self.d_prices_tm,
                    &self.d_weights,
                    &self.d_first_valids,
                    self.period,
                    self.cols,
                    self.rows,
                    &mut self.d_out_tm,
                )
                .expect("trima many-series kernel");
            self.cuda.stream.synchronize().expect("trima sync");
        }
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaTrima::new(0).expect("cuda trima");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = TrimaParams { period: Some(64) };
        let (first_valids, period) =
            CudaTrima::prepare_many_series_inputs(&data_tm, cols, rows, &params)
                .expect("trima prepare many-series inputs");

        let weights = CudaTrima::compute_weights(period);

        let d_prices_tm = DeviceBuffer::from_slice(&data_tm).expect("d_prices_tm");
        let d_weights = DeviceBuffer::from_slice(&weights).expect("d_weights");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out_tm");
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(TrimaManyDevState {
            cuda,
            d_prices_tm,
            d_weights,
            d_first_valids,
            cols,
            rows,
            period,
            d_out_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "trima",
                "one_series_many_params",
                "trima_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "trima",
                "many_series_one_param",
                "trima_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
