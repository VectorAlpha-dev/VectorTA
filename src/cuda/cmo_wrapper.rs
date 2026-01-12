#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32;
use crate::indicators::cmo::{CmoBatchRange, CmoParams};
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

#[derive(thiserror::Error, Debug)]
pub enum CudaCmoError {
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
pub struct CudaCmoPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaCmoPolicy {
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

pub struct CudaCmo {
    module: Module,
    stream: Stream,
    context: std::sync::Arc<Context>,
    device_id: u32,
    policy: CudaCmoPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaCmo {
    pub fn new(device_id: usize) -> Result<Self, CudaCmoError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = std::sync::Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/cmo_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => match Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                Ok(m) => m,
                Err(_) => Module::from_ptx(ptx, &[])?,
            },
        };

        let _ = cust::context::CurrentContext::set_cache_config(CacheConfig::PreferL1);

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy: CudaCmoPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn set_policy(&mut self, policy: CudaCmoPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaCmoPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
    pub fn context_arc(&self) -> std::sync::Arc<Context> {
        self.context.clone()
    }
    pub fn device_id(&self) -> u32 {
        self.device_id
    }
    pub fn synchronize(&self) -> Result<(), CudaCmoError> {
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
                    eprintln!("[DEBUG] CMO batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaCmo)).debug_batch_logged = true;
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
                    eprintln!("[DEBUG] CMO many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaCmo)).debug_many_logged = true;
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
        if let Some((free, _total)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    fn prepare_batch_inputs(
        prices: &[f32],
        sweep: &CmoBatchRange,
    ) -> Result<(Vec<CmoParams>, usize, usize), CudaCmoError> {
        if prices.is_empty() {
            return Err(CudaCmoError::InvalidInput("empty data".into()));
        }
        let len = prices.len();
        let first_valid = prices
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaCmoError::InvalidInput("all values are NaN".into()))?;

        fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
            if step == 0 || start == end {
                return vec![start];
            }
            let mut vals = Vec::new();
            if start < end {
                let mut x = start;
                while x <= end {
                    vals.push(x);
                    let next = x.saturating_add(step);
                    if next == x {
                        break;
                    }
                    x = next;
                }
            } else {
                let mut x = start;
                loop {
                    vals.push(x);
                    if x <= end {
                        break;
                    }
                    let next = x.saturating_sub(step);
                    if next >= x {
                        break;
                    }
                    x = next;
                }
            }
            vals
        }
        let periods: Vec<usize> = axis_usize(sweep.period);
        if periods.is_empty() {
            return Err(CudaCmoError::InvalidInput("no periods".into()));
        }
        let combos: Vec<CmoParams> = periods
            .iter()
            .map(|&p| CmoParams { period: Some(p) })
            .collect();

        let max_p = *periods.iter().max().unwrap();
        if len - first_valid <= max_p {
            return Err(CudaCmoError::InvalidInput(format!(
                "not enough valid data (needed > {}, valid = {})",
                max_p,
                len - first_valid
            )));
        }
        Ok((combos, first_valid, len))
    }

    fn _unused_prefix_build(_prices: &[f32], _first_valid: usize) -> (Vec<f64>, Vec<f64>) {
        unimplemented!()
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaCmoError> {
        let mut func: Function = self.module.get_function("cmo_batch_f32").map_err(|_| {
            CudaCmoError::MissingKernelSymbol {
                name: "cmo_batch_f32",
            }
        })?;

        let _ = func.set_cache_config(CacheConfig::PreferL1);

        let block_x: u32 = match std::env::var("CMO_BLOCK_X").ok().as_deref() {
            Some(s) if s != "auto" => s
                .parse::<u32>()
                .ok()
                .filter(|&v| v > 0 && v % 32 == 0)
                .unwrap_or(256),
            _ => {
                let n = n_combos as u32;
                if n >= 256 { 256 } else { ((n + 31) / 32) * 32 }.max(32)
            }
        };

        let warps_per_block = (block_x / 32).max(1);
        let grid_x = ((n_combos as u32) + warps_per_block - 1) / warps_per_block;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        if block_x > 1024 {
            return Err(CudaCmoError::LaunchConfigTooLarge {
                gx: grid_x.max(1),
                gy: 1,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }

        unsafe {
            (*(self as *const _ as *mut CudaCmo)).last_batch =
                Some(BatchKernelSelected::OneD { block_x });
        }
        self.maybe_log_batch_debug();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    pub fn cmo_batch_dev(
        &self,
        prices: &[f32],
        sweep: &CmoBatchRange,
    ) -> Result<DeviceArrayF32, CudaCmoError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(prices, sweep)?;

        let rows = combos.len();
        let out_elems = rows
            .checked_mul(len)
            .ok_or_else(|| CudaCmoError::InvalidInput("size overflow".into()))?;
        let bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|b| {
                b.checked_add(
                    rows.checked_mul(std::mem::size_of::<i32>())
                        .unwrap_or(usize::MAX),
                )
            })
            .and_then(|b| {
                b.checked_add(
                    out_elems
                        .checked_mul(std::mem::size_of::<f32>())
                        .unwrap_or(usize::MAX),
                )
            })
            .ok_or_else(|| CudaCmoError::InvalidInput("size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(bytes, headroom) {
            let (free, _) = mem_get_info().unwrap_or((0, 0));
            return Err(CudaCmoError::OutOfMemory {
                required: bytes,
                free,
                headroom,
            });
        }

        let periods_i32: Vec<i32> = combos
            .iter()
            .map(|c| c.period.unwrap_or(14) as i32)
            .collect();

        let h_prices = LockedBuffer::from_slice(prices)?;
        let h_p = LockedBuffer::from_slice(&periods_i32)?;

        let mut d_prices = unsafe { DeviceBuffer::<f32>::uninitialized_async(len, &self.stream) }?;
        let mut d_periods =
            unsafe { DeviceBuffer::<i32>::uninitialized_async(rows, &self.stream) }?;
        let mut d_out =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(rows * len, &self.stream) }?;

        unsafe {
            d_prices.async_copy_from(&h_prices, &self.stream)?;
            d_periods.async_copy_from(&h_p, &self.stream)?;
        }

        self.launch_batch_kernel(&d_prices, &d_periods, len, rows, first_valid, &mut d_out)?;

        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols: len,
        })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &CmoParams,
    ) -> Result<(Vec<i32>, usize), CudaCmoError> {
        if cols == 0 || rows == 0 {
            return Err(CudaCmoError::InvalidInput(
                "matrix dimensions must be positive".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaCmoError::InvalidInput("matrix shape mismatch".into()));
        }
        let period = params.period.unwrap_or(14);
        if period == 0 || period > rows {
            return Err(CudaCmoError::InvalidInput(
                "invalid period for many-series".into(),
            ));
        }

        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for r in 0..rows {
                let v = data_tm_f32[r * cols + s];
                if !v.is_nan() {
                    fv = Some(r);
                    break;
                }
            }
            let fv =
                fv.ok_or_else(|| CudaCmoError::InvalidInput(format!("series {} all NaN", s)))?;
            if rows - fv <= period {
                return Err(CudaCmoError::InvalidInput(format!(
                    "series {}: not enough valid data (needed > {}, valid = {})",
                    s,
                    period,
                    rows - fv
                )));
            }
            first_valids[s] = fv as i32;
        }
        Ok((first_valids, period))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaCmoError> {
        let func = self
            .module
            .get_function("cmo_many_series_one_param_f32")
            .map_err(|_| CudaCmoError::MissingKernelSymbol {
                name: "cmo_many_series_one_param_f32",
            })?;

        let block_x: u32 = match std::env::var("CMO_MS_BLOCK_X").ok().as_deref() {
            Some(s) if s != "auto" => s
                .parse::<u32>()
                .ok()
                .filter(|&v| v > 0 && v % 32 == 0)
                .unwrap_or(256),
            _ => {
                let n = cols as u32;
                if n >= 256 { 256 } else { ((n + 31) / 32) * 32 }.max(32)
            }
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        if block_x > 1024 {
            return Err(CudaCmoError::LaunchConfigTooLarge {
                gx: grid_x.max(1),
                gy: 1,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }

        unsafe {
            (*(self as *const _ as *mut CudaCmo)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();

        unsafe {
            let mut p_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut f_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut period_i = period as i32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_ptr as *mut _ as *mut c_void,
                &mut f_ptr as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    pub fn cmo_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &CmoParams,
    ) -> Result<DeviceArrayF32, CudaCmoError> {
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaCmoError::InvalidInput("size overflow".into()))?;
        let bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|b| {
                b.checked_add(
                    first_valids
                        .len()
                        .checked_mul(std::mem::size_of::<i32>())
                        .unwrap_or(usize::MAX),
                )
            })
            .and_then(|b| {
                b.checked_add(
                    elems
                        .checked_mul(std::mem::size_of::<f32>())
                        .unwrap_or(usize::MAX),
                )
            })
            .ok_or_else(|| CudaCmoError::InvalidInput("size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(bytes, headroom) {
            let (free, _) = mem_get_info().unwrap_or((0, 0));
            return Err(CudaCmoError::OutOfMemory {
                required: bytes,
                free,
                headroom,
            });
        }

        let h_prices = LockedBuffer::from_slice(data_tm_f32)?;
        let h_first = LockedBuffer::from_slice(&first_valids)?;

        let mut d_prices_tm =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(cols * rows, &self.stream) }?;
        let mut d_first = unsafe { DeviceBuffer::<i32>::uninitialized_async(cols, &self.stream) }?;
        let mut d_out_tm =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(cols * rows, &self.stream) }?;

        unsafe {
            d_prices_tm.async_copy_from(&h_prices, &self.stream)?;
            d_first.async_copy_from(&h_first, &self.stream)?;
        }

        self.launch_many_series_kernel(&d_prices_tm, &d_first, cols, rows, period, &mut d_out_tm)?;

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
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;
    const MANY_SERIES_COLS: usize = 250;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let periods_bytes = PARAM_SWEEP * std::mem::size_of::<i32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + periods_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = elems * std::mem::size_of::<f32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct BatchDeviceState {
        cuda: CudaCmo,
        d_prices: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_out: DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
    }
    impl CudaBenchState for BatchDeviceState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_prices,
                    &self.d_periods,
                    self.len,
                    self.n_combos,
                    self.first_valid,
                    &mut self.d_out,
                )
                .expect("cmo launch_batch_kernel");
            self.cuda.synchronize().expect("cmo sync");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = CmoBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
        };

        let (combos, first_valid, len) =
            CudaCmo::prepare_batch_inputs(&price, &sweep).expect("cmo prepare_batch_inputs");
        let n_combos = combos.len();
        let periods_i32: Vec<i32> = combos
            .iter()
            .map(|c| c.period.unwrap_or(14) as i32)
            .collect();

        let cuda = CudaCmo::new(0).expect("cuda");
        let d_prices =
            unsafe { DeviceBuffer::from_slice_async(&price, &cuda.stream) }.expect("d_prices");
        let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("d_periods");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * len) }.expect("d_out");
        cuda.synchronize().expect("sync after prep");
        Box::new(BatchDeviceState {
            cuda,
            d_prices,
            d_periods,
            d_out,
            len,
            first_valid,
            n_combos,
        })
    }

    struct ManySeriesDeviceState {
        cuda: CudaCmo,
        d_prices_tm: DeviceBuffer<f32>,
        d_first: DeviceBuffer<i32>,
        d_out_tm: DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        period: usize,
    }
    impl CudaBenchState for ManySeriesDeviceState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series_kernel(
                    &self.d_prices_tm,
                    &self.d_first,
                    self.cols,
                    self.rows,
                    self.period,
                    &mut self.d_out_tm,
                )
                .expect("cmo launch_many_series_kernel");
            self.cuda.synchronize().expect("cmo sync");
        }
    }
    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let period = 64usize;
        let first_valids: Vec<i32> = (0..cols).map(|i| i as i32).collect();

        let cuda = CudaCmo::new(0).expect("cuda");
        let d_prices_tm =
            unsafe { DeviceBuffer::from_slice_async(&data_tm, &cuda.stream) }.expect("d_prices_tm");
        let d_first = DeviceBuffer::from_slice(&first_valids).expect("d_first");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out_tm");
        cuda.synchronize().expect("sync after prep");
        Box::new(ManySeriesDeviceState {
            cuda,
            d_prices_tm,
            d_first,
            d_out_tm,
            cols,
            rows,
            period,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "cmo",
                "one_series_many_params",
                "cmo_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "cmo",
                "many_series_one_param",
                "cmo_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
