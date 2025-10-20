#![cfg(feature = "cuda")]

//! CUDA wrapper for Ease of Movement (EMV).
//!
//! Parity points with ALMA/CWMA wrappers:
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/emv_kernel.ptx"))
//! - Stream NON_BLOCKING and simple OneD policies with introspection
//! - VRAM checks and bounds guards; warmup/NaN semantics match scalar
//! - Public device entry points that return VRAM-resident DeviceArrayF32

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

#[derive(Debug)]
pub enum CudaEmvError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaEmvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaEmvError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaEmvError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaEmvError {}

// Minimal policy surface mirroring common wrappers
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
    _context: Context,
    policy: CudaEmvPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaEmv {
    pub fn new(device_id: usize) -> Result<Self, CudaEmvError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaEmvError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaEmvError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaEmvError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/emv_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaEmvError::Cuda(e.to_string()))?;

        // Favor L1 cache by default for read-mostly workloads
        let _ = cust::context::CurrentContext::set_cache_config(CacheConfig::PreferL1);

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaEmvError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
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
        self.stream
            .synchronize()
            .map_err(|e| CudaEmvError::Cuda(e.to_string()))
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

    // ---- Batch path (one series × many params; EMV has no params) ----

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
            .find(|&i| !(high[i].is_nan() || low[i].is_nan() || volume[i].is_nan()))
            .ok_or_else(|| CudaEmvError::InvalidInput("all values are NaN".into()))?;
        let valid_after = (first..len)
            .filter(|&i| !(high[i].is_nan() || low[i].is_nan() || volume[i].is_nan()))
            .count();
        if valid_after < 2 {
            return Err(CudaEmvError::InvalidInput(format!(
                "not enough valid data: need at least 2, found {}",
                valid_after
            )));
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
        let mut func: Function = self
            .module
            .get_function("emv_batch_f32")
            .map_err(|e| CudaEmvError::Cuda(e.to_string()))?;

        let _ = func.set_cache_config(CacheConfig::PreferL1);

        // Block size policy: env EMV_BLOCK_X or occupancy suggestion
        let block_x: u32 = match std::env::var("EMV_BLOCK_X").ok().as_deref() {
            Some("auto") | None => {
                let (_min_grid, suggested) = func
                    .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
                    .map_err(|e| CudaEmvError::Cuda(e.to_string()))?;
                suggested
            }
            Some(s) => s.parse::<u32>().ok().filter(|&v| v > 0).unwrap_or(128),
        };
        let grid_x = ((n_rows as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

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
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaEmvError::Cuda(e.to_string()))?;
        }

        // Introspection: record selected batch kernel
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

        // VRAM estimate: 3 inputs + 1 output
        let bytes = (3 * len + len) * std::mem::size_of::<f32>();
        let headroom = 64 * 1024 * 1024; // 64MB
        if !Self::will_fit(bytes, headroom) {
            return Err(CudaEmvError::InvalidInput(
                "insufficient VRAM for EMV batch".into(),
            ));
        }

        // Use pinned host buffers for better H2D throughput
        let h_high =
            LockedBuffer::from_slice(high).map_err(|e| CudaEmvError::Cuda(e.to_string()))?;
        let h_low = LockedBuffer::from_slice(low).map_err(|e| CudaEmvError::Cuda(e.to_string()))?;
        let h_vol =
            LockedBuffer::from_slice(volume).map_err(|e| CudaEmvError::Cuda(e.to_string()))?;

        let mut d_high = unsafe { DeviceBuffer::<f32>::uninitialized_async(len, &self.stream) }
            .map_err(|e| CudaEmvError::Cuda(e.to_string()))?;
        let mut d_low = unsafe { DeviceBuffer::<f32>::uninitialized_async(len, &self.stream) }
            .map_err(|e| CudaEmvError::Cuda(e.to_string()))?;
        let mut d_vol = unsafe { DeviceBuffer::<f32>::uninitialized_async(len, &self.stream) }
            .map_err(|e| CudaEmvError::Cuda(e.to_string()))?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized_async(len, &self.stream) }
            .map_err(|e| CudaEmvError::Cuda(e.to_string()))?;

        unsafe {
            d_high
                .async_copy_from(&h_high, &self.stream)
                .map_err(|e| CudaEmvError::Cuda(e.to_string()))?;
            d_low
                .async_copy_from(&h_low, &self.stream)
                .map_err(|e| CudaEmvError::Cuda(e.to_string()))?;
            d_vol
                .async_copy_from(&h_vol, &self.stream)
                .map_err(|e| CudaEmvError::Cuda(e.to_string()))?;
        }

        self.launch_batch_kernel(&d_high, &d_low, &d_vol, len, 1, first, &mut d_out)?;

        self.stream
            .synchronize()
            .map_err(|e| CudaEmvError::Cuda(e.to_string()))?;
        // Selection introspection optional; keep None to avoid &mut self requirement.
        self.maybe_log_batch_debug();

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: 1,
            cols: len,
        })
    }

    // ---- Many-series × one-param (time-major) ----

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
        if high_tm.len() != cols * rows
            || low_tm.len() != cols * rows
            || vol_tm.len() != cols * rows
        {
            return Err(CudaEmvError::InvalidInput("matrix shape mismatch".into()));
        }
        let mut fv = vec![0i32; cols];
        for s in 0..cols {
            let mut found = -1i32;
            for r in 0..rows {
                let idx = r * cols + s;
                if !(high_tm[idx].is_nan() || low_tm[idx].is_nan() || vol_tm[idx].is_nan()) {
                    found = r as i32;
                    break;
                }
            }
            if found < 0 {
                return Err(CudaEmvError::InvalidInput(format!(
                    "all NaN in series {}",
                    s
                )));
            }
            // Require at least two valid points
            let mut valid = 0usize;
            for r in (found as usize)..rows {
                let idx = r * cols + s;
                if !(high_tm[idx].is_nan() || low_tm[idx].is_nan() || vol_tm[idx].is_nan()) {
                    valid += 1;
                }
            }
            if valid < 2 {
                return Err(CudaEmvError::InvalidInput(format!(
                    "not enough valid data in series {}: need >=2, found {}",
                    s, valid
                )));
            }
            fv[s] = found;
        }
        Ok(fv)
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
        let func: Function = self
            .module
            .get_function("emv_many_series_one_param_f32")
            .map_err(|e| CudaEmvError::Cuda(e.to_string()))?;

        // Block size policy: env EMV_BLOCK_X or occupancy suggestion
        let block_x: u32 = match std::env::var("EMV_BLOCK_X").ok().as_deref() {
            Some("auto") | None => {
                let (_min_grid, suggested) = func
                    .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
                    .map_err(|e| CudaEmvError::Cuda(e.to_string()))?;
                suggested
            }
            Some(s) => s.parse::<u32>().ok().filter(|&v| v > 0).unwrap_or(256),
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

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
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaEmvError::Cuda(e.to_string()))?;
        }
        // Introspection: record selected many-series kernel
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

        // VRAM estimate: 3 inputs + first_valids + out
        let elems = cols * rows;
        let bytes =
            (3 * elems + elems) * std::mem::size_of::<f32>() + cols * std::mem::size_of::<i32>();
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(bytes, headroom) {
            return Err(CudaEmvError::InvalidInput(
                "insufficient VRAM for EMV many-series".into(),
            ));
        }

        let d_high_tm =
            DeviceBuffer::from_slice(high_tm).map_err(|e| CudaEmvError::Cuda(e.to_string()))?;
        let d_low_tm =
            DeviceBuffer::from_slice(low_tm).map_err(|e| CudaEmvError::Cuda(e.to_string()))?;
        let d_vol_tm =
            DeviceBuffer::from_slice(vol_tm).map_err(|e| CudaEmvError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaEmvError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(|e| CudaEmvError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_high_tm,
            &d_low_tm,
            &d_vol_tm,
            &d_first,
            cols,
            rows,
            &mut d_out_tm,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaEmvError::Cuda(e.to_string()))?;
        // Selection introspection optional; keep None to avoid &mut self requirement.
        self.maybe_log_many_debug();

        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows,
            cols,
        })
    }
}

// ---------- Bench profiles ----------
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

    struct BatchState {
        cuda: CudaEmv,
        high: Vec<f32>,
        low: Vec<f32>,
        vol: Vec<f32>,
    }
    impl CudaBenchState for BatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .emv_batch_dev(&self.high, &self.low, &self.vol)
                .expect("emv_batch_dev");
        }
    }
    fn prep_one_series() -> Box<dyn CudaBenchState> {
        let price = gen_series(ONE_SERIES_LEN);
        let (high, low) = synth_hl_from_price(&price);
        let vol = synth_volume(ONE_SERIES_LEN);
        let cuda = CudaEmv::new(0).expect("cuda");
        Box::new(BatchState {
            cuda,
            high,
            low,
            vol,
        })
    }

    struct ManyState {
        cuda: CudaEmv,
        high_tm: Vec<f32>,
        low_tm: Vec<f32>,
        vol_tm: Vec<f32>,
        cols: usize,
        rows: usize,
    }
    impl CudaBenchState for ManyState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .emv_many_series_one_param_time_major_dev(
                    &self.high_tm,
                    &self.low_tm,
                    &self.vol_tm,
                    self.cols,
                    self.rows,
                )
                .expect("emv_many_series_one_param_time_major_dev");
        }
    }
    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let price_tm = gen_time_major_prices(cols, rows);
        let (high_tm, low_tm) = synth_hl_from_price(&price_tm);
        let vol_tm = gen_time_major_volumes(cols, rows);
        let cuda = CudaEmv::new(0).expect("cuda");
        Box::new(ManyState {
            cuda,
            high_tm,
            low_tm,
            vol_tm,
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
