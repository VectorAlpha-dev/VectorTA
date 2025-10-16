//! CUDA scaffolding for the Predictive Moving Average (PMA).
//!
//! Mirrors ALMA-style GPU surface: one-series × many-params (synthetic combos)
//! and many-series × one-param (time-major). This PMA variant matches
//! src/indicators/pma.rs semantics (no 1-bar lag; warmups at first+6 and first+9).

#![cfg(feature = "cuda")]

use super::DeviceArrayF32;
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::indicators::pma::PmaBatchRange;

// Kernel policy (parity with ALMA/CWMA wrappers)
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
    Tiled2D { tx: u32, ty: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaPmaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaPmaPolicy {
    fn default() -> Self {
        Self { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Auto }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
    Tiled { tile: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
    Tiled2D { tx: u32, ty: u32 },
}

#[derive(Debug)]
pub enum CudaPmaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaPmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaPmaError::Cuda(e) => write!(f, "CUDA error: {e}"),
            CudaPmaError::InvalidInput(e) => write!(f, "Invalid input: {e}"),
        }
    }
}

impl std::error::Error for CudaPmaError {}

/// VRAM-backed pair of PMA outputs (predict + trigger)
pub struct DevicePmaPair {
    pub predict: DeviceArrayF32,
    pub trigger: DeviceArrayF32,
}

impl DevicePmaPair {
    #[inline]
    pub fn rows(&self) -> usize { self.predict.rows }
    #[inline]
    pub fn cols(&self) -> usize { self.predict.cols }
    #[inline]
    pub fn len(&self) -> usize { self.predict.len() }
}

pub struct CudaPma {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaPmaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

// Helper container for batch preparation
struct BatchInputs {
    combos: usize,
    first_valid: usize,
    series_len: usize,
}

impl CudaPma {
    pub fn new(device_id: usize) -> Result<Self, CudaPmaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaPmaError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32).map_err(|e| CudaPmaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaPmaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/pma_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaPmaError::Cuda(e.to_string()))?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaPmaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaPmaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(device_id: usize, policy: CudaPmaPolicy) -> Result<Self, CudaPmaError> {
        let mut s = Self::new(device_id)?; s.policy = policy; Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaPmaPolicy) { self.policy = policy; }
    pub fn policy(&self) -> &CudaPmaPolicy { &self.policy }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> { self.last_batch }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> { self.last_many }
    pub fn synchronize(&self) -> Result<(), CudaPmaError> {
        self.stream.synchronize().map_err(|e| CudaPmaError::Cuda(e.to_string()))
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") { Ok(v) => v != "0" && v.to_lowercase() != "false", Err(_) => true }
    }
    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> { mem_get_info().ok() }
    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() { return true; }
        if let Some((free, _t)) = Self::device_mem_info() { required_bytes.saturating_add(headroom_bytes) <= free } else { true }
    }

    fn maybe_log_batch_debug(&self) {
        static GLOBAL: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_scen = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scen || !GLOBAL.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] PMA batch selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaPma)).debug_batch_logged = true; }
            }
        }
    }
    fn maybe_log_many_debug(&self) {
        static GLOBAL: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per_scen = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scen || !GLOBAL.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] PMA many-series selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaPma)).debug_many_logged = true; }
            }
        }
    }

    // ---------------- Batch (one series × many params) ----------------
    fn prepare_batch_inputs(prices: &[f32], _sweep: &PmaBatchRange) -> Result<BatchInputs, CudaPmaError> {
        if prices.is_empty() { return Err(CudaPmaError::InvalidInput("empty price series".into())); }
        let first_valid = prices.iter().position(|v| !v.is_nan())
            .ok_or_else(|| CudaPmaError::InvalidInput("all values are NaN".into()))?;
        const MIN_REQUIRED: usize = 7; // 7 samples for first predict
        if prices.len() - first_valid < MIN_REQUIRED {
            return Err(CudaPmaError::InvalidInput(format!(
                "not enough valid data (needed >= {MIN_REQUIRED}, valid = {})", prices.len() - first_valid
            )));
        }
        // PMA has no tunable params; keep one synthetic combo for API parity.
        Ok(BatchInputs { combos: 1, first_valid, series_len: prices.len() })
    }

    fn run_batch_kernel(&self,
        prices: &[f32], inputs: &BatchInputs
    ) -> Result<DevicePmaPair, CudaPmaError> {
        let prices_bytes = inputs.series_len * core::mem::size_of::<f32>();
        let out_bytes = inputs.combos * inputs.series_len * core::mem::size_of::<f32>();
        let required = prices_bytes + 2 * out_bytes;
        let headroom = 64 * 1024 * 1024; // 64MB safety
        if !Self::will_fit(required, headroom) {
            return Err(CudaPmaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let mut d_prices: DeviceBuffer<f32> = DeviceBuffer::from_slice(prices)
            .map_err(|e| CudaPmaError::Cuda(e.to_string()))?;
        let mut d_predict: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(inputs.combos * inputs.series_len) }
            .map_err(|e| CudaPmaError::Cuda(e.to_string()))?;
        let mut d_trigger: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(inputs.combos * inputs.series_len) }
            .map_err(|e| CudaPmaError::Cuda(e.to_string()))?;

        self.launch_batch_kernel_select(&d_prices, inputs.series_len, inputs.combos, inputs.first_valid, &mut d_predict, &mut d_trigger)?;

        Ok(DevicePmaPair {
            predict: DeviceArrayF32 { buf: d_predict, rows: inputs.combos, cols: inputs.series_len },
            trigger: DeviceArrayF32 { buf: d_trigger, rows: inputs.combos, cols: inputs.series_len },
        })
    }

    fn launch_batch_kernel_select(&self,
        d_prices: &DeviceBuffer<f32>, series_len: usize, n_combos: usize, first_valid: usize,
        d_predict: &mut DeviceBuffer<f32>, d_trigger: &mut DeviceBuffer<f32>
    ) -> Result<(), CudaPmaError> {
        let (fname, block, grid, sel) = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => (
                "pma_batch_f32",
                BlockSize::xy(block_x.max(1), 1),
                GridSize::xyz(n_combos as u32, 1, 1),
                Some(BatchKernelSelected::Plain { block_x }),
            ),
            BatchKernelPolicy::Tiled { tile } => {
                let sym = if tile >= 256 { "pma_batch_tiled_f32_tile256" } else { "pma_batch_tiled_f32_tile128" };
                let name = if self.module.get_function(sym).is_ok() { sym } else { "pma_batch_f32" };
                (name, BlockSize::xy(1, 1), GridSize::xyz(n_combos as u32, 1, 1), Some(BatchKernelSelected::Tiled { tile }))
            }
            BatchKernelPolicy::Auto => {
                // Default to simple 1D launch; computation is strictly sequential per combo.
                ("pma_batch_f32", BlockSize::xy(1, 1), GridSize::xyz(n_combos as u32, 1, 1), Some(BatchKernelSelected::Plain { block_x: 1 }))
            }
        };

        if let Some(s) = sel { unsafe { (*(self as *const _ as *mut CudaPma)).last_batch = Some(s); } }
        self.maybe_log_batch_debug();

        let func = self.module.get_function(fname).map_err(|e| CudaPmaError::Cuda(e.to_string()))?;
        let mut args: [*mut c_void; 6] = [
            &mut d_prices.as_device_ptr().as_raw() as *mut _ as *mut c_void,
            &mut (series_len as i32) as *mut _ as *mut c_void,
            &mut (n_combos as i32) as *mut _ as *mut c_void,
            &mut (first_valid as i32) as *mut _ as *mut c_void,
            &mut d_predict.as_device_ptr().as_raw() as *mut _ as *mut c_void,
            &mut d_trigger.as_device_ptr().as_raw() as *mut _ as *mut c_void,
        ];
        unsafe { self.stream.launch(&func, grid, block, 0, &mut args) }
            .map_err(|e| CudaPmaError::Cuda(e.to_string()))
    }

    pub fn pma_batch_dev(&self, prices: &[f32], sweep: &PmaBatchRange) -> Result<DevicePmaPair, CudaPmaError> {
        let inputs = Self::prepare_batch_inputs(prices, sweep)?; self.run_batch_kernel(prices, &inputs)
    }

    pub fn pma_batch_into_host_f32(&self, prices: &[f32], sweep: &PmaBatchRange, out_predict: &mut [f32], out_trigger: &mut [f32])
        -> Result<(usize, usize), CudaPmaError>
    {
        let inputs = Self::prepare_batch_inputs(prices, sweep)?;
        let expected = inputs.series_len * inputs.combos;
        if out_predict.len() != expected || out_trigger.len() != expected {
            return Err(CudaPmaError::InvalidInput(format!("output slice wrong length: got p={}, t={}, expected={}", out_predict.len(), out_trigger.len(), expected)));
        }
        let pair = self.run_batch_kernel(prices, &inputs)?;
        pair.predict.buf.copy_to(out_predict).map_err(|e| CudaPmaError::Cuda(e.to_string()))?;
        pair.trigger.buf.copy_to(out_trigger).map_err(|e| CudaPmaError::Cuda(e.to_string()))?;
        Ok((pair.rows(), pair.cols()))
    }

    // ---------------- Many series × one param (time-major) ----------------
    fn prepare_many_series_inputs(prices_tm: &[f32], cols: usize, rows: usize) -> Result<Vec<i32>, CudaPmaError> {
        if cols == 0 || rows == 0 { return Err(CudaPmaError::InvalidInput("num_series or series_len is zero".into())); }
        if prices_tm.len() != cols * rows { return Err(CudaPmaError::InvalidInput(format!("data length {} != cols*rows {}", prices_tm.len(), cols*rows))); }
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for r in 0..rows {
                let v = prices_tm[r * cols + s]; if !v.is_nan() { fv = Some(r); break; }
            }
            let idx = fv.ok_or_else(|| CudaPmaError::InvalidInput(format!("series {} is entirely NaN", s)))?;
            if rows - idx < 7 { return Err(CudaPmaError::InvalidInput(format!("series {} lacks warmup samples (valid = {})", s, rows - idx))); }
            first_valids[s] = idx as i32;
        }
        Ok(first_valids)
    }

    fn launch_many_series_kernel_select(&self,
        d_prices_tm: &DeviceBuffer<f32>, cols: usize, rows: usize, d_first_valids: &DeviceBuffer<i32>,
        d_predict_tm: &mut DeviceBuffer<f32>, d_trigger_tm: &mut DeviceBuffer<f32>
    ) -> Result<(), CudaPmaError> {
        let (fname, block, grid, sel) = match self.policy.many_series {
            ManySeriesKernelPolicy::Tiled2D { tx, ty } => (
                match (tx, ty) { (1,4) => "pma_ms1p_tiled_f32_tx1_ty4", (1,2) => "pma_ms1p_tiled_f32_tx1_ty2", _ => "pma_many_series_one_param_f32" },
                BlockSize::xyz(tx.max(1), ty.max(1), 1),
                { let gx = ((cols as u32) + ty - 1) / ty; GridSize::xyz(gx, 1, 1) },
                Some(ManySeriesKernelSelected::Tiled2D { tx, ty })
            ),
            ManySeriesKernelPolicy::OneD { block_x } => (
                "pma_many_series_one_param_f32", BlockSize::xy(block_x.max(1), 1), GridSize::xyz(cols as u32, 1, 1), Some(ManySeriesKernelSelected::OneD { block_x })
            ),
            ManySeriesKernelPolicy::Auto => {
                if cols >= 16 && self.module.get_function("pma_ms1p_tiled_f32_tx1_ty4").is_ok() {
                    let gx = ((cols as u32) + 4 - 1) / 4; ("pma_ms1p_tiled_f32_tx1_ty4", BlockSize::xyz(1,4,1), GridSize::xyz(gx,1,1), Some(ManySeriesKernelSelected::Tiled2D { tx:1, ty:4 }))
                } else if cols >= 8 && self.module.get_function("pma_ms1p_tiled_f32_tx1_ty2").is_ok() {
                    let gx = ((cols as u32) + 2 - 1) / 2; ("pma_ms1p_tiled_f32_tx1_ty2", BlockSize::xyz(1,2,1), GridSize::xyz(gx,1,1), Some(ManySeriesKernelSelected::Tiled2D { tx:1, ty:2 }))
                } else {
                    ("pma_many_series_one_param_f32", BlockSize::xy(1,1), GridSize::xyz(cols as u32, 1, 1), Some(ManySeriesKernelSelected::OneD { block_x: 1 }))
                }
            }
        };
        if let Some(s) = sel { unsafe { (*(self as *const _ as *mut CudaPma)).last_many = Some(s); } }
        self.maybe_log_many_debug();

        let func = self.module.get_function(fname).map_err(|e| CudaPmaError::Cuda(e.to_string()))?;
        let mut args: [*mut c_void; 6] = [
            &mut d_prices_tm.as_device_ptr().as_raw() as *mut _ as *mut c_void,
            &mut (cols as i32) as *mut _ as *mut c_void,
            &mut (rows as i32) as *mut _ as *mut c_void,
            &mut d_first_valids.as_device_ptr().as_raw() as *mut _ as *mut c_void,
            &mut d_predict_tm.as_device_ptr().as_raw() as *mut _ as *mut c_void,
            &mut d_trigger_tm.as_device_ptr().as_raw() as *mut _ as *mut c_void,
        ];
        unsafe { self.stream.launch(&func, grid, block, 0, &mut args) }
            .map_err(|e| CudaPmaError::Cuda(e.to_string()))
    }

    pub fn pma_many_series_one_param_time_major_dev(&self, prices_tm: &[f32], cols: usize, rows: usize)
        -> Result<DevicePmaPair, CudaPmaError>
    {
        let first_valids = Self::prepare_many_series_inputs(prices_tm, cols, rows)?;
        let prices_bytes = cols * rows * core::mem::size_of::<f32>();
        let first_bytes = cols * core::mem::size_of::<i32>();
        let out_bytes = cols * rows * core::mem::size_of::<f32>();
        let required = prices_bytes + first_bytes + 2 * out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaPmaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let mut d_prices_tm: DeviceBuffer<f32> = DeviceBuffer::from_slice(prices_tm)
            .map_err(|e| CudaPmaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaPmaError::Cuda(e.to_string()))?;
        let mut d_predict_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaPmaError::Cuda(e.to_string()))?;
        let mut d_trigger_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaPmaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel_select(&d_prices_tm, cols, rows, &d_first_valids, &mut d_predict_tm, &mut d_trigger_tm)?;

        Ok(DevicePmaPair {
            predict: DeviceArrayF32 { buf: d_predict_tm, rows, cols },
            trigger: DeviceArrayF32 { buf: d_trigger_tm, rows, cols },
        })
    }
}

// ---------- Bench profiles ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const MANY_SERIES_COLS: usize = 256;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * core::mem::size_of::<f32>();
        // combos fixed = 1; two outputs
        let out_bytes = 2 * ONE_SERIES_LEN * core::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = elems * core::mem::size_of::<f32>();
        let out_bytes = 2 * elems * core::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct PmaBatchState { cuda: CudaPma, price: Vec<f32>, sweep: PmaBatchRange }
    impl CudaBenchState for PmaBatchState { fn launch(&mut self) { let _ = self.cuda.pma_batch_dev(&self.price, &self.sweep).unwrap(); } }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaPma::new(0).expect("cuda pma");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = PmaBatchRange::default();
        Box::new(PmaBatchState { cuda, price, sweep })
    }

    struct PmaManyState { cuda: CudaPma, data_tm: Vec<f32>, cols: usize, rows: usize }
    impl CudaBenchState for PmaManyState { fn launch(&mut self) { let _ = self.cuda.pma_many_series_one_param_time_major_dev(&self.data_tm, self.cols, self.rows).unwrap(); } }
    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaPma::new(0).expect("cuda pma");
        let cols = MANY_SERIES_COLS; let rows = MANY_SERIES_LEN; let data_tm = gen_time_major_prices(cols, rows);
        Box::new(PmaManyState { cuda, data_tm, cols, rows })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new("pma", "one_series_many_params", "pma_cuda_batch_dev", "1m_x1", prep_one_series_many_params)
                .with_sample_size(10)
                .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new("pma", "many_series_one_param", "pma_cuda_many_series_one_param", "256x1m", prep_many_series_one_param)
                .with_sample_size(5)
                .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
