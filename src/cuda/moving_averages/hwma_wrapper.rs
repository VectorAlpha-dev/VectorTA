//! CUDA wrapper for Holt-Winters Moving Average (HWMA) kernels.
//!
//! Aligned with the ALMA wrapper conventions:
//! - JIT options with DetermineTargetFromContext and O2 fallback.
//! - Policy enums + introspection for selected kernels.
//! - VRAM estimation (with ~64MB headroom) and early failure when insufficient.
//! - Chunked launches for large parameter sweeps.
//! - NON_BLOCKING stream and zero-copy DeviceArrayF32 return type.
//!
//! HWMA is a recurrence (time‑marching) indicator: each thread scans time for a
//! single parameter combo (batch) or a single series (many‑series).

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::hwma::{expand_grid, HwmaBatchRange, HwmaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
pub enum CudaHwmaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaHwmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaHwmaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaHwmaError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaHwmaError {}

// -------- Kernel policy + introspection (subset for recurrence) --------

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
pub struct CudaHwmaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaHwmaPolicy {
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
}
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaHwma {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaHwmaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaHwma {
    pub fn new(device_id: usize) -> Result<Self, CudaHwmaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaHwmaError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaHwmaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaHwmaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/hwma_kernel.ptx"));
        // Match ALMA: prefer DetermineTargetFromContext + O2; fall back to simpler modes.
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => match Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                Ok(m) => m,
                Err(_) => {
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaHwmaError::Cuda(e.to_string()))?
                }
            },
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaHwmaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaHwmaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(
        device_id: usize,
        policy: CudaHwmaPolicy,
    ) -> Result<Self, CudaHwmaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaHwmaPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaHwmaPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
    pub fn synchronize(&self) -> Result<(), CudaHwmaError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaHwmaError::Cuda(e.to_string()))
    }

    // VRAM helpers
    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
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
    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] HWMA batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaHwma)).debug_batch_logged = true;
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
                let per = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] HWMA many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaHwma)).debug_many_logged = true;
                }
            }
        }
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &HwmaBatchRange,
    ) -> Result<(Vec<HwmaParams>, usize, usize), CudaHwmaError> {
        if data_f32.is_empty() {
            return Err(CudaHwmaError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaHwmaError::InvalidInput("all values are NaN".into()))?;
        let len = data_f32.len();

        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaHwmaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        for (idx, prm) in combos.iter().enumerate() {
            let na = prm.na.unwrap_or(0.2);
            let nb = prm.nb.unwrap_or(0.1);
            let nc = prm.nc.unwrap_or(0.1);
            if !na.is_finite() || !nb.is_finite() || !nc.is_finite() {
                return Err(CudaHwmaError::InvalidInput(format!(
                    "params[{}] contain non-finite values",
                    idx
                )));
            }
            if !(na > 0.0 && na < 1.0 && nb > 0.0 && nb < 1.0 && nc > 0.0 && nc < 1.0) {
                return Err(CudaHwmaError::InvalidInput(format!(
                    "params[{}] must lie in (0,1): na={}, nb={}, nc={}",
                    idx, na, nb, nc
                )));
            }
        }

        Ok((combos, first_valid, len))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_nas: &DeviceBuffer<f32>,
        d_nbs: &DeviceBuffer<f32>,
        d_ncs: &DeviceBuffer<f32>,
        first_valid: usize,
        series_len: usize,
        n_combos: usize,
        start_combo: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaHwmaError> {
        if series_len == 0 || n_combos == 0 {
            return Err(CudaHwmaError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }
        if series_len > i32::MAX as usize || n_combos > i32::MAX as usize {
            return Err(CudaHwmaError::InvalidInput(
                "series_len or n_combos exceed i32::MAX".into(),
            ));
        }
        if first_valid > i32::MAX as usize {
            return Err(CudaHwmaError::InvalidInput(
                "first_valid exceeds i32::MAX".into(),
            ));
        }

        let func = self
            .module
            .get_function("hwma_batch_f32")
            .map_err(|e| CudaHwmaError::Cuda(e.to_string()))?;

        // Policy/env override for block size
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            BatchKernelPolicy::Auto => std::env::var("HWMA_BLOCK_X")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .filter(|&v| matches!(v, 64 | 128 | 256 | 512))
                .unwrap_or(128),
        };
        let grid_x = ((n_combos as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        // Introspection
        unsafe {
            (*(self as *const _ as *mut CudaHwma)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            // Offset param/output pointers by start_combo when chunking
            let mut nas_ptr = d_nas.as_device_ptr().add(start_combo).as_raw();
            let mut nbs_ptr = d_nbs.as_device_ptr().add(start_combo).as_raw();
            let mut ncs_ptr = d_ncs.as_device_ptr().add(start_combo).as_raw();
            let mut first_valid_i = first_valid as i32;
            let mut series_len_i = series_len as i32;
            let mut n_combos_i = n_combos as i32;
            let mut out_ptr = d_out.as_device_ptr().add(start_combo * series_len).as_raw();
            let args: &mut [*mut std::ffi::c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut std::ffi::c_void,
                &mut nas_ptr as *mut _ as *mut std::ffi::c_void,
                &mut nbs_ptr as *mut _ as *mut std::ffi::c_void,
                &mut ncs_ptr as *mut _ as *mut std::ffi::c_void,
                &mut first_valid_i as *mut _ as *mut std::ffi::c_void,
                &mut series_len_i as *mut _ as *mut std::ffi::c_void,
                &mut n_combos_i as *mut _ as *mut std::ffi::c_void,
                &mut out_ptr as *mut _ as *mut std::ffi::c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaHwmaError::Cuda(e.to_string()))?;
        }

        Ok(())
    }

    pub fn hwma_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_nas: &DeviceBuffer<f32>,
        d_nbs: &DeviceBuffer<f32>,
        d_ncs: &DeviceBuffer<f32>,
        first_valid: usize,
        series_len: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaHwmaError> {
        self.launch_batch_kernel(
            d_prices,
            d_nas,
            d_nbs,
            d_ncs,
            first_valid,
            series_len,
            n_combos,
            0,
            d_out,
        )
    }

    pub fn hwma_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &HwmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaHwmaError> {
        let (combos, first_valid, series_len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = combos.len();

        // VRAM estimate: prices + 3 param arrays + output
        let prices_bytes = series_len * std::mem::size_of::<f32>();
        let params_bytes = 3 * n_combos * std::mem::size_of::<f32>();
        let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + params_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaHwmaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaHwmaError::Cuda(e.to_string()))?;

        let mut nas = Vec::with_capacity(n_combos);
        let mut nbs = Vec::with_capacity(n_combos);
        let mut ncs = Vec::with_capacity(n_combos);
        for prm in &combos {
            nas.push(prm.na.unwrap_or(0.2) as f32);
            nbs.push(prm.nb.unwrap_or(0.1) as f32);
            ncs.push(prm.nc.unwrap_or(0.1) as f32);
        }
        let d_nas =
            DeviceBuffer::from_slice(&nas).map_err(|e| CudaHwmaError::Cuda(e.to_string()))?;
        let d_nbs =
            DeviceBuffer::from_slice(&nbs).map_err(|e| CudaHwmaError::Cuda(e.to_string()))?;
        let d_ncs =
            DeviceBuffer::from_slice(&ncs).map_err(|e| CudaHwmaError::Cuda(e.to_string()))?;

        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaHwmaError::Cuda(e.to_string()))?;

        // Chunk to <= 65_535 combos per launch (mirrors ALMA grid.y chunking behavior)
        const MAX_CHUNK: usize = 65_535;
        let mut launched = 0usize;
        while launched < n_combos {
            let len = (n_combos - launched).min(MAX_CHUNK);
            self.launch_batch_kernel(
                &d_prices,
                &d_nas,
                &d_nbs,
                &d_ncs,
                first_valid,
                series_len,
                len,
                launched,
                &mut d_out,
            )?;
            launched += len;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaHwmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &HwmaParams,
    ) -> Result<(Vec<i32>, f32, f32, f32), CudaHwmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaHwmaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaHwmaError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }

        let na = params.na.unwrap_or(0.2);
        let nb = params.nb.unwrap_or(0.1);
        let nc = params.nc.unwrap_or(0.1);
        if !na.is_finite() || !nb.is_finite() || !nc.is_finite() {
            return Err(CudaHwmaError::InvalidInput(
                "parameters must be finite".into(),
            ));
        }
        if !(na > 0.0 && na < 1.0 && nb > 0.0 && nb < 1.0 && nc > 0.0 && nc < 1.0) {
            return Err(CudaHwmaError::InvalidInput(format!(
                "parameters must lie in (0,1): na={}, nb={}, nc={}",
                na, nb, nc
            )));
        }

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut found = None;
            for row in 0..rows {
                let idx = row * cols + series;
                if !data_tm_f32[idx].is_nan() {
                    found = Some(row);
                    break;
                }
            }
            let fv = found.ok_or_else(|| {
                CudaHwmaError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            if fv > i32::MAX as usize {
                return Err(CudaHwmaError::InvalidInput(
                    "first_valid exceeds i32::MAX".into(),
                ));
            }
            first_valids[series] = fv as i32;
        }

        Ok((first_valids, na as f32, nb as f32, nc as f32))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        na: f32,
        nb: f32,
        nc: f32,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaHwmaError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaHwmaError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if num_series > i32::MAX as usize || series_len > i32::MAX as usize {
            return Err(CudaHwmaError::InvalidInput(
                "num_series or series_len exceed i32::MAX".into(),
            ));
        }

        let func = self
            .module
            .get_function("hwma_multi_series_one_param_f32")
            .map_err(|e| CudaHwmaError::Cuda(e.to_string()))?;

        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            ManySeriesKernelPolicy::Auto => std::env::var("HWMA_MS_BLOCK_X")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .filter(|&v| matches!(v, 64 | 128 | 256 | 512))
                .unwrap_or(128),
        };
        let grid_x = ((num_series as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            (*(self as *const _ as *mut CudaHwma)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut na_f = na;
            let mut nb_f = nb;
            let mut nc_f = nc;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut std::ffi::c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut std::ffi::c_void,
                &mut na_f as *mut _ as *mut std::ffi::c_void,
                &mut nb_f as *mut _ as *mut std::ffi::c_void,
                &mut nc_f as *mut _ as *mut std::ffi::c_void,
                &mut num_series_i as *mut _ as *mut std::ffi::c_void,
                &mut series_len_i as *mut _ as *mut std::ffi::c_void,
                &mut first_valids_ptr as *mut _ as *mut std::ffi::c_void,
                &mut out_ptr as *mut _ as *mut std::ffi::c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaHwmaError::Cuda(e.to_string()))?;
        }

        Ok(())
    }

    pub fn hwma_multi_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        na: f32,
        nb: f32,
        nc: f32,
        num_series: i32,
        series_len: i32,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaHwmaError> {
        if num_series <= 0 || series_len <= 0 {
            return Err(CudaHwmaError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_prices_tm,
            na,
            nb,
            nc,
            num_series as usize,
            series_len as usize,
            d_first_valids,
            d_out_tm,
        )
    }

    pub fn hwma_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &HwmaParams,
    ) -> Result<DeviceArrayF32, CudaHwmaError> {
        let (first_valids, na, nb, nc) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        // VRAM: prices + first_valids + output
        let prices_bytes = cols * rows * std::mem::size_of::<f32>();
        let first_bytes = cols * std::mem::size_of::<i32>();
        let out_bytes = cols * rows * std::mem::size_of::<f32>();
        let required = prices_bytes + first_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaHwmaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaHwmaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaHwmaError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaHwmaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            na,
            nb,
            nc,
            cols,
            rows,
            &d_first_valids,
            &mut d_out_tm,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaHwmaError::Cuda(e.to_string()))?;

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
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        hwma_benches,
        CudaHwma,
        crate::indicators::moving_averages::hwma::HwmaBatchRange,
        crate::indicators::moving_averages::hwma::HwmaParams,
        hwma_batch_dev,
        hwma_multi_series_one_param_time_major_dev,
        crate::indicators::moving_averages::hwma::HwmaBatchRange {
            na: (0.05, 0.05 + (PARAM_SWEEP as f64 - 1.0) * 0.001, 0.001),
            nb: (0.1, 0.1, 0.0),
            nc: (0.1, 0.1, 0.0)
        },
        crate::indicators::moving_averages::hwma::HwmaParams {
            na: Some(0.2),
            nb: Some(0.1),
            nc: Some(0.1)
        },
        "hwma",
        "hwma"
    );
    pub use hwma_benches::bench_profiles;
}
