//! CUDA support for the Slope Adaptive Moving Average (SAMA).
//!
//! Mirrors the ALMA/CWMA CUDA API surface by exposing zero‑copy device entry
//! points for both the one‑series × many‑parameter sweep and the time‑major
//! many‑series × one‑parameter scenario. Kernels operate fully in FP32 and
//! reuse host‑prepared alpha coefficients to keep GPU‑side work focused on the
//! adaptive recurrence itself. Wrapper adds ALMA‑parity features: policy enums,
//! JIT options (DetermineTargetFromContext + O2 with fallbacks), NON_BLOCKING
//! stream, VRAM checks with ~64MB headroom, chunked launches, and BENCH_DEBUG
//! logging of selected kernels.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
// Reuse shared policy enums exported by CWMA for ALMA‑parity API shape.
use super::{BatchKernelPolicy, ManySeriesKernelPolicy};
use crate::indicators::moving_averages::sama::{SamaBatchRange, SamaParams};
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

#[derive(Debug)]
pub enum CudaSamaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaSamaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaSamaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaSamaError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}

impl std::error::Error for CudaSamaError {}

pub struct CudaSama {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    // Policy + introspection for parity with ALMA/CWMA wrappers
    policy: CudaSamaPolicy,
    last_batch: Option<SamaBatchKernelSelected>,
    last_many: Option<SamaManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

// -------- Kernel selection policy (explicit for tests; Auto for production) --------

#[derive(Clone, Copy, Debug)]
pub struct CudaSamaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaSamaPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

// -------- Introspection (selected kernel) --------

#[derive(Clone, Copy, Debug)]
pub enum SamaBatchKernelSelected {
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum SamaManySeriesKernelSelected {
    OneD { block_x: u32 },
}

struct PreparedSamaBatch {
    combos: Vec<SamaParams>,
    first_valid: usize,
    series_len: usize,
    lengths_i32: Vec<i32>,
    min_alphas: Vec<f32>,
    maj_alphas: Vec<f32>,
    first_valids: Vec<i32>,
}

struct PreparedSamaManySeries {
    first_valids: Vec<i32>,
    length: i32,
    min_alpha: f32,
    maj_alpha: f32,
}

impl CudaSama {
    pub fn new(device_id: usize) -> Result<Self, CudaSamaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaSamaError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaSamaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaSamaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/sama_kernel.ptx"));
        // Match ALMA/CWMA JIT policy for broad driver support and perf
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
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaSamaError::Cuda(e.to_string()))?
                }
            }
        };

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaSamaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaSamaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    /// Create with an explicit policy (mirrors ALMA/CWMA convenience).
    pub fn new_with_policy(
        device_id: usize,
        policy: CudaSamaPolicy,
    ) -> Result<Self, CudaSamaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }

    #[inline]
    pub fn set_policy(&mut self, policy: CudaSamaPolicy) {
        self.policy = policy;
    }
    #[inline]
    pub fn policy(&self) -> &CudaSamaPolicy {
        &self.policy
    }
    #[inline]
    pub fn selected_batch_kernel(&self) -> Option<SamaBatchKernelSelected> {
        self.last_batch
    }
    #[inline]
    pub fn selected_many_series_kernel(&self) -> Option<SamaManySeriesKernelSelected> {
        self.last_many
    }

    pub fn sama_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &SamaBatchRange,
    ) -> Result<DeviceArrayF32, CudaSamaError> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = prepared.combos.len();

        // VRAM estimate (prices + params + output)
        let prices_bytes = prepared.series_len * std::mem::size_of::<f32>();
        let params_bytes = n_combos
            * (std::mem::size_of::<i32>() // lengths
                + 2 * std::mem::size_of::<f32>() // alphas
                + std::mem::size_of::<i32>()); // first_valids
        let out_bytes = n_combos * prepared.series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + params_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024; // ~64MB
        if !Self::will_fit(required, headroom) {
            return Err(CudaSamaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaSamaError::Cuda(e.to_string()))?;
        let d_lengths = DeviceBuffer::from_slice(&prepared.lengths_i32)
            .map_err(|e| CudaSamaError::Cuda(e.to_string()))?;
        let d_min = DeviceBuffer::from_slice(&prepared.min_alphas)
            .map_err(|e| CudaSamaError::Cuda(e.to_string()))?;
        let d_maj = DeviceBuffer::from_slice(&prepared.maj_alphas)
            .map_err(|e| CudaSamaError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&prepared.first_valids)
            .map_err(|e| CudaSamaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(prepared.series_len * n_combos)
                .map_err(|e| CudaSamaError::Cuda(e.to_string()))?
        };

        self.launch_batch_kernel_sliced(
            &d_prices,
            &d_lengths,
            &d_min,
            &d_maj,
            &d_first,
            prepared.series_len,
            n_combos,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: prepared.series_len,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sama_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_lengths: &DeviceBuffer<i32>,
        d_min_alphas: &DeviceBuffer<f32>,
        d_maj_alphas: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSamaError> {
        if series_len == 0 {
            return Err(CudaSamaError::InvalidInput(
                "series_len must be positive".into(),
            ));
        }
        if n_combos == 0 {
            return Err(CudaSamaError::InvalidInput(
                "n_combos must be positive".into(),
            ));
        }
        if d_lengths.len() != n_combos
            || d_min_alphas.len() != n_combos
            || d_maj_alphas.len() != n_combos
            || d_first_valids.len() != n_combos
        {
            return Err(CudaSamaError::InvalidInput(
                "device buffer length mismatch".into(),
            ));
        }
        if d_prices.len() != series_len {
            return Err(CudaSamaError::InvalidInput(
                "prices length must equal series_len".into(),
            ));
        }
        if d_out.len() != n_combos * series_len {
            return Err(CudaSamaError::InvalidInput(
                "output buffer length mismatch".into(),
            ));
        }

        self.launch_batch_kernel_sliced(
            d_prices,
            d_lengths,
            d_min_alphas,
            d_maj_alphas,
            d_first_valids,
            series_len,
            n_combos,
            d_out,
        )
    }

    pub fn sama_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &SamaBatchRange,
        out_flat: &mut [f32],
    ) -> Result<(), CudaSamaError> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        if out_flat.len() != prepared.series_len * prepared.combos.len() {
            return Err(CudaSamaError::InvalidInput(
                "output slice length mismatch".into(),
            ));
        }
        let handle = self.sama_batch_dev(data_f32, sweep)?;
        handle
            .buf
            .copy_to(out_flat)
            .map_err(|e| CudaSamaError::Cuda(e.to_string()))
    }

    pub fn sama_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &SamaParams,
    ) -> Result<DeviceArrayF32, CudaSamaError> {
        let prepared =
            Self::prepare_many_series_inputs(data_tm_f32, num_series, series_len, params)?;

        // VRAM estimate (prices + first_valids + out)
        let prices_bytes = num_series * series_len * std::mem::size_of::<f32>();
        let fv_bytes = num_series * std::mem::size_of::<i32>();
        let out_bytes = num_series * series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + fv_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaSamaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaSamaError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&prepared.first_valids)
            .map_err(|e| CudaSamaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(num_series * series_len)
                .map_err(|e| CudaSamaError::Cuda(e.to_string()))?
        };

        self.launch_many_series_kernel(
            &d_prices,
            &d_first,
            prepared.length,
            prepared.min_alpha,
            prepared.maj_alpha,
            num_series,
            series_len,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: series_len,
            cols: num_series,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sama_many_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        length: i32,
        min_alpha: f32,
        maj_alpha: f32,
        num_series: i32,
        series_len: i32,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSamaError> {
        if num_series <= 0 || series_len <= 0 {
            return Err(CudaSamaError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if length <= 0 {
            return Err(CudaSamaError::InvalidInput(
                "length must be positive".into(),
            ));
        }
        if d_first_valids.len() != num_series as usize {
            return Err(CudaSamaError::InvalidInput(
                "first_valids buffer length mismatch".into(),
            ));
        }
        let total = num_series as usize * series_len as usize;
        if d_prices_tm.len() != total || d_out_tm.len() != total {
            return Err(CudaSamaError::InvalidInput(
                "time-major buffer length mismatch".into(),
            ));
        }

        self.launch_many_series_kernel(
            d_prices_tm,
            d_first_valids,
            length,
            min_alpha,
            maj_alpha,
            num_series as usize,
            series_len as usize,
            d_out_tm,
        )
    }

    pub fn sama_many_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &SamaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaSamaError> {
        if out_tm.len() != num_series * series_len {
            return Err(CudaSamaError::InvalidInput(
                "output slice length mismatch".into(),
            ));
        }
        let handle = self.sama_many_series_one_param_time_major_dev(
            data_tm_f32,
            num_series,
            series_len,
            params,
        )?;
        handle
            .buf
            .copy_to(out_tm)
            .map_err(|e| CudaSamaError::Cuda(e.to_string()))
    }

    // Launch helper that chunks across parameter combos to respect grid limits
    // and supports large sweeps. Each chunk updates param/output pointers.
    fn launch_batch_kernel_sliced(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_lengths: &DeviceBuffer<i32>,
        d_min_alphas: &DeviceBuffer<f32>,
        d_maj_alphas: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSamaError> {
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Auto
            | BatchKernelPolicy::Plain { .. }
            | BatchKernelPolicy::Tiled { .. } => 256u32,
        };
        let block: BlockSize = (block_x, 1, 1).into();

        let func = self
            .module
            .get_function("sama_batch_f32")
            .map_err(|e| CudaSamaError::Cuda(e.to_string()))?;

        // Select kernel once for introspection/debugging
        unsafe {
            (*(self as *const _ as *mut CudaSama)).last_batch =
                Some(SamaBatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        // Chunk across combos to keep launch sizes manageable (<= 65_535 per slice)
        const MAX_SLICE: usize = 65_535; // conservative even though grid.x allows much larger
        let mut start = 0usize;
        while start < n_combos {
            let len = (n_combos - start).min(MAX_SLICE);
            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                let mut lengths_ptr = d_lengths.as_device_ptr().add(start).as_raw();
                let mut min_ptr = d_min_alphas.as_device_ptr().add(start).as_raw();
                let mut maj_ptr = d_maj_alphas.as_device_ptr().add(start).as_raw();
                let mut first_ptr = d_first_valids.as_device_ptr().add(start).as_raw();
                let mut series_len_i = series_len as i32;
                let mut combos_i = len as i32;
                let mut out_ptr = d_out.as_device_ptr().add(start * series_len).as_raw();
                let mut args: [*mut c_void; 8] = [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut lengths_ptr as *mut _ as *mut c_void,
                    &mut min_ptr as *mut _ as *mut c_void,
                    &mut maj_ptr as *mut _ as *mut c_void,
                    &mut first_ptr as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut combos_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                let grid: GridSize = (len as u32, 1, 1).into();
                self.stream
                    .launch(&func, grid, block, 0, &mut args)
                    .map_err(|e| CudaSamaError::Cuda(e.to_string()))?;
            }
            start += len;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaSamaError::Cuda(e.to_string()))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        length: i32,
        min_alpha: f32,
        maj_alpha: f32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSamaError> {
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto
            | ManySeriesKernelPolicy::OneD { .. }
            | ManySeriesKernelPolicy::Tiled2D { .. } => 256u32,
        };
        let block: BlockSize = (block_x, 1, 1).into();
        let grid: GridSize = (num_series as u32, 1, 1).into();

        let func = self
            .module
            .get_function("sama_many_series_one_param_f32")
            .map_err(|e| CudaSamaError::Cuda(e.to_string()))?;

        // Introspection for benches/debug
        unsafe {
            (*(self as *const _ as *mut CudaSama)).last_many =
                Some(SamaManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut length_i = length;
            let mut min_a = min_alpha;
            let mut maj_a = maj_alpha;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 8] = [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut length_i as *mut _ as *mut c_void,
                &mut min_a as *mut _ as *mut c_void,
                &mut maj_a as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(|e| CudaSamaError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaSamaError::Cuda(e.to_string()))
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &SamaBatchRange,
    ) -> Result<PreparedSamaBatch, CudaSamaError> {
        if data_f32.is_empty() {
            return Err(CudaSamaError::InvalidInput("input data is empty".into()));
        }

        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaSamaError::InvalidInput(
                "no parameter combinations provided".into(),
            ));
        }

        let series_len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| v.is_finite())
            .ok_or_else(|| CudaSamaError::InvalidInput("all values are NaN".into()))?;
        if series_len - first_valid < 1 {
            return Err(CudaSamaError::InvalidInput(
                "not enough valid data to start computation".into(),
            ));
        }

        let mut lengths_i32 = Vec::with_capacity(combos.len());
        let mut min_alphas = Vec::with_capacity(combos.len());
        let mut maj_alphas = Vec::with_capacity(combos.len());
        let mut first_valids = Vec::with_capacity(combos.len());

        for params in &combos {
            let length = params.length.unwrap_or(200);
            let maj_length = params.maj_length.unwrap_or(14);
            let min_length = params.min_length.unwrap_or(6);

            if length == 0 || maj_length == 0 || min_length == 0 {
                return Err(CudaSamaError::InvalidInput(
                    "length, maj_length, and min_length must be positive".into(),
                ));
            }
            if length + 1 > series_len {
                return Err(CudaSamaError::InvalidInput(format!(
                    "length {} exceeds available data {}",
                    length + 1,
                    series_len
                )));
            }

            let min_alpha = 2.0f32 / (min_length as f32 + 1.0f32);
            let maj_alpha = 2.0f32 / (maj_length as f32 + 1.0f32);

            lengths_i32.push(length as i32);
            min_alphas.push(min_alpha);
            maj_alphas.push(maj_alpha);
            first_valids.push(first_valid as i32);
        }

        Ok(PreparedSamaBatch {
            combos,
            first_valid,
            series_len,
            lengths_i32,
            min_alphas,
            maj_alphas,
            first_valids,
        })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &SamaParams,
    ) -> Result<PreparedSamaManySeries, CudaSamaError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaSamaError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if data_tm_f32.len() != num_series * series_len {
            return Err(CudaSamaError::InvalidInput(
                "time-major slice length mismatch".into(),
            ));
        }

        let length = params.length.unwrap_or(200) as i32;
        let maj_length = params.maj_length.unwrap_or(14);
        let min_length = params.min_length.unwrap_or(6);
        if length <= 0 || maj_length == 0 || min_length == 0 {
            return Err(CudaSamaError::InvalidInput(
                "length, maj_length, and min_length must be positive".into(),
            ));
        }
        if (length as usize) + 1 > series_len {
            return Err(CudaSamaError::InvalidInput(format!(
                "length {} exceeds available data {}",
                length as usize + 1,
                series_len
            )));
        }

        let mut first_valids = Vec::with_capacity(num_series);
        for series in 0..num_series {
            let mut fv = None;
            for t in 0..series_len {
                let v = data_tm_f32[t * num_series + series];
                if v.is_finite() {
                    fv = Some(t as i32);
                    break;
                }
            }
            let fv = fv.ok_or_else(|| {
                CudaSamaError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            if series_len - (fv as usize) < 1 {
                return Err(CudaSamaError::InvalidInput(format!(
                    "series {} does not have enough valid data",
                    series
                )));
            }
            first_valids.push(fv);
        }

        let min_alpha = 2.0f32 / (min_length as f32 + 1.0f32);
        let maj_alpha = 2.0f32 / (maj_length as f32 + 1.0f32);

        Ok(PreparedSamaManySeries {
            first_valids,
            length,
            min_alpha,
            maj_alpha,
        })
    }
    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
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
                    eprintln!("[DEBUG] SAMA batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaSama)).debug_batch_logged = true;
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
                    eprintln!("[DEBUG] SAMA many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaSama)).debug_many_logged = true;
                }
            }
        }
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        sama_benches,
        CudaSama,
        crate::indicators::moving_averages::sama::SamaBatchRange,
        crate::indicators::moving_averages::sama::SamaParams,
        sama_batch_dev,
        sama_many_series_one_param_time_major_dev,
        crate::indicators::moving_averages::sama::SamaBatchRange {
            length: (64, 64 + PARAM_SWEEP - 1, 1),
            maj_length: (14, 14, 0),
            min_length: (6, 6, 0)
        },
        crate::indicators::moving_averages::sama::SamaParams {
            length: Some(64),
            maj_length: Some(14),
            min_length: Some(6)
        },
        "sama",
        "sama"
    );
    pub use sama_benches::bench_profiles;
}

fn expand_grid(range: &SamaBatchRange) -> Vec<SamaParams> {
    fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }

    let lengths = axis(range.length);
    let maj_lengths = axis(range.maj_length);
    let min_lengths = axis(range.min_length);

    let mut out = Vec::with_capacity(lengths.len() * maj_lengths.len() * min_lengths.len());
    for &len in &lengths {
        for &maj in &maj_lengths {
            for &min in &min_lengths {
                out.push(SamaParams {
                    length: Some(len),
                    maj_length: Some(maj),
                    min_length: Some(min),
                });
            }
        }
    }
    out
}
