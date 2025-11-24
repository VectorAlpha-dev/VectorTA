//! CUDA support for the Williams' %R (WILLR) indicator.
//!
//! Mirrors the CPU batching API by accepting a single price series with many
//! period combinations. Kernels operate in FP32 and replicate the scalar
//! semantics (warm-up NaNs, NaN propagation, zero denominator handling).

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::willr::{
    build_willr_gpu_tables, WillrBatchRange, WillrGpuTables, WillrParams,
};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaWillrError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

pub struct CudaWillr {
    module: Module,
    stream: Stream,
    ctx: Arc<Context>,
    device_id: u32,
}

/// Device-resident sparse-table buffers for WILLR, reusable across runs.
pub struct WillrGpuTablesDev {
    d_log2: DeviceBuffer<i32>,
    d_level_offsets: DeviceBuffer<i32>,
    d_st_max: DeviceBuffer<f32>,
    d_st_min: DeviceBuffer<f32>,
    d_nan_psum: DeviceBuffer<i32>,
    pub series_len: usize,
    pub first_valid: usize,
    pub level_count: usize,
}

struct PreparedWillrBatch {
    combos: Vec<WillrParams>,
    first_valid: usize,
    series_len: usize,
    tables: WillrGpuTables,
}

impl CudaWillr {
    pub fn new(device_id: usize) -> Result<Self, CudaWillrError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/willr_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            // Prefer higher driver JIT optimization; fallback below if unsupported.
            ModuleJitOption::OptLevel(OptLevel::O3),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => Module::from_ptx(ptx, &[])?,
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            ctx: context,
            device_id: device_id as u32,
        })
    }

    /// Expose context/device for Python interop and tests.
    #[inline]
    pub fn context(&self) -> Arc<Context> {
        self.ctx.clone()
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }

    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaWillrError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Ok((free, _total)) = mem_get_info() {
            if required_bytes.saturating_add(headroom_bytes) > free {
                return Err(CudaWillrError::OutOfMemory {
                    required: required_bytes,
                    free,
                    headroom: headroom_bytes,
                });
            }
        }
        Ok(())
    }

    pub fn willr_batch_dev(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        close_f32: &[f32],
        sweep: &WillrBatchRange,
    ) -> Result<DeviceArrayF32, CudaWillrError> {
        let prepared = Self::prepare_batch_inputs(high_f32, low_f32, close_f32, sweep)?;
        let n_combos = prepared.combos.len();
        let periods: Vec<i32> = prepared
            .combos
            .iter()
            .map(|p| p.period.unwrap_or(0) as i32)
            .collect();

        // VRAM estimate: close + tables + periods + output
        let n = prepared.series_len;
        let elems = n
            .checked_mul(n_combos)
            .ok_or_else(|| CudaWillrError::InvalidInput("series_len*n_combos overflow".into()))?;
        let f32_bytes = core::mem::size_of::<f32>();
        let i32_bytes = core::mem::size_of::<i32>();
        let bytes_close = n
            .checked_mul(f32_bytes)
            .ok_or_else(|| CudaWillrError::InvalidInput("size overflow".into()))?;
        let bytes_periods = n_combos
            .checked_mul(i32_bytes)
            .ok_or_else(|| CudaWillrError::InvalidInput("size overflow".into()))?;
        let bytes_log2 = prepared.tables.log2.len()
            .checked_mul(i32_bytes)
            .ok_or_else(|| CudaWillrError::InvalidInput("size overflow".into()))?;
        let bytes_offsets = prepared.tables.level_offsets.len()
            .checked_mul(i32_bytes)
            .ok_or_else(|| CudaWillrError::InvalidInput("size overflow".into()))?;
        let bytes_st_max = prepared.tables.st_max.len()
            .checked_mul(f32_bytes)
            .ok_or_else(|| CudaWillrError::InvalidInput("size overflow".into()))?;
        let bytes_st_min = prepared.tables.st_min.len()
            .checked_mul(f32_bytes)
            .ok_or_else(|| CudaWillrError::InvalidInput("size overflow".into()))?;
        let bytes_nan_psum = prepared.tables.nan_psum.len()
            .checked_mul(i32_bytes)
            .ok_or_else(|| CudaWillrError::InvalidInput("size overflow".into()))?;
        let bytes_out = elems
            .checked_mul(f32_bytes)
            .ok_or_else(|| CudaWillrError::InvalidInput("size overflow".into()))?;
        let required = bytes_close
            .checked_add(bytes_periods)
            .and_then(|v| v.checked_add(bytes_log2))
            .and_then(|v| v.checked_add(bytes_offsets))
            .and_then(|v| v.checked_add(bytes_st_max))
            .and_then(|v| v.checked_add(bytes_st_min))
            .and_then(|v| v.checked_add(bytes_nan_psum))
            .and_then(|v| v.checked_add(bytes_out))
            .ok_or_else(|| CudaWillrError::InvalidInput("size overflow".into()))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        let d_close = DeviceBuffer::from_slice(close_f32)?;
        let d_periods = DeviceBuffer::from_slice(&periods)?;
        let d_log2 = DeviceBuffer::from_slice(&prepared.tables.log2)?;
        let d_offsets = DeviceBuffer::from_slice(&prepared.tables.level_offsets)?;
        let d_st_max = DeviceBuffer::from_slice(&prepared.tables.st_max)?;
        let d_st_min = DeviceBuffer::from_slice(&prepared.tables.st_min)?;
        let d_nan_psum = DeviceBuffer::from_slice(&prepared.tables.nan_psum)?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems)? };

        self.launch_batch_kernel(
            &d_close,
            &d_periods,
            &d_log2,
            &d_offsets,
            &d_st_max,
            &d_st_min,
            &d_nan_psum,
            prepared.series_len,
            prepared.first_valid,
            prepared.tables.level_offsets.len() - 1,
            n_combos,
            &mut d_out,
        )?;
        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: prepared.series_len,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn willr_batch_device(
        &self,
        d_close: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_log2: &DeviceBuffer<i32>,
        d_offsets: &DeviceBuffer<i32>,
        d_st_max: &DeviceBuffer<f32>,
        d_st_min: &DeviceBuffer<f32>,
        d_nan_psum: &DeviceBuffer<i32>,
        series_len: i32,
        first_valid: i32,
        n_combos: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWillrError> {
        if series_len <= 0 || n_combos <= 0 {
            return Err(CudaWillrError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }
        if first_valid < 0 || first_valid >= series_len {
            return Err(CudaWillrError::InvalidInput(format!(
                "first_valid out of range: {} (len {})",
                first_valid, series_len
            )));
        }

        let level_count = d_offsets
            .len()
            .checked_sub(1)
            .ok_or_else(|| CudaWillrError::InvalidInput("level offsets is empty".into()))?;

        self.launch_batch_kernel(
            d_close,
            d_periods,
            d_log2,
            d_offsets,
            d_st_max,
            d_st_min,
            d_nan_psum,
            series_len as usize,
            first_valid as usize,
            level_count,
            n_combos as usize,
            d_out,
        )
    }

    fn prepare_batch_inputs(
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &WillrBatchRange,
    ) -> Result<PreparedWillrBatch, CudaWillrError> {
        let len = high.len();
        if len == 0 || low.len() != len || close.len() != len {
            return Err(CudaWillrError::InvalidInput(
                "input slices are empty or mismatched".into(),
            ));
        }

        let combos = expand_periods(sweep)?;

        let first_valid = (0..len)
            .find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
            .ok_or_else(|| CudaWillrError::InvalidInput("all values are NaN".into()))?;

        let max_period = combos
            .iter()
            .map(|p| p.period.unwrap_or(0))
            .max()
            .unwrap_or(0);
        if max_period == 0 {
            return Err(CudaWillrError::InvalidInput(
                "period must be positive".into(),
            ));
        }

        let valid = len - first_valid;
        if valid < max_period {
            return Err(CudaWillrError::InvalidInput(format!(
                "not enough valid data: needed >= {}, have {}",
                max_period, valid
            )));
        }

        let tables = build_willr_gpu_tables(high, low);

        Ok(PreparedWillrBatch {
            combos,
            first_valid,
            series_len: len,
            tables,
        })
    }

    fn launch_batch_kernel(
        &self,
        d_close: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_log2: &DeviceBuffer<i32>,
        d_offsets: &DeviceBuffer<i32>,
        d_st_max: &DeviceBuffer<f32>,
        d_st_min: &DeviceBuffer<f32>,
        d_nan_psum: &DeviceBuffer<i32>,
        series_len: usize,
        first_valid: usize,
        level_count: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWillrError> {
        self.launch_batch_kernel_raw(
            d_close,
            d_periods,
            d_log2,
            d_offsets,
            d_st_max,
            d_st_min,
            d_nan_psum,
            series_len,
            first_valid,
            level_count,
            n_combos,
            d_out,
        )
    }

    /// Build sparse tables on host (from H/L) and upload them once.
    /// Returns a reusable device handle that also carries series_len, first_valid and level_count.
    pub fn prepare_tables_device(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
    ) -> Result<WillrGpuTablesDev, CudaWillrError> {
        let len = high.len();
        if len == 0 || low.len() != len || close.len() != len {
            return Err(CudaWillrError::InvalidInput(
                "input slices are empty or mismatched".into(),
            ));
        }

        // earliest index where all H/L/C are non-NaN (preserves warmup semantics)
        let first_valid = (0..len)
            .find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
            .ok_or_else(|| CudaWillrError::InvalidInput("all values are NaN".into()))?;

        // Build sparse tables (host) once for this series.
        let tables = build_willr_gpu_tables(high, low);

        // Upload once; keep device buffers around for reuse.
        let f32_bytes = core::mem::size_of::<f32>();
        let i32_bytes = core::mem::size_of::<i32>();
        let bytes_log2 = tables.log2.len()
            .checked_mul(i32_bytes)
            .ok_or_else(|| CudaWillrError::InvalidInput("size overflow".into()))?;
        let bytes_offsets = tables.level_offsets.len()
            .checked_mul(i32_bytes)
            .ok_or_else(|| CudaWillrError::InvalidInput("size overflow".into()))?;
        let bytes_st_max = tables.st_max.len()
            .checked_mul(f32_bytes)
            .ok_or_else(|| CudaWillrError::InvalidInput("size overflow".into()))?;
        let bytes_st_min = tables.st_min.len()
            .checked_mul(f32_bytes)
            .ok_or_else(|| CudaWillrError::InvalidInput("size overflow".into()))?;
        let bytes_nan_psum = tables.nan_psum.len()
            .checked_mul(i32_bytes)
            .ok_or_else(|| CudaWillrError::InvalidInput("size overflow".into()))?;
        let required = bytes_log2
            .checked_add(bytes_offsets)
            .and_then(|v| v.checked_add(bytes_st_max))
            .and_then(|v| v.checked_add(bytes_st_min))
            .and_then(|v| v.checked_add(bytes_nan_psum))
            .ok_or_else(|| CudaWillrError::InvalidInput("size overflow".into()))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        let d_log2 = DeviceBuffer::from_slice(&tables.log2)?;
        let d_level_offsets = DeviceBuffer::from_slice(&tables.level_offsets)?;
        let d_st_max = DeviceBuffer::from_slice(&tables.st_max)?;
        let d_st_min = DeviceBuffer::from_slice(&tables.st_min)?;
        let d_nan_psum = DeviceBuffer::from_slice(&tables.nan_psum)?;

        let level_count = tables
            .level_offsets
            .len()
            .checked_sub(1)
            .ok_or_else(|| CudaWillrError::InvalidInput("level offsets is empty".into()))?;

        Ok(WillrGpuTablesDev {
            d_log2,
            d_level_offsets,
            d_st_max,
            d_st_min,
            d_nan_psum,
            series_len: len,
            first_valid,
            level_count,
        })
    }

    /// Compute WILLR for many periods using previously uploaded device tables.
    /// This avoids re-uploading st_max/st_min/nan_psum/log2/offsets on every run.
    pub fn willr_batch_dev_with_tables(
        &self,
        close_f32: &[f32],
        sweep: &WillrBatchRange,
        dev_tables: &WillrGpuTablesDev,
    ) -> Result<DeviceArrayF32, CudaWillrError> {
        if close_f32.len() != dev_tables.series_len {
            return Err(CudaWillrError::InvalidInput(format!(
                "close length {} != series_len {}",
                close_f32.len(),
                dev_tables.series_len
            )));
        }

        // Expand period sweep → device vector
        let combos = expand_periods(sweep)?;
        let periods: Vec<i32> = combos
            .iter()
            .map(|p| p.period.unwrap_or(0) as i32)
            .collect();
        let n_combos = periods.len();

        // VRAM estimate: close + periods + output (tables already resident)
        let n = dev_tables.series_len;
        let elems = n
            .checked_mul(n_combos)
            .ok_or_else(|| CudaWillrError::InvalidInput("series_len*n_combos overflow".into()))?;
        let f32_bytes = core::mem::size_of::<f32>();
        let i32_bytes = core::mem::size_of::<i32>();
        let bytes_close = n
            .checked_mul(f32_bytes)
            .ok_or_else(|| CudaWillrError::InvalidInput("size overflow".into()))?;
        let bytes_periods = n_combos
            .checked_mul(i32_bytes)
            .ok_or_else(|| CudaWillrError::InvalidInput("size overflow".into()))?;
        let bytes_out = elems
            .checked_mul(f32_bytes)
            .ok_or_else(|| CudaWillrError::InvalidInput("size overflow".into()))?;
        let required = bytes_close
            .checked_add(bytes_periods)
            .and_then(|v| v.checked_add(bytes_out))
            .ok_or_else(|| CudaWillrError::InvalidInput("size overflow".into()))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        let d_close = DeviceBuffer::from_slice(close_f32)?;
        let d_periods = DeviceBuffer::from_slice(&periods)?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems)? };

        // Launch using the reusable tables.
        self.launch_batch_kernel_raw(
            &d_close,
            &d_periods,
            &dev_tables.d_log2,
            &dev_tables.d_level_offsets,
            &dev_tables.d_st_max,
            &dev_tables.d_st_min,
            &dev_tables.d_nan_psum,
            dev_tables.series_len,
            dev_tables.first_valid,
            dev_tables.level_count,
            n_combos,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: dev_tables.series_len,
        })
    }

    #[inline]
    fn block_for_time_parallel(series_len: usize) -> u32 {
        if series_len >= (1 << 20) {
            512
        } else if series_len >= (1 << 14) {
            256
        } else {
            128
        }
    }

    fn launch_batch_kernel_raw(
        &self,
        d_close: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_log2: &DeviceBuffer<i32>,
        d_offsets: &DeviceBuffer<i32>,
        d_st_max: &DeviceBuffer<f32>,
        d_st_min: &DeviceBuffer<f32>,
        d_nan_psum: &DeviceBuffer<i32>,
        series_len: usize,
        first_valid: usize,
        level_count: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWillrError> {
        if n_combos == 0 {
            return Ok(());
        }

        let func = self
            .module
            .get_function("willr_batch_f32")
            .map_err(|_| CudaWillrError::MissingKernelSymbol { name: "willr_batch_f32" })?;

        let block_x: u32 = Self::block_for_time_parallel(series_len);
        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut log2_ptr = d_log2.as_device_ptr().as_raw();
            let mut offsets_ptr = d_offsets.as_device_ptr().as_raw();
            let mut st_max_ptr = d_st_max.as_device_ptr().as_raw();
            let mut st_min_ptr = d_st_min.as_device_ptr().as_raw();
            let mut nan_psum_ptr = d_nan_psum.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut first_valid_i = first_valid as i32;
            let mut level_count_i = level_count as i32;
            let mut n_combos_i = n_combos as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut close_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut log2_ptr as *mut _ as *mut c_void,
                &mut offsets_ptr as *mut _ as *mut c_void,
                &mut st_max_ptr as *mut _ as *mut c_void,
                &mut st_min_ptr as *mut _ as *mut c_void,
                &mut nan_psum_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut level_count_i as *mut _ as *mut c_void,
                &mut n_combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    // ----- Many-series × one-param (time-major) -----

    pub fn willr_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaWillrError> {
        let (first_valids, cols, rows, period) =
            Self::prepare_many_series_inputs(high_tm, low_tm, close_tm, cols, rows, period)?;

        // VRAM estimate: 3 inputs + first_valids + output
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaWillrError::InvalidInput("cols*rows overflow".into()))?;
        let f32_bytes = core::mem::size_of::<f32>();
        let i32_bytes = core::mem::size_of::<i32>();
        let in_bytes = 3usize
            .checked_mul(elems)
            .and_then(|v| v.checked_mul(f32_bytes))
            .ok_or_else(|| CudaWillrError::InvalidInput("size overflow".into()))?;
        let first_bytes = cols
            .checked_mul(i32_bytes)
            .ok_or_else(|| CudaWillrError::InvalidInput("size overflow".into()))?;
        let out_bytes = elems
            .checked_mul(f32_bytes)
            .ok_or_else(|| CudaWillrError::InvalidInput("size overflow".into()))?;
        let required = in_bytes
            .checked_add(first_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaWillrError::InvalidInput("size overflow".into()))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        let d_high = DeviceBuffer::from_slice(high_tm)?;
        let d_low = DeviceBuffer::from_slice(low_tm)?;
        let d_close = DeviceBuffer::from_slice(close_tm)?;
        let d_first = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems)? };

        self.willr_many_series_one_param_device(
            &d_high,
            &d_low,
            &d_close,
            cols as i32,
            rows as i32,
            period as i32,
            &d_first,
            &mut d_out,
        )?;

        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn willr_many_series_one_param_device(
        &self,
        d_high_tm: &DeviceBuffer<f32>,
        d_low_tm: &DeviceBuffer<f32>,
        d_close_tm: &DeviceBuffer<f32>,
        cols: i32,
        rows: i32,
        period: i32,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWillrError> {
        if cols <= 0 || rows <= 0 || period <= 0 {
            return Err(CudaWillrError::InvalidInput(
                "cols, rows, period must be positive".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_high_tm,
            d_low_tm,
            d_close_tm,
            cols as usize,
            rows as usize,
            period as usize,
            d_first_valids,
            d_out_tm,
        )
    }

    fn prepare_many_series_inputs(
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<(Vec<i32>, usize, usize, usize), CudaWillrError> {
        if cols == 0 || rows == 0 {
            return Err(CudaWillrError::InvalidInput(
                "cols and rows must be > 0".into(),
            ));
        }
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaWillrError::InvalidInput("cols*rows overflow".into()))?;
        if high_tm.len() != elems || low_tm.len() != elems || close_tm.len() != elems {
            return Err(CudaWillrError::InvalidInput(
                "inputs must be length cols*rows (time-major)".into(),
            ));
        }
        if period == 0 {
            return Err(CudaWillrError::InvalidInput("period must be > 0".into()));
        }

        // Per-series first valid index where all three inputs are non-NaN.
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = -1i32;
            for t in 0..rows {
                let idx = t * cols + s;
                if !high_tm[idx].is_nan() && !low_tm[idx].is_nan() && !close_tm[idx].is_nan() {
                    fv = t as i32;
                    break;
                }
            }
            if fv < 0 || (rows as i32 - fv) < period as i32 {
                return Err(CudaWillrError::InvalidInput(format!(
                    "series {} lacks enough valid data (fv={}, rows={}, period={})",
                    s, fv, rows, period
                )));
            }
            first_valids[s] = fv;
        }

        Ok((first_valids, cols, rows, period))
    }

    fn launch_many_series_kernel(
        &self,
        d_high_tm: &DeviceBuffer<f32>,
        d_low_tm: &DeviceBuffer<f32>,
        d_close_tm: &DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        period: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWillrError> {
        let block_x: u32 = 256;
        let grid_x: u32 = (((cols as u32) + block_x - 1) / block_x).max(1);
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        let func = self
            .module
            .get_function("willr_many_series_one_param_time_major_f32")
            .map_err(|_| CudaWillrError::MissingKernelSymbol {
                name: "willr_many_series_one_param_time_major_f32",
            })?;

        unsafe {
            let mut high_ptr = d_high_tm.as_device_ptr().as_raw();
            let mut low_ptr = d_low_tm.as_device_ptr().as_raw();
            let mut close_ptr = d_close_tm.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut period_i = period as i32;
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }
}

// ---------- Bench profiles (batch only) ----------

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;

    fn bytes_one_series_many_params() -> usize {
        // close + precomputed tables (derived from synthetic H/L); count worst-case generous
        let in_bytes = 3 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    fn synth_hlc_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i];
            if v.is_nan() {
                continue;
            }
            let x = i as f32 * 0.0023;
            let off = (0.0029 * x.sin()).abs() + 0.1;
            high[i] = v + off;
            low[i] = v - off;
        }
        (high, low)
    }

    struct WillrBatchState {
        cuda: CudaWillr,
        high: Vec<f32>,
        low: Vec<f32>,
        close: Vec<f32>,
        sweep: WillrBatchRange,
    }
    impl CudaBenchState for WillrBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .willr_batch_dev(&self.high, &self.low, &self.close, &self.sweep)
                .expect("willr batch");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaWillr::new(0).expect("cuda willr");
        let close = gen_series(ONE_SERIES_LEN);
        let (high, low) = synth_hlc_from_close(&close);
        let sweep = WillrBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
        };
        Box::new(WillrBatchState {
            cuda,
            high,
            low,
            close,
            sweep,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "willr",
            "one_series_many_params",
            "willr_cuda_batch_dev",
            "1m_x_250",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}

fn expand_periods(range: &WillrBatchRange) -> Result<Vec<WillrParams>, CudaWillrError> {
    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, CudaWillrError> {
        // Treat zero step as static; allow reversed bounds; error on empty.
        if step == 0 {
            return Ok(vec![start]);
        }
        if start == end {
            return Ok(vec![start]);
        }
        let mut vals = Vec::new();
        if start < end {
            let mut v = start;
            while v <= end {
                vals.push(v);
                match v.checked_add(step) {
                    Some(next) => {
                        if next == v {
                            break;
                        }
                        v = next;
                    }
                    None => break,
                }
            }
        } else {
            let mut v = start;
            while v >= end {
                vals.push(v);
                if v == 0 {
                    break;
                }
                let next = v.saturating_sub(step);
                if next == v {
                    break;
                }
                v = next;
                if v < end {
                    break;
                }
            }
        }
        if vals.is_empty() {
            return Err(CudaWillrError::InvalidInput(format!(
                "invalid range: start={}, end={}, step={}",
                start, end, step
            )));
        }
        Ok(vals)
    }

    let periods = axis_usize(range.period)?;
    Ok(periods
        .into_iter()
        .map(|p| WillrParams { period: Some(p) })
        .collect())
}
