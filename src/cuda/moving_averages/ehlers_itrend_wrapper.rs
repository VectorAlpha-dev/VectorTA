//! CUDA support for the Ehlers Instantaneous Trend (ITrend) indicator.
//!
//! Provides zero-copy device handles for both batch parameter sweeps and
//! time-major many-series execution, mirroring the ALMA CUDA API surface. The
//! kernels operate purely in FP32 while this wrapper validates parameters,
//! prepares grid ranges, and stages device buffers.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::ehlers_itrend::{
    EhlersITrendBatchRange, EhlersITrendParams,
};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::DeviceBuffer;
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::convert::TryFrom;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaEhlersITrendError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaEhlersITrendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaEhlersITrendError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaEhlersITrendError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}

impl std::error::Error for CudaEhlersITrendError {}

pub struct CudaEhlersITrend {
    module: Module,
    stream: Stream,
    _context: Context,
}

struct PreparedEhlersBatch {
    combos: Vec<EhlersITrendParams>,
    warmups: Vec<i32>,
    max_dcs: Vec<i32>,
    first_valid: usize,
    series_len: usize,
    max_shared_dc: usize,
}

struct PreparedEhlersManySeries {
    first_valids: Vec<i32>,
    warmup: usize,
    max_dc: usize,
    num_series: usize,
    series_len: usize,
}

impl CudaEhlersITrend {
    pub fn new(device_id: usize) -> Result<Self, CudaEhlersITrendError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaEhlersITrendError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaEhlersITrendError::Cuda(e.to_string()))?;
        let context =
            Context::new(device).map_err(|e| CudaEhlersITrendError::Cuda(e.to_string()))?;

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/ehlers_itrend_kernel.ptx"));
        let module =
            Module::from_ptx(ptx, &[]).map_err(|e| CudaEhlersITrendError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaEhlersITrendError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    pub fn ehlers_itrend_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &EhlersITrendBatchRange,
    ) -> Result<DeviceArrayF32, CudaEhlersITrendError> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = prepared.combos.len();

        let d_prices = DeviceBuffer::from_slice(data_f32)
            .map_err(|e| CudaEhlersITrendError::Cuda(e.to_string()))?;
        let d_warmups = DeviceBuffer::from_slice(&prepared.warmups)
            .map_err(|e| CudaEhlersITrendError::Cuda(e.to_string()))?;
        let d_max_dcs = DeviceBuffer::from_slice(&prepared.max_dcs)
            .map_err(|e| CudaEhlersITrendError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(prepared.series_len * n_combos)
                .map_err(|e| CudaEhlersITrendError::Cuda(e.to_string()))?
        };

        self.launch_batch_kernel(
            &d_prices,
            &d_warmups,
            &d_max_dcs,
            prepared.series_len,
            prepared.first_valid,
            n_combos,
            prepared.max_shared_dc,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: prepared.series_len,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ehlers_itrend_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_warmups: &DeviceBuffer<i32>,
        d_max_dcs: &DeviceBuffer<i32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        max_shared_dc: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhlersITrendError> {
        if series_len == 0 {
            return Err(CudaEhlersITrendError::InvalidInput(
                "series_len must be positive".into(),
            ));
        }
        if n_combos == 0 {
            return Err(CudaEhlersITrendError::InvalidInput(
                "n_combos must be positive".into(),
            ));
        }
        if max_shared_dc == 0 {
            return Err(CudaEhlersITrendError::InvalidInput(
                "max_shared_dc must be positive".into(),
            ));
        }
        if first_valid >= series_len {
            return Err(CudaEhlersITrendError::InvalidInput(format!(
                "first_valid {} out of range for len {}",
                first_valid, series_len
            )));
        }
        if d_prices.len() != series_len {
            return Err(CudaEhlersITrendError::InvalidInput(
                "prices buffer length mismatch".into(),
            ));
        }
        if d_warmups.len() != n_combos || d_max_dcs.len() != n_combos {
            return Err(CudaEhlersITrendError::InvalidInput(
                "parameter buffer length mismatch".into(),
            ));
        }
        if d_out.len() != n_combos * series_len {
            return Err(CudaEhlersITrendError::InvalidInput(
                "output buffer length mismatch".into(),
            ));
        }

        self.launch_batch_kernel(
            d_prices,
            d_warmups,
            d_max_dcs,
            series_len,
            first_valid,
            n_combos,
            max_shared_dc,
            d_out,
        )
    }

    pub fn ehlers_itrend_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &EhlersITrendParams,
    ) -> Result<DeviceArrayF32, CudaEhlersITrendError> {
        let prepared =
            Self::prepare_many_series_inputs(data_tm_f32, num_series, series_len, params)?;

        let d_prices = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaEhlersITrendError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&prepared.first_valids)
            .map_err(|e| CudaEhlersITrendError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(num_series * series_len)
                .map_err(|e| CudaEhlersITrendError::Cuda(e.to_string()))?
        };

        self.launch_many_series_kernel(
            &d_prices,
            &d_first_valids,
            prepared.num_series,
            prepared.series_len,
            prepared.warmup,
            prepared.max_dc,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: prepared.series_len,
            cols: prepared.num_series,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ehlers_itrend_many_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        num_series: usize,
        series_len: usize,
        warmup: usize,
        max_dc: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhlersITrendError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaEhlersITrendError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if warmup == 0 {
            return Err(CudaEhlersITrendError::InvalidInput(
                "warmup must be positive".into(),
            ));
        }
        if max_dc == 0 {
            return Err(CudaEhlersITrendError::InvalidInput(
                "max_dc must be positive".into(),
            ));
        }
        if d_prices_tm.len() != num_series * series_len {
            return Err(CudaEhlersITrendError::InvalidInput(
                "time-major prices length mismatch".into(),
            ));
        }
        if d_first_valids.len() != num_series {
            return Err(CudaEhlersITrendError::InvalidInput(
                "first_valids length mismatch".into(),
            ));
        }
        if d_out_tm.len() != num_series * series_len {
            return Err(CudaEhlersITrendError::InvalidInput(
                "output buffer length mismatch".into(),
            ));
        }

        self.launch_many_series_kernel(
            d_prices_tm,
            d_first_valids,
            num_series,
            series_len,
            warmup,
            max_dc,
            d_out_tm,
        )
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_warmups: &DeviceBuffer<i32>,
        d_max_dcs: &DeviceBuffer<i32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        max_shared_dc: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhlersITrendError> {
        if n_combos == 0 {
            return Ok(());
        }

        let func = self
            .module
            .get_function("ehlers_itrend_batch_f32")
            .map_err(|e| CudaEhlersITrendError::Cuda(e.to_string()))?;

        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (128, 1, 1).into();
        let shared_bytes = (max_shared_dc * std::mem::size_of::<f32>()) as u32;

        let mut series_len_i = i32::try_from(series_len).map_err(|_| {
            CudaEhlersITrendError::InvalidInput("series_len exceeds i32::MAX".into())
        })?;
        let mut first_valid_i = i32::try_from(first_valid).map_err(|_| {
            CudaEhlersITrendError::InvalidInput("first_valid exceeds i32::MAX".into())
        })?;
        let mut combos_i = i32::try_from(n_combos)
            .map_err(|_| CudaEhlersITrendError::InvalidInput("n_combos exceeds i32::MAX".into()))?;
        let mut max_shared_dc_i = i32::try_from(max_shared_dc).map_err(|_| {
            CudaEhlersITrendError::InvalidInput("max_shared_dc exceeds i32::MAX".into())
        })?;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut warm_ptr = d_warmups.as_device_ptr().as_raw();
            let mut max_dc_ptr = d_max_dcs.as_device_ptr().as_raw();
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 8] = [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut warm_ptr as *mut _ as *mut c_void,
                &mut max_dc_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut max_shared_dc_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes, &mut args)
                .map_err(|e| CudaEhlersITrendError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaEhlersITrendError::Cuda(e.to_string()))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        num_series: usize,
        series_len: usize,
        warmup: usize,
        max_dc: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhlersITrendError> {
        if num_series == 0 {
            return Ok(());
        }

        let func = self
            .module
            .get_function("ehlers_itrend_many_series_one_param_f32")
            .map_err(|e| CudaEhlersITrendError::Cuda(e.to_string()))?;

        let grid: GridSize = (num_series as u32, 1, 1).into();
        let block: BlockSize = (128, 1, 1).into();
        let shared_bytes = (max_dc * std::mem::size_of::<f32>()) as u32;

        let mut num_series_i = i32::try_from(num_series).map_err(|_| {
            CudaEhlersITrendError::InvalidInput("num_series exceeds i32::MAX".into())
        })?;
        let mut series_len_i = i32::try_from(series_len).map_err(|_| {
            CudaEhlersITrendError::InvalidInput("series_len exceeds i32::MAX".into())
        })?;
        let mut warmup_i = i32::try_from(warmup)
            .map_err(|_| CudaEhlersITrendError::InvalidInput("warmup exceeds i32::MAX".into()))?;
        let mut max_dc_i = i32::try_from(max_dc)
            .map_err(|_| CudaEhlersITrendError::InvalidInput("max_dc exceeds i32::MAX".into()))?;

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 7] = [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut warmup_i as *mut _ as *mut c_void,
                &mut max_dc_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes, &mut args)
                .map_err(|e| CudaEhlersITrendError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaEhlersITrendError::Cuda(e.to_string()))
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &EhlersITrendBatchRange,
    ) -> Result<PreparedEhlersBatch, CudaEhlersITrendError> {
        if data_f32.is_empty() {
            return Err(CudaEhlersITrendError::InvalidInput(
                "input series may not be empty".into(),
            ));
        }

        let combos = expand_grid_cuda(sweep)?;
        if combos.is_empty() {
            return Err(CudaEhlersITrendError::InvalidInput(
                "parameter sweep produced no combinations".into(),
            ));
        }

        let series_len = data_f32.len();
        let first_valid = data_f32.iter().position(|v| v.is_finite()).ok_or_else(|| {
            CudaEhlersITrendError::InvalidInput("all input values are NaN".into())
        })?;

        let mut warmups = Vec::with_capacity(combos.len());
        let mut max_dcs = Vec::with_capacity(combos.len());
        let mut max_shared_dc = 0usize;

        for params in &combos {
            let warmup = params.warmup_bars.unwrap_or(12);
            let max_dc = params.max_dc_period.unwrap_or(50);
            if warmup == 0 {
                return Err(CudaEhlersITrendError::InvalidInput(
                    "warmup_bars must be positive".into(),
                ));
            }
            if max_dc == 0 {
                return Err(CudaEhlersITrendError::InvalidInput(
                    "max_dc_period must be positive".into(),
                ));
            }
            if warmup > i32::MAX as usize {
                return Err(CudaEhlersITrendError::InvalidInput(
                    "warmup_bars exceeds i32::MAX".into(),
                ));
            }
            if max_dc > i32::MAX as usize {
                return Err(CudaEhlersITrendError::InvalidInput(
                    "max_dc_period exceeds i32::MAX".into(),
                ));
            }
            if series_len - first_valid < warmup {
                return Err(CudaEhlersITrendError::InvalidInput(format!(
                    "not enough valid samples after first_valid={} for warmup {}",
                    first_valid, warmup
                )));
            }
            warmups.push(warmup as i32);
            max_dcs.push(max_dc as i32);
            if max_dc > max_shared_dc {
                max_shared_dc = max_dc;
            }
        }

        Ok(PreparedEhlersBatch {
            combos,
            warmups,
            max_dcs,
            first_valid,
            series_len,
            max_shared_dc,
        })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &EhlersITrendParams,
    ) -> Result<PreparedEhlersManySeries, CudaEhlersITrendError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaEhlersITrendError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if data_tm_f32.len() != num_series * series_len {
            return Err(CudaEhlersITrendError::InvalidInput(
                "time-major slice length mismatch".into(),
            ));
        }

        let warmup = params.warmup_bars.unwrap_or(12);
        let max_dc = params.max_dc_period.unwrap_or(50);
        if warmup == 0 {
            return Err(CudaEhlersITrendError::InvalidInput(
                "warmup_bars must be positive".into(),
            ));
        }
        if max_dc == 0 {
            return Err(CudaEhlersITrendError::InvalidInput(
                "max_dc_period must be positive".into(),
            ));
        }
        if warmup > i32::MAX as usize {
            return Err(CudaEhlersITrendError::InvalidInput(
                "warmup_bars exceeds i32::MAX".into(),
            ));
        }
        if max_dc > i32::MAX as usize {
            return Err(CudaEhlersITrendError::InvalidInput(
                "max_dc_period exceeds i32::MAX".into(),
            ));
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
                CudaEhlersITrendError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            let remaining = series_len - fv as usize;
            if remaining < warmup {
                return Err(CudaEhlersITrendError::InvalidInput(format!(
                    "series {} lacks warmup samples: need {}, have {} after first_valid",
                    series, warmup, remaining
                )));
            }
            first_valids.push(fv);
        }

        Ok(PreparedEhlersManySeries {
            first_valids,
            warmup,
            max_dc,
            num_series,
            series_len,
        })
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        ehlers_itrend_benches,
        CudaEhlersITrend,
        crate::indicators::moving_averages::ehlers_itrend::EhlersITrendBatchRange,
        crate::indicators::moving_averages::ehlers_itrend::EhlersITrendParams,
        ehlers_itrend_batch_dev,
        ehlers_itrend_many_series_one_param_time_major_dev,
        crate::indicators::moving_averages::ehlers_itrend::EhlersITrendBatchRange { warmup_bars: (12, 12 + PARAM_SWEEP - 1, 1), max_dc_period: (50, 50, 0) },
        crate::indicators::moving_averages::ehlers_itrend::EhlersITrendParams { warmup_bars: Some(32), max_dc_period: Some(50) },
        "ehlers_itrend",
        "ehlers_itrend"
    );
    pub use ehlers_itrend_benches::bench_profiles;
}

fn expand_grid_cuda(
    range: &EhlersITrendBatchRange,
) -> Result<Vec<EhlersITrendParams>, CudaEhlersITrendError> {
    fn axis(tuple: (usize, usize, usize)) -> Option<Vec<usize>> {
        let (start, end, step) = tuple;
        if step == 0 {
            if start == end {
                Some(vec![start])
            } else {
                None
            }
        } else if start > end {
            None
        } else {
            Some((start..=end).step_by(step).collect())
        }
    }

    let warmups = axis(range.warmup_bars)
        .ok_or_else(|| CudaEhlersITrendError::InvalidInput("invalid warmup range".into()))?;
    let max_dcs = axis(range.max_dc_period)
        .ok_or_else(|| CudaEhlersITrendError::InvalidInput("invalid max_dc range".into()))?;

    let mut combos = Vec::with_capacity(warmups.len() * max_dcs.len());
    for &w in &warmups {
        for &m in &max_dcs {
            combos.push(EhlersITrendParams {
                warmup_bars: Some(w),
                max_dc_period: Some(m),
            });
        }
    }
    Ok(combos)
}
