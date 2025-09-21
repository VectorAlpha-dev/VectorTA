//! CUDA support for the Slope Adaptive Moving Average (SAMA).
//!
//! Mirrors the ALMA CUDA API surface by exposing zero-copy device entry
//! points for both the one-series × many-parameter sweep and the time-major
//! many-series × one-parameter scenario. Kernels operate fully in FP32 and
//! reuse host-prepared alpha coefficients to keep GPU-side work focused on the
//! adaptive recurrence itself.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::sama::{SamaBatchRange, SamaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::DeviceBuffer;
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;

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

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/sama_kernel.ptx"));
        let module = Module::from_ptx(ptx, &[]).map_err(|e| CudaSamaError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaSamaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    pub fn sama_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &SamaBatchRange,
    ) -> Result<DeviceArrayF32, CudaSamaError> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = prepared.combos.len();

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

        self.launch_batch_kernel(
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

        self.launch_batch_kernel(
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

    fn launch_batch_kernel(
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
        let block: BlockSize = (256, 1, 1).into();
        let grid: GridSize = (n_combos as u32, 1, 1).into();

        let func = self
            .module
            .get_function("sama_batch_f32")
            .map_err(|e| CudaSamaError::Cuda(e.to_string()))?;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut lengths_ptr = d_lengths.as_device_ptr().as_raw();
            let mut min_ptr = d_min_alphas.as_device_ptr().as_raw();
            let mut maj_ptr = d_maj_alphas.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut combos_i = n_combos as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
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
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(|e| CudaSamaError::Cuda(e.to_string()))?;
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
        let block: BlockSize = (256, 1, 1).into();
        let grid: GridSize = (num_series as u32, 1, 1).into();

        let func = self
            .module
            .get_function("sama_many_series_one_param_f32")
            .map_err(|e| CudaSamaError::Cuda(e.to_string()))?;

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
