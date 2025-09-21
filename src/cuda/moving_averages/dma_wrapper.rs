//! CUDA scaffolding for the DMA (Dickson Moving Average) kernels.
//!
//! The implementation mirrors the scalar Rust path but executes batches of
//! parameter combinations on the GPU. Each parameter set is processed within a
//! dedicated block using sequential evaluation to keep the adaptive EMA search
//! exact while still benefiting from device-resident data.

#![cfg(feature = "cuda")]

use super::DeviceArrayF32;
use crate::indicators::moving_averages::dma::{DmaBatchRange, DmaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::DeviceBuffer;
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::sys as cu;
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaDmaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaDmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaDmaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaDmaError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}

impl std::error::Error for CudaDmaError {}

pub struct CudaDma {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaDma {
    pub fn new(device_id: usize) -> Result<Self, CudaDmaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaDmaError::Cuda(e.to_string()))?;

        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaDmaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaDmaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/dma_kernel.ptx"));
        let module = Module::from_ptx(ptx, &[]).map_err(|e| CudaDmaError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaDmaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
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
        unsafe {
            let mut free: usize = 0;
            let mut total: usize = 0;
            let res = cu::cuMemGetInfo_v2(&mut free as *mut usize, &mut total as *mut usize);
            if res == cu::CUresult::CUDA_SUCCESS {
                Some((free, total))
            } else {
                None
            }
        }
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

    pub fn dma_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &DmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaDmaError> {
        let inputs = Self::prepare_batch_inputs(data_f32, sweep)?;
        self.run_batch_kernel(data_f32, &inputs)
    }

    pub fn dma_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &DmaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<DmaParams>), CudaDmaError> {
        let inputs = Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = inputs.series_len * inputs.hull_lengths.len();
        if out.len() != expected {
            return Err(CudaDmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                expected
            )));
        }
        let arr = self.run_batch_kernel(data_f32, &inputs)?;
        arr.buf
            .copy_to(out)
            .map_err(|e| CudaDmaError::Cuda(e.to_string()))?;
        Ok((arr.rows, arr.cols, inputs.combos))
    }

    pub fn dma_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_hulls: &DeviceBuffer<i32>,
        d_emas: &DeviceBuffer<i32>,
        d_gain_limits: &DeviceBuffer<i32>,
        d_types: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        max_sqrt_len: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaDmaError> {
        if series_len == 0 || n_combos == 0 {
            return Err(CudaDmaError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }
        if series_len > i32::MAX as usize {
            return Err(CudaDmaError::InvalidInput(
                "series too long for kernel argument width".into(),
            ));
        }
        self.launch_batch_kernel(
            d_prices,
            d_hulls,
            d_emas,
            d_gain_limits,
            d_types,
            series_len,
            n_combos,
            first_valid,
            max_sqrt_len,
            d_out,
        )
    }

    pub fn dma_many_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        hull_length: i32,
        ema_length: i32,
        ema_gain_limit: i32,
        hull_type: i32,
        series_len: usize,
        num_series: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaDmaError> {
        if hull_length <= 0 || ema_length <= 0 {
            return Err(CudaDmaError::InvalidInput(
                "hull_length and ema_length must be positive".into(),
            ));
        }
        if series_len == 0 || num_series == 0 {
            return Err(CudaDmaError::InvalidInput(
                "series_len and num_series must be positive".into(),
            ));
        }
        if ema_gain_limit < 0 {
            return Err(CudaDmaError::InvalidInput(
                "ema_gain_limit must be non-negative".into(),
            ));
        }
        if hull_type != 0 && hull_type != 1 {
            return Err(CudaDmaError::InvalidInput(
                "hull_type must be 0 (WMA) or 1 (EMA)".into(),
            ));
        }
        let sqrt_len = ((hull_length as f64).sqrt().round() as usize).max(1);
        self.launch_many_series_kernel(
            d_prices_tm,
            hull_length as usize,
            ema_length as usize,
            ema_gain_limit as usize,
            hull_type,
            series_len,
            num_series,
            d_first_valids,
            sqrt_len,
            d_out_tm,
        )
    }

    pub fn dma_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &DmaParams,
    ) -> Result<DeviceArrayF32, CudaDmaError> {
        let (first_valids, hull_length, ema_length, ema_gain_limit, hull_type, sqrt_len) =
            Self::prepare_many_series_inputs(data_tm_f32, num_series, series_len, params)?;
        self.run_many_series_kernel(
            data_tm_f32,
            num_series,
            series_len,
            &first_valids,
            hull_length,
            ema_length,
            ema_gain_limit,
            hull_type,
            sqrt_len,
        )
    }

    pub fn dma_many_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &DmaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaDmaError> {
        if out_tm.len() != data_tm_f32.len() {
            return Err(CudaDmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out_tm.len(),
                data_tm_f32.len()
            )));
        }
        let (first_valids, hull_length, ema_length, ema_gain_limit, hull_type, sqrt_len) =
            Self::prepare_many_series_inputs(data_tm_f32, num_series, series_len, params)?;
        let arr = self.run_many_series_kernel(
            data_tm_f32,
            num_series,
            series_len,
            &first_valids,
            hull_length,
            ema_length,
            ema_gain_limit,
            hull_type,
            sqrt_len,
        )?;
        arr.buf
            .copy_to(out_tm)
            .map_err(|e| CudaDmaError::Cuda(e.to_string()))
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        inputs: &BatchInputs,
    ) -> Result<DeviceArrayF32, CudaDmaError> {
        let n_combos = inputs.hull_lengths.len();
        let series_len = inputs.series_len;
        let first_valid = inputs.first_valid;
        let max_sqrt_len = inputs.max_sqrt_len;

        let prices_bytes = series_len * std::mem::size_of::<f32>();
        let hull_bytes = n_combos * std::mem::size_of::<i32>();
        let ema_bytes = hull_bytes;
        let gain_bytes = hull_bytes;
        let type_bytes = hull_bytes;
        let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + hull_bytes + ema_bytes + gain_bytes + type_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaDmaError::InvalidInput(
                "not enough device memory for DMA batch".into(),
            ));
        }

        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaDmaError::Cuda(e.to_string()))?;
        let d_hulls = DeviceBuffer::from_slice(&inputs.hull_lengths)
            .map_err(|e| CudaDmaError::Cuda(e.to_string()))?;
        let d_emas = DeviceBuffer::from_slice(&inputs.ema_lengths)
            .map_err(|e| CudaDmaError::Cuda(e.to_string()))?;
        let d_gains = DeviceBuffer::from_slice(&inputs.ema_gain_limits)
            .map_err(|e| CudaDmaError::Cuda(e.to_string()))?;
        let d_types = DeviceBuffer::from_slice(&inputs.hull_types)
            .map_err(|e| CudaDmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaDmaError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_prices,
            &d_hulls,
            &d_emas,
            &d_gains,
            &d_types,
            series_len,
            n_combos,
            first_valid,
            max_sqrt_len,
            &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaDmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    fn run_many_series_kernel(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        first_valids: &[i32],
        hull_length: usize,
        ema_length: usize,
        ema_gain_limit: usize,
        hull_type: i32,
        sqrt_len: usize,
    ) -> Result<DeviceArrayF32, CudaDmaError> {
        let prices_bytes = data_tm_f32.len() * std::mem::size_of::<f32>();
        let first_valid_bytes = first_valids.len() * std::mem::size_of::<i32>();
        let out_bytes = data_tm_f32.len() * std::mem::size_of::<f32>();
        let required = prices_bytes + first_valid_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaDmaError::InvalidInput(
                "not enough device memory for DMA many-series".into(),
            ));
        }

        let d_prices =
            DeviceBuffer::from_slice(data_tm_f32).map_err(|e| CudaDmaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(first_valids)
            .map_err(|e| CudaDmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(data_tm_f32.len()) }
                .map_err(|e| CudaDmaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices,
            hull_length,
            ema_length,
            ema_gain_limit,
            hull_type,
            series_len,
            num_series,
            &d_first_valids,
            sqrt_len,
            &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaDmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: series_len,
            cols: num_series,
        })
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_hulls: &DeviceBuffer<i32>,
        d_emas: &DeviceBuffer<i32>,
        d_gains: &DeviceBuffer<i32>,
        d_types: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        max_sqrt_len: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaDmaError> {
        let func = self
            .module
            .get_function("dma_batch_f32")
            .map_err(|e| CudaDmaError::Cuda(e.to_string()))?;

        let block: BlockSize = (32, 1, 1).into();
        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let shared_bytes = (max_sqrt_len * std::mem::size_of::<f32>()) as u32;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut hull_ptr = d_hulls.as_device_ptr().as_raw();
            let mut ema_ptr = d_emas.as_device_ptr().as_raw();
            let mut gain_ptr = d_gains.as_device_ptr().as_raw();
            let mut type_ptr = d_types.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut n_combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut hull_ptr as *mut _ as *mut c_void,
                &mut ema_ptr as *mut _ as *mut c_void,
                &mut gain_ptr as *mut _ as *mut c_void,
                &mut type_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut n_combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes, args)
                .map_err(|e| CudaDmaError::Cuda(e.to_string()))?
        }
        Ok(())
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        hull_length: usize,
        ema_length: usize,
        ema_gain_limit: usize,
        hull_type: i32,
        series_len: usize,
        num_series: usize,
        d_first_valids: &DeviceBuffer<i32>,
        sqrt_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaDmaError> {
        let func = self
            .module
            .get_function("dma_many_series_one_param_f32")
            .map_err(|e| CudaDmaError::Cuda(e.to_string()))?;

        let block: BlockSize = (32, 1, 1).into();
        let grid: GridSize = (num_series as u32, 1, 1).into();
        let shared_bytes = (sqrt_len * std::mem::size_of::<f32>()) as u32;

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut hull_len_i = hull_length as i32;
            let mut ema_len_i = ema_length as i32;
            let mut gain_i = ema_gain_limit as i32;
            let mut hull_type_i = hull_type;
            let mut series_len_i = series_len as i32;
            let mut num_series_i = num_series as i32;
            let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut sqrt_len_i = sqrt_len as i32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut hull_len_i as *mut _ as *mut c_void,
                &mut ema_len_i as *mut _ as *mut c_void,
                &mut gain_i as *mut _ as *mut c_void,
                &mut hull_type_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut first_valids_ptr as *mut _ as *mut c_void,
                &mut sqrt_len_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes, args)
                .map_err(|e| CudaDmaError::Cuda(e.to_string()))?
        }
        Ok(())
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &DmaBatchRange,
    ) -> Result<BatchInputs, CudaDmaError> {
        if data_f32.is_empty() {
            return Err(CudaDmaError::InvalidInput("empty data".into()));
        }

        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaDmaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let series_len = data_f32.len();
        if series_len > i32::MAX as usize {
            return Err(CudaDmaError::InvalidInput(
                "series too long for kernel argument width".into(),
            ));
        }

        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaDmaError::InvalidInput("all values are NaN".into()))?;
        let valid = series_len - first_valid;

        let mut hull_lengths = Vec::with_capacity(combos.len());
        let mut ema_lengths = Vec::with_capacity(combos.len());
        let mut ema_gain_limits = Vec::with_capacity(combos.len());
        let mut hull_types = Vec::with_capacity(combos.len());
        let mut max_sqrt_len = 0usize;

        for prm in &combos {
            let hull_len = prm.hull_length.unwrap_or(0);
            let ema_len = prm.ema_length.unwrap_or(0);
            let gain_limit = prm.ema_gain_limit.unwrap_or(0);
            let hull_ma_type = prm
                .hull_ma_type
                .as_deref()
                .unwrap_or("WMA")
                .to_ascii_uppercase();

            if hull_len == 0 || hull_len > series_len {
                return Err(CudaDmaError::InvalidInput(format!(
                    "invalid hull length {} for data len {}",
                    hull_len, series_len
                )));
            }
            if ema_len == 0 || ema_len > series_len {
                return Err(CudaDmaError::InvalidInput(format!(
                    "invalid ema length {} for data len {}",
                    ema_len, series_len
                )));
            }
            let sqrt_len = ((hull_len as f64).sqrt().round()) as usize;
            let needed = hull_len.max(ema_len) + sqrt_len;
            if valid < needed {
                return Err(CudaDmaError::InvalidInput(format!(
                    "not enough valid data (needed >= {}, valid = {})",
                    needed, valid
                )));
            }

            let hull_tag = match hull_ma_type.as_str() {
                "WMA" => 0,
                "EMA" => 1,
                other => {
                    return Err(CudaDmaError::InvalidInput(format!(
                        "unsupported hull_ma_type {}",
                        other
                    )))
                }
            };

            if hull_len > i32::MAX as usize || ema_len > i32::MAX as usize {
                return Err(CudaDmaError::InvalidInput(
                    "parameter length exceeds kernel limits".into(),
                ));
            }
            if gain_limit > i32::MAX as usize {
                return Err(CudaDmaError::InvalidInput(
                    "ema_gain_limit exceeds kernel limits".into(),
                ));
            }

            hull_lengths.push(hull_len as i32);
            ema_lengths.push(ema_len as i32);
            ema_gain_limits.push(gain_limit as i32);
            hull_types.push(hull_tag);
            max_sqrt_len = max_sqrt_len.max(sqrt_len);
        }

        if max_sqrt_len == 0 {
            max_sqrt_len = 1;
        }

        Ok(BatchInputs {
            combos,
            hull_lengths,
            ema_lengths,
            ema_gain_limits,
            hull_types,
            first_valid,
            series_len,
            max_sqrt_len,
        })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &DmaParams,
    ) -> Result<(Vec<i32>, usize, usize, usize, i32, usize), CudaDmaError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaDmaError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if data_tm_f32.len() != num_series * series_len {
            return Err(CudaDmaError::InvalidInput(format!(
                "data length {} != num_series * series_len {}",
                data_tm_f32.len(),
                num_series * series_len
            )));
        }

        let hull_length = params.hull_length.unwrap_or(7);
        let ema_length = params.ema_length.unwrap_or(20);
        let ema_gain_limit = params.ema_gain_limit.unwrap_or(50);
        if hull_length == 0 || ema_length == 0 {
            return Err(CudaDmaError::InvalidInput(
                "hull_length and ema_length must be positive".into(),
            ));
        }
        let hull_ma_type = params
            .hull_ma_type
            .as_deref()
            .unwrap_or("WMA")
            .to_ascii_uppercase();
        let hull_type_tag = match hull_ma_type.as_str() {
            "WMA" => 0,
            "EMA" => 1,
            other => {
                return Err(CudaDmaError::InvalidInput(format!(
                    "unsupported hull_ma_type {}",
                    other
                )))
            }
        };

        if hull_length > i32::MAX as usize
            || ema_length > i32::MAX as usize
            || ema_gain_limit > i32::MAX as usize
        {
            return Err(CudaDmaError::InvalidInput(
                "parameter exceeds kernel argument width".into(),
            ));
        }

        let sqrt_len = ((hull_length as f64).sqrt().round() as usize).max(1);
        let needed = hull_length.max(ema_length) + sqrt_len;

        let mut first_valids = vec![0i32; num_series];
        for series in 0..num_series {
            let mut fv = None;
            for t in 0..series_len {
                let v = data_tm_f32[t * num_series + series];
                if !v.is_nan() {
                    fv = Some(t);
                    break;
                }
            }
            let first = fv.ok_or_else(|| {
                CudaDmaError::InvalidInput(format!("series {} all values are NaN", series))
            })?;
            if series_len - first < needed {
                return Err(CudaDmaError::InvalidInput(format!(
                    "series {} not enough valid data (needed >= {}, valid = {})",
                    series,
                    needed,
                    series_len - first
                )));
            }
            first_valids[series] = first as i32;
        }

        Ok((
            first_valids,
            hull_length,
            ema_length,
            ema_gain_limit,
            hull_type_tag,
            sqrt_len,
        ))
    }
}

fn axis_values((start, end, step): (usize, usize, usize)) -> Vec<usize> {
    if step == 0 || start == end {
        return vec![start];
    }
    (start..=end).step_by(step).collect()
}

fn expand_grid(range: &DmaBatchRange) -> Vec<DmaParams> {
    let hull_lengths = axis_values(range.hull_length);
    let ema_lengths = axis_values(range.ema_length);
    let ema_gain_limits = axis_values(range.ema_gain_limit);

    let mut combos = Vec::new();
    for &h in &hull_lengths {
        for &e in &ema_lengths {
            for &g in &ema_gain_limits {
                combos.push(DmaParams {
                    hull_length: Some(h),
                    ema_length: Some(e),
                    ema_gain_limit: Some(g),
                    hull_ma_type: Some(range.hull_ma_type.clone()),
                });
            }
        }
    }
    combos
}

struct BatchInputs {
    combos: Vec<DmaParams>,
    hull_lengths: Vec<i32>,
    ema_lengths: Vec<i32>,
    ema_gain_limits: Vec<i32>,
    hull_types: Vec<i32>,
    first_valid: usize,
    series_len: usize,
    max_sqrt_len: usize,
}
