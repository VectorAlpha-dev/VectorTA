#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::edcf::{EdcfBatchRange, EdcfParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{CopyDestination, DeviceBuffer};
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaEdcfError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaEdcfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaEdcfError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaEdcfError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}

impl std::error::Error for CudaEdcfError {}

pub struct CudaEdcf {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaEdcf {
    pub fn new(device_id: usize) -> Result<Self, CudaEdcfError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;

        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/edcf_kernel.ptx"));
        let module = Module::from_ptx(ptx, &[]).map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    pub fn edcf_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &EdcfBatchRange,
    ) -> Result<DeviceArrayF32, CudaEdcfError> {
        let (combos, first_valid, series_len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = combos.len();
        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;
        let mut d_dist: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(series_len) }
            .map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;

        self.edcf_batch_device_impl(
            &d_prices,
            &combos,
            first_valid,
            series_len,
            &mut d_dist,
            &mut d_out,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    pub fn edcf_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &EdcfBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<EdcfParams>), CudaEdcfError> {
        let (combos, first_valid, series_len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos.len() * series_len;
        if out.len() != expected {
            return Err(CudaEdcfError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                expected
            )));
        }
        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(expected) }
            .map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;
        let mut d_dist: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(series_len) }
            .map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;

        self.edcf_batch_device_impl(
            &d_prices,
            &combos,
            first_valid,
            series_len,
            &mut d_dist,
            &mut d_out,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;

        d_out
            .copy_to(out)
            .map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;

        Ok((combos.len(), series_len, combos))
    }

    pub fn edcf_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        combos: &[EdcfParams],
        first_valid: usize,
        series_len: usize,
        d_dist: &mut DeviceBuffer<f32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEdcfError> {
        self.edcf_batch_device_impl(d_prices, combos, first_valid, series_len, d_dist, d_out)?;
        self.stream
            .synchronize()
            .map_err(|e| CudaEdcfError::Cuda(e.to_string()))
    }

    fn edcf_batch_device_impl(
        &self,
        d_prices: &DeviceBuffer<f32>,
        combos: &[EdcfParams],
        first_valid: usize,
        series_len: usize,
        d_dist: &mut DeviceBuffer<f32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEdcfError> {
        if combos.is_empty() {
            return Err(CudaEdcfError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        if d_dist.len() < series_len {
            return Err(CudaEdcfError::InvalidInput(format!(
                "dist buffer too small: got {}, need {}",
                d_dist.len(),
                series_len
            )));
        }
        if d_out.len() != combos.len() * series_len {
            return Err(CudaEdcfError::InvalidInput(format!(
                "output buffer wrong length: got {}, expected {}",
                d_out.len(),
                combos.len() * series_len
            )));
        }
        if series_len == 0 {
            return Err(CudaEdcfError::InvalidInput("series_len is zero".into()));
        }
        if series_len > i32::MAX as usize {
            return Err(CudaEdcfError::InvalidInput(
                "series_len exceeds i32::MAX (unsupported)".into(),
            ));
        }

        let compute_fn = self
            .module
            .get_function("edcf_compute_dist_f32")
            .map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;
        let apply_fn = self
            .module
            .get_function("edcf_apply_weights_f32")
            .map_err(|e| CudaEdcfError::Cuda(e.to_string()))?;

        for (row_idx, params) in combos.iter().enumerate() {
            let period = params.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaEdcfError::InvalidInput(format!(
                    "invalid period at combo {}: {}",
                    row_idx, period
                )));
            }
            if period > series_len {
                return Err(CudaEdcfError::InvalidInput(format!(
                    "period {} exceeds series length {}",
                    period, series_len
                )));
            }
            let needed = 2 * period;
            if series_len - first_valid < needed {
                return Err(CudaEdcfError::InvalidInput(format!(
                    "not enough valid data (needed >= {}, valid = {})",
                    needed,
                    series_len - first_valid
                )));
            }

            self.launch_compute_dist(
                &compute_fn,
                d_prices,
                series_len,
                period,
                first_valid,
                d_dist,
            )?;

            let offset_elems = row_idx * series_len;
            let row_ptr =
                d_out.as_device_ptr().as_raw() + (offset_elems * std::mem::size_of::<f32>()) as u64;
            self.launch_apply_weights(
                &apply_fn,
                d_prices,
                d_dist,
                series_len,
                period,
                first_valid,
                row_ptr,
            )?;
        }

        Ok(())
    }

    fn launch_compute_dist(
        &self,
        func: &cust::function::Function,
        d_prices: &DeviceBuffer<f32>,
        len: usize,
        period: usize,
        first_valid: usize,
        d_dist: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEdcfError> {
        let block_x: u32 = 256;
        let mut grid_x = ((len as u32) + block_x - 1) / block_x;
        if grid_x == 0 {
            grid_x = 1;
        }
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut period_i = period as i32;
            let mut first_i = first_valid as i32;
            let mut dist_ptr = d_dist.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut dist_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(func, grid, block, 0, args)
                .map_err(|e| CudaEdcfError::Cuda(e.to_string()))?
        }
        Ok(())
    }

    fn launch_apply_weights(
        &self,
        func: &cust::function::Function,
        d_prices: &DeviceBuffer<f32>,
        d_dist: &DeviceBuffer<f32>,
        len: usize,
        period: usize,
        first_valid: usize,
        out_row_ptr: u64,
    ) -> Result<(), CudaEdcfError> {
        let block_x: u32 = 256;
        let mut grid_x = ((len as u32) + block_x - 1) / block_x;
        if grid_x == 0 {
            grid_x = 1;
        }
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut dist_ptr = d_dist.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut period_i = period as i32;
            let mut first_i = first_valid as i32;
            let mut out_ptr = out_row_ptr;
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut dist_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(func, grid, block, 0, args)
                .map_err(|e| CudaEdcfError::Cuda(e.to_string()))?
        }
        Ok(())
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &EdcfBatchRange,
    ) -> Result<(Vec<EdcfParams>, usize, usize), CudaEdcfError> {
        if data_f32.is_empty() {
            return Err(CudaEdcfError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaEdcfError::InvalidInput("all values are NaN".into()))?;
        let series_len = data_f32.len();

        let combos = Self::expand_range(sweep);
        if combos.is_empty() {
            return Err(CudaEdcfError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        for (idx, prm) in combos.iter().enumerate() {
            let period = prm.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaEdcfError::InvalidInput(format!(
                    "invalid period at combo {}: {}",
                    idx, period
                )));
            }
            if period > series_len {
                return Err(CudaEdcfError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, series_len
                )));
            }
            let needed = 2 * period;
            if series_len - first_valid < needed {
                return Err(CudaEdcfError::InvalidInput(format!(
                    "not enough valid data (needed >= {}, valid = {})",
                    needed,
                    series_len - first_valid
                )));
            }
        }

        Ok((combos, first_valid, series_len))
    }

    fn expand_range(sweep: &EdcfBatchRange) -> Vec<EdcfParams> {
        let (start, end, step) = sweep.period;
        if step == 0 || start == end {
            return vec![EdcfParams {
                period: Some(start),
            }];
        }
        let mut periods = Vec::new();
        let mut value = start;
        while value <= end {
            periods.push(EdcfParams {
                period: Some(value),
            });
            value = match value.checked_add(step) {
                Some(v) => v,
                None => break,
            };
        }
        periods
    }
}
