#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::vwap::{
    expand_grid_vwap, first_valid_vwap_index, parse_anchor, VwapBatchRange, VwapParams,
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
pub enum CudaVwapError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaVwapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaVwapError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaVwapError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaVwapError {}

pub struct CudaVwap {
    module: Module,
    stream: Stream,
    _context: Context,
}

struct PreparedBatch {
    combos: Vec<VwapParams>,
    counts: Vec<i32>,
    unit_codes: Vec<i32>,
    divisors: Vec<i64>,
    first_valids: Vec<i32>,
    month_ids: Option<Vec<i32>>,
    series_len: usize,
}

impl CudaVwap {
    pub fn new(device_id: usize) -> Result<Self, CudaVwapError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaVwapError::Cuda(e.to_string()))?;

        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaVwapError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaVwapError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/vwap_kernel.ptx"));
        let module = Module::from_ptx(ptx, &[]).map_err(|e| CudaVwapError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaVwapError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    fn prepare_batch_inputs(
        timestamps: &[i64],
        volumes: &[f64],
        prices: &[f64],
        sweep: &VwapBatchRange,
    ) -> Result<PreparedBatch, CudaVwapError> {
        if timestamps.len() != volumes.len() || volumes.len() != prices.len() {
            return Err(CudaVwapError::InvalidInput(
                "timestamps, volumes, and prices must have equal length".into(),
            ));
        }
        if timestamps.is_empty() {
            return Err(CudaVwapError::InvalidInput("empty input series".into()));
        }

        let combos = expand_grid_vwap(sweep);
        if combos.is_empty() {
            return Err(CudaVwapError::InvalidInput(
                "no parameter combinations after anchor expansion".into(),
            ));
        }

        let mut counts = Vec::with_capacity(combos.len());
        let mut unit_codes = Vec::with_capacity(combos.len());
        let mut divisors = Vec::with_capacity(combos.len());
        let mut first_valids = Vec::with_capacity(combos.len());
        let mut needs_months = false;

        for params in &combos {
            let anchor = params.anchor.as_deref().unwrap_or("1d");
            let (count_u32, unit_char) =
                parse_anchor(anchor).map_err(|e| CudaVwapError::InvalidInput(e.to_string()))?;
            if count_u32 == 0 {
                return Err(CudaVwapError::InvalidInput(format!(
                    "anchor '{}' resolved to zero count",
                    anchor
                )));
            }
            let count = i32::try_from(count_u32)
                .map_err(|_| CudaVwapError::InvalidInput("count exceeds i32::MAX".into()))?;

            let (unit_code, divisor) = match unit_char {
                'm' => (0, (count as i64).saturating_mul(60_000)),
                'h' => (1, (count as i64).saturating_mul(3_600_000)),
                'd' => (2, (count as i64).saturating_mul(86_400_000)),
                'M' => {
                    needs_months = true;
                    (3, count as i64)
                }
                other => {
                    return Err(CudaVwapError::InvalidInput(format!(
                        "unsupported anchor unit '{}'",
                        other
                    )))
                }
            };

            if divisor <= 0 {
                return Err(CudaVwapError::InvalidInput(format!(
                    "non-positive divisor derived from anchor '{}'",
                    anchor
                )));
            }

            let warm = first_valid_vwap_index(timestamps, volumes, count_u32, unit_char);
            let warm_i32 = i32::try_from(warm).unwrap_or(i32::MAX);

            counts.push(count);
            unit_codes.push(unit_code);
            divisors.push(divisor);
            first_valids.push(warm_i32);
        }

        let month_ids = if needs_months {
            Some(Self::compute_month_ids(timestamps)?)
        } else {
            None
        };

        Ok(PreparedBatch {
            combos,
            counts,
            unit_codes,
            divisors,
            first_valids,
            month_ids,
            series_len: prices.len(),
        })
    }

    fn compute_month_ids(timestamps: &[i64]) -> Result<Vec<i32>, CudaVwapError> {
        use crate::indicators::moving_averages::vwap::floor_to_month;

        let mut out = Vec::with_capacity(timestamps.len());
        for &ts in timestamps {
            let month = match floor_to_month(ts, 1) {
                Ok(v) => v,
                Err(_) => i64::MIN,
            };
            let clamped = month.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
            out.push(clamped);
        }
        Ok(out)
    }

    fn launch_kernel(
        &self,
        d_timestamps: &DeviceBuffer<i64>,
        d_volumes: &DeviceBuffer<f32>,
        d_prices: &DeviceBuffer<f32>,
        d_counts: &DeviceBuffer<i32>,
        d_unit_codes: &DeviceBuffer<i32>,
        d_divisors: &DeviceBuffer<i64>,
        d_first_valids: &DeviceBuffer<i32>,
        month_ids_ptr: u64,
        d_out: &mut DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
    ) -> Result<(), CudaVwapError> {
        if series_len > i32::MAX as usize {
            return Err(CudaVwapError::InvalidInput(
                "series length exceeds i32::MAX (unsupported by kernel)".into(),
            ));
        }
        if n_combos > i32::MAX as usize {
            return Err(CudaVwapError::InvalidInput(
                "number of parameter combos exceeds i32::MAX".into(),
            ));
        }

        let func = self
            .module
            .get_function("vwap_batch_f32")
            .map_err(|e| CudaVwapError::Cuda(e.to_string()))?;

        let block_x = 128u32;
        let grid_x = ((n_combos as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut ts_ptr = d_timestamps.as_device_ptr().as_raw();
            let mut vol_ptr = d_volumes.as_device_ptr().as_raw();
            let mut price_ptr = d_prices.as_device_ptr().as_raw();
            let mut count_ptr = d_counts.as_device_ptr().as_raw();
            let mut unit_ptr = d_unit_codes.as_device_ptr().as_raw();
            let mut div_ptr = d_divisors.as_device_ptr().as_raw();
            let mut warm_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut month_ptr = month_ids_ptr;
            let mut series_len_i = series_len as i32;
            let mut n_combos_i = n_combos as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut ts_ptr as *mut _ as *mut c_void,
                &mut vol_ptr as *mut _ as *mut c_void,
                &mut price_ptr as *mut _ as *mut c_void,
                &mut count_ptr as *mut _ as *mut c_void,
                &mut unit_ptr as *mut _ as *mut c_void,
                &mut div_ptr as *mut _ as *mut c_void,
                &mut warm_ptr as *mut _ as *mut c_void,
                &mut month_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut n_combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaVwapError::Cuda(e.to_string()))?;
        }

        Ok(())
    }

    pub fn vwap_batch_dev(
        &self,
        timestamps: &[i64],
        volumes: &[f64],
        prices: &[f64],
        sweep: &VwapBatchRange,
    ) -> Result<DeviceArrayF32, CudaVwapError> {
        let PreparedBatch {
            combos,
            counts,
            unit_codes,
            divisors,
            first_valids,
            month_ids,
            series_len,
        } = Self::prepare_batch_inputs(timestamps, volumes, prices, sweep)?;
        let n_combos = combos.len();

        let prices_f32: Vec<f32> = prices.iter().map(|&v| v as f32).collect();
        let volumes_f32: Vec<f32> = volumes.iter().map(|&v| v as f32).collect();

        let d_timestamps =
            DeviceBuffer::from_slice(timestamps).map_err(|e| CudaVwapError::Cuda(e.to_string()))?;
        let d_volumes = DeviceBuffer::from_slice(&volumes_f32)
            .map_err(|e| CudaVwapError::Cuda(e.to_string()))?;
        let d_prices = DeviceBuffer::from_slice(&prices_f32)
            .map_err(|e| CudaVwapError::Cuda(e.to_string()))?;
        let d_counts =
            DeviceBuffer::from_slice(&counts).map_err(|e| CudaVwapError::Cuda(e.to_string()))?;
        let d_unit_codes = DeviceBuffer::from_slice(&unit_codes)
            .map_err(|e| CudaVwapError::Cuda(e.to_string()))?;
        let d_divisors =
            DeviceBuffer::from_slice(&divisors).map_err(|e| CudaVwapError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaVwapError::Cuda(e.to_string()))?;
        let mut d_month_ids = if let Some(ids) = month_ids {
            Some(DeviceBuffer::from_slice(&ids).map_err(|e| CudaVwapError::Cuda(e.to_string()))?)
        } else {
            None
        };
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaVwapError::Cuda(e.to_string()))?;

        let month_ptr = d_month_ids
            .as_mut()
            .map(|buf| buf.as_device_ptr().as_raw())
            .unwrap_or(0);

        self.launch_kernel(
            &d_timestamps,
            &d_volumes,
            &d_prices,
            &d_counts,
            &d_unit_codes,
            &d_divisors,
            &d_first_valids,
            month_ptr,
            &mut d_out,
            series_len,
            n_combos,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaVwapError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    pub fn vwap_batch_into_host_f32(
        &self,
        timestamps: &[i64],
        volumes: &[f64],
        prices: &[f64],
        sweep: &VwapBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<VwapParams>), CudaVwapError> {
        let PreparedBatch {
            combos,
            counts,
            unit_codes,
            divisors,
            first_valids,
            month_ids,
            series_len,
        } = Self::prepare_batch_inputs(timestamps, volumes, prices, sweep)?;
        let n_combos = combos.len();
        let expected = n_combos * series_len;
        if out.len() != expected {
            return Err(CudaVwapError::InvalidInput(format!(
                "output slice wrong length: got {}, expected {}",
                out.len(),
                expected
            )));
        }

        let prices_f32: Vec<f32> = prices.iter().map(|&v| v as f32).collect();
        let volumes_f32: Vec<f32> = volumes.iter().map(|&v| v as f32).collect();

        let d_timestamps =
            DeviceBuffer::from_slice(timestamps).map_err(|e| CudaVwapError::Cuda(e.to_string()))?;
        let d_volumes = DeviceBuffer::from_slice(&volumes_f32)
            .map_err(|e| CudaVwapError::Cuda(e.to_string()))?;
        let d_prices = DeviceBuffer::from_slice(&prices_f32)
            .map_err(|e| CudaVwapError::Cuda(e.to_string()))?;
        let d_counts =
            DeviceBuffer::from_slice(&counts).map_err(|e| CudaVwapError::Cuda(e.to_string()))?;
        let d_unit_codes = DeviceBuffer::from_slice(&unit_codes)
            .map_err(|e| CudaVwapError::Cuda(e.to_string()))?;
        let d_divisors =
            DeviceBuffer::from_slice(&divisors).map_err(|e| CudaVwapError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaVwapError::Cuda(e.to_string()))?;
        let mut d_month_ids = if let Some(ids) = month_ids {
            Some(DeviceBuffer::from_slice(&ids).map_err(|e| CudaVwapError::Cuda(e.to_string()))?)
        } else {
            None
        };
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(expected) }
            .map_err(|e| CudaVwapError::Cuda(e.to_string()))?;

        let month_ptr = d_month_ids
            .as_mut()
            .map(|buf| buf.as_device_ptr().as_raw())
            .unwrap_or(0);

        self.launch_kernel(
            &d_timestamps,
            &d_volumes,
            &d_prices,
            &d_counts,
            &d_unit_codes,
            &d_divisors,
            &d_first_valids,
            month_ptr,
            &mut d_out,
            series_len,
            n_combos,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaVwapError::Cuda(e.to_string()))?;

        d_out
            .copy_to(out)
            .map_err(|e| CudaVwapError::Cuda(e.to_string()))?;

        Ok((n_combos, series_len, combos))
    }

    pub fn vwap_batch_device(
        &self,
        d_timestamps: &DeviceBuffer<i64>,
        d_volumes: &DeviceBuffer<f32>,
        d_prices: &DeviceBuffer<f32>,
        counts: &[i32],
        unit_codes: &[i32],
        divisors: &[i64],
        first_valids: &[i32],
        month_ids: Option<&DeviceBuffer<i32>>,
        series_len: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVwapError> {
        let n_combos = counts.len();
        let d_counts =
            DeviceBuffer::from_slice(counts).map_err(|e| CudaVwapError::Cuda(e.to_string()))?;
        let d_unit_codes =
            DeviceBuffer::from_slice(unit_codes).map_err(|e| CudaVwapError::Cuda(e.to_string()))?;
        let d_divisors =
            DeviceBuffer::from_slice(divisors).map_err(|e| CudaVwapError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(first_valids)
            .map_err(|e| CudaVwapError::Cuda(e.to_string()))?;

        self.vwap_batch_device_with_params(
            d_timestamps,
            d_volumes,
            d_prices,
            &d_counts,
            &d_unit_codes,
            &d_divisors,
            &d_first_valids,
            month_ids,
            series_len,
            n_combos,
            d_out,
        )
    }

    pub fn vwap_batch_device_with_params(
        &self,
        d_timestamps: &DeviceBuffer<i64>,
        d_volumes: &DeviceBuffer<f32>,
        d_prices: &DeviceBuffer<f32>,
        d_counts: &DeviceBuffer<i32>,
        d_unit_codes: &DeviceBuffer<i32>,
        d_divisors: &DeviceBuffer<i64>,
        d_first_valids: &DeviceBuffer<i32>,
        month_ids: Option<&DeviceBuffer<i32>>,
        series_len: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVwapError> {
        if d_counts.len() != n_combos
            || d_unit_codes.len() != n_combos
            || d_divisors.len() != n_combos
            || d_first_valids.len() != n_combos
        {
            return Err(CudaVwapError::InvalidInput(
                "parameter buffer length mismatch".into(),
            ));
        }
        if d_out.len() != n_combos * series_len {
            return Err(CudaVwapError::InvalidInput(format!(
                "output buffer wrong length: got {}, expected {}",
                d_out.len(),
                n_combos * series_len
            )));
        }

        let month_ptr = month_ids
            .map(|buf| buf.as_device_ptr().as_raw())
            .unwrap_or(0);

        self.launch_kernel(
            d_timestamps,
            d_volumes,
            d_prices,
            d_counts,
            d_unit_codes,
            d_divisors,
            d_first_valids,
            month_ptr,
            d_out,
            series_len,
            n_combos,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaVwapError::Cuda(e.to_string()))
    }
}
