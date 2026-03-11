#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::sgf::{
    build_endpoint_sgf_weights, effective_period, expand_grid, SgfBatchRange, SgfParams,
};
use cust::context::Context;
use cust::device::Device;
use cust::error::CudaError;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::{c_void, CString};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaSgfError {
    #[error(transparent)]
    Cuda(#[from] CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("insufficient VRAM: required={required}B, free={free}B, headroom={headroom}B")]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
}

pub struct CudaSgf {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    has_const_weights: bool,
    max_period_const: usize,
}

impl CudaSgf {
    pub fn new(device_id: usize) -> Result<Self, CudaSgfError> {
        cust::init(CudaFlags::empty()).map_err(CudaSgfError::Cuda)?;
        let device = Device::get_device(device_id as u32).map_err(CudaSgfError::Cuda)?;
        let context = Arc::new(Context::new(device).map_err(CudaSgfError::Cuda)?);
        let ptx = include_str!(concat!(env!("OUT_DIR"), "/sgf_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(CudaSgfError::Cuda)?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).map_err(CudaSgfError::Cuda)?;

        const SGF_MAX_PERIOD_RS: usize = 4096;
        let has_const_weights = module
            .get_global::<[f32; SGF_MAX_PERIOD_RS]>(&CString::new("c_sgf_weights").unwrap())
            .is_ok();

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            has_const_weights,
            max_period_const: if has_const_weights {
                SGF_MAX_PERIOD_RS
            } else {
                0
            },
        })
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> {
        self.context.clone()
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    pub fn synchronize(&self) -> Result<(), CudaSgfError> {
        self.stream.synchronize().map_err(Into::into)
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaSgfError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaSgfError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            }
        }
        Ok(())
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &SgfBatchRange,
    ) -> Result<(Vec<i32>, Vec<i32>, Vec<f32>, usize, usize, usize), CudaSgfError> {
        if data_f32.is_empty() {
            return Err(CudaSgfError::InvalidInput("empty data".into()));
        }
        let first = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaSgfError::InvalidInput("all values are NaN".into()))?;
        let combos = expand_grid(sweep).map_err(|e| CudaSgfError::InvalidInput(e.to_string()))?;
        let len = data_f32.len();
        let mut max_period = 0usize;
        let mut periods = Vec::with_capacity(combos.len());
        let mut warms = Vec::with_capacity(combos.len());
        let mut periods_usize = Vec::with_capacity(combos.len());

        for combo in &combos {
            let requested_period = combo.period.unwrap_or(21);
            let poly_order = combo.poly_order.unwrap_or(2);
            let period = effective_period(requested_period);
            if period < 3 || period > len {
                return Err(CudaSgfError::InvalidInput(format!(
                    "invalid effective period {} for len {}",
                    period, len
                )));
            }
            if poly_order >= period {
                return Err(CudaSgfError::InvalidInput(format!(
                    "poly_order {} must be < effective period {}",
                    poly_order, period
                )));
            }
            if len - first < period {
                return Err(CudaSgfError::InvalidInput(format!(
                    "not enough valid data: needed {}, have {}",
                    period,
                    len - first
                )));
            }
            max_period = max_period.max(period);
            periods.push(period as i32);
            periods_usize.push(period);
            warms.push((first + period - 1) as i32);
        }

        let mut weights_flat = vec![0.0f32; combos.len() * max_period];
        for (row, combo) in combos.iter().enumerate() {
            let weights = build_endpoint_sgf_weights(
                combo.period.unwrap_or(21),
                combo.poly_order.unwrap_or(2),
            )
            .map_err(|e| CudaSgfError::InvalidInput(e.to_string()))?;
            let period = periods_usize[row];
            let row_off = row * max_period;
            for idx in 0..period {
                weights_flat[row_off + idx] = weights[idx] as f32;
            }
        }

        Ok((periods, warms, weights_flat, len, combos.len(), max_period))
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &SgfParams,
    ) -> Result<(Vec<i32>, usize, Vec<f32>), CudaSgfError> {
        if cols == 0 || rows == 0 || data_tm_f32.len() != cols * rows {
            return Err(CudaSgfError::InvalidInput(
                "invalid time-major shape".to_string(),
            ));
        }
        let requested_period = params.period.unwrap_or(21);
        let poly_order = params.poly_order.unwrap_or(2);
        let period = effective_period(requested_period);
        if period < 3 || poly_order >= period {
            return Err(CudaSgfError::InvalidInput(format!(
                "invalid period/poly_order pair: period={}, poly_order={}",
                requested_period, poly_order
            )));
        }

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut found = None;
            for row in 0..rows {
                if !data_tm_f32[row * cols + series].is_nan() {
                    found = Some(row);
                    break;
                }
            }
            let first = found.ok_or_else(|| {
                CudaSgfError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            if rows - first < period {
                return Err(CudaSgfError::InvalidInput(format!(
                    "series {} lacks enough valid data",
                    series
                )));
            }
            first_valids[series] = first as i32;
        }

        let weights = build_endpoint_sgf_weights(requested_period, poly_order)
            .map_err(|e| CudaSgfError::InvalidInput(e.to_string()))?
            .iter()
            .map(|&x| x as f32)
            .collect();

        Ok((first_valids, period, weights))
    }

    fn upload_const_weights(&self, period: usize, weights: &[f32]) -> Result<(), CudaSgfError> {
        if !self.has_const_weights {
            return Ok(());
        }
        if period > self.max_period_const {
            return Err(CudaSgfError::InvalidInput(format!(
                "period {} exceeds compiled SGF_MAX_PERIOD {}",
                period, self.max_period_const
            )));
        }
        const SGF_MAX_PERIOD_RS: usize = 4096;
        let mut host = [0f32; SGF_MAX_PERIOD_RS];
        host[..period].copy_from_slice(weights);
        let mut symbol = self
            .module
            .get_global::<[f32; SGF_MAX_PERIOD_RS]>(&CString::new("c_sgf_weights").unwrap())
            .map_err(CudaSgfError::Cuda)?;
        symbol.copy_from(&host).map_err(CudaSgfError::Cuda)?;
        Ok(())
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warms: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSgfError> {
        let func = self.module.get_function("sgf_batch_f32").map_err(|_| {
            CudaSgfError::MissingKernelSymbol {
                name: "sgf_batch_f32",
            }
        })?;
        let block_x = 128u32;
        let grid_x = (series_len as u32).div_ceil(block_x);
        let grid: GridSize = (grid_x.max(1), n_combos as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        let shared_bytes =
            ((max_period + block_x as usize + max_period) * std::mem::size_of::<f32>()) as u32;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut weights_ptr = d_weights.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut warms_ptr = d_warms.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut n_combos_i = n_combos as i32;
            let mut max_period_i = max_period as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut weights_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut warms_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut n_combos_i as *mut _ as *mut c_void,
                &mut max_period_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes, args)
                .map_err(CudaSgfError::Cuda)?;
        }
        Ok(())
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_weights_opt: Option<&DeviceBuffer<f32>>,
        d_first_valids: &DeviceBuffer<i32>,
        period: usize,
        cols: usize,
        rows: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSgfError> {
        let func = self
            .module
            .get_function("sgf_multi_series_one_param_f32")
            .map_err(|_| CudaSgfError::MissingKernelSymbol {
                name: "sgf_multi_series_one_param_f32",
            })?;
        let tx = 128u32;
        let ty = 4u32;
        let grid: GridSize = (
            ((rows as u32).div_ceil(tx)).max(1),
            ((cols as u32).div_ceil(ty)).max(1),
            1,
        )
            .into();
        let block: BlockSize = (tx, ty, 1).into();
        let shared_floats = ((tx as usize) + period - 1) * (ty as usize)
            + if self.has_const_weights { 0 } else { period };
        let shared_bytes = (shared_floats * std::mem::size_of::<f32>()) as u32;

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut weights_ptr = d_weights_opt
                .map(|w| w.as_device_ptr().as_raw())
                .unwrap_or(0);
            let mut period_i = period as i32;
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut weights_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_valids_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes, args)
                .map_err(CudaSgfError::Cuda)?;
        }
        Ok(())
    }

    pub fn sgf_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warms: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSgfError> {
        self.launch_batch_kernel(
            d_prices, d_weights, d_periods, d_warms, series_len, n_combos, max_period, d_out,
        )
    }

    pub fn sgf_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &SgfBatchRange,
    ) -> Result<DeviceArrayF32, CudaSgfError> {
        let (periods, warms, weights_flat, series_len, n_combos, max_period) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let required = data_f32.len() * 4
            + periods.len() * 4
            + warms.len() * 4
            + weights_flat.len() * 4
            + n_combos * series_len * 4;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        let d_prices = DeviceBuffer::from_slice(data_f32).map_err(CudaSgfError::Cuda)?;
        let d_weights = DeviceBuffer::from_slice(&weights_flat).map_err(CudaSgfError::Cuda)?;
        let d_periods = DeviceBuffer::from_slice(&periods).map_err(CudaSgfError::Cuda)?;
        let d_warms = DeviceBuffer::from_slice(&warms).map_err(CudaSgfError::Cuda)?;
        let mut d_out = unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
            .map_err(CudaSgfError::Cuda)?;

        self.launch_batch_kernel(
            &d_prices, &d_weights, &d_periods, &d_warms, series_len, n_combos, max_period,
            &mut d_out,
        )?;
        self.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    pub fn sgf_multi_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        period: i32,
        num_series: i32,
        series_len: i32,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSgfError> {
        if self.has_const_weights {
            let mut host_w = vec![0f32; period as usize];
            d_weights.copy_to(&mut host_w).map_err(CudaSgfError::Cuda)?;
            self.upload_const_weights(period as usize, &host_w)?;
            self.launch_many_series_kernel(
                d_prices_tm,
                None,
                d_first_valids,
                period as usize,
                num_series as usize,
                series_len as usize,
                d_out_tm,
            )
        } else {
            self.launch_many_series_kernel(
                d_prices_tm,
                Some(d_weights),
                d_first_valids,
                period as usize,
                num_series as usize,
                series_len as usize,
                d_out_tm,
            )
        }
    }

    pub fn sgf_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &SgfParams,
    ) -> Result<DeviceArrayF32, CudaSgfError> {
        let (first_valids, period, weights) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let required = data_tm_f32.len() * 4
            + first_valids.len() * 4
            + weights.len() * 4
            + data_tm_f32.len() * 4;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        if self.has_const_weights {
            self.upload_const_weights(period, &weights)?;
        }
        let d_prices = DeviceBuffer::from_slice(data_tm_f32).map_err(CudaSgfError::Cuda)?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).map_err(CudaSgfError::Cuda)?;
        let d_weights = if self.has_const_weights {
            None
        } else {
            Some(DeviceBuffer::from_slice(&weights).map_err(CudaSgfError::Cuda)?)
        };
        let mut d_out =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.map_err(CudaSgfError::Cuda)?;
        self.launch_many_series_kernel(
            &d_prices,
            d_weights.as_ref(),
            &d_first_valids,
            period,
            cols,
            rows,
            &mut d_out,
        )?;
        self.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }
}
