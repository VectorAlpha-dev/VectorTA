#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::vwmacd::{VwmacdBatchRange, VwmacdParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;

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
pub struct CudaVwmacdPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaVwmacdPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum CudaVwmacdError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

pub struct DeviceVwmacdTriplet {
    pub macd: DeviceArrayF32,
    pub signal: DeviceArrayF32,
    pub hist: DeviceArrayF32,
}
impl DeviceVwmacdTriplet {
    #[inline]
    pub fn rows(&self) -> usize {
        self.macd.rows
    }
    #[inline]
    pub fn cols(&self) -> usize {
        self.macd.cols
    }
}

pub struct CudaVwmacd {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaVwmacdPolicy,
}

pub struct CudaVwmacdBatchPlan {
    combos: Vec<VwmacdParams>,
    d_pv: DeviceBuffer<f64>,
    d_vol: DeviceBuffer<f64>,
    d_fasts: DeviceBuffer<i32>,
    d_slows: DeviceBuffer<i32>,
    d_sigs: DeviceBuffer<i32>,
    d_macd: DeviceBuffer<f32>,
    d_signal: DeviceBuffer<f32>,
    d_hist: DeviceBuffer<f32>,
    rows: usize,
    cols: usize,
    device_id: u32,
    first_valid: usize,
}
impl CudaVwmacdBatchPlan {
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    #[inline]
    pub fn params(&self) -> &[VwmacdParams] {
        &self.combos
    }

    #[inline]
    pub fn outputs(&self) -> (&DeviceBuffer<f32>, &DeviceBuffer<f32>, &DeviceBuffer<f32>) {
        (&self.d_macd, &self.d_signal, &self.d_hist)
    }

    pub fn into_device_triplet_and_params(self) -> (DeviceVwmacdTriplet, Vec<VwmacdParams>) {
        (
            DeviceVwmacdTriplet {
                macd: DeviceArrayF32 {
                    buf: self.d_macd,
                    rows: self.rows,
                    cols: self.cols,
                },
                signal: DeviceArrayF32 {
                    buf: self.d_signal,
                    rows: self.rows,
                    cols: self.cols,
                },
                hist: DeviceArrayF32 {
                    buf: self.d_hist,
                    rows: self.rows,
                    cols: self.cols,
                },
            },
            self.combos,
        )
    }
}

impl CudaVwmacd {
    pub fn new(device_id: usize) -> Result<Self, CudaVwmacdError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Context::new(device)?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/vwmacd_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = crate::load_cuda_embedded_module!("vwmacd_kernel")?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaVwmacdPolicy::default(),
        })
    }

    #[inline]
    pub fn set_policy(&mut self, policy: CudaVwmacdPolicy) {
        self.policy = policy;
    }
    #[inline]
    pub fn policy(&self) -> &CudaVwmacdPolicy {
        &self.policy
    }
    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaVwmacdError> {
        self.stream.synchronize().map_err(Into::into)
    }
    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    fn launch_build_prefix_sums_one_series_f64(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_volumes: &DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        d_pv: &mut DeviceBuffer<f64>,
        d_vol: &mut DeviceBuffer<f64>,
    ) -> Result<(), CudaVwmacdError> {
        let func = self
            .module
            .get_function("vwmacd_build_prefix_one_series_f64")
            .map_err(|_| CudaVwmacdError::MissingKernelSymbol {
                name: "vwmacd_build_prefix_one_series_f64",
            })?;
        let grid: GridSize = (1, 1, 1).into();
        let block: BlockSize = (1, 1, 1).into();
        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut volumes_ptr = d_volumes.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut first_i = first_valid as i32;
            let mut pv_ptr = d_pv.as_device_ptr().as_raw();
            let mut vol_ptr = d_vol.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut volumes_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut pv_ptr as *mut _ as *mut c_void,
                &mut vol_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    fn launch_build_prefix_sums_time_major_f64(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_volumes_tm: &DeviceBuffer<f32>,
        d_firsts: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        d_pv_tm: &mut DeviceBuffer<f64>,
        d_vol_tm: &mut DeviceBuffer<f64>,
    ) -> Result<(), CudaVwmacdError> {
        let func = self
            .module
            .get_function("vwmacd_build_prefix_time_major_f64")
            .map_err(|_| CudaVwmacdError::MissingKernelSymbol {
                name: "vwmacd_build_prefix_time_major_f64",
            })?;
        let block_x = 256u32;
        let cols_u32 = u32::try_from(cols)
            .map_err(|_| CudaVwmacdError::InvalidInput("cols exceeds CUDA grid limit".into()))?;
        let grid_x = (cols_u32 + block_x - 1) / block_x;
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut volumes_ptr = d_volumes_tm.as_device_ptr().as_raw();
            let mut firsts_ptr = d_firsts.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut pv_ptr = d_pv_tm.as_device_ptr().as_raw();
            let mut vol_ptr = d_vol_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut volumes_ptr as *mut _ as *mut c_void,
                &mut firsts_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut pv_ptr as *mut _ as *mut c_void,
                &mut vol_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaVwmacdError> {
        if let Ok((free, _)) = mem_get_info() {
            let required = required_bytes
                .checked_add(headroom_bytes)
                .ok_or_else(|| CudaVwmacdError::InvalidInput("size overflow".into()))?;
            if required > free {
                return Err(CudaVwmacdError::OutOfMemory {
                    required,
                    free,
                    headroom: headroom_bytes,
                });
            }
        }
        Ok(())
    }

    pub fn vwmacd_batch_dev(
        &self,
        prices_f32: &[f32],
        volumes_f32: &[f32],
        sweep: &VwmacdBatchRange,
    ) -> Result<(DeviceVwmacdTriplet, Vec<VwmacdParams>), CudaVwmacdError> {
        let len = prices_f32.len();
        if len == 0 || volumes_f32.len() != len {
            return Err(CudaVwmacdError::InvalidInput(
                "mismatched or empty inputs".into(),
            ));
        }
        let first_valid = first_valid_pair_f32(prices_f32, volumes_f32)
            .ok_or_else(|| CudaVwmacdError::InvalidInput("all values are NaN".into()))?;
        let d_prices = DeviceBuffer::from_slice(prices_f32)?;
        let d_volumes = DeviceBuffer::from_slice(volumes_f32)?;
        let result = self.vwmacd_batch_dev_from_device_inputs(
            &d_prices,
            &d_volumes,
            len,
            first_valid,
            sweep,
        )?;
        self.synchronize()?;
        Ok(result)
    }

    pub fn vwmacd_batch_dev_from_device_inputs(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_volumes: &DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        sweep: &VwmacdBatchRange,
    ) -> Result<(DeviceVwmacdTriplet, Vec<VwmacdParams>), CudaVwmacdError> {
        if len == 0 || d_prices.len() != len || d_volumes.len() != len {
            return Err(CudaVwmacdError::InvalidInput(
                "mismatched or empty inputs".into(),
            ));
        }
        if first_valid >= len {
            return Err(CudaVwmacdError::InvalidInput(
                "first_valid out of range".into(),
            ));
        }

        let mut plan = self.prepare_vwmacd_batch_plan(len, first_valid, sweep)?;
        self.launch_vwmacd_batch_plan(d_prices, d_volumes, &mut plan)?;
        Ok(plan.into_device_triplet_and_params())
    }

    pub fn prepare_vwmacd_batch_plan(
        &self,
        len: usize,
        first_valid: usize,
        sweep: &VwmacdBatchRange,
    ) -> Result<CudaVwmacdBatchPlan, CudaVwmacdError> {
        if len == 0 {
            return Err(CudaVwmacdError::InvalidInput("empty input".into()));
        }
        if first_valid >= len {
            return Err(CudaVwmacdError::InvalidInput(
                "first_valid out of range".into(),
            ));
        }
        if !sweep.fast_ma_type.eq_ignore_ascii_case("sma")
            || !sweep.slow_ma_type.eq_ignore_ascii_case("sma")
            || !sweep.signal_ma_type.eq_ignore_ascii_case("ema")
        {
            return Err(CudaVwmacdError::InvalidPolicy(
                "CUDA VWMACD supports fast=\"sma\", slow=\"sma\", signal=\"ema\" only",
            ));
        }

        let combos = expand_grid(sweep)?;

        let mut max_macd_warm = 0usize;
        for c in &combos {
            let f = c.fast_period.unwrap();
            let s = c.slow_period.unwrap();
            let macd_warm = first_valid
                .checked_add(f.max(s).saturating_sub(1))
                .ok_or_else(|| CudaVwmacdError::InvalidInput("size overflow".into()))?;
            if macd_warm > max_macd_warm {
                max_macd_warm = macd_warm;
            }
        }
        if len <= max_macd_warm {
            return Err(CudaVwmacdError::InvalidInput(
                "not enough valid data".into(),
            ));
        }

        let rows = combos.len();
        let fasts: Vec<i32> = combos
            .iter()
            .map(|c| c.fast_period.unwrap() as i32)
            .collect();
        let slows: Vec<i32> = combos
            .iter()
            .map(|c| c.slow_period.unwrap() as i32)
            .collect();
        let sigs: Vec<i32> = combos
            .iter()
            .map(|c| c.signal_period.unwrap() as i32)
            .collect();

        let f64_sz = std::mem::size_of::<f64>();
        let f32_sz = std::mem::size_of::<f32>();
        let i32_sz = std::mem::size_of::<i32>();
        let prefix_bytes = len
            .checked_mul(f64_sz)
            .and_then(|b| b.checked_add(len.checked_mul(f64_sz)?))
            .ok_or_else(|| CudaVwmacdError::InvalidInput("size overflow".into()))?;
        let param_len = fasts
            .len()
            .checked_add(slows.len())
            .and_then(|n| n.checked_add(sigs.len()))
            .ok_or_else(|| CudaVwmacdError::InvalidInput("size overflow".into()))?;
        let param_bytes = param_len
            .checked_mul(i32_sz)
            .ok_or_else(|| CudaVwmacdError::InvalidInput("size overflow".into()))?;
        let elems = rows
            .checked_mul(len)
            .ok_or_else(|| CudaVwmacdError::InvalidInput("size overflow".into()))?;
        let out_bytes = elems
            .checked_mul(3)
            .and_then(|n| n.checked_mul(f32_sz))
            .ok_or_else(|| CudaVwmacdError::InvalidInput("size overflow".into()))?;
        let bytes = prefix_bytes
            .checked_add(param_bytes)
            .and_then(|b| b.checked_add(out_bytes))
            .ok_or_else(|| CudaVwmacdError::InvalidInput("size overflow".into()))?;
        CudaVwmacd::will_fit(bytes, 64 * 1024 * 1024)?;

        let d_pv: DeviceBuffer<f64> = unsafe { DeviceBuffer::uninitialized(len) }?;
        let d_vol: DeviceBuffer<f64> = unsafe { DeviceBuffer::uninitialized(len) }?;
        let d_fasts = DeviceBuffer::from_slice(&fasts)?;
        let d_slows = DeviceBuffer::from_slice(&slows)?;
        let d_sigs = DeviceBuffer::from_slice(&sigs)?;

        let d_macd: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }?;
        let d_signal: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }?;
        let d_hist: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }?;

        Ok(CudaVwmacdBatchPlan {
            combos,
            d_pv,
            d_vol,
            d_fasts,
            d_slows,
            d_sigs,
            d_macd,
            d_signal,
            d_hist,
            rows,
            cols: len,
            device_id: self.device_id,
            first_valid,
        })
    }

    pub fn launch_vwmacd_batch_plan(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_volumes: &DeviceBuffer<f32>,
        plan: &mut CudaVwmacdBatchPlan,
    ) -> Result<(), CudaVwmacdError> {
        if d_prices.len() != plan.cols || d_volumes.len() != plan.cols {
            return Err(CudaVwmacdError::InvalidInput(
                "device input length mismatch".into(),
            ));
        }
        self.launch_build_prefix_sums_one_series_f64(
            d_prices,
            d_volumes,
            plan.cols,
            plan.first_valid,
            &mut plan.d_pv,
            &mut plan.d_vol,
        )?;

        self.launch_batch(
            &plan.d_pv,
            &plan.d_vol,
            &plan.d_fasts,
            &plan.d_slows,
            &plan.d_sigs,
            plan.cols,
            plan.first_valid,
            plan.rows,
            &mut plan.d_macd,
            &mut plan.d_signal,
            &mut plan.d_hist,
        )
    }

    pub fn vwmacd_many_series_one_param_time_major_dev(
        &self,
        prices_tm_f32: &[f32],
        volumes_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &VwmacdParams,
    ) -> Result<DeviceVwmacdTriplet, CudaVwmacdError> {
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaVwmacdError::InvalidInput("size overflow".into()))?;
        if cols == 0
            || rows == 0
            || prices_tm_f32.len() != expected
            || volumes_tm_f32.len() != expected
        {
            return Err(CudaVwmacdError::InvalidInput(
                "invalid time-major inputs".into(),
            ));
        }
        let f = params.fast_period.unwrap_or(12);
        let s = params.slow_period.unwrap_or(26);
        let g = params.signal_period.unwrap_or(9);
        if f == 0 || s == 0 || g == 0 {
            return Err(CudaVwmacdError::InvalidInput("zero period".into()));
        }
        if !params
            .fast_ma_type
            .as_deref()
            .unwrap_or("sma")
            .eq_ignore_ascii_case("sma")
            || !params
                .slow_ma_type
                .as_deref()
                .unwrap_or("sma")
                .eq_ignore_ascii_case("sma")
            || !params
                .signal_ma_type
                .as_deref()
                .unwrap_or("ema")
                .eq_ignore_ascii_case("ema")
        {
            return Err(CudaVwmacdError::InvalidPolicy(
                "CUDA VWMACD supports fast=\"sma\", slow=\"sma\", signal=\"ema\" only",
            ));
        }

        let first_valids = first_valids_time_major_f32(prices_tm_f32, volumes_tm_f32, cols, rows);

        let mut ok = false;
        for &fv in &first_valids {
            if (rows as i32 - fv) as usize > f.max(s) {
                ok = true;
                break;
            }
        }
        if !ok {
            return Err(CudaVwmacdError::InvalidInput(
                "not enough valid data".into(),
            ));
        }

        let f64_sz = std::mem::size_of::<f64>();
        let f32_sz = std::mem::size_of::<f32>();
        let i32_sz = std::mem::size_of::<i32>();
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaVwmacdError::InvalidInput("size overflow".into()))?;
        let input_bytes = elems
            .checked_mul(2)
            .and_then(|n| n.checked_mul(f32_sz))
            .ok_or_else(|| CudaVwmacdError::InvalidInput("size overflow".into()))?;
        let prefix_bytes = elems
            .checked_mul(2)
            .and_then(|n| n.checked_mul(f64_sz))
            .ok_or_else(|| CudaVwmacdError::InvalidInput("size overflow".into()))?;
        let param_bytes = first_valids
            .len()
            .checked_mul(i32_sz)
            .ok_or_else(|| CudaVwmacdError::InvalidInput("size overflow".into()))?;
        let out_bytes = elems
            .checked_mul(3)
            .and_then(|n| n.checked_mul(f32_sz))
            .ok_or_else(|| CudaVwmacdError::InvalidInput("size overflow".into()))?;
        let bytes = prefix_bytes
            .checked_add(input_bytes)
            .and_then(|b| b.checked_add(param_bytes))
            .and_then(|b| b.checked_add(out_bytes))
            .ok_or_else(|| CudaVwmacdError::InvalidInput("size overflow".into()))?;
        CudaVwmacd::will_fit(bytes, 64 * 1024 * 1024)?;

        let d_prices = DeviceBuffer::from_slice(prices_tm_f32)?;
        let d_volumes = DeviceBuffer::from_slice(volumes_tm_f32)?;
        let h_firsts = LockedBuffer::from_slice(&first_valids)?;
        let d_first: DeviceBuffer<i32> =
            unsafe { DeviceBuffer::from_slice_async(&*h_firsts, &self.stream) }?;
        let mut d_pv: DeviceBuffer<f64> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }?;
        let mut d_vol: DeviceBuffer<f64> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }?;

        self.launch_build_prefix_sums_time_major_f64(
            &d_prices, &d_volumes, &d_first, cols, rows, &mut d_pv, &mut d_vol,
        )?;

        let mut d_macd: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }?;
        let mut d_signal: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }?;
        let mut d_hist: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }?;

        self.launch_many_series(
            &d_pv,
            &d_vol,
            &d_first,
            f,
            s,
            g,
            cols,
            rows,
            &mut d_macd,
            &mut d_signal,
            &mut d_hist,
        )?;
        self.synchronize()?;

        Ok(DeviceVwmacdTriplet {
            macd: DeviceArrayF32 {
                buf: d_macd,
                rows,
                cols,
            },
            signal: DeviceArrayF32 {
                buf: d_signal,
                rows,
                cols,
            },
            hist: DeviceArrayF32 {
                buf: d_hist,
                rows,
                cols,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch(
        &self,
        d_pv: &DeviceBuffer<f64>,
        d_vol: &DeviceBuffer<f64>,
        d_fasts: &DeviceBuffer<i32>,
        d_slows: &DeviceBuffer<i32>,
        d_sigs: &DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        rows: usize,
        d_macd: &mut DeviceBuffer<f32>,
        d_signal: &mut DeviceBuffer<f32>,
        d_hist: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVwmacdError> {
        let macd_func = self
            .module
            .get_function("vwmacd_batch_macd_tiled_f32")
            .map_err(|_| CudaVwmacdError::MissingKernelSymbol {
                name: "vwmacd_batch_macd_tiled_f32",
            })?;
        let signal_func = self
            .module
            .get_function("vwmacd_batch_signal_serial_f32")
            .map_err(|_| CudaVwmacdError::MissingKernelSymbol {
                name: "vwmacd_batch_signal_serial_f32",
            })?;
        let hist_func = self
            .module
            .get_function("vwmacd_batch_hist_tiled_f32")
            .map_err(|_| CudaVwmacdError::MissingKernelSymbol {
                name: "vwmacd_batch_hist_tiled_f32",
            })?;
        let tile_block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            _ => 256,
        };
        if tile_block_x == 0 {
            return Err(CudaVwmacdError::InvalidPolicy("block_x must be nonzero"));
        }
        let len_u32 = u32::try_from(len)
            .map_err(|_| CudaVwmacdError::InvalidInput("len exceeds CUDA grid limit".into()))?;
        let rows_u32 = u32::try_from(rows)
            .map_err(|_| CudaVwmacdError::InvalidInput("rows exceeds CUDA grid limit".into()))?;
        let tile_grid_x = (len_u32 + tile_block_x - 1) / tile_block_x;
        let tile_grid: GridSize = (tile_grid_x, rows_u32, 1).into();
        let tile_block: BlockSize = (tile_block_x, 1, 1).into();
        let row_block_x = 64u32;
        let row_grid_x = (rows_u32 + row_block_x - 1) / row_block_x;
        let row_grid: GridSize = (row_grid_x, 1, 1).into();
        let row_block: BlockSize = (row_block_x, 1, 1).into();

        unsafe {
            let mut pv_ptr = d_pv.as_device_ptr().as_raw();
            let mut vol_ptr = d_vol.as_device_ptr().as_raw();
            let mut f_ptr = d_fasts.as_device_ptr().as_raw();
            let mut s_ptr = d_slows.as_device_ptr().as_raw();
            let mut g_ptr = d_sigs.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut fv_i = first_valid as i32;
            let mut rows_i = rows as i32;
            let mut macd_ptr = d_macd.as_device_ptr().as_raw();
            let mut sig_ptr = d_signal.as_device_ptr().as_raw();
            let mut hist_ptr = d_hist.as_device_ptr().as_raw();
            let macd_args: &mut [*mut c_void] = &mut [
                &mut pv_ptr as *mut _ as *mut c_void,
                &mut vol_ptr as *mut _ as *mut c_void,
                &mut f_ptr as *mut _ as *mut c_void,
                &mut s_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut fv_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut macd_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&macd_func, tile_grid, tile_block, 0, macd_args)?;

            let signal_args: &mut [*mut c_void] = &mut [
                &mut macd_ptr as *mut _ as *mut c_void,
                &mut f_ptr as *mut _ as *mut c_void,
                &mut s_ptr as *mut _ as *mut c_void,
                &mut g_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut fv_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut sig_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&signal_func, row_grid, row_block, 0, signal_args)?;

            let hist_args: &mut [*mut c_void] = &mut [
                &mut macd_ptr as *mut _ as *mut c_void,
                &mut f_ptr as *mut _ as *mut c_void,
                &mut s_ptr as *mut _ as *mut c_void,
                &mut g_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut fv_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut sig_ptr as *mut _ as *mut c_void,
                &mut hist_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&hist_func, tile_grid, tile_block, 0, hist_args)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series(
        &self,
        d_pv_tm: &DeviceBuffer<f64>,
        d_vol_tm: &DeviceBuffer<f64>,
        d_first: &DeviceBuffer<i32>,
        fast: usize,
        slow: usize,
        signal: usize,
        cols: usize,
        rows: usize,
        d_macd_tm: &mut DeviceBuffer<f32>,
        d_signal_tm: &mut DeviceBuffer<f32>,
        d_hist_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVwmacdError> {
        let func = self
            .module
            .get_function("vwmacd_many_series_one_param_time_major_f32")
            .map_err(|_| CudaVwmacdError::MissingKernelSymbol {
                name: "vwmacd_many_series_one_param_time_major_f32",
            })?;
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            _ => 256,
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut pv_ptr = d_pv_tm.as_device_ptr().as_raw();
            let mut vol_ptr = d_vol_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first.as_device_ptr().as_raw();
            let mut f_i = fast as i32;
            let mut s_i = slow as i32;
            let mut g_i = signal as i32;
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut macd_ptr = d_macd_tm.as_device_ptr().as_raw();
            let mut signal_ptr = d_signal_tm.as_device_ptr().as_raw();
            let mut hist_ptr = d_hist_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut pv_ptr as *mut _ as *mut c_void,
                &mut vol_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut f_i as *mut _ as *mut c_void,
                &mut s_i as *mut _ as *mut c_void,
                &mut g_i as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut macd_ptr as *mut _ as *mut c_void,
                &mut signal_ptr as *mut _ as *mut c_void,
                &mut hist_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }
}

fn first_valid_pair_f32(close: &[f32], volume: &[f32]) -> Option<usize> {
    close
        .iter()
        .zip(volume)
        .position(|(c, v)| !c.is_nan() && !v.is_nan())
}

fn compute_prefix_sums(
    prices: &[f32],
    volumes: &[f32],
    first_valid: usize,
    len: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut pv_prefix = vec![0f64; len];
    let mut vol_prefix = vec![0f64; len];
    let mut acc_pv = 0f64;
    let mut acc_vol = 0f64;
    for i in first_valid..len {
        let p = prices[i] as f64;
        let v = volumes[i] as f64;
        if p.is_nan() || v.is_nan() || acc_pv.is_nan() || acc_vol.is_nan() {
            acc_pv = f64::NAN;
            acc_vol = f64::NAN;
        } else {
            acc_pv += p * v;
            acc_vol += v;
        }
        pv_prefix[i] = acc_pv;
        vol_prefix[i] = acc_vol;
    }
    (pv_prefix, vol_prefix)
}

fn compute_prefix_sums_time_major(
    prices_tm: &[f32],
    volumes_tm: &[f32],
    cols: usize,
    rows: usize,
    first_valids: &[i32],
) -> (Vec<f64>, Vec<f64>) {
    let mut pv_prefix = vec![0f64; prices_tm.len()];
    let mut vol_prefix = vec![0f64; volumes_tm.len()];
    for series in 0..cols {
        let fv = first_valids[series].max(0) as usize;
        let mut acc_pv = 0f64;
        let mut acc_vol = 0f64;
        for r in 0..rows {
            let idx = r * cols + series;
            if r >= fv {
                let p = prices_tm[idx] as f64;
                let v = volumes_tm[idx] as f64;
                if p.is_nan() || v.is_nan() || acc_pv.is_nan() || acc_vol.is_nan() {
                    acc_pv = f64::NAN;
                    acc_vol = f64::NAN;
                } else {
                    acc_pv += p * v;
                    acc_vol += v;
                }
            }
            pv_prefix[idx] = acc_pv;
            vol_prefix[idx] = acc_vol;
        }
    }
    (pv_prefix, vol_prefix)
}

fn first_valids_time_major_f32(
    prices_tm: &[f32],
    volumes_tm: &[f32],
    cols: usize,
    rows: usize,
) -> Vec<i32> {
    let mut out = vec![0i32; cols];
    for series in 0..cols {
        let mut fv: i32 = -1;
        for r in 0..rows {
            let idx = r * cols + series;
            let c = prices_tm[idx];
            let v = volumes_tm[idx];
            if !c.is_nan() && !v.is_nan() {
                fv = r as i32;
                break;
            }
        }
        out[series] = if fv < 0 { rows as i32 } else { fv };
    }
    out
}

fn expand_grid(r: &VwmacdBatchRange) -> Result<Vec<VwmacdParams>, CudaVwmacdError> {
    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, CudaVwmacdError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let st = step.max(1);
            let mut v = Vec::new();
            let mut cur = start;
            while cur <= end {
                v.push(cur);
                let next = cur.saturating_add(st);
                if next == cur {
                    break;
                }
                cur = next;
            }
            if v.is_empty() {
                return Err(CudaVwmacdError::InvalidInput(format!(
                    "invalid range: start={start}, end={end}, step={step}"
                )));
            }
            return Ok(v);
        }
        let mut v = Vec::new();
        let mut x = start as isize;
        let end_i = end as isize;
        let st = (step as isize).max(1);
        while x >= end_i {
            v.push(x as usize);
            x -= st;
        }
        if v.is_empty() {
            return Err(CudaVwmacdError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        Ok(v)
    }

    let fasts = axis_usize(r.fast)?;
    let slows = axis_usize(r.slow)?;
    let signals = axis_usize(r.signal)?;

    let cap = fasts
        .len()
        .checked_mul(slows.len())
        .and_then(|x| x.checked_mul(signals.len()))
        .ok_or_else(|| CudaVwmacdError::InvalidInput("size overflow".into()))?;
    if cap == 0 {
        return Err(CudaVwmacdError::InvalidInput(
            "empty parameter sweep".into(),
        ));
    }

    let mut out = Vec::with_capacity(cap);
    for &f in &fasts {
        for &s in &slows {
            for &g in &signals {
                out.push(VwmacdParams {
                    fast_period: Some(f),
                    slow_period: Some(s),
                    signal_period: Some(g),
                    fast_ma_type: Some(r.fast_ma_type.clone()),
                    slow_ma_type: Some(r.slow_ma_type.clone()),
                    signal_ma_type: Some(r.signal_ma_type.clone()),
                });
            }
        }
    }
    Ok(out)
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices, gen_time_major_volumes};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const MANY_SERIES_COLS: usize = 128;
    const MANY_SERIES_LEN: usize = 100_000;

    fn bytes_one_series_many_params() -> usize {
        let rows = 250usize;
        let in_b = 2 * ONE_SERIES_LEN * 4;
        let out_b = 3 * rows * ONE_SERIES_LEN * 4;
        in_b + out_b + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_LEN * MANY_SERIES_COLS;
        3 * elems * 4 + 2 * elems * 8 + 64 * 1024 * 1024
    }

    struct BatchDevState {
        cuda: CudaVwmacd,
        d_pv: DeviceBuffer<f64>,
        d_vol: DeviceBuffer<f64>,
        d_fasts: DeviceBuffer<i32>,
        d_slows: DeviceBuffer<i32>,
        d_sigs: DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        rows: usize,
        d_macd: DeviceBuffer<f32>,
        d_signal: DeviceBuffer<f32>,
        d_hist: DeviceBuffer<f32>,
    }
    impl CudaBenchState for BatchDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch(
                    &self.d_pv,
                    &self.d_vol,
                    &self.d_fasts,
                    &self.d_slows,
                    &self.d_sigs,
                    self.len,
                    self.first_valid,
                    self.rows,
                    &mut self.d_macd,
                    &mut self.d_signal,
                    &mut self.d_hist,
                )
                .expect("vwmacd batch kernel");
            self.cuda.synchronize().expect("vwmacd sync");
        }
    }
    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        let mut v = Vec::new();
        v.push(
            CudaBenchScenario::new(
                "vwmacd",
                "one_series_many_params",
                "vwmacd_cuda_batch_dev",
                "1m_x_250",
                || {
                    let cuda = CudaVwmacd::new(0).unwrap();
                    let price = gen_series(ONE_SERIES_LEN);
                    let mut vol = gen_series(ONE_SERIES_LEN);
                    for x in &mut vol {
                        if x.is_finite() {
                            *x = x.abs() * 100.0 + 10.0;
                        }
                    }
                    let sweep = VwmacdBatchRange {
                        fast: (8, 57, 1),
                        slow: (16, 20, 1),
                        signal: (9, 9, 0),
                        fast_ma_type: "sma".into(),
                        slow_ma_type: "sma".into(),
                        signal_ma_type: "ema".into(),
                    };
                    let combos = expand_grid(&sweep).expect("vwmacd expand grid");
                    let rows = combos.len();
                    let first_valid =
                        first_valid_pair_f32(&price, &vol).expect("vwmacd first_valid");
                    let (pv_prefix, vol_prefix) =
                        compute_prefix_sums(&price, &vol, first_valid, price.len());
                    let fasts: Vec<i32> = combos
                        .iter()
                        .map(|c| c.fast_period.unwrap_or(0) as i32)
                        .collect();
                    let slows: Vec<i32> = combos
                        .iter()
                        .map(|c| c.slow_period.unwrap_or(0) as i32)
                        .collect();
                    let sigs: Vec<i32> = combos
                        .iter()
                        .map(|c| c.signal_period.unwrap_or(0) as i32)
                        .collect();

                    let d_pv = DeviceBuffer::from_slice(&pv_prefix).expect("d_pv");
                    let d_vol = DeviceBuffer::from_slice(&vol_prefix).expect("d_vol");
                    let d_fasts = DeviceBuffer::from_slice(&fasts).expect("d_fasts");
                    let d_slows = DeviceBuffer::from_slice(&slows).expect("d_slows");
                    let d_sigs = DeviceBuffer::from_slice(&sigs).expect("d_sigs");
                    let elems = rows.checked_mul(price.len()).expect("rows*len");
                    let d_macd: DeviceBuffer<f32> =
                        unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_macd");
                    let d_signal: DeviceBuffer<f32> =
                        unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_signal");
                    let d_hist: DeviceBuffer<f32> =
                        unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_hist");
                    cuda.synchronize().expect("sync after prep");

                    Box::new(BatchDevState {
                        cuda,
                        d_pv,
                        d_vol,
                        d_fasts,
                        d_slows,
                        d_sigs,
                        len: price.len(),
                        first_valid,
                        rows,
                        d_macd,
                        d_signal,
                        d_hist,
                    })
                },
            )
            .with_mem_required(bytes_one_series_many_params()),
        );

        v.push(
            CudaBenchScenario::new(
                "vwmacd",
                "many_series_one_param",
                "vwmacd_cuda_many_series_one_param",
                "128x100k",
                || {
                    let cuda = CudaVwmacd::new(0).unwrap();
                    let price = gen_time_major_prices(MANY_SERIES_COLS, MANY_SERIES_LEN);
                    let mut vol = gen_time_major_volumes(MANY_SERIES_COLS, MANY_SERIES_LEN);
                    for x in &mut vol {
                        if x.is_finite() {
                            *x = x.abs() * 50.0 + 5.0;
                        }
                    }
                    let params = VwmacdParams {
                        fast_period: Some(12),
                        slow_period: Some(26),
                        signal_period: Some(9),
                        fast_ma_type: Some("sma".into()),
                        slow_ma_type: Some("sma".into()),
                        signal_ma_type: Some("ema".into()),
                    };

                    struct S {
                        cuda: CudaVwmacd,
                        d_pv_tm: DeviceBuffer<f64>,
                        d_vol_tm: DeviceBuffer<f64>,
                        d_first: DeviceBuffer<i32>,
                        fast: usize,
                        slow: usize,
                        signal: usize,
                        cols: usize,
                        rows: usize,
                        d_macd: DeviceBuffer<f32>,
                        d_signal: DeviceBuffer<f32>,
                        d_hist: DeviceBuffer<f32>,
                    }
                    impl CudaBenchState for S {
                        fn launch(&mut self) {
                            self.cuda
                                .launch_many_series(
                                    &self.d_pv_tm,
                                    &self.d_vol_tm,
                                    &self.d_first,
                                    self.fast,
                                    self.slow,
                                    self.signal,
                                    self.cols,
                                    self.rows,
                                    &mut self.d_macd,
                                    &mut self.d_signal,
                                    &mut self.d_hist,
                                )
                                .expect("vwmacd many-series kernel");
                            self.cuda.synchronize().expect("vwmacd sync");
                        }
                    }

                    let first_valids = first_valids_time_major_f32(
                        &price,
                        &vol,
                        MANY_SERIES_COLS,
                        MANY_SERIES_LEN,
                    );
                    let (pv_prefix_tm, vol_prefix_tm) = compute_prefix_sums_time_major(
                        &price,
                        &vol,
                        MANY_SERIES_COLS,
                        MANY_SERIES_LEN,
                        &first_valids,
                    );
                    let d_pv_tm = DeviceBuffer::from_slice(&pv_prefix_tm).expect("d_pv_tm");
                    let d_vol_tm = DeviceBuffer::from_slice(&vol_prefix_tm).expect("d_vol_tm");
                    let d_first = DeviceBuffer::from_slice(&first_valids).expect("d_first");
                    let elems = MANY_SERIES_COLS
                        .checked_mul(MANY_SERIES_LEN)
                        .expect("elems");
                    let d_macd: DeviceBuffer<f32> =
                        unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_macd");
                    let d_signal: DeviceBuffer<f32> =
                        unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_signal");
                    let d_hist: DeviceBuffer<f32> =
                        unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_hist");
                    cuda.synchronize().expect("sync after prep");

                    Box::new(S {
                        cuda,
                        d_pv_tm,
                        d_vol_tm,
                        d_first,
                        fast: params.fast_period.unwrap_or(12),
                        slow: params.slow_period.unwrap_or(26),
                        signal: params.signal_period.unwrap_or(9),
                        cols: MANY_SERIES_COLS,
                        rows: MANY_SERIES_LEN,
                        d_macd,
                        d_signal,
                        d_hist,
                    })
                },
            )
            .with_mem_required(bytes_many_series_one_param()),
        );
        v
    }
}
