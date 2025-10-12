//! CUDA scaffolding for the Volume Weighted Moving Average (VWMA).
//!
//! The GPU implementation reuses precomputed prefix sums of price*volume and
//! volume so that each thread only performs two subtractions and one division
//! per output element. Parameter sweeps map to blocks in the Y dimension while
//! threads along X march across the time axis.

#![cfg(feature = "cuda")]

use super::DeviceArrayF32;
use crate::indicators::moving_averages::vwma::{VwmaBatchRange, VwmaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::DeviceBuffer;
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::sys as cu;
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaVwmaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaVwmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaVwmaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaVwmaError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}

impl std::error::Error for CudaVwmaError {}

pub struct CudaVwma {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaVwma {
    pub fn new(device_id: usize) -> Result<Self, CudaVwmaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaVwmaError::Cuda(e.to_string()))?;

        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaVwmaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaVwmaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/vwma_kernel.ptx"));
        // Prefer context-derived target and O2 JIT, with graceful fallback
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
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaVwmaError::Cuda(e.to_string()))?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaVwmaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    pub fn vwma_batch_dev(
        &self,
        prices: &[f32],
        volumes: &[f32],
        sweep: &VwmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaVwmaError> {
        let inputs = Self::prepare_batch_inputs(prices, volumes, sweep)?;
        self.run_batch_kernel(prices, volumes, &inputs)
    }

    pub fn vwma_batch_into_host_f32(
        &self,
        prices: &[f32],
        volumes: &[f32],
        sweep: &VwmaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<VwmaParams>), CudaVwmaError> {
        let inputs = Self::prepare_batch_inputs(prices, volumes, sweep)?;
        let expected = inputs.series_len * inputs.combos.len();
        if out.len() != expected {
            return Err(CudaVwmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                expected
            )));
        }

        let arr = self.run_batch_kernel(prices, volumes, &inputs)?;
        arr.buf
            .copy_to(out)
            .map_err(|e| CudaVwmaError::Cuda(e.to_string()))?;
        Ok((arr.rows, arr.cols, inputs.combos))
    }

    pub fn vwma_batch_device(
        &self,
        d_pv_prefix: &DeviceBuffer<f64>,
        d_vol_prefix: &DeviceBuffer<f64>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVwmaError> {
        if series_len == 0 || n_combos == 0 {
            return Err(CudaVwmaError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }
        if series_len > i32::MAX as usize {
            return Err(CudaVwmaError::InvalidInput(
                "series too long for kernel argument width".into(),
            ));
        }
        if n_combos > i32::MAX as usize {
            return Err(CudaVwmaError::InvalidInput(
                "too many parameter combinations".into(),
            ));
        }

        self.launch_batch_kernel(
            d_pv_prefix,
            d_vol_prefix,
            d_periods,
            series_len,
            n_combos,
            first_valid,
            d_out,
        )
    }

    pub fn vwma_many_series_one_param_device(
        &self,
        d_pv_prefix_tm: &DeviceBuffer<f64>,
        d_vol_prefix_tm: &DeviceBuffer<f64>,
        period: usize,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVwmaError> {
        if period == 0 || num_series == 0 || series_len == 0 {
            return Err(CudaVwmaError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        if period > i32::MAX as usize
            || num_series > i32::MAX as usize
            || series_len > i32::MAX as usize
        {
            return Err(CudaVwmaError::InvalidInput(
                "arguments exceed kernel limits".into(),
            ));
        }

        self.launch_many_series_kernel(
            d_pv_prefix_tm,
            d_vol_prefix_tm,
            period,
            num_series,
            series_len,
            d_first_valids,
            d_out_tm,
        )
    }

    pub fn vwma_many_series_one_param_time_major_dev(
        &self,
        prices_tm_f32: &[f32],
        volumes_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaVwmaError> {
        let inputs =
            Self::prepare_many_series_inputs(prices_tm_f32, volumes_tm_f32, cols, rows, period)?;
        self.run_many_series_kernel(prices_tm_f32, volumes_tm_f32, cols, rows, period, &inputs)
    }

    pub fn vwma_many_series_one_param_time_major_into_host_f32(
        &self,
        prices_tm_f32: &[f32],
        volumes_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
        out_tm: &mut [f32],
    ) -> Result<(), CudaVwmaError> {
        if out_tm.len() != cols * rows {
            return Err(CudaVwmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out_tm.len(),
                cols * rows
            )));
        }

        let inputs =
            Self::prepare_many_series_inputs(prices_tm_f32, volumes_tm_f32, cols, rows, period)?;
        let arr = self.run_many_series_kernel(
            prices_tm_f32,
            volumes_tm_f32,
            cols,
            rows,
            period,
            &inputs,
        )?;
        arr.buf
            .copy_to(out_tm)
            .map_err(|e| CudaVwmaError::Cuda(e.to_string()))
    }

    fn run_batch_kernel(
        &self,
        prices: &[f32],
        volumes: &[f32],
        inputs: &BatchInputs,
    ) -> Result<DeviceArrayF32, CudaVwmaError> {
        let n_combos = inputs.combos.len();
        let series_len = inputs.series_len;
        let first_valid = inputs.first_valid;

        let (pv_prefix, vol_prefix) = compute_prefix_sums(prices, volumes, first_valid, series_len);

        let pv_bytes = pv_prefix.len() * std::mem::size_of::<f64>();
        let vol_bytes = vol_prefix.len() * std::mem::size_of::<f64>();
        let period_bytes = n_combos * std::mem::size_of::<i32>();
        let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
        let required = pv_bytes + vol_bytes + period_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaVwmaError::InvalidInput(
                "not enough device memory for VWMA batch".into(),
            ));
        }

        let d_pv =
            DeviceBuffer::from_slice(&pv_prefix).map_err(|e| CudaVwmaError::Cuda(e.to_string()))?;
        let d_vol = DeviceBuffer::from_slice(&vol_prefix)
            .map_err(|e| CudaVwmaError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&inputs.periods)
            .map_err(|e| CudaVwmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaVwmaError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_pv,
            &d_vol,
            &d_periods,
            series_len,
            n_combos,
            first_valid,
            &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaVwmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    fn launch_batch_kernel(
        &self,
        d_pv_prefix: &DeviceBuffer<f64>,
        d_vol_prefix: &DeviceBuffer<f64>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVwmaError> {
        if series_len > i32::MAX as usize {
            return Err(CudaVwmaError::InvalidInput(
                "series too long for kernel argument width".into(),
            ));
        }
        if n_combos > i32::MAX as usize {
            return Err(CudaVwmaError::InvalidInput(
                "too many parameter combinations".into(),
            ));
        }

        let func = self
            .module
            .get_function("vwma_batch_f32")
            .map_err(|e| CudaVwmaError::Cuda(e.to_string()))?;

        const MAX_GRID_Y: usize = 65_535;
        let block_x: u32 = 256;
        let grid_x = ((series_len as u32) + block_x - 1) / block_x;
        let block: BlockSize = (block_x, 1, 1).into();

        let total_out_elems = (n_combos as u64) * (series_len as u64);
        let out_base = d_out.as_device_ptr().as_raw();
        let periods_base = d_periods.as_device_ptr().as_raw();
        let pv_base = d_pv_prefix.as_device_ptr().as_raw();
        let vol_base = d_vol_prefix.as_device_ptr().as_raw();

        let mut start = 0usize;
        while start < n_combos {
            let chunk = (n_combos - start).min(MAX_GRID_Y);
            let grid: GridSize = (grid_x, chunk as u32, 1).into();

            unsafe {
                let mut pv_ptr = pv_base;
                let mut vol_ptr = vol_base;
                let mut periods_ptr = periods_base
                    .saturating_add((start as u64) * (std::mem::size_of::<i32>() as u64));
                let mut series_len_i = series_len as i32;
                let mut combos_i = chunk as i32;
                let mut first_valid_i = first_valid as i32;
                let mut out_ptr = out_base.saturating_add(
                    (start as u64) * (series_len as u64) * (std::mem::size_of::<f32>() as u64),
                );
                let args: &mut [*mut c_void] = &mut [
                    &mut pv_ptr as *mut _ as *mut c_void,
                    &mut vol_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut combos_i as *mut _ as *mut c_void,
                    &mut first_valid_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaVwmaError::Cuda(e.to_string()))?;
            }
            start += chunk;
        }
        Ok(())
    }

    fn launch_many_series_kernel(
        &self,
        d_pv_prefix_tm: &DeviceBuffer<f64>,
        d_vol_prefix_tm: &DeviceBuffer<f64>,
        period: usize,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVwmaError> {
        let func = self
            .module
            .get_function("vwma_multi_series_one_param_f32")
            .map_err(|e| CudaVwmaError::Cuda(e.to_string()))?;

        let block_x: u32 = 128;
        let grid_x = ((series_len as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x, num_series as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut pv_ptr = d_pv_prefix_tm.as_device_ptr().as_raw();
            let mut vol_ptr = d_vol_prefix_tm.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut pv_ptr as *mut _ as *mut c_void,
                &mut vol_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_valids_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaVwmaError::Cuda(e.to_string()))?
        }
        Ok(())
    }

    fn run_many_series_kernel(
        &self,
        prices_tm_f32: &[f32],
        volumes_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
        inputs: &ManySeriesPrepared,
    ) -> Result<DeviceArrayF32, CudaVwmaError> {
        let series_len = rows;
        let num_series = cols;

        let pv_bytes = inputs.pv_prefix_tm.len() * std::mem::size_of::<f64>();
        let vol_bytes = inputs.vol_prefix_tm.len() * std::mem::size_of::<f64>();
        let first_valid_bytes = inputs.first_valids.len() * std::mem::size_of::<i32>();
        let out_bytes = prices_tm_f32.len() * std::mem::size_of::<f32>();
        let required = pv_bytes + vol_bytes + first_valid_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaVwmaError::InvalidInput(
                "not enough device memory for VWMA many-series".into(),
            ));
        }

        let d_pv = DeviceBuffer::from_slice(&inputs.pv_prefix_tm)
            .map_err(|e| CudaVwmaError::Cuda(e.to_string()))?;
        let d_vol = DeviceBuffer::from_slice(&inputs.vol_prefix_tm)
            .map_err(|e| CudaVwmaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&inputs.first_valids)
            .map_err(|e| CudaVwmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(prices_tm_f32.len()) }
                .map_err(|e| CudaVwmaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_pv,
            &d_vol,
            period,
            num_series,
            series_len,
            &d_first_valids,
            &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaVwmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: series_len,
            cols: num_series,
        })
    }

    fn prepare_many_series_inputs(
        prices_tm_f32: &[f32],
        volumes_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<ManySeriesPrepared, CudaVwmaError> {
        if prices_tm_f32.len() != volumes_tm_f32.len() {
            return Err(CudaVwmaError::InvalidInput(
                "price/volume matrix length mismatch".into(),
            ));
        }
        if cols == 0 || rows == 0 {
            return Err(CudaVwmaError::InvalidInput(
                "matrix dimensions must be positive".into(),
            ));
        }
        if prices_tm_f32.len() != cols * rows {
            return Err(CudaVwmaError::InvalidInput("matrix shape mismatch".into()));
        }
        if period == 0 {
            return Err(CudaVwmaError::InvalidInput(
                "period must be positive".into(),
            ));
        }

        let mut first_valids = vec![0i32; cols];
        for series_idx in 0..cols {
            let mut fv = None;
            for row in 0..rows {
                let idx = row * cols + series_idx;
                let p = prices_tm_f32[idx];
                let v = volumes_tm_f32[idx];
                if !p.is_nan() && !v.is_nan() {
                    fv = Some(row);
                    break;
                }
            }
            let val = fv.ok_or_else(|| {
                CudaVwmaError::InvalidInput(format!(
                    "series {} has all NaN price/volume pairs",
                    series_idx
                ))
            })?;
            if rows - val < period {
                return Err(CudaVwmaError::InvalidInput(format!(
                    "series {} lacks data: needed >= {}, valid = {}",
                    series_idx,
                    period,
                    rows - val
                )));
            }
            first_valids[series_idx] = val as i32;
        }

        let (pv_prefix_tm, vol_prefix_tm) = compute_prefix_sums_time_major(
            prices_tm_f32,
            volumes_tm_f32,
            cols,
            rows,
            &first_valids,
        );

        Ok(ManySeriesPrepared {
            first_valids,
            pv_prefix_tm,
            vol_prefix_tm,
        })
    }

    fn prepare_batch_inputs(
        prices: &[f32],
        volumes: &[f32],
        sweep: &VwmaBatchRange,
    ) -> Result<BatchInputs, CudaVwmaError> {
        if prices.is_empty() {
            return Err(CudaVwmaError::InvalidInput("empty prices".into()));
        }
        if prices.len() != volumes.len() {
            return Err(CudaVwmaError::InvalidInput(format!(
                "price/volume length mismatch: {} vs {}",
                prices.len(),
                volumes.len()
            )));
        }

        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaVwmaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let series_len = prices.len();
        let first_valid = prices
            .iter()
            .zip(volumes.iter())
            .position(|(&p, &v)| !p.is_nan() && !v.is_nan())
            .ok_or_else(|| CudaVwmaError::InvalidInput("all price/volume pairs are NaN".into()))?;

        let max_period = combos
            .iter()
            .map(|c| c.period.unwrap_or(0))
            .max()
            .unwrap_or(0);
        if max_period == 0 {
            return Err(CudaVwmaError::InvalidInput(
                "invalid period (zero) in sweep".into(),
            ));
        }
        if series_len - first_valid < max_period {
            return Err(CudaVwmaError::InvalidInput(format!(
                "not enough valid data (needed >= {}, valid = {})",
                max_period,
                series_len - first_valid
            )));
        }

        let mut periods = Vec::with_capacity(combos.len());
        for prm in &combos {
            let period = prm.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaVwmaError::InvalidInput(
                    "period must be positive".into(),
                ));
            }
            if period > i32::MAX as usize {
                return Err(CudaVwmaError::InvalidInput(
                    "period too large for kernel argument".into(),
                ));
            }
            periods.push(period as i32);
        }

        Ok(BatchInputs {
            combos,
            periods,
            first_valid,
            series_len,
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
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices, gen_time_major_volumes};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;
    const MANY_SERIES_COLS: usize = 250;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = 2 * ONE_SERIES_LEN * std::mem::size_of::<f32>(); // price + volume
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = 2 * elems * std::mem::size_of::<f32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct VwmaBatchState {
        cuda: CudaVwma,
        price: Vec<f32>,
        volume: Vec<f32>,
        sweep: VwmaBatchRange,
    }
    impl CudaBenchState for VwmaBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .vwma_batch_dev(&self.price, &self.volume, &self.sweep)
                .expect("vwma batch launch");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaVwma::new(0).expect("cuda vwma");
        let price = gen_series(ONE_SERIES_LEN);
        let volume = gen_series(ONE_SERIES_LEN)
            .into_iter()
            .map(|v| {
                if v.is_nan() {
                    v
                } else {
                    (v.abs() + 1.0) * 500.0
                }
            })
            .collect::<Vec<f32>>();
        let sweep = VwmaBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
        };
        Box::new(VwmaBatchState {
            cuda,
            price,
            volume,
            sweep,
        })
    }

    struct VwmaManyState {
        cuda: CudaVwma,
        price_tm: Vec<f32>,
        vol_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        period: usize,
    }
    impl CudaBenchState for VwmaManyState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .vwma_many_series_one_param_time_major_dev(
                    &self.price_tm,
                    &self.vol_tm,
                    self.cols,
                    self.rows,
                    self.period,
                )
                .expect("vwma many-series launch");
        }
    }
    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaVwma::new(0).expect("cuda vwma");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let price_tm = gen_time_major_prices(cols, rows);
        let vol_tm = gen_time_major_volumes(cols, rows);
        Box::new(VwmaManyState {
            cuda,
            price_tm,
            vol_tm,
            cols,
            rows,
            period: 64,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "vwma",
                "one_series_many_params",
                "vwma_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "vwma",
                "many_series_one_param",
                "vwma_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}

struct BatchInputs {
    combos: Vec<VwmaParams>,
    periods: Vec<i32>,
    first_valid: usize,
    series_len: usize,
}

struct ManySeriesPrepared {
    first_valids: Vec<i32>,
    pv_prefix_tm: Vec<f64>,
    vol_prefix_tm: Vec<f64>,
}

fn compute_prefix_sums(
    prices: &[f32],
    volumes: &[f32],
    first_valid: usize,
    series_len: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut pv_prefix = vec![0f64; series_len];
    let mut vol_prefix = vec![0f64; series_len];
    // Accumulate in f64 for better numerical agreement with CPU f64 path
    let mut acc_pv = 0f64;
    let mut acc_vol = 0f64;

    for i in first_valid..series_len {
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

    for series_idx in 0..cols {
        let fv = first_valids[series_idx].max(0) as usize;
        // Accumulate in f64 for better agreement
        let mut acc_pv = 0f64;
        let mut acc_vol = 0f64;
        for row in 0..rows {
            let idx = row * cols + series_idx;
            if row >= fv {
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

fn expand_grid(r: &VwmaBatchRange) -> Vec<VwmaParams> {
    let (start, end, step) = r.period;
    if step == 0 || start == end {
        return vec![VwmaParams {
            period: Some(start),
        }];
    }

    (start..=end)
        .step_by(step)
        .map(|p| VwmaParams { period: Some(p) })
        .collect()
}
