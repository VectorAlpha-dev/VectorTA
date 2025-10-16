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
use cust::memory::DeviceBuffer;
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::sys as cu;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaWillrError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaWillrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaWillrError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaWillrError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}

impl std::error::Error for CudaWillrError {}

pub struct CudaWillr {
    module: Module,
    stream: Stream,
    _context: Context,
}

struct PreparedWillrBatch {
    combos: Vec<WillrParams>,
    first_valid: usize,
    series_len: usize,
    tables: WillrGpuTables,
}

impl CudaWillr {
    pub fn new(device_id: usize) -> Result<Self, CudaWillrError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaWillrError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaWillrError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaWillrError::Cuda(e.to_string()))?;

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/willr_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => Module::from_ptx(ptx, &[]).map_err(|e| CudaWillrError::Cuda(e.to_string()))?,
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaWillrError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }

    fn device_mem_info() -> Option<(usize, usize)> {
        unsafe {
            let mut free: usize = 0;
            let mut total: usize = 0;
            let res = cu::cuMemGetInfo_v2(&mut free as *mut usize, &mut total as *mut usize);
            if res == cu::CUresult::CUDA_SUCCESS { Some((free, total)) } else { None }
        }
    }

    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Some((free, _)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
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

        let d_close =
            DeviceBuffer::from_slice(close_f32).map_err(|e| CudaWillrError::Cuda(e.to_string()))?;
        let d_periods =
            DeviceBuffer::from_slice(&periods).map_err(|e| CudaWillrError::Cuda(e.to_string()))?;
        let d_log2 = DeviceBuffer::from_slice(&prepared.tables.log2)
            .map_err(|e| CudaWillrError::Cuda(e.to_string()))?;
        let d_offsets = DeviceBuffer::from_slice(&prepared.tables.level_offsets)
            .map_err(|e| CudaWillrError::Cuda(e.to_string()))?;
        let d_st_max = DeviceBuffer::from_slice(&prepared.tables.st_max)
            .map_err(|e| CudaWillrError::Cuda(e.to_string()))?;
        let d_st_min = DeviceBuffer::from_slice(&prepared.tables.st_min)
            .map_err(|e| CudaWillrError::Cuda(e.to_string()))?;
        let d_nan_psum = DeviceBuffer::from_slice(&prepared.tables.nan_psum)
            .map_err(|e| CudaWillrError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(prepared.series_len * n_combos)
                .map_err(|e| CudaWillrError::Cuda(e.to_string()))?
        };

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

        let combos = expand_periods(sweep);
        if combos.is_empty() {
            return Err(CudaWillrError::InvalidInput(
                "no period combinations".into(),
            ));
        }

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
        if n_combos == 0 {
            return Ok(());
        }

        let func = self
            .module
            .get_function("willr_batch_f32")
            .map_err(|e| CudaWillrError::Cuda(e.to_string()))?;

        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (256, 1, 1).into();

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

            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaWillrError::Cuda(e.to_string()))?;
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
        let in_bytes = 3 * elems * std::mem::size_of::<f32>();
        let first_bytes = cols * std::mem::size_of::<i32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        let required = in_bytes + first_bytes + out_bytes;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaWillrError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                required as f64 / (1024.0 * 1024.0)
            )));
        }

        let d_high = DeviceBuffer::from_slice(high_tm)
            .map_err(|e| CudaWillrError::Cuda(e.to_string()))?;
        let d_low = DeviceBuffer::from_slice(low_tm)
            .map_err(|e| CudaWillrError::Cuda(e.to_string()))?;
        let d_close = DeviceBuffer::from_slice(close_tm)
            .map_err(|e| CudaWillrError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaWillrError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(|e| CudaWillrError::Cuda(e.to_string()))?;

        self.willr_many_series_one_param_device(
            &d_high, &d_low, &d_close, cols as i32, rows as i32, period as i32, &d_first, &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaWillrError::Cuda(e.to_string()))?;

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
            return Err(CudaWillrError::InvalidInput("cols and rows must be > 0".into()));
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
            .map_err(|e| CudaWillrError::Cuda(e.to_string()))?;

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
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaWillrError::Cuda(e.to_string()))?;
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

fn expand_periods(range: &WillrBatchRange) -> Vec<WillrParams> {
    let (start, end, step) = range.period;
    if step == 0 || start == end {
        return vec![WillrParams {
            period: Some(start),
        }];
    }
    let mut periods = Vec::new();
    let mut value = start;
    while value <= end {
        periods.push(WillrParams {
            period: Some(value),
        });
        value = value.saturating_add(step);
    }
    periods
}
