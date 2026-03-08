#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::yang_zhang_volatility::{
    YangZhangVolatilityBatchRange, YangZhangVolatilityParams,
};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::mem::size_of;
use std::sync::Arc;
use thiserror::Error;

const PREP_BLOCK_X: u32 = 256;
const BATCH_BLOCK_X: u32 = 256;
const MANY_SERIES_BLOCK_X: u32 = 256;
const MAX_GRID_Y: usize = 65_535;

#[derive(Debug, Error)]
pub enum CudaYangZhangVolatilityError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
}

pub struct DeviceArrayF32Pair {
    pub yz: DeviceArrayF32,
    pub rs: DeviceArrayF32,
}

impl DeviceArrayF32Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.yz.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.yz.cols
    }
}

pub struct CudaYangZhangBatchResult {
    pub outputs: DeviceArrayF32Pair,
    pub combos: Vec<YangZhangVolatilityParams>,
}

pub struct CudaYangZhangPreparedSeries {
    d_prefix_valid: DeviceBuffer<i32>,
    d_prefix_rs: DeviceBuffer<f32>,
    d_prefix_o: DeviceBuffer<f32>,
    d_prefix_oo: DeviceBuffer<f32>,
    d_prefix_c: DeviceBuffer<f32>,
    d_prefix_cc: DeviceBuffer<f32>,
    series_len: usize,
    first_valid: usize,
}

impl CudaYangZhangPreparedSeries {
    #[inline]
    pub fn series_len(&self) -> usize {
        self.series_len
    }

    #[inline]
    pub fn first_valid(&self) -> usize {
        self.first_valid
    }
}

pub struct CudaYangZhangPreparedBatch {
    pub outputs: DeviceArrayF32Pair,
    pub combos: Vec<YangZhangVolatilityParams>,
    d_lookbacks: DeviceBuffer<i32>,
    d_k_overrides: DeviceBuffer<i32>,
    d_k_values: DeviceBuffer<f32>,
}

impl CudaYangZhangPreparedBatch {
    #[inline]
    pub fn rows(&self) -> usize {
        self.outputs.rows()
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.outputs.cols()
    }
}

pub struct CudaYangZhangVolatility {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
}

impl CudaYangZhangVolatility {
    pub fn new(device_id: usize) -> Result<Self, CudaYangZhangVolatilityError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(
            env!("OUT_DIR"),
            "/yang_zhang_volatility_kernel.ptx"
        ));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
        })
    }

    #[inline]
    pub fn context_arc_clone(&self) -> Arc<Context> {
        self._context.clone()
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    pub fn synchronize(&self) -> Result<(), CudaYangZhangVolatilityError> {
        self.stream.synchronize().map_err(Into::into)
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }

    #[inline]
    fn bytes_for<T>(elems: usize) -> Result<usize, CudaYangZhangVolatilityError> {
        elems.checked_mul(size_of::<T>()).ok_or_else(|| {
            CudaYangZhangVolatilityError::InvalidInput("byte size overflow".to_string())
        })
    }

    #[inline]
    fn grid_x_for_len(len: usize, block_x: u32) -> u32 {
        let blocks = ((len as u64) + (block_x as u64) - 1) / (block_x as u64);
        blocks.max(1).min(u32::MAX as u64) as u32
    }

    #[inline]
    fn default_headroom() -> usize {
        env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024)
    }

    #[inline]
    fn will_fit(bytes_needed: usize, headroom: usize) -> Result<(), CudaYangZhangVolatilityError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Ok((free, _)) = mem_get_info() {
            if bytes_needed.saturating_add(headroom) <= free {
                Ok(())
            } else {
                Err(CudaYangZhangVolatilityError::OutOfMemory {
                    required: bytes_needed,
                    free,
                    headroom,
                })
            }
        } else {
            Ok(())
        }
    }

    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, CudaYangZhangVolatilityError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let st = step.max(1);
        if start < end {
            let mut v = Vec::new();
            let mut x = start;
            while x <= end {
                v.push(x);
                match x.checked_add(st) {
                    Some(next) if next != x => x = next,
                    _ => break,
                }
            }
            if v.is_empty() {
                return Err(CudaYangZhangVolatilityError::InvalidRange {
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                });
            }
            Ok(v)
        } else {
            let mut v = Vec::new();
            let mut x = start;
            loop {
                v.push(x);
                if x == end {
                    break;
                }
                let next = x.saturating_sub(st);
                if next == x || next < end {
                    break;
                }
                x = next;
            }
            if v.is_empty() {
                return Err(CudaYangZhangVolatilityError::InvalidRange {
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                });
            }
            Ok(v)
        }
    }

    fn axis_f64(
        (start, end, step): (f64, f64, f64),
    ) -> Result<Vec<f64>, CudaYangZhangVolatilityError> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }
        let st = step.abs();
        if !st.is_finite() {
            return Err(CudaYangZhangVolatilityError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        if start < end {
            let mut v = Vec::new();
            let mut x = start;
            while x <= end + 1e-12 {
                v.push(x);
                x += st;
            }
            if v.is_empty() {
                return Err(CudaYangZhangVolatilityError::InvalidRange {
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                });
            }
            Ok(v)
        } else {
            let mut v = Vec::new();
            let mut x = start;
            while x >= end - 1e-12 {
                v.push(x);
                x -= st;
            }
            if v.is_empty() {
                return Err(CudaYangZhangVolatilityError::InvalidRange {
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                });
            }
            Ok(v)
        }
    }

    fn expand_grid_checked(
        sweep: &YangZhangVolatilityBatchRange,
    ) -> Result<Vec<YangZhangVolatilityParams>, CudaYangZhangVolatilityError> {
        let lookbacks = Self::axis_usize(sweep.lookback)?;
        let ks = if sweep.k_override {
            Self::axis_f64(sweep.k)?
        } else {
            vec![sweep.k.0]
        };
        let mut out = Vec::with_capacity(lookbacks.len().saturating_mul(ks.len()));
        for &lb in &lookbacks {
            for &k in &ks {
                out.push(YangZhangVolatilityParams {
                    lookback: Some(lb),
                    k_override: Some(sweep.k_override),
                    k: Some(k),
                });
            }
        }
        if out.is_empty() {
            return Err(CudaYangZhangVolatilityError::InvalidInput(
                "no parameter combinations".to_string(),
            ));
        }
        Ok(out)
    }

    fn find_first_valid_ohlc(
        open: &[f32],
        high: &[f32],
        low: &[f32],
        close: &[f32],
    ) -> Option<usize> {
        for i in 0..open.len() {
            let o = open[i];
            let h = high[i];
            let l = low[i];
            let c = close[i];
            if o.is_finite()
                && h.is_finite()
                && l.is_finite()
                && c.is_finite()
                && o > 0.0
                && h > 0.0
                && l > 0.0
                && c > 0.0
            {
                return Some(i);
            }
        }
        None
    }

    fn build_combo_buffers(
        sweep: &YangZhangVolatilityBatchRange,
        len: usize,
        first_valid: usize,
    ) -> Result<
        (
            Vec<YangZhangVolatilityParams>,
            Vec<i32>,
            Vec<i32>,
            Vec<f32>,
            usize,
        ),
        CudaYangZhangVolatilityError,
    > {
        let combos = Self::expand_grid_checked(sweep)?;
        let mut lookbacks_i32 = Vec::with_capacity(combos.len());
        let mut k_overrides_i32 = Vec::with_capacity(combos.len());
        let mut k_values_f32 = Vec::with_capacity(combos.len());
        let mut max_lb = 0usize;

        for combo in &combos {
            let lb = combo.lookback.unwrap_or(14);
            if lb == 0 || lb > len {
                return Err(CudaYangZhangVolatilityError::InvalidInput(format!(
                    "invalid lookback {lb} for len {len}"
                )));
            }
            if len.saturating_sub(first_valid) < lb + 1 {
                return Err(CudaYangZhangVolatilityError::InvalidInput(format!(
                    "not enough valid data for lookback {lb}"
                )));
            }
            let ko = combo.k_override.unwrap_or(false);
            let kv = combo.k.unwrap_or(0.34);
            if ko && (!kv.is_finite() || !(0.0..=1.0).contains(&kv)) {
                return Err(CudaYangZhangVolatilityError::InvalidInput(format!(
                    "invalid k value {kv}"
                )));
            }
            max_lb = max_lb.max(lb);
            lookbacks_i32.push(lb as i32);
            k_overrides_i32.push(if ko { 1 } else { 0 });
            k_values_f32.push(kv as f32);
        }

        Ok((combos, lookbacks_i32, k_overrides_i32, k_values_f32, max_lb))
    }

    fn prepared_series_bytes(series_len: usize) -> Result<usize, CudaYangZhangVolatilityError> {
        let prefix_len = series_len.checked_add(1).ok_or_else(|| {
            CudaYangZhangVolatilityError::InvalidInput("series len overflow".to_string())
        })?;
        let valid_bytes = Self::bytes_for::<i32>(prefix_len)?;
        let float_prefix_bytes = Self::bytes_for::<f32>(prefix_len)?
            .checked_mul(5)
            .ok_or_else(|| {
                CudaYangZhangVolatilityError::InvalidInput("byte size overflow".to_string())
            })?;
        valid_bytes.checked_add(float_prefix_bytes).ok_or_else(|| {
            CudaYangZhangVolatilityError::InvalidInput("byte size overflow".to_string())
        })
    }

    fn prepare_series_build_bytes(
        series_len: usize,
    ) -> Result<usize, CudaYangZhangVolatilityError> {
        let input_bytes = Self::bytes_for::<f32>(series_len)?
            .checked_mul(4)
            .ok_or_else(|| {
                CudaYangZhangVolatilityError::InvalidInput("byte size overflow".to_string())
            })?;
        let temp_bytes = Self::bytes_for::<i32>(series_len)?
            .checked_add(
                Self::bytes_for::<f32>(series_len)?
                    .checked_mul(3)
                    .ok_or_else(|| {
                        CudaYangZhangVolatilityError::InvalidInput("byte size overflow".to_string())
                    })?,
            )
            .ok_or_else(|| {
                CudaYangZhangVolatilityError::InvalidInput("byte size overflow".to_string())
            })?;
        input_bytes
            .checked_add(temp_bytes)
            .and_then(|v| v.checked_add(Self::prepared_series_bytes(series_len).ok()?))
            .ok_or_else(|| {
                CudaYangZhangVolatilityError::InvalidInput("byte size overflow".to_string())
            })
    }

    fn prepared_batch_bytes(
        n_combos: usize,
        series_len: usize,
    ) -> Result<usize, CudaYangZhangVolatilityError> {
        let out_elems = n_combos.checked_mul(series_len).ok_or_else(|| {
            CudaYangZhangVolatilityError::InvalidInput("rows*cols overflow".to_string())
        })?;
        let out_bytes = Self::bytes_for::<f32>(out_elems)?
            .checked_mul(2)
            .ok_or_else(|| {
                CudaYangZhangVolatilityError::InvalidInput("byte size overflow".to_string())
            })?;
        let param_bytes = Self::bytes_for::<i32>(n_combos)?
            .checked_add(Self::bytes_for::<i32>(n_combos)?)
            .and_then(|v| v.checked_add(Self::bytes_for::<f32>(n_combos).ok()?))
            .ok_or_else(|| {
                CudaYangZhangVolatilityError::InvalidInput("byte size overflow".to_string())
            })?;
        out_bytes.checked_add(param_bytes).ok_or_else(|| {
            CudaYangZhangVolatilityError::InvalidInput("byte size overflow".to_string())
        })
    }

    fn launch_prepare_terms_kernel(
        &self,
        d_open: &DeviceBuffer<f32>,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        series_len: usize,
        d_valid: &mut DeviceBuffer<i32>,
        d_rs_terms: &mut DeviceBuffer<f32>,
        d_oret_terms: &mut DeviceBuffer<f32>,
        d_cret_terms: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaYangZhangVolatilityError> {
        if series_len == 0 {
            return Ok(());
        }

        let func = self
            .module
            .get_function("yang_zhang_precompute_terms_f32")
            .map_err(|_| CudaYangZhangVolatilityError::MissingKernelSymbol {
                name: "yang_zhang_precompute_terms_f32",
            })?;

        let grid: GridSize = (Self::grid_x_for_len(series_len, PREP_BLOCK_X), 1u32, 1u32).into();
        let block: BlockSize = (PREP_BLOCK_X, 1u32, 1u32).into();
        let stream = &self.stream;
        unsafe {
            launch!(
                func<<<grid, block, 0, stream>>>(
                    d_open.as_device_ptr(),
                    d_high.as_device_ptr(),
                    d_low.as_device_ptr(),
                    d_close.as_device_ptr(),
                    series_len as i32,
                    d_valid.as_device_ptr(),
                    d_rs_terms.as_device_ptr(),
                    d_oret_terms.as_device_ptr(),
                    d_cret_terms.as_device_ptr()
                )
            )?;
        }

        Ok(())
    }

    fn launch_prefix_kernel(
        &self,
        d_valid: &DeviceBuffer<i32>,
        d_rs_terms: &DeviceBuffer<f32>,
        d_oret_terms: &DeviceBuffer<f32>,
        d_cret_terms: &DeviceBuffer<f32>,
        series_len: usize,
        d_prefix_valid: &mut DeviceBuffer<i32>,
        d_prefix_rs: &mut DeviceBuffer<f32>,
        d_prefix_o: &mut DeviceBuffer<f32>,
        d_prefix_oo: &mut DeviceBuffer<f32>,
        d_prefix_c: &mut DeviceBuffer<f32>,
        d_prefix_cc: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaYangZhangVolatilityError> {
        let func = self
            .module
            .get_function("yang_zhang_prefix_terms_f32")
            .map_err(|_| CudaYangZhangVolatilityError::MissingKernelSymbol {
                name: "yang_zhang_prefix_terms_f32",
            })?;

        let grid: GridSize = (1u32, 1u32, 1u32).into();
        let block: BlockSize = (1u32, 1u32, 1u32).into();
        let stream = &self.stream;
        unsafe {
            launch!(
                func<<<grid, block, 0, stream>>>(
                    d_valid.as_device_ptr(),
                    d_rs_terms.as_device_ptr(),
                    d_oret_terms.as_device_ptr(),
                    d_cret_terms.as_device_ptr(),
                    series_len as i32,
                    d_prefix_valid.as_device_ptr(),
                    d_prefix_rs.as_device_ptr(),
                    d_prefix_o.as_device_ptr(),
                    d_prefix_oo.as_device_ptr(),
                    d_prefix_c.as_device_ptr(),
                    d_prefix_cc.as_device_ptr()
                )
            )?;
        }

        Ok(())
    }

    fn launch_batch_kernel(
        &self,
        prepared_series: &CudaYangZhangPreparedSeries,
        prepared_batch: &mut CudaYangZhangPreparedBatch,
    ) -> Result<(), CudaYangZhangVolatilityError> {
        let n_combos = prepared_batch.combos.len();
        if n_combos == 0 || prepared_series.series_len == 0 {
            return Ok(());
        }

        let func = self
            .module
            .get_function("yang_zhang_volatility_batch_prefix_f32")
            .map_err(|_| CudaYangZhangVolatilityError::MissingKernelSymbol {
                name: "yang_zhang_volatility_batch_prefix_f32",
            })?;

        let block: BlockSize = (BATCH_BLOCK_X, 1u32, 1u32).into();
        let grid_x = Self::grid_x_for_len(prepared_series.series_len, BATCH_BLOCK_X);
        let mut launched = 0usize;
        while launched < n_combos {
            let chunk = (n_combos - launched).min(MAX_GRID_Y);
            let grid: GridSize = (grid_x, chunk as u32, 1u32).into();
            let stream = &self.stream;
            unsafe {
                launch!(
                    func<<<grid, block, 0, stream>>>(
                        prepared_batch.d_lookbacks.as_device_ptr().add(launched),
                        prepared_batch.d_k_overrides.as_device_ptr().add(launched),
                        prepared_batch.d_k_values.as_device_ptr().add(launched),
                        prepared_series.series_len as i32,
                        prepared_series.first_valid as i32,
                        chunk as i32,
                        prepared_series.d_prefix_valid.as_device_ptr(),
                        prepared_series.d_prefix_rs.as_device_ptr(),
                        prepared_series.d_prefix_o.as_device_ptr(),
                        prepared_series.d_prefix_oo.as_device_ptr(),
                        prepared_series.d_prefix_c.as_device_ptr(),
                        prepared_series.d_prefix_cc.as_device_ptr(),
                        prepared_batch
                            .outputs
                            .yz
                            .buf
                            .as_device_ptr()
                            .add(launched * prepared_series.series_len),
                        prepared_batch
                            .outputs
                            .rs
                            .buf
                            .as_device_ptr()
                            .add(launched * prepared_series.series_len)
                    )
                )?;
            }
            launched += chunk;
        }

        Ok(())
    }

    pub fn prepare_batch_series(
        &self,
        open: &[f32],
        high: &[f32],
        low: &[f32],
        close: &[f32],
    ) -> Result<CudaYangZhangPreparedSeries, CudaYangZhangVolatilityError> {
        if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
            return Err(CudaYangZhangVolatilityError::InvalidInput(
                "OHLC length mismatch".to_string(),
            ));
        }
        let len = close.len();
        if len == 0 {
            return Err(CudaYangZhangVolatilityError::InvalidInput(
                "empty input".to_string(),
            ));
        }

        let first_valid = Self::find_first_valid_ohlc(open, high, low, close).ok_or_else(|| {
            CudaYangZhangVolatilityError::InvalidInput("all values are invalid".to_string())
        })?;

        Self::will_fit(
            Self::prepare_series_build_bytes(len)?,
            Self::default_headroom(),
        )?;

        let d_open = unsafe { DeviceBuffer::from_slice_async(open, &self.stream) }?;
        let d_high = unsafe { DeviceBuffer::from_slice_async(high, &self.stream) }?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(low, &self.stream) }?;
        let d_close = unsafe { DeviceBuffer::from_slice_async(close, &self.stream) }?;

        let mut d_valid: DeviceBuffer<i32> =
            unsafe { DeviceBuffer::uninitialized_async(len, &self.stream) }?;
        let mut d_rs_terms: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(len, &self.stream) }?;
        let mut d_oret_terms: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(len, &self.stream) }?;
        let mut d_cret_terms: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(len, &self.stream) }?;

        let prefix_len = len.checked_add(1).ok_or_else(|| {
            CudaYangZhangVolatilityError::InvalidInput("series len overflow".to_string())
        })?;
        let mut d_prefix_valid: DeviceBuffer<i32> =
            unsafe { DeviceBuffer::uninitialized_async(prefix_len, &self.stream) }?;
        let mut d_prefix_rs: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(prefix_len, &self.stream) }?;
        let mut d_prefix_o: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(prefix_len, &self.stream) }?;
        let mut d_prefix_oo: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(prefix_len, &self.stream) }?;
        let mut d_prefix_c: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(prefix_len, &self.stream) }?;
        let mut d_prefix_cc: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(prefix_len, &self.stream) }?;

        self.launch_prepare_terms_kernel(
            &d_open,
            &d_high,
            &d_low,
            &d_close,
            len,
            &mut d_valid,
            &mut d_rs_terms,
            &mut d_oret_terms,
            &mut d_cret_terms,
        )?;
        self.launch_prefix_kernel(
            &d_valid,
            &d_rs_terms,
            &d_oret_terms,
            &d_cret_terms,
            len,
            &mut d_prefix_valid,
            &mut d_prefix_rs,
            &mut d_prefix_o,
            &mut d_prefix_oo,
            &mut d_prefix_c,
            &mut d_prefix_cc,
        )?;
        self.synchronize()?;

        Ok(CudaYangZhangPreparedSeries {
            d_prefix_valid,
            d_prefix_rs,
            d_prefix_o,
            d_prefix_oo,
            d_prefix_c,
            d_prefix_cc,
            series_len: len,
            first_valid,
        })
    }

    pub fn prepare_batch_series_device(
        &self,
        d_open: &DeviceBuffer<f32>,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        first_valid: usize,
    ) -> Result<CudaYangZhangPreparedSeries, CudaYangZhangVolatilityError> {
        let len = d_close.len();
        if len == 0 {
            return Err(CudaYangZhangVolatilityError::InvalidInput(
                "empty input".to_string(),
            ));
        }
        if d_open.len() != len || d_high.len() != len || d_low.len() != len {
            return Err(CudaYangZhangVolatilityError::InvalidInput(
                "OHLC length mismatch".to_string(),
            ));
        }
        if first_valid >= len {
            return Err(CudaYangZhangVolatilityError::InvalidInput(
                "first_valid out of range".to_string(),
            ));
        }

        Self::will_fit(
            Self::prepare_series_build_bytes(len)?,
            Self::default_headroom(),
        )?;

        let mut d_valid: DeviceBuffer<i32> =
            unsafe { DeviceBuffer::uninitialized_async(len, &self.stream) }?;
        let mut d_rs_terms: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(len, &self.stream) }?;
        let mut d_oret_terms: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(len, &self.stream) }?;
        let mut d_cret_terms: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(len, &self.stream) }?;

        let prefix_len = len.checked_add(1).ok_or_else(|| {
            CudaYangZhangVolatilityError::InvalidInput("series len overflow".to_string())
        })?;
        let mut d_prefix_valid: DeviceBuffer<i32> =
            unsafe { DeviceBuffer::uninitialized_async(prefix_len, &self.stream) }?;
        let mut d_prefix_rs: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(prefix_len, &self.stream) }?;
        let mut d_prefix_o: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(prefix_len, &self.stream) }?;
        let mut d_prefix_oo: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(prefix_len, &self.stream) }?;
        let mut d_prefix_c: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(prefix_len, &self.stream) }?;
        let mut d_prefix_cc: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(prefix_len, &self.stream) }?;

        self.launch_prepare_terms_kernel(
            d_open,
            d_high,
            d_low,
            d_close,
            len,
            &mut d_valid,
            &mut d_rs_terms,
            &mut d_oret_terms,
            &mut d_cret_terms,
        )?;
        self.launch_prefix_kernel(
            &d_valid,
            &d_rs_terms,
            &d_oret_terms,
            &d_cret_terms,
            len,
            &mut d_prefix_valid,
            &mut d_prefix_rs,
            &mut d_prefix_o,
            &mut d_prefix_oo,
            &mut d_prefix_c,
            &mut d_prefix_cc,
        )?;
        self.synchronize()?;

        Ok(CudaYangZhangPreparedSeries {
            d_prefix_valid,
            d_prefix_rs,
            d_prefix_o,
            d_prefix_oo,
            d_prefix_c,
            d_prefix_cc,
            series_len: len,
            first_valid,
        })
    }

    pub fn prepare_batch_run(
        &self,
        prepared_series: &CudaYangZhangPreparedSeries,
        sweep: &YangZhangVolatilityBatchRange,
    ) -> Result<CudaYangZhangPreparedBatch, CudaYangZhangVolatilityError> {
        let (combos, lookbacks_i32, k_overrides_i32, k_values_f32, max_lb) =
            Self::build_combo_buffers(
                sweep,
                prepared_series.series_len,
                prepared_series.first_valid,
            )?;

        if max_lb == 0
            || max_lb > prepared_series.series_len
            || prepared_series
                .series_len
                .saturating_sub(prepared_series.first_valid)
                < max_lb + 1
        {
            return Err(CudaYangZhangVolatilityError::InvalidInput(
                "not enough valid data for lookback".to_string(),
            ));
        }

        let n_combos = combos.len();
        Self::will_fit(
            Self::prepared_batch_bytes(n_combos, prepared_series.series_len)?,
            Self::default_headroom(),
        )?;

        let d_lookbacks = unsafe { DeviceBuffer::from_slice_async(&lookbacks_i32, &self.stream) }?;
        let d_k_overrides =
            unsafe { DeviceBuffer::from_slice_async(&k_overrides_i32, &self.stream) }?;
        let d_k_values = unsafe { DeviceBuffer::from_slice_async(&k_values_f32, &self.stream) }?;

        let out_elems = n_combos
            .checked_mul(prepared_series.series_len)
            .ok_or_else(|| {
                CudaYangZhangVolatilityError::InvalidInput("rows*cols overflow".to_string())
            })?;
        let d_yz: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_elems, &self.stream) }?;
        let d_rs: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_elems, &self.stream) }?;

        Ok(CudaYangZhangPreparedBatch {
            outputs: DeviceArrayF32Pair {
                yz: DeviceArrayF32 {
                    buf: d_yz,
                    rows: n_combos,
                    cols: prepared_series.series_len,
                },
                rs: DeviceArrayF32 {
                    buf: d_rs,
                    rows: n_combos,
                    cols: prepared_series.series_len,
                },
            },
            combos,
            d_lookbacks,
            d_k_overrides,
            d_k_values,
        })
    }

    pub fn launch_prepared_batch(
        &self,
        prepared_series: &CudaYangZhangPreparedSeries,
        prepared_batch: &mut CudaYangZhangPreparedBatch,
    ) -> Result<(), CudaYangZhangVolatilityError> {
        self.launch_batch_kernel(prepared_series, prepared_batch)
    }

    pub fn yang_zhang_volatility_batch_dev(
        &self,
        open: &[f32],
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &YangZhangVolatilityBatchRange,
    ) -> Result<CudaYangZhangBatchResult, CudaYangZhangVolatilityError> {
        let prepared_series = self.prepare_batch_series(open, high, low, close)?;
        let mut prepared_batch = self.prepare_batch_run(&prepared_series, sweep)?;
        self.launch_prepared_batch(&prepared_series, &mut prepared_batch)?;
        self.synchronize()?;

        Ok(CudaYangZhangBatchResult {
            outputs: prepared_batch.outputs,
            combos: prepared_batch.combos,
        })
    }

    pub fn yang_zhang_volatility_batch_from_device(
        &self,
        d_open: &DeviceBuffer<f32>,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        first_valid: usize,
        sweep: &YangZhangVolatilityBatchRange,
    ) -> Result<CudaYangZhangBatchResult, CudaYangZhangVolatilityError> {
        let prepared_series =
            self.prepare_batch_series_device(d_open, d_high, d_low, d_close, first_valid)?;
        let mut prepared_batch = self.prepare_batch_run(&prepared_series, sweep)?;
        self.launch_prepared_batch(&prepared_series, &mut prepared_batch)?;
        self.synchronize()?;

        Ok(CudaYangZhangBatchResult {
            outputs: prepared_batch.outputs,
            combos: prepared_batch.combos,
        })
    }

    pub fn yang_zhang_volatility_many_series_one_param_time_major_dev(
        &self,
        open_tm: &[f32],
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        lookback: usize,
        k_override: bool,
        k: f32,
    ) -> Result<DeviceArrayF32Pair, CudaYangZhangVolatilityError> {
        if cols == 0 || rows == 0 {
            return Err(CudaYangZhangVolatilityError::InvalidInput(
                "invalid matrix dims".to_string(),
            ));
        }
        let total = cols.checked_mul(rows).ok_or_else(|| {
            CudaYangZhangVolatilityError::InvalidInput("rows*cols overflow".to_string())
        })?;
        if open_tm.len() != total
            || high_tm.len() != total
            || low_tm.len() != total
            || close_tm.len() != total
        {
            return Err(CudaYangZhangVolatilityError::InvalidInput(
                "matrix input length mismatch".to_string(),
            ));
        }
        if lookback == 0 || lookback > rows {
            return Err(CudaYangZhangVolatilityError::InvalidInput(
                "invalid lookback".to_string(),
            ));
        }
        if k_override && (!k.is_finite() || !(0.0..=1.0).contains(&k)) {
            return Err(CudaYangZhangVolatilityError::InvalidInput(
                "invalid k override".to_string(),
            ));
        }

        let mut first_valids = vec![-1i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                let o = open_tm[idx];
                let h = high_tm[idx];
                let l = low_tm[idx];
                let c = close_tm[idx];
                if o.is_finite()
                    && h.is_finite()
                    && l.is_finite()
                    && c.is_finite()
                    && o > 0.0
                    && h > 0.0
                    && l > 0.0
                    && c > 0.0
                {
                    first_valids[s] = t as i32;
                    break;
                }
            }
        }

        let in_bytes = total
            .checked_mul(4)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                CudaYangZhangVolatilityError::InvalidInput("byte size overflow".to_string())
            })?;
        let first_bytes = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| {
                CudaYangZhangVolatilityError::InvalidInput("byte size overflow".to_string())
            })?;
        let out_bytes = total
            .checked_mul(2)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                CudaYangZhangVolatilityError::InvalidInput("byte size overflow".to_string())
            })?;
        let required = in_bytes
            .checked_add(first_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| {
                CudaYangZhangVolatilityError::InvalidInput("byte size overflow".to_string())
            })?;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        let d_open = unsafe { DeviceBuffer::from_slice_async(open_tm, &self.stream) }?;
        let d_high = unsafe { DeviceBuffer::from_slice_async(high_tm, &self.stream) }?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(low_tm, &self.stream) }?;
        let d_close = unsafe { DeviceBuffer::from_slice_async(close_tm, &self.stream) }?;
        let d_first = unsafe { DeviceBuffer::from_slice_async(&first_valids, &self.stream) }?;

        let mut d_yz: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(total, &self.stream) }?;
        let mut d_rs: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(total, &self.stream) }?;

        let func = self
            .module
            .get_function("yang_zhang_volatility_many_series_one_param_f32")
            .map_err(|_| CudaYangZhangVolatilityError::MissingKernelSymbol {
                name: "yang_zhang_volatility_many_series_one_param_f32",
            })?;
        let grid: GridSize = (cols as u32, 1u32, 1u32).into();
        let block: BlockSize = (MANY_SERIES_BLOCK_X, 1u32, 1u32).into();
        let stream = &self.stream;
        unsafe {
            launch!(
                func<<<grid, block, 0, stream>>>(
                    d_open.as_device_ptr(),
                    d_high.as_device_ptr(),
                    d_low.as_device_ptr(),
                    d_close.as_device_ptr(),
                    d_first.as_device_ptr(),
                    lookback as i32,
                    if k_override { 1 } else { 0 },
                    k,
                    cols as i32,
                    rows as i32,
                    d_yz.as_device_ptr(),
                    d_rs.as_device_ptr()
                )
            )?;
        }

        self.synchronize()?;

        Ok(DeviceArrayF32Pair {
            yz: DeviceArrayF32 {
                buf: d_yz,
                rows,
                cols,
            },
            rs: DeviceArrayF32 {
                buf: d_rs,
                rows,
                cols,
            },
        })
    }

    pub fn yang_zhang_volatility_batch_into_host_f32(
        &self,
        open: &[f32],
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &YangZhangVolatilityBatchRange,
        out_yz: &mut [f32],
        out_rs: &mut [f32],
    ) -> Result<(usize, usize, Vec<YangZhangVolatilityParams>), CudaYangZhangVolatilityError> {
        let result = self.yang_zhang_volatility_batch_dev(open, high, low, close, sweep)?;
        let rows = result.outputs.rows();
        let cols = result.outputs.cols();
        let expected = rows.checked_mul(cols).ok_or_else(|| {
            CudaYangZhangVolatilityError::InvalidInput("rows*cols overflow".to_string())
        })?;
        if out_yz.len() != expected || out_rs.len() != expected {
            return Err(CudaYangZhangVolatilityError::InvalidInput(format!(
                "output length mismatch: yz={}, rs={}, expected={expected}",
                out_yz.len(),
                out_rs.len()
            )));
        }
        result.outputs.yz.buf.copy_to(out_yz)?;
        result.outputs.rs.buf.copy_to(out_rs)?;
        Ok((rows, cols, result.combos))
    }

    pub fn yang_zhang_volatility_batch_into_pinned_host_f32(
        &self,
        open: &[f32],
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &YangZhangVolatilityBatchRange,
    ) -> Result<
        (
            LockedBuffer<f32>,
            LockedBuffer<f32>,
            usize,
            usize,
            Vec<YangZhangVolatilityParams>,
        ),
        CudaYangZhangVolatilityError,
    > {
        let result = self.yang_zhang_volatility_batch_dev(open, high, low, close, sweep)?;
        let rows = result.outputs.rows();
        let cols = result.outputs.cols();
        let total = rows.checked_mul(cols).ok_or_else(|| {
            CudaYangZhangVolatilityError::InvalidInput("rows*cols overflow".to_string())
        })?;
        let mut pinned_yz: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(total)? };
        let mut pinned_rs: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(total)? };
        unsafe {
            result
                .outputs
                .yz
                .buf
                .async_copy_to(pinned_yz.as_mut_slice(), &self.stream)?;
            result
                .outputs
                .rs
                .buf
                .async_copy_to(pinned_rs.as_mut_slice(), &self.stream)?;
        }
        self.stream.synchronize()?;
        Ok((pinned_yz, pinned_rs, rows, cols, result.combos))
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;

    fn bytes_one_series_many_params() -> usize {
        let prefix_bytes =
            (ONE_SERIES_LEN + 1).saturating_mul(size_of::<i32>() + 5 * size_of::<f32>());
        let param_bytes = PARAM_SWEEP.saturating_mul(2 * size_of::<i32>() + size_of::<f32>());
        let out_bytes = ONE_SERIES_LEN
            .saturating_mul(PARAM_SWEEP)
            .saturating_mul(2)
            .saturating_mul(size_of::<f32>());
        prefix_bytes
            .saturating_add(param_bytes)
            .saturating_add(out_bytes)
            .saturating_add(64 * 1024 * 1024)
    }

    fn gen_ohlc(len: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut open = vec![0.0f32; len];
        let mut high = vec![0.0f32; len];
        let mut low = vec![0.0f32; len];
        let mut close = vec![0.0f32; len];
        let mut prev = 1000.0f32;
        for i in 0..len {
            let x = i as f32;
            let drift = 0.0002f32 * x;
            let wave = (x * 0.0013f32).sin() * 2.0 + (x * 0.00037f32).cos() * 1.3;
            let o = (prev + drift + wave).max(1.0);
            let c = (o + (x * 0.0021f32).sin() * 0.7).max(1.0);
            let hi = o.max(c) + 0.35 + (x * 0.0011f32).cos().abs() * 0.08;
            let lo = (o.min(c) - 0.35 - (x * 0.0017f32).sin().abs() * 0.08).max(0.01);
            open[i] = o;
            high[i] = hi;
            low[i] = lo;
            close[i] = c.max(0.01);
            prev = close[i];
        }
        (open, high, low, close)
    }

    struct YangZhangBatchState {
        cuda: CudaYangZhangVolatility,
        prepared_series: CudaYangZhangPreparedSeries,
        prepared_batch: CudaYangZhangPreparedBatch,
    }

    impl CudaBenchState for YangZhangBatchState {
        fn launch(&mut self) {
            self.cuda
                .launch_prepared_batch(&self.prepared_series, &mut self.prepared_batch)
                .expect("yang_zhang_volatility batch launch");
            self.cuda.synchronize().expect("yang_zhang_volatility sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaYangZhangVolatility::new(0).expect("cuda yang_zhang_volatility");
        let (open, high, low, close) = gen_ohlc(ONE_SERIES_LEN);
        let sweep = YangZhangVolatilityBatchRange {
            lookback: (10, 10 + PARAM_SWEEP - 1, 1),
            k_override: false,
            k: (0.34, 0.34, 0.0),
        };
        let prepared_series = cuda
            .prepare_batch_series(&open, &high, &low, &close)
            .expect("prepare series");
        let prepared_batch = cuda
            .prepare_batch_run(&prepared_series, &sweep)
            .expect("prepare batch");

        Box::new(YangZhangBatchState {
            cuda,
            prepared_series,
            prepared_batch,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "yang_zhang_volatility",
            "one_series_many_params",
            "yang_zhang_volatility_cuda_batch_dev",
            "1m_x_250",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}
