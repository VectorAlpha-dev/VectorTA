#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::rogers_satchell_volatility::{
    RogersSatchellVolatilityBatchRange, RogersSatchellVolatilityParams,
};
use cust::context::Context;
use cust::device::Device;
use cust::memory::{CopyDestination, DeviceBuffer};
use cust::prelude::*;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaRogersSatchellVolatilityError {
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
}

pub struct DeviceArrayF32Pair {
    pub rs: DeviceArrayF32,
    pub signal: DeviceArrayF32,
}

pub struct CudaRogersSatchellBatchResult {
    pub outputs: DeviceArrayF32Pair,
    pub combos: Vec<RogersSatchellVolatilityParams>,
}

pub struct CudaRogersSatchellManySeriesResult {
    pub rs: DeviceArrayF32,
    pub signal: DeviceArrayF32,
}

pub struct CudaRogersSatchellVolatility {
    _context: Arc<Context>,
    device_id: u32,
}

impl CudaRogersSatchellVolatility {
    pub fn new(device_id: usize) -> Result<Self, CudaRogersSatchellVolatilityError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        Ok(Self {
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

    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, CudaRogersSatchellVolatilityError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let step = step.max(1);
        if start < end {
            let mut values = Vec::new();
            let mut current = start;
            while current <= end {
                values.push(current);
                match current.checked_add(step) {
                    Some(next) if next != current => current = next,
                    _ => break,
                }
            }
            if values.is_empty() {
                return Err(CudaRogersSatchellVolatilityError::InvalidRange {
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                });
            }
            Ok(values)
        } else {
            let mut values = Vec::new();
            let mut current = start;
            loop {
                values.push(current);
                if current == end {
                    break;
                }
                let next = current.saturating_sub(step);
                if next == current || next < end {
                    break;
                }
                current = next;
            }
            if values.is_empty() {
                return Err(CudaRogersSatchellVolatilityError::InvalidRange {
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                });
            }
            Ok(values)
        }
    }

    fn expand_grid(
        sweep: &RogersSatchellVolatilityBatchRange,
    ) -> Result<Vec<RogersSatchellVolatilityParams>, CudaRogersSatchellVolatilityError> {
        let lookbacks = Self::axis_usize(sweep.lookback)?;
        let signal_lengths = Self::axis_usize(sweep.signal_length)?;
        let mut combos = Vec::with_capacity(lookbacks.len() * signal_lengths.len());
        for &lookback in &lookbacks {
            for &signal_length in &signal_lengths {
                combos.push(RogersSatchellVolatilityParams {
                    lookback: Some(lookback),
                    signal_length: Some(signal_length),
                });
            }
        }
        Ok(combos)
    }

    fn compute_signal_row(rs: &[f32], signal_length: usize) -> Vec<f32> {
        let len = rs.len();
        let mut out = vec![f32::NAN; len];
        if signal_length == 0 || len == 0 {
            return out;
        }
        let mut sum = 0.0f64;
        let mut valid = 0usize;
        for i in 0..len {
            let value = rs[i];
            if value.is_finite() {
                sum += value as f64;
                valid += 1;
            }
            if i >= signal_length {
                let old = rs[i - signal_length];
                if old.is_finite() {
                    sum -= old as f64;
                    valid -= 1;
                }
            }
            if i + 1 >= signal_length && valid == signal_length {
                out[i] = (sum / signal_length as f64) as f32;
            }
        }
        out
    }

    #[inline]
    fn rs_term(open: f32, high: f32, low: f32, close: f32) -> Option<f64> {
        if !(open.is_finite() && high.is_finite() && low.is_finite() && close.is_finite()) {
            return None;
        }
        if open <= 0.0 || high <= 0.0 || low <= 0.0 || close <= 0.0 {
            return None;
        }
        Some(
            ((high / close).ln() as f64) * ((high / open).ln() as f64)
                + ((low / close).ln() as f64) * ((low / open).ln() as f64),
        )
    }

    fn compute_rs_row(
        open: &[f32],
        high: &[f32],
        low: &[f32],
        close: &[f32],
        lookback: usize,
    ) -> Vec<f32> {
        let len = close.len();
        let mut out = vec![f32::NAN; len];
        let mut prefix_valid = vec![0usize; len + 1];
        let mut prefix_sum = vec![0.0f64; len + 1];

        for i in 0..len {
            prefix_valid[i + 1] = prefix_valid[i];
            prefix_sum[i + 1] = prefix_sum[i];
            if let Some(term) = Self::rs_term(open[i], high[i], low[i], close[i]) {
                prefix_valid[i + 1] += 1;
                prefix_sum[i + 1] += term;
            }
        }

        let warm = lookback.saturating_sub(1).min(len);
        for t in warm..len {
            let end = t + 1;
            let start = end - lookback;
            if prefix_valid[end] - prefix_valid[start] == lookback {
                let mut variance = (prefix_sum[end] - prefix_sum[start]) / lookback as f64;
                if variance < 0.0 {
                    variance = 0.0;
                }
                out[t] = variance.sqrt() as f32;
            }
        }
        out
    }

    fn compute_batch_host(
        open: &[f32],
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &RogersSatchellVolatilityBatchRange,
    ) -> Result<
        (Vec<f32>, Vec<f32>, Vec<RogersSatchellVolatilityParams>),
        CudaRogersSatchellVolatilityError,
    > {
        let combos = Self::expand_grid(sweep)?;
        let cols = close.len();
        let total = combos.len().checked_mul(cols).ok_or_else(|| {
            CudaRogersSatchellVolatilityError::InvalidInput("rows*cols overflow".to_string())
        })?;
        let mut rs_out = vec![f32::NAN; total];
        let mut signal_out = vec![f32::NAN; total];

        for (row, combo) in combos.iter().enumerate() {
            let lookback = combo.lookback.unwrap_or(8);
            let signal_length = combo.signal_length.unwrap_or(8);
            let rs = Self::compute_rs_row(open, high, low, close, lookback);
            let signal = Self::compute_signal_row(&rs, signal_length);
            let rs_dst = &mut rs_out[row * cols..(row + 1) * cols];
            let signal_dst = &mut signal_out[row * cols..(row + 1) * cols];
            rs_dst.copy_from_slice(&rs);
            signal_dst.copy_from_slice(&signal);
        }

        Ok((rs_out, signal_out, combos))
    }

    pub fn rogers_satchell_volatility_batch_dev(
        &self,
        open: &[f32],
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &RogersSatchellVolatilityBatchRange,
    ) -> Result<CudaRogersSatchellBatchResult, CudaRogersSatchellVolatilityError> {
        if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
            return Err(CudaRogersSatchellVolatilityError::InvalidInput(
                "OHLC slice length mismatch".to_string(),
            ));
        }

        let cols = close.len();
        let (host_rs, host_signal, combos) =
            Self::compute_batch_host(open, high, low, close, sweep)?;
        let d_rs = DeviceBuffer::from_slice(&host_rs)?;
        let d_signal = DeviceBuffer::from_slice(&host_signal)?;

        Ok(CudaRogersSatchellBatchResult {
            outputs: DeviceArrayF32Pair {
                rs: DeviceArrayF32 {
                    buf: d_rs,
                    rows: combos.len(),
                    cols,
                },
                signal: DeviceArrayF32 {
                    buf: d_signal,
                    rows: combos.len(),
                    cols,
                },
            },
            combos,
        })
    }

    pub fn rogers_satchell_volatility_batch_from_device(
        &self,
        open: &DeviceBuffer<f32>,
        high: &DeviceBuffer<f32>,
        low: &DeviceBuffer<f32>,
        close: &DeviceBuffer<f32>,
        _first_valid: usize,
        sweep: &RogersSatchellVolatilityBatchRange,
    ) -> Result<CudaRogersSatchellBatchResult, CudaRogersSatchellVolatilityError> {
        if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
            return Err(CudaRogersSatchellVolatilityError::InvalidInput(
                "device OHLC length mismatch".to_string(),
            ));
        }
        let mut host_open = vec![0f32; open.len()];
        let mut host_high = vec![0f32; high.len()];
        let mut host_low = vec![0f32; low.len()];
        let mut host_close = vec![0f32; close.len()];
        open.copy_to(&mut host_open)?;
        high.copy_to(&mut host_high)?;
        low.copy_to(&mut host_low)?;
        close.copy_to(&mut host_close)?;

        let cols = host_close.len();
        let (host_rs, host_signal, combos) =
            Self::compute_batch_host(&host_open, &host_high, &host_low, &host_close, sweep)?;
        let d_rs = DeviceBuffer::from_slice(&host_rs)?;
        let d_signal = DeviceBuffer::from_slice(&host_signal)?;

        Ok(CudaRogersSatchellBatchResult {
            outputs: DeviceArrayF32Pair {
                rs: DeviceArrayF32 {
                    buf: d_rs,
                    rows: combos.len(),
                    cols,
                },
                signal: DeviceArrayF32 {
                    buf: d_signal,
                    rows: combos.len(),
                    cols,
                },
            },
            combos,
        })
    }

    pub fn rogers_satchell_volatility_many_series_one_param_time_major_dev(
        &self,
        open_tm: &[f32],
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        lookback: usize,
        signal_length: usize,
    ) -> Result<CudaRogersSatchellManySeriesResult, CudaRogersSatchellVolatilityError> {
        if open_tm.len() != high_tm.len()
            || open_tm.len() != low_tm.len()
            || open_tm.len() != close_tm.len()
            || open_tm.len() != cols.saturating_mul(rows)
        {
            return Err(CudaRogersSatchellVolatilityError::InvalidInput(
                "time-major OHLC shape mismatch".to_string(),
            ));
        }

        let mut host_rs = vec![f32::NAN; open_tm.len()];
        let mut host_signal = vec![f32::NAN; open_tm.len()];
        for series in 0..cols {
            let mut open = vec![0f32; rows];
            let mut high = vec![0f32; rows];
            let mut low = vec![0f32; rows];
            let mut close = vec![0f32; rows];
            for t in 0..rows {
                let idx = t * cols + series;
                open[t] = open_tm[idx];
                high[t] = high_tm[idx];
                low[t] = low_tm[idx];
                close[t] = close_tm[idx];
            }
            let rs = Self::compute_rs_row(&open, &high, &low, &close, lookback);
            let signal = Self::compute_signal_row(&rs, signal_length);
            for t in 0..rows {
                let idx = t * cols + series;
                host_rs[idx] = rs[t];
                host_signal[idx] = signal[t];
            }
        }
        let d_rs = DeviceBuffer::from_slice(&host_rs)?;
        let d_signal = DeviceBuffer::from_slice(&host_signal)?;

        Ok(CudaRogersSatchellManySeriesResult {
            rs: DeviceArrayF32 {
                buf: d_rs,
                rows,
                cols,
            },
            signal: DeviceArrayF32 {
                buf: d_signal,
                rows,
                cols,
            },
        })
    }
}
