//! # Volume Weighted Average Price (VWAP)
//!
//! VWAP computes the average price of a security weighted by traded volume.
//! You can customize the anchor (e.g., `1m`, `4h`, `2d`, or `1M`) for
//! aggregation periods in minutes, hours, days, or months. This implementation
//! supports flexible input, streaming, batch/grid evaluation, and SIMD feature stubs.
//!
//! ## Parameters
//! - **anchor**: Defines the grouping period (e.g., `1m`, `4h`, `1d`, `1M`). Defaults to `"1d"`.
//!
//! ## Errors
//! - **MismatchTimestampsPricesVolumes**: vwap: Mismatch in length of timestamps, prices, or volumes.
//! - **NoData**: vwap: No data available for VWAP calculation.
//! - **MismatchPricesVolumes**: vwap: Mismatch in length of prices and volumes.
//! - **ParseAnchorError**: vwap: Error parsing the anchor string.
//! - **UnsupportedAnchorUnit**: vwap: The specified anchor unit is not supported.
//! - **MonthConversionError**: vwap: Error converting timestamp to month-based anchor.
//!
//! ## Returns
//! - **`Ok(VwapOutput)`** on success, containing a `Vec<f64>` matching the input length.
//! - **`Err(VwapError)`** otherwise.
//!
//! ## Developer Notes
//! - **AVX2/AVX512 kernels**: Currently stubs calling scalar implementation
//! - **Streaming update**: O(1) - maintains running sums for current anchor period
//! - **Memory optimization**: Uses `alloc_with_nan_prefix` for zero-copy allocation
//! - **Current status**:
//!   - Scalar path optimized (pointer indexing, small unroll, fused `mul_add`).
//!   - SIMD kept as stubs via runtime selection (segmented prefix + int division limits wins).
//!   - Batch path now reuses a precomputed `pv[i] = price[i]*volume[i]` across rows.
//! - **Notes**:
//!   - Decision: keep SIMD disabled by default for VWAP; runtime selection short-circuits to scalar.
//!   - Rationale: per-element bucket IDs require integer division; segmented prefix-sum makes effective SIMD complex and data-dependent; stubs ensure correctness.

//! Decision: Streaming kernel uses boundary-only division and lazy month cutoffs; matches scalar/batch outputs and reduces per-tick overhead.

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::CudaVwap;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::vwap_wrapper::VwapDeviceArrayF32;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
use chrono::{Datelike, NaiveDate, NaiveDateTime};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

/// VWAP input data
#[derive(Debug, Clone)]
pub enum VwapData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    CandlesPlusPrices {
        candles: &'a Candles,
        prices: &'a [f64],
    },
    Slice {
        timestamps: &'a [i64],
        volumes: &'a [f64],
        prices: &'a [f64],
    },
}

/// VWAP parameters
#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct VwapParams {
    pub anchor: Option<String>,
}

impl Default for VwapParams {
    fn default() -> Self {
        Self {
            anchor: Some("1d".to_string()),
        }
    }
}

/// VWAP input
#[derive(Debug, Clone)]
pub struct VwapInput<'a> {
    pub data: VwapData<'a>,
    pub params: VwapParams,
}

impl<'a> VwapInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: VwapParams) -> Self {
        Self {
            data: VwapData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_candles_plus_prices(
        candles: &'a Candles,
        prices: &'a [f64],
        params: VwapParams,
    ) -> Self {
        Self {
            data: VwapData::CandlesPlusPrices { candles, prices },
            params,
        }
    }

    #[inline]
    pub fn from_slice(
        timestamps: &'a [i64],
        volumes: &'a [f64],
        prices: &'a [f64],
        params: VwapParams,
    ) -> Self {
        Self {
            data: VwapData::Slice {
                timestamps,
                volumes,
                prices,
            },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: VwapData::Candles {
                candles,
                source: "hlc3",
            },
            params: VwapParams::default(),
        }
    }

    #[inline]
    pub fn get_anchor(&self) -> &str {
        self.params.anchor.as_deref().unwrap_or("1d")
    }
}

/// VWAP output
#[derive(Debug, Clone)]
pub struct VwapOutput {
    pub values: Vec<f64>,
}

/// VWAP builder
#[derive(Clone, Debug)]
pub struct VwapBuilder {
    anchor: Option<String>,
    kernel: Kernel,
}

impl Default for VwapBuilder {
    fn default() -> Self {
        Self {
            anchor: None,
            kernel: Kernel::Auto,
        }
    }
}

impl VwapBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn anchor(mut self, s: impl Into<String>) -> Self {
        self.anchor = Some(s.into());
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<VwapOutput, VwapError> {
        let params = VwapParams {
            anchor: self.anchor,
        };
        let input = VwapInput::from_candles(candles, "hlc3", params);
        vwap_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        timestamps: &[i64],
        volumes: &[f64],
        prices: &[f64],
    ) -> Result<VwapOutput, VwapError> {
        let params = VwapParams {
            anchor: self.anchor,
        };
        let input = VwapInput::from_slice(timestamps, volumes, prices, params);
        vwap_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<VwapStream, VwapError> {
        let params = VwapParams {
            anchor: self.anchor,
        };
        VwapStream::try_new(params)
    }

    #[inline(always)]
    pub fn apply_candles_with_source(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<VwapOutput, VwapError> {
        let params = VwapParams {
            anchor: self.anchor,
        };
        let input = VwapInput::from_candles(candles, source, params);
        vwap_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn with_default_candles(candles: &Candles) -> Result<VwapOutput, VwapError> {
        VwapBuilder::new().apply_candles_with_source(candles, "hlc3")
    }
}

#[derive(Debug, Error)]
pub enum VwapError {
    #[error("vwap: Input data slice is empty.")]
    EmptyInputData,
    #[error("vwap: Mismatch in length of timestamps ({timestamps}), prices ({prices}), or volumes ({volumes}).")]
    MismatchTimestampsPricesVolumes {
        timestamps: usize,
        prices: usize,
        volumes: usize,
    },
    #[error("vwap: No data for VWAP calculation.")]
    NoData,
    #[error("vwap: Mismatch in length of prices ({prices}) and volumes ({volumes}).")]
    MismatchPricesVolumes { prices: usize, volumes: usize },
    #[error("vwap: Error parsing anchor: {msg}")]
    ParseAnchorError { msg: String },
    #[error("vwap: Unsupported anchor unit '{unit_char}'.")]
    UnsupportedAnchorUnit { unit_char: char },
    #[error("vwap: Error converting timestamp {ts_ms} to month-based anchor.")]
    MonthConversionError { ts_ms: i64 },
    // Enriched errors for batch/interop
    #[error("vwap: Output length mismatch (expected {expected}, got {got}).")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("vwap: Invalid kernel for batch path: {0:?}.")]
    InvalidKernelForBatch(Kernel),
    #[error("vwap: Invalid range expansion (start='{start}', end='{end}', step={step}).")]
    InvalidRange { start: String, end: String, step: u32 },
    #[error("vwap: Not enough valid data (needed {needed}, have {valid}).")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("vwap: All input values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn vwap(input: &VwapInput) -> Result<VwapOutput, VwapError> {
    vwap_with_kernel(input, Kernel::Auto)
}

pub fn vwap_with_kernel(input: &VwapInput, kernel: Kernel) -> Result<VwapOutput, VwapError> {
    let (timestamps, volumes, prices) = match &input.data {
        VwapData::Candles { candles, source } => {
            let timestamps: &[i64] = candles
                .get_timestamp()
                .map_err(|e| VwapError::ParseAnchorError { msg: e.to_string() })?;
            let prices: &[f64] = source_type(candles, source);
            let vols: &[f64] = candles
                .select_candle_field("volume")
                .map_err(|e| VwapError::ParseAnchorError { msg: e.to_string() })?;
            (timestamps, vols, prices)
        }
        VwapData::CandlesPlusPrices { candles, prices } => {
            let timestamps: &[i64] = candles
                .get_timestamp()
                .map_err(|e| VwapError::ParseAnchorError { msg: e.to_string() })?;
            let vols: &[f64] = candles
                .select_candle_field("volume")
                .map_err(|e| VwapError::ParseAnchorError { msg: e.to_string() })?;
            (timestamps, vols, *prices)
        }
        VwapData::Slice {
            timestamps,
            volumes,
            prices,
        } => (*timestamps, *volumes, *prices),
    };

    let n = prices.len();
    if timestamps.len() != n || volumes.len() != n {
        return Err(VwapError::MismatchTimestampsPricesVolumes {
            timestamps: timestamps.len(),
            prices: n,
            volumes: volumes.len(),
        });
    }
    if n == 0 {
        return Err(VwapError::NoData);
    }

    let (count, unit_char) = parse_anchor(input.get_anchor())
        .map_err(|e| VwapError::ParseAnchorError { msg: e.to_string() })?;

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut values = alloc_with_nan_prefix(n, 0);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                vwap_scalar(timestamps, volumes, prices, count, unit_char, &mut values)?
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                vwap_avx2(timestamps, volumes, prices, count, unit_char, &mut values)?
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                vwap_avx512(timestamps, volumes, prices, count, unit_char, &mut values)?
            }
            _ => unreachable!(),
        }
    }

    Ok(VwapOutput { values })
}

/// Write VWAP values directly to output slice - no allocations
#[inline]
pub fn vwap_into_slice(dst: &mut [f64], input: &VwapInput, kern: Kernel) -> Result<(), VwapError> {
    let (timestamps, volumes, prices) = match &input.data {
        VwapData::Candles { candles, source } => {
            let timestamps: &[i64] = candles
                .get_timestamp()
                .map_err(|e| VwapError::ParseAnchorError { msg: e.to_string() })?;
            let prices: &[f64] = source_type(candles, source);
            let vols: &[f64] = candles
                .select_candle_field("volume")
                .map_err(|e| VwapError::ParseAnchorError { msg: e.to_string() })?;
            (timestamps, vols, prices)
        }
        VwapData::CandlesPlusPrices { candles, prices } => {
            let timestamps: &[i64] = candles
                .get_timestamp()
                .map_err(|e| VwapError::ParseAnchorError { msg: e.to_string() })?;
            let vols: &[f64] = candles
                .select_candle_field("volume")
                .map_err(|e| VwapError::ParseAnchorError { msg: e.to_string() })?;
            (timestamps, vols, *prices)
        }
        VwapData::Slice {
            timestamps,
            volumes,
            prices,
        } => (*timestamps, *volumes, *prices),
    };

    let n = prices.len();
    if dst.len() != n {
        return Err(VwapError::OutputLengthMismatch {
            expected: n,
            got: dst.len(),
        });
    }
    if timestamps.len() != n || volumes.len() != n {
        return Err(VwapError::MismatchTimestampsPricesVolumes {
            timestamps: timestamps.len(),
            prices: n,
            volumes: volumes.len(),
        });
    }
    if n == 0 {
        return Err(VwapError::NoData);
    }

    let (count, unit_char) = parse_anchor(input.get_anchor())
        .map_err(|e| VwapError::ParseAnchorError { msg: e.to_string() })?;

    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                vwap_scalar(timestamps, volumes, prices, count, unit_char, dst)?
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                vwap_avx2(timestamps, volumes, prices, count, unit_char, dst)?
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                vwap_avx512(timestamps, volumes, prices, count, unit_char, dst)?
            }
            _ => unreachable!(),
        }
    }

    // VWAP doesn't have a warmup period in the traditional sense,
    // but we may need to fill initial values with NaN until first anchor reset
    // This is handled inside the kernel functions

    Ok(())
}

/// Writes VWAP values into a caller-provided slice without allocating.
///
/// - Preserves NaN warmups exactly as the Vec-returning API does.
/// - The output slice length must equal the input length; returns an error on mismatch.
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn vwap_into(input: &VwapInput, out: &mut [f64]) -> Result<(), VwapError> {
    // Delegate to the existing compute-into function with automatic kernel selection.
    vwap_into_slice(out, input, Kernel::Auto)
}

#[inline(always)]
pub fn vwap_scalar(
    timestamps: &[i64],
    volumes: &[f64],
    prices: &[f64],
    count: u32,
    unit_char: char,
    out: &mut [f64],
) -> Result<(), VwapError> {
    debug_assert_eq!(out.len(), prices.len(), "output slice length mismatch");

    // Running state for the current anchor bucket
    let mut current_group_id: i64 = i64::MIN; // sentinel that won't match real gids
    let mut volume_sum: f64 = 0.0;
    let mut vol_price_sum: f64 = 0.0;

    let n = prices.len();
    if n == 0 {
        return Ok(());
    }

    // Fast path for minute/hour/day anchors: single integer division per element
    unsafe {
        let ts_ptr = timestamps.as_ptr();
        let vol_ptr = volumes.as_ptr();
        let pr_ptr = prices.as_ptr();
        let out_ptr = out.as_mut_ptr();

        if unit_char == 'm' || unit_char == 'h' || unit_char == 'd' {
            const MINUTE_MS: i64 = 60_000;
            const HOUR_MS: i64 = 3_600_000;
            const DAY_MS: i64 = 86_400_000;

            let unit_ms: i64 = match unit_char {
                'm' => MINUTE_MS,
                'h' => HOUR_MS,
                _ => DAY_MS, // 'd'
            };
            let bucket_ms: i64 = (count as i64) * unit_ms;

            // Unroll by 2 to reduce loop overhead
            let mut i: usize = 0;
            let unroll_end = n & !1usize; // round down to even

            while i < unroll_end {
                // ---- iteration i ----
                let ts0 = *ts_ptr.add(i);
                let gid0 = ts0 / bucket_ms;
                if gid0 != current_group_id {
                    current_group_id = gid0;
                    volume_sum = 0.0;
                    vol_price_sum = 0.0;
                }
                let v0 = *vol_ptr.add(i);
                let p0 = *pr_ptr.add(i);
                volume_sum += v0;
                vol_price_sum = p0.mul_add(v0, vol_price_sum);
                *out_ptr.add(i) = if volume_sum > 0.0 {
                    vol_price_sum / volume_sum
                } else {
                    f64::NAN
                };

                // ---- iteration i+1 ----
                let idx1 = i + 1;
                let ts1 = *ts_ptr.add(idx1);
                let gid1 = ts1 / bucket_ms;
                if gid1 != current_group_id {
                    current_group_id = gid1;
                    volume_sum = 0.0;
                    vol_price_sum = 0.0;
                }
                let v1 = *vol_ptr.add(idx1);
                let p1 = *pr_ptr.add(idx1);
                volume_sum += v1;
                vol_price_sum = p1.mul_add(v1, vol_price_sum);
                *out_ptr.add(idx1) = if volume_sum > 0.0 {
                    vol_price_sum / volume_sum
                } else {
                    f64::NAN
                };

                i += 2;
            }

            // Remainder
            if unroll_end != n {
                let ts = *ts_ptr.add(unroll_end);
                let gid = ts / bucket_ms;
                if gid != current_group_id {
                    current_group_id = gid;
                    volume_sum = 0.0;
                    vol_price_sum = 0.0;
                }
                let v = *vol_ptr.add(unroll_end);
                let p = *pr_ptr.add(unroll_end);
                volume_sum += v;
                vol_price_sum = p.mul_add(v, vol_price_sum);
                *out_ptr.add(unroll_end) = if volume_sum > 0.0 {
                    vol_price_sum / volume_sum
                } else {
                    f64::NAN
                };
            }

            return Ok(());
        }

        // Month-based grouping: precise calendar bucketing per element
        if unit_char == 'M' {
            let mut i: usize = 0;
            while i < n {
                let ts = *ts_ptr.add(i);
                let gid = match floor_to_month(ts, count) {
                    Ok(g) => g,
                    Err(_) => return Err(VwapError::MonthConversionError { ts_ms: ts }),
                };
                if gid != current_group_id {
                    current_group_id = gid;
                    volume_sum = 0.0;
                    vol_price_sum = 0.0;
                }
                let v = *vol_ptr.add(i);
                let p = *pr_ptr.add(i);
                volume_sum += v;
                vol_price_sum = p.mul_add(v, vol_price_sum);
                *out_ptr.add(i) = if volume_sum > 0.0 {
                    vol_price_sum / volume_sum
                } else {
                    f64::NAN
                };
                i += 1;
            }
            return Ok(());
        }
    }

    Err(VwapError::UnsupportedAnchorUnit { unit_char })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vwap_avx2(
    timestamps: &[i64],
    volumes: &[f64],
    prices: &[f64],
    count: u32,
    unit_char: char,
    out: &mut [f64],
) -> Result<(), VwapError> {
    vwap_scalar(timestamps, volumes, prices, count, unit_char, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vwap_avx512(
    timestamps: &[i64],
    volumes: &[f64],
    prices: &[f64],
    count: u32,
    unit_char: char,
    out: &mut [f64],
) -> Result<(), VwapError> {
    vwap_scalar(timestamps, volumes, prices, count, unit_char, out)
}

/// Streaming VWAP (per anchor bucket) — O(1) update with minimized divisions.
/// Divisions:
///   - 'm'/'h'/'d': at most one integer division when entering a new window,
///     zero divisions for all subsequent ticks inside the same window.
///   - 'M' (months): no chrono conversion on every tick; we compute the next
///     boundary only when the tick crosses it.
#[derive(Debug, Clone)]
pub struct VwapStream {
    // configuration
    anchor: String,
    count: u32,
    unit_char: char,

    // fast path for linear-time anchors (m/h/d)
    bucket_ms: i64,   // 0 for 'M' anchors
    next_cutoff: i64, // absolute timestamp (ms) of the next bucket boundary
    current_group_id: i64,

    // accumulators
    volume_sum: f64,
    vol_price_sum: f64,
}

impl VwapStream {
    #[inline]
    pub fn try_new(params: VwapParams) -> Result<Self, VwapError> {
        let anchor = params.anchor.unwrap_or_else(|| "1d".to_string());
        let (count, unit_char) = parse_anchor(&anchor)
            .map_err(|e| VwapError::ParseAnchorError { msg: e.to_string() })?;

        // Precompute bucket size for m/h/d; 0 means month-based path.
        const MINUTE_MS: i64 = 60_000;
        const HOUR_MS: i64 = 3_600_000;
        const DAY_MS: i64 = 86_400_000;

        let bucket_ms = match unit_char {
            'm' => (count as i64).saturating_mul(MINUTE_MS),
            'h' => (count as i64).saturating_mul(HOUR_MS),
            'd' => (count as i64).saturating_mul(DAY_MS),
            'M' => 0,
            _ => return Err(VwapError::UnsupportedAnchorUnit { unit_char }),
        };

        Ok(Self {
            anchor,
            count,
            unit_char,
            bucket_ms,
            next_cutoff: i64::MIN,      // lazy-init on first tick
            current_group_id: i64::MIN, // lazy-init on first tick
            volume_sum: 0.0,
            vol_price_sum: 0.0,
        })
    }

    /// Amortized O(1) update with zero divisions inside a window for m/h/d.
    /// Returns `Some(vwap)` if cumulative volume in the current window > 0, else `None`.
    #[inline(always)]
    pub fn update(&mut self, timestamp: i64, price: f64, volume: f64) -> Option<f64> {
        match self.unit_char {
            'm' | 'h' | 'd' => self.update_linear(timestamp, price, volume),
            'M' => self.update_month(timestamp, price, volume),
            _ => None,
        }
    }

    // --------- m/h/d path (no chrono on tick; divisions only on boundary) ---------

    #[inline(always)]
    fn init_linear(&mut self, ts: i64) {
        // One division to seed the stream.
        let gid = ts / self.bucket_ms;
        self.current_group_id = gid;
        // next_cutoff = start_of_window + bucket_ms = (gid + 1) * bucket_ms
        self.next_cutoff = gid.saturating_add(1).saturating_mul(self.bucket_ms);
        self.volume_sum = 0.0;
        self.vol_price_sum = 0.0;
    }

    #[inline(always)]
    fn roll_linear_to(&mut self, ts: i64) {
        // We crossed one or more windows. Compute how many with a single division, once.
        // k = number of whole buckets to jump forward (>= 1).
        let delta = ts.saturating_sub(self.next_cutoff);
        let k = (delta / self.bucket_ms).saturating_add(1);
        self.current_group_id = self.current_group_id.saturating_add(k);
        self.next_cutoff = self
            .next_cutoff
            .saturating_add(self.bucket_ms.saturating_mul(k));
        self.volume_sum = 0.0;
        self.vol_price_sum = 0.0;
    }

    #[inline(always)]
    fn update_linear(&mut self, ts: i64, price: f64, volume: f64) -> Option<f64> {
        debug_assert!(self.bucket_ms > 0);
        if self.current_group_id == i64::MIN {
            self.init_linear(ts);
        } else if ts >= self.next_cutoff {
            self.roll_linear_to(ts); // one division per boundary crossing
        }

        // Accumulate with FMA to reduce rounding error and improve throughput.
        self.volume_sum += volume;
        self.vol_price_sum = price.mul_add(volume, self.vol_price_sum);

        if self.volume_sum > 0.0 {
            Some(self.vol_price_sum / self.volume_sum)
        } else {
            None
        }
    }

    // ---------------------------- 'M' (months) path ----------------------------

    // Helper: compute current month-group id and the next bucket boundary timestamp in ms.
    // gid definition matches `floor_to_month(ts, count)`.
    #[inline]
    fn month_gid_and_next_cutoff(&self, ts_ms: i64) -> Result<(i64, i64), VwapError> {
        // Convert to calendar time once (only when boundary init/crossing).
        let seconds = ts_ms / 1000;
        let nanos = ((ts_ms % 1000) * 1_000_000) as u32;
        let dt = NaiveDateTime::from_timestamp_opt(seconds, nanos)
            .ok_or_else(|| VwapError::MonthConversionError { ts_ms })?;

        let year = dt.year();
        let month = dt.month() as i32; // 1..=12
        let total_months = (year - 1970) as i64 * 12 + (month - 1) as i64;
        let gid = total_months / (self.count as i64);

        // Compute next bucket's first day 00:00:00 as ms since epoch.
        let next_bucket_months = gid.saturating_add(1).saturating_mul(self.count as i64);

        let next_year = 1970 + next_bucket_months.div_euclid(12);
        let next_month0 = next_bucket_months.rem_euclid(12); // 0..=11
        let next_date = NaiveDate::from_ymd_opt(next_year as i32, (next_month0 + 1) as u32, 1)
            .ok_or_else(|| VwapError::MonthConversionError { ts_ms })?;
        let next_dt = next_date
            .and_hms_opt(0, 0, 0)
            .ok_or_else(|| VwapError::MonthConversionError { ts_ms })?;

        // chrono NaiveDateTime::timestamp() gives seconds since epoch; multiply to ms.
        let next_ms = next_dt.timestamp().saturating_mul(1000);

        Ok((gid, next_ms))
    }

    #[inline(always)]
    fn init_month(&mut self, ts: i64) -> Option<()> {
        match self.month_gid_and_next_cutoff(ts) {
            Ok((gid, next_ms)) => {
                self.current_group_id = gid;
                self.next_cutoff = next_ms;
                self.volume_sum = 0.0;
                self.vol_price_sum = 0.0;
                Some(())
            }
            Err(_) => None,
        }
    }

    #[inline(always)]
    fn update_month(&mut self, ts: i64, price: f64, volume: f64) -> Option<f64> {
        if self.current_group_id == i64::MIN {
            if self.init_month(ts).is_none() {
                return None;
            }
        } else if ts >= self.next_cutoff {
            // We may have skipped multiple month-groups; recompute gid and next cutoff
            match self.month_gid_and_next_cutoff(ts) {
                Ok((gid, next_ms)) => {
                    self.current_group_id = gid;
                    self.next_cutoff = next_ms;
                    self.volume_sum = 0.0;
                    self.vol_price_sum = 0.0;
                }
                Err(_) => return None,
            }
        }

        self.volume_sum += volume;
        self.vol_price_sum = price.mul_add(volume, self.vol_price_sum);

        if self.volume_sum > 0.0 {
            Some(self.vol_price_sum / self.volume_sum)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct VwapBatchRange {
    pub anchor: (String, String, u32),
}

impl Default for VwapBatchRange {
    fn default() -> Self {
        Self {
            anchor: ("1d".to_string(), "1d".to_string(), 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct VwapBatchBuilder {
    range: VwapBatchRange,
    kernel: Kernel,
}

impl VwapBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn anchor_range(
        mut self,
        start: impl Into<String>,
        end: impl Into<String>,
        step: u32,
    ) -> Self {
        self.range.anchor = (start.into(), end.into(), step);
        self
    }
    #[inline]
    pub fn anchor_static(mut self, val: impl Into<String>) -> Self {
        let s = val.into();
        self.range.anchor = (s.clone(), s, 0);
        self
    }
    pub fn apply_slice(
        self,
        timestamps: &[i64],
        volumes: &[f64],
        prices: &[f64],
    ) -> Result<VwapBatchOutput, VwapError> {
        vwap_batch_with_kernel(timestamps, volumes, prices, &self.range, self.kernel)
    }
}

pub fn vwap_batch_with_kernel(
    timestamps: &[i64],
    volumes: &[f64],
    prices: &[f64],
    sweep: &VwapBatchRange,
    k: Kernel,
) -> Result<VwapBatchOutput, VwapError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(VwapError::InvalidKernelForBatch(other)),
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };

    vwap_batch_inner(timestamps, volumes, prices, sweep, simd, true)
}

#[derive(Clone, Debug)]
pub struct VwapBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VwapParams>,
    pub rows: usize,
    pub cols: usize,
}

impl VwapBatchOutput {
    pub fn row_for_anchor(&self, a: &str) -> Option<usize> {
        self.combos
            .iter()
            .position(|p| p.anchor.as_deref() == Some(a))
    }

    pub fn values_for(&self, a: &str) -> Option<&[f64]> {
        self.row_for_anchor(a).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
pub(crate) fn expand_grid_vwap(r: &VwapBatchRange) -> Vec<VwapParams> {
    // Static or equal bounds → single value
    if r.anchor.2 == 0 || r.anchor.0 == r.anchor.1 {
        return vec![VwapParams {
            anchor: Some(r.anchor.0.clone()),
        }];
    }
    // Robust expansion: support reversed bounds and explicit step
    let step = r.anchor.2.max(1);
    let start = anchor_to_num_and_unit(&r.anchor.0);
    let end = anchor_to_num_and_unit(&r.anchor.1);
    let mut result = Vec::new();
    if let (Some((mut a, unit_a)), Some((b, unit_b))) = (start, end) {
        if unit_a != unit_b {
            // Unit mismatch → no combos; caller will convert to typed error
            return result;
        }
        if a <= b {
            while a <= b {
                result.push(VwapParams {
                    anchor: Some(format!("{}{}", a, unit_a)),
                });
                a = a.saturating_add(step);
            }
        } else {
            // Reversed bounds: descend
            while a >= b {
                result.push(VwapParams {
                    anchor: Some(format!("{}{}", a, unit_a)),
                });
                if a < step { break; }
                a -= step;
            }
        }
    }
    result
}

fn anchor_to_num_and_unit(anchor: &str) -> Option<(u32, char)> {
    let mut idx = 0;
    for (pos, ch) in anchor.char_indices() {
        if !ch.is_ascii_digit() {
            idx = pos;
            break;
        }
    }
    if idx == 0 {
        return None;
    }
    let num = anchor[..idx].parse::<u32>().ok()?;
    let unit = anchor[idx..].chars().next()?;
    Some((num, unit))
}

#[inline]
pub(crate) fn first_valid_vwap_index(
    timestamps: &[i64],
    volumes: &[f64],
    count: u32,
    unit: char,
) -> usize {
    if timestamps.is_empty() {
        return 0;
    }
    let mut cur_gid = i64::MIN;
    let mut vsum = 0.0;
    for i in 0..timestamps.len() {
        let ts = timestamps[i];
        let gid = match unit {
            'm' => ts / ((count as i64) * 60_000),
            'h' => ts / ((count as i64) * 3_600_000),
            'd' => ts / ((count as i64) * 86_400_000),
            'M' => floor_to_month(ts, count).unwrap_or(i64::MIN),
            _ => i64::MIN,
        };
        if gid != cur_gid {
            cur_gid = gid;
            vsum = 0.0;
        }
        vsum += volumes[i];
        if vsum > 0.0 {
            return i;
        }
    }
    0
}

#[inline(always)]
pub fn vwap_batch_slice(
    timestamps: &[i64],
    volumes: &[f64],
    prices: &[f64],
    sweep: &VwapBatchRange,
    kern: Kernel,
) -> Result<VwapBatchOutput, VwapError> {
    vwap_batch_inner(timestamps, volumes, prices, sweep, kern, false)
}

#[inline(always)]
pub fn vwap_batch_par_slice(
    timestamps: &[i64],
    volumes: &[f64],
    prices: &[f64],
    sweep: &VwapBatchRange,
    kern: Kernel,
) -> Result<VwapBatchOutput, VwapError> {
    vwap_batch_inner(timestamps, volumes, prices, sweep, kern, true)
}

fn vwap_batch_inner(
    timestamps: &[i64],
    volumes: &[f64],
    prices: &[f64],
    sweep: &VwapBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<VwapBatchOutput, VwapError> {
    let combos = expand_grid_vwap(sweep);
    if combos.is_empty() {
        return Err(VwapError::InvalidRange {
            start: sweep.anchor.0.clone(),
            end: sweep.anchor.1.clone(),
            step: sweep.anchor.2,
        });
    }

    let rows = combos.len();
    let cols = prices.len();

    let mut raw = make_uninit_matrix(rows, cols); // Vec<MaybeUninit<f64>>

    // compute warm prefix per row
    let mut warm: Vec<usize> = Vec::with_capacity(rows);
    for prm in &combos {
        let (cnt, unit) = parse_anchor(prm.anchor.as_deref().unwrap_or("1d"))
            .map_err(|e| VwapError::ParseAnchorError { msg: e.to_string() })?;
        let w = first_valid_vwap_index(timestamps, volumes, cnt, unit);
        warm.push(w);
    }
    init_matrix_prefixes(&mut raw, cols, &warm);

    // Shared precompute across rows: price*volume vector to halve inner-loop FLOPs
    // Safe: computed once per batch and reused for all parameter rows.
    let pv: Vec<f64> = prices
        .iter()
        .zip(volumes.iter())
        .map(|(&p, &v)| p * v)
        .collect();

    // ---------- 2. closure to fill one row in-place --------------------------
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let params = combos.get(row).unwrap();
        let (count, unit_char) = parse_anchor(params.anchor.as_deref().unwrap_or("1d"))
            .map_err(|e| VwapError::ParseAnchorError { msg: e.to_string() })
            .unwrap();

        // cast this row to &mut [f64]
        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            Kernel::Scalar => {
                vwap_row_scalar_pv(timestamps, volumes, &pv, count, unit_char, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => vwap_row_avx2(timestamps, volumes, prices, count, unit_char, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => {
                vwap_row_avx512(timestamps, volumes, prices, count, unit_char, out_row)
            }
            _ => unreachable!(),
        }
    };

    // ---------- 3. run every row, writing directly into `raw` ----------------
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            raw.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in raw.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in raw.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    // ---------- 4. all elements written – turn into Vec<f64> -----------------
    let mut raw_guard = core::mem::ManuallyDrop::new(raw);
    let values: Vec<f64> = unsafe {
        Vec::from_raw_parts(
            raw_guard.as_mut_ptr() as *mut f64,
            raw_guard.len(),
            raw_guard.capacity(),
        )
    };

    Ok(VwapBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn vwap_batch_inner_into(
    timestamps: &[i64],
    volumes: &[f64],
    prices: &[f64],
    sweep: &VwapBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<VwapParams>, VwapError> {
    if timestamps.len() != volumes.len() || volumes.len() != prices.len() {
        return Err(VwapError::MismatchTimestampsPricesVolumes {
            timestamps: timestamps.len(),
            prices: prices.len(),
            volumes: volumes.len(),
        });
    }
    let combos = expand_grid_vwap(sweep);
    if combos.is_empty() {
        return Err(VwapError::InvalidRange {
            start: sweep.anchor.0.clone(),
            end: sweep.anchor.1.clone(),
            step: sweep.anchor.2,
        });
    }

    let rows = combos.len();
    let cols = prices.len();

    // Ensure output buffer is the right size with checked arithmetic
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| VwapError::InvalidRange {
            start: sweep.anchor.0.clone(),
            end: sweep.anchor.1.clone(),
            step: sweep.anchor.2,
        })?;
    if out.len() != expected {
        return Err(VwapError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    // Shared precompute across rows: price*volume vector
    let pv: Vec<f64> = prices
        .iter()
        .zip(volumes.iter())
        .map(|(&p, &v)| p * v)
        .collect();

    // ---------- 2. closure to fill one row in-place --------------------------
    let do_row = |row: usize, dst: &mut [f64]| unsafe {
        let params = combos.get(row).unwrap();
        let (count, unit_char) = parse_anchor(params.anchor.as_deref().unwrap_or("1d"))
            .map_err(|e| VwapError::ParseAnchorError { msg: e.to_string() })
            .unwrap();

        // Use the slice directly
        let out_row = dst;

        match kern {
            Kernel::Scalar => {
                vwap_row_scalar_pv(timestamps, volumes, &pv, count, unit_char, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => vwap_row_avx2(timestamps, volumes, prices, count, unit_char, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => {
                vwap_row_avx512(timestamps, volumes, prices, count, unit_char, out_row)
            }
            _ => unreachable!(),
        }
    };

    // ---------- 3. run every row, writing directly into `out` ----------------
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in out.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in out.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(combos)
}

#[inline]
pub(crate) fn parse_anchor(anchor: &str) -> Result<(u32, char), Box<dyn std::error::Error>> {
    let mut idx = 0;
    for (pos, ch) in anchor.char_indices() {
        if !ch.is_ascii_digit() {
            idx = pos;
            break;
        }
    }
    if idx == 0 {
        return Err(format!("No numeric portion found in anchor '{}'", anchor).into());
    }
    let num_part = &anchor[..idx];
    let unit_part = &anchor[idx..];
    let count = num_part
        .parse::<u32>()
        .map_err(|_| format!("Failed parsing numeric portion '{}'", num_part))?;
    if unit_part.len() != 1 {
        return Err(format!("Anchor unit must be 1 char (found '{}')", unit_part).into());
    }
    let mut unit_char = unit_part.chars().next().unwrap();
    unit_char = match unit_char {
        'H' => 'h',
        'D' => 'd',
        c => c,
    };
    match unit_char {
        'm' | 'h' | 'd' | 'M' => Ok((count, unit_char)),
        _ => Err(format!("Unsupported unit '{}'", unit_char).into()),
    }
}

#[inline]
pub(crate) fn floor_to_month(ts_ms: i64, count: u32) -> Result<i64, Box<dyn Error>> {
    // Convert timestamp to real calendar month
    let seconds = ts_ms / 1000;
    let nanos = ((ts_ms % 1000) * 1_000_000) as u32;

    let dt = NaiveDateTime::from_timestamp_opt(seconds, nanos)
        .ok_or_else(|| format!("Invalid timestamp: {}", ts_ms))?;

    // Calculate total months since epoch (Jan 1970 = month 0)
    let year = dt.year();
    let month = dt.month() as i32;
    let total_months = (year - 1970) as i64 * 12 + (month - 1) as i64;

    if count == 1 {
        // For 1M anchor, return the month count directly
        Ok(total_months)
    } else {
        // For multi-month anchors (e.g., 3M for quarters), group by count
        Ok(total_months / (count as i64))
    }
}

/// Row-parallel helpers
#[inline(always)]
pub unsafe fn vwap_row_scalar(
    timestamps: &[i64],
    volumes: &[f64],
    prices: &[f64],
    count: u32,
    unit_char: char,
    out: &mut [f64],
) {
    vwap_scalar(timestamps, volumes, prices, count, unit_char, out);
}

#[inline(always)]
unsafe fn vwap_row_scalar_pv(
    timestamps: &[i64],
    volumes: &[f64],
    pv: &[f64],
    count: u32,
    unit_char: char,
    out: &mut [f64],
) {
    debug_assert_eq!(pv.len(), out.len(), "pv length must match out len");
    // Re-implement the scalar loop but reuse precomputed pv[i] instead of price*volume
    let n = out.len();
    if n == 0 {
        return;
    }
    let mut current_group_id: i64 = i64::MIN;
    let mut volume_sum: f64 = 0.0;
    let mut vol_price_sum: f64 = 0.0;

    let ts_ptr = timestamps.as_ptr();
    let vol_ptr = volumes.as_ptr();
    let pv_ptr = pv.as_ptr();
    let out_ptr = out.as_mut_ptr();

    if unit_char == 'm' || unit_char == 'h' || unit_char == 'd' {
        const MINUTE_MS: i64 = 60_000;
        const HOUR_MS: i64 = 3_600_000;
        const DAY_MS: i64 = 86_400_000;
        let unit_ms: i64 = match unit_char {
            'm' => MINUTE_MS,
            'h' => HOUR_MS,
            _ => DAY_MS,
        };
        let bucket_ms: i64 = (count as i64) * unit_ms;

        let mut i: usize = 0;
        let unroll_end = n & !1usize;
        while i < unroll_end {
            let ts0 = *ts_ptr.add(i);
            let gid0 = ts0 / bucket_ms;
            if gid0 != current_group_id {
                current_group_id = gid0;
                volume_sum = 0.0;
                vol_price_sum = 0.0;
            }
            let v0 = *vol_ptr.add(i);
            let pv0 = *pv_ptr.add(i);
            volume_sum += v0;
            vol_price_sum += pv0;
            *out_ptr.add(i) = if volume_sum > 0.0 {
                vol_price_sum / volume_sum
            } else {
                f64::NAN
            };

            let idx1 = i + 1;
            let ts1 = *ts_ptr.add(idx1);
            let gid1 = ts1 / bucket_ms;
            if gid1 != current_group_id {
                current_group_id = gid1;
                volume_sum = 0.0;
                vol_price_sum = 0.0;
            }
            let v1 = *vol_ptr.add(idx1);
            let pv1 = *pv_ptr.add(idx1);
            volume_sum += v1;
            vol_price_sum += pv1;
            *out_ptr.add(idx1) = if volume_sum > 0.0 {
                vol_price_sum / volume_sum
            } else {
                f64::NAN
            };

            i += 2;
        }
        if unroll_end != n {
            let ts = *ts_ptr.add(unroll_end);
            let gid = ts / bucket_ms;
            if gid != current_group_id {
                current_group_id = gid;
                volume_sum = 0.0;
                vol_price_sum = 0.0;
            }
            let v = *vol_ptr.add(unroll_end);
            let pvx = *pv_ptr.add(unroll_end);
            volume_sum += v;
            vol_price_sum += pvx;
            *out_ptr.add(unroll_end) = if volume_sum > 0.0 {
                vol_price_sum / volume_sum
            } else {
                f64::NAN
            };
        }
        return;
    }

    if unit_char == 'M' {
        let mut i: usize = 0;
        while i < n {
            let ts = *ts_ptr.add(i);
            let gid = match floor_to_month(ts, count) {
                Ok(g) => g,
                Err(_) => return,
            };
            if gid != current_group_id {
                current_group_id = gid;
                volume_sum = 0.0;
                vol_price_sum = 0.0;
            }
            let v = *vol_ptr.add(i);
            let pvx = *pv_ptr.add(i);
            volume_sum += v;
            vol_price_sum += pvx;
            *out_ptr.add(i) = if volume_sum > 0.0 {
                vol_price_sum / volume_sum
            } else {
                f64::NAN
            };
            i += 1;
        }
        return;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vwap_row_avx2(
    timestamps: &[i64],
    volumes: &[f64],
    prices: &[f64],
    count: u32,
    unit_char: char,
    out: &mut [f64],
) {
    vwap_row_scalar(timestamps, volumes, prices, count, unit_char, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vwap_row_avx512(
    timestamps: &[i64],
    volumes: &[f64],
    prices: &[f64],
    count: u32,
    unit_char: char,
    out: &mut [f64],
) {
    vwap_row_scalar(timestamps, volumes, prices, count, unit_char, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vwap_row_avx512_short(
    timestamps: &[i64],
    volumes: &[f64],
    prices: &[f64],
    count: u32,
    unit_char: char,
    out: &mut [f64],
) {
    vwap_row_scalar(timestamps, volumes, prices, count, unit_char, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vwap_row_avx512_long(
    timestamps: &[i64],
    volumes: &[f64],
    prices: &[f64],
    count: u32,
    unit_char: char,
    out: &mut [f64],
) {
    vwap_row_scalar(timestamps, volumes, prices, count, unit_char, out);
}

// expand_grid function, not public
#[inline(always)]
fn expand_grid(r: &VwapBatchRange) -> Vec<VwapParams> {
    expand_grid_vwap(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_vwap_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params_default = VwapParams { anchor: None };
        let input_default = VwapInput::from_candles(&candles, "hlc3", params_default);
        let output_default = vwap_with_kernel(&input_default, kernel)?;
        assert_eq!(output_default.values.len(), candles.close.len());
        Ok(())
    }

    #[test]
    fn test_vwap_into_matches_api() -> Result<(), Box<dyn Error>> {
        // Prepare a non-trivial slice (~512 bars) from real data
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Extract sources identical to the normal API path (timestamps, volume, hlc3)
        let timestamps = candles.get_timestamp().map_err(|e| e.to_string())?;
        let volumes = candles
            .select_candle_field("volume")
            .map_err(|e| e.to_string())?;
        let prices = source_type(&candles, "hlc3");

        // Pick the last 512 bars (or all if shorter)
        let take = 512usize.min(prices.len());
        let start = prices.len().saturating_sub(take);
        let ts_slice = &timestamps[start..start + take];
        let vol_slice = &volumes[start..start + take];
        let price_slice = &prices[start..start + take];

        // Build input with default params (anchor defaults to "1d")
        let params = VwapParams { anchor: None };
        let input = VwapInput::from_slice(ts_slice, vol_slice, price_slice, params);

        // Baseline via existing Vec-returning API
        let baseline = vwap(&input)?.values;

        // Preallocate output buffer and compute via new into API
        let mut out = vec![0.0; price_slice.len()];
        vwap_into(&input, &mut out)?;

        assert_eq!(baseline.len(), out.len());

        // Helper: NaN == NaN; otherwise exact or tight epsilon
        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b) || ((a - b).abs() <= 1e-12)
        }

        for i in 0..out.len() {
            assert!(
                eq_or_both_nan(baseline[i], out[i]),
                "Mismatch at {}: baseline={}, into={}",
                i,
                baseline[i],
                out[i]
            );
        }

        Ok(())
    }

    fn check_vwap_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let expected_last_five_vwap = [
            59353.05963230107,
            59330.15815713043,
            59289.94649532547,
            59274.6155462414,
            58730.0,
        ];
        let file_path: &str = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = VwapParams {
            anchor: Some("1D".to_string()),
        };
        let input = VwapInput::from_candles(&candles, "hlc3", params);
        let result = vwap_with_kernel(&input, kernel)?;
        assert!(result.values.len() >= 5, "Not enough data points for test");
        let start_idx = result.values.len() - 5;
        for (i, &vwap_val) in result.values[start_idx..].iter().enumerate() {
            let exp_val = expected_last_five_vwap[i];
            assert!(
                (vwap_val - exp_val).abs() < 1e-5,
                "[{}] VWAP mismatch at index {}: expected {}, got {}",
                test_name,
                i,
                exp_val,
                vwap_val
            );
        }
        Ok(())
    }

    fn check_vwap_candles_plus_prices(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let source_prices = candles.get_calculated_field("hl2").unwrap();
        let params = VwapParams {
            anchor: Some("1d".to_string()),
        };
        let input = VwapInput::from_candles_plus_prices(&candles, source_prices, params);
        let result = vwap_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        Ok(())
    }

    fn check_vwap_anchor_parsing_error(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = VwapParams {
            anchor: Some("xyz".to_string()),
        };
        let input = VwapInput::from_candles(&candles, "hlc3", params);
        let result = vwap_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_vwap_slice_data_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = VwapParams {
            anchor: Some("1d".to_string()),
        };
        let first_input = VwapInput::from_candles(&candles, "close", first_params);
        let first_result = vwap_with_kernel(&first_input, kernel)?;
        let second_params = VwapParams {
            anchor: Some("1h".to_string()),
        };
        let source_prices = &first_result.values;
        let second_input =
            VwapInput::from_candles_plus_prices(&candles, source_prices, second_params);
        let second_result = vwap_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_vwap_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = VwapInput::with_default_candles(&candles);
        let result = vwap_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        for &val in &result.values {
            if !val.is_nan() {
                assert!(val.is_finite());
            }
        }
        Ok(())
    }

    fn check_vwap_with_default_candles(
        test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = VwapInput::with_default_candles(&candles);
        match input.data {
            VwapData::Candles { source, .. } => {
                assert_eq!(source, "hlc3");
            }
            _ => panic!("Expected VwapData::Candles"),
        }
        let anchor = input.get_anchor();
        assert_eq!(anchor, "1d");
        Ok(())
    }

    fn check_vwap_with_default_params(
        test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        let default_params = VwapParams::default();
        assert_eq!(default_params.anchor, Some("1d".to_string()));
        Ok(())
    }

    macro_rules! generate_all_vwap_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                    }
                )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test]
                    fn [<$test_fn _avx2_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                    }
                    #[test]
                    fn [<$test_fn _avx512_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                    }
                )*
            }
        }
    }

    // Check for poison values in single output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_vwap_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with multiple parameter combinations to increase coverage
        let test_anchors = vec!["1m", "5m", "15m", "30m", "1h", "4h", "1d", "3d"];
        let test_sources = vec!["close", "open", "high", "low", "hl2", "hlc3", "ohlc4"];

        for anchor in test_anchors {
            for source in &test_sources {
                let params = VwapParams {
                    anchor: Some(anchor.to_string()),
                };
                let input = VwapInput::from_candles(&candles, source, params);
                let output = vwap_with_kernel(&input, kernel)?;

                // Check every value for poison patterns
                for (i, &val) in output.values.iter().enumerate() {
                    // Skip NaN values as they're expected in the warmup period
                    if val.is_nan() {
                        continue;
                    }

                    let bits = val.to_bits();

                    // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
                    if bits == 0x11111111_11111111 {
                        panic!(
                            "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} (anchor={}, source={})",
                            test_name, val, bits, i, anchor, source
                        );
                    }

                    // Check for init_matrix_prefixes poison (0x22222222_22222222)
                    if bits == 0x22222222_22222222 {
                        panic!(
                            "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} (anchor={}, source={})",
                            test_name, val, bits, i, anchor, source
                        );
                    }

                    // Check for make_uninit_matrix poison (0x33333333_33333333)
                    if bits == 0x33333333_33333333 {
                        panic!(
                            "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} (anchor={}, source={})",
                            test_name, val, bits, i, anchor, source
                        );
                    }
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_vwap_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    fn check_vwap_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Load real market data for realistic testing
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Get all necessary data
        let timestamps = candles.get_timestamp().unwrap();
        let volumes = candles.select_candle_field("volume").unwrap();
        let close_prices = &candles.close;
        let high_prices = &candles.high;
        let low_prices = &candles.low;

        // Anchor periods to test (m, h, d, M are supported)
        let anchor_periods = vec!["30m", "1h", "4h", "12h", "1d", "2d", "3d"];

        // Strategy: test various anchor and data slice combinations
        let strat = (
            0usize..anchor_periods.len(),                 // anchor index
            0usize..timestamps.len().saturating_sub(200), // starting index
            100usize..=200,                               // length of data slice to use
        );

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(anchor_idx, start_idx, slice_len)| {
                // Ensure we have valid slice bounds
                let end_idx = (start_idx + slice_len).min(timestamps.len());
                if end_idx <= start_idx || end_idx - start_idx < 10 {
                    return Ok(()); // Skip invalid combinations
                }

                let anchor = anchor_periods[anchor_idx];
                let ts_slice = &timestamps[start_idx..end_idx];
                let vol_slice = &volumes[start_idx..end_idx];
                let price_slice = &close_prices[start_idx..end_idx];
                let high_slice = &high_prices[start_idx..end_idx];
                let low_slice = &low_prices[start_idx..end_idx];

                let params = VwapParams {
                    anchor: Some(anchor.to_string()),
                };
                let input = VwapInput::from_slice(ts_slice, vol_slice, price_slice, params.clone());

                // Test the specified kernel
                let result = vwap_with_kernel(&input, kernel);

                // Also compute with scalar kernel for reference
                let scalar_input =
                    VwapInput::from_slice(ts_slice, vol_slice, price_slice, params.clone());
                let scalar_result = vwap_with_kernel(&scalar_input, Kernel::Scalar);

                // Both should succeed or fail together
                match (result, scalar_result) {
                    (Ok(VwapOutput { values: out }), Ok(VwapOutput { values: ref_out })) => {
                        // Verify output length
                        prop_assert_eq!(out.len(), price_slice.len());
                        prop_assert_eq!(ref_out.len(), price_slice.len());

                        // Parse anchor to understand grouping
                        let (count, unit_char) = parse_anchor(anchor).unwrap();

                        // Test specific properties for valid outputs
                        for i in 0..out.len() {
                            let y = out[i];
                            let r = ref_out[i];

                            // Kernel consistency check
                            if y.is_finite() && r.is_finite() {
                                let y_bits = y.to_bits();
                                let r_bits = r.to_bits();
                                let ulp_diff: u64 = y_bits.abs_diff(r_bits);
                                prop_assert!(
                                    (y - r).abs() <= 1e-9 || ulp_diff <= 5,
                                    "Kernel mismatch at {}: {} vs {} (ULP={})",
                                    i,
                                    y,
                                    r,
                                    ulp_diff
                                );
                            } else if y.is_nan() && r.is_nan() {
                                // Both NaN is fine (zero volume case)
                                continue;
                            } else {
                                prop_assert_eq!(
                                    y.is_nan(),
                                    r.is_nan(),
                                    "NaN mismatch at {}: {} vs {}",
                                    i,
                                    y,
                                    r
                                );
                            }

                            // VWAP should be positive for positive prices and volumes
                            if y.is_finite() && vol_slice[i] > 0.0 && price_slice[i] > 0.0 {
                                prop_assert!(
									y > 0.0,
									"VWAP should be positive at {} for positive price {} and volume {}",
									i, price_slice[i], vol_slice[i]
								);
                            }
                        }

                        // Test VWAP formula property: Find anchor boundaries and verify
                        let mut current_group_id = -1_i64;
                        let mut group_start = 0;

                        for i in 0..ts_slice.len() {
                            let ts_ms = ts_slice[i];
                            let group_id = match unit_char {
                                'm' => ts_ms / ((count as i64) * 60_000),
                                'h' => ts_ms / ((count as i64) * 3_600_000),
                                'd' => ts_ms / ((count as i64) * 86_400_000),
                                'M' => floor_to_month(ts_ms, count).unwrap_or(-1),
                                _ => -1,
                            };

                            // Check if we've crossed an anchor boundary
                            if group_id != current_group_id {
                                // Verify the previous group if it exists
                                if current_group_id != -1 && i > group_start {
                                    // Calculate expected VWAP for this group
                                    let mut vol_sum = 0.0;
                                    let mut vol_price_sum = 0.0;
                                    let mut min_price = f64::INFINITY;
                                    let mut max_price = f64::NEG_INFINITY;

                                    for j in group_start..i {
                                        vol_sum += vol_slice[j];
                                        vol_price_sum += vol_slice[j] * price_slice[j];
                                        min_price = min_price.min(low_slice[j]);
                                        max_price = max_price.max(high_slice[j]);
                                    }

                                    if vol_sum > 0.0 {
                                        let expected_vwap = vol_price_sum / vol_sum;
                                        let actual_vwap = out[i - 1]; // Last value of previous group

                                        // VWAP should match the formula
                                        prop_assert!(
											(actual_vwap - expected_vwap).abs() < 1e-6,
											"VWAP formula mismatch at group ending at {}: {} vs expected {}",
											i - 1, actual_vwap, expected_vwap
										);

                                        // VWAP should be within price bounds of the period
                                        // Using low/high for bounds check since VWAP uses close prices
                                        prop_assert!(
                                            actual_vwap >= min_price - 1e-9
                                                && actual_vwap <= max_price + 1e-9,
                                            "VWAP {} outside price bounds [{}, {}] at {}",
                                            actual_vwap,
                                            min_price,
                                            max_price,
                                            i - 1
                                        );
                                    }
                                }

                                current_group_id = group_id;
                                group_start = i;
                            }
                        }

                        // Test streaming consistency for smaller slices
                        if price_slice.len() <= 50 {
                            let mut stream = VwapStream::try_new(params).unwrap();
                            let mut stream_values = Vec::with_capacity(price_slice.len());

                            for i in 0..price_slice.len() {
                                match stream.update(ts_slice[i], price_slice[i], vol_slice[i]) {
                                    Some(val) => stream_values.push(val),
                                    None => stream_values.push(f64::NAN),
                                }
                            }

                            // Compare streaming output with batch output
                            for (i, (&batch_val, &stream_val)) in
                                out.iter().zip(stream_values.iter()).enumerate()
                            {
                                if batch_val.is_nan() && stream_val.is_nan() {
                                    continue;
                                }
                                if batch_val.is_finite() && stream_val.is_finite() {
                                    prop_assert!(
                                        (batch_val - stream_val).abs() < 1e-9,
                                        "Streaming mismatch at {}: batch={} vs stream={}",
                                        i,
                                        batch_val,
                                        stream_val
                                    );
                                }
                            }
                        }

                        // Test volume weighting property: VWAP is cumulative within anchor period
                        // Create synthetic timestamps that are guaranteed to be in the same anchor period
                        {
                            // Use a base timestamp and add small increments
                            let base_ts = 1609459200000_i64; // Jan 1, 2021 00:00:00 UTC
                            let test_ts = vec![
                                base_ts,           // Start of period
                                base_ts + 3600000, // +1 hour
                                base_ts + 7200000, // +2 hours
                            ];
                            let test_prices = vec![100.0, 200.0, 300.0];
                            let test_volumes = vec![1.0, 2.0, 3.0];

                            // Use 1d anchor to ensure all 3 are in same period
                            let test_params = VwapParams {
                                anchor: Some("1d".to_string()),
                            };
                            let test_input = VwapInput::from_slice(
                                &test_ts,
                                &test_volumes,
                                &test_prices,
                                test_params,
                            );

                            if let Ok(VwapOutput { values: test_out }) =
                                vwap_with_kernel(&test_input, kernel)
                            {
                                // VWAP is cumulative:
                                // At index 0: 100*1 / 1 = 100
                                // At index 1: (100*1 + 200*2) / (1+2) = 500/3 = 166.67...
                                // At index 2: (100*1 + 200*2 + 300*3) / (1+2+3) = 1400/6 = 233.33...

                                if test_out.len() >= 3 {
                                    if test_out[0].is_finite() {
                                        prop_assert!(
                                            (test_out[0] - 100.0).abs() < 1e-9,
                                            "VWAP at index 0 should be 100, got {}",
                                            test_out[0]
                                        );
                                    }
                                    if test_out[1].is_finite() {
                                        let expected_1 = 500.0 / 3.0;
                                        prop_assert!(
                                            (test_out[1] - expected_1).abs() < 1e-9,
                                            "VWAP at index 1 should be {}, got {}",
                                            expected_1,
                                            test_out[1]
                                        );
                                    }
                                    if test_out[2].is_finite() {
                                        let expected_2 = 1400.0 / 6.0;
                                        prop_assert!(
                                            (test_out[2] - expected_2).abs() < 1e-9,
                                            "VWAP at index 2 should be {}, got {}",
                                            expected_2,
                                            test_out[2]
                                        );
                                    }
                                }
                            }
                        }
                    }
                    (Err(e1), Err(e2)) => {
                        // Both kernels should fail with similar errors
                        prop_assert_eq!(
                            std::mem::discriminant(&e1),
                            std::mem::discriminant(&e2),
                            "Different error types: {:?} vs {:?}",
                            e1,
                            e2
                        );
                    }
                    _ => {
                        prop_assert!(
                            false,
                            "Kernel consistency failure: one succeeded, one failed"
                        );
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    generate_all_vwap_tests!(
        check_vwap_partial_params,
        check_vwap_accuracy,
        check_vwap_candles_plus_prices,
        check_vwap_anchor_parsing_error,
        check_vwap_slice_data_reinput,
        check_vwap_nan_handling,
        check_vwap_with_default_candles,
        check_vwap_with_default_params,
        check_vwap_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_vwap_tests!(check_vwap_property);
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let timestamps = c.get_timestamp().unwrap();
        let prices = c.get_calculated_field("hlc3").unwrap();
        let volumes = c.select_candle_field("volume").unwrap();

        let output = VwapBatchBuilder::new()
            .kernel(kernel)
            .apply_slice(timestamps, volumes, prices)?;

        let def = VwapParams::default();
        let row = output
            .combos
            .iter()
            .position(|p| p.anchor == def.anchor)
            .expect("default row missing");
        let row_values = &output.values[row * output.cols..(row + 1) * output.cols];

        assert_eq!(row_values.len(), c.close.len());

        let expected = [
            59353.05963230107,
            59330.15815713043,
            59289.94649532547,
            59274.6155462414,
            58730.0,
        ];
        let start = row_values.len() - 5;
        for (i, &v) in row_values[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-5,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
        Ok(())
    }

    fn check_batch_anchor_grid(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let timestamps = c.get_timestamp().unwrap();
        let prices = c.get_calculated_field("hlc3").unwrap();
        let volumes = c.select_candle_field("volume").unwrap();

        let batch = VwapBatchBuilder::new()
            .kernel(kernel)
            .anchor_range("1d", "3d", 1)
            .apply_slice(timestamps, volumes, prices)?;

        assert_eq!(batch.cols, c.close.len());
        assert!(batch.rows >= 1 && batch.rows <= 3);

        let anchors: Vec<_> = batch
            .combos
            .iter()
            .map(|p| p.anchor.clone().unwrap())
            .collect();
        assert_eq!(
            anchors,
            vec!["1d".to_string(), "2d".to_string(), "3d".to_string()]
        );
        Ok(())
    }

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]()      {
                    let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]()        {
                    let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]()      {
                    let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch);
                }
                #[test] fn [<$fn_name _auto_detect>]() {
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }
    // Check for poison values in batch output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let timestamps = c.get_timestamp().unwrap();
        let volumes = c.select_candle_field("volume").unwrap();

        // Test batch with multiple parameter range combinations
        let anchor_ranges = vec![
            ("1m", "5m", 1),  // Minute anchors
            ("1h", "6h", 1),  // Hour anchors
            ("1d", "5d", 1),  // Day anchors
            ("7d", "14d", 7), // Week-like anchors (using days)
        ];

        let test_sources = vec!["close", "open", "high", "low", "hl2", "hlc3", "ohlc4"];

        for (start, end, step) in anchor_ranges {
            for source in &test_sources {
                let prices = match *source {
                    "close" => c.select_candle_field("close").unwrap(),
                    "open" => c.select_candle_field("open").unwrap(),
                    "high" => c.select_candle_field("high").unwrap(),
                    "low" => c.select_candle_field("low").unwrap(),
                    _ => c.get_calculated_field(source).unwrap(),
                };

                let output = VwapBatchBuilder::new()
                    .kernel(kernel)
                    .anchor_range(start, end, step)
                    .apply_slice(timestamps, volumes, prices)?;

                // Check every value in the entire batch matrix for poison patterns
                for (idx, &val) in output.values.iter().enumerate() {
                    // Skip NaN values as they're expected in warmup periods
                    if val.is_nan() {
                        continue;
                    }

                    let bits = val.to_bits();
                    let row = idx / output.cols;
                    let col = idx % output.cols;

                    // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
                    if bits == 0x11111111_11111111 {
                        panic!(
                            "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with anchor_range({},{},{}) source={}",
                            test, val, bits, row, col, idx, start, end, step, source
                        );
                    }

                    // Check for init_matrix_prefixes poison (0x22222222_22222222)
                    if bits == 0x22222222_22222222 {
                        panic!(
                            "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) with anchor_range({},{},{}) source={}",
                            test, val, bits, row, col, idx, start, end, step, source
                        );
                    }

                    // Check for make_uninit_matrix poison (0x33333333_33333333)
                    if bits == 0x33333333_33333333 {
                        panic!(
                            "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with anchor_range({},{},{}) source={}",
                            test, val, bits, row, col, idx, start, end, step, source
                        );
                    }
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_anchor_grid);
    gen_batch_tests!(check_batch_no_poison);
}

#[cfg(feature = "python")]
#[pyfunction(name = "vwap")]
#[pyo3(signature = (timestamps, volumes, prices, anchor=None, kernel=None))]
/// Compute the Volume Weighted Average Price (VWAP) of the input data.
///
/// VWAP calculates the average price of a security weighted by traded volume,
/// grouped by the specified time anchor (e.g., "1d", "4h", "1M").
///
/// Parameters:
/// -----------
/// timestamps : np.ndarray
///     Array of Unix timestamps in milliseconds (int64).
/// volumes : np.ndarray
///     Array of trading volumes (float64).
/// prices : np.ndarray
///     Array of prices (float64).
/// anchor : str, optional
///     Time period for grouping (e.g., "1d", "4h", "1M"). Default is "1d".
/// kernel : str, optional
///     Computation kernel to use: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// np.ndarray
///     Array of VWAP values, same length as input.
///
/// Raises:
/// -------
/// ValueError
///     If array lengths don't match, anchor is invalid, or data is empty.
pub fn vwap_py<'py>(
    py: Python<'py>,
    timestamps: numpy::PyReadonlyArray1<'py, i64>,
    volumes: numpy::PyReadonlyArray1<'py, f64>,
    prices: numpy::PyReadonlyArray1<'py, f64>,
    anchor: Option<&str>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let ts_slice = timestamps.as_slice()?;
    let vol_slice = volumes.as_slice()?;
    let price_slice = prices.as_slice()?;

    // Parse and validate kernel before allow_threads
    let kern = crate::utilities::kernel_validation::validate_kernel(kernel, false)?;

    // Build input struct
    let params = VwapParams {
        anchor: anchor.map(|s| s.to_string()),
    };
    let vwap_in = VwapInput::from_slice(ts_slice, vol_slice, price_slice, params);

    // Get Vec<f64> from Rust function with all computation inside allow_threads
    let result_vec: Vec<f64> = py
        .allow_threads(|| vwap_with_kernel(&vwap_in, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Zero-copy transfer to NumPy
    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "VwapStream")]
pub struct VwapStreamPy {
    stream: VwapStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl VwapStreamPy {
    #[new]
    fn new(anchor: Option<&str>) -> PyResult<Self> {
        let params = VwapParams {
            anchor: anchor.map(|s| s.to_string()),
        };
        let stream =
            VwapStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(VwapStreamPy { stream })
    }

    /// Updates the stream with new values and returns the calculated VWAP value.
    /// Returns `None` if volume sum is zero.
    fn update(&mut self, timestamp: i64, price: f64, volume: f64) -> Option<f64> {
        self.stream.update(timestamp, price, volume)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "vwap_batch")]
#[pyo3(signature = (timestamps, volumes, prices, anchor_range, kernel=None))]
/// Compute VWAP for multiple anchor parameter combinations in a single pass.
///
/// Parameters:
/// -----------
/// timestamps : np.ndarray
///     Array of Unix timestamps in milliseconds (int64).
/// volumes : np.ndarray
///     Array of trading volumes (float64).
/// prices : np.ndarray
///     Array of prices (float64).
/// anchor_range : tuple
///     (start, end, step) for anchor values to compute (e.g., ("1d", "3d", 1)).
/// kernel : str, optional
///     Computation kernel: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// dict
///     Dictionary with 'values' (2D array) and 'anchors' array.
pub fn vwap_batch_py<'py>(
    py: Python<'py>,
    timestamps: numpy::PyReadonlyArray1<'py, i64>,
    volumes: numpy::PyReadonlyArray1<'py, f64>,
    prices: numpy::PyReadonlyArray1<'py, f64>,
    anchor_range: (String, String, u32),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let ts_slice = timestamps.as_slice()?;
    let vol_slice = volumes.as_slice()?;
    let price_slice = prices.as_slice()?;

    // Validate kernel before allow_threads
    let kern = crate::utilities::kernel_validation::validate_kernel(kernel, true)?;

    let sweep = VwapBatchRange {
        anchor: (anchor_range.0, anchor_range.1, anchor_range.2),
    };

    let combos = expand_grid_vwap(&sweep);
    let rows = combos.len();
    let cols = price_slice.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            let kernel = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };
            let simd = match kernel {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                _ => unreachable!(),
            };
            vwap_batch_inner_into(
                ts_slice,
                vol_slice,
                price_slice,
                &sweep,
                simd,
                true,
                slice_out,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;

    // Create Python list of anchor strings
    let anchors_list = PyList::new(
        py,
        combos
            .iter()
            .map(|p| p.anchor.clone().unwrap_or_else(|| "1d".to_string())),
    )?;
    dict.set_item("anchors", anchors_list)?;

    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "vwap_cuda_batch_dev")]
#[pyo3(signature = (timestamps, volumes, prices, anchor_range, device_id=0))]
pub fn vwap_cuda_batch_dev_py(
    py: Python<'_>,
    timestamps: numpy::PyReadonlyArray1<'_, i64>,
    volumes: numpy::PyReadonlyArray1<'_, f64>,
    prices: numpy::PyReadonlyArray1<'_, f64>,
    anchor_range: (String, String, u32),
    device_id: usize,
) -> PyResult<DeviceArrayF32VwapPy> {
    use numpy::PyArrayMethods;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let ts_slice = timestamps.as_slice()?;
    let vol_slice = volumes.as_slice()?;
    let price_slice = prices.as_slice()?;

    if ts_slice.len() != vol_slice.len() || vol_slice.len() != price_slice.len() {
        return Err(PyValueError::new_err(
            "timestamps, volumes, and prices must share the same length",
        ));
    }

    let (start, end, step) = anchor_range;
    let sweep = VwapBatchRange {
        anchor: (start, end, step),
    };

    let (inner, dev) = py
        .allow_threads(|| -> Result<_, crate::cuda::moving_averages::vwap_wrapper::CudaVwapError> {
            let cuda = CudaVwap::new(device_id)?;
            let arr = cuda.vwap_batch_dev_retaining_ctx(ts_slice, vol_slice, price_slice, &sweep)?;
            Ok((arr, cuda.device_id()))
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(DeviceArrayF32VwapPy { inner: Some(inner), device_id: dev })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "vwap_cuda_many_series_one_param_dev")]
#[pyo3(signature = (timestamps, prices_tm, volumes_tm, anchor, device_id=0))]
pub fn vwap_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    timestamps: numpy::PyReadonlyArray1<'_, i64>,
    prices_tm: numpy::PyReadonlyArray2<'_, f64>,
    volumes_tm: numpy::PyReadonlyArray2<'_, f64>,
    anchor: String,
    device_id: usize,
) -> PyResult<DeviceArrayF32VwapPy> {
    use numpy::PyArrayMethods;
    use numpy::PyUntypedArrayMethods;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let ts_slice = timestamps.as_slice()?;
    let p_shape = prices_tm.shape();
    let v_shape = volumes_tm.shape();
    if p_shape != v_shape {
        return Err(PyValueError::new_err(
            "prices_tm and volumes_tm shapes must match",
        ));
    }
    let rows = p_shape[0];
    let cols = p_shape[1];
    if ts_slice.len() != rows {
        return Err(PyValueError::new_err(
            "timestamps length must equal rows of matrices",
        ));
    }
    let prices_flat = prices_tm.as_slice()?;
    let volumes_flat = volumes_tm.as_slice()?;

    let (inner, dev) = py
        .allow_threads(|| -> Result<_, crate::cuda::moving_averages::vwap_wrapper::CudaVwapError> {
            let cuda = CudaVwap::new(device_id)?;
            let arr = cuda.vwap_many_series_one_param_time_major_dev_retaining_ctx(
                ts_slice,
                volumes_flat,
                prices_flat,
                cols,
                rows,
                &anchor,
            )?;
            Ok((arr, cuda.device_id()))
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(DeviceArrayF32VwapPy { inner: Some(inner), device_id: dev })
}

// ---------------- VWAP Python device handle (CAI v3 + DLPack) ----------------
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", name = "DeviceArrayF32Vwap", unsendable)]
pub struct DeviceArrayF32VwapPy {
    pub(crate) inner: Option<VwapDeviceArrayF32>,
    pub(crate) device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl DeviceArrayF32VwapPy {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("buffer already exported via __dlpack__"))?;
        let d = PyDict::new(py);
        d.set_item("shape", (inner.rows, inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item(
            "strides",
            (
                inner.cols * std::mem::size_of::<f32>(),
                std::mem::size_of::<f32>(),
            ),
        )?;
        let ptr_val: usize = if inner.rows == 0 || inner.cols == 0 { 0 } else { inner.buf.as_device_ptr().as_raw() as usize };
        d.set_item("data", (ptr_val, false))?;
        // Producing stream synchronized in wrapper; omit 'stream' per CAI v3
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> PyResult<(i32, i32)> {
        Ok((2, self.device_id as i32))
    }

    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<PyObject>,
        max_version: Option<PyObject>,
        dl_device: Option<PyObject>,
        copy: Option<PyObject>,
    ) -> PyResult<PyObject> {
        // 1. Determine and validate target device.
        let (kdl, alloc_dev) = self.__dlpack_device__()?;
        if let Some(dev_obj) = dl_device.as_ref() {
            if let Ok((dev_ty, dev_id)) = dev_obj.extract::<(i32, i32)>(py) {
                if dev_ty != kdl || dev_id != alloc_dev {
                    let wants_copy = copy
                        .as_ref()
                        .and_then(|c| c.extract::<bool>(py).ok())
                        .unwrap_or(false);
                    if wants_copy {
                        return Err(PyValueError::new_err(
                            "device copy not implemented for __dlpack__",
                        ));
                    } else {
                        return Err(PyValueError::new_err("dl_device mismatch for __dlpack__"));
                    }
                }
            }
        }
        let _ = stream;

        // 2. Move the underlying VRAM handle out; the DLPack capsule will own it.
        let inner = self
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("buffer already exported via __dlpack__"))?;

        let rows = inner.rows;
        let cols = inner.cols;
        let buf = inner.buf;

        // 3. Forward to the shared helper, preserving max_version negotiation.
        let max_version_bound = max_version.map(|obj| obj.into_bound(py));
        export_f32_cuda_dlpack_2d(py, buf, rows, cols, alloc_dev, max_version_bound)
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vwap_js(
    timestamps: &[f64], // Note: JS numbers are f64, we'll convert internally
    volumes: &[f64],
    prices: &[f64],
    anchor: Option<String>,
    kernel: Option<String>,
) -> Result<Vec<f64>, JsValue> {
    // Convert timestamps from f64 to i64 with validation
    let ts_i64: Vec<i64> = timestamps
        .iter()
        .map(|&t| {
            if t.is_nan() || t.is_infinite() || t < 0.0 {
                return Err(JsValue::from_str(&format!("Invalid timestamp: {}", t)));
            }
            Ok(t as i64)
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Parse kernel string to enum - following ALMA pattern
    let kern = match kernel.as_deref() {
        None | Some("auto") => Kernel::Auto, // Use Auto like ALMA
        Some("scalar") => Kernel::Scalar,
        Some("scalar_batch") => Kernel::ScalarBatch,
        Some(k) => return Err(JsValue::from_str(&format!("Unknown kernel: {}", k))),
    };

    let params = VwapParams { anchor };
    let input = VwapInput::from_slice(&ts_i64, volumes, prices, params);

    // Single allocation following ALMA pattern
    let mut output = vec![0.0; prices.len()];

    vwap_into_slice(&mut output, &input, kern).map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vwap_into(
    timestamps_ptr: *const f64,
    volumes_ptr: *const f64,
    prices_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    anchor: Option<String>,
) -> Result<(), JsValue> {
    if timestamps_ptr.is_null()
        || volumes_ptr.is_null()
        || prices_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str("null pointer passed to vwap_into"));
    }

    unsafe {
        let ts_f64 = std::slice::from_raw_parts(timestamps_ptr, len);
        let vols = std::slice::from_raw_parts(volumes_ptr, len);
        let pric = std::slice::from_raw_parts(prices_ptr, len);

        // convert timestamps to i64 with basic checks
        let mut ts_i64 = Vec::with_capacity(len);
        ts_i64.set_len(len);
        for i in 0..len {
            let t = ts_f64[i];
            if !t.is_finite() || t < 0.0 {
                return Err(JsValue::from_str("invalid timestamp"));
            }
            *ts_i64.get_unchecked_mut(i) = t as i64;
        }

        let params = VwapParams { anchor };
        let input = VwapInput::from_slice(&ts_i64, vols, pric, params);

        if core::ptr::eq(prices_ptr, out_ptr as *const f64) {
            // overlapping: compute to temp then copy
            let mut tmp = vec![0.0; len];
            vwap_into_slice(&mut tmp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&tmp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            vwap_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vwap_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vwap_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct VwapBatchConfig {
    pub anchor_range: (String, String, u32),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct VwapBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VwapParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "vwap_batch")]
pub fn vwap_batch_unified_js(
    timestamps: &[f64],
    volumes: &[f64],
    prices: &[f64],
    start: String,
    end: String,
    step: u32,
) -> Result<JsValue, JsValue> {
    // f64->i64 timestamps
    let ts_i64: Vec<i64> = timestamps
        .iter()
        .map(|&t| {
            if !t.is_finite() || t < 0.0 {
                return Err(JsValue::from_str("invalid timestamp"));
            }
            Ok(t as i64)
        })
        .collect::<Result<_, _>>()?;

    let sweep = VwapBatchRange {
        anchor: (start, end, step),
    };

    // Map Auto to concrete kernel to avoid unreachable!() in match
    let kernel = match detect_best_kernel() {
        Kernel::Auto => Kernel::Scalar,
        k => k,
    };

    let out = vwap_batch_inner(&ts_i64, volumes, prices, &sweep, kernel, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js = VwapBatchJsOutput {
        values: out.values,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    };
    serde_wasm_bindgen::to_value(&js).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vwap_batch_into(
    timestamps_ptr: *const f64,
    volumes_ptr: *const f64,
    prices_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    anchor_start: String,
    anchor_end: String,
    anchor_step: u32,
) -> Result<usize, JsValue> {
    if timestamps_ptr.is_null()
        || volumes_ptr.is_null()
        || prices_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str("null pointer passed to vwap_batch_into"));
    }

    unsafe {
        let timestamps_f64 = std::slice::from_raw_parts(timestamps_ptr, len);
        let volumes = std::slice::from_raw_parts(volumes_ptr, len);
        let prices = std::slice::from_raw_parts(prices_ptr, len);

        // Convert timestamps from f64 to i64
        let ts_i64: Vec<i64> = timestamps_f64.iter().map(|&t| t as i64).collect();

        let sweep = VwapBatchRange {
            anchor: (anchor_start, anchor_end, anchor_step),
        };

        let combos = expand_grid_vwap(&sweep);
        let rows = combos.len();
        let cols = len;

        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

        // Use the efficient batch_inner_into function
        vwap_batch_inner_into(
            &ts_i64,
            volumes,
            prices,
            &sweep,
            detect_best_kernel(),
            false,
            out,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vwap_batch_metadata_js(
    anchor_start: String,
    anchor_end: String,
    anchor_step: u32,
) -> Result<Vec<String>, JsValue> {
    let sweep = VwapBatchRange {
        anchor: (anchor_start, anchor_end, anchor_step),
    };

    let combos = expand_grid_vwap(&sweep);
    let metadata: Vec<String> = combos
        .iter()
        .map(|c| c.anchor.clone().unwrap_or_else(|| "1d".to_string()))
        .collect();

    Ok(metadata)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[deprecated(
    since = "1.0.0",
    note = "For anchor state reuse patterns, use the fast/unsafe API with persistent buffers"
)]
pub struct VwapContext {
    anchor: String,
    // In a real implementation, this would store anchor group state
    // For now, we just store the anchor pattern to match ALMA's API
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[allow(deprecated)]
impl VwapContext {
    #[wasm_bindgen(constructor)]
    #[deprecated(
        since = "1.0.0",
        note = "For anchor state reuse patterns, use the fast/unsafe API with persistent buffers"
    )]
    pub fn new(anchor: String) -> Result<VwapContext, JsValue> {
        if anchor.is_empty() {
            return Err(JsValue::from_str("Invalid anchor: empty string"));
        }

        // Validate anchor format
        let _ = parse_anchor(&anchor).map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(VwapContext { anchor })
    }

    pub fn update_into(
        &self,
        timestamps_ptr: *const f64,
        volumes_ptr: *const f64,
        prices_ptr: *const f64,
        out_ptr: *mut f64,
        len: usize,
    ) -> Result<(), JsValue> {
        if timestamps_ptr.is_null()
            || volumes_ptr.is_null()
            || prices_ptr.is_null()
            || out_ptr.is_null()
        {
            return Err(JsValue::from_str("null pointer passed to update_into"));
        }

        unsafe {
            let timestamps_f64 = std::slice::from_raw_parts(timestamps_ptr, len);
            let volumes = std::slice::from_raw_parts(volumes_ptr, len);
            let prices = std::slice::from_raw_parts(prices_ptr, len);

            // Convert timestamps from f64 to i64 with validation
            let ts_i64: Vec<i64> = timestamps_f64
                .iter()
                .map(|&t| {
                    if t.is_nan() || t.is_infinite() || t < 0.0 {
                        return Err(JsValue::from_str(&format!("Invalid timestamp: {}", t)));
                    }
                    Ok(t as i64)
                })
                .collect::<Result<Vec<_>, _>>()?;

            let params = VwapParams {
                anchor: Some(self.anchor.clone()),
            };
            let input = VwapInput::from_slice(&ts_i64, volumes, prices, params);

            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            vwap_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}
