//! # Bollinger Bands Indicator
//!
//! Volatility bands using a moving average and deviation (stddev or alternatives).
//!
//! ## Parameters
//! - **period**: MA window size (default: 20)
//! - **devup**: Upward deviation multiplier (default: 2.0)
//! - **devdn**: Downward deviation multiplier (default: 2.0)
//! - **matype**: Moving average type as string (default: "sma")
//! - **devtype**: Deviation type (0=stddev, 1=mean_ad, 2=median_ad; default: 0)
//!
//! ## Returns
//! - **BollingerBandsOutput** with upper, middle, and lower bands.
//! - Proper error types for invalid input, params, or kernel mismatch.
//!
//! ## Developer Notes
//! - **AVX2/AVX512 kernels**: Accelerate the final blend step (upper/middle/lower)
//!   for non-SMA (and SMA with non-stddev) by vectorizing the writes; fall back
//!   to the optimized scalar SMA+stddev path where loop-carried dependencies
//!   dominate and SIMD provides little benefit.
//! - **Streaming**: O(1) for SMA/EMA with stddev via ring buffer; generic
//!   fallback (mean_ad/median_ad or exotic MAs) uses an O(period) path without
//!   per-tick allocs. Matches batch outputs after warmup.
//! - **Memory optimization**: ✅ Uses alloc_with_nan_prefix (zero-copy) for all three bands
//! - **Batch operations**: ✅ Implemented with parallel processing support
//! - **Row-specific batch**: Not specialized for SMA+stddev; a future optimization is
//!   to share prefix sums across rows (periods/devups/devdns) when beneficial.
use crate::indicators::deviation::{deviation, DevInput, DevParams};
use crate::indicators::moving_averages::ma::{ma, MaData};
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum BollingerBandsData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

impl<'a> AsRef<[f64]> for BollingerBandsInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            BollingerBandsData::Slice(s) => s,
            BollingerBandsData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BollingerBandsOutput {
    pub upper_band: Vec<f64>,
    pub middle_band: Vec<f64>,
    pub lower_band: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(serde::Serialize, serde::Deserialize))]
pub struct BollingerBandsParams {
    pub period: Option<usize>,
    pub devup: Option<f64>,
    pub devdn: Option<f64>,
    pub matype: Option<String>,
    pub devtype: Option<usize>,
}

impl Default for BollingerBandsParams {
    fn default() -> Self {
        Self {
            period: Some(20),
            devup: Some(2.0),
            devdn: Some(2.0),
            matype: Some("sma".to_string()),
            devtype: Some(0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BollingerBandsInput<'a> {
    pub data: BollingerBandsData<'a>,
    pub params: BollingerBandsParams,
}

impl<'a> BollingerBandsInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: BollingerBandsParams) -> Self {
        Self {
            data: BollingerBandsData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: BollingerBandsParams) -> Self {
        Self {
            data: BollingerBandsData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", BollingerBandsParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(20)
    }
    #[inline]
    pub fn get_devup(&self) -> f64 {
        self.params.devup.unwrap_or(2.0)
    }
    #[inline]
    pub fn get_devdn(&self) -> f64 {
        self.params.devdn.unwrap_or(2.0)
    }
    #[inline]
    pub fn get_matype(&self) -> String {
        self.params.matype.clone().unwrap_or_else(|| "sma".into())
    }
    #[inline]
    pub fn get_devtype(&self) -> usize {
        self.params.devtype.unwrap_or(0)
    }
}

#[derive(Clone, Debug)]
pub struct BollingerBandsBuilder {
    period: Option<usize>,
    devup: Option<f64>,
    devdn: Option<f64>,
    matype: Option<String>,
    devtype: Option<usize>,
    kernel: Kernel,
}

impl Default for BollingerBandsBuilder {
    fn default() -> Self {
        Self {
            period: None,
            devup: None,
            devdn: None,
            matype: None,
            devtype: None,
            kernel: Kernel::Auto,
        }
    }
}

impl BollingerBandsBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn period(mut self, n: usize) -> Self {
        self.period = Some(n);
        self
    }
    #[inline(always)]
    pub fn devup(mut self, x: f64) -> Self {
        self.devup = Some(x);
        self
    }
    #[inline(always)]
    pub fn devdn(mut self, x: f64) -> Self {
        self.devdn = Some(x);
        self
    }
    #[inline(always)]
    pub fn matype(mut self, x: &str) -> Self {
        self.matype = Some(x.into());
        self
    }
    #[inline(always)]
    pub fn devtype(mut self, t: usize) -> Self {
        self.devtype = Some(t);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<BollingerBandsOutput, BollingerBandsError> {
        let p = BollingerBandsParams {
            period: self.period,
            devup: self.devup,
            devdn: self.devdn,
            matype: self.matype,
            devtype: self.devtype,
        };
        let i = BollingerBandsInput::from_candles(c, "close", p);
        bollinger_bands_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<BollingerBandsOutput, BollingerBandsError> {
        let p = BollingerBandsParams {
            period: self.period,
            devup: self.devup,
            devdn: self.devdn,
            matype: self.matype,
            devtype: self.devtype,
        };
        let i = BollingerBandsInput::from_slice(d, p);
        bollinger_bands_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<BollingerBandsStream, BollingerBandsError> {
        let p = BollingerBandsParams {
            period: self.period,
            devup: self.devup,
            devdn: self.devdn,
            matype: self.matype,
            devtype: self.devtype,
        };
        BollingerBandsStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum BollingerBandsError {
    #[error("bollinger_bands: Empty data provided.")]
    EmptyData,
    #[error("bollinger_bands: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("bollinger_bands: All values are NaN.")]
    AllValuesNaN,
    #[error("bollinger_bands: Underlying MA or Deviation function failed: {0}")]
    UnderlyingFunctionFailed(String),
    #[error("bollinger_bands: Not enough valid data for period: needed={needed}, valid={valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "bollinger_bands: Kernel must be a batch variant for batch operations. Got: {kernel:?}"
    )]
    KernelNotBatch { kernel: Kernel },
}

#[inline]
pub fn bollinger_bands(
    input: &BollingerBandsInput,
) -> Result<BollingerBandsOutput, BollingerBandsError> {
    bollinger_bands_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn bb_prepare<'a>(
    input: &'a BollingerBandsInput,
    kernel: Kernel,
) -> Result<
    (
        &'a [f64], // data
        usize,     // period
        f64,       // devup
        f64,       // devdn
        String,    // matype
        usize,     // devtype
        usize,     // first valid index
        Kernel,    // final kernel
    ),
    BollingerBandsError,
> {
    let data: &[f64] = input.as_ref();
    if data.is_empty() {
        return Err(BollingerBandsError::EmptyData);
    }

    let period = input.get_period();
    let devup = input.get_devup();
    let devdn = input.get_devdn();
    let matype = input.get_matype();
    let devtype = input.get_devtype();

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(BollingerBandsError::AllValuesNaN)?;

    if period == 0 || period > data.len() {
        return Err(BollingerBandsError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    if (data.len() - first) < period {
        return Err(BollingerBandsError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    Ok((data, period, devup, devdn, matype, devtype, first, chosen))
}

/// Advanced API: Compute Bollinger Bands directly into pre-allocated buffers.
///
/// This is a low-level API for power users who want to manage their own memory allocation.
/// The caller is responsible for:
/// - Ensuring output slices have the same length as input data
/// - Pre-filling output slices with NaN for the warmup period (0..first+period-1)
///
/// # Arguments
/// * `data` - Input price data
/// * `period` - MA window size
/// * `devup` - Upper band deviation multiplier
/// * `devdn` - Lower band deviation multiplier
/// * `matype` - Moving average type ("sma", "ema", etc.)
/// * `devtype` - Deviation type (0=stddev, 1=mean_ad, 2=median_ad)
/// * `first` - Index of first non-NaN value in data
/// * `kernel` - SIMD kernel to use
/// * `out_u` - Output buffer for upper band (must be same length as data)
/// * `out_m` - Output buffer for middle band (must be same length as data)
/// * `out_l` - Output buffer for lower band (must be same length as data)
///
/// # Safety
/// This function writes to indices [first+period-1..data.len()) in the output buffers.
/// Caller must ensure the buffers are properly sized and the warmup period is handled.
#[inline(always)]
pub fn bollinger_bands_compute_into(
    data: &[f64],
    period: usize,
    devup: f64,
    devdn: f64,
    matype: &str,
    devtype: usize,
    first: usize,
    kernel: Kernel,
    out_u: &mut [f64],
    out_m: &mut [f64],
    out_l: &mut [f64],
) -> Result<(), BollingerBandsError> {
    // Use classic kernel for SMA
    if matype == "sma" || matype == "SMA" {
        unsafe {
            bb_row_scalar_classic_sma(
                data, period, devtype, devup, devdn, first, out_u, out_m, out_l,
            );
        }
        return Ok(());
    }

    // Use SIMD row kernels when requested and available; otherwise, fall back to scalar.
    match kernel {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 => unsafe {
            bb_row_avx2(
                data, matype, period, devtype, devup, devdn, first, out_u, out_m, out_l,
            );
            Ok(())
        },
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 => unsafe {
            bb_row_avx512(
                data, matype, period, devtype, devup, devdn, first, out_u, out_m, out_l,
            );
            Ok(())
        },
        _ => {
            // Precompute middle and deviation once; jam the blend in a tight scalar loop.
            let middle = ma(matype, MaData::Slice(data), period)
                .map_err(|e| BollingerBandsError::UnderlyingFunctionFailed(e.to_string()))?;
            let dev_values = deviation(&DevInput::from_slice(
                data,
                DevParams {
                    period: Some(period),
                    devtype: Some(devtype),
                },
            ))
            .map_err(|e| BollingerBandsError::UnderlyingFunctionFailed(e.to_string()))?;

            let start = first + period - 1;
            let n = data.len();
            let mut i = start;
            while i < n {
                let m = unsafe { *middle.get_unchecked(i) };
                let d = unsafe { *dev_values.values.get_unchecked(i) };
                unsafe {
                    *out_m.get_unchecked_mut(i) = m;
                    *out_u.get_unchecked_mut(i) = devup.mul_add(d, m);
                    *out_l.get_unchecked_mut(i) = (-devdn).mul_add(d, m);
                }
                i += 1;
            }
            Ok(())
        }
    }
}

pub fn bollinger_bands_with_kernel(
    input: &BollingerBandsInput,
    kernel: Kernel,
) -> Result<BollingerBandsOutput, BollingerBandsError> {
    let (data, period, devup, devdn, matype, devtype, first, chosen) = bb_prepare(input, kernel)?;
    let warm = first + period - 1;

    let mut upper = alloc_with_nan_prefix(data.len(), warm);
    let mut middle = alloc_with_nan_prefix(data.len(), warm);
    let mut lower = alloc_with_nan_prefix(data.len(), warm);

    bollinger_bands_compute_into(
        data,
        period,
        devup,
        devdn,
        &matype,
        devtype,
        first,
        chosen,
        &mut upper,
        &mut middle,
        &mut lower,
    )?;

    Ok(BollingerBandsOutput {
        upper_band: upper,
        middle_band: middle,
        lower_band: lower,
    })
}

#[inline]
pub fn bollinger_bands_into_slices(
    out_u: &mut [f64],
    out_m: &mut [f64],
    out_l: &mut [f64],
    input: &BollingerBandsInput,
    kern: Kernel,
) -> Result<(), BollingerBandsError> {
    let (data, period, devup, devdn, matype, devtype, first, chosen) = bb_prepare(input, kern)?;

    if out_u.len() != data.len() || out_m.len() != data.len() || out_l.len() != data.len() {
        return Err(BollingerBandsError::InvalidPeriod {
            period: out_m.len(),
            data_len: data.len(),
        });
    }

    bollinger_bands_compute_into(
        data, period, devup, devdn, &matype, devtype, first, chosen, out_u, out_m, out_l,
    )?;

    let warm = first + period - 1;
    out_u[..warm].fill(f64::NAN);
    out_m[..warm].fill(f64::NAN);
    out_l[..warm].fill(f64::NAN);
    Ok(())
}

#[inline]
pub unsafe fn bollinger_bands_scalar(
    data: &[f64],
    matype: &str,
    ma_data: MaData,
    period: usize,
    devtype: usize,
    dev_input: DevInput,
    devup: f64,
    devdn: f64,
) -> Result<BollingerBandsOutput, BollingerBandsError> {
    let middle = ma(matype, ma_data, period)
        .map_err(|e| BollingerBandsError::UnderlyingFunctionFailed(e.to_string()))?;
    let dev_values = deviation(&dev_input)
        .map_err(|e| BollingerBandsError::UnderlyingFunctionFailed(e.to_string()))?;

    let first = data.iter().position(|x| !x.is_nan()).unwrap();
    let warmup_period = first + period - 1;
    let mut upper_band = alloc_with_nan_prefix(data.len(), warmup_period);
    let mut middle_band = alloc_with_nan_prefix(data.len(), warmup_period);
    let mut lower_band = alloc_with_nan_prefix(data.len(), warmup_period);
    for i in (first + period - 1)..data.len() {
        middle_band[i] = middle[i];
        upper_band[i] = middle[i] + devup * dev_values[i];
        lower_band[i] = middle[i] - devdn * dev_values[i];
    }
    Ok(BollingerBandsOutput {
        upper_band,
        middle_band,
        lower_band,
    })
}

// NOTE: AVX512 implementation currently delegates to scalar as a safe fallback.
// Future optimization: implement actual AVX512 SIMD operations.
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
pub unsafe fn bollinger_bands_avx512(
    data: &[f64],
    matype: &str,
    ma_data: MaData,
    period: usize,
    devtype: usize,
    dev_input: DevInput,
    devup: f64,
    devdn: f64,
) -> Result<BollingerBandsOutput, BollingerBandsError> {
    if period <= 32 {
        bollinger_bands_avx512_short(
            data, matype, ma_data, period, devtype, dev_input, devup, devdn,
        )
    } else {
        bollinger_bands_avx512_long(
            data, matype, ma_data, period, devtype, dev_input, devup, devdn,
        )
    }
}

// Stub implementation - delegates to scalar
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
pub unsafe fn bollinger_bands_avx512_short(
    data: &[f64],
    matype: &str,
    ma_data: MaData,
    period: usize,
    devtype: usize,
    dev_input: DevInput,
    devup: f64,
    devdn: f64,
) -> Result<BollingerBandsOutput, BollingerBandsError> {
    bollinger_bands_scalar(
        data, matype, ma_data, period, devtype, dev_input, devup, devdn,
    )
}

// Stub implementation - delegates to scalar
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
pub unsafe fn bollinger_bands_avx512_long(
    data: &[f64],
    matype: &str,
    ma_data: MaData,
    period: usize,
    devtype: usize,
    dev_input: DevInput,
    devup: f64,
    devdn: f64,
) -> Result<BollingerBandsOutput, BollingerBandsError> {
    bollinger_bands_scalar(
        data, matype, ma_data, period, devtype, dev_input, devup, devdn,
    )
}

// NOTE: AVX2 implementation currently delegates to scalar as a safe fallback.
// Future optimization: implement actual AVX2 SIMD operations.
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
pub unsafe fn bollinger_bands_avx2(
    data: &[f64],
    matype: &str,
    ma_data: MaData,
    period: usize,
    devtype: usize,
    dev_input: DevInput,
    devup: f64,
    devdn: f64,
) -> Result<BollingerBandsOutput, BollingerBandsError> {
    bollinger_bands_scalar(
        data, matype, ma_data, period, devtype, dev_input, devup, devdn,
    )
}

/// Classic kernel - optimized loop-jammed implementation for SMA + stddev (devtype=0).
/// Falls back to generic computation for non-stddev deviation types to preserve correctness.
///
/// Safety:
/// - Caller guarantees `out_*` slices have length == `data.len()`.
/// - Warm-up NaN prefix is handled by the caller (already done before invoking this).
unsafe fn bb_row_scalar_classic_sma(
    data: &[f64],
    period: usize,
    devtype: usize,
    devup: f64,
    devdn: f64,
    first: usize,
    out_u: &mut [f64],
    out_m: &mut [f64],
    out_l: &mut [f64],
) {
    let n = data.len();
    let warm = first + period - 1;
    if warm >= n {
        return;
    }

    // If deviation is not standard deviation (devtype != 0), use the generic path
    // to preserve correctness for mean_ad / median_ad.
    if devtype != 0 {
        let ma_data = MaData::Slice(data);
        let dev_input = DevInput::from_slice(
            data,
            DevParams {
                period: Some(period),
                devtype: Some(devtype),
            },
        );

        let middle = ma("sma", ma_data, period)
            .expect("MA(sma) computation failed in bb_row_scalar_classic_sma");
        let dev_values = deviation(&dev_input)
            .expect("Deviation computation failed in bb_row_scalar_classic_sma");

        let mut i = warm;
        while i < n {
            let m = *middle.get_unchecked(i);
            let d = *dev_values.values.get_unchecked(i);
            *out_m.get_unchecked_mut(i) = m;
            *out_u.get_unchecked_mut(i) = devup.mul_add(d, m);
            *out_l.get_unchecked_mut(i) = (-devdn).mul_add(d, m);
            i += 1;
        }
        return;
    }

    // ----------- Fast rolling SMA + stddev (devtype == 0) -----------
    let inv_n = 1.0 / (period as f64);
    let use_sample = false; // population variance per repo convention

    // Compute initial window sums using raw pointers to avoid bounds checks.
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;

    let mut p = data.as_ptr().add(first);
    for _ in 0..period {
        let v = unsafe { *p };
        sum += v;
        sum_sq = v.mul_add(v, sum_sq);
        p = unsafe { p.add(1) };
    }

    // Emit first valid point at index `warm`.
    let mean = sum * inv_n;
    // Population variance: Var = E[x^2] - (E[x])^2
    let var0 = if !use_sample {
        (-mean).mul_add(mean, sum_sq * inv_n)
    } else if period > 1 {
        (sum_sq - sum * mean) / ((period - 1) as f64)
    } else {
        0.0
    };
    let std0 = var0.max(0.0).sqrt();

    unsafe {
        *out_m.get_unchecked_mut(warm) = mean;
        *out_u.get_unchecked_mut(warm) = devup.mul_add(std0, mean);
        *out_l.get_unchecked_mut(warm) = (-devdn).mul_add(std0, mean);
    }

    // Rolling update with two moving pointers for old/new elements.
    let mut i = warm + 1;
    let mut p_new = unsafe { data.as_ptr().add(i) };
    let mut p_old = unsafe { data.as_ptr().add(i - period) };
    while i < n {
        let new = unsafe { *p_new };
        let old = unsafe { *p_old };

        // Update sum and sum of squares
        sum += new - old;
        let tmp = sum_sq - old * old;
        sum_sq = new.mul_add(new, tmp);

        let m = sum * inv_n;
        let var = if !use_sample {
            (-m).mul_add(m, sum_sq * inv_n)
        } else if period > 1 {
            (sum_sq - sum * m) / ((period - 1) as f64)
        } else {
            0.0
        };
        let sd = var.max(0.0).sqrt();

        unsafe {
            *out_m.get_unchecked_mut(i) = m;
            *out_u.get_unchecked_mut(i) = devup.mul_add(sd, m);
            *out_l.get_unchecked_mut(i) = (-devdn).mul_add(sd, m);
        }

        i += 1;
        p_new = unsafe { p_new.add(1) };
        p_old = unsafe { p_old.add(1) };
    }
}

/// Regular kernel - uses function calls for flexibility with any MA type,
/// but jams the final blend and uses pointer math + FMA for speed.
unsafe fn bb_row_scalar(
    data: &[f64],
    matype: &str,
    period: usize,
    devtype: usize,
    devup: f64,
    devdn: f64,
    first: usize,
    out_u: &mut [f64],
    out_m: &mut [f64],
    out_l: &mut [f64],
) {
    if matype.eq_ignore_ascii_case("sma") {
        bb_row_scalar_classic_sma(
            data, period, devtype, devup, devdn, first, out_u, out_m, out_l,
        );
        return;
    }

    let ma_data = MaData::Slice(data);
    let dev_input = DevInput::from_slice(
        data,
        DevParams {
            period: Some(period),
            devtype: Some(devtype),
        },
    );

    let middle = ma(matype, ma_data, period).expect("MA computation failed in bb_row_scalar");
    let dev_values = deviation(&dev_input).expect("Deviation computation failed in bb_row_scalar");

    let n = data.len();
    let warm = first + period - 1;
    if warm >= n {
        return;
    }

    let mut i = warm;
    while i < n {
        let m = *middle.get_unchecked(i);
        let d = *dev_values.values.get_unchecked(i);

        *out_m.get_unchecked_mut(i) = m;
        *out_u.get_unchecked_mut(i) = devup.mul_add(d, m);
        *out_l.get_unchecked_mut(i) = (-devdn).mul_add(d, m);

        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
unsafe fn bb_row_avx2(
    data: &[f64],
    matype: &str,
    period: usize,
    devtype: usize,
    devup: f64,
    devdn: f64,
    first: usize,
    out_u: &mut [f64],
    out_m: &mut [f64],
    out_l: &mut [f64],
) {
    use core::arch::x86_64::*;

    let n = data.len();
    let warm = first + period - 1;
    if warm >= n {
        return;
    }

    // For the SMA+stddev case, the scalar rolling kernel is optimal.
    if matype.eq_ignore_ascii_case("sma") && devtype == 0 {
        bb_row_scalar_classic_sma(
            data, period, devtype, devup, devdn, first, out_u, out_m, out_l,
        );
        return;
    }

    // Precompute middle & deviation with flexible engines.
    let ma_data = MaData::Slice(data);
    let dev_input = DevInput::from_slice(
        data,
        DevParams {
            period: Some(period),
            devtype: Some(devtype),
        },
    );
    let middle = ma(matype, ma_data, period).expect("MA computation failed in bb_row_avx2");
    let dev_values = deviation(&dev_input).expect("Deviation computation failed in bb_row_avx2");

    let du = unsafe { _mm256_set1_pd(devup) };
    let dd = unsafe { _mm256_set1_pd(devdn) };

    let mut i = warm;
    let step = 4usize;

    // Vectorized loop
    while i + step <= n {
        let m = unsafe { _mm256_loadu_pd(middle.as_ptr().add(i)) };
        let dv = unsafe { _mm256_loadu_pd(dev_values.values.as_ptr().add(i)) };

        // upper = m + devup * dv
        // lower = m - devdn * dv
        #[cfg(target_feature = "fma")]
        let up = unsafe { _mm256_fmadd_pd(du, dv, m) };
        #[cfg(not(target_feature = "fma"))]
        let up = unsafe { _mm256_add_pd(m, _mm256_mul_pd(du, dv)) };

        #[cfg(target_feature = "fma")]
        let lo = unsafe { _mm256_fnmadd_pd(dd, dv, m) }; // m - dd*dv
        #[cfg(not(target_feature = "fma"))]
        let lo = unsafe { _mm256_sub_pd(m, _mm256_mul_pd(dd, dv)) };

        unsafe {
            _mm256_storeu_pd(out_m.as_mut_ptr().add(i), m);
            _mm256_storeu_pd(out_u.as_mut_ptr().add(i), up);
            _mm256_storeu_pd(out_l.as_mut_ptr().add(i), lo);
        }

        i += step;
    }

    // Scalar tail
    while i < n {
        let m = unsafe { *middle.get_unchecked(i) };
        let d = unsafe { *dev_values.values.get_unchecked(i) };
        unsafe {
            *out_m.get_unchecked_mut(i) = m;
            *out_u.get_unchecked_mut(i) = devup.mul_add(d, m);
            *out_l.get_unchecked_mut(i) = (-devdn).mul_add(d, m);
        }
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
unsafe fn bb_row_avx512(
    data: &[f64],
    matype: &str,
    period: usize,
    devtype: usize,
    devup: f64,
    devdn: f64,
    first: usize,
    out_u: &mut [f64],
    out_m: &mut [f64],
    out_l: &mut [f64],
) {
    use core::arch::x86_64::*;

    let n = data.len();
    let warm = first + period - 1;
    if warm >= n {
        return;
    }

    // For the SMA+stddev case, the scalar rolling kernel is optimal.
    if matype.eq_ignore_ascii_case("sma") && devtype == 0 {
        bb_row_scalar_classic_sma(
            data, period, devtype, devup, devdn, first, out_u, out_m, out_l,
        );
        return;
    }

    // Precompute middle & deviation with flexible engines.
    let ma_data = MaData::Slice(data);
    let dev_input = DevInput::from_slice(
        data,
        DevParams {
            period: Some(period),
            devtype: Some(devtype),
        },
    );
    let middle = ma(matype, ma_data, period).expect("MA computation failed in bb_row_avx512");
    let dev_values = deviation(&dev_input).expect("Deviation computation failed in bb_row_avx512");

    let du = unsafe { _mm512_set1_pd(devup) };
    let dd = unsafe { _mm512_set1_pd(devdn) };

    let mut i = warm;
    let step = 8usize;

    while i + step <= n {
        let m = unsafe { _mm512_loadu_pd(middle.as_ptr().add(i)) };
        let dv = unsafe { _mm512_loadu_pd(dev_values.values.as_ptr().add(i)) };

        // upper = m + devup * dv
        // lower = m - devdn * dv
        #[cfg(target_feature = "fma")]
        let up = unsafe { _mm512_fmadd_pd(du, dv, m) };
        #[cfg(not(target_feature = "fma"))]
        let up = unsafe { _mm512_add_pd(m, _mm512_mul_pd(du, dv)) };

        #[cfg(target_feature = "fma")]
        let lo = unsafe { _mm512_fnmadd_pd(dd, dv, m) }; // m - dd*dv
        #[cfg(not(target_feature = "fma"))]
        let lo = unsafe { _mm512_sub_pd(m, _mm512_mul_pd(dd, dv)) };

        unsafe {
            _mm512_storeu_pd(out_m.as_mut_ptr().add(i), m);
            _mm512_storeu_pd(out_u.as_mut_ptr().add(i), up);
            _mm512_storeu_pd(out_l.as_mut_ptr().add(i), lo);
        }

        i += step;
    }

    // Scalar tail
    while i < n {
        let m = unsafe { *middle.get_unchecked(i) };
        let d = unsafe { *dev_values.values.get_unchecked(i) };
        unsafe {
            *out_m.get_unchecked_mut(i) = m;
            *out_u.get_unchecked_mut(i) = devup.mul_add(d, m);
            *out_l.get_unchecked_mut(i) = (-devdn).mul_add(d, m);
        }
        i += 1;
    }
}

#[derive(Debug, Clone)]
pub struct BollingerBandsStream {
    // public configuration (unchanged API)
    pub period: usize,
    pub devup: f64,
    pub devdn: f64,
    pub matype: String,
    pub devtype: usize,

    // --- internal state for O(1) fast paths ---
    buf: Vec<f64>, // ring buffer capacity == period
    idx: usize,    // write index (oldest element position)
    len: usize,    // number of valid samples in buf (<= period)
    sum: f64,      // Σ x
    sum_sq: f64,   // Σ x^2
    inv_n: f64,    // 1.0 / period (cached)

    // EMA state (only used when matype == "ema")
    alpha: f64, // 2.0 / (period + 1.0)
    ema: f64,
    ema_seeded: bool,

    // scratch for generic fallback path (re-used, no allocs per tick)
    scratch: Vec<f64>,

    // dispatch
    fast_path: FastPath,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum FastPath {
    SmaStddev,
    EmaStddev,
    Generic,
}

impl BollingerBandsStream {
    pub fn try_new(params: BollingerBandsParams) -> Result<Self, BollingerBandsError> {
        let period = params.period.unwrap_or(20);
        if period == 0 {
            return Err(BollingerBandsError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let devup = params.devup.unwrap_or(2.0);
        let devdn = params.devdn.unwrap_or(2.0);
        let matype = params.matype.unwrap_or_else(|| "sma".to_string());
        let devtype = params.devtype.unwrap_or(0);

        let mt_lc = matype.to_ascii_lowercase();
        let fast_path = if devtype == 0 && (mt_lc == "sma") {
            FastPath::SmaStddev
        } else if devtype == 0 && (mt_lc == "ema") {
            FastPath::EmaStddev
        } else {
            FastPath::Generic
        };

        let inv_n = 1.0 / (period as f64);
        let alpha = 2.0 / ((period as f64) + 1.0); // standard EMA smoothing
        Ok(Self {
            period,
            devup,
            devdn,
            matype,
            devtype,
            buf: vec![0.0; period],
            idx: 0,
            len: 0,
            sum: 0.0,
            sum_sq: 0.0,
            inv_n,
            alpha,
            ema: 0.0,
            ema_seeded: false,
            scratch: Vec::with_capacity(period),
            fast_path,
        })
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.idx = 0;
        self.len = 0;
        self.sum = 0.0;
        self.sum_sq = 0.0;
        self.ema = 0.0;
        self.ema_seeded = false;
        // buffer values may remain; len=0 ensures they are ignored until refilled
    }

    /// Build a contiguous (oldest..newest) view of the current window into `self.scratch`.
    /// Used only by the Generic fallback to call `ma()` / `deviation()`.
    #[inline(always)]
    fn window_contiguous<'a>(&'a mut self) -> &'a [f64] {
        self.scratch.clear();
        if self.len < self.period {
            // During warmup, just use what we have in direct order [0..len)
            self.scratch.extend_from_slice(&self.buf[..self.len]);
        } else {
            // Full window: [idx..end) then [0..idx)
            self.scratch.extend_from_slice(&self.buf[self.idx..]);
            if self.idx != 0 {
                self.scratch.extend_from_slice(&self.buf[..self.idx]);
            }
        }
        debug_assert_eq!(self.scratch.len(), self.len.min(self.period));
        &self.scratch
    }

    /// O(1) standard deviation (population) from running sums over the current window.
    #[inline(always)]
    fn stddev_current(&self) -> f64 {
        // Var = E[x^2] - (E[x])^2, with E[.] over the window, population variance (no Bessel's)
        let mean = self.sum * self.inv_n;
        let var = (self.sum_sq * self.inv_n) - mean * mean;
        // Numerically safe clamp against tiny negative due to FP error:
        let v = if var > 0.0 { var } else { 0.0 };
        v.sqrt()
    }

    /// Push one value into the window; updates sums and ring pointers in O(1).
    /// Returns (old, had_old) where `old` is the evicted sample when the window is full.
    #[inline(always)]
    fn push(&mut self, x: f64) -> (f64, bool) {
        if self.len < self.period {
            self.buf[self.idx] = x;
            self.idx += 1;
            if self.idx == self.period {
                self.idx = 0;
            }
            self.len += 1;
            self.sum += x;
            self.sum_sq = x.mul_add(x, self.sum_sq); // sum_sq += x*x
            (0.0, false)
        } else {
            // overwrite oldest (at idx)
            let old = self.buf[self.idx];
            self.buf[self.idx] = x;
            self.idx += 1;
            if self.idx == self.period {
                self.idx = 0;
            }

            // O(1) rolling updates
            self.sum += x - old;
            // sum_sq = sum_sq - old^2 + x^2  (use FMA where possible)
            let tmp = self.sum_sq - old * old;
            self.sum_sq = x.mul_add(x, tmp);
            (old, true)
        }
    }

    /// O(1) stream update. Returns (upper, middle, lower) once the window is full.
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64, f64)> {
        // Treat non-finite as a break in the series; this mirrors batch behavior
        // where computation starts at the first finite element.
        if !value.is_finite() {
            self.reset();
            return None;
        }

        // Insert and update ring/sums
        let _ = self.push(value);

        // Warm‑up: wait until we have a full window
        if self.len < self.period {
            return None;
        }

        match self.fast_path {
            FastPath::SmaStddev => {
                // middle = SMA, deviation = window stddev (population)
                let mean = self.sum * self.inv_n;
                let sd = self.stddev_current();
                let up = self.devup.mul_add(sd, mean);
                let lo = (-self.devdn).mul_add(sd, mean);
                Some((up, mean, lo))
            }
            FastPath::EmaStddev => {
                // deviation = window stddev (same as batch); middle = EMA seeded with first SMA
                if !self.ema_seeded {
                    self.ema = self.sum * self.inv_n; // seed EMA with SMA of first window
                    self.ema_seeded = true;
                } else {
                    // ema = α*new + (1-α)*ema
                    // (value is the newest sample that just entered the window)
                    self.ema = self.alpha.mul_add(value, (1.0 - self.alpha) * self.ema);
                }
                let sd = self.stddev_current();
                let up = self.devup.mul_add(sd, self.ema);
                let lo = (-self.devdn).mul_add(sd, self.ema);
                Some((up, self.ema, lo))
            }
            FastPath::Generic => {
                // Fall back to generic engines for non-stddev deviations or non-SMA/EMA MAs.
                // Still avoids O(period) shifting thanks to the ring; only one copy into scratch.
                let matype = self.matype.clone();
                let period = self.period;
                let devtype = self.devtype;
                let devup = self.devup;
                let devdn = self.devdn;
                let win = self.window_contiguous();

                // middle
                let mid = ma(&matype, MaData::Slice(win), period)
                    .ok()
                    .and_then(|v| v.last().copied())
                    .unwrap_or(f64::NAN);

                // deviation (devtype may be mean_ad or median_ad)
                let dev = deviation(&DevInput::from_slice(
                    win,
                    DevParams {
                        period: Some(period),
                        devtype: Some(devtype),
                    },
                ))
                .ok()
                .and_then(|v| v.values.last().copied())
                .unwrap_or(f64::NAN);

                let up = devup.mul_add(dev, mid);
                let lo = (-devdn).mul_add(dev, mid);
                Some((up, mid, lo))
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct BollingerBandsBatchRange {
    pub period: (usize, usize, usize),
    pub devup: (f64, f64, f64),
    pub devdn: (f64, f64, f64),
    pub matype: (String, String, usize),
    pub devtype: (usize, usize, usize),
}

impl Default for BollingerBandsBatchRange {
    fn default() -> Self {
        Self {
            period: (20, 20, 0),
            devup: (2.0, 2.0, 0.0),
            devdn: (2.0, 2.0, 0.0),
            matype: ("sma".to_string(), "sma".to_string(), 0),
            devtype: (0, 0, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct BollingerBandsBatchBuilder {
    range: BollingerBandsBatchRange,
    kernel: Kernel,
}

impl BollingerBandsBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }
    pub fn period_static(mut self, p: usize) -> Self {
        self.range.period = (p, p, 0);
        self
    }
    pub fn devup_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.devup = (start, end, step);
        self
    }
    pub fn devup_static(mut self, x: f64) -> Self {
        self.range.devup = (x, x, 0.0);
        self
    }
    pub fn devdn_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.devdn = (start, end, step);
        self
    }
    pub fn devdn_static(mut self, x: f64) -> Self {
        self.range.devdn = (x, x, 0.0);
        self
    }
    pub fn matype_static(mut self, m: &str) -> Self {
        self.range.matype = (m.into(), m.into(), 0);
        self
    }
    pub fn devtype_static(mut self, d: usize) -> Self {
        self.range.devtype = (d, d, 0);
        self
    }

    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<BollingerBandsBatchOutput, BollingerBandsError> {
        bollinger_bands_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(
        data: &[f64],
        k: Kernel,
    ) -> Result<BollingerBandsBatchOutput, BollingerBandsError> {
        Self::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(
        self,
        c: &Candles,
        src: &str,
    ) -> Result<BollingerBandsBatchOutput, BollingerBandsError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(
        c: &Candles,
    ) -> Result<BollingerBandsBatchOutput, BollingerBandsError> {
        Self::new().kernel(Kernel::Auto).apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct BollingerBandsBatchOutput {
    pub upper: Vec<f64>,
    pub middle: Vec<f64>,
    pub lower: Vec<f64>,
    pub combos: Vec<BollingerBandsParams>,
    pub rows: usize,
    pub cols: usize,
}
impl BollingerBandsBatchOutput {
    pub fn row_for_params(&self, p: &BollingerBandsParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period == p.period
                && (c.devup.unwrap_or(2.0) - p.devup.unwrap_or(2.0)).abs() < 1e-12
                && (c.devdn.unwrap_or(2.0) - p.devdn.unwrap_or(2.0)).abs() < 1e-12
                && c.matype == p.matype
                && c.devtype == p.devtype
        })
    }
    pub fn bands_for(&self, p: &BollingerBandsParams) -> Option<(&[f64], &[f64], &[f64])> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            (
                &self.upper[start..start + self.cols],
                &self.middle[start..start + self.cols],
                &self.lower[start..start + self.cols],
            )
        })
    }
}

fn expand_grid(r: &BollingerBandsBatchRange) -> Vec<BollingerBandsParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return vec![start];
        }
        let mut v = Vec::new();
        let mut x = start;
        while x <= end + 1e-12 {
            v.push(x);
            x += step;
        }
        v
    }
    fn axis_str((start, end, _step): (String, String, usize)) -> Vec<String> {
        if start == end {
            vec![start.clone()]
        } else {
            vec![start, end]
        }
    }

    let periods = axis_usize(r.period);
    let devups = axis_f64(r.devup);
    let devdns = axis_f64(r.devdn);
    let matypes = axis_str(r.matype.clone());
    let devtypes = axis_usize(r.devtype);

    let mut out = Vec::with_capacity(
        periods.len() * devups.len() * devdns.len() * matypes.len() * devtypes.len(),
    );
    for &p in &periods {
        for &u in &devups {
            for &d in &devdns {
                for m in &matypes {
                    for &t in &devtypes {
                        out.push(BollingerBandsParams {
                            period: Some(p),
                            devup: Some(u),
                            devdn: Some(d),
                            matype: Some(m.clone()),
                            devtype: Some(t),
                        });
                    }
                }
            }
        }
    }
    out
}

pub fn bollinger_bands_batch_with_kernel(
    data: &[f64],
    sweep: &BollingerBandsBatchRange,
    k: Kernel,
) -> Result<BollingerBandsBatchOutput, BollingerBandsError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(BollingerBandsError::KernelNotBatch { kernel: k });
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    bollinger_bands_batch_par_slice(data, sweep, simd)
}

pub fn bollinger_bands_batch_slice(
    data: &[f64],
    sweep: &BollingerBandsBatchRange,
    kern: Kernel,
) -> Result<BollingerBandsBatchOutput, BollingerBandsError> {
    bollinger_bands_batch_inner(data, sweep, kern, false)
}

pub fn bollinger_bands_batch_par_slice(
    data: &[f64],
    sweep: &BollingerBandsBatchRange,
    kern: Kernel,
) -> Result<BollingerBandsBatchOutput, BollingerBandsError> {
    bollinger_bands_batch_inner(data, sweep, kern, true)
}

fn bollinger_bands_batch_inner(
    data: &[f64],
    sweep: &BollingerBandsBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<BollingerBandsBatchOutput, BollingerBandsError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(BollingerBandsError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(BollingerBandsError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(BollingerBandsError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();

    // Allocate uninitialized matrices
    let mut upper_mu = make_uninit_matrix(rows, cols);
    let mut middle_mu = make_uninit_matrix(rows, cols);
    let mut lower_mu = make_uninit_matrix(rows, cols);

    // Calculate warmup periods for each row using single first
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();

    // Initialize NaN prefixes
    init_matrix_prefixes(&mut upper_mu, cols, &warm);
    init_matrix_prefixes(&mut middle_mu, cols, &warm);
    init_matrix_prefixes(&mut lower_mu, cols, &warm);

    // Convert to mutable slices for computation
    let mut upper_guard = ManuallyDrop::new(upper_mu);
    let mut middle_guard = ManuallyDrop::new(middle_mu);
    let mut lower_guard = ManuallyDrop::new(lower_mu);

    let upper: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(upper_guard.as_mut_ptr() as *mut f64, upper_guard.len())
    };
    let middle: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(middle_guard.as_mut_ptr() as *mut f64, middle_guard.len())
    };
    let lower: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(lower_guard.as_mut_ptr() as *mut f64, lower_guard.len())
    };

    let do_row = |row: usize, out_u: &mut [f64], out_m: &mut [f64], out_l: &mut [f64]| unsafe {
        let p = combos[row].period.unwrap();
        let du = combos[row].devup.unwrap();
        let dd = combos[row].devdn.unwrap();
        let mt = combos[row].matype.clone().unwrap();
        let dt = combos[row].devtype.unwrap();

        let ma_data = MaData::Slice(data);
        let dev_input = DevInput::from_slice(
            data,
            DevParams {
                period: Some(p),
                devtype: Some(dt),
            },
        );

        match kern {
            Kernel::Scalar => {
                let first = data.iter().position(|x| !x.is_nan()).unwrap();
                bb_row_scalar(data, &mt, p, dt, du, dd, first, out_u, out_m, out_l);
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => {
                let first = data.iter().position(|x| !x.is_nan()).unwrap();
                bb_row_avx2(data, &mt, p, dt, du, dd, first, out_u, out_m, out_l);
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => {
                let first = data.iter().position(|x| !x.is_nan()).unwrap();
                bb_row_avx512(data, &mt, p, dt, du, dd, first, out_u, out_m, out_l);
            }
            _ => unreachable!(),
        };
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            upper
                .par_chunks_mut(cols)
                .zip(middle.par_chunks_mut(cols))
                .zip(lower.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, ((u, m), l))| do_row(row, u, m, l));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, ((u, m), l)) in upper
                .chunks_mut(cols)
                .zip(middle.chunks_mut(cols))
                .zip(lower.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, u, m, l);
            }
        }
    } else {
        for (row, ((u, m), l)) in upper
            .chunks_mut(cols)
            .zip(middle.chunks_mut(cols))
            .zip(lower.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, u, m, l);
        }
    }

    // Reclaim as Vec<f64>
    let upper_vec = unsafe {
        Vec::from_raw_parts(
            upper_guard.as_mut_ptr() as *mut f64,
            upper_guard.len(),
            upper_guard.capacity(),
        )
    };
    let middle_vec = unsafe {
        Vec::from_raw_parts(
            middle_guard.as_mut_ptr() as *mut f64,
            middle_guard.len(),
            middle_guard.capacity(),
        )
    };
    let lower_vec = unsafe {
        Vec::from_raw_parts(
            lower_guard.as_mut_ptr() as *mut f64,
            lower_guard.len(),
            lower_guard.capacity(),
        )
    };

    Ok(BollingerBandsBatchOutput {
        upper: upper_vec,
        middle: middle_vec,
        lower: lower_vec,
        combos,
        rows,
        cols,
    })
}

/// Compute Bollinger Bands batch operation directly into pre-allocated buffers.
///
/// This function is optimized for Python/WASM bindings where we want to avoid
/// intermediate allocations. The output buffer must have size rows * cols where
/// rows is the number of parameter combinations and cols is the data length.
///
/// Returns the parameter combinations that were successfully computed.
#[inline(always)]
pub fn bollinger_bands_batch_inner_into(
    data: &[f64],
    sweep: &BollingerBandsBatchRange,
    kern: Kernel,
    parallel: bool,
    out_upper: &mut [f64],
    out_middle: &mut [f64],
    out_lower: &mut [f64],
) -> Result<Vec<BollingerBandsParams>, BollingerBandsError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(BollingerBandsError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(BollingerBandsError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(BollingerBandsError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let cols = data.len();

    let do_row = |row: usize, out_u: &mut [f64], out_m: &mut [f64], out_l: &mut [f64]| unsafe {
        let p = combos[row].period.unwrap();
        let du = combos[row].devup.unwrap();
        let dd = combos[row].devdn.unwrap();
        let mt = combos[row].matype.clone().unwrap();
        let dt = combos[row].devtype.unwrap();

        // Fill NaN prefix
        if first + p > 0 {
            let nan_end = (first + p - 1).min(data.len());
            out_u[..nan_end].fill(f64::NAN);
            out_m[..nan_end].fill(f64::NAN);
            out_l[..nan_end].fill(f64::NAN);
        }

        // Compute into buffers
        let _ = bollinger_bands_compute_into(
            data, p, du, dd, &mt, dt, first, kern, out_u, out_m, out_l,
        );
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_upper
                .par_chunks_mut(cols)
                .zip(out_middle.par_chunks_mut(cols))
                .zip(out_lower.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, ((u, m), l))| do_row(row, u, m, l));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, ((u, m), l)) in out_upper
                .chunks_mut(cols)
                .zip(out_middle.chunks_mut(cols))
                .zip(out_lower.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, u, m, l);
            }
        }
    } else {
        for (row, ((u, m), l)) in out_upper
            .chunks_mut(cols)
            .zip(out_middle.chunks_mut(cols))
            .zip(out_lower.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, u, m, l);
        }
    }

    Ok(combos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_bb_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let partial_params = BollingerBandsParams {
            period: Some(22),
            devup: None,
            devdn: None,
            matype: Some("sma".to_string()),
            devtype: None,
        };
        let input_partial = BollingerBandsInput::from_candles(&candles, "close", partial_params);
        let output_partial = bollinger_bands_with_kernel(&input_partial, kernel)?;
        assert_eq!(output_partial.upper_band.len(), candles.close.len());
        assert_eq!(output_partial.middle_band.len(), candles.close.len());
        assert_eq!(output_partial.lower_band.len(), candles.close.len());
        Ok(())
    }

    fn check_bb_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = BollingerBandsInput::with_default_candles(&candles);
        let result = bollinger_bands_with_kernel(&input, kernel)?;
        let expected_middle = [
            59403.199999999975,
            59423.24999999998,
            59370.49999999998,
            59371.39999999998,
            59351.299999999974,
        ];
        let expected_lower = [
            58299.51497247008,
            58351.47038179873,
            58332.65135978715,
            58334.33194052157,
            58275.767369163135,
        ];
        let expected_upper = [
            60506.88502752987,
            60495.029618201224,
            60408.348640212804,
            60408.468059478386,
            60426.83263083681,
        ];

        let start_idx = result.middle_band.len() - 5;
        for i in 0..5 {
            let actual_mid = result.middle_band[start_idx + i];
            let actual_low = result.lower_band[start_idx + i];
            let actual_up = result.upper_band[start_idx + i];
            assert!(
                (actual_mid - expected_middle[i]).abs() < 1e-4,
                "[{}] BB middle mismatch at i={}: {} vs {}",
                test_name,
                i,
                actual_mid,
                expected_middle[i]
            );
            assert!(
                (actual_low - expected_lower[i]).abs() < 1e-4,
                "[{}] BB lower mismatch at i={}: {} vs {}",
                test_name,
                i,
                actual_low,
                expected_lower[i]
            );
            assert!(
                (actual_up - expected_upper[i]).abs() < 1e-4,
                "[{}] BB upper mismatch at i={}: {} vs {}",
                test_name,
                i,
                actual_up,
                expected_upper[i]
            );
        }
        Ok(())
    }

    fn check_bb_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = BollingerBandsInput::with_default_candles(&candles);
        match input.data {
            BollingerBandsData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected BollingerBandsData::Candles"),
        }
        let output = bollinger_bands_with_kernel(&input, kernel)?;
        assert_eq!(output.middle_band.len(), candles.close.len());
        Ok(())
    }

    fn check_bb_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0];
        let params = BollingerBandsParams {
            period: Some(0),
            ..BollingerBandsParams::default()
        };
        let input = BollingerBandsInput::from_slice(&data, params);
        let res = bollinger_bands_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] BB should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_bb_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0];
        let params = BollingerBandsParams {
            period: Some(10),
            ..BollingerBandsParams::default()
        };
        let input = BollingerBandsInput::from_slice(&data, params);
        let res = bollinger_bands_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] BB should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_bb_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [42.0];
        let input = BollingerBandsInput::from_slice(&data, BollingerBandsParams::default());
        let res = bollinger_bands_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] BB should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_bb_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = BollingerBandsParams {
            period: Some(20),
            ..BollingerBandsParams::default()
        };
        let first_input = BollingerBandsInput::from_candles(&candles, "close", first_params);
        let first_result = bollinger_bands_with_kernel(&first_input, kernel)?;

        let second_params = BollingerBandsParams {
            period: Some(10),
            ..BollingerBandsParams::default()
        };
        let second_input =
            BollingerBandsInput::from_slice(&first_result.middle_band, second_params);
        let second_result = bollinger_bands_with_kernel(&second_input, kernel)?;

        assert_eq!(
            second_result.middle_band.len(),
            first_result.middle_band.len()
        );
        Ok(())
    }

    fn check_bb_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = BollingerBandsParams {
            period: Some(20),
            ..BollingerBandsParams::default()
        };
        let input = BollingerBandsInput::from_candles(&candles, "close", params);
        let result = bollinger_bands_with_kernel(&input, kernel)?;
        let check_index = 240;
        if result.middle_band.len() > check_index {
            for i in check_index..result.middle_band.len() {
                assert!(
                    !result.middle_band[i].is_nan(),
                    "[{}] BB NaN middle idx {}",
                    test_name,
                    i
                );
                assert!(
                    !result.upper_band[i].is_nan(),
                    "[{}] BB NaN upper idx {}",
                    test_name,
                    i
                );
                assert!(
                    !result.lower_band[i].is_nan(),
                    "[{}] BB NaN lower idx {}",
                    test_name,
                    i
                );
            }
        }
        Ok(())
    }

    fn check_bb_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = BollingerBandsParams::default();
        let period = params.period.unwrap_or(20);
        let devup = params.devup.unwrap_or(2.0);
        let devdn = params.devdn.unwrap_or(2.0);

        let input = BollingerBandsInput::from_candles(&candles, "close", params.clone());
        let batch_output = bollinger_bands_with_kernel(&input, kernel)?;

        let mut stream = BollingerBandsStream::try_new(params)?;
        let mut stream_upper = Vec::with_capacity(candles.close.len());
        let mut stream_middle = Vec::with_capacity(candles.close.len());
        let mut stream_lower = Vec::with_capacity(candles.close.len());
        for &v in &candles.close {
            match stream.update(v) {
                Some((up, mid, low)) => {
                    stream_upper.push(up);
                    stream_middle.push(mid);
                    stream_lower.push(low);
                }
                None => {
                    stream_upper.push(f64::NAN);
                    stream_middle.push(f64::NAN);
                    stream_lower.push(f64::NAN);
                }
            }
        }

        for (i, (bu, bm, bl)) in itertools::izip!(
            &batch_output.upper_band,
            &batch_output.middle_band,
            &batch_output.lower_band
        )
        .enumerate()
        {
            if bu.is_nan() && stream_upper[i].is_nan() {
                continue;
            }
            assert!(
                (*bu - stream_upper[i]).abs() < 1e-6,
                "[{}] BB stream/upper mismatch at idx {}",
                test_name,
                i
            );
            assert!(
                (*bm - stream_middle[i]).abs() < 1e-6,
                "[{}] BB stream/middle mismatch at idx {}",
                test_name,
                i
            );
            assert!(
                (*bl - stream_lower[i]).abs() < 1e-6,
                "[{}] BB stream/lower mismatch at idx {}",
                test_name,
                i
            );
        }
        Ok(())
    }

    // Check for poison values in single output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_bb_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with multiple parameter combinations to increase chance of catching bugs
        let params_list = vec![
            BollingerBandsParams::default(),
            BollingerBandsParams {
                period: Some(10),
                devup: Some(1.5),
                devdn: Some(1.5),
                matype: Some("ema".to_string()),
                devtype: Some(1),
            },
            BollingerBandsParams {
                period: Some(30),
                devup: Some(3.0),
                devdn: Some(2.0),
                matype: Some("sma".to_string()),
                devtype: Some(2),
            },
        ];

        for params in params_list {
            let input = BollingerBandsInput::from_candles(&candles, "close", params.clone());
            let output = bollinger_bands_with_kernel(&input, kernel)?;

            // Check all three bands for poison patterns
            for (band_name, band_data) in [
                ("upper", &output.upper_band),
                ("middle", &output.middle_band),
                ("lower", &output.lower_band),
            ] {
                for (i, &val) in band_data.iter().enumerate() {
                    // Skip NaN values as they're expected in the warmup period
                    if val.is_nan() {
                        continue;
                    }

                    let bits = val.to_bits();

                    // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
                    if bits == 0x11111111_11111111 {
                        panic!(
                            "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in {} band with params: period={}, devup={}, devdn={}, matype={}, devtype={}",
                            test_name, val, bits, i, band_name,
                            params.period.unwrap_or(20),
                            params.devup.unwrap_or(2.0),
                            params.devdn.unwrap_or(2.0),
                            params.matype.as_ref().unwrap_or(&"sma".to_string()),
                            params.devtype.unwrap_or(0)
                        );
                    }

                    // Check for init_matrix_prefixes poison (0x22222222_22222222)
                    if bits == 0x22222222_22222222 {
                        panic!(
                            "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in {} band with params: period={}, devup={}, devdn={}, matype={}, devtype={}",
                            test_name, val, bits, i, band_name,
                            params.period.unwrap_or(20),
                            params.devup.unwrap_or(2.0),
                            params.devdn.unwrap_or(2.0),
                            params.matype.as_ref().unwrap_or(&"sma".to_string()),
                            params.devtype.unwrap_or(0)
                        );
                    }

                    // Check for make_uninit_matrix poison (0x33333333_33333333)
                    if bits == 0x33333333_33333333 {
                        panic!(
                            "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in {} band with params: period={}, devup={}, devdn={}, matype={}, devtype={}",
                            test_name, val, bits, i, band_name,
                            params.period.unwrap_or(20),
                            params.devup.unwrap_or(2.0),
                            params.devdn.unwrap_or(2.0),
                            params.matype.as_ref().unwrap_or(&"sma".to_string()),
                            params.devtype.unwrap_or(0)
                        );
                    }
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_bb_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_bollinger_bands_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Note: This test currently only validates with SMA and standard deviation (devtype=0).
        // Future enhancement could test other MA types (EMA, WMA, etc.) and deviation types
        // (mean_ad, median_ad), each with their own specific invariants.

        // Generate test strategy: period, data, devup, devdn
        let strat = (1usize..=100).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    period..400,
                ),
                Just(period),
                0.0f64..5.0f64, // devup multiplier
                0.0f64..5.0f64, // devdn multiplier
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period, devup, devdn)| {
                // Create input with SMA as default (most common)
                let params = BollingerBandsParams {
                    period: Some(period),
                    devup: Some(devup),
                    devdn: Some(devdn),
                    matype: Some("sma".to_string()),
                    devtype: Some(0), // Standard deviation
                };
                let input = BollingerBandsInput::from_slice(&data, params);

                // Compute with specified kernel
                let result = bollinger_bands_with_kernel(&input, kernel).unwrap();
                let upper = &result.upper_band;
                let middle = &result.middle_band;
                let lower = &result.lower_band;

                // Also compute with scalar kernel for reference
                let ref_result = bollinger_bands_with_kernel(&input, Kernel::Scalar).unwrap();
                let ref_upper = &ref_result.upper_band;
                let ref_middle = &ref_result.middle_band;
                let ref_lower = &ref_result.lower_band;

                // Find first valid index
                let first_valid = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
                let warmup_end = first_valid + period - 1;

                // Property 1: Warmup period should be NaN
                for i in 0..warmup_end.min(data.len()) {
                    prop_assert!(upper[i].is_nan(), "Upper band should be NaN at idx {}", i);
                    prop_assert!(middle[i].is_nan(), "Middle band should be NaN at idx {}", i);
                    prop_assert!(lower[i].is_nan(), "Lower band should be NaN at idx {}", i);
                }

                // Test properties for valid output range
                for i in warmup_end..data.len() {
                    let u = upper[i];
                    let m = middle[i];
                    let l = lower[i];

                    // Property 2: Band ordering (when bands are finite and devup/devdn >= 0)
                    if u.is_finite()
                        && m.is_finite()
                        && l.is_finite()
                        && devup >= 0.0
                        && devdn >= 0.0
                    {
                        prop_assert!(
                            u >= m - 1e-9,
                            "Upper band {} should be >= middle band {} at idx {}",
                            u,
                            m,
                            i
                        );
                        prop_assert!(
                            m >= l - 1e-9,
                            "Middle band {} should be >= lower band {} at idx {}",
                            m,
                            l,
                            i
                        );
                    }

                    // Property 3: Middle band should be within window range
                    if m.is_finite() {
                        let window_start = i.saturating_sub(period - 1);
                        let window = &data[window_start..=i];
                        let valid_values: Vec<f64> =
                            window.iter().filter(|x| x.is_finite()).copied().collect();

                        if !valid_values.is_empty() {
                            let window_min =
                                valid_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                            let window_max = valid_values
                                .iter()
                                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                            // For period=1, use larger tolerance due to floating-point precision
                            let tolerance = if period == 1 { 1e-6 } else { 1e-9 };

                            // Middle band (moving average) should be within data range
                            prop_assert!(
                                m >= window_min - tolerance && m <= window_max + tolerance,
                                "Middle band {} not in window range [{}, {}] at idx {}",
                                m,
                                window_min,
                                window_max,
                                i
                            );
                        }
                    }

                    // Property 4: Cross-kernel consistency
                    let ru = ref_upper[i];
                    let rm = ref_middle[i];
                    let rl = ref_lower[i];

                    // Check upper band consistency
                    if u.is_finite() && ru.is_finite() {
                        let ulp_diff = u.to_bits().abs_diff(ru.to_bits());
                        prop_assert!(
                            (u - ru).abs() <= 1e-9 || ulp_diff <= 4,
                            "Upper band mismatch at idx {}: {} vs {} (ULP={})",
                            i,
                            u,
                            ru,
                            ulp_diff
                        );
                    } else {
                        prop_assert_eq!(u.is_nan(), ru.is_nan(), "Upper NaN mismatch at idx {}", i);
                    }

                    // Check middle band consistency
                    if m.is_finite() && rm.is_finite() {
                        let ulp_diff = m.to_bits().abs_diff(rm.to_bits());
                        prop_assert!(
                            (m - rm).abs() <= 1e-9 || ulp_diff <= 4,
                            "Middle band mismatch at idx {}: {} vs {} (ULP={})",
                            i,
                            m,
                            rm,
                            ulp_diff
                        );
                    } else {
                        prop_assert_eq!(
                            m.is_nan(),
                            rm.is_nan(),
                            "Middle NaN mismatch at idx {}",
                            i
                        );
                    }

                    // Check lower band consistency
                    if l.is_finite() && rl.is_finite() {
                        let ulp_diff = l.to_bits().abs_diff(rl.to_bits());
                        prop_assert!(
                            (l - rl).abs() <= 1e-9 || ulp_diff <= 4,
                            "Lower band mismatch at idx {}: {} vs {} (ULP={})",
                            i,
                            l,
                            rl,
                            ulp_diff
                        );
                    } else {
                        prop_assert_eq!(l.is_nan(), rl.is_nan(), "Lower NaN mismatch at idx {}", i);
                    }
                }

                // Property 5: Period=2 edge case (smallest valid period for stddev)
                if period == 2 && warmup_end < data.len() {
                    // With period=2, we can validate basic properties
                    for i in warmup_end..data.len() {
                        if upper[i].is_finite() && middle[i].is_finite() && lower[i].is_finite() {
                            // Bands should still follow ordering
                            prop_assert!(
                                upper[i] >= middle[i] - 1e-9,
                                "Period=2: upper band should be >= middle at idx {}",
                                i
                            );
                            prop_assert!(
                                middle[i] >= lower[i] - 1e-9,
                                "Period=2: middle band should be >= lower at idx {}",
                                i
                            );
                        }
                    }
                }

                // Property 6: Constant data property
                let all_same = data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12);
                if all_same && data.len() > 0 && data[0].is_finite() {
                    let const_val = data[0];
                    for i in warmup_end..data.len() {
                        if upper[i].is_finite() {
                            // With constant data, std dev = 0, so all bands equal the constant
                            prop_assert!(
                                (upper[i] - const_val).abs() <= 1e-9,
                                "Constant data: upper band should be {} at idx {}",
                                const_val,
                                i
                            );
                            prop_assert!(
                                (middle[i] - const_val).abs() <= 1e-9,
                                "Constant data: middle band should be {} at idx {}",
                                const_val,
                                i
                            );
                            prop_assert!(
                                (lower[i] - const_val).abs() <= 1e-9,
                                "Constant data: lower band should be {} at idx {}",
                                const_val,
                                i
                            );
                        }
                    }
                }

                // Property 7: Band width relationship
                if devup > 0.0 && devdn > 0.0 && warmup_end < data.len() {
                    for i in warmup_end..data.len() {
                        if upper[i].is_finite() && middle[i].is_finite() && lower[i].is_finite() {
                            let upper_width = upper[i] - middle[i];
                            let lower_width = middle[i] - lower[i];

                            // Band widths should be proportional to deviation multipliers
                            if upper_width > 1e-12 && lower_width > 1e-12 {
                                let ratio = (upper_width / lower_width) / (devup / devdn);
                                prop_assert!(
                                    (ratio - 1.0).abs() <= 1e-4,
                                    "Band width ratio mismatch at idx {}: expected {}, got {}",
                                    i,
                                    devup / devdn,
                                    upper_width / lower_width
                                );
                            }
                        }
                    }
                }

                // Property 8: Symmetry with equal multipliers
                if (devup - devdn).abs() < 1e-12 && devup > 0.0 {
                    for i in warmup_end..data.len() {
                        if upper[i].is_finite() && middle[i].is_finite() && lower[i].is_finite() {
                            let upper_dist = upper[i] - middle[i];
                            let lower_dist = middle[i] - lower[i];
                            prop_assert!(
                                (upper_dist - lower_dist).abs() <= 1e-9,
                                "Bands should be symmetric at idx {}: upper_dist={}, lower_dist={}",
                                i,
                                upper_dist,
                                lower_dist
                            );
                        }
                    }
                }

                // Property 9: Finite output for finite input
                let all_finite = data.iter().all(|x| x.is_finite());
                if all_finite {
                    for i in warmup_end..data.len() {
                        prop_assert!(
                            upper[i].is_finite(),
                            "Upper band should be finite at idx {} with finite input",
                            i
                        );
                        prop_assert!(
                            middle[i].is_finite(),
                            "Middle band should be finite at idx {} with finite input",
                            i
                        );
                        prop_assert!(
                            lower[i].is_finite(),
                            "Lower band should be finite at idx {} with finite input",
                            i
                        );
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_bb_tests {
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

    generate_all_bb_tests!(
        check_bb_partial_params,
        check_bb_accuracy,
        check_bb_default_candles,
        check_bb_zero_period,
        check_bb_period_exceeds_length,
        check_bb_very_small_dataset,
        check_bb_reinput,
        check_bb_nan_handling,
        check_bb_streaming,
        check_bb_no_poison
    );

    // Generate property tests only when proptest feature is enabled
    #[cfg(feature = "proptest")]
    generate_all_bb_tests!(check_bollinger_bands_property);

    #[test]
    fn test_kernel_not_batch_error() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sweep = BollingerBandsBatchRange::default();

        // Test that a non-batch kernel returns the correct error
        let result = bollinger_bands_batch_with_kernel(&data, &sweep, Kernel::Scalar);
        assert!(matches!(
            result,
            Err(BollingerBandsError::KernelNotBatch {
                kernel: Kernel::Scalar
            })
        ));

        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        {
            let result = bollinger_bands_batch_with_kernel(&data, &sweep, Kernel::Avx2);
            assert!(matches!(
                result,
                Err(BollingerBandsError::KernelNotBatch {
                    kernel: Kernel::Avx2
                })
            ));
        }
    }

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = BollingerBandsBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = BollingerBandsParams::default();
        let (_up, mid, _low) = output.bands_for(&def).expect("default row missing");

        assert_eq!(mid.len(), c.close.len());
        Ok(())
    }

    // Check for poison values in batch output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test batch with multiple parameter combinations
        let output = BollingerBandsBatchBuilder::new()
            .kernel(kernel)
            .period_range(10, 30, 10) // Tests periods: 10, 20, 30
            .devup_range(1.0, 3.0, 1.0) // Tests devup: 1.0, 2.0, 3.0
            .devdn_range(1.0, 2.0, 0.5) // Tests devdn: 1.0, 1.5, 2.0
            .matype_static("sma")
            .devtype_static(0)
            .apply_candles(&c, "close")?;

        // Check every value in all three batch matrices for poison patterns
        for (band_name, band_data) in [
            ("upper", &output.upper),
            ("middle", &output.middle),
            ("lower", &output.lower),
        ] {
            for (idx, &val) in band_data.iter().enumerate() {
                // Skip NaN values as they're expected in warmup periods
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let params = &output.combos[row];

                // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) in {} band. Params: period={}, devup={}, devdn={}",
                        test, val, bits, row, col, idx, band_name,
                        params.period.unwrap_or(20),
                        params.devup.unwrap_or(2.0),
                        params.devdn.unwrap_or(2.0)
                    );
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) in {} band. Params: period={}, devup={}, devdn={}",
                        test, val, bits, row, col, idx, band_name,
                        params.period.unwrap_or(20),
                        params.devup.unwrap_or(2.0),
                        params.devdn.unwrap_or(2.0)
                    );
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) in {} band. Params: period={}, devup={}, devdn={}",
                        test, val, bits, row, col, idx, band_name,
                        params.period.unwrap_or(20),
                        params.devup.unwrap_or(2.0),
                        params.devdn.unwrap_or(2.0)
                    );
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(
        _test: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
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
    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}

#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
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

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct BollingerJsResult {
    values: Vec<f64>, // [upper..., middle..., lower...]
    rows: usize,      // 3
    cols: usize,      // data length
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl BollingerJsResult {
    #[wasm_bindgen(getter)]
    pub fn values(&self) -> Vec<f64> {
        self.values.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[wasm_bindgen(getter)]
    pub fn cols(&self) -> usize {
        self.cols
    }
}

#[cfg(feature = "wasm")]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct BollingerBatchJsOutput {
    pub upper: Vec<f64>,
    pub middle: Vec<f64>,
    pub lower: Vec<f64>,
    pub combos: Vec<BollingerBandsParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "python")]
#[pyfunction(name = "bollinger_bands")]
#[pyo3(signature = (data, period=20, devup=2.0, devdn=2.0, matype="sma", devtype=0, kernel=None))]
pub fn bollinger_bands_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    devup: f64,
    devdn: f64,
    matype: &str,
    devtype: usize,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, numpy::PyArray1<f64>>,
    Bound<'py, numpy::PyArray1<f64>>,
    Bound<'py, numpy::PyArray1<f64>>,
)> {
    use numpy::PyArrayMethods;

    let slice_in = data.as_slice()?;
    let n = slice_in.len();

    let out_u = unsafe { numpy::PyArray1::<f64>::new(py, [n], false) };
    let out_m = unsafe { numpy::PyArray1::<f64>::new(py, [n], false) };
    let out_l = unsafe { numpy::PyArray1::<f64>::new(py, [n], false) };

    let kern = validate_kernel(kernel, false)?;
    let input = BollingerBandsInput::from_slice(
        slice_in,
        BollingerBandsParams {
            period: Some(period),
            devup: Some(devup),
            devdn: Some(devdn),
            matype: Some(matype.to_string()),
            devtype: Some(devtype),
        },
    );

    {
        let u = unsafe { out_u.as_slice_mut()? };
        let m = unsafe { out_m.as_slice_mut()? };
        let l = unsafe { out_l.as_slice_mut()? };
        py.allow_threads(|| bollinger_bands_into_slices(u, m, l, &input, kern))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    }
    Ok((out_u, out_m, out_l))
}

#[cfg(feature = "python")]
#[pyclass(name = "BollingerBandsStream")]
pub struct BollingerBandsStreamPy {
    stream: BollingerBandsStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl BollingerBandsStreamPy {
    #[new]
    #[pyo3(signature = (period=20, devup=2.0, devdn=2.0, matype="sma", devtype=0))]
    fn new(period: usize, devup: f64, devdn: f64, matype: &str, devtype: usize) -> PyResult<Self> {
        let params = BollingerBandsParams {
            period: Some(period),
            devup: Some(devup),
            devdn: Some(devdn),
            matype: Some(matype.to_string()),
            devtype: Some(devtype),
        };
        let stream = BollingerBandsStream::try_new(params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(BollingerBandsStreamPy { stream })
    }

    /// Updates the stream with a new value and returns the calculated Bollinger Bands.
    /// Returns (upper, middle, lower) tuple or None if the buffer is not yet full.
    fn update(&mut self, value: f64) -> Option<(f64, f64, f64)> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "bollinger_bands_batch")]
#[pyo3(signature = (data, period_range=(20, 20, 0), devup_range=(2.0, 2.0, 0.0), devdn_range=(2.0, 2.0, 0.0), matype="sma", devtype_range=(0,0,0), kernel=None))]
pub fn bollinger_bands_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    devup_range: (f64, f64, f64),
    devdn_range: (f64, f64, f64),
    matype: &str,
    devtype_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let sweep = BollingerBandsBatchRange {
        period: period_range,
        devup: devup_range,
        devdn: devdn_range,
        matype: (matype.to_string(), matype.to_string(), 0),
        devtype: devtype_range,
    };

    // 1. Expand grid once to know rows*cols
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    // 2. Pre-allocate uninitialized NumPy arrays (1-D, will reshape later)
    // NOTE: PyArray1::new() creates uninitialized memory, not zero-initialized
    let upper_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let middle_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let lower_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };

    let slice_upper = unsafe { upper_arr.as_slice_mut()? };
    let slice_middle = unsafe { middle_arr.as_slice_mut()? };
    let slice_lower = unsafe { lower_arr.as_slice_mut()? };

    // Use kernel validation for safety
    let kern = validate_kernel(kernel, true)?;

    // 3. Heavy work without the GIL
    let combos_clone = py
        .allow_threads(
            || -> Result<Vec<BollingerBandsParams>, BollingerBandsError> {
                // Resolve Kernel::Auto to a specific kernel
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

                // Use the new function that writes directly to output buffers
                bollinger_bands_batch_inner_into(
                    slice_in,
                    &sweep,
                    simd,
                    true, // parallel
                    slice_upper,
                    slice_middle,
                    slice_lower,
                )
            },
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // 4. Build dict with the GIL
    let dict = PyDict::new(py);
    dict.set_item("upper", upper_arr.reshape((rows, cols))?)?;
    dict.set_item("middle", middle_arr.reshape((rows, cols))?)?;
    dict.set_item("lower", lower_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "periods",
        combos_clone
            .iter()
            .map(|p| p.period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "devups",
        combos_clone
            .iter()
            .map(|p| p.devup.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "devdns",
        combos_clone
            .iter()
            .map(|p| p.devdn.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "matypes",
        combos_clone
            .iter()
            .map(|p| p.matype.as_ref().unwrap().clone())
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "devtypes",
        combos_clone
            .iter()
            .map(|p| p.devtype.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn bollinger_bands_js(
    data: &[f64],
    period: usize,
    devup: f64,
    devdn: f64,
    matype: String,
    devtype: usize,
) -> Result<BollingerJsResult, JsValue> {
    let input = BollingerBandsInput::from_slice(
        data,
        BollingerBandsParams {
            period: Some(period),
            devup: Some(devup),
            devdn: Some(devdn),
            matype: Some(matype),
            devtype: Some(devtype),
        },
    );

    let mut u = vec![0.0; data.len()];
    let mut m = vec![0.0; data.len()];
    let mut l = vec![0.0; data.len()];
    bollinger_bands_into_slices(&mut u, &mut m, &mut l, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let mut values = u;
    values.extend_from_slice(&m);
    values.extend_from_slice(&l);
    Ok(BollingerJsResult {
        values,
        rows: 3,
        cols: data.len(),
    })
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn bollinger_bands_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
    devup_start: f64,
    devup_end: f64,
    devup_step: f64,
    devdn_start: f64,
    devdn_end: f64,
    devdn_step: f64,
    matype: &str,
    devtype: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = BollingerBandsBatchRange {
        period: (period_start, period_end, period_step),
        devup: (devup_start, devup_end, devup_step),
        devdn: (devdn_start, devdn_end, devdn_step),
        matype: (matype.to_string(), matype.to_string(), 0),
        devtype: (devtype, devtype, 0),
    };

    // Expand grid and allocate output vectors
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = data.len();

    // Find first valid index
    let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);

    // Allocate uninitialized matrices
    let mut upper_mu = make_uninit_matrix(rows, cols);
    let mut middle_mu = make_uninit_matrix(rows, cols);
    let mut lower_mu = make_uninit_matrix(rows, cols);

    // Calculate warmup periods for each row
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();

    // Initialize NaN prefixes
    init_matrix_prefixes(&mut upper_mu, cols, &warm);
    init_matrix_prefixes(&mut middle_mu, cols, &warm);
    init_matrix_prefixes(&mut lower_mu, cols, &warm);

    // Convert to mutable slices
    let mut upper_guard = ManuallyDrop::new(upper_mu);
    let mut middle_guard = ManuallyDrop::new(middle_mu);
    let mut lower_guard = ManuallyDrop::new(lower_mu);

    let upper_vec: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(upper_guard.as_mut_ptr() as *mut f64, upper_guard.len())
    };
    let middle_vec: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(middle_guard.as_mut_ptr() as *mut f64, middle_guard.len())
    };
    let lower_vec: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(lower_guard.as_mut_ptr() as *mut f64, lower_guard.len())
    };

    // Process each row
    for (i, combo) in combos.iter().enumerate() {
        let row_start = i * cols;
        let out_u = &mut upper_vec[row_start..row_start + cols];
        let out_m = &mut middle_vec[row_start..row_start + cols];
        let out_l = &mut lower_vec[row_start..row_start + cols];

        let p = combo.period.unwrap();
        let du = combo.devup.unwrap();
        let dd = combo.devdn.unwrap();
        let mt = combo.matype.as_ref().unwrap();
        let dt = combo.devtype.unwrap();

        // Fill NaN prefix if needed
        if first + p > 0 {
            let nan_end = (first + p - 1).min(cols);
            out_u[..nan_end].fill(f64::NAN);
            out_m[..nan_end].fill(f64::NAN);
            out_l[..nan_end].fill(f64::NAN);
        }

        // Compute into buffers
        bollinger_bands_compute_into(
            data,
            p,
            du,
            dd,
            mt,
            dt,
            first,
            Kernel::Scalar,
            out_u,
            out_m,
            out_l,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    // Reclaim as Vec<f64>
    let upper_vec = unsafe {
        Vec::from_raw_parts(
            upper_guard.as_mut_ptr() as *mut f64,
            upper_guard.len(),
            upper_guard.capacity(),
        )
    };
    let middle_vec = unsafe {
        Vec::from_raw_parts(
            middle_guard.as_mut_ptr() as *mut f64,
            middle_guard.len(),
            middle_guard.capacity(),
        )
    };
    let lower_vec = unsafe {
        Vec::from_raw_parts(
            lower_guard.as_mut_ptr() as *mut f64,
            lower_guard.len(),
            lower_guard.capacity(),
        )
    };

    // Flatten into single output vector
    let mut result = upper_vec;
    result.extend(middle_vec);
    result.extend(lower_vec);
    Ok(result)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn bollinger_bands_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
    devup_start: f64,
    devup_end: f64,
    devup_step: f64,
    devdn_start: f64,
    devdn_end: f64,
    devdn_step: f64,
    matype: &str,
    devtype: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = BollingerBandsBatchRange {
        period: (period_start, period_end, period_step),
        devup: (devup_start, devup_end, devup_step),
        devdn: (devdn_start, devdn_end, devdn_step),
        matype: (matype.to_string(), matype.to_string(), 0),
        devtype: (devtype, devtype, 0),
    };

    let combos = expand_grid(&sweep);
    let mut metadata = Vec::with_capacity(combos.len() * 4);

    for combo in combos {
        metadata.push(combo.period.unwrap() as f64);
        metadata.push(combo.devup.unwrap());
        metadata.push(combo.devdn.unwrap());
        metadata.push(combo.devtype.unwrap() as f64);
    }

    Ok(metadata)
}

// New ergonomic WASM API
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct BollingerBandsBatchConfig {
    pub period_range: (usize, usize, usize),
    pub devup_range: (f64, f64, f64),
    pub devdn_range: (f64, f64, f64),
    pub matype: String,
    pub devtype: usize,
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct BollingerBandsBatchJsOutput {
    pub upper: Vec<f64>,
    pub middle: Vec<f64>,
    pub lower: Vec<f64>,
    pub combos: Vec<BollingerBandsParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = bollinger_bands_batch)]
pub fn bollinger_bands_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    // 1. Deserialize the configuration object from JavaScript
    let config: BollingerBandsBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = BollingerBandsBatchRange {
        period: config.period_range,
        devup: config.devup_range,
        devdn: config.devdn_range,
        matype: (config.matype.clone(), config.matype, 0),
        devtype: (config.devtype, config.devtype, 0),
    };

    // 2. Run the existing core logic
    let output = bollinger_bands_batch_inner(data, &sweep, Kernel::Scalar, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // 3. Create the structured output
    let js_output = BollingerBandsBatchJsOutput {
        upper: output.upper,
        middle: output.middle,
        lower: output.lower,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    // 4. Serialize the output struct into a JavaScript object
    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn bollinger_bands_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(len);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn bollinger_bands_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn bollinger_bands_into(
    in_ptr: *const f64,
    out_u: *mut f64,
    out_m: *mut f64,
    out_l: *mut f64,
    len: usize,
    period: usize,
    devup: f64,
    devdn: f64,
    matype: String,
    devtype: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_u.is_null() || out_m.is_null() || out_l.is_null() {
        return Err(JsValue::from_str("null pointer"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let mut u = std::slice::from_raw_parts_mut(out_u, len);
        let mut m = std::slice::from_raw_parts_mut(out_m, len);
        let mut l = std::slice::from_raw_parts_mut(out_l, len);

        let input = BollingerBandsInput::from_slice(
            data,
            BollingerBandsParams {
                period: Some(period),
                devup: Some(devup),
                devdn: Some(devdn),
                matype: Some(matype),
                devtype: Some(devtype),
            },
        );

        // Protect against aliasing input/output
        if in_ptr == out_u as *const f64
            || in_ptr == out_m as *const f64
            || in_ptr == out_l as *const f64
        {
            let mut tu = vec![0.0; len];
            let mut tm = vec![0.0; len];
            let mut tl = vec![0.0; len];
            bollinger_bands_into_slices(&mut tu, &mut tm, &mut tl, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            u.copy_from_slice(&tu);
            m.copy_from_slice(&tm);
            l.copy_from_slice(&tl);
        } else {
            bollinger_bands_into_slices(&mut u, &mut m, &mut l, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }
    Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn bollinger_bands_batch_into(
    in_ptr: *const f64,
    upper_ptr: *mut f64,
    middle_ptr: *mut f64,
    lower_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
    devup_start: f64,
    devup_end: f64,
    devup_step: f64,
    devdn_start: f64,
    devdn_end: f64,
    devdn_step: f64,
    matype: &str,
    devtype: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || upper_ptr.is_null() || middle_ptr.is_null() || lower_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = BollingerBandsBatchRange {
            period: (period_start, period_end, period_step),
            devup: (devup_start, devup_end, devup_step),
            devdn: (devdn_start, devdn_end, devdn_step),
            matype: (matype.to_string(), matype.to_string(), 0),
            devtype: (devtype, devtype, 0),
        };

        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let cols = len;

        let upper_out = std::slice::from_raw_parts_mut(upper_ptr, rows * cols);
        let middle_out = std::slice::from_raw_parts_mut(middle_ptr, rows * cols);
        let lower_out = std::slice::from_raw_parts_mut(lower_ptr, rows * cols);

        bollinger_bands_batch_inner_into(
            data,
            &sweep,
            Kernel::Auto,
            false,
            upper_out,
            middle_out,
            lower_out,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = bb_batch_unified)]
pub fn bb_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    #[derive(serde::Deserialize)]
    struct Cfg {
        period_range: (usize, usize, usize),
        devup_range: (f64, f64, f64),
        devdn_range: (f64, f64, f64),
        matype: String,
        devtype_range: (usize, usize, usize),
    }
    let c: Cfg = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;

    let sweep = BollingerBandsBatchRange {
        period: c.period_range,
        devup: c.devup_range,
        devdn: c.devdn_range,
        matype: (c.matype.clone(), c.matype, 0),
        devtype: c.devtype_range,
    };
    let out = bollinger_bands_batch_inner(data, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js = BollingerBatchJsOutput {
        upper: out.upper,
        middle: out.middle,
        lower: out.lower,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    };
    serde_wasm_bindgen::to_value(&js)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

// ==================== PYTHON CUDA BINDINGS ====================
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::{cuda_available, CudaBollingerBands};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;
#[cfg(all(feature = "python", feature = "cuda"))]
use numpy::PyReadonlyArray1;
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::{pyfunction, PyResult, Python};
#[cfg(all(feature = "python", feature = "cuda"))]
// PyValueError already imported above under `#[cfg(feature = "python")]`.
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "bollinger_bands_cuda_batch_dev")]
#[pyo3(signature = (data_f32, period_range, devup_range, devdn_range, device_id=0))]
pub fn bollinger_bands_cuda_batch_dev_py(
    py: Python<'_>,
    data_f32: PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    devup_range: (f64, f64, f64),
    devdn_range: (f64, f64, f64),
    device_id: usize,
) -> PyResult<(DeviceArrayF32Py, DeviceArrayF32Py, DeviceArrayF32Py)> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let slice = data_f32.as_slice()?;
    let sweep = BollingerBandsBatchRange {
        period: period_range,
        devup: devup_range,
        devdn: devdn_range,
        matype: ("sma".to_string(), "sma".to_string(), 0),
        devtype: (0, 0, 0),
    };
    let (up, mid, lo) = py.allow_threads(|| {
        let cuda =
            CudaBollingerBands::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.bollinger_bands_batch_dev(slice, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok((
        DeviceArrayF32Py { inner: up },
        DeviceArrayF32Py { inner: mid },
        DeviceArrayF32Py { inner: lo },
    ))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "bollinger_bands_cuda_many_series_one_param_dev")]
#[pyo3(signature = (prices_tm_f32, cols, rows, period, devup, devdn, device_id=0))]
pub fn bollinger_bands_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    prices_tm_f32: PyReadonlyArray1<'_, f32>,
    cols: usize,
    rows: usize,
    period: usize,
    devup: f32,
    devdn: f32,
    device_id: usize,
) -> PyResult<(DeviceArrayF32Py, DeviceArrayF32Py, DeviceArrayF32Py)> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let tm = prices_tm_f32.as_slice()?;
    let (up, mid, lo) = py.allow_threads(|| {
        let cuda =
            CudaBollingerBands::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.bollinger_bands_many_series_one_param_time_major_dev(
            tm, cols, rows, period, devup, devdn,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok((
        DeviceArrayF32Py { inner: up },
        DeviceArrayF32Py { inner: mid },
        DeviceArrayF32Py { inner: lo },
    ))
}
