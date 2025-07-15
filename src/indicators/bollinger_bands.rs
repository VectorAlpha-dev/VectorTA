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
use crate::indicators::deviation::{deviation, DevInput, DevParams};
use crate::indicators::moving_averages::ma::{ma, MaData};
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
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
) -> Result<(
    &'a [f64],      // data
    usize,          // period
    f64,            // devup
    f64,            // devdn
    String,         // matype
    usize,          // devtype
    usize,          // first valid index
    Kernel,         // final kernel
), BollingerBandsError> {
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
    unsafe {
        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => 
                bb_row_scalar(data, matype, period, devtype, devup, devdn, first, out_u, out_m, out_l),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => 
                bb_row_avx2(data, matype, period, devtype, devup, devdn, first, out_u, out_m, out_l),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => 
                bb_row_avx512(data, matype, period, devtype, devup, devdn, first, out_u, out_m, out_l),
            _ => unreachable!(),
        }
    }
    Ok(())
}

pub fn bollinger_bands_with_kernel(
    input: &BollingerBandsInput,
    kernel: Kernel,
) -> Result<BollingerBandsOutput, BollingerBandsError> {
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

    let ma_data = match &input.data {
        BollingerBandsData::Candles { candles, source } => MaData::Candles { candles, source },
        BollingerBandsData::Slice(slice) => MaData::Slice(slice),
    };
    let dev_input = DevInput::from_slice(
        data,
        DevParams {
            period: Some(period),
            devtype: Some(devtype),
        },
    );

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => bollinger_bands_scalar(
                data, &matype, ma_data, period, devtype, dev_input, devup, devdn,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => bollinger_bands_avx2(
                data, &matype, ma_data, period, devtype, dev_input, devup, devdn,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => bollinger_bands_avx512(
                data, &matype, ma_data, period, devtype, dev_input, devup, devdn,
            ),
            _ => unreachable!(),
        }
    }
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

    let mut upper_band = vec![f64::NAN; data.len()];
    let mut middle_band = vec![f64::NAN; data.len()];
    let mut lower_band = vec![f64::NAN; data.len()];
    let first = data.iter().position(|x| !x.is_nan()).unwrap();
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
    // SAFETY: This function must write to all elements from (first + period - 1) onwards
    let ma_data = MaData::Slice(data);
    let dev_input = DevInput::from_slice(
        data,
        DevParams {
            period: Some(period),
            devtype: Some(devtype),
        },
    );
    
    let middle = ma(matype, ma_data, period).unwrap();
    let dev_values = deviation(&dev_input).unwrap();
    
    for i in (first + period - 1)..data.len() {
        let m = middle[i];
        let d = dev_values[i];
        out_m[i] = m;
        out_u[i] = m + devup * d;
        out_l[i] = m - devdn * d;
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
    bb_row_scalar(data, matype, period, devtype, devup, devdn, first, out_u, out_m, out_l)
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
    bb_row_scalar(data, matype, period, devtype, devup, devdn, first, out_u, out_m, out_l)
}

#[derive(Debug, Clone)]
pub struct BollingerBandsStream {
    pub period: usize,
    pub devup: f64,
    pub devdn: f64,
    pub matype: String,
    pub devtype: usize,
    pub buffer: Vec<f64>,
}

impl BollingerBandsStream {
    pub fn try_new(params: BollingerBandsParams) -> Result<Self, BollingerBandsError> {
        let period = params.period.unwrap_or(20);
        let devup = params.devup.unwrap_or(2.0);
        let devdn = params.devdn.unwrap_or(2.0);
        let matype = params.matype.unwrap_or_else(|| "sma".to_string());
        let devtype = params.devtype.unwrap_or(0);

        Ok(Self {
            period,
            devup,
            devdn,
            matype,
            devtype,
            buffer: Vec::with_capacity(period),
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64, f64)> {
        if self.buffer.len() == self.period {
            self.buffer.remove(0);
        }
        self.buffer.push(value);
        if self.buffer.len() < self.period || self.buffer.iter().all(|x| x.is_nan()) {
            return None;
        }
        let data = self.buffer.as_slice();
        let ma_data = MaData::Slice(data);
        let dev_input = DevInput::from_slice(
            data,
            DevParams {
                period: Some(self.period),
                devtype: Some(self.devtype),
            },
        );

        let mid = ma(&self.matype, ma_data, self.period)
            .ok()
            .and_then(|v| v.last().copied())
            .unwrap_or(f64::NAN);
        let dev = deviation(&dev_input)
            .ok()
            .and_then(|v| v.last().copied())
            .unwrap_or(f64::NAN);

        Some((mid + self.devup * dev, mid, mid - self.devdn * dev))
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
            return Err(BollingerBandsError::InvalidPeriod {
                period: 0,
                data_len: 0,
            });
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
    let mut upper = vec![f64::NAN; rows * cols];
    let mut middle = vec![f64::NAN; rows * cols];
    let mut lower = vec![f64::NAN; rows * cols];

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

    Ok(BollingerBandsBatchOutput {
        upper,
        middle,
        lower,
        combos,
        rows,
        cols,
    })
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
        check_bb_streaming
    );

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
}

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

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
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    use numpy::{PyArray1, PyArrayMethods};

    let slice_in = data.as_slice()?; // zero-copy, read-only view

    // Use kernel validation for safety
    let kern = validate_kernel(kernel, false)?;

    // ---------- build params & prepare once -----------------------------------------
    let params = BollingerBandsParams {
        period: Some(period),
        devup: Some(devup),
        devdn: Some(devdn),
        matype: Some(matype.to_string()),
        devtype: Some(devtype),
    };
    let bb_in = BollingerBandsInput::from_slice(slice_in, params);
    
    // Get prepared values without kernel dispatch repetition
    let (d, per, du, dd, mt, dt, first, chosen) = bb_prepare(&bb_in, kern)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // ---------- allocate uninitialized NumPy output buffers -------------------------
    // NOTE: PyArray1::new() creates uninitialized memory, not zero-initialized
    let upper_arr = unsafe { PyArray1::<f64>::new(py, [d.len()], false) };
    let middle_arr = unsafe { PyArray1::<f64>::new(py, [d.len()], false) };
    let lower_arr = unsafe { PyArray1::<f64>::new(py, [d.len()], false) };
    
    let slice_upper = unsafe { upper_arr.as_slice_mut()? };
    let slice_middle = unsafe { middle_arr.as_slice_mut()? };
    let slice_lower = unsafe { lower_arr.as_slice_mut()? };

    // ---------- heavy lifting without the GIL --------------------------------------
    py.allow_threads(|| -> Result<(), BollingerBandsError> {
        // SAFETY: We must write to ALL elements before returning to Python
        // 1. Write the NaN prefix once
        if first + per > 0 {
            let nan_end = (first + per - 1).min(d.len());
            slice_upper[..nan_end].fill(f64::NAN);
            slice_middle[..nan_end].fill(f64::NAN);
            slice_lower[..nan_end].fill(f64::NAN);
        }
        
        // 2. Compute directly into NumPy buffers
        bollinger_bands_compute_into(
            d, per, du, dd, &mt, dt, first, chosen,
            slice_upper, slice_middle, slice_lower,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((upper_arr, middle_arr, lower_arr))
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
#[pyo3(signature = (data, period_range=(20, 20, 0), devup_range=(2.0, 2.0, 0.0), devdn_range=(2.0, 2.0, 0.0), matype="sma", devtype=0, kernel=None))]
pub fn bollinger_bands_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    devup_range: (f64, f64, f64),
    devdn_range: (f64, f64, f64),
    matype: &str,
    devtype: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;

    let sweep = BollingerBandsBatchRange {
        period: period_range,
        devup: devup_range,
        devdn: devdn_range,
        matype: (matype.to_string(), matype.to_string(), 0),
        devtype: (devtype, devtype, 0),
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
    let combos_clone = py.allow_threads(|| -> Result<Vec<BollingerBandsParams>, BollingerBandsError> {
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
        
        // Process each row directly into the output buffers
        let first = slice_in.iter().position(|x| !x.is_nan()).unwrap_or(0);
        
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            slice_upper
                .par_chunks_mut(cols)
                .zip(slice_middle.par_chunks_mut(cols))
                .zip(slice_lower.par_chunks_mut(cols))
                .zip(combos.par_iter())
                .for_each(|(((out_u, out_m), out_l), combo)| {
                    let p = combo.period.unwrap();
                    let du = combo.devup.unwrap();
                    let dd = combo.devdn.unwrap();
                    let mt = combo.matype.as_ref().unwrap();
                    let dt = combo.devtype.unwrap();
                    
                    // Fill NaN prefix
                    if first + p > 0 {
                        let nan_end = (first + p - 1).min(slice_in.len());
                        out_u[..nan_end].fill(f64::NAN);
                        out_m[..nan_end].fill(f64::NAN);
                        out_l[..nan_end].fill(f64::NAN);
                    }
                    
                    // Compute into buffers
                    let _ = bollinger_bands_compute_into(
                        slice_in, p, du, dd, mt, dt, first, simd,
                        out_u, out_m, out_l,
                    );
                });
        }
        
        #[cfg(target_arch = "wasm32")]
        {
            for (i, combo) in combos.iter().enumerate() {
                let row_start = i * cols;
                let out_u = &mut slice_upper[row_start..row_start + cols];
                let out_m = &mut slice_middle[row_start..row_start + cols];
                let out_l = &mut slice_lower[row_start..row_start + cols];
                
                let p = combo.period.unwrap();
                let du = combo.devup.unwrap();
                let dd = combo.devdn.unwrap();
                let mt = combo.matype.as_ref().unwrap();
                let dt = combo.devtype.unwrap();
                
                // Fill NaN prefix
                if first + p > 0 {
                    let nan_end = (first + p - 1).min(slice_in.len());
                    out_u[..nan_end].fill(f64::NAN);
                    out_m[..nan_end].fill(f64::NAN);
                    out_l[..nan_end].fill(f64::NAN);
                }
                
                // Compute into buffers
                let _ = bollinger_bands_compute_into(
                    slice_in, p, du, dd, mt, dt, first, simd,
                    out_u, out_m, out_l,
                );
            }
        }
        
        Ok(combos)
    })
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
    matype: &str,
    devtype: usize,
) -> Result<Vec<f64>, JsValue> {
    let mut up = vec![f64::NAN; data.len()];
    let mut mid = vec![f64::NAN; data.len()];
    let mut low = vec![f64::NAN; data.len()];
    
    // prepare once
    let params = BollingerBandsParams {
        period: Some(period),
        devup: Some(devup),
        devdn: Some(devdn),
        matype: Some(matype.to_string()),
        devtype: Some(devtype),
    };
    let bb_in = BollingerBandsInput::from_slice(data, params);
    let (d, per, du, dd, mt, dt, first, chosen) = bb_prepare(&bb_in, Kernel::Scalar)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    // Compute into pre-allocated vectors
    bollinger_bands_compute_into(d, per, du, dd, &mt, dt, first, chosen,
                                &mut up, &mut mid, &mut low)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    // Flatten the three bands into a single vector - same convention as before
    let mut out = up;
    out.extend(mid);
    out.extend(low);
    Ok(out)
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
    
    let mut upper_vec = vec![f64::NAN; rows * cols];
    let mut middle_vec = vec![f64::NAN; rows * cols];
    let mut lower_vec = vec![f64::NAN; rows * cols];
    
    // Find first valid index
    let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
    
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
            data, p, du, dd, mt, dt, first, Kernel::Scalar,
            out_u, out_m, out_l,
        ).map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    
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
