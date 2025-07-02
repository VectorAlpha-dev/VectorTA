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

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_kernel, detect_best_batch_kernel, alloc_with_nan_prefix, make_uninit_matrix, init_matrix_prefixes};
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;
use chrono::{Datelike, NaiveDateTime, Utc};
#[cfg(not(target_arch = "wasm32"))]
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use std::mem::MaybeUninit;

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
}

#[derive(Debug, Error)]
pub enum VwapError {
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
    if n != volumes.len() {
        return Err(VwapError::MismatchPricesVolumes {
            prices: n,
            volumes: volumes.len(),
        });
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
            Kernel::Scalar | Kernel::ScalarBatch =>
                vwap_scalar(timestamps, volumes, prices, count, unit_char, &mut values)?,
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch =>
                vwap_avx2(timestamps, volumes, prices, count, unit_char, &mut values)?,
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch =>
                vwap_avx512(timestamps, volumes, prices, count, unit_char, &mut values)?,
            _ => unreachable!(),
        }
    }

    Ok(VwapOutput { values })
}

#[inline(always)]
pub fn vwap_scalar(
    timestamps: &[i64],
    volumes: &[f64],
    prices: &[f64],
    count: u32,
    unit_char: char,
    out:       &mut [f64],
) -> Result<(), VwapError> {
    debug_assert_eq!(out.len(), prices.len(), "output slice length mismatch");

    let mut current_group_id = -1_i64;
    let mut volume_sum       = 0.0;
    let mut vol_price_sum    = 0.0;

    for i in 0..prices.len() {
        let ts_ms   = timestamps[i];
        let price   = prices[i];
        let volume  = volumes[i];
        let group_id = match unit_char {
            'm' => ts_ms / ((count as i64) * 60_000),
            'h' => ts_ms / ((count as i64) * 3_600_000),
            'd' => ts_ms / ((count as i64) * 86_400_000),
            'M' => floor_to_month(ts_ms, count)
                     .map_err(|_| VwapError::MonthConversionError { ts_ms })?,
            _   => return Err(VwapError::UnsupportedAnchorUnit { unit_char }),
        };

        if group_id != current_group_id {
            current_group_id = group_id;
            volume_sum    = 0.0;
            vol_price_sum = 0.0;
        }
        volume_sum    += volume;
        vol_price_sum += volume * price;

        out[i] = if volume_sum > 0.0 {
            vol_price_sum / volume_sum
        
            } else {
            f64::NAN
        };
    }
    Ok(())
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
) -> Result<(),  VwapError> {
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
) -> Result<(),  VwapError> {
    vwap_scalar(timestamps, volumes, prices, count, unit_char, out)
}

/// Streaming VWAP (per anchor bucket)
#[derive(Debug, Clone)]
pub struct VwapStream {
    anchor: String,
    count: u32,
    unit_char: char,
    current_group_id: i64,
    volume_sum: f64,
    vol_price_sum: f64,
}

impl VwapStream {
    pub fn try_new(params: VwapParams) -> Result<Self, VwapError> {
        let anchor = params.anchor.unwrap_or_else(|| "1d".to_string());
        let (count, unit_char) = parse_anchor(&anchor)
            .map_err(|e| VwapError::ParseAnchorError { msg: e.to_string() })?;
        Ok(Self {
            anchor,
            count,
            unit_char,
            current_group_id: -1,
            volume_sum: 0.0,
            vol_price_sum: 0.0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, timestamp: i64, price: f64, volume: f64) -> Option<f64> {
        let group_id = match self.unit_char {
            'm' => (timestamp / (self.count as i64 * 60_000)),
            'h' => (timestamp / (self.count as i64 * 3_600_000)),
            'd' => (timestamp / (self.count as i64 * 86_400_000)),
            'M' => match floor_to_month(timestamp, self.count) {
                Ok(g) => g,
                Err(_) => return None,
            },
            _ => return None,
        };
        if group_id != self.current_group_id {
            self.current_group_id = group_id;
            self.volume_sum = 0.0;
            self.vol_price_sum = 0.0;
        }
        self.volume_sum += volume;
        self.vol_price_sum += volume * price;
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
    pub fn anchor_range(mut self, start: impl Into<String>, end: impl Into<String>, step: u32) -> Self {
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
        _ => {
            return Err(VwapError::NoData);
        }
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

#[inline(always)]
fn expand_grid_vwap(r: &VwapBatchRange) -> Vec<VwapParams> {
    if r.anchor.2 == 0 || r.anchor.0 == r.anchor.1 {
        return vec![VwapParams {
            anchor: Some(r.anchor.0.clone()),
        }];
    }
    // e.g. anchor: ("1d", "3d", 1) = ["1d", "2d", "3d"]
    let mut result = vec![];
    let mut start = anchor_to_num_and_unit(&r.anchor.0);
    let end = anchor_to_num_and_unit(&r.anchor.1);
    let step = r.anchor.2;
    if let (Some((mut n, unit)), Some((e, _))) = (start, end) {
        while n <= e {
            result.push(VwapParams {
                anchor: Some(format!("{}{}", n, unit)),
            });
            n += step;
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
    if idx == 0 { return None; }
    let num = anchor[..idx].parse::<u32>().ok()?;
    let unit = anchor[idx..].chars().next()?;
    Some((num, unit))
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
        return Err(VwapError::NoData);
    }

    let rows = combos.len();
    let cols = prices.len();

    let mut raw = make_uninit_matrix(rows, cols);          // Vec<MaybeUninit<f64>>

    // optional: NaN prefixes – none needed, but this exercises the helper
    let warm: Vec<usize> = vec![0; rows];
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

    // ---------- 2. closure to fill one row in-place --------------------------
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let params = combos.get(row).unwrap();
        let (count, unit_char) = parse_anchor(params.anchor.as_deref().unwrap_or("1d"))
            .map_err(|e| VwapError::ParseAnchorError { msg: e.to_string() })
            .unwrap();

        // cast this row to &mut [f64]
        let out_row = core::slice::from_raw_parts_mut(
            dst_mu.as_mut_ptr() as *mut f64,
            dst_mu.len(),
        );

        match kern {
            Kernel::Scalar =>
                vwap_row_scalar (timestamps, volumes, prices, count, unit_char, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2   =>
                vwap_row_avx2  (timestamps, volumes, prices, count, unit_char, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 =>
                vwap_row_avx512(timestamps, volumes, prices, count, unit_char, out_row),
            _ => unreachable!(),
        }
    };

    // ---------- 3. run every row, writing directly into `raw` ----------------
    if parallel {

        #[cfg(not(target_arch = "wasm32"))] {

        raw.par_chunks_mut(cols)

                    .enumerate()

                    .for_each(|(row, slice)| do_row(row, slice));

        }

        #[cfg(target_arch = "wasm32")] {

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
    let values: Vec<f64> = unsafe { std::mem::transmute(raw) };

    Ok(VwapBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline]
fn parse_anchor(anchor: &str) -> Result<(u32, char), Box<dyn std::error::Error>> {
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
fn floor_to_month(ts_ms: i64, count: u32) -> Result<i64, Box<dyn Error>> {
    let naive = NaiveDateTime::from_timestamp(ts_ms / 1000, ((ts_ms % 1000) * 1_000_000) as u32);
    let dt_utc = chrono::DateTime::<Utc>::from_utc(naive, Utc);
    let year = dt_utc.year();
    let month = dt_utc.month();
    if count == 1 {
        let group_id = (year as i64) * 12 + (month as i64 - 1);
        Ok(group_id)
    
        } else {
        let quarter_group = (month as i64 - 1) / (count as i64);
        let group_id = (year as i64) * 100 + quarter_group;
        Ok(group_id)
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
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

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

    fn check_vwap_candles_plus_prices(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
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

    fn check_vwap_anchor_parsing_error(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
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

    fn check_vwap_slice_data_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
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

    fn check_vwap_with_default_candles(test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
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

    fn check_vwap_with_default_params(test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
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

    generate_all_vwap_tests!(
        check_vwap_partial_params,
        check_vwap_accuracy,
        check_vwap_candles_plus_prices,
        check_vwap_anchor_parsing_error,
        check_vwap_slice_data_reinput,
        check_vwap_nan_handling,
        check_vwap_with_default_candles,
        check_vwap_with_default_params
    );
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
    let row = output.combos.iter().position(|p| p.anchor == def.anchor)
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

        let anchors: Vec<_> = batch.combos.iter()
            .map(|p| p.anchor.clone().unwrap())
            .collect();
        assert_eq!(anchors, vec!["1d".to_string(), "2d".to_string(), "3d".to_string()]);
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
    gen_batch_tests!(check_batch_anchor_grid);
}
