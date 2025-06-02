//! # Ease of Movement (EMV)
//!
//! Measures how easily price moves given volume. Returns a vector of EMV values (length
//! = input length). Leading NaNs are returned before the first valid calculation.
//!
//! ## Errors
//! - **AllValuesNaN**: All input values are `NaN`.
//! - **EmptyData**: Input data is empty.
//! - **NotEnoughData**: Fewer than 2 valid data points.
//!
//! ## Returns
//! - **`Ok(EmvOutput)`** on success (`values: Vec<f64>`)
//! - **`Err(EmvError)`** otherwise
//!
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use thiserror::Error;
use std::error::Error;

#[derive(Debug, Clone)]
pub enum EmvData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct EmvOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct EmvParams;

#[derive(Debug, Clone)]
pub struct EmvInput<'a> {
    pub data: EmvData<'a>,
    pub params: EmvParams,
}

impl<'a> EmvInput<'a> {
    #[inline(always)]
    pub fn from_candles(candles: &'a Candles) -> Self {
        Self {
            data: EmvData::Candles { candles },
            params: EmvParams,
        }
    }

    #[inline(always)]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
    ) -> Self {
        Self {
            data: EmvData::Slices {
                high,
                low,
                close,
                volume,
            },
            params: EmvParams,
        }
    }

    #[inline(always)]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles)
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct EmvBuilder {
    kernel: Kernel,
}

impl EmvBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<EmvOutput, EmvError> {
        let input = EmvInput::from_candles(c);
        emv_with_kernel(&input, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<EmvOutput, EmvError> {
        let input = EmvInput::from_slices(high, low, close, volume);
        emv_with_kernel(&input, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<EmvStream, EmvError> {
        EmvStream::try_new()
    }
}

#[derive(Debug, Error)]
pub enum EmvError {
    #[error("emv: Empty data provided.")]
    EmptyData,
    #[error("emv: Not enough data: needed at least 2 valid points, found {valid}.")]
    NotEnoughData { valid: usize },
    #[error("emv: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn emv(input: &EmvInput) -> Result<EmvOutput, EmvError> {
    emv_with_kernel(input, Kernel::Auto)
}

pub fn emv_with_kernel(input: &EmvInput, kernel: Kernel) -> Result<EmvOutput, EmvError> {
    let (high, low, close, volume) = match &input.data {
        EmvData::Candles { candles } => {
            let high = source_type(candles, "high");
            let low = source_type(candles, "low");
            let close = source_type(candles, "close");
            let volume = source_type(candles, "volume");
            (high, low, close, volume)
        }
        EmvData::Slices {
            high,
            low,
            close,
            volume,
        } => (*high, *low, *close, *volume),
    };

    if high.is_empty() || low.is_empty() || volume.is_empty() {
        return Err(EmvError::EmptyData);
    }
    let len = high.len().min(low.len()).min(volume.len());
    if len == 0 {
        return Err(EmvError::EmptyData);
    }

    let first = (0..len).find(|&i| !(high[i].is_nan() || low[i].is_nan() || volume[i].is_nan()));
    let first = match first {
        Some(idx) => idx,
        None => return Err(EmvError::AllValuesNaN),
    };

    let mut valid_count = 0_usize;
    for i in first..len {
        if !(high[i].is_nan() || low[i].is_nan() || volume[i].is_nan()) {
            valid_count += 1;
        }
    }
    if valid_count < 2 {
        return Err(EmvError::NotEnoughData { valid: valid_count });
    }

    let mut out = vec![f64::NAN; len];
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                emv_scalar(high, low, volume, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                emv_avx2(high, low, volume, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                emv_avx512(high, low, volume, first, &mut out)
            }
            _ => unreachable!(),
        }
    }
    Ok(EmvOutput { values: out })
}

#[inline]
pub fn emv_scalar(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    first: usize,
    out: &mut [f64],
) {
    let len = high.len().min(low.len()).min(volume.len());
    let mut last_mid = 0.5 * (high[first] + low[first]);
    for i in (first + 1)..len {
        if high[i].is_nan() || low[i].is_nan() || volume[i].is_nan() {
            out[i] = f64::NAN;
            continue;
        }
        let current_mid = 0.5 * (high[i] + low[i]);
        let range = high[i] - low[i];
        if range == 0.0 {
            out[i] = f64::NAN;
            last_mid = current_mid;
            continue;
        }
        let br = volume[i] / 10000.0 / range;
        out[i] = (current_mid - last_mid) / br;
        last_mid = current_mid;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn emv_avx512(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    first: usize,
    out: &mut [f64],
) {
    if high.len() <= 32 {
        unsafe { emv_avx512_short(high, low, volume, first, out) }
    } else {
        unsafe { emv_avx512_long(high, low, volume, first, out) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn emv_avx2(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    first: usize,
    out: &mut [f64],
) {
    emv_scalar(high, low, volume, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn emv_avx512_short(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    first: usize,
    out: &mut [f64],
) {
    emv_scalar(high, low, volume, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn emv_avx512_long(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    first: usize,
    out: &mut [f64],
) {
    emv_scalar(high, low, volume, first, out);
}

#[derive(Debug, Clone)]
pub struct EmvStream {
    last_mid: Option<f64>,
    initialized: bool,
}

impl EmvStream {
    pub fn try_new() -> Result<Self, EmvError> {
        Ok(Self {
            last_mid: None,
            initialized: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, volume: f64) -> Option<f64> {
        if high.is_nan() || low.is_nan() || volume.is_nan() {
            return None;
        }
        let current_mid = 0.5 * (high + low);
        if !self.initialized {
            self.last_mid = Some(current_mid);
            self.initialized = true;
            return None;
        }
        let last_mid = self.last_mid.unwrap();
        let range = high - low;
        if range == 0.0 {
            self.last_mid = Some(current_mid);
            return None;
        }
        let br = volume / 10000.0 / range;
        let out = (current_mid - last_mid) / br;
        self.last_mid = Some(current_mid);
        Some(out)
    }
}

#[derive(Clone, Debug)]
pub struct EmvBatchRange {}

impl Default for EmvBatchRange {
    fn default() -> Self {
        Self {}
    }
}

#[derive(Clone, Debug, Default)]
pub struct EmvBatchBuilder {
    kernel: Kernel,
    _range: EmvBatchRange,
}

impl EmvBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<EmvBatchOutput, EmvError> {
        emv_batch_with_kernel(high, low, close, volume, self.kernel)
    }

    pub fn with_default_slices(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
        k: Kernel,
    ) -> Result<EmvBatchOutput, EmvError> {
        EmvBatchBuilder::new().kernel(k).apply_slices(high, low, close, volume)
    }

    pub fn apply_candles(self, c: &Candles) -> Result<EmvBatchOutput, EmvError> {
        let high = source_type(c, "high");
        let low = source_type(c, "low");
        let close = source_type(c, "close");
        let volume = source_type(c, "volume");
        self.apply_slices(high, low, close, volume)
    }

    pub fn with_default_candles(c: &Candles, k: Kernel) -> Result<EmvBatchOutput, EmvError> {
        EmvBatchBuilder::new().kernel(k).apply_candles(c)
    }
}

pub fn emv_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    _close: &[f64],
    volume: &[f64],
    kernel: Kernel,
) -> Result<EmvBatchOutput, EmvError> {
    let simd = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => Kernel::ScalarBatch,
    };
    emv_batch_par_slice(high, low, volume, simd)
}

#[derive(Clone, Debug)]
pub struct EmvBatchOutput {
    pub values: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[inline(always)]
fn expand_grid(_r: &EmvBatchRange) -> Vec<()> {
    vec![()]
}

#[inline(always)]
pub fn emv_batch_slice(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    kern: Kernel,
) -> Result<EmvBatchOutput, EmvError> {
    emv_batch_inner(high, low, volume, kern, false)
}

#[inline(always)]
pub fn emv_batch_par_slice(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    kern: Kernel,
) -> Result<EmvBatchOutput, EmvError> {
    emv_batch_inner(high, low, volume, kern, true)
}

fn emv_batch_inner(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    kern: Kernel,
    _parallel: bool,
) -> Result<EmvBatchOutput, EmvError> {
    let len = high.len().min(low.len()).min(volume.len());
    let mut out = vec![f64::NAN; len];
    let first = (0..len).find(|&i| !(high[i].is_nan() || low[i].is_nan() || volume[i].is_nan()));
    let first = match first {
        Some(idx) => idx,
        None => return Err(EmvError::AllValuesNaN),
    };

    unsafe {
        match kern {
            Kernel::ScalarBatch | Kernel::Scalar => {
                emv_scalar(high, low, volume, first, &mut out);
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2Batch | Kernel::Avx2 => {
                emv_avx2(high, low, volume, first, &mut out);
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512Batch | Kernel::Avx512 => {
                emv_avx512(high, low, volume, first, &mut out);
            }
            _ => unreachable!(),
        }
    }

    Ok(EmvBatchOutput {
        values: out,
        rows: 1,
        cols: len,
    })
}

#[inline(always)]
pub fn emv_row_scalar(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    first: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    emv_scalar(high, low, volume, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn emv_row_avx2(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    first: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    emv_scalar(high, low, volume, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn emv_row_avx512(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    first: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    emv_avx512(high, low, volume, first, out);
}


#[inline(always)]
fn expand_grid_emv(_r: &EmvBatchRange) -> Vec<()> {
    vec![()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_emv_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = EmvInput::from_candles(&candles);
        let output = emv_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        let expected_last_five_emv = [
            -6488905.579799851,
            2371436.7401001123,
            -3855069.958128531,
            1051939.877943717,
            -8519287.22257077,
        ];
        let start = output.values.len().saturating_sub(5);
        for (i, &val) in output.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five_emv[i]).abs();
            let tol = expected_last_five_emv[i].abs() * 0.0001;
            assert!(
                diff <= tol,
                "[{}] EMV {:?} mismatch at idx {}: got {}, expected {}, diff={}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five_emv[i],
                diff
            );
        }
        Ok(())
    }

    fn check_emv_with_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = EmvInput::with_default_candles(&candles);
        let output = emv_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_emv_empty_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let input = EmvInput::from_slices(&empty, &empty, &empty, &empty);
        let result = emv_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_emv_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let nan_arr = [f64::NAN, f64::NAN];
        let input = EmvInput::from_slices(&nan_arr, &nan_arr, &nan_arr, &nan_arr);
        let result = emv_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_emv_not_enough_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10000.0, f64::NAN];
        let low = [9990.0, f64::NAN];
        let close = [9995.0, f64::NAN];
        let volume = [1_000_000.0, f64::NAN];
        let input = EmvInput::from_slices(&high, &low, &close, &volume);
        let result = emv_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_emv_basic_calculation(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 12.0, 13.0, 15.0];
        let low = [5.0, 7.0, 8.0, 10.0];
        let close = [7.5, 9.0, 10.5, 12.5];
        let volume = [10000.0, 20000.0, 25000.0, 30000.0];
        let input = EmvInput::from_slices(&high, &low, &close, &volume);
        let output = emv_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), 4);
        assert!(output.values[0].is_nan());
        for &val in &output.values[1..] {
            assert!(!val.is_nan());
        }
        Ok(())
    }

    fn check_emv_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let high = source_type(&candles, "high");
        let low = source_type(&candles, "low");
        let volume = source_type(&candles, "volume");

        let output = emv_with_kernel(&EmvInput::from_candles(&candles), kernel)?.values;

        let mut stream = EmvStream::try_new()?;
        let mut stream_values = Vec::with_capacity(high.len());
        for i in 0..high.len() {
            match stream.update(high[i], low[i], volume[i]) {
                Some(val) => stream_values.push(val),
                None => stream_values.push(f64::NAN),
            }
        }
        assert_eq!(output.len(), stream_values.len());
        for (b, s) in output.iter().zip(stream_values.iter()) {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] EMV streaming f64 mismatch: batch={}, stream={}, diff={}",
                test_name,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_emv_tests {
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

    generate_all_emv_tests!(
        check_emv_accuracy,
        check_emv_with_default_candles,
        check_emv_empty_data,
        check_emv_all_nan,
        check_emv_not_enough_data,
        check_emv_basic_calculation,
        check_emv_streaming
    );

    fn check_batch_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = EmvBatchBuilder::new().kernel(kernel).apply_candles(&c)?;
        assert_eq!(output.values.len(), c.close.len());
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
    gen_batch_tests!(check_batch_row);
}
