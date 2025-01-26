/// # Volume Weighted MACD (VWMACD)
///
/// A variant of MACD that uses volume-weighted moving averages (VWMA) instead of
/// traditional moving averages. This implementation follows the Python reference
/// exactly, using five separate moving-average calls:
/// 1) `SMA(close * volume, slow_period)`
/// 2) `SMA(volume, slow_period)`
/// 3) `SMA(close * volume, fast_period)`
/// 4) `SMA(volume, fast_period)`
/// 5) `EMA(vwmacd_line, signal_period)`
///
/// Then, `vwma_slow = slow_sma_cv / slow_sma_v`, `vwma_fast = fast_sma_cv / fast_sma_v`,
/// `macd = vwma_fast - vwma_slow`, `signal = EMA(macd)`, and `hist = macd - signal`.
///
/// ## Parameters
/// - **fast_period**: Fast VWMA window size. Defaults to 12.
/// - **slow_period**: Slow VWMA window size. Defaults to 26.
/// - **signal_period**: Window size for EMA of the VWMACD line. Defaults to 9.
///
/// ## Errors
/// - **EmptyData**: No data provided.
/// - **InvalidPeriod**: Any period is zero or exceeds the data length.
/// - **AllValuesNaN**: All input values (close or volume) are NaN.
/// - **Sma** / **Ema**: Underlying SMA/EMA errors (period / data mismatch, etc.).
///
/// ## Returns
/// - **`Ok(VwmacdOutput)`** on success, containing three `Vec<f64>`:
///   - `macd`: the VWMACD line
///   - `signal`: the signal line
///   - `hist`: the histogram
/// - **`Err(VwmacdError)`** otherwise.
use crate::indicators::ema::{ema, EmaData, EmaError, EmaInput, EmaParams};
use crate::indicators::sma::{sma, SmaData, SmaError, SmaInput, SmaParams};
use crate::utilities::data_loader::{source_type, Candles};

#[derive(Debug, Clone)]
pub enum VwmacdData<'a> {
    Candles {
        candles: &'a Candles,
        close_source: &'a str,
        volume_source: &'a str,
    },
    Slices {
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct VwmacdOutput {
    pub macd: Vec<f64>,
    pub signal: Vec<f64>,
    pub hist: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct VwmacdParams {
    pub fast_period: Option<usize>,
    pub slow_period: Option<usize>,
    pub signal_period: Option<usize>,
}

impl Default for VwmacdParams {
    fn default() -> Self {
        Self {
            fast_period: Some(12),
            slow_period: Some(26),
            signal_period: Some(9),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VwmacdInput<'a> {
    pub data: VwmacdData<'a>,
    pub params: VwmacdParams,
}

impl<'a> VwmacdInput<'a> {
    pub fn from_candles(
        candles: &'a Candles,
        close_source: &'a str,
        volume_source: &'a str,
        params: VwmacdParams,
    ) -> Self {
        Self {
            data: VwmacdData::Candles {
                candles,
                close_source,
                volume_source,
            },
            params,
        }
    }

    pub fn from_slices(close: &'a [f64], volume: &'a [f64], params: VwmacdParams) -> Self {
        Self {
            data: VwmacdData::Slices { close, volume },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: VwmacdData::Candles {
                candles,
                close_source: "close",
                volume_source: "volume",
            },
            params: VwmacdParams::default(),
        }
    }

    pub fn get_fast(&self) -> usize {
        self.params
            .fast_period
            .unwrap_or_else(|| VwmacdParams::default().fast_period.unwrap())
    }

    pub fn get_slow(&self) -> usize {
        self.params
            .slow_period
            .unwrap_or_else(|| VwmacdParams::default().slow_period.unwrap())
    }

    pub fn get_signal(&self) -> usize {
        self.params
            .signal_period
            .unwrap_or_else(|| VwmacdParams::default().signal_period.unwrap())
    }
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum VwmacdError {
    #[error("vwmacd: Empty data provided.")]
    EmptyData,
    #[error("vwmacd: Invalid period. fast = {fast}, slow = {slow}, signal = {signal}, data_len = {data_len}")]
    InvalidPeriod {
        fast: usize,
        slow: usize,
        signal: usize,
        data_len: usize,
    },
    #[error("vwmacd: All values are NaN (either close or volume).")]
    AllValuesNaN,
    #[error("vwmacd: SMA error: {0}")]
    Sma(#[from] SmaError),
    #[error("vwmacd: EMA error: {0}")]
    Ema(#[from] EmaError),
}

#[inline]
pub fn vwmacd(input: &VwmacdInput) -> Result<VwmacdOutput, VwmacdError> {
    let (close, volume) = match &input.data {
        VwmacdData::Candles {
            candles,
            close_source,
            volume_source,
        } => {
            let c = source_type(candles, close_source);
            let v = source_type(candles, volume_source);
            (c, v)
        }
        VwmacdData::Slices { close, volume } => (*close, *volume),
    };

    if close.is_empty() || volume.is_empty() {
        return Err(VwmacdError::EmptyData);
    }

    let fast = input.get_fast();
    let slow = input.get_slow();
    let signal = input.get_signal();
    let data_len = close.len();

    if fast == 0
        || slow == 0
        || signal == 0
        || fast > data_len
        || slow > data_len
        || signal > data_len
    {
        return Err(VwmacdError::InvalidPeriod {
            fast,
            slow,
            signal,
            data_len,
        });
    }

    if !close.iter().any(|x| !x.is_nan()) || !volume.iter().any(|x| !x.is_nan()) {
        return Err(VwmacdError::AllValuesNaN);
    }

    let mut close_x_volume = vec![f64::NAN; data_len];
    for i in 0..data_len {
        close_x_volume[i] = close[i] * volume[i];
    }

    let slow_sma_cv = sma(&SmaInput::from_slice(
        &close_x_volume,
        SmaParams { period: Some(slow) },
    ))?
    .values;
    let slow_sma_v = sma(&SmaInput::from_slice(
        &volume,
        SmaParams { period: Some(slow) },
    ))?
    .values;

    let mut vwma_slow = vec![f64::NAN; data_len];
    for i in 0..data_len {
        let denom = slow_sma_v[i];
        if !denom.is_nan() && denom != 0.0 {
            vwma_slow[i] = slow_sma_cv[i] / denom;
        }
    }

    let fast_sma_cv = sma(&SmaInput::from_slice(
        &close_x_volume,
        SmaParams { period: Some(fast) },
    ))?
    .values;
    let fast_sma_v = sma(&SmaInput::from_slice(
        &volume,
        SmaParams { period: Some(fast) },
    ))?
    .values;

    let mut vwma_fast = vec![f64::NAN; data_len];
    for i in 0..data_len {
        let denom = fast_sma_v[i];
        if !denom.is_nan() && denom != 0.0 {
            vwma_fast[i] = fast_sma_cv[i] / denom;
        }
    }

    let mut macd_values = vec![f64::NAN; data_len];
    for i in 0..data_len {
        if !vwma_fast[i].is_nan() && !vwma_slow[i].is_nan() {
            macd_values[i] = vwma_fast[i] - vwma_slow[i];
        }
    }

    let signal_values = ema(&EmaInput::from_slice(
        &macd_values,
        EmaParams {
            period: Some(signal),
        },
    ))?
    .values;

    let mut hist_values = vec![f64::NAN; data_len];
    for i in 0..data_len {
        if !macd_values[i].is_nan() && !signal_values[i].is_nan() {
            hist_values[i] = macd_values[i] - signal_values[i];
        }
    }

    Ok(VwmacdOutput {
        macd: macd_values,
        signal: signal_values,
        hist: hist_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_vwmacd_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = VwmacdParams {
            fast_period: None,
            slow_period: None,
            signal_period: None,
        };
        let input_default = VwmacdInput::from_candles(&candles, "close", "volume", default_params);
        let output_default = vwmacd(&input_default).expect("Failed VWMACD with default params");
        assert_eq!(output_default.macd.len(), candles.close.len());

        let custom_params = VwmacdParams {
            fast_period: Some(10),
            slow_period: Some(20),
            signal_period: Some(5),
        };
        let input_custom = VwmacdInput::from_candles(&candles, "close", "volume", custom_params);
        let output_custom = vwmacd(&input_custom).expect("Failed VWMACD with custom params");
        assert_eq!(output_custom.macd.len(), candles.close.len());
    }

    #[test]
    #[ignore]
    fn test_vwmacd_accuracy_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = VwmacdInput::with_default_candles(&candles);
        let result = vwmacd(&input).expect("Failed to calculate VWMACD");

        let last_five = &result.macd[result.macd.len().saturating_sub(5)..];
        let expected = [
            -0.40358334248536065,
            -0.16292768139917702,
            -0.4792942916867958,
            -0.1188231211518107,
            -3.3492674990910025,
        ];
        for (i, &val) in last_five.iter().enumerate() {
            assert!(
                (val - expected[i]).abs() < 1e-7,
                "Mismatch at index {}: got {}, expected {}",
                i,
                val,
                expected[i]
            );
        }
    }

    #[test]
    fn test_vwmacd_all_nan_data() {
        let close_data = [f64::NAN, f64::NAN];
        let vol_data = [f64::NAN, f64::NAN];
        let params = VwmacdParams::default();
        let input = VwmacdInput::from_slices(&close_data, &vol_data, params);
        let result = vwmacd(&input);
        assert!(result.is_err(), "Expected AllValuesNaN error");
    }

    #[test]
    fn test_vwmacd_with_zero_period() {
        let close_data = [10.0, 20.0, 30.0];
        let vol_data = [1.0, 1.0, 1.0];
        let params = VwmacdParams {
            fast_period: Some(0),
            slow_period: Some(26),
            signal_period: Some(9),
        };
        let input = VwmacdInput::from_slices(&close_data, &vol_data, params);
        let result = vwmacd(&input);
        assert!(result.is_err(), "Expected an error for zero fast period");
    }

    #[test]
    fn test_vwmacd_with_period_exceeding_data_length() {
        let close_data = [10.0, 20.0, 30.0];
        let vol_data = [100.0, 200.0, 300.0];
        let params = VwmacdParams {
            fast_period: Some(12),
            slow_period: Some(26),
            signal_period: Some(9),
        };
        let input = VwmacdInput::from_slices(&close_data, &vol_data, params);
        let result = vwmacd(&input);
        assert!(
            result.is_err(),
            "Expected error for fast/slow period > data length"
        );
    }
}
