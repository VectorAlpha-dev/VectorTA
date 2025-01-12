use crate::utilities::data_loader::{source_type, Candles};
/// # Moving Average Convergence/Divergence (MACD)
///
/// A trend-following momentum indicator that shows the relationship between two moving averages of a data series.
/// The MACD is calculated by subtracting the "slow" moving average from the "fast" moving average. A "signal" moving
/// average of the MACD line is then plotted on top of the MACD line, which can function as a trigger for buy/sell signals.
///
/// ## Parameters
/// - **fast_period**: The short moving average period. Defaults to 12.
/// - **slow_period**: The long moving average period. Defaults to 26.
/// - **signal_period**: The signal line moving average period. Defaults to 9.
/// - **ma_type**: The type of moving average used for the MACD calculation. Defaults to "ema".
///
/// ## Errors
/// - **EmptyData**: macd: Input data slice is empty.
/// - **InvalidPeriod**: macd: One or more periods is zero or exceeds the data length.
/// - **NotEnoughValidData**: macd: Fewer valid (non-`NaN`) data points remain after the first valid index than needed.
/// - **AllValuesNaN**: macd: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(MacdOutput)`** on success, containing three `Vec<f64>` (macd, signal, hist), each matching the input length,
///   with leading `NaN`s until the respective moving averages become available.
/// - **`Err(MacdError)`** otherwise.
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum MacdData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct MacdOutput {
    pub macd: Vec<f64>,
    pub signal: Vec<f64>,
    pub hist: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MacdParams {
    pub fast_period: Option<usize>,
    pub slow_period: Option<usize>,
    pub signal_period: Option<usize>,
    pub ma_type: Option<String>,
}

impl Default for MacdParams {
    fn default() -> Self {
        Self {
            fast_period: Some(12),
            slow_period: Some(26),
            signal_period: Some(9),
            ma_type: Some("ema".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MacdInput<'a> {
    pub data: MacdData<'a>,
    pub params: MacdParams,
}

impl<'a> MacdInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: MacdParams) -> Self {
        Self {
            data: MacdData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: MacdParams) -> Self {
        Self {
            data: MacdData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: MacdData::Candles {
                candles,
                source: "close",
            },
            params: MacdParams::default(),
        }
    }

    pub fn get_fast_period(&self) -> usize {
        self.params
            .fast_period
            .unwrap_or_else(|| MacdParams::default().fast_period.unwrap())
    }

    pub fn get_slow_period(&self) -> usize {
        self.params
            .slow_period
            .unwrap_or_else(|| MacdParams::default().slow_period.unwrap())
    }

    pub fn get_signal_period(&self) -> usize {
        self.params
            .signal_period
            .unwrap_or_else(|| MacdParams::default().signal_period.unwrap())
    }

    pub fn get_ma_type(&self) -> String {
        self.params
            .ma_type
            .clone()
            .unwrap_or_else(|| MacdParams::default().ma_type.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum MacdError {
    #[error("macd: Empty data provided.")]
    EmptyData,
    #[error("macd: Invalid period (fast={fast}, slow={slow}, signal={signal}), data length = {data_len}")]
    InvalidPeriod {
        fast: usize,
        slow: usize,
        signal: usize,
        data_len: usize,
    },
    #[error("macd: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("macd: All values are NaN.")]
    AllValuesNaN,
}

#[derive(Debug, Clone)]
pub struct MacdOutputBuilder {
    macd: Vec<f64>,
    signal: Vec<f64>,
    hist: Vec<f64>,
}

impl MacdOutputBuilder {
    pub fn new(len: usize) -> Self {
        Self {
            macd: vec![f64::NAN; len],
            signal: vec![f64::NAN; len],
            hist: vec![f64::NAN; len],
        }
    }

    pub fn set_macd(&mut self, idx: usize, val: f64) {
        self.macd[idx] = val;
    }

    pub fn set_signal(&mut self, idx: usize, val: f64) {
        self.signal[idx] = val;
    }

    pub fn set_hist(&mut self, idx: usize, val: f64) {
        self.hist[idx] = val;
    }

    pub fn build(self) -> MacdOutput {
        MacdOutput {
            macd: self.macd,
            signal: self.signal,
            hist: self.hist,
        }
    }
}

#[inline]
pub fn macd(input: &MacdInput) -> Result<MacdOutput, MacdError> {
    let data: &[f64] = match &input.data {
        MacdData::Candles { candles, source } => source_type(candles, source),
        MacdData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(MacdError::EmptyData);
    }

    let fast_period = input.get_fast_period();
    let slow_period = input.get_slow_period();
    let signal_period = input.get_signal_period();
    let length = data.len();

    if fast_period == 0
        || slow_period == 0
        || signal_period == 0
        || fast_period > length
        || slow_period > length
        || signal_period > length
    {
        return Err(MacdError::InvalidPeriod {
            fast: fast_period,
            slow: slow_period,
            signal: signal_period,
            data_len: length,
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(MacdError::AllValuesNaN),
    };

    if (data.len() - first_valid_idx) < slow_period {
        return Err(MacdError::NotEnoughValidData {
            needed: slow_period,
            valid: data.len() - first_valid_idx,
        });
    }

    let mut output_builder = MacdOutputBuilder::new(data.len());
    let ma_type = input.get_ma_type();

    let fast_ma = crate::indicators::moving_averages::ma::ma(
        &ma_type,
        crate::indicators::moving_averages::ma::MaData::Slice(data),
        fast_period,
    )
    .map_err(|_| MacdError::AllValuesNaN)?;
    let slow_ma = crate::indicators::moving_averages::ma::ma(
        &ma_type,
        crate::indicators::moving_averages::ma::MaData::Slice(data),
        slow_period,
    )
    .map_err(|_| MacdError::AllValuesNaN)?;

    let mut macd_line = vec![f64::NAN; data.len()];
    for i in first_valid_idx..length {
        if fast_ma[i].is_nan() || slow_ma[i].is_nan() {
            continue;
        }
        macd_line[i] = fast_ma[i] - slow_ma[i];
        output_builder.set_macd(i, macd_line[i]);
    }

    if (data.len() - first_valid_idx) < (slow_period + signal_period - 1) {
        return Ok(output_builder.build());
    }

    let signal_ma = crate::indicators::moving_averages::ma::ma(
        &ma_type,
        crate::indicators::moving_averages::ma::MaData::Slice(&macd_line),
        signal_period,
    )
    .map_err(|_| MacdError::AllValuesNaN)?;

    for i in first_valid_idx..length {
        if macd_line[i].is_nan() || signal_ma[i].is_nan() {
            continue;
        }
        let hist_val = macd_line[i] - signal_ma[i];
        output_builder.set_signal(i, signal_ma[i]);
        output_builder.set_hist(i, hist_val);
    }

    Ok(output_builder.build())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::moving_averages::ma::{ma, MaData};
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_macd_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = MacdParams {
            fast_period: None,
            slow_period: None,
            signal_period: None,
            ma_type: None,
        };
        let input_default = MacdInput::from_candles(&candles, "close", default_params);
        let output_default = macd(&input_default).expect("Failed MACD with default params");
        assert_eq!(output_default.macd.len(), candles.close.len());
        assert_eq!(output_default.signal.len(), candles.close.len());
        assert_eq!(output_default.hist.len(), candles.close.len());

        let params_custom = MacdParams {
            fast_period: Some(10),
            slow_period: Some(20),
            signal_period: Some(8),
            ma_type: Some("ema".to_string()),
        };
        let input_custom = MacdInput::from_candles(&candles, "hl2", params_custom);
        let output_custom = macd(&input_custom).expect("Failed MACD with custom params");
        assert_eq!(output_custom.macd.len(), candles.close.len());
        assert_eq!(output_custom.signal.len(), candles.close.len());
        assert_eq!(output_custom.hist.len(), candles.close.len());
    }

    #[test]
    fn test_macd_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = MacdParams {
            fast_period: Some(12),
            slow_period: Some(26),
            signal_period: Some(9),
            ma_type: Some("ema".to_string()),
        };
        let input = MacdInput::from_candles(&candles, "close", params);
        let macd_result = macd(&input).expect("Failed to calculate MACD");

        assert_eq!(macd_result.macd.len(), candles.close.len());
        assert_eq!(macd_result.signal.len(), candles.close.len());
        assert_eq!(macd_result.hist.len(), candles.close.len());

        let expected_macd_last_five = [
            -629.8674025082801,
            -600.2986584356258,
            -581.6188884820076,
            -551.1020443476082,
            -560.798510688488,
        ];
        let expected_signal_last_five = [
            -721.9744591891067,
            -697.6392990384105,
            -674.4352169271299,
            -649.7685824112256,
            -631.9745680666781,
        ];
        let expected_hist_last_five = [
            92.10705668082664,
            97.34064060278467,
            92.81632844512228,
            98.6665380636174,
            71.17605737819008,
        ];

        let len = macd_result.macd.len();
        assert!(len >= 5, "MACD length too short for final check");
        let start_idx = len - 5;

        for i in 0..5 {
            let macd_val = macd_result.macd[start_idx + i];
            let signal_val = macd_result.signal[start_idx + i];
            let hist_val = macd_result.hist[start_idx + i];

            let macd_exp = expected_macd_last_five[i];
            let signal_exp = expected_signal_last_five[i];
            let hist_exp = expected_hist_last_five[i];

            assert!(
                (macd_val - macd_exp).abs() < 1e-1,
                "MACD mismatch at index {}: expected {}, got {}",
                start_idx + i,
                macd_exp,
                macd_val
            );
            assert!(
                (signal_val - signal_exp).abs() < 1e-1,
                "Signal mismatch at index {}: expected {}, got {}",
                start_idx + i,
                signal_exp,
                signal_val
            );
            assert!(
                (hist_val - hist_exp).abs() < 1e-1,
                "Hist mismatch at index {}: expected {}, got {}",
                start_idx + i,
                hist_exp,
                hist_val
            );
        }
    }

    #[test]
    fn test_macd_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = MacdParams {
            fast_period: Some(0),
            slow_period: Some(26),
            signal_period: Some(9),
            ma_type: Some("ema".to_string()),
        };
        let input = MacdInput::from_slice(&input_data, params);

        let result = macd(&input);
        assert!(result.is_err(), "Expected an error for zero fast period");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Invalid period"),
                "Expected 'Invalid period' error message, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_macd_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = MacdParams {
            fast_period: Some(12),
            slow_period: Some(26),
            signal_period: Some(9),
            ma_type: Some("ema".to_string()),
        };
        let input = MacdInput::from_slice(&input_data, params);

        let result = macd(&input);
        assert!(
            result.is_err(),
            "Expected an error for MACD period > data.len()"
        );
    }

    #[test]
    fn test_macd_very_small_data_set() {
        let input_data = [42.0];
        let params = MacdParams {
            fast_period: Some(12),
            slow_period: Some(26),
            signal_period: Some(9),
            ma_type: Some("ema".to_string()),
        };
        let input = MacdInput::from_slice(&input_data, params);

        let result = macd(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than slow period"
        );
    }

    #[test]
    fn test_macd_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = MacdParams {
            fast_period: Some(12),
            slow_period: Some(26),
            signal_period: Some(9),
            ma_type: Some("ema".to_string()),
        };
        let first_input = MacdInput::from_candles(&candles, "close", first_params);
        let first_result = macd(&first_input).expect("Failed to calculate first MACD");

        assert_eq!(first_result.macd.len(), candles.close.len());
        assert_eq!(first_result.signal.len(), candles.close.len());
        assert_eq!(first_result.hist.len(), candles.close.len());

        let second_params = MacdParams {
            fast_period: Some(12),
            slow_period: Some(26),
            signal_period: Some(9),
            ma_type: Some("ema".to_string()),
        };
        let second_input = MacdInput::from_slice(&first_result.macd, second_params);
        let second_result = macd(&second_input).expect("Failed to calculate second MACD");

        assert_eq!(second_result.macd.len(), first_result.macd.len());
        assert_eq!(second_result.signal.len(), first_result.signal.len());
        assert_eq!(second_result.hist.len(), first_result.hist.len());

        for i in 52..second_result.macd.len() {
            assert!(
                !second_result.macd[i].is_nan(),
                "Expected no NaN after index 52, but found NaN at index {}",
                i
            );
        }
    }

    #[test]
    fn test_macd_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = &candles.close;

        let params = MacdParams {
            fast_period: Some(12),
            slow_period: Some(26),
            signal_period: Some(9),
            ma_type: Some("ema".to_string()),
        };
        let input = MacdInput::from_candles(&candles, "close", params);
        let macd_result = macd(&input).expect("Failed to calculate MACD");

        assert_eq!(macd_result.macd.len(), close_prices.len());
        assert_eq!(macd_result.signal.len(), close_prices.len());
        assert_eq!(macd_result.hist.len(), close_prices.len());

        if macd_result.macd.len() > 240 {
            for i in 240..macd_result.macd.len() {
                assert!(
                    !macd_result.macd[i].is_nan(),
                    "Expected no NaN after index 240 in macd, found NaN at {}",
                    i
                );
                assert!(
                    !macd_result.signal[i].is_nan(),
                    "Expected no NaN after index 240 in signal, found NaN at {}",
                    i
                );
                assert!(
                    !macd_result.hist[i].is_nan(),
                    "Expected no NaN after index 240 in hist, found NaN at {}",
                    i
                );
            }
        }
    }

    #[test]
    fn test_macd_params_with_default_params() {
        let default_params = MacdParams::default();
        assert_eq!(
            default_params.fast_period,
            Some(12),
            "Expected fast_period=12 in default params"
        );
        assert_eq!(
            default_params.slow_period,
            Some(26),
            "Expected slow_period=26 in default params"
        );
        assert_eq!(
            default_params.signal_period,
            Some(9),
            "Expected signal_period=9 in default params"
        );
        assert_eq!(
            default_params.ma_type,
            Some("ema".to_string()),
            "Expected ma_type=ema in default params"
        );
    }

    #[test]
    fn test_macd_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = MacdInput::with_default_candles(&candles);
        match input.data {
            MacdData::Candles { source, .. } => {
                assert_eq!(source, "close", "Expected default source to be 'close'");
            }
            _ => panic!("Expected MacdData::Candles variant"),
        }
    }
}
