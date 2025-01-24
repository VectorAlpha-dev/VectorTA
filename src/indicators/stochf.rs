use crate::indicators::sma::{sma, SmaData, SmaInput, SmaOutput, SmaParams};
/// # Stochastic Fast (StochF)
///
/// A momentum indicator comparing a security’s closing price to its price range (high-low) over
/// a specified lookback (`fastk_period`). It then applies a moving average (`fastd_period`) on
/// the %K values to obtain %D. This variant of the Stochastic oscillator is known as "fast"
/// because it uses shorter averaging, making it more sensitive to price changes.
///
/// ## Parameters
/// - **fastk_period**: Lookback period for the highest high and lowest low. Defaults to 5.
/// - **fastd_period**: Period for the moving average of %K. Defaults to 3.
/// - **fastd_matype**: Moving average type (only SMA=0 supported here). Defaults to 0.
///
/// ## Errors
/// - **EmptyData**: stochf: Input data slice(s) are empty.
/// - **InvalidPeriod**: stochf: A provided period is zero or exceeds the data length.
/// - **AllValuesNaN**: stochf: All input data values are `NaN`.
/// - **NotEnoughValidData**: stochf: Not enough valid (non-NaN) data remain after the first valid index.
///
/// ## Returns
/// - **`Ok(StochfOutput)`** on success, containing two `Vec<f64>` (%K and %D) matching the input length,
///   with leading `NaN`s until each series can be computed.
/// - **`Err(StochfError)`** otherwise.
use crate::utilities::data_loader::Candles;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum StochfData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct StochfOutput {
    pub k: Vec<f64>,
    pub d: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct StochfParams {
    pub fastk_period: Option<usize>,
    pub fastd_period: Option<usize>,
    pub fastd_matype: Option<usize>,
}

impl Default for StochfParams {
    fn default() -> Self {
        Self {
            fastk_period: Some(5),
            fastd_period: Some(3),
            fastd_matype: Some(0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StochfInput<'a> {
    pub data: StochfData<'a>,
    pub params: StochfParams,
}

impl<'a> StochfInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: StochfParams) -> Self {
        Self {
            data: StochfData::Candles { candles },
            params,
        }
    }

    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: StochfParams,
    ) -> Self {
        Self {
            data: StochfData::Slices { high, low, close },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: StochfData::Candles { candles },
            params: StochfParams::default(),
        }
    }

    pub fn get_fastk_period(&self) -> usize {
        self.params
            .fastk_period
            .unwrap_or_else(|| StochfParams::default().fastk_period.unwrap())
    }

    pub fn get_fastd_period(&self) -> usize {
        self.params
            .fastd_period
            .unwrap_or_else(|| StochfParams::default().fastd_period.unwrap())
    }

    pub fn get_fastd_matype(&self) -> usize {
        self.params
            .fastd_matype
            .unwrap_or_else(|| StochfParams::default().fastd_matype.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum StochfError {
    #[error("stochf: Empty data provided.")]
    EmptyData,
    #[error("stochf: Invalid period (fastk={fastk}, fastd={fastd}), data length={data_len}.")]
    InvalidPeriod {
        fastk: usize,
        fastd: usize,
        data_len: usize,
    },
    #[error("stochf: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "stochf: Not enough valid data after first valid index (needed={needed}, valid={valid})."
    )]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn stochf(input: &StochfInput) -> Result<StochfOutput, StochfError> {
    let (high, low, close) = match &input.data {
        StochfData::Candles { candles } => {
            let high = candles
                .select_candle_field("high")
                .map_err(|_| StochfError::EmptyData)?;
            let low = candles
                .select_candle_field("low")
                .map_err(|_| StochfError::EmptyData)?;
            let close = candles
                .select_candle_field("close")
                .map_err(|_| StochfError::EmptyData)?;
            (high, low, close)
        }
        StochfData::Slices { high, low, close } => (*high, *low, *close),
    };

    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(StochfError::EmptyData);
    }

    let len = high.len();
    if low.len() != len || close.len() != len {
        return Err(StochfError::EmptyData);
    }

    let fastk_period = input.get_fastk_period();
    let fastd_period = input.get_fastd_period();
    let matype = input.get_fastd_matype();

    if fastk_period == 0 || fastd_period == 0 || fastk_period > len || fastd_period > len {
        return Err(StochfError::InvalidPeriod {
            fastk: fastk_period,
            fastd: fastd_period,
            data_len: len,
        });
    }

    let first_valid_idx = {
        let mut idx_opt = None;
        for i in 0..len {
            if !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan() {
                idx_opt = Some(i);
                break;
            }
        }
        match idx_opt {
            Some(i) => i,
            None => return Err(StochfError::AllValuesNaN),
        }
    };

    if (len - first_valid_idx) < fastk_period {
        return Err(StochfError::NotEnoughValidData {
            needed: fastk_period,
            valid: len - first_valid_idx,
        });
    }

    let mut k_vals = vec![f64::NAN; len];
    let mut d_vals = vec![f64::NAN; len];

    for i in first_valid_idx..len {
        if i < first_valid_idx + fastk_period - 1 {
            continue;
        }
        let start = i + 1 - fastk_period;
        let (hh, ll) = {
            let mut max_h = f64::NEG_INFINITY;
            let mut min_l = f64::INFINITY;
            for j in start..=i {
                let h_j = high[j];
                let l_j = low[j];
                if h_j > max_h {
                    max_h = h_j;
                }
                if l_j < min_l {
                    min_l = l_j;
                }
            }
            (max_h, min_l)
        };
        if hh == ll {
            if close[i] == hh {
                k_vals[i] = 100.0;
            } else {
                k_vals[i] = 0.0;
            }
        } else {
            k_vals[i] = 100.0 * (close[i] - ll) / (hh - ll);
        }
    }

    if matype != 0 {
        return Err(StochfError::InvalidPeriod {
            fastk: fastk_period,
            fastd: fastd_period,
            data_len: len,
        });
    }

    let d_input = SmaInput {
        data: SmaData::Slice(&k_vals),
        params: SmaParams {
            period: Some(fastd_period),
        },
    };
    let SmaOutput { values: d_result } = match sma(&d_input) {
        Ok(res) => res,
        Err(_) => {
            return Err(StochfError::NotEnoughValidData {
                needed: fastd_period,
                valid: len - first_valid_idx,
            })
        }
    };
    for i in 0..len {
        d_vals[i] = d_result[i];
    }

    Ok(StochfOutput {
        k: k_vals,
        d: d_vals,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_stochf_basic_functionality() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = StochfParams {
            fastk_period: Some(5),
            fastd_period: Some(3),
            fastd_matype: Some(0),
        };
        let input = StochfInput::from_candles(&candles, params);
        let result = stochf(&input).expect("Failed to compute StochF");
        assert_eq!(result.k.len(), candles.close.len());
        assert_eq!(result.d.len(), candles.close.len());
    }

    #[test]
    fn test_stochf_accuracy_last_five() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = StochfParams {
            fastk_period: Some(5),
            fastd_period: Some(3),
            fastd_matype: Some(0),
        };
        let input = StochfInput::from_candles(&candles, params);
        let output = stochf(&input).expect("Failed to compute StochF");

        let expected_k = [
            80.6987399770905,
            40.88471849865952,
            15.507246376811594,
            36.920529801324506,
            32.1880650994575,
        ];
        let expected_d = [
            70.99960994145033,
            61.44725644908976,
            45.696901617520815,
            31.104164892265487,
            28.205280425864817,
        ];

        let k_len = output.k.len();
        let d_len = output.d.len();
        assert!(
            k_len >= 5 && d_len >= 5,
            "Not enough data to test last 5 values"
        );

        let k_slice = &output.k[k_len - 5..];
        let d_slice = &output.d[d_len - 5..];

        for i in 0..5 {
            let diff_k = (k_slice[i] - expected_k[i]).abs();
            let diff_d = (d_slice[i] - expected_d[i]).abs();
            assert!(
                diff_k < 1e-4,
                "Mismatch in K at last 5 index {}: expected {}, got {}",
                i,
                expected_k[i],
                k_slice[i]
            );
            assert!(
                diff_d < 1e-4,
                "Mismatch in D at last 5 index {}: expected {}, got {}",
                i,
                expected_d[i],
                d_slice[i]
            );
        }
    }

    #[test]
    fn test_stochf_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = StochfInput::with_default_candles(&candles);
        let output = stochf(&input).expect("Failed to compute StochF with defaults");
        assert_eq!(output.k.len(), candles.close.len());
        assert_eq!(output.d.len(), candles.close.len());
    }

    #[test]
    fn test_stochf_empty_data() {
        let params = StochfParams::default();
        let input = StochfInput::from_slices(&[], &[], &[], params);
        let result = stochf(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Empty data"),
                "Expected EmptyData error, got {}",
                e
            );
        }
    }

    #[test]
    fn test_stochf_zero_period() {
        let input_data = [10.0, 20.0, 30.0, 40.0, 50.0];
        let params = StochfParams {
            fastk_period: Some(0),
            fastd_period: Some(3),
            fastd_matype: Some(0),
        };
        let input = StochfInput::from_slices(&input_data, &input_data, &input_data, params);
        let result = stochf(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Invalid period"),
                "Expected 'Invalid period' error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_stochf_period_exceeding_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = StochfParams {
            fastk_period: Some(10),
            fastd_period: Some(3),
            fastd_matype: Some(0),
        };
        let input = StochfInput::from_slices(&input_data, &input_data, &input_data, params);
        let result = stochf(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_stochf_all_nan() {
        let input_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = StochfParams::default();
        let input = StochfInput::from_slices(&input_data, &input_data, &input_data, params);
        let result = stochf(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_stochf_not_enough_valid_data() {
        let input_data = [f64::NAN, 10.0, 20.0, 30.0];
        let params = StochfParams {
            fastk_period: Some(5),
            fastd_period: Some(3),
            fastd_matype: Some(0),
        };
        let input = StochfInput::from_slices(&input_data, &input_data, &input_data, params);
        let result = stochf(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_stochf_slice_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let first_params = StochfParams {
            fastk_period: Some(5),
            fastd_period: Some(3),
            fastd_matype: Some(0),
        };
        let first_input = StochfInput::from_candles(&candles, first_params);
        let first_result = stochf(&first_input).expect("Failed to compute StochF");

        let second_params = StochfParams {
            fastk_period: Some(5),
            fastd_period: Some(3),
            fastd_matype: Some(0),
        };
        let second_input = StochfInput::from_slices(
            &first_result.k,
            &first_result.k,
            &first_result.k,
            second_params,
        );
        let second_result = stochf(&second_input).expect("Failed to compute StochF second time");
        assert_eq!(second_result.k.len(), first_result.k.len());
        assert_eq!(second_result.d.len(), first_result.d.len());
    }
}
