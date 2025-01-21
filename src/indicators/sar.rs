/// # Parabolic SAR (SAR)
///
/// The Parabolic SAR is a trend-following indicator that uses a system of
/// progressively rising (in an uptrend) or falling (in a downtrend) dots.
///
/// ## Parameters
/// - **acceleration**: Acceleration factor. Defaults to 0.02.
/// - **maximum**: Maximum acceleration. Defaults to 0.2.
///
/// ## Errors
/// - **EmptyData**: sar: Input data slice is empty.
/// - **AllValuesNaN**: sar: All high/low values are `NaN`.
/// - **NotEnoughValidData**: sar: Fewer than 2 valid (non-`NaN`) data points remain
///   after the first valid index.
///
/// ## Returns
/// - **`Ok(SarOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the calculation starts.
/// - **`Err(SarError)`** otherwise.
use crate::utilities::data_loader::Candles;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum SarData<'a> {
    Candles { candles: &'a Candles },
    Slices { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct SarOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SarParams {
    pub acceleration: Option<f64>,
    pub maximum: Option<f64>,
}

impl Default for SarParams {
    fn default() -> Self {
        Self {
            acceleration: Some(0.02),
            maximum: Some(0.2),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SarInput<'a> {
    pub data: SarData<'a>,
    pub params: SarParams,
}

impl<'a> SarInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: SarParams) -> Result<Self, SarError> {
        if candles.high.is_empty() || candles.low.is_empty() {
            return Err(SarError::EmptyData);
        }
        Ok(Self {
            data: SarData::Candles { candles },
            params,
        })
    }

    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        params: SarParams,
    ) -> Result<Self, SarError> {
        if high.is_empty() || low.is_empty() {
            return Err(SarError::EmptyData);
        }
        Ok(Self {
            data: SarData::Slices { high, low },
            params,
        })
    }

    pub fn with_default_candles(candles: &'a Candles) -> Result<Self, SarError> {
        Self::from_candles(candles, SarParams::default())
    }

    pub fn get_acceleration(&self) -> f64 {
        self.params
            .acceleration
            .unwrap_or_else(|| SarParams::default().acceleration.unwrap())
    }

    pub fn get_maximum(&self) -> f64 {
        self.params
            .maximum
            .unwrap_or_else(|| SarParams::default().maximum.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum SarError {
    #[error("sar: Empty data provided for SAR.")]
    EmptyData,
    #[error("sar: All values are NaN.")]
    AllValuesNaN,
    #[error("sar: Not enough valid data. needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

pub fn sar(input: &SarInput) -> Result<SarOutput, SarError> {
    let (high, low) = match &input.data {
        SarData::Candles { candles } => (candles.high.as_slice(), candles.low.as_slice()),
        SarData::Slices { high, low } => (*high, *low),
    };

    if high.is_empty() || low.is_empty() {
        return Err(SarError::EmptyData);
    }

    let first_valid_idx = high
        .iter()
        .zip(low.iter())
        .position(|(&h, &l)| !h.is_nan() && !l.is_nan());
    let first_valid_idx = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(SarError::AllValuesNaN),
    };

    if (high.len() - first_valid_idx) < 2 {
        return Err(SarError::NotEnoughValidData {
            needed: 2,
            valid: high.len() - first_valid_idx,
        });
    }

    let mut values = vec![f64::NAN; high.len()];
    let acc_init = input.get_acceleration();
    let acc_max = input.get_maximum();

    let mut trend_up;
    let mut sar;
    let mut ep;
    let i0 = first_valid_idx;
    let i1 = i0 + 1;
    if high[i1] > high[i0] {
        trend_up = true;
        sar = low[i0];
        ep = high[i1];
    } else {
        trend_up = false;
        sar = high[i0];
        ep = low[i1];
    }

    let mut acc = acc_init;
    values[i0] = f64::NAN;
    values[i1] = sar;

    for i in (i1..high.len()).skip(1) {
        let mut next_sar = sar + acc * (ep - sar);
        if trend_up {
            if low[i] < next_sar {
                trend_up = false;
                next_sar = ep;
                ep = low[i];
                acc = acc_init;
            } else {
                if high[i] > ep {
                    ep = high[i];
                    acc = (acc + acc_init).min(acc_max);
                }
                let prev = i.saturating_sub(1);
                let pre_prev = i.saturating_sub(2);
                if prev < high.len() {
                    next_sar = next_sar.min(low[prev]);
                }
                if pre_prev < high.len() {
                    next_sar = next_sar.min(low[pre_prev]);
                }
            }
        } else {
            if high[i] > next_sar {
                trend_up = true;
                next_sar = ep;
                ep = high[i];
                acc = acc_init;
            } else {
                if low[i] < ep {
                    ep = low[i];
                    acc = (acc + acc_init).min(acc_max);
                }
                let prev = i.saturating_sub(1);
                let pre_prev = i.saturating_sub(2);
                if prev < high.len() {
                    next_sar = next_sar.max(high[prev]);
                }
                if pre_prev < high.len() {
                    next_sar = next_sar.max(high[pre_prev]);
                }
            }
        }
        values[i] = next_sar;
        sar = next_sar;
    }

    Ok(SarOutput { values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_sar_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = SarParams {
            acceleration: None,
            maximum: None,
        };
        let input_default = SarInput::from_candles(&candles, default_params)
            .expect("Failed to create SAR input with default params");
        let output_default = sar(&input_default).expect("Failed SAR with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let custom_params = SarParams {
            acceleration: Some(0.03),
            maximum: Some(0.25),
        };
        let input_custom = SarInput::from_candles(&candles, custom_params)
            .expect("Failed to create SAR input with custom params");
        let output_custom = sar(&input_custom).expect("Failed SAR with custom params");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_sar_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = SarParams {
            acceleration: Some(0.02),
            maximum: Some(0.2),
        };
        let input = SarInput::from_candles(&candles, params).expect("Failed SAR input creation");
        let sar_result = sar(&input).expect("Failed to calculate SAR");
        assert_eq!(sar_result.values.len(), candles.close.len());

        let expected_last_five_sar = [
            60370.00224209362,
            60220.362107568006,
            60079.70038111392,
            59947.478358247085,
            59823.189656752256,
        ];
        assert!(sar_result.values.len() >= 5);
        let start_index = sar_result.values.len() - 5;
        let actual_last_five = &sar_result.values[start_index..];
        for (i, &val) in actual_last_five.iter().enumerate() {
            let exp_val = expected_last_five_sar[i];
            assert!(
                (val - exp_val).abs() < 1e-4,
                "Mismatch at last five index {}: expected {}, got {}",
                i,
                exp_val,
                val
            );
        }
    }

    #[test]
    fn test_sar_from_slices() {
        let high = [50000.0, 50500.0, 51000.0];
        let low = [49000.0, 49500.0, 49900.0];
        let params = SarParams::default();
        let input = SarInput::from_slices(&high, &low, params).expect("Failed slices input");
        let result = sar(&input).expect("Failed SAR calculation from slices");
        assert_eq!(result.values.len(), high.len());
    }

    #[test]
    fn test_sar_all_nan() {
        let high = [f64::NAN, f64::NAN, f64::NAN];
        let low = [f64::NAN, f64::NAN, f64::NAN];
        let params = SarParams::default();
        let input = SarInput::from_slices(&high, &low, params).unwrap();
        let result = sar(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("All values are NaN"));
        }
    }
}
