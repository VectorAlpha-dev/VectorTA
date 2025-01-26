/// # Volume Price Trend (VPT)
///
/// This version matches Jesse's implementation exactly, using a shifted array approach:
///
/// ```python
/// vpt_val = volume * ((price - shift(price,1)) / shift(price,1))
/// res = shift(vpt_val,1) + vpt_val
/// ```
///
/// Note that this differs from the *typical* "cumulative from bar 0" formula.
/// Jesse's method tends to produce smaller magnitudes. If your goal is to
/// match code from jesse's `vpt` function exactly, use this approach.
///
/// ## Errors
/// - **EmptyData**: vpt: Input data slice is empty.
/// - **AllValuesNaN**: vpt: All input data values are `NaN`.
/// - **NotEnoughValidData**: vpt: Fewer than 2 valid (non-`NaN`) price points remain.
///
/// ## Returns
/// - **`Ok(VptOutput)`** on success, containing an array that directly matches
///   Jesse's final array (with the same shifting behavior).
/// - **`Err(VptError)`** on failure.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum VptData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slices {
        price: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct VptOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct VptParams;

#[derive(Debug, Clone)]
pub struct VptInput<'a> {
    pub data: VptData<'a>,
    pub params: VptParams,
}

impl<'a> VptInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str) -> Self {
        Self {
            data: VptData::Candles { candles, source },
            params: VptParams::default(),
        }
    }

    pub fn from_slices(price: &'a [f64], volume: &'a [f64]) -> Self {
        Self {
            data: VptData::Slices { price, volume },
            params: VptParams::default(),
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: VptData::Candles {
                candles,
                source: "close",
            },
            params: VptParams::default(),
        }
    }
}

#[derive(Debug, Error)]
pub enum VptError {
    #[error("vpt: Empty data provided.")]
    EmptyData,
    #[error("vpt: All price/volume values are NaN.")]
    AllValuesNaN,
    #[error("vpt: Not enough valid data (fewer than 2 valid points).")]
    NotEnoughValidData,
}

#[inline]
pub fn vpt(input: &VptInput) -> Result<VptOutput, VptError> {
    let (price, volume) = match &input.data {
        VptData::Candles { candles, source } => {
            let price = source_type(candles, source);
            let vol = candles
                .select_candle_field("volume")
                .map_err(|_| VptError::EmptyData)?;
            (price, vol)
        }
        VptData::Slices { price, volume } => (*price, *volume),
    };

    if price.is_empty() || volume.is_empty() || price.len() != volume.len() {
        return Err(VptError::EmptyData);
    }

    let valid_count = price
        .iter()
        .zip(volume.iter())
        .filter(|(&p, &v)| !(p.is_nan() || v.is_nan()))
        .count();

    if valid_count == 0 {
        return Err(VptError::AllValuesNaN);
    }
    if valid_count < 2 {
        return Err(VptError::NotEnoughValidData);
    }

    let n = price.len();
    let mut vpt_val = vec![f64::NAN; n];
    for i in 1..n {
        let p0 = price[i - 1];
        let p1 = price[i];
        let v1 = volume[i];
        if p0.is_nan() || p0 == 0.0 || p1.is_nan() || v1.is_nan() {
            vpt_val[i] = f64::NAN;
        } else {
            vpt_val[i] = v1 * ((p1 - p0) / p0);
        }
    }

    let mut res = vec![f64::NAN; n];
    for i in 1..n {
        let shifted = vpt_val[i - 1];
        if vpt_val[i].is_nan() || shifted.is_nan() {
            res[i] = f64::NAN;
        } else {
            res[i] = vpt_val[i] + shifted;
        }
    }

    Ok(VptOutput { values: res })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_vpt_basic_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = VptInput::from_candles(&candles, "close");
        let output = vpt(&input).expect("Failed to calculate VPT");
        assert_eq!(output.values.len(), candles.close.len());
    }

    #[test]
    fn test_vpt_basic_slices() {
        let price = [1.0, 1.1, 1.05, 1.2, 1.3];
        let volume = [1000.0, 1100.0, 1200.0, 1300.0, 1400.0];
        let input = VptInput::from_slices(&price, &volume);
        let output = vpt(&input).expect("Failed to calculate VPT from slices");
        assert_eq!(output.values.len(), price.len());
    }

    #[test]
    fn test_vpt_not_enough_data() {
        let price = [100.0];
        let volume = [500.0];
        let input = VptInput::from_slices(&price, &volume);
        let result = vpt(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_vpt_empty_data() {
        let price: [f64; 0] = [];
        let volume: [f64; 0] = [];
        let input = VptInput::from_slices(&price, &volume);
        let result = vpt(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_vpt_all_nan() {
        let price = [f64::NAN, f64::NAN, f64::NAN];
        let volume = [f64::NAN, f64::NAN, f64::NAN];
        let input = VptInput::from_slices(&price, &volume);
        let result = vpt(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_vpt_accuracy_from_csv() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = VptInput::from_candles(&candles, "close");
        let output = vpt(&input).expect("Failed to calculate VPT");

        let expected_last_five = [
            -0.40358334248536065,
            -0.16292768139917702,
            -0.4792942916867958,
            -0.1188231211518107,
            -3.3492674990910025,
        ];

        assert!(
            output.values.len() >= 5,
            "VPT array is too short to compare last 5 values"
        );
        let start_index = output.values.len() - 5;
        for (i, &value) in output.values[start_index..].iter().enumerate() {
            let expected_value = expected_last_five[i];
            assert!(
                (value - expected_value).abs() < 1e-3,
                "VPT mismatch at final bars, index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
    }
}
