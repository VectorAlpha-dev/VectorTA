/// # Elder's Force Index (EFI)
///
/// The Elder's Force Index (EFI) measures the power behind a price move using both price change and volume.
/// EFI is typically calculated by taking the difference in price (current - previous) multiplied by volume,
/// and then applying an EMA to that result.
///
/// ## Parameters
/// - **period**: The window size for the EMA. Defaults to 13.
///
/// ## Errors
/// - **EmptyData**: efi: Input data slice is empty or volumes are missing.
/// - **InvalidPeriod**: efi: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: efi: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index in the difference array (or fewer than 2 data points total to begin calculating).
/// - **AllValuesNaN**: efi: All input data values (or volumes) are `NaN`.
///
/// ## Returns
/// - **`Ok(EfiOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the EFI window is valid. The very first calculated difference also leads to `NaN`
///   since EFI requires at least two points to compute the initial price difference.
/// - **`Err(EfiError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum EfiData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice {
        price: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct EfiOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct EfiParams {
    pub period: Option<usize>,
}

impl Default for EfiParams {
    fn default() -> Self {
        Self { period: Some(13) }
    }
}

#[derive(Debug, Clone)]
pub struct EfiInput<'a> {
    pub data: EfiData<'a>,
    pub params: EfiParams,
}

impl<'a> EfiInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: EfiParams) -> Self {
        Self {
            data: EfiData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slices(price: &'a [f64], volume: &'a [f64], params: EfiParams) -> Self {
        Self {
            data: EfiData::Slice { price, volume },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: EfiData::Candles {
                candles,
                source: "close",
            },
            params: EfiParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| EfiParams::default().period.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum EfiError {
    #[error("efi: Empty data provided.")]
    EmptyData,
    #[error("efi: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("efi: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("efi: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn efi(input: &EfiInput) -> Result<EfiOutput, EfiError> {
    let (price, volume): (&[f64], &[f64]) = match &input.data {
        EfiData::Candles { candles, source } => {
            let p = source_type(candles, source);
            let v = &candles.volume;
            (p, v)
        }
        EfiData::Slice { price, volume } => (price, volume),
    };

    if price.is_empty() || volume.is_empty() {
        return Err(EfiError::EmptyData);
    }
    if price.len() != volume.len() {
        return Err(EfiError::EmptyData);
    }
    let len = price.len();
    let period = input.get_period();
    if period == 0 || period > len {
        return Err(EfiError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    let first_valid_idx = price
        .iter()
        .zip(volume.iter())
        .position(|(p, v)| !p.is_nan() && !v.is_nan());
    if first_valid_idx.is_none() {
        return Err(EfiError::AllValuesNaN);
    }
    let first_valid_idx = first_valid_idx.unwrap();

    if (len - first_valid_idx) < 2 {
        return Err(EfiError::NotEnoughValidData {
            needed: 2,
            valid: len - first_valid_idx,
        });
    }

    let alpha = 2.0 / (period as f64 + 1.0);
    let mut efi_values = vec![f64::NAN; len];
    let mut valid_dif_idx = None;
    for i in (first_valid_idx + 1)..len {
        if !price[i].is_nan() && !price[i - 1].is_nan() && !volume[i].is_nan() {
            efi_values[i] = (price[i] - price[i - 1]) * volume[i];
            valid_dif_idx = Some(i);
            break;
        }
    }
    let start_idx = match valid_dif_idx {
        Some(idx) => idx,
        None => return Err(EfiError::AllValuesNaN),
    };

    if (len - start_idx) < period {
        return Err(EfiError::NotEnoughValidData {
            needed: period,
            valid: len - start_idx,
        });
    }

    for i in (start_idx + 1)..len {
        let prev_ema = efi_values[i - 1];
        if price[i].is_nan() || price[i - 1].is_nan() || volume[i].is_nan() {
            efi_values[i] = prev_ema;
        } else {
            let current_dif = (price[i] - price[i - 1]) * volume[i];
            efi_values[i] = alpha * current_dif + (1.0 - alpha) * prev_ema;
        }
    }

    Ok(EfiOutput { values: efi_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_efi_empty_data() {
        let price: [f64; 0] = [];
        let volume: [f64; 0] = [];
        let params = EfiParams::default();
        let input = EfiInput::from_slices(&price, &volume, params);
        let result = efi(&input);
        assert!(result.is_err(), "Expected error for empty data");
        assert!(
            matches!(result, Err(EfiError::EmptyData)),
            "Should return EfiError::EmptyData"
        );
    }

    #[test]
    fn test_efi_mismatched_length() {
        let price = [10.0, 20.0];
        let volume = [100.0];
        let params = EfiParams::default();
        let input = EfiInput::from_slices(&price, &volume, params);
        let result = efi(&input);
        assert!(
            result.is_err(),
            "Expected error for mismatched slice lengths"
        );
        assert!(
            matches!(result, Err(EfiError::EmptyData)),
            "Should return EfiError::EmptyData for mismatched lengths"
        );
    }

    #[test]
    fn test_efi_zero_period() {
        let price = [10.0, 20.0, 30.0];
        let volume = [100.0, 200.0, 300.0];
        let params = EfiParams { period: Some(0) };
        let input = EfiInput::from_slices(&price, &volume, params);
        let result = efi(&input);
        assert!(result.is_err(), "Expected an error for zero period");
        assert!(
            matches!(result, Err(EfiError::InvalidPeriod { .. })),
            "Should return EfiError::InvalidPeriod"
        );
    }

    #[test]
    fn test_efi_period_exceeding_length() {
        let price = [10.0, 20.0, 30.0];
        let volume = [100.0, 200.0, 300.0];
        let params = EfiParams { period: Some(10) };
        let input = EfiInput::from_slices(&price, &volume, params);
        let result = efi(&input);
        assert!(
            result.is_err(),
            "Expected an error for period > data length"
        );
        assert!(
            matches!(result, Err(EfiError::InvalidPeriod { .. })),
            "Should return EfiError::InvalidPeriod"
        );
    }

    #[test]
    fn test_efi_all_nan() {
        let price = [f64::NAN, f64::NAN, f64::NAN];
        let volume = [f64::NAN, f64::NAN, f64::NAN];
        let params = EfiParams::default();
        let input = EfiInput::from_slices(&price, &volume, params);
        let result = efi(&input);
        assert!(result.is_err(), "Expected an error for all NaN data");
    }

    #[test]
    fn test_efi_not_enough_data() {
        let price = [10.0];
        let volume = [500.0];
        let params = EfiParams::default();
        let input = EfiInput::from_slices(&price, &volume, params);
        let result = efi(&input);
        assert!(
            result.is_err(),
            "Expected an error with fewer than 2 data points"
        );
    }

    #[test]
    fn test_efi_default_params() {
        let default_params = EfiParams::default();
        assert_eq!(
            default_params.period,
            Some(13),
            "Expected default period to be 13"
        );
    }

    #[test]
    fn test_efi_from_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = EfiParams::default();

        let input = EfiInput::from_candles(&candles, "close", params);
        let result = efi(&input).expect("Failed to calculate EFI from candles");
        assert_eq!(result.values.len(), candles.close.len());
    }

    #[test]
    fn test_efi_known_values_check() {
        let price = [
            50000.0, 50100.0, 50250.0, 50120.0, 50050.0, 49900.0, 49850.0, 49920.0, 49700.0,
            49650.0,
        ];
        let volume = [
            500.0, 520.0, 530.0, 515.0, 510.0, 505.0, 503.0, 510.0, 490.0, 480.0,
        ];
        let params = EfiParams { period: Some(5) };
        let input = EfiInput::from_slices(&price, &volume, params);
        let result = efi(&input).expect("Failed to calculate EFI");
        let values = result.values;
        assert_eq!(values.len(), price.len());
        for i in 5..values.len() {
            assert!(!values[i].is_nan(), "Expected value at index {}", i);
        }
    }

    #[test]
    fn test_efi_provided_sample_values() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = EfiParams { period: Some(13) };
        let input = EfiInput::from_candles(&candles, "close", params);
        let result = efi(&input).expect("Failed to calculate EFI");
        let values = result.values;
        assert_eq!(values.len(), candles.close.len());
        let last_five = &values[values.len().saturating_sub(5)..];
        let expected = [
            -44604.382026531224,
            -39811.02321812391,
            -36599.9671820205,
            -29903.28014503471,
            -55406.09054645832,
        ];
        for (i, &val) in last_five.iter().enumerate() {
            assert!(
                (val - expected[i]).abs() < 1.0,
                "Mismatch at index {}: expected {}, got {}",
                i,
                expected[i],
                val
            );
        }
    }

    #[test]
    fn test_efi_recalculation_via_slice() {
        let price = [100.0, 102.0, 101.0, 103.0, 105.0, 107.0];
        let volume = [50.0, 60.0, 55.0, 65.0, 70.0, 75.0];
        let first_params = EfiParams { period: Some(3) };
        let first_input = EfiInput::from_slices(&price, &volume, first_params);
        let first_result = efi(&first_input).expect("Failed to calculate first EFI");
        assert_eq!(first_result.values.len(), price.len());
        let second_params = EfiParams { period: Some(2) };
        let second_input = EfiInput::from_slices(&first_result.values, &volume, second_params);
        let second_result = efi(&second_input).expect("Failed to calculate second EFI");
        assert_eq!(second_result.values.len(), price.len());
        for i in 2..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "Expected no NaN in second EFI at index {}",
                i
            );
        }
    }
}
