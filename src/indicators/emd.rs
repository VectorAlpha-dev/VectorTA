/// # Empirical Mode Decomposition (EMD)
///
/// Implements the Empirical Mode Decomposition indicator as described by
/// John F. Ehlers and Ric Way. This version uses a band-pass filter and
/// simple moving averages to produce three output bands:
/// - **upperband**: A scaled SMA of detected peaks
/// - **middleband**: An SMA of the band-passed signal
/// - **lowerband**: A scaled SMA of detected valleys
///
/// ## Parameters
/// - **period**: The base window for the band-pass filter. Defaults to 20.
/// - **delta**: Used in the band-pass filter phase calculation. Defaults to 0.5.
/// - **fraction**: Scaling factor for peaks and valleys. Defaults to 0.1.
///
/// ## Errors
/// - **EmptyData**: emd: Input data slice is empty.
/// - **InvalidPeriod**: emd: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: emd: Fewer than the maximum needed (`max(2*period, 50)`) valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: emd: All input data values are `NaN`.
/// - **CandleFieldError**: emd: Candle field not found or could not be extracted.
///
/// ## Returns
/// - **`Ok(EmdOutput)`** on success, containing three `Vec<f64>` matching the input length,
///   with leading `NaN`s until the computation can begin.
/// - **`Err(EmdError)`** otherwise.
use crate::utilities::data_loader::{read_candles_from_csv, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum EmdData<'a> {
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
pub struct EmdOutput {
    pub upperband: Vec<f64>,
    pub middleband: Vec<f64>,
    pub lowerband: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct EmdParams {
    pub period: Option<usize>,
    pub delta: Option<f64>,
    pub fraction: Option<f64>,
}

impl Default for EmdParams {
    fn default() -> Self {
        Self {
            period: Some(20),
            delta: Some(0.5),
            fraction: Some(0.1),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmdInput<'a> {
    pub data: EmdData<'a>,
    pub params: EmdParams,
}

impl<'a> EmdInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: EmdParams) -> Self {
        Self {
            data: EmdData::Candles { candles },
            params,
        }
    }

    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
        params: EmdParams,
    ) -> Self {
        Self {
            data: EmdData::Slices {
                high,
                low,
                close,
                volume,
            },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: EmdData::Candles { candles },
            params: EmdParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| EmdParams::default().period.unwrap())
    }

    pub fn get_delta(&self) -> f64 {
        self.params
            .delta
            .unwrap_or_else(|| EmdParams::default().delta.unwrap())
    }

    pub fn get_fraction(&self) -> f64 {
        self.params
            .fraction
            .unwrap_or_else(|| EmdParams::default().fraction.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum EmdError {
    #[error("emd: Empty data provided.")]
    EmptyData,
    #[error("emd: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("emd: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("emd: All values are NaN.")]
    AllValuesNaN,
    #[error("emd: Candle field not found or could not be extracted.")]
    CandleFieldError,
}

#[inline]
pub fn emd(input: &EmdInput) -> Result<EmdOutput, EmdError> {
    let (high, low, _close, _volume) = match &input.data {
        EmdData::Candles { candles } => {
            let high = candles
                .select_candle_field("high")
                .map_err(|_| EmdError::CandleFieldError)?;
            let low = candles
                .select_candle_field("low")
                .map_err(|_| EmdError::CandleFieldError)?;
            let close = candles
                .select_candle_field("close")
                .map_err(|_| EmdError::CandleFieldError)?;
            let volume = candles
                .select_candle_field("volume")
                .map_err(|_| EmdError::CandleFieldError)?;
            (high, low, close, volume)
        }
        EmdData::Slices {
            high,
            low,
            close,
            volume,
        } => (*high, *low, *close, *volume),
    };

    let len = high.len();
    if len == 0 {
        return Err(EmdError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > len {
        return Err(EmdError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    let needed = (2 * period).max(50);
    let first_valid_idx = match (0..len).find(|&i| !high[i].is_nan() && !low[i].is_nan()) {
        Some(idx) => idx,
        None => return Err(EmdError::AllValuesNaN),
    };
    if (len - first_valid_idx) < needed {
        return Err(EmdError::NotEnoughValidData {
            needed,
            valid: len - first_valid_idx,
        });
    }

    let mut upperband = vec![f64::NAN; len];
    let mut middleband = vec![f64::NAN; len];
    let mut lowerband = vec![f64::NAN; len];

    let delta = input.get_delta();
    let fraction = input.get_fraction();
    let beta = (2.0 * std::f64::consts::PI / period as f64).cos();
    let gamma = 1.0 / ((4.0 * std::f64::consts::PI * delta / period as f64).cos());
    let alpha = gamma - (gamma * gamma - 1.0).sqrt();

    let half_one_minus_alpha = 0.5 * (1.0 - alpha);

    let per_up_low = 50;
    let per_mid = 2 * period;

    let mut sum_up = 0.0;
    let mut sum_mb = 0.0;
    let mut sum_low = 0.0;

    let mut sp_ring = vec![0.0; per_up_low];
    let mut sv_ring = vec![0.0; per_up_low];
    let mut bp_ring = vec![0.0; per_mid];
    let mut idx_up_low = 0_usize;
    let mut idx_mid = 0_usize;

    let mut bp_prev1 = 0.0;
    let mut bp_prev2 = 0.0;

    let mut peak_prev = 0.0;
    let mut valley_prev = 0.0;
    let mut initialized = false;

    let up_low_sub = per_up_low - 1;
    let mid_sub = per_mid - 1;

    for i in 0..len {
        if i < first_valid_idx {
            continue;
        }

        let price = (high[i] + low[i]) * 0.5;

        if !initialized {
            bp_prev1 = price;
            bp_prev2 = price;
            peak_prev = price;
            valley_prev = price;
            initialized = true;
        }

        let bp_curr = if i >= first_valid_idx + 2 {
            let price_i2 = (high[i - 2] + low[i - 2]) * 0.5;
            half_one_minus_alpha * (price - price_i2) + beta * (1.0 + alpha) * bp_prev1
                - alpha * bp_prev2
        } else {
            price
        };

        let mut peak_curr = peak_prev;
        let mut valley_curr = valley_prev;
        if i >= first_valid_idx + 2 {
            if bp_prev1 > bp_curr && bp_prev1 > bp_prev2 {
                peak_curr = bp_prev1;
            }
            if bp_prev1 < bp_curr && bp_prev1 < bp_prev2 {
                valley_curr = bp_prev1;
            }
        }

        let sp = peak_curr * fraction;
        let sv = valley_curr * fraction;

        sum_up += sp;
        sum_low += sv;
        sum_mb += bp_curr;

        let old_sp = sp_ring[idx_up_low];
        let old_sv = sv_ring[idx_up_low];
        let old_bp = bp_ring[idx_mid];

        sp_ring[idx_up_low] = sp;
        sv_ring[idx_up_low] = sv;
        bp_ring[idx_mid] = bp_curr;

        if i >= first_valid_idx + per_up_low {
            sum_up -= old_sp;
            sum_low -= old_sv;
        }
        if i >= first_valid_idx + per_mid {
            sum_mb -= old_bp;
        }

        idx_up_low += 1;
        if idx_up_low == per_up_low {
            idx_up_low = 0;
        }
        idx_mid += 1;
        if idx_mid == per_mid {
            idx_mid = 0;
        }

        if i >= first_valid_idx + up_low_sub {
            upperband[i] = sum_up / per_up_low as f64;
            lowerband[i] = sum_low / per_up_low as f64;
        }
        if i >= first_valid_idx + mid_sub {
            middleband[i] = sum_mb / per_mid as f64;
        }

        bp_prev2 = bp_prev1;
        bp_prev1 = bp_curr;
        peak_prev = peak_curr;
        valley_prev = valley_curr;
    }

    Ok(EmdOutput {
        upperband,
        middleband,
        lowerband,
    })
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emd_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = EmdParams {
            period: Some(20),
            delta: Some(0.5),
            fraction: Some(0.1),
        };
        let input = EmdInput::from_candles(&candles, params);
        let emd_result = emd(&input).expect("Failed to calculate EMD");

        assert_eq!(
            emd_result.upperband.len(),
            candles.close.len(),
            "EMD upperband length mismatch"
        );
        assert_eq!(
            emd_result.middleband.len(),
            candles.close.len(),
            "EMD middleband length mismatch"
        );
        assert_eq!(
            emd_result.lowerband.len(),
            candles.close.len(),
            "EMD lowerband length mismatch"
        );

        let expected_last_five_upper = [
            50.33760237677157,
            50.28850695686447,
            50.23941153695737,
            50.19031611705027,
            48.709744457737344,
        ];
        let expected_last_five_middle = [
            -368.71064280396706,
            -399.11033986231377,
            -421.9368852621732,
            -437.879217150269,
            -447.3257167904511,
        ];
        let expected_last_five_lower = [
            -60.67834136221248,
            -60.93110347122829,
            -61.68154077026321,
            -62.43197806929814,
            -63.18241536833306,
        ];

        let len = candles.close.len();
        assert!(
            len >= 5,
            "Not enough data to test last 5 EMD values; have {} points",
            len
        );

        let start_idx = len - 5;
        let actual_ub = &emd_result.upperband[start_idx..];
        let actual_mb = &emd_result.middleband[start_idx..];
        let actual_lb = &emd_result.lowerband[start_idx..];

        for i in 0..5 {
            assert!(
                (actual_ub[i] - expected_last_five_upper[i]).abs() < 1e-6,
                "Upperband mismatch at index {}: expected {}, got {}",
                i,
                expected_last_five_upper[i],
                actual_ub[i]
            );
            assert!(
                (actual_mb[i] - expected_last_five_middle[i]).abs() < 1e-6,
                "Middleband mismatch at index {}: expected {}, got {}",
                i,
                expected_last_five_middle[i],
                actual_mb[i]
            );
            assert!(
                (actual_lb[i] - expected_last_five_lower[i]).abs() < 1e-6,
                "Lowerband mismatch at index {}: expected {}, got {}",
                i,
                expected_last_five_lower[i],
                actual_lb[i]
            );
        }
    }

    #[test]
    fn test_emd_empty_data() {
        let empty_data: [f64; 0] = [];
        let params = EmdParams::default();
        let input =
            EmdInput::from_slices(&empty_data, &empty_data, &empty_data, &empty_data, params);
        let result = emd(&input);
        assert!(result.is_err(), "Expected error on empty data");
    }

    #[test]
    fn test_emd_all_nans() {
        let data = [f64::NAN, f64::NAN, f64::NAN];
        let params = EmdParams::default();
        let input = EmdInput::from_slices(&data, &data, &data, &data, params);
        let result = emd(&input);
        assert!(result.is_err(), "Expected error for all-NaN data");
    }

    #[test]
    fn test_emd_invalid_period() {
        let data = [1.0, 2.0, 3.0];
        let params = EmdParams {
            period: Some(0),
            ..Default::default()
        };
        let input = EmdInput::from_slices(&data, &data, &data, &data, params);
        let result = emd(&input);
        assert!(result.is_err(), "Expected error for zero period");
    }

    #[test]
    fn test_emd_not_enough_valid_data() {
        let data = vec![10.0; 10];
        let params = EmdParams {
            period: Some(20),
            ..Default::default()
        };
        let input = EmdInput::from_slices(&data, &data, &data, &data, params);
        let result = emd(&input);
        assert!(result.is_err(), "Expected error for not enough valid data");
    }

    #[test]
    fn test_emd_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = EmdInput::with_default_candles(&candles);
        let result = emd(&input);
        assert!(
            result.is_ok(),
            "Expected EMD to succeed with default params"
        );
    }
}
