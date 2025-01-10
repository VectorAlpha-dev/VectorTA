/// # Fractal Adaptive Moving Average (FRAMA)
///
/// A moving average that adapts its smoothing factor based on a fractal dimension calculation.
/// This implementation follows the Python reference from the provided `frama` function,
/// using `high`, `low`, and `close` data to calculate the fractal dimension over a specified
/// window, then adapting the smoothing coefficient accordingly.
///
/// ## Parameters
/// - **window**: The window size for the FRAMA calculation (must be even). Defaults to 10.
/// - **sc**: Slow constant. Defaults to 300.
/// - **fc**: Fast constant. Defaults to 1.
///
/// ## Errors
/// - **EmptyData**: frama: Input data slice is empty.
/// - **InvalidWindow**: frama: `window` is zero or exceeds the data length.
/// - **NotEnoughValidData**: frama: Fewer than `window` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: frama: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(FramaOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the calculation window is filled.
/// - **`Err(FramaError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum FramaData<'a> {
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
pub struct FramaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct FramaParams {
    pub window: Option<usize>,
    pub sc: Option<usize>,
    pub fc: Option<usize>,
}

impl Default for FramaParams {
    fn default() -> Self {
        Self {
            window: Some(10),
            sc: Some(300),
            fc: Some(1),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FramaInput<'a> {
    pub data: FramaData<'a>,
    pub params: FramaParams,
}

impl<'a> FramaInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: FramaParams) -> Self {
        Self {
            data: FramaData::Candles { candles },
            params,
        }
    }

    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: FramaParams,
    ) -> Self {
        Self {
            data: FramaData::Slices { high, low, close },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: FramaData::Candles { candles },
            params: FramaParams::default(),
        }
    }

    pub fn get_window(&self) -> usize {
        self.params
            .window
            .unwrap_or_else(|| FramaParams::default().window.unwrap())
    }

    pub fn get_sc(&self) -> usize {
        self.params
            .sc
            .unwrap_or_else(|| FramaParams::default().sc.unwrap())
    }

    pub fn get_fc(&self) -> usize {
        self.params
            .fc
            .unwrap_or_else(|| FramaParams::default().fc.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum FramaError {
    #[error("frama: Empty data provided for FRAMA.")]
    EmptyData,
    #[error("frama: Invalid window: window = {window}, data length = {data_len}")]
    InvalidWindow { window: usize, data_len: usize },
    #[error("frama: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("frama: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn frama(input: &FramaInput) -> Result<FramaOutput, FramaError> {
    let (high, low, close) = match &input.data {
        FramaData::Candles { candles } => {
            let h = candles.select_candle_field("high").unwrap();
            let l = candles.select_candle_field("low").unwrap();
            let c = candles.select_candle_field("close").unwrap();
            (h, l, c)
        }
        FramaData::Slices { high, low, close } => (*high, *low, *close),
    };

    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(FramaError::EmptyData);
    }

    let window = input.get_window();
    let sc = input.get_sc();
    let fc = input.get_fc();
    let data_len = high.len();

    if window == 0 || window > data_len {
        return Err(FramaError::InvalidWindow { window, data_len });
    }

    let first_valid_idx =
        (0..data_len).find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan());
    if first_valid_idx.is_none() {
        return Err(FramaError::AllValuesNaN);
    }
    let first_valid_idx = first_valid_idx.unwrap();
    if (data_len - first_valid_idx) < window {
        return Err(FramaError::NotEnoughValidData {
            needed: window,
            valid: data_len - first_valid_idx,
        });
    }

    let mut frama_values = vec![f64::NAN; data_len];
    let mut d_values = vec![f64::NAN; data_len];
    let mut alpha_values = vec![f64::NAN; data_len];

    let mut n = window;
    if n % 2 == 1 {
        n += 1;
    }

    let w = (2.0 / (sc as f64 + 1.0)).ln();
    let mut sum_init = 0.0;
    for i in first_valid_idx..(first_valid_idx + n) {
        sum_init += close[i];
    }
    frama_values[first_valid_idx + n - 1] = sum_init / n as f64;

    for i in (first_valid_idx + n)..data_len {
        if !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan() {
            let seg_start = i - n;
            let half = n / 2;
            let mid = seg_start + half;

            let (mut max1, mut min1) = (f64::MIN, f64::MAX);
            let (mut max2, mut min2) = (f64::MIN, f64::MAX);
            let (mut max3, mut min3) = (f64::MIN, f64::MAX);

            for j in seg_start..i {
                if j < mid {
                    if high[j] > max2 {
                        max2 = high[j];
                    }
                    if low[j] < min2 {
                        min2 = low[j];
                    }
                } else {
                    if high[j] > max1 {
                        max1 = high[j];
                    }
                    if low[j] < min1 {
                        min1 = low[j];
                    }
                }
                if high[j] > max3 {
                    max3 = high[j];
                }
                if low[j] < min3 {
                    min3 = low[j];
                }
            }

            let n1 = (max1 - min1) / half as f64;
            let n2 = (max2 - min2) / half as f64;
            let n3 = (max3 - min3) / n as f64;

            if n1 > 0.0 && n2 > 0.0 && n3 > 0.0 {
                d_values[i] = ((n1 + n2).ln() - n3.ln()) / 2_f64.ln();
            } else {
                d_values[i] = d_values[i - 1];
            }

            let mut old_alpha = (w * (d_values[i] - 1.0)).exp();
            if old_alpha < 0.1 {
                old_alpha = 0.1;
            }
            if old_alpha > 1.0 {
                old_alpha = 1.0;
            }

            let old_n = (2.0 - old_alpha) / old_alpha;
            let new_n = ((sc as f64 - fc as f64) * ((old_n - 1.0) / (sc as f64 - 1.0))) + fc as f64;
            let mut alpha_ = 2.0 / (new_n + 1.0);
            let sc_floor = 2.0 / (sc as f64 + 1.0);
            if alpha_ < sc_floor {
                alpha_ = sc_floor;
            }
            if alpha_ > 1.0 {
                alpha_ = 1.0;
            }
            alpha_values[i] = alpha_;

            frama_values[i] = alpha_ * close[i] + (1.0 - alpha_) * frama_values[i - 1];
        } else {
            d_values[i] = d_values[i - 1];
            alpha_values[i] = alpha_values[i - 1];
            frama_values[i] = frama_values[i - 1];
        }
    }

    Ok(FramaOutput {
        values: frama_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_frama_defaults_from_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = FramaInput::with_default_candles(&candles);
        let result = frama(&input).expect("FRAMA calculation failed with defaults");
        assert_eq!(result.values.len(), candles.close.len());
    }

    #[test]
    fn test_frama_params() {
        let params = FramaParams {
            window: Some(10),
            sc: Some(300),
            fc: Some(1),
        };
        assert_eq!(params.window, Some(10));
        assert_eq!(params.sc, Some(300));
        assert_eq!(params.fc, Some(1));

        let default_params = FramaParams::default();
        assert_eq!(default_params.window, Some(10));
        assert_eq!(default_params.sc, Some(300));
        assert_eq!(default_params.fc, Some(1));
    }

    #[test]
    fn test_frama_output_length() {
        let input_data = [10.0, 20.0, 30.0, 40.0, 50.0];
        let params = FramaParams {
            window: Some(4),
            sc: Some(300),
            fc: Some(1),
        };
        let input = FramaInput::from_slices(&input_data, &input_data, &input_data, params);
        let result = frama(&input).expect("FRAMA calculation failed");
        assert_eq!(result.values.len(), input_data.len());
    }

    #[test]
    fn test_frama_with_zero_window() {
        let input_data = [10.0, 20.0, 30.0];
        let params = FramaParams {
            window: Some(0),
            sc: Some(300),
            fc: Some(1),
        };
        let input = FramaInput::from_slices(&input_data, &input_data, &input_data, params);
        let result = frama(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Invalid window"));
        }
    }

    #[test]
    fn test_frama_with_window_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = FramaParams {
            window: Some(10),
            sc: Some(300),
            fc: Some(1),
        };
        let input = FramaInput::from_slices(&input_data, &input_data, &input_data, params);
        let result = frama(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Invalid window"));
        }
    }

    #[test]
    fn test_frama_not_enough_valid_data() {
        let input_data = [f64::NAN, 20.0, 30.0];
        let params = FramaParams {
            window: Some(3),
            sc: Some(300),
            fc: Some(1),
        };
        let input = FramaInput::from_slices(&input_data, &input_data, &input_data, params);
        let result = frama(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Not enough valid data"));
        }
    }

    #[test]
    fn test_frama_all_nan() {
        let nan_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = FramaParams::default();
        let input = FramaInput::from_slices(&nan_data, &nan_data, &nan_data, params);
        let result = frama(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_frama_expected_values() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = FramaParams {
            window: Some(10),
            sc: Some(300),
            fc: Some(1),
        };
        let input = FramaInput::from_candles(&candles, params);
        let result = frama(&input).expect("Failed to calculate FRAMA");
        let expected = [
            59337.23056930512,
            59321.607512374605,
            59286.677929994796,
            59268.00202402624,
            59160.03888720062,
        ];
        assert!(result.values.len() >= 5);
        let start_index = result.values.len() - 5;
        let last_five = &result.values[start_index..];
        for (i, &val) in last_five.iter().enumerate() {
            let exp: f64 = expected[i];
            if !val.is_nan() && !exp.is_nan() {
                assert!(
                    (val - exp).abs() < 1e-1,
                    "FRAMA mismatch at index {}: expected {}, got {}",
                    i,
                    exp,
                    val
                );
            }
        }
    }
}
