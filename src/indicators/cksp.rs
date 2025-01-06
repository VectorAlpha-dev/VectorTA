/// # Chande Kroll Stop (CKSP)
///
/// Computes two stop lines (long and short) using ATR and rolling maxima/minima:
/// - `long_stop[i] = rolling_max(high, q)[i] - x * atr[i]`, then another rolling max over that result
/// - `short_stop[i] = rolling_min(low, q)[i] + x * atr[i]`, then another rolling min over that result
///
/// ## Parameters
/// - **p**: Period for ATR. Defaults to 10.
/// - **x**: Multiplier for ATR. Defaults to 1.0.
/// - **q**: Window size for rolling max/min. Defaults to 9.
///
/// ## Errors
/// - **NoData**: cksp: Data is empty.
/// - **NotEnoughData**: cksp: Not enough data for the provided parameters.
/// - **InconsistentLengths**: cksp: high, low, close slices have different lengths.
///
/// ## Returns
/// `Ok(CkspOutput)` on success, containing:
/// - `long_values`: Vector of the long stop line.
/// - `short_values`: Vector of the short stop line.
///
/// `Err(CkspError)` otherwise.
use crate::utilities::data_loader::Candles;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum CkspData<'a> {
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
pub struct CkspOutput {
    pub long_values: Vec<f64>,
    pub short_values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct CkspParams {
    pub p: Option<usize>,
    pub x: Option<f64>,
    pub q: Option<usize>,
}

impl Default for CkspParams {
    fn default() -> Self {
        Self {
            p: Some(10),
            x: Some(1.0),
            q: Some(9),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CkspInput<'a> {
    pub data: CkspData<'a>,
    pub params: CkspParams,
}

impl<'a> CkspInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: CkspParams) -> Self {
        Self {
            data: CkspData::Candles { candles },
            params,
        }
    }

    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: CkspParams,
    ) -> Self {
        Self {
            data: CkspData::Slices { high, low, close },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: CkspData::Candles { candles },
            params: CkspParams::default(),
        }
    }

    pub fn get_p(&self) -> usize {
        self.params.p.unwrap_or(10)
    }

    pub fn get_x(&self) -> f64 {
        self.params.x.unwrap_or(1.0)
    }

    pub fn get_q(&self) -> usize {
        self.params.q.unwrap_or(9)
    }
}

#[derive(Debug, Error)]
pub enum CkspError {
    #[error("cksp: Data is empty.")]
    NoData,
    #[error("cksp: Not enough data for p={p}, q={q}, data_len={data_len}.")]
    NotEnoughData { p: usize, q: usize, data_len: usize },
    #[error("cksp: Inconsistent lengths.")]
    InconsistentLengths,
    #[error("cksp: Candle field error: {0}")]
    CandleFieldError(String),
}

#[inline]
pub fn cksp(input: &CkspInput) -> Result<CkspOutput, CkspError> {
    let (high, low, close) = match &input.data {
        CkspData::Candles { candles } => {
            let high = candles
                .select_candle_field("high")
                .map_err(|e| CkspError::CandleFieldError(e.to_string()))?;
            let low = candles
                .select_candle_field("low")
                .map_err(|e| CkspError::CandleFieldError(e.to_string()))?;
            let close = candles
                .select_candle_field("close")
                .map_err(|e| CkspError::CandleFieldError(e.to_string()))?;
            (high, low, close)
        }
        CkspData::Slices { high, low, close } => {
            if high.len() != low.len() || low.len() != close.len() {
                return Err(CkspError::InconsistentLengths);
            }
            (*high, *low, *close)
        }
    };

    let p = input.get_p();
    let x = input.get_x();
    let q = input.get_q();

    let len = close.len();
    if len == 0 {
        return Err(CkspError::NoData);
    }

    if p > len || q > len {
        return Err(CkspError::NotEnoughData {
            p,
            q,
            data_len: len,
        });
    }

    let mut atr_values = vec![f64::NAN; len];
    let alpha = 1.0 / (p as f64);
    let mut sum_tr = 0.0;
    let mut rma = f64::NAN;

    for i in 0..len {
        let tr = if i == 0 {
            high[0] - low[0]
        } else {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            hl.max(hc).max(lc)
        };
        if i < p {
            sum_tr += tr;
            if i == p - 1 {
                rma = sum_tr / p as f64;
                atr_values[i] = rma;
            }
        } else {
            rma += alpha * (tr - rma);
            atr_values[i] = rma;
        }
    }

    fn rolling_max(src: &[f64], window: usize) -> Vec<f64> {
        let mut output = vec![f64::NAN; src.len()];
        let mut deque = std::collections::VecDeque::<(usize, f64)>::new();

        for i in 0..src.len() {
            let val = src[i];
            while let Some((_, v)) = deque.back() {
                if *v <= val {
                    deque.pop_back();
                } else {
                    break;
                }
            }
            deque.push_back((i, val));
            let start = i.saturating_sub(window - 1);
            while let Some(&(idx, _)) = deque.front() {
                if idx < start {
                    deque.pop_front();
                } else {
                    break;
                }
            }
            if i >= window - 1 {
                if let Some(&(_, v)) = deque.front() {
                    output[i] = v;
                }
            }
        }
        output
    }

    fn rolling_min(src: &[f64], window: usize) -> Vec<f64> {
        let mut output = vec![f64::NAN; src.len()];
        let mut deque = std::collections::VecDeque::<(usize, f64)>::new();

        for i in 0..src.len() {
            let val = src[i];
            while let Some((_, v)) = deque.back() {
                if *v >= val {
                    deque.pop_back();
                } else {
                    break;
                }
            }
            deque.push_back((i, val));
            let start = i.saturating_sub(window - 1);
            while let Some(&(idx, _)) = deque.front() {
                if idx < start {
                    deque.pop_front();
                } else {
                    break;
                }
            }
            if i >= window - 1 {
                if let Some(&(_, v)) = deque.front() {
                    output[i] = v;
                }
            }
        }
        output
    }

    let max_high_q = rolling_max(high, q);
    let min_low_q = rolling_min(low, q);

    let mut ls0 = vec![f64::NAN; len];
    for i in 0..len {
        if !max_high_q[i].is_nan() && !atr_values[i].is_nan() {
            ls0[i] = max_high_q[i] - x * atr_values[i];
        }
    }
    let ls0_rolling = rolling_max(&ls0, q);

    let mut ss0 = vec![f64::NAN; len];
    for i in 0..len {
        if !min_low_q[i].is_nan() && !atr_values[i].is_nan() {
            ss0[i] = min_low_q[i] + x * atr_values[i];
        }
    }
    let ss0_rolling = rolling_min(&ss0, q);

    Ok(CkspOutput {
        long_values: ls0_rolling,
        short_values: ss0_rolling,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_cksp_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = CkspParams {
            p: None,
            x: None,
            q: None,
        };
        let input_default = CkspInput::from_candles(&candles, default_params);
        let output_default = cksp(&input_default).expect("Failed CKSP with default params");
        assert_eq!(output_default.long_values.len(), candles.close.len());
        assert_eq!(output_default.short_values.len(), candles.close.len());

        let params_custom = CkspParams {
            p: Some(5),
            x: Some(2.0),
            q: Some(4),
        };
        let input_custom = CkspInput::from_candles(&candles, params_custom);
        let output_custom = cksp(&input_custom).expect("Failed CKSP custom params");
        assert_eq!(output_custom.long_values.len(), candles.close.len());
        assert_eq!(output_custom.short_values.len(), candles.close.len());
    }

    #[test]
    fn test_cksp_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = &candles.close;

        let params = CkspParams {
            p: Some(10),
            x: Some(1.0),
            q: Some(9),
        };
        let input = CkspInput::from_candles(&candles, params);
        let cksp_result = cksp(&input).expect("Failed to calculate CKSP");

        assert_eq!(cksp_result.long_values.len(), close_prices.len());
        assert_eq!(cksp_result.short_values.len(), close_prices.len());

        let expected_long_last_5 = [
            60306.66197802568,
            60306.66197802568,
            60306.66197802568,
            60203.29578022311,
            60201.57958198072,
        ];
        let l_start = cksp_result.long_values.len() - 5;
        let long_slice = &cksp_result.long_values[l_start..];
        for (i, &val) in long_slice.iter().enumerate() {
            let exp_val = expected_long_last_5[i];
            assert!(
                (val - exp_val).abs() < 1e-5,
                "CKSP long mismatch at index {}: expected {}, got {}",
                i,
                exp_val,
                val
            );
        }

        let expected_short_last_5 = [
            58757.826484736055,
            58701.74383626245,
            58656.36945263621,
            58611.03250737258,
            58611.03250737258,
        ];
        let s_start = cksp_result.short_values.len() - 5;
        let short_slice = &cksp_result.short_values[s_start..];
        for (i, &val) in short_slice.iter().enumerate() {
            let exp_val = expected_short_last_5[i];
            assert!(
                (val - exp_val).abs() < 1e-5,
                "CKSP short mismatch at index {}: expected {}, got {}",
                i,
                exp_val,
                val
            );
        }
    }

    #[test]
    fn test_cksp_params_with_default_params() {
        let default_params = CkspParams::default();
        assert_eq!(default_params.p, Some(10));
        assert_eq!(default_params.x, Some(1.0));
        assert_eq!(default_params.q, Some(9));
    }

    #[test]
    fn test_cksp_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = CkspInput::with_default_candles(&candles);
        match input.data {
            CkspData::Candles { .. } => {}
            _ => panic!("Expected CkspData::Candles variant"),
        }
    }

    #[test]
    fn test_cksp_with_zero_p() {
        let data_h = [10.0, 11.0, 12.0];
        let data_l = [9.0, 10.0, 10.5];
        let data_c = [9.5, 10.5, 11.0];
        let params = CkspParams {
            p: Some(0),
            x: Some(1.0),
            q: Some(9),
        };
        let input = CkspInput::from_slices(&data_h, &data_l, &data_c, params);
        let result = cksp(&input);
        assert!(result.is_err(), "Expected NotEnoughData for p=0");
    }

    #[test]
    fn test_cksp_with_inconsistent_lengths() {
        let data_h = [10.0, 11.0, 12.0];
        let data_l = [8.0, 9.0];
        let data_c = [9.0, 10.0, 11.0];
        let params = CkspParams {
            p: Some(10),
            x: Some(1.0),
            q: Some(9),
        };
        let input = CkspInput::from_slices(&data_h, &data_l, &data_c, params);
        let result = cksp(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_cksp_very_small_data_set() {
        let data_h = [42.0];
        let data_l = [41.0];
        let data_c = [41.5];
        let params = CkspParams {
            p: Some(10),
            x: Some(1.0),
            q: Some(9),
        };
        let input = CkspInput::from_slices(&data_h, &data_l, &data_c, params);
        let result = cksp(&input);
        assert!(
            result.is_err(),
            "Expected NotEnoughData for 1 candle < p or q"
        );
    }

    #[test]
    fn test_cksp_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = CkspParams {
            p: Some(10),
            x: Some(1.0),
            q: Some(9),
        };
        let first_input = CkspInput::from_candles(&candles, first_params);
        let first_result = cksp(&first_input).expect("Failed first CKSP");
        assert_eq!(first_result.long_values.len(), candles.close.len());
        assert_eq!(first_result.short_values.len(), candles.close.len());

        let second_params = CkspParams {
            p: Some(10),
            x: Some(1.0),
            q: Some(9),
        };
        let dummy_close = vec![0.0; first_result.long_values.len()];
        let second_input = CkspInput::from_slices(
            &first_result.long_values,
            &first_result.short_values,
            &dummy_close,
            second_params,
        );
        let second_result = cksp(&second_input).expect("Failed second CKSP re-input");
        assert_eq!(second_result.long_values.len(), dummy_close.len());
        assert_eq!(second_result.short_values.len(), dummy_close.len());
    }

    #[test]
    fn test_cksp_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = CkspParams {
            p: Some(10),
            x: Some(1.0),
            q: Some(9),
        };
        let input = CkspInput::from_candles(&candles, params);
        let cksp_result = cksp(&input).expect("Failed to calculate CKSP");

        let len = candles.close.len();
        assert_eq!(cksp_result.long_values.len(), len);
        assert_eq!(cksp_result.short_values.len(), len);

        if len > 240 {
            for i in 240..len {
                assert!(
                    !cksp_result.long_values[i].is_nan(),
                    "Expected no NaN in long_values after index 240, found NaN at {}",
                    i
                );
                assert!(
                    !cksp_result.short_values[i].is_nan(),
                    "Expected no NaN in short_values after index 240, found NaN at {}",
                    i
                );
            }
        }
    }
}
