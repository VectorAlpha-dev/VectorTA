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
            let h = candles
                .select_candle_field("high")
                .map_err(|e| CkspError::CandleFieldError(e.to_string()))?;
            let l = candles
                .select_candle_field("low")
                .map_err(|e| CkspError::CandleFieldError(e.to_string()))?;
            let c = candles
                .select_candle_field("close")
                .map_err(|e| CkspError::CandleFieldError(e.to_string()))?;
            (h, l, c)
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
    let size = close.len();
    if size == 0 {
        return Err(CkspError::NoData);
    }
    if p > size || q > size {
        return Err(CkspError::NotEnoughData {
            p,
            q,
            data_len: size,
        });
    }
    let first_valid_idx = match close.iter().position(|&v| !v.is_nan()) {
        Some(idx) => idx,
        None => return Err(CkspError::NoData),
    };
    let mut long_values = vec![f64::NAN; size];
    let mut short_values = vec![f64::NAN; size];
    let mut atr = vec![0.0; size];
    let mut sum_tr = 0.0;
    let mut rma = 0.0;
    let alpha = 1.0 / (p as f64);
    let mut dq_h = std::collections::VecDeque::<(usize, f64)>::new();
    let mut dq_ls0 = std::collections::VecDeque::<(usize, f64)>::new();
    let mut dq_l = std::collections::VecDeque::<(usize, f64)>::new();
    let mut dq_ss0 = std::collections::VecDeque::<(usize, f64)>::new();
    for i in 0..size {
        if i < first_valid_idx {
            continue;
        }
        let tr = if i == first_valid_idx {
            high[i] - low[i]
        } else {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            hl.max(hc).max(lc)
        };
        if i - first_valid_idx < p {
            sum_tr += tr;
            if i - first_valid_idx == p - 1 {
                rma = sum_tr / p as f64;
                atr[i] = rma;
            }
        } else {
            rma += alpha * (tr - rma);
            atr[i] = rma;
        }
        while let Some((_, v)) = dq_h.back() {
            if *v <= high[i] {
                dq_h.pop_back();
            } else {
                break;
            }
        }
        dq_h.push_back((i, high[i]));
        let start_h = i.saturating_sub(q - 1);
        while let Some(&(idx, _)) = dq_h.front() {
            if idx < start_h {
                dq_h.pop_front();
            } else {
                break;
            }
        }
        while let Some((_, v)) = dq_l.back() {
            if *v >= low[i] {
                dq_l.pop_back();
            } else {
                break;
            }
        }
        dq_l.push_back((i, low[i]));
        let start_l = i.saturating_sub(q - 1);
        while let Some(&(idx, _)) = dq_l.front() {
            if idx < start_l {
                dq_l.pop_front();
            } else {
                break;
            }
        }
        if atr[i] != 0.0 && i >= first_valid_idx + p - 1 {
            if let (Some(&(_, mh)), Some(&(_, ml))) = (dq_h.front(), dq_l.front()) {
                let ls0_val = mh - x * atr[i];
                let ss0_val = ml + x * atr[i];
                while let Some((_, val)) = dq_ls0.back() {
                    if *val <= ls0_val {
                        dq_ls0.pop_back();
                    } else {
                        break;
                    }
                }
                dq_ls0.push_back((i, ls0_val));
                let start_ls0 = i.saturating_sub(q - 1);
                while let Some(&(idx, _)) = dq_ls0.front() {
                    if idx < start_ls0 {
                        dq_ls0.pop_front();
                    } else {
                        break;
                    }
                }
                if let Some(&(_, mx)) = dq_ls0.front() {
                    long_values[i] = mx;
                }
                while let Some((_, val)) = dq_ss0.back() {
                    if *val >= ss0_val {
                        dq_ss0.pop_back();
                    } else {
                        break;
                    }
                }
                dq_ss0.push_back((i, ss0_val));
                let start_ss0 = i.saturating_sub(q - 1);
                while let Some(&(idx, _)) = dq_ss0.front() {
                    if idx < start_ss0 {
                        dq_ss0.pop_front();
                    } else {
                        break;
                    }
                }
                if let Some(&(_, mn)) = dq_ss0.front() {
                    short_values[i] = mn;
                }
            }
        }
    }
    Ok(CkspOutput {
        long_values,
        short_values,
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
