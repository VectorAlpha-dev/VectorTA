/// # Damiani Volatmeter
///
/// A volatility indicator that uses two ATR calculations (`vis_atr` and `sed_atr`) and two standard
/// deviation windows (`vis_std` and `sed_std`) to determine market volatility (`vol`) and an
/// "anti-trend" threshold offset (`anti`). If `vol` is above `anti`, the market is considered
/// trending.
///
/// ## Parameters
/// - **vis_atr**: ATR period for the volatility line. Defaults to 13.
/// - **vis_std**: Standard deviation window for the volatility line. Defaults to 20.
/// - **sed_atr**: ATR period for the "sedation" line. Defaults to 40.
/// - **sed_std**: Standard deviation window for the "sedation" line. Defaults to 100.
/// - **threshold**: Constant used to shift the "anti" line. Defaults to 1.4.
///
/// ## Errors
/// - **EmptyData**: No data available.
/// - **InvalidPeriod**: One or more period parameters are zero or exceed the data length.
/// - **NotEnoughValidData**: Not enough valid (non-`NaN`) data points remain after the first valid index.
/// - **AllValuesNaN**: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(DamianiVolatmeterOutput)`** on success, containing two `Vec<f64>` (`vol`, `anti`) matching
///   the input length, each with leading `NaN`s until the required lookback windows are satisfied.
/// - **`Err(DamianiVolatmeterError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum DamianiVolatmeterData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct DamianiVolatmeterOutput {
    pub vol: Vec<f64>,
    pub anti: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct DamianiVolatmeterParams {
    pub vis_atr: Option<usize>,
    pub vis_std: Option<usize>,
    pub sed_atr: Option<usize>,
    pub sed_std: Option<usize>,
    pub threshold: Option<f64>,
}

impl Default for DamianiVolatmeterParams {
    fn default() -> Self {
        Self {
            vis_atr: Some(13),
            vis_std: Some(20),
            sed_atr: Some(40),
            sed_std: Some(100),
            threshold: Some(1.4),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DamianiVolatmeterInput<'a> {
    pub data: DamianiVolatmeterData<'a>,
    pub params: DamianiVolatmeterParams,
}

impl<'a> DamianiVolatmeterInput<'a> {
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: DamianiVolatmeterParams,
    ) -> Self {
        Self {
            data: DamianiVolatmeterData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: DamianiVolatmeterParams) -> Self {
        Self {
            data: DamianiVolatmeterData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: DamianiVolatmeterData::Candles {
                candles,
                source: "close",
            },
            params: DamianiVolatmeterParams::default(),
        }
    }

    fn get_vis_atr(&self) -> usize {
        self.params
            .vis_atr
            .unwrap_or_else(|| DamianiVolatmeterParams::default().vis_atr.unwrap())
    }

    fn get_vis_std(&self) -> usize {
        self.params
            .vis_std
            .unwrap_or_else(|| DamianiVolatmeterParams::default().vis_std.unwrap())
    }

    fn get_sed_atr(&self) -> usize {
        self.params
            .sed_atr
            .unwrap_or_else(|| DamianiVolatmeterParams::default().sed_atr.unwrap())
    }

    fn get_sed_std(&self) -> usize {
        self.params
            .sed_std
            .unwrap_or_else(|| DamianiVolatmeterParams::default().sed_std.unwrap())
    }

    fn get_threshold(&self) -> f64 {
        self.params
            .threshold
            .unwrap_or_else(|| DamianiVolatmeterParams::default().threshold.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum DamianiVolatmeterError {
    #[error("damiani_volatmeter: Empty data provided.")]
    EmptyData,
    #[error("damiani_volatmeter: Invalid period: data length = {data_len}, vis_atr = {vis_atr}, vis_std = {vis_std}, sed_atr = {sed_atr}, sed_std = {sed_std}")]
    InvalidPeriod {
        data_len: usize,
        vis_atr: usize,
        vis_std: usize,
        sed_atr: usize,
        sed_std: usize,
    },
    #[error("damiani_volatmeter: Not enough valid data after first non-NaN index. needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("damiani_volatmeter: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn damiani_volatmeter(
    input: &DamianiVolatmeterInput,
) -> Result<DamianiVolatmeterOutput, DamianiVolatmeterError> {
    let (source, high, low, close) = match &input.data {
        DamianiVolatmeterData::Candles { candles, source } => {
            let s = source_type(candles, source);
            if s.is_empty() || candles.close.is_empty() {
                return Err(DamianiVolatmeterError::EmptyData);
            }
            (
                s,
                source_type(candles, "high"),
                source_type(candles, "low"),
                source_type(candles, "close"),
            )
        }
        DamianiVolatmeterData::Slice(slice) => {
            if slice.is_empty() {
                return Err(DamianiVolatmeterError::EmptyData);
            }
            return Err(DamianiVolatmeterError::EmptyData);
        }
    };

    let length = source.len();
    if length == 0 {
        return Err(DamianiVolatmeterError::EmptyData);
    }

    let vis_atr = input.get_vis_atr();
    let vis_std = input.get_vis_std();
    let sed_atr = input.get_sed_atr();
    let sed_std = input.get_sed_std();
    let threshold = input.get_threshold();

    if vis_atr == 0
        || vis_std == 0
        || sed_atr == 0
        || sed_std == 0
        || vis_atr > length
        || vis_std > length
        || sed_atr > length
        || sed_std > length
    {
        return Err(DamianiVolatmeterError::InvalidPeriod {
            data_len: length,
            vis_atr,
            vis_std,
            sed_atr,
            sed_std,
        });
    }

    let first_valid_idx = match source.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(DamianiVolatmeterError::AllValuesNaN),
    };

    let needed = *[vis_atr, vis_std, sed_atr, sed_std, 3]
        .iter()
        .max()
        .unwrap_or(&0);

    if (length - first_valid_idx) < needed {
        return Err(DamianiVolatmeterError::NotEnoughValidData {
            needed,
            valid: length - first_valid_idx,
        });
    }

    let mut vol = vec![f64::NAN; length];
    let mut anti = vec![f64::NAN; length];

    let mut atr_vis_val = f64::NAN;
    let mut atr_sed_val = f64::NAN;
    let mut sum_vis = 0.0;
    let mut sum_sed = 0.0;
    let mut prev_close = close[0];

    let mut ring_vis = vec![0.0; vis_std];
    let mut ring_sed = vec![0.0; sed_std];
    let mut sum_vis_std = 0.0;
    let mut sum_sq_vis_std = 0.0;
    let mut sum_sed_std = 0.0;
    let mut sum_sq_sed_std = 0.0;
    let mut idx_vis = 0;
    let mut idx_sed = 0;
    let mut filled_vis = 0;
    let mut filled_sed = 0;

    #[inline]
    fn ring_update(
        val: f64,
        ring: &mut [f64],
        idx: &mut usize,
        filled: &mut usize,
        sum_x: &mut f64,
        sum_x2: &mut f64,
    ) {
        let old_val = ring[*idx];
        ring[*idx] = val;
        *idx = (*idx + 1) % ring.len();

        if *filled < ring.len() {
            *filled += 1;
            *sum_x += val;
            *sum_x2 += val * val;
        } else {
            *sum_x = *sum_x - old_val + val;
            *sum_x2 = *sum_x2 - old_val * old_val + val * val;
        }
    }

    #[inline]
    fn compute_pop_std(sum: f64, sum_sq: f64, count: usize) -> f64 {
        if count == 0 {
            return 0.0;
        }
        let mean = sum / count as f64;
        let mean_sq = sum_sq / count as f64;
        let var = mean_sq - mean * mean;
        if var <= 0.0 {
            0.0
        } else {
            var.sqrt()
        }
    }

    let lag_s = 0.5_f64;

    for i in 0..length {
        if i > 0 {
            let mut tr = high[i] - low[i];
            let mut tr2 = (high[i] - prev_close).abs();
            let mut tr3 = (low[i] - prev_close).abs();
            if tr.is_nan() {
                tr = 0.0;
            }
            if tr2.is_nan() {
                tr2 = 0.0;
            }
            if tr3.is_nan() {
                tr3 = 0.0;
            }
            let true_range = tr.max(tr2).max(tr3);
            prev_close = close[i];

            if i < vis_atr {
                sum_vis += true_range;
                if i == vis_atr - 1 {
                    atr_vis_val = sum_vis / vis_atr as f64;
                }
            } else if atr_vis_val.is_finite() {
                atr_vis_val = ((vis_atr - 1) as f64 * atr_vis_val + true_range) / vis_atr as f64;
            }

            if i < sed_atr {
                sum_sed += true_range;
                if i == sed_atr - 1 {
                    atr_sed_val = sum_sed / sed_atr as f64;
                }
            } else if atr_sed_val.is_finite() {
                atr_sed_val = ((sed_atr - 1) as f64 * atr_sed_val + true_range) / sed_atr as f64;
            }
        }

        let new_val = if source[i].is_nan() { 0.0 } else { source[i] };
        ring_update(
            new_val,
            &mut ring_vis,
            &mut idx_vis,
            &mut filled_vis,
            &mut sum_vis_std,
            &mut sum_sq_vis_std,
        );
        ring_update(
            new_val,
            &mut ring_sed,
            &mut idx_sed,
            &mut filled_sed,
            &mut sum_sed_std,
            &mut sum_sq_sed_std,
        );

        if i >= needed {
            let p1 = if i >= 1 && !vol[i - 1].is_nan() {
                vol[i - 1]
            } else {
                0.0
            };
            let p3 = if i >= 3 && !vol[i - 3].is_nan() {
                vol[i - 3]
            } else {
                0.0
            };

            let sed_safe = if atr_sed_val.is_finite() && atr_sed_val != 0.0 {
                atr_sed_val
            } else {
                atr_sed_val + f64::EPSILON
            };
            vol[i] = (atr_vis_val / sed_safe) + lag_s * (p1 - p3);

            if filled_vis == vis_std && filled_sed == sed_std {
                let std_vis_val = compute_pop_std(sum_vis_std, sum_sq_vis_std, vis_std);
                let std_sed_val = compute_pop_std(sum_sed_std, sum_sq_sed_std, sed_std);
                let ratio = if std_sed_val != 0.0 {
                    std_vis_val / std_sed_val
                } else {
                    std_vis_val / (std_sed_val + f64::EPSILON)
                };
                anti[i] = threshold - ratio;
            }
        }
    }

    Ok(DamianiVolatmeterOutput { vol, anti })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_damiani_volatmeter_basic() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = DamianiVolatmeterParams::default();
        let input = DamianiVolatmeterInput::from_candles(&candles, "close", params);
        let output = damiani_volatmeter(&input).expect("Failed to calculate DamianiVolatmeter");

        assert_eq!(output.vol.len(), candles.close.len());
        assert_eq!(output.anti.len(), candles.close.len());
    }

    #[test]
    fn test_damiani_volatmeter_last_five_values() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = DamianiVolatmeterParams::default();
        let input = DamianiVolatmeterInput::from_candles(&candles, "close", params);
        let output = damiani_volatmeter(&input).expect("Failed to calculate DamianiVolatmeter");

        let n = output.vol.len();
        assert!(n >= 5);

        let expected_vol = [
            0.9009485470514558,
            0.8333604467044887,
            0.815318380178986,
            0.8276892636184923,
            0.879447954127426,
        ];
        let expected_anti = [
            1.1227721577887388,
            1.1250333024152703,
            1.1325501989919875,
            1.1403866079746106,
            1.1392919184055932,
        ];
        let start_idx = n - 5;

        for i in 0..5 {
            let vol_val = output.vol[start_idx + i];
            let anti_val = output.anti[start_idx + i];
            let diff_vol = (vol_val - expected_vol[i]).abs();
            let diff_anti = (anti_val - expected_anti[i]).abs();

            assert!(
                diff_vol < 1e-2,
                "vol mismatch at index {}: expected {}, got {}",
                start_idx + i,
                expected_vol[i],
                vol_val
            );
            assert!(
                diff_anti < 1e-2,
                "anti mismatch at index {}: expected {}, got {}",
                start_idx + i,
                expected_anti[i],
                anti_val
            );
        }
    }

    #[test]
    fn test_damiani_volatmeter_with_zero_period() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let mut params = DamianiVolatmeterParams::default();
        params.vis_atr = Some(0);

        let input = DamianiVolatmeterInput::from_candles(&candles, "close", params);
        let result = damiani_volatmeter(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_damiani_volatmeter_params_with_defaults() {
        let default_params = DamianiVolatmeterParams::default();
        assert_eq!(default_params.vis_atr, Some(13));
        assert_eq!(default_params.vis_std, Some(20));
        assert_eq!(default_params.sed_atr, Some(40));
        assert_eq!(default_params.sed_std, Some(100));
        assert_eq!(default_params.threshold, Some(1.4));
    }

    #[test]
    fn test_damiani_volatmeter_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = DamianiVolatmeterInput::with_default_candles(&candles);
        match input.data {
            DamianiVolatmeterData::Candles { source, .. } => {
                assert_eq!(source, "close", "Expected default source to be 'close'");
            }
            _ => panic!("Expected DamianiVolatmeterData::Candles variant"),
        }
    }
}
