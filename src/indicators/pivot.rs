/// # Pivot Points (PIVOT)
///
/// A set of support (S) and resistance (R) levels calculated from High, Low, Close, (and Open) prices.
/// Commonly used to identify possible support and resistance levels and guide trade entries or exits.
///
/// ## Parameters
/// - **mode**: Determines the calculation method.
///   - 0 = Standard / Floor
///   - 1 = Fibonacci
///   - 2 = Demark
///   - 3 = Camarilla (default)
///   - 4 = Woodie
///
/// ## Errors
/// - **EmptyData**: pivot: One or more fields (High, Low, Close, possibly Open) is empty.
/// - **AllValuesNaN**: pivot: All input data values are `NaN`.
/// - **NotEnoughValidData**: pivot: No valid (non-`NaN`) data points remain to compute pivot points.
///
/// ## Returns
/// - **`Ok(PivotOutput)`** on success, containing `Vec<f64>` for each pivot level (r4, r3, r2, r1, pp, s1, s2, s3, s4),
///   each matching the length of the input.
/// - **`Err(PivotError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum PivotData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        open: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct PivotParams {
    pub mode: Option<usize>,
}

impl Default for PivotParams {
    fn default() -> Self {
        Self { mode: Some(3) }
    }
}

#[derive(Debug, Clone)]
pub struct PivotInput<'a> {
    pub data: PivotData<'a>,
    pub params: PivotParams,
}

impl<'a> PivotInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: PivotParams) -> Self {
        Self {
            data: PivotData::Candles { candles },
            params,
        }
    }

    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        open: &'a [f64],
        params: PivotParams,
    ) -> Self {
        Self {
            data: PivotData::Slices {
                high,
                low,
                close,
                open,
            },
            params,
        }
    }

    pub fn get_mode(&self) -> usize {
        self.params
            .mode
            .unwrap_or_else(|| PivotParams::default().mode.unwrap())
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: PivotData::Candles { candles },
            params: PivotParams::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PivotOutput {
    pub r4: Vec<f64>,
    pub r3: Vec<f64>,
    pub r2: Vec<f64>,
    pub r1: Vec<f64>,
    pub pp: Vec<f64>,
    pub s1: Vec<f64>,
    pub s2: Vec<f64>,
    pub s3: Vec<f64>,
    pub s4: Vec<f64>,
}

#[derive(Debug, Error)]
pub enum PivotError {
    #[error("pivot: One or more required fields is empty.")]
    EmptyData,
    #[error("pivot: All values are NaN.")]
    AllValuesNaN,
    #[error("pivot: Not enough valid data after the first valid index.")]
    NotEnoughValidData,
}

#[inline]
pub fn pivot(input: &PivotInput) -> Result<PivotOutput, PivotError> {
    let (high, low, close, open) = match &input.data {
        PivotData::Candles { candles } => {
            let high = source_type(candles, "high");
            let low = source_type(candles, "low");
            let close = source_type(candles, "close");
            let open = source_type(candles, "open");
            (high, low, close, open)
        }
        PivotData::Slices {
            high,
            low,
            close,
            open,
        } => (*high, *low, *close, *open),
    };

    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(PivotError::EmptyData);
    }

    let len = high.len();
    if low.len() != len || close.len() != len {
        return Err(PivotError::EmptyData);
    }

    let mut r4 = vec![f64::NAN; len];
    let mut r3 = vec![f64::NAN; len];
    let mut r2 = vec![f64::NAN; len];
    let mut r1 = vec![f64::NAN; len];
    let mut pp = vec![f64::NAN; len];
    let mut s1 = vec![f64::NAN; len];
    let mut s2 = vec![f64::NAN; len];
    let mut s3 = vec![f64::NAN; len];
    let mut s4 = vec![f64::NAN; len];

    let mode = input.get_mode();

    let mut first_valid_idx = None;
    for i in 0..len {
        let h = high[i];
        let l = low[i];
        let c = close[i];
        if !(h.is_nan() || l.is_nan() || c.is_nan()) {
            first_valid_idx = Some(i);
            break;
        }
    }

    let first_valid_idx = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(PivotError::AllValuesNaN),
    };

    if first_valid_idx >= len {
        return Err(PivotError::NotEnoughValidData);
    }

    for i in first_valid_idx..len {
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let o = open[i];
        if h.is_nan() || l.is_nan() || c.is_nan() {
            continue;
        }

        let p = match mode {
            0 | 1 | 2 | 3 | 4 => {
                if mode == 2 {
                    if c < o {
                        (h + 2.0 * l + c) / 4.0
                    } else if c > o {
                        (2.0 * h + l + c) / 4.0
                    } else {
                        (h + l + 2.0 * c) / 4.0
                    }
                } else if mode == 4 {
                    (h + l + (2.0 * o)) / 4.0
                } else {
                    (h + l + c) / 3.0
                }
            }
            _ => (h + l + c) / 3.0,
        };

        pp[i] = p;

        match mode {
            0 => {
                r1[i] = 2.0 * p - l;
                r2[i] = p + (h - l);
                s1[i] = 2.0 * p - h;
                s2[i] = p - (h - l);
            }
            1 => {
                r1[i] = p + 0.382 * (h - l);
                r2[i] = p + 0.618 * (h - l);
                r3[i] = p + 1.0 * (h - l);
                s1[i] = p - 0.382 * (h - l);
                s2[i] = p - 0.618 * (h - l);
                s3[i] = p - 1.0 * (h - l);
            }
            2 => {
                s1[i] = if c < o {
                    (h + 2.0 * l + c) / 2.0 - h
                } else if c > o {
                    (2.0 * h + l + c) / 2.0 - h
                } else {
                    (h + l + 2.0 * c) / 2.0 - h
                };
                r1[i] = if c < o {
                    (h + 2.0 * l + c) / 2.0 - l
                } else if c > o {
                    (2.0 * h + l + c) / 2.0 - l
                } else {
                    (h + l + 2.0 * c) / 2.0 - l
                };
            }
            3 => {
                r4[i] = (0.55 * (h - l)) + c;
                r3[i] = (0.275 * (h - l)) + c;
                r2[i] = (0.183 * (h - l)) + c;
                r1[i] = (0.0916 * (h - l)) + c;
                s1[i] = c - (0.0916 * (h - l));
                s2[i] = c - (0.183 * (h - l));
                s3[i] = c - (0.275 * (h - l));
                s4[i] = c - (0.55 * (h - l));
            }
            4 => {
                r3[i] = h + 2.0 * (p - l);
                r4[i] = r3[i] + (h - l);
                r2[i] = p + (h - l);
                r1[i] = 2.0 * p - l;
                s1[i] = 2.0 * p - h;
                s2[i] = p - (h - l);
                s3[i] = l - 2.0 * (h - p);
                s4[i] = s3[i] - (h - l);
            }
            _ => {}
        }
    }

    Ok(PivotOutput {
        r4,
        r3,
        r2,
        r1,
        pp,
        s1,
        s2,
        s3,
        s4,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_pivot_default_mode_camarilla() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = PivotParams { mode: None };
        let input = PivotInput::from_candles(&candles, params);
        let result = pivot(&input).expect("Failed to calculate PIVOT with default Camarilla");

        assert_eq!(result.r4.len(), candles.close.len());
        assert_eq!(result.r3.len(), candles.close.len());
        assert_eq!(result.r2.len(), candles.close.len());
        assert_eq!(result.r1.len(), candles.close.len());
        assert_eq!(result.pp.len(), candles.close.len());
        assert_eq!(result.s1.len(), candles.close.len());
        assert_eq!(result.s2.len(), candles.close.len());
        assert_eq!(result.s3.len(), candles.close.len());
        assert_eq!(result.s4.len(), candles.close.len());

        let last_five_r4 = &result.r4[result.r4.len().saturating_sub(5)..];
        let expected_r4 = [59466.5, 59357.55, 59243.6, 59334.85, 59170.35];
        for (i, &val) in last_five_r4.iter().enumerate() {
            let exp = expected_r4[i];
            assert!(
                (val - exp).abs() < 1e-1,
                "Camarilla r4 mismatch at index {}, expected {}, got {}",
                i,
                exp,
                val
            );
        }

        let last_five_r3 = &result.r3[result.r3.len().saturating_sub(5)..];
        let expected_r3 = [59375.75, 59269.275, 59141.3, 59244.925, 58912.675];
        for (i, &val) in last_five_r3.iter().enumerate() {
            let exp = expected_r3[i];
            assert!(
                (val - exp).abs() < 1e-1,
                "Camarilla r3 mismatch at index {}, expected {}, got {}",
                i,
                exp,
                val
            );
        }

        let last_five_r2 = &result.r2[result.r2.len().saturating_sub(5)..];
        let expected_r2 = [59345.39, 59239.743, 59107.076, 59214.841, 58826.471];
        for (i, &val) in last_five_r2.iter().enumerate() {
            let exp = expected_r2[i];
            assert!(
                (val - exp).abs() < 1e-1,
                "Camarilla r2 mismatch at index {}, expected {}, got {}",
                i,
                exp,
                val
            );
        }

        let last_five_r1 = &result.r1[result.r1.len().saturating_sub(5)..];
        let expected_r1 = [59315.228, 59210.4036, 59073.0752, 59184.9532, 58740.8292];
        for (i, &val) in last_five_r1.iter().enumerate() {
            let exp = expected_r1[i];
            assert!(
                (val - exp).abs() < 1e-1,
                "Camarilla r1 mismatch at index {}, expected {}, got {}",
                i,
                exp,
                val
            );
        }

        let last_five_s1 = &result.s1[result.s1.len().saturating_sub(5)..];
        let expected_s1 = [59254.772, 59151.5964, 59004.9248, 59125.0468, 58569.1708];
        for (i, &val) in last_five_s1.iter().enumerate() {
            let exp = expected_s1[i];
            assert!(
                (val - exp).abs() < 1e-1,
                "Camarilla s1 mismatch at index {}, expected {}, got {}",
                i,
                exp,
                val
            );
        }

        let last_five_s2 = &result.s2[result.s2.len().saturating_sub(5)..];
        let expected_s2 = [59224.61, 59122.257, 58970.924, 59095.159, 58483.529];
        for (i, &val) in last_five_s2.iter().enumerate() {
            let exp = expected_s2[i];
            assert!(
                (val - exp).abs() < 1e-1,
                "Camarilla s2 mismatch at index {}, expected {}, got {}",
                i,
                exp,
                val
            );
        }

        let last_five_s3 = &result.s3[result.s3.len().saturating_sub(5)..];
        let expected_s3 = [59194.25, 59092.725, 58936.7, 59065.075, 58397.325];
        for (i, &val) in last_five_s3.iter().enumerate() {
            let exp = expected_s3[i];
            assert!(
                (val - exp).abs() < 1e-1,
                "Camarilla s3 mismatch at index {}, expected {}, got {}",
                i,
                exp,
                val
            );
        }

        let last_five_s4 = &result.s4[result.s4.len().saturating_sub(5)..];
        let expected_s4 = [59103.5, 59004.45, 58834.4, 58975.15, 58139.65];
        for (i, &val) in last_five_s4.iter().enumerate() {
            let exp = expected_s4[i];
            assert!(
                (val - exp).abs() < 1e-1,
                "Camarilla s4 mismatch at index {}, expected {}, got {}",
                i,
                exp,
                val
            );
        }
    }

    #[test]
    fn test_pivot_nan_values() {
        let high = [10.0, f64::NAN, 30.0];
        let low = [9.0, 8.5, f64::NAN];
        let close = [9.5, 9.0, 29.0];
        let open = [9.1, 8.8, 28.5];

        let params = PivotParams { mode: Some(3) };
        let input = PivotInput::from_slices(&high, &low, &close, &open, params);
        let result = pivot(&input).expect("Pivot calculation should succeed on partial NaN");
        assert_eq!(result.pp.len(), high.len());
    }

    #[test]
    fn test_pivot_no_data() {
        let high: [f64; 0] = [];
        let low: [f64; 0] = [];
        let close: [f64; 0] = [];
        let open: [f64; 0] = [];
        let params = PivotParams { mode: Some(3) };
        let input = PivotInput::from_slices(&high, &low, &close, &open, params);
        let result = pivot(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("One or more required fields"),
                "Expected 'EmptyData' error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_pivot_all_nan() {
        let high = [f64::NAN, f64::NAN];
        let low = [f64::NAN, f64::NAN];
        let close = [f64::NAN, f64::NAN];
        let open = [f64::NAN, f64::NAN];
        let params = PivotParams { mode: Some(3) };
        let input = PivotInput::from_slices(&high, &low, &close, &open, params);
        let result = pivot(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("All values are NaN"),
                "Expected 'AllValuesNaN' error, got: {}",
                e
            );
        }
    }
}
