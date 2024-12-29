use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub enum GaussianData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct GaussianOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct GaussianParams {
    pub period: Option<usize>,
    pub poles: Option<usize>,
}

impl Default for GaussianParams {
    fn default() -> Self {
        Self {
            period: Some(14),
            poles: Some(4),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GaussianInput<'a> {
    pub data: GaussianData<'a>,
    pub params: GaussianParams,
}

impl<'a> GaussianInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: GaussianParams) -> Self {
        Self {
            data: GaussianData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: GaussianParams) -> Self {
        Self {
            data: GaussianData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: GaussianData::Candles {
                candles,
                source: "close",
            },
            params: GaussianParams::default(),
        }
    }
}

pub fn gaussian(input: &GaussianInput) -> Result<GaussianOutput, Box<dyn Error>> {
    let period = input.params.period.unwrap_or(14);
    let poles = input.params.poles.unwrap_or(4);

    let data: &[f64] = match &input.data {
        GaussianData::Candles { candles, source } => source_type(candles, source),
        GaussianData::Slice(slice) => slice,
    };

    let n: usize = data.len();
    if n == 0 {
        return Err("No data provided to Gaussian filter.".into());
    }
    if !(1..=4).contains(&poles) {
        return Err("Gaussian filter poles must be in 1..4.".into());
    }
    if data.len() < input.params.period.unwrap_or(14) {
        return Err("Gaussian filter period is longer than the data.".into());
    }

    use std::f64::consts::PI;
    let beta = {
        let numerator = 1.0 - (2.0 * PI / period as f64).cos();
        let denominator = (2.0_f64).powf(1.0 / poles as f64) - 1.0;
        numerator / denominator
    };
    let alpha = {
        let tmp = beta * beta + 2.0 * beta;
        -beta + tmp.sqrt()
    };

    let output_vals = match poles {
        1 => gaussian_poles1(data, n, alpha),
        2 => gaussian_poles2(data, n, alpha),
        3 => gaussian_poles3(data, n, alpha),
        4 => gaussian_poles4(data, n, alpha),
        _ => unreachable!(),
    };

    Ok(GaussianOutput {
        values: output_vals,
    })
}

#[inline(always)]
fn gaussian_poles1(data: &[f64], n: usize, alpha: f64) -> Vec<f64> {
    let c0 = alpha;
    let c1 = 1.0 - alpha;

    let mut fil = vec![0.0; 1 + n];

    for i in 0..n {
        fil[i + 1] = c0 * data[i] + c1 * fil[i];
    }

    fil[1..1 + n].to_vec()
}

#[inline(always)]
fn gaussian_poles2(data: &[f64], n: usize, alpha: f64) -> Vec<f64> {
    let a2 = alpha * alpha;
    let one_a = 1.0 - alpha;
    let c0 = a2;
    let c1 = 2.0 * one_a;
    let c2 = -(one_a * one_a);

    let mut fil = vec![0.0; 2 + n];

    for i in 0..n {
        fil[i + 2] = c0 * data[i] + c1 * fil[i + 1] + c2 * fil[i];
    }

    fil[2..2 + n].to_vec()
}

#[inline(always)]
fn gaussian_poles3(data: &[f64], n: usize, alpha: f64) -> Vec<f64> {
    let a3 = alpha * alpha * alpha;
    let one_a = 1.0 - alpha;
    let one_a2 = one_a * one_a;
    let c0 = a3;
    let c1 = 3.0 * one_a;
    let c2 = -3.0 * one_a2;
    let c3 = one_a2 * one_a;

    let mut fil = vec![0.0; 3 + n];

    for i in 0..n {
        fil[i + 3] = c0 * data[i] + c1 * fil[i + 2] + c2 * fil[i + 1] + c3 * fil[i];
    }

    fil[3..3 + n].to_vec()
}

#[inline(always)]
fn gaussian_poles4(data: &[f64], n: usize, alpha: f64) -> Vec<f64> {
    let a4 = alpha * alpha * alpha * alpha;
    let one_a = 1.0 - alpha;
    let one_a2 = one_a * one_a;
    let one_a3 = one_a2 * one_a;
    let c0 = a4;
    let c1 = 4.0 * one_a;
    let c2 = -6.0 * one_a2;
    let c3 = 4.0 * one_a3;
    let c4 = -(one_a3 * one_a);

    let mut fil = vec![0.0; 4 + n];

    for i in 0..n {
        fil[i + 4] =
            c0 * data[i] + c1 * fil[i + 3] + c2 * fil[i + 2] + c3 * fil[i + 1] + c4 * fil[i];
    }

    fil[4..4 + n].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_gaussian_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = GaussianParams {
            period: Some(14),
            poles: Some(4),
        };
        let input = GaussianInput::from_candles(&candles, "close", params);

        let gaussian_result = gaussian(&input).expect("Failed to calculate Gaussian filter");

        let expected_last_five = [
            59221.90637814869,
            59236.15215167245,
            59207.10087088464,
            59178.48276885589,
            59085.36983209433,
        ];
        let len = gaussian_result.values.len();
        assert!(len >= 5, "Not enough Gaussian filter values for the test");
        assert_eq!(
            len,
            candles.close.len(),
            "Gaussian filter output length does not match input length"
        );
        let start_index = len - 5;
        let last_five = &gaussian_result.values[start_index..];

        for (i, &val) in last_five.iter().enumerate() {
            let exp = expected_last_five[i];
            assert!(
                (val - exp).abs() < 1e-4,
                "Gaussian filter mismatch at last-five index {}: expected {}, got {}",
                i,
                exp,
                val
            );
        }

        let skip = input.params.poles.unwrap_or(4);
        for val in gaussian_result.values.iter().skip(skip) {
            assert!(
                val.is_finite(),
                "Gaussian output should be finite once settled."
            );
        }
    }
    #[test]
    fn test_gaussian_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = GaussianInput::with_default_candles(&candles);
        match input.data {
            GaussianData::Candles { source, .. } => {
                assert_eq!(source, "close");
            }
            _ => panic!("Expected GaussianData::Candles"),
        }
        let period = input.params.period.unwrap_or(14);
        let poles = input.params.poles.unwrap_or(4);
        assert_eq!(period, 14);
        assert_eq!(poles, 4);
    }

    #[test]
    fn test_gaussian_with_default_params() {
        let default_params = GaussianParams::default();
        assert_eq!(default_params.period, Some(14));
        assert_eq!(default_params.poles, Some(4));
    }

    #[test]
    fn test_gaussian_with_no_data() {
        let data: [f64; 0] = [];
        let input = GaussianInput::from_slice(
            &data,
            GaussianParams {
                period: Some(14),
                poles: Some(4),
            },
        );
        let result = gaussian(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e
                .to_string()
                .contains("No data provided to Gaussian filter"));
        }
    }

    #[test]
    fn test_gaussian_with_out_of_range_poles() {
        let data = [10.0, 20.0, 30.0];
        let input = GaussianInput::from_slice(
            &data,
            GaussianParams {
                period: Some(14),
                poles: Some(5),
            },
        );
        let result = gaussian(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e
                .to_string()
                .contains("Gaussian filter poles must be in 1..4."));
        }
    }

    #[test]
    fn test_gaussian_very_small_data_set() {
        let data = [42.0];
        let input = GaussianInput::from_slice(
            &data,
            GaussianParams {
                period: Some(14),
                poles: Some(4),
            },
        );
        let result = gaussian(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("period is longer than the data."));
        }
    }

    #[test]
    fn test_gaussian_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let first_input = GaussianInput::from_candles(
            &candles,
            "close",
            GaussianParams {
                period: Some(14),
                poles: Some(4),
            },
        );
        let first_result = gaussian(&first_input).expect("First Gaussian filter failed");
        let second_input = GaussianInput::from_slice(
            &first_result.values,
            GaussianParams {
                period: Some(7),
                poles: Some(2),
            },
        );
        let second_result = gaussian(&second_input).expect("Second Gaussian filter failed");
        assert_eq!(second_result.values.len(), first_result.values.len());
    }

    #[test]
    fn test_gaussian_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = GaussianInput::from_candles(
            &candles,
            "close",
            GaussianParams {
                period: None,
                poles: None,
            },
        );
        let result = gaussian(&input).expect("Gaussian calculation failed");
        assert_eq!(result.values.len(), candles.close.len());
    }

    #[test]
    fn test_gaussian_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = GaussianInput::from_candles(
            &candles,
            "close",
            GaussianParams {
                period: Some(14),
                poles: Some(4),
            },
        );
        let result = gaussian(&input).expect("Gaussian calculation failed");
        let start_index = input.params.poles.unwrap_or(4);
        for i in start_index..result.values.len() {
            assert!(!result.values[i].is_nan());
        }
    }
}
