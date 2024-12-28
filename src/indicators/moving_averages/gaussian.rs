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

    fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }

    fn get_poles(&self) -> usize {
        self.params.poles.unwrap_or(4)
    }
}

pub fn gaussian(input: &GaussianInput) -> Result<GaussianOutput, Box<dyn Error>> {
    let data: &[f64] = match &input.data {
        GaussianData::Candles { candles, source } => source_type(candles, source),
        GaussianData::Slice(slice) => slice,
    };

    let n: usize = data.len();
    if n == 0 {
        return Err("No data provided to Gaussian filter.".into());
    }

    let period = input.get_period();
    let poles = input.get_poles();
    if !(1..=4).contains(&poles) {
        return Err("Gaussian filter poles must be in 1..4.".into());
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

        let skip = input.get_poles();
        for val in gaussian_result.values.iter().skip(skip) {
            assert!(
                val.is_finite(),
                "Gaussian output should be finite once settled."
            );
        }
    }
}
