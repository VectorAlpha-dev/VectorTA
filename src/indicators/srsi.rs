use crate::indicators::rsi::{rsi, RsiData, RsiError, RsiInput, RsiOutput, RsiParams};
use crate::indicators::stoch::{
    stoch, StochData, StochError, StochInput, StochOutput, StochParams,
};
/// # Stochastic RSI (SRSI)
///
/// A momentum oscillator that applies the Stochastic formula to the RSI values instead of price data.
///
/// ## Parameters
/// - **rsi_period**: The period for RSI calculation. Defaults to 14.
/// - **stoch_period**: The period for Stochastic calculation on RSI. Defaults to 14.
/// - **k**: The period for the slow K moving average in Stochastic. Defaults to 3.
/// - **d**: The period for the slow D moving average in Stochastic. Defaults to 3.
/// - **source**: The candle field to be used (if using candles). Defaults to `"close"`.
///
/// ## Errors
/// - **RsiError**: If the RSI calculation fails.
/// - **StochError**: If the Stochastic calculation (on the computed RSI) fails.
///
/// ## Returns
/// - **`Ok(SrsiOutput)`** on success, containing vectors `k` and `d` (both matching the input length),
///   with leading `NaN`s until each component can be calculated.
/// - **`Err(SrsiError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum SrsiData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct SrsiOutput {
    pub k: Vec<f64>,
    pub d: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SrsiParams {
    pub rsi_period: Option<usize>,
    pub stoch_period: Option<usize>,
    pub k: Option<usize>,
    pub d: Option<usize>,
    pub source: Option<String>,
}

impl Default for SrsiParams {
    fn default() -> Self {
        Self {
            rsi_period: Some(14),
            stoch_period: Some(14),
            k: Some(3),
            d: Some(3),
            source: Some("close".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SrsiInput<'a> {
    pub data: SrsiData<'a>,
    pub params: SrsiParams,
}

impl<'a> SrsiInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: SrsiParams) -> Self {
        Self {
            data: SrsiData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: SrsiParams) -> Self {
        Self {
            data: SrsiData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: SrsiData::Candles {
                candles,
                source: "close",
            },
            params: SrsiParams::default(),
        }
    }

    pub fn get_rsi_period(&self) -> usize {
        self.params
            .rsi_period
            .unwrap_or_else(|| SrsiParams::default().rsi_period.unwrap())
    }

    pub fn get_stoch_period(&self) -> usize {
        self.params
            .stoch_period
            .unwrap_or_else(|| SrsiParams::default().stoch_period.unwrap())
    }

    pub fn get_k(&self) -> usize {
        self.params
            .k
            .unwrap_or_else(|| SrsiParams::default().k.unwrap())
    }

    pub fn get_d(&self) -> usize {
        self.params
            .d
            .unwrap_or_else(|| SrsiParams::default().d.unwrap())
    }

    pub fn get_source(&self) -> String {
        self.params
            .source
            .clone()
            .unwrap_or_else(|| SrsiParams::default().source.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum SrsiError {
    #[error("srsi: Error from RSI calculation: {0}")]
    RsiError(#[from] RsiError),
    #[error("srsi: Error from Stochastic calculation: {0}")]
    StochError(#[from] StochError),
}

#[inline]
pub fn srsi(input: &SrsiInput) -> Result<SrsiOutput, SrsiError> {
    let rsi_period = input.get_rsi_period();
    let stoch_period = input.get_stoch_period();
    let k_len = input.get_k();
    let d_len = input.get_d();

    let rsi_values = match &input.data {
        SrsiData::Candles { candles, source } => {
            let rsi_input = RsiInput::from_candles(
                candles,
                source,
                RsiParams {
                    period: Some(rsi_period),
                },
            );
            rsi(&rsi_input)?.values
        }
        SrsiData::Slice(slice) => {
            let rsi_input = RsiInput::from_slice(
                slice,
                RsiParams {
                    period: Some(rsi_period),
                },
            );
            rsi(&rsi_input)?.values
        }
    };

    let stoch_input = StochInput {
        data: crate::indicators::stoch::StochData::Slices {
            high: &rsi_values,
            low: &rsi_values,
            close: &rsi_values,
        },
        params: StochParams {
            fastk_period: Some(stoch_period),
            slowk_period: Some(k_len),
            slowk_ma_type: Some("sma".to_string()),
            slowd_period: Some(d_len),
            slowd_ma_type: Some("sma".to_string()),
        },
    };

    let stoch_output: StochOutput = stoch(&stoch_input)?;
    Ok(SrsiOutput {
        k: stoch_output.k,
        d: stoch_output.d,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_srsi_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = SrsiInput::with_default_candles(&candles);
        let output = srsi(&input).expect("Failed SRSI with default params");
        assert_eq!(output.k.len(), candles.close.len());
        assert_eq!(output.d.len(), candles.close.len());
    }

    #[test]
    fn test_srsi_custom_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = SrsiParams {
            rsi_period: Some(10),
            stoch_period: Some(10),
            k: Some(4),
            d: Some(4),
            source: Some("hlc3".to_string()),
        };
        let input = SrsiInput::from_candles(&candles, "hlc3", params);
        let output = srsi(&input).expect("Failed SRSI with custom params");
        assert_eq!(output.k.len(), candles.close.len());
        assert_eq!(output.d.len(), candles.close.len());
    }

    #[test]
    fn test_srsi_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = SrsiParams::default();
        let input = SrsiInput::from_candles(&candles, "close", params);
        let result = srsi(&input).expect("Failed to calculate SRSI");

        assert_eq!(result.k.len(), candles.close.len());
        assert_eq!(result.d.len(), candles.close.len());

        let last_five_k = [
            65.52066633236464,
            61.22507053191985,
            57.220471530042644,
            64.61344854988147,
            60.66534359318523,
        ];
        let last_five_d = [
            64.33503158970049,
            64.42143544464182,
            61.32206946477942,
            61.01966353728503,
            60.83308789104016,
        ];
        assert!(result.k.len() >= 5 && result.d.len() >= 5);
        let k_slice = &result.k[result.k.len() - 5..];
        let d_slice = &result.d[result.d.len() - 5..];
        for i in 0..5 {
            let diff_k = (k_slice[i] - last_five_k[i]).abs();
            let diff_d = (d_slice[i] - last_five_d[i]).abs();
            assert!(
                diff_k < 1e-6,
                "Mismatch in SRSI K at index {}: got {}, expected {}",
                i,
                k_slice[i],
                last_five_k[i]
            );
            assert!(
                diff_d < 1e-6,
                "Mismatch in SRSI D at index {}: got {}, expected {}",
                i,
                d_slice[i],
                last_five_d[i]
            );
        }
    }

    #[test]
    fn test_srsi_from_slice() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let slice_data = candles.close.as_slice();
        let params = SrsiParams {
            rsi_period: Some(3),
            stoch_period: Some(3),
            k: Some(2),
            d: Some(2),
            source: Some("close".to_string()),
        };
        let input = SrsiInput::from_slice(&slice_data, params);
        let output = srsi(&input).expect("Failed SRSI from slice");
        assert_eq!(output.k.len(), slice_data.len());
        assert_eq!(output.d.len(), slice_data.len());
    }
}
