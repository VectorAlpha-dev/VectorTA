/// # Volume Price Confirmation Index (VPCI)
///
/// VPCI aims to confirm price movements using volume-weighted moving averages,
/// comparing price and volume trends to identify confluence or divergence.
/// It uses the existing SMA function to compute necessary moving averages.
///
/// ## Parameters
/// - **short_range**: Window size for the short-term average. Defaults to 5.
/// - **long_range**: Window size for the long-term average. Defaults to 25.
///
/// ## Errors
/// - **SmaError**: Wraps any SMA errors (e.g., empty data, invalid period).
///
/// ## Returns
/// - **`Ok(VpciOutput)`** on success, containing two `Vec<f64>` (VPCI and VPCIS) matching the input length,
///   with leading `NaN`s until the moving average windows are satisfied.
/// - **`Err(VpciError)`** otherwise.
use crate::indicators::sma::{sma, SmaData, SmaError, SmaInput, SmaParams};
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum VpciData<'a> {
    Candles {
        candles: &'a Candles,
        close_source: &'a str,
        volume_source: &'a str,
    },
    Slices {
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct VpciOutput {
    pub vpci: Vec<f64>,
    pub vpcis: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct VpciParams {
    pub short_range: Option<usize>,
    pub long_range: Option<usize>,
}

impl Default for VpciParams {
    fn default() -> Self {
        Self {
            short_range: Some(5),
            long_range: Some(25),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VpciInput<'a> {
    pub data: VpciData<'a>,
    pub params: VpciParams,
}

impl<'a> VpciInput<'a> {
    pub fn from_candles(
        candles: &'a Candles,
        close_source: &'a str,
        volume_source: &'a str,
        params: VpciParams,
    ) -> Self {
        Self {
            data: VpciData::Candles {
                candles,
                close_source,
                volume_source,
            },
            params,
        }
    }

    pub fn from_slices(close: &'a [f64], volume: &'a [f64], params: VpciParams) -> Self {
        Self {
            data: VpciData::Slices { close, volume },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: VpciData::Candles {
                candles,
                close_source: "close",
                volume_source: "volume",
            },
            params: VpciParams::default(),
        }
    }

    pub fn get_short_range(&self) -> usize {
        self.params
            .short_range
            .unwrap_or_else(|| VpciParams::default().short_range.unwrap())
    }

    pub fn get_long_range(&self) -> usize {
        self.params
            .long_range
            .unwrap_or_else(|| VpciParams::default().long_range.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum VpciError {
    #[error("vpci: SMA error: {0}")]
    SmaError(#[from] SmaError),
}

#[inline]
pub fn vpci(input: &VpciInput) -> Result<VpciOutput, VpciError> {
    let (close, volume) = match &input.data {
        VpciData::Candles {
            candles,
            close_source,
            volume_source,
        } => {
            let c = source_type(candles, close_source);
            let v = source_type(candles, volume_source);
            (c, v)
        }
        VpciData::Slices { close, volume } => (*close, *volume),
    };

    let short_range = input.get_short_range();
    let long_range = input.get_long_range();

    let close_volume_product: Vec<f64> = close
        .iter()
        .zip(volume.iter())
        .map(|(c, v)| c * v)
        .collect();

    let sma_close_long = sma(&SmaInput {
        data: SmaData::Slice(&close),
        params: SmaParams {
            period: Some(long_range),
        },
    })?
    .values;
    let sma_close_short = sma(&SmaInput {
        data: SmaData::Slice(&close),
        params: SmaParams {
            period: Some(short_range),
        },
    })?
    .values;
    let sma_volume_long = sma(&SmaInput {
        data: SmaData::Slice(&volume),
        params: SmaParams {
            period: Some(long_range),
        },
    })?
    .values;
    let sma_volume_short = sma(&SmaInput {
        data: SmaData::Slice(&volume),
        params: SmaParams {
            period: Some(short_range),
        },
    })?
    .values;
    let sma_close_vol_long = sma(&SmaInput {
        data: SmaData::Slice(&close_volume_product),
        params: SmaParams {
            period: Some(long_range),
        },
    })?
    .values;
    let sma_close_vol_short = sma(&SmaInput {
        data: SmaData::Slice(&close_volume_product),
        params: SmaParams {
            period: Some(short_range),
        },
    })?
    .values;

    let mut vpci_vals = vec![f64::NAN; close.len()];
    let mut vpcis_vals = vec![f64::NAN; close.len()];
    let mut vwma_long = vec![f64::NAN; close.len()];
    let mut vwma_short = vec![f64::NAN; close.len()];

    for i in 0..close.len() {
        if !sma_volume_long[i].is_nan() && sma_volume_long[i] != 0.0 {
            vwma_long[i] = sma_close_vol_long[i] / sma_volume_long[i];
        }
        if !sma_volume_short[i].is_nan() && sma_volume_short[i] != 0.0 {
            vwma_short[i] = sma_close_vol_short[i] / sma_volume_short[i];
        }
    }

    let mut vpci_times_vol = vec![f64::NAN; close.len()];

    for i in 0..close.len() {
        let vpc = vwma_long[i] - sma_close_long[i];
        let vpr = if !sma_close_short[i].is_nan() && sma_close_short[i] != 0.0 {
            vwma_short[i] / sma_close_short[i]
        } else {
            f64::NAN
        };
        let vm = if !sma_volume_long[i].is_nan() && sma_volume_long[i] != 0.0 {
            sma_volume_short[i] / sma_volume_long[i]
        } else {
            f64::NAN
        };
        let val = vpc * vpr * vm;
        vpci_vals[i] = val;
        if !val.is_nan() && !volume[i].is_nan() {
            vpci_times_vol[i] = val * volume[i];
        }
    }

    let sma_vpci_times_vol_short = sma(&SmaInput {
        data: SmaData::Slice(&vpci_times_vol),
        params: SmaParams {
            period: Some(short_range),
        },
    })?
    .values;

    for i in 0..close.len() {
        if !sma_volume_short[i].is_nan() && sma_volume_short[i] != 0.0 {
            vpcis_vals[i] = sma_vpci_times_vol_short[i] / sma_volume_short[i];
        }
    }

    Ok(VpciOutput {
        vpci: vpci_vals,
        vpcis: vpcis_vals,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_vpci_default_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = VpciInput::with_default_candles(&candles);
        let output = vpci(&input).expect("Failed to calculate VPCI with default params");
        assert_eq!(output.vpci.len(), candles.close.len());
        assert_eq!(output.vpcis.len(), candles.close.len());
    }

    #[test]
    fn test_vpci_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = VpciParams {
            short_range: Some(3),
            long_range: None,
        };
        let input = VpciInput::from_candles(&candles, "close", "volume", params);
        let output = vpci(&input).expect("Failed to calculate VPCI with partial params");
        assert_eq!(output.vpci.len(), candles.close.len());
        assert_eq!(output.vpcis.len(), candles.close.len());
    }

    #[test]
    fn test_vpci_slice_input() {
        let close_data = [10.0, 12.0, 14.0, 13.0, 15.0];
        let volume_data = [100.0, 200.0, 300.0, 250.0, 400.0];
        let params = VpciParams {
            short_range: Some(2),
            long_range: Some(3),
        };
        let input = VpciInput::from_slices(&close_data, &volume_data, params);
        let output = vpci(&input).expect("Failed to calculate VPCI from slices");
        assert_eq!(output.vpci.len(), close_data.len());
        assert_eq!(output.vpcis.len(), close_data.len());
    }

    #[test]
    fn test_vpci_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = VpciParams {
            short_range: Some(5),
            long_range: Some(25),
        };
        let input = VpciInput::from_candles(&candles, "close", "volume", params);
        let output = vpci(&input).expect("Failed to calculate VPCI");

        let vpci_len = output.vpci.len();
        let vpcis_len = output.vpcis.len();
        assert_eq!(vpci_len, candles.close.len());
        assert_eq!(vpcis_len, candles.close.len());

        let vpci_last_five = &output.vpci[vpci_len.saturating_sub(5)..];
        let vpcis_last_five = &output.vpcis[vpcis_len.saturating_sub(5)..];
        let expected_vpci = [
            -319.65148214323426,
            -133.61700649928346,
            -144.76194155503174,
            -83.55576212490328,
            -169.53504207700533,
        ];
        let expected_vpcis = [
            -1049.2826640115732,
            -694.1067814399748,
            -519.6960416662324,
            -330.9401404636258,
            -173.004986803695,
        ];
        for (i, &val) in vpci_last_five.iter().enumerate() {
            let diff = (val - expected_vpci[i]).abs();
            assert!(
                diff < 1e-1,
                "VPCI mismatch at index {}: expected {}, got {}",
                i,
                expected_vpci[i],
                val
            );
        }
        for (i, &val) in vpcis_last_five.iter().enumerate() {
            let diff = (val - expected_vpcis[i]).abs();
            assert!(
                diff < 1e-1,
                "VPCIS mismatch at index {}: expected {}, got {}",
                i,
                expected_vpcis[i],
                val
            );
        }
    }
}
