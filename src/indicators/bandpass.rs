use crate::indicators::highpass::{highpass, HighPassInput, HighPassParams};
use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct BandPassParams {
    pub period: Option<usize>,
    pub bandwidth: Option<f64>,
}

impl BandPassParams {
    pub fn with_default_params() -> Self {
        BandPassParams {
            period: None,
            bandwidth: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BandPassInput<'a> {
    pub candles: &'a Candles,
    pub source: &'a str,
    pub params: BandPassParams,
}

impl<'a> BandPassInput<'a> {
    pub fn new(candles: &'a Candles, source: &'a str, params: BandPassParams) -> Self {
        BandPassInput {
            candles,
            source,
            params,
        }
    }

    pub fn with_default_params(candles: &'a Candles) -> Self {
        BandPassInput {
            candles,
            source: "close",
            params: BandPassParams::with_default_params(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BandPassOutput {
    pub bp: Vec<f64>,
    pub bp_normalized: Vec<f64>,
    pub signal: Vec<f64>,
    pub trigger: Vec<f64>,
}

#[inline]
pub fn bandpass(input: &BandPassInput) -> Result<BandPassOutput, Box<dyn Error>> {
    let data: &[f64] = source_type(input.candles, input.source);
    let len: usize = data.len();
    let period: usize = input.params.period.unwrap_or(20);
    let bandwidth: f64 = input.params.bandwidth.unwrap_or(0.3);

    if len == 0 {
        return Err("No data available.".into());
    }
    if period < 2 {
        return Err("BandPass period must be >= 2".into());
    }

    let hp_period_f = 4.0 * (period as f64) / bandwidth;
    let hp_period = hp_period_f.round() as usize;
    if hp_period < 2 {
        return Err("hp_period is too small after rounding.".into());
    }

    let mut hp_params = HighPassParams::with_default_params();
    hp_params.period = Some(hp_period);

    let hp_input = HighPassInput::from_slice(data, hp_params);
    let hp_result = highpass(&hp_input)?;
    let hp = hp_result.values;

    let beta = (2.0 * PI / period as f64).cos();
    let gamma = (2.0 * PI * bandwidth / period as f64).cos();
    let alpha = 1.0 / gamma - ((1.0 / (gamma * gamma)) - 1.0).sqrt();

    let mut bp = hp.clone();
    if len >= 2 {
        for i in 2..len {
            bp[i] = 0.5 * (1.0 - alpha) * hp[i] - 0.5 * (1.0 - alpha) * hp[i - 2]
                + beta * (1.0 + alpha) * bp[i - 1]
                - alpha * bp[i - 2];
        }
    }

    let k = 0.991;
    let mut peak_prev = 0.0;
    let mut bp_normalized = vec![0.0; len];
    for i in 0..len {
        peak_prev *= k;
        let abs_bp = bp[i].abs();
        if abs_bp > peak_prev {
            peak_prev = abs_bp;
        }
        if peak_prev != 0.0 {
            bp_normalized[i] = bp[i] / peak_prev;
        } else {
            bp_normalized[i] = 0.0;
        }
    }

    let trigger_period_f = (period as f64 / bandwidth) / 1.5;
    let trigger_period = trigger_period_f.round() as usize;
    if trigger_period < 2 {
        return Err("trigger_period is too small after rounding.".into());
    }
    let mut trigger_params = HighPassParams::with_default_params();
    trigger_params.period = Some(trigger_period);
    let trigger_input = HighPassInput::from_slice(&bp_normalized, trigger_params);

    let trigger_result = highpass(&trigger_input)?;
    let trigger = trigger_result.values;

    let mut signal = vec![0.0; len];
    for i in 0..len {
        let bn = bp_normalized[i];
        let tr = trigger[i];
        if bn < tr {
            signal[i] = 1.0;
        } else if bn > tr {
            signal[i] = -1.0;
        } else {
            signal[i] = 0.0;
        }
    }

    Ok(BandPassOutput {
        bp,
        bp_normalized,
        signal,
        trigger,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    #[test]
    fn test_bandpass_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        // Use default BandPass parameters
        let default_input = BandPassInput::with_default_params(&candles);
        let default_output = bandpass(&default_input).expect("Failed bandpass with default params");
        assert_eq!(default_output.bp.len(), candles.close.len());

        // Custom parameters
        let custom_params = BandPassParams {
            period: Some(30),
            bandwidth: Some(0.5),
        };
        let custom_input = BandPassInput::new(&candles, "hl2", custom_params);
        let custom_output = bandpass(&custom_input).expect("Failed bandpass with custom params");
        assert_eq!(custom_output.bp.len(), candles.close.len());
    }
    #[test]
    fn test_bandpass_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = BandPassInput::with_default_params(&candles);
        let result = bandpass(&input).expect("Failed to calculate bandpass");

        let expected_bp_last_five = [
            -236.23678021132827,
            -247.4846395608195,
            -242.3788746078502,
            -212.89589193350128,
            -179.97293838509464,
        ];

        let expected_bp_normalized_last_five = [
            -0.4399672555578846,
            -0.4651011734720517,
            -0.4596426251402882,
            -0.40739824942488945,
            -0.3475245023284841,
        ];

        let expected_signal_last_five = [-1.0, 1.0, 1.0, 1.0, 1.0];

        let expected_trigger_last_five = [
            -0.4746908356434579,
            -0.4353877348116954,
            -0.3727126131420441,
            -0.2746336628365846,
            -0.18240018384226137,
        ];

        assert!(result.bp.len() >= 5, "Not enough bp values");
        assert!(
            result.bp_normalized.len() >= 5,
            "Not enough bp_normalized values"
        );
        assert!(result.signal.len() >= 5, "Not enough signal values");
        assert!(result.trigger.len() >= 5, "Not enough trigger values");

        let start_bp = result.bp.len().saturating_sub(5);
        let start_bpn = result.bp_normalized.len().saturating_sub(5);
        let start_sig = result.signal.len().saturating_sub(5);
        let start_trg = result.trigger.len().saturating_sub(5);

        let bp_last_five = &result.bp[start_bp..];
        let bp_normalized_last_five = &result.bp_normalized[start_bpn..];
        let signal_last_five = &result.signal[start_sig..];
        let trigger_last_five = &result.trigger[start_trg..];

        assert_eq!(
            result.bp.len(),
            candles.close.len(),
            "BandPass output length does not match input length"
        );

        assert_eq!(
            result.bp_normalized.len(),
            candles.close.len(),
            "BandPass Normalized output length does not match input length"
        );

        assert_eq!(
            result.signal.len(),
            candles.close.len(),
            "Signal output length does not match input length"
        );

        assert_eq!(
            result.trigger.len(),
            candles.close.len(),
            "Trigger output length does not match input length"
        );

        for (i, &value) in bp_last_five.iter().enumerate() {
            assert!(
                (value - expected_bp_last_five[i]).abs() < 1e-1,
                "BP value mismatch at index {}: expected {}, got {}",
                i,
                expected_bp_last_five[i],
                value
            );
        }

        for (i, &value) in bp_normalized_last_five.iter().enumerate() {
            assert!(
                (value - expected_bp_normalized_last_five[i]).abs() < 1e-1,
                "BP Normalized value mismatch at index {}: expected {}, got {}",
                i,
                expected_bp_normalized_last_five[i],
                value
            );
        }

        for (i, &value) in signal_last_five.iter().enumerate() {
            assert!(
                (value - expected_signal_last_five[i]).abs() < 1e-1,
                "Signal value mismatch at index {}: expected {}, got {}",
                i,
                expected_signal_last_five[i],
                value
            );
        }

        for (i, &value) in trigger_last_five.iter().enumerate() {
            assert!(
                (value - expected_trigger_last_five[i]).abs() < 1e-1,
                "Trigger value mismatch at index {}: expected {}, got {}",
                i,
                expected_trigger_last_five[i],
                value
            );
        }

        for val in &result.bp {
            assert!(val.is_finite(), "BP output should be finite");
        }
        for val in &result.bp_normalized {
            assert!(val.is_finite(), "BP Normalized output should be finite");
        }
        for val in &result.signal {
            assert!(val.is_finite(), "Signal output should be finite");
        }
        for val in &result.trigger {
            assert!(val.is_finite(), "Trigger output should be finite");
        }
    }
}
