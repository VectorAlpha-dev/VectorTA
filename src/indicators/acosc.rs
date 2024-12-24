use crate::indicators::data_loader::Candles;
use std::error::Error;

#[derive(Debug, Clone, Default)]
pub struct AcoscParams {}

#[derive(Debug, Clone)]
pub struct AcoscInput<'a> {
    pub candles: &'a Candles,
    pub params: AcoscParams,
}

impl<'a> AcoscInput<'a> {
    pub fn new(candles: &'a Candles, params: AcoscParams) -> Self {
        AcoscInput { candles, params }
    }

    pub fn with_default_params(candles: &'a Candles) -> Self {
        AcoscInput {
            candles,
            params: AcoscParams::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AcoscOutput {
    pub osc: Vec<f64>,
    pub change: Vec<f64>,
}

#[inline]
pub fn calculate_acosc(input: &AcoscInput) -> Result<AcoscOutput, Box<dyn Error>> {
    let candles = input.candles;

    let high_prices = candles.select_candle_field("high")?;
    let low_prices = candles.select_candle_field("low")?;

    let len = low_prices.len();
    const PERIOD_SMA5: usize = 5;
    const PERIOD_SMA34: usize = 34;
    const INV_PERIOD_SMA5: f64 = 1.0 / PERIOD_SMA5 as f64;
    const INV_PERIOD_SMA34: f64 = 1.0 / PERIOD_SMA34 as f64;
    const REQUIRED_LENGTH: usize = PERIOD_SMA34 + PERIOD_SMA5;

    if len < REQUIRED_LENGTH {
        return Err("Not enough data points to calculate AC oscillator".into());
    }

    let mut osc = vec![f64::NAN; len];
    let mut change = vec![f64::NAN; len];

    let mut queue_sma5 = [0.0; PERIOD_SMA5];
    let mut queue_sma34 = [0.0; PERIOD_SMA34];
    let mut queue_sma5_ao = [0.0; PERIOD_SMA5];

    let mut sum_sma5 = 0.0;
    let mut sum_sma34 = 0.0;
    let mut sum_sma5_ao = 0.0;

    let mut idx_sma5 = 0;
    let mut idx_sma34 = 0;
    let mut idx_sma5_ao = 0;

    for i in 0..PERIOD_SMA34 {
        let medprice = (high_prices[i] + low_prices[i]) * 0.5;

        sum_sma34 += medprice;
        queue_sma34[i] = medprice;

        if i < PERIOD_SMA5 {
            sum_sma5 += medprice;
            queue_sma5[i] = medprice;
        }
    }

    for i in PERIOD_SMA34..(PERIOD_SMA34 + PERIOD_SMA5 - 1) {
        let medprice = (high_prices[i] + low_prices[i]) * 0.5;

        sum_sma34 += medprice - queue_sma34[idx_sma34];
        queue_sma34[idx_sma34] = medprice;
        idx_sma34 += 1;
        if idx_sma34 == PERIOD_SMA34 {
            idx_sma34 = 0;
        }
        let sma34 = sum_sma34 * INV_PERIOD_SMA34;

        sum_sma5 += medprice - queue_sma5[idx_sma5];
        queue_sma5[idx_sma5] = medprice;
        idx_sma5 += 1;
        if idx_sma5 == PERIOD_SMA5 {
            idx_sma5 = 0;
        }
        let sma5 = sum_sma5 * INV_PERIOD_SMA5;

        let ao = sma5 - sma34;

        sum_sma5_ao += ao;
        queue_sma5_ao[idx_sma5_ao] = ao;
        idx_sma5_ao += 1;
    }
    if idx_sma5_ao == PERIOD_SMA5 {
        idx_sma5_ao = 0;
    }

    let mut prev_res = 0.0;

    for i in (PERIOD_SMA34 + PERIOD_SMA5 - 1)..len {
        let medprice = (high_prices[i] + low_prices[i]) * 0.5;

        sum_sma34 += medprice - queue_sma34[idx_sma34];
        queue_sma34[idx_sma34] = medprice;
        idx_sma34 += 1;
        if idx_sma34 == PERIOD_SMA34 {
            idx_sma34 = 0;
        }
        let sma34 = sum_sma34 * INV_PERIOD_SMA34;

        sum_sma5 += medprice - queue_sma5[idx_sma5];
        queue_sma5[idx_sma5] = medprice;
        idx_sma5 += 1;
        if idx_sma5 == PERIOD_SMA5 {
            idx_sma5 = 0;
        }
        let sma5 = sum_sma5 * INV_PERIOD_SMA5;

        let ao = sma5 - sma34;

        let old_ao = queue_sma5_ao[idx_sma5_ao];
        sum_sma5_ao += ao - old_ao;
        queue_sma5_ao[idx_sma5_ao] = ao;
        idx_sma5_ao += 1;
        if idx_sma5_ao == PERIOD_SMA5 {
            idx_sma5_ao = 0;
        }
        let sma5_ao = sum_sma5_ao * INV_PERIOD_SMA5;

        let res = ao - sma5_ao;
        let mom = res - prev_res;
        prev_res = res;

        osc[i] = res;
        change[i] = mom;
    }

    Ok(AcoscOutput { osc, change })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::data_loader::read_candles_from_csv;

    #[test]
    fn test_acosc_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = AcoscInput::with_default_params(&candles);
        let acosc_result = calculate_acosc(&input).expect("Failed to calculate acosc");

        assert_eq!(
            acosc_result.osc.len(),
            candles.close.len(),
            "ACOSC output length (osc) does not match input length"
        );
        assert_eq!(
            acosc_result.change.len(),
            candles.close.len(),
            "ACOSC output length (change) does not match input length"
        );

        let expected_last_five_acosc_osc = [273.30, 383.72, 357.7, 291.25, 176.84];
        let expected_last_five_acosc_change = [49.6, 110.4, -26.0, -66.5, -114.4];

        assert!(acosc_result.osc.len() >= 5);
        assert!(acosc_result.change.len() >= 5);

        let start_index_osc = acosc_result.osc.len().saturating_sub(5);
        let result_last_five_acosc_osc = &acosc_result.osc[start_index_osc..];

        let start_index_change = acosc_result.change.len().saturating_sub(5);
        let result_last_five_acosc_change = &acosc_result.change[start_index_change..];

        for (i, &value) in result_last_five_acosc_osc.iter().enumerate() {
            assert!(
                (value - expected_last_five_acosc_osc[i]).abs() < 1e-1,
                "acosc osc value mismatch at index {}: expected {}, got {}",
                i,
                expected_last_five_acosc_osc[i],
                value
            );
        }

        for (i, &value) in result_last_five_acosc_change.iter().enumerate() {
            assert!(
                (value - expected_last_five_acosc_change[i]).abs() < 1e-1,
                "acosc change value mismatch at index {}: expected {}, got {}",
                i,
                expected_last_five_acosc_change[i],
                value
            );
        }
    }
}
