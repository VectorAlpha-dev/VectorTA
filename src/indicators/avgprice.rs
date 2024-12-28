use crate::utilities::data_loader::Candles;
use std::error::Error;

#[derive(Debug, Clone)]
pub enum AvgPriceData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct AvgPriceInput<'a> {
    pub data: AvgPriceData<'a>,
}

impl<'a> AvgPriceInput<'a> {
    pub fn from_candles(candles: &'a Candles) -> Self {
        Self {
            data: AvgPriceData::Candles { candles },
        }
    }

    pub fn from_slices(open: &'a [f64], high: &'a [f64], low: &'a [f64], close: &'a [f64]) -> Self {
        Self {
            data: AvgPriceData::Slices {
                open,
                high,
                low,
                close,
            },
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: AvgPriceData::Candles { candles },
        }
    }
}

#[derive(Debug, Clone)]
pub struct AvgPriceOutput {
    pub values: Vec<f64>,
}

#[inline]
pub fn avgprice(input: &AvgPriceInput) -> Result<AvgPriceOutput, Box<dyn Error>> {
    let (open, high, low, close) = match &input.data {
        AvgPriceData::Candles { candles } => {
            if candles.close.is_empty() {
                return Err("No candles available.".into());
            }
            let open: &[f64] = candles.select_candle_field("open")?;
            let high: &[f64] = candles.select_candle_field("high")?;
            let low: &[f64] = candles.select_candle_field("low")?;
            let close: &[f64] = candles.select_candle_field("close")?;
            (open, high, low, close)
        }
        AvgPriceData::Slices {
            open,
            high,
            low,
            close,
        } => {
            if open.is_empty() {
                return Err("Input slices have zero length.".into());
            }
            if open.len() != high.len() || high.len() != low.len() || low.len() != close.len() {
                return Err("Inconsistent slice lengths.".into());
            }
            (*open, *high, *low, *close)
        }
    };
    let len: usize = close.len();
    if len == 0 {
        return Err("No candles available.".into());
    }

    let mut values = Vec::with_capacity(len);
    for i in 0..len {
        let sum = open[i] + high[i] + low[i] + close[i];
        values.push(sum / 4.0);
    }

    Ok(AvgPriceOutput { values })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avgprice_accuracy() {
        let candles = Candles {
            timestamp: vec![1, 2, 3, 4, 5],
            open: vec![100., 101., 102., 103., 104.],
            high: vec![110., 111., 112., 113., 114.],
            low: vec![90., 91., 92., 93., 94.],
            close: vec![105., 106., 107., 108., 109.],
            volume: vec![1000., 1000., 1000., 1000., 1000.],
            hl2: vec![100., 101., 102., 103., 104.],
            hlc3: vec![100., 101., 102., 103., 104.],
            ohlc4: vec![100., 101., 102., 103., 104.],
            hlcc4: vec![100., 101., 102., 103., 104.],
        };

        let input = AvgPriceInput::from_candles(&candles);
        let result = avgprice(&input).expect("Failed to calculate avgprice");
        assert_eq!(
            result.values.len(),
            candles.close.len(),
            "Output length should match input length"
        );
        let expected = [101.25, 102.25, 103.25, 104.25, 105.25];

        assert_eq!(result.values.len(), 5);
        for (i, &val) in result.values.iter().enumerate() {
            assert!(
                (val - expected[i]).abs() < 1e-2,
                "Mismatch at index {}: expected {}, got {}",
                i,
                expected[i],
                val
            );
        }
    }
}
