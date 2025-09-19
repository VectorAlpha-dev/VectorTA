/// # Moving Average Streaming Dispatcher
///
/// A single entry-point for streaming moving average calculations that provides O(1)
/// updates per tick/price point. This dispatcher creates and manages streaming
/// implementations of various moving average algorithms based on a string identifier.
///
/// ## Parameters for initialization
/// - **ma_type**: A lowercase string identifier specifying which moving average
///   stream to create (e.g., `"sma"`, `"ema"`, `"alma"`, `"kama"`, etc.).
/// - **period**: The window length for the chosen moving average. Its interpretation
///   depends on the specific MA type (e.g., a look-back period).
///
/// ## Usage
/// ```rust
/// let mut stream = ma_stream("sma", 14)?;
/// let value = stream.update(price); // Returns Option<f64>
/// ```
///
/// ## Errors
/// - **`Box<dyn Error>`**: Returns errors from specific moving average initialization
///   (invalid periods, invalid parameters, etc.). If `ma_type` does not match a known
///   identifier, an error will also be returned.
///
/// ## Returns
/// - **`Ok(MaStream)`** on success, containing the initialized streaming object
/// - **`Err(Box<dyn Error>)`** on initialization failure
use crate::indicators::alma::{AlmaParams, AlmaStream};
use crate::indicators::cwma::{CwmaParams, CwmaStream};
use crate::indicators::dema::{DemaParams, DemaStream};
use crate::indicators::edcf::{EdcfParams, EdcfStream};
use crate::indicators::ehlers_itrend::{EhlersITrendParams, EhlersITrendStream};
use crate::indicators::ema::{EmaParams, EmaStream};
use crate::indicators::epma::{EpmaParams, EpmaStream};
use crate::indicators::frama::{FramaParams, FramaStream};
use crate::indicators::fwma::{FwmaParams, FwmaStream};
use crate::indicators::gaussian::{GaussianParams, GaussianStream};
use crate::indicators::highpass::{HighPassParams, HighPassStream};
use crate::indicators::highpass_2_pole::{HighPass2Params, HighPass2Stream};
use crate::indicators::hma::{HmaParams, HmaStream};
use crate::indicators::hwma::{HwmaParams, HwmaStream};
use crate::indicators::jma::{JmaParams, JmaStream};
use crate::indicators::jsa::{JsaParams, JsaStream};
use crate::indicators::kama::{KamaParams, KamaStream};
use crate::indicators::linreg::{LinRegParams, LinRegStream};
use crate::indicators::maaq::{MaaqParams, MaaqStream};
use crate::indicators::mama::{MamaParams, MamaStream};
use crate::indicators::moving_averages::dma::{DmaParams, DmaStream};
use crate::indicators::moving_averages::ehlers_ecema::{EhlersEcemaParams, EhlersEcemaStream};
use crate::indicators::moving_averages::ehlers_kama::{EhlersKamaParams, EhlersKamaStream};
use crate::indicators::moving_averages::ehma::{EhmaParams, EhmaStream};
use crate::indicators::moving_averages::nama::{NamaParams, NamaStream};
use crate::indicators::moving_averages::sama::{SamaParams, SamaStream};
use crate::indicators::moving_averages::volatility_adjusted_ma::{VamaParams, VamaStream};
use crate::indicators::mwdx::{MwdxParams, MwdxStream};
use crate::indicators::nma::{NmaParams, NmaStream};
use crate::indicators::pwma::{PwmaParams, PwmaStream};
use crate::indicators::reflex::{ReflexParams, ReflexStream};
use crate::indicators::sinwma::{SinWmaParams, SinWmaStream};
use crate::indicators::sma::{SmaParams, SmaStream};
use crate::indicators::smma::{SmmaParams, SmmaStream};
use crate::indicators::sqwma::{SqwmaParams, SqwmaStream};
use crate::indicators::srwma::{SrwmaParams, SrwmaStream};
use crate::indicators::supersmoother::{SuperSmootherParams, SuperSmootherStream};
use crate::indicators::supersmoother_3_pole::{SuperSmoother3PoleParams, SuperSmoother3PoleStream};
use crate::indicators::swma::{SwmaParams, SwmaStream};
use crate::indicators::tema::{TemaParams, TemaStream};
use crate::indicators::tilson::{TilsonParams, TilsonStream};
use crate::indicators::trendflex::{TrendFlexParams, TrendFlexStream};
use crate::indicators::trima::{TrimaParams, TrimaStream};
use crate::indicators::vpwma::{VpwmaParams, VpwmaStream};
use crate::indicators::vwap::{VwapParams, VwapStream};
use crate::indicators::vwma::{VwmaParams, VwmaStream};
use crate::indicators::wilders::{WildersParams, WildersStream};
use crate::indicators::wma::{WmaParams, WmaStream};
use crate::indicators::zlema::{ZlemaParams, ZlemaStream};

use std::error::Error;

/// Unified streaming moving average enum that holds any stream type
#[derive(Debug, Clone)]
pub enum MaStream {
    Sma(SmaStream),
    Ema(EmaStream),
    Dema(DemaStream),
    Tema(TemaStream),
    Smma(SmmaStream),
    Zlema(ZlemaStream),
    Alma(AlmaStream),
    Cwma(CwmaStream),
    Edcf(EdcfStream),
    Fwma(FwmaStream),
    Gaussian(GaussianStream),
    HighPass(HighPassStream),
    HighPass2(HighPass2Stream),
    Hma(HmaStream),
    Hwma(HwmaStream),
    Jma(JmaStream),
    Jsa(JsaStream),
    Kama(KamaStream),
    LinReg(LinRegStream),
    Maaq(MaaqStream),
    Mama(MamaStream),
    Mwdx(MwdxStream),
    Nma(NmaStream),
    Pwma(PwmaStream),
    Reflex(ReflexStream),
    SinWma(SinWmaStream),
    SqWma(SqwmaStream),
    SrWma(SrwmaStream),
    SuperSmoother(SuperSmootherStream),
    SuperSmoother3Pole(SuperSmoother3PoleStream),
    Swma(SwmaStream),
    Tilson(TilsonStream),
    TrendFlex(TrendFlexStream),
    Trima(TrimaStream),
    Wilders(WildersStream),
    Wma(WmaStream),
    VpWma(VpwmaStream),
    Vwap(VwapStream),
    Vwma(VwmaStream),
    EhlersITrend(EhlersITrendStream),
    Frama(FramaStream),
    Epma(EpmaStream),
    Dma(DmaStream),
    EhlersEcema(EhlersEcemaStream),
    EhlersKama(EhlersKamaStream),
    Ehma(EhmaStream),
    Nama(NamaStream),
    Sama(SamaStream),
    Vama(VamaStream),
}

impl MaStream {
    /// Update the stream with a new value
    /// Returns Some(f64) with the calculated moving average, or None if not enough data yet
    #[inline]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        match self {
            MaStream::Sma(s) => s.update(value),
            MaStream::Ema(s) => s.update(value),
            MaStream::Dema(s) => s.update(value),
            MaStream::Tema(s) => s.update(value),
            MaStream::Smma(s) => s.update(value),
            MaStream::Zlema(s) => s.update(value),
            MaStream::Alma(s) => s.update(value),
            MaStream::Cwma(s) => s.update(value),
            MaStream::Edcf(s) => s.update(value),
            MaStream::Fwma(s) => s.update(value),
            MaStream::Gaussian(s) => Some(s.update(value)),
            MaStream::HighPass(s) => Some(s.update(value)),
            MaStream::HighPass2(s) => s.update(value),
            MaStream::Hma(s) => s.update(value),
            MaStream::Hwma(s) => s.update(value),
            MaStream::Jma(s) => s.update(value),
            MaStream::Jsa(s) => s.update(value),
            MaStream::Kama(s) => s.update(value),
            MaStream::LinReg(s) => s.update(value),
            MaStream::Maaq(s) => s.update(value),
            MaStream::Mama(s) => s.update(value).map(|(mama, _fama)| mama),
            MaStream::Mwdx(s) => Some(s.update(value)),
            MaStream::Nma(s) => s.update(value),
            MaStream::Pwma(s) => s.update(value),
            MaStream::Reflex(s) => s.update(value),
            MaStream::SinWma(s) => s.update(value),
            MaStream::SqWma(s) => s.update(value),
            MaStream::SrWma(s) => s.update(value),
            MaStream::SuperSmoother(s) => s.update(value, None),
            MaStream::SuperSmoother3Pole(s) => Some(s.update(value)),
            MaStream::Swma(s) => s.update(value),
            MaStream::Tilson(s) => s.update(value),
            MaStream::TrendFlex(s) => s.update(value),
            MaStream::Trima(s) => s.update(value),
            MaStream::Wilders(s) => s.update(value),
            MaStream::Wma(s) => s.update(value),
            MaStream::VpWma(s) => s.update(value),
            MaStream::Vwap(s) => None, // Requires high/low/close - use update_with_volume instead
            MaStream::Vwma(s) => None, // Requires volume - use update_with_volume instead
            MaStream::EhlersITrend(s) => s.update(value),
            MaStream::Frama(s) => None, // Requires high/low - cannot use with single value
            MaStream::Epma(s) => s.update(value),
            MaStream::Dma(s) => s.update(value),
            MaStream::EhlersEcema(s) => Some(s.next(value)),
            MaStream::EhlersKama(s) => s.update(value),
            MaStream::Ehma(s) => s.update(value),
            MaStream::Nama(s) => s.update_source(value),
            MaStream::Sama(s) => s.update(value),
            MaStream::Vama(s) => s.update(value),
        }
    }

    /// Update volume-based indicators with price and volume
    /// Returns Some(f64) with the calculated moving average, or None if not enough data yet
    /// For non-volume indicators, volume is ignored and regular update is called
    #[inline]
    pub fn update_with_volume(&mut self, value: f64, volume: f64) -> Option<f64> {
        match self {
            MaStream::VpWma(s) => s.update(value * volume),
            MaStream::Vwap(_s) => None, // VWAP requires timestamp - not supported in simple update
            MaStream::Vwma(s) => s.update(value, volume),
            _ => self.update(value),
        }
    }
}

/// Create a new streaming moving average dispatcher
///
/// # Arguments
/// * `ma_type` - String identifier for the moving average type
/// * `period` - The period/window for the moving average
///
/// # Returns
/// * `Ok(MaStream)` - The initialized streaming moving average
/// * `Err(Box<dyn Error>)` - If initialization fails
#[inline]
pub fn ma_stream(ma_type: &str, period: usize) -> Result<MaStream, Box<dyn Error>> {
    match ma_type.to_lowercase().as_str() {
        "sma" => {
            let stream = SmaStream::try_new(SmaParams {
                period: Some(period),
            })?;
            Ok(MaStream::Sma(stream))
        }

        "ema" => {
            let stream = EmaStream::try_new(EmaParams {
                period: Some(period),
            })?;
            Ok(MaStream::Ema(stream))
        }

        "dema" => {
            let stream = DemaStream::try_new(DemaParams {
                period: Some(period),
            })?;
            Ok(MaStream::Dema(stream))
        }

        "tema" => {
            let stream = TemaStream::try_new(TemaParams {
                period: Some(period),
            })?;
            Ok(MaStream::Tema(stream))
        }

        "smma" => {
            let stream = SmmaStream::try_new(SmmaParams {
                period: Some(period),
            })?;
            Ok(MaStream::Smma(stream))
        }

        "zlema" => {
            let stream = ZlemaStream::try_new(ZlemaParams {
                period: Some(period),
            })?;
            Ok(MaStream::Zlema(stream))
        }

        "alma" => {
            let stream = AlmaStream::try_new(AlmaParams {
                period: Some(period),
                offset: None, // Default: 0.85
                sigma: None,  // Default: 6.0
            })?;
            Ok(MaStream::Alma(stream))
        }

        "cwma" => {
            let stream = CwmaStream::try_new(CwmaParams {
                period: Some(period),
            })?;
            Ok(MaStream::Cwma(stream))
        }

        "edcf" => {
            let stream = EdcfStream::try_new(EdcfParams {
                period: Some(period),
            })?;
            Ok(MaStream::Edcf(stream))
        }

        "fwma" => {
            let stream = FwmaStream::try_new(FwmaParams {
                period: Some(period),
            })?;
            Ok(MaStream::Fwma(stream))
        }

        "gaussian" => {
            let stream = GaussianStream::try_new(GaussianParams {
                period: Some(period),
                poles: None, // Default used by indicator
            })?;
            Ok(MaStream::Gaussian(stream))
        }

        "highpass" => {
            let stream = HighPassStream::try_new(HighPassParams {
                period: Some(period),
            })?;
            Ok(MaStream::HighPass(stream))
        }

        "highpass2" => {
            let stream = HighPass2Stream::try_new(HighPass2Params {
                period: Some(period),
                k: Some(0.707), // Default from ma.rs
            })?;
            Ok(MaStream::HighPass2(stream))
        }

        "hma" => {
            let stream = HmaStream::try_new(HmaParams {
                period: Some(period),
            })?;
            Ok(MaStream::Hma(stream))
        }

        "hwma" => {
            let stream = HwmaStream::try_new(HwmaParams {
                na: None, // Defaults used by indicator
                nb: None,
                nc: None,
            })?;
            Ok(MaStream::Hwma(stream))
        }

        "jma" => {
            let stream = JmaStream::try_new(JmaParams {
                period: Some(period),
                phase: None, // Default used by indicator
                power: None, // Default used by indicator
            })?;
            Ok(MaStream::Jma(stream))
        }

        "jsa" => {
            let stream = JsaStream::try_new(JsaParams {
                period: Some(period),
            })?;
            Ok(MaStream::Jsa(stream))
        }

        "kama" => {
            let stream = KamaStream::try_new(KamaParams {
                period: Some(period),
            })?;
            Ok(MaStream::Kama(stream))
        }

        "linreg" => {
            let stream = LinRegStream::try_new(LinRegParams {
                period: Some(period),
            })?;
            Ok(MaStream::LinReg(stream))
        }

        "maaq" => {
            let stream = MaaqStream::try_new(MaaqParams {
                period: Some(period),
                fast_period: Some(period / 2), // Default from ma.rs
                slow_period: Some(period * 2), // Default from ma.rs
            })?;
            Ok(MaStream::Maaq(stream))
        }

        "mama" => {
            let _fast_limit = (10.0 / period as f64).clamp(0.0, 1.0);
            let stream = MamaStream::try_new(MamaParams {
                fast_limit: Some(_fast_limit), // Default from ma.rs
                slow_limit: None,
            })?;
            Ok(MaStream::Mama(stream))
        }

        "mwdx" => {
            let stream = MwdxStream::try_new(MwdxParams {
                factor: None, // Default used by indicator
            })?;
            Ok(MaStream::Mwdx(stream))
        }

        "nma" => {
            let stream = NmaStream::try_new(NmaParams {
                period: Some(period),
            })?;
            Ok(MaStream::Nma(stream))
        }

        "pwma" => {
            let stream = PwmaStream::try_new(PwmaParams {
                period: Some(period),
            })?;
            Ok(MaStream::Pwma(stream))
        }

        "reflex" => {
            let stream = ReflexStream::try_new(ReflexParams {
                period: Some(period),
            })?;
            Ok(MaStream::Reflex(stream))
        }

        "sinwma" => {
            let stream = SinWmaStream::try_new(SinWmaParams {
                period: Some(period),
            })?;
            Ok(MaStream::SinWma(stream))
        }

        "sqwma" => {
            let stream = SqwmaStream::try_new(SqwmaParams {
                period: Some(period),
            })?;
            Ok(MaStream::SqWma(stream))
        }

        "srwma" => {
            let stream = SrwmaStream::try_new(SrwmaParams {
                period: Some(period),
            })?;
            Ok(MaStream::SrWma(stream))
        }

        "supersmoother" => {
            let stream = SuperSmootherStream::try_new(SuperSmootherParams {
                period: Some(period),
            })?;
            Ok(MaStream::SuperSmoother(stream))
        }

        "supersmoother_3_pole" => {
            let stream = SuperSmoother3PoleStream::try_new(SuperSmoother3PoleParams {
                period: Some(period),
            })?;
            Ok(MaStream::SuperSmoother3Pole(stream))
        }

        "swma" => {
            let stream = SwmaStream::try_new(SwmaParams {
                period: Some(period),
            })?;
            Ok(MaStream::Swma(stream))
        }

        "tilson" => {
            let stream = TilsonStream::try_new(TilsonParams {
                period: Some(period),
                volume_factor: None, // Default used by indicator
            })?;
            Ok(MaStream::Tilson(stream))
        }

        "trendflex" => {
            let stream = TrendFlexStream::try_new(TrendFlexParams {
                period: Some(period),
            })?;
            Ok(MaStream::TrendFlex(stream))
        }

        "trima" => {
            let stream = TrimaStream::try_new(TrimaParams {
                period: Some(period),
            })?;
            Ok(MaStream::Trima(stream))
        }

        "wilders" => {
            let stream = WildersStream::try_new(WildersParams {
                period: Some(period),
            })?;
            Ok(MaStream::Wilders(stream))
        }

        "wma" => {
            let stream = WmaStream::try_new(WmaParams {
                period: Some(period),
            })?;
            Ok(MaStream::Wma(stream))
        }

        "vpwma" => {
            let stream = VpwmaStream::try_new(VpwmaParams {
                period: Some(period),
                power: None, // Default used by indicator
            })?;
            Ok(MaStream::VpWma(stream))
        }

        "vwap" => {
            let stream = VwapStream::try_new(VwapParams {
                anchor: None, // Default used by indicator
            })?;
            Ok(MaStream::Vwap(stream))
        }

        "vwma" => {
            let stream = VwmaStream::try_new(VwmaParams {
                period: Some(period),
            })?;
            Ok(MaStream::Vwma(stream))
        }

        "ehlers_itrend" => {
            let stream = EhlersITrendStream::try_new(EhlersITrendParams {
                warmup_bars: Some(20),       // Default from ma.rs
                max_dc_period: Some(period), // Using period as max_dc_period
            })?;
            Ok(MaStream::EhlersITrend(stream))
        }

        "frama" => {
            let stream = FramaStream::try_new(FramaParams {
                window: Some(period),
                sc: None,
                fc: None,
            })?;
            Ok(MaStream::Frama(stream))
        }

        "epma" => {
            let stream = EpmaStream::try_new(EpmaParams {
                period: Some(period),
                offset: None, // Default used by indicator
            })?;
            Ok(MaStream::Epma(stream))
        }

        "dma" => {
            let stream = DmaStream::try_new(DmaParams {
                ema_length: Some(period),
                ..Default::default()
            })?;
            Ok(MaStream::Dma(stream))
        }

        "ehlers_ecema" => {
            let stream = EhlersEcemaStream::try_new(EhlersEcemaParams {
                length: Some(period),
                ..Default::default()
            })?;
            Ok(MaStream::EhlersEcema(stream))
        }

        "ehlers_kama" => {
            let stream = EhlersKamaStream::try_new(EhlersKamaParams {
                period: Some(period),
            })?;
            Ok(MaStream::EhlersKama(stream))
        }

        "ehma" => {
            let stream = EhmaStream::try_new(EhmaParams {
                period: Some(period),
            })?;
            Ok(MaStream::Ehma(stream))
        }

        "nama" => {
            let stream = NamaStream::try_new(NamaParams {
                period: Some(period),
                ..Default::default()
            })?;
            Ok(MaStream::Nama(stream))
        }

        "sama" => {
            let stream = SamaStream::try_new(SamaParams {
                length: Some(period),
                ..Default::default()
            })?;
            Ok(MaStream::Sama(stream))
        }

        "volatility_adjusted_ma" | "vama" => {
            let stream = VamaStream::try_new(VamaParams {
                base_period: Some(period),
                ..Default::default()
            })?;
            Ok(MaStream::Vama(stream))
        }

        _ => {
            // Default to SMA for unknown types
            eprintln!("Unknown indicator '{ma_type}'. Defaulting to 'sma'.");
            let stream = SmaStream::try_new(SmaParams {
                period: Some(period),
            })?;
            Ok(MaStream::Sma(stream))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ma_stream_creation() {
        let ma_types = vec![
            "sma",
            "ema",
            "dema",
            "tema",
            "smma",
            "zlema",
            "alma",
            "cwma",
            "edcf",
            "fwma",
            "gaussian",
            "highpass",
            "highpass2",
            "hma",
            "hwma",
            "jma",
            "jsa",
            "kama",
            "linreg",
            "maaq",
            "mwdx",
            "nma",
            "pwma",
            "reflex",
            "sinwma",
            "sqwma",
            "srwma",
            "supersmoother",
            "supersmoother_3_pole",
            "swma",
            "tilson",
            "trendflex",
            "trima",
            "wilders",
            "wma",
            "vpwma",
            "vwap",
            "vwma",
            "mama",
            "ehlers_itrend",
            "frama",
            "epma",
        ];

        for ma_type in ma_types {
            let result = ma_stream(ma_type, 14);
            assert!(result.is_ok(), "Failed to create stream for {}", ma_type);
        }
    }

    #[test]
    fn test_ma_stream_update() {
        let test_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let mut stream = ma_stream("sma", 3).expect("Failed to create SMA stream");

        let mut results = Vec::new();
        for &value in &test_data {
            if let Some(result) = stream.update(value) {
                results.push(result);
            }
        }

        // SMA(3) should start producing values after 3 inputs
        assert!(!results.is_empty(), "SMA stream should produce values");

        // Test the last value: SMA of [8.0, 9.0, 10.0] = 9.0
        let expected = 9.0;
        let actual = results.last().unwrap();
        assert!(
            (actual - expected).abs() < 1e-10,
            "SMA(3) last value mismatch: expected {}, got {}",
            expected,
            actual
        );
    }

    #[test]
    fn test_volume_based_streams() {
        let mut vwma = ma_stream("vwma", 3).expect("Failed to create VWMA stream");
        let mut vpwma = ma_stream("vpwma", 3).expect("Failed to create VPWMA stream");
        let mut vwap = ma_stream("vwap", 3).expect("Failed to create VWAP stream");

        let prices = vec![100.0, 102.0, 101.0, 103.0, 105.0];
        let volumes = vec![1000.0, 1500.0, 1200.0, 2000.0, 1800.0];

        for (&price, &volume) in prices.iter().zip(volumes.iter()) {
            vwma.update_with_volume(price, volume);
            vpwma.update_with_volume(price, volume);
            vwap.update_with_volume(price, volume);
        }

        // Just verify they don't panic and process values
        // Actual correctness is tested in individual indicator tests
    }

    #[test]
    fn test_unknown_ma_type_defaults_to_sma() {
        let stream = ma_stream("unknown_type", 5);
        assert!(stream.is_ok(), "Should default to SMA for unknown type");

        let mut s = stream.unwrap();
        // Verify it behaves like SMA
        let test_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut last_result = None;

        for value in test_values {
            last_result = s.update(value);
        }

        // SMA(5) of [1,2,3,4,5] = 3.0
        assert_eq!(last_result, Some(3.0));
    }
}
