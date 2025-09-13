/// # Moving Average Dispatcher
///
/// A single entry-point function that dispatches the call to various moving average
/// implementations, based on a string identifier (`ma_type`). Supported identifiers
/// include `"sma"`, `"ema"`, `"wma"`, etc., and each corresponds to a specific
/// underlying moving average algorithm. This allows you to switch between different
/// moving average types at runtime with a uniform interface.
///
/// ## Parameters
/// - **ma_type**: A lowercase string identifier specifying which moving average
///   function to call (e.g., `"sma"`, `"ema"`, `"alma"`, `"kama"`, etc.).
/// - **data**: An enum that can contain either candle data + source or a raw slice of `f64`.
/// - **period**: The window length for the chosen moving average. Its interpretation
///   depends on the specific MA type (e.g., a look-back period).
///
/// ## Errors
/// - **`Box<dyn Error>`**: Propagates any errors returned by the specific moving average
///   function. This may include invalid periods, missing or mismatched data fields,
///   insufficient data length, etc. If `ma_type` does not match a known identifier,
///   an error will also be returned.
///
/// ## Returns
/// - **`Ok(Vec<f64>)`** on success, containing the computed moving average values
///   of length matching the input data.
/// - **`Err(Box<dyn Error>)`** otherwise.
use crate::indicators::alma::{alma, AlmaData, AlmaInput, AlmaParams};
use crate::indicators::cwma::{cwma, CwmaData, CwmaInput, CwmaParams};
use crate::indicators::dema::{dema, DemaData, DemaInput, DemaParams};
use crate::indicators::edcf::{edcf, EdcfData, EdcfInput, EdcfParams};
use crate::indicators::ehlers_itrend::{ehlers_itrend, EhlersITrendData, EhlersITrendInput, EhlersITrendParams};
use crate::indicators::ema::{ema, EmaData, EmaInput, EmaParams};
use crate::indicators::epma::{epma, EpmaData, EpmaInput, EpmaParams};
use crate::indicators::fwma::{fwma, FwmaData, FwmaInput, FwmaParams};
use crate::indicators::gaussian::{gaussian, GaussianData, GaussianInput, GaussianParams};
use crate::indicators::highpass::{highpass, HighPassData, HighPassInput, HighPassParams};
use crate::indicators::highpass_2_pole::{highpass_2_pole, HighPass2Data, HighPass2Input, HighPass2Params};
use crate::indicators::hma::{hma, HmaData, HmaInput, HmaParams};
use crate::indicators::hwma::{hwma, HwmaData, HwmaInput, HwmaParams};
use crate::indicators::jma::{jma, JmaData, JmaInput, JmaParams};
use crate::indicators::jsa::{jsa, JsaData, JsaInput, JsaParams};
use crate::indicators::kama::{kama, KamaData, KamaInput, KamaParams};
use crate::indicators::linreg::{linreg, LinRegData, LinRegInput, LinRegParams};
use crate::indicators::maaq::{maaq, MaaqData, MaaqInput, MaaqParams};
use crate::indicators::mama::{mama, MamaData, MamaInput, MamaParams};
use crate::indicators::mwdx::{mwdx, MwdxData, MwdxInput, MwdxParams};
use crate::indicators::nma::{nma, NmaData, NmaInput, NmaParams};
use crate::indicators::pwma::{pwma, PwmaData, PwmaInput, PwmaParams};
use crate::indicators::reflex::{reflex, ReflexData, ReflexInput, ReflexParams};
use crate::indicators::sinwma::{sinwma, SinWmaData, SinWmaInput, SinWmaParams};
use crate::indicators::sma::{sma, SmaData, SmaInput, SmaParams};
use crate::indicators::smma::{smma, SmmaData, SmmaInput, SmmaParams};
use crate::indicators::sqwma::{sqwma, SqwmaData, SqwmaInput, SqwmaParams};
use crate::indicators::srwma::{srwma, SrwmaData, SrwmaInput, SrwmaParams};
use crate::indicators::supersmoother::{supersmoother, SuperSmootherData, SuperSmootherInput, SuperSmootherParams};
use crate::indicators::supersmoother_3_pole::{
	supersmoother_3_pole, SuperSmoother3PoleData, SuperSmoother3PoleInput, SuperSmoother3PoleParams,
};
use crate::indicators::swma::{swma, SwmaData, SwmaInput, SwmaParams};
use crate::indicators::tema::{tema, TemaData, TemaInput, TemaParams};
use crate::indicators::tilson::{tilson, TilsonData, TilsonInput, TilsonParams};
use crate::indicators::trendflex::{trendflex, TrendFlexData, TrendFlexInput, TrendFlexParams};
use crate::indicators::trima::{trima, TrimaData, TrimaInput, TrimaParams};
use crate::indicators::vpwma::{vpwma, VpwmaData, VpwmaInput, VpwmaParams};
use crate::indicators::vwap::{vwap, VwapData, VwapInput, VwapParams};
use crate::indicators::vwma::{vwma, VwmaData, VwmaInput, VwmaParams};
use crate::indicators::wilders::{wilders, WildersData, WildersInput, WildersParams};
use crate::indicators::wma::{wma, WmaData, WmaInput, WmaParams};
use crate::indicators::zlema::{zlema, ZlemaData, ZlemaInput, ZlemaParams};
use crate::indicators::moving_averages::dma::{dma, DmaData, DmaInput, DmaParams};
use crate::indicators::moving_averages::ehlers_ecema::{ehlers_ecema, EhlersEcemaData, EhlersEcemaInput, EhlersEcemaParams};
use crate::indicators::moving_averages::ehlers_kama::{ehlers_kama, EhlersKamaData, EhlersKamaInput, EhlersKamaParams};
use crate::indicators::moving_averages::ehma::{ehma, EhmaData, EhmaInput, EhmaParams};
use crate::indicators::moving_averages::nama::{nama, NamaData, NamaInput, NamaParams};
use crate::indicators::moving_averages::sama::{sama, SamaData, SamaInput, SamaParams};
use crate::indicators::moving_averages::volatility_adjusted_ma::{vama, VamaData, VamaInput, VamaParams};
use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use std::error::Error;

// Import _with_kernel functions
use crate::indicators::alma::alma_with_kernel;
use crate::indicators::cwma::cwma_with_kernel;
use crate::indicators::dema::dema_with_kernel;
use crate::indicators::edcf::edcf_with_kernel;
use crate::indicators::ehlers_itrend::ehlers_itrend_with_kernel;
use crate::indicators::ema::ema_with_kernel;
use crate::indicators::epma::epma_with_kernel;
use crate::indicators::fwma::fwma_with_kernel;
use crate::indicators::gaussian::gaussian_with_kernel;
use crate::indicators::highpass::highpass_with_kernel;
use crate::indicators::highpass_2_pole::highpass_2_pole_with_kernel;
use crate::indicators::hma::hma_with_kernel;
use crate::indicators::hwma::hwma_with_kernel;
use crate::indicators::jma::jma_with_kernel;
use crate::indicators::jsa::jsa_with_kernel;
use crate::indicators::kama::kama_with_kernel;
use crate::indicators::linreg::linreg_with_kernel;
use crate::indicators::maaq::maaq_with_kernel;
use crate::indicators::mama::mama_with_kernel;
use crate::indicators::mwdx::mwdx_with_kernel;
use crate::indicators::nma::nma_with_kernel;
use crate::indicators::pwma::pwma_with_kernel;
use crate::indicators::reflex::reflex_with_kernel;
use crate::indicators::sinwma::sinwma_with_kernel;
use crate::indicators::sma::sma_with_kernel;
use crate::indicators::smma::smma_with_kernel;
use crate::indicators::sqwma::sqwma_with_kernel;
use crate::indicators::srwma::srwma_with_kernel;
use crate::indicators::supersmoother::supersmoother_with_kernel;
use crate::indicators::supersmoother_3_pole::supersmoother_3_pole_with_kernel;
use crate::indicators::swma::swma_with_kernel;
use crate::indicators::tema::tema_with_kernel;
use crate::indicators::tilson::tilson_with_kernel;
use crate::indicators::trendflex::trendflex_with_kernel;
use crate::indicators::trima::trima_with_kernel;
use crate::indicators::vpwma::vpwma_with_kernel;
use crate::indicators::vwap::vwap_with_kernel;
use crate::indicators::vwma::vwma_with_kernel;
use crate::indicators::wilders::wilders_with_kernel;
use crate::indicators::wma::wma_with_kernel;
use crate::indicators::zlema::zlema_with_kernel;
use crate::indicators::moving_averages::dma::dma_with_kernel;
use crate::indicators::moving_averages::ehlers_ecema::ehlers_ecema_with_kernel;
use crate::indicators::moving_averages::ehlers_kama::ehlers_kama_with_kernel;
use crate::indicators::moving_averages::ehma::ehma_with_kernel;
use crate::indicators::moving_averages::nama::nama_with_kernel;
use crate::indicators::moving_averages::sama::sama_with_kernel;
use crate::indicators::moving_averages::volatility_adjusted_ma::vama_with_kernel;

#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "python")]
use numpy::{PyArray1, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone)]
pub enum MaData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[inline]
pub fn ma<'a>(ma_type: &str, data: MaData<'a>, period: usize) -> Result<Vec<f64>, Box<dyn Error>> {
	match ma_type.to_lowercase().as_str() {
		"sma" => {
			let input = match data {
				MaData::Candles { candles, source } => SmaInput {
					data: SmaData::Candles { candles, source },
					params: SmaParams { period: Some(period) },
				},
				MaData::Slice(slice) => SmaInput {
					data: SmaData::Slice(slice),
					params: SmaParams { period: Some(period) },
				},
			};
			let output = sma(&input)?;
			Ok(output.values)
		}

		"alma" => {
			let input = match data {
				MaData::Candles { candles, source } => AlmaInput {
					data: AlmaData::Candles { candles, source },
					params: AlmaParams {
						period: Some(period),
						offset: None,
						sigma: None,
					},
				},
				MaData::Slice(slice) => AlmaInput {
					data: AlmaData::Slice(slice),
					params: AlmaParams {
						period: Some(period),
						offset: None,
						sigma: None,
					},
				},
			};
			let output = alma(&input)?;
			Ok(output.values)
		}

		"cwma" => {
			let input = match data {
				MaData::Candles { candles, source } => CwmaInput {
					data: CwmaData::Candles { candles, source },
					params: CwmaParams { period: Some(period) },
				},
				MaData::Slice(slice) => CwmaInput {
					data: CwmaData::Slice(slice),
					params: CwmaParams { period: Some(period) },
				},
			};
			let output = cwma(&input)?;
			Ok(output.values)
		}

		"dema" => {
			let input = match data {
				MaData::Candles { candles, source } => DemaInput {
					data: DemaData::Candles { candles, source },
					params: DemaParams { period: Some(period) },
				},
				MaData::Slice(slice) => DemaInput {
					data: DemaData::Slice(slice),
					params: DemaParams { period: Some(period) },
				},
			};
			let output = dema(&input)?;
			Ok(output.values)
		}

		"edcf" => {
			let input = match data {
				MaData::Candles { candles, source } => EdcfInput {
					data: EdcfData::Candles { candles, source },
					params: EdcfParams { period: Some(period) },
				},
				MaData::Slice(slice) => EdcfInput {
					data: EdcfData::Slice(slice),
					params: EdcfParams { period: Some(period) },
				},
			};
			let output = edcf(&input)?;
			Ok(output.values)
		}

		"ema" => {
			let input = match data {
				MaData::Candles { candles, source } => EmaInput {
					data: EmaData::Candles { candles, source },
					params: EmaParams { period: Some(period) },
				},
				MaData::Slice(slice) => EmaInput {
					data: EmaData::Slice(slice),
					params: EmaParams { period: Some(period) },
				},
			};
			let output = ema(&input)?;
			Ok(output.values)
		}

		"epma" => {
			let input = match data {
				MaData::Candles { candles, source } => EpmaInput {
					data: EpmaData::Candles { candles, source },
					params: EpmaParams {
						period: Some(period),
						offset: None,
					},
				},
				MaData::Slice(slice) => EpmaInput {
					data: EpmaData::Slice(slice),
					params: EpmaParams {
						period: Some(period),
						offset: None,
					},
				},
			};
			let output = epma(&input)?;
			Ok(output.values)
		}

		"fwma" => {
			let input = match data {
				MaData::Candles { candles, source } => FwmaInput {
					data: FwmaData::Candles { candles, source },
					params: FwmaParams { period: Some(period) },
				},
				MaData::Slice(slice) => FwmaInput {
					data: FwmaData::Slice(slice),
					params: FwmaParams { period: Some(period) },
				},
			};
			let output = fwma(&input)?;
			Ok(output.values)
		}

		"gaussian" => {
			let input = match data {
				MaData::Candles { candles, source } => GaussianInput {
					data: GaussianData::Candles { candles, source },
					params: GaussianParams {
						period: Some(period),
						poles: None,
					},
				},
				MaData::Slice(slice) => GaussianInput {
					data: GaussianData::Slice(slice),
					params: GaussianParams {
						period: Some(period),
						poles: None,
					},
				},
			};
			let output = gaussian(&input)?;
			Ok(output.values)
		}

		"highpass" => {
			let input = match data {
				MaData::Candles { candles, source } => HighPassInput {
					data: HighPassData::Candles { candles, source },
					params: HighPassParams { period: Some(period) },
				},
				MaData::Slice(slice) => HighPassInput {
					data: HighPassData::Slice(slice),
					params: HighPassParams { period: Some(period) },
				},
			};
			let output = highpass(&input)?;
			Ok(output.values)
		}

		"highpass2" => {
			let input = match data {
				MaData::Candles { candles, source } => HighPass2Input {
					data: HighPass2Data::Candles { candles, source },
					params: HighPass2Params {
						period: Some(period),
						k: Some(0.707),
					},
				},
				MaData::Slice(slice) => HighPass2Input {
					data: HighPass2Data::Slice(slice),
					params: HighPass2Params {
						period: Some(period),
						k: Some(0.707),
					},
				},
			};
			let output = highpass_2_pole(&input)?;
			Ok(output.values)
		}

		"hma" => {
			let input = match data {
				MaData::Candles { candles, source } => HmaInput {
					data: HmaData::Candles { candles, source },
					params: HmaParams { period: Some(period) },
				},
				MaData::Slice(slice) => HmaInput {
					data: HmaData::Slice(slice),
					params: HmaParams { period: Some(period) },
				},
			};
			let output = hma(&input)?;
			Ok(output.values)
		}

		"ehlers_itrend" => {
			let input = match data {
				MaData::Candles { candles, source } => EhlersITrendInput {
					data: EhlersITrendData::Candles { candles, source },
					params: EhlersITrendParams {
						warmup_bars: Some(20),
						max_dc_period: Some(period),
					},
				},
				MaData::Slice(slice) => EhlersITrendInput {
					data: EhlersITrendData::Slice(slice),
					params: EhlersITrendParams {
						warmup_bars: Some(20),
						max_dc_period: Some(period),
					},
				},
			};
			let output = ehlers_itrend(&input)?;
			Ok(output.values)
		}

		"hwma" => {
			let input = match data {
				MaData::Candles { candles, source } => HwmaInput {
					data: HwmaData::Candles { candles, source },
					params: HwmaParams {
						na: None,
						nb: None,
						nc: None,
					},
				},
				MaData::Slice(slice) => HwmaInput {
					data: HwmaData::Slice(slice),
					params: HwmaParams {
						na: None,
						nb: None,
						nc: None,
					},
				},
			};
			let output = hwma(&input)?;
			Ok(output.values)
		}

		"jma" => {
			let input = match data {
				MaData::Candles { candles, source } => JmaInput {
					data: JmaData::Candles { candles, source },
					params: JmaParams {
						period: Some(period),
						phase: None,
						power: None,
					},
				},
				MaData::Slice(slice) => JmaInput {
					data: JmaData::Slice(slice),
					params: JmaParams {
						period: Some(period),
						phase: None,
						power: None,
					},
				},
			};
			let output = jma(&input)?;
			Ok(output.values)
		}

		"jsa" => {
			let input = match data {
				MaData::Candles { candles, source } => JsaInput {
					data: JsaData::Candles { candles, source },
					params: JsaParams { period: Some(period) },
				},
				MaData::Slice(slice) => JsaInput {
					data: JsaData::Slice(slice),
					params: JsaParams { period: Some(period) },
				},
			};
			let output = jsa(&input)?;
			Ok(output.values)
		}

		"kama" => {
			let input = match data {
				MaData::Candles { candles, source } => KamaInput {
					data: KamaData::Candles { candles, source },
					params: KamaParams { period: Some(period) },
				},
				MaData::Slice(slice) => KamaInput {
					data: KamaData::Slice(slice),
					params: KamaParams { period: Some(period) },
				},
			};
			let output = kama(&input)?;
			Ok(output.values)
		}

		"linreg" => {
			let input = match data {
				MaData::Candles { candles, source } => LinRegInput {
					data: LinRegData::Candles { candles, source },
					params: LinRegParams { period: Some(period) },
				},
				MaData::Slice(s) => LinRegInput {
					data: LinRegData::Slice(s),
					params: LinRegParams { period: Some(period) },
				},
			};
			let output = linreg(&input)?;
			Ok(output.values)
		}

		"maaq" => {
			let input = match data {
				MaData::Candles { candles, source } => MaaqInput {
					data: MaaqData::Candles { candles, source },
					params: MaaqParams {
						period: Some(period),
						fast_period: Some(period / 2),
						slow_period: Some(period * 2),
					},
				},
				MaData::Slice(s) => MaaqInput {
					data: MaaqData::Slice(s),
					params: MaaqParams {
						period: Some(period),
						fast_period: Some(period / 2),
						slow_period: Some(period * 2),
					},
				},
			};
			let output = maaq(&input)?;
			Ok(output.values)
		}

		"mama" => {
			// Use default MAMA parameters rather than trying to derive from period
			// MAMA is an adaptive algorithm that doesn't use a fixed period
			let input = match data {
				MaData::Candles { candles, source } => MamaInput::from_candles(candles, source, MamaParams::default()),
				MaData::Slice(s) => MamaInput::from_slice(s, MamaParams::default()),
			};
			let output = mama(&input)?;
			Ok(output.mama_values)
		}

		"mwdx" => {
			let input = match data {
				MaData::Candles { candles, source } => MwdxInput {
					data: MwdxData::Candles { candles, source },
					params: MwdxParams { factor: None },
				},
				MaData::Slice(s) => MwdxInput {
					data: MwdxData::Slice(s),
					params: MwdxParams { factor: None },
				},
			};
			let output = mwdx(&input)?;
			Ok(output.values)
		}

		"nma" => {
			let input = match data {
				MaData::Candles { candles, source } => NmaInput {
					data: NmaData::Candles { candles, source },
					params: NmaParams { period: Some(period) },
				},
				MaData::Slice(s) => NmaInput {
					data: NmaData::Slice(s),
					params: NmaParams { period: Some(period) },
				},
			};
			let output = nma(&input)?;
			Ok(output.values)
		}

		"pwma" => {
			let input = match data {
				MaData::Candles { candles, source } => PwmaInput {
					data: PwmaData::Candles { candles, source },
					params: PwmaParams { period: Some(period) },
				},
				MaData::Slice(s) => PwmaInput {
					data: PwmaData::Slice(s),
					params: PwmaParams { period: Some(period) },
				},
			};
			let output = pwma(&input)?;
			Ok(output.values)
		}

		"reflex" => {
			let input = match data {
				MaData::Candles { candles, source } => ReflexInput {
					data: ReflexData::Candles { candles, source },
					params: ReflexParams { period: Some(period) },
				},
				MaData::Slice(s) => ReflexInput {
					data: ReflexData::Slice(s),
					params: ReflexParams { period: Some(period) },
				},
			};
			let output = reflex(&input)?;
			Ok(output.values)
		}

		"sinwma" => {
			let input = match data {
				MaData::Candles { candles, source } => SinWmaInput {
					data: SinWmaData::Candles { candles, source },
					params: SinWmaParams { period: Some(period) },
				},
				MaData::Slice(s) => SinWmaInput {
					data: SinWmaData::Slice(s),
					params: SinWmaParams { period: Some(period) },
				},
			};
			let output = sinwma(&input)?;
			Ok(output.values)
		}

		"smma" => {
			let input = match data {
				MaData::Candles { candles, source } => SmmaInput {
					data: SmmaData::Candles { candles, source },
					params: SmmaParams { period: Some(period) },
				},
				MaData::Slice(s) => SmmaInput {
					data: SmmaData::Slice(s),
					params: SmmaParams { period: Some(period) },
				},
			};
			let output = smma(&input)?;
			Ok(output.values)
		}

		"sqwma" => {
			let input = match data {
				MaData::Candles { candles, source } => SqwmaInput {
					data: SqwmaData::Candles { candles, source },
					params: SqwmaParams { period: Some(period) },
				},
				MaData::Slice(s) => SqwmaInput {
					data: SqwmaData::Slice(s),
					params: SqwmaParams { period: Some(period) },
				},
			};
			let output = sqwma(&input)?;
			Ok(output.values)
		}

		"srwma" => {
			let input = match data {
				MaData::Candles { candles, source } => SrwmaInput {
					data: SrwmaData::Candles { candles, source },
					params: SrwmaParams { period: Some(period) },
				},
				MaData::Slice(s) => SrwmaInput {
					data: SrwmaData::Slice(s),
					params: SrwmaParams { period: Some(period) },
				},
			};
			let output = srwma(&input)?;
			Ok(output.values)
		}

		"supersmoother" => {
			let input = match data {
				MaData::Candles { candles, source } => SuperSmootherInput {
					data: SuperSmootherData::Candles { candles, source },
					params: SuperSmootherParams { period: Some(period) },
				},
				MaData::Slice(s) => SuperSmootherInput {
					data: SuperSmootherData::Slice(s),
					params: SuperSmootherParams { period: Some(period) },
				},
			};
			let output = supersmoother(&input)?;
			Ok(output.values)
		}

		"supersmoother_3_pole" => {
			let input = match data {
				MaData::Candles { candles, source } => SuperSmoother3PoleInput {
					data: SuperSmoother3PoleData::Candles { candles, source },
					params: SuperSmoother3PoleParams { period: Some(period) },
				},
				MaData::Slice(s) => SuperSmoother3PoleInput {
					data: SuperSmoother3PoleData::Slice(s),
					params: SuperSmoother3PoleParams { period: Some(period) },
				},
			};
			let output = supersmoother_3_pole(&input)?;
			Ok(output.values)
		}

		"swma" => {
			let input = match data {
				MaData::Candles { candles, source } => SwmaInput {
					data: SwmaData::Candles { candles, source },
					params: SwmaParams { period: Some(period) },
				},
				MaData::Slice(s) => SwmaInput {
					data: SwmaData::Slice(s),
					params: SwmaParams { period: Some(period) },
				},
			};
			let output = swma(&input)?;
			Ok(output.values)
		}

		"tema" => {
			let input = match data {
				MaData::Candles { candles, source } => TemaInput {
					data: TemaData::Candles { candles, source },
					params: TemaParams { period: Some(period) },
				},
				MaData::Slice(s) => TemaInput {
					data: TemaData::Slice(s),
					params: TemaParams { period: Some(period) },
				},
			};
			let output = tema(&input)?;
			Ok(output.values)
		}

		"tilson" => {
			let input = match data {
				MaData::Candles { candles, source } => TilsonInput {
					data: TilsonData::Candles { candles, source },
					params: TilsonParams {
						period: Some(period),
						volume_factor: None,
					},
				},
				MaData::Slice(s) => TilsonInput {
					data: TilsonData::Slice(s),
					params: TilsonParams {
						period: Some(period),
						volume_factor: None,
					},
				},
			};
			let output = tilson(&input)?;
			Ok(output.values)
		}

		"trendflex" => {
			let input = match data {
				MaData::Candles { candles, source } => TrendFlexInput {
					data: TrendFlexData::Candles { candles, source },
					params: TrendFlexParams { period: Some(period) },
				},
				MaData::Slice(s) => TrendFlexInput {
					data: TrendFlexData::Slice(s),
					params: TrendFlexParams { period: Some(period) },
				},
			};
			let output = trendflex(&input)?;
			Ok(output.values)
		}

		"trima" => {
			let input = match data {
				MaData::Candles { candles, source } => TrimaInput {
					data: TrimaData::Candles { candles, source },
					params: TrimaParams { period: Some(period) },
				},
				MaData::Slice(s) => TrimaInput {
					data: TrimaData::Slice(s),
					params: TrimaParams { period: Some(period) },
				},
			};
			let output = trima(&input)?;
			Ok(output.values)
		}

		"vpwma" => {
			if let MaData::Candles { candles, source } = data {
				let input = VpwmaInput {
					data: VpwmaData::Candles { candles, source },
					params: VpwmaParams {
						period: Some(period),
						power: None,
					},
				};
				let output = vpwma(&input)?;
				Ok(output.values)
			} else {
				eprintln!("Unknown data type for 'vpwma'. Defaulting to 'sma'.");

				let input = match data {
					MaData::Candles { candles, source } => {
						SmaInput::from_candles(candles, source, SmaParams { period: Some(period) })
					}
					MaData::Slice(slice) => SmaInput::from_slice(slice, SmaParams { period: Some(period) }),
				};
				let output = sma(&input)?;
				Ok(output.values)
			}
		}

		"vwap" => {
			if let MaData::Candles { candles, source } = data {
				let input = VwapInput {
					data: VwapData::Candles { candles, source },
					params: VwapParams { anchor: None },
				};
				let output = vwap(&input)?;
				Ok(output.values)
			} else {
				eprintln!("Unknown data type for 'vwap'. Defaulting to 'sma'.");

				let input = match data {
					MaData::Candles { candles, source } => {
						SmaInput::from_candles(candles, source, SmaParams { period: Some(period) })
					}
					MaData::Slice(slice) => SmaInput::from_slice(slice, SmaParams { period: Some(period) }),
				};
				let output = sma(&input)?;
				Ok(output.values)
			}
		}
		"vwma" => {
			if let MaData::Candles { candles, source } = data {
				let input = VwmaInput {
					data: VwmaData::Candles { candles, source },
					params: VwmaParams { period: Some(period) },
				};
				let output = vwma(&input)?;
				Ok(output.values)
			} else {
				eprintln!("Unknown data type for 'vwma'. Defaulting to 'sma'.");

				let input = match data {
					MaData::Candles { candles, source } => {
						SmaInput::from_candles(candles, source, SmaParams { period: Some(period) })
					}
					MaData::Slice(slice) => SmaInput::from_slice(slice, SmaParams { period: Some(period) }),
				};
				let output = sma(&input)?;
				Ok(output.values)
			}
		}

		"wilders" => {
			let input = match data {
				MaData::Candles { candles, source } => WildersInput {
					data: WildersData::Candles { candles, source },
					params: WildersParams { period: Some(period) },
				},
				MaData::Slice(s) => WildersInput {
					data: WildersData::Slice(s),
					params: WildersParams { period: Some(period) },
				},
			};
			let output = wilders(&input)?;
			Ok(output.values)
		}

		"wma" => {
			let input = match data {
				MaData::Candles { candles, source } => WmaInput {
					data: WmaData::Candles { candles, source },
					params: WmaParams { period: Some(period) },
				},
				MaData::Slice(s) => WmaInput {
					data: WmaData::Slice(s),
					params: WmaParams { period: Some(period) },
				},
			};
			let output = wma(&input)?;
			Ok(output.values)
		}

		"zlema" => {
			let input = match data {
				MaData::Candles { candles, source } => ZlemaInput {
					data: ZlemaData::Candles { candles, source },
					params: ZlemaParams { period: Some(period) },
				},
				MaData::Slice(s) => ZlemaInput {
					data: ZlemaData::Slice(s),
					params: ZlemaParams { period: Some(period) },
				},
			};
			let output = zlema(&input)?;
			Ok(output.values)
		}

		"buff_averages" => {
			// BuffAverages requires volume data and returns dual outputs, not suitable for simple MA selector
			return Err("buff_averages requires volume data, use the indicator directly".into());
		}

		"dma" => {
			let input = match data {
				MaData::Candles { candles, source } => DmaInput {
					data: DmaData::Candles { candles, source },
					params: DmaParams {
						ema_length: Some(period),
						..Default::default()
					},
				},
				MaData::Slice(s) => DmaInput {
					data: DmaData::Slice(s),
					params: DmaParams {
						ema_length: Some(period),
						..Default::default()
					},
				},
			};
			let output = dma(&input)?;
			Ok(output.values)
		}

		"ehlers_ecema" => {
			let input = match data {
				MaData::Candles { candles, source } => EhlersEcemaInput {
					data: EhlersEcemaData::Candles { candles, source },
					params: EhlersEcemaParams {
						length: Some(period),
						..Default::default()
					},
				},
				MaData::Slice(s) => EhlersEcemaInput {
					data: EhlersEcemaData::Slice(s),
					params: EhlersEcemaParams {
						length: Some(period),
						..Default::default()
					},
				},
			};
			let output = ehlers_ecema(&input)?;
			Ok(output.values)
		}

		"ehlers_kama" => {
			let input = match data {
				MaData::Candles { candles, source } => EhlersKamaInput {
					data: EhlersKamaData::Candles { candles, source },
					params: EhlersKamaParams {
						period: Some(period),
						..Default::default()
					},
				},
				MaData::Slice(s) => EhlersKamaInput {
					data: EhlersKamaData::Slice(s),
					params: EhlersKamaParams {
						period: Some(period),
						..Default::default()
					},
				},
			};
			let output = ehlers_kama(&input)?;
			Ok(output.values)
		}

		"ehlers_pma" => {
			// EhlersPma returns dual outputs (predict/trigger), not suitable for simple MA selector
			return Err("ehlers_pma returns dual outputs, use the indicator directly".into());
		}

		"ehma" => {
			let input = match data {
				MaData::Candles { candles, source } => EhmaInput {
					data: EhmaData::Candles { candles, source },
					params: EhmaParams {
						period: Some(period),
						..Default::default()
					},
				},
				MaData::Slice(s) => EhmaInput {
					data: EhmaData::Slice(s),
					params: EhmaParams {
						period: Some(period),
						..Default::default()
					},
				},
			};
			let output = ehma(&input)?;
			Ok(output.values)
		}

		"frama" => {
			// Frama requires high/low data, not suitable for simple MA selector  
			return Err("frama requires high/low data, use the indicator directly".into());
		}

		"nama" => {
			let input = match data {
				MaData::Candles { candles, source } => NamaInput {
					data: NamaData::Candles { candles, source },
					params: NamaParams {
						period: Some(period),
						..Default::default()
					},
				},
				MaData::Slice(s) => NamaInput {
					data: NamaData::Slice(s),
					params: NamaParams {
						period: Some(period),
						..Default::default()
					},
				},
			};
			let output = nama(&input)?;
			Ok(output.values)
		}

		"sama" => {
			let input = match data {
				MaData::Candles { candles, source } => SamaInput {
					data: SamaData::Candles { candles, source },
					params: SamaParams {
						length: Some(period),
						..Default::default()
					},
				},
				MaData::Slice(s) => SamaInput {
					data: SamaData::Slice(s),
					params: SamaParams {
						length: Some(period),
						..Default::default()
					},
				},
			};
			let output = sama(&input)?;
			Ok(output.values)
		}

		"tradjema" => {
			// TradjemaData requires high/low/close data, not suitable for simple MA selector
			return Err("tradjema requires high/low/close data, use the indicator directly".into());
		}

		"uma" => {
			// UMA requires volume data, not suitable for simple MA selector
			return Err("uma requires volume data, use the indicator directly".into());
		}

		"volatility_adjusted_ma" => {
			let input = match data {
				MaData::Candles { candles, source } => VamaInput {
					data: VamaData::Candles { candles, source },
					params: VamaParams {
						base_period: Some(period),
						..Default::default()
					},
				},
				MaData::Slice(s) => VamaInput {
					data: VamaData::Slice(s),
					params: VamaParams {
						base_period: Some(period),
						..Default::default()
					},
				},
			};
			let output = vama(&input)?;
			Ok(output.values)
		}

		"volume_adjusted_ma" => {
			// VolumeAdjustedMa requires volume data, not suitable for simple MA selector
			return Err("volume_adjusted_ma requires volume data, use the indicator directly".into());
		}

		_ => {
			return Err(format!("Unknown moving average type: {}", ma_type).into());
		}
	}
}

#[inline]
pub fn ma_with_kernel<'a>(
	ma_type: &str,
	data: MaData<'a>,
	period: usize,
	kernel: Kernel,
) -> Result<Vec<f64>, Box<dyn Error>> {
	match ma_type.to_lowercase().as_str() {
		"sma" => {
			let input = match data {
				MaData::Candles { candles, source } => SmaInput {
					data: SmaData::Candles { candles, source },
					params: SmaParams { period: Some(period) },
				},
				MaData::Slice(slice) => SmaInput {
					data: SmaData::Slice(slice),
					params: SmaParams { period: Some(period) },
				},
			};
			let output = sma_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"alma" => {
			let input = match data {
				MaData::Candles { candles, source } => AlmaInput {
					data: AlmaData::Candles { candles, source },
					params: AlmaParams {
						period: Some(period),
						offset: None,
						sigma: None,
					},
				},
				MaData::Slice(slice) => AlmaInput {
					data: AlmaData::Slice(slice),
					params: AlmaParams {
						period: Some(period),
						offset: None,
						sigma: None,
					},
				},
			};
			let output = alma_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"cwma" => {
			let input = match data {
				MaData::Candles { candles, source } => CwmaInput {
					data: CwmaData::Candles { candles, source },
					params: CwmaParams { period: Some(period) },
				},
				MaData::Slice(slice) => CwmaInput {
					data: CwmaData::Slice(slice),
					params: CwmaParams { period: Some(period) },
				},
			};
			let output = cwma_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"dema" => {
			let input = match data {
				MaData::Candles { candles, source } => DemaInput {
					data: DemaData::Candles { candles, source },
					params: DemaParams { period: Some(period) },
				},
				MaData::Slice(slice) => DemaInput {
					data: DemaData::Slice(slice),
					params: DemaParams { period: Some(period) },
				},
			};
			let output = dema_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"edcf" => {
			let input = match data {
				MaData::Candles { candles, source } => EdcfInput {
					data: EdcfData::Candles { candles, source },
					params: EdcfParams { period: Some(period) },
				},
				MaData::Slice(slice) => EdcfInput {
					data: EdcfData::Slice(slice),
					params: EdcfParams { period: Some(period) },
				},
			};
			let output = edcf_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"ema" => {
			let input = match data {
				MaData::Candles { candles, source } => EmaInput {
					data: EmaData::Candles { candles, source },
					params: EmaParams { period: Some(period) },
				},
				MaData::Slice(slice) => EmaInput {
					data: EmaData::Slice(slice),
					params: EmaParams { period: Some(period) },
				},
			};
			let output = ema_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"epma" => {
			let input = match data {
				MaData::Candles { candles, source } => EpmaInput {
					data: EpmaData::Candles { candles, source },
					params: EpmaParams {
						period: Some(period),
						offset: None,
					},
				},
				MaData::Slice(slice) => EpmaInput {
					data: EpmaData::Slice(slice),
					params: EpmaParams {
						period: Some(period),
						offset: None,
					},
				},
			};
			let output = epma_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"fwma" => {
			let input = match data {
				MaData::Candles { candles, source } => FwmaInput {
					data: FwmaData::Candles { candles, source },
					params: FwmaParams { period: Some(period) },
				},
				MaData::Slice(slice) => FwmaInput {
					data: FwmaData::Slice(slice),
					params: FwmaParams { period: Some(period) },
				},
			};
			let output = fwma_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"gaussian" => {
			let input = match data {
				MaData::Candles { candles, source } => GaussianInput {
					data: GaussianData::Candles { candles, source },
					params: GaussianParams {
						period: Some(period),
						poles: None,
					},
				},
				MaData::Slice(slice) => GaussianInput {
					data: GaussianData::Slice(slice),
					params: GaussianParams {
						period: Some(period),
						poles: None,
					},
				},
			};
			let output = gaussian_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"highpass" => {
			let input = match data {
				MaData::Candles { candles, source } => HighPassInput {
					data: HighPassData::Candles { candles, source },
					params: HighPassParams { period: Some(period) },
				},
				MaData::Slice(slice) => HighPassInput {
					data: HighPassData::Slice(slice),
					params: HighPassParams { period: Some(period) },
				},
			};
			let output = highpass_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"highpass2" => {
			let input = match data {
				MaData::Candles { candles, source } => HighPass2Input {
					data: HighPass2Data::Candles { candles, source },
					params: HighPass2Params {
						period: Some(period),
						k: Some(0.707),
					},
				},
				MaData::Slice(slice) => HighPass2Input {
					data: HighPass2Data::Slice(slice),
					params: HighPass2Params {
						period: Some(period),
						k: Some(0.707),
					},
				},
			};
			let output = highpass_2_pole_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"hma" => {
			let input = match data {
				MaData::Candles { candles, source } => HmaInput {
					data: HmaData::Candles { candles, source },
					params: HmaParams { period: Some(period) },
				},
				MaData::Slice(slice) => HmaInput {
					data: HmaData::Slice(slice),
					params: HmaParams { period: Some(period) },
				},
			};
			let output = hma_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"ehlers_itrend" => {
			let input = match data {
				MaData::Candles { candles, source } => EhlersITrendInput {
					data: EhlersITrendData::Candles { candles, source },
					params: EhlersITrendParams {
						warmup_bars: Some(12),
						max_dc_period: Some(50),
					},
				},
				MaData::Slice(slice) => EhlersITrendInput {
					data: EhlersITrendData::Slice(slice),
					params: EhlersITrendParams {
						warmup_bars: Some(12),
						max_dc_period: Some(50),
					},
				},
			};
			let output = ehlers_itrend_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"hwma" => {
			let input = match data {
				MaData::Candles { candles, source } => HwmaInput {
					data: HwmaData::Candles { candles, source },
					params: HwmaParams {
						na: None,
						nb: None,
						nc: None,
					},
				},
				MaData::Slice(slice) => HwmaInput {
					data: HwmaData::Slice(slice),
					params: HwmaParams {
						na: None,
						nb: None,
						nc: None,
					},
				},
			};
			let output = hwma_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"jma" => {
			let input = match data {
				MaData::Candles { candles, source } => JmaInput {
					data: JmaData::Candles { candles, source },
					params: JmaParams {
						period: Some(period),
						phase: None,
						power: None,
					},
				},
				MaData::Slice(slice) => JmaInput {
					data: JmaData::Slice(slice),
					params: JmaParams {
						period: Some(period),
						phase: None,
						power: None,
					},
				},
			};
			let output = jma_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"jsa" => {
			let input = match data {
				MaData::Candles { candles, source } => JsaInput {
					data: JsaData::Candles { candles, source },
					params: JsaParams { period: Some(period) },
				},
				MaData::Slice(slice) => JsaInput {
					data: JsaData::Slice(slice),
					params: JsaParams { period: Some(period) },
				},
			};
			let output = jsa_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"kama" => {
			let input = match data {
				MaData::Candles { candles, source } => KamaInput {
					data: KamaData::Candles { candles, source },
					params: KamaParams { period: Some(period) },
				},
				MaData::Slice(slice) => KamaInput {
					data: KamaData::Slice(slice),
					params: KamaParams { period: Some(period) },
				},
			};
			let output = kama_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"linreg" => {
			let input = match data {
				MaData::Candles { candles, source } => LinRegInput {
					data: LinRegData::Candles { candles, source },
					params: LinRegParams { period: Some(period) },
				},
				MaData::Slice(slice) => LinRegInput {
					data: LinRegData::Slice(slice),
					params: LinRegParams { period: Some(period) },
				},
			};
			let output = linreg_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"maaq" => {
			let input = match data {
				MaData::Candles { candles, source } => MaaqInput {
					data: MaaqData::Candles { candles, source },
					params: MaaqParams {
						period: Some(period),
						fast_period: None,
						slow_period: None,
					},
				},
				MaData::Slice(slice) => MaaqInput {
					data: MaaqData::Slice(slice),
					params: MaaqParams {
						period: Some(period),
						fast_period: None,
						slow_period: None,
					},
				},
			};
			let output = maaq_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"mama" => {
			let input = match data {
				MaData::Candles { candles, source } => MamaInput::from_candles(candles, source, MamaParams::default()),
				MaData::Slice(slice) => MamaInput::from_slice(slice, MamaParams::default()),
			};
			let output = mama_with_kernel(&input, kernel)?;
			Ok(output.mama_values)
		}

		"mwdx" => {
			let input = match data {
				MaData::Candles { candles, source } => {
					MwdxInput::from_candles(candles, source, MwdxParams { factor: Some(0.2) })
				}
				MaData::Slice(slice) => MwdxInput::from_slice(slice, MwdxParams { factor: Some(0.2) }),
			};
			let output = mwdx_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"nma" => {
			let input = match data {
				MaData::Candles { candles, source } => NmaInput {
					data: NmaData::Candles { candles, source },
					params: NmaParams { period: Some(period) },
				},
				MaData::Slice(slice) => NmaInput {
					data: NmaData::Slice(slice),
					params: NmaParams { period: Some(period) },
				},
			};
			let output = nma_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"pwma" => {
			let input = match data {
				MaData::Candles { candles, source } => PwmaInput {
					data: PwmaData::Candles { candles, source },
					params: PwmaParams { period: Some(period) },
				},
				MaData::Slice(slice) => PwmaInput {
					data: PwmaData::Slice(slice),
					params: PwmaParams { period: Some(period) },
				},
			};
			let output = pwma_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"reflex" => {
			let input = match data {
				MaData::Candles { candles, source } => ReflexInput {
					data: ReflexData::Candles { candles, source },
					params: ReflexParams { period: Some(period) },
				},
				MaData::Slice(slice) => ReflexInput {
					data: ReflexData::Slice(slice),
					params: ReflexParams { period: Some(period) },
				},
			};
			let output = reflex_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"sinwma" => {
			let input = match data {
				MaData::Candles { candles, source } => SinWmaInput {
					data: SinWmaData::Candles { candles, source },
					params: SinWmaParams { period: Some(period) },
				},
				MaData::Slice(slice) => SinWmaInput {
					data: SinWmaData::Slice(slice),
					params: SinWmaParams { period: Some(period) },
				},
			};
			let output = sinwma_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"sqwma" => {
			let input = match data {
				MaData::Candles { candles, source } => SqwmaInput {
					data: SqwmaData::Candles { candles, source },
					params: SqwmaParams { period: Some(period) },
				},
				MaData::Slice(slice) => SqwmaInput {
					data: SqwmaData::Slice(slice),
					params: SqwmaParams { period: Some(period) },
				},
			};
			let output = sqwma_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"srwma" => {
			let input = match data {
				MaData::Candles { candles, source } => SrwmaInput {
					data: SrwmaData::Candles { candles, source },
					params: SrwmaParams { period: Some(period) },
				},
				MaData::Slice(slice) => SrwmaInput {
					data: SrwmaData::Slice(slice),
					params: SrwmaParams { period: Some(period) },
				},
			};
			let output = srwma_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"smma" => {
			let input = match data {
				MaData::Candles { candles, source } => SmmaInput {
					data: SmmaData::Candles { candles, source },
					params: SmmaParams { period: Some(period) },
				},
				MaData::Slice(slice) => SmmaInput {
					data: SmmaData::Slice(slice),
					params: SmmaParams { period: Some(period) },
				},
			};
			let output = smma_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"supersmoother" => {
			let input = match data {
				MaData::Candles { candles, source } => SuperSmootherInput {
					data: SuperSmootherData::Candles { candles, source },
					params: SuperSmootherParams { period: Some(period) },
				},
				MaData::Slice(slice) => SuperSmootherInput {
					data: SuperSmootherData::Slice(slice),
					params: SuperSmootherParams { period: Some(period) },
				},
			};
			let output = supersmoother_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"supersmoother_3_pole" => {
			let input = match data {
				MaData::Candles { candles, source } => SuperSmoother3PoleInput {
					data: SuperSmoother3PoleData::Candles { candles, source },
					params: SuperSmoother3PoleParams { period: Some(period) },
				},
				MaData::Slice(slice) => SuperSmoother3PoleInput {
					data: SuperSmoother3PoleData::Slice(slice),
					params: SuperSmoother3PoleParams { period: Some(period) },
				},
			};
			let output = supersmoother_3_pole_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"swma" => {
			let input = match data {
				MaData::Candles { candles, source } => SwmaInput {
					data: SwmaData::Candles { candles, source },
					params: SwmaParams { period: Some(period) },
				},
				MaData::Slice(slice) => SwmaInput {
					data: SwmaData::Slice(slice),
					params: SwmaParams { period: Some(period) },
				},
			};
			let output = swma_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"tema" => {
			let input = match data {
				MaData::Candles { candles, source } => TemaInput {
					data: TemaData::Candles { candles, source },
					params: TemaParams { period: Some(period) },
				},
				MaData::Slice(slice) => TemaInput {
					data: TemaData::Slice(slice),
					params: TemaParams { period: Some(period) },
				},
			};
			let output = tema_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"tilson" => {
			let input = match data {
				MaData::Candles { candles, source } => TilsonInput {
					data: TilsonData::Candles { candles, source },
					params: TilsonParams {
						period: Some(period),
						volume_factor: None,
					},
				},
				MaData::Slice(slice) => TilsonInput {
					data: TilsonData::Slice(slice),
					params: TilsonParams {
						period: Some(period),
						volume_factor: None,
					},
				},
			};
			let output = tilson_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"trendflex" => {
			let input = match data {
				MaData::Candles { candles, source } => TrendFlexInput {
					data: TrendFlexData::Candles { candles, source },
					params: TrendFlexParams { period: Some(period) },
				},
				MaData::Slice(slice) => TrendFlexInput {
					data: TrendFlexData::Slice(slice),
					params: TrendFlexParams { period: Some(period) },
				},
			};
			let output = trendflex_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"trima" => {
			let input = match data {
				MaData::Candles { candles, source } => TrimaInput {
					data: TrimaData::Candles { candles, source },
					params: TrimaParams { period: Some(period) },
				},
				MaData::Slice(slice) => TrimaInput {
					data: TrimaData::Slice(slice),
					params: TrimaParams { period: Some(period) },
				},
			};
			let output = trima_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"wilders" => {
			let input = match data {
				MaData::Candles { candles, source } => WildersInput {
					data: WildersData::Candles { candles, source },
					params: WildersParams { period: Some(period) },
				},
				MaData::Slice(slice) => WildersInput {
					data: WildersData::Slice(slice),
					params: WildersParams { period: Some(period) },
				},
			};
			let output = wilders_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"wma" => {
			let input = match data {
				MaData::Candles { candles, source } => WmaInput {
					data: WmaData::Candles { candles, source },
					params: WmaParams { period: Some(period) },
				},
				MaData::Slice(slice) => WmaInput {
					data: WmaData::Slice(slice),
					params: WmaParams { period: Some(period) },
				},
			};
			let output = wma_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"vpwma" => {
			let input = match data {
				MaData::Candles { candles, source } => VpwmaInput {
					data: VpwmaData::Candles { candles, source },
					params: VpwmaParams {
						period: Some(period),
						power: Some(0.382),
					},
				},
				MaData::Slice(_) => {
					return Err("VPWMA requires candle data with volume".into());
				}
			};
			let output = vpwma_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"vwap" => {
			let input = match data {
				MaData::Candles { candles, .. } => VwapInput {
					data: VwapData::Candles {
						candles,
						source: "hlc3",
					},
					params: VwapParams::default(),
				},
				MaData::Slice(_) => {
					return Err("VWAP requires candle data with high, low, close, and volume".into());
				}
			};
			let output = vwap_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"vwma" => {
			let input = match data {
				MaData::Candles { candles, source } => VwmaInput {
					data: VwmaData::Candles { candles, source },
					params: VwmaParams { period: Some(period) },
				},
				MaData::Slice(_) => {
					return Err("VWMA requires candle data with volume".into());
				}
			};
			let output = vwma_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"zlema" => {
			let input = match data {
				MaData::Candles { candles, source } => ZlemaInput {
					data: ZlemaData::Candles { candles, source },
					params: ZlemaParams { period: Some(period) },
				},
				MaData::Slice(slice) => ZlemaInput {
					data: ZlemaData::Slice(slice),
					params: ZlemaParams { period: Some(period) },
				},
			};
			let output = zlema_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"buff_averages" => {
			// BuffAverages requires volume data and returns dual outputs, not suitable for simple MA selector
			return Err("buff_averages requires volume data, use the indicator directly".into());
		}

		"dma" => {
			let input = match data {
				MaData::Candles { candles, source } => DmaInput {
					data: DmaData::Candles { candles, source },
					params: DmaParams {
						ema_length: Some(period),
						..Default::default()
					},
				},
				MaData::Slice(slice) => DmaInput {
					data: DmaData::Slice(slice),
					params: DmaParams {
						ema_length: Some(period),
						..Default::default()
					},
				},
			};
			let output = dma_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"ehlers_ecema" => {
			let input = match data {
				MaData::Candles { candles, source } => EhlersEcemaInput {
					data: EhlersEcemaData::Candles { candles, source },
					params: EhlersEcemaParams {
						length: Some(period),
						..Default::default()
					},
				},
				MaData::Slice(slice) => EhlersEcemaInput {
					data: EhlersEcemaData::Slice(slice),
					params: EhlersEcemaParams {
						length: Some(period),
						..Default::default()
					},
				},
			};
			let output = ehlers_ecema_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"ehlers_kama" => {
			let input = match data {
				MaData::Candles { candles, source } => EhlersKamaInput {
					data: EhlersKamaData::Candles { candles, source },
					params: EhlersKamaParams {
						period: Some(period),
						..Default::default()
					},
				},
				MaData::Slice(slice) => EhlersKamaInput {
					data: EhlersKamaData::Slice(slice),
					params: EhlersKamaParams {
						period: Some(period),
						..Default::default()
					},
				},
			};
			let output = ehlers_kama_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"ehlers_pma" => {
			// EhlersPma returns dual outputs (predict/trigger), not suitable for simple MA selector
			return Err("ehlers_pma returns dual outputs, use the indicator directly".into());
		}

		"ehma" => {
			let input = match data {
				MaData::Candles { candles, source } => EhmaInput {
					data: EhmaData::Candles { candles, source },
					params: EhmaParams {
						period: Some(period),
						..Default::default()
					},
				},
				MaData::Slice(slice) => EhmaInput {
					data: EhmaData::Slice(slice),
					params: EhmaParams {
						period: Some(period),
						..Default::default()
					},
				},
			};
			let output = ehma_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"frama" => {
			// Frama requires high/low data, not suitable for simple MA selector  
			return Err("frama requires high/low data, use the indicator directly".into());
		}

		"nama" => {
			let input = match data {
				MaData::Candles { candles, source } => NamaInput {
					data: NamaData::Candles { candles, source },
					params: NamaParams {
						period: Some(period),
						..Default::default()
					},
				},
				MaData::Slice(slice) => NamaInput {
					data: NamaData::Slice(slice),
					params: NamaParams {
						period: Some(period),
						..Default::default()
					},
				},
			};
			let output = nama_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"sama" => {
			let input = match data {
				MaData::Candles { candles, source } => SamaInput {
					data: SamaData::Candles { candles, source },
					params: SamaParams {
						length: Some(period),
						..Default::default()
					},
				},
				MaData::Slice(slice) => SamaInput {
					data: SamaData::Slice(slice),
					params: SamaParams {
						length: Some(period),
						..Default::default()
					},
				},
			};
			let output = sama_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"tradjema" => {
			// TradjemaData requires high/low/close data, not suitable for simple MA selector
			return Err("tradjema requires high/low/close data, use the indicator directly".into());
		}

		"uma" => {
			// UMA requires volume data, not suitable for simple MA selector
			return Err("uma requires volume data, use the indicator directly".into());
		}

		"volatility_adjusted_ma" => {
			let input = match data {
				MaData::Candles { candles, source } => VamaInput {
					data: VamaData::Candles { candles, source },
					params: VamaParams {
						base_period: Some(period),
						..Default::default()
					},
				},
				MaData::Slice(slice) => VamaInput {
					data: VamaData::Slice(slice),
					params: VamaParams {
						base_period: Some(period),
						..Default::default()
					},
				},
			};
			let output = vama_with_kernel(&input, kernel)?;
			Ok(output.values)
		}

		"volume_adjusted_ma" => {
			// VolumeAdjustedMa requires volume data, not suitable for simple MA selector
			return Err("volume_adjusted_ma requires volume data, use the indicator directly".into());
		}

		_ => {
			return Err(format!("Unknown moving average type: {}", ma_type).into());
		}
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "ma")]
#[pyo3(signature = (data, ma_type, period, kernel=None))]
pub fn ma_py<'py>(
	py: Python<'py>,
	data: PyReadonlyArray1<'py, f64>,
	ma_type: &str,
	period: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?;

	// Get Vec<f64> from Rust function
	let result_vec: Vec<f64> = py
		.allow_threads(|| -> Result<Vec<f64>, Box<dyn Error + Send + Sync>> {
			// Try the requested MA type first
			match ma_with_kernel(ma_type, MaData::Slice(slice_in), period, kern) {
				Ok(result) => Ok(result),
				Err(e) => {
					// If it's an unknown MA type error, default to SMA
					if e.to_string().contains("Unknown moving average type") {
						ma_with_kernel("sma", MaData::Slice(slice_in), period, kern).map_err(
							|e| -> Box<dyn Error + Send + Sync> {
								Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
							},
						)
					} else {
						// For other errors, propagate them
						Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())) as Box<dyn Error + Send + Sync>)
					}
				}
			}
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// Zero-copy transfer to NumPy
	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "ma")]
pub fn ma_js(data: &[f64], ma_type: &str, period: usize) -> Result<Vec<f64>, JsValue> {
	// Try the requested MA type first
	match ma(ma_type, MaData::Slice(data), period) {
		Ok(result) => Ok(result),
		Err(e) => {
			// If it's an unknown MA type error, default to SMA
			if e.to_string().contains("Unknown moving average type") {
				ma("sma", MaData::Slice(data), period).map_err(|e| JsValue::from_str(&e.to_string()))
			} else {
				// For other errors, propagate them
				Err(JsValue::from_str(&e.to_string()))
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::utilities::data_loader::read_candles_from_csv;

	#[test]
	fn test_all_ma_variants() {
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
		];

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

		for &ma_type in &ma_types {
			let period = 80;
			let candles_result = ma(
				ma_type,
				MaData::Candles {
					candles: &candles,
					source: "close",
				},
				period,
			)
			.unwrap_or_else(|err| panic!("`ma({})` failed with error: {}", ma_type, err));

			assert_eq!(
				candles_result.len(),
				candles.close.len(),
				"MA output length for '{}' mismatch",
				ma_type
			);

			// Check candles_result for NaN values after warmup
			// MAMA has a warmup of 10, most others vary
			let skip_amount = if ma_type == "mama" { 10 } else { 960 };
			for (i, &value) in candles_result.iter().enumerate().skip(skip_amount) {
				assert!(!value.is_nan(), "MA result for '{}' at index {} is NaN", ma_type, i);
			}

			// Skip the double-processing test for MAMA since it's an adaptive algorithm
			// that doesn't handle NaN inputs well and running MAMA on MAMA output
			// isn't a meaningful operation
			if ma_type != "mama" {
				let slice_result = ma(ma_type, MaData::Slice(&candles_result), 60)
					.unwrap_or_else(|err| panic!("`ma({})` failed with error: {}", ma_type, err));

				assert_eq!(
					slice_result.len(),
					candles.close.len(),
					"MA output length for '{}' mismatch",
					ma_type
				);

				for (i, &value) in slice_result.iter().enumerate().skip(960) {
					assert!(!value.is_nan(), "MA result for '{}' at index {} is NaN", ma_type, i);
				}
			}
		}
	}
}
