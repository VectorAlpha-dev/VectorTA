use crate::indicators::alma::{alma, AlmaData, AlmaInput, AlmaParams};
use crate::indicators::cwma::{cwma, CwmaData, CwmaInput, CwmaParams};
use crate::indicators::dema::{dema, DemaData, DemaInput, DemaParams};
use crate::indicators::edcf::{edcf, EdcfData, EdcfInput, EdcfParams};
use crate::indicators::ehlers_itrend::{
    ehlers_itrend, EhlersITrendData, EhlersITrendInput, EhlersITrendParams,
};
use crate::indicators::ema::{ema, EmaData, EmaInput, EmaParams};
use crate::indicators::epma::{epma, EpmaData, EpmaInput, EpmaParams};
use crate::indicators::fwma::{fwma, FwmaData, FwmaInput, FwmaParams};
use crate::indicators::gaussian::{gaussian, GaussianData, GaussianInput, GaussianParams};
use crate::indicators::highpass::{highpass, HighPassData, HighPassInput, HighPassParams};
use crate::indicators::highpass_2_pole::{
    highpass_2_pole, HighPass2Data, HighPass2Input, HighPass2Params,
};
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
use crate::indicators::supersmoother::{
    supersmoother, SuperSmootherData, SuperSmootherInput, SuperSmootherParams,
};
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
use crate::utilities::data_loader::Candles;
use std::error::Error;

#[derive(Debug, Clone)]
pub enum MaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

pub fn ma<'a>(ma_type: &str, data: MaData<'a>, period: usize) -> Result<Vec<f64>, Box<dyn Error>> {
    match ma_type.to_lowercase().as_str() {
        "sma" => {
            let input = match data {
                MaData::Candles { candles, source } => SmaInput {
                    data: SmaData::Candles { candles, source },
                    params: SmaParams {
                        period: Some(period),
                    },
                },
                MaData::Slice(slice) => SmaInput {
                    data: SmaData::Slice(slice),
                    params: SmaParams {
                        period: Some(period),
                    },
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
                    params: CwmaParams {
                        period: Some(period),
                    },
                },
                MaData::Slice(slice) => CwmaInput {
                    data: CwmaData::Slice(slice),
                    params: CwmaParams {
                        period: Some(period),
                    },
                },
            };
            let output = cwma(&input)?;
            Ok(output.values)
        }

        "dema" => {
            let input = match data {
                MaData::Candles { candles, source } => DemaInput {
                    data: DemaData::Candles { candles, source },
                    params: DemaParams {
                        period: Some(period),
                    },
                },
                MaData::Slice(slice) => DemaInput {
                    data: DemaData::Slice(slice),
                    params: DemaParams {
                        period: Some(period),
                    },
                },
            };
            let output = dema(&input)?;
            Ok(output.values)
        }

        "edcf" => {
            let input = match data {
                MaData::Candles { candles, source } => EdcfInput {
                    data: EdcfData::Candles { candles, source },
                    params: EdcfParams {
                        period: Some(period),
                    },
                },
                MaData::Slice(slice) => EdcfInput {
                    data: EdcfData::Slice(slice),
                    params: EdcfParams {
                        period: Some(period),
                    },
                },
            };
            let output = edcf(&input)?;
            Ok(output.values)
        }

        "ema" => {
            let input = match data {
                MaData::Candles { candles, source } => EmaInput {
                    data: EmaData::Candles { candles, source },
                    params: EmaParams {
                        period: Some(period),
                    },
                },
                MaData::Slice(slice) => EmaInput {
                    data: EmaData::Slice(slice),
                    params: EmaParams {
                        period: Some(period),
                    },
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
                    params: FwmaParams {
                        period: Some(period),
                    },
                },
                MaData::Slice(slice) => FwmaInput {
                    data: FwmaData::Slice(slice),
                    params: FwmaParams {
                        period: Some(period),
                    },
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
                    params: HighPassParams {
                        period: Some(period),
                    },
                },
                MaData::Slice(slice) => HighPassInput {
                    data: HighPassData::Slice(slice),
                    params: HighPassParams {
                        period: Some(period),
                    },
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
                        k: Some(0.5),
                    },
                },
                MaData::Slice(slice) => HighPass2Input {
                    data: HighPass2Data::Slice(slice),
                    params: HighPass2Params {
                        period: Some(period),
                        k: Some(0.5),
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
                    params: HmaParams {
                        period: Some(period),
                    },
                },
                MaData::Slice(slice) => HmaInput {
                    data: HmaData::Slice(slice),
                    params: HmaParams {
                        period: Some(period),
                    },
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
                    params: JsaParams {
                        period: Some(period),
                    },
                },
                MaData::Slice(slice) => JsaInput {
                    data: JsaData::Slice(slice),
                    params: JsaParams {
                        period: Some(period),
                    },
                },
            };
            let output = jsa(&input)?;
            Ok(output.values)
        }

        "kama" => {
            let input = match data {
                MaData::Candles { candles, source } => KamaInput {
                    data: KamaData::Candles { candles, source },
                    params: KamaParams {
                        period: Some(period),
                    },
                },
                MaData::Slice(slice) => KamaInput {
                    data: KamaData::Slice(slice),
                    params: KamaParams {
                        period: Some(period),
                    },
                },
            };
            let output = kama(&input)?;
            Ok(output.values)
        }

        "linreg" => {
            let input = match data {
                MaData::Candles { candles, source } => LinRegInput {
                    data: LinRegData::Candles { candles, source },
                    params: LinRegParams {
                        period: Some(period),
                    },
                },
                MaData::Slice(s) => LinRegInput {
                    data: LinRegData::Slice(s),
                    params: LinRegParams {
                        period: Some(period),
                    },
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
            let _fast_limit = (10.0 / period as f64).clamp(0.0, 1.0);
            let input = match data {
                MaData::Candles { candles, source } => MamaInput {
                    data: MamaData::Candles { candles, source },
                    params: MamaParams {
                        fast_limit: Some(_fast_limit),
                        slow_limit: None,
                    },
                },
                MaData::Slice(s) => MamaInput {
                    data: MamaData::Slice(s),
                    params: MamaParams {
                        fast_limit: Some(_fast_limit),
                        slow_limit: None,
                    },
                },
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
                    params: NmaParams {
                        period: Some(period),
                    },
                },
                MaData::Slice(s) => NmaInput {
                    data: NmaData::Slice(s),
                    params: NmaParams {
                        period: Some(period),
                    },
                },
            };
            let output = nma(&input)?;
            Ok(output.values)
        }

        "pwma" => {
            let input = match data {
                MaData::Candles { candles, source } => PwmaInput {
                    data: PwmaData::Candles { candles, source },
                    params: PwmaParams {
                        period: Some(period),
                    },
                },
                MaData::Slice(s) => PwmaInput {
                    data: PwmaData::Slice(s),
                    params: PwmaParams {
                        period: Some(period),
                    },
                },
            };
            let output = pwma(&input)?;
            Ok(output.values)
        }

        "reflex" => {
            let input = match data {
                MaData::Candles { candles, source } => ReflexInput {
                    data: ReflexData::Candles { candles, source },
                    params: ReflexParams {
                        period: Some(period),
                    },
                },
                MaData::Slice(s) => ReflexInput {
                    data: ReflexData::Slice(s),
                    params: ReflexParams {
                        period: Some(period),
                    },
                },
            };
            let output = reflex(&input)?;
            Ok(output.values)
        }

        "sinwma" => {
            let input = match data {
                MaData::Candles { candles, source } => SinWmaInput {
                    data: SinWmaData::Candles { candles, source },
                    params: SinWmaParams {
                        period: Some(period),
                    },
                },
                MaData::Slice(s) => SinWmaInput {
                    data: SinWmaData::Slice(s),
                    params: SinWmaParams {
                        period: Some(period),
                    },
                },
            };
            let output = sinwma(&input)?;
            Ok(output.values)
        }

        "smma" => {
            let input = match data {
                MaData::Candles { candles, source } => SmmaInput {
                    data: SmmaData::Candles { candles, source },
                    params: SmmaParams {
                        period: Some(period),
                    },
                },
                MaData::Slice(s) => SmmaInput {
                    data: SmmaData::Slice(s),
                    params: SmmaParams {
                        period: Some(period),
                    },
                },
            };
            let output = smma(&input)?;
            Ok(output.values)
        }

        "sqwma" => {
            let input = match data {
                MaData::Candles { candles, source } => SqwmaInput {
                    data: SqwmaData::Candles { candles, source },
                    params: SqwmaParams {
                        period: Some(period),
                    },
                },
                MaData::Slice(s) => SqwmaInput {
                    data: SqwmaData::Slice(s),
                    params: SqwmaParams {
                        period: Some(period),
                    },
                },
            };
            let output = sqwma(&input)?;
            Ok(output.values)
        }

        "srwma" => {
            let input = match data {
                MaData::Candles { candles, source } => SrwmaInput {
                    data: SrwmaData::Candles { candles, source },
                    params: SrwmaParams {
                        period: Some(period),
                    },
                },
                MaData::Slice(s) => SrwmaInput {
                    data: SrwmaData::Slice(s),
                    params: SrwmaParams {
                        period: Some(period),
                    },
                },
            };
            let output = srwma(&input)?;
            Ok(output.values)
        }

        "supersmoother" => {
            let input = match data {
                MaData::Candles { candles, source } => SuperSmootherInput {
                    data: SuperSmootherData::Candles { candles, source },
                    params: SuperSmootherParams {
                        period: Some(period),
                    },
                },
                MaData::Slice(s) => SuperSmootherInput {
                    data: SuperSmootherData::Slice(s),
                    params: SuperSmootherParams {
                        period: Some(period),
                    },
                },
            };
            let output = supersmoother(&input)?;
            Ok(output.values)
        }

        "supersmoother_3_pole" => {
            let input = match data {
                MaData::Candles { candles, source } => SuperSmoother3PoleInput {
                    data: SuperSmoother3PoleData::Candles { candles, source },
                    params: SuperSmoother3PoleParams {
                        period: Some(period),
                    },
                },
                MaData::Slice(s) => SuperSmoother3PoleInput {
                    data: SuperSmoother3PoleData::Slice(s),
                    params: SuperSmoother3PoleParams {
                        period: Some(period),
                    },
                },
            };
            let output = supersmoother_3_pole(&input)?;
            Ok(output.values)
        }

        "swma" => {
            let input = match data {
                MaData::Candles { candles, source } => SwmaInput {
                    data: SwmaData::Candles { candles, source },
                    params: SwmaParams {
                        period: Some(period),
                    },
                },
                MaData::Slice(s) => SwmaInput {
                    data: SwmaData::Slice(s),
                    params: SwmaParams {
                        period: Some(period),
                    },
                },
            };
            let output = swma(&input)?;
            Ok(output.values)
        }

        "tema" => {
            let input = match data {
                MaData::Candles { candles, source } => TemaInput {
                    data: TemaData::Candles { candles, source },
                    params: TemaParams {
                        period: Some(period),
                    },
                },
                MaData::Slice(s) => TemaInput {
                    data: TemaData::Slice(s),
                    params: TemaParams {
                        period: Some(period),
                    },
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
                    params: TrendFlexParams {
                        period: Some(period),
                    },
                },
                MaData::Slice(s) => TrendFlexInput {
                    data: TrendFlexData::Slice(s),
                    params: TrendFlexParams {
                        period: Some(period),
                    },
                },
            };
            let output = trendflex(&input)?;
            Ok(output.values)
        }

        "trima" => {
            let input = match data {
                MaData::Candles { candles, source } => TrimaInput {
                    data: TrimaData::Candles { candles, source },
                    params: TrimaParams {
                        period: Some(period),
                    },
                },
                MaData::Slice(s) => TrimaInput {
                    data: TrimaData::Slice(s),
                    params: TrimaParams {
                        period: Some(period),
                    },
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
                // Default case
                eprintln!("Unknown data type for 'vpwma'. Defaulting to 'sma'.");

                let input = match data {
                    MaData::Candles { candles, source } => SmaInput::from_candles(
                        candles,
                        source,
                        SmaParams {
                            period: Some(period),
                        },
                    ),
                    MaData::Slice(slice) => SmaInput::from_slice(
                        slice,
                        SmaParams {
                            period: Some(period),
                        },
                    ),
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
                // Default case
                eprintln!("Unknown data type for 'vwap'. Defaulting to 'sma'.");

                let input = match data {
                    MaData::Candles { candles, source } => SmaInput::from_candles(
                        candles,
                        source,
                        SmaParams {
                            period: Some(period),
                        },
                    ),
                    MaData::Slice(slice) => SmaInput::from_slice(
                        slice,
                        SmaParams {
                            period: Some(period),
                        },
                    ),
                };
                let output = sma(&input)?;
                Ok(output.values)
            }
        }
        "vwma" => {
            if let MaData::Candles { candles, source } = data {
                let input = VwmaInput {
                    data: VwmaData::Candles { candles, source },
                    params: VwmaParams {
                        period: Some(period),
                    },
                };
                let output = vwma(&input)?;
                Ok(output.values)
            } else {
                // Default case
                eprintln!("Unknown data type for 'vpwma'. Defaulting to 'sma'.");

                let input = match data {
                    MaData::Candles { candles, source } => SmaInput::from_candles(
                        candles,
                        source,
                        SmaParams {
                            period: Some(period),
                        },
                    ),
                    MaData::Slice(slice) => SmaInput::from_slice(
                        slice,
                        SmaParams {
                            period: Some(period),
                        },
                    ),
                };
                let output = sma(&input)?;
                Ok(output.values)
            }
        }

        "wilders" => {
            let input = match data {
                MaData::Candles { candles, source } => WildersInput {
                    data: WildersData::Candles { candles, source },
                    params: WildersParams {
                        period: Some(period),
                    },
                },
                MaData::Slice(s) => WildersInput {
                    data: WildersData::Slice(s),
                    params: WildersParams {
                        period: Some(period),
                    },
                },
            };
            let output = wilders(&input)?;
            Ok(output.values)
        }

        "wma" => {
            let input = match data {
                MaData::Candles { candles, source } => WmaInput {
                    data: WmaData::Candles { candles, source },
                    params: WmaParams {
                        period: Some(period),
                    },
                },
                MaData::Slice(s) => WmaInput {
                    data: WmaData::Slice(s),
                    params: WmaParams {
                        period: Some(period),
                    },
                },
            };
            let output = wma(&input)?;
            Ok(output.values)
        }

        "zlema" => {
            let input = match data {
                MaData::Candles { candles, source } => ZlemaInput {
                    data: ZlemaData::Candles { candles, source },
                    params: ZlemaParams {
                        period: Some(period),
                    },
                },
                MaData::Slice(s) => ZlemaInput {
                    data: ZlemaData::Slice(s),
                    params: ZlemaParams {
                        period: Some(period),
                    },
                },
            };
            let output = zlema(&input)?;
            Ok(output.values)
        }

        _ => {
            eprintln!("Unknown indicator '{ma_type}'. Defaulting to 'sma'.");

            let input = match data {
                MaData::Candles { candles, source } => SmaInput::from_candles(
                    candles,
                    source,
                    SmaParams {
                        period: Some(period),
                    },
                ),
                MaData::Slice(slice) => SmaInput::from_slice(
                    slice,
                    SmaParams {
                        period: Some(period),
                    },
                ),
            };
            let output = sma(&input)?;
            Ok(output.values)
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
            "kama",
            "linreg",
            "maaq",
            "mama",
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
        ];

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        for &ma_type in &ma_types {
            let period = 14;
            let candles_result = ma(
                ma_type,
                MaData::Candles {
                    candles: &candles,
                    source: "close",
                },
                period,
            )
            .unwrap_or_else(|err| panic!("`ma({})` failed with error: {}", ma_type, err));

            let slice_result = ma(ma_type, MaData::Slice(&candles_result), period)
                .unwrap_or_else(|err| panic!("`ma({})` failed with error: {}", ma_type, err));

            assert_eq!(
                candles_result.len(),
                candles.close.len(),
                "MA output length for '{}' mismatch",
                ma_type
            );

            for (i, &value) in candles_result.iter().enumerate().skip(240) {
                assert!(
                    !value.is_nan(),
                    "MA result for '{}' at index {} is NaN",
                    ma_type,
                    i
                );
            }

            assert_eq!(
                slice_result.len(),
                candles.close.len(),
                "MA output length for '{}' mismatch",
                ma_type
            );

            for (i, &value) in slice_result.iter().enumerate().skip(240) {
                assert!(
                    !value.is_nan(),
                    "MA result for '{}' at index {} is NaN",
                    ma_type,
                    i
                );
            }
        }
    }
}
