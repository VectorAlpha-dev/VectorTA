use my_project::indicators::acosc::{acosc, AcoscData, AcoscInput, AcoscParams};
use my_project::indicators::ad::{ad, AdData, AdInput, AdParams};
use my_project::indicators::adosc::{adosc, AdoscData, AdoscInput, AdoscParams};
use my_project::indicators::adx::{adx, AdxData, AdxInput, AdxParams};
use my_project::indicators::adxr::{adxr, AdxrData, AdxrInput, AdxrParams};
use my_project::indicators::alligator::{alligator, AlligatorInput, AlligatorParams};
use my_project::indicators::ao::{ao, AoData, AoInput, AoParams};
use my_project::indicators::apo::{apo, ApoInput, ApoParams};
use my_project::indicators::aroon::{aroon, AroonData, AroonInput, AroonParams};
use my_project::indicators::aroonosc::{aroon_osc, AroonOscData, AroonOscInput, AroonOscParams};
use my_project::indicators::atr::{atr, AtrData, AtrInput, AtrParams};
use my_project::indicators::bandpass::{bandpass, BandPassInput, BandPassParams};
use my_project::indicators::bollinger_bands::{
    bollinger_bands, BollingerBandsInput, BollingerBandsParams,
};
use my_project::indicators::bollinger_bands_width::{
    bollinger_bands_width, BollingerBandsWidthInput, BollingerBandsWidthParams,
};
use my_project::indicators::bop::{bop, BopInput, BopParams};
use my_project::indicators::cci::{cci, CciInput, CciParams};
use my_project::indicators::cfo::{cfo, CfoInput, CfoParams};
use my_project::indicators::cg::{cg, CgInput, CgParams};
use my_project::indicators::chande::{chande, ChandeData, ChandeInput, ChandeParams};
use my_project::indicators::chop::{chop, ChopData, ChopInput, ChopParams};
use my_project::indicators::cmo::{cmo, CmoInput, CmoParams};
use my_project::indicators::correl_hl::{correl_hl, CorrelHlData, CorrelHlInput, CorrelHlParams};
use my_project::indicators::cvi::{cvi, CviInput, CviParams};
/// Binary to generate reference outputs for indicator testing
/// This is used by Python and WASM tests to verify their outputs match Rust
use my_project::indicators::damiani_volatmeter::{
    damiani_volatmeter, DamianiVolatmeterInput, DamianiVolatmeterParams,
};
use my_project::indicators::decycler::{decycler, DecyclerInput, DecyclerParams};
use my_project::indicators::deviation::{deviation, DeviationInput, DeviationParams};
use my_project::indicators::devstop::{devstop, DevStopData, DevStopInput, DevStopParams};
use my_project::indicators::di::{di, DiData, DiInput, DiParams};
use my_project::indicators::dpo::{dpo, DpoInput, DpoParams};
use my_project::indicators::emv::{emv, EmvInput};
use my_project::indicators::er::{er, ErInput, ErParams};
use my_project::indicators::eri::{eri, EriData, EriInput, EriParams};
use my_project::indicators::fisher::{fisher, FisherInput, FisherParams};
use my_project::indicators::kst::{kst, KstInput, KstParams};
use my_project::indicators::kurtosis::{kurtosis, KurtosisInput, KurtosisParams};
use my_project::indicators::linearreg_intercept::{
    linearreg_intercept, LinearRegInterceptInput, LinearRegInterceptParams,
};
use my_project::indicators::macz::{macz, MaczInput, MaczParams};
use my_project::indicators::marketefi::{
    marketefi, MarketefiData, MarketefiInput, MarketefiParams,
};
use my_project::indicators::mass::{mass, MassInput, MassParams};
use my_project::indicators::mfi::{mfi, MfiData, MfiInput, MfiParams};
use my_project::indicators::midpoint::{midpoint, MidpointInput, MidpointParams};
use my_project::indicators::midprice::{midprice, MidpriceInput, MidpriceParams};
use my_project::indicators::moving_averages::alma::{alma, AlmaInput, AlmaParams};
use my_project::indicators::moving_averages::cwma::{cwma, CwmaInput, CwmaParams};
use my_project::indicators::moving_averages::dema::{dema, DemaInput, DemaParams};
use my_project::indicators::moving_averages::edcf::{edcf, EdcfInput, EdcfParams};
use my_project::indicators::moving_averages::ehlers_ecema::{
    ehlers_ecema, EhlersEcemaInput, EhlersEcemaParams,
};
use my_project::indicators::moving_averages::ehlers_itrend::{
    ehlers_itrend, EhlersITrendInput, EhlersITrendParams,
};
use my_project::indicators::moving_averages::ema::{ema, EmaInput, EmaParams};
use my_project::indicators::moving_averages::epma::{epma, EpmaInput, EpmaParams};
use my_project::indicators::moving_averages::frama::{frama, FramaInput, FramaParams};
use my_project::indicators::moving_averages::fwma::{fwma, FwmaInput, FwmaParams};
use my_project::indicators::moving_averages::gaussian::{gaussian, GaussianInput, GaussianParams};
use my_project::indicators::moving_averages::highpass::{highpass, HighPassInput, HighPassParams};
use my_project::indicators::moving_averages::highpass_2_pole::{
    highpass_2_pole, HighPass2Input, HighPass2Params,
};
use my_project::indicators::moving_averages::hma::{hma, HmaInput, HmaParams};
use my_project::indicators::moving_averages::hwma::{hwma, HwmaInput, HwmaParams};
use my_project::indicators::moving_averages::jma::{jma, JmaInput, JmaParams};
use my_project::indicators::moving_averages::jsa::{jsa, JsaInput, JsaParams};
use my_project::indicators::moving_averages::kama::{kama, KamaInput, KamaParams};
use my_project::indicators::moving_averages::linreg::{linreg, LinRegInput, LinRegParams};
use my_project::indicators::moving_averages::volatility_adjusted_ma::{
    vama, VamaInput, VamaParams,
};
use my_project::indicators::moving_averages::volume_adjusted_ma::{
    VolumeAdjustedMa as volu_ma,
    VolumeAdjustedMaInput as VoluMaInput,
    VolumeAdjustedMaParams as VoluMaParams,
};
use my_project::indicators::moving_averages::maaq::{maaq, MaaqInput, MaaqParams};
use my_project::indicators::moving_averages::mama::{mama, MamaInput, MamaParams};
use my_project::indicators::moving_averages::mwdx::{mwdx, MwdxInput, MwdxParams};
use my_project::indicators::moving_averages::nma::{nma, NmaInput, NmaParams};
use my_project::indicators::moving_averages::pwma::{pwma, PwmaInput, PwmaParams};
use my_project::indicators::moving_averages::reflex::{reflex, ReflexInput, ReflexParams};
use my_project::indicators::moving_averages::sama::{sama, SamaInput, SamaParams};
use my_project::indicators::moving_averages::sinwma::{sinwma, SinWmaInput, SinWmaParams};
use my_project::indicators::moving_averages::sma::{sma, SmaInput, SmaParams};
use my_project::indicators::moving_averages::smma::{smma, SmmaInput, SmmaParams};
use my_project::indicators::moving_averages::sqwma::{sqwma, SqwmaInput, SqwmaParams};
use my_project::indicators::moving_averages::srwma::{srwma, SrwmaInput, SrwmaParams};
use my_project::indicators::moving_averages::supersmoother::{
    supersmoother, SuperSmootherInput, SuperSmootherParams,
};
use my_project::indicators::moving_averages::supersmoother_3_pole::{
    supersmoother_3_pole, SuperSmoother3PoleInput, SuperSmoother3PoleParams,
};
use my_project::indicators::moving_averages::swma::{swma, SwmaInput, SwmaParams};
use my_project::indicators::moving_averages::tema::{tema, TemaInput, TemaParams};
use my_project::indicators::moving_averages::tilson::{tilson, TilsonInput, TilsonParams};
use my_project::indicators::moving_averages::trendflex::{
    trendflex, TrendFlexInput, TrendFlexParams,
};
use my_project::indicators::moving_averages::trima::{trima, TrimaInput, TrimaParams};
use my_project::indicators::moving_averages::vpwma::{vpwma, VpwmaInput, VpwmaParams};
use my_project::indicators::moving_averages::vwap::{vwap, VwapInput, VwapParams};
use my_project::indicators::moving_averages::vwma::{vwma, VwmaInput, VwmaParams};
use my_project::indicators::moving_averages::wilders::{wilders, WildersInput, WildersParams};
use my_project::indicators::moving_averages::wma::{wma, WmaInput, WmaParams};
use my_project::indicators::moving_averages::zlema::{zlema, ZlemaInput, ZlemaParams};
use my_project::indicators::pma::{pma, PmaInput, PmaParams};
use my_project::indicators::ppo::{ppo, PpoInput, PpoParams};
use my_project::indicators::roc::{roc, RocInput, RocParams};
use my_project::indicators::rocp::{rocp, RocpInput, RocpParams};
use my_project::indicators::rsi::{rsi, RsiInput, RsiParams};
use my_project::indicators::rsx::{rsx, RsxInput, RsxParams};
use my_project::indicators::rvi::{rvi, RviInput, RviParams};
use my_project::indicators::squeeze_momentum::{
    squeeze_momentum, SqueezeMomentumInput, SqueezeMomentumParams,
};
use my_project::indicators::stddev::{stddev, StdDevInput, StdDevParams};
use my_project::indicators::tsf::{tsf, TsfInput, TsfParams};
use my_project::indicators::ui::{ui, UiInput, UiParams};
use my_project::indicators::var::{var, VarInput, VarParams};
use my_project::indicators::vpci::{vpci, VpciInput, VpciParams};
use my_project::indicators::vpt::{vpt, VptInput};
use my_project::indicators::wclprice::{wclprice, WclpriceInput};
use my_project::utilities::data_loader::read_candles_from_csv;
use serde_json::json;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <indicator_name> [source]", args[0]);
        eprintln!("Available indicators: ad, acosc, adx, adosc, adxr, alligator, alma, ao, apo, aroon, aroonosc, atr, bandpass, bollinger_bands, bollinger_bands_width, bop, cci, cfo, cg, chop, cwma, decycler, dema, devstop, di, edcf, ehlers_itrend, ema, epma, eri, fisher, frama, fwma, gaussian, highpass_2_pole, highpass, hma, hwma, jma, jsa, kama, kst, kurtosis, linreg, maaq, macz, mama, marketefi, midpoint, midprice, mfi, mwdx, nma, pma, ppo, rsx, pwma, reflex, roc, rocp, rsi, rvi, rvi, sama, sinwma, sma, smma, squeeze_momentum, sqwma, srwma, stddev, supersmoother_3_pole, supersmoother, swma, tema, tilson, trendflex, trima, var, vpci, tsf, ui, vwap, vwma, vpwma, wclprice, wilders, wma, zlema");
        eprintln!("Available sources: open, high, low, close, volume, hl2, hlc3, ohlc4, hlcc4");
        std::process::exit(1);
    }

    let indicator = &args[1];
    let source = args.get(2).map(|s| s.as_str()).unwrap_or("close");

    // Load test data
    let candles = read_candles_from_csv("src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv")?;

    let output = match indicator.as_str() {
        "alma" => {
            let params = AlmaParams::default();
            let period = params.period.unwrap_or(9);
            let offset = params.offset.unwrap_or(0.85);
            let sigma = params.sigma.unwrap_or(6.0);
            let input = AlmaInput::from_candles(&candles, source, params);
            let result = alma(&input)?;
            json!({
                "indicator": "alma",
                "source": source,
                "params": {
                    "period": period,
                    "offset": offset,
                    "sigma": sigma
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "cwma" => {
            let params = CwmaParams::default();
            let period = params.period.unwrap_or(14);
            let input = CwmaInput::from_candles(&candles, source, params);
            let result = cwma(&input)?;
            json!({
                "indicator": "cwma",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "decycler" => {
            let params = DecyclerParams::default();
            let hp_period = params.hp_period.unwrap_or(125);
            let k = params.k.unwrap_or(0.707);
            let input = DecyclerInput::from_candles(&candles, source, params);
            let result = decycler(&input)?;
            json!({
                "indicator": "decycler",
                "source": source,
                "params": {
                    "hp_period": hp_period,
                    "k": k
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "deviation" => {
            let params = DeviationParams::default();
            let period = params.period.unwrap_or(9);
            let devtype = params.devtype.unwrap_or(0);
            let input = DeviationInput::from_candles(&candles, source, params);
            let result = deviation(&input)?;
            json!({
                "indicator": "deviation",
                "source": source,
                "params": {
                    "period": period,
                    "devtype": devtype
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "emv" => {
            // EMV requires OHLCV data
            let high = candles.select_candle_field("high")?;
            let low = candles.select_candle_field("low")?;
            let close = candles.select_candle_field("close")?;
            let volume = candles.select_candle_field("volume")?;
            let input = EmvInput::from_slices(high, low, close, volume);
            let result = emv(&input)?;
            json!({
                "indicator": "emv",
                "source": "ohlcv",
                "params": {},
                "values": result.values,
                "length": result.values.len()
            })
        }
        "devstop" => {
            // DevStop requires high/low data
            let params = DevStopParams::default();
            let period = params.period.unwrap_or(20);
            let mult = params.mult.unwrap_or(0.0);
            let devtype = params.devtype.unwrap_or(0);
            let direction = params.direction.clone().unwrap_or("long".to_string());
            let ma_type = params.ma_type.clone().unwrap_or("sma".to_string());

            let high = &candles.high;
            let low = &candles.low;
            let input = DevStopInput {
                data: DevStopData::SliceHL(high, low),
                params,
            };
            let result = devstop(&input)?;
            json!({
                "indicator": "devstop",
                "source": "hl",
                "params": {
                    "period": period,
                    "mult": mult,
                    "devtype": devtype,
                    "direction": direction,
                    "ma_type": ma_type
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "er" => {
            let params = ErParams::default();
            let period = params.period.unwrap_or(5);
            let input = ErInput::from_candles(&candles, source, params);
            let result = er(&input)?;
            json!({
                "indicator": "er",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "dema" => {
            let params = DemaParams::default();
            let period = params.period.unwrap_or(21);
            let input = DemaInput::from_candles(&candles, source, params);
            let result = dema(&input)?;
            json!({
                "indicator": "dema",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "dpo" => {
            let params = DpoParams::default();
            let period = params.period.unwrap_or(5);
            let input = DpoInput::from_candles(&candles, source, params);
            let result = dpo(&input)?;
            json!({
                "indicator": "dpo",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "damiani_volatmeter" => {
            let params = DamianiVolatmeterParams::default();
            let vis_atr = params.vis_atr.unwrap_or(13);
            let vis_std = params.vis_std.unwrap_or(20);
            let sed_atr = params.sed_atr.unwrap_or(40);
            let sed_std = params.sed_std.unwrap_or(100);
            let threshold = params.threshold.unwrap_or(1.4);
            let input = DamianiVolatmeterInput::from_candles(&candles, source, params);
            let result = damiani_volatmeter(&input)?;
            json!({
                "indicator": "damiani_volatmeter",
                "source": source,
                "params": {
                    "vis_atr": vis_atr,
                    "vis_std": vis_std,
                    "sed_atr": sed_atr,
                    "sed_std": sed_std,
                    "threshold": threshold
                },
                "vol": result.vol,
                "anti": result.anti,
                "length": result.vol.len()
            })
        }
        "di" => {
            if source != "hlc" {
                eprintln!("DI indicator requires 'hlc' source");
                std::process::exit(1);
            }
            let params = DiParams::default();
            let period = params.period.unwrap_or(14);
            let input = DiInput {
                data: DiData::Candles { candles: &candles },
                params,
            };
            let result = di(&input)?;
            json!({
                "indicator": "di",
                "source": source,
                "params": {
                    "period": period
                },
                "plus": result.plus,
                "minus": result.minus,
                "length": result.plus.len()
            })
        }
        "edcf" => {
            let params = EdcfParams::default();
            let period = params.period.unwrap_or(15);
            let input = EdcfInput::from_candles(&candles, source, params);
            let result = edcf(&input)?;
            json!({
                "indicator": "edcf",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "ehlers_itrend" => {
            let params = EhlersITrendParams::default();
            let warmup_bars = params.warmup_bars.unwrap_or(12);
            let max_dc_period = params.max_dc_period.unwrap_or(50);
            let input = EhlersITrendInput::from_candles(&candles, source, params);
            let result = ehlers_itrend(&input)?;
            json!({
                "indicator": "ehlers_itrend",
                "source": source,
                "params": {
                    "warmup_bars": warmup_bars,
                    "max_dc_period": max_dc_period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "ehlers_ecema" => {
            let params = EhlersEcemaParams::default();
            let length = params.length.unwrap_or(20);
            let gain_limit = params.gain_limit.unwrap_or(50);
            let pine_compatible = params.pine_compatible.unwrap_or(false);
            let confirmed_only = params.confirmed_only.unwrap_or(false);
            let input = EhlersEcemaInput::from_candles(&candles, source, params);
            let result = ehlers_ecema(&input)?;
            json!({
                "indicator": "ehlers_ecema",
                "source": source,
                "params": {
                    "length": length,
                    "gain_limit": gain_limit,
                    "pine_compatible": pine_compatible,
                    "confirmed_only": confirmed_only
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "ema" => {
            let params = EmaParams::default();
            let period = params.period.unwrap_or(9);
            let input = EmaInput::from_candles(&candles, source, params);
            let result = ema(&input)?;
            json!({
                "indicator": "ema",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "epma" => {
            let params = EpmaParams::default();
            let period = params.period.unwrap_or(11);
            let offset = params.offset.unwrap_or(4);
            let input = EpmaInput::from_candles(&candles, source, params);
            let result = epma(&input)?;
            json!({
                "indicator": "epma",
                "source": source,
                "params": {
                    "period": period,
                    "offset": offset
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "eri" => {
            let params = EriParams::default();
            let period = params.period.unwrap_or(13);
            let ma_type = params.ma_type.clone().unwrap_or("ema".to_string());
            let input = EriInput {
                data: EriData::Candles {
                    candles: &candles,
                    source,
                },
                params,
            };
            let result = eri(&input)?;
            json!({
                "indicator": "eri",
                "source": source,
                "params": {
                    "period": period,
                    "ma_type": ma_type
                },
                "bull": result.bull,
                "bear": result.bear,
                "length": result.bull.len()
            })
        }
        "fisher" => {
            let params = FisherParams::default();
            let period = params.period.unwrap_or(9);
            let input = FisherInput::from_candles(&candles, params);
            let result = fisher(&input)?;
            json!({
                "indicator": "fisher",
                "source": "high,low",  // Fisher uses high and low
                "params": {
                    "period": period
                },
                "fisher": result.fisher,
                "signal": result.signal,
                "length": result.fisher.len()
            })
        }
        "frama" => {
            let params = FramaParams::default();
            // Use the true FRAMA defaults from FramaParams::default()
            // Window defaults to 10, sc to 300, fc to 1.
            let window = params.window.unwrap_or(10);
            let sc = params.sc.unwrap_or(300);
            let fc = params.fc.unwrap_or(1);
            let input = FramaInput::from_candles(&candles, params);
            let result = frama(&input)?;
            json!({
                "indicator": "frama",
                "source": "high,low,close", // FRAMA uses multiple price sources
                "params": {
                    "window": window,
                    "sc": sc,
                    "fc": fc
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "fwma" => {
            let params = FwmaParams::default();
            let period = params.period.unwrap_or(5);
            let input = FwmaInput::from_candles(&candles, source, params);
            let result = fwma(&input)?;
            json!({
                "indicator": "fwma",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "gaussian" => {
            let params = GaussianParams::default();
            let period = params.period.unwrap_or(14);
            let poles = params.poles.unwrap_or(4);
            let input = GaussianInput::from_candles(&candles, source, params);
            let result = gaussian(&input)?;
            json!({
                "indicator": "gaussian",
                "source": source,
                "params": {
                    "period": period,
                    "poles": poles
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "highpass_2_pole" => {
            let params = HighPass2Params::default();
            let period = params.period.unwrap_or(48);
            let k = params.k.unwrap_or(0.707);
            let input = HighPass2Input::from_candles(&candles, source, params);
            let result = highpass_2_pole(&input)?;
            json!({
                "indicator": "highpass_2_pole",
                "source": source,
                "params": {
                    "period": period,
                    "k": k
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "highpass" => {
            let params = HighPassParams::default();
            let period = params.period.unwrap_or(48);
            let input = HighPassInput::from_candles(&candles, source, params);
            let result = highpass(&input)?;
            json!({
                "indicator": "highpass",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "hma" => {
            let params = HmaParams::default();
            let period = params.period.unwrap_or(5);
            let input = HmaInput::from_candles(&candles, source, params);
            let result = hma(&input)?;
            json!({
                "indicator": "hma",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "hwma" => {
            let params = HwmaParams::default();
            let na = params.na.unwrap_or(0.2);
            let nb = params.nb.unwrap_or(0.1);
            let nc = params.nc.unwrap_or(0.1);
            let input = HwmaInput::from_candles(&candles, source, params);
            let result = hwma(&input)?;
            json!({
                "indicator": "hwma",
                "source": source,
                "params": {
                    "na": na,
                    "nb": nb,
                    "nc": nc
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "jma" => {
            let params = JmaParams::default();
            let period = params.period.unwrap_or(7);
            let phase = params.phase.unwrap_or(50.0);
            let power = params.power.unwrap_or(2);
            let input = JmaInput::from_candles(&candles, source, params);
            let result = jma(&input)?;
            json!({
                "indicator": "jma",
                "source": source,
                "params": {
                    "period": period,
                    "phase": phase,
                    "power": power
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "jsa" => {
            let params = JsaParams::default();
            let period = params.period.unwrap_or(30);
            let input = JsaInput::from_candles(&candles, source, params);
            let result = jsa(&input)?;
            json!({
                "indicator": "jsa",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "kurtosis" => {
            let params = KurtosisParams::default();
            let period = params.period.unwrap_or(5);
            let input = KurtosisInput::from_candles(&candles, source, params);
            let result = kurtosis(&input)?;
            json!({
                "indicator": "kurtosis",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "kst" => {
            let params = KstParams::default();
            let input = KstInput::from_candles(&candles, source, params);
            let result = kst(&input)?;
            json!({
                "indicator": "kst",
                "source": source,
                "params": {
                    "sma_period1": params.sma_period1.unwrap_or(10),
                    "sma_period2": params.sma_period2.unwrap_or(10),
                    "sma_period3": params.sma_period3.unwrap_or(10),
                    "sma_period4": params.sma_period4.unwrap_or(15),
                    "roc_period1": params.roc_period1.unwrap_or(10),
                    "roc_period2": params.roc_period2.unwrap_or(15),
                    "roc_period3": params.roc_period3.unwrap_or(20),
                    "roc_period4": params.roc_period4.unwrap_or(30),
                    "signal_period": params.signal_period.unwrap_or(9)
                },
                "line": result.line,
                "signal": result.signal,
                "length": result.line.len()
            })
        }
        "kst_line" => {
            let params = KstParams::default();
            let input = KstInput::from_candles(&candles, source, params);
            let result = kst(&input)?;
            json!({
                "indicator": "kst_line",
                "source": source,
                "params": {
                    "sma_period1": params.sma_period1.unwrap_or(10),
                    "sma_period2": params.sma_period2.unwrap_or(10),
                    "sma_period3": params.sma_period3.unwrap_or(10),
                    "sma_period4": params.sma_period4.unwrap_or(15),
                    "roc_period1": params.roc_period1.unwrap_or(10),
                    "roc_period2": params.roc_period2.unwrap_or(15),
                    "roc_period3": params.roc_period3.unwrap_or(20),
                    "roc_period4": params.roc_period4.unwrap_or(30),
                    "signal_period": params.signal_period.unwrap_or(9)
                },
                "values": result.line,
                "length": result.line.len()
            })
        }
        "kst_signal" => {
            let params = KstParams::default();
            let input = KstInput::from_candles(&candles, source, params);
            let result = kst(&input)?;
            json!({
                "indicator": "kst_signal",
                "source": source,
                "params": {
                    "sma_period1": params.sma_period1.unwrap_or(10),
                    "sma_period2": params.sma_period2.unwrap_or(10),
                    "sma_period3": params.sma_period3.unwrap_or(10),
                    "sma_period4": params.sma_period4.unwrap_or(15),
                    "roc_period1": params.roc_period1.unwrap_or(10),
                    "roc_period2": params.roc_period2.unwrap_or(15),
                    "roc_period3": params.roc_period3.unwrap_or(20),
                    "roc_period4": params.roc_period4.unwrap_or(30),
                    "signal_period": params.signal_period.unwrap_or(9)
                },
                "values": result.signal,
                "length": result.signal.len()
            })
        }
        "kama" => {
            let params = KamaParams::default();
            let period = params.period.unwrap_or(30);
            let input = KamaInput::from_candles(&candles, source, params);
            let result = kama(&input)?;
            json!({
                "indicator": "kama",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "linreg" => {
            let params = LinRegParams::default();
            let period = params.period.unwrap_or(14);
            let input = LinRegInput::from_candles(&candles, source, params);
            let result = linreg(&input)?;
            json!({
                "indicator": "linreg",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "linearreg_intercept" => {
            let params = LinearRegInterceptParams::default();
            let period = params.period.unwrap_or(14);
            let input = LinearRegInterceptInput::from_candles(&candles, source, params);
            let result = linearreg_intercept(&input)?;
            json!({
                "indicator": "linearreg_intercept",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "maaq" => {
            let params = MaaqParams::default();
            let period = params.period.unwrap_or(11);
            let fast_period = params.fast_period.unwrap_or(2);
            let slow_period = params.slow_period.unwrap_or(30);
            let input = MaaqInput::from_candles(&candles, source, params);
            let result = maaq(&input)?;
            json!({
                "indicator": "maaq",
                "source": source,
                "params": {
                    "period": period,
                    "fast_period": fast_period,
                    "slow_period": slow_period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "macz" => {
            let params = MaczParams::default();
            let fast_length = params.fast_length.unwrap_or(12);
            let slow_length = params.slow_length.unwrap_or(25);
            let signal_length = params.signal_length.unwrap_or(9);
            let lengthz = params.lengthz.unwrap_or(20);
            let length_stdev = params.length_stdev.unwrap_or(25);
            let a = params.a.unwrap_or(1.0);
            let b = params.b.unwrap_or(1.0);
            let use_lag = params.use_lag.unwrap_or(false);
            let gamma = params.gamma.unwrap_or(0.02);
            let input = MaczInput::from_candles(&candles, source, params);
            let result = macz(&input)?;
            json!({
                "indicator": "macz",
                "source": source,
                "params": {
                    "fast_length": fast_length,
                    "slow_length": slow_length,
                    "signal_length": signal_length,
                    "lengthz": lengthz,
                    "length_stdev": length_stdev,
                    "a": a,
                    "b": b,
                    "use_lag": use_lag,
                    "gamma": gamma
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "mama" => {
            let params = MamaParams::default();
            let fast_limit = params.fast_limit.unwrap_or(0.5);
            let slow_limit = params.slow_limit.unwrap_or(0.05);
            let input = MamaInput::from_candles(&candles, source, params);
            let result = mama(&input)?;
            json!({
                "indicator": "mama",
                "source": source,
                "params": {
                    "fast_limit": fast_limit,
                    "slow_limit": slow_limit
                },
                "mama_values": result.mama_values,
                "fama_values": result.fama_values,
                "length": result.mama_values.len()
            })
        }
        "midpoint" => {
            let params = MidpointParams::default();
            let period = params.period.unwrap_or(14);
            let input = MidpointInput::from_candles(&candles, source, params);
            let result = midpoint(&input)?;
            json!({
                "indicator": "midpoint",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "midprice" => {
            if source != "hl" {
                eprintln!("Midprice indicator requires 'hl' source");
                std::process::exit(1);
            }
            let params = MidpriceParams::default();
            let period = params.period.unwrap_or(14);
            let input = MidpriceInput::from_candles(&candles, "high", "low", params);
            let result = midprice(&input)?;
            json!({
                "indicator": "midprice",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "marketefi" => {
            if source != "hlv" {
                eprintln!("MarketEFI indicator requires 'hlv' source");
                std::process::exit(1);
            }
            let params = MarketefiParams::default();
            let input = MarketefiInput {
                data: MarketefiData::Candles {
                    candles: &candles,
                    source_high: "high",
                    source_low: "low",
                    source_volume: "volume",
                },
                params,
            };
            let result = marketefi(&input)?;
            json!({
                "indicator": "marketefi",
                "source": source,
                "params": {},
                "values": result.values,
                "length": result.values.len()
            })
        }
        "mass" => {
            // Mass Index requires high and low data
            if !source.contains(",") {
                eprintln!("Mass Index requires 'high,low' source");
                std::process::exit(1);
            }
            let params = MassParams::default();
            let period = params.period.unwrap_or(5);
            let input = MassInput::from_candles(&candles, "high", "low", params);
            let result = mass(&input)?;
            json!({
                "indicator": "mass",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "mfi" => {
            // MFI requires typical price and volume
            if source != "hlc3_volume" {
                eprintln!("MFI indicator requires 'hlc3_volume' source");
                std::process::exit(1);
            }
            let params = MfiParams::default();
            let period = params.period.unwrap_or(14);

            // Calculate typical price directly from candles fields
            let typical_price: Vec<f64> = candles
                .high
                .iter()
                .zip(candles.low.iter())
                .zip(candles.close.iter())
                .map(|((h, l), c)| (h + l + c) / 3.0)
                .collect();

            let input = MfiInput {
                data: MfiData::Slices {
                    typical_price: &typical_price,
                    volume: &candles.volume,
                },
                params,
            };
            let result = mfi(&input)?;
            json!({
                "indicator": "mfi",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "mwdx" => {
            let params = MwdxParams::default();
            let factor = params.factor.unwrap_or(0.2);
            let input = MwdxInput::from_candles(&candles, source, params);
            let result = mwdx(&input)?;
            json!({
                "indicator": "mwdx",
                "source": source,
                "params": {
                    "factor": factor
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "nma" => {
            let params = NmaParams::default();
            let period = params.period.unwrap_or(40);
            let input = NmaInput::from_candles(&candles, source, params);
            let result = nma(&input)?;
            json!({
                "indicator": "nma",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "ppo" => {
            let params = PpoParams::default();
            let fast_period = params.fast_period.unwrap_or(12);
            let slow_period = params.slow_period.unwrap_or(26);
            let ma_type = params.ma_type.clone().unwrap_or_else(|| "sma".to_string());
            let input = PpoInput::from_candles(&candles, source, params);
            let result = ppo(&input)?;
            json!({
                "indicator": "ppo",
                "source": source,
                "params": {
                    "fast_period": fast_period,
                    "slow_period": slow_period,
                    "ma_type": ma_type
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "rsx" => {
            let params = RsxParams::default();
            let period = params.period.unwrap_or(14);
            let input = RsxInput::from_candles(&candles, source, params);
            let result = rsx(&input)?;
            json!({
                "indicator": "rsx",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "pma" => {
            let params = PmaParams::default();
            let input = PmaInput::from_candles(&candles, source, params);
            let result = pma(&input)?;
            // PMA returns two arrays: predict and trigger
            // For consistency with other indicators, we'll return predict as the main values
            json!({
                "indicator": "pma",
                "source": source,
                "params": {},
                "values": result.predict,
                "trigger": result.trigger,
                "length": result.predict.len()
            })
        }
        "pwma" => {
            let params = PwmaParams::default();
            let period = params.period.unwrap_or(5);
            let input = PwmaInput::from_candles(&candles, source, params);
            let result = pwma(&input)?;
            json!({
                "indicator": "pwma",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "reflex" => {
            let params = ReflexParams::default();
            let period = params.period.unwrap_or(20);
            let input = ReflexInput::from_candles(&candles, source, params);
            let result = reflex(&input)?;
            json!({
                "indicator": "reflex",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "sinwma" => {
            let params = SinWmaParams::default();
            let period = params.period.unwrap_or(14);
            let input = SinWmaInput::from_candles(&candles, source, params);
            let result = sinwma(&input)?;
            json!({
                "indicator": "sinwma",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "sma" => {
            let params = SmaParams::default();
            let period = params.period.unwrap_or(9);
            let input = SmaInput::from_candles(&candles, source, params);
            let result = sma(&input)?;
            json!({
                "indicator": "sma",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "smma" => {
            let params = SmmaParams::default();
            let period = params.period.unwrap_or(7);
            let input = SmmaInput::from_candles(&candles, source, params);
            let result = smma(&input)?;
            json!({
                "indicator": "smma",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "sqwma" => {
            let params = SqwmaParams::default();
            let period = params.period.unwrap_or(14);
            let input = SqwmaInput::from_candles(&candles, source, params);
            let result = sqwma(&input)?;
            json!({
                "indicator": "sqwma",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "srwma" => {
            let params = SrwmaParams::default();
            let period = params.period.unwrap_or(14);
            let input = SrwmaInput::from_candles(&candles, source, params);
            let result = srwma(&input)?;
            json!({
                "indicator": "srwma",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "stddev" => {
            let params = StdDevParams::default();
            let period = params.period.unwrap_or(5);
            let nbdev = params.nbdev.unwrap_or(1.0);
            let input = StdDevInput::from_candles(&candles, source, params);
            let result = stddev(&input)?;
            json!({
                "indicator": "stddev",
                "source": source,
                "params": {
                    "period": period,
                    "nbdev": nbdev
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "var" => {
            let params = VarParams::default();
            let period = params.period.unwrap_or(14);
            let nbdev = params.nbdev.unwrap_or(1.0);
            let input = VarInput::from_candles(&candles, source, params);
            let result = var(&input)?;
            json!({
                "indicator": "var",
                "source": source,
                "params": {
                    "period": period,
                    "nbdev": nbdev
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "supersmoother_3_pole" => {
            let params = SuperSmoother3PoleParams::default();
            let period = params.period.unwrap_or(14);
            let input = SuperSmoother3PoleInput::from_candles(&candles, source, params);
            let result = supersmoother_3_pole(&input)?;
            json!({
                "indicator": "supersmoother_3_pole",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "supersmoother" => {
            let params = SuperSmootherParams::default();
            let period = params.period.unwrap_or(14);
            let input = SuperSmootherInput::from_candles(&candles, source, params);
            let result = supersmoother(&input)?;
            json!({
                "indicator": "supersmoother",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "swma" => {
            let params = SwmaParams::default();
            let period = params.period.unwrap_or(5);
            let input = SwmaInput::from_candles(&candles, source, params);
            let result = swma(&input)?;
            json!({
                "indicator": "swma",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "tema" => {
            let params = TemaParams::default();
            let period = params.period.unwrap_or(9);
            let input = TemaInput::from_candles(&candles, source, params);
            let result = tema(&input)?;
            json!({
                "indicator": "tema",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "tilson" => {
            let params = TilsonParams::default();
            let period = params.period.unwrap_or(5);
            let volume_factor = params.volume_factor.unwrap_or(0.0);
            let input = TilsonInput::from_candles(&candles, source, params);
            let result = tilson(&input)?;
            json!({
                "indicator": "tilson",
                "source": source,
                "params": {
                    "period": period,
                    "volume_factor": volume_factor
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "trendflex" => {
            let params = TrendFlexParams::default();
            let period = params.period.unwrap_or(20);
            let input = TrendFlexInput::from_candles(&candles, source, params);
            let result = trendflex(&input)?;
            json!({
                "indicator": "trendflex",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "trima" => {
            let params = TrimaParams::default();
            let period = params.period.unwrap_or(30);
            let input = TrimaInput::from_candles(&candles, source, params);
            let result = trima(&input)?;
            json!({
                "indicator": "trima",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "roc" => {
            let params = RocParams::default();
            let period = params.period.unwrap_or(9);
            let input = RocInput::from_candles(&candles, source, params);
            let result = roc(&input)?;
            json!({
                "indicator": "roc",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "rocp" => {
            let params = RocpParams::default();
            let period = params.period.unwrap_or(10);
            let input = RocpInput::from_candles(&candles, source, params);
            let result = rocp(&input)?;
            json!({
                "indicator": "rocp",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "rsi" => {
            let params = RsiParams::default();
            let period = params.period.unwrap_or(14);
            let input = RsiInput::from_candles(&candles, source, params);
            let result = rsi(&input)?;
            json!({
                "indicator": "rsi",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "rvi" => {
            let params = RviParams::default();
            let period = params.period.unwrap_or(10);
            let ma_len = params.ma_len.unwrap_or(14);
            let matype = params.matype.unwrap_or(1);
            let devtype = params.devtype.unwrap_or(0);
            let input = RviInput::from_candles(&candles, source, params);
            let result = rvi(&input)?;
            json!({
                "indicator": "rvi",
                "source": source,
                "params": {
                    "period": period,
                    "ma_len": ma_len,
                    "matype": matype,
                    "devtype": devtype
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "tsf" => {
            let params = TsfParams::default();
            let period = params.period.unwrap_or(14);
            let input = TsfInput::from_candles(&candles, source, params);
            let result = tsf(&input)?;
            json!({
                "indicator": "tsf",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "ui" => {
            let params = UiParams::default();
            let period = params.period.unwrap_or(14);
            let scalar = params.scalar.unwrap_or(100.0);
            let input = UiInput::from_candles(&candles, source, params);
            let result = ui(&input)?;
            json!({
                "indicator": "ui",
                "source": source,
                "params": {
                    "period": period,
                    "scalar": scalar
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "squeeze_momentum" => {
            let params = SqueezeMomentumParams::default();
            let length_bb = params.length_bb.unwrap_or(20);
            let mult_bb = params.mult_bb.unwrap_or(2.0);
            let length_kc = params.length_kc.unwrap_or(20);
            let mult_kc = params.mult_kc.unwrap_or(1.5);
            let input = SqueezeMomentumInput::from_candles(&candles, params);
            let result = squeeze_momentum(&input)?;
            // For squeeze_momentum, we return the momentum values as the main output
            // This matches what the Python test expects
            json!({
                "indicator": "squeeze_momentum",
                "source": "hlc",
                "params": {
                    "length_bb": length_bb,
                    "mult_bb": mult_bb,
                    "length_kc": length_kc,
                    "mult_kc": mult_kc
                },
                "values": result.momentum,
                "length": result.momentum.len()
            })
        }
        "vwap" => {
            let params = VwapParams::default();
            let anchor = params.anchor.clone().unwrap_or_else(|| "1d".to_string());
            let input = VwapInput::from_candles(&candles, "hlcv", params);
            let result = vwap(&input)?;
            json!({
                "indicator": "vwap",
                "source": "hlcv",
                "params": {
                    "anchor": anchor
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "vwma" => {
            let params = VwmaParams::default();
            let period = params.period.unwrap_or(20);
            let input = VwmaInput::from_candles(&candles, source, params);
            let result = vwma(&input)?;
            json!({
                "indicator": "vwma",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "vpci" => {
            let params = VpciParams::default();
            let short_range = params.short_range.unwrap_or(5);
            let long_range = params.long_range.unwrap_or(25);
            let input = VpciInput::from_candles(&candles, "close", "volume", params);
            let result = vpci(&input)?;
            json!({
                "indicator": "vpci",
                "source": "close",
                "params": {
                    "short_range": short_range,
                    "long_range": long_range
                },
                "vpci": result.vpci,
                "vpcis": result.vpcis,
                "length": result.vpci.len()
            })
        }
        "vpt" => {
            // VPT requires price and volume data
            let volume = candles.select_candle_field("volume")?;
            let price = candles.select_candle_field(source)?;
            let input = VptInput::from_slices(price, volume);
            let result = vpt(&input)?;
            json!({
                "indicator": "vpt",
                "source": source,
                "params": {},
                "values": result.values,
                "length": result.values.len()
            })
        }
        "vpwma" => {
            let params = VpwmaParams::default();
            let period = params.period.unwrap_or(20);
            let power = params.power.unwrap_or(1.0);
            let input = VpwmaInput::from_candles(&candles, source, params);
            let result = vpwma(&input)?;
            json!({
                "indicator": "vpwma",
                "source": source,
                "params": {
                    "period": period,
                    "power": power
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "wilders" => {
            let params = WildersParams::default();
            let period = params.period.unwrap_or(14);
            let input = WildersInput::from_candles(&candles, source, params);
            let result = wilders(&input)?;
            json!({
                "indicator": "wilders",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "wma" => {
            let params = WmaParams::default();
            let period = params.period.unwrap_or(9);
            let input = WmaInput::from_candles(&candles, source, params);
            let result = wma(&input)?;
            json!({
                "indicator": "wma",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "zlema" => {
            let params = ZlemaParams::default();
            let period = params.period.unwrap_or(14);
            let input = ZlemaInput::from_candles(&candles, source, params);
            let result = zlema(&input)?;
            json!({
                "indicator": "zlema",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "ad" => {
            if source != "ohlcv" {
                eprintln!("AD indicator requires 'ohlcv' source");
                std::process::exit(1);
            }
            let params = AdParams::default();
            let data = AdData::Candles { candles: &candles };
            let input = AdInput { data, params };
            let result = ad(&input)?;
            json!({
                "indicator": "ad",
                "source": source,
                "params": {},
                "values": result.values,
                "length": result.values.len()
            })
        }
        "acosc" => {
            if source != "high_low" {
                eprintln!("ACOSC indicator requires 'high_low' source");
                std::process::exit(1);
            }
            let params = AcoscParams::default();
            let data = AcoscData::Candles { candles: &candles };
            let input = AcoscInput { data, params };
            let result = acosc(&input)?;
            json!({
                "indicator": "acosc",
                "source": source,
                "params": {},
                "osc": result.osc,
                "change": result.change,
                "length": result.osc.len()
            })
        }
        "adx" => {
            if source != "ohlc" {
                eprintln!("ADX indicator requires 'ohlc' source");
                std::process::exit(1);
            }
            let params = AdxParams::default();
            let input = AdxInput {
                data: AdxData::Candles { candles: &candles },
                params,
            };
            let result = adx(&input)?;
            json!({
                "indicator": "adx",
                "source": source,
                "params": {},
                "values": result.values,
                "length": result.values.len()
            })
        }
        "adosc" => {
            if source != "hlcv" {
                eprintln!("ADOSC indicator requires 'hlcv' source");
                std::process::exit(1);
            }
            let params = AdoscParams::default();
            let input = AdoscInput {
                data: AdoscData::Candles { candles: &candles },
                params,
            };
            let result = adosc(&input)?;
            json!({
                "indicator": "adosc",
                "source": source,
                "params": {},
                "values": result.values,
                "length": result.values.len()
            })
        }
        "adxr" => {
            if source != "hlc" {
                eprintln!("ADXR indicator requires 'hlc' source");
                std::process::exit(1);
            }
            let params = AdxrParams::default();
            let input = AdxrInput {
                data: AdxrData::Candles { candles: &candles },
                params,
            };
            let result = adxr(&input)?;
            json!({
                "indicator": "adxr",
                "source": source,
                "params": {},
                "values": result.values,
                "length": result.values.len()
            })
        }
        "alligator" => {
            let params = AlligatorParams::default();
            let jaw_period = params.jaw_period.unwrap_or(13);
            let teeth_period = params.teeth_period.unwrap_or(8);
            let lips_period = params.lips_period.unwrap_or(5);
            let jaw_offset = params.jaw_offset.unwrap_or(8);
            let teeth_offset = params.teeth_offset.unwrap_or(5);
            let lips_offset = params.lips_offset.unwrap_or(3);
            let input = AlligatorInput::from_candles(&candles, source, params);
            let result = alligator(&input)?;
            json!({
                "indicator": "alligator",
                "source": source,
                "params": {
                    "jaw_period": jaw_period,
                    "teeth_period": teeth_period,
                    "lips_period": lips_period,
                    "jaw_offset": jaw_offset,
                    "teeth_offset": teeth_offset,
                    "lips_offset": lips_offset
                },
                "jaw": result.jaw,
                "teeth": result.teeth,
                "lips": result.lips,
                "length": result.jaw.len()
            })
        }
        "ao" => {
            if source != "high_low" {
                eprintln!("AO indicator requires 'high_low' source");
                std::process::exit(1);
            }
            let params = AoParams::default();
            let input = AoInput {
                data: AoData::Candles {
                    candles: &candles,
                    source: "hl2",
                },
                params,
            };
            let result = ao(&input)?;
            json!({
                "indicator": "ao",
                "source": source,
                "params": {},
                "values": result.values,
                "length": result.values.len()
            })
        }
        "apo" => {
            let params = ApoParams::default();
            let short_period = params.short_period.unwrap_or(10);
            let long_period = params.long_period.unwrap_or(20);
            let input = ApoInput::from_candles(&candles, source, params);
            let result = apo(&input)?;
            json!({
                "indicator": "apo",
                "source": source,
                "params": {
                    "short_period": short_period,
                    "long_period": long_period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "aroon" => {
            if source != "high_low" {
                eprintln!("Aroon indicator requires 'high_low' source");
                std::process::exit(1);
            }
            let params = AroonParams::default();
            let length = params.length.unwrap_or(14);
            let input = AroonInput {
                data: AroonData::Candles { candles: &candles },
                params,
            };
            let result = aroon(&input)?;
            json!({
                "indicator": "aroon",
                "source": source,
                "params": {
                    "length": length
                },
                "aroon_down": result.aroon_down,
                "aroon_up": result.aroon_up,
                "length": result.aroon_up.len()
            })
        }
        "aroonosc" => {
            if source != "high_low" {
                eprintln!("Aroon Oscillator requires 'high_low' source");
                std::process::exit(1);
            }
            let params = AroonOscParams::default();
            let length = params.length.unwrap_or(14);
            let input = AroonOscInput {
                data: AroonOscData::Candles { candles: &candles },
                params,
            };
            let result = aroon_osc(&input)?;
            json!({
                "indicator": "aroonosc",
                "source": source,
                "params": {
                    "length": length
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "atr" => {
            if source != "ohlc" {
                eprintln!("ATR indicator requires 'ohlc' source");
                std::process::exit(1);
            }
            let params = AtrParams::default();
            let length = params.length.unwrap_or(14);
            let input = AtrInput {
                data: AtrData::Candles { candles: &candles },
                params,
            };
            let result = atr(&input)?;
            json!({
                "indicator": "atr",
                "source": source,
                "params": {
                    "length": length
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "bandpass" => {
            let params = BandPassParams::default();
            let period = params.period.unwrap_or(20);
            let bandwidth = params.bandwidth.unwrap_or(0.3);
            let input = BandPassInput::from_candles(&candles, source, params);
            let result = bandpass(&input)?;
            json!({
                "indicator": "bandpass",
                "source": source,
                "params": {
                    "period": period,
                    "bandwidth": bandwidth
                },
                "bp": result.bp,
                "bp_normalized": result.bp_normalized,
                "signal": result.signal,
                "trigger": result.trigger,
                "length": result.bp.len()
            })
        }
        "bollinger_bands" => {
            let params = BollingerBandsParams::default();
            let period = params.period.unwrap_or(20);
            let devup = params.devup.unwrap_or(2.0);
            let devdn = params.devdn.unwrap_or(2.0);
            let matype = params.matype.clone().unwrap_or("sma".to_string());
            let devtype = params.devtype.unwrap_or(0);
            let input = BollingerBandsInput::from_candles(&candles, source, params);
            let result = bollinger_bands(&input)?;
            json!({
                "indicator": "bollinger_bands",
                "source": source,
                "params": {
                    "period": period,
                    "devup": devup,
                    "devdn": devdn,
                    "matype": matype,
                    "devtype": devtype
                },
                "upper_band": result.upper_band,
                "middle_band": result.middle_band,
                "lower_band": result.lower_band,
                "length": result.upper_band.len()
            })
        }
        "bollinger_bands_width" => {
            let params = BollingerBandsWidthParams::default();
            let period = params.period.unwrap_or(20);
            let devup = params.devup.unwrap_or(2.0);
            let devdn = params.devdn.unwrap_or(2.0);
            let matype = params.matype.clone().unwrap_or("sma".to_string());
            let devtype = params.devtype.unwrap_or(0);
            let input = BollingerBandsWidthInput::from_candles(&candles, source, params);
            let result = bollinger_bands_width(&input)?;
            json!({
                "indicator": "bollinger_bands_width",
                "source": source,
                "params": {
                    "period": period,
                    "devup": devup,
                    "devdn": devdn,
                    "matype": matype,
                    "devtype": devtype
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "cfo" => {
            let params = CfoParams::default();
            let period = params.period.unwrap_or(14);
            let scalar = params.scalar.unwrap_or(100.0);
            let input = CfoInput::from_candles(&candles, source, params);
            let result = cfo(&input)?;
            json!({
                "indicator": "cfo",
                "source": source,
                "params": {
                    "period": period,
                    "scalar": scalar
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "bop" => {
            // BOP requires OHLC data
            let input = BopInput::from_candles(&candles, BopParams::default());
            let result = bop(&input)?;
            json!({
                "indicator": "bop",
                "source": "ohlc",
                "params": {},
                "values": result.values,
                "length": result.values.len()
            })
        }
        "cci" => {
            let params = CciParams::default();
            let period = params.period.unwrap_or(14);
            let input = CciInput::from_candles(&candles, source, params);
            let result = cci(&input)?;
            json!({
                "indicator": "cci",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "cg" => {
            let params = CgParams::default();
            let period = params.period.unwrap_or(10);
            let input = CgInput::from_candles(&candles, source, params);
            let result = cg(&input)?;
            json!({
                "indicator": "cg",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "cvi" => {
            if source != "hl" {
                eprintln!("CVI indicator requires 'hl' source");
                std::process::exit(1);
            }
            let params = CviParams::default();
            let period = params.period.unwrap_or(10);
            let input = CviInput::from_candles(&candles, params);
            let result = cvi(&input)?;
            json!({
                "indicator": "cvi",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "wclprice" => {
            if source != "hlc" {
                eprintln!("WCLPRICE indicator requires 'hlc' source");
                std::process::exit(1);
            }
            let input = WclpriceInput::from_candles(&candles);
            let result = wclprice(&input)?;
            json!({
                "indicator": "wclprice",
                "source": "hlc",
                "params": {},
                "values": result.values,
                "length": result.values.len()
            })
        }
        "chande" => {
            if source != "candles" {
                eprintln!("Chande indicator requires 'candles' source");
                std::process::exit(1);
            }
            let params = ChandeParams::default();
            let period = params.period.unwrap_or(22);
            let mult = params.mult.unwrap_or(3.0);
            let direction = params.direction.clone().unwrap_or("long".to_string());
            let input = ChandeInput {
                data: ChandeData::Candles { candles: &candles },
                params,
            };
            let result = chande(&input)?;
            json!({
                "indicator": "chande",
                "source": source,
                "params": {
                    "period": period,
                    "mult": mult,
                    "direction": direction
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "chop" => {
            if source != "hlc" {
                eprintln!("CHOP indicator requires 'hlc' source");
                std::process::exit(1);
            }
            let params = ChopParams::default();
            let period = params.period.unwrap_or(14);
            let scalar = params.scalar.unwrap_or(100.0);
            let drift = params.drift.unwrap_or(1);

            let high = candles.select_candle_field("high")?;
            let low = candles.select_candle_field("low")?;
            let close = candles.select_candle_field("close")?;

            let input = ChopInput {
                data: ChopData::Slice { high, low, close },
                params,
            };
            let result = chop(&input)?;
            json!({
                "indicator": "chop",
                "source": source,
                "params": {
                    "period": period,
                    "scalar": scalar,
                    "drift": drift
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "cmo" => {
            let params = CmoParams::default();
            let period = params.period.unwrap_or(14);
            let input = CmoInput::from_candles(&candles, source, params);
            let result = cmo(&input)?;
            serde_json::json!({
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "correl_hl" => {
            // correl_hl takes high,low as source
            if source != "high,low" {
                eprintln!("CORREL_HL indicator requires 'high,low' source");
                std::process::exit(1);
            }
            let params = CorrelHlParams::default();
            let period = params.period.unwrap_or(9);
            let input = CorrelHlInput {
                data: CorrelHlData::Candles { candles: &candles },
                params,
            };
            let result = correl_hl(&input)?;
            json!({
                "indicator": "correl_hl",
                "source": source,
                "params": {
                    "period": period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "sama" => {
            let params = SamaParams::default();
            let length = params.length.unwrap_or(200);
            let maj_length = params.maj_length.unwrap_or(14);
            let min_length = params.min_length.unwrap_or(6);
            let input = SamaInput::from_candles(&candles, source, params);
            let result = sama(&input)?;
            json!({
                "indicator": "sama",
                "source": source,
                "params": {
                    "length": length,
                    "maj_length": maj_length,
                    "min_length": min_length
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "vama" => {
            // Volatility Adjusted MA uses only price; accept any source but default to close
            let price = candles.select_candle_field(match source {
                "open" | "high" | "low" | "close" | "hl2" | "hlc3" | "ohlc4" | "hlcc4" => source,
                _ => "close",
            })?;
            let params = VamaParams::default();
            let base_period = params.base_period.unwrap_or(113);
            let vol_period = params.vol_period.unwrap_or(51);
            let smoothing = params.smoothing.unwrap_or(true);
            let smooth_type = params.smooth_type.unwrap_or(3);
            let smooth_period = params.smooth_period.unwrap_or(5);
            let input = VamaInput::from_slice(price, params);
            let result = vama(&input)?;
            json!({
                "indicator": "vama",
                "source": "close",
                "params": {
                    "base_period": base_period,
                    "vol_period": vol_period,
                    "smoothing": smoothing,
                    "smooth_type": smooth_type,
                    "smooth_period": smooth_period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        "volume_adjusted_ma" => {
            // Volume Adjusted MA requires price and volume; tests pass 'close_volume'
            let price = candles.select_candle_field("close")?;
            let volume = candles.select_candle_field("volume")?;
            let params = VoluMaParams::default();
            let length = params.length.unwrap_or(13);
            let vi_factor = params.vi_factor.unwrap_or(0.67);
            let strict = params.strict.unwrap_or(true);
            let sample_period = params.sample_period.unwrap_or(0);
            let input = VoluMaInput::from_slices(price, volume, params);
            let result = volu_ma(&input)?;
            json!({
                "indicator": "volume_adjusted_ma",
                "source": "close_volume",
                "params": {
                    "length": length,
                    "vi_factor": vi_factor,
                    "strict": strict,
                    "sample_period": sample_period
                },
                "values": result.values,
                "length": result.values.len()
            })
        }
        _ => {
            eprintln!("Unknown indicator: {}", indicator);
            std::process::exit(1);
        }
    };

    // Output as JSON
    println!("{}", serde_json::to_string_pretty(&output)?);

    Ok(())
}
