/// Binary to generate reference outputs for indicator testing
/// This is used by Python and WASM tests to verify their outputs match Rust
use my_project::indicators::moving_averages::alma::{alma, AlmaInput, AlmaParams};
use my_project::indicators::moving_averages::cwma::{cwma, CwmaInput, CwmaParams};
use my_project::indicators::moving_averages::dema::{dema, DemaInput, DemaParams};
use my_project::indicators::moving_averages::edcf::{edcf, EdcfInput, EdcfParams};
use my_project::indicators::moving_averages::ehlers_itrend::{ehlers_itrend, EhlersITrendInput, EhlersITrendParams};
use my_project::indicators::moving_averages::ema::{ema, EmaInput, EmaParams};
use my_project::indicators::moving_averages::epma::{epma, EpmaInput, EpmaParams};
use my_project::indicators::moving_averages::frama::{frama, FramaInput, FramaParams};
use my_project::indicators::moving_averages::fwma::{fwma, FwmaInput, FwmaParams};
use my_project::indicators::moving_averages::gaussian::{gaussian, GaussianInput, GaussianParams};
use my_project::indicators::moving_averages::highpass_2_pole::{highpass_2_pole, HighPass2Input, HighPass2Params};
use my_project::indicators::moving_averages::highpass::{highpass, HighPassInput, HighPassParams};
use my_project::indicators::moving_averages::hma::{hma, HmaInput, HmaParams};
use my_project::indicators::moving_averages::hwma::{hwma, HwmaInput, HwmaParams};
use my_project::indicators::moving_averages::jma::{jma, JmaInput, JmaParams};
use my_project::indicators::moving_averages::jsa::{jsa, JsaInput, JsaParams};
use my_project::indicators::moving_averages::kama::{kama, KamaInput, KamaParams};
use my_project::indicators::moving_averages::linreg::{linreg, LinRegInput, LinRegParams};
use my_project::indicators::moving_averages::maaq::{maaq, MaaqInput, MaaqParams};
use my_project::indicators::moving_averages::mama::{mama, MamaInput, MamaParams};
use my_project::indicators::moving_averages::mwdx::{mwdx, MwdxInput, MwdxParams};
use my_project::indicators::moving_averages::nma::{nma, NmaInput, NmaParams};
use my_project::indicators::moving_averages::pwma::{pwma, PwmaInput, PwmaParams};
use my_project::indicators::moving_averages::reflex::{reflex, ReflexInput, ReflexParams};
use my_project::indicators::moving_averages::sinwma::{sinwma, SinWmaInput, SinWmaParams};
use my_project::indicators::moving_averages::sma::{sma, SmaInput, SmaParams};
use my_project::indicators::moving_averages::smma::{smma, SmmaInput, SmmaParams};
use my_project::indicators::moving_averages::sqwma::{sqwma, SqwmaInput, SqwmaParams};
use my_project::indicators::moving_averages::srwma::{srwma, SrwmaInput, SrwmaParams};
use my_project::indicators::moving_averages::supersmoother_3_pole::{supersmoother_3_pole, SuperSmoother3PoleInput, SuperSmoother3PoleParams};
use my_project::indicators::moving_averages::supersmoother::{supersmoother, SuperSmootherInput, SuperSmootherParams};
use my_project::indicators::moving_averages::swma::{swma, SwmaInput, SwmaParams};
use my_project::indicators::moving_averages::tema::{tema, TemaInput, TemaParams};
use my_project::indicators::moving_averages::tilson::{tilson, TilsonInput, TilsonParams};
use my_project::indicators::moving_averages::trendflex::{trendflex, TrendFlexInput, TrendFlexParams};
use my_project::indicators::moving_averages::trima::{trima, TrimaInput, TrimaParams};
use my_project::indicators::moving_averages::vwap::{vwap, VwapInput, VwapParams};
use my_project::indicators::moving_averages::vwma::{vwma, VwmaInput, VwmaParams};
use my_project::indicators::moving_averages::vpwma::{vpwma, VpwmaInput, VpwmaParams};
use my_project::indicators::moving_averages::wilders::{wilders, WildersInput, WildersParams};
use my_project::indicators::moving_averages::wma::{wma, WmaInput, WmaParams};
use my_project::indicators::moving_averages::zlema::{zlema, ZlemaInput, ZlemaParams};
use my_project::indicators::ad::{ad, AdInput, AdParams, AdData};
use my_project::indicators::acosc::{acosc, AcoscInput, AcoscParams, AcoscData};
use my_project::indicators::adx::{adx, AdxInput, AdxParams, AdxData};
use my_project::indicators::adosc::{adosc, AdoscInput, AdoscParams, AdoscData};
use my_project::indicators::adxr::{adxr, AdxrInput, AdxrParams, AdxrData};
use my_project::indicators::alligator::{alligator, AlligatorInput, AlligatorParams};
use my_project::indicators::ao::{ao, AoInput, AoParams, AoData};
use my_project::indicators::apo::{apo, ApoInput, ApoParams};
use my_project::indicators::aroon::{aroon, AroonInput, AroonParams, AroonData};
use my_project::indicators::aroonosc::{aroon_osc, AroonOscInput, AroonOscParams, AroonOscData};
use my_project::indicators::atr::{atr, AtrInput, AtrParams, AtrData};
use my_project::indicators::bandpass::{bandpass, BandPassInput, BandPassParams};
use my_project::indicators::bollinger_bands::{bollinger_bands, BollingerBandsInput, BollingerBandsParams};
use my_project::indicators::bollinger_bands_width::{bollinger_bands_width, BollingerBandsWidthInput, BollingerBandsWidthParams};
use my_project::indicators::bop::{bop, BopInput, BopParams, BopData};
use my_project::utilities::data_loader::read_candles_from_csv;
use serde_json::json;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <indicator_name> [source]", args[0]);
        eprintln!("Available indicators: ad, acosc, adx, adosc, adxr, alligator, alma, ao, apo, aroon, aroonosc, atr, bandpass, bollinger_bands, bollinger_bands_width, bop, cwma, dema, edcf, ehlers_itrend, ema, epma, frama, fwma, gaussian, highpass_2_pole, highpass, hma, hwma, jma, jsa, kama, linreg, maaq, mama, mwdx, nma, pwma, reflex, sinwma, sma, smma, sqwma, srwma, supersmoother_3_pole, supersmoother, swma, tema, tilson, trendflex, trima, vwap, vwma, vpwma, wilders, wma, zlema");
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
        "frama" => {
            let params = FramaParams::default();
            let window = params.window.unwrap_or(10);
            let sc = params.sc.unwrap_or(200);
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
        "ao" => {
            if source != "high_low" {
                eprintln!("AO indicator requires 'high_low' source");
                std::process::exit(1);
            }
            let params = AoParams::default();
            let input = AoInput {
                data: AoData::Candles { candles: &candles, source: "hl2" },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
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
        },
        _ => {
            eprintln!("Unknown indicator: {}", indicator);
            std::process::exit(1);
        }
    };
    
    // Output as JSON
    println!("{}", serde_json::to_string_pretty(&output)?);
    
    Ok(())
}