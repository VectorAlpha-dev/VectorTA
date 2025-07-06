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
use my_project::utilities::data_loader::read_candles_from_csv;
use serde_json::json;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <indicator_name> [source]", args[0]);
        eprintln!("Available indicators: alma, cwma, dema, edcf, ehlers_itrend, ema, epma, frama, fwma, gaussian, highpass_2_pole, highpass, hma, hwma, jma, jsa, kama, linreg, maaq, mama");
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
        _ => {
            eprintln!("Unknown indicator: {}", indicator);
            std::process::exit(1);
        }
    };
    
    // Output as JSON
    println!("{}", serde_json::to_string_pretty(&output)?);
    
    Ok(())
}