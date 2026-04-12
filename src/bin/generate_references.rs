extern crate vector_ta;

use vector_ta::indicators::acosc::{acosc, AcoscData, AcoscInput, AcoscParams};
use vector_ta::indicators::ad::{ad, AdData, AdInput, AdParams};
use vector_ta::indicators::adaptive_bounds_rsi::{
    adaptive_bounds_rsi, AdaptiveBoundsRsiInput, AdaptiveBoundsRsiParams,
};
use vector_ta::indicators::adjustable_ma_alternating_extremities::{
    adjustable_ma_alternating_extremities, AdjustableMaAlternatingExtremitiesInput,
    AdjustableMaAlternatingExtremitiesParams,
};
use vector_ta::indicators::adosc::{adosc, AdoscData, AdoscInput, AdoscParams};
use vector_ta::indicators::adx::{adx, AdxData, AdxInput, AdxParams};
use vector_ta::indicators::adxr::{adxr, AdxrData, AdxrInput, AdxrParams};
use vector_ta::indicators::alligator::{alligator, AlligatorInput, AlligatorParams};
use vector_ta::indicators::ao::{ao, AoData, AoInput, AoParams};
use vector_ta::indicators::apo::{apo, ApoInput, ApoParams};
use vector_ta::indicators::aroon::{aroon, AroonData, AroonInput, AroonParams};
use vector_ta::indicators::aroonosc::{aroon_osc, AroonOscData, AroonOscInput, AroonOscParams};
use vector_ta::indicators::atr::{atr, AtrData, AtrInput, AtrParams};
use vector_ta::indicators::bandpass::{bandpass, BandPassInput, BandPassParams};
use vector_ta::indicators::bollinger_bands::{
    bollinger_bands, BollingerBandsInput, BollingerBandsParams,
};
use vector_ta::indicators::bollinger_bands_width::{
    bollinger_bands_width, BollingerBandsWidthInput, BollingerBandsWidthParams,
};
use vector_ta::indicators::bop::{bop, BopInput, BopParams};
use vector_ta::indicators::cci::{cci, CciInput, CciParams};
use vector_ta::indicators::cfo::{cfo, CfoInput, CfoParams};
use vector_ta::indicators::cg::{cg, CgInput, CgParams};
use vector_ta::indicators::chande::{chande, ChandeData, ChandeInput, ChandeParams};
use vector_ta::indicators::chop::{chop, ChopData, ChopInput, ChopParams};
use vector_ta::indicators::cmo::{cmo, CmoInput, CmoParams};
use vector_ta::indicators::correl_hl::{correl_hl, CorrelHlData, CorrelHlInput, CorrelHlParams};
use vector_ta::indicators::cvi::{cvi, CviInput, CviParams};

use serde_json::json;
use std::env;
use vector_ta::indicators::damiani_volatmeter::{
    damiani_volatmeter, DamianiVolatmeterInput, DamianiVolatmeterParams,
};
use vector_ta::indicators::decycler::{decycler, DecyclerInput, DecyclerParams};
use vector_ta::indicators::deviation::{deviation, DeviationInput, DeviationParams};
use vector_ta::indicators::devstop::{devstop, DevStopData, DevStopInput, DevStopParams};
use vector_ta::indicators::di::{di, DiData, DiInput, DiParams};
use vector_ta::indicators::dpo::{dpo, DpoInput, DpoParams};
use vector_ta::indicators::emv::{emv, EmvInput};
use vector_ta::indicators::er::{er, ErInput, ErParams};
use vector_ta::indicators::eri::{eri, EriData, EriInput, EriParams};
use vector_ta::indicators::fisher::{fisher, FisherInput, FisherParams};
use vector_ta::indicators::forward_backward_exponential_oscillator::{
    forward_backward_exponential_oscillator, ForwardBackwardExponentialOscillatorInput,
    ForwardBackwardExponentialOscillatorParams,
};
use vector_ta::indicators::kst::{kst, KstInput, KstParams};
use vector_ta::indicators::kurtosis::{kurtosis, KurtosisInput, KurtosisParams};
use vector_ta::indicators::linearreg_intercept::{
    linearreg_intercept, LinearRegInterceptInput, LinearRegInterceptParams,
};
use vector_ta::indicators::macz::{macz, MaczInput, MaczParams};
use vector_ta::indicators::market_structure_confluence::{
    market_structure_confluence, MarketStructureConfluenceInput, MarketStructureConfluenceParams,
};
use vector_ta::indicators::marketefi::{marketefi, MarketefiData, MarketefiInput, MarketefiParams};
use vector_ta::indicators::mass::{mass, MassInput, MassParams};
use vector_ta::indicators::mfi::{mfi, MfiData, MfiInput, MfiParams};
use vector_ta::indicators::midpoint::{midpoint, MidpointInput, MidpointParams};
use vector_ta::indicators::midprice::{midprice, MidpriceInput, MidpriceParams};
use vector_ta::indicators::moving_averages::alma::{alma, AlmaInput, AlmaParams};
use vector_ta::indicators::moving_averages::cwma::{cwma, CwmaInput, CwmaParams};
use vector_ta::indicators::moving_averages::dema::{dema, DemaInput, DemaParams};
use vector_ta::indicators::moving_averages::edcf::{edcf, EdcfInput, EdcfParams};
use vector_ta::indicators::moving_averages::ehlers_ecema::{
    ehlers_ecema, EhlersEcemaInput, EhlersEcemaParams,
};
use vector_ta::indicators::moving_averages::ehlers_itrend::{
    ehlers_itrend, EhlersITrendInput, EhlersITrendParams,
};
use vector_ta::indicators::moving_averages::ema::{ema, EmaInput, EmaParams};
use vector_ta::indicators::moving_averages::epma::{epma, EpmaInput, EpmaParams};
use vector_ta::indicators::moving_averages::frama::{frama, FramaInput, FramaParams};
use vector_ta::indicators::moving_averages::fwma::{fwma, FwmaInput, FwmaParams};
use vector_ta::indicators::moving_averages::gaussian::{gaussian, GaussianInput, GaussianParams};
use vector_ta::indicators::moving_averages::highpass::{highpass, HighPassInput, HighPassParams};
use vector_ta::indicators::moving_averages::highpass_2_pole::{
    highpass_2_pole, HighPass2Input, HighPass2Params,
};
use vector_ta::indicators::moving_averages::hma::{hma, HmaInput, HmaParams};
use vector_ta::indicators::moving_averages::hwma::{hwma, HwmaInput, HwmaParams};
use vector_ta::indicators::moving_averages::jma::{jma, JmaInput, JmaParams};
use vector_ta::indicators::moving_averages::jsa::{jsa, JsaInput, JsaParams};
use vector_ta::indicators::moving_averages::kama::{kama, KamaInput, KamaParams};
use vector_ta::indicators::moving_averages::linreg::{linreg, LinRegInput, LinRegParams};
use vector_ta::indicators::moving_averages::maaq::{maaq, MaaqInput, MaaqParams};
use vector_ta::indicators::moving_averages::mama::{mama, MamaInput, MamaParams};
use vector_ta::indicators::moving_averages::mwdx::{mwdx, MwdxInput, MwdxParams};
use vector_ta::indicators::moving_averages::nma::{nma, NmaInput, NmaParams};
use vector_ta::indicators::moving_averages::pwma::{pwma, PwmaInput, PwmaParams};
use vector_ta::indicators::moving_averages::reflex::{reflex, ReflexInput, ReflexParams};
use vector_ta::indicators::moving_averages::sama::{sama, SamaInput, SamaParams};
use vector_ta::indicators::moving_averages::sinwma::{sinwma, SinWmaInput, SinWmaParams};
use vector_ta::indicators::moving_averages::sma::{sma, SmaInput, SmaParams};
use vector_ta::indicators::moving_averages::smma::{smma, SmmaInput, SmmaParams};
use vector_ta::indicators::moving_averages::sqwma::{sqwma, SqwmaInput, SqwmaParams};
use vector_ta::indicators::moving_averages::srwma::{srwma, SrwmaInput, SrwmaParams};
use vector_ta::indicators::moving_averages::supersmoother::{
    supersmoother, SuperSmootherInput, SuperSmootherParams,
};
use vector_ta::indicators::moving_averages::supersmoother_3_pole::{
    supersmoother_3_pole, SuperSmoother3PoleInput, SuperSmoother3PoleParams,
};
use vector_ta::indicators::moving_averages::swma::{swma, SwmaInput, SwmaParams};
use vector_ta::indicators::moving_averages::tema::{tema, TemaInput, TemaParams};
use vector_ta::indicators::moving_averages::tilson::{tilson, TilsonInput, TilsonParams};
use vector_ta::indicators::moving_averages::trendflex::{
    trendflex, TrendFlexInput, TrendFlexParams,
};
use vector_ta::indicators::moving_averages::trima::{trima, TrimaInput, TrimaParams};
use vector_ta::indicators::moving_averages::volatility_adjusted_ma::{vama, VamaInput, VamaParams};
use vector_ta::indicators::moving_averages::volume_adjusted_ma::{
    VolumeAdjustedMa as volu_ma, VolumeAdjustedMaInput as VoluMaInput,
    VolumeAdjustedMaParams as VoluMaParams,
};
use vector_ta::indicators::moving_averages::vpwma::{vpwma, VpwmaInput, VpwmaParams};
use vector_ta::indicators::moving_averages::vwap::{vwap, VwapInput, VwapParams};
use vector_ta::indicators::moving_averages::vwma::{vwma, VwmaInput, VwmaParams};
use vector_ta::indicators::moving_averages::wilders::{wilders, WildersInput, WildersParams};
use vector_ta::indicators::moving_averages::wma::{wma, WmaInput, WmaParams};
use vector_ta::indicators::moving_averages::zlema::{zlema, ZlemaInput, ZlemaParams};
use vector_ta::indicators::pma::{pma, PmaInput, PmaParams};
use vector_ta::indicators::ppo::{ppo, PpoInput, PpoParams};
use vector_ta::indicators::qqe_weighted_oscillator::{
    qqe_weighted_oscillator, QqeWeightedOscillatorInput, QqeWeightedOscillatorParams,
};
use vector_ta::indicators::range_filtered_trend_signals::{
    range_filtered_trend_signals, RangeFilteredTrendSignalsInput, RangeFilteredTrendSignalsParams,
};
use vector_ta::indicators::range_oscillator::{
    range_oscillator, RangeOscillatorInput, RangeOscillatorParams,
};
use vector_ta::indicators::roc::{roc, RocInput, RocParams};
use vector_ta::indicators::rocp::{rocp, RocpInput, RocpParams};
use vector_ta::indicators::rsi::{rsi, RsiInput, RsiParams};
use vector_ta::indicators::rsx::{rsx, RsxInput, RsxParams};
use vector_ta::indicators::rvi::{rvi, RviInput, RviParams};
use vector_ta::indicators::squeeze_momentum::{
    squeeze_momentum, SqueezeMomentumInput, SqueezeMomentumParams,
};
use vector_ta::indicators::stddev::{stddev, StdDevInput, StdDevParams};
use vector_ta::indicators::tsf::{tsf, TsfInput, TsfParams};
use vector_ta::indicators::ui::{ui, UiInput, UiParams};
use vector_ta::indicators::var::{var, VarInput, VarParams};
use vector_ta::indicators::volume_weighted_relative_strength_index::{
    volume_weighted_relative_strength_index, VolumeWeightedRelativeStrengthIndexInput,
    VolumeWeightedRelativeStrengthIndexParams,
};
use vector_ta::indicators::vpci::{vpci, VpciInput, VpciParams};
use vector_ta::indicators::vpt::{vpt, VptInput};
use vector_ta::indicators::wclprice::{wclprice, WclpriceInput};
use vector_ta::utilities::data_loader::read_candles_from_csv;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <indicator_name> [source]", args[0]);
        eprintln!("Available indicators: ad, acosc, adx, adosc, adxr, adaptive_bounds_rsi, adjustable_ma_alternating_extremities, alligator, alma, ao, apo, aroon, aroonosc, atr, bandpass, bollinger_bands, bollinger_bands_width, bop, cci, cfo, cg, chop, cwma, decycler, dema, devstop, di, edcf, ehlers_itrend, ema, epma, eri, fisher, forward_backward_exponential_oscillator, frama, fwma, gaussian, highpass_2_pole, highpass, hma, hwma, jma, jsa, kama, kst, kurtosis, linreg, maaq, macz, mama, market_structure_confluence, marketefi, midpoint, midprice, mfi, mwdx, nma, pma, ppo, qqe_weighted_oscillator, range_filtered_trend_signals, range_oscillator, rsx, pwma, reflex, roc, rocp, rsi, rvi, rvi, sama, sinwma, sma, smma, squeeze_momentum, sqwma, srwma, stddev, supersmoother_3_pole, supersmoother, swma, tema, tilson, trendflex, trima, var, volume_weighted_relative_strength_index, vpci, tsf, ui, vwap, vwma, vpwma, wclprice, wilders, wma, zlema");
        eprintln!("Available sources: open, high, low, close, volume, hl2, hlc3, ohlc4, hlcc4");
        std::process::exit(1);
    }

    let indicator = &args[1];
    let source = args.get(2).map(|s| s.as_str()).unwrap_or("close");

    let candles = read_candles_from_csv("src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv")?;

    let output = match indicator.as_str() {
        "adaptive_bounds_rsi" => {
            let params = AdaptiveBoundsRsiParams::default();
            let rsi_length = params.rsi_length.unwrap_or(14);
            let alpha = params.alpha.unwrap_or(0.1);
            let input = AdaptiveBoundsRsiInput::from_candles(&candles, source, params);
            let result = adaptive_bounds_rsi(&input)?;
            json!({
                "indicator": "adaptive_bounds_rsi",
                "source": source,
                "params": {
                    "rsi_length": rsi_length,
                    "alpha": alpha
                },
                "rsi": result.rsi,
                "lower_bound": result.lower_bound,
                "lower_mid": result.lower_mid,
                "mid": result.mid,
                "upper_mid": result.upper_mid,
                "upper_bound": result.upper_bound,
                "regime": result.regime,
                "regime_flip": result.regime_flip,
                "lower_signal": result.lower_signal,
                "upper_signal": result.upper_signal,
                "length": candles.close.len()
            })
        }
        "adjustable_ma_alternating_extremities" => {
            let params = AdjustableMaAlternatingExtremitiesParams::default();
            let length = params.length.unwrap_or(50);
            let mult = params.mult.unwrap_or(2.0);
            let alpha = params.alpha.unwrap_or(1.0);
            let beta = params.beta.unwrap_or(0.5);
            let input = AdjustableMaAlternatingExtremitiesInput::from_slices(
                candles.select_candle_field("high")?,
                candles.select_candle_field("low")?,
                candles.select_candle_field("close")?,
                params,
            );
            let result = adjustable_ma_alternating_extremities(&input)?;
            json!({
                "indicator": "adjustable_ma_alternating_extremities",
                "source": "ohlc",
                "params": {
                    "length": length,
                    "mult": mult,
                    "alpha": alpha,
                    "beta": beta
                },
                "ma": result.ma,
                "upper": result.upper,
                "lower": result.lower,
                "extremity": result.extremity,
                "state": result.state,
                "changed": result.changed,
                "smoothed_open": result.smoothed_open,
                "smoothed_high": result.smoothed_high,
                "smoothed_low": result.smoothed_low,
                "smoothed_close": result.smoothed_close,
                "length": candles.close.len()
            })
        }
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
        "qqe_weighted_oscillator" => {
            let params = QqeWeightedOscillatorParams::default();
            let length = params.length.unwrap_or(14);
            let factor = params.factor.unwrap_or(4.236);
            let smooth = params.smooth.unwrap_or(5);
            let weight = params.weight.unwrap_or(2.0);
            let input = QqeWeightedOscillatorInput::from_candles(&candles, source, params);
            let result = qqe_weighted_oscillator(&input)?;
            json!({
                "indicator": "qqe_weighted_oscillator",
                "source": source,
                "params": {
                    "length": length,
                    "factor": factor,
                    "smooth": smooth,
                    "weight": weight
                },
                "rsi": result.rsi,
                "trailing_stop": result.trailing_stop,
                "length": candles.close.len()
            })
        }
        "range_oscillator" => {
            let params = RangeOscillatorParams::default();
            let length = params.length.unwrap_or(50);
            let mult = params.mult.unwrap_or(2.0);
            let input = RangeOscillatorInput::from_slices(
                candles.select_candle_field("high")?,
                candles.select_candle_field("low")?,
                candles.select_candle_field("close")?,
                params,
            );
            let result = range_oscillator(&input)?;
            json!({
                "indicator": "range_oscillator",
                "source": "ohlc",
                "params": {
                    "length": length,
                    "mult": mult
                },
                "oscillator": result.oscillator,
                "ma": result.ma,
                "upper_band": result.upper_band,
                "lower_band": result.lower_band,
                "range_width": result.range_width,
                "in_range": result.in_range,
                "trend": result.trend,
                "break_up": result.break_up,
                "break_down": result.break_down,
                "length": candles.close.len()
            })
        }
        "market_structure_confluence" => {
            let params = MarketStructureConfluenceParams::default();
            let swing_size = params.swing_size.unwrap_or(10);
            let bos_confirmation = params
                .bos_confirmation
                .clone()
                .unwrap_or_else(|| "Candle Close".to_string());
            let basis_length = params.basis_length.unwrap_or(100);
            let atr_length = params.atr_length.unwrap_or(14);
            let atr_smooth = params.atr_smooth.unwrap_or(21);
            let vol_mult = params.vol_mult.unwrap_or(2.0);
            let input = MarketStructureConfluenceInput::from_slices(
                candles.select_candle_field("high")?,
                candles.select_candle_field("low")?,
                candles.select_candle_field("close")?,
                params,
            );
            let result = market_structure_confluence(&input)?;
            json!({
                "indicator": "market_structure_confluence",
                "source": "ohlc",
                "params": {
                    "swing_size": swing_size,
                    "bos_confirmation": bos_confirmation,
                    "basis_length": basis_length,
                    "atr_length": atr_length,
                    "atr_smooth": atr_smooth,
                    "vol_mult": vol_mult
                },
                "basis": result.basis,
                "upper_band": result.upper_band,
                "lower_band": result.lower_band,
                "structure_direction": result.structure_direction,
                "bullish_arrow": result.bullish_arrow,
                "bearish_arrow": result.bearish_arrow,
                "bullish_change": result.bullish_change,
                "bearish_change": result.bearish_change,
                "hh": result.hh,
                "lh": result.lh,
                "hl": result.hl,
                "ll": result.ll,
                "bullish_bos": result.bullish_bos,
                "bullish_choch": result.bullish_choch,
                "bearish_bos": result.bearish_bos,
                "bearish_choch": result.bearish_choch,
                "length": candles.close.len()
            })
        }
        "range_filtered_trend_signals" => {
            let params = RangeFilteredTrendSignalsParams::default();
            let kalman_alpha = params.kalman_alpha.unwrap_or(0.01);
            let kalman_beta = params.kalman_beta.unwrap_or(0.1);
            let kalman_period = params.kalman_period.unwrap_or(77);
            let dev = params.dev.unwrap_or(1.2);
            let supertrend_factor = params.supertrend_factor.unwrap_or(0.7);
            let supertrend_atr_period = params.supertrend_atr_period.unwrap_or(7);
            let input = RangeFilteredTrendSignalsInput::from_slices(
                candles.select_candle_field("high")?,
                candles.select_candle_field("low")?,
                candles.select_candle_field("close")?,
                params,
            );
            let result = range_filtered_trend_signals(&input)?;
            json!({
                "indicator": "range_filtered_trend_signals",
                "source": "ohlc",
                "params": {
                    "kalman_alpha": kalman_alpha,
                    "kalman_beta": kalman_beta,
                    "kalman_period": kalman_period,
                    "dev": dev,
                    "supertrend_factor": supertrend_factor,
                    "supertrend_atr_period": supertrend_atr_period
                },
                "kalman": result.kalman,
                "supertrend": result.supertrend,
                "upper_band": result.upper_band,
                "lower_band": result.lower_band,
                "trend": result.trend,
                "kalman_trend": result.kalman_trend,
                "state": result.state,
                "market_trending": result.market_trending,
                "market_ranging": result.market_ranging,
                "short_term_bullish": result.short_term_bullish,
                "short_term_bearish": result.short_term_bearish,
                "long_term_bullish": result.long_term_bullish,
                "long_term_bearish": result.long_term_bearish,
                "length": candles.close.len()
            })
        }
        "volume_weighted_relative_strength_index" => {
            if source != "close_volume" {
                eprintln!("volume_weighted_relative_strength_index requires 'close_volume' source");
                std::process::exit(1);
            }
            let params = VolumeWeightedRelativeStrengthIndexParams::default();
            let rsi_length = params.rsi_length.unwrap_or(14);
            let range_length = params.range_length.unwrap_or(10);
            let ma_length = params.ma_length.unwrap_or(14);
            let ma_type = params.ma_type.clone().unwrap_or_else(|| "EMA".to_string());
            let input = VolumeWeightedRelativeStrengthIndexInput::from_slices(
                candles.select_candle_field("close")?,
                candles.select_candle_field("volume")?,
                params,
            );
            let result = volume_weighted_relative_strength_index(&input)?;
            json!({
                "indicator": "volume_weighted_relative_strength_index",
                "source": "close_volume",
                "params": {
                    "rsi_length": rsi_length,
                    "range_length": range_length,
                    "ma_length": ma_length,
                    "ma_type": ma_type
                },
                "rsi": result.rsi,
                "consolidation_strength": result.consolidation_strength,
                "rsi_ma": result.rsi_ma,
                "bearish_tp": result.bearish_tp,
                "bullish_tp": result.bullish_tp,
                "length": candles.close.len()
            })
        }
        "forward_backward_exponential_oscillator" => {
            let params = ForwardBackwardExponentialOscillatorParams::default();
            let length = params.length.unwrap_or(20);
            let smooth = params.smooth.unwrap_or(10);
            let input =
                ForwardBackwardExponentialOscillatorInput::from_candles(&candles, source, params);
            let result = forward_backward_exponential_oscillator(&input)?;
            json!({
                "indicator": "forward_backward_exponential_oscillator",
                "source": source,
                "params": {
                    "length": length,
                    "smooth": smooth
                },
                "forward_backward": result.forward_backward,
                "backward": result.backward,
                "histogram": result.histogram,
                "length": candles.close.len()
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
                "source": "high,low",
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

            let window = params.window.unwrap_or(10);
            let sc = params.sc.unwrap_or(300);
            let fc = params.fc.unwrap_or(1);
            let input = FramaInput::from_candles(&candles, params);
            let result = frama(&input)?;
            json!({
                "indicator": "frama",
                "source": "high,low,close",
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
            if source != "hlc3_volume" {
                eprintln!("MFI indicator requires 'hlc3_volume' source");
                std::process::exit(1);
            }
            let params = MfiParams::default();
            let period = params.period.unwrap_or(14);

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

    println!("{}", serde_json::to_string_pretty(&output)?);

    Ok(())
}
