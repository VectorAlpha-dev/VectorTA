use std::os::raw::{c_double, c_int};
use std::slice;
use my_project::indicators::moving_averages::{
    sma, ema, dema, tema, wma, kama, trima, hma, vwma, wilders, zlema, linreg
};
use my_project::indicators::{
    rsi, atr, bollinger_bands, macd, adx, cci, stoch, aroon,
    apo, cmo, dpo, mom, ppo, roc, willr, ad, adosc, obv, mfi,
    ao, bop, natr, stddev, var, ultosc, adxr, aroonosc, di, dm,
    dx, rocp, rocr, srsi, stochf, trix, fisher, fosc, kvo, mass,
    medprice, midpoint, midprice, nvi, pvi, qstick, sar, tsf, linearreg_angle,
    vidya, vosc, wad, wclprice
};
use my_project::utilities::data_loader::Candles;

/// FFI wrapper for Rust SMA indicator
#[no_mangle]
pub unsafe extern "C" fn rust_sma(
    size: c_int,
    input: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if input.is_null() || output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = sma::SmaParams {
        period: Some(period as usize),
    };

    let sma_input = sma::SmaInput::from_slice(input_slice, params);

    match sma::sma(&sma_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust EMA indicator
#[no_mangle]
pub unsafe extern "C" fn rust_ema(
    size: c_int,
    input: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if input.is_null() || output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = ema::EmaParams {
        period: Some(period as usize),
    };

    let ema_input = ema::EmaInput::from_slice(input_slice, params);

    match ema::ema(&ema_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust RSI indicator
#[no_mangle]
pub unsafe extern "C" fn rust_rsi(
    size: c_int,
    input: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if input.is_null() || output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = rsi::RsiParams {
        period: Some(period as usize),
    };

    let rsi_input = rsi::RsiInput::from_slice(input_slice, params);

    match rsi::rsi(&rsi_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust ATR indicator
#[no_mangle]
pub unsafe extern "C" fn rust_atr(
    size: c_int,
    high: *const c_double,
    low: *const c_double,
    close: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if high.is_null() || low.is_null() || close.is_null() || output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let high_slice = slice::from_raw_parts(high, size as usize);
    let low_slice = slice::from_raw_parts(low, size as usize);
    let close_slice = slice::from_raw_parts(close, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = atr::AtrParams {
        length: Some(period as usize),
    };

    // ATR needs a Candles structure, we'll create a minimal one
    let mut hl2 = vec![0.0; size as usize];
    let mut hlc3 = vec![0.0; size as usize];
    let mut ohlc4 = vec![0.0; size as usize];
    let mut hlcc4 = vec![0.0; size as usize];

    for i in 0..size as usize {
        hl2[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlc3[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 3.0;
        ohlc4[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 4.0; // Using close as open substitute
        hlcc4[i] = (high_slice[i] + low_slice[i] + close_slice[i] + close_slice[i]) / 4.0;
    }

    let candles = Candles {
        high: high_slice.to_vec(),
        low: low_slice.to_vec(),
        close: close_slice.to_vec(),
        open: vec![0.0; size as usize], // ATR doesn't use open
        volume: vec![0.0; size as usize], // ATR doesn't use volume
        timestamp: vec![0; size as usize],
        hl2,
        hlc3,
        ohlc4,
        hlcc4,
    };

    let atr_input = atr::AtrInput::from_candles(&candles, params);

    match atr::atr(&atr_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust Bollinger Bands indicator
#[no_mangle]
pub unsafe extern "C" fn rust_bbands(
    size: c_int,
    input: *const c_double,
    period: c_int,
    stddev: c_double,
    output_lower: *mut c_double,
    output_middle: *mut c_double,
    output_upper: *mut c_double,
) -> c_int {
    if input.is_null() || output_lower.is_null() || output_middle.is_null() ||
       output_upper.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let lower_slice = slice::from_raw_parts_mut(output_lower, size as usize);
    let middle_slice = slice::from_raw_parts_mut(output_middle, size as usize);
    let upper_slice = slice::from_raw_parts_mut(output_upper, size as usize);

    let params = bollinger_bands::BollingerBandsParams {
        period: Some(period as usize),
        devup: Some(stddev),
        devdn: Some(stddev),
        matype: Some("sma".to_string()),
        devtype: Some(0),
    };

    let bb_input = bollinger_bands::BollingerBandsInput::from_slice(input_slice, params);

    match bollinger_bands::bollinger_bands(&bb_input) {
        Ok(result) => {
            lower_slice.copy_from_slice(&result.lower_band);
            middle_slice.copy_from_slice(&result.middle_band);
            upper_slice.copy_from_slice(&result.upper_band);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust MACD indicator
#[no_mangle]
pub unsafe extern "C" fn rust_macd(
    size: c_int,
    input: *const c_double,
    short_period: c_int,
    long_period: c_int,
    signal_period: c_int,
    output_macd: *mut c_double,
    output_signal: *mut c_double,
    output_histogram: *mut c_double,
) -> c_int {
    if input.is_null() || output_macd.is_null() || output_signal.is_null() ||
       output_histogram.is_null() || size <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let macd_slice = slice::from_raw_parts_mut(output_macd, size as usize);
    let signal_slice = slice::from_raw_parts_mut(output_signal, size as usize);
    let histogram_slice = slice::from_raw_parts_mut(output_histogram, size as usize);

    let params = macd::MacdParams {
        fast_period: Some(short_period as usize),
        slow_period: Some(long_period as usize),
        signal_period: Some(signal_period as usize),
        ma_type: Some("ema".to_string()),
    };

    let macd_input = macd::MacdInput::from_slice(input_slice, params);

    match macd::macd(&macd_input) {
        Ok(result) => {
            macd_slice.copy_from_slice(&result.macd);
            signal_slice.copy_from_slice(&result.signal);
            histogram_slice.copy_from_slice(&result.hist);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust ADX indicator
#[no_mangle]
pub unsafe extern "C" fn rust_adx(
    size: c_int,
    high: *const c_double,
    low: *const c_double,
    close: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if high.is_null() || low.is_null() || close.is_null() || output.is_null() ||
       size <= 0 || period <= 0 {
        return -1;
    }

    let high_slice = slice::from_raw_parts(high, size as usize);
    let low_slice = slice::from_raw_parts(low, size as usize);
    let close_slice = slice::from_raw_parts(close, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = adx::AdxParams {
        period: Some(period as usize),
    };

    // ADX needs a Candles structure
    let mut hl2 = vec![0.0; size as usize];
    let mut hlc3 = vec![0.0; size as usize];
    let mut ohlc4 = vec![0.0; size as usize];
    let mut hlcc4 = vec![0.0; size as usize];

    for i in 0..size as usize {
        hl2[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlc3[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 3.0;
        ohlc4[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 4.0;
        hlcc4[i] = (high_slice[i] + low_slice[i] + close_slice[i] + close_slice[i]) / 4.0;
    }

    let candles = Candles {
        high: high_slice.to_vec(),
        low: low_slice.to_vec(),
        close: close_slice.to_vec(),
        open: vec![0.0; size as usize],
        volume: vec![0.0; size as usize],
        timestamp: vec![0; size as usize],
        hl2,
        hlc3,
        ohlc4,
        hlcc4,
    };

    let adx_input = adx::AdxInput::from_candles(&candles, params);

    match adx::adx(&adx_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust CCI indicator
#[no_mangle]
pub unsafe extern "C" fn rust_cci(
    size: c_int,
    high: *const c_double,
    low: *const c_double,
    close: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if high.is_null() || low.is_null() || close.is_null() || output.is_null() ||
       size <= 0 || period <= 0 {
        return -1;
    }

    let high_slice = slice::from_raw_parts(high, size as usize);
    let low_slice = slice::from_raw_parts(low, size as usize);
    let close_slice = slice::from_raw_parts(close, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = cci::CciParams {
        period: Some(period as usize),
    };

    // CCI needs a Candles structure for typical price calculation
    let mut hl2 = vec![0.0; size as usize];
    let mut hlc3 = vec![0.0; size as usize];
    let mut ohlc4 = vec![0.0; size as usize];
    let mut hlcc4 = vec![0.0; size as usize];

    for i in 0..size as usize {
        hl2[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlc3[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 3.0;
        ohlc4[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 4.0;
        hlcc4[i] = (high_slice[i] + low_slice[i] + close_slice[i] + close_slice[i]) / 4.0;
    }

    let candles = Candles {
        high: high_slice.to_vec(),
        low: low_slice.to_vec(),
        close: close_slice.to_vec(),
        open: vec![0.0; size as usize],
        volume: vec![0.0; size as usize],
        timestamp: vec![0; size as usize],
        hl2,
        hlc3,
        ohlc4,
        hlcc4,
    };

    let cci_input = cci::CciInput::from_candles(&candles, "hlc3", params);

    match cci::cci(&cci_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust DEMA indicator
#[no_mangle]
pub unsafe extern "C" fn rust_dema(
    size: c_int,
    input: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if input.is_null() || output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = dema::DemaParams {
        period: Some(period as usize),
    };

    let dema_input = dema::DemaInput::from_slice(input_slice, params);

    match dema::dema(&dema_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust TEMA indicator
#[no_mangle]
pub unsafe extern "C" fn rust_tema(
    size: c_int,
    input: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if input.is_null() || output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = tema::TemaParams {
        period: Some(period as usize),
    };

    let tema_input = tema::TemaInput::from_slice(input_slice, params);

    match tema::tema(&tema_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust WMA indicator
#[no_mangle]
pub unsafe extern "C" fn rust_wma(
    size: c_int,
    input: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if input.is_null() || output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = wma::WmaParams {
        period: Some(period as usize),
    };

    let wma_input = wma::WmaInput::from_slice(input_slice, params);

    match wma::wma(&wma_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust KAMA indicator
#[no_mangle]
pub unsafe extern "C" fn rust_kama(
    size: c_int,
    input: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if input.is_null() || output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = kama::KamaParams {
        period: Some(period as usize),
    };

    let kama_input = kama::KamaInput::from_slice(input_slice, params);

    match kama::kama(&kama_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust TRIMA indicator
#[no_mangle]
pub unsafe extern "C" fn rust_trima(
    size: c_int,
    input: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if input.is_null() || output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = trima::TrimaParams {
        period: Some(period as usize),
    };

    let trima_input = trima::TrimaInput::from_slice(input_slice, params);

    match trima::trima(&trima_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust HMA indicator
#[no_mangle]
pub unsafe extern "C" fn rust_hma(
    size: c_int,
    input: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if input.is_null() || output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = hma::HmaParams {
        period: Some(period as usize),
    };

    let hma_input = hma::HmaInput::from_slice(input_slice, params);

    match hma::hma(&hma_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust Stochastic indicator
#[no_mangle]
pub unsafe extern "C" fn rust_stoch(
    size: c_int,
    high: *const c_double,
    low: *const c_double,
    close: *const c_double,
    k_period: c_int,
    k_smooth: c_int,
    d_smooth: c_int,
    output_k: *mut c_double,
    output_d: *mut c_double,
) -> c_int {
    if high.is_null() || low.is_null() || close.is_null() ||
       output_k.is_null() || output_d.is_null() || size <= 0 {
        return -1;
    }

    let high_slice = slice::from_raw_parts(high, size as usize);
    let low_slice = slice::from_raw_parts(low, size as usize);
    let close_slice = slice::from_raw_parts(close, size as usize);
    let k_slice = slice::from_raw_parts_mut(output_k, size as usize);
    let d_slice = slice::from_raw_parts_mut(output_d, size as usize);

    let params = stoch::StochParams {
        fastk_period: Some(k_period as usize),
        slowk_period: Some(k_smooth as usize),
        slowk_ma_type: Some("sma".to_string()),
        slowd_period: Some(d_smooth as usize),
        slowd_ma_type: Some("sma".to_string()),
    };

    // Create Candles structure
    let mut hl2 = vec![0.0; size as usize];
    let mut hlc3 = vec![0.0; size as usize];
    let mut ohlc4 = vec![0.0; size as usize];
    let mut hlcc4 = vec![0.0; size as usize];

    for i in 0..size as usize {
        hl2[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlc3[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 3.0;
        ohlc4[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 4.0;
        hlcc4[i] = (high_slice[i] + low_slice[i] + close_slice[i] + close_slice[i]) / 4.0;
    }

    let candles = Candles {
        high: high_slice.to_vec(),
        low: low_slice.to_vec(),
        close: close_slice.to_vec(),
        open: vec![0.0; size as usize],
        volume: vec![0.0; size as usize],
        timestamp: vec![0; size as usize],
        hl2,
        hlc3,
        ohlc4,
        hlcc4,
    };

    let stoch_input = stoch::StochInput::from_candles(&candles, params);

    match stoch::stoch(&stoch_input) {
        Ok(result) => {
            k_slice.copy_from_slice(&result.k);
            d_slice.copy_from_slice(&result.d);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust AD (Accumulation/Distribution) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_ad(
    size: c_int,
    high: *const c_double,
    low: *const c_double,
    close: *const c_double,
    volume: *const c_double,
    output: *mut c_double,
) -> c_int {
    if high.is_null() || low.is_null() || close.is_null() || volume.is_null() ||
       output.is_null() || size <= 0 {
        return -1;
    }

    let high_slice = slice::from_raw_parts(high, size as usize);
    let low_slice = slice::from_raw_parts(low, size as usize);
    let close_slice = slice::from_raw_parts(close, size as usize);
    let volume_slice = slice::from_raw_parts(volume, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    // Create Candles structure
    let mut hl2 = vec![0.0; size as usize];
    let mut hlc3 = vec![0.0; size as usize];
    let mut ohlc4 = vec![0.0; size as usize];
    let mut hlcc4 = vec![0.0; size as usize];

    for i in 0..size as usize {
        hl2[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlc3[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 3.0;
        ohlc4[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 4.0;
        hlcc4[i] = (high_slice[i] + low_slice[i] + close_slice[i] + close_slice[i]) / 4.0;
    }

    let candles = Candles {
        high: high_slice.to_vec(),
        low: low_slice.to_vec(),
        close: close_slice.to_vec(),
        open: vec![0.0; size as usize],
        volume: volume_slice.to_vec(),
        timestamp: vec![0; size as usize],
        hl2,
        hlc3,
        ohlc4,
        hlcc4,
    };

    let ad_input = ad::AdInput::from_candles(&candles, ad::AdParams {});

    match ad::ad(&ad_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust ADOSC (Chaikin A/D Oscillator) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_adosc(
    size: c_int,
    high: *const c_double,
    low: *const c_double,
    close: *const c_double,
    volume: *const c_double,
    fast_period: c_int,
    slow_period: c_int,
    output: *mut c_double,
) -> c_int {
    if high.is_null() || low.is_null() || close.is_null() || volume.is_null() ||
       output.is_null() || size <= 0 {
        return -1;
    }

    let high_slice = slice::from_raw_parts(high, size as usize);
    let low_slice = slice::from_raw_parts(low, size as usize);
    let close_slice = slice::from_raw_parts(close, size as usize);
    let volume_slice = slice::from_raw_parts(volume, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = adosc::AdoscParams {
        short_period: Some(fast_period as usize),
        long_period: Some(slow_period as usize),
    };

    // Create Candles structure
    let mut hl2 = vec![0.0; size as usize];
    let mut hlc3 = vec![0.0; size as usize];
    let mut ohlc4 = vec![0.0; size as usize];
    let mut hlcc4 = vec![0.0; size as usize];

    for i in 0..size as usize {
        hl2[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlc3[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 3.0;
        ohlc4[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 4.0;
        hlcc4[i] = (high_slice[i] + low_slice[i] + close_slice[i] + close_slice[i]) / 4.0;
    }

    let candles = Candles {
        high: high_slice.to_vec(),
        low: low_slice.to_vec(),
        close: close_slice.to_vec(),
        open: vec![0.0; size as usize],
        volume: volume_slice.to_vec(),
        timestamp: vec![0; size as usize],
        hl2,
        hlc3,
        ohlc4,
        hlcc4,
    };

    let adosc_input = adosc::AdoscInput::from_candles(&candles, params);

    match adosc::adosc(&adosc_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust ADXR (Average Directional Movement Index Rating) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_adxr(
    size: c_int,
    high: *const c_double,
    low: *const c_double,
    close: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if high.is_null() || low.is_null() || close.is_null() || output.is_null() ||
       size <= 0 || period <= 0 {
        return -1;
    }

    let high_slice = slice::from_raw_parts(high, size as usize);
    let low_slice = slice::from_raw_parts(low, size as usize);
    let close_slice = slice::from_raw_parts(close, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = adxr::AdxrParams {
        period: Some(period as usize),
    };

    // Create Candles structure
    let mut hl2 = vec![0.0; size as usize];
    let mut hlc3 = vec![0.0; size as usize];
    let mut ohlc4 = vec![0.0; size as usize];
    let mut hlcc4 = vec![0.0; size as usize];

    for i in 0..size as usize {
        hl2[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlc3[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 3.0;
        ohlc4[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 4.0;
        hlcc4[i] = (high_slice[i] + low_slice[i] + close_slice[i] + close_slice[i]) / 4.0;
    }

    let candles = Candles {
        high: high_slice.to_vec(),
        low: low_slice.to_vec(),
        close: close_slice.to_vec(),
        open: vec![0.0; size as usize],
        volume: vec![0.0; size as usize],
        timestamp: vec![0; size as usize],
        hl2,
        hlc3,
        ohlc4,
        hlcc4,
    };

    let adxr_input = adxr::AdxrInput::from_candles(&candles, params);

    match adxr::adxr(&adxr_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust AO (Awesome Oscillator) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_ao(
    size: c_int,
    high: *const c_double,
    low: *const c_double,
    output: *mut c_double,
) -> c_int {
    if high.is_null() || low.is_null() || output.is_null() || size <= 0 {
        return -1;
    }

    let high_slice = slice::from_raw_parts(high, size as usize);
    let low_slice = slice::from_raw_parts(low, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = ao::AoParams {
        short_period: Some(5),
        long_period: Some(34),
    };

    // Create Candles structure
    let mut hl2 = vec![0.0; size as usize];
    let mut hlc3 = vec![0.0; size as usize];
    let mut ohlc4 = vec![0.0; size as usize];
    let mut hlcc4 = vec![0.0; size as usize];

    for i in 0..size as usize {
        hl2[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlc3[i] = (high_slice[i] + low_slice[i]) / 2.0;
        ohlc4[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlcc4[i] = (high_slice[i] + low_slice[i]) / 2.0;
    }

    let candles = Candles {
        high: high_slice.to_vec(),
        low: low_slice.to_vec(),
        close: vec![0.0; size as usize],
        open: vec![0.0; size as usize],
        volume: vec![0.0; size as usize],
        timestamp: vec![0; size as usize],
        hl2,
        hlc3,
        ohlc4,
        hlcc4,
    };

    let ao_input = ao::AoInput::from_candles(&candles, "hl2", params);

    match ao::ao(&ao_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust APO (Absolute Price Oscillator) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_apo(
    size: c_int,
    input: *const c_double,
    fast_period: c_int,
    slow_period: c_int,
    output: *mut c_double,
) -> c_int {
    if input.is_null() || output.is_null() || size <= 0 || fast_period <= 0 || slow_period <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = apo::ApoParams {
        short_period: Some(fast_period as usize),
        long_period: Some(slow_period as usize),
    };

    let apo_input = apo::ApoInput::from_slice(input_slice, params);

    match apo::apo(&apo_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust Aroon indicator
#[no_mangle]
pub unsafe extern "C" fn rust_aroon(
    size: c_int,
    high: *const c_double,
    low: *const c_double,
    period: c_int,
    output_down: *mut c_double,
    output_up: *mut c_double,
) -> c_int {
    if high.is_null() || low.is_null() || output_down.is_null() ||
       output_up.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let high_slice = slice::from_raw_parts(high, size as usize);
    let low_slice = slice::from_raw_parts(low, size as usize);
    let down_slice = slice::from_raw_parts_mut(output_down, size as usize);
    let up_slice = slice::from_raw_parts_mut(output_up, size as usize);

    let params = aroon::AroonParams {
        length: Some(period as usize),
    };

    // Create minimal Candles structure
    let mut hl2 = vec![0.0; size as usize];
    let mut hlc3 = vec![0.0; size as usize];
    let mut ohlc4 = vec![0.0; size as usize];
    let mut hlcc4 = vec![0.0; size as usize];

    for i in 0..size as usize {
        hl2[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlc3[i] = (high_slice[i] + low_slice[i]) / 3.0;
        ohlc4[i] = (high_slice[i] + low_slice[i]) / 4.0;
        hlcc4[i] = (high_slice[i] + low_slice[i]) / 4.0;
    }

    let candles = Candles {
        high: high_slice.to_vec(),
        low: low_slice.to_vec(),
        close: vec![0.0; size as usize],
        open: vec![0.0; size as usize],
        volume: vec![0.0; size as usize],
        timestamp: vec![0; size as usize],
        hl2,
        hlc3,
        ohlc4,
        hlcc4,
    };

    let aroon_input = aroon::AroonInput::from_candles(&candles, params);

    match aroon::aroon(&aroon_input) {
        Ok(result) => {
            down_slice.copy_from_slice(&result.aroon_down);
            up_slice.copy_from_slice(&result.aroon_up);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust AROONOSC (Aroon Oscillator) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_aroonosc(
    size: c_int,
    high: *const c_double,
    low: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if high.is_null() || low.is_null() || output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let high_slice = slice::from_raw_parts(high, size as usize);
    let low_slice = slice::from_raw_parts(low, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = aroonosc::AroonOscParams {
        length: Some(period as usize),
    };

    // Create Candles structure
    let mut hl2 = vec![0.0; size as usize];
    let mut hlc3 = vec![0.0; size as usize];
    let mut ohlc4 = vec![0.0; size as usize];
    let mut hlcc4 = vec![0.0; size as usize];

    for i in 0..size as usize {
        hl2[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlc3[i] = (high_slice[i] + low_slice[i]) / 2.0;
        ohlc4[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlcc4[i] = (high_slice[i] + low_slice[i]) / 2.0;
    }

    let candles = Candles {
        high: high_slice.to_vec(),
        low: low_slice.to_vec(),
        close: vec![0.0; size as usize],
        open: vec![0.0; size as usize],
        volume: vec![0.0; size as usize],
        timestamp: vec![0; size as usize],
        hl2,
        hlc3,
        ohlc4,
        hlcc4,
    };

    let aroonosc_input = aroonosc::AroonOscInput::from_candles(&candles, params);

    match aroonosc::aroon_osc(&aroonosc_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust BOP (Balance of Power) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_bop(
    size: c_int,
    open: *const c_double,
    high: *const c_double,
    low: *const c_double,
    close: *const c_double,
    output: *mut c_double,
) -> c_int {
    if open.is_null() || high.is_null() || low.is_null() || close.is_null() ||
       output.is_null() || size <= 0 {
        return -1;
    }

    let open_slice = slice::from_raw_parts(open, size as usize);
    let high_slice = slice::from_raw_parts(high, size as usize);
    let low_slice = slice::from_raw_parts(low, size as usize);
    let close_slice = slice::from_raw_parts(close, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    // Create Candles structure
    let mut hl2 = vec![0.0; size as usize];
    let mut hlc3 = vec![0.0; size as usize];
    let mut ohlc4 = vec![0.0; size as usize];
    let mut hlcc4 = vec![0.0; size as usize];

    for i in 0..size as usize {
        hl2[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlc3[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 3.0;
        ohlc4[i] = (open_slice[i] + high_slice[i] + low_slice[i] + close_slice[i]) / 4.0;
        hlcc4[i] = (high_slice[i] + low_slice[i] + close_slice[i] + close_slice[i]) / 4.0;
    }

    let candles = Candles {
        high: high_slice.to_vec(),
        low: low_slice.to_vec(),
        close: close_slice.to_vec(),
        open: open_slice.to_vec(),
        volume: vec![0.0; size as usize],
        timestamp: vec![0; size as usize],
        hl2,
        hlc3,
        ohlc4,
        hlcc4,
    };

    let bop_input = bop::BopInput::from_candles(&candles, bop::BopParams {});

    match bop::bop(&bop_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust CMO (Chande Momentum Oscillator) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_cmo(
    size: c_int,
    input: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if input.is_null() || output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = cmo::CmoParams {
        period: Some(period as usize),
    };

    let cmo_input = cmo::CmoInput::from_slice(input_slice, params);

    match cmo::cmo(&cmo_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust DI (Directional Indicator) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_di(
    size: c_int,
    high: *const c_double,
    low: *const c_double,
    close: *const c_double,
    period: c_int,
    output_plus: *mut c_double,
    output_minus: *mut c_double,
) -> c_int {
    if high.is_null() || low.is_null() || close.is_null() ||
       output_plus.is_null() || output_minus.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let high_slice = slice::from_raw_parts(high, size as usize);
    let low_slice = slice::from_raw_parts(low, size as usize);
    let close_slice = slice::from_raw_parts(close, size as usize);
    let plus_slice = slice::from_raw_parts_mut(output_plus, size as usize);
    let minus_slice = slice::from_raw_parts_mut(output_minus, size as usize);

    let params = di::DiParams {
        period: Some(period as usize),
    };

    // Create Candles structure
    let mut hl2 = vec![0.0; size as usize];
    let mut hlc3 = vec![0.0; size as usize];
    let mut ohlc4 = vec![0.0; size as usize];
    let mut hlcc4 = vec![0.0; size as usize];

    for i in 0..size as usize {
        hl2[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlc3[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 3.0;
        ohlc4[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 4.0;
        hlcc4[i] = (high_slice[i] + low_slice[i] + close_slice[i] + close_slice[i]) / 4.0;
    }

    let candles = Candles {
        high: high_slice.to_vec(),
        low: low_slice.to_vec(),
        close: close_slice.to_vec(),
        open: vec![0.0; size as usize],
        volume: vec![0.0; size as usize],
        timestamp: vec![0; size as usize],
        hl2,
        hlc3,
        ohlc4,
        hlcc4,
    };

    let di_input = di::DiInput::from_candles(&candles, params);

    match di::di(&di_input) {
        Ok(result) => {
            plus_slice.copy_from_slice(&result.plus);
            minus_slice.copy_from_slice(&result.minus);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust DPO (Detrended Price Oscillator) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_dpo(
    size: c_int,
    input: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if input.is_null() || output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = dpo::DpoParams {
        period: Some(period as usize),
    };

    let dpo_input = dpo::DpoInput::from_slice(input_slice, params);

    match dpo::dpo(&dpo_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust DX (Directional Movement Index) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_dx(
    size: c_int,
    high: *const c_double,
    low: *const c_double,
    close: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if high.is_null() || low.is_null() || close.is_null() || output.is_null() ||
       size <= 0 || period <= 0 {
        return -1;
    }

    let high_slice = slice::from_raw_parts(high, size as usize);
    let low_slice = slice::from_raw_parts(low, size as usize);
    let close_slice = slice::from_raw_parts(close, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = dx::DxParams {
        period: Some(period as usize),
    };

    // Create Candles structure
    let mut hl2 = vec![0.0; size as usize];
    let mut hlc3 = vec![0.0; size as usize];
    let mut ohlc4 = vec![0.0; size as usize];
    let mut hlcc4 = vec![0.0; size as usize];

    for i in 0..size as usize {
        hl2[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlc3[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 3.0;
        ohlc4[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 4.0;
        hlcc4[i] = (high_slice[i] + low_slice[i] + close_slice[i] + close_slice[i]) / 4.0;
    }

    let candles = Candles {
        high: high_slice.to_vec(),
        low: low_slice.to_vec(),
        close: close_slice.to_vec(),
        open: vec![0.0; size as usize],
        volume: vec![0.0; size as usize],
        timestamp: vec![0; size as usize],
        hl2,
        hlc3,
        ohlc4,
        hlcc4,
    };

    let dx_input = dx::DxInput::from_candles(&candles, params);

    match dx::dx(&dx_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust FISHER (Fisher Transform) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_fisher(
    size: c_int,
    high: *const c_double,
    low: *const c_double,
    period: c_int,
    output_fisher: *mut c_double,
    output_signal: *mut c_double,
) -> c_int {
    if high.is_null() || low.is_null() || output_fisher.is_null() ||
       output_signal.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let high_slice = slice::from_raw_parts(high, size as usize);
    let low_slice = slice::from_raw_parts(low, size as usize);
    let fisher_slice = slice::from_raw_parts_mut(output_fisher, size as usize);
    let signal_slice = slice::from_raw_parts_mut(output_signal, size as usize);

    let params = fisher::FisherParams {
        period: Some(period as usize),
    };

    // Create Candles structure
    let mut hl2 = vec![0.0; size as usize];
    let mut hlc3 = vec![0.0; size as usize];
    let mut ohlc4 = vec![0.0; size as usize];
    let mut hlcc4 = vec![0.0; size as usize];

    for i in 0..size as usize {
        hl2[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlc3[i] = (high_slice[i] + low_slice[i]) / 2.0;
        ohlc4[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlcc4[i] = (high_slice[i] + low_slice[i]) / 2.0;
    }

    let candles = Candles {
        high: high_slice.to_vec(),
        low: low_slice.to_vec(),
        close: vec![0.0; size as usize],
        open: vec![0.0; size as usize],
        volume: vec![0.0; size as usize],
        timestamp: vec![0; size as usize],
        hl2,
        hlc3,
        ohlc4,
        hlcc4,
    };

    let fisher_input = fisher::FisherInput::from_candles(&candles, params);

    match fisher::fisher(&fisher_input) {
        Ok(result) => {
            fisher_slice.copy_from_slice(&result.fisher);
            signal_slice.copy_from_slice(&result.signal);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust MFI (Money Flow Index) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_mfi(
    size: c_int,
    high: *const c_double,
    low: *const c_double,
    close: *const c_double,
    volume: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if high.is_null() || low.is_null() || close.is_null() || volume.is_null() ||
       output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let high_slice = slice::from_raw_parts(high, size as usize);
    let low_slice = slice::from_raw_parts(low, size as usize);
    let close_slice = slice::from_raw_parts(close, size as usize);
    let volume_slice = slice::from_raw_parts(volume, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = mfi::MfiParams {
        period: Some(period as usize),
    };

    // Create Candles structure
    let mut hl2 = vec![0.0; size as usize];
    let mut hlc3 = vec![0.0; size as usize];
    let mut ohlc4 = vec![0.0; size as usize];
    let mut hlcc4 = vec![0.0; size as usize];

    for i in 0..size as usize {
        hl2[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlc3[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 3.0;
        ohlc4[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 4.0;
        hlcc4[i] = (high_slice[i] + low_slice[i] + close_slice[i] + close_slice[i]) / 4.0;
    }

    let candles = Candles {
        high: high_slice.to_vec(),
        low: low_slice.to_vec(),
        close: close_slice.to_vec(),
        open: vec![0.0; size as usize],
        volume: volume_slice.to_vec(),
        timestamp: vec![0; size as usize],
        hl2,
        hlc3,
        ohlc4,
        hlcc4,
    };

    let mfi_input = mfi::MfiInput::from_candles(&candles, "hlc3", params);

    match mfi::mfi(&mfi_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust MOM (Momentum) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_mom(
    size: c_int,
    input: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if input.is_null() || output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = mom::MomParams {
        period: Some(period as usize),
    };

    let mom_input = mom::MomInput::from_slice(input_slice, params);

    match mom::mom(&mom_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust NATR (Normalized Average True Range) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_natr(
    size: c_int,
    high: *const c_double,
    low: *const c_double,
    close: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if high.is_null() || low.is_null() || close.is_null() || output.is_null() ||
       size <= 0 || period <= 0 {
        return -1;
    }

    let high_slice = slice::from_raw_parts(high, size as usize);
    let low_slice = slice::from_raw_parts(low, size as usize);
    let close_slice = slice::from_raw_parts(close, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = natr::NatrParams {
        period: Some(period as usize),
    };

    // Create Candles structure
    let mut hl2 = vec![0.0; size as usize];
    let mut hlc3 = vec![0.0; size as usize];
    let mut ohlc4 = vec![0.0; size as usize];
    let mut hlcc4 = vec![0.0; size as usize];

    for i in 0..size as usize {
        hl2[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlc3[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 3.0;
        ohlc4[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 4.0;
        hlcc4[i] = (high_slice[i] + low_slice[i] + close_slice[i] + close_slice[i]) / 4.0;
    }

    let candles = Candles {
        high: high_slice.to_vec(),
        low: low_slice.to_vec(),
        close: close_slice.to_vec(),
        open: vec![0.0; size as usize],
        volume: vec![0.0; size as usize],
        timestamp: vec![0; size as usize],
        hl2,
        hlc3,
        ohlc4,
        hlcc4,
    };

    let natr_input = natr::NatrInput::from_candles(&candles, params);

    match natr::natr(&natr_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust OBV (On Balance Volume) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_obv(
    size: c_int,
    close: *const c_double,
    volume: *const c_double,
    output: *mut c_double,
) -> c_int {
    if close.is_null() || volume.is_null() || output.is_null() || size <= 0 {
        return -1;
    }

    let close_slice = slice::from_raw_parts(close, size as usize);
    let volume_slice = slice::from_raw_parts(volume, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    // Create Candles structure
    let mut hl2 = vec![0.0; size as usize];
    let mut hlc3 = vec![0.0; size as usize];
    let mut ohlc4 = vec![0.0; size as usize];
    let mut hlcc4 = vec![0.0; size as usize];

    for i in 0..size as usize {
        hl2[i] = close_slice[i];
        hlc3[i] = close_slice[i];
        ohlc4[i] = close_slice[i];
        hlcc4[i] = close_slice[i];
    }

    let candles = Candles {
        high: vec![0.0; size as usize],
        low: vec![0.0; size as usize],
        close: close_slice.to_vec(),
        open: vec![0.0; size as usize],
        volume: volume_slice.to_vec(),
        timestamp: vec![0; size as usize],
        hl2,
        hlc3,
        ohlc4,
        hlcc4,
    };

    let obv_input = obv::ObvInput::from_candles(&candles, obv::ObvParams {});

    match obv::obv(&obv_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust PPO (Percentage Price Oscillator) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_ppo(
    size: c_int,
    input: *const c_double,
    fast_period: c_int,
    slow_period: c_int,
    _signal_period: c_int,
    output_ppo: *mut c_double,
    output_signal: *mut c_double,
    output_hist: *mut c_double,
) -> c_int {
    if input.is_null() || output_ppo.is_null() || output_signal.is_null() ||
       output_hist.is_null() || size <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let ppo_slice = slice::from_raw_parts_mut(output_ppo, size as usize);
    let signal_slice = slice::from_raw_parts_mut(output_signal, size as usize);
    let hist_slice = slice::from_raw_parts_mut(output_hist, size as usize);

    let params = ppo::PpoParams {
        fast_period: Some(fast_period as usize),
        slow_period: Some(slow_period as usize),
        ma_type: Some("ema".to_string()),
    };

    let ppo_input = ppo::PpoInput::from_slice(input_slice, params);

    match ppo::ppo(&ppo_input) {
        Ok(result) => {
            // PPO only returns a single output
            ppo_slice.copy_from_slice(&result.values);
            // Fill signal and histogram with NaN for now
            for i in 0..size as usize {
                signal_slice[i] = f64::NAN;
                hist_slice[i] = f64::NAN;
            }
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust ROC (Rate of Change) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_roc(
    size: c_int,
    input: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if input.is_null() || output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = roc::RocParams {
        period: Some(period as usize),
    };

    let roc_input = roc::RocInput::from_slice(input_slice, params);

    match roc::roc(&roc_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust ROCR (Rate of Change Ratio) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_rocr(
    size: c_int,
    input: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if input.is_null() || output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = rocr::RocrParams {
        period: Some(period as usize),
    };

    let rocr_input = rocr::RocrInput::from_slice(input_slice, params);

    match rocr::rocr(&rocr_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust ROCP (Rate of Change Percentage) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_rocp(
    size: c_int,
    input: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if input.is_null() || output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = rocp::RocpParams {
        period: Some(period as usize),
    };

    let rocp_input = rocp::RocpInput::from_slice(input_slice, params);

    match rocp::rocp(&rocp_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust STDDEV (Standard Deviation) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_stddev(
    size: c_int,
    input: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if input.is_null() || output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = stddev::StdDevParams {
        period: Some(period as usize),
        nbdev: Some(1.0),
    };

    let stddev_input = stddev::StdDevInput::from_slice(input_slice, params);

    match stddev::stddev(&stddev_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust ULTOSC (Ultimate Oscillator) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_ultosc(
    size: c_int,
    high: *const c_double,
    low: *const c_double,
    close: *const c_double,
    period1: c_int,
    period2: c_int,
    period3: c_int,
    output: *mut c_double,
) -> c_int {
    if high.is_null() || low.is_null() || close.is_null() || output.is_null() ||
       size <= 0 || period1 <= 0 || period2 <= 0 || period3 <= 0 {
        return -1;
    }

    let high_slice = slice::from_raw_parts(high, size as usize);
    let low_slice = slice::from_raw_parts(low, size as usize);
    let close_slice = slice::from_raw_parts(close, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = ultosc::UltOscParams {
        timeperiod1: Some(period1 as usize),
        timeperiod2: Some(period2 as usize),
        timeperiod3: Some(period3 as usize),
    };

    // Create Candles structure
    let mut hl2 = vec![0.0; size as usize];
    let mut hlc3 = vec![0.0; size as usize];
    let mut ohlc4 = vec![0.0; size as usize];
    let mut hlcc4 = vec![0.0; size as usize];

    for i in 0..size as usize {
        hl2[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlc3[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 3.0;
        ohlc4[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 4.0;
        hlcc4[i] = (high_slice[i] + low_slice[i] + close_slice[i] + close_slice[i]) / 4.0;
    }

    let candles = Candles {
        high: high_slice.to_vec(),
        low: low_slice.to_vec(),
        close: close_slice.to_vec(),
        open: vec![0.0; size as usize],
        volume: vec![0.0; size as usize],
        timestamp: vec![0; size as usize],
        hl2,
        hlc3,
        ohlc4,
        hlcc4,
    };

    let ultosc_input = ultosc::UltOscInput::from_candles(&candles, "high", "low", "close", params);

    match ultosc::ultosc(&ultosc_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust VAR (Variance) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_var(
    size: c_int,
    input: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if input.is_null() || output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = var::VarParams {
        period: Some(period as usize),
        nbdev: Some(1.0),
    };

    let var_input = var::VarInput::from_slice(input_slice, params);

    match var::var(&var_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust WILLR (Williams %R) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_willr(
    size: c_int,
    high: *const c_double,
    low: *const c_double,
    close: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if high.is_null() || low.is_null() || close.is_null() || output.is_null() ||
       size <= 0 || period <= 0 {
        return -1;
    }

    let high_slice = slice::from_raw_parts(high, size as usize);
    let low_slice = slice::from_raw_parts(low, size as usize);
    let close_slice = slice::from_raw_parts(close, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = willr::WillrParams {
        period: Some(period as usize),
    };

    // Create Candles structure
    let mut hl2 = vec![0.0; size as usize];
    let mut hlc3 = vec![0.0; size as usize];
    let mut ohlc4 = vec![0.0; size as usize];
    let mut hlcc4 = vec![0.0; size as usize];

    for i in 0..size as usize {
        hl2[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlc3[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 3.0;
        ohlc4[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 4.0;
        hlcc4[i] = (high_slice[i] + low_slice[i] + close_slice[i] + close_slice[i]) / 4.0;
    }

    let candles = Candles {
        high: high_slice.to_vec(),
        low: low_slice.to_vec(),
        close: close_slice.to_vec(),
        open: vec![0.0; size as usize],
        volume: vec![0.0; size as usize],
        timestamp: vec![0; size as usize],
        hl2,
        hlc3,
        ohlc4,
        hlcc4,
    };

    let willr_input = willr::WillrInput::from_candles(&candles, params);

    match willr::willr(&willr_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust DM (Directional Movement) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_dm(
    size: c_int,
    high: *const c_double,
    low: *const c_double,
    period: c_int,
    output_plus: *mut c_double,
    output_minus: *mut c_double,
) -> c_int {
    if high.is_null() || low.is_null() || output_plus.is_null() ||
       output_minus.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let high_slice = slice::from_raw_parts(high, size as usize);
    let low_slice = slice::from_raw_parts(low, size as usize);
    let plus_slice = slice::from_raw_parts_mut(output_plus, size as usize);
    let minus_slice = slice::from_raw_parts_mut(output_minus, size as usize);

    let params = dm::DmParams {
        period: Some(period as usize),
    };

    // Create Candles structure
    let mut hl2 = vec![0.0; size as usize];
    let mut hlc3 = vec![0.0; size as usize];
    let mut ohlc4 = vec![0.0; size as usize];
    let mut hlcc4 = vec![0.0; size as usize];

    for i in 0..size as usize {
        hl2[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlc3[i] = (high_slice[i] + low_slice[i]) / 2.0;
        ohlc4[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlcc4[i] = (high_slice[i] + low_slice[i]) / 2.0;
    }

    let candles = Candles {
        high: high_slice.to_vec(),
        low: low_slice.to_vec(),
        close: vec![0.0; size as usize],
        open: vec![0.0; size as usize],
        volume: vec![0.0; size as usize],
        timestamp: vec![0; size as usize],
        hl2,
        hlc3,
        ohlc4,
        hlcc4,
    };

    let dm_input = dm::DmInput::from_candles(&candles, params);

    match dm::dm(&dm_input) {
        Ok(result) => {
            plus_slice.copy_from_slice(&result.plus);
            minus_slice.copy_from_slice(&result.minus);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust FOSC (Forecast Oscillator) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_fosc(
    size: c_int,
    input: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if input.is_null() || output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = fosc::FoscParams {
        period: Some(period as usize),
    };

    let fosc_input = fosc::FoscInput::from_slice(input_slice, params);

    match fosc::fosc(&fosc_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust KVO (Klinger Volume Oscillator) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_kvo(
    size: c_int,
    high: *const c_double,
    low: *const c_double,
    close: *const c_double,
    volume: *const c_double,
    short_period: c_int,
    long_period: c_int,
    output_kvo: *mut c_double,
    output_signal: *mut c_double,
) -> c_int {
    if high.is_null() || low.is_null() || close.is_null() || volume.is_null() ||
       output_kvo.is_null() || output_signal.is_null() || size <= 0 {
        return -1;
    }

    let high_slice = slice::from_raw_parts(high, size as usize);
    let low_slice = slice::from_raw_parts(low, size as usize);
    let close_slice = slice::from_raw_parts(close, size as usize);
    let volume_slice = slice::from_raw_parts(volume, size as usize);
    let kvo_slice = slice::from_raw_parts_mut(output_kvo, size as usize);
    let signal_slice = slice::from_raw_parts_mut(output_signal, size as usize);

    let params = kvo::KvoParams {
        short_period: Some(short_period as usize),
        long_period: Some(long_period as usize),
    };

    // Create Candles structure
    let mut hl2 = vec![0.0; size as usize];
    let mut hlc3 = vec![0.0; size as usize];
    let mut ohlc4 = vec![0.0; size as usize];
    let mut hlcc4 = vec![0.0; size as usize];

    for i in 0..size as usize {
        hl2[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlc3[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 3.0;
        ohlc4[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 4.0;
        hlcc4[i] = (high_slice[i] + low_slice[i] + close_slice[i] + close_slice[i]) / 4.0;
    }

    let candles = Candles {
        high: high_slice.to_vec(),
        low: low_slice.to_vec(),
        close: close_slice.to_vec(),
        open: vec![0.0; size as usize],
        volume: volume_slice.to_vec(),
        timestamp: vec![0; size as usize],
        hl2,
        hlc3,
        ohlc4,
        hlcc4,
    };

    let kvo_input = kvo::KvoInput::from_candles(&candles, params);

    match kvo::kvo(&kvo_input) {
        Ok(result) => {
            kvo_slice.copy_from_slice(&result.values);
            // KVO only returns a single output, fill signal with NaN
            for i in 0..size as usize {
                signal_slice[i] = f64::NAN;
            }
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust LINREG (Linear Regression) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_linreg(
    size: c_int,
    input: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if input.is_null() || output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = linreg::LinRegParams {
        period: Some(period as usize),
    };

    let linreg_input = linreg::LinRegInput::from_slice(input_slice, params);

    match linreg::linreg(&linreg_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust MASS (Mass Index) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_mass(
    size: c_int,
    high: *const c_double,
    low: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if high.is_null() || low.is_null() || output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let high_slice = slice::from_raw_parts(high, size as usize);
    let low_slice = slice::from_raw_parts(low, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = mass::MassParams {
        period: Some(period as usize),
    };

    // Create Candles structure
    let mut hl2 = vec![0.0; size as usize];
    let mut hlc3 = vec![0.0; size as usize];
    let mut ohlc4 = vec![0.0; size as usize];
    let mut hlcc4 = vec![0.0; size as usize];

    for i in 0..size as usize {
        hl2[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlc3[i] = (high_slice[i] + low_slice[i]) / 2.0;
        ohlc4[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlcc4[i] = (high_slice[i] + low_slice[i]) / 2.0;
    }

    let candles = Candles {
        high: high_slice.to_vec(),
        low: low_slice.to_vec(),
        close: vec![0.0; size as usize],
        open: vec![0.0; size as usize],
        volume: vec![0.0; size as usize],
        timestamp: vec![0; size as usize],
        hl2,
        hlc3,
        ohlc4,
        hlcc4,
    };

    let mass_input = mass::MassInput::from_candles(&candles, "high", "low", params);

    match mass::mass(&mass_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust MEDPRICE (Median Price) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_medprice(
    size: c_int,
    high: *const c_double,
    low: *const c_double,
    output: *mut c_double,
) -> c_int {
    if high.is_null() || low.is_null() || output.is_null() || size <= 0 {
        return -1;
    }

    let high_slice = slice::from_raw_parts(high, size as usize);
    let low_slice = slice::from_raw_parts(low, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    // Create Candles structure
    let mut hl2 = vec![0.0; size as usize];
    let mut hlc3 = vec![0.0; size as usize];
    let mut ohlc4 = vec![0.0; size as usize];
    let mut hlcc4 = vec![0.0; size as usize];

    for i in 0..size as usize {
        hl2[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlc3[i] = (high_slice[i] + low_slice[i]) / 2.0;
        ohlc4[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlcc4[i] = (high_slice[i] + low_slice[i]) / 2.0;
    }

    let candles = Candles {
        high: high_slice.to_vec(),
        low: low_slice.to_vec(),
        close: vec![0.0; size as usize],
        open: vec![0.0; size as usize],
        volume: vec![0.0; size as usize],
        timestamp: vec![0; size as usize],
        hl2,
        hlc3,
        ohlc4,
        hlcc4,
    };

    let medprice_input = medprice::MedpriceInput::from_candles(&candles, "high", "low", medprice::MedpriceParams {});

    match medprice::medprice(&medprice_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust MIDPOINT indicator
#[no_mangle]
pub unsafe extern "C" fn rust_midpoint(
    size: c_int,
    input: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if input.is_null() || output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = midpoint::MidpointParams {
        period: Some(period as usize),
    };

    let midpoint_input = midpoint::MidpointInput::from_slice(input_slice, params);

    match midpoint::midpoint(&midpoint_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust MIDPRICE indicator
#[no_mangle]
pub unsafe extern "C" fn rust_midprice(
    size: c_int,
    high: *const c_double,
    low: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if high.is_null() || low.is_null() || output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let high_slice = slice::from_raw_parts(high, size as usize);
    let low_slice = slice::from_raw_parts(low, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = midprice::MidpriceParams {
        period: Some(period as usize),
    };

    // Create Candles structure
    let mut hl2 = vec![0.0; size as usize];
    let mut hlc3 = vec![0.0; size as usize];
    let mut ohlc4 = vec![0.0; size as usize];
    let mut hlcc4 = vec![0.0; size as usize];

    for i in 0..size as usize {
        hl2[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlc3[i] = (high_slice[i] + low_slice[i]) / 2.0;
        ohlc4[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlcc4[i] = (high_slice[i] + low_slice[i]) / 2.0;
    }

    let candles = Candles {
        high: high_slice.to_vec(),
        low: low_slice.to_vec(),
        close: vec![0.0; size as usize],
        open: vec![0.0; size as usize],
        volume: vec![0.0; size as usize],
        timestamp: vec![0; size as usize],
        hl2,
        hlc3,
        ohlc4,
        hlcc4,
    };

    let midprice_input = midprice::MidpriceInput::from_candles(&candles, "high", "low", params);

    match midprice::midprice(&midprice_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust NVI (Negative Volume Index) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_nvi(
    size: c_int,
    close: *const c_double,
    volume: *const c_double,
    output: *mut c_double,
) -> c_int {
    if close.is_null() || volume.is_null() || output.is_null() || size <= 0 {
        return -1;
    }

    let close_slice = slice::from_raw_parts(close, size as usize);
    let volume_slice = slice::from_raw_parts(volume, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    // Create Candles structure
    let mut hl2 = vec![0.0; size as usize];
    let mut hlc3 = vec![0.0; size as usize];
    let mut ohlc4 = vec![0.0; size as usize];
    let mut hlcc4 = vec![0.0; size as usize];

    for i in 0..size as usize {
        hl2[i] = close_slice[i];
        hlc3[i] = close_slice[i];
        ohlc4[i] = close_slice[i];
        hlcc4[i] = close_slice[i];
    }

    let candles = Candles {
        high: vec![0.0; size as usize],
        low: vec![0.0; size as usize],
        close: close_slice.to_vec(),
        open: vec![0.0; size as usize],
        volume: volume_slice.to_vec(),
        timestamp: vec![0; size as usize],
        hl2,
        hlc3,
        ohlc4,
        hlcc4,
    };

    let nvi_input = nvi::NviInput::from_candles(&candles, "close", nvi::NviParams {});

    match nvi::nvi(&nvi_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust PVI (Positive Volume Index) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_pvi(
    size: c_int,
    close: *const c_double,
    volume: *const c_double,
    output: *mut c_double,
) -> c_int {
    if close.is_null() || volume.is_null() || output.is_null() || size <= 0 {
        return -1;
    }

    let close_slice = slice::from_raw_parts(close, size as usize);
    let volume_slice = slice::from_raw_parts(volume, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    // Create Candles structure
    let mut hl2 = vec![0.0; size as usize];
    let mut hlc3 = vec![0.0; size as usize];
    let mut ohlc4 = vec![0.0; size as usize];
    let mut hlcc4 = vec![0.0; size as usize];

    for i in 0..size as usize {
        hl2[i] = close_slice[i];
        hlc3[i] = close_slice[i];
        ohlc4[i] = close_slice[i];
        hlcc4[i] = close_slice[i];
    }

    let candles = Candles {
        high: vec![0.0; size as usize],
        low: vec![0.0; size as usize],
        close: close_slice.to_vec(),
        open: vec![0.0; size as usize],
        volume: volume_slice.to_vec(),
        timestamp: vec![0; size as usize],
        hl2,
        hlc3,
        ohlc4,
        hlcc4,
    };

    let pvi_input = pvi::PviInput::from_candles(&candles, "close", "volume", pvi::PviParams { initial_value: Some(1000.0) });

    match pvi::pvi(&pvi_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust QSTICK indicator
#[no_mangle]
pub unsafe extern "C" fn rust_qstick(
    size: c_int,
    open: *const c_double,
    close: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if open.is_null() || close.is_null() || output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let open_slice = slice::from_raw_parts(open, size as usize);
    let close_slice = slice::from_raw_parts(close, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = qstick::QstickParams {
        period: Some(period as usize),
    };

    // Create Candles structure
    let mut hl2 = vec![0.0; size as usize];
    let mut hlc3 = vec![0.0; size as usize];
    let mut ohlc4 = vec![0.0; size as usize];
    let mut hlcc4 = vec![0.0; size as usize];

    for i in 0..size as usize {
        hl2[i] = (open_slice[i] + close_slice[i]) / 2.0;
        hlc3[i] = (open_slice[i] + close_slice[i]) / 2.0;
        ohlc4[i] = (open_slice[i] + close_slice[i]) / 2.0;
        hlcc4[i] = (open_slice[i] + close_slice[i]) / 2.0;
    }

    let candles = Candles {
        high: vec![0.0; size as usize],
        low: vec![0.0; size as usize],
        close: close_slice.to_vec(),
        open: open_slice.to_vec(),
        volume: vec![0.0; size as usize],
        timestamp: vec![0; size as usize],
        hl2,
        hlc3,
        ohlc4,
        hlcc4,
    };

    let qstick_input = qstick::QstickInput::from_candles(&candles, "open", "close", params);

    match qstick::qstick(&qstick_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust SAR (Parabolic SAR) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_sar(
    size: c_int,
    high: *const c_double,
    low: *const c_double,
    accel_start: c_double,
    accel_max: c_double,
    output: *mut c_double,
) -> c_int {
    if high.is_null() || low.is_null() || output.is_null() || size <= 0 {
        return -1;
    }

    let high_slice = slice::from_raw_parts(high, size as usize);
    let low_slice = slice::from_raw_parts(low, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = sar::SarParams {
        acceleration: Some(accel_start),
        maximum: Some(accel_max),
    };

    // Create Candles structure
    let mut hl2 = vec![0.0; size as usize];
    let mut hlc3 = vec![0.0; size as usize];
    let mut ohlc4 = vec![0.0; size as usize];
    let mut hlcc4 = vec![0.0; size as usize];

    for i in 0..size as usize {
        hl2[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlc3[i] = (high_slice[i] + low_slice[i]) / 2.0;
        ohlc4[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlcc4[i] = (high_slice[i] + low_slice[i]) / 2.0;
    }

    let candles = Candles {
        high: high_slice.to_vec(),
        low: low_slice.to_vec(),
        close: vec![0.0; size as usize],
        open: vec![0.0; size as usize],
        volume: vec![0.0; size as usize],
        timestamp: vec![0; size as usize],
        hl2,
        hlc3,
        ohlc4,
        hlcc4,
    };

    let sar_input = sar::SarInput::from_candles(&candles, params);

    match sar::sar(&sar_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust SRSI (Stochastic RSI) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_srsi(
    size: c_int,
    input: *const c_double,
    rsi_period: c_int,
    stoch_period: c_int,
    k_period: c_int,
    d_period: c_int,
    output_k: *mut c_double,
    output_d: *mut c_double,
) -> c_int {
    if input.is_null() || output_k.is_null() || output_d.is_null() || size <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let k_slice = slice::from_raw_parts_mut(output_k, size as usize);
    let d_slice = slice::from_raw_parts_mut(output_d, size as usize);

    let params = srsi::SrsiParams {
        source: Some("close".to_string()),
        rsi_period: Some(rsi_period as usize),
        stoch_period: Some(stoch_period as usize),
        k: Some(k_period as usize),
        d: Some(d_period as usize),
    };

    let srsi_input = srsi::SrsiInput::from_slice(input_slice, params);

    match srsi::srsi(&srsi_input) {
        Ok(result) => {
            k_slice.copy_from_slice(&result.k);
            d_slice.copy_from_slice(&result.d);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust STOCHF (Stochastic Fast) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_stochf(
    size: c_int,
    high: *const c_double,
    low: *const c_double,
    close: *const c_double,
    fastk_period: c_int,
    fastd_period: c_int,
    output_k: *mut c_double,
    output_d: *mut c_double,
) -> c_int {
    if high.is_null() || low.is_null() || close.is_null() ||
       output_k.is_null() || output_d.is_null() || size <= 0 {
        return -1;
    }

    let high_slice = slice::from_raw_parts(high, size as usize);
    let low_slice = slice::from_raw_parts(low, size as usize);
    let close_slice = slice::from_raw_parts(close, size as usize);
    let k_slice = slice::from_raw_parts_mut(output_k, size as usize);
    let d_slice = slice::from_raw_parts_mut(output_d, size as usize);

    let params = stochf::StochfParams {
        fastk_period: Some(fastk_period as usize),
        fastd_period: Some(fastd_period as usize),
        fastd_matype: Some(0), // 0 = SMA
    };

    // Create Candles structure
    let mut hl2 = vec![0.0; size as usize];
    let mut hlc3 = vec![0.0; size as usize];
    let mut ohlc4 = vec![0.0; size as usize];
    let mut hlcc4 = vec![0.0; size as usize];

    for i in 0..size as usize {
        hl2[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlc3[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 3.0;
        ohlc4[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 4.0;
        hlcc4[i] = (high_slice[i] + low_slice[i] + close_slice[i] + close_slice[i]) / 4.0;
    }

    let candles = Candles {
        high: high_slice.to_vec(),
        low: low_slice.to_vec(),
        close: close_slice.to_vec(),
        open: vec![0.0; size as usize],
        volume: vec![0.0; size as usize],
        timestamp: vec![0; size as usize],
        hl2,
        hlc3,
        ohlc4,
        hlcc4,
    };

    let stochf_input = stochf::StochfInput::from_candles(&candles, params);

    match stochf::stochf(&stochf_input) {
        Ok(result) => {
            k_slice.copy_from_slice(&result.k);
            d_slice.copy_from_slice(&result.d);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust TRIX indicator
#[no_mangle]
pub unsafe extern "C" fn rust_trix(
    size: c_int,
    input: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if input.is_null() || output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = trix::TrixParams {
        period: Some(period as usize),
    };

    let trix_input = trix::TrixInput::from_slice(input_slice, params);

    match trix::trix(&trix_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust TSF (Time Series Forecast) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_tsf(
    size: c_int,
    input: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if input.is_null() || output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = tsf::TsfParams {
        period: Some(period as usize),
    };

    let tsf_input = tsf::TsfInput::from_slice(input_slice, params);

    match tsf::tsf(&tsf_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust VIDYA indicator
#[no_mangle]
pub unsafe extern "C" fn rust_vidya(
    size: c_int,
    input: *const c_double,
    short_period: c_int,
    long_period: c_int,
    alpha: c_double,
    output: *mut c_double,
) -> c_int {
    if input.is_null() || output.is_null() || size <= 0 || short_period <= 0 || long_period <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = vidya::VidyaParams {
        short_period: Some(short_period as usize),
        long_period: Some(long_period as usize),
        alpha: Some(alpha),
    };

    let vidya_input = vidya::VidyaInput::from_slice(input_slice, params);

    match vidya::vidya(&vidya_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust VOSC (Volume Oscillator) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_vosc(
    size: c_int,
    volume: *const c_double,
    short_period: c_int,
    long_period: c_int,
    output: *mut c_double,
) -> c_int {
    if volume.is_null() || output.is_null() || size <= 0 || short_period <= 0 || long_period <= 0 {
        return -1;
    }

    let volume_slice = slice::from_raw_parts(volume, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = vosc::VoscParams {
        short_period: Some(short_period as usize),
        long_period: Some(long_period as usize),
    };

    let vosc_input = vosc::VoscInput::from_slice(volume_slice, params);

    match vosc::vosc(&vosc_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust VWMA (Volume Weighted Moving Average) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_vwma(
    size: c_int,
    close: *const c_double,
    volume: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if close.is_null() || volume.is_null() || output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let close_slice = slice::from_raw_parts(close, size as usize);
    let volume_slice = slice::from_raw_parts(volume, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = vwma::VwmaParams {
        period: Some(period as usize),
    };

    // Create Candles structure
    let mut hl2 = vec![0.0; size as usize];
    let mut hlc3 = vec![0.0; size as usize];
    let mut ohlc4 = vec![0.0; size as usize];
    let mut hlcc4 = vec![0.0; size as usize];

    for i in 0..size as usize {
        hl2[i] = close_slice[i];
        hlc3[i] = close_slice[i];
        ohlc4[i] = close_slice[i];
        hlcc4[i] = close_slice[i];
    }

    let candles = Candles {
        high: vec![0.0; size as usize],
        low: vec![0.0; size as usize],
        close: close_slice.to_vec(),
        open: vec![0.0; size as usize],
        volume: volume_slice.to_vec(),
        timestamp: vec![0; size as usize],
        hl2,
        hlc3,
        ohlc4,
        hlcc4,
    };

    let vwma_input = vwma::VwmaInput::from_candles(&candles, "close", params);

    match vwma::vwma(&vwma_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust WAD (Williams Accumulation/Distribution) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_wad(
    size: c_int,
    high: *const c_double,
    low: *const c_double,
    close: *const c_double,
    output: *mut c_double,
) -> c_int {
    if high.is_null() || low.is_null() || close.is_null() || output.is_null() || size <= 0 {
        return -1;
    }

    let high_slice = slice::from_raw_parts(high, size as usize);
    let low_slice = slice::from_raw_parts(low, size as usize);
    let close_slice = slice::from_raw_parts(close, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    // Create Candles structure
    let mut hl2 = vec![0.0; size as usize];
    let mut hlc3 = vec![0.0; size as usize];
    let mut ohlc4 = vec![0.0; size as usize];
    let mut hlcc4 = vec![0.0; size as usize];

    for i in 0..size as usize {
        hl2[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlc3[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 3.0;
        ohlc4[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 4.0;
        hlcc4[i] = (high_slice[i] + low_slice[i] + close_slice[i] + close_slice[i]) / 4.0;
    }

    let candles = Candles {
        high: high_slice.to_vec(),
        low: low_slice.to_vec(),
        close: close_slice.to_vec(),
        open: vec![0.0; size as usize],
        volume: vec![0.0; size as usize],
        timestamp: vec![0; size as usize],
        hl2,
        hlc3,
        ohlc4,
        hlcc4,
    };

    let wad_input = wad::WadInput::from_candles(&candles);

    match wad::wad(&wad_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust WCLPRICE (Weighted Close Price) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_wclprice(
    size: c_int,
    high: *const c_double,
    low: *const c_double,
    close: *const c_double,
    output: *mut c_double,
) -> c_int {
    if high.is_null() || low.is_null() || close.is_null() || output.is_null() || size <= 0 {
        return -1;
    }

    let high_slice = slice::from_raw_parts(high, size as usize);
    let low_slice = slice::from_raw_parts(low, size as usize);
    let close_slice = slice::from_raw_parts(close, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    // Create Candles structure - wclprice is (high + low + 2*close) / 4
    let mut hl2 = vec![0.0; size as usize];
    let mut hlc3 = vec![0.0; size as usize];
    let mut ohlc4 = vec![0.0; size as usize];
    let mut hlcc4 = vec![0.0; size as usize];

    for i in 0..size as usize {
        hl2[i] = (high_slice[i] + low_slice[i]) / 2.0;
        hlc3[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 3.0;
        ohlc4[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 4.0;
        hlcc4[i] = (high_slice[i] + low_slice[i] + close_slice[i] + close_slice[i]) / 4.0;
    }

    let candles = Candles {
        high: high_slice.to_vec(),
        low: low_slice.to_vec(),
        close: close_slice.to_vec(),
        open: vec![0.0; size as usize],
        volume: vec![0.0; size as usize],
        timestamp: vec![0; size as usize],
        hl2,
        hlc3,
        ohlc4,
        hlcc4,
    };

    let wclprice_input = wclprice::WclpriceInput::from_candles(&candles);

    match wclprice::wclprice(&wclprice_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust WILDERS (Wilders Smoothing) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_wilders(
    size: c_int,
    input: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if input.is_null() || output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = wilders::WildersParams {
        period: Some(period as usize),
    };

    let wilders_input = wilders::WildersInput::from_slice(input_slice, params);

    match wilders::wilders(&wilders_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust ZLEMA (Zero Lag EMA) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_zlema(
    size: c_int,
    input: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if input.is_null() || output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = zlema::ZlemaParams {
        period: Some(period as usize),
    };

    let zlema_input = zlema::ZlemaInput::from_slice(input_slice, params);

    match zlema::zlema(&zlema_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust Linear Regression Slope indicator
#[no_mangle]
pub unsafe extern "C" fn rust_linearreg_slope(
    size: c_int,
    input: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if input.is_null() || output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = my_project::indicators::linearreg_slope::LinearRegSlopeParams {
        period: Some(period as usize),
    };

    let linreg_input = my_project::indicators::linearreg_slope::LinearRegSlopeInput::from_slice(input_slice, params);

    match my_project::indicators::linearreg_slope::linearreg_slope(&linreg_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust Linear Regression Intercept indicator
#[no_mangle]
pub unsafe extern "C" fn rust_linearreg_intercept(
    size: c_int,
    input: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if input.is_null() || output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = my_project::indicators::linearreg_intercept::LinearRegInterceptParams {
        period: Some(period as usize),
    };

    let linreg_input = my_project::indicators::linearreg_intercept::LinearRegInterceptInput::from_slice(input_slice, params);

    match my_project::indicators::linearreg_intercept::linearreg_intercept(&linreg_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust EMV (Ease of Movement) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_emv(
    size: c_int,
    high: *const c_double,
    low: *const c_double,
    volume: *const c_double,
    output: *mut c_double,
) -> c_int {
    if high.is_null() || low.is_null() || volume.is_null() || output.is_null() || size <= 0 {
        return -1;
    }

    let high_slice = slice::from_raw_parts(high, size as usize);
    let low_slice = slice::from_raw_parts(low, size as usize);
    let volume_slice = slice::from_raw_parts(volume, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    // Create candles data
    let candles = Candles {
        high: high_slice.to_vec(),
        low: low_slice.to_vec(),
        close: vec![0.0; size as usize], // EMV doesn't use close
        open: vec![0.0; size as usize],  // EMV doesn't use open
        volume: volume_slice.to_vec(),
        timestamp: vec![0; size as usize],
        hl2: vec![0.0; size as usize],
        hlc3: vec![0.0; size as usize],
        ohlc4: vec![0.0; size as usize],
        hlcc4: vec![0.0; size as usize],
    };

    let emv_input = my_project::indicators::emv::EmvInput::from_candles(&candles);

    match my_project::indicators::emv::emv(&emv_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust CVI (Chaikin Volatility Index) indicator
#[no_mangle]
pub unsafe extern "C" fn rust_cvi(
    size: c_int,
    high: *const c_double,
    low: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if high.is_null() || low.is_null() || output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let high_slice = slice::from_raw_parts(high, size as usize);
    let low_slice = slice::from_raw_parts(low, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    // Create candles data
    let candles = Candles {
        high: high_slice.to_vec(),
        low: low_slice.to_vec(),
        close: vec![0.0; size as usize],
        open: vec![0.0; size as usize],
        volume: vec![0.0; size as usize],
        timestamp: vec![0; size as usize],
        hl2: vec![0.0; size as usize],
        hlc3: vec![0.0; size as usize],
        ohlc4: vec![0.0; size as usize],
        hlcc4: vec![0.0; size as usize],
    };

    let params = my_project::indicators::cvi::CviParams {
        period: Some(period as usize),
    };

    let cvi_input = my_project::indicators::cvi::CviInput::from_candles(&candles, params);

    match my_project::indicators::cvi::cvi(&cvi_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}

/// FFI wrapper for Rust True Range indicator
#[no_mangle]
pub unsafe extern "C" fn rust_tr(
    size: c_int,
    high: *const c_double,
    low: *const c_double,
    close: *const c_double,
    output: *mut c_double,
) -> c_int {
    if high.is_null() || low.is_null() || close.is_null() || output.is_null() || size <= 0 {
        return -1;
    }

    let high_slice = slice::from_raw_parts(high, size as usize);
    let low_slice = slice::from_raw_parts(low, size as usize);
    let close_slice = slice::from_raw_parts(close, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    // Calculate true range manually since we might not have a dedicated TR indicator
    output_slice[0] = high_slice[0] - low_slice[0];
    for i in 1..size as usize {
        let hl = high_slice[i] - low_slice[i];
        let hc = (high_slice[i] - close_slice[i - 1]).abs();
        let lc = (low_slice[i] - close_slice[i - 1]).abs();
        output_slice[i] = hl.max(hc).max(lc);
    }

    0
}

/// FFI wrapper for Rust Average Price indicator
#[no_mangle]
pub unsafe extern "C" fn rust_avgprice(
    size: c_int,
    open: *const c_double,
    high: *const c_double,
    low: *const c_double,
    close: *const c_double,
    output: *mut c_double,
) -> c_int {
    if open.is_null() || high.is_null() || low.is_null() || close.is_null() || output.is_null() || size <= 0 {
        return -1;
    }

    let open_slice = slice::from_raw_parts(open, size as usize);
    let high_slice = slice::from_raw_parts(high, size as usize);
    let low_slice = slice::from_raw_parts(low, size as usize);
    let close_slice = slice::from_raw_parts(close, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    // Calculate average price (OHLC/4)
    for i in 0..size as usize {
        output_slice[i] = (open_slice[i] + high_slice[i] + low_slice[i] + close_slice[i]) / 4.0;
    }

    0
}

/// FFI wrapper for Rust Typical Price indicator
#[no_mangle]
pub unsafe extern "C" fn rust_typprice(
    size: c_int,
    high: *const c_double,
    low: *const c_double,
    close: *const c_double,
    output: *mut c_double,
) -> c_int {
    if high.is_null() || low.is_null() || close.is_null() || output.is_null() || size <= 0 {
        return -1;
    }

    let high_slice = slice::from_raw_parts(high, size as usize);
    let low_slice = slice::from_raw_parts(low, size as usize);
    let close_slice = slice::from_raw_parts(close, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    // Calculate typical price (HLC/3)
    for i in 0..size as usize {
        output_slice[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 3.0;
    }

    0
}

/// FFI wrapper for Rust LINEARREG_ANGLE indicator
#[no_mangle]
pub unsafe extern "C" fn rust_linearreg_angle(
    size: c_int,
    input: *const c_double,
    period: c_int,
    output: *mut c_double,
) -> c_int {
    if input.is_null() || output.is_null() || size <= 0 || period <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);

    let params = linearreg_angle::Linearreg_angleParams {
        period: Some(period as usize),
    };

    let angle_input = linearreg_angle::Linearreg_angleInput::from_slice(input_slice, params);

    match linearreg_angle::linearreg_angle(&angle_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}
