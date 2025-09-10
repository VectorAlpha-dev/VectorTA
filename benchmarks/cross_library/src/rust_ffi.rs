use std::os::raw::{c_double, c_int};
use std::slice;
use my_project::indicators::moving_averages::{
    sma, ema, dema, tema, wma, kama, trima, hma, vwma, wilders, zlema
};
use my_project::indicators::{
    rsi, atr, bollinger_bands, macd, adx, cci, stoch, aroon,
    apo, cmo, dpo, mom, ppo, roc, willr, ad, adosc, obv, mfi,
    ao, bop, natr, stddev, var, ultosc
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