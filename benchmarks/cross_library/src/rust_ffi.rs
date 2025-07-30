use std::os::raw::{c_double, c_int};
use std::slice;
use my_project::indicators;

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
    
    let params = indicators::sma::SmaParams {
        period: Some(period as usize),
    };
    
    let sma_input = indicators::sma::SmaInput::from_slice(input_slice, params);
    
    match indicators::sma::sma(&sma_input) {
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
    
    let params = indicators::ema::EmaParams {
        period: Some(period as usize),
    };
    
    let ema_input = indicators::ema::EmaInput::from_slice(input_slice, params);
    
    match indicators::ema::ema(&ema_input) {
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
    
    let params = indicators::rsi::RsiParams {
        period: Some(period as usize),
    };
    
    let rsi_input = indicators::rsi::RsiInput::from_slice(input_slice, params);
    
    match indicators::rsi::rsi(&rsi_input) {
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
    
    let params = indicators::atr::AtrParams {
        period: Some(period as usize),
    };
    
    let atr_input = indicators::atr::AtrInput::from_slices(
        high_slice,
        low_slice,
        close_slice,
        params
    );
    
    match indicators::atr::atr(&atr_input) {
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
    
    let params = indicators::bollinger_bands::BollingerBandsParams {
        period: Some(period as usize),
        multiplier_upper: Some(stddev),
        multiplier_lower: Some(stddev),
        ma_type: Some("sma".to_string()),
        ddof: Some(0),
    };
    
    let bb_input = indicators::bollinger_bands::BollingerBandsInput::from_slice(input_slice, params);
    
    match indicators::bollinger_bands::bollinger_bands(&bb_input) {
        Ok(result) => {
            lower_slice.copy_from_slice(&result.lower);
            middle_slice.copy_from_slice(&result.middle);
            upper_slice.copy_from_slice(&result.upper);
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
    
    let params = indicators::macd::MacdParams {
        short_period: Some(short_period as usize),
        long_period: Some(long_period as usize),
        signal_period: Some(signal_period as usize),
    };
    
    let macd_input = indicators::macd::MacdInput::from_slice(input_slice, params);
    
    match indicators::macd::macd(&macd_input) {
        Ok(result) => {
            macd_slice.copy_from_slice(&result.macd);
            signal_slice.copy_from_slice(&result.signal);
            histogram_slice.copy_from_slice(&result.histogram);
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
    
    let params = indicators::adx::AdxParams {
        period: Some(period as usize),
    };
    
    let adx_input = indicators::adx::AdxInput::from_slices(
        high_slice,
        low_slice,
        close_slice,
        params
    );
    
    match indicators::adx::adx(&adx_input) {
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
    
    // Calculate typical price
    let mut typical_price = vec![0.0; size as usize];
    for i in 0..size as usize {
        typical_price[i] = (high_slice[i] + low_slice[i] + close_slice[i]) / 3.0;
    }
    
    let params = indicators::cci::CciParams {
        period: Some(period as usize),
    };
    
    let cci_input = indicators::cci::CciInput::from_slice(&typical_price, params);
    
    match indicators::cci::cci(&cci_input) {
        Ok(result) => {
            output_slice.copy_from_slice(&result.values);
            0
        }
        Err(_) => -1,
    }
}