use std::os::raw::{c_double, c_int};

// Use the auto-generated bindings
use crate::*;

// We don't need to redeclare the functions - they're already in the generated bindings
// Just create safe wrappers using the existing declarations
// Remove all the extern declarations since they're in the generated bindings
/*
    pub fn TA_SMA(
        startIdx: c_int,
        endIdx: c_int,
        inReal: *const c_double,
        optInTimePeriod: c_int,
        outBegIdx: *mut c_int,
        outNBElement: *mut c_int,
        outReal: *mut c_double,
    ) -> c_int;

    // Exponential Moving Average
    pub fn TA_EMA(
        startIdx: c_int,
        endIdx: c_int,
        inReal: *const c_double,
        optInTimePeriod: c_int,
        outBegIdx: *mut c_int,
        outNBElement: *mut c_int,
        outReal: *mut c_double,
    ) -> c_int;

    // Relative Strength Index
    pub fn TA_RSI(
        startIdx: c_int,
        endIdx: c_int,
        inReal: *const c_double,
        optInTimePeriod: c_int,
        outBegIdx: *mut c_int,
        outNBElement: *mut c_int,
        outReal: *mut c_double,
    ) -> c_int;

    // Bollinger Bands
    pub fn TA_BBANDS(
        startIdx: c_int,
        endIdx: c_int,
        inReal: *const c_double,
        optInTimePeriod: c_int,
        optInNbDevUp: c_double,
        optInNbDevDn: c_double,
        optInMAType: TA_MAType,
        outBegIdx: *mut c_int,
        outNBElement: *mut c_int,
        outRealUpperBand: *mut c_double,
        outRealMiddleBand: *mut c_double,
        outRealLowerBand: *mut c_double,
    ) -> c_int;

    // MACD
    pub fn TA_MACD(
        startIdx: c_int,
        endIdx: c_int,
        inReal: *const c_double,
        optInFastPeriod: c_int,
        optInSlowPeriod: c_int,
        optInSignalPeriod: c_int,
        outBegIdx: *mut c_int,
        outNBElement: *mut c_int,
        outMACD: *mut c_double,
        outMACDSignal: *mut c_double,
        outMACDHist: *mut c_double,
    ) -> c_int;

    // Average True Range
    pub fn TA_ATR(
        startIdx: c_int,
        endIdx: c_int,
        inHigh: *const c_double,
        inLow: *const c_double,
        inClose: *const c_double,
        optInTimePeriod: c_int,
        outBegIdx: *mut c_int,
        outNBElement: *mut c_int,
        outReal: *mut c_double,
    ) -> c_int;

    // Stochastic
    pub fn TA_STOCH(
        startIdx: c_int,
        endIdx: c_int,
        inHigh: *const c_double,
        inLow: *const c_double,
        inClose: *const c_double,
        optInFastK_Period: c_int,
        optInSlowK_Period: c_int,
        optInSlowK_MAType: TA_MAType,
        optInSlowD_Period: c_int,
        optInSlowD_MAType: TA_MAType,
        outBegIdx: *mut c_int,
        outNBElement: *mut c_int,
        outSlowK: *mut c_double,
        outSlowD: *mut c_double,
    ) -> c_int;

    // Aroon
    pub fn TA_AROON(
        startIdx: c_int,
        endIdx: c_int,
        inHigh: *const c_double,
        inLow: *const c_double,
        optInTimePeriod: c_int,
        outBegIdx: *mut c_int,
        outNBElement: *mut c_int,
        outAroonDown: *mut c_double,
        outAroonUp: *mut c_double,
    ) -> c_int;

    // ADX
    pub fn TA_ADX(
        startIdx: c_int,
        endIdx: c_int,
        inHigh: *const c_double,
        inLow: *const c_double,
        inClose: *const c_double,
        optInTimePeriod: c_int,
        outBegIdx: *mut c_int,
        outNBElement: *mut c_int,
        outReal: *mut c_double,
    ) -> c_int;

    // CCI
    pub fn TA_CCI(
        startIdx: c_int,
        endIdx: c_int,
        inHigh: *const c_double,
        inLow: *const c_double,
        inClose: *const c_double,
        optInTimePeriod: c_int,
        outBegIdx: *mut c_int,
        outNBElement: *mut c_int,
        outReal: *mut c_double,
    ) -> c_int;

    // Additional Moving Averages
    pub fn TA_DEMA(
        startIdx: c_int,
        endIdx: c_int,
        inReal: *const c_double,
        optInTimePeriod: c_int,
        outBegIdx: *mut c_int,
        outNBElement: *mut c_int,
        outReal: *mut c_double,
    ) -> c_int;

    pub fn TA_TEMA(
        startIdx: c_int,
        endIdx: c_int,
        inReal: *const c_double,
        optInTimePeriod: c_int,
        outBegIdx: *mut c_int,
        outNBElement: *mut c_int,
        outReal: *mut c_double,
    ) -> c_int;

    pub fn TA_WMA(
        startIdx: c_int,
        endIdx: c_int,
        inReal: *const c_double,
        optInTimePeriod: c_int,
        outBegIdx: *mut c_int,
        outNBElement: *mut c_int,
        outReal: *mut c_double,
    ) -> c_int;

    pub fn TA_KAMA(
        startIdx: c_int,
        endIdx: c_int,
        inReal: *const c_double,
        optInTimePeriod: c_int,
        outBegIdx: *mut c_int,
        outNBElement: *mut c_int,
        outReal: *mut c_double,
    ) -> c_int;

    pub fn TA_TRIMA(
        startIdx: c_int,
        endIdx: c_int,
        inReal: *const c_double,
        optInTimePeriod: c_int,
        outBegIdx: *mut c_int,
        outNBElement: *mut c_int,
        outReal: *mut c_double,
    ) -> c_int;

    // HMA is not in TA-LIB, but we'll add other indicators

    // Momentum indicators
    pub fn TA_APO(
        startIdx: c_int,
        endIdx: c_int,
        inReal: *const c_double,
        optInFastPeriod: c_int,
        optInSlowPeriod: c_int,
        optInMAType: TA_MAType,
        outBegIdx: *mut c_int,
        outNBElement: *mut c_int,
        outReal: *mut c_double,
    ) -> c_int;

    pub fn TA_CMO(
        startIdx: c_int,
        endIdx: c_int,
        inReal: *const c_double,
        optInTimePeriod: c_int,
        outBegIdx: *mut c_int,
        outNBElement: *mut c_int,
        outReal: *mut c_double,
    ) -> c_int;

    pub fn TA_MOM(
        startIdx: c_int,
        endIdx: c_int,
        inReal: *const c_double,
        optInTimePeriod: c_int,
        outBegIdx: *mut c_int,
        outNBElement: *mut c_int,
        outReal: *mut c_double,
    ) -> c_int;

    pub fn TA_PPO(
        startIdx: c_int,
        endIdx: c_int,
        inReal: *const c_double,
        optInFastPeriod: c_int,
        optInSlowPeriod: c_int,
        optInMAType: TA_MAType,
        outBegIdx: *mut c_int,
        outNBElement: *mut c_int,
        outReal: *mut c_double,
    ) -> c_int;

    pub fn TA_ROC(
        startIdx: c_int,
        endIdx: c_int,
        inReal: *const c_double,
        optInTimePeriod: c_int,
        outBegIdx: *mut c_int,
        outNBElement: *mut c_int,
        outReal: *mut c_double,
    ) -> c_int;

    pub fn TA_ROCR(
        startIdx: c_int,
        endIdx: c_int,
        inReal: *const c_double,
        optInTimePeriod: c_int,
        outBegIdx: *mut c_int,
        outNBElement: *mut c_int,
        outReal: *mut c_double,
    ) -> c_int;

    pub fn TA_WILLR(
        startIdx: c_int,
        endIdx: c_int,
        inHigh: *const c_double,
        inLow: *const c_double,
        inClose: *const c_double,
        optInTimePeriod: c_int,
        outBegIdx: *mut c_int,
        outNBElement: *mut c_int,
        outReal: *mut c_double,
    ) -> c_int;

    // Volume indicators
    pub fn TA_AD(
        startIdx: c_int,
        endIdx: c_int,
        inHigh: *const c_double,
        inLow: *const c_double,
        inClose: *const c_double,
        inVolume: *const c_double,
        outBegIdx: *mut c_int,
        outNBElement: *mut c_int,
        outReal: *mut c_double,
    ) -> c_int;

    pub fn TA_ADOSC(
        startIdx: c_int,
        endIdx: c_int,
        inHigh: *const c_double,
        inLow: *const c_double,
        inClose: *const c_double,
        inVolume: *const c_double,
        optInFastPeriod: c_int,
        optInSlowPeriod: c_int,
        outBegIdx: *mut c_int,
        outNBElement: *mut c_int,
        outReal: *mut c_double,
    ) -> c_int;

    pub fn TA_OBV(
        startIdx: c_int,
        endIdx: c_int,
        inReal: *const c_double,
        inVolume: *const c_double,
        outBegIdx: *mut c_int,
        outNBElement: *mut c_int,
        outReal: *mut c_double,
    ) -> c_int;

    pub fn TA_MFI(
        startIdx: c_int,
        endIdx: c_int,
        inHigh: *const c_double,
        inLow: *const c_double,
        inClose: *const c_double,
        inVolume: *const c_double,
        optInTimePeriod: c_int,
        outBegIdx: *mut c_int,
        outNBElement: *mut c_int,
        outReal: *mut c_double,
    ) -> c_int;

    // Other indicators
    pub fn TA_NATR(
        startIdx: c_int,
        endIdx: c_int,
        inHigh: *const c_double,
        inLow: *const c_double,
        inClose: *const c_double,
        optInTimePeriod: c_int,
        outBegIdx: *mut c_int,
        outNBElement: *mut c_int,
        outReal: *mut c_double,
    ) -> c_int;

    pub fn TA_STDDEV(
        startIdx: c_int,
        endIdx: c_int,
        inReal: *const c_double,
        optInTimePeriod: c_int,
        optInNbDev: c_double,
        outBegIdx: *mut c_int,
        outNBElement: *mut c_int,
        outReal: *mut c_double,
    ) -> c_int;

    pub fn TA_VAR(
        startIdx: c_int,
        endIdx: c_int,
        inReal: *const c_double,
        optInTimePeriod: c_int,
        optInNbDev: c_double,
        outBegIdx: *mut c_int,
        outNBElement: *mut c_int,
        outReal: *mut c_double,
    ) -> c_int;

    pub fn TA_ULTOSC(
        startIdx: c_int,
        endIdx: c_int,
        inHigh: *const c_double,
        inLow: *const c_double,
        inClose: *const c_double,
        optInTimePeriod1: c_int,
        optInTimePeriod2: c_int,
        optInTimePeriod3: c_int,
        outBegIdx: *mut c_int,
        outNBElement: *mut c_int,
        outReal: *mut c_double,
    ) -> c_int;

    pub fn TA_BOP(
        startIdx: c_int,
        endIdx: c_int,
        inOpen: *const c_double,
        inHigh: *const c_double,
        inLow: *const c_double,
        inClose: *const c_double,
        outBegIdx: *mut c_int,
        outNBElement: *mut c_int,
        outReal: *mut c_double,
    ) -> c_int;

*/

// Safe wrapper functions for TA-LIB indicators
pub unsafe fn talib_sma(
    input: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = input.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_SMA(
        0,
        size - 1,
        input.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_SMA failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

pub unsafe fn talib_ema(
    input: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = input.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_EMA(
        0,
        size - 1,
        input.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_EMA failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

pub unsafe fn talib_rsi(
    input: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = input.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_RSI(
        0,
        size - 1,
        input.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_RSI failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

pub unsafe fn talib_bbands(
    input: &[f64],
    period: i32,
    devup: f64,
    devdn: f64,
    output_upper: &mut [f64],
    output_middle: &mut [f64],
    output_lower: &mut [f64],
) -> Result<(), String> {
    let size = input.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_BBANDS(
        0,
        size - 1,
        input.as_ptr(),
        period,
        devup,
        devdn,
        TA_MAType_TA_MAType_SMA,
        &mut out_beg_idx,
        &mut out_nb_element,
        output_upper.as_mut_ptr().add(out_beg_idx as usize),
        output_middle.as_mut_ptr().add(out_beg_idx as usize),
        output_lower.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_BBANDS failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output_upper[i] = f64::NAN;
        output_middle[i] = f64::NAN;
        output_lower[i] = f64::NAN;
    }

    Ok(())
}

pub unsafe fn talib_macd(
    input: &[f64],
    fast_period: i32,
    slow_period: i32,
    signal_period: i32,
    output_macd: &mut [f64],
    output_signal: &mut [f64],
    output_hist: &mut [f64],
) -> Result<(), String> {
    let size = input.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_MACD(
        0,
        size - 1,
        input.as_ptr(),
        fast_period,
        slow_period,
        signal_period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output_macd.as_mut_ptr().add(out_beg_idx as usize),
        output_signal.as_mut_ptr().add(out_beg_idx as usize),
        output_hist.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_MACD failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output_macd[i] = f64::NAN;
        output_signal[i] = f64::NAN;
        output_hist[i] = f64::NAN;
    }

    Ok(())
}

pub unsafe fn talib_atr(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = high.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_ATR(
        0,
        size - 1,
        high.as_ptr(),
        low.as_ptr(),
        close.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_ATR failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

pub unsafe fn talib_stoch(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    fastk_period: i32,
    slowk_period: i32,
    slowd_period: i32,
    output_k: &mut [f64],
    output_d: &mut [f64],
) -> Result<(), String> {
    let size = high.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_STOCH(
        0,
        size - 1,
        high.as_ptr(),
        low.as_ptr(),
        close.as_ptr(),
        fastk_period,
        slowk_period,
        TA_MAType_TA_MAType_SMA,
        slowd_period,
        TA_MAType_TA_MAType_SMA,
        &mut out_beg_idx,
        &mut out_nb_element,
        output_k.as_mut_ptr().add(out_beg_idx as usize),
        output_d.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_STOCH failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output_k[i] = f64::NAN;
        output_d[i] = f64::NAN;
    }

    Ok(())
}

// AD - Accumulation/Distribution Line
pub unsafe fn talib_ad(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    output: &mut [f64],
) -> Result<(), String> {
    let size = high.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_AD(
        0,
        size - 1,
        high.as_ptr(),
        low.as_ptr(),
        close.as_ptr(),
        volume.as_ptr(),
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_AD failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// ADOSC - Accumulation/Distribution Oscillator
pub unsafe fn talib_adosc(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    fast_period: i32,
    slow_period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = high.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_ADOSC(
        0,
        size - 1,
        high.as_ptr(),
        low.as_ptr(),
        close.as_ptr(),
        volume.as_ptr(),
        fast_period,
        slow_period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_ADOSC failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// ADX - Average Directional Movement Index
pub unsafe fn talib_adx(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = high.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_ADX(
        0,
        size - 1,
        high.as_ptr(),
        low.as_ptr(),
        close.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_ADX failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// ADXR - Average Directional Movement Index Rating
pub unsafe fn talib_adxr(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = high.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_ADXR(
        0,
        size - 1,
        high.as_ptr(),
        low.as_ptr(),
        close.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_ADXR failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// APO - Absolute Price Oscillator
pub unsafe fn talib_apo(
    input: &[f64],
    fast_period: i32,
    slow_period: i32,
    ma_type: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = input.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_APO(
        0,
        size - 1,
        input.as_ptr(),
        fast_period,
        slow_period,
        ma_type as TA_MAType,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_APO failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// AROON - Aroon
pub unsafe fn talib_aroon(
    high: &[f64],
    low: &[f64],
    period: i32,
    output_down: &mut [f64],
    output_up: &mut [f64],
) -> Result<(), String> {
    let size = high.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_AROON(
        0,
        size - 1,
        high.as_ptr(),
        low.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output_down.as_mut_ptr().add(out_beg_idx as usize),
        output_up.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_AROON failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output_down[i] = f64::NAN;
        output_up[i] = f64::NAN;
    }

    Ok(())
}

// AROONOSC - Aroon Oscillator
pub unsafe fn talib_aroonosc(
    high: &[f64],
    low: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = high.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_AROONOSC(
        0,
        size - 1,
        high.as_ptr(),
        low.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_AROONOSC failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// BOP - Balance of Power
pub unsafe fn talib_bop(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    output: &mut [f64],
) -> Result<(), String> {
    let size = high.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_BOP(
        0,
        size - 1,
        open.as_ptr(),
        high.as_ptr(),
        low.as_ptr(),
        close.as_ptr(),
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_BOP failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// CCI - Commodity Channel Index
pub unsafe fn talib_cci(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = high.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_CCI(
        0,
        size - 1,
        high.as_ptr(),
        low.as_ptr(),
        close.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_CCI failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// CMO - Chande Momentum Oscillator
pub unsafe fn talib_cmo(
    input: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = input.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_CMO(
        0,
        size - 1,
        input.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_CMO failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// DEMA - Double Exponential Moving Average
pub unsafe fn talib_dema(
    input: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = input.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_DEMA(
        0,
        size - 1,
        input.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_DEMA failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// DX - Directional Movement Index
pub unsafe fn talib_dx(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = high.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_DX(
        0,
        size - 1,
        high.as_ptr(),
        low.as_ptr(),
        close.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_DX failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// PLUS_DI - Plus Directional Indicator
pub unsafe fn talib_plus_di(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = high.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_PLUS_DI(
        0,
        size - 1,
        high.as_ptr(),
        low.as_ptr(),
        close.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_PLUS_DI failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// MINUS_DI - Minus Directional Indicator
pub unsafe fn talib_minus_di(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = high.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_MINUS_DI(
        0,
        size - 1,
        high.as_ptr(),
        low.as_ptr(),
        close.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_MINUS_DI failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// PLUS_DM - Plus Directional Movement
pub unsafe fn talib_plus_dm(
    high: &[f64],
    low: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = high.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_PLUS_DM(
        0,
        size - 1,
        high.as_ptr(),
        low.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_PLUS_DM failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// MINUS_DM - Minus Directional Movement
pub unsafe fn talib_minus_dm(
    high: &[f64],
    low: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = high.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_MINUS_DM(
        0,
        size - 1,
        high.as_ptr(),
        low.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_MINUS_DM failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// KAMA - Kaufman Adaptive Moving Average
pub unsafe fn talib_kama(
    input: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = input.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_KAMA(
        0,
        size - 1,
        input.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_KAMA failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// LINEARREG - Linear Regression
pub unsafe fn talib_linearreg(
    input: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = input.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_LINEARREG(
        0,
        size - 1,
        input.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_LINEARREG failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// MEDPRICE - Median Price
pub unsafe fn talib_medprice(
    high: &[f64],
    low: &[f64],
    output: &mut [f64],
) -> Result<(), String> {
    let size = high.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_MEDPRICE(
        0,
        size - 1,
        high.as_ptr(),
        low.as_ptr(),
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_MEDPRICE failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// MFI - Money Flow Index
pub unsafe fn talib_mfi(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = high.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_MFI(
        0,
        size - 1,
        high.as_ptr(),
        low.as_ptr(),
        close.as_ptr(),
        volume.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_MFI failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// MIDPOINT - Midpoint over period
pub unsafe fn talib_midpoint(
    input: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = input.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_MIDPOINT(
        0,
        size - 1,
        input.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_MIDPOINT failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// MIDPRICE - Midpoint Price over period
pub unsafe fn talib_midprice(
    high: &[f64],
    low: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = high.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_MIDPRICE(
        0,
        size - 1,
        high.as_ptr(),
        low.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_MIDPRICE failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// MOM - Momentum
pub unsafe fn talib_mom(
    input: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = input.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_MOM(
        0,
        size - 1,
        input.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_MOM failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// NATR - Normalized Average True Range
pub unsafe fn talib_natr(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = high.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_NATR(
        0,
        size - 1,
        high.as_ptr(),
        low.as_ptr(),
        close.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_NATR failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// OBV - On Balance Volume
pub unsafe fn talib_obv(
    input: &[f64],
    volume: &[f64],
    output: &mut [f64],
) -> Result<(), String> {
    let size = input.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_OBV(
        0,
        size - 1,
        input.as_ptr(),
        volume.as_ptr(),
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_OBV failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// PPO - Percentage Price Oscillator
pub unsafe fn talib_ppo(
    input: &[f64],
    fast_period: i32,
    slow_period: i32,
    ma_type: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = input.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_PPO(
        0,
        size - 1,
        input.as_ptr(),
        fast_period,
        slow_period,
        ma_type as TA_MAType,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_PPO failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// ROC - Rate of change
pub unsafe fn talib_roc(
    input: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = input.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_ROC(
        0,
        size - 1,
        input.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_ROC failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// ROCP - Rate of change Percentage
pub unsafe fn talib_rocp(
    input: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = input.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_ROCP(
        0,
        size - 1,
        input.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_ROCP failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// ROCR - Rate of change ratio
pub unsafe fn talib_rocr(
    input: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = input.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_ROCR(
        0,
        size - 1,
        input.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_ROCR failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// SAR - Parabolic SAR
pub unsafe fn talib_sar(
    high: &[f64],
    low: &[f64],
    acceleration: f64,
    maximum: f64,
    output: &mut [f64],
) -> Result<(), String> {
    let size = high.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_SAR(
        0,
        size - 1,
        high.as_ptr(),
        low.as_ptr(),
        acceleration,
        maximum,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_SAR failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// STDDEV - Standard Deviation
pub unsafe fn talib_stddev(
    input: &[f64],
    period: i32,
    nb_dev: f64,
    output: &mut [f64],
) -> Result<(), String> {
    let size = input.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_STDDEV(
        0,
        size - 1,
        input.as_ptr(),
        period,
        nb_dev,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_STDDEV failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// STOCHF - Stochastic Fast
pub unsafe fn talib_stochf(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    fastk_period: i32,
    fastd_period: i32,
    fastd_matype: i32,
    output_k: &mut [f64],
    output_d: &mut [f64],
) -> Result<(), String> {
    let size = high.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_STOCHF(
        0,
        size - 1,
        high.as_ptr(),
        low.as_ptr(),
        close.as_ptr(),
        fastk_period,
        fastd_period,
        fastd_matype as TA_MAType,
        &mut out_beg_idx,
        &mut out_nb_element,
        output_k.as_mut_ptr().add(out_beg_idx as usize),
        output_d.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_STOCHF failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output_k[i] = f64::NAN;
        output_d[i] = f64::NAN;
    }

    Ok(())
}

// STOCHRSI - Stochastic Relative Strength Index
pub unsafe fn talib_stochrsi(
    input: &[f64],
    period: i32,
    fastk_period: i32,
    fastd_period: i32,
    fastd_matype: i32,
    output_k: &mut [f64],
    output_d: &mut [f64],
) -> Result<(), String> {
    let size = input.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_STOCHRSI(
        0,
        size - 1,
        input.as_ptr(),
        period,
        fastk_period,
        fastd_period,
        fastd_matype as TA_MAType,
        &mut out_beg_idx,
        &mut out_nb_element,
        output_k.as_mut_ptr().add(out_beg_idx as usize),
        output_d.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_STOCHRSI failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output_k[i] = f64::NAN;
        output_d[i] = f64::NAN;
    }

    Ok(())
}

// TEMA - Triple Exponential Moving Average
pub unsafe fn talib_tema(
    input: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = input.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_TEMA(
        0,
        size - 1,
        input.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_TEMA failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// TRIMA - Triangular Moving Average
pub unsafe fn talib_trima(
    input: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = input.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_TRIMA(
        0,
        size - 1,
        input.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_TRIMA failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
pub unsafe fn talib_trix(
    input: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = input.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_TRIX(
        0,
        size - 1,
        input.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_TRIX failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// TSF - Time Series Forecast
pub unsafe fn talib_tsf(
    input: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = input.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_TSF(
        0,
        size - 1,
        input.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_TSF failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// ULTOSC - Ultimate Oscillator
pub unsafe fn talib_ultosc(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period1: i32,
    period2: i32,
    period3: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = high.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_ULTOSC(
        0,
        size - 1,
        high.as_ptr(),
        low.as_ptr(),
        close.as_ptr(),
        period1,
        period2,
        period3,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_ULTOSC failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// VAR - Variance
pub unsafe fn talib_var(
    input: &[f64],
    period: i32,
    nb_dev: f64,
    output: &mut [f64],
) -> Result<(), String> {
    let size = input.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_VAR(
        0,
        size - 1,
        input.as_ptr(),
        period,
        nb_dev,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_VAR failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// WCLPRICE - Weighted Close Price
pub unsafe fn talib_wclprice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    output: &mut [f64],
) -> Result<(), String> {
    let size = high.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_WCLPRICE(
        0,
        size - 1,
        high.as_ptr(),
        low.as_ptr(),
        close.as_ptr(),
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_WCLPRICE failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// WILLR - Williams' %R
pub unsafe fn talib_willr(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = high.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_WILLR(
        0,
        size - 1,
        high.as_ptr(),
        low.as_ptr(),
        close.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_WILLR failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// WMA - Weighted Moving Average
pub unsafe fn talib_wma(
    input: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = input.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_WMA(
        0,
        size - 1,
        input.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_WMA failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// LINEARREG_SLOPE - Linear Regression Slope
pub unsafe fn talib_linearreg_slope(
    input: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = input.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_LINEARREG_SLOPE(
        0,
        size - 1,
        input.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_LINEARREG_SLOPE failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// LINEARREG_INTERCEPT - Linear Regression Intercept
pub unsafe fn talib_linearreg_intercept(
    input: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = input.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_LINEARREG_INTERCEPT(
        0,
        size - 1,
        input.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_LINEARREG_INTERCEPT failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// LINEARREG_ANGLE - Linear Regression Angle
pub unsafe fn talib_linearreg_angle(
    input: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = input.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_LINEARREG_ANGLE(
        0,
        size - 1,
        input.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_LINEARREG_ANGLE failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// TYPPRICE - Typical Price
pub unsafe fn talib_typprice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    output: &mut [f64],
) -> Result<(), String> {
    let size = high.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_TYPPRICE(
        0,
        size - 1,
        high.as_ptr(),
        low.as_ptr(),
        close.as_ptr(),
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_TYPPRICE failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// AVGPRICE - Average Price
pub unsafe fn talib_avgprice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    output: &mut [f64],
) -> Result<(), String> {
    let size = high.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_AVGPRICE(
        0,
        size - 1,
        open.as_ptr(),
        high.as_ptr(),
        low.as_ptr(),
        close.as_ptr(),
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_AVGPRICE failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// TRANGE - True Range
pub unsafe fn talib_trange(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    output: &mut [f64],
) -> Result<(), String> {
    let size = high.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_TRANGE(
        0,
        size - 1,
        high.as_ptr(),
        low.as_ptr(),
        close.as_ptr(),
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_TRANGE failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// MIN - Lowest value over a period
pub unsafe fn talib_min(
    input: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = input.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_MIN(
        0,
        size - 1,
        input.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_MIN failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// MAX - Highest value over a period
pub unsafe fn talib_max(
    input: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = input.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_MAX(
        0,
        size - 1,
        input.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_MAX failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}

// SUM - Summation
pub unsafe fn talib_sum(
    input: &[f64],
    period: i32,
    output: &mut [f64],
) -> Result<(), String> {
    let size = input.len() as c_int;
    let mut out_beg_idx: c_int = 0;
    let mut out_nb_element: c_int = 0;

    let ret = TA_SUM(
        0,
        size - 1,
        input.as_ptr(),
        period,
        &mut out_beg_idx,
        &mut out_nb_element,
        output.as_mut_ptr().add(out_beg_idx as usize),
    );

    if ret != TA_RetCode_TA_SUCCESS {
        return Err(format!("TA_SUM failed with code {:?}", ret));
    }

    // Fill the beginning with NaN
    for i in 0..out_beg_idx as usize {
        output[i] = f64::NAN;
    }

    Ok(())
}