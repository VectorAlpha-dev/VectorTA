pub mod acosc;
pub mod ad;
pub mod adosc;
pub mod adx;
pub mod adxr;
pub mod alligator;
pub mod ao;
pub mod apo;
pub mod aroon;
pub mod aroonosc;
pub mod atr;
pub mod avgprice;
pub mod bandpass;
pub mod bollinger_bands;
pub mod bop;
pub mod cci;
pub mod deviation;
pub mod moving_averages;
pub mod roc;
pub mod rocp;
pub mod rsi;
pub use moving_averages::{
    alma, cwma, dema, edcf, ehlers_itrend, ema, epma, fwma, gaussian, highpass, highpass_2_pole,
    hma, hwma, jma, jsa, kama, linreg, maaq, mama, mwdx, nma, pwma, reflex, sinwma, sma, smma,
    sqwma, srwma, supersmoother, supersmoother_3_pole, swma, tema, tilson, trendflex, trima, vpwma,
    vwap, vwma, wilders, wma, zlema,
};
