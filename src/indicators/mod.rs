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
pub mod bollinger_bands_width;
pub mod bop;
pub mod cci;
pub mod cfo;
pub mod cg;
pub mod chande;
pub mod chop;
pub mod cksp;
pub mod cmo;
pub mod coppock;
pub mod correl_hl;
pub mod correlation_cycle;
pub mod cvi;
pub mod damiani_volatmeter;
pub mod dec_osc;
pub mod decycler;
pub mod deviation;
pub mod devstop;
pub mod di;
pub mod dm;
pub mod donchian;
pub mod dpo;
pub mod dti;
pub mod dx;
pub mod efi;
pub mod emd;
pub mod emv;
pub mod er;
pub mod eri;
pub mod fisher;
pub mod moving_averages;
pub mod roc;
pub mod rocp;
pub mod rocr;
pub mod rsi;
pub mod utility_functions;
pub use moving_averages::{
    alma, cwma, dema, edcf, ehlers_itrend, ema, epma, fwma, gaussian, highpass, highpass_2_pole,
    hma, hwma, jma, jsa, kama, linreg, maaq, mama, mwdx, nma, pwma, reflex, sinwma, sma, smma,
    sqwma, srwma, supersmoother, supersmoother_3_pole, swma, tema, tilson, trendflex, trima, vpwma,
    vwap, vwma, wilders, wma, zlema,
};
