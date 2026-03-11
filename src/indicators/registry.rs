use crate::indicators::moving_averages::param_schema::{ma_param_schema, MaParamKind};
use crate::indicators::moving_averages::registry::list_moving_averages;
use once_cell::sync::Lazy;
use serde::Serialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IndicatorParamKind {
    Int,
    Float,
    Bool,
    EnumString,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamValueStatic {
    Int(i64),
    Float(f64),
    Bool(bool),
    EnumString(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IndicatorValueType {
    F64,
    F32,
    I32,
    Bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IndicatorInputKind {
    Slice,
    Candles,
    Ohlc,
    Ohlcv,
    HighLow,
    CloseVolume,
}

#[derive(Debug, Clone, Serialize)]
pub struct IndicatorParamInfo {
    pub key: &'static str,
    pub label: &'static str,
    pub kind: IndicatorParamKind,
    pub required: bool,
    pub default: Option<ParamValueStatic>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub step: Option<f64>,
    pub enum_values: &'static [&'static str],
    pub notes: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct IndicatorOutputInfo {
    pub id: &'static str,
    pub label: &'static str,
    pub value_type: IndicatorValueType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct IndicatorCapabilities {
    pub supports_cpu_single: bool,
    pub supports_cpu_batch: bool,
    pub supports_cuda_single: bool,
    pub supports_cuda_batch: bool,
    pub supports_cuda_vram: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct IndicatorInfo {
    pub id: &'static str,
    pub label: &'static str,
    pub category: &'static str,
    pub dynamic_strategy_eligible: bool,
    pub input_kind: IndicatorInputKind,
    pub outputs: Vec<IndicatorOutputInfo>,
    pub params: Vec<IndicatorParamInfo>,
    pub capabilities: IndicatorCapabilities,
    pub notes: Option<&'static str>,
}

const BUCKET_B_INDICATORS: &[&str] = &[
    "acosc",
    "adaptive_bounds_rsi",
    "adjustable_ma_alternating_extremities",
    "alligator",
    "alphatrend",
    "aroon",
    "aso",
    "bandpass",
    "bollinger_bands_width",
    "chande",
    "chandelier_exit",
    "cksp",
    "correlation_cycle",
    "damiani_volatmeter",
    "di",
    "dm",
    "donchian",
    "ehlers_adaptive_cg",
    "adaptive_momentum_oscillator",
    "dvdiqqe",
    "exponential_trend",
    "trend_flow_trail",
    "range_breakout_signals",
    "emd",
    "eri",
    "fisher",
    "forward_backward_exponential_oscillator",
    "fvg_trailing_stop",
    "gatorosc",
    "halftrend",
    "hypertrend",
    "adaptive_schaff_trend_cycle",
    "smoothed_gaussian_trend_filter",
    "ict_propulsion_block",
    "kdj",
    "keltner",
    "kst",
    "lpc",
    "mab",
    "macz",
    "mama",
    "minmax",
    "msw",
    "nadaraya_watson_envelope",
    "otto",
    "pma",
    "prb",
    "qqe",
    "qqe_weighted_oscillator",
    "market_structure_confluence",
    "range_filtered_trend_signals",
    "range_oscillator",
    "volume_weighted_relative_strength_index",
    "range_filter",
    "rsmk",
    "squeeze_momentum",
    "srsi",
    "supertrend",
    "supertrend_oscillator",
    "vi",
    "voss",
    "wavetrend",
    "wto",
    "ehlers_pma",
    "buff_averages",
    "vwap",
    "pivot",
    "normalized_volume_true_range",
];

const EMPTY_ENUM_VALUES: &[&str] = &[];
const ENUM_VALUES_TRUE_FALSE: &[&str] = &["true", "false"];
const ENUM_VALUES_CLOSE_WICK: &[&str] = &["close", "wick"];
const ENUM_VALUES_MA_OUTPUT: &[&str] = &["mama", "fama"];
const ENUM_VALUES_PMA_OUTPUT: &[&str] = &["predict", "trigger"];
const ENUM_VALUES_EHLERS_ADAPTIVE_CG_OUTPUT: &[&str] = &["cg", "trigger"];
const ENUM_VALUES_ADAPTIVE_MOMENTUM_OSCILLATOR_OUTPUT: &[&str] = &["amo", "ama"];
const ENUM_VALUES_BUFF_OUTPUT: &[&str] = &["fast", "slow"];
const ENUM_VALUES_EDCT3_OUTPUT: &[&str] = &["corrected", "t3"];
const ENUM_VALUES_LMA_SMOOTHING: &[&str] = &["ema", "sma", "wma", "rma", "vwma"];
const ENUM_VALUES_VDO_SESSION_MODE: &[&str] =
    &["4_hours", "daily", "weekly", "rolling_bars", "rolling_days"];
const ENUM_VALUES_VDO_DEVIATION_MODE: &[&str] = &["percent", "absolute", "zscore"];
const ENUM_VALUES_ICHI_NORMALIZE_MODE: &[&str] = &["all", "window", "disabled"];
const ENUM_VALUES_PMARP_MA_TYPE: &[&str] = &["sma", "ema", "hma", "rma", "vwma"];
const ENUM_VALUES_PMARP_LINE_MODE: &[&str] = &["pmar", "pmarp"];
const ENUM_VALUES_SMOOTH_THEIL_SEN_STAT_STYLE: &[&str] = &["mean", "smooth_median", "median"];
const ENUM_VALUES_SMOOTH_THEIL_SEN_DEVIATION_STYLE: &[&str] = &["mad", "rmsd"];
const ENUM_VALUES_BULLS_V_BEARS_MA_TYPE: &[&str] = &["ema", "sma", "wma"];
const ENUM_VALUES_BULLS_V_BEARS_CALCULATION_METHOD: &[&str] = &["normalized", "raw"];
const ENUM_VALUES_MOVING_AVERAGE_CROSS_PROBABILITY_MA_TYPE: &[&str] = &["ema", "sma"];
const ENUM_VALUES_PRICE_SOURCE: &[&str] = &[
    "open", "high", "low", "close", "volume", "hl2", "hlc3", "ohlc4", "hlcc4", "hlcc",
];
const ENUM_VALUES_NORMALIZED_VOLUME_TRUE_RANGE_OUTPUT: &[&str] = &[
    "normalized_volume",
    "normalized_true_range",
    "baseline",
    "atr",
    "average_volume",
];
const ENUM_VALUES_STATISTICAL_TRAILING_STOP_BASE_LEVEL: &[&str] =
    &["level0", "level1", "level2", "level3"];

const OUTPUT_VALUE_F64: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "value",
    label: "Value",
    value_type: IndicatorValueType::F64,
};

const OUTPUT_VALUE_BOOL: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "value",
    label: "Value",
    value_type: IndicatorValueType::Bool,
};

const OUTPUT_MATRIX: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "matrix",
    label: "Matrix",
    value_type: IndicatorValueType::Bool,
};

const OUTPUT_MACD: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "macd",
    label: "MACD",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_SIGNAL: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "signal",
    label: "Signal",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_EMA: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "ema",
    label: "EMA",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_CG: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "cg",
    label: "CG",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_AMO: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "amo",
    label: "AMO",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_AMA: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "ama",
    label: "AMA",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_HIST: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "hist",
    label: "Histogram",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_UPPER: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "upper",
    label: "Upper",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_MIDDLE: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "middle",
    label: "Middle",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_AVERAGE: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "average",
    label: "Average",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_LOWER: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "lower",
    label: "Lower",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_K: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "k",
    label: "K",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_D: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "d",
    label: "D",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_VPCI: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "vpci",
    label: "VPCI",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_VPCIS: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "vpcis",
    label: "VPCIS",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_MOMENTUM: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "momentum",
    label: "Momentum",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_SQUEEZE: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "squeeze",
    label: "Squeeze",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_MAMA: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "mama",
    label: "MAMA",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_FAMA: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "fama",
    label: "FAMA",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_PREDICT: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "predict",
    label: "Predict",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_TRIGGER: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "trigger",
    label: "Trigger",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_CORRECTED: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "corrected",
    label: "Corrected",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_T3: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "t3",
    label: "T3",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_CYCLE: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "cycle",
    label: "Cycle",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_BULL: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "bull",
    label: "Bull",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_BEAR: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "bear",
    label: "Bear",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_FAST: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "fast",
    label: "Fast",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_SLOW: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "slow",
    label: "Slow",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_PLUS: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "plus",
    label: "Plus",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_MINUS: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "minus",
    label: "Minus",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_UP: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "up",
    label: "Up",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_DOWN: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "down",
    label: "Down",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_LEVEL: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "level",
    label: "Level",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_BAND: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "band",
    label: "Band",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_ANCHOR_LINE: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "anchor",
    label: "Anchor",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_SWITCH_PRICE_LINE: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "switch_price",
    label: "Switch Price",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_BIAS: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "bias",
    label: "Bias",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_TREND: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "trend",
    label: "Trend",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_CHANGED: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "changed",
    label: "Changed",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_J: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "j",
    label: "J",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_MOMENTUM_SIGNAL: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "signal",
    label: "Signal",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_WT1: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "wt1",
    label: "WT1",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_WT2: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "wt2",
    label: "WT2",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_WT_DIFF: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "wt_diff",
    label: "WT Diff",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_WAVETREND1: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "wavetrend1",
    label: "WaveTrend 1",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_WAVETREND2: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "wavetrend2",
    label: "WaveTrend 2",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_WAVETREND: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "wavetrend",
    label: "WaveTrend",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_HISTOGRAM: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "histogram",
    label: "Histogram",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_YZ: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "yz",
    label: "YZ",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_RS: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "rs",
    label: "RS",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_VOLATILITY: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "volatility",
    label: "Volatility",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_VARIANCE: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "variance",
    label: "Variance",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_TMF: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "tmf",
    label: "TMF",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_SMOOTHED: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "smoothed",
    label: "Smoothed",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_HVP: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "hvp",
    label: "HVP",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_HVP_SMA: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "hvp_sma",
    label: "HVP SMA",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_EDF: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "edf",
    label: "EDF",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_KBW: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "kbw",
    label: "KBW",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_KBW_SMA: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "kbw_sma",
    label: "KBW SMA",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_MMI: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "mmi",
    label: "MMI",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_MMI_SMOOTHED: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "mmi_smoothed",
    label: "MMI Smoothed",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_PRICE_DENSITY: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "price_density",
    label: "Price Density",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_PRICE_DENSITY_PERCENT: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "price_density_percent",
    label: "Price Density Percent",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_IMPULSE_MACD: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "impulse_macd",
    label: "Impulse MACD",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_MOMENTUM_RATIO_OSCILLATOR: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "line",
    label: "Momentum Ratio Oscillator",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_SUPERTREND_OSCILLATOR: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "oscillator",
    label: "Oscillator",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_CONV_ACCELERATION: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "conv_acceleration",
    label: "Convolution Acceleration",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_PLUS_TCF: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "plus_tcf",
    label: "Plus TCF",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_MINUS_TCF: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "minus_tcf",
    label: "Minus TCF",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_VQI_SUM: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "vqi_sum",
    label: "VQI Sum",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_FAST_SMA: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "fast_sma",
    label: "Fast SMA",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_SLOW_SMA: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "slow_sma",
    label: "Slow SMA",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_SLOPE: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "slope",
    label: "Slope",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_INTERCEPT: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "intercept",
    label: "Intercept",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_DEVIATION: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "deviation",
    label: "Deviation",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_MA: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "ma",
    label: "MA",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_BULLISH_SIGNAL: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "bullish_signal",
    label: "Bullish Signal",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_BEARISH_SIGNAL: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "bearish_signal",
    label: "Bearish Signal",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_ZERO_CROSS_UP: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "zero_cross_up",
    label: "Zero Cross Up",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_ZERO_CROSS_DOWN: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "zero_cross_down",
    label: "Zero Cross Down",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_BULLISH_REVERSAL: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "bullish_reversal",
    label: "Bullish Reversal",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_BEARISH_REVERSAL: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "bearish_reversal",
    label: "Bearish Reversal",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_FORECAST: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "forecast",
    label: "Forecast",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_DIRECTION: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "direction",
    label: "Direction",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_EXTREMITY: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "extremity",
    label: "Extremity",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_STATE: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "state",
    label: "State",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_SMOOTHED_OPEN: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "smoothed_open",
    label: "Smoothed Open",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_SMOOTHED_HIGH: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "smoothed_high",
    label: "Smoothed High",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_SMOOTHED_LOW: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "smoothed_low",
    label: "Smoothed Low",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_SMOOTHED_CLOSE: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "smoothed_close",
    label: "Smoothed Close",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_RSI_LINE: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "rsi",
    label: "RSI",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_TRAILING_STOP: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "trailing_stop",
    label: "Trailing Stop",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_FORWARD_BACKWARD: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "forward_backward",
    label: "Forward-Backward",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_BACKWARD_LINE: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "backward",
    label: "Backward",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_LOWER_MID: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "lower_mid",
    label: "Lower Mid",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_UPPER_MID: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "upper_mid",
    label: "Upper Mid",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_REGIME: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "regime",
    label: "Regime",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_REGIME_FLIP: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "regime_flip",
    label: "Regime Flip",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_LOWER_SIGNAL: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "lower_signal",
    label: "Lower Signal",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_UPPER_SIGNAL: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "upper_signal",
    label: "Upper Signal",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_NORMALIZED_VOLUME: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "normalized_volume",
    label: "Normalized Volume",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_NORMALIZED_TRUE_RANGE: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "normalized_true_range",
    label: "Normalized True Range",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_BASELINE: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "baseline",
    label: "Baseline",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_AVERAGE_VOLUME: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "average_volume",
    label: "Average Volume",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_ATR_LINE: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "atr",
    label: "ATR",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_RANGE_TOP: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "range_top",
    label: "Range Top",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_RANGE_BOTTOM: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "range_bottom",
    label: "Range Bottom",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_BULLISH: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "bullish",
    label: "Bullish",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_EXTRA_BULLISH: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "extra_bullish",
    label: "Extra Bullish",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_BEARISH: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "bearish",
    label: "Bearish",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_EXTRA_BEARISH: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "extra_bearish",
    label: "Extra Bearish",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_UPTREND_BASE: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "uptrend_base",
    label: "Uptrend Base",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_DOWNTREND_BASE: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "downtrend_base",
    label: "Downtrend Base",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_UPTREND_EXTENSION: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "uptrend_extension",
    label: "Uptrend Extension",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_DOWNTREND_EXTENSION: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "downtrend_extension",
    label: "Downtrend Extension",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_BULLISH_CHANGE: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "bullish_change",
    label: "Bullish Change",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_BEARISH_CHANGE: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "bearish_change",
    label: "Bearish Change",
    value_type: IndicatorValueType::F64,
};

const OUTPUTS_VALUE_F64: &[IndicatorOutputInfo] = &[OUTPUT_VALUE_F64];
const OUTPUTS_VALUE_BOOL: &[IndicatorOutputInfo] = &[OUTPUT_VALUE_BOOL];
const OUTPUTS_MATRIX_BOOL: &[IndicatorOutputInfo] = &[OUTPUT_MATRIX];
const OUTPUTS_MACD: &[IndicatorOutputInfo] = &[OUTPUT_MACD, OUTPUT_SIGNAL, OUTPUT_HIST];
const OUTPUTS_BOLLINGER: &[IndicatorOutputInfo] = &[OUTPUT_UPPER, OUTPUT_MIDDLE, OUTPUT_LOWER];
const OUTPUTS_STOCH: &[IndicatorOutputInfo] = &[OUTPUT_K, OUTPUT_D];
const OUTPUTS_VPCI: &[IndicatorOutputInfo] = &[OUTPUT_VPCI, OUTPUT_VPCIS];
const OUTPUTS_TTM_SQUEEZE: &[IndicatorOutputInfo] = &[OUTPUT_MOMENTUM, OUTPUT_SQUEEZE];
const OUTPUTS_MAMA: &[IndicatorOutputInfo] = &[OUTPUT_MAMA, OUTPUT_FAMA];
const OUTPUTS_EHLERS_PMA: &[IndicatorOutputInfo] = &[OUTPUT_PREDICT, OUTPUT_TRIGGER];
const OUTPUTS_EHLERS_ADAPTIVE_CG: &[IndicatorOutputInfo] = &[OUTPUT_CG, OUTPUT_TRIGGER];
const OUTPUTS_ADAPTIVE_MOMENTUM_OSCILLATOR: &[IndicatorOutputInfo] = &[OUTPUT_AMO, OUTPUT_AMA];
const OUTPUTS_BUFF_AVERAGES: &[IndicatorOutputInfo] = &[OUTPUT_FAST, OUTPUT_SLOW];
const OUTPUTS_EDCT3: &[IndicatorOutputInfo] = &[OUTPUT_CORRECTED, OUTPUT_T3];
const OUTPUTS_PLUS_MINUS: &[IndicatorOutputInfo] = &[OUTPUT_PLUS, OUTPUT_MINUS];
const OUTPUTS_UP_DOWN: &[IndicatorOutputInfo] = &[OUTPUT_UP, OUTPUT_DOWN];
const OUTPUTS_STATISTICAL_TRAILING_STOP: &[IndicatorOutputInfo] = &[
    OUTPUT_LEVEL,
    OUTPUT_ANCHOR_LINE,
    OUTPUT_BIAS,
    OUTPUT_CHANGED,
];
const OUTPUTS_SUPERTREND_RECOVERY: &[IndicatorOutputInfo] = &[
    OUTPUT_BAND,
    OUTPUT_SWITCH_PRICE_LINE,
    OUTPUT_TREND,
    OUTPUT_CHANGED,
];
const OUTPUTS_RANGE_BREAKOUT_SIGNALS: &[IndicatorOutputInfo] = &[
    OUTPUT_RANGE_TOP,
    OUTPUT_RANGE_BOTTOM,
    OUTPUT_BULLISH,
    OUTPUT_EXTRA_BULLISH,
    OUTPUT_BEARISH,
    OUTPUT_EXTRA_BEARISH,
];
const OUTPUTS_EXPONENTIAL_TREND: &[IndicatorOutputInfo] = &[
    OUTPUT_UPTREND_BASE,
    OUTPUT_DOWNTREND_BASE,
    OUTPUT_UPTREND_EXTENSION,
    OUTPUT_DOWNTREND_EXTENSION,
    OUTPUT_BULLISH_CHANGE,
    OUTPUT_BEARISH_CHANGE,
];
const OUTPUT_ALPHA_TRAIL: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "alpha_trail",
    label: "AlphaTrail",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_ALPHA_TRAIL_BULLISH: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "alpha_trail_bullish",
    label: "AlphaTrail Bullish",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_ALPHA_TRAIL_BEARISH: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "alpha_trail_bearish",
    label: "AlphaTrail Bearish",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_ALPHA_DIR: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "alpha_dir",
    label: "AlphaTrail Direction",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_MFI: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "mfi",
    label: "Money Flow Index",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_TP_UPPER: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "tp_upper",
    label: "TP Upper",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_TP_LOWER: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "tp_lower",
    label: "TP Lower",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_ALPHA_TRAIL_BULLISH_SWITCH: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "alpha_trail_bullish_switch",
    label: "AlphaTrail Bullish Switch",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_ALPHA_TRAIL_BEARISH_SWITCH: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "alpha_trail_bearish_switch",
    label: "AlphaTrail Bearish Switch",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_MFI_OVERBOUGHT: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "mfi_overbought",
    label: "MFI Overbought",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_MFI_OVERSOLD: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "mfi_oversold",
    label: "MFI Oversold",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_MFI_CROSS_UP_MID: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "mfi_cross_up_mid",
    label: "MFI Cross Up Mid",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_MFI_CROSS_DOWN_MID: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "mfi_cross_down_mid",
    label: "MFI Cross Down Mid",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_PRICE_CROSS_ALPHA_TRAIL_UP: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "price_cross_alpha_trail_up",
    label: "Price Cross AlphaTrail Up",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_PRICE_CROSS_ALPHA_TRAIL_DOWN: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "price_cross_alpha_trail_down",
    label: "Price Cross AlphaTrail Down",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_MFI_ABOVE_90: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "mfi_above_90",
    label: "MFI Above 90",
    value_type: IndicatorValueType::F64,
};
const OUTPUT_MFI_BELOW_10: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "mfi_below_10",
    label: "MFI Below 10",
    value_type: IndicatorValueType::F64,
};
const OUTPUTS_TREND_FLOW_TRAIL: &[IndicatorOutputInfo] = &[
    OUTPUT_ALPHA_TRAIL,
    OUTPUT_ALPHA_TRAIL_BULLISH,
    OUTPUT_ALPHA_TRAIL_BEARISH,
    OUTPUT_ALPHA_DIR,
    OUTPUT_MFI,
    OUTPUT_TP_UPPER,
    OUTPUT_TP_LOWER,
    OUTPUT_ALPHA_TRAIL_BULLISH_SWITCH,
    OUTPUT_ALPHA_TRAIL_BEARISH_SWITCH,
    OUTPUT_MFI_OVERBOUGHT,
    OUTPUT_MFI_OVERSOLD,
    OUTPUT_MFI_CROSS_UP_MID,
    OUTPUT_MFI_CROSS_DOWN_MID,
    OUTPUT_PRICE_CROSS_ALPHA_TRAIL_UP,
    OUTPUT_PRICE_CROSS_ALPHA_TRAIL_DOWN,
    OUTPUT_MFI_ABOVE_90,
    OUTPUT_MFI_BELOW_10,
];
const OUTPUTS_STANDARDIZED_PSAR_OSCILLATOR: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "oscillator",
        label: "Oscillator",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "ma",
        label: "MA",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "bullish_reversal",
        label: "Bullish Reversal",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "bearish_reversal",
        label: "Bearish Reversal",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "regular_bullish",
        label: "Regular Bullish",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "regular_bearish",
        label: "Regular Bearish",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "bullish_weakening",
        label: "Bullish Weakening",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "bearish_weakening",
        label: "Bearish Weakening",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_VDUBUS_DIVERGENCE_WAVE_PATTERN_GENERATOR: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "fast_standard",
        label: "Fast Standard",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "fast_climax",
        label: "Fast Climax",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "fast_rounded",
        label: "Fast Rounded",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "fast_predator",
        label: "Fast Predator",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "slow_standard",
        label: "Slow Standard",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "slow_climax",
        label: "Slow Climax",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "slow_rounded",
        label: "Slow Rounded",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "slow_predator",
        label: "Slow Predator",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "opposing_force",
        label: "Opposing Force",
        value_type: IndicatorValueType::F64,
    },
    OUTPUT_MACD,
    OUTPUT_SIGNAL,
    OUTPUT_HIST,
];
const OUTPUTS_TREND_CHANGED: &[IndicatorOutputInfo] = &[OUTPUT_TREND, OUTPUT_CHANGED];
const OUTPUTS_KDJ: &[IndicatorOutputInfo] = &[OUTPUT_K, OUTPUT_D, OUTPUT_J];
const OUTPUTS_SQUEEZE_MOMENTUM: &[IndicatorOutputInfo] =
    &[OUTPUT_MOMENTUM, OUTPUT_SQUEEZE, OUTPUT_MOMENTUM_SIGNAL];
const OUTPUTS_WTO: &[IndicatorOutputInfo] =
    &[OUTPUT_WAVETREND1, OUTPUT_WAVETREND2, OUTPUT_HISTOGRAM];
const OUTPUTS_WAVETREND: &[IndicatorOutputInfo] = &[OUTPUT_WT1, OUTPUT_WT2, OUTPUT_WT_DIFF];
const OUTPUTS_MOD_GOD_MODE: &[IndicatorOutputInfo] =
    &[OUTPUT_WAVETREND, OUTPUT_SIGNAL, OUTPUT_HISTOGRAM];
const OUTPUTS_YANG_ZHANG: &[IndicatorOutputInfo] = &[OUTPUT_YZ, OUTPUT_RS];
const OUTPUTS_PARKINSON: &[IndicatorOutputInfo] = &[OUTPUT_VOLATILITY, OUTPUT_VARIANCE];
const OUTPUTS_TWIGGS_MONEY_FLOW: &[IndicatorOutputInfo] = &[OUTPUT_TMF, OUTPUT_SMOOTHED];
const OUTPUTS_HVP: &[IndicatorOutputInfo] = &[OUTPUT_HVP, OUTPUT_HVP_SMA];
const OUTPUTS_EHLERS_DETRENDING_FILTER: &[IndicatorOutputInfo] = &[OUTPUT_EDF, OUTPUT_SIGNAL];
const OUTPUTS_KELTNER_CHANNEL_WIDTH_OSCILLATOR: &[IndicatorOutputInfo] =
    &[OUTPUT_KBW, OUTPUT_KBW_SMA];
const OUTPUTS_MARKET_MEANNESS_INDEX: &[IndicatorOutputInfo] = &[OUTPUT_MMI, OUTPUT_MMI_SMOOTHED];
const OUTPUTS_PRICE_DENSITY_MARKET_NOISE: &[IndicatorOutputInfo] =
    &[OUTPUT_PRICE_DENSITY, OUTPUT_PRICE_DENSITY_PERCENT];
const OUTPUTS_IMPULSE_MACD: &[IndicatorOutputInfo] =
    &[OUTPUT_IMPULSE_MACD, OUTPUT_HISTOGRAM, OUTPUT_SIGNAL];
const OUTPUTS_MOMENTUM_RATIO_OSCILLATOR: &[IndicatorOutputInfo] =
    &[OUTPUT_MOMENTUM_RATIO_OSCILLATOR, OUTPUT_SIGNAL];
const OUTPUTS_SUPERTREND_OSCILLATOR: &[IndicatorOutputInfo] = &[
    OUTPUT_SUPERTREND_OSCILLATOR,
    OUTPUT_SIGNAL,
    OUTPUT_HISTOGRAM,
];
const OUTPUTS_HYPERTREND: &[IndicatorOutputInfo] = &[
    OUTPUT_AVERAGE,
    OUTPUT_UPPER,
    OUTPUT_LOWER,
    OUTPUT_TREND,
    OUTPUT_CHANGED,
];
const OUTPUTS_LOGARITHMIC_MOVING_AVERAGE: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "lma",
        label: "LMA",
        value_type: IndicatorValueType::F64,
    },
    OUTPUT_SIGNAL,
    IndicatorOutputInfo {
        id: "position",
        label: "Position",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "momentum_confirmed",
        label: "Momentum Confirmed",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_ADAPTIVE_SCHAFF_TREND_CYCLE: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "stc",
        label: "STC",
        value_type: IndicatorValueType::F64,
    },
    OUTPUT_HISTOGRAM,
];
const OUTPUTS_SMOOTHED_GAUSSIAN_TREND_FILTER: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "filter",
        label: "Filter",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "supertrend",
        label: "SuperTrend",
        value_type: IndicatorValueType::F64,
    },
    OUTPUT_TREND,
    IndicatorOutputInfo {
        id: "ranging",
        label: "Ranging",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_STOCHASTIC_ADAPTIVE_D: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "standard_d",
        label: "Standard %D",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "adaptive_d",
        label: "Adaptive %D",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "difference",
        label: "Difference",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_ICT_PROPULSION_BLOCK: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "bullish_high",
        label: "Bullish High",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "bullish_low",
        label: "Bullish Low",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "bullish_kind",
        label: "Bullish Kind",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "bullish_active",
        label: "Bullish Active",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "bullish_mitigated",
        label: "Bullish Mitigated",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "bullish_new",
        label: "Bullish New",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "bearish_high",
        label: "Bearish High",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "bearish_low",
        label: "Bearish Low",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "bearish_kind",
        label: "Bearish Kind",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "bearish_active",
        label: "Bearish Active",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "bearish_mitigated",
        label: "Bearish Mitigated",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "bearish_new",
        label: "Bearish New",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_LEAVITT_CONVOLUTION_ACCELERATION: &[IndicatorOutputInfo] =
    &[OUTPUT_CONV_ACCELERATION, OUTPUT_SIGNAL];
const OUTPUTS_TREND_CONTINUATION_FACTOR: &[IndicatorOutputInfo] =
    &[OUTPUT_PLUS_TCF, OUTPUT_MINUS_TCF];
const OUTPUTS_VOLUME_WEIGHTED_STOCHASTIC_RSI: &[IndicatorOutputInfo] = &[OUTPUT_K, OUTPUT_D];
const OUTPUTS_VOLATILITY_QUALITY_INDEX: &[IndicatorOutputInfo] =
    &[OUTPUT_VQI_SUM, OUTPUT_FAST_SMA, OUTPUT_SLOW_SMA];
const OUTPUTS_ANDEAN_OSCILLATOR: &[IndicatorOutputInfo] =
    &[OUTPUT_BULL, OUTPUT_BEAR, OUTPUT_SIGNAL];
const OUTPUTS_CYCLE_CHANNEL_OSCILLATOR: &[IndicatorOutputInfo] = &[OUTPUT_FAST, OUTPUT_SLOW];
const OUTPUTS_DAILY_FACTOR: &[IndicatorOutputInfo] = &[OUTPUT_VALUE_F64, OUTPUT_EMA, OUTPUT_SIGNAL];
const OUTPUTS_MOVING_AVERAGE_CROSS_PROBABILITY: &[IndicatorOutputInfo] = &[
    OUTPUT_VALUE_F64,
    IndicatorOutputInfo {
        id: "slow_ma",
        label: "Slow MA",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "fast_ma",
        label: "Fast MA",
        value_type: IndicatorValueType::F64,
    },
    OUTPUT_FORECAST,
    OUTPUT_UPPER,
    OUTPUT_LOWER,
    OUTPUT_DIRECTION,
];
const OUTPUTS_BULLS_V_BEARS: &[IndicatorOutputInfo] = &[
    OUTPUT_VALUE_F64,
    OUTPUT_BULL,
    OUTPUT_BEAR,
    OUTPUT_MA,
    OUTPUT_UPPER,
    OUTPUT_LOWER,
    OUTPUT_BULLISH_SIGNAL,
    OUTPUT_BEARISH_SIGNAL,
    OUTPUT_ZERO_CROSS_UP,
    OUTPUT_ZERO_CROSS_DOWN,
];
const OUTPUTS_ADJUSTABLE_MA_ALTERNATING_EXTREMITIES: &[IndicatorOutputInfo] = &[
    OUTPUT_MA,
    OUTPUT_UPPER,
    OUTPUT_LOWER,
    OUTPUT_EXTREMITY,
    OUTPUT_STATE,
    OUTPUT_CHANGED,
    OUTPUT_SMOOTHED_OPEN,
    OUTPUT_SMOOTHED_HIGH,
    OUTPUT_SMOOTHED_LOW,
    OUTPUT_SMOOTHED_CLOSE,
];
const OUTPUTS_QQE_WEIGHTED_OSCILLATOR: &[IndicatorOutputInfo] =
    &[OUTPUT_RSI_LINE, OUTPUT_TRAILING_STOP];
const OUTPUTS_ADAPTIVE_BOUNDS_RSI: &[IndicatorOutputInfo] = &[
    OUTPUT_RSI_LINE,
    OUTPUT_LOWER,
    OUTPUT_LOWER_MID,
    OUTPUT_MIDDLE,
    OUTPUT_UPPER_MID,
    OUTPUT_UPPER,
    OUTPUT_REGIME,
    OUTPUT_REGIME_FLIP,
    OUTPUT_LOWER_SIGNAL,
    OUTPUT_UPPER_SIGNAL,
];
const OUTPUTS_FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR: &[IndicatorOutputInfo] = &[
    OUTPUT_FORWARD_BACKWARD,
    OUTPUT_BACKWARD_LINE,
    OUTPUT_HISTOGRAM,
];
const OUTPUTS_RANGE_OSCILLATOR: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "oscillator",
        label: "Oscillator",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "ma",
        label: "MA",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "upper_band",
        label: "Upper Band",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "lower_band",
        label: "Lower Band",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "range_width",
        label: "Range Width",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "in_range",
        label: "In Range",
        value_type: IndicatorValueType::F64,
    },
    OUTPUT_TREND,
    IndicatorOutputInfo {
        id: "break_up",
        label: "Break Up",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "break_down",
        label: "Break Down",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_RANGE_FILTERED_TREND_SIGNALS: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "kalman",
        label: "Kalman",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "supertrend",
        label: "Supertrend",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "upper_band",
        label: "Upper Band",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "lower_band",
        label: "Lower Band",
        value_type: IndicatorValueType::F64,
    },
    OUTPUT_TREND,
    IndicatorOutputInfo {
        id: "kalman_trend",
        label: "Kalman Trend",
        value_type: IndicatorValueType::F64,
    },
    OUTPUT_STATE,
    IndicatorOutputInfo {
        id: "market_trending",
        label: "Market Trending",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "market_ranging",
        label: "Market Ranging",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "short_term_bullish",
        label: "Short Term Bullish",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "short_term_bearish",
        label: "Short Term Bearish",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "long_term_bullish",
        label: "Long Term Bullish",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "long_term_bearish",
        label: "Long Term Bearish",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_MARKET_STRUCTURE_CONFLUENCE: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "basis",
        label: "Basis",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "upper_band",
        label: "Upper Band",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "lower_band",
        label: "Lower Band",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "structure_direction",
        label: "Structure Direction",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "bullish_arrow",
        label: "Bullish Arrow",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "bearish_arrow",
        label: "Bearish Arrow",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "bullish_change",
        label: "Bullish Change",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "bearish_change",
        label: "Bearish Change",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "hh",
        label: "HH",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "lh",
        label: "LH",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "hl",
        label: "HL",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "ll",
        label: "LL",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "bullish_bos",
        label: "Bullish BOS",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "bullish_choch",
        label: "Bullish CHoCH",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "bearish_bos",
        label: "Bearish BOS",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "bearish_choch",
        label: "Bearish CHoCH",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_VOLUME_WEIGHTED_RELATIVE_STRENGTH_INDEX: &[IndicatorOutputInfo] = &[
    OUTPUT_RSI_LINE,
    IndicatorOutputInfo {
        id: "consolidation_strength",
        label: "Consolidation Strength",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "rsi_ma",
        label: "RSI MA",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "bearish_tp",
        label: "Bearish TP",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "bullish_tp",
        label: "Bullish TP",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_SMOOTH_THEIL_SEN: &[IndicatorOutputInfo] = &[
    OUTPUT_VALUE_F64,
    OUTPUT_UPPER,
    OUTPUT_LOWER,
    OUTPUT_SLOPE,
    OUTPUT_INTERCEPT,
    OUTPUT_DEVIATION,
];
const OUTPUTS_REGRESSION_SLOPE_OSCILLATOR: &[IndicatorOutputInfo] = &[
    OUTPUT_VALUE_F64,
    OUTPUT_SIGNAL,
    OUTPUT_BULLISH_REVERSAL,
    OUTPUT_BEARISH_REVERSAL,
];
const OUTPUTS_EHLERS_SIMPLE_CYCLE_INDICATOR: &[IndicatorOutputInfo] =
    &[OUTPUT_CYCLE, OUTPUT_TRIGGER];
const OUTPUTS_RANDOM_WALK_INDEX: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "high",
        label: "High",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "low",
        label: "Low",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_PRICE_MOVING_AVERAGE_RATIO_PERCENTILE: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "pmar",
        label: "PMAR",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "pmarp",
        label: "PMARP",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "plotline",
        label: "Plotline",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "signal",
        label: "Signal",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "pmar_high",
        label: "Historical PMAR High",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "pmar_low",
        label: "Historical PMAR Low",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "scaled_pmar",
        label: "Scaled PMAR",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_RELATIVE_STRENGTH_INDEX_WAVE_INDICATOR: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "rsi_ma1",
        label: "RSI MA1",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "rsi_ma2",
        label: "RSI MA2",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "rsi_ma3",
        label: "RSI MA3",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "rsi_ma4",
        label: "RSI MA4",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "state",
        label: "State",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_MESA_STOCHASTIC_MULTI_LENGTH: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "mesa_1",
        label: "MESA 1",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "mesa_2",
        label: "MESA 2",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "mesa_3",
        label: "MESA 3",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "mesa_4",
        label: "MESA 4",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "trigger_1",
        label: "Trigger 1",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "trigger_2",
        label: "Trigger 2",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "trigger_3",
        label: "Trigger 3",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "trigger_4",
        label: "Trigger 4",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_SPEARMAN_CORRELATION: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "raw",
        label: "Raw",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "smoothed",
        label: "Smoothed",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_VWAP_DEVIATION_OSCILLATOR: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "osc",
        label: "Oscillator",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "std1",
        label: "Std 1",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "std2",
        label: "Std 2",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "std3",
        label: "Std 3",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_ICHIMOKU_OSCILLATOR: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "signal",
        label: "Signal",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "ma",
        label: "MA",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "conversion",
        label: "Conversion",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "base",
        label: "Base",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "chikou",
        label: "Chikou",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "current_kumo_a",
        label: "Current Kumo A",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "current_kumo_b",
        label: "Current Kumo B",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "future_kumo_a",
        label: "Future Kumo A",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "future_kumo_b",
        label: "Future Kumo B",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "max_level",
        label: "Max Level",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "high_level",
        label: "High Level",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "low_level",
        label: "Low Level",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "min_level",
        label: "Min Level",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_NORMALIZED_VOLUME_TRUE_RANGE: &[IndicatorOutputInfo] = &[
    OUTPUT_NORMALIZED_VOLUME,
    OUTPUT_NORMALIZED_TRUE_RANGE,
    OUTPUT_BASELINE,
    OUTPUT_ATR_LINE,
    OUTPUT_AVERAGE_VOLUME,
];
const OUTPUTS_ACOSC: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "osc",
        label: "Oscillator",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "change",
        label: "Change",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_ALLIGATOR: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "jaw",
        label: "Jaw",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "teeth",
        label: "Teeth",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "lips",
        label: "Lips",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_K1_K2: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "k1",
        label: "K1",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "k2",
        label: "K2",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_BULLS_BEARS: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "bulls",
        label: "Bulls",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "bears",
        label: "Bears",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_BANDPASS: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "bp",
        label: "BandPass",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "bp_normalized",
        label: "Normalized",
        value_type: IndicatorValueType::F64,
    },
    OUTPUT_SIGNAL,
    OUTPUT_TRIGGER,
];
const OUTPUTS_LONG_SHORT_STOP: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "long_stop",
        label: "Long Stop",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "short_stop",
        label: "Short Stop",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_LONG_SHORT_VALUES: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "long_values",
        label: "Long",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "short_values",
        label: "Short",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_CORRELATION_CYCLE: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "real",
        label: "Real",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "imag",
        label: "Imag",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "angle",
        label: "Angle",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "state",
        label: "State",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_VOL_ANTI: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "vol",
        label: "Vol",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "anti",
        label: "Anti",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_DVDIQQE: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "dvdi",
        label: "DVDI",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "fast_tl",
        label: "Fast TL",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "slow_tl",
        label: "Slow TL",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "center_line",
        label: "Center Line",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_UPPER_MIDDLE_LOWER_BAND: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "upperband",
        label: "Upper",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "middleband",
        label: "Middle",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "lowerband",
        label: "Lower",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_BULL_BEAR: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "bull",
        label: "Bull",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "bear",
        label: "Bear",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_FVG_TS: &[IndicatorOutputInfo] = &[
    OUTPUT_UPPER,
    OUTPUT_LOWER,
    IndicatorOutputInfo {
        id: "upper_ts",
        label: "Upper TS",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "lower_ts",
        label: "Lower TS",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_GATOROSC: &[IndicatorOutputInfo] = &[
    OUTPUT_UPPER,
    OUTPUT_LOWER,
    IndicatorOutputInfo {
        id: "upper_change",
        label: "Upper Change",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "lower_change",
        label: "Lower Change",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_HALFTREND: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "halftrend",
        label: "HalfTrend",
        value_type: IndicatorValueType::F64,
    },
    OUTPUT_TREND,
    IndicatorOutputInfo {
        id: "atr_high",
        label: "ATR High",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "atr_low",
        label: "ATR Low",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "buy_signal",
        label: "Buy",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "sell_signal",
        label: "Sell",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_LINE_SIGNAL: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "line",
        label: "Line",
        value_type: IndicatorValueType::F64,
    },
    OUTPUT_SIGNAL,
];
const OUTPUTS_FISHER: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "fisher",
        label: "Fisher",
        value_type: IndicatorValueType::F64,
    },
    OUTPUT_SIGNAL,
];
const OUTPUTS_UPPER_LOWER: &[IndicatorOutputInfo] = &[OUTPUT_UPPER, OUTPUT_LOWER];
const OUTPUTS_FILTER_BANDS: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "filter",
        label: "Filter",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "high_band",
        label: "High Band",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "low_band",
        label: "Low Band",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_MINMAX: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "is_min",
        label: "Is Min",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "is_max",
        label: "Is Max",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "last_min",
        label: "Last Min",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "last_max",
        label: "Last Max",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_SINE_LEAD: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "sine",
        label: "Sine",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "lead",
        label: "Lead",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_HOTT_LOTT: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "hott",
        label: "HOTT",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "lott",
        label: "LOTT",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_PRB: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "values",
        label: "Value",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "upper_band",
        label: "Upper",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "lower_band",
        label: "Lower",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_INDICATOR_SIGNAL: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "indicator",
        label: "Indicator",
        value_type: IndicatorValueType::F64,
    },
    OUTPUT_SIGNAL,
];
const OUTPUTS_VOSS: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "voss",
        label: "Voss",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "filt",
        label: "Filter",
        value_type: IndicatorValueType::F64,
    },
];
const OUTPUTS_PIVOT: &[IndicatorOutputInfo] = &[
    IndicatorOutputInfo {
        id: "pp",
        label: "PP",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "r1",
        label: "R1",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "r2",
        label: "R2",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "r3",
        label: "R3",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "r4",
        label: "R4",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "s1",
        label: "S1",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "s2",
        label: "S2",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "s3",
        label: "S3",
        value_type: IndicatorValueType::F64,
    },
    IndicatorOutputInfo {
        id: "s4",
        label: "S4",
        value_type: IndicatorValueType::F64,
    },
];

const PARAM_PERIOD: IndicatorParamInfo = IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: true,
    default: Some(ParamValueStatic::Int(14)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
};

const PARAMS_PERIOD_ONLY: &[IndicatorParamInfo] = &[PARAM_PERIOD];

const PARAM_OUTPUT_MAMA: IndicatorParamInfo = IndicatorParamInfo {
    key: "output",
    label: "Output",
    kind: IndicatorParamKind::EnumString,
    required: false,
    default: Some(ParamValueStatic::EnumString("mama")),
    min: None,
    max: None,
    step: None,
    enum_values: ENUM_VALUES_MA_OUTPUT,
    notes: None,
};

const PARAM_OUTPUT_EHLERS_PMA: IndicatorParamInfo = IndicatorParamInfo {
    key: "output",
    label: "Output",
    kind: IndicatorParamKind::EnumString,
    required: false,
    default: Some(ParamValueStatic::EnumString("predict")),
    min: None,
    max: None,
    step: None,
    enum_values: ENUM_VALUES_PMA_OUTPUT,
    notes: None,
};

const PARAM_OUTPUT_EHLERS_ADAPTIVE_CG: IndicatorParamInfo = IndicatorParamInfo {
    key: "output",
    label: "Output",
    kind: IndicatorParamKind::EnumString,
    required: false,
    default: Some(ParamValueStatic::EnumString("cg")),
    min: None,
    max: None,
    step: None,
    enum_values: ENUM_VALUES_EHLERS_ADAPTIVE_CG_OUTPUT,
    notes: None,
};

const PARAM_OUTPUT_ADAPTIVE_MOMENTUM_OSCILLATOR: IndicatorParamInfo = IndicatorParamInfo {
    key: "output",
    label: "Output",
    kind: IndicatorParamKind::EnumString,
    required: false,
    default: Some(ParamValueStatic::EnumString("amo")),
    min: None,
    max: None,
    step: None,
    enum_values: ENUM_VALUES_ADAPTIVE_MOMENTUM_OSCILLATOR_OUTPUT,
    notes: None,
};

const PARAM_OUTPUT_BUFF_AVERAGES: IndicatorParamInfo = IndicatorParamInfo {
    key: "output",
    label: "Output",
    kind: IndicatorParamKind::EnumString,
    required: false,
    default: Some(ParamValueStatic::EnumString("fast")),
    min: None,
    max: None,
    step: None,
    enum_values: ENUM_VALUES_BUFF_OUTPUT,
    notes: None,
};

const PARAM_OUTPUT_EDCT3: IndicatorParamInfo = IndicatorParamInfo {
    key: "output",
    label: "Output",
    kind: IndicatorParamKind::EnumString,
    required: false,
    default: Some(ParamValueStatic::EnumString("corrected")),
    min: None,
    max: None,
    step: None,
    enum_values: ENUM_VALUES_EDCT3_OUTPUT,
    notes: None,
};

const PARAM_OUTPUT_NORMALIZED_VOLUME_TRUE_RANGE: IndicatorParamInfo = IndicatorParamInfo {
    key: "output",
    label: "Output",
    kind: IndicatorParamKind::EnumString,
    required: false,
    default: Some(ParamValueStatic::EnumString("normalized_volume")),
    min: None,
    max: None,
    step: None,
    enum_values: ENUM_VALUES_NORMALIZED_VOLUME_TRUE_RANGE_OUTPUT,
    notes: None,
};

const PARAM_ANCHOR: IndicatorParamInfo = IndicatorParamInfo {
    key: "anchor",
    label: "Anchor",
    kind: IndicatorParamKind::EnumString,
    required: false,
    default: Some(ParamValueStatic::EnumString("1d")),
    min: None,
    max: None,
    step: None,
    enum_values: EMPTY_ENUM_VALUES,
    notes: Some("Anchor string for session boundary"),
};

const PARAM_STRICT: IndicatorParamInfo = IndicatorParamInfo {
    key: "strict",
    label: "Strict",
    kind: IndicatorParamKind::Bool,
    required: false,
    default: Some(ParamValueStatic::Bool(false)),
    min: None,
    max: None,
    step: None,
    enum_values: ENUM_VALUES_TRUE_FALSE,
    notes: None,
};

const PARAM_NONE: &[IndicatorParamInfo] = &[];

const PARAM_RSI_PERIOD: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(14)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_ROC_PERIOD: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(9)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_LINEAR_CORRELATION_OSCILLATOR_PERIOD: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(14)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_EHLERS_FM_DEMODULATOR_PERIOD: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(30)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: Some("Published default is 30."),
}];

const PARAM_ADOSC: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "short_period",
        label: "Short Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(3)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "long_period",
        label: "Long Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_AO: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "short_period",
        label: "Short Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(5)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "long_period",
        label: "Long Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(34)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_EFI_PERIOD: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(13)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_MFI_PERIOD: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(14)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_MASS_PERIOD: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(5)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_KVO: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "short_period",
        label: "Short Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(2)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "long_period",
        label: "Long Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(5)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_VOSC: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "short_period",
        label: "Short Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(2)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "long_period",
        label: "Long Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(5)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_MOM_PERIOD: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(10)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_CMO_PERIOD: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(14)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_ROCP_PERIOD: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(10)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_ROCR_PERIOD: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(10)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_PPO: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "fast_period",
        label: "Fast Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(12)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "slow_period",
        label: "Slow Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(26)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "ma_type",
        label: "MA Type",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("sma")),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_TRIX_PERIOD: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(18)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_TSI: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "long_period",
        label: "Long Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(25)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "short_period",
        label: "Short Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(13)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_STDDEV: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "period",
        label: "Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(5)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "nbdev",
        label: "NB Dev",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(1.0)),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_WILLR_PERIOD: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(14)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_ULTOSC: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "timeperiod1",
        label: "Time Period 1",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(7)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "timeperiod2",
        label: "Time Period 2",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "timeperiod3",
        label: "Time Period 3",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(28)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_RANGE_BREAKOUT_SIGNALS: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "range_length",
        label: "Range Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(20)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "confirmation_length",
        label: "Confirmation Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(5)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Uses bar-direction volume as the up/down volume proxy because TradingView lower-timeframe volume requests are unavailable in vector-ta."),
    },
];

const PARAM_EXPONENTIAL_TREND: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "exp_rate",
        label: "Exponential Rate",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(0.00003)),
        min: Some(0.0),
        max: Some(0.5),
        step: Some(0.01),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "initial_distance",
        label: "Initial Distance",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(4.0)),
        min: Some(0.1),
        max: None,
        step: Some(0.1),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("The supplied Pine source seeds trend state at bar index 100 after a fixed ATR(14)-based supertrend-style initialization."),
    },
    IndicatorParamInfo {
        key: "width_multiplier",
        label: "Width Multiplier",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(1.0)),
        min: Some(0.1),
        max: None,
        step: Some(0.1),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];
const PARAM_TREND_FLOW_TRAIL: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "alpha_length",
        label: "AlphaTrail Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(33)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: &[],
        notes: None,
    },
    IndicatorParamInfo {
        key: "alpha_multiplier",
        label: "AlphaTrail Multiplier",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(3.3)),
        min: Some(0.1),
        max: None,
        step: Some(0.1),
        enum_values: &[],
        notes: None,
    },
    IndicatorParamInfo {
        key: "mfi_length",
        label: "MFI Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: Some(2000.0),
        step: Some(1.0),
        enum_values: &[],
        notes: None,
    },
];

const PARAM_APO: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "short_period",
        label: "Short Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "long_period",
        label: "Long Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(20)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_CCI_PERIOD: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(14)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_CCI_CYCLE: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "length",
        label: "Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "factor",
        label: "Factor",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(0.5)),
        min: Some(0.0),
        max: Some(1.0),
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_CFO: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "period",
        label: "Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "scalar",
        label: "Scalar",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(100.0)),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_ER_PERIOD: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(5)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_KURTOSIS_PERIOD: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(5)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_NATR_PERIOD: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(14)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_REVERSE_RSI: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "rsi_length",
        label: "RSI Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "rsi_level",
        label: "RSI Level",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(50.0)),
        min: Some(0.0),
        max: Some(100.0),
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_QSTICK: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(5)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_MEAN_AD_PERIOD: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(5)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_MEDIUM_AD_PERIOD: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(5)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_DEVIATION: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "period",
        label: "Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(9)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "devtype",
        label: "Dev Type",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(0)),
        min: Some(0.0),
        max: Some(2.0),
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_DPO_PERIOD: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(5)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_LRSI_ALPHA: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "alpha",
    label: "Alpha",
    kind: IndicatorParamKind::Float,
    required: false,
    default: Some(ParamValueStatic::Float(0.2)),
    min: Some(0.0),
    max: Some(1.0),
    step: None,
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_VELOCITY: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "length",
        label: "Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(21)),
        min: Some(2.0),
        max: Some(60.0),
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("TradingView script default source is HLCC4."),
    },
    IndicatorParamInfo {
        key: "smooth_length",
        label: "Smoothing Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(5)),
        min: Some(1.0),
        max: Some(9.0),
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_ADAPTIVE_MOMENTUM_OSCILLATOR: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "length",
        label: "Data Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "smoothing_length",
        label: "Smoothing Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(9)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    PARAM_OUTPUT_ADAPTIVE_MOMENTUM_OSCILLATOR,
];

const PARAM_NORMALIZED_VOLUME_TRUE_RANGE: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "true_range_style",
        label: "True Range Style",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("body")),
        min: None,
        max: None,
        step: None,
        enum_values: &["body", "hl", "delta"],
        notes: None,
    },
    IndicatorParamInfo {
        key: "outlier_range",
        label: "Outlier Range",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(5.0)),
        min: Some(0.5),
        max: None,
        step: Some(0.25),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "atr_length",
        label: "ATR Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(2.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "volume_length",
        label: "Average Volume Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(2.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    PARAM_OUTPUT_NORMALIZED_VOLUME_TRUE_RANGE,
];

const PARAM_PVI: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "initial_value",
    label: "Initial Value",
    kind: IndicatorParamKind::Float,
    required: false,
    default: Some(ParamValueStatic::Float(1000.0)),
    min: None,
    max: None,
    step: None,
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_PFE: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "period",
        label: "Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "smoothing",
        label: "Smoothing",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(5)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_PERCENTILE_NEAREST_RANK: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "length",
        label: "Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(15)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "percentage",
        label: "Percentage",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(50.0)),
        min: Some(0.0),
        max: Some(100.0),
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_UI: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "period",
        label: "Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "scalar",
        label: "Scalar",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(100.0)),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_ZSCORE: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "period",
        label: "Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "ma_type",
        label: "MA Type",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("sma")),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "nbdev",
        label: "NB Dev",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(1.0)),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "devtype",
        label: "Dev Type",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(0)),
        min: Some(0.0),
        max: Some(2.0),
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_MIDPOINT_PERIOD: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(14)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_MIDPRICE_PERIOD: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(14)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_TSF_PERIOD: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(14)),
    min: Some(2.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_VAR: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "period",
        label: "Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "nbdev",
        label: "NB Dev",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(1.0)),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_ADX_PERIOD: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(14)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_DX_PERIOD: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(14)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_ATR_PERIOD: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "length",
    label: "Length",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(14)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_PSYCHOLOGICAL_LINE: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "length",
    label: "Length",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(20)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_RANK_CORRELATION_INDEX: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "length",
    label: "Length",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(12)),
    min: Some(2.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_LINEAR_REGRESSION_INTENSITY: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "lookback_period",
        label: "Lookback Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(12)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "range_tolerance",
        label: "Range Tolerance",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(90.0)),
        min: Some(0.0),
        max: Some(100.0),
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "linreg_length",
        label: "Linear Regression Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(90)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_TREND_FOLLOWER: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "matype",
        label: "MA Type",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("ema")),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "trend_period",
        label: "Trend Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(20)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "ma_period",
        label: "MA Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(20)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "channel_rate_percent",
        label: "Channel Rate %",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(1.0)),
        min: Some(0.0000001),
        max: None,
        step: Some(0.1),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "use_linear_regression",
        label: "Use Linear Regression",
        kind: IndicatorParamKind::Bool,
        required: false,
        default: Some(ParamValueStatic::Bool(true)),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_TRUE_FALSE,
        notes: None,
    },
    IndicatorParamInfo {
        key: "linear_regression_period",
        label: "Linear Regression Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(5)),
        min: Some(2.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_TREND_DIRECTION_FORCE_INDEX: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "length",
    label: "Length",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(10)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_TREND_CONTINUATION_FACTOR: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "length",
    label: "Length",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(35)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_EWMA_VOLATILITY: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "lambda",
    label: "Lambda",
    kind: IndicatorParamKind::Float,
    required: false,
    default: Some(ParamValueStatic::Float(0.94)),
    min: Some(0.0),
    max: Some(0.999_999_999),
    step: Some(0.01),
    enum_values: EMPTY_ENUM_VALUES,
    notes: Some("Pine-faithful EMA mapping from lambda to rounded EMA length"),
}];

const PARAM_ACCUMULATION_SWING_INDEX: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "daily_limit",
    label: "Daily Limit",
    kind: IndicatorParamKind::Float,
    required: false,
    default: Some(ParamValueStatic::Float(10_000.0)),
    min: Some(0.000_000_001),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: Some("Pine-faithful Wilder ASI daily limit divisor"),
}];

const PARAM_DAILY_FACTOR: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "threshold_level",
    label: "Threshold Level",
    kind: IndicatorParamKind::Float,
    required: false,
    default: Some(ParamValueStatic::Float(0.35)),
    min: Some(0.0),
    max: Some(1.0),
    step: Some(0.01),
    enum_values: EMPTY_ENUM_VALUES,
    notes: Some(
        "Daily Factor threshold used for the strong signal state; EMA length is fixed at 14",
    ),
}];

const PARAM_MOVING_AVERAGE_CROSS_PROBABILITY: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "ma_type",
        label: "MA Type",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("ema")),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_MOVING_AVERAGE_CROSS_PROBABILITY_MA_TYPE,
        notes: Some("Moving-average mode used for the fast and slow baseline pair"),
    },
    IndicatorParamInfo {
        key: "smoothing_window",
        label: "Smoothing Window",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(7)),
        min: Some(2.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some(
            "Hull moving-average and standard-deviation lookback used for the forecast envelope",
        ),
    },
    IndicatorParamInfo {
        key: "slow_length",
        label: "Slow Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(30)),
        min: Some(2.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Slow moving-average length; must remain greater than fast_length"),
    },
    IndicatorParamInfo {
        key: "fast_length",
        label: "Fast Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Fast moving-average length used by the crossover simulation"),
    },
    IndicatorParamInfo {
        key: "resolution",
        label: "Resolution",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(50)),
        min: Some(2.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Number of simulated future prices sampled across the forecast envelope"),
    },
];

const PARAM_BULLS_V_BEARS: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "period",
        label: "Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Moving average period used for the Bulls and Bears pressure baseline"),
    },
    IndicatorParamInfo {
        key: "ma_type",
        label: "MA Type",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("ema")),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_BULLS_V_BEARS_MA_TYPE,
        notes: Some("Moving average type used for the baseline"),
    },
    IndicatorParamInfo {
        key: "calculation_method",
        label: "Calculation Method",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("normalized")),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_BULLS_V_BEARS_CALCULATION_METHOD,
        notes: Some(
            "Normalized mode reproduces the v4-style scaling; raw mode uses rolling raw thresholds",
        ),
    },
    IndicatorParamInfo {
        key: "normalized_bars_back",
        label: "Normalized Bars Back",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(120)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Lookback window used for bull and bear normalization"),
    },
    IndicatorParamInfo {
        key: "raw_rolling_period",
        label: "Raw Rolling Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(50)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Rolling window used to derive raw mode dynamic thresholds"),
    },
    IndicatorParamInfo {
        key: "raw_threshold_percentile",
        label: "Raw Threshold Percentile",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(95.0)),
        min: Some(80.0),
        max: Some(99.0),
        step: Some(0.5),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Percentile used to map the raw rolling range into upper and lower thresholds"),
    },
    IndicatorParamInfo {
        key: "threshold_level",
        label: "Threshold Level",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(80.0)),
        min: Some(0.0),
        max: Some(100.0),
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Fixed normalized-mode threshold level"),
    },
];

const PARAM_REGRESSION_SLOPE_OSCILLATOR: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "min_range",
        label: "Min Range",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10)),
        min: Some(2.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Shortest regression lookback sampled by the oscillator"),
    },
    IndicatorParamInfo {
        key: "max_range",
        label: "Max Range",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(100)),
        min: Some(2.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Longest regression lookback sampled by the oscillator"),
    },
    IndicatorParamInfo {
        key: "step",
        label: "Step",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(5)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Increment between sampled regression lookbacks"),
    },
    IndicatorParamInfo {
        key: "signal_line",
        label: "Signal Line",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(7)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("SMA length applied to the oscillator"),
    },
];

const PARAM_SMOOTH_THEIL_SEN: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "source",
        label: "Source",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("close")),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_PRICE_SOURCE,
        notes: Some("Source series used by the Theil-Sen regression"),
    },
    IndicatorParamInfo {
        key: "length",
        label: "Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(25)),
        min: Some(2.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Regression window length"),
    },
    IndicatorParamInfo {
        key: "offset",
        label: "Offset",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(0)),
        min: Some(0.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Lookback offset used for the regression anchor"),
    },
    IndicatorParamInfo {
        key: "multiplier",
        label: "Multiplier",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(2.0)),
        min: Some(0.0),
        max: None,
        step: Some(0.125),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Deviation-band scale factor"),
    },
    IndicatorParamInfo {
        key: "slope_style",
        label: "Slope Style",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("smooth_median")),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_SMOOTH_THEIL_SEN_STAT_STYLE,
        notes: Some("Estimator used for pairwise slope aggregation"),
    },
    IndicatorParamInfo {
        key: "residual_style",
        label: "Residual Style",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("smooth_median")),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_SMOOTH_THEIL_SEN_STAT_STYLE,
        notes: Some("Estimator used for intercept aggregation"),
    },
    IndicatorParamInfo {
        key: "deviation_style",
        label: "Deviation Style",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("mad")),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_SMOOTH_THEIL_SEN_DEVIATION_STYLE,
        notes: Some("Deviation method used for the bands"),
    },
    IndicatorParamInfo {
        key: "mad_style",
        label: "MAD Style",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("smooth_median")),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_SMOOTH_THEIL_SEN_STAT_STYLE,
        notes: Some("Estimator used when deviation_style is mad"),
    },
    IndicatorParamInfo {
        key: "include_prediction_in_deviation",
        label: "Include Prediction In Deviation",
        kind: IndicatorParamKind::Bool,
        required: false,
        default: Some(ParamValueStatic::Bool(false)),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_TRUE_FALSE,
        notes: Some("Include the extrapolated offset region in deviation estimation"),
    },
];

const PARAM_L2_EHLERS_SIGNAL_TO_NOISE: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "source",
        label: "Source",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("hl2")),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_PRICE_SOURCE,
        notes: Some("Source used for the Hilbert-transform signal component"),
    },
    IndicatorParamInfo {
        key: "smooth_period",
        label: "Smooth Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Hilbert-transform scaling period used by the Ehlers SNR calculation"),
    },
];

const PARAM_EHLERS_SMOOTHED_ADAPTIVE_MOMENTUM: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "source",
        label: "Source",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("hl2")),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_PRICE_SOURCE,
        notes: Some("Price source used by the Pine-faithful ESAM core"),
    },
    IndicatorParamInfo {
        key: "alpha",
        label: "Alpha",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(0.07)),
        min: Some(0.0),
        max: None,
        step: Some(0.01),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Pine-faithful high-pass alpha term"),
    },
    IndicatorParamInfo {
        key: "cutoff",
        label: "Cutoff",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(8.0)),
        min: Some(0.000_000_001),
        max: None,
        step: Some(0.1),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Pine-faithful smoothing cutoff used by the final 3-pole filter"),
    },
];

const PARAM_EHLERS_ADAPTIVE_CYBER_CYCLE: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "source",
        label: "Source",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("hl2")),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_PRICE_SOURCE,
        notes: Some("Price source used by the Adaptive Cyber Cycle core"),
    },
    IndicatorParamInfo {
        key: "alpha",
        label: "Alpha",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(0.07)),
        min: Some(0.0),
        max: Some(1.0),
        step: Some(0.01),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Fixed alpha term used by the recursive cycle-estimation stage"),
    },
];

const PARAM_EHLERS_SIMPLE_CYCLE_INDICATOR: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "source",
        label: "Source",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("hl2")),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_PRICE_SOURCE,
        notes: Some("Price source used by the Pine-faithful simple cycle core"),
    },
    IndicatorParamInfo {
        key: "alpha",
        label: "Alpha",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(0.07)),
        min: Some(0.0),
        max: Some(1.0),
        step: Some(0.01),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Pine-faithful recursive alpha term used by the cycle filter"),
    },
];

const PARAM_L1_EHLERS_PHASOR: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "domestic_cycle_length",
    label: "Domestic Cycle Length",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(15)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: Some("Sliding phasor window length used by the Pine-faithful blackcat phase-angle core"),
}];

const PARAM_ANDEAN_OSCILLATOR: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "length",
        label: "Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(50)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Envelope EMA length used by the Pine-faithful Andean core"),
    },
    IndicatorParamInfo {
        key: "signal_length",
        label: "Signal Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(9)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("EMA length applied to max(bull, bear) for the Pine-faithful signal line"),
    },
];

const PARAM_RANDOM_WALK_INDEX: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "length",
    label: "Length",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(14)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: Some("Pine-faithful single ATR/history length used by the everget script"),
}];

const PARAM_PRICE_MOVING_AVERAGE_RATIO_PERCENTILE: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "source",
        label: "Source",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("close")),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_PRICE_SOURCE,
        notes: Some("Price source used for PMAR and PMARP"),
    },
    IndicatorParamInfo {
        key: "ma_length",
        label: "PMAR Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(20)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Moving average length used for PMAR"),
    },
    IndicatorParamInfo {
        key: "ma_type",
        label: "PMAR MA Type",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("vwma")),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_PMARP_MA_TYPE,
        notes: Some("Moving average type used for PMAR"),
    },
    IndicatorParamInfo {
        key: "pmarp_lookback",
        label: "PMARP Lookback",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(350)),
        min: Some(1.0),
        max: Some(1900.0),
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Lookback used by the PMARP percentile count"),
    },
    IndicatorParamInfo {
        key: "signal_ma_length",
        label: "Signal MA Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(20)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Signal moving average length"),
    },
    IndicatorParamInfo {
        key: "signal_ma_type",
        label: "Signal MA Type",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("sma")),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_PMARP_MA_TYPE,
        notes: Some("Moving average type used for the signal line"),
    },
    IndicatorParamInfo {
        key: "line_mode",
        label: "Line Mode",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("pmarp")),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_PMARP_LINE_MODE,
        notes: Some("Main plotted line: raw PMAR or PMARP percentile"),
    },
];

const PARAM_RELATIVE_STRENGTH_INDEX_WAVE_INDICATOR: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "source",
        label: "Source",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("hlcc4")),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_PRICE_SOURCE,
        notes: Some("Price source used for the custom RSI leg"),
    },
    IndicatorParamInfo {
        key: "rsi_length",
        label: "RSI Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("RSI length used for source, high, and low RSI series"),
    },
    IndicatorParamInfo {
        key: "length1",
        label: "Length 1",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(2)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("WMA length used for the first RSI wave line"),
    },
    IndicatorParamInfo {
        key: "length2",
        label: "Length 2",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(5)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("WMA length used for the second RSI wave line"),
    },
    IndicatorParamInfo {
        key: "length3",
        label: "Length 3",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(9)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("WMA length used for the third RSI wave line"),
    },
    IndicatorParamInfo {
        key: "length4",
        label: "Length 4",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(13)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("WMA length used for the fourth RSI wave line"),
    },
];

const PARAM_MESA_STOCHASTIC_MULTI_LENGTH: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "source",
        label: "Source",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("close")),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_PRICE_SOURCE,
        notes: Some("Price source used by the shared MESA filter stage"),
    },
    IndicatorParamInfo {
        key: "length_1",
        label: "Length 1",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(48)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("First Pine-faithful MESA stochastic lookback"),
    },
    IndicatorParamInfo {
        key: "length_2",
        label: "Length 2",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(21)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Second Pine-faithful MESA stochastic lookback"),
    },
    IndicatorParamInfo {
        key: "length_3",
        label: "Length 3",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(9)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Third Pine-faithful MESA stochastic lookback"),
    },
    IndicatorParamInfo {
        key: "length_4",
        label: "Length 4",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(6)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Fourth Pine-faithful MESA stochastic lookback"),
    },
    IndicatorParamInfo {
        key: "trigger_length",
        label: "Trigger Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(2)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("SMA length used for all four trigger lines"),
    },
];

const PARAM_SPEARMAN_CORRELATION: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "source",
        label: "Source",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("close")),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_PRICE_SOURCE,
        notes: Some("Primary candle source used for the return series"),
    },
    IndicatorParamInfo {
        key: "comparison_source",
        label: "Comparison Source",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("open")),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_PRICE_SOURCE,
        notes: Some("Secondary candle source used for the comparison return series"),
    },
    IndicatorParamInfo {
        key: "lookback",
        label: "Lookback",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(30)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Return-window length used for the Spearman rank correlation"),
    },
    IndicatorParamInfo {
        key: "smoothing_length",
        label: "Smoothing Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(3)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Simple moving average length applied to the raw Spearman series"),
    },
];

const PARAM_TREND_TRIGGER_FACTOR: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "length",
    label: "Length",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(15)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: Some("Pine-faithful lookback length; the source input in the script is unused"),
}];

const PARAM_CYCLE_CHANNEL_OSCILLATOR: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "source",
        label: "Source",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("close")),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_PRICE_SOURCE,
        notes: Some("Candle source used for the short and medium RMA center lines"),
    },
    IndicatorParamInfo {
        key: "short_cycle_length",
        label: "Short Cycle Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10)),
        min: Some(2.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Pine-faithful raw short cycle length; the internal RMA period is floor(length / 2)"),
    },
    IndicatorParamInfo {
        key: "medium_cycle_length",
        label: "Medium Cycle Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(30)),
        min: Some(2.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Pine-faithful raw medium cycle length; the internal ATR and RMA period is floor(length / 2)"),
    },
    IndicatorParamInfo {
        key: "short_multiplier",
        label: "Short Multiplier",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(1.0)),
        min: Some(0.0),
        max: None,
        step: Some(0.1),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Preserved for Pine parity; in the published oscillator script this multiplier cancels out of the fast/slow outputs"),
    },
    IndicatorParamInfo {
        key: "medium_multiplier",
        label: "Medium Multiplier",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(3.0)),
        min: Some(0.0),
        max: None,
        step: Some(0.1),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("ATR channel multiplier used by the medium cycle oscillator denominator"),
    },
];

const PARAM_VOLATILITY_QUALITY_INDEX: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "fast_length",
        label: "Fast Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(9)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Pine-faithful fast SMA length over cumulative VQI"),
    },
    IndicatorParamInfo {
        key: "slow_length",
        label: "Slow Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(200)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Pine-faithful slow SMA length over cumulative VQI"),
    },
];

const PARAM_VOLUME_ZONE_OSCILLATOR: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "length",
        label: "Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(2.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Pine-faithful EMA length for signed and total volume"),
    },
    IndicatorParamInfo {
        key: "intraday_smoothing",
        label: "Intraday Smoothing",
        kind: IndicatorParamKind::Bool,
        required: false,
        default: Some(ParamValueStatic::Bool(true)),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_TRUE_FALSE,
        notes: Some("Applies the optional Pine intraday EMA smoothing pass"),
    },
    IndicatorParamInfo {
        key: "noise_filter",
        label: "Noise Filter",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(4)),
        min: Some(2.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("EMA length used by the optional intraday smoothing pass"),
    },
];

const PARAM_VWAP_DEVIATION_OSCILLATOR: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "session_mode",
        label: "Session Mode",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("rolling_bars")),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_VDO_SESSION_MODE,
        notes: None,
    },
    IndicatorParamInfo {
        key: "rolling_period",
        label: "Rolling Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(20)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "rolling_days",
        label: "Rolling Days",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(30)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "use_close",
        label: "Use Close",
        kind: IndicatorParamKind::Bool,
        required: false,
        default: Some(ParamValueStatic::Bool(false)),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_TRUE_FALSE,
        notes: None,
    },
    IndicatorParamInfo {
        key: "deviation_mode",
        label: "Deviation Mode",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("absolute")),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_VDO_DEVIATION_MODE,
        notes: None,
    },
    IndicatorParamInfo {
        key: "z_window",
        label: "Z Window",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(50)),
        min: Some(5.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "pct_vol_lookback",
        label: "Percent Volatility Lookback",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(100)),
        min: Some(10.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "pct_min_sigma",
        label: "Percent Minimum Sigma",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(0.1)),
        min: Some(0.01),
        max: None,
        step: Some(0.01),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "abs_vol_lookback",
        label: "Absolute Volatility Lookback",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(100)),
        min: Some(10.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_ICHIMOKU_OSCILLATOR: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "source",
        label: "Source",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("close")),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Signal source used for the oscillator signal and chikou inputs"),
    },
    IndicatorParamInfo {
        key: "conversion_periods",
        label: "Conversion Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(9)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "base_periods",
        label: "Base Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(26)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "lagging_span_periods",
        label: "Leading Span B Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(52)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "displacement",
        label: "Displacement",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(26)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "ma_length",
        label: "MA Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(12)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "smoothing_length",
        label: "Smoothing Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(3)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "extra_smoothing",
        label: "Extra Smoothing",
        kind: IndicatorParamKind::Bool,
        required: false,
        default: Some(ParamValueStatic::Bool(true)),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_TRUE_FALSE,
        notes: None,
    },
    IndicatorParamInfo {
        key: "normalize",
        label: "Normalize",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("window")),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_ICHI_NORMALIZE_MODE,
        notes: None,
    },
    IndicatorParamInfo {
        key: "window_size",
        label: "Window Size",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(20)),
        min: Some(5.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "clamp",
        label: "Clamp",
        kind: IndicatorParamKind::Bool,
        required: false,
        default: Some(ParamValueStatic::Bool(true)),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_TRUE_FALSE,
        notes: None,
    },
    IndicatorParamInfo {
        key: "top_band",
        label: "Top Band",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(2.0)),
        min: Some(0.0),
        max: None,
        step: Some(0.25),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "mid_band",
        label: "Mid Band",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(1.5)),
        min: Some(0.0),
        max: None,
        step: Some(0.25),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_FOSC_PERIOD: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(5)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_IFT_RSI: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "rsi_period",
        label: "RSI Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(5)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "wma_period",
        label: "WMA Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(9)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_DEC_OSC: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "hp_period",
        label: "HP Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(125)),
        min: Some(3.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "k",
        label: "K",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(1.0)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Must be positive"),
    },
];

const PARAM_DECYCLER: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "hp_period",
        label: "HP Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(125)),
        min: Some(2.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "k",
        label: "K",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(0.707)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Must be positive"),
    },
];

const PARAM_VIDYA: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "short_period",
        label: "Short Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(2)),
        min: Some(2.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "long_period",
        label: "Long Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(5)),
        min: Some(2.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "alpha",
        label: "Alpha",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(0.2)),
        min: Some(0.0),
        max: Some(1.0),
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_VLMA: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "min_period",
        label: "Min Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(5)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "max_period",
        label: "Max Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(50)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "matype",
        label: "MA Type",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("sma")),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "devtype",
        label: "Deviation Type",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(0)),
        min: Some(0.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_LINEARREG_ANGLE_PERIOD: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(14)),
    min: Some(2.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_LINEARREG_INTERCEPT_PERIOD: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(14)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_LINEARREG_SLOPE_PERIOD: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(14)),
    min: Some(2.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_CG_PERIOD: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(10)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_MACD: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "fast_period",
        label: "Fast Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(12)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "slow_period",
        label: "Slow Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(26)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "signal_period",
        label: "Signal Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(9)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_ADAPTIVE_MACD: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "length",
        label: "R2 Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(20)),
        min: Some(2.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "fast_period",
        label: "Fast Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10)),
        min: Some(2.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "slow_period",
        label: "Slow Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(20)),
        min: Some(2.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "signal_period",
        label: "Signal Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(9)),
        min: Some(2.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_BOLLINGER: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "period",
        label: "Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(20)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "devup",
        label: "Dev Up",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(2.0)),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "devdn",
        label: "Dev Down",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(2.0)),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_STOCH: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "fastk_period",
        label: "Fast K Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "slowk_period",
        label: "Slow K Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(3)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "slowd_period",
        label: "Slow D Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(3)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_STOCHF: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "fastk_period",
        label: "Fast K Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(5)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "fastd_period",
        label: "Fast D Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(3)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_VW_MACD: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "fast",
        label: "Fast",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(12)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "slow",
        label: "Slow",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(26)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "signal",
        label: "Signal",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(9)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_VPCI: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "short_range",
        label: "Short Range",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(5)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "long_range",
        label: "Long Range",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(20)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_TTM_TREND: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(6)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_TTM_SQUEEZE: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "length",
        label: "Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(20)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "bb_mult",
        label: "BB Mult",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(2.0)),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "kc_high",
        label: "KC High",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(1.0)),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "kc_mid",
        label: "KC Mid",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(1.5)),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "kc_low",
        label: "KC Low",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(2.0)),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_DI: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(14)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_DTI: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "r",
        label: "R",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "s",
        label: "S",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "u",
        label: "U",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(5)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_DM: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(14)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_DONCHIAN: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(20)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_SUPERTREND: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "period",
        label: "Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "factor",
        label: "Factor",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(3.0)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_ADJUSTABLE_MA_ALTERNATING_EXTREMITIES: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "length",
        label: "Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(50)),
        min: Some(2.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "mult",
        label: "Multiplier",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(2.0)),
        min: Some(1.0),
        max: None,
        step: Some(0.1),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "alpha",
        label: "Lag",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(1.0)),
        min: Some(0.0),
        max: None,
        step: Some(0.1),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "beta",
        label: "Overshoot",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(0.5)),
        min: Some(0.0),
        max: None,
        step: Some(0.1),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_SUPERTREND_RECOVERY: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "atr_length",
        label: "ATR Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "multiplier",
        label: "Base Multiplier",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(3.0)),
        min: Some(0.1),
        max: None,
        step: Some(0.1),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "alpha_percent",
        label: "Recovery Alpha (%)",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(5.0)),
        min: Some(0.1),
        max: Some(100.0),
        step: Some(0.1),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Percentage weight applied to price when the active trend is at a loss relative to the latest switch price"),
    },
    IndicatorParamInfo {
        key: "threshold_atr",
        label: "Recovery Threshold (xATR)",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(1.0)),
        min: Some(0.0),
        max: None,
        step: Some(0.1),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Recovery logic activates only when the loss exceeds this many ATRs from the switch price"),
    },
];

const PARAM_STANDARDIZED_PSAR_OSCILLATOR: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "start",
        label: "Start",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(0.02)),
        min: Some(0.0),
        max: None,
        step: Some(0.0001),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "increment",
        label: "Increment",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(0.0005)),
        min: Some(0.0),
        max: None,
        step: Some(0.0001),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "maximum",
        label: "Max Value",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(0.2)),
        min: Some(0.0),
        max: None,
        step: Some(0.0001),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "standardization_length",
        label: "Standardization Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(21)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "wma_length",
        label: "WMA Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(40)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "wma_lag",
        label: "WMA Lag Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(3)),
        min: Some(0.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "pivot_left",
        label: "Divergence Pivot Detection Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(15)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "pivot_right",
        label: "Divergence Pivot Confirmation Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(1)),
        min: Some(0.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "plot_bullish",
        label: "Plot Bullish Divergences",
        kind: IndicatorParamKind::Bool,
        required: false,
        default: Some(ParamValueStatic::Bool(true)),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_TRUE_FALSE,
        notes: None,
    },
    IndicatorParamInfo {
        key: "plot_bearish",
        label: "Plot Bearish Divergences",
        kind: IndicatorParamKind::Bool,
        required: false,
        default: Some(ParamValueStatic::Bool(true)),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_TRUE_FALSE,
        notes: None,
    },
];

const PARAM_KELTNER: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "period",
        label: "Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(20)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "multiplier",
        label: "Multiplier",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(2.0)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "ma_type",
        label: "MA Type",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("ema")),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_AROON: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "length",
    label: "Length",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(14)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_SRSI: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "rsi_period",
        label: "RSI Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "stoch_period",
        label: "Stoch Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "k",
        label: "K",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(3)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "d",
        label: "D",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(3)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "source",
        label: "Source",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("close")),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_STOCHASTIC_CONNORS_RSI: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "stoch_length",
        label: "Stochastic Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(3)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "smooth_k",
        label: "K",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(3)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "smooth_d",
        label: "D",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(3)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "rsi_length",
        label: "RSI Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(3)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "updown_length",
        label: "Updown RSI Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(2)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "roc_length",
        label: "ROC Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(100)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_CHOP: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "period",
        label: "Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "scalar",
        label: "Scalar",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(100.0)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "drift",
        label: "Drift",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(1)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_SQUEEZE_MOMENTUM: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "length_bb",
        label: "BB Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(20)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "mult_bb",
        label: "BB Mult",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(2.0)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "length_kc",
        label: "KC Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(20)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "mult_kc",
        label: "KC Mult",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(1.5)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_WTO: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "channel_length",
        label: "Channel Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "average_length",
        label: "Average Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(21)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_WAVETREND: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "channel_length",
        label: "Channel Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(9)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "average_length",
        label: "Average Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(12)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "ma_length",
        label: "MA Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(3)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "factor",
        label: "Factor",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(0.015)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_YANG_ZHANG: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "lookback",
        label: "Lookback",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "k_override",
        label: "K Override",
        kind: IndicatorParamKind::Bool,
        required: false,
        default: Some(ParamValueStatic::Bool(false)),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_TRUE_FALSE,
        notes: None,
    },
    IndicatorParamInfo {
        key: "k",
        label: "K",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(0.34)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_PARKINSON: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(8)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];
const PARAM_VOLATILITY_RATIO_ADAPTIVE_RSX: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "period",
        label: "Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Volatility ratio lookback and adaptive RSX base period."),
    },
    IndicatorParamInfo {
        key: "speed",
        label: "Speed",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(0.5)),
        min: Some(0.0),
        max: Some(1.0),
        step: Some(0.1),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Pine `spd` parameter."),
    },
];
const PARAM_ON_BALANCE_VOLUME_OSCILLATOR: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "obv_length",
        label: "OBV Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(20)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Rolling sum window for signed volume and total volume."),
    },
    IndicatorParamInfo {
        key: "ema_length",
        label: "EMA Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(9)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("EMA smoothing length applied to the oscillator line."),
    },
];
const PARAM_HISTORICAL_VOLATILITY_PERCENTILE: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "length",
        label: "Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(21)),
        min: Some(2.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Sample HV lookback and HVP SMA length."),
    },
    IndicatorParamInfo {
        key: "annual_length",
        label: "Annual Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(252)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Used for annualization and HVP percentile window."),
    },
];
const PARAM_KELTNER_CHANNEL_WIDTH_OSCILLATOR: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "length",
        label: "Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(20)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Lookback for the centerline and width SMA."),
    },
    IndicatorParamInfo {
        key: "multiplier",
        label: "Multiplier",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(2.0)),
        min: Some(0.0),
        max: None,
        step: Some(0.1),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Band multiplier applied to the chosen Keltner range."),
    },
    IndicatorParamInfo {
        key: "use_exponential",
        label: "Use Exponential",
        kind: IndicatorParamKind::Bool,
        required: false,
        default: Some(ParamValueStatic::Bool(true)),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("When false, the centerline uses SMA instead of EMA."),
    },
    IndicatorParamInfo {
        key: "bands_style",
        label: "Bands Style",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("Average True Range")),
        min: None,
        max: None,
        step: None,
        enum_values: &["Average True Range", "True Range", "Range"],
        notes: Some("Matches the Pine band-style switch."),
    },
    IndicatorParamInfo {
        key: "atr_length",
        label: "ATR Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Used when `bands_style` is `Average True Range`."),
    },
];
const PARAM_MARKET_MEANNESS_INDEX: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "length",
        label: "Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(300)),
        min: Some(6.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Rolling period used for both the raw MMI and the smoothing SMA."),
    },
    IndicatorParamInfo {
        key: "source_mode",
        label: "Source Mode",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("Price")),
        min: None,
        max: None,
        step: None,
        enum_values: &["Price", "Change"],
        notes: Some("Matches the Pine source selector: close price or close-open change."),
    },
];
const PARAM_PRICE_DENSITY_MARKET_NOISE: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "length",
        label: "Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Rolling price density lookback."),
    },
    IndicatorParamInfo {
        key: "eval_period",
        label: "Eval Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(200)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Percentile-rank lookback for relative price density."),
    },
];
const PARAM_MOMENTUM_RATIO_OSCILLATOR: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(50)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: Some("Pine `inpPeriod` used by the recursive EMA-ratio oscillator core."),
}];
const PARAM_SUPERTREND_OSCILLATOR: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "length",
        label: "Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("ATR and SuperTrend lookback length."),
    },
    IndicatorParamInfo {
        key: "mult",
        label: "Multiplier",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(2.0)),
        min: Some(0.0000001),
        max: None,
        step: Some(0.1),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("ATR multiplier applied to the SuperTrend bands."),
    },
    IndicatorParamInfo {
        key: "smooth",
        label: "Smooth",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(72)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("EMA length applied to oscillator minus adaptive signal."),
    },
];
const PARAM_HYPERTREND: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "factor",
        label: "Multiplicative Factor",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(5.0)),
        min: Some(0.0000001),
        max: None,
        step: Some(0.1),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("ATR(200) multiplier used by the HyperTrend recursion."),
    },
    IndicatorParamInfo {
        key: "slope",
        label: "Slope",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(14.0)),
        min: Some(0.0000001),
        max: None,
        step: Some(0.1),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Controls the recursive drift step when price remains within the ATR gate."),
    },
    IndicatorParamInfo {
        key: "width_percent",
        label: "Width %",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(80.0)),
        min: Some(0.0),
        max: Some(100.0),
        step: Some(0.1),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Band width percentage applied to the held ATR amplitude."),
    },
];
const PARAM_LOGARITHMIC_MOVING_AVERAGE: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "period",
        label: "Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(100)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some(
            "Window length used by the logarithmic weighting and trend-quality calculations.",
        ),
    },
    IndicatorParamInfo {
        key: "steepness",
        label: "Steepness",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(2.5)),
        min: Some(0.0000001),
        max: None,
        step: Some(0.1),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Controls how aggressively logarithmic weights decay into the past."),
    },
    IndicatorParamInfo {
        key: "ma_type",
        label: "Smoothing MA Type",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("ema")),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_LMA_SMOOTHING,
        notes: Some("Final signal smoother: ema, sma, wma, rma, or vwma."),
    },
    IndicatorParamInfo {
        key: "smooth",
        label: "Signal Smoothing",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Lookback for the final smoothing pass applied to the LMA signal."),
    },
    IndicatorParamInfo {
        key: "momentum_weight",
        label: "Momentum Weight",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(1.2)),
        min: Some(0.0000001),
        max: None,
        step: Some(0.1),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some(
            "Multiplier applied when signal direction agrees with annualized log momentum.",
        ),
    },
    IndicatorParamInfo {
        key: "long_threshold",
        label: "Long Threshold",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(0.5)),
        min: None,
        max: None,
        step: Some(0.1),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Upper threshold used for bullish position classification."),
    },
    IndicatorParamInfo {
        key: "short_threshold",
        label: "Short Threshold",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(-0.5)),
        min: None,
        max: None,
        step: Some(0.1),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Lower threshold used for bearish position classification."),
    },
];
const PARAM_ICT_PROPULSION_BLOCK: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "swing_length",
        label: "Swing Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(3)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Right-side swing confirmation length used by the order-block state machine."),
    },
    IndicatorParamInfo {
        key: "mitigation_price",
        label: "Mitigation Price",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("close")),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_CLOSE_WICK,
        notes: Some("Use closing-price or wick breaches when marking blocks as mitigated."),
    },
];
const PARAM_EHLERS_DETRENDING_FILTER: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "length",
    label: "Length",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(10)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: Some("Cosine-window detrending length applied to the recursive EDF prefilter."),
}];
const PARAM_LEAVITT_CONVOLUTION_ACCELERATION: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "length",
        label: "Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(70)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Projection length used by the Leavitt linear-regression forecast."),
    },
    IndicatorParamInfo {
        key: "norm_length",
        label: "Normalization Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(150)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Lookback used to z-normalize the convolution slope."),
    },
    IndicatorParamInfo {
        key: "use_norm_hyperbolic",
        label: "Use Hyperbolic Normalization",
        kind: IndicatorParamKind::Bool,
        required: false,
        default: Some(ParamValueStatic::Bool(true)),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_TRUE_FALSE,
        notes: Some("Switch between hyperbolic and logistic soft normalization."),
    },
];
const PARAM_INSYNC_INDEX: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "emo_divisor",
        label: "EMO Divisor",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10000)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Scaling divisor used by the EMO / EMV-style subcomponent."),
    },
    IndicatorParamInfo {
        key: "emo_length",
        label: "EMO Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("SMA length applied to the raw EMO series."),
    },
    IndicatorParamInfo {
        key: "fast_length",
        label: "Fast Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(12)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Fast EMA length used by the MACD subcomponent."),
    },
    IndicatorParamInfo {
        key: "slow_length",
        label: "Slow Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(26)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Slow EMA length used by the MACD subcomponent."),
    },
    IndicatorParamInfo {
        key: "mfi_length",
        label: "MFI Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(20)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Money Flow Index lookback."),
    },
    IndicatorParamInfo {
        key: "bb_length",
        label: "BB Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(20)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Bollinger Bands lookback."),
    },
    IndicatorParamInfo {
        key: "bb_multiplier",
        label: "BB Multiplier",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(2.0)),
        min: Some(0.000001),
        max: None,
        step: Some(0.1),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Bollinger Bands standard-deviation multiplier."),
    },
    IndicatorParamInfo {
        key: "cci_length",
        label: "CCI Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("CCI lookback applied to close."),
    },
    IndicatorParamInfo {
        key: "dpo_length",
        label: "DPO Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(18)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Detrended Price Oscillator lookback."),
    },
    IndicatorParamInfo {
        key: "roc_length",
        label: "ROC Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Rate-of-change lookback."),
    },
    IndicatorParamInfo {
        key: "rsi_length",
        label: "RSI Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("RSI lookback."),
    },
    IndicatorParamInfo {
        key: "stoch_length",
        label: "Stoch Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Stochastic fast-window lookback."),
    },
    IndicatorParamInfo {
        key: "stoch_d_length",
        label: "Stoch D Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(3)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Stochastic D smoothing length."),
    },
    IndicatorParamInfo {
        key: "stoch_k_length",
        label: "Stoch K Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(1)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Stochastic K smoothing length."),
    },
    IndicatorParamInfo {
        key: "sma_length",
        label: "SMA Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Shared smoothing length for the EMO, MACD, DPO, and ROC subcomponents."),
    },
];
const PARAM_IMPULSE_MACD: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "length_ma",
        label: "MA Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(34)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Lookback used by the SMMA(high/low) envelopes and the double-EMA midline."),
    },
    IndicatorParamInfo {
        key: "length_signal",
        label: "Signal Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(9)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("SMA smoothing period applied to the impulse line."),
    },
];
const PARAM_TWIGGS_MONEY_FLOW: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "length",
        label: "Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(21)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Base moving-average lookback for the Twiggs Money Flow ratio."),
    },
    IndicatorParamInfo {
        key: "smoothing_length",
        label: "Smoothing Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(0.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("HMA smoothing period for the secondary output."),
    },
    IndicatorParamInfo {
        key: "ma_type",
        label: "MA Type",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("WMA")),
        min: None,
        max: None,
        step: None,
        enum_values: &["SMA", "EMA", "WMA", "VWMA"],
        notes: Some("Moving-average type for both the numerator and denominator."),
    },
];
const PARAM_VOLUME_WEIGHTED_STOCHASTIC_RSI: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "rsi_length",
        label: "RSI Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Lookback for the volume-weighted RSI core."),
    },
    IndicatorParamInfo {
        key: "stoch_length",
        label: "Stochastic Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Lookback for stochastic normalization of the weighted RSI."),
    },
    IndicatorParamInfo {
        key: "k_length",
        label: "K Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(3)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Smoothing length for the K line."),
    },
    IndicatorParamInfo {
        key: "d_length",
        label: "D Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(3)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Smoothing length for the D line."),
    },
    IndicatorParamInfo {
        key: "ma_type",
        label: "MA Type",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("WSMA")),
        min: None,
        max: None,
        step: None,
        enum_values: &[
            "WSMA", "SMMA", "RMA", "WILDERS", "SMA", "EMA", "WMA", "VWMA",
        ],
        notes: Some("Weighted smoothing type applied to K and D."),
    },
];
const PARAM_STOCHASTIC_ADAPTIVE_D: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "k_length",
        label: "Stochastic Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(20)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Lookback used by the stochastic window over pre-smoothed prices."),
    },
    IndicatorParamInfo {
        key: "d_smoothing",
        label: "%K Smoothing",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(9)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("SMA length applied to raw stochastic values before attenuation."),
    },
    IndicatorParamInfo {
        key: "pre_smooth",
        label: "Price Pre-Smoothing",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(20)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("SMA length applied to high, low, and close before stochastic calculation."),
    },
    IndicatorParamInfo {
        key: "attenuation",
        label: "Attenuation Factor",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(2.0)),
        min: Some(0.1),
        max: None,
        step: Some(0.1),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Controls the adaptive smoothing speed and input attenuation."),
    },
];
const PARAM_ADAPTIVE_SCHAFF_TREND_CYCLE: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "adaptive_length",
        label: "Adaptive Logic Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(55)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Lookback for the adaptive correlation term in the recursive MACD core."),
    },
    IndicatorParamInfo {
        key: "stc_length",
        label: "STC Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(12)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Lookback used by both stochastic normalization passes."),
    },
    IndicatorParamInfo {
        key: "smoothing_factor",
        label: "STC Sensitivity",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(0.45)),
        min: Some(0.0001),
        max: Some(1.0),
        step: Some(0.0001),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Recursive smoothing coefficient applied to both stochastic stages."),
    },
    IndicatorParamInfo {
        key: "fast_length",
        label: "MACD Fast Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(26)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Fast alpha used by the adaptive recursive MACD filter."),
    },
    IndicatorParamInfo {
        key: "slow_length",
        label: "MACD Slow Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(50)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some(
            "Slow alpha used by the adaptive recursive MACD filter and histogram normalization.",
        ),
    },
];
const PARAM_SMOOTHED_GAUSSIAN_TREND_FILTER: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "gaussian_length",
        label: "Gaussian Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(15)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Period used to derive the Gaussian alpha coefficient."),
    },
    IndicatorParamInfo {
        key: "poles",
        label: "Poles",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(3)),
        min: Some(1.0),
        max: Some(4.0),
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Gaussian filter order."),
    },
    IndicatorParamInfo {
        key: "smoothing_length",
        label: "Smoothing Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(22)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Linear-regression smoothing length applied to the Gaussian output."),
    },
    IndicatorParamInfo {
        key: "linreg_offset",
        label: "Flatten Multiplier",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(7)),
        min: Some(0.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Linear-regression offset used to flatten the final filter output."),
    },
];

const PARAM_VI: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(14)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_KDJ: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "fast_k_period",
        label: "Fast K Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(9)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "slow_k_period",
        label: "Slow K Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(3)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "slow_k_ma_type",
        label: "Slow K MA Type",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("sma")),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "slow_d_period",
        label: "Slow D Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(3)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "slow_d_ma_type",
        label: "Slow D MA Type",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("sma")),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_ACOSC: &[IndicatorParamInfo] = PARAM_NONE;

const PARAM_ALLIGATOR: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "jaw_period",
        label: "Jaw Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(13)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "jaw_offset",
        label: "Jaw Offset",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(8)),
        min: Some(0.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "teeth_period",
        label: "Teeth Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(8)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "teeth_offset",
        label: "Teeth Offset",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(5)),
        min: Some(0.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "lips_period",
        label: "Lips Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(5)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "lips_offset",
        label: "Lips Offset",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(3)),
        min: Some(0.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_ALPHATREND: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "coeff",
        label: "Coeff",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(1.0)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "period",
        label: "Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "no_volume",
        label: "No Volume",
        kind: IndicatorParamKind::Bool,
        required: false,
        default: Some(ParamValueStatic::Bool(false)),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_TRUE_FALSE,
        notes: None,
    },
];

const PARAM_ASO: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "period",
        label: "Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "mode",
        label: "Mode",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(0)),
        min: Some(0.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_AVSL: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "fast_period",
        label: "Fast Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(12)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "slow_period",
        label: "Slow Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(26)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "multiplier",
        label: "Multiplier",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(2.0)),
        min: Some(0.0),
        max: None,
        step: Some(0.1),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_BANDPASS: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "period",
        label: "Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(20)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "bandwidth",
        label: "Bandwidth",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(0.3)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_CHANDE: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "period",
        label: "Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(22)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "mult",
        label: "Multiplier",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(3.0)),
        min: Some(0.0),
        max: None,
        step: Some(0.1),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "direction",
        label: "Direction",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("long")),
        min: None,
        max: None,
        step: None,
        enum_values: &["long", "short"],
        notes: None,
    },
];

const PARAM_CHANDELIER_EXIT: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "period",
        label: "Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(22)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "mult",
        label: "Multiplier",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(3.0)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "use_close",
        label: "Use Close",
        kind: IndicatorParamKind::Bool,
        required: false,
        default: Some(ParamValueStatic::Bool(true)),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_TRUE_FALSE,
        notes: None,
    },
];

const PARAM_CKSP: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "p",
        label: "P",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "x",
        label: "X",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(1.0)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "q",
        label: "Q",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(9)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_CORRELATION_CYCLE: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "period",
        label: "Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(20)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "threshold",
        label: "Threshold",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(9.0)),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_CORREL_HL: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(9)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_DAMIANI_VOLATMETER: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "vis_atr",
        label: "Vis ATR",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(13)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "vis_std",
        label: "Vis STD",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(20)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "sed_atr",
        label: "Sed ATR",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(40)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "sed_std",
        label: "Sed STD",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(100)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "threshold",
        label: "Threshold",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(1.4)),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_DVDIQQE: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "period",
        label: "Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(13)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "smoothing_period",
        label: "Smoothing Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(6)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "fast_multiplier",
        label: "Fast Multiplier",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(2.618)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "slow_multiplier",
        label: "Slow Multiplier",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(4.236)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "volume_type",
        label: "Volume Type",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("default")),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "center_type",
        label: "Center Type",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("dynamic")),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "tick_size",
        label: "Tick Size",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(0.01)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_EMD: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "period",
        label: "Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(20)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "delta",
        label: "Delta",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(0.5)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "fraction",
        label: "Fraction",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(0.1)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_ERI: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "period",
        label: "Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(13)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "ma_type",
        label: "MA Type",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("ema")),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_FISHER: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(9)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_FVG_TRAILING_STOP: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "unmitigated_fvg_lookback",
        label: "FVG Lookback",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(5)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "smoothing_length",
        label: "Smoothing Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(9)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "reset_on_cross",
        label: "Reset On Cross",
        kind: IndicatorParamKind::Bool,
        required: false,
        default: Some(ParamValueStatic::Bool(false)),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_TRUE_FALSE,
        notes: None,
    },
];

const PARAM_GATOROSC: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "jaws_length",
        label: "Jaws Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(13)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "jaws_shift",
        label: "Jaws Shift",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(8)),
        min: Some(0.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "teeth_length",
        label: "Teeth Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(8)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "teeth_shift",
        label: "Teeth Shift",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(5)),
        min: Some(0.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "lips_length",
        label: "Lips Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(5)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "lips_shift",
        label: "Lips Shift",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(3)),
        min: Some(0.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_HALFTREND: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "amplitude",
        label: "Amplitude",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(2)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "channel_deviation",
        label: "Channel Deviation",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(2.0)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "atr_period",
        label: "ATR Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(100)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_SAFEZONESTOP: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "period",
        label: "Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(22)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "mult",
        label: "Multiplier",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(2.5)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "max_lookback",
        label: "Max Lookback",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(3)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "direction",
        label: "Direction",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("long")),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_DEVSTOP: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "period",
        label: "Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(20)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "mult",
        label: "Multiplier",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(0.0)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "devtype",
        label: "Dev Type",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(0)),
        min: Some(0.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "direction",
        label: "Direction",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("long")),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "ma_type",
        label: "MA Type",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("sma")),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_MOD_GOD_MODE: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "n1",
        label: "N1",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(17)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "n2",
        label: "N2",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(6)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "n3",
        label: "N3",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(4)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "mode",
        label: "Mode",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("tradition_mg")),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "use_volume",
        label: "Use Volume",
        kind: IndicatorParamKind::Bool,
        required: false,
        default: Some(ParamValueStatic::Bool(true)),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_TRUE_FALSE,
        notes: None,
    },
];

const PARAM_KST: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "sma_period1",
        label: "SMA 1",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "sma_period2",
        label: "SMA 2",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "sma_period3",
        label: "SMA 3",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "sma_period4",
        label: "SMA 4",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(15)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "roc_period1",
        label: "ROC 1",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "roc_period2",
        label: "ROC 2",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(15)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "roc_period3",
        label: "ROC 3",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(20)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "roc_period4",
        label: "ROC 4",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(30)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "signal_period",
        label: "Signal Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(9)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_KAUFMANSTOP: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "period",
        label: "Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(22)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "mult",
        label: "Multiplier",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(2.0)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "direction",
        label: "Direction",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("long")),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "ma_type",
        label: "MA Type",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("sma")),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_LPC: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "cutoff_type",
        label: "Cutoff Type",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("adaptive")),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "fixed_period",
        label: "Fixed Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(20)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "max_cycle_limit",
        label: "Max Cycle Limit",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(60)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "cycle_mult",
        label: "Cycle Mult",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(1.0)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "tr_mult",
        label: "TR Mult",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(1.0)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_MAB: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "fast_period",
        label: "Fast Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "slow_period",
        label: "Slow Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(50)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "devup",
        label: "Dev Up",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(1.0)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "devdn",
        label: "Dev Down",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(1.0)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "fast_ma_type",
        label: "Fast MA Type",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("sma")),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "slow_ma_type",
        label: "Slow MA Type",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("sma")),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_MACZ: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "fast_length",
        label: "Fast Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(12)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "slow_length",
        label: "Slow Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(25)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "signal_length",
        label: "Signal Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(9)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "lengthz",
        label: "Length Z",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(20)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "length_stdev",
        label: "Length StdDev",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(25)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "a",
        label: "A",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(1.0)),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "b",
        label: "B",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(1.0)),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "use_lag",
        label: "Use Lag",
        kind: IndicatorParamKind::Bool,
        required: false,
        default: Some(ParamValueStatic::Bool(false)),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_TRUE_FALSE,
        notes: None,
    },
    IndicatorParamInfo {
        key: "gamma",
        label: "Gamma",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(0.02)),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_MINMAX: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "order",
    label: "Order",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(3)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_MSW: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "period",
    label: "Period",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(5)),
    min: Some(1.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const PARAM_NWE: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "bandwidth",
        label: "Bandwidth",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(8.0)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "multiplier",
        label: "Multiplier",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(3.0)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "lookback",
        label: "Lookback",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(500)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_OTT: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "period",
        label: "Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(2)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "percent",
        label: "Percent",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(1.4)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "ma_type",
        label: "MA Type",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("VAR")),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_OTTO: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "ott_period",
        label: "OTT Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(2)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "ott_percent",
        label: "OTT Percent",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(0.6)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "fast_vidya_length",
        label: "Fast VIDYA Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "slow_vidya_length",
        label: "Slow VIDYA Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(25)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "correcting_constant",
        label: "Correcting Constant",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(100000.0)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "ma_type",
        label: "MA Type",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("VAR")),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_PMA: &[IndicatorParamInfo] = PARAM_NONE;

const PARAM_EHLERS_ADAPTIVE_CG: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "alpha",
        label: "Alpha",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(0.07)),
        min: Some(0.0),
        max: Some(1.0),
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    PARAM_OUTPUT_EHLERS_ADAPTIVE_CG,
];

const PARAM_PRB: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "smooth_data",
        label: "Smooth Data",
        kind: IndicatorParamKind::Bool,
        required: false,
        default: Some(ParamValueStatic::Bool(true)),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_TRUE_FALSE,
        notes: None,
    },
    IndicatorParamInfo {
        key: "smooth_period",
        label: "Smooth Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "regression_period",
        label: "Regression Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(100)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "polynomial_order",
        label: "Polynomial Order",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(2)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "regression_offset",
        label: "Regression Offset",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(0)),
        min: None,
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "ndev",
        label: "NDev",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(2.0)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "equ_from",
        label: "Equ From",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(0)),
        min: Some(0.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_POLYNOMIAL_REGRESSION_EXTRAPOLATION: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "length",
        label: "Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(100)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "extrapolate",
        label: "Extrapolate",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10)),
        min: Some(0.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "degree",
        label: "Degree",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(3)),
        min: Some(0.0),
        max: Some(8.0),
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Must satisfy degree + 1 <= length"),
    },
];

const PARAM_STATISTICAL_TRAILING_STOP: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "data_length",
        label: "Data Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some("Corrected from the provided source so the data_length input controls the true-range window"),
    },
    IndicatorParamInfo {
        key: "normalization_length",
        label: "Distribution Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(100)),
        min: Some(10.0),
        max: Some(5000.0),
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "base_level",
        label: "Base Level",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("level2")),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_STATISTICAL_TRAILING_STOP_BASE_LEVEL,
        notes: None,
    },
];

const PARAM_GEOMETRIC_BIAS_OSCILLATOR: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "length",
        label: "Window Size",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(100)),
        min: Some(10.0),
        max: Some(500.0),
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "multiplier",
        label: "ATR Multiplier",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(2.0)),
        min: Some(0.1),
        max: None,
        step: Some(0.1),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some(
            "Used as the RDP simplification threshold against ATR-normalized price geometry",
        ),
    },
    IndicatorParamInfo {
        key: "atr_length",
        label: "ATR Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "smooth",
        label: "Smoothing",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(1)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_VDUBUS_DIVERGENCE_WAVE_PATTERN_GENERATOR: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "fast_depth",
        label: "Fast Depth",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(9)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "slow_depth",
        label: "Slow Depth",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(24)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "fast_length",
        label: "Fast Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(21)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "slow_length",
        label: "Slow Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(34)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "signal_length",
        label: "Signal Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(5)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "lookback",
        label: "Momentum Pivot Lookback",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(3)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "err_tol",
        label: "Harmonic Tolerance",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(0.15)),
        min: Some(0.01),
        max: Some(0.5),
        step: Some(0.01),
        enum_values: EMPTY_ENUM_VALUES,
        notes: Some(
            "Uses corrected XAD ratio instead of the constant 1.27 shown in the provided source",
        ),
    },
    IndicatorParamInfo {
        key: "show_standard",
        label: "Show Standard",
        kind: IndicatorParamKind::Bool,
        required: false,
        default: Some(ParamValueStatic::Bool(true)),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_TRUE_FALSE,
        notes: None,
    },
    IndicatorParamInfo {
        key: "show_climax",
        label: "Show Climax",
        kind: IndicatorParamKind::Bool,
        required: false,
        default: Some(ParamValueStatic::Bool(true)),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_TRUE_FALSE,
        notes: None,
    },
    IndicatorParamInfo {
        key: "show_rounded",
        label: "Show Rounded",
        kind: IndicatorParamKind::Bool,
        required: false,
        default: Some(ParamValueStatic::Bool(true)),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_TRUE_FALSE,
        notes: None,
    },
    IndicatorParamInfo {
        key: "show_predator",
        label: "Show Predator",
        kind: IndicatorParamKind::Bool,
        required: false,
        default: Some(ParamValueStatic::Bool(true)),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_TRUE_FALSE,
        notes: None,
    },
    IndicatorParamInfo {
        key: "show_gartley",
        label: "Show Gartley",
        kind: IndicatorParamKind::Bool,
        required: false,
        default: Some(ParamValueStatic::Bool(false)),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_TRUE_FALSE,
        notes: None,
    },
    IndicatorParamInfo {
        key: "show_bat",
        label: "Show Bat",
        kind: IndicatorParamKind::Bool,
        required: false,
        default: Some(ParamValueStatic::Bool(false)),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_TRUE_FALSE,
        notes: None,
    },
    IndicatorParamInfo {
        key: "show_butterfly",
        label: "Show Butterfly",
        kind: IndicatorParamKind::Bool,
        required: false,
        default: Some(ParamValueStatic::Bool(false)),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_TRUE_FALSE,
        notes: None,
    },
    IndicatorParamInfo {
        key: "show_crab",
        label: "Show Crab",
        kind: IndicatorParamKind::Bool,
        required: false,
        default: Some(ParamValueStatic::Bool(false)),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_TRUE_FALSE,
        notes: None,
    },
    IndicatorParamInfo {
        key: "show_deep",
        label: "Show Deep",
        kind: IndicatorParamKind::Bool,
        required: false,
        default: Some(ParamValueStatic::Bool(false)),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_TRUE_FALSE,
        notes: None,
    },
    IndicatorParamInfo {
        key: "show_hs",
        label: "Show H&S",
        kind: IndicatorParamKind::Bool,
        required: false,
        default: Some(ParamValueStatic::Bool(true)),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_TRUE_FALSE,
        notes: None,
    },
];

const PARAM_QQE: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "rsi_period",
        label: "RSI Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "smoothing_factor",
        label: "Smoothing Factor",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(5)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "fast_factor",
        label: "Fast Factor",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(4.236)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_QQE_WEIGHTED_OSCILLATOR: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "length",
        label: "Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "factor",
        label: "Factor",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(4.236)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "smooth",
        label: "Smooth",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(5)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "weight",
        label: "Weight",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(2.0)),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "length",
        label: "Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(20)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "smooth",
        label: "Smoothing",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_ADAPTIVE_BOUNDS_RSI: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "rsi_length",
        label: "RSI Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "alpha",
        label: "Learning Rate",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(0.1)),
        min: Some(0.001),
        max: Some(1.0),
        step: Some(0.001),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_RANGE_OSCILLATOR: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "length",
        label: "Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(50)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "mult",
        label: "Multiplier",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(2.0)),
        min: Some(0.1),
        max: None,
        step: Some(0.1),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];
const PARAM_RANGE_FILTERED_TREND_SIGNALS: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "kalman_alpha",
        label: "Kalman Alpha",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(0.01)),
        min: Some(0.0),
        max: None,
        step: Some(0.01),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "kalman_beta",
        label: "Kalman Beta",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(0.1)),
        min: Some(0.0),
        max: None,
        step: Some(0.1),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "kalman_period",
        label: "Kalman Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(77)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "dev",
        label: "Deviation",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(1.2)),
        min: Some(0.0),
        max: None,
        step: Some(0.1),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "supertrend_factor",
        label: "Supertrend Factor",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(0.7)),
        min: Some(0.0),
        max: None,
        step: Some(0.1),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "supertrend_atr_period",
        label: "Supertrend ATR Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(7)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];
const ENUM_VALUES_MSC_BOS_CONFIRMATION: &[&str] = &["Candle Close", "Wicks"];
const PARAM_MARKET_STRUCTURE_CONFLUENCE: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "swing_size",
        label: "Time Horizon",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10)),
        min: Some(2.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "bos_confirmation",
        label: "BOS Confirmation",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("Candle Close")),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_MSC_BOS_CONFIRMATION,
        notes: None,
    },
    IndicatorParamInfo {
        key: "basis_length",
        label: "Basis Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(100)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "atr_length",
        label: "ATR Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "atr_smooth",
        label: "ATR Smoothing",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(21)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "vol_mult",
        label: "Volatility Multiplier",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(2.0)),
        min: Some(0.0),
        max: None,
        step: Some(0.1),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];
const ENUM_VALUES_VWRSI_MA_TYPE: &[&str] = &["EMA", "SMA", "HMA", "SMMA (RMA)", "WMA", "VWMA"];
const PARAM_VOLUME_WEIGHTED_RELATIVE_STRENGTH_INDEX: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "rsi_length",
        label: "RSI Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "range_length",
        label: "Consolidation Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "ma_length",
        label: "RSI MA Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "ma_type",
        label: "RSI MA Type",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("EMA")),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_VWRSI_MA_TYPE,
        notes: None,
    },
];

const PARAM_RANGE_FILTER: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "range_size",
        label: "Range Size",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(2.618)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "range_period",
        label: "Range Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "smooth_range",
        label: "Smooth Range",
        kind: IndicatorParamKind::Bool,
        required: false,
        default: Some(ParamValueStatic::Bool(true)),
        min: None,
        max: None,
        step: None,
        enum_values: ENUM_VALUES_TRUE_FALSE,
        notes: None,
    },
    IndicatorParamInfo {
        key: "smooth_period",
        label: "Smooth Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(27)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_RSMK: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "lookback",
        label: "Lookback",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(90)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "period",
        label: "Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(3)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "signal_period",
        label: "Signal Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(20)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "matype",
        label: "MA Type",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("ema")),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "signal_matype",
        label: "Signal MA Type",
        kind: IndicatorParamKind::EnumString,
        required: false,
        default: Some(ParamValueStatic::EnumString("ema")),
        min: None,
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_VOSS: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "period",
        label: "Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(20)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "predict",
        label: "Predict",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(3)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "bandwidth",
        label: "Bandwidth",
        kind: IndicatorParamKind::Float,
        required: false,
        default: Some(ParamValueStatic::Float(0.25)),
        min: Some(0.0),
        max: None,
        step: None,
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_STC: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "fast_period",
        label: "Fast Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(23)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "slow_period",
        label: "Slow Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(50)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "k_period",
        label: "K Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "d_period",
        label: "D Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(3)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_RVI: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "period",
        label: "Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "ma_len",
        label: "MA Length",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "matype",
        label: "MA Type",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(1)),
        min: Some(0.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "devtype",
        label: "Deviation Type",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(0)),
        min: Some(0.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_COPPOCK: &[IndicatorParamInfo] = &[
    IndicatorParamInfo {
        key: "short_roc_period",
        label: "Short ROC Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(11)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "long_roc_period",
        label: "Long ROC Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(14)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
    IndicatorParamInfo {
        key: "ma_period",
        label: "MA Period",
        kind: IndicatorParamKind::Int,
        required: false,
        default: Some(ParamValueStatic::Int(10)),
        min: Some(1.0),
        max: None,
        step: Some(1.0),
        enum_values: EMPTY_ENUM_VALUES,
        notes: None,
    },
];

const PARAM_PIVOT: &[IndicatorParamInfo] = &[IndicatorParamInfo {
    key: "mode",
    label: "Mode",
    kind: IndicatorParamKind::Int,
    required: false,
    default: Some(ParamValueStatic::Int(3)),
    min: Some(0.0),
    max: None,
    step: Some(1.0),
    enum_values: EMPTY_ENUM_VALUES,
    notes: None,
}];

const SUPPLEMENTAL_SEED_NOTE: &str =
    "Phase 1 seed metadata; parameter and capability metadata will expand.";

struct SupplementalIndicatorSeed {
    id: &'static str,
    label: &'static str,
    category: &'static str,
    input_kind: IndicatorInputKind,
    outputs: &'static [IndicatorOutputInfo],
    params: &'static [IndicatorParamInfo],
}

const SUPPLEMENTAL_INDICATORS: &[SupplementalIndicatorSeed] = &[
    SupplementalIndicatorSeed {
        id: "adx",
        label: "ADX",
        category: "trend",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_ADX_PERIOD,
    },
    SupplementalIndicatorSeed {
        id: "adxr",
        label: "ADXR",
        category: "trend",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_ADX_PERIOD,
    },
    SupplementalIndicatorSeed {
        id: "dx",
        label: "DX",
        category: "trend",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_DX_PERIOD,
    },
    SupplementalIndicatorSeed {
        id: "di",
        label: "DI",
        category: "trend",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_PLUS_MINUS,
        params: PARAM_DI,
    },
    SupplementalIndicatorSeed {
        id: "dm",
        label: "DM",
        category: "trend",
        input_kind: IndicatorInputKind::HighLow,
        outputs: OUTPUTS_PLUS_MINUS,
        params: PARAM_DM,
    },
    SupplementalIndicatorSeed {
        id: "vi",
        label: "VI",
        category: "trend",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_PLUS_MINUS,
        params: PARAM_VI,
    },
    SupplementalIndicatorSeed {
        id: "donchian",
        label: "Donchian",
        category: "volatility",
        input_kind: IndicatorInputKind::HighLow,
        outputs: OUTPUTS_BOLLINGER,
        params: PARAM_DONCHIAN,
    },
    SupplementalIndicatorSeed {
        id: "supertrend",
        label: "SuperTrend",
        category: "trend",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_TREND_CHANGED,
        params: PARAM_SUPERTREND,
    },
    SupplementalIndicatorSeed {
        id: "adjustable_ma_alternating_extremities",
        label: "Adjustable MA & Alternating Extremities",
        category: "trend",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_ADJUSTABLE_MA_ALTERNATING_EXTREMITIES,
        params: PARAM_ADJUSTABLE_MA_ALTERNATING_EXTREMITIES,
    },
    SupplementalIndicatorSeed {
        id: "supertrend_recovery",
        label: "SuperTrend Recovery",
        category: "trend",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_SUPERTREND_RECOVERY,
        params: PARAM_SUPERTREND_RECOVERY,
    },
    SupplementalIndicatorSeed {
        id: "keltner",
        label: "Keltner",
        category: "volatility",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_BOLLINGER,
        params: PARAM_KELTNER,
    },
    SupplementalIndicatorSeed {
        id: "aroon",
        label: "Aroon",
        category: "trend",
        input_kind: IndicatorInputKind::HighLow,
        outputs: OUTPUTS_UP_DOWN,
        params: PARAM_AROON,
    },
    SupplementalIndicatorSeed {
        id: "aroonosc",
        label: "Aroon Oscillator",
        category: "momentum",
        input_kind: IndicatorInputKind::HighLow,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_AROON,
    },
    SupplementalIndicatorSeed {
        id: "srsi",
        label: "Stochastic RSI",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_STOCH,
        params: PARAM_SRSI,
    },
    SupplementalIndicatorSeed {
        id: "stochastic_connors_rsi",
        label: "Stochastic Connors RSI",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_STOCH,
        params: PARAM_STOCHASTIC_CONNORS_RSI,
    },
    SupplementalIndicatorSeed {
        id: "kdj",
        label: "KDJ",
        category: "momentum",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_KDJ,
        params: PARAM_KDJ,
    },
    SupplementalIndicatorSeed {
        id: "squeeze_momentum",
        label: "Squeeze Momentum",
        category: "momentum",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_SQUEEZE_MOMENTUM,
        params: PARAM_SQUEEZE_MOMENTUM,
    },
    SupplementalIndicatorSeed {
        id: "wavetrend",
        label: "WaveTrend",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_WAVETREND,
        params: PARAM_WAVETREND,
    },
    SupplementalIndicatorSeed {
        id: "wto",
        label: "WTO",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_WTO,
        params: PARAM_WTO,
    },
    SupplementalIndicatorSeed {
        id: "accumulation_swing_index",
        label: "Accumulation Swing Index",
        category: "volatility",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_ACCUMULATION_SWING_INDEX,
    },
    SupplementalIndicatorSeed {
        id: "daily_factor",
        label: "Daily Factor",
        category: "momentum",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_DAILY_FACTOR,
        params: PARAM_DAILY_FACTOR,
    },
    SupplementalIndicatorSeed {
        id: "moving_average_cross_probability",
        label: "Moving Average Cross Probability",
        category: "momentum",
        input_kind: IndicatorInputKind::Candles,
        outputs: OUTPUTS_MOVING_AVERAGE_CROSS_PROBABILITY,
        params: PARAM_MOVING_AVERAGE_CROSS_PROBABILITY,
    },
    SupplementalIndicatorSeed {
        id: "bulls_v_bears",
        label: "Bulls v Bears",
        category: "momentum",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_BULLS_V_BEARS,
        params: PARAM_BULLS_V_BEARS,
    },
    SupplementalIndicatorSeed {
        id: "regression_slope_oscillator",
        label: "Regression Slope Oscillator",
        category: "momentum",
        input_kind: IndicatorInputKind::Candles,
        outputs: OUTPUTS_REGRESSION_SLOPE_OSCILLATOR,
        params: PARAM_REGRESSION_SLOPE_OSCILLATOR,
    },
    SupplementalIndicatorSeed {
        id: "smooth_theil_sen",
        label: "Smooth Theil-Sen",
        category: "trend",
        input_kind: IndicatorInputKind::Candles,
        outputs: OUTPUTS_SMOOTH_THEIL_SEN,
        params: PARAM_SMOOTH_THEIL_SEN,
    },
    SupplementalIndicatorSeed {
        id: "l2_ehlers_signal_to_noise",
        label: "L2 Ehlers Signal to Noise",
        category: "cycle",
        input_kind: IndicatorInputKind::Candles,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_L2_EHLERS_SIGNAL_TO_NOISE,
    },
    SupplementalIndicatorSeed {
        id: "ehlers_smoothed_adaptive_momentum",
        label: "Ehlers Smoothed Adaptive Momentum",
        category: "cycle",
        input_kind: IndicatorInputKind::Candles,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_EHLERS_SMOOTHED_ADAPTIVE_MOMENTUM,
    },
    SupplementalIndicatorSeed {
        id: "ehlers_adaptive_cyber_cycle",
        label: "Ehlers Adaptive Cyber Cycle",
        category: "cycle",
        input_kind: IndicatorInputKind::Candles,
        outputs: OUTPUTS_EHLERS_SIMPLE_CYCLE_INDICATOR,
        params: PARAM_EHLERS_ADAPTIVE_CYBER_CYCLE,
    },
    SupplementalIndicatorSeed {
        id: "ehlers_simple_cycle_indicator",
        label: "Ehlers Simple Cycle Indicator",
        category: "cycle",
        input_kind: IndicatorInputKind::Candles,
        outputs: OUTPUTS_EHLERS_SIMPLE_CYCLE_INDICATOR,
        params: PARAM_EHLERS_SIMPLE_CYCLE_INDICATOR,
    },
    SupplementalIndicatorSeed {
        id: "l1_ehlers_phasor",
        label: "L1 Ehlers Phasor",
        category: "cycle",
        input_kind: IndicatorInputKind::Candles,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_L1_EHLERS_PHASOR,
    },
    SupplementalIndicatorSeed {
        id: "andean_oscillator",
        label: "Andean Oscillator",
        category: "trend",
        input_kind: IndicatorInputKind::Candles,
        outputs: OUTPUTS_ANDEAN_OSCILLATOR,
        params: PARAM_ANDEAN_OSCILLATOR,
    },
    SupplementalIndicatorSeed {
        id: "cycle_channel_oscillator",
        label: "Cycle Channel Oscillator",
        category: "cycle",
        input_kind: IndicatorInputKind::Candles,
        outputs: OUTPUTS_CYCLE_CHANNEL_OSCILLATOR,
        params: PARAM_CYCLE_CHANNEL_OSCILLATOR,
    },
    SupplementalIndicatorSeed {
        id: "ewma_volatility",
        label: "EWMA Volatility",
        category: "volatility",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_EWMA_VOLATILITY,
    },
    SupplementalIndicatorSeed {
        id: "ichimoku_oscillator",
        label: "Ichimoku Oscillator",
        category: "trend",
        input_kind: IndicatorInputKind::Candles,
        outputs: OUTPUTS_ICHIMOKU_OSCILLATOR,
        params: PARAM_ICHIMOKU_OSCILLATOR,
    },
    SupplementalIndicatorSeed {
        id: "random_walk_index",
        label: "Random Walk Index",
        category: "volatility",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_RANDOM_WALK_INDEX,
        params: PARAM_RANDOM_WALK_INDEX,
    },
    SupplementalIndicatorSeed {
        id: "price_moving_average_ratio_percentile",
        label: "Price Moving Average Ratio & Percentile",
        category: "momentum",
        input_kind: IndicatorInputKind::Candles,
        outputs: OUTPUTS_PRICE_MOVING_AVERAGE_RATIO_PERCENTILE,
        params: PARAM_PRICE_MOVING_AVERAGE_RATIO_PERCENTILE,
    },
    SupplementalIndicatorSeed {
        id: "relative_strength_index_wave_indicator",
        label: "Relative Strength Index Wave Indicator",
        category: "momentum",
        input_kind: IndicatorInputKind::Candles,
        outputs: OUTPUTS_RELATIVE_STRENGTH_INDEX_WAVE_INDICATOR,
        params: PARAM_RELATIVE_STRENGTH_INDEX_WAVE_INDICATOR,
    },
    SupplementalIndicatorSeed {
        id: "mesa_stochastic_multi_length",
        label: "MESA Stochastic Multi Length",
        category: "momentum",
        input_kind: IndicatorInputKind::Candles,
        outputs: OUTPUTS_MESA_STOCHASTIC_MULTI_LENGTH,
        params: PARAM_MESA_STOCHASTIC_MULTI_LENGTH,
    },
    SupplementalIndicatorSeed {
        id: "spearman_correlation",
        label: "Spearman Correlation",
        category: "relative_strength",
        input_kind: IndicatorInputKind::Candles,
        outputs: OUTPUTS_SPEARMAN_CORRELATION,
        params: PARAM_SPEARMAN_CORRELATION,
    },
    SupplementalIndicatorSeed {
        id: "trend_trigger_factor",
        label: "Trend Trigger Factor",
        category: "momentum",
        input_kind: IndicatorInputKind::HighLow,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_TREND_TRIGGER_FACTOR,
    },
    SupplementalIndicatorSeed {
        id: "volatility_quality_index",
        label: "Volatility Quality Index",
        category: "volatility",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_VOLATILITY_QUALITY_INDEX,
        params: PARAM_VOLATILITY_QUALITY_INDEX,
    },
    SupplementalIndicatorSeed {
        id: "yang_zhang_volatility",
        label: "Yang-Zhang Volatility",
        category: "volatility",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_YANG_ZHANG,
        params: PARAM_YANG_ZHANG,
    },
    SupplementalIndicatorSeed {
        id: "historical_volatility_percentile",
        label: "Historical Volatility Percentile",
        category: "volatility",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_HVP,
        params: PARAM_HISTORICAL_VOLATILITY_PERCENTILE,
    },
    SupplementalIndicatorSeed {
        id: "ehlers_detrending_filter",
        label: "Ehlers Detrending Filter",
        category: "trend",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_EHLERS_DETRENDING_FILTER,
        params: PARAM_EHLERS_DETRENDING_FILTER,
    },
    SupplementalIndicatorSeed {
        id: "keltner_channel_width_oscillator",
        label: "Keltner Channel Width Oscillator",
        category: "volatility",
        input_kind: IndicatorInputKind::Candles,
        outputs: OUTPUTS_KELTNER_CHANNEL_WIDTH_OSCILLATOR,
        params: PARAM_KELTNER_CHANNEL_WIDTH_OSCILLATOR,
    },
    SupplementalIndicatorSeed {
        id: "market_meanness_index",
        label: "Market Meanness Index",
        category: "statistics",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_MARKET_MEANNESS_INDEX,
        params: PARAM_MARKET_MEANNESS_INDEX,
    },
    SupplementalIndicatorSeed {
        id: "price_density_market_noise",
        label: "Price Density (Market Noise)",
        category: "volatility",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_PRICE_DENSITY_MARKET_NOISE,
        params: PARAM_PRICE_DENSITY_MARKET_NOISE,
    },
    SupplementalIndicatorSeed {
        id: "momentum_ratio_oscillator",
        label: "Momentum Ratio Oscillator",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_MOMENTUM_RATIO_OSCILLATOR,
        params: PARAM_MOMENTUM_RATIO_OSCILLATOR,
    },
    SupplementalIndicatorSeed {
        id: "hypertrend",
        label: "HyperTrend",
        category: "trend",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_HYPERTREND,
        params: PARAM_HYPERTREND,
    },
    SupplementalIndicatorSeed {
        id: "adaptive_schaff_trend_cycle",
        label: "Adaptive Schaff Trend Cycle",
        category: "momentum",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_ADAPTIVE_SCHAFF_TREND_CYCLE,
        params: PARAM_ADAPTIVE_SCHAFF_TREND_CYCLE,
    },
    SupplementalIndicatorSeed {
        id: "smoothed_gaussian_trend_filter",
        label: "Smoothed Gaussian Trend Filter",
        category: "trend",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_SMOOTHED_GAUSSIAN_TREND_FILTER,
        params: PARAM_SMOOTHED_GAUSSIAN_TREND_FILTER,
    },
    SupplementalIndicatorSeed {
        id: "logarithmic_moving_average",
        label: "Logarithmic Moving Average",
        category: "moving_averages",
        input_kind: IndicatorInputKind::CloseVolume,
        outputs: OUTPUTS_LOGARITHMIC_MOVING_AVERAGE,
        params: PARAM_LOGARITHMIC_MOVING_AVERAGE,
    },
    SupplementalIndicatorSeed {
        id: "ict_propulsion_block",
        label: "ICT Propulsion Block",
        category: "trend",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_ICT_PROPULSION_BLOCK,
        params: PARAM_ICT_PROPULSION_BLOCK,
    },
    SupplementalIndicatorSeed {
        id: "supertrend_oscillator",
        label: "SuperTrend Oscillator",
        category: "momentum",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_SUPERTREND_OSCILLATOR,
        params: PARAM_SUPERTREND_OSCILLATOR,
    },
    SupplementalIndicatorSeed {
        id: "leavitt_convolution_acceleration",
        label: "Leavitt Convolution Acceleration",
        category: "trend",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_LEAVITT_CONVOLUTION_ACCELERATION,
        params: PARAM_LEAVITT_CONVOLUTION_ACCELERATION,
    },
    SupplementalIndicatorSeed {
        id: "impulse_macd",
        label: "Impulse MACD",
        category: "momentum",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_IMPULSE_MACD,
        params: PARAM_IMPULSE_MACD,
    },
    SupplementalIndicatorSeed {
        id: "insync_index",
        label: "Insync Index",
        category: "momentum",
        input_kind: IndicatorInputKind::Ohlcv,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_INSYNC_INDEX,
    },
    SupplementalIndicatorSeed {
        id: "volatility_ratio_adaptive_rsx",
        label: "Volatility Ratio Adaptive RSX",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_LINE_SIGNAL,
        params: PARAM_VOLATILITY_RATIO_ADAPTIVE_RSX,
    },
    SupplementalIndicatorSeed {
        id: "on_balance_volume_oscillator",
        label: "On Balance Volume Oscillator",
        category: "volume",
        input_kind: IndicatorInputKind::CloseVolume,
        outputs: OUTPUTS_LINE_SIGNAL,
        params: PARAM_ON_BALANCE_VOLUME_OSCILLATOR,
    },
    SupplementalIndicatorSeed {
        id: "parkinson_volatility",
        label: "Parkinson Volatility",
        category: "volatility",
        input_kind: IndicatorInputKind::HighLow,
        outputs: OUTPUTS_PARKINSON,
        params: PARAM_PARKINSON,
    },
    SupplementalIndicatorSeed {
        id: "pretty_good_oscillator",
        label: "Pretty Good Oscillator",
        category: "momentum",
        input_kind: IndicatorInputKind::Candles,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_ATR_PERIOD,
    },
    SupplementalIndicatorSeed {
        id: "psychological_line",
        label: "Psychological Line",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_PSYCHOLOGICAL_LINE,
    },
    SupplementalIndicatorSeed {
        id: "rank_correlation_index",
        label: "Rank Correlation Index",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_RANK_CORRELATION_INDEX,
    },
    SupplementalIndicatorSeed {
        id: "linear_regression_intensity",
        label: "Linear Regression Intensity",
        category: "trend",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_LINEAR_REGRESSION_INTENSITY,
    },
    SupplementalIndicatorSeed {
        id: "trend_continuation_factor",
        label: "Trend Continuation Factor",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_TREND_CONTINUATION_FACTOR,
        params: PARAM_TREND_CONTINUATION_FACTOR,
    },
    SupplementalIndicatorSeed {
        id: "trend_follower",
        label: "Trend Follower",
        category: "trend",
        input_kind: IndicatorInputKind::Ohlcv,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_TREND_FOLLOWER,
    },
    SupplementalIndicatorSeed {
        id: "trend_direction_force_index",
        label: "Trend Direction Force Index",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_TREND_DIRECTION_FORCE_INDEX,
    },
    SupplementalIndicatorSeed {
        id: "twiggs_money_flow",
        label: "Twiggs Money Flow",
        category: "volume",
        input_kind: IndicatorInputKind::Ohlcv,
        outputs: OUTPUTS_TWIGGS_MONEY_FLOW,
        params: PARAM_TWIGGS_MONEY_FLOW,
    },
    SupplementalIndicatorSeed {
        id: "volume_weighted_stochastic_rsi",
        label: "Volume Weighted Stochastic RSI",
        category: "momentum",
        input_kind: IndicatorInputKind::CloseVolume,
        outputs: OUTPUTS_VOLUME_WEIGHTED_STOCHASTIC_RSI,
        params: PARAM_VOLUME_WEIGHTED_STOCHASTIC_RSI,
    },
    SupplementalIndicatorSeed {
        id: "stochastic_adaptive_d",
        label: "Stochastic Adaptive %D",
        category: "momentum",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_STOCHASTIC_ADAPTIVE_D,
        params: PARAM_STOCHASTIC_ADAPTIVE_D,
    },
    SupplementalIndicatorSeed {
        id: "volume_zone_oscillator",
        label: "Volume Zone Oscillator",
        category: "volume",
        input_kind: IndicatorInputKind::CloseVolume,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_VOLUME_ZONE_OSCILLATOR,
    },
    SupplementalIndicatorSeed {
        id: "vwap_deviation_oscillator",
        label: "VWAP Deviation Oscillator",
        category: "volatility",
        input_kind: IndicatorInputKind::Candles,
        outputs: OUTPUTS_VWAP_DEVIATION_OSCILLATOR,
        params: PARAM_VWAP_DEVIATION_OSCILLATOR,
    },
    SupplementalIndicatorSeed {
        id: "atr",
        label: "ATR",
        category: "volatility",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_ATR_PERIOD,
    },
    SupplementalIndicatorSeed {
        id: "ad",
        label: "AD",
        category: "volume",
        input_kind: IndicatorInputKind::Ohlcv,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_NONE,
    },
    SupplementalIndicatorSeed {
        id: "adosc",
        label: "ADOSC",
        category: "volume",
        input_kind: IndicatorInputKind::Ohlcv,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_ADOSC,
    },
    SupplementalIndicatorSeed {
        id: "ao",
        label: "AO",
        category: "momentum",
        input_kind: IndicatorInputKind::HighLow,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_AO,
    },
    SupplementalIndicatorSeed {
        id: "bop",
        label: "BOP",
        category: "momentum",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_NONE,
    },
    SupplementalIndicatorSeed {
        id: "emv",
        label: "EMV",
        category: "volume",
        input_kind: IndicatorInputKind::Ohlcv,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_NONE,
    },
    SupplementalIndicatorSeed {
        id: "efi",
        label: "EFI",
        category: "volume",
        input_kind: IndicatorInputKind::CloseVolume,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_EFI_PERIOD,
    },
    SupplementalIndicatorSeed {
        id: "dti",
        label: "DTI",
        category: "momentum",
        input_kind: IndicatorInputKind::HighLow,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_DTI,
    },
    SupplementalIndicatorSeed {
        id: "mfi",
        label: "MFI",
        category: "volume",
        input_kind: IndicatorInputKind::CloseVolume,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_MFI_PERIOD,
    },
    SupplementalIndicatorSeed {
        id: "mass",
        label: "MASS",
        category: "volatility",
        input_kind: IndicatorInputKind::HighLow,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_MASS_PERIOD,
    },
    SupplementalIndicatorSeed {
        id: "kvo",
        label: "KVO",
        category: "volume",
        input_kind: IndicatorInputKind::Ohlcv,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_KVO,
    },
    SupplementalIndicatorSeed {
        id: "vosc",
        label: "VOSC",
        category: "volume",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_VOSC,
    },
    SupplementalIndicatorSeed {
        id: "rsi",
        label: "RSI",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_RSI_PERIOD,
    },
    SupplementalIndicatorSeed {
        id: "rsx",
        label: "RSX",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_RSI_PERIOD,
    },
    SupplementalIndicatorSeed {
        id: "roc",
        label: "ROC",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_ROC_PERIOD,
    },
    SupplementalIndicatorSeed {
        id: "linear_correlation_oscillator",
        label: "Linear Correlation Oscillator",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_LINEAR_CORRELATION_OSCILLATOR_PERIOD,
    },
    SupplementalIndicatorSeed {
        id: "ehlers_fm_demodulator",
        label: "Ehlers FM Demodulator",
        category: "momentum",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_EHLERS_FM_DEMODULATOR_PERIOD,
    },
    SupplementalIndicatorSeed {
        id: "ehlers_adaptive_cg",
        label: "Ehlers Adaptive CG",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_EHLERS_ADAPTIVE_CG,
        params: PARAM_EHLERS_ADAPTIVE_CG,
    },
    SupplementalIndicatorSeed {
        id: "adaptive_momentum_oscillator",
        label: "Adaptive Momentum Oscillator",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_ADAPTIVE_MOMENTUM_OSCILLATOR,
        params: PARAM_ADAPTIVE_MOMENTUM_OSCILLATOR,
    },
    SupplementalIndicatorSeed {
        id: "velocity",
        label: "Velocity",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_VELOCITY,
    },
    SupplementalIndicatorSeed {
        id: "normalized_volume_true_range",
        label: "Normalized Volume True Range",
        category: "volume",
        input_kind: IndicatorInputKind::Ohlcv,
        outputs: OUTPUTS_NORMALIZED_VOLUME_TRUE_RANGE,
        params: PARAM_NORMALIZED_VOLUME_TRUE_RANGE,
    },
    SupplementalIndicatorSeed {
        id: "exponential_trend",
        label: "Exponential Trend",
        category: "trend",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_EXPONENTIAL_TREND,
        params: PARAM_EXPONENTIAL_TREND,
    },
    SupplementalIndicatorSeed {
        id: "trend_flow_trail",
        label: "Trend Flow Trail",
        category: "trend",
        input_kind: IndicatorInputKind::Ohlcv,
        outputs: OUTPUTS_TREND_FLOW_TRAIL,
        params: PARAM_TREND_FLOW_TRAIL,
    },
    SupplementalIndicatorSeed {
        id: "range_breakout_signals",
        label: "Range Breakout Signals",
        category: "trend",
        input_kind: IndicatorInputKind::Ohlcv,
        outputs: OUTPUTS_RANGE_BREAKOUT_SIGNALS,
        params: PARAM_RANGE_BREAKOUT_SIGNALS,
    },
    SupplementalIndicatorSeed {
        id: "apo",
        label: "APO",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_APO,
    },
    SupplementalIndicatorSeed {
        id: "cci",
        label: "CCI",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_CCI_PERIOD,
    },
    SupplementalIndicatorSeed {
        id: "cci_cycle",
        label: "CCI Cycle",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_CCI_CYCLE,
    },
    SupplementalIndicatorSeed {
        id: "cfo",
        label: "CFO",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_CFO,
    },
    SupplementalIndicatorSeed {
        id: "cg",
        label: "CG",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_CG_PERIOD,
    },
    SupplementalIndicatorSeed {
        id: "er",
        label: "ER",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_ER_PERIOD,
    },
    SupplementalIndicatorSeed {
        id: "kurtosis",
        label: "Kurtosis",
        category: "statistics",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_KURTOSIS_PERIOD,
    },
    SupplementalIndicatorSeed {
        id: "natr",
        label: "NATR",
        category: "volatility",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_NATR_PERIOD,
    },
    SupplementalIndicatorSeed {
        id: "mean_ad",
        label: "Mean AD",
        category: "statistics",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_MEAN_AD_PERIOD,
    },
    SupplementalIndicatorSeed {
        id: "medium_ad",
        label: "Medium AD",
        category: "statistics",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_MEDIUM_AD_PERIOD,
    },
    SupplementalIndicatorSeed {
        id: "deviation",
        label: "Deviation",
        category: "statistics",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_DEVIATION,
    },
    SupplementalIndicatorSeed {
        id: "dpo",
        label: "DPO",
        category: "trend",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_DPO_PERIOD,
    },
    SupplementalIndicatorSeed {
        id: "fosc",
        label: "FOSC",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_FOSC_PERIOD,
    },
    SupplementalIndicatorSeed {
        id: "ift_rsi",
        label: "IFT RSI",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_IFT_RSI,
    },
    SupplementalIndicatorSeed {
        id: "linearreg_angle",
        label: "Linear Regression Angle",
        category: "trend",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_LINEARREG_ANGLE_PERIOD,
    },
    SupplementalIndicatorSeed {
        id: "linearreg_intercept",
        label: "Linear Regression Intercept",
        category: "trend",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_LINEARREG_INTERCEPT_PERIOD,
    },
    SupplementalIndicatorSeed {
        id: "linearreg_slope",
        label: "Linear Regression Slope",
        category: "trend",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_LINEARREG_SLOPE_PERIOD,
    },
    SupplementalIndicatorSeed {
        id: "pfe",
        label: "PFE",
        category: "trend",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_PFE,
    },
    SupplementalIndicatorSeed {
        id: "qstick",
        label: "QStick",
        category: "trend",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_QSTICK,
    },
    SupplementalIndicatorSeed {
        id: "reverse_rsi",
        label: "Reverse RSI",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_REVERSE_RSI,
    },
    SupplementalIndicatorSeed {
        id: "percentile_nearest_rank",
        label: "Percentile Nearest Rank",
        category: "statistics",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_PERCENTILE_NEAREST_RANK,
    },
    SupplementalIndicatorSeed {
        id: "ui",
        label: "UI",
        category: "volatility",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_UI,
    },
    SupplementalIndicatorSeed {
        id: "zscore",
        label: "Zscore",
        category: "statistics",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_ZSCORE,
    },
    SupplementalIndicatorSeed {
        id: "medprice",
        label: "Medprice",
        category: "price",
        input_kind: IndicatorInputKind::HighLow,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_NONE,
    },
    SupplementalIndicatorSeed {
        id: "midpoint",
        label: "Midpoint",
        category: "price",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_MIDPOINT_PERIOD,
    },
    SupplementalIndicatorSeed {
        id: "midprice",
        label: "Midprice",
        category: "price",
        input_kind: IndicatorInputKind::HighLow,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_MIDPRICE_PERIOD,
    },
    SupplementalIndicatorSeed {
        id: "wclprice",
        label: "WCLPRICE",
        category: "price",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_NONE,
    },
    SupplementalIndicatorSeed {
        id: "obv",
        label: "OBV",
        category: "volume",
        input_kind: IndicatorInputKind::CloseVolume,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_NONE,
    },
    SupplementalIndicatorSeed {
        id: "vpt",
        label: "VPT",
        category: "volume",
        input_kind: IndicatorInputKind::CloseVolume,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_NONE,
    },
    SupplementalIndicatorSeed {
        id: "nvi",
        label: "NVI",
        category: "volume",
        input_kind: IndicatorInputKind::CloseVolume,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_NONE,
    },
    SupplementalIndicatorSeed {
        id: "pvi",
        label: "PVI",
        category: "volume",
        input_kind: IndicatorInputKind::CloseVolume,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_PVI,
    },
    SupplementalIndicatorSeed {
        id: "mom",
        label: "MOM",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_MOM_PERIOD,
    },
    SupplementalIndicatorSeed {
        id: "cmo",
        label: "CMO",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_CMO_PERIOD,
    },
    SupplementalIndicatorSeed {
        id: "dec_osc",
        label: "Dec Osc",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_DEC_OSC,
    },
    SupplementalIndicatorSeed {
        id: "lrsi",
        label: "LRSI",
        category: "momentum",
        input_kind: IndicatorInputKind::HighLow,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_LRSI_ALPHA,
    },
    SupplementalIndicatorSeed {
        id: "rocp",
        label: "ROCP",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_ROCP_PERIOD,
    },
    SupplementalIndicatorSeed {
        id: "rocr",
        label: "ROCR",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_ROCR_PERIOD,
    },
    SupplementalIndicatorSeed {
        id: "tsf",
        label: "TSF",
        category: "trend",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_TSF_PERIOD,
    },
    SupplementalIndicatorSeed {
        id: "adaptive_macd",
        label: "Adaptive MACD",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_MACD,
        params: PARAM_ADAPTIVE_MACD,
    },
    SupplementalIndicatorSeed {
        id: "polynomial_regression_extrapolation",
        label: "Polynomial Regression Extrapolation",
        category: "trend",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_POLYNOMIAL_REGRESSION_EXTRAPOLATION,
    },
    SupplementalIndicatorSeed {
        id: "statistical_trailing_stop",
        label: "Statistical Trailing Stop",
        category: "trend",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_STATISTICAL_TRAILING_STOP,
        params: PARAM_STATISTICAL_TRAILING_STOP,
    },
    SupplementalIndicatorSeed {
        id: "standardized_psar_oscillator",
        label: "Standardized PSAR Oscillator",
        category: "momentum",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_STANDARDIZED_PSAR_OSCILLATOR,
        params: PARAM_STANDARDIZED_PSAR_OSCILLATOR,
    },
    SupplementalIndicatorSeed {
        id: "geometric_bias_oscillator",
        label: "Geometric Bias Oscillator",
        category: "momentum",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_GEOMETRIC_BIAS_OSCILLATOR,
    },
    SupplementalIndicatorSeed {
        id: "vdubus_divergence_wave_pattern_generator",
        label: "Vdubus Divergence Wave Pattern Generator",
        category: "pattern_recognition",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_VDUBUS_DIVERGENCE_WAVE_PATTERN_GENERATOR,
        params: PARAM_VDUBUS_DIVERGENCE_WAVE_PATTERN_GENERATOR,
    },
    SupplementalIndicatorSeed {
        id: "ppo",
        label: "PPO",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_PPO,
    },
    SupplementalIndicatorSeed {
        id: "trix",
        label: "TRIX",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_TRIX_PERIOD,
    },
    SupplementalIndicatorSeed {
        id: "tsi",
        label: "TSI",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_TSI,
    },
    SupplementalIndicatorSeed {
        id: "stddev",
        label: "StdDev",
        category: "volatility",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_STDDEV,
    },
    SupplementalIndicatorSeed {
        id: "var",
        label: "VAR",
        category: "volatility",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_VAR,
    },
    SupplementalIndicatorSeed {
        id: "willr",
        label: "WILLR",
        category: "momentum",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_WILLR_PERIOD,
    },
    SupplementalIndicatorSeed {
        id: "ultosc",
        label: "ULTOSC",
        category: "momentum",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_ULTOSC,
    },
    SupplementalIndicatorSeed {
        id: "macd",
        label: "MACD",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_MACD,
        params: PARAM_MACD,
    },
    SupplementalIndicatorSeed {
        id: "bollinger_bands",
        label: "Bollinger Bands",
        category: "volatility",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_BOLLINGER,
        params: PARAM_BOLLINGER,
    },
    SupplementalIndicatorSeed {
        id: "bollinger_bands_width",
        label: "Bollinger Bands Width",
        category: "volatility",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_BOLLINGER,
    },
    SupplementalIndicatorSeed {
        id: "stoch",
        label: "Stochastic",
        category: "momentum",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_STOCH,
        params: PARAM_STOCH,
    },
    SupplementalIndicatorSeed {
        id: "stochf",
        label: "Fast Stochastic",
        category: "momentum",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_STOCH,
        params: PARAM_STOCHF,
    },
    SupplementalIndicatorSeed {
        id: "vwmacd",
        label: "VWMACD",
        category: "volume",
        input_kind: IndicatorInputKind::CloseVolume,
        outputs: OUTPUTS_MACD,
        params: PARAM_VW_MACD,
    },
    SupplementalIndicatorSeed {
        id: "vpci",
        label: "VPCI",
        category: "volume",
        input_kind: IndicatorInputKind::CloseVolume,
        outputs: OUTPUTS_VPCI,
        params: PARAM_VPCI,
    },
    SupplementalIndicatorSeed {
        id: "ttm_trend",
        label: "TTM Trend",
        category: "trend",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_VALUE_BOOL,
        params: PARAM_TTM_TREND,
    },
    SupplementalIndicatorSeed {
        id: "ttm_squeeze",
        label: "TTM Squeeze",
        category: "momentum",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_TTM_SQUEEZE,
        params: PARAM_TTM_SQUEEZE,
    },
    SupplementalIndicatorSeed {
        id: "acosc",
        label: "Acosc",
        category: "momentum",
        input_kind: IndicatorInputKind::HighLow,
        outputs: OUTPUTS_ACOSC,
        params: PARAM_ACOSC,
    },
    SupplementalIndicatorSeed {
        id: "alligator",
        label: "Alligator",
        category: "trend",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_ALLIGATOR,
        params: PARAM_ALLIGATOR,
    },
    SupplementalIndicatorSeed {
        id: "alphatrend",
        label: "AlphaTrend",
        category: "trend",
        input_kind: IndicatorInputKind::Ohlcv,
        outputs: OUTPUTS_K1_K2,
        params: PARAM_ALPHATREND,
    },
    SupplementalIndicatorSeed {
        id: "aso",
        label: "ASO",
        category: "momentum",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_BULLS_BEARS,
        params: PARAM_ASO,
    },
    SupplementalIndicatorSeed {
        id: "avsl",
        label: "AVSL",
        category: "trend",
        input_kind: IndicatorInputKind::Ohlcv,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_AVSL,
    },
    SupplementalIndicatorSeed {
        id: "bandpass",
        label: "BandPass",
        category: "cycle",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_BANDPASS,
        params: PARAM_BANDPASS,
    },
    SupplementalIndicatorSeed {
        id: "chande",
        label: "Chande",
        category: "trend",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_CHANDE,
    },
    SupplementalIndicatorSeed {
        id: "chandelier_exit",
        label: "Chandelier Exit",
        category: "trend",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_LONG_SHORT_STOP,
        params: PARAM_CHANDELIER_EXIT,
    },
    SupplementalIndicatorSeed {
        id: "cksp",
        label: "CKSP",
        category: "trend",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_LONG_SHORT_VALUES,
        params: PARAM_CKSP,
    },
    SupplementalIndicatorSeed {
        id: "correlation_cycle",
        label: "Correlation Cycle",
        category: "cycle",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_CORRELATION_CYCLE,
        params: PARAM_CORRELATION_CYCLE,
    },
    SupplementalIndicatorSeed {
        id: "correl_hl",
        label: "Correl HL",
        category: "statistics",
        input_kind: IndicatorInputKind::HighLow,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_CORREL_HL,
    },
    SupplementalIndicatorSeed {
        id: "decycler",
        label: "Decycler",
        category: "trend",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_DECYCLER,
    },
    SupplementalIndicatorSeed {
        id: "damiani_volatmeter",
        label: "Damiani Volatmeter",
        category: "volatility",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VOL_ANTI,
        params: PARAM_DAMIANI_VOLATMETER,
    },
    SupplementalIndicatorSeed {
        id: "dvdiqqe",
        label: "DVDIQQE",
        category: "volume",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_DVDIQQE,
        params: PARAM_DVDIQQE,
    },
    SupplementalIndicatorSeed {
        id: "emd",
        label: "EMD",
        category: "volatility",
        input_kind: IndicatorInputKind::Ohlcv,
        outputs: OUTPUTS_UPPER_MIDDLE_LOWER_BAND,
        params: PARAM_EMD,
    },
    SupplementalIndicatorSeed {
        id: "eri",
        label: "ERI",
        category: "trend",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_BULL_BEAR,
        params: PARAM_ERI,
    },
    SupplementalIndicatorSeed {
        id: "fisher",
        label: "Fisher",
        category: "momentum",
        input_kind: IndicatorInputKind::HighLow,
        outputs: OUTPUTS_FISHER,
        params: PARAM_FISHER,
    },
    SupplementalIndicatorSeed {
        id: "fvg_trailing_stop",
        label: "FVG Trailing Stop",
        category: "trend",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_FVG_TS,
        params: PARAM_FVG_TRAILING_STOP,
    },
    SupplementalIndicatorSeed {
        id: "gatorosc",
        label: "Gator Oscillator",
        category: "trend",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_GATOROSC,
        params: PARAM_GATOROSC,
    },
    SupplementalIndicatorSeed {
        id: "halftrend",
        label: "HalfTrend",
        category: "trend",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_HALFTREND,
        params: PARAM_HALFTREND,
    },
    SupplementalIndicatorSeed {
        id: "kst",
        label: "KST",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_LINE_SIGNAL,
        params: PARAM_KST,
    },
    SupplementalIndicatorSeed {
        id: "kaufmanstop",
        label: "Kaufmanstop",
        category: "trend",
        input_kind: IndicatorInputKind::HighLow,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_KAUFMANSTOP,
    },
    SupplementalIndicatorSeed {
        id: "lpc",
        label: "LPC",
        category: "cycle",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_FILTER_BANDS,
        params: PARAM_LPC,
    },
    SupplementalIndicatorSeed {
        id: "mab",
        label: "MAB",
        category: "volatility",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_UPPER_MIDDLE_LOWER_BAND,
        params: PARAM_MAB,
    },
    SupplementalIndicatorSeed {
        id: "macz",
        label: "MACZ",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_MACZ,
    },
    SupplementalIndicatorSeed {
        id: "minmax",
        label: "MinMax",
        category: "pattern",
        input_kind: IndicatorInputKind::HighLow,
        outputs: OUTPUTS_MINMAX,
        params: PARAM_MINMAX,
    },
    SupplementalIndicatorSeed {
        id: "mod_god_mode",
        label: "Mod God Mode",
        category: "momentum",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_MOD_GOD_MODE,
        params: PARAM_MOD_GOD_MODE,
    },
    SupplementalIndicatorSeed {
        id: "pattern_recognition",
        label: "Pattern Recognition",
        category: "pattern",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_MATRIX_BOOL,
        params: PARAM_NONE,
    },
    SupplementalIndicatorSeed {
        id: "msw",
        label: "MSW",
        category: "cycle",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_SINE_LEAD,
        params: PARAM_MSW,
    },
    SupplementalIndicatorSeed {
        id: "nadaraya_watson_envelope",
        label: "Nadaraya Watson Envelope",
        category: "volatility",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_UPPER_LOWER,
        params: PARAM_NWE,
    },
    SupplementalIndicatorSeed {
        id: "ott",
        label: "OTT",
        category: "trend",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_OTT,
    },
    SupplementalIndicatorSeed {
        id: "otto",
        label: "OTTO",
        category: "trend",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_HOTT_LOTT,
        params: PARAM_OTTO,
    },
    SupplementalIndicatorSeed {
        id: "vidya",
        label: "VIDYA",
        category: "trend",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_VIDYA,
    },
    SupplementalIndicatorSeed {
        id: "vlma",
        label: "VLMA",
        category: "trend",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_VLMA,
    },
    SupplementalIndicatorSeed {
        id: "pma",
        label: "PMA",
        category: "trend",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_EHLERS_PMA,
        params: PARAM_PMA,
    },
    SupplementalIndicatorSeed {
        id: "prb",
        label: "PRB",
        category: "statistics",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_PRB,
        params: PARAM_PRB,
    },
    SupplementalIndicatorSeed {
        id: "qqe",
        label: "QQE",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: &[OUTPUT_FAST, OUTPUT_SLOW],
        params: PARAM_QQE,
    },
    SupplementalIndicatorSeed {
        id: "adaptive_bounds_rsi",
        label: "Adaptive Bounds RSI",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_ADAPTIVE_BOUNDS_RSI,
        params: PARAM_ADAPTIVE_BOUNDS_RSI,
    },
    SupplementalIndicatorSeed {
        id: "forward_backward_exponential_oscillator",
        label: "Forward-Backward Exponential Oscillator",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR,
        params: PARAM_FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR,
    },
    SupplementalIndicatorSeed {
        id: "qqe_weighted_oscillator",
        label: "QQE Weighted Oscillator",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_QQE_WEIGHTED_OSCILLATOR,
        params: PARAM_QQE_WEIGHTED_OSCILLATOR,
    },
    SupplementalIndicatorSeed {
        id: "market_structure_confluence",
        label: "Market Structure Confluence",
        category: "trend",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_MARKET_STRUCTURE_CONFLUENCE,
        params: PARAM_MARKET_STRUCTURE_CONFLUENCE,
    },
    SupplementalIndicatorSeed {
        id: "range_filtered_trend_signals",
        label: "Range Filtered Trend Signals",
        category: "trend",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_RANGE_FILTERED_TREND_SIGNALS,
        params: PARAM_RANGE_FILTERED_TREND_SIGNALS,
    },
    SupplementalIndicatorSeed {
        id: "range_oscillator",
        label: "Range Oscillator",
        category: "volatility",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_RANGE_OSCILLATOR,
        params: PARAM_RANGE_OSCILLATOR,
    },
    SupplementalIndicatorSeed {
        id: "volume_weighted_relative_strength_index",
        label: "Volume Weighted Relative Strength Index",
        category: "momentum",
        input_kind: IndicatorInputKind::CloseVolume,
        outputs: OUTPUTS_VOLUME_WEIGHTED_RELATIVE_STRENGTH_INDEX,
        params: PARAM_VOLUME_WEIGHTED_RELATIVE_STRENGTH_INDEX,
    },
    SupplementalIndicatorSeed {
        id: "range_filter",
        label: "Range Filter",
        category: "volatility",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_FILTER_BANDS,
        params: PARAM_RANGE_FILTER,
    },
    SupplementalIndicatorSeed {
        id: "coppock",
        label: "Coppock",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_COPPOCK,
    },
    SupplementalIndicatorSeed {
        id: "rsmk",
        label: "RSMK",
        category: "relative_strength",
        input_kind: IndicatorInputKind::CloseVolume,
        outputs: OUTPUTS_INDICATOR_SIGNAL,
        params: PARAM_RSMK,
    },
    SupplementalIndicatorSeed {
        id: "voss",
        label: "Voss",
        category: "cycle",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VOSS,
        params: PARAM_VOSS,
    },
    SupplementalIndicatorSeed {
        id: "stc",
        label: "STC",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_STC,
    },
    SupplementalIndicatorSeed {
        id: "rvi",
        label: "RVI",
        category: "volatility",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_RVI,
    },
    SupplementalIndicatorSeed {
        id: "safezonestop",
        label: "SafeZoneStop",
        category: "trend",
        input_kind: IndicatorInputKind::HighLow,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_SAFEZONESTOP,
    },
    SupplementalIndicatorSeed {
        id: "chop",
        label: "CHOP",
        category: "trend",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_CHOP,
    },
    SupplementalIndicatorSeed {
        id: "devstop",
        label: "DevStop",
        category: "trend",
        input_kind: IndicatorInputKind::HighLow,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_DEVSTOP,
    },
    SupplementalIndicatorSeed {
        id: "net_myrsi",
        label: "NET_MyRSI",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAMS_PERIOD_ONLY,
    },
    SupplementalIndicatorSeed {
        id: "wad",
        label: "WAD",
        category: "volume",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_NONE,
    },
    SupplementalIndicatorSeed {
        id: "pivot",
        label: "Pivot",
        category: "price",
        input_kind: IndicatorInputKind::Ohlc,
        outputs: OUTPUTS_PIVOT,
        params: PARAM_PIVOT,
    },
];

fn supplemental_supports_cpu_batch(id: &str) -> bool {
    matches!(
        id,
        "adx"
            | "adxr"
            | "atr"
            | "ad"
            | "adosc"
            | "ao"
            | "dti"
            | "dx"
            | "di"
            | "dm"
            | "vi"
            | "donchian"
            | "supertrend"
            | "adjustable_ma_alternating_extremities"
            | "supertrend_recovery"
            | "keltner"
            | "aroon"
            | "srsi"
            | "kdj"
            | "squeeze_momentum"
            | "wavetrend"
            | "wto"
            | "accumulation_swing_index"
            | "daily_factor"
            | "moving_average_cross_probability"
            | "bulls_v_bears"
            | "regression_slope_oscillator"
            | "smooth_theil_sen"
            | "market_structure_confluence"
            | "forward_backward_exponential_oscillator"
            | "qqe_weighted_oscillator"
            | "range_filtered_trend_signals"
            | "range_oscillator"
            | "volume_weighted_relative_strength_index"
            | "l2_ehlers_signal_to_noise"
            | "ehlers_smoothed_adaptive_momentum"
            | "ehlers_adaptive_cyber_cycle"
            | "ehlers_simple_cycle_indicator"
            | "l1_ehlers_phasor"
            | "andean_oscillator"
            | "cycle_channel_oscillator"
            | "ewma_volatility"
            | "ichimoku_oscillator"
            | "mesa_stochastic_multi_length"
            | "price_moving_average_ratio_percentile"
            | "relative_strength_index_wave_indicator"
            | "random_walk_index"
            | "spearman_correlation"
            | "trend_trigger_factor"
            | "volatility_quality_index"
            | "vwap_deviation_oscillator"
            | "yang_zhang_volatility"
            | "historical_volatility_percentile"
            | "ehlers_detrending_filter"
            | "keltner_channel_width_oscillator"
            | "market_meanness_index"
            | "price_density_market_noise"
            | "momentum_ratio_oscillator"
            | "hypertrend"
            | "adaptive_schaff_trend_cycle"
            | "smoothed_gaussian_trend_filter"
            | "logarithmic_moving_average"
            | "ict_propulsion_block"
            | "supertrend_oscillator"
            | "leavitt_convolution_acceleration"
            | "impulse_macd"
            | "insync_index"
            | "volatility_ratio_adaptive_rsx"
            | "on_balance_volume_oscillator"
            | "parkinson_volatility"
            | "psychological_line"
            | "rank_correlation_index"
            | "trend_follower"
            | "trend_direction_force_index"
            | "linear_regression_intensity"
            | "trend_continuation_factor"
            | "pretty_good_oscillator"
            | "twiggs_money_flow"
            | "volume_weighted_stochastic_rsi"
            | "stochastic_adaptive_d"
            | "stochastic_connors_rsi"
            | "volume_zone_oscillator"
            | "bop"
            | "emv"
            | "efi"
            | "mfi"
            | "mass"
            | "kvo"
            | "wad"
            | "vosc"
            | "rvi"
            | "coppock"
            | "rsi"
            | "roc"
            | "linear_correlation_oscillator"
            | "ehlers_fm_demodulator"
            | "ehlers_adaptive_cg"
            | "adaptive_momentum_oscillator"
            | "velocity"
            | "normalized_volume_true_range"
            | "exponential_trend"
            | "trend_flow_trail"
            | "range_breakout_signals"
            | "apo"
            | "cci"
            | "cci_cycle"
            | "cfo"
            | "cg"
            | "er"
            | "kurtosis"
            | "natr"
            | "net_myrsi"
            | "mean_ad"
            | "medium_ad"
            | "deviation"
            | "mod_god_mode"
            | "dpo"
            | "lrsi"
            | "fosc"
            | "ift_rsi"
            | "linearreg_angle"
            | "linearreg_intercept"
            | "linearreg_slope"
            | "pfe"
            | "percentile_nearest_rank"
            | "ui"
            | "zscore"
            | "medprice"
            | "midpoint"
            | "midprice"
            | "wclprice"
            | "obv"
            | "vpt"
            | "nvi"
            | "pvi"
            | "mom"
            | "cmo"
            | "rocp"
            | "rocr"
            | "tsf"
            | "adaptive_macd"
            | "polynomial_regression_extrapolation"
            | "statistical_trailing_stop"
            | "standardized_psar_oscillator"
            | "geometric_bias_oscillator"
            | "ppo"
            | "trix"
            | "tsi"
            | "stddev"
            | "var"
            | "willr"
            | "ultosc"
            | "macd"
            | "bollinger_bands"
            | "bollinger_bands_width"
            | "stoch"
            | "stochf"
            | "vwmacd"
            | "stc"
            | "vpci"
            | "ttm_trend"
            | "ttm_squeeze"
            | "acosc"
            | "alligator"
            | "alphatrend"
            | "aso"
            | "bandpass"
            | "chande"
            | "chandelier_exit"
            | "cksp"
            | "coppock"
            | "correl_hl"
            | "correlation_cycle"
            | "ehlers_adaptive_cg"
            | "normalized_volume_true_range"
            | "exponential_trend"
            | "trend_flow_trail"
            | "range_breakout_signals"
            | "chop"
            | "damiani_volatmeter"
            | "dvdiqqe"
            | "emd"
            | "eri"
            | "fisher"
            | "fvg_trailing_stop"
            | "gatorosc"
            | "halftrend"
            | "kst"
            | "lpc"
            | "mab"
            | "macz"
            | "minmax"
            | "msw"
            | "nadaraya_watson_envelope"
            | "otto"
            | "vidya"
            | "vlma"
            | "pma"
            | "prb"
            | "qqe"
            | "qqe_weighted_oscillator"
            | "range_filter"
            | "rsmk"
            | "safezonestop"
            | "devstop"
            | "voss"
            | "pivot"
    )
}

fn supplemental_supports_cuda_single(id: &str) -> bool {
    matches!(id, "pattern_recognition")
}

fn supplemental_supports_cuda_batch(id: &str) -> bool {
    matches!(
        id,
        "acosc"
            | "adosc"
            | "adx"
            | "adxr"
            | "alligator"
            | "alphatrend"
            | "ao"
            | "apo"
            | "aroon"
            | "aroonosc"
            | "aso"
            | "atr"
            | "avsl"
            | "bandpass"
            | "bollinger_bands"
            | "bollinger_bands_width"
            | "bop"
            | "cci"
            | "cci_cycle"
            | "cfo"
            | "cg"
            | "chande"
            | "chandelier_exit"
            | "chop"
            | "cksp"
            | "cmo"
            | "coppock"
            | "correl_hl"
            | "correlation_cycle"
            | "cvi"
            | "damiani_volatmeter"
            | "dec_osc"
            | "decycler"
            | "deviation"
            | "devstop"
            | "di"
            | "dm"
            | "donchian"
            | "dpo"
            | "dti"
            | "dvdiqqe"
            | "dx"
            | "efi"
            | "emd"
            | "emv"
            | "er"
            | "eri"
            | "fisher"
            | "fosc"
            | "fvg_trailing_stop"
            | "gatorosc"
            | "halftrend"
            | "ift_rsi"
            | "kaufmanstop"
            | "kdj"
            | "keltner"
            | "kst"
            | "kurtosis"
            | "kvo"
            | "linearreg_angle"
            | "linearreg_intercept"
            | "linearreg_slope"
            | "lpc"
            | "lrsi"
            | "mab"
            | "macd"
            | "macz"
            | "marketefi"
            | "mass"
            | "mean_ad"
            | "medium_ad"
            | "medprice"
            | "mfi"
            | "minmax"
            | "mod_god_mode"
            | "mom"
            | "msw"
            | "nadaraya_watson_envelope"
            | "natr"
            | "net_myrsi"
            | "nvi"
            | "obv"
            | "ott"
            | "otto"
            | "percentile_nearest_rank"
            | "pfe"
            | "pivot"
            | "pma"
            | "ppo"
            | "prb"
            | "pvi"
            | "qqe"
            | "qstick"
            | "range_filter"
            | "reverse_rsi"
            | "roc"
            | "rocp"
            | "rocr"
            | "rsi"
            | "rsmk"
            | "rsx"
            | "rvi"
            | "safezonestop"
            | "sar"
            | "squeeze_momentum"
            | "srsi"
            | "stc"
            | "stddev"
            | "stoch"
            | "stochf"
            | "supertrend"
            | "trix"
            | "tsf"
            | "tsi"
            | "ttm_squeeze"
            | "ttm_trend"
            | "ui"
            | "ultosc"
            | "var"
            | "vi"
            | "vidya"
            | "vlma"
            | "vosc"
            | "voss"
            | "vpci"
            | "vpt"
            | "vwmacd"
            | "wad"
            | "wavetrend"
            | "wclprice"
            | "willr"
            | "wto"
            | "ewma_volatility"
            | "yang_zhang_volatility"
            | "parkinson_volatility"
            | "zscore"
    )
}

fn supplemental_supports_cuda_vram(id: &str) -> bool {
    matches!(id, "pattern_recognition") || supplemental_supports_cuda_batch(id)
}

pub fn is_bucket_b_indicator(id: &str) -> bool {
    BUCKET_B_INDICATORS
        .iter()
        .any(|item| item.eq_ignore_ascii_case(id))
}

static INDICATOR_REGISTRY: Lazy<Vec<IndicatorInfo>> = Lazy::new(build_registry);
static INDICATOR_EXACT_INDEX: Lazy<HashMap<&'static str, usize>> = Lazy::new(|| {
    let mut map = HashMap::with_capacity(INDICATOR_REGISTRY.len());
    for (idx, info) in INDICATOR_REGISTRY.iter().enumerate() {
        map.insert(info.id, idx);
    }
    map
});

fn ma_outputs_for(ma_id: &str) -> Vec<IndicatorOutputInfo> {
    match ma_id {
        "mama" => OUTPUTS_MAMA.to_vec(),
        "ehlers_pma" => OUTPUTS_EHLERS_PMA.to_vec(),
        "ehlers_undersampled_double_moving_average" => OUTPUTS_BUFF_AVERAGES.to_vec(),
        "buff_averages" => OUTPUTS_BUFF_AVERAGES.to_vec(),
        "ema_deviation_corrected_t3" => OUTPUTS_EDCT3.to_vec(),
        _ => OUTPUTS_VALUE_F64.to_vec(),
    }
}

fn ma_params_for(ma_id: &str, period_based: bool) -> Vec<IndicatorParamInfo> {
    let mut params = Vec::new();
    if period_based {
        params.push(PARAM_PERIOD);
    }
    for item in ma_param_schema(ma_id).iter() {
        let kind = match item.kind {
            MaParamKind::Float => IndicatorParamKind::Float,
            MaParamKind::Int => IndicatorParamKind::Int,
        };
        let default = match kind {
            IndicatorParamKind::Float => Some(ParamValueStatic::Float(item.default)),
            IndicatorParamKind::Int => Some(ParamValueStatic::Int(item.default as i64)),
            IndicatorParamKind::Bool | IndicatorParamKind::EnumString => None,
        };
        params.push(IndicatorParamInfo {
            key: item.key,
            label: item.label,
            kind,
            required: false,
            default,
            min: item.min,
            max: item.max,
            step: item.step,
            enum_values: EMPTY_ENUM_VALUES,
            notes: item.notes,
        });
    }
    match ma_id {
        "mama" => params.push(PARAM_OUTPUT_MAMA),
        "ehlers_pma" => params.push(PARAM_OUTPUT_EHLERS_PMA),
        "ehlers_undersampled_double_moving_average" => params.push(PARAM_OUTPUT_BUFF_AVERAGES),
        "buff_averages" => params.push(PARAM_OUTPUT_BUFF_AVERAGES),
        "ema_deviation_corrected_t3" => params.push(PARAM_OUTPUT_EDCT3),
        "vwap" => params.push(PARAM_ANCHOR),
        "volume_adjusted_ma" => params.push(PARAM_STRICT),
        _ => {}
    }
    params
}

fn build_registry() -> Vec<IndicatorInfo> {
    let mut out = Vec::new();

    for ma in list_moving_averages().iter() {
        out.push(IndicatorInfo {
            id: ma.id,
            label: ma.label,
            category: "moving_averages",
            dynamic_strategy_eligible: true,
            input_kind: if ma.requires_candles {
                IndicatorInputKind::Candles
            } else {
                IndicatorInputKind::Slice
            },
            outputs: ma_outputs_for(ma.id),
            params: ma_params_for(ma.id, ma.period_based),
            capabilities: IndicatorCapabilities {
                supports_cpu_single: ma.supports_cpu_single,
                supports_cpu_batch: ma.supports_cpu_batch,
                supports_cuda_single: ma.supports_cuda_single,
                supports_cuda_batch: ma.supports_cuda_sweep,
                supports_cuda_vram: ma.supports_cuda_sweep,
            },
            notes: ma.notes,
        });
    }

    for seed in SUPPLEMENTAL_INDICATORS.iter() {
        let info = IndicatorInfo {
            id: seed.id,
            label: seed.label,
            category: seed.category,
            dynamic_strategy_eligible: true,
            input_kind: seed.input_kind,
            outputs: seed.outputs.to_vec(),
            params: seed.params.to_vec(),
            capabilities: IndicatorCapabilities {
                supports_cpu_single: true,
                supports_cpu_batch: supplemental_supports_cpu_batch(seed.id),
                supports_cuda_single: supplemental_supports_cuda_single(seed.id),
                supports_cuda_batch: supplemental_supports_cuda_batch(seed.id),
                supports_cuda_vram: supplemental_supports_cuda_vram(seed.id),
            },
            notes: Some(SUPPLEMENTAL_SEED_NOTE),
        };

        if let Some(existing) = out
            .iter_mut()
            .find(|item| item.id.eq_ignore_ascii_case(seed.id))
        {
            *existing = info;
        } else {
            out.push(info);
        }
    }

    out.sort_by(|a, b| a.id.cmp(b.id));
    out
}

pub fn list_indicators() -> &'static [IndicatorInfo] {
    INDICATOR_REGISTRY.as_slice()
}

pub fn get_indicator(id: &str) -> Option<&'static IndicatorInfo> {
    let indicators = list_indicators();
    if let Some(idx) = INDICATOR_EXACT_INDEX.get(id).copied() {
        return Some(&indicators[idx]);
    }
    if let Ok(idx) = indicators.binary_search_by(|info| info.id.cmp(id)) {
        return Some(&indicators[idx]);
    }
    indicators
        .iter()
        .find(|info| info.id.eq_ignore_ascii_case(id))
}

pub fn indicator_param_schema(id: &str) -> Option<&'static [IndicatorParamInfo]> {
    get_indicator(id).map(|info| info.params.as_slice())
}

pub fn indicator_output_schema(id: &str) -> Option<&'static [IndicatorOutputInfo]> {
    get_indicator(id).map(|info| info.outputs.as_slice())
}

pub fn indicator_capabilities(id: &str) -> Option<IndicatorCapabilities> {
    get_indicator(id).map(|info| info.capabilities)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_is_non_empty() {
        assert!(!list_indicators().is_empty());
    }

    #[test]
    fn ids_are_unique_case_insensitive() {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        for info in list_indicators().iter() {
            let lower = info.id.to_ascii_lowercase();
            assert!(seen.insert(lower), "duplicate id {}", info.id);
        }
    }

    #[test]
    fn all_registered_entries_have_output_schema() {
        for info in list_indicators().iter() {
            assert!(
                !info.outputs.is_empty(),
                "indicator {} has no output schema",
                info.id
            );
        }
    }

    #[test]
    fn ma_registry_is_mirrored() {
        for ma in list_moving_averages().iter() {
            assert!(
                get_indicator(ma.id).is_some(),
                "missing moving average {} in global registry",
                ma.id
            );
        }
    }

    #[test]
    fn lookup_is_case_insensitive() {
        assert!(get_indicator("SMA").is_some());
        assert!(get_indicator("sma").is_some());
    }

    #[test]
    fn schema_accessors_work() {
        assert!(indicator_output_schema("macd").is_some());
        assert!(indicator_param_schema("sma").is_some());
        assert!(indicator_capabilities("sma").is_some());
        assert!(indicator_output_schema("not_real").is_none());
    }

    #[test]
    fn pattern_recognition_capability_is_registered_as_non_batch() {
        let info = get_indicator("pattern_recognition").unwrap();
        assert_eq!(info.input_kind, IndicatorInputKind::Ohlc);
        assert_eq!(info.outputs.len(), 1);
        assert_eq!(info.outputs[0].id, "matrix");
        assert!(info.capabilities.supports_cpu_single);
        assert!(!info.capabilities.supports_cpu_batch);
        assert!(info.capabilities.supports_cuda_single);
        assert!(!info.capabilities.supports_cuda_batch);
        assert!(info.capabilities.supports_cuda_vram);
    }

    #[test]
    fn dec_osc_and_rsx_are_registered_for_cuda_batch() {
        let dec_osc = get_indicator("dec_osc").unwrap();
        assert_eq!(dec_osc.input_kind, IndicatorInputKind::Slice);
        assert_eq!(dec_osc.outputs.len(), 1);
        assert_eq!(dec_osc.outputs[0].id, "value");
        assert!(dec_osc.capabilities.supports_cuda_batch);
        assert!(dec_osc.capabilities.supports_cuda_vram);

        let rsx = get_indicator("rsx").unwrap();
        assert_eq!(rsx.input_kind, IndicatorInputKind::Slice);
        assert_eq!(rsx.outputs.len(), 1);
        assert_eq!(rsx.outputs[0].id, "value");
        assert!(rsx.capabilities.supports_cuda_batch);
        assert!(rsx.capabilities.supports_cuda_vram);
    }

    #[test]
    fn chande_is_registered_for_cpu_and_cuda_batch() {
        assert!(supplemental_supports_cpu_batch("chande"));
        assert!(supplemental_supports_cuda_batch("chande"));
        let chande = get_indicator("chande").unwrap();
        assert_eq!(chande.input_kind, IndicatorInputKind::Ohlc);
        assert_eq!(chande.outputs.len(), 1);
        assert_eq!(chande.outputs[0].id, "value");
        assert!(chande.capabilities.supports_cpu_batch);
        assert!(chande.capabilities.supports_cuda_batch);
        assert!(chande.capabilities.supports_cuda_vram);
    }

    #[test]
    fn bucket_b_ma_capabilities_follow_ma_registry() {
        let mama = indicator_capabilities("mama").unwrap();
        assert!(mama.supports_cpu_batch);
        assert!(mama.supports_cuda_batch);
        assert!(mama.supports_cuda_vram);

        let vwap = indicator_capabilities("vwap").unwrap();
        assert!(vwap.supports_cpu_batch);
        assert!(vwap.supports_cuda_batch);
        assert!(vwap.supports_cuda_vram);
    }

    #[test]
    fn bucket_membership_lookup_is_case_insensitive() {
        assert!(is_bucket_b_indicator("MAMA"));
        assert!(is_bucket_b_indicator("pivot"));
        assert!(!is_bucket_b_indicator("sma"));
        assert!(!is_bucket_b_indicator("adx"));
    }

    #[test]
    fn lrsi_is_registered_for_cuda_batch() {
        let lrsi = get_indicator("lrsi").unwrap();
        assert_eq!(lrsi.input_kind, IndicatorInputKind::HighLow);
        assert_eq!(lrsi.outputs.len(), 1);
        assert_eq!(lrsi.outputs[0].id, "value");
        assert!(lrsi.capabilities.supports_cpu_batch);
        assert!(lrsi.capabilities.supports_cuda_batch);
        assert!(lrsi.capabilities.supports_cuda_vram);
    }
}
