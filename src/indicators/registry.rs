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
    "alligator",
    "alphatrend",
    "aroon",
    "aso",
    "bandpass",
    "chandelier_exit",
    "cksp",
    "correlation_cycle",
    "damiani_volatmeter",
    "di",
    "dm",
    "donchian",
    "dvdiqqe",
    "emd",
    "eri",
    "fisher",
    "fvg_trailing_stop",
    "gatorosc",
    "halftrend",
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
    "range_filter",
    "rsmk",
    "squeeze_momentum",
    "srsi",
    "supertrend",
    "vi",
    "voss",
    "wavetrend",
    "wto",
    "ehlers_pma",
    "buff_averages",
    "vwap",
    "pivot",
];

const EMPTY_ENUM_VALUES: &[&str] = &[];
const ENUM_VALUES_TRUE_FALSE: &[&str] = &["true", "false"];
const ENUM_VALUES_MA_OUTPUT: &[&str] = &["mama", "fama"];
const ENUM_VALUES_PMA_OUTPUT: &[&str] = &["predict", "trigger"];
const ENUM_VALUES_BUFF_OUTPUT: &[&str] = &["fast", "slow"];

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
const OUTPUT_HISTOGRAM: IndicatorOutputInfo = IndicatorOutputInfo {
    id: "histogram",
    label: "Histogram",
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
const OUTPUTS_BUFF_AVERAGES: &[IndicatorOutputInfo] = &[OUTPUT_FAST, OUTPUT_SLOW];
const OUTPUTS_PLUS_MINUS: &[IndicatorOutputInfo] = &[OUTPUT_PLUS, OUTPUT_MINUS];
const OUTPUTS_UP_DOWN: &[IndicatorOutputInfo] = &[OUTPUT_UP, OUTPUT_DOWN];
const OUTPUTS_TREND_CHANGED: &[IndicatorOutputInfo] = &[OUTPUT_TREND, OUTPUT_CHANGED];
const OUTPUTS_KDJ: &[IndicatorOutputInfo] = &[OUTPUT_K, OUTPUT_D, OUTPUT_J];
const OUTPUTS_SQUEEZE_MOMENTUM: &[IndicatorOutputInfo] =
    &[OUTPUT_MOMENTUM, OUTPUT_SQUEEZE, OUTPUT_MOMENTUM_SIGNAL];
const OUTPUTS_WTO: &[IndicatorOutputInfo] =
    &[OUTPUT_WAVETREND1, OUTPUT_WAVETREND2, OUTPUT_HISTOGRAM];
const OUTPUTS_WAVETREND: &[IndicatorOutputInfo] = &[OUTPUT_WT1, OUTPUT_WT2, OUTPUT_WT_DIFF];
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
        id: "srsi",
        label: "Stochastic RSI",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_STOCH,
        params: PARAM_SRSI,
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
        id: "roc",
        label: "ROC",
        category: "momentum",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_VALUE_F64,
        params: PARAM_ROC_PERIOD,
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
        id: "bandpass",
        label: "BandPass",
        category: "cycle",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_BANDPASS,
        params: PARAM_BANDPASS,
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
        id: "otto",
        label: "OTTO",
        category: "trend",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_HOTT_LOTT,
        params: PARAM_OTTO,
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
        id: "range_filter",
        label: "Range Filter",
        category: "volatility",
        input_kind: IndicatorInputKind::Slice,
        outputs: OUTPUTS_FILTER_BANDS,
        params: PARAM_RANGE_FILTER,
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
            | "atr"
            | "ad"
            | "adosc"
            | "ao"
            | "dx"
            | "di"
            | "dm"
            | "vi"
            | "donchian"
            | "supertrend"
            | "keltner"
            | "aroon"
            | "srsi"
            | "kdj"
            | "squeeze_momentum"
            | "wavetrend"
            | "wto"
            | "bop"
            | "emv"
            | "efi"
            | "mfi"
            | "mass"
            | "kvo"
            | "vosc"
            | "rsi"
            | "roc"
            | "apo"
            | "cci"
            | "cfo"
            | "cg"
            | "er"
            | "kurtosis"
            | "natr"
            | "mean_ad"
            | "medium_ad"
            | "deviation"
            | "dpo"
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
            | "ppo"
            | "trix"
            | "tsi"
            | "stddev"
            | "var"
            | "willr"
            | "ultosc"
            | "macd"
            | "bollinger_bands"
            | "stoch"
            | "stochf"
            | "vwmacd"
            | "vpci"
            | "ttm_trend"
            | "ttm_squeeze"
            | "acosc"
            | "alligator"
            | "alphatrend"
            | "aso"
            | "bandpass"
            | "chandelier_exit"
            | "cksp"
            | "correlation_cycle"
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
            | "pma"
            | "prb"
            | "qqe"
            | "range_filter"
            | "rsmk"
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
            "acosc" | "adosc" | "adx" | "adxr" | "alligator" | "alphatrend" | "ao" | "apo" |
            "aroon" | "aroonosc" | "aso" | "atr" | "avsl" | "bandpass" | "bollinger_bands" |
            "bollinger_bands_width" | "bop" | "cci" | "cci_cycle" | "cfo" | "cg" | "chande" |
            "chandelier_exit" | "chop" | "cksp" | "cmo" | "coppock" | "correl_hl" |
            "correlation_cycle" | "cvi" | "damiani_volatmeter" | "dec_osc" | "decycler" |
            "deviation" | "devstop" | "di" | "dm" | "donchian" | "dpo" | "dti" | "dvdiqqe" | "dx" |
            "efi" | "emd" | "emv" | "er" | "eri" | "fisher" | "fosc" | "fvg_trailing_stop" |
            "gatorosc" | "halftrend" | "ift_rsi" | "kaufmanstop" | "kdj" | "keltner" | "kst" |
            "kurtosis" | "kvo" | "linearreg_angle" | "linearreg_intercept" | "linearreg_slope" |
            "lpc" | "lrsi" | "mab" | "macd" | "macz" | "marketefi" | "mass" | "mean_ad" |
            "medium_ad" | "medprice" | "mfi" | "minmax" | "mod_god_mode" | "mom" | "msw" |
            "nadaraya_watson_envelope" | "natr" | "net_myrsi" | "nvi" | "obv" | "ott" | "otto" |
            "percentile_nearest_rank" | "pfe" | "pivot" | "pma" | "ppo" | "prb" | "pvi" | "qqe" |
            "qstick" | "range_filter" | "reverse_rsi" | "roc" | "rocp" | "rocr" | "rsi" | "rsmk" |
            "rsx" | "rvi" | "safezonestop" | "sar" | "squeeze_momentum" | "srsi" | "stc" |
            "stddev" | "stoch" | "stochf" | "supertrend" | "trix" | "tsf" | "tsi" | "ttm_squeeze" |
            "ttm_trend" | "ui" | "ultosc" | "var" | "vi" | "vidya" | "vlma" | "vosc" | "voss" |
            "vpci" | "vpt" | "vwmacd" | "wad" | "wavetrend" | "wclprice" | "willr" | "wto" |
            "zscore"
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
        "buff_averages" => OUTPUTS_BUFF_AVERAGES.to_vec(),
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
        "buff_averages" => params.push(PARAM_OUTPUT_BUFF_AVERAGES),
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
        out.push(IndicatorInfo {
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
        });
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
}
