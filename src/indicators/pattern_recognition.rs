use crate::utilities::data_loader::Candles;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::export_u64_cuda_dlpack_2d;
#[cfg(all(feature = "python", feature = "cuda"))]
pub use crate::utilities::dlpack_cuda::DeviceArrayF32Py;
use crate::utilities::enums::Kernel;
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::memory::{CopyDestination, DeviceBuffer};
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArrayMethods};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};
use std::mem::MaybeUninit;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;
use thiserror::Error;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone)]
pub enum PatternData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

impl Default for PatternType {
    fn default() -> Self {
        PatternType::Cdl2Crows
    }
}

#[derive(Debug, Clone, Default)]
pub struct PatternParams {
    pub pattern_type: PatternType,
    pub penetration: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternType {
    Cdl2Crows,
    Cdl3BlackCrows,
    Cdl3Inside,
    Cdl3LineStrike,
    Cdl3Outside,
    Cdl3StarsInSouth,
    Cdl3WhiteSoldiers,
    CdlAbandonedBaby,
    CdlAdvanceBlock,
    CdlBeltHold,
    CdlBreakaway,
    CdlClosingMarubozu,
    CdlConcealBabySwall,
    CdlCounterAttack,
    CdlDarkCloudCover,
    CdlDoji,
    CdlDojiStar,
    CdlDragonflyDoji,
    CdlEngulfing,
    CdlEveningDojiStar,
    CdlEveningStar,
    CdlGapSideSideWhite,
    CdlGravestoneDoji,
    CdlHammer,
    CdlHangingMan,
    CdlHarami,
    CdlHaramiCross,
    CdlHighWave,
    CdlHikkake,
    CdlHikkakeMod,
    CdlHomingPigeon,
    CdlIdentical3Crows,
    CdlInNeck,
    CdlInvertedHammer,
    CdlKicking,
    CdlKickingByLength,
    CdlLadderBottom,
    CdlLongLeggedDoji,
    CdlLongLine,
    CdlMarubozu,
    CdlMatchingLow,
    CdlMatHold,
    CdlMorningDojiStar,
    CdlMorningStar,
    CdlOnNeck,
    CdlPiercing,
    CdlRickshawMan,
    CdlRiseFall3Methods,
    CdlSeparatingLines,
    CdlShootingStar,
    CdlShortLine,
    CdlSpinningTop,
    CdlStalledPattern,
    CdlStickSandwich,
    CdlTakuri,
    CdlTasukiGap,
    CdlThrusting,
    CdlTristar,
    CdlUnique3River,
    CdlUpsideGap2Crows,
    CdlXSideGap3Methods,
}

#[derive(Debug, Clone)]
pub struct PatternInput<'a> {
    pub data: PatternData<'a>,
    pub params: PatternParams,
}

impl<'a> PatternInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: PatternParams) -> Self {
        Self {
            data: PatternData::Candles { candles },
            params,
        }
    }

    pub fn from_slices(
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: PatternParams,
    ) -> Self {
        Self {
            data: PatternData::Slices {
                open,
                high,
                low,
                close,
            },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles, pattern_type: PatternType) -> Self {
        Self {
            data: PatternData::Candles { candles },
            params: PatternParams {
                pattern_type,
                ..Default::default()
            },
        }
    }

    pub fn with_default_slices(
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        pattern_type: PatternType,
    ) -> Self {
        Self {
            data: PatternData::Slices {
                open,
                high,
                low,
                close,
            },
            params: PatternParams {
                pattern_type,
                ..Default::default()
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct PatternOutput {
    pub values: Vec<i8>,
}

#[derive(Debug, Clone, Default)]
pub struct PatternRecognitionParams {}

#[derive(Debug, Clone)]
pub enum PatternRecognitionData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct PatternRecognitionInput<'a> {
    pub data: PatternRecognitionData<'a>,
    pub params: PatternRecognitionParams,
}

impl<'a> PatternRecognitionInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: PatternRecognitionParams) -> Self {
        Self {
            data: PatternRecognitionData::Candles { candles },
            params,
        }
    }

    pub fn from_slices(
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: PatternRecognitionParams,
    ) -> Self {
        Self {
            data: PatternRecognitionData::Slices {
                open,
                high,
                low,
                close,
            },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, PatternRecognitionParams::default())
    }

    pub fn with_default_slices(
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    ) -> Self {
        Self::from_slices(open, high, low, close, PatternRecognitionParams::default())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PatternSpec {
    pub id: &'static str,
    pub row_index: usize,
    pub category: &'static str,
}

#[derive(Debug, Clone)]
pub struct PatternRecognitionOutput {
    pub rows: usize,
    pub cols: usize,
    pub values_u8: Vec<u8>,
    pub pattern_ids: Vec<&'static str>,
    pub warmup: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedPatternRecognitionOutput {
    pub rows: usize,
    pub cols: usize,
    pub words_per_row: usize,
    pub words_u64: Vec<u64>,
    pub pattern_ids: Vec<&'static str>,
    pub warmup: Option<usize>,
}

#[derive(Debug, Clone)]
struct SharedPatternPrimitives {
    body: Vec<f64>,
    range: Vec<f64>,
    upper_shadow: Vec<f64>,
    lower_shadow: Vec<f64>,
    direction: Vec<i8>,
    body_gap_up: Vec<u8>,
    body_gap_down: Vec<u8>,
}

fn build_shared_primitives(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> SharedPatternPrimitives {
    let len = close.len();
    let mut body = Vec::with_capacity(len);
    let mut range = Vec::with_capacity(len);
    let mut upper_shadow = Vec::with_capacity(len);
    let mut lower_shadow = Vec::with_capacity(len);
    let mut direction = Vec::with_capacity(len);
    let mut body_gap_up = Vec::with_capacity(len);
    let mut body_gap_down = Vec::with_capacity(len);

    for i in 0..len {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];

        body.push((c - o).abs());
        range.push(h - l);
        upper_shadow.push(if c >= o { h - c } else { h - o });
        lower_shadow.push(if c >= o { o - l } else { c - l });
        direction.push(if c >= o { 1 } else { -1 });

        if i == 0 {
            body_gap_up.push(0);
            body_gap_down.push(0);
        } else {
            let cur_min = o.min(c);
            let cur_max = o.max(c);
            let prev_min = open[i - 1].min(close[i - 1]);
            let prev_max = open[i - 1].max(close[i - 1]);
            body_gap_up.push((cur_min > prev_max) as u8);
            body_gap_down.push((cur_max < prev_min) as u8);
        }
    }

    SharedPatternPrimitives {
        body,
        range,
        upper_shadow,
        lower_shadow,
        direction,
        body_gap_up,
        body_gap_down,
    }
}

impl PatternRecognitionOutput {
    pub fn to_bitmask_u64(&self) -> PackedPatternRecognitionOutput {
        let words_per_row = self.cols.div_ceil(64);
        let mut words_u64 = vec![0u64; self.rows.saturating_mul(words_per_row)];

        for row in 0..self.rows {
            let src_start = row * self.cols;
            let src = &self.values_u8[src_start..src_start + self.cols];
            let dst_start = row * words_per_row;
            let dst = &mut words_u64[dst_start..dst_start + words_per_row];

            for (idx, v) in src.iter().enumerate() {
                if *v != 0 {
                    let word = idx / 64;
                    let bit = idx % 64;
                    dst[word] |= 1u64 << bit;
                }
            }
        }

        PackedPatternRecognitionOutput {
            rows: self.rows,
            cols: self.cols,
            words_per_row,
            words_u64,
            pattern_ids: self.pattern_ids.clone(),
            warmup: self.warmup,
        }
    }
}

#[derive(Debug, Error)]
pub enum PatternRecognitionError {
    #[error(
        "pattern_recognition: data length mismatch: open={open}, high={high}, low={low}, close={close}"
    )]
    DataLengthMismatch {
        open: usize,
        high: usize,
        low: usize,
        close: usize,
    },

    #[error("pattern_recognition: output length mismatch for `{pattern_id}`: expected {expected}, got {got}")]
    OutputLengthMismatch {
        pattern_id: &'static str,
        expected: usize,
        got: usize,
    },

    #[error(transparent)]
    Pattern(#[from] PatternError),
}

#[derive(Clone, Copy)]
struct PatternRunner {
    id: &'static str,
    pattern_type: PatternType,
    category: &'static str,
    run: fn(&PatternInput<'_>) -> Result<PatternOutput, PatternError>,
}

const PATTERN_RUNNERS: [PatternRunner; 61] = [
    PatternRunner {
        id: "cdl2crows",
        pattern_type: PatternType::Cdl2Crows,
        category: "candlestick",
        run: cdl2crows,
    },
    PatternRunner {
        id: "cdl3blackcrows",
        pattern_type: PatternType::Cdl3BlackCrows,
        category: "candlestick",
        run: cdl3blackcrows,
    },
    PatternRunner {
        id: "cdl3inside",
        pattern_type: PatternType::Cdl3Inside,
        category: "candlestick",
        run: cdl3inside,
    },
    PatternRunner {
        id: "cdl3linestrike",
        pattern_type: PatternType::Cdl3LineStrike,
        category: "candlestick",
        run: cdl3linestrike,
    },
    PatternRunner {
        id: "cdl3outside",
        pattern_type: PatternType::Cdl3Outside,
        category: "candlestick",
        run: cdl3outside,
    },
    PatternRunner {
        id: "cdl3starsinsouth",
        pattern_type: PatternType::Cdl3StarsInSouth,
        category: "candlestick",
        run: cdl3starsinsouth,
    },
    PatternRunner {
        id: "cdl3whitesoldiers",
        pattern_type: PatternType::Cdl3WhiteSoldiers,
        category: "candlestick",
        run: cdl3whitesoldiers,
    },
    PatternRunner {
        id: "cdlabandonedbaby",
        pattern_type: PatternType::CdlAbandonedBaby,
        category: "candlestick",
        run: cdlabandonedbaby,
    },
    PatternRunner {
        id: "cdladvanceblock",
        pattern_type: PatternType::CdlAdvanceBlock,
        category: "candlestick",
        run: cdladvanceblock,
    },
    PatternRunner {
        id: "cdlbelthold",
        pattern_type: PatternType::CdlBeltHold,
        category: "candlestick",
        run: cdlbelthold,
    },
    PatternRunner {
        id: "cdlbreakaway",
        pattern_type: PatternType::CdlBreakaway,
        category: "candlestick",
        run: cdlbreakaway,
    },
    PatternRunner {
        id: "cdlclosingmarubozu",
        pattern_type: PatternType::CdlClosingMarubozu,
        category: "candlestick",
        run: cdlclosingmarubozu,
    },
    PatternRunner {
        id: "cdlconcealbabyswall",
        pattern_type: PatternType::CdlConcealBabySwall,
        category: "candlestick",
        run: cdlconcealbabyswall,
    },
    PatternRunner {
        id: "cdlcounterattack",
        pattern_type: PatternType::CdlCounterAttack,
        category: "candlestick",
        run: cdlcounterattack,
    },
    PatternRunner {
        id: "cdldarkcloudcover",
        pattern_type: PatternType::CdlDarkCloudCover,
        category: "candlestick",
        run: cdldarkcloudcover,
    },
    PatternRunner {
        id: "cdldoji",
        pattern_type: PatternType::CdlDoji,
        category: "candlestick",
        run: cdldoji,
    },
    PatternRunner {
        id: "cdldojistar",
        pattern_type: PatternType::CdlDojiStar,
        category: "candlestick",
        run: cdldojistar,
    },
    PatternRunner {
        id: "cdldragonflydoji",
        pattern_type: PatternType::CdlDragonflyDoji,
        category: "candlestick",
        run: cdldragonflydoji,
    },
    PatternRunner {
        id: "cdlengulfing",
        pattern_type: PatternType::CdlEngulfing,
        category: "candlestick",
        run: cdlengulfing,
    },
    PatternRunner {
        id: "cdleveningdojistar",
        pattern_type: PatternType::CdlEveningDojiStar,
        category: "candlestick",
        run: cdleveningdojistar,
    },
    PatternRunner {
        id: "cdleveningstar",
        pattern_type: PatternType::CdlEveningStar,
        category: "candlestick",
        run: cdleveningstar,
    },
    PatternRunner {
        id: "cdlmorningstar",
        pattern_type: PatternType::CdlMorningStar,
        category: "candlestick",
        run: cdlmorningstar,
    },
    PatternRunner {
        id: "cdlgravestonedoji",
        pattern_type: PatternType::CdlGravestoneDoji,
        category: "candlestick",
        run: cdlgravestonedoji,
    },
    PatternRunner {
        id: "cdlhammer",
        pattern_type: PatternType::CdlHammer,
        category: "candlestick",
        run: cdlhammer,
    },
    PatternRunner {
        id: "cdlhangingman",
        pattern_type: PatternType::CdlHangingMan,
        category: "candlestick",
        run: cdlhangingman,
    },
    PatternRunner {
        id: "cdlharami",
        pattern_type: PatternType::CdlHarami,
        category: "candlestick",
        run: cdlharami,
    },
    PatternRunner {
        id: "cdlharamicross",
        pattern_type: PatternType::CdlHaramiCross,
        category: "candlestick",
        run: cdlharamicross,
    },
    PatternRunner {
        id: "cdlhighwave",
        pattern_type: PatternType::CdlHighWave,
        category: "candlestick",
        run: cdlhighwave,
    },
    PatternRunner {
        id: "cdlinvertedhammer",
        pattern_type: PatternType::CdlInvertedHammer,
        category: "candlestick",
        run: cdlinvertedhammer,
    },
    PatternRunner {
        id: "cdllongleggeddoji",
        pattern_type: PatternType::CdlLongLeggedDoji,
        category: "candlestick",
        run: cdllongleggeddoji,
    },
    PatternRunner {
        id: "cdllongline",
        pattern_type: PatternType::CdlLongLine,
        category: "candlestick",
        run: cdllongline,
    },
    PatternRunner {
        id: "cdlmarubozu",
        pattern_type: PatternType::CdlMarubozu,
        category: "candlestick",
        run: cdlmarubozu,
    },
    PatternRunner {
        id: "cdlrickshawman",
        pattern_type: PatternType::CdlRickshawMan,
        category: "candlestick",
        run: cdlrickshawman,
    },
    PatternRunner {
        id: "cdlshootingstar",
        pattern_type: PatternType::CdlShootingStar,
        category: "candlestick",
        run: cdlshootingstar,
    },
    PatternRunner {
        id: "cdlshortline",
        pattern_type: PatternType::CdlShortLine,
        category: "candlestick",
        run: cdlshortline,
    },
    PatternRunner {
        id: "cdlspinningtop",
        pattern_type: PatternType::CdlSpinningTop,
        category: "candlestick",
        run: cdlspinningtop,
    },
    PatternRunner {
        id: "cdltakuri",
        pattern_type: PatternType::CdlTakuri,
        category: "candlestick",
        run: cdltakuri,
    },
    PatternRunner {
        id: "cdlhomingpigeon",
        pattern_type: PatternType::CdlHomingPigeon,
        category: "candlestick",
        run: cdlhomingpigeon,
    },
    PatternRunner {
        id: "cdlmatchinglow",
        pattern_type: PatternType::CdlMatchingLow,
        category: "candlestick",
        run: cdlmatchinglow,
    },
    PatternRunner {
        id: "cdlinneck",
        pattern_type: PatternType::CdlInNeck,
        category: "candlestick",
        run: cdlinneck,
    },
    PatternRunner {
        id: "cdlonneck",
        pattern_type: PatternType::CdlOnNeck,
        category: "candlestick",
        run: cdlonneck,
    },
    PatternRunner {
        id: "cdlpiercing",
        pattern_type: PatternType::CdlPiercing,
        category: "candlestick",
        run: cdlpiercing,
    },
    PatternRunner {
        id: "cdlthrusting",
        pattern_type: PatternType::CdlThrusting,
        category: "candlestick",
        run: cdlthrusting,
    },
    PatternRunner {
        id: "cdlmorningdojistar",
        pattern_type: PatternType::CdlMorningDojiStar,
        category: "candlestick",
        run: cdlmorningdojistar,
    },
    PatternRunner {
        id: "cdltristar",
        pattern_type: PatternType::CdlTristar,
        category: "candlestick",
        run: cdltristar,
    },
    PatternRunner {
        id: "cdlidentical3crows",
        pattern_type: PatternType::CdlIdentical3Crows,
        category: "candlestick",
        run: cdlidentical3crows,
    },
    PatternRunner {
        id: "cdlsticksandwich",
        pattern_type: PatternType::CdlStickSandwich,
        category: "candlestick",
        run: cdlsticksandwich,
    },
    PatternRunner {
        id: "cdlseparatinglines",
        pattern_type: PatternType::CdlSeparatingLines,
        category: "candlestick",
        run: cdlseparatinglines,
    },
    PatternRunner {
        id: "cdlgapsidesidewhite",
        pattern_type: PatternType::CdlGapSideSideWhite,
        category: "candlestick",
        run: cdlgapsidesidewhite,
    },
    PatternRunner {
        id: "cdlhikkake",
        pattern_type: PatternType::CdlHikkake,
        category: "candlestick",
        run: cdlhikkake,
    },
    PatternRunner {
        id: "cdlhikkakemod",
        pattern_type: PatternType::CdlHikkakeMod,
        category: "candlestick",
        run: cdlhikkakemod,
    },
    PatternRunner {
        id: "cdlkicking",
        pattern_type: PatternType::CdlKicking,
        category: "candlestick",
        run: cdlkicking,
    },
    PatternRunner {
        id: "cdlkickingbylength",
        pattern_type: PatternType::CdlKickingByLength,
        category: "candlestick",
        run: cdlkickingbylength,
    },
    PatternRunner {
        id: "cdlladderbottom",
        pattern_type: PatternType::CdlLadderBottom,
        category: "candlestick",
        run: cdlladderbottom,
    },
    PatternRunner {
        id: "cdlmathold",
        pattern_type: PatternType::CdlMatHold,
        category: "candlestick",
        run: cdlmathold,
    },
    PatternRunner {
        id: "cdlrisefall3methods",
        pattern_type: PatternType::CdlRiseFall3Methods,
        category: "candlestick",
        run: cdlrisefall3methods,
    },
    PatternRunner {
        id: "cdlstalledpattern",
        pattern_type: PatternType::CdlStalledPattern,
        category: "candlestick",
        run: cdlstalledpattern,
    },
    PatternRunner {
        id: "cdltasukigap",
        pattern_type: PatternType::CdlTasukiGap,
        category: "candlestick",
        run: cdltasukigap,
    },
    PatternRunner {
        id: "cdlunique3river",
        pattern_type: PatternType::CdlUnique3River,
        category: "candlestick",
        run: cdlunique3river,
    },
    PatternRunner {
        id: "cdlupsidegap2crows",
        pattern_type: PatternType::CdlUpsideGap2Crows,
        category: "candlestick",
        run: cdlupsidegap2crows,
    },
    PatternRunner {
        id: "cdlxsidegap3methods",
        pattern_type: PatternType::CdlXSideGap3Methods,
        category: "candlestick",
        run: cdlxsidegap3methods,
    },
];

const PATTERN_SPECS: [PatternSpec; 61] = [
    PatternSpec {
        id: "cdl2crows",
        row_index: 0,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdl3blackcrows",
        row_index: 1,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdl3inside",
        row_index: 2,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdl3linestrike",
        row_index: 3,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdl3outside",
        row_index: 4,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdl3starsinsouth",
        row_index: 5,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdl3whitesoldiers",
        row_index: 6,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlabandonedbaby",
        row_index: 7,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdladvanceblock",
        row_index: 8,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlbelthold",
        row_index: 9,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlbreakaway",
        row_index: 10,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlclosingmarubozu",
        row_index: 11,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlconcealbabyswall",
        row_index: 12,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlcounterattack",
        row_index: 13,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdldarkcloudcover",
        row_index: 14,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdldoji",
        row_index: 15,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdldojistar",
        row_index: 16,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdldragonflydoji",
        row_index: 17,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlengulfing",
        row_index: 18,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdleveningdojistar",
        row_index: 19,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdleveningstar",
        row_index: 20,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlmorningstar",
        row_index: 21,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlgravestonedoji",
        row_index: 22,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlhammer",
        row_index: 23,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlhangingman",
        row_index: 24,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlharami",
        row_index: 25,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlharamicross",
        row_index: 26,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlhighwave",
        row_index: 27,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlinvertedhammer",
        row_index: 28,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdllongleggeddoji",
        row_index: 29,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdllongline",
        row_index: 30,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlmarubozu",
        row_index: 31,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlrickshawman",
        row_index: 32,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlshootingstar",
        row_index: 33,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlshortline",
        row_index: 34,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlspinningtop",
        row_index: 35,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdltakuri",
        row_index: 36,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlhomingpigeon",
        row_index: 37,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlmatchinglow",
        row_index: 38,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlinneck",
        row_index: 39,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlonneck",
        row_index: 40,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlpiercing",
        row_index: 41,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlthrusting",
        row_index: 42,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlmorningdojistar",
        row_index: 43,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdltristar",
        row_index: 44,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlidentical3crows",
        row_index: 45,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlsticksandwich",
        row_index: 46,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlseparatinglines",
        row_index: 47,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlgapsidesidewhite",
        row_index: 48,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlhikkake",
        row_index: 49,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlhikkakemod",
        row_index: 50,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlkicking",
        row_index: 51,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlkickingbylength",
        row_index: 52,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlladderbottom",
        row_index: 53,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlmathold",
        row_index: 54,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlrisefall3methods",
        row_index: 55,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlstalledpattern",
        row_index: 56,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdltasukigap",
        row_index: 57,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlunique3river",
        row_index: 58,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlupsidegap2crows",
        row_index: 59,
        category: "candlestick",
    },
    PatternSpec {
        id: "cdlxsidegap3methods",
        row_index: 60,
        category: "candlestick",
    },
];

pub fn list_patterns() -> &'static [PatternSpec] {
    &PATTERN_SPECS
}

pub fn pattern_type_from_id(id: &str) -> Option<PatternType> {
    PATTERN_RUNNERS
        .iter()
        .find(|runner| runner.id.eq_ignore_ascii_case(id))
        .map(|runner| runner.pattern_type)
}

pub fn pattern(input: &PatternInput<'_>) -> Result<PatternOutput, PatternError> {
    pattern_with_kernel(input, Kernel::Auto)
}

pub fn pattern_with_kernel(
    input: &PatternInput<'_>,
    kernel: Kernel,
) -> Result<PatternOutput, PatternError> {
    let _ = kernel;
    PATTERN_RUNNERS
        .iter()
        .find(|runner| runner.pattern_type == input.params.pattern_type)
        .map(|runner| (runner.run)(input))
        .unwrap_or(Err(PatternError::Unknown))
}

pub fn pattern_recognition(
    input: &PatternRecognitionInput<'_>,
) -> Result<PatternRecognitionOutput, PatternRecognitionError> {
    pattern_recognition_with_kernel(input, Kernel::Auto)
}

pub fn pattern_recognition_with_kernel(
    input: &PatternRecognitionInput<'_>,
    kernel: Kernel,
) -> Result<PatternRecognitionOutput, PatternRecognitionError> {
    let _ = kernel;
    let (pattern_data, cols) = match &input.data {
        PatternRecognitionData::Candles { candles } => {
            (PatternData::Candles { candles }, candles.close.len())
        }
        PatternRecognitionData::Slices {
            open,
            high,
            low,
            close,
        } => {
            if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
                return Err(PatternRecognitionError::DataLengthMismatch {
                    open: open.len(),
                    high: high.len(),
                    low: low.len(),
                    close: close.len(),
                });
            }
            (
                PatternData::Slices {
                    open,
                    high,
                    low,
                    close,
                },
                close.len(),
            )
        }
    };

    let rows = PATTERN_RUNNERS.len();
    let mut matrix = make_uninit_u8_matrix(rows, cols);
    let mut pattern_input = PatternInput {
        data: pattern_data,
        params: PatternParams {
            pattern_type: PatternType::default(),
            penetration: 0.0,
        },
    };

    for (row, runner) in PATTERN_RUNNERS.iter().enumerate() {
        pattern_input.params.pattern_type = runner.pattern_type;
        let out = (runner.run)(&pattern_input)?;
        if out.values.len() != cols {
            return Err(PatternRecognitionError::OutputLengthMismatch {
                pattern_id: runner.id,
                expected: cols,
                got: out.values.len(),
            });
        }
        let offset = row * cols;
        let dst = &mut matrix[offset..offset + cols];
        for idx in 0..cols {
            unsafe {
                dst.get_unchecked_mut(idx)
                    .write((*out.values.get_unchecked(idx) != 0) as u8);
            }
        }
    }

    let pattern_ids = PATTERN_RUNNERS.iter().map(|x| x.id).collect();
    let values_u8 = unsafe { assume_init_u8(matrix) };

    Ok(PatternRecognitionOutput {
        rows,
        cols,
        values_u8,
        pattern_ids,
        warmup: None,
    })
}

pub fn extract_pattern_series<'a>(
    output: &'a PatternRecognitionOutput,
    pattern_id: &str,
) -> Option<&'a [u8]> {
    let row = output.pattern_ids.iter().position(|id| *id == pattern_id)?;
    let start = row.checked_mul(output.cols)?;
    let end = start.checked_add(output.cols)?;
    output.values_u8.get(start..end)
}

pub fn pattern_hit(
    output: &PatternRecognitionOutput,
    pattern_id: &str,
    bar: usize,
) -> Option<bool> {
    let row = output.pattern_ids.iter().position(|id| *id == pattern_id)?;
    if bar >= output.cols {
        return None;
    }
    let idx = row.checked_mul(output.cols)?.checked_add(bar)?;
    output.values_u8.get(idx).map(|x| *x != 0)
}

fn make_uninit_u8_matrix(rows: usize, cols: usize) -> Vec<MaybeUninit<u8>> {
    let total = rows
        .checked_mul(cols)
        .expect("rows * cols overflowed usize");

    let mut v: Vec<MaybeUninit<u8>> = Vec::new();
    v.try_reserve_exact(total)
        .expect("OOM in make_uninit_u8_matrix");

    #[cfg(not(debug_assertions))]
    unsafe {
        v.set_len(total);
    }

    #[cfg(debug_assertions)]
    {
        for _ in 0..total {
            v.push(MaybeUninit::new(0xCD));
        }
    }

    v
}

unsafe fn assume_init_u8(mut v: Vec<MaybeUninit<u8>>) -> Vec<u8> {
    let ptr = v.as_mut_ptr() as *mut u8;
    let len = v.len();
    let cap = v.capacity();
    std::mem::forget(v);
    Vec::from_raw_parts(ptr, len, cap)
}

#[derive(Debug, Error)]
pub enum PatternError {
    #[error("pattern_recognition: Not enough data points. Length={len}, pattern={pattern:?}")]
    NotEnoughData { len: usize, pattern: PatternType },

    #[error("pattern_recognition: Candle field error: {0}")]
    CandleFieldError(String),

    #[error("pattern_recognition: Unknown error occurred.")]
    Unknown,
}

#[inline(always)]
fn candle_color(open: f64, close: f64) -> i32 {
    if close >= open {
        1
    } else {
        -1
    }
}

#[inline(always)]
fn real_body(open: f64, close: f64) -> f64 {
    (close - open).abs()
}

#[inline(always)]
fn candle_range(open: f64, close: f64) -> f64 {
    real_body(open, close)
}

#[inline(always)]
fn upper_shadow(open: f64, high: f64, close: f64) -> f64 {
    if close >= open {
        high - close
    } else {
        high - open
    }
}

#[inline(always)]
fn lower_shadow(open: f64, low: f64, close: f64) -> f64 {
    if close >= open {
        open - low
    } else {
        close - low
    }
}

#[inline]
fn input_ohlc<'a>(
    data: &'a PatternData<'a>,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], &'a [f64]), PatternError> {
    match data {
        PatternData::Candles { candles } => {
            Ok((&candles.open, &candles.high, &candles.low, &candles.close))
        }
        PatternData::Slices {
            open,
            high,
            low,
            close,
        } => {
            if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
                return Err(PatternError::CandleFieldError(
                    "open/high/low/close length mismatch".to_string(),
                ));
            }
            Ok((open, high, low, close))
        }
    }
}

#[inline]
pub fn cdl2crows(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    const BODY_LONG_PERIOD: usize = 10;

    let (open, high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let lookback_total = 2 + BODY_LONG_PERIOD;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type.clone(),
        });
    }

    let mut out = vec![0i8; size];

    let mut body_long_period_total = 0.0;
    let body_long_trailing_start = 0;
    let body_long_trailing_end = BODY_LONG_PERIOD;
    for i in body_long_trailing_start..body_long_trailing_end {
        body_long_period_total += candle_range(open[i], close[i]);
    }

    for i in lookback_total..size {
        let first_color = candle_color(open[i - 2], close[i - 2]);
        let first_body = real_body(open[i - 2], close[i - 2]);
        let body_long_avg = body_long_period_total / (BODY_LONG_PERIOD as f64);

        let second_color = candle_color(open[i - 1], close[i - 1]);
        let third_color = candle_color(open[i], close[i]);

        let second_rb_min = open[i - 1].min(close[i - 1]);
        let first_rb_max = open[i - 2].max(close[i - 2]);
        let real_body_gap_up = second_rb_min > first_rb_max;

        let third_opens_in_2nd_body = open[i] < open[i - 1] && open[i] > close[i - 1];

        let third_closes_in_1st_body = close[i] > open[i - 2] && close[i] < close[i - 2];

        if (first_color == 1)
            && (first_body > body_long_avg)
            && (second_color == -1)
            && real_body_gap_up
            && (third_color == -1)
            && third_opens_in_2nd_body
            && third_closes_in_1st_body
        {
            out[i] = -100;
        } else {
            out[i] = 0;
        }

        let old_idx = i - lookback_total;
        let new_idx = i - 2;
        body_long_period_total += candle_range(open[new_idx], close[new_idx])
            - candle_range(open[old_idx], close[old_idx]);
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdl3blackcrows(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    const SHADOW_VERY_SHORT_PERIOD: usize = 10;

    let (open, high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let lookback_total = 3 + SHADOW_VERY_SHORT_PERIOD;
    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type.clone(),
        });
    }

    let mut out = vec![0i8; size];

    fn candle_color(o: f64, c: f64) -> i8 {
        if c >= o {
            1
        } else {
            -1
        }
    }

    fn lower_shadow(o: f64, c: f64, l: f64) -> f64 {
        if c < o {
            c - l
        } else {
            o - l
        }
    }

    let mut sum2 = 0.0;
    let mut sum1 = 0.0;
    let mut sum0 = 0.0;
    for i in 0..SHADOW_VERY_SHORT_PERIOD {
        sum2 += lower_shadow(open[i], close[i], low[i]);
        sum1 += lower_shadow(open[i + 1], close[i + 1], low[i + 1]);
        sum0 += lower_shadow(open[i + 2], close[i + 2], low[i + 2]);
    }

    for i in lookback_total..size {
        let avg2 = sum2 / (SHADOW_VERY_SHORT_PERIOD as f64);
        let avg1 = sum1 / (SHADOW_VERY_SHORT_PERIOD as f64);
        let avg0 = sum0 / (SHADOW_VERY_SHORT_PERIOD as f64);

        if candle_color(open[i - 3], close[i - 3]) == 1
            && candle_color(open[i - 2], close[i - 2]) == -1
            && lower_shadow(open[i - 2], close[i - 2], low[i - 2]) < avg2
            && candle_color(open[i - 1], close[i - 1]) == -1
            && lower_shadow(open[i - 1], close[i - 1], low[i - 1]) < avg1
            && candle_color(open[i], close[i]) == -1
            && lower_shadow(open[i], close[i], low[i]) < avg0
            && open[i - 1] < open[i - 2]
            && open[i - 1] > close[i - 2]
            && open[i] < open[i - 1]
            && open[i] > close[i - 1]
            && high[i - 3] > close[i - 2]
            && close[i - 2] > close[i - 1]
            && close[i - 1] > close[i]
        {
            out[i] = -100;
        } else {
            out[i] = 0;
        }

        let old_idx2 = i - lookback_total;
        let new_idx2 = i - 2;
        sum2 += lower_shadow(open[new_idx2], close[new_idx2], low[new_idx2])
            - lower_shadow(open[old_idx2], close[old_idx2], low[old_idx2]);

        let old_idx1 = i - lookback_total + 1;
        let new_idx1 = i - 1;
        sum1 += lower_shadow(open[new_idx1], close[new_idx1], low[new_idx1])
            - lower_shadow(open[old_idx1], close[old_idx1], low[old_idx1]);

        let old_idx0 = i - lookback_total + 2;
        let new_idx0 = i;
        sum0 += lower_shadow(open[new_idx0], close[new_idx0], low[new_idx0])
            - lower_shadow(open[old_idx0], close[old_idx0], low[old_idx0]);
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdl3inside(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    const BODY_LONG_PERIOD: usize = 10;
    const BODY_SHORT_PERIOD: usize = 10;

    let (open, high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let lookback_total = 2 + BODY_LONG_PERIOD.max(BODY_SHORT_PERIOD);
    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type.clone(),
        });
    }

    fn candle_color(o: f64, c: f64) -> i8 {
        if c >= o {
            1
        } else {
            -1
        }
    }

    fn candle_range(o: f64, c: f64) -> f64 {
        (c - o).abs()
    }

    fn real_body(o: f64, c: f64) -> f64 {
        (c - o).abs()
    }

    fn max2(a: f64, b: f64) -> f64 {
        if a > b {
            a
        } else {
            b
        }
    }

    fn min2(a: f64, b: f64) -> f64 {
        if a < b {
            a
        } else {
            b
        }
    }

    let mut out = vec![0i8; size];

    let mut body_long_period_total = 0.0;
    let mut body_short_period_total = 0.0;

    for i in 0..BODY_LONG_PERIOD {
        body_long_period_total += candle_range(open[i], close[i]);
    }
    for i in 0..BODY_SHORT_PERIOD {
        body_short_period_total += candle_range(open[i], close[i]);
    }

    for i in lookback_total..size {
        let avg_body_long = body_long_period_total / BODY_LONG_PERIOD as f64;
        let avg_body_short = body_short_period_total / BODY_SHORT_PERIOD as f64;

        if real_body(open[i - 2], close[i - 2]) > avg_body_long
            && real_body(open[i - 1], close[i - 1]) <= avg_body_short
            && max2(close[i - 1], open[i - 1]) < max2(close[i - 2], open[i - 2])
            && min2(close[i - 1], open[i - 1]) > min2(close[i - 2], open[i - 2])
            && ((candle_color(open[i - 2], close[i - 2]) == 1
                && candle_color(open[i], close[i]) == -1
                && close[i] < open[i - 2])
                || (candle_color(open[i - 2], close[i - 2]) == -1
                    && candle_color(open[i], close[i]) == 1
                    && close[i] > open[i - 2]))
        {
            out[i] = -candle_color(open[i - 2], close[i - 2]) * 100;
        } else {
            out[i] = 0;
        }

        let old_idx_long = i - lookback_total;
        body_long_period_total += candle_range(open[i - 2], close[i - 2])
            - candle_range(open[old_idx_long], close[old_idx_long]);

        let old_idx_short = i - lookback_total + 1;
        body_short_period_total += candle_range(open[i - 1], close[i - 1])
            - candle_range(open[old_idx_short], close[old_idx_short]);
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdl3linestrike(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    const NEAR_PERIOD: usize = 10;

    let (open, high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let lookback_total = 3 + NEAR_PERIOD;
    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type.clone(),
        });
    }

    fn candle_color(o: f64, c: f64) -> i8 {
        if c >= o {
            1
        } else {
            -1
        }
    }

    fn candle_range(o: f64, c: f64) -> f64 {
        (c - o).abs()
    }

    fn max2(a: f64, b: f64) -> f64 {
        if a > b {
            a
        } else {
            b
        }
    }

    fn min2(a: f64, b: f64) -> f64 {
        if a < b {
            a
        } else {
            b
        }
    }

    let mut out = vec![0i8; size];
    let mut sum3 = 0.0;
    let mut sum2 = 0.0;

    for i in 0..NEAR_PERIOD {
        sum3 += candle_range(open[i], close[i]);
        sum2 += candle_range(open[i + 1], close[i + 1]);
    }

    for i in lookback_total..size {
        let avg3 = sum3 / (NEAR_PERIOD as f64);
        let avg2 = sum2 / (NEAR_PERIOD as f64);

        if candle_color(open[i - 3], close[i - 3]) == candle_color(open[i - 2], close[i - 2])
            && candle_color(open[i - 2], close[i - 2]) == candle_color(open[i - 1], close[i - 1])
            && candle_color(open[i], close[i]) == -candle_color(open[i - 1], close[i - 1])
            && open[i - 2] >= min2(open[i - 3], close[i - 3]) - avg3
            && open[i - 2] <= max2(open[i - 3], close[i - 3]) + avg3
            && open[i - 1] >= min2(open[i - 2], close[i - 2]) - avg2
            && open[i - 1] <= max2(open[i - 2], close[i - 2]) + avg2
            && ((candle_color(open[i - 1], close[i - 1]) == 1
                && close[i - 1] > close[i - 2]
                && close[i - 2] > close[i - 3]
                && open[i] > close[i - 1]
                && close[i] < open[i - 3])
                || (candle_color(open[i - 1], close[i - 1]) == -1
                    && close[i - 1] < close[i - 2]
                    && close[i - 2] < close[i - 3]
                    && open[i] < close[i - 1]
                    && close[i] > open[i - 3]))
        {
            out[i] = candle_color(open[i - 1], close[i - 1]) * 100;
        } else {
            out[i] = 0;
        }

        let old_idx3 = i - lookback_total;
        let new_idx3 = i - 3;
        sum3 += candle_range(open[new_idx3], close[new_idx3])
            - candle_range(open[old_idx3], close[old_idx3]);

        let old_idx2 = i - lookback_total + 1;
        let new_idx2 = i - 2;
        sum2 += candle_range(open[new_idx2], close[new_idx2])
            - candle_range(open[old_idx2], close[old_idx2]);
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdl3outside(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, high, low, close) = input_ohlc(&input.data)?;

    fn candle_color(o: f64, c: f64) -> i8 {
        if c >= o {
            1
        } else {
            -1
        }
    }

    let size = open.len();
    let lookback_total = 2;

    if size < lookback_total + 1 {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type.clone(),
        });
    }

    let mut out = vec![0i8; size];

    for i in lookback_total..size {
        let second_candle_color = candle_color(open[i - 1], close[i - 1]);
        let first_candle_color = candle_color(open[i - 2], close[i - 2]);

        let white_engulfs_black = second_candle_color == 1
            && first_candle_color == -1
            && close[i - 1] > open[i - 2]
            && open[i - 1] < close[i - 2]
            && close[i] > close[i - 1];

        let black_engulfs_white = second_candle_color == -1
            && first_candle_color == 1
            && open[i - 1] > close[i - 2]
            && close[i - 1] < open[i - 2]
            && close[i] < close[i - 1];

        if white_engulfs_black || black_engulfs_white {
            out[i] = second_candle_color * 100;
        } else {
            out[i] = 0;
        }
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdl3starsinsouth(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    const BODY_LONG_PERIOD: usize = 10;
    const SHADOW_LONG_PERIOD: usize = 10;
    const SHADOW_VERY_SHORT_PERIOD: usize = 10;
    const BODY_SHORT_PERIOD: usize = 10;

    let (open, high, low, close) = input_ohlc(&input.data)?;

    fn candle_color(o: f64, c: f64) -> i8 {
        if c >= o {
            1
        } else {
            -1
        }
    }
    fn real_body(o: f64, c: f64) -> f64 {
        (c - o).abs()
    }
    fn lower_shadow(o: f64, c: f64, l: f64) -> f64 {
        if c < o {
            c - l
        } else {
            o - l
        }
    }
    fn upper_shadow(o: f64, c: f64, h: f64) -> f64 {
        if c < o {
            h - o
        } else {
            h - c
        }
    }
    fn candle_range(o: f64, c: f64) -> f64 {
        (c - o).abs()
    }

    let size = open.len();
    let lookback_total = 2 + BODY_LONG_PERIOD
        .max(SHADOW_LONG_PERIOD)
        .max(SHADOW_VERY_SHORT_PERIOD)
        .max(BODY_SHORT_PERIOD);

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type.clone(),
        });
    }

    let mut out = vec![0i8; size];

    let mut body_long_sum = 0.0;
    let mut shadow_long_sum = 0.0;
    let mut shadow_very_short_sum_1 = 0.0;
    let mut shadow_very_short_sum_0 = 0.0;
    let mut body_short_sum = 0.0;

    let body_long_trail_start = lookback_total - BODY_LONG_PERIOD;
    for idx in body_long_trail_start..lookback_total {
        let ref_index = if idx >= 2 { idx - 2 } else { 0 };
        body_long_sum += candle_range(open[ref_index], close[ref_index]);
    }

    let shadow_long_trail_start = lookback_total - SHADOW_LONG_PERIOD;
    for idx in shadow_long_trail_start..lookback_total {
        let ref_index = if idx >= 2 { idx - 2 } else { 0 };
        shadow_long_sum += candle_range(open[ref_index], close[ref_index]);
    }

    let shadow_very_short_trail_start = lookback_total - SHADOW_VERY_SHORT_PERIOD;
    for idx in shadow_very_short_trail_start..lookback_total {
        let ref_index_1 = if idx >= 1 { idx - 1 } else { 0 };
        shadow_very_short_sum_1 +=
            lower_shadow(open[ref_index_1], close[ref_index_1], low[ref_index_1]);

        shadow_very_short_sum_0 += lower_shadow(open[idx], close[idx], low[idx]);
    }

    let body_short_trail_start = lookback_total - BODY_SHORT_PERIOD;
    for idx in body_short_trail_start..lookback_total {
        body_short_sum += candle_range(open[idx], close[idx]);
    }
    for i in lookback_total..size {
        let avg_body_long = body_long_sum / (BODY_LONG_PERIOD as f64);
        let avg_shadow_long = shadow_long_sum / (SHADOW_LONG_PERIOD as f64);
        let avg_shadow_very_short_1 = shadow_very_short_sum_1 / (SHADOW_VERY_SHORT_PERIOD as f64);
        let avg_shadow_very_short_0 = shadow_very_short_sum_0 / (SHADOW_VERY_SHORT_PERIOD as f64);
        let avg_body_short = body_short_sum / (BODY_SHORT_PERIOD as f64);

        if candle_color(open[i - 2], close[i - 2]) == -1
            && candle_color(open[i - 1], close[i - 1]) == -1
            && candle_color(open[i], close[i]) == -1
            && real_body(open[i - 2], close[i - 2]) > avg_body_long
            && lower_shadow(open[i - 2], close[i - 2], low[i - 2]) > avg_shadow_long
            && real_body(open[i - 1], close[i - 1]) < real_body(open[i - 2], close[i - 2])
            && open[i - 1] > close[i - 2]
            && open[i - 1] <= high[i - 2]
            && low[i - 1] < close[i - 2]
            && low[i - 1] >= low[i - 2]
            && lower_shadow(open[i - 1], close[i - 1], low[i - 1]) > avg_shadow_very_short_1
            && real_body(open[i], close[i]) < avg_body_short
            && lower_shadow(open[i], close[i], low[i]) < avg_shadow_very_short_0
            && upper_shadow(open[i], close[i], high[i]) < avg_shadow_very_short_0
            && low[i] > low[i - 1]
            && high[i] < high[i - 1]
        {
            out[i] = 100;
        } else {
            out[i] = 0;
        }

        let old_idx = i - lookback_total;

        {
            let old_ref = if old_idx >= 2 { old_idx - 2 } else { 0 };
            let new_ref = i - 2;
            body_long_sum += candle_range(open[new_ref], close[new_ref])
                - candle_range(open[old_ref], close[old_ref]);
        }

        {
            let old_ref = if old_idx >= 2 { old_idx - 2 } else { 0 };
            let new_ref = i - 2;
            shadow_long_sum += candle_range(open[new_ref], close[new_ref])
                - candle_range(open[old_ref], close[old_ref]);
        }

        {
            let old_ref_1 = if old_idx >= 1 { old_idx - 1 } else { 0 };
            let new_ref_1 = i - 1;
            shadow_very_short_sum_1 +=
                lower_shadow(open[new_ref_1], close[new_ref_1], low[new_ref_1])
                    - lower_shadow(open[old_ref_1], close[old_ref_1], low[old_ref_1]);
        }
        {
            shadow_very_short_sum_0 += lower_shadow(open[i], close[i], low[i])
                - lower_shadow(open[old_idx], close[old_idx], low[old_idx]);
        }

        {
            body_short_sum +=
                candle_range(open[i], close[i]) - candle_range(open[old_idx], close[old_idx]);
        }
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdl3whitesoldiers(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    const SHADOW_VERY_SHORT_PERIOD: usize = 10;
    const NEAR_PERIOD: usize = 10;
    const FAR_PERIOD: usize = 10;
    const BODY_SHORT_PERIOD: usize = 10;

    let (open, high, low, close) = input_ohlc(&input.data)?;

    fn candle_color(o: f64, c: f64) -> i8 {
        if c >= o {
            1
        } else {
            -1
        }
    }

    fn candle_range(o: f64, c: f64) -> f64 {
        (c - o).abs()
    }

    fn real_body(o: f64, c: f64) -> f64 {
        (c - o).abs()
    }

    fn upper_shadow(o: f64, c: f64, h: f64) -> f64 {
        if c < o {
            h - o
        } else {
            h - c
        }
    }

    fn max2(a: f64, b: f64) -> f64 {
        if a > b {
            a
        } else {
            b
        }
    }

    let size = open.len();
    let lookback_total = 2 + SHADOW_VERY_SHORT_PERIOD
        .max(NEAR_PERIOD)
        .max(FAR_PERIOD)
        .max(BODY_SHORT_PERIOD);

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type.clone(),
        });
    }

    let mut out = vec![0i8; size];
    let mut shadow_very_short_sum = [0.0; 3];
    let mut near_sum = [0.0; 3];
    let mut far_sum = [0.0; 3];
    let mut body_short_sum = 0.0;

    for i in 0..SHADOW_VERY_SHORT_PERIOD {
        shadow_very_short_sum[2] += upper_shadow(open[i], close[i], high[i]);
        if i + 1 < size {
            shadow_very_short_sum[1] += upper_shadow(open[i + 1], close[i + 1], high[i + 1]);
        }
        if i + 2 < size {
            shadow_very_short_sum[0] += upper_shadow(open[i + 2], close[i + 2], high[i + 2]);
        }
    }
    for i in 0..NEAR_PERIOD {
        if i + 2 < size {
            near_sum[2] += candle_range(open[i + 2], close[i + 2]);
        }
        if i + 1 < size {
            near_sum[1] += candle_range(open[i + 1], close[i + 1]);
        }
    }
    for i in 0..FAR_PERIOD {
        if i + 2 < size {
            far_sum[2] += candle_range(open[i + 2], close[i + 2]);
        }
        if i + 1 < size {
            far_sum[1] += candle_range(open[i + 1], close[i + 1]);
        }
    }
    for i in 0..BODY_SHORT_PERIOD {
        body_short_sum += candle_range(open[i], close[i]);
    }

    for i in lookback_total..size {
        let avg_sv_2 = shadow_very_short_sum[2] / (SHADOW_VERY_SHORT_PERIOD as f64);
        let avg_sv_1 = shadow_very_short_sum[1] / (SHADOW_VERY_SHORT_PERIOD as f64);
        let avg_sv_0 = shadow_very_short_sum[0] / (SHADOW_VERY_SHORT_PERIOD as f64);
        let avg_near_2 = near_sum[2] / (NEAR_PERIOD as f64);
        let avg_near_1 = near_sum[1] / (NEAR_PERIOD as f64);
        let avg_far_2 = far_sum[2] / (FAR_PERIOD as f64);
        let avg_far_1 = far_sum[1] / (FAR_PERIOD as f64);
        let avg_body_short = body_short_sum / (BODY_SHORT_PERIOD as f64);

        if candle_color(open[i - 2], close[i - 2]) == 1
            && upper_shadow(open[i - 2], close[i - 2], high[i - 2]) < avg_sv_2
            && candle_color(open[i - 1], close[i - 1]) == 1
            && upper_shadow(open[i - 1], close[i - 1], high[i - 1]) < avg_sv_1
            && candle_color(open[i], close[i]) == 1
            && upper_shadow(open[i], close[i], high[i]) < avg_sv_0
            && close[i] > close[i - 1]
            && close[i - 1] > close[i - 2]
            && open[i - 1] > open[i - 2]
            && open[i - 1] <= close[i - 2] + avg_near_2
            && open[i] > open[i - 1]
            && open[i] <= close[i - 1] + avg_near_1
            && real_body(open[i - 1], close[i - 1])
                > real_body(open[i - 2], close[i - 2]) - avg_far_2
            && real_body(open[i], close[i]) > real_body(open[i - 1], close[i - 1]) - avg_far_1
            && real_body(open[i], close[i]) > avg_body_short
        {
            out[i] = 100;
        } else {
            out[i] = 0;
        }

        let old_idx = i - lookback_total;
        shadow_very_short_sum[2] += upper_shadow(open[i - 2], close[i - 2], high[i - 2])
            - upper_shadow(
                open[old_idx.saturating_sub(2)],
                close[old_idx.saturating_sub(2)],
                high[old_idx.saturating_sub(2)],
            );
        shadow_very_short_sum[1] += upper_shadow(open[i - 1], close[i - 1], high[i - 1])
            - upper_shadow(
                open[old_idx.saturating_sub(1)],
                close[old_idx.saturating_sub(1)],
                high[old_idx.saturating_sub(1)],
            );
        shadow_very_short_sum[0] += upper_shadow(open[i], close[i], high[i])
            - upper_shadow(open[old_idx], close[old_idx], high[old_idx]);

        far_sum[2] += candle_range(open[i - 2], close[i - 2])
            - candle_range(
                open[old_idx.saturating_sub(2)],
                close[old_idx.saturating_sub(2)],
            );
        far_sum[1] += candle_range(open[i - 1], close[i - 1])
            - candle_range(
                open[old_idx.saturating_sub(1)],
                close[old_idx.saturating_sub(1)],
            );

        near_sum[2] += candle_range(open[i - 2], close[i - 2])
            - candle_range(
                open[old_idx.saturating_sub(2)],
                close[old_idx.saturating_sub(2)],
            );
        near_sum[1] += candle_range(open[i - 1], close[i - 1])
            - candle_range(
                open[old_idx.saturating_sub(1)],
                close[old_idx.saturating_sub(1)],
            );

        body_short_sum +=
            candle_range(open[i], close[i]) - candle_range(open[old_idx], close[old_idx]);
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlabandonedbaby(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    const BODY_LONG_PERIOD: usize = 10;
    const BODY_DOJI_PERIOD: usize = 10;
    const BODY_SHORT_PERIOD: usize = 10;

    let (open, high, low, close) = input_ohlc(&input.data)?;

    let penetration = input.params.penetration;

    fn candle_color(o: f64, c: f64) -> i8 {
        if c >= o {
            1
        } else {
            -1
        }
    }

    fn real_body(o: f64, c: f64) -> f64 {
        (c - o).abs()
    }

    fn candle_range(o: f64, c: f64) -> f64 {
        (c - o).abs()
    }

    fn candle_gap_up(idx1: usize, idx2: usize, low: &[f64], high: &[f64]) -> bool {
        low[idx1] > high[idx2]
    }

    fn candle_gap_down(idx1: usize, idx2: usize, low: &[f64], high: &[f64]) -> bool {
        high[idx1] < low[idx2]
    }

    let size = open.len();
    let lookback_total = 2 + BODY_LONG_PERIOD
        .max(BODY_DOJI_PERIOD)
        .max(BODY_SHORT_PERIOD);

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type.clone(),
        });
    }

    let mut out = vec![0i8; size];
    let mut body_long_sum = 0.0;
    let mut body_doji_sum = 0.0;
    let mut body_short_sum = 0.0;

    for i in 0..BODY_LONG_PERIOD {
        body_long_sum += candle_range(open[i], close[i]);
    }
    for i in 0..BODY_DOJI_PERIOD {
        body_doji_sum += candle_range(open[i], close[i]);
    }
    for i in 0..BODY_SHORT_PERIOD {
        body_short_sum += candle_range(open[i], close[i]);
    }

    for i in lookback_total..size {
        let avg_body_long = body_long_sum / BODY_LONG_PERIOD as f64;
        let avg_body_doji = body_doji_sum / BODY_DOJI_PERIOD as f64;
        let avg_body_short = body_short_sum / BODY_SHORT_PERIOD as f64;

        if real_body(open[i - 2], close[i - 2]) > avg_body_long
            && real_body(open[i - 1], close[i - 1]) <= avg_body_doji
            && real_body(open[i], close[i]) > avg_body_short
            && ((candle_color(open[i - 2], close[i - 2]) == 1
                && candle_color(open[i], close[i]) == -1
                && close[i] < close[i - 2] - real_body(open[i - 2], close[i - 2]) * penetration
                && candle_gap_up(i - 1, i - 2, &low, &high)
                && candle_gap_down(i, i - 1, &low, &high))
                || (candle_color(open[i - 2], close[i - 2]) == -1
                    && candle_color(open[i], close[i]) == 1
                    && close[i]
                        > close[i - 2] + real_body(open[i - 2], close[i - 2]) * penetration
                    && candle_gap_down(i - 1, i - 2, &low, &high)
                    && candle_gap_up(i, i - 1, &low, &high)))
        {
            out[i] = candle_color(open[i], close[i]) * 100;
        } else {
            out[i] = 0;
        }

        let old_idx = i - lookback_total;
        body_long_sum += candle_range(open[i - 2], close[i - 2])
            - candle_range(
                open[old_idx.saturating_sub(2)],
                close[old_idx.saturating_sub(2)],
            );
        body_doji_sum += candle_range(open[i - 1], close[i - 1])
            - candle_range(
                open[old_idx.saturating_sub(1)],
                close[old_idx.saturating_sub(1)],
            );
        body_short_sum +=
            candle_range(open[i], close[i]) - candle_range(open[old_idx], close[old_idx]);
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdladvanceblock(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let shadow_short_period = 10;
    let shadow_long_period = 10;
    let near_period = 5;
    let far_period = 5;
    let body_long_period = 10;
    let lookback_total = 2 + shadow_short_period
        .max(shadow_long_period)
        .max(near_period)
        .max(far_period)
        .max(body_long_period);

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type.clone(),
        });
    }

    let mut out = vec![0i8; size];

    let mut shadow_short_period_total = [0.0; 3];
    let mut shadow_long_period_total = [0.0; 2];
    let mut near_period_total = [0.0; 3];
    let mut far_period_total = [0.0; 3];
    let mut body_long_period_total = 0.0;

    #[inline(always)]
    fn upper_shadow(o: f64, h: f64, c: f64) -> f64 {
        if c >= o {
            h - c
        } else {
            h - o
        }
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    let start_idx = lookback_total;
    let mut shadow_short_trailing_idx = start_idx.saturating_sub(shadow_short_period);
    let mut shadow_long_trailing_idx = start_idx.saturating_sub(shadow_long_period);
    let mut near_trailing_idx = start_idx.saturating_sub(near_period);
    let mut far_trailing_idx = start_idx.saturating_sub(far_period);
    let mut body_long_trailing_idx = start_idx.saturating_sub(body_long_period);

    let mut i = shadow_short_trailing_idx;
    while i < start_idx {
        shadow_short_period_total[2] += upper_shadow(
            open[i.saturating_sub(2)],
            high[i.saturating_sub(2)],
            close[i.saturating_sub(2)],
        );
        shadow_short_period_total[1] += upper_shadow(
            open[i.saturating_sub(1)],
            high[i.saturating_sub(1)],
            close[i.saturating_sub(1)],
        );
        shadow_short_period_total[0] += upper_shadow(open[i], high[i], close[i]);
        i += 1;
    }
    i = shadow_long_trailing_idx;
    while i < start_idx {
        shadow_long_period_total[1] += upper_shadow(
            open[i.saturating_sub(1)],
            high[i.saturating_sub(1)],
            close[i.saturating_sub(1)],
        );
        shadow_long_period_total[0] += upper_shadow(open[i], high[i], close[i]);
        i += 1;
    }
    i = near_trailing_idx;
    while i < start_idx {
        near_period_total[2] += real_body(open[i.saturating_sub(2)], close[i.saturating_sub(2)]);
        near_period_total[1] += real_body(open[i.saturating_sub(1)], close[i.saturating_sub(1)]);
        i += 1;
    }
    i = far_trailing_idx;
    while i < start_idx {
        far_period_total[2] += real_body(open[i.saturating_sub(2)], close[i.saturating_sub(2)]);
        far_period_total[1] += real_body(open[i.saturating_sub(1)], close[i.saturating_sub(1)]);
        i += 1;
    }
    i = body_long_trailing_idx;
    while i < start_idx {
        body_long_period_total += real_body(open[i.saturating_sub(2)], close[i.saturating_sub(2)]);
        i += 1;
    }

    let mut idx = start_idx;
    while idx < size {
        let c1_color = candle_color(open[idx - 2], close[idx - 2]);
        let c2_color = candle_color(open[idx - 1], close[idx - 1]);
        let c3_color = candle_color(open[idx], close[idx]);

        if c1_color == 1
            && c2_color == 1
            && c3_color == 1
            && close[idx] > close[idx - 1]
            && close[idx - 1] > close[idx - 2]
            && open[idx - 1] > open[idx - 2]
            && open[idx - 1] <= close[idx - 2] + candle_average(near_period_total[2], near_period)
            && open[idx] > open[idx - 1]
            && open[idx] <= close[idx - 1] + candle_average(near_period_total[1], near_period)
            && real_body(open[idx - 2], close[idx - 2])
                > candle_average(body_long_period_total, body_long_period)
            && upper_shadow(open[idx - 2], high[idx - 2], close[idx - 2])
                < candle_average(shadow_short_period_total[2], shadow_short_period)
            && ((real_body(open[idx - 1], close[idx - 1])
                < real_body(open[idx - 2], close[idx - 2])
                    - candle_average(far_period_total[2], far_period)
                && real_body(open[idx], close[idx])
                    < real_body(open[idx - 1], close[idx - 1])
                        + candle_average(near_period_total[1], near_period))
                || (real_body(open[idx], close[idx])
                    < real_body(open[idx - 1], close[idx - 1])
                        - candle_average(far_period_total[1], far_period))
                || (real_body(open[idx], close[idx]) < real_body(open[idx - 1], close[idx - 1])
                    && real_body(open[idx - 1], close[idx - 1])
                        < real_body(open[idx - 2], close[idx - 2])
                    && (upper_shadow(open[idx], high[idx], close[idx])
                        > candle_average(shadow_short_period_total[0], shadow_short_period)
                        || upper_shadow(open[idx - 1], high[idx - 1], close[idx - 1])
                            > candle_average(shadow_short_period_total[1], shadow_short_period)))
                || (real_body(open[idx], close[idx]) < real_body(open[idx - 1], close[idx - 1])
                    && upper_shadow(open[idx], high[idx], close[idx])
                        > candle_average(shadow_long_period_total[0], shadow_long_period)))
        {
            out[idx] = -100;
        }

        for tot_idx in (0..=2).rev() {
            if tot_idx < 3 {
                shadow_short_period_total[tot_idx] += upper_shadow(
                    open[idx.saturating_sub(tot_idx)],
                    high[idx.saturating_sub(tot_idx)],
                    close[idx.saturating_sub(tot_idx)],
                ) - upper_shadow(
                    open[shadow_short_trailing_idx.saturating_sub(tot_idx)],
                    high[shadow_short_trailing_idx.saturating_sub(tot_idx)],
                    close[shadow_short_trailing_idx.saturating_sub(tot_idx)],
                );
            }
        }

        for tot_idx in (0..=1).rev() {
            shadow_long_period_total[tot_idx] += upper_shadow(
                open[idx.saturating_sub(tot_idx)],
                high[idx.saturating_sub(tot_idx)],
                close[idx.saturating_sub(tot_idx)],
            ) - upper_shadow(
                open[shadow_long_trailing_idx.saturating_sub(tot_idx)],
                high[shadow_long_trailing_idx.saturating_sub(tot_idx)],
                close[shadow_long_trailing_idx.saturating_sub(tot_idx)],
            );
        }

        for tot_idx in (1..=2).rev() {
            far_period_total[tot_idx] += real_body(
                open[idx.saturating_sub(tot_idx)],
                close[idx.saturating_sub(tot_idx)],
            ) - real_body(
                open[far_trailing_idx.saturating_sub(tot_idx)],
                close[far_trailing_idx.saturating_sub(tot_idx)],
            );
            near_period_total[tot_idx] += real_body(
                open[idx.saturating_sub(tot_idx)],
                close[idx.saturating_sub(tot_idx)],
            ) - real_body(
                open[near_trailing_idx.saturating_sub(tot_idx)],
                close[near_trailing_idx.saturating_sub(tot_idx)],
            );
        }

        body_long_period_total += real_body(open[idx - 2], close[idx - 2])
            - real_body(
                open[body_long_trailing_idx.saturating_sub(2)],
                close[body_long_trailing_idx.saturating_sub(2)],
            );

        idx += 1;
        shadow_short_trailing_idx += 1;
        shadow_long_trailing_idx += 1;
        near_trailing_idx += 1;
        far_trailing_idx += 1;
        body_long_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlbelthold(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_long_period = 10;
    let shadow_very_short_period = 10;
    let lookback_total = body_long_period.max(shadow_very_short_period);

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type.clone(),
        });
    }

    let mut out = vec![0i8; size];
    let mut body_long_period_total = 0.0;
    let mut shadow_very_short_period_total = 0.0;

    #[inline(always)]
    fn lower_shadow(o: f64, l: f64, c: f64) -> f64 {
        if c >= o {
            o - l
        } else {
            c - l
        }
    }

    #[inline(always)]
    fn upper_shadow(o: f64, h: f64, c: f64) -> f64 {
        if c >= o {
            h - c
        } else {
            h - o
        }
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    let mut start_idx = lookback_total;
    let mut body_long_trailing_idx = start_idx.saturating_sub(body_long_period);
    let mut shadow_very_short_trailing_idx = start_idx.saturating_sub(shadow_very_short_period);

    let mut i = body_long_trailing_idx;
    while i < start_idx {
        body_long_period_total += real_body(open[i], close[i]);
        i += 1;
    }

    i = shadow_very_short_trailing_idx;
    while i < start_idx {
        let color = candle_color(open[i], close[i]);
        shadow_very_short_period_total += if color == 1 {
            lower_shadow(open[i], low[i], close[i])
        } else {
            upper_shadow(open[i], high[i], close[i])
        };
        i += 1;
    }

    while start_idx < size {
        let color = candle_color(open[start_idx], close[start_idx]);
        if real_body(open[start_idx], close[start_idx])
            > candle_average(body_long_period_total, body_long_period)
            && ((color == 1
                && lower_shadow(open[start_idx], low[start_idx], close[start_idx])
                    < candle_average(shadow_very_short_period_total, shadow_very_short_period))
                || (color == -1
                    && upper_shadow(open[start_idx], high[start_idx], close[start_idx])
                        < candle_average(shadow_very_short_period_total, shadow_very_short_period)))
        {
            out[start_idx] = (color as i8) * 100;
        }

        body_long_period_total += real_body(open[start_idx], close[start_idx])
            - real_body(open[body_long_trailing_idx], close[body_long_trailing_idx]);

        let trailing_color = candle_color(open[start_idx], close[start_idx]);
        let new_range = if trailing_color == 1 {
            lower_shadow(open[start_idx], low[start_idx], close[start_idx])
        } else {
            upper_shadow(open[start_idx], high[start_idx], close[start_idx])
        };
        let old_range_color = candle_color(
            open[shadow_very_short_trailing_idx],
            close[shadow_very_short_trailing_idx],
        );
        let old_range = if old_range_color == 1 {
            lower_shadow(
                open[shadow_very_short_trailing_idx],
                low[shadow_very_short_trailing_idx],
                close[shadow_very_short_trailing_idx],
            )
        } else {
            upper_shadow(
                open[shadow_very_short_trailing_idx],
                high[shadow_very_short_trailing_idx],
                close[shadow_very_short_trailing_idx],
            )
        };

        shadow_very_short_period_total += new_range - old_range;

        start_idx += 1;
        body_long_trailing_idx += 1;
        shadow_very_short_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlbreakaway(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_long_period = 10;
    let lookback_total = 4 + body_long_period;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type.clone(),
        });
    }

    let mut out = vec![0i8; size];
    let mut body_long_period_total = 0.0;

    #[inline(always)]
    fn candle_range(o: f64, c: f64) -> f64 {
        (c - o).abs()
    }

    #[inline(always)]
    fn gap_up(op_curr: f64, cl_curr: f64, op_prev: f64, cl_prev: f64) -> bool {
        op_curr.min(cl_curr) > op_prev.max(cl_prev)
    }

    #[inline(always)]
    fn gap_down(op_curr: f64, cl_curr: f64, op_prev: f64, cl_prev: f64) -> bool {
        op_curr.max(cl_curr) < op_prev.min(cl_prev)
    }

    let mut start_idx = lookback_total;
    let mut body_long_trailing_idx = start_idx.saturating_sub(body_long_period);

    let mut i = body_long_trailing_idx;
    while i < start_idx {
        body_long_period_total += candle_range(open[i - 4], close[i - 4]);
        i += 1;
    }

    while start_idx < size {
        let first_long = (close[start_idx - 4] - open[start_idx - 4]).abs()
            > body_long_period_total / body_long_period as f64;
        let c1 = candle_color(open[start_idx - 4], close[start_idx - 4]);
        let c2 = candle_color(open[start_idx - 3], close[start_idx - 3]);
        let c3 = candle_color(open[start_idx - 2], close[start_idx - 2]);
        let c4 = candle_color(open[start_idx - 1], close[start_idx - 1]);
        let c5 = candle_color(open[start_idx], close[start_idx]);

        if first_long
            && c1 == c2
            && c2 == c4
            && c4 == -c5
            && ((c1 == -1
                && gap_down(
                    open[start_idx - 3],
                    close[start_idx - 3],
                    open[start_idx - 4],
                    close[start_idx - 4],
                )
                && high[start_idx - 2] < high[start_idx - 3]
                && low[start_idx - 2] < low[start_idx - 3]
                && high[start_idx - 1] < high[start_idx - 2]
                && low[start_idx - 1] < low[start_idx - 2]
                && close[start_idx] > open[start_idx - 3]
                && close[start_idx] < close[start_idx - 4])
                || (c1 == 1
                    && gap_up(
                        open[start_idx - 3],
                        close[start_idx - 3],
                        open[start_idx - 4],
                        close[start_idx - 4],
                    )
                    && high[start_idx - 2] > high[start_idx - 3]
                    && low[start_idx - 2] > low[start_idx - 3]
                    && high[start_idx - 1] > high[start_idx - 2]
                    && low[start_idx - 1] > low[start_idx - 2]
                    && close[start_idx] < open[start_idx - 3]
                    && close[start_idx] > close[start_idx - 4]))
        {
            out[start_idx] = (c5 as i8) * 100;
        }

        body_long_period_total += candle_range(open[start_idx - 4], close[start_idx - 4])
            - candle_range(
                open[body_long_trailing_idx - 4],
                close[body_long_trailing_idx - 4],
            );

        start_idx += 1;
        body_long_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlclosingmarubozu(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_long_period = 10;
    let shadow_very_short_period = 10;
    let lookback_total = body_long_period.max(shadow_very_short_period);

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type.clone(),
        });
    }

    let mut out = vec![0i8; size];
    let mut body_long_period_total = 0.0;
    let mut shadow_very_short_period_total = 0.0;

    #[inline(always)]
    fn lower_shadow(o: f64, l: f64, c: f64) -> f64 {
        if c >= o {
            o - l
        } else {
            c - l
        }
    }

    #[inline(always)]
    fn upper_shadow(o: f64, h: f64, c: f64) -> f64 {
        if c >= o {
            h - c
        } else {
            h - o
        }
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    let mut start_idx = lookback_total;
    let mut body_long_trailing_idx = start_idx.saturating_sub(body_long_period);
    let mut shadow_very_short_trailing_idx = start_idx.saturating_sub(shadow_very_short_period);

    let mut i = body_long_trailing_idx;
    while i < start_idx {
        body_long_period_total += real_body(open[i], close[i]);
        i += 1;
    }

    i = shadow_very_short_trailing_idx;
    while i < start_idx {
        let color = candle_color(open[i], close[i]);
        shadow_very_short_period_total += if color == 1 {
            upper_shadow(open[i], high[i], close[i])
        } else {
            lower_shadow(open[i], low[i], close[i])
        };
        i += 1;
    }

    while start_idx < size {
        let color = candle_color(open[start_idx], close[start_idx]);
        if real_body(open[start_idx], close[start_idx])
            > candle_average(body_long_period_total, body_long_period)
            && ((color == 1
                && upper_shadow(open[start_idx], high[start_idx], close[start_idx])
                    < candle_average(shadow_very_short_period_total, shadow_very_short_period))
                || (color == -1
                    && lower_shadow(open[start_idx], low[start_idx], close[start_idx])
                        < candle_average(shadow_very_short_period_total, shadow_very_short_period)))
        {
            out[start_idx] = (color as i8) * 100;
        }

        body_long_period_total += real_body(open[start_idx], close[start_idx])
            - real_body(open[body_long_trailing_idx], close[body_long_trailing_idx]);

        let trailing_color = candle_color(open[start_idx], close[start_idx]);
        let new_shadow = if trailing_color == 1 {
            upper_shadow(open[start_idx], high[start_idx], close[start_idx])
        } else {
            lower_shadow(open[start_idx], low[start_idx], close[start_idx])
        };

        let old_color = candle_color(
            open[shadow_very_short_trailing_idx],
            close[shadow_very_short_trailing_idx],
        );
        let old_shadow = if old_color == 1 {
            upper_shadow(
                open[shadow_very_short_trailing_idx],
                high[shadow_very_short_trailing_idx],
                close[shadow_very_short_trailing_idx],
            )
        } else {
            lower_shadow(
                open[shadow_very_short_trailing_idx],
                low[shadow_very_short_trailing_idx],
                close[shadow_very_short_trailing_idx],
            )
        };

        shadow_very_short_period_total += new_shadow - old_shadow;

        start_idx += 1;
        body_long_trailing_idx += 1;
        shadow_very_short_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlconcealbabyswall(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let shadow_very_short_period = 10;
    let lookback_total = 3 + shadow_very_short_period;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type.clone(),
        });
    }

    let mut out = vec![0i8; size];
    let mut shadow_very_short_period_total = [0.0; 4];

    #[inline(always)]
    fn upper_shadow(o: f64, h: f64, c: f64) -> f64 {
        if c >= o {
            h - c
        } else {
            h - o
        }
    }

    #[inline(always)]
    fn lower_shadow(o: f64, l: f64, c: f64) -> f64 {
        if c >= o {
            o - l
        } else {
            c - l
        }
    }

    #[inline(always)]
    fn candle_color(o: f64, c: f64) -> i32 {
        if c >= o {
            1
        } else {
            -1
        }
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    #[inline(always)]
    fn real_body_gap_down(o1: f64, c1: f64, o2: f64, c2: f64) -> bool {
        o1.max(c1) < o2.min(c2)
    }

    let mut start_idx = lookback_total;
    let mut shadow_very_short_trailing_idx = start_idx.saturating_sub(shadow_very_short_period);

    let mut i = shadow_very_short_trailing_idx;
    while i < start_idx {
        shadow_very_short_period_total[3] += upper_shadow(open[i - 3], high[i - 3], close[i - 3])
            .max(lower_shadow(open[i - 3], low[i - 3], close[i - 3]));
        shadow_very_short_period_total[2] += upper_shadow(open[i - 2], high[i - 2], close[i - 2])
            .max(lower_shadow(open[i - 2], low[i - 2], close[i - 2]));
        shadow_very_short_period_total[1] += upper_shadow(open[i - 1], high[i - 1], close[i - 1])
            .max(lower_shadow(open[i - 1], low[i - 1], close[i - 1]));
        i += 1;
    }

    while start_idx < size {
        let c1 = candle_color(open[start_idx - 3], close[start_idx - 3]);
        let c2 = candle_color(open[start_idx - 2], close[start_idx - 2]);
        let c3 = candle_color(open[start_idx - 1], close[start_idx - 1]);
        let c4 = candle_color(open[start_idx], close[start_idx]);

        let upper1 = upper_shadow(
            open[start_idx - 3],
            high[start_idx - 3],
            close[start_idx - 3],
        );
        let lower1 = lower_shadow(
            open[start_idx - 3],
            low[start_idx - 3],
            close[start_idx - 3],
        );
        let upper2 = upper_shadow(
            open[start_idx - 2],
            high[start_idx - 2],
            close[start_idx - 2],
        );
        let lower2 = lower_shadow(
            open[start_idx - 2],
            low[start_idx - 2],
            close[start_idx - 2],
        );
        let upper3 = upper_shadow(
            open[start_idx - 1],
            high[start_idx - 1],
            close[start_idx - 1],
        );

        if c1 == -1
            && c2 == -1
            && c3 == -1
            && c4 == -1
            && lower1 < candle_average(shadow_very_short_period_total[3], shadow_very_short_period)
            && upper1 < candle_average(shadow_very_short_period_total[3], shadow_very_short_period)
            && lower2 < candle_average(shadow_very_short_period_total[2], shadow_very_short_period)
            && upper2 < candle_average(shadow_very_short_period_total[2], shadow_very_short_period)
            && real_body_gap_down(
                open[start_idx - 1],
                close[start_idx - 1],
                open[start_idx - 2],
                close[start_idx - 2],
            )
            && upper3 > candle_average(shadow_very_short_period_total[1], shadow_very_short_period)
            && high[start_idx - 1] > close[start_idx - 2]
            && high[start_idx] > high[start_idx - 1]
            && low[start_idx] < low[start_idx - 1]
        {
            out[start_idx] = 100;
        }

        for tot_idx in (1..=3).rev() {
            let current_upper = upper_shadow(
                open[start_idx - tot_idx],
                high[start_idx - tot_idx],
                close[start_idx - tot_idx],
            );
            let current_lower = lower_shadow(
                open[start_idx - tot_idx],
                low[start_idx - tot_idx],
                close[start_idx - tot_idx],
            );
            let new_val = current_upper.max(current_lower);

            let old_upper = upper_shadow(
                open[shadow_very_short_trailing_idx - tot_idx],
                high[shadow_very_short_trailing_idx - tot_idx],
                close[shadow_very_short_trailing_idx - tot_idx],
            );
            let old_lower = lower_shadow(
                open[shadow_very_short_trailing_idx - tot_idx],
                low[shadow_very_short_trailing_idx - tot_idx],
                close[shadow_very_short_trailing_idx - tot_idx],
            );
            let old_val = old_upper.max(old_lower);

            shadow_very_short_period_total[tot_idx] += new_val - old_val;
        }

        start_idx += 1;
        shadow_very_short_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlcounterattack(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_long_period = 10;
    let equal_period = 10;
    let lookback_total = 1 + body_long_period.max(equal_period);

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type.clone(),
        });
    }

    let mut out = vec![0i8; size];
    let mut equal_period_total = 0.0;
    let mut body_long_period_total = [0.0; 2];

    #[inline(always)]
    fn real_body(o: f64, c: f64) -> f64 {
        (c - o).abs()
    }

    #[inline(always)]
    fn candle_color(o: f64, c: f64) -> i32 {
        if c >= o {
            1
        } else {
            -1
        }
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    let mut start_idx = lookback_total;
    let mut equal_trailing_idx = start_idx.saturating_sub(equal_period);
    let mut body_long_trailing_idx = start_idx.saturating_sub(body_long_period);

    let mut i = equal_trailing_idx;
    while i < start_idx {
        equal_period_total += real_body(open[i - 1], close[i - 1]);
        i += 1;
    }

    i = body_long_trailing_idx;
    while i < start_idx {
        body_long_period_total[1] += real_body(open[i - 1], close[i - 1]);
        body_long_period_total[0] += real_body(open[i], close[i]);
        i += 1;
    }

    while start_idx < size {
        let c1 = candle_color(open[start_idx - 1], close[start_idx - 1]);
        let c2 = candle_color(open[start_idx], close[start_idx]);
        let rb1 = real_body(open[start_idx - 1], close[start_idx - 1]);
        let rb2 = real_body(open[start_idx], close[start_idx]);
        let eq_avg = candle_average(equal_period_total, equal_period);
        let body1_avg = candle_average(body_long_period_total[1], body_long_period);
        let body2_avg = candle_average(body_long_period_total[0], body_long_period);

        if c1 == -c2
            && rb1 > body1_avg
            && rb2 > body2_avg
            && close[start_idx] <= close[start_idx - 1] + eq_avg
            && close[start_idx] >= close[start_idx - 1] - eq_avg
        {
            out[start_idx] = (c2 as i8) * 100;
        }

        equal_period_total += real_body(open[start_idx - 1], close[start_idx - 1])
            - real_body(open[equal_trailing_idx - 1], close[equal_trailing_idx - 1]);

        for tot_idx in (0..=1).rev() {
            body_long_period_total[tot_idx] +=
                real_body(open[start_idx - tot_idx], close[start_idx - tot_idx])
                    - real_body(
                        open[body_long_trailing_idx - tot_idx],
                        close[body_long_trailing_idx - tot_idx],
                    );
        }

        start_idx += 1;
        equal_trailing_idx += 1;
        body_long_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdldarkcloudcover(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, high, _, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_long_period = 10;
    let penetration = if input.params.penetration == 0.0 {
        0.5
    } else {
        input.params.penetration
    };
    let lookback_total = 1 + body_long_period;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type.clone(),
        });
    }

    let mut out = vec![0i8; size];
    let mut body_long_period_total = 0.0;

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    let mut start_idx = lookback_total;
    let mut body_long_trailing_idx = start_idx.saturating_sub(body_long_period);

    let mut i = body_long_trailing_idx;
    while i < start_idx {
        body_long_period_total += real_body(open[i - 1], close[i - 1]);
        i += 1;
    }

    while start_idx < size {
        if candle_color(open[start_idx - 1], close[start_idx - 1]) == 1
            && real_body(open[start_idx - 1], close[start_idx - 1])
                > candle_average(body_long_period_total, body_long_period)
            && candle_color(open[start_idx], close[start_idx]) == -1
            && open[start_idx] > high[start_idx - 1]
            && close[start_idx] > open[start_idx - 1]
            && close[start_idx]
                < close[start_idx - 1]
                    - real_body(open[start_idx - 1], close[start_idx - 1]) * penetration
        {
            out[start_idx] = -100;
        }

        body_long_period_total += real_body(open[start_idx - 1], close[start_idx - 1])
            - real_body(
                open[body_long_trailing_idx - 1],
                close[body_long_trailing_idx - 1],
            );

        start_idx += 1;
        body_long_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdldoji(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, _, _, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_doji_period = 10;
    let lookback_total = body_doji_period;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type.clone(),
        });
    }

    let mut out = vec![0i8; size];
    let mut body_doji_period_total = 0.0;

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    let mut start_idx = lookback_total;
    let mut body_doji_trailing_idx = start_idx.saturating_sub(body_doji_period);

    let mut i = body_doji_trailing_idx;
    while i < start_idx {
        body_doji_period_total += candle_range(open[i], close[i]);
        i += 1;
    }

    while start_idx < size {
        let avg_body = candle_average(body_doji_period_total, body_doji_period);
        if real_body(open[start_idx], close[start_idx]) <= avg_body {
            out[start_idx] = 100;
        }

        body_doji_period_total += candle_range(open[start_idx], close[start_idx])
            - candle_range(open[body_doji_trailing_idx], close[body_doji_trailing_idx]);

        start_idx += 1;
        body_doji_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdldojistar(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_long_period = 10;
    let body_doji_period = 10;
    let lookback_total = 1 + body_long_period.max(body_doji_period);

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type.clone(),
        });
    }

    let mut out = vec![0i8; size];
    let mut body_long_period_total = 0.0;
    let mut body_doji_period_total = 0.0;

    #[inline(always)]
    fn gap_up(current_open: f64, current_close: f64, prev_open: f64, prev_close: f64) -> bool {
        current_open.min(current_close) > prev_open.max(prev_close)
    }

    #[inline(always)]
    fn gap_down(current_open: f64, current_close: f64, prev_open: f64, prev_close: f64) -> bool {
        current_open.max(current_close) < prev_open.min(prev_close)
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    let mut start_idx = lookback_total;
    let mut body_long_trailing_idx = start_idx.saturating_sub(1 + body_long_period);
    let mut body_doji_trailing_idx = start_idx.saturating_sub(body_doji_period);

    let mut i = body_long_trailing_idx;
    while i < start_idx - 1 {
        body_long_period_total += real_body(open[i], close[i]);
        i += 1;
    }

    i = body_doji_trailing_idx;
    while i < start_idx {
        body_doji_period_total += real_body(open[i], close[i]);
        i += 1;
    }

    while start_idx < size {
        if real_body(open[start_idx - 1], close[start_idx - 1])
            > candle_average(body_long_period_total, body_long_period)
            && real_body(open[start_idx], close[start_idx])
                <= candle_average(body_doji_period_total, body_doji_period)
            && ((candle_color(open[start_idx - 1], close[start_idx - 1]) == 1
                && gap_up(
                    open[start_idx],
                    close[start_idx],
                    open[start_idx - 1],
                    close[start_idx - 1],
                ))
                || (candle_color(open[start_idx - 1], close[start_idx - 1]) == -1
                    && gap_down(
                        open[start_idx],
                        close[start_idx],
                        open[start_idx - 1],
                        close[start_idx - 1],
                    )))
        {
            out[start_idx] = -candle_color(open[start_idx - 1], close[start_idx - 1]) as i8 * 100;
        }

        body_long_period_total += real_body(open[start_idx - 1], close[start_idx - 1])
            - real_body(open[body_long_trailing_idx], close[body_long_trailing_idx]);

        body_doji_period_total += real_body(open[start_idx], close[start_idx])
            - real_body(open[body_doji_trailing_idx], close[body_doji_trailing_idx]);

        start_idx += 1;
        body_long_trailing_idx += 1;
        body_doji_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdldragonflydoji(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_doji_period = 10;
    let shadow_very_short_period = 10;
    let lookback_total = body_doji_period.max(shadow_very_short_period);

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type.clone(),
        });
    }

    let mut out = vec![0i8; size];
    let mut body_doji_period_total = 0.0;
    let mut shadow_very_short_period_total = 0.0;

    #[inline(always)]
    fn upper_shadow(o: f64, h: f64, c: f64) -> f64 {
        if c >= o {
            h - c
        } else {
            h - o
        }
    }

    #[inline(always)]
    fn lower_shadow(o: f64, l: f64, c: f64) -> f64 {
        if c >= o {
            o - l
        } else {
            c - l
        }
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    let mut start_idx = lookback_total;
    let mut body_doji_trailing_idx = start_idx.saturating_sub(body_doji_period);
    let mut shadow_very_short_trailing_idx = start_idx.saturating_sub(shadow_very_short_period);

    let mut i = body_doji_trailing_idx;
    while i < start_idx {
        body_doji_period_total += candle_range(open[i], close[i]);
        i += 1;
    }

    i = shadow_very_short_trailing_idx;
    while i < start_idx {
        shadow_very_short_period_total +=
            (upper_shadow(open[i], high[i], close[i])).max(lower_shadow(open[i], low[i], close[i]));
        i += 1;
    }

    while start_idx < size {
        let rb = real_body(open[start_idx], close[start_idx]);
        let us = upper_shadow(open[start_idx], high[start_idx], close[start_idx]);
        let ls = lower_shadow(open[start_idx], low[start_idx], close[start_idx]);
        let avg_body_doji = candle_average(body_doji_period_total, body_doji_period);
        let avg_shadow_very_short =
            candle_average(shadow_very_short_period_total, shadow_very_short_period);

        if rb <= avg_body_doji && us < avg_shadow_very_short && ls > avg_shadow_very_short {
            out[start_idx] = 100;
        }

        body_doji_period_total += candle_range(open[start_idx], close[start_idx])
            - candle_range(open[body_doji_trailing_idx], close[body_doji_trailing_idx]);

        let current_shadow_sum = (upper_shadow(open[start_idx], high[start_idx], close[start_idx]))
            .max(lower_shadow(
                open[start_idx],
                low[start_idx],
                close[start_idx],
            ));
        let trailing_shadow_sum = (upper_shadow(
            open[shadow_very_short_trailing_idx],
            high[shadow_very_short_trailing_idx],
            close[shadow_very_short_trailing_idx],
        ))
        .max(lower_shadow(
            open[shadow_very_short_trailing_idx],
            low[shadow_very_short_trailing_idx],
            close[shadow_very_short_trailing_idx],
        ));

        shadow_very_short_period_total += current_shadow_sum - trailing_shadow_sum;

        start_idx += 1;
        body_doji_trailing_idx += 1;
        shadow_very_short_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlengulfing(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, _, _, close) = input_ohlc(&input.data)?;

    let size = open.len();
    if size < 2 {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type.clone(),
        });
    }

    let mut out = vec![0i8; size];
    for i in 1..size {
        let c1 = candle_color(open[i - 1], close[i - 1]);
        let c2 = candle_color(open[i], close[i]);
        if (c2 == 1
            && c1 == -1
            && ((close[i] >= open[i - 1] && open[i] < close[i - 1])
                || (close[i] > open[i - 1] && open[i] <= close[i - 1])))
            || (c2 == -1
                && c1 == 1
                && ((open[i] >= close[i - 1] && close[i] < open[i - 1])
                    || (open[i] > close[i - 1] && close[i] <= open[i - 1])))
        {
            if (open[i] - close[i - 1]).abs() > f64::EPSILON
                && (close[i] - open[i - 1]).abs() > f64::EPSILON
            {
                out[i] = (c2 as i8) * 100;
            } else {
                out[i] = (c2 as i8) * 80;
            }
        }
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdleveningdojistar(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, _, _, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_long_period = 10;
    let body_doji_period = 10;
    let body_short_period = 10;
    let penetration = if input.params.penetration == 0.0 {
        0.3
    } else {
        input.params.penetration
    };
    let lookback_total = 2 + body_long_period
        .max(body_doji_period)
        .max(body_short_period);

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type.clone(),
        });
    }

    let mut out = vec![0i8; size];
    let mut body_long_period_total = 0.0;
    let mut body_doji_period_total = 0.0;
    let mut body_short_period_total = 0.0;

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    #[inline(always)]
    fn gap_up(current_open: f64, current_close: f64, prev_open: f64, prev_close: f64) -> bool {
        current_open.min(current_close) > prev_open.max(prev_close)
    }

    let mut start_idx = lookback_total;
    let mut body_long_trailing_idx = start_idx.saturating_sub(2 + body_long_period);
    let mut body_doji_trailing_idx = start_idx.saturating_sub(1 + body_doji_period);
    let mut body_short_trailing_idx = start_idx.saturating_sub(body_short_period);

    let mut i = body_long_trailing_idx;
    while i < start_idx - 2 {
        body_long_period_total += real_body(open[i], close[i]);
        i += 1;
    }

    i = body_doji_trailing_idx;
    while i < start_idx - 1 {
        body_doji_period_total += real_body(open[i], close[i]);
        i += 1;
    }

    i = body_short_trailing_idx;
    while i < start_idx {
        body_short_period_total += real_body(open[i], close[i]);
        i += 1;
    }

    while start_idx < size {
        if real_body(open[start_idx - 2], close[start_idx - 2])
            > candle_average(body_long_period_total, body_long_period)
            && candle_color(open[start_idx - 2], close[start_idx - 2]) == 1
            && real_body(open[start_idx - 1], close[start_idx - 1])
                <= candle_average(body_doji_period_total, body_doji_period)
            && gap_up(
                open[start_idx - 1],
                close[start_idx - 1],
                open[start_idx - 2],
                close[start_idx - 2],
            )
            && real_body(open[start_idx], close[start_idx])
                > candle_average(body_short_period_total, body_short_period)
            && candle_color(open[start_idx], close[start_idx]) == -1
            && close[start_idx]
                < close[start_idx - 2]
                    - real_body(open[start_idx - 2], close[start_idx - 2]) * penetration
        {
            out[start_idx] = -100;
        }

        body_long_period_total += real_body(open[start_idx - 2], close[start_idx - 2])
            - real_body(open[body_long_trailing_idx], close[body_long_trailing_idx]);

        body_doji_period_total += real_body(open[start_idx - 1], close[start_idx - 1])
            - real_body(open[body_doji_trailing_idx], close[body_doji_trailing_idx]);

        body_short_period_total += real_body(open[start_idx], close[start_idx])
            - real_body(
                open[body_short_trailing_idx],
                close[body_short_trailing_idx],
            );

        start_idx += 1;
        body_long_trailing_idx += 1;
        body_doji_trailing_idx += 1;
        body_short_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdleveningstar(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, _, _, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_long_period = 10;
    let body_short_period = 10;
    let penetration = if input.params.penetration == 0.0 {
        0.3
    } else {
        input.params.penetration
    };
    let lookback_total = 2 + body_long_period.max(body_short_period);

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    let mut out = vec![0i8; size];
    let mut body_long_period_total = 0.0;
    let mut body_short_period_total = 0.0;
    let mut body_short_period_total2 = 0.0;

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    #[inline(always)]
    fn gap_up(current_open: f64, current_close: f64, prev_open: f64, prev_close: f64) -> bool {
        current_open.min(current_close) > prev_open.max(prev_close)
    }

    let mut start_idx = lookback_total;
    let mut body_long_trailing_idx = start_idx.saturating_sub(2 + body_long_period);
    let mut body_short_trailing_idx = start_idx.saturating_sub(1 + body_short_period);

    let mut i = body_long_trailing_idx;
    while i < start_idx - 2 {
        body_long_period_total += real_body(open[i], close[i]);
        i += 1;
    }

    i = body_short_trailing_idx;
    while i < start_idx - 1 {
        body_short_period_total += real_body(open[i], close[i]);
        body_short_period_total2 += real_body(open[i + 1], close[i + 1]);
        i += 1;
    }

    while start_idx < size {
        if real_body(open[start_idx - 2], close[start_idx - 2])
            > candle_average(body_long_period_total, body_long_period)
            && candle_color(open[start_idx - 2], close[start_idx - 2]) == 1
            && real_body(open[start_idx - 1], close[start_idx - 1])
                <= candle_average(body_short_period_total, body_short_period)
            && gap_up(
                open[start_idx - 1],
                close[start_idx - 1],
                open[start_idx - 2],
                close[start_idx - 2],
            )
            && real_body(open[start_idx], close[start_idx])
                > candle_average(body_short_period_total2, body_short_period)
            && candle_color(open[start_idx], close[start_idx]) == -1
            && close[start_idx]
                < close[start_idx - 2]
                    - real_body(open[start_idx - 2], close[start_idx - 2]) * penetration
        {
            out[start_idx] = -100;
        }

        body_long_period_total += real_body(open[start_idx - 2], close[start_idx - 2])
            - real_body(open[body_long_trailing_idx], close[body_long_trailing_idx]);
        body_short_period_total += real_body(open[start_idx - 1], close[start_idx - 1])
            - real_body(
                open[body_short_trailing_idx],
                close[body_short_trailing_idx],
            );
        body_short_period_total2 += real_body(open[start_idx], close[start_idx])
            - real_body(
                open[body_short_trailing_idx + 1],
                close[body_short_trailing_idx + 1],
            );

        start_idx += 1;
        body_long_trailing_idx += 1;
        body_short_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlmorningstar(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, _, _, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_long_period = 10;
    let body_short_period = 10;
    let penetration = if input.params.penetration == 0.0 {
        0.3
    } else {
        input.params.penetration
    };
    let lookback_total = 2 + body_long_period.max(body_short_period);

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    let mut out = vec![0i8; size];
    let mut body_long_period_total = 0.0;
    let mut body_short_period_total = 0.0;
    let mut body_short_period_total2 = 0.0;

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    #[inline(always)]
    fn gap_down(current_open: f64, current_close: f64, prev_open: f64, prev_close: f64) -> bool {
        current_open.max(current_close) < prev_open.min(prev_close)
    }

    let mut start_idx = lookback_total;
    let mut body_long_trailing_idx = start_idx.saturating_sub(2 + body_long_period);
    let mut body_short_trailing_idx = start_idx.saturating_sub(1 + body_short_period);

    let mut i = body_long_trailing_idx;
    while i < start_idx - 2 {
        body_long_period_total += real_body(open[i], close[i]);
        i += 1;
    }

    i = body_short_trailing_idx;
    while i < start_idx - 1 {
        body_short_period_total += real_body(open[i], close[i]);
        body_short_period_total2 += real_body(open[i + 1], close[i + 1]);
        i += 1;
    }

    while start_idx < size {
        if real_body(open[start_idx - 2], close[start_idx - 2])
            > candle_average(body_long_period_total, body_long_period)
            && candle_color(open[start_idx - 2], close[start_idx - 2]) == -1
            && real_body(open[start_idx - 1], close[start_idx - 1])
                <= candle_average(body_short_period_total, body_short_period)
            && gap_down(
                open[start_idx - 1],
                close[start_idx - 1],
                open[start_idx - 2],
                close[start_idx - 2],
            )
            && real_body(open[start_idx], close[start_idx])
                > candle_average(body_short_period_total2, body_short_period)
            && candle_color(open[start_idx], close[start_idx]) == 1
            && close[start_idx]
                > close[start_idx - 2]
                    + real_body(open[start_idx - 2], close[start_idx - 2]) * penetration
        {
            out[start_idx] = 100;
        }

        body_long_period_total += real_body(open[start_idx - 2], close[start_idx - 2])
            - real_body(open[body_long_trailing_idx], close[body_long_trailing_idx]);
        body_short_period_total += real_body(open[start_idx - 1], close[start_idx - 1])
            - real_body(
                open[body_short_trailing_idx],
                close[body_short_trailing_idx],
            );
        body_short_period_total2 += real_body(open[start_idx], close[start_idx])
            - real_body(
                open[body_short_trailing_idx + 1],
                close[body_short_trailing_idx + 1],
            );

        start_idx += 1;
        body_long_trailing_idx += 1;
        body_short_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlgravestonedoji(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_doji_period = 10;
    let shadow_very_short_period = 10;
    let lookback_total = body_doji_period.max(shadow_very_short_period);

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    let mut out = vec![0i8; size];
    let mut body_doji_period_total = 0.0;
    let mut shadow_very_short_period_total = 0.0;

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    let mut start_idx = lookback_total;
    let mut body_doji_trailing_idx = start_idx.saturating_sub(body_doji_period);
    let mut shadow_very_short_trailing_idx = start_idx.saturating_sub(shadow_very_short_period);

    let mut i = body_doji_trailing_idx;
    while i < start_idx {
        body_doji_period_total += candle_range(open[i], close[i]);
        i += 1;
    }

    i = shadow_very_short_trailing_idx;
    while i < start_idx {
        shadow_very_short_period_total += upper_shadow(open[i], high[i], close[i]);
        i += 1;
    }

    while start_idx < size {
        let avg_body_doji = candle_average(body_doji_period_total, body_doji_period);
        let avg_shadow_very_short =
            candle_average(shadow_very_short_period_total, shadow_very_short_period);
        let rb = real_body(open[start_idx], close[start_idx]);
        let ls = lower_shadow(open[start_idx], low[start_idx], close[start_idx]);
        let us = upper_shadow(open[start_idx], high[start_idx], close[start_idx]);

        if rb <= avg_body_doji && ls < avg_shadow_very_short && us > avg_shadow_very_short {
            out[start_idx] = 100;
        }

        body_doji_period_total += candle_range(open[start_idx], close[start_idx])
            - candle_range(open[body_doji_trailing_idx], close[body_doji_trailing_idx]);
        shadow_very_short_period_total +=
            upper_shadow(open[start_idx], high[start_idx], close[start_idx])
                - upper_shadow(
                    open[shadow_very_short_trailing_idx],
                    high[shadow_very_short_trailing_idx],
                    close[shadow_very_short_trailing_idx],
                );

        start_idx += 1;
        body_doji_trailing_idx += 1;
        shadow_very_short_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlhammer(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_short_period = 10;
    let shadow_long_period = 10;
    let shadow_very_short_period = 10;
    let near_period = 10;
    let lookback_total = body_short_period
        .max(shadow_long_period)
        .max(shadow_very_short_period)
        .max(near_period)
        + 1;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    let mut out = vec![0i8; size];
    let mut body_period_total = 0.0;
    let mut shadow_long_period_total = 0.0;
    let mut shadow_very_short_period_total = 0.0;
    let mut near_period_total = 0.0;

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    let mut start_idx = lookback_total;
    let mut body_trailing_idx = start_idx.saturating_sub(body_short_period);
    let mut shadow_long_trailing_idx = start_idx.saturating_sub(shadow_long_period);
    let mut shadow_very_short_trailing_idx = start_idx.saturating_sub(shadow_very_short_period);
    let mut near_trailing_idx = start_idx.saturating_sub(1 + near_period);

    let mut i = body_trailing_idx;
    while i < start_idx {
        body_period_total += candle_range(open[i], close[i]);
        i += 1;
    }
    i = shadow_long_trailing_idx;
    while i < start_idx {
        shadow_long_period_total += lower_shadow(open[i], low[i], close[i]);
        i += 1;
    }
    i = shadow_very_short_trailing_idx;
    while i < start_idx {
        shadow_very_short_period_total += upper_shadow(open[i], high[i], close[i]);
        i += 1;
    }
    i = near_trailing_idx;
    while i < start_idx - 1 {
        near_period_total += candle_range(open[i], close[i]);
        i += 1;
    }

    while start_idx < size {
        let rb = real_body(open[start_idx], close[start_idx]);
        let ls = lower_shadow(open[start_idx], low[start_idx], close[start_idx]);
        let us = upper_shadow(open[start_idx], high[start_idx], close[start_idx]);
        let rb_low = open[start_idx].min(close[start_idx]);
        if rb < candle_average(body_period_total, body_short_period)
            && ls > candle_average(shadow_long_period_total, shadow_long_period)
            && us < candle_average(shadow_very_short_period_total, shadow_very_short_period)
            && rb_low <= low[start_idx - 1] + candle_average(near_period_total, near_period)
        {
            out[start_idx] = 100;
        }

        body_period_total += candle_range(open[start_idx], close[start_idx])
            - candle_range(open[body_trailing_idx], close[body_trailing_idx]);
        shadow_long_period_total += lower_shadow(open[start_idx], low[start_idx], close[start_idx])
            - lower_shadow(
                open[shadow_long_trailing_idx],
                low[shadow_long_trailing_idx],
                close[shadow_long_trailing_idx],
            );
        shadow_very_short_period_total +=
            upper_shadow(open[start_idx], high[start_idx], close[start_idx])
                - upper_shadow(
                    open[shadow_very_short_trailing_idx],
                    high[shadow_very_short_trailing_idx],
                    close[shadow_very_short_trailing_idx],
                );
        near_period_total += candle_range(open[start_idx - 1], close[start_idx - 1])
            - candle_range(open[near_trailing_idx], close[near_trailing_idx]);

        start_idx += 1;
        body_trailing_idx += 1;
        shadow_long_trailing_idx += 1;
        shadow_very_short_trailing_idx += 1;
        near_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlhangingman(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_short_period = 10;
    let shadow_long_period = 10;
    let shadow_very_short_period = 10;
    let near_period = 10;
    let lookback_total = body_short_period
        .max(shadow_long_period)
        .max(shadow_very_short_period)
        .max(near_period)
        + 1;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    let mut out = vec![0i8; size];
    let mut body_period_total = 0.0;
    let mut shadow_long_period_total = 0.0;
    let mut shadow_very_short_period_total = 0.0;
    let mut near_period_total = 0.0;

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    let mut start_idx = lookback_total;
    let mut body_trailing_idx = start_idx.saturating_sub(body_short_period);
    let mut shadow_long_trailing_idx = start_idx.saturating_sub(shadow_long_period);
    let mut shadow_very_short_trailing_idx = start_idx.saturating_sub(shadow_very_short_period);
    let mut near_trailing_idx = start_idx.saturating_sub(1 + near_period);

    let mut i = body_trailing_idx;
    while i < start_idx {
        body_period_total += candle_range(open[i], close[i]);
        i += 1;
    }
    i = shadow_long_trailing_idx;
    while i < start_idx {
        shadow_long_period_total += lower_shadow(open[i], low[i], close[i]);
        i += 1;
    }
    i = shadow_very_short_trailing_idx;
    while i < start_idx {
        shadow_very_short_period_total += upper_shadow(open[i], high[i], close[i]);
        i += 1;
    }
    i = near_trailing_idx;
    while i < start_idx - 1 {
        near_period_total += candle_range(open[i], close[i]);
        i += 1;
    }

    while start_idx < size {
        let rb = real_body(open[start_idx], close[start_idx]);
        let ls = lower_shadow(open[start_idx], low[start_idx], close[start_idx]);
        let us = upper_shadow(open[start_idx], high[start_idx], close[start_idx]);
        let rb_low = open[start_idx].min(close[start_idx]);
        if rb < candle_average(body_period_total, body_short_period)
            && ls > candle_average(shadow_long_period_total, shadow_long_period)
            && us < candle_average(shadow_very_short_period_total, shadow_very_short_period)
            && rb_low >= high[start_idx - 1] - candle_average(near_period_total, near_period)
        {
            out[start_idx] = -100;
        }

        body_period_total += candle_range(open[start_idx], close[start_idx])
            - candle_range(open[body_trailing_idx], close[body_trailing_idx]);
        shadow_long_period_total += lower_shadow(open[start_idx], low[start_idx], close[start_idx])
            - lower_shadow(
                open[shadow_long_trailing_idx],
                low[shadow_long_trailing_idx],
                close[shadow_long_trailing_idx],
            );
        shadow_very_short_period_total +=
            upper_shadow(open[start_idx], high[start_idx], close[start_idx])
                - upper_shadow(
                    open[shadow_very_short_trailing_idx],
                    high[shadow_very_short_trailing_idx],
                    close[shadow_very_short_trailing_idx],
                );
        near_period_total += candle_range(open[start_idx - 1], close[start_idx - 1])
            - candle_range(open[near_trailing_idx], close[near_trailing_idx]);

        start_idx += 1;
        body_trailing_idx += 1;
        shadow_long_trailing_idx += 1;
        shadow_very_short_trailing_idx += 1;
        near_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlharami(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, _, _, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_long_period = 10;
    let body_short_period = 10;
    let lookback_total = body_long_period.max(body_short_period) + 1;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    let mut out = vec![0i8; size];
    let mut body_long_period_total = 0.0;
    let mut body_short_period_total = 0.0;
    let mut body_long_trailing_idx = lookback_total - 1 - body_long_period;
    let mut body_short_trailing_idx = lookback_total - body_short_period;
    let mut i = body_long_trailing_idx;
    while i < lookback_total - 1 {
        body_long_period_total += real_body(open[i], close[i]);
        i += 1;
    }
    i = body_short_trailing_idx;
    while i < lookback_total {
        body_short_period_total += real_body(open[i], close[i]);
        i += 1;
    }
    i = lookback_total;
    while i < size {
        if real_body(open[i - 1], close[i - 1])
            > candle_average(body_long_period_total, body_long_period)
            && real_body(open[i], close[i])
                <= candle_average(body_short_period_total, body_short_period)
        {
            let hi0 = open[i - 1].max(close[i - 1]);
            let lo0 = open[i - 1].min(close[i - 1]);
            let hi1 = open[i].max(close[i]);
            let lo1 = open[i].min(close[i]);
            let sign = -(candle_color(open[i - 1], close[i - 1]) as i8);
            if hi1 < hi0 && lo1 > lo0 {
                out[i] = sign * 100;
            } else if hi1 <= hi0 && lo1 >= lo0 {
                out[i] = sign * 80;
            }
        }

        body_long_period_total += real_body(open[i - 1], close[i - 1])
            - real_body(open[body_long_trailing_idx], close[body_long_trailing_idx]);
        body_short_period_total += real_body(open[i], close[i])
            - real_body(
                open[body_short_trailing_idx],
                close[body_short_trailing_idx],
            );
        i += 1;
        body_long_trailing_idx += 1;
        body_short_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlharamicross(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, _, _, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_long_period = 10;
    let body_doji_period = 10;
    let lookback_total = body_long_period.max(body_doji_period) + 1;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    let mut out = vec![0i8; size];
    let mut body_long_period_total = 0.0;
    let mut body_doji_period_total = 0.0;
    let mut body_long_trailing_idx = lookback_total - 1 - body_long_period;
    let mut body_doji_trailing_idx = lookback_total - body_doji_period;
    let mut i = body_long_trailing_idx;
    while i < lookback_total - 1 {
        body_long_period_total += real_body(open[i], close[i]);
        i += 1;
    }
    i = body_doji_trailing_idx;
    while i < lookback_total {
        body_doji_period_total += real_body(open[i], close[i]);
        i += 1;
    }
    i = lookback_total;
    while i < size {
        if real_body(open[i - 1], close[i - 1])
            > candle_average(body_long_period_total, body_long_period)
            && real_body(open[i], close[i])
                <= candle_average(body_doji_period_total, body_doji_period)
        {
            let hi0 = open[i - 1].max(close[i - 1]);
            let lo0 = open[i - 1].min(close[i - 1]);
            let hi1 = open[i].max(close[i]);
            let lo1 = open[i].min(close[i]);
            let sign = -(candle_color(open[i - 1], close[i - 1]) as i8);
            if hi1 < hi0 && lo1 > lo0 {
                out[i] = sign * 100;
            } else if hi1 <= hi0 && lo1 >= lo0 {
                out[i] = sign * 80;
            }
        }

        body_long_period_total += real_body(open[i - 1], close[i - 1])
            - real_body(open[body_long_trailing_idx], close[body_long_trailing_idx]);
        body_doji_period_total += real_body(open[i], close[i])
            - real_body(open[body_doji_trailing_idx], close[body_doji_trailing_idx]);
        i += 1;
        body_long_trailing_idx += 1;
        body_doji_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlhighwave(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_short_period = 10;
    let shadow_very_long_period = 10;
    let lookback_total = body_short_period.max(shadow_very_long_period);

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    let mut out = vec![0i8; size];
    let mut body_period_total = 0.0;
    let mut shadow_period_total = 0.0;
    let mut body_trailing_idx = lookback_total - body_short_period;
    let mut shadow_trailing_idx = lookback_total - shadow_very_long_period;

    let mut i = body_trailing_idx;
    while i < lookback_total {
        body_period_total += real_body(open[i], close[i]);
        i += 1;
    }
    i = shadow_trailing_idx;
    while i < lookback_total {
        shadow_period_total += upper_shadow(open[i], high[i], close[i]);
        i += 1;
    }

    i = lookback_total;
    while i < size {
        let rb = real_body(open[i], close[i]);
        let us = upper_shadow(open[i], high[i], close[i]);
        let ls = lower_shadow(open[i], low[i], close[i]);
        if rb < candle_average(body_period_total, body_short_period)
            && us > candle_average(shadow_period_total, shadow_very_long_period)
            && ls > candle_average(shadow_period_total, shadow_very_long_period)
        {
            out[i] = (candle_color(open[i], close[i]) as i8) * 100;
        }

        body_period_total += real_body(open[i], close[i])
            - real_body(open[body_trailing_idx], close[body_trailing_idx]);
        shadow_period_total += upper_shadow(open[i], high[i], close[i])
            - upper_shadow(
                open[shadow_trailing_idx],
                high[shadow_trailing_idx],
                close[shadow_trailing_idx],
            );
        i += 1;
        body_trailing_idx += 1;
        shadow_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlinvertedhammer(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_short_period = 10;
    let shadow_long_period = 10;
    let shadow_very_short_period = 10;
    let lookback_total = body_short_period
        .max(shadow_long_period)
        .max(shadow_very_short_period)
        + 1;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    #[inline(always)]
    fn real_body_gap_down(
        curr_open: f64,
        curr_close: f64,
        prev_open: f64,
        prev_close: f64,
    ) -> bool {
        curr_open.max(curr_close) < prev_open.min(prev_close)
    }

    let mut out = vec![0i8; size];
    let mut body_period_total = 0.0;
    let mut shadow_long_period_total = 0.0;
    let mut shadow_very_short_period_total = 0.0;
    let mut body_trailing_idx = lookback_total - 1 - body_short_period;
    let mut shadow_long_trailing_idx = lookback_total - 1 - shadow_long_period;
    let mut shadow_very_short_trailing_idx = lookback_total - 1 - shadow_very_short_period;

    let mut i = body_trailing_idx;
    while i < lookback_total - 1 {
        body_period_total += real_body(open[i], close[i]);
        i += 1;
    }
    i = shadow_long_trailing_idx;
    while i < lookback_total - 1 {
        shadow_long_period_total += upper_shadow(open[i], high[i], close[i]);
        i += 1;
    }
    i = shadow_very_short_trailing_idx;
    while i < lookback_total - 1 {
        shadow_very_short_period_total += lower_shadow(open[i], low[i], close[i]);
        i += 1;
    }

    i = lookback_total;
    while i < size {
        if real_body(open[i], close[i]) < candle_average(body_period_total, body_short_period)
            && upper_shadow(open[i], high[i], close[i])
                > candle_average(shadow_long_period_total, shadow_long_period)
            && lower_shadow(open[i], low[i], close[i])
                < candle_average(shadow_very_short_period_total, shadow_very_short_period)
            && real_body_gap_down(open[i], close[i], open[i - 1], close[i - 1])
        {
            out[i] = 100;
        }

        body_period_total += real_body(open[i], close[i])
            - real_body(open[body_trailing_idx], close[body_trailing_idx]);
        shadow_long_period_total += upper_shadow(open[i], high[i], close[i])
            - upper_shadow(
                open[shadow_long_trailing_idx],
                high[shadow_long_trailing_idx],
                close[shadow_long_trailing_idx],
            );
        shadow_very_short_period_total += lower_shadow(open[i], low[i], close[i])
            - lower_shadow(
                open[shadow_very_short_trailing_idx],
                low[shadow_very_short_trailing_idx],
                close[shadow_very_short_trailing_idx],
            );

        i += 1;
        body_trailing_idx += 1;
        shadow_long_trailing_idx += 1;
        shadow_very_short_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdllongleggeddoji(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_doji_period = 10;
    let shadow_long_period = 10;
    let lookback_total = body_doji_period.max(shadow_long_period);

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    let mut out = vec![0i8; size];
    let mut body_doji_period_total = 0.0;
    let mut shadow_long_period_total = 0.0;
    let mut body_doji_trailing_idx = lookback_total - body_doji_period;
    let mut shadow_long_trailing_idx = lookback_total - shadow_long_period;

    let mut i = body_doji_trailing_idx;
    while i < lookback_total {
        body_doji_period_total += real_body(open[i], close[i]);
        i += 1;
    }
    i = shadow_long_trailing_idx;
    while i < lookback_total {
        shadow_long_period_total += upper_shadow(open[i], high[i], close[i]);
        i += 1;
    }

    i = lookback_total;
    while i < size {
        let rb = real_body(open[i], close[i]);
        let us = upper_shadow(open[i], high[i], close[i]);
        let ls = lower_shadow(open[i], low[i], close[i]);
        if rb <= candle_average(body_doji_period_total, body_doji_period)
            && (ls > candle_average(shadow_long_period_total, shadow_long_period)
                || us > candle_average(shadow_long_period_total, shadow_long_period))
        {
            out[i] = 100;
        }

        body_doji_period_total += real_body(open[i], close[i])
            - real_body(open[body_doji_trailing_idx], close[body_doji_trailing_idx]);
        shadow_long_period_total += upper_shadow(open[i], high[i], close[i])
            - upper_shadow(
                open[shadow_long_trailing_idx],
                high[shadow_long_trailing_idx],
                close[shadow_long_trailing_idx],
            );

        i += 1;
        body_doji_trailing_idx += 1;
        shadow_long_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdllongline(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_long_period = 10;
    let shadow_short_period = 10;
    let lookback_total = body_long_period.max(shadow_short_period);

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    let mut out = vec![0i8; size];
    let mut body_period_total = 0.0;
    let mut shadow_period_total = 0.0;
    let mut body_trailing_idx = lookback_total - body_long_period;
    let mut shadow_trailing_idx = lookback_total - shadow_short_period;

    let mut i = body_trailing_idx;
    while i < lookback_total {
        body_period_total += real_body(open[i], close[i]);
        i += 1;
    }
    i = shadow_trailing_idx;
    while i < lookback_total {
        shadow_period_total += upper_shadow(open[i], high[i], close[i]);
        i += 1;
    }

    i = lookback_total;
    while i < size {
        if real_body(open[i], close[i]) > candle_average(body_period_total, body_long_period)
            && upper_shadow(open[i], high[i], close[i])
                < candle_average(shadow_period_total, shadow_short_period)
            && lower_shadow(open[i], low[i], close[i])
                < candle_average(shadow_period_total, shadow_short_period)
        {
            out[i] = (candle_color(open[i], close[i]) as i8) * 100;
        }

        body_period_total += real_body(open[i], close[i])
            - real_body(open[body_trailing_idx], close[body_trailing_idx]);
        shadow_period_total += upper_shadow(open[i], high[i], close[i])
            - upper_shadow(
                open[shadow_trailing_idx],
                high[shadow_trailing_idx],
                close[shadow_trailing_idx],
            );

        i += 1;
        body_trailing_idx += 1;
        shadow_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlmarubozu(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_long_period = 10;
    let shadow_very_short_period = 10;
    let lookback_total = body_long_period.max(shadow_very_short_period);

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    let mut out = vec![0i8; size];
    let mut body_long_period_total = 0.0;
    let mut shadow_very_short_period_total = 0.0;
    let mut body_long_trailing_idx = lookback_total - body_long_period;
    let mut shadow_very_short_trailing_idx = lookback_total - shadow_very_short_period;

    let mut i = body_long_trailing_idx;
    while i < lookback_total {
        body_long_period_total += real_body(open[i], close[i]);
        i += 1;
    }
    i = shadow_very_short_trailing_idx;
    while i < lookback_total {
        shadow_very_short_period_total += upper_shadow(open[i], high[i], close[i]);
        i += 1;
    }

    i = lookback_total;
    while i < size {
        if real_body(open[i], close[i]) > candle_average(body_long_period_total, body_long_period)
            && upper_shadow(open[i], high[i], close[i])
                < candle_average(shadow_very_short_period_total, shadow_very_short_period)
            && lower_shadow(open[i], low[i], close[i])
                < candle_average(shadow_very_short_period_total, shadow_very_short_period)
        {
            out[i] = (candle_color(open[i], close[i]) as i8) * 100;
        }

        body_long_period_total += real_body(open[i], close[i])
            - real_body(open[body_long_trailing_idx], close[body_long_trailing_idx]);
        shadow_very_short_period_total += upper_shadow(open[i], high[i], close[i])
            - upper_shadow(
                open[shadow_very_short_trailing_idx],
                high[shadow_very_short_trailing_idx],
                close[shadow_very_short_trailing_idx],
            );

        i += 1;
        body_long_trailing_idx += 1;
        shadow_very_short_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlrickshawman(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_doji_period = 10;
    let shadow_long_period = 10;
    let near_period = 5;
    let lookback_total = body_doji_period.max(shadow_long_period).max(near_period);

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    let mut out = vec![0i8; size];
    let mut body_doji_period_total = 0.0;
    let mut shadow_long_period_total = 0.0;
    let mut near_period_total = 0.0;
    let mut body_doji_trailing_idx = lookback_total - body_doji_period;
    let mut shadow_long_trailing_idx = lookback_total - shadow_long_period;
    let mut near_trailing_idx = lookback_total - near_period;

    let mut i = body_doji_trailing_idx;
    while i < lookback_total {
        body_doji_period_total += real_body(open[i], close[i]);
        i += 1;
    }
    i = shadow_long_trailing_idx;
    while i < lookback_total {
        shadow_long_period_total += upper_shadow(open[i], high[i], close[i]);
        i += 1;
    }
    i = near_trailing_idx;
    while i < lookback_total {
        near_period_total += candle_range(open[i], close[i]);
        i += 1;
    }

    i = lookback_total;
    while i < size {
        let rb = real_body(open[i], close[i]);
        let us = upper_shadow(open[i], high[i], close[i]);
        let ls = lower_shadow(open[i], low[i], close[i]);
        let hl_mid = low[i] + (high[i] - low[i]) * 0.5;
        let body_low = open[i].min(close[i]);
        let body_high = open[i].max(close[i]);
        let near_avg = candle_average(near_period_total, near_period);

        if rb <= candle_average(body_doji_period_total, body_doji_period)
            && ls > candle_average(shadow_long_period_total, shadow_long_period)
            && us > candle_average(shadow_long_period_total, shadow_long_period)
            && body_low <= hl_mid + near_avg
            && body_high >= hl_mid - near_avg
        {
            out[i] = 100;
        }

        body_doji_period_total += real_body(open[i], close[i])
            - real_body(open[body_doji_trailing_idx], close[body_doji_trailing_idx]);
        shadow_long_period_total += upper_shadow(open[i], high[i], close[i])
            - upper_shadow(
                open[shadow_long_trailing_idx],
                high[shadow_long_trailing_idx],
                close[shadow_long_trailing_idx],
            );
        near_period_total += candle_range(open[i], close[i])
            - candle_range(open[near_trailing_idx], close[near_trailing_idx]);

        i += 1;
        body_doji_trailing_idx += 1;
        shadow_long_trailing_idx += 1;
        near_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlshootingstar(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_short_period = 10;
    let shadow_long_period = 10;
    let shadow_very_short_period = 10;
    let lookback_total = body_short_period
        .max(shadow_long_period)
        .max(shadow_very_short_period)
        + 1;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    #[inline(always)]
    fn real_body_gap_up(curr_open: f64, curr_close: f64, prev_open: f64, prev_close: f64) -> bool {
        curr_open.min(curr_close) > prev_open.max(prev_close)
    }

    let mut out = vec![0i8; size];
    let mut body_period_total = 0.0;
    let mut shadow_long_period_total = 0.0;
    let mut shadow_very_short_period_total = 0.0;
    let mut body_trailing_idx = lookback_total - 1 - body_short_period;
    let mut shadow_long_trailing_idx = lookback_total - 1 - shadow_long_period;
    let mut shadow_very_short_trailing_idx = lookback_total - 1 - shadow_very_short_period;

    let mut i = body_trailing_idx;
    while i < lookback_total - 1 {
        body_period_total += real_body(open[i], close[i]);
        i += 1;
    }
    i = shadow_long_trailing_idx;
    while i < lookback_total - 1 {
        shadow_long_period_total += upper_shadow(open[i], high[i], close[i]);
        i += 1;
    }
    i = shadow_very_short_trailing_idx;
    while i < lookback_total - 1 {
        shadow_very_short_period_total += lower_shadow(open[i], low[i], close[i]);
        i += 1;
    }

    i = lookback_total;
    while i < size {
        if real_body(open[i], close[i]) < candle_average(body_period_total, body_short_period)
            && upper_shadow(open[i], high[i], close[i])
                > candle_average(shadow_long_period_total, shadow_long_period)
            && lower_shadow(open[i], low[i], close[i])
                < candle_average(shadow_very_short_period_total, shadow_very_short_period)
            && real_body_gap_up(open[i], close[i], open[i - 1], close[i - 1])
        {
            out[i] = -100;
        }

        body_period_total += real_body(open[i], close[i])
            - real_body(open[body_trailing_idx], close[body_trailing_idx]);
        shadow_long_period_total += upper_shadow(open[i], high[i], close[i])
            - upper_shadow(
                open[shadow_long_trailing_idx],
                high[shadow_long_trailing_idx],
                close[shadow_long_trailing_idx],
            );
        shadow_very_short_period_total += lower_shadow(open[i], low[i], close[i])
            - lower_shadow(
                open[shadow_very_short_trailing_idx],
                low[shadow_very_short_trailing_idx],
                close[shadow_very_short_trailing_idx],
            );

        i += 1;
        body_trailing_idx += 1;
        shadow_long_trailing_idx += 1;
        shadow_very_short_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlshortline(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_short_period = 10;
    let shadow_short_period = 10;
    let lookback_total = body_short_period.max(shadow_short_period);

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    let mut out = vec![0i8; size];
    let mut body_period_total = 0.0;
    let mut shadow_period_total = 0.0;
    let mut body_trailing_idx = lookback_total - body_short_period;
    let mut shadow_trailing_idx = lookback_total - shadow_short_period;

    let mut i = body_trailing_idx;
    while i < lookback_total {
        body_period_total += real_body(open[i], close[i]);
        i += 1;
    }
    i = shadow_trailing_idx;
    while i < lookback_total {
        shadow_period_total += upper_shadow(open[i], high[i], close[i]);
        i += 1;
    }

    i = lookback_total;
    while i < size {
        if real_body(open[i], close[i]) < candle_average(body_period_total, body_short_period)
            && upper_shadow(open[i], high[i], close[i])
                < candle_average(shadow_period_total, shadow_short_period)
            && lower_shadow(open[i], low[i], close[i])
                < candle_average(shadow_period_total, shadow_short_period)
        {
            out[i] = (candle_color(open[i], close[i]) as i8) * 100;
        }

        body_period_total += real_body(open[i], close[i])
            - real_body(open[body_trailing_idx], close[body_trailing_idx]);
        shadow_period_total += upper_shadow(open[i], high[i], close[i])
            - upper_shadow(
                open[shadow_trailing_idx],
                high[shadow_trailing_idx],
                close[shadow_trailing_idx],
            );

        i += 1;
        body_trailing_idx += 1;
        shadow_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlspinningtop(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_short_period = 10;
    let lookback_total = body_short_period;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    let mut out = vec![0i8; size];
    let mut body_period_total = 0.0;
    let mut body_trailing_idx = lookback_total - body_short_period;

    let mut i = body_trailing_idx;
    while i < lookback_total {
        body_period_total += real_body(open[i], close[i]);
        i += 1;
    }

    i = lookback_total;
    while i < size {
        let rb = real_body(open[i], close[i]);
        let us = upper_shadow(open[i], high[i], close[i]);
        let ls = lower_shadow(open[i], low[i], close[i]);
        if rb < candle_average(body_period_total, body_short_period) && us > rb && ls > rb {
            out[i] = (candle_color(open[i], close[i]) as i8) * 100;
        }

        body_period_total += real_body(open[i], close[i])
            - real_body(open[body_trailing_idx], close[body_trailing_idx]);

        i += 1;
        body_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdltakuri(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_doji_period = 10;
    let shadow_very_short_period = 10;
    let shadow_very_long_period = 10;
    let lookback_total = body_doji_period
        .max(shadow_very_short_period)
        .max(shadow_very_long_period);

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    let mut out = vec![0i8; size];
    let mut body_doji_period_total = 0.0;
    let mut shadow_very_short_period_total = 0.0;
    let mut shadow_very_long_period_total = 0.0;
    let mut body_doji_trailing_idx = lookback_total - body_doji_period;
    let mut shadow_very_short_trailing_idx = lookback_total - shadow_very_short_period;
    let mut shadow_very_long_trailing_idx = lookback_total - shadow_very_long_period;

    let mut i = body_doji_trailing_idx;
    while i < lookback_total {
        body_doji_period_total += real_body(open[i], close[i]);
        i += 1;
    }
    i = shadow_very_short_trailing_idx;
    while i < lookback_total {
        shadow_very_short_period_total += upper_shadow(open[i], high[i], close[i]);
        i += 1;
    }
    i = shadow_very_long_trailing_idx;
    while i < lookback_total {
        shadow_very_long_period_total += lower_shadow(open[i], low[i], close[i]);
        i += 1;
    }

    i = lookback_total;
    while i < size {
        if real_body(open[i], close[i]) <= candle_average(body_doji_period_total, body_doji_period)
            && upper_shadow(open[i], high[i], close[i])
                < candle_average(shadow_very_short_period_total, shadow_very_short_period)
            && lower_shadow(open[i], low[i], close[i])
                > candle_average(shadow_very_long_period_total, shadow_very_long_period)
        {
            out[i] = 100;
        }

        body_doji_period_total += real_body(open[i], close[i])
            - real_body(open[body_doji_trailing_idx], close[body_doji_trailing_idx]);
        shadow_very_short_period_total += upper_shadow(open[i], high[i], close[i])
            - upper_shadow(
                open[shadow_very_short_trailing_idx],
                high[shadow_very_short_trailing_idx],
                close[shadow_very_short_trailing_idx],
            );
        shadow_very_long_period_total += lower_shadow(open[i], low[i], close[i])
            - lower_shadow(
                open[shadow_very_long_trailing_idx],
                low[shadow_very_long_trailing_idx],
                close[shadow_very_long_trailing_idx],
            );

        i += 1;
        body_doji_trailing_idx += 1;
        shadow_very_short_trailing_idx += 1;
        shadow_very_long_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlhomingpigeon(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, _high, _low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_long_period = 10;
    let body_short_period = 10;
    let lookback_total = body_long_period.max(body_short_period) + 1;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    let mut out = vec![0i8; size];
    let mut body_long_period_total = 0.0;
    let mut body_short_period_total = 0.0;
    let mut body_long_trailing_idx = lookback_total - body_long_period;
    let mut body_short_trailing_idx = lookback_total - body_short_period;

    let mut i = body_long_trailing_idx;
    while i < lookback_total {
        body_long_period_total += real_body(open[i - 1], close[i - 1]);
        i += 1;
    }
    i = body_short_trailing_idx;
    while i < lookback_total {
        body_short_period_total += real_body(open[i], close[i]);
        i += 1;
    }

    i = lookback_total;
    while i < size {
        if candle_color(open[i - 1], close[i - 1]) == -1
            && candle_color(open[i], close[i]) == -1
            && real_body(open[i - 1], close[i - 1])
                > candle_average(body_long_period_total, body_long_period)
            && real_body(open[i], close[i])
                <= candle_average(body_short_period_total, body_short_period)
            && open[i] < open[i - 1]
            && close[i] > close[i - 1]
        {
            out[i] = 100;
        }

        body_long_period_total += real_body(open[i - 1], close[i - 1])
            - real_body(
                open[body_long_trailing_idx - 1],
                close[body_long_trailing_idx - 1],
            );
        body_short_period_total += real_body(open[i], close[i])
            - real_body(
                open[body_short_trailing_idx],
                close[body_short_trailing_idx],
            );

        i += 1;
        body_long_trailing_idx += 1;
        body_short_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlmatchinglow(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, _high, _low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let equal_period = 10;
    let lookback_total = equal_period + 1;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    let mut out = vec![0i8; size];
    let mut equal_period_total = 0.0;
    let mut equal_trailing_idx = lookback_total - equal_period;

    let mut i = equal_trailing_idx;
    while i < lookback_total {
        equal_period_total += real_body(open[i - 1], close[i - 1]);
        i += 1;
    }

    i = lookback_total;
    while i < size {
        let eq_avg = candle_average(equal_period_total, equal_period);
        if candle_color(open[i - 1], close[i - 1]) == -1
            && candle_color(open[i], close[i]) == -1
            && close[i] <= close[i - 1] + eq_avg
            && close[i] >= close[i - 1] - eq_avg
        {
            out[i] = 100;
        }

        equal_period_total += real_body(open[i - 1], close[i - 1])
            - real_body(open[equal_trailing_idx - 1], close[equal_trailing_idx - 1]);
        i += 1;
        equal_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlinneck(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, _high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let equal_period = 10;
    let body_long_period = 10;
    let lookback_total = equal_period.max(body_long_period) + 1;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    let mut out = vec![0i8; size];
    let mut equal_period_total = 0.0;
    let mut body_long_period_total = 0.0;
    let mut equal_trailing_idx = lookback_total - equal_period;
    let mut body_long_trailing_idx = lookback_total - body_long_period;

    let mut i = equal_trailing_idx;
    while i < lookback_total {
        equal_period_total += real_body(open[i - 1], close[i - 1]);
        i += 1;
    }
    i = body_long_trailing_idx;
    while i < lookback_total {
        body_long_period_total += real_body(open[i - 1], close[i - 1]);
        i += 1;
    }

    i = lookback_total;
    while i < size {
        let eq_avg = candle_average(equal_period_total, equal_period);
        if candle_color(open[i - 1], close[i - 1]) == -1
            && real_body(open[i - 1], close[i - 1])
                > candle_average(body_long_period_total, body_long_period)
            && candle_color(open[i], close[i]) == 1
            && open[i] < low[i - 1]
            && close[i] <= close[i - 1] + eq_avg
            && close[i] >= close[i - 1]
        {
            out[i] = -100;
        }

        equal_period_total += real_body(open[i - 1], close[i - 1])
            - real_body(open[equal_trailing_idx - 1], close[equal_trailing_idx - 1]);
        body_long_period_total += real_body(open[i - 1], close[i - 1])
            - real_body(
                open[body_long_trailing_idx - 1],
                close[body_long_trailing_idx - 1],
            );

        i += 1;
        equal_trailing_idx += 1;
        body_long_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlonneck(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, _high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let equal_period = 10;
    let body_long_period = 10;
    let lookback_total = equal_period.max(body_long_period) + 1;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    let mut out = vec![0i8; size];
    let mut equal_period_total = 0.0;
    let mut body_long_period_total = 0.0;
    let mut equal_trailing_idx = lookback_total - equal_period;
    let mut body_long_trailing_idx = lookback_total - body_long_period;

    let mut i = equal_trailing_idx;
    while i < lookback_total {
        equal_period_total += real_body(open[i - 1], close[i - 1]);
        i += 1;
    }
    i = body_long_trailing_idx;
    while i < lookback_total {
        body_long_period_total += real_body(open[i - 1], close[i - 1]);
        i += 1;
    }

    i = lookback_total;
    while i < size {
        let eq_avg = candle_average(equal_period_total, equal_period);
        if candle_color(open[i - 1], close[i - 1]) == -1
            && real_body(open[i - 1], close[i - 1])
                > candle_average(body_long_period_total, body_long_period)
            && candle_color(open[i], close[i]) == 1
            && open[i] < low[i - 1]
            && close[i] <= low[i - 1] + eq_avg
            && close[i] >= low[i - 1] - eq_avg
        {
            out[i] = -100;
        }

        equal_period_total += real_body(open[i - 1], close[i - 1])
            - real_body(open[equal_trailing_idx - 1], close[equal_trailing_idx - 1]);
        body_long_period_total += real_body(open[i - 1], close[i - 1])
            - real_body(
                open[body_long_trailing_idx - 1],
                close[body_long_trailing_idx - 1],
            );

        i += 1;
        equal_trailing_idx += 1;
        body_long_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlpiercing(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, _high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_long_period = 10;
    let lookback_total = body_long_period + 1;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    let mut out = vec![0i8; size];
    let mut body_long_period_total_prev = 0.0;
    let mut body_long_period_total_curr = 0.0;
    let mut body_long_trailing_idx = lookback_total - body_long_period;

    let mut i = body_long_trailing_idx;
    while i < lookback_total {
        body_long_period_total_prev += real_body(open[i - 1], close[i - 1]);
        body_long_period_total_curr += real_body(open[i], close[i]);
        i += 1;
    }

    i = lookback_total;
    while i < size {
        if candle_color(open[i - 1], close[i - 1]) == -1
            && real_body(open[i - 1], close[i - 1])
                > candle_average(body_long_period_total_prev, body_long_period)
            && candle_color(open[i], close[i]) == 1
            && real_body(open[i], close[i])
                > candle_average(body_long_period_total_curr, body_long_period)
            && open[i] < low[i - 1]
            && close[i] < open[i - 1]
            && close[i] > close[i - 1] + real_body(open[i - 1], close[i - 1]) * 0.5
        {
            out[i] = 100;
        }

        body_long_period_total_prev += real_body(open[i - 1], close[i - 1])
            - real_body(
                open[body_long_trailing_idx - 1],
                close[body_long_trailing_idx - 1],
            );
        body_long_period_total_curr += real_body(open[i], close[i])
            - real_body(open[body_long_trailing_idx], close[body_long_trailing_idx]);

        i += 1;
        body_long_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlthrusting(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, _high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let equal_period = 10;
    let body_long_period = 10;
    let lookback_total = equal_period.max(body_long_period) + 1;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    let mut out = vec![0i8; size];
    let mut equal_period_total = 0.0;
    let mut body_long_period_total = 0.0;
    let mut equal_trailing_idx = lookback_total - equal_period;
    let mut body_long_trailing_idx = lookback_total - body_long_period;

    let mut i = equal_trailing_idx;
    while i < lookback_total {
        equal_period_total += real_body(open[i - 1], close[i - 1]);
        i += 1;
    }
    i = body_long_trailing_idx;
    while i < lookback_total {
        body_long_period_total += real_body(open[i - 1], close[i - 1]);
        i += 1;
    }

    i = lookback_total;
    while i < size {
        if candle_color(open[i - 1], close[i - 1]) == -1
            && real_body(open[i - 1], close[i - 1])
                > candle_average(body_long_period_total, body_long_period)
            && candle_color(open[i], close[i]) == 1
            && open[i] < low[i - 1]
            && close[i] > close[i - 1] + candle_average(equal_period_total, equal_period)
            && close[i] <= close[i - 1] + real_body(open[i - 1], close[i - 1]) * 0.5
        {
            out[i] = -100;
        }

        equal_period_total += real_body(open[i - 1], close[i - 1])
            - real_body(open[equal_trailing_idx - 1], close[equal_trailing_idx - 1]);
        body_long_period_total += real_body(open[i - 1], close[i - 1])
            - real_body(
                open[body_long_trailing_idx - 1],
                close[body_long_trailing_idx - 1],
            );

        i += 1;
        equal_trailing_idx += 1;
        body_long_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlmorningdojistar(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, _high, _low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_long_period = 10;
    let body_doji_period = 10;
    let body_short_period = 10;
    let penetration = if input.params.penetration == 0.0 {
        0.3
    } else {
        input.params.penetration
    };
    let lookback_total = 2 + body_long_period
        .max(body_doji_period)
        .max(body_short_period);

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    #[inline(always)]
    fn real_body_gap_down(
        curr_open: f64,
        curr_close: f64,
        prev_open: f64,
        prev_close: f64,
    ) -> bool {
        curr_open.max(curr_close) < prev_open.min(prev_close)
    }

    let mut out = vec![0i8; size];
    let mut body_long_period_total = 0.0;
    let mut body_doji_period_total = 0.0;
    let mut body_short_period_total = 0.0;
    let mut body_long_trailing_idx = lookback_total - 2 - body_long_period;
    let mut body_doji_trailing_idx = lookback_total - 1 - body_doji_period;
    let mut body_short_trailing_idx = lookback_total - body_short_period;

    let mut i = body_long_trailing_idx;
    while i < lookback_total - 2 {
        body_long_period_total += real_body(open[i], close[i]);
        i += 1;
    }
    i = body_doji_trailing_idx;
    while i < lookback_total - 1 {
        body_doji_period_total += real_body(open[i], close[i]);
        i += 1;
    }
    i = body_short_trailing_idx;
    while i < lookback_total {
        body_short_period_total += real_body(open[i], close[i]);
        i += 1;
    }

    i = lookback_total;
    while i < size {
        if real_body(open[i - 2], close[i - 2])
            > candle_average(body_long_period_total, body_long_period)
            && candle_color(open[i - 2], close[i - 2]) == -1
            && real_body(open[i - 1], close[i - 1])
                <= candle_average(body_doji_period_total, body_doji_period)
            && real_body_gap_down(open[i - 1], close[i - 1], open[i - 2], close[i - 2])
            && real_body(open[i], close[i])
                > candle_average(body_short_period_total, body_short_period)
            && candle_color(open[i], close[i]) == 1
            && close[i] > close[i - 2] + real_body(open[i - 2], close[i - 2]) * penetration
        {
            out[i] = 100;
        }

        body_long_period_total += real_body(open[i - 2], close[i - 2])
            - real_body(open[body_long_trailing_idx], close[body_long_trailing_idx]);
        body_doji_period_total += real_body(open[i - 1], close[i - 1])
            - real_body(open[body_doji_trailing_idx], close[body_doji_trailing_idx]);
        body_short_period_total += real_body(open[i], close[i])
            - real_body(
                open[body_short_trailing_idx],
                close[body_short_trailing_idx],
            );

        i += 1;
        body_long_trailing_idx += 1;
        body_doji_trailing_idx += 1;
        body_short_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdltristar(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, _high, _low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_doji_period = 10;
    let lookback_total = body_doji_period + 2;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    #[inline(always)]
    fn real_body_gap_up(curr_open: f64, curr_close: f64, prev_open: f64, prev_close: f64) -> bool {
        curr_open.min(curr_close) > prev_open.max(prev_close)
    }

    #[inline(always)]
    fn real_body_gap_down(
        curr_open: f64,
        curr_close: f64,
        prev_open: f64,
        prev_close: f64,
    ) -> bool {
        curr_open.max(curr_close) < prev_open.min(prev_close)
    }

    let mut out = vec![0i8; size];
    let mut body_period_total = 0.0;
    let mut body_trailing_idx = lookback_total - 2 - body_doji_period;

    let mut i = body_trailing_idx;
    while i < lookback_total - 2 {
        body_period_total += real_body(open[i], close[i]);
        i += 1;
    }

    i = lookback_total;
    while i < size {
        let doji_avg = candle_average(body_period_total, body_doji_period);
        if real_body(open[i - 2], close[i - 2]) <= doji_avg
            && real_body(open[i - 1], close[i - 1]) <= doji_avg
            && real_body(open[i], close[i]) <= doji_avg
        {
            let mut value = 0i8;
            if real_body_gap_up(open[i - 1], close[i - 1], open[i - 2], close[i - 2])
                && open[i].max(close[i]) < open[i - 1].max(close[i - 1])
            {
                value = -100;
            }
            if real_body_gap_down(open[i - 1], close[i - 1], open[i - 2], close[i - 2])
                && open[i].min(close[i]) > open[i - 1].min(close[i - 1])
            {
                value = 100;
            }
            out[i] = value;
        }

        body_period_total += real_body(open[i - 2], close[i - 2])
            - real_body(open[body_trailing_idx], close[body_trailing_idx]);
        i += 1;
        body_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlidentical3crows(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, _high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let shadow_very_short_period = 10;
    let equal_period = 10;
    let lookback_total = shadow_very_short_period.max(equal_period) + 2;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    let mut out = vec![0i8; size];
    let mut shadow_total_2 = 0.0;
    let mut shadow_total_1 = 0.0;
    let mut shadow_total_0 = 0.0;
    let mut equal_total_2 = 0.0;
    let mut equal_total_1 = 0.0;
    let mut shadow_trailing_idx = lookback_total - shadow_very_short_period;
    let mut equal_trailing_idx = lookback_total - equal_period;

    let mut i = shadow_trailing_idx;
    while i < lookback_total {
        shadow_total_2 += lower_shadow(open[i - 2], low[i - 2], close[i - 2]);
        shadow_total_1 += lower_shadow(open[i - 1], low[i - 1], close[i - 1]);
        shadow_total_0 += lower_shadow(open[i], low[i], close[i]);
        i += 1;
    }
    i = equal_trailing_idx;
    while i < lookback_total {
        equal_total_2 += real_body(open[i - 2], close[i - 2]);
        equal_total_1 += real_body(open[i - 1], close[i - 1]);
        i += 1;
    }

    i = lookback_total;
    while i < size {
        if candle_color(open[i - 2], close[i - 2]) == -1
            && lower_shadow(open[i - 2], low[i - 2], close[i - 2])
                < candle_average(shadow_total_2, shadow_very_short_period)
            && candle_color(open[i - 1], close[i - 1]) == -1
            && lower_shadow(open[i - 1], low[i - 1], close[i - 1])
                < candle_average(shadow_total_1, shadow_very_short_period)
            && candle_color(open[i], close[i]) == -1
            && lower_shadow(open[i], low[i], close[i])
                < candle_average(shadow_total_0, shadow_very_short_period)
            && close[i - 2] > close[i - 1]
            && close[i - 1] > close[i]
            && open[i - 1] <= close[i - 2] + candle_average(equal_total_2, equal_period)
            && open[i - 1] >= close[i - 2] - candle_average(equal_total_2, equal_period)
            && open[i] <= close[i - 1] + candle_average(equal_total_1, equal_period)
            && open[i] >= close[i - 1] - candle_average(equal_total_1, equal_period)
        {
            out[i] = -100;
        }

        shadow_total_2 += lower_shadow(open[i - 2], low[i - 2], close[i - 2])
            - lower_shadow(
                open[shadow_trailing_idx - 2],
                low[shadow_trailing_idx - 2],
                close[shadow_trailing_idx - 2],
            );
        shadow_total_1 += lower_shadow(open[i - 1], low[i - 1], close[i - 1])
            - lower_shadow(
                open[shadow_trailing_idx - 1],
                low[shadow_trailing_idx - 1],
                close[shadow_trailing_idx - 1],
            );
        shadow_total_0 += lower_shadow(open[i], low[i], close[i])
            - lower_shadow(
                open[shadow_trailing_idx],
                low[shadow_trailing_idx],
                close[shadow_trailing_idx],
            );

        equal_total_2 += real_body(open[i - 2], close[i - 2])
            - real_body(open[equal_trailing_idx - 2], close[equal_trailing_idx - 2]);
        equal_total_1 += real_body(open[i - 1], close[i - 1])
            - real_body(open[equal_trailing_idx - 1], close[equal_trailing_idx - 1]);

        i += 1;
        shadow_trailing_idx += 1;
        equal_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlsticksandwich(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, _high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let equal_period = 10;
    let lookback_total = equal_period + 2;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    let mut out = vec![0i8; size];
    let mut equal_period_total = 0.0;
    let mut equal_trailing_idx = lookback_total - equal_period;

    let mut i = equal_trailing_idx;
    while i < lookback_total {
        equal_period_total += real_body(open[i - 2], close[i - 2]);
        i += 1;
    }

    i = lookback_total;
    while i < size {
        let eq_avg = candle_average(equal_period_total, equal_period);
        if candle_color(open[i - 2], close[i - 2]) == -1
            && candle_color(open[i - 1], close[i - 1]) == 1
            && candle_color(open[i], close[i]) == -1
            && low[i - 1] > close[i - 2]
            && close[i] <= close[i - 2] + eq_avg
            && close[i] >= close[i - 2] - eq_avg
        {
            out[i] = 100;
        }

        equal_period_total += real_body(open[i - 2], close[i - 2])
            - real_body(open[equal_trailing_idx - 2], close[equal_trailing_idx - 2]);
        i += 1;
        equal_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlseparatinglines(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let shadow_very_short_period = 10;
    let body_long_period = 10;
    let equal_period = 10;
    let lookback_total = shadow_very_short_period
        .max(body_long_period)
        .max(equal_period)
        + 1;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    #[inline(always)]
    fn max_shadow(open: f64, high: f64, low: f64, close: f64) -> f64 {
        upper_shadow(open, high, close).max(lower_shadow(open, low, close))
    }

    let mut out = vec![0i8; size];
    let mut shadow_very_short_period_total = 0.0;
    let mut body_long_period_total = 0.0;
    let mut equal_period_total = 0.0;
    let mut shadow_very_short_trailing_idx = lookback_total - shadow_very_short_period;
    let mut body_long_trailing_idx = lookback_total - body_long_period;
    let mut equal_trailing_idx = lookback_total - equal_period;

    let mut i = shadow_very_short_trailing_idx;
    while i < lookback_total {
        shadow_very_short_period_total += max_shadow(open[i], high[i], low[i], close[i]);
        i += 1;
    }
    i = body_long_trailing_idx;
    while i < lookback_total {
        body_long_period_total += real_body(open[i], close[i]);
        i += 1;
    }
    i = equal_trailing_idx;
    while i < lookback_total {
        equal_period_total += real_body(open[i - 1], close[i - 1]);
        i += 1;
    }

    i = lookback_total;
    while i < size {
        if candle_color(open[i - 1], close[i - 1]) == -candle_color(open[i], close[i])
            && open[i] <= open[i - 1] + candle_average(equal_period_total, equal_period)
            && open[i] >= open[i - 1] - candle_average(equal_period_total, equal_period)
            && real_body(open[i], close[i])
                > candle_average(body_long_period_total, body_long_period)
            && ((candle_color(open[i], close[i]) == 1
                && lower_shadow(open[i], low[i], close[i])
                    < candle_average(shadow_very_short_period_total, shadow_very_short_period))
                || (candle_color(open[i], close[i]) == -1
                    && upper_shadow(open[i], high[i], close[i])
                        < candle_average(shadow_very_short_period_total, shadow_very_short_period)))
        {
            out[i] = (candle_color(open[i], close[i]) as i8) * 100;
        }

        shadow_very_short_period_total += max_shadow(open[i], high[i], low[i], close[i])
            - max_shadow(
                open[shadow_very_short_trailing_idx],
                high[shadow_very_short_trailing_idx],
                low[shadow_very_short_trailing_idx],
                close[shadow_very_short_trailing_idx],
            );
        body_long_period_total += real_body(open[i], close[i])
            - real_body(open[body_long_trailing_idx], close[body_long_trailing_idx]);
        equal_period_total += real_body(open[i - 1], close[i - 1])
            - real_body(open[equal_trailing_idx - 1], close[equal_trailing_idx - 1]);

        i += 1;
        shadow_very_short_trailing_idx += 1;
        body_long_trailing_idx += 1;
        equal_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlgapsidesidewhite(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, _high, _low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let near_period = 10;
    let equal_period = 10;
    let lookback_total = near_period.max(equal_period) + 2;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    #[inline(always)]
    fn real_body_gap_up(curr_open: f64, curr_close: f64, prev_open: f64, prev_close: f64) -> bool {
        curr_open.min(curr_close) > prev_open.max(prev_close)
    }

    #[inline(always)]
    fn real_body_gap_down(
        curr_open: f64,
        curr_close: f64,
        prev_open: f64,
        prev_close: f64,
    ) -> bool {
        curr_open.max(curr_close) < prev_open.min(prev_close)
    }

    let mut out = vec![0i8; size];
    let mut near_period_total = 0.0;
    let mut equal_period_total = 0.0;
    let mut near_trailing_idx = lookback_total - near_period;
    let mut equal_trailing_idx = lookback_total - equal_period;

    let mut i = near_trailing_idx;
    while i < lookback_total {
        near_period_total += real_body(open[i - 1], close[i - 1]);
        i += 1;
    }
    i = equal_trailing_idx;
    while i < lookback_total {
        equal_period_total += real_body(open[i - 1], close[i - 1]);
        i += 1;
    }

    i = lookback_total;
    while i < size {
        let gap_up_1 = real_body_gap_up(open[i - 1], close[i - 1], open[i - 2], close[i - 2]);
        let gap_down_1 = real_body_gap_down(open[i - 1], close[i - 1], open[i - 2], close[i - 2]);
        if ((gap_up_1 && real_body_gap_up(open[i], close[i], open[i - 2], close[i - 2]))
            || (gap_down_1 && real_body_gap_down(open[i], close[i], open[i - 2], close[i - 2])))
            && candle_color(open[i - 1], close[i - 1]) == 1
            && candle_color(open[i], close[i]) == 1
            && real_body(open[i], close[i])
                >= real_body(open[i - 1], close[i - 1])
                    - candle_average(near_period_total, near_period)
            && real_body(open[i], close[i])
                <= real_body(open[i - 1], close[i - 1])
                    + candle_average(near_period_total, near_period)
            && open[i] >= open[i - 1] - candle_average(equal_period_total, equal_period)
            && open[i] <= open[i - 1] + candle_average(equal_period_total, equal_period)
        {
            out[i] = if gap_up_1 { 100 } else { -100 };
        }

        near_period_total += real_body(open[i - 1], close[i - 1])
            - real_body(open[near_trailing_idx - 1], close[near_trailing_idx - 1]);
        equal_period_total += real_body(open[i - 1], close[i - 1])
            - real_body(open[equal_trailing_idx - 1], close[equal_trailing_idx - 1]);

        i += 1;
        near_trailing_idx += 1;
        equal_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlhikkake(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (_open, high, low, close) = input_ohlc(&input.data)?;

    let size = high.len();
    let lookback_total = 5;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    let mut out = vec![0i8; size];
    let mut pattern_idx = usize::MAX;
    let mut pattern_result = 0i8;

    let mut i = lookback_total - 3;
    while i < lookback_total {
        if high[i - 1] < high[i - 2]
            && low[i - 1] > low[i - 2]
            && ((high[i] < high[i - 1] && low[i] < low[i - 1])
                || (high[i] > high[i - 1] && low[i] > low[i - 1]))
        {
            pattern_result = if high[i] < high[i - 1] { 100 } else { -100 };
            pattern_idx = i;
        } else if pattern_idx != usize::MAX
            && pattern_idx > 0
            && i <= pattern_idx + 3
            && ((pattern_result > 0 && close[i] > high[pattern_idx - 1])
                || (pattern_result < 0 && close[i] < low[pattern_idx - 1]))
        {
            pattern_idx = usize::MAX;
        }
        i += 1;
    }

    i = lookback_total;
    while i < size {
        if high[i - 1] < high[i - 2]
            && low[i - 1] > low[i - 2]
            && ((high[i] < high[i - 1] && low[i] < low[i - 1])
                || (high[i] > high[i - 1] && low[i] > low[i - 1]))
        {
            pattern_result = if high[i] < high[i - 1] { 100 } else { -100 };
            pattern_idx = i;
            out[i] = pattern_result;
        } else if pattern_idx != usize::MAX
            && pattern_idx > 0
            && i <= pattern_idx + 3
            && ((pattern_result > 0 && close[i] > high[pattern_idx - 1])
                || (pattern_result < 0 && close[i] < low[pattern_idx - 1]))
        {
            out[i] = pattern_result;
            pattern_idx = usize::MAX;
        }

        i += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlhikkakemod(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let near_period = 10;
    let lookback_total = near_period.max(1) + 5;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    let mut out = vec![0i8; size];
    let mut near_period_total = 0.0;
    let mut near_trailing_idx = lookback_total - 3 - near_period;
    let mut pattern_idx = usize::MAX;
    let mut pattern_result = 0i8;

    let mut i = near_trailing_idx;
    while i < lookback_total - 3 {
        near_period_total += real_body(open[i - 2], close[i - 2]);
        i += 1;
    }

    i = lookback_total - 3;
    while i < lookback_total {
        if high[i - 2] < high[i - 3]
            && low[i - 2] > low[i - 3]
            && high[i - 1] < high[i - 2]
            && low[i - 1] > low[i - 2]
            && ((high[i] < high[i - 1]
                && low[i] < low[i - 1]
                && close[i - 2] <= low[i - 2] + candle_average(near_period_total, near_period))
                || (high[i] > high[i - 1]
                    && low[i] > low[i - 1]
                    && close[i - 2]
                        >= high[i - 2] - candle_average(near_period_total, near_period)))
        {
            pattern_result = if high[i] < high[i - 1] { 100 } else { -100 };
            pattern_idx = i;
        } else if pattern_idx != usize::MAX
            && pattern_idx > 0
            && i <= pattern_idx + 3
            && ((pattern_result > 0 && close[i] > high[pattern_idx - 1])
                || (pattern_result < 0 && close[i] < low[pattern_idx - 1]))
        {
            pattern_idx = usize::MAX;
        }

        near_period_total += real_body(open[i - 2], close[i - 2])
            - real_body(open[near_trailing_idx - 2], close[near_trailing_idx - 2]);
        near_trailing_idx += 1;
        i += 1;
    }

    i = lookback_total;
    while i < size {
        if high[i - 2] < high[i - 3]
            && low[i - 2] > low[i - 3]
            && high[i - 1] < high[i - 2]
            && low[i - 1] > low[i - 2]
            && ((high[i] < high[i - 1]
                && low[i] < low[i - 1]
                && close[i - 2] <= low[i - 2] + candle_average(near_period_total, near_period))
                || (high[i] > high[i - 1]
                    && low[i] > low[i - 1]
                    && close[i - 2]
                        >= high[i - 2] - candle_average(near_period_total, near_period)))
        {
            pattern_result = if high[i] < high[i - 1] { 100 } else { -100 };
            pattern_idx = i;
            out[i] = pattern_result;
        } else if pattern_idx != usize::MAX
            && pattern_idx > 0
            && i <= pattern_idx + 3
            && ((pattern_result > 0 && close[i] > high[pattern_idx - 1])
                || (pattern_result < 0 && close[i] < low[pattern_idx - 1]))
        {
            out[i] = pattern_result;
            pattern_idx = usize::MAX;
        }

        near_period_total += real_body(open[i - 2], close[i - 2])
            - real_body(open[near_trailing_idx - 2], close[near_trailing_idx - 2]);
        near_trailing_idx += 1;
        i += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlkicking(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let shadow_very_short_period = 10;
    let body_long_period = 10;
    let lookback_total = shadow_very_short_period.max(body_long_period) + 1;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    #[inline(always)]
    fn candle_gap_up(curr_open: f64, curr_close: f64, prev_open: f64, prev_close: f64) -> bool {
        curr_open.min(curr_close) > prev_open.max(prev_close)
    }

    #[inline(always)]
    fn candle_gap_down(curr_open: f64, curr_close: f64, prev_open: f64, prev_close: f64) -> bool {
        curr_open.max(curr_close) < prev_open.min(prev_close)
    }

    let mut out = vec![0i8; size];
    let mut shadow_total_prev = 0.0;
    let mut shadow_total_curr = 0.0;
    let mut body_total_prev = 0.0;
    let mut body_total_curr = 0.0;
    let mut shadow_trailing_idx = lookback_total - shadow_very_short_period;
    let mut body_trailing_idx = lookback_total - body_long_period;

    let mut i = shadow_trailing_idx;
    while i < lookback_total {
        shadow_total_prev += upper_shadow(open[i - 1], high[i - 1], close[i - 1])
            .max(lower_shadow(open[i - 1], low[i - 1], close[i - 1]));
        shadow_total_curr +=
            upper_shadow(open[i], high[i], close[i]).max(lower_shadow(open[i], low[i], close[i]));
        i += 1;
    }
    i = body_trailing_idx;
    while i < lookback_total {
        body_total_prev += real_body(open[i - 1], close[i - 1]);
        body_total_curr += real_body(open[i], close[i]);
        i += 1;
    }

    i = lookback_total;
    while i < size {
        if candle_color(open[i - 1], close[i - 1]) == -candle_color(open[i], close[i])
            && real_body(open[i - 1], close[i - 1])
                > candle_average(body_total_prev, body_long_period)
            && upper_shadow(open[i - 1], high[i - 1], close[i - 1]).max(lower_shadow(
                open[i - 1],
                low[i - 1],
                close[i - 1],
            )) < candle_average(shadow_total_prev, shadow_very_short_period)
            && real_body(open[i], close[i]) > candle_average(body_total_curr, body_long_period)
            && upper_shadow(open[i], high[i], close[i]).max(lower_shadow(open[i], low[i], close[i]))
                < candle_average(shadow_total_curr, shadow_very_short_period)
            && ((candle_color(open[i - 1], close[i - 1]) == -1
                && candle_gap_up(open[i], close[i], open[i - 1], close[i - 1]))
                || (candle_color(open[i - 1], close[i - 1]) == 1
                    && candle_gap_down(open[i], close[i], open[i - 1], close[i - 1])))
        {
            out[i] = (candle_color(open[i], close[i]) as i8) * 100;
        }

        shadow_total_prev += upper_shadow(open[i - 1], high[i - 1], close[i - 1])
            .max(lower_shadow(open[i - 1], low[i - 1], close[i - 1]))
            - upper_shadow(
                open[shadow_trailing_idx - 1],
                high[shadow_trailing_idx - 1],
                close[shadow_trailing_idx - 1],
            )
            .max(lower_shadow(
                open[shadow_trailing_idx - 1],
                low[shadow_trailing_idx - 1],
                close[shadow_trailing_idx - 1],
            ));
        shadow_total_curr += upper_shadow(open[i], high[i], close[i])
            .max(lower_shadow(open[i], low[i], close[i]))
            - upper_shadow(
                open[shadow_trailing_idx],
                high[shadow_trailing_idx],
                close[shadow_trailing_idx],
            )
            .max(lower_shadow(
                open[shadow_trailing_idx],
                low[shadow_trailing_idx],
                close[shadow_trailing_idx],
            ));
        body_total_prev += real_body(open[i - 1], close[i - 1])
            - real_body(open[body_trailing_idx - 1], close[body_trailing_idx - 1]);
        body_total_curr += real_body(open[i], close[i])
            - real_body(open[body_trailing_idx], close[body_trailing_idx]);

        i += 1;
        shadow_trailing_idx += 1;
        body_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlkickingbylength(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let shadow_very_short_period = 10;
    let body_long_period = 10;
    let lookback_total = shadow_very_short_period.max(body_long_period) + 1;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    #[inline(always)]
    fn candle_gap_up(curr_open: f64, curr_close: f64, prev_open: f64, prev_close: f64) -> bool {
        curr_open.min(curr_close) > prev_open.max(prev_close)
    }

    #[inline(always)]
    fn candle_gap_down(curr_open: f64, curr_close: f64, prev_open: f64, prev_close: f64) -> bool {
        curr_open.max(curr_close) < prev_open.min(prev_close)
    }

    let mut out = vec![0i8; size];
    let mut shadow_total_prev = 0.0;
    let mut shadow_total_curr = 0.0;
    let mut body_total_prev = 0.0;
    let mut body_total_curr = 0.0;
    let mut shadow_trailing_idx = lookback_total - shadow_very_short_period;
    let mut body_trailing_idx = lookback_total - body_long_period;

    let mut i = shadow_trailing_idx;
    while i < lookback_total {
        shadow_total_prev += upper_shadow(open[i - 1], high[i - 1], close[i - 1])
            .max(lower_shadow(open[i - 1], low[i - 1], close[i - 1]));
        shadow_total_curr +=
            upper_shadow(open[i], high[i], close[i]).max(lower_shadow(open[i], low[i], close[i]));
        i += 1;
    }
    i = body_trailing_idx;
    while i < lookback_total {
        body_total_prev += real_body(open[i - 1], close[i - 1]);
        body_total_curr += real_body(open[i], close[i]);
        i += 1;
    }

    i = lookback_total;
    while i < size {
        if candle_color(open[i - 1], close[i - 1]) == -candle_color(open[i], close[i])
            && real_body(open[i - 1], close[i - 1])
                > candle_average(body_total_prev, body_long_period)
            && upper_shadow(open[i - 1], high[i - 1], close[i - 1]).max(lower_shadow(
                open[i - 1],
                low[i - 1],
                close[i - 1],
            )) < candle_average(shadow_total_prev, shadow_very_short_period)
            && real_body(open[i], close[i]) > candle_average(body_total_curr, body_long_period)
            && upper_shadow(open[i], high[i], close[i]).max(lower_shadow(open[i], low[i], close[i]))
                < candle_average(shadow_total_curr, shadow_very_short_period)
            && ((candle_color(open[i - 1], close[i - 1]) == -1
                && candle_gap_up(open[i], close[i], open[i - 1], close[i - 1]))
                || (candle_color(open[i - 1], close[i - 1]) == 1
                    && candle_gap_down(open[i], close[i], open[i - 1], close[i - 1])))
        {
            out[i] = if real_body(open[i], close[i]) > real_body(open[i - 1], close[i - 1]) {
                (candle_color(open[i], close[i]) as i8) * 100
            } else {
                (candle_color(open[i - 1], close[i - 1]) as i8) * 100
            };
        }

        shadow_total_prev += upper_shadow(open[i - 1], high[i - 1], close[i - 1])
            .max(lower_shadow(open[i - 1], low[i - 1], close[i - 1]))
            - upper_shadow(
                open[shadow_trailing_idx - 1],
                high[shadow_trailing_idx - 1],
                close[shadow_trailing_idx - 1],
            )
            .max(lower_shadow(
                open[shadow_trailing_idx - 1],
                low[shadow_trailing_idx - 1],
                close[shadow_trailing_idx - 1],
            ));
        shadow_total_curr += upper_shadow(open[i], high[i], close[i])
            .max(lower_shadow(open[i], low[i], close[i]))
            - upper_shadow(
                open[shadow_trailing_idx],
                high[shadow_trailing_idx],
                close[shadow_trailing_idx],
            )
            .max(lower_shadow(
                open[shadow_trailing_idx],
                low[shadow_trailing_idx],
                close[shadow_trailing_idx],
            ));
        body_total_prev += real_body(open[i - 1], close[i - 1])
            - real_body(open[body_trailing_idx - 1], close[body_trailing_idx - 1]);
        body_total_curr += real_body(open[i], close[i])
            - real_body(open[body_trailing_idx], close[body_trailing_idx]);

        i += 1;
        shadow_trailing_idx += 1;
        body_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlladderbottom(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, high, _low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let shadow_very_short_period = 10;
    let lookback_total = shadow_very_short_period + 4;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    let mut out = vec![0i8; size];
    let mut shadow_very_short_total = 0.0;
    let mut shadow_trailing_idx = lookback_total - shadow_very_short_period;

    let mut i = shadow_trailing_idx;
    while i < lookback_total {
        shadow_very_short_total += upper_shadow(open[i - 1], high[i - 1], close[i - 1]);
        i += 1;
    }

    i = lookback_total;
    while i < size {
        if candle_color(open[i - 4], close[i - 4]) == -1
            && candle_color(open[i - 3], close[i - 3]) == -1
            && candle_color(open[i - 2], close[i - 2]) == -1
            && open[i - 4] > open[i - 3]
            && open[i - 3] > open[i - 2]
            && close[i - 4] > close[i - 3]
            && close[i - 3] > close[i - 2]
            && candle_color(open[i - 1], close[i - 1]) == -1
            && upper_shadow(open[i - 1], high[i - 1], close[i - 1])
                > candle_average(shadow_very_short_total, shadow_very_short_period)
            && candle_color(open[i], close[i]) == 1
            && open[i] > open[i - 1]
            && close[i] > high[i - 1]
        {
            out[i] = 100;
        }

        shadow_very_short_total += upper_shadow(open[i - 1], high[i - 1], close[i - 1])
            - upper_shadow(
                open[shadow_trailing_idx - 1],
                high[shadow_trailing_idx - 1],
                close[shadow_trailing_idx - 1],
            );
        i += 1;
        shadow_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlmathold(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, high, _low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_short_period = 10;
    let body_long_period = 10;
    let penetration = if input.params.penetration == 0.0 {
        0.5
    } else {
        input.params.penetration
    };
    let lookback_total = body_short_period.max(body_long_period) + 4;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    #[inline(always)]
    fn real_body_gap_up(curr_open: f64, curr_close: f64, prev_open: f64, prev_close: f64) -> bool {
        curr_open.min(curr_close) > prev_open.max(prev_close)
    }

    let mut out = vec![0i8; size];
    let mut body_total_4 = 0.0;
    let mut body_total_3 = 0.0;
    let mut body_total_2 = 0.0;
    let mut body_total_1 = 0.0;
    let mut body_short_trailing_idx = lookback_total - body_short_period;
    let mut body_long_trailing_idx = lookback_total - body_long_period;

    let mut i = body_short_trailing_idx;
    while i < lookback_total {
        body_total_3 += real_body(open[i - 3], close[i - 3]);
        body_total_2 += real_body(open[i - 2], close[i - 2]);
        body_total_1 += real_body(open[i - 1], close[i - 1]);
        i += 1;
    }
    i = body_long_trailing_idx;
    while i < lookback_total {
        body_total_4 += real_body(open[i - 4], close[i - 4]);
        i += 1;
    }

    i = lookback_total;
    while i < size {
        if real_body(open[i - 4], close[i - 4]) > candle_average(body_total_4, body_long_period)
            && real_body(open[i - 3], close[i - 3])
                < candle_average(body_total_3, body_short_period)
            && real_body(open[i - 2], close[i - 2])
                < candle_average(body_total_2, body_short_period)
            && real_body(open[i - 1], close[i - 1])
                < candle_average(body_total_1, body_short_period)
            && candle_color(open[i - 4], close[i - 4]) == 1
            && candle_color(open[i - 3], close[i - 3]) == -1
            && candle_color(open[i], close[i]) == 1
            && real_body_gap_up(open[i - 3], close[i - 3], open[i - 4], close[i - 4])
            && open[i - 2].min(close[i - 2]) < close[i - 4]
            && open[i - 1].min(close[i - 1]) < close[i - 4]
            && open[i - 2].min(close[i - 2])
                > close[i - 4] - real_body(open[i - 4], close[i - 4]) * penetration
            && open[i - 1].min(close[i - 1])
                > close[i - 4] - real_body(open[i - 4], close[i - 4]) * penetration
            && open[i - 2].max(close[i - 2]) < open[i - 3]
            && open[i - 1].max(close[i - 1]) < open[i - 2].max(close[i - 2])
            && open[i] > close[i - 1]
            && close[i] > high[i - 3].max(high[i - 2]).max(high[i - 1])
        {
            out[i] = 100;
        }

        body_total_4 += real_body(open[i - 4], close[i - 4])
            - real_body(
                open[body_long_trailing_idx - 4],
                close[body_long_trailing_idx - 4],
            );
        body_total_3 += real_body(open[i - 3], close[i - 3])
            - real_body(
                open[body_short_trailing_idx - 3],
                close[body_short_trailing_idx - 3],
            );
        body_total_2 += real_body(open[i - 2], close[i - 2])
            - real_body(
                open[body_short_trailing_idx - 2],
                close[body_short_trailing_idx - 2],
            );
        body_total_1 += real_body(open[i - 1], close[i - 1])
            - real_body(
                open[body_short_trailing_idx - 1],
                close[body_short_trailing_idx - 1],
            );

        i += 1;
        body_short_trailing_idx += 1;
        body_long_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlrisefall3methods(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_short_period = 10;
    let body_long_period = 10;
    let lookback_total = body_short_period.max(body_long_period) + 4;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    let mut out = vec![0i8; size];
    let mut body_total_4 = 0.0;
    let mut body_total_3 = 0.0;
    let mut body_total_2 = 0.0;
    let mut body_total_1 = 0.0;
    let mut body_total_0 = 0.0;
    let mut body_short_trailing_idx = lookback_total - body_short_period;
    let mut body_long_trailing_idx = lookback_total - body_long_period;

    let mut i = body_short_trailing_idx;
    while i < lookback_total {
        body_total_3 += real_body(open[i - 3], close[i - 3]);
        body_total_2 += real_body(open[i - 2], close[i - 2]);
        body_total_1 += real_body(open[i - 1], close[i - 1]);
        i += 1;
    }
    i = body_long_trailing_idx;
    while i < lookback_total {
        body_total_4 += real_body(open[i - 4], close[i - 4]);
        body_total_0 += real_body(open[i], close[i]);
        i += 1;
    }

    i = lookback_total;
    while i < size {
        let c4 = candle_color(open[i - 4], close[i - 4]);
        if real_body(open[i - 4], close[i - 4]) > candle_average(body_total_4, body_long_period)
            && real_body(open[i - 3], close[i - 3])
                < candle_average(body_total_3, body_short_period)
            && real_body(open[i - 2], close[i - 2])
                < candle_average(body_total_2, body_short_period)
            && real_body(open[i - 1], close[i - 1])
                < candle_average(body_total_1, body_short_period)
            && real_body(open[i], close[i]) > candle_average(body_total_0, body_long_period)
            && c4 == -candle_color(open[i - 3], close[i - 3])
            && candle_color(open[i - 3], close[i - 3]) == candle_color(open[i - 2], close[i - 2])
            && candle_color(open[i - 2], close[i - 2]) == candle_color(open[i - 1], close[i - 1])
            && candle_color(open[i - 1], close[i - 1]) == -candle_color(open[i], close[i])
            && open[i - 3].min(close[i - 3]) < high[i - 4]
            && open[i - 3].max(close[i - 3]) > low[i - 4]
            && open[i - 2].min(close[i - 2]) < high[i - 4]
            && open[i - 2].max(close[i - 2]) > low[i - 4]
            && open[i - 1].min(close[i - 1]) < high[i - 4]
            && open[i - 1].max(close[i - 1]) > low[i - 4]
            && close[i - 2] * (c4 as f64) < close[i - 3] * (c4 as f64)
            && close[i - 1] * (c4 as f64) < close[i - 2] * (c4 as f64)
            && open[i] * (c4 as f64) > close[i - 1] * (c4 as f64)
            && close[i] * (c4 as f64) > close[i - 4] * (c4 as f64)
        {
            out[i] = (c4 as i8) * 100;
        }

        body_total_4 += real_body(open[i - 4], close[i - 4])
            - real_body(
                open[body_long_trailing_idx - 4],
                close[body_long_trailing_idx - 4],
            );
        body_total_3 += real_body(open[i - 3], close[i - 3])
            - real_body(
                open[body_short_trailing_idx - 3],
                close[body_short_trailing_idx - 3],
            );
        body_total_2 += real_body(open[i - 2], close[i - 2])
            - real_body(
                open[body_short_trailing_idx - 2],
                close[body_short_trailing_idx - 2],
            );
        body_total_1 += real_body(open[i - 1], close[i - 1])
            - real_body(
                open[body_short_trailing_idx - 1],
                close[body_short_trailing_idx - 1],
            );
        body_total_0 += real_body(open[i], close[i])
            - real_body(open[body_long_trailing_idx], close[body_long_trailing_idx]);

        i += 1;
        body_short_trailing_idx += 1;
        body_long_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlstalledpattern(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, high, _low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_long_period = 10;
    let body_short_period = 10;
    let shadow_very_short_period = 10;
    let near_period = 10;
    let lookback_total = body_long_period
        .max(body_short_period)
        .max(shadow_very_short_period)
        .max(near_period)
        + 2;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    let mut out = vec![0i8; size];
    let mut body_long_total_2 = 0.0;
    let mut body_long_total_1 = 0.0;
    let mut body_short_total = 0.0;
    let mut shadow_vs_total = 0.0;
    let mut near_total_2 = 0.0;
    let mut near_total_1 = 0.0;
    let mut body_long_trailing_idx = lookback_total - body_long_period;
    let mut body_short_trailing_idx = lookback_total - body_short_period;
    let mut shadow_vs_trailing_idx = lookback_total - shadow_very_short_period;
    let mut near_trailing_idx = lookback_total - near_period;

    let mut i = body_long_trailing_idx;
    while i < lookback_total {
        body_long_total_2 += real_body(open[i - 2], close[i - 2]);
        body_long_total_1 += real_body(open[i - 1], close[i - 1]);
        i += 1;
    }
    i = body_short_trailing_idx;
    while i < lookback_total {
        body_short_total += real_body(open[i], close[i]);
        i += 1;
    }
    i = shadow_vs_trailing_idx;
    while i < lookback_total {
        shadow_vs_total += upper_shadow(open[i - 1], high[i - 1], close[i - 1]);
        i += 1;
    }
    i = near_trailing_idx;
    while i < lookback_total {
        near_total_2 += real_body(open[i - 2], close[i - 2]);
        near_total_1 += real_body(open[i - 1], close[i - 1]);
        i += 1;
    }

    i = lookback_total;
    while i < size {
        if candle_color(open[i - 2], close[i - 2]) == 1
            && candle_color(open[i - 1], close[i - 1]) == 1
            && candle_color(open[i], close[i]) == 1
            && close[i] > close[i - 1]
            && close[i - 1] > close[i - 2]
            && real_body(open[i - 2], close[i - 2])
                > candle_average(body_long_total_2, body_long_period)
            && real_body(open[i - 1], close[i - 1])
                > candle_average(body_long_total_1, body_long_period)
            && upper_shadow(open[i - 1], high[i - 1], close[i - 1])
                < candle_average(shadow_vs_total, shadow_very_short_period)
            && open[i - 1] > open[i - 2]
            && open[i - 1] <= close[i - 2] + candle_average(near_total_2, near_period)
            && real_body(open[i], close[i]) < candle_average(body_short_total, body_short_period)
            && open[i]
                >= close[i - 1]
                    - real_body(open[i], close[i])
                    - candle_average(near_total_1, near_period)
        {
            out[i] = -100;
        }

        body_long_total_2 += real_body(open[i - 2], close[i - 2])
            - real_body(
                open[body_long_trailing_idx - 2],
                close[body_long_trailing_idx - 2],
            );
        body_long_total_1 += real_body(open[i - 1], close[i - 1])
            - real_body(
                open[body_long_trailing_idx - 1],
                close[body_long_trailing_idx - 1],
            );
        body_short_total += real_body(open[i], close[i])
            - real_body(
                open[body_short_trailing_idx],
                close[body_short_trailing_idx],
            );
        shadow_vs_total += upper_shadow(open[i - 1], high[i - 1], close[i - 1])
            - upper_shadow(
                open[shadow_vs_trailing_idx - 1],
                high[shadow_vs_trailing_idx - 1],
                close[shadow_vs_trailing_idx - 1],
            );
        near_total_2 += real_body(open[i - 2], close[i - 2])
            - real_body(open[near_trailing_idx - 2], close[near_trailing_idx - 2]);
        near_total_1 += real_body(open[i - 1], close[i - 1])
            - real_body(open[near_trailing_idx - 1], close[near_trailing_idx - 1]);

        i += 1;
        body_long_trailing_idx += 1;
        body_short_trailing_idx += 1;
        shadow_vs_trailing_idx += 1;
        near_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdltasukigap(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, _high, _low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let near_period = 10;
    let lookback_total = near_period + 2;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    #[inline(always)]
    fn real_body_gap_up(curr_open: f64, curr_close: f64, prev_open: f64, prev_close: f64) -> bool {
        curr_open.min(curr_close) > prev_open.max(prev_close)
    }

    #[inline(always)]
    fn real_body_gap_down(
        curr_open: f64,
        curr_close: f64,
        prev_open: f64,
        prev_close: f64,
    ) -> bool {
        curr_open.max(curr_close) < prev_open.min(prev_close)
    }

    let mut out = vec![0i8; size];
    let mut near_period_total = 0.0;
    let mut near_trailing_idx = lookback_total - near_period;

    let mut i = near_trailing_idx;
    while i < lookback_total {
        near_period_total += real_body(open[i - 1], close[i - 1]);
        i += 1;
    }

    i = lookback_total;
    while i < size {
        if (real_body_gap_up(open[i - 1], close[i - 1], open[i - 2], close[i - 2])
            && candle_color(open[i - 1], close[i - 1]) == 1
            && candle_color(open[i], close[i]) == -1
            && open[i] < close[i - 1]
            && open[i] > open[i - 1]
            && close[i] < open[i - 1]
            && close[i] > close[i - 2].max(open[i - 2])
            && (real_body(open[i - 1], close[i - 1]) - real_body(open[i], close[i])).abs()
                < candle_average(near_period_total, near_period))
            || (real_body_gap_down(open[i - 1], close[i - 1], open[i - 2], close[i - 2])
                && candle_color(open[i - 1], close[i - 1]) == -1
                && candle_color(open[i], close[i]) == 1
                && open[i] < open[i - 1]
                && open[i] > close[i - 1]
                && close[i] > open[i - 1]
                && close[i] < close[i - 2].min(open[i - 2])
                && (real_body(open[i - 1], close[i - 1]) - real_body(open[i], close[i])).abs()
                    < candle_average(near_period_total, near_period))
        {
            out[i] = (candle_color(open[i - 1], close[i - 1]) as i8) * 100;
        }

        near_period_total += real_body(open[i - 1], close[i - 1])
            - real_body(open[near_trailing_idx - 1], close[near_trailing_idx - 1]);
        i += 1;
        near_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlunique3river(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, _high, low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_short_period = 10;
    let body_long_period = 10;
    let lookback_total = body_short_period.max(body_long_period) + 2;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    let mut out = vec![0i8; size];
    let mut body_long_total = 0.0;
    let mut body_short_total = 0.0;
    let mut body_long_trailing_idx = lookback_total - 2 - body_long_period;
    let mut body_short_trailing_idx = lookback_total - body_short_period;

    let mut i = body_long_trailing_idx;
    while i < lookback_total - 2 {
        body_long_total += real_body(open[i], close[i]);
        i += 1;
    }
    i = body_short_trailing_idx;
    while i < lookback_total {
        body_short_total += real_body(open[i], close[i]);
        i += 1;
    }

    i = lookback_total;
    while i < size {
        if real_body(open[i - 2], close[i - 2]) > candle_average(body_long_total, body_long_period)
            && candle_color(open[i - 2], close[i - 2]) == -1
            && candle_color(open[i - 1], close[i - 1]) == -1
            && close[i - 1] > close[i - 2]
            && open[i - 1] <= open[i - 2]
            && low[i - 1] < low[i - 2]
            && real_body(open[i], close[i]) < candle_average(body_short_total, body_short_period)
            && candle_color(open[i], close[i]) == 1
            && open[i] > low[i - 1]
        {
            out[i] = 100;
        }

        body_long_total += real_body(open[i - 2], close[i - 2])
            - real_body(open[body_long_trailing_idx], close[body_long_trailing_idx]);
        body_short_total += real_body(open[i], close[i])
            - real_body(
                open[body_short_trailing_idx],
                close[body_short_trailing_idx],
            );
        i += 1;
        body_long_trailing_idx += 1;
        body_short_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlupsidegap2crows(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, _high, _low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let body_short_period = 10;
    let body_long_period = 10;
    let lookback_total = body_short_period.max(body_long_period) + 2;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn candle_average(sum: f64, period: usize) -> f64 {
        if period == 0 {
            0.0
        } else {
            sum / period as f64
        }
    }

    #[inline(always)]
    fn real_body_gap_up(curr_open: f64, curr_close: f64, prev_open: f64, prev_close: f64) -> bool {
        curr_open.min(curr_close) > prev_open.max(prev_close)
    }

    let mut out = vec![0i8; size];
    let mut body_long_total = 0.0;
    let mut body_short_total = 0.0;
    let mut body_long_trailing_idx = lookback_total - 2 - body_long_period;
    let mut body_short_trailing_idx = lookback_total - 1 - body_short_period;

    let mut i = body_long_trailing_idx;
    while i < lookback_total - 2 {
        body_long_total += real_body(open[i], close[i]);
        i += 1;
    }
    i = body_short_trailing_idx;
    while i < lookback_total - 1 {
        body_short_total += real_body(open[i], close[i]);
        i += 1;
    }

    i = lookback_total;
    while i < size {
        if candle_color(open[i - 2], close[i - 2]) == 1
            && real_body(open[i - 2], close[i - 2])
                > candle_average(body_long_total, body_long_period)
            && candle_color(open[i - 1], close[i - 1]) == -1
            && real_body(open[i - 1], close[i - 1])
                <= candle_average(body_short_total, body_short_period)
            && real_body_gap_up(open[i - 1], close[i - 1], open[i - 2], close[i - 2])
            && candle_color(open[i], close[i]) == -1
            && open[i] > open[i - 1]
            && close[i] < close[i - 1]
            && close[i] > close[i - 2]
        {
            out[i] = -100;
        }

        body_long_total += real_body(open[i - 2], close[i - 2])
            - real_body(open[body_long_trailing_idx], close[body_long_trailing_idx]);
        body_short_total += real_body(open[i - 1], close[i - 1])
            - real_body(
                open[body_short_trailing_idx],
                close[body_short_trailing_idx],
            );
        i += 1;
        body_long_trailing_idx += 1;
        body_short_trailing_idx += 1;
    }

    Ok(PatternOutput { values: out })
}

#[inline]
pub fn cdlxsidegap3methods(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    let (open, _high, _low, close) = input_ohlc(&input.data)?;

    let size = open.len();
    let lookback_total = 2;

    if size < lookback_total {
        return Err(PatternError::NotEnoughData {
            len: size,
            pattern: input.params.pattern_type,
        });
    }

    #[inline(always)]
    fn real_body_gap_up(curr_open: f64, curr_close: f64, prev_open: f64, prev_close: f64) -> bool {
        curr_open.min(curr_close) > prev_open.max(prev_close)
    }

    #[inline(always)]
    fn real_body_gap_down(
        curr_open: f64,
        curr_close: f64,
        prev_open: f64,
        prev_close: f64,
    ) -> bool {
        curr_open.max(curr_close) < prev_open.min(prev_close)
    }

    let mut out = vec![0i8; size];
    let mut i = lookback_total;
    while i < size {
        if candle_color(open[i - 2], close[i - 2]) == candle_color(open[i - 1], close[i - 1])
            && candle_color(open[i - 1], close[i - 1]) == -candle_color(open[i], close[i])
            && open[i] < close[i - 1].max(open[i - 1])
            && open[i] > close[i - 1].min(open[i - 1])
            && close[i] < close[i - 2].max(open[i - 2])
            && close[i] > close[i - 2].min(open[i - 2])
            && ((candle_color(open[i - 2], close[i - 2]) == 1
                && real_body_gap_up(open[i - 1], close[i - 1], open[i - 2], close[i - 2]))
                || (candle_color(open[i - 2], close[i - 2]) == -1
                    && real_body_gap_down(open[i - 1], close[i - 1], open[i - 2], close[i - 2])))
        {
            out[i] = (candle_color(open[i - 2], close[i - 2]) as i8) * 100;
        }

        i += 1;
    }

    Ok(PatternOutput { values: out })
}

#[cfg(feature = "python")]
#[pyfunction(name = "pattern_recognition")]
#[pyo3(signature = (open, high, low, close, kernel=None))]
pub fn pattern_recognition_py<'py>(
    py: Python<'py>,
    open: numpy::PyReadonlyArray1<'py, f64>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    close: numpy::PyReadonlyArray1<'py, f64>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let open_slice = open.as_slice()?;
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let input = PatternRecognitionInput::with_default_slices(
        open_slice,
        high_slice,
        low_slice,
        close_slice,
    );
    let output = py
        .allow_threads(|| pattern_recognition_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    let rows = output.rows;
    let cols = output.cols;
    let values = output.values_u8.into_pyarray(py);
    dict.set_item("values", values.reshape((rows, cols))?)?;
    dict.set_item(
        "pattern_ids",
        PyList::new(py, output.pattern_ids.iter().copied())?,
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    dict.set_item("warmup", output.warmup)?;
    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "pattern_recognition_cuda_batch_dev")]
#[pyo3(signature = (open_f32, high_f32, low_f32, close_f32, device_id=0))]
pub fn pattern_recognition_cuda_batch_dev_py(
    py: Python<'_>,
    open_f32: numpy::PyReadonlyArray1<'_, f32>,
    high_f32: numpy::PyReadonlyArray1<'_, f32>,
    low_f32: numpy::PyReadonlyArray1<'_, f32>,
    close_f32: numpy::PyReadonlyArray1<'_, f32>,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use crate::cuda::pattern_recognition_wrapper::CudaPatternRecognition;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let open_slice = open_f32.as_slice()?;
    let high_slice = high_f32.as_slice()?;
    let low_slice = low_f32.as_slice()?;
    let close_slice = close_f32.as_slice()?;

    let (inner, ctx, dev_id) = py.allow_threads(|| {
        let cuda = CudaPatternRecognition::new(device_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        let dev_id = cuda.device_id();
        let d_f32 = cuda
            .compute_native_matrix_f32_device_from_host_inputs(
                open_slice,
                high_slice,
                low_slice,
                close_slice,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, PyErr>((d_f32, ctx, dev_id))
    })?;

    Ok(DeviceArrayF32Py {
        inner,
        _ctx: Some(ctx),
        device_id: Some(dev_id),
    })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "pattern_recognition_cuda_host_f32")]
#[pyo3(signature = (open_f32, high_f32, low_f32, close_f32, device_id=0))]
pub fn pattern_recognition_cuda_host_f32_py<'py>(
    py: Python<'py>,
    open_f32: numpy::PyReadonlyArray1<'py, f32>,
    high_f32: numpy::PyReadonlyArray1<'py, f32>,
    low_f32: numpy::PyReadonlyArray1<'py, f32>,
    close_f32: numpy::PyReadonlyArray1<'py, f32>,
    device_id: usize,
) -> PyResult<Bound<'py, PyDict>> {
    use crate::cuda::cuda_available;
    use crate::cuda::pattern_recognition_wrapper::CudaPatternRecognition;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let open_slice = open_f32.as_slice()?;
    let high_slice = high_f32.as_slice()?;
    let low_slice = low_f32.as_slice()?;
    let close_slice = close_f32.as_slice()?;

    let (values_f32, pattern_ids, rows, cols) = py.allow_threads(|| {
        let cuda = CudaPatternRecognition::new(device_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let native_ids = CudaPatternRecognition::native_supported_pattern_ids();
        let rows = native_ids.len();
        let cols = close_slice.len();
        let d_u8 = cuda
            .compute_native_matrix_device_from_host_inputs(
                open_slice,
                high_slice,
                low_slice,
                close_slice,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.synchronize()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut host_u8 = vec![0u8; rows.saturating_mul(cols)];
        d_u8.copy_to(host_u8.as_mut_slice())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut values_f32 = Vec::with_capacity(host_u8.len());
        values_f32.extend(host_u8.into_iter().map(|x| x as f32));
        let pattern_ids = native_ids
            .iter()
            .map(|x| (*x).to_string())
            .collect::<Vec<_>>();
        Ok::<_, PyErr>((values_f32, pattern_ids, rows, cols))
    })?;

    let dict = PyDict::new(py);
    let values = values_f32.into_pyarray(py);
    dict.set_item("values", values.reshape((rows, cols))?)?;
    dict.set_item("pattern_ids", pattern_ids)?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    dict.set_item("warmup", py.None())?;
    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "vector_ta", unsendable)]
pub struct PatternRecognitionDeviceBitmaskU64Py {
    pub(crate) buf: Option<DeviceBuffer<u64>>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) words_per_row: usize,
    pub(crate) _ctx: Arc<Context>,
    pub(crate) device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl PatternRecognitionDeviceBitmaskU64Py {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("shape", (self.rows, self.words_per_row))?;
        d.set_item("typestr", "<u8")?;
        d.set_item(
            "strides",
            (
                self.words_per_row * std::mem::size_of::<u64>(),
                std::mem::size_of::<u64>(),
            ),
        )?;
        let ptr = self
            .buf
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("buffer already exported via __dlpack__"))?
            .as_device_ptr()
            .as_raw() as usize;
        d.set_item("data", (ptr, false))?;
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (2, self.device_id as i32)
    }

    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<pyo3::PyObject>,
        max_version: Option<pyo3::PyObject>,
        dl_device: Option<pyo3::PyObject>,
        copy: Option<pyo3::PyObject>,
    ) -> PyResult<PyObject> {
        let (kdl, alloc_dev) = self.__dlpack_device__();
        if let Some(dev_obj) = dl_device.as_ref() {
            if let Ok((dev_ty, dev_id)) = dev_obj.extract::<(i32, i32)>(py) {
                if dev_ty != kdl || dev_id != alloc_dev {
                    let wants_copy = copy
                        .as_ref()
                        .and_then(|c| c.extract::<bool>(py).ok())
                        .unwrap_or(false);
                    if wants_copy {
                        return Err(PyValueError::new_err(
                            "device copy not implemented for __dlpack__",
                        ));
                    } else {
                        return Err(PyValueError::new_err("dl_device mismatch for __dlpack__"));
                    }
                }
            }
        }

        if let Some(obj) = stream.as_ref() {
            if let Ok(i) = obj.extract::<i64>(py) {
                if i == 0 {
                    return Err(PyValueError::new_err(
                        "__dlpack__: stream 0 is disallowed for CUDA",
                    ));
                }
            }
        }

        let buf = self
            .buf
            .take()
            .ok_or_else(|| PyValueError::new_err("__dlpack__ may only be called once"))?;
        let rows = self.rows;
        let cols = self.words_per_row;
        let max_version_bound = max_version.map(|obj| obj.into_bound(py));
        export_u64_cuda_dlpack_2d(py, buf, rows, cols, alloc_dev, max_version_bound)
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "pattern_recognition_cuda_bitmask_dev")]
#[pyo3(signature = (open_f32, high_f32, low_f32, close_f32, device_id=0))]
pub fn pattern_recognition_cuda_bitmask_dev_py<'py>(
    py: Python<'py>,
    open_f32: numpy::PyReadonlyArray1<'py, f32>,
    high_f32: numpy::PyReadonlyArray1<'py, f32>,
    low_f32: numpy::PyReadonlyArray1<'py, f32>,
    close_f32: numpy::PyReadonlyArray1<'py, f32>,
    device_id: usize,
) -> PyResult<Bound<'py, PyDict>> {
    use crate::cuda::cuda_available;
    use crate::cuda::pattern_recognition_wrapper::CudaPatternRecognition;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let open_slice = open_f32.as_slice()?;
    let high_slice = high_f32.as_slice()?;
    let low_slice = low_f32.as_slice()?;
    let close_slice = close_f32.as_slice()?;

    let (bitmask, pattern_ids) = py.allow_threads(|| {
        let cuda = CudaPatternRecognition::new(device_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let bitmask = cuda
            .compute_native_matrix_bitmask_u64_device_from_host_inputs(
                open_slice,
                high_slice,
                low_slice,
                close_slice,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let pattern_ids = CudaPatternRecognition::native_supported_pattern_ids()
            .iter()
            .map(|x| (*x).to_string())
            .collect::<Vec<_>>();
        Ok::<_, PyErr>((bitmask, pattern_ids))
    })?;

    let handle = Py::new(
        py,
        PatternRecognitionDeviceBitmaskU64Py {
            buf: Some(bitmask.buf),
            rows: bitmask.rows,
            cols: bitmask.cols,
            words_per_row: bitmask.words_per_row,
            _ctx: bitmask.ctx,
            device_id: bitmask.device_id,
        },
    )?;

    let dict = PyDict::new(py);
    dict.set_item("values", handle)?;
    dict.set_item("pattern_ids", pattern_ids)?;
    dict.set_item("rows", bitmask.rows)?;
    dict.set_item("cols", bitmask.cols)?;
    dict.set_item("words_per_row", bitmask.words_per_row)?;
    dict.set_item("warmup", py.None())?;
    Ok(dict)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
struct PatternRecognitionJsOutput {
    values: Vec<u8>,
    pattern_ids: Vec<String>,
    rows: usize,
    cols: usize,
    warmup: Option<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
struct PatternRecognitionIntoMeta {
    pattern_ids: Vec<String>,
    rows: usize,
    cols: usize,
    warmup: Option<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn pattern_recognition_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> Result<JsValue, JsValue> {
    let input = PatternRecognitionInput::with_default_slices(open, high, low, close);
    let output = pattern_recognition(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let js_output = PatternRecognitionJsOutput {
        values: output.values_u8,
        pattern_ids: output
            .pattern_ids
            .into_iter()
            .map(|x| x.to_string())
            .collect(),
        rows: output.rows,
        cols: output.cols,
        warmup: output.warmup,
    };
    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn pattern_recognition_alloc(len: usize) -> *mut u8 {
    let mut vec = Vec::<u8>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn pattern_recognition_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, len);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn pattern_recognition_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut u8,
    len: usize,
    out_len: usize,
) -> Result<JsValue, JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to pattern_recognition_into",
        ));
    }
    if len == 0 {
        return Err(JsValue::from_str("len must be > 0"));
    }

    let rows = PATTERN_RUNNERS.len();
    let expected_out_len = rows
        .checked_mul(len)
        .ok_or_else(|| JsValue::from_str("rows*len overflow"))?;
    if out_len < expected_out_len {
        return Err(JsValue::from_str("output buffer too small"));
    }

    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let output = pattern_recognition(&PatternRecognitionInput::with_default_slices(
            open, high, low, close,
        ))
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        if output.values_u8.len() != expected_out_len {
            return Err(JsValue::from_str("unexpected output length"));
        }
        let out_slice = std::slice::from_raw_parts_mut(out_ptr, expected_out_len);
        out_slice.copy_from_slice(&output.values_u8);

        let meta = PatternRecognitionIntoMeta {
            pattern_ids: output
                .pattern_ids
                .into_iter()
                .map(|x| x.to_string())
                .collect(),
            rows: output.rows,
            cols: output.cols,
            warmup: output.warmup,
        };
        serde_wasm_bindgen::to_value(&meta)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn pattern_recognition_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = pattern_recognition_js(open, high, low, close)?;
    crate::write_wasm_object_f64_outputs("pattern_recognition_output_into_js", &value, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_candles(len: usize) -> Candles {
        let mut timestamp = Vec::with_capacity(len);
        let mut open = Vec::with_capacity(len);
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        let mut close = Vec::with_capacity(len);
        let mut volume = Vec::with_capacity(len);

        for i in 0..len {
            let base = 100.0 + (i as f64 * 0.01) + (((i % 11) as f64) - 5.0) * 0.12;
            let o = base + (((i % 5) as f64) - 2.0) * 0.05;
            let c = base + (((i % 7) as f64) - 3.0) * 0.04;
            let h = o.max(c) + 0.08 + ((i % 3) as f64) * 0.01;
            let l = o.min(c) - 0.08 - ((i % 4) as f64) * 0.01;

            timestamp.push(i as i64);
            open.push(o);
            high.push(h);
            low.push(l);
            close.push(c);
            volume.push(1000.0 + i as f64);
        }

        Candles::new(timestamp, open, high, low, close, volume)
    }

    fn adversarial_candles(len: usize) -> Candles {
        let mut timestamp = Vec::with_capacity(len);
        let mut open = Vec::with_capacity(len);
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        let mut close = Vec::with_capacity(len);
        let mut volume = Vec::with_capacity(len);

        let mut prev_close: f64 = 100.0;
        for i in 0..len {
            let band = i % 12;
            let (o, c, hi_pad, lo_pad): (f64, f64, f64, f64) = match band {
                0 => (prev_close + 5.0, prev_close + 5.0, 0.0, 0.0),
                1 => (prev_close + 1.2, prev_close - 0.8, 0.3, 0.7),
                2 => (prev_close - 3.5, prev_close + 2.2, 1.4, 1.1),
                3 => (prev_close, prev_close, 0.02, 0.02),
                4 => (prev_close + 0.01, prev_close - 0.01, 3.5, 3.5),
                5 => (prev_close + 8.0, prev_close + 9.5, 0.5, 0.2),
                6 => (prev_close - 7.0, prev_close - 8.2, 0.6, 0.3),
                7 => (prev_close + 0.5, prev_close + 0.6, 2.2, 0.1),
                8 => (prev_close - 0.6, prev_close - 0.5, 0.1, 2.2),
                9 => (prev_close + 0.8, prev_close + 0.8, 0.01, 0.01),
                10 => (prev_close - 0.8, prev_close - 0.8, 0.01, 0.01),
                _ => (prev_close + 1.5, prev_close - 1.5, 0.4, 0.4),
            };

            let h = o.max(c) + hi_pad;
            let l = o.min(c) - lo_pad;

            timestamp.push(i as i64);
            open.push(o);
            high.push(h);
            low.push(l);
            close.push(c);
            volume.push(1000.0 + ((i * 17) % 113) as f64);
            prev_close = c;
        }

        Candles::new(timestamp, open, high, low, close, volume)
    }

    #[test]
    fn pattern_specs_align_with_runners() {
        assert_eq!(PATTERN_SPECS.len(), PATTERN_RUNNERS.len());
        for (idx, spec) in PATTERN_SPECS.iter().enumerate() {
            assert_eq!(spec.row_index, idx);
            assert_eq!(spec.id, PATTERN_RUNNERS[idx].id);
            assert_eq!(spec.category, PATTERN_RUNNERS[idx].category);
            assert_eq!(
                pattern_type_from_id(spec.id),
                Some(PATTERN_RUNNERS[idx].pattern_type)
            );
        }
    }

    #[test]
    fn shared_primitive_pass_matches_scalar_helpers() {
        let candles = adversarial_candles(128);
        let prim =
            build_shared_primitives(&candles.open, &candles.high, &candles.low, &candles.close);

        assert_eq!(prim.body.len(), candles.close.len());
        assert_eq!(prim.range.len(), candles.close.len());
        assert_eq!(prim.upper_shadow.len(), candles.close.len());
        assert_eq!(prim.lower_shadow.len(), candles.close.len());
        assert_eq!(prim.direction.len(), candles.close.len());
        assert_eq!(prim.body_gap_up.len(), candles.close.len());
        assert_eq!(prim.body_gap_down.len(), candles.close.len());

        for i in 0..candles.close.len() {
            let o = candles.open[i];
            let h = candles.high[i];
            let l = candles.low[i];
            let c = candles.close[i];

            assert!((prim.body[i] - real_body(o, c)).abs() <= 1e-12);
            assert!((prim.range[i] - (h - l)).abs() <= 1e-12);
            assert!((prim.upper_shadow[i] - upper_shadow(o, h, c)).abs() <= 1e-12);
            assert!((prim.lower_shadow[i] - lower_shadow(o, l, c)).abs() <= 1e-12);
            assert_eq!(prim.direction[i], candle_color(o, c) as i8);

            if i == 0 {
                assert_eq!(prim.body_gap_up[i], 0);
                assert_eq!(prim.body_gap_down[i], 0);
            } else {
                let cur_min = o.min(c);
                let cur_max = o.max(c);
                let prev_min = candles.open[i - 1].min(candles.close[i - 1]);
                let prev_max = candles.open[i - 1].max(candles.close[i - 1]);
                assert_eq!(prim.body_gap_up[i], (cur_min > prev_max) as u8);
                assert_eq!(prim.body_gap_down[i], (cur_max < prev_min) as u8);
            }
        }
    }

    #[test]
    fn pattern_dynamic_route_matches_direct_function() {
        let candles = synthetic_candles(256);
        let input = PatternInput::with_default_candles(&candles, PatternType::CdlDoji);
        let direct = cdldoji(&input).unwrap();
        let routed = pattern(&input).unwrap();
        let routed_kernel = pattern_with_kernel(&input, Kernel::Scalar).unwrap();

        assert_eq!(direct.values, routed.values);
        assert_eq!(direct.values, routed_kernel.values);
    }

    #[test]
    fn pattern_recognition_output_contract_holds() {
        let candles = synthetic_candles(320);
        let input = PatternRecognitionInput::with_default_candles(&candles);
        let out = pattern_recognition(&input).unwrap();

        assert_eq!(out.rows, PATTERN_RUNNERS.len());
        assert_eq!(out.cols, candles.close.len());
        assert_eq!(out.values_u8.len(), out.rows * out.cols);
        assert_eq!(out.pattern_ids.len(), out.rows);
        assert!(out.values_u8.iter().all(|v| *v == 0 || *v == 1));

        for spec in list_patterns() {
            assert_eq!(out.pattern_ids[spec.row_index], spec.id);
        }
    }

    #[test]
    fn pattern_recognition_matches_direct_pattern_functions() {
        let candles = synthetic_candles(400);
        let input = PatternRecognitionInput::with_default_candles(&candles);
        let out = pattern_recognition_with_kernel(&input, Kernel::Scalar).unwrap();

        for (row, runner) in PATTERN_RUNNERS.iter().enumerate() {
            let direct_input = PatternInput::from_candles(
                &candles,
                PatternParams {
                    pattern_type: runner.pattern_type.clone(),
                    penetration: 0.0,
                },
            );
            let direct = (runner.run)(&direct_input).unwrap();
            let series = extract_pattern_series(&out, runner.id).unwrap();
            for (idx, v) in direct.values.iter().enumerate() {
                let mapped = if *v == 0 { 0 } else { 1 };
                assert_eq!(series[idx], mapped, "pattern={} bar={}", runner.id, idx);
                let hit = pattern_hit(&out, runner.id, idx).unwrap();
                assert_eq!(hit, mapped != 0, "pattern={} bar={}", runner.id, idx);
            }
            assert_eq!(out.pattern_ids[row], runner.id);
        }
    }

    #[test]
    fn pattern_recognition_matches_direct_on_adversarial_fixture() {
        let candles = adversarial_candles(320);
        let input = PatternRecognitionInput::with_default_candles(&candles);
        let out = pattern_recognition_with_kernel(&input, Kernel::Scalar).unwrap();

        for runner in PATTERN_RUNNERS.iter() {
            let direct_input = PatternInput::from_candles(
                &candles,
                PatternParams {
                    pattern_type: runner.pattern_type,
                    penetration: 0.0,
                },
            );
            let direct = (runner.run)(&direct_input).unwrap();
            let series = extract_pattern_series(&out, runner.id).unwrap();
            for (idx, v) in direct.values.iter().enumerate() {
                let mapped = if *v == 0 { 0 } else { 1 };
                assert_eq!(series[idx], mapped, "pattern={} bar={}", runner.id, idx);
            }
        }
    }

    #[test]
    fn from_slices_matches_from_candles() {
        let candles = synthetic_candles(256);
        let from_candles =
            pattern_recognition(&PatternRecognitionInput::with_default_candles(&candles)).unwrap();
        let from_slices = pattern_recognition(&PatternRecognitionInput::with_default_slices(
            &candles.open,
            &candles.high,
            &candles.low,
            &candles.close,
        ))
        .unwrap();

        assert_eq!(from_candles.rows, from_slices.rows);
        assert_eq!(from_candles.cols, from_slices.cols);
        assert_eq!(from_candles.pattern_ids, from_slices.pattern_ids);
        assert_eq!(from_candles.values_u8, from_slices.values_u8);
    }

    #[test]
    fn packed_bitmask_export_matches_dense_values() {
        let candles = synthetic_candles(191);
        let out =
            pattern_recognition(&PatternRecognitionInput::with_default_candles(&candles)).unwrap();
        let packed = out.to_bitmask_u64();

        assert_eq!(packed.rows, out.rows);
        assert_eq!(packed.cols, out.cols);
        assert_eq!(packed.pattern_ids, out.pattern_ids);
        assert_eq!(packed.warmup, out.warmup);
        assert_eq!(packed.words_per_row, out.cols.div_ceil(64));
        assert_eq!(packed.words_u64.len(), packed.rows * packed.words_per_row);

        for row in 0..out.rows {
            for col in 0..out.cols {
                let dense_idx = row * out.cols + col;
                let word = row * packed.words_per_row + (col / 64);
                let bit = col % 64;
                let packed_hit = ((packed.words_u64[word] >> bit) & 1) != 0;
                assert_eq!(packed_hit, out.values_u8[dense_idx] != 0);
            }
        }
    }

    #[test]
    fn from_slices_rejects_length_mismatch() {
        let candles = synthetic_candles(64);
        let res = pattern_recognition(&PatternRecognitionInput::from_slices(
            &candles.open[..63],
            &candles.high,
            &candles.low,
            &candles.close,
            PatternRecognitionParams::default(),
        ));

        match res {
            Err(PatternRecognitionError::DataLengthMismatch {
                open,
                high,
                low,
                close,
            }) => {
                assert_eq!(open, 63);
                assert_eq!(high, 64);
                assert_eq!(low, 64);
                assert_eq!(close, 64);
            }
            other => panic!("expected DataLengthMismatch, got {:?}", other),
        }
    }

    #[test]
    fn direct_pattern_function_accepts_slice_input() {
        let candles = synthetic_candles(256);
        let params = PatternParams {
            pattern_type: PatternType::CdlEngulfing,
            penetration: 0.0,
        };
        let from_candles =
            cdlengulfing(&PatternInput::from_candles(&candles, params.clone())).unwrap();
        let from_slices = cdlengulfing(&PatternInput::from_slices(
            &candles.open,
            &candles.high,
            &candles.low,
            &candles.close,
            params,
        ))
        .unwrap();
        assert_eq!(from_candles.values, from_slices.values);
    }
}
