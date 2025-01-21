use crate::utilities::data_loader::Candles;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum PatternData<'a> {
    Candles { candles: &'a Candles },
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

#[derive(Debug, Clone)]
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

    pub fn with_default_candles(candles: &'a Candles, pattern_type: PatternType) -> Self {
        Self {
            data: PatternData::Candles { candles },
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

#[inline]
pub fn cdl2crows(input: &PatternInput) -> Result<PatternOutput, PatternError> {
    const BODY_LONG_PERIOD: usize = 10;

    let (open, high, low, close) = match &input.data {
        PatternData::Candles { candles } => {
            let open = candles
                .select_candle_field("open")
                .map_err(|e| PatternError::CandleFieldError(e.to_string()))?;
            let high = candles
                .select_candle_field("high")
                .map_err(|e| PatternError::CandleFieldError(e.to_string()))?;
            let low = candles
                .select_candle_field("low")
                .map_err(|e| PatternError::CandleFieldError(e.to_string()))?;
            let close = candles
                .select_candle_field("close")
                .map_err(|e| PatternError::CandleFieldError(e.to_string()))?;

            (open, high, low, close)
        }
    };

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

    let (open, high, low, close) = match &input.data {
        PatternData::Candles { candles } => {
            let open = candles
                .select_candle_field("open")
                .map_err(|e| PatternError::CandleFieldError(e.to_string()))?;
            let high = candles
                .select_candle_field("high")
                .map_err(|e| PatternError::CandleFieldError(e.to_string()))?;
            let low = candles
                .select_candle_field("low")
                .map_err(|e| PatternError::CandleFieldError(e.to_string()))?;
            let close = candles
                .select_candle_field("close")
                .map_err(|e| PatternError::CandleFieldError(e.to_string()))?;

            (open, high, low, close)
        }
    };

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

    let (open, high, low, close) = match &input.data {
        PatternData::Candles { candles } => {
            let open = candles
                .select_candle_field("open")
                .map_err(|e| PatternError::CandleFieldError(e.to_string()))?;
            let high = candles
                .select_candle_field("high")
                .map_err(|e| PatternError::CandleFieldError(e.to_string()))?;
            let low = candles
                .select_candle_field("low")
                .map_err(|e| PatternError::CandleFieldError(e.to_string()))?;
            let close = candles
                .select_candle_field("close")
                .map_err(|e| PatternError::CandleFieldError(e.to_string()))?;

            (open, high, low, close)
        }
    };

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

    let (open, high, low, close) = match &input.data {
        PatternData::Candles { candles } => {
            let open = candles
                .select_candle_field("open")
                .map_err(|e| PatternError::CandleFieldError(e.to_string()))?;
            let high = candles
                .select_candle_field("high")
                .map_err(|e| PatternError::CandleFieldError(e.to_string()))?;
            let low = candles
                .select_candle_field("low")
                .map_err(|e| PatternError::CandleFieldError(e.to_string()))?;
            let close = candles
                .select_candle_field("close")
                .map_err(|e| PatternError::CandleFieldError(e.to_string()))?;

            (open, high, low, close)
        }
    };

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
