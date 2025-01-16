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
    pub values: Vec<f64>,
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
