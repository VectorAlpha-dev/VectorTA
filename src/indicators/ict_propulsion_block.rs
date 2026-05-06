#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

use std::collections::VecDeque;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

const DEFAULT_SWING_LENGTH: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    serde(rename_all = "snake_case")
)]
pub enum IctPropulsionBlockMitigationPrice {
    Close,
    Wick,
}

impl Default for IctPropulsionBlockMitigationPrice {
    fn default() -> Self {
        Self::Close
    }
}

impl IctPropulsionBlockMitigationPrice {
    #[inline]
    fn as_str(self) -> &'static str {
        match self {
            Self::Close => "close",
            Self::Wick => "wick",
        }
    }
}

#[derive(Debug, Clone)]
pub enum IctPropulsionBlockData<'a> {
    Candles(&'a Candles),
    Slices {
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct IctPropulsionBlockOutput {
    pub bullish_high: Vec<f64>,
    pub bullish_low: Vec<f64>,
    pub bullish_kind: Vec<f64>,
    pub bullish_active: Vec<f64>,
    pub bullish_mitigated: Vec<f64>,
    pub bullish_new: Vec<f64>,
    pub bearish_high: Vec<f64>,
    pub bearish_low: Vec<f64>,
    pub bearish_kind: Vec<f64>,
    pub bearish_active: Vec<f64>,
    pub bearish_mitigated: Vec<f64>,
    pub bearish_new: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct IctPropulsionBlockParams {
    pub swing_length: Option<usize>,
    pub mitigation_price: Option<IctPropulsionBlockMitigationPrice>,
}

impl Default for IctPropulsionBlockParams {
    fn default() -> Self {
        Self {
            swing_length: Some(DEFAULT_SWING_LENGTH),
            mitigation_price: Some(IctPropulsionBlockMitigationPrice::Close),
        }
    }
}

#[derive(Debug, Clone)]
pub struct IctPropulsionBlockInput<'a> {
    pub data: IctPropulsionBlockData<'a>,
    pub params: IctPropulsionBlockParams,
}

impl<'a> IctPropulsionBlockInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: IctPropulsionBlockParams) -> Self {
        Self {
            data: IctPropulsionBlockData::Candles(candles),
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: IctPropulsionBlockParams,
    ) -> Self {
        Self {
            data: IctPropulsionBlockData::Slices {
                open,
                high,
                low,
                close,
            },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, IctPropulsionBlockParams::default())
    }

    #[inline]
    pub fn get_swing_length(&self) -> usize {
        self.params.swing_length.unwrap_or(DEFAULT_SWING_LENGTH)
    }

    #[inline]
    pub fn get_mitigation_price(&self) -> IctPropulsionBlockMitigationPrice {
        self.params
            .mitigation_price
            .unwrap_or(IctPropulsionBlockMitigationPrice::Close)
    }

    #[inline]
    pub fn as_refs(&'a self) -> (&'a [f64], &'a [f64], &'a [f64], &'a [f64]) {
        match &self.data {
            IctPropulsionBlockData::Candles(candles) => {
                (&candles.open, &candles.high, &candles.low, &candles.close)
            }
            IctPropulsionBlockData::Slices {
                open,
                high,
                low,
                close,
            } => (*open, *high, *low, *close),
        }
    }
}

#[derive(Clone, Debug)]
pub struct IctPropulsionBlockBuilder {
    swing_length: Option<usize>,
    mitigation_price: Option<IctPropulsionBlockMitigationPrice>,
    kernel: Kernel,
}

impl Default for IctPropulsionBlockBuilder {
    fn default() -> Self {
        Self {
            swing_length: None,
            mitigation_price: None,
            kernel: Kernel::Auto,
        }
    }
}

impl IctPropulsionBlockBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn swing_length(mut self, value: usize) -> Self {
        self.swing_length = Some(value);
        self
    }

    #[inline]
    pub fn mitigation_price(mut self, value: IctPropulsionBlockMitigationPrice) -> Self {
        self.mitigation_price = Some(value);
        self
    }

    #[inline]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<IctPropulsionBlockOutput, IctPropulsionBlockError> {
        let input = IctPropulsionBlockInput::from_candles(
            candles,
            IctPropulsionBlockParams {
                swing_length: self.swing_length,
                mitigation_price: self.mitigation_price,
            },
        );
        ict_propulsion_block_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<IctPropulsionBlockOutput, IctPropulsionBlockError> {
        let input = IctPropulsionBlockInput::from_slices(
            open,
            high,
            low,
            close,
            IctPropulsionBlockParams {
                swing_length: self.swing_length,
                mitigation_price: self.mitigation_price,
            },
        );
        ict_propulsion_block_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(self) -> Result<IctPropulsionBlockStream, IctPropulsionBlockError> {
        IctPropulsionBlockStream::try_new(IctPropulsionBlockParams {
            swing_length: self.swing_length,
            mitigation_price: self.mitigation_price,
        })
    }
}

#[derive(Debug, Error)]
pub enum IctPropulsionBlockError {
    #[error("ict_propulsion_block: Empty input data.")]
    EmptyInputData,
    #[error(
        "ict_propulsion_block: Input length mismatch: open={open}, high={high}, low={low}, close={close}"
    )]
    DataLengthMismatch {
        open: usize,
        high: usize,
        low: usize,
        close: usize,
    },
    #[error("ict_propulsion_block: All input values are invalid.")]
    AllValuesNaN,
    #[error("ict_propulsion_block: Invalid swing_length: {swing_length}")]
    InvalidSwingLength { swing_length: usize },
    #[error("ict_propulsion_block: Invalid mitigation_price: {mitigation_price}")]
    InvalidMitigationPrice { mitigation_price: String },
    #[error("ict_propulsion_block: Output length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("ict_propulsion_block: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("ict_propulsion_block: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Copy, Debug)]
struct SwingState {
    value: f64,
    index: usize,
    cross: bool,
}

impl SwingState {
    #[inline]
    fn na() -> Self {
        Self {
            value: f64::NAN,
            index: 0,
            cross: false,
        }
    }

    #[inline]
    fn is_valid(self) -> bool {
        self.value.is_finite()
    }
}

#[derive(Clone, Copy, Debug)]
struct BlockSeed {
    index: usize,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
}

#[derive(Clone, Copy, Debug)]
struct BlockState {
    start_index: usize,
    end_index: usize,
    confirmed_index: usize,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    is_propulsion: bool,
    is_active: bool,
    is_mitigated: bool,
}

impl BlockState {
    #[inline]
    fn new(seed: BlockSeed, confirmed_index: usize, is_propulsion: bool) -> Self {
        Self {
            start_index: seed.index,
            end_index: confirmed_index,
            confirmed_index,
            open: seed.open,
            high: seed.high,
            low: seed.low,
            close: seed.close,
            is_propulsion,
            is_active: true,
            is_mitigated: false,
        }
    }

    #[inline]
    fn kind_value(self) -> f64 {
        if self.is_propulsion {
            2.0
        } else {
            1.0
        }
    }
}

#[inline(always)]
fn valid_bar(open: f64, high: f64, low: f64, close: f64) -> bool {
    open.is_finite() && high.is_finite() && low.is_finite() && close.is_finite() && high >= low
}

#[inline(always)]
fn validate_lengths(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> Result<(), IctPropulsionBlockError> {
    if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(IctPropulsionBlockError::EmptyInputData);
    }
    if open.len() != high.len() || high.len() != low.len() || low.len() != close.len() {
        return Err(IctPropulsionBlockError::DataLengthMismatch {
            open: open.len(),
            high: high.len(),
            low: low.len(),
            close: close.len(),
        });
    }
    Ok(())
}

#[inline(always)]
fn first_valid_bar(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Option<usize> {
    (0..close.len()).find(|&i| valid_bar(open[i], high[i], low[i], close[i]))
}

#[inline(always)]
fn validate_params(
    swing_length: usize,
    mitigation_price: IctPropulsionBlockMitigationPrice,
) -> Result<(), IctPropulsionBlockError> {
    if swing_length == 0 {
        return Err(IctPropulsionBlockError::InvalidSwingLength { swing_length });
    }
    match mitigation_price {
        IctPropulsionBlockMitigationPrice::Close | IctPropulsionBlockMitigationPrice::Wick => {}
    }
    Ok(())
}

#[inline(always)]
fn push_front_limited(blocks: &mut Vec<BlockState>, block: BlockState) {
    blocks.insert(0, block);
    if blocks.len() > 2 {
        blocks.truncate(2);
    }
}

#[inline(always)]
fn reset_deque(deque: &mut VecDeque<usize>) {
    deque.clear();
}

#[inline(always)]
fn select_bullish_seed(
    current: usize,
    swing_index: usize,
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> BlockSeed {
    let mut best = BlockSeed {
        index: current - 1,
        open: open[current - 1],
        high: high[current - 1],
        low: low[current - 1],
        close: close[current - 1],
    };
    let diff = current.saturating_sub(swing_index);
    for offset in 1..diff {
        let idx = current - offset;
        if open[idx] > close[idx] && low[idx] <= best.low {
            best = BlockSeed {
                index: idx,
                open: open[idx],
                high: high[idx],
                low: low[idx],
                close: close[idx],
            };
        }
    }
    best
}

#[inline(always)]
fn select_bearish_seed(
    current: usize,
    swing_index: usize,
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> BlockSeed {
    let mut best = BlockSeed {
        index: current - 1,
        open: open[current - 1],
        high: high[current - 1],
        low: low[current - 1],
        close: close[current - 1],
    };
    let diff = current.saturating_sub(swing_index);
    for offset in 1..diff {
        let idx = current - offset;
        if open[idx] < close[idx] && high[idx] >= best.high {
            best = BlockSeed {
                index: idx,
                open: open[idx],
                high: high[idx],
                low: low[idx],
                close: close[idx],
            };
        }
    }
    best
}

#[inline(always)]
fn maybe_insert_bullish_order_block(
    blocks: &mut Vec<BlockState>,
    seed: BlockSeed,
    current: usize,
) -> bool {
    if blocks.is_empty() {
        push_front_limited(blocks, BlockState::new(seed, current, false));
        return true;
    }

    if blocks[0].is_mitigated
        && blocks[0].is_propulsion
        && blocks.len() > 1
        && !blocks[1].is_propulsion
    {
        blocks[1].is_mitigated = true;
    }

    let recent = blocks[0];
    if !(recent.is_mitigated
        || (!recent.is_mitigated && seed.high > recent.high && seed.index > recent.start_index))
    {
        return false;
    }

    push_front_limited(blocks, BlockState::new(seed, current, false));
    if blocks.len() > 1 {
        blocks[1].is_active = false;
        if seed.index <= blocks[1].end_index
            && blocks[0].low <= blocks[1].high
            && blocks[0].high > blocks[1].high
        {
            blocks[0].is_propulsion = true;
        }
    }
    true
}

#[inline(always)]
fn maybe_insert_bearish_order_block(
    blocks: &mut Vec<BlockState>,
    seed: BlockSeed,
    current: usize,
) -> bool {
    if blocks.is_empty() {
        push_front_limited(blocks, BlockState::new(seed, current, false));
        return true;
    }

    if blocks[0].is_mitigated
        && blocks[0].is_propulsion
        && blocks.len() > 1
        && !blocks[1].is_propulsion
    {
        blocks[1].is_mitigated = true;
    }

    let recent = blocks[0];
    if !(recent.is_mitigated
        || (!recent.is_mitigated && seed.low < recent.low && seed.index > recent.start_index))
    {
        return false;
    }

    push_front_limited(blocks, BlockState::new(seed, current, false));
    if blocks.len() > 1 {
        blocks[1].is_active = false;
        if seed.index <= blocks[1].end_index
            && blocks[0].high >= blocks[1].low
            && blocks[0].low < blocks[1].low
        {
            blocks[0].is_propulsion = true;
        }
    }
    true
}

#[inline(always)]
fn insert_bullish_propulsion(
    blocks: &mut Vec<BlockState>,
    breach_index: usize,
    breach_high: f64,
    current: usize,
    open: &[f64],
    low: &[f64],
    close: &[f64],
) -> bool {
    if blocks.is_empty() {
        return false;
    }
    blocks[0].is_active = false;
    blocks[0].end_index = current;
    let seed = BlockSeed {
        index: breach_index,
        open: open[breach_index],
        high: breach_high,
        low: low[breach_index],
        close: close[breach_index],
    };
    push_front_limited(blocks, BlockState::new(seed, current, true));
    true
}

#[inline(always)]
fn insert_bearish_propulsion(
    blocks: &mut Vec<BlockState>,
    breach_index: usize,
    breach_low: f64,
    current: usize,
    open: &[f64],
    high: &[f64],
    close: &[f64],
) -> bool {
    if blocks.is_empty() {
        return false;
    }
    blocks[0].is_active = false;
    blocks[0].end_index = current;
    let seed = BlockSeed {
        index: breach_index,
        open: open[breach_index],
        high: high[breach_index],
        low: breach_low,
        close: close[breach_index],
    };
    push_front_limited(blocks, BlockState::new(seed, current, true));
    true
}

#[inline(always)]
fn write_snapshot(
    block: Option<&BlockState>,
    new_flag: f64,
    out_high: &mut [f64],
    out_low: &mut [f64],
    out_kind: &mut [f64],
    out_active: &mut [f64],
    out_mitigated: &mut [f64],
    out_new: &mut [f64],
    index: usize,
) {
    if let Some(block) = block {
        out_high[index] = block.high;
        out_low[index] = block.low;
        out_kind[index] = block.kind_value();
        out_active[index] = if block.is_active { 1.0 } else { 0.0 };
        out_mitigated[index] = if block.is_mitigated { 1.0 } else { 0.0 };
        out_new[index] = new_flag;
    } else {
        out_high[index] = f64::NAN;
        out_low[index] = f64::NAN;
        out_kind[index] = 0.0;
        out_active[index] = 0.0;
        out_mitigated[index] = 0.0;
        out_new[index] = new_flag;
    }
}

#[inline(always)]
fn normalize_kernel(kernel: Kernel) -> Kernel {
    match kernel {
        Kernel::Auto => Kernel::Scalar,
        other if other.is_batch() => other.to_non_batch(),
        other => other,
    }
}

#[allow(clippy::too_many_arguments)]
fn ict_propulsion_block_row_scalar(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    swing_length: usize,
    mitigation_price: IctPropulsionBlockMitigationPrice,
    out_bullish_high: &mut [f64],
    out_bullish_low: &mut [f64],
    out_bullish_kind: &mut [f64],
    out_bullish_active: &mut [f64],
    out_bullish_mitigated: &mut [f64],
    out_bullish_new: &mut [f64],
    out_bearish_high: &mut [f64],
    out_bearish_low: &mut [f64],
    out_bearish_kind: &mut [f64],
    out_bearish_active: &mut [f64],
    out_bearish_mitigated: &mut [f64],
    out_bearish_new: &mut [f64],
) {
    let len = close.len();
    let mut swing_os = 0i8;
    let mut swing_high = SwingState::na();
    let mut swing_low = SwingState::na();
    let mut bullish_breach = SwingState::na();
    let mut bearish_breach = SwingState::na();
    let mut bullish_breach_low_prev = f64::NAN;
    let mut bullish_breach_high_prev = f64::NAN;
    let mut bullish_breach_index_prev = 0usize;
    let mut bearish_breach_low_prev = f64::NAN;
    let mut bearish_breach_high_prev = f64::NAN;
    let mut bearish_breach_index_prev = 0usize;
    let mut bullish_blocks: Vec<BlockState> = Vec::with_capacity(2);
    let mut bearish_blocks: Vec<BlockState> = Vec::with_capacity(2);
    let mut max_high_window: VecDeque<usize> = VecDeque::with_capacity(swing_length + 1);
    let mut min_low_window: VecDeque<usize> = VecDeque::with_capacity(swing_length + 1);

    for i in 0..len {
        if !valid_bar(open[i], high[i], low[i], close[i]) {
            out_bullish_high[i] = f64::NAN;
            out_bullish_low[i] = f64::NAN;
            out_bullish_kind[i] = f64::NAN;
            out_bullish_active[i] = f64::NAN;
            out_bullish_mitigated[i] = f64::NAN;
            out_bullish_new[i] = f64::NAN;
            out_bearish_high[i] = f64::NAN;
            out_bearish_low[i] = f64::NAN;
            out_bearish_kind[i] = f64::NAN;
            out_bearish_active[i] = f64::NAN;
            out_bearish_mitigated[i] = f64::NAN;
            out_bearish_new[i] = f64::NAN;
            swing_os = 0;
            swing_high = SwingState::na();
            swing_low = SwingState::na();
            bullish_breach = SwingState::na();
            bearish_breach = SwingState::na();
            bullish_breach_low_prev = f64::NAN;
            bullish_breach_high_prev = f64::NAN;
            bearish_breach_low_prev = f64::NAN;
            bearish_breach_high_prev = f64::NAN;
            bullish_blocks.clear();
            bearish_blocks.clear();
            reset_deque(&mut max_high_window);
            reset_deque(&mut min_low_window);
            continue;
        }

        while let Some(&idx) = max_high_window.back() {
            if high[idx] <= high[i] {
                max_high_window.pop_back();
            } else {
                break;
            }
        }
        max_high_window.push_back(i);

        while let Some(&idx) = min_low_window.back() {
            if low[idx] >= low[i] {
                min_low_window.pop_back();
            } else {
                break;
            }
        }
        min_low_window.push_back(i);

        let window_start = i.saturating_sub(swing_length.saturating_sub(1));
        while let Some(&idx) = max_high_window.front() {
            if idx < window_start {
                max_high_window.pop_front();
            } else {
                break;
            }
        }
        while let Some(&idx) = min_low_window.front() {
            if idx < window_start {
                min_low_window.pop_front();
            } else {
                break;
            }
        }

        if i >= swing_length {
            let candidate = i - swing_length;
            let upper = high[*max_high_window.front().unwrap()];
            let lower = low[*min_low_window.front().unwrap()];
            let mut next_os = swing_os;
            if high[candidate] > upper {
                next_os = 0;
            } else if low[candidate] < lower {
                next_os = 1;
            }

            if next_os == 0 && swing_os != 0 {
                swing_high = SwingState {
                    value: high[candidate],
                    index: candidate,
                    cross: false,
                };
            }
            if next_os == 1 && swing_os != 1 {
                swing_low = SwingState {
                    value: low[candidate],
                    index: candidate,
                    cross: false,
                };
            }
            swing_os = next_os;
        }

        let mut breach_low = low[i];
        let mut breach_high = high[i];
        let mut breach_index = i;
        if let Some(current) = bullish_blocks.first() {
            let condition = low[i] <= current.high
                && low[i] > current.low
                && i > current.confirmed_index
                && !current.is_mitigated
                && current.is_active
                && !current.is_propulsion
                && open[i] > current.high;
            if condition {
                let prev_low = if bullish_breach_low_prev.is_finite() {
                    bullish_breach_low_prev
                } else {
                    low[i]
                };
                breach_low = low[i].min(prev_low);
                if breach_low == low[i] || !bullish_breach_high_prev.is_finite() {
                    breach_high = high[i];
                    breach_index = i;
                } else {
                    breach_high = bullish_breach_high_prev;
                    breach_index = bullish_breach_index_prev;
                }
                bullish_breach = SwingState {
                    value: breach_high,
                    index: breach_index,
                    cross: false,
                };
            }
        }
        bullish_breach_low_prev = breach_low;
        bullish_breach_high_prev = breach_high;
        bullish_breach_index_prev = breach_index;

        let mut bear_breach_low = low[i];
        let mut bear_breach_high = high[i];
        let mut bear_breach_index = i;
        if let Some(current) = bearish_blocks.first() {
            let condition = high[i] >= current.low
                && high[i] < current.high
                && i > current.confirmed_index
                && !current.is_mitigated
                && current.is_active
                && !current.is_propulsion
                && open[i] < current.low;
            if condition {
                let prev_high = if bearish_breach_high_prev.is_finite() {
                    bearish_breach_high_prev
                } else {
                    high[i]
                };
                bear_breach_high = high[i].max(prev_high);
                if bear_breach_high == high[i] || !bearish_breach_low_prev.is_finite() {
                    bear_breach_low = low[i];
                    bear_breach_index = i;
                } else {
                    bear_breach_low = bearish_breach_low_prev;
                    bear_breach_index = bearish_breach_index_prev;
                }
                bearish_breach = SwingState {
                    value: bear_breach_low,
                    index: bear_breach_index,
                    cross: false,
                };
            }
        }
        bearish_breach_low_prev = bear_breach_low;
        bearish_breach_high_prev = bear_breach_high;
        bearish_breach_index_prev = bear_breach_index;

        let mut bullish_new = 0.0;
        let mut bearish_new = 0.0;

        if swing_high.is_valid()
            && !swing_high.cross
            && close[i] > swing_high.value
            && i > swing_high.index
        {
            swing_high.cross = true;
            let seed = select_bullish_seed(i, swing_high.index, open, high, low, close);
            if maybe_insert_bullish_order_block(&mut bullish_blocks, seed, i) {
                bullish_new = 1.0;
            }
        }

        if let Some(recent) = bullish_blocks.first() {
            let create_pb = bullish_breach.is_valid()
                && close[i] > bullish_breach.value
                && !bullish_breach.cross
                && !recent.is_mitigated
                && bullish_breach.index > recent.confirmed_index;
            if create_pb {
                bullish_breach.cross = true;
                if insert_bullish_propulsion(
                    &mut bullish_blocks,
                    bullish_breach.index,
                    bullish_breach.value,
                    i,
                    open,
                    low,
                    close,
                ) {
                    bullish_new = 1.0;
                }
            }
        }

        for block in &mut bullish_blocks {
            if block.is_active && !block.is_mitigated {
                let mitigated = match mitigation_price {
                    IctPropulsionBlockMitigationPrice::Close => close[i] < block.low,
                    IctPropulsionBlockMitigationPrice::Wick => low[i] < block.low,
                };
                if mitigated {
                    block.is_mitigated = true;
                }
                block.end_index = i;
            }
        }

        if swing_low.is_valid()
            && !swing_low.cross
            && close[i] < swing_low.value
            && i > swing_low.index
        {
            swing_low.cross = true;
            let seed = select_bearish_seed(i, swing_low.index, open, high, low, close);
            if maybe_insert_bearish_order_block(&mut bearish_blocks, seed, i) {
                bearish_new = 1.0;
            }
        }

        if let Some(recent) = bearish_blocks.first() {
            let create_pb = bearish_breach.is_valid()
                && close[i] < bearish_breach.value
                && !bearish_breach.cross
                && !recent.is_mitigated
                && bearish_breach.index > recent.confirmed_index;
            if create_pb {
                bearish_breach.cross = true;
                if insert_bearish_propulsion(
                    &mut bearish_blocks,
                    bearish_breach.index,
                    bearish_breach.value,
                    i,
                    open,
                    high,
                    close,
                ) {
                    bearish_new = 1.0;
                }
            }
        }

        for block in &mut bearish_blocks {
            if block.is_active && !block.is_mitigated {
                let mitigated = match mitigation_price {
                    IctPropulsionBlockMitigationPrice::Close => close[i] > block.high,
                    IctPropulsionBlockMitigationPrice::Wick => high[i] > block.high,
                };
                if mitigated {
                    block.is_mitigated = true;
                }
                block.end_index = i;
            }
        }

        write_snapshot(
            bullish_blocks.first(),
            bullish_new,
            out_bullish_high,
            out_bullish_low,
            out_bullish_kind,
            out_bullish_active,
            out_bullish_mitigated,
            out_bullish_new,
            i,
        );
        write_snapshot(
            bearish_blocks.first(),
            bearish_new,
            out_bearish_high,
            out_bearish_low,
            out_bearish_kind,
            out_bearish_active,
            out_bearish_mitigated,
            out_bearish_new,
            i,
        );
    }
}

#[inline]
pub fn ict_propulsion_block(
    input: &IctPropulsionBlockInput,
) -> Result<IctPropulsionBlockOutput, IctPropulsionBlockError> {
    ict_propulsion_block_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn ict_propulsion_block_with_kernel(
    input: &IctPropulsionBlockInput,
    kernel: Kernel,
) -> Result<IctPropulsionBlockOutput, IctPropulsionBlockError> {
    let (open, high, low, close) = input.as_refs();
    validate_lengths(open, high, low, close)?;
    let swing_length = input.get_swing_length();
    let mitigation_price = input.get_mitigation_price();
    validate_params(swing_length, mitigation_price)?;
    let first_valid =
        first_valid_bar(open, high, low, close).ok_or(IctPropulsionBlockError::AllValuesNaN)?;
    let _kernel = normalize_kernel(kernel);
    let len = close.len();

    let mut bullish_high = alloc_with_nan_prefix(len, first_valid);
    let mut bullish_low = alloc_with_nan_prefix(len, first_valid);
    let mut bullish_kind = alloc_with_nan_prefix(len, first_valid);
    let mut bullish_active = alloc_with_nan_prefix(len, first_valid);
    let mut bullish_mitigated = alloc_with_nan_prefix(len, first_valid);
    let mut bullish_new = alloc_with_nan_prefix(len, first_valid);
    let mut bearish_high = alloc_with_nan_prefix(len, first_valid);
    let mut bearish_low = alloc_with_nan_prefix(len, first_valid);
    let mut bearish_kind = alloc_with_nan_prefix(len, first_valid);
    let mut bearish_active = alloc_with_nan_prefix(len, first_valid);
    let mut bearish_mitigated = alloc_with_nan_prefix(len, first_valid);
    let mut bearish_new = alloc_with_nan_prefix(len, first_valid);

    ict_propulsion_block_row_scalar(
        open,
        high,
        low,
        close,
        swing_length,
        mitigation_price,
        &mut bullish_high,
        &mut bullish_low,
        &mut bullish_kind,
        &mut bullish_active,
        &mut bullish_mitigated,
        &mut bullish_new,
        &mut bearish_high,
        &mut bearish_low,
        &mut bearish_kind,
        &mut bearish_active,
        &mut bearish_mitigated,
        &mut bearish_new,
    );

    Ok(IctPropulsionBlockOutput {
        bullish_high,
        bullish_low,
        bullish_kind,
        bullish_active,
        bullish_mitigated,
        bullish_new,
        bearish_high,
        bearish_low,
        bearish_kind,
        bearish_active,
        bearish_mitigated,
        bearish_new,
    })
}

#[inline]
pub fn ict_propulsion_block_into_slice(
    out_bullish_high: &mut [f64],
    out_bullish_low: &mut [f64],
    out_bullish_kind: &mut [f64],
    out_bullish_active: &mut [f64],
    out_bullish_mitigated: &mut [f64],
    out_bullish_new: &mut [f64],
    out_bearish_high: &mut [f64],
    out_bearish_low: &mut [f64],
    out_bearish_kind: &mut [f64],
    out_bearish_active: &mut [f64],
    out_bearish_mitigated: &mut [f64],
    out_bearish_new: &mut [f64],
    input: &IctPropulsionBlockInput,
    kernel: Kernel,
) -> Result<(), IctPropulsionBlockError> {
    let (open, high, low, close) = input.as_refs();
    validate_lengths(open, high, low, close)?;
    let len = close.len();
    if out_bullish_high.len() != len
        || out_bullish_low.len() != len
        || out_bullish_kind.len() != len
        || out_bullish_active.len() != len
        || out_bullish_mitigated.len() != len
        || out_bullish_new.len() != len
        || out_bearish_high.len() != len
        || out_bearish_low.len() != len
        || out_bearish_kind.len() != len
        || out_bearish_active.len() != len
        || out_bearish_mitigated.len() != len
        || out_bearish_new.len() != len
    {
        return Err(IctPropulsionBlockError::OutputLengthMismatch {
            expected: len,
            got: out_bullish_high
                .len()
                .max(out_bullish_low.len())
                .max(out_bullish_kind.len())
                .max(out_bullish_active.len())
                .max(out_bullish_mitigated.len())
                .max(out_bullish_new.len())
                .max(out_bearish_high.len())
                .max(out_bearish_low.len())
                .max(out_bearish_kind.len())
                .max(out_bearish_active.len())
                .max(out_bearish_mitigated.len())
                .max(out_bearish_new.len()),
        });
    }

    let swing_length = input.get_swing_length();
    let mitigation_price = input.get_mitigation_price();
    validate_params(swing_length, mitigation_price)?;
    let _kernel = normalize_kernel(kernel);

    ict_propulsion_block_row_scalar(
        open,
        high,
        low,
        close,
        swing_length,
        mitigation_price,
        out_bullish_high,
        out_bullish_low,
        out_bullish_kind,
        out_bullish_active,
        out_bullish_mitigated,
        out_bullish_new,
        out_bearish_high,
        out_bearish_low,
        out_bearish_kind,
        out_bearish_active,
        out_bearish_mitigated,
        out_bearish_new,
    );
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn ict_propulsion_block_into(
    input: &IctPropulsionBlockInput,
    out_bullish_high: &mut [f64],
    out_bullish_low: &mut [f64],
    out_bullish_kind: &mut [f64],
    out_bullish_active: &mut [f64],
    out_bullish_mitigated: &mut [f64],
    out_bullish_new: &mut [f64],
    out_bearish_high: &mut [f64],
    out_bearish_low: &mut [f64],
    out_bearish_kind: &mut [f64],
    out_bearish_active: &mut [f64],
    out_bearish_mitigated: &mut [f64],
    out_bearish_new: &mut [f64],
) -> Result<(), IctPropulsionBlockError> {
    ict_propulsion_block_into_slice(
        out_bullish_high,
        out_bullish_low,
        out_bullish_kind,
        out_bullish_active,
        out_bullish_mitigated,
        out_bullish_new,
        out_bearish_high,
        out_bearish_low,
        out_bearish_kind,
        out_bearish_active,
        out_bearish_mitigated,
        out_bearish_new,
        input,
        Kernel::Auto,
    )
}

#[derive(Clone, Debug)]
pub struct IctPropulsionBlockStream {
    swing_length: usize,
    mitigation_price: IctPropulsionBlockMitigationPrice,
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
}

impl IctPropulsionBlockStream {
    #[inline]
    pub fn try_new(params: IctPropulsionBlockParams) -> Result<Self, IctPropulsionBlockError> {
        let swing_length = params.swing_length.unwrap_or(DEFAULT_SWING_LENGTH);
        let mitigation_price = params
            .mitigation_price
            .unwrap_or(IctPropulsionBlockMitigationPrice::Close);
        validate_params(swing_length, mitigation_price)?;
        Ok(Self {
            swing_length,
            mitigation_price,
            open: Vec::new(),
            high: Vec::new(),
            low: Vec::new(),
            close: Vec::new(),
        })
    }

    #[inline]
    pub fn update(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64)> {
        if !valid_bar(open, high, low, close) {
            self.open.clear();
            self.high.clear();
            self.low.clear();
            self.close.clear();
            return None;
        }

        self.open.push(open);
        self.high.push(high);
        self.low.push(low);
        self.close.push(close);

        let input = IctPropulsionBlockInput::from_slices(
            &self.open,
            &self.high,
            &self.low,
            &self.close,
            IctPropulsionBlockParams {
                swing_length: Some(self.swing_length),
                mitigation_price: Some(self.mitigation_price),
            },
        );
        let out = ict_propulsion_block_with_kernel(&input, Kernel::Scalar).ok()?;
        let last = self.close.len() - 1;
        Some((
            out.bullish_high[last],
            out.bullish_low[last],
            out.bullish_kind[last],
            out.bullish_active[last],
            out.bullish_mitigated[last],
            out.bullish_new[last],
            out.bearish_high[last],
            out.bearish_low[last],
            out.bearish_kind[last],
            out.bearish_active[last],
            out.bearish_mitigated[last],
            out.bearish_new[last],
        ))
    }
}

#[derive(Clone, Debug)]
pub struct IctPropulsionBlockBatchRange {
    pub swing_length: (usize, usize, usize),
    pub mitigation_price: (bool, bool),
}

impl Default for IctPropulsionBlockBatchRange {
    fn default() -> Self {
        Self {
            swing_length: (DEFAULT_SWING_LENGTH, DEFAULT_SWING_LENGTH, 0),
            mitigation_price: (true, false),
        }
    }
}

#[derive(Clone, Debug)]
pub struct IctPropulsionBlockBatchOutput {
    pub bullish_high: Vec<f64>,
    pub bullish_low: Vec<f64>,
    pub bullish_kind: Vec<f64>,
    pub bullish_active: Vec<f64>,
    pub bullish_mitigated: Vec<f64>,
    pub bullish_new: Vec<f64>,
    pub bearish_high: Vec<f64>,
    pub bearish_low: Vec<f64>,
    pub bearish_kind: Vec<f64>,
    pub bearish_active: Vec<f64>,
    pub bearish_mitigated: Vec<f64>,
    pub bearish_new: Vec<f64>,
    pub combos: Vec<IctPropulsionBlockParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug)]
pub struct IctPropulsionBlockBatchBuilder {
    range: IctPropulsionBlockBatchRange,
    kernel: Kernel,
}

impl Default for IctPropulsionBlockBatchBuilder {
    fn default() -> Self {
        Self {
            range: IctPropulsionBlockBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl IctPropulsionBlockBatchBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn swing_length_range(mut self, range: (usize, usize, usize)) -> Self {
        self.range.swing_length = range;
        self
    }

    #[inline]
    pub fn mitigation_price_toggle(mut self, include_close: bool, include_wick: bool) -> Self {
        self.range.mitigation_price = (include_close, include_wick);
        self
    }

    #[inline]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline]
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<IctPropulsionBlockBatchOutput, IctPropulsionBlockError> {
        ict_propulsion_block_batch_with_kernel(open, high, low, close, &self.range, self.kernel)
    }

    #[inline]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<IctPropulsionBlockBatchOutput, IctPropulsionBlockError> {
        ict_propulsion_block_batch_with_kernel(
            &candles.open,
            &candles.high,
            &candles.low,
            &candles.close,
            &self.range,
            self.kernel,
        )
    }
}

#[inline]
fn expand_axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, IctPropulsionBlockError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }
    let mut out = Vec::new();
    if start <= end {
        let mut value = start;
        while value <= end {
            out.push(value);
            match value.checked_add(step) {
                Some(next) => value = next,
                None => break,
            }
        }
    } else {
        let mut value = start;
        loop {
            if value < end {
                break;
            }
            out.push(value);
            match value.checked_sub(step) {
                Some(next) => value = next,
                None => break,
            }
        }
    }
    if out.is_empty() {
        return Err(IctPropulsionBlockError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

#[inline]
pub fn expand_grid_ict_propulsion_block(
    range: &IctPropulsionBlockBatchRange,
) -> Result<Vec<IctPropulsionBlockParams>, IctPropulsionBlockError> {
    let swing_lengths = expand_axis_usize(range.swing_length)?;
    let mut mitigation_prices = Vec::new();
    if range.mitigation_price.0 {
        mitigation_prices.push(IctPropulsionBlockMitigationPrice::Close);
    }
    if range.mitigation_price.1 {
        mitigation_prices.push(IctPropulsionBlockMitigationPrice::Wick);
    }
    if mitigation_prices.is_empty() {
        mitigation_prices.push(IctPropulsionBlockMitigationPrice::Close);
    }

    let mut out = Vec::with_capacity(swing_lengths.len().saturating_mul(mitigation_prices.len()));
    for &swing_length in &swing_lengths {
        for &mitigation_price in &mitigation_prices {
            out.push(IctPropulsionBlockParams {
                swing_length: Some(swing_length),
                mitigation_price: Some(mitigation_price),
            });
        }
    }
    Ok(out)
}

#[inline]
pub fn ict_propulsion_block_batch_with_kernel(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &IctPropulsionBlockBatchRange,
    kernel: Kernel,
) -> Result<IctPropulsionBlockBatchOutput, IctPropulsionBlockError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(IctPropulsionBlockError::InvalidKernelForBatch(other)),
    };
    ict_propulsion_block_batch_par_slice(open, high, low, close, sweep, batch_kernel.to_non_batch())
}

#[inline]
pub fn ict_propulsion_block_batch_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &IctPropulsionBlockBatchRange,
    kernel: Kernel,
) -> Result<IctPropulsionBlockBatchOutput, IctPropulsionBlockError> {
    ict_propulsion_block_batch_inner(open, high, low, close, sweep, kernel)
}

#[inline]
pub fn ict_propulsion_block_batch_par_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &IctPropulsionBlockBatchRange,
    kernel: Kernel,
) -> Result<IctPropulsionBlockBatchOutput, IctPropulsionBlockError> {
    ict_propulsion_block_batch_inner(open, high, low, close, sweep, kernel)
}

fn ict_propulsion_block_batch_inner(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &IctPropulsionBlockBatchRange,
    _kernel: Kernel,
) -> Result<IctPropulsionBlockBatchOutput, IctPropulsionBlockError> {
    validate_lengths(open, high, low, close)?;
    let combos = expand_grid_ict_propulsion_block(sweep)?;
    for params in &combos {
        validate_params(
            params.swing_length.unwrap_or(DEFAULT_SWING_LENGTH),
            params
                .mitigation_price
                .unwrap_or(IctPropulsionBlockMitigationPrice::Close),
        )?;
    }

    let _first_valid =
        first_valid_bar(open, high, low, close).ok_or(IctPropulsionBlockError::AllValuesNaN)?;
    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or(IctPropulsionBlockError::OutputLengthMismatch {
            expected: usize::MAX,
            got: 0,
        })?;

    let bullish_high_matrix = make_uninit_matrix(rows, cols);
    let bullish_low_matrix = make_uninit_matrix(rows, cols);
    let bullish_kind_matrix = make_uninit_matrix(rows, cols);
    let bullish_active_matrix = make_uninit_matrix(rows, cols);
    let bullish_mitigated_matrix = make_uninit_matrix(rows, cols);
    let bullish_new_matrix = make_uninit_matrix(rows, cols);
    let bearish_high_matrix = make_uninit_matrix(rows, cols);
    let bearish_low_matrix = make_uninit_matrix(rows, cols);
    let bearish_kind_matrix = make_uninit_matrix(rows, cols);
    let bearish_active_matrix = make_uninit_matrix(rows, cols);
    let bearish_mitigated_matrix = make_uninit_matrix(rows, cols);
    let bearish_new_matrix = make_uninit_matrix(rows, cols);

    let mut bullish_high_guard = ManuallyDrop::new(bullish_high_matrix);
    let mut bullish_low_guard = ManuallyDrop::new(bullish_low_matrix);
    let mut bullish_kind_guard = ManuallyDrop::new(bullish_kind_matrix);
    let mut bullish_active_guard = ManuallyDrop::new(bullish_active_matrix);
    let mut bullish_mitigated_guard = ManuallyDrop::new(bullish_mitigated_matrix);
    let mut bullish_new_guard = ManuallyDrop::new(bullish_new_matrix);
    let mut bearish_high_guard = ManuallyDrop::new(bearish_high_matrix);
    let mut bearish_low_guard = ManuallyDrop::new(bearish_low_matrix);
    let mut bearish_kind_guard = ManuallyDrop::new(bearish_kind_matrix);
    let mut bearish_active_guard = ManuallyDrop::new(bearish_active_matrix);
    let mut bearish_mitigated_guard = ManuallyDrop::new(bearish_mitigated_matrix);
    let mut bearish_new_guard = ManuallyDrop::new(bearish_new_matrix);

    let bullish_high_mu: &mut [MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(bullish_high_guard.as_mut_ptr(), bullish_high_guard.len())
    };
    let bullish_low_mu: &mut [MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(bullish_low_guard.as_mut_ptr(), bullish_low_guard.len())
    };
    let bullish_kind_mu: &mut [MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(bullish_kind_guard.as_mut_ptr(), bullish_kind_guard.len())
    };
    let bullish_active_mu: &mut [MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(
            bullish_active_guard.as_mut_ptr(),
            bullish_active_guard.len(),
        )
    };
    let bullish_mitigated_mu: &mut [MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(
            bullish_mitigated_guard.as_mut_ptr(),
            bullish_mitigated_guard.len(),
        )
    };
    let bullish_new_mu: &mut [MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(bullish_new_guard.as_mut_ptr(), bullish_new_guard.len())
    };
    let bearish_high_mu: &mut [MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(bearish_high_guard.as_mut_ptr(), bearish_high_guard.len())
    };
    let bearish_low_mu: &mut [MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(bearish_low_guard.as_mut_ptr(), bearish_low_guard.len())
    };
    let bearish_kind_mu: &mut [MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(bearish_kind_guard.as_mut_ptr(), bearish_kind_guard.len())
    };
    let bearish_active_mu: &mut [MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(
            bearish_active_guard.as_mut_ptr(),
            bearish_active_guard.len(),
        )
    };
    let bearish_mitigated_mu: &mut [MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(
            bearish_mitigated_guard.as_mut_ptr(),
            bearish_mitigated_guard.len(),
        )
    };
    let bearish_new_mu: &mut [MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(bearish_new_guard.as_mut_ptr(), bearish_new_guard.len())
    };

    for row in 0..rows {
        let base = row * cols;
        let out_bullish_high = unsafe {
            std::slice::from_raw_parts_mut(
                bullish_high_mu[base..base + cols].as_mut_ptr() as *mut f64,
                cols,
            )
        };
        let out_bullish_low = unsafe {
            std::slice::from_raw_parts_mut(
                bullish_low_mu[base..base + cols].as_mut_ptr() as *mut f64,
                cols,
            )
        };
        let out_bullish_kind = unsafe {
            std::slice::from_raw_parts_mut(
                bullish_kind_mu[base..base + cols].as_mut_ptr() as *mut f64,
                cols,
            )
        };
        let out_bullish_active = unsafe {
            std::slice::from_raw_parts_mut(
                bullish_active_mu[base..base + cols].as_mut_ptr() as *mut f64,
                cols,
            )
        };
        let out_bullish_mitigated = unsafe {
            std::slice::from_raw_parts_mut(
                bullish_mitigated_mu[base..base + cols].as_mut_ptr() as *mut f64,
                cols,
            )
        };
        let out_bullish_new = unsafe {
            std::slice::from_raw_parts_mut(
                bullish_new_mu[base..base + cols].as_mut_ptr() as *mut f64,
                cols,
            )
        };
        let out_bearish_high = unsafe {
            std::slice::from_raw_parts_mut(
                bearish_high_mu[base..base + cols].as_mut_ptr() as *mut f64,
                cols,
            )
        };
        let out_bearish_low = unsafe {
            std::slice::from_raw_parts_mut(
                bearish_low_mu[base..base + cols].as_mut_ptr() as *mut f64,
                cols,
            )
        };
        let out_bearish_kind = unsafe {
            std::slice::from_raw_parts_mut(
                bearish_kind_mu[base..base + cols].as_mut_ptr() as *mut f64,
                cols,
            )
        };
        let out_bearish_active = unsafe {
            std::slice::from_raw_parts_mut(
                bearish_active_mu[base..base + cols].as_mut_ptr() as *mut f64,
                cols,
            )
        };
        let out_bearish_mitigated = unsafe {
            std::slice::from_raw_parts_mut(
                bearish_mitigated_mu[base..base + cols].as_mut_ptr() as *mut f64,
                cols,
            )
        };
        let out_bearish_new = unsafe {
            std::slice::from_raw_parts_mut(
                bearish_new_mu[base..base + cols].as_mut_ptr() as *mut f64,
                cols,
            )
        };

        ict_propulsion_block_row_scalar(
            open,
            high,
            low,
            close,
            combos[row].swing_length.unwrap_or(DEFAULT_SWING_LENGTH),
            combos[row]
                .mitigation_price
                .unwrap_or(IctPropulsionBlockMitigationPrice::Close),
            out_bullish_high,
            out_bullish_low,
            out_bullish_kind,
            out_bullish_active,
            out_bullish_mitigated,
            out_bullish_new,
            out_bearish_high,
            out_bearish_low,
            out_bearish_kind,
            out_bearish_active,
            out_bearish_mitigated,
            out_bearish_new,
        );
    }

    let bullish_high = unsafe {
        Vec::from_raw_parts(
            bullish_high_guard.as_mut_ptr() as *mut f64,
            total,
            bullish_high_guard.capacity(),
        )
    };
    let bullish_low = unsafe {
        Vec::from_raw_parts(
            bullish_low_guard.as_mut_ptr() as *mut f64,
            total,
            bullish_low_guard.capacity(),
        )
    };
    let bullish_kind = unsafe {
        Vec::from_raw_parts(
            bullish_kind_guard.as_mut_ptr() as *mut f64,
            total,
            bullish_kind_guard.capacity(),
        )
    };
    let bullish_active = unsafe {
        Vec::from_raw_parts(
            bullish_active_guard.as_mut_ptr() as *mut f64,
            total,
            bullish_active_guard.capacity(),
        )
    };
    let bullish_mitigated = unsafe {
        Vec::from_raw_parts(
            bullish_mitigated_guard.as_mut_ptr() as *mut f64,
            total,
            bullish_mitigated_guard.capacity(),
        )
    };
    let bullish_new = unsafe {
        Vec::from_raw_parts(
            bullish_new_guard.as_mut_ptr() as *mut f64,
            total,
            bullish_new_guard.capacity(),
        )
    };
    let bearish_high = unsafe {
        Vec::from_raw_parts(
            bearish_high_guard.as_mut_ptr() as *mut f64,
            total,
            bearish_high_guard.capacity(),
        )
    };
    let bearish_low = unsafe {
        Vec::from_raw_parts(
            bearish_low_guard.as_mut_ptr() as *mut f64,
            total,
            bearish_low_guard.capacity(),
        )
    };
    let bearish_kind = unsafe {
        Vec::from_raw_parts(
            bearish_kind_guard.as_mut_ptr() as *mut f64,
            total,
            bearish_kind_guard.capacity(),
        )
    };
    let bearish_active = unsafe {
        Vec::from_raw_parts(
            bearish_active_guard.as_mut_ptr() as *mut f64,
            total,
            bearish_active_guard.capacity(),
        )
    };
    let bearish_mitigated = unsafe {
        Vec::from_raw_parts(
            bearish_mitigated_guard.as_mut_ptr() as *mut f64,
            total,
            bearish_mitigated_guard.capacity(),
        )
    };
    let bearish_new = unsafe {
        Vec::from_raw_parts(
            bearish_new_guard.as_mut_ptr() as *mut f64,
            total,
            bearish_new_guard.capacity(),
        )
    };

    Ok(IctPropulsionBlockBatchOutput {
        bullish_high,
        bullish_low,
        bullish_kind,
        bullish_active,
        bullish_mitigated,
        bullish_new,
        bearish_high,
        bearish_low,
        bearish_kind,
        bearish_active,
        bearish_mitigated,
        bearish_new,
        combos,
        rows,
        cols,
    })
}

#[allow(clippy::too_many_arguments)]
fn ict_propulsion_block_batch_inner_into(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &IctPropulsionBlockBatchRange,
    kernel: Kernel,
    out_bullish_high: &mut [f64],
    out_bullish_low: &mut [f64],
    out_bullish_kind: &mut [f64],
    out_bullish_active: &mut [f64],
    out_bullish_mitigated: &mut [f64],
    out_bullish_new: &mut [f64],
    out_bearish_high: &mut [f64],
    out_bearish_low: &mut [f64],
    out_bearish_kind: &mut [f64],
    out_bearish_active: &mut [f64],
    out_bearish_mitigated: &mut [f64],
    out_bearish_new: &mut [f64],
) -> Result<Vec<IctPropulsionBlockParams>, IctPropulsionBlockError> {
    validate_lengths(open, high, low, close)?;
    let combos = expand_grid_ict_propulsion_block(sweep)?;
    for params in &combos {
        validate_params(
            params.swing_length.unwrap_or(DEFAULT_SWING_LENGTH),
            params
                .mitigation_price
                .unwrap_or(IctPropulsionBlockMitigationPrice::Close),
        )?;
    }
    let _first_valid =
        first_valid_bar(open, high, low, close).ok_or(IctPropulsionBlockError::AllValuesNaN)?;
    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or(IctPropulsionBlockError::OutputLengthMismatch {
            expected: usize::MAX,
            got: 0,
        })?;

    if out_bullish_high.len() != total
        || out_bullish_low.len() != total
        || out_bullish_kind.len() != total
        || out_bullish_active.len() != total
        || out_bullish_mitigated.len() != total
        || out_bullish_new.len() != total
        || out_bearish_high.len() != total
        || out_bearish_low.len() != total
        || out_bearish_kind.len() != total
        || out_bearish_active.len() != total
        || out_bearish_mitigated.len() != total
        || out_bearish_new.len() != total
    {
        return Err(IctPropulsionBlockError::OutputLengthMismatch {
            expected: total,
            got: out_bullish_high
                .len()
                .max(out_bullish_low.len())
                .max(out_bullish_kind.len())
                .max(out_bullish_active.len())
                .max(out_bullish_mitigated.len())
                .max(out_bullish_new.len())
                .max(out_bearish_high.len())
                .max(out_bearish_low.len())
                .max(out_bearish_kind.len())
                .max(out_bearish_active.len())
                .max(out_bearish_mitigated.len())
                .max(out_bearish_new.len()),
        });
    }

    let _kernel = kernel;
    for row in 0..rows {
        let base = row * cols;
        ict_propulsion_block_row_scalar(
            open,
            high,
            low,
            close,
            combos[row].swing_length.unwrap_or(DEFAULT_SWING_LENGTH),
            combos[row]
                .mitigation_price
                .unwrap_or(IctPropulsionBlockMitigationPrice::Close),
            &mut out_bullish_high[base..base + cols],
            &mut out_bullish_low[base..base + cols],
            &mut out_bullish_kind[base..base + cols],
            &mut out_bullish_active[base..base + cols],
            &mut out_bullish_mitigated[base..base + cols],
            &mut out_bullish_new[base..base + cols],
            &mut out_bearish_high[base..base + cols],
            &mut out_bearish_low[base..base + cols],
            &mut out_bearish_kind[base..base + cols],
            &mut out_bearish_active[base..base + cols],
            &mut out_bearish_mitigated[base..base + cols],
            &mut out_bearish_new[base..base + cols],
        );
    }
    Ok(combos)
}

fn parse_mitigation_price(
    value: &str,
) -> Result<IctPropulsionBlockMitigationPrice, IctPropulsionBlockError> {
    if value.eq_ignore_ascii_case("close") || value.eq_ignore_ascii_case("closing_price") {
        return Ok(IctPropulsionBlockMitigationPrice::Close);
    }
    if value.eq_ignore_ascii_case("wick") {
        return Ok(IctPropulsionBlockMitigationPrice::Wick);
    }
    Err(IctPropulsionBlockError::InvalidMitigationPrice {
        mitigation_price: value.to_string(),
    })
}

#[cfg(feature = "python")]
#[pyfunction(name = "ict_propulsion_block")]
#[pyo3(signature = (open, high, low, close, swing_length=DEFAULT_SWING_LENGTH, mitigation_price="close", kernel=None))]
pub fn ict_propulsion_block_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    swing_length: usize,
    mitigation_price: &str,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let open = open.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let input = IctPropulsionBlockInput::from_slices(
        open,
        high,
        low,
        close,
        IctPropulsionBlockParams {
            swing_length: Some(swing_length),
            mitigation_price: Some(
                parse_mitigation_price(mitigation_price)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            ),
        },
    );
    let kernel = validate_kernel(kernel, false)?;
    let out = py
        .allow_threads(|| ict_propulsion_block_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.bullish_high.into_pyarray(py),
        out.bullish_low.into_pyarray(py),
        out.bullish_kind.into_pyarray(py),
        out.bullish_active.into_pyarray(py),
        out.bullish_mitigated.into_pyarray(py),
        out.bullish_new.into_pyarray(py),
        out.bearish_high.into_pyarray(py),
        out.bearish_low.into_pyarray(py),
        out.bearish_kind.into_pyarray(py),
        out.bearish_active.into_pyarray(py),
        out.bearish_mitigated.into_pyarray(py),
        out.bearish_new.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "IctPropulsionBlockStream")]
pub struct IctPropulsionBlockStreamPy {
    stream: IctPropulsionBlockStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl IctPropulsionBlockStreamPy {
    #[new]
    #[pyo3(signature = (swing_length=DEFAULT_SWING_LENGTH, mitigation_price="close"))]
    fn new(swing_length: usize, mitigation_price: &str) -> PyResult<Self> {
        let stream = IctPropulsionBlockStream::try_new(IctPropulsionBlockParams {
            swing_length: Some(swing_length),
            mitigation_price: Some(
                parse_mitigation_price(mitigation_price)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            ),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64)> {
        self.stream.update(open, high, low, close)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "ict_propulsion_block_batch")]
#[pyo3(signature = (open, high, low, close, swing_length_range, mitigation_price_toggle=(true, false), kernel=None))]
pub fn ict_propulsion_block_batch_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    swing_length_range: (usize, usize, usize),
    mitigation_price_toggle: (bool, bool),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let open = open.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let sweep = IctPropulsionBlockBatchRange {
        swing_length: swing_length_range,
        mitigation_price: mitigation_price_toggle,
    };
    let combos = expand_grid_ict_propulsion_block(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let bullish_high_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let bullish_low_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let bullish_kind_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let bullish_active_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let bullish_mitigated_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let bullish_new_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let bearish_high_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let bearish_low_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let bearish_kind_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let bearish_active_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let bearish_mitigated_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let bearish_new_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };

    let out_bullish_high = unsafe { bullish_high_arr.as_slice_mut()? };
    let out_bullish_low = unsafe { bullish_low_arr.as_slice_mut()? };
    let out_bullish_kind = unsafe { bullish_kind_arr.as_slice_mut()? };
    let out_bullish_active = unsafe { bullish_active_arr.as_slice_mut()? };
    let out_bullish_mitigated = unsafe { bullish_mitigated_arr.as_slice_mut()? };
    let out_bullish_new = unsafe { bullish_new_arr.as_slice_mut()? };
    let out_bearish_high = unsafe { bearish_high_arr.as_slice_mut()? };
    let out_bearish_low = unsafe { bearish_low_arr.as_slice_mut()? };
    let out_bearish_kind = unsafe { bearish_kind_arr.as_slice_mut()? };
    let out_bearish_active = unsafe { bearish_active_arr.as_slice_mut()? };
    let out_bearish_mitigated = unsafe { bearish_mitigated_arr.as_slice_mut()? };
    let out_bearish_new = unsafe { bearish_new_arr.as_slice_mut()? };

    let kernel = validate_kernel(kernel, true)?;
    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        ict_propulsion_block_batch_inner_into(
            open,
            high,
            low,
            close,
            &sweep,
            batch_kernel.to_non_batch(),
            out_bullish_high,
            out_bullish_low,
            out_bullish_kind,
            out_bullish_active,
            out_bullish_mitigated,
            out_bullish_new,
            out_bearish_high,
            out_bearish_low,
            out_bearish_kind,
            out_bearish_active,
            out_bearish_mitigated,
            out_bearish_new,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let swing_lengths: Vec<u64> = combos
        .iter()
        .map(|params| params.swing_length.unwrap_or(DEFAULT_SWING_LENGTH) as u64)
        .collect();
    let mitigation_prices: Vec<&str> = combos
        .iter()
        .map(|params| {
            params
                .mitigation_price
                .unwrap_or(IctPropulsionBlockMitigationPrice::Close)
                .as_str()
        })
        .collect();

    let dict = PyDict::new(py);
    dict.set_item("bullish_high", bullish_high_arr.reshape((rows, cols))?)?;
    dict.set_item("bullish_low", bullish_low_arr.reshape((rows, cols))?)?;
    dict.set_item("bullish_kind", bullish_kind_arr.reshape((rows, cols))?)?;
    dict.set_item("bullish_active", bullish_active_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "bullish_mitigated",
        bullish_mitigated_arr.reshape((rows, cols))?,
    )?;
    dict.set_item("bullish_new", bullish_new_arr.reshape((rows, cols))?)?;
    dict.set_item("bearish_high", bearish_high_arr.reshape((rows, cols))?)?;
    dict.set_item("bearish_low", bearish_low_arr.reshape((rows, cols))?)?;
    dict.set_item("bearish_kind", bearish_kind_arr.reshape((rows, cols))?)?;
    dict.set_item("bearish_active", bearish_active_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "bearish_mitigated",
        bearish_mitigated_arr.reshape((rows, cols))?,
    )?;
    dict.set_item("bearish_new", bearish_new_arr.reshape((rows, cols))?)?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    dict.set_item("swing_lengths", swing_lengths.into_pyarray(py))?;
    dict.set_item("mitigation_prices", PyList::new(py, mitigation_prices)?)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_ict_propulsion_block_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ict_propulsion_block_py, m)?)?;
    m.add_function(wrap_pyfunction!(ict_propulsion_block_batch_py, m)?)?;
    m.add_class::<IctPropulsionBlockStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IctPropulsionBlockJsOutput {
    bullish_high: Vec<f64>,
    bullish_low: Vec<f64>,
    bullish_kind: Vec<f64>,
    bullish_active: Vec<f64>,
    bullish_mitigated: Vec<f64>,
    bullish_new: Vec<f64>,
    bearish_high: Vec<f64>,
    bearish_low: Vec<f64>,
    bearish_kind: Vec<f64>,
    bearish_active: Vec<f64>,
    bearish_mitigated: Vec<f64>,
    bearish_new: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IctPropulsionBlockBatchConfig {
    swing_length_range: Vec<usize>,
    mitigation_price_toggle: Vec<bool>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IctPropulsionBlockBatchJsOutput {
    bullish_high: Vec<f64>,
    bullish_low: Vec<f64>,
    bullish_kind: Vec<f64>,
    bullish_active: Vec<f64>,
    bullish_mitigated: Vec<f64>,
    bullish_new: Vec<f64>,
    bearish_high: Vec<f64>,
    bearish_low: Vec<f64>,
    bearish_kind: Vec<f64>,
    bearish_active: Vec<f64>,
    bearish_mitigated: Vec<f64>,
    bearish_new: Vec<f64>,
    rows: usize,
    cols: usize,
    combos: Vec<IctPropulsionBlockParams>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "ict_propulsion_block")]
pub fn ict_propulsion_block_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    swing_length: usize,
    mitigation_price: &str,
) -> Result<JsValue, JsValue> {
    let input = IctPropulsionBlockInput::from_slices(
        open,
        high,
        low,
        close,
        IctPropulsionBlockParams {
            swing_length: Some(swing_length),
            mitigation_price: Some(
                parse_mitigation_price(mitigation_price)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?,
            ),
        },
    );
    let out = ict_propulsion_block_with_kernel(&input, Kernel::Scalar)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&IctPropulsionBlockJsOutput {
        bullish_high: out.bullish_high,
        bullish_low: out.bullish_low,
        bullish_kind: out.bullish_kind,
        bullish_active: out.bullish_active,
        bullish_mitigated: out.bullish_mitigated,
        bullish_new: out.bullish_new,
        bearish_high: out.bearish_high,
        bearish_low: out.bearish_low,
        bearish_kind: out.bearish_kind,
        bearish_active: out.bearish_active,
        bearish_mitigated: out.bearish_mitigated,
        bearish_new: out.bearish_new,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ict_propulsion_block_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    swing_length: usize,
    mitigation_price: &str,
) -> Result<(), JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to ict_propulsion_block_into",
        ));
    }

    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len * 12);
        let (out_bullish_high, rest) = out.split_at_mut(len);
        let (out_bullish_low, rest) = rest.split_at_mut(len);
        let (out_bullish_kind, rest) = rest.split_at_mut(len);
        let (out_bullish_active, rest) = rest.split_at_mut(len);
        let (out_bullish_mitigated, rest) = rest.split_at_mut(len);
        let (out_bullish_new, rest) = rest.split_at_mut(len);
        let (out_bearish_high, rest) = rest.split_at_mut(len);
        let (out_bearish_low, rest) = rest.split_at_mut(len);
        let (out_bearish_kind, rest) = rest.split_at_mut(len);
        let (out_bearish_active, rest) = rest.split_at_mut(len);
        let (out_bearish_mitigated, out_bearish_new) = rest.split_at_mut(len);
        let input = IctPropulsionBlockInput::from_slices(
            open,
            high,
            low,
            close,
            IctPropulsionBlockParams {
                swing_length: Some(swing_length),
                mitigation_price: Some(
                    parse_mitigation_price(mitigation_price)
                        .map_err(|e| JsValue::from_str(&e.to_string()))?,
                ),
            },
        );
        ict_propulsion_block_into_slice(
            out_bullish_high,
            out_bullish_low,
            out_bullish_kind,
            out_bullish_active,
            out_bullish_mitigated,
            out_bullish_new,
            out_bearish_high,
            out_bearish_low,
            out_bearish_kind,
            out_bearish_active,
            out_bearish_mitigated,
            out_bearish_new,
            &input,
            Kernel::Scalar,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "ict_propulsion_block_into_host")]
pub fn ict_propulsion_block_into_host(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    out_ptr: *mut f64,
    swing_length: usize,
    mitigation_price: &str,
) -> Result<(), JsValue> {
    if out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to ict_propulsion_block_into_host",
        ));
    }

    unsafe {
        let len = close.len();
        let out = std::slice::from_raw_parts_mut(out_ptr, len * 12);
        let (out_bullish_high, rest) = out.split_at_mut(len);
        let (out_bullish_low, rest) = rest.split_at_mut(len);
        let (out_bullish_kind, rest) = rest.split_at_mut(len);
        let (out_bullish_active, rest) = rest.split_at_mut(len);
        let (out_bullish_mitigated, rest) = rest.split_at_mut(len);
        let (out_bullish_new, rest) = rest.split_at_mut(len);
        let (out_bearish_high, rest) = rest.split_at_mut(len);
        let (out_bearish_low, rest) = rest.split_at_mut(len);
        let (out_bearish_kind, rest) = rest.split_at_mut(len);
        let (out_bearish_active, rest) = rest.split_at_mut(len);
        let (out_bearish_mitigated, out_bearish_new) = rest.split_at_mut(len);
        let input = IctPropulsionBlockInput::from_slices(
            open,
            high,
            low,
            close,
            IctPropulsionBlockParams {
                swing_length: Some(swing_length),
                mitigation_price: Some(
                    parse_mitigation_price(mitigation_price)
                        .map_err(|e| JsValue::from_str(&e.to_string()))?,
                ),
            },
        );
        ict_propulsion_block_into_slice(
            out_bullish_high,
            out_bullish_low,
            out_bullish_kind,
            out_bullish_active,
            out_bullish_mitigated,
            out_bullish_new,
            out_bearish_high,
            out_bearish_low,
            out_bearish_kind,
            out_bearish_active,
            out_bearish_mitigated,
            out_bearish_new,
            &input,
            Kernel::Scalar,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ict_propulsion_block_alloc(len: usize) -> *mut f64 {
    let mut buf = vec![0.0_f64; len * 12];
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ict_propulsion_block_free(ptr: *mut f64, len: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, len * 12);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "ict_propulsion_block_batch")]
pub fn ict_propulsion_block_batch_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: IctPropulsionBlockBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.swing_length_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: swing_length_range must have exactly 3 elements [start, end, step]",
        ));
    }
    if config.mitigation_price_toggle.len() != 2 {
        return Err(JsValue::from_str(
            "Invalid config: mitigation_price_toggle must have exactly 2 booleans [include_close, include_wick]",
        ));
    }

    let sweep = IctPropulsionBlockBatchRange {
        swing_length: (
            config.swing_length_range[0],
            config.swing_length_range[1],
            config.swing_length_range[2],
        ),
        mitigation_price: (
            config.mitigation_price_toggle[0],
            config.mitigation_price_toggle[1],
        ),
    };
    let out =
        ict_propulsion_block_batch_with_kernel(open, high, low, close, &sweep, Kernel::ScalarBatch)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&IctPropulsionBlockBatchJsOutput {
        bullish_high: out.bullish_high,
        bullish_low: out.bullish_low,
        bullish_kind: out.bullish_kind,
        bullish_active: out.bullish_active,
        bullish_mitigated: out.bullish_mitigated,
        bullish_new: out.bullish_new,
        bearish_high: out.bearish_high,
        bearish_low: out.bearish_low,
        bearish_kind: out.bearish_kind,
        bearish_active: out.bearish_active,
        bearish_mitigated: out.bearish_mitigated,
        bearish_new: out.bearish_new,
        rows: out.rows,
        cols: out.cols,
        combos: out.combos,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn ict_propulsion_block_batch_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    bullish_high_ptr: *mut f64,
    bullish_low_ptr: *mut f64,
    bullish_kind_ptr: *mut f64,
    bullish_active_ptr: *mut f64,
    bullish_mitigated_ptr: *mut f64,
    bullish_new_ptr: *mut f64,
    bearish_high_ptr: *mut f64,
    bearish_low_ptr: *mut f64,
    bearish_kind_ptr: *mut f64,
    bearish_active_ptr: *mut f64,
    bearish_mitigated_ptr: *mut f64,
    bearish_new_ptr: *mut f64,
    len: usize,
    swing_start: usize,
    swing_end: usize,
    swing_step: usize,
    include_close: bool,
    include_wick: bool,
) -> Result<usize, JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || bullish_high_ptr.is_null()
        || bullish_low_ptr.is_null()
        || bullish_kind_ptr.is_null()
        || bullish_active_ptr.is_null()
        || bullish_mitigated_ptr.is_null()
        || bullish_new_ptr.is_null()
        || bearish_high_ptr.is_null()
        || bearish_low_ptr.is_null()
        || bearish_kind_ptr.is_null()
        || bearish_active_ptr.is_null()
        || bearish_mitigated_ptr.is_null()
        || bearish_new_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to ict_propulsion_block_batch_into",
        ));
    }

    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let sweep = IctPropulsionBlockBatchRange {
            swing_length: (swing_start, swing_end, swing_step),
            mitigation_price: (include_close, include_wick),
        };
        let combos = expand_grid_ict_propulsion_block(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let total = combos.len().checked_mul(len).ok_or_else(|| {
            JsValue::from_str("rows*cols overflow in ict_propulsion_block_batch_into")
        })?;

        let out_bullish_high = std::slice::from_raw_parts_mut(bullish_high_ptr, total);
        let out_bullish_low = std::slice::from_raw_parts_mut(bullish_low_ptr, total);
        let out_bullish_kind = std::slice::from_raw_parts_mut(bullish_kind_ptr, total);
        let out_bullish_active = std::slice::from_raw_parts_mut(bullish_active_ptr, total);
        let out_bullish_mitigated = std::slice::from_raw_parts_mut(bullish_mitigated_ptr, total);
        let out_bullish_new = std::slice::from_raw_parts_mut(bullish_new_ptr, total);
        let out_bearish_high = std::slice::from_raw_parts_mut(bearish_high_ptr, total);
        let out_bearish_low = std::slice::from_raw_parts_mut(bearish_low_ptr, total);
        let out_bearish_kind = std::slice::from_raw_parts_mut(bearish_kind_ptr, total);
        let out_bearish_active = std::slice::from_raw_parts_mut(bearish_active_ptr, total);
        let out_bearish_mitigated = std::slice::from_raw_parts_mut(bearish_mitigated_ptr, total);
        let out_bearish_new = std::slice::from_raw_parts_mut(bearish_new_ptr, total);

        ict_propulsion_block_batch_inner_into(
            open,
            high,
            low,
            close,
            &sweep,
            Kernel::Scalar,
            out_bullish_high,
            out_bullish_low,
            out_bullish_kind,
            out_bullish_active,
            out_bullish_mitigated,
            out_bullish_new,
            out_bearish_high,
            out_bearish_low,
            out_bearish_kind,
            out_bearish_active,
            out_bearish_mitigated,
            out_bearish_new,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(combos.len())
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ict_propulsion_block_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    swing_length: usize,
    mitigation_price: &str,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = ict_propulsion_block_js(open, high, low, close, swing_length, mitigation_price)?;
    crate::write_wasm_object_f64_outputs("ict_propulsion_block_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ict_propulsion_block_batch_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = ict_propulsion_block_batch_js(open, high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "ict_propulsion_block_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::dispatch::{
        compute_cpu_batch, IndicatorBatchRequest, IndicatorDataRef, IndicatorParamSet, ParamKV,
        ParamValue,
    };
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::utilities::enums::Kernel;

    fn load_candles() -> Candles {
        read_candles_from_csv("src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv")
            .expect("test candles")
    }

    fn eq_or_both_nan(lhs: &[f64], rhs: &[f64]) -> bool {
        lhs.iter()
            .zip(rhs.iter())
            .all(|(a, b)| (a.is_nan() && b.is_nan()) || a == b)
    }

    #[test]
    fn output_contract() {
        let candles = load_candles();
        let input = IctPropulsionBlockInput::from_slices(
            &candles.open[..320],
            &candles.high[..320],
            &candles.low[..320],
            &candles.close[..320],
            IctPropulsionBlockParams::default(),
        );
        let out = ict_propulsion_block(&input).expect("ict_propulsion_block");
        assert_eq!(out.bullish_high.len(), 320);
        assert_eq!(out.bearish_high.len(), 320);
        assert!(out
            .bullish_kind
            .iter()
            .any(|v| v.is_finite() && (*v == 1.0 || *v == 2.0)));
        for &kind in out.bullish_kind.iter().chain(out.bearish_kind.iter()) {
            assert!(kind.is_nan() || kind == 0.0 || kind == 1.0 || kind == 2.0);
        }
    }

    #[test]
    fn invalid_params() {
        let candles = load_candles();
        let input = IctPropulsionBlockInput::from_slices(
            &candles.open[..64],
            &candles.high[..64],
            &candles.low[..64],
            &candles.close[..64],
            IctPropulsionBlockParams {
                swing_length: Some(0),
                mitigation_price: Some(IctPropulsionBlockMitigationPrice::Close),
            },
        );
        assert!(matches!(
            ict_propulsion_block(&input),
            Err(IctPropulsionBlockError::InvalidSwingLength { swing_length: 0 })
        ));
    }

    #[test]
    fn into_matches_direct() {
        let candles = load_candles();
        let input = IctPropulsionBlockInput::from_slices(
            &candles.open[..220],
            &candles.high[..220],
            &candles.low[..220],
            &candles.close[..220],
            IctPropulsionBlockParams::default(),
        );
        let direct = ict_propulsion_block(&input).expect("direct");
        let mut bullish_high = vec![f64::NAN; 220];
        let mut bullish_low = vec![f64::NAN; 220];
        let mut bullish_kind = vec![f64::NAN; 220];
        let mut bullish_active = vec![f64::NAN; 220];
        let mut bullish_mitigated = vec![f64::NAN; 220];
        let mut bullish_new = vec![f64::NAN; 220];
        let mut bearish_high = vec![f64::NAN; 220];
        let mut bearish_low = vec![f64::NAN; 220];
        let mut bearish_kind = vec![f64::NAN; 220];
        let mut bearish_active = vec![f64::NAN; 220];
        let mut bearish_mitigated = vec![f64::NAN; 220];
        let mut bearish_new = vec![f64::NAN; 220];

        ict_propulsion_block_into_slice(
            &mut bullish_high,
            &mut bullish_low,
            &mut bullish_kind,
            &mut bullish_active,
            &mut bullish_mitigated,
            &mut bullish_new,
            &mut bearish_high,
            &mut bearish_low,
            &mut bearish_kind,
            &mut bearish_active,
            &mut bearish_mitigated,
            &mut bearish_new,
            &input,
            Kernel::Scalar,
        )
        .expect("into");

        assert!(eq_or_both_nan(&bullish_high, &direct.bullish_high));
        assert!(eq_or_both_nan(&bearish_kind, &direct.bearish_kind));
        assert!(eq_or_both_nan(&bullish_new, &direct.bullish_new));
    }

    #[test]
    fn stream_matches_batch() {
        let candles = load_candles();
        let open = &candles.open[..180];
        let high = &candles.high[..180];
        let low = &candles.low[..180];
        let close = &candles.close[..180];
        let input = IctPropulsionBlockInput::from_slices(
            open,
            high,
            low,
            close,
            IctPropulsionBlockParams::default(),
        );
        let batch = ict_propulsion_block(&input).expect("batch");
        let mut stream =
            IctPropulsionBlockStream::try_new(IctPropulsionBlockParams::default()).expect("stream");
        let mut bullish_high = Vec::new();
        let mut bearish_new = Vec::new();
        for i in 0..open.len() {
            let out = stream
                .update(open[i], high[i], low[i], close[i])
                .expect("stream update");
            bullish_high.push(out.0);
            bearish_new.push(out.11);
        }
        assert!(eq_or_both_nan(&bullish_high, &batch.bullish_high));
        assert!(eq_or_both_nan(&bearish_new, &batch.bearish_new));
    }

    #[test]
    fn batch_single_param_matches_single() {
        let candles = load_candles();
        let open = &candles.open[..160];
        let high = &candles.high[..160];
        let low = &candles.low[..160];
        let close = &candles.close[..160];
        let batch = ict_propulsion_block_batch_with_kernel(
            open,
            high,
            low,
            close,
            &IctPropulsionBlockBatchRange {
                swing_length: (3, 3, 0),
                mitigation_price: (true, false),
            },
            Kernel::ScalarBatch,
        )
        .expect("batch");
        let single = ict_propulsion_block(&IctPropulsionBlockInput::from_slices(
            open,
            high,
            low,
            close,
            IctPropulsionBlockParams::default(),
        ))
        .expect("single");
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        assert!(eq_or_both_nan(
            &batch.bullish_high[..close.len()],
            &single.bullish_high[..]
        ));
        assert!(eq_or_both_nan(
            &batch.bearish_low[..close.len()],
            &single.bearish_low[..]
        ));
    }

    #[test]
    fn dispatch_matches_direct() {
        let candles = load_candles();
        let combos = [IndicatorParamSet {
            params: &[
                ParamKV {
                    key: "swing_length",
                    value: ParamValue::Int(3),
                },
                ParamKV {
                    key: "mitigation_price",
                    value: ParamValue::EnumString("close"),
                },
            ],
        }];
        let dispatched = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "ict_propulsion_block",
            output_id: Some("bullish_high"),
            data: IndicatorDataRef::Ohlc {
                open: &candles.open[..160],
                high: &candles.high[..160],
                low: &candles.low[..160],
                close: &candles.close[..160],
            },
            combos: &combos,
            kernel: Kernel::ScalarBatch,
        })
        .expect("dispatch");
        let direct = ict_propulsion_block(&IctPropulsionBlockInput::from_slices(
            &candles.open[..160],
            &candles.high[..160],
            &candles.low[..160],
            &candles.close[..160],
            IctPropulsionBlockParams::default(),
        ))
        .expect("direct");
        assert!(eq_or_both_nan(
            &dispatched.values_f64.expect("f64"),
            &direct.bullish_high
        ));
    }
}
