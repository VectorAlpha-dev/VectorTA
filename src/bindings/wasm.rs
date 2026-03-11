#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use crate::indicators::adaptive_bounds_rsi::{
    adaptive_bounds_rsi_batch_js, adaptive_bounds_rsi_js,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use crate::indicators::adjustable_ma_alternating_extremities::{
    adjustable_ma_alternating_extremities_batch_js, adjustable_ma_alternating_extremities_js,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use crate::indicators::bulls_v_bears::{
    bulls_v_bears_alloc, bulls_v_bears_batch_into, bulls_v_bears_batch_js, bulls_v_bears_free,
    bulls_v_bears_into, bulls_v_bears_js,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use crate::indicators::daily_factor::{
    daily_factor_alloc, daily_factor_batch_into, daily_factor_batch_js, daily_factor_free,
    daily_factor_into, daily_factor_js,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use crate::indicators::ehlers_adaptive_cyber_cycle::{
    ehlers_adaptive_cyber_cycle_alloc, ehlers_adaptive_cyber_cycle_batch_into,
    ehlers_adaptive_cyber_cycle_batch_js, ehlers_adaptive_cyber_cycle_free,
    ehlers_adaptive_cyber_cycle_into, ehlers_adaptive_cyber_cycle_js,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use crate::indicators::forward_backward_exponential_oscillator::{
    forward_backward_exponential_oscillator_batch_js, forward_backward_exponential_oscillator_js,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use crate::indicators::market_structure_confluence::{
    market_structure_confluence_batch_js, market_structure_confluence_js,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use crate::indicators::moving_average_cross_probability::{
    moving_average_cross_probability_alloc, moving_average_cross_probability_batch_into,
    moving_average_cross_probability_batch_js, moving_average_cross_probability_free,
    moving_average_cross_probability_into, moving_average_cross_probability_js,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use crate::indicators::qqe_weighted_oscillator::{
    qqe_weighted_oscillator_batch_js, qqe_weighted_oscillator_js,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use crate::indicators::range_filtered_trend_signals::{
    range_filtered_trend_signals_batch_js, range_filtered_trend_signals_js,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use crate::indicators::range_oscillator::{range_oscillator_batch_js, range_oscillator_js};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use crate::indicators::regression_slope_oscillator::{
    regression_slope_oscillator_alloc, regression_slope_oscillator_batch_into,
    regression_slope_oscillator_batch_js, regression_slope_oscillator_free,
    regression_slope_oscillator_into, regression_slope_oscillator_js,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use crate::indicators::smooth_theil_sen::{
    smooth_theil_sen_alloc, smooth_theil_sen_batch_into, smooth_theil_sen_batch_js,
    smooth_theil_sen_free, smooth_theil_sen_into, smooth_theil_sen_js,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use crate::indicators::volume_weighted_relative_strength_index::{
    volume_weighted_relative_strength_index_batch_js, volume_weighted_relative_strength_index_js,
};pub use crate::indicators::exponential_trend::{
    exponential_trend_batch_unified_js as exponential_trend_batch, exponential_trend_js,
    ExponentialTrendStreamWasm,
};
pub use crate::indicators::range_breakout_signals::{
    range_breakout_signals_batch_unified_js as range_breakout_signals_batch,
    range_breakout_signals_js, RangeBreakoutSignalsStreamWasm,
};
pub use crate::indicators::trend_flow_trail::{
    trend_flow_trail_batch_unified_js as trend_flow_trail_batch, trend_flow_trail_js,
    TrendFlowTrailStreamWasm,
};
