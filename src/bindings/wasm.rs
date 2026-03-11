#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use crate::indicators::adaptive_schaff_trend_cycle::{
    adaptive_schaff_trend_cycle_alloc, adaptive_schaff_trend_cycle_batch_into,
    adaptive_schaff_trend_cycle_batch_js, adaptive_schaff_trend_cycle_free,
    adaptive_schaff_trend_cycle_into, adaptive_schaff_trend_cycle_into_host,
    adaptive_schaff_trend_cycle_js,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use crate::indicators::hypertrend::{
    hypertrend_alloc, hypertrend_batch_into, hypertrend_batch_js, hypertrend_free, hypertrend_into,
    hypertrend_into_host, hypertrend_js,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use crate::indicators::ict_propulsion_block::{
    ict_propulsion_block_alloc, ict_propulsion_block_batch_into, ict_propulsion_block_batch_js,
    ict_propulsion_block_free, ict_propulsion_block_into, ict_propulsion_block_into_host,
    ict_propulsion_block_js,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use crate::indicators::moving_averages::logarithmic_moving_average::{
    logarithmic_moving_average_alloc, logarithmic_moving_average_batch_into,
    logarithmic_moving_average_batch_js, logarithmic_moving_average_free,
    logarithmic_moving_average_into, logarithmic_moving_average_into_host,
    logarithmic_moving_average_js,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use crate::indicators::smoothed_gaussian_trend_filter::{
    smoothed_gaussian_trend_filter_alloc, smoothed_gaussian_trend_filter_batch_into,
    smoothed_gaussian_trend_filter_batch_js, smoothed_gaussian_trend_filter_free,
    smoothed_gaussian_trend_filter_into, smoothed_gaussian_trend_filter_into_host,
    smoothed_gaussian_trend_filter_js,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use crate::indicators::stochastic_adaptive_d::{
    stochastic_adaptive_d_alloc, stochastic_adaptive_d_batch_into, stochastic_adaptive_d_batch_js,
    stochastic_adaptive_d_free, stochastic_adaptive_d_into, stochastic_adaptive_d_into_host,
    stochastic_adaptive_d_js,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use crate::indicators::supertrend_oscillator::{
    supertrend_oscillator_alloc, supertrend_oscillator_batch_into, supertrend_oscillator_batch_js,
    supertrend_oscillator_free, supertrend_oscillator_into, supertrend_oscillator_into_host,
    supertrend_oscillator_js,
};
