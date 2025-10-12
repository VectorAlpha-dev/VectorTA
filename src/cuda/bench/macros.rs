#![cfg(feature = "cuda")]

/// Macro to define two standard CUDA benchmark scenarios (one-series × many-params,
/// and many-series × one-param) for a moving average wrapper that follows the
/// conventional API shape.
///
/// Parameters:
/// - mod name to contain the benches (to avoid polluting wrapper namespace)
/// - wrapper type path (e.g., crate::cuda::moving_averages::CudaSma)
/// - range and params type paths (from indicators module)
/// - batch function ident on the wrapper (e.g., sma_batch_dev)
/// - many-series function ident on the wrapper (e.g., sma_multi_series_one_param_time_major_dev)
/// - an expression that builds the batch range (uses the constants defined within)
/// - an expression that builds the single-parameter params for many-series
/// - indicator short name (e.g., "sma") and group base label (e.g., "sma")
#[macro_export]
macro_rules! define_ma_period_benches {
    (
        $modname:ident,
        $WrapperTy:path,
        $RangeTy:path,
        $ParamsTy:path,
        $batch_fn:ident,
        $many_fn:ident,
        $make_range:expr,
        $make_params:expr,
        $indicator:expr,
        $group_base:expr
    ) => {
        pub mod $modname {
            use super::*;
            use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
            use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

            const ONE_SERIES_LEN: usize = 1_000_000;
            const PARAM_SWEEP: usize = 250;
            const MANY_SERIES_COLS: usize = 250;
            const MANY_SERIES_LEN: usize = 1_000_000;

            fn bytes_one_series_many_params() -> usize {
                let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
                let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
                in_bytes + out_bytes + 64 * 1024 * 1024
            }
            fn bytes_many_series_one_param() -> usize {
                let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
                let in_bytes = elems * std::mem::size_of::<f32>();
                let out_bytes = elems * std::mem::size_of::<f32>();
                in_bytes + out_bytes + 64 * 1024 * 1024
            }

            struct BatchState {
                cuda: $WrapperTy,
                price: Vec<f32>,
                sweep: $RangeTy,
            }
            impl CudaBenchState for BatchState {
                fn launch(&mut self) {
                    let _ = self
                        .cuda
                        .$batch_fn(&self.price, &self.sweep)
                        .expect("batch launch");
                }
            }
            fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
                let cuda = <$WrapperTy>::new(0).expect("cuda");
                let price = gen_series(ONE_SERIES_LEN);
                let sweep: $RangeTy = $make_range;
                Box::new(BatchState { cuda, price, sweep })
            }

            struct ManyState {
                cuda: $WrapperTy,
                data_tm: Vec<f32>,
                cols: usize,
                rows: usize,
                params: $ParamsTy,
            }
            impl CudaBenchState for ManyState {
                fn launch(&mut self) {
                    let _ = self
                        .cuda
                        .$many_fn(&self.data_tm, self.cols, self.rows, &self.params)
                        .expect("many-series launch");
                }
            }
            fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
                let cuda = <$WrapperTy>::new(0).expect("cuda");
                let cols = MANY_SERIES_COLS;
                let rows = MANY_SERIES_LEN;
                let data_tm = gen_time_major_prices(cols, rows);
                let params: $ParamsTy = $make_params;
                Box::new(ManyState {
                    cuda,
                    data_tm,
                    cols,
                    rows,
                    params,
                })
            }

            pub fn bench_profiles() -> Vec<CudaBenchScenario> {
                vec![
                    CudaBenchScenario::new(
                        $indicator,
                        "one_series_many_params",
                        concat!($group_base, "_cuda_batch_dev"),
                        "1m_x_250",
                        prep_one_series_many_params,
                    )
                    .with_sample_size(10)
                    .with_mem_required(bytes_one_series_many_params()),
                    CudaBenchScenario::new(
                        $indicator,
                        "many_series_one_param",
                        concat!($group_base, "_cuda_many_series_one_param"),
                        "250x1m",
                        prep_many_series_one_param,
                    )
                    .with_sample_size(5)
                    .with_mem_required(bytes_many_series_one_param()),
                ]
            }
        }
    };
}

/// Macro to define a batch-only benchmark (for wrappers that do not provide a
/// many-series path).
#[macro_export]
macro_rules! define_ma_period_benches_batch_only {
    (
        $modname:ident,
        $WrapperTy:path,
        $RangeTy:path,
        $batch_fn:ident,
        $make_range:expr,
        $indicator:expr,
        $group_base:expr
    ) => {
        pub mod $modname {
            use super::*;
            use crate::cuda::bench::helpers::gen_series;
            use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

            const ONE_SERIES_LEN: usize = 1_000_000;
            const PARAM_SWEEP: usize = 250;

            fn bytes_one_series_many_params() -> usize {
                let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
                let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
                in_bytes + out_bytes + 64 * 1024 * 1024
            }

            struct BatchState {
                cuda: $WrapperTy,
                price: Vec<f32>,
                sweep: $RangeTy,
            }
            impl CudaBenchState for BatchState {
                fn launch(&mut self) {
                    let _ = self
                        .cuda
                        .$batch_fn(&self.price, &self.sweep)
                        .expect("batch launch");
                }
            }
            fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
                let cuda = <$WrapperTy>::new(0).expect("cuda");
                let price = gen_series(ONE_SERIES_LEN);
                let sweep: $RangeTy = $make_range;
                Box::new(BatchState { cuda, price, sweep })
            }

            pub fn bench_profiles() -> Vec<CudaBenchScenario> {
                vec![CudaBenchScenario::new(
                    $indicator,
                    "one_series_many_params",
                    concat!($group_base, "_cuda_batch_dev"),
                    "1m_x_250",
                    prep_one_series_many_params,
                )
                .with_sample_size(10)
                .with_mem_required(bytes_one_series_many_params())]
            }
        }
    };
}
