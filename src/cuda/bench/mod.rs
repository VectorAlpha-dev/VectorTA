#![cfg(feature = "cuda")]

pub mod helpers;
pub mod macros;

/// Trait implemented by CUDA benchmark state holders returned by wrapper-provided
/// preparation routines. The `launch` method executes one kernel invocation.
pub trait CudaBenchState {
    fn launch(&mut self);
}

/// Declarative description of a single CUDA benchmark scenario. Wrappers expose
/// these so the Criterion harness can iterate over every registered scenario
/// without hardcoding per-indicator setup in the bench crate.
pub struct CudaBenchScenario {
    /// Logical indicator identifier (e.g., "alma", "ema", ...).
    pub indicator: &'static str,
    /// Scenario key (e.g., "one_series_many_params").
    pub scenario: &'static str,
    /// Criterion benchmark group label.
    pub group: &'static str,
    /// Criterion benchmark id displayed within the group.
    pub bench_id: &'static str,
    /// Optional label used when emitting skip messages (defaults to `group`).
    pub skip_label: Option<&'static str>,
    /// Optional sample size override for the Criterion group.
    pub sample_size: Option<usize>,
    /// Approximate VRAM required (in bytes) to run the scenario, including any
    /// safety headroom.
    pub mem_required: Option<usize>,
    /// Optional inner iteration count to repeat the kernel multiple times per
    /// Criterion iteration (useful for very small workloads to reduce noise).
    pub inner_iters: Option<usize>,
    /// Preparation function returning the state needed for repeated kernel
    /// launches. The state owns its device buffers for the benchmark lifetime.
    pub prep: fn() -> Box<dyn CudaBenchState>,
}

impl CudaBenchScenario {
    /// Helper to build a scenario with the common required fields.
    pub const fn new(
        indicator: &'static str,
        scenario: &'static str,
        group: &'static str,
        bench_id: &'static str,
        prep: fn() -> Box<dyn CudaBenchState>,
    ) -> Self {
        Self {
            indicator,
            scenario,
            group,
            bench_id,
            skip_label: None,
            sample_size: None,
            mem_required: None,
            inner_iters: None,
            prep,
        }
    }

    /// Attach a skip label used when memory checks prevent running the bench.
    pub const fn with_skip_label(mut self, skip_label: &'static str) -> Self {
        self.skip_label = Some(skip_label);
        self
    }

    /// Attach a sample size override.
    pub const fn with_sample_size(mut self, sample_size: usize) -> Self {
        self.sample_size = Some(sample_size);
        self
    }

    /// Attach a VRAM requirement estimate (bytes).
    pub const fn with_mem_required(mut self, bytes: usize) -> Self {
        self.mem_required = Some(bytes);
        self
    }

    /// Attach an inner iteration count (repeats kernel in one bench iter).
    pub const fn with_inner_iters(mut self, iters: usize) -> Self {
        self.inner_iters = Some(iters);
        self
    }
}
