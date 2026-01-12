#![cfg(feature = "cuda")]

pub mod helpers;
pub mod macros;

pub trait CudaBenchState {
    fn launch(&mut self);
}

pub struct CudaBenchScenario {
    pub indicator: &'static str,

    pub scenario: &'static str,

    pub group: &'static str,

    pub bench_id: &'static str,

    pub skip_label: Option<&'static str>,

    pub sample_size: Option<usize>,

    pub mem_required: Option<usize>,

    pub inner_iters: Option<usize>,

    pub prep: fn() -> Box<dyn CudaBenchState>,
}

impl CudaBenchScenario {
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

    pub const fn with_skip_label(mut self, skip_label: &'static str) -> Self {
        self.skip_label = Some(skip_label);
        self
    }

    pub const fn with_sample_size(mut self, sample_size: usize) -> Self {
        self.sample_size = Some(sample_size);
        self
    }

    pub const fn with_mem_required(mut self, bytes: usize) -> Self {
        self.mem_required = Some(bytes);
        self
    }

    pub const fn with_inner_iters(mut self, iters: usize) -> Self {
        self.inner_iters = Some(iters);
        self
    }
}
