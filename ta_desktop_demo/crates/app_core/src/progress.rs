use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct DoubleMaProgressPayload {
    pub processed_pairs: usize,
    pub total_pairs: usize,
    pub phase: &'static str,
}

pub trait ProgressSink: Send + Sync {
    fn emit_double_ma_progress(&self, payload: DoubleMaProgressPayload);
}

pub fn emit_double_ma_progress(
    sink: Option<&dyn ProgressSink>,
    processed_pairs: usize,
    total_pairs: usize,
    phase: &'static str,
) {
    let Some(sink) = sink else { return };
    sink.emit_double_ma_progress(DoubleMaProgressPayload {
        processed_pairs,
        total_pairs,
        phase,
    });
}

