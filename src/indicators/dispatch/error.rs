use crate::indicators::registry::IndicatorInputKind;
use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum IndicatorDispatchError {
    #[error("unknown indicator: {id}")]
    UnknownIndicator { id: String },
    #[error("unknown output '{output}' for indicator '{indicator}'")]
    UnknownOutput { indicator: String, output: String },
    #[error("missing required input {input:?} for indicator '{indicator}'")]
    MissingRequiredInput {
        indicator: String,
        input: IndicatorInputKind,
    },
    #[error("invalid parameter '{key}' for indicator '{indicator}': {reason}")]
    InvalidParam {
        indicator: String,
        key: String,
        reason: String,
    },
    #[error("unsupported capability '{capability}' for indicator '{indicator}'")]
    UnsupportedCapability {
        indicator: String,
        capability: &'static str,
    },
    #[error("data length mismatch: {details}")]
    DataLengthMismatch { details: String },
    #[error("kernel unavailable: {details}")]
    KernelUnavailable { details: String },
    #[error("compute failed for indicator '{indicator}': {details}")]
    ComputeFailed { indicator: String, details: String },
}
