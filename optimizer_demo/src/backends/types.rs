use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
pub struct OptimizeRequest {
    pub backend: Backend,
    pub series: Option<Vec<f64>>,         
    pub synthetic_len: Option<usize>,
    pub fast_period: (usize, usize, usize),
    pub slow_period: (usize, usize, usize),
    pub fast_type: Option<String>,        
    pub slow_type: Option<String>,
    
    pub open: Option<Vec<f64>>,
    pub high: Option<Vec<f64>>,
    pub low: Option<Vec<f64>>,
    pub close: Option<Vec<f64>>, 
    pub volume: Option<Vec<f64>>,
    pub timestamps: Option<Vec<i64>>,     
    
    pub fast_params: Option<serde_json::Value>,
    pub slow_params: Option<serde_json::Value>,
    pub anchor: Option<String>,           
    pub offset: f64,
    pub sigma: f64,
    pub commission: f32,
    pub metrics: usize, 
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Backend { Cpu, Gpu }

#[derive(Debug, Clone, Serialize)]
pub struct OptimizeResponseMeta {
    pub fast_periods: Vec<usize>,
    pub slow_periods: Vec<usize>,
    pub metrics: Vec<&'static str>,
    pub rows: usize,
    pub cols: usize,
    pub axes: Vec<AxisMeta>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OptimizeResponse {
    pub meta: OptimizeResponseMeta,
    // Flattened row-major [rows * cols * M] fp32 for compactness
    pub values: Vec<f32>,
    // Number of layers for extra dimensions (>= 1)
    pub layers: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct AxisMeta {
    pub name: String,   // e.g., "fast_period", "slow_period", "fast.phase", "slow.power"
    pub values: Vec<f64>,
}

pub fn expand_range((s, e, step): (usize, usize, usize)) -> Vec<usize> {
    if step == 0 || s == e { return vec![s]; }
    (s..=e).step_by(step).collect()
}
