use serde::Serialize;

use my_project::indicators::moving_averages::param_schema::MaParamInfo;

#[derive(Debug, Clone, Serialize)]
pub struct MaInfo {
    pub id: String,
    pub label: String,
    pub supports_cpu: bool,
    pub supports_cuda: bool,
    pub supports_cuda_ma_sweep: bool,
    pub supports_cuda_kernel: bool,
    pub requires_candles: bool,
    pub period_based: bool,
    pub single_output: bool,
    pub params: Vec<MaParamInfo>,
    pub notes: Option<String>,
}

fn is_cuda_kernel_supported(id: &str) -> bool {
    matches!(id, "sma" | "ema" | "wma" | "alma")
}

pub fn list_mas() -> Vec<MaInfo> {
    my_project::indicators::moving_averages::registry::list_moving_averages()
        .iter()
        .map(|m| MaInfo {
            id: m.id.to_string(),
            label: m.label.to_string(),
            supports_cpu: m.supports_cpu_single,
            supports_cuda: m.supports_cuda_single,
            supports_cuda_ma_sweep: m.supports_cuda_sweep,
            supports_cuda_kernel: is_cuda_kernel_supported(m.id),
            requires_candles: m.requires_candles,
            period_based: m.period_based,
            single_output: m.single_output,
            params: my_project::indicators::moving_averages::param_schema::ma_param_schema(m.id)
                .to_vec(),
            notes: m.notes.map(|s| s.to_string()),
        })
        .collect()
}
