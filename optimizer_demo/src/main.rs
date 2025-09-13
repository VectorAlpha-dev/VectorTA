mod backends {
    pub mod types;
    pub mod cpu;
    #[cfg(feature = "gpu")] pub mod gpu;
}

use axum::{routing::post, Json, Router};
use tower_http::services::ServeDir;
use backends::types::{Backend, OptimizeRequest, OptimizeResponse};
use serde_json::json;
use std::net::SocketAddr;

#[tokio::main]
async fn main() {
    let static_service = ServeDir::new("web/dist").fallback(ServeDir::new("static"));
    let app = Router::new()
        .route("/api/optimize", post(optimize))
        .nest_service("/wasm", ServeDir::new("pkg/pkg"))
        .fallback_service(static_service);
    let addr: SocketAddr = "127.0.0.1:8088".parse().unwrap();
    println!("Optimizer demo backend listening on http://{}", addr);
    axum::Server::bind(&addr).serve(app.into_make_service()).await.unwrap();
}

async fn optimize(Json(req): Json<OptimizeRequest>) -> Json<serde_json::Value> {
    // If GPU selected but non-ALMA types chosen, fall back to CPU backend
    let wants_gpu = matches!(req.backend, Backend::Gpu);
    let fast_ty = req.fast_type.clone().unwrap_or_else(|| "alma".to_string()).to_ascii_lowercase();
    let slow_ty = req.slow_type.clone().unwrap_or_else(|| "alma".to_string()).to_ascii_lowercase();
    let non_alma = fast_ty.as_str() != "alma" || slow_ty.as_str() != "alma";

    let result: anyhow::Result<OptimizeResponse> = if wants_gpu && non_alma {
        backends::cpu::run_cpu(req)
    } else {
        match req.backend {
            Backend::Cpu => backends::cpu::run_cpu(req),
            #[cfg(feature = "gpu")]
            Backend::Gpu => backends::gpu::run_gpu(req),
            #[cfg(not(feature = "gpu"))]
            Backend::Gpu => Err(anyhow::anyhow!("GPU backend not built")),
        }
    };
    match result {
        Ok(res) => Json(json!({ "ok": true, "data": res })),
        Err(e) => Json(json!({ "ok": false, "error": e.to_string() })),
    }
}
