use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Compile the CUDA backtest kernel when the `gpu` feature is enabled
    let gpu_enabled = env::var_os("CARGO_FEATURE_GPU").is_some();
    if !gpu_enabled { return; }

    println!("cargo:rerun-if-changed=src/kernels/double_crossover.cu");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");

    let nvcc = match which::which("nvcc") {
        Ok(p) => p,
        Err(_) => {
            println!("cargo:warning=nvcc not found; building without GPU kernel");
            return;
        }
    };
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_out = out_dir.join("double_crossover.ptx");
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "compute_61".to_string());
    let status = Command::new(nvcc)
        .args(["-ptx", "-O3", "--use_fast_math"])
        .arg(format!("-arch={}", arch))
        .args(["-o", ptx_out.to_str().unwrap()])
        .arg("src/kernels/double_crossover.cu")
        .status()
        .expect("failed to run nvcc");
    if !status.success() {
        panic!("nvcc failed to compile double_crossover.cu");
    }
}

