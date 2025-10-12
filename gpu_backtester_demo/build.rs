use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-changed=src/kernels/double_crossover.cu");

    // Try to compile the demo's CUDA backtest kernel to PTX if nvcc is present.
    let nvcc = match which::which("nvcc") {
        Ok(path) => path,
        Err(_) => {
            println!("cargo:warning=nvcc not found; building demo without compiling backtest kernel");
            return;
        }
    };

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let ptx_out = out_dir.join("double_crossover.ptx");

    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "compute_61".to_string());

    let status = Command::new(nvcc)
        .arg("-ptx")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg(format!("-arch={}", arch))
        .arg("-o")
        .arg(&ptx_out)
        .arg("src/kernels/double_crossover.cu")
        .status()
        .expect("failed to spawn nvcc");

    if !status.success() {
        panic!("nvcc failed to compile double_crossover.cu to PTX");
    }
}

