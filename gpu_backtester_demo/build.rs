use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/kernels/double_crossover.cu");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=NVCC_ARGS");

    let nvcc = find_nvcc();
    let arch = resolve_cuda_arch();
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let kernel = Path::new("src/kernels/double_crossover.cu");
    let ptx_out = out_dir.join("double_crossover.ptx");

    let mut cmd = Command::new(&nvcc);
    cmd.arg("-std=c++17")
        .arg("--expt-relaxed-constexpr")
        .arg("--extended-lambda")
        .arg("-ptx")
        .arg("-O3")
        .arg(format!("-arch={}", arch))
        .arg("-o")
        .arg(&ptx_out)
        .arg(kernel);

    if cfg!(target_os = "windows") {
        cmd.arg("-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH");
        cmd.arg("-allow-unsupported-compiler");
    }

    if let Ok(extra) = env::var("NVCC_ARGS") {
        for tok in extra.split_whitespace() {
            if !tok.is_empty() { cmd.arg(tok); }
        }
    }

    eprintln!("[gpu_backtester_demo] nvcc: {:?}", cmd);
    let out = cmd.output().expect("failed to run nvcc");
    if !out.status.success() {
        eprintln!("stdout: {}", String::from_utf8_lossy(&out.stdout));
        eprintln!("stderr: {}", String::from_utf8_lossy(&out.stderr));
        panic!("nvcc failed");
    }
}

fn resolve_cuda_arch() -> String {
    fn to_compute(tag: &str) -> String {
        let t = tag.trim().to_lowercase();
        if t.starts_with("compute_") { return t; }
        if t.starts_with("sm_") { return format!("compute_{}", &t[3..]); }
        let digits = if t.contains('.') { t.replace('.', "") } else { t };
        if digits.chars().all(|c| c.is_ascii_digit()) && !digits.is_empty() {
            return format!("compute_{}", digits);
        }
        "compute_80".into()
    }
    to_compute(&env::var("CUDA_ARCH").unwrap_or_else(|_| "compute_89".into()))
}

fn find_nvcc() -> PathBuf {
    if let Ok(p) = env::var("NVCC") {
        let pb = PathBuf::from(p);
        if pb.exists() { return pb; }
    }
    if let Ok(cuda) = env::var("CUDA_PATH").or_else(|_| env::var("CUDA_HOME")) {
        let cand = if cfg!(target_os = "windows") { Path::new(&cuda).join("bin/nvcc.exe") } else { Path::new(&cuda).join("bin/nvcc") };
        if cand.exists() { return cand; }
    }
    which::which("nvcc").unwrap_or_else(|_| if cfg!(target_os = "windows") { PathBuf::from("nvcc.exe") } else { PathBuf::from("nvcc") })
}

