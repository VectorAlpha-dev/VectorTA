use std::env;
use std::path::PathBuf;

fn main() {
    #[cfg(feature = "cuda")]
    compile_cuda_kernels();
}

#[cfg(feature = "cuda")]
fn compile_cuda_kernels() {
    println!("cargo:rerun-if-changed=kernels/cuda");
    println!("cargo:rerun-if-env-changed=CUDA_FILTER");

    let cuda_path = find_cuda_path();

    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=cudart");

    // Existing helpers
    compile_alma_kernel(&cuda_path);
    compile_cwma_kernel(&cuda_path);
    compile_epma_kernel(&cuda_path);
    compile_ehlers_ecema_kernel(&cuda_path);
    compile_kama_kernel(&cuda_path);
    compile_highpass_kernel(&cuda_path);
    compile_nama_kernel(&cuda_path);
    compile_wma_kernel(&cuda_path);
    compile_sinwma_kernel(&cuda_path);
    compile_tradjema_kernel(&cuda_path);
    compile_volume_adjusted_ma_kernel(&cuda_path);
    compile_supersmoother_3_pole_kernel(&cuda_path);
    compile_wto_kernel(&cuda_path);

    // Additional kernels required by wrappers under feature `cuda`
    // Moving averages (broad set)
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/buff_averages_kernel.cu", "buff_averages_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/dema_kernel.cu", "dema_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/dma_kernel.cu", "dma_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/edcf_kernel.cu", "edcf_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/ehlers_itrend_kernel.cu", "ehlers_itrend_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/ehlers_kama_kernel.cu", "ehlers_kama_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/ehlers_pma_kernel.cu", "ehlers_pma_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/ehma_kernel.cu", "ehma_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/ema_kernel.cu", "ema_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/frama_kernel.cu", "frama_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/fwma_kernel.cu", "fwma_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/gaussian_kernel.cu", "gaussian_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/highpass2_kernel.cu", "highpass2_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/hma_kernel.cu", "hma_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/hwma_kernel.cu", "hwma_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/jma_kernel.cu", "jma_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/jsa_kernel.cu", "jsa_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/linreg_kernel.cu", "linreg_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/maaq_kernel.cu", "maaq_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/mama_kernel.cu", "mama_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/mwdx_kernel.cu", "mwdx_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/nma_kernel.cu", "nma_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/pwma_kernel.cu", "pwma_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/reflex_kernel.cu", "reflex_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/sama_kernel.cu", "sama_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/sma_kernel.cu", "sma_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/smma_kernel.cu", "smma_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/sqwma_kernel.cu", "sqwma_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/srwma_kernel.cu", "srwma_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/supersmoother_kernel.cu", "supersmoother_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/swma_kernel.cu", "swma_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/tema_kernel.cu", "tema_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/tilson_kernel.cu", "tilson_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/trendflex_kernel.cu", "trendflex_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/trima_kernel.cu", "trima_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/uma_kernel.cu", "uma_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/vama_kernel.cu", "vama_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/vpwma_kernel.cu", "vpwma_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/vwap_kernel.cu", "vwap_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/vwma_kernel.cu", "vwma_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/wilders_kernel.cu", "wilders_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/moving_averages/zlema_kernel.cu", "zlema_kernel.ptx");

    // Non-MA
    compile_kernel(&cuda_path, "kernels/cuda/wad_kernel.cu", "wad_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/oscillators/willr_kernel.cu", "willr_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/wavetrend_kernel.cu", "wavetrend_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/wclprice_kernel.cu", "wclprice_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/zscore_kernel.cu", "zscore_kernel.ptx");
}

#[cfg(feature = "cuda")]
fn find_cuda_path() -> String {
    env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| {
            if cfg!(target_os = "windows") {
                let paths = [
                    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9",
                    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3",
                    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1",
                    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0",
                    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8",
                ];

                for path in &paths {
                    if std::path::Path::new(path).exists() {
                        eprintln!("Found CUDA at: {}", path);
                        return path.to_string();
                    }
                }

                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1".to_string()
            } else {
                "/usr/local/cuda".to_string()
            }
        })
}

#[cfg(feature = "cuda")]
fn compile_alma_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/alma_kernel.cu",
        "alma_kernel.ptx",
    );
}

#[cfg(feature = "cuda")]
fn compile_cwma_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/cwma_kernel.cu",
        "cwma_kernel.ptx",
    );
}

#[cfg(feature = "cuda")]
fn compile_epma_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/epma_kernel.cu",
        "epma_kernel.ptx",
    );
}

#[cfg(feature = "cuda")]
fn compile_ehlers_ecema_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/ehlers_ecema_kernel.cu",
        "ehlers_ecema_kernel.ptx",
    );
}

#[cfg(feature = "cuda")]
fn compile_kama_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/kama_kernel.cu",
        "kama_kernel.ptx",
    );
}

#[cfg(feature = "cuda")]
fn compile_highpass_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/highpass_kernel.cu",
        "highpass_kernel.ptx",
    );
}

#[cfg(feature = "cuda")]
fn compile_nama_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/nama_kernel.cu",
        "nama_kernel.ptx",
    );
}

#[cfg(feature = "cuda")]
fn compile_wma_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/wma_kernel.cu",
        "wma_kernel.ptx",
    );
}

#[cfg(feature = "cuda")]
fn compile_sinwma_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/sinwma_kernel.cu",
        "sinwma_kernel.ptx",
    );
}

#[cfg(feature = "cuda")]
fn compile_tradjema_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/tradjema_kernel.cu",
        "tradjema_kernel.ptx",
    );
}

#[cfg(feature = "cuda")]
fn compile_volume_adjusted_ma_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/volume_adjusted_ma_kernel.cu",
        "volume_adjusted_ma_kernel.ptx",
    );
}

#[cfg(feature = "cuda")]
fn compile_supersmoother_3_pole_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/supersmoother_3_pole_kernel.cu",
        "supersmoother_3_pole_kernel.ptx",
    );
}

#[cfg(feature = "cuda")]
fn compile_wto_kernel(cuda_path: &str) {
    compile_kernel(cuda_path, "kernels/cuda/wto_kernel.cu", "wto_kernel.ptx");
}

#[cfg(feature = "cuda")]
fn compile_kernel(cuda_path: &str, rel_src: &str, ptx_name: &str) {
    use std::process::Command;

    println!("cargo:rerun-if-changed={rel_src}");

    // Optional filter to speed up local iteration (e.g., CUDA_FILTER=gaussian,alma)
    if let Ok(filter) = env::var("CUDA_FILTER") {
        if !filter_allows(&filter, rel_src) {
            eprintln!("[build.rs] CUDA_FILTER matched; skipping NVCC for {rel_src}");
            return;
        }
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    if cfg!(target_os = "windows") && env::var("VCINSTALLDIR").is_err() {
        eprintln!(
            "Warning: VCINSTALLDIR not set. CUDA compilation may require running inside a Visual Studio Developer Command Prompt."
        );
    }

    let nvcc = if cfg!(target_os = "windows") {
        format!("{}/bin/nvcc.exe", cuda_path)
    } else {
        format!("{}/bin/nvcc", cuda_path)
    };

    let ptx_path = out_dir.join(ptx_name);

    let mut cmd = Command::new(&nvcc);
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_86".to_string());

    cmd.args(&[
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--extended-lambda",
        "-ptx",
        "-O3",
        // Avoid --use_fast_math to improve numerical parity with CPU for Reflex
        "-arch",
        &arch,
        "-o",
        ptx_path.to_str().expect("ptx path"),
        rel_src,
    ]);

    if cfg!(target_os = "windows") {
        cmd.arg("-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH");
        cmd.arg("-allow-unsupported-compiler");

        if let Ok(vs_path) = find_vs_installation() {
            cmd.arg("-ccbin").arg(vs_path);
        }
    }

    eprintln!("Running nvcc command: {:?}", cmd);

    let output = cmd.output().expect("Failed to execute nvcc");

    if !output.status.success() {
        eprintln!("CUDA compilation failed for {rel_src}!");
        eprintln!("stdout: {}", String::from_utf8_lossy(&output.stdout));
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));

        let allow_placeholder = env::var("CUDA_PLACEHOLDER_ON_FAIL").map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false);
        if allow_placeholder {
            eprintln!("[build.rs] Emitting placeholder PTX for {rel_src} at {}", ptx_path.display());
            std::fs::write(&ptx_path, placeholder_ptx()).expect("write placeholder PTX");
            return;
        }

        if cfg!(target_os = "windows")
            && String::from_utf8_lossy(&output.stderr).contains("Cannot find compiler 'cl.exe'")
        {
            eprintln!("\n=== CUDA Build Error: Missing Visual Studio C++ Compiler ===");
            eprintln!("nvcc requires the Microsoft Visual C++ compiler (cl.exe) to be available.");
            eprintln!("Install Visual Studio Build Tools 2022 or run cargo from a Developer Command Prompt.");
            eprintln!("===========================================================\n");
        }

        panic!("nvcc compilation failed");
    }

    println!("Successfully compiled {rel_src} to {}", ptx_path.display());
}

#[cfg(feature = "cuda")]
fn filter_allows(filter: &str, rel_src: &str) -> bool {
    let tokens: Vec<&str> = filter
        .split(|c| c == ',' || c == ' ')
        .filter(|s| !s.is_empty())
        .collect();
    if tokens.is_empty() { return true; }
    tokens.iter().any(|t| rel_src.contains(t))
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
fn find_vs_installation() -> Result<String, ()> {
    let vs_paths = [
        "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC",
        "C:/Program Files/Microsoft Visual Studio/2022/Professional/VC/Tools/MSVC",
        "C:/Program Files/Microsoft Visual Studio/2022/Enterprise/VC/Tools/MSVC",
        "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC",
        "C:/Program Files/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC",
        "C:/Program Files/Microsoft Visual Studio/2019/Professional/VC/Tools/MSVC",
        "C:/Program Files/Microsoft Visual Studio/2019/Enterprise/VC/Tools/MSVC",
    ];

    for vs_base in &vs_paths {
        if let Ok(entries) = std::fs::read_dir(vs_base) {
            if let Some(msvc_version) = entries
                .filter_map(|e| e.ok())
                .filter_map(|e| e.file_name().into_string().ok())
                .filter(|name| name.starts_with("14."))
                .max()
            {
                let cl_path = format!("{}/{}/bin/Hostx64/x64", vs_base, msvc_version);
                if std::path::Path::new(&format!("{}/cl.exe", cl_path)).exists() {
                    eprintln!("Found cl.exe at: {}", cl_path);
                    return Ok(cl_path);
                }
            }
        }
    }

    Err(())
}

#[cfg(all(feature = "cuda", not(target_os = "windows")))]
fn find_vs_installation() -> Result<String, ()> {
    Err(())
}

#[cfg(feature = "cuda")]
fn placeholder_ptx() -> &'static str {
    // Minimal valid PTX that defines no entry points; suitable for satisfying include_str!
    // and Module::from_ptx() when the functions are never looked up.
    r#".version 8.0
.target sm_50
.address_size 64
// placeholder
"#
}
