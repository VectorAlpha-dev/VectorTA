use std::env;
use std::path::PathBuf;

fn main() {
    // Only compile CUDA PTX when the crate feature `cuda` is enabled.
    // Cargo exposes active features to build.rs via env vars like CARGO_FEATURE_CUDA.
    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        compile_cuda_kernels();
    } else {
        // Keep a single, quiet note for clarity in verbose logs.
        println!("cargo:warning=feature `cuda` not enabled; skipping PTX build");
    }
}

fn compile_cuda_kernels() {
    println!("cargo:rerun-if-changed=kernels/cuda");
    // Re-run on environment changes that affect CUDA build behavior
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_ARCHS");
    println!("cargo:rerun-if-env-changed=CUDA_FILTER");
    println!("cargo:rerun-if-env-changed=CUDA_KERNEL_DIR");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=NVCC_ARGS");
    println!("cargo:rerun-if-env-changed=CUDA_DEBUG");
    println!("cargo:rerun-if-env-changed=CUDA_FAST_MATH");
    // Placeholder PTX on fail is disabled for focused CUDA development.

    let cuda_path = find_cuda_path();
    // No runtime linkage to cudart is required; PTX is JIT-loaded at runtime.
    // Leave link directives out to avoid coupling to a specific toolkit layout.

    // Existing helpers
    compile_alma_kernel(&cuda_path);
    compile_cwma_kernel(&cuda_path);
    compile_epma_kernel(&cuda_path);
    compile_cora_wave_kernel(&cuda_path);
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
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/buff_averages_kernel.cu",
        "buff_averages_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/dema_kernel.cu",
        "dema_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/dma_kernel.cu",
        "dma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/edcf_kernel.cu",
        "edcf_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/ehlers_itrend_kernel.cu",
        "ehlers_itrend_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/ehlers_kama_kernel.cu",
        "ehlers_kama_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/ehlers_pma_kernel.cu",
        "ehlers_pma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/ehma_kernel.cu",
        "ehma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/ema_kernel.cu",
        "ema_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/apo_kernel.cu",
        "apo_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/frama_kernel.cu",
        "frama_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/fwma_kernel.cu",
        "fwma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/gaussian_kernel.cu",
        "gaussian_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/highpass2_kernel.cu",
        "highpass2_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/hma_kernel.cu",
        "hma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/hwma_kernel.cu",
        "hwma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/jma_kernel.cu",
        "jma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/jsa_kernel.cu",
        "jsa_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/linreg_kernel.cu",
        "linreg_kernel.ptx",
    );
    // Linear Regression Slope
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/linearreg_slope_kernel.cu",
        "linearreg_slope_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/maaq_kernel.cu",
        "maaq_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/mama_kernel.cu",
        "mama_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/mwdx_kernel.cu",
        "mwdx_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/nma_kernel.cu",
        "nma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/pwma_kernel.cu",
        "pwma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/reflex_kernel.cu",
        "reflex_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/sama_kernel.cu",
        "sama_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/sma_kernel.cu",
        "sma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/smma_kernel.cu",
        "smma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/sqwma_kernel.cu",
        "sqwma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/srwma_kernel.cu",
        "srwma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/supersmoother_kernel.cu",
        "supersmoother_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/swma_kernel.cu",
        "swma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/tema_kernel.cu",
        "tema_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/tilson_kernel.cu",
        "tilson_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/trendflex_kernel.cu",
        "trendflex_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/trima_kernel.cu",
        "trima_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/trix_kernel.cu",
        "trix_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/uma_kernel.cu",
        "uma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/vlma_kernel.cu",
        "vlma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/vama_kernel.cu",
        "vama_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/vpwma_kernel.cu",
        "vpwma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/vwap_kernel.cu",
        "vwap_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/vwma_kernel.cu",
        "vwma_kernel.ptx",
    );
    // VWMACD (Volume-Weighted MACD)
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/vwmacd_kernel.cu",
        "vwmacd_kernel.ptx",
    );
    // AVSL (Anti-Volume Stop Loss)
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/avsl_kernel.cu",
        "avsl_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/wilders_kernel.cu",
        "wilders_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/zlema_kernel.cu",
        "zlema_kernel.ptx",
    );

    // MAC-Z (ZVWAP + MACD/Stddev composite)
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/macz_kernel.cu",
        "macz_kernel.ptx",
    );

    // Non-MA
    compile_kernel(&cuda_path, "kernels/cuda/wad_kernel.cu", "wad_kernel.ptx");
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/willr_kernel.cu",
        "willr_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/wavetrend_kernel.cu",
        "wavetrend_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/cci_cycle_kernel.cu",
        "cci_cycle_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/msw_kernel.cu",
        "msw_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/kst_kernel.cu",
        "kst_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/qqe_kernel.cu",
        "qqe_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/rocp_kernel.cu",
        "rocp_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/rvi_kernel.cu",
        "rvi_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/stc_kernel.cu",
        "stc_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/wclprice_kernel.cu",
        "wclprice_kernel.ptx",
    );
    // Price transforms
    compile_kernel(
        &cuda_path,
        "kernels/cuda/medprice_kernel.cu",
        "medprice_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/zscore_kernel.cu",
        "zscore_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/adx_kernel.cu",
        "adx_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/dm_kernel.cu",
        "dm_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/chandelier_exit_kernel.cu",
        "chandelier_exit_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/damiani_volatmeter_kernel.cu",
        "damiani_volatmeter_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/dx_kernel.cu",
        "dx_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/eri_kernel.cu",
        "eri_kernel.ptx",
    );
    // OBV (On-Balance Volume)
    compile_kernel(
        &cuda_path,
        "kernels/cuda/obv_kernel.cu",
        "obv_kernel.ptx",
    );
    // HalfTrend indicator
    compile_kernel(
        &cuda_path,
        "kernels/cuda/halftrend_kernel.cu",
        "halftrend_kernel.ptx",
    );
    // Pivot indicator
    compile_kernel(
        &cuda_path,
        "kernels/cuda/pivot_kernel.cu",
        "pivot_kernel.ptx",
    );
    // Ulcer Index (UI)
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ui_kernel.cu",
        "ui_kernel.ptx",
    );
}

fn find_cuda_path() -> String {
    env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| {
            if cfg!(target_os = "windows") {
                use std::fs;
                let base = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA";
                if let Ok(entries) = fs::read_dir(base) {
                    // Pick highest version directory (e.g., v13.0 preferred over v12.3)
                    let mut best: Option<(u32, u32, String)> = None;
                    for e in entries.flatten() {
                        if let Ok(name) = e.file_name().into_string() {
                            // Expect names like "v13.0", "v12.3"
                            if let Some(stripped) = name.strip_prefix('v') {
                                let mut it = stripped.split('.');
                                let major = it.next().and_then(|s| s.parse::<u32>().ok());
                                let minor =
                                    it.next().and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);
                                if let Some(maj) = major {
                                    let cand = (maj, minor, format!("{base}/{}", name));
                                    if let Some(cur) = &best {
                                        if cand.0 > cur.0 || (cand.0 == cur.0 && cand.1 > cur.1) {
                                            best = Some(cand);
                                        }
                                    } else {
                                        best = Some(cand);
                                    }
                                }
                            }
                        }
                    }
                    if let Some((_, _, path)) = best {
                        eprintln!("Found CUDA at: {}", path);
                        return path;
                    }
                }
                // Fallback to a reasonable default if discovery fails
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0".to_string()
            } else {
                "/usr/local/cuda".to_string()
            }
        })
}

fn compile_alma_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/alma_kernel.cu",
        "alma_kernel.ptx",
    );
}

fn compile_cwma_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/cwma_kernel.cu",
        "cwma_kernel.ptx",
    );
}

fn compile_cora_wave_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/cora_wave_kernel.cu",
        "cora_wave_kernel.ptx",
    );
}

fn compile_epma_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/epma_kernel.cu",
        "epma_kernel.ptx",
    );
}

fn compile_ehlers_ecema_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/ehlers_ecema_kernel.cu",
        "ehlers_ecema_kernel.ptx",
    );
}

fn compile_kama_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/kama_kernel.cu",
        "kama_kernel.ptx",
    );
}

fn compile_highpass_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/highpass_kernel.cu",
        "highpass_kernel.ptx",
    );
}

fn compile_nama_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/nama_kernel.cu",
        "nama_kernel.ptx",
    );
}

fn compile_wma_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/wma_kernel.cu",
        "wma_kernel.ptx",
    );
}

fn compile_sinwma_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/sinwma_kernel.cu",
        "sinwma_kernel.ptx",
    );
}

fn compile_tradjema_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/tradjema_kernel.cu",
        "tradjema_kernel.ptx",
    );
}

fn compile_volume_adjusted_ma_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/volume_adjusted_ma_kernel.cu",
        "volume_adjusted_ma_kernel.ptx",
    );
}

fn compile_supersmoother_3_pole_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/supersmoother_3_pole_kernel.cu",
        "supersmoother_3_pole_kernel.ptx",
    );
}

fn compile_wto_kernel(cuda_path: &str) {
    compile_kernel(cuda_path, "kernels/cuda/wto_kernel.cu", "wto_kernel.ptx");
}

fn compile_kernel(cuda_path: &str, rel_src: &str, ptx_name: &str) {
    use std::process::Command;

    // Allow overriding the kernels root directory
    let src_path = if let Ok(root) = env::var("CUDA_KERNEL_DIR") {
        let root = root.trim_end_matches(['/', '\\']);
        let prefix = "kernels/cuda/";
        if rel_src.starts_with(prefix) {
            format!("{}/{}", root, &rel_src[prefix.len()..])
        } else {
            rel_src.to_string()
        }
    } else {
        rel_src.to_string()
    };

    println!("cargo:rerun-if-changed={}", src_path);

    // Optional filter: only compile kernels whose path contains any of these substrings
    if let Ok(filt) = env::var("CUDA_FILTER") {
        let mut any = false;
        for tok in filt.split(|c: char| c == ',' || c.is_ascii_whitespace()) {
            let t = tok.trim();
            if !t.is_empty() && rel_src.contains(t) {
                any = true;
                break;
            }
        }
        if !any {
            eprintln!("Skipping {} due to CUDA_FILTER", rel_src);
            return;
        }
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    if cfg!(target_os = "windows") && env::var("VCINSTALLDIR").is_err() {
        eprintln!(
            "Warning: VCINSTALLDIR not set. CUDA compilation may require running inside a Visual Studio Developer Command Prompt."
        );
    }

    // Resolve nvcc: NVCC env var wins; else cuda_path/bin/nvcc; else rely on PATH
    let nvcc = if let Ok(nvcc_env) = env::var("NVCC") {
        nvcc_env
    } else if cfg!(target_os = "windows") {
        format!("{}/bin/nvcc.exe", cuda_path)
    } else {
        format!("{}/bin/nvcc", cuda_path)
    };

    let ptx_path = out_dir.join(ptx_name);

    let mut cmd = Command::new(&nvcc);

    // Arch selection: first non-empty from CUDA_ARCHS, else CUDA_ARCH, else default compute_89
    fn normalize_arch(s: &str) -> String {
        let t = s.trim();
        if t.is_empty() {
            return String::new();
        }
        // Accept forms: 89, 8.9, sm_89, compute_89
        if t.starts_with("sm_") {
            // Prefer compute_XX for -ptx
            return t.replacen("sm_", "compute_", 1);
        }
        if t.starts_with("compute_") {
            return t.to_string();
        }
        let digits: String = t.chars().filter(|c| c.is_ascii_digit()).collect();
        if digits.len() >= 2 {
            return format!("compute_{}{}", &digits[0..1], &digits[1..2]);
        }
        // Fallback: as-is
        t.to_string()
    }

    let arch = {
        if let Ok(list) = env::var("CUDA_ARCHS") {
            let first = list
                .split(|c: char| c == ',' || c.is_ascii_whitespace())
                .find(|t| !t.trim().is_empty())
                .map(|s| normalize_arch(s));
            first
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "compute_89".to_string())
        } else if let Ok(a) = env::var("CUDA_ARCH") {
            let n = normalize_arch(&a);
            if n.is_empty() {
                "compute_89".to_string()
            } else {
                n
            }
        } else {
            "compute_89".to_string()
        }
    };

    cmd.args(&[
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--extended-lambda",
        "-ptx",
        "-O3",
    ]);

    // CUDA_FAST_MATH=1 adds fast math; =0 disables
    match env::var("CUDA_FAST_MATH").as_deref() {
        Ok("0") => {}
        _ => {
            cmd.arg("--use_fast_math");
        }
    }

    // Debug line info when requested
    if env::var("CUDA_DEBUG").ok().as_deref() == Some("1") {
        cmd.arg("-lineinfo");
    }

    // No per‑indicator compile‑time overrides. HMA fast paths are now enabled
    // by kernel defaults or internal heuristics; see kernels/cuda/moving_averages/hma_kernel.cu.

    cmd.args(&[
        "-arch",
        &arch,
        "-o",
        ptx_path.to_str().expect("ptx path"),
        &src_path,
    ]);

    // Extra NVCC_ARGS passthrough
    if let Ok(extra) = env::var("NVCC_ARGS") {
        for tok in extra.split_whitespace() {
            if !tok.is_empty() {
                cmd.arg(tok);
            }
        }
    }

    if cfg!(target_os = "windows") {
        cmd.arg("-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH");
        cmd.arg("-allow-unsupported-compiler");

        if let Ok(vs_path) = find_vs_installation() {
            cmd.arg("-ccbin").arg(vs_path);
        }
    }

    eprintln!("Running nvcc command: {:?}", cmd);

    let mut output = cmd.output().expect("Failed to execute nvcc");

    // If arch unsupported, retry with compute_80
    if !output.status.success() {
        let out_s = String::from_utf8_lossy(&output.stdout);
        let err_s = String::from_utf8_lossy(&output.stderr);
        let maybe_arch_fail = err_s.contains("unsupported gpu architecture")
            || err_s.contains("Value 'compute_")
            || out_s.contains("unsupported gpu architecture");

        if arch != "compute_80" && maybe_arch_fail {
            eprintln!(
                "Falling back to -arch=compute_80 for {rel_src} (nvcc doesn't support {})",
                arch
            );
            let mut cmd2 = Command::new(&nvcc);
            cmd2.args(&[
                "-std=c++17",
                "--expt-relaxed-constexpr",
                "--extended-lambda",
                "-ptx",
                "-O3",
            ]);
            if env::var("CUDA_FAST_MATH").ok().as_deref() != Some("0") {
                cmd2.arg("--use_fast_math");
            }
            if env::var("CUDA_DEBUG").ok().as_deref() == Some("1") {
                cmd2.arg("-lineinfo");
            }
            // No per‑indicator overrides in fallback path either.
            cmd2.args(&[
                "-arch",
                "compute_80",
                "-o",
                ptx_path.to_str().expect("ptx path"),
                &src_path,
            ]);
            if let Ok(extra) = env::var("NVCC_ARGS") {
                for tok in extra.split_whitespace() {
                    if !tok.is_empty() {
                        cmd2.arg(tok);
                    }
                }
            }
            if cfg!(target_os = "windows") {
                cmd2.arg("-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH");
                cmd2.arg("-allow-unsupported-compiler");
                if let Ok(vs_path) = find_vs_installation() {
                    cmd2.arg("-ccbin").arg(vs_path);
                }
            }
            eprintln!("Running nvcc command: {:?}", cmd2);
            output = cmd2.output().expect("Failed to execute nvcc (fallback)");
        }
    }

    if !output.status.success() {
        eprintln!("CUDA compilation failed for {rel_src}!");
        eprintln!("stdout: {}", String::from_utf8_lossy(&output.stdout));
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));

        // Placeholder PTX emission removed (focus on strict CUDA builds).

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

    println!(
        "Successfully compiled {} to {}",
        src_path,
        ptx_path.display()
    );
}

#[cfg(target_os = "windows")]
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

#[cfg(not(target_os = "windows"))]
fn find_vs_installation() -> Result<String, ()> {
    Err(())
}

// Note: kept intentionally strict; no placeholder PTX emission path.
