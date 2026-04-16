use std::collections::HashSet;
use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels/cuda");
    println!("cargo:rerun-if-changed=kernels/ptx");
    println!("cargo:rerun-if-changed=kernels/cubin");

    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        if env::var("CARGO_FEATURE_CUDA_BUILD_PTX").is_ok() {
            compile_cuda_kernels();
        } else {
            stage_prebuilt_ptx();
        }
    }

    if is_nightly() {
        println!("cargo:rustc-cfg=rustc_is_nightly");
    }
}

fn is_nightly() -> bool {
    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let output = Command::new(rustc).arg("--version").output();
    if let Ok(output) = output {
        if let Ok(stdout) = String::from_utf8(output.stdout) {
            return stdout.contains("nightly");
        }
    }
    false
}

fn sm89_cubin_name_for_ptx(ptx_name: &str) -> String {
    if let Some(stem) = ptx_name.strip_suffix(".ptx") {
        format!("{stem}_sm89.cubin")
    } else {
        format!("{ptx_name}_sm89.cubin")
    }
}

fn stage_prebuilt_ptx() {
    println!("cargo:rerun-if-env-changed=VECTOR_TA_PREBUILT_PTX_DIR");
    println!("cargo:rerun-if-env-changed=VECTOR_TA_PREBUILT_CUBIN_DIR");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    let ptx_dir = if let Ok(dir) = env::var("VECTOR_TA_PREBUILT_PTX_DIR") {
        PathBuf::from(dir)
    } else {
        manifest_dir.join("kernels/ptx/compute_89")
    };

    let cubin_dir = if let Ok(dir) = env::var("VECTOR_TA_PREBUILT_CUBIN_DIR") {
        PathBuf::from(dir)
    } else {
        manifest_dir.join("kernels/cubin/sm_89")
    };

    if !ptx_dir.is_dir() {
        panic!(
            "Prebuilt PTX directory not found: {}. \
Enable `--features cuda-build-ptx` to compile PTX and cubin artifacts with nvcc, or set VECTOR_TA_PREBUILT_PTX_DIR to a directory containing *.ptx files.",
            ptx_dir.display()
        );
    }

    let mut ptx_files: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(&ptx_dir).expect("read prebuilt PTX dir") {
        let entry = entry.expect("read prebuilt PTX dir entry");
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("ptx") {
            ptx_files.push(path);
        }
    }

    let mut cubin_files: Vec<PathBuf> = Vec::new();
    if cubin_dir.is_dir() {
        for entry in std::fs::read_dir(&cubin_dir).expect("read prebuilt cubin dir") {
            let entry = entry.expect("read prebuilt cubin dir entry");
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("cubin") {
                cubin_files.push(path);
            }
        }
    } else {
        println!(
            "cargo:warning=Prebuilt cubin directory not found: {}. Continuing with PTX-only staging.",
            cubin_dir.display()
        );
    }

    if ptx_files.is_empty() {
        panic!(
            "No prebuilt PTX files (*.ptx) found in {}. \
Enable `--features cuda-build-ptx` to compile PTX artifacts with nvcc.",
            ptx_dir.display()
        );
    }

    for src in ptx_files {
        println!("cargo:rerun-if-changed={}", src.display());
        let file_name = src
            .file_name()
            .expect("PTX file name")
            .to_string_lossy()
            .to_string();
        let dst = out_dir.join(&file_name);
        std::fs::copy(&src, &dst).unwrap_or_else(|e| {
            panic!(
                "Failed copying prebuilt PTX {} -> {}: {e}",
                src.display(),
                dst.display()
            )
        });
    }

    let mut staged_cubins = HashSet::new();
    for src in cubin_files {
        println!("cargo:rerun-if-changed={}", src.display());
        let file_name = src
            .file_name()
            .expect("cubin file name")
            .to_string_lossy()
            .to_string();
        let dst = out_dir.join(&file_name);
        std::fs::copy(&src, &dst).unwrap_or_else(|e| {
            panic!(
                "Failed copying prebuilt cubin {} -> {}: {e}",
                src.display(),
                dst.display()
            )
        });
        staged_cubins.insert(file_name);
    }

    for entry in std::fs::read_dir(&out_dir).expect("read staged PTX dir") {
        let entry = entry.expect("read staged PTX dir entry");
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("ptx") {
            continue;
        }
        let ptx_name = path
            .file_name()
            .expect("PTX file name")
            .to_string_lossy()
            .to_string();
        let cubin_name = sm89_cubin_name_for_ptx(&ptx_name);
        if staged_cubins.contains(&cubin_name) {
            continue;
        }
        std::fs::write(out_dir.join(cubin_name), []).expect("write placeholder cubin");
    }
}

fn compile_cuda_kernels() {
    println!("cargo:rerun-if-changed=kernels/cuda");

    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_ARCHS");
    println!("cargo:rerun-if-env-changed=CUDA_FILTER");
    println!("cargo:rerun-if-env-changed=CUDA_KERNEL_DIR");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=NVCC_ARGS");
    println!("cargo:rerun-if-env-changed=CUDA_DEBUG");
    println!("cargo:rerun-if-env-changed=CUDA_FAST_MATH");
    println!("cargo:rerun-if-env-changed=VECTOR_TA_PREBUILD_PTX_DIR");
    println!("cargo:rerun-if-env-changed=VECTOR_TA_PREBUILD_CUBIN_DIR");

    let cuda_path = find_cuda_path();

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
        "kernels/cuda/moving_averages/pma_kernel.cu",
        "pma_kernel.ptx",
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
        "kernels/cuda/moving_averages/decycler_kernel.cu",
        "decycler_kernel.ptx",
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
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/linearreg_intercept_kernel.cu",
        "linearreg_intercept_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/linearreg_slope_kernel.cu",
        "linearreg_slope_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/tsf_kernel.cu",
        "tsf_kernel.ptx",
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
        "kernels/cuda/moving_averages/vidya_kernel.cu",
        "vidya_kernel.ptx",
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
        "kernels/cuda/moving_averages/sgf_kernel.cu",
        "sgf_kernel.ptx",
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
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/vidya_kernel.cu",
        "vidya_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/vwmacd_kernel.cu",
        "vwmacd_kernel.ptx",
    );

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
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/alligator_kernel.cu",
        "alligator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/correlation_cycle_kernel.cu",
        "correlation_cycle_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/otto_kernel.cu",
        "otto_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/mab_kernel.cu",
        "mab_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/rsmk_kernel.cu",
        "rsmk_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/mean_ad_kernel.cu",
        "mean_ad_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/macz_kernel.cu",
        "macz_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/qstick_kernel.cu",
        "qstick_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/ott_kernel.cu",
        "ott_kernel.ptx",
    );

    compile_kernel(&cuda_path, "kernels/cuda/wad_kernel.cu", "wad_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/var_kernel.cu", "var_kernel.ptx");
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/adosc_kernel.cu",
        "adosc_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/ao_kernel.cu",
        "ao_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/bop_kernel.cu",
        "bop_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/coppock_kernel.cu",
        "coppock_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/gatorosc_kernel.cu",
        "gatorosc_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/macd_kernel.cu",
        "macd_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/mom_kernel.cu",
        "mom_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/roc_kernel.cu",
        "roc_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/rsx_kernel.cu",
        "rsx_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/srsi_kernel.cu",
        "srsi_kernel.ptx",
    );

    compile_kernel(&cuda_path, "kernels/cuda/vosc_kernel.cu", "vosc_kernel.ptx");

    compile_kernel(
        &cuda_path,
        "kernels/cuda/safezonestop_kernel.cu",
        "safezonestop_kernel.ptx",
    );

    compile_kernel(&cuda_path, "kernels/cuda/rocr_kernel.cu", "rocr_kernel.ptx");
    compile_kernel(
        &cuda_path,
        "kernels/cuda/nadaraya_watson_envelope_kernel.cu",
        "nadaraya_watson_envelope_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/mfi_kernel.cu",
        "mfi_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/willr_kernel.cu",
        "willr_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/acosc_kernel.cu",
        "acosc_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/aroonosc_kernel.cu",
        "aroonosc_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/cfo_kernel.cu",
        "cfo_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/fosc_kernel.cu",
        "fosc_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/dpo_kernel.cu",
        "dpo_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/lrsi_kernel.cu",
        "lrsi_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/ppo_kernel.cu",
        "ppo_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/tsi_kernel.cu",
        "tsi_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/stoch_kernel.cu",
        "stoch_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/aso_kernel.cu",
        "aso_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/cg_kernel.cu",
        "cg_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/cmo_kernel.cu",
        "cmo_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/reverse_rsi_kernel.cu",
        "reverse_rsi_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/rsi_kernel.cu",
        "rsi_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/dti_kernel.cu",
        "dti_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/emv_kernel.cu",
        "emv_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/kdj_kernel.cu",
        "kdj_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/stochf_kernel.cu",
        "stochf_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/squeeze_momentum_kernel.cu",
        "squeeze_momentum_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/ttm_squeeze_kernel.cu",
        "ttm_squeeze_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/cci_kernel.cu",
        "cci_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/chop_kernel.cu",
        "chop_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/dec_osc_kernel.cu",
        "dec_osc_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/fisher_kernel.cu",
        "fisher_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/ift_rsi_kernel.cu",
        "ift_rsi_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/ultosc_kernel.cu",
        "ultosc_kernel.ptx",
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
        "kernels/cuda/moving_averages/wclprice_kernel.cu",
        "wclprice_kernel.ptx",
    );
    compile_kernel(&cuda_path, "kernels/cuda/sar_kernel.cu", "sar_kernel.ptx");
    compile_kernel(
        &cuda_path,
        "kernels/cuda/alphatrend_kernel.cu",
        "alphatrend_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/medprice_kernel.cu",
        "medprice_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/pattern_recognition_kernel.cu",
        "pattern_recognition_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/bandpass_kernel.cu",
        "bandpass_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/aroon_kernel.cu",
        "aroon_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/zscore_kernel.cu",
        "zscore_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/yang_zhang_volatility_kernel.cu",
        "yang_zhang_volatility_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/garman_klass_volatility_kernel.cu",
        "garman_klass_volatility_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/parkinson_volatility_kernel.cu",
        "parkinson_volatility_kernel.ptx",
    );
    compile_kernel(&cuda_path, "kernels/cuda/voss_kernel.cu", "voss_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/cksp_kernel.cu", "cksp_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/emd_kernel.cu", "emd_kernel.ptx");
    compile_kernel(
        &cuda_path,
        "kernels/cuda/emd_trend_kernel.cu",
        "emd_trend_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/minmax_kernel.cu",
        "minmax_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/bollinger_bands_width_kernel.cu",
        "bollinger_bands_width_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/deviation_kernel.cu",
        "deviation_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/range_filter_kernel.cu",
        "range_filter_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/kaufmanstop_kernel.cu",
        "kaufmanstop_kernel.ptx",
    );
    compile_kernel(&cuda_path, "kernels/cuda/mass_kernel.cu", "mass_kernel.ptx");
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/kvo_kernel.cu",
        "kvo_kernel.ptx",
    );

    compile_kernel(&cuda_path, "kernels/cuda/natr_kernel.cu", "natr_kernel.ptx");
    compile_kernel(
        &cuda_path,
        "kernels/cuda/linearreg_angle_kernel.cu",
        "linearreg_angle_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/net_myrsi_kernel.cu",
        "net_myrsi_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/percentile_nearest_rank_kernel.cu",
        "percentile_nearest_rank_kernel.ptx",
    );

    compile_kernel(&cuda_path, "kernels/cuda/prb_kernel.cu", "prb_kernel.ptx");

    compile_kernel(&cuda_path, "kernels/cuda/vi_kernel.cu", "vi_kernel.ptx");

    compile_kernel(&cuda_path, "kernels/cuda/vpci_kernel.cu", "vpci_kernel.ptx");

    compile_kernel(
        &cuda_path,
        "kernels/cuda/mod_god_mode_kernel.cu",
        "mod_god_mode_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/bollinger_bands_kernel.cu",
        "bollinger_bands_kernel.ptx",
    );
    compile_kernel(&cuda_path, "kernels/cuda/ad_kernel.cu", "ad_kernel.ptx");
    compile_kernel(
        &cuda_path,
        "kernels/cuda/devstop_kernel.cu",
        "devstop_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/fvg_trailing_stop_kernel.cu",
        "fvg_trailing_stop_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/ttm_trend_kernel.cu",
        "ttm_trend_kernel.ptx",
    );

    compile_kernel(&cuda_path, "kernels/cuda/nvi_kernel.cu", "nvi_kernel.ptx");

    compile_kernel(&cuda_path, "kernels/cuda/pvi_kernel.cu", "pvi_kernel.ptx");

    compile_kernel(&cuda_path, "kernels/cuda/vpt_kernel.cu", "vpt_kernel.ptx");

    compile_kernel(
        &cuda_path,
        "kernels/cuda/supertrend_kernel.cu",
        "supertrend_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/medium_ad_kernel.cu",
        "medium_ad_kernel.ptx",
    );

    compile_kernel(&cuda_path, "kernels/cuda/di_kernel.cu", "di_kernel.ptx");

    compile_kernel(&cuda_path, "kernels/cuda/atr_kernel.cu", "atr_kernel.ptx");
    compile_kernel(
        &cuda_path,
        "kernels/cuda/atr_percentile_kernel.cu",
        "atr_percentile_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/chande_kernel.cu",
        "chande_kernel.ptx",
    );

    compile_kernel(&cuda_path, "kernels/cuda/cvi_kernel.cu", "cvi_kernel.ptx");
    compile_kernel(
        &cuda_path,
        "kernels/cuda/cycle_channel_oscillator_kernel.cu",
        "cycle_channel_oscillator_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/dvdiqqe_kernel.cu",
        "dvdiqqe_kernel.ptx",
    );

    compile_kernel(&cuda_path, "kernels/cuda/er_kernel.cu", "er_kernel.ptx");

    compile_kernel(&cuda_path, "kernels/cuda/pfe_kernel.cu", "pfe_kernel.ptx");

    compile_kernel(
        &cuda_path,
        "kernels/cuda/keltner_kernel.cu",
        "keltner_kernel.ptx",
    );
    compile_kernel(&cuda_path, "kernels/cuda/adx_kernel.cu", "adx_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/dm_kernel.cu", "dm_kernel.ptx");
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
    compile_kernel(&cuda_path, "kernels/cuda/dx_kernel.cu", "dx_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/eri_kernel.cu", "eri_kernel.ptx");

    compile_kernel(&cuda_path, "kernels/cuda/obv_kernel.cu", "obv_kernel.ptx");
    compile_kernel(
        &cuda_path,
        "kernels/cuda/advance_decline_line_kernel.cu",
        "advance_decline_line_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/bull_power_vs_bear_power_kernel.cu",
        "bull_power_vs_bear_power_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/bulls_v_bears_kernel.cu",
        "bulls_v_bears_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/psychological_line_kernel.cu",
        "psychological_line_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/historical_volatility_kernel.cu",
        "historical_volatility_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/historical_volatility_rank_kernel.cu",
        "historical_volatility_rank_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/historical_volatility_percentile_kernel.cu",
        "historical_volatility_percentile_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/half_causal_estimator_kernel.cu",
        "half_causal_estimator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/vertical_horizontal_filter_kernel.cu",
        "vertical_horizontal_filter_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/gopalakrishnan_range_index_kernel.cu",
        "gopalakrishnan_range_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/grover_llorens_cycle_oscillator_kernel.cu",
        "grover_llorens_cycle_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/dual_ulcer_index_kernel.cu",
        "dual_ulcer_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ewma_volatility_kernel.cu",
        "ewma_volatility_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/momentum_ratio_oscillator_kernel.cu",
        "momentum_ratio_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/on_balance_volume_oscillator_kernel.cu",
        "on_balance_volume_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/pretty_good_oscillator_kernel.cu",
        "pretty_good_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/rolling_z_score_trend_kernel.cu",
        "rolling_z_score_trend_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/rank_correlation_index_kernel.cu",
        "rank_correlation_index_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/halftrend_kernel.cu",
        "halftrend_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/pivot_kernel.cu",
        "pivot_kernel.ptx",
    );

    compile_kernel(&cuda_path, "kernels/cuda/ui_kernel.cu", "ui_kernel.ptx");

    compile_kernel(
        &cuda_path,
        "kernels/cuda/stddev_kernel.cu",
        "stddev_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/donchian_channel_width_kernel.cu",
        "donchian_channel_width_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/donchian_kernel.cu",
        "donchian_kernel.ptx",
    );

    compile_kernel(&cuda_path, "kernels/cuda/adxr_kernel.cu", "adxr_kernel.ptx");

    compile_kernel(
        &cuda_path,
        "kernels/cuda/correl_hl_kernel.cu",
        "correl_hl_kernel.ptx",
    );

    compile_kernel(&cuda_path, "kernels/cuda/efi_kernel.cu", "efi_kernel.ptx");

    compile_kernel(
        &cuda_path,
        "kernels/cuda/marketefi_kernel.cu",
        "marketefi_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/kurtosis_kernel.cu",
        "kurtosis_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/velocity_kernel.cu",
        "velocity_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/velocity_acceleration_indicator_kernel.cu",
        "velocity_acceleration_indicator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/random_walk_index_kernel.cu",
        "random_walk_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/regression_slope_oscillator_kernel.cu",
        "regression_slope_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/reversal_signals_kernel.cu",
        "reversal_signals_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/relative_strength_index_wave_indicator_kernel.cu",
        "relative_strength_index_wave_indicator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/monotonicity_index_kernel.cu",
        "monotonicity_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/market_meanness_index_kernel.cu",
        "market_meanness_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/macd_wave_signal_pro_kernel.cu",
        "macd_wave_signal_pro_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/market_structure_trailing_stop_kernel.cu",
        "market_structure_trailing_stop_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_average_cross_probability_kernel.cu",
        "moving_average_cross_probability_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/multi_length_stochastic_average_kernel.cu",
        "multi_length_stochastic_average_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/absolute_strength_index_oscillator_kernel.cu",
        "absolute_strength_index_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/accumulation_swing_index_kernel.cu",
        "accumulation_swing_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/autocorrelation_indicator_kernel.cu",
        "autocorrelation_indicator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/adaptive_bounds_rsi_kernel.cu",
        "adaptive_bounds_rsi_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/adaptive_schaff_trend_cycle_kernel.cu",
        "adaptive_schaff_trend_cycle_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/adjustable_ma_alternating_extremities_kernel.cu",
        "adjustable_ma_alternating_extremities_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/adaptive_momentum_oscillator_kernel.cu",
        "adaptive_momentum_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/cyberpunk_value_trend_analyzer_kernel.cu",
        "cyberpunk_value_trend_analyzer_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/candle_strength_oscillator_kernel.cu",
        "candle_strength_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/hema_trend_levels_kernel.cu",
        "hema_trend_levels_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/daily_factor_kernel.cu",
        "daily_factor_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/demand_index_kernel.cu",
        "demand_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/decisionpoint_breadth_swenlin_trading_oscillator_kernel.cu",
        "decisionpoint_breadth_swenlin_trading_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/directional_imbalance_index_kernel.cu",
        "directional_imbalance_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ehlers_fm_demodulator_kernel.cu",
        "ehlers_fm_demodulator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ehlers_autocorrelation_periodogram_kernel.cu",
        "ehlers_autocorrelation_periodogram_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/fractal_dimension_index_kernel.cu",
        "fractal_dimension_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/fvg_positioning_average_kernel.cu",
        "fvg_positioning_average_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/volume_energy_reservoirs_kernel.cu",
        "volume_energy_reservoirs_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/fibonacci_entry_bands_kernel.cu",
        "fibonacci_entry_bands_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/fibonacci_trailing_stop_kernel.cu",
        "fibonacci_trailing_stop_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/disparity_index_kernel.cu",
        "disparity_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/dynamic_momentum_index_kernel.cu",
        "dynamic_momentum_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/forward_backward_exponential_oscillator_kernel.cu",
        "forward_backward_exponential_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ehlers_simple_cycle_indicator_kernel.cu",
        "ehlers_simple_cycle_indicator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/evasive_supertrend_kernel.cu",
        "evasive_supertrend_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/l1_ehlers_phasor_kernel.cu",
        "l1_ehlers_phasor_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/l2_ehlers_signal_to_noise_kernel.cu",
        "l2_ehlers_signal_to_noise_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/leavitt_convolution_acceleration_kernel.cu",
        "leavitt_convolution_acceleration_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/linear_correlation_oscillator_kernel.cu",
        "linear_correlation_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ehlers_adaptive_cg_kernel.cu",
        "ehlers_adaptive_cg_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ehlers_detrending_filter_kernel.cu",
        "ehlers_detrending_filter_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ehlers_data_sampling_relative_strength_indicator_kernel.cu",
        "ehlers_data_sampling_relative_strength_indicator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ehlers_linear_extrapolation_predictor_kernel.cu",
        "ehlers_linear_extrapolation_predictor_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ehlers_smoothed_adaptive_momentum_kernel.cu",
        "ehlers_smoothed_adaptive_momentum_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ehlers_adaptive_cyber_cycle_kernel.cu",
        "ehlers_adaptive_cyber_cycle_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/adaptive_bandpass_trigger_oscillator_kernel.cu",
        "adaptive_bandpass_trigger_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/adaptive_macd_kernel.cu",
        "adaptive_macd_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/andean_oscillator_kernel.cu",
        "andean_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/exponential_trend_kernel.cu",
        "exponential_trend_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/didi_index_kernel.cu",
        "didi_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/geometric_bias_oscillator_kernel.cu",
        "geometric_bias_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/gmma_oscillator_kernel.cu",
        "gmma_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/hypertrend_kernel.cu",
        "hypertrend_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/hull_butterfly_oscillator_kernel.cu",
        "hull_butterfly_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/intraday_momentum_index_kernel.cu",
        "intraday_momentum_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/kairi_relative_index_kernel.cu",
        "kairi_relative_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/keltner_channel_width_oscillator_kernel.cu",
        "keltner_channel_width_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/impulse_macd_kernel.cu",
        "impulse_macd_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/linear_regression_intensity_kernel.cu",
        "linear_regression_intensity_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/nonlinear_regression_zero_lag_moving_average_kernel.cu",
        "nonlinear_regression_zero_lag_moving_average_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/neighboring_trailing_stop_kernel.cu",
        "neighboring_trailing_stop_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/polynomial_regression_extrapolation_kernel.cu",
        "polynomial_regression_extrapolation_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/range_oscillator_kernel.cu",
        "range_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/rolling_skewness_kurtosis_kernel.cu",
        "rolling_skewness_kurtosis_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/price_density_market_noise_kernel.cu",
        "price_density_market_noise_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/premier_rsi_oscillator_kernel.cu",
        "premier_rsi_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/spearman_correlation_kernel.cu",
        "spearman_correlation_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/squeeze_index_kernel.cu",
        "squeeze_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/statistical_trailing_stop_kernel.cu",
        "statistical_trailing_stop_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/projection_oscillator_kernel.cu",
        "projection_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/qqe_weighted_oscillator_kernel.cu",
        "qqe_weighted_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/range_breakout_signals_kernel.cu",
        "range_breakout_signals_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/range_filtered_trend_signals_kernel.cu",
        "range_filtered_trend_signals_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/stochastic_distance_kernel.cu",
        "stochastic_distance_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/stochastic_adaptive_d_kernel.cu",
        "stochastic_adaptive_d_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/stochastic_connors_rsi_kernel.cu",
        "stochastic_connors_rsi_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/stochastic_money_flow_index_kernel.cu",
        "stochastic_money_flow_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/smoothed_gaussian_trend_filter_kernel.cu",
        "smoothed_gaussian_trend_filter_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/trend_continuation_factor_kernel.cu",
        "trend_continuation_factor_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/trend_follower_kernel.cu",
        "trend_follower_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/trend_direction_force_index_kernel.cu",
        "trend_direction_force_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/trend_flow_trail_kernel.cu",
        "trend_flow_trail_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/trend_trigger_factor_kernel.cu",
        "trend_trigger_factor_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/twiggs_money_flow_kernel.cu",
        "twiggs_money_flow_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/volume_zone_oscillator_kernel.cu",
        "volume_zone_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/volume_weighted_rsi_kernel.cu",
        "volume_weighted_rsi_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/volume_weighted_relative_strength_index_kernel.cu",
        "volume_weighted_relative_strength_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/price_moving_average_ratio_percentile_kernel.cu",
        "price_moving_average_ratio_percentile_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/mesa_stochastic_multi_length_kernel.cu",
        "mesa_stochastic_multi_length_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/normalized_volume_true_range_kernel.cu",
        "normalized_volume_true_range_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/normalized_resonator_kernel.cu",
        "normalized_resonator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/volatility_quality_index_kernel.cu",
        "volatility_quality_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/goertzel_cycle_composite_wave_kernel.cu",
        "goertzel_cycle_composite_wave_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ict_propulsion_block_kernel.cu",
        "ict_propulsion_block_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ichimoku_oscillator_kernel.cu",
        "ichimoku_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/insync_index_kernel.cu",
        "insync_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/kase_peak_oscillator_with_divergences_kernel.cu",
        "kase_peak_oscillator_with_divergences_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/market_structure_confluence_kernel.cu",
        "market_structure_confluence_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/possible_rsi_kernel.cu",
        "possible_rsi_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/smooth_theil_sen_kernel.cu",
        "smooth_theil_sen_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/vdubus_divergence_wave_pattern_generator_kernel.cu",
        "vdubus_divergence_wave_pattern_generator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/volatility_ratio_adaptive_rsx_kernel.cu",
        "volatility_ratio_adaptive_rsx_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/velocity_acceleration_convergence_divergence_indicator_kernel.cu",
        "velocity_acceleration_convergence_divergence_indicator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/vwap_deviation_oscillator_kernel.cu",
        "vwap_deviation_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/vwap_zscore_with_signals_kernel.cu",
        "vwap_zscore_with_signals_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/volume_weighted_stochastic_rsi_kernel.cu",
        "volume_weighted_stochastic_rsi_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/supertrend_recovery_kernel.cu",
        "supertrend_recovery_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/supertrend_oscillator_kernel.cu",
        "supertrend_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/standardized_psar_oscillator_kernel.cu",
        "standardized_psar_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/zig_zag_channels_kernel.cu",
        "zig_zag_channels_kernel.ptx",
    );

    compile_kernel(&cuda_path, "kernels/cuda/lpc_kernel.cu", "lpc_kernel.ptx");
}

fn find_cuda_path() -> String {
    env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| {
            if cfg!(target_os = "windows") {
                use std::fs;
                let base = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA";
                if let Ok(entries) = fs::read_dir(base) {
                    let mut best: Option<(u32, u32, String)> = None;
                    for e in entries.flatten() {
                        if let Ok(name) = e.file_name().into_string() {
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

#[cfg(target_os = "windows")]
fn append_windows_nvcc_host_args(cmd: &mut std::process::Command) {
    cmd.arg("-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH");
    cmd.arg("-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING");
    cmd.arg("-allow-unsupported-compiler");
    cmd.arg("-Xcompiler").arg("/Zc:preprocessor");

    if let Ok(vs_path) = find_vs_installation() {
        cmd.arg("-ccbin").arg(vs_path);
    }
}

#[cfg(not(target_os = "windows"))]
fn append_windows_nvcc_host_args(_cmd: &mut std::process::Command) {}

fn compile_kernel(cuda_path: &str, rel_src: &str, ptx_name: &str) {
    use std::process::Command;

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

    let cubin_name = sm89_cubin_name_for_ptx(ptx_name);

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let ptx_path = out_dir.join(ptx_name);
    let cubin_path = out_dir.join(&cubin_name);

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

            let placeholder = ".version 7.0
.target compute_80
.address_size 64
// placeholder PTX (no kernels)
";
            std::fs::write(&ptx_path, placeholder).expect("write placeholder PTX");
            std::fs::write(&cubin_path, []).expect("write placeholder cubin");
            return;
        }
    }

    if cfg!(target_os = "windows") && env::var("VCINSTALLDIR").is_err() {
        eprintln!(
            "Warning: VCINSTALLDIR not set. CUDA compilation may require running inside a Visual Studio Developer Command Prompt."
        );
    }

    let nvcc = if let Ok(nvcc_env) = env::var("NVCC") {
        nvcc_env
    } else if cfg!(target_os = "windows") {
        format!("{}/bin/nvcc.exe", cuda_path)
    } else {
        format!("{}/bin/nvcc", cuda_path)
    };

    fn normalize_arch(s: &str) -> String {
        let t = s.trim();
        if t.is_empty() {
            return String::new();
        }

        if t.starts_with("sm_") {
            return t.replacen("sm_", "compute_", 1);
        }
        if t.starts_with("compute_") {
            return t.to_string();
        }
        let digits: String = t.chars().filter(|c| c.is_ascii_digit()).collect();
        if digits.len() >= 2 {
            return format!("compute_{}{}", &digits[0..1], &digits[1..2]);
        }

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

    let mut cmd = Command::new(&nvcc);

    cmd.args(&[
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--extended-lambda",
        "-ptx",
        "-O3",
    ]);

    match env::var("CUDA_FAST_MATH").as_deref() {
        Ok("0") => {}
        _ => {
            cmd.arg("--use_fast_math");
        }
    }

    if env::var("CUDA_DEBUG").ok().as_deref() == Some("1") {
        cmd.arg("-lineinfo");
    }

    cmd.args(&[
        "-arch",
        &arch,
        "-o",
        ptx_path.to_str().expect("ptx path"),
        &src_path,
    ]);

    if let Ok(extra) = env::var("NVCC_ARGS") {
        for tok in extra.split_whitespace() {
            if !tok.is_empty() {
                cmd.arg(tok);
            }
        }
    }

    if cfg!(target_os = "windows") {
        append_windows_nvcc_host_args(&mut cmd);
    }

    eprintln!("Running nvcc command: {:?}", cmd);

    let mut output = cmd.output().expect("Failed to execute nvcc");

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
                append_windows_nvcc_host_args(&mut cmd2);
            }
            eprintln!("Running nvcc command: {:?}", cmd2);
            output = cmd2.output().expect("Failed to execute nvcc (fallback)");
        }
    }

    if !output.status.success() {
        eprintln!("CUDA compilation failed for {rel_src}!");
        eprintln!("stdout: {}", String::from_utf8_lossy(&output.stdout));
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));

        if cfg!(target_os = "windows")
            && String::from_utf8_lossy(&output.stderr).contains("Cannot find compiler 'cl.exe'")
        {
            eprintln!(
                "
=== CUDA Build Error: Missing Visual Studio C++ Compiler ==="
            );
            eprintln!("nvcc requires the Microsoft Visual C++ compiler (cl.exe) to be available.");
            eprintln!("Install Visual Studio Build Tools 2022 or run cargo from a Developer Command Prompt.");
            eprintln!(
                "===========================================================
"
            );
        }

        panic!("nvcc compilation failed");
    }

    println!(
        "Successfully compiled {} to {}",
        src_path,
        ptx_path.display()
    );

    let mut cubin_cmd = Command::new(&nvcc);
    cubin_cmd.args(&[
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--extended-lambda",
        "-cubin",
        "-O3",
    ]);

    match env::var("CUDA_FAST_MATH").as_deref() {
        Ok("0") => {}
        _ => {
            cubin_cmd.arg("--use_fast_math");
        }
    }

    if env::var("CUDA_DEBUG").ok().as_deref() == Some("1") {
        cubin_cmd.arg("-lineinfo");
    }

    cubin_cmd.args(&[
        "-arch",
        "sm_89",
        "-o",
        cubin_path.to_str().expect("cubin path"),
        &src_path,
    ]);

    if let Ok(extra) = env::var("NVCC_ARGS") {
        for tok in extra.split_whitespace() {
            if !tok.is_empty() {
                cubin_cmd.arg(tok);
            }
        }
    }

    if cfg!(target_os = "windows") {
        append_windows_nvcc_host_args(&mut cubin_cmd);
    }

    eprintln!("Running nvcc command: {:?}", cubin_cmd);

    let cubin_output = cubin_cmd
        .output()
        .expect("Failed to execute nvcc for cubin");

    if !cubin_output.status.success() {
        eprintln!("CUDA cubin compilation failed for {rel_src}!");
        eprintln!("stdout: {}", String::from_utf8_lossy(&cubin_output.stdout));
        eprintln!("stderr: {}", String::from_utf8_lossy(&cubin_output.stderr));
        panic!("nvcc cubin compilation failed");
    }

    println!(
        "Successfully compiled {} to {}",
        src_path,
        cubin_path.display()
    );

    if let Ok(prebuild_dir) = env::var("VECTOR_TA_PREBUILD_PTX_DIR") {
        let prebuild_dir = PathBuf::from(prebuild_dir);
        std::fs::create_dir_all(&prebuild_dir).expect("create VECTOR_TA_PREBUILD_PTX_DIR");
        let dst = prebuild_dir.join(ptx_name);
        std::fs::copy(&ptx_path, &dst).unwrap_or_else(|e| {
            panic!(
                "Failed copying compiled PTX {} -> {}: {e}",
                ptx_path.display(),
                dst.display()
            )
        });
    }

    if let Ok(prebuild_dir) = env::var("VECTOR_TA_PREBUILD_CUBIN_DIR") {
        let prebuild_dir = PathBuf::from(prebuild_dir);
        std::fs::create_dir_all(&prebuild_dir).expect("create VECTOR_TA_PREBUILD_CUBIN_DIR");
        let dst = prebuild_dir.join(&cubin_name);
        std::fs::copy(&cubin_path, &dst).unwrap_or_else(|e| {
            panic!(
                "Failed copying compiled cubin {} -> {}: {e}",
                cubin_path.display(),
                dst.display()
            )
        });
    }
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
