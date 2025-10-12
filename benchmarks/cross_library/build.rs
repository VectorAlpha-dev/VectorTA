use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../third_party/tulipindicators");

    // Build Tulip Indicators
    build_tulip();

    // Generate bindings for Tulip
    generate_tulip_bindings();

    // TA-Lib requires manual installation on Windows
    // Users should download from https://ta-lib.org and set LIB/INCLUDE paths
    #[cfg(feature = "talib")]
    {
        build_talib();
        generate_talib_bindings();
    }
}

fn build_tulip() {
    let tulip_path = "../../third_party/tulipindicators";

    // Build Tulip indicators C library
    let mut build = cc::Build::new();

    build
        .include(format!("{}/indicators", tulip_path))
        .file(format!("{}/indicators/abs.c", tulip_path))
        .file(format!("{}/indicators/acos.c", tulip_path))
        .file(format!("{}/indicators/ad.c", tulip_path))
        .file(format!("{}/indicators/add.c", tulip_path))
        .file(format!("{}/indicators/adosc.c", tulip_path))
        .file(format!("{}/indicators/adx.c", tulip_path))
        .file(format!("{}/indicators/adxr.c", tulip_path))
        .file(format!("{}/indicators/ao.c", tulip_path))
        .file(format!("{}/indicators/apo.c", tulip_path))
        .file(format!("{}/indicators/aroon.c", tulip_path))
        .file(format!("{}/indicators/aroonosc.c", tulip_path))
        .file(format!("{}/indicators/asin.c", tulip_path))
        .file(format!("{}/indicators/atan.c", tulip_path))
        .file(format!("{}/indicators/atr.c", tulip_path))
        .file(format!("{}/indicators/avgprice.c", tulip_path))
        .file(format!("{}/indicators/bbands.c", tulip_path))
        .file(format!("{}/indicators/bop.c", tulip_path))
        .file(format!("{}/indicators/cci.c", tulip_path))
        .file(format!("{}/indicators/ceil.c", tulip_path))
        .file(format!("{}/indicators/cmo.c", tulip_path))
        .file(format!("{}/indicators/cos.c", tulip_path))
        .file(format!("{}/indicators/cosh.c", tulip_path))
        .file(format!("{}/indicators/crossany.c", tulip_path))
        .file(format!("{}/indicators/crossover.c", tulip_path))
        .file(format!("{}/indicators/cvi.c", tulip_path))
        .file(format!("{}/indicators/decay.c", tulip_path))
        .file(format!("{}/indicators/dema.c", tulip_path))
        .file(format!("{}/indicators/di.c", tulip_path))
        .file(format!("{}/indicators/div.c", tulip_path))
        .file(format!("{}/indicators/dm.c", tulip_path))
        .file(format!("{}/indicators/dpo.c", tulip_path))
        .file(format!("{}/indicators/dx.c", tulip_path))
        .file(format!("{}/indicators/edecay.c", tulip_path))
        .file(format!("{}/indicators/ema.c", tulip_path))
        .file(format!("{}/indicators/emv.c", tulip_path))
        .file(format!("{}/indicators/exp.c", tulip_path))
        .file(format!("{}/indicators/fisher.c", tulip_path))
        .file(format!("{}/indicators/floor.c", tulip_path))
        .file(format!("{}/indicators/fosc.c", tulip_path))
        .file(format!("{}/indicators/hma.c", tulip_path))
        .file(format!("{}/indicators/kama.c", tulip_path))
        .file(format!("{}/indicators/kvo.c", tulip_path))
        .file(format!("{}/indicators/lag.c", tulip_path))
        .file(format!("{}/indicators/linreg.c", tulip_path))
        .file(format!("{}/indicators/linregintercept.c", tulip_path))
        .file(format!("{}/indicators/linregslope.c", tulip_path))
        .file(format!("{}/indicators/ln.c", tulip_path))
        .file(format!("{}/indicators/log10.c", tulip_path))
        .file(format!("{}/indicators/macd.c", tulip_path))
        .file(format!("{}/indicators/marketfi.c", tulip_path))
        .file(format!("{}/indicators/mass.c", tulip_path))
        .file(format!("{}/indicators/max.c", tulip_path))
        .file(format!("{}/indicators/md.c", tulip_path))
        .file(format!("{}/indicators/medprice.c", tulip_path))
        .file(format!("{}/indicators/mfi.c", tulip_path))
        .file(format!("{}/indicators/min.c", tulip_path))
        .file(format!("{}/indicators/mom.c", tulip_path))
        .file(format!("{}/indicators/msw.c", tulip_path))
        .file(format!("{}/indicators/mul.c", tulip_path))
        .file(format!("{}/indicators/natr.c", tulip_path))
        .file(format!("{}/indicators/nvi.c", tulip_path))
        .file(format!("{}/indicators/obv.c", tulip_path))
        .file(format!("{}/indicators/ppo.c", tulip_path))
        .file(format!("{}/indicators/psar.c", tulip_path))
        .file(format!("{}/indicators/pvi.c", tulip_path))
        .file(format!("{}/indicators/qstick.c", tulip_path))
        .file(format!("{}/indicators/roc.c", tulip_path))
        .file(format!("{}/indicators/rocr.c", tulip_path))
        .file(format!("{}/indicators/round.c", tulip_path))
        .file(format!("{}/indicators/rsi.c", tulip_path))
        .file(format!("{}/indicators/sin.c", tulip_path))
        .file(format!("{}/indicators/sinh.c", tulip_path))
        .file(format!("{}/indicators/sma.c", tulip_path))
        .file(format!("{}/indicators/sqrt.c", tulip_path))
        .file(format!("{}/indicators/stddev.c", tulip_path))
        .file(format!("{}/indicators/stderr.c", tulip_path))
        .file(format!("{}/indicators/stoch.c", tulip_path))
        .file(format!("{}/indicators/stochrsi.c", tulip_path))
        .file(format!("{}/indicators/sub.c", tulip_path))
        .file(format!("{}/indicators/sum.c", tulip_path))
        .file(format!("{}/indicators/tan.c", tulip_path))
        .file(format!("{}/indicators/tanh.c", tulip_path))
        .file(format!("{}/indicators/tema.c", tulip_path))
        .file(format!("{}/indicators/todeg.c", tulip_path))
        .file(format!("{}/indicators/torad.c", tulip_path))
        .file(format!("{}/indicators/tr.c", tulip_path))
        .file(format!("{}/indicators/trima.c", tulip_path))
        .file(format!("{}/indicators/trix.c", tulip_path))
        .file(format!("{}/indicators/trunc.c", tulip_path))
        .file(format!("{}/indicators/tsf.c", tulip_path))
        .file(format!("{}/indicators/typprice.c", tulip_path))
        .file(format!("{}/indicators/ultosc.c", tulip_path))
        .file(format!("{}/indicators/var.c", tulip_path))
        .file(format!("{}/indicators/vhf.c", tulip_path))
        .file(format!("{}/indicators/vidya.c", tulip_path))
        .file(format!("{}/indicators/volatility.c", tulip_path))
        .file(format!("{}/indicators/vosc.c", tulip_path))
        .file(format!("{}/indicators/vwma.c", tulip_path))
        .file(format!("{}/indicators/wad.c", tulip_path))
        .file(format!("{}/indicators/wcprice.c", tulip_path))
        .file(format!("{}/indicators/wilders.c", tulip_path))
        .file(format!("{}/indicators/willr.c", tulip_path))
        .file(format!("{}/indicators/wma.c", tulip_path))
        .file(format!("{}/indicators/zlema.c", tulip_path))
        // Add utility files needed by some indicators
        .file(format!("{}/utils/buffer.c", tulip_path))
        // Add the main indicator list - use indicators.c instead of tiamalgamation.c for MSVC compatibility
        .file(format!("{}/indicators.c", tulip_path));

    // Set optimization flags
    build
        .opt_level(3)
        .flag_if_supported("-march=native")
        .flag_if_supported("/arch:AVX2");

    // Compile the library
    build.compile("tulipindicators");

    // Link the library
    println!("cargo:rustc-link-lib=static=tulipindicators");
}

fn generate_tulip_bindings() {
    let tulip_path = "../../third_party/tulipindicators";

    let bindings = bindgen::Builder::default()
        .header(format!("{}/indicators.h", tulip_path))
        // Include necessary types
        .allowlist_type("ti_indicator_info")
        .allowlist_function("ti_.*")
        .allowlist_var("ti_.*")
        .allowlist_type("TI_.*")
        // Generate bindings
        .generate()
        .expect("Unable to generate Tulip bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("tulip_bindings.rs"))
        .expect("Couldn't write Tulip bindings!");
}

#[cfg(feature = "talib")]
fn build_talib() {
    // TA-Lib is more complex to build automatically, especially on Windows
    // Users should install it manually and set environment variables

    // Check for TA-Lib installation
    if let Ok(talib_path) = env::var("TALIB_PATH") {
        println!("cargo:rustc-link-search={}/lib", talib_path);
        // TA-LIB 0.6.x uses 'ta-lib' instead of 'ta_lib'
        println!("cargo:rustc-link-lib=ta-lib");
    } else {
        println!("cargo:warning=TA-Lib not found. Set TALIB_PATH environment variable.");
        println!("cargo:warning=Download from https://ta-lib.org");
    }
}

#[cfg(feature = "talib")]
fn generate_talib_bindings() {
    if let Ok(talib_path) = env::var("TALIB_PATH") {
        let bindings = bindgen::Builder::default()
            .header(format!("{}/include/ta_libc.h", talib_path))
            .allowlist_function("TA_.*")
            .allowlist_var("TA_.*")
            .generate()
            .expect("Unable to generate TA-Lib bindings");

        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        bindings
            .write_to_file(out_path.join("talib_bindings.rs"))
            .expect("Couldn't write TA-Lib bindings!");
    } else {
        // Create dummy bindings if TA-Lib is not available
        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        std::fs::write(
            out_path.join("talib_bindings.rs"),
            "// TA-Lib bindings not generated - library not found\n"
        ).expect("Couldn't write dummy TA-Lib bindings!");
    }
}