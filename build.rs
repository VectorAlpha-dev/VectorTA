// build.rs
use std::{env, path::PathBuf};

fn main() {
    // 1. Configure & build SLEEF via CMake + Ninja
    let dst = cmake::Config::new("third_party/sleef")
        // Force Ninja to avoid MSBuild file-tracker path-length problems
        .generator("Ninja")
        // Enable only the vector ISAs SLEEF supports in its public API
        .define("SLEEF_ENABLE_AVX2",    "ON")
        .define("SLEEF_ENABLE_AVX512F", "ON")
        // Workaround CMake ≥4 policy error on Windows
        .define("CMAKE_POLICY_VERSION_MINIMUM", "3.5")
        // Prevent manifest.rc generation & linking
        .define("CMAKE_EXE_LINKER_FLAGS",    "/MANIFEST:NO")
        .define("CMAKE_SHARED_LINKER_FLAGS", "/MANIFEST:NO")
        .build();

    // Tell Rust where to find and link the static SLEEF lib
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=sleef");

    // 2. Generate Rust bindings from the installed sleef.h
    let header_path = dst.join("include").join("sleef.h");
    let bindings = bindgen::Builder::default()
        .header(header_path.to_str().unwrap())
        // Only pull in the AVX2 / AVX-512F entry points
        .allowlist_function("Sleef_[a-z0-9_]+avx(2|512f)")
        .generate()
        .expect("Unable to generate SLEEF bindings");

    // Write the bindings to $OUT_DIR/bindings.rs
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write SLEEF bindings!");
}
