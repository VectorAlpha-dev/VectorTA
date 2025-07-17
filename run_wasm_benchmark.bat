@echo off
REM Run WASM benchmarks with optimizations

cd /d "C:\Rust Projects\my_project-bindings-4"

echo Building WASM with optimizations...
echo =======================================

REM Set build flags for optimization (no LTO for WASM)
REM Don't use target-cpu=native for WASM as it's not applicable
set RUSTFLAGS=-C opt-level=3 -C embed-bitcode=yes
set CARGO_PROFILE_RELEASE_LTO=off

REM Build WASM module
call wasm-pack build --target nodejs --features wasm

echo.
echo Running WASM benchmarks...
echo =======================================

REM Run benchmarks with GC exposed for better control
node --expose-gc benchmarks\wasm_benchmark.js %*

pause