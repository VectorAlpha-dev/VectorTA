@echo off
REM Run WASM benchmarks with optimizations

cd /d "C:\Rust Projects\my_project-bindings-4"

echo Building WASM with optimizations...
echo =======================================

REM Set build flags for optimization
set RUSTFLAGS=-C target-cpu=native -C opt-level=3 -C lto=fat -C embed-bitcode=yes

REM Build WASM module
call wasm-pack build --target nodejs --features wasm

echo.
echo Running WASM benchmarks...
echo =======================================

REM Run benchmarks with GC exposed for better control
node --expose-gc benchmarks\wasm_benchmark.js %*

pause