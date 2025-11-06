# CUDA Integration Guide

This guide explains how to re-enable CUDA support in the Rust-Backtester project.

## Files to Restore

1. **Core CUDA Implementation**
   - Copy `experimental/cuda/moving_averages/` to `src/cuda/moving_averages/`
   - Copy `experimental/cuda/mod.rs` to `src/cuda/mod.rs`

2. **Build Script**
   - Replace `build.rs` with `experimental/cuda/build.rs.cuda`

3. **Modified Files**
   - `src/utilities/enums.rs`: Use `experimental/cuda/enums.rs.cuda` as reference
   - `src/utilities/helpers.rs`: Use `experimental/cuda/helpers.rs.cuda` as reference
   - `src/indicators/moving_averages/alma.rs`: Use `experimental/cuda/alma.rs.cuda` as reference
   - `src/indicators/moving_averages/gaussian.rs`: Use `experimental/cuda/gaussian.rs.cuda` as reference
   - `benches/indicator_benchmark.rs`: Use `experimental/cuda/indicator_benchmark.rs.cuda` as reference

## Cargo.toml Changes

Add to dependencies:
```toml
cudarc = { version = "0.12", optional = true, features = ["cuda-version-from-build-system"] }
```

Add to features:
```toml
cuda = ["cudarc", "cc"]
```

Add to build-dependencies:
```toml
cc = { version = "1.0", optional = true }
```

## Module Registration

1. In `src/lib.rs`, add:
```rust
#[cfg(feature = "cuda")]
pub mod cuda;
```

2. In `src/main.rs`, add:
```rust
#[cfg(feature = "cuda")]
mod cuda;
```

## Key Changes in Each File

### src/utilities/enums.rs
- Add `CudaBatch` variant to `Kernel` enum
- Update `is_batch()` method to include `CudaBatch`

### src/utilities/helpers.rs
- Add CUDA detection in `skip_if_unsupported!` macro
- Never auto-select CUDA in `detect_best_batch_kernel()`

### src/indicators/moving_averages/alma.rs
- Add CUDA kernel handling in `alma_batch_with_kernel()`
- Add CUDA fallback in batch processing match
- Add `alma_batch_cuda()` function
- Add CUDA test variants in test macros
- Add `check_alma_cuda_precision()` test

### benches/indicator_benchmark.rs
- Add `alma_batch_cudabatch()` function
- Add CUDA benchmark group

## Testing CUDA Integration

```bash
# Build with CUDA
cargo build --features cuda

# Run tests
cargo test --lib --features cuda alma_cuda

# Run benchmarks (Windows with VS Developer Prompt)
cargo bench --features cuda alma_batch
```

## Important Notes

1. CUDA is never auto-selected due to precision concerns
2. Only batch operations are supported (no single series)
3. Requires explicit opt-in with `.kernel(Kernel::CudaBatch)`
4. Falls back to CPU when CUDA unavailable
5. Windows requires Visual Studio C++ compiler
6. WSL2 has known issues with cudarc 0.12