# Build Configuration Fixes

This document summarizes the fixes applied to resolve build issues with Python and WASM targets.

## Issue Summary

1. **LTO Error**: "lto can only be run for executables, cdylibs and static library outputs"
   - WASM builds were failing because LTO was enabled globally
   - WASM cdylib targets don't support LTO

2. **CPU Feature Warnings**: Various unrecognized CPU features when building for WASM
   - Features like `+avx512vbmi2`, `+rdpid` etc. are x86-specific
   - These were being applied to WASM builds incorrectly

3. **Build-std Error**: "error: -Zbuild-std requires --target"
   - The `[unstable]` section in `.cargo/config.toml` was applying globally
   - This broke regular cargo commands

## Fixes Applied

### 1. Removed Global Flags

**`.cargo/config.toml`**:
- Removed `-Zdylib-lto` from global `[build]` rustflags
- Commented out `[unstable]` section (only needed for WASM parallel builds)

### 2. Target-Specific Build Scripts

Created separate build scripts for different targets:

**`build_python.bat`**:
```batch
set CARGO_PROFILE_RELEASE_LTO=thin
maturin develop --features python,nightly-avx --release
```

**`build_wasm.bat`**:
```batch
set CARGO_PROFILE_RELEASE_LTO=off
wasm-pack build --target nodejs --features wasm
```

### 3. Updated Existing Scripts

**`test_bindings.bat`**:
- Added `set CARGO_PROFILE_RELEASE_LTO=off` for WASM builds
- Added `set RUSTFLAGS=-Zdylib-lto` for Python builds

**`run_wasm_benchmark.bat`**:
- Removed `-C target-cpu=native` (not applicable to WASM)
- Added `set CARGO_PROFILE_RELEASE_LTO=off`

**`build_wasm_parallel.bat`**:
- Added `set CARGO_PROFILE_RELEASE_LTO=off`

### 4. Cargo.toml Changes

- Removed `lto = "thin"` from `[profile.release]`
- Set `lto = false` in `[profile.wasm]` (though this profile isn't used by wasm-pack)

## How to Build

### Python Bindings:
```batch
build_python.bat
# or
set RUSTFLAGS=-Zdylib-lto
maturin develop --features python,nightly-avx --release
```

### WASM Bindings:
```batch
build_wasm.bat
# or
set CARGO_PROFILE_RELEASE_LTO=off
wasm-pack build --target nodejs --features wasm
```

### WASM with Parallel Support:
```batch
build_wasm_parallel.bat
# Note: Requires uncommenting [unstable] section in .cargo/config.toml
```

## Testing

Run tests with:
```batch
test_bindings.bat alma
```

All ALMA tests should pass:
- ✅ 29 tests passing
- ⏭️ 2 tests skipped (parallel features not built)
- ❌ 0 tests failing

## Key Learnings

1. **LTO and WASM**: Link-Time Optimization cannot be used with WASM cdylib targets
2. **Environment Variables**: `CARGO_PROFILE_RELEASE_LTO=off` disables LTO for specific builds
3. **Target Features**: CPU-specific features should not be applied to WASM builds
4. **Build-std**: The `[unstable]` build-std feature should only be used when specifically needed

## Future Considerations

1. Consider using Cargo profiles more effectively (e.g., custom profiles for WASM)
2. Move build configuration to a build.rs script for more control
3. Use feature flags to conditionally enable LTO based on target