# Critical Review: Tilson WASM Optimization Issues

## ❌ CRITICAL OPTIMIZATIONS MISSED

After a thorough review comparing tilson.rs with alma.rs, I've identified several critical optimizations that I missed which would explain why performance doesn't match expectations.

### 1. **No SIMD128 Support for WebAssembly** ⚠️

**ALMA has:**
```rust
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn alma_simd128(data: &[f64], weights: &[f64], period: usize, first_val: usize, inv_norm: f64, out: &mut [f64]) {
    use core::arch::wasm32::*;
    // Optimized SIMD128 implementation processing 2 f64 values at a time
}
```

**TILSON is missing:**
- No SIMD128 implementation at all
- Falls back to scalar implementation in WASM
- This alone could account for 2-4x performance difference

### 2. **Wrong Kernel Selection in WASM Functions** ⚠️

**ALMA pattern:**
```rust
// alma_compute_into checks for SIMD128 first
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
{
    if matches!(kernel, Kernel::Scalar | Kernel::ScalarBatch) {
        alma_simd128(data, weights, period, first, inv_n, out);
        return;
    }
}
```

**TILSON issues:**
- `tilson_js` uses `Kernel::Auto` instead of checking for WASM environment
- No automatic SIMD128 selection when available
- `tilson_batch_js` and `tilson_batch_unified_js` also use `Kernel::Auto`

### 3. **Inefficient Kernel Detection in Context API** ⚠️

**TILSON TilsonContext:**
```rust
kernel: detect_best_kernel(),  // This returns AVX on x86, not useful in WASM!
```

Should be:
```rust
kernel: Kernel::Scalar,  // Or check for SIMD128 support
```

### 4. **Missing Optimized tilson_compute_into Pattern** ⚠️

**TILSON compute function doesn't check for WASM SIMD128:**
```rust
fn tilson_compute_into(...) -> Result<(), TilsonError> {
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => tilson_scalar(...),
            // No SIMD128 check!
        }
    }
}
```

## Required Fixes

### Fix 1: Implement tilson_simd128
```rust
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn tilson_simd128(data: &[f64], period: usize, v_factor: f64, first: usize, out: &mut [f64]) {
    use core::arch::wasm32::*;
    
    // Compute coefficients
    let c1 = -v_factor.powi(3);
    let c2 = 3.0 * v_factor.powi(2) + 3.0 * v_factor.powi(3);
    let c3 = -6.0 * v_factor.powi(2) - 3.0 * v_factor - 3.0 * v_factor.powi(3);
    let c4 = 1.0 + 3.0 * v_factor + v_factor.powi(3) + 3.0 * v_factor.powi(2);
    
    // SIMD128 optimized EMA calculations...
    // Process 2 values at a time where possible
}
```

### Fix 2: Update tilson_compute_into
```rust
fn tilson_compute_into(...) -> Result<(), TilsonError> {
    unsafe {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if matches!(chosen, Kernel::Scalar | Kernel::ScalarBatch) {
                tilson_simd128(data, period, v_factor, first, out)?;
                return Ok(());
            }
        }
        
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => tilson_scalar(...),
            // ... rest of kernels
        }
    }
}
```

### Fix 3: Update WASM Functions
```rust
pub fn tilson_js(data: &[f64], period: usize, volume_factor: Option<f64>) -> Result<Vec<f64>, JsValue> {
    let params = TilsonParams {
        period: Some(period),
        volume_factor: volume_factor.or(Some(0.0)),
    };
    let input = TilsonInput::from_slice(data, params);

    // Use Scalar kernel which will auto-upgrade to SIMD128 if available
    tilson_with_kernel(&input, Kernel::Scalar)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}
```

### Fix 4: Update TilsonContext
```rust
Ok(TilsonContext {
    period,
    c1,
    c2,
    c3,
    c4,
    kernel: Kernel::Scalar,  // Not detect_best_kernel()
    // ...
})
```

## Performance Impact

With these fixes, expected performance improvements:
- **SIMD128 implementation**: 2-3x faster than scalar in WASM
- **Proper kernel selection**: Ensures SIMD128 is used when available
- **Overall**: Should achieve the target 2x difference between WASM and native Rust

## Conclusion

I failed to properly analyze alma.rs's WASM-specific optimizations. The SIMD128 support is crucial for achieving competitive performance in WebAssembly environments. Without it, tilson.rs is running purely scalar code while alma.rs is using vectorized operations, explaining the performance gap.