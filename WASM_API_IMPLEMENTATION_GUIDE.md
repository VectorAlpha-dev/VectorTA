# WASM API Implementation Guide

This guide defines the standard WASM binding patterns for Rust-Backtester indicators, based on the optimized ALMA implementation that achieved 38% performance improvement.

## Core Principles

1. **Two APIs Only**: Safe (`indicator_js`) and Fast (`indicator_into`)
2. **Zero Intermediate Allocations**: Use the `_into_slice` helper pattern
3. **Handle Aliasing**: Always check if input/output pointers are the same
4. **Consistent Naming**: `indicator_js`, `indicator_into`, `indicator_alloc`, `indicator_free`

## Implementation Pattern

### 1. Core Helper (MANDATORY)
```rust
/// Write directly to output slice - no allocations
pub fn indicator_into_slice(
    dst: &mut [f64], 
    input: &IndicatorInput, 
    kern: Kernel
) -> Result<(), IndicatorError> {
    let (data, params, warmup, ...) = indicator_prepare(input, kern)?;
    
    if dst.len() != data.len() {
        return Err(IndicatorError::InvalidLength { ... });
    }
    
    indicator_compute_into(data, params, dst);
    
    // Fill warmup with NaN
    for v in &mut dst[..warmup] {
        *v = f64::NAN;
    }
    Ok(())
}
```

### 2. Safe API
```rust
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn indicator_js(data: &[f64], param1: usize, param2: f64) -> Result<Vec<f64>, JsValue> {
    let params = IndicatorParams { param1: Some(param1), param2: Some(param2) };
    let input = IndicatorInput::from_slice(data, params);
    
    let mut output = vec![0.0; data.len()];  // Single allocation
    indicator_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    Ok(output)
}
```

### 3. Fast API with Aliasing Detection
```rust
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn indicator_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    param1: usize,
    param2: f64,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = IndicatorParams { param1: Some(param1), param2: Some(param2) };
        let input = IndicatorInput::from_slice(data, params);
        
        if in_ptr == out_ptr {  // CRITICAL: Aliasing check
            let mut temp = vec![0.0; len];
            indicator_into_slice(&mut temp, &input, Kernel::Auto)?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            indicator_into_slice(out, &input, Kernel::Auto)?;
        }
        Ok(())
    }
}
```

### 4. Memory Management
```rust
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn indicator_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn indicator_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe { let _ = Vec::from_raw_parts(ptr, len, len); }
    }
}
```

## Multiple Output Indicators

```rust
// Safe API - return flattened array
#[wasm_bindgen]
pub struct MultiResult {
    values: Vec<f64>, // [upper..., middle..., lower...]
    rows: usize,      // 3 for bollinger
    cols: usize,      // data length
}

// Fast API - separate pointers
pub fn bollinger_into(
    in_ptr: *const f64,
    upper_ptr: *mut f64,
    middle_ptr: *mut f64,
    lower_ptr: *mut f64,
    len: usize,
    period: usize,
    std_dev: f64,
) -> Result<(), JsValue> {
    // Check aliasing for ALL output pointers
}
```

## Batch Processing

```rust
#[derive(Serialize, Deserialize)]
pub struct BatchConfig {
    pub param1_range: (usize, usize, usize), // (start, end, step)
}

#[wasm_bindgen(js_name = indicator_batch)]
pub fn indicator_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: BatchConfig = serde_wasm_bindgen::from_value(config)?;
    // Return flattened results with metadata
}
```

## Critical Requirements

1. **Always validate parameters** before computation
2. **Never allocate twice** - use `_into_slice` pattern
3. **Always check aliasing** when using pointers
4. **Use Vec allocator** for WASM, not system allocator
5. **Handle warmup consistently** - fill with NaN

## Testing Checklist

- [ ] Safe API returns correct values
- [ ] Fast API handles in-place operations (aliasing)
- [ ] Memory allocation/deallocation works
- [ ] Batch API processes parameter ranges
- [ ] Add to `wasm_indicator_benchmark.js`
- [ ] Performance: Fast API is 1.4-1.8x faster

## Common Mistakes

```rust
// ❌ WRONG: Double allocation
let result = indicator(&input)?;  // Allocates
Ok(result.values)                 // Returns allocation

// ✅ RIGHT: Single allocation
let mut output = vec![0.0; len];
indicator_into_slice(&mut output, &input)?;
Ok(output)

// ❌ WRONG: No aliasing check
alma_compute_into(data, out);  // Corrupts on in-place

// ✅ RIGHT: Handle aliasing
if in_ptr == out_ptr {
    // Use temp buffer
}
```

## SIMD Implementation Rules

1. **SIMD128 for WASM**: Only implement if AVX512 kernel exists and is NOT a stub
2. **Add to macro tests**: If SIMD128 is added, include `simd128` in the indicator's macro unit tests
3. **Follow ALMA pattern**: Check `alma.rs` for SIMD128 implementation example

## Integration Requirements

1. **Add to benchmark**: Update `benchmarks/wasm_indicator_benchmark.js` with new indicator config
2. **Update tests**: Add to existing `tests/wasm/test_indicator.js` or create if missing
3. **Export functions**: Update `src/wasm.rs` to export all new WASM functions

## Performance Targets

- Safe API: Baseline (1 allocation)
- Fast API: 1.4-1.8x faster (0 allocations)
- Batch API: Optimized for parameter sweeps

**IMPORTANT**: Use `src/indicators/moving_averages/alma.rs` as the reference implementation throughout development. It demonstrates all patterns correctly including SIMD128, testing, and optimization techniques.