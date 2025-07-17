# WASM API Implementation Guide for Rust-Backtester Indicators

This guide provides a standardized approach for implementing WASM bindings across all indicators in the Rust-Backtester project. Follow these patterns to ensure consistency and prevent API drift.

## Table of Contents
1. [API Patterns Overview](#api-patterns-overview)
2. [Single Output Indicators](#single-output-indicators)
3. [Multiple Output Indicators](#multiple-output-indicators)
4. [Implementation Checklist](#implementation-checklist)
5. [Common Pitfalls](#common-pitfalls)

## API Patterns Overview

All indicators should implement these four API patterns when adding WASM support:

1. **Standard API** - Simple, safe, memory-managed
2. **Context API** - Stateful computation with cached parameters
3. **Zero-Copy API** - Direct pointer manipulation
4. **Pre-Allocated Buffer API** - Maximum performance with manual memory management

## Single Output Indicators

For indicators that return a single array (e.g., SMA, EMA, RSI):

### 1. Standard API Implementation

```rust
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn indicator_name(
    data: &[f64],
    param1: usize,
    param2: f64,  // Add parameters as needed
) -> Result<Vec<f64>, JsValue> {
    match indicator_name_impl(&IndicatorInput {
        data: IndicatorData::Slice(data),
        params: IndicatorParams {
            param1,
            param2,
        },
    }) {
        Ok(output) => Ok(output.values),
        Err(e) => Err(JsValue::from_str(&e.to_string())),
    }
}
```

### 2. Context API Implementation

```rust
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct IndicatorContext {
    // Store pre-computed values
    param1: usize,
    param2: f64,
    warmup_period: usize,
    // Add any cached computations
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl IndicatorContext {
    #[wasm_bindgen(constructor)]
    pub fn new(param1: usize, param2: f64) -> Result<IndicatorContext, JsValue> {
        // Validate parameters
        if param1 < 2 {
            return Err(JsValue::from_str("param1 must be at least 2"));
        }
        
        // Calculate warmup period
        let warmup_period = param1 - 1;
        
        Ok(IndicatorContext {
            param1,
            param2,
            warmup_period,
        })
    }
    
    pub fn get_warmup_period(&self) -> usize {
        self.warmup_period
    }
    
    pub fn update(&self, data: &[f64]) -> Result<Vec<f64>, JsValue> {
        self.process_internal(data, false)
    }
    
    pub fn update_into(
        &self,
        in_ptr: *const f64,
        out_ptr: *mut f64,
        len: usize,
    ) -> Result<(), JsValue> {
        if in_ptr.is_null() || out_ptr.is_null() {
            return Err(JsValue::from_str("Null pointer provided"));
        }
        
        unsafe {
            let data = std::slice::from_raw_parts(in_ptr, len);
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            
            // Check for aliasing (same buffer for input/output)
            if in_ptr == out_ptr as *const f64 {
                // Use temporary buffer to avoid corruption
                let mut temp = vec![0.0; len];
                self.compute_into(data, &mut temp)?;
                out.copy_from_slice(&temp);
            } else {
                self.compute_into(data, out)?;
            }
        }
        
        Ok(())
    }
    
    fn compute_into(&self, data: &[f64], out: &mut [f64]) -> Result<(), JsValue> {
        // Your indicator computation logic here
        // Write NaN values for warmup period
        // Compute values for valid indices
        Ok(())
    }
}
```

### 3. Zero-Copy API Implementation

```rust
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn indicator_name_into(
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
        let out = std::slice::from_raw_parts_mut(out_ptr, len);
        
        // Compute directly into output buffer
        match indicator_name_impl(&IndicatorInput {
            data: IndicatorData::Slice(data),
            params: IndicatorParams { param1, param2 },
        }) {
            Ok(result) => {
                out.copy_from_slice(&result.values);
                Ok(())
            }
            Err(e) => Err(JsValue::from_str(&e.to_string())),
        }
    }
}
```

### 4. Pre-Allocated Buffer API Implementation

```rust
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn indicator_alloc(len: usize) -> *mut f64 {
    let layout = Layout::array::<f64>(len).expect("Invalid layout");
    unsafe {
        let ptr = alloc(layout) as *mut f64;
        if ptr.is_null() {
            panic!("Failed to allocate memory");
        }
        ptr
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn indicator_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let layout = Layout::array::<f64>(len).expect("Invalid layout");
            dealloc(ptr as *mut u8, layout);
        }
    }
}

// Use the same indicator_name_into function from Zero-Copy API
```

## Multiple Output Indicators

For indicators that return multiple arrays (e.g., Bollinger Bands, MACD):

### 1. Standard API with Object Return

```rust
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct IndicatorResult {
    upper: Vec<f64>,
    middle: Vec<f64>,
    lower: Vec<f64>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl IndicatorResult {
    #[wasm_bindgen(getter)]
    pub fn upper(&self) -> Vec<f64> {
        self.upper.clone()
    }
    
    #[wasm_bindgen(getter)]
    pub fn middle(&self) -> Vec<f64> {
        self.middle.clone()
    }
    
    #[wasm_bindgen(getter)]
    pub fn lower(&self) -> Vec<f64> {
        self.lower.clone()
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn bollinger_bands(
    data: &[f64],
    period: usize,
    std_dev: f64,
) -> Result<IndicatorResult, JsValue> {
    // Implementation
    Ok(IndicatorResult {
        upper,
        middle,
        lower,
    })
}
```

### 2. Zero-Copy API for Multiple Outputs

```rust
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn bollinger_bands_into(
    in_ptr: *const f64,
    upper_ptr: *mut f64,
    middle_ptr: *mut f64,
    lower_ptr: *mut f64,
    len: usize,
    period: usize,
    std_dev: f64,
) -> Result<(), JsValue> {
    // Validate all pointers
    if in_ptr.is_null() || upper_ptr.is_null() || middle_ptr.is_null() || lower_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let upper = std::slice::from_raw_parts_mut(upper_ptr, len);
        let middle = std::slice::from_raw_parts_mut(middle_ptr, len);
        let lower = std::slice::from_raw_parts_mut(lower_ptr, len);
        
        // Compute all outputs
        // Handle aliasing if any output buffer is the same as input
    }
    
    Ok(())
}
```

### 3. Allocation Functions for Multiple Outputs

```rust
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct BollingerBuffers {
    upper_ptr: *mut f64,
    middle_ptr: *mut f64,
    lower_ptr: *mut f64,
    len: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn bollinger_alloc_all(len: usize) -> BollingerBuffers {
    BollingerBuffers {
        upper_ptr: indicator_alloc(len),
        middle_ptr: indicator_alloc(len),
        lower_ptr: indicator_alloc(len),
        len,
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn bollinger_free_all(buffers: &BollingerBuffers) {
    indicator_free(buffers.upper_ptr, buffers.len);
    indicator_free(buffers.middle_ptr, buffers.len);
    indicator_free(buffers.lower_ptr, buffers.len);
}
```

## Implementation Checklist

When adding WASM support to an indicator:

- [ ] Add `#[cfg(feature = "wasm")]` guards for all WASM-specific code
- [ ] Implement Standard API with proper error handling
- [ ] Add parameter validation before any allocations
- [ ] Document warmup period in function comments
- [ ] Implement Context API if indicator has expensive pre-computations
- [ ] Add Zero-Copy API with aliasing checks
- [ ] Implement allocation/deallocation functions
- [ ] Add tests in `tests/wasm/test_indicator_name.js`
- [ ] Update `src/wasm.rs` to export new functions
- [ ] Run `wasm-pack build --target nodejs --features wasm` to verify

## Common Pitfalls

### 1. Memory Aliasing
Always check if input and output pointers are the same:
```rust
if in_ptr == out_ptr as *const f64 {
    // Use temporary buffer
}
```

### 2. Null Pointer Checks
Always validate pointers before use:
```rust
if ptr.is_null() {
    return Err(JsValue::from_str("Null pointer provided"));
}
```

### 3. Parameter Validation
Validate parameters before allocation to avoid wasting memory:
```rust
if period > data.len() {
    return Err(JsValue::from_str("Period exceeds data length"));
}
```

### 4. Warmup Period Handling
Always document and handle warmup periods consistently:
```rust
// Fill warmup period with NaN
for i in 0..warmup_period {
    out[i] = f64::NAN;
}
```

### 5. Error Messages
Provide clear, actionable error messages:
```rust
Err(JsValue::from_str(&format!(
    "Period must be between 2 and {}, got {}",
    max_period, period
)))
```

## Testing Template

Create `tests/wasm/test_indicator_name.js`:

```javascript
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const rust_wasm = require('../../pkg/rust_backtester.js');
import { describe, it } from 'node:test';
import assert from 'node:assert';

describe('indicator_name WASM', () => {
    it('should calculate correctly', () => {
        const data = [/* test data */];
        const result = rust_wasm.indicator_name(data, param1, param2);
        
        // Verify warmup period
        for (let i = 0; i < expectedWarmup; i++) {
            assert(isNaN(result[i]));
        }
        
        // Verify calculations
        assert.strictEqual(result[expectedWarmup], expectedValue);
    });
    
    it('should handle zero-copy API', () => {
        // Test with pre-allocated buffers
    });
    
    it('should handle context API', () => {
        const ctx = new rust_wasm.IndicatorContext(param1, param2);
        const result = ctx.update(data);
        // Verify results
    });
});
```

## Notes

- This guide assumes indicators follow the standard pattern with `IndicatorInput` and `IndicatorOutput` structures
- SIMD optimizations are handled automatically by the Rust implementation when available
- The zero-copy and pre-allocated APIs are primarily for performance-critical applications
- Most users should use the Standard or Context APIs for safety and simplicity