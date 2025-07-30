# GatorOsc vs ALMA Implementation Comparison Report

## Executive Summary

This report compares the implementation patterns between `gatorosc.rs` and `alma.rs`, focusing on API structure, memory operations, and optimization patterns. The analysis excludes scalar/AVX2/AVX512 kernels and WASM-specific bindings.

## 1. API Structure

### Input/Output Structs
Both indicators follow nearly identical patterns:

**ALMA:**
- `AlmaData<'a>` enum with `Candles` and `Slice` variants
- `AlmaInput<'a>` with data and params
- `AlmaOutput` with single `values: Vec<f64>`
- `AlmaParams` with `period`, `offset`, `sigma`

**GatorOsc:**
- `GatorOscData<'a>` enum with `Candles` and `Slice` variants  
- `GatorOscInput<'a>` with data and params
- `GatorOscOutput` with four outputs: `upper`, `lower`, `upper_change`, `lower_change`
- `GatorOscParams` with 6 parameters (jaws/teeth/lips length and shift)

### Builder Pattern
Both implement the builder pattern identically:
- Default constructors
- Method chaining for parameters
- `apply()` and `apply_slice()` methods
- `into_stream()` for streaming interface
- Kernel selection support

### Error Types
Both use thiserror with similar error variants:
- `AllValuesNaN`
- `InvalidSettings` / `InvalidPeriod`
- `NotEnoughValidData`
- ALMA has additional: `EmptyInputData`, `InvalidSigma`, `InvalidOffset`

## 2. Memory Operations Analysis

### ✅ CORRECT: Output Vector Allocation
Both correctly use `alloc_with_nan_prefix()` for output vectors:

**ALMA (line 314):**
```rust
let mut out = alloc_with_nan_prefix(data.len(), first + period - 1);
```

**GatorOsc (lines 254-257):**
```rust
let mut upper = alloc_with_nan_prefix(data.len(), first + jaws_length.max(teeth_length) - 1);
let mut lower = alloc_with_nan_prefix(data.len(), first + teeth_length.max(lips_length) - 1);
let mut upper_change = alloc_with_nan_prefix(data.len(), first + jaws_length.max(teeth_length));
let mut lower_change = alloc_with_nan_prefix(data.len(), first + teeth_length.max(lips_length));
```

### ✅ CORRECT: Batch Operations
Both correctly use `make_uninit_matrix()` and `init_matrix_prefixes()`:

**ALMA (lines 896-905):**
```rust
let mut buf_mu = make_uninit_matrix(rows, cols);
let warm: Vec<usize> = combos.iter().map(|c| ...).collect();
init_matrix_prefixes(&mut buf_mu, cols, &warm);
```

**GatorOsc (lines 1025-1049):**
```rust
let mut upper_buf = make_uninit_matrix(rows, cols);
let mut lower_buf = make_uninit_matrix(rows, cols);
// ... warmup period calculations ...
init_matrix_prefixes(&mut upper_buf, cols, &warmup_periods);
```

### ⚠️ MINOR ISSUE: Small Fixed-Size Allocations

**ALMA (lines 292-294):**
```rust
let mut weights: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, period);
weights.resize(period, 0.0);
```
This is acceptable as weights are small (size = period << data.len()).

**GatorOsc (lines 310-315):**
```rust
let mut jaws_ring: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, max_shift + 1);
let mut teeth_ring: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, max_shift + 1);
let mut lips_ring: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, max_shift + 1);
jaws_ring.resize(max_shift + 1, f64::NAN);
teeth_ring.resize(max_shift + 1, f64::NAN);
lips_ring.resize(max_shift + 1, f64::NAN);
```
These ring buffers are also acceptable as they're small fixed-size allocations.

### ⚠️ POTENTIAL ISSUE: Stream Implementation

**GatorOsc Stream (lines 754-757):**
```rust
buf: {
    let mut buf: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, jaws_shift.max(teeth_shift).max(lips_shift) + 1);
    buf.resize(jaws_shift.max(teeth_shift).max(lips_shift) + 1, f64::NAN);
    buf.to_vec()  // ⚠️ COPY OPERATION
},
```
The `.to_vec()` call creates an unnecessary copy. This should be avoided.

**GatorOsc EmaStream (line 832):**
```rust
state: Vec::new(),  // ✅ OK - grows incrementally
```
This is fine as it starts empty and grows as needed.

**ALMA Stream (line 670):**
```rust
buffer: vec![f64::NAN; period],  // ✅ OK - small fixed size
```
This is acceptable as it's a small fixed-size buffer.

## 3. Optimization Patterns

### Zero-Copy Patterns
Both indicators properly implement zero-copy patterns:
- Direct computation into pre-allocated output buffers
- `_into_slice` variants that write directly to user-provided buffers
- Batch operations that minimize allocations

### Helper Function Usage
Both correctly use the helper functions:
- `alloc_with_nan_prefix()` for output allocation
- `make_uninit_matrix()` for batch matrix allocation
- `init_matrix_prefixes()` for NaN prefix initialization
- `detect_best_kernel()` for SIMD selection

### Batch Implementation
Both have sophisticated batch implementations:
- Parameter grid expansion
- Parallel processing support (via rayon)
- Zero-allocation batch functions (`_batch_inner_into`)

## 4. Core Function Comparison

### Main Entry Points
- `alma()` / `gatorosc()` - Auto kernel selection
- `alma_with_kernel()` / `gatorosc_with_kernel()` - Manual kernel selection
- Both follow prepare → allocate → compute pattern

### Python Bindings (Non-WASM)
Both implement Python bindings similarly:
- Pre-allocate numpy arrays for batch operations
- Use `allow_threads()` for GIL release
- Zero-copy where possible

## 5. Key Differences

1. **Output Complexity**: GatorOsc has 4 outputs vs ALMA's 1
2. **Parameter Count**: GatorOsc has 6 parameters vs ALMA's 3
3. **Internal State**: GatorOsc uses ring buffers for shift operations
4. **Error Granularity**: ALMA has more specific error types
5. **Stream Buffer**: GatorOsc unnecessarily copies AVec to Vec

## 6. Recommendations

1. **Fix GatorOsc Stream Buffer**: Remove the `.to_vec()` call in line 757:
   ```rust
   buf: buf,  // Instead of buf.to_vec()
   ```
   And change the field type from `Vec<f64>` to `AVec<f64>`.

2. **Consider Error Consolidation**: GatorOsc could benefit from more specific error types like ALMA.

3. **Both implementations are otherwise excellent** and follow the zero-copy patterns correctly.

## Conclusion

Both indicators demonstrate high-quality implementation with proper memory management and optimization patterns. The only significant issue is the unnecessary copy operation in GatorOsc's stream buffer initialization. Otherwise, both indicators correctly implement zero-copy patterns, use helper functions appropriately, and follow consistent API design.