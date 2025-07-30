# Kurtosis vs ALMA Implementation Comparison Report

## Executive Summary

This report compares the `kurtosis.rs` implementation against the reference `alma.rs` implementation to identify optimization gaps and missing features. While kurtosis.rs follows many of the best practices from alma.rs, there are several areas where optimizations and features could be added.

## 1. Memory Allocation Analysis

### ✅ Correct Usage
- **Primary output allocation**: Both use `alloc_with_nan_prefix()` correctly
  ```rust
  // Both ALMA and Kurtosis:
  let mut out = alloc_with_nan_prefix(len, first + period - 1);
  ```

- **Batch matrix allocation**: Both use `make_uninit_matrix()` and `init_matrix_prefixes()` correctly
  ```rust
  // Both implementations:
  let mut values_mu = make_uninit_matrix(rows, cols);
  init_matrix_prefixes(&mut values_mu, cols, &warmup_periods);
  ```

### ⚠️ Minor Issues in Kurtosis
- **Stream buffer allocation**: Uses regular `vec!` instead of `AVec`
  ```rust
  // Kurtosis:
  buffer: vec![f64::NAN; period],  // Line 648
  
  // ALMA uses AVec for better alignment:
  buffer: vec![f64::NAN; period],  // But also uses vec! (line 670)
  ```

### ✅ No Major Memory Issues
- No large Vec allocations proportional to data length found
- Follows the zero-copy memory operation requirements

## 2. Helper Function Usage

### ✅ Correctly Used
- `alloc_with_nan_prefix` - Used in main function
- `make_uninit_matrix` - Used in batch processing
- `init_matrix_prefixes` - Used to initialize warmup periods
- `detect_best_kernel` / `detect_best_batch_kernel` - Properly used

### ⚠️ Missing Optimizations
- No use of `AVec` for cache-aligned operations in weight/coefficient storage
- ALMA uses `AVec::with_capacity(CACHELINE_ALIGN, period)` for weights

## 3. Python Binding Comparison

### ✅ Implemented Features
- Basic `kurtosis_py` function with proper GIL handling
- Stream class `KurtosisStreamPy` 
- Batch processing `kurtosis_batch_py`
- Uses `py.allow_threads()` for releasing GIL during computation

### ❌ Missing Optimizations
1. **Direct slice output in batch**: Kurtosis allocates output array correctly but ALMA has more sophisticated handling
2. **Parameter validation**: Less comprehensive than ALMA
3. **Error handling**: Less detailed error messages

### Python Binding Differences:
```python
# ALMA has 3 parameters per combo:
dict.set_item("periods", ...)
dict.set_item("offsets", ...)  
dict.set_item("sigmas", ...)

# Kurtosis only has 1:
dict.set_item("periods", ...)
```

## 4. API Consistency

### ✅ Consistent Features
- Input structures follow same pattern (`KurtosisInput` vs `AlmaInput`)
- Builder pattern implemented (`KurtosisBuilder`)
- Stream support (`KurtosisStream`)
- Batch processing structures

### ❌ Missing Features
1. **WASM-specific functions**: No `kurtosis_alloc`, `kurtosis_free`, or `kurtosis_into` functions
2. **Unified batch config**: No structured config object for WASM batch operations
3. **Extended builder methods**: ALMA builder has more configuration options

## 5. Missing Functions and Patterns

### Functions Present in ALMA but Missing in Kurtosis:

1. **Memory Management Functions** (WASM):
   - `alma_alloc()` - Custom allocator
   - `alma_free()` - Custom deallocator
   - `alma_into()` - In-place computation with raw pointers

2. **Advanced Batch Processing**:
   - `alma_batch_into()` - Batch processing into pre-allocated memory
   - More sophisticated parameter sweep with 3D grid (period × offset × sigma)

3. **SIMD Optimizations**:
   - ALMA has actual AVX2/AVX512 implementations
   - Kurtosis AVX functions just call scalar version:
     ```rust
     // Kurtosis line 307:
     pub fn kurtosis_avx2(...) {
         unsafe { kurtosis_scalar(...) }  // No actual SIMD!
     }
     ```

4. **Helper Functions**:
   - `round_up8()` - For SIMD alignment
   - Actual SIMD register operations

## 6. Specific Optimization Opportunities

### High Priority:
1. **Implement actual SIMD kernels** - Current AVX2/AVX512 functions are placeholders
2. **Add WASM memory management functions** - For zero-copy operations in JavaScript
3. **Use AVec for cache alignment** - Especially for temporary buffers

### Medium Priority:
1. **Enhance error messages** - More specific error contexts
2. **Add more builder configuration options**
3. **Implement unified batch config for WASM**

### Low Priority:
1. **Add parameter validation helpers**
2. **Enhance streaming API with more features**
3. **Add more comprehensive tests**

## 7. Code Quality Observations

### Strengths:
- Follows the mandatory zero-copy requirements
- Proper error handling structure
- Good test coverage
- Consistent API design

### Areas for Improvement:
- SIMD implementations are stubs
- Less sophisticated parameter validation
- Fewer optimization paths
- Missing some advanced features

## Recommendations

1. **Immediate Actions**:
   - Implement actual SIMD kernels for AVX2/AVX512
   - Add WASM memory management functions
   - Use AVec for coefficient storage

2. **Near-term Improvements**:
   - Enhance batch processing with more sophisticated parameter handling
   - Add missing WASM batch configuration structures
   - Improve error messages with more context

3. **Long-term Enhancements**:
   - Consider adding more complex parameter sweeps if applicable
   - Implement additional optimization paths
   - Add performance benchmarks comparing implementations

## Conclusion

The kurtosis.rs implementation follows most of the critical patterns from alma.rs, particularly around memory management and API design. The main gaps are in SIMD optimization implementation and some advanced features primarily related to WASM support. The code quality is good, but there's room for performance improvements through actual SIMD implementations and better cache alignment strategies.