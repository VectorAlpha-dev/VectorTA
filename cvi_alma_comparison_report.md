# CVI vs ALMA Indicator Comparison Report

## Executive Summary

This report compares the CVI (Chaikin's Volatility) and ALMA (Arnaud Legoux Moving Average) indicators to identify discrepancies and missing implementations. Overall, CVI has several missing features and API inconsistencies compared to ALMA.

## 1. API Surface Comparison

### Public Structs and Enums

#### CVI
```rust
pub enum CviData<'a> {
    Candles { candles: &'a Candles },
    Slices { high: &'a [f64], low: &'a [f64] },
}
```

#### ALMA
```rust
pub enum AlmaData<'a> {
    Candles { candles: &'a Candles, source: &'a str },
    Slice(&'a [f64]),
}
```

**DISCREPANCY**: CVI's `CviData::Candles` variant is missing the `source` field that ALMA has. This prevents CVI from selecting different price sources (open/high/low/close) from candles.

### AsRef Implementation

#### CVI
**MISSING**: CVI does not implement `AsRef<[f64]>` for `CviInput`.

#### ALMA
```rust
impl<'a> AsRef<[f64]> for AlmaInput<'a> {
    fn as_ref(&self) -> &[f64] { ... }
}
```

### Builder Pattern Methods

Both indicators have similar builder patterns with `new()`, parameter setters, and `apply()` methods. However:

#### CVI Builder
- Has `apply()` for candles and `apply_slice()` for slices
- `into_stream()` takes `initial_high` and `initial_low` parameters

#### ALMA Builder  
- Has `apply()` for candles and `apply_slice()` for slices
- `into_stream()` takes no parameters (creates from params)

### Error Types

Both have appropriate error enums, but with different variants:

#### CVI Errors
- `EmptyData`
- `InvalidPeriod`
- `NotEnoughValidData`
- `AllValuesNaN`

#### ALMA Errors
- `EmptyInputData`
- `AllValuesNaN`
- `InvalidPeriod`
- `NotEnoughValidData`
- `InvalidSigma`
- `InvalidOffset`

The error types are appropriate for each indicator's requirements.

## 2. Memory and Performance Patterns

### Helper Function Usage

Both indicators correctly use:
- ✅ `alloc_with_nan_prefix()` for output allocation
- ✅ `make_uninit_matrix()` for batch operations
- ✅ `init_matrix_prefixes()` for batch warmup handling
- ✅ `detect_best_kernel()` and `detect_best_batch_kernel()`

### Memory Allocation Patterns

#### CVI
- ✅ Correctly uses `alloc_with_nan_prefix()` for main output
- ❌ **VIOLATION**: Uses `vec![0.0; period]` for lag buffer (line 241)
- ❌ **VIOLATION**: Uses `vec![0.0; period]` in streaming (line 610)

#### ALMA
- ✅ Correctly uses `AVec::with_capacity()` for weights (line 293)
- ✅ Uses proper buffer allocation in streaming

**ISSUE**: CVI should use `AVec` for its lag buffers to maintain consistency with performance patterns.

### Batch Operations

Both implement batch operations correctly with:
- Proper use of `make_uninit_matrix()` and `init_matrix_prefixes()`
- Parallel processing support with rayon
- WASM compatibility checks

## 3. Python Bindings

### Main Function Signatures

#### CVI
```python
#[pyfunction(name = "cvi")]
#[pyo3(signature = (high, low, period, kernel=None))]
pub fn cvi_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>>
```

#### ALMA
```python
#[pyfunction(name = "alma")]
#[pyo3(signature = (data, period, offset, sigma, kernel=None))]
pub fn alma_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    offset: f64,
    sigma: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>>
```

Both follow similar patterns with appropriate parameter differences.

### Batch Function Implementations

#### CVI Batch
```python
#[pyfunction(name = "cvi_batch")]
#[pyo3(signature = (high, low, period_range, kernel=None))]
```
- Returns dict with "values" and "periods"

#### ALMA Batch
```python
#[pyfunction(name = "alma_batch")]
#[pyo3(signature = (data, period_range, offset_range, sigma_range, kernel=None))]
```
- Returns dict with "values", "periods", "offsets", and "sigmas"

**DISCREPANCY**: ALMA batch returns all parameter combinations in the dict, while CVI only returns periods. CVI is missing offset/sigma equivalents in its return dict.

### Stream Class Implementations

Both implement PyClass wrappers correctly:
- CVI: `CviStreamPy` with `new(period, initial_high, initial_low)`
- ALMA: `AlmaStreamPy` with `new(period, offset, sigma)`

## 4. Code Quality

### Documentation

Both have comprehensive documentation with:
- Module-level docs explaining the indicator
- Parameter descriptions
- Error conditions
- Return value descriptions

### Error Handling

Both handle errors consistently:
- Input validation
- NaN checking
- Bounds checking
- Appropriate error messages

### Inline Annotations

Both use appropriate inline annotations:
- `#[inline]` for small functions
- `#[inline(always)]` for critical hot-path functions
- `#[target_feature]` for SIMD functions

### Feature Gates

Both correctly use feature gates:
- `#[cfg(feature = "python")]` for Python bindings
- `#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]` for AVX
- `#[cfg(target_arch = "wasm32")]` for WASM compatibility

## 5. Missing Implementations in CVI

1. **AsRef trait**: CVI should implement `AsRef<[f64]>` for `CviInput`
2. **Source field**: `CviData::Candles` should include a `source` field
3. **Memory optimization**: Lag buffers should use `AVec` instead of `Vec`
4. **Batch output**: Should include all parameters in the return dict, not just periods
5. **Helper function**: Missing `expand_grid_cvi` is defined but seems redundant

## 6. Recommendations

1. **High Priority**:
   - Add `source` field to `CviData::Candles` variant
   - Implement `AsRef<[f64]>` for `CviInput`
   - Replace `vec![]` allocations with `AVec` for lag buffers

2. **Medium Priority**:
   - Enhance batch output to include all parameter combinations
   - Add source type selection support to CVI

3. **Low Priority**:
   - Consider removing redundant `expand_grid_cvi` function
   - Align error enum naming conventions

## Conclusion

While CVI implements most core functionality correctly, it lacks some API features and optimizations present in ALMA. The main issues are:
1. Missing source selection for candles
2. Missing AsRef implementation  
3. Non-optimal memory allocations for buffers
4. Incomplete batch output information

These issues should be addressed to maintain consistency across the indicator library.