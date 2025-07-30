# Decycler vs ALMA Function Comparison

## Overview
This document compares every function in Decycler with ALMA (excluding scalar/avx2/avx512 kernels) to check for API signatures, optimization patterns, and feature completeness.

## Comparison Table

| Function/Struct/Trait | Present in Both? | API Match? | Optimizations Match? | Notes |
|----------------------|------------------|------------|---------------------|-------|
| **Core Types** | | | | |
| `AsRef<[f64]> for Input` | ✓ Yes | ✓ Yes | N/A | Both implement AsRef trait identically |
| `Data<'a>` enum | ✓ Yes | ✓ Yes | N/A | Both have Candles/Slice variants |
| `Output` struct | ✓ Yes | ✓ Yes | N/A | Both have `values: Vec<f64>` |
| `Params` struct | ✓ Yes | ✗ No | N/A | Different params: ALMA has period/offset/sigma, Decycler has hp_period/k |
| `Default for Params` | ✓ Yes | ✓ Yes | N/A | Both implement Default trait |
| `Input<'a>` struct | ✓ Yes | ✓ Yes | N/A | Same structure with data/params fields |
| **Input Methods** | | | | |
| `Input::from_candles()` | ✓ Yes | ✓ Yes | N/A | Identical signatures |
| `Input::from_slice()` | ✓ Yes | ✓ Yes | N/A | Identical signatures |
| `Input::with_default_candles()` | ✓ Yes | ✓ Yes | N/A | Identical signatures |
| `Input::get_period()` | ✗ No | N/A | N/A | ALMA only - Decycler has `get_hp_period()` instead |
| `Input::get_hp_period()` | ✗ No | N/A | N/A | Decycler only |
| `Input::get_offset()` | ✗ No | N/A | N/A | ALMA only |
| `Input::get_sigma()` | ✗ No | N/A | N/A | ALMA only |
| `Input::get_k()` | ✗ No | N/A | N/A | Decycler only |
| **Builder Pattern** | | | | |
| `Builder` struct | ✓ Yes | ✓ Yes | N/A | Both have builder pattern |
| `Default for Builder` | ✓ Yes | ✓ Yes | N/A | Both implement Default |
| `Builder::new()` | ✓ Yes | ✓ Yes | N/A | Identical |
| `Builder::period()` | ✗ No | N/A | N/A | ALMA only |
| `Builder::hp_period()` | ✗ No | N/A | N/A | Decycler only |
| `Builder::offset()` | ✗ No | N/A | N/A | ALMA only |
| `Builder::sigma()` | ✗ No | N/A | N/A | ALMA only |
| `Builder::k()` | ✗ No | N/A | N/A | Decycler only |
| `Builder::kernel()` | ✓ Yes | ✓ Yes | N/A | Identical |
| `Builder::apply()` | ✓ Yes | ✓ Yes | N/A | Identical - applies to Candles |
| `Builder::apply_slice()` | ✓ Yes | ✓ Yes | N/A | Identical - applies to slice |
| `Builder::into_stream()` | ✓ Yes | ✓ Yes | N/A | Both support streaming |
| **Error Types** | | | | |
| `Error` enum | ✓ Yes | ✗ No | N/A | Different error variants for different params |
| **Core Functions** | | | | |
| `indicator()` | ✓ Yes | ✓ Yes | N/A | Both have main entry function (alma/decycler) |
| `indicator_with_kernel()` | ✓ Yes | ✓ Yes | ✓ Yes | Both use detect_best_kernel() |
| `indicator_into_slice()` | ✓ Yes | ✓ Yes | ✗ No | ALMA better - uses dst parameter, Decycler writes to out |
| `indicator_scalar()` | ✓ Yes | ✗ No | ✗ No | Different implementations - Decycler allocates hp vector |
| **Memory Optimization** | | | | |
| Uses `alloc_with_nan_prefix()` | ✓ Yes | ✓ Yes | ✓ Yes | Both use for output vectors |
| Uses `make_uninit_matrix()` | ✓ Yes | ✓ Yes | ✓ Yes | Both use for batch operations |
| Uses `init_matrix_prefixes()` | ✓ Yes | ✓ Yes | ✓ Yes | Both use for batch NaN initialization |
| Uses `AVec` for alignment | ✓ Yes | ✓ Yes | ✗ No | ALMA uses more extensively for weights |
| **Streaming Support** | | | | |
| `Stream` struct | ✓ Yes | ✗ No | ✗ No | Different fields based on algorithm |
| `Stream::try_new()` | ✓ Yes | ✓ Yes | N/A | Both have constructor with error handling |
| `Stream::update()` | ✓ Yes | ✓ Yes | N/A | Both return Option<f64> |
| **Batch Processing** | | | | |
| `BatchRange` struct | ✓ Yes | ✗ No | N/A | Different params (period/offset/sigma vs hp_period/k) |
| `Default for BatchRange` | ✓ Yes | ✓ Yes | N/A | Both implement Default |
| `BatchBuilder` struct | ✓ Yes | ✓ Yes | N/A | Similar structure |
| `BatchBuilder::new()` | ✓ Yes | ✓ Yes | N/A | Identical |
| `BatchBuilder::kernel()` | ✓ Yes | ✓ Yes | N/A | Identical |
| `BatchBuilder::*_range()` | ✓ Yes | ✗ No | N/A | Different parameter methods |
| `BatchBuilder::*_static()` | ✓ Yes | ✗ No | N/A | Different parameter methods |
| `BatchBuilder::apply_slice()` | ✓ Yes | ✓ Yes | N/A | Identical signatures |
| `BatchBuilder::apply_candles()` | ✓ Yes | ✓ Yes | N/A | Identical signatures |
| `BatchBuilder::with_default_slice()` | ✓ Yes | ✓ Yes | N/A | Static method, identical pattern |
| `BatchBuilder::with_default_candles()` | ✓ Yes | ✓ Yes | N/A | Static method, identical pattern |
| `BatchOutput` struct | ✓ Yes | ✓ Yes | N/A | Same fields: values/combos/rows/cols |
| `BatchOutput::row_for_params()` | ✓ Yes | ✓ Yes | N/A | Same functionality |
| `BatchOutput::values_for()` | ✓ Yes | ✓ Yes | N/A | Same functionality |
| `batch_with_kernel()` | ✓ Yes | ✓ Yes | ✓ Yes | Both use detect_best_batch_kernel() |
| `batch_slice()` | ✓ Yes | ✓ Yes | N/A | Serial batch processing |
| `batch_par_slice()` | ✓ Yes | ✓ Yes | N/A | Parallel batch processing |
| `expand_grid()` | ✓ Yes | ✓ Yes | N/A | Both have grid expansion logic |
| **Python Bindings** | | | | |
| `#[pyfunction] indicator_py()` | ✓ Yes | ✓ Yes | ✓ Yes | Both use allow_threads, return PyArray |
| `#[pyclass] StreamPy` | ✓ Yes | ✓ Yes | N/A | Both wrap native Stream |
| `StreamPy::new()` | ✓ Yes | ✓ Yes | N/A | Constructor with params |
| `StreamPy::update()` | ✓ Yes | ✓ Yes | N/A | Returns Option<f64> |
| `#[pyfunction] batch_py()` | ✓ Yes | ✓ Yes | ✗ No | Similar but Decycler has batch_inner_into optimization |
| **WASM Bindings** | | | | |
| `indicator_js()` | ✗ No | N/A | N/A | ALMA only |
| `BatchConfig` struct | ✗ No | N/A | N/A | ALMA only |
| `BatchJsOutput` struct | ✗ No | N/A | N/A | ALMA only |
| `batch_unified_js()` | ✗ No | N/A | N/A | ALMA only |
| `alloc()` | ✗ No | N/A | N/A | ALMA only - manual memory management |
| `free()` | ✗ No | N/A | N/A | ALMA only - manual memory management |
| `into()` | ✗ No | N/A | N/A | ALMA only - in-place computation |
| `Context` struct | ✗ No | N/A | N/A | ALMA only - stateful WASM context |
| `batch_into()` | ✗ No | N/A | N/A | ALMA only - batch in-place |

## Key Differences Summary

### 1. **API Completeness**
- **ALMA has more complete WASM bindings**: Includes js functions, manual memory management, and Context for stateful operations
- **Decycler missing**: WASM-specific functions beyond basic Python bindings

### 2. **Optimization Patterns**
- **Both use core optimizations correctly**: `alloc_with_nan_prefix`, `make_uninit_matrix`, `init_matrix_prefixes`
- **ALMA better**: Uses `AVec` for weight vectors (cache alignment)
- **Decycler issue**: Allocates full-size hp vector (`vec![0.0; data.len()]`) in scalar implementation - potential optimization opportunity

### 3. **Feature Differences**
- **ALMA unique**: 
  - WASM-specific APIs (alloc/free/into/Context)
  - More sophisticated weight calculation with normalization
  - Better memory management in scalar kernel
- **Decycler unique**:
  - `batch_inner_into()` for Python - writes directly to output buffer
  - Simpler algorithm allows some optimizations

### 4. **Code Quality**
- **Both follow standards**: Proper error handling, builder pattern, streaming support
- **ALMA superior**: More complete API surface, better WASM integration
- **Decycler adequate**: Meets requirements but less feature-rich

## Recommendations for Decycler

1. **Add WASM bindings**: Implement `decycler_js()`, manual memory management functions
2. **Optimize scalar implementation**: Consider using rolling window instead of full hp vector
3. **Use AVec for small vectors**: Even though hp values are algorithm-specific, cache alignment could help
4. **Add Context API**: For stateful WASM operations if needed
5. **Consider in-place operations**: Add `_into()` variants for better memory efficiency