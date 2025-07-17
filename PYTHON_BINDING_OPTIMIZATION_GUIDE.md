# Python Binding Optimization Guide for Rust-Backtester Indicators

This guide provides a systematic approach to optimize Python bindings for all indicators to match the performance level achieved in alma.rs. The optimization reduces binding overhead from ~60% to near 0% by using zero-copy transfers and eliminating redundant operations.

## Table of Contents
1. [Core Optimization Principles](#core-optimization-principles)
2. [Implementation Patterns](#implementation-patterns)
3. [Special Cases](#special-cases)
4. [Required Imports](#required-imports)
5. [Optimization Checklist](#optimization-checklist)
6. [Performance Validation](#performance-validation)
7. [Common Pitfalls](#common-pitfalls)

## Core Optimization Principles

### 1. Use `Vec<f64>::into_pyarray()` for Zero-Copy Transfer
- **Never** pre-allocate NumPy arrays and copy data
- **Always** return Rust `Vec<f64>` directly as NumPy arrays
- This eliminates one memory allocation and copy operation

### 2. Remove Redundant NaN Filling
- The Rust functions already use `alloc_with_nan_prefix()`
- **Never** manually fill NaN values in Python bindings
- Trust the Rust implementation to handle warmup periods correctly

### 3. Keep Computation Inside `py.allow_threads()`
- All heavy computation must happen without the GIL
- Only type conversions and final array creation should hold the GIL

## Implementation Patterns

### Pattern 1: Single Output Indicator

#### ❌ Before (Inefficient):
```rust
#[pyfunction]
pub fn indicator_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    param1: usize,
    param2: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice_in = data.as_slice()?;
    let params = IndicatorParams { 
        param1: Some(param1), 
        param2: Some(param2) 
    };
    let input = IndicatorInput::from_slice(slice_in, params);

    // BAD: Pre-allocating NumPy array
    let out_arr = unsafe { PyArray1::<f64>::new(py, [slice_in.len()], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    py.allow_threads(|| -> Result<(), IndicatorError> {
        // BAD: Manual NaN filling
        slice_out[..warmup].fill(f64::NAN);
        // BAD: Writing directly to NumPy array
        indicator_compute_into(data, params, slice_out);
        Ok(())
    })?;

    Ok(out_arr)
}
```

#### ✅ After (Optimized):
```rust
#[pyfunction]
#[pyo3(signature = (data, param1, param2, kernel=None))]
pub fn indicator_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    param1: usize,
    param2: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;  // Validate before allow_threads
    
    let params = IndicatorParams { 
        param1: Some(param1), 
        param2: Some(param2) 
    };
    let input = IndicatorInput::from_slice(slice_in, params);

    // GOOD: Get Vec<f64> from Rust function
    let result_vec: Vec<f64> = py.allow_threads(|| {
        indicator_with_kernel(&input, kern)
            .map(|o| o.values)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // GOOD: Zero-copy transfer to NumPy
    Ok(result_vec.into_pyarray(py))
}
```

### Pattern 2: Multiple Output Indicator

For indicators that return multiple arrays (e.g., Bollinger Bands returns upper, middle, lower):

#### ❌ Before (Inefficient):
```rust
#[pyfunction]
pub fn indicator_multi_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    param: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    // BAD: Pre-allocating multiple arrays
    let upper_arr = unsafe { PyArray1::<f64>::new(py, [data.len()], false) };
    let middle_arr = unsafe { PyArray1::<f64>::new(py, [data.len()], false) };
    let lower_arr = unsafe { PyArray1::<f64>::new(py, [data.len()], false) };
    // ... manual copying ...
}
```

#### ✅ After (Optimized):
```rust
#[pyfunction]
#[pyo3(signature = (data, param, kernel=None))]
pub fn indicator_multi_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    param: usize,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    
    let params = IndicatorParams { param: Some(param) };
    let input = IndicatorInput::from_slice(slice_in, params);

    // GOOD: Get all vectors from Rust
    let (upper_vec, middle_vec, lower_vec) = py.allow_threads(|| {
        indicator_with_kernel(&input, kern)
            .map(|o| (o.upper, o.middle, o.lower))
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // GOOD: Zero-copy transfer for each output
    Ok((
        upper_vec.into_pyarray(py),
        middle_vec.into_pyarray(py),
        lower_vec.into_pyarray(py)
    ))
}
```

### Pattern 3: Batch Operations

Key Points for Batch:
- Pre-allocate NumPy array for batch (this is acceptable since we write directly)
- Use `_batch_inner_into` variant that writes directly to the buffer
- Use `into_pyarray()` for parameter arrays

#### ✅ Optimized Batch Pattern:
```rust
#[pyfunction]
#[pyo3(signature = (data, param_range, kernel=None))]
pub fn indicator_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    param_range: (usize, usize, usize),  // (start, end, step)
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;  // true for batch operations
    let sweep = IndicatorBatchRange { param: param_range };

    // Calculate dimensions
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    // Pre-allocate output array (OK for batch operations)
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Compute without GIL
    let combos = py.allow_threads(|| {
        // Handle kernel selection for batch operations
        let kernel = match kern {
            Kernel::Auto => detect_best_batch_kernel(),
            k => k,
        };
        
        // For indicators with SIMD kernels, map batch kernels to regular kernels
        // let simd = match kernel {
        //     Kernel::Avx512Batch => Kernel::Avx512,
        //     Kernel::Avx2Batch => Kernel::Avx2,
        //     Kernel::ScalarBatch => Kernel::Scalar,
        //     _ => kernel,
        // };
        
        indicator_batch_inner_into(slice_in, &sweep, kernel, true, slice_out)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Build result dictionary
    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    
    // For single-parameter indicators:
    dict.set_item(
        "params",
        combos.iter()
            .map(|p| p.param.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py)
    )?;
    
    // For multi-parameter indicators like ALMA:
    // dict.set_item("periods", combos.iter().map(|p| p.period.unwrap() as u64).collect::<Vec<_>>().into_pyarray(py))?;
    // dict.set_item("offsets", combos.iter().map(|p| p.offset.unwrap()).collect::<Vec<_>>().into_pyarray(py))?;
    // dict.set_item("sigmas", combos.iter().map(|p| p.sigma.unwrap()).collect::<Vec<_>>().into_pyarray(py))?;

    Ok(dict)
}
```

### Pattern 4: Streaming Indicator

```rust
#[pyclass(name = "IndicatorStream")]
pub struct IndicatorStreamPy {
    inner: IndicatorStream,
}

#[pymethods]
impl IndicatorStreamPy {
    #[new]
    pub fn new(param1: usize, param2: f64) -> PyResult<Self> {
        let params = IndicatorParams {
            param1: Some(param1),
            param2: Some(param2),
        };
        let inner = IndicatorStream::try_new(params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(IndicatorStreamPy { inner })
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.inner.update(value)
    }
}
```

## Special Cases

### 1. Indicators with Multiple Required Inputs

Some indicators need high, low, close arrays. Handle similarly:

```rust
#[pyfunction]
#[pyo3(signature = (high, low, close, param, kernel=None))]
pub fn indicator_hlc_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    param: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let result_vec = py.allow_threads(|| {
        indicator_hlc(high_slice, low_slice, close_slice, param, kern)
            .map(|o| o.values)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}
```

### 2. Optional Kernel Parameter

Always validate kernel parameter outside `allow_threads`:
```rust
let kern = validate_kernel(kernel, false)?;  // Before allow_threads
```

### 3. Return Types

- Single output: Return `PyArray1<f64>`
- Multiple outputs: Return tuple of `PyArray1<f64>`
- Batch operations: Return `PyDict` with "values" and parameter arrays

## Required Imports

Always ensure these imports are present:

```rust
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::types::PyDict;  // Only for batch operations
```

## Optimization Checklist

When optimizing an indicator's Python binding:

- [ ] Remove any `PyArray1::new()` followed by `copy_from_slice()`
- [ ] Remove any manual NaN filling (lines like `slice[..warmup].fill(f64::NAN)`)
- [ ] Use the existing `indicator_with_kernel()` function that returns a struct with `Vec<f64>`
- [ ] Add `IntoPyArray` to imports if not present
- [ ] Use `.into_pyarray(py)` for zero-copy transfer
- [ ] Keep all computation inside `py.allow_threads()`
- [ ] Validate parameters (like kernel) before entering `allow_threads`
- [ ] For multi-output indicators, return tuple of arrays using `into_pyarray()` for each
- [ ] For batch operations, keep the pre-allocated array pattern but use `into_pyarray()` for parameter arrays
- [ ] Add proper PyO3 signature macro: `#[pyo3(signature = (...))]`
- [ ] Include optional `kernel` parameter with default `None`

## Performance Validation

After optimization, verify performance with:

```python
import time
import numpy as np
import rust_backtester as ta

data = np.random.randn(1_000_000).astype(np.float64)

# Warmup
for _ in range(10):
    _ = ta.indicator(data, params...)

# Benchmark
times = []
for _ in range(100):
    start = time.perf_counter()
    _ = ta.indicator(data, params...)
    times.append((time.perf_counter() - start) * 1000)

print(f"Median time: {np.median(times):.2f} ms")
```

Expected improvement: 30-50% reduction in execution time.

## Common Pitfalls

### ❌ Don't Do This:
1. **Don't mix old and new patterns** - Either use full zero-copy or don't
2. **Don't hold GIL during computation** - Only for array creation
3. **Don't forget error mapping** - Use `.map_err(|e| PyValueError::new_err(e.to_string()))`
4. **Don't change the public API** - Keep the same function signatures
5. **Don't remove the kernel parameter** - Even if optional, maintain API compatibility
6. **Don't use mutable NumPy arrays** - Always use `PyReadonlyArray1` for inputs
7. **Don't manually manage warmup** - Let Rust handle NaN prefixes

### ✅ Do This:
1. **Use zero-copy transfers** - `Vec::into_pyarray()`
2. **Release GIL during computation** - `py.allow_threads()`
3. **Validate inputs early** - Before entering `allow_threads`
4. **Return appropriate types** - Single array, tuple, or dict as needed
5. **Include kernel parameter** - Even if users rarely use it
6. **Trust Rust allocations** - Use `alloc_with_nan_prefix` patterns

## Example: Complete ALMA Implementation

Here's the complete optimized ALMA implementation as a reference:

### Single Calculation
```rust
#[pyfunction(name = "alma")]
#[pyo3(signature = (data, period, offset, sigma, kernel=None))]
pub fn alma_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    offset: f64,
    sigma: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};
    
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    
    let params = AlmaParams {
        period: Some(period),
        offset: Some(offset),
        sigma: Some(sigma),
    };
    let alma_in = AlmaInput::from_slice(slice_in, params);
    
    let result_vec: Vec<f64> = py.allow_threads(|| {
        alma_with_kernel(&alma_in, kern).map(|o| o.values)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    
    Ok(result_vec.into_pyarray(py))
}
```

### Batch Calculation
```rust
#[pyfunction(name = "alma_batch")]
#[pyo3(signature = (data, period_range, offset_range, sigma_range, kernel=None))]
pub fn alma_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    offset_range: (f64, f64, f64),
    sigma_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;
    
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    
    let sweep = AlmaBatchRange {
        period: period_range,
        offset: offset_range,
        sigma: sigma_range,
    };
    
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();
    
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };
    
    let combos = py.allow_threads(|| {
        let kernel = match kern {
            Kernel::Auto => detect_best_batch_kernel(),
            k => k,
        };
        let simd = match kernel {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            _ => unreachable!(),
        };
        alma_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    
    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item("periods", combos.iter().map(|p| p.period.unwrap() as u64).collect::<Vec<_>>().into_pyarray(py))?;
    dict.set_item("offsets", combos.iter().map(|p| p.offset.unwrap()).collect::<Vec<_>>().into_pyarray(py))?;
    dict.set_item("sigmas", combos.iter().map(|p| p.sigma.unwrap()).collect::<Vec<_>>().into_pyarray(py))?;
    
    Ok(dict)
}
```

### Streaming Class
```rust
#[pyclass(name = "AlmaStream")]
pub struct AlmaStreamPy {
    stream: AlmaStream,
}

#[pymethods]
impl AlmaStreamPy {
    #[new]
    fn new(period: usize, offset: f64, sigma: f64) -> PyResult<Self> {
        let params = AlmaParams {
            period: Some(period),
            offset: Some(offset),
            sigma: Some(sigma),
        };
        let stream = AlmaStream::try_new(params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(AlmaStreamPy { stream })
    }
    
    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}
```

## Summary

The key changes for optimization are:
1. Removed `PyArray1::new()` + `as_slice_mut()` + manual NaN filling
2. Added `IntoPyArray` import
3. Changed to return `result_vec.into_pyarray(py)`
4. Kept all computation in `py.allow_threads()`
5. Added kernel parameter support

Following this guide should achieve similar performance improvements (from ~60% overhead to <10% overhead) for all indicators.