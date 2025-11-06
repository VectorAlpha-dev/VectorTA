# Python and WASM Bindings Test Setup (Updated for PyO3 0.20+)

## Key API Changes in Recent PyO3/numpy Versions

### 1. **Function Signatures**
- Python functions need unique names when exposed to Python
- Use `#[pyo3(name = "python_name")]` to control the Python-visible name
- Return `Py<T>` instead of `&'py T` for owned Python objects

### 2. **Array Access**
- Import `PyArrayMethods` trait for array methods
- Use `PyReadonlyArray1` for input arrays
- Use `.into_pyarray_bound(py)` to create new arrays
- Use `.bind(py)` to get a bound reference from `Py<T>`

### 3. **Module Registration**
- Create a separate registration function for each indicator
- Use `Bound<'_, PyModule>` instead of `&PyModule`
- Add to main module in lib.rs

## Example Setup for Your Project

### 1. **Update lib.rs** to register Python modules:

```rust
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn ta_indicators(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register each indicator's functions
    indicators::zscore::register_zscore_module(m)?;
    // indicators::alma::register_alma_module(m)?;
    // Add other indicators here...
    Ok(())
}
```

### 2. **Running Tests**

```bash
# First, ensure you have the right dependencies
pip install maturin pytest numpy

# Build and install the Python module
maturin develop --features python --release

# Run Rust tests with Python features
cargo test --features python

# Run Python integration tests
pytest tests/test_zscore_python.py -v
```

### 3. **Example Python Test File**

```python
# tests/test_zscore_python.py
import pytest
import numpy as np
import ta_indicators

def test_zscore_basic():
    """Test basic zscore calculation"""
    data = np.array([10.0, 20.0, 30.0, 40.0, 50.0] * 3, dtype=np.float64)
    result = ta_indicators.zscore(data, period=14, ma_type="sma", nbdev=1.0, devtype=0)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == len(data)
    assert np.isnan(result[:13]).all()  # First 13 should be NaN
    assert not np.isnan(result[13])     # 14th should have a value

def test_zscore_stream():
    """Test streaming zscore calculation"""
    stream = ta_indicators.ZscoreStream(period=10, ma_type="sma", nbdev=1.0, devtype=0)
    
    values = []
    for i in range(20):
        val = stream.update(float(i))
        values.append(val)
    
    # First 9 should be None
    assert all(v is None for v in values[:9])
    # 10th and beyond should have values
    assert all(v is not None for v in values[9:])

def test_zscore_batch():
    """Test batch zscore calculation"""
    data = np.random.randn(100)
    result = ta_indicators.zscore_batch(
        data,
        period_range=(10, 20, 5),
        ma_type="sma",
        nbdev_range=(1.0, 2.0, 0.5),
        devtype_range=(0, 0, 1)
    )
    
    assert isinstance(result, dict)
    assert 'values' in result
    assert 'periods' in result
    assert 'nbdevs' in result
    assert 'devtypes' in result
    
    # Check that values is a 2D array
    assert result['values'].ndim == 2

def test_zscore_errors():
    """Test error handling"""
    # Empty data should raise
    with pytest.raises(ValueError):
        ta_indicators.zscore(np.array([]), period=14, ma_type="sma", nbdev=1.0, devtype=0)
    
    # All NaN should raise
    with pytest.raises(ValueError):
        ta_indicators.zscore(np.full(10, np.nan), period=5, ma_type="sma", nbdev=1.0, devtype=0)
```

### 4. **WASM Testing Setup**

For WASM tests, ensure your `Cargo.toml` has:

```toml
[target.'cfg(target_arch = "wasm32")'.dev-dependencies]
wasm-bindgen-test = "0.3"
```

Run WASM tests with:
```bash
# Install wasm-pack if needed
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Run tests in Node.js
wasm-pack test --node --features wasm

# Run tests in browser
wasm-pack test --chrome --headless --features wasm
```

### 5. **Common Issues and Solutions**

1. **"PyModule has no method add_function"**
   - Use `Bound<'_, PyModule>` instead of `&PyModule`
   - Use the registration pattern shown above

2. **"as_slice_mut() not found"**
   - Import `PyArrayMethods` trait
   - For read-only access, use `PyReadonlyArray1`
   - Create new arrays and return them instead of modifying in-place

3. **"cannot convert to Python object"**
   - Return `Py<T>` types for Python objects
   - Use `.into_pyarray_bound(py).into()` for arrays
   - Use `.into()` to convert `Bound<T>` to `Py<T>`

4. **Module not found in Python**
   - Ensure the module name in `Cargo.toml` matches your import
   - Check that `maturin develop` completed successfully
   - Verify Python can find the built module (check `sys.path`)

### 6. **GitHub Actions CI Example**

```yaml
name: Test Python and WASM Bindings

on: [push, pull_request]

jobs:
  test-python:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.10', '3.12']
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable
      - name: Install dependencies
        run: |
          pip install maturin pytest numpy
      - name: Build and test
        run: |
          maturin develop --features python
          pytest tests/test_*_python.py -v
          
  test-wasm:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable
      - name: Install wasm-pack
        run: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
      - name: Run WASM tests
        run: |
          wasm-pack test --node --features wasm
          wasm-pack test --chrome --headless --features wasm
```

This setup ensures your Python and WASM bindings are tested as conveniently as your Rust tests!