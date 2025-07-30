# LRSI vs ALMA Implementation Comparison

## Key Differences Found

### 1. Memory Allocation Violations in LRSI

**LRSI Issues (Lines violating zero-copy requirement):**
- Line 185: `let mut price = Vec::with_capacity(high.len());` - Allocates vector proportional to input size
- Line 227-230: Multiple calls to `alloc_with_nan_prefix(n, first)` for l0, l1, l2, l3 vectors
- Line 540: `let mut price = Vec::with_capacity(high.len());` - Repeated in batch function
- Line 742: `let mut stream_values = Vec::with_capacity(high.len());` - In streaming test
- Line 867: `let mut price = Vec::with_capacity(high.len());` - In batch_inner_into

**ALMA Compliance:**
- Line 314: `let mut out = alloc_with_nan_prefix(data.len(), first + period - 1);` - Only for output vector
- Line 2254: `let mut output = vec![0.0; data.len()];` - Only in WASM binding (acceptable for binding layer)
- Uses helper functions properly for batch operations

### 2. Python Binding Optimization

**LRSI:**
- Has optimized `lrsi_batch_inner_into` function that writes directly to output slice
- Uses `PyArray1::new` with unsafe allocation and direct slice manipulation
- Properly uses `py.allow_threads` for GIL release

**ALMA:**
- Has similar `alma_batch_inner_into` function for direct output writing
- Same optimization pattern with PyArray1 and direct slice access
- Also properly releases GIL

### 3. API Parity

Both have complete feature parity:
- Builder pattern ✓
- Streaming API ✓
- Batch operations ✓
- Grid search support ✓
- Python bindings with streaming ✓
- WASM bindings ✓
- AVX2/AVX512 stubs ✓

### 4. Helper Function Usage

**LRSI:**
- Uses `alloc_with_nan_prefix` for output vectors ✓
- Uses `make_uninit_matrix` and `init_matrix_prefixes` for batch operations ✓
- BUT violates the rule by allocating intermediate vectors of size n

**ALMA:**
- Properly uses helper functions only for output vectors
- Does not allocate large intermediate vectors

## Specific Violations to Fix in LRSI

1. **Price calculation vectors** - These should be computed on-the-fly or use a different approach
2. **Laguerre filter state vectors (l0, l1, l2, l3)** - These are full-size vectors that violate the zero-copy requirement
3. **Test streaming values vector** - Should compare values one by one instead of collecting

## Recommendation

LRSI needs refactoring to eliminate all allocations of vectors proportional to input size, except for the final output vector. The Laguerre filter implementation should either:
1. Compute values on-the-fly without storing full history
2. Use a sliding window approach
3. Reuse the output buffer for intermediate calculations