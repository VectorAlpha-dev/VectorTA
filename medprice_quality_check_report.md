# MEDPRICE.RS Quality Check Report

## Comprehensive Comparison with ALMA.RS

### 1. Core API Structure

#### Input Structures

**ALMA:**
```rust
pub enum AlmaData<'a> {
    Candles { candles: &'a Candles, source: &'a str },
    Slice(&'a [f64]),
}

pub struct AlmaInput<'a> {
    pub data: AlmaData<'a>,
    pub params: AlmaParams,
}
```

**MEDPRICE:**
```rust
pub enum MedpriceData<'a> {
    Candles {
        candles: &'a Candles,
        high_source: &'a str,
        low_source: &'a str,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
    },
}

pub struct MedpriceInput<'a> {
    pub data: MedpriceData<'a>,
    pub params: MedpriceParams,
}
```

✅ **Assessment:** MEDPRICE correctly adapts the pattern for dual-source requirements (high/low)

#### Parameter Structures

**ALMA:**
```rust
pub struct AlmaParams {
    pub period: Option<usize>,
    pub offset: Option<f64>,
    pub sigma: Option<f64>,
}
```

**MEDPRICE:**
```rust
pub struct MedpriceParams;  // Empty struct
```

✅ **Assessment:** Correct - MEDPRICE has no parameters

#### Builder Patterns

**ALMA:**
- Has period/offset/sigma setters
- Supports kernel selection
- Provides apply methods for both candles and slices

**MEDPRICE:**
- Only has kernel setter (correct for parameterless indicator)
- Provides apply methods for both candles and slices
- Has `into_stream()` method

✅ **Assessment:** Appropriate adaptation for parameterless indicator

### 2. Memory Operations Analysis

#### Output Allocation

**ALMA:**
```rust
let mut out = alloc_with_nan_prefix(data.len(), first + period - 1);
```

**MEDPRICE:**
```rust
let mut out = alloc_with_nan_prefix(high.len(), first_valid_idx);
```

✅ **Assessment:** Correctly uses zero-copy helper function

#### Batch Operations

**ALMA:**
```rust
let mut buf_mu = make_uninit_matrix(rows, cols);
init_matrix_prefixes(&mut buf_mu, cols, &warm);
```

**MEDPRICE:**
```rust
let mut buf_mu = make_uninit_matrix(rows, cols);
let warmup_periods = vec![first];
init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);
```

✅ **Assessment:** Correctly uses uninitialized memory helpers

#### Vector Allocations Check

**MEDPRICE Line 428:** 
```rust
let warmup_periods = vec![first];
```
⚠️ **Issue:** Small allocation, but could be avoided

**MEDPRICE Line 463:**
```rust
let combos = vec![MedpriceParams::default()];
```
✅ **OK:** Small fixed-size allocation for parameter tracking

**MEDPRICE Lines 670, 859, 904:**
```rust
buffer: vec![f64::NAN; period],  // Line 670 in AlmaStream
let mut output = vec![0.0; high.len()];  // Lines 859, 904 in WASM
```
❌ **Critical Issues:**
- Line 670: Stream buffer allocation is acceptable (small, fixed size)
- Lines 859, 904: WASM functions allocate full-size output vectors - violates zero-copy principle

### 3. Function Signatures Comparison

#### Main Computation Functions

**ALMA:**
```rust
pub fn alma(input: &AlmaInput) -> Result<AlmaOutput, AlmaError>
pub fn alma_with_kernel(input: &AlmaInput, kernel: Kernel) -> Result<AlmaOutput, AlmaError>
```

**MEDPRICE:**
```rust
pub fn medprice(input: &MedpriceInput) -> Result<MedpriceOutput, MedpriceError>
pub fn medprice_with_kernel(input: &MedpriceInput, kernel: Kernel) -> Result<MedpriceOutput, MedpriceError>
```

✅ **Assessment:** Consistent pattern

#### Compute Into Functions

**ALMA:**
```rust
fn alma_compute_into(data: &[f64], weights: &[f64], period: usize, first: usize, inv_n: f64, kernel: Kernel, out: &mut [f64])
```

**MEDPRICE:**
```rust
pub fn medprice_compute_into(high: &[f64], low: &[f64], kernel: Kernel, out: &mut [f64]) -> Result<(), MedpriceError>
```

⚠️ **Difference:** MEDPRICE's compute_into is public and returns Result - inconsistent with ALMA's private helper pattern

### 4. Optimization Patterns

#### Kernel Detection
Both use identical patterns:
```rust
let chosen = match kernel {
    Kernel::Auto => detect_best_kernel(),
    other => other,
};
```

✅ **Assessment:** Consistent

#### Warmup Period Handling

**ALMA:** Complex calculation based on parameters
**MEDPRICE:** Simple first valid index search

✅ **Assessment:** Appropriate for indicator requirements

### 5. Error Handling

**ALMA Errors:**
- EmptyInputData
- AllValuesNaN
- InvalidPeriod
- NotEnoughValidData
- InvalidSigma
- InvalidOffset

**MEDPRICE Errors:**
- EmptyData
- DifferentLength (for high/low mismatch)
- AllValuesNaN

✅ **Assessment:** Appropriate error cases for indicator requirements

### 6. Streaming Implementation

**ALMA:** Complex circular buffer with weights
**MEDPRICE:** Simple stateless calculation

```rust
pub fn update(&mut self, high: f64, low: f64) -> Option<f64> {
    if high.is_nan() || low.is_nan() {
        return None;
    }
    Some((high + low) * 0.5)
}
```

✅ **Assessment:** Correct stateless implementation

### 7. Batch Processing

**MEDPRICE Batch:**
- Simplified expand_grid (returns single params)
- No parameter ranges (uses dummy range for API compatibility)
- Correctly uses uninitialized memory pattern

✅ **Assessment:** Appropriate simplification for parameterless indicator

## Critical Issues Found

### 1. WASM Memory Allocations
```rust
// Lines 859, 904 - VIOLATES ZERO-COPY PRINCIPLE
let mut output = vec![0.0; high.len()];
```

### 2. Public compute_into Function
The `medprice_compute_into` function is public while ALMA's equivalent is private. This breaks encapsulation.

### 3. Minor Allocation
```rust
let warmup_periods = vec![first];  // Could use array
```

## Recommendations

1. **Fix WASM functions** to use provided output buffers instead of allocating
2. Make `medprice_compute_into` private or document why it needs to be public
3. Replace `vec![first]` with `&[first]` or inline array
4. Consider adding `#[inline]` attributes to match ALMA's optimization hints

## Overall Assessment

MEDPRICE generally follows ALMA's patterns well, with appropriate adaptations for a parameterless, dual-source indicator. The main issues are:
- WASM function memory allocations (critical)
- Minor API inconsistencies
- Small optimization opportunities

The core computation and memory management patterns are sound, but the WASM bindings need immediate attention to comply with the zero-copy requirement.