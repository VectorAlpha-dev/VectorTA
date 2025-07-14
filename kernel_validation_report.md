# Kernel Validation Standardization Report

## Current State

### Indicators using `validate_kernel` (Safe approach):
1. **AD** - src/indicators/ad.rs
2. **ACOSC** - src/indicators/acosc.rs  
3. **VWAP** - src/indicators/moving_averages/vwap.rs
4. **VWMA** - src/indicators/moving_averages/vwma.rs
5. **WMA** - src/indicators/moving_averages/wma.rs

### Indicators using direct string-to-enum (Potentially unsafe):
1. **ALMA** - src/indicators/moving_averages/alma.rs
2. **VPWMA** - src/indicators/moving_averages/vpwma.rs
3. **ZLEMA** - src/indicators/moving_averages/zlema.rs
4. **WILDERS** - src/indicators/moving_averages/wilders.rs

## The Issue

The direct string-to-enum approach can cause segfaults if:
- User requests "avx2" on a CPU that doesn't support AVX2
- User requests "avx512" on a CPU that doesn't support AVX512
- The build was compiled without nightly-avx feature

## The Solution

The `validate_kernel` function in `src/utilities/kernel_validation.rs`:
- Checks CPU feature support at runtime
- Returns appropriate error messages if features aren't available
- Handles both single and batch kernel variants

## Standardization Approach

### Option 1: Update all indicators to use validate_kernel
**Pros:**
- Consistent error handling across all indicators
- Prevents segfaults from unsupported SIMD instructions
- Clear error messages for users

**Cons:**
- Requires changes to 4+ indicators
- Slightly more overhead (negligible)

### Option 2: Update validate_kernel logic into alma_prepare/compute functions
**Pros:**
- Centralized validation
- No changes to individual indicator files

**Cons:**
- Would require refactoring core compute functions
- More complex implementation

## Recommendation

**Use Option 1**: Update ALMA, VPWMA, ZLEMA, and WILDERS to use the validate_kernel helper.

This ensures:
1. Safety - no segfaults from unsupported instructions
2. Consistency - all indicators behave the same way
3. User-friendly - clear error messages
4. Minimal changes - just update the kernel parsing in Python functions

## Implementation Plan

For each indicator (ALMA, VPWMA, ZLEMA, WILDERS):

1. Add import:
```rust
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
```

2. Replace kernel parsing:
```rust
// OLD:
let kern = match kernel {
    None | Some("auto") => Kernel::Auto,
    Some("scalar") => Kernel::Scalar,
    Some("avx2") => Kernel::Avx2,
    Some("avx512") => Kernel::Avx512,
    Some(k) => return Err(PyValueError::new_err(format!("Unknown kernel: {}", k))),
};

// NEW:
let kern = validate_kernel(kernel, false)?;
```

3. For batch functions, use `validate_kernel(kernel, true)?`

This standardization will ensure all indicators are safe and consistent in their kernel validation.