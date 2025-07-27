# Detailed Comparison: CKSP.rs vs ALMA.rs

This report compares the implementation quality between `src/indicators/cksp.rs` and `src/indicators/moving_averages/alma.rs`, excluding scalar/avx2/avx512 kernel implementations.

## 1. Module Structure and Imports

### ALMA.rs
- **Python imports**: Grouped at top with feature gate
- **WASM imports**: Grouped with feature gate  
- **Core imports**: Well organized by category
- **Placement**: AsRef trait implementation appears early (line 33)

### CKSP.rs
- **Python imports**: Properly feature-gated (lines 35-44)
- **WASM imports**: Properly feature-gated (lines 46-49)
- **Core imports**: Well organized
- **Missing**: No early AsRef implementation

**Gap**: CKSP places AsRef implementation later (line 122) instead of early like ALMA

## 2. Input/Output/Params Structs and Traits

### ALMA.rs
- **Input enum**: `AlmaData` supports both Slice and Candles with source selection
- **Params**: Has Default implementation with sensible defaults
- **Builder methods**: `from_candles`, `from_slice`, `with_default_candles`
- **Getters**: Clean getter methods for parameters

### CKSP.rs  
- **Input enum**: `CkspData` supports Candles and Slices (high/low/close)
- **Params**: Has Default implementation
- **Builder methods**: `from_candles`, `from_slices`, `with_default_candles`
- **Getters**: Clean getter methods

**Gap**: CKSP uses plural "Slices" vs ALMA's singular "Slice" - minor inconsistency

## 3. Builder Pattern Implementation

### ALMA.rs
- **Fields**: period, offset, sigma, kernel
- **Methods**: Fluent API with `apply`, `apply_slice`, `into_stream`
- **Kernel support**: Explicit kernel parameter

### CKSP.rs
- **Fields**: p, x, q, kernel  
- **Methods**: Fluent API with `apply`, `apply_slices`, `into_stream`
- **Kernel support**: Explicit kernel parameter

**Quality**: Both implementations are equivalent

## 4. Error Types and Handling

### ALMA.rs
- **Error variants**: 7 specific error types
  - EmptyInputData, AllValuesNaN, InvalidPeriod, NotEnoughValidData, InvalidSigma, InvalidOffset
- **Error messages**: Detailed with context
- **thiserror**: Used for derive

### CKSP.rs
- **Error variants**: 5 error types
  - NoData, NotEnoughData, InconsistentLengths, InvalidParam, CandleFieldError
- **Error messages**: Good detail
- **thiserror**: Used for derive

**Gap**: ALMA has more specific error types (e.g., separate InvalidSigma/InvalidOffset vs generic InvalidParam)

## 5. Main Compute Functions and Signatures

### ALMA.rs
```rust
pub fn alma(input: &AlmaInput) -> Result<AlmaOutput, AlmaError>
pub fn alma_with_kernel(input: &AlmaInput, kernel: Kernel) -> Result<AlmaOutput, AlmaError>
```

### CKSP.rs
```rust
pub fn cksp(input: &CkspInput) -> Result<CkspOutput, CkspError>
pub fn cksp_with_kernel(input: &CkspInput, kernel: Kernel) -> Result<CkspOutput, CkspError>
```

**Quality**: Identical pattern

## 6. Helper Functions

### ALMA.rs
- **alma_prepare**: Extracts and validates parameters, prepares weights
- **alma_compute_into**: Core computation logic
- **alma_into_slice**: Computes directly into provided slice
- **Memory**: Uses `alloc_with_nan_prefix` correctly

### CKSP.rs
- **cksp_compute_into**: Core computation logic
- **Memory**: Uses `alloc_with_nan_prefix` correctly
- **Missing**: No separate prepare function - validation mixed with computation

**Gap**: ALMA separates preparation/validation from computation; CKSP mixes them

## 7. Streaming API Implementation

### ALMA.rs
- **State**: period, weights, inv_norm, buffer, head, filled
- **Ring buffer**: Proper circular buffer implementation
- **update**: Returns Option<f64>
- **Validation**: In try_new constructor

### CKSP.rs  
- **State**: Complex with multiple deques for different calculations
- **update**: Returns Option<(f64, f64)> for long/short values
- **Validation**: In try_new constructor
- **Dependencies**: Uses VecDeque collections

**Quality**: Both appropriate for their algorithms

## 8. Batch API Implementation

### ALMA.rs
- **Builder**: AlmaBatchBuilder with range methods
- **Range struct**: AlmaBatchRange with defaults
- **Output**: AlmaBatchOutput with helper methods
- **Functions**: alma_batch_slice, alma_batch_par_slice, alma_batch_inner
- **Memory**: Proper use of make_uninit_matrix and init_matrix_prefixes
- **Parameter expansion**: expand_grid function
- **Row functions**: alma_row_scalar, alma_row_avx2, alma_row_avx512

### CKSP.rs
- **Builder**: CkspBatchBuilder with range methods  
- **Range struct**: CkspBatchRange with defaults
- **Output**: CkspBatchOutput with helper methods (includes both long/short)
- **Functions**: cksp_batch_slice, cksp_batch_par_slice, cksp_batch_inner
- **Memory**: Proper use of make_uninit_matrix and init_matrix_prefixes
- **Parameter expansion**: expand_grid function
- **Row functions**: cksp_row_scalar, cksp_row_avx2, cksp_row_avx512

**Quality**: Nearly identical implementation patterns

## 9. Test Structure and Coverage

### ALMA.rs Tests
1. check_alma_partial_params
2. check_alma_accuracy
3. check_alma_default_candles
4. check_alma_zero_period
5. check_alma_period_exceeds_length
6. check_alma_very_small_dataset
7. check_alma_empty_input
8. check_alma_invalid_sigma
9. check_alma_invalid_offset
10. check_alma_reinput
11. check_alma_nan_handling
12. check_alma_streaming
13. check_alma_no_poison (debug only)

### CKSP.rs Tests
1. check_cksp_partial_params
2. check_cksp_accuracy
3. check_cksp_default_candles
4. check_cksp_zero_period
5. check_cksp_period_exceeds_length
6. check_cksp_very_small_dataset
7. check_cksp_reinput
8. check_cksp_nan_handling
9. check_cksp_streaming
10. check_cksp_no_poison (debug only)

**Gaps**: 
- CKSP missing: empty_input test, invalid parameter-specific tests
- CKSP has: inconsistent_lengths test (not in ALMA)

## 10. Documentation Quality

### ALMA.rs
- Module-level documentation: Missing
- Function documentation: Minimal
- Error documentation: In error enum
- Parameter documentation: In error messages

### CKSP.rs
- Module-level documentation: Comprehensive (lines 1-19)
- Function documentation: Minimal  
- Error documentation: In module docs and error enum
- Parameter documentation: In module docs

**Gap**: CKSP has better module-level documentation; ALMA lacks it entirely

## Summary of Missing Features and Quality Gaps

### CKSP.rs is missing:
1. Early AsRef trait placement (minor style issue)
2. Separate prepare function for validation/setup
3. More specific error types (uses generic InvalidParam)
4. Empty input test case
5. Parameter-specific validation tests

### CKSP.rs advantages:
1. Better module-level documentation
2. InconsistentLengths error handling

### Overall Assessment:
Both implementations follow very similar patterns and meet the quality standards. The main differences are:
- ALMA has slightly better code organization (prepare function)
- ALMA has more granular error types
- CKSP has better documentation
- Both properly implement all required APIs (streaming, batch, builder)
- Both use proper memory allocation patterns
- Both have comprehensive test coverage

The implementations are largely equivalent in quality, with only minor differences in style and organization.