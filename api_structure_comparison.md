# API Structure Comparison: damiani_volatmeter.rs vs alma.rs

## Overview
Both indicators follow a consistent API design pattern with similar structures, though damiani_volatmeter has some additional complexity due to its multi-output nature.

## 1. Input/Output/Params Struct Patterns

### Data Enum Structure
Both use identical enum patterns for flexible data input:

**ALMA:**
```rust
pub enum AlmaData<'a> {
    Candles { candles: &'a Candles, source: &'a str },
    Slice(&'a [f64]),
}
```

**Damiani Volatmeter:**
```rust
pub enum DamianiVolatmeterData<'a> {
    Candles { candles: &'a Candles, source: &'a str },
    Slice(&'a [f64]),
}
```

### Output Structures
Key difference: ALMA has single output, Damiani has dual outputs.

**ALMA:**
```rust
pub struct AlmaOutput {
    pub values: Vec<f64>,
}
```

**Damiani Volatmeter:**
```rust
pub struct DamianiVolatmeterOutput {
    pub vol: Vec<f64>,
    pub anti: Vec<f64>,
}
```

### Parameters Structures
Both use Option<T> pattern for optional parameters with defaults:

**ALMA:**
```rust
pub struct AlmaParams {
    pub period: Option<usize>,
    pub offset: Option<f64>,
    pub sigma: Option<f64>,
}
```

**Damiani Volatmeter:**
```rust
pub struct DamianiVolatmeterParams {
    pub vis_atr: Option<usize>,
    pub vis_std: Option<usize>,
    pub sed_atr: Option<usize>,
    pub sed_std: Option<usize>,
    pub threshold: Option<f64>,
}
```

### Input Structures
Identical pattern with convenience methods:

**Both have:**
- `from_candles()` - Create from candle data with source selection
- `from_slice()` - Create from raw f64 slice
- `with_default_candles()` - Quick creation with default params
- Getter methods for each parameter with defaults

## 2. Builder Pattern Implementation

Both implement identical builder patterns:

### Structure
- Same field layout (parameters + kernel selection)
- Default implementation
- Method chaining for all parameters
- `apply()` and `apply_slice()` methods
- `into_stream()` method for streaming computation

**Key Methods (identical in both):**
```rust
pub fn new() -> Self
pub fn period/vis_atr/etc(mut self, n: usize) -> Self
pub fn kernel(mut self, k: Kernel) -> Self
pub fn apply(self, c: &Candles) -> Result<Output, Error>
pub fn apply_slice(self, d: &[f64]) -> Result<Output, Error>
pub fn into_stream(...) -> Result<Stream, Error>
```

## 3. Error Handling Approach

Both use thiserror-derived error enums with similar patterns:

### Common Error Types
- `EmptyData/EmptyInputData` - Empty input slice
- `AllValuesNaN` - All input values are NaN
- `InvalidPeriod` - Period validation errors
- `NotEnoughValidData` - Insufficient data after NaN removal

### Differences
**ALMA** has additional parameter-specific errors:
- `InvalidSigma` - Sigma validation
- `InvalidOffset` - Offset validation

**Damiani** combines all period errors into one:
- Single `InvalidPeriod` with all four period parameters

## 4. Public API Methods

### Core Functions (identical pattern):
```rust
// Main entry point
pub fn indicator_name(input: &Input) -> Result<Output, Error>

// With kernel selection
pub fn indicator_name_with_kernel(input: &Input, kernel: Kernel) -> Result<Output, Error>
```

### AsRef Implementation
Both implement `AsRef<[f64]>` for their input types with identical logic.

## 5. Stream/Context Patterns

### Stream Support
- **ALMA**: `AlmaStream` with `try_new()` constructor
- **Damiani**: `DamianiVolatmeterStream` with `new_from_candles()` constructor

Both builders have `into_stream()` methods, though parameters differ slightly:
- ALMA: No parameters needed
- Damiani: Requires candles and source parameters

## Key Similarities

1. **Consistent Naming**: Both follow `IndicatorName{Input,Output,Params,Error,Builder,Stream}` pattern
2. **Feature Gates**: Identical conditional compilation for Python/WASM features
3. **SIMD Support**: Both prepare for AVX2/AVX512 optimizations
4. **Memory Helpers**: Both import the same uninitialized memory helpers
5. **Default Trait**: Both implement Default for params with sensible defaults
6. **Builder Pattern**: Nearly identical implementation
7. **Error Handling**: Similar error types and patterns

## Key Differences

1. **Output Complexity**: Damiani has dual outputs (vol, anti) vs ALMA's single output
2. **Parameter Count**: Damiani has 5 parameters vs ALMA's 3
3. **Data Requirements**: Damiani's prepare function extracts high/low/close vs ALMA's single series
4. **Stream Construction**: Different approaches to stream initialization
5. **Error Specificity**: ALMA has more granular parameter validation errors

## Conclusion

Both indicators follow the same architectural blueprint with minor variations to accommodate their specific requirements. The damiani_volatmeter successfully adapts ALMA's patterns while handling its additional complexity (multiple outputs, more parameters, multi-series input).