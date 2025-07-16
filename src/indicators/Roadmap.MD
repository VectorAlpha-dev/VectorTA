# Testing Standardization Roadmap

## Current State Analysis

### Testing Inconsistencies Identified

During the implementation of poison value detection tests across indicators, significant inconsistencies were discovered in parameter testing patterns:

| Indicator | Single Test Periods | Batch Configurations |
|-----------|-------------------|---------------------|
| supersmoother_3_pole | `[5, 10, 14, 20, 30, 50, 100, 200]` | `[(5,15,5), (10,30,10), (20,100,20), (7,7,1), (3,50,1)]` |
| supersmoother | `[3, 7, 10, 14, 20, 30, 50, 100, 200]` | `[(3,10,2), (10,30,10), (20,100,20), (5,5,1), (2,50,1)]` |
| swma | `[1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100]` | `[(1,10,1), (3,9,3), (5,25,5), (10,50,10), (2,2,1), (1,30,2)]` |
| zlema | `[1, 2, 3, 5, 7, 10, 14, 20, 21, 30, 50, 100, 200]` | `[(1,10,1), (3,21,3), (2,20,2), (10,50,10), (7,7,1), (8,8,1), (5,100,5)]` |

## Problems Identified

1. **Inconsistent Coverage**: Different indicators test different ranges and edge cases
2. **Maintenance Burden**: No centralized configuration for test parameters
3. **API Drift Risk**: Inconsistencies may mask actual API differences between indicators
4. **Missing Documentation**: No clear rationale for why certain indicators test specific parameters

## Recommendations

### 1. Implement Standardized Testing Framework

Create a centralized test configuration module:

```rust
// src/indicators/testing/mod.rs
pub const STANDARD_PERIODS: &[usize] = &[1, 2, 3, 5, 10, 14, 20, 30, 50, 100];

pub const STANDARD_BATCH_CONFIGS: &[(usize, usize, usize)] = &[
    (1, 10, 1),    // Dense small range
    (5, 25, 5),    // Medium sparse range
    (10, 50, 10),  // Large sparse range
    (7, 7, 1),     // Single period test
];
```

### 2. Support Indicator-Specific Extensions

Allow indicators to extend standard tests when mathematically justified:

```rust
pub fn get_indicator_test_periods(indicator_type: IndicatorType) -> Vec<usize> {
    let mut periods = STANDARD_PERIODS.to_vec();
    
    match indicator_type {
        IndicatorType::Zlema => periods.extend(&[21, 8]), // Odd/even for lag calculation
        IndicatorType::LongMemory => periods.push(200),   // Indicators needing long lookback
        _ => {}
    }
    
    periods
}
```

### 3. Document Test Rationale

Each indicator should document why it needs special test cases:

```rust
/// ZLEMA requires both odd and even period testing due to lag = (period - 1) / 2
/// affecting calculation behavior differently for odd vs even periods.
```

## Implementation Plan

### Phase 1: Standardization (Immediate)
- [ ] Create `src/indicators/testing/mod.rs` with standard configurations
- [ ] Refactor all poison detection tests to use standard configurations
- [ ] Document any indicator-specific requirements

### Phase 2: Validation (Short-term)
- [ ] Audit all existing tests for parameter coverage
- [ ] Ensure minimum parameter coverage across all indicators
- [ ] Add missing edge case tests where identified

### Phase 3: Automation (Long-term)
- [ ] Create macro for generating standardized tests
- [ ] Implement property-based testing for parameter ranges
- [ ] Add CI checks to enforce testing standards

## Success Metrics

- All indicators use the same base parameter set for testing
- Indicator-specific extensions are documented and justified
- Test maintenance time reduced by 50%
- Zero undetected poison values in production

## Next Steps

1. Review and approve this roadmap
2. Create the testing module with standard configurations
3. Systematically update all indicator tests
4. Document the new testing standards in CLAUDE.md