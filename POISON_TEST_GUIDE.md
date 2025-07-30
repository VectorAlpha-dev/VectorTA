# Poison Test Guide for Rust-TA Indicators

## Overview

Poison tests detect uninitialized memory usage by checking if special debug-only "poison" values appear in indicator outputs. These tests only run in debug builds and are critical for ensuring memory safety.

## Poison Values

Three distinct patterns are used by our memory allocation helpers:

| Helper Function | Poison Value | Hex Pattern | Float Representation |
|----------------|--------------|-------------|---------------------|
| `alloc_with_nan_prefix` | Output buffer poison | `0x11111111_11111111` | ~2.2612e-308 |
| `init_matrix_prefixes` | Matrix initialization | `0x22222222_22222222` | ~4.5224e-308 |
| `make_uninit_matrix` | Matrix allocation | `0x33333333_33333333` | ~6.7837e-308 |

## Implementation Pattern

### Single Indicator Poison Test

```rust
#[cfg(debug_assertions)]
fn check_INDICATOR_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
    skip_if_unsupported!(kernel, test_name);
    
    let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
    let candles = read_candles_from_csv(file_path)?;
    
    // Define comprehensive parameter combinations
    let test_params = vec![
        IndicatorParams::default(),
        IndicatorParams { period: Some(5), /* other params */ },
        IndicatorParams { period: Some(20), /* other params */ },
        // Add edge cases: min period, max period, boundary conditions
        IndicatorParams { period: Some(2), /* minimum viable */ },
        IndicatorParams { period: Some(100), /* large period */ },
        // Add variations of other parameters if applicable
    ];
    
    for (param_idx, params) in test_params.iter().enumerate() {
        let input = IndicatorInput::from_candles(&candles, "close", params.clone());
        let output = indicator_with_kernel(&input, kernel)?;
        
        for (i, &val) in output.values.iter().enumerate() {
            if val.is_nan() {
                continue; // NaN values are expected during warmup
            }
            
            let bits = val.to_bits();
            
            // Check all three poison patterns
            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
                     with params: {:?} (param set {})",
                    test_name, val, bits, i, params, param_idx
                );
            }
            
            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
                     with params: {:?} (param set {})",
                    test_name, val, bits, i, params, param_idx
                );
            }
            
            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
                     with params: {:?} (param set {})",
                    test_name, val, bits, i, params, param_idx
                );
            }
        }
    }
    
    Ok(())
}

#[cfg(not(debug_assertions))]
fn check_INDICATOR_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
    Ok(()) // No-op in release builds
}
```

### Batch Indicator Poison Test

```rust
#[cfg(debug_assertions)]
fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
    skip_if_unsupported!(kernel, test);
    
    let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
    let c = read_candles_from_csv(file)?;
    
    // Test various parameter sweep configurations
    let test_configs = vec![
        // (period_start, period_end, period_step, other_param_ranges...)
        (2, 10, 2),      // Small periods
        (5, 25, 5),      // Medium periods
        (30, 60, 15),    // Large periods
        (2, 5, 1),       // Dense small range
        // Add more based on your indicator's parameters
    ];
    
    for (cfg_idx, config) in test_configs.iter().enumerate() {
        let output = IndicatorBatchBuilder::new()
            .kernel(kernel)
            .period_range(config.0, config.1, config.2)
            // Add other parameter ranges as needed
            .apply_candles(&c, "close")?;
        
        for (idx, &val) in output.values.iter().enumerate() {
            if val.is_nan() {
                continue;
            }
            
            let bits = val.to_bits();
            let row = idx / output.cols;
            let col = idx % output.cols;
            let combo = &output.combos[row];
            
            // Check all three poison patterns with detailed context
            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
                     at row {} col {} (flat index {}) with params: {:?}",
                    test, cfg_idx, val, bits, row, col, idx, combo
                );
            }
            
            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
                     at row {} col {} (flat index {}) with params: {:?}",
                    test, cfg_idx, val, bits, row, col, idx, combo
                );
            }
            
            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
                     at row {} col {} (flat index {}) with params: {:?}",
                    test, cfg_idx, val, bits, row, col, idx, combo
                );
            }
        }
    }
    
    Ok(())
}
```

## Integration with Test Framework

Add poison tests to your test generation macros:

```rust
// For single indicator tests
generate_all_INDICATOR_tests!(
    check_INDICATOR_accuracy,
    check_INDICATOR_empty_input,
    // ... other tests ...
    check_INDICATOR_no_poison  // Add this
);

// For batch tests
gen_batch_tests!(check_batch_default_row);
gen_batch_tests!(check_batch_sweep);
gen_batch_tests!(check_batch_no_poison);  // Add this
```

## Key Requirements

1. **Parameter Coverage**: Test diverse parameter combinations including:
   - Default parameters
   - Minimum viable parameters
   - Maximum reasonable parameters
   - Edge cases specific to your indicator
   - Common use cases

2. **Error Messages**: Include:
   - Test name and kernel type
   - Exact poison value and hex representation
   - Index/location where found
   - Complete parameter set that triggered it
   - For batch: row, column, and flat index

3. **Performance**: Only runs in debug builds (`#[cfg(debug_assertions)]`)

4. **NaN Handling**: Skip NaN values - they're expected during warmup periods

## Example Adaptation

For an RSI indicator with period parameter:

```rust
let test_params = vec![
    RsiParams::default(),                    // period: 14
    RsiParams { period: Some(2) },          // minimum
    RsiParams { period: Some(7) },          // small
    RsiParams { period: Some(21) },         // medium
    RsiParams { period: Some(50) },         // large
    RsiParams { period: Some(200) },        // very large
];
```

For a Bollinger Bands indicator with period and num_std:

```rust
let test_params = vec![
    BollingerParams::default(),             // period: 20, num_std: 2.0
    BollingerParams { period: Some(10), num_std: Some(1.0) },
    BollingerParams { period: Some(20), num_std: Some(3.0) },
    BollingerParams { period: Some(50), num_std: Some(2.5) },
];
```

## Common Pitfalls

1. **Don't check float equality** - always use `to_bits()` for poison detection
2. **Don't skip debug builds** - poison values only exist in debug mode
3. **Don't forget edge cases** - test minimum/maximum parameter values
4. **Include parameter context** - error messages must identify which params failed

## Testing

Run poison tests with:
```bash
cargo test --features nightly-avx --lib indicators::INDICATOR_NAME::check_INDICATOR_no_poison -- --nocapture
```

The `--nocapture` flag ensures you see the detailed panic messages if poison is found.