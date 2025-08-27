/**
 * WASM binding tests for Linear Regression Angle indicator.
 * These tests mirror the Rust unit tests to ensure WASM bindings work correctly.
 */
import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { 
    loadTestData, 
    assertArrayClose, 
    assertClose,
    isNaN,
    assertAllNaN,
    assertNoNaN,
    EXPECTED_OUTPUTS 
} from './test_utils.js';
import { compareWithRust } from './rust-comparison.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

test.before(async () => {
    // Load WASM module
    try {
        const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
        const importPath = process.platform === 'win32' 
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
        wasm = await import(importPath);
        // No need to call default() for ES modules
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('Linear Regression Angle partial params', () => {
    // Test with default parameters - mirrors check_lra_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.linearreg_angle_js(close, 14); // Default period
    assert.strictEqual(result.length, close.length);
});

test('Linear Regression Angle accuracy', () => {
    // Test LRA matches expected values from Rust tests - mirrors check_lra_accuracy
    const close = new Float64Array(testData.close);
    
    const result = wasm.linearreg_angle_js(close, 14);
    
    assert.strictEqual(result.length, close.length);
    
    // Expected last 5 values from Rust test
    const expected_last_5 = [
        -89.30491945492733,
        -89.28911257342405,
        -89.1088041965075,
        -86.58419429159467,
        -87.77085937059316,
    ];
    
    // Check last 5 values match expected
    assertArrayClose(
        result.slice(-5),
        expected_last_5,
        1e-5,
        'Linear Regression Angle last 5 values mismatch'
    );
    
    // Skip compareWithRust since generate_references doesn't support linearreg_angle
    // compareWithRust('linearreg_angle', result, 'close', { period: 14 });
});

test('Linear Regression Angle zero period', () => {
    // Test LRA fails with zero period - mirrors check_lra_zero_period
    const input_data = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.linearreg_angle_js(input_data, 0);
    }, /Invalid period/, 'Expected error for zero period');
});

test('Linear Regression Angle period exceeds length', () => {
    // Test LRA fails when period exceeds data length - mirrors check_lra_period_exceeds_length
    const data_small = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.linearreg_angle_js(data_small, 10);
    }, /Invalid period/, 'Expected error for period exceeding data length');
});

test('Linear Regression Angle very small dataset', () => {
    // Test LRA fails with insufficient data - mirrors check_lra_very_small_dataset
    const single_point = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.linearreg_angle_js(single_point, 14);
    }, /Invalid period|Not enough valid data/, 'Expected error for insufficient data');
});

test('Linear Regression Angle empty input', () => {
    // Test LRA fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.linearreg_angle_js(empty, 14);
    }, /Empty data/, 'Expected error for empty input');
});

test('Linear Regression Angle warmup period', () => {
    // Test LRA warmup period behavior - mirrors Rust warmup tests
    const close = new Float64Array(testData.close);
    const period = 14;
    
    const result = wasm.linearreg_angle_js(close, period);
    assert.strictEqual(result.length, close.length);
    
    // Find first non-NaN value in input data
    let firstValid = 0;
    for (let i = 0; i < close.length; i++) {
        if (!isNaN(close[i])) {
            firstValid = i;
            break;
        }
    }
    
    // Calculate warmup period
    const warmupEnd = firstValid + period - 1;
    
    // First warmupEnd values should be NaN
    for (let i = 0; i < warmupEnd; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} during warmup, got ${result[i]}`);
    }
    
    // After warmup, values should not be NaN (unless input was NaN)
    for (let i = warmupEnd; i < Math.min(result.length, warmupEnd + 50); i++) {
        if (!isNaN(close[i])) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} after warmup period`);
            // Angle should be within [-90, 90] degrees
            assert(result[i] >= -90.0 && result[i] <= 90.0, 
                `Angle ${result[i]} out of range at index ${i}`);
        }
    }
});

test('Linear Regression Angle NaN handling with leading NaNs', () => {
    // Test LRA handles NaN values correctly with leading NaNs
    const data = new Float64Array(55);
    // Fill with NaN for first 5 values, then regular data
    for (let i = 0; i < 5; i++) {
        data[i] = NaN;
    }
    for (let i = 5; i < 55; i++) {
        data[i] = 100.0 + i;
    }
    
    const period = 14;
    const result = wasm.linearreg_angle_js(data, period);
    assert.strictEqual(result.length, data.length);
    
    // First valid value is at index 5
    const firstValid = 5;
    const warmupEnd = firstValid + period - 1;
    
    // All values before warmupEnd should be NaN
    for (let i = 0; i < warmupEnd; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} before warmup end`);
    }
    
    // Values after warmup should not be NaN
    for (let i = warmupEnd; i < result.length; i++) {
        assert(!isNaN(result[i]), `Unexpected NaN at index ${i} after warmup`);
    }
});

test('Linear Regression Angle all NaN input', () => {
    // Test LRA with all NaN values
    const all_nan = new Float64Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.linearreg_angle_js(all_nan, 14);
    }, /All values are NaN/, 'Expected error for all NaN input');
});

test.skip('Linear Regression Angle fast API (no aliasing)', () => {
    // SKIP: wasm-bindgen doesn't export memory directly, cannot access wasm.memory.buffer
    // This test pattern is incompatible with wasm-bindgen's memory management
});

test.skip('Linear Regression Angle fast API (with aliasing)', () => {
    // SKIP: wasm-bindgen doesn't export memory directly, cannot access wasm.memory.buffer
    // This test pattern is incompatible with wasm-bindgen's memory management
});

test('Linear Regression Angle batch single period', async () => {
    // Test batch processing with single period
    const close = new Float64Array(testData.close);
    
    // Batch with single period
    const config = {
        period_range: [14, 14, 0] // Single period
    };
    
    const batch_result = await wasm.linearreg_angle_batch(close, config);
    
    // Should have one row
    assert.strictEqual(batch_result.rows, 1);
    assert.strictEqual(batch_result.cols, close.length);
    assert.strictEqual(batch_result.combos.length, 1);
    assert.strictEqual(batch_result.combos[0].period, 14);
    
    // Single batch result should match regular call
    const single_result = wasm.linearreg_angle_js(close, 14);
    const batch_values = batch_result.values.slice(0, close.length); // First row
    assertArrayClose(
        batch_values,
        single_result,
        1e-10,
        'Batch vs single calculation mismatch'
    );
});

test('Linear Regression Angle batch multiple periods', async () => {
    // Test batch processing with multiple periods - mirrors check_batch_grid_search
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    
    // Multiple periods: 10, 12, 14, 16
    const config = {
        period_range: [10, 16, 2] // period range
    };
    
    const batch_result = await wasm.linearreg_angle_batch(close, config);
    
    // Should have 4 rows (10, 12, 14, 16)
    assert.strictEqual(batch_result.rows, 4);
    assert.strictEqual(batch_result.cols, 100);
    assert.strictEqual(batch_result.combos.length, 4);
    
    // Verify periods are correct
    const expected_periods = [10, 12, 14, 16];
    for (let i = 0; i < expected_periods.length; i++) {
        assert.strictEqual(batch_result.combos[i].period, expected_periods[i], 
            `Period mismatch at index ${i}`);
    }
    
    // Verify each row matches individual calculation
    for (let i = 0; i < expected_periods.length; i++) {
        const period = expected_periods[i];
        const single_result = wasm.linearreg_angle_js(close, period);
        const row_start = i * 100;
        const row_end = row_start + 100;
        const batch_row = batch_result.values.slice(row_start, row_end);
        
        assertArrayClose(
            batch_row,
            single_result,
            1e-10,
            `Period ${period} batch vs single mismatch`
        );
    }
});

test('Linear Regression Angle memory management', () => {
    // Test allocation and deallocation
    const len = 1000;
    
    // Test multiple allocations and frees
    const ptrs = [];
    for (let i = 0; i < 10; i++) {
        ptrs.push(wasm.linearreg_angle_alloc(len));
    }
    
    // Free all allocations
    for (const ptr of ptrs) {
        wasm.linearreg_angle_free(ptr, len);
    }
    
    // Test null pointer handling
    wasm.linearreg_angle_free(0, len); // Should not crash
});

test('Linear Regression Angle batch period static', async () => {
    // Test batch with single static period - mirrors check_batch_period_static
    const close = new Float64Array(testData.close);
    
    // Static period = 14
    const config = {
        period_range: [14, 14, 0] // Static period
    };
    
    const batch_result = await wasm.linearreg_angle_batch(close, config);
    
    // Should have exactly 1 row
    assert.strictEqual(batch_result.rows, 1);
    assert.strictEqual(batch_result.cols, close.length);
    assert.strictEqual(batch_result.combos.length, 1);
    assert.strictEqual(batch_result.combos[0].period, 14);
    
    // Last value should match expected
    const expected_last = -87.77085937059316;
    const last_value = batch_result.values[batch_result.values.length - 1];
    assertClose(last_value, expected_last, 1e-5, 'Static period batch last value mismatch');
});

test('Linear Regression Angle batch metadata', async () => {
    // Test that batch result includes correct parameter combinations
    const close = new Float64Array(20); // Small dataset
    close.fill(100);
    for (let i = 0; i < 20; i++) {
        close[i] = 100 + i; // Simple linear data
    }
    
    const config = {
        period_range: [5, 7, 1] // periods: 5, 6, 7
    };
    
    const result = await wasm.linearreg_angle_batch(close, config);
    
    // Should have 3 combinations
    assert.strictEqual(result.combos.length, 3);
    assert.strictEqual(result.rows, 3);
    
    // Check each combination
    const expected_periods = [5, 6, 7];
    for (let i = 0; i < expected_periods.length; i++) {
        assert.strictEqual(result.combos[i].period, expected_periods[i], 
            `Period mismatch at combo ${i}`);
    }
});

test('Linear Regression Angle batch edge cases', async () => {
    // Test edge cases for batch processing
    const small_data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    // Single value sweep (step=0)
    const single_batch = await wasm.linearreg_angle_batch(small_data, {
        period_range: [5, 5, 0]
    });
    
    assert.strictEqual(single_batch.rows, 1);
    assert.strictEqual(single_batch.cols, 10);
    assert.strictEqual(single_batch.combos.length, 1);
    
    // Step larger than range
    const large_step = await wasm.linearreg_angle_batch(small_data, {
        period_range: [5, 7, 10] // Step > range
    });
    
    // Should only have period=5
    assert.strictEqual(large_step.rows, 1);
    assert.strictEqual(large_step.combos[0].period, 5);
    
    // Empty data should throw
    const empty = new Float64Array([]);
    await assert.rejects(async () => {
        await wasm.linearreg_angle_batch(empty, { period_range: [5, 5, 0] });
    }, /All values are NaN|Empty data/, 'Expected error for empty data');
});

test('Linear Regression Angle batch warmup validation', async () => {
    // Test batch warmup periods for different parameters
    const close = new Float64Array(testData.close.slice(0, 50));
    
    // Multiple periods with different warmup requirements
    const config = {
        period_range: [5, 15, 5] // periods: 5, 10, 15
    };
    
    const batch_result = await wasm.linearreg_angle_batch(close, config);
    
    assert.strictEqual(batch_result.rows, 3);
    
    // Check warmup for each period
    const periods = [5, 10, 15];
    for (let i = 0; i < periods.length; i++) {
        const period = periods[i];
        const row_start = i * 50;
        const row = batch_result.values.slice(row_start, row_start + 50);
        const warmup_end = period - 1;
        
        // Check NaN during warmup
        for (let j = 0; j < warmup_end; j++) {
            assert(isNaN(row[j]), `Period ${period}: Expected NaN at index ${j}`);
        }
        
        // Check non-NaN after warmup
        for (let j = warmup_end; j < Math.min(row.length, warmup_end + 10); j++) {
            assert(!isNaN(row[j]), `Period ${period}: Unexpected NaN at index ${j}`);
        }
    }
});

test('Linear Regression Angle batch error handling', async () => {
    const close = new Float64Array(10);
    close.fill(100);
    
    // Invalid config structure
    await assert.rejects(async () => {
        await wasm.linearreg_angle_batch(close, {
            period_range: [9, 9] // Missing step
        });
    }, /Invalid config/, 'Expected error for invalid config');
    
    // Invalid data type in config
    await assert.rejects(async () => {
        await wasm.linearreg_angle_batch(close, {
            period_range: "invalid"
        });
    }, /Invalid config/, 'Expected error for invalid data type');
    
    // Period exceeding data length
    await assert.rejects(async () => {
        await wasm.linearreg_angle_batch(close, {
            period_range: [15, 15, 0] // Period > data length
        });
    }, /Invalid period|Not enough valid data/, 'Expected error for period exceeding data');
});

test('Linear Regression Angle null pointer handling', () => {
    // Test fast API with null pointers
    assert.throws(() => {
        wasm.linearreg_angle_into(0, 100, 50, 14); // null input pointer
    }, /Null pointer/, 'Expected error for null input pointer');
    
    assert.throws(() => {
        wasm.linearreg_angle_into(100, 0, 50, 14); // null output pointer
    }, /Null pointer/, 'Expected error for null output pointer');
});