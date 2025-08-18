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

test('Linear Regression Angle reinput', () => {
    // Test LRA applied twice (re-input) - mirrors check_lra_reinput
    const close = new Float64Array(testData.close);
    
    // First pass
    const first_result = wasm.linearreg_angle_js(close, 14);
    assert.strictEqual(first_result.length, close.length);
    
    // Second pass - apply to output of first pass
    const second_result = wasm.linearreg_angle_js(new Float64Array(first_result), 14);
    assert.strictEqual(second_result.length, first_result.length);
});

test('Linear Regression Angle NaN handling', () => {
    // Test LRA handles NaN values correctly
    const close = new Float64Array(testData.close);
    
    const result = wasm.linearreg_angle_js(close, 14);
    assert.strictEqual(result.length, close.length);
    
    // First period-1 values should be NaN (warmup period)
    const warmup = 14 - 1; // period - 1
    for (let i = 0; i < warmup; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} during warmup`);
    }
    
    // After warmup, values should not be NaN (unless input was NaN)
    for (let i = warmup; i < result.length; i++) {
        if (!isNaN(close[i])) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} after warmup`);
        }
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
    // Test batch processing with multiple periods
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
    
    // Verify each row matches individual calculation
    const periods = [10, 12, 14, 16];
    for (let i = 0; i < periods.length; i++) {
        const period = periods[i];
        const single_result = wasm.linearreg_angle_js(close, period);
        const row_start = i * 100;
        const row_end = row_start + 100;
        const batch_row = batch_result.values.slice(row_start, row_end);
        
        assertArrayClose(
            batch_row,
            single_result,
            1e-10,
            `Period ${period} mismatch`
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

test('Linear Regression Angle null pointer handling', () => {
    // Test fast API with null pointers
    assert.throws(() => {
        wasm.linearreg_angle_into(0, 100, 50, 14); // null input pointer
    }, /Null pointer/, 'Expected error for null input pointer');
    
    assert.throws(() => {
        wasm.linearreg_angle_into(100, 0, 50, 14); // null output pointer
    }, /Null pointer/, 'Expected error for null output pointer');
});