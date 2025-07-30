/**
 * WASM binding tests for Damiani Volatmeter indicator.
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

test('Damiani Volatmeter partial params', () => {
    // Test with default parameters - mirrors check_damiani_partial_params
    const close = new Float64Array(testData.close);
    
    // Call with all default params
    const result = wasm.damiani_volatmeter_js(close, 13, 20, 40, 100, 1.4);
    assert.strictEqual(result.length, close.length * 2); // vol and anti
    
    // Split result
    const vol = result.slice(0, close.length);
    const anti = result.slice(close.length);
    
    assert.strictEqual(vol.length, close.length);
    assert.strictEqual(anti.length, close.length);
});

test('Damiani Volatmeter accuracy', () => {
    // Test matches expected values from Rust tests - mirrors check_damiani_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS['damiani_volatmeter'];
    
    const result = wasm.damiani_volatmeter_js(close, 13, 20, 40, 100, 1.4);
    
    // Split result
    const vol = result.slice(0, close.length);
    const anti = result.slice(close.length);
    
    // Check last 5 values match expected
    assertArrayClose(
        vol.slice(-5), 
        expected['vol_last_5_values'],
        1e-2,  // Same tolerance as Rust tests
        "Damiani Volatmeter vol last 5 values mismatch"
    );
    assertArrayClose(
        anti.slice(-5), 
        expected['anti_last_5_values'],
        1e-2,  // Same tolerance as Rust tests
        "Damiani Volatmeter anti last 5 values mismatch"
    );
});

test('Damiani Volatmeter zero period', () => {
    // Test fails with zero period - mirrors check_damiani_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0].concat(Array(120).fill(50.0)));
    
    assert.throws(() => {
        wasm.damiani_volatmeter_js(inputData, 0, 20, 40, 100, 1.4);
    }, /Invalid period/);
});

test('Damiani Volatmeter period exceeds length', () => {
    // Test fails when period exceeds data length - mirrors check_damiani_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.damiani_volatmeter_js(dataSmall, 99999, 20, 40, 100, 1.4);
    }, /Invalid period/);
});

test('Damiani Volatmeter very small dataset', () => {
    // Test fails with insufficient data - mirrors check_damiani_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.damiani_volatmeter_js(singlePoint, 9, 9, 9, 9, 1.4);
    }, /Invalid period|Not enough valid data/);
});

test('Damiani Volatmeter empty input', () => {
    // Test fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.damiani_volatmeter_js(empty, 13, 20, 40, 100, 1.4);
    }, /Empty data/);
});

test('Damiani Volatmeter all NaN input', () => {
    // Test fails with all NaN input
    const allNan = new Float64Array(150).fill(NaN);
    
    assert.throws(() => {
        wasm.damiani_volatmeter_js(allNan, 13, 20, 40, 100, 1.4);
    }, /All values are NaN/);
});

test('Damiani Volatmeter NaN handling', () => {
    // Test handles NaN values correctly
    const close = new Float64Array(testData.close);
    
    const result = wasm.damiani_volatmeter_js(close, 13, 20, 40, 100, 1.4);
    const vol = result.slice(0, close.length);
    const anti = result.slice(close.length);
    
    // Check that vol values start appearing after warmup
    // Maximum warmup is max(vis_atr, vis_std, sed_atr, sed_std, 3) = 100
    let firstNonNanVol = -1;
    for (let i = 0; i < vol.length; i++) {
        if (!isNaN(vol[i])) {
            firstNonNanVol = i;
            break;
        }
    }
    assert(firstNonNanVol >= 100, `Vol values appeared too early at index ${firstNonNanVol}`);
    
    // Anti values need both stddev windows filled
    let firstNonNanAnti = -1;
    for (let i = 0; i < anti.length; i++) {
        if (!isNaN(anti[i])) {
            firstNonNanAnti = i;
            break;
        }
    }
    assert(firstNonNanAnti >= 100, `Anti values appeared too early at index ${firstNonNanAnti}`);
});

test('Damiani Volatmeter fast API (into)', () => {
    // Test the fast unsafe API
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Allocate memory for outputs
    const volPtr = wasm.damiani_volatmeter_alloc(len);
    const antiPtr = wasm.damiani_volatmeter_alloc(len);
    
    try {
        // Compute into allocated memory
        wasm.damiani_volatmeter_into(
            close, volPtr, antiPtr, len,
            13, 20, 40, 100, 1.4
        );
        
        // Read results back - need to create typed arrays from pointers
        const memory = wasm.memory;
        const vol = new Float64Array(memory.buffer, volPtr, len);
        const anti = new Float64Array(memory.buffer, antiPtr, len);
        
        // Compare with safe API
        const safeResult = wasm.damiani_volatmeter_js(close, 13, 20, 40, 100, 1.4);
        const safeVol = safeResult.slice(0, len);
        const safeAnti = safeResult.slice(len);
        
        assertArrayClose(vol, safeVol, 1e-12, "Fast API vol mismatch");
        assertArrayClose(anti, safeAnti, 1e-12, "Fast API anti mismatch");
    } finally {
        // Clean up
        wasm.damiani_volatmeter_free(volPtr, len);
        wasm.damiani_volatmeter_free(antiPtr, len);
    }
});

test('Damiani Volatmeter fast API aliasing (vol)', () => {
    // Test aliasing detection when input and vol output are the same
    const data = new Float64Array(testData.close);
    const len = data.length;
    
    // Allocate memory for anti
    const antiPtr = wasm.damiani_volatmeter_alloc(len);
    
    try {
        // Use data array as both input and vol output (aliased)
        wasm.damiani_volatmeter_into(
            data, data, antiPtr, len,
            13, 20, 40, 100, 1.4
        );
        
        // Results should still be correct (implementation should handle aliasing)
        const memory = wasm.memory;
        const anti = new Float64Array(memory.buffer, antiPtr, len);
        
        // Compare with safe API
        const originalData = new Float64Array(testData.close);
        const safeResult = wasm.damiani_volatmeter_js(originalData, 13, 20, 40, 100, 1.4);
        const safeVol = safeResult.slice(0, len);
        const safeAnti = safeResult.slice(len);
        
        assertArrayClose(data, safeVol, 1e-12, "Aliased vol mismatch");
        assertArrayClose(anti, safeAnti, 1e-12, "Aliased anti mismatch");
    } finally {
        wasm.damiani_volatmeter_free(antiPtr, len);
    }
});

test('Damiani Volatmeter fast API aliasing (anti)', () => {
    // Test aliasing detection when input and anti output are the same
    const data = new Float64Array(testData.close);
    const len = data.length;
    
    // Allocate memory for vol
    const volPtr = wasm.damiani_volatmeter_alloc(len);
    
    try {
        // Use data array as both input and anti output (aliased)
        wasm.damiani_volatmeter_into(
            data, volPtr, data, len,
            13, 20, 40, 100, 1.4
        );
        
        // Results should still be correct (implementation should handle aliasing)
        const memory = wasm.memory;
        const vol = new Float64Array(memory.buffer, volPtr, len);
        
        // Compare with safe API
        const originalData = new Float64Array(testData.close);
        const safeResult = wasm.damiani_volatmeter_js(originalData, 13, 20, 40, 100, 1.4);
        const safeVol = safeResult.slice(0, len);
        const safeAnti = safeResult.slice(len);
        
        assertArrayClose(vol, safeVol, 1e-12, "Aliased vol mismatch");
        assertArrayClose(data, safeAnti, 1e-12, "Aliased anti mismatch");
    } finally {
        wasm.damiani_volatmeter_free(volPtr, len);
    }
});

test('Damiani Volatmeter fast API vol/anti same pointer error', () => {
    // Test error when vol and anti pointers are the same
    const data = new Float64Array(testData.close);
    const len = data.length;
    
    // Allocate single buffer for both outputs
    const ptr = wasm.damiani_volatmeter_alloc(len);
    
    try {
        assert.throws(() => {
            wasm.damiani_volatmeter_into(
                data, ptr, ptr, len,  // vol and anti same pointer
                13, 20, 40, 100, 1.4
            );
        }, /vol_ptr and anti_ptr cannot be the same/);
    } finally {
        wasm.damiani_volatmeter_free(ptr, len);
    }
});

test('Damiani Volatmeter batch processing', () => {
    // Test batch processing with parameter sweeps
    const close = new Float64Array(testData.close);
    
    const config = {
        vis_atr_range: [13, 40, 1],  // Default sweep from Rust
        vis_std_range: [20, 40, 1],  // Default sweep from Rust
        sed_atr_range: [40, 40, 0],  // Single value
        sed_std_range: [100, 100, 0], // Single value
        threshold_range: [1.4, 1.4, 0.0] // Single value
    };
    
    const result = wasm.damiani_volatmeter_batch(close, config);
    
    assert(result.vol);
    assert(result.anti);
    assert(result.combos);
    assert.strictEqual(result.cols, close.length);
    
    // Check that we have the expected number of combinations
    // vis_atr: 13 to 40 step 1 = 28 values
    // vis_std: 20 to 40 step 1 = 21 values
    // sed_atr: 40 (single value)
    // sed_std: 100 (single value)
    // threshold: 1.4 (single value)
    // Total: 28 * 21 * 1 * 1 * 1 = 588 combinations
    const expectedRows = 28 * 21;  // 588
    assert.strictEqual(result.rows, expectedRows);
    assert.strictEqual(result.vol.length, expectedRows * close.length);
    assert.strictEqual(result.anti.length, expectedRows * close.length);
    assert.strictEqual(result.combos.length, expectedRows);
});

test('Damiani Volatmeter different thresholds', () => {
    // Test with different threshold values
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    const thresholds = [0.5, 1.0, 1.4, 2.0];
    const results = [];
    
    for (const thresh of thresholds) {
        const result = wasm.damiani_volatmeter_js(close, 13, 20, 40, 100, thresh);
        const vol = result.slice(0, len);
        const anti = result.slice(len);
        results.push({ vol, anti });
    }
    
    // Anti values should change with threshold (since anti = threshold - ratio)
    for (let i = 0; i < thresholds.length - 1; i++) {
        const anti1 = results[i].anti;
        const anti2 = results[i + 1].anti;
        
        // Find indices where both are not NaN
        const validIndices = [];
        for (let j = 0; j < len; j++) {
            if (!isNaN(anti1[j]) && !isNaN(anti2[j])) {
                validIndices.push(j);
            }
        }
        
        if (validIndices.length > 0) {
            // The difference should be approximately the threshold difference
            const expectedDiff = thresholds[i + 1] - thresholds[i];
            for (const idx of validIndices) {
                const diff = anti2[idx] - anti1[idx];
                assertClose(diff, expectedDiff, 1e-10, 1e-10,
                    `Anti values should differ by threshold difference at index ${idx}`);
            }
        }
    }
});

test('Damiani Volatmeter batch fast API (batch_into)', () => {
    // Test the fast batch API
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Small sweep for testing
    const vis_atr_start = 10, vis_atr_end = 12, vis_atr_step = 1;  // 3 values
    const vis_std_start = 18, vis_std_end = 20, vis_std_step = 1;  // 3 values
    const sed_atr_start = 40, sed_atr_end = 40, sed_atr_step = 0;  // 1 value
    const sed_std_start = 100, sed_std_end = 100, sed_std_step = 0; // 1 value
    const threshold_start = 1.4, threshold_end = 1.4, threshold_step = 0.0; // 1 value
    
    const expectedRows = 3 * 3 * 1 * 1 * 1;  // 9 combinations
    
    // Allocate memory for outputs
    const volPtr = wasm.damiani_volatmeter_alloc(expectedRows * len);
    const antiPtr = wasm.damiani_volatmeter_alloc(expectedRows * len);
    
    try {
        const rows = wasm.damiani_volatmeter_batch_into(
            close, volPtr, antiPtr, len,
            vis_atr_start, vis_atr_end, vis_atr_step,
            vis_std_start, vis_std_end, vis_std_step,
            sed_atr_start, sed_atr_end, sed_atr_step,
            sed_std_start, sed_std_end, sed_std_step,
            threshold_start, threshold_end, threshold_step
        );
        
        assert.strictEqual(rows, expectedRows, "Unexpected number of rows");
        
        // Read results back
        const memory = wasm.memory;
        const vol = new Float64Array(memory.buffer, volPtr, expectedRows * len);
        const anti = new Float64Array(memory.buffer, antiPtr, expectedRows * len);
        
        // Test a few values are not NaN after warmup
        for (let row = 0; row < expectedRows; row++) {
            const rowStart = row * len;
            const rowEnd = rowStart + len;
            const volRow = vol.slice(rowStart, rowEnd);
            const antiRow = anti.slice(rowStart, rowEnd);
            
            // Check some values exist after warmup
            const nonNanVol = volRow.slice(120).some(v => !isNaN(v));
            const nonNanAnti = antiRow.slice(120).some(v => !isNaN(v));
            assert(nonNanVol, `Row ${row} vol has no non-NaN values after warmup`);
            assert(nonNanAnti, `Row ${row} anti has no non-NaN values after warmup`);
        }
    } finally {
        wasm.damiani_volatmeter_free(volPtr, expectedRows * len);
        wasm.damiani_volatmeter_free(antiPtr, expectedRows * len);
    }
});

test('Damiani Volatmeter batch fast API aliasing', () => {
    // Test batch_into with aliasing
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Small sweep
    const expectedRows = 9;  // 3 * 3 * 1 * 1 * 1
    
    // Allocate memory for outputs
    const volPtr = wasm.damiani_volatmeter_alloc(expectedRows * len);
    const antiPtr = wasm.damiani_volatmeter_alloc(expectedRows * len);
    
    try {
        // Test error when vol and anti are same
        assert.throws(() => {
            wasm.damiani_volatmeter_batch_into(
                close, volPtr, volPtr, len,  // same pointer for vol and anti
                10, 12, 1, 18, 20, 1, 40, 40, 0, 100, 100, 0, 1.4, 1.4, 0.0
            );
        }, /vol_ptr and anti_ptr cannot be the same/);
        
        // Test aliasing with input (should work)
        const rows = wasm.damiani_volatmeter_batch_into(
            close, close, antiPtr, len,  // input aliased with vol output
            10, 12, 1, 18, 20, 1, 40, 40, 0, 100, 100, 0, 1.4, 1.4, 0.0
        );
        
        assert.strictEqual(rows, expectedRows, "Aliased batch should work");
    } finally {
        wasm.damiani_volatmeter_free(volPtr, expectedRows * len);
        wasm.damiani_volatmeter_free(antiPtr, expectedRows * len);
    }
});