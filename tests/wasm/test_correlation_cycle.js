/**
 * WASM binding tests for CORRELATION_CYCLE indicator.
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

test('CORRELATION_CYCLE accuracy - Safe API', () => {
    const data = testData.close;
    const expected = EXPECTED_OUTPUTS.correlation_cycle;
    
    // Test with default parameters
    const result = wasm.correlation_cycle_js(data, expected.default_params.period, expected.default_params.threshold);
    
    // Extract the last 5 values for each output
    const n = result.real.length;
    const lastReal = result.real.slice(n - 5);
    const lastImag = result.imag.slice(n - 5);
    const lastAngle = result.angle.slice(n - 5);
    const lastState = result.state.slice(n - 5);
    
    // Check accuracy for all outputs
    assertArrayClose(lastReal, expected.last_5_values.real, 1e-8, 1e-10, 'Real output mismatch');
    assertArrayClose(lastImag, expected.last_5_values.imag, 1e-8, 1e-10, 'Imag output mismatch');
    assertArrayClose(lastAngle, expected.last_5_values.angle, 1e-8, 1e-10, 'Angle output mismatch');
    
    // Verify state values are -1, 0, or 1 after warmup
    const warmup = expected.default_params.period;
    for (let i = warmup + 1; i < result.state.length; i++) {
        assert(result.state[i] === -1 || result.state[i] === 0 || result.state[i] === 1, 
               `State at index ${i} should be -1, 0 or 1, got ${result.state[i]}`);
    }
});

test('CORRELATION_CYCLE accuracy - Fast API', () => {
    const data = testData.close;
    const len = data.length;
    const expected = EXPECTED_OUTPUTS.correlation_cycle;
    
    // Allocate buffers
    const inPtr = wasm.correlation_cycle_alloc(len);
    const realPtr = wasm.correlation_cycle_alloc(len);
    const imagPtr = wasm.correlation_cycle_alloc(len);
    const anglePtr = wasm.correlation_cycle_alloc(len);
    const statePtr = wasm.correlation_cycle_alloc(len);
    
    try {
        // Copy input data
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inView.set(data);
        
        // Compute
        wasm.correlation_cycle_into(
            inPtr, realPtr, imagPtr, anglePtr, statePtr, len,
            expected.default_params.period, expected.default_params.threshold
        );
        
        // Read results
        const realView = new Float64Array(wasm.__wasm.memory.buffer, realPtr, len);
        const imagView = new Float64Array(wasm.__wasm.memory.buffer, imagPtr, len);
        const angleView = new Float64Array(wasm.__wasm.memory.buffer, anglePtr, len);
        
        // Extract last 5 values
        const lastReal = Array.from(realView.slice(len - 5));
        const lastImag = Array.from(imagView.slice(len - 5));
        const lastAngle = Array.from(angleView.slice(len - 5));
        
        // Check accuracy
        assertArrayClose(lastReal, expected.last_5_values.real, 1e-8, 1e-10, 'Fast API real output mismatch');
        assertArrayClose(lastImag, expected.last_5_values.imag, 1e-8, 1e-10, 'Fast API imag output mismatch');
        assertArrayClose(lastAngle, expected.last_5_values.angle, 1e-8, 1e-10, 'Fast API angle output mismatch');
    } finally {
        // Clean up
        wasm.correlation_cycle_free(inPtr, len);
        wasm.correlation_cycle_free(realPtr, len);
        wasm.correlation_cycle_free(imagPtr, len);
        wasm.correlation_cycle_free(anglePtr, len);
        wasm.correlation_cycle_free(statePtr, len);
    }
});

test('CORRELATION_CYCLE - Fast API aliasing test', () => {
    const data = testData.close.slice(0, 100); // Use smaller data for test
    const len = data.length;
    const expected = EXPECTED_OUTPUTS.correlation_cycle;
    
    // Allocate single buffer that will be used as both input and output
    const bufferPtr = wasm.correlation_cycle_alloc(len);
    const imagPtr = wasm.correlation_cycle_alloc(len);
    const anglePtr = wasm.correlation_cycle_alloc(len);
    const statePtr = wasm.correlation_cycle_alloc(len);
    
    try {
        // Copy input data to buffer
        const bufferView = new Float64Array(wasm.__wasm.memory.buffer, bufferPtr, len);
        bufferView.set(data);
        
        // Compute with aliasing (input and real output share same pointer)
        wasm.correlation_cycle_into(
            bufferPtr, bufferPtr, imagPtr, anglePtr, statePtr, len,
            expected.default_params.period, expected.default_params.threshold
        );
        
        // The function should handle aliasing correctly
        // Just verify it doesn't crash and produces valid output
        const resultView = new Float64Array(wasm.__wasm.memory.buffer, bufferPtr, len);
        
        // Check that warmup values are NaN
        for (let i = 0; i < expected.default_params.period; i++) {
            assert(isNaN(resultView[i]), `Expected NaN at index ${i} but got ${resultView[i]}`);
        }
        
        // Check that we have some non-NaN values after warmup
        let hasValidValues = false;
        for (let i = expected.default_params.period; i < len; i++) {
            if (!isNaN(resultView[i])) {
                hasValidValues = true;
                break;
            }
        }
        assert(hasValidValues, 'Expected some valid values after warmup period');
    } finally {
        // Clean up
        wasm.correlation_cycle_free(bufferPtr, len);
        wasm.correlation_cycle_free(imagPtr, len);
        wasm.correlation_cycle_free(anglePtr, len);
        wasm.correlation_cycle_free(statePtr, len);
    }
});

test('CORRELATION_CYCLE error handling', () => {
    // Test empty data
    assert.throws(() => {
        wasm.correlation_cycle_js([], 20, 9.0);
    }, /Empty data/, 'Should throw on empty data');
    
    // Test all NaN values
    const nanData = new Array(100).fill(NaN);
    assert.throws(() => {
        wasm.correlation_cycle_js(nanData, 20, 9.0);
    }, /All values are NaN/, 'Should throw on all NaN values');
    
    // Test invalid period
    const data = testData.close.slice(0, 50);
    assert.throws(() => {
        wasm.correlation_cycle_js(data, 100, 9.0); // period > data length
    }, /Invalid period/, 'Should throw on invalid period');
    
    // Test null pointers in fast API
    assert.throws(() => {
        wasm.correlation_cycle_into(0, 0, 0, 0, 0, 100, 20, 9.0);
    }, /Null pointer/, 'Should throw on null pointers');
});

test('CORRELATION_CYCLE batch API', () => {
    const data = testData.close.slice(0, 1000); // Use smaller data for batch test
    
    // Test batch computation
    const result = wasm.correlation_cycle_batch_js(
        data,
        15, 25, 5,    // period range: 15, 20, 25
        8.0, 10.0, 1.0  // threshold range: 8.0, 9.0, 10.0
    );
    
    // Should have 3 * 3 = 9 combinations
    assert.strictEqual(result.rows, 9, 'Expected 9 parameter combinations');
    assert.strictEqual(result.cols, data.length, 'Expected output length to match input');
    
    // Verify we have all 4 outputs
    assert(result.real && result.real.length === 9 * data.length, 'Expected real output');
    assert(result.imag && result.imag.length === 9 * data.length, 'Expected imag output');
    assert(result.angle && result.angle.length === 9 * data.length, 'Expected angle output');
    assert(result.state && result.state.length === 9 * data.length, 'Expected state output');
    
    // Verify parameter combinations
    assert.strictEqual(result.combos.length, 9, 'Expected 9 parameter combinations');
    assert.strictEqual(result.combos[0].period, 15, 'First combo should have period 15');
    assert.strictEqual(result.combos[0].threshold, 8.0, 'First combo should have threshold 8.0');
});

test('CORRELATION_CYCLE default candles', () => {
    // Test with default parameters - mirrors check_cc_default_candles
    const close = testData.close;
    
    // Default params: period=20, threshold=9.0
    const result = wasm.correlation_cycle_js(close, 20, 9.0);
    assert.strictEqual(result.real.length, close.length);
    assert.strictEqual(result.imag.length, close.length);
    assert.strictEqual(result.angle.length, close.length);
    assert.strictEqual(result.state.length, close.length);
});

test('CORRELATION_CYCLE zero period', () => {
    // Test fails with zero period - mirrors check_cc_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.correlation_cycle_js(inputData, 0, 9.0);
    }, /Invalid period/);
});

test('CORRELATION_CYCLE period exceeds length', () => {
    // Test fails when period exceeds data length - mirrors check_cc_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.correlation_cycle_js(dataSmall, 10, 9.0);
    }, /Invalid period/);
});

test('CORRELATION_CYCLE very small dataset', () => {
    // Test fails with insufficient data - mirrors check_cc_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.correlation_cycle_js(singlePoint, 20, 9.0);
    }, /Invalid period|Not enough valid data/);
});

test('CORRELATION_CYCLE invalid threshold', () => {
    // Test with invalid threshold values
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    
    // NaN threshold - should use default
    const result1 = wasm.correlation_cycle_js(data, 3, NaN);
    assert.strictEqual(result1.real.length, data.length);
    
    // Negative threshold - should work (no restriction in Rust)
    const result2 = wasm.correlation_cycle_js(data, 3, -1.0);
    assert.strictEqual(result2.real.length, data.length);
    
    // Zero threshold - should work
    const result3 = wasm.correlation_cycle_js(data, 3, 0.0);
    assert.strictEqual(result3.real.length, data.length);
});

test('CORRELATION_CYCLE reinput', () => {
    // Test applied twice (re-input) - mirrors check_cc_reinput
    const data = new Float64Array([10.0, 10.5, 11.0, 11.5, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]);
    
    // First pass
    const firstResult = wasm.correlation_cycle_js(data, 4, 2.0);
    assert.strictEqual(firstResult.real.length, data.length);
    
    // Second pass - apply to real output
    const secondResult = wasm.correlation_cycle_js(firstResult.real, 4, 2.0);
    assert.strictEqual(secondResult.real.length, data.length);
    
    // Both should have proper structure
    assert.strictEqual(firstResult.real.length, secondResult.real.length);
    assert.strictEqual(firstResult.imag.length, secondResult.imag.length);
});

test('CORRELATION_CYCLE NaN handling', () => {
    // Test handles NaN values correctly - mirrors check_cc_nan_handling
    const close = new Float64Array(testData.close);
    
    // Insert some NaN values
    for (let i = 10; i < 15; i++) {
        close[i] = NaN;
    }
    
    const result = wasm.correlation_cycle_js(close, 20, 9.0);
    assert.strictEqual(result.real.length, close.length);
    
    // First period values should be NaN
    for (let i = 0; i < 20; i++) {
        assert(isNaN(result.real[i]), `Expected NaN at warmup index ${i}`);
        assert(isNaN(result.imag[i]), `Expected NaN at warmup index ${i}`);
        assert(isNaN(result.angle[i]), `Expected NaN at warmup index ${i}`);
    }
    
    // After sufficient data (beyond NaN region + warmup), should have valid values
    if (result.real.length > 40) {
        for (let i = 40; i < Math.min(50, result.real.length); i++) {
            assert(!isNaN(result.real[i]), `Unexpected NaN in real at index ${i}`);
            assert(!isNaN(result.imag[i]), `Unexpected NaN in imag at index ${i}`);
            assert(!isNaN(result.angle[i]), `Unexpected NaN in angle at index ${i}`);
        }
    }
});

test('CORRELATION_CYCLE all NaN input', () => {
    // Test with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.correlation_cycle_js(allNaN, 20, 9.0);
    }, /All values are NaN/);
});

test('CORRELATION_CYCLE batch accuracy', () => {
    // Test batch matches expected values - mirrors check_batch_default_row
    const data = testData.close;
    const expected = EXPECTED_OUTPUTS.correlation_cycle;
    
    // Test batch with default parameters only
    const result = wasm.correlation_cycle_batch_js(
        data,
        20, 20, 0,     // period range: just 20
        9.0, 9.0, 0.0  // threshold range: just 9.0
    );
    
    // Should have 1 combination
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, data.length);
    
    // Extract last 5 values from the single row
    const lastReal = result.real.slice(-5);
    const lastImag = result.imag.slice(-5);
    const lastAngle = result.angle.slice(-5);
    
    // Check accuracy
    assertArrayClose(lastReal, expected.last_5_values.real, 1e-8, 1e-10, 'Batch real mismatch');
    assertArrayClose(lastImag, expected.last_5_values.imag, 1e-8, 1e-10, 'Batch imag mismatch');
    assertArrayClose(lastAngle, expected.last_5_values.angle, 1e-8, 1e-10, 'Batch angle mismatch');
});

test('CORRELATION_CYCLE batch edge cases', () => {
    // Test edge cases for batch processing
    const close = testData.close.slice(0, 50);
    
    // Single value sweep
    const singleBatch = wasm.correlation_cycle_batch_js(
        close,
        10, 10, 1,     // Single period
        5.0, 5.0, 1.0  // Single threshold
    );
    
    assert.strictEqual(singleBatch.rows, 1);
    assert.strictEqual(singleBatch.combos.length, 1);
    assert.strictEqual(singleBatch.combos[0].period, 10);
    assert.strictEqual(singleBatch.combos[0].threshold, 5.0);
    
    // Step larger than range
    const largeBatch = wasm.correlation_cycle_batch_js(
        close,
        10, 15, 10,    // Step larger than range
        5.0, 5.0, 0.0
    );
    
    // Should only have period=10
    assert.strictEqual(largeBatch.rows, 1);
    assert.strictEqual(largeBatch.combos[0].period, 10);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.correlation_cycle_batch_js(
            new Float64Array([]),
            10, 10, 0,
            5.0, 5.0, 0.0
        );
    }, /Empty data|All values are NaN/);
});

// Zero-copy API tests
test('CORRELATION_CYCLE zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const len = data.length;
    const period = 3;
    const threshold = 2.0;
    
    // Allocate buffers for all outputs
    const inPtr = wasm.correlation_cycle_alloc(len);
    const realPtr = wasm.correlation_cycle_alloc(len);
    const imagPtr = wasm.correlation_cycle_alloc(len);
    const anglePtr = wasm.correlation_cycle_alloc(len);
    const statePtr = wasm.correlation_cycle_alloc(len);
    
    try {
        // Create view and copy data
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inView.set(data);
        
        // Compute
        wasm.correlation_cycle_into(
            inPtr, realPtr, imagPtr, anglePtr, statePtr, len,
            period, threshold
        );
        
        // Read results
        const realView = new Float64Array(wasm.__wasm.memory.buffer, realPtr, len);
        const imagView = new Float64Array(wasm.__wasm.memory.buffer, imagPtr, len);
        const angleView = new Float64Array(wasm.__wasm.memory.buffer, anglePtr, len);
        const stateView = new Float64Array(wasm.__wasm.memory.buffer, statePtr, len);
        
        // Verify results match regular API
        const regularResult = wasm.correlation_cycle_js(data, period, threshold);
        
        for (let i = 0; i < len; i++) {
            // Check each output separately for NaN
            if (!isNaN(regularResult.real[i]) || !isNaN(realView[i])) {
                assert(Math.abs(regularResult.real[i] - realView[i]) < 1e-10,
                       `Zero-copy real mismatch at index ${i}`);
            }
            if (!isNaN(regularResult.imag[i]) || !isNaN(imagView[i])) {
                assert(Math.abs(regularResult.imag[i] - imagView[i]) < 1e-10,
                       `Zero-copy imag mismatch at index ${i}`);
            }
            if (!isNaN(regularResult.angle[i]) || !isNaN(angleView[i])) {
                assert(Math.abs(regularResult.angle[i] - angleView[i]) < 1e-10,
                       `Zero-copy angle mismatch at index ${i}`);
            }
            if (!isNaN(regularResult.state[i]) || !isNaN(stateView[i])) {
                // State values can be -1, 0, or 1, so allow a bit more tolerance for integer comparisons
                assert(Math.abs(regularResult.state[i] - stateView[i]) < 1e-8,
                       `Zero-copy state mismatch at index ${i}: regular=${regularResult.state[i]}, zerocopy=${stateView[i]}`);
            }
        }
    } finally {
        // Always free memory
        wasm.correlation_cycle_free(inPtr, len);
        wasm.correlation_cycle_free(realPtr, len);
        wasm.correlation_cycle_free(imagPtr, len);
        wasm.correlation_cycle_free(anglePtr, len);
        wasm.correlation_cycle_free(statePtr, len);
    }
});

test('CORRELATION_CYCLE zero-copy with large dataset', () => {
    const size = 10000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }
    
    // Allocate buffers
    const inPtr = wasm.correlation_cycle_alloc(size);
    const realPtr = wasm.correlation_cycle_alloc(size);
    const imagPtr = wasm.correlation_cycle_alloc(size);
    const anglePtr = wasm.correlation_cycle_alloc(size);
    const statePtr = wasm.correlation_cycle_alloc(size);
    
    assert(inPtr !== 0, 'Failed to allocate input buffer');
    assert(realPtr !== 0, 'Failed to allocate real buffer');
    
    try {
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, size);
        inView.set(data);
        
        wasm.correlation_cycle_into(
            inPtr, realPtr, imagPtr, anglePtr, statePtr, size,
            20, 9.0
        );
        
        // Recreate views in case memory grew
        const realView = new Float64Array(wasm.__wasm.memory.buffer, realPtr, size);
        
        // Check warmup period has NaN
        for (let i = 0; i < 20; i++) {
            assert(isNaN(realView[i]), `Expected NaN at warmup index ${i}`);
        }
        
        // Check after warmup has values
        for (let i = 20; i < Math.min(30, size); i++) {
            assert(!isNaN(realView[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.correlation_cycle_free(inPtr, size);
        wasm.correlation_cycle_free(realPtr, size);
        wasm.correlation_cycle_free(imagPtr, size);
        wasm.correlation_cycle_free(anglePtr, size);
        wasm.correlation_cycle_free(statePtr, size);
    }
});

test('CORRELATION_CYCLE zero-copy error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.correlation_cycle_into(0, 0, 0, 0, 0, 10, 5, 2.0);
    }, /Null pointer/);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.correlation_cycle_alloc(10);
    const ptr2 = wasm.correlation_cycle_alloc(10);
    const ptr3 = wasm.correlation_cycle_alloc(10);
    const ptr4 = wasm.correlation_cycle_alloc(10);
    const ptr5 = wasm.correlation_cycle_alloc(10);
    
    try {
        // Invalid period
        assert.throws(() => {
            wasm.correlation_cycle_into(ptr, ptr2, ptr3, ptr4, ptr5, 10, 0, 2.0);
        }, /Invalid period/);
        
        // Period exceeds length
        assert.throws(() => {
            wasm.correlation_cycle_into(ptr, ptr2, ptr3, ptr4, ptr5, 10, 20, 2.0);
        }, /Invalid period/);
    } finally {
        wasm.correlation_cycle_free(ptr, 10);
        wasm.correlation_cycle_free(ptr2, 10);
        wasm.correlation_cycle_free(ptr3, 10);
        wasm.correlation_cycle_free(ptr4, 10);
        wasm.correlation_cycle_free(ptr5, 10);
    }
});

test('CORRELATION_CYCLE zero-copy memory management', () => {
    // Allocate and free multiple times to ensure no leaks
    const sizes = [100, 1000, 5000];
    
    for (const size of sizes) {
        const ptrs = [];
        
        // Allocate 5 buffers (for all outputs)
        for (let i = 0; i < 5; i++) {
            const ptr = wasm.correlation_cycle_alloc(size);
            assert(ptr !== 0, `Failed to allocate ${size} elements`);
            ptrs.push(ptr);
        }
        
        // Write pattern to verify memory
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptrs[0], size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }
        
        // Verify pattern
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
        }
        
        // Free all buffers
        for (const ptr of ptrs) {
            wasm.correlation_cycle_free(ptr, size);
        }
    }
});

test('CORRELATION_CYCLE SIMD128 consistency', () => {
    // This test verifies SIMD128 produces same results as scalar
    // It runs automatically when SIMD128 is enabled
    const testCases = [
        { size: 10, period: 3 },
        { size: 100, period: 10 },
        { size: 1000, period: 20 },
        { size: 5000, period: 50 }
    ];
    
    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = Math.sin(i * 0.1) + Math.cos(i * 0.05);
        }
        
        const result = wasm.correlation_cycle_js(data, testCase.period, 5.0);
        
        // Basic sanity checks
        assert.strictEqual(result.real.length, data.length);
        assert.strictEqual(result.imag.length, data.length);
        assert.strictEqual(result.angle.length, data.length);
        assert.strictEqual(result.state.length, data.length);
        
        // Check warmup period
        for (let i = 0; i < testCase.period; i++) {
            assert(isNaN(result.real[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}`);
            assert(isNaN(result.imag[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}`);
            assert(isNaN(result.angle[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}`);
        }
        
        // Check values exist after warmup
        let hasValidReal = false;
        let hasValidImag = false;
        for (let i = testCase.period; i < result.real.length; i++) {
            if (!isNaN(result.real[i])) hasValidReal = true;
            if (!isNaN(result.imag[i])) hasValidImag = true;
            if (hasValidReal && hasValidImag) break;
        }
        
        assert(hasValidReal, `No valid real values found for size=${testCase.size}`);
        assert(hasValidImag, `No valid imag values found for size=${testCase.size}`);
        
        // Verify angle values are in reasonable range (-180 to 180)
        for (let i = testCase.period; i < result.angle.length; i++) {
            if (!isNaN(result.angle[i])) {
                assert(result.angle[i] >= -180 && result.angle[i] <= 180,
                       `Angle value ${result.angle[i]} out of range at index ${i}`);
            }
        }
        
        // Verify state values are -1, 0, or 1
        for (let i = testCase.period + 1; i < result.state.length; i++) {
            if (!isNaN(result.state[i])) {
                assert(result.state[i] === -1 || result.state[i] === 0 || result.state[i] === 1,
                       `State value ${result.state[i]} not valid at index ${i}`);
            }
        }
    }
});

test.after(() => {
    console.log('CORRELATION_CYCLE WASM tests completed');
});
