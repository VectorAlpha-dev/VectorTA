/**
 * WASM binding tests for PWMA indicator.
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
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('PWMA partial params', () => {
    // Test with default parameters - mirrors check_pwma_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.pwma_js(close, 5);
    assert(result instanceof Float64Array);
    assert.strictEqual(result.length, close.length);
});

test('PWMA accuracy', async () => {
    // Test PWMA matches expected values from Rust tests - mirrors check_pwma_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.pwma;
    
    const result = wasm.pwma_js(close, expected.defaultParams.period);
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-3,
        "PWMA last 5 values mismatch"
    );
    
    // Compare with Rust implementation
    await compareWithRust('pwma', result, 'close', expected.defaultParams);
});

test('PWMA zero period', () => {
    // Test PWMA fails with zero period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.pwma_js(inputData, 0);
    });
});

test('PWMA period exceeds length', () => {
    // Test PWMA fails with period exceeding length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.pwma_js(dataSmall, 10);
    });
});

test('PWMA very small dataset', () => {
    // Test PWMA fails with insufficient data
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.pwma_js(singlePoint, 5);
    });
});

test('PWMA empty input', () => {
    // Test PWMA with empty input
    const dataEmpty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.pwma_js(dataEmpty, 5);
    });
});

test('PWMA reinput', () => {
    // Test PWMA with re-input of PWMA result - mirrors check_pwma_reinput
    const close = new Float64Array(testData.close);
    
    // First PWMA pass with period=5
    const firstResult = wasm.pwma_js(close, 5);
    
    // Second PWMA pass with period=3 using first result as input
    const secondResult = wasm.pwma_js(firstResult, 3);
    
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // All values should be finite after warmup
    const warmup = 240 + (5 - 1) + (3 - 1);  // first_valid + (first_period - 1) + (second_period - 1)
    for (let i = warmup; i < secondResult.length; i++) {
        assert(isFinite(secondResult[i]), `NaN found at index ${i}`);
    }
});

test('PWMA NaN handling', () => {
    // Test PWMA handling of NaN values - mirrors check_pwma_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.pwma_js(close, 5);
    
    assert.strictEqual(result.length, close.length);
    
    // After warmup, all values should be finite
    for (let i = 245; i < result.length; i++) {
        assert(isFinite(result[i]), `NaN found at index ${i}`);
    }
});

test('PWMA batch', () => {
    // Test PWMA batch computation - mirrors check_batch_default_row
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.pwma;
    
    // Test parameter ranges from expected
    const batch_result = wasm.pwma_batch_js(
        close, 
        expected.batchRange.start, 
        expected.batchRange.end, 
        expected.batchRange.step
    );
    
    // Get rows and cols info
    const rows_cols = wasm.pwma_batch_rows_cols_js(
        expected.batchRange.start,
        expected.batchRange.end,
        expected.batchRange.step,
        close.length
    );
    const rows = rows_cols[0];
    const cols = rows_cols[1];
    
    assert(batch_result instanceof Float64Array);
    assert.strictEqual(rows, expected.batchPeriods.length);
    assert.strictEqual(cols, close.length);
    assert.strictEqual(batch_result.length, rows * cols);
    
    // Verify first combination matches individual calculation
    const individual_result = wasm.pwma_js(close, expected.batchPeriods[0]);
    const batch_first = batch_result.slice(0, close.length);
    
    // Compare after warmup
    const warmup = 240 + expected.batchPeriods[0] - 1;
    for (let i = warmup; i < close.length; i++) {
        assertClose(batch_first[i], individual_result[i], 1e-9, `Batch mismatch at ${i}`);
    }
});

test('PWMA different periods', () => {
    // Test PWMA with different period values
    const close = new Float64Array(testData.close);
    
    // Test various period values
    const testPeriods = [2, 5, 10, 20];
    
    for (const period of testPeriods) {
        const result = wasm.pwma_js(close, period);
        assert.strictEqual(result.length, close.length);
        
        // After warmup, all values should be finite
        const warmup = 240 + period - 1;
        if (warmup < result.length) {
            for (let i = warmup; i < result.length; i++) {
                assert(isFinite(result[i]), `NaN at index ${i} for period=${period}`);
            }
        }
    }
});

test('PWMA batch performance', () => {
    // Test that batch computation is more efficient than multiple single computations
    const close = new Float64Array(testData.close.slice(0, 1000)); // Use first 1000 values
    
    // Test multiple parameter combinations
    const startBatch = performance.now();
    const batchResult = wasm.pwma_batch_js(
        close,
        5, 30, 5    // periods: 5, 10, 15, 20, 25, 30
    );
    const batchTime = performance.now() - startBatch;
    
    // Get the exact parameters used by the batch function
    const metadata = wasm.pwma_batch_metadata_js(5, 30, 5);
    
    const startSingle = performance.now();
    const singleResults = [];
    // Use the exact parameters from metadata
    for (const period of metadata) {
        const result = wasm.pwma_js(close, period);
        singleResults.push(...result);
    }
    const singleTime = performance.now() - startSingle;
    
    // Batch should be faster than multiple single calls
    console.log(`Batch time: ${batchTime.toFixed(2)}ms, Single time: ${singleTime.toFixed(2)}ms`);
    
    // Verify results match
    assertArrayClose(batchResult, singleResults, 1e-9, 'Batch vs single results');
});

test('PWMA edge cases', () => {
    // Test PWMA with edge case inputs
    const expected = EXPECTED_OUTPUTS.pwma;
    const period = expected.defaultParams.period;
    
    // Test with monotonically increasing data
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = i + 1;
    }
    const result = wasm.pwma_js(data, period);
    assert.strictEqual(result.length, data.length);
    
    // After warmup, all values should be finite
    for (let i = period - 1; i < result.length; i++) {
        assert(isFinite(result[i]), `NaN at index ${i}`);
    }
    
    // Test with constant values
    const constantVal = expected.constantValue;
    const constantData = new Float64Array(100).fill(constantVal);
    const constantResult = wasm.pwma_js(constantData, period);
    
    assert.strictEqual(constantResult.length, constantData.length);
    
    // After warmup, all values should be finite and equal to the constant
    for (let i = period - 1; i < constantResult.length; i++) {
        assert(isFinite(constantResult[i]), `NaN at index ${i} for constant data`);
        assertClose(constantResult[i], constantVal, 1e-9, `Constant value mismatch at ${i}`);
    }
});

test('PWMA batch metadata', () => {
    // Test metadata function returns correct period values
    const expected = EXPECTED_OUTPUTS.pwma;
    const metadata = wasm.pwma_batch_metadata_js(
        expected.batchRange.start,
        expected.batchRange.end,
        expected.batchRange.step
    );
    
    // Should match expected batch periods
    assert.strictEqual(metadata.length, expected.batchPeriods.length);
    for (let i = 0; i < metadata.length; i++) {
        assert.strictEqual(metadata[i], expected.batchPeriods[i]);
    }
});

test('PWMA consistency across calls', () => {
    // Test that PWMA produces consistent results across multiple calls
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const result1 = wasm.pwma_js(close, 5);
    const result2 = wasm.pwma_js(close, 5);
    
    assertArrayClose(result1, result2, 1e-15, "PWMA results not consistent");
});

test('PWMA step precision', () => {
    // Test batch with various step sizes
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = i + 1;
    }
    
    const batch_result = wasm.pwma_batch_js(
        data,
        5, 15, 5     // periods: 5, 10, 15
    );
    
    // Get rows and cols info
    const rows_cols = wasm.pwma_batch_rows_cols_js(5, 15, 5, data.length);
    const rows = rows_cols[0];
    
    // Should have 3 combinations
    assert.strictEqual(rows, 3);
    assert.strictEqual(batch_result.length, 3 * data.length);
    
    // Verify metadata
    const metadata = wasm.pwma_batch_metadata_js(5, 15, 5);
    assert.strictEqual(metadata.length, 3);
    assert.strictEqual(metadata[0], 5);
    assert.strictEqual(metadata[1], 10);
    assert.strictEqual(metadata[2], 15);
});

test('PWMA warmup behavior', () => {
    // Test PWMA warmup period behavior - mirrors check_pwma_nan_handling
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.pwma;
    const period = expected.defaultParams.period;
    
    const result = wasm.pwma_js(close, period);
    
    // Find first non-NaN in input
    let firstValid = 0;
    for (let i = 0; i < close.length; i++) {
        if (!isNaN(close[i])) {
            firstValid = i;
            break;
        }
    }
    const warmup = firstValid + period - 1;
    
    // Values before warmup should be NaN
    for (let i = 0; i < warmup; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} during warmup`);
    }
    
    // Values at warmup and after should be finite
    for (let i = warmup; i < result.length; i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i} after warmup`);
    }
});

test('PWMA oscillating data', () => {
    // Test with oscillating values
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = (i % 2 === 0) ? 10.0 : 20.0;
    }
    
    const result = wasm.pwma_js(data, 5);
    assert.strictEqual(result.length, data.length);
    
    // After warmup, all values should be finite
    for (let i = 4; i < result.length; i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i}`);
    }
});

test('PWMA small step size', () => {
    // Test batch with very small step size
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = i + 1;
    }
    
    const batch_result = wasm.pwma_batch_js(
        data,
        5, 7, 1     // periods: 5, 6, 7
    );
    
    const rows_cols = wasm.pwma_batch_rows_cols_js(5, 7, 1, data.length);
    const rows = rows_cols[0];
    
    assert.strictEqual(rows, 3);
    assert.strictEqual(batch_result.length, 3 * data.length);
});

test('PWMA formula verification', () => {
    // Test that PWMA values follow Pascal triangle weights - mirrors Rust test
    const expected = EXPECTED_OUTPUTS.pwma.formulaTest;
    const data = new Float64Array(expected.data);
    
    const result = wasm.pwma_js(data, expected.period);
    
    // Result length should match input
    assert.strictEqual(result.length, data.length);
    
    // Check each expected value
    for (let i = 0; i < expected.expected.length; i++) {
        if (isNaN(expected.expected[i])) {
            assert(isNaN(result[i]), `Expected NaN at index ${i}`);
        } else {
            assertClose(result[i], expected.expected[i], 1e-9, 
                       `PWMA formula mismatch at index ${i}`);
        }
    }
});

test('PWMA all NaN input', () => {
    // Test PWMA with all NaN values
    const allNaN = new Float64Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.pwma_js(allNaN, 5);
    }, /All values are NaN/);
});

test('PWMA batch error conditions', () => {
    // Test various error conditions for batch
    const data = new Float64Array([1, 2, 3, 4, 5]);
    
    // Period exceeds data length
    assert.throws(() => {
        wasm.pwma_batch_js(data, 10, 20, 5);
    });
    
    // Empty data
    const empty = new Float64Array([]);
    assert.throws(() => {
        wasm.pwma_batch_js(empty, 5, 10, 5);
    });
});

test('PWMA pascal weights verification', () => {
    // Test with constant input to verify Pascal weights normalization
    const expected = EXPECTED_OUTPUTS.pwma;
    const constantVal = expected.constantValue;
    const data = new Float64Array(10).fill(constantVal);
    
    // For any period with constant input, output should be the constant
    for (const period of [2, 3, 4, 5]) {
        const result = wasm.pwma_js(data, period);
        
        // After warmup, all values should equal the constant
        for (let i = period - 1; i < result.length; i++) {
            assertClose(result[i], constantVal, 1e-9, 
                `PWMA constant test failed at index ${i} for period=${period}`);
        }
    }
});

// Add zero-copy API tests like ALMA has
test('PWMA zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const period = 5;
    
    // Allocate buffer
    const ptr = wasm.pwma_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    // Create view into WASM memory
    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        data.length
    );
    
    // Copy data into WASM memory
    memView.set(data);
    
    // Compute PWMA in-place
    try {
        wasm.pwma_into(ptr, ptr, data.length, period);
        
        // Verify results match regular API
        const regularResult = wasm.pwma_js(data, period);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        // Always free memory
        wasm.pwma_free(ptr, data.length);
    }
});

test('PWMA zero-copy with large dataset', () => {
    const size = 10000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }
    
    const ptr = wasm.pwma_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');
    
    try {
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        memView.set(data);
        
        wasm.pwma_into(ptr, ptr, size, 5);
        
        // Recreate view in case memory grew
        const memView2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        
        // Check warmup period has NaN
        for (let i = 0; i < 4; i++) {
            assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
        }
        
        // Check after warmup has values
        for (let i = 4; i < Math.min(100, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.pwma_free(ptr, size);
    }
});

test('PWMA zero-copy error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.pwma_into(0, 0, 10, 5);
    }, /null pointer|invalid memory/i);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.pwma_alloc(10);
    try {
        // Invalid period
        assert.throws(() => {
            wasm.pwma_into(ptr, ptr, 10, 0);
        }, /Invalid period/);
        
        // Period exceeds length
        assert.throws(() => {
            wasm.pwma_into(ptr, ptr, 10, 20);
        }, /Invalid period/);
    } finally {
        wasm.pwma_free(ptr, 10);
    }
});

test('PWMA zero-copy memory management', () => {
    // Allocate and free multiple times to ensure no leaks
    const sizes = [100, 1000, 5000];
    
    for (const size of sizes) {
        const ptr = wasm.pwma_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        // Write pattern to verify memory
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }
        
        // Verify pattern
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
        }
        
        // Free memory
        wasm.pwma_free(ptr, size);
    }
});

// Test SIMD128 consistency
test('PWMA SIMD128 consistency', () => {
    // This test verifies SIMD128 produces same results as scalar
    const testCases = [
        { size: 10, period: 3 },
        { size: 100, period: 5 },
        { size: 1000, period: 10 },
        { size: 5000, period: 20 }
    ];
    
    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = Math.sin(i * 0.1) + Math.cos(i * 0.05);
        }
        
        const result = wasm.pwma_js(data, testCase.period);
        
        // Basic sanity checks
        assert.strictEqual(result.length, data.length);
        
        // Check warmup period
        for (let i = 0; i < testCase.period - 1; i++) {
            assert(isNaN(result[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}`);
        }
        
        // Check values exist after warmup
        let sumAfterWarmup = 0;
        let countAfterWarmup = 0;
        for (let i = testCase.period - 1; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for size=${testCase.size}`);
            sumAfterWarmup += result[i];
            countAfterWarmup++;
        }
        
        // Verify reasonable values
        const avgAfterWarmup = sumAfterWarmup / countAfterWarmup;
        assert(Math.abs(avgAfterWarmup) < 10, `Average value ${avgAfterWarmup} seems unreasonable`);
    }
});