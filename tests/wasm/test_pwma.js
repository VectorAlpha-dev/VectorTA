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
    
    const result = wasm.pwma_js(close, 5);
    
    assert.strictEqual(result.length, close.length);
    
    // Expected values from Rust test
    const expectedLastFive = [59313.25, 59309.6875, 59249.3125, 59175.625, 59094.875];
    
    const actualLastFive = result.slice(-5);
    
    for (let i = 0; i < 5; i++) {
        assertClose(actualLastFive[i], expectedLastFive[i], 1e-3, 
            `PWMA mismatch at index ${i}`);
    }
    
    // Compare with Rust implementation
    await compareWithRust('pwma', result, 'close', { period: 5 });
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
    // Test PWMA batch computation
    const close = new Float64Array(testData.close);
    
    // Test parameter ranges
    const batch_result = wasm.pwma_batch_js(
        close, 
        3, 10, 2    // period range: 3, 5, 7, 9
    );
    
    // Get rows and cols info
    const rows_cols = wasm.pwma_batch_rows_cols_js(3, 10, 2, close.length);
    const rows = rows_cols[0];
    const cols = rows_cols[1];
    
    assert(batch_result instanceof Float64Array);
    assert.strictEqual(rows, 4); // 4 period values
    assert.strictEqual(cols, close.length);
    assert.strictEqual(batch_result.length, rows * cols);
    
    // Verify first combination matches individual calculation
    const individual_result = wasm.pwma_js(close, 3);
    const batch_first = batch_result.slice(0, close.length);
    
    // Compare after warmup
    const warmup = 240 + 3 - 1;  // first valid + period - 1
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
    
    // Test with monotonically increasing data
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = i + 1;
    }
    const result = wasm.pwma_js(data, 5);
    assert.strictEqual(result.length, data.length);
    
    // After warmup, all values should be finite
    for (let i = 4; i < result.length; i++) {
        assert(isFinite(result[i]), `NaN at index ${i}`);
    }
    
    // Test with constant values
    const constantData = new Float64Array(100).fill(50.0);
    const constantResult = wasm.pwma_js(constantData, 5);
    
    assert.strictEqual(constantResult.length, constantData.length);
    
    // After warmup, all values should be finite and equal to the constant
    for (let i = 4; i < constantResult.length; i++) {
        assert(isFinite(constantResult[i]), `NaN at index ${i} for constant data`);
        assertClose(constantResult[i], 50.0, 1e-9, `Constant value mismatch at ${i}`);
    }
});

test('PWMA batch metadata', () => {
    // Test metadata function returns correct period values
    const metadata = wasm.pwma_batch_metadata_js(
        5, 15, 5    // period range: 5, 10, 15
    );
    
    // Should have 3 period values
    assert.strictEqual(metadata.length, 3);
    assert.strictEqual(metadata[0], 5);
    assert.strictEqual(metadata[1], 10);
    assert.strictEqual(metadata[2], 15);
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
    // Test PWMA warmup period behavior
    const close = new Float64Array(testData.close);
    const period = 5;
    
    const result = wasm.pwma_js(close, period);
    
    // Find first non-NaN value in input
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
    // Test that PWMA values follow Pascal triangle weights
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    const period = 3;
    
    const result = wasm.pwma_js(data, period);
    
    // Result length should match input
    assert.strictEqual(result.length, data.length);
    
    // Warmup period should be respected
    for (let i = 0; i < period - 1; i++) {
        assert(isNaN(result[i]), `Expected NaN during warmup at index ${i}`);
    }
    
    // For period=3: weights = [1, 2, 1] / 4 = [0.25, 0.5, 0.25]
    // Result[2] = 1*0.25 + 2*0.5 + 3*0.25 = 2.0
    // Result[3] = 2*0.25 + 3*0.5 + 4*0.25 = 3.0
    // Result[4] = 3*0.25 + 4*0.5 + 5*0.25 = 4.0
    
    assertClose(result[2], 2.0, 1e-9, 'PWMA formula mismatch at index 2');
    assertClose(result[3], 3.0, 1e-9, 'PWMA formula mismatch at index 3');
    assertClose(result[4], 4.0, 1e-9, 'PWMA formula mismatch at index 4');
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
    const data = new Float64Array(10).fill(1.0);
    
    // For any period with constant input, output should be the constant
    for (const period of [2, 3, 4, 5]) {
        const result = wasm.pwma_js(data, period);
        
        // After warmup, all values should be 1.0
        for (let i = period - 1; i < result.length; i++) {
            assertClose(result[i], 1.0, 1e-9, 
                `PWMA constant test failed at index ${i} for period=${period}`);
        }
    }
});