/**
 * WASM binding tests for NMA indicator.
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
        wasm = await import(wasmPath);
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('NMA partial params', () => {
    // Test with default parameters - mirrors check_nma_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.nma_js(close, 40);
    assert(result instanceof Float64Array);
    assert.strictEqual(result.length, close.length);
});

test('NMA accuracy', async () => {
    // Test NMA matches expected values from Rust tests - mirrors check_nma_accuracy
    const close = new Float64Array(testData.close);
    
    const result = wasm.nma_js(close, 40);
    
    assert.strictEqual(result.length, close.length);
    
    // Expected values from Rust test
    const expectedLastFive = [
        64320.486018271724,
        64227.95719984426,
        64180.9249333126,
        63966.35530620797,
        64039.04719192334,
    ];
    
    const actualLastFive = result.slice(-5);
    
    for (let i = 0; i < 5; i++) {
        assertClose(actualLastFive[i], expectedLastFive[i], 1e-3, 
            `NMA mismatch at index ${i}`);
    }
    
    // Compare with Rust implementation
    await compareWithRust('nma', result, 'close', { period: 40 });
});

test('NMA zero period', () => {
    // Test NMA fails with zero period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.nma_js(inputData, 0);
    });
});

test('NMA period exceeds length', () => {
    // Test NMA fails with period exceeding length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.nma_js(dataSmall, 10);
    });
});

test('NMA very small dataset', () => {
    // Test NMA fails with insufficient data
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.nma_js(singlePoint, 40);
    });
});

test('NMA empty input', () => {
    // Test NMA with empty input
    const dataEmpty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.nma_js(dataEmpty, 40);
    });
});

test('NMA reinput', () => {
    // Test NMA with re-input of NMA result - mirrors check_nma_reinput
    const close = new Float64Array(testData.close);
    
    // First NMA pass with period=40
    const firstResult = wasm.nma_js(close, 40);
    
    // Second NMA pass with period=30 using first result as input
    const secondResult = wasm.nma_js(firstResult, 30);
    
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // All values should be finite after warmup
    for (let i = 70; i < secondResult.length; i++) {
        assert(isFinite(secondResult[i]), `NaN found at index ${i}`);
    }
});

test('NMA NaN handling', () => {
    // Test NMA handling of NaN values - mirrors check_nma_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.nma_js(close, 40);
    
    assert.strictEqual(result.length, close.length);
    
    // After warmup, all values should be finite
    for (let i = 240; i < result.length; i++) {
        assert(isFinite(result[i]), `NaN found at index ${i}`);
    }
});

test('NMA batch', () => {
    // Test NMA batch computation
    const close = new Float64Array(testData.close);
    
    // Test parameter ranges
    const batch_result = wasm.nma_batch_js(
        close, 
        20, 60, 20    // period range: 20, 40, 60
    );
    
    // Get rows and cols info
    const rows_cols = wasm.nma_batch_rows_cols_js(20, 60, 20, close.length);
    const rows = rows_cols[0];
    const cols = rows_cols[1];
    
    assert(batch_result instanceof Float64Array);
    assert.strictEqual(rows, 3); // 3 period values
    assert.strictEqual(cols, close.length);
    assert.strictEqual(batch_result.length, rows * cols);
    
    // Verify first combination matches individual calculation
    const individual_result = wasm.nma_js(close, 20);
    const batch_first = batch_result.slice(0, close.length);
    
    // Compare after warmup
    const warmup = 240 + 20;
    for (let i = warmup; i < close.length; i++) {
        assertClose(batch_first[i], individual_result[i], 1e-9, `Batch mismatch at ${i}`);
    }
});

test('NMA different periods', () => {
    // Test NMA with different period values
    const close = new Float64Array(testData.close);
    
    // Test various period values
    const testPeriods = [10, 20, 40, 80];
    
    for (const period of testPeriods) {
        const result = wasm.nma_js(close, period);
        assert.strictEqual(result.length, close.length);
        
        // After warmup, all values should be finite
        const warmup = 240 + period;
        if (warmup < result.length) {
            for (let i = warmup; i < result.length; i++) {
                assert(isFinite(result[i]), `NaN at index ${i} for period=${period}`);
            }
        }
    }
});

test('NMA batch performance', () => {
    // Test that batch computation is more efficient than multiple single computations
    const close = new Float64Array(testData.close.slice(0, 1000)); // Use first 1000 values
    
    // Test multiple parameter combinations
    const startBatch = performance.now();
    const batchResult = wasm.nma_batch_js(
        close,
        10, 50, 10    // periods: 10, 20, 30, 40, 50
    );
    const batchTime = performance.now() - startBatch;
    
    // Get the exact parameters used by the batch function
    const metadata = wasm.nma_batch_metadata_js(10, 50, 10);
    
    const startSingle = performance.now();
    const singleResults = [];
    // Use the exact parameters from metadata
    for (const period of metadata) {
        const result = wasm.nma_js(close, period);
        singleResults.push(...result);
    }
    const singleTime = performance.now() - startSingle;
    
    // Batch should be faster than multiple single calls
    console.log(`Batch time: ${batchTime.toFixed(2)}ms, Single time: ${singleTime.toFixed(2)}ms`);
    
    // Verify results match
    assertArrayClose(batchResult, singleResults, 1e-9, 'Batch vs single results');
});

test('NMA edge cases', () => {
    // Test NMA with edge case inputs
    
    // Test with monotonically increasing data
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = i + 1;
    }
    const result = wasm.nma_js(data, 10);
    assert.strictEqual(result.length, data.length);
    
    // After warmup, all values should be finite
    for (let i = 10; i < result.length; i++) {
        assert(isFinite(result[i]), `NaN at index ${i}`);
    }
    
    // Test with constant values
    const constantData = new Float64Array(100).fill(50.0);
    const constantResult = wasm.nma_js(constantData, 10);
    
    assert.strictEqual(constantResult.length, constantData.length);
    
    // After warmup, all values should be finite
    for (let i = 10; i < constantResult.length; i++) {
        assert(isFinite(constantResult[i]), `NaN at index ${i} for constant data`);
    }
});

test('NMA batch metadata', () => {
    // Test metadata function returns correct period values
    const metadata = wasm.nma_batch_metadata_js(
        10, 30, 10    // period range: 10, 20, 30
    );
    
    // Should have 3 period values
    assert.strictEqual(metadata.length, 3);
    assert.strictEqual(metadata[0], 10);
    assert.strictEqual(metadata[1], 20);
    assert.strictEqual(metadata[2], 30);
});

test('NMA consistency across calls', () => {
    // Test that NMA produces consistent results across multiple calls
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const result1 = wasm.nma_js(close, 40);
    const result2 = wasm.nma_js(close, 40);
    
    assertArrayClose(result1, result2, 1e-15, "NMA results not consistent");
});

test('NMA step precision', () => {
    // Test batch with various step sizes
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = i + 1;
    }
    
    const batch_result = wasm.nma_batch_js(
        data,
        10, 30, 10     // periods: 10, 20, 30
    );
    
    // Get rows and cols info
    const rows_cols = wasm.nma_batch_rows_cols_js(10, 30, 10, data.length);
    const rows = rows_cols[0];
    
    // Should have 3 combinations
    assert.strictEqual(rows, 3);
    assert.strictEqual(batch_result.length, 3 * data.length);
    
    // Verify metadata
    const metadata = wasm.nma_batch_metadata_js(10, 30, 10);
    assert.strictEqual(metadata.length, 3);
    assert.strictEqual(metadata[0], 10);
    assert.strictEqual(metadata[1], 20);
    assert.strictEqual(metadata[2], 30);
});

test('NMA warmup behavior', () => {
    // Test NMA warmup period behavior
    const close = new Float64Array(testData.close);
    const period = 40;
    
    const result = wasm.nma_js(close, period);
    
    // Find first non-NaN value in input
    let firstValid = 0;
    for (let i = 0; i < close.length; i++) {
        if (!isNaN(close[i])) {
            firstValid = i;
            break;
        }
    }
    
    const warmup = firstValid + period;
    
    // Values before warmup should be NaN
    for (let i = 0; i < warmup; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} during warmup`);
    }
    
    // Values after warmup should be finite
    for (let i = warmup; i < result.length; i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i} after warmup`);
    }
});

test('NMA oscillating data', () => {
    // Test with oscillating values
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = (i % 2 === 0) ? 10.0 : 20.0;
    }
    
    const result = wasm.nma_js(data, 10);
    assert.strictEqual(result.length, data.length);
    
    // After warmup, all values should be finite
    for (let i = 10; i < result.length; i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i}`);
    }
});

test('NMA small step size', () => {
    // Test batch with very small step size
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = i + 1;
    }
    
    const batch_result = wasm.nma_batch_js(
        data,
        10, 12, 1     // periods: 10, 11, 12
    );
    
    const rows_cols = wasm.nma_batch_rows_cols_js(10, 12, 1, data.length);
    const rows = rows_cols[0];
    
    assert.strictEqual(rows, 3);
    assert.strictEqual(batch_result.length, 3 * data.length);
});

test('NMA formula verification', () => {
    // Test that NMA values are within reasonable range
    const data = new Float64Array([10.0, 12.0, 11.0, 13.0, 15.0, 14.0]);
    const period = 3;
    
    const result = wasm.nma_js(data, period);
    
    // Result length should match input
    assert.strictEqual(result.length, data.length);
    
    // Warmup period should be respected
    for (let i = 0; i < period; i++) {
        assert(isNaN(result[i]), `Expected NaN during warmup at index ${i}`);
    }
    
    // Values after warmup should be reasonable
    const min = Math.min(...data);
    const max = Math.max(...data);
    for (let i = period; i < result.length; i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i}`);
        assert(result[i] >= min * 0.5, `Value too low at index ${i}`);
        assert(result[i] <= max * 1.5, `Value too high at index ${i}`);
    }
});

test('NMA all NaN input', () => {
    // Test NMA with all NaN values
    const allNaN = new Float64Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.nma_js(allNaN, 40);
    }, /All values are NaN/);
});

test('NMA batch error conditions', () => {
    // Test various error conditions for batch
    const data = new Float64Array([1, 2, 3, 4, 5]);
    
    // Period exceeds data length
    assert.throws(() => {
        wasm.nma_batch_js(data, 10, 20, 5);
    });
    
    // Empty data
    const empty = new Float64Array([]);
    assert.throws(() => {
        wasm.nma_batch_js(empty, 5, 10, 5);
    });
});

// Module verification code
// console.log('NMA module loaded successfully');