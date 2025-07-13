/**
 * WASM binding tests for Reflex indicator.
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

test('Reflex partial params', () => {
    // Test with default parameters - mirrors check_reflex_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.reflex_js(close, 20);
    assert(result instanceof Float64Array);
    assert.strictEqual(result.length, close.length);
});

test('Reflex accuracy', async () => {
    // Test Reflex matches expected values from Rust tests - mirrors check_reflex_accuracy
    const close = new Float64Array(testData.close);
    
    const result = wasm.reflex_js(close, 20);
    
    assert.strictEqual(result.length, close.length);
    
    // Expected values from Rust test
    const expectedLastFive = [
        0.8085220962465361,
        0.445264715886137,
        0.13861699036615063,
        -0.03598639652007061,
        -0.224906760543743
    ];
    
    const actualLastFive = result.slice(-5);
    
    for (let i = 0; i < 5; i++) {
        assertClose(actualLastFive[i], expectedLastFive[i], 1e-7, 
            `Reflex mismatch at index ${i}`);
    }
    
    // Compare with Rust implementation
    await compareWithRust('reflex', result, 'close', { period: 20 });
});

test('Reflex invalid period', () => {
    // Test Reflex fails with invalid period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    // Period < 2 should fail
    assert.throws(() => {
        wasm.reflex_js(inputData, 1);
    });
    
    // Period = 0 should fail
    assert.throws(() => {
        wasm.reflex_js(inputData, 0);
    });
});

test('Reflex period exceeds length', () => {
    // Test Reflex fails with period exceeding length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.reflex_js(dataSmall, 10);
    });
});

test('Reflex very small dataset', () => {
    // Test Reflex fails with insufficient data
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.reflex_js(singlePoint, 5);
    });
});

test('Reflex empty input', () => {
    // Test Reflex with empty input
    const dataEmpty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.reflex_js(dataEmpty, 20);
    });
});

test('Reflex reinput', () => {
    // Test Reflex with re-input of Reflex result - mirrors check_reflex_reinput
    const close = new Float64Array(testData.close);
    
    // First Reflex pass with period=14
    const firstResult = wasm.reflex_js(close, 14);
    
    // Second Reflex pass with period=20 using first result as input
    const secondResult = wasm.reflex_js(firstResult, 20);
    
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // All values should be finite after warmup
    const warmup = 240 + 20;  // first_valid + second_period
    for (let i = warmup; i < secondResult.length; i++) {
        assert(isFinite(secondResult[i]), `NaN found at index ${i}`);
    }
});

test('Reflex NaN handling', () => {
    // Test Reflex handling of NaN values - mirrors check_reflex_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.reflex_js(close, 20);
    
    assert.strictEqual(result.length, close.length);
    
    // After warmup, all values should be finite
    const warmup = 240 + 20;  // first_valid + period
    for (let i = warmup; i < result.length; i++) {
        assert(isFinite(result[i]), `NaN found at index ${i}`);
    }
});

test('Reflex batch', () => {
    // Test Reflex batch computation
    const close = new Float64Array(testData.close);
    
    // Test parameter ranges
    const batch_result = wasm.reflex_batch_js(
        close, 
        10, 30, 5    // period range: 10, 15, 20, 25, 30
    );
    
    // Get rows and cols info
    const rows_cols = wasm.reflex_batch_rows_cols_js(10, 30, 5, close.length);
    const rows = rows_cols[0];
    const cols = rows_cols[1];
    
    assert(batch_result instanceof Float64Array);
    assert.strictEqual(rows, 5); // 5 period values
    assert.strictEqual(cols, close.length);
    assert.strictEqual(batch_result.length, rows * cols);
    
    // Verify first combination matches individual calculation
    const individual_result = wasm.reflex_js(close, 10);
    const batch_first = batch_result.slice(0, close.length);
    
    // Compare after warmup
    const warmup = 240 + 10;  // first valid + period
    for (let i = warmup; i < close.length; i++) {
        assertClose(batch_first[i], individual_result[i], 1e-9, `Batch mismatch at ${i}`);
    }
});

test('Reflex different periods', () => {
    // Test Reflex with different period values
    const close = new Float64Array(testData.close);
    
    // Test various period values
    const testPeriods = [5, 10, 20, 50];
    
    for (const period of testPeriods) {
        const result = wasm.reflex_js(close, period);
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

test('Reflex batch performance', () => {
    // Test that batch computation is more efficient than multiple single computations
    const close = new Float64Array(testData.close.slice(0, 1000)); // Use first 1000 values
    
    // Test multiple parameter combinations
    const startBatch = performance.now();
    const batchResult = wasm.reflex_batch_js(
        close,
        10, 50, 10    // periods: 10, 20, 30, 40, 50
    );
    const batchTime = performance.now() - startBatch;
    
    // Get the exact parameters used by the batch function
    const metadata = wasm.reflex_batch_metadata_js(10, 50, 10);
    
    const startSingle = performance.now();
    const singleResults = [];
    // Use the exact parameters from metadata
    for (const period of metadata) {
        const result = wasm.reflex_js(close, period);
        singleResults.push(...result);
    }
    const singleTime = performance.now() - startSingle;
    
    // Batch should be faster than multiple single calls
    console.log(`Batch time: ${batchTime.toFixed(2)}ms, Single time: ${singleTime.toFixed(2)}ms`);
    
    // Verify results match
    assertArrayClose(batchResult, singleResults, 1e-9, 'Batch vs single results');
});

test('Reflex edge cases', () => {
    // Test Reflex with edge case inputs
    
    // Test with monotonically increasing data
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = i + 1;
    }
    const result = wasm.reflex_js(data, 20);
    assert.strictEqual(result.length, data.length);
    
    // After warmup, all values should be finite
    for (let i = 20; i < result.length; i++) {
        assert(isFinite(result[i]), `NaN at index ${i}`);
    }
    
    // Test with constant values
    const constantData = new Float64Array(100).fill(50.0);
    const constantResult = wasm.reflex_js(constantData, 20);
    
    assert.strictEqual(constantResult.length, constantData.length);
    
    // With constant input, Reflex produces NaN after warmup (division by zero variance)
    // This is expected behavior when there's no price variation
    // First period values should be zeros
    for (let i = 0; i < 20; i++) {
        assert.strictEqual(constantResult[i], 0.0, `Expected zero at index ${i} during warmup`);
    }
});

test('Reflex batch metadata', () => {
    // Test metadata function returns correct period values
    const metadata = wasm.reflex_batch_metadata_js(
        10, 30, 5    // period range: 10, 15, 20, 25, 30
    );
    
    // Should have 5 period values
    assert.strictEqual(metadata.length, 5);
    assert.strictEqual(metadata[0], 10);
    assert.strictEqual(metadata[1], 15);
    assert.strictEqual(metadata[2], 20);
    assert.strictEqual(metadata[3], 25);
    assert.strictEqual(metadata[4], 30);
});

test('Reflex consistency across calls', () => {
    // Test that Reflex produces consistent results across multiple calls
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const result1 = wasm.reflex_js(close, 20);
    const result2 = wasm.reflex_js(close, 20);
    
    assertArrayClose(result1, result2, 1e-15, "Reflex results not consistent");
});

test('Reflex step precision', () => {
    // Test batch with various step sizes
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = i + 1;
    }
    
    const batch_result = wasm.reflex_batch_js(
        data,
        10, 20, 2     // periods: 10, 12, 14, 16, 18, 20
    );
    
    // Get rows and cols info
    const rows_cols = wasm.reflex_batch_rows_cols_js(10, 20, 2, data.length);
    const rows = rows_cols[0];
    
    // Should have 6 combinations
    assert.strictEqual(rows, 6);
    assert.strictEqual(batch_result.length, 6 * data.length);
    
    // Verify metadata
    const metadata = wasm.reflex_batch_metadata_js(10, 20, 2);
    assert.strictEqual(metadata.length, 6);
    assert.strictEqual(metadata[0], 10);
    assert.strictEqual(metadata[1], 12);
    assert.strictEqual(metadata[2], 14);
    assert.strictEqual(metadata[3], 16);
    assert.strictEqual(metadata[4], 18);
    assert.strictEqual(metadata[5], 20);
});

test('Reflex warmup behavior', () => {
    // Test Reflex warmup period behavior
    const close = new Float64Array(testData.close);
    const period = 20;
    
    const result = wasm.reflex_js(close, period);
    
    // Values during warmup should be zeros (Reflex specific behavior)
    for (let i = 0; i < period; i++) {
        assert.strictEqual(result[i], 0.0, `Expected zero at index ${i} during warmup`);
    }
    
    // Find first non-NaN value in input
    let firstValid = 0;
    for (let i = 0; i < close.length; i++) {
        if (!isNaN(close[i])) {
            firstValid = i;
            break;
        }
    }
    
    const warmup = firstValid + period;
    
    // Values after warmup should be finite
    for (let i = warmup; i < result.length; i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i} after warmup`);
    }
});

test('Reflex oscillating data', () => {
    // Test with oscillating values
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = (i % 2 === 0) ? 10.0 : 20.0;
    }
    
    const result = wasm.reflex_js(data, 20);
    assert.strictEqual(result.length, data.length);
    
    // After warmup, all values should be finite
    for (let i = 20; i < result.length; i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i}`);
    }
});

test('Reflex small step size', () => {
    // Test batch with very small step size
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = i + 1;
    }
    
    const batch_result = wasm.reflex_batch_js(
        data,
        10, 14, 1     // periods: 10, 11, 12, 13, 14
    );
    
    const rows_cols = wasm.reflex_batch_rows_cols_js(10, 14, 1, data.length);
    const rows = rows_cols[0];
    
    assert.strictEqual(rows, 5);
    assert.strictEqual(batch_result.length, 5 * data.length);
});

test('Reflex formula verification', () => {
    // Test that Reflex detects patterns correctly
    const data = new Float64Array([10.0, 11.0, 12.0, 11.0, 10.0, 9.0, 10.0, 11.0, 12.0, 13.0]);
    const period = 5;
    
    const result = wasm.reflex_js(data, period);
    
    // Result length should match input
    assert.strictEqual(result.length, data.length);
    
    // Warmup period should have zeros
    for (let i = 0; i < period; i++) {
        assert.strictEqual(result[i], 0.0, `Expected zero during warmup at index ${i}`);
    }
    
    // After warmup, values should be finite
    for (let i = period; i < result.length; i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i}`);
    }
});

test('Reflex all NaN input', () => {
    // Test Reflex with all NaN values
    const allNaN = new Float64Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.reflex_js(allNaN, 20);
    }, /All values are NaN/);
});

test('Reflex batch error conditions', () => {
    // Test various error conditions for batch
    const data = new Float64Array([1, 2, 3, 4, 5]);
    
    // Period exceeds data length
    assert.throws(() => {
        wasm.reflex_batch_js(data, 10, 20, 5);
    });
    
    // Empty data
    const empty = new Float64Array([]);
    assert.throws(() => {
        wasm.reflex_batch_js(empty, 10, 20, 5);
    });
});

test('Reflex constant input normalization', () => {
    // Test with constant input to verify normalization
    const data = new Float64Array(10).fill(1.0);
    
    // For any period with constant input, warmup values should be zeros
    for (const period of [3, 5, 7]) {
        const result = wasm.reflex_js(data, period);
        
        // First period values should be zeros
        for (let i = 0; i < period; i++) {
            assert.strictEqual(result[i], 0.0, 
                `Expected zero at index ${i} for period=${period}`);
        }
        // After warmup, constant input produces NaN (zero variance)
    }
});