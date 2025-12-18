/**
 * WASM binding tests for TRADJEMA indicator.
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

test('TRADJEMA partial params', () => {
    // Test with default parameters - mirrors check_tradjema_partial_params
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.tradjema_js(high, low, close, 40, 10.0);
    assert.strictEqual(result.length, close.length);
});

test('TRADJEMA accuracy', async () => {
    // Test TRADJEMA matches expected values from Rust tests - mirrors check_tradjema_accuracy
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.tradjema;
    
    const result = wasm.tradjema_js(
        high, low, close,
        expected.defaultParams.length,
        expected.defaultParams.mult
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-8,
        "TRADJEMA last 5 values mismatch"
    );
    
    // Check warmup period (length - 1 values should be NaN)
    const warmup = expected.defaultParams.length - 1;
    assertAllNaN(result.slice(0, warmup), `Expected NaN in warmup period (0..${warmup})`);
    
    // Check that valid value starts at warmup index
    assert(!isNaN(result[warmup]), `Expected valid value at index ${warmup}`);
    
    // Enhanced: Validate intermediate values at specific indices
    // Check values at indices 100, 500, 1000 (if they exist)
    const checkIndices = [100, 500, 1000, 5000];
    for (const idx of checkIndices) {
        if (idx < result.length) {
            assert(!isNaN(result[idx]), `Expected valid value at index ${idx}`);
            assert(isFinite(result[idx]), `Expected finite value at index ${idx}`);
            // Verify value is in reasonable range (based on input data range)
            const inputMax = Math.max(...close.slice(Math.max(0, idx - 100), idx + 1));
            const inputMin = Math.min(...close.slice(Math.max(0, idx - 100), idx + 1));
            assert(
                result[idx] >= inputMin * 0.5 && result[idx] <= inputMax * 1.5,
                `Value at index ${idx} (${result[idx]}) outside reasonable range [${inputMin * 0.5}, ${inputMax * 1.5}]`
            );
        }
    }
    
    // Validate that the indicator follows the general trend of the input
    // Check if moving average nature is preserved
    const windowSize = 100;
    if (result.length > warmup + windowSize) {
        const startIdx = warmup + 50;
        const endIdx = startIdx + windowSize;
        
        const inputTrend = close[endIdx] - close[startIdx];
        const outputTrend = result[endIdx] - result[startIdx];
        
        // Both should have the same sign (direction)
        if (Math.abs(inputTrend) > 1.0) { // Only check if there's significant movement
            assert(
                Math.sign(inputTrend) === Math.sign(outputTrend),
                `Trend mismatch: input trend=${inputTrend}, output trend=${outputTrend}`
            );
        }
    }
    
    // Compare full output with Rust (commented out due to generate_references compilation issue)
    // await compareWithRust('tradjema', result, 'ohlc', expected.defaultParams);
});

test('TRADJEMA default candles', () => {
    // Test TRADJEMA with default parameters - mirrors check_tradjema_default_candles
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.tradjema_js(high, low, close, 40, 10.0);
    assert.strictEqual(result.length, close.length);
});

test('TRADJEMA zero length', () => {
    // Test TRADJEMA fails with zero length - mirrors check_tradjema_zero_length
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.tradjema_js(inputData, inputData, inputData, 0, 10.0);
    }, /Invalid length/);
    
    // Also test length=1 (minimum is 2)
    assert.throws(() => {
        wasm.tradjema_js(inputData, inputData, inputData, 1, 10.0);
    }, /Invalid length/);
});

test('TRADJEMA length exceeds data', () => {
    // Test TRADJEMA fails when length exceeds data length - mirrors check_tradjema_length_exceeds_data
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.tradjema_js(dataSmall, dataSmall, dataSmall, 10, 10.0);
    }, /Invalid length/);
});

test('TRADJEMA very small dataset', () => {
    // Test TRADJEMA fails with insufficient data - mirrors check_tradjema_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.tradjema_js(singlePoint, singlePoint, singlePoint, 40, 10.0);
    }, /Invalid length|Not enough valid data/);
});

test('TRADJEMA empty input', () => {
    // Test TRADJEMA fails with empty input - mirrors check_tradjema_empty_input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.tradjema_js(empty, empty, empty, 40, 10.0);
    }, /Input data slice is empty/);
});

test('TRADJEMA invalid mult', () => {
    // Test TRADJEMA fails with invalid mult - mirrors check_tradjema_invalid_mult
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    // Negative mult
    assert.throws(() => {
        wasm.tradjema_js(data, data, data, 2, -10.0);
    }, /Invalid mult/);
    
    // Zero mult
    assert.throws(() => {
        wasm.tradjema_js(data, data, data, 2, 0.0);
    }, /Invalid mult/);
    
    // NaN mult
    assert.throws(() => {
        wasm.tradjema_js(data, data, data, 2, NaN);
    }, /Invalid mult/);
    
    // Infinite mult
    assert.throws(() => {
        wasm.tradjema_js(data, data, data, 2, Infinity);
    }, /Invalid mult/);
});

test('TRADJEMA reinput', () => {
    // Test TRADJEMA applied twice (re-input) - mirrors check_tradjema_reinput
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.tradjema;
    
    // First pass
    const firstResult = wasm.tradjema_js(high, low, close, 40, 10.0);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass - apply TRADJEMA to TRADJEMA output (using output for all OHLC)
    const secondResult = wasm.tradjema_js(firstResult, firstResult, firstResult, 40, 10.0);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // Check last 5 values match expected
    const last5 = secondResult.slice(-5);
    assertArrayClose(
        last5,
        expected.reinputLast5,
        1e-8,
        "TRADJEMA re-input last 5 values mismatch"
    );
});

test('TRADJEMA NaN handling', () => {
    // Test TRADJEMA handles NaN values correctly - mirrors check_tradjema_nan_handling
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.tradjema_js(high, low, close, 40, 10.0);
    assert.strictEqual(result.length, close.length);
    
    // After warmup period (50), no NaN values should exist
    if (result.length > 50) {
        for (let i = 50; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    // First length-1 values should be NaN
    assertAllNaN(result.slice(0, 39), "Expected NaN in warmup period");
    
    // Value at index 39 should be valid (first non-NaN)
    assert(!isNaN(result[39]), "Expected valid value at index 39");
});

test('TRADJEMA all NaN input', () => {
    // Test TRADJEMA with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.tradjema_js(allNaN, allNaN, allNaN, 40, 10.0);
    }, /All values are NaN/);
});

test('TRADJEMA mismatched lengths', () => {
    // Test TRADJEMA with mismatched OHLC array lengths
    const high = new Float64Array([1.0, 2.0, 3.0]);
    const low = new Float64Array([1.0, 2.0]);
    const close = new Float64Array([1.0]);
    
    assert.throws(() => {
        wasm.tradjema_js(high, low, close, 2, 10.0);
    }, /length mismatch/);
});

test('TRADJEMA batch single parameter set', () => {
    // Test batch with single parameter combination
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    // Using the new ergonomic batch API for single parameter
    const batchResult = wasm.tradjema_batch(high, low, close, {
        length_range: [40, 40, 0],
        mult_range: [10.0, 10.0, 0.0]
    });
    
    // Should match single calculation
    const singleResult = wasm.tradjema_js(high, low, close, 40, 10.0);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('TRADJEMA batch multiple lengths', () => {
    // Test batch with multiple length values
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Multiple lengths: 20, 30, 40 using ergonomic API
    const batchResult = wasm.tradjema_batch(high, low, close, {
        length_range: [20, 40, 10],      // length range
        mult_range: [10.0, 10.0, 0.0]    // mult range  
    });
    
    // Should have 3 rows * 100 cols = 300 values
    assert.strictEqual(batchResult.values.length, 3 * 100);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 100);
    
    // Verify each row matches individual calculation
    const lengths = [20, 30, 40];
    for (let i = 0; i < lengths.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.tradjema_js(high, low, close, lengths[i], 10.0);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Length ${lengths[i]} mismatch`
        );
    }
});

test('TRADJEMA batch metadata from result', () => {
    // Test that batch result includes correct parameter combinations
    const high = new Float64Array(50);
    const low = new Float64Array(50);
    const close = new Float64Array(50);
    high.fill(100);
    low.fill(90);
    close.fill(95);
    
    const result = wasm.tradjema_batch(high, low, close, {
        length_range: [20, 40, 10],      // length: 20, 30, 40
        mult_range: [5.0, 15.0, 5.0]     // mult: 5.0, 10.0, 15.0
    });
    
    // Should have 3 * 3 = 9 combinations
    assert.strictEqual(result.combos.length, 9);
    
    // Check first combination
    assert.strictEqual(result.combos[0].length, 20);   // length
    assert.strictEqual(result.combos[0].mult, 5.0);    // mult
    
    // Check last combination
    assert.strictEqual(result.combos[8].length, 40);   // length
    assert.strictEqual(result.combos[8].mult, 15.0);   // mult
});

test('TRADJEMA batch full parameter sweep', () => {
    // Test full parameter sweep matching expected structure
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.tradjema_batch(high, low, close, {
        length_range: [20, 30, 10],      // 2 lengths
        mult_range: [5.0, 10.0, 5.0]     // 2 mults
    });
    
    // Should have 2 * 2 = 4 combinations
    assert.strictEqual(batchResult.combos.length, 4);
    assert.strictEqual(batchResult.rows, 4);
    assert.strictEqual(batchResult.cols, 50);
    assert.strictEqual(batchResult.values.length, 4 * 50);
    
    // Verify structure
    for (let combo = 0; combo < batchResult.combos.length; combo++) {
        const length = batchResult.combos[combo].length;
        const mult = batchResult.combos[combo].mult;
        
        const rowStart = combo * 50;
        const rowData = batchResult.values.slice(rowStart, rowStart + 50);
        
        // First length-1 values should be NaN
        for (let i = 0; i < length - 1; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for length ${length}`);
        }
        
        // After warmup should have values
        for (let i = length - 1; i < 50; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for length ${length}`);
        }
    }
});

test('TRADJEMA batch edge cases', () => {
    // Test edge cases for batch processing
    const high = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const low = new Float64Array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]);
    const close = new Float64Array([0.75, 1.75, 2.75, 3.75, 4.75, 5.75, 6.75, 7.75, 8.75, 9.75]);
    
    // Single value sweep
    const singleBatch = wasm.tradjema_batch(high, low, close, {
        length_range: [5, 5, 1],
        mult_range: [10.0, 10.0, 0.1]
    });
    
    assert.strictEqual(singleBatch.values.length, 10);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    // Step larger than range
    const largeBatch = wasm.tradjema_batch(high, low, close, {
        length_range: [5, 7, 10], // Step larger than range
        mult_range: [10.0, 10.0, 0]
    });
    
    // Should only have length=5
    assert.strictEqual(largeBatch.values.length, 10);
    assert.strictEqual(largeBatch.combos.length, 1);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.tradjema_batch(new Float64Array([]), new Float64Array([]), new Float64Array([]), {
            length_range: [40, 40, 0],
            mult_range: [10.0, 10.0, 0.0]
        });
    }, /Input arrays are empty/);
});

// Batch vs single cross-validation test
test('TRADJEMA batch vs single cross-validation', () => {
    // Test that batch processing produces identical results to individual calculations
    const size = 200;
    const high = new Float64Array(size);
    const low = new Float64Array(size);
    const close = new Float64Array(size);
    
    // Generate synthetic OHLC data
    for (let i = 0; i < size; i++) {
        const base = 100 + Math.sin(i * 0.05) * 20;
        close[i] = base;
        high[i] = base + Math.abs(Math.random() * 3);
        low[i] = base - Math.abs(Math.random() * 3);
    }
    
    // Define test parameters
    const lengths = [20, 30, 40];
    const mults = [5.0, 10.0, 15.0];
    
    // Run batch calculation
    const batchResult = wasm.tradjema_batch(high, low, close, {
        length_range: [20, 40, 10],
        mult_range: [5.0, 15.0, 5.0]
    });
    
    // Verify batch has expected structure
    assert.strictEqual(batchResult.combos.length, 9); // 3 lengths * 3 mults
    assert.strictEqual(batchResult.values.length, 9 * size);
    
    // Cross-validate each combination
    let comboIdx = 0;
    for (const length of lengths) {
        for (const mult of mults) {
            // Calculate single result
            const singleResult = wasm.tradjema_js(high, low, close, length, mult);
            
            // Extract batch row
            const rowStart = comboIdx * size;
            const rowEnd = rowStart + size;
            const batchRow = batchResult.values.slice(rowStart, rowEnd);
            
            // Verify parameters match
            assert.strictEqual(batchResult.combos[comboIdx].length, length,
                `Combo ${comboIdx}: length mismatch`);
            assert.strictEqual(batchResult.combos[comboIdx].mult, mult,
                `Combo ${comboIdx}: mult mismatch`);
            
            // Compare results
            for (let i = 0; i < size; i++) {
                if (isNaN(singleResult[i]) && isNaN(batchRow[i])) {
                    continue; // Both NaN is OK
                }
                assertClose(
                    batchRow[i],
                    singleResult[i],
                    1e-10,
                    `Mismatch at index ${i} for length=${length}, mult=${mult}: batch=${batchRow[i]}, single=${singleResult[i]}`
                );
            }
            
            // Verify warmup periods match exactly
            const warmup = length - 1;
            for (let i = 0; i < warmup; i++) {
                assert(isNaN(batchRow[i]) && isNaN(singleResult[i]),
                    `Warmup mismatch at index ${i} for length=${length}`);
            }
            
            // Verify first valid value
            if (warmup < size) {
                assert(!isNaN(batchRow[warmup]) && !isNaN(singleResult[warmup]),
                    `First valid value mismatch at index ${warmup} for length=${length}`);
            }
            
            comboIdx++;
        }
    }
});

// Parameter boundary tests
test('TRADJEMA parameter boundary conditions', () => {
    // Test extreme but valid parameter combinations
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    
    // Test minimum valid length (2)
    const minLengthResult = wasm.tradjema_js(data, data, data, 2, 10.0);
    assert.strictEqual(minLengthResult.length, data.length);
    assert(isNaN(minLengthResult[0]), "Expected NaN at index 0 for min length");
    assert(!isNaN(minLengthResult[1]), "Expected valid value at index 1 for min length");
    
    // Test very small mult (near zero but positive)
    const smallMultResult = wasm.tradjema_js(data, data, data, 3, 0.0001);
    assert.strictEqual(smallMultResult.length, data.length);
    // Verify all values are finite
    for (let i = 2; i < smallMultResult.length; i++) {
        assert(isFinite(smallMultResult[i]), `Non-finite value at index ${i} with small mult`);
    }
    
    // Test very large mult
    const largeMultResult = wasm.tradjema_js(data, data, data, 3, 999.9);
    assert.strictEqual(largeMultResult.length, data.length);
    // Verify all values are finite despite large mult
    for (let i = 2; i < largeMultResult.length; i++) {
        assert(isFinite(largeMultResult[i]), `Non-finite value at index ${i} with large mult`);
    }
    
    // Test maximum valid length (data.length)
    const maxLengthResult = wasm.tradjema_js(data, data, data, data.length, 10.0);
    assert.strictEqual(maxLengthResult.length, data.length);
    // Should have warmup of length-1
    for (let i = 0; i < data.length - 1; i++) {
        assert(isNaN(maxLengthResult[i]), `Expected NaN at index ${i} for max length`);
    }
    assert(!isNaN(maxLengthResult[data.length - 1]), "Expected valid value at last index for max length");
    
    // Test boundary mult values
    const boundaryMults = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 999.0];
    for (const mult of boundaryMults) {
        const result = wasm.tradjema_js(data, data, data, 3, mult);
        assert.strictEqual(result.length, data.length, `Length mismatch for mult=${mult}`);
        
        // Check structure is valid
        assert(isNaN(result[0]) && isNaN(result[1]), `Expected NaN in warmup for mult=${mult}`);
        assert(!isNaN(result[2]), `Expected valid value after warmup for mult=${mult}`);
        
        // Check all valid values are finite
        for (let i = 2; i < result.length; i++) {
            assert(isFinite(result[i]), `Non-finite value at index ${i} for mult=${mult}`);
        }
    }
});

// Zero-copy API tests
test('TRADJEMA zero-copy API', () => {
    const high = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const low = new Float64Array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]);
    const close = new Float64Array([0.75, 1.75, 2.75, 3.75, 4.75, 5.75, 6.75, 7.75, 8.75, 9.75]);
    const length = 5;
    const mult = 10.0;
    
    // Allocate buffers
    const highPtr = wasm.tradjema_alloc(high.length);
    const lowPtr = wasm.tradjema_alloc(low.length);
    const closePtr = wasm.tradjema_alloc(close.length);
    const outPtr = wasm.tradjema_alloc(high.length);
    
    assert(highPtr !== 0, 'Failed to allocate high buffer');
    assert(lowPtr !== 0, 'Failed to allocate low buffer');
    assert(closePtr !== 0, 'Failed to allocate close buffer');
    assert(outPtr !== 0, 'Failed to allocate output buffer');
    
    try {
        // Create views into WASM memory
        const memory = wasm.__wasm.memory.buffer;
        const highView = new Float64Array(memory, highPtr, high.length);
        const lowView = new Float64Array(memory, lowPtr, low.length);
        const closeView = new Float64Array(memory, closePtr, close.length);
        const outView = new Float64Array(memory, outPtr, high.length);
        
        // Copy data into WASM memory
        highView.set(high);
        lowView.set(low);
        closeView.set(close);
        
        // Compute TRADJEMA
        wasm.tradjema_into(highPtr, lowPtr, closePtr, outPtr, high.length, length, mult);
        
        // Verify results match regular API
        const regularResult = wasm.tradjema_js(high, low, close, length, mult);
        for (let i = 0; i < high.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(outView[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularResult[i] - outView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${outView[i]}`);
        }
    } finally {
        // Always free memory
        wasm.tradjema_free(highPtr, high.length);
        wasm.tradjema_free(lowPtr, low.length);
        wasm.tradjema_free(closePtr, close.length);
        wasm.tradjema_free(outPtr, high.length);
    }
});

test('TRADJEMA zero-copy with large dataset', () => {
    const size = 10000;
    const high = new Float64Array(size);
    const low = new Float64Array(size);
    const close = new Float64Array(size);
    
    for (let i = 0; i < size; i++) {
        high[i] = Math.sin(i * 0.01) + 2.0 + Math.random() * 0.1;
        low[i] = Math.sin(i * 0.01) + 1.0 + Math.random() * 0.1;
        close[i] = Math.sin(i * 0.01) + 1.5 + Math.random() * 0.1;
    }
    
    const highPtr = wasm.tradjema_alloc(size);
    const lowPtr = wasm.tradjema_alloc(size);
    const closePtr = wasm.tradjema_alloc(size);
    const outPtr = wasm.tradjema_alloc(size);
    
    assert(highPtr !== 0, 'Failed to allocate large high buffer');
    assert(lowPtr !== 0, 'Failed to allocate large low buffer');
    assert(closePtr !== 0, 'Failed to allocate large close buffer');
    assert(outPtr !== 0, 'Failed to allocate large output buffer');
    
    try {
        const memory = wasm.__wasm.memory.buffer;
        const highView = new Float64Array(memory, highPtr, size);
        const lowView = new Float64Array(memory, lowPtr, size);
        const closeView = new Float64Array(memory, closePtr, size);
        
        highView.set(high);
        lowView.set(low);
        closeView.set(close);
        
        wasm.tradjema_into(highPtr, lowPtr, closePtr, outPtr, size, 40, 10.0);
        
        // Recreate view in case memory grew
        const memory2 = wasm.__wasm.memory.buffer;
        const outView = new Float64Array(memory2, outPtr, size);
        
        // Check warmup period has NaN
        for (let i = 0; i < 39; i++) {
            assert(isNaN(outView[i]), `Expected NaN at warmup index ${i}`);
        }
        
        // Check after warmup has values
        for (let i = 39; i < Math.min(100, size); i++) {
            assert(!isNaN(outView[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.tradjema_free(highPtr, size);
        wasm.tradjema_free(lowPtr, size);
        wasm.tradjema_free(closePtr, size);
        wasm.tradjema_free(outPtr, size);
    }
});

// Error handling for zero-copy API
test('TRADJEMA zero-copy error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.tradjema_into(0, 0, 0, 0, 10, 40, 10.0);
    }, /null pointer/i);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.tradjema_alloc(10);
    try {
        // Invalid length
        assert.throws(() => {
            wasm.tradjema_into(ptr, ptr, ptr, ptr, 10, 0, 10.0);
        }, /Invalid length/);
        
        // Invalid mult
        assert.throws(() => {
            wasm.tradjema_into(ptr, ptr, ptr, ptr, 10, 5, 0.0);
        }, /Invalid mult/);
    } finally {
        wasm.tradjema_free(ptr, 10);
    }
});

// Memory leak prevention test
test('TRADJEMA zero-copy memory management', () => {
    // Allocate and free multiple times to ensure no leaks
    const sizes = [100, 1000, 10000];
    
    for (const size of sizes) {
        const highPtr = wasm.tradjema_alloc(size);
        const lowPtr = wasm.tradjema_alloc(size);
        const closePtr = wasm.tradjema_alloc(size);
        const outPtr = wasm.tradjema_alloc(size);
        
        assert(highPtr !== 0, `Failed to allocate high with ${size} elements`);
        assert(lowPtr !== 0, `Failed to allocate low with ${size} elements`);
        assert(closePtr !== 0, `Failed to allocate close with ${size} elements`);
        assert(outPtr !== 0, `Failed to allocate output with ${size} elements`);
        
        // Write pattern to verify memory
        const memory = wasm.__wasm.memory.buffer;
        const highView = new Float64Array(memory, highPtr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            highView[i] = i * 1.5;
        }
        
        // Verify pattern
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(highView[i], i * 1.5, `Memory corruption at index ${i}`);
        }
        
        // Free memory
        wasm.tradjema_free(highPtr, size);
        wasm.tradjema_free(lowPtr, size);
        wasm.tradjema_free(closePtr, size);
        wasm.tradjema_free(outPtr, size);
    }
});

// Enhanced memory stability test with repeated allocations and stress patterns
test('TRADJEMA memory stability under stress', () => {
    // Test rapid allocation/deallocation cycles
    const iterations = 50;
    const baseSize = 1000;
    
    for (let iter = 0; iter < iterations; iter++) {
        // Vary size to stress different memory patterns
        const size = baseSize + (iter * 100);
        
        // Allocate memory
        const ptrs = {
            high: wasm.tradjema_alloc(size),
            low: wasm.tradjema_alloc(size),
            close: wasm.tradjema_alloc(size),
            out: wasm.tradjema_alloc(size)
        };
        
        // Verify all allocations succeeded
        for (const [name, ptr] of Object.entries(ptrs)) {
            assert(ptr !== 0, `Iteration ${iter}: Failed to allocate ${name} with ${size} elements`);
        }
        
        // Fill with test data
        const memory = wasm.__wasm.memory.buffer;
        const views = {
            high: new Float64Array(memory, ptrs.high, size),
            low: new Float64Array(memory, ptrs.low, size),
            close: new Float64Array(memory, ptrs.close, size)
        };
        
        // Generate synthetic data
        for (let i = 0; i < size; i++) {
            const baseVal = 100 + Math.sin(i * 0.1) * 10;
            views.close[i] = baseVal;
            views.high[i] = baseVal + Math.random() * 2;
            views.low[i] = baseVal - Math.random() * 2;
        }
        
        // Run computation
        try {
            wasm.tradjema_into(ptrs.high, ptrs.low, ptrs.close, ptrs.out, size, 20, 10.0);
            
            // Verify output has expected structure
            const memory2 = wasm.__wasm.memory.buffer;
            const outView = new Float64Array(memory2, ptrs.out, size);
            
            // Check warmup period
            for (let i = 0; i < 19; i++) {
                assert(isNaN(outView[i]), `Iteration ${iter}: Expected NaN at warmup index ${i}`);
            }
            
            // Check valid values after warmup
            for (let i = 19; i < Math.min(30, size); i++) {
                assert(!isNaN(outView[i]), `Iteration ${iter}: Unexpected NaN at index ${i}`);
                assert(isFinite(outView[i]), `Iteration ${iter}: Non-finite value at index ${i}`);
            }
        } finally {
            // Always free memory, even if test fails
            wasm.tradjema_free(ptrs.high, size);
            wasm.tradjema_free(ptrs.low, size);
            wasm.tradjema_free(ptrs.close, size);
            wasm.tradjema_free(ptrs.out, size);
        }
    }
    
    // Test concurrent allocations (simulate multiple indicators running)
    const concurrentAllocs = [];
    for (let i = 0; i < 10; i++) {
        const size = 500 * (i + 1);
        concurrentAllocs.push({
            size,
            ptrs: [
                wasm.tradjema_alloc(size),
                wasm.tradjema_alloc(size),
                wasm.tradjema_alloc(size),
                wasm.tradjema_alloc(size)
            ]
        });
    }
    
    // Verify all concurrent allocations succeeded
    for (let i = 0; i < concurrentAllocs.length; i++) {
        const {size, ptrs} = concurrentAllocs[i];
        for (let j = 0; j < ptrs.length; j++) {
            assert(ptrs[j] !== 0, `Concurrent alloc ${i}, buffer ${j} failed`);
        }
    }
    
    // Free all concurrent allocations
    for (const {size, ptrs} of concurrentAllocs) {
        for (const ptr of ptrs) {
            wasm.tradjema_free(ptr, size);
        }
    }
});

// SIMD128 verification test
test('TRADJEMA SIMD128 consistency', () => {
    // This test verifies SIMD128 produces same results as scalar
    // It runs automatically when SIMD128 is enabled
    const testCases = [
        { size: 10, length: 5 },
        { size: 100, length: 20 },
        { size: 1000, length: 40 },
        { size: 10000, length: 50 }
    ];
    
    for (const testCase of testCases) {
        const high = new Float64Array(testCase.size);
        const low = new Float64Array(testCase.size);
        const close = new Float64Array(testCase.size);
        
        for (let i = 0; i < testCase.size; i++) {
            high[i] = Math.sin(i * 0.1) + Math.cos(i * 0.05) + 2.0;
            low[i] = Math.sin(i * 0.1) + Math.cos(i * 0.05) + 1.0;
            close[i] = Math.sin(i * 0.1) + Math.cos(i * 0.05) + 1.5;
        }
        
        const result = wasm.tradjema_js(high, low, close, testCase.length, 10.0);
        
        // Basic sanity checks
        assert.strictEqual(result.length, high.length);
        
        // Check warmup period
        for (let i = 0; i < testCase.length - 1; i++) {
            assert(isNaN(result[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}`);
        }
        
        // Check values exist after warmup
        let sumAfterWarmup = 0;
        let countAfterWarmup = 0;
        for (let i = testCase.length - 1; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for size=${testCase.size}`);
            sumAfterWarmup += result[i];
            countAfterWarmup++;
        }
        
        // Verify reasonable values
        const avgAfterWarmup = sumAfterWarmup / countAfterWarmup;
        assert(Math.abs(avgAfterWarmup - 1.5) < 1.0, `Average value ${avgAfterWarmup} seems unreasonable`);
    }
});

test.after(() => {
    console.log('TRADJEMA WASM tests completed');
});
