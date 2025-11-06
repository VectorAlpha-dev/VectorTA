/**
 * WASM binding tests for Ehlers KAMA indicator.
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

test('Ehlers KAMA partial params', () => {
    // Test with default parameters - mirrors test_ehlers_kama_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.ehlers_kama_js(close, 20);
    assert.strictEqual(result.length, close.length);
});

test('Ehlers KAMA accuracy', async () => {
    // Test Ehlers KAMA matches expected values from Rust tests - mirrors test_ehlers_kama_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.ehlersKama;
    
    const result = wasm.ehlers_kama_js(
        close,
        expected.defaultParams.period
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 6 values, first 5 match expected (Pine non-repainting alignment)
    const last6 = result.slice(-6);
    const checkValues = last6.slice(0, 5);
    assertArrayClose(
        checkValues,
        expected.last5Values,
        1e-8,
        "Ehlers KAMA last 5 values mismatch"
    );
    
    // TODO: Enable once ehlers_kama is added to generate_references binary
    // await compareWithRust('ehlers_kama', result, 'close', expected.defaultParams);
});

test('Ehlers KAMA default', () => {
    // Test Ehlers KAMA with default parameters - mirrors test_ehlers_kama_default
    const close = new Float64Array(testData.close);
    
    const result = wasm.ehlers_kama_js(close, 20);
    assert.strictEqual(result.length, close.length);
});

test('Ehlers KAMA zero period', () => {
    // Test Ehlers KAMA fails with zero period - mirrors test_ehlers_kama_invalid_period
    const inputData = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]);
    
    assert.throws(() => {
        wasm.ehlers_kama_js(inputData, 0);
    }, /Invalid period/);
});

test('Ehlers KAMA period exceeds length', () => {
    // Test Ehlers KAMA fails when period exceeds data length - mirrors test_ehlers_kama_invalid_period
    const dataSmall = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0]);
    
    assert.throws(() => {
        wasm.ehlers_kama_js(dataSmall, 10);
    }, /Invalid period/);
});

test('Ehlers KAMA very small dataset', () => {
    // Test Ehlers KAMA fails with insufficient data - mirrors test_ehlers_kama_invalid_period
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.ehlers_kama_js(singlePoint, 20);
    }, /Invalid period|Not enough valid data/);
});

test('Ehlers KAMA empty input', () => {
    // Test Ehlers KAMA fails with empty input - mirrors test_ehlers_kama_empty_input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.ehlers_kama_js(empty, 20);
    }, /Input data slice is empty/);
});

test('Ehlers KAMA all NaN', () => {
    // Test Ehlers KAMA fails with all NaN values - mirrors test_ehlers_kama_all_nan
    const data = new Float64Array(10).fill(NaN);
    
    assert.throws(() => {
        wasm.ehlers_kama_js(data, 5);
    }, /All input data is NaN|All values are NaN/);
});

test('Ehlers KAMA not enough valid data', () => {
    // Test Ehlers KAMA fails with insufficient valid data
    const data = new Float64Array([NaN, NaN, NaN, NaN, 1.0, 2.0, 3.0]);
    
    assert.throws(() => {
        wasm.ehlers_kama_js(data, 5);
    }, /Not enough valid data|Invalid period/);
});

test('Ehlers KAMA with NaN prefix', () => {
    // Test Ehlers KAMA handles NaN prefix correctly
    const dataWithNaN = new Float64Array([NaN, NaN, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
    
    const result = wasm.ehlers_kama_js(dataWithNaN, 3);
    assert.strictEqual(result.length, dataWithNaN.length);
    
    // First few values should be NaN due to warmup
    assert(isNaN(result[0]));
    assert(isNaN(result[1]));
    assert(isNaN(result[2]));
    assert(isNaN(result[3]));
    
    // After warmup, should have valid values
    assert(!isNaN(result[4]));
});

test('Ehlers KAMA memory management', () => {
    // Test Ehlers KAMA memory allocation and deallocation functions
    const len = 100;
    
    // Allocate memory
    const ptr = wasm.ehlers_kama_alloc(len);
    assert(ptr !== 0, "Failed to allocate memory");
    
    // Free memory
    assert.doesNotThrow(() => {
        wasm.ehlers_kama_free(ptr, len);
    });
});

test('Ehlers KAMA into with null pointers', () => {
    // Test that into function properly validates null pointers
    assert.throws(() => {
        wasm.ehlers_kama_into(0, 0, 10, 5);
    }, /null pointer/);
});

test('Ehlers KAMA batch with NaN handling', () => {
    // Test batch processing with NaN values
    const data = new Float64Array([NaN, NaN].concat(Array.from({length: 48}, (_, i) => i + 1)));
    
    const result = wasm.ehlers_kama_batch(data, {
        period_range: [10, 20, 10]  // periods 10, 20
    });
    
    // Both parameter combinations should handle NaN correctly
    for (let rowIdx = 0; rowIdx < 2; rowIdx++) {
        const rowStart = rowIdx * 50;
        const row = result.values.slice(rowStart, rowStart + 50);
        // First values should be NaN due to input NaN and warmup
        assert(isNaN(row[0]));
        assert(isNaN(row[1]));
    }
});

test('Ehlers KAMA NaN handling', () => {
    // Test Ehlers KAMA handles NaN values correctly - mirrors check_ehlers_kama_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.ehlers_kama_js(close, 20);
    assert.strictEqual(result.length, close.length);
    
    // First period-1 values should be NaN (warmup period)
    assertAllNaN(result.slice(0, 19), "Expected NaN in warmup period");
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('Ehlers KAMA warmup period validation', () => {
    // Test that warmup period is exactly period-1 NaN values
    const close = new Float64Array(testData.close);
    const period = 20;
    
    const result = wasm.ehlers_kama_js(close, period);
    
    // Count NaN values at the start
    let nanCount = 0;
    for (const val of result) {
        if (isNaN(val)) {
            nanCount++;
        } else {
            break;
        }
    }
    
    // Should have exactly period-1 NaN values
    assert.strictEqual(nanCount, period - 1, `Expected ${period-1} NaN values for warmup, got ${nanCount}`);
    
    // First non-NaN should be at index period-1
    assert(!isNaN(result[period-1]), `Expected valid value at index ${period-1}`);
});

test('Ehlers KAMA batch single parameter', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    // Using the ergonomic batch API for single parameter
    const batchResult = wasm.ehlers_kama_batch(close, {
        period_range: [20, 20, 0]  // Default period only
    });
    
    // Should match single calculation
    const singleResult = wasm.ehlers_kama_js(close, 20);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    // Match Rust property tests which allow ~1e-9 (or a few ULPs)
    assertArrayClose(batchResult.values, singleResult, 1e-9, "Batch vs single mismatch");
});

test('Ehlers KAMA batch multiple periods', () => {
    // Test batch with multiple period values
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    
    // Multiple periods: 10, 15, 20 using ergonomic API
    const batchResult = wasm.ehlers_kama_batch(close, {
        period_range: [10, 20, 5]  // period range: 10, 15, 20
    });
    
    // Should have 3 rows * 100 cols = 300 values
    assert.strictEqual(batchResult.values.length, 3 * 100);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 100);
    
    // Verify each row matches individual calculation
    const periods = [10, 15, 20];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.ehlers_kama_js(close, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('Ehlers KAMA batch metadata from result', () => {
    // Test that batch result includes correct parameter combinations
    const close = new Float64Array(50); // Small test data
    close.fill(100);
    
    const result = wasm.ehlers_kama_batch(close, {
        period_range: [10, 30, 10]  // period: 10, 20, 30
    });
    
    // Should have 3 combinations
    assert.strictEqual(result.combos.length, 3);
    
    // Check combinations
    assert.strictEqual(result.combos[0].period, 10);
    assert.strictEqual(result.combos[1].period, 20);
    assert.strictEqual(result.combos[2].period, 30);
});

test('Ehlers KAMA batch into function', () => {
    // Test Ehlers KAMA batch direct memory manipulation
    const data = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0,
                                  110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0]);
    const len = data.length;
    const periods = [5, 10];
    const numRows = periods.length;
    
    // Allocate memory
    const inPtr = wasm.ehlers_kama_alloc(len);
    const outPtr = wasm.ehlers_kama_alloc(len * numRows);
    
    try {
        // Create memory views
        const memory = wasm.__wasm ? wasm.__wasm.memory : wasm.memory;
        const inputArray = new Float64Array(memory.buffer, inPtr, len);
        const outputArray = new Float64Array(memory.buffer, outPtr, len * numRows);
        
        // Copy data to input buffer
        inputArray.set(data);
        
        // Process batch data
        const resultRows = wasm.ehlers_kama_batch_into(inPtr, outPtr, len, 5, 10, 5);
        assert.strictEqual(resultRows, numRows, "Batch should return correct number of rows");
        
        // Verify each row matches individual calculation
        for (let i = 0; i < numRows; i++) {
            const rowStart = i * len;
            const rowEnd = rowStart + len;
            const rowData = Array.from(outputArray.slice(rowStart, rowEnd));
            
            const expected = wasm.ehlers_kama_js(data, periods[i]);
            for (let j = 0; j < len; j++) {
                if (isNaN(expected[j])) {
                    assert(isNaN(rowData[j]), `Row ${i} mismatch at index ${j}: expected NaN, got ${rowData[j]}`);
                } else {
                    assertClose(rowData[j], expected[j], 1e-10, `Row ${i} mismatch at index ${j}`);
                }
            }
        }
    } finally {
        // Clean up
        wasm.ehlers_kama_free(inPtr, len);
        wasm.ehlers_kama_free(outPtr, len * numRows);
    }
});

test('Ehlers KAMA into function', () => {
    // Test Ehlers KAMA direct memory manipulation function
    const data = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]);
    const len = data.length;
    
    // Allocate input and output buffers
    const inPtr = wasm.ehlers_kama_alloc(len);
    const outPtr = wasm.ehlers_kama_alloc(len);
    
    try {
        // Create memory views
        const memory = wasm.__wasm ? wasm.__wasm.memory : wasm.memory;
        const inputArray = new Float64Array(memory.buffer, inPtr, len);
        const outputArray = new Float64Array(memory.buffer, outPtr, len);
        
        // Copy data to input buffer
        inputArray.set(data);
        
        // Process data
        wasm.ehlers_kama_into(inPtr, outPtr, len, 3);
        
        // Check output
        assert.strictEqual(outputArray.length, data.length);
        
        // Compare with regular function
        const expected = wasm.ehlers_kama_js(data, 3);
        for (let i = 0; i < len; i++) {
            if (isNaN(expected[i])) {
                assert(isNaN(outputArray[i]), `Mismatch at index ${i}: expected NaN, got ${outputArray[i]}`);
            } else {
                assertClose(outputArray[i], expected[i], 1e-10, `Mismatch at index ${i}`);
            }
        }
    } finally {
        // Clean up
        wasm.ehlers_kama_free(inPtr, len);
        wasm.ehlers_kama_free(outPtr, len);
    }
});
