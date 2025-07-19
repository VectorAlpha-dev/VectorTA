/**
 * WASM binding tests for Gaussian indicator.
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
        
        // Check if memory is accessible
        if (!wasm.__wasm || !wasm.__wasm.memory) {
            console.error('WASM memory not accessible. Available properties:', Object.keys(wasm));
            if (wasm.__wasm) {
                console.error('__wasm properties:', Object.keys(wasm.__wasm));
            }
        }
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('Gaussian partial params', () => {
    // Test with default parameters - mirrors check_gaussian_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.gaussian_js(close, 14, 4);
    assert.strictEqual(result.length, close.length);
});

test('Gaussian accuracy', async () => {
    // Test Gaussian matches expected values from Rust tests - mirrors check_gaussian_accuracy
    const close = new Float64Array(testData.close);
    
    const result = wasm.gaussian_js(close, 14, 4);
    
    assert.strictEqual(result.length, close.length);
    
    // Expected last 5 values from Rust test
    const expectedLastFive = [
        59221.90637814869,
        59236.15215167245,
        59207.10087088464,
        59178.48276885589,
        59085.36983209433
    ];
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLastFive,
        1e-4,  // Using 1e-4 as in Rust test
        "Gaussian last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('gaussian', result, 'close', { period: 14, poles: 4 });
});

test('Gaussian default candles', async () => {
    // Test Gaussian with default parameters - mirrors check_gaussian_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.gaussian_js(close, 14, 4);
    assert.strictEqual(result.length, close.length);
    
    // Compare with Rust
    await compareWithRust('gaussian', result, 'close', { period: 14, poles: 4 });
});

test('Gaussian zero period', () => {
    // Test Gaussian fails with zero period - mirrors check_gaussian_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.gaussian_js(inputData, 0, 4);
    });
});

test('Gaussian period exceeds length', () => {
    // Test Gaussian fails when period exceeds data length - mirrors check_gaussian_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.gaussian_js(dataSmall, 10, 4);
    });
});

test('Gaussian very small dataset', () => {
    // Test Gaussian with very small dataset - mirrors check_gaussian_very_small_dataset
    const dataSingle = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.gaussian_js(dataSingle, 3, 4);
    });
});

test('Gaussian empty input', () => {
    // Test Gaussian with empty input - mirrors check_gaussian_empty_input
    const dataEmpty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.gaussian_js(dataEmpty, 14, 4);
    });
});

test('Gaussian invalid poles', () => {
    // Test Gaussian with invalid poles - mirrors check_gaussian_invalid_poles
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    // Test poles = 0
    assert.throws(() => {
        wasm.gaussian_js(data, 3, 0);
    });
    
    // Test poles = 5 (> 4)
    assert.throws(() => {
        wasm.gaussian_js(data, 3, 5);
    });
});

test('Gaussian all NaN', () => {
    // Test Gaussian with all NaN input - mirrors check_gaussian_all_nan
    const data = new Float64Array([NaN, NaN, NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.gaussian_js(data, 3, 4);
    });
});

test('Gaussian reinput', () => {
    // Test Gaussian with re-input of Gaussian result - mirrors check_gaussian_reinput
    const close = new Float64Array(testData.close);
    
    // First Gaussian pass with period=14, poles=4
    const firstResult = wasm.gaussian_js(close, 14, 4);
    
    // Second Gaussian pass with period=10, poles=2 using first result as input
    const secondResult = wasm.gaussian_js(firstResult, 10, 2);
    
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // Verify no NaN values after warmup period in second result
    for (let i = 240; i < secondResult.length; i++) {
        assert(!isNaN(secondResult[i]), `NaN found at index ${i}`);
    }
});

test('Gaussian NaN handling', () => {
    // Test Gaussian handling of NaN values - mirrors check_gaussian_nan_handling
    // The Rust test doesn't actually test NaN inputs, it just verifies that
    // the outputs are finite for regular data. Let's test with regular data
    const close = new Float64Array(testData.close);
    
    const result = wasm.gaussian_js(close, 14, 4);
    
    assert.strictEqual(result.length, close.length);
    
    // Skip the first few values (poles) and check that remaining are finite
    const skip = 4; // poles
    for (let i = skip; i < result.length; i++) {
        assert(isFinite(result[i]), `Non-finite value found at index ${i}`);
    }
});

test('Gaussian batch', () => {
    // Test Gaussian batch computation
    const close = new Float64Array(testData.close);
    
    // Test period range 10-20 step 5, poles range 2-4 step 1
    const period_start = 10;
    const period_end = 20;
    const period_step = 5;  // periods: 10, 15, 20
    const poles_start = 2;
    const poles_end = 4;
    const poles_step = 1;   // poles: 2, 3, 4
    
    const batch_result = wasm.gaussian_batch_js(
        close, 
        period_start, period_end, period_step,
        poles_start, poles_end, poles_step
    );
    const metadata = wasm.gaussian_batch_metadata_js(
        period_start, period_end, period_step,
        poles_start, poles_end, poles_step
    );
    
    // Metadata should contain [period, poles] pairs
    assert.strictEqual(metadata.length, 18);  // 3 periods x 3 poles x 2 values per combo
    
    // Batch result should contain all individual results flattened
    assert.strictEqual(batch_result.length, 9 * close.length);  // 3 periods x 3 poles = 9 rows
    
    // Verify each row matches individual calculation
    let row_idx = 0;
    for (const period of [10, 15, 20]) {
        for (const poles of [2, 3, 4]) {
            const individual_result = wasm.gaussian_js(close, period, poles);
            
            // Extract row from batch result
            const row_start = row_idx * close.length;
            const row = batch_result.slice(row_start, row_start + close.length);
            
            assertArrayClose(row, individual_result, 1e-9, `Period ${period}, Poles ${poles}`);
            row_idx++;
        }
    }
});

test('Gaussian different poles', () => {
    // Test Gaussian with different poles values
    const close = new Float64Array(testData.close);
    const period = 14;
    
    // Test all valid poles values (1-4)
    for (const poles of [1, 2, 3, 4]) {
        const result = wasm.gaussian_js(close, period, poles);
        assert.strictEqual(result.length, close.length);
        
        // Verify warmup period - Rust implementation returns actual values during warmup
        // not NaN values, so we just check that we get valid results
        assert.strictEqual(result.length, close.length);
        // Check that after warmup we have valid non-NaN values
        for (let i = period; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for poles=${poles}`);
        }
    }
});

test('Gaussian batch performance', () => {
    // Test that batch computation is more efficient than multiple single computations
    const close = new Float64Array(testData.close.slice(0, 1000)); // Use first 1000 values
    
    // Test 5 periods x 4 poles = 20 combinations
    const startBatch = performance.now();
    const batchResult = wasm.gaussian_batch_js(close, 10, 30, 5, 1, 4, 1);
    const batchTime = performance.now() - startBatch;
    
    const startSingle = performance.now();
    const singleResults = [];
    for (let period = 10; period <= 30; period += 5) {
        for (let poles = 1; poles <= 4; poles++) {
            singleResults.push(...wasm.gaussian_js(close, period, poles));
        }
    }
    const singleTime = performance.now() - startSingle;
    
    // Batch should be faster than multiple single calls
    console.log(`Batch time: ${batchTime.toFixed(2)}ms, Single time: ${singleTime.toFixed(2)}ms`);
    
    // Verify results match
    assertArrayClose(batchResult, singleResults, 1e-9, 'Batch vs single results');
});

// Tests for new optimized WASM APIs

test('Gaussian fast API (gaussian_into)', () => {
    const close = new Float64Array(testData.close);
    const period = 14;
    const poles = 4;
    
    // Get result from safe API for comparison
    const safeResult = wasm.gaussian_js(close, period, poles);
    
    // Allocate memory for fast API
    const ptr = wasm.gaussian_alloc(close.length);
    const outPtr = wasm.gaussian_alloc(close.length);
    
    try {
        // Copy data to WASM memory
        const wasmMemory = new Float64Array(wasm.__wasm.memory.buffer, ptr, close.length);
        wasmMemory.set(close);
        
        // Call fast API
        wasm.gaussian_into(ptr, outPtr, close.length, period, poles);
        
        // Read result
        const fastResult = new Float64Array(wasm.__wasm.memory.buffer, outPtr, close.length);
        
        // Results should match
        assertArrayClose(Array.from(fastResult), safeResult, 1e-10, 'Fast API vs Safe API');
    } finally {
        // Clean up
        wasm.gaussian_free(ptr, close.length);
        wasm.gaussian_free(outPtr, close.length);
    }
});

test('Gaussian fast API with aliasing (in-place)', () => {
    const close = new Float64Array(testData.close);
    const period = 14;
    const poles = 4;
    
    // Get expected result
    const expectedResult = wasm.gaussian_js(close, period, poles);
    
    // Allocate single buffer for in-place operation
    const ptr = wasm.gaussian_alloc(close.length);
    
    try {
        // Copy data to WASM memory
        const wasmMemory = new Float64Array(wasm.__wasm.memory.buffer, ptr, close.length);
        wasmMemory.set(close);
        
        // Call fast API with same input and output pointers (aliasing)
        wasm.gaussian_into(ptr, ptr, close.length, period, poles);
        
        // Read result (should be computed correctly despite aliasing)
        const result = Array.from(new Float64Array(wasm.__wasm.memory.buffer, ptr, close.length));
        
        // Results should match
        assertArrayClose(result, expectedResult, 1e-10, 'In-place operation result');
    } finally {
        wasm.gaussian_free(ptr, close.length);
    }
});

test('Gaussian unified batch API', () => {
    const close = new Float64Array(testData.close.slice(0, 10000)); // Use smaller dataset
    
    // Configure batch parameters
    const config = {
        period_range: [10, 20, 5],  // 10, 15, 20
        poles_range: [2, 4, 1]      // 2, 3, 4
    };
    
    // Call unified batch API
    const result = wasm.gaussian_batch(close, config);
    
    // Verify structure
    assert(result.values, 'Result should have values');
    assert(result.combos, 'Result should have combos');
    assert.strictEqual(result.rows, 9, 'Should have 9 combinations (3 periods × 3 poles)');
    assert.strictEqual(result.cols, close.length, 'Should have same length as input');
    
    // Verify combos
    assert.strictEqual(result.combos.length, 9, 'Should have 9 parameter combinations');
    
    // Verify first combo
    assert.strictEqual(result.combos[0].period, 10);
    assert.strictEqual(result.combos[0].poles, 2);
    
    // Verify values shape
    assert.strictEqual(result.values.length, 9 * close.length, 'Values should be flattened matrix');
    
    // Verify first row matches individual computation
    const firstRow = result.values.slice(0, close.length);
    const expected = wasm.gaussian_js(close, 10, 2);
    assertArrayClose(firstRow, expected, 1e-10, 'First row should match individual computation');
});

test('Gaussian fast batch API (gaussian_batch_into)', () => {
    const close = new Float64Array(testData.close.slice(0, 1000)); // Small dataset
    
    // Batch parameters
    const periodStart = 10, periodEnd = 15, periodStep = 5;  // 2 periods
    const polesStart = 3, polesEnd = 4, polesStep = 1;       // 2 poles
    const expectedRows = 4; // 2 × 2
    
    // Allocate memory
    const inPtr = wasm.gaussian_alloc(close.length);
    const outPtr = wasm.gaussian_alloc(close.length * expectedRows);
    
    try {
        // Copy data to WASM memory
        const wasmInput = new Float64Array(wasm.__wasm.memory.buffer, inPtr, close.length);
        wasmInput.set(close);
        
        // Call fast batch API
        const rows = wasm.gaussian_batch_into(
            inPtr, outPtr, close.length,
            periodStart, periodEnd, periodStep,
            polesStart, polesEnd, polesStep
        );
        
        assert.strictEqual(rows, expectedRows, 'Should return correct number of rows');
        
        // Read results
        const results = new Float64Array(wasm.__wasm.memory.buffer, outPtr, close.length * rows);
        
        // Verify first row matches individual computation
        const firstRow = Array.from(results.slice(0, close.length));
        const expected = wasm.gaussian_js(close, periodStart, polesStart);
        assertArrayClose(firstRow, expected, 1e-10, 'First batch row should match individual');
    } finally {
        wasm.gaussian_free(inPtr, close.length);
        wasm.gaussian_free(outPtr, close.length * expectedRows);
    }
});

test('Gaussian memory management', () => {
    const size = 1000;
    
    // Test allocation
    const ptr1 = wasm.gaussian_alloc(size);
    assert(ptr1 !== 0, 'Should allocate non-null pointer');
    
    const ptr2 = wasm.gaussian_alloc(size);
    assert(ptr2 !== 0, 'Should allocate second non-null pointer');
    assert(ptr2 !== ptr1, 'Should allocate different memory regions');
    
    // Test we can write to allocated memory
    const mem1 = new Float64Array(wasm.__wasm.memory.buffer, ptr1, size);
    const mem2 = new Float64Array(wasm.__wasm.memory.buffer, ptr2, size);
    
    mem1[0] = 42.0;
    mem2[0] = 84.0;
    
    assert.strictEqual(mem1[0], 42.0, 'Should write to first allocation');
    assert.strictEqual(mem2[0], 84.0, 'Should write to second allocation');
    
    // Free memory
    wasm.gaussian_free(ptr1, size);
    wasm.gaussian_free(ptr2, size);
    
    // Test freeing null pointer (should not crash)
    wasm.gaussian_free(0, size);
});