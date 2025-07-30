/**
 * WASM binding tests for VLMA indicator.
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

test('VLMA partial params', () => {
    // Test with default parameters - mirrors check_vlma_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.vlma_js(close, 5, 50, "sma", 0);
    assert.strictEqual(result.length, close.length);
});

test('VLMA accuracy', async () => {
    // Test VLMA matches expected values from Rust tests - mirrors check_vlma_accuracy
    const close = new Float64Array(testData.close);
    
    const result = wasm.vlma_js(close, 5, 50, "sma", 0);
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected (from Rust test)
    const expectedLast5 = [
        59376.252799490234,
        59343.71066624187,
        59292.92555520155,
        59269.93796266796,
        59167.4483022233,
    ];
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLast5,
        1e-1,  // Less strict tolerance for VLMA
        "VLMA last 5 values mismatch"
    );
});

test('VLMA zero or inverted periods', () => {
    // Test VLMA fails with zero or inverted periods - mirrors check_vlma_zero_or_inverted_periods
    const inputData = new Float64Array([10.0, 20.0, 30.0, 40.0]);
    
    // Test min_period > max_period
    assert.throws(() => {
        wasm.vlma_js(inputData, 10, 5, "sma", 0);
    }, /min_period.*is greater than max_period/);
    
    // Test zero max_period
    assert.throws(() => {
        wasm.vlma_js(inputData, 5, 0, "sma", 0);
    }, /Invalid period/);
});

test('VLMA not enough data', () => {
    // Test VLMA fails with insufficient data - mirrors check_vlma_not_enough_data
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.vlma_js(inputData, 5, 10, "sma", 0);
    }, /Invalid period|Not enough valid data/);
});

test('VLMA all NaN', () => {
    // Test VLMA fails with all NaN input - mirrors check_vlma_all_nan
    const inputData = new Float64Array([NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.vlma_js(inputData, 2, 3, "sma", 0);
    }, /All values are NaN/);
});

test('VLMA empty input', () => {
    // Test VLMA fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.vlma_js(empty, 5, 50, "sma", 0);
    }, /Empty data/);
});

test('VLMA slice reinput', () => {
    // Test VLMA can process its own output - mirrors check_vlma_slice_reinput
    const close = new Float64Array(testData.close);
    
    // First pass
    const firstResult = wasm.vlma_js(close, 5, 20, "ema", 1);
    
    // Second pass - apply VLMA to VLMA output
    const secondResult = wasm.vlma_js(firstResult, 5, 20, "ema", 1);
    
    assert.strictEqual(secondResult.length, firstResult.length);
});

test('VLMA batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    // Using the new ergonomic batch API for single parameter
    const batchResult = wasm.vlma_batch(close, {
        min_period_range: [5, 5, 0],
        max_period_range: [50, 50, 0],
        devtype_range: [0, 0, 0],
        matype: "sma"
    });
    
    // Should match single calculation
    const singleResult = wasm.vlma_js(close, 5, 50, "sma", 0);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('VLMA batch multiple periods', () => {
    // Test batch with multiple period values
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    
    // Multiple min periods: 5, 10, 15
    const batchResult = wasm.vlma_batch(close, {
        min_period_range: [5, 15, 5],     // min period range
        max_period_range: [50, 50, 0],    // max period range  
        devtype_range: [0, 0, 0],         // devtype range
        matype: "sma"
    });
    
    // Should have 3 rows * 100 cols = 300 values
    assert.strictEqual(batchResult.values.length, 3 * 100);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 100);
    
    // Verify each row matches individual calculation
    const minPeriods = [5, 10, 15];
    for (let i = 0; i < minPeriods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.vlma_js(close, minPeriods[i], 50, "sma", 0);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Min period ${minPeriods[i]} mismatch`
        );
    }
});

test('VLMA batch metadata from result', () => {
    // Test that batch result includes correct parameter combinations
    const close = new Float64Array(60); // Need enough data for max_period
    close.fill(100);
    
    const result = wasm.vlma_batch(close, {
        min_period_range: [5, 10, 5],      // min_period: 5, 10
        max_period_range: [30, 40, 10],    // max_period: 30, 40
        devtype_range: [0, 2, 1],          // devtype: 0, 1, 2
        matype: "sma"
    });
    
    // Should have 2 * 2 * 3 = 12 combinations
    assert.strictEqual(result.combos.length, 12);
    
    // Check first combination
    assert.strictEqual(result.combos[0].min_period, 5);
    assert.strictEqual(result.combos[0].max_period, 30);
    assert.strictEqual(result.combos[0].devtype, 0);
    assert.strictEqual(result.combos[0].matype, "sma");
    
    // Check last combination
    assert.strictEqual(result.combos[11].min_period, 10);
    assert.strictEqual(result.combos[11].max_period, 40);
    assert.strictEqual(result.combos[11].devtype, 2);
    assert.strictEqual(result.combos[11].matype, "sma");
});

// Fast/unsafe API tests

test('VLMA zero-copy API in-place operation', () => {
    // Test in-place operation with aliasing
    const data = new Float64Array([
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
        110, 120, 130, 140, 150, 160, 170, 180, 190, 200
    ]);
    const minPeriod = 3;
    const maxPeriod = 10;
    const matype = "sma";
    const devtype = 0;
    
    // Allocate buffer
    const ptr = wasm.vlma_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    // Create view into WASM memory
    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        data.length
    );
    
    // Copy data into WASM memory
    memView.set(data);
    
    // Compute VLMA in-place
    try {
        wasm.vlma_into(ptr, ptr, data.length, minPeriod, maxPeriod, matype, devtype);
        
        // Verify results match regular API
        const regularResult = wasm.vlma_js(data, minPeriod, maxPeriod, matype, devtype);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue;
            }
            assertClose(memView[i], regularResult[i], 1e-10, `Mismatch at index ${i}`);
        }
    } finally {
        wasm.vlma_free(ptr, data.length);
    }
});

test('VLMA zero-copy API separate buffers', () => {
    // Test with separate input and output buffers
    const size = 100;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }
    
    const inPtr = wasm.vlma_alloc(size);
    const outPtr = wasm.vlma_alloc(size);
    assert(inPtr !== 0, 'Failed to allocate input buffer');
    assert(outPtr !== 0, 'Failed to allocate output buffer');
    
    try {
        // Copy data to input buffer
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, size);
        inView.set(data);
        
        // Compute VLMA
        wasm.vlma_into(inPtr, outPtr, size, 5, 20, "ema", 1);
        
        // Get output view
        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, size);
        
        // Verify results match regular API
        const regularResult = wasm.vlma_js(data, 5, 20, "ema", 1);
        assertArrayClose(outView, regularResult, 1e-10, "Zero-copy mismatch");
        
    } finally {
        wasm.vlma_free(inPtr, size);
        wasm.vlma_free(outPtr, size);
    }
});

test('VLMA zero-copy error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.vlma_into(0, 0, 10, 5, 50, "sma", 0);
    }, /null pointer|Null pointer provided/i);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.vlma_alloc(10);
    try {
        // Invalid period
        assert.throws(() => {
            wasm.vlma_into(ptr, ptr, 10, 20, 5, "sma", 0);
        }, /min_period.*is greater than max_period/);
        
        // Zero max period
        assert.throws(() => {
            wasm.vlma_into(ptr, ptr, 10, 5, 0, "sma", 0);
        }, /Invalid period/);
    } finally {
        wasm.vlma_free(ptr, 10);
    }
});

test('VLMA zero-copy memory management', () => {
    // Allocate and free multiple times to ensure no leaks
    const sizes = [100, 1000, 10000, 100000];
    
    for (const size of sizes) {
        const ptr = wasm.vlma_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        // Write pattern to verify memory
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }
        
        // Verify pattern
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Pattern mismatch at ${i}`);
        }
        
        wasm.vlma_free(ptr, size);
    }
});

test('VLMA batch zero-copy API', () => {
    // Test batch processing with zero-copy API
    const size = 50;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.1) * 100 + 100;
    }
    
    // Calculate expected output size
    const minPeriods = 3; // 5, 10, 15
    const maxPeriods = 1; // 50
    const devtypes = 2;   // 0, 1
    const totalCombos = minPeriods * maxPeriods * devtypes;
    const totalSize = totalCombos * size;
    
    const inPtr = wasm.vlma_alloc(size);
    const outPtr = wasm.vlma_alloc(totalSize);
    
    try {
        // Copy data to input buffer
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, size);
        inView.set(data);
        
        // Compute batch
        const numCombos = wasm.vlma_batch_into(
            inPtr, outPtr, size,
            5, 15, 5,    // min_period range
            50, 50, 0,   // max_period range
            0, 1, 1,     // devtype range
            "sma"
        );
        
        assert.strictEqual(numCombos, totalCombos, 'Unexpected number of combinations');
        
        // Get output view
        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, totalSize);
        
        // Verify at least the first combination
        const firstRowResult = new Float64Array(size);
        for (let i = 0; i < size; i++) {
            firstRowResult[i] = outView[i];
        }
        
        const expectedFirst = wasm.vlma_js(data, 5, 50, "sma", 0);
        assertArrayClose(firstRowResult, expectedFirst, 1e-10, "First batch row mismatch");
        
    } finally {
        wasm.vlma_free(inPtr, size);
        wasm.vlma_free(outPtr, totalSize);
    }
});