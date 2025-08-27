/**
 * WASM binding tests for ZSCORE indicator.
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
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('ZSCORE basic candles', () => {
    // Test with default parameters - mirrors check_zscore_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.zscore_js(close, 14, "sma", 1.0, 0);
    assert.strictEqual(result.length, close.length);
});

test('ZSCORE with custom parameters', () => {
    const close = new Float64Array(testData.close);
    
    // Test with custom parameters
    const result = wasm.zscore_js(close, 20, "ema", 2.0, 0);
    assert.strictEqual(result.length, close.length);
    
    // Check that warmup period values are NaN (first period-1 values)
    for (let i = 0; i < 19; i++) {
        assert(isNaN(result[i]));
    }
    
    // After warmup, should have values
    assert(!isNaN(result[19]));
});

test('ZSCORE zero period', () => {
    // Test ZSCORE fails with zero period - mirrors check_zscore_with_zero_period
    const data = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.zscore_js(data, 0, "sma", 1.0, 0);
    }, /Invalid period/);
});

test('ZSCORE period exceeds length', () => {
    // Test ZSCORE fails when period exceeds data length - mirrors check_zscore_period_exceeds_length
    const data = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.zscore_js(data, 10, "sma", 1.0, 0);
    }, /Invalid period/);
});

test('ZSCORE very small dataset', () => {
    // Test ZSCORE fails with insufficient data - mirrors check_zscore_very_small_dataset
    const data = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.zscore_js(data, 14, "sma", 1.0, 0);
    }, /Invalid period/);
});

test('ZSCORE all NaN', () => {
    // Test ZSCORE fails with all NaN values - mirrors check_zscore_all_nan
    const data = new Float64Array([NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.zscore_js(data, 2, "sma", 1.0, 0);
    }, /All values are NaN/);
});

test('ZSCORE fast API - basic', () => {
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Allocate memory for input and output
    const inPtr = wasm.zscore_alloc(len);
    const outPtr = wasm.zscore_alloc(len);
    assert(inPtr !== 0, 'Failed to allocate input memory');
    assert(outPtr !== 0, 'Failed to allocate output memory');
    
    try {
        // Copy data to WASM memory
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inView.set(close);
        
        // Compute zscore using fast API
        wasm.zscore_into(inPtr, outPtr, len, 14, "sma", 1.0, 0);
        
        // Create view of output (may need to recreate after potential memory growth)
        const resultBuffer = wasm.__wasm.memory.buffer;
        const result = new Float64Array(resultBuffer, outPtr, len);
        
        // Convert to regular array for comparison
        const resultArray = Array.from(result);
        
        // Compare with safe API
        const expected = wasm.zscore_js(close, 14, "sma", 1.0, 0);
        assertArrayClose(resultArray, expected, 1e-10, "Fast API mismatch");
        
    } finally {
        wasm.zscore_free(inPtr, len);
        wasm.zscore_free(outPtr, len);
    }
});

test('ZSCORE fast API - in-place', () => {
    const data = new Float64Array(testData.close.slice(0, 100));
    const len = data.length;
    
    // Allocate buffer and copy data
    const ptr = wasm.zscore_alloc(len);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    try {
        // Copy data to WASM memory
        const buffer = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        buffer.set(data);
        
        // Compute in-place (same pointer for input and output)
        wasm.zscore_into(ptr, ptr, len, 14, "sma", 1.0, 0);
        
        // Re-create view after potential memory growth
        const result = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        
        // Compare with safe API
        const expected = wasm.zscore_js(data, 14, "sma", 1.0, 0);
        assertArrayClose(result, expected, 1e-10, "In-place computation mismatch");
        
    } finally {
        wasm.zscore_free(ptr, len);
    }
});

test('ZSCORE batch processing', async () => {
    const close = new Float64Array(testData.close.slice(0, 500));
    
    // Call batch fast API directly with individual parameters
    const inPtr = wasm.zscore_alloc(close.length);
    
    try {
        // Copy data to WASM memory
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, close.length);
        inView.set(close);
        
        // Calculate expected output size
        const nPeriods = 3; // (20-10)/5 + 1
        const nNbdevs = 3;  // (2.0-1.0)/0.5 + 1
        const nCombos = nPeriods * nNbdevs;
        
        // Allocate output buffer
        const outPtr = wasm.zscore_alloc(nCombos * close.length);
        
        try {
            // Call batch function
            const actualCombos = wasm.zscore_batch_into(
                inPtr, outPtr, close.length,
                10, 20, 5,     // period_start, period_end, period_step
                "sma",         // ma_type
                1.0, 2.0, 0.5, // nbdev_start, nbdev_end, nbdev_step
                0, 0, 0        // devtype_start, devtype_end, devtype_step
            );
            
            assert.strictEqual(actualCombos, 9, 'Should return 9 combinations');
            
            // Create view of results
            const results = new Float64Array(wasm.__wasm.memory.buffer, outPtr, nCombos * close.length);
            
            // Verify first row has proper warmup
            let firstNonNaN = -1;
            for (let i = 0; i < close.length; i++) {
                if (!isNaN(results[i])) {
                    firstNonNaN = i;
                    break;
                }
            }
            assert(firstNonNaN >= 9, 'First row should have warmup period of at least 9 (period-1)');
            
        } finally {
            wasm.zscore_free(outPtr, nCombos * close.length);
        }
    } finally {
        wasm.zscore_free(inPtr, close.length);
    }
});

test('ZSCORE batch fast API', () => {
    const close = new Float64Array(testData.close.slice(0, 100));
    const len = close.length;
    
    // Calculate expected combinations
    const nPeriods = Math.floor((20 - 10) / 5) + 1; // 3
    const nNbdevs = Math.floor((2.0 - 1.0) / 0.5) + 1; // 3
    const nCombos = nPeriods * nNbdevs; // 9
    
    // Allocate input and output buffers
    const inPtr = wasm.zscore_alloc(len);
    const outPtr = wasm.zscore_alloc(nCombos * len);
    assert(inPtr !== 0, 'Failed to allocate input memory');
    assert(outPtr !== 0, 'Failed to allocate output memory');
    
    try {
        // Copy data to WASM memory
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inView.set(close);
        
        const resultCombos = wasm.zscore_batch_into(
            inPtr,
            outPtr,
            len,
            10, 20, 5,  // period range
            "sma",
            1.0, 2.0, 0.5,  // nbdev range
            0, 0, 0  // devtype range
        );
        
        assert.strictEqual(resultCombos, nCombos, 'Should return correct number of combinations');
        
        // Verify some output values exist
        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, nCombos * len);
        assert(result.some(v => !isNaN(v)), 'Should have some non-NaN values');
        
    } finally {
        wasm.zscore_free(inPtr, len);
        wasm.zscore_free(outPtr, nCombos * len);
    }
});

test('ZSCORE different MA types', () => {
    const close = new Float64Array(testData.close.slice(0, 100));
    const maTypes = ["sma", "ema", "wma"];
    
    for (const maType of maTypes) {
        try {
            const result = wasm.zscore_js(close, 14, maType, 1.0, 0);
            assert.strictEqual(result.length, close.length);
            
            // Check warmup period
            assert(isNaN(result[0]));
            // After warmup should have values
            assert(!isNaN(result[20]));
        } catch (e) {
            // Some MA types might not be supported
            assert(e.message.includes("Unknown MA") || e.message.includes("Invalid"));
        }
    }
});

test('ZSCORE deviation types', () => {
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // devtype: 0 = stddev, 1 = mean abs dev, 2 = median abs dev
    for (const devtype of [0, 1, 2]) {
        const result = wasm.zscore_js(close, 14, "sma", 1.0, devtype);
        assert.strictEqual(result.length, close.length);
    }
});