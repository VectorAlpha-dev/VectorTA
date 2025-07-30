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
    
    // Check that warmup period values are NaN
    for (let i = 0; i < 20; i++) {
        assert(isNaN(result[i]));
    }
    
    // After warmup, should have values
    assert(!isNaN(result[20]));
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
    
    // Allocate output buffer
    const outPtr = wasm.zscore_alloc(len);
    
    try {
        // Get pointer to input data
        const inPtr = close.byteOffset;
        
        // Compute zscore using fast API
        wasm.zscore_into(close, outPtr, len, 14, "sma", 1.0, 0);
        
        // Create view of output
        const result = new Float64Array(wasm.memory.buffer, outPtr, len);
        
        // Compare with safe API
        const expected = wasm.zscore_js(close, 14, "sma", 1.0, 0);
        assertArrayClose(result, expected, 1e-10, "Fast API mismatch");
        
    } finally {
        wasm.zscore_free(outPtr, len);
    }
});

test('ZSCORE fast API - in-place', () => {
    const data = new Float64Array(testData.close.slice(0, 100));
    const len = data.length;
    
    // Allocate buffer and copy data
    const ptr = wasm.zscore_alloc(len);
    const buffer = new Float64Array(wasm.memory.buffer, ptr, len);
    buffer.set(data);
    
    try {
        // Compute in-place (same pointer for input and output)
        wasm.zscore_into(buffer, ptr, len, 14, "sma", 1.0, 0);
        
        // Compare with safe API
        const expected = wasm.zscore_js(data, 14, "sma", 1.0, 0);
        assertArrayClose(buffer, expected, 1e-10, "In-place computation mismatch");
        
    } finally {
        wasm.zscore_free(ptr, len);
    }
});

test('ZSCORE batch processing', async () => {
    const close = new Float64Array(testData.close.slice(0, 500));
    
    const config = {
        period_range: [10, 20, 5],  // 10, 15, 20
        ma_type: "sma",
        nbdev_range: [1.0, 2.0, 0.5], // 1.0, 1.5, 2.0
        devtype_range: [0, 0, 0]  // only stddev
    };
    
    const result = wasm.zscore_batch(close, config);
    
    assert(result.values, 'Should have values array');
    assert(result.combos, 'Should have combos array');
    assert.strictEqual(result.rows, 9, 'Should have 9 combinations (3 periods * 3 nbdevs)');
    assert.strictEqual(result.cols, close.length, 'Should have same columns as input');
    assert.strictEqual(result.values.length, 9 * close.length, 'Values array size mismatch');
    
    // Check that combos match expected
    assert.strictEqual(result.combos.length, 9);
    assert.strictEqual(result.combos[0].period, 10);
    assert.strictEqual(result.combos[0].nbdev, 1.0);
    assert.strictEqual(result.combos[8].period, 20);
    assert.strictEqual(result.combos[8].nbdev, 2.0);
});

test('ZSCORE batch fast API', () => {
    const close = new Float64Array(testData.close.slice(0, 100));
    const len = close.length;
    
    // Calculate expected combinations
    const nPeriods = Math.floor((20 - 10) / 5) + 1; // 3
    const nNbdevs = Math.floor((2.0 - 1.0) / 0.5) + 1; // 3
    const nCombos = nPeriods * nNbdevs; // 9
    
    // Allocate output buffer
    const outPtr = wasm.zscore_alloc(nCombos * len);
    
    try {
        const resultCombos = wasm.zscore_batch_into(
            close,
            outPtr,
            len,
            10, 20, 5,  // period range
            "sma",
            1.0, 2.0, 0.5,  // nbdev range
            0, 0, 0  // devtype range
        );
        
        assert.strictEqual(resultCombos, nCombos, 'Should return correct number of combinations');
        
        // Verify some output values exist
        const result = new Float64Array(wasm.memory.buffer, outPtr, nCombos * len);
        assert(result.some(v => !isNaN(v)), 'Should have some non-NaN values');
        
    } finally {
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