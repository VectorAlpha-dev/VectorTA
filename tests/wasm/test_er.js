/**
 * WASM binding tests for ER (Kaufman Efficiency Ratio) indicator.
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

test('ER partial params', () => {
    // Test with default parameters - mirrors check_er_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.er_js(close, 5);
    assert.strictEqual(result.length, close.length);
});

test('ER default candles', () => {
    // Test ER with default parameters - mirrors check_er_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.er_js(close, 5);
    assert.strictEqual(result.length, close.length);
});

test('ER zero period', () => {
    // Test ER fails with zero period - mirrors check_er_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.er_js(inputData, 0);
    }, /Invalid period/);
});

test('ER period exceeds length', () => {
    // Test ER fails when period exceeds data length - mirrors check_er_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.er_js(dataSmall, 10);
    }, /Invalid period/);
});

test('ER very small dataset', () => {
    // Test ER fails with insufficient data - mirrors check_er_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.er_js(singlePoint, 5);
    }, /Invalid period|Not enough valid data/);
});

test('ER empty input', () => {
    // Test ER fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.er_js(empty, 5);
    }, /All values are NaN/);
});

test('ER all NaN input', () => {
    // Test ER fails with all NaN values
    const allNan = new Float64Array(100);
    allNan.fill(NaN);
    
    assert.throws(() => {
        wasm.er_js(allNan, 5);
    }, /All values are NaN/);
});

test('ER NaN handling', () => {
    // Test ER handles NaN values correctly - mirrors check_er_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.er_js(close, 5);
    assert.strictEqual(result.length, close.length);
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        const nanCount = result.slice(240).filter(v => isNaN(v)).length;
        assert.strictEqual(nanCount, 0, "Found unexpected NaN after warmup period");
    }
    
    // First period-1 values should be NaN
    assert(isNaN(result[0]), "Expected NaN at index 0");
    assert(isNaN(result[1]), "Expected NaN at index 1");
    assert(isNaN(result[2]), "Expected NaN at index 2");
    assert(isNaN(result[3]), "Expected NaN at index 3");
});

test('ER reinput', () => {
    // Test ER applied twice (re-input) - mirrors check_er_reinput
    const close = new Float64Array(testData.close);
    
    // First pass
    const firstResult = wasm.er_js(close, 5);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass - apply ER to ER output
    const secondResult = wasm.er_js(firstResult, 5);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // Check that values are still in valid range (0.0 to 1.0)
    const validValues = secondResult.filter(v => !isNaN(v));
    validValues.forEach(v => {
        assert(v >= 0.0 && v <= 1.0, `ER value ${v} outside valid range [0.0, 1.0]`);
    });
});

test('ER consistency', () => {
    // Test ER produces consistent results for known data
    // Simple test case with clear trend
    const trendingData = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const result = wasm.er_js(trendingData, 5);
    
    // For perfectly trending data, ER should be close to 1
    const validValues = result.slice(4); // Skip warmup
    validValues.forEach(v => {
        assert(v > 0.9, `Expected ER > 0.9 for trending data, got ${v}`);
    });
    
    // Choppy data
    const choppyData = new Float64Array([1, 5, 2, 6, 3, 7, 4, 8, 5, 9]);
    const result2 = wasm.er_js(choppyData, 5);
    
    // For choppy data, ER should be lower
    const validValues2 = result2.slice(4); // Skip warmup
    validValues2.forEach(v => {
        assert(v < 0.5, `Expected ER < 0.5 for choppy data, got ${v}`);
    });
});

test('ER fast API - basic calculation', async () => {
    // Test the fast/unsafe API
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Allocate input and output buffers
    const inPtr = wasm.er_alloc(len);
    const outPtr = wasm.er_alloc(len);
    
    try {
        // Create typed arrays from WASM memory
        const memory = new Float64Array(wasm.__wasm.memory.buffer);
        const inOffset = inPtr / 8;
        const outOffset = outPtr / 8;
        
        // Copy data to input buffer
        memory.set(close, inOffset);
        
        // Run computation
        wasm.er_into(inPtr, outPtr, len, 5);
        
        // Extract output
        const result = new Float64Array(memory.buffer, outPtr, len);
        const resultCopy = new Float64Array(result); // Copy before freeing
        
        // Compare with safe API
        const safeResult = wasm.er_js(close, 5);
        assertArrayClose(resultCopy, safeResult, 1e-10, "Fast API result differs from safe API");
        
    } finally {
        // Clean up
        wasm.er_free(inPtr, len);
        wasm.er_free(outPtr, len);
    }
});

test('ER fast API - in-place operation (aliasing)', async () => {
    // Test that fast API handles aliasing correctly
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Allocate single buffer for both input and output
    const ptr = wasm.er_alloc(len);
    
    try {
        // Create typed array from WASM memory
        const memory = new Float64Array(wasm.__wasm.memory.buffer);
        const offset = ptr / 8;
        
        // Copy data to buffer
        memory.set(close, offset);
        
        // Run in-place computation (same pointer for input and output)
        wasm.er_into(ptr, ptr, len, 5);
        
        // Extract result
        const result = new Float64Array(memory.buffer, ptr, len);
        const resultCopy = new Float64Array(result); // Copy before freeing
        
        // Compare with safe API
        const safeResult = wasm.er_js(close, 5);
        assertArrayClose(resultCopy, safeResult, 1e-10, "In-place operation result differs from safe API");
        
    } finally {
        // Clean up
        wasm.er_free(ptr, len);
    }
});

test('ER batch - single period', () => {
    // Test batch operation with single period
    const close = new Float64Array(testData.close.slice(0, 100)); // Use subset for speed
    
    const config = {
        period_range: [5, 5, 0] // Single period = 5
    };
    
    const result = wasm.er_batch(close, config);
    
    assert.strictEqual(typeof result, 'object');
    assert(Array.isArray(result.values));
    assert(Array.isArray(result.combos));
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.values.length, close.length);
    
    // Should match single calculation
    const singleResult = wasm.er_js(close, 5);
    assertArrayClose(
        new Float64Array(result.values),
        singleResult,
        1e-10,
        "Batch single period differs from single calculation"
    );
});

test('ER batch - multiple periods', () => {
    // Test batch operation with multiple periods
    const close = new Float64Array(testData.close.slice(0, 100)); // Use subset for speed
    
    const config = {
        period_range: [5, 15, 5] // 3 periods: 5, 10, 15
    };
    
    const result = wasm.er_batch(close, config);
    
    assert.strictEqual(result.rows, 3);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.values.length, 3 * close.length);
    assert.strictEqual(result.combos.length, 3);
    
    // Verify each row matches individual calculation
    const periods = [5, 10, 15];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * close.length;
        const rowEnd = rowStart + close.length;
        const rowData = new Float64Array(result.values.slice(rowStart, rowEnd));
        
        const singleResult = wasm.er_js(close, periods[i]);
        assertArrayClose(
            rowData,
            singleResult,
            1e-10,
            `Batch row ${i} (period=${periods[i]}) differs from single calculation`
        );
    }
});

test('ER batch - fast API', () => {
    // Test batch fast API
    const close = new Float64Array(testData.close.slice(0, 100));
    const len = close.length;
    
    // Calculate output size: 3 periods × data length
    const periods = [5, 10, 15];
    const outputLen = periods.length * len;
    
    // Allocate buffers
    const inPtr = wasm.er_alloc(len);
    const outPtr = wasm.er_alloc(outputLen);
    
    try {
        const memory = new Float64Array(wasm.__wasm.memory.buffer);
        const inOffset = inPtr / 8;
        const outOffset = outPtr / 8;
        
        // Copy data to input buffer
        memory.set(close, inOffset);
        
        // Run batch computation
        const rows = wasm.er_batch_into(inPtr, outPtr, len, 5, 15, 5);
        
        assert.strictEqual(rows, 3);
        
        // Extract and verify output
        const result = new Float64Array(memory.buffer, outPtr, outputLen);
        const resultCopy = new Float64Array(result);
        
        // Compare with safe batch API
        const config = { period_range: [5, 15, 5] };
        const safeResult = wasm.er_batch(close, config);
        
        assertArrayClose(
            resultCopy,
            new Float64Array(safeResult.values),
            1e-10,
            "Fast batch API differs from safe batch API"
        );
        
    } finally {
        wasm.er_free(inPtr, len);
        wasm.er_free(outPtr, outputLen);
    }
});