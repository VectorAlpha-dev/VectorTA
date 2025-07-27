/**
 * WASM binding tests for CHOP indicator.
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

test('CHOP basic functionality', () => {
    // Test basic CHOP functionality
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    // Test with default parameters
    const result = wasm.chop_js(high, low, close, 14, 100.0, 1);
    
    assert.strictEqual(result.length, close.length);
    assert.ok(result instanceof Float64Array);
    
    // First period-1 values should be NaN
    for (let i = 0; i < 13; i++) {
        assert.ok(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
    
    // Values after warmup should be finite
    const validValues = result.slice(13).filter(v => !isNaN(v));
    assert.ok(validValues.length > 0, "Should have valid values after warmup");
    
    // CHOP values should be in reasonable range (0-100 typically)
    for (const val of validValues) {
        assert.ok(val >= 0, `CHOP value ${val} should be non-negative`);
        assert.ok(val <= 200, `CHOP value ${val} seems too large`);
    }
});

test('CHOP with custom parameters', () => {
    // Test CHOP with custom parameters
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    // Test with custom parameters
    const period = 20;
    const scalar = 50.0;
    const drift = 2;
    
    const result = wasm.chop_js(high, low, close, period, scalar, drift);
    
    assert.strictEqual(result.length, close.length);
    
    // First period-1 values should be NaN
    for (let i = 0; i < period - 1; i++) {
        assert.ok(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
    
    // Check that parameters affect the result
    const resultDefault = wasm.chop_js(high, low, close, 14, 100.0, 1);
    let differentCount = 0;
    for (let i = period; i < result.length; i++) {
        if (!isNaN(result[i]) && !isNaN(resultDefault[i]) && 
            Math.abs(result[i] - resultDefault[i]) > 1e-10) {
            differentCount++;
        }
    }
    assert.ok(differentCount > 0, "Custom parameters should produce different results");
});

test('CHOP edge cases', () => {
    // Empty arrays
    assert.throws(() => {
        wasm.chop_js(new Float64Array([]), new Float64Array([]), new Float64Array([]), 14, 100.0, 1);
    }, /EmptyData|empty/i);
    
    // Period exceeds data length
    const smallData = new Float64Array([1.0, 2.0, 3.0]);
    assert.throws(() => {
        wasm.chop_js(smallData, smallData, smallData, 10, 100.0, 1);
    }, /Invalid period/);
    
    // Zero period
    const data = new Float64Array(100).fill(50.0);
    assert.throws(() => {
        wasm.chop_js(data, data, data, 0, 100.0, 1);
    }, /Invalid period/);
});

test('CHOP fast API (in-place)', () => {
    // Test the fast API with in-place operation
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Allocate buffers for inputs and output
    const highPtr = wasm.chop_alloc(len);
    const lowPtr = wasm.chop_alloc(len);
    const closePtr = wasm.chop_alloc(len);
    const outPtr = wasm.chop_alloc(len);
    
    try {
        // Copy data to WASM memory
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        const closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        
        highView.set(high);
        lowView.set(low);
        closeView.set(close);
        
        // Test normal operation first
        wasm.chop_into(
            highPtr,
            lowPtr,
            closePtr,
            outPtr,
            len,
            14, 100.0, 1
        );
        
        // Read result
        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        
        // Compare with safe API
        const expected = wasm.chop_js(high, low, close, 14, 100.0, 1);
        assertArrayClose(result, expected, 1e-10, "Fast API should match safe API");
        
        // Test in-place operation (output = close)
        wasm.chop_into(
            highPtr,
            lowPtr,
            closePtr,
            closePtr,  // in-place on close
            len,
            14, 100.0, 1
        );
        
        // Verify in-place result
        const closeResult = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        assertArrayClose(closeResult, expected, 1e-10, "In-place operation should match safe API");
        
    } finally {
        wasm.chop_free(highPtr, len);
        wasm.chop_free(lowPtr, len);
        wasm.chop_free(closePtr, len);
        wasm.chop_free(outPtr, len);
    }
});

test('CHOP batch processing', () => {
    // Test batch processing
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const config = {
        period_range: [10, 20, 5],    // 10, 15, 20
        scalar_range: [50.0, 100.0, 50.0], // 50, 100
        drift_range: [1, 2, 1]        // 1, 2
    };
    
    const result = wasm.chop_batch(high, low, close, config);
    
    assert.ok(result.values);
    assert.ok(result.combos);
    assert.ok(result.rows);
    assert.ok(result.cols);
    
    // Should have 3 * 2 * 2 = 12 combinations
    assert.strictEqual(result.rows, 12);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.combos.length, 12);
    assert.strictEqual(result.values.length, 12 * close.length);
    
    // Verify parameter combinations
    const combos = result.combos;
    const periods = combos.map(c => c.period).filter((v, i, a) => a.indexOf(v) === i);
    const scalars = combos.map(c => c.scalar).filter((v, i, a) => a.indexOf(v) === i);
    const drifts = combos.map(c => c.drift).filter((v, i, a) => a.indexOf(v) === i);
    
    assert.deepStrictEqual(periods.sort((a,b) => a-b), [10, 15, 20]);
    assert.deepStrictEqual(scalars.sort((a,b) => a-b), [50.0, 100.0]);
    assert.deepStrictEqual(drifts.sort((a,b) => a-b), [1, 2]);
});

test('CHOP batch fast API', () => {
    // Test fast batch API
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    const len = close.length;
    
    // Calculate expected number of combinations
    const expectedRows = 3 * 2 * 2; // period * scalar * drift combinations
    const totalSize = expectedRows * len;
    
    // Allocate input buffers
    const highPtr = wasm.chop_alloc(len);
    const lowPtr = wasm.chop_alloc(len);
    const closePtr = wasm.chop_alloc(len);
    
    // Allocate output buffer
    const outPtr = wasm.chop_alloc(totalSize);
    
    try {
        // Copy data to WASM memory
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        const closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        
        highView.set(high);
        lowView.set(low);
        closeView.set(close);
        
        const rows = wasm.chop_batch_into(
            highPtr,
            lowPtr,
            closePtr,
            outPtr,
            len,
            10, 20, 5,      // period range
            50.0, 100.0, 50.0, // scalar range
            1, 2, 1         // drift range
        );
        
        assert.strictEqual(rows, expectedRows);
        
        // Read result
        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, totalSize);
        
        // Verify first row matches single calculation
        const firstRow = result.slice(0, len);
        const expected = wasm.chop_js(high, low, close, 10, 50.0, 1);
        assertArrayClose(firstRow, expected, 1e-10, "First batch row should match single calculation");
        
    } finally {
        wasm.chop_free(highPtr, len);
        wasm.chop_free(lowPtr, len);
        wasm.chop_free(closePtr, len);
        wasm.chop_free(outPtr, totalSize);
    }
});

test('CHOP accuracy', () => {
    // Test CHOP accuracy against expected values
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.chop_js(high, low, close, 14, 100.0, 1);
    
    // Expected last 5 values from Python test
    const expectedLast5 = [
        49.98214330294626,
        48.90450693742312,
        46.63648608318844,
        46.19823574588033,
        56.22876423352909,
    ];
    
    const last5 = result.slice(-5);
    assertArrayClose(last5, expectedLast5, 1e-4, "CHOP last 5 values should match expected");
});

test('CHOP memory management', () => {
    // Test allocation and deallocation
    const sizes = [10, 100, 1000, 10000];
    
    for (const size of sizes) {
        const ptr = wasm.chop_alloc(size);
        assert.ok(ptr !== 0, `Should allocate memory for size ${size}`);
        
        // Write some data
        const arr = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < size; i++) {
            arr[i] = i * 1.5;
        }
        
        // Free memory
        assert.doesNotThrow(() => {
            wasm.chop_free(ptr, size);
        }, `Should free memory for size ${size}`);
    }
    
    // Test null pointer in free
    assert.doesNotThrow(() => {
        wasm.chop_free(0, 100);
    }, "Should handle null pointer in free");
});

test('CHOP null pointer handling', () => {
    // Test null pointer errors
    const len = 100;
    
    assert.throws(() => {
        wasm.chop_into(0, 0, 0, 0, len, 14, 100.0, 1);
    }, /Null pointer/);
    
    const validPtr = wasm.chop_alloc(len);
    try {
        assert.throws(() => {
            wasm.chop_into(validPtr, 0, validPtr, validPtr, len, 14, 100.0, 1);
        }, /Null pointer/);
    } finally {
        wasm.chop_free(validPtr, len);
    }
});

test('CHOP NaN handling', () => {
    // Test with NaN values in input
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    // Insert some NaN values
    high[10] = NaN;
    high[11] = NaN;
    low[15] = NaN;
    close[5] = NaN;
    
    const result = wasm.chop_js(high, low, close, 14, 100.0, 1);
    
    assert.strictEqual(result.length, close.length);
    
    // Should handle NaN values gracefully
    // Values after sufficient non-NaN data should be valid
    const validFromIndex = 30; // After NaN values and warmup
    const validValues = result.slice(validFromIndex).filter(v => !isNaN(v));
    assert.ok(validValues.length > 0, "Should have valid values after NaN inputs");
});