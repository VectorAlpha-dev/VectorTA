/**
 * WASM binding tests for ZLEMA indicator.
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
        // No need to call default() for ES modules
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('ZLEMA partial params', () => {
    const data = testData.close;
    
    // Test with period 14
    const result = wasm.zlema_js(data, 14);
    assert.strictEqual(result.length, data.length, 'Output length should match input');
});

test('ZLEMA accuracy', () => {
    const data = testData.close;
    const expected = EXPECTED_OUTPUTS.zlema;
    
    const result = wasm.zlema_js(data, expected.defaultParams.period);
    
    assert.strictEqual(result.length, data.length, 'Output length should match input');
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(last5, expected.last5Values, 1e-1, 'ZLEMA last 5 values mismatch');
});

test('ZLEMA zero period', () => {
    const data = [10.0, 20.0, 30.0];
    
    assert.throws(() => {
        wasm.zlema_js(data, 0);
    }, /Invalid period/, 'Should throw error for zero period');
});

test('ZLEMA period exceeds length', () => {
    const data = [10.0, 20.0, 30.0];
    
    assert.throws(() => {
        wasm.zlema_js(data, 10);
    }, /Invalid period/, 'Should throw error when period exceeds data length');
});

test('ZLEMA very small dataset', () => {
    const data = [42.0];
    
    assert.throws(() => {
        wasm.zlema_js(data, 14);
    }, /Invalid period/, 'Should throw error for insufficient data');
});

test('ZLEMA reinput', () => {
    const data = testData.close;
    
    // First pass with period 21
    const firstResult = wasm.zlema_js(data, 21);
    
    // Second pass with period 14 on first result
    const secondResult = wasm.zlema_js(firstResult, 14);
    
    assert.strictEqual(secondResult.length, firstResult.length, 'Output length should match input');
    
    // Find first non-NaN index in second result
    let firstValidIdx = 0;
    while (firstValidIdx < secondResult.length && isNaN(secondResult[firstValidIdx])) {
        firstValidIdx++;
    }
    
    // Check that values after the first valid index are finite
    for (let idx = firstValidIdx; idx < secondResult.length; idx++) {
        assert(isFinite(secondResult[idx]), `NaN found at index ${idx} after first valid at ${firstValidIdx}`);
    }
});

test('ZLEMA nan handling', () => {
    const data = testData.close;
    
    const result = wasm.zlema_js(data, 14);
    assert.strictEqual(result.length, data.length, 'Output length should match input');
    
    // Check that values after warmup period don't have NaN
    if (result.length > 20) {
        for (let i = 20; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('ZLEMA batch processing', () => {
    const data = testData.close;
    
    const result = wasm.zlema_batch_js(
        data,
        14,  // period_start
        40,  // period_end
        1    // period_step
    );
    
    // Result should be flattened 2D array (27 rows * data length)
    assert.strictEqual(result.length, 27 * data.length, 
        'Batch result should have 27 rows × data length values');
    
    // Get metadata to verify periods
    const metadata = wasm.zlema_batch_metadata_js(14, 40, 1);
    assert.strictEqual(metadata.length, 27, 'Metadata should have 27 periods');
    assert.strictEqual(metadata[0], 14, 'First period should be 14');
    assert.strictEqual(metadata[26], 40, 'Last period should be 40');
    
    // Check that first row (period 14) matches single ZLEMA calculation
    const single_zlema = wasm.zlema_js(data, 14);
    const first_row = result.slice(0, data.length);
    assertArrayClose(first_row, single_zlema, 1e-9, 'Batch period 14 row should match single calculation');
});

test('ZLEMA all nan input', () => {
    const data = [NaN, NaN, NaN];
    
    assert.throws(() => {
        wasm.zlema_js(data, 2);
    }, /All values are NaN/, 'Should throw error for all NaN input');
});

test('ZLEMA batch metadata', () => {
    // Test metadata generation
    const metadata = wasm.zlema_batch_metadata_js(10, 20, 2);
    
    // Should have periods: 10, 12, 14, 16, 18, 20
    assert.strictEqual(metadata.length, 6, 'Metadata should have 6 periods');
    assert.deepStrictEqual(Array.from(metadata), [10, 12, 14, 16, 18, 20], 
        'Metadata should contain correct periods');
});

test('ZLEMA batch single period', () => {
    const data = testData.close;
    
    // Test batch with single period (start == end)
    const result = wasm.zlema_batch_js(
        data,
        14,  // period_start
        14,  // period_end
        0    // period_step (ignored when start == end)
    );
    
    // Should have only 1 row
    assert.strictEqual(result.length, data.length, 
        'Single period batch should have 1 row × data length values');
    
    // Should match single ZLEMA calculation
    const single_zlema = wasm.zlema_js(data, 14);
    assertArrayClose(result, single_zlema, 1e-9, 'Single period batch should match single calculation');
});

test('ZLEMA empty input', () => {
    const data = [];
    
    assert.throws(() => {
        wasm.zlema_js(data, 14);
    }, /All values are NaN/, 'Should throw error for empty input');
});

test('ZLEMA fast API basic', () => {
    const data = testData.close;
    const period = 14;
    
    // Allocate memory
    const inPtr = wasm.zlema_alloc(data.length);
    const outPtr = wasm.zlema_alloc(data.length);
    
    try {
        // Copy data to WASM memory
        const memory = new Float64Array(wasm.__wasm.memory.buffer);
        memory.set(data, inPtr / 8);
        
        // Compute
        wasm.zlema_into(inPtr, outPtr, data.length, period);
        
        // Read results
        const result = Array.from(memory.slice(outPtr / 8, outPtr / 8 + data.length));
        
        // Compare with safe API
        const safeResult = wasm.zlema_js(data, period);
        assertArrayClose(result, safeResult, 1e-9, 'Fast API should match safe API');
    } finally {
        // Clean up
        wasm.zlema_free(inPtr, data.length);
        wasm.zlema_free(outPtr, data.length);
    }
});

test('ZLEMA fast API in-place (aliasing)', () => {
    const data = testData.close;
    const period = 14;
    
    // Allocate memory
    const ptr = wasm.zlema_alloc(data.length);
    
    try {
        // Copy data to WASM memory
        const memory = new Float64Array(wasm.__wasm.memory.buffer);
        memory.set(data, ptr / 8);
        
        // Compute in-place (same pointer for input and output)
        wasm.zlema_into(ptr, ptr, data.length, period);
        
        // Read results
        const result = Array.from(memory.slice(ptr / 8, ptr / 8 + data.length));
        
        // Compare with safe API
        const safeResult = wasm.zlema_js(data, period);
        assertArrayClose(result, safeResult, 1e-9, 'In-place computation should match safe API');
    } finally {
        // Clean up
        wasm.zlema_free(ptr, data.length);
    }
});

test('ZLEMA unified batch API', () => {
    const data = testData.close;
    
    // Test with config object
    const config = {
        period_range: [14, 20, 2]  // 14, 16, 18, 20
    };
    
    const result = wasm.zlema_batch(data, config);
    
    assert(result.values, 'Result should have values array');
    assert(result.combos, 'Result should have combos array');
    assert.strictEqual(result.rows, 4, 'Should have 4 rows (periods)');
    assert.strictEqual(result.cols, data.length, 'Columns should match data length');
    assert.strictEqual(result.values.length, 4 * data.length, 'Values should be flattened 2D array');
    
    // Check that first row matches single calculation
    const firstRowValues = result.values.slice(0, data.length);
    const singleResult = wasm.zlema_js(data, 14);
    assertArrayClose(firstRowValues, singleResult, 1e-9, 'First row should match period=14 calculation');
});

test('ZLEMA batch fast API', () => {
    const data = testData.close;
    const periodStart = 10;
    const periodEnd = 20;
    const periodStep = 5;  // 10, 15, 20 = 3 rows
    
    // Allocate memory
    const inPtr = wasm.zlema_alloc(data.length);
    const outPtr = wasm.zlema_alloc(3 * data.length);  // 3 rows
    
    try {
        // Copy data to WASM memory
        const memory = new Float64Array(wasm.__wasm.memory.buffer);
        memory.set(data, inPtr / 8);
        
        // Compute batch
        const rows = wasm.zlema_batch_into(inPtr, outPtr, data.length, periodStart, periodEnd, periodStep);
        assert.strictEqual(rows, 3, 'Should return 3 rows');
        
        // Read results
        const result = Array.from(memory.slice(outPtr / 8, outPtr / 8 + 3 * data.length));
        
        // Check first row (period=10)
        const firstRow = result.slice(0, data.length);
        const expected10 = wasm.zlema_js(data, 10);
        assertArrayClose(firstRow, expected10, 1e-9, 'First batch row should match period=10');
    } finally {
        // Clean up
        wasm.zlema_free(inPtr, data.length);
        wasm.zlema_free(outPtr, 3 * data.length);
    }
});

test.after(() => {
    console.log('ZLEMA WASM tests completed');
});