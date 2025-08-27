/**
 * WASM binding tests for VOSS indicator.
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
    assertNoNaN
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

test('VOSS partial params', () => {
    // Test with default parameters - mirrors check_voss_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.voss_js(close, 20, 3, 0.25);
    // VOSS returns object with voss and filt arrays
    assert(result.voss, 'Should have voss array');
    assert(result.filt, 'Should have filt array');
    assert.strictEqual(result.voss.length, close.length);
    assert.strictEqual(result.filt.length, close.length);
});

test('VOSS accuracy', async () => {
    // Test VOSS matches expected values from Rust tests - mirrors check_voss_accuracy
    const close = new Float64Array(testData.close);
    
    const result = wasm.voss_js(close, 20, 3, 0.25);
    
    assert(result.voss, 'Should have voss array');
    assert(result.filt, 'Should have filt array');
    assert.strictEqual(result.voss.length, close.length);
    assert.strictEqual(result.filt.length, close.length);
    
    // Get results from object
    const voss = result.voss;
    const filt = result.filt;
    
    // Expected values from Rust tests
    const expectedVossLast5 = [
        -290.430249544605,
        -269.74949153549596,
        -241.08179139844515,
        -149.2113276943419,
        -138.60361772412466,
    ];
    const expectedFiltLast5 = [
        -228.0283989610523,
        -257.79056527053103,
        -270.3220395771822,
        -257.4282859799144,
        -235.78021136041997,
    ];
    
    // Check last 5 values match expected
    assertArrayClose(
        voss.slice(-5),
        expectedVossLast5,
        1e-1,  // Using 1e-1 like in Rust tests
        "VOSS last 5 values mismatch"
    );
    
    assertArrayClose(
        filt.slice(-5),
        expectedFiltLast5,
        1e-1,  // Using 1e-1 like in Rust tests
        "Filt last 5 values mismatch"
    );
});

test('VOSS default candles', () => {
    // Test VOSS with default parameters - mirrors check_voss_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.voss_js(close, 20, 3, 0.25);
    assert(result.voss, 'Should have voss array');
    assert(result.filt, 'Should have filt array');
    assert.strictEqual(result.voss.length, close.length);
    assert.strictEqual(result.filt.length, close.length);
});

test('VOSS zero period', () => {
    // Test VOSS fails with zero period - mirrors check_voss_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.voss_js(inputData, 0, 3, 0.25);
    }, /Invalid period/);
});

test('VOSS period exceeds length', () => {
    // Test VOSS fails when period exceeds data length - mirrors check_voss_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.voss_js(dataSmall, 10, 3, 0.25);
    }, /Invalid period/);
});

test('VOSS very small dataset', () => {
    // Test VOSS fails with insufficient data - mirrors check_voss_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.voss_js(singlePoint, 20, 3, 0.25);
    }, /Invalid period|Not enough valid data/);
});

test('VOSS empty input', () => {
    // Test VOSS fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.voss_js(empty, 20, 3, 0.25);
    }, /Empty|empty/);
});

test('VOSS reinput', () => {
    // Test VOSS applied twice (re-input) - mirrors check_voss_reinput
    const close = new Float64Array(testData.close);
    
    // First pass
    const firstResult = wasm.voss_js(close, 10, 2, 0.2);
    assert(firstResult.voss, 'Should have voss array');
    assert(firstResult.filt, 'Should have filt array');
    assert.strictEqual(firstResult.voss.length, close.length);
    assert.strictEqual(firstResult.filt.length, close.length);
    
    // Extract first voss output for second pass
    const firstVoss = firstResult.voss;
    
    // Second pass - apply VOSS to VOSS output
    const secondResult = wasm.voss_js(firstVoss, 10, 2, 0.2);
    assert(secondResult.voss, 'Should have voss array');
    assert(secondResult.filt, 'Should have filt array');
    assert.strictEqual(secondResult.voss.length, firstVoss.length);
    assert.strictEqual(secondResult.filt.length, firstVoss.length);
});

test('VOSS all NaN input', () => {
    // Test VOSS with all NaN values
    const allNan = new Float64Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.voss_js(allNan, 20, 3, 0.25);
    }, /All values are NaN|All NaN/);
});

test('VOSS fast API', () => {
    // Test fast/unsafe API
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Allocate buffers
    const inPtr = wasm.voss_alloc(len);
    const vossPtr = wasm.voss_alloc(len);
    const filtPtr = wasm.voss_alloc(len);
    
    assert(inPtr !== 0, 'Failed to allocate input memory');
    assert(vossPtr !== 0, 'Failed to allocate voss memory');
    assert(filtPtr !== 0, 'Failed to allocate filt memory');
    
    try {
        // Copy input data to allocated memory
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inView.set(close);
        
        // Compute
        wasm.voss_into(inPtr, vossPtr, filtPtr, len, 20, 3, 0.25);
        
        // Read results immediately and copy to regular arrays
        const vossView = new Float64Array(wasm.__wasm.memory.buffer, vossPtr, len);
        const filtView = new Float64Array(wasm.__wasm.memory.buffer, filtPtr, len);
        const voss = Array.from(vossView);  // Copy to avoid detachment
        const filt = Array.from(filtView);  // Copy to avoid detachment
        
        // Compare with safe API (might grow memory)
        const safeResult = wasm.voss_js(close, 20, 3, 0.25);
        
        assert.strictEqual(voss.length, len, 'Voss array has wrong length');
        assert.strictEqual(filt.length, len, 'Filt array has wrong length');
        const safeVoss = safeResult.voss;
        const safeFilt = safeResult.filt;
        
        assert(Array.isArray(safeVoss) || safeVoss instanceof Float64Array, 'safeVoss should be an array');
        assert(Array.isArray(safeFilt) || safeFilt instanceof Float64Array, 'safeFilt should be an array');
        assert.strictEqual(safeVoss.length, len, 'safeVoss has wrong length');
        assert.strictEqual(safeFilt.length, len, 'safeFilt has wrong length');
        
        // Debug info
        if (voss.length === 0) {
            console.log('voss is empty!');
            console.log('vossPtr:', vossPtr);
            console.log('wasm.__wasm.memory.buffer.byteLength:', wasm.__wasm.memory.buffer.byteLength);
        }
        
        assertArrayClose(voss, safeVoss, 1e-10, "Fast API voss mismatch");
        assertArrayClose(filt, safeFilt, 1e-10, "Fast API filt mismatch");
    } finally {
        // Free memory
        wasm.voss_free(inPtr, len);
        wasm.voss_free(vossPtr, len);
        wasm.voss_free(filtPtr, len);
    }
});

test('VOSS fast API with aliasing', () => {
    // Test fast API with in-place operation (aliasing)
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Create a copy for comparison
    const closeCopy = new Float64Array(close);
    
    // Allocate one buffer and use it for both input and voss output
    const ptr = wasm.voss_alloc(len);
    const filtPtr = wasm.voss_alloc(len);
    
    try {
        // Copy input data to allocated memory
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        memView.set(close);
        
        // Compute with aliasing (input and voss output share same memory)
        wasm.voss_into(ptr, ptr, filtPtr, len, 20, 3, 0.25);
        
        // Get results
        const voss = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        const filt = new Float64Array(wasm.__wasm.memory.buffer, filtPtr, len);
        
        // Verify against safe API
        const safeResult = wasm.voss_js(closeCopy, 20, 3, 0.25);
        const safeVoss = safeResult.voss;
        const safeFilt = safeResult.filt;
        
        assertArrayClose(voss, safeVoss, 1e-10, "Aliased voss mismatch");
        assertArrayClose(filt, safeFilt, 1e-10, "Aliased filt mismatch");
    } finally {
        wasm.voss_free(ptr, len);
        wasm.voss_free(filtPtr, len);
    }
});

test('VOSS batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    // Single parameter set
    const config = {
        period_range: [20, 20, 0],
        predict_range: [3, 3, 0],
        bandwidth_range: [0.25, 0.25, 0.0]
    };
    
    const result = wasm.voss_batch(close, config);
    
    // Should match single calculation
    const singleResult = wasm.voss_js(close, 20, 3, 0.25);
    
    assert(result.voss, 'Batch result should have voss');
    assert(result.filt, 'Batch result should have filt');
    assert(result.combos, 'Batch result should have combos');
    assert(result.rows, 'Batch result should have rows');
    assert(result.cols, 'Batch result should have cols');
    
    // Check shapes
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.voss.length, close.length);
    assert.strictEqual(result.filt.length, close.length);
    
    // Check values match single calculation
    const singleVoss = singleResult.voss;
    const singleFilt = singleResult.filt;
    
    assertArrayClose(result.voss, singleVoss, 1e-10);
    assertArrayClose(result.filt, singleFilt, 1e-10);
});

test('VOSS batch multiple periods', () => {
    // Test batch with multiple period values
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    
    // Multiple periods: 10, 12, 14
    const config = {
        period_range: [10, 14, 2],
        predict_range: [3, 3, 0],
        bandwidth_range: [0.25, 0.25, 0.0]
    };
    
    const result = wasm.voss_batch(close, config);
    
    // Should have 3 rows
    assert.strictEqual(result.rows, 3);
    assert.strictEqual(result.cols, 100);
    assert.strictEqual(result.voss.length, 300);
    assert.strictEqual(result.filt.length, 300);
    assert.strictEqual(result.combos.length, 3);
    
    // Verify each row matches individual calculation
    const periods = [10, 12, 14];
    for (let i = 0; i < 3; i++) {
        const singleResult = wasm.voss_js(close, periods[i], 3, 0.25);
        const singleVoss = singleResult.voss;
        const singleFilt = singleResult.filt;
        
        const rowVoss = result.voss.slice(i * 100, (i + 1) * 100);
        const rowFilt = result.filt.slice(i * 100, (i + 1) * 100);
        
        assertArrayClose(rowVoss, singleVoss, 1e-10);
        assertArrayClose(rowFilt, singleFilt, 1e-10);
    }
});

test('VOSS batch full parameter sweep', () => {
    // Test full parameter sweep
    const close = new Float64Array(testData.close.slice(0, 50)); // Small dataset for speed
    
    const config = {
        period_range: [10, 12, 2],      // 2 periods
        predict_range: [2, 3, 1],        // 2 predicts
        bandwidth_range: [0.2, 0.3, 0.1] // 2 bandwidths
    };
    
    const result = wasm.voss_batch(close, config);
    
    // Should have 2 * 2 * 2 = 8 combinations
    assert.strictEqual(result.rows, 8);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.combos.length, 8);
    
    // Verify parameter combinations
    const expectedCombos = [
        [10, 2, 0.2], [10, 2, 0.3],
        [10, 3, 0.2], [10, 3, 0.3],
        [12, 2, 0.2], [12, 2, 0.3],
        [12, 3, 0.2], [12, 3, 0.3],
    ];
    
    for (let i = 0; i < 8; i++) {
        assert.strictEqual(result.combos[i].period || 20, expectedCombos[i][0]);
        assert.strictEqual(result.combos[i].predict || 3, expectedCombos[i][1]);
        assertClose(result.combos[i].bandwidth || 0.25, expectedCombos[i][2], 1e-10);
    }
});