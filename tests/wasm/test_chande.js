/**
 * WASM binding tests for Chande indicator.
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

test('Chande partial params', () => {
    // Test with default parameters - mirrors check_chande_partial_params
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.chande_js(high, low, close, 22, 3.0, 'long');
    assert.strictEqual(result.length, close.length);
});

test('Chande accuracy', async () => {
    // Test Chande matches expected values from Rust tests - mirrors check_chande_accuracy
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.chande;
    
    const result = wasm.chande_js(
        high, low, close,
        expected.defaultParams.period,
        expected.defaultParams.mult,
        expected.defaultParams.direction
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-8,
        "Chande last 5 values mismatch"
    );
    
    // First 21 values should be NaN (period - 1)
    const warmupPeriod = expected.defaultParams.period - 1;
    for (let i = 0; i < warmupPeriod; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} during warmup period`);
    }
    
    // Compare full output with Rust
    await compareWithRust('chande', result, 'candles', expected.defaultParams);
});

test('Chande zero period', () => {
    // Test Chande fails with zero period - mirrors check_chande_zero_period
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    const close = new Float64Array([8.0, 18.0, 28.0]);
    
    assert.throws(() => {
        wasm.chande_js(high, low, close, 0, 3.0, 'long');
    }, /Invalid period/);
});

test('Chande period exceeds length', () => {
    // Test Chande fails when period exceeds data length - mirrors check_chande_period_exceeds_length
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    const close = new Float64Array([8.0, 18.0, 28.0]);
    
    assert.throws(() => {
        wasm.chande_js(high, low, close, 10, 3.0, 'long');
    }, /Invalid period/);
});

test('Chande bad direction', () => {
    // Test Chande fails with bad direction - mirrors check_chande_bad_direction
    const high = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0]);
    const low = new Float64Array([5.0, 15.0, 25.0, 35.0, 45.0]);
    const close = new Float64Array([8.0, 18.0, 28.0, 38.0, 48.0]);
    
    assert.throws(() => {
        wasm.chande_js(high, low, close, 2, 3.0, 'bad');
    }, /Invalid direction/);
});

test('Chande empty input', () => {
    // Test Chande fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.chande_js(empty, empty, empty, 22, 3.0, 'long');
    }, /All values are NaN/);
});

test('Chande mismatched lengths', () => {
    // Test Chande fails with mismatched input lengths
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0]);
    const close = new Float64Array([8.0, 18.0, 28.0]);
    
    assert.throws(() => {
        wasm.chande_js(high, low, close, 2, 3.0, 'long');
    }, /Not enough valid data/);
});

test('Chande directions', () => {
    // Test Chande with different directions
    const high = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0]);
    const low = new Float64Array([5.0, 15.0, 25.0, 35.0, 45.0]);
    const close = new Float64Array([8.0, 18.0, 28.0, 38.0, 48.0]);
    
    // Test long direction
    const resultLong = wasm.chande_js(high, low, close, 3, 2.0, 'long');
    assert.strictEqual(resultLong.length, close.length);
    assert(isNaN(resultLong[0]));  // First period-1 values should be NaN
    assert(isNaN(resultLong[1]));
    
    // Test short direction
    const resultShort = wasm.chande_js(high, low, close, 3, 2.0, 'short');
    assert.strictEqual(resultShort.length, close.length);
    assert(isNaN(resultShort[0]));  // First period-1 values should be NaN
    assert(isNaN(resultShort[1]));
    
    // Results should be different for long vs short
    assert.notStrictEqual(resultLong[2], resultShort[2]);
});

test('Chande batch single params', () => {
    // Test Chande batch with single parameter set
    const high = new Float64Array(testData.high.slice(0, 100));  // Use smaller dataset for speed
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Single parameter set
    const result = wasm.chande_batch_js(
        high, low, close,
        22, 22, 0,  // period_start, period_end, period_step
        3.0, 3.0, 0.0,  // mult_start, mult_end, mult_step
        'long'
    );
    
    // Should have exactly one row
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, 100);
    assert.strictEqual(result.values.length, 100);
    
    // Values should match single calculation
    const singleResult = wasm.chande_js(high, low, close, 22, 3.0, 'long');
    assertArrayClose(
        result.values,
        singleResult,
        1e-10,
        "Batch vs single mismatch"
    );
    
    // Check parameter arrays
    assert.strictEqual(result.periods.length, 1);
    assert.strictEqual(result.periods[0], 22);
    assert.strictEqual(result.mults.length, 1);
    assert.strictEqual(result.mults[0], 3.0);
    assert.strictEqual(result.directions.length, 1);
    assert.strictEqual(result.directions[0], 'long');
});

test('Chande batch multiple params', () => {
    // Test Chande batch with multiple parameter combinations
    const high = new Float64Array(testData.high.slice(0, 50));  // Use smaller dataset for speed
    const low = new Float64Array(testData.low.slice(0, 50));
    const close = new Float64Array(testData.close.slice(0, 50));
    
    // Multiple parameters
    const result = wasm.chande_batch_js(
        high, low, close,
        10, 20, 10,      // period: 10, 20
        2.0, 3.0, 0.5,   // mult: 2.0, 2.5, 3.0
        'short'
    );
    
    // Should have 2 * 3 = 6 rows
    assert.strictEqual(result.rows, 6);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.values.length, 6 * 50);
    
    // Check parameter arrays
    assert.strictEqual(result.periods.length, 6);
    assert.strictEqual(result.mults.length, 6);
    assert.strictEqual(result.directions.length, 6);
    
    // All directions should be 'short'
    assert(result.directions.every(d => d === 'short'));
    
    // Verify each row matches individual calculation
    const expectedParams = [
        [10, 2.0], [10, 2.5], [10, 3.0],
        [20, 2.0], [20, 2.5], [20, 3.0]
    ];
    
    for (let i = 0; i < expectedParams.length; i++) {
        const [period, mult] = expectedParams[i];
        const singleResult = wasm.chande_js(high, low, close, period, mult, 'short');
        const batchRow = result.values.slice(i * 50, (i + 1) * 50);
        
        assertArrayClose(
            batchRow,
            singleResult,
            1e-10,
            `Batch row ${i} (period=${period}, mult=${mult}) mismatch`
        );
    }
});

test('Chande zero-copy API', () => {
    // Test the fast/zero-copy API
    const len = 100;
    const high = new Float64Array(testData.high.slice(0, len));
    const low = new Float64Array(testData.low.slice(0, len));
    const close = new Float64Array(testData.close.slice(0, len));
    
    // Allocate memory for inputs and output
    const highPtr = wasm.chande_alloc(len);
    const lowPtr = wasm.chande_alloc(len);
    const closePtr = wasm.chande_alloc(len);
    const outPtr = wasm.chande_alloc(len);
    
    try {
        // Copy input data to WASM memory
        const highMem = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowMem = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        const closeMem = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        highMem.set(high);
        lowMem.set(low);
        closeMem.set(close);
        
        // Compute using zero-copy API
        wasm.chande_into(
            highPtr,
            lowPtr,
            closePtr,
            outPtr,
            len,
            22,
            3.0,
            'long'
        );
        
        // Read result from pointer
        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        
        // Compare with safe API
        const expected = wasm.chande_js(high, low, close, 22, 3.0, 'long');
        assertArrayClose(
            result,
            expected,
            1e-10,
            "Zero-copy API mismatch"
        );
    } finally {
        // Clean up allocated memory
        wasm.chande_free(highPtr, len);
        wasm.chande_free(lowPtr, len);
        wasm.chande_free(closePtr, len);
        wasm.chande_free(outPtr, len);
    }
});

test('Chande zero-copy API with aliasing', () => {
    // Test zero-copy API when output pointer aliases input
    const len = 100;
    const high = new Float64Array(testData.high.slice(0, len));
    const low = new Float64Array(testData.low.slice(0, len));
    const close = new Float64Array(testData.close.slice(0, len));
    
    // Allocate memory for inputs
    const highPtr = wasm.chande_alloc(len);
    const lowPtr = wasm.chande_alloc(len);
    const closePtr = wasm.chande_alloc(len);
    
    try {
        // Copy input data to WASM memory
        const highMem = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowMem = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        const closeMem = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        highMem.set(high);
        lowMem.set(low);
        closeMem.set(close);
        
        // Compute in-place (output overwrites close)
        wasm.chande_into(
            highPtr,
            lowPtr,
            closePtr,
            closePtr,  // Same as close - aliasing!
            len,
            22,
            3.0,
            'long'
        );
        
        // Read result from close pointer (which now contains output)
        const result = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        
        // Compare with safe API
        const expected = wasm.chande_js(high, low, close, 22, 3.0, 'long');
        assertArrayClose(
            result,
            expected,
            1e-10,
            "Zero-copy API with aliasing mismatch"
        );
    } finally {
        // Clean up allocated memory
        wasm.chande_free(highPtr, len);
        wasm.chande_free(lowPtr, len);
        wasm.chande_free(closePtr, len);
    }
});

test('Chande batch zero-copy API', () => {
    // Test batch computation with zero-copy API
    const len = 50;
    const high = new Float64Array(testData.high.slice(0, len));
    const low = new Float64Array(testData.low.slice(0, len));
    const close = new Float64Array(testData.close.slice(0, len));
    
    // Parameters for 2x3 = 6 combinations
    const periodStart = 10, periodEnd = 20, periodStep = 10;
    const multStart = 2.0, multEnd = 3.0, multStep = 0.5;
    
    // Calculate expected rows
    const expectedRows = 2 * 3;  // 2 periods * 3 mults
    
    // Allocate memory for inputs and output
    const highPtr = wasm.chande_alloc(len);
    const lowPtr = wasm.chande_alloc(len);
    const closePtr = wasm.chande_alloc(len);
    const outPtr = wasm.chande_alloc(expectedRows * len);
    
    try {
        // Copy input data to WASM memory
        const highMem = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowMem = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        const closeMem = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        highMem.set(high);
        lowMem.set(low);
        closeMem.set(close);
        
        // Compute batch using zero-copy API
        const rows = wasm.chande_batch_into(
            highPtr,
            lowPtr,
            closePtr,
            outPtr,
            len,
            periodStart, periodEnd, periodStep,
            multStart, multEnd, multStep,
            'long'
        );
        
        assert.strictEqual(rows, expectedRows);
        
        // Read result from pointer
        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, rows * len);
        
        // Compare with safe batch API
        const expected = wasm.chande_batch_js(
            high, low, close,
            periodStart, periodEnd, periodStep,
            multStart, multEnd, multStep,
            'long'
        );
        
        assertArrayClose(
            result,
            expected.values,
            1e-10,
            "Batch zero-copy API mismatch"
        );
    } finally {
        // Clean up allocated memory
        wasm.chande_free(highPtr, len);
        wasm.chande_free(lowPtr, len);
        wasm.chande_free(closePtr, len);
        wasm.chande_free(outPtr, expectedRows * len);
    }
});