/**
 * WASM binding tests for CORREL_HL indicator.
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

test('CORREL_HL partial params', () => {
    // Test with default parameters - mirrors check_correl_hl_partial_params
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.correl_hl_js(high, low, 9);
    assert.strictEqual(result.length, high.length);
});

test('CORREL_HL accuracy', async () => {
    // Test CORREL_HL matches expected values from Rust tests - mirrors check_correl_hl_accuracy
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const expected = EXPECTED_OUTPUTS.correl_hl;
    
    const result = wasm.correl_hl_js(
        high,
        low,
        expected.default_params.period
    );
    
    assert.strictEqual(result.length, high.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last_5_values,
        1e-7,  // CORREL_HL uses 1e-7 tolerance in Rust tests
        "CORREL_HL last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('correl_hl', result, ['high', 'low'], expected.default_params);
});

test('CORREL_HL from candles', () => {
    // Test CORREL_HL with candle data - mirrors check_correl_hl_from_candles
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.correl_hl_js(high, low, 9);
    assert.strictEqual(result.length, high.length);
});

test('CORREL_HL zero period', () => {
    // Test CORREL_HL fails with zero period - mirrors check_correl_hl_zero_period
    const high = new Float64Array([1.0, 2.0, 3.0]);
    const low = new Float64Array([1.0, 2.0, 3.0]);
    
    assert.throws(() => {
        wasm.correl_hl_js(high, low, 0);
    }, /Invalid period/);
});

test('CORREL_HL period exceeds length', () => {
    // Test CORREL_HL fails when period exceeds data length - mirrors check_correl_hl_period_exceeds_length
    const high = new Float64Array([1.0, 2.0, 3.0]);
    const low = new Float64Array([1.0, 2.0, 3.0]);
    
    assert.throws(() => {
        wasm.correl_hl_js(high, low, 10);
    }, /Invalid period/);
});

test('CORREL_HL data length mismatch', () => {
    // Test CORREL_HL fails on length mismatch - mirrors check_correl_hl_data_length_mismatch
    const high = new Float64Array([1.0, 2.0, 3.0]);
    const low = new Float64Array([1.0, 2.0]);
    
    assert.throws(() => {
        wasm.correl_hl_js(high, low, 2);
    }, /Data length mismatch/);
});

test('CORREL_HL all NaN', () => {
    // Test CORREL_HL fails on all NaN - mirrors check_correl_hl_all_nan
    const high = new Float64Array([NaN, NaN, NaN]);
    const low = new Float64Array([NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.correl_hl_js(high, low, 2);
    }, /All values are NaN/);
});

test('CORREL_HL empty input', () => {
    // Test CORREL_HL fails with empty input
    const high = new Float64Array([]);
    const low = new Float64Array([]);
    
    assert.throws(() => {
        wasm.correl_hl_js(high, low, 9);
    }, /Empty data/);
});

test('CORREL_HL reinput', () => {
    // Test CORREL_HL reinput - mirrors check_correl_hl_reinput
    const high = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    const low = new Float64Array([0.5, 1.0, 1.5, 2.0, 2.5]);
    
    const firstResult = wasm.correl_hl_js(high, low, 2);
    const secondResult = wasm.correl_hl_js(firstResult, low, 2);
    assert.strictEqual(secondResult.length, low.length);
});

test('CORREL_HL NaN handling', () => {
    // Test CORREL_HL handles NaN values correctly
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.correl_hl_js(high, low, 9);
    assert.strictEqual(result.length, high.length);
    
    // First period values should be NaN
    assertAllNaN(result.slice(0, 8));
    
    // After warmup period, should have valid values
    assertNoNaN(result.slice(8));
});

test('CORREL_HL fast API (in-place)', () => {
    // Test fast API with in-place operation (aliasing)
    const high = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    const low = new Float64Array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]);
    const period = 3;
    
    // Get expected results from safe API
    const expected = wasm.correl_hl_js(high, low, period);
    
    // Allocate output buffer
    const len = high.length;
    const outPtr = wasm.correl_hl_alloc(len);
    
    try {
        // First, test normal operation (no aliasing)
        wasm.correl_hl_into(high, low, outPtr, len, period);
        
        // Read results from WASM memory
        const memory = new Float64Array(wasm.memory.buffer, outPtr, len);
        const result = new Float64Array(memory);
        
        // Should match safe API
        assertArrayClose(result, expected, 1e-10, "Fast API mismatch");
        
        // Now test aliasing - use high pointer as output
        const highCopy = new Float64Array(high);
        wasm.correl_hl_into(highCopy, low, highCopy, len, period);
        
        // Should still produce same results
        assertArrayClose(highCopy, expected, 1e-10, "Fast API aliasing mismatch");
    } finally {
        wasm.correl_hl_free(outPtr, len);
    }
});

test('CORREL_HL batch single period', () => {
    // Test CORREL_HL batch with single period
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    
    const config = {
        period_range: [9, 9, 1]
    };
    
    const result = wasm.correl_hl_batch(high, low, config);
    
    // Check structure
    assert(result.values, 'Missing values in batch result');
    assert(result.periods, 'Missing periods in batch result');
    assert(result.rows, 'Missing rows in batch result');
    assert(result.cols, 'Missing cols in batch result');
    
    // Check dimensions
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, 100);
    assert.strictEqual(result.values.length, 100);
    assert.strictEqual(result.periods.length, 1);
    assert.strictEqual(result.periods[0], 9);
    
    // Compare with single calculation
    const singleResult = wasm.correl_hl_js(high, low, 9);
    assertArrayClose(result.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('CORREL_HL batch multiple periods', () => {
    // Test CORREL_HL batch with multiple periods
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    
    const config = {
        period_range: [5, 15, 5]  // periods: 5, 10, 15
    };
    
    const result = wasm.correl_hl_batch(high, low, config);
    
    // Check dimensions
    assert.strictEqual(result.rows, 3);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.values.length, 150);  // 3 * 50
    assert.strictEqual(result.periods.length, 3);
    assert.deepStrictEqual(Array.from(result.periods), [5, 10, 15]);
    
    // Verify each row has correct warmup period
    for (let i = 0; i < 3; i++) {
        const period = result.periods[i];
        const rowStart = i * 50;
        const rowEnd = rowStart + 50;
        const row = result.values.slice(rowStart, rowEnd);
        
        // First period-1 values should be NaN
        assertAllNaN(row.slice(0, period - 1));
        // After warmup should have valid values
        assertNoNaN(row.slice(period - 1));
    }
});

test('CORREL_HL batch fast API', () => {
    // Test batch fast API
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const len = high.length;
    
    // Define batch parameters
    const periodStart = 5;
    const periodEnd = 15;
    const periodStep = 5;
    const expectedRows = 3; // periods: 5, 10, 15
    
    // Allocate output buffer
    const outPtr = wasm.correl_hl_alloc(len * expectedRows);
    
    try {
        const rows = wasm.correl_hl_batch_into(
            high, low, outPtr, len,
            periodStart, periodEnd, periodStep
        );
        
        assert.strictEqual(rows, expectedRows);
        
        // Read results from WASM memory
        const memory = new Float64Array(wasm.memory.buffer, outPtr, len * rows);
        const result = new Float64Array(memory);
        
        // Compare with safe batch API
        const config = {
            period_range: [periodStart, periodEnd, periodStep]
        };
        const safeBatchResult = wasm.correl_hl_batch(high, low, config);
        
        assertArrayClose(
            result,
            safeBatchResult.values,
            1e-10,
            "Fast batch API mismatch"
        );
    } finally {
        wasm.correl_hl_free(outPtr, len * expectedRows);
    }
});