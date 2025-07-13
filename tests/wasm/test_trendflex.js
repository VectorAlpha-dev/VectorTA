/**
 * WASM binding tests for TrendFlex indicator.
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

// Expected outputs for TrendFlex
const EXPECTED_OUTPUTS = {
    trendflex: {
        default_params: { period: 20 },
        last_5_values: [
            -0.19724678008015128,
            -0.1238001236481444,
            -0.10515389737087717,
            -0.1149541079904878,
            -0.16006869484450567,
        ]
    }
};

test.before(async () => {
    // Load WASM module
    try {
        const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
        const importPath = process.platform === 'win32' 
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
        wasm = await import(importPath);
    } catch (error) {
        console.error('Failed to load WASM module:', error);
        throw error;
    }
    
    // Load test data
    testData = loadTestData();
});

test('trendflex_partial_params', () => {
    const close = testData.close;
    
    // Test with default period of 20
    const result = wasm.trendflex_js(close, 20);
    assert.equal(result.length, close.length);
});

test('trendflex_accuracy', () => {
    const close = testData.close;
    const expected = EXPECTED_OUTPUTS.trendflex;
    
    const result = wasm.trendflex_js(close, expected.default_params.period);
    
    assert.equal(result.length, close.length);
    
    // Check last 5 values
    const last5 = result.slice(-5);
    expected.last_5_values.forEach((expectedVal, i) => {
        assertClose(last5[i], expectedVal, 1e-8, `TrendFlex mismatch at index ${i}`);
    });
    
    // Compare with Rust
    compareWithRust('trendflex', result, 'close', expected.default_params);
});

test('trendflex_default_candles', () => {
    const close = testData.close;
    
    // Default params: period=20
    const result = wasm.trendflex_js(close, 20);
    assert.equal(result.length, close.length);
});

test('trendflex_zero_period', () => {
    const inputData = [10.0, 20.0, 30.0];
    
    assert.throws(() => {
        wasm.trendflex_js(inputData, 0);
    }, /period = 0/);
});

test('trendflex_period_exceeds_length', () => {
    const dataSmall = [10.0, 20.0, 30.0];
    
    assert.throws(() => {
        wasm.trendflex_js(dataSmall, 10);
    }, /period > data len/);
});

test('trendflex_very_small_dataset', () => {
    const singlePoint = [42.0];
    
    assert.throws(() => {
        wasm.trendflex_js(singlePoint, 9);
    }, /period > data len/);
});

test('trendflex_empty_input', () => {
    const empty = [];
    
    assert.throws(() => {
        wasm.trendflex_js(empty, 20);
    }, /No data provided/);
});

test('trendflex_reinput', () => {
    const close = testData.close;
    
    // First pass
    const firstResult = wasm.trendflex_js(close, 20);
    assert.equal(firstResult.length, close.length);
    
    // Second pass - apply TrendFlex to TrendFlex output
    const secondResult = wasm.trendflex_js(firstResult, 10);
    assert.equal(secondResult.length, firstResult.length);
    
    // Check for NaN handling after warmup
    if (secondResult.length > 240) {
        for (let i = 240; i < secondResult.length; i++) {
            assert.ok(!isNaN(secondResult[i]), `Unexpected NaN at index ${i}`);
        }
    }
});

test('trendflex_nan_handling', () => {
    const close = testData.close;
    
    const result = wasm.trendflex_js(close, 20);
    assert.equal(result.length, close.length);
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert.ok(!isNaN(result[i]), `Unexpected NaN at index ${i}`);
        }
    }
    
    // First 19 values should be NaN (period-1)
    for (let i = 0; i < 19; i++) {
        assert.ok(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
});

test('trendflex_batch', () => {
    const close = testData.close;
    
    const result = wasm.trendflex_batch_js(close, 20, 20, 0);
    const metadata = wasm.trendflex_batch_metadata_js(20, 20, 0);
    
    // Should have 1 period value
    assert.equal(metadata.length, 1);
    assert.equal(metadata[0], 20);
    
    // Result should be flattened array (1 row × data length)
    assert.equal(result.length, close.length);
    
    // Check last 5 values
    const expected = EXPECTED_OUTPUTS.trendflex.last_5_values;
    const last5 = result.slice(-5);
    expected.forEach((expectedVal, i) => {
        assertClose(last5[i], expectedVal, 1e-8, `TrendFlex batch mismatch at index ${i}`);
    });
});

test('trendflex_batch_multiple_periods', () => {
    const close = testData.close;
    
    const result = wasm.trendflex_batch_js(close, 10, 30, 10);
    const metadata = wasm.trendflex_batch_metadata_js(10, 30, 10);
    
    // Should have 3 periods: 10, 20, 30
    assert.equal(metadata.length, 3);
    assert.deepEqual(Array.from(metadata), [10, 20, 30]);
    
    // Result should be flattened array (3 rows × data length)
    assert.equal(result.length, 3 * close.length);
});

test('trendflex_all_nan_input', () => {
    const allNan = new Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.trendflex_js(allNan, 20);
    }, /All values are NaN/);
});