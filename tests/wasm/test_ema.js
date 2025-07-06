/**
 * WASM binding tests for EMA indicator.
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
        wasm = await import(wasmPath);
        // No need to call default() for ES modules
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('EMA partial params', () => {
    // Test with default parameters - mirrors check_ema_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.ema_js(close, 9);
    assert.strictEqual(result.length, close.length);
});

test('EMA accuracy', () => {
    // Test EMA matches expected values from Rust tests - mirrors check_ema_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.ema;
    
    const result = wasm.ema_js(
        close,
        expected.defaultParams.period
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.lastFive,
        1e-1,
        "EMA last 5 values mismatch"
    );
});

test('EMA default candles', () => {
    // Test EMA with default candle data - mirrors check_ema_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.ema_js(close, 9);
    assert.strictEqual(result.length, close.length);
});

test('EMA zero period', () => {
    // Test EMA fails with zero period - mirrors check_ema_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.ema_js(inputData, 0);
    }, "Expected error for zero period");
});

test('EMA period exceeds length', () => {
    // Test EMA fails when period exceeds data length - mirrors check_ema_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.ema_js(dataSmall, 10);
    }, "Expected error for period exceeding length");
});

test('EMA very small dataset', () => {
    // Test EMA with single data point - mirrors check_ema_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.ema_js(singlePoint, 9);
    }, "Expected error for insufficient data");
});

test('EMA NaN handling', () => {
    // Test EMA handles NaN values correctly - mirrors check_ema_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.ema_js(close, 9);
    assert.strictEqual(result.length, close.length);
    
    // Check that values after warm-up period are not NaN
    if (result.length > 30) {
        for (let i = 30; i < result.length; i++) {
            assert.ok(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('EMA batch', () => {
    // Test EMA batch computation
    const close = new Float64Array(testData.close);
    
    // Test batch with period range
    const result = wasm.ema_batch_js(
        close,
        5,    // period_start
        20,   // period_end  
        5     // period_step
    );
    
    // Batch result is a flat array of all results concatenated
    const expectedPeriods = [5, 10, 15, 20];
    const expectedLength = expectedPeriods.length * close.length;
    assert.strictEqual(result.length, expectedLength);
    
    // Get metadata to verify periods
    const metadata = wasm.ema_batch_metadata_js(5, 20, 5);
    assert.strictEqual(metadata.length, expectedPeriods.length);
    for (let i = 0; i < expectedPeriods.length; i++) {
        assert.strictEqual(metadata[i], expectedPeriods[i]);
    }
});