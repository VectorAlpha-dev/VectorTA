/**
 * WASM binding tests for Ehlers PMA indicator.
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

test('Ehlers PMA accuracy', () => {
    // Test Ehlers PMA matches expected values from Rust tests - mirrors check_ehlers_pma_accuracy
    const close = new Float64Array(testData.close);
    
    // Call the WASM function - returns object with values, rows, cols
    const result = wasm.ehlers_pma(close);
    
    // Extract predict and trigger from flat result
    const predict = result.values.slice(0, close.length);
    const trigger = result.values.slice(close.length);
    
    assert.strictEqual(predict.length, close.length);
    assert.strictEqual(trigger.length, close.length);
    
    // Reference values from Rust tests (using close source with TradingView parity)
    const expectedPredictLastFive = [
        59161.97066327,
        59240.51785714,
        59260.29846939,
        59225.19005102,
        59192.78443878
    ];
    const expectedTriggerLastFive = [
        59020.56403061,
        59141.96938776,
        59214.56709184,
        59232.46619898,
        59220.78227041
    ];
    
    // Check last 5 values match expected
    assertArrayClose(
        predict.slice(-5),
        expectedPredictLastFive,
        1e-6,
        "Ehlers PMA predict last 5 values mismatch"
    );
    
    assertArrayClose(
        trigger.slice(-5),
        expectedTriggerLastFive,
        1e-6,
        "Ehlers PMA trigger last 5 values mismatch"
    );
});

test('Ehlers PMA default candles', () => {
    // Test Ehlers PMA with default parameters - mirrors check_ehlers_pma_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.ehlers_pma(close);
    
    // Should return 2 * length values
    assert.strictEqual(result.values.length, close.length * 2);
});

test('Ehlers PMA empty input', () => {
    // Test Ehlers PMA fails with empty input - mirrors check_ehlers_pma_empty_input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.ehlers_pma(empty);
    }, /Input data slice is empty/);
});

test('Ehlers PMA all NaN values', () => {
    // Test Ehlers PMA fails with all NaN values - mirrors check_ehlers_pma_all_nan
    const allNan = new Float64Array(20).fill(NaN);
    
    assert.throws(() => {
        wasm.ehlers_pma(allNan);
    }, /All values are NaN/);
});

test('Ehlers PMA insufficient data', () => {
    // Test Ehlers PMA fails with insufficient data - mirrors check_ehlers_pma_insufficient_data
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    assert.throws(() => {
        wasm.ehlers_pma(data);
    }, /Not enough valid data/);
});

test('Ehlers PMA very small dataset', () => {
    // Test Ehlers PMA fails with very small dataset - mirrors check_ehlers_pma_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.ehlers_pma(singlePoint);
    }, /Not enough valid data/);
    
    const twoPoints = new Float64Array([42.0, 43.0]);
    
    assert.throws(() => {
        wasm.ehlers_pma(twoPoints);
    }, /Not enough valid data/);
});

test('Ehlers PMA NaN handling', () => {
    // Test Ehlers PMA handles NaN values correctly - mirrors check_ehlers_pma_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.ehlers_pma(close);
    const predict = result.values.slice(0, close.length);
    const trigger = result.values.slice(close.length);
    
    assert.strictEqual(predict.length, close.length);
    assert.strictEqual(trigger.length, close.length);
    
    // After warmup period (20), no NaN values should exist
    if (predict.length > 20) {
        for (let i = 20; i < predict.length; i++) {
            assert(!isNaN(predict[i]), `Found unexpected NaN in predict at index ${i}`);
        }
        for (let i = 20; i < trigger.length; i++) {
            assert(!isNaN(trigger[i]), `Found unexpected NaN in trigger at index ${i}`);
        }
    }
    
    // First 12 values should be NaN for predict (WASM binding warmup)
    for (let i = 0; i < 12; i++) {
        assert(isNaN(predict[i]), `Expected NaN in predict warmup at index ${i}`);
    }
    
    // First 15 values should be NaN for trigger (WASM binding warmup)
    for (let i = 0; i < 15; i++) {
        assert(isNaN(trigger[i]), `Expected NaN in trigger warmup at index ${i}`);
    }
});

test('Ehlers PMA with minimum required data', () => {
    // Test with minimum 14 values required (with 1-bar lag)
    const data = new Float64Array(Array.from({length: 14}, (_, i) => 100.0 + i));
    
    const result = wasm.ehlers_pma(data);
    assert.strictEqual(result.values.length, data.length * 2);
    
    const predict = result.values.slice(0, data.length);
    const trigger = result.values.slice(data.length);
    
    // Should have at least one valid predict value
    const validPredict = [];
    for (let i = 0; i < predict.length; i++) {
        if (!isNaN(predict[i])) {
            validPredict.push(predict[i]);
        }
    }
    assert(validPredict.length > 0, 'Should have at least one valid predict value');
});

test('Ehlers PMA with constant values', () => {
    // Test with constant input
    const data = new Float64Array(50).fill(100.0);
    
    const result = wasm.ehlers_pma(data);
    const predict = result.values.slice(0, data.length);
    
    // Find first valid predict value
    let validIdx = -1;
    for (let i = 0; i < predict.length; i++) {
        if (!isNaN(predict[i])) {
            validIdx = i;
            break;
        }
    }
    
    assert(validIdx !== -1, 'Should have valid predict values');
    
    // With constant input, predict should stabilize near the constant
    assertClose(predict[validIdx], 100.0, 1e-9, 'Predict should equal constant value');
});

test('Ehlers PMA crossover detection', () => {
    // Use 200 candles for crossover detection
    const close = new Float64Array(testData.close.slice(0, 200));
    
    const result = wasm.ehlers_pma(close);
    const predict = result.values.slice(0, close.length);
    const trigger = result.values.slice(close.length);
    
    // Check for crossovers
    const crossovers = [];
    for (let i = 17; i < predict.length - 1; i++) {
        if (!isNaN(predict[i]) && !isNaN(trigger[i]) && 
            !isNaN(predict[i-1]) && !isNaN(trigger[i-1])) {
            if (predict[i-1] <= trigger[i-1] && predict[i] > trigger[i]) {
                crossovers.push({type: 'bullish', index: i});
            } else if (predict[i-1] >= trigger[i-1] && predict[i] < trigger[i]) {
                crossovers.push({type: 'bearish', index: i});
            }
        }
    }
    
    // Should detect at least one crossover in 200 candles
    assert(crossovers.length > 0, 'Should detect at least one crossover in real data');
});

test('Ehlers PMA with NaN in middle of data', () => {
    // Create data with NaN early enough that it eventually falls out of the 7-period window
    // Need at least 25 values: NaN at position 7, and we need 7 more values after the last
    // value that uses the NaN, plus the warmup period of 13
    const data = new Float64Array(30);
    for (let i = 0; i < data.length; i++) {
        data[i] = 100.0 + i;
    }
    
    // Put NaN at position 7
    data[7] = NaN;
    
    const result = wasm.ehlers_pma(data);
    assert.strictEqual(result.values.length, data.length * 2);
    
    const predict = result.values.slice(0, data.length);
    
    // NaN at position 7 will affect calculations from position 8 through 14 (inclusive)
    // because at position 14, we use data[13] through data[7] for WMA1
    // Starting from position 15, the NaN is no longer in the 7-period window
    
    // Check that NaN appears where expected (warmup + affected region)
    for (let i = 0; i < 15; i++) {
        assert(isNaN(predict[i]), `Expected NaN at index ${i} during warmup/affected period`);
    }
    
    // Check that calculation resumes after NaN falls out of window
    // With 30 values and NaN only affecting up to index 14, we should have valid values from 15 onwards
    let hasValidAfterNaN = false;
    for (let i = 15; i < predict.length; i++) {
        if (!isNaN(predict[i])) {
            hasValidAfterNaN = true;
            break;
        }
    }
    assert(hasValidAfterNaN, 'Should have valid values after NaN falls out of 7-period window');
});