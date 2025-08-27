/**
 * WASM binding tests for Stoch indicator.
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

test('Stoch partial params', () => {
    // Test with default parameters - mirrors check_stoch_partial_params
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    // Default params: fastk_period=14, slowk_period=3, slowk_ma_type="sma", 
    // slowd_period=3, slowd_ma_type="sma"
    const result = wasm.stoch(high, low, close, 14, 3, "sma", 3, "sma");
    
    assert.strictEqual(result.values.length, high.length * 2); // K and D concatenated
    assert.strictEqual(result.rows, 2);
    assert.strictEqual(result.cols, high.length);
});

test('Stoch accuracy', () => {
    // Test Stoch matches expected values from Rust tests - mirrors check_stoch_accuracy
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    // Expected values from Rust test
    const expectedK = [
        42.51122827572717,
        40.13864479593807,
        37.853934778363374,
        37.337021714266086,
        36.26053890551548,
    ];
    const expectedD = [
        41.36561869426493,
        41.7691857059163,
        40.16793595000925,
        38.44320042952222,
        37.15049846604803,
    ];
    
    const result = wasm.stoch(high, low, close, 14, 3, "sma", 3, "sma");
    
    // Extract K and D from flattened result
    const k = result.values.slice(0, result.cols);
    const d = result.values.slice(result.cols);
    
    assert.strictEqual(k.length, close.length);
    assert.strictEqual(d.length, close.length);
    
    // Check last 5 values match expected
    const lastK = k.slice(-5);
    const lastD = d.slice(-5);
    
    assertArrayClose(
        lastK,
        expectedK,
        1e-6,
        "Stoch K last 5 values mismatch"
    );
    
    assertArrayClose(
        lastD,
        expectedD,
        1e-6,
        "Stoch D last 5 values mismatch"
    );
});

test('Stoch default candles', () => {
    // Test Stoch with default parameters - mirrors check_stoch_default_candles
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.stoch(high, low, close, 14, 3, "sma", 3, "sma");
    assert.strictEqual(result.values.length, high.length * 2);
});

test('Stoch zero period', () => {
    // Test Stoch fails with zero period - mirrors check_stoch_zero_period
    const high = new Float64Array([10.0, 11.0, 12.0]);
    const low = new Float64Array([9.0, 9.5, 10.5]);
    const close = new Float64Array([9.5, 10.6, 11.5]);
    
    assert.throws(() => {
        wasm.stoch(high, low, close, 0, 3, "sma", 3, "sma");
    }, /Invalid period/);
});

test('Stoch period exceeds length', () => {
    // Test Stoch fails when period exceeds data length - mirrors check_stoch_period_exceeds_length
    const high = new Float64Array([10.0, 11.0, 12.0]);
    const low = new Float64Array([9.0, 9.5, 10.5]);
    const close = new Float64Array([9.5, 10.6, 11.5]);
    
    assert.throws(() => {
        wasm.stoch(high, low, close, 10, 3, "sma", 3, "sma");
    }, /Invalid period/);
});

test('Stoch all NaN', () => {
    // Test Stoch fails with all NaN values - mirrors check_stoch_all_nan
    const nanData = new Float64Array(20).fill(NaN);
    
    assert.throws(() => {
        wasm.stoch(nanData, nanData, nanData, 14, 3, "sma", 3, "sma");
    }, /All values are NaN/);
});

test('Stoch empty input', () => {
    // Test Stoch fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.stoch(empty, empty, empty, 14, 3, "sma", 3, "sma");
    }, /Empty data/);
});

test('Stoch mismatched lengths', () => {
    // Test Stoch fails with mismatched input lengths
    const high = new Float64Array([10.0, 11.0, 12.0]);
    const low = new Float64Array([9.0, 9.5]); // Different length
    const close = new Float64Array([9.5, 10.6, 11.5]);
    
    assert.throws(() => {
        wasm.stoch(high, low, close, 14, 3, "sma", 3, "sma");
    }, /Mismatched length/);
});

test('Stoch batch single parameter', () => {
    // Test Stoch batch with single parameter set - mirrors check_batch_default_row
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    // Single parameter set (default values)
    const result = wasm.stoch_batch(
        high, low, close,
        14, 14, 0, // fastk_range
        3, 3, 0,   // slowk_range  
        "sma",     // slowk_ma_type
        3, 3, 0,   // slowd_range
        "sma"      // slowd_ma_type
    );
    
    assert.ok(result.values);
    assert.ok(result.combos);
    assert.strictEqual(result.rows_per_combo, 2); // K and D
    assert.strictEqual(result.cols, close.length);
    
    // Should have 1 combo
    assert.strictEqual(result.combos.length, 1);
    
    // Values should be [all K rows..., all D rows...]
    // With 1 combo and 2 rows per combo = 2 total rows
    assert.strictEqual(result.values.length, 2 * close.length);
});

test('Stoch batch multiple parameters', () => {
    // Test Stoch batch with parameter sweep
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Test with parameter sweep
    const result = wasm.stoch_batch(
        high, low, close,
        10, 20, 5,  // fastk_range: 10, 15, 20
        2, 4, 1,    // slowk_range: 2, 3, 4
        "sma",      // slowk_ma_type
        2, 4, 1,    // slowd_range: 2, 3, 4
        "sma"       // slowd_ma_type
    );
    
    // Should have 3 * 3 * 3 = 27 parameter combinations
    const expectedCombos = 3 * 3 * 3;
    assert.strictEqual(result.combos.length, expectedCombos);
    assert.strictEqual(result.rows_per_combo, 2);
    assert.strictEqual(result.cols, close.length);
    
    // Total values = combos * rows_per_combo * cols
    assert.strictEqual(result.values.length, expectedCombos * 2 * close.length);
});

test('Stoch different MA types', () => {
    // Test Stoch with different MA types for smoothing
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Test with SMA (default)
    const resultSMA = wasm.stoch(high, low, close, 14, 3, "sma", 3, "sma");
    const kSMA = resultSMA.values.slice(0, resultSMA.cols);
    const dSMA = resultSMA.values.slice(resultSMA.cols);
    
    // Test with EMA
    const resultEMA = wasm.stoch(high, low, close, 14, 3, "ema", 3, "ema");
    const kEMA = resultEMA.values.slice(0, resultEMA.cols);
    const dEMA = resultEMA.values.slice(resultEMA.cols);
    
    // Results should be different
    assert.strictEqual(kSMA.length, kEMA.length);
    assert.strictEqual(dSMA.length, dEMA.length);
    
    // Values should differ after warmup
    const warmup = 20;
    let kDiffCount = 0;
    let dDiffCount = 0;
    
    for (let i = warmup; i < kSMA.length; i++) {
        if (!isNaN(kSMA[i]) && !isNaN(kEMA[i])) {
            if (Math.abs(kSMA[i] - kEMA[i]) > 1e-6) {
                kDiffCount++;
            }
        }
        if (!isNaN(dSMA[i]) && !isNaN(dEMA[i])) {
            if (Math.abs(dSMA[i] - dEMA[i]) > 1e-6) {
                dDiffCount++;
            }
        }
    }
    
    // At least some values should be different
    assert.ok(kDiffCount > 0, "K values should differ between SMA and EMA");
    assert.ok(dDiffCount > 0, "D values should differ between SMA and EMA");
});