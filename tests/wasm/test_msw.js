/**
 * WASM binding tests for MSW (Mesa Sine Wave) indicator.
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
        // No need to call default() for ES modules
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('MSW partial params', () => {
    // Test with default parameters - mirrors check_msw_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.msw_js(close, 5);
    assert(result.sine, 'Should have sine array');
    assert(result.lead, 'Should have lead array');
    assert.strictEqual(result.sine.length, close.length);
    assert.strictEqual(result.lead.length, close.length);
});

test('MSW accuracy', async () => {
    // Test MSW matches expected values from Rust tests - mirrors check_msw_accuracy
    const close = new Float64Array(testData.close);
    const period = 5;
    
    const result = wasm.msw_js(close, period);
    
    assert.strictEqual(result.sine.length, close.length);
    assert.strictEqual(result.lead.length, close.length);
    
    // Expected values from Rust test
    const expectedLastFiveSine = [
        -0.49733966449848194,
        -0.8909425976991894,
        -0.709353328514554,
        -0.40483478076837887,
        -0.8817006719953886,
    ];
    const expectedLastFiveLead = [
        -0.9651269132969991,
        -0.30888310410390457,
        -0.003182174183612666,
        0.36030983330963545,
        -0.28983704937461496,
    ];
    
    // Check last 5 values match expected
    const last5Sine = Array.from(result.sine.slice(-5));
    const last5Lead = Array.from(result.lead.slice(-5));
    
    assertArrayClose(
        last5Sine,
        expectedLastFiveSine,
        1e-1,  // MSW uses 1e-1 tolerance in Rust tests
        "MSW sine last 5 values mismatch"
    );
    assertArrayClose(
        last5Lead,
        expectedLastFiveLead,
        1e-1,  // MSW uses 1e-1 tolerance in Rust tests
        "MSW lead last 5 values mismatch"
    );
});

test('MSW default candles', () => {
    // Test MSW with default parameters - mirrors check_msw_default_candles
    const close = new Float64Array(testData.close);
    
    // Default params: period=5
    const result = wasm.msw_js(close, 5);
    assert.strictEqual(result.sine.length, close.length);
    assert.strictEqual(result.lead.length, close.length);
});

test('MSW zero period', () => {
    // Test MSW fails with zero period - mirrors check_msw_zero_period
    const data = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.msw_js(data, 0);
    }, /Invalid period/);
});

test('MSW period exceeds length', () => {
    // Test MSW fails when period exceeds data length - mirrors check_msw_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.msw_js(dataSmall, 10);
    }, /Invalid period/);
});

test('MSW very small dataset', () => {
    // Test MSW fails with insufficient data - mirrors check_msw_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.msw_js(singlePoint, 5);
    }, /Invalid period|Not enough valid data/);
});

test('MSW empty input', () => {
    // Test MSW fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.msw_js(empty, 5);
    }, /Empty data/);
});

test('MSW NaN handling', () => {
    // Test MSW handles NaN values correctly - mirrors check_msw_nan_handling
    const close = new Float64Array(testData.close);
    const period = 5;
    
    const result = wasm.msw_js(close, period);
    assert.strictEqual(result.sine.length, close.length);
    assert.strictEqual(result.lead.length, close.length);
    
    // First `period-1` values should be NaN
    const expectedWarmup = period - 1;  // for period=5, warmup is 4
    assertAllNaN(Array.from(result.sine.slice(0, expectedWarmup)), "Expected NaN in sine warmup period");
    assertAllNaN(Array.from(result.lead.slice(0, expectedWarmup)), "Expected NaN in lead warmup period");
    
    // After warmup period, no NaN values should exist
    if (result.sine.length > expectedWarmup) {
        const nonNanStart = Math.max(expectedWarmup, 240);  // Skip initial NaN values in data
        if (result.sine.length > nonNanStart) {
            for (let i = nonNanStart; i < result.sine.length; i++) {
                assert(!isNaN(result.sine[i]), `Found unexpected NaN in sine at index ${i}`);
                assert(!isNaN(result.lead[i]), `Found unexpected NaN in lead at index ${i}`);
            }
        }
    }
});

test('MSW all NaN input', () => {
    // Test MSW with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.msw_js(allNaN, 5);
    }, /All values are NaN/);
});

test('MSW mixed NaN input', () => {
    // Test MSW with mixed NaN values
    const mixedData = new Float64Array([NaN, NaN, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0]);
    const period = 3;
    
    const result = wasm.msw_js(mixedData, period);
    assert.strictEqual(result.sine.length, mixedData.length);
    assert.strictEqual(result.lead.length, mixedData.length);
    
    // First values should be NaN due to input NaN and warmup
    assert(isNaN(result.sine[0]));
    assert(isNaN(result.sine[1]));
    assert(isNaN(result.lead[0]));
    assert(isNaN(result.lead[1]));
    
    // After warmup from first valid value, should have real values
    // First valid value is at index 2, so warmup ends at index 2 + period - 1 = 4
    for (let i = 4; i < result.sine.length; i++) {
        assert(!isNaN(result.sine[i]), `Unexpected NaN in sine at index ${i}`);
        assert(!isNaN(result.lead[i]), `Unexpected NaN in lead at index ${i}`);
    }
});

test('MSW simple predictable pattern', () => {
    // Test MSW with a simple pattern
    const simpleData = new Float64Array([1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5]);
    const period = 5;
    
    const result = wasm.msw_js(simpleData, period);
    assert.strictEqual(result.sine.length, simpleData.length);
    assert.strictEqual(result.lead.length, simpleData.length);
    
    // Check warmup period
    for (let i = 0; i < period - 1; i++) {
        assert(isNaN(result.sine[i]), `Expected NaN in sine at index ${i}`);
        assert(isNaN(result.lead[i]), `Expected NaN in lead at index ${i}`);
    }
    
    // After warmup, all values should be valid
    for (let i = period - 1; i < result.sine.length; i++) {
        assert(!isNaN(result.sine[i]), `Unexpected NaN in sine at index ${i}`);
        assert(!isNaN(result.lead[i]), `Unexpected NaN in lead at index ${i}`);
        
        // Sine values should be between -1 and 1
        assert(result.sine[i] >= -1.0 && result.sine[i] <= 1.0, 
               `Sine value ${result.sine[i]} at index ${i} is out of range [-1, 1]`);
        assert(result.lead[i] >= -1.0 && result.lead[i] <= 1.0, 
               `Lead value ${result.lead[i]} at index ${i} is out of range [-1, 1]`);
    }
});