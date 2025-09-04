/**
 * WASM binding tests for VAMA indicator.
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
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('VAMA fast accuracy', () => {
    // Test VAMA with fast parameters (length=13) matches expected values
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    const expected = EXPECTED_OUTPUTS.vama;
    
    const result = wasm.vama_js(
        close,
        volume,
        expected.defaultParams.length,
        expected.defaultParams.viFactor,
        expected.defaultParams.strict,
        expected.defaultParams.samplePeriod
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.fastValues,
        1e-6,
        "VAMA fast last 5 values mismatch"
    );
});

test('VAMA slow accuracy', () => {
    // Test VAMA with slow parameters (length=55) matches expected values
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    const expected = EXPECTED_OUTPUTS.vama;
    
    const result = wasm.vama_js(
        close,
        volume,
        expected.slowParams.length,
        expected.slowParams.viFactor,
        expected.slowParams.strict,
        expected.slowParams.samplePeriod
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.slowValues,
        1e-6,
        "VAMA slow last 5 values mismatch"
    );
});

test('VAMA default params', () => {
    // Test with default parameters
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    const result = wasm.vama_js(close, volume, 13, 0.67, true, 0);
    assert.strictEqual(result.length, close.length);
});

test('VAMA empty input', () => {
    // Test VAMA fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.vama_js(empty, empty, 13, 0.67, true, 0);
    }, /[Ee]mpty/);
});

test('VAMA all NaN input', () => {
    // Test VAMA with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    const volume = new Float64Array(100);
    volume.fill(100);
    
    assert.throws(() => {
        wasm.vama_js(allNaN, volume, 13, 0.67, true, 0);
    }, /All values are NaN/);
});

test('VAMA mismatched lengths', () => {
    // Test VAMA fails when price and volume have different lengths
    const price = new Float64Array([10.0, 20.0, 30.0]);
    const volume = new Float64Array([100.0, 200.0]);  // Different length
    
    assert.throws(() => {
        wasm.vama_js(price, volume, 13, 0.67, true, 0);
    }, /length mismatch/);
});

test('VAMA invalid period', () => {
    // Test VAMA fails with zero period
    const price = new Float64Array([10.0, 20.0, 30.0]);
    const volume = new Float64Array([100.0, 200.0, 300.0]);
    
    assert.throws(() => {
        wasm.vama_js(price, volume, 0, 0.67, true, 0);
    }, /Invalid period/);
});

test('VAMA invalid vi_factor', () => {
    // Test VAMA fails with invalid vi_factor
    const price = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0]);
    const volume = new Float64Array([100.0, 200.0, 300.0, 400.0, 500.0]);
    
    // Zero vi_factor
    assert.throws(() => {
        wasm.vama_js(price, volume, 2, 0.0, true, 0);
    }, /Invalid vi_factor/);
    
    // Negative vi_factor
    assert.throws(() => {
        wasm.vama_js(price, volume, 2, -1.0, true, 0);
    }, /Invalid vi_factor/);
});

test('VAMA period exceeds length', () => {
    // Test VAMA fails when period exceeds data length
    const smallPrice = new Float64Array([10.0, 20.0, 30.0]);
    const smallVolume = new Float64Array([100.0, 200.0, 300.0]);
    
    assert.throws(() => {
        wasm.vama_js(smallPrice, smallVolume, 10, 0.67, true, 0);
    }, /Invalid period|Not enough/);
});

test('VAMA NaN handling', () => {
    // Test VAMA handles NaN values correctly
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    const expected = EXPECTED_OUTPUTS.vama;
    
    const result = wasm.vama_js(close, volume, 13, 0.67, true, 0);
    assert.strictEqual(result.length, close.length);
    
    // After warmup period, no NaN values should exist
    const warmup = expected.warmupPeriod; // Should be 12 (length - 1)
    
    if (result.length > warmup) {
        for (let i = warmup + 1; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    // First warmup values should be NaN
    assertAllNaN(result.slice(0, warmup), "Expected NaN in warmup period");
});

test('VAMA strict vs non-strict', () => {
    // Test VAMA with strict=True vs strict=False
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset
    const volume = new Float64Array(testData.volume.slice(0, 100));
    
    // Test with strict=true
    const resultStrict = wasm.vama_js(close, volume, 13, 0.67, true, 0);
    
    // Test with strict=false
    const resultNonStrict = wasm.vama_js(close, volume, 13, 0.67, false, 0);
    
    assert.strictEqual(resultStrict.length, close.length);
    assert.strictEqual(resultNonStrict.length, close.length);
    
    // Results may differ but both should be valid
    let hasValidStrict = false;
    let hasValidNonStrict = false;
    for (let i = 13; i < resultStrict.length; i++) {
        if (!isNaN(resultStrict[i])) hasValidStrict = true;
        if (!isNaN(resultNonStrict[i])) hasValidNonStrict = true;
    }
    assert(hasValidStrict, "Strict mode should produce valid values");
    assert(hasValidNonStrict, "Non-strict mode should produce valid values");
});

test('VAMA sample period', () => {
    // Test VAMA with different sample periods
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset
    const volume = new Float64Array(testData.volume.slice(0, 100));
    
    // Test with sample_period=0 (all bars)
    const resultAll = wasm.vama_js(close, volume, 13, 0.67, true, 0);
    
    // Test with fixed sample_period
    const resultFixed = wasm.vama_js(close, volume, 13, 0.67, true, 20);
    
    assert.strictEqual(resultAll.length, close.length);
    assert.strictEqual(resultFixed.length, close.length);
    
    // Results may differ but both should be valid
    let hasValidAll = false;
    let hasValidFixed = false;
    for (let i = 13; i < resultAll.length; i++) {
        if (!isNaN(resultAll[i])) hasValidAll = true;
        if (!isNaN(resultFixed[i])) hasValidFixed = true;
    }
    assert(hasValidAll, "sample_period=0 should produce valid values");
    assert(hasValidFixed, "Fixed sample_period should produce valid values");
});

test('VAMA different vi_factors', () => {
    // Test VAMA with different vi_factor values
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset
    const volume = new Float64Array(testData.volume.slice(0, 100));
    
    // Test with different vi_factors
    const result1 = wasm.vama_js(close, volume, 13, 0.5, true, 0);
    const result2 = wasm.vama_js(close, volume, 13, 0.67, true, 0);
    const result3 = wasm.vama_js(close, volume, 13, 1.0, true, 0);
    
    assert.strictEqual(result1.length, close.length);
    assert.strictEqual(result2.length, close.length);
    assert.strictEqual(result3.length, close.length);
    
    // Results should differ with different vi_factors
    let hasDifference12 = false;
    let hasDifference23 = false;
    
    for (let i = close.length - 10; i < close.length; i++) {
        if (result1[i] !== result2[i]) hasDifference12 = true;
        if (result2[i] !== result3[i]) hasDifference23 = true;
    }
    
    assert(hasDifference12, "Different vi_factors should produce different results");
    assert(hasDifference23, "Different vi_factors should produce different results");
});

test('VAMA constant volume', () => {
    // Test VAMA with constant volume
    // Create price series with some variation
    const priceData = [];
    for (let i = 0; i < 5; i++) {
        priceData.push(50.0, 51.0, 49.0, 52.0, 48.0, 53.0, 47.0, 54.0, 46.0, 55.0);
    }
    const price = new Float64Array(priceData);
    
    // Constant volume
    const volume = new Float64Array(50);
    volume.fill(1000.0);
    
    const result = wasm.vama_js(price, volume, 5, 0.67, true, 0);
    
    assert.strictEqual(result.length, price.length);
    
    // With constant volume, VAMA should still produce valid results
    let hasValidValues = false;
    for (let i = 5; i < result.length; i++) {
        if (!isNaN(result[i])) {
            hasValidValues = true;
            break;
        }
    }
    assert(hasValidValues, "VAMA should produce valid values with constant volume");
});