/**
 * WASM binding tests for Nadaraya-Watson Envelope indicator
 */

import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { 
    loadTestData, 
    assertArrayClose,
    isNaN,
    assertAllNaN
} from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

test.before(async () => {
    
    try {
        const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
        const importPath = process.platform === 'win32' 
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
        wasm = await import(importPath);
    } catch (e) {
        console.error('WASM module not built. Run: wasm-pack build --target nodejs --features wasm');
        process.exit(1);
    }
    
    
    testData = await loadTestData();
});


const expectedUpper = [62141.41569185, 62108.86018850, 62069.70106389, 62045.52821051, 61980.68541380];
const expectedLower = [56560.88881720, 56530.68399610, 56490.03377396, 56465.39492722, 56394.51167599];

test('NWE accuracy test with reference values', () => {
    const close = testData.close;
    
    
    const result = wasm.nadaraya_watson_envelope(close, 8.0, 3.0, 500);
    
    assert.ok(result.upper, 'Result should have upper envelope');
    assert.ok(result.lower, 'Result should have lower envelope');
    assert.strictEqual(result.upper.length, close.length, 'Upper envelope length should match input');
    assert.strictEqual(result.lower.length, close.length, 'Lower envelope length should match input');
    
    
    const upperLast5 = result.upper.slice(-5);
    const lowerLast5 = result.lower.slice(-5);
    
    
    assertArrayClose(upperLast5, expectedUpper, 1e-8, 'Upper envelope last 5 values');
    assertArrayClose(lowerLast5, expectedLower, 1e-8, 'Lower envelope last 5 values');
});

test('NWE with default parameters', () => {
    const close = testData.close;
    const result = wasm.nadaraya_watson_envelope(close, 8.0, 3.0, 500);
    
    assert.strictEqual(result.upper.length, close.length);
    assert.strictEqual(result.lower.length, close.length);
    
    
    for (let i = 10; i < close.length; i++) {
        if (!isNaN(result.upper[i]) && !isNaN(result.lower[i])) {
            assert.ok(result.upper[i] > result.lower[i], 
                `Upper should be greater than lower at index ${i}`);
        }
    }
});

test('NWE with different parameters', () => {
    const close = testData.close;
    const closeSlice = close.slice(0, 1100);  
    const lookback = 100;  
    
    
    const result1 = wasm.nadaraya_watson_envelope(closeSlice, 4.0, 3.0, lookback);
    const result2 = wasm.nadaraya_watson_envelope(closeSlice, 16.0, 3.0, lookback);
    
    assert.strictEqual(result1.upper.length, closeSlice.length);
    assert.strictEqual(result2.upper.length, closeSlice.length);
    
    
    let isDifferent = false;
    for (let i = 0; i < closeSlice.length; i++) {
        if (!isNaN(result1.upper[i]) && !isNaN(result2.upper[i])) {
            if (Math.abs(result1.upper[i] - result2.upper[i]) > 1e-10) {
                isDifferent = true;
                break;
            }
        }
    }
    assert.ok(isDifferent, 'Different bandwidth should produce different results');
    
    
    const result3 = wasm.nadaraya_watson_envelope(closeSlice, 8.0, 1.0, lookback);
    const result4 = wasm.nadaraya_watson_envelope(closeSlice, 8.0, 5.0, lookback);
    
    
    
    const warmup = lookback - 1 + 498;  
    let foundWiderBand = false;
    for (let i = warmup; i < Math.min(warmup + 100, closeSlice.length); i++) {
        if (!isNaN(result3.upper[i]) && !isNaN(result3.lower[i]) &&
            !isNaN(result4.upper[i]) && !isNaN(result4.lower[i])) {
            const bandWidth3 = result3.upper[i] - result3.lower[i];
            const bandWidth4 = result4.upper[i] - result4.lower[i];
            if (bandWidth4 > bandWidth3) {
                foundWiderBand = true;
                break;
            }
        }
    }
    assert.ok(foundWiderBand, 'Larger multiplier should produce wider bands');
});

test('NWE flat array output format', () => {
    const close = testData.close;
    const closeSlice = close.slice(0, 1000);  
    const flatResult = wasm.nadaraya_watson_envelope_flat(closeSlice, 8.0, 3.0, 500);
    
    
    assert.ok(flatResult, 'Should return result object');
    assert.ok(Array.isArray(flatResult.values), 'Should have values array');
    assert.strictEqual(flatResult.rows, 2, 'Should have 2 rows (upper and lower)');
    assert.strictEqual(flatResult.cols, closeSlice.length, 'Should have correct column count');
    assert.strictEqual(flatResult.values.length, closeSlice.length * 2, 'Should return upper and lower concatenated');
    
    
    const upper = flatResult.values.slice(0, closeSlice.length);
    const lower = flatResult.values.slice(closeSlice.length);
    
    
    const warmup = 499 + 498;
    for (let i = warmup; i < closeSlice.length; i++) {
        if (!isNaN(upper[i]) && !isNaN(lower[i])) {
            assert.ok(upper[i] > lower[i], `Upper should be greater than lower at index ${i}`);
            break; 
        }
    }
});

test('NWE batch processing', () => {
    const close = testData.close;
    const closeSlice = close.slice(0, 100);
    
    
    const result = wasm.nadaraya_watson_envelope_batch(
        closeSlice,
        [8.0, 8.0, 0.0],  
        [3.0, 3.0, 0.0],  
        [500, 500, 0]     
    );
    
    assert.ok(result.upper, 'Should have upper values');
    assert.ok(result.lower, 'Should have lower values');
    assert.ok(result.bandwidths, 'Should have bandwidths');
    assert.ok(result.multipliers, 'Should have multipliers');
    assert.ok(result.lookbacks, 'Should have lookbacks');
    
    assert.strictEqual(result.rows, 1, 'Should have 1 combination');
    assert.strictEqual(result.cols, closeSlice.length, 'Should have correct data length');
    assert.strictEqual(result.upper.length, closeSlice.length, 'Upper should have correct length');
    assert.strictEqual(result.lower.length, closeSlice.length, 'Lower should have correct length');
    
    
    const multiResult = wasm.nadaraya_watson_envelope_batch(
        closeSlice,
        [6.0, 10.0, 2.0],  
        [2.0, 4.0, 1.0],   
        [400, 500, 100]    
    );
    
    assert.strictEqual(multiResult.rows, 18, 'Should have 3*3*2=18 combinations');
    assert.strictEqual(multiResult.cols, closeSlice.length, 'Should have correct data length');
    assert.strictEqual(multiResult.upper.length, 18 * closeSlice.length, 'Upper flattened size correct');
    assert.strictEqual(multiResult.lower.length, 18 * closeSlice.length, 'Lower flattened size correct');
    assert.strictEqual(multiResult.bandwidths.length, 18, 'Should have 18 bandwidth values');
    assert.strictEqual(multiResult.multipliers.length, 18, 'Should have 18 multiplier values');
    assert.strictEqual(multiResult.lookbacks.length, 18, 'Should have 18 lookback values');
});

test('NWE with empty input should throw', () => {
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.nadaraya_watson_envelope(empty, 8.0, 3.0, 500);
    }, /Input data slice is empty/);
});

test('NWE with invalid bandwidth should throw', () => {
    const data = new Float64Array([1, 2, 3, 4, 5]);
    
    
    assert.throws(() => {
        wasm.nadaraya_watson_envelope(data, 0.0, 3.0, 500);
    }, /Invalid bandwidth/);
    
    
    assert.throws(() => {
        wasm.nadaraya_watson_envelope(data, -1.0, 3.0, 500);
    }, /Invalid bandwidth/);
});

test('NWE with invalid multiplier should throw', () => {
    const data = new Float64Array([1, 2, 3, 4, 5]);
    
    
    assert.throws(() => {
        wasm.nadaraya_watson_envelope(data, 8.0, -1.0, 500);
    }, /Invalid multiplier/);
});

test('NWE with invalid lookback should throw', () => {
    const data = new Float64Array([1, 2, 3, 4, 5]);
    
    
    assert.throws(() => {
        wasm.nadaraya_watson_envelope(data, 8.0, 3.0, 0);
    }, /Invalid lookback/);
});

test('NWE with small dataset - mirrors check_nwe_very_small_dataset', () => {
    
    
    const single = new Float64Array([42.0]);
    assert.throws(() => {
        wasm.nadaraya_watson_envelope(single, 8.0, 3.0, 500);
    }, /Not enough valid data/);
    
    
    const small = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    assert.throws(() => {
        wasm.nadaraya_watson_envelope(small, 8.0, 3.0, 10);
    }, /Not enough valid data/);
});

test('NWE with all NaN should throw', () => {
    const allNaN = new Float64Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.nadaraya_watson_envelope(allNaN, 8.0, 3.0, 500);
    }, /All values are NaN/);
});

test('NWE warmup period calculation - mirrors check_nwe_warmup_period', () => {
    
    const data = new Float64Array(1000);
    for (let i = 0; i < 1000; i++) {
        data[i] = 50000.0 + Math.sin(i) * 100.0;
    }
    
    
    const result = wasm.nadaraya_watson_envelope(data, 8.0, 3.0, 500);
    
    
    
    const WARMUP = 499 + 498;
    
    
    for (let i = 0; i < WARMUP; i++) {
        assert.ok(isNaN(result.upper[i]), `Upper should be NaN at ${i} during warmup`);
        assert.ok(isNaN(result.lower[i]), `Lower should be NaN at ${i} during warmup`);
    }
    
    
    if (data.length > WARMUP) {
        assert.ok(!isNaN(result.upper[WARMUP]), `Upper should not be NaN at ${WARMUP}`);
        assert.ok(!isNaN(result.lower[WARMUP]), `Lower should not be NaN at ${WARMUP}`);
    }
});

test('NWE memory allocation/deallocation', () => {
    const close = testData.close;
    const data = close.slice(0, 100);
    const len = data.length;
    
    
    const ptr = wasm.nadaraya_watson_envelope_alloc(len);
    assert.ok(ptr !== 0, 'Should allocate memory');
    
    
    assert.doesNotThrow(() => {
        wasm.nadaraya_watson_envelope_free(ptr, len);
    });
});

console.log('All Nadaraya-Watson Envelope tests passed!');