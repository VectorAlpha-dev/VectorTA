/**
 * WASM binding tests for StochF indicator.
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
    
    try {
        const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
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

test('StochF partial params', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.stochf_js(high, low, close, 5, 3, 0);
    assert.strictEqual(result.length, high.length * 2); 
});

test('StochF accuracy', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.stochf_js(high, low, close, 5, 3, 0);
    
    
    const k = result.slice(0, high.length);
    const d = result.slice(high.length);
    
    assert.strictEqual(k.length, close.length);
    assert.strictEqual(d.length, close.length);
    
    
    const expected_k = [
        80.6987399770905,
        40.88471849865952,
        15.507246376811594,
        36.920529801324506,
        32.1880650994575,
    ];
    const expected_d = [
        70.99960994145033,
        61.44725644908976,
        45.696901617520815,
        31.104164892265487,
        28.205280425864817,
    ];
    
    
    assertArrayClose(
        k.slice(-5), 
        expected_k,
        1e-4,
        "StochF K last 5 values mismatch"
    );
    assertArrayClose(
        d.slice(-5), 
        expected_d,
        1e-4,
        "StochF D last 5 values mismatch"
    );
});

test('StochF zero period', () => {
    
    const data = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0]);
    
    assert.throws(() => {
        wasm.stochf_js(data, data, data, 0, 3, 0);
    }, /Invalid period/);
});

test('StochF period exceeds length', () => {
    
    const data = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.stochf_js(data, data, data, 10, 3, 0);
    }, /Invalid period/);
});

test('StochF empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.stochf_js(empty, empty, empty, 5, 3, 0);
    }, /Empty data|Input data slice is empty/);
});

test('StochF all NaN input', () => {
    
    const all_nan = new Float64Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.stochf_js(all_nan, all_nan, all_nan, 5, 3, 0);
    }, /All values are NaN/);
});

test('StochF nan handling', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.stochf_js(high, low, close, 5, 3, 0);
    const k = result.slice(0, high.length);
    const d = result.slice(high.length);
    
    
    
    
    if (k.length > 10) {
        
        for (let i = 10; i < k.length; i++) {
            assert(!isNaN(k[i]), `Found unexpected NaN in K at index ${i}`);
        }
        
        
        for (let i = 10; i < d.length; i++) {
            assert(!isNaN(d[i]), `Found unexpected NaN in D at index ${i}`);
        }
    }
    
    
    for (let i = 0; i < Math.min(4, k.length); i++) {
        assert(isNaN(k[i]), `Expected NaN in K warmup period at index ${i}`);
    }
    
    
    for (let i = 0; i < Math.min(6, d.length); i++) {
        assert(isNaN(d[i]), `Expected NaN in D warmup period at index ${i}`);
    }
});

test('StochF fast API', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    
    const result = wasm.stochf_js(high, low, close, 5, 3, 0);
    
    
    assert.strictEqual(result.length, high.length * 2);
    
    
    const k_result = result.slice(0, high.length);
    const d_result = result.slice(high.length);
    
    
    assert.ok(isNaN(k_result[0]), 'K should have NaN at index 0');
    assert.ok(isNaN(k_result[1]), 'K should have NaN at index 1');
    assert.ok(isNaN(k_result[2]), 'K should have NaN at index 2');
    assert.ok(isNaN(k_result[3]), 'K should have NaN at index 3');
    
    assert.ok(isNaN(d_result[0]), 'D should have NaN at index 0');
    assert.ok(isNaN(d_result[1]), 'D should have NaN at index 1');
    assert.ok(isNaN(d_result[2]), 'D should have NaN at index 2');
    assert.ok(isNaN(d_result[3]), 'D should have NaN at index 3');
    assert.ok(isNaN(d_result[4]), 'D should have NaN at index 4');
    assert.ok(isNaN(d_result[5]), 'D should have NaN at index 5');
    
    
    let hasValidK = false;
    let hasValidD = false;
    for (let i = 10; i < high.length; i++) {
        if (!isNaN(k_result[i])) hasValidK = true;
        if (!isNaN(d_result[i])) hasValidD = true;
        if (hasValidK && hasValidD) break;
    }
    
    assert.ok(hasValidK, 'K should have valid values after warmup');
    assert.ok(hasValidD, 'D should have valid values after warmup');
});

test('StochF batch single parameter set', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    
    const config = {
        fastk_range: [5, 5, 0],
        fastd_range: [3, 3, 0],
        fastd_matype: 0
    };
    
    const batch_result = wasm.stochf_batch(high, low, close, config);
    
    
    const single_result = wasm.stochf_js(high, low, close, 5, 3, 0);
    
    assert.strictEqual(batch_result.rows, 1);
    assert.strictEqual(batch_result.cols, close.length);
    assert.strictEqual(batch_result.k_values.length, close.length);
    assert.strictEqual(batch_result.d_values.length, close.length);
    
    const single_k = single_result.slice(0, close.length);
    const single_d = single_result.slice(close.length);
    
    assertArrayClose(
        batch_result.k_values, 
        single_k, 
        1e-10, 
        "Batch vs single K mismatch"
    );
    assertArrayClose(
        batch_result.d_values, 
        single_d, 
        1e-10, 
        "Batch vs single D mismatch"
    );
});

test('StochF batch multiple periods', () => {
    
    const high = new Float64Array(testData.high.slice(0, 100)); 
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    
    const config = {
        fastk_range: [5, 9, 2],    
        fastd_range: [3, 3, 0],     
        fastd_matype: 0
    };
    
    const batch_result = wasm.stochf_batch(high, low, close, config);
    
    
    assert.strictEqual(batch_result.rows, 3);
    assert.strictEqual(batch_result.cols, 100);
    assert.strictEqual(batch_result.k_values.length, 300);
    assert.strictEqual(batch_result.d_values.length, 300);
    assert.strictEqual(batch_result.combos.length, 3);
    
    
    const fastk_periods = [5, 7, 9];
    for (let i = 0; i < fastk_periods.length; i++) {
        const single_result = wasm.stochf_js(high, low, close, fastk_periods[i], 3, 0);
        const single_k = single_result.slice(0, 100);
        const single_d = single_result.slice(100);
        
        const batch_k = batch_result.k_values.slice(i * 100, (i + 1) * 100);
        const batch_d = batch_result.d_values.slice(i * 100, (i + 1) * 100);
        
        assertArrayClose(
            batch_k, 
            single_k, 
            1e-10, 
            `FastK period ${fastk_periods[i]} K mismatch`
        );
        assertArrayClose(
            batch_d, 
            single_d, 
            1e-10, 
            `FastK period ${fastk_periods[i]} D mismatch`
        );
    }
});

test('StochF batch metadata', () => {
    
    const high = new Float64Array(testData.high.slice(0, 20));
    const low = new Float64Array(testData.low.slice(0, 20));
    const close = new Float64Array(testData.close.slice(0, 20));
    
    const config = {
        fastk_range: [5, 7, 2],     
        fastd_range: [3, 4, 1],     
        fastd_matype: 0
    };
    
    const result = wasm.stochf_batch(high, low, close, config);
    
    
    assert.strictEqual(result.combos.length, 4);
    
    
    const expected_combos = [
        {fastk_period: 5, fastd_period: 3},
        {fastk_period: 5, fastd_period: 4},
        {fastk_period: 7, fastd_period: 3},
        {fastk_period: 7, fastd_period: 4}
    ];
    
    for (let i = 0; i < expected_combos.length; i++) {
        assert.strictEqual(result.combos[i].fastk_period, expected_combos[i].fastk_period);
        assert.strictEqual(result.combos[i].fastd_period, expected_combos[i].fastd_period);
        assert.strictEqual(result.combos[i].fastd_matype, 0);
    }
});