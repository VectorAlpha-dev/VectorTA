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

test('StochF partial params', () => {
    // Test with default parameters - mirrors check_stochf_partial_params
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.stochf_js(high, low, close, 5, 3, 0);
    assert.strictEqual(result.length, high.length * 2); // k and d values
});

test('StochF accuracy', () => {
    // Test StochF matches expected values from Rust tests
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.stochf_js(high, low, close, 5, 3, 0);
    
    // Split result into k and d arrays
    const k = result.slice(0, high.length);
    const d = result.slice(high.length);
    
    assert.strictEqual(k.length, close.length);
    assert.strictEqual(d.length, close.length);
    
    // Expected values from Rust tests
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
    
    // Check last 5 values match expected
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
    // Test StochF fails with zero period
    const data = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0]);
    
    assert.throws(() => {
        wasm.stochf_js(data, data, data, 0, 3, 0);
    }, /Invalid period/);
});

test('StochF period exceeds length', () => {
    // Test StochF fails when period exceeds data length
    const data = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.stochf_js(data, data, data, 10, 3, 0);
    }, /Invalid period/);
});

test('StochF empty input', () => {
    // Test StochF fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.stochf_js(empty, empty, empty, 5, 3, 0);
    }, /Empty data|Input data slice is empty/);
});

test('StochF all NaN input', () => {
    // Test StochF with all NaN values
    const all_nan = new Float64Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.stochf_js(all_nan, all_nan, all_nan, 5, 3, 0);
    }, /All values are NaN/);
});

test('StochF nan handling', () => {
    // Test StochF handles NaN values correctly
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.stochf_js(high, low, close, 5, 3, 0);
    const k = result.slice(0, high.length);
    const d = result.slice(high.length);
    
    // After warmup period, no NaN values should exist
    // Warmup for K is fastk_period - 1 = 4
    // Warmup for D is fastk_period - 1 + fastd_period - 1 = 6
    if (k.length > 10) {
        // Check K values after warmup
        for (let i = 10; i < k.length; i++) {
            assert(!isNaN(k[i]), `Found unexpected NaN in K at index ${i}`);
        }
        
        // Check D values after warmup
        for (let i = 10; i < d.length; i++) {
            assert(!isNaN(d[i]), `Found unexpected NaN in D at index ${i}`);
        }
    }
    
    // First fastk_period-1 values should be NaN for K
    for (let i = 0; i < Math.min(4, k.length); i++) {
        assert(isNaN(k[i]), `Expected NaN in K warmup period at index ${i}`);
    }
    
    // First fastk_period-1 + fastd_period-1 values should be NaN for D
    for (let i = 0; i < Math.min(6, d.length); i++) {
        assert(isNaN(d[i]), `Expected NaN in D warmup period at index ${i}`);
    }
});

test('StochF fast API', () => {
    // Test the fast API - simplified test without direct memory access
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    // Use the regular JS API which handles memory internally
    const result = wasm.stochf_js(high, low, close, 5, 3, 0);
    
    // Result should contain both k and d values concatenated
    assert.strictEqual(result.length, high.length * 2);
    
    // Split the result into k and d
    const k_result = result.slice(0, high.length);
    const d_result = result.slice(high.length);
    
    // Verify warmup periods have NaN
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
    
    // Verify some values are not NaN after warmup
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
    // Test batch with single parameter combination
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    // Using single parameter set
    const config = {
        fastk_range: [5, 5, 0],
        fastd_range: [3, 3, 0],
        fastd_matype: 0
    };
    
    const batch_result = wasm.stochf_batch(high, low, close, config);
    
    // Should match single calculation
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
    // Test batch with multiple period values
    const high = new Float64Array(testData.high.slice(0, 100)); // Use smaller dataset for speed
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Multiple periods: fastk=[5,7,9], fastd=[3,3,3]
    const config = {
        fastk_range: [5, 9, 2],    // 5, 7, 9
        fastd_range: [3, 3, 0],     // 3
        fastd_matype: 0
    };
    
    const batch_result = wasm.stochf_batch(high, low, close, config);
    
    // Should have 3 rows * 100 cols
    assert.strictEqual(batch_result.rows, 3);
    assert.strictEqual(batch_result.cols, 100);
    assert.strictEqual(batch_result.k_values.length, 300);
    assert.strictEqual(batch_result.d_values.length, 300);
    assert.strictEqual(batch_result.combos.length, 3);
    
    // Verify each row matches individual calculation
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
    // Test that batch result includes correct parameter combinations
    const high = new Float64Array(testData.high.slice(0, 20));
    const low = new Float64Array(testData.low.slice(0, 20));
    const close = new Float64Array(testData.close.slice(0, 20));
    
    const config = {
        fastk_range: [5, 7, 2],     // 5, 7
        fastd_range: [3, 4, 1],     // 3, 4
        fastd_matype: 0
    };
    
    const result = wasm.stochf_batch(high, low, close, config);
    
    // Should have 2 * 2 = 4 combinations
    assert.strictEqual(result.combos.length, 4);
    
    // Check combinations
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