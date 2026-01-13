
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

test('NET_MYRSI partial params', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.net_myrsi_js(close, 14);
    assert.strictEqual(result.length, close.length);
});

test('NET_MYRSI accuracy', () => {

    const close = new Float64Array(testData.close);


    const period = 14;

    const result = wasm.net_myrsi_js(close, period);

    assert.strictEqual(result.length, close.length);


    const expected_last_five = [
        0.64835165,
        0.49450549,
        0.29670330,
        0.07692308,
        -0.07692308,
    ];


    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected_last_five,
        1e-7,
        "NET_MYRSI last 5 values mismatch"
    );
});

test('NET_MYRSI default params', () => {

    const close = new Float64Array(testData.close);


    const result = wasm.net_myrsi_js(close, 14);
    assert.strictEqual(result.length, close.length);
});

test('NET_MYRSI zero period', () => {

    const input_data = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.net_myrsi_js(input_data, 0);
    }, /Invalid period/, 'Should throw error for zero period');
});

test('NET_MYRSI period exceeds length', () => {

    const data_small = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.net_myrsi_js(data_small, 10);
    }, /Invalid period/, 'Should throw error when period exceeds data length');
});

test('NET_MYRSI very small dataset', () => {


    const data_small = new Float64Array([10.0, 20.0, 30.0, 15.0, 25.0]);

    const result = wasm.net_myrsi_js(data_small, 3);
    assert.strictEqual(result.length, data_small.length);


    assert(isNaN(result[0]), 'First value should be NaN');
    assert(isNaN(result[1]), 'Second value should be NaN');

    assert(!result.every(isNaN), 'Should have some valid values');
});

test('NET_MYRSI empty input', () => {

    const input_data = new Float64Array([]);

    assert.throws(() => {
        wasm.net_myrsi_js(input_data, 14);
    }, /Input data slice is empty/, 'Should throw error for empty input');
});

test('NET_MYRSI all NaN', () => {

    const input_data = new Float64Array(30);
    input_data.fill(NaN);

    assert.throws(() => {
        wasm.net_myrsi_js(input_data, 14);
    }, /All values are NaN/, 'Should throw error for all NaN input');
});

test('NET_MYRSI insufficient data', () => {


    const input_data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);

    assert.throws(() => {
        wasm.net_myrsi_js(input_data, 10);
    }, /(Invalid period|Not enough valid data)/, 'Should throw error for insufficient data');
});

test('NET_MYRSI nan handling', () => {


    const data = new Float64Array(30);
    for (let i = 0; i < 10; i++) {
        data[i] = i + 1.0;
    }
    for (let i = 10; i < 30; i++) {
        data[i] = data[i - 1] + 1.0;
    }

    const period = 14;


    const dataWithNaN = new Float64Array(data);
    dataWithNaN[15] = NaN;

    const result = wasm.net_myrsi_js(dataWithNaN, period);
    assert.strictEqual(result.length, dataWithNaN.length);





    let hasValidBefore = false;
    for (let i = 0; i < 15 && i < result.length; i++) {
        if (!isNaN(result[i])) {
            hasValidBefore = true;
            break;
        }
    }
    assert(hasValidBefore, 'Should have valid values before NaN');

    let hasValidAfter = false;
    for (let i = 16; i < result.length; i++) {
        if (!isNaN(result[i])) {
            hasValidAfter = true;
            break;
        }
    }
    assert(hasValidAfter, 'Should have valid values after NaN');


    const dataMultiNaN = new Float64Array(data);
    dataMultiNaN[10] = NaN;
    dataMultiNaN[20] = NaN;


    const result2 = wasm.net_myrsi_js(dataMultiNaN, period);
    assert.strictEqual(result2.length, dataMultiNaN.length);
    assert(Array.isArray(result2) || result2 instanceof Float64Array, 'Should return array');
});

test('NET_MYRSI warmup nans', () => {

    const close = new Float64Array(testData.close);
    const period = 14;


    let first_valid = 0;
    for (let i = 0; i < close.length; i++) {
        if (!isNaN(close[i])) {
            first_valid = i;
            break;
        }
    }

    const result = wasm.net_myrsi_js(close, period);



    const warmup = first_valid + period - 1;


    for (let i = 0; i < warmup; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} (warmup period)`);
    }



    const actual_start = first_valid + period;
    if (actual_start < result.length) {
        assert(!isNaN(result[actual_start]),
               `Expected valid value at index ${actual_start} (first computed value)`);
    }


    if (actual_start + 5 < result.length) {
        for (let i = actual_start; i < actual_start + 5; i++) {
            assert(!isNaN(result[i]), `Expected valid value at index ${i} (after warmup)`);
        }
    }
});

test('NET_MYRSI with NaN prefix', () => {

    const close = new Float64Array(testData.close);


    close[0] = NaN;
    close[1] = NaN;
    close[2] = NaN;

    const result = wasm.net_myrsi_js(close, 14);
    assert.strictEqual(result.length, close.length);


    assert(isNaN(result[0]), 'First value should be NaN');
    assert(isNaN(result[1]), 'Second value should be NaN');
    assert(isNaN(result[2]), 'Third value should be NaN');
});

test('NET_MYRSI memory allocation functions', () => {

    const len = 100;


    const ptr = wasm.net_myrsi_alloc(len);
    assert(ptr !== 0, 'Pointer should not be null');


    wasm.net_myrsi_free(ptr, len);

});

test('NET_MYRSI consistency check', () => {

    const close = new Float64Array(testData.close);
    const period = 14;

    const result1 = wasm.net_myrsi_js(close, period);
    const result2 = wasm.net_myrsi_js(close, period);

    assertArrayClose(
        result1,
        result2,
        1e-10,
        "Results should be identical for same input"
    );
});

test('NET_MYRSI different periods', () => {

    const close = new Float64Array(testData.close);

    const result10 = wasm.net_myrsi_js(close, 10);
    const result14 = wasm.net_myrsi_js(close, 14);
    const result20 = wasm.net_myrsi_js(close, 20);

    assert.strictEqual(result10.length, close.length);
    assert.strictEqual(result14.length, close.length);
    assert.strictEqual(result20.length, close.length);


    const last10 = result10[result10.length - 1];
    const last14 = result14[result14.length - 1];
    const last20 = result20[result20.length - 1];

    assert(Math.abs(last10 - last14) > 1e-10, 'Results should differ for different periods');
    assert(Math.abs(last14 - last20) > 1e-10, 'Results should differ for different periods');
});





