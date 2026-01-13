
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

test('REVERSE_RSI partial params', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.reverse_rsi_js(close, 14, 50.0);
    assert.strictEqual(result.length, close.length);
});

test('REVERSE_RSI accuracy', () => {


    const close = new Float64Array(testData.close);


    const rsiLength = 14;
    const rsiLevel = 50.0;

    const result = wasm.reverse_rsi_js(close, rsiLength, rsiLevel);


    assert.strictEqual(result.length, close.length);



    const expectedLast5 = [
        60124.655535277416, 60064.68013990046, 60001.56012990757, 59932.80583491417, 59877.248275277445
    ];

    const last5 = result.slice(-6, -1);
    for (let i = 0; i < 5; i++) {
        assertClose(
            last5[i],
            expectedLast5[i],
            1e-6,
            `REVERSE_RSI last 5 values mismatch at index ${i}`
        );
    }
});

test('REVERSE_RSI default params', () => {

    const close = new Float64Array(testData.close);


    const result = wasm.reverse_rsi_js(close, 14, 50.0);
    assert.strictEqual(result.length, close.length);
});

test('REVERSE_RSI zero period', () => {

    const input_data = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.reverse_rsi_js(input_data, 0, 50.0);
    }, /Invalid period/, 'Should throw error for zero period');
});

test('REVERSE_RSI period exceeds length', () => {

    const data_small = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.reverse_rsi_js(data_small, 10, 50.0);
    }, /Invalid period/, 'Should throw error when period exceeds data length');
});

test('REVERSE_RSI invalid level', () => {

    const input_data = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0, 10.0, 20.0, 30.0, 40.0, 50.0,
                                          10.0, 20.0, 30.0, 40.0, 50.0, 10.0, 20.0, 30.0, 40.0, 50.0,
                                          10.0, 20.0, 30.0, 40.0, 50.0]);


    assert.throws(() => {
        wasm.reverse_rsi_js(input_data, 14, 150.0);
    }, /Invalid RSI level/, 'Should throw error for RSI level > 100');


    assert.throws(() => {
        wasm.reverse_rsi_js(input_data, 14, -10.0);
    }, /Invalid RSI level/, 'Should throw error for negative RSI level');
});

test('REVERSE_RSI edge levels', () => {

    const input_data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        input_data[i] = 10.0 + (i % 5) * 10.0;
    }


    const result01 = wasm.reverse_rsi_js(input_data, 14, 0.01);
    assert.strictEqual(result01.length, input_data.length);

    assert(!result01.every(isNaN), 'Should have some valid values for RSI level = 0.01');


    const result1 = wasm.reverse_rsi_js(input_data, 14, 1.0);
    assert.strictEqual(result1.length, input_data.length);

    assert(!result1.every(isNaN), 'Should have some valid values for RSI level = 1.0');


    const result99 = wasm.reverse_rsi_js(input_data, 14, 99.0);
    assert.strictEqual(result99.length, input_data.length);

    assert(!result99.every(isNaN), 'Should have some valid values for RSI level = 99.0');
});

test('REVERSE_RSI various levels', () => {

    const input_data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        input_data[i] = 10.0 + i;
    }

    const levels = [20.0, 30.0, 50.0, 70.0, 80.0];
    const results = [];

    for (const level of levels) {
        const result = wasm.reverse_rsi_js(input_data, 14, level);
        assert.strictEqual(result.length, input_data.length);
        results.push(result);
    }


    for (let i = 0; i < results.length - 1; i++) {

        let foundDifference = false;
        for (let j = 0; j < results[i].length; j++) {
            if (!isNaN(results[i][j]) && !isNaN(results[i+1][j])) {
                if (Math.abs(results[i][j] - results[i+1][j]) > 1e-10) {
                    foundDifference = true;
                    break;
                }
            }
        }
        assert(foundDifference, `Different RSI levels (${levels[i]} vs ${levels[i+1]}) should produce different results`);
    }
});

test('REVERSE_RSI empty input', () => {

    const input_data = new Float64Array([]);

    assert.throws(() => {
        wasm.reverse_rsi_js(input_data, 14, 50.0);
    }, /Input data slice is empty/, 'Should throw error for empty input');
});

test('REVERSE_RSI all NaN', () => {

    const input_data = new Float64Array(30);
    input_data.fill(NaN);

    assert.throws(() => {
        wasm.reverse_rsi_js(input_data, 14, 50.0);
    }, /All values are NaN/, 'Should throw error for all NaN input');
});

test('REVERSE_RSI insufficient data', () => {



    const input_data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);

    assert.throws(() => {
        wasm.reverse_rsi_js(input_data, 14, 50.0);
    }, /(Invalid period|Not enough valid data)/, 'Should throw error for insufficient data');
});

test('REVERSE_RSI nan handling', () => {


    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = i + 1.0;
    }

    const rsiLength = 14;
    const rsiLevel = 50.0;


    const dataWithNaN = new Float64Array(data);
    dataWithNaN[25] = NaN;

    const result = wasm.reverse_rsi_js(dataWithNaN, rsiLength, rsiLevel);
    assert.strictEqual(result.length, dataWithNaN.length);



    let hasValidAfter = false;
    for (let i = 26; i < result.length; i++) {
        if (!isNaN(result[i])) {
            hasValidAfter = true;
            break;
        }
    }
    assert(hasValidAfter, 'Should have valid values after warmup/NaN');


    const dataMultiNaN = new Float64Array(data);
    dataMultiNaN[20] = NaN;
    dataMultiNaN[30] = NaN;


    const result2 = wasm.reverse_rsi_js(dataMultiNaN, rsiLength, rsiLevel);
    assert.strictEqual(result2.length, dataMultiNaN.length);
    assert(result2 instanceof Float64Array, 'Should return Float64Array');
});

test('REVERSE_RSI warmup nans', () => {

    const close = new Float64Array(testData.close);
    const rsiLength = 14;
    const rsiLevel = 50.0;


    let firstValid = 0;
    for (let i = 0; i < close.length; i++) {
        if (!isNaN(close[i])) {
            firstValid = i;
            break;
        }
    }

    const result = wasm.reverse_rsi_js(close, rsiLength, rsiLevel);


    for (let i = 0; i < firstValid; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} (before first valid data)`);
    }
});

test('REVERSE_RSI memory management with into', () => {

    const close = new Float64Array(testData.close.slice(0, 50));
    const rsiLength = 14;
    const rsiLevel = 50.0;


    const outPtr = wasm.reverse_rsi_alloc(close.length);


    const inPtr = wasm.reverse_rsi_alloc(close.length);
    const wasmMemory = new Float64Array(wasm.__wasm.memory.buffer);
    const inOffset = inPtr / 8;


    for (let i = 0; i < close.length; i++) {
        wasmMemory[inOffset + i] = close[i];
    }


    wasm.reverse_rsi_into(inPtr, outPtr, close.length, rsiLength, rsiLevel);


    const outOffset = outPtr / 8;
    const output = new Float64Array(close.length);
    for (let i = 0; i < close.length; i++) {
        output[i] = wasmMemory[outOffset + i];
    }


    wasm.reverse_rsi_free(inPtr, close.length);
    wasm.reverse_rsi_free(outPtr, close.length);


    const expected = wasm.reverse_rsi_js(close, rsiLength, rsiLevel);
    assertArrayClose(output, expected, 1e-10, 'Memory-managed version should match normal version');
});

test('REVERSE_RSI batch processing', () => {

    const close = new Float64Array(testData.close.slice(0, 100));


    const config = {
        rsi_length_range: [10, 20, 5],
        rsi_level_range: [50.0, 50.0, 0.0]
    };

    const result = wasm.reverse_rsi_batch(close, config);


    assert(result.values, 'Should have values array');
    assert(result.combos, 'Should have combos array');
    assert.strictEqual(result.rows, 3, 'Should have 3 parameter combinations');
    assert.strictEqual(result.cols, close.length, 'Should have same columns as input');


    assert.strictEqual(result.combos.length, 3, 'Should have 3 combinations');
    const expectedLengths = [10, 15, 20];
    for (let i = 0; i < 3; i++) {
        assert.strictEqual(result.combos[i].rsi_length, expectedLengths[i],
            `Combo ${i} should have rsi_length ${expectedLengths[i]}`);
        assert.strictEqual(result.combos[i].rsi_level, 50.0,
            `Combo ${i} should have rsi_level 50.0`);
    }


    for (let row = 0; row < result.rows; row++) {
        const rowStart = row * result.cols;
        const rowData = result.values.slice(rowStart, rowStart + result.cols);


        let hasValid = false;
        for (let i = 0; i < rowData.length; i++) {
            if (!isNaN(rowData[i])) {
                hasValid = true;
                break;
            }
        }
        assert(hasValid, `Row ${row} should have some valid values`);
    }


    if (wasm.reverse_rsi_batch_into) {
        const len = close.length;


        const inPtr = wasm.reverse_rsi_alloc(len);
        const outPtr = wasm.reverse_rsi_alloc(len * 3);


        const wasmMemory = new Float64Array(wasm.__wasm.memory.buffer);
        const inOffset = inPtr / 8;
        for (let i = 0; i < len; i++) {
            wasmMemory[inOffset + i] = close[i];
        }


        const numRows = wasm.reverse_rsi_batch_into(
            inPtr, outPtr, len,
            10, 20, 5,
            50.0, 50.0, 0.0
        );

        assert.strictEqual(numRows, 3, 'Should return 3 rows');


        const outOffset = outPtr / 8;
        const output = new Float64Array(len * 3);
        for (let i = 0; i < output.length; i++) {
            output[i] = wasmMemory[outOffset + i];
        }


        wasm.reverse_rsi_free(inPtr, len);
        wasm.reverse_rsi_free(outPtr, len * 3);


        for (let row = 0; row < 3; row++) {
            const rowStart = row * len;
            const rowData = output.slice(rowStart, rowStart + len);

            let hasValid = false;
            for (let i = 0; i < rowData.length; i++) {
                if (!isNaN(rowData[i])) {
                    hasValid = true;
                    break;
                }
            }
            assert(hasValid, `Batch_into row ${row} should have some valid values`);
        }
    }
});

test('REVERSE_RSI numerical precision', () => {



    const extremeData = new Float64Array(40);
    for (let i = 0; i < 10; i++) {
        extremeData[i * 4] = 1e-10;
        extremeData[i * 4 + 1] = 1e10;
        extremeData[i * 4 + 2] = 1e-10;
        extremeData[i * 4 + 3] = 1e10;
    }

    const result1 = wasm.reverse_rsi_js(extremeData, 5, 50.0);
    assert.strictEqual(result1.length, extremeData.length);

    const validValues1 = result1.filter(v => !isNaN(v));
    assert(!validValues1.some(v => !isFinite(v)), 'Should not produce infinity');


    const smallDiffData = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        smallDiffData[i] = 100.0 + i * 1e-10;
    }

    const result2 = wasm.reverse_rsi_js(smallDiffData, 10, 50.0);
    assert.strictEqual(result2.length, smallDiffData.length);

    const validValues2 = result2.filter(v => !isNaN(v));
    if (validValues2.length > 0) {
        assert(!validValues2.some(v => !isFinite(v)), 'Should not produce infinity');
    }


    const constantData = new Float64Array(30);
    constantData.fill(100.0);

    const result3 = wasm.reverse_rsi_js(constantData, 10, 50.0);
    assert.strictEqual(result3.length, constantData.length);

    const validValues3 = result3.filter(v => !isNaN(v));
    if (validValues3.length > 0) {
        assert(!validValues3.some(v => !isFinite(v)), 'Should not produce infinity with constant values');
    }
});