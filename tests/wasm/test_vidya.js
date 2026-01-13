
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

test('VIDYA accuracy', (t) => {
    const close = testData.close;


    const result = wasm.vidya_js(close, 2, 5, 0.2);
    assert.strictEqual(result.length, close.length, 'Output length should match input length');


    for (let i = 0; i < 3; i++) {
        assert(isNaN(result[i]), `Value at index ${i} should be NaN during warmup`);
    }


    assertNoNaN(result.slice(3), 'Should have no NaN values after warmup period');


    if (result.length >= 5) {
        const expected_last_five = [
            59553.42785306692,
            59503.60445032524,
            59451.72283651444,
            59413.222561244685,
            59239.716526894175
        ];

        const start = result.length - 5;
        for (let i = 0; i < 5; i++) {
            assertClose(result[start + i], expected_last_five[i], 1e-1,
                `Last 5 values[${i}]`);
        }
    }
});

test('VIDYA fast API (vidya_into)', (t) => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const len = data.length;


    const inPtr = wasm.vidya_alloc(len);
    const outPtr = wasm.vidya_alloc(len);

    try {

        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inView.set(data);


        wasm.vidya_into(inPtr, outPtr, len, 2, 3, 0.2);


        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        assert.strictEqual(result.length, len);


        assert(isNaN(result[0]));


        inView.set(data);
        wasm.vidya_into(inPtr, inPtr, len, 2, 3, 0.2);


        const inPlaceResult = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);


        for (let i = 0; i < len; i++) {
            if (isNaN(result[i])) {
                assert(isNaN(inPlaceResult[i]), `In-place result at ${i} should also be NaN`);
            } else {
                assertClose(inPlaceResult[i], result[i], 1e-10, `In-place vs out-of-place at ${i}`);
            }
        }
    } finally {
        wasm.vidya_free(inPtr, len);
        wasm.vidya_free(outPtr, len);
    }
});




test('VIDYA error handling', (t) => {

    assert.throws(() => wasm.vidya_js([], 2, 5, 0.2), 'Should throw on empty data');


    assert.throws(() => wasm.vidya_js([1, 2, 3], 0, 5, 0.2), 'Should throw on invalid short period');
    assert.throws(() => wasm.vidya_js([1, 2, 3], 3, 2, 0.2), 'Should throw when short > long');
    assert.throws(() => wasm.vidya_js([1, 2, 3], 2, 5, -0.1), 'Should throw on negative alpha');
    assert.throws(() => wasm.vidya_js([1, 2, 3], 2, 5, 1.1), 'Should throw on alpha > 1');


    assert.throws(() => wasm.vidya_js([1, 2], 2, 5, 0.2), 'Should throw when data length < long period');
});

test('VIDYA zero-copy memory management', (t) => {
    const sizes = [100, 1000];

    for (const size of sizes) {
        const data = new Float64Array(size);
        for (let i = 0; i < size; i++) {
            data[i] = Math.sin(i * 0.1) * 100 + 50;
        }


        const inPtr = wasm.vidya_alloc(size);
        const outPtr = wasm.vidya_alloc(size);

        try {

            const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, size);
            inView.set(data);


            for (let alpha = 0.1; alpha <= 0.5; alpha += 0.1) {
                wasm.vidya_into(inPtr, outPtr, size, 2, 5, alpha);
                const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, size);


                assert.strictEqual(result.length, size);
                assert(isNaN(result[0]) || isNaN(result[1]) || isNaN(result[2]), 'Should have NaN values in warmup');
            }
        } finally {
            wasm.vidya_free(inPtr, size);
            wasm.vidya_free(outPtr, size);
        }
    }
});







test('VIDYA parameter validation', (t) => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);


    assert.throws(() => wasm.vidya_js(data, 1, 2, 0.5), 'Should throw on short_period=1');


    const result2 = wasm.vidya_js(data, 4, 5, 0.5);
    assert.strictEqual(result2.length, data.length);
});

test('VIDYA consistency across different data sizes', (t) => {
    const sizes = [10, 50, 100, 500];

    for (const size of sizes) {
        const data = new Float64Array(size);
        for (let i = 0; i < size; i++) {
            data[i] = Math.sin(i * 0.1) * 100 + 50;
        }

        const result = wasm.vidya_js(data, 2, 5, 0.2);


        assert.strictEqual(result.length, data.length);


        for (let i = 0; i < 3; i++) {
            assert(isNaN(result[i]), `Warmup at ${i} should be NaN`);
        }


        assertNoNaN(result.slice(3), 'No NaN after warmup');
    }
});

test('VIDYA reinput test', (t) => {

    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = Math.sin(i * 0.1) * 50 + 100;
    }


    const first = wasm.vidya_js(data, 2, 5, 0.2);


    const second = wasm.vidya_js(first, 2, 5, 0.2);


    assert.strictEqual(second.length, data.length);


    let nanCount = 0;
    for (let i = 0; i < second.length; i++) {
        if (isNaN(second[i])) nanCount++;
        else break;
    }


    assert(nanCount >= 3, `Should have at least 3 NaN values, got ${nanCount}`);
});
