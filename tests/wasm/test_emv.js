
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

test('EMV basic calculation', () => {

    const high = new Float64Array([10.0, 12.0, 13.0, 15.0]);
    const low = new Float64Array([5.0, 7.0, 8.0, 10.0]);
    const close = new Float64Array([7.5, 9.0, 10.5, 12.5]);
    const volume = new Float64Array([10000.0, 20000.0, 25000.0, 30000.0]);

    const result = wasm.emv_js(high, low, close, volume);
    assert.strictEqual(result.length, 4);
    assert(isNaN(result[0]));
    assert(!isNaN(result[1]));







    assertClose(result[1], 5.0, 0.01, "EMV calculation at index 1");
});

test('EMV accuracy', async () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);

    const result = wasm.emv_js(high, low, close, volume);
    assert.strictEqual(result.length, high.length);


    const expected_last_five = [
        -6488905.579799851,
        2371436.7401001123,
        -3855069.958128531,
        1051939.877943717,
        -8519287.22257077,
    ];

    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected_last_five,
        100,
        "EMV last 5 values mismatch"
    );


    await compareWithRust('emv', result, 'ohlcv');
});

test('EMV warmup period', () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);

    const result = wasm.emv_js(high, low, close, volume);


    assert(isNaN(result[0]), "First EMV value should be NaN (warmup)");


    let firstValid = null;
    for (let i = 0; i < high.length; i++) {
        if (!isNaN(high[i]) && !isNaN(low[i]) && !isNaN(volume[i])) {
            firstValid = i;
            break;
        }
    }

    if (firstValid !== null && firstValid + 1 < result.length) {

        assert(!isNaN(result[firstValid + 1]),
               `Expected valid EMV at index ${firstValid + 1} after first valid data`);
    }
});

test('EMV empty data', () => {

    const empty = new Float64Array([]);

    assert.throws(() => {
        wasm.emv_js(empty, empty, empty, empty);
    }, /input data slice is empty|Empty data|EmptyData/i);
});

test('EMV all NaN', () => {

    const nanArr = new Float64Array([NaN, NaN]);

    assert.throws(() => {
        wasm.emv_js(nanArr, nanArr, nanArr, nanArr);
    }, /all values are nan|AllValuesNaN/i);
});

test('EMV not enough data', () => {

    const high = new Float64Array([10000.0, NaN]);
    const low = new Float64Array([9990.0, NaN]);
    const close = new Float64Array([9995.0, NaN]);
    const volume = new Float64Array([1_000_000.0, NaN]);

    assert.throws(() => {
        wasm.emv_js(high, low, close, volume);
    }, /not enough valid data|Not enough data|NotEnoughData/i);
});

test('EMV partial NaN handling', () => {

    const high = new Float64Array([NaN, 12.0, 15.0, NaN, 13.0, 16.0]);
    const low = new Float64Array([NaN, 9.0, 11.0, NaN, 10.0, 12.0]);
    const close = new Float64Array([NaN, 10.0, 13.0, NaN, 11.5, 14.0]);
    const volume = new Float64Array([NaN, 10000.0, 20000.0, NaN, 15000.0, 25000.0]);

    const result = wasm.emv_js(high, low, close, volume);


    assert.strictEqual(result.length, high.length);


    assert(isNaN(result[0]));
    assert(isNaN(result[1]));


    assert(!isNaN(result[2]));
});

test('EMV zero range handling', () => {

    const high = new Float64Array([10.0, 10.0, 12.0, 13.0]);
    const low = new Float64Array([9.0, 10.0, 11.0, 12.0]);
    const close = new Float64Array([9.5, 10.0, 11.5, 12.5]);
    const volume = new Float64Array([1000.0, 2000.0, 3000.0, 4000.0]);

    const result = wasm.emv_js(high, low, close, volume);


    assert(isNaN(result[1]), "Expected NaN when range is zero");


    assert(!isNaN(result[2]));
});

test('EMV mismatched lengths', () => {

    const high = new Float64Array([10.0, 12.0, 13.0]);
    const low = new Float64Array([9.0, 11.0]);
    const close = new Float64Array([9.5, 11.5, 12.0]);
    const volume = new Float64Array([1000.0, 2000.0, 3000.0]);


    const result = wasm.emv_js(high, low, close, volume);
    assert.strictEqual(result.length, 2);
});

test('EMV batch operations', () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);


    const config = {};

    const result = wasm.emv_batch(high, low, close, volume, config);

    assert(result.values, "Expected values array");
    assert.strictEqual(result.rows, 1, "EMV batch should have 1 row (no parameter sweep)");
    assert.strictEqual(result.cols, high.length, "Expected cols to match input length");


    const singleResult = wasm.emv_js(high, low, close, volume);
    assertArrayClose(
        new Float64Array(result.values),
        singleResult,
        1e-10,
        "Batch values should match single calculation"
    );


    const expected_last_five = [
        -6488905.579799851,
        2371436.7401001123,
        -3855069.958128531,
        1051939.877943717,
        -8519287.22257077,
    ];

    const batchLast5 = result.values.slice(-5);
    assertArrayClose(
        batchLast5,
        expected_last_five,
        100,
        "EMV batch last 5 values mismatch"
    );
});

test('EMV SIMD consistency', () => {

    const testCases = [
        { size: 10 },
        { size: 100 },
        { size: 1000 },
        { size: 10000 }
    ];

    for (const testCase of testCases) {

        const high = new Float64Array(testCase.size);
        const low = new Float64Array(testCase.size);
        const close = new Float64Array(testCase.size);
        const volume = new Float64Array(testCase.size);

        for (let i = 0; i < testCase.size; i++) {
            const base = 100 + Math.sin(i * 0.1) * 10;
            high[i] = base + Math.random() * 5;
            low[i] = base - Math.random() * 5;
            close[i] = (high[i] + low[i]) / 2;
            volume[i] = 10000 + Math.random() * 5000;
        }

        const result = wasm.emv_js(high, low, close, volume);


        assert.strictEqual(result.length, testCase.size);


        assert(isNaN(result[0]), `Expected NaN at warmup index 0 for size=${testCase.size}`);


        let sumAfterWarmup = 0;
        let countAfterWarmup = 0;
        for (let i = 1; i < result.length; i++) {
            if (!isNaN(result[i])) {
                sumAfterWarmup += Math.abs(result[i]);
                countAfterWarmup++;
            }
        }


        assert(countAfterWarmup > 0, `No valid values found for size=${testCase.size}`);


        const avgAfterWarmup = sumAfterWarmup / countAfterWarmup;
        assert(isFinite(avgAfterWarmup), `Average value ${avgAfterWarmup} is not finite`);
    }
});

test('EMV zero-copy (fast) API', () => {

    const size = 100;


    const highPtr = wasm.emv_alloc(size);
    const lowPtr = wasm.emv_alloc(size);
    const closePtr = wasm.emv_alloc(size);
    const volumePtr = wasm.emv_alloc(size);
    const outputPtr = wasm.emv_alloc(size);

    assert(highPtr !== 0, "Failed to allocate high buffer");
    assert(lowPtr !== 0, "Failed to allocate low buffer");
    assert(closePtr !== 0, "Failed to allocate close buffer");
    assert(volumePtr !== 0, "Failed to allocate volume buffer");
    assert(outputPtr !== 0, "Failed to allocate output buffer");

    try {

        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, size);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, size);
        const closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, size);
        const volumeView = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, size);
        const outputView = new Float64Array(wasm.__wasm.memory.buffer, outputPtr, size);


        for (let i = 0; i < size; i++) {
            const base = 100 + Math.sin(i * 0.1) * 10;
            highView[i] = base + 2;
            lowView[i] = base - 2;
            closeView[i] = base;
            volumeView[i] = 10000 + i * 100;
        }


        wasm.emv_into(
            highPtr,
            lowPtr,
            closePtr,
            volumePtr,
            outputPtr,
            size
        );


        assert(isNaN(outputView[0]), "First value should be NaN");
        assert(!isNaN(outputView[10]), "Should have valid values after warmup");


        const highArray = new Float64Array(highView);
        const lowArray = new Float64Array(lowView);
        const closeArray = new Float64Array(closeView);
        const volumeArray = new Float64Array(volumeView);

        const regularResult = wasm.emv_js(highArray, lowArray, closeArray, volumeArray);

        for (let i = 0; i < size; i++) {
            if (isNaN(regularResult[i]) && isNaN(outputView[i])) {
                continue;
            }
            assert(Math.abs(regularResult[i] - outputView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${outputView[i]}`);
        }
    } finally {

        wasm.emv_free(highPtr, size);
        wasm.emv_free(lowPtr, size);
        wasm.emv_free(closePtr, size);
        wasm.emv_free(volumePtr, size);
        wasm.emv_free(outputPtr, size);
    }
});

test('EMV zero-copy API with aliasing', () => {

    const size = 50;


    const highPtr = wasm.emv_alloc(size);
    const lowPtr = wasm.emv_alloc(size);
    const closePtr = wasm.emv_alloc(size);
    const volumePtr = wasm.emv_alloc(size);

    assert(highPtr !== 0 && lowPtr !== 0 && closePtr !== 0 && volumePtr !== 0,
           "Failed to allocate memory");

    try {

        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, size);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, size);
        const closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, size);
        const volumeView = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, size);


        for (let i = 0; i < size; i++) {
            highView[i] = 100 + i;
            lowView[i] = 90 + i;
            closeView[i] = 95 + i;
            volumeView[i] = 10000;
        }


        const originalClose = new Float64Array(closeView);


        wasm.emv_into(
            highPtr,
            lowPtr,
            closePtr,
            volumePtr,
            closePtr,
            size
        );


        assert(isNaN(closeView[0]), "First EMV value should be NaN");
        assert(closeView[1] !== originalClose[1], "Values should have changed");
    } finally {
        wasm.emv_free(highPtr, size);
        wasm.emv_free(lowPtr, size);
        wasm.emv_free(closePtr, size);
        wasm.emv_free(volumePtr, size);
    }
});

test('EMV memory leak prevention', () => {

    const sizes = [100, 1000, 10000, 100000];

    for (const size of sizes) {
        const ptr = wasm.emv_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);


        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }


        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
        }


        wasm.emv_free(ptr, size);
    }


    for (let round = 0; round < 10; round++) {
        const ptrs = [];


        for (let i = 0; i < 5; i++) {
            const ptr = wasm.emv_alloc(1000);
            assert(ptr !== 0, `Failed to allocate in round ${round}, buffer ${i}`);
            ptrs.push(ptr);
        }


        for (const ptr of ptrs) {
            wasm.emv_free(ptr, 1000);
        }
    }
});

test('EMV batch into (fast API)', () => {

    const size = 100;


    const highPtr = wasm.emv_alloc(size);
    const lowPtr = wasm.emv_alloc(size);
    const closePtr = wasm.emv_alloc(size);
    const volumePtr = wasm.emv_alloc(size);
    const outputPtr = wasm.emv_alloc(size);

    try {

        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, size);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, size);
        const closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, size);
        const volumeView = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, size);
        const outputView = new Float64Array(wasm.__wasm.memory.buffer, outputPtr, size);

        for (let i = 0; i < size; i++) {
            highView[i] = 100 + i * 0.5;
            lowView[i] = 95 + i * 0.5;
            closeView[i] = 97.5 + i * 0.5;
            volumeView[i] = 10000;
        }

        const rows = wasm.emv_batch_into(
            highPtr,
            lowPtr,
            closePtr,
            volumePtr,
            outputPtr,
            size
        );

        assert.strictEqual(rows, 1, "Expected 1 row for EMV batch");
        assert(isNaN(outputView[0]), "First value should be NaN");
        assert(!isNaN(outputView[10]), "Should have valid values after warmup");
    } finally {
        wasm.emv_free(highPtr, size);
        wasm.emv_free(lowPtr, size);
        wasm.emv_free(closePtr, size);
        wasm.emv_free(volumePtr, size);
        wasm.emv_free(outputPtr, size);
    }
});

test('EMV null pointer handling', () => {

    assert.throws(() => {
        wasm.emv_into(0, 0, 0, 0, 0, 100);
    }, /null pointer/);
});

test.after(() => {
    console.log('EMV WASM tests completed');
});
