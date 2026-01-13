
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

test('KVO accuracy', () => {
    const high = testData.high;
    const low = testData.low;
    const close = testData.close;
    const volume = testData.volume;


    const result = wasm.kvo_js(high, low, close, volume, 2, 5);

    assert.strictEqual(result.length, close.length, 'Result length should match input length');


    const expectedLastFive = [
        -246.42698280402647,
        530.8651474164992,
        237.2148311016648,
        608.8044103976362,
        -6339.615516805162,
    ];

    const actualLastFive = result.slice(-5);
    assertArrayClose(actualLastFive, expectedLastFive, 1e-1, 'KVO last 5 values mismatch');
});

test('KVO with default parameters', () => {
    const high = testData.high;
    const low = testData.low;
    const close = testData.close;
    const volume = testData.volume;


    const result = wasm.kvo_js(high, low, close, volume, 2, 5);
    assert(result && typeof result.length === 'number', 'Result should have a length property');
    assert.strictEqual(result.length, close.length, 'Result length should match input length');
});

test('KVO error handling - zero period', () => {
    const high = testData.high;
    const low = testData.low;
    const close = testData.close;
    const volume = testData.volume;

    assert.throws(
        () => wasm.kvo_js(high, low, close, volume, 0, 5),
        /Invalid/,
        'Should throw error for zero short period'
    );
});

test('KVO error handling - invalid period', () => {
    const high = testData.high;
    const low = testData.low;
    const close = testData.close;
    const volume = testData.volume;

    assert.throws(
        () => wasm.kvo_js(high, low, close, volume, 5, 2),
        /Invalid/,
        'Should throw error when long_period < short_period'
    );
});

test('KVO error handling - insufficient data', () => {
    const singlePoint = new Float64Array([42.0]);

    assert.throws(
        () => wasm.kvo_js(singlePoint, singlePoint, singlePoint, singlePoint, 2, 5),
        /Not enough valid data/,
        'Should throw error for insufficient data'
    );
});

test('KVO error handling - empty input', () => {
    const empty = new Float64Array([]);

    assert.throws(
        () => wasm.kvo_js(empty, empty, empty, empty, 2, 5),
        /empty/i,
        'Should throw error for empty input'
    );
});

test('KVO error handling - all NaN input', () => {
    const allNaN = new Float64Array(100).fill(NaN);

    assert.throws(
        () => wasm.kvo_js(allNaN, allNaN, allNaN, allNaN, 2, 5),
        /All values are NaN/,
        'Should throw error for all NaN values'
    );
});

test('KVO NaN handling', () => {
    const high = testData.high;
    const low = testData.low;
    const close = testData.close;
    const volume = testData.volume;

    const result = wasm.kvo_js(high, low, close, volume, 2, 5);
    assert.strictEqual(result.length, close.length, 'Result length should match input length');


    let nanCount = 0;
    for (let i = 0; i < result.length; i++) {
        if (isNaN(result[i])) {
            nanCount++;
        } else {
            break;
        }
    }


    assert(nanCount > 0, 'Should have NaN values during warmup period');
});

test('KVO batch processing', () => {
    const high = testData.high.slice(0, 100);
    const low = testData.low.slice(0, 100);
    const close = testData.close.slice(0, 100);
    const volume = testData.volume.slice(0, 100);


    const config = {
        short_period_range: [2, 2, 0],
        long_period_range: [5, 5, 0]
    };

    const result = wasm.kvo_batch(high, low, close, volume, config);

    assert(result.values, 'Result should have values array');
    assert(result.combos, 'Result should have combos array');
    assert.strictEqual(result.rows, 1, 'Should have 1 row for single parameter set');
    assert.strictEqual(result.cols, 100, 'Should have 100 columns');


    const singleResult = wasm.kvo_js(high, low, close, volume, 2, 5);
    assertArrayClose(
        result.values,
        singleResult,
        1e-10,
        'Batch result should match single calculation'
    );
});

test('KVO batch with multiple parameters', () => {
    const high = testData.high.slice(0, 50);
    const low = testData.low.slice(0, 50);
    const close = testData.close.slice(0, 50);
    const volume = testData.volume.slice(0, 50);


    const config = {
        short_period_range: [2, 3, 1],
        long_period_range: [5, 6, 1]
    };

    const result = wasm.kvo_batch(high, low, close, volume, config);


    assert.strictEqual(result.rows, 4, 'Should have 4 parameter combinations');
    assert.strictEqual(result.cols, 50, 'Should have 50 columns');
    assert.strictEqual(result.values.length, 200, 'Should have 4 * 50 = 200 values');


    const firstRow = result.values.slice(0, 50);
    const singleResult = wasm.kvo_js(high, low, close, volume, 2, 5);

    assertArrayClose(firstRow, singleResult, 1e-10, 'First batch row should match single calculation');
});

test('KVO memory allocation/deallocation', () => {
    const len = 1000;
    const ptr = wasm.kvo_alloc(len);

    assert(ptr !== 0, 'Allocated pointer should not be null');


    wasm.kvo_free(ptr, len);


    const ptrs = [];
    for (let i = 0; i < 10; i++) {
        ptrs.push(wasm.kvo_alloc(100));
    }


    ptrs.forEach(p => wasm.kvo_free(p, 100));
});

test('KVO fast API (kvo_into)', () => {
    const high = testData.high.slice(0, 100);
    const low = testData.low.slice(0, 100);
    const close = testData.close.slice(0, 100);
    const volume = testData.volume.slice(0, 100);
    const len = high.length;


    const highPtr = wasm.kvo_alloc(len);
    const lowPtr = wasm.kvo_alloc(len);
    const closePtr = wasm.kvo_alloc(len);
    const volumePtr = wasm.kvo_alloc(len);
    const outPtr = wasm.kvo_alloc(len);

    try {

        const highMem = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowMem = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        const closeMem = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        const volumeMem = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, len);

        highMem.set(new Float64Array(high));
        lowMem.set(new Float64Array(low));
        closeMem.set(new Float64Array(close));
        volumeMem.set(new Float64Array(volume));


        wasm.kvo_into(
            highPtr,
            lowPtr,
            closePtr,
            volumePtr,
            outPtr,
            len,
            2,
            5
        );


        const memory = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        const result = Array.from(memory);


        const expected = wasm.kvo_js(high, low, close, volume, 2, 5);
        assertArrayClose(result, expected, 1e-14, 'Fast API should match safe API');

    } finally {
        wasm.kvo_free(highPtr, len);
        wasm.kvo_free(lowPtr, len);
        wasm.kvo_free(closePtr, len);
        wasm.kvo_free(volumePtr, len);
        wasm.kvo_free(outPtr, len);
    }
});

test('KVO batch metadata verification', () => {
    const high = testData.high.slice(0, 50);
    const low = testData.low.slice(0, 50);
    const close = testData.close.slice(0, 50);
    const volume = testData.volume.slice(0, 50);


    const config = {
        short_period_range: [2, 4, 1],
        long_period_range: [5, 7, 2]
    };

    const result = wasm.kvo_batch(high, low, close, volume, config);


    assert.strictEqual(result.rows, 6, 'Should have 6 parameter combinations');
    assert.strictEqual(result.cols, 50, 'Should have 50 columns');
    assert.strictEqual(result.combos.length, 6, 'Should have 6 combo entries');


    const expectedCombos = [
        { short_period: 2, long_period: 5 },
        { short_period: 2, long_period: 7 },
        { short_period: 3, long_period: 5 },
        { short_period: 3, long_period: 7 },
        { short_period: 4, long_period: 5 },
        { short_period: 4, long_period: 7 }
    ];

    for (let i = 0; i < expectedCombos.length; i++) {
        assert.strictEqual(
            result.combos[i].short_period,
            expectedCombos[i].short_period,
            `Combo ${i} short_period mismatch`
        );
        assert.strictEqual(
            result.combos[i].long_period,
            expectedCombos[i].long_period,
            `Combo ${i} long_period mismatch`
        );
    }
});

test('KVO batch edge cases', () => {
    const size = 20;
    const high = new Float64Array(size);
    const low = new Float64Array(size);
    const close = new Float64Array(size);
    const volume = new Float64Array(size);


    for (let i = 0; i < size; i++) {
        high[i] = 100 + i;
        low[i] = 95 + i;
        close[i] = 97.5 + i;
        volume[i] = 1000 + i * 10;
    }


    const singleConfig = {
        short_period_range: [2, 2, 0],
        long_period_range: [5, 5, 0]
    };
    const singleResult = wasm.kvo_batch(high, low, close, volume, singleConfig);
    assert.strictEqual(singleResult.rows, 1, 'Single combo should have 1 row');
    assert.strictEqual(singleResult.combos.length, 1, 'Single combo should have 1 entry');


    const largeStepConfig = {
        short_period_range: [2, 3, 10],
        long_period_range: [5, 6, 10]
    };
    const largeStepResult = wasm.kvo_batch(high, low, close, volume, largeStepConfig);
    assert.strictEqual(largeStepResult.rows, 1, 'Large step should give 1 row');
    assert.strictEqual(largeStepResult.combos[0].short_period, 2, 'Should use start value');
    assert.strictEqual(largeStepResult.combos[0].long_period, 5, 'Should use start value');


    const allNaN = new Float64Array(size).fill(NaN);
    assert.throws(
        () => wasm.kvo_batch(allNaN, allNaN, allNaN, allNaN, singleConfig),
        /All values are NaN/,
        'Should throw error for all NaN values'
    );
});

test('KVO zero-copy batch API (kvo_batch_into)', () => {
    const high = testData.high.slice(0, 50);
    const low = testData.low.slice(0, 50);
    const close = testData.close.slice(0, 50);
    const volume = testData.volume.slice(0, 50);
    const len = high.length;


    const shortStart = 2, shortEnd = 3, shortStep = 1;
    const longStart = 5, longEnd = 6, longStep = 1;
    const rows = 2 * 2;
    const totalSize = rows * len;


    const highPtr = wasm.kvo_alloc(len);
    const lowPtr = wasm.kvo_alloc(len);
    const closePtr = wasm.kvo_alloc(len);
    const volumePtr = wasm.kvo_alloc(len);
    const outPtr = wasm.kvo_alloc(totalSize);

    try {

        const highMem = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowMem = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        const closeMem = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        const volumeMem = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, len);

        highMem.set(new Float64Array(high));
        lowMem.set(new Float64Array(low));
        closeMem.set(new Float64Array(close));
        volumeMem.set(new Float64Array(volume));


        const actualRows = wasm.kvo_batch_into(
            highPtr, lowPtr, closePtr, volumePtr,
            outPtr, len,
            shortStart, shortEnd, shortStep,
            longStart, longEnd, longStep
        );

        assert.strictEqual(actualRows, rows, 'Should return correct number of rows');


        const resultMem = new Float64Array(wasm.__wasm.memory.buffer, outPtr, totalSize);
        const result = Array.from(resultMem);


        const firstRow = result.slice(0, len);
        const expected = wasm.kvo_js(high, low, close, volume, 2, 5);
        assertArrayClose(firstRow, expected, 1e-10, 'First batch row should match single calc');

    } finally {
        wasm.kvo_free(highPtr, len);
        wasm.kvo_free(lowPtr, len);
        wasm.kvo_free(closePtr, len);
        wasm.kvo_free(volumePtr, len);
        wasm.kvo_free(outPtr, totalSize);
    }
});

test('KVO memory management', () => {

    const sizes = [100, 1000, 10000];
    const ptrs = [];


    for (const size of sizes) {
        const ptr = wasm.kvo_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        ptrs.push({ ptr, size });


        const mem = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            mem[i] = i * 1.5;
        }
    }


    for (const { ptr, size } of ptrs) {
        const mem = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(mem[i], i * 1.5, `Memory corruption at index ${i}`);
        }
    }


    for (const { ptr, size } of ptrs) {
        wasm.kvo_free(ptr, size);
    }


    const largeSize = 1000000;
    const largePtr = wasm.kvo_alloc(largeSize);
    assert(largePtr !== 0, 'Failed to allocate large buffer');
    wasm.kvo_free(largePtr, largeSize);
});

test('KVO SIMD consistency verification', () => {

    const testCases = [
        { size: 10 },
        { size: 100 },
        { size: 1000 },
        { size: 5000 }
    ];

    for (const testCase of testCases) {
        const size = testCase.size;
        const high = new Float64Array(size);
        const low = new Float64Array(size);
        const close = new Float64Array(size);
        const volume = new Float64Array(size);


        for (let i = 0; i < size; i++) {
            const base = 100 + Math.sin(i * 0.1) * 10;
            high[i] = base + Math.random() * 2;
            low[i] = base - Math.random() * 2;
            close[i] = (high[i] + low[i]) / 2;
            volume[i] = 1000 + Math.random() * 500;
        }


        const result = wasm.kvo_js(high, low, close, volume, 2, 5);


        assert.strictEqual(result.length, size, `Result length mismatch for size ${size}`);


        assert(isNaN(result[0]), `Expected NaN at warmup index 0 for size ${size}`);
        assert(!isNaN(result[1]), `Expected valid value at index 1 for size ${size}`);


        let hasValidValues = false;
        for (let i = 1; i < result.length; i++) {
            if (!isNaN(result[i])) {
                hasValidValues = true;

                assert(isFinite(result[i]), `Non-finite value at index ${i} for size ${size}`);
            }
        }
        assert(hasValidValues, `No valid values found for size ${size}`);
    }
});

test('KVO warmup period verification', () => {
    const size = 50;
    const high = new Float64Array(size);
    const low = new Float64Array(size);
    const close = new Float64Array(size);
    const volume = new Float64Array(size);


    for (let i = 0; i < size; i++) {
        high[i] = 100 + i;
        low[i] = 95 + i;
        close[i] = 97.5 + i;
        volume[i] = 1000 + i * 10;
    }

    const result = wasm.kvo_js(high, low, close, volume, 2, 5);



    assert(isNaN(result[0]), 'First value should be NaN during warmup');
    assert(!isNaN(result[1]), 'Second value should be valid after warmup');


    for (let i = 0; i < 5; i++) {
        high[i] = NaN;
        low[i] = NaN;
        close[i] = NaN;
        volume[i] = NaN;
    }

    const resultNaN = wasm.kvo_js(high, low, close, volume, 2, 5);


    for (let i = 0; i < 6; i++) {
        assert(isNaN(resultNaN[i]), `Expected NaN at index ${i} during warmup with NaN input`);
    }
    assert(!isNaN(resultNaN[6]), 'Expected valid value at index 6 after warmup');
});

test.after(() => {
    console.log('KVO WASM tests completed');
});
