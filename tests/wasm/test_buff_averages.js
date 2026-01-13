
import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';
import { createRequire } from 'module';
import {
    loadTestData,
    assertArrayClose,
    assertClose,
    isNaN,
    assertAllNaN,
    assertNoNaN
} from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

test.before(async () => {

    try {
        const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
        const wasmBinPath = path.join(__dirname, '../../pkg/vector_ta_bg.wasm');
        const importPath = process.platform === 'win32'
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
        const imported = await import(importPath);

        const wasmModule = (imported.default && typeof imported.default === 'object') ? imported.default : imported;
        if (typeof imported.default === 'function') {
            await imported.default();
        } else if (typeof wasmModule.default === 'function') {
            await wasmModule.default();
        } else if (typeof wasmModule.initSync === 'function' && fs.existsSync(wasmBinPath)) {
            wasmModule.initSync(fs.readFileSync(wasmBinPath));
        }
        wasm = wasmModule;
    } catch (error) {
        try {
            const require = createRequire(import.meta.url);
            wasm = require(path.join(__dirname, '../../pkg/vector_ta.js'));
        } catch {
            console.error('Failed to load WASM module. Run "wasm-pack build --target nodejs --out-name vector_ta --features wasm" first');
            throw error;
        }
    }

    testData = loadTestData();
});

test('Buff Averages accuracy', () => {

    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);


    const result = wasm.buff_averages_js(close, volume, 5, 20);


    const halfLen = close.length;
    const fast_buff = result.slice(0, halfLen);
    const slow_buff = result.slice(halfLen);

    assert.strictEqual(fast_buff.length, close.length);
    assert.strictEqual(slow_buff.length, close.length);


    const expected_fast = [
        58740.30855637,
        59132.28418702,
        59309.76658172,
        59266.10492431,
        59194.11908892,
    ];

    const expected_slow = [
        59209.26229392,
        59201.87047432,
        59217.15739355,
        59195.74527194,
        59196.26139533,
    ];


    const last6_fast = fast_buff.slice(-6);
    const last6_slow = slow_buff.slice(-6);
    const last5_fast = last6_fast.slice(0, 5);
    const last5_slow = last6_slow.slice(0, 5);

    assertArrayClose(
        last5_fast,
        expected_fast,
        1e-3,
        "Buff Averages fast buffer last 5 values mismatch"
    );

    assertArrayClose(
        last5_slow,
        expected_slow,
        1e-3,
        "Buff Averages slow buffer last 5 values mismatch"
    );
});

test('Buff Averages partial params', () => {

    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);


    const result = wasm.buff_averages_js(close, volume, 5, 20);

    assert.strictEqual(result.length, close.length * 2);
});

test('Buff Averages NaN handling', () => {

    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);

    const result = wasm.buff_averages_js(close, volume, 5, 20);

    const halfLen = close.length;
    const fast_buff = result.slice(0, halfLen);
    const slow_buff = result.slice(halfLen);


    let firstValid = 0;
    for (let i = 0; i < close.length; i++) {
        if (!isNaN(close[i])) {
            firstValid = i;
            break;
        }
    }

    const warmupPeriod = firstValid + 20 - 1;


    assertAllNaN(fast_buff.slice(0, warmupPeriod), "Expected NaN in fast buffer warmup period");
    assertAllNaN(slow_buff.slice(0, warmupPeriod), "Expected NaN in slow buffer warmup period");


    assertNoNaN(fast_buff.slice(warmupPeriod), "Found unexpected NaN in fast buffer after warmup");
    assertNoNaN(slow_buff.slice(warmupPeriod), "Found unexpected NaN in slow buffer after warmup");
});

test('Buff Averages empty input', () => {

    const empty = new Float64Array([]);

    assert.throws(() => {
        wasm.buff_averages_js(empty, empty, 5, 20);
    }, /empty/i, "Should throw error for empty input");
});

test('Buff Averages all NaN', () => {

    const nanData = new Float64Array(100);
    const nanVolume = new Float64Array(100);
    nanData.fill(NaN);
    nanVolume.fill(NaN);

    assert.throws(() => {
        wasm.buff_averages_js(nanData, nanVolume, 5, 20);
    }, /All values are NaN/i, "Should throw error for all NaN values");
});

test('Buff Averages zero period', () => {

    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    const volumeData = new Float64Array([100.0, 200.0, 300.0]);

    assert.throws(() => {
        wasm.buff_averages_js(inputData, volumeData, 0, 20);
    }, /Invalid period/, "Should throw error for zero fast period");
});

test('Buff Averages period exceeds length', () => {

    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    const volumeSmall = new Float64Array([100.0, 200.0, 300.0]);

    assert.throws(() => {
        wasm.buff_averages_js(dataSmall, volumeSmall, 5, 10);
    }, /Invalid period|Not enough valid data/, "Should throw error for period exceeding data length");
});

test('Buff Averages very small dataset', () => {

    const singlePoint = new Float64Array([42.0]);
    const singleVolume = new Float64Array([100.0]);

    assert.throws(() => {
        wasm.buff_averages_js(singlePoint, singleVolume, 5, 20);
    }, /Invalid period|Not enough valid data/, "Should throw error for insufficient data");
});

test('Buff Averages mismatched lengths', () => {

    const priceData = new Float64Array([10.0, 20.0, 30.0]);
    const volumeData = new Float64Array([100.0, 200.0]);

    assert.throws(() => {
        wasm.buff_averages_js(priceData, volumeData, 2, 2);
    }, /mismatched|different length/i, "Should throw error for mismatched data lengths");
});

test('Buff Averages unified API', () => {

    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);

    const result = wasm.buff_averages(close, volume, 5, 20);


    assert(result.values, 'Should have values array');
    assert(typeof result.rows === 'number', 'Should have rows count');
    assert(typeof result.cols === 'number', 'Should have cols count');


    assert.strictEqual(result.rows, 2);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.values.length, 2 * close.length);


    const fast_buff = result.values.slice(0, result.cols);
    const slow_buff = result.values.slice(result.cols);


    const expected_fast = [
        58740.30855637,
        59132.28418702,
        59309.76658172,
        59266.10492431,
        59194.11908892,
    ];

    const expected_slow = [
        59209.26229392,
        59201.87047432,
        59217.15739355,
        59195.74527194,
        59196.26139533,
    ];

    assertArrayClose(
        fast_buff.slice(-6, -1),
        expected_fast,
        1e-3,
        "Unified API fast buffer mismatch"
    );

    assertArrayClose(
        slow_buff.slice(-6, -1),
        expected_slow,
        1e-3,
        "Unified API slow buffer mismatch"
    );
});

test('Buff Averages batch single', () => {

    const close = new Float64Array(testData.close.slice(0, 100));
    const volume = new Float64Array(testData.volume.slice(0, 100));

    const result = wasm.buff_averages_batch(
        close,
        volume,
        [5, 5, 1],
        [20, 20, 1]
    );


    assert(result.values, 'Should have values array');
    assert(result.rows, 'Should have rows count');
    assert(result.cols, 'Should have cols count');
    assert(result.fast_periods, 'Should have fast_periods array');
    assert(result.slow_periods, 'Should have slow_periods array');


    assert.strictEqual(result.fast_periods.length, 1);
    assert.strictEqual(result.slow_periods.length, 1);
    assert.strictEqual(result.rows, 2);
    assert.strictEqual(result.cols, 100);


    assert.strictEqual(result.fast_periods[0], 5);
    assert.strictEqual(result.slow_periods[0], 20);


    const regularResult = wasm.buff_averages_js(close, volume, 5, 20);
    const fastRegular = regularResult.slice(0, 100);
    const slowRegular = regularResult.slice(100);

    const fastBatch = result.values.slice(0, 100);
    const slowBatch = result.values.slice(100);

    assertArrayClose(fastBatch, fastRegular, 1e-10, "Batch vs regular fast mismatch");
    assertArrayClose(slowBatch, slowRegular, 1e-10, "Batch vs regular slow mismatch");
});

test('Buff Averages batch multiple', () => {

    const close = new Float64Array(testData.close.slice(0, 50));
    const volume = new Float64Array(testData.volume.slice(0, 50));

    const result = wasm.buff_averages_batch(
        close,
        volume,
        [3, 5, 2],
        [15, 20, 5]
    );


    assert.strictEqual(result.fast_periods.length, 4);
    assert.strictEqual(result.slow_periods.length, 4);
    assert.strictEqual(result.rows, 8);
    assert.strictEqual(result.cols, 50);


    const expectedCombos = [
        {fast: 3, slow: 15},
        {fast: 3, slow: 20},
        {fast: 5, slow: 15},
        {fast: 5, slow: 20}
    ];

    for (let i = 0; i < expectedCombos.length; i++) {
        assert.strictEqual(result.fast_periods[i], expectedCombos[i].fast);
        assert.strictEqual(result.slow_periods[i], expectedCombos[i].slow);


        const fastRow = result.values.slice(i * 50, (i + 1) * 50);
        const slowRow = result.values.slice((4 + i) * 50, (4 + i + 1) * 50);


        const regularResult = wasm.buff_averages_js(
            close,
            volume,
            expectedCombos[i].fast,
            expectedCombos[i].slow
        );
        const fastRegular = regularResult.slice(0, 50);
        const slowRegular = regularResult.slice(50);

        assertArrayClose(
            fastRow,
            fastRegular,
            1e-10,
            `Fast mismatch for combo (${expectedCombos[i].fast}, ${expectedCombos[i].slow})`
        );
        assertArrayClose(
            slowRow,
            slowRegular,
            1e-10,
            `Slow mismatch for combo (${expectedCombos[i].fast}, ${expectedCombos[i].slow})`
        );
    }
});

test('Buff Averages batch edge cases', () => {

    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]);
    const volume = new Float64Array(20);
    volume.fill(1.0);


    const result1 = wasm.buff_averages_batch(
        data,
        volume,
        [5, 7, 10],
        [10, 10, 1]
    );

    assert.strictEqual(result1.fast_periods.length, 1);
    assert.strictEqual(result1.fast_periods[0], 5);
    assert.strictEqual(result1.slow_periods[0], 10);


    assert.throws(() => {
        wasm.buff_averages_batch(
            new Float64Array([]),
            new Float64Array([]),
            [5, 5, 1],
            [20, 20, 1]
        );
    }, /empty/i);


    assert.throws(() => {
        wasm.buff_averages_batch(
            data,
            volume,
            [5, 5],
            [20, 20, 1]
        );
    }, /must each have 3 elements/);
});

test('Buff Averages batch warmup periods', () => {

    const close = new Float64Array(testData.close.slice(0, 50));
    const volume = new Float64Array(testData.volume.slice(0, 50));

    const result = wasm.buff_averages_batch(
        close,
        volume,
        [3, 5, 2],
        [10, 20, 10]
    );


    const combos = [
        {fast: 3, slow: 10, warmup: 9},
        {fast: 3, slow: 20, warmup: 19},
        {fast: 5, slow: 10, warmup: 9},
        {fast: 5, slow: 20, warmup: 19}
    ];

    for (let i = 0; i < combos.length; i++) {
        const fastRow = result.values.slice(i * 50, (i + 1) * 50);
        const slowRow = result.values.slice((4 + i) * 50, (4 + i + 1) * 50);
        const warmup = combos[i].warmup;


        for (let j = 0; j < warmup; j++) {
            assert(isNaN(fastRow[j]),
                `Expected NaN in fast warmup at index ${j} for combo (${combos[i].fast}, ${combos[i].slow})`);
            assert(isNaN(slowRow[j]),
                `Expected NaN in slow warmup at index ${j} for combo (${combos[i].fast}, ${combos[i].slow})`);
        }


        for (let j = warmup; j < 50; j++) {
            assert(!isNaN(fastRow[j]),
                `Unexpected NaN after fast warmup at index ${j} for combo (${combos[i].fast}, ${combos[i].slow})`);
            assert(!isNaN(slowRow[j]),
                `Unexpected NaN after slow warmup at index ${j} for combo (${combos[i].fast}, ${combos[i].slow})`);
        }
    }
});

test('Buff Averages zero-copy API', () => {

    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]);
    const volume = new Float64Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
    const len = data.length;


    const outPtr = wasm.buff_averages_alloc(len);
    assert(outPtr !== 0, 'Failed to allocate memory');
    wasm.buff_averages_free(outPtr, len);


    const result = wasm.buff_averages_js(data, volume, 5, 10);
    assert.strictEqual(result.length, len * 2);

    const fastResult = result.slice(0, len);
    const slowResult = result.slice(len);


    for (let i = 0; i < 9; i++) {
        assert(isNaN(fastResult[i]), `Expected NaN in fast at index ${i}`);
        assert(isNaN(slowResult[i]), `Expected NaN in slow at index ${i}`);
    }


    
});

test('Buff Averages zero-copy error handling', () => {

    const ptr = wasm.buff_averages_alloc(10);
    assert(ptr !== 0, 'Should allocate memory');
    wasm.buff_averages_free(ptr, 10);


    


    assert.throws(() => {
        wasm.buff_averages_js(
            new Float64Array([]),
            new Float64Array([]),
            5, 20
        );
    }, /empty/i, 'Should fail with empty input');

    assert.throws(() => {
        wasm.buff_averages_js(
            new Float64Array([1, 2, 3]),
            new Float64Array([1, 2]),
            5, 20
        );
    }, /mismatch|different/, 'Should fail with mismatched lengths');

    assert.throws(() => {
        wasm.buff_averages_js(
            new Float64Array([1, 2, 3]),
            new Float64Array([1, 2, 3]),
            0, 20
        );
    }, /period/, 'Should fail with zero period');
});

test('Buff Averages zero-copy large dataset', () => {

    const size = 10000;
    const data = new Float64Array(size);
    const volume = new Float64Array(size);


    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) * 100 + 1000;
        volume[i] = Math.random() * 1000 + 100;
    }


    const outPtr = wasm.buff_averages_alloc(size);
    assert(outPtr !== 0, 'Failed to allocate large buffer');
    wasm.buff_averages_free(outPtr, size);


    const startTime = performance.now();
    const result = wasm.buff_averages_js(data, volume, 50, 200);
    const endTime = performance.now();
    console.log(`Large dataset (${size} points): ${(endTime - startTime).toFixed(2)}ms`);

    assert.strictEqual(result.length, size * 2);

    const fastResult = result.slice(0, size);
    const slowResult = result.slice(size);


    for (let i = 0; i < 199; i++) {
        assert(isNaN(fastResult[i]), `Expected NaN at fast warmup index ${i}`);
        assert(isNaN(slowResult[i]), `Expected NaN at slow warmup index ${i}`);
    }


    for (let i = 199; i < Math.min(300, size); i++) {
        assert(!isNaN(fastResult[i]), `Unexpected NaN at fast index ${i}`);
        assert(!isNaN(slowResult[i]), `Unexpected NaN at slow index ${i}`);
    }

    
});




test('Buff Averages Context API (deprecated)', () => {

    if (!wasm.BuffAveragesContext) {
        console.log('BuffAveragesContext not available, skipping test');
        return;
    }

    const ctx = new wasm.BuffAveragesContext(5, 20);
    assert(ctx, 'Failed to create BuffAveragesContext');


    assert.strictEqual(ctx.get_warmup_period(), 19, 'Warmup period should be slow_period - 1');


    const size = 50;
    const data = new Float64Array(size);
    const volume = new Float64Array(size);

    for (let i = 0; i < size; i++) {
        data[i] = i + 1;
        volume[i] = 1;
    }


    const result = ctx.compute(data, volume);
    assert(result, 'Should compute with context');
    assert.strictEqual(result.length, size * 2);

    const fastResult = result.slice(0, size);
    const slowResult = result.slice(size);


    for (let i = 0; i < 19; i++) {
        assert(isNaN(fastResult[i]), `Expected NaN in fast at index ${i}`);
        assert(isNaN(slowResult[i]), `Expected NaN in slow at index ${i}`);
    }


    for (let i = 19; i < size; i++) {
        assert(!isNaN(fastResult[i]), `Unexpected NaN in fast at index ${i}`);
        assert(!isNaN(slowResult[i]), `Unexpected NaN in slow at index ${i}`);
    }
});

test.after(() => {
    console.log('Buff Averages WASM tests completed');
});
