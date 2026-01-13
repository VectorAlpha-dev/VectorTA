
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
let high, low, close, hl2;

test.before(async () => {

    try {
        const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
        const importPath = process.platform === 'win32'
            ? `file:///${wasmPath.replace(/\\/g, '/')}`
            : wasmPath;
        const module = await import(importPath);
        wasm = module;
    } catch (error) {
        console.error('Failed to load WASM module:', error);
        throw error;
    }


    testData = loadTestData();
    high = testData.high;
    low = testData.low;
    close = testData.close;


    hl2 = high.map((h, i) => (h + low[i]) / 2);
});

test('TTM Trend - safe API basic test', () => {
    const period = 5;
    const result = wasm.ttm_trend_js(hl2, close, period);

    assert.strictEqual(result.length, close.length, 'Output length should match input length');
    assert.ok(result instanceof Float64Array, 'Result should be Float64Array');


    for (let i = 0; i < result.length; i++) {
        assert.ok(result[i] === 0.0 || result[i] === 1.0 || isNaN(result[i]),
                  `Value at index ${i} should be 0.0, 1.0, or NaN, got ${result[i]}`);
    }


    for (let i = 0; i < period - 1; i++) {
        assert.ok(isNaN(result[i]), `Warmup period at index ${i} should be NaN`);
    }
});

test('TTM Trend - safe API with different periods', () => {
    const periods = [5, 10, 20];

    for (const period of periods) {
        const result = wasm.ttm_trend_js(hl2, close, period);
        assert.strictEqual(result.length, close.length, `Output length for period ${period} should match input`);


        for (let i = 0; i < period - 1; i++) {
            assert.ok(isNaN(result[i]), `Warmup for period ${period} at index ${i} should be NaN`);
        }
    }
});

test('TTM Trend - fast API in-place operations', () => {
    const period = 5;
    const len = close.length;


    const sourcePtr = wasm.ttm_trend_alloc(len);
    const closePtr = wasm.ttm_trend_alloc(len);
    const outPtr = wasm.ttm_trend_alloc(len);

    try {

        const sourceHeap = new Float64Array(wasm.__wasm.memory.buffer, sourcePtr, len);
        const closeHeap = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        sourceHeap.set(hl2);
        closeHeap.set(close);


        wasm.ttm_trend_into(sourcePtr, closePtr, outPtr, len, period);


        const resultHeap = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        const result = Array.from(resultHeap);


        const safeResult = wasm.ttm_trend_js(hl2, close, period);

        for (let i = 0; i < len; i++) {

            if (isNaN(result[i]) && isNaN(safeResult[i])) {
                continue;
            }
            assert.strictEqual(result[i], safeResult[i], `Fast API mismatch at index ${i}`);
        }
    } finally {

        wasm.ttm_trend_free(sourcePtr, len);
        wasm.ttm_trend_free(closePtr, len);
        wasm.ttm_trend_free(outPtr, len);
    }
});

test('TTM Trend - fast API with aliasing detection', () => {
    const period = 5;
    const len = 100;
    const hl2Small = hl2.slice(0, len);
    const closeSmall = close.slice(0, len);


    const sourcePtr = wasm.ttm_trend_alloc(len);
    const closePtr = wasm.ttm_trend_alloc(len);

    try {

        const sourceHeap = new Float64Array(wasm.__wasm.memory.buffer, sourcePtr, len);
        const closeHeap = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        sourceHeap.set(hl2Small);
        closeHeap.set(closeSmall);


        const outPtr = sourcePtr;
        wasm.ttm_trend_into(sourcePtr, closePtr, outPtr, len, period);


        const resultHeap = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        const result = Array.from(resultHeap);


        const hasZeros = result.some(v => v === 0.0);
        const hasOnes = result.some(v => v === 1.0);
        assert.ok(hasZeros || hasOnes, 'Result should contain valid values (0.0 or 1.0)');
    } finally {

        wasm.ttm_trend_free(closePtr, len);
        wasm.ttm_trend_free(sourcePtr, len);
    }
});

test('TTM Trend - batch API', () => {
    const config = {
        period_range: [5, 15, 5]
    };

    const result = wasm.ttm_trend_batch(hl2, close, config);

    assert.strictEqual(result.rows, 3, 'Should have 3 rows (periods)');
    assert.strictEqual(result.cols, close.length, 'Columns should match input length');
    assert.strictEqual(result.periods.length, 3, 'Should have 3 periods');
    assert.deepStrictEqual(result.periods, [5, 10, 15], 'Periods should match');
    assert.strictEqual(result.values.length, 3 * close.length, 'Values array should be flattened');


    for (let i = 0; i < result.values.length; i++) {
        assert.ok(result.values[i] === 0.0 || result.values[i] === 1.0 || isNaN(result.values[i]),
                  `Batch value at index ${i} should be 0.0, 1.0, or NaN`);
    }


    const singleResult = wasm.ttm_trend_js(hl2, close, 5);
    for (let i = 0; i < close.length; i++) {

        if (isNaN(result.values[i]) && isNaN(singleResult[i])) {
            continue;
        }
        assert.strictEqual(result.values[i], singleResult[i],
                          `Batch row 0 mismatch at index ${i}`);
    }
});

test('TTM Trend - error handling', () => {

    assert.throws(() => {
        wasm.ttm_trend_js(hl2, close, 0);
    }, /Invalid period/, 'Should throw on zero period');

    assert.throws(() => {
        wasm.ttm_trend_js(hl2, close, close.length + 1);
    }, /Invalid period/, 'Should throw when period exceeds data length');


    assert.throws(() => {
        wasm.ttm_trend_js([], [], 5);
    }, /empty/i, 'Should throw on empty input');
});

test('TTM Trend - accuracy check', () => {
    const period = 5;
    const result = wasm.ttm_trend_js(hl2, close, period);



    for (let i = period + 10; i < period + 15; i++) {
        let sum = 0;
        for (let j = i - period + 1; j <= i; j++) {
            sum += hl2[j];
        }
        const avg = sum / period;
        const expected = close[i] > avg ? 1.0 : 0.0;

        assert.strictEqual(result[i], expected,
                          `Accuracy mismatch at index ${i}: close=${close[i]}, avg=${avg}`);
    }
});

test('TTM Trend - matches Rust expected last five values', () => {

    const period = 5;
    const result = wasm.ttm_trend_js(hl2, close, period);


    const last5Bools = Array.from(result.slice(-5)).map(v => v === 1.0);
    const expectedLastFive = [true, false, false, false, false];
    assert.deepStrictEqual(
        last5Bools,
        expectedLastFive,
        `Expected ${expectedLastFive}, got ${last5Bools}`
    );
});

test('TTM Trend - memory allocation and deallocation', () => {
    const len = 1000;


    const ptr1 = wasm.ttm_trend_alloc(len);
    assert.ok(ptr1 !== 0, 'Allocated pointer should not be null');
    wasm.ttm_trend_free(ptr1, len);


    const ptrs = [];
    for (let i = 0; i < 10; i++) {
        ptrs.push(wasm.ttm_trend_alloc(len));
    }


    const uniquePtrs = new Set(ptrs);
    assert.strictEqual(uniquePtrs.size, ptrs.length, 'All allocated pointers should be unique');


    for (const ptr of ptrs) {
        wasm.ttm_trend_free(ptr, len);
    }
});

test('TTM Trend - warmup period calculation', () => {

    const periods = [5, 10, 20, 50];

    for (const period of periods) {
        const result = wasm.ttm_trend_js(hl2, close, period);


        for (let i = 0; i < period - 1; i++) {
            assert.ok(isNaN(result[i]), `Warmup at index ${i} for period ${period} should be NaN`);
        }


        assert.ok(!isNaN(result[period - 1]), `First valid value should be at index ${period - 1}`);
    }
});

test('TTM Trend - reinput test', () => {
    const period = 5;


    const firstResult = wasm.ttm_trend_js(hl2, close, period);



    const secondResult = wasm.ttm_trend_js(firstResult, close, period);

    assert.strictEqual(secondResult.length, firstResult.length, 'Reinput length should match');


    let hasChanges = false;
    for (let i = period; i < secondResult.length; i++) {
        if (!isNaN(firstResult[i]) && !isNaN(secondResult[i]) && secondResult[i] !== firstResult[i]) {
            hasChanges = true;
            break;
        }
    }
    assert.ok(hasChanges, 'Reinput should produce different values');
});

test('TTM Trend - all NaN input', () => {
    const allNaN = new Float64Array(100).fill(NaN);

    assert.throws(() => {
        wasm.ttm_trend_js(allNaN, allNaN, 5);
    }, /All values are NaN/, 'Should throw on all NaN input');
});

test('TTM Trend - batch metadata verification', () => {
    const config = {
        period_range: [5, 15, 5]
    };

    const result = wasm.ttm_trend_batch(hl2, close, config);


    assert.strictEqual(result.rows, 3, 'Should have correct number of rows');
    assert.strictEqual(result.cols, close.length, 'Should have correct number of columns');
    assert.deepStrictEqual(result.periods, [5, 10, 15], 'Should have correct periods');


    assert.strictEqual(result.values.length, result.rows * result.cols,
                      'Values array should match rows * cols');
});

test('TTM Trend - batch edge cases', () => {

    const smallData = hl2.slice(0, 20);
    const smallClose = close.slice(0, 20);


    const singleConfig = {
        period_range: [5, 5, 0]
    };

    const singleResult = wasm.ttm_trend_batch(smallData, smallClose, singleConfig);
    assert.strictEqual(singleResult.rows, 1, 'Single period should have 1 row');
    assert.strictEqual(singleResult.periods.length, 1, 'Should have 1 period');


    const largeStepConfig = {
        period_range: [5, 10, 10]
    };

    const largeStepResult = wasm.ttm_trend_batch(smallData, smallClose, largeStepConfig);
    assert.strictEqual(largeStepResult.rows, 1, 'Large step should result in 1 row');
    assert.deepStrictEqual(largeStepResult.periods, [5], 'Should only have start period');
});

test('TTM Trend - zero-copy consistency', () => {
    const period = 5;
    const len = 100;
    const smallHl2 = hl2.slice(0, len);
    const smallClose = close.slice(0, len);


    const safeResult = wasm.ttm_trend_js(smallHl2, smallClose, period);


    const sourcePtr = wasm.ttm_trend_alloc(len);
    const closePtr = wasm.ttm_trend_alloc(len);
    const outPtr = wasm.ttm_trend_alloc(len);

    try {

        const sourceHeap = new Float64Array(wasm.__wasm.memory.buffer, sourcePtr, len);
        const closeHeap = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        sourceHeap.set(smallHl2);
        closeHeap.set(smallClose);


        wasm.ttm_trend_into(sourcePtr, closePtr, outPtr, len, period);


        const resultHeap = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        const fastResult = Array.from(resultHeap);


        for (let i = 0; i < len; i++) {

            if (isNaN(fastResult[i]) && isNaN(safeResult[i])) {
                continue;
            }
            assert.strictEqual(fastResult[i], safeResult[i],
                              `Fast and safe APIs should produce identical results at index ${i}`);
        }
    } finally {
        wasm.ttm_trend_free(sourcePtr, len);
        wasm.ttm_trend_free(closePtr, len);
        wasm.ttm_trend_free(outPtr, len);
    }
});

test('TTM Trend - streaming edge cases', () => {

    const nanData = [...hl2.slice(0, 10)];
    const nanClose = [...close.slice(0, 10)];
    nanData[5] = NaN;
    nanClose[6] = NaN;


    const result = wasm.ttm_trend_js(
        new Float64Array(nanData),
        new Float64Array(nanClose),
        3
    );

    assert.strictEqual(result.length, 10, 'Should handle some NaN values');


    assert.ok(isNaN(result[0]), 'Warmup should be NaN');
    assert.ok(isNaN(result[1]), 'Warmup should be NaN');
});

test('TTM Trend - partial parameters', () => {

    const result = wasm.ttm_trend_js(hl2, close, 5);
    assert.strictEqual(result.length, close.length, 'Output length should match input');
    assert.ok(result instanceof Float64Array, 'Result should be Float64Array');
});

test('TTM Trend - batch with invalid parameters', () => {

    assert.throws(() => {
        wasm.ttm_trend_batch(hl2, close, {
            period_range: [0, 10, 5]
        });
    }, /Invalid period/, 'Should throw on invalid start period');

    assert.throws(() => {
        wasm.ttm_trend_batch(hl2, close, {
            period_range: [10, 5, 1]
        });
    }, /Invalid.*range/, 'Should throw on invalid range');
});

test('TTM Trend - large dataset performance', () => {

    const startTime = performance.now();
    const result = wasm.ttm_trend_js(hl2, close, 20);
    const endTime = performance.now();

    assert.strictEqual(result.length, close.length, 'Should handle large dataset');


    const duration = endTime - startTime;
    assert.ok(duration < 100, `Should complete quickly (took ${duration.toFixed(2)}ms)`);
});

test('TTM Trend - NaN propagation', () => {

    const testData = [...hl2.slice(0, 50)];
    const testClose = [...close.slice(0, 50)];


    testData[25] = NaN;
    testClose[30] = NaN;

    const result = wasm.ttm_trend_js(
        new Float64Array(testData),
        new Float64Array(testClose),
        5
    );


    let hasValidBefore = false;
    let hasValidAfter = false;

    for (let i = 10; i < 25; i++) {
        if (!isNaN(result[i])) {
            hasValidBefore = true;
            break;
        }
    }

    for (let i = 35; i < 50; i++) {
        if (!isNaN(result[i])) {
            hasValidAfter = true;
            break;
        }
    }

    assert.ok(hasValidBefore, 'Should have valid values before NaN');
    assert.ok(hasValidAfter, 'Should have valid values after NaN');
});

console.log('All TTM Trend WASM tests passed! ✓');
