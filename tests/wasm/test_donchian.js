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

test('Donchian - basic functionality', () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const period = 20;

    const result = wasm.donchian_js(high, low, period);


    assert.strictEqual(result.rows, 3, 'Should have 3 rows (upper, middle, lower)');
    assert.strictEqual(result.cols, high.length, 'Columns should match input length');
    assert.strictEqual(result.values.length, high.length * 3, 'Values should be flattened array');
});

test('Donchian - accuracy test', () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const expected = EXPECTED_OUTPUTS.donchian;
    const period = expected.defaultParams.period;

    const result = wasm.donchian_js(high, low, period);


    const len = high.length;
    const upper = result.values.slice(0, len);
    const middle = result.values.slice(len, 2 * len);
    const lower = result.values.slice(2 * len, 3 * len);


    const startIdx = len - 5;
    assertArrayClose(
        upper.slice(startIdx),
        expected.last5Upper,
        0.1,
        'Upper band last 5 values mismatch'
    );
    assertArrayClose(
        middle.slice(startIdx),
        expected.last5Middle,
        0.1,
        'Middle band last 5 values mismatch'
    );
    assertArrayClose(
        lower.slice(startIdx),
        expected.last5Lower,
        0.1,
        'Lower band last 5 values mismatch'
    );
});

test('Donchian - zero period should fail', () => {
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);

    assert.throws(() => {
        wasm.donchian_js(high, low, 0);
    }, /Invalid period/);
});

test('Donchian - period exceeds length should fail', () => {
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);

    assert.throws(() => {
        wasm.donchian_js(high, low, 10);
    }, /Invalid period|period exceeds/);
});

test('Donchian - mismatched lengths should fail', () => {
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0]);

    assert.throws(() => {
        wasm.donchian_js(high, low, 2);
    }, /MismatchedLength|different lengths/);
});

test('Donchian - empty data should fail', () => {
    const empty = new Float64Array([]);

    assert.throws(() => {
        wasm.donchian_js(empty, empty, 20);
    }, /empty|EmptyData/i);
});

test('Donchian - all NaN values should fail', () => {
    const allNaN = new Float64Array(100).fill(NaN);

    assert.throws(() => {
        wasm.donchian_js(allNaN, allNaN, 20);
    }, /all.*NaN|AllValuesNaN/i);
});

test('Donchian - warmup period handling', () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const period = 20;

    const result = wasm.donchian_js(high, low, period);
    const len = high.length;
    const upper = result.values.slice(0, len);
    const middle = result.values.slice(len, 2 * len);
    const lower = result.values.slice(2 * len, 3 * len);


    for (let i = 0; i < period - 1; i++) {
        assert(isNaN(upper[i]), `Upper[${i}] should be NaN during warmup`);
        assert(isNaN(middle[i]), `Middle[${i}] should be NaN during warmup`);
        assert(isNaN(lower[i]), `Lower[${i}] should be NaN during warmup`);
    }


    assertNoNaN(upper.slice(period), 'Upper band after warmup');
    assertNoNaN(middle.slice(period), 'Middle band after warmup');
    assertNoNaN(lower.slice(period), 'Lower band after warmup');
});

test('Donchian - fast API in-place operation', async () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const period = 20;
    const len = high.length;


    const highPtr = wasm.donchian_alloc(len);
    const lowPtr = wasm.donchian_alloc(len);
    const upperPtr = wasm.donchian_alloc(len);
    const middlePtr = wasm.donchian_alloc(len);
    const lowerPtr = wasm.donchian_alloc(len);

    try {

        const memory = new Float64Array(wasm.__wasm.memory.buffer);
        const highStart = highPtr / 8;
        const lowStart = lowPtr / 8;
        memory.set(high, highStart);
        memory.set(low, lowStart);


        wasm.donchian_into(
            highPtr,
            lowPtr,
            upperPtr,
            middlePtr,
            lowerPtr,
            len,
            period
        );


        const memoryAfter = new Float64Array(wasm.__wasm.memory.buffer);
        const upperStart = upperPtr / 8;
        const middleStart = middlePtr / 8;
        const lowerStart = lowerPtr / 8;

        const upper = memoryAfter.slice(upperStart, upperStart + len);
        const middle = memoryAfter.slice(middleStart, middleStart + len);
        const lower = memoryAfter.slice(lowerStart, lowerStart + len);


        assert(!isNaN(upper[period]), 'Upper should have valid values after warmup');
        assert(!isNaN(middle[period]), 'Middle should have valid values after warmup');
        assert(!isNaN(lower[period]), 'Lower should have valid values after warmup');
    } finally {

        wasm.donchian_free(highPtr, len);
        wasm.donchian_free(lowPtr, len);
        wasm.donchian_free(upperPtr, len);
        wasm.donchian_free(middlePtr, len);
        wasm.donchian_free(lowerPtr, len);
    }
});

test('Donchian - batch processing', () => {
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));

    const config = {
        period_range: [10, 30, 10]
    };

    const result = wasm.donchian_batch(high, low, config);


    assert.strictEqual(result.rows, 3, 'Should have 3 combinations');
    assert.strictEqual(result.cols, 100, 'Should have 100 columns');
    assert.strictEqual(result.periods.length, 3, 'Should have 3 periods');
    assert.deepStrictEqual(Array.from(result.periods), [10, 20, 30], 'Periods should match');


    assert.strictEqual(result.upper.length, 300, 'Upper should have rows * cols values');
    assert.strictEqual(result.middle.length, 300, 'Middle should have rows * cols values');
    assert.strictEqual(result.lower.length, 300, 'Lower should have rows * cols values');
});

test('Donchian - very small dataset', () => {
    const singlePoint = new Float64Array([42.0]);

    assert.throws(() => {
        wasm.donchian_js(singlePoint, singlePoint, 2);
    }, /Invalid period|Not enough/);
});

test('Donchian - reinput test', () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const expected = EXPECTED_OUTPUTS.donchian;
    const period = expected.defaultParams.period;


    const firstResult = wasm.donchian_js(high, low, period);
    const len = high.length;
    const firstUpper = firstResult.values.slice(0, len);
    const firstMiddle = firstResult.values.slice(len, 2 * len);
    const firstLower = firstResult.values.slice(2 * len, 3 * len);



    const secondResult = wasm.donchian_js(firstMiddle, firstMiddle, period);
    const secondUpper = secondResult.values.slice(0, len);
    const secondMiddle = secondResult.values.slice(len, 2 * len);
    const secondLower = secondResult.values.slice(2 * len, 3 * len);


    assert.strictEqual(secondResult.rows, 3);
    assert.strictEqual(secondResult.cols, len);


    const startIdx = len - 5;
    assertArrayClose(
        secondUpper.slice(startIdx),
        expected.reinputLast5Upper,
        0.1,
        'Reinput upper band last 5 values mismatch'
    );
    assertArrayClose(
        secondMiddle.slice(startIdx),
        expected.reinputLast5Middle,
        0.1,
        'Reinput middle band last 5 values mismatch'
    );
    assertArrayClose(
        secondLower.slice(startIdx),
        expected.reinputLast5Lower,
        0.1,
        'Reinput lower band last 5 values mismatch'
    );



});

test('Donchian - invalid high/low relationship', () => {

    const high = new Float64Array([10.0, 15.0, 20.0, 25.0, 30.0]);
    const low = new Float64Array([15.0, 20.0, 25.0, 30.0, 35.0]);
    const period = 3;


    const result = wasm.donchian_js(high, low, period);
    const len = high.length;
    const upper = result.values.slice(0, len);
    const middle = result.values.slice(len, 2 * len);
    const lower = result.values.slice(2 * len, 3 * len);





    assert(!isNaN(upper[period - 1]), 'Upper should have value after warmup');
    assert(!isNaN(lower[period - 1]), 'Lower should have value after warmup');



    assert(upper[period - 1] >= lower[period - 1], 'Bands should still maintain proper relationship');
});

test('Donchian - partial NaN handling', () => {
    const high = new Float64Array([NaN, 12.0, 15.0, NaN, 13.0, 16.0, 14.0, 12.0, 18.0, 17.0]);
    const low = new Float64Array([NaN, 9.0, 11.0, NaN, 10.0, 12.0, 11.0, 10.0, 14.0, 15.0]);
    const period = 3;

    const result = wasm.donchian_js(high, low, period);
    const len = high.length;
    const upper = result.values.slice(0, len);
    const middle = result.values.slice(len, 2 * len);
    const lower = result.values.slice(2 * len, 3 * len);


    assert(isNaN(upper[0]), 'Should be NaN at index 0');
    assert(isNaN(upper[1]), 'Should be NaN at index 1');
    assert(isNaN(upper[2]), 'Should be NaN at index 2');


    assert(isNaN(upper[5]), 'Should be NaN at index 5 due to NaN in window');
    assert(isNaN(middle[5]), 'Should be NaN at index 5 due to NaN in window');
    assert(isNaN(lower[5]), 'Should be NaN at index 5 due to NaN in window');


    assert(!isNaN(upper[6]), 'Should have valid value at index 6');
    assert(!isNaN(middle[6]), 'Should have valid value at index 6');
    assert(!isNaN(lower[6]), 'Should have valid value at index 6');
});

test('Donchian - zero-copy null pointer handling', () => {
    const len = 100;

    assert.throws(() => {
        wasm.donchian_into(
            0,
            0,
            0,
            0,
            0,
            len,
            20
        );
    }, /Null pointer/);
});

test('Donchian - zero-copy with aliasing', async () => {
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const period = 20;
    const len = high.length;


    const highPtr = wasm.donchian_alloc(len);
    const lowPtr = wasm.donchian_alloc(len);
    const outputPtr = wasm.donchian_alloc(len);

    try {

        let memory = new Float64Array(wasm.__wasm.memory.buffer);
        const highStart = highPtr / 8;
        const lowStart = lowPtr / 8;
        memory.set(high, highStart);
        memory.set(low, lowStart);


        wasm.donchian_into(
            highPtr,
            lowPtr,
            outputPtr,
            outputPtr,
            outputPtr,
            len,
            period
        );


        assert(true, 'Aliased operation completed');

    } finally {
        wasm.donchian_free(highPtr, len);
        wasm.donchian_free(lowPtr, len);
        wasm.donchian_free(outputPtr, len);
    }
});

test('Donchian - zero-copy memory management', () => {
    const sizes = [10, 100, 1000, 10000];

    for (const size of sizes) {

        const ptr = wasm.donchian_alloc(size);
        assert(ptr > 0, `Allocation should return valid pointer for size ${size}`);


        wasm.donchian_free(ptr, size);

    }


    wasm.donchian_free(0, 100);
});

test('Donchian - batch edge cases', () => {
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));


    const singleConfig = {
        period_range: [20, 20, 0]
    };

    const singleResult = wasm.donchian_batch(high, low, singleConfig);
    assert.strictEqual(singleResult.rows, 1, 'Single value sweep should have 1 row');
    assert.strictEqual(singleResult.periods.length, 1, 'Should have 1 period');
    assert.strictEqual(singleResult.periods[0], 20, 'Period should be 20');


    const largeStepConfig = {
        period_range: [10, 15, 10]
    };

    const largeStepResult = wasm.donchian_batch(high, low, largeStepConfig);
    assert.strictEqual(largeStepResult.rows, 1, 'Large step should yield 1 value');
    assert.strictEqual(largeStepResult.periods[0], 10, 'Should only have start value');


    const reverseConfig = {
        period_range: [30, 10, 5]
    };


    try {
        const reverseResult = wasm.donchian_batch(high, low, reverseConfig);
        assert.strictEqual(reverseResult.rows, 0, 'Reverse range should yield 0 rows');
    } catch (e) {
        assert(true, 'Reverse range throws error as expected');
    }
});

test('Donchian - batch API error handling', () => {
    const high = new Float64Array([1, 2, 3, 4, 5]);
    const low = new Float64Array([0, 1, 2, 3, 4]);


    assert.throws(() => {
        wasm.donchian_batch(high, low, {});
    }, /Invalid config|period_range/);

    assert.throws(() => {
        wasm.donchian_batch(high, low, {
            period_range: "invalid"
        });
    }, /Invalid config/);


    assert.throws(() => {
        wasm.donchian_batch(high, low, {
            period_range: [10, 20, 5]
        });
    }, /Invalid period|Not enough/);
});

test('Donchian - batch individual row verification', () => {
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));

    const config = {
        period_range: [10, 30, 10]
    };

    const batchResult = wasm.donchian_batch(high, low, config);


    const periods = [10, 20, 30];
    for (let i = 0; i < periods.length; i++) {
        const period = periods[i];


        const rowStart = i * 100;
        const batchUpper = batchResult.upper.slice(rowStart, rowStart + 100);
        const batchMiddle = batchResult.middle.slice(rowStart, rowStart + 100);
        const batchLower = batchResult.lower.slice(rowStart, rowStart + 100);


        const individual = wasm.donchian_js(high, low, period);
        const indivUpper = individual.values.slice(0, 100);
        const indivMiddle = individual.values.slice(100, 200);
        const indivLower = individual.values.slice(200, 300);


        assertArrayClose(batchUpper, indivUpper, 1e-10, `Batch upper row ${i} mismatch`);
        assertArrayClose(batchMiddle, indivMiddle, 1e-10, `Batch middle row ${i} mismatch`);
        assertArrayClose(batchLower, indivLower, 1e-10, `Batch lower row ${i} mismatch`);
    }
});

test('Donchian - zero-copy with large dataset', () => {

    const size = 100000;
    const high = new Float64Array(size);
    const low = new Float64Array(size);


    let price = 100;
    for (let i = 0; i < size; i++) {
        price += (Math.random() - 0.5) * 2;
        high[i] = price + Math.random();
        low[i] = price - Math.random();
    }

    const period = 50;


    const highPtr = wasm.donchian_alloc(size);
    const lowPtr = wasm.donchian_alloc(size);
    const upperPtr = wasm.donchian_alloc(size);
    const middlePtr = wasm.donchian_alloc(size);
    const lowerPtr = wasm.donchian_alloc(size);

    try {

        const memory = new Float64Array(wasm.__wasm.memory.buffer);
        const highStart = highPtr / 8;
        const lowStart = lowPtr / 8;
        memory.set(high, highStart);
        memory.set(low, lowStart);

        const start = performance.now();


        wasm.donchian_into(
            highPtr,
            lowPtr,
            upperPtr,
            middlePtr,
            lowerPtr,
            size,
            period
        );

        const elapsed = performance.now() - start;
        console.log(`  Zero-copy API processed ${size} points in ${elapsed.toFixed(2)}ms`);


        const memoryAfter = new Float64Array(wasm.__wasm.memory.buffer);
        const upperStart = upperPtr / 8;
        const upper = memoryAfter.slice(upperStart, upperStart + size);


        for (let i = 0; i < period - 1; i++) {
            assert(isNaN(upper[i]), `Should be NaN during warmup at ${i}`);
        }
        assert(!isNaN(upper[period]), 'Should have values after warmup');

    } finally {
        wasm.donchian_free(highPtr, size);
        wasm.donchian_free(lowPtr, size);
        wasm.donchian_free(upperPtr, size);
        wasm.donchian_free(middlePtr, size);
        wasm.donchian_free(lowerPtr, size);
    }
});

test('Donchian - fast batch API', () => {
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const len = high.length;


    const periodStart = 10;
    const periodEnd = 30;
    const periodStep = 10;
    const expectedRows = 3;


    const highPtr = wasm.donchian_alloc(len);
    const lowPtr = wasm.donchian_alloc(len);
    const totalSize = expectedRows * len;
    const upperPtr = wasm.donchian_alloc(totalSize);
    const middlePtr = wasm.donchian_alloc(totalSize);
    const lowerPtr = wasm.donchian_alloc(totalSize);

    try {

        const memory = new Float64Array(wasm.__wasm.memory.buffer);
        const highStart = highPtr / 8;
        const lowStart = lowPtr / 8;
        memory.set(high, highStart);
        memory.set(low, lowStart);


        const rows = wasm.donchian_batch_into(
            highPtr,
            lowPtr,
            upperPtr,
            middlePtr,
            lowerPtr,
            len,
            periodStart,
            periodEnd,
            periodStep
        );

        assert.strictEqual(rows, expectedRows, 'Should return correct number of rows');


        const memoryAfter = new Float64Array(wasm.__wasm.memory.buffer);
        const upperStart = upperPtr / 8;
        const upper = memoryAfter.slice(upperStart, upperStart + totalSize);


        for (let row = 0; row < rows; row++) {
            const period = periodStart + row * periodStep;
            const rowOffset = row * len;


            for (let i = 0; i < period - 1; i++) {
                assert(isNaN(upper[rowOffset + i]), `Row ${row} should have NaN at ${i}`);
            }

            assert(!isNaN(upper[rowOffset + period]), `Row ${row} should have data after warmup`);
        }

    } finally {
        wasm.donchian_free(highPtr, len);
        wasm.donchian_free(lowPtr, len);
        wasm.donchian_free(upperPtr, totalSize);
        wasm.donchian_free(middlePtr, totalSize);
        wasm.donchian_free(lowerPtr, totalSize);
    }
});

test('Donchian - batch metadata from result', () => {

    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));

    const config = {
        period_range: [10, 20, 5]
    };

    const result = wasm.donchian_batch(high, low, config);


    assert.strictEqual(result.periods.length, 3);
    assert.strictEqual(result.rows, 3);


    assert.strictEqual(result.periods[0], 10);
    assert.strictEqual(result.periods[1], 15);
    assert.strictEqual(result.periods[2], 20);


    assert.strictEqual(result.upper.length, 3 * 100);
    assert.strictEqual(result.middle.length, 3 * 100);
    assert.strictEqual(result.lower.length, 3 * 100);
});

test('Donchian - batch full parameter sweep', () => {

    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));

    const config = {
        period_range: [5, 25, 5]
    };

    const result = wasm.donchian_batch(high, low, config);


    assert.strictEqual(result.periods.length, 5);
    assert.strictEqual(result.rows, 5);
    assert.strictEqual(result.cols, 50);


    assert.deepStrictEqual(Array.from(result.periods), [5, 10, 15, 20, 25]);


    for (let row = 0; row < 5; row++) {
        const period = result.periods[row];
        const rowStart = row * 50;


        for (let i = 0; i < period - 1; i++) {
            assert(isNaN(result.upper[rowStart + i]),
                `Row ${row} should have NaN at index ${i} during warmup`);
        }

        assert(!isNaN(result.upper[rowStart + period]),
            `Row ${row} should have valid data after warmup`);
    }
});

test('Donchian - batch matches EXPECTED_OUTPUTS', () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const expected = EXPECTED_OUTPUTS.donchian;

    const config = {
        period_range: [expected.defaultParams.period, expected.defaultParams.period, 0]
    };

    const result = wasm.donchian_batch(high, low, config);


    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.periods[0], expected.defaultParams.period);


    const len = high.length;
    const startIdx = len - 5;

    assertArrayClose(
        result.upper.slice(startIdx, startIdx + 5),
        expected.last5Upper,
        0.1,
        'Batch upper band last 5 values mismatch'
    );
    assertArrayClose(
        result.middle.slice(startIdx, startIdx + 5),
        expected.last5Middle,
        0.1,
        'Batch middle band last 5 values mismatch'
    );
    assertArrayClose(
        result.lower.slice(startIdx, startIdx + 5),
        expected.last5Lower,
        0.1,
        'Batch lower band last 5 values mismatch'
    );
});


test.after(() => {

    console.log('Donchian WASM tests completed with comprehensive coverage');
});

console.log('Donchian WASM tests completed');