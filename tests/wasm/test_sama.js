
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

test('SAMA partial params', () => {

    const close = new Float64Array(testData.close);


    const result = wasm.sama_js(close, 200, 14, 6);
    assert.strictEqual(result.length, close.length);


    const result2 = wasm.sama_js(close, 50, 14, 6);
    assert.strictEqual(result2.length, close.length);
});

test('SAMA accuracy', async () => {

    const close = new Float64Array(testData.close.slice(0, 300));
    const expected = EXPECTED_OUTPUTS.sama;


    const result = wasm.sama_js(
        close,
        expected.defaultParams.length,
        expected.defaultParams.majLength,
        expected.defaultParams.minLength
    );

    assert.strictEqual(result.length, close.length);


    const validValues = result.filter(v => !isNaN(v));
    if (validValues.length >= 5) {
        const last5 = validValues.slice(-5);
        assertArrayClose(
            last5,
            expected.last5Values,
            1e-8,
            "SAMA last 5 values mismatch (default params)"
        );
    }


    const result2 = wasm.sama_js(
        close,
        expected.testParams.length,
        expected.testParams.majLength,
        expected.testParams.minLength
    );

    const validValues2 = result2.filter(v => !isNaN(v));
    assert(validValues2.length >= 5, "Should have at least 5 valid values with length=50");
    const last5Test = validValues2.slice(-5);
    assertArrayClose(
        last5Test,
        expected.testLast5,
        1e-8,
        "SAMA last 5 values mismatch (test params)"
    );
});

test('SAMA default candles', () => {

    const close = new Float64Array(testData.close);


    const result = wasm.sama_js(close, 200, 14, 6);
    assert.strictEqual(result.length, close.length);



    assertNoNaN(result.slice(0, 200), "Should have valid values from start");
});

test('SAMA zero period', () => {

    const inputData = new Float64Array([10.0, 20.0, 30.0]);


    assert.throws(() => {
        wasm.sama_js(inputData, 0, 14, 6);
    }, /Invalid period/);


    assert.throws(() => {
        wasm.sama_js(inputData, 10, 0, 6);
    }, /Invalid period/);


    assert.throws(() => {
        wasm.sama_js(inputData, 10, 14, 0);
    }, /Invalid period/);
});

test('SAMA period exceeds length', () => {

    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.sama_js(dataSmall, 10, 14, 6);
    }, /Invalid period|Not enough valid data/);
});

test('SAMA very small dataset', () => {

    const singlePoint = new Float64Array([42.0]);

    assert.throws(() => {
        wasm.sama_js(singlePoint, 200, 14, 6);
    }, /Invalid period|Not enough valid data/);
});

test('SAMA empty input', () => {

    const empty = new Float64Array([]);

    assert.throws(() => {
        wasm.sama_js(empty, 200, 14, 6);
    }, /Input data slice is empty/);
});

test('SAMA all NaN input', () => {

    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);

    assert.throws(() => {
        wasm.sama_js(allNaN, 50, 14, 6);
    }, /All values are NaN/);
});

test('SAMA reinput', () => {

    const close = new Float64Array(testData.close.slice(0, 300));
    const expected = EXPECTED_OUTPUTS.sama;


    const firstResult = wasm.sama_js(
        close,
        expected.testParams.length,
        expected.testParams.majLength,
        expected.testParams.minLength
    );
    assert.strictEqual(firstResult.length, close.length);


    const secondResult = wasm.sama_js(
        firstResult,
        expected.testParams.length,
        expected.testParams.majLength,
        expected.testParams.minLength
    );
    assert.strictEqual(secondResult.length, firstResult.length);


    const validReinput = secondResult.filter(v => !isNaN(v));
    assert(validReinput.length >= 5, "Should have at least 5 valid reinput values");
    const last5Reinput = validReinput.slice(-5);
    assertArrayClose(
        last5Reinput,
        expected.reinputLast5,
        1e-8,
        "SAMA re-input last 5 values mismatch"
    );
});

test('SAMA NaN handling', () => {

    const close = new Float64Array(testData.close);


    const result = wasm.sama_js(close, 50, 14, 6);
    assert.strictEqual(result.length, close.length);


    let firstValid = -1;
    for (let i = 0; i < close.length; i++) {
        if (!isNaN(close[i])) {
            firstValid = i;
            break;
        }
    }

    if (firstValid >= 0) {
        const warmupPeriod = firstValid + 50;


        if (warmupPeriod <= result.length) {

            assertNoNaN(result.slice(0, Math.min(10, result.length)), "Should have valid values from start");


            if (warmupPeriod < result.length - 10) {

                const afterWarmup = result.slice(warmupPeriod, warmupPeriod + 10);
                let hasValidValues = false;
                for (let v of afterWarmup) {
                    if (!isNaN(v)) {
                        hasValidValues = true;
                        break;
                    }
                }
                assert(hasValidValues, "Should have some valid values after warmup");
            }
        }
    }
});

test('SAMA batch single parameter set', () => {

    const close = new Float64Array(testData.close.slice(0, 300));
    const expected = EXPECTED_OUTPUTS.sama;


    const batchResult = wasm.sama_batch(close, {
        length_range: [200, 200, 0],
        maj_length_range: [14, 14, 0],
        min_length_range: [6, 6, 0]
    });


    assert(batchResult.values, 'Should have values array');
    assert(batchResult.combos, 'Should have combos array');
    assert(typeof batchResult.rows === 'number', 'Should have rows count');
    assert(typeof batchResult.cols === 'number', 'Should have cols count');


    assert.strictEqual(batchResult.rows, 1);
    assert.strictEqual(batchResult.cols, close.length);
    assert.strictEqual(batchResult.combos.length, 1);
    assert.strictEqual(batchResult.values.length, close.length);


    const combo = batchResult.combos[0];
    assert.strictEqual(combo.length, 200);
    assert.strictEqual(combo.maj_length, 14);
    assert.strictEqual(combo.min_length, 6);


    const validValues = batchResult.values.filter(v => !isNaN(v));
    if (validValues.length >= 5) {
        const last5 = validValues.slice(-5);
        assertArrayClose(
            last5,
            expected.last5Values,
            1e-8,
            "SAMA batch default row mismatch"
        );
    }
});

test('SAMA batch multiple parameters', () => {

    const close = new Float64Array(testData.close.slice(0, 100));


    const batchResult = wasm.sama_batch(close, {
        length_range: [40, 50, 5],
        maj_length_range: [12, 14, 1],
        min_length_range: [4, 6, 1]
    });


    assert.strictEqual(batchResult.rows, 27);
    assert.strictEqual(batchResult.cols, 100);
    assert.strictEqual(batchResult.combos.length, 27);
    assert.strictEqual(batchResult.values.length, 27 * 100);


    assert.strictEqual(batchResult.combos[0].length, 40);
    assert.strictEqual(batchResult.combos[0].maj_length, 12);
    assert.strictEqual(batchResult.combos[0].min_length, 4);


    assert.strictEqual(batchResult.combos[26].length, 50);
    assert.strictEqual(batchResult.combos[26].maj_length, 14);
    assert.strictEqual(batchResult.combos[26].min_length, 6);


    for (let i = 0; i < Math.min(3, batchResult.combos.length); i++) {
        const combo = batchResult.combos[i];
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.values.slice(rowStart, rowEnd);

        const singleResult = wasm.sama_js(close, combo.length, combo.maj_length, combo.min_length);
        assertArrayClose(
            rowData,
            singleResult,
            1e-10,
            `Row ${i} (length=${combo.length}, maj=${combo.maj_length}, min=${combo.min_length}) mismatch`
        );
    }
});

test('SAMA batch metadata from result', () => {

    const close = new Float64Array(60);
    close.fill(100);

    const result = wasm.sama_batch(close, {
        length_range: [40, 50, 10],
        maj_length_range: [12, 14, 2],
        min_length_range: [5, 6, 1]
    });


    assert.strictEqual(result.combos.length, 8);


    assert.strictEqual(result.combos[0].length, 40);
    assert.strictEqual(result.combos[0].maj_length, 12);
    assert.strictEqual(result.combos[0].min_length, 5);


    assert.strictEqual(result.combos[7].length, 50);
    assert.strictEqual(result.combos[7].maj_length, 14);
    assert.strictEqual(result.combos[7].min_length, 6);
});

test('SAMA batch full parameter sweep', () => {

    const close = new Float64Array(testData.close.slice(0, 60));

    const batchResult = wasm.sama_batch(close, {
        length_range: [45, 50, 5],
        maj_length_range: [13, 14, 1],
        min_length_range: [5, 5, 0]
    });


    assert.strictEqual(batchResult.combos.length, 4);
    assert.strictEqual(batchResult.rows, 4);
    assert.strictEqual(batchResult.cols, 60);
    assert.strictEqual(batchResult.values.length, 4 * 60);


    for (let combo = 0; combo < batchResult.combos.length; combo++) {
        const length = batchResult.combos[combo].length;
        const rowStart = combo * 60;
        const rowData = batchResult.values.slice(rowStart, rowStart + 60);



        for (let i = 0; i < length; i++) {
            assert(!isNaN(rowData[i]), `Should have valid value at index ${i} for length ${length}`);
        }


        for (let i = length; i < Math.min(length + 5, 60); i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for length ${length}`);
        }
    }
});

test('SAMA batch edge cases', () => {

    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);


    const singleBatch = wasm.sama_batch(close, {
        length_range: [10, 10, 1],
        maj_length_range: [5, 5, 0],
        min_length_range: [3, 3, 0]
    });

    assert.strictEqual(singleBatch.values.length, 15);
    assert.strictEqual(singleBatch.combos.length, 1);


    const largeBatch = wasm.sama_batch(close, {
        length_range: [10, 12, 10],
        maj_length_range: [5, 5, 0],
        min_length_range: [3, 3, 0]
    });


    assert.strictEqual(largeBatch.values.length, 15);
    assert.strictEqual(largeBatch.combos.length, 1);


    assert.throws(() => {
        wasm.sama_batch(new Float64Array([]), {
            length_range: [10, 10, 0],
            maj_length_range: [5, 5, 0],
            min_length_range: [3, 3, 0]
        });
    }, /All values are NaN|Input data slice is empty/);
});

test('SAMA batch vs single calculation', () => {

    const close = new Float64Array(testData.close.slice(0, 100));


    const singleResult = wasm.sama_js(close, 45, 13, 5);


    const batchResult = wasm.sama_batch(close, {
        length_range: [45, 45, 0],
        maj_length_range: [13, 13, 0],
        min_length_range: [5, 5, 0]
    });


    assertArrayClose(
        batchResult.values,
        singleResult,
        1e-10,
        "Batch vs single calculation mismatch"
    );
});


test('SAMA zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    const length = 10;
    const majLength = 5;
    const minLength = 3;


    const ptr = wasm.sama_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');


    const memory = wasm.__wasm.memory;
    const memView = new Float64Array(
        memory.buffer,
        ptr,
        data.length
    );


    memView.set(data);


    try {
        wasm.sama_into(ptr, ptr, data.length, length, majLength, minLength);


        const regularResult = wasm.sama_js(data, length, majLength, minLength);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue;
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {

        wasm.sama_free(ptr, data.length);
    }
});

test('SAMA zero-copy with large dataset', () => {
    const size = 10000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }

    const ptr = wasm.sama_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');

    try {
        const memory = wasm.__wasm.memory;
        const memView = new Float64Array(memory.buffer, ptr, size);
        memView.set(data);

        wasm.sama_into(ptr, ptr, size, 50, 14, 6);


        const memory2 = wasm.__wasm.memory;
        const memView2 = new Float64Array(memory2.buffer, ptr, size);


        for (let i = 0; i < 50; i++) {
            assert(!isNaN(memView2[i]), `Should have valid value at index ${i}`);
        }


        for (let i = 50; i < Math.min(60, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.sama_free(ptr, size);
    }
});

test('SAMA batch_into zero-copy', () => {
    const size = 100;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = 50 + Math.sin(i * 0.1) * 10;
    }


    const inPtr = wasm.sama_alloc(size);
    assert(inPtr !== 0, 'Failed to allocate input buffer');


    const memory = wasm.__wasm.memory;
    const inView = new Float64Array(memory.buffer, inPtr, size);
    inView.set(data);


    const numCombos = 2 * 2 * 2;
    const outPtr = wasm.sama_alloc(size * numCombos);
    assert(outPtr !== 0, 'Failed to allocate output buffer');

    try {

        const rowCount = wasm.sama_batch_into(
            inPtr, outPtr, size,
            40, 45, 5,
            12, 13, 1,
            4, 5, 1
        );

        assert.strictEqual(rowCount, numCombos, 'Should return correct row count');


        const memory2 = wasm.__wasm.memory;
        const outView = new Float64Array(memory2.buffer, outPtr, size * numCombos);


        const singleResult = wasm.sama_js(data, 40, 12, 4);
        const firstRow = Array.from(outView.slice(0, size));
        assertArrayClose(firstRow, singleResult, 1e-10, 'First row should match single calc');

    } finally {
        wasm.sama_free(inPtr, size);
        wasm.sama_free(outPtr, size * numCombos);
    }
});


test('SAMA SIMD128 consistency', () => {


    const testCases = [
        { size: 15, length: 10, majLength: 5, minLength: 3 },
        { size: 100, length: 50, majLength: 14, minLength: 6 },
        { size: 1000, length: 200, majLength: 20, minLength: 10 },
        { size: 5000, length: 500, majLength: 50, minLength: 25 }
    ];

    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = 50 + Math.sin(i * 0.1) * 10 + Math.cos(i * 0.05) * 5;
        }

        const result = wasm.sama_js(data, testCase.length, testCase.majLength, testCase.minLength);


        assert.strictEqual(result.length, data.length);


        for (let i = 0; i < testCase.length; i++) {
            assert(!isNaN(result[i]), `Should have valid value at index ${i} for size=${testCase.size}`);
        }


        let sumAfterWarmup = 0;
        let countAfterWarmup = 0;
        for (let i = testCase.length; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for size=${testCase.size}`);
            sumAfterWarmup += result[i];
            countAfterWarmup++;
        }


        if (countAfterWarmup > 0) {
            const avgAfterWarmup = sumAfterWarmup / countAfterWarmup;
            assert(avgAfterWarmup > 0, `Average value ${avgAfterWarmup} should be positive`);
            assert(avgAfterWarmup < 100, `Average value ${avgAfterWarmup} seems too large`);
        }
    }
});


test('SAMA zero-copy error handling', () => {

    assert.throws(() => {
        wasm.sama_into(0, 0, 10, 50, 14, 6);
    }, /null pointer|invalid memory/i);


    const ptr = wasm.sama_alloc(10);
    try {

        assert.throws(() => {
            wasm.sama_into(ptr, ptr, 10, 0, 14, 6);
        }, /Invalid period/);


        assert.throws(() => {
            wasm.sama_into(ptr, ptr, 10, 10, 0, 6);
        }, /Invalid period/);


        assert.throws(() => {
            wasm.sama_into(ptr, ptr, 10, 10, 14, 0);
        }, /Invalid period/);
    } finally {
        wasm.sama_free(ptr, 10);
    }
});


test('SAMA zero-copy memory management', () => {

    const sizes = [100, 1000, 10000];

    for (const size of sizes) {
        const ptr = wasm.sama_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);


        const memory = wasm.__wasm.memory;
        const memView = new Float64Array(memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }


        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
        }


        wasm.sama_free(ptr, size);
    }
});

test('SAMA batch error handling', () => {
    const close = new Float64Array(testData.close.slice(0, 10));


    assert.throws(() => {
        wasm.sama_batch(close, {
            length_range: [40, 40],
            maj_length_range: [12, 12, 0],
            min_length_range: [5, 5, 0]
        });
    }, /Invalid config/);


    assert.throws(() => {
        wasm.sama_batch(close, {
            length_range: [40, 40, 0],
            maj_length_range: [12, 12, 0]

        });
    }, /Invalid config/);


    assert.throws(() => {
        wasm.sama_batch(close, {
            length_range: "invalid",
            maj_length_range: [12, 12, 0],
            min_length_range: [5, 5, 0]
        });
    }, /Invalid config/);
});


test('SAMA wasm() function compatibility', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);


    const result1 = wasm.sama_js(data, 10, 5, 3);
    assert.strictEqual(result1.length, data.length);


    const result2 = wasm.sama_js(data, 5, 3, 2);
    assert.strictEqual(result2.length, data.length);


    const result3 = wasm.sama_js(data, 14, 7, 4);
    assert.strictEqual(result3.length, data.length);
});

test.after(() => {
    console.log('SAMA WASM tests completed');
});
