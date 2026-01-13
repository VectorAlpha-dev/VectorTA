
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

test('LRSI partial params', () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);

    const result = wasm.lrsi_js(high, low, 0.2);
    assert.strictEqual(result.length, high.length);
});

test('LRSI accuracy', () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const alpha = 0.2;

    const result = wasm.lrsi_js(
        high,
        low,
        alpha
    );

    assert.strictEqual(result.length, high.length);


    let firstValid = null;
    for (let i = 0; i < high.length; i++) {
        const price = (high[i] + low[i]) / 2.0;
        if (!isNaN(price)) {
            firstValid = i;
            break;
        }
    }

    if (firstValid !== null) {
        const warmupEnd = firstValid + 3;


        if (warmupEnd > 0) {
            assertAllNaN(result.slice(0, warmupEnd), `Expected NaN in warmup period [0:${warmupEnd}]`);
        }


        if (result.length > warmupEnd + 10) {
            for (let i = warmupEnd; i < Math.min(warmupEnd + 50, result.length); i++) {
                if (!isNaN(result[i])) {
                    assert(result[i] >= 0.0 && result[i] <= 1.0,
                        `LRSI value at index ${i} (${result[i]}) should be between 0 and 1`);
                }
            }
        }
    }


    const expectedLast5 = [0.0, 0.0, 0.0, 0.0, 0.0];
    const last5 = Array.from(result.slice(result.length - 5));
    assertArrayClose(last5, expectedLast5, 1e-9, 'LRSI last 5 values mismatch vs Rust');
});

test('LRSI default candles', () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);


    const result = wasm.lrsi_js(high, low, 0.2);
    assert.strictEqual(result.length, high.length);


    let foundNonNaN = false;
    for (let i = 3; i < result.length; i++) {
        if (!isNaN(result[i])) {
            foundNonNaN = true;
            break;
        }
    }
    assert(foundNonNaN, 'No valid values found after warmup period');
});

test('LRSI invalid alpha', () => {

    const high = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    const low = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);


    assert.throws(() => {
        wasm.lrsi_js(high, low, 1.2);
    }, /Invalid alpha/);


    assert.throws(() => {
        wasm.lrsi_js(high, low, 0.0);
    }, /Invalid alpha/);


    assert.throws(() => {
        wasm.lrsi_js(high, low, -0.1);
    }, /Invalid alpha/);
});

test('LRSI empty data', () => {

    const high = new Float64Array([]);
    const low = new Float64Array([]);

    assert.throws(() => {
        wasm.lrsi_js(high, low, 0.2);
    }, /All values are NaN|Empty/);
});

test('LRSI all NaN', () => {

    const high = new Float64Array([NaN, NaN, NaN]);
    const low = new Float64Array([NaN, NaN, NaN]);

    assert.throws(() => {
        wasm.lrsi_js(high, low, 0.2);
    }, /All values are NaN/);
});

test('LRSI very small dataset', () => {

    const high = new Float64Array([1.0, 1.0]);
    const low = new Float64Array([1.0, 1.0]);

    assert.throws(() => {
        wasm.lrsi_js(high, low, 0.2);
    }, /Not enough valid data/);
});

test('LRSI NaN handling', () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);

    const result = wasm.lrsi_js(high, low, 0.2);
    assert.strictEqual(result.length, high.length);


    let firstValid = null;
    for (let i = 0; i < high.length; i++) {
        const price = (high[i] + low[i]) / 2.0;
        if (!isNaN(price)) {
            firstValid = i;
            break;
        }
    }

    if (firstValid !== null) {
        const warmupEnd = firstValid + 3;


        if (warmupEnd > 0) {
            assertAllNaN(result.slice(0, warmupEnd), `Expected NaN in warmup period [0:${warmupEnd}]`);
        }


        let hasValues = false;
        for (let i = warmupEnd; i < Math.min(result.length, warmupEnd + 20); i++) {
            if (!isNaN(result[i])) {
                hasValues = true;
                break;
            }
        }
        assert(hasValues, "Expected valid values after warmup period");
    }
});


test('LRSI batch - single parameter', () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);

    const batchResult = wasm.lrsi_batch(high, low, {
        alpha_range: [0.2, 0.2, 0]
    });


    const singleResult = wasm.lrsi_js(high, low, 0.2);

    assert.strictEqual(batchResult.values.length, singleResult.length);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('LRSI batch - multiple parameters', () => {

    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));


    const batchResult = wasm.lrsi_batch(high, low, {
        alpha_range: [0.1, 0.3, 0.1]
    });


    assert.strictEqual(batchResult.values.length, 3 * 100);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 100);


    assert.strictEqual(batchResult.combos.length, 3);


    assertClose(batchResult.combos[0].alpha, 0.1, 1e-10, 'First alpha');
    assertClose(batchResult.combos[1].alpha, 0.2, 1e-10, 'Second alpha');
    assertClose(batchResult.combos[2].alpha, 0.3, 1e-10, 'Third alpha');
});

test('LRSI batch - parameter sweep', () => {

    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));

    const batchResult = wasm.lrsi_batch(high, low, {
        alpha_range: [0.1, 0.5, 0.1]
    });


    assert.strictEqual(batchResult.combos.length, 5);
    assert.strictEqual(batchResult.rows, 5);
    assert.strictEqual(batchResult.cols, 50);
    assert.strictEqual(batchResult.values.length, 5 * 50);


    for (let i = 0; i < 5; i++) {
        const expectedAlpha = 0.1 + i * 0.1;
        assert(Math.abs(batchResult.combos[i].alpha - expectedAlpha) < 1e-10,
               `Combo ${i} alpha mismatch: expected ${expectedAlpha}, got ${batchResult.combos[i].alpha}`);
    }
});

test('LRSI batch - edge cases', () => {

    const high = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const low = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);


    const singleBatch = wasm.lrsi_batch(high, low, {
        alpha_range: [0.2, 0.2, 0.1]
    });

    assert.strictEqual(singleBatch.values.length, 10);
    assert.strictEqual(singleBatch.combos.length, 1);


    const largeBatch = wasm.lrsi_batch(high, low, {
        alpha_range: [0.2, 0.3, 0.5]
    });


    assert.strictEqual(largeBatch.values.length, 10);
    assert.strictEqual(largeBatch.combos.length, 1);


    assert.throws(() => {
        wasm.lrsi_batch(new Float64Array([]), new Float64Array([]), {
            alpha_range: [0.2, 0.2, 0]
        });
    }, /Empty data|Empty input/);
});

test('LRSI batch - error handling', () => {
    const high = new Float64Array(testData.high.slice(0, 10));
    const low = new Float64Array(testData.low.slice(0, 10));


    assert.throws(() => {
        wasm.lrsi_batch(high, low, {
            alpha_range: [0.2, 0.2]
        });
    }, /Invalid config/);


    assert.throws(() => {
        wasm.lrsi_batch(high, low, {

        });
    }, /Invalid config/);


    assert.throws(() => {
        wasm.lrsi_batch(high, low, {
            alpha_range: "invalid"
        });
    }, /Invalid config/);
});


test('LRSI zero-copy API', () => {

    const data = testData.high.slice(0, 100);
    const high = new Float64Array(data);
    const low = new Float64Array(testData.low.slice(0, 100));
    const alpha = 0.2;


    const ptr = wasm.lrsi_alloc(high.length);
    assert(ptr !== 0, 'Failed to allocate memory');


    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        high.length
    );


    memView.set(high);


    try {

        const lowPtr = wasm.lrsi_alloc(low.length);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, low.length);
        lowView.set(low);

        wasm.lrsi_into(ptr, lowPtr, ptr, high.length, alpha);


        const regularResult = wasm.lrsi_js(high, low, alpha);
        for (let i = 0; i < high.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue;
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }

        wasm.lrsi_free(lowPtr, low.length);
    } finally {

        wasm.lrsi_free(ptr, high.length);
    }
});

test('LRSI zero-copy with large dataset', () => {
    const size = 10000;
    const high = new Float64Array(size);
    const low = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        high[i] = Math.sin(i * 0.01) + Math.random() * 0.1 + 1;
        low[i] = high[i] - Math.random() * 0.2;
    }

    const highPtr = wasm.lrsi_alloc(size);
    const lowPtr = wasm.lrsi_alloc(size);
    const outPtr = wasm.lrsi_alloc(size);
    assert(highPtr !== 0 && lowPtr !== 0 && outPtr !== 0, 'Failed to allocate large buffers');

    try {
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, size);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, size);
        highView.set(high);
        lowView.set(low);

        wasm.lrsi_into(highPtr, lowPtr, outPtr, size, 0.2);


        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, size);



        for (let i = 0; i < 3; i++) {
            assert(isNaN(outView[i]), `Expected NaN at warmup index ${i}`);
        }


        for (let i = 3; i < Math.min(10, size); i++) {
            assert(!isNaN(outView[i]), `Unexpected NaN at index ${i}`);
            assert(outView[i] >= 0.0 && outView[i] <= 1.0,
                `LRSI value at index ${i} (${outView[i]}) should be between 0 and 1`);
        }
    } finally {
        wasm.lrsi_free(highPtr, size);
        wasm.lrsi_free(lowPtr, size);
        wasm.lrsi_free(outPtr, size);
    }
});

test('LRSI zero-copy error handling', () => {

    assert.throws(() => {
        wasm.lrsi_into(0, 0, 0, 10, 0.2);
    }, /null pointer|invalid memory/i);


    const ptr1 = wasm.lrsi_alloc(10);
    const ptr2 = wasm.lrsi_alloc(10);
    const ptr3 = wasm.lrsi_alloc(10);
    try {

        assert.throws(() => {
            wasm.lrsi_into(ptr1, ptr2, ptr3, 10, 0.0);
        }, /Invalid alpha/);


        assert.throws(() => {
            wasm.lrsi_into(ptr1, ptr2, ptr3, 10, 1.5);
        }, /Invalid alpha/);
    } finally {
        wasm.lrsi_free(ptr1, 10);
        wasm.lrsi_free(ptr2, 10);
        wasm.lrsi_free(ptr3, 10);
    }
});


test('LRSI batch_into zero-copy API', () => {
    const size = 100;
    const high = new Float64Array(size);
    const low = new Float64Array(size);


    for (let i = 0; i < size; i++) {
        high[i] = Math.sin(i * 0.1) + 10;
        low[i] = high[i] - Math.random() * 0.5;
    }


    const alphaStart = 0.1;
    const alphaEnd = 0.3;
    const alphaStep = 0.1;
    const numRows = 3;

    const highPtr = wasm.lrsi_alloc(size);
    const lowPtr = wasm.lrsi_alloc(size);
    const outPtr = wasm.lrsi_alloc(size * numRows);

    assert(highPtr !== 0 && lowPtr !== 0 && outPtr !== 0, 'Failed to allocate batch buffers');

    try {

        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, size);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, size);
        highView.set(high);
        lowView.set(low);


        const actualRows = wasm.lrsi_batch_into(
            highPtr, lowPtr, outPtr, size,
            alphaStart, alphaEnd, alphaStep
        );

        assert.strictEqual(actualRows, numRows, 'Unexpected number of rows returned');


        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, size * numRows);


        for (let row = 0; row < numRows; row++) {
            const rowStart = row * size;


            for (let i = 0; i < 3; i++) {
                assert(isNaN(outView[rowStart + i]),
                    `Row ${row}: Expected NaN at warmup index ${i}`);
            }


            for (let i = 3; i < Math.min(10, size); i++) {
                const val = outView[rowStart + i];
                assert(!isNaN(val), `Row ${row}: Unexpected NaN at index ${i}`);
                assert(val >= 0.0 && val <= 1.0,
                    `Row ${row}: LRSI value at index ${i} (${val}) should be between 0 and 1`);
            }
        }


        const regularBatch = wasm.lrsi_batch(high, low, {
            alpha_range: [alphaStart, alphaEnd, alphaStep]
        });

        assert.strictEqual(regularBatch.rows, numRows);


        for (let i = 0; i < Math.min(100, size * numRows); i++) {
            const regular = regularBatch.values[i];
            const zerocopy = outView[i];
            if (isNaN(regular) && isNaN(zerocopy)) continue;
            assert(Math.abs(regular - zerocopy) < 1e-10,
                `Mismatch at index ${i}: regular=${regular}, zerocopy=${zerocopy}`);
        }

    } finally {
        wasm.lrsi_free(highPtr, size);
        wasm.lrsi_free(lowPtr, size);
        wasm.lrsi_free(outPtr, size * numRows);
    }
});

test('LRSI batch_into error handling', () => {

    assert.throws(() => {
        wasm.lrsi_batch_into(0, 0, 0, 10, 0.1, 0.3, 0.1);
    }, /null pointer/i);


    const size = 10;
    const rows = 3;
    const ptr1 = wasm.lrsi_alloc(size);
    const ptr2 = wasm.lrsi_alloc(size);
    const ptr3 = wasm.lrsi_alloc(rows * size);

    try {

        assert.throws(() => {
            wasm.lrsi_batch_into(ptr1, ptr2, ptr3, size, 1.5, 2.0, 0.1);
        }, /Invalid alpha/);


        assert.throws(() => {
            wasm.lrsi_batch_into(ptr1, ptr2, ptr3, size, 0.0, 0.5, 0.1);
        }, /Invalid alpha/);

    } finally {
        wasm.lrsi_free(ptr1, size);
        wasm.lrsi_free(ptr2, size);
        wasm.lrsi_free(ptr3, rows * size);
    }
});


test('LRSI zero-copy memory management', () => {

    const sizes = [100, 1000, 10000];

    for (const size of sizes) {
        const ptr = wasm.lrsi_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);


        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }


        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Memory pattern mismatch at index ${i}`);
        }


        wasm.lrsi_free(ptr, size);
    }
});

test.after(() => {
    console.log('LRSI WASM tests completed');
});
