
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
    assertNoNaN
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

test('SAR partial params', () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);


    const result = wasm.sar_js(high, low, 0.02, 0.2);
    assert.strictEqual(result.length, high.length);
});

test('SAR accuracy', () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);

    const result = wasm.sar_js(high, low, 0.02, 0.2);

    assert.strictEqual(result.length, high.length);


    const expected_last_five = [
        60370.00224209362,
        60220.362107568006,
        60079.70038111392,
        59947.478358247085,
        59823.189656752256,
    ];

    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected_last_five,
        1e-4,
        "SAR last 5 values mismatch"
    );
});

test('SAR from slices', () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);

    const result = wasm.sar_js(high, low, 0.02, 0.2);
    assert.strictEqual(result.length, high.length);
});

test('SAR all NaN', () => {

    const high = new Float64Array(100).fill(NaN);
    const low = new Float64Array(100).fill(NaN);

    assert.throws(() => {
        wasm.sar_js(high, low, 0.02, 0.2);
    }, /All values are NaN/);
});

test('SAR empty input', () => {

    const empty = new Float64Array([]);

    assert.throws(() => {
        wasm.sar_js(empty, empty, 0.02, 0.2);
    }, /empty/i);
});

test('SAR mismatched lengths', () => {

    const high = new Float64Array([1.0, 2.0, 3.0]);
    const low = new Float64Array([1.0, 2.0]);


    const result = wasm.sar_js(high, low, 0.02, 0.2);
    assert.strictEqual(result.length, 2, 'Result should have length of minimum input');
});

test('SAR invalid acceleration', () => {

    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);


    assert.throws(() => {
        wasm.sar_js(data, data, 0.0, 0.2);
    }, /Invalid acceleration/);


    assert.throws(() => {
        wasm.sar_js(data, data, -0.02, 0.2);
    }, /Invalid acceleration/);


    assert.throws(() => {
        wasm.sar_js(data, data, NaN, 0.2);
    }, /Invalid acceleration/);
});

test('SAR invalid maximum', () => {

    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);


    assert.throws(() => {
        wasm.sar_js(data, data, 0.02, 0.0);
    }, /Invalid maximum/);


    assert.throws(() => {
        wasm.sar_js(data, data, 0.02, -0.2);
    }, /Invalid maximum/);


    assert.throws(() => {
        wasm.sar_js(data, data, 0.02, NaN);
    }, /Invalid maximum/);
});

test('SAR NaN handling', () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);


    for (let i = 0; i < 5; i++) {
        high[i] = NaN;
        low[i] = NaN;
    }

    const result = wasm.sar_js(high, low, 0.02, 0.2);
    assert.strictEqual(result.length, high.length);


    for (let i = 0; i < 6; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
});

test('SAR batch single parameter set', () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);


    const config = {
        acceleration_range: [0.02, 0.02, 0],
        maximum_range: [0.2, 0.2, 0]
    };

    const batch_result = wasm.sar_batch(high, low, config);


    const single_result = wasm.sar_js(high, low, 0.02, 0.2);

    assert(batch_result.values, 'Batch result should have values');
    assert(batch_result.combos, 'Batch result should have combos');
    assert.strictEqual(batch_result.rows, 1);
    assert.strictEqual(batch_result.cols, high.length);


    const batch_values = batch_result.values.slice(0, high.length);
    assertArrayClose(batch_values, single_result, 1e-10, "Batch vs single mismatch");
});

test('SAR batch multiple parameters', () => {

    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));


    const config = {
        acceleration_range: [0.01, 0.03, 0.01],
        maximum_range: [0.1, 0.3, 0.1]
    };

    const batch_result = wasm.sar_batch(high, low, config);


    assert.strictEqual(batch_result.rows, 9);
    assert.strictEqual(batch_result.cols, 100);
    assert.strictEqual(batch_result.values.length, 9 * 100);
    assert.strictEqual(batch_result.combos.length, 9);


    const expected_accs = [0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.03, 0.03];
    const expected_maxs = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3];

    for (let i = 0; i < 9; i++) {
        assertClose(batch_result.combos[i].acceleration, expected_accs[i], 1e-10);
        assertClose(batch_result.combos[i].maximum, expected_maxs[i], 1e-10);
    }
});

test('SAR fast API (in-place)', () => {

    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const len = high.length;


    const high_ptr = wasm.sar_alloc(len);
    const low_ptr = wasm.sar_alloc(len);
    const out_ptr = wasm.sar_alloc(len);

    try {

        const memory = new Float64Array(wasm.__wasm.memory.buffer);
        memory.set(high, high_ptr / 8);
        memory.set(low, low_ptr / 8);


        wasm.sar_into(high_ptr, low_ptr, out_ptr, len, 0.02, 0.2);


        const result = new Float64Array(wasm.__wasm.memory.buffer, out_ptr, len);
        const result_copy = new Float64Array(result);


        const expected = wasm.sar_js(high, low, 0.02, 0.2);
        assertArrayClose(result_copy, expected, 1e-10, "Fast API mismatch");
    } finally {

        wasm.sar_free(high_ptr, len);
        wasm.sar_free(low_ptr, len);
        wasm.sar_free(out_ptr, len);
    }
});

test('SAR fast API with aliasing', () => {

    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const len = high.length;


    const high_ptr = wasm.sar_alloc(len);
    const low_ptr = wasm.sar_alloc(len);

    try {

        const memory = new Float64Array(wasm.__wasm.memory.buffer);
        memory.set(high, high_ptr / 8);
        memory.set(low, low_ptr / 8);


        wasm.sar_into(high_ptr, low_ptr, high_ptr, len, 0.02, 0.2);


        const result = new Float64Array(wasm.__wasm.memory.buffer, high_ptr, len);
        const result_copy = new Float64Array(result);


        const expected = wasm.sar_js(high, low, 0.02, 0.2);
        assertArrayClose(result_copy, expected, 1e-10, "Fast API aliasing mismatch");
    } finally {

        wasm.sar_free(high_ptr, len);
        wasm.sar_free(low_ptr, len);
    }
});


test('SAR SIMD128 consistency', () => {


    const testCases = [
        { size: 10, acceleration: 0.02, maximum: 0.2 },
        { size: 100, acceleration: 0.01, maximum: 0.1 },
        { size: 1000, acceleration: 0.03, maximum: 0.3 },
        { size: 10000, acceleration: 0.02, maximum: 0.2 }
    ];

    for (const testCase of testCases) {

        const high = new Float64Array(testCase.size);
        const low = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            const base = 100 + Math.sin(i * 0.1) * 10;
            high[i] = base + Math.abs(Math.sin(i * 0.05)) * 5;
            low[i] = base - Math.abs(Math.sin(i * 0.05)) * 5;
        }

        const result = wasm.sar_js(high, low, testCase.acceleration, testCase.maximum);


        assert.strictEqual(result.length, high.length);


        let hasValidValues = false;
        for (let i = 0; i < Math.min(10, result.length); i++) {
            if (!isNaN(result[i])) {
                hasValidValues = true;
                break;
            }
        }
        assert(hasValidValues, `No valid values found in first 10 elements for size=${testCase.size}`);


        for (let i = 0; i < result.length; i++) {
            if (!isNaN(result[i]) && !isNaN(high[i]) && !isNaN(low[i])) {

                const minPrice = Math.min(...low.slice(Math.max(0, i-50), i+1).filter(v => !isNaN(v)));
                const maxPrice = Math.max(...high.slice(Math.max(0, i-50), i+1).filter(v => !isNaN(v)));
                assert(result[i] >= minPrice * 0.9 && result[i] <= maxPrice * 1.1,
                       `SAR value ${result[i]} at index ${i} outside reasonable range [${minPrice * 0.9}, ${maxPrice * 1.1}]`);
            }
        }
    }
});


test('SAR zero-copy error handling', () => {

    assert.throws(() => {
        wasm.sar_into(0, 0, 0, 10, 0.02, 0.2);
    }, /null pointer|invalid memory/i);


    const ptr = wasm.sar_alloc(10);
    const ptr2 = wasm.sar_alloc(10);
    try {

        assert.throws(() => {
            wasm.sar_into(ptr, ptr2, ptr, 10, 0.0, 0.2);
        }, /Invalid acceleration/);


        assert.throws(() => {
            wasm.sar_into(ptr, ptr2, ptr, 10, 0.02, 0.0);
        }, /Invalid maximum/);


        assert.throws(() => {
            wasm.sar_into(ptr, ptr2, ptr, 10, -0.02, 0.2);
        }, /Invalid acceleration/);
    } finally {
        wasm.sar_free(ptr, 10);
        wasm.sar_free(ptr2, 10);
    }
});


test('SAR zero-copy with large dataset', () => {
    const size = 100000;
    const high = new Float64Array(size);
    const low = new Float64Array(size);


    for (let i = 0; i < size; i++) {
        const base = 100 + Math.sin(i * 0.001) * 20;
        high[i] = base + Math.abs(Math.sin(i * 0.01)) * 2;
        low[i] = base - Math.abs(Math.sin(i * 0.01)) * 2;
    }

    const high_ptr = wasm.sar_alloc(size);
    const low_ptr = wasm.sar_alloc(size);
    const out_ptr = wasm.sar_alloc(size);

    assert(high_ptr !== 0, 'Failed to allocate high buffer');
    assert(low_ptr !== 0, 'Failed to allocate low buffer');
    assert(out_ptr !== 0, 'Failed to allocate output buffer');

    try {

        const highView = new Float64Array(wasm.__wasm.memory.buffer, high_ptr, size);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, low_ptr, size);
        highView.set(high);
        lowView.set(low);


        wasm.sar_into(high_ptr, low_ptr, out_ptr, size, 0.02, 0.2);


        const outView = new Float64Array(wasm.__wasm.memory.buffer, out_ptr, size);


        let validCount = 0;
        for (let i = 0; i < Math.min(1000, size); i++) {
            if (!isNaN(outView[i])) {
                validCount++;
            }
        }
        assert(validCount > 900, `Expected mostly valid values, got ${validCount}/1000`);


        for (let i = 0; i < Math.min(100, size); i++) {
            if (!isNaN(outView[i])) {
                assert(outView[i] > 0, `SAR value ${outView[i]} at index ${i} should be positive`);
                assert(outView[i] < 1000, `SAR value ${outView[i]} at index ${i} seems too large`);
            }
        }
    } finally {
        wasm.sar_free(high_ptr, size);
        wasm.sar_free(low_ptr, size);
        wasm.sar_free(out_ptr, size);
    }
});


test('SAR zero-copy memory management', () => {

    const sizes = [100, 1000, 10000, 50000];

    for (const size of sizes) {
        const ptr = wasm.sar_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);


        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 2.5;
        }


        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 2.5, `Memory corruption at index ${i}`);
        }


        wasm.sar_free(ptr, size);
    }
});


test('SAR batch edge cases', () => {
    const high = new Float64Array([100, 105, 103, 107, 110, 108, 112, 115, 113, 117]);
    const low = new Float64Array([98, 100, 101, 102, 105, 104, 108, 110, 109, 112]);


    const singleBatch = wasm.sar_batch(high, low, {
        acceleration_range: [0.02, 0.02, 0.1],
        maximum_range: [0.2, 0.2, 0.1]
    });

    assert.strictEqual(singleBatch.values.length, 10);
    assert.strictEqual(singleBatch.combos.length, 1);
    assert.strictEqual(singleBatch.rows, 1);
    assert.strictEqual(singleBatch.cols, 10);


    const largeBatch = wasm.sar_batch(high, low, {
        acceleration_range: [0.01, 0.02, 0.05],
        maximum_range: [0.1, 0.2, 0.5]
    });


    assert.strictEqual(largeBatch.combos.length, 1);
    assert.strictEqual(largeBatch.combos[0].acceleration, 0.01);
    assert.strictEqual(largeBatch.combos[0].maximum, 0.1);


    assert.throws(() => {
        wasm.sar_batch(new Float64Array([]), new Float64Array([]), {
            acceleration_range: [0.02, 0.02, 0],
            maximum_range: [0.2, 0.2, 0]
        });
    }, /Empty data|All values are NaN/);
});


test('SAR batch error handling', () => {
    const high = new Float64Array(testData.high.slice(0, 10));
    const low = new Float64Array(testData.low.slice(0, 10));


    assert.throws(() => {
        wasm.sar_batch(high, low, {
            acceleration_range: [0.02, 0.02],
            maximum_range: [0.2, 0.2, 0]
        });
    }, /Invalid config/);


    assert.throws(() => {
        wasm.sar_batch(high, low, {
            acceleration_range: [0.02, 0.02, 0]

        });
    }, /Invalid config/);


    assert.throws(() => {
        wasm.sar_batch(high, low, {
            acceleration_range: "invalid",
            maximum_range: [0.2, 0.2, 0]
        });
    }, /Invalid config/);


    const shortLow = new Float64Array(5);
    const batchResult = wasm.sar_batch(high, shortLow, {
        acceleration_range: [0.02, 0.02, 0],
        maximum_range: [0.2, 0.2, 0]
    });

    assert.strictEqual(batchResult.cols, 5, 'Batch should use minimum length');
});

test.after(() => {
    console.log('SAR WASM tests completed');
});
