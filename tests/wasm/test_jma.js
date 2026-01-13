
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

test('JMA partial params', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.jma_js(close, 7, 50.0, 2);
    assert.strictEqual(result.length, close.length);
});

test('JMA accuracy', async () => {

    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.jma;

    const result = wasm.jma_js(
        close,
        expected.defaultParams.period,
        expected.defaultParams.phase,
        expected.defaultParams.power
    );

    assert.strictEqual(result.length, close.length);


    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-6,
        "JMA last 5 values mismatch"
    );


    await compareWithRust('jma', result, 'close', expected.defaultParams);
});

test('JMA default candles', async () => {

    const close = new Float64Array(testData.close);

    const result = wasm.jma_js(close, 7, 50.0, 2);
    assert.strictEqual(result.length, close.length);


    await compareWithRust('jma', result, 'close', { period: 7, phase: 50.0, power: 2 });
});

test('JMA zero period', () => {

    const inputData = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.jma_js(inputData, 0, 50.0, 2);
    });
});

test('JMA period exceeds length', () => {

    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.jma_js(dataSmall, 10, 50.0, 2);
    });
});

test('JMA very small dataset', () => {

    const dataSingle = new Float64Array([42.0]);

    assert.throws(() => {
        wasm.jma_js(dataSingle, 7, 50.0, 2);
    });
});

test('JMA empty input', () => {

    const dataEmpty = new Float64Array([]);

    assert.throws(() => {
        wasm.jma_js(dataEmpty, 7, 50.0, 2);
    });
});

test('JMA all NaN', () => {

    const data = new Float64Array([NaN, NaN, NaN, NaN, NaN]);

    assert.throws(() => {
        wasm.jma_js(data, 3, 50.0, 2);
    });
});

test('JMA invalid phase', () => {

    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);


    assert.throws(() => {
        wasm.jma_js(data, 3, NaN, 2);
    });


    assert.throws(() => {
        wasm.jma_js(data, 3, Infinity, 2);
    });
});

test('JMA reinput', () => {

    const close = new Float64Array(testData.close);


    const firstResult = wasm.jma_js(close, 7, 50.0, 2);


    const secondResult = wasm.jma_js(firstResult, 7, 50.0, 2);

    assert.strictEqual(secondResult.length, firstResult.length);
});

test('JMA NaN handling', () => {

    const dataWithNan = new Float64Array([NaN, NaN, NaN, ...testData.close.slice(0, 50)]);
    const period = 7;

    const result = wasm.jma_js(dataWithNan, period, 50.0, 2);

    assert.strictEqual(result.length, dataWithNan.length);


    assertAllNaN(result.slice(0, 3), "Expected NaN before first valid data");


    assert(isFinite(result[3]), "Expected finite value at first valid");


    for (let i = 3; i < result.length; i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i}`);
    }
});

test('JMA NaN handling (original)', () => {

    const close = new Float64Array(testData.close);
    const period = 7;

    const result = wasm.jma_js(close, period, 50.0, 2);

    assert.strictEqual(result.length, close.length);


    let finiteCount = 0;
    for (let val of result) {
        if (isFinite(val)) finiteCount++;
    }
    assert(finiteCount > close.length - period * 2, "Too many non-finite values");
});

test('JMA batch', () => {

    const close = new Float64Array(testData.close);


    const period_start = 5, period_end = 9, period_step = 2;
    const phase_start = 40.0, phase_end = 60.0, phase_step = 10.0;
    const power_start = 1, power_end = 3, power_step = 1;

    const batch_result = wasm.jma_batch_js(
        close,
        period_start, period_end, period_step,
        phase_start, phase_end, phase_step,
        power_start, power_end, power_step
    );
    const metadata = wasm.jma_batch_metadata_js(
        period_start, period_end, period_step,
        phase_start, phase_end, phase_step,
        power_start, power_end, power_step
    );


    assert.strictEqual(metadata.length, 27 * 3);


    assert.strictEqual(batch_result.length, 27 * close.length);


    const individual_result = wasm.jma_js(close, 5, 40.0, 1);


    const row = batch_result.slice(0, close.length);

    assertArrayClose(row, individual_result, 1e-9, 'First combination');
});

test('JMA different params', () => {

    const close = new Float64Array(testData.close);


    const paramSets = [
        [5, 0.0, 1],
        [7, 50.0, 2],
        [10, 100.0, 3],
        [14, -100.0, 2],
    ];

    for (const [period, phase, power] of paramSets) {
        const result = wasm.jma_js(close, period, phase, power);
        assert.strictEqual(result.length, close.length);


        let finiteCount = 0;
        for (let val of result) {
            if (isFinite(val)) finiteCount++;
        }
        assert(finiteCount > close.length - period * 2,
            `Too many non-finite values for params (${period}, ${phase}, ${power})`);
    }
});

test('JMA batch performance', () => {

    const close = new Float64Array(testData.close.slice(0, 1000));


    const startBatch = performance.now();
    const batchResult = wasm.jma_batch_js(
        close,
        5, 7, 2,
        40.0, 50.0, 10.0,
        1, 2, 1
    );
    const batchTime = performance.now() - startBatch;

    const startSingle = performance.now();
    const singleResults = [];
    for (let period = 5; period <= 7; period += 2) {
        for (let phase = 40.0; phase <= 50.0; phase += 10.0) {
            for (let power = 1; power <= 2; power += 1) {
                singleResults.push(...wasm.jma_js(close, period, phase, power));
            }
        }
    }
    const singleTime = performance.now() - startSingle;


    console.log(`Batch time: ${batchTime.toFixed(2)}ms, Single time: ${singleTime.toFixed(2)}ms`);


    assertArrayClose(batchResult, singleResults, 1e-9, 'Batch vs single results');
});

test('JMA phase range', () => {

    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);


    for (const phase of [-100.0, -50.0, 0.0, 50.0, 100.0]) {
        const result = wasm.jma_js(data, 3, phase, 2);
        assert.strictEqual(result.length, data.length);

        let finiteCount = 0;
        for (let val of result) {
            if (isFinite(val)) finiteCount++;
        }
        assert(finiteCount > 0, `No finite values for phase ${phase}`);
    }
});

test('JMA power values', () => {

    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);


    for (const power of [1, 2, 3, 5, 10]) {
        const result = wasm.jma_js(data, 3, 50.0, power);
        assert.strictEqual(result.length, data.length);

        let finiteCount = 0;
        for (let val of result) {
            if (isFinite(val)) finiteCount++;
        }
        assert(finiteCount > 0, `No finite values for power ${power}`);
    }
});

test('JMA single param batch', () => {

    const close = new Float64Array(testData.close.slice(0, 100));


    const batch_result = wasm.jma_batch_js(
        close,
        5, 9, 2,
        50.0, 50.0, 0.0,
        2, 2, 0
    );


    assert.strictEqual(batch_result.length, 3 * close.length);


    for (let i = 0; i < 3; i++) {
        const period = 5 + i * 2;
        const start = i * close.length;
        const row = batch_result.slice(start, start + close.length);

        const individual = wasm.jma_js(close, period, 50.0, 2);
        assertArrayClose(row, individual, 1e-9, `Period ${period}`);
    }
});

test('JMA batch metadata', () => {

    const metadata = wasm.jma_batch_metadata_js(
        5, 7, 2,
        40.0, 50.0, 10.0,
        1, 2, 1
    );



    assert.strictEqual(metadata.length, 24);


    const expectedFirstCombinations = [
        5, 40.0, 1,
        5, 40.0, 2,
        5, 50.0, 1,
        5, 50.0, 2,
    ];

    for (let i = 0; i < expectedFirstCombinations.length; i++) {
        assertClose(
            metadata[i],
            expectedFirstCombinations[i],
            1e-9,
            `Metadata value at index ${i}`
        );
    }
});

test('JMA warmup behavior', () => {


    const data = new Float64Array([NaN, NaN, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    const period = 3;

    const result = wasm.jma_js(data, period, 50.0, 2);


    assert(isNaN(result[0]), "Index 0 should be NaN");
    assert(isNaN(result[1]), "Index 1 should be NaN");


    assert(isFinite(result[2]), "Index 2 (first valid) should have a value");


    for (let i = 3; i < result.length; i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i}`);
    }


    const cleanData = new Float64Array(testData.close.slice(0, 20));
    const cleanResult = wasm.jma_js(cleanData, 7, 50.0, 2);


    assert(isFinite(cleanResult[0]), "JMA should output at first valid (index 0 for clean data)");
});

test('JMA consistency across calls', () => {

    const close = new Float64Array(testData.close.slice(0, 100));

    const result1 = wasm.jma_js(close, 7, 50.0, 2);
    const result2 = wasm.jma_js(close, 7, 50.0, 2);

    assertArrayClose(result1, result2, 1e-15, "JMA results not consistent");
});

test('JMA parameter boundaries', () => {

    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);


    const result1 = wasm.jma_js(data, 1, 50.0, 2);
    assert.strictEqual(result1.length, data.length);


    const smallData = new Float64Array([1, 2, 3, 4, 5]);
    const result2 = wasm.jma_js(smallData, 4, 50.0, 2);
    assert.strictEqual(result2.length, smallData.length);
});


test('JMA zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const period = 3;
    const phase = 50.0;
    const power = 2;


    const ptr = wasm.jma_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');


    const memory = wasm.__wasm ? wasm.__wasm.memory : (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory);
    const memView = new Float64Array(
        memory.buffer,
        ptr,
        data.length
    );


    memView.set(data);


    try {
        wasm.jma_into(ptr, ptr, data.length, period, phase, power);


        const regularResult = wasm.jma_js(data, period, phase, power);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue;
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {

        wasm.jma_free(ptr, data.length);
    }
});

test('JMA zero-copy with NaN data', () => {
    const data = new Float64Array([NaN, NaN, 1, 2, 3, 4, 5, 6, 7, 8]);
    const period = 3;
    const phase = 50.0;
    const power = 2;

    const ptr = wasm.jma_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');

    try {
        const memory = wasm.__wasm ? wasm.__wasm.memory : (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory);
        const memView = new Float64Array(memory.buffer, ptr, data.length);
        memView.set(data);

        wasm.jma_into(ptr, ptr, data.length, period, phase, power);


        const memory2 = wasm.__wasm ? wasm.__wasm.memory : (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory);
        const memView2 = new Float64Array(memory2.buffer, ptr, data.length);


        assert(isNaN(memView2[0]), "Expected NaN at index 0");
        assert(isNaN(memView2[1]), "Expected NaN at index 1");


        assert(isFinite(memView2[2]), "Expected finite value at first valid (index 2)");
    } finally {
        wasm.jma_free(ptr, data.length);
    }
});

test('JMA zero-copy error handling', () => {

    assert.throws(() => {
        wasm.jma_into(0, 0, 10, 3, 50.0, 2);
    }, /null pointer/i);


    const ptr = wasm.jma_alloc(10);
    try {

        assert.throws(() => {
            wasm.jma_into(ptr, ptr, 10, 0, 50.0, 2);
        }, /Invalid period/);


        assert.throws(() => {
            wasm.jma_into(ptr, ptr, 10, 3, NaN, 2);
        }, /Invalid phase/);
    } finally {
        wasm.jma_free(ptr, 10);
    }
});

test('JMA zero-copy memory management', () => {

    const sizes = [100, 1000, 10000];

    for (const size of sizes) {
        const ptr = wasm.jma_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);


        const memory = wasm.__wasm ? wasm.__wasm.memory : (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory);
        const memView = new Float64Array(memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }


        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
        }


        wasm.jma_free(ptr, size);
    }
});


test('JMA batch - new ergonomic API with single parameter', () => {
    const close = new Float64Array(testData.close.slice(0, 100));

    const result = wasm.jma_batch(close, {
        period_range: [7, 7, 0],
        phase_range: [50.0, 50.0, 0],
        power_range: [2, 2, 0]
    });


    assert(result.values, 'Should have values array');
    assert(result.combos, 'Should have combos array');
    assert(typeof result.rows === 'number', 'Should have rows count');
    assert(typeof result.cols === 'number', 'Should have cols count');


    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.combos.length, 1);
    assert.strictEqual(result.values.length, close.length);


    const combo = result.combos[0];
    assert.strictEqual(combo.period, 7);
    assert.strictEqual(combo.phase, 50.0);
    assert.strictEqual(combo.power, 2);


    const singleResult = wasm.jma_js(close, 7, 50.0, 2);
    assertArrayClose(result.values, singleResult, 1e-10, 'Batch vs single mismatch');
});

test('JMA batch - new API with multiple parameters', () => {
    const close = new Float64Array(testData.close.slice(0, 50));

    const result = wasm.jma_batch(close, {
        period_range: [5, 7, 2],
        phase_range: [40.0, 50.0, 10.0],
        power_range: [2, 2, 0]
    });


    assert.strictEqual(result.rows, 4);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.combos.length, 4);
    assert.strictEqual(result.values.length, 200);


    const expectedCombos = [
        { period: 5, phase: 40.0, power: 2 },
        { period: 5, phase: 50.0, power: 2 },
        { period: 7, phase: 40.0, power: 2 },
        { period: 7, phase: 50.0, power: 2 }
    ];

    for (let i = 0; i < expectedCombos.length; i++) {
        assert.strictEqual(result.combos[i].period, expectedCombos[i].period);
        assert.strictEqual(result.combos[i].phase, expectedCombos[i].phase);
        assert.strictEqual(result.combos[i].power, expectedCombos[i].power);
    }


    const firstRow = result.values.slice(0, result.cols);
    const singleResult = wasm.jma_js(close, 5, 40.0, 2);
    assertArrayClose(firstRow, singleResult, 1e-10, 'First row mismatch');
});

test('JMA batch - with NaN data', () => {

    const data = new Float64Array([NaN, NaN, ...Array.from({length: 20}, (_, i) => i + 1)]);

    const result = wasm.jma_batch(data, {
        period_range: [3, 5, 2],
        phase_range: [50.0, 50.0, 0],
        power_range: [2, 2, 0]
    });

    assert.strictEqual(result.rows, 2);
    assert.strictEqual(result.cols, data.length);


    for (let row = 0; row < 2; row++) {
        const rowStart = row * data.length;
        assert(isNaN(result.values[rowStart]), `Row ${row} index 0 should be NaN`);
        assert(isNaN(result.values[rowStart + 1]), `Row ${row} index 1 should be NaN`);

        assert(isFinite(result.values[rowStart + 2]), `Row ${row} index 2 should be finite`);
    }
});
