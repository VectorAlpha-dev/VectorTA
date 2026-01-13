
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

test('Aroon partial params', () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);

    const result = wasm.aroon_js(high, low, 14);
    assert(result.up, 'Should have up array');
    assert(result.down, 'Should have down array');
    assert.strictEqual(result.up.length, high.length);
    assert.strictEqual(result.down.length, low.length);
});

test('Aroon accuracy', async () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const expected = EXPECTED_OUTPUTS.aroon;

    const result = wasm.aroon_js(
        high, low,
        expected.defaultParams.length || 14
    );

    assert.strictEqual(result.up.length, high.length);
    assert.strictEqual(result.down.length, low.length);


    const last5Up = Array.from(result.up.slice(-5));
    const last5Down = Array.from(result.down.slice(-5));

    assertArrayClose(
        last5Up,
        expected.last5Up,
        1e-2,
        "Aroon up last 5 values mismatch"
    );
    assertArrayClose(
        last5Down,
        expected.last5Down,
        1e-2,
        "Aroon down last 5 values mismatch"
    );




});

test('Aroon default candles', () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);

    const result = wasm.aroon_js(high, low, 14);
    assert.strictEqual(result.up.length, high.length);
    assert.strictEqual(result.down.length, low.length);
});

test('Aroon zero length', () => {

    const high = new Float64Array([10.0, 11.0, 12.0]);
    const low = new Float64Array([9.0, 10.0, 11.0]);

    assert.throws(() => {
        wasm.aroon_js(high, low, 0);
    }, /Invalid length/);
});

test('Aroon length exceeds data', () => {

    const high = new Float64Array([10.0, 11.0, 12.0]);
    const low = new Float64Array([9.0, 10.0, 11.0]);

    assert.throws(() => {
        wasm.aroon_js(high, low, 14);
    }, /Invalid length/);
});

test('Aroon very small dataset', () => {

    const high = new Float64Array([100.0]);
    const low = new Float64Array([99.5]);

    assert.throws(() => {
        wasm.aroon_js(high, low, 14);
    }, /Invalid length|Not enough valid data/);
});

test('Aroon empty input', () => {

    const empty = new Float64Array([]);

    assert.throws(() => {
        wasm.aroon_js(empty, empty, 14);
    }, /Input data slice is empty/);
});

test('Aroon mismatched lengths', () => {

    const high = new Float64Array([10.0, 11.0, 12.0]);
    const low = new Float64Array([9.0, 10.0]);

    assert.throws(() => {
        wasm.aroon_js(high, low, 2);
    }, /Mismatch in high\/low slice length/);
});

test('Aroon reinput', () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);


    const firstResult = wasm.aroon_js(high, low, 14);
    assert.strictEqual(firstResult.up.length, high.length);
    assert.strictEqual(firstResult.down.length, low.length);


    const secondResult = wasm.aroon_js(high, low, 5);
    assert.strictEqual(secondResult.up.length, high.length);
    assert.strictEqual(secondResult.down.length, low.length);
});

test('Aroon NaN handling', () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);

    const result = wasm.aroon_js(high, low, 14);
    assert.strictEqual(result.up.length, high.length);
    assert.strictEqual(result.down.length, low.length);


    if (result.up.length > 240) {
        for (let i = 240; i < result.up.length; i++) {
            assert(!isNaN(result.up[i]), `Found unexpected NaN in up at index ${i}`);
            assert(!isNaN(result.down[i]), `Found unexpected NaN in down at index ${i}`);
        }
    }


    assertAllNaN(Array.from(result.up.slice(0, 14)), "Expected NaN in up warmup period");
    assertAllNaN(Array.from(result.down.slice(0, 14)), "Expected NaN in down warmup period");
});

test('Aroon all NaN input', () => {

    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);


    assert.throws(() => {
        wasm.aroon_js(allNaN, allNaN, 14);
    }, /All values are NaN/, 'Expected AllValuesNaN error');
});

test('Aroon batch single parameter set', () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);


    const batchResult = wasm.aroon_batch_js(
        high, low,
        14, 14, 0
    );


    const singleResult = wasm.aroon_js(high, low, 14);

    assert.strictEqual(batchResult.up.length, singleResult.up.length);
    assert.strictEqual(batchResult.down.length, singleResult.down.length);
    assertArrayClose(Array.from(batchResult.up), Array.from(singleResult.up), 1e-10, "Batch vs single up mismatch");
    assertArrayClose(Array.from(batchResult.down), Array.from(singleResult.down), 1e-10, "Batch vs single down mismatch");
});

test('Aroon batch multiple lengths', () => {

    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));


    const batchResult = wasm.aroon_batch_js(
        high, low,
        10, 20, 5
    );


    assert.strictEqual(batchResult.up.length, 3 * 100);
    assert.strictEqual(batchResult.down.length, 3 * 100);


    const lengths = [10, 15, 20];
    for (let i = 0; i < lengths.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowUp = batchResult.up.slice(rowStart, rowEnd);
        const rowDown = batchResult.down.slice(rowStart, rowEnd);

        const singleResult = wasm.aroon_js(high, low, lengths[i]);
        assertArrayClose(
            Array.from(rowUp),
            Array.from(singleResult.up),
            1e-10,
            `Length ${lengths[i]} up mismatch`
        );
        assertArrayClose(
            Array.from(rowDown),
            Array.from(singleResult.down),
            1e-10,
            `Length ${lengths[i]} down mismatch`
        );
    }
});

test('Aroon batch metadata', () => {

    const metadata = wasm.aroon_batch_metadata_js(
        10, 20, 5
    );


    assert.strictEqual(metadata.length, 3);


    assert.strictEqual(metadata[0], 10);
    assert.strictEqual(metadata[1], 15);
    assert.strictEqual(metadata[2], 20);
});

test('Aroon batch full parameter sweep', () => {

    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));

    const batchResult = wasm.aroon_batch_js(
        high, low,
        10, 14, 2
    );

    const metadata = wasm.aroon_batch_metadata_js(10, 14, 2);


    const numCombos = metadata.length;
    assert.strictEqual(numCombos, 3);
    assert.strictEqual(batchResult.up.length, 3 * 50);
    assert.strictEqual(batchResult.down.length, 3 * 50);


    for (let combo = 0; combo < numCombos; combo++) {
        const length = metadata[combo];
        const rowStart = combo * 50;
        const rowUp = Array.from(batchResult.up.slice(rowStart, rowStart + 50));
        const rowDown = Array.from(batchResult.down.slice(rowStart, rowStart + 50));


        for (let i = 0; i < Math.min(length, 50); i++) {
            assert(isNaN(rowUp[i]), `Expected NaN in up at warmup index ${i} for length ${length}`);
            assert(isNaN(rowDown[i]), `Expected NaN in down at warmup index ${i} for length ${length}`);
        }


        if (length < 50) {
            for (let i = length; i < 50; i++) {
                assert(!isNaN(rowUp[i]), `Unexpected NaN in up at index ${i} for length ${length}`);
                assert(!isNaN(rowDown[i]), `Unexpected NaN in down at index ${i} for length ${length}`);
            }
        }
    }
});

test('Aroon batch edge cases', () => {

    const high = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const low = new Float64Array([0.9, 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9, 8.9, 9.9]);


    const singleBatch = wasm.aroon_batch_js(
        high, low,
        5, 5, 1
    );

    assert.strictEqual(singleBatch.up.length, 10);
    assert.strictEqual(singleBatch.down.length, 10);


    const largeBatch = wasm.aroon_batch_js(
        high, low,
        5, 7, 10
    );


    assert.strictEqual(largeBatch.up.length, 10);
    assert.strictEqual(largeBatch.down.length, 10);


    assert.throws(() => {
        wasm.aroon_batch_js(
            new Float64Array([]),
            new Float64Array([]),
            9, 9, 0
        );
    }, /All values are NaN|Empty input data|Invalid length|Not enough valid data/);
});


test('Aroon batch - new ergonomic API with single parameter', () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);

    const result = wasm.aroon_batch(high, low, {
        length_range: [14, 14, 0]
    });


    assert(result.up, 'Should have up array');
    assert(result.down, 'Should have down array');
    assert(result.combos, 'Should have combos array');
    assert(typeof result.rows === 'number', 'Should have rows count');
    assert(typeof result.cols === 'number', 'Should have cols count');


    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, high.length);
    assert.strictEqual(result.combos.length, 1);
    assert.strictEqual(result.up.length, high.length);
    assert.strictEqual(result.down.length, low.length);


    const combo = result.combos[0];
    assert.strictEqual(combo.length, 14);


    const oldResult = wasm.aroon_js(high, low, 14);
    for (let i = 0; i < oldResult.up.length; i++) {
        if (isNaN(oldResult.up[i]) && isNaN(result.up[i])) {
            continue;
        }
        assert(Math.abs(oldResult.up[i] - result.up[i]) < 1e-10,
               `Up value mismatch at index ${i}`);
    }
    for (let i = 0; i < oldResult.down.length; i++) {
        if (isNaN(oldResult.down[i]) && isNaN(result.down[i])) {
            continue;
        }
        assert(Math.abs(oldResult.down[i] - result.down[i]) < 1e-10,
               `Down value mismatch at index ${i}`);
    }
});

test('Aroon batch - new API with multiple parameters', () => {

    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));

    const result = wasm.aroon_batch(high, low, {
        length_range: [10, 14, 2]
    });


    assert.strictEqual(result.rows, 3);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.combos.length, 3);
    assert.strictEqual(result.up.length, 150);
    assert.strictEqual(result.down.length, 150);


    const expectedLengths = [10, 12, 14];

    for (let i = 0; i < expectedLengths.length; i++) {
        assert.strictEqual(result.combos[i].length, expectedLengths[i]);
    }


    const secondRowUp = result.up.slice(result.cols, 2 * result.cols);
    const secondRowDown = result.down.slice(result.cols, 2 * result.cols);
    assert.strictEqual(secondRowUp.length, 50);
    assert.strictEqual(secondRowDown.length, 50);


    const oldResult = wasm.aroon_js(high, low, 10);
    const firstRowUp = result.up.slice(0, result.cols);
    const firstRowDown = result.down.slice(0, result.cols);
    for (let i = 0; i < oldResult.up.length; i++) {
        if (isNaN(oldResult.up[i]) && isNaN(firstRowUp[i])) {
            continue;
        }
        assert(Math.abs(oldResult.up[i] - firstRowUp[i]) < 1e-10,
               `Up value mismatch at index ${i}`);
    }
    for (let i = 0; i < oldResult.down.length; i++) {
        if (isNaN(oldResult.down[i]) && isNaN(firstRowDown[i])) {
            continue;
        }
        assert(Math.abs(oldResult.down[i] - firstRowDown[i]) < 1e-10,
               `Down value mismatch at index ${i}`);
    }
});

test('Aroon batch - new API matches old API results', () => {

    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));

    const params = {
        length_range: [10, 15, 5]
    };


    const oldResult = wasm.aroon_batch_js(
        high, low,
        params.length_range[0], params.length_range[1], params.length_range[2]
    );


    const newResult = wasm.aroon_batch(high, low, params);


    assert.strictEqual(oldResult.up.length, newResult.up.length);
    assert.strictEqual(oldResult.down.length, newResult.down.length);

    for (let i = 0; i < oldResult.up.length; i++) {
        if (isNaN(oldResult.up[i]) && isNaN(newResult.up[i])) {
            continue;
        }
        assert(Math.abs(oldResult.up[i] - newResult.up[i]) < 1e-10,
               `Up value mismatch at index ${i}: old=${oldResult.up[i]}, new=${newResult.up[i]}`);
    }
    for (let i = 0; i < oldResult.down.length; i++) {
        if (isNaN(oldResult.down[i]) && isNaN(newResult.down[i])) {
            continue;
        }
        assert(Math.abs(oldResult.down[i] - newResult.down[i]) < 1e-10,
               `Down value mismatch at index ${i}: old=${oldResult.down[i]}, new=${newResult.down[i]}`);
    }
});

test('Aroon batch - new API error handling', () => {
    const high = new Float64Array(testData.high.slice(0, 10));
    const low = new Float64Array(testData.low.slice(0, 10));


    assert.throws(() => {
        wasm.aroon_batch(high, low, {
            length_range: [9, 9]
        });
    }, /Invalid config/);


    assert.throws(() => {
        wasm.aroon_batch(high, low, {

        });
    }, /Invalid config/);


    assert.throws(() => {
        wasm.aroon_batch(high, low, {
            length_range: "invalid"
        });
    }, /Invalid config/);
});



test.after(() => {
    console.log('Aroon WASM tests completed');
});
