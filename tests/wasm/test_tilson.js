import test from 'node:test';
import assert from 'assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
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
const __dirname = dirname(__filename);

let wasm;
let testData;

test.before(async () => {

    try {
        const wasmPath = join(__dirname, '../../pkg/vector_ta.js');
        const importPath = process.platform === 'win32'
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
        wasm = await import(importPath);
    } catch (e2) {
        try {
            const { createRequire } = await import('node:module');
            const require = createRequire(import.meta.url);
            wasm = require(join(__dirname, '../../pkg/vector_ta.js'));
        } catch {
            console.error('Failed to load WASM module. Run "wasm-pack build --target nodejs --out-name vector_ta --features wasm" first');
            throw e2;
        }
    }

    testData = loadTestData();
});

test.describe('Tilson T3 Moving Average', () => {
    test('Tilson accuracy', async () => {

        const close = new Float64Array(testData.close);
        const expected = EXPECTED_OUTPUTS.tilson;

        const result = wasm.tilson_js(
            close,
            expected.defaultParams.period,
            expected.defaultParams.volume_factor
        );

        assert.strictEqual(result.length, close.length);


        const last5 = result.slice(-5);
        assertArrayClose(
            last5,
            expected.last5Values,
            1e-8,
            "Tilson last 5 values mismatch"
        );




        await compareWithRust('tilson', result, 'close', expected.defaultParams, 1e-13);
    });

    test('Tilson default params', () => {

        const close = new Float64Array(testData.close);

        const result = wasm.tilson_js(close, 5, 0.0);
        assert.strictEqual(result.length, close.length);


        const warmup = 6 * (5 - 1);
        for (let i = 0; i < warmup; i++) {
            assert(isNaN(result[i]), `Index ${i} should be NaN during warmup`);
        }
        if (result.length > warmup) {
            for (let i = warmup; i < result.length; i++) {
                assert(!isNaN(result[i]), `Index ${i} should not be NaN after warmup`);
            }
        }
    });

    test('basic functionality', () => {
        const data = new Float64Array([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30
        ]);
        const period = 5;
        const volumeFactor = 0.0;

        const result = wasm.tilson_js(data, period, volumeFactor);

        assert(result instanceof Float64Array, 'Result should be Float64Array');
        assert.strictEqual(result.length, data.length, 'Result length should match input');


        for (let i = 0; i < 24; i++) {
            assert(isNaN(result[i]), `Index ${i} should be NaN during warmup`);
        }


        for (let i = 24; i < result.length; i++) {
            assert(!isNaN(result[i]), `Index ${i} should not be NaN after warmup`);
        }
    });

    test('error handling - empty input', () => {
        const data = new Float64Array(0);
        const period = 5;
        const volumeFactor = 0.0;

        assert.throws(
            () => wasm.tilson_js(data, period, volumeFactor),
            /Input data slice is empty/,
            'Should throw error for empty input'
        );
    });

    test('error handling - all NaN values', () => {
        const data = new Float64Array(10).fill(NaN);
        const period = 5;
        const volumeFactor = 0.0;

        assert.throws(
            () => wasm.tilson_js(data, period, volumeFactor),
            /All values are NaN/,
            'Should throw error for all NaN values'
        );
    });

    test('error handling - invalid period', () => {
        const data = new Float64Array([1, 2, 3, 4, 5]);


        assert.throws(
            () => wasm.tilson_js(data, 6, 0.0),
            /Invalid period/,
            'Should throw error when period exceeds data length'
        );


        assert.throws(
            () => wasm.tilson_js(data, 0, 0.0),
            /Invalid period/,
            'Should throw error for zero period'
        );
    });

    test('error handling - not enough valid data', () => {

        const data = new Float64Array([NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 1, 2]);
        const period = 5;
        const volumeFactor = 0.0;

        assert.throws(
            () => wasm.tilson_js(data, period, volumeFactor),
            /Not enough valid data/,
            'Should throw error when not enough valid data after NaN values'
        );
    });

    test('error handling - invalid volume factor', () => {
        const data = new Float64Array(100);
        for (let i = 0; i < 100; i++) {
            data[i] = Math.random() * 100;
        }


        assert.throws(
            () => wasm.tilson_js(data, 5, NaN),
            /Invalid volume factor/,
            'Should throw error for NaN volume factor'
        );


        assert.throws(
            () => wasm.tilson_js(data, 5, Infinity),
            /Invalid volume factor/,
            'Should throw error for infinite volume factor'
        );
    });

    test('leading NaN values', () => {
        const data = new Float64Array([
            NaN, NaN, 1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26
        ]);
        const period = 3;
        const volumeFactor = 0.5;

        const result = wasm.tilson_js(data, period, volumeFactor);



        for (let i = 0; i < 14; i++) {
            assert(isNaN(result[i]), `Index ${i} should be NaN`);
        }
        for (let i = 14; i < result.length; i++) {
            assert(!isNaN(result[i]), `Index ${i} should not be NaN`);
        }
    });

    test('Tilson zero period', () => {

        const inputData = new Float64Array([10.0, 20.0, 30.0]);

        assert.throws(() => {
            wasm.tilson_js(inputData, 0, 0.0);
        }, /Invalid period/);
    });

    test('Tilson period exceeds length', () => {

        const dataSmall = new Float64Array([10.0, 20.0, 30.0]);

        assert.throws(() => {
            wasm.tilson_js(dataSmall, 10, 0.0);
        }, /Invalid period/);
    });

    test('Tilson very small dataset', () => {

        const singlePoint = new Float64Array([42.0]);

        assert.throws(() => {
            wasm.tilson_js(singlePoint, 5, 0.0);
        }, /Invalid period|Not enough valid data/);
    });

    test('Tilson NaN handling', () => {

        const close = new Float64Array(testData.close);

        const result = wasm.tilson_js(close, 5, 0.0);
        assert.strictEqual(result.length, close.length);


        if (result.length > 240) {
            for (let i = 240; i < result.length; i++) {
                assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
            }
        }


        assertAllNaN(result.slice(0, 24), "Expected NaN in warmup period");
    });

    test('Tilson all NaN input', () => {

        const allNaN = new Float64Array(100);
        allNaN.fill(NaN);

        assert.throws(() => {
            wasm.tilson_js(allNaN, 5, 0.0);
        }, /All values are NaN/);
    });

    test('batch calculation', () => {
        const data = new Float64Array(100);
        for (let i = 0; i < 100; i++) {
            data[i] = Math.random() * 100;
        }

        const periodStart = 3;
        const periodEnd = 7;
        const periodStep = 2;
        const vFactorStart = 0.0;
        const vFactorEnd = 0.6;
        const vFactorStep = 0.3;

        const values = wasm.tilson_batch_js(data, periodStart, periodEnd, periodStep,
                                           vFactorStart, vFactorEnd, vFactorStep);
        const metadata = wasm.tilson_batch_metadata_js(periodStart, periodEnd, periodStep,
                                                       vFactorStart, vFactorEnd, vFactorStep);


        const expectedPeriods = [3, 5, 7];
        const expectedVFactors = [0.0, 0.3, 0.6];
        const expectedCombos = expectedPeriods.length * expectedVFactors.length;

        const rows = expectedCombos;
        const cols = data.length;

        assert.strictEqual(metadata.length, rows * 2, 'Metadata should contain periods and volume factors');
        assert.strictEqual(values.length, rows * cols, 'Values array size should be rows*cols');


        const periods = new Float64Array(metadata.slice(0, rows));
        const vFactors = new Float64Array(metadata.slice(rows));


        let comboIdx = 0;
        for (const period of expectedPeriods) {
            for (const vFactor of expectedVFactors) {
                assert.strictEqual(periods[comboIdx], period, `Period mismatch at ${comboIdx}`);
                assert(Math.abs(vFactors[comboIdx] - vFactor) < 1e-10, `Volume factor mismatch at ${comboIdx}`);
                comboIdx++;
            }
        }
    });

    test('warmup period validation', () => {
        const data = new Float64Array(100);
        for (let i = 0; i < 100; i++) {
            data[i] = i + 1;
        }


        const periods = [2, 3, 4, 5, 6];
        for (const period of periods) {
            const result = wasm.tilson_js(data, period, 0.0);
            const warmup = 6 * (period - 1);


            for (let i = 0; i < warmup && i < result.length; i++) {
                assert(isNaN(result[i]), `Period ${period}: Index ${i} should be NaN`);
            }


            for (let i = warmup; i < result.length; i++) {
                assert(!isNaN(result[i]), `Period ${period}: Index ${i} should not be NaN`);
            }
        }
    });

    test('edge cases', () => {
        const data = new Float64Array(100);
        for (let i = 0; i < 100; i++) {
            data[i] = Math.random() * 100;
        }


        const result1 = wasm.tilson_js(data, 1, 0.0);

        assert(!isNaN(result1[0]), 'Period 1 should have no NaN values');


        const result2 = wasm.tilson_js(data, 5, 1.0);
        const warmup = 6 * (5 - 1);
        for (let i = 0; i < warmup; i++) {
            assert(isNaN(result2[i]), `Index ${i} should be NaN`);
        }
        for (let i = warmup; i < result2.length; i++) {
            assert(!isNaN(result2[i]), `Index ${i} should not be NaN`);
        }
    });

    test('Tilson reinput', () => {

        const close = new Float64Array(testData.close);
        const expected = EXPECTED_OUTPUTS.tilson;


        const firstResult = wasm.tilson_js(close, 5, 0.0);
        assert.strictEqual(firstResult.length, close.length);


        const secondResult = wasm.tilson_js(firstResult, 3, 0.7);
        assert.strictEqual(secondResult.length, firstResult.length);


        const last5 = secondResult.slice(-5);
        assertArrayClose(
            last5,
            expected.reinputLast5,
            1e-8,
            "Tilson re-input last 5 values mismatch"
        );
    });

    test('Tilson batch single parameter set', () => {

        const close = new Float64Array(testData.close);


        const batchResult = wasm.tilson_batch(close, {
            period_range: [5, 5, 0],
            volume_factor_range: [0.0, 0.0, 0]
        });


        const singleResult = wasm.tilson_js(close, 5, 0.0);

        assert.strictEqual(batchResult.values.length, singleResult.length);
        assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
    });

    test('batch with single combination', () => {
        const data = new Float64Array(50);
        for (let i = 0; i < 50; i++) {
            data[i] = Math.random() * 100;
        }


        const values = wasm.tilson_batch_js(data, 5, 5, 0, 0.7, 0.7, 0.0);
        const metadata = wasm.tilson_batch_metadata_js(5, 5, 0, 0.7, 0.7, 0.0);

        assert.strictEqual(values.length, data.length, 'Single batch should have data length');
        assert.strictEqual(metadata.length, 2, 'Metadata should have 2 values (period and volume factor)');
        assert.strictEqual(metadata[0], 5, 'Period should be 5');
        assert(Math.abs(metadata[1] - 0.7) < 1e-10, 'Volume factor should be 0.7');


        const singleResult = wasm.tilson_js(data, 5, 0.7);
        assertArrayClose(values, singleResult, 1e-10, 'Batch with single combo should match single calc');
    });

    test('volume factor effect', () => {
        const data = new Float64Array(100);
        for (let i = 0; i < 100; i++) {
            data[i] = Math.sin(i * 0.1) * 50 + 50 + Math.random() * 5;
        }
        const period = 5;


        const result0 = wasm.tilson_js(data, period, 0.0);
        const result5 = wasm.tilson_js(data, period, 0.5);
        const result9 = wasm.tilson_js(data, period, 0.9);

        const warmup = 6 * (period - 1);


        let differences0_5 = 0;
        let differences5_9 = 0;

        for (let i = warmup; i < data.length; i++) {
            if (Math.abs(result0[i] - result5[i]) > 1e-10) differences0_5++;
            if (Math.abs(result5[i] - result9[i]) > 1e-10) differences5_9++;
        }

        assert(differences0_5 > 0, 'Volume factor 0.0 vs 0.5 should produce different results');
        assert(differences5_9 > 0, 'Volume factor 0.5 vs 0.9 should produce different results');
    });

    test('Tilson batch multiple periods', () => {

        const close = new Float64Array(testData.close.slice(0, 100));


        const batchResult = wasm.tilson_batch(close, {
            period_range: [3, 7, 2],
            volume_factor_range: [0.0, 0.6, 0.3]
        });


        assert.strictEqual(batchResult.rows, 9);
        assert.strictEqual(batchResult.cols, 100);
        assert.strictEqual(batchResult.values.length, 9 * 100);


        const periods = [3, 5, 7];
        const vFactors = [0.0, 0.3, 0.6];
        let comboIdx = 0;
        for (const period of periods) {
            for (const vFactor of vFactors) {
                const rowStart = comboIdx * 100;
                const rowEnd = rowStart + 100;
                const rowData = batchResult.values.slice(rowStart, rowEnd);

                const singleResult = wasm.tilson_js(close, period, vFactor);


                let validCount = 0;
                for (let i = 0; i < 100; i++) {
                    if (!isNaN(rowData[i]) && !isNaN(singleResult[i])) {
                        validCount++;
                        assert(Math.abs(rowData[i] - singleResult[i]) < 1e-10,
                            `Mismatch at index ${i} for period=${period}, v_factor=${vFactor}`);
                    }
                }
                assert(validCount > 0, `No valid values to compare for period=${period}, v_factor=${vFactor}`);
                comboIdx++;
            }
        }
    });

    test('unified batch API with config object', () => {
        const data = new Float64Array(100);
        for (let i = 0; i < 100; i++) {
            data[i] = Math.random() * 100;
        }

        const config = {
            period_range: [3, 7, 2],
            volume_factor_range: [0.0, 0.6, 0.3]
        };

        const result = wasm.tilson_batch(data, config);


        const expectedPeriods = [3, 5, 7];
        const expectedVFactors = [0.0, 0.3, 0.6];
        const expectedCombos = expectedPeriods.length * expectedVFactors.length;

        assert.strictEqual(result.combos.length, expectedCombos, 'Should have correct number of combinations');
        assert.strictEqual(result.rows, expectedCombos, 'Rows should match combinations');
        assert.strictEqual(result.cols, data.length, 'Cols should match data length');
        assert.strictEqual(result.values.length, expectedCombos * data.length, 'Values array size should be rows*cols');


        let comboIdx = 0;
        for (const period of expectedPeriods) {
            for (const vFactor of expectedVFactors) {
                assert.strictEqual(result.combos[comboIdx].period, period, `Period mismatch at ${comboIdx}`);
                assert(Math.abs(result.combos[comboIdx].volume_factor - vFactor) < 1e-10, `Volume factor mismatch at ${comboIdx}`);
                comboIdx++;
            }
        }
    });

    test('zero-copy API', () => {
        const data = new Float64Array([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30
        ]);
        const period = 5;
        const volumeFactor = 0.0;


        const ptr = wasm.tilson_alloc(data.length);
        assert(ptr !== 0, 'Failed to allocate memory');


        const memView = new Float64Array(
            wasm.__wasm.memory.buffer,
            ptr,
            data.length
        );


        memView.set(data);


        try {
            wasm.tilson_into(ptr, ptr, data.length, period, volumeFactor);


            const regularResult = wasm.tilson_js(data, period, volumeFactor);
            for (let i = 0; i < data.length; i++) {
                if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                    continue;
                }
                assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                       `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
            }
        } finally {

            wasm.tilson_free(ptr, data.length);
        }
    });

    test('zero-copy batch API', () => {
        const data = new Float64Array(50);
        for (let i = 0; i < 50; i++) {
            data[i] = Math.random() * 100;
        }

        const periodStart = 3;
        const periodEnd = 5;
        const periodStep = 2;
        const vFactorStart = 0.0;
        const vFactorEnd = 0.6;
        const vFactorStep = 0.6;


        const expectedRows = 4;
        const totalSize = expectedRows * data.length;


        const inPtr = wasm.tilson_alloc(data.length);
        const outPtr = wasm.tilson_alloc(totalSize);

        assert(inPtr !== 0, 'Failed to allocate input buffer');
        assert(outPtr !== 0, 'Failed to allocate output buffer');

        try {

            const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, data.length);
            inView.set(data);


            const rows = wasm.tilson_batch_into(
                inPtr, outPtr, data.length,
                periodStart, periodEnd, periodStep,
                vFactorStart, vFactorEnd, vFactorStep
            );

            assert.strictEqual(rows, expectedRows, 'Should return correct number of rows');


            const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, totalSize);
            let hasNonNaN = false;
            for (let i = 0; i < totalSize; i++) {
                if (!isNaN(outView[i])) {
                    hasNonNaN = true;
                    break;
                }
            }
            assert(hasNonNaN, 'Output should contain some non-NaN values');
        } finally {
            wasm.tilson_free(inPtr, data.length);
            wasm.tilson_free(outPtr, totalSize);
        }
    });

    test('deprecated TilsonContext API', () => {
        const period = 5;
        const volumeFactor = 0.7;


        const context = new wasm.TilsonContext(period, volumeFactor);


        assert.strictEqual(context.get_warmup_period(), 6 * (period - 1), 'Warmup period should be 6*(period-1)');


        const testData = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30];
        const results = [];

        for (let i = 0; i < testData.length; i++) {
            const result = context.update(testData[i]);
            results.push(result === undefined ? null : result);
        }


        const warmup = 6 * (period - 1);
        for (let i = 0; i < warmup && i < results.length; i++) {
            assert(results[i] === null, `Should return null during warmup at index ${i}`);
        }


        for (let i = warmup; i < results.length; i++) {
            assert(results[i] !== null && !isNaN(results[i]), `Should return valid value after warmup at index ${i}`);
        }


        context.reset();
        const afterReset = context.update(10.0);
        assert(afterReset === null || afterReset === undefined, 'Should return null/undefined after reset');


        assert.throws(() => {
            new wasm.TilsonContext(0, 0.5);
        }, /Invalid period/, 'Should throw for invalid period');

        assert.throws(() => {
            new wasm.TilsonContext(5, NaN);
        }, /Invalid volume factor/, 'Should throw for NaN volume factor');
    });
});


if (import.meta.url === `file://${process.argv[1]}`) {
    console.log('Running Tilson T3 tests...');
}
