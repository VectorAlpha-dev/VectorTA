
import test from 'node:test';
import assert from 'node:assert';
import { loadTestData, EXPECTED_OUTPUTS, assertArrayClose, assertClose } from './test_utils.js';
import { compareWithRust } from './rust-comparison.js';


import * as wasm from '../../pkg/vector_ta.js';

test.describe('MAC-Z WASM Binding Tests', () => {
    let testData;

    test.before(() => {
        testData = loadTestData();
    });

    test('MAC-Z partial params - check_macz_partial_params', () => {
        const close = testData.close;
        const volume = testData.volume || null;


        const result = wasm.macz_js(
            close,
            20,
            30,
            10,
            20,
            20,
            2.0,
            -1.0,
            false,
            0.02
        );

        assert.strictEqual(result.length, close.length, "Result length should match input");
    });

    test('MAC-Z accuracy - check_macz_accuracy', () => {
        const close = testData.close;
        const params = EXPECTED_OUTPUTS.macz.defaultParams;
        const expected = EXPECTED_OUTPUTS.macz;

        const result = wasm.macz_js(
            close,
            params.fast_length,
            params.slow_length,
            params.signal_length,
            params.lengthz,
            params.length_stdev,
            params.a,
            params.b,
            params.use_lag,
            params.gamma
        );

        assert.strictEqual(result.length, close.length, "Result length should match input");


        const last5 = result.slice(-5);
        assertArrayClose(last5, expected.last5Values, 1e-9, "MAC-Z last 5 values mismatch");
    });

    test('MAC-Z default candles - check_macz_default_candles', () => {
        const close = testData.close;


        const result = wasm.macz_js(
            close,
            20,
            30,
            10,
            20,
            20,
            2.0,
            -1.0,
            false,
            0.02
        );

        assert.strictEqual(result.length, close.length, "Result length should match input");
    });

    test('MAC-Z zero fast_length error - check_macz_zero_fast_length', () => {
        const inputData = new Float64Array([10.0, 20.0, 30.0]);

        assert.throws(
            () => wasm.macz_js(
                inputData,
                0,
                30,
                10,
                20,
                20,
                2.0,
                -1.0,
                false,
                0.02
            ),
            /Invalid period/i,
            "Should throw error for zero fast_length"
        );
    });

    test('MAC-Z period exceeds length error - check_macz_period_exceeds_length', () => {
        const smallData = new Float64Array([10.0, 20.0, 30.0]);

        assert.throws(
            () => wasm.macz_js(
                smallData,
                20,
                100,
                10,
                20,
                20,
                2.0,
                -1.0,
                false,
                0.02
            ),
            /Not enough valid data/i,
            "Should throw error when period exceeds data length"
        );
    });

    test('MAC-Z very small dataset error - check_macz_very_small_dataset', () => {
        const singlePoint = new Float64Array([42.0]);

        assert.throws(
            () => wasm.macz_js(
                singlePoint,
                20,
                30,
                10,
                20,
                20,
                2.0,
                -1.0,
                false,
                0.02
            ),
            /Not enough valid data/i,
            "Should throw error for insufficient data"
        );
    });

    test('MAC-Z empty input error - check_macz_empty_input', () => {
        const empty = new Float64Array([]);

        assert.throws(
            () => wasm.macz_js(
                empty,
                20,
                30,
                10,
                20,
                20,
                2.0,
                -1.0,
                false,
                0.02
            ),
            /Input data slice is empty/i,
            "Should throw error for empty input"
        );
    });

    test('MAC-Z invalid A error - check_macz_invalid_a', () => {
        const data = new Float64Array(Array(60).fill(0).map((_, i) => i % 3 + 1.0));


        assert.throws(
            () => wasm.macz_js(
                data,
                20,
                30,
                10,
                20,
                20,
                3.0,
                -1.0,
                false,
                0.02
            ),
            /A out of range/i,
            "Should throw error for A > 2.0"
        );


        assert.throws(
            () => wasm.macz_js(
                data,
                20,
                30,
                10,
                20,
                20,
                -3.0,
                -1.0,
                false,
                0.02
            ),
            /A out of range/i,
            "Should throw error for A < -2.0"
        );
    });

    test('MAC-Z invalid B error - check_macz_invalid_b', () => {
        const data = new Float64Array(Array(60).fill(0).map((_, i) => i % 3 + 1.0));


        assert.throws(
            () => wasm.macz_js(
                data,
                20,
                30,
                10,
                20,
                20,
                2.0,
                3.0,
                false,
                0.02
            ),
            /B out of range/i,
            "Should throw error for B > 2.0"
        );


        assert.throws(
            () => wasm.macz_js(
                data,
                20,
                30,
                10,
                20,
                20,
                2.0,
                -3.0,
                false,
                0.02
            ),
            /B out of range/i,
            "Should throw error for B < -2.0"
        );
    });

    test('MAC-Z invalid gamma error - check_macz_invalid_gamma', () => {
        const data = new Float64Array(Array(60).fill(0).map((_, i) => i % 3 + 1.0));


        assert.throws(
            () => wasm.macz_js(
                data,
                20,
                30,
                10,
                20,
                20,
                2.0,
                -1.0,
                false,
                1.5
            ),
            /Invalid gamma/i,
            "Should throw error for gamma >= 1.0"
        );


        assert.throws(
            () => wasm.macz_js(
                data,
                20,
                30,
                10,
                20,
                20,
                2.0,
                -1.0,
                false,
                -0.1
            ),
            /Invalid gamma/i,
            "Should throw error for gamma < 0.0"
        );
    });

    test('MAC-Z NaN handling - check_macz_nan_handling', () => {
        const close = testData.close;
        const params = EXPECTED_OUTPUTS.macz.defaultParams;
        const expected = EXPECTED_OUTPUTS.macz;

        const result = wasm.macz_js(
            close,
            params.fast_length,
            params.slow_length,
            params.signal_length,
            params.lengthz,
            params.length_stdev,
            params.a,
            params.b,
            params.use_lag,
            params.gamma
        );

        assert.strictEqual(result.length, close.length, "Result length should match input");


        const warmup = expected.warmupPeriod;
        if (result.length > warmup) {
            for (let i = warmup; i < result.length; i++) {
                assert.ok(!isNaN(result[i]), `Found unexpected NaN at index ${i} after warmup period`);
            }
        }


        for (let i = 0; i < warmup; i++) {
            assert.ok(isNaN(result[i]), `Expected NaN at index ${i} in warmup period`);
        }
    });

    test('MAC-Z all NaN input error', () => {
        const allNan = new Float64Array(100).fill(NaN);

        assert.throws(
            () => wasm.macz_js(
                allNan,
                20,
                30,
                10,
                20,
                20,
                2.0,
                -1.0,
                false,
                0.02
            ),
            /All values are NaN/i,
            "Should throw error for all NaN values"
        );
    });

    test('MAC-Z batch processing - check_batch_default_row', async () => {
        const close = testData.close;
        const expected = EXPECTED_OUTPUTS.macz;
        const volume = testData.volume;


        const batchResult = wasm.macz_batch(
            close,
            volume,
            [12, 12, 0],
            [25, 25, 0],
            [9, 9, 0],
            [20, 20, 0],
            [25, 25, 0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [false, false, false],
            [0.02, 0.02, 0.0]
        );


        assert.ok(batchResult.values, "Should have values array");
        assert.ok(batchResult.fast_lengths, "Should have fast_lengths array");
        assert.ok(batchResult.slow_lengths, "Should have slow_lengths array");
        assert.ok(batchResult.signal_lengths, "Should have signal_lengths array");


        assert.strictEqual(batchResult.values.length, 1, "Should have 1 parameter combination");
        assert.strictEqual(batchResult.values[0].length, close.length, "Result length should match input");


        const defaultRow = batchResult.values[0];
        await compareWithRust('macz', defaultRow, 'close', expected.defaultParams, 1e-8);
    });

    test('MAC-Z zero-copy API', () => {
        const close = testData.close;
        const params = EXPECTED_OUTPUTS.macz.defaultParams;


        const outputPtr = wasm.allocate_f64_array(close.length);

        try {

            wasm.macz(
                close,
                outputPtr,
                params.fast_length,
                params.slow_length,
                params.signal_length,
                params.lengthz,
                params.length_stdev,
                params.a,
                params.b,
                params.use_lag,
                params.gamma
            );


            const result = wasm.read_f64_array(outputPtr, close.length);

            assert.strictEqual(result.length, close.length, "Result length should match input");


            const last5 = result.slice(-5);
            assertArrayClose(
                last5,
                EXPECTED_OUTPUTS.macz.last5Values,
                1e-9,
                "MAC-Z zero-copy last 5 values"
            );
        } finally {

            wasm.deallocate_f64_array(outputPtr);
        }
    });

    test('MAC-Z batch zero-copy API', () => {
        const close = testData.close;
        const numCombinations = 1;


        const outputPtr = wasm.allocate_f64_matrix(numCombinations, close.length);

        try {

            const numResults = wasm.macz_batch_zero_copy(
                close,
                null,
                outputPtr,
                [12, 12, 0],
                [25, 25, 0],
                [9, 9, 0],
                [20, 20, 0],
                [25, 25, 0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [false, false, false],
                [0.02, 0.02, 0.0]
            );

            assert.strictEqual(numResults, numCombinations, "Should return correct number of combinations");


            const result = wasm.read_f64_matrix(outputPtr, numCombinations, close.length);


            assert.strictEqual(result.length, numCombinations, "Should have correct number of rows");
            assert.strictEqual(result[0].length, close.length, "Each row should match input length");


            const last5 = result[0].slice(-5);
            assertArrayClose(
                last5,
                EXPECTED_OUTPUTS.macz.last5Values,
                1e-9,
                "MAC-Z batch zero-copy last 5 values"
            );
        } finally {

            wasm.deallocate_f64_matrix(outputPtr);
        }
    });

    test('MAC-Z with volume data', () => {
        const close = testData.close;
        const volume = testData.volume || new Float64Array(close.length).fill(1000.0);
        const params = EXPECTED_OUTPUTS.macz.defaultParams;



        const result = wasm.macz_js(
            close,
            params.fast_length,
            params.slow_length,
            params.signal_length,
            params.lengthz,
            params.length_stdev,
            params.a,
            params.b,
            params.use_lag,
            params.gamma
        );

        assert.strictEqual(result.length, close.length, "Result length should match input");


        const hasValidValues = result.some(v => !isNaN(v));
        assert.ok(hasValidValues, "Result should not be all NaN");
    });
});
