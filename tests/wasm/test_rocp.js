
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

test('ROCP Safe API accuracy', () => {
    const period = 10;
    const result = wasm.rocp_js(testData.close, period);

    assert.strictEqual(result.length, testData.close.length);


    const expectedLastFive = [
        -0.0022551709049293996,
        -0.005561903481650759,
        -0.003275201323586514,
        -0.004945415398072297,
        -0.015045927020537019,
    ];

    const startIdx = result.length - 5;
    assertArrayClose(
        result.slice(startIdx),
        expectedLastFive,
        1e-9,
        'ROCP last 5 values mismatch'
    );
});

test('ROCP Fast API with aliasing', () => {
    const period = 10;
    const len = testData.close.length;


    const ptr = wasm.rocp_alloc(len);

    try {

        const memory = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        memory.set(testData.close);


        wasm.rocp_into(ptr, ptr, len, period);


        const memory2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);


        const result = [...memory2];


        const safeResult = wasm.rocp_js(testData.close, period);
        assertArrayClose(
            result,
            safeResult,
            1e-9,
            'Fast API with aliasing mismatch'
        );
    } finally {
        wasm.rocp_free(ptr, len);
    }
});

test('ROCP Fast API without aliasing', () => {
    const period = 10;
    const len = testData.close.length;


    const inPtr = wasm.rocp_alloc(len);
    const outPtr = wasm.rocp_alloc(len);

    try {

        const inMemory = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        const outMemory = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);


        inMemory.set(testData.close);


        wasm.rocp_into(inPtr, outPtr, len, period);


        const outMemory2 = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);


        const result = [...outMemory2];


        const safeResult = wasm.rocp_js(testData.close, period);
        assertArrayClose(
            result,
            safeResult,
            1e-9,
            'Fast API without aliasing mismatch'
        );
    } finally {
        wasm.rocp_free(inPtr, len);
        wasm.rocp_free(outPtr, len);
    }
});

test('ROCP error handling - zero period', () => {
    const data = [10.0, 20.0, 30.0];

    assert.throws(
        () => wasm.rocp_js(data, 0),
        /Invalid period/,
        'Should fail with zero period'
    );
});

test('ROCP error handling - period exceeds length', () => {
    const data = [10.0, 20.0, 30.0];

    assert.throws(
        () => wasm.rocp_js(data, 10),
        /Invalid period/,
        'Should fail when period exceeds data length'
    );
});

test('ROCP error handling - empty input', () => {
    const data = [];

    assert.throws(
        () => wasm.rocp_js(data, 10),
        /Input data slice is empty|Invalid period|All values are NaN/,
        'Should fail with empty input'
    );
});

test('ROCP Batch API - small config', () => {
    const config = {
        period_range: [9, 15, 3]
    };

    const result = wasm.rocp_batch(testData.close, config);

    assert(result.values, 'Batch result should have values');
    assert(result.combos, 'Batch result should have combos');
    assert.strictEqual(result.combos.length, 3, 'Should have 3 combinations');
    assert.strictEqual(result.rows, 3);
    assert.strictEqual(result.cols, testData.close.length);


    const periods = result.combos.map(c => c.period);
    assert.deepStrictEqual(periods, [9, 12, 15]);


    const firstRow = result.values.slice(0, result.cols);
    const singleResult = wasm.rocp_js(testData.close, 9);
    assertArrayClose(firstRow, singleResult, 1e-9, 'Batch first row mismatch');
});

test('ROCP Batch Fast API', () => {
    const periodStart = 5;
    const periodEnd = 15;
    const periodStep = 5;
    const len = testData.close.length;


    const expectedRows = Math.floor((periodEnd - periodStart) / periodStep) + 1;
    const totalSize = expectedRows * len;


    const inPtr = wasm.rocp_alloc(len);
    const outPtr = wasm.rocp_alloc(totalSize);

    try {

        const inMemory = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inMemory.set(testData.close);


        const rows = wasm.rocp_batch_into(
            inPtr,
            outPtr,
            len,
            periodStart,
            periodEnd,
            periodStep
        );

        assert.strictEqual(rows, expectedRows, 'Batch rows mismatch');


        const outMemory = new Float64Array(wasm.__wasm.memory.buffer, outPtr, totalSize);


        const firstRow = outMemory.slice(0, len);
        const singleResult = wasm.rocp_js(testData.close, periodStart);
        assertArrayClose(
            Array.from(firstRow),
            singleResult,
            1e-9,
            'Batch fast API first row mismatch'
        );
    } finally {
        wasm.rocp_free(inPtr, len);
        wasm.rocp_free(outPtr, totalSize);
    }
});

test('ROCP NaN handling', () => {
    const period = 9;
    const result = wasm.rocp_js(testData.close, period);


    for (let i = 0; i < period; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} during warmup`);
    }


    const validStartIdx = Math.max(240, period);
    if (result.length > validStartIdx) {
        for (let i = validStartIdx; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i}`);
        }
    }
});

test.after(() => {
    console.log('ROCP WASM tests completed');
});
