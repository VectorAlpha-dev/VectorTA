
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


function extractMswResults(result) {
    const sine = result.values.slice(0, result.cols);
    const lead = result.values.slice(result.cols);
    return { sine, lead };
}

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

test('MSW partial params', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.msw_js(close, 5);
    assert(result.values, 'Should have values array');
    assert.strictEqual(result.rows, 2, 'Should have 2 rows (sine and lead)');
    assert.strictEqual(result.cols, close.length, 'Columns should match input length');
    assert.strictEqual(result.values.length, 2 * close.length, 'Values should contain sine and lead');


    const sine = result.values.slice(0, close.length);
    const lead = result.values.slice(close.length);
    assert.strictEqual(sine.length, close.length);
    assert.strictEqual(lead.length, close.length);
});

test('MSW accuracy', async () => {

    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.msw;
    const period = expected.defaultParams.period;

    const result = wasm.msw_js(close, period);
    const { sine, lead } = extractMswResults(result);

    assert.strictEqual(sine.length, close.length);
    assert.strictEqual(lead.length, close.length);


    const last5Sine = Array.from(sine.slice(-5));
    const last5Lead = Array.from(lead.slice(-5));

    assertArrayClose(
        last5Sine,
        expected.last5Sine,
        1e-1,
        "MSW sine last 5 values mismatch"
    );
    assertArrayClose(
        last5Lead,
        expected.last5Lead,
        1e-1,
        "MSW lead last 5 values mismatch"
    );
});

test('MSW default candles', () => {

    const close = new Float64Array(testData.close);


    const result = wasm.msw_js(close, 5);
    const { sine, lead } = extractMswResults(result);
    assert.strictEqual(sine.length, close.length);
    assert.strictEqual(lead.length, close.length);
});

test('MSW zero period', () => {

    const data = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.msw_js(data, 0);
    }, /Invalid period/);
});

test('MSW period exceeds length', () => {

    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.msw_js(dataSmall, 10);
    }, /Invalid period/);
});

test('MSW very small dataset', () => {

    const singlePoint = new Float64Array([42.0]);

    assert.throws(() => {
        wasm.msw_js(singlePoint, 5);
    }, /Invalid period|Not enough valid data/);
});

test('MSW empty input', () => {

    const empty = new Float64Array([]);

    assert.throws(() => {
        wasm.msw_js(empty, 5);
    }, /Empty data/);
});

test('MSW NaN handling', () => {

    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.msw;
    const period = expected.defaultParams.period;

    const result = wasm.msw_js(close, period);
    const { sine, lead } = extractMswResults(result);
    assert.strictEqual(sine.length, close.length);
    assert.strictEqual(lead.length, close.length);


    const expectedWarmup = expected.warmupPeriod;
    assertAllNaN(Array.from(sine.slice(0, expectedWarmup)), "Expected NaN in sine warmup period");
    assertAllNaN(Array.from(lead.slice(0, expectedWarmup)), "Expected NaN in lead warmup period");


    if (sine.length > expectedWarmup) {
        const nonNanStart = Math.max(expectedWarmup, 240);
        if (sine.length > nonNanStart) {
            for (let i = nonNanStart; i < sine.length; i++) {
                assert(!isNaN(sine[i]), `Found unexpected NaN in sine at index ${i}`);
                assert(!isNaN(lead[i]), `Found unexpected NaN in lead at index ${i}`);
            }
        }
    }
});

test('MSW all NaN input', () => {

    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);

    assert.throws(() => {
        wasm.msw_js(allNaN, 5);
    }, /All values are NaN/);
});

test('MSW mixed NaN input', () => {

    const mixedData = new Float64Array([NaN, NaN, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0]);
    const period = 3;

    const result = wasm.msw_js(mixedData, period);
    const { sine, lead } = extractMswResults(result);
    assert.strictEqual(sine.length, mixedData.length);
    assert.strictEqual(lead.length, mixedData.length);


    assert(isNaN(sine[0]));
    assert(isNaN(sine[1]));
    assert(isNaN(lead[0]));
    assert(isNaN(lead[1]));



    for (let i = 4; i < sine.length; i++) {
        assert(!isNaN(sine[i]), `Unexpected NaN in sine at index ${i}`);
        assert(!isNaN(lead[i]), `Unexpected NaN in lead at index ${i}`);
    }
});

test('MSW simple predictable pattern', () => {

    const simpleData = new Float64Array([1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5]);
    const period = 5;

    const result = wasm.msw_js(simpleData, period);
    const { sine, lead } = extractMswResults(result);
    assert.strictEqual(sine.length, simpleData.length);
    assert.strictEqual(lead.length, simpleData.length);


    for (let i = 0; i < period - 1; i++) {
        assert(isNaN(sine[i]), `Expected NaN in sine at index ${i}`);
        assert(isNaN(lead[i]), `Expected NaN in lead at index ${i}`);
    }


    for (let i = period - 1; i < sine.length; i++) {
        assert(!isNaN(sine[i]), `Unexpected NaN in sine at index ${i}`);
        assert(!isNaN(lead[i]), `Unexpected NaN in lead at index ${i}`);


        assert(sine[i] >= -1.0 && sine[i] <= 1.0,
               `Sine value ${sine[i]} at index ${i} is out of range [-1, 1]`);
        assert(lead[i] >= -1.0 && lead[i] <= 1.0,
               `Lead value ${lead[i]} at index ${i} is out of range [-1, 1]`);
    }
});












test('MSW SIMD128 consistency', () => {

    const testCases = [
        { size: 10, period: 3 },
        { size: 50, period: 5 },
        { size: 100, period: 10 },
        { size: 500, period: 20 }
    ];

    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = Math.sin(i * 0.1) * Math.cos(i * 0.05) + i * 0.01;
        }

        const result = wasm.msw_js(data, testCase.period);
        const { sine, lead } = extractMswResults(result);


        assert.strictEqual(sine.length, data.length);
        assert.strictEqual(lead.length, data.length);


        for (let i = 0; i < testCase.period - 1; i++) {
            assert(isNaN(sine[i]), `Expected NaN at sine warmup index ${i} for size=${testCase.size}`);
            assert(isNaN(lead[i]), `Expected NaN at lead warmup index ${i} for size=${testCase.size}`);
        }


        let sineSum = 0;
        let leadSum = 0;
        let count = 0;
        for (let i = testCase.period - 1; i < sine.length; i++) {
            assert(!isNaN(sine[i]), `Unexpected NaN in sine at index ${i} for size=${testCase.size}`);
            assert(!isNaN(lead[i]), `Unexpected NaN in lead at index ${i} for size=${testCase.size}`);


            assert(sine[i] >= -1.0 && sine[i] <= 1.0,
                   `Sine value ${sine[i]} out of range at index ${i}`);
            assert(lead[i] >= -1.0 && lead[i] <= 1.0,
                   `Lead value ${lead[i]} out of range at index ${i}`);

            sineSum += sine[i];
            leadSum += lead[i];
            count++;
        }


        const avgSine = sineSum / count;
        const avgLead = leadSum / count;
        assert(Math.abs(avgSine) < 1.0, `Average sine ${avgSine} seems unreasonable`);
        assert(Math.abs(avgLead) < 1.0, `Average lead ${avgLead} seems unreasonable`);
    }
});

test.after(() => {
    console.log('MSW WASM tests completed');
});