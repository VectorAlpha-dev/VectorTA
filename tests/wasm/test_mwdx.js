/**
 * WASM binding tests for MWDX indicator.
 * These tests mirror the Rust unit tests to ensure WASM bindings work correctly.
 */
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

test('MWDX partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.mwdx_js(close, 0.2);
    assert(result instanceof Float64Array);
    assert.strictEqual(result.length, close.length);
});

test('MWDX accuracy', async () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.mwdx;
    
    const result = wasm.mwdx_js(
        close,
        expected.defaultParams.factor
    );
    
    assert.strictEqual(result.length, close.length);
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-7,
        "MWDX last 5 values mismatch"
    );
    
    
    await compareWithRust('mwdx', result, 'close', expected.defaultParams);
});

test('MWDX zero factor', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.mwdx_js(inputData, 0.0);
    }, /Factor must be greater than 0/);
});

test('MWDX negative factor', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.mwdx_js(inputData, -0.5);
    }, /Factor must be greater than 0/);
});

test('MWDX NaN factor', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.mwdx_js(inputData, NaN);
    }, /Factor must be greater than 0/);
});

test('MWDX very small dataset', () => {
    
    const data = new Float64Array([42.0]);
    
    const result = wasm.mwdx_js(data, 0.2);
    assert.strictEqual(result.length, 1);
    assert.strictEqual(result[0], 42.0);
});

test('MWDX empty input', () => {
    
    const dataEmpty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.mwdx_js(dataEmpty, 0.2);
    }, /No input data was provided/);
});

test('MWDX reinput', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const firstResult = wasm.mwdx_js(close, 0.2);
    
    
    const secondResult = wasm.mwdx_js(firstResult, 0.3);
    
    assert.strictEqual(secondResult.length, firstResult.length);
    
    
    for (let i = 0; i < secondResult.length; i++) {
        assert(isFinite(secondResult[i]), `NaN found at index ${i}`);
    }
});

test('MWDX NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.mwdx_js(close, 0.2);
    
    assert.strictEqual(result.length, close.length);
    
    
    for (let i = 0; i < result.length; i++) {
        assert(isFinite(result[i]), `NaN found at index ${i}`);
    }
});

test('MWDX batch', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const batch_result = wasm.mwdx_batch_js(
        close, 
        0.1, 0.5, 0.1    
    );
    
    
    const rows_cols = wasm.mwdx_batch_rows_cols_js(0.1, 0.5, 0.1, close.length);
    const rows = rows_cols[0];
    const cols = rows_cols[1];
    
    assert(batch_result instanceof Float64Array);
    assert.strictEqual(rows, 5); 
    assert.strictEqual(cols, close.length);
    assert.strictEqual(batch_result.length, rows * cols);
    
    
    const individual_result = wasm.mwdx_js(close, 0.1);
    const batch_first = batch_result.slice(0, close.length);
    
    assertArrayClose(batch_first, individual_result, 1e-9, 'MWDX first combination');
});

test('MWDX different factors', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const testFactors = [0.1, 0.2, 0.5, 0.9];
    
    for (const factor of testFactors) {
        const result = wasm.mwdx_js(close, factor);
        assert.strictEqual(result.length, close.length);
        
        
        let finiteCount = 0;
        for (let i = 0; i < result.length; i++) {
            if (isFinite(result[i])) finiteCount++;
        }
        assert.strictEqual(finiteCount, close.length, 
            `Found NaN values for factor=${factor}`);
    }
});

test('MWDX batch performance', () => {
    
    const close = new Float64Array(testData.close.slice(0, 1000)); 
    
    
    const startBatch = performance.now();
    const batchResult = wasm.mwdx_batch_js(
        close,
        0.1, 0.9, 0.2    
    );
    const batchTime = performance.now() - startBatch;
    
    
    const metadata = wasm.mwdx_batch_metadata_js(0.1, 0.9, 0.2);
    
    const startSingle = performance.now();
    const singleResults = [];
    
    for (const factor of metadata) {
        const result = wasm.mwdx_js(close, factor);
        singleResults.push(...result);
    }
    const singleTime = performance.now() - startSingle;
    
    
    console.log(`Batch time: ${batchTime.toFixed(2)}ms, Single time: ${singleTime.toFixed(2)}ms`);
    
    
    assertArrayClose(batchResult, singleResults, 1e-9, 'Batch vs single results');
});

test('MWDX edge cases', () => {
    
    
    
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = i + 1;
    }
    const result = wasm.mwdx_js(data, 0.2);
    assert.strictEqual(result.length, data.length);
    
    
    for (let i = 0; i < result.length; i++) {
        assert(isFinite(result[i]), `NaN at index ${i}`);
    }
    
    
    const constantData = new Float64Array(100).fill(50.0);
    const constantResult = wasm.mwdx_js(constantData, 0.2);
    
    assert.strictEqual(constantResult.length, constantData.length);
    
    
    for (let i = 10; i < constantResult.length; i++) {
        assertClose(constantResult[i], 50.0, 1e-6, 
            `Constant value failed at index ${i}`);
    }
});

test('MWDX batch metadata', () => {
    
    const metadata = wasm.mwdx_batch_metadata_js(
        0.2, 0.6, 0.2    
    );
    
    
    assert.strictEqual(metadata.length, 3);
    assertClose(metadata[0], 0.2, 1e-9);
    assertClose(metadata[1], 0.4, 1e-9);
    assertClose(metadata[2], 0.6, 1e-9);
});

test('MWDX consistency across calls', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const result1 = wasm.mwdx_js(close, 0.2);
    const result2 = wasm.mwdx_js(close, 0.2);
    
    assertArrayClose(result1, result2, 1e-15, "MWDX results not consistent");
});

test('MWDX step precision', () => {
    
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = i + 1;
    }
    
    const batch_result = wasm.mwdx_batch_js(
        data,
        0.2, 0.8, 0.3     
    );
    
    
    const rows_cols = wasm.mwdx_batch_rows_cols_js(0.2, 0.8, 0.3, data.length);
    const rows = rows_cols[0];
    
    
    assert.strictEqual(rows, 3);
    assert.strictEqual(batch_result.length, 3 * data.length);
    
    
    const metadata = wasm.mwdx_batch_metadata_js(0.2, 0.8, 0.3);
    assert.strictEqual(metadata.length, 3);
    assertClose(metadata[0], 0.2, 1e-9);
    assertClose(metadata[1], 0.5, 1e-9);
    assertClose(metadata[2], 0.8, 1e-9);
});

test('MWDX streaming simulation', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    const factor = 0.2;
    
    
    const batchResult = wasm.mwdx_js(close, factor);
    
    
    assert.strictEqual(batchResult.length, close.length);
    
    
    for (let i = 0; i < batchResult.length; i++) {
        assert(isFinite(batchResult[i]), `Expected finite value at index ${i}`);
    }
    
    
    assertClose(batchResult[0], close[0], 1e-9, "First value mismatch");
});

test('MWDX high factor', () => {
    
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = Math.sin(i * 0.1) * 100 + 1000; 
    }
    
    const result = wasm.mwdx_js(data, 0.95);
    
    assert.strictEqual(result.length, data.length);
    
    
    for (let i = 0; i < result.length; i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i}`);
    }
});

test('MWDX low factor', () => {
    
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = Math.sin(i * 0.1) * 100 + 1000; 
    }
    
    const result = wasm.mwdx_js(data, 0.01);
    
    assert.strictEqual(result.length, data.length);
    
    
    for (let i = 0; i < result.length; i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i}`);
    }
});

test('MWDX NaN prefix', () => {
    
    const data = new Float64Array([NaN, NaN, 10.0, 20.0, 30.0, 40.0]);
    
    const result = wasm.mwdx_js(data, 0.2);
    assert.strictEqual(result.length, data.length);
    
    
    
    
    
    
    
    
    
    
    
    assert(isNaN(result[0]), 'Expected NaN at index 0');
    assert(isNaN(result[1]), 'Expected NaN at index 1');
    
    
    assert.strictEqual(result[2], 10.0);
    assert(Math.abs(result[3] - 12.0) < 1e-10, `Expected 12.0 at index 3, got ${result[3]}`);
    assert(Math.abs(result[4] - 15.6) < 1e-10, `Expected 15.6 at index 4, got ${result[4]}`);
    assert(Math.abs(result[5] - 20.48) < 1e-10, `Expected 20.48 at index 5, got ${result[5]}`);
    
    
    const cleanData = new Float64Array([10.0, 20.0, 30.0, 40.0]);
    const cleanResult = wasm.mwdx_js(cleanData, 0.2);
    
    
    for (let i = 0; i < cleanResult.length; i++) {
        assert(isFinite(cleanResult[i]), `Expected finite value at index ${i} for clean data`);
    }
});

test('MWDX formula verification', () => {
    
    const data = new Float64Array([10.0, 20.0, 30.0, 40.0]);
    const factor = 0.3;
    
    const result = wasm.mwdx_js(data, factor);
    
    
    const expected = [data[0]]; 
    for (let i = 1; i < data.length; i++) {
        const val = factor * data[i] + (1 - factor) * expected[i-1];
        expected.push(val);
    }
    
    assertArrayClose(result, expected, 1e-12, 'Formula verification failed');
});

test('MWDX all NaN input', () => {
    
    const allNaN = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        allNaN[i] = NaN;
    }
    
    
    const result = wasm.mwdx_js(allNaN, 0.2);
    for (let i = 0; i < result.length; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
});

test('MWDX oscillating data', () => {
    
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = (i % 2 === 0) ? 10.0 : 20.0;
    }
    
    const result = wasm.mwdx_js(data, 0.5);
    assert.strictEqual(result.length, data.length);
    
    
    for (let i = 0; i < result.length; i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i}`);
    }
});

test('MWDX small step size', () => {
    
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = i + 1;
    }
    
    const batch_result = wasm.mwdx_batch_js(
        data,
        0.1, 0.2, 0.05     
    );
    
    const rows_cols = wasm.mwdx_batch_rows_cols_js(0.1, 0.2, 0.05, data.length);
    const rows = rows_cols[0];
    
    assert.strictEqual(rows, 3);
    assert.strictEqual(batch_result.length, 3 * data.length);
});