/**
 * WASM binding tests for ROCR indicator.
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

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

test.before(async () => {
    
    try {
        const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
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

test('ROCR partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.rocr_js(close, 10);
    assert.strictEqual(result.length, close.length);
});

test('ROCR accuracy', () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.rocr;
    
    const result = wasm.rocr_js(close, expected.defaultParams.period);
    
    assert.strictEqual(result.length, close.length);
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-8,
        "ROCR last 5 values mismatch"
    );
});

test('ROCR default period', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.rocr_js(close, 10);
    assert.strictEqual(result.length, close.length);
});

test('ROCR zero period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.rocr_js(inputData, 0);
    }, /Invalid period/);
});

test('ROCR period exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.rocr_js(dataSmall, 10);
    }, /Invalid period/);
});

test('ROCR very small dataset', () => {
    
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.rocr_js(singlePoint, 9);
    }, /Invalid period|Not enough/);
});

test('ROCR empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.rocr_js(empty, 9);
    }, /Empty data/);
});

test('ROCR all NaN input', () => {
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.rocr_js(allNaN, 9);
    }, /All values are NaN/);
});

test('ROCR reinput', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const firstResult = wasm.rocr_js(close, 14);
    assert.strictEqual(firstResult.length, close.length);
    
    
    const secondResult = wasm.rocr_js(firstResult, 14);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    
    if (secondResult.length > 28) {
        for (let i = 28; i < secondResult.length; i++) {
            assert(!isNaN(secondResult[i]), `Found unexpected NaN at index ${i} after double warmup`);
        }
    }
});

test('ROCR NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.rocr_js(close, 9);
    assert.strictEqual(result.length, close.length);
    
    
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    
    assertAllNaN(result.slice(0, 9), "Expected NaN in warmup period");
});

test('ROCR batch single parameter', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const batchResult = wasm.rocr_batch(close, {
        period_range: [10, 10, 0]
    });
    
    
    const singleResult = wasm.rocr_js(close, 10);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    assert.strictEqual(batchResult.rows, 1);
    assert.strictEqual(batchResult.cols, close.length);
    
    
    const rowData = batchResult.values.slice(0, close.length);
    assertArrayClose(rowData, singleResult, 1e-10, "Batch vs single mismatch");
});

test('ROCR batch multiple periods', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    
    
    const batchResult = wasm.rocr_batch(close, {
        period_range: [5, 15, 5]
    });
    
    
    assert.strictEqual(batchResult.values.length, 3 * 100);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 100);
    
    
    const periods = [5, 10, 15];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.rocr_js(close, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Batch row ${i} (period=${periods[i]}) mismatch`
        );
    }
});

test('ROCR edge cases', () => {
    
    const dataWithZeros = new Float64Array([1.0, 2.0, 0.0, 4.0, 5.0, 6.0, 0.0, 8.0, 9.0, 10.0]);
    const result = wasm.rocr_js(dataWithZeros, 2);
    assert.strictEqual(result.length, dataWithZeros.length);
    
    
    assert.strictEqual(result[2], 0.0, "Expected 0 when dividing by 0");
    assert.strictEqual(result[6], 0.0, "Expected 0 when dividing by 0");
    
    
    const smallResult = wasm.rocr_js(new Float64Array(testData.close.slice(0, 20)), 1);
    assert.strictEqual(smallResult.length, 20);
    
    
    const largeData = new Float64Array(testData.close.slice(0, 200));
    const largeResult = wasm.rocr_js(largeData, 100);
    assert.strictEqual(largeResult.length, 200);
    
    
    for (let i = 0; i < 100; i++) {
        assert(isNaN(largeResult[i]), `Expected NaN at index ${i} during warmup`);
    }
    
    
    for (let i = 100; i < 200; i++) {
        assert(!isNaN(largeResult[i]), `Expected valid value at index ${i} after warmup`);
    }
});

test('ROCR batch configuration', () => {
    
    const close = new Float64Array(testData.close.slice(0, 50));
    
    
    const result1 = wasm.rocr_batch(close, {
        period_range: [5, 20, 5]  
    });
    assert.strictEqual(result1.rows, 4);
    assert.strictEqual(result1.combos.length, 4);
    assert.strictEqual(result1.combos[0].period, 5);
    assert.strictEqual(result1.combos[1].period, 10);
    assert.strictEqual(result1.combos[2].period, 15);
    assert.strictEqual(result1.combos[3].period, 20);
    
    
    const result2 = wasm.rocr_batch(close, {
        period_range: [7, 7, 0]
    });
    assert.strictEqual(result2.rows, 1);
    assert.strictEqual(result2.combos[0].period, 7);
    
    
    const result3 = wasm.rocr_batch(close, {
        period_range: [8, 10, 1]  
    });
    assert.strictEqual(result3.rows, 3);
});

test.after(() => {
    console.log('ROCR WASM tests completed');
});