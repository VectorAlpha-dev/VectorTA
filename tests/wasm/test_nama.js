/**
 * WASM binding tests for NAMA indicator.
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

test('NAMA basic', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.nama_js(close, 30);
    assert.strictEqual(result.length, close.length);
    
    
    for (let i = 0; i < 29; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} during warmup`);
    }
    
    
    let nonNanCount = 0;
    for (let i = 29; i < result.length; i++) {
        if (!isNaN(result[i])) nonNanCount++;
    }
    assert.strictEqual(nonNanCount, close.length - 29, 'All values after warmup should be non-NaN');
});

test('NAMA accuracy', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const result = wasm.nama_js(close, 30);
    
    assert.strictEqual(result.length, close.length);
    
    
    
    const expectedLastFive = [
        59248.42400839,
        59226.18226649,
        59167.91952826,
        59163.80438196,
        59009.01273427
    ];
    
    
    const last5Actual = result.slice(-5);
    assertArrayClose(last5Actual, expectedLastFive, 1e-8,
                    "NAMA last 5 values mismatch");
    
    
    
});




test('NAMA edge cases', () => {
    
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    const result = wasm.nama_js(data, 3);
    assert.strictEqual(result.length, 5);
    
    
    assert(isNaN(result[0]));
    assert(isNaN(result[1]));
    assert(!isNaN(result[2]));
    
    
    const constant = new Float64Array(50).fill(100.0);
    const resultConst = wasm.nama_js(constant, 10);
    assert.strictEqual(resultConst.length, 50);
    
    
    for (let i = 9; i < 50; i++) {  
        assert(isFinite(resultConst[i]));
    }
    
    
    const alternating = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        alternating[i] = i % 2 === 0 ? 100.0 : 50.0;
    }
    const resultAlt = wasm.nama_js(alternating, 5);
    assert.strictEqual(resultAlt.length, 50);
});

test('NAMA infinity handling', () => {
    
    const dataInf = new Float64Array([100.0, 102.0, 101.0, Infinity, 105.0, 104.0, 106.0, 108.0]);
    
    
    const result = wasm.nama_js(dataInf, 3);
    assert.strictEqual(result.length, dataInf.length);
    
    
    let nonNanCount = 0;
    for (let i = 0; i < result.length; i++) {
        if (!isNaN(result[i])) nonNanCount++;
    }
    assert(nonNanCount > 0, 'Should have some non-NaN values despite infinity');
    
    
    const dataNegInf = new Float64Array([100.0, 102.0, 101.0, -Infinity, 105.0, 104.0, 106.0, 108.0]);
    const resultNegInf = wasm.nama_js(dataNegInf, 3);
    assert.strictEqual(resultNegInf.length, dataNegInf.length);
});

test('NAMA large dataset', () => {
    
    const largeData = new Float64Array(10000);
    for (let i = 0; i < 10000; i++) {
        largeData[i] = 100 + Math.sin(i / 100) * 10;
    }
    
    
    const result = wasm.nama_js(largeData, 50);
    assert.strictEqual(result.length, largeData.length);
    
    
    for (let i = 0; i < 49; i++) {  
        assert(isNaN(result[i]), `Expected NaN at index ${i} during warmup`);
    }
    
    
    let nonNanCount = 0;
    for (let i = 49; i < result.length; i++) {
        if (!isNaN(result[i])) nonNanCount++;
    }
    assert.strictEqual(nonNanCount, largeData.length - 49, 'All values after warmup should be non-NaN');
});

test('NAMA zero period', () => {
    const data = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.nama_js(data, 0);
    }, /Invalid (period|length)/);  
});

test('NAMA period exceeds length', () => {
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.nama_js(dataSmall, 10);
    }, /Invalid (period|length)/);  
});

test('NAMA very small dataset', () => {
    const singlePoint = new Float64Array([42.0]);
    
    
    const result = wasm.nama_js(singlePoint, 1);
    assert.strictEqual(result.length, 1);
    
    
    assert.throws(() => {
        wasm.nama_js(singlePoint, 2);
    }, /Invalid (period|length)|Not enough valid data/);  
});

test('NAMA empty input', () => {
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.nama_js(empty, 5);
    }, /Input data slice is empty|Empty input/);
});

test('NAMA NaN handling', () => {
    
    const dataNan = new Float64Array([NaN, NaN, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    
    const result = wasm.nama_js(dataNan, 3);
    assert.strictEqual(result.length, dataNan.length);
    
    
    assert(isNaN(result[0]));
    assert(isNaN(result[1]));
    
    
    
    
    for (let i = 0; i < 4; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} during warmup`);
    }
    
    
    for (let i = 4; i < dataNan.length; i++) {
        assert(!isNaN(result[i]), `Expected non-NaN at index ${i} after warmup`);
    }
});

test('NAMA batch processing', () => {
    const close = new Float64Array(testData.close.slice(0, 100));  
    
    
    const resultSingle = wasm.nama_batch(close, {
        period_range: [30, 30, 0]
    });
    assert(resultSingle, 'Batch result should exist');
    assert(resultSingle.values, 'Should have values array');
    assert(resultSingle.combos, 'Should have parameter combinations');
    
    
    assert.strictEqual(resultSingle.combos.length, 1);
    assert.strictEqual(resultSingle.combos[0].period, 30);
    assert.strictEqual(resultSingle.values.length, 100);  
    
    
    const singleRow = resultSingle.values;
    for (let i = 0; i < 29; i++) {  
        assert(isNaN(singleRow[i]), `Expected NaN at index ${i} during warmup`);
    }
    
    
    let nonNanCount = 0;
    for (let i = 29; i < 100; i++) {
        if (!isNaN(singleRow[i])) nonNanCount++;
    }
    assert(nonNanCount > 0, 'Should have non-NaN values after warmup');
    
    
    const resultMulti = wasm.nama_batch(close, {
        period_range: [10, 30, 10]
    });
    assert(resultMulti.combos.length === 3, 'Should have 3 periods');
    const periods = resultMulti.combos.map(c => c.period);
    assert.deepStrictEqual(periods, [10, 20, 30]);
    
    
    assert.strictEqual(resultMulti.values.length, 300);  
});

test('NAMA zero-copy API', () => {
    if (!wasm.nama_alloc || !wasm.nama_into || !wasm.nama_free) {
        console.log('Zero-copy API not available, skipping test');
        return;
    }
    
    const data = new Float64Array([
        100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0,
        107.0, 109.0, 111.0, 110.0, 112.0, 114.0, 113.0, 115.0
    ]);
    
    
    const inPtr = wasm.nama_alloc(data.length);
    const outPtr = wasm.nama_alloc(data.length);
    assert(inPtr, 'Should allocate input memory');
    assert(outPtr, 'Should allocate output memory');
    
    try {
        
        const memory = wasm.__wasm.memory;
        
        
        const inputArray = new Float64Array(memory.buffer, inPtr, data.length);
        inputArray.set(data);
        
        
        wasm.nama_into(inPtr, outPtr, data.length, 5);
        
        
        const result = new Float64Array(memory.buffer, outPtr, data.length);
        
        
        const expected = wasm.nama_js(data, 5);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(expected[i])) {
                assert(isNaN(result[i]), `Expected NaN at index ${i}`);
            } else {
                assertClose(result[i], expected[i], 1e-10);
            }
        }
    } finally {
        
        wasm.nama_free(inPtr, data.length);
        wasm.nama_free(outPtr, data.length);
    }
});



/*
test('NAMA streaming', () => {
    // Use same test data as Rust
    const data = new Float64Array([
        100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0,
        107.0, 109.0, 111.0, 110.0, 112.0, 114.0, 113.0, 115.0
    ]);
    const period = 5;
    
    if (!wasm.NamaStream) {
        console.log('Streaming API not available, skipping test');
        return;
    }
    
    // Batch calculation
    const batchResult = wasm.nama_js(data, period);
    
    // Streaming calculation
    const stream = new wasm.NamaStream(period);
    const streamValues = [];
    
    for (const price of data) {
        const result = stream.update(price);
        streamValues.push(result !== null && result !== undefined ? result : NaN);
    }
    
    // Compare batch vs streaming
    assert.strictEqual(batchResult.length, streamValues.length);
    
    // Compare values after warmup (index 4 onwards for period=5)
    const warmup = 4;  // first + period - 1 = 0 + 5 - 1 = 4
    for (let i = warmup; i < data.length; i++) {
        if (isFinite(batchResult[i]) && isFinite(streamValues[i])) {
            assertClose(batchResult[i], streamValues[i], 1e-10,
                       `NAMA streaming mismatch at index ${i}`);
        }
    }
});
*/

test('NAMA all NaN input', () => {
    const allNan = new Float64Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.nama_js(allNan, 30);
    }, /All values are NaN/);
});


test('NAMA vs Rust reference values', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.nama_js(close, 30);
    
    
    const expectedLastFive = [
        59248.42400839,
        59226.18226649,
        59167.91952826,
        59163.80438196,
        59009.01273427
    ];
    
    
    const last5 = result.slice(-5);
    
    
    for (let i = 0; i < expectedLastFive.length; i++) {
        assertClose(last5[i], expectedLastFive[i], 1e-8,
                   `NAMA value mismatch with Rust at position ${i} of last 5`);
    }
    
    
    const warmupPeriod = 29; 
    for (let i = 0; i < warmupPeriod; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} during warmup`);
    }
    
    
    for (let i = warmupPeriod; i < result.length; i++) {
        assert(!isNaN(result[i]) && isFinite(result[i]), 
               `Expected valid value at index ${i} after warmup`);
    }
});

test.after(() => {
    
});
