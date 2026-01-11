/**
 * WASM binding tests for Chandelier Exit indicator.
 * These tests mirror the Rust unit tests to ensure WASM bindings work correctly.
 */
import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';
import { parse } from 'csv-parse/sync';

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
        console.error('Failed to load WASM module:', error);
        throw error;
    }
    
    
    const csvPath = path.join(__dirname, '../../src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv');
    try {
        const csvContent = fs.readFileSync(csvPath, 'utf-8');
        const records = parse(csvContent, {
            columns: false, 
            skip_empty_lines: true,
            from_line: 2  
        });
        
        
        testData = {
            high: new Float64Array(records.map(r => parseFloat(r[3]))),
            low: new Float64Array(records.map(r => parseFloat(r[4]))),
            close: new Float64Array(records.map(r => parseFloat(r[2])))
        };
    } catch (error) {
        console.error('Failed to load test data:', error);
        throw error;
    }
});


const EXPECTED_SHORT_STOP = [68719.23648167, 68705.54391432, 68244.42828185, 67599.49972358, 66883.02246342];


function assertArrayClose(actual, expected, tolerance, msg) {
    assert.strictEqual(actual.length, expected.length, `${msg}: array length mismatch`);
    for (let i = 0; i < actual.length; i++) {
        const diff = Math.abs(actual[i] - expected[i]);
        assert(diff < tolerance, `${msg}: value at index ${i} mismatch - expected ${expected[i]}, got ${actual[i]}, diff ${diff}`);
    }
}

function assertClose(actual, expected, tolerance, msg) {
    const diff = Math.abs(actual - expected);
    assert(diff < tolerance, `${msg}: expected ${expected}, got ${actual}, diff ${diff}`);
}

test('chandelier_exit partial params', () => {
    
    const high = testData.high.slice(0, 100);
    const low = testData.low.slice(0, 100);
    const close = testData.close.slice(0, 100);
    
    const result = wasm.chandelier_exit_wasm(high, low, close, 22, 3.0, true);
    assert(result, 'Should return a result');
    assert.strictEqual(result.long_stop.length, 100);
    assert.strictEqual(result.short_stop.length, 100);
});

test('chandelier_exit accuracy', async () => {
    
    const result = wasm.chandelier_exit_wasm(
        testData.high,
        testData.low,
        testData.close,
        22,     
        3.0,    
        true    
    );
    
    assert(result, 'Should return a result');
    assert(result.long_stop, 'Should have long_stop array');
    assert(result.short_stop, 'Should have short_stop array');
    assert.strictEqual(result.long_stop.length, testData.close.length);
    assert.strictEqual(result.short_stop.length, testData.close.length);
    
    
    const expectedIndices = [15386, 15387, 15388, 15389, 15390];
    
    if (result.short_stop.length > Math.max(...expectedIndices)) {
        
        const actualValues = expectedIndices.map(i => result.short_stop[i]);
        
        assertArrayClose(actualValues, EXPECTED_SHORT_STOP, 1e-5, 'Short stop accuracy at indices 15386-15390');
    } else {
        
        let hasNonNaN = false;
        for (let i = 22; i < result.short_stop.length; i++) {
            if (!isNaN(result.short_stop[i])) {
                hasNonNaN = true;
                break;
            }
        }
        assert(hasNonNaN, 'Should have some non-NaN short_stop values after warmup');
    }
});

test('chandelier_exit default candles', () => {
    
    const high = testData.high.slice(0, 100);
    const low = testData.low.slice(0, 100);
    const close = testData.close.slice(0, 100);
    
    const result = wasm.chandelier_exit_wasm(high, low, close, 22, 3.0, true);
    assert.strictEqual(result.long_stop.length, 100);
    assert.strictEqual(result.short_stop.length, 100);
});

test('chandelier_exit zero period', () => {
    
    const high = new Float64Array([10.0, 11.0, 12.0]);
    const low = new Float64Array([9.0, 10.0, 11.0]);
    const close = new Float64Array([9.5, 10.5, 11.5]);
    
    assert.throws(() => {
        wasm.chandelier_exit_wasm(high, low, close, 0, 3.0, true);
    }, /Invalid period/);
});

test('chandelier_exit period exceeds length', () => {
    
    const high = new Float64Array([10.0, 11.0, 12.0]);
    const low = new Float64Array([9.0, 10.0, 11.0]);
    const close = new Float64Array([9.5, 10.5, 11.5]);
    
    assert.throws(() => {
        wasm.chandelier_exit_wasm(high, low, close, 20, 3.0, true);
    }, /Invalid period/);
});

test('chandelier_exit very small dataset', () => {
    
    const high = new Float64Array([10.0]);
    const low = new Float64Array([9.0]);
    const close = new Float64Array([9.5]);
    
    assert.throws(() => {
        wasm.chandelier_exit_wasm(high, low, close, 22, 3.0, true);
    }, /Invalid period|Not enough valid data/);
});

test('chandelier_exit empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.chandelier_exit_wasm(empty, empty, empty, 22, 3.0, true);
    }, /Input data slice is empty/);
});

test('chandelier_exit invalid mult', () => {
    
    const high = testData.high.slice(0, 30);
    const low = testData.low.slice(0, 30);
    const close = testData.close.slice(0, 30);
    
    
    let result = wasm.chandelier_exit_wasm(high, low, close, 10, -2.0, true);
    assert.strictEqual(result.long_stop.length, 30);
    assert.strictEqual(result.short_stop.length, 30);
    
    
    result = wasm.chandelier_exit_wasm(high, low, close, 10, 0.0, true);
    assert.strictEqual(result.long_stop.length, 30);
    assert.strictEqual(result.short_stop.length, 30);
});

test('chandelier_exit NaN handling', () => {
    
    const high = testData.high.slice(0, 50);
    const low = testData.low.slice(0, 50);
    const close = testData.close.slice(0, 50);
    
    const result = wasm.chandelier_exit_wasm(high, low, close, 10, 2.0, true);
    assert.strictEqual(result.long_stop.length, 50);
    assert.strictEqual(result.short_stop.length, 50);
    
    
    for (let i = 0; i < 9; i++) {
        assert(isNaN(result.long_stop[i]), `Expected NaN at warmup index ${i} for long_stop`);
        assert(isNaN(result.short_stop[i]), `Expected NaN at warmup index ${i} for short_stop`);
    }
    
    
    let hasValidAfterWarmup = false;
    for (let i = 9; i < 50; i++) {
        if (!isNaN(result.long_stop[i]) || !isNaN(result.short_stop[i])) {
            hasValidAfterWarmup = true;
            break;
        }
    }
    assert(hasValidAfterWarmup, 'Should have valid values after warmup');
});

test('chandelier_exit all NaN input', () => {
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.chandelier_exit_wasm(allNaN, allNaN, allNaN, 22, 3.0, true);
    }, /All values are NaN/);
});

test('chandelier_exit with custom parameters', () => {
    const result = wasm.chandelier_exit_wasm(
        testData.high.slice(0, 100),
        testData.low.slice(0, 100),
        testData.close.slice(0, 100),
        14,     
        2.0,    
        false   
    );
    
    assert(result, 'Should return a result');
    assert.strictEqual(result.long_stop.length, 100, 'Should match input length');
    assert.strictEqual(result.short_stop.length, 100, 'Should match input length');
    
    
    let hasValidLong = false;
    let hasValidShort = false;
    for (let i = 14; i < 100; i++) {
        if (!isNaN(result.long_stop[i])) hasValidLong = true;
        if (!isNaN(result.short_stop[i])) hasValidShort = true;
    }
    assert(hasValidLong || hasValidShort, 'Should have valid values after warmup period');
});

test('chandelier_exit streaming', () => {
    const stream = new wasm.ChandelierExitStreamWasm(22, 3.0, true);
    
    
    let lastResult = null;
    for (let i = 0; i < Math.min(50, testData.close.length); i++) {
        const result = stream.update(
            testData.high[i],
            testData.low[i],
            testData.close[i]
        );
        if (result !== null) {
            lastResult = result;
        }
    }
    
    assert(lastResult, 'Should have produced results after warmup');
    
    
    assert(typeof lastResult.long_stop === 'number' || isNaN(lastResult.long_stop), 'Should have long_stop value');
    assert(typeof lastResult.short_stop === 'number' || isNaN(lastResult.short_stop), 'Should have short_stop value');
    
    
    stream.reset();
    let resultAfterReset = null;
    for (let i = 0; i < 30; i++) {
        resultAfterReset = stream.update(
            testData.high[i],
            testData.low[i],
            testData.close[i]
        );
    }
    
    
    assert(resultAfterReset !== null, 'Should produce results after reset and feeding enough data');
});

test('chandelier_exit batch single parameter set', () => {
    
    const high = testData.high.slice(0, 100);
    const low = testData.low.slice(0, 100);
    const close = testData.close.slice(0, 100);
    
    const batchResult = wasm.ce_batch(high, low, close, {
        period_range: [22, 22, 0],
        mult_range: [3.0, 3.0, 0],
        use_close: true
    });
    
    assert(batchResult, 'Should return batch result');
    assert(batchResult.values, 'Should have values array');
    assert(batchResult.combos, 'Should have combos array');
    assert.strictEqual(batchResult.rows, 2); 
    assert.strictEqual(batchResult.cols, 100);
    assert.strictEqual(batchResult.combos.length, 1);
    
    
    const singleResult = wasm.chandelier_exit_wasm(high, low, close, 22, 3.0, true);
    
    
    const batchLong = batchResult.values.slice(0, 100);
    const batchShort = batchResult.values.slice(100, 200);
    
    
    for (let i = 0; i < 100; i++) {
        if (isNaN(singleResult.long_stop[i]) && isNaN(batchLong[i])) continue;
        if (!isNaN(singleResult.long_stop[i])) {
            assertClose(batchLong[i], singleResult.long_stop[i], 1e-10, `Batch long vs single mismatch at ${i}`);
        }
        
        if (isNaN(singleResult.short_stop[i]) && isNaN(batchShort[i])) continue;
        if (!isNaN(singleResult.short_stop[i])) {
            assertClose(batchShort[i], singleResult.short_stop[i], 1e-10, `Batch short vs single mismatch at ${i}`);
        }
    }
});

test('chandelier_exit batch multiple parameters', () => {
    
    const high = testData.high.slice(0, 50);
    const low = testData.low.slice(0, 50);
    const close = testData.close.slice(0, 50);
    
    const batchResult = wasm.ce_batch(high, low, close, {
        period_range: [10, 20, 10],  
        mult_range: [2.0, 3.0, 1.0],  
        use_close: true
    });
    
    
    
    assert.strictEqual(batchResult.rows, 8);
    assert.strictEqual(batchResult.cols, 50);
    assert.strictEqual(batchResult.combos.length, 4);
    
    
    const expectedParams = [
        { period: 10, mult: 2.0 },
        { period: 10, mult: 3.0 },
        { period: 20, mult: 2.0 },
        { period: 20, mult: 3.0 }
    ];
    
    for (let i = 0; i < 4; i++) {
        assert.strictEqual(batchResult.combos[i].period, expectedParams[i].period);
        assertClose(batchResult.combos[i].mult, expectedParams[i].mult, 1e-10, `mult[${i}]`);
        assert.strictEqual(batchResult.combos[i].use_close, true);
    }
});

test('chandelier_exit batch edge cases', () => {
    
    const high = new Float64Array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19]);
    const low = new Float64Array([9, 10, 11, 12, 13, 14, 15, 16, 17, 18]);
    const close = new Float64Array([9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5]);
    
    
    const singleBatch = wasm.ce_batch(high, low, close, {
        period_range: [5, 5, 1],
        mult_range: [2.0, 2.0, 0.1],
        use_close: true
    });
    
    assert.strictEqual(singleBatch.rows, 2);  
    assert.strictEqual(singleBatch.cols, 10);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    
    const largeBatch = wasm.ce_batch(high, low, close, {
        period_range: [5, 7, 10], 
        mult_range: [2.0, 2.0, 0],
        use_close: true
    });
    
    
    assert.strictEqual(largeBatch.rows, 2);
    assert.strictEqual(largeBatch.combos.length, 1);
    assert.strictEqual(largeBatch.combos[0].period, 5);
});

test('chandelier_exit zero-copy API', () => {
    const high = new Float64Array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19]);
    const low = new Float64Array([9, 10, 11, 12, 13, 14, 15, 16, 17, 18]);
    const close = new Float64Array([9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5]);
    const period = 5;
    const mult = 2.0;
    const use_close = true;
    
    
    const highPtr = wasm.ce_alloc(high.length);
    const lowPtr = wasm.ce_alloc(low.length);
    const closePtr = wasm.ce_alloc(close.length);
    const outPtr = wasm.ce_alloc(high.length * 2); 
    
    assert(highPtr !== 0, 'Failed to allocate high buffer');
    assert(lowPtr !== 0, 'Failed to allocate low buffer');
    assert(closePtr !== 0, 'Failed to allocate close buffer');
    assert(outPtr !== 0, 'Failed to allocate output buffer');
    
    try {
        
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, high.length);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, low.length);
        const closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, close.length);
        
        
        highView.set(high);
        lowView.set(low);
        closeView.set(close);
        
        
        wasm.ce_into(highPtr, lowPtr, closePtr, outPtr, high.length, period, mult, use_close);
        
        
        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, high.length * 2);
        
        
        const regularResult = wasm.chandelier_exit_wasm(high, low, close, period, mult, use_close);
        
        
        for (let i = 0; i < high.length; i++) {
            if (isNaN(regularResult.long_stop[i]) && isNaN(outView[i])) continue;
            if (!isNaN(regularResult.long_stop[i])) {
                assertClose(outView[i], regularResult.long_stop[i], 1e-10, 
                          `Zero-copy long mismatch at ${i}`);
            }
        }
        
        
        for (let i = 0; i < high.length; i++) {
            if (isNaN(regularResult.short_stop[i]) && isNaN(outView[high.length + i])) continue;
            if (!isNaN(regularResult.short_stop[i])) {
                assertClose(outView[high.length + i], regularResult.short_stop[i], 1e-10,
                          `Zero-copy short mismatch at ${i}`);
            }
        }
    } finally {
        
        wasm.ce_free(highPtr, high.length);
        wasm.ce_free(lowPtr, low.length);
        wasm.ce_free(closePtr, close.length);
        wasm.ce_free(outPtr, high.length * 2);
    }
});

test('chandelier_exit zero-copy error handling', () => {
    
    assert.throws(() => {
        wasm.ce_into(0, 0, 0, 0, 10, 5, 2.0, true);
    }, /null pointer/i);
    
    
    const ptr = wasm.ce_alloc(30); 
    try {
        
        assert.throws(() => {
            wasm.ce_into(ptr, ptr + 80, ptr + 160, ptr + 240, 10, 0, 2.0, true);
        }, /Invalid period/);
    } finally {
        wasm.ce_free(ptr, 30);
    }
});

test('chandelier_exit edge cases', () => {
    
    const minData = 30;
    const result = wasm.chandelier_exit_wasm(
        testData.high.slice(0, minData),
        testData.low.slice(0, minData),
        testData.close.slice(0, minData),
        22,
        3.0,
        true
    );
    
    assert(result, 'Should handle minimum data');
    assert.strictEqual(result.long_stop.length, minData, 'Should return correct length');
    assert.strictEqual(result.short_stop.length, minData, 'Should return correct length');
    
    
    for (let i = 0; i < 21; i++) {
        assert(isNaN(result.long_stop[i]), `long_stop[${i}] should be NaN during warmup`);
        assert(isNaN(result.short_stop[i]), `short_stop[${i}] should be NaN during warmup`);
    }
    
    
    let hasValidValues = false;
    for (let i = 21; i < minData; i++) {
        if (!isNaN(result.long_stop[i]) || !isNaN(result.short_stop[i])) {
            hasValidValues = true;
            break;
        }
    }
    assert(hasValidValues, 'Should have valid values after warmup period');
});

test.after(() => {
    console.log('Chandelier Exit WASM tests completed');
});
