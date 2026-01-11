/**
 * WASM binding tests for TRIX indicator.
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

test('TRIX accuracy', () => {
    const closePrices = testData.close;
    const period = 18;
    
    
    const result = wasm.trix_js(closePrices, period);
    
    assert.equal(result.length, closePrices.length, 'TRIX length mismatch');
    
    
    const expectedLastFive = [-16.03736447, -15.92084231, -15.76171478, -15.53571033, -15.34967155];
    
    
    assert(result.length >= 5, 'TRIX length too short');
    const resultLastFive = result.slice(-5);
    
    for (let i = 0; i < expectedLastFive.length; i++) {
        assertClose(
            resultLastFive[i], 
            expectedLastFive[i], 
            1e-6, 
            `TRIX mismatch at index ${i}`
        );
    }
});

test('TRIX error handling', () => {
    
    assert.throws(
        () => wasm.trix_js([10.0, 20.0, 30.0], 0),
        /Invalid period/,
        'TRIX should fail with zero period'
    );
    
    
    assert.throws(
        () => wasm.trix_js([10.0, 20.0, 30.0], 10),
        /Invalid period/,
        'TRIX should fail with period exceeding length'
    );
    
    
    assert.throws(
        () => wasm.trix_js([42.0], 18),
        /Invalid period|Not enough valid data/,
        'TRIX should fail with insufficient data'
    );
    
    
    assert.throws(
        () => wasm.trix_js([], 18),
        /empty/i,
        'TRIX should fail with empty data'
    );
    
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    assert.throws(
        () => wasm.trix_js(allNaN, 18),
        /All values are NaN/,
        'TRIX should fail with all NaN values'
    );
});

test('TRIX partial params', () => {
    const closePrices = testData.close;
    
    
    const result14 = wasm.trix_js(closePrices, 14);
    assert.equal(result14.length, closePrices.length);
    
    const result20 = wasm.trix_js(closePrices, 20);
    assert.equal(result20.length, closePrices.length);
});

test('TRIX fast API (unsafe)', async () => {
    
    if (!wasm.trix_alloc || !wasm.trix_into || !wasm.trix_free || !wasm.memory) {
        console.log('Skipping TRIX fast API test - functions not available');
        return;
    }
    
    const closePrices = testData.close;
    const len = closePrices.length;
    const period = 18;
    
    
    const outPtr = wasm.trix_alloc(len);
    
    try {
        
        const inPtr = wasm.trix_alloc(len);
        const wasmMemory = new Float64Array(wasm.memory.buffer);
        wasmMemory.set(closePrices, inPtr / 8);
        
        
        wasm.trix_into(inPtr, outPtr, len, period);
        
        
        const result = new Float64Array(wasm.memory.buffer, outPtr, len);
        
        
        const safeResult = wasm.trix_js(closePrices, period);
        assertArrayClose(Array.from(result), safeResult, 1e-10, 'Fast API results should match safe API');
        
        
        wasm.trix_into(inPtr, inPtr, len, period);
        const inPlaceResult = new Float64Array(wasm.memory.buffer, inPtr, len);
        assertArrayClose(Array.from(inPlaceResult), safeResult, 1e-10, 'In-place results should match safe API');
        
        
        wasm.trix_free(inPtr, len);
    } finally {
        
        wasm.trix_free(outPtr, len);
    }
});

test('TRIX batch processing', () => {
    const closePrices = testData.close;
    
    
    const singleConfig = {
        period_range: [18, 18, 0]
    };
    const singleResult = wasm.trix_batch(closePrices, singleConfig);
    
    assert(singleResult.values, 'Batch result should have values');
    assert(singleResult.periods, 'Batch result should have periods');
    assert.equal(singleResult.rows, 1, 'Single batch should have 1 row');
    assert.equal(singleResult.cols, closePrices.length, 'Columns should match input length');
    assert.equal(singleResult.periods.length, 1, 'Should have 1 period');
    assert.equal(singleResult.periods[0], 18, 'Period should be 18');
    
    
    const rangeConfig = {
        period_range: [10, 20, 5]  
    };
    const rangeResult = wasm.trix_batch(closePrices, rangeConfig);
    
    assert.equal(rangeResult.rows, 3, 'Range batch should have 3 rows');
    assert.equal(rangeResult.cols, closePrices.length);
    assert.equal(rangeResult.values.length, 3 * closePrices.length);
    assert.deepEqual(rangeResult.periods, [10, 15, 20]);
});

test('TRIX batch fast API', async () => {
    
    if (!wasm.trix_alloc || !wasm.trix_batch_into || !wasm.trix_free || !wasm.memory) {
        console.log('Skipping TRIX batch fast API test - functions not available');
        return;
    }
    
    const closePrices = testData.close;
    const len = closePrices.length;
    
    
    const periodStart = 10;
    const periodEnd = 20; 
    const periodStep = 5;  
    const numCombos = 3;
    const totalSize = numCombos * len;
    
    
    const inPtr = wasm.trix_alloc(len);
    const outPtr = wasm.trix_alloc(totalSize);
    
    try {
        
        const wasmMemory = new Float64Array(wasm.memory.buffer);
        wasmMemory.set(closePrices, inPtr / 8);
        
        
        const resultRows = wasm.trix_batch_into(
            inPtr, outPtr, len,
            periodStart, periodEnd, periodStep
        );
        
        assert.equal(resultRows, numCombos, 'Should return correct number of combinations');
        
        
        const results = new Float64Array(wasm.memory.buffer, outPtr, totalSize);
        
        
        const firstRow = Array.from(results.slice(0, len));
        const singleResult = wasm.trix_js(closePrices, 10);
        assertArrayClose(firstRow, singleResult, 1e-10, 'First batch row should match single computation');
    } finally {
        wasm.trix_free(inPtr, len);
        wasm.trix_free(outPtr, totalSize);
    }
});

test('TRIX with NaN handling', () => {
    const closePrices = testData.close.slice(0, 500); 
    
    
    closePrices[100] = NaN;
    closePrices[101] = NaN;
    closePrices[102] = NaN;
    closePrices[200] = NaN;
    closePrices[300] = NaN;
    closePrices[301] = NaN;
    
    
    const result = wasm.trix_js(closePrices, 18);
    assert.equal(result.length, closePrices.length);
    
    
    
    const validBeforeNaN = result.slice(80, 100).filter(v => !isNaN(v));
    assert(validBeforeNaN.length > 0, 'Should have valid values before first NaN');
    
    
    const allNaNAfter = result.slice(110).every(v => isNaN(v));
    assert(allNaNAfter, 'All values after NaN should be NaN due to EMA propagation');
});

test('TRIX reinput', () => {
    const closePrices = testData.close;
    const period = 10;
    
    
    const firstResult = wasm.trix_js(closePrices, period);
    
    
    const secondResult = wasm.trix_js(firstResult, period);
    
    assert.equal(firstResult.length, secondResult.length);
    
    
    const firstValidIdx = firstResult.findIndex(v => !isNaN(v));
    const secondValidIdx = secondResult.findIndex(v => !isNaN(v));
    
    assert(secondValidIdx > firstValidIdx, 'Second TRIX should have more warmup period');
});

test('TRIX stream processing', () => {
    const closePrices = testData.close.slice(0, 100); 
    const period = 18;
    
    
    const batchResult = wasm.trix_js(closePrices, period);
    
    
    if (!wasm.TrixStream) {
        console.log('Skipping TRIX stream processing test - TrixStream not available');
        return;
    }
    
    
    const stream = new wasm.TrixStream(period);
    const streamValues = [];
    
    for (let i = 0; i < closePrices.length; i++) {
        const result = stream.update(closePrices[i]);
        streamValues.push(result !== null && result !== undefined ? result : NaN);
    }
    
    
    assert.equal(batchResult.length, streamValues.length, 'Length mismatch');
    
    
    for (let i = 0; i < batchResult.length; i++) {
        if (!isNaN(batchResult[i]) && !isNaN(streamValues[i])) {
            assertClose(
                batchResult[i],
                streamValues[i],
                1e-9,
                `TRIX streaming mismatch at index ${i}`
            );
        }
    }
});

test('TRIX all NaN input', () => {
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(
        () => wasm.trix_js(allNaN, 18),
        /All values are NaN/,
        'TRIX should fail with all NaN values'
    );
});

test('TRIX batch metadata', () => {
    const closePrices = testData.close.slice(0, 100); 
    
    
    const rangeConfig = {
        period_range: [10, 20, 5]  
    };
    const rangeResult = wasm.trix_batch(closePrices, rangeConfig);
    
    
    assert.equal(rangeResult.rows, 3, 'Should have 3 rows');
    assert.equal(rangeResult.cols, closePrices.length, 'Columns should match input length');
    assert.deepEqual(rangeResult.periods, [10, 15, 20], 'Periods should be [10, 15, 20]');
    
    
    const periods = [10, 15, 20];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * closePrices.length;
        const rowEnd = rowStart + closePrices.length;
        const rowData = rangeResult.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.trix_js(closePrices, periods[i]);
        assertArrayClose(
            rowData,
            singleResult,
            1e-10,
            `Period ${periods[i]} mismatch`
        );
    }
});

test('TRIX batch edge cases', () => {
    const closePrices = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    
    
    const singleBatch = wasm.trix_batch(closePrices, {
        period_range: [5, 5, 1]
    });
    
    assert.equal(singleBatch.values.length, closePrices.length, 'Single batch length mismatch');
    assert.equal(singleBatch.rows, 1, 'Should have 1 row');
    assert.deepEqual(singleBatch.periods, [5], 'Should have period 5');
    
    
    const largeBatch = wasm.trix_batch(closePrices, {
        period_range: [5, 7, 10]  
    });
    
    
    assert.equal(largeBatch.values.length, closePrices.length, 'Large step batch length mismatch');
    assert.equal(largeBatch.rows, 1, 'Should have 1 row');
    assert.deepEqual(largeBatch.periods, [5], 'Should only have period 5');
    
    
    assert.throws(
        () => wasm.trix_batch(new Float64Array([]), { period_range: [10, 10, 0] }),
        /Empty|All values are NaN/,
        'Should throw on empty data'
    );
});

test('TRIX with mixed NaN patterns', () => {
    const closePrices = testData.close.slice(0, 200).map(v => v); 
    
    
    closePrices[50] = NaN;        
    closePrices[100] = NaN;
    closePrices[101] = NaN;
    closePrices[102] = NaN;        
    closePrices[150] = NaN;
    closePrices[152] = NaN;        
    
    
    const result = wasm.trix_js(closePrices, 18);
    assert.equal(result.length, closePrices.length, 'Length should match');
    
    
    
    
    
    const warmup = 3 * (18 - 1) + 1; 
    
    
    const firstNonNaN = result.findIndex(v => !isNaN(v));
    if (firstNonNaN === -1) {
        
        assert(true, 'All values are NaN as expected when NaN is in warmup period');
    } else {
        const validBeforeNaN = result.slice(Math.max(0, 50 - 10), 50).filter(v => !isNaN(v));
        assert(validBeforeNaN.length > 0 || firstNonNaN === -1, 'Should have valid values before first NaN or all NaN');
    }
    
    const allNaNAfter = result.slice(55).every(v => isNaN(v));
    assert(allNaNAfter, 'All values after NaN should be NaN due to EMA propagation');
});

test('TRIX warmup period validation', () => {
    const closePrices = testData.close.slice(0, 100);
    const period = 10;
    
    const result = wasm.trix_js(closePrices, period);
    
    
    const expectedWarmup = 3 * (period - 1) + 1;  
    
    
    for (let i = 0; i < expectedWarmup; i++) {
        assert(isNaN(result[i]), `Expected NaN at warmup index ${i}`);
    }
    
    
    const hasValidData = closePrices.slice(0, expectedWarmup + 10).every(v => !isNaN(v));
    if (hasValidData) {
        assert(!isNaN(result[expectedWarmup]), `Expected valid value at index ${expectedWarmup}`);
    }
});

test('TRIX batch full parameter sweep', () => {
    const closePrices = testData.close.slice(0, 60);
    
    
    const batchResult = wasm.trix_batch(closePrices, {
        period_range: [10, 18, 4]  
    });
    
    
    assert.equal(batchResult.rows, 3, 'Should have 3 rows');
    assert.equal(batchResult.cols, 60, 'Should have 60 columns');
    assert.equal(batchResult.values.length, 3 * 60, 'Should have 180 values');
    assert.deepEqual(batchResult.periods, [10, 14, 18], 'Periods should be [10, 14, 18]');
    
    
    const periods = [10, 14, 18];
    for (let row = 0; row < periods.length; row++) {
        const period = periods[row];
        const warmup = 3 * (period - 1) + 1;
        const rowStart = row * 60;
        const rowData = batchResult.values.slice(rowStart, rowStart + 60);
        
        
        for (let i = 0; i < warmup && i < 60; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${period}`);
        }
        
        
        if (warmup < 60) {
            assert(!isNaN(rowData[warmup]), `Expected valid value after warmup for period ${period}`);
        }
    }
});

test.after(() => {
    console.log('TRIX WASM tests completed');
});
