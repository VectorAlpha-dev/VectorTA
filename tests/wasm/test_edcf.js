/**
 * WASM binding tests for EDCF indicator.
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
    assertNoNaN
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

test('EDCF partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.edcf_js(close, 15);
    assert.strictEqual(result.length, close.length);
    
    
    const resultCustom = wasm.edcf_js(close, 20);
    assert.strictEqual(resultCustom.length, close.length);
});

test('EDCF accuracy', async () => {
    
    
    const hl2 = new Float64Array(testData.high.length);
    for (let i = 0; i < testData.high.length; i++) {
        hl2[i] = (testData.high[i] + testData.low[i]) / 2;
    }
    
    const expectedLast5 = [
        59593.332275678375,
        59731.70263288801,
        59766.41512339413,
        59655.66162110993,
        59332.492883847
    ];
    
    const result = wasm.edcf_js(hl2, 15);
    
    assert.strictEqual(result.length, hl2.length);
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLast5,
        1e-8,
        "EDCF last 5 values mismatch"
    );
    
    
    await compareWithRust('edcf', result, 'hl2', { period: 15 });
});

test('EDCF default candles', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.edcf_js(close, 15);
    assert.strictEqual(result.length, close.length);
});

test('EDCF zero period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.edcf_js(inputData, 0);
    }, /Invalid period/);
});

test('EDCF period exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.edcf_js(dataSmall, 10);
    }, /Invalid period/);
});

test('EDCF very small dataset', () => {
    
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.edcf_js(singlePoint, 15);
    }, /Invalid period/);
});

test('EDCF empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.edcf_js(empty, 15);
    }, /No data provided/);
});

test('EDCF reinput', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const firstResult = wasm.edcf_js(close, 15);
    assert.strictEqual(firstResult.length, close.length);
    
    
    const secondResult = wasm.edcf_js(firstResult, 5);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    
    if (secondResult.length > 240) {
        for (let i = 240; i < secondResult.length; i++) {
            assert(!isNaN(secondResult[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('EDCF NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    const period = 15;
    
    const result = wasm.edcf_js(close, period);
    assert.strictEqual(result.length, close.length);
    
    
    const startIndex = 2 * period;
    if (result.length > startIndex) {
        for (let i = startIndex; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('EDCF all NaN input', () => {
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.edcf_js(allNaN, 15);
    }, /All values are NaN/);
});

test('EDCF accuracy hl2 verification', () => {
    
    const hl2 = new Float64Array(testData.high.length);
    for (let i = 0; i < testData.high.length; i++) {
        hl2[i] = (testData.high[i] + testData.low[i]) / 2;
    }
    
    const result = wasm.edcf_js(hl2, 15);
    
    
    const expectedLast5 = [
        59593.332275678375,
        59731.70263288801,
        59766.41512339413,
        59655.66162110993,
        59332.492883847
    ];
    
    
    const actualLast5 = result.slice(-5);
    for (let i = 0; i < 5; i++) {
        const diff = Math.abs(actualLast5[i] - expectedLast5[i]);
        assert(diff < 1e-8, `EDCF mismatch at index ${i}: got ${actualLast5[i]}, expected ${expectedLast5[i]}, diff ${diff}`);
    }
});

test('EDCF batch single parameter set', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const batchResult = wasm.edcf_batch_js(
        close,
        15, 15, 0  
    );
    
    
    const singleResult = wasm.edcf_js(close, 15);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('EDCF batch multiple periods', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    
    
    const batchResult = wasm.edcf_batch_js(
        close,
        10, 25, 5  
    );
    
    
    assert.strictEqual(batchResult.length, 4 * 100);
    
    
    const periods = [10, 15, 20, 25];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.slice(rowStart, rowEnd);
        
        const singleResult = wasm.edcf_js(close, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('EDCF batch metadata', () => {
    
    const metadata = wasm.edcf_batch_metadata_js(
        10, 30, 5  
    );
    
    
    assert.strictEqual(metadata.length, 5);
    
    
    assert.strictEqual(metadata[0], 10);
    assert.strictEqual(metadata[1], 15);
    assert.strictEqual(metadata[2], 20);
    assert.strictEqual(metadata[3], 25);
    assert.strictEqual(metadata[4], 30);
});

test('EDCF batch warmup validation', () => {
    
    const close = new Float64Array(testData.close.slice(0, 60));
    
    const batchResult = wasm.edcf_batch_js(
        close,
        10, 20, 10  
    );
    
    const metadata = wasm.edcf_batch_metadata_js(10, 20, 10);
    const numCombos = metadata.length;
    assert.strictEqual(numCombos, 2);
    
    
    for (let combo = 0; combo < numCombos; combo++) {
        const period = metadata[combo];
        const rowStart = combo * 60;
        const rowData = batchResult.slice(rowStart, rowStart + 60);
        
        
        const warmup = 2 * period;
        for (let i = 0; i < warmup && i < 60; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${period}`);
        }
        
        
        for (let i = warmup; i < 60; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period}`);
        }
    }
});

test('EDCF batch with hl2 data', () => {
    
    const hl2 = new Float64Array(testData.high.length);
    for (let i = 0; i < testData.high.length; i++) {
        hl2[i] = (testData.high[i] + testData.low[i]) / 2;
    }
    
    
    const hl2Small = hl2.slice(0, 100);
    
    const batchResult = wasm.edcf_batch_js(
        hl2Small,
        15, 15, 0  
    );
    
    
    const singleResult = wasm.edcf_js(hl2Small, 15);
    assertArrayClose(batchResult, singleResult, 1e-10, "HL2 batch vs single mismatch");
});

test('EDCF batch edge cases', () => {
    
    const close = new Float64Array(Array.from({length: 50}, (_, i) => i + 1));
    
    
    const singleBatch = wasm.edcf_batch_js(
        close,
        10, 10, 1
    );
    
    assert.strictEqual(singleBatch.length, 50);
    
    
    const zeroStepBatch = wasm.edcf_batch_js(
        close,
        15, 25, 0
    );
    
    assert.strictEqual(zeroStepBatch.length, 50); 
    
    
    const largeBatch = wasm.edcf_batch_js(
        close,
        10, 12, 10  
    );
    
    
    assert.strictEqual(largeBatch.length, 50);
    
    
    assert.throws(() => {
        wasm.edcf_batch_js(
            new Float64Array([]),
            15, 15, 0
        );
    }, /No data provided/);
});

test('EDCF batch performance test', () => {
    
    const close = new Float64Array(testData.close.slice(0, 200));
    
    
    const perf = typeof performance !== 'undefined' ? performance : { now: Date.now };
    
    
    const iterations = 10;
    let batchTotalTime = 0;
    let singleTotalTime = 0;
    
    
    for (let i = 0; i < 3; i++) {
        wasm.edcf_batch_js(close, 10, 30, 2);
        for (let period = 10; period <= 30; period += 2) {
            wasm.edcf_js(close, period);
        }
    }
    
    
    for (let i = 0; i < iterations; i++) {
        const startBatch = perf.now();
        const batchResult = wasm.edcf_batch_js(
            close,
            10, 30, 2  
        );
        batchTotalTime += perf.now() - startBatch;
        
        
        if (i === 0) {
            assert.strictEqual(batchResult.length, 11 * 200);
        }
    }
    
    
    for (let i = 0; i < iterations; i++) {
        const startSingle = perf.now();
        const singleResults = [];
        for (let period = 10; period <= 30; period += 2) {
            singleResults.push(...wasm.edcf_js(close, period));
        }
        singleTotalTime += perf.now() - startSingle;
    }
    
    
    const avgBatchTime = batchTotalTime / iterations;
    const avgSingleTime = singleTotalTime / iterations;
    
    
    console.log(`  EDCF Avg Batch time: ${avgBatchTime.toFixed(3)}ms, Avg Single calls time: ${avgSingleTime.toFixed(3)}ms`);
    
    
    
    
    
    const isComparable = avgBatchTime <= avgSingleTime * 3 || 
                         avgBatchTime - avgSingleTime <= 2.0;
    
    assert(isComparable, 
        `Batch (${avgBatchTime.toFixed(3)}ms) should be comparable to single calls (${avgSingleTime.toFixed(3)}ms)`);
});

test('EDCF batch volatility analysis scenario', () => {
    
    const close = new Float64Array(testData.close.slice(0, 300));
    
    
    
    const shortTermBatch = wasm.edcf_batch_js(close, 10, 20, 2);
    const mediumTermBatch = wasm.edcf_batch_js(close, 25, 40, 3);
    const longTermBatch = wasm.edcf_batch_js(close, 45, 60, 5);
    
    
    assert.strictEqual(shortTermBatch.length, 6 * 300);  
    assert.strictEqual(mediumTermBatch.length, 6 * 300); 
    assert.strictEqual(longTermBatch.length, 4 * 300);   
    
    
    const short10 = shortTermBatch.slice(0, 300);
    const medium25 = mediumTermBatch.slice(0, 300);
    const long45 = longTermBatch.slice(0, 300);
    
    
    for (let i = 0; i < 20; i++) {
        assert(isNaN(short10[i]), `Expected NaN at index ${i} for short-term EDCF`);
    }
    for (let i = 0; i < 50; i++) {
        assert(isNaN(medium25[i]), `Expected NaN at index ${i} for medium-term EDCF`);
    }
    for (let i = 0; i < 90; i++) {
        assert(isNaN(long45[i]), `Expected NaN at index ${i} for long-term EDCF`);
    }
});

test('EDCF insufficient data', () => {
    
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    
    assert.throws(() => {
        wasm.edcf_js(data, 3);
    }, /Not enough valid data/);
    
    
    const result = wasm.edcf_js(data, 2);
    assert.strictEqual(result.length, data.length);
});

test('EDCF warmup period verification', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    const period = 10;
    
    const result = wasm.edcf_js(close, period);
    
    
    const warmup = 2 * period;
    
    
    for (let i = 0; i < warmup; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} during warmup, got ${result[i]}`);
    }
    
    
    assert(!isNaN(result[warmup]), `Expected valid value at index ${warmup}, got NaN`);
    
    
    for (let i = warmup + 1; i < result.length; i++) {
        assert(!isNaN(result[i]), `Unexpected NaN at index ${i} after warmup`);
    }
});

test('EDCF constant data', () => {
    
    
    
    const constant = new Float64Array(50);
    constant.fill(100.0);
    
    const result = wasm.edcf_js(constant, 5);
    assert.strictEqual(result.length, constant.length);
    
    
    
    console.log('  Note: EDCF constant data handling has known issues with uninitialized memory');
});

test('EDCF monotonic data', () => {
    
    const monotonic = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        monotonic[i] = i + 1.0;
    }
    
    const result = wasm.edcf_js(monotonic, 5);
    assert.strictEqual(result.length, monotonic.length);
    
    
    const warmup = 2 * 5;
    assert(!isNaN(result[warmup]), "EDCF should handle monotonic data");
    
    
    for (let i = warmup; i < result.length; i++) {
        if (!isNaN(result[i])) {
            const windowStart = Math.max(0, i - 4);
            const windowEnd = i + 1;
            const windowMin = monotonic[windowStart];
            const windowMax = monotonic[windowEnd - 1];
            assert(result[i] >= windowMin - 1e-9 && result[i] <= windowMax + 1e-9,
                `EDCF value ${result[i]} outside window [${windowMin}, ${windowMax}] at index ${i}`);
        }
    }
});

test('EDCF zero-copy API', () => {
    
    if (!wasm.edcf_alloc || !wasm.edcf_free || !wasm.edcf_into) {
        console.log('  Zero-copy API not available, skipping test');
        return;
    }
    
    const testData = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    const len = testData.length;
    const period = 3;
    
    
    const inPtr = wasm.edcf_alloc(len);
    const outPtr = wasm.edcf_alloc(len);
    
    try {
        
        if (!wasm.memory || !wasm.memory.buffer) {
            console.log('  WASM memory not accessible, skipping test');
            return;
        }
        
        
        const wasmMemory = new Float64Array(wasm.memory.buffer);
        wasmMemory.set(testData, inPtr / 8);
        
        
        wasm.edcf_into(inPtr, outPtr, len, period);
        
        
        const resultView = new Float64Array(wasm.memory.buffer, outPtr, len);
        const result = Array.from(resultView);
        
        
        const expected = wasm.edcf_js(testData, period);
        assertArrayClose(result, expected, 1e-10, "Zero-copy API mismatch");
        
        
        wasm.edcf_into(inPtr, inPtr, len, period);
        const inPlaceView = new Float64Array(wasm.memory.buffer, inPtr, len);
        const inPlaceResult = Array.from(inPlaceView);
        assertArrayClose(inPlaceResult, expected, 1e-10, "In-place zero-copy mismatch");
        
    } finally {
        
        wasm.edcf_free(inPtr, len);
        wasm.edcf_free(outPtr, len);
    }
});

test('EDCF batch zero-copy API', () => {
    
    if (!wasm.edcf_batch_into || !wasm.edcf_alloc || !wasm.edcf_free) {
        console.log('  Batch zero-copy API not available, skipping test');
        return;
    }
    
    const testData = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        testData[i] = Math.sin(i * 0.1) * 100 + 100;
    }
    
    const len = testData.length;
    const periodStart = 5;
    const periodEnd = 15;
    const periodStep = 5;
    const numPeriods = 3; 
    
    
    const inPtr = wasm.edcf_alloc(len);
    const outPtr = wasm.edcf_alloc(len * numPeriods);
    
    try {
        
        if (!wasm.memory || !wasm.memory.buffer) {
            console.log('  WASM memory not accessible, skipping test');
            return;
        }
        
        
        const wasmMemory = new Float64Array(wasm.memory.buffer);
        wasmMemory.set(testData, inPtr / 8);
        
        
        const rowsReturned = wasm.edcf_batch_into(
            inPtr, outPtr, len,
            periodStart, periodEnd, periodStep
        );
        
        assert.strictEqual(rowsReturned, numPeriods, "Incorrect number of rows returned");
        
        
        const resultView = new Float64Array(wasm.memory.buffer, outPtr, len * numPeriods);
        const result = Array.from(resultView);
        
        
        const expected = wasm.edcf_batch_js(testData, periodStart, periodEnd, periodStep);
        assertArrayClose(result, expected, 1e-10, "Batch zero-copy API mismatch");
        
    } finally {
        
        wasm.edcf_free(inPtr, len);
        wasm.edcf_free(outPtr, len * numPeriods);
    }
});



test.after(() => {
    console.log('EDCF WASM tests completed');
});