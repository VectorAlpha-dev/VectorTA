/**
 * WASM binding tests for CCI indicator.
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

test('CCI partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.cci_js(close, 14);
    assert.strictEqual(result.length, close.length);
    
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    
    const hl2 = new Float64Array(high.length);
    for (let i = 0; i < high.length; i++) {
        hl2[i] = (high[i] + low[i]) / 2;
    }
    
    const resultHl2 = wasm.cci_js(hl2, 20);
    assert.strictEqual(resultHl2.length, hl2.length);
    
    
    const hlc3 = new Float64Array(high.length);
    for (let i = 0; i < high.length; i++) {
        hlc3[i] = (high[i] + low[i] + close[i]) / 3;
    }
    
    const resultHlc3 = wasm.cci_js(hlc3, 9);
    assert.strictEqual(resultHlc3.length, hlc3.length);
});

test('CCI accuracy', async () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.cci;
    
    
    const hlc3 = new Float64Array(high.length);
    for (let i = 0; i < high.length; i++) {
        hlc3[i] = (high[i] + low[i] + close[i]) / 3;
    }
    
    const result = wasm.cci_js(
        hlc3,
        expected.defaultParams.period
    );
    
    assert.strictEqual(result.length, hlc3.length);
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-6,
        "CCI last 5 values mismatch"
    );
    
    
    const period = expected.defaultParams.period;
    for (let i = 0; i < period - 1; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} for initial period warm-up`);
    }
    
    
    await compareWithRust('cci', result, 'hlc3', expected.defaultParams);
});

test('CCI default candles', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    
    const hlc3 = new Float64Array(high.length);
    for (let i = 0; i < high.length; i++) {
        hlc3[i] = (high[i] + low[i] + close[i]) / 3;
    }
    
    const result = wasm.cci_js(hlc3, 14);
    assert.strictEqual(result.length, hlc3.length);
});

test('CCI zero period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.cci_js(inputData, 0);
    }, /Invalid period/);
});

test('CCI period exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.cci_js(dataSmall, 10);
    }, /Invalid period/);
});

test('CCI very small dataset', () => {
    
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.cci_js(singlePoint, 9);
    }, /Invalid period|Not enough valid data/);
});

test('CCI empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.cci_js(empty, 14);
    }, /Input data slice is empty/);
});

test('CCI reinput', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const firstResult = wasm.cci_js(close, 14);
    assert.strictEqual(firstResult.length, close.length);
    
    
    const secondResult = wasm.cci_js(firstResult, 14);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    
    if (secondResult.length > 28) {
        for (let i = 28; i < secondResult.length; i++) {
            assert(!isNaN(secondResult[i]), `Expected no NaN after index 28, found NaN at index ${i}`);
        }
    }
});

test('CCI NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.cci_js(close, 14);
    assert.strictEqual(result.length, close.length);
    
    
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    
    assertAllNaN(result.slice(0, 13), "Expected NaN in warmup period");
});

test('CCI all NaN input', () => {
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.cci_js(allNaN, 14);
    }, /All values are NaN/);
});

test('CCI batch single parameter set', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const batchResult = wasm.cci_batch_js(
        close,
        14, 14, 0      
    );
    
    
    const singleResult = wasm.cci_js(close, 14);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('CCI batch multiple periods', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    
    
    const batchResult = wasm.cci_batch_js(
        close,
        10, 20, 2      
    );
    
    
    assert.strictEqual(batchResult.length, 6 * 100);
    
    
    const periods = [10, 12, 14, 16, 18, 20];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.slice(rowStart, rowEnd);
        
        const singleResult = wasm.cci_js(close, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('CCI batch metadata', () => {
    
    const metadata = wasm.cci_batch_metadata_js(
        10, 20, 2      
    );
    
    
    assert.strictEqual(metadata.length, 6);
    
    
    const expected = [10, 12, 14, 16, 18, 20];
    for (let i = 0; i < expected.length; i++) {
        assert.strictEqual(metadata[i], expected[i]);
    }
});

test('CCI batch full parameter sweep', () => {
    
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.cci_batch_js(
        close,
        10, 14, 2      
    );
    
    const metadata = wasm.cci_batch_metadata_js(
        10, 14, 2
    );
    
    
    const numCombos = metadata.length;
    assert.strictEqual(numCombos, 3);
    assert.strictEqual(batchResult.length, 3 * 50);
    
    
    for (let combo = 0; combo < numCombos; combo++) {
        const period = metadata[combo];
        
        const rowStart = combo * 50;
        const rowData = batchResult.slice(rowStart, rowStart + 50);
        
        
        for (let i = 0; i < period - 1; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${period}`);
        }
        
        
        for (let i = period - 1; i < 50; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period}`);
        }
    }
});

test('CCI batch edge cases', () => {
    
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    
    const singleBatch = wasm.cci_batch_js(
        close,
        5, 5, 1
    );
    
    assert.strictEqual(singleBatch.length, 10);
    
    
    const largeBatch = wasm.cci_batch_js(
        close,
        5, 7, 10 
    );
    
    
    assert.strictEqual(largeBatch.length, 10);
    
    
    assert.throws(() => {
        wasm.cci_batch_js(
            new Float64Array([]),
            14, 14, 0
        );
    }, /Input data slice is empty/);
});


test('CCI batch - new ergonomic API with single parameter', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    
    const hlc3 = new Float64Array(high.length);
    for (let i = 0; i < high.length; i++) {
        hlc3[i] = (high[i] + low[i] + close[i]) / 3;
    }
    
    const result = wasm.cci_batch(hlc3, {
        period_range: [14, 14, 0]
    });
    
    
    assert(result.values, 'Should have values array');
    assert(result.combos, 'Should have combos array');
    assert(typeof result.rows === 'number', 'Should have rows count');
    assert(typeof result.cols === 'number', 'Should have cols count');
    
    
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, hlc3.length);
    assert.strictEqual(result.combos.length, 1);
    assert.strictEqual(result.values.length, hlc3.length);
    
    
    const combo = result.combos[0];
    assert.strictEqual(combo.period, 14);
    
    
    const oldResult = wasm.cci_js(hlc3, 14);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(result.values[i])) {
            continue; 
        }
        assert(Math.abs(oldResult[i] - result.values[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
    
    
    const expected = EXPECTED_OUTPUTS.cci;
    const last5 = result.values.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-6,
        "CCI new API last 5 values mismatch"
    );
});

test('CCI batch - new API with multiple parameters', () => {
    
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const result = wasm.cci_batch(close, {
        period_range: [10, 14, 2]      
    });
    
    
    assert.strictEqual(result.rows, 3);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.combos.length, 3);
    assert.strictEqual(result.values.length, 150);
    
    
    const expectedCombos = [
        { period: 10 },
        { period: 12 },
        { period: 14 }
    ];
    
    for (let i = 0; i < expectedCombos.length; i++) {
        assert.strictEqual(result.combos[i].period, expectedCombos[i].period);
    }
    
    
    const secondRow = result.values.slice(result.cols, 2 * result.cols);
    assert.strictEqual(secondRow.length, 50);
    
    
    const oldResult = wasm.cci_js(close, 10);
    const firstRow = result.values.slice(0, result.cols);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(firstRow[i])) {
            continue; 
        }
        assert(Math.abs(oldResult[i] - firstRow[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});

test('CCI batch - new API matches old API results', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const params = {
        period_range: [12, 18, 3]
    };
    
    
    const oldValues = wasm.cci_batch_js(
        close,
        params.period_range[0], params.period_range[1], params.period_range[2]
    );
    
    
    const newResult = wasm.cci_batch(close, params);
    
    
    assert.strictEqual(oldValues.length, newResult.values.length);
    
    for (let i = 0; i < oldValues.length; i++) {
        if (isNaN(oldValues[i]) && isNaN(newResult.values[i])) {
            continue; 
        }
        assert(Math.abs(oldValues[i] - newResult.values[i]) < 1e-10,
               `Value mismatch at index ${i}: old=${oldValues[i]}, new=${newResult.values[i]}`);
    }
});

test('CCI batch - new API error handling', () => {
    const close = new Float64Array(testData.close.slice(0, 10));
    
    
    assert.throws(() => {
        wasm.cci_batch(close, {
            period_range: [14, 14] 
        });
    }, /Invalid config/);
    
    
    assert.throws(() => {
        wasm.cci_batch(close, {
            
        });
    }, /Invalid config/);
    
    
    assert.throws(() => {
        wasm.cci_batch(close, {
            period_range: "invalid"
        });
    }, /Invalid config/);
});


test('CCI zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    const period = 5;
    
    
    const ptr = wasm.cci_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    
    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        data.length
    );
    
    
    memView.set(data);
    
    
    try {
        wasm.cci_into(ptr, ptr, data.length, period);
        
        
        const regularResult = wasm.cci_js(data, period);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; 
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        
        wasm.cci_free(ptr, data.length);
    }
});

test('CCI zero-copy with large dataset', () => {
    const size = 10000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = 100 + Math.sin(i * 0.01) * 10 + Math.random() * 2;
    }
    
    const ptr = wasm.cci_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');
    
    try {
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        memView.set(data);
        
        wasm.cci_into(ptr, ptr, size, 14);
        
        
        const memView2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        
        
        for (let i = 0; i < 13; i++) {
            assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
        }
        
        
        for (let i = 13; i < Math.min(100, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.cci_free(ptr, size);
    }
});

test('CCI zero-copy error handling', () => {
    
    assert.throws(() => {
        wasm.cci_into(0, 0, 10, 14);
    }, /null pointer/i);
    
    
    const ptr = wasm.cci_alloc(10);
    try {
        
        assert.throws(() => {
            wasm.cci_into(ptr, ptr, 10, 0);
        }, /Invalid period/);
        
        
        assert.throws(() => {
            wasm.cci_into(ptr, ptr, 10, 20);
        }, /Invalid period/);
    } finally {
        wasm.cci_free(ptr, 10);
    }
});

test('CCI zero-copy memory management', () => {
    
    const sizes = [100, 1000, 5000];
    
    for (const size of sizes) {
        const ptr = wasm.cci_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 2.5;
        }
        
        
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 2.5, `Memory corruption at index ${i}`);
        }
        
        
        wasm.cci_free(ptr, size);
    }
});

test('CCI batch zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]);
    const periodStart = 5;
    const periodEnd = 7;
    const periodStep = 1;
    
    
    const numPeriods = Math.floor((periodEnd - periodStart) / periodStep) + 1;
    const outputSize = numPeriods * data.length;
    
    
    const inPtr = wasm.cci_alloc(data.length);
    const outPtr = wasm.cci_alloc(outputSize);
    
    assert(inPtr !== 0, 'Failed to allocate input buffer');
    assert(outPtr !== 0, 'Failed to allocate output buffer');
    
    try {
        
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, data.length);
        inView.set(data);
        
        
        const rows = wasm.cci_batch_into(inPtr, outPtr, data.length, periodStart, periodEnd, periodStep);
        assert.strictEqual(rows, numPeriods, 'Unexpected number of rows');
        
        
        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, outputSize);
        
        
        const regularResult = wasm.cci_batch_js(data, periodStart, periodEnd, periodStep);
        
        
        assert.strictEqual(regularResult.length, outputSize, 'Length mismatch');
        
        
        for (let i = 0; i < outputSize; i++) {
            const regular = regularResult[i];
            const zeroCopy = outView[i];
            
            if (isNaN(regular) && isNaN(zeroCopy)) {
                continue; 
            }
            
            if (isNaN(regular) || isNaN(zeroCopy)) {
                assert(false, `NaN mismatch at index ${i}: regular=${regular}, zerocopy=${zeroCopy}`);
            }
            
            const diff = Math.abs(regular - zeroCopy);
            assert(diff < 1e-10,
                   `Batch zero-copy mismatch at index ${i}: regular=${regular}, zerocopy=${zeroCopy}, diff=${diff}`);
        }
    } finally {
        wasm.cci_free(inPtr, data.length);
        wasm.cci_free(outPtr, outputSize);
    }
});


test('CCI SIMD128 consistency', () => {
    
    
    const testCases = [
        { size: 20, period: 5 },
        { size: 100, period: 14 },
        { size: 1000, period: 20 },
        { size: 5000, period: 50 }
    ];
    
    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            
            data[i] = 100 + Math.sin(i * 0.1) * 5 + Math.cos(i * 0.05) * 3;
        }
        
        const result = wasm.cci_js(data, testCase.period);
        
        
        assert.strictEqual(result.length, data.length);
        
        
        for (let i = 0; i < testCase.period - 1; i++) {
            assert(isNaN(result[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}`);
        }
        
        
        let sumAfterWarmup = 0;
        let countAfterWarmup = 0;
        for (let i = testCase.period - 1; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for size=${testCase.size}`);
            sumAfterWarmup += result[i];
            countAfterWarmup++;
        }
        
        
        const avgAfterWarmup = sumAfterWarmup / countAfterWarmup;
        assert(Math.abs(avgAfterWarmup) < 200, `Average CCI value ${avgAfterWarmup} seems unreasonable`);
    }
});

test('CCI constant price test', () => {
    
    const data = new Float64Array(100);
    data.fill(50.0);
    
    const result = wasm.cci_js(data, 14);
    
    
    for (let i = 13; i < result.length; i++) {
        assert(Math.abs(result[i]) < 1e-9, 
               `CCI should be ~0 for constant prices, got ${result[i]} at index ${i}`);
    }
});

test('CCI extreme values test', () => {
    
    const largeData = new Float64Array([1e10, 1e10 + 1, 1e10 + 2, 1e10 + 3, 1e10 + 4, 1e10 + 5]);
    const resultLarge = wasm.cci_js(largeData, 3);
    assert.strictEqual(resultLarge.length, largeData.length);
    
    
    for (let i = 0; i < resultLarge.length; i++) {
        if (!isNaN(resultLarge[i])) {
            assert(isFinite(resultLarge[i]), `CCI produced infinite value at index ${i}`);
        }
    }
    
    
    const smallData = new Float64Array([1e-10, 2e-10, 3e-10, 4e-10, 5e-10, 6e-10]);
    const resultSmall = wasm.cci_js(smallData, 3);
    assert.strictEqual(resultSmall.length, smallData.length);
    
    
    for (let i = 0; i < resultSmall.length; i++) {
        if (!isNaN(resultSmall[i])) {
            assert(isFinite(resultSmall[i]), `CCI produced infinite value at index ${i}`);
        }
    }
});





test.after(() => {
    console.log('CCI WASM tests completed');
});