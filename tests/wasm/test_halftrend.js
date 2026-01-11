/**
 * WASM binding tests for HALFTREND indicator.
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


let wasm;
let testData;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

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

test('HALFTREND partial params', () => {
    
    const { high, low, close } = testData;
    const expected = EXPECTED_OUTPUTS.halftrend;
    
    const result = wasm.halftrend(
        high, low, close,
        expected.defaultParams.amplitude,
        expected.defaultParams.channelDeviation,
        expected.defaultParams.atrPeriod
    );
    
    assert(result.values, 'Should have values output');
    assert.strictEqual(result.cols, high.length);
    assert.strictEqual(result.rows, 6);
});

test('HALFTREND accuracy', async () => {
    
    const { high, low, close } = testData;
    const expected = EXPECTED_OUTPUTS.halftrend;
    
    const result = wasm.halftrend(
        high, low, close,
        expected.defaultParams.amplitude,
        expected.defaultParams.channelDeviation,
        expected.defaultParams.atrPeriod
    );
    
    
    assert(result.values, 'Should have values output');
    assert.strictEqual(result.rows, 6, 'Should have 6 rows');
    assert.strictEqual(result.cols, high.length, 'Cols should match input length');
    assert.strictEqual(result.values.length, 6 * high.length, 'Values array should be flattened');
    
    
    const cols = result.cols;
    const halftrend = result.values.slice(0, cols);
    const trend = result.values.slice(cols, 2 * cols);
    const atr_high = result.values.slice(2 * cols, 3 * cols);
    const atr_low = result.values.slice(3 * cols, 4 * cols);
    const buy_signal = result.values.slice(4 * cols, 5 * cols);
    const sell_signal = result.values.slice(5 * cols, 6 * cols);
    
    
    const testIndices = expected.testIndices;
    const expectedHalftrend = expected.expectedHalftrend;
    const expectedTrend = expected.expectedTrend;
    
    testIndices.forEach((idx, i) => {
        assertClose(
            halftrend[idx], 
            expectedHalftrend[i], 
            1.0,
            `HalfTrend mismatch at index ${idx}`
        );
        assertClose(
            trend[idx], 
            expectedTrend[i], 
            0.01,
            `Trend mismatch at index ${idx}`
        );
    });
    
    
    
});

test('HALFTREND default candles', () => {
    
    const { high, low, close } = testData;
    
    
    const result = wasm.halftrend(high, low, close, 2, 2.0, 100);
    
    assert(result.values, 'Should have values output');
    assert.strictEqual(result.cols, high.length, 'Output length should match input');
    assert.strictEqual(result.rows, 6, 'Should have 6 output series');
});

test('HALFTREND zero amplitude', () => {
    
    const inputData = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    assert.throws(() => {
        wasm.halftrend(inputData, inputData, inputData, 0, 2.0, 100);
    }, /Invalid period.*period = 0/);
});

test('HALFTREND period exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.halftrend(dataSmall, dataSmall, dataSmall, 10, 2.0, 100);
    }, /Invalid period.*period = 100/);
});

test('HALFTREND very small dataset', () => {
    
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.halftrend(singlePoint, singlePoint, singlePoint, 2, 2.0, 100);
    }, /Invalid period.*period = 100|Not enough valid data/);
});

test('HALFTREND empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.halftrend(empty, empty, empty, 2, 2.0, 100);
    }, /Empty input data|Input data slice is empty/);
});

test('HALFTREND invalid channel deviation', () => {
    
    const data = new Float64Array([1.0, 2.0, 3.0]);
    
    
    assert.throws(() => {
        wasm.halftrend(data, data, data, 2, 0.0, 100);
    }, /Invalid channel_deviation/);
    
    
    assert.throws(() => {
        wasm.halftrend(data, data, data, 2, -1.0, 100);
    }, /Invalid channel_deviation/);
});

test('HALFTREND invalid ATR period', () => {
    
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    assert.throws(() => {
        wasm.halftrend(data, data, data, 2, 2.0, 0);
    }, /Invalid period.*period = 0/);
});

test('HALFTREND NaN handling', () => {
    
    const { high, low, close } = testData;
    const expected = EXPECTED_OUTPUTS.halftrend;
    
    const result = wasm.halftrend(high, low, close, 2, 2.0, 100);
    assert.strictEqual(result.cols, high.length);
    
    
    const halftrend = result.values.slice(0, result.cols);
    const trend = result.values.slice(result.cols, 2 * result.cols);
    
    
    const warmupPeriod = expected.warmupPeriod;
    for (let i = 0; i < warmupPeriod; i++) {
        assert(isNaN(halftrend[i]), `Expected NaN at warmup index ${i}`);
        assert(isNaN(trend[i]), `Expected NaN in trend at warmup index ${i}`);
    }
    
    
    if (halftrend.length > warmupPeriod + 10) {
        for (let i = warmupPeriod; i < warmupPeriod + 10; i++) {
            assert(!isNaN(halftrend[i]), `Unexpected NaN at index ${i}`);
            assert(!isNaN(trend[i]), `Unexpected NaN in trend at index ${i}`);
        }
    }
});

test('HALFTREND all NaN input', () => {
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.halftrend(allNaN, allNaN, allNaN, 2, 2.0, 100);
    }, /All values are NaN|All NaN|All input values/);
});

test('HALFTREND mismatched array lengths', () => {
    
    const high = new Float64Array([1, 2, 3]);
    const low = new Float64Array([1, 2]);
    const close = new Float64Array([1, 2, 3]);
    
    assert.throws(() => {
        wasm.halftrend(high, low, close, 2, 2.0, 100);
    }, /Mismatched|lengths|size|Not enough valid/);
});

test('HALFTREND custom params', () => {
    
    const { high, low, close } = testData;
    
    
    const result = wasm.halftrend(
        high, low, close,
        3,      
        2.5,    
        50      
    );
    
    
    assert(result.values, 'Should have values output');
    assert.strictEqual(result.cols, high.length, 'Cols should match input length');
    assert.strictEqual(result.rows, 6, 'Should have 6 output series');
    
    
    const halftrend = result.values.slice(0, result.cols);
    
    
    const warmupPeriod = 49;
    let nonNanCount = 0;
    for (let i = warmupPeriod; i < Math.min(halftrend.length, warmupPeriod + 100); i++) {
        if (!isNaN(halftrend[i])) {
            nonNanCount++;
        }
    }
    assert(nonNanCount > 0, 'Should have non-NaN values after warmup period');
});

test('HALFTREND warmup period verification', () => {
    
    const data = new Float64Array(500);
    for (let i = 0; i < 500; i++) {
        data[i] = 100 + Math.sin(i * 0.1) * 10;
    }
    
    const testCases = [
        { amplitude: 2, atrPeriod: 100, expectedWarmup: 99 },  
        { amplitude: 50, atrPeriod: 20, expectedWarmup: 49 },  
        { amplitude: 10, atrPeriod: 10, expectedWarmup: 9 },   
    ];
    
    for (const testCase of testCases) {
        const result = wasm.halftrend(
            data, data, data,
            testCase.amplitude,
            2.0,
            testCase.atrPeriod
        );
        
        
        const halftrend = result.values.slice(0, result.cols);
        
        
        let nanCount = 0;
        for (let i = 0; i <= testCase.expectedWarmup; i++) {
            if (isNaN(halftrend[i])) {
                nanCount++;
            }
        }
        assert(
            nanCount >= testCase.expectedWarmup,
            `Expected at least ${testCase.expectedWarmup} NaN values for amplitude=${testCase.amplitude}, atrPeriod=${testCase.atrPeriod}`
        );
    }
});

test('HALFTREND signal detection', () => {
    
    const { high, low, close } = testData;
    
    const result = wasm.halftrend(high, low, close, 2, 2.0, 100);
    
    
    const buy_signal = result.values.slice(4 * result.cols, 5 * result.cols);
    const sell_signal = result.values.slice(5 * result.cols, 6 * result.cols);
    
    
    let buyNanCount = 0;
    let sellNanCount = 0;
    
    for (let i = 0; i < buy_signal.length; i++) {
        if (isNaN(buy_signal[i])) buyNanCount++;
        if (isNaN(sell_signal[i])) sellNanCount++;
    }
    
    
    assert(buyNanCount > high.length * 0.95, 'Buy signals should be sparse (mostly NaN)');
    assert(sellNanCount > high.length * 0.95, 'Sell signals should be sparse (mostly NaN)');
    
    
    const buySignalCount = buy_signal.length - buyNanCount;
    const sellSignalCount = sell_signal.length - sellNanCount;
    
    
    assert(buySignalCount > 0, 'Should have at least one buy signal');
    assert(sellSignalCount > 0, 'Should have at least one sell signal');
});


test('HALFTREND batch single parameter set', () => {
    
    const { high, low, close } = testData;
    const expected = EXPECTED_OUTPUTS.halftrend;
    
    
    const batchResult = wasm.halftrend_batch(high, low, close, {
        amplitude_range: [expected.defaultParams.amplitude, expected.defaultParams.amplitude, 0],
        channel_deviation_range: [expected.defaultParams.channelDeviation, expected.defaultParams.channelDeviation, 0],
        atr_period_range: [expected.defaultParams.atrPeriod, expected.defaultParams.atrPeriod, 0]
    });
    
    
    assert(batchResult.values, 'Should have values array');
    assert.strictEqual(batchResult.rows, 6, 'Should have 6 output series');
    assert.strictEqual(batchResult.cols, high.length, 'Cols should match input length');
    assert.strictEqual(batchResult.values.length, 6 * high.length, 'Values should be flattened');
    
    
    const halftrend = batchResult.values.slice(0, batchResult.cols);
    const trend = batchResult.values.slice(batchResult.cols, 2 * batchResult.cols);
    
    
    const testIndices = expected.testIndices;
    const expectedHalftrend = expected.expectedHalftrend;
    const expectedTrend = expected.expectedTrend;
    
    testIndices.forEach((idx, i) => {
        assertClose(
            halftrend[idx],
            expectedHalftrend[i],
            1.0,
            `Batch halftrend mismatch at index ${idx}`
        );
        assertClose(
            trend[idx],
            expectedTrend[i],
            0.01,
            `Batch trend mismatch at index ${idx}`
        );
    });
});

test('HALFTREND batch multiple parameters', () => {
    
    const { high, low, close } = testData;
    const testHigh = high.slice(0, 100);
    const testLow = low.slice(0, 100);
    const testClose = close.slice(0, 100);
    
    const batchResult = wasm.halftrend_batch(testHigh, testLow, testClose, {
        amplitude_range: [2, 4, 1],  
        channel_deviation_range: [2.0, 2.5, 0.5],  
        atr_period_range: [50, 50, 0]  
    });
    
    
    const expectedCombos = 6;
    assert.strictEqual(batchResult.combos ? batchResult.combos.length : (batchResult.rows / 6), expectedCombos);
    assert.strictEqual(batchResult.cols, 100, 'Cols should match input length');
    
    
    assert(batchResult.values, 'Should have values array');
    assert(batchResult.values.length > 0, 'Values array should not be empty');
});


test('HALFTREND zero-copy API', () => {
    const size = 100;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = 100 + Math.sin(i * 0.1) * 10;
    }
    
    
    const highPtr = wasm.halftrend_alloc(size);
    const lowPtr = wasm.halftrend_alloc(size);
    const closePtr = wasm.halftrend_alloc(size);
    const outPtr = wasm.halftrend_alloc(size * 6); 
    
    try {
        assert(highPtr !== 0, 'Failed to allocate high buffer');
        assert(lowPtr !== 0, 'Failed to allocate low buffer');
        assert(closePtr !== 0, 'Failed to allocate close buffer');
        assert(outPtr !== 0, 'Failed to allocate output buffer');
        
        
        const memoryBuffer = wasm.__wasm.memory.buffer;
        const highView = new Float64Array(memoryBuffer, highPtr, size);
        const lowView = new Float64Array(memoryBuffer, lowPtr, size);
        const closeView = new Float64Array(memoryBuffer, closePtr, size);
        
        
        highView.set(data);
        lowView.set(data);
        closeView.set(data);
        
        
        wasm.halftrend_into(
            highPtr, lowPtr, closePtr,
            outPtr,
            size, 2, 2.0, 50
        );
        
        
        const htView = new Float64Array(memoryBuffer, outPtr, size);
        
        
        let hasNaN = false;
        for (let i = 0; i < 49; i++) {
            if (isNaN(htView[i])) hasNaN = true;
        }
        assert(hasNaN, 'Should have NaN in warmup period');
        
        
        let hasValues = false;
        for (let i = 50; i < Math.min(60, size); i++) {
            if (!isNaN(htView[i])) hasValues = true;
        }
        assert(hasValues, 'Should have values after warmup');
    } finally {
        
        wasm.halftrend_free(highPtr, size);
        wasm.halftrend_free(lowPtr, size);
        wasm.halftrend_free(closePtr, size);
        wasm.halftrend_free(outPtr, size * 6);
    }
});


test('HALFTREND reinput', () => {
    
    const { high, low, close } = testData;
    
    
    const firstResult = wasm.halftrend(high, low, close, 2, 2.0, 100);
    assert(firstResult.values, 'First pass should produce output');
    
    
    const halftrend1 = firstResult.values.slice(0, firstResult.cols);
    
    
    const secondResult = wasm.halftrend(halftrend1, halftrend1, halftrend1, 2, 2.0, 100);
    assert(secondResult.values, 'Second pass should produce output');
    
    
    const halftrend2 = secondResult.values.slice(0, secondResult.cols);
    
    
    assert.strictEqual(halftrend2.length, halftrend1.length, 'Length should be preserved');
    
    
    let differences = 0;
    for (let i = 200; i < Math.min(300, halftrend2.length); i++) {
        if (!isNaN(halftrend1[i]) && !isNaN(halftrend2[i])) {
            if (Math.abs(halftrend1[i] - halftrend2[i]) > 1e-10) {
                differences++;
            }
        }
    }
    assert(differences > 0, 'Reinput should produce different values due to re-smoothing');
});


test('HALFTREND not enough valid data', () => {
    
    const n = 10;
    const highData = new Float64Array(n);
    const lowData = new Float64Array(n);
    const closeData = new Float64Array(n);
    
    
    highData.fill(NaN);
    lowData.fill(NaN);
    closeData.fill(NaN);
    highData[5] = 1.0;
    lowData[5] = 1.0;
    closeData[5] = 1.0;
    
    assert.throws(() => {
        wasm.halftrend(highData, lowData, closeData, 9, 2.0, 9);
    }, /Not enough valid data/);
});

test('HALFTREND batch metadata verification', () => {
    
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = 100 + Math.sin(i * 0.1) * 10;
    }
    
    const result = wasm.halftrend_batch(data, data, data, {
        amplitude_range: [2, 4, 1],           
        channel_deviation_range: [2.0, 2.5, 0.5],  
        atr_period_range: [10, 20, 10]        
    });
    
    
    if (result.combos) {
        assert.strictEqual(result.combos.length, 12, 'Should have 12 parameter combinations');
        
        
        assert.strictEqual(result.combos[0].amplitude, 2);
        assert.strictEqual(result.combos[0].channel_deviation, 2.0);
        assert.strictEqual(result.combos[0].atr_period, 10);
        
        
        assert.strictEqual(result.combos[11].amplitude, 4);
        assertClose(result.combos[11].channel_deviation, 2.5, 1e-10, 'channel_deviation mismatch');
        assert.strictEqual(result.combos[11].atr_period, 20);
    }
});

test('HALFTREND zero-copy error handling', () => {
    
    assert.throws(() => {
        wasm.halftrend_into(0, 0, 0, 0, 10, 2, 2.0, 100);
    }, /null pointer|invalid memory/i);
    
    
    const ptr = wasm.halftrend_alloc(10);
    const out_ptr = wasm.halftrend_alloc(60); 
    try {
        
        assert.throws(() => {
            wasm.halftrend_into(
                ptr, ptr, ptr,
                out_ptr,
                10, 0, 2.0, 100
            );
        }, /Invalid period/);
        
        
        assert.throws(() => {
            wasm.halftrend_into(
                ptr, ptr, ptr,
                out_ptr,
                10, 2, 0.0, 100
            );
        }, /Invalid channel_deviation/);
    } finally {
        wasm.halftrend_free(ptr, 10);
        wasm.halftrend_free(out_ptr, 60);
    }
});

test('HALFTREND memory management stress test', () => {
    
    const sizes = [100, 1000, 10000];
    
    for (const size of sizes) {
        
        const buffers = [];
        for (let i = 0; i < 9; i++) {  
            const ptr = wasm.halftrend_alloc(size);
            assert(ptr !== 0, `Failed to allocate buffer ${i} of size ${size}`);
            buffers.push(ptr);
        }
        
        
        const memoryBuffer = wasm.__wasm.memory.buffer;
        const view = new Float64Array(memoryBuffer, buffers[0], size);
        for (let i = 0; i < Math.min(10, size); i++) {
            view[i] = i * 2.5;
        }
        
        
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(view[i], i * 2.5, `Memory corruption at index ${i}`);
        }
        
        
        for (const ptr of buffers) {
            wasm.halftrend_free(ptr, size);
        }
    }
});

test('HALFTREND SIMD128 consistency', () => {
    
    const testCases = [
        { size: 10, amplitude: 2, atrPeriod: 5 },
        { size: 100, amplitude: 5, atrPeriod: 20 },
        { size: 1000, amplitude: 10, atrPeriod: 50 }
    ];
    
    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = 100 + Math.sin(i * 0.1) * 10 + Math.cos(i * 0.05) * 5;
        }
        
        const result = wasm.halftrend(
            data, data, data,
            testCase.amplitude, 2.0, testCase.atrPeriod
        );
        
        
        assert(result, `Result should not be null for size=${testCase.size}`);
        assert(result.values, `Result should have values for size=${testCase.size}`);
        
        
        assert.strictEqual(result.cols, data.length);
        assert.strictEqual(result.rows, 6);
        
        
        const halftrend = result.values.slice(0, result.cols);
        
        
        const warmup = Math.max(testCase.amplitude, testCase.atrPeriod) - 1;
        for (let i = 0; i < Math.min(warmup, halftrend.length); i++) {
            assert(isNaN(halftrend[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}`);
        }
        
        
        let validCount = 0;
        for (let i = warmup; i < halftrend.length; i++) {
            if (!isNaN(halftrend[i])) {
                validCount++;
            }
        }
        assert(validCount > 0, `Should have valid values after warmup for size=${testCase.size}`);
    }
});

test.after(() => {
    console.log('HALFTREND WASM tests completed');
});
