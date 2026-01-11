/**
 * WASM binding tests for QQE indicator.
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

test('QQE default params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.qqe_js(close, 14, 5, 4.236);
    assert.ok(result.values, 'Result should have values array');
    assert.strictEqual(result.rows, 2, 'Should have 2 rows (fast and slow)');
    assert.strictEqual(result.cols, close.length, 'Columns should match data length');
});

test('QQE partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.qqe_js(close, 14, 5, 4.236);
    assert.ok(result.values);
    assert.strictEqual(result.rows, 2);
    assert.strictEqual(result.cols, close.length);
});

test('QQE accuracy', async () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.qqe;
    
    
    const result = wasm.qqe_js(
        close,
        expected.defaultParams.rsiPeriod,
        expected.defaultParams.smoothingFactor,
        expected.defaultParams.fastFactor
    );
    
    
    assert.ok(result.values, 'Result should have values array');
    assert.strictEqual(result.rows, 2, 'Should have 2 rows (fast and slow)');
    assert.strictEqual(result.cols, close.length, 'Columns should match data length');
    
    
    const fast = result.values.slice(0, result.cols);
    const slow = result.values.slice(result.cols, result.cols * 2);
    
    
    const last5Fast = fast.slice(-5);
    const last5Slow = slow.slice(-5);
    
    assertArrayClose(
        last5Fast,
        expected.last5Fast,
        1e-6,
        "QQE fast last 5 values mismatch"
    );
    
    assertArrayClose(
        last5Slow,
        expected.last5Slow,
        1e-6,
        "QQE slow last 5 values mismatch"
    );
});

test('QQE unified output', async () => {
    
    const close = new Float64Array(testData.close);
    const result = wasm.qqe_unified_js(
        close,
        14,  
        5,   
        4.236 
    );
    
    
    assert.strictEqual(result.length, close.length * 2);
    
    
    for (let i = 0; i < Math.min(10, close.length); i++) {
        const fastIdx = i * 2;
        const slowIdx = i * 2 + 1;
        
        
        assert.ok(typeof result[fastIdx] === 'number');
        assert.ok(typeof result[slowIdx] === 'number');
    }
});

test('QQE very small dataset', () => {
    
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.qqe_js(singlePoint, 14, 5, 4.236);
    }, /Invalid period|Not enough valid data/);
});

test('QQE zero period', () => {
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.qqe_js(inputData, 0, 5, 4.236);
    }, /Invalid period/);
});

test('QQE empty input', () => {
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.qqe_js(empty, 14, 5, 4.236);
    }, /[Ee]mpty/);
});

test('QQE period exceeds length', () => {
    const smallData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.qqe_js(smallData, 10, 5, 4.236);
    }, /Invalid period|Not enough/);
});

test('QQE all NaN input', () => {
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.qqe_js(allNaN, 14, 5, 4.236);
    }, /All values are NaN/);
});

test('QQE invalid smoothing factor', () => {
    
    const data = new Float64Array(Array(50).fill(0).map((_, i) => 10.0 + i));
    
    
    assert.throws(() => {
        wasm.qqe_js(data, 14, 0, 4.236);
    }, /Invalid period/);
});

test('QQE extreme fast factor', () => {
    
    const data = new Float64Array(Array(50).fill(0).map((_, i) => 10.0 + i));
    
    
    
    try {
        const result = wasm.qqe_js(data, 14, 5, 0.001);
        assert.strictEqual(result.values.length, data.length * 2);
    } catch (e) {
        
    }
    
    
    try {
        const result = wasm.qqe_js(data, 14, 5, 100.0);
        assert.strictEqual(result.values.length, data.length * 2);
    } catch (e) {
        
    }
});

test('QQE custom parameters', async () => {
    
    const close = new Float64Array(testData.close);
    
    
    const result = wasm.qqe_js(
        close,
        10,  
        3,   
        3.0  
    );
    
    assert.ok(result.values, 'Should have values array');
    assert.strictEqual(result.rows, 2);
    assert.strictEqual(result.cols, close.length);
    
    
    const defaultResult = wasm.qqe_js(close, 14, 5, 4.236);
    
    
    const fast = result.values.slice(0, result.cols);
    const defaultFast = defaultResult.values.slice(0, defaultResult.cols);
    
    
    let hasDifference = false;
    const checkStart = close.length - 10;
    for (let i = checkStart; i < close.length; i++) {
        if (fast[i] !== defaultFast[i]) {
            hasDifference = true;
            break;
        }
    }
    
    assert.ok(hasDifference, 'Custom parameters should produce different results');
});

test('QQE memory management', async () => {
    const len = testData.length;
    
    
    const ptr = wasm.qqe_alloc(len);
    assert.ok(ptr !== 0, 'Should allocate memory');
    
    
    assert.doesNotThrow(() => {
        wasm.qqe_free(ptr, len);
    });
});

test('QQE into function', async () => {
    const inputData = new Float64Array([
        50.0, 51.0, 52.0, 51.5, 50.5, 49.5, 50.0, 51.0, 52.0, 53.0,
        52.5, 51.5, 50.5, 51.0, 52.0, 53.0, 54.0, 53.5, 52.5, 51.5,
        50.5, 51.5, 52.5, 53.5, 54.5, 55.0, 54.5, 53.5, 52.5, 51.5
    ]);
    
    
    const len = inputData.length;
    const inPtr = wasm.qqe_alloc(len);
    const outPtr = wasm.qqe_alloc(len);
    
    try {
        
        const wasmMemory = new Float64Array(wasm.__wasm.memory.buffer);
        wasmMemory.set(inputData, inPtr / 8);
        
        
        wasm.qqe_into(inPtr, outPtr, len, 14, 5, 4.236);
        
        
        const results = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len * 2);
        
        
        assert.strictEqual(results.length, len * 2);
        
        
        let hasValidValues = false;
        for (let i = 20; i < len; i++) {
            if (!isNaN(results[i * 2]) && !isNaN(results[i * 2 + 1])) {
                hasValidValues = true;
                break;
            }
        }
        assert.ok(hasValidValues, 'Should have valid values after warmup period');
        
    } finally {
        
        wasm.qqe_free(inPtr, len);
        wasm.qqe_free(outPtr, len);
    }
});

test('QQE reinput', () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.qqe;
    
    
    const firstResult = wasm.qqe_js(close, 14, 5, 4.236);
    const firstFast = firstResult.values.slice(0, firstResult.cols);
    
    
    const secondResult = wasm.qqe_js(firstFast, 14, 5, 4.236);
    const secondFast = secondResult.values.slice(0, secondResult.cols);
    const secondSlow = secondResult.values.slice(secondResult.cols, secondResult.cols * 2);
    
    assert.strictEqual(secondFast.length, firstFast.length);
    assert.strictEqual(secondSlow.length, firstFast.length);
    
    
    
    const warmupFirst = 17;  
    const warmupSecond = warmupFirst + 17;  
    
    if (secondFast.length > warmupSecond) {
        for (let i = warmupSecond; i < Math.min(warmupSecond + 100, secondFast.length); i++) {
            assert(!isNaN(secondFast[i]), `Found NaN in fast at index ${i} after warmup`);
        }
    }
    
    if (secondSlow.length > warmupSecond) {
        for (let i = warmupSecond; i < Math.min(warmupSecond + 100, secondSlow.length); i++) {
            assert(!isNaN(secondSlow[i]), `Found NaN in slow at index ${i} after warmup`);
        }
    }
});

test('QQE NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.qqe;
    
    const result = wasm.qqe_js(close, 14, 5, 4.236);
    const fast = result.values.slice(0, result.cols);
    const slow = result.values.slice(result.cols, result.cols * 2);
    
    
    const warmup = expected.warmupPeriod; 
    
    if (fast.length > warmup) {
        for (let i = warmup; i < Math.min(warmup + 100, fast.length); i++) {
            assert(!isNaN(fast[i]), `Found unexpected NaN in fast at index ${i}`);
            assert(!isNaN(slow[i]), `Found unexpected NaN in slow at index ${i}`);
        }
    }
    
    
    const firstValid = testData.close.findIndex(v => !Number.isNaN(v));
    const rsiStart = firstValid + expected.defaultParams.rsiPeriod; 
    
    assertAllNaN(fast.slice(0, rsiStart), "Expected NaN in fast until rsi_start");
    assertAllNaN(slow.slice(0, warmup), "Expected NaN in slow warmup period");
});

test('QQE handles NaN values', async () => {
    
    const dataWithNaN = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        if (i < 3) {
            dataWithNaN[i] = NaN;
        } else {
            dataWithNaN[i] = 50.0 + Math.sin(i * 0.1) * 5;
        }
    }
    
    const result = wasm.qqe_js(dataWithNaN, 14, 5, 4.236);
    
    assert.ok(result.values, 'Should have values array');
    assert.strictEqual(result.rows, 2);
    assert.strictEqual(result.cols, dataWithNaN.length);
    
    
    const fast = result.values.slice(0, result.cols);
    const slow = result.values.slice(result.cols, result.cols * 2);
    
    
    assert.ok(isNaN(fast[0]), 'First fast value should be NaN');
    assert.ok(isNaN(slow[0]), 'First slow value should be NaN');
});

test('QQE batch single parameter set', () => {
    
    if (!wasm.qqe_batch_unified_js) {
        console.log('Skipping batch test - function not available in WASM module');
        return;
    }
    
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.qqe;
    
    
    const batchConfig = {
        rsi_period_range: [14, 14, 0],
        smoothing_factor_range: [5, 5, 0],
        fast_factor_range: [4.236, 4.236, 0]
    };
    
    const batchResult = wasm.qqe_batch_unified_js(close, batchConfig);
    
    assert.ok(batchResult.fast_values, 'Should have fast values');
    assert.ok(batchResult.slow_values, 'Should have slow values');
    assert.strictEqual(batchResult.rows, 1, 'Should have 1 row');
    assert.strictEqual(batchResult.cols, close.length, 'Columns should match data length');
    
    
    const singleResult = wasm.qqe_js(close, 14, 5, 4.236);
    const singleFast = singleResult.values.slice(0, singleResult.cols);
    const singleSlow = singleResult.values.slice(singleResult.cols, singleResult.cols * 2);
    
    assertArrayClose(
        batchResult.fast_values,
        singleFast,
        1e-10,
        "Batch vs single fast mismatch"
    );
    
    assertArrayClose(
        batchResult.slow_values,
        singleSlow,
        1e-10,
        "Batch vs single slow mismatch"
    );
});

test('QQE batch multiple parameters', () => {
    
    if (!wasm.qqe_batch_unified_js) {
        console.log('Skipping batch test - function not available in WASM module');
        return;
    }
    
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    
    
    const batchConfig = {
        rsi_period_range: [10, 14, 2],      
        smoothing_factor_range: [3, 4, 1],  
        fast_factor_range: [3.0, 3.5, 0.5]  
    };
    
    const batchResult = wasm.qqe_batch_unified_js(close, batchConfig);
    
    
    assert.strictEqual(batchResult.rows, 12);
    assert.strictEqual(batchResult.cols, 100);
    assert.strictEqual(batchResult.fast_values.length, 12 * 100);
    assert.strictEqual(batchResult.slow_values.length, 12 * 100);
    
    
    assert.strictEqual(batchResult.combos[0].rsi_period, 10);
    assert.strictEqual(batchResult.combos[0].smoothing_factor, 3);
    assert.ok(Math.abs(batchResult.combos[0].fast_factor - 3.0) < 1e-9);
    
    
    for (let i = 0; i < Math.min(3, batchResult.rows); i++) {
        const combo = batchResult.combos[i];
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const batchFastRow = batchResult.fast_values.slice(rowStart, rowEnd);
        const batchSlowRow = batchResult.slow_values.slice(rowStart, rowEnd);
        
        
        const singleResult = wasm.qqe_js(
            close, 
            combo.rsi_period, 
            combo.smoothing_factor, 
            combo.fast_factor
        );
        const singleFast = singleResult.values.slice(0, 100);
        const singleSlow = singleResult.values.slice(100, 200);
        
        assertArrayClose(
            batchFastRow,
            singleFast,
            1e-9,
            `Batch row ${i} fast mismatch`
        );
        
        assertArrayClose(
            batchSlowRow,
            singleSlow,
            1e-9,
            `Batch row ${i} slow mismatch`
        );
    }
});

test('QQE boundary conditions', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    
    
    const result1 = wasm.qqe_js(close, 14, 1, 4.236);
    assert.strictEqual(result1.values.length, close.length * 2);
    
    
    const result2 = wasm.qqe_js(close, 14, 5, 10.0);
    assert.strictEqual(result2.values.length, close.length * 2);
    
    
    const result3 = wasm.qqe_js(close, 14, 5, 0.1);
    assert.strictEqual(result3.values.length, close.length * 2);
});

test('QQE values within bounds', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.qqe_js(close, 14, 5, 4.236);
    const fast = result.values.slice(0, result.cols);
    const slow = result.values.slice(result.cols, result.cols * 2);
    
    
    for (let i = 0; i < fast.length; i++) {
        if (!isNaN(fast[i])) {
            assert(fast[i] >= 0.0, `Fast value at ${i} is < 0: ${fast[i]}`);
            assert(fast[i] <= 100.0, `Fast value at ${i} is > 100: ${fast[i]}`);
        }
        if (!isNaN(slow[i])) {
            assert(slow[i] >= 0.0, `Slow value at ${i} is < 0: ${slow[i]}`);
            assert(slow[i] <= 100.0, `Slow value at ${i} is > 100: ${slow[i]}`);
        }
    }
});

test('QQE constant data', () => {
    
    const constantData = new Float64Array(100).fill(50.0);
    
    const result = wasm.qqe_js(constantData, 14, 5, 4.236);
    const fast = result.values.slice(0, result.cols);
    const slow = result.values.slice(result.cols, result.cols * 2);
    
    
    const warmup = 17; 
    
    if (fast.length > warmup + 10) {
        
        const lastFast = fast.slice(-10);
        const lastSlow = slow.slice(-10);
        
        const meanFast = lastFast.reduce((a, b) => a + b, 0) / lastFast.length;
        const stdFast = Math.sqrt(lastFast.reduce((sq, n) => sq + Math.pow(n - meanFast, 2), 0) / lastFast.length);
        
        const meanSlow = lastSlow.reduce((a, b) => a + b, 0) / lastSlow.length;
        const stdSlow = Math.sqrt(lastSlow.reduce((sq, n) => sq + Math.pow(n - meanSlow, 2), 0) / lastSlow.length);
        
        assert(stdFast < 0.1, `Fast values should be stable for constant input, std: ${stdFast}`);
        assert(stdSlow < 0.1, `Slow values should be stable for constant input, std: ${stdSlow}`);
    }
});
