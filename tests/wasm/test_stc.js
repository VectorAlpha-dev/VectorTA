/**
 * WASM binding tests for STC indicator.
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

test('STC default params', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const result = wasm.stc_js(close, 23, 50, 10, 3, "ema", "ema");
    assert.strictEqual(result.length, close.length);
    
    
    const warmup = 50; 
    let hasValidValues = false;
    for (let i = warmup; i < result.length; i++) {
        if (!isNaN(result[i])) {
            hasValidValues = true;
            break;
        }
    }
    assert(hasValidValues, "Expected some valid values after warmup period");
});

test('STC with custom params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.stc_js(close, 12, 26, 9, 3, "sma", "sma");
    assert.strictEqual(result.length, close.length);
});

test('STC accuracy', async () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.stc_js(close, 23, 50, 10, 3, "ema", "ema");
    assert.strictEqual(result.length, close.length);
    
    
    for (let i = 0; i < result.length; i++) {
        if (!isNaN(result[i])) {
            assert(result[i] >= -0.1, `STC value ${result[i]} should be >= 0`);
            assert(result[i] <= 100.1, `STC value ${result[i]} should be <= 100`);
        }
    }
    
    
    if (EXPECTED_OUTPUTS.stc) {
        const expected = EXPECTED_OUTPUTS.stc;
        const last5 = result.slice(-5);
        assertArrayClose(
            last5,
            expected.last5Values,
            1e-6,
            "STC last 5 values mismatch"
        );
        await compareWithRust('stc', result, 'close', expected.defaultParams);
    }
});

test('STC zero period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0]);
    
    assert.throws(() => {
        wasm.stc_js(inputData, 0, 50, 10, 3, "ema", "ema");
    }, /Not enough valid data|Invalid|Empty/);
    
    assert.throws(() => {
        wasm.stc_js(inputData, 23, 0, 10, 3, "ema", "ema");
    }, /Not enough valid data|Invalid|Empty/);
});

test('STC period exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.stc_js(dataSmall, 10, 50, 10, 3, "ema", "ema");
    }, /Not enough valid data/);
});

test('STC empty data', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.stc_js(empty, 23, 50, 10, 3, "ema", "ema");
    }, /Empty data/);
});

test('STC all NaN', () => {
    
    const allNaN = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        allNaN[i] = NaN;
    }
    
    assert.throws(() => {
        wasm.stc_js(allNaN, 23, 50, 10, 3, "ema", "ema");
    }, /All values are NaN/);
});

test('STC NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    
    
    for (let i = 10; i < 20; i++) {
        close[i] = NaN;
    }
    
    const result = wasm.stc_js(close, 23, 50, 10, 3, "ema", "ema");
    assert.strictEqual(result.length, close.length);
    
    
    let hasValidValues = false;
    for (let i = 100; i < result.length; i++) {
        if (!isNaN(result[i])) {
            hasValidValues = true;
            break;
        }
    }
    assert(hasValidValues, "Expected some valid values after NaN section");
});

test('STC fast API', () => {
    
    const close = new Float64Array(testData.close.slice(0, 500));
    const len = close.length;
    
    
    const inPtr = wasm.stc_alloc(len);
    const outPtr = wasm.stc_alloc(len);
    assert(inPtr !== 0, 'Failed to allocate input buffer');
    assert(outPtr !== 0, 'Failed to allocate output buffer');
    
    try {
        
        const memory = wasm.__wasm.memory.buffer;
        new Float64Array(memory, inPtr, len).set(close);
        
        
        wasm.stc_into(
            inPtr,
            outPtr,
            len,
            23, 50, 10, 3,
            "ema", "ema"
        );
        
        
        const memory2 = wasm.__wasm.memory.buffer;
        const result = new Float64Array(memory2, outPtr, len);
        
        assert.strictEqual(result.length, len);
        
        
        const regular = wasm.stc_js(close, 23, 50, 10, 3, "ema", "ema");
        for (let i = 0; i < len; i++) {
            if (isNaN(regular[i]) && isNaN(result[i])) continue;
            assertClose(result[i], regular[i], 1e-10, `stc_into mismatch at ${i}`);
        }
    } finally {
        
        wasm.stc_free(inPtr, len);
        wasm.stc_free(outPtr, len);
    }
});

test('STC fast API in-place', () => {
    
    const close = new Float64Array(testData.close.slice(0, 500));
    const len = close.length;
    
    const inOutPtr = wasm.stc_alloc(len);
    assert(inOutPtr !== 0, 'Failed to allocate in/out buffer');

    try {
        const regular = wasm.stc_js(close, 23, 50, 10, 3, "ema", "ema");

        const memory = wasm.__wasm.memory.buffer;
        new Float64Array(memory, inOutPtr, len).set(close);

        
        wasm.stc_into(inOutPtr, inOutPtr, len, 23, 50, 10, 3, "ema", "ema");

        const memory2 = wasm.__wasm.memory.buffer;
        const result = new Float64Array(memory2, inOutPtr, len);

        for (let i = 0; i < len; i++) {
            if (isNaN(regular[i]) && isNaN(result[i])) continue;
            assertClose(result[i], regular[i], 1e-10, `stc_into aliasing mismatch at ${i}`);
        }
    } finally {
        wasm.stc_free(inOutPtr, len);
    }
});

test('STC batch processing', async () => {
    
    const close = new Float64Array(testData.close);
    
    const config = {
        fast_period_range: [20, 30, 5],    
        slow_period_range: [45, 55, 5],    
        k_period_range: [10, 10, 1],       
        d_period_range: [3, 3, 1]          
    };
    
    const result = wasm.stc_batch(close, config);
    
    assert(result.values, "Expected values in result");
    assert(result.combos, "Expected combos in result");
    assert(result.rows, "Expected rows in result");
    assert(result.cols, "Expected cols in result");
    
    
    assert.strictEqual(result.rows, 9);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.values.length, 9 * close.length);
    assert.strictEqual(result.combos.length, 9);
    
    
    const firstRow = result.values.slice(0, close.length);
    const singleResult = wasm.stc_js(close, 20, 45, 10, 3, "ema", "ema");
    
    assertArrayClose(
        firstRow,
        singleResult,
        1e-10,
        "Batch first row vs single calculation mismatch"
    );
});

test('STC batch single param', () => {
    
    const close = new Float64Array(testData.close);
    
    const config = {
        fast_period_range: [23, 23, 0],
        slow_period_range: [50, 50, 0],
        k_period_range: [10, 10, 0],
        d_period_range: [3, 3, 0]
    };
    
    const result = wasm.stc_batch(close, config);
    
    
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, close.length);
    
    
    const singleResult = wasm.stc_js(close, 23, 50, 10, 3, "ema", "ema");
    assertArrayClose(
        result.values,
        singleResult,
        1e-10,
        "Batch single param vs single calculation mismatch"
    );
});
