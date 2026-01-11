/**
 * WASM binding tests for ZSCORE indicator.
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

test('ZSCORE basic candles', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.zscore_js(close, 14, "sma", 1.0, 0);
    assert.strictEqual(result.length, close.length);
});

test('ZSCORE accuracy vs Rust reference (last 5 values)', () => {
    
    const close = new Float64Array(testData.close);
    const out = wasm.zscore_js(close, 14, 'sma', 1.0, 0);

    const start = Math.max(0, out.length - 5);
    const last5 = out.slice(start);
    const expected = EXPECTED_OUTPUTS.zscore.last5Values;

    
    for (let i = 0; i < 5; i++) {
        assertClose(last5[i], expected[i], 1e-8, `Zscore mismatch at tail idx ${i}`);
    }
});

test('ZSCORE with custom parameters', () => {
    const close = new Float64Array(testData.close);
    
    
    const result = wasm.zscore_js(close, 20, "ema", 2.0, 0);
    assert.strictEqual(result.length, close.length);
    
    
    for (let i = 0; i < 19; i++) {
        assert(isNaN(result[i]));
    }
    
    
    assert(!isNaN(result[19]));
});

test('ZSCORE zero period', () => {
    
    const data = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.zscore_js(data, 0, "sma", 1.0, 0);
    }, /Invalid period/);
});

test('ZSCORE period exceeds length', () => {
    
    const data = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.zscore_js(data, 10, "sma", 1.0, 0);
    }, /Invalid period/);
});

test('ZSCORE very small dataset', () => {
    
    const data = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.zscore_js(data, 14, "sma", 1.0, 0);
    }, /Invalid period/);
});

test('ZSCORE all NaN', () => {
    
    const data = new Float64Array([NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.zscore_js(data, 2, "sma", 1.0, 0);
    }, /All values are NaN/);
});

test('ZSCORE fast API - basic', () => {
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    
    const inPtr = wasm.zscore_alloc(len);
    const outPtr = wasm.zscore_alloc(len);
    assert(inPtr !== 0, 'Failed to allocate input memory');
    assert(outPtr !== 0, 'Failed to allocate output memory');
    
    try {
        
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inView.set(close);
        
        
        wasm.zscore_into(inPtr, outPtr, len, 14, "sma", 1.0, 0);
        
        
        const resultBuffer = wasm.__wasm.memory.buffer;
        const result = new Float64Array(resultBuffer, outPtr, len);
        
        
        const resultArray = Array.from(result);
        
        
        const expected = wasm.zscore_js(close, 14, "sma", 1.0, 0);
        assertArrayClose(resultArray, expected, 1e-10, "Fast API mismatch");
        
    } finally {
        wasm.zscore_free(inPtr, len);
        wasm.zscore_free(outPtr, len);
    }
});

test('ZSCORE fast API - in-place', () => {
    const data = new Float64Array(testData.close.slice(0, 100));
    const len = data.length;
    
    
    const ptr = wasm.zscore_alloc(len);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    try {
        
        const buffer = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        buffer.set(data);
        
        
        wasm.zscore_into(ptr, ptr, len, 14, "sma", 1.0, 0);
        
        
        const result = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        
        
        const expected = wasm.zscore_js(data, 14, "sma", 1.0, 0);
        assertArrayClose(result, expected, 1e-10, "In-place computation mismatch");
        
    } finally {
        wasm.zscore_free(ptr, len);
    }
});

test('ZSCORE batch processing', async () => {
    const close = new Float64Array(testData.close.slice(0, 500));
    
    
    const inPtr = wasm.zscore_alloc(close.length);
    
    try {
        
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, close.length);
        inView.set(close);
        
        
        const nPeriods = 3; 
        const nNbdevs = 3;  
        const nCombos = nPeriods * nNbdevs;
        
        
        const outPtr = wasm.zscore_alloc(nCombos * close.length);
        
        try {
            
            const actualCombos = wasm.zscore_batch_into(
                inPtr, outPtr, close.length,
                10, 20, 5,     
                "sma",         
                1.0, 2.0, 0.5, 
                0, 0, 0        
            );
            
            assert.strictEqual(actualCombos, 9, 'Should return 9 combinations');
            
            
            const results = new Float64Array(wasm.__wasm.memory.buffer, outPtr, nCombos * close.length);
            
            
            let firstNonNaN = -1;
            for (let i = 0; i < close.length; i++) {
                if (!isNaN(results[i])) {
                    firstNonNaN = i;
                    break;
                }
            }
            assert(firstNonNaN >= 9, 'First row should have warmup period of at least 9 (period-1)');
            
        } finally {
            wasm.zscore_free(outPtr, nCombos * close.length);
        }
    } finally {
        wasm.zscore_free(inPtr, close.length);
    }
});

test('ZSCORE batch fast API', () => {
    const close = new Float64Array(testData.close.slice(0, 100));
    const len = close.length;
    
    
    const nPeriods = Math.floor((20 - 10) / 5) + 1; 
    const nNbdevs = Math.floor((2.0 - 1.0) / 0.5) + 1; 
    const nCombos = nPeriods * nNbdevs; 
    
    
    const inPtr = wasm.zscore_alloc(len);
    const outPtr = wasm.zscore_alloc(nCombos * len);
    assert(inPtr !== 0, 'Failed to allocate input memory');
    assert(outPtr !== 0, 'Failed to allocate output memory');
    
    try {
        
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inView.set(close);
        
        const resultCombos = wasm.zscore_batch_into(
            inPtr,
            outPtr,
            len,
            10, 20, 5,  
            "sma",
            1.0, 2.0, 0.5,  
            0, 0, 0  
        );
        
        assert.strictEqual(resultCombos, nCombos, 'Should return correct number of combinations');
        
        
        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, nCombos * len);
        assert(result.some(v => !isNaN(v)), 'Should have some non-NaN values');
        
    } finally {
        wasm.zscore_free(inPtr, len);
        wasm.zscore_free(outPtr, nCombos * len);
    }
});

test('ZSCORE different MA types', () => {
    const close = new Float64Array(testData.close.slice(0, 100));
    const maTypes = ["sma", "ema", "wma"];
    
    for (const maType of maTypes) {
        try {
            const result = wasm.zscore_js(close, 14, maType, 1.0, 0);
            assert.strictEqual(result.length, close.length);
            
            
            assert(isNaN(result[0]));
            
            assert(!isNaN(result[20]));
        } catch (e) {
            
            assert(e.message.includes("Unknown MA") || e.message.includes("Invalid"));
        }
    }
});

test('ZSCORE deviation types', () => {
    const close = new Float64Array(testData.close.slice(0, 100));
    
    
    for (const devtype of [0, 1, 2]) {
        const result = wasm.zscore_js(close, 14, "sma", 1.0, devtype);
        assert.strictEqual(result.length, close.length);
    }
});
