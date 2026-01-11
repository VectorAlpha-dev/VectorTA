/**
 * WASM binding tests for SRSI indicator.
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

test('SRSI partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.srsi_js(close, 14, 14, 3, 3);
    assert.strictEqual(result.length, close.length * 2); 
});

test('SRSI accuracy', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const expected_k = [
        65.52066633236464,
        61.22507053191985,
        57.220471530042644,
        64.61344854988147,
        60.66534359318523,
    ];
    const expected_d = [
        64.33503158970049,
        64.42143544464182,
        61.32206946477942,
        61.01966353728503,
        60.83308789104016,
    ];
    
    
    const result = wasm.srsi_js(close, 14, 14, 3, 3);
    
    
    const k_values = result.slice(0, close.length);
    const d_values = result.slice(close.length);
    
    assert.strictEqual(k_values.length, close.length);
    assert.strictEqual(d_values.length, close.length);
    
    
    const k_last5 = k_values.slice(-5);
    const d_last5 = d_values.slice(-5);
    
    assertArrayClose(k_last5, expected_k, 1e-6, "SRSI K last 5 values mismatch");
    assertArrayClose(d_last5, expected_d, 1e-6, "SRSI D last 5 values mismatch");
});

test('SRSI custom params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.srsi_js(close, 10, 10, 4, 4);
    assert.strictEqual(result.length, close.length * 2);
});

test('SRSI from slice', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.srsi_js(close, 3, 3, 2, 2);
    assert.strictEqual(result.length, close.length * 2);
});

test('SRSI zero period', () => {
    const input_data = new Float64Array([10.0, 11.0, 12.0]);
    
    assert.throws(() => {
        wasm.srsi_js(input_data, 0, 0, 0, 0);
    }, /Invalid period/);
});

test('SRSI insufficient data', () => {
    const input_data = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.srsi_js(input_data, 90, 3, 20, 20);
    }, /Not enough/);
});

test('SRSI empty input', () => {
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.srsi_js(empty, 14, 14, 3, 3);
    }, /empty/);
});

test('SRSI fast API (srsi_into)', () => {
    const close = new Float64Array(testData.close.slice(0, 500));
    const len = close.length;

    const inPtr = wasm.srsi_alloc(len);
    const kPtr = wasm.srsi_alloc(len);
    const dPtr = wasm.srsi_alloc(len);
    assert(inPtr !== 0, 'Failed to allocate input buffer');
    assert(kPtr !== 0, 'Failed to allocate k buffer');
    assert(dPtr !== 0, 'Failed to allocate d buffer');

    try {
        
        const memory = wasm.__wasm.memory.buffer;
        const inView = new Float64Array(memory, inPtr, len);
        inView.set(close);

        
        wasm.srsi_into(inPtr, kPtr, dPtr, len, 14, 14, 3, 3);

        
        const memory2 = wasm.__wasm.memory.buffer;
        const kView = new Float64Array(memory2, kPtr, len);
        const dView = new Float64Array(memory2, dPtr, len);

        
        const regular = wasm.srsi_js(close, 14, 14, 3, 3);
        const regularK = regular.slice(0, len);
        const regularD = regular.slice(len);

        for (let i = 0; i < len; i++) {
            if (isNaN(regularK[i]) && isNaN(kView[i])) continue;
            if (isNaN(regularD[i]) && isNaN(dView[i])) continue;
            assertClose(kView[i], regularK[i], 1e-10, `srsi_into k mismatch at ${i}`);
            assertClose(dView[i], regularD[i], 1e-10, `srsi_into d mismatch at ${i}`);
        }
    } finally {
        wasm.srsi_free(inPtr, len);
        wasm.srsi_free(kPtr, len);
        wasm.srsi_free(dPtr, len);
    }
});

test('SRSI fast API with aliasing', () => {
    const close = new Float64Array(testData.close.slice(0, 500));
    const len = close.length;

    const inOutPtr = wasm.srsi_alloc(len);
    const dPtr = wasm.srsi_alloc(len);
    assert(inOutPtr !== 0, 'Failed to allocate in/out buffer');
    assert(dPtr !== 0, 'Failed to allocate d buffer');

    try {
        
        const regular = wasm.srsi_js(close, 14, 14, 3, 3);
        const regularK = regular.slice(0, len);
        const regularD = regular.slice(len);

        
        const memory = wasm.__wasm.memory.buffer;
        const inOutView = new Float64Array(memory, inOutPtr, len);
        inOutView.set(close);

        wasm.srsi_into(inOutPtr, inOutPtr, dPtr, len, 14, 14, 3, 3);

        const memory2 = wasm.__wasm.memory.buffer;
        const kView = new Float64Array(memory2, inOutPtr, len);
        const dView = new Float64Array(memory2, dPtr, len);

        for (let i = 0; i < len; i++) {
            if (isNaN(regularK[i]) && isNaN(kView[i])) continue;
            if (isNaN(regularD[i]) && isNaN(dView[i])) continue;
            assertClose(kView[i], regularK[i], 1e-10, `aliasing k mismatch at ${i}`);
            assertClose(dView[i], regularD[i], 1e-10, `aliasing d mismatch at ${i}`);
        }
    } finally {
        wasm.srsi_free(inOutPtr, len);
        wasm.srsi_free(dPtr, len);
    }
});

test('SRSI batch operation', () => {
    const close = new Float64Array(testData.close.slice(0, 1000)); 
    
    const config = {
        rsi_period_range: [14, 14, 0],  
        stoch_period_range: [14, 14, 0], 
        k_range: [3, 3, 0],  
        d_range: [3, 3, 0]   
    };
    
    const result = wasm.srsi_batch(close, config);
    
    assert(result.k_values);
    assert(result.d_values);
    assert(result.combos);
    assert.strictEqual(result.rows, 1); 
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.k_values.length, close.length);
    assert.strictEqual(result.d_values.length, close.length);
});

test('SRSI batch with multiple params', () => {
    const close = new Float64Array(testData.close.slice(0, 500)); 
    
    const config = {
        rsi_period_range: [10, 14, 2],    
        stoch_period_range: [10, 14, 2],  
        k_range: [2, 4, 1],               
        d_range: [2, 3, 1]                
    };
    
    const result = wasm.srsi_batch(close, config);
    
    
    const expected_rows = 3 * 3 * 3 * 2;
    assert.strictEqual(result.rows, expected_rows);
    assert.strictEqual(result.combos.length, expected_rows);
    assert.strictEqual(result.k_values.length, expected_rows * close.length);
    assert.strictEqual(result.d_values.length, expected_rows * close.length);
});

test('SRSI memory allocation and deallocation', () => {
    const len = 1000;
    
    
    const ptr = wasm.srsi_alloc(len);
    assert(ptr !== 0, 'Allocation should return non-zero pointer');
    
    
    assert.doesNotThrow(() => {
        wasm.srsi_free(ptr, len);
    });
});

test('SRSI batch fast API', () => {
    const close = new Float64Array(testData.close.slice(0, 500));
    
    
    const rsi_periods = 3;    
    const stoch_periods = 3;  
    const k_periods = 3;      
    const d_periods = 2;      
    const expected_rows = rsi_periods * stoch_periods * k_periods * d_periods;
    
    const len = close.length;
    const total = expected_rows * len;

    const inPtr = wasm.srsi_alloc(len);
    const kPtr = wasm.srsi_alloc(total);
    const dPtr = wasm.srsi_alloc(total);
    assert(inPtr !== 0, 'Failed to allocate input buffer');
    assert(kPtr !== 0, 'Failed to allocate k buffer');
    assert(dPtr !== 0, 'Failed to allocate d buffer');

    try {
        const memory = wasm.__wasm.memory.buffer;
        new Float64Array(memory, inPtr, len).set(close);

        const rows = wasm.srsi_batch_into(
            inPtr,
            kPtr,
            dPtr,
            len,
            10, 14, 2,    
            10, 14, 2,    
            2, 4, 1,      
            2, 3, 1       
        );

        assert.strictEqual(rows, expected_rows);

        const memory2 = wasm.__wasm.memory.buffer;
        const kView = new Float64Array(memory2, kPtr, total);
        const dView = new Float64Array(memory2, dPtr, total);

        
        assert(!isNaN(kView[50]));
        assert(!isNaN(dView[50]));
    } finally {
        wasm.srsi_free(inPtr, len);
        wasm.srsi_free(kPtr, total);
        wasm.srsi_free(dPtr, total);
    }
});
