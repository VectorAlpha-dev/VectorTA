/**
 * WASM binding tests for RSX indicator.
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

test('RSX partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.rsx_js(close, 14);
    assert.strictEqual(result.length, close.length);
});

test('RSX accuracy', async () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.rsx;
    
    const result = wasm.rsx_js(
        close,
        expected.default_params.period
    );
    
    assert.strictEqual(result.length, close.length);
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last_5_values,
        0.1,  
        "RSX last 5 values mismatch"
    );
    
    
    await compareWithRust('rsx', result, 'close', expected.default_params);
});

test('RSX default candles', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.rsx_js(close, 14);
    assert.strictEqual(result.length, close.length);
});

test('RSX zero period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.rsx_js(inputData, 0);
    }, /Invalid period/);
});

test('RSX period exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.rsx_js(dataSmall, 10);
    }, /Invalid period/);
});

test('RSX all NaN', () => {
    
    const allNan = new Float64Array(10).fill(NaN);
    
    assert.throws(() => {
        wasm.rsx_js(allNan, 3);
    }, /All values are NaN/);
});

test('RSX NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.rsx_js(close, 14);
    assert.strictEqual(result.length, close.length);
    
    
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i}`);
        }
    }
});

test('RSX empty input', () => {
    
    const emptyData = new Float64Array([]);
    
    assert.throws(() => {
        wasm.rsx_js(emptyData, 14);
    }, /All values are NaN|Invalid period|empty/i);
});

test('RSX fast API (rsx_into)', () => {
    
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    
    const outPtr = wasm.rsx_alloc(len);
    const inPtr = wasm.rsx_alloc(len);
    
    try {
        
        const memory = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        memory.set(close);
        
        
        wasm.rsx_into(inPtr, outPtr, len, 14);
        
        
        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        const resultCopy = new Float64Array(result);
        
        
        const safeResult = wasm.rsx_js(close, 14);
        assertArrayClose(resultCopy, safeResult, 1e-10, "Fast API mismatch with safe API");
    } finally {
        
        wasm.rsx_free(inPtr, len);
        wasm.rsx_free(outPtr, len);
    }
});

test('RSX fast API in-place', () => {
    
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    
    const ptr = wasm.rsx_alloc(len);
    
    try {
        
        const memory = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        memory.set(close);
        
        
        wasm.rsx_into(ptr, ptr, len, 14);
        
        
        const result = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        const resultCopy = new Float64Array(result);
        
        
        const safeResult = wasm.rsx_js(close, 14);
        assertArrayClose(resultCopy, safeResult, 1e-10, "In-place operation mismatch");
    } finally {
        
        wasm.rsx_free(ptr, len);
    }
});

test('RSX memory allocation/deallocation', () => {
    
    const len = 1000;
    
    
    const ptr = wasm.rsx_alloc(len);
    assert(ptr !== 0, "Allocation returned null pointer");
    
    
    const memory = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
    for (let i = 0; i < len; i++) {
        memory[i] = i * 1.5;
    }
    
    
    assert.strictEqual(memory[0], 0);
    assert.strictEqual(memory[10], 15);
    
    
    wasm.rsx_free(ptr, len);
    
    
    const ptr2 = wasm.rsx_alloc(len);
    assert(ptr2 !== 0, "Re-allocation failed");
    wasm.rsx_free(ptr2, len);
});

test('RSX null pointer handling', () => {
    
    assert.throws(() => {
        wasm.rsx_into(0, 0, 100, 14);
    }, /null pointer/);
});

test('RSX batch API', () => {
    
    const close = new Float64Array(testData.close);
    
    const config = {
        period_range: [10, 20, 5]  
    };
    
    const result = wasm.rsx_batch(close, config);
    
    assert(result.values, "Batch result should have values");
    assert(result.combos, "Batch result should have combos");
    assert.strictEqual(result.rows, 3, "Should have 3 parameter combinations");
    assert.strictEqual(result.cols, close.length, "Columns should match data length");
    assert.strictEqual(result.values.length, result.rows * result.cols, "Values array size mismatch");
    assert.strictEqual(result.combos.length, 3, "Should have 3 parameter combinations");
    
    
    assert.strictEqual(result.combos[0].period, 10);
    assert.strictEqual(result.combos[1].period, 15);
    assert.strictEqual(result.combos[2].period, 20);
});

test('RSX batch fast API', () => {
    
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    
    const periodStart = 10, periodEnd = 20, periodStep = 5;
    const expectedRows = Math.floor((periodEnd - periodStart) / periodStep) + 1; 
    const outputSize = expectedRows * len;
    
    
    const inPtr = wasm.rsx_alloc(len);
    const outPtr = wasm.rsx_alloc(outputSize);
    
    try {
        
        const memory = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        memory.set(close);
        
        
        const rows = wasm.rsx_batch_into(
            inPtr, outPtr, len,
            periodStart, periodEnd, periodStep
        );
        
        assert.strictEqual(rows, expectedRows, "Row count mismatch");
        
        
        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, outputSize);
        
        
        assert(isNaN(result[0]), "First value should be NaN");
        
        
        assert(result.length === outputSize, "Output size mismatch");
    } finally {
        
        wasm.rsx_free(inPtr, len);
        wasm.rsx_free(outPtr, outputSize);
    }
});