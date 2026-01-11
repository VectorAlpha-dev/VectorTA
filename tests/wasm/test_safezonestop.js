/**
 * WASM binding tests for SafeZoneStop indicator.
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

test('SafeZoneStop partial params', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    
    const result = wasm.safezonestop_js(high, low, 14, 2.5, 3, "short");
    assert.strictEqual(result.length, high.length);
});

test('SafeZoneStop accuracy', async () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.safezonestop_js(high, low, 22, 2.5, 3, "long");
    
    assert.strictEqual(result.length, high.length);
    
    
    const expectedLast5 = [
        45331.180007991,
        45712.94455308232,
        46019.94707339676,
        46461.767660969635,
        46461.767660969635,
    ];
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLast5,
        1e-4,
        "SafeZoneStop last 5 values mismatch"
    );
});

test('SafeZoneStop default params', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    
    const result = wasm.safezonestop_js(high, low, 22, 2.5, 3, "long");
    assert.strictEqual(result.length, high.length);
});

test('SafeZoneStop zero period', () => {
    
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    
    assert.throws(() => {
        wasm.safezonestop_js(high, low, 0, 2.5, 3, "long");
    }, /Invalid period/);
});

test('SafeZoneStop mismatched lengths', () => {
    
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0]); 
    
    assert.throws(() => {
        wasm.safezonestop_js(high, low, 22, 2.5, 3, "long");
    }, /Mismatched lengths/);
});

test('SafeZoneStop invalid direction', () => {
    
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    
    assert.throws(() => {
        wasm.safezonestop_js(high, low, 2, 2.5, 3, "invalid");
    }, /Invalid direction/);
});

test('SafeZoneStop nan handling', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.safezonestop_js(high, low, 22, 2.5, 3, "long");
    assert.strictEqual(result.length, high.length);
    
    
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('SafeZoneStop fast API (into)', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const len = high.length;
    
    
    const highPtr = wasm.safezonestop_alloc(len);
    const lowPtr = wasm.safezonestop_alloc(len);
    const outPtr = wasm.safezonestop_alloc(len);
    
    try {
        
        const highMem = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowMem = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        highMem.set(high);
        lowMem.set(low);
        
        
        wasm.safezonestop_into(
            highPtr,
            lowPtr,
            outPtr,
            len,
            22,
            2.5,
            3,
            "long"
        );
        
        
        const memory = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        const result = Array.from(memory);
        
        
        const safeResult = wasm.safezonestop_js(high, low, 22, 2.5, 3, "long");
        assertArrayClose(result, safeResult, 1e-10, "Fast API mismatch");
    } finally {
        
        wasm.safezonestop_free(highPtr, len);
        wasm.safezonestop_free(lowPtr, len);
        wasm.safezonestop_free(outPtr, len);
    }
});

test('SafeZoneStop fast API aliasing - high', () => {
    
    const high = new Float64Array(testData.high.slice(0, 100)); 
    const low = new Float64Array(testData.low.slice(0, 100));
    const len = high.length;
    
    
    const highCopy = new Float64Array(high);
    
    
    const highPtr = wasm.safezonestop_alloc(len);
    const lowPtr = wasm.safezonestop_alloc(len);
    
    try {
        
        const highMem = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowMem = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        highMem.set(high);
        lowMem.set(low);
        
        
        wasm.safezonestop_into(
            highPtr,
            lowPtr,
            highPtr, 
            len,
            22,
            2.5,
            3,
            "long"
        );
        
        
        const result = Array.from(new Float64Array(wasm.__wasm.memory.buffer, highPtr, len));
        
        
        const expected = wasm.safezonestop_js(highCopy, low, 22, 2.5, 3, "long");
        assertArrayClose(result, expected, 1e-10, "Aliasing test failed");
    } finally {
        
        wasm.safezonestop_free(highPtr, len);
        wasm.safezonestop_free(lowPtr, len);
    }
});

test('SafeZoneStop batch API', () => {
    
    const high = new Float64Array(testData.high.slice(0, 100)); 
    const low = new Float64Array(testData.low.slice(0, 100));
    
    const config = {
        period_range: [14, 30, 8],
        mult_range: [2.0, 3.0, 0.5],
        max_lookback_range: [2, 4, 1],
        direction: "long"
    };
    
    const result = wasm.safezonestop_batch(high, low, config);
    
    
    assert(result.values, "Missing values in batch result");
    assert(result.combos, "Missing combos in batch result");
    assert(result.rows > 0, "Invalid rows count");
    assert(result.cols === 100, "Invalid cols count");
    
    
    assert.strictEqual(result.values.length, result.rows * result.cols);
    assert.strictEqual(result.combos.length, result.rows);
});

test('SafeZoneStop batch single parameter set', () => {
    
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    
    const config = {
        period_range: [22, 22, 0],
        mult_range: [2.5, 2.5, 0],
        max_lookback_range: [3, 3, 0],
        direction: "long"
    };
    
    const batchResult = wasm.safezonestop_batch(high, low, config);
    
    
    const singleResult = wasm.safezonestop_js(high, low, 22, 2.5, 3, "long");
    
    assert.strictEqual(batchResult.rows, 1);
    assert.strictEqual(batchResult.cols, 50);
    
    const batchValues = batchResult.values.slice(0, 50);
    assertArrayClose(
        batchValues,
        singleResult,
        1e-10,
        "Batch vs single mismatch"
    );
});

test('SafeZoneStop memory allocation/deallocation', () => {
    
    const len = 1000;
    
    
    const ptr = wasm.safezonestop_alloc(len);
    assert(ptr > 0, "Invalid pointer returned");
    
    
    const memory = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
    for (let i = 0; i < len; i++) {
        memory[i] = i * 1.5;
    }
    
    
    assert.strictEqual(memory[0], 0);
    assert.strictEqual(memory[999], 999 * 1.5);
    
    
    wasm.safezonestop_free(ptr, len);
    
});

test('SafeZoneStop reinput', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    
    const firstResult = wasm.safezonestop_js(high, low, 22, 2.5, 3, "long");
    assert.strictEqual(firstResult.length, high.length);
    
    
    const secondResult = wasm.safezonestop_js(firstResult, firstResult, 22, 2.5, 3, "long");
    assert.strictEqual(secondResult.length, firstResult.length);
    
    
    
    let hasDifference = false;
    for (let i = 0; i < firstResult.length; i++) {
        if (!isNaN(firstResult[i]) && !isNaN(secondResult[i])) {
            if (Math.abs(firstResult[i] - secondResult[i]) > 1e-10) {
                hasDifference = true;
                break;
            }
        }
    }
    assert(hasDifference, "Reinput should produce different values");
});

test('SafeZoneStop all NaN input', () => {
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.safezonestop_js(allNaN, allNaN, 22, 2.5, 3, "long");
    }, /All values are NaN/);
});