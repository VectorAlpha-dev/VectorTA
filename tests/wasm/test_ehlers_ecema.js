/**
 * WASM binding tests for EHLERS_ECEMA indicator.
 * These tests mirror the Rust unit tests to ensure WASM bindings work correctly.
 */
import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';
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
let wasmInst; 
let testData;
let HAS_WASM_MEMORY = false;

test.before(async () => {
    
    try {
        const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
        const importPath = process.platform === 'win32'
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
        wasm = await import(importPath);
        
        const wasmBgPath = path.join(path.dirname(importPath.replace('file:///', '/')), 'vector_ta_bg.wasm');
        const wasmBytes = fs.readFileSync(wasmBgPath);
        if (typeof wasm.initSync === 'function') {
            wasmInst = wasm.initSync(wasmBytes);
        } else if (typeof wasm.default === 'function') {
            
            wasmInst = await wasm.default(wasmBytes);
        }

        
        HAS_WASM_MEMORY = !!(wasmInst && wasmInst.memory);
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('EHLERS_ECEMA partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.ehlers_ecema_js(close, 20, 50);
    assert.strictEqual(result.length, close.length);
});

test('EHLERS_ECEMA accuracy', () => {
    
    const expected = EXPECTED_OUTPUTS.ehlersEcema;
    
    const data = new Float64Array(testData.close);
    
    
    const length = expected.defaultParams.length;
    const gainLimit = expected.defaultParams.gainLimit;
    
    
    const result = wasm.ehlers_ecema_js(data, length, gainLimit);
    
    assert.strictEqual(result.length, data.length);
    
    
    for (let i = 0; i < expected.warmupPeriod; i++) {
        assert(isNaN(result[i]), `Expected NaN during warmup at index ${i}`);
    }
    
    
    assert(!isNaN(result[expected.warmupPeriod]), "Expected valid value after warmup");
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-8,
        "EHLERS_ECEMA last 5 values mismatch"
    );
});

test('EHLERS_ECEMA Pine mode accuracy', () => {
    if (!HAS_WASM_MEMORY) {
        console.log('Skipping Pine mode zero-copy test: wasm memory export unavailable');
        return;
    }
    
    const expected = EXPECTED_OUTPUTS.ehlersEcema;
    
    const data = new Float64Array(testData.close);
    
    
    if (wasm.ehlers_ecema_into_ex) {
        const len = data.length;
        const ptr = wasm.ehlers_ecema_alloc(len);
        
        
        const memory = wasmInst.memory;
        const memView = new Float64Array(memory.buffer, ptr, len);
        memView.set(data);
        
        
        const outPtr = wasm.ehlers_ecema_alloc(len);
        
        
        wasm.ehlers_ecema_into_ex(ptr, outPtr, len, 20, 50, true, false);
        
        
        const result = new Float64Array(wasmInst.memory.buffer, outPtr, len);
        const resultCopy = Array.from(result);
        
        
        wasm.ehlers_ecema_free(ptr, len);
        wasm.ehlers_ecema_free(outPtr, len);
        
        
        const last5 = resultCopy.slice(-5);
        assertArrayClose(
            last5,
            expected.pineModeLast5,
            1e-8,
            "EHLERS_ECEMA Pine mode last 5 values mismatch"
        );
        
        
        assert(!isNaN(resultCopy[0]), "Pine mode should have valid value at index 0");
    }
});

test('EHLERS_ECEMA default candles', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.ehlers_ecema_js(close, 20, 50);
    assert.strictEqual(result.length, close.length);
});

test('EHLERS_ECEMA zero period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.ehlers_ecema_js(inputData, 0, 50);
    }, /Invalid/);
});

test('EHLERS_ECEMA zero gain limit', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.ehlers_ecema_js(inputData, 2, 0);
    }, /Invalid gain limit/);
});

test('EHLERS_ECEMA period exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.ehlers_ecema_js(dataSmall, 10, 50);
    }, /Invalid/);
});

test('EHLERS_ECEMA very small dataset', () => {
    
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.ehlers_ecema_js(singlePoint, 20, 50);
    }, /Invalid|Not enough/);
});

test('EHLERS_ECEMA empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.ehlers_ecema_js(empty, 20, 50);
    }, /empty|Empty/);
});

test('EHLERS_ECEMA all NaN input', () => {
    
    const allNan = new Float64Array([NaN, NaN, NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.ehlers_ecema_js(allNan, 2, 50);
    }, /All values are NaN/);
});

test('EHLERS_ECEMA invalid gain limit', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0]);
    
    
    assert.throws(() => {
        wasm.ehlers_ecema_js(inputData, 3, 0);
    }, /Invalid gain limit/);
    
    
    
    
});

test('EHLERS_ECEMA reinput', () => {
    
    const expected = EXPECTED_OUTPUTS.ehlersEcema;
    
    const data = new Float64Array(testData.close);
    
    
    const length = expected.reinputParams.length;
    const gainLimit = expected.reinputParams.gainLimit;
    
    
    const firstResult = wasm.ehlers_ecema_js(data, length, gainLimit);
    assert.strictEqual(firstResult.length, data.length);
    
    
    const secondResult = wasm.ehlers_ecema_js(firstResult, length, gainLimit);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    
    const warmupPeriod = length - 1;
    for (let i = 0; i < warmupPeriod; i++) {
        assert(isNaN(firstResult[i]), `First pass should have NaN in warmup at index ${i}`);
    }
    
    
    
    
    const secondWarmup = warmupPeriod + warmupPeriod;
    for (let i = 0; i < secondWarmup; i++) {
        assert(isNaN(secondResult[i]), `Second pass should have NaN in extended warmup at index ${i}`);
    }
    
    
    assert(!isNaN(firstResult[warmupPeriod]), "First pass should have valid values after warmup");
    
    const validIndices = secondResult.reduce((acc, val, idx) => {
        if (!isNaN(val)) acc.push(idx);
        return acc;
    }, []);
    assert(validIndices.length > 0, "Second pass should have some valid values");
    
    
    const last5 = secondResult.slice(-5);
    assertArrayClose(
        last5,
        expected.reinputLast5,
        1e-8,
        "EHLERS_ECEMA re-input last 5 values mismatch"
    );
});

test('EHLERS_ECEMA NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.ehlers_ecema_js(close, 20, 50);
    assert.strictEqual(result.length, close.length);
    
    
    const warmupPeriod = 19; 
    for (let i = 0; i < warmupPeriod; i++) {
        assert(isNaN(result[i]), `Expected NaN in warmup period at index ${i}`);
    }
    
    
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN after warmup period at index ${i}`);
        }
    }
    
    
    if (result.length > warmupPeriod) {
        assert(!isNaN(result[warmupPeriod]), `Expected valid value at index ${warmupPeriod}`);
    }
});

test('EHLERS_ECEMA memory management', () => {
    if (!HAS_WASM_MEMORY) {
        console.log('Skipping memory management test: wasm memory export unavailable');
        return;
    }
    
    const expected = EXPECTED_OUTPUTS.ehlersEcema;
    
    const data = new Float64Array(testData.close.slice(0, 25));
    
    const len = data.length;
    const ptr = wasm.ehlers_ecema_alloc(len);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    
    const memView = new Float64Array(wasmInst.memory.buffer, ptr, len);
    memView.set(data);
    
    
    const outPtr = wasm.ehlers_ecema_alloc(len);
    
    
    wasm.ehlers_ecema_into(ptr, outPtr, len, 20, 50);
    
    
    const result = new Float64Array(wasmInst.memory.buffer, outPtr, len);
    const resultCopy = Array.from(result);
    
    
    wasm.ehlers_ecema_free(ptr, len);
    wasm.ehlers_ecema_free(outPtr, len);
    
    
    const regularResult = wasm.ehlers_ecema_js(data, 20, 50);
    assertArrayClose(resultCopy, regularResult, 1e-10, "Memory management test failed");
});

test('EHLERS_ECEMA in-place computation', () => {
    if (!HAS_WASM_MEMORY) {
        console.log('Skipping in-place computation test: wasm memory export unavailable');
        return;
    }
    
    const expected = EXPECTED_OUTPUTS.ehlersEcema;
    
    const data = new Float64Array(testData.close.slice(0, 25));
    
    const len = data.length;
    const ptr = wasm.ehlers_ecema_alloc(len);
    
    
    const memView = new Float64Array(wasmInst.memory.buffer, ptr, len);
    memView.set(data);
    
    
    wasm.ehlers_ecema_into(ptr, ptr, len, 20, 50);
    
    
    const result = new Float64Array(wasmInst.memory.buffer, ptr, len);
    const resultCopy = Array.from(result);
    
    
    wasm.ehlers_ecema_free(ptr, len);
    
    
    const regularResult = wasm.ehlers_ecema_js(data, 20, 50);
    assertArrayClose(resultCopy, regularResult, 1e-10, "In-place computation test failed");
});


test('EHLERS_ECEMA batch single parameter set', () => {
    
    if (!wasm.ehlers_ecema_batch) {
        console.log('EHLERS_ECEMA batch API not yet available');
        return;
    }
    
    const expected = EXPECTED_OUTPUTS.ehlersEcema;
    const close = new Float64Array(testData.close);
    
    
    const batchResult = wasm.ehlers_ecema_batch(close, {
        length_range: [expected.defaultParams.length, expected.defaultParams.length, 0],
        gain_limit_range: [expected.defaultParams.gainLimit, expected.defaultParams.gainLimit, 0]
    });
    
    
    const singleResult = wasm.ehlers_ecema_js(close, expected.defaultParams.length, expected.defaultParams.gainLimit);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
    
    
    assert.strictEqual(batchResult.rows, 1);
    assert.strictEqual(batchResult.cols, close.length);
    assert.strictEqual(batchResult.combos.length, 1);
    assert.strictEqual(batchResult.combos[0].length, expected.defaultParams.length);
    assert.strictEqual(batchResult.combos[0].gain_limit, expected.defaultParams.gainLimit);
});

test('EHLERS_ECEMA batch multiple parameters', () => {
    
    if (!wasm.ehlers_ecema_batch) {
        console.log('EHLERS_ECEMA batch API not yet available');
        return;
    }
    
    const expected = EXPECTED_OUTPUTS.ehlersEcema;
    const close = new Float64Array(testData.close.slice(0, 100)); 
    
    
    const batchResult = wasm.ehlers_ecema_batch(close, {
        length_range: expected.batchParams.lengthRange,
        gain_limit_range: expected.batchParams.gainLimitRange
    });
    
    
    assert.strictEqual(batchResult.rows, expected.batchCombinations);
    assert.strictEqual(batchResult.cols, 100);
    assert.strictEqual(batchResult.values.length, expected.batchCombinations * 100);
    
    
    assert.strictEqual(batchResult.combos.length, expected.batchCombinations);
    
    
    const lengths = [15, 20, 25];
    const gains = [40, 50, 60];
    let rowIdx = 0;
    
    for (const length of lengths) {
        for (const gain of gains) {
            const rowStart = rowIdx * 100;
            const rowEnd = rowStart + 100;
            const rowData = batchResult.values.slice(rowStart, rowEnd);
            
            const singleResult = wasm.ehlers_ecema_js(close, length, gain);
            assertArrayClose(
                rowData,
                singleResult,
                1e-10,
                `Length ${length}, gain ${gain} mismatch`
            );
            rowIdx++;
        }
    }
});


test('EHLERS_ECEMA_into_ex with mode flags', () => {
    if (!HAS_WASM_MEMORY) {
        console.log('Skipping into_ex flags test: wasm memory export unavailable');
        return;
    }
    
    if (!wasm.ehlers_ecema_into_ex) {
        console.log('ehlers_ecema_into_ex not yet available');
        return;
    }
    
    
    const data = new Float64Array(testData.close.slice(0, 25));
    
    const len = data.length;
    
    
    const testCases = [
        { pine: false, confirmed: false, desc: "Regular mode" },
        { pine: true, confirmed: false, desc: "Pine mode" },
        { pine: false, confirmed: true, desc: "Confirmed mode" },
        { pine: true, confirmed: true, desc: "Pine + Confirmed mode" }
    ];
    
    for (const testCase of testCases) {
        const inPtr = wasm.ehlers_ecema_alloc(len);
        const outPtr = wasm.ehlers_ecema_alloc(len);
        
        
        const inMem = new Float64Array(wasmInst.memory.buffer, inPtr, len);
        inMem.set(data);
        
        
        wasm.ehlers_ecema_into_ex(
            inPtr, outPtr, len, 20, 50,
            testCase.pine, testCase.confirmed
        );
        
        
        const outMem = new Float64Array(wasmInst.memory.buffer, outPtr, len);
        const result = Array.from(outMem);
        
        
        wasm.ehlers_ecema_free(inPtr, len);
        wasm.ehlers_ecema_free(outPtr, len);
        
        
        assert.strictEqual(result.length, len, `${testCase.desc}: Length mismatch`);
        
        
        if (testCase.pine) {
            
            assert(!isNaN(result[0]), `${testCase.desc}: Should have value at index 0`);
        } else {
            
            for (let i = 0; i < 19; i++) {
                assert(isNaN(result[i]), `${testCase.desc}: Expected NaN at index ${i}`);
            }
            assert(!isNaN(result[19]), `${testCase.desc}: Expected value at index 19`);
        }
    }
});


test('EHLERS_ECEMA zero-copy memory management', () => {
    if (!HAS_WASM_MEMORY) {
        console.log('Skipping zero-copy memory test: wasm memory export unavailable');
        return;
    }
    
    const sizes = [100, 1000, 10000];
    
    for (const size of sizes) {
        const ptr = wasm.ehlers_ecema_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        
        const memView = new Float64Array(wasmInst.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 2.5;
        }
        
        
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 2.5, `Memory corruption at index ${i}`);
        }
        
        
        wasm.ehlers_ecema_free(ptr, size);
    }
});


test('EHLERS_ECEMA SIMD128 consistency', () => {
    
    const testCases = [
        { size: 25, length: 10, gain: 30 },
        { size: 100, length: 20, gain: 50 },
        { size: 1000, length: 30, gain: 70 }
    ];
    
    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = Math.sin(i * 0.1) * 100 + 50000;
        }
        
        const result = wasm.ehlers_ecema_js(data, testCase.length, testCase.gain);
        
        
        assert.strictEqual(result.length, data.length);
        
        
        for (let i = 0; i < testCase.length - 1; i++) {
            assert(isNaN(result[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}`);
        }
        
        
        let sumAfterWarmup = 0;
        let countAfterWarmup = 0;
        for (let i = testCase.length - 1; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for size=${testCase.size}`);
            sumAfterWarmup += result[i];
            countAfterWarmup++;
        }
        
        
        const avgAfterWarmup = sumAfterWarmup / countAfterWarmup;
        assert(avgAfterWarmup > 40000 && avgAfterWarmup < 60000, 
               `Average value ${avgAfterWarmup} seems unreasonable`);
    }
});
