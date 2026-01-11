/**
 * WASM binding tests for MAMA indicator.
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


function splitMamaResult(result, dataLength) {
    if (result.values && Array.isArray(result.values)) {
        
        const mama = result.values.slice(0, dataLength);
        const fama = result.values.slice(dataLength);
        return { mama, fama };
    } else if (result instanceof Float64Array) {
        
        const mama = result.slice(0, dataLength);
        const fama = result.slice(dataLength);
        return { mama, fama };
    } else {
        throw new Error('Unknown MAMA result format');
    }
}

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

test('MAMA partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.mama(close, 0.5, 0.05);
    assert(result.values, 'Should have values array');
    assert.strictEqual(result.rows, 2); 
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.values.length, close.length * 2); 
    
    
    const { mama, fama } = splitMamaResult(result, close.length);
    assert.strictEqual(mama.length, close.length);
    assert.strictEqual(fama.length, close.length);
});

test('MAMA accuracy', async () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.mama(close, 0.5, 0.05);
    const { mama, fama } = splitMamaResult(result, close.length);
    
    assert.strictEqual(mama.length, close.length);
    assert.strictEqual(fama.length, close.length);
    
    
    for (let i = 0; i < 10; i++) {
        assert(isNaN(mama[i]), `Expected NaN at warmup index ${i} for MAMA`);
        assert(isNaN(fama[i]), `Expected NaN at warmup index ${i} for FAMA`);
    }
    
    
    for (let i = 10; i < Math.min(20, mama.length); i++) {
        assert(isFinite(mama[i]), `MAMA NaN at index ${i}`);
        assert(isFinite(fama[i]), `FAMA NaN at index ${i}`);
    }
    
    
    
    
    const mamaLast5 = mama.slice(-5);
    const famaLast5 = fama.slice(-5);
    
    
    for (let i = 0; i < 5; i++) {
        assert(isFinite(mamaLast5[i]), `MAMA last 5 value ${i} should be finite`);
        assert(isFinite(famaLast5[i]), `FAMA last 5 value ${i} should be finite`);
        assert(mamaLast5[i] > 0, `MAMA last 5 value ${i} should be positive`);
        assert(famaLast5[i] > 0, `FAMA last 5 value ${i} should be positive`);
    }
});

test('MAMA matches Rust reference last-5', async () => {
    
    const close = new Float64Array(testData.close);
    const result = wasm.mama(close, 0.5, 0.05);
    const { mama, fama } = splitMamaResult(result, close.length);
    const { getRustOutput } = await import('./rust-comparison.js');
    const rustOut = await getRustOutput('mama');
    const last5 = 5;
    assertArrayClose(mama.slice(-last5), rustOut.mama_values.slice(-last5), 1e-1, 'MAMA last5 vs Rust');
    assertArrayClose(fama.slice(-last5), rustOut.fama_values.slice(-last5), 2e1, 'FAMA last5 vs Rust');
});

test('MAMA default candles', async () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.mama(close, 0.5, 0.05);
    const { mama, fama } = splitMamaResult(result, close.length);
    
    assert.strictEqual(mama.length, close.length);
    assert.strictEqual(fama.length, close.length);
    
    
});

test('MAMA invalid fast limit', () => {
    
    const inputData = new Float64Array(new Array(30).fill(0).map((_, i) => (i % 3 + 1) * 10));
    
    
    assert.throws(() => {
        wasm.mama(inputData, 0.0, 0.05);
    });
    
    
    assert.throws(() => {
        wasm.mama(inputData, -0.5, 0.05);
    });
});

test('MAMA invalid slow limit', () => {
    
    const inputData = new Float64Array(new Array(30).fill(0).map((_, i) => (i % 3 + 1) * 10));
    
    
    assert.throws(() => {
        wasm.mama(inputData, 0.5, 0.0);
    });
    
    
    assert.throws(() => {
        wasm.mama(inputData, 0.5, -0.05);
    });
});

test('MAMA insufficient data', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.mama(dataSmall, 0.5, 0.05);
    });
});

test('MAMA very small dataset', () => {
    
    const dataMin = new Float64Array(10).fill(42.0);
    
    const result = wasm.mama(dataMin, 0.5, 0.05);
    const { mama, fama } = splitMamaResult(result, dataMin.length);
    
    assert.strictEqual(mama.length, 10);
    assert.strictEqual(fama.length, 10);
    
    
    for (let i = 0; i < 10; i++) {
        assert(isNaN(mama[i]), `Expected NaN at index ${i} for MAMA`);
        assert(isNaN(fama[i]), `Expected NaN at index ${i} for FAMA`);
    }
});

test('MAMA empty input', () => {
    
    const dataEmpty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.mama(dataEmpty, 0.5, 0.05);
    });
});

test('MAMA reinput', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const firstResult = wasm.mama(close, 0.5, 0.05);
    const { mama: firstMama } = splitMamaResult(firstResult, close.length);
    
    
    const secondResult = wasm.mama(new Float64Array(firstMama), 0.7, 0.1);
    const { mama: secondMama, fama: secondFama } = splitMamaResult(secondResult, firstMama.length);
    
    assert.strictEqual(secondMama.length, firstMama.length);
    assert.strictEqual(secondFama.length, firstMama.length);
});

test('MAMA NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.mama(close, 0.5, 0.05);
    const { mama, fama } = splitMamaResult(result, close.length);
    
    assert.strictEqual(mama.length, close.length);
    assert.strictEqual(fama.length, close.length);
    
    
    for (let i = 0; i < 10; i++) {
        assert(isNaN(mama[i]), `Expected NaN at warmup index ${i} for MAMA`);
        assert(isNaN(fama[i]), `Expected NaN at warmup index ${i} for FAMA`);
    }
    
    
    if (mama.length > 10) {
        for (let i = 10; i < mama.length; i++) {
            assert(isFinite(mama[i]), `MAMA NaN at index ${i}`);
            assert(isFinite(fama[i]), `FAMA NaN at index ${i}`);
        }
    }
});

test('MAMA batch', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const batch_result = wasm.mama_batch(
        close, 
        0.3, 0.7, 0.2,    
        0.03, 0.07, 0.02  
    );
    
    
    assert(batch_result.mama, 'Should have mama array');
    assert(batch_result.fama, 'Should have fama array');
    assert(batch_result.combos, 'Should have combos array');
    assert.strictEqual(batch_result.rows, 9); 
    assert.strictEqual(batch_result.cols, close.length);
    assert.strictEqual(batch_result.combos.length, 9);
    
    
    assert.strictEqual(batch_result.mama.length, 9 * close.length);
    assert.strictEqual(batch_result.fama.length, 9 * close.length);
    
    
    
});

test('MAMA different params', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const testCases = [
        [0.3, 0.03],
        [0.5, 0.05],  
        [0.7, 0.07],
        [0.9, 0.1],
    ];
    
    for (const [fast_lim, slow_lim] of testCases) {
        const result = wasm.mama(close, fast_lim, slow_lim);
        const { mama, fama } = splitMamaResult(result, close.length);
        
        assert.strictEqual(mama.length, close.length);
        assert.strictEqual(fama.length, close.length);
        
        
        for (let i = 0; i < 10; i++) {
            assert(isNaN(mama[i]), `Expected NaN at warmup index ${i} for MAMA with params=(${fast_lim}, ${slow_lim})`);
            assert(isNaN(fama[i]), `Expected NaN at warmup index ${i} for FAMA with params=(${fast_lim}, ${slow_lim})`);
        }
        for (let i = 10; i < mama.length; i++) {
            assert(isFinite(mama[i]), `Found NaN at index ${i} for MAMA with params=(${fast_lim}, ${slow_lim})`);
            assert(isFinite(fama[i]), `Found NaN at index ${i} for FAMA with params=(${fast_lim}, ${slow_lim})`);
        }
    }
});

test('MAMA batch performance', () => {
    
    const close = new Float64Array(testData.close.slice(0, 1000)); 
    
    
    const startBatch = performance.now();
    const batchResult = wasm.mama_batch(
        close,
        0.3, 0.7, 0.1,    
        0.04, 0.06, 0.01  
    );
    const batchTime = performance.now() - startBatch;
    
    const startSingle = performance.now();
    const singleMamaResults = [];
    const singleFamaResults = [];
    
    for (const combo of batchResult.combos) {
        const result = wasm.mama(close, combo.fast_limit, combo.slow_limit);
        const { mama, fama } = splitMamaResult(result, close.length);
        singleMamaResults.push(...mama);
        singleFamaResults.push(...fama);
    }
    const singleTime = performance.now() - startSingle;
    
    
    console.log(`Batch time: ${batchTime.toFixed(2)}ms, Single time: ${singleTime.toFixed(2)}ms`);
    
    
    
    
    const rows = batchResult.rows;
    const cols = close.length;
    if (!(Array.isArray(batchResult.combos) && batchResult.combos.length === rows)) {
        throw new Error('Combos metadata missing');
    }
    if (!(batchResult.mama.length === rows * cols && batchResult.fama.length === rows * cols)) {
        throw new Error('Flattened output shape mismatch');
    }
});

test('MAMA edge cases', () => {
    
    
    
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = i + 1;
    }
    const result = wasm.mama(data, 0.5, 0.05);
    const { mama, fama } = splitMamaResult(result, data.length);
    assert.strictEqual(mama.length, data.length);
    assert.strictEqual(fama.length, data.length);
    
    
    for (let i = 0; i < 10; i++) {
        assert(isNaN(mama[i]), `Expected NaN at warmup index ${i}`);
        assert(isNaN(fama[i]), `Expected NaN at warmup index ${i}`);
    }
    for (let i = 10; i < mama.length; i++) {
        assert(isFinite(mama[i]), `MAMA NaN at index ${i}`);
        assert(isFinite(fama[i]), `FAMA NaN at index ${i}`);
    }
    
    
    const constantData = new Float64Array(100).fill(50.0);
    const constantResult = wasm.mama(constantData, 0.5, 0.05);
    const { mama: constantMama, fama: constantFama } = splitMamaResult(constantResult, constantData.length);
    
    assert.strictEqual(constantMama.length, constantData.length);
    assert.strictEqual(constantFama.length, constantData.length);
    
    
    for (let i = 20; i < constantMama.length; i++) {
        assertClose(constantMama[i], 50.0, 1e-6, `MAMA constant value failed at index ${i}`);
        assertClose(constantFama[i], 50.0, 1e-6, `FAMA constant value failed at index ${i}`);
    }
});

test('MAMA batch metadata', () => {
    
    const data = new Float64Array(20);  
    for (let i = 0; i < 20; i++) data[i] = i + 1;
    
    const result = wasm.mama_batch(
        data,
        0.4, 0.6, 0.2,    
        0.04, 0.06, 0.02  
    );
    
    
    assert.strictEqual(result.combos.length, 4);
    assert.strictEqual(result.rows, 4);
    assert.strictEqual(result.cols, data.length);
    
    
    assert.strictEqual(result.combos[0].fast_limit, 0.4);
    assert.strictEqual(result.combos[0].slow_limit, 0.04);
    
    assert.strictEqual(result.combos[1].fast_limit, 0.4);
    assert.strictEqual(result.combos[1].slow_limit, 0.06);
    
    assertClose(result.combos[2].fast_limit, 0.6, 1e-10, 'Fast limit 2');
    assert.strictEqual(result.combos[2].slow_limit, 0.04);
    
    assertClose(result.combos[3].fast_limit, 0.6, 1e-10, 'Fast limit 3');
    assert.strictEqual(result.combos[3].slow_limit, 0.06);
});

test('MAMA warmup period calculation', () => {
    
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const result = wasm.mama(close, 0.5, 0.05);
    const { mama, fama } = splitMamaResult(result, close.length);
    
    
    
    for (let i = 0; i < 10; i++) {
        assert(isNaN(mama[i]), `Expected NaN at warmup index ${i} for MAMA`);
        assert(isNaN(fama[i]), `Expected NaN at warmup index ${i} for FAMA`);
    }
    
    for (let i = 10; i < mama.length; i++) {
        assert(isFinite(mama[i]), `Expected finite value at index ${i} for MAMA`);
        assert(isFinite(fama[i]), `Expected finite value at index ${i} for FAMA`);
    }
});

test('MAMA consistency across calls', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const result1 = wasm.mama(close, 0.5, 0.05);
    const result2 = wasm.mama(close, 0.5, 0.05);
    
    assertArrayClose(result1, result2, 1e-15, "MAMA results not consistent");
});

test('MAMA parameter step precision', () => {
    
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = i + 1;
    }
    
    const batch_result = wasm.mama_batch(
        data,
        0.4, 0.5, 0.1,     
        0.04, 0.05, 0.01   
    );
    
    
    assert.strictEqual(batch_result.rows, 4);
    assert.strictEqual(batch_result.combos.length, 4);
    assert.strictEqual(batch_result.mama.length, 4 * data.length);
    assert.strictEqual(batch_result.fama.length, 4 * data.length);
    
    
    const expectedCombos = [
        { fast_limit: 0.4, slow_limit: 0.04 },
        { fast_limit: 0.4, slow_limit: 0.05 },
        { fast_limit: 0.5, slow_limit: 0.04 },
        { fast_limit: 0.5, slow_limit: 0.05 }
    ];
    
    for (let i = 0; i < 4; i++) {
        assertClose(batch_result.combos[i].fast_limit, expectedCombos[i].fast_limit, 1e-10, `Fast limit ${i}`);
        assertClose(batch_result.combos[i].slow_limit, expectedCombos[i].slow_limit, 1e-10, `Slow limit ${i}`);
    }
});

test('MAMA streaming simulation', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    const fast_limit = 0.5;
    const slow_limit = 0.05;
    
    
    const batchResult = wasm.mama(close, fast_limit, slow_limit);
    const { mama, fama } = splitMamaResult(batchResult, close.length);
    
    
    assert.strictEqual(mama.length, close.length);
    assert.strictEqual(fama.length, close.length);
    
    
    for (let i = 0; i < 10; i++) {
        assert(isNaN(mama[i]), `Expected NaN at warmup index ${i} for MAMA`);
        assert(isNaN(fama[i]), `Expected NaN at warmup index ${i} for FAMA`);
    }
    for (let i = 10; i < mama.length; i++) {
        assert(isFinite(mama[i]), `Expected finite value at index ${i} for MAMA`);
        assert(isFinite(fama[i]), `Expected finite value at index ${i} for FAMA`);
    }
    
    
    let hasDifferentMama = false;
    let hasDifferentFama = false;
    for (let i = 0; i < close.length; i++) {
        if (Math.abs(mama[i] - close[i]) > 1e-9) {
            hasDifferentMama = true;
        }
        if (Math.abs(fama[i] - close[i]) > 1e-9) {
            hasDifferentFama = true;
        }
    }
    assert(hasDifferentMama, "MAMA should produce smoothed values");
    assert(hasDifferentFama, "FAMA should produce smoothed values");
});

test('MAMA large parameter range', () => {
    
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = Math.sin(i * 0.1) * 100 + 1000; 
    }
    
    const result = wasm.mama(data, 0.9, 0.01);
    const { mama, fama } = splitMamaResult(result, data.length);
    
    assert.strictEqual(mama.length, data.length);
    assert.strictEqual(fama.length, data.length);
    
    
    for (let i = 0; i < 10; i++) {
        assert(isNaN(mama[i]), `Expected NaN at warmup index ${i} for MAMA`);
        assert(isNaN(fama[i]), `Expected NaN at warmup index ${i} for FAMA`);
    }
    for (let i = 10; i < mama.length; i++) {
        assert(isFinite(mama[i]), `Expected finite value at index ${i} for MAMA`);
        assert(isFinite(fama[i]), `Expected finite value at index ${i} for FAMA`);
    }
});

test('MAMA all NaN input', () => {
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    
    try {
        const result = wasm.mama(allNaN, 0.5, 0.05);
        
        const { mama, fama } = splitMamaResult(result, allNaN.length);
        for (let i = 0; i < mama.length; i++) {
            assert(isNaN(mama[i]), `Expected NaN at index ${i} for all-NaN input`);
            assert(isNaN(fama[i]), `Expected NaN at index ${i} for all-NaN input`);
        }
    } catch (e) {
        
        assert(e.message.includes('NaN') || e.message.includes('mama'), 'Should throw appropriate error for all NaN input');
    }
});

test('MAMA zero-copy API', () => {
    
    if (typeof wasm.mama_alloc !== 'function' || typeof wasm.mama_into !== 'function' || typeof wasm.mama_free !== 'function') {
        console.log('MAMA zero-copy API not available, skipping test');
        return;
    }
    
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    const fast_limit = 0.5;
    const slow_limit = 0.05;
    
    
    const inPtr = wasm.mama_alloc(data.length);
    const outMamaPtr = wasm.mama_alloc(data.length);
    const outFamaPtr = wasm.mama_alloc(data.length);
    
    assert(inPtr !== 0, 'Failed to allocate input memory');
    assert(outMamaPtr !== 0, 'Failed to allocate MAMA output memory');
    assert(outFamaPtr !== 0, 'Failed to allocate FAMA output memory');
    
    
    const memory = wasm.__wasm.memory;
    const inView = new Float64Array(memory.buffer, inPtr, data.length);
    const mamaView = new Float64Array(memory.buffer, outMamaPtr, data.length);
    const famaView = new Float64Array(memory.buffer, outFamaPtr, data.length);
    
    
    inView.set(data);
    
    
    try {
        wasm.mama_into(inPtr, outMamaPtr, outFamaPtr, data.length, fast_limit, slow_limit);
        
        
        const memory2 = wasm.__wasm.memory;
        const mamaResult = new Float64Array(memory2.buffer, outMamaPtr, data.length);
        const famaResult = new Float64Array(memory2.buffer, outFamaPtr, data.length);
        
        
        const regularResult = wasm.mama(data, fast_limit, slow_limit);
        const { mama: regularMama, fama: regularFama } = splitMamaResult(regularResult, data.length);
        
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularMama[i]) && isNaN(mamaResult[i])) {
                continue; 
            }
            assert(Math.abs(regularMama[i] - mamaResult[i]) < 1e-10,
                   `MAMA zero-copy mismatch at index ${i}: regular=${regularMama[i]}, zerocopy=${mamaResult[i]}`);
            assert(Math.abs(regularFama[i] - famaResult[i]) < 1e-10,
                   `FAMA zero-copy mismatch at index ${i}: regular=${regularFama[i]}, zerocopy=${famaResult[i]}`);
        }
    } finally {
        
        wasm.mama_free(inPtr, data.length);
        wasm.mama_free(outMamaPtr, data.length);
        wasm.mama_free(outFamaPtr, data.length);
    }
});

test('MAMA structured result API', () => {
    
    if (typeof wasm.mama_result !== 'function') {
        console.log('MAMA structured result API not available, skipping test');
        return;
    }
    
    const data = new Float64Array(testData.close.slice(0, 100));
    
    const result = wasm.mama_result(data, 0.5, 0.05);
    
    
    assert(result.values, 'Should have values array');
    assert(typeof result.rows === 'number', 'Should have rows count');
    assert(typeof result.cols === 'number', 'Should have cols count');
    
    
    assert.strictEqual(result.rows, 2); 
    assert.strictEqual(result.cols, data.length);
    assert.strictEqual(result.values.length, 2 * data.length);
    
    
    const mama = result.values.slice(0, data.length);
    const fama = result.values.slice(data.length);
    
    
    for (let i = 0; i < 10; i++) {
        assert(isNaN(mama[i]), `Expected NaN at warmup index ${i} for MAMA`);
        assert(isNaN(fama[i]), `Expected NaN at warmup index ${i} for FAMA`);
    }
});

test('MAMA FAMA relationship', () => {
    
    const close = new Float64Array(testData.close.slice(0, 200));
    
    const result = wasm.mama(close, 0.5, 0.05);
    const { mama, fama } = splitMamaResult(result, close.length);
    
    
    
    let mamaMean = 0, famaMean = 0;
    let count = 0;
    for (let i = 20; i < mama.length; i++) {
        mamaMean += mama[i];
        famaMean += fama[i];
        count++;
    }
    mamaMean /= count;
    famaMean /= count;
    
    let mamaVar = 0, famaVar = 0;
    for (let i = 20; i < mama.length; i++) {
        mamaVar += Math.pow(mama[i] - mamaMean, 2);
        famaVar += Math.pow(fama[i] - famaMean, 2);
    }
    mamaVar /= count;
    famaVar /= count;
    
    
    console.log(`MAMA variance: ${mamaVar.toFixed(6)}, FAMA variance: ${famaVar.toFixed(6)}`);
    assert(famaVar < mamaVar * 1.1, "FAMA should be smoother than MAMA");
});

test.after(() => {
    console.log('MAMA WASM tests completed');
});
