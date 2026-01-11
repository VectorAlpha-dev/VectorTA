/**
 * WASM binding tests for Damiani Volatmeter indicator.
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
import { getRustOutput } from './rust-comparison.js';

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

test('Damiani Volatmeter partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const result = wasm.damiani_volatmeter_js(close, 13, 20, 40, 100, 1.4);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.rows, 2); 
    assert.strictEqual(result.values.length, close.length * 2);
    
    
    const vol = result.values.slice(0, close.length);
    const anti = result.values.slice(close.length);
    
    assert.strictEqual(vol.length, close.length);
    assert.strictEqual(anti.length, close.length);
});

test('Damiani Volatmeter accuracy (close-only binding path)', async () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.damiani_volatmeter;
    
    const result = wasm.damiani_volatmeter_js(
        close, 
        expected.defaultParams.vis_atr,
        expected.defaultParams.vis_std,
        expected.defaultParams.sed_atr,
        expected.defaultParams.sed_std,
        expected.defaultParams.threshold
    );
    
    
    const vol = result.values.slice(0, close.length);
    const anti = result.values.slice(close.length);
    
    
    assertArrayClose(
        vol.slice(-5), 
        expected.volLast5Values,
        1e-2,  
        "Damiani Volatmeter vol last 5 values mismatch"
    );
    assertArrayClose(
        anti.slice(-5), 
        expected.antiLast5Values,
        1e-2,  
        "Damiani Volatmeter anti last 5 values mismatch"
    );
    
    
    
});

test('Damiani Volatmeter Rust reference parity (candles-based)', async () => {
    
    const expected = EXPECTED_OUTPUTS.damiani_volatmeter;
    const rust = await getRustOutput('damiani_volatmeter', 'close'); 

    const n = rust.length;
    const start = n - 5;
    const rustVolLast5 = rust.vol.slice(start);
    const rustAntiLast5 = rust.anti.slice(start);

    
    assertArrayClose(rustVolLast5, expected.rustVolLast5Values, 1e-2, 'Rust vol last 5 mismatch');
    assertArrayClose(rustAntiLast5, expected.rustAntiLast5Values, 1e-2, 'Rust anti last 5 mismatch');
});

test('Damiani Volatmeter zero period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0].concat(Array(120).fill(50.0)));
    
    assert.throws(() => {
        wasm.damiani_volatmeter_js(inputData, 0, 20, 40, 100, 1.4);
    }, /Invalid period/);
});

test('Damiani Volatmeter period exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.damiani_volatmeter_js(dataSmall, 99999, 20, 40, 100, 1.4);
    }, /Invalid period/);
});

test('Damiani Volatmeter very small dataset', () => {
    
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.damiani_volatmeter_js(singlePoint, 9, 9, 9, 9, 1.4);
    }, /Invalid period|Not enough valid data/);
});

test('Damiani Volatmeter empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.damiani_volatmeter_js(empty, 13, 20, 40, 100, 1.4);
    }, /Empty data/);
});

test('Damiani Volatmeter all NaN input', () => {
    
    const allNan = new Float64Array(150).fill(NaN);
    
    assert.throws(() => {
        wasm.damiani_volatmeter_js(allNan, 13, 20, 40, 100, 1.4);
    }, /All values are NaN/);
});

test('Damiani Volatmeter NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.damiani_volatmeter;
    
    const result = wasm.damiani_volatmeter_js(
        close,
        expected.defaultParams.vis_atr,
        expected.defaultParams.vis_std,
        expected.defaultParams.sed_atr,
        expected.defaultParams.sed_std,
        expected.defaultParams.threshold
    );
    const vol = result.values.slice(0, close.length);
    const anti = result.values.slice(close.length);
    
    const warmup = expected.warmupPeriod;
    
    
    let firstNonNanVol = -1;
    for (let i = 0; i < vol.length; i++) {
        if (!isNaN(vol[i])) {
            firstNonNanVol = i;
            break;
        }
    }
    assert.strictEqual(firstNonNanVol, warmup - 1, `Vol first non-NaN at index ${firstNonNanVol}, expected ${warmup - 1}`);
    
    
    let firstNonNanAnti = -1;
    for (let i = 0; i < anti.length; i++) {
        if (!isNaN(anti[i])) {
            firstNonNanAnti = i;
            break;
        }
    }
    assert.strictEqual(firstNonNanAnti, warmup - 1, `Anti first non-NaN at index ${firstNonNanAnti}, expected ${warmup - 1}`);
    
    
    assertAllNaN(vol.slice(0, warmup - 1), `Expected NaN in vol warmup period [0:${warmup - 1})`);
    assertAllNaN(anti.slice(0, warmup - 1), `Expected NaN in anti warmup period [0:${warmup - 1})`);
    
    
    if (vol.length > warmup + 100) {
        assertNoNaN(vol.slice(warmup + 100), "Found unexpected NaN in vol after extended warmup");
        assertNoNaN(anti.slice(warmup + 100), "Found unexpected NaN in anti after extended warmup");
    }
});

test('Damiani Volatmeter fast API (into)', () => {
    
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    
    const inPtr = wasm.damiani_volatmeter_alloc(len);
    const volPtr = wasm.damiani_volatmeter_alloc(len);
    const antiPtr = wasm.damiani_volatmeter_alloc(len);
    
    
    assert(inPtr !== 0, "Failed to allocate input buffer");
    assert(volPtr !== 0, "Failed to allocate vol buffer");
    assert(antiPtr !== 0, "Failed to allocate anti buffer");
    
    try {
        
        const inMem = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inMem.set(close);
        
        
        wasm.damiani_volatmeter_into(
            inPtr, volPtr, antiPtr, len,
            13, 20, 40, 100, 1.4
        );
        
        
        
        const volView = new Float64Array(wasm.__wasm.memory.buffer, volPtr, len);
        const antiView = new Float64Array(wasm.__wasm.memory.buffer, antiPtr, len);
        const vol = Array.from(volView);
        const anti = Array.from(antiView);
        
        
        assert.strictEqual(vol.length, len, "Vol array length mismatch");
        assert.strictEqual(anti.length, len, "Anti array length mismatch");
        
        
        const safeResult = wasm.damiani_volatmeter_js(close, 13, 20, 40, 100, 1.4);
        const safeVol = safeResult.values.slice(0, len);
        const safeAnti = safeResult.values.slice(len, 2 * len);
        
        
        assertArrayClose(vol, safeVol, 1e-12, "Fast API vol mismatch");
        assertArrayClose(anti, safeAnti, 1e-12, "Fast API anti mismatch");
    } finally {
        
        wasm.damiani_volatmeter_free(inPtr, len);
        wasm.damiani_volatmeter_free(volPtr, len);
        wasm.damiani_volatmeter_free(antiPtr, len);
    }
});

test.skip('Damiani Volatmeter fast API aliasing (vol)', () => {
    
    const data = new Float64Array(testData.close);
    const len = data.length;
    
    
    const inPtr = wasm.damiani_volatmeter_alloc(len);
    const antiPtr = wasm.damiani_volatmeter_alloc(len);
    
    try {
        
        const inMem = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inMem.set(data);
        
        
        wasm.damiani_volatmeter_into(
            inPtr, inPtr, antiPtr, len,
            13, 20, 40, 100, 1.4
        );
        
        
        const volView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);  
        const antiView = new Float64Array(wasm.__wasm.memory.buffer, antiPtr, len);
        const vol = Array.from(volView);
        const anti = Array.from(antiView);
        
        
        const freshData = new Float64Array(testData.close);
        const safeResult = wasm.damiani_volatmeter_js(freshData, 13, 20, 40, 100, 1.4);
        const safeVol = safeResult.values.slice(0, len);
        const safeAnti = safeResult.values.slice(len, 2 * len);
        
        assertArrayClose(vol, safeVol, 1e-12, "Aliased vol mismatch");
        assertArrayClose(anti, safeAnti, 1e-12, "Aliased anti mismatch");
    } finally {
        wasm.damiani_volatmeter_free(inPtr, len);
        wasm.damiani_volatmeter_free(antiPtr, len);
    }
});

test.skip('Damiani Volatmeter fast API aliasing (anti)', () => {
    
    const data = new Float64Array(testData.close);
    const len = data.length;
    
    
    const inPtr = wasm.damiani_volatmeter_alloc(len);
    const volPtr = wasm.damiani_volatmeter_alloc(len);
    
    try {
        
        const inMem = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inMem.set(data);
        
        
        wasm.damiani_volatmeter_into(
            inPtr, volPtr, inPtr, len,
            13, 20, 40, 100, 1.4
        );
        
        
        const volView = new Float64Array(wasm.__wasm.memory.buffer, volPtr, len);
        const antiView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);  
        const vol = Array.from(volView);
        const anti = Array.from(antiView);
        
        
        const freshData = new Float64Array(testData.close);
        const safeResult = wasm.damiani_volatmeter_js(freshData, 13, 20, 40, 100, 1.4);
        const safeVol = safeResult.values.slice(0, len);
        const safeAnti = safeResult.values.slice(len, 2 * len);
        
        assertArrayClose(vol, safeVol, 1e-12, "Aliased vol mismatch");
        assertArrayClose(anti, safeAnti, 1e-12, "Aliased anti mismatch");
    } finally {
        wasm.damiani_volatmeter_free(inPtr, len);
        wasm.damiani_volatmeter_free(volPtr, len);
    }
});

test('Damiani Volatmeter fast API vol/anti same pointer error', () => {
    
    const data = new Float64Array(testData.close);
    const len = data.length;
    
    
    const inPtr = wasm.damiani_volatmeter_alloc(len);
    const ptr = wasm.damiani_volatmeter_alloc(len);
    
    try {
        
        const inMem = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inMem.set(data);
        
        assert.throws(() => {
            wasm.damiani_volatmeter_into(
                inPtr, ptr, ptr, len,  
                13, 20, 40, 100, 1.4
            );
        }, /vol_ptr and anti_ptr cannot be the same/);
    } finally {
        wasm.damiani_volatmeter_free(inPtr, len);
        wasm.damiani_volatmeter_free(ptr, len);
    }
});

test('Damiani Volatmeter batch processing', () => {
    
    const close = new Float64Array(testData.close);
    
    const config = {
        vis_atr: [13, 40, 1],  
        vis_std: [20, 40, 1],  
        sed_atr: [40, 40, 0],  
        sed_std: [100, 100, 0], 
        threshold: [1.4, 1.4, 0.0] 
    };
    
    const result = wasm.damiani_volatmeter_batch(close, config);
    
    assert(result.vol);
    assert(result.anti);
    assert(result.combos);
    assert.strictEqual(result.cols, close.length);
    
    
    
    
    
    
    
    
    const expectedRows = 28 * 21;  
    assert.strictEqual(result.rows, expectedRows);
    assert.strictEqual(result.vol.length, expectedRows * close.length);
    assert.strictEqual(result.anti.length, expectedRows * close.length);
    assert.strictEqual(result.combos.length, expectedRows);
});

test('Damiani Volatmeter different thresholds', () => {
    
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    const thresholds = [0.5, 1.0, 1.4, 2.0];
    const results = [];
    
    for (const thresh of thresholds) {
        const result = wasm.damiani_volatmeter_js(close, 13, 20, 40, 100, thresh);
        const vol = result.values.slice(0, len);
        const anti = result.values.slice(len);
        results.push({ vol, anti });
    }
    
    
    for (let i = 0; i < thresholds.length - 1; i++) {
        const anti1 = results[i].anti;
        const anti2 = results[i + 1].anti;
        
        
        const validIndices = [];
        for (let j = 0; j < len; j++) {
            if (!isNaN(anti1[j]) && !isNaN(anti2[j])) {
                validIndices.push(j);
            }
        }
        
        if (validIndices.length > 0) {
            
            const expectedDiff = thresholds[i + 1] - thresholds[i];
            for (const idx of validIndices) {
                const diff = anti2[idx] - anti1[idx];
                assertClose(diff, expectedDiff, 1e-10, 1e-10,
                    `Anti values should differ by threshold difference at index ${idx}`);
            }
        }
    }
});

test('Damiani Volatmeter batch fast API (batch_into)', () => {
    
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    
    const vis_atr_start = 10, vis_atr_end = 12, vis_atr_step = 1;  
    const vis_std_start = 18, vis_std_end = 20, vis_std_step = 1;  
    const sed_atr_start = 40, sed_atr_end = 40, sed_atr_step = 0;  
    const sed_std_start = 100, sed_std_end = 100, sed_std_step = 0; 
    const threshold_start = 1.4, threshold_end = 1.4, threshold_step = 0.0; 
    
    const expectedRows = 3 * 3 * 1 * 1 * 1;  
    
    
    const inPtr = wasm.damiani_volatmeter_alloc(len);
    const volPtr = wasm.damiani_volatmeter_alloc(expectedRows * len);
    const antiPtr = wasm.damiani_volatmeter_alloc(expectedRows * len);
    
    try {
        
        const inMem = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inMem.set(close);
        
        const rows = wasm.damiani_volatmeter_batch_into(
            inPtr, volPtr, antiPtr, len,
            vis_atr_start, vis_atr_end, vis_atr_step,
            vis_std_start, vis_std_end, vis_std_step,
            sed_atr_start, sed_atr_end, sed_atr_step,
            sed_std_start, sed_std_end, sed_std_step,
            threshold_start, threshold_end, threshold_step
        );
        
        assert.strictEqual(rows, expectedRows, "Unexpected number of rows");
        
        
        const volView = new Float64Array(wasm.__wasm.memory.buffer, volPtr, expectedRows * len);
        const antiView = new Float64Array(wasm.__wasm.memory.buffer, antiPtr, expectedRows * len);
        const vol = Array.from(volView);
        const anti = Array.from(antiView);
        
        
        for (let row = 0; row < expectedRows; row++) {
            const rowStart = row * len;
            const rowEnd = rowStart + len;
            const volRow = vol.slice(rowStart, rowEnd);
            const antiRow = anti.slice(rowStart, rowEnd);
            
            
            const nonNanVol = volRow.slice(120).some(v => !isNaN(v));
            const nonNanAnti = antiRow.slice(120).some(v => !isNaN(v));
            assert(nonNanVol, `Row ${row} vol has no non-NaN values after warmup`);
            assert(nonNanAnti, `Row ${row} anti has no non-NaN values after warmup`);
        }
    } finally {
        wasm.damiani_volatmeter_free(inPtr, len);
        wasm.damiani_volatmeter_free(volPtr, expectedRows * len);
        wasm.damiani_volatmeter_free(antiPtr, expectedRows * len);
    }
});

test('Damiani Volatmeter warmup verification', () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.damiani_volatmeter;
    
    
    const result = wasm.damiani_volatmeter_js(
        close,
        expected.defaultParams.vis_atr,
        expected.defaultParams.vis_std,
        expected.defaultParams.sed_atr,
        expected.defaultParams.sed_std,
        expected.defaultParams.threshold
    );
    const vol = result.values.slice(0, close.length);
    const anti = result.values.slice(close.length);
    
    
    const calculatedWarmup = Math.max(13, 20, 40, 100, 3) + 1;  
    assert.strictEqual(calculatedWarmup, expected.warmupPeriod);
    
    
    for (let i = 0; i < calculatedWarmup - 1; i++) {
        assert(isNaN(vol[i]), `Vol should be NaN at index ${i}`);
        assert(isNaN(anti[i]), `Anti should be NaN at index ${i}`);
    }
    
    
    assert(!isNaN(vol[calculatedWarmup - 1]), `Vol should not be NaN at index ${calculatedWarmup - 1}`);
    assert(!isNaN(anti[calculatedWarmup - 1]), `Anti should not be NaN at index ${calculatedWarmup - 1}`);
    
    
    const result2 = wasm.damiani_volatmeter_js(close, 5, 10, 15, 20, 1.0);
    const vol2 = result2.values.slice(0, close.length);
    const calculatedWarmup2 = Math.max(5, 10, 15, 20, 3) + 1;  
    
    
    for (let i = 0; i < calculatedWarmup2 - 1; i++) {
        assert(isNaN(vol2[i]), `Vol2 should be NaN at index ${i}`);
    }
    assert(!isNaN(vol2[calculatedWarmup2 - 1]), `Vol2 should not be NaN at index ${calculatedWarmup2 - 1}`);
});

test('Damiani Volatmeter batch metadata validation', () => {
    
    
    const close = new Float64Array(testData.close.slice(0, 150));
    
    const config = {
        vis_atr: [10, 14, 2],    
        vis_std: [18, 22, 2],    
        sed_atr: [40, 40, 0],    
        sed_std: [100, 100, 0],  
        threshold: [1.4, 1.4, 0.0] 
    };
    
    const result = wasm.damiani_volatmeter_batch(close, config);
    
    
    assert.strictEqual(result.combos.length, 9);
    assert.strictEqual(result.rows, 9);
    assert.strictEqual(result.cols, 150);
    
    
    assert.strictEqual(result.combos[0].vis_atr, 10);
    assert.strictEqual(result.combos[0].vis_std, 18);
    assert.strictEqual(result.combos[0].sed_atr, 40);
    assert.strictEqual(result.combos[0].sed_std, 100);
    assert.strictEqual(result.combos[0].threshold, 1.4);
    
    
    assert.strictEqual(result.combos[8].vis_atr, 14);
    assert.strictEqual(result.combos[8].vis_std, 22);
    assert.strictEqual(result.combos[8].sed_atr, 40);
    assert.strictEqual(result.combos[8].sed_std, 100);
    assert.strictEqual(result.combos[8].threshold, 1.4);
    
    
    for (let combo = 0; combo < result.combos.length; combo++) {
        const params = result.combos[combo];
        const warmup = Math.max(params.vis_atr, params.vis_std, params.sed_atr, params.sed_std, 3) + 1;
        
        const rowStart = combo * 150;
        const volRow = result.vol.slice(rowStart, rowStart + 150);
        const antiRow = result.anti.slice(rowStart, rowStart + 150);
        
        
        for (let i = 0; i < Math.min(warmup - 1, 150); i++) {
            assert(isNaN(volRow[i]), `Expected NaN at warmup index ${i} for combo ${combo}`);
            assert(isNaN(antiRow[i]), `Expected NaN at warmup index ${i} for combo ${combo}`);
        }
        
        
        if (warmup - 1 < 150) {
            assert(!isNaN(volRow[warmup - 1]), `Unexpected NaN at index ${warmup - 1} for combo ${combo}`);
            assert(!isNaN(antiRow[warmup - 1]), `Unexpected NaN at index ${warmup - 1} for combo ${combo}`);
        }
    }
});

test('Damiani Volatmeter batch accuracy', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    const expected = EXPECTED_OUTPUTS.damiani_volatmeter;
    
    
    const singleResult = wasm.damiani_volatmeter_js(
        close,
        expected.defaultParams.vis_atr,
        expected.defaultParams.vis_std,
        expected.defaultParams.sed_atr,
        expected.defaultParams.sed_std,
        expected.defaultParams.threshold
    );
    const singleVol = singleResult.values.slice(0, 100);
    const singleAnti = singleResult.values.slice(100, 200);
    
    
    const config = {
        vis_atr: [13, 13, 0],
        vis_std: [20, 20, 0],
        sed_atr: [40, 40, 0],
        sed_std: [100, 100, 0],
        threshold: [1.4, 1.4, 0.0]
    };
    
    const batchResult = wasm.damiani_volatmeter_batch(close, config);
    
    assert.strictEqual(batchResult.rows, 1);
    const batchVol = batchResult.vol.slice(0, 100);
    const batchAnti = batchResult.anti.slice(0, 100);
    
    
    assertArrayClose(batchVol, singleVol, 1e-10, "Batch vol doesn't match single calculation");
    assertArrayClose(batchAnti, singleAnti, 1e-10, "Batch anti doesn't match single calculation");
});

test('Damiani Volatmeter memory growth handling', () => {
    
    const size = 10000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }
    
    const inPtr = wasm.damiani_volatmeter_alloc(size);
    const volPtr = wasm.damiani_volatmeter_alloc(size);
    const antiPtr = wasm.damiani_volatmeter_alloc(size);
    
    try {
        
        const inMem = new Float64Array(wasm.__wasm.memory.buffer, inPtr, size);
        inMem.set(data);
        
        
        const originalMemSize = wasm.__wasm.memory.buffer.byteLength;
        
        
        wasm.damiani_volatmeter_into(
            inPtr, volPtr, antiPtr, size,
            13, 20, 40, 100, 1.4
        );
        
        
        const newMemSize = wasm.__wasm.memory.buffer.byteLength;
        
        
        const vol = new Float64Array(wasm.__wasm.memory.buffer, volPtr, size);
        const anti = new Float64Array(wasm.__wasm.memory.buffer, antiPtr, size);
        
        
        let nonNanCount = 0;
        for (let i = 101; i < Math.min(200, size); i++) {
            if (!isNaN(vol[i]) && !isNaN(anti[i])) {
                nonNanCount++;
            }
        }
        assert(nonNanCount > 50, "Should have valid values after warmup");
        
        if (newMemSize > originalMemSize) {
            console.log(`Memory grew from ${originalMemSize} to ${newMemSize} bytes`);
        }
    } finally {
        wasm.damiani_volatmeter_free(inPtr, size);
        wasm.damiani_volatmeter_free(volPtr, size);
        wasm.damiani_volatmeter_free(antiPtr, size);
    }
});

test('Damiani Volatmeter SIMD128 consistency', () => {
    
    const testCases = [
        { size: 50, params: { vis_atr: 5, vis_std: 10, sed_atr: 15, sed_std: 20, threshold: 1.0 } },
        { size: 100, params: { vis_atr: 13, vis_std: 20, sed_atr: 40, sed_std: 100, threshold: 1.4 } },
        { size: 500, params: { vis_atr: 20, vis_std: 30, sed_atr: 50, sed_std: 80, threshold: 1.5 } }
    ];
    
    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = 100 + Math.sin(i * 0.1) * 10 + Math.cos(i * 0.05) * 5;
        }
        
        const result = wasm.damiani_volatmeter_js(
            data,
            testCase.params.vis_atr,
            testCase.params.vis_std,
            testCase.params.sed_atr,
            testCase.params.sed_std,
            testCase.params.threshold
        );
        
        const vol = result.values.slice(0, data.length);
        const anti = result.values.slice(data.length);
        
        
        assert.strictEqual(vol.length, data.length);
        assert.strictEqual(anti.length, data.length);
        
        
        const warmup = Math.max(
            testCase.params.vis_atr,
            testCase.params.vis_std,
            testCase.params.sed_atr,
            testCase.params.sed_std,
            3
        ) + 1;
        
        for (let i = 0; i < Math.min(warmup - 1, vol.length); i++) {
            assert(isNaN(vol[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}`);
            assert(isNaN(anti[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}`);
        }
        
        
        if (warmup - 1 < vol.length) {
            let validCount = 0;
            for (let i = warmup - 1; i < vol.length; i++) {
                if (!isNaN(vol[i]) && !isNaN(anti[i])) {
                    validCount++;
                }
            }
            assert(validCount > 0, `No valid values after warmup for size=${testCase.size}`);
        }
    }
});

test('Damiani Volatmeter partial NaN handling', () => {
    
    const close = new Float64Array(testData.close.slice(0, 200));
    
    
    for (let i = 50; i < 60; i++) {
        close[i] = NaN;  
    }
    close[150] = NaN;  
    
    const result = wasm.damiani_volatmeter_js(close, 13, 20, 40, 100, 1.4);
    const vol = result.values.slice(0, close.length);
    const anti = result.values.slice(close.length);
    
    assert.strictEqual(vol.length, close.length);
    assert.strictEqual(anti.length, close.length);
    
    
    
    
    
    
    
    for (let i = 50; i < 60; i++) {
        
        assert(isNaN(vol[i]), `Vol should be NaN at index ${i} (direct NaN input)`);
        assert(isNaN(anti[i]), `Anti should be NaN at index ${i} (direct NaN input)`);
    }
    
    
    
    
    
    
    
    for (let i = 60; i < 100; i++) {
        
        assert(isNaN(vol[i]), `Vol should be NaN at index ${i} due to NaN in window`);
        assert(isNaN(anti[i]), `Anti should be NaN at index ${i} due to NaN in window`);
    }
    
    
    
    
    
    
    
    
    
    
    let nanAt150 = isNaN(vol[150]);
    if (nanAt150) {
        
        for (let i = 151; i < Math.min(200, vol.length); i++) {
            if (!isNaN(vol[i])) {
                
                
                break;
            }
        }
    }
    
});

test('Damiani Volatmeter invalid parameters', () => {
    
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    
    assert.throws(() => {
        wasm.damiani_volatmeter_js(data, -1, 20, 40, 100, 1.4);
    }, /Invalid period/);
    
    
    assert.throws(() => {
        wasm.damiani_volatmeter_js(data, 13, 0, 40, 100, 1.4);
    }, /Invalid period/);
    
    
    assert.throws(() => {
        wasm.damiani_volatmeter_js(data, 13, 20, -5, 100, 1.4);
    }, /Invalid period/);
    
    
    assert.throws(() => {
        wasm.damiani_volatmeter_js(data, 13, 20, 40, 0, 1.4);
    }, /Invalid period/);
    
    
    
    const nanResult = wasm.damiani_volatmeter_js(data, 2, 2, 2, 2, NaN);
    const nanAnti = nanResult.values.slice(data.length);
    assert(nanAnti.slice(2).every(v => isNaN(v)), "Expected NaN anti values with NaN threshold");
    
    
    const negResult = wasm.damiani_volatmeter_js(data, 2, 2, 2, 2, -1.0);
    assert.strictEqual(negResult.values.length, data.length * 2);
});

test('Damiani Volatmeter batch fast API aliasing', () => {
    
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    
    const expectedRows = 9;  
    
    
    const inPtr = wasm.damiani_volatmeter_alloc(len);
    const volPtr = wasm.damiani_volatmeter_alloc(expectedRows * len);
    const antiPtr = wasm.damiani_volatmeter_alloc(expectedRows * len);
    
    try {
        
        const inMem = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inMem.set(close);
        
        
        assert.throws(() => {
            wasm.damiani_volatmeter_batch_into(
                inPtr, volPtr, volPtr, len,  
                10, 12, 1, 18, 20, 1, 40, 40, 0, 100, 100, 0, 1.4, 1.4, 0.0
            );
        }, /vol_ptr and anti_ptr cannot be the same/);
        
        
        const inPtr2 = wasm.damiani_volatmeter_alloc(len);
        const volPtr2 = wasm.damiani_volatmeter_alloc(expectedRows * len);
        const antiPtr2 = wasm.damiani_volatmeter_alloc(expectedRows * len);
        
        
        const inMem2 = new Float64Array(wasm.__wasm.memory.buffer, inPtr2, len);
        inMem2.set(close);
        
        
        const rows = wasm.damiani_volatmeter_batch_into(
            inPtr2, volPtr2, antiPtr2, len,  
            10, 12, 1, 18, 20, 1, 40, 40, 0, 100, 100, 0, 1.4, 1.4, 0.0
        );
        
        assert.strictEqual(rows, expectedRows, "Batch should work");
        
        
        wasm.damiani_volatmeter_free(inPtr2, len);
        wasm.damiani_volatmeter_free(volPtr2, expectedRows * len);
        wasm.damiani_volatmeter_free(antiPtr2, expectedRows * len);
    } finally {
        wasm.damiani_volatmeter_free(inPtr, len);
        wasm.damiani_volatmeter_free(volPtr, expectedRows * len);
        wasm.damiani_volatmeter_free(antiPtr, expectedRows * len);
    }
});
