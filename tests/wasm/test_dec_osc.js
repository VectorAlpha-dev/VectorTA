/**
 * WASM binding tests for Decycler Oscillator (DEC_OSC) indicator.
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

test('DEC_OSC partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.dec_osc_js(close, 125, 1.0);
    assert.strictEqual(result.length, close.length);
});

test('DEC_OSC accuracy', async () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.dec_osc_js(close, 125, 1.0);
    
    assert.strictEqual(result.length, close.length);
    
    
    const expectedLast5 = [
        -1.5036367540303395,
        -1.4037875172207006,
        -1.3174199471429475,
        -1.2245874070642693,
        -1.1638422627265639,
    ];
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLast5,
        1e-7,
        "DEC_OSC last 5 values mismatch"
    );
    
    
    
    
});

test('DEC_OSC default candles', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.dec_osc_js(close, 125, 1.0);
    assert.strictEqual(result.length, close.length);
});

test('DEC_OSC zero period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.dec_osc_js(inputData, 0, 1.0);
    }, /Invalid period/);
});

test('DEC_OSC period exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.dec_osc_js(dataSmall, 10, 1.0);
    }, /Invalid period/);
});

test('DEC_OSC very small dataset', () => {
    
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.dec_osc_js(singlePoint, 125, 1.0);
    }, /Invalid period|Not enough valid data/);
});

test('DEC_OSC empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.dec_osc_js(empty, 125, 1.0);
    }, /Input data slice is empty/);
});

test('DEC_OSC invalid k', () => {
    
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    
    assert.throws(() => {
        wasm.dec_osc_js(data, 2, 0.0);
    }, /Invalid K/);
    
    
    assert.throws(() => {
        wasm.dec_osc_js(data, 2, -1.0);
    }, /Invalid K/);
    
    
    assert.throws(() => {
        wasm.dec_osc_js(data, 2, NaN);
    }, /Invalid K/);
});

test('DEC_OSC reinput', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const firstResult = wasm.dec_osc_js(close, 50, 1.0);
    assert.strictEqual(firstResult.length, close.length);
    
    
    const secondResult = wasm.dec_osc_js(firstResult, 50, 1.0);
    assert.strictEqual(secondResult.length, firstResult.length);
});

test('DEC_OSC NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    const hpPeriod = 10;
    
    
    const result = wasm.dec_osc_js(close, hpPeriod, 1.0);
    assert.strictEqual(result.length, close.length);
    
    
    const warmupPeriod = 2;
    
    
    assertAllNaN(result.slice(0, warmupPeriod), `Expected NaN in first ${warmupPeriod} values (warmup period)`);
    
    
    if (result.length > warmupPeriod + 100) {
        for (let i = warmupPeriod + 100; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i} (after warmup)`);
        }
    }
    
    
    const dataWithNaN = new Float64Array(close);
    for (let i = 0; i < 5; i++) {
        dataWithNaN[i] = NaN;
    }
    
    const resultWithNaN = wasm.dec_osc_js(dataWithNaN, hpPeriod, 1.0);
    assert.strictEqual(resultWithNaN.length, dataWithNaN.length);
    
    
    assertAllNaN(resultWithNaN.slice(0, 5), "Expected NaN propagation from input NaNs");
});

test('DEC_OSC warmup period', () => {
    
    const close = new Float64Array(testData.close);
    const testPeriods = [5, 10, 20, 50, 125];
    
    for (const hpPeriod of testPeriods) {
        const result = wasm.dec_osc_js(close, hpPeriod, 1.0);
        
        
        const expectedWarmup = 2;
        
        
        for (let i = 0; i < Math.min(expectedWarmup, result.length); i++) {
            assert(isNaN(result[i]), `Expected NaN at index ${i} for hp_period=${hpPeriod}`);
        }
        
        
        if (result.length > expectedWarmup) {
            assert(!isNaN(result[expectedWarmup]), `Unexpected NaN at index ${expectedWarmup} (after warmup) for hp_period=${hpPeriod}`);
        }
    }
});

test('DEC_OSC batch calculation', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const result = wasm.dec_osc_batch(close, {
        hp_period_range: [100, 150, 25],
        k_range: [0.5, 1.5, 0.5]
    });
    
    
    assert(result.values !== undefined, 'Result should have values');
    assert(result.combos !== undefined, 'Result should have combos');
    assert(result.rows !== undefined, 'Result should have rows');
    assert(result.cols !== undefined, 'Result should have cols');
    
    
    const expectedCombinations = 3 * 3; 
    assert.strictEqual(result.rows, expectedCombinations);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.values.length, expectedCombinations * close.length);
    assert.strictEqual(result.combos.length, expectedCombinations);
    
    
    const expectedPeriods = [100, 100, 100, 125, 125, 125, 150, 150, 150];
    const expectedKs = [0.5, 1.0, 1.5, 0.5, 1.0, 1.5, 0.5, 1.0, 1.5];
    
    for (let i = 0; i < expectedCombinations; i++) {
        assert.strictEqual(result.combos[i].hp_period, expectedPeriods[i]);
        assertClose(result.combos[i].k, expectedKs[i], 1e-10, `k value mismatch at index ${i}`);
        
        
        const rowStart = i * result.cols;
        const rowEnd = rowStart + result.cols;
        const rowData = result.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.dec_osc_js(close, result.combos[i].hp_period, result.combos[i].k);
        assertArrayClose(
            rowData,
            singleResult,
            1e-10,
            `Batch row ${i} (hp_period=${result.combos[i].hp_period}, k=${result.combos[i].k}) doesn't match single calculation`
        );
    }
});

test('DEC_OSC batch single parameter', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const result = wasm.dec_osc_batch(close, {
        hp_period_range: [125, 125, 0],
        k_range: [1.0, 1.0, 0]
    });
    
    
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.combos.length, 1);
    assert.strictEqual(result.values.length, close.length);
    
    
    assert.strictEqual(result.combos[0].hp_period, 125);
    assert.strictEqual(result.combos[0].k, 1.0);
    
    
    const singleResult = wasm.dec_osc_js(close, 125, 1.0);
    assertArrayClose(result.values, singleResult, 1e-10, "Batch single param doesn't match single calculation");
    
    
    const expectedLast5 = [
        -1.5036367540303395,
        -1.4037875172207006,
        -1.3174199471429475,
        -1.2245874070642693,
        -1.1638422627265639,
    ];
    const last5 = result.values.slice(-5);
    assertArrayClose(last5, expectedLast5, 1e-7, "Batch default params last 5 values mismatch");
});

test('DEC_OSC batch metadata from result', () => {
    
    const close = new Float64Array(20); 
    close.fill(100);
    
    const result = wasm.dec_osc_batch(close, {
        hp_period_range: [10, 15, 5],    
        k_range: [0.5, 1.0, 0.25]        
    });
    
    
    assert.strictEqual(result.combos.length, 6);
    
    
    assert.strictEqual(result.combos[0].hp_period, 10);
    assert.strictEqual(result.combos[0].k, 0.5);
    
    
    assert.strictEqual(result.combos[5].hp_period, 15);
    assertClose(result.combos[5].k, 1.0, 1e-10, "k mismatch");
});

test('DEC_OSC edge cases', () => {
    
    
    
    const sameValues = new Float64Array(100).fill(50.0);
    const result1 = wasm.dec_osc_js(sameValues, 10, 1.0);
    assert.strictEqual(result1.length, sameValues.length);
    
    
    const increasing = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        increasing[i] = i;
    }
    const result2 = wasm.dec_osc_js(increasing, 10, 1.0);
    assert.strictEqual(result2.length, increasing.length);
    
    
    const alternating = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        alternating[i] = i % 2 === 0 ? 10.0 : 20.0;
    }
    const result3 = wasm.dec_osc_js(alternating, 10, 1.0);
    assert.strictEqual(result3.length, alternating.length);
});

test('DEC_OSC batch edge cases', () => {
    
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    
    
    const singleBatch = wasm.dec_osc_batch(close, {
        hp_period_range: [5, 5, 1],
        k_range: [1.0, 1.0, 0.1]
    });
    
    assert.strictEqual(singleBatch.values.length, 15);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    
    const largeBatch = wasm.dec_osc_batch(close, {
        hp_period_range: [5, 7, 10], 
        k_range: [1.0, 1.0, 0]
    });
    
    
    assert.strictEqual(largeBatch.values.length, 15);
    assert.strictEqual(largeBatch.combos.length, 1);
    assert.strictEqual(largeBatch.combos[0].hp_period, 5);
    
    
    assert.throws(() => {
        wasm.dec_osc_batch(new Float64Array([]), {
            hp_period_range: [10, 10, 0],
            k_range: [1.0, 1.0, 0]
        });
    }, /All values are NaN|Input data slice is empty/);
});

test('DEC_OSC batch API error handling', () => {
    const close = new Float64Array(testData.close.slice(0, 10));
    
    
    assert.throws(() => {
        wasm.dec_osc_batch(close, {
            hp_period_range: [10, 10], 
            k_range: [1.0, 1.0, 0]
        });
    }, /Invalid config/);
    
    
    assert.throws(() => {
        wasm.dec_osc_batch(close, {
            hp_period_range: [10, 10, 0]
            
        });
    }, /Invalid config/);
    
    
    assert.throws(() => {
        wasm.dec_osc_batch(close, {
            hp_period_range: "invalid",
            k_range: [1.0, 1.0, 0]
        });
    }, /Invalid config/);
});

test('DEC_OSC zero-copy API', () => {
    
    const close = new Float64Array(testData.close);
    const len = close.length;
    const hpPeriod = 125;
    const k = 1.0;
    
    
    const ptr = wasm.dec_osc_alloc(len);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    try {
        
        const memView = new Float64Array(
            wasm.__wasm.memory.buffer,
            ptr,
            len
        );
        
        
        memView.set(close);
        
        
        wasm.dec_osc_into(ptr, ptr, len, hpPeriod, k);
        
        
        const regularResult = wasm.dec_osc_js(close, hpPeriod, k);
        for (let i = 0; i < len; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; 
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        
        wasm.dec_osc_free(ptr, len);
    }
});

test('DEC_OSC zero-copy with large dataset', () => {
    
    const size = 100000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }
    
    const ptr = wasm.dec_osc_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');
    
    try {
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        memView.set(data);
        
        wasm.dec_osc_into(ptr, ptr, size, 125, 1.0);
        
        
        const memView2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        
        
        const warmupPeriod = 2;
        for (let i = 0; i < warmupPeriod; i++) {
            assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
        }
        
        
        for (let i = warmupPeriod; i < Math.min(warmupPeriod + 100, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.dec_osc_free(ptr, size);
    }
});

test('DEC_OSC zero-copy error handling', () => {
    
    assert.throws(() => {
        wasm.dec_osc_into(0, 0, 10, 10, 1.0);
    }, /null pointer|invalid memory/i);
    
    
    const ptr = wasm.dec_osc_alloc(10);
    try {
        
        assert.throws(() => {
            wasm.dec_osc_into(ptr, ptr, 10, 0, 1.0);
        }, /Invalid period/);
        
        
        assert.throws(() => {
            wasm.dec_osc_into(ptr, ptr, 10, 5, 0.0);
        }, /Invalid K/);
    } finally {
        wasm.dec_osc_free(ptr, 10);
    }
});

test('DEC_OSC memory management', () => {
    
    const sizes = [100, 1000, 10000, 100000];
    
    for (const size of sizes) {
        const ptr = wasm.dec_osc_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }
        
        
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
        }
        
        
        wasm.dec_osc_free(ptr, size);
    }
});

test('DEC_OSC SIMD128 consistency', () => {
    
    
    const testCases = [
        { size: 10, hp_period: 5 },
        { size: 100, hp_period: 10 },
        { size: 1000, hp_period: 50 },
        { size: 10000, hp_period: 125 }
    ];
    
    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = Math.sin(i * 0.1) + Math.cos(i * 0.05);
        }
        
        const result = wasm.dec_osc_js(data, testCase.hp_period, 1.0);
        
        
        assert.strictEqual(result.length, data.length);
        
        
        const warmupPeriod = 2;
        for (let i = 0; i < warmupPeriod && i < result.length; i++) {
            assert(isNaN(result[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}`);
        }
        
        
        let sumAfterWarmup = 0;
        let countAfterWarmup = 0;
        for (let i = warmupPeriod; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for size=${testCase.size}`);
            sumAfterWarmup += result[i];
            countAfterWarmup++;
        }
        
        
        
        if (countAfterWarmup > 0) {
            const avgAfterWarmup = sumAfterWarmup / countAfterWarmup;
            
            assert(Math.abs(avgAfterWarmup) < 10000, `Average value ${avgAfterWarmup} seems unreasonable`);
        }
    }
});

test('DEC_OSC all NaN input', () => {
    
    const allNaN = new Float64Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.dec_osc_js(allNaN, 10, 1.0);
    }, /All values are NaN/);
});
