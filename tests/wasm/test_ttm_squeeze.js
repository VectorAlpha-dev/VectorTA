/**
 * WASM binding tests for TTM Squeeze indicator.
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


function extractTtmSqueezeResult(result) {
    
    
    const momentum = result.values.slice(0, result.cols);
    const squeeze = result.values.slice(result.cols, 2 * result.cols);
    return { momentum, squeeze };
}


const TTM_EXPECTED = {
    defaultParams: {
        length: 20,
        bb_mult: 2.0,
        kc_mult_high: 1.0,
        kc_mult_mid: 1.5,
        kc_mult_low: 2.0
    },
    
    
    momentum_first5: [
        -167.98676428571423,  
        -154.99159285714336,  
        -148.98427857142892,  
        -131.80910714285744,  
        -89.35822142857162,   
    ],
    squeeze_first5: [0.0, 0.0, 0.0, 0.0, 1.0],  
    warmupPeriod: 19 
};

test('TTM Squeeze partial params', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const rawResult = wasm.ttm_squeeze(high, low, close, 20, 2.0, 1.0, 1.5, 2.0);
    assert.ok(rawResult, 'Should return a result');
    assert.ok(rawResult.values, 'Should have values array');
    assert.strictEqual(rawResult.rows, 2, 'Should have 2 rows (momentum and squeeze)');
    assert.strictEqual(rawResult.cols, close.length, 'Cols should match input length');
    
    const result = extractTtmSqueezeResult(rawResult);
    assert.strictEqual(result.momentum.length, close.length);
    assert.strictEqual(result.squeeze.length, close.length);
});

test('TTM Squeeze accuracy', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const rawResult = wasm.ttm_squeeze(
        high, low, close,
        TTM_EXPECTED.defaultParams.length,
        TTM_EXPECTED.defaultParams.bb_mult,
        TTM_EXPECTED.defaultParams.kc_mult_high,
        TTM_EXPECTED.defaultParams.kc_mult_mid,
        TTM_EXPECTED.defaultParams.kc_mult_low
    );
    
    const result = extractTtmSqueezeResult(rawResult);
    assert.strictEqual(result.momentum.length, close.length);
    assert.strictEqual(result.squeeze.length, close.length);
    
    
    const startIdx = TTM_EXPECTED.warmupPeriod;
    for (let i = 0; i < TTM_EXPECTED.momentum_first5.length; i++) {
        const actual = result.momentum[startIdx + i];
        const expected = TTM_EXPECTED.momentum_first5[i];
        assertClose(actual, expected, 1e-8, `Momentum mismatch at index ${i}`);
    }
    
    
    for (let i = 0; i < TTM_EXPECTED.squeeze_first5.length; i++) {
        const actual = result.squeeze[startIdx + i];
        const expected = TTM_EXPECTED.squeeze_first5[i];
        assert.strictEqual(actual, expected, `Squeeze mismatch at index ${i}`);
    }
});

test('TTM Squeeze default candles', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const rawResult = wasm.ttm_squeeze(high, low, close, 20, 2.0, 1.0, 1.5, 2.0);
    const result = extractTtmSqueezeResult(rawResult);
    assert.strictEqual(result.momentum.length, close.length);
    assert.strictEqual(result.squeeze.length, close.length);
});

test('TTM Squeeze zero period', () => {
    
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([9.0, 19.0, 29.0]);
    const close = new Float64Array([9.5, 19.5, 29.5]);
    
    assert.throws(() => {
        wasm.ttm_squeeze(high, low, close, 0, 2.0, 1.0, 1.5, 2.0);
    }, /Invalid period|period cannot be zero/);
});

test('TTM Squeeze period exceeds length', () => {
    
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([9.0, 19.0, 29.0]);
    const close = new Float64Array([9.5, 19.5, 29.5]);
    
    assert.throws(() => {
        wasm.ttm_squeeze(high, low, close, 10, 2.0, 1.0, 1.5, 2.0);
    }, /Invalid period|Not enough data/);
});

test('TTM Squeeze very small dataset', () => {
    
    const high = new Float64Array([42.0]);
    const low = new Float64Array([41.0]);
    const close = new Float64Array([41.5]);
    
    assert.throws(() => {
        wasm.ttm_squeeze(high, low, close, 20, 2.0, 1.0, 1.5, 2.0);
    }, /Invalid period|Not enough valid data/);
});

test('TTM Squeeze empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.ttm_squeeze(empty, empty, empty, 20, 2.0, 1.0, 1.5, 2.0);
    }, /Input data slice is empty|empty/);
});

test('TTM Squeeze all NaN input', () => {
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.ttm_squeeze(allNaN, allNaN, allNaN, 20, 2.0, 1.0, 1.5, 2.0);
    }, /All values are NaN|Not enough valid data/);
});

test('TTM Squeeze inconsistent slices', () => {
    
    const high = new Float64Array([1, 2, 3, 4, 5]);
    const low = new Float64Array([1, 2, 3]);
    const close = new Float64Array([1, 2, 3, 4, 5]);
    
    assert.throws(() => {
        wasm.ttm_squeeze(high, low, close, 2, 2.0, 1.0, 1.5, 2.0);
    }, /Inconsistent slice lengths|mismatched/);
});

test('TTM Squeeze NaN handling', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const rawResult = wasm.ttm_squeeze(high, low, close, 20, 2.0, 1.0, 1.5, 2.0);
    const result = extractTtmSqueezeResult(rawResult);
    assert.strictEqual(result.momentum.length, close.length);
    assert.strictEqual(result.squeeze.length, close.length);
    
    
    if (result.momentum.length > 40) {
        for (let i = 40; i < result.momentum.length; i++) {
            assert(!isNaN(result.momentum[i]), `Found unexpected NaN in momentum at index ${i}`);
            assert(!isNaN(result.squeeze[i]), `Found unexpected NaN in squeeze at index ${i}`);
        }
    }
    
    
    for (let i = 0; i < TTM_EXPECTED.warmupPeriod; i++) {
        assert(isNaN(result.momentum[i]), `Expected NaN in momentum warmup at index ${i}`);
        assert(isNaN(result.squeeze[i]), `Expected NaN in squeeze warmup at index ${i}`);
    }
});

test('TTM Squeeze with custom parameters', () => {
    
    const n = 100;
    const high = new Float64Array(n);
    const low = new Float64Array(n);
    const close = new Float64Array(n);
    
    
    for (let i = 0; i < n; i++) {
        high[i] = 100 + Math.sin(i * 0.1) * 10 + Math.random() * 2;
        low[i] = high[i] - 2 - Math.random();
        close[i] = (high[i] + low[i]) / 2 + (Math.random() - 0.5) * 0.5;
    }
    
    
    const rawResult = wasm.ttm_squeeze(
        high, low, close,
        30,   
        2.5,  
        1.2,  
        1.8,  
        2.5   
    );
    
    assert.ok(rawResult, 'Should return a result');
    const result = extractTtmSqueezeResult(rawResult);
    assert.strictEqual(result.momentum.length, n);
    assert.strictEqual(result.squeeze.length, n);
    
    
    assert(isNaN(result.momentum[0]), 'First momentum value should be NaN');
    assert(isNaN(result.squeeze[0]), 'First squeeze value should be NaN');
    
    
    assert(!isNaN(result.momentum[n - 1]), 'Last momentum value should be valid');
    assert(!isNaN(result.squeeze[n - 1]), 'Last squeeze value should be valid');
    
    
    for (let i = 29; i < n; i++) {
        if (!isNaN(result.squeeze[i])) {
            assert(result.squeeze[i] >= 0 && result.squeeze[i] <= 3,
                `Squeeze value at ${i} should be between 0 and 3, got ${result.squeeze[i]}`);
        }
    }
});


test('TTM Squeeze batch single parameter set', () => {
    
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const batchResult = wasm.ttm_squeeze_batch(high, low, close, {
        length_range: [20, 20, 0],
        bb_mult_range: [2.0, 2.0, 0],
        kc_high_range: [1.0, 1.0, 0],
        kc_mid_range: [1.5, 1.5, 0],
        kc_low_range: [2.0, 2.0, 0]
    });
    
    assert(batchResult.values, 'Should have values array');
    assert(batchResult.combos, 'Should have combos array');
    assert.strictEqual(batchResult.combos.length, 1);
    
    assert.strictEqual(batchResult.rows, 2);  
    assert.strictEqual(batchResult.cols, 100);
    
    
    assert.strictEqual(batchResult.values.length, 2 * 100); 
    
    
    const singleRawResult = wasm.ttm_squeeze(high, low, close, 20, 2.0, 1.0, 1.5, 2.0);
    const singleResult = extractTtmSqueezeResult(singleRawResult);
    
    
    const batchMomentum = batchResult.values.slice(0, 100);
    const batchSqueeze = batchResult.values.slice(100, 200);
    
    for (let i = 0; i < 100; i++) {
        if (isNaN(singleResult.momentum[i]) && isNaN(batchMomentum[i])) continue;
        assertClose(batchMomentum[i], singleResult.momentum[i], 1e-8, `Momentum mismatch at ${i}`);
        
        if (isNaN(singleResult.squeeze[i]) && isNaN(batchSqueeze[i])) continue;
        assertClose(batchSqueeze[i], singleResult.squeeze[i], 1e-8, `Squeeze mismatch at ${i}`);
    }
});

test('TTM Squeeze batch multiple parameters', () => {
    
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const result = wasm.ttm_squeeze_batch(high, low, close, {
        length_range: [20, 22, 2],    
        bb_mult_range: [2.0, 2.5, 0.5], 
        kc_high_range: [1.0, 1.0, 0],
        kc_mid_range: [1.5, 1.5, 0],
        kc_low_range: [2.0, 2.0, 0]
    });
    
    
    assert.strictEqual(result.combos.length, 4);
    
    assert.strictEqual(result.rows, 8);  
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.values.length, 8 * 50); 
    
    
    const expectedCombos = [
        { length: 20, bb_mult: 2.0 },
        { length: 20, bb_mult: 2.5 },
        { length: 22, bb_mult: 2.0 },
        { length: 22, bb_mult: 2.5 }
    ];
    
    for (let i = 0; i < expectedCombos.length; i++) {
        assert.strictEqual(result.combos[i].length, expectedCombos[i].length);
        assertClose(result.combos[i].bb_mult, expectedCombos[i].bb_mult, 1e-8);
    }
});

test('TTM Squeeze batch metadata from result', () => {
    
    const high = new Float64Array(30);
    const low = new Float64Array(30); 
    const close = new Float64Array(30);
    
    
    for (let i = 0; i < 30; i++) {
        high[i] = 100 + i;
        low[i] = 99 + i;
        close[i] = 99.5 + i;
    }
    
    const result = wasm.ttm_squeeze_batch(high, low, close, {
        length_range: [20, 24, 2],      
        bb_mult_range: [2.0, 2.0, 0],   
        kc_high_range: [1.0, 1.2, 0.2], 
        kc_mid_range: [1.5, 1.5, 0],    
        kc_low_range: [2.0, 2.0, 0]     
    });
    
    
    assert.strictEqual(result.combos.length, 6);
    
    
    assert.strictEqual(result.combos[0].length, 20);
    assert.strictEqual(result.combos[0].bb_mult, 2.0);
    assert.strictEqual(result.combos[0].kc_mult_high, 1.0);
    
    
    assert.strictEqual(result.combos[5].length, 24);
    assertClose(result.combos[5].kc_mult_high, 1.2, 1e-8);
});

test('TTM Squeeze batch edge cases', () => {
    
    const high = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]);
    const low = high.map(v => v - 0.1);
    const close = high.map(v => v - 0.05);
    
    
    const singleBatch = wasm.ttm_squeeze_batch(high, low, close, {
        length_range: [20, 20, 1],
        bb_mult_range: [2.0, 2.0, 0.1],
        kc_high_range: [1.0, 1.0, 0],
        kc_mid_range: [1.5, 1.5, 0],
        kc_low_range: [2.0, 2.0, 0]
    });
    
    assert.strictEqual(singleBatch.combos.length, 1);
    assert.strictEqual(singleBatch.values.length, 2 * 21); 
    
    
    assert.throws(() => {
        wasm.ttm_squeeze_batch(new Float64Array([]), new Float64Array([]), new Float64Array([]), {
            length_range: [20, 20, 0],
            bb_mult_range: [2.0, 2.0, 0],
            kc_high_range: [1.0, 1.0, 0],
            kc_mid_range: [1.5, 1.5, 0],
            kc_low_range: [2.0, 2.0, 0]
        });
    }, /empty|All values are NaN|insufficient/i);
});


test('TTM Squeeze zero-copy API', () => {
    const high = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]);
    const low = new Float64Array(high.length);
    const close = new Float64Array(high.length);
    
    for (let i = 0; i < high.length; i++) {
        low[i] = high[i] - 0.2;
        close[i] = high[i] - 0.1;
    }
    
    const length = 20;
    const bb_mult = 2.0;
    const kc_mult_high = 1.0;
    const kc_mult_mid = 1.5;
    const kc_mult_low = 2.0;
    
    
    const ptrMom = wasm.ttm_squeeze_alloc(high.length);
    const ptrSqz = wasm.ttm_squeeze_alloc(high.length);
    assert(ptrMom !== 0, 'Failed to allocate momentum buffer');
    assert(ptrSqz !== 0, 'Failed to allocate squeeze buffer');
    
    try {
        
        const memory = wasm.__wasm.memory.buffer;
        
        
        const ptrHigh = wasm.ttm_squeeze_alloc(high.length);
        const ptrLow = wasm.ttm_squeeze_alloc(low.length);
        const ptrClose = wasm.ttm_squeeze_alloc(close.length);
        
        const memHigh = new Float64Array(memory, ptrHigh, high.length);
        const memLow = new Float64Array(memory, ptrLow, low.length);
        const memClose = new Float64Array(memory, ptrClose, close.length);
        
        memHigh.set(high);
        memLow.set(low);
        memClose.set(close);
        
        
        wasm.ttm_squeeze_into_ptrs(
            ptrHigh, ptrLow, ptrClose,
            ptrMom, ptrSqz,
            high.length,
            length, bb_mult,
            kc_mult_high, kc_mult_mid, kc_mult_low
        );
        
        
        const memory2 = wasm.__wasm.memory.buffer;
        const memMom = new Float64Array(memory2, ptrMom, high.length);
        const memSqz = new Float64Array(memory2, ptrSqz, high.length);
        
        
        const regularRawResult = wasm.ttm_squeeze(
            high, low, close,
            length, bb_mult,
            kc_mult_high, kc_mult_mid, kc_mult_low
        );
        const regularResult = extractTtmSqueezeResult(regularRawResult);
        
        for (let i = 0; i < high.length; i++) {
            if (isNaN(regularResult.momentum[i]) && isNaN(memMom[i])) continue;
            assertClose(memMom[i], regularResult.momentum[i], 1e-8,
                `Zero-copy momentum mismatch at index ${i}`);
            
            if (isNaN(regularResult.squeeze[i]) && isNaN(memSqz[i])) continue;
            assertClose(memSqz[i], regularResult.squeeze[i], 1e-8,
                `Zero-copy squeeze mismatch at index ${i}`);
        }
        
        
        wasm.ttm_squeeze_free(ptrHigh, high.length);
        wasm.ttm_squeeze_free(ptrLow, low.length);
        wasm.ttm_squeeze_free(ptrClose, close.length);
    } finally {
        
        wasm.ttm_squeeze_free(ptrMom, high.length);
        wasm.ttm_squeeze_free(ptrSqz, high.length);
    }
});

test('TTM Squeeze zero-copy with large dataset', () => {
    const size = 10000;
    const high = new Float64Array(size);
    const low = new Float64Array(size);
    const close = new Float64Array(size);
    
    for (let i = 0; i < size; i++) {
        high[i] = 100 + Math.sin(i * 0.01) + Math.random() * 0.1;
        low[i] = high[i] - 1 - Math.random() * 0.1;
        close[i] = (high[i] + low[i]) / 2;
    }
    
    const ptrMom = wasm.ttm_squeeze_alloc(size);
    const ptrSqz = wasm.ttm_squeeze_alloc(size);
    assert(ptrMom !== 0, 'Failed to allocate large momentum buffer');
    assert(ptrSqz !== 0, 'Failed to allocate large squeeze buffer');
    
    try {
        
        const ptrHigh = wasm.ttm_squeeze_alloc(size);
        const ptrLow = wasm.ttm_squeeze_alloc(size);
        const ptrClose = wasm.ttm_squeeze_alloc(size);
        
        const memory = wasm.__wasm.memory.buffer;
        const memHigh = new Float64Array(memory, ptrHigh, size);
        const memLow = new Float64Array(memory, ptrLow, size);
        const memClose = new Float64Array(memory, ptrClose, size);
        
        memHigh.set(high);
        memLow.set(low);
        memClose.set(close);
        
        wasm.ttm_squeeze_into_ptrs(
            ptrHigh, ptrLow, ptrClose,
            ptrMom, ptrSqz,
            size,
            20, 2.0, 1.0, 1.5, 2.0
        );
        
        
        const memory2 = wasm.__wasm.memory.buffer;
        const memMom = new Float64Array(memory2, ptrMom, size);
        const memSqz = new Float64Array(memory2, ptrSqz, size);
        
        
        for (let i = 0; i < 19; i++) {
            assert(isNaN(memMom[i]), `Expected NaN in momentum at warmup index ${i}`);
            assert(isNaN(memSqz[i]), `Expected NaN in squeeze at warmup index ${i}`);
        }
        
        
        for (let i = 19; i < Math.min(100, size); i++) {
            assert(!isNaN(memMom[i]), `Unexpected NaN in momentum at index ${i}`);
            assert(!isNaN(memSqz[i]), `Unexpected NaN in squeeze at index ${i}`);
        }
        
        
        wasm.ttm_squeeze_free(ptrHigh, size);
        wasm.ttm_squeeze_free(ptrLow, size);
        wasm.ttm_squeeze_free(ptrClose, size);
    } finally {
        wasm.ttm_squeeze_free(ptrMom, size);
        wasm.ttm_squeeze_free(ptrSqz, size);
    }
});

test('TTM Squeeze zero-copy error handling', () => {
    
    assert.throws(() => {
        wasm.ttm_squeeze_into_ptrs(0, 0, 0, 0, 0, 10, 20, 2.0, 1.0, 1.5, 2.0);
    }, /empty|null pointer|invalid memory/i);
    
    
    const ptr = wasm.ttm_squeeze_alloc(10);
    try {
        
        assert.throws(() => {
            wasm.ttm_squeeze_into_ptrs(ptr, ptr, ptr, ptr, ptr, 10, 0, 2.0, 1.0, 1.5, 2.0);
        }, /Invalid period/);
        
        
        assert.throws(() => {
            wasm.ttm_squeeze_into_ptrs(ptr, ptr, ptr, ptr, ptr, 10, 5, 0.0, 1.0, 1.5, 2.0);
        }, /Invalid.*mult/);
    } finally {
        wasm.ttm_squeeze_free(ptr, 10);
    }
});

test('TTM Squeeze memory leak prevention', () => {
    
    const sizes = [100, 1000, 10000];
    
    for (const size of sizes) {
        const ptrMom = wasm.ttm_squeeze_alloc(size);
        const ptrSqz = wasm.ttm_squeeze_alloc(size);
        assert(ptrMom !== 0, `Failed to allocate ${size} momentum elements`);
        assert(ptrSqz !== 0, `Failed to allocate ${size} squeeze elements`);
        
        
        const memory = wasm.__wasm.memory.buffer;
        const memMom = new Float64Array(memory, ptrMom, size);
        const memSqz = new Float64Array(memory, ptrSqz, size);
        
        for (let i = 0; i < Math.min(10, size); i++) {
            memMom[i] = i * 1.5;
            memSqz[i] = i * 2.5;
        }
        
        
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memMom[i], i * 1.5, `Momentum memory corruption at index ${i}`);
            assert.strictEqual(memSqz[i], i * 2.5, `Squeeze memory corruption at index ${i}`);
        }
        
        
        wasm.ttm_squeeze_free(ptrMom, size);
        wasm.ttm_squeeze_free(ptrSqz, size);
    }
});


test('TTM Squeeze SIMD128 consistency', () => {
    
    const testCases = [
        { size: 21, length: 20 },
        { size: 100, length: 20 },
        { size: 1000, length: 50 }
    ];
    
    for (const testCase of testCases) {
        const high = new Float64Array(testCase.size);
        const low = new Float64Array(testCase.size);
        const close = new Float64Array(testCase.size);
        
        for (let i = 0; i < testCase.size; i++) {
            high[i] = 100 + Math.sin(i * 0.1) + Math.cos(i * 0.05);
            low[i] = high[i] - 1 - Math.abs(Math.sin(i * 0.2));
            close[i] = (high[i] + low[i]) / 2;
        }
        
        const rawResult = wasm.ttm_squeeze(
            high, low, close,
            testCase.length, 2.0, 1.0, 1.5, 2.0
        );
        const result = extractTtmSqueezeResult(rawResult);
        
        
        assert.strictEqual(result.momentum.length, testCase.size);
        assert.strictEqual(result.squeeze.length, testCase.size);
        
        
        for (let i = 0; i < testCase.length - 1; i++) {
            assert(isNaN(result.momentum[i]), `Expected NaN in momentum at warmup index ${i} for size=${testCase.size}`);
            assert(isNaN(result.squeeze[i]), `Expected NaN in squeeze at warmup index ${i} for size=${testCase.size}`);
        }
        
        
        let sumMomentum = 0;
        let countMomentum = 0;
        for (let i = testCase.length - 1; i < result.momentum.length; i++) {
            assert(!isNaN(result.momentum[i]), `Unexpected NaN in momentum at index ${i} for size=${testCase.size}`);
            assert(!isNaN(result.squeeze[i]), `Unexpected NaN in squeeze at index ${i} for size=${testCase.size}`);
            
            
            assert(result.squeeze[i] >= 0 && result.squeeze[i] <= 3,
                `Squeeze value ${result.squeeze[i]} out of range at index ${i}`);
            
            sumMomentum += result.momentum[i];
            countMomentum++;
        }
        
        
        const avgMomentum = sumMomentum / countMomentum;
        assert(Math.abs(avgMomentum) < 1000, `Average momentum ${avgMomentum} seems unreasonable`);
    }
});

test.after(() => {
    console.log('TTM Squeeze WASM tests completed');
});
