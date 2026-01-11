/**
 * WASM binding tests for AVSL (Anti-Volume Stop Loss) indicator.
 * These tests mirror the Rust unit tests to ensure WASM bindings work correctly.
 */
import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';
import { parse } from 'csv-parse/sync';

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
        console.error('Failed to load WASM module:', error);
        throw error;
    }
    
    
    const csvPath = path.join(__dirname, '../../src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv');
    const csvContent = fs.readFileSync(csvPath, 'utf-8');
    const records = parse(csvContent, {
        columns: false,  
        skip_empty_lines: true,
        from: 2  
    });
    
    
    testData = {
        close: new Float64Array(records.map(r => parseFloat(r[2]))),  
        low: new Float64Array(records.map(r => parseFloat(r[4]))),    
        volume: new Float64Array(records.map(r => parseFloat(r[5])))  
    };
});

test('AVSL accuracy', async () => {
    const { close, low, volume } = testData;
    
    
    const expectedLastFive = [
        56471.61721191,
        56267.11946706,
        56079.12004921,
        55910.07971214,
        55765.37864229,
    ];
    
    
    const result = wasm.avsl_js(
        close,
        low,
        volume,
        12,  
        26,  
        2.0  
    );
    
    assert.strictEqual(result.length, close.length, 'Result length should match input length');
    
    
    const last5 = result.slice(-5);
    for (let i = 0; i < 5; i++) {
        const actual = last5[i];
        const expected = expectedLastFive[i];
        const tolerance = Math.abs(expected) * 0.01; 
        const diff = Math.abs(actual - expected);
        
        assert.ok(
            diff < tolerance,
            `AVSL value mismatch at index ${i}: got ${actual}, expected ${expected}, diff ${diff}`
        );
    }
});

test('AVSL empty input', () => {
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.avsl_js(empty, empty, empty, 12, 26, 2.0);
    }, /empty/i);
});

test('AVSL mismatched lengths', () => {
    const close = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([9.0, 19.0]); 
    const volume = new Float64Array([100.0, 200.0, 300.0]);
    
    assert.throws(() => {
        wasm.avsl_js(close, low, volume, 12, 26, 2.0);
    }, /mismatch/i);
});

test('AVSL invalid period', () => {
    const data = new Float64Array([10.0, 20.0, 30.0]);
    const volume = new Float64Array([100.0, 200.0, 300.0]);
    
    
    assert.throws(() => {
        wasm.avsl_js(data, data, volume, 0, 26, 2.0);
    }, /Invalid period/);
    
    
    assert.throws(() => {
        wasm.avsl_js(data, data, volume, 12, 100, 2.0);
    }, /Invalid period/);
});

test('AVSL all NaN', () => {
    const nanData = new Float64Array([NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.avsl_js(nanData, nanData, nanData, 12, 26, 2.0);
    }, /NaN/i);
});

test('AVSL different parameters', async () => {
    const { close, low, volume } = testData;
    
    
    const result1 = wasm.avsl_js(close, low, volume, 10, 20, 2.0);
    assert.strictEqual(result1.length, close.length);
    
    
    const result2 = wasm.avsl_js(close, low, volume, 12, 26, 1.5);
    assert.strictEqual(result2.length, close.length);
    
    
    const last10_1 = result1.slice(-10);
    const last10_2 = result2.slice(-10);
    
    let isDifferent = false;
    for (let i = 0; i < 10; i++) {
        if (Math.abs(last10_1[i] - last10_2[i]) > 0.0001) {
            isDifferent = true;
            break;
        }
    }
    assert.ok(isDifferent, 'Results should differ with different parameters');
});

test('AVSL invalid multiplier', () => {
    const data = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0]);
    const volume = new Float64Array([100.0, 200.0, 300.0, 400.0, 500.0]);
    
    
    assert.throws(() => {
        wasm.avsl_js(data, data, volume, 2, 3, -1.0);
    }, /Invalid multiplier/);
    
    
    assert.throws(() => {
        wasm.avsl_js(data, data, volume, 2, 3, 0.0);
    }, /Invalid multiplier/);
    
    
    assert.throws(() => {
        wasm.avsl_js(data, data, volume, 2, 3, NaN);
    }, /Invalid multiplier/);
    
    
    assert.throws(() => {
        wasm.avsl_js(data, data, volume, 2, 3, Infinity);
    }, /Invalid multiplier/);
});

test('AVSL warmup period verification', async () => {
    const { close, low, volume } = testData;
    const fastPeriod = 12;
    const slowPeriod = 26;
    
    const result = wasm.avsl_js(close, low, volume, fastPeriod, slowPeriod, 2.0);
    
    
    let firstValid = -1;
    for (let i = 0; i < result.length; i++) {
        if (!isNaN(result[i])) {
            firstValid = i;
            break;
        }
    }
    
    
    const expectedWarmup = slowPeriod - 1;
    assert.ok(
        firstValid >= expectedWarmup,
        `First valid value at index ${firstValid}, expected >= ${expectedWarmup}`
    );
    
    
    for (let i = 0; i < expectedWarmup && i < result.length; i++) {
        assert.ok(isNaN(result[i]), `Expected NaN at warmup index ${i}`);
    }
    
    
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert.ok(!isNaN(result[i]), `Unexpected NaN at index ${i} after warmup`);
        }
    }
});

test('AVSL batch processing', async () => {
    const { close, low, volume } = testData;
    
    
    const batchResult = wasm.avsl_batch(close, low, volume, {
        fast_range: [12, 12, 0],
        slow_range: [26, 26, 0],
        mult_range: [2.0, 2.0, 0.0]
    });
    
    assert.ok(batchResult.values, 'Should have values array');
    assert.ok(batchResult.combos, 'Should have combos array');
    assert.strictEqual(typeof batchResult.rows, 'number', 'Should have rows count');
    assert.strictEqual(typeof batchResult.cols, 'number', 'Should have cols count');
    
    
    assert.strictEqual(batchResult.rows, 1);
    assert.strictEqual(batchResult.cols, close.length);
    assert.strictEqual(batchResult.combos.length, 1);
    assert.strictEqual(batchResult.values.length, close.length);
    
    
    const defaultRow = batchResult.values;
    
    
    const expectedLastFive = [
        56471.61721191,
        56267.11946706,
        56079.12004921,
        55910.07971214,
        55765.37864229,
    ];
    
    
    const last5 = defaultRow.slice(-5);
    for (let i = 0; i < 5; i++) {
        const actual = last5[i];
        const expected = expectedLastFive[i];
        const tolerance = Math.abs(expected) * 0.01;
        const diff = Math.abs(actual - expected);
        
        assert.ok(
            diff < tolerance,
            `Batch default row mismatch at index ${i}: got ${actual}, expected ${expected}, diff ${diff}`
        );
    }
});

test('AVSL batch multiple parameters', () => {
    const close = new Float64Array(testData.close.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const volume = new Float64Array(testData.volume.slice(0, 100));
    
    const result = wasm.avsl_batch(close, low, volume, {
        fast_range: [10, 15, 5],  
        slow_range: [20, 30, 10], 
        mult_range: [1.5, 2.5, 0.5] 
    });
    
    
    assert.strictEqual(result.rows, 12);
    assert.strictEqual(result.cols, 100);
    assert.strictEqual(result.combos.length, 12);
    assert.strictEqual(result.values.length, 12 * 100);
    
    
    assert.strictEqual(result.combos[0].fast_period, 10);
    assert.strictEqual(result.combos[0].slow_period, 20);
    assert.strictEqual(result.combos[0].multiplier, 1.5);
    
    
    assert.strictEqual(result.combos[11].fast_period, 15);
    assert.strictEqual(result.combos[11].slow_period, 30);
    assert.strictEqual(result.combos[11].multiplier, 2.5);
});

test('AVSL context API', () => {
    const fastPeriod = 12;
    const slowPeriod = 26;
    const multiplier = 2.0;
    
    
    const context = new wasm.AvslContext(fastPeriod, slowPeriod, multiplier);
    
    
    assert.strictEqual(context.get_warmup_period(), slowPeriod - 1);
    
    
    assert.throws(() => {
        new wasm.AvslContext(0, slowPeriod, multiplier);
    }, /Invalid fast period/);
    
    assert.throws(() => {
        new wasm.AvslContext(fastPeriod, 0, multiplier);
    }, /Invalid slow period/);
    
    assert.throws(() => {
        new wasm.AvslContext(fastPeriod, slowPeriod, -1.0);
    }, /Invalid multiplier/);
});

test('AVSL zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const volume = new Float64Array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
    const fastPeriod = 3;
    const slowPeriod = 5;
    const multiplier = 2.0;
    
    
    const closePtr = wasm.avsl_alloc(data.length);
    const lowPtr = wasm.avsl_alloc(data.length);
    const volPtr = wasm.avsl_alloc(data.length);
    const outPtr = wasm.avsl_alloc(data.length);
    
    assert.ok(closePtr !== 0, 'Failed to allocate close buffer');
    assert.ok(lowPtr !== 0, 'Failed to allocate low buffer');
    assert.ok(volPtr !== 0, 'Failed to allocate volume buffer');
    assert.ok(outPtr !== 0, 'Failed to allocate output buffer');
    
    try {
        
        const closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, data.length);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, data.length);
        const volView = new Float64Array(wasm.__wasm.memory.buffer, volPtr, data.length);
        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, data.length);
        
        
        closeView.set(data);
        lowView.set(data); 
        volView.set(volume);
        
        
        wasm.avsl_into(closePtr, lowPtr, volPtr, outPtr, data.length, fastPeriod, slowPeriod, multiplier);
        
        
        const regularResult = wasm.avsl_js(data, data, volume, fastPeriod, slowPeriod, multiplier);
        
        
        const outView2 = new Float64Array(wasm.__wasm.memory.buffer, outPtr, data.length);
        
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(outView2[i])) {
                continue; 
            }
            assert.ok(
                Math.abs(regularResult[i] - outView2[i]) < 1e-10,
                `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${outView2[i]}`
            );
        }
    } finally {
        
        wasm.avsl_free(closePtr, data.length);
        wasm.avsl_free(lowPtr, data.length);
        wasm.avsl_free(volPtr, data.length);
        wasm.avsl_free(outPtr, data.length);
    }
});

test('AVSL zero-copy in-place operation', () => {
    const size = 100;
    const close = new Float64Array(size);
    const low = new Float64Array(size);
    const volume = new Float64Array(size);
    
    
    for (let i = 0; i < size; i++) {
        close[i] = Math.sin(i * 0.1) + 100;
        low[i] = close[i] - Math.random() * 2;
        volume[i] = 1000 + Math.random() * 500;
    }
    
    const closePtr = wasm.avsl_alloc(size);
    const lowPtr = wasm.avsl_alloc(size);
    const volPtr = wasm.avsl_alloc(size);
    
    try {
        const closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, size);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, size);
        const volView = new Float64Array(wasm.__wasm.memory.buffer, volPtr, size);
        
        closeView.set(close);
        lowView.set(low);
        volView.set(volume);
        
        
        wasm.avsl_into(closePtr, lowPtr, volPtr, closePtr, size, 12, 26, 2.0);
        
        
        const resultView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, size);
        
        
        
        
        
        
        const warmupEnd = 50; 
        for (let i = 0; i < warmupEnd && i < size; i++) {
            assert.ok(isNaN(resultView[i]), `Expected NaN at warmup index ${i}`);
        }
        
        
        for (let i = warmupEnd; i < Math.min(warmupEnd + 10, size); i++) {
            assert.ok(!isNaN(resultView[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.avsl_free(closePtr, size);
        wasm.avsl_free(lowPtr, size);
        wasm.avsl_free(volPtr, size);
    }
});

test('AVSL SIMD128 consistency', () => {
    
    const testCases = [
        { size: 10, fast: 3, slow: 5 },
        { size: 100, fast: 12, slow: 26 },
        { size: 1000, fast: 20, slow: 50 }
    ];
    
    for (const testCase of testCases) {
        const close = new Float64Array(testCase.size);
        const low = new Float64Array(testCase.size);
        const volume = new Float64Array(testCase.size);
        
        for (let i = 0; i < testCase.size; i++) {
            close[i] = Math.sin(i * 0.1) + Math.cos(i * 0.05) + 100;
            low[i] = close[i] - Math.abs(Math.sin(i * 0.2));
            volume[i] = 1000 + Math.sin(i * 0.3) * 500;
        }
        
        const result = wasm.avsl_js(close, low, volume, testCase.fast, testCase.slow, 2.0);
        
        
        assert.strictEqual(result.length, close.length);
        
        
        const expectedWarmup = testCase.slow - 1;
        for (let i = 0; i < expectedWarmup && i < result.length; i++) {
            assert.ok(isNaN(result[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}`);
        }
        
        
        let hasValidValues = false;
        for (let i = testCase.slow; i < result.length; i++) {
            if (!isNaN(result[i])) {
                hasValidValues = true;
                break;
            }
        }
        assert.ok(hasValidValues, `Should have valid values after warmup for size=${testCase.size}`);
    }
});

test('AVSL memory allocation/deallocation', () => {
    const len = 1000;
    
    
    const ptr = wasm.avsl_alloc(len);
    assert.ok(ptr !== 0, 'Should allocate non-zero pointer');
    
    
    wasm.avsl_free(ptr, len);
    
    
    const ptrs = [];
    for (let i = 0; i < 5; i++) {
        ptrs.push(wasm.avsl_alloc(100));
    }
    
    
    for (const p of ptrs) {
        wasm.avsl_free(p, 100);
    }
});

test('AVSL batch error handling', () => {
    const close = new Float64Array(10);
    const low = new Float64Array(10);
    const volume = new Float64Array(10);
    close.fill(100);
    low.fill(99);
    volume.fill(1000);
    
    
    assert.throws(() => {
        wasm.avsl_batch(close, low, volume, {
            fast_range: [12, 12], 
            slow_range: [26, 26, 0],
            mult_range: [2.0, 2.0, 0]
        });
    }, /Invalid config/);
    
    
    assert.throws(() => {
        wasm.avsl_batch(close, low, volume, {
            fast_range: [12, 12, 0],
            slow_range: [26, 26, 0]
            
        });
    }, /Invalid config/);
    
    
    assert.throws(() => {
        wasm.avsl_batch(close, low, volume, {
            fast_range: "invalid",
            slow_range: [26, 26, 0],
            mult_range: [2.0, 2.0, 0]
        });
    }, /Invalid config/);
});

test('AVSL very small dataset', () => {
    const smallData = new Float64Array([10.0, 11.0, 12.0, 13.0, 14.0]);
    const smallVolume = new Float64Array([100.0, 110.0, 120.0, 130.0, 140.0]);
    
    
    assert.throws(() => {
        wasm.avsl_js(smallData, smallData, smallVolume, 12, 26, 2.0);
    }, /Invalid period|Not enough valid data/);
    
    
    const result = wasm.avsl_js(smallData, smallData, smallVolume, 2, 3, 2.0);
    assert.strictEqual(result.length, smallData.length);
});
