/**
 * WASM binding tests for CoRa Wave indicator.
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

test('CoRa Wave partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.cora_wave_js(close, 20, 2.0, true);
    assert.strictEqual(result.length, close.length);
});

test('CoRa Wave accuracy', () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.coraWave;
    
    const result = wasm.cora_wave_js(
        close,
        expected.defaultParams.period,
        expected.defaultParams.r_multi,
        expected.defaultParams.smooth
    );
    
    assert.strictEqual(result.length, close.length);
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-8,  
        "CoRa Wave last 5 values mismatch"
    );
});

test('CoRa Wave default candles', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.cora_wave_js(close, 20, 2.0, true);
    assert.strictEqual(result.length, close.length);
});

test('CoRa Wave zero period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.cora_wave_js(inputData, 0, 2.0, true);
    }, /Invalid period/);
});

test('CoRa Wave period exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.cora_wave_js(dataSmall, 10, 2.0, true);
    }, /Invalid period/);
});

test('CoRa Wave very small dataset', () => {
    
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.cora_wave_js(singlePoint, 20, 2.0, true);
    }, /Invalid period|Not enough valid data/);
});

test('CoRa Wave empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.cora_wave_js(empty, 20, 2.0, true);
    }, /empty/i);
});

test('CoRa Wave invalid r_multi', () => {
    
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    
    assert.throws(() => {
        wasm.cora_wave_js(data, 2, NaN, false);
    }, /Invalid r_multi/);
    
    
    assert.throws(() => {
        wasm.cora_wave_js(data, 2, -1.0, false);
    }, /Invalid r_multi/);
    
    
    const resultZero = wasm.cora_wave_js(data, 2, 0.0, false);
    assert.strictEqual(resultZero.length, data.length);
});

test('CoRa Wave NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.coraWave;
    
    const result = wasm.cora_wave_js(
        close,
        expected.defaultParams.period,
        expected.defaultParams.r_multi,
        expected.defaultParams.smooth
    );
    
    assert.strictEqual(result.length, close.length);
    
    
    const warmup = expected.warmupPeriod;  
    if (result.length > warmup + 100) {
        for (let i = warmup + 100; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    
    assertAllNaN(result.slice(0, warmup), "Expected NaN in warmup period");
});

test('CoRa Wave all NaN input', () => {
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.cora_wave_js(allNaN, 20, 2.0, true);
    }, /All values are NaN/);
});

test('CoRa Wave without smoothing', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    
    const resultSmooth = wasm.cora_wave_js(close, 20, 2.0, true);
    const resultRaw = wasm.cora_wave_js(close, 20, 2.0, false);
    
    assert.strictEqual(resultSmooth.length, close.length);
    assert.strictEqual(resultRaw.length, close.length);
    
    
    
    let foundDifference = false;
    for (let i = 30; i < resultSmooth.length; i++) {
        if (!isNaN(resultSmooth[i]) && !isNaN(resultRaw[i])) {
            if (Math.abs(resultSmooth[i] - resultRaw[i]) > 1e-10) {
                foundDifference = true;
                break;
            }
        }
    }
    assert(foundDifference, 'Smoothed and raw values should be different');
});

test('CoRa Wave with different r_multi values', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    
    const result1 = wasm.cora_wave_js(close, 20, 1.0, false);
    const result2 = wasm.cora_wave_js(close, 20, 2.0, false);
    const result3 = wasm.cora_wave_js(close, 20, 3.0, false);
    
    
    let foundDifference12 = false;
    let foundDifference23 = false;
    for (let i = 30; i < result1.length; i++) {
        if (!isNaN(result1[i]) && !isNaN(result2[i]) && !isNaN(result3[i])) {
            if (Math.abs(result1[i] - result2[i]) > 1e-10) {
                foundDifference12 = true;
            }
            if (Math.abs(result2[i] - result3[i]) > 1e-10) {
                foundDifference23 = true;
            }
            if (foundDifference12 && foundDifference23) {
                break;
            }
        }
    }
    assert(foundDifference12, 'Different r_multi values (1.0 vs 2.0) should produce different results');
    assert(foundDifference23, 'Different r_multi values (2.0 vs 3.0) should produce different results');
});

test('CoRa Wave batch single parameter set', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    
    
    const batchResult = wasm.cora_wave_batch(close, {
        period_range: [20, 20, 0],
        r_multi_range: [2.0, 2.0, 0],
        smooth: false  
    });
    
    
    const singleResult = wasm.cora_wave_js(close, 20, 2.0, false);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('CoRa Wave batch multiple periods', () => {
    
    const close = new Float64Array(testData.close.slice(0, 50)); 
    
    
    const batchResult = wasm.cora_wave_batch(close, {
        period_range: [15, 20, 5],      
        r_multi_range: [2.0, 2.0, 0],   
        smooth: false                    
    });
    
    
    assert.strictEqual(batchResult.values.length, 2 * 50);
    assert.strictEqual(batchResult.rows, 2);
    assert.strictEqual(batchResult.cols, 50);
    
    
    const periods = [15, 20];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 50;
        const rowEnd = rowStart + 50;
        const rowData = batchResult.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.cora_wave_js(close, periods[i], 2.0, false);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('CoRa Wave batch metadata from result', () => {
    
    const close = new Float64Array(25); 
    close.fill(100);
    
    const result = wasm.cora_wave_batch(close, {
        period_range: [15, 20, 5],      
        r_multi_range: [1.5, 2.0, 0.5], 
        smooth: false                    
    });
    
    
    assert.strictEqual(result.combos.length, 4);
    
    
    assert.strictEqual(result.combos[0].period, 15);
    assert.strictEqual(result.combos[0].r_multi, 1.5);
    assert.strictEqual(result.combos[0].smooth, false);
    
    
    assert.strictEqual(result.combos[3].period, 20);
    assert.strictEqual(result.combos[3].r_multi, 2.0);
    assert.strictEqual(result.combos[3].smooth, false);
});

test('CoRa Wave batch full parameter sweep', () => {
    
    const close = new Float64Array(testData.close.slice(0, 30));
    
    const batchResult = wasm.cora_wave_batch(close, {
        period_range: [10, 15, 5],      
        r_multi_range: [1.0, 1.5, 0.5], 
        smooth: false                    
    });
    
    
    assert.strictEqual(batchResult.combos.length, 4);
    assert.strictEqual(batchResult.rows, 4);
    assert.strictEqual(batchResult.cols, 30);
    assert.strictEqual(batchResult.values.length, 4 * 30);
    
    
    for (let combo = 0; combo < batchResult.combos.length; combo++) {
        const period = batchResult.combos[combo].period;
        const r_multi = batchResult.combos[combo].r_multi;
        const smooth = batchResult.combos[combo].smooth;
        
        const rowStart = combo * 30;
        const rowData = batchResult.values.slice(rowStart, rowStart + 30);
        
        
        for (let i = 0; i < period - 1; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${period}`);
        }
        
        
        for (let i = period - 1; i < 30; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period}`);
        }
    }
});

test('CoRa Wave batch edge cases', () => {
    
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    
    const singleBatch = wasm.cora_wave_batch(close, {
        period_range: [5, 5, 1],
        r_multi_range: [2.0, 2.0, 0.1],
        smooth: false  
    });
    
    assert.strictEqual(singleBatch.values.length, 10);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    
    const largeBatch = wasm.cora_wave_batch(close, {
        period_range: [5, 7, 10], 
        r_multi_range: [2.0, 2.0, 0],
        smooth: false  
    });
    
    
    assert.strictEqual(largeBatch.values.length, 10);
    assert.strictEqual(largeBatch.combos.length, 1);
    
    
    assert.throws(() => {
        wasm.cora_wave_batch(new Float64Array([]), {
            period_range: [20, 20, 0],
            r_multi_range: [2.0, 2.0, 0],
            smooth: true  
        });
    }, /All values are NaN/);
});

test('CoRa Wave batch - API error handling', () => {
    const close = new Float64Array(testData.close.slice(0, 25));
    
    
    assert.throws(() => {
        wasm.cora_wave_batch(close, {
            period_range: [20, 20], 
            r_multi_range: [2.0, 2.0, 0],
            smooth: true
        });
    }, /Invalid config/);
    
    
    assert.throws(() => {
        wasm.cora_wave_batch(close, {
            period_range: [20, 20, 0],
            r_multi_range: [2.0, 2.0, 0]
            
        });
    }, /Invalid config/);
    
    
    assert.throws(() => {
        wasm.cora_wave_batch(close, {
            period_range: "invalid",
            r_multi_range: [2.0, 2.0, 0],
            smooth: true
        });
    }, /Invalid config/);
});


test('CoRa Wave zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const period = 5;
    const r_multi = 2.0;
    const smooth = false;
    
    
    const ptr = wasm.cora_wave_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    
    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        data.length
    );
    
    
    memView.set(data);
    
    
    try {
        wasm.cora_wave_into(ptr, ptr, data.length, period, r_multi, smooth);
        
        
        const regularResult = wasm.cora_wave_js(data, period, r_multi, smooth);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; 
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        
        wasm.cora_wave_free(ptr, data.length);
    }
});

test('CoRa Wave zero-copy with large dataset', () => {
    const size = 10000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }
    
    const ptr = wasm.cora_wave_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');
    
    try {
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        memView.set(data);
        
        wasm.cora_wave_into(ptr, ptr, size, 20, 2.0, true);
        
        
        const memView2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        
        
        
        for (let i = 0; i < 22; i++) {
            assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
        }
        
        
        for (let i = 22; i < Math.min(100, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.cora_wave_free(ptr, size);
    }
});


test('CoRa Wave zero-copy error handling', () => {
    
    assert.throws(() => {
        wasm.cora_wave_into(0, 0, 10, 20, 2.0, true);
    }, /null pointer|invalid memory/i);
    
    
    const ptr = wasm.cora_wave_alloc(10);
    try {
        
        assert.throws(() => {
            wasm.cora_wave_into(ptr, ptr, 10, 0, 2.0, true);
        }, /Invalid period/);
        
        
        assert.throws(() => {
            wasm.cora_wave_into(ptr, ptr, 10, 5, -1.0, false);
        }, /Invalid r_multi/);
    } finally {
        wasm.cora_wave_free(ptr, 10);
    }
});


test('CoRa Wave zero-copy memory management', () => {
    
    const sizes = [100, 1000, 10000];
    
    for (const size of sizes) {
        const ptr = wasm.cora_wave_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }
        
        
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
        }
        
        
        wasm.cora_wave_free(ptr, size);
    }
});


test('CoRa Wave SIMD128 consistency', () => {
    
    
    const testCases = [
        { size: 10, period: 5 },
        { size: 100, period: 10 },
        { size: 1000, period: 20 },
        { size: 5000, period: 30 }
    ];
    
    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = Math.sin(i * 0.1) + Math.cos(i * 0.05);
        }
        
        const result = wasm.cora_wave_js(data, testCase.period, 2.0, true);
        
        
        assert.strictEqual(result.length, data.length);
        
        
        
        const smoothPeriod = Math.max(1, Math.round(Math.sqrt(testCase.period)));
        const warmup = testCase.period - 1 + Math.max(0, smoothPeriod - 1);
        for (let i = 0; i < Math.min(warmup, result.length); i++) {
            assert(isNaN(result[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}, period=${testCase.period}`);
        }
        
        
        if (result.length > warmup + 10) {
            let sumAfterWarmup = 0;
            let countAfterWarmup = 0;
            for (let i = warmup; i < Math.min(warmup + 100, result.length); i++) {
                assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for size=${testCase.size}`);
                sumAfterWarmup += result[i];
                countAfterWarmup++;
            }
            
            
            const avgAfterWarmup = sumAfterWarmup / countAfterWarmup;
            assert(Math.abs(avgAfterWarmup) < 100, `Average value ${avgAfterWarmup} seems unreasonable`);
        }
    }
});


test('CoRa Wave with leading NaN values', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    
    for (let i = 0; i < 5; i++) {
        close[i] = NaN;
    }
    
    const result = wasm.cora_wave_js(close, 20, 2.0, true);
    assert.strictEqual(result.length, close.length);
    
    
    const expectedNanCount = 5 + 22;
    for (let i = 0; i < expectedNanCount; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
    
    
    if (result.length > expectedNanCount + 10) {
        for (let i = expectedNanCount + 10; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i}`);
        }
    }
});

test('CoRa Wave batch with expected parameters from test_utils', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    const expected = EXPECTED_OUTPUTS.coraWave;
    
    const batchResult = wasm.cora_wave_batch(close, {
        period_range: expected.batchRange.periodRange,
        r_multi_range: expected.batchRange.rMultiRange,
        smooth: expected.batchRange.smooth
    });
    
    
    assert.strictEqual(batchResult.combos.length, 9);
    assert.strictEqual(batchResult.rows, 9);
    assert.strictEqual(batchResult.cols, 100);
    
    
    let defaultIdx = -1;
    for (let i = 0; i < batchResult.combos.length; i++) {
        if (batchResult.combos[i].period === 20 && 
            Math.abs(batchResult.combos[i].r_multi - 2.0) < 1e-10 &&
            batchResult.combos[i].smooth === false) {
            defaultIdx = i;
            break;
        }
    }
    
    assert(defaultIdx >= 0, 'Default parameters not found in batch result');
    
    
    const singleResult = wasm.cora_wave_js(close, 20, 2.0, false);
    const batchRow = batchResult.values.slice(defaultIdx * 100, (defaultIdx + 1) * 100);
    
    
    for (let i = 0; i < singleResult.length; i++) {
        if (!isNaN(batchRow[i]) && !isNaN(singleResult[i])) {
            assertClose(batchRow[i], singleResult[i], 1e-8, 
                       `Batch row doesn't match single calculation at index ${i}`);
        }
    }
});

test('CoRa Wave performance characteristics', () => {
    
    
    const trendUp = new Float64Array(100);
    const trendDown = new Float64Array(100);
    const choppy = new Float64Array(100);
    
    for (let i = 0; i < 100; i++) {
        trendUp[i] = 100 + i;  
        trendDown[i] = 200 - i;  
        choppy[i] = 100 + 10 * (i % 2);  
    }
    
    
    const resultChoppy = wasm.cora_wave_js(choppy, 10, 2.0, true);
    
    
    let validResult = [];
    for (let i = 0; i < resultChoppy.length; i++) {
        if (!isNaN(resultChoppy[i])) {
            validResult.push(resultChoppy[i]);
        }
    }
    
    if (validResult.length > 20) {
        
        const inputSlice = choppy.slice(choppy.length - validResult.length);
        let inputMean = 0, outputMean = 0;
        for (let i = 0; i < validResult.length; i++) {
            inputMean += inputSlice[i];
            outputMean += validResult[i];
        }
        inputMean /= validResult.length;
        outputMean /= validResult.length;
        
        let inputVar = 0, outputVar = 0;
        for (let i = 0; i < validResult.length; i++) {
            inputVar += Math.pow(inputSlice[i] - inputMean, 2);
            outputVar += Math.pow(validResult[i] - outputMean, 2);
        }
        inputVar /= validResult.length;
        outputVar /= validResult.length;
        
        assert(outputVar < inputVar, 'CoRa Wave should smooth choppy data');
    }
    
    
    const resultUp = wasm.cora_wave_js(trendUp, 10, 2.0, true);
    let validUp = [];
    for (let i = 0; i < resultUp.length; i++) {
        if (!isNaN(resultUp[i])) {
            validUp.push(resultUp[i]);
        }
    }
    
    if (validUp.length > 2) {
        
        let increasingCount = 0;
        for (let i = 1; i < validUp.length; i++) {
            if (validUp[i] > validUp[i-1]) {
                increasingCount++;
            }
        }
        const increasingRatio = increasingCount / (validUp.length - 1);
        assert(increasingRatio > 0.7, `CoRa Wave should follow upward trend (ratio: ${increasingRatio})`);
    }
});

test('CoRa Wave edge cases with extreme values', () => {
    
    const close = new Float64Array(testData.close.slice(0, 20));
    let result = wasm.cora_wave_js(close, 2, 2.0, false);
    assert.strictEqual(result.length, close.length);
    
    
    result = wasm.cora_wave_js(close, close.length, 2.0, false);
    assert.strictEqual(result.length, close.length);
    
    for (let i = 0; i < result.length - 1; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
    assert(!isNaN(result[result.length - 1]), 'Last value should not be NaN');
    
    
    result = wasm.cora_wave_js(close, 5, 10.0, false);
    assert.strictEqual(result.length, close.length);
    
    
    result = wasm.cora_wave_js(close, 5, 0.001, false);
    assert.strictEqual(result.length, close.length);
    
    
    result = wasm.cora_wave_js(close, 5, 0.0, false);
    assert.strictEqual(result.length, close.length);
});

test.after(() => {
    console.log('CoRa Wave WASM tests completed');
});
