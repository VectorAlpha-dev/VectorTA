/**
 * WASM binding tests for DMA (Dickson Moving Average) indicator.
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
        console.error('Failed to load WASM module:', error);
        throw error;
    }
    
    
    testData = loadTestData();
});

test('DMA partial params', () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.dma;
    
    const result = wasm.dma_js(
        close,
        expected.defaultParams.hull_length,
        expected.defaultParams.ema_length,
        expected.defaultParams.ema_gain_limit,
        expected.defaultParams.hull_ma_type
    );
    assert.strictEqual(result.length, close.length);
    
    
    const hullLen = expected.defaultParams.hull_length;
    const emaLen = expected.defaultParams.ema_length;
    const warmup = Math.max(hullLen, emaLen) - 1;
    
    assertAllNaN(result.slice(0, warmup), `Expected NaN during warmup period (first ${warmup} values)`);
    
    
    let nonNanCount = 0;
    for (let i = warmup; i < result.length; i++) {
        if (!isNaN(result[i])) nonNanCount++;
    }
    assert(nonNanCount > close.length - warmup - 10, "Should have values after warmup");
});

test('DMA accuracy', () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.dma;
    
    const result = wasm.dma_js(
        close,
        expected.defaultParams.hull_length,
        expected.defaultParams.ema_length,
        expected.defaultParams.ema_gain_limit,
        expected.defaultParams.hull_ma_type
    );
    
    assert.strictEqual(result.length, close.length);
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        0.001,  
        "DMA last 5 values mismatch"
    );
});

test('DMA default candles', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const result = wasm.dma_js(close, 7, 20, 50, "WMA");
    assert.strictEqual(result.length, close.length);
    
    
    const warmup = Math.max(7, 20) - 1;
    
    assertAllNaN(result.slice(0, warmup), `Expected NaN in first ${warmup} values`);
    
    
    let hasValues = false;
    for (let i = warmup; i < result.length; i++) {
        if (!isNaN(result[i])) {
            hasValues = true;
            break;
        }
    }
    assert(hasValues, "Should have values after warmup");
});

test('DMA zero hull period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.dma_js(inputData, 0, 20, 50, "WMA");
    }, /Invalid period/);
});

test('DMA zero ema period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.dma_js(inputData, 7, 0, 50, "WMA");
    }, /Invalid period/);
});

test('DMA empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.dma_js(empty, 7, 20, 50, "WMA");
    }, /empty/i);
});

test('DMA all NaN', () => {
    
    const nanData = new Float64Array([NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.dma_js(nanData, 7, 20, 50, "WMA");
    }, /NaN/i);
});

test('DMA invalid hull type', () => {
    
    const inputData = new Float64Array(Array(50).fill(10.0));
    
    assert.throws(() => {
        wasm.dma_js(inputData, 7, 20, 50, "INVALID");
    }, /Invalid Hull MA type/);
});

test('DMA EMA hull type', () => {
    
    const inputData = new Float64Array(Array.from({length: 100}, (_, i) => i));
    
    const result = wasm.dma_js(
        inputData,
        7,    
        20,   
        50,   
        "EMA" 
    );
    
    assert.strictEqual(result.length, inputData.length);
    
    const nonNanCount = result.filter(x => !isNaN(x)).length;
    assert(nonNanCount > 0, 'Should produce non-NaN values after warmup');
});

test('DMA period exceeds length', () => {
    
    const smallData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.dma_js(smallData, 10, 20, 50, "WMA");
    }, /Invalid period/);
});

test('DMA insufficient data', () => {
    
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.dma_js(singlePoint, 7, 20, 50, "WMA");
    });
});

test('DMA NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.dma;
    
    const result = wasm.dma_js(
        close,
        expected.defaultParams.hull_length,
        expected.defaultParams.ema_length,
        expected.defaultParams.ema_gain_limit,
        expected.defaultParams.hull_ma_type
    );
    
    assert.strictEqual(result.length, close.length);
    
    
    const hullLen = expected.defaultParams.hull_length;
    const emaLen = expected.defaultParams.ema_length;
    const warmup = Math.max(hullLen, emaLen) - 1;
    
    
    if (result.length > warmup + 100) {
        
        for (let i = warmup + 100; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    
    assertAllNaN(result.slice(0, warmup), `Expected NaN in warmup period (first ${warmup} values)`);
});




test('DMA constant input', () => {
    
    const constantVal = EXPECTED_OUTPUTS.dma.constantValue;
    const inputData = new Float64Array(100).fill(constantVal);
    
    const result = wasm.dma_js(inputData, 7, 20, 50, "WMA");
    
    assert.strictEqual(result.length, inputData.length);
    
    
    
    const last10 = result.slice(-10);
    const validValues = last10.filter(x => !isNaN(x));
    
    if (validValues.length > 0) {
        
        validValues.forEach((val, i) => {
            assertClose(
                val, 
                constantVal,
                0.01,  
                `DMA should converge to constant input value at index ${i}`
            );
        });
    }
});

test('DMA mixed hull types', () => {
    
    const inputData = new Float64Array(Array.from({length: 100}, (_, i) => i * 10 + Math.sin(i)));
    const expected = EXPECTED_OUTPUTS.dma;
    
    const results = {};
    for (const hullType of expected.hullMaTypes) {
        const result = wasm.dma_js(
            inputData,
            7,    
            20,   
            50,   
            hullType
        );
        
        assert.strictEqual(result.length, inputData.length, `Result length mismatch for hull_type=${hullType}`);
        
        
        const nonNanCount = result.filter(x => !isNaN(x)).length;
        assert(nonNanCount > 0, `Should produce values for hull_type=${hullType}`);
        
        results[hullType] = result;
    }
    
    
    if (results['EMA'] && results['WMA']) {
        let hasDifference = false;
        for (let i = 0; i < results['EMA'].length; i++) {
            if (!isNaN(results['EMA'][i]) && !isNaN(results['WMA'][i])) {
                if (Math.abs(results['EMA'][i] - results['WMA'][i]) > 1e-10) {
                    hasDifference = true;
                    break;
                }
            }
        }
        assert(hasDifference, "EMA and WMA hull types should produce different results");
    }
});

test('DMA gain limit edge cases', () => {
    
    const inputData = new Float64Array(Array.from({length: 50}, (_, i) => i));
    
    
    const resultZero = wasm.dma_js(
        inputData,
        7,    
        20,   
        0,    
        "WMA"
    );
    assert.strictEqual(resultZero.length, inputData.length, "Zero gain limit should produce output");
    
    
    const resultLarge = wasm.dma_js(
        inputData,
        7,     
        20,    
        1000,  
        "WMA"
    );
    assert.strictEqual(resultLarge.length, inputData.length, "Large gain limit should produce output");
});

test('DMA trending data', () => {
    
    
    const uptrend = new Float64Array(Array.from({length: 100}, (_, i) => i));
    const resultUp = wasm.dma_js(uptrend, 7, 20, 50, "WMA");
    assert.strictEqual(resultUp.length, uptrend.length);
    
    
    const downtrend = new Float64Array(Array.from({length: 100}, (_, i) => 100 - i));
    const resultDown = wasm.dma_js(downtrend, 7, 20, 50, "WMA");
    assert.strictEqual(resultDown.length, downtrend.length);
    
    
    const warmup = 19; 
    
    
    if (resultUp.length > warmup + 10) {
        const last10 = resultUp.slice(-10);
        const valid = last10.filter(x => !isNaN(x));
        if (valid.length > 1) {
            
            let increasingCount = 0;
            for (let i = 1; i < valid.length; i++) {
                if (valid[i] > valid[i-1]) increasingCount++;
            }
            assert(increasingCount > valid.length / 2, "DMA should follow uptrend");
        }
    }
});

test('DMA oscillating data', () => {
    
    
    const points = 100;
    const oscillating = new Float64Array(Array.from({length: points}, (_, i) => {
        const x = (i / points) * 4 * Math.PI;
        return 50.0 + 10.0 * Math.sin(x);
    }));
    
    const result = wasm.dma_js(
        oscillating,
        7,    
        20,   
        50,   
        "WMA"
    );
    
    assert.strictEqual(result.length, oscillating.length);
    
    
    const warmup = 19;
    if (result.length > warmup) {
        const validResult = result.slice(warmup).filter(x => !isNaN(x));
        const validInput = oscillating.slice(warmup);
        
        
        const mean = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
        const variance = arr => {
            const m = mean(arr);
            return arr.reduce((a, b) => a + Math.pow(b - m, 2), 0) / arr.length;
        };
        
        if (validResult.length > 0) {
            const resultVar = variance(validResult);
            const inputVar = variance(validInput);
            
            
            assert(resultVar < inputVar, "DMA should smooth oscillating data");
        }
    }
});

test('DMA extreme ratios', () => {
    
    const inputData = new Float64Array(Array.from({length: 100}, (_, i) => i * 2 + Math.random()));
    
    
    const result1 = wasm.dma_js(
        inputData,
        3,    
        50,   
        50,   
        "WMA"
    );
    assert.strictEqual(result1.length, inputData.length);
    
    
    const result2 = wasm.dma_js(
        inputData,
        50,   
        3,    
        50,   
        "WMA"
    );
    assert.strictEqual(result2.length, inputData.length);
    
    
    let hasDifference = false;
    for (let i = 0; i < result1.length; i++) {
        if (!isNaN(result1[i]) && !isNaN(result2[i])) {
            if (Math.abs(result1[i] - result2[i]) > 0.01) {
                hasDifference = true;
                break;
            }
        }
    }
    assert(hasDifference, "Different hull/ema ratios should produce different results");
});

test('DMA zero-copy API', () => {
    
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]);
    const hullLength = 5;
    const emaLength = 10;
    const emaGainLimit = 50;
    const hullMaType = "WMA";
    
    
    const ptr = wasm.dma_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    
    const memoryBuffer = wasm.__wasm.memory.buffer;
    const memView = new Float64Array(
        memoryBuffer,
        ptr,
        data.length
    );
    
    
    memView.set(data);
    
    try {
        
        wasm.dma_into(
            ptr,
            ptr,
            data.length,
            hullLength,
            emaLength,
            emaGainLimit,
            hullMaType
        );
        
        
        const regularResult = wasm.dma_js(data, hullLength, emaLength, emaGainLimit, hullMaType);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; 
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        
        wasm.dma_free(ptr, data.length);
    }
});

test('DMA zero-copy error handling', () => {
    
    assert.throws(() => {
        wasm.dma_into(0, 0, 10, 7, 20, 50, "WMA");
    }, /null pointer|invalid memory/i);
    
    
    const ptr = wasm.dma_alloc(10);
    try {
        
        assert.throws(() => {
            wasm.dma_into(ptr, ptr, 10, 0, 20, 50, "WMA");
        }, /Invalid period/);
        
        
        assert.throws(() => {
            wasm.dma_into(ptr, ptr, 10, 7, 0, 50, "WMA");
        }, /Invalid period/);
        
        
        
        assert.throws(() => {
            wasm.dma_into(ptr, ptr, 10, 7, 20, 50, "INVALID");
        });
    } finally {
        wasm.dma_free(ptr, 10);
    }
});

test('DMA memory management', () => {
    
    const sizes = [100, 1000, 10000];
    
    for (const size of sizes) {
        const ptr = wasm.dma_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        
        const memoryBuffer = wasm.__wasm.memory.buffer;
        const memView = new Float64Array(memoryBuffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }
        
        
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
        }
        
        
        wasm.dma_free(ptr, size);
    }
});

test.after(() => {
    console.log('DMA WASM tests completed');
});
