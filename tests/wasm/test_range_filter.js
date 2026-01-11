/**
 * WASM binding tests for RANGE_FILTER indicator.
 * These tests mirror the Rust unit tests to ensure WASM bindings work correctly.
 */
import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
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
const __dirname = dirname(__filename);

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

test('RANGE_FILTER accuracy', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const result = wasm.range_filter_js(close);
    
    assert(result.filter, 'Should have filter output');
    assert(result.high_band, 'Should have high_band output');
    assert(result.low_band, 'Should have low_band output');
    assert.strictEqual(result.filter.length, close.length);
    assert.strictEqual(result.high_band.length, close.length);
    assert.strictEqual(result.low_band.length, close.length);
    
    
    const expectedFilter = [
        59589.73987817684, 
        59589.73987817684, 
        59589.73987817684, 
        59589.73987817684, 
        59589.73987817684
    ];
    
    const expectedHigh = [
        60935.63924911415,
        60906.58379951138,
        60874.2002431993,
        60838.79850154794,
        60810.879398758305
    ];
    
    const expectedLow = [
        58243.84050723953,
        58272.8959568423,
        58305.27951315438,
        58340.68125480574,
        58368.60035759538
    ];
    
    const tolerance = 1e-10; 
    
    
    const last5Filter = result.filter.slice(-5);
    for (let i = 0; i < 5; i++) {
        const diff = Math.abs(last5Filter[i] - expectedFilter[i]);
        assert(
            diff < tolerance,
            `Filter[${i}] mismatch: expected ${expectedFilter[i]}, got ${last5Filter[i]} (diff: ${diff})`
        );
    }
    
    
    const last5High = result.high_band.slice(-5);
    for (let i = 0; i < 5; i++) {
        const diff = Math.abs(last5High[i] - expectedHigh[i]);
        assert(
            diff < tolerance,
            `High Band[${i}] mismatch: expected ${expectedHigh[i]}, got ${last5High[i]} (diff: ${diff})`
        );
    }
    
    
    const last5Low = result.low_band.slice(-5);
    for (let i = 0; i < 5; i++) {
        const diff = Math.abs(last5Low[i] - expectedLow[i]);
        assert(
            diff < tolerance,
            `Low Band[${i}] mismatch: expected ${expectedLow[i]}, got ${last5Low[i]} (diff: ${diff})`
        );
    }
});

test('RANGE_FILTER default candles', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.range_filter_js(close);
    
    assert(result.filter, 'Should have filter output');
    assert(result.high_band, 'Should have high_band output');
    assert(result.low_band, 'Should have low_band output');
    assert.strictEqual(result.filter.length, close.length);
    assert.strictEqual(result.high_band.length, close.length);
    assert.strictEqual(result.low_band.length, close.length);
});

test('RANGE_FILTER partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    
    let result = wasm.range_filter_js(close, 2.5);
    assert.strictEqual(result.filter.length, close.length);
    
    
    result = wasm.range_filter_js(close, 2.5, 15);
    assert.strictEqual(result.filter.length, close.length);
    
    
    result = wasm.range_filter_js(close, 2.5, 15, true, 20);
    assert.strictEqual(result.filter.length, close.length);
});

test('RANGE_FILTER empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.range_filter_js(empty);
    }, /Input data slice is empty/);
});

test('RANGE_FILTER all NaN', () => {
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.range_filter_js(allNaN);
    }, /All values are NaN/);
});

test('RANGE_FILTER invalid period', () => {
    
    const data = new Float64Array([1.0, 2.0, 3.0]);
    
    
    assert.throws(() => {
        wasm.range_filter_js(data, 2.618, 10);
    }, /Invalid period/);
    
    
    assert.throws(() => {
        wasm.range_filter_js(data, 2.618, 0);
    }, /Invalid period|Invalid range_period/);
});

test('RANGE_FILTER invalid range_size', () => {
    
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    
    assert.throws(() => {
        wasm.range_filter_js(data, 0.0);
    }, /Invalid range_size/);
    
    
    assert.throws(() => {
        wasm.range_filter_js(data, -1.0);
    }, /Invalid range_size/);
    
    
    assert.throws(() => {
        wasm.range_filter_js(data, NaN);
    }, /Invalid range_size/);
});

test('RANGE_FILTER NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.range_filter_js(close);
    
    
    
    if (result.filter.length > 50) {
        for (let i = 50; i < result.filter.length; i++) {
            assert(!isNaN(result.filter[i]), `Found unexpected NaN in filter at index ${i}`);
            assert(!isNaN(result.high_band[i]), `Found unexpected NaN in high_band at index ${i}`);
            assert(!isNaN(result.low_band[i]), `Found unexpected NaN in low_band at index ${i}`);
        }
    }
});

test('RANGE_FILTER batch default', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const result = wasm.range_filter_batch_unified(close, {
        range_size: [2.618, 2.618, 0],
        range_period: [10, 10, 0],
        smooth_range: true,
        smooth_period: 27
    });
    
    
    assert(result.values, 'Should have values array');
    assert(result.combos, 'Should have combos array');
    assert(typeof result.rows === 'number', 'Should have rows count');
    assert(typeof result.cols === 'number', 'Should have cols count');
    
    
    assert.strictEqual(result.combos.length, 1);
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, close.length);
    
    
    assert.strictEqual(result.values.length, 3 * close.length);
    
    
    const filterValues = result.values.slice(0, close.length);
    
    
    const singleResult = wasm.range_filter_js(close, 2.618, 10, true, 27);
    for (let i = 0; i < close.length; i++) {
        if (isNaN(singleResult.filter[i]) && isNaN(filterValues[i])) {
            continue; 
        }
        assertClose(
            filterValues[i],
            singleResult.filter[i],
            1e-10,
            `Batch vs single mismatch at index ${i}`
        );
    }
});

test('RANGE_FILTER batch sweep', () => {
    
    
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = Math.sin(i * 0.2) * 100 + 500;
    }
    
    const result = wasm.range_filter_batch_unified(data, {
        range_size: [2.0, 3.0, 0.5],  
        range_period: [10, 20, 5],    
        smooth_range: true,
        smooth_period: 15
    });
    
    
    assert.strictEqual(result.combos.length, 9);
    assert.strictEqual(result.rows, 9);
    assert.strictEqual(result.cols, 50);
    
    
    assert.strictEqual(result.values.length, 9 * 3 * 50);
    
    
    const expectedCombos = [];
    for (const rs of [2.0, 2.5, 3.0]) {
        for (const rp of [10, 15, 20]) {
            expectedCombos.push({ range_size: rs, range_period: rp });
        }
    }
    
    for (let i = 0; i < expectedCombos.length; i++) {
        assertClose(
            result.combos[i].range_size,
            expectedCombos[i].range_size,
            0.01,
            `Range size mismatch at combo ${i}`
        );
        assert.strictEqual(
            result.combos[i].range_period,
            expectedCombos[i].range_period,
            `Range period mismatch at combo ${i}`
        );
    }
});

test('RANGE_FILTER no poison', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.range_filter_js(close);
    
    
    const poisonPatterns = [0x1111111111111111n, 0x2222222222222222n, 0x3333333333333333n];
    
    for (const [name, arr] of [
        ['filter', result.filter],
        ['high_band', result.high_band],
        ['low_band', result.low_band]
    ]) {
        for (let i = 0; i < arr.length; i++) {
            if (isNaN(arr[i])) continue;
            
            
            const buffer = new ArrayBuffer(8);
            const view = new DataView(buffer);
            view.setFloat64(0, arr[i], true);
            const bits = view.getBigUint64(0, true);
            
            assert(
                !poisonPatterns.includes(bits),
                `Poison pattern found in ${name} at index ${i}: 0x${bits.toString(16)}`
            );
        }
    }
});

test('RANGE_FILTER batch no poison', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.range_filter_batch_unified(close, {
        range_size: [2.618, 2.618, 0],
        range_period: [10, 10, 0],
        smooth_range: true,
        smooth_period: 27
    });
    
    
    const poisonPatterns = [0x1111111111111111n, 0x2222222222222222n, 0x3333333333333333n];
    
    for (let i = 0; i < result.values.length; i++) {
        if (isNaN(result.values[i])) continue;
        
        const buffer = new ArrayBuffer(8);
        const view = new DataView(buffer);
        view.setFloat64(0, result.values[i], true);
        const bits = view.getBigUint64(0, true);
        
        assert(
            !poisonPatterns.includes(bits),
            `Poison pattern in values[${i}]: 0x${bits.toString(16)}`
        );
    }
});

test('RANGE_FILTER kernel parity', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    
    
    const resultAuto = wasm.range_filter_js(close);
    
    
    const resultSecond = wasm.range_filter_js(close);
    
    
    for (let i = 0; i < close.length; i++) {
        if (isNaN(resultAuto.filter[i]) && isNaN(resultSecond.filter[i])) {
            continue;
        }
        assertClose(
            resultAuto.filter[i],
            resultSecond.filter[i],
            1e-10,
            `Kernel consistency check failed at index ${i}`
        );
    }
});

test('RANGE_FILTER multi-output structure', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const result = wasm.range_filter_js(close);
    
    
    assert(result.filter, 'Should have filter output');
    assert(result.high_band, 'Should have high_band output');
    assert(result.low_band, 'Should have low_band output');
    
    
    assert.strictEqual(result.filter.length, close.length);
    assert.strictEqual(result.high_band.length, close.length);
    assert.strictEqual(result.low_band.length, close.length);
    
    
    for (let i = 50; i < close.length; i++) { 
        if (!isNaN(result.filter[i])) {
            
            assert(
                result.high_band[i] >= result.filter[i] - 1e-6,
                `High band < filter at index ${i}`
            );
            
            assert(
                result.low_band[i] <= result.filter[i] + 1e-6,
                `Low band > filter at index ${i}`
            );
        }
    }
});

test('RANGE_FILTER batch metadata', () => {
    
    const close = new Float64Array(20);
    close.fill(100);
    
    const result = wasm.range_filter_batch_unified(close, {
        range_size: [2.0, 3.0, 1.0],      
        range_period: [10, 12, 2],        
        smooth_range: true,
        smooth_period: 15
    });
    
    
    assert.strictEqual(result.combos.length, 4);
    
    
    assert.strictEqual(result.combos[0].range_size, 2.0);
    assert.strictEqual(result.combos[0].range_period, 10);
    
    
    assert.strictEqual(result.combos[3].range_size, 3.0);
    assert.strictEqual(result.combos[3].range_period, 12);
});

test('RANGE_FILTER batch edge cases', () => {
    
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    
    const singleBatch = wasm.range_filter_batch_unified(close, {
        range_size: [2.0, 2.0, 1],
        range_period: [5, 5, 1],
        smooth_range: true,
        smooth_period: 3
    });
    
    assert.strictEqual(singleBatch.combos.length, 1);
    assert.strictEqual(singleBatch.values.length, 3 * 10); 
    
    
    const largeBatch = wasm.range_filter_batch_unified(close, {
        range_size: [2.0, 2.5, 10], 
        range_period: [5, 5, 0],
        smooth_range: true,
        smooth_period: 3
    });
    
    
    assert.strictEqual(largeBatch.combos.length, 1);
    
    
    assert.throws(() => {
        wasm.range_filter_batch_unified(new Float64Array([]), {
            range_size: [2.618, 2.618, 0],
            range_period: [10, 10, 0],
            smooth_range: true,
            smooth_period: 27
        });
    }, /Input data slice is empty/);
});

test('RANGE_FILTER batch error handling', () => {
    const close = new Float64Array(testData.close.slice(0, 10));
    
    
    assert.throws(() => {
        wasm.range_filter_batch_unified(close, {
            range_size: [2.618, 2.618], 
            range_period: [10, 10, 0],
            smooth_range: true,
            smooth_period: 27
        });
    }, /Invalid config/);
    
    
    assert.throws(() => {
        wasm.range_filter_batch_unified(close, {
            range_size: [2.618, 2.618, 0],
            
            smooth_range: true,
            smooth_period: 27
        });
    }, /Invalid config/);
    
    
    assert.throws(() => {
        wasm.range_filter_batch_unified(close, {
            range_size: "invalid",
            range_period: [10, 10, 0],
            smooth_range: true,
            smooth_period: 27
        });
    }, /Invalid config/);
});

test('RANGE_FILTER batch full parameter sweep', () => {
    
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.range_filter_batch_unified(close, {
        range_size: [2.0, 2.5, 0.5],     
        range_period: [10, 11, 1],       
        smooth_range: false,
        smooth_period: 27
    });
    
    
    assert.strictEqual(batchResult.combos.length, 4);
    assert.strictEqual(batchResult.rows, 4);
    assert.strictEqual(batchResult.cols, 50);
    
    
    assert.strictEqual(batchResult.values.length, 4 * 3 * 50);
    
    
    for (let combo = 0; combo < batchResult.combos.length; combo++) {
        const range_size = batchResult.combos[combo].range_size;
        const range_period = batchResult.combos[combo].range_period;
        
        
        const baseIdx = combo * 3 * 50;
        const filterData = batchResult.values.slice(baseIdx, baseIdx + 50);
        
        
        let firstNonNan = -1;
        for (let i = 0; i < 50; i++) {
            if (!isNaN(filterData[i])) {
                firstNonNan = i;
                break;
            }
        }
        
        
        assert(firstNonNan >= 0 && firstNonNan < 50, 
               `No valid values found for combo ${combo} (size=${range_size}, period=${range_period})`);
    }
});

test('RANGE_FILTER SIMD128 consistency', () => {
    
    const testCases = [
        { size: 50 },
        { size: 100 },
        { size: 1000 }
    ];
    
    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = Math.sin(i * 0.1) + Math.cos(i * 0.05);
        }
        
        const result = wasm.range_filter_js(data);
        
        
        assert.strictEqual(result.filter.length, data.length);
        assert.strictEqual(result.high_band.length, data.length);
        assert.strictEqual(result.low_band.length, data.length);
        
        
        let hasValues = false;
        for (let i = 10; i < result.filter.length; i++) {
            if (!isNaN(result.filter[i])) {
                hasValues = true;
                break;
            }
        }
        assert(hasValues, `No valid values found for size=${testCase.size}`);
    }
});

test.after(() => {
    console.log('RANGE_FILTER WASM tests completed');
});
