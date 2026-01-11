/**
 * WASM binding tests for ADX indicator.
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

test('ADX partial params', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.adx_js(high, low, close, 14);
    assert.strictEqual(result.length, close.length);
});

test('ADX accuracy', async () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.adx;
    
    const result = wasm.adx_js(
        high,
        low,
        close,
        expected.defaultParams.period
    );
    
    assert.strictEqual(result.length, close.length);
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-1,  
        "ADX last 5 values mismatch"
    );
    
    
    await compareWithRust('adx', result, 'ohlc', expected.defaultParams);
});

test('ADX default candles', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.adx_js(high, low, close, 14);
    assert.strictEqual(result.length, close.length);
});

test('ADX zero period', () => {
    
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    const close = new Float64Array([9.0, 19.0, 29.0]);
    
    assert.throws(() => {
        wasm.adx_js(high, low, close, 0);
    }, /Invalid period: period = 0/);
});

test('ADX period exceeds length', () => {
    
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    const close = new Float64Array([9.0, 19.0, 29.0]);
    
    assert.throws(() => {
        wasm.adx_js(high, low, close, 10);
    }, /Invalid period: period = 10/);
});

test('ADX very small dataset', () => {
    
    const high = new Float64Array([42.0]);
    const low = new Float64Array([41.0]);
    const close = new Float64Array([40.5]);
    
    assert.throws(() => {
        wasm.adx_js(high, low, close, 14);
    }, /Invalid period|Not enough valid data/);
});

test('ADX input length mismatch', () => {
    
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0]);  
    const close = new Float64Array([9.0, 19.0, 29.0]);
    
    assert.throws(() => {
        wasm.adx_js(high, low, close, 14);
    }, /Input arrays must have the same length/);
});

test('ADX all NaN input', () => {
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.adx_js(allNaN, allNaN, allNaN, 14);
    }, /All values are NaN/);
});


test('ADX NaN handling', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.adx_js(high, low, close, 14);
    assert.strictEqual(result.length, close.length);
    
    
    const warmupPeriod = 2 * 14 - 1;
    
    
    assertAllNaN(result.slice(0, warmupPeriod), `Expected NaN in first ${warmupPeriod} values`);
    
    
    if (result.length > warmupPeriod + 10) {
        for (let i = warmupPeriod + 10; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i} after warmup period`);
        }
    }
});

test('ADX leading NaN values', () => {
    
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    
    for (let i = 0; i < 5; i++) {
        high[i] = NaN;
        low[i] = NaN;
        close[i] = NaN;
    }
    
    const result = wasm.adx_js(high, low, close, 14);
    assert.strictEqual(result.length, close.length);
    
    
    
    assertAllNaN(result.slice(0, 5), "Expected NaN where input has NaN");
});

test('ADX batch single parameter set', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    
    const batchResult = wasm.adx_batch_js(
        high,
        low,
        close,
        14, 14, 0  
    );
    
    
    const singleResult = wasm.adx_js(high, low, close, 14);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('ADX batch multiple periods', () => {
    
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    
    const batchResult = wasm.adx_batch_js(
        high,
        low,
        close,
        10, 18, 4  
    );
    
    
    assert.strictEqual(batchResult.length, 3 * 100);
    
    
    const periods = [10, 14, 18];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.slice(rowStart, rowEnd);
        
        const singleResult = wasm.adx_js(high, low, close, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('ADX batch metadata', () => {
    
    const metadata = wasm.adx_batch_metadata_js(
        10, 18, 4  
    );
    
    
    assert.strictEqual(metadata.length, 3);
    
    
    assert.strictEqual(metadata[0], 10);
    assert.strictEqual(metadata[1], 14);
    assert.strictEqual(metadata[2], 18);
});

test('ADX batch full parameter sweep', () => {
    
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.adx_batch_js(
        high,
        low,
        close,
        10, 20, 5  
    );
    
    const metadata = wasm.adx_batch_metadata_js(10, 20, 5);
    
    
    assert.strictEqual(metadata.length, 3);
    assert.strictEqual(batchResult.length, 3 * 50);
    
    
    for (let combo = 0; combo < metadata.length; combo++) {
        const period = metadata[combo];
        const rowStart = combo * 50;
        const rowData = batchResult.slice(rowStart, rowStart + 50);
        
        
        const warmupPeriod = 2 * period - 1;
        
        
        for (let i = 0; i < Math.min(warmupPeriod, 50); i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${period}`);
        }
        
        
        if (50 > warmupPeriod + 5) {
            let hasValues = false;
            for (let i = warmupPeriod; i < 50; i++) {
                if (!isNaN(rowData[i])) {
                    hasValues = true;
                    break;
                }
            }
            assert(hasValues, `Expected some non-NaN values after warmup for period ${period}`);
        }
    }
});

test('ADX batch edge cases', () => {
    
    const high = new Float64Array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19]);
    const low = new Float64Array([9, 10, 11, 12, 13, 14, 15, 16, 17, 18]);
    const close = new Float64Array([9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5]);
    
    
    const singleBatch = wasm.adx_batch_js(
        high,
        low,
        close,
        5, 5, 1
    );
    
    assert.strictEqual(singleBatch.length, 10);
    
    
    const largeBatch = wasm.adx_batch_js(
        high,
        low,
        close,
        5, 7, 10  
    );
    
    
    assert.strictEqual(largeBatch.length, 10);
    
    
    assert.throws(() => {
        wasm.adx_batch_js(
            new Float64Array([]),
            new Float64Array([]),
            new Float64Array([]),
            14, 14, 0
        );
    }, /Input data slice is empty|All values are NaN|unreachable|RuntimeError/);
});

test('ADX batch invalid params', () => {
    
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    const close = new Float64Array(testData.close.slice(0, 50));
    
    
    assert.throws(() => {
        wasm.adx_batch_js(
            high,
            low,
            close,
            100, 100, 0
        );
    }, /Not enough valid data/);
});


test('ADX batch - new ergonomic API with single parameter', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.adx_batch(high, low, close, {
        period_range: [14, 14, 0]
    });
    
    
    assert(result.values, 'Should have values array');
    assert(result.combos, 'Should have combos array');
    assert(typeof result.rows === 'number', 'Should have rows count');
    assert(typeof result.cols === 'number', 'Should have cols count');
    
    
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.combos.length, 1);
    assert.strictEqual(result.values.length, close.length);
    
    
    const combo = result.combos[0];
    assert.strictEqual(combo.period, 14);
    
    
    const oldResult = wasm.adx_js(high, low, close, 14);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(result.values[i])) {
            continue; 
        }
        assert(Math.abs(oldResult[i] - result.values[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});

test('ADX batch - new API with multiple parameters', () => {
    
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const result = wasm.adx_batch(high, low, close, {
        period_range: [10, 18, 4]  
    });
    
    
    assert.strictEqual(result.rows, 3);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.combos.length, 3);
    assert.strictEqual(result.values.length, 150);
    
    
    const expectedPeriods = [10, 14, 18];
    for (let i = 0; i < expectedPeriods.length; i++) {
        assert.strictEqual(result.combos[i].period, expectedPeriods[i]);
    }
    
    
    const secondRow = result.values.slice(result.cols, 2 * result.cols);
    assert.strictEqual(secondRow.length, 50);
    
    
    const oldResult = wasm.adx_js(high, low, close, 10);
    const firstRow = result.values.slice(0, result.cols);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(firstRow[i])) {
            continue; 
        }
        assert(Math.abs(oldResult[i] - firstRow[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});

test('ADX batch - new API matches old API results', () => {
    
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const params = {
        period_range: [10, 20, 5]
    };
    
    
    const oldValues = wasm.adx_batch_js(
        high,
        low,
        close,
        params.period_range[0], params.period_range[1], params.period_range[2]
    );
    
    
    const newResult = wasm.adx_batch(high, low, close, params);
    
    
    assert.strictEqual(oldValues.length, newResult.values.length);
    
    for (let i = 0; i < oldValues.length; i++) {
        if (isNaN(oldValues[i]) && isNaN(newResult.values[i])) {
            continue; 
        }
        assert(Math.abs(oldValues[i] - newResult.values[i]) < 1e-10,
               `Value mismatch at index ${i}: old=${oldValues[i]}, new=${newResult.values[i]}`);
    }
});

test('ADX batch - new API error handling', () => {
    const high = new Float64Array(testData.high.slice(0, 10));
    const low = new Float64Array(testData.low.slice(0, 10));
    const close = new Float64Array(testData.close.slice(0, 10));
    
    
    assert.throws(() => {
        wasm.adx_batch(high, low, close, {
            period_range: [14, 14]  
        });
    }, /Invalid config/);
    
    
    assert.throws(() => {
        wasm.adx_batch(high, low, close, {
            
        });
    }, /Invalid config/);
    
    
    assert.throws(() => {
        wasm.adx_batch(high, low, close, {
            period_range: "invalid"
        });
    }, /Invalid config/);
});

test('ADX warmup behavior', () => {
    
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    const close = new Float64Array(testData.close.slice(0, 50));
    const period = 14;
    
    const result = wasm.adx_js(high, low, close, period);
    
    
    
    
    
    const warmupPeriod = 2 * period - 1;  
    
    
    for (let i = 0; i < warmupPeriod; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} during warmup`);
    }
    
    
    if (result.length > warmupPeriod) {
        assert(!isNaN(result[warmupPeriod]), 
               `Expected first valid value at index ${warmupPeriod}`);
    }
    
    
    if (result.length > warmupPeriod + 5) {
        for (let i = warmupPeriod; i < result.length; i++) {
            assert(!isNaN(result[i]), 
                   `Expected all non-NaN values after warmup period at index ${i}`);
        }
    }
});



test.after(() => {
    console.log('ADX WASM tests completed');
});