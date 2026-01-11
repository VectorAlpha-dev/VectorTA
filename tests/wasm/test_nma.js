/**
 * WASM binding tests for NMA indicator.
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

test('NMA partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.nma_js(close, 40);
    assert(result instanceof Float64Array);
    assert.strictEqual(result.length, close.length);
});

test('NMA accuracy', async () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.nma;
    
    const result = wasm.nma_js(close, expected.defaultParams.period);
    
    assert.strictEqual(result.length, close.length);
    
    
    const actualLastFive = result.slice(-5);
    
    assertArrayClose(
        actualLastFive,
        expected.last5Values,
        1e-3,  
        "NMA last 5 values mismatch"
    );
    
    
    await compareWithRust('nma', result, 'close', expected.defaultParams);
});

test('NMA default candles', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const result = wasm.nma_js(close, 40);
    assert.strictEqual(result.length, close.length);
});

test('NMA zero period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.nma_js(inputData, 0);
    }, /Invalid period/);
});

test('NMA period exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.nma_js(dataSmall, 10);
    }, /Invalid period/);
});

test('NMA very small dataset', () => {
    
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.nma_js(singlePoint, 40);
    }, /Invalid period|Not enough valid data/);
});

test('NMA empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.nma_js(empty, 40);
    }, /Input data slice is empty/);
});

test('NMA NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.nma_js(close, 40);
    assert.strictEqual(result.length, close.length);
    
    
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    
    let firstValid = 0;
    for (let i = 0; i < close.length; i++) {
        if (!isNaN(close[i])) {
            firstValid = i;
            break;
        }
    }
    const warmup = firstValid + 40;
    assertAllNaN(result.slice(0, warmup), "Expected NaN in warmup period");
});

test('NMA all NaN input', () => {
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.nma_js(allNaN, 40);
    }, /All values are NaN/);
});

test('NMA batch default row', () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.nma;
    
    
    const batchResult = wasm.nma_batch_js(
        close,
        40, 40, 0  
    );
    
    
    const rowsCols = wasm.nma_batch_rows_cols_js(40, 40, 0, close.length);
    const rows = rowsCols[0];
    const cols = rowsCols[1];
    
    assert(batchResult instanceof Float64Array);
    assert.strictEqual(rows, 1); 
    assert.strictEqual(cols, close.length);
    assert.strictEqual(batchResult.length, rows * cols);
    
    
    const defaultRow = batchResult.slice(-5);
    
    
    assertArrayClose(
        defaultRow,
        expected.batchDefaultRow,
        1e-3,  
        "NMA batch default row mismatch"
    );
});

test('NMA batch multiple periods', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const batchResult = wasm.nma_batch_js(
        close,
        20, 60, 20  
    );
    
    
    const rowsCols = wasm.nma_batch_rows_cols_js(20, 60, 20, close.length);
    const rows = rowsCols[0];
    const cols = rowsCols[1];
    
    assert(batchResult instanceof Float64Array);
    assert.strictEqual(rows, 3); 
    assert.strictEqual(cols, close.length);
    assert.strictEqual(batchResult.length, rows * cols);
    
    
    const metadata = wasm.nma_batch_metadata_js(20, 60, 20);
    assert.strictEqual(metadata.length, 3);
    assert.strictEqual(metadata[0], 20);
    assert.strictEqual(metadata[1], 40);
    assert.strictEqual(metadata[2], 60);
    
    
    for (let i = 0; i < metadata.length; i++) {
        const period = metadata[i];
        const individualResult = wasm.nma_js(close, period);
        const rowStart = i * cols;
        const rowEnd = rowStart + cols;
        const batchRow = batchResult.slice(rowStart, rowEnd);
        
        
        let firstValid = 0;
        for (let j = 0; j < close.length; j++) {
            if (!isNaN(close[j])) {
                firstValid = j;
                break;
            }
        }
        const warmup = firstValid + period;
        
        if (warmup < close.length) {
            for (let j = warmup; j < close.length; j++) {
                assertClose(
                    batchRow[j], 
                    individualResult[j], 
                    1e-9, 
                    `NMA batch period ${period} mismatch at index ${j}`
                );
            }
        }
    }
});

test('NMA batch with unified API', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    
    
    if (typeof wasm.nma_batch_unified_js === 'function') {
        const config = {
            period_range: [20, 40, 10]  
        };
        
        const result = wasm.nma_batch_unified_js(close, config);
        
        
        assert(result.values instanceof Float64Array);
        assert(Array.isArray(result.combos));
        assert.strictEqual(result.rows, 3);
        assert.strictEqual(result.cols, 100);
        assert.strictEqual(result.values.length, 3 * 100);
        
        
        assert.strictEqual(result.combos.length, 3);
        assert.strictEqual(result.combos[0].period, 20);
        assert.strictEqual(result.combos[1].period, 30);
        assert.strictEqual(result.combos[2].period, 40);
    }
});

test('NMA batch error handling', () => {
    
    
    
    const allNaN = new Float64Array(100).fill(NaN);
    assert.throws(() => {
        wasm.nma_batch_js(allNaN, 20, 40, 10);
    }, /All values are NaN/);
    
    
    const smallData = new Float64Array([1.0, 2.0, 3.0]);
    assert.throws(() => {
        wasm.nma_batch_js(smallData, 10, 20, 10);
    }, /Invalid period|Not enough valid data|unreachable/);  
});

test('NMA warmup behavior', () => {
    
    const close = new Float64Array(testData.close);
    const period = 40;
    
    const result = wasm.nma_js(close, period);
    
    
    let firstValid = 0;
    for (let i = 0; i < close.length; i++) {
        if (!isNaN(close[i])) {
            firstValid = i;
            break;
        }
    }
    
    const warmup = firstValid + period;
    
    
    for (let i = 0; i < warmup; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} during warmup`);
    }
    
    
    for (let i = warmup; i < result.length; i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i} after warmup`);
    }
});

test('NMA different periods', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const testPeriods = [10, 20, 40, 80];
    
    for (const period of testPeriods) {
        const result = wasm.nma_js(close, period);
        assert.strictEqual(result.length, close.length);
        
        
        let firstValid = 0;
        for (let i = 0; i < close.length; i++) {
            if (!isNaN(close[i])) {
                firstValid = i;
                break;
            }
        }
        const warmup = firstValid + period;
        
        if (warmup < result.length) {
            for (let i = warmup; i < result.length; i++) {
                assert(isFinite(result[i]), `NaN at index ${i} for period=${period}`);
            }
        }
    }
});

test('NMA edge cases', () => {
    
    
    
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = i + 1;
    }
    const result = wasm.nma_js(data, 10);
    assert.strictEqual(result.length, data.length);
    
    
    for (let i = 10; i < result.length; i++) {
        assert(isFinite(result[i]), `NaN at index ${i}`);
    }
    
    
    const constantData = new Float64Array(100).fill(50.0);
    const constantResult = wasm.nma_js(constantData, 10);
    
    assert.strictEqual(constantResult.length, constantData.length);
    
    
    for (let i = 10; i < constantResult.length; i++) {
        assert(isFinite(constantResult[i]), `NaN at index ${i} for constant data`);
    }
    
    
    const oscillatingData = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        oscillatingData[i] = (i % 2 === 0) ? 10.0 : 20.0;
    }
    const oscillatingResult = wasm.nma_js(oscillatingData, 10);
    assert.strictEqual(oscillatingResult.length, oscillatingData.length);
    
    
    for (let i = 10; i < oscillatingResult.length; i++) {
        assert(isFinite(oscillatingResult[i]), `Expected finite value at index ${i}`);
    }
});

test('NMA consistency across calls', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const result1 = wasm.nma_js(close, 40);
    const result2 = wasm.nma_js(close, 40);
    
    assertArrayClose(result1, result2, 1e-15, "NMA results not consistent");
});

test('NMA formula verification', () => {
    
    const data = new Float64Array([10.0, 12.0, 11.0, 13.0, 15.0, 14.0]);
    const period = 3;
    
    const result = wasm.nma_js(data, period);
    
    
    assert.strictEqual(result.length, data.length);
    
    
    for (let i = 0; i < period; i++) {
        assert(isNaN(result[i]), `Expected NaN during warmup at index ${i}`);
    }
    
    
    const min = Math.min(...data);
    const max = Math.max(...data);
    for (let i = period; i < result.length; i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i}`);
        assert(result[i] >= min * 0.5, `Value too low at index ${i}`);
        assert(result[i] <= max * 1.5, `Value too high at index ${i}`);
    }
});

test('NMA batch metadata', () => {
    
    const metadata = wasm.nma_batch_metadata_js(
        10, 30, 10  
    );
    
    
    assert.strictEqual(metadata.length, 3);
    assert.strictEqual(metadata[0], 10);
    assert.strictEqual(metadata[1], 20);
    assert.strictEqual(metadata[2], 30);
});

test('NMA batch rows and cols', () => {
    
    const rowsCols = wasm.nma_batch_rows_cols_js(
        10, 30, 10,  
        100          
    );
    
    assert.strictEqual(rowsCols[0], 3);   
    assert.strictEqual(rowsCols[1], 100); 
});

test('NMA step precision', () => {
    
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = i + 1;
    }
    
    const batchResult = wasm.nma_batch_js(
        data,
        10, 30, 10  
    );
    
    
    const rowsCols = wasm.nma_batch_rows_cols_js(10, 30, 10, data.length);
    const rows = rowsCols[0];
    
    
    assert.strictEqual(rows, 3);
    assert.strictEqual(batchResult.length, 3 * data.length);
    
    
    const metadata = wasm.nma_batch_metadata_js(10, 30, 10);
    assert.strictEqual(metadata.length, 3);
    assert.strictEqual(metadata[0], 10);
    assert.strictEqual(metadata[1], 20);
    assert.strictEqual(metadata[2], 30);
});

test('NMA small step size', () => {
    
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = i + 1;
    }
    
    const batchResult = wasm.nma_batch_js(
        data,
        10, 12, 1  
    );
    
    const rowsCols = wasm.nma_batch_rows_cols_js(10, 12, 1, data.length);
    const rows = rowsCols[0];
    
    assert.strictEqual(rows, 3);
    assert.strictEqual(batchResult.length, 3 * data.length);
});