/**
 * WASM binding tests for CFO indicator.
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

test('CFO partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.cfo_js(close, 14, 100.0);
    assert.strictEqual(result.length, close.length);
});

test('CFO accuracy', async () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.cfo;
    
    const result = wasm.cfo_js(
        close,
        expected.defaultParams.period,
        expected.defaultParams.scalar
    );
    
    assert.strictEqual(result.length, close.length);
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-6,
        "CFO last 5 values mismatch"
    );
    
    
    await compareWithRust('cfo', result, 'close', expected.defaultParams);
});

test('CFO default candles', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.cfo_js(close, 14, 100.0);
    assert.strictEqual(result.length, close.length);
});

test('CFO zero period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.cfo_js(inputData, 0, 100.0);
    }, /Invalid period/);
});

test('CFO period exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.cfo_js(dataSmall, 10, 100.0);
    }, /Invalid period/);
});

test('CFO very small dataset', () => {
    
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.cfo_js(singlePoint, 14, 100.0);
    }, /Invalid period|Not enough valid data/);
});

test('CFO empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.cfo_js(empty, 14, 100.0);
    }, /No data provided/);
});

test('CFO reinput', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const firstResult = wasm.cfo_js(close, 14, 100.0);
    assert.strictEqual(firstResult.length, close.length);
    
    
    const secondResult = wasm.cfo_js(firstResult, 14, 100.0);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    
    if (secondResult.length > 240) {
        for (let i = 240; i < secondResult.length; i++) {
            assert(!isNaN(secondResult[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('CFO NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.cfo_js(close, 14, 100.0);
    assert.strictEqual(result.length, close.length);
    
    
    let firstValid = 0;
    for (let i = 0; i < close.length; i++) {
        if (!isNaN(close[i])) {
            firstValid = i;
            break;
        }
    }
    const warmupPeriod = firstValid + 14 - 1;
    
    
    assertAllNaN(result.slice(0, warmupPeriod), `Expected NaN in warmup period [0:${warmupPeriod})`);
    
    
    if (result.length > warmupPeriod) {
        let hasValidValue = false;
        for (let i = warmupPeriod; i < result.length; i++) {
            if (!isNaN(result[i])) {
                hasValidValue = true;
                break;
            }
        }
        assert(hasValidValue, "Expected some valid values after warmup");
    }
});

test('CFO all NaN input', () => {
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.cfo_js(allNaN, 14, 100.0);
    }, /All values are NaN/);
});

test('CFO batch single parameter set', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const batchResult = wasm.cfo_batch_js(
        close,
        14, 14, 0,      
        100.0, 100.0, 0 
    );
    
    
    const singleResult = wasm.cfo_js(close, 14, 100.0);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    
    assertArrayClose(batchResult, singleResult, 1e-7, "Batch vs single mismatch");
});

test('CFO batch multiple periods', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    
    
    const batchResult = wasm.cfo_batch_js(
        close,
        10, 14, 2,      
        100.0, 100.0, 0 
    );
    
    
    assert.strictEqual(batchResult.length, 3 * 100);
    
    
    const periods = [10, 12, 14];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.slice(rowStart, rowEnd);
        
        const singleResult = wasm.cfo_js(close, periods[i], 100.0);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('CFO batch metadata', () => {
    
    const metadata = wasm.cfo_batch_metadata_js(
        10, 14, 2,      
        50.0, 150.0, 50.0 
    );
    
    
    
    assert.strictEqual(metadata.length, 9 * 2);
    
    
    assert.strictEqual(metadata[0], 10);    
    assert.strictEqual(metadata[1], 50.0);  
    
    
    assert.strictEqual(metadata[16], 14);   
    assert.strictEqual(metadata[17], 150.0); 
});

test('CFO batch full parameter sweep', () => {
    
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.cfo_batch_js(
        close,
        10, 12, 2,      
        50.0, 100.0, 50.0  
    );
    
    const metadata = wasm.cfo_batch_metadata_js(
        10, 12, 2,
        50.0, 100.0, 50.0
    );
    
    
    assert.strictEqual(batchResult.length, 4 * 50);
    assert.strictEqual(metadata.length, 4 * 2);
});

test('CFO batch unified API', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const config = {
        period_range: [10, 14, 2],
        scalar_range: [50.0, 150.0, 50.0]
    };
    
    const result = wasm.cfo_batch(close, config);
    
    
    assert(result.values, "Missing values field");
    assert(result.combos, "Missing combos field");
    assert(result.rows, "Missing rows field");
    assert(result.cols, "Missing cols field");
    
    
    assert.strictEqual(result.rows, 9);
    assert.strictEqual(result.cols, 100);
    assert.strictEqual(result.values.length, 9 * 100);
    assert.strictEqual(result.combos.length, 9);
    
    
    result.combos.forEach(combo => {
        assert(combo.hasOwnProperty('period'), "Combo missing period");
        assert(combo.hasOwnProperty('scalar'), "Combo missing scalar");
    });
});

test('CFO constant values', () => {
    
    const constant = new Float64Array(50);
    constant.fill(42.0);
    
    const result = wasm.cfo_js(constant, 14, 100.0);
    assert.strictEqual(result.length, constant.length);
    
    
    
    let foundNonNaN = false;
    for (let i = 0; i < result.length; i++) {
        if (!isNaN(result[i])) {
            foundNonNaN = true;
            assert(Math.abs(result[i]) < 1e-10, `CFO should be ~0 for constant series at index ${i}, got ${result[i]}`);
        }
    }
    assert(foundNonNaN, "Should have at least one non-NaN value");
});

test('CFO linear trend', () => {
    
    const x = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        x[i] = i;
    }
    
    const result = wasm.cfo_js(x, 14, 100.0);
    assert.strictEqual(result.length, x.length);
    
    
    
    let foundNonNaN = false;
    for (let i = 0; i < result.length; i++) {
        if (!isNaN(result[i])) {
            foundNonNaN = true;
            assert(Math.abs(result[i]) < 1e-10, `CFO should be ~0 for linear trend at index ${i}, got ${result[i]}`);
        }
    }
    assert(foundNonNaN, "Should have at least one non-NaN value");
});

test('CFO scalar edge cases', () => {
    
    const data = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0]);
    
    
    const result1 = wasm.cfo_js(data, 2, NaN);
    let allNaN = true;
    for (let i = 0; i < result1.length; i++) {
        if (!isNaN(result1[i])) {
            allNaN = false;
            break;
        }
    }
    assert(allNaN, "NaN scalar should produce all NaN output");
    
    
    const result2 = wasm.cfo_js(data, 2, Infinity);
    
    for (let i = 1; i < result2.length; i++) { 
        assert(isNaN(result2[i]) || !isFinite(result2[i]), 
               `Infinite scalar should produce inf/nan at index ${i}`);
    }
    
    
    const result3 = wasm.cfo_js(data, 2, 0.0);
    for (let i = 1; i < result3.length; i++) {
        assert.strictEqual(result3[i], 0.0, 
                          `Zero scalar should produce zero at index ${i}`);
    }
});

test('CFO negative scalar', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    
    
    const resultPos = wasm.cfo_js(close, 14, 100.0);
    const resultNeg = wasm.cfo_js(close, 14, -100.0);
    
    assert.strictEqual(resultPos.length, resultNeg.length);
    
    
    for (let i = 0; i < resultPos.length; i++) {
        if (isNaN(resultPos[i]) && isNaN(resultNeg[i])) {
            continue;
        }
        assertClose(-resultPos[i], resultNeg[i], 1e-10, 
                   `Negative scalar mismatch at index ${i}`);
    }
});

test('CFO warmup period', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const periods = [5, 10, 14, 20, 30];
    for (const period of periods) {
        const result = wasm.cfo_js(close, period, 100.0);
        
        
        let firstValid = 0;
        for (let i = 0; i < close.length; i++) {
            if (!isNaN(close[i])) {
                firstValid = i;
                break;
            }
        }
        const expectedWarmup = firstValid + period - 1;
        
        
        for (let i = 0; i < expectedWarmup && i < result.length; i++) {
            assert(isNaN(result[i]), 
                  `Period ${period}: Expected NaN at index ${i} (before warmup ${expectedWarmup})`);
        }
        
        
        if (expectedWarmup < result.length) {
            assert(!isNaN(result[expectedWarmup]), 
                  `Period ${period}: Expected valid value at index ${expectedWarmup}`);
        }
    }
});

test('CFO edge values', () => {
    
    
    
    const dataWithZero = new Float64Array([10.0, 20.0, 30.0, 0.0, 40.0, 50.0, 60.0]);
    const result1 = wasm.cfo_js(dataWithZero, 3, 100.0);
    assert(isNaN(result1[3]), "Expected NaN when current value is 0");
    
    
    const dataWithInf = new Float64Array([10.0, 20.0, 30.0, Infinity, 40.0, 50.0, 60.0]);
    const result2 = wasm.cfo_js(dataWithInf, 3, 100.0);
    assert(isNaN(result2[3]), "Expected NaN when current value is inf");
    
    
    const dataSmall = new Float64Array([1e-10, 2e-10, 3e-10, 4e-10, 5e-10]);
    const result3 = wasm.cfo_js(dataSmall, 2, 100.0);
    assert.strictEqual(result3.length, dataSmall.length);
    
    
    let hasValid = false;
    for (let i = 0; i < result3.length; i++) {
        if (!isNaN(result3[i])) {
            hasValid = true;
            break;
        }
    }
    assert(hasValid, "Should handle very small values");
});






/*
test('CFO zero-copy API', () => {
    // Test zero-copy pointer-based API
    const close = new Float64Array(testData.close.slice(0, 100));
    const len = close.length;
    
    // Allocate memory for output
    const outPtr = wasm.cfo_alloc(len);
    assert(outPtr !== 0, "Should allocate memory");
    
    try {
        // Get input pointer from TypedArray
        const inPtr = wasm.__wbindgen_malloc(len * 8);
        const mem = new Float64Array(wasm.memory.buffer, inPtr, len);
        mem.set(close);
        
        // Call zero-copy function
        wasm.cfo_into(inPtr, outPtr, len, 14, 100.0);
        
        // Read results
        const result = new Float64Array(wasm.memory.buffer, outPtr, len);
        
        // Compare with regular API
        const expected = wasm.cfo_js(close, 14, 100.0);
        assertArrayClose(result, expected, 1e-10, "Zero-copy should match regular API");
        
        // Clean up input memory
        wasm.__wbindgen_free(inPtr, len * 8);
    } finally {
        // Always free output memory
        wasm.cfo_free(outPtr, len);
    }
});
*/

/*
test('CFO zero-copy batch', () => {
    // Test zero-copy batch API
    const close = new Float64Array(testData.close.slice(0, 50));
    const len = close.length;
    const rows = 2 * 2; // 2 periods * 2 scalars
    
    // Allocate memory for output
    const outPtr = wasm.cfo_alloc(rows * len);
    assert(outPtr !== 0, "Should allocate memory for batch");
    
    try {
        // Get input pointer
        const inPtr = wasm.__wbindgen_malloc(len * 8);
        const mem = new Float64Array(wasm.memory.buffer, inPtr, len);
        mem.set(close);
        
        // Call batch zero-copy function
        const numRows = wasm.cfo_batch_into(
            inPtr, outPtr, len,
            10, 12, 2,      // period range
            50.0, 100.0, 50.0  // scalar range
        );
        
        assert.strictEqual(numRows, rows, "Should return correct number of rows");
        
        // Read results
        const result = new Float64Array(wasm.memory.buffer, outPtr, rows * len);
        
        // Compare with regular batch API
        const expected = wasm.cfo_batch_js(
            close,
            10, 12, 2,
            50.0, 100.0, 50.0
        );
        assertArrayClose(result, expected, 1e-10, "Zero-copy batch should match regular batch");
        
        // Clean up input memory
        wasm.__wbindgen_free(inPtr, len * 8);
    } finally {
        // Always free output memory
        wasm.cfo_free(outPtr, rows * len);
    }
});

test('CFO memory management', () => {
    // Test proper memory allocation and deallocation
    const sizes = [10, 100, 1000];
    
    for (const size of sizes) {
        // Allocate memory
        const ptr = wasm.cfo_alloc(size);
        assert(ptr !== 0, `Should allocate memory for size ${size}`);
        
        // Write some data to verify memory is accessible
        const mem = new Float64Array(wasm.memory.buffer, ptr, size);
        for (let i = 0; i < size; i++) {
            mem[i] = i;
        }
        
        // Verify data was written
        assert.strictEqual(mem[0], 0);
        assert.strictEqual(mem[size - 1], size - 1);
        
        // Free memory
        wasm.cfo_free(ptr, size);
        // Note: We can't really verify the memory is freed, but at least
        // make sure the free function doesn't crash
    }
});
*/

test('CFO batch error handling', () => {
    
    const close = new Float64Array(testData.close.slice(0, 50));
    
    
    assert.throws(() => {
        wasm.cfo_batch(close, { invalid: "config" });
    }, /Invalid config/);
    
    
    assert.throws(() => {
        wasm.cfo_batch(close, { period_range: [10, 12, 2] });
    }, /Invalid config/);
    
    
    assert.throws(() => {
        wasm.cfo_batch(close, {
            period_range: [0, 0, 0],  
            scalar_range: [100.0, 100.0, 0.0]
        });
    }, /Invalid period/);
});
