/**
 * WASM binding tests for CG (Center of Gravity) indicator.
 * These tests mirror the Rust unit tests to ensure WASM bindings work correctly.
 * 
 * Warmup Period: CG requires period + 1 valid data points.
 * Output starts at index: first_valid + period
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

test('CG partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.cg_js(close, 12);
    assert.strictEqual(result.length, close.length);
});

test('CG accuracy', async () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.cg;
    
    const result = wasm.cg_js(
        close,
        expected.defaultParams.period
    );
    
    assert.strictEqual(result.length, close.length);
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-4,  
        "CG last 5 values mismatch"
    );
    
    
    await compareWithRust('cg', result, 'close', expected.defaultParams);
});

test('CG default candles', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.cg_js(close, 10);
    assert.strictEqual(result.length, close.length);
});

test('CG zero period', () => {
    
    const data = new Float64Array([1.0, 2.0, 3.0]);
    
    assert.throws(() => {
        wasm.cg_js(data, 0);
    }, /Invalid period/);
});

test('CG period exceeds length', () => {
    
    const data = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.cg_js(data, 10);
    }, /Invalid period/);
});

test('CG very small dataset', () => {
    
    const data = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.cg_js(data, 10);
    }, /Invalid period|Not enough valid data/);
});

test('CG empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.cg_js(empty, 10);
    }, /Empty data/);
});

test('CG NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.cg_js(close, 10);
    assert.strictEqual(result.length, close.length);
    
    
    const checkIdx = 240;
    if (result.length > checkIdx) {
        
        let foundValid = false;
        for (let i = checkIdx; i < result.length; i++) {
            if (!isNaN(result[i])) {
                foundValid = true;
                break;
            }
        }
        assert(foundValid, `All CG values from index ${checkIdx} onward are NaN.`);
    }
    
    
    assertAllNaN(result.slice(0, 10), "Expected NaN in warmup period");
});

test('CG all NaN input', () => {
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.cg_js(allNaN, 10);
    }, /All values are NaN/);
});


test('CG batch full parameter sweep', () => {
    
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const result = wasm.cg_batch(close, {
        period_range: [10, 14, 2]  
    });
    
    
    assert.strictEqual(result.rows, 3);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.values.length, 3 * 50);
    
    
    for (let combo = 0; combo < result.rows; combo++) {
        const period = result.combos[combo].period;
        
        const rowStart = combo * 50;
        const rowData = result.values.slice(rowStart, rowStart + 50);
        
        
        for (let i = 0; i < period; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${period}`);
        }
        
        
        for (let i = period; i < 50; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period}`);
        }
    }
});

test('CG batch edge cases', () => {
    
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    
    
    const singleResult = wasm.cg_batch(close, {
        period_range: [5, 5, 1]
    });
    
    assert.strictEqual(singleResult.rows, 1);
    assert.strictEqual(singleResult.values.length, 12);
    
    
    const largeResult = wasm.cg_batch(close, {
        period_range: [5, 7, 10] 
    });
    
    
    assert.strictEqual(largeResult.rows, 1);
    assert.strictEqual(largeResult.values.length, 12);
    
    
    assert.throws(() => {
        wasm.cg_batch(new Float64Array([]), {
            period_range: [10, 10, 0]
        });
    }, /All values are NaN/);
});


test('CG batch - new ergonomic API with single parameter', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.cg_batch(close, {
        period_range: [10, 10, 0]
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
    assert.strictEqual(combo.period, 10);
    
    
    const oldResult = wasm.cg_js(close, 10);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(result.values[i])) {
            continue; 
        }
        assert(Math.abs(oldResult[i] - result.values[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});

test('CG batch - new API with multiple parameters', () => {
    
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const result = wasm.cg_batch(close, {
        period_range: [10, 14, 2]      
    });
    
    
    assert.strictEqual(result.rows, 3);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.combos.length, 3);
    assert.strictEqual(result.values.length, 150);
    
    
    const expectedPeriods = [10, 12, 14];
    
    for (let i = 0; i < expectedPeriods.length; i++) {
        assert.strictEqual(result.combos[i].period, expectedPeriods[i]);
    }
    
    
    const secondRow = result.values.slice(result.cols, 2 * result.cols);
    assert.strictEqual(secondRow.length, 50);
    
    
    const oldResult = wasm.cg_js(close, 10);
    const firstRow = result.values.slice(0, result.cols);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(firstRow[i])) {
            continue; 
        }
        assert(Math.abs(oldResult[i] - firstRow[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});


test('CG batch - new API error handling', () => {
    const close = new Float64Array(testData.close.slice(0, 10));
    
    
    assert.throws(() => {
        wasm.cg_batch(close, {
            period_range: [10, 10], 
        });
    }, /Invalid config/);
    
    
    assert.throws(() => {
        wasm.cg_batch(close, {
            
        });
    }, /Invalid config/);
    
    
    assert.throws(() => {
        wasm.cg_batch(close, {
            period_range: "invalid"
        });
    }, /Invalid config/);
});

test('CG warmup period', () => {
    
    const close = new Float64Array(testData.close);
    const period = 10;
    
    const result = wasm.cg_js(close, period);
    
    
    
    assertAllNaN(result.slice(0, period), `Expected NaN in first ${period} values`);
    
    
    if (close.length > period) {
        assert(!isNaN(result[period]), `Expected valid value at index ${period}`);
    }
});

test('CG edge case small period', () => {
    
    const close = new Float64Array(testData.close.slice(0, 20));
    
    
    const result = wasm.cg_js(close, 2);
    assert.strictEqual(result.length, close.length);
    
    
    assertAllNaN(result.slice(0, 2));
    
    assert(!isNaN(result[2]));
});

test('CG NaN injection at specific positions', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    
    
    for (let i = 20; i < 25; i++) {
        close[i] = NaN;
    }
    
    const result = wasm.cg_js(close, 10);
    assert.strictEqual(result.length, close.length);
    
    
    
    
    assertAllNaN(result.slice(0, 10));
    
    
    
    let hasNonNaNAfterInjection = false;
    for (let i = 30; i < result.length; i++) {
        if (!isNaN(result[i]) && result[i] !== 0.0) {
            hasNonNaNAfterInjection = true;
            break;
        }
    }
    assert(hasNonNaNAfterInjection, "Expected non-zero values after NaN injection");
});

test('CG batch accuracy verification', () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.cg;
    
    
    const result = wasm.cg_batch(close, {
        period_range: [10, 10, 0]
    });
    
    
    const last5 = result.values.slice(result.cols - 5, result.cols);
    
    
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-4,
        "CG batch accuracy mismatch"
    );
});

test('CG batch comprehensive parameter sweep', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const result = wasm.cg_batch(close, {
        period_range: [5, 20, 5]  
    });
    
    
    assert.strictEqual(result.rows, 4);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.combos.length, 4);
    
    
    const expectedPeriods = [5, 10, 15, 20];
    for (let i = 0; i < expectedPeriods.length; i++) {
        assert.strictEqual(result.combos[i].period, expectedPeriods[i]);
    }
    
    
    for (let i = 0; i < result.rows; i++) {
        const period = result.combos[i].period;
        const rowStart = i * result.cols;
        const rowData = result.values.slice(rowStart, rowStart + result.cols);
        
        
        assertAllNaN(rowData.slice(0, period));
        
        
        if (close.length > period) {
            assert(!isNaN(rowData[period]));
        }
    }
});

test('CG numerical stability with extreme values', () => {
    
    const largeData = new Float64Array([1e15, 1e15, 1e15, 1e15, 1e15, 1e15]);
    const resultLarge = wasm.cg_js(largeData, 2);
    assert.strictEqual(resultLarge.length, largeData.length);
    
    assert(!isNaN(resultLarge[2]));  
    
    
    const smallData = new Float64Array([1e-15, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15]);
    const resultSmall = wasm.cg_js(smallData, 2);
    assert.strictEqual(resultSmall.length, smallData.length);
    assert(!isNaN(resultSmall[2]));  
});

test('CG full dataset test', () => {
    
    const close = new Float64Array(testData.close);  
    
    const result = wasm.cg_js(close, 10);
    assert.strictEqual(result.length, close.length);
    
    
    assertAllNaN(result.slice(0, 10));
    
    
    for (let i = 10; i < result.length; i++) {
        assert(!isNaN(result[i]), `Unexpected NaN at index ${i}`);
    }
});

test('CG batch with invalid parameters', () => {
    const close = new Float64Array(testData.close.slice(0, 10));
    
    
    
    assert.throws(() => {
        wasm.cg_batch(close, {
            period_range: [15, 20, 5]  
        });
    }, /Not enough valid data|Invalid period/);
});

test('CG batch memory validation', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    
    
    for (let iteration = 0; iteration < 3; iteration++) {
        const result = wasm.cg_batch(close, {
            period_range: [5, 15, 5]
        });
        
        
        assert.strictEqual(result.rows, 3);
        assert.strictEqual(result.cols, 100);
        
        
        for (let i = 0; i < result.values.length; i++) {
            const val = result.values[i];
            if (!isNaN(val)) {
                
                assert(Math.abs(val) < 1e100, `Unreasonable value at index ${i}: ${val}`);
            }
        }
    }
});



test.after(() => {
    console.log('CG WASM tests completed');
});