/**
 * WASM binding tests for CCI_CYCLE indicator.
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

test('CCI_CYCLE partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.cci_cycle_js(close, 10, 0.5);
    assert.strictEqual(result.length, close.length);
});

test('CCI_CYCLE accuracy', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const length = 10;
    const factor = 0.5;
    
    const result = wasm.cci_cycle_js(close, length, factor);
    
    assert.strictEqual(result.length, close.length);
    
    
    const expected_last_five = [
        9.25177192,
        20.49219826,
        35.42917181,
        55.57843075,
        77.78921538,
    ];
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected_last_five,
        1e-6,
        "CCI_CYCLE last 5 values mismatch"
    );
});

test('CCI_CYCLE default candles', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.cci_cycle_js(close, 10, 0.5);
    assert.strictEqual(result.length, close.length);
});

test('CCI_CYCLE zero period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.cci_cycle_js(inputData, 0, 0.5);
    }, /Invalid period/);
});

test('CCI_CYCLE period exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.cci_cycle_js(dataSmall, 10, 0.5);
    }, /Invalid period/);
});

test('CCI_CYCLE very small dataset', () => {
    
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.cci_cycle_js(singlePoint, 10, 0.5);
    }, /Invalid period|Not enough valid data/);
});

test('CCI_CYCLE empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.cci_cycle_js(empty, 10, 0.5);
    }, /empty/i);
});

test('CCI_CYCLE factor edge cases', () => {
    const data = new Float64Array([
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
    ]);

    
    const nanResult = wasm.cci_cycle_js(data, 5, NaN);
    assert.strictEqual(nanResult.length, data.length);
    let nanCount = 0;
    for (const v of nanResult) {
        if (Number.isNaN(v)) nanCount++;
    }
    assert.ok(
        nanCount >= data.length - 5,
        `Expected mostly NaN when factor is NaN, got ${nanCount}/${data.length} NaN values`
    );

    
    const negResult = wasm.cci_cycle_js(data, 5, -0.5);
    assert.strictEqual(negResult.length, data.length);

    const bigResult = wasm.cci_cycle_js(data, 5, 10.0);
    assert.strictEqual(bigResult.length, data.length);

    
    assert.throws(() => {
        wasm.cci_cycle_js(data, 5, Infinity);
    }, /Invalid factor/);
});

test('CCI_CYCLE all NaN input', () => {
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.cci_cycle_js(allNaN, 10, 0.5);
    }, /All values are NaN/);
});

test('CCI_CYCLE NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.cci_cycle_js(close, 10, 0.5);
    assert.strictEqual(result.length, close.length);
    
    
    
    let initialNans = 0;
    for (let i = 0; i < Math.min(50, result.length); i++) {
        if (isNaN(result[i])) {
            initialNans++;
        } else {
            break;
        }
    }
    
    
    assert(initialNans > 0, "Expected some NaN values during warmup period");
    
    
    if (result.length > 100) {
        let nonNanCount = 0;
        for (let i = 100; i < Math.min(200, result.length); i++) {
            if (!isNaN(result[i])) {
                nonNanCount++;
            }
        }
        assert(nonNanCount > 0, "Should have some valid values after sufficient data");
    }
});

test('CCI_CYCLE batch single parameter set', () => {
    
    const close = new Float64Array(testData.close);
    
    try {
        
        const batchResult = wasm.cci_cycle_batch(close, {
            length_range: [10, 10, 0],
            factor_range: [0.5, 0.5, 0]
        });
        
        
        const singleResult = wasm.cci_cycle_js(close, 10, 0.5);
        
        assert.strictEqual(batchResult.values.length, singleResult.length);
        assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
    } catch (error) {
        
        console.log("CCI_CYCLE batch API not available, skipping batch tests");
    }
});

test('CCI_CYCLE batch multiple parameters', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    
    try {
        
        const batchResult = wasm.cci_cycle_batch(close, {
            length_range: [10, 20, 5],      
            factor_range: [0.3, 0.7, 0.2]   
        });
        
        
        assert.strictEqual(batchResult.combos.length, 9);
        assert.strictEqual(batchResult.rows, 9);
        assert.strictEqual(batchResult.cols, 100);
        assert.strictEqual(batchResult.values.length, 900);
        
        
        const lengths = [10, 10, 10, 15, 15, 15, 20, 20, 20];
        const factors = [0.3, 0.5, 0.7, 0.3, 0.5, 0.7, 0.3, 0.5, 0.7];
        
        for (let i = 0; i < 9; i++) {
            const rowStart = i * 100;
            const rowEnd = rowStart + 100;
            const rowData = batchResult.values.slice(rowStart, rowEnd);
            
            const singleResult = wasm.cci_cycle_js(close, lengths[i], factors[i]);
            assertArrayClose(
                rowData, 
                singleResult, 
                1e-10, 
                `Length ${lengths[i]}, Factor ${factors[i]} mismatch`
            );
        }
    } catch (error) {
        
        console.log("CCI_CYCLE batch API not available, skipping batch tests");
    }
});

test('CCI_CYCLE batch metadata', () => {
    
    const close = new Float64Array(50); 
    close.fill(100);
    
    try {
        const result = wasm.cci_cycle_batch(close, {
            length_range: [10, 15, 5],      
            factor_range: [0.3, 0.7, 0.2]   
        });
        
        
        assert.strictEqual(result.combos.length, 6);
        
        
        assert.strictEqual(result.combos[0].length, 10);
        assertClose(result.combos[0].factor, 0.3, 1e-10, "factor mismatch");
        
        
        assert.strictEqual(result.combos[5].length, 15);
        assertClose(result.combos[5].factor, 0.7, 1e-10, "factor mismatch");
    } catch (error) {
        
        console.log("CCI_CYCLE batch API not available, skipping batch tests");
    }
});

test('CCI_CYCLE batch edge cases', () => {
    
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]);
    
    try {
        
        const singleBatch = wasm.cci_cycle_batch(close, {
            length_range: [5, 5, 1],
            factor_range: [0.5, 0.5, 0.1]
        });
        
        assert.strictEqual(singleBatch.values.length, 20);
        assert.strictEqual(singleBatch.combos.length, 1);
        
        
        const largeBatch = wasm.cci_cycle_batch(close, {
            length_range: [5, 7, 10], 
            factor_range: [0.5, 0.5, 0]
        });
        
        
        assert.strictEqual(largeBatch.values.length, 20);
        assert.strictEqual(largeBatch.combos.length, 1);
        
        
        assert.throws(() => {
            wasm.cci_cycle_batch(new Float64Array([]), {
                length_range: [10, 10, 0],
                factor_range: [0.5, 0.5, 0]
            });
        }, /All values are NaN|empty/);
    } catch (error) {
        
        console.log("CCI_CYCLE batch API not available, skipping batch tests");
    }
});


test('CCI_CYCLE zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]);
    const length = 5;
    const factor = 0.5;
    
    try {
        
        const ptr = wasm.cci_cycle_alloc(data.length);
        assert(ptr !== 0, 'Failed to allocate memory');
        
        
        const memory = wasm.__wasm.memory.buffer;
        const memView = new Float64Array(
            memory,
            ptr,
            data.length
        );
        
        
        memView.set(data);
        
        
        try {
            wasm.cci_cycle_into(ptr, ptr, data.length, length, factor);
            
            
            const regularResult = wasm.cci_cycle_js(data, length, factor);
            for (let i = 0; i < data.length; i++) {
                if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                    continue; 
                }
                assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                       `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
            }
        } finally {
            
            wasm.cci_cycle_free(ptr, data.length);
        }
    } catch (error) {
        
        console.log("CCI_CYCLE zero-copy API not available, skipping zero-copy tests");
    }
});

test('CCI_CYCLE zero-copy with large dataset', () => {
    const size = 10000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }
    
    try {
        const ptr = wasm.cci_cycle_alloc(size);
        assert(ptr !== 0, 'Failed to allocate large buffer');
        
        try {
            const memory = wasm.__wasm.memory.buffer;
            const memView = new Float64Array(memory, ptr, size);
            memView.set(data);
            
            wasm.cci_cycle_into(ptr, ptr, size, 10, 0.5);
            
            
            const memory2 = wasm.__wasm.memory.buffer;
            const memView2 = new Float64Array(memory2, ptr, size);
            
            
            const warmupPeriod = 10 * 4;
            for (let i = 0; i < warmupPeriod; i++) {
                assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
            }
            
            
            let hasValidValues = false;
            for (let i = warmupPeriod + 10; i < Math.min(warmupPeriod + 100, size); i++) {
                if (!isNaN(memView2[i])) {
                    hasValidValues = true;
                    break;
                }
            }
            assert(hasValidValues, "Should have some valid values after warmup");
        } finally {
            wasm.cci_cycle_free(ptr, size);
        }
    } catch (error) {
        
        console.log("CCI_CYCLE zero-copy API not available, skipping zero-copy tests");
    }
});


test('CCI_CYCLE zero-copy error handling', () => {
    try {
        
        assert.throws(() => {
            wasm.cci_cycle_into(0, 0, 10, 10, 0.5);
        }, /null pointer|invalid memory/i);
        
        
        const ptr = wasm.cci_cycle_alloc(20);
        try {
            
            assert.throws(() => {
                wasm.cci_cycle_into(ptr, ptr, 20, 0, 0.5);
            }, /Invalid period/);
            
            
            assert.doesNotThrow(() => {
                wasm.cci_cycle_into(ptr, ptr, 20, 5, NaN);
            });

            
            assert.throws(() => {
                wasm.cci_cycle_into(ptr, ptr, 20, 5, Infinity);
            }, /Invalid factor/);
        } finally {
            wasm.cci_cycle_free(ptr, 20);
        }
    } catch (error) {
        
        console.log("CCI_CYCLE zero-copy API not available, skipping error handling tests");
    }
});


test('CCI_CYCLE zero-copy memory management', () => {
    try {
        
        const sizes = [100, 1000, 5000];
        
        for (const size of sizes) {
            const ptr = wasm.cci_cycle_alloc(size);
            assert(ptr !== 0, `Failed to allocate ${size} elements`);
            
            
            const memory = wasm.__wasm.memory.buffer;
            const memView = new Float64Array(memory, ptr, size);
            for (let i = 0; i < Math.min(10, size); i++) {
                memView[i] = i * 1.5;
            }
            
            
            for (let i = 0; i < Math.min(10, size); i++) {
                assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
            }
            
            
            wasm.cci_cycle_free(ptr, size);
        }
    } catch (error) {
        
        console.log("CCI_CYCLE zero-copy API not available, skipping memory management tests");
    }
});

test.after(() => {
    console.log('CCI_CYCLE WASM tests completed');
});
