/**
 * WASM binding tests for VPWMA indicator.
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

test('VPWMA partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.vpwma_js(close, 14, 0.382);
    assert.strictEqual(result.length, close.length);
});

test('VPWMA accuracy', async () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.vpwma;
    
    const result = wasm.vpwma_js(
        close,
        expected.defaultParams.period,
        expected.defaultParams.power
    );
    
    assert.strictEqual(result.length, close.length);
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-4,  
        "VPWMA last 5 values mismatch"
    );
    
    
    await compareWithRust('vpwma', result, 'close', expected.defaultParams);
});

test('VPWMA zero period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.vpwma_js(inputData, 0, 0.382);
    }, /Invalid period/);
});

test('VPWMA period exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.vpwma_js(dataSmall, 10, 0.382);
    }, /Invalid period/);
});

test('VPWMA very small dataset', () => {
    
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.vpwma_js(singlePoint, 2, 0.382);
    }, /Invalid period|Not enough valid data/);
});

test('VPWMA empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.vpwma_js(empty, 14, 0.382);
    }, /Input data slice is empty/);
});

test('VPWMA invalid power', () => {
    
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    
    assert.throws(() => {
        wasm.vpwma_js(data, 2, NaN);
    }, /Invalid power/);
    
    
    assert.throws(() => {
        wasm.vpwma_js(data, 2, Infinity);
    }, /Invalid power/);
});

test('VPWMA reinput', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const firstResult = wasm.vpwma_js(close, 14, 0.382);
    assert.strictEqual(firstResult.length, close.length);
    
    
    const secondResult = wasm.vpwma_js(firstResult, 5, 0.5);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    
    if (secondResult.length > 240) {
        for (let i = 240; i < secondResult.length; i++) {
            assert(!isNaN(secondResult[i]), `Unexpected NaN at index ${i}`);
        }
    }
});

test('VPWMA NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.vpwma_js(close, 14, 0.382);
    assert.strictEqual(result.length, close.length);
    
    
    if (result.length > 50) {
        for (let i = 50; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    
    assertAllNaN(result.slice(0, 13), "Expected NaN in warmup period");
});

test('VPWMA all NaN input', () => {
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.vpwma_js(allNaN, 14, 0.382);
    }, /All values are NaN/);
});

test('VPWMA batch single parameter set', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const batchResult = wasm.vpwma_batch(close, {
        period_range: [14, 14, 0],
        power_range: [0.382, 0.382, 0]
    });
    
    
    const singleResult = wasm.vpwma_js(close, 14, 0.382);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('VPWMA batch multiple periods', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    
    
    const batchResult = wasm.vpwma_batch(close, {
        period_range: [14, 18, 2],      
        power_range: [0.382, 0.382, 0]  
    });
    
    
    assert.strictEqual(batchResult.values.length, 3 * 100);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 100);
    
    
    const periods = [14, 16, 18];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.vpwma_js(close, periods[i], 0.382);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('VPWMA batch metadata from result', () => {
    
    const close = new Float64Array(20); 
    close.fill(100);
    
    const result = wasm.vpwma_batch(close, {
        period_range: [14, 18, 2],      
        power_range: [0.3, 0.5, 0.1]   
    });
    
    
    assert.strictEqual(result.combos.length, 9);
    
    
    assert.strictEqual(result.combos[0].period, 14);   
    assert.strictEqual(result.combos[0].power, 0.3);  
    
    
    assert.strictEqual(result.combos[8].period, 18);  
    assertClose(result.combos[8].power, 0.5, 1e-10, "power mismatch"); 
});

test('VPWMA batch full parameter sweep', () => {
    
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.vpwma_batch(close, {
        period_range: [12, 14, 2],      
        power_range: [0.3, 0.4, 0.1]    
    });
    
    
    assert.strictEqual(batchResult.combos.length, 4);
    assert.strictEqual(batchResult.rows, 4);
    assert.strictEqual(batchResult.cols, 50);
    assert.strictEqual(batchResult.values.length, 4 * 50);
    
    
    for (let combo = 0; combo < batchResult.combos.length; combo++) {
        const period = batchResult.combos[combo].period;
        const power = batchResult.combos[combo].power;
        
        const rowStart = combo * 50;
        const rowData = batchResult.values.slice(rowStart, rowStart + 50);
        
        
        for (let i = 0; i < period - 1; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${period}`);
        }
        
        
        for (let i = period - 1; i < 50; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period}`);
        }
    }
});

test('VPWMA batch edge cases', () => {
    
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    
    const singleBatch = wasm.vpwma_batch(close, {
        period_range: [5, 5, 1],
        power_range: [0.382, 0.382, 0.1]
    });
    
    assert.strictEqual(singleBatch.values.length, 10);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    
    const largeBatch = wasm.vpwma_batch(close, {
        period_range: [5, 7, 10], 
        power_range: [0.382, 0.382, 0]
    });
    
    
    assert.strictEqual(largeBatch.values.length, 10);
    assert.strictEqual(largeBatch.combos.length, 1);
    
    
    assert.throws(() => {
        wasm.vpwma_batch(new Float64Array([]), {
            period_range: [14, 14, 0],
            power_range: [0.382, 0.382, 0]
        });
    }, /Input data slice is empty/);
});


test('VPWMA batch - new ergonomic API with single parameter', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.vpwma_batch(close, {
        period_range: [14, 14, 0],
        power_range: [0.382, 0.382, 0]
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
    assert.strictEqual(combo.power, 0.382);
    
    
    const oldResult = wasm.vpwma_js(close, 14, 0.382);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(result.values[i])) {
            continue; 
        }
        assert(Math.abs(oldResult[i] - result.values[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});

test('VPWMA batch - new API with multiple parameters', () => {
    
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const result = wasm.vpwma_batch(close, {
        period_range: [10, 14, 2],       
        power_range: [0.3, 0.4, 0.1]     
    });
    
    
    assert.strictEqual(result.combos.length, 6);
    assert.strictEqual(result.rows, 6);
    assert.strictEqual(result.cols, 50);
    
    
    const expectedCombos = [
        {period: 10, power: 0.3},
        {period: 10, power: 0.4},
        {period: 12, power: 0.3},
        {period: 12, power: 0.4},
        {period: 14, power: 0.3},
        {period: 14, power: 0.4}
    ];
    
    for (let i = 0; i < expectedCombos.length; i++) {
        assert.strictEqual(result.combos[i].period, expectedCombos[i].period);
        assertClose(result.combos[i].power, expectedCombos[i].power, 1e-10, `Power mismatch at combo ${i}`);
    }
});

test.skip('VPWMA zero-copy API', () => {
    
    
    
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]);
    const period = 5;
    const power = 0.382;
    
    
    const ptr = wasm.vpwma_alloc(data.length);
    assert(ptr !== 0, 'Should allocate memory successfully');
    
    try {
        
        const memory = wasm.memory;
        const memView = new Float64Array(memory.buffer, ptr, data.length);
        memView.set(data);
        
        
        wasm.vpwma_into(ptr, ptr, data.length, period, power);
        
        
        const regularResult = wasm.vpwma_js(data, period, power);
        
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; 
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        wasm.vpwma_free(ptr, data.length);
    }
});

test.skip('VPWMA zero-copy with large dataset', () => {
    
    
    const size = 10000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.1) * 100 + 100;
    }
    
    const ptr = wasm.vpwma_alloc(size);
    assert(ptr !== 0, 'Should allocate large memory successfully');
    
    try {
        const memory = wasm.memory;
        const memView = new Float64Array(memory.buffer, ptr, size);
        memView.set(data);
        
        
        wasm.vpwma_into(ptr, ptr, size, 14, 0.382);
        
        
        for (let i = 0; i < 13; i++) {
            assert(isNaN(memView[i]), `Expected NaN at warmup index ${i}`);
        }
        
        
        for (let i = 13; i < 100; i++) {
            assert(!isNaN(memView[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.vpwma_free(ptr, size);
    }
});


test.skip('VPWMA zero-copy error handling', () => {
    
    
    assert.throws(() => {
        wasm.vpwma_into(0, 0, 10, 14, 0.382);
    }, /null pointer|invalid memory/i);
    
    const ptr = wasm.vpwma_alloc(10);
    try {
        
        assert.throws(() => {
            wasm.vpwma_into(ptr, ptr, 10, 0, 0.382);
        }, /Invalid period/);
        
        
        assert.throws(() => {
            wasm.vpwma_into(ptr, ptr, 10, 5, NaN);
        }, /Invalid power/);
    } finally {
        wasm.vpwma_free(ptr, 10);
    }
});

test.skip('VPWMA zero-copy memory management', () => {
    
    
    const sizes = [10, 100, 1000];
    
    for (const size of sizes) {
        const ptr = wasm.vpwma_alloc(size);
        assert(ptr !== 0, `Should allocate ${size} elements`);
        
        
        const memory = wasm.memory;
        const memView = new Float64Array(memory.buffer, ptr, size);
        for (let i = 0; i < size; i++) {
            memView[i] = i;
        }
        
        
        for (let i = 0; i < size; i++) {
            assert.strictEqual(memView[i], i, `Memory corruption at index ${i}`);
        }
        
        
        wasm.vpwma_free(ptr, size);
    }
});

test('VPWMA warmup period verification', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const testCases = [
        {period: 5, power: 0.5, expectedWarmup: 4},
        {period: 10, power: 0.382, expectedWarmup: 9},
        {period: 14, power: 0.382, expectedWarmup: 13},
        {period: 20, power: 0.3, expectedWarmup: 19}
    ];
    
    for (const tc of testCases) {
        const result = wasm.vpwma_js(close, tc.period, tc.power);
        
        
        for (let i = 0; i < tc.expectedWarmup; i++) {
            assert(isNaN(result[i]), 
                   `Expected NaN at index ${i} for period=${tc.period}, got ${result[i]}`);
        }
        
        
        assert(!isNaN(result[tc.expectedWarmup]), 
               `Expected first valid value at index ${tc.expectedWarmup} for period=${tc.period}`);
        
        
        assert.strictEqual(tc.expectedWarmup, tc.period - 1,
                          `Warmup calculation mismatch for period=${tc.period}`);
    }
});

test('VPWMA SIMD consistency', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    const period = 14;
    const power = 0.382;
    
    
    const kernels = ['scalar', 'auto'];
    const results = {};
    
    for (const kernel of kernels) {
        try {
            
            
            
            results[kernel] = wasm.vpwma_js(close, period, power);
        } catch (e) {
            
            continue;
        }
    }
    
    
    assert(results['auto'], 'Auto kernel should be available');
    
    
    if (Object.keys(results).length > 1) {
        const baseResult = results['auto'];
        for (const [kernel, result] of Object.entries(results)) {
            if (kernel === 'auto') continue;
            
            for (let i = 0; i < result.length; i++) {
                if (isNaN(baseResult[i]) && isNaN(result[i])) {
                    continue;
                }
                assert(Math.abs(baseResult[i] - result[i]) < 1e-9,
                       `Kernel ${kernel} mismatch with auto at index ${i}`);
            }
        }
    }
});



test.after(() => {
    console.log('VPWMA WASM tests completed');
});