/**
 * WASM binding tests for SuperSmoother indicator.
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

test('SuperSmoother partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.supersmoother_js(close, 14);
    assert.strictEqual(result.length, close.length);
});

test('SuperSmoother accuracy', async () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.supersmoother;
    
    const result = wasm.supersmoother_js(
        close,
        expected.defaultParams.period
    );
    
    assert.strictEqual(result.length, close.length);
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-8,
        "SuperSmoother last 5 values mismatch"
    );
    
    
    await compareWithRust('supersmoother', result, 'close', expected.defaultParams);
});

test('SuperSmoother default candles', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.supersmoother_js(close, 14);
    assert.strictEqual(result.length, close.length);
});

test('SuperSmoother zero period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.supersmoother_js(inputData, 0);
    }, /Invalid period/);
});

test('SuperSmoother period exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.supersmoother_js(dataSmall, 10);
    }, /Invalid period/);
});

test('SuperSmoother very small dataset', () => {
    
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.supersmoother_js(singlePoint, 14);
    }, /Invalid period|Not enough valid data/);
});

test('SuperSmoother empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.supersmoother_js(empty, 14);
    }, /Empty data/);
});

test('SuperSmoother reinput', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const firstResult = wasm.supersmoother_js(close, 14);
    assert.strictEqual(firstResult.length, close.length);
    
    
    const secondResult = wasm.supersmoother_js(firstResult, 10);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    
    
});

test('SuperSmoother NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.supersmoother_js(close, 14);
    assert.strictEqual(result.length, close.length);
    
    
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    
    assertAllNaN(result.slice(0, 13), "Expected NaN in warmup period");
    
    
    assert(!isNaN(result[13]), "Value at index 13 should be initialized");
    if (result.length > 14) {
        assert(!isNaN(result[14]), "Value at index 14 should be initialized");
    }
});

test('SuperSmoother all NaN input', () => {
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.supersmoother_js(allNaN, 14);
    }, /All values are NaN/);
});

test('SuperSmoother batch single parameter', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const batchResult = wasm.supersmoother_batch_js(close, 14, 14, 0);
    
    
    const metadata = wasm.supersmoother_batch_metadata_js(14, 14, 0);
    assert.strictEqual(metadata.length, 1);
    assert.strictEqual(metadata[0], 14);
    
    
    const singleResult = wasm.supersmoother_js(close, 14);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('SuperSmoother batch multiple periods', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    
    
    const batchResult = wasm.supersmoother_batch_js(close, 10, 20, 5);
    
    
    const metadata = wasm.supersmoother_batch_metadata_js(10, 20, 5);
    assert.strictEqual(metadata.length, 3);
    
    
    assert.strictEqual(batchResult.length, 3 * 100);
    
    
    const periods = [10, 15, 20];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.slice(rowStart, rowEnd);
        
        const singleResult = wasm.supersmoother_js(close, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('SuperSmoother batch metadata', () => {
    
    const metadata = wasm.supersmoother_batch_metadata_js(5, 15, 5);
    
    
    assert.strictEqual(metadata.length, 3);
    assert.strictEqual(metadata[0], 5);
    assert.strictEqual(metadata[1], 10);
    assert.strictEqual(metadata[2], 15);
    
    
    const singleMeta = wasm.supersmoother_batch_metadata_js(7, 7, 0);
    assert.strictEqual(singleMeta.length, 1);
    assert.strictEqual(singleMeta[0], 7);
});

test('SuperSmoother batch edge cases', () => {
    
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    
    const singleBatch = wasm.supersmoother_batch_js(close, 5, 5, 0);
    const singleMeta = wasm.supersmoother_batch_metadata_js(5, 5, 0);
    
    assert.strictEqual(singleBatch.length, 10);
    assert.strictEqual(singleMeta.length, 1);
    
    
    const largeBatch = wasm.supersmoother_batch_js(close, 5, 7, 10); 
    const largeMeta = wasm.supersmoother_batch_metadata_js(5, 7, 10);
    
    
    assert.strictEqual(largeBatch.length, 10);
    assert.strictEqual(largeMeta.length, 1);
    assert.strictEqual(largeMeta[0], 5);
});

test('SuperSmoother leading NaNs', () => {
    
    const data = new Float64Array(20);
    for (let i = 0; i < 5; i++) {
        data[i] = NaN;
    }
    for (let i = 5; i < 20; i++) {
        data[i] = i - 4; 
    }
    
    const period = 3;
    const result = wasm.supersmoother_js(data, period);
    
    
    
    
    
    
    
    
    for (let i = 0; i < 5; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} where input is NaN`);
    }
    
    
    for (let i = 5; i < 7; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} due to warmup`);
    }
    
    
    assert.strictEqual(result[7], data[7], "Expected initial value at index 7");
    assert.strictEqual(result[8], data[8], "Expected initial value at index 8");
});

test('SuperSmoother consistency', () => {
    
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const period = 3;
    
    
    const result1 = wasm.supersmoother_js(data, period);
    const result2 = wasm.supersmoother_js(data, period);
    
    
    for (let i = 0; i < result1.length; i++) {
        if (isNaN(result1[i]) && isNaN(result2[i])) continue;
        assert.strictEqual(result1[i], result2[i], `Inconsistent result at index ${i}`);
    }
});


test('SuperSmoother zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const period = 5;
    
    
    const ptr = wasm.supersmoother_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    
    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        data.length
    );
    
    
    memView.set(data);
    
    
    try {
        wasm.supersmoother_into(ptr, ptr, data.length, period);
        
        
        const regularResult = wasm.supersmoother_js(data, period);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; 
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        
        wasm.supersmoother_free(ptr, data.length);
    }
});

test('SuperSmoother zero-copy with large dataset', () => {
    const size = 10000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }
    
    const ptr = wasm.supersmoother_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');
    
    try {
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        memView.set(data);
        
        wasm.supersmoother_into(ptr, ptr, size, 14);
        
        
        const memView2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        
        
        for (let i = 0; i < 13; i++) {
            assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
        }
        
        
        for (let i = 13; i < Math.min(100, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.supersmoother_free(ptr, size);
    }
});


test('SuperSmoother zero-copy error handling', () => {
    
    assert.throws(() => {
        wasm.supersmoother_into(0, 0, 10, 14);
    }, /null pointer|invalid memory/i);
    
    
    const ptr = wasm.supersmoother_alloc(10);
    try {
        
        assert.throws(() => {
            wasm.supersmoother_into(ptr, ptr, 10, 0);
        }, /Invalid period/);
        
        
        assert.throws(() => {
            wasm.supersmoother_into(ptr, ptr, 10, 20);
        }, /Invalid period/);
    } finally {
        wasm.supersmoother_free(ptr, 10);
    }
});


test('SuperSmoother zero-copy memory management', () => {
    
    const sizes = [100, 1000, 10000];
    
    for (const size of sizes) {
        const ptr = wasm.supersmoother_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }
        
        
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
        }
        
        
        wasm.supersmoother_free(ptr, size);
    }
});


test('SuperSmoother batch - new unified API', () => {
    
    const close = new Float64Array(testData.close.slice(0, 50));
    
    
    if (typeof wasm.supersmoother_batch !== 'function') {
        console.log('New unified batch API not yet implemented');
        return;
    }
    
    
    const result = wasm.supersmoother_batch(close, {
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
});

test('SuperSmoother batch full parameter sweep', () => {
    
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.supersmoother_batch_js(close, 10, 20, 10);
    const metadata = wasm.supersmoother_batch_metadata_js(10, 20, 10);
    
    
    assert.strictEqual(metadata.length, 2);
    assert.strictEqual(batchResult.length, 2 * 50);
    
    
    for (let combo = 0; combo < metadata.length; combo++) {
        const period = metadata[combo];
        const rowStart = combo * 50;
        const rowData = batchResult.slice(rowStart, rowStart + 50);
        
        
        for (let i = 0; i < period - 1; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${period}`);
        }
        
        
        for (let i = period - 1; i < 50; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period}`);
        }
    }
});

test('Compare SuperSmoother with Rust implementation', async () => {
    const close = new Float64Array(testData.close);
    const period = 14;
    
    
    const wasmResult = wasm.supersmoother_js(close, period);
    
    
    const result = await compareWithRust('supersmoother', wasmResult, 'close', { period });
    
    
    assert.ok(result, 'Comparison with Rust succeeded');
});

test.after(() => {
    console.log('SuperSmoother WASM tests completed');
});