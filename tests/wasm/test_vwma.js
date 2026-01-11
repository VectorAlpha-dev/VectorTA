/**
 * WASM binding tests for VWMA indicator.
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

test('VWMA partial params', () => {
    
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    const result = wasm.vwma_js(close, volume, 20);
    assert.strictEqual(result.length, close.length);
    
    
    const result_custom = wasm.vwma_js(close, volume, 10);
    assert.strictEqual(result_custom.length, close.length);
});

test('VWMA accuracy', async () => {
    
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    const expected = EXPECTED_OUTPUTS.vwma;
    
    const result = wasm.vwma_js(
        close,
        volume,
        expected.defaultParams.period
    );
    
    assert.strictEqual(result.length, close.length);
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-3,  
        "VWMA last 5 values mismatch"
    );
    
    
    await compareWithRust('vwma', result, 'close', expected.defaultParams);
});

test('VWMA price volume mismatch', () => {
    
    const prices = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0]);
    const volumes = new Float64Array([100.0, 200.0, 300.0]);  
    
    assert.throws(() => {
        wasm.vwma_js(prices, volumes, 3);
    }, /Price and volume mismatch/);
});

test('VWMA invalid period', () => {
    
    const prices = new Float64Array([10.0, 20.0, 30.0]);
    const volumes = new Float64Array([100.0, 200.0, 300.0]);
    
    
    assert.throws(() => {
        wasm.vwma_js(prices, volumes, 0);
    }, /Invalid period/);
    
    
    assert.throws(() => {
        wasm.vwma_js(prices, volumes, 10);
    }, /Invalid period/);
});

test('VWMA all NaN values', () => {
    
    const prices = new Float64Array(10);
    const volumes = new Float64Array(10);
    prices.fill(NaN);
    volumes.fill(NaN);
    
    assert.throws(() => {
        wasm.vwma_js(prices, volumes, 5);
    }, /All/);
});

test('VWMA not enough valid data', () => {
    
    
    const prices = new Float64Array([NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 10.0, 20.0]);
    const volumes = new Float64Array([NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 100.0, 200.0]);
    
    assert.throws(() => {
        wasm.vwma_js(prices, volumes, 5);
    }, /Not enough valid/);
});

test('VWMA with default candles', () => {
    
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    
    const result = wasm.vwma_js(close, volume, 20);
    assert.strictEqual(result.length, close.length);
});

test('VWMA candles plus prices', () => {
    
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    
    const custom_prices = close.map(v => v * 1.001);
    
    const result = wasm.vwma_js(custom_prices, volume, 20);
    assert.strictEqual(result.length, custom_prices.length);
});

test('VWMA slice reinput', () => {
    
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    
    const firstResult = wasm.vwma_js(close, volume, 20);
    assert.strictEqual(firstResult.length, close.length);
    
    
    const secondResult = wasm.vwma_js(firstResult, volume, 10);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    
    
    
    const expectedWarmup = 28;
    for (let i = expectedWarmup; i < secondResult.length; i++) {
        assert(!isNaN(secondResult[i]), `Unexpected NaN at index ${i}`);
    }
});

test('VWMA NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    const result = wasm.vwma_js(close, volume, 20);
    assert.strictEqual(result.length, close.length);
    
    
    assertAllNaN(result.slice(0, 19), "Expected NaN in warmup period");
    
    
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('VWMA batch single parameter set', () => {
    
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    
    const batchResult = wasm.vwma_batch_js(
        close,
        volume,
        20, 20, 0  
    );
    
    
    const singleResult = wasm.vwma_js(close, volume, 20);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('VWMA batch multiple periods', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    const volume = new Float64Array(testData.volume.slice(0, 100));
    
    
    const batchResult = wasm.vwma_batch_js(
        close,
        volume,
        10, 20, 5  
    );
    
    
    assert.strictEqual(batchResult.length, 3 * 100);
    
    
    const periods = [10, 15, 20];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.slice(rowStart, rowEnd);
        
        const singleResult = wasm.vwma_js(close, volume, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('VWMA batch metadata', () => {
    
    const metadata = wasm.vwma_batch_metadata_js(
        10, 30, 5  
    );
    
    
    assert.strictEqual(metadata.length, 5);
    
    
    assert.strictEqual(metadata[0], 10);
    assert.strictEqual(metadata[1], 15);
    assert.strictEqual(metadata[2], 20);
    assert.strictEqual(metadata[3], 25);
    assert.strictEqual(metadata[4], 30);
});

test('VWMA batch edge cases', () => {
    
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const volume = new Float64Array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
    
    
    const singleBatch = wasm.vwma_batch_js(
        close,
        volume,
        5, 5, 1
    );
    
    assert.strictEqual(singleBatch.length, 10);
    
    
    const largeBatch = wasm.vwma_batch_js(
        close,
        volume,
        5, 7, 10  
    );
    
    
    assert.strictEqual(largeBatch.length, 10);
    
    
    assert.throws(() => {
        wasm.vwma_batch_js(
            new Float64Array([]),
            new Float64Array([]),
            10, 10, 0
        );
    }, /empty/i);
});



test('VWMA fast/unsafe API basic', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    const volume = new Float64Array(testData.volume.slice(0, 100));
    
    
    const closePtr = wasm.vwma_alloc(100);
    const volumePtr = wasm.vwma_alloc(100);
    const outPtr = wasm.vwma_alloc(100);
    
    try {
        
        const memory = wasm.__wasm?.memory || wasm.memory;
        if (!memory) {
            throw new Error('WASM memory not accessible');
        }
        
        
        const closeView = new Float64Array(memory.buffer, closePtr, 100);
        const volumeView = new Float64Array(memory.buffer, volumePtr, 100);
        closeView.set(close);
        volumeView.set(volume);
        
        
        wasm.vwma_into(
            closePtr,
            volumePtr,
            outPtr,
            100,
            20  
        );
        
        
        const result = new Float64Array(memory.buffer, outPtr, 100);
        const resultCopy = new Float64Array(result); 
        
        
        const safeResult = wasm.vwma_js(close, volume, 20);
        assertArrayClose(resultCopy, safeResult, 1e-10, "Fast vs safe API mismatch");
    } finally {
        
        wasm.vwma_free(closePtr, 100);
        wasm.vwma_free(volumePtr, 100);
        wasm.vwma_free(outPtr, 100);
    }
});

test('VWMA fast API in-place operation (aliasing)', () => {
    
    const prices = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const volumes = new Float64Array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
    
    
    const pricesCopy = new Float64Array(prices);
    
    
    const pricePtr = wasm.vwma_alloc(10);
    const volumePtr = wasm.vwma_alloc(10);
    
    try {
        
        const memory = wasm.__wasm?.memory || wasm.memory;
        if (!memory) {
            throw new Error('WASM memory not accessible');
        }
        
        
        const priceView = new Float64Array(memory.buffer, pricePtr, 10);
        const volumeView = new Float64Array(memory.buffer, volumePtr, 10);
        priceView.set(prices);
        volumeView.set(volumes);
        
        
        wasm.vwma_into(
            pricePtr,
            volumePtr,
            pricePtr,  
            10,
            3  
        );
        
        
        const result = new Float64Array(memory.buffer, pricePtr, 10);
        const resultCopy = new Float64Array(result); 
        
        
        const expected = wasm.vwma_js(pricesCopy, volumes, 3);
        assertArrayClose(resultCopy, expected, 1e-10, "In-place operation mismatch");
    } finally {
        
        wasm.vwma_free(pricePtr, 10);
        wasm.vwma_free(volumePtr, 10);
    }
});

test('VWMA fast API null pointer handling', () => {
    
    assert.throws(() => {
        wasm.vwma_into(0, 0, 0, 10, 5);
    }, /Null pointer/);
});

test('VWMA memory allocation and deallocation', () => {
    
    const sizes = [10, 100, 1000];
    
    
    const memory = wasm.__wasm?.memory || wasm.memory;
    if (!memory) {
        throw new Error('WASM memory not accessible');
    }
    
    for (const size of sizes) {
        const ptr = wasm.vwma_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        
        const arr = new Float64Array(memory.buffer, ptr, size);
        for (let i = 0; i < size; i++) {
            arr[i] = i;
        }
        
        
        for (let i = 0; i < size; i++) {
            assert.strictEqual(arr[i], i);
        }
        
        
        wasm.vwma_free(ptr, size);
    }
});

test('VWMA batch unified API with serde config', () => {
    
    const close = new Float64Array(testData.close.slice(0, 50));
    const volume = new Float64Array(testData.volume.slice(0, 50));
    
    const config = {
        period_range: [10, 20, 5]  
    };
    
    const result = wasm.vwma_batch(close, volume, config);
    
    
    assert(result.values instanceof Array, "Expected values array");
    assert(result.combos instanceof Array, "Expected combos array");
    assert.strictEqual(result.rows, 3, "Expected 3 rows (periods: 10, 15, 20)");
    assert.strictEqual(result.cols, 50, "Expected 50 columns");
    assert.strictEqual(result.values.length, 150, "Expected 150 total values");
    
    
    assert.strictEqual(result.combos.length, 3);
    assert.strictEqual(result.combos[0].period, 10);
    assert.strictEqual(result.combos[1].period, 15);
    assert.strictEqual(result.combos[2].period, 20);
});

test('VWMA batch fast API', () => {
    
    const close = new Float64Array(testData.close.slice(0, 30));
    const volume = new Float64Array(testData.volume.slice(0, 30));
    
    
    const memory = wasm.__wasm?.memory || wasm.memory;
    if (!memory) {
        throw new Error('WASM memory not accessible');
    }
    
    
    const closePtr = wasm.vwma_alloc(30);
    const volumePtr = wasm.vwma_alloc(30);
    
    
    const outPtr = wasm.vwma_alloc(60);
    
    try {
        
        const closeView = new Float64Array(memory.buffer, closePtr, 30);
        const volumeView = new Float64Array(memory.buffer, volumePtr, 30);
        closeView.set(close);
        volumeView.set(volume);
        
        const rows = wasm.vwma_batch_into(
            closePtr,
            volumePtr,
            outPtr,
            30,
            10, 15, 5  
        );
        
        assert.strictEqual(rows, 2, "Expected 2 rows");
        
        
        const result = new Float64Array(memory.buffer, outPtr, 60);
        
        
        const expected1 = wasm.vwma_js(close, volume, 10);
        const expected2 = wasm.vwma_js(close, volume, 15);
        
        assertArrayClose(
            result.slice(0, 30), 
            expected1, 
            1e-10, 
            "Batch row 1 mismatch"
        );
        assertArrayClose(
            result.slice(30, 60), 
            expected2, 
            1e-10, 
            "Batch row 2 mismatch"
        );
    } finally {
        wasm.vwma_free(closePtr, 30);
        wasm.vwma_free(volumePtr, 30);
        wasm.vwma_free(outPtr, 60);
    }
});

test('VWMA batch fast API with aliasing', () => {
    
    const prices = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const volumes = new Float64Array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
    
    
    const memory = wasm.__wasm?.memory || wasm.memory;
    if (!memory) {
        throw new Error('WASM memory not accessible');
    }
    
    
    const pricePtr = wasm.vwma_alloc(10);
    const volumePtr = wasm.vwma_alloc(10);
    
    try {
        
        const priceView = new Float64Array(memory.buffer, pricePtr, 10);
        const volumeView = new Float64Array(memory.buffer, volumePtr, 10);
        priceView.set(prices);
        volumeView.set(volumes);
        
        
        const pricesCopy = new Float64Array(prices);
        const volumesCopy = new Float64Array(volumes);
        
        
        
        const rows = wasm.vwma_batch_into(
            pricePtr,
            volumePtr,
            pricePtr,  
            10,
            3, 3, 1  
        );
        
        assert.strictEqual(rows, 1, "Expected 1 row");
        
        
        const result = new Float64Array(memory.buffer, pricePtr, 10);
        const resultCopy = new Float64Array(result); 
        
        
        const expected = wasm.vwma_js(pricesCopy, volumesCopy, 3);
        assertArrayClose(
            resultCopy, 
            expected, 
            1e-10, 
            "Batch aliasing mismatch"
        );
    } finally {
        wasm.vwma_free(pricePtr, 10);
        wasm.vwma_free(volumePtr, 10);
    }
});

test('VWMA zero volume', () => {
    
    const prices = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]);
    const volumes = new Float64Array([100.0, 0.0, 300.0, 0.0, 500.0, 0.0, 700.0, 0.0, 900.0, 0.0]);
    
    const result = wasm.vwma_js(prices, volumes, 3);
    assert.strictEqual(result.length, prices.length);
    
    
    
    assert(!isNaN(result[2]), "Index 2 should have valid value with non-zero volumes in window");
});

test('VWMA partial NaN data', () => {
    
    const close = new Float64Array(testData.close.slice(0, 200));
    const volume = new Float64Array(testData.volume.slice(0, 200));
    
    
    for (let i = 100; i < 110; i++) {
        close[i] = NaN;
        volume[i] = NaN;
    }
    
    
    
    const result = wasm.vwma_js(close, volume, 20);
    assert.strictEqual(result.length, close.length);
    
    
    assertAllNaN(result.slice(0, 19), "Expected NaN in warmup period");
    
    
    
});

test('VWMA warmup period verification', () => {
    
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    const period = 20;
    
    const result = wasm.vwma_js(close, volume, period);
    
    
    
    assertAllNaN(result.slice(0, 19), "First 19 values should be NaN");
    assert(!isNaN(result[19]), "Index 19 should be first valid value");
});

test('VWMA batch multiple parameter sweeps', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    const volume = new Float64Array(testData.volume.slice(0, 100));
    
    
    const batchResult = wasm.vwma_batch_js(
        close,
        volume,
        10, 30, 10  
    );
    
    
    assert.strictEqual(batchResult.length, 3 * 100);
    
    
    const periods = [10, 20, 30];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.slice(rowStart, rowEnd);
        
        const singleResult = wasm.vwma_js(close, volume, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch in batch`
        );
    }
});

test.after(() => {
    console.log('VWMA WASM tests completed');
});
