/**
 * WASM binding tests for MFI (Money Flow Index) indicator.
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

test('MFI partial params', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    
    const typicalPrice = new Float64Array(high.length);
    for (let i = 0; i < high.length; i++) {
        typicalPrice[i] = (high[i] + low[i] + close[i]) / 3.0;
    }
    
    const result = wasm.mfi_js(typicalPrice, volume, 14);
    assert.strictEqual(result.length, typicalPrice.length);
});

test('MFI accuracy', async () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    const expected = EXPECTED_OUTPUTS.mfi;
    
    
    const typicalPrice = new Float64Array(high.length);
    for (let i = 0; i < high.length; i++) {
        typicalPrice[i] = (high[i] + low[i] + close[i]) / 3.0;
    }
    
    const result = wasm.mfi_js(
        typicalPrice,
        volume,
        expected.defaultParams.period
    );
    
    assert.strictEqual(result.length, typicalPrice.length);
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-1,  
        "MFI last 5 values mismatch"
    );
    
    
    await compareWithRust('mfi', result, 'hlc3_volume', expected.defaultParams);
});

test('MFI default candles', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    
    const typicalPrice = new Float64Array(high.length);
    for (let i = 0; i < high.length; i++) {
        typicalPrice[i] = (high[i] + low[i] + close[i]) / 3.0;
    }
    
    const result = wasm.mfi_js(typicalPrice, volume, 14);
    assert.strictEqual(result.length, typicalPrice.length);
});

test('MFI zero period', () => {
    
    const data = new Float64Array(20).fill(100);
    const volume = new Float64Array(20).fill(1000);
    
    assert.throws(() => {
        wasm.mfi_js(data, volume, 0);
    }, /Invalid period/);
});

test('MFI period exceeds length', () => {
    
    const input_high = [1.0, 2.0, 3.0];
    const input_low = [0.5, 1.5, 2.5];
    const input_close = [0.8, 1.8, 2.8];
    const input_volume = [100.0, 200.0, 300.0];
    const typicalPrice = new Float64Array(3);
    for (let i = 0; i < 3; i++) {
        typicalPrice[i] = (input_high[i] + input_low[i] + input_close[i]) / 3.0;
    }
    const volume = new Float64Array(input_volume);
    
    assert.throws(() => {
        wasm.mfi_js(typicalPrice, volume, 10);
    }, /Invalid period/);
});

test('MFI very small dataset', () => {
    
    const typicalPrice = new Float64Array([(1.0 + 0.5 + 0.8) / 3.0]);
    const volume = new Float64Array([100.0]);
    
    assert.throws(() => {
        wasm.mfi_js(typicalPrice, volume, 14);
    }, /Invalid period/);
});

test('MFI empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.mfi_js(empty, empty, 14);
    }, /empty/i);
});

test('MFI mismatched array lengths', () => {
    
    const typicalPrice = new Float64Array([1, 2, 3]);
    const volume = new Float64Array([1, 2]);
    
    assert.throws(() => {
        wasm.mfi_js(typicalPrice, volume, 14);
    }, /empty/i);
});

test('MFI NaN handling', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    
    const typicalPrice = new Float64Array(high.length);
    for (let i = 0; i < high.length; i++) {
        typicalPrice[i] = (high[i] + low[i] + close[i]) / 3.0;
    }
    
    const result = wasm.mfi_js(typicalPrice, volume, 14);
    assert.strictEqual(result.length, typicalPrice.length);
    
    
    assertAllNaN(result.slice(0, 13), "Expected NaN in warmup period");
    
    
    if (result.length > 13) {
        
        let hasValidValues = false;
        for (let i = 13; i < Math.min(100, result.length); i++) {
            if (!isNaN(result[i])) {
                hasValidValues = true;
                
                assert(result[i] >= 0.0 && result[i] <= 100.0,
                    `MFI value ${result[i]} at index ${i} out of bounds [0, 100]`);
            }
        }
        assert(hasValidValues, "Expected valid values after warmup period");
    }
});

test('MFI with NaN input', () => {
    
    const n = 30;
    const typicalPrice = new Float64Array(n);
    const volume = new Float64Array(n).fill(1000);
    for (let i = 0; i < n; i++) {
        typicalPrice[i] = i;
    }
    typicalPrice[5] = NaN; 

    const period = 14;
    const result = wasm.mfi_js(typicalPrice, volume, period);
    assert.strictEqual(result.length, n);

    
    
    
    for (let i = 0; i < period - 1; i++) {
        assert(isNaN(result[i]), `Expected NaN at warmup index ${i}`);
    }
    for (let i = period - 1; i < n; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} due to seed NaN`);
    }
});

test('MFI all NaN input', () => {
    
    const data = new Float64Array(20).fill(NaN);
    const volume = new Float64Array(20).fill(NaN);
    
    assert.throws(() => {
        wasm.mfi_js(data, volume, 14);
    }, /All values are NaN/);
});

test('MFI zero volume', () => {
    
    const n = 30;
    const typicalPrice = new Float64Array(n);
    const volume = new Float64Array(n).fill(0);
    
    
    for (let i = 0; i < n; i++) {
        typicalPrice[i] = 100 + i;
    }
    
    const result = wasm.mfi_js(typicalPrice, volume, 14);
    
    
    for (let i = 13; i < result.length; i++) {
        assert(Math.abs(result[i] - 0.0) < 1e-10,
            `Expected MFI to be 0 with zero volume, got ${result[i]} at index ${i}`);
    }
});

test('MFI batch single parameter set', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    
    const typicalPrice = new Float64Array(high.length);
    for (let i = 0; i < high.length; i++) {
        typicalPrice[i] = (high[i] + low[i] + close[i]) / 3.0;
    }
    
    
    const config = {
        period_range: [14, 14, 0]  
    };
    
    const batchResult = wasm.mfi_batch(typicalPrice, volume, config);
    
    
    assert(batchResult.values, "Expected values in batch result");
    assert(batchResult.combos, "Expected combos in batch result");
    assert.strictEqual(batchResult.rows, 1, "Expected 1 row");
    assert.strictEqual(batchResult.cols, typicalPrice.length, "Expected cols to match data length");
    assert.strictEqual(batchResult.combos.length, 1, "Expected 1 parameter combination");
    
    
    const singleResult = wasm.mfi_js(typicalPrice, volume, 14);
    const firstRow = batchResult.values.slice(0, batchResult.cols);
    assertArrayClose(
        firstRow,
        singleResult,
        1e-10,
        "Batch should match single calculation"
    );
});

test('MFI batch multiple periods', () => {
    
    const high = new Float64Array(testData.high.slice(0, 100)); 
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    const volume = new Float64Array(testData.volume.slice(0, 100));
    
    
    const typicalPrice = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        typicalPrice[i] = (high[i] + low[i] + close[i]) / 3.0;
    }
    
    
    const config = {
        period_range: [10, 20, 5]
    };
    
    const result = wasm.mfi_batch(typicalPrice, volume, config);
    
    
    assert.strictEqual(result.rows, 3, "Expected 3 rows (periods)");
    assert.strictEqual(result.cols, 100, "Expected 100 columns");
    assert.strictEqual(result.combos.length, 3, "Expected 3 combinations");
    assert.strictEqual(result.values.length, 300, "Expected 300 total values");
    
    
    const periods = [10, 15, 20];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = result.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.mfi_js(typicalPrice, volume, periods[i]);
        assertArrayClose(
            rowData,
            singleResult,
            1e-10,
            `Period ${periods[i]} mismatch`
        );
        
        
        for (let j = 0; j < periods[i] - 1; j++) {
            assert(isNaN(rowData[j]), `Expected NaN at warmup index ${j} for period ${periods[i]}`);
        }
    }
});

test('MFI batch metadata', () => {
    
    const typicalPrice = new Float64Array(50).fill(100);
    const volume = new Float64Array(50).fill(1000);
    
    const result = wasm.mfi_batch(typicalPrice, volume, {
        period_range: [10, 20, 5]  
    });
    
    
    assert.strictEqual(result.combos.length, 3);
    
    
    assert.strictEqual(result.combos[0].period, 10);
    assert.strictEqual(result.combos[1].period, 15);
    assert.strictEqual(result.combos[2].period, 20);
});

test('MFI batch edge cases', () => {
    
    const typicalPrice = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const volume = new Float64Array(10).fill(100);
    
    
    const singleBatch = wasm.mfi_batch(typicalPrice, volume, {
        period_range: [5, 5, 1]
    });
    
    assert.strictEqual(singleBatch.values.length, 10);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    
    const largeBatch = wasm.mfi_batch(typicalPrice, volume, {
        period_range: [5, 7, 10]  
    });
    
    
    assert.strictEqual(largeBatch.values.length, 10);
    assert.strictEqual(largeBatch.combos.length, 1);
    
    
    assert.throws(() => {
        wasm.mfi_batch(new Float64Array([]), new Float64Array([]), {
            period_range: [14, 14, 0]
        });
    }, /All values are NaN/);
});

test('MFI memory allocation/deallocation', () => {
    
    const len = 1000;
    
    
    const ptr = wasm.mfi_alloc(len);
    assert(ptr !== 0, "Expected non-null pointer from mfi_alloc");
    
    
    wasm.mfi_free(ptr, len);
    
    
    wasm.mfi_free(0, len); 
});

test('MFI fast API (in-place)', { skip: 'WASM memory not directly accessible in ES modules' }, () => {
    
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    const volume = new Float64Array(testData.volume.slice(0, 100));
    
    
    const typicalPrice = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        typicalPrice[i] = (high[i] + low[i] + close[i]) / 3.0;
    }
    
    
    const expected = wasm.mfi_js(typicalPrice, volume, 14);
    
    
    const len = typicalPrice.length;
    const tpPtr = wasm.mfi_alloc(len);
    const volPtr = wasm.mfi_alloc(len);
    const outPtr = wasm.mfi_alloc(len);
    
    try {
        
        const tpMem = new Float64Array(wasm.memory.buffer, tpPtr, len);
        const volMem = new Float64Array(wasm.memory.buffer, volPtr, len);
        tpMem.set(typicalPrice);
        volMem.set(volume);
        
        
        wasm.mfi_into(
            tpPtr,
            volPtr,
            outPtr,
            len,
            14
        );
        
        
        const result = new Float64Array(wasm.memory.buffer, outPtr, len);
        
        
        assertArrayClose(
            Array.from(result),
            Array.from(expected),
            1e-10,
            "Fast API result mismatch"
        );
    } finally {
        wasm.mfi_free(tpPtr, len);
        wasm.mfi_free(volPtr, len);
        wasm.mfi_free(outPtr, len);
    }
});

test('MFI fast API with aliasing', { skip: 'WASM memory not directly accessible in ES modules' }, () => {
    
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    const volume = new Float64Array(testData.volume.slice(0, 100));
    
    
    const typicalPrice = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        typicalPrice[i] = (high[i] + low[i] + close[i]) / 3.0;
    }
    const typicalPriceCopy = new Float64Array(typicalPrice);
    
    
    const expected = wasm.mfi_js(typicalPrice, volume, 14);
    
    
    const len = 100;
    const tpPtr = wasm.mfi_alloc(len);
    const volPtr = wasm.mfi_alloc(len);
    
    try {
        
        const tpMem = new Float64Array(wasm.memory.buffer, tpPtr, len);
        const volMem = new Float64Array(wasm.memory.buffer, volPtr, len);
        tpMem.set(typicalPriceCopy);
        volMem.set(volume);
        
        
        wasm.mfi_into(
            tpPtr,
            volPtr,
            tpPtr,  
            len,
            14
        );
        
        
        const result = new Float64Array(wasm.memory.buffer, tpPtr, len);
        
        
        assertArrayClose(
            Array.from(result),
            Array.from(expected),
            1e-10,
            "Fast API aliasing result mismatch"
        );
    } finally {
        wasm.mfi_free(tpPtr, len);
        wasm.mfi_free(volPtr, len);
    }
});

test('MFI batch fast API', { skip: 'WASM memory not directly accessible in ES modules' }, () => {
    
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    const close = new Float64Array(testData.close.slice(0, 50));
    const volume = new Float64Array(testData.volume.slice(0, 50));
    
    
    const typicalPrice = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        typicalPrice[i] = (high[i] + low[i] + close[i]) / 3.0;
    }
    
    
    const periods = { start: 10, end: 20, step: 5 }; 
    const numCombos = 3;
    const totalSize = numCombos * 50;
    
    
    const tpPtr = wasm.mfi_alloc(50);
    const volPtr = wasm.mfi_alloc(50);
    const outPtr = wasm.mfi_alloc(totalSize);
    
    try {
        
        const tpMem = new Float64Array(wasm.memory.buffer, tpPtr, 50);
        const volMem = new Float64Array(wasm.memory.buffer, volPtr, 50);
        tpMem.set(typicalPrice);
        volMem.set(volume);
        
        
        const rows = wasm.mfi_batch_into(
            tpPtr,
            volPtr,
            outPtr,
            50,
            periods.start,
            periods.end,
            periods.step
        );
        
        assert.strictEqual(rows, 3, "Expected 3 rows returned");
        
        
        const result = new Float64Array(wasm.memory.buffer, outPtr, totalSize);
        
        
        const singleResult = wasm.mfi_js(typicalPrice, volume, 10);
        const firstRow = Array.from(result.slice(0, 50));
        assertArrayClose(
            firstRow,
            Array.from(singleResult),
            1e-10,
            "Batch fast API first row mismatch"
        );
    } finally {
        wasm.mfi_free(tpPtr, 50);
        wasm.mfi_free(volPtr, 50);
        wasm.mfi_free(outPtr, totalSize);
    }
});

test('MFI zero-copy error handling', () => {
    
    assert.throws(() => {
        wasm.mfi_into(0, 0, 0, 10, 14);
    }, /null pointer|Null pointer/i);
    
    
    const ptr = wasm.mfi_alloc(10);
    const volPtr = wasm.mfi_alloc(10);
    try {
        
        assert.throws(() => {
            wasm.mfi_into(ptr, volPtr, ptr, 10, 0);
        }, /Invalid period/);
        
        
        assert.throws(() => {
            wasm.mfi_into(ptr, volPtr, ptr, 10, 20);
        }, /Invalid period/);
    } finally {
        wasm.mfi_free(ptr, 10);
        wasm.mfi_free(volPtr, 10);
    }
});

test('MFI memory leak prevention', { skip: 'WASM memory not directly accessible in ES modules' }, () => {
    
    const sizes = [100, 1000, 10000];
    
    for (const size of sizes) {
        const ptr = wasm.mfi_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        
        const memView = new Float64Array(wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }
        
        
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
        }
        
        
        wasm.mfi_free(ptr, size);
    }
});

test.after(() => {
    console.log('MFI WASM tests completed');
});
