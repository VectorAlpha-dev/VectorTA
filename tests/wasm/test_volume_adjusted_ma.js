/**
 * WASM binding tests for VolumeAdjustedMa indicator.
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

test('VolumeAdjustedMa partial params', () => {
    
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    const result = wasm.volume_adjusted_ma_js(close, volume, 13, 0.67, true, 0);
    assert.strictEqual(result.length, close.length);
});

test('VolumeAdjustedMa accuracy', async () => {
    
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    const expected = EXPECTED_OUTPUTS.volume_adjusted_ma;
    
    const result = wasm.volume_adjusted_ma_js(
        close,
        volume,
        expected.defaultParams.length,
        expected.defaultParams.viFactor,
        expected.defaultParams.strict,
        expected.defaultParams.samplePeriod
    );
    
    assert.strictEqual(result.length, close.length);
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.fastValues,
        1e-6,
        "VolumeAdjustedMa last 5 values mismatch"
    );
    
    
    
});

test('VolumeAdjustedMa slow', () => {
    
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    const expected = EXPECTED_OUTPUTS.volume_adjusted_ma;
    
    const result = wasm.volume_adjusted_ma_js(
        close,
        volume,
        expected.slowParams.length,
        expected.slowParams.viFactor,
        expected.slowParams.strict,
        expected.slowParams.samplePeriod
    );
    
    assert.strictEqual(result.length, close.length);
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.slowValues,
        1e-6,
        "VolumeAdjustedMa slow last 5 values mismatch"
    );
});

test('VolumeAdjustedMa default candles', () => {
    
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    const result = wasm.volume_adjusted_ma_js(close, volume, 13, 0.67, true, 0);
    assert.strictEqual(result.length, close.length);
});

test('VolumeAdjustedMa zero period', () => {
    
    const price = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0]);
    const volume = new Float64Array([100.0, 200.0, 300.0, 400.0, 500.0]);
    
    assert.throws(() => {
        wasm.volume_adjusted_ma_js(price, volume, 0, 0.67, true, 0);
    }, /Invalid period/);
});

test('VolumeAdjustedMa empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.volume_adjusted_ma_js(empty, empty, 13, 0.67, true, 0);
    }, /Input data slice is empty/);
});

test('VolumeAdjustedMa all NaN input', () => {
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    const volume = new Float64Array(100);
    volume.fill(100);
    
    assert.throws(() => {
        wasm.volume_adjusted_ma_js(allNaN, volume, 13, 0.67, true, 0);
    }, /All values are NaN/);
});

test('VolumeAdjustedMa mismatched lengths', () => {
    
    const price = new Float64Array([10.0, 20.0, 30.0]);
    const volume = new Float64Array([100.0, 200.0]);  
    
    assert.throws(() => {
        wasm.volume_adjusted_ma_js(price, volume, 13, 0.67, true, 0);
    }, /length mismatch/);
});

test('VolumeAdjustedMa invalid period', () => {
    
    const price = new Float64Array([10.0, 20.0, 30.0]);
    const volume = new Float64Array([100.0, 200.0, 300.0]);
    
    assert.throws(() => {
        wasm.volume_adjusted_ma_js(price, volume, 0, 0.67, true, 0);
    }, /Invalid period/);
});

test('VolumeAdjustedMa invalid vi_factor', () => {
    
    const price = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0]);
    const volume = new Float64Array([100.0, 200.0, 300.0, 400.0, 500.0]);
    
    
    assert.throws(() => {
        wasm.volume_adjusted_ma_js(price, volume, 2, 0.0, true, 0);
    }, /Invalid vi_factor/);
    
    
    assert.throws(() => {
        wasm.volume_adjusted_ma_js(price, volume, 2, -1.0, true, 0);
    }, /Invalid vi_factor/);
});

test('VolumeAdjustedMa period exceeds length', () => {
    
    const smallPrice = new Float64Array([10.0, 20.0, 30.0]);
    const smallVolume = new Float64Array([100.0, 200.0, 300.0]);
    
    assert.throws(() => {
        wasm.volume_adjusted_ma_js(smallPrice, smallVolume, 10, 0.67, true, 0);
    }, /Invalid period|Not enough/);
});

test('VolumeAdjustedMa NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    const expected = EXPECTED_OUTPUTS.volume_adjusted_ma;
    
    const result = wasm.volume_adjusted_ma_js(close, volume, 13, 0.67, true, 0);
    assert.strictEqual(result.length, close.length);
    
    
    const warmup = expected.warmupPeriod; 
    
    if (result.length > warmup) {
        for (let i = warmup + 1; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    
    assertAllNaN(result.slice(0, warmup), "Expected NaN in warmup period");
});

test('VolumeAdjustedMa strict vs non-strict', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    const volume = new Float64Array(testData.volume.slice(0, 100));
    
    
    const resultStrict = wasm.volume_adjusted_ma_js(close, volume, 13, 0.67, true, 0);
    
    
    const resultNonStrict = wasm.volume_adjusted_ma_js(close, volume, 13, 0.67, false, 0);
    
    assert.strictEqual(resultStrict.length, close.length);
    assert.strictEqual(resultNonStrict.length, close.length);
    
    
    let hasValidStrict = false;
    let hasValidNonStrict = false;
    for (let i = 13; i < resultStrict.length; i++) {
        if (!isNaN(resultStrict[i])) hasValidStrict = true;
        if (!isNaN(resultNonStrict[i])) hasValidNonStrict = true;
    }
    assert(hasValidStrict, "Strict mode should produce valid values");
    assert(hasValidNonStrict, "Non-strict mode should produce valid values");
});

test('VolumeAdjustedMa sample period', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    const volume = new Float64Array(testData.volume.slice(0, 100));
    
    
    const resultAll = wasm.volume_adjusted_ma_js(close, volume, 13, 0.67, true, 0);
    
    
    const resultFixed = wasm.volume_adjusted_ma_js(close, volume, 13, 0.67, true, 20);
    
    assert.strictEqual(resultAll.length, close.length);
    assert.strictEqual(resultFixed.length, close.length);
    
    
    let hasValidAll = false;
    let hasValidFixed = false;
    for (let i = 13; i < resultAll.length; i++) {
        if (!isNaN(resultAll[i])) hasValidAll = true;
        if (!isNaN(resultFixed[i])) hasValidFixed = true;
    }
    assert(hasValidAll, "sample_period=0 should produce valid values");
    assert(hasValidFixed, "Fixed sample_period should produce valid values");
});

test('VolumeAdjustedMa different vi_factors', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    const volume = new Float64Array(testData.volume.slice(0, 100));
    
    
    const result1 = wasm.volume_adjusted_ma_js(close, volume, 13, 0.5, true, 0);
    const result2 = wasm.volume_adjusted_ma_js(close, volume, 13, 0.67, true, 0);
    const result3 = wasm.volume_adjusted_ma_js(close, volume, 13, 1.0, true, 0);
    
    assert.strictEqual(result1.length, close.length);
    assert.strictEqual(result2.length, close.length);
    assert.strictEqual(result3.length, close.length);
    
    
    let hasDifference12 = false;
    let hasDifference23 = false;
    
    for (let i = close.length - 10; i < close.length; i++) {
        if (result1[i] !== result2[i]) hasDifference12 = true;
        if (result2[i] !== result3[i]) hasDifference23 = true;
    }
    
    assert(hasDifference12, "Different vi_factors should produce different results");
    assert(hasDifference23, "Different vi_factors should produce different results");
});

test('VolumeAdjustedMa very small dataset', () => {
    
    const singlePrice = new Float64Array([42.0]);
    const singleVolume = new Float64Array([100.0]);
    
    assert.throws(() => {
        wasm.volume_adjusted_ma_js(singlePrice, singleVolume, 13, 0.67, true, 0);
    }, /Invalid period|Not enough valid data/);
});

test('VolumeAdjustedMa zero length', () => {
    
    const price = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0]);
    const volume = new Float64Array([100.0, 200.0, 300.0, 400.0, 500.0]);
    
    assert.throws(() => {
        wasm.volume_adjusted_ma_js(price, volume, 0, 0.67, true, 0);
    }, /Invalid period/);
});

test('VolumeAdjustedMa length exceeds data', () => {
    
    const price = new Float64Array([10.0, 20.0, 30.0]);
    const volume = new Float64Array([100.0, 200.0, 300.0]);
    
    assert.throws(() => {
        wasm.volume_adjusted_ma_js(price, volume, 10, 0.67, true, 0);
    }, /Invalid period/);
});

test('VolumeAdjustedMa batch single parameter set', () => {
    
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    
    const batchResult = wasm.volume_adjusted_ma_batch(close, volume, {
        length_range: [13, 13, 0],
        vi_factor_range: [0.67, 0.67, 0],
        strict: true,
        sample_period_range: [0, 0, 0]
    });
    
    
    const singleResult = wasm.volume_adjusted_ma_js(close, volume, 13, 0.67, true, 0);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('VolumeAdjustedMa batch multiple periods', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    const volume = new Float64Array(testData.volume.slice(0, 100));
    
    
    const batchResult = wasm.volume_adjusted_ma_batch(close, volume, {
        length_range: [13, 17, 2],         
        vi_factor_range: [0.67, 0.67, 0],  
        strict: true,                       
        sample_period_range: [0, 0, 0]     
    });
    
    
    assert.strictEqual(batchResult.values.length, 3 * 100);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 100);
    
    
    const periods = [13, 15, 17];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.volume_adjusted_ma_js(close, volume, periods[i], 0.67, true, 0);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('VolumeAdjustedMa batch metadata from result', () => {
    
    const close = new Float64Array(20);
    close.fill(100);
    const volume = new Float64Array(20);
    volume.fill(1000);
    
    const result = wasm.volume_adjusted_ma_batch(close, volume, {
        length_range: [10, 14, 2],        
        vi_factor_range: [0.5, 0.7, 0.1], 
        strict: true,
        sample_period_range: [0, 10, 5]   
    });
    
    
    assert.strictEqual(result.combos.length, 27);
    
    
    assert.strictEqual(result.combos[0].length, 10);
    assert.strictEqual(result.combos[0].vi_factor, 0.5);
    assert.strictEqual(result.combos[0].strict, true);
    assert.strictEqual(result.combos[0].sample_period, 0);
    
    
    assert.strictEqual(result.combos[26].length, 14);
    assertClose(result.combos[26].vi_factor, 0.7, 1e-10, "vi_factor mismatch");
    assert.strictEqual(result.combos[26].strict, true);
    assert.strictEqual(result.combos[26].sample_period, 10);
});

test('VolumeAdjustedMa batch full parameter sweep', () => {
    
    const close = new Float64Array(testData.close.slice(0, 50));
    const volume = new Float64Array(testData.volume.slice(0, 50));
    
    const batchResult = wasm.volume_adjusted_ma_batch(close, volume, {
        length_range: [10, 12, 2],         
        vi_factor_range: [0.6, 0.7, 0.1],  
        strict: false,                      
        sample_period_range: [0, 0, 0]     
    });
    
    
    assert.strictEqual(batchResult.combos.length, 4);
    assert.strictEqual(batchResult.rows, 4);
    assert.strictEqual(batchResult.cols, 50);
    assert.strictEqual(batchResult.values.length, 4 * 50);
    
    
    for (let combo = 0; combo < batchResult.combos.length; combo++) {
        const length = batchResult.combos[combo].length;
        const vi_factor = batchResult.combos[combo].vi_factor;
        const sample_period = batchResult.combos[combo].sample_period;
        
        const rowStart = combo * 50;
        const rowData = batchResult.values.slice(rowStart, rowStart + 50);
        
        
        for (let i = 0; i < length - 1; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for length ${length}`);
        }
        
        
        
        for (let i = length; i < 50; i++) {
            
            if (i >= length) {
                assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for length ${length}`);
            }
        }
    }
});

test('VolumeAdjustedMa batch edge cases', () => {
    
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const volume = new Float64Array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100]);
    
    
    const singleBatch = wasm.volume_adjusted_ma_batch(close, volume, {
        length_range: [5, 5, 1],
        vi_factor_range: [0.67, 0.67, 0.1],
        strict: true,
        sample_period_range: [0, 0, 1]
    });
    
    assert.strictEqual(singleBatch.values.length, 10);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    
    const largeBatch = wasm.volume_adjusted_ma_batch(close, volume, {
        length_range: [5, 7, 10], 
        vi_factor_range: [0.67, 0.67, 0],
        strict: true,
        sample_period_range: [0, 0, 0]
    });
    
    
    assert.strictEqual(largeBatch.values.length, 10);
    assert.strictEqual(largeBatch.combos.length, 1);
    
    
    assert.throws(() => {
        wasm.volume_adjusted_ma_batch(new Float64Array([]), new Float64Array([]), {
            length_range: [13, 13, 0],
            vi_factor_range: [0.67, 0.67, 0],
            strict: true,
            sample_period_range: [0, 0, 0]
        });
    }, /Input data slice is empty|All values are NaN/);
});


test('VolumeAdjustedMa zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const volume = new Float64Array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100]);
    const length = 5;
    const vi_factor = 0.67;
    const strict = true;
    const sample_period = 0;
    
    
    const dataPtr = wasm.volume_adjusted_ma_alloc(data.length);
    const volPtr = wasm.volume_adjusted_ma_alloc(volume.length);
    assert(dataPtr !== 0, 'Failed to allocate data memory');
    assert(volPtr !== 0, 'Failed to allocate volume memory');
    
    
    const memory = wasm.__wasm.memory.buffer;
    const dataView = new Float64Array(
        memory,
        dataPtr,
        data.length
    );
    const volView = new Float64Array(
        memory,
        volPtr,
        volume.length
    );
    
    
    dataView.set(data);
    volView.set(volume);
    
    
    try {
        wasm.volume_adjusted_ma_into(dataPtr, volPtr, dataPtr, data.length, length, vi_factor, strict, sample_period);
        
        
        const regularResult = wasm.volume_adjusted_ma_js(data, volume, length, vi_factor, strict, sample_period);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(dataView[i])) {
                continue; 
            }
            assert(Math.abs(regularResult[i] - dataView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${dataView[i]}`);
        }
    } finally {
        
        wasm.volume_adjusted_ma_free(dataPtr, data.length);
        wasm.volume_adjusted_ma_free(volPtr, volume.length);
    }
});

test('VolumeAdjustedMa zero-copy with large dataset', () => {
    const size = 10000;
    const data = new Float64Array(size);
    const volume = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
        volume[i] = 900 + Math.random() * 200;
    }
    
    const dataPtr = wasm.volume_adjusted_ma_alloc(size);
    const volPtr = wasm.volume_adjusted_ma_alloc(size);
    assert(dataPtr !== 0, 'Failed to allocate large data buffer');
    assert(volPtr !== 0, 'Failed to allocate large volume buffer');
    
    try {
        const memory = wasm.__wasm.memory.buffer;
        const dataView = new Float64Array(memory, dataPtr, size);
        const volView = new Float64Array(memory, volPtr, size);
        dataView.set(data);
        volView.set(volume);
        
        wasm.volume_adjusted_ma_into(dataPtr, volPtr, dataPtr, size, 13, 0.67, true, 0);
        
        
        const memory2 = wasm.__wasm.memory.buffer;
        const dataView2 = new Float64Array(memory2, dataPtr, size);
        
        
        for (let i = 0; i < 12; i++) {
            assert(isNaN(dataView2[i]), `Expected NaN at warmup index ${i}`);
        }
        
        
        for (let i = 12; i < Math.min(100, size); i++) {
            assert(!isNaN(dataView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.volume_adjusted_ma_free(dataPtr, size);
        wasm.volume_adjusted_ma_free(volPtr, size);
    }
});


test('VolumeAdjustedMa zero-copy error handling', () => {
    
    assert.throws(() => {
        wasm.volume_adjusted_ma_into(0, 0, 0, 10, 13, 0.67, true, 0);
    }, /null pointer|invalid memory/i);
    
    
    const dataPtr = wasm.volume_adjusted_ma_alloc(10);
    const volPtr = wasm.volume_adjusted_ma_alloc(10);
    try {
        
        assert.throws(() => {
            wasm.volume_adjusted_ma_into(dataPtr, volPtr, dataPtr, 10, 0, 0.67, true, 0);
        }, /Invalid period/);
        
        
        assert.throws(() => {
            wasm.volume_adjusted_ma_into(dataPtr, volPtr, dataPtr, 10, 5, 0.0, true, 0);
        }, /Invalid vi_factor/);
    } finally {
        wasm.volume_adjusted_ma_free(dataPtr, 10);
        wasm.volume_adjusted_ma_free(volPtr, 10);
    }
});


test('VolumeAdjustedMa zero-copy memory management', () => {
    
    const sizes = [100, 1000, 10000];
    
    for (const size of sizes) {
        const dataPtr = wasm.volume_adjusted_ma_alloc(size);
        const volPtr = wasm.volume_adjusted_ma_alloc(size);
        assert(dataPtr !== 0, `Failed to allocate ${size} data elements`);
        assert(volPtr !== 0, `Failed to allocate ${size} volume elements`);
        
        
        const memory = wasm.__wasm.memory.buffer;
        const dataView = new Float64Array(memory, dataPtr, size);
        const volView = new Float64Array(memory, volPtr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            dataView[i] = i * 1.5;
            volView[i] = i * 100.0;
        }
        
        
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(dataView[i], i * 1.5, `Data memory corruption at index ${i}`);
            assert.strictEqual(volView[i], i * 100.0, `Volume memory corruption at index ${i}`);
        }
        
        
        wasm.volume_adjusted_ma_free(dataPtr, size);
        wasm.volume_adjusted_ma_free(volPtr, size);
    }
});

test('VolumeAdjustedMa constant volume', () => {
    
    
    const priceData = [];
    for (let i = 0; i < 5; i++) {
        priceData.push(50.0, 51.0, 49.0, 52.0, 48.0, 53.0, 47.0, 54.0, 46.0, 55.0);
    }
    const price = new Float64Array(priceData);
    
    
    const volume = new Float64Array(50);
    volume.fill(1000.0);
    
    const result = wasm.volume_adjusted_ma_js(price, volume, 5, 0.67, true, 0);
    
    assert.strictEqual(result.length, price.length);
    
    
    let hasValidValues = false;
    for (let i = 5; i < result.length; i++) {
        if (!isNaN(result[i])) {
            hasValidValues = true;
            break;
        }
    }
    assert(hasValidValues, "VolumeAdjustedMa should produce valid values with constant volume");
});

test.after(() => {
    console.log('VolumeAdjustedMa WASM tests completed');
});
