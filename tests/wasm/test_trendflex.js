/**
 * WASM binding tests for TrendFlex indicator.
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
    assertNoNaN 
} from './test_utils.js';
import { compareWithRust } from './rust-comparison.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;


const EXPECTED_OUTPUTS = {
    trendflex: {
        defaultParams: { period: 20 },
        last5Values: [
            -0.19724678008015128,
            -0.1238001236481444,
            -0.10515389737087717,
            -0.1149541079904878,
            -0.16006869484450567,
        ]
    }
};

test.before(async () => {
    
    try {
        const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
        const importPath = process.platform === 'win32' 
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
        wasm = await import(importPath);
    } catch (error) {
        console.error('Failed to load WASM module:', error);
        throw error;
    }
    
    
    testData = loadTestData();
});

test('trendflex_partial_params', () => {
    const close = testData.close;
    
    
    const result = wasm.trendflex_js(close, 20);
    assert.equal(result.length, close.length);
});

test('trendflex_accuracy', () => {
    const close = testData.close;
    const expected = EXPECTED_OUTPUTS.trendflex;
    
    const result = wasm.trendflex_js(close, expected.defaultParams.period);
    
    assert.equal(result.length, close.length);
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-8,
        "TrendFlex last 5 values mismatch"
    );
    
    
    compareWithRust('trendflex', result, 'close', expected.defaultParams);
});

test('trendflex_default_candles', () => {
    const close = testData.close;
    
    
    const result = wasm.trendflex_js(close, 20);
    assert.equal(result.length, close.length);
});

test('trendflex_zero_period', () => {
    const inputData = [10.0, 20.0, 30.0];
    
    assert.throws(() => {
        wasm.trendflex_js(inputData, 0);
    }, /period = 0|ZeroTrendFlexPeriod/);
});

test('trendflex_period_exceeds_length', () => {
    const dataSmall = [10.0, 20.0, 30.0];
    
    assert.throws(() => {
        wasm.trendflex_js(dataSmall, 10);
    }, /period > data len|TrendFlexPeriodExceedsData/);
});

test('trendflex_very_small_dataset', () => {
    const singlePoint = [42.0];
    
    assert.throws(() => {
        wasm.trendflex_js(singlePoint, 9);
    }, /period > data len|TrendFlexPeriodExceedsData/);
});

test('trendflex_empty_input', () => {
    const empty = [];
    
    assert.throws(() => {
        wasm.trendflex_js(empty, 20);
    }, /No data provided|NoDataProvided/);
});

test('trendflex_reinput', () => {
    const close = testData.close;
    
    
    const firstResult = wasm.trendflex_js(close, 20);
    assert.equal(firstResult.length, close.length);
    
    
    const secondResult = wasm.trendflex_js(firstResult, 10);
    assert.equal(secondResult.length, firstResult.length);
    
    
    if (secondResult.length > 240) {
        for (let i = 240; i < secondResult.length; i++) {
            assert.ok(!isNaN(secondResult[i]), `Unexpected NaN at index ${i}`);
        }
    }
});

test('trendflex_nan_handling', () => {
    const close = testData.close;
    
    const result = wasm.trendflex_js(close, 20);
    assert.equal(result.length, close.length);
    
    
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert.ok(!isNaN(result[i]), `Unexpected NaN at index ${i}`);
        }
    }
    
    
    const firstValid = 0; 
    const warmup = firstValid + 20;
    
    
    assertAllNaN(result.slice(0, warmup), `Expected NaN in warmup period [0:${warmup})`);
    
    assert.ok(!isNaN(result[warmup]), `Expected valid value at index ${warmup}`);
});

test('trendflex_batch_single_param', () => {
    const close = testData.close;
    
    const result = wasm.trendflex_batch_js(close, 20, 20, 0);
    const metadata = wasm.trendflex_batch_metadata_js(20, 20, 0);
    
    
    assert.equal(metadata.length, 1);
    assert.equal(metadata[0], 20);
    
    
    assert.equal(result.length, close.length);
    
    
    const expected = EXPECTED_OUTPUTS.trendflex.last5Values;
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected,
        1e-8,
        "TrendFlex batch last 5 values mismatch"
    );
    
    
    const singleResult = wasm.trendflex_js(close, 20);
    assertArrayClose(
        result,
        singleResult,
        1e-10,
        "Batch vs single calculation mismatch"
    );
});

test('trendflex_batch_multiple_periods', () => {
    const close = testData.close.slice(0, 100); 
    
    const result = wasm.trendflex_batch_js(close, 10, 30, 10);
    const metadata = wasm.trendflex_batch_metadata_js(10, 30, 10);
    
    
    assert.equal(metadata.length, 3);
    assert.deepEqual(Array.from(metadata), [10, 20, 30]);
    
    
    assert.equal(result.length, 3 * close.length);
    
    
    const periods = [10, 20, 30];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * close.length;
        const rowEnd = rowStart + close.length;
        const rowData = result.slice(rowStart, rowEnd);
        
        const singleResult = wasm.trendflex_js(close, periods[i]);
        assertArrayClose(
            rowData,
            singleResult,
            1e-10,
            `Period ${periods[i]} mismatch`
        );
        
        
        const warmup = periods[i]; 
        assertAllNaN(rowData.slice(0, warmup), `Expected NaN in warmup [0:${warmup}) for period=${periods[i]}`);
        assert.ok(!isNaN(rowData[warmup]), `Expected valid value at index ${warmup} for period=${periods[i]}`);
    }
});

test('trendflex_all_nan_input', () => {
    const allNan = new Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.trendflex_js(allNan, 20);
    }, /All values are NaN|AllValuesNaN/);
});


test('trendflex_batch_ergonomic_single_parameter', () => {
    const close = testData.close;
    
    
    const batchResult = wasm.trendflex_batch(close, {
        period_range: [20, 20, 0]
    });
    
    
    const singleResult = wasm.trendflex_js(close, 20);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('trendflex_batch_ergonomic_multiple_periods', () => {
    const close = testData.close.slice(0, 100);
    
    
    const batchResult = wasm.trendflex_batch(close, {
        period_range: [10, 30, 10]      
    });
    
    
    assert.strictEqual(batchResult.values.length, 3 * 100);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 100);
    
    
    assert.strictEqual(batchResult.combos.length, 3);
    assert.strictEqual(batchResult.combos[0].period, 10);
    assert.strictEqual(batchResult.combos[1].period, 20);
    assert.strictEqual(batchResult.combos[2].period, 30);
});

test('trendflex_batch_edge_cases', () => {
    const close = new Float64Array(10);
    close.fill(100);
    
    
    const singleBatch = wasm.trendflex_batch(close, {
        period_range: [5, 5, 1]
    });
    
    assert.strictEqual(singleBatch.values.length, 10);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    
    const largeBatch = wasm.trendflex_batch(close, {
        period_range: [5, 7, 10] 
    });
    
    
    assert.strictEqual(largeBatch.values.length, 10);
    assert.strictEqual(largeBatch.combos.length, 1);
    
    
    assert.throws(() => {
        wasm.trendflex_batch(new Float64Array([]), {
            period_range: [20, 20, 0]
        });
    }, /All values are NaN/);
});


test('trendflex_zero_copy_basic', () => {
    
    const data = new Float64Array(testData.close.slice(0, 100));
    const period = 20;
    
    
    const ptr = wasm.trendflex_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    
    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        data.length
    );
    
    
    memView.set(data);
    
    
    try {
        wasm.trendflex_into(ptr, ptr, data.length, period);
        
        
        
        const updatedMemView = new Float64Array(wasm.__wasm.memory.buffer, ptr, data.length);
        
        
        const regularResult = wasm.trendflex_js(data, period);
        for (let i = 0; i < data.length; i++) {
            
            const memValue = updatedMemView[i];
            const regValue = regularResult[i];
            
            if ((isNaN(regValue) || regValue === undefined) && 
                (isNaN(memValue) || memValue === undefined)) {
                continue; 
            }
            
            assert(Math.abs(regValue - memValue) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regValue}, zerocopy=${memValue}`);
        }
    } finally {
        
        wasm.trendflex_free(ptr, data.length);
    }
});

test('trendflex_zero_copy_error_handling', () => {
    
    assert.throws(() => {
        wasm.trendflex_into(0, 0, 10, 20);
    }, /null pointer|invalid memory/i);
    
    
    const ptr = wasm.trendflex_alloc(10);
    try {
        
        assert.throws(() => {
            wasm.trendflex_into(ptr, ptr, 10, 0);
        }, /Invalid period/);
        
        
        assert.throws(() => {
            wasm.trendflex_into(ptr, ptr, 10, 20);
        }, /Invalid period/);
    } finally {
        wasm.trendflex_free(ptr, 10);
    }
});

test('trendflex_batch_into', () => {
    const data = new Float64Array(testData.close.slice(0, 100));
    const period_start = 10;
    const period_end = 30;
    const period_step = 10;
    
    
    const expected_combos = 3;
    const total_size = expected_combos * data.length;
    
    
    const in_ptr = wasm.trendflex_alloc(data.length);
    const out_ptr = wasm.trendflex_alloc(total_size);
    
    try {
        
        const memory = wasm.__wasm.memory.buffer;
        const inView = new Float64Array(memory, in_ptr, data.length);
        inView.set(data);
        
        
        const n_combos = wasm.trendflex_batch_into(
            in_ptr, out_ptr, data.length,
            period_start, period_end, period_step
        );
        
        assert.strictEqual(n_combos, expected_combos);
        
        
        const memory2 = wasm.__wasm.memory.buffer;
        const outView = new Float64Array(memory2, out_ptr, total_size);
        
        
        const regularBatch = wasm.trendflex_batch(data, {
            period_range: [period_start, period_end, period_step]
        });
        
        assertArrayClose(
            Array.from(outView),
            regularBatch.values,
            1e-10,
            "Batch into vs regular batch mismatch"
        );
    } finally {
        wasm.trendflex_free(in_ptr, data.length);
        wasm.trendflex_free(out_ptr, total_size);
    }
});



test('trendflex_zero_copy_large_dataset', () => {
    const size = 100000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }
    
    const ptr = wasm.trendflex_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');
    
    try {
        const memory = wasm.__wasm.memory.buffer;
        const memView = new Float64Array(memory, ptr, size);
        memView.set(data);
        
        wasm.trendflex_into(ptr, ptr, size, 20);
        
        
        const memory2 = wasm.__wasm.memory.buffer;
        const memView2 = new Float64Array(memory2, ptr, size);
        
        
        for (let i = 0; i < 20; i++) {
            assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
        }
        
        
        for (let i = 20; i < Math.min(100, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.trendflex_free(ptr, size);
    }
});

test('trendflex_memory_management', () => {
    
    const sizes = [100, 1000, 10000, 50000];
    
    for (const size of sizes) {
        const ptr = wasm.trendflex_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        
        const memory = wasm.__wasm.memory.buffer;
        const memView = new Float64Array(memory, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }
        
        
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
        }
        
        
        wasm.trendflex_free(ptr, size);
    }
});

test('trendflex_simd128_consistency', () => {
    
    
    const testCases = [
        { size: 10, period: 5 },
        { size: 100, period: 20 },
        { size: 1000, period: 30 },
        { size: 5000, period: 50 }
    ];
    
    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = Math.sin(i * 0.1) + Math.cos(i * 0.05);
        }
        
        const result = wasm.trendflex_js(data, testCase.period);
        
        
        assert.strictEqual(result.length, data.length);
        
        
        for (let i = 0; i < testCase.period; i++) {
            assert(isNaN(result[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}`);
        }
        
        
        let sumAfterWarmup = 0;
        let countAfterWarmup = 0;
        for (let i = testCase.period; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for size=${testCase.size}`);
            sumAfterWarmup += result[i];
            countAfterWarmup++;
        }
        
        
        const avgAfterWarmup = sumAfterWarmup / countAfterWarmup;
        assert(Math.abs(avgAfterWarmup) < 10, `Average value ${avgAfterWarmup} seems unreasonable`);
    }
});

test('trendflex_batch_metadata_comprehensive', () => {
    const close = new Float64Array(50);
    close.fill(100);
    
    const result = wasm.trendflex_batch(close, {
        period_range: [10, 20, 5]  
    });
    
    
    assert.strictEqual(result.combos.length, 3);
    assert.strictEqual(result.rows, 3);
    assert.strictEqual(result.cols, 50);
    
    
    const expectedPeriods = [10, 15, 20];
    for (let i = 0; i < expectedPeriods.length; i++) {
        assert.strictEqual(result.combos[i].period, expectedPeriods[i]);
    }
    
    
    for (let i = 0; i < result.rows; i++) {
        const rowStart = i * result.cols;
        const rowEnd = rowStart + result.cols;
        const rowData = result.values.slice(rowStart, rowEnd);
        
        
        const period = result.combos[i].period;
        for (let j = 0; j < period; j++) {
            assert(isNaN(rowData[j]), `Expected NaN at index ${j} for period ${period}`);
        }
        
        
        if (period < rowData.length) {
            assert(!isNaN(rowData[period]), `Expected valid value at index ${period} for period ${period}`);
        }
    }
});

test('trendflex_invalid_period_comprehensive', () => {
    const data = new Float64Array([1, 2, 3, 4, 5]);
    
    
    assert.throws(() => {
        wasm.trendflex_js(data, 0);
    }, /period = 0|ZeroTrendFlexPeriod/, 'Should reject period=0');
    
    
    assert.throws(() => {
        wasm.trendflex_js(data, 10);
    }, /period > data len|TrendFlexPeriodExceedsData/, 'Should reject period > length');
    
    
    assert.throws(() => {
        wasm.trendflex_js(data, data.length);
    }, /period > data len|TrendFlexPeriodExceedsData/, 'Should reject period = length');
});

test('trendflex_warmup_calculation_comprehensive', () => {
    
    const testPeriods = [5, 10, 20, 30];
    const close = testData.close.slice(0, 100); 
    
    for (const period of testPeriods) {
        if (period >= close.length) continue;
        
        const result = wasm.trendflex_js(close, period);
        
        
        const firstValid = 0;
        const warmup = firstValid + period;
        
        
        for (let i = 0; i < warmup; i++) {
            assert(isNaN(result[i]), `Expected NaN at index ${i} for period=${period}`);
        }
        
        
        if (warmup < result.length) {
            assert(!isNaN(result[warmup]), `Expected valid value at index ${warmup} for period=${period}`);
        }
    }
});
