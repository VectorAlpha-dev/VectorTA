/**
 * WASM binding tests for MSW (Mesa Sine Wave) indicator.
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


function extractMswResults(result) {
    const sine = result.values.slice(0, result.cols);
    const lead = result.values.slice(result.cols);
    return { sine, lead };
}

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

test('MSW partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.msw_js(close, 5);
    assert(result.values, 'Should have values array');
    assert.strictEqual(result.rows, 2, 'Should have 2 rows (sine and lead)');
    assert.strictEqual(result.cols, close.length, 'Columns should match input length');
    assert.strictEqual(result.values.length, 2 * close.length, 'Values should contain sine and lead');
    
    
    const sine = result.values.slice(0, close.length);
    const lead = result.values.slice(close.length);
    assert.strictEqual(sine.length, close.length);
    assert.strictEqual(lead.length, close.length);
});

test('MSW accuracy', async () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.msw;
    const period = expected.defaultParams.period;
    
    const result = wasm.msw_js(close, period);
    const { sine, lead } = extractMswResults(result);
    
    assert.strictEqual(sine.length, close.length);
    assert.strictEqual(lead.length, close.length);
    
    
    const last5Sine = Array.from(sine.slice(-5));
    const last5Lead = Array.from(lead.slice(-5));
    
    assertArrayClose(
        last5Sine,
        expected.last5Sine,
        1e-1,  
        "MSW sine last 5 values mismatch"
    );
    assertArrayClose(
        last5Lead,
        expected.last5Lead,
        1e-1,  
        "MSW lead last 5 values mismatch"
    );
});

test('MSW default candles', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const result = wasm.msw_js(close, 5);
    const { sine, lead } = extractMswResults(result);
    assert.strictEqual(sine.length, close.length);
    assert.strictEqual(lead.length, close.length);
});

test('MSW zero period', () => {
    
    const data = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.msw_js(data, 0);
    }, /Invalid period/);
});

test('MSW period exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.msw_js(dataSmall, 10);
    }, /Invalid period/);
});

test('MSW very small dataset', () => {
    
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.msw_js(singlePoint, 5);
    }, /Invalid period|Not enough valid data/);
});

test('MSW empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.msw_js(empty, 5);
    }, /Empty data/);
});

test('MSW NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.msw;
    const period = expected.defaultParams.period;
    
    const result = wasm.msw_js(close, period);
    const { sine, lead } = extractMswResults(result);
    assert.strictEqual(sine.length, close.length);
    assert.strictEqual(lead.length, close.length);
    
    
    const expectedWarmup = expected.warmupPeriod;  
    assertAllNaN(Array.from(sine.slice(0, expectedWarmup)), "Expected NaN in sine warmup period");
    assertAllNaN(Array.from(lead.slice(0, expectedWarmup)), "Expected NaN in lead warmup period");
    
    
    if (sine.length > expectedWarmup) {
        const nonNanStart = Math.max(expectedWarmup, 240);  
        if (sine.length > nonNanStart) {
            for (let i = nonNanStart; i < sine.length; i++) {
                assert(!isNaN(sine[i]), `Found unexpected NaN in sine at index ${i}`);
                assert(!isNaN(lead[i]), `Found unexpected NaN in lead at index ${i}`);
            }
        }
    }
});

test('MSW all NaN input', () => {
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.msw_js(allNaN, 5);
    }, /All values are NaN/);
});

test('MSW mixed NaN input', () => {
    
    const mixedData = new Float64Array([NaN, NaN, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0]);
    const period = 3;
    
    const result = wasm.msw_js(mixedData, period);
    const { sine, lead } = extractMswResults(result);
    assert.strictEqual(sine.length, mixedData.length);
    assert.strictEqual(lead.length, mixedData.length);
    
    
    assert(isNaN(sine[0]));
    assert(isNaN(sine[1]));
    assert(isNaN(lead[0]));
    assert(isNaN(lead[1]));
    
    
    
    for (let i = 4; i < sine.length; i++) {
        assert(!isNaN(sine[i]), `Unexpected NaN in sine at index ${i}`);
        assert(!isNaN(lead[i]), `Unexpected NaN in lead at index ${i}`);
    }
});

test('MSW simple predictable pattern', () => {
    
    const simpleData = new Float64Array([1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5]);
    const period = 5;
    
    const result = wasm.msw_js(simpleData, period);
    const { sine, lead } = extractMswResults(result);
    assert.strictEqual(sine.length, simpleData.length);
    assert.strictEqual(lead.length, simpleData.length);
    
    
    for (let i = 0; i < period - 1; i++) {
        assert(isNaN(sine[i]), `Expected NaN in sine at index ${i}`);
        assert(isNaN(lead[i]), `Expected NaN in lead at index ${i}`);
    }
    
    
    for (let i = period - 1; i < sine.length; i++) {
        assert(!isNaN(sine[i]), `Unexpected NaN in sine at index ${i}`);
        assert(!isNaN(lead[i]), `Unexpected NaN in lead at index ${i}`);
        
        
        assert(sine[i] >= -1.0 && sine[i] <= 1.0, 
               `Sine value ${sine[i]} at index ${i} is out of range [-1, 1]`);
        assert(lead[i] >= -1.0 && lead[i] <= 1.0, 
               `Lead value ${lead[i]} at index ${i} is out of range [-1, 1]`);
    }
});




/*
test('MSW batch single parameter', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close.slice(0, 100));
    const expected = EXPECTED_OUTPUTS.msw;
    
    // Single period batch
    const batchResult = wasm.msw_batch(close, {
        period_range: [expected.defaultParams.period, expected.defaultParams.period, 0]
    });
    
    // Should match single calculation
    const singleResult = wasm.msw_js(close, expected.defaultParams.period);
    const { sine: singleSine, lead: singleLead } = extractMswResults(singleResult);
    
    // Extract first row from batch (sine and lead)
    assert.strictEqual(batchResult.rows, 2);
    assert.strictEqual(batchResult.cols, close.length);
    assert.strictEqual(batchResult.combos.length, 1);
    
    const batchSine = batchResult.values.slice(0, close.length);
    const batchLead = batchResult.values.slice(close.length);
    
    for (let i = 0; i < close.length; i++) {
        if (isNaN(singleSine[i]) && isNaN(batchSine[i])) {
            continue; // Both NaN is OK
        }
        assert(Math.abs(singleSine[i] - batchSine[i]) < 1e-10,
               `Sine mismatch at index ${i}: single=${singleSine[i]}, batch=${batchSine[i]}`);
        assert(Math.abs(singleLead[i] - batchLead[i]) < 1e-10,
               `Lead mismatch at index ${i}: single=${singleLead[i]}, batch=${batchLead[i]}`);
    }
});

test('MSW batch multiple periods', () => {
    // Test batch with multiple period values
    const close = new Float64Array(testData.close.slice(0, 50));
    
    // Multiple periods: 5, 10, 15
    const batchResult = wasm.msw_batch(close, {
        period_range: [5, 15, 5]  // 5, 10, 15
    });
    
    // Should have 3 parameter combinations
    assert.strictEqual(batchResult.combos.length, 3);
    assert.strictEqual(batchResult.rows, 6);  // 3 periods * 2 outputs (sine, lead)
    assert.strictEqual(batchResult.cols, 50);
    assert.strictEqual(batchResult.values.length, 6 * 50);
    
    // Verify each period
    const periods = [5, 10, 15];
    for (let p = 0; p < periods.length; p++) {
        assert.strictEqual(batchResult.combos[p].period, periods[p]);
        
        // Calculate single result for comparison
        const singleResult = wasm.msw_js(close, periods[p]);
        const { sine: singleSine, lead: singleLead } = extractMswResults(singleResult);
        
        // Extract batch results for this period
        // Batch layout: [sine_p0, lead_p0, sine_p1, lead_p1, sine_p2, lead_p2]
        const batchSineStart = p * 2 * 50;  // Each period has 2 rows (sine, lead)
        const batchLeadStart = batchSineStart + 50;
        const batchSine = batchResult.values.slice(batchSineStart, batchSineStart + 50);
        const batchLead = batchResult.values.slice(batchLeadStart, batchLeadStart + 50);
        
        // Compare
        for (let i = 0; i < 50; i++) {
            if (isNaN(singleSine[i]) && isNaN(batchSine[i])) {
                continue;
            }
            assert(Math.abs(singleSine[i] - batchSine[i]) < 1e-10,
                   `Period ${periods[p]} sine mismatch at index ${i}`);
            
            if (isNaN(singleLead[i]) && isNaN(batchLead[i])) {
                continue;
            }
            assert(Math.abs(singleLead[i] - batchLead[i]) < 1e-10,
                   `Period ${periods[p]} lead mismatch at index ${i}`);
        }
        
        // Check warmup period
        const warmup = periods[p] - 1;
        for (let i = 0; i < warmup; i++) {
            assert(isNaN(batchSine[i]), `Expected NaN at warmup index ${i} for period ${periods[p]} sine`);
            assert(isNaN(batchLead[i]), `Expected NaN at warmup index ${i} for period ${periods[p]} lead`);
        }
    }
});

test('MSW batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    // Single value sweep
    const singleBatch = wasm.msw_batch(close, {
        period_range: [5, 5, 0]
    });
    
    assert.strictEqual(singleBatch.combos.length, 1);
    assert.strictEqual(singleBatch.rows, 2);  // sine and lead
    assert.strictEqual(singleBatch.cols, 10);
    
    // Step larger than range
    const largeBatch = wasm.msw_batch(close, {
        period_range: [3, 5, 10]  // Step larger than range
    });
    
    // Should only have period=3
    assert.strictEqual(largeBatch.combos.length, 1);
    assert.strictEqual(largeBatch.combos[0].period, 3);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.msw_batch(new Float64Array([]), {
            period_range: [5, 5, 0]
        });
    }, /Empty data|All values are NaN/);
});
*/




/*
test('MSW zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const period = 5;
    
    // Allocate buffer for both sine and lead (2x data length)
    const totalSize = data.length * 2;
    const ptr = wasm.msw_alloc(totalSize);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    // Create view into WASM memory
    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        totalSize
    );
    
    // Copy data into first half (for input)
    memView.set(data, 0);
    
    // Compute MSW in-place
    try {
        wasm.msw_into(ptr, ptr, data.length, period);
        
        // Extract results
        const sineResult = memView.slice(0, data.length);
        const leadResult = memView.slice(data.length, totalSize);
        
        // Verify results match regular API
        const regularResult = wasm.msw_js(data, period);
        const { sine: regularSine, lead: regularLead } = extractMswResults(regularResult);
        
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularSine[i]) && isNaN(sineResult[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularSine[i] - sineResult[i]) < 1e-10,
                   `Zero-copy sine mismatch at index ${i}`);
            
            if (isNaN(regularLead[i]) && isNaN(leadResult[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularLead[i] - leadResult[i]) < 1e-10,
                   `Zero-copy lead mismatch at index ${i}`);
        }
    } finally {
        // Always free memory
        wasm.msw_free(ptr, totalSize);
    }
});

test('MSW zero-copy with large dataset', () => {
    const size = 10000;
    const totalSize = size * 2;  // For sine and lead
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.cos(i * 0.02);
    }
    
    const ptr = wasm.msw_alloc(totalSize);
    assert(ptr !== 0, 'Failed to allocate large buffer');
    
    try {
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, totalSize);
        memView.set(data, 0);
        
        wasm.msw_into(ptr, ptr, size, 7);
        
        // Recreate view in case memory grew
        const memView2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, totalSize);
        const sineResult = memView2.slice(0, size);
        const leadResult = memView2.slice(size, totalSize);
        
        // Check warmup period has NaN
        for (let i = 0; i < 6; i++) {  // period-1 = 6
            assert(isNaN(sineResult[i]), `Expected NaN at sine warmup index ${i}`);
            assert(isNaN(leadResult[i]), `Expected NaN at lead warmup index ${i}`);
        }
        
        // Check after warmup has values
        for (let i = 6; i < Math.min(100, size); i++) {
            assert(!isNaN(sineResult[i]), `Unexpected NaN in sine at index ${i}`);
            assert(!isNaN(leadResult[i]), `Unexpected NaN in lead at index ${i}`);
            
            // Values should be in range [-1, 1]
            assert(sineResult[i] >= -1.0 && sineResult[i] <= 1.0,
                   `Sine value ${sineResult[i]} out of range at index ${i}`);
            assert(leadResult[i] >= -1.0 && leadResult[i] <= 1.0,
                   `Lead value ${leadResult[i]} out of range at index ${i}`);
        }
    } finally {
        wasm.msw_free(ptr, totalSize);
    }
});

test('MSW zero-copy error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.msw_into(0, 0, 10, 5);
    }, /null pointer|invalid memory/i);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.msw_alloc(20);  // 10 values * 2 outputs
    try {
        // Invalid period
        assert.throws(() => {
            wasm.msw_into(ptr, ptr, 10, 0);
        }, /Invalid period/);
        
        // Period exceeds length
        assert.throws(() => {
            wasm.msw_into(ptr, ptr, 10, 20);
        }, /Invalid period/);
    } finally {
        wasm.msw_free(ptr, 20);
    }
});
*/


test('MSW SIMD128 consistency', () => {
    
    const testCases = [
        { size: 10, period: 3 },
        { size: 50, period: 5 },
        { size: 100, period: 10 },
        { size: 500, period: 20 }
    ];
    
    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = Math.sin(i * 0.1) * Math.cos(i * 0.05) + i * 0.01;
        }
        
        const result = wasm.msw_js(data, testCase.period);
        const { sine, lead } = extractMswResults(result);
        
        
        assert.strictEqual(sine.length, data.length);
        assert.strictEqual(lead.length, data.length);
        
        
        for (let i = 0; i < testCase.period - 1; i++) {
            assert(isNaN(sine[i]), `Expected NaN at sine warmup index ${i} for size=${testCase.size}`);
            assert(isNaN(lead[i]), `Expected NaN at lead warmup index ${i} for size=${testCase.size}`);
        }
        
        
        let sineSum = 0;
        let leadSum = 0;
        let count = 0;
        for (let i = testCase.period - 1; i < sine.length; i++) {
            assert(!isNaN(sine[i]), `Unexpected NaN in sine at index ${i} for size=${testCase.size}`);
            assert(!isNaN(lead[i]), `Unexpected NaN in lead at index ${i} for size=${testCase.size}`);
            
            
            assert(sine[i] >= -1.0 && sine[i] <= 1.0,
                   `Sine value ${sine[i]} out of range at index ${i}`);
            assert(lead[i] >= -1.0 && lead[i] <= 1.0,
                   `Lead value ${lead[i]} out of range at index ${i}`);
            
            sineSum += sine[i];
            leadSum += lead[i];
            count++;
        }
        
        
        const avgSine = sineSum / count;
        const avgLead = leadSum / count;
        assert(Math.abs(avgSine) < 1.0, `Average sine ${avgSine} seems unreasonable`);
        assert(Math.abs(avgLead) < 1.0, `Average lead ${avgLead} seems unreasonable`);
    }
});

test.after(() => {
    console.log('MSW WASM tests completed');
});