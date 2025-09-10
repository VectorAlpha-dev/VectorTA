/**
 * WASM binding tests for EHMA (Ehlers Hann Moving Average) indicator.
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
    // Load WASM module
    try {
        const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
        const importPath = process.platform === 'win32' 
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
        wasm = await import(importPath);
        // No need to call default() for ES modules
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('EHMA accuracy', () => {
    // Test EHMA calculation produces consistent values
    const expected = EXPECTED_OUTPUTS.ehma || {};
    const data = new Float64Array(
        expected.testData || [
            59500.0, 59450.0, 59420.0, 59380.0, 59350.0, 
            59320.0, 59310.0, 59300.0, 59280.0, 59260.0,
            59250.0, 59240.0, 59230.0, 59220.0, 59210.0,
            59200.0, 59190.0, 59180.0,
        ]
    );
    
    const result = wasm.ehma_wasm(data, 14);
    
    assert.strictEqual(result.length, data.length, "Result length should match input length");
    
    // Check first 13 values are NaN (warmup period)
    for (let i = 0; i < 13; i++) {
        assert(isNaN(result[i]), `Value at index ${i} should be NaN`);
    }
    
    // Check values from index 13 onwards are valid
    for (let i = 13; i < result.length; i++) {
        assert(!isNaN(result[i]), `Value at index ${i} should not be NaN`);
        assert(isFinite(result[i]), `Value at index ${i} should be finite`);
    }
    
    // Verify specific calculation for consistency
    const expectedValueAt13 = expected.expectedValueAt13 || 59309.748;
    const actual13 = result[13];
    assertClose(
        actual13, 
        expectedValueAt13,
        0.001,
        `EHMA value at index 13`
    );
    
    // Check that values are within reasonable range
    const minVal = Math.min(...data);
    const maxVal = Math.max(...data);
    const tolerance = (maxVal - minVal) * 0.1;
    
    for (let i = 13; i < result.length; i++) {
        assert(
            result[i] >= minVal - tolerance && result[i] <= maxVal + tolerance,
            `Value ${result[i]} at index ${i} is outside reasonable range`
        );
    }
});

test('EHMA default period', () => {
    // Test EHMA with default period
    const expected = EXPECTED_OUTPUTS.ehma || {};
    const defaultPeriod = expected.defaultParams?.period || 14;
    const warmupPeriod = expected.warmupPeriod || (defaultPeriod - 1);
    
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const result = wasm.ehma_wasm(close, defaultPeriod);
    assert.strictEqual(result.length, close.length);
    
    // Check warmup period
    for (let i = 0; i < warmupPeriod; i++) {
        assert(isNaN(result[i]), `Value at index ${i} should be NaN`);
    }
    
    // Values from warmup period onwards should be valid
    for (let i = warmupPeriod; i < result.length; i++) {
        assert(!isNaN(result[i]), `Value at index ${i} should not be NaN`);
    }
});

test('EHMA empty input', () => {
    // Test EHMA fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.ehma_wasm(empty, 14);
    }, /Input data slice is empty/);
});

test('EHMA all NaN', () => {
    // Test EHMA fails with all NaN values
    const nanData = new Float64Array([NaN, NaN, NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.ehma_wasm(nanData, 5);
    }, /All values are NaN/);
});

test('EHMA invalid period', () => {
    // Test EHMA fails with invalid period
    const data = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0]);
    
    // Period zero
    assert.throws(() => {
        wasm.ehma_wasm(data, 0);
    }, /Invalid period/);
    
    // Period larger than data
    assert.throws(() => {
        wasm.ehma_wasm(data, 10);
    }, /Invalid period/);
});

test('EHMA not enough valid data', () => {
    // Test EHMA fails when not enough valid data
    // First 10 values are NaN, only 5 valid values for period=14
    const data = new Float64Array(15);
    for (let i = 0; i < 10; i++) {
        data[i] = NaN;
    }
    for (let i = 10; i < 15; i++) {
        data[i] = i - 9; // 1.0, 2.0, 3.0, 4.0, 5.0
    }
    
    assert.throws(() => {
        wasm.ehma_wasm(data, 14);
    }, /Not enough valid data/);
});

test('EHMA different periods', () => {
    // Test EHMA with different period values
    const expected = EXPECTED_OUTPUTS.ehma || {};
    const batchPeriods = expected.batchPeriods || [10, 14, 20, 28];
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const results = {};
    for (const period of batchPeriods) {
        results[period] = wasm.ehma_wasm(close, period);
    }
    
    // All should have same length as input
    for (const [period, result] of Object.entries(results)) {
        assert.strictEqual(result.length, close.length, 
            `Result for period ${period} has wrong length`);
        
        // Check warmup period
        const warmup = period - 1;
        for (let i = 0; i < warmup; i++) {
            assert(isNaN(result[i]), `Period ${period}: result[${i}] should be NaN`);
        }
        assert(!isNaN(result[warmup]), `Period ${period}: result[${warmup}] should not be NaN`);
    }
    
    // Results should be different with different periods
    if (batchPeriods.length >= 2) {
        const maxWarmup = Math.max(...batchPeriods) - 1;
        for (let i = 0; i < batchPeriods.length - 1; i++) {
            const p1 = batchPeriods[i];
            const p2 = batchPeriods[i + 1];
            const r1 = results[p1];
            const r2 = results[p2];
            
            let different = false;
            for (let j = maxWarmup; j < maxWarmup + 10 && j < close.length; j++) {
                if (Math.abs(r1[j] - r2[j]) > 1e-10) {
                    different = true;
                    break;
                }
            }
            assert(different, `Results with periods ${p1} and ${p2} should be different`);
        }
    }
});

test('EHMA with NaN values', () => {
    // Test EHMA handles some NaN values correctly
    const data = new Float64Array([
        NaN, NaN, 100.0, 101.0, 102.0,
        103.0, 104.0, 105.0, 106.0, 107.0,
        108.0, 109.0, 110.0, 111.0, 112.0,
        113.0, 114.0, 115.0, 116.0, 117.0
    ]);
    
    const result = wasm.ehma_wasm(data, 10);
    
    assert.strictEqual(result.length, data.length);
    // Should handle leading NaNs and start calculating when enough valid data
    // First valid index is 2, so warmup ends at 2 + 10 - 1 = 11
    for (let i = 0; i < 11; i++) {
        assert(isNaN(result[i]), `Value at index ${i} should be NaN`);
    }
    for (let i = 11; i < result.length; i++) {
        assert(!isNaN(result[i]), `Value at index ${i} should not be NaN`);
    }
});

test('EHMA streaming', () => {
    // Test EHMA streaming functionality
    const data = new Float64Array([
        59500.0, 59450.0, 59420.0, 59380.0, 59350.0, 
        59320.0, 59310.0, 59300.0, 59280.0, 59260.0,
        59250.0, 59240.0, 59230.0, 59220.0, 59210.0,
        59200.0, 59190.0, 59180.0,
    ]);
    
    // Create stream with period 14
    const stream = new wasm.EhmaWasmStream(14);
    
    // Process data through stream
    const streamResults = [];
    for (const value of data) {
        streamResults.push(stream.update(value));
    }
    
    // Compare with batch calculation
    const batchResult = wasm.ehma_wasm(data, 14);
    
    // Convert null to NaN for comparison
    const streamResultsWithNaN = streamResults.map(v => v === null ? NaN : v);
    
    // Results should match
    assertArrayClose(
        streamResultsWithNaN,
        Array.from(batchResult),
        1e-10,
        "Stream and batch results should match"
    );
    
    // Test reset
    stream.reset();
    // After reset, first value should be null again
    const firstAfterReset = stream.update(100.0);
    assert(firstAfterReset === null, "First value after reset should be null");
    
    // Test that stream continues to work after reset
    const streamAfterReset = [];
    for (let i = 0; i < data.length; i++) {
        streamAfterReset.push(stream.update(data[i]));
    }
    
    // Convert null to NaN for comparison
    const streamAfterResetWithNaN = streamAfterReset.map(v => v === null ? NaN : v);
    
    // Should match original batch result
    assertArrayClose(
        streamAfterResetWithNaN,
        Array.from(batchResult),
        1e-10,
        "Stream after reset should match batch results"
    );
});

test('EHMA with real market data', () => {
    // Test EHMA with real market data
    const expected = EXPECTED_OUTPUTS.ehma || {};
    const batchPeriods = expected.batchPeriods || [10, 14, 20, 28];
    const close = new Float64Array(testData.close.slice(0, 500));
    
    // Test with different periods
    const results = {};
    for (const period of batchPeriods) {
        results[period] = wasm.ehma_wasm(close, period);
    }
    
    // Check each period result
    const minVal = Math.min(...close);
    const maxVal = Math.max(...close);
    const tolerance = (maxVal - minVal) * 0.1;
    
    for (const [period, result] of Object.entries(results)) {
        assert.strictEqual(result.length, close.length, 
            `Period ${period}: Result length mismatch`);
        
        // EHMA should smooth the data
        // Check that the indicator produces reasonable values
        let validCount = 0;
        const warmup = period - 1;
        
        for (let i = warmup; i < result.length; i++) {
            if (!isNaN(result[i])) {
                validCount++;
                assert(isFinite(result[i]), `Period ${period}: result[${i}] should be finite`);
                
                // Values should be within reasonable range
                assert(
                    result[i] >= minVal - tolerance && result[i] <= maxVal + tolerance,
                    `Period ${period}: Value ${result[i]} at index ${i} is outside reasonable range`
                );
            }
        }
        
        assert(validCount > 0, `Should have valid values for period ${period}`);
    }
});

test('EHMA consistency', () => {
    // Test that running EHMA multiple times produces same results
    const expected = EXPECTED_OUTPUTS.ehma || {};
    const defaultPeriod = expected.defaultParams?.period || 14;
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const result1 = wasm.ehma_wasm(close, defaultPeriod);
    const result2 = wasm.ehma_wasm(close, defaultPeriod);
    const result3 = wasm.ehma_wasm(close, defaultPeriod);
    
    assertArrayClose(
        Array.from(result1),
        Array.from(result2),
        1e-15,
        "First and second run should produce identical results"
    );
    
    assertArrayClose(
        Array.from(result2),
        Array.from(result3),
        1e-15,
        "Second and third run should produce identical results"
    );
});

test('EHMA batch processing', () => {
    // Test EHMA batch processing with multiple periods
    const expected = EXPECTED_OUTPUTS.ehma || {};
    const batchRange = expected.batchRange || [10, 30, 10];
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Test batch function if available
    if (wasm.ehma_batch_wasm) {
        const batchResult = wasm.ehma_batch_wasm(
            close, 
            batchRange[0],  // start
            batchRange[1],  // stop
            batchRange[2]   // step
        );
        
        // Verify structure
        assert(batchResult.periods, "Batch result should have periods");
        assert(batchResult.values, "Batch result should have values");
        
        // Check periods
        const expectedPeriods = [];
        for (let p = batchRange[0]; p <= batchRange[1]; p += batchRange[2]) {
            expectedPeriods.push(p);
        }
        
        assert.strictEqual(batchResult.periods.length, expectedPeriods.length,
            "Number of periods should match expected");
        
        // Each row should have proper warmup
        for (let i = 0; i < expectedPeriods.length; i++) {
            const period = expectedPeriods[i];
            const row = batchResult.values[i];
            const warmup = period - 1;
            
            // Check NaN pattern
            for (let j = 0; j < warmup; j++) {
                assert(isNaN(row[j]), 
                    `Period ${period}: Value at index ${j} should be NaN`);
            }
            for (let j = warmup; j < Math.min(warmup + 5, row.length); j++) {
                assert(!isNaN(row[j]), 
                    `Period ${period}: Value at index ${j} should not be NaN`);
            }
        }
    }
});

test('EHMA reinput', () => {
    // Test EHMA applied to its own output
    const expected = EXPECTED_OUTPUTS.ehma || {};
    const defaultPeriod = expected.defaultParams?.period || 14;
    const close = new Float64Array(testData.close.slice(0, 200));
    
    // First pass
    const firstResult = wasm.ehma_wasm(close, defaultPeriod);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass - apply EHMA to EHMA output
    const secondResult = wasm.ehma_wasm(firstResult, defaultPeriod);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // Check that warmup period compounds
    const firstWarmup = defaultPeriod - 1;
    const secondWarmup = firstWarmup + defaultPeriod - 1;
    
    // First pass should have NaN only in first warmup period
    for (let i = 0; i < firstWarmup; i++) {
        assert(isNaN(firstResult[i]), `First pass: Value at ${i} should be NaN`);
    }
    
    // Second pass should have extended warmup
    for (let i = 0; i < secondWarmup; i++) {
        assert(isNaN(secondResult[i]), `Second pass: Value at ${i} should be NaN`);
    }
    
    // Values after warmup should be valid
    for (let i = secondWarmup; i < Math.min(secondWarmup + 5, secondResult.length); i++) {
        assert(!isNaN(secondResult[i]), `Second pass: Value at ${i} should not be NaN`);
    }
});