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

test('EHMA accuracy', () => {
    
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
    
    
    for (let i = 0; i < 13; i++) {
        assert(isNaN(result[i]), `Value at index ${i} should be NaN`);
    }
    
    
    for (let i = 13; i < result.length; i++) {
        assert(!isNaN(result[i]), `Value at index ${i} should not be NaN`);
        assert(isFinite(result[i]), `Value at index ${i} should be finite`);
    }
    
    
    const expectedValueAt13 = expected.expectedValueAt13 || 59309.748;
    const actual13 = result[13];
    assertClose(
        actual13, 
        expectedValueAt13,
        0.001,
        `EHMA value at index 13`
    );
    
    
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
    
    const expected = EXPECTED_OUTPUTS.ehma || {};
    const defaultPeriod = expected.defaultParams?.period || 14;
    const warmupPeriod = expected.warmupPeriod || (defaultPeriod - 1);
    
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const result = wasm.ehma_wasm(close, defaultPeriod);
    assert.strictEqual(result.length, close.length);
    
    
    for (let i = 0; i < warmupPeriod; i++) {
        assert(isNaN(result[i]), `Value at index ${i} should be NaN`);
    }
    
    
    for (let i = warmupPeriod; i < result.length; i++) {
        assert(!isNaN(result[i]), `Value at index ${i} should not be NaN`);
    }
});

test('EHMA empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.ehma_wasm(empty, 14);
    }, /Input data slice is empty/);
});

test('EHMA all NaN', () => {
    
    const nanData = new Float64Array([NaN, NaN, NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.ehma_wasm(nanData, 5);
    }, /All values are NaN/);
});

test('EHMA invalid period', () => {
    
    const data = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0]);
    
    
    assert.throws(() => {
        wasm.ehma_wasm(data, 0);
    }, /Invalid period/);
    
    
    assert.throws(() => {
        wasm.ehma_wasm(data, 10);
    }, /Invalid period/);
});

test('EHMA not enough valid data', () => {
    
    
    const data = new Float64Array(15);
    for (let i = 0; i < 10; i++) {
        data[i] = NaN;
    }
    for (let i = 10; i < 15; i++) {
        data[i] = i - 9; 
    }
    
    assert.throws(() => {
        wasm.ehma_wasm(data, 14);
    }, /Not enough valid data/);
});

test('EHMA different periods', () => {
    
    const expected = EXPECTED_OUTPUTS.ehma || {};
    const batchPeriods = expected.batchPeriods || [10, 14, 20, 28];
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const results = {};
    for (const period of batchPeriods) {
        results[period] = wasm.ehma_wasm(close, period);
    }
    
    
    for (const [period, result] of Object.entries(results)) {
        assert.strictEqual(result.length, close.length, 
            `Result for period ${period} has wrong length`);
        
        
        const warmup = period - 1;
        for (let i = 0; i < warmup; i++) {
            assert(isNaN(result[i]), `Period ${period}: result[${i}] should be NaN`);
        }
        assert(!isNaN(result[warmup]), `Period ${period}: result[${warmup}] should not be NaN`);
    }
    
    
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
    
    const data = new Float64Array([
        NaN, NaN, 100.0, 101.0, 102.0,
        103.0, 104.0, 105.0, 106.0, 107.0,
        108.0, 109.0, 110.0, 111.0, 112.0,
        113.0, 114.0, 115.0, 116.0, 117.0
    ]);
    
    const result = wasm.ehma_wasm(data, 10);
    
    assert.strictEqual(result.length, data.length);
    
    
    for (let i = 0; i < 11; i++) {
        assert(isNaN(result[i]), `Value at index ${i} should be NaN`);
    }
    for (let i = 11; i < result.length; i++) {
        assert(!isNaN(result[i]), `Value at index ${i} should not be NaN`);
    }
});

test('EHMA streaming', () => {
    
    const data = new Float64Array([
        59500.0, 59450.0, 59420.0, 59380.0, 59350.0, 
        59320.0, 59310.0, 59300.0, 59280.0, 59260.0,
        59250.0, 59240.0, 59230.0, 59220.0, 59210.0,
        59200.0, 59190.0, 59180.0,
    ]);
    
    
    const stream = new wasm.EhmaWasmStream(14);
    
    
    const streamResults = [];
    for (const value of data) {
        streamResults.push(stream.update(value));
    }
    
    
    const batchResult = wasm.ehma_wasm(data, 14);
    
    
    const streamResultsWithNaN = streamResults.map(v => v === null ? NaN : v);
    
    
    assertArrayClose(
        streamResultsWithNaN,
        Array.from(batchResult),
        1e-10,
        "Stream and batch results should match"
    );
    
    
    stream.reset();
    
    const firstAfterReset = stream.update(100.0);
    assert(firstAfterReset === null, "First value after reset should be null");
    
    
    const streamAfterReset = [];
    for (let i = 0; i < data.length; i++) {
        streamAfterReset.push(stream.update(data[i]));
    }
    
    
    const streamAfterResetWithNaN = streamAfterReset.map(v => v === null ? NaN : v);
    
    
    assertArrayClose(
        streamAfterResetWithNaN,
        Array.from(batchResult),
        1e-10,
        "Stream after reset should match batch results"
    );
});

test('EHMA with real market data', () => {
    
    const expected = EXPECTED_OUTPUTS.ehma || {};
    const batchPeriods = expected.batchPeriods || [10, 14, 20, 28];
    const close = new Float64Array(testData.close.slice(0, 500));
    
    
    const results = {};
    for (const period of batchPeriods) {
        results[period] = wasm.ehma_wasm(close, period);
    }
    
    
    const minVal = Math.min(...close);
    const maxVal = Math.max(...close);
    const tolerance = (maxVal - minVal) * 0.1;
    
    for (const [period, result] of Object.entries(results)) {
        assert.strictEqual(result.length, close.length, 
            `Period ${period}: Result length mismatch`);
        
        
        
        let validCount = 0;
        const warmup = period - 1;
        
        for (let i = warmup; i < result.length; i++) {
            if (!isNaN(result[i])) {
                validCount++;
                assert(isFinite(result[i]), `Period ${period}: result[${i}] should be finite`);
                
                
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
    
    const expected = EXPECTED_OUTPUTS.ehma || {};
    const batchRange = expected.batchRange || [10, 30, 10];
    const close = new Float64Array(testData.close.slice(0, 100));
    
    
    if (wasm.ehma_batch_wasm) {
        const batchResult = wasm.ehma_batch_wasm(
            close, 
            batchRange[0],  
            batchRange[1],  
            batchRange[2]   
        );
        
        
        assert(batchResult.periods, "Batch result should have periods");
        assert(batchResult.values, "Batch result should have values");
        
        
        const expectedPeriods = [];
        for (let p = batchRange[0]; p <= batchRange[1]; p += batchRange[2]) {
            expectedPeriods.push(p);
        }
        
        assert.strictEqual(batchResult.periods.length, expectedPeriods.length,
            "Number of periods should match expected");
        
        
        for (let i = 0; i < expectedPeriods.length; i++) {
            const period = expectedPeriods[i];
            const row = batchResult.values[i];
            const warmup = period - 1;
            
            
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
    
    const expected = EXPECTED_OUTPUTS.ehma || {};
    const defaultPeriod = expected.defaultParams?.period || 14;
    const close = new Float64Array(testData.close.slice(0, 200));
    
    
    const firstResult = wasm.ehma_wasm(close, defaultPeriod);
    assert.strictEqual(firstResult.length, close.length);
    
    
    const secondResult = wasm.ehma_wasm(firstResult, defaultPeriod);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    
    const firstWarmup = defaultPeriod - 1;
    const secondWarmup = firstWarmup + defaultPeriod - 1;
    
    
    for (let i = 0; i < firstWarmup; i++) {
        assert(isNaN(firstResult[i]), `First pass: Value at ${i} should be NaN`);
    }
    
    
    for (let i = 0; i < secondWarmup; i++) {
        assert(isNaN(secondResult[i]), `Second pass: Value at ${i} should be NaN`);
    }
    
    
    for (let i = secondWarmup; i < Math.min(secondWarmup + 5, secondResult.length); i++) {
        assert(!isNaN(secondResult[i]), `Second pass: Value at ${i} should not be NaN`);
    }
});