import test from 'node:test';
import assert from 'node:assert';
import * as wasm from '../../pkg/my_project.js';
import { loadTestData, assertClose, EXPECTED_OUTPUTS } from './test_utils.js';

test.describe('LPC WASM Tests', () => {
    const testData = loadTestData();
    const expected = EXPECTED_OUTPUTS.lpc;
    
    test.it('should calculate LPC correctly with default parameters', () => {
        const result = wasm.lpc_wasm(
            testData.high,
            testData.low,
            testData.close,
            testData.close, // Using close as source
            expected.default_params.cutoff_type,
            expected.default_params.fixed_period,
            expected.default_params.max_cycle_limit,
            expected.default_params.cycle_mult,
            expected.default_params.tr_mult
        );
        
        assert(result.filter, 'Result should have filter array');
        assert(result.high_band, 'Result should have high_band array');
        assert(result.low_band, 'Result should have low_band array');
        assert.strictEqual(result.filter.length, testData.close.length, 'Filter length should match input');
        assert.strictEqual(result.high_band.length, testData.close.length, 'High band length should match input');
        assert.strictEqual(result.low_band.length, testData.close.length, 'Low band length should match input');
        
        // Check last 5 values match expected
        const filterLast5 = result.filter.slice(-5);
        const highLast5 = result.high_band.slice(-5);
        const lowLast5 = result.low_band.slice(-5);
        
        // Check all three outputs with tight tolerance
        assertClose(filterLast5, expected.last_5_filter, 1e-8, 'LPC Filter last 5 values');
        assertClose(highLast5, expected.last_5_high_band, 1e-8, 'LPC High Band last 5 values');
        assertClose(lowLast5, expected.last_5_low_band, 1e-8, 'LPC Low Band last 5 values');
    });
    
    test.it('should work with fixed cutoff type', () => {
        const result = wasm.lpc_wasm(
            testData.high.slice(0, 100),
            testData.low.slice(0, 100),
            testData.close.slice(0, 100),
            testData.close.slice(0, 100),
            'fixed',
            20,
            60,
            1.0,
            1.0
        );
        
        assert(result.filter, 'Result should have filter array');
        assert(result.high_band, 'Result should have high_band array');
        assert(result.low_band, 'Result should have low_band array');
        assert.strictEqual(result.filter.length, 100, 'Output length should be 100');
        
        // Verify bands relationship
        for (let i = 20; i < 100; i++) {
            if (!isNaN(result.filter[i]) && !isNaN(result.high_band[i]) && !isNaN(result.low_band[i])) {
                assert(result.low_band[i] <= result.filter[i], 
                    `Filter should be >= low band at index ${i}`);
                assert(result.filter[i] <= result.high_band[i], 
                    `Filter should be <= high band at index ${i}`);
            }
        }
    });
    
    test.it('should work with adaptive cutoff type', () => {
        const result = wasm.lpc_wasm(
            testData.high.slice(0, 100),
            testData.low.slice(0, 100),
            testData.close.slice(0, 100),
            testData.close.slice(0, 100),
            'adaptive',
            20,
            60,
            1.0,
            1.0
        );
        
        assert(result.filter, 'Result should have filter array');
        assert(result.high_band, 'Result should have high_band array');
        assert(result.low_band, 'Result should have low_band array');
        assert.strictEqual(result.filter.length, 100, 'Output length should be 100');
    });
    
    test.it('should produce different results for adaptive vs fixed', () => {
        const dataSlice = 200;
        const adaptiveResult = wasm.lpc_wasm(
            testData.high.slice(0, dataSlice),
            testData.low.slice(0, dataSlice),
            testData.close.slice(0, dataSlice),
            testData.close.slice(0, dataSlice),
            'adaptive',
            20,
            60,
            1.0,
            1.0
        );
        
        const fixedResult = wasm.lpc_wasm(
            testData.high.slice(0, dataSlice),
            testData.low.slice(0, dataSlice),
            testData.close.slice(0, dataSlice),
            testData.close.slice(0, dataSlice),
            'fixed',
            20,
            60,
            1.0,
            1.0
        );
        
        // Count differences
        let differences = 0;
        for (let i = 100; i < dataSlice; i++) {
            if (!isNaN(adaptiveResult.filter[i]) && !isNaN(fixedResult.filter[i])) {
                if (Math.abs(adaptiveResult.filter[i] - fixedResult.filter[i]) > 1e-6) {
                    differences++;
                }
            }
        }
        
        assert(differences > 10, 'Adaptive and fixed modes should produce different results');
    });
    
    test.it('should handle different multiplier values', () => {
        const dataSlice = 100;
        
        // Test with cycle_mult = 2.0
        const result1 = wasm.lpc_wasm(
            testData.high.slice(0, dataSlice),
            testData.low.slice(0, dataSlice),
            testData.close.slice(0, dataSlice),
            testData.close.slice(0, dataSlice),
            'adaptive',
            20,
            60,
            2.0, // cycle_mult
            1.0  // tr_mult
        );
        
        // Test with tr_mult = 2.0
        const result2 = wasm.lpc_wasm(
            testData.high.slice(0, dataSlice),
            testData.low.slice(0, dataSlice),
            testData.close.slice(0, dataSlice),
            testData.close.slice(0, dataSlice),
            'adaptive',
            20,
            60,
            1.0, // cycle_mult
            2.0  // tr_mult
        );
        
        // Check that tr_mult affects band width
        const checkIdx = 50;
        if (!isNaN(result1.high_band[checkIdx]) && !isNaN(result2.high_band[checkIdx])) {
            const bandWidth1 = result1.high_band[checkIdx] - result1.low_band[checkIdx];
            const bandWidth2 = result2.high_band[checkIdx] - result2.low_band[checkIdx];
            assert(bandWidth2 > bandWidth1 * 1.5, 
                'Higher tr_mult should produce wider bands');
        }
    });
    
    test.it('should handle empty input', () => {
        assert.throws(() => {
            wasm.lpc_wasm([], [], [], [], 'adaptive', 20, 60, 1.0, 1.0);
        }, /empty/i, 'Should throw error for empty input');
    });
    
    test.it('should handle invalid cutoff type', () => {
        const data = [1, 2, 3, 4, 5];
        assert.throws(() => {
            wasm.lpc_wasm(data, data, data, data, 'invalid', 20, 60, 1.0, 1.0);
        }, /Invalid cutoff type/i, 'Should throw error for invalid cutoff type');
    });
    
    test.it('should handle invalid period', () => {
        const data = [1, 2, 3];
        assert.throws(() => {
            wasm.lpc_wasm(data, data, data, data, 'fixed', 0, 60, 1.0, 1.0);
        }, /Invalid period/i, 'Should throw error for zero period');
    });
    
    test.it('should handle period exceeding data length', () => {
        const data = [1, 2, 3];
        assert.throws(() => {
            wasm.lpc_wasm(data, data, data, data, 'fixed', 10, 60, 1.0, 1.0);
        }, /Invalid period/i, 'Should throw error when period exceeds data length');
    });
    
    test.it('should handle mismatched array lengths', () => {
        const high = [1, 2, 3];
        const low = [1, 2];  // Different length
        const close = [1, 2, 3];
        const src = [1, 2, 3];
        
        assert.throws(() => {
            wasm.lpc_wasm(high, low, close, src, 'fixed', 2, 60, 1.0, 1.0);
        }, /missing|mismatch/i, 'Should throw error for mismatched lengths');
    });
    
    test.it('should handle all NaN input with error', () => {
        const nanData = new Array(10).fill(NaN);
        assert.throws(() => {
            wasm.lpc_wasm(
                nanData,
                nanData,
                nanData,
                nanData,
                'fixed',
                5,
                60,
                1.0,
                1.0
            );
        }, /All values are NaN/i, 'Should throw error for all NaN input');
    });
    
    test.it('should calculate channel bands correctly', () => {
        const result = wasm.lpc_wasm(
            testData.high.slice(0, 100),
            testData.low.slice(0, 100),
            testData.close.slice(0, 100),
            testData.close.slice(0, 100),
            'fixed',
            10,
            60,
            1.0,
            1.0
        );
        
        // After warmup, verify the bands make sense
        for (let i = 20; i < 100; i++) {
            if (!isNaN(result.filter[i]) && !isNaN(result.high_band[i]) && !isNaN(result.low_band[i])) {
                // High band should be above low band
                assert(result.high_band[i] > result.low_band[i], 
                    `High band should be above low band at index ${i}`);
                
                // Filter should be between bands
                assert(result.filter[i] >= result.low_band[i] && result.filter[i] <= result.high_band[i],
                    `Filter should be between bands at index ${i}`);
                
                // Band width should be reasonable (not too wide)
                const bandWidth = result.high_band[i] - result.low_band[i];
                const filterValue = result.filter[i];
                assert(bandWidth < filterValue * 2, 
                    `Band width should be reasonable at index ${i}`);
            }
        }
    });
    
    test.it('should handle warmup period correctly', () => {
        const result = wasm.lpc_wasm(
            testData.high,
            testData.low,
            testData.close,
            testData.close,
            'fixed',
            20,
            60,
            1.0,
            1.0
        );
        
        // LPC starts computing from the first valid data point
        // Unlike some indicators, it doesn't enforce a warmup period of NaN values
        // unless the input data itself has NaN values
        
        // After sufficient data (e.g., index 240), no NaN values should exist
        if (result.filter.length > 240) {
            for (let i = 240; i < result.filter.length; i++) {
                assert(!isNaN(result.filter[i]), `Found unexpected NaN in filter at index ${i}`);
                assert(!isNaN(result.high_band[i]), `Found unexpected NaN in high_band at index ${i}`);
                assert(!isNaN(result.low_band[i]), `Found unexpected NaN in low_band at index ${i}`);
            }
        }
        
        // Verify that bands maintain proper relationship throughout
        for (let i = 20; i < Math.min(100, result.filter.length); i++) {
            if (!isNaN(result.filter[i]) && !isNaN(result.high_band[i]) && !isNaN(result.low_band[i])) {
                assert(result.low_band[i] <= result.filter[i], 
                    `Filter should be >= low band at index ${i}`);
                assert(result.filter[i] <= result.high_band[i], 
                    `Filter should be <= high band at index ${i}`);
            }
        }
    });
    
    test.it('should test lpc flattened output format', () => {
        // lpc returns a single flattened array with [filter, high_band, low_band] concatenated
        const flatResult = wasm.lpc(
            testData.high.slice(0, 100),
            testData.low.slice(0, 100),
            testData.close.slice(0, 100),
            testData.close.slice(0, 100),
            'fixed',
            20,
            60,
            1.0,
            1.0
        );
        
        assert(flatResult instanceof Float64Array || Array.isArray(flatResult), 'lpc should return an array or Float64Array');
        assert.strictEqual(flatResult.length, 100 * 3, 'Flattened array should have length = input_length * 3');
        
        // Extract the three components
        const filterValues = flatResult.slice(0, 100);
        const highBandValues = flatResult.slice(100, 200);
        const lowBandValues = flatResult.slice(200, 300);
        
        // Verify the values match lpc_wasm output
        const structuredResult = wasm.lpc_wasm(
            testData.high.slice(0, 100),
            testData.low.slice(0, 100),
            testData.close.slice(0, 100),
            testData.close.slice(0, 100),
            'fixed',
            20,
            60,
            1.0,
            1.0
        );
        
        assertClose(filterValues, structuredResult.filter, 1e-10, 'Filter values should match');
        assertClose(highBandValues, structuredResult.high_band, 1e-10, 'High band values should match');
        assertClose(lowBandValues, structuredResult.low_band, 1e-10, 'Low band values should match');
    });
    
    test.it('should test batch processing with single parameter combination', () => {
        const result = wasm.lpc_batch(
            testData.high,
            testData.low,
            testData.close,
            testData.close,
            {
                fixed_period_range: [10, 12, 1],  // 10, 11, 12
                cycle_mult_range: [1.0, 1.0, 0.0],  // single value
                tr_mult_range: [1.0, 1.0, 0.0],  // single value
                cutoff_type: 'fixed',
                max_cycle_limit: 60
            }
        );
        
        assert(result.values, 'Batch result should have values field');
        assert(result.fixed_periods, 'Batch result should have fixed_periods field');
        assert(result.cycle_mults, 'Batch result should have cycle_mults field');
        assert(result.tr_mults, 'Batch result should have tr_mults field');
        assert(result.rows, 'Batch result should have rows field');
        assert(result.cols, 'Batch result should have cols field');
        assert(result.order, 'Batch result should have order field');
        
        // Check dimensions
        const combos = 3;  // 10, 11, 12
        const expectedRows = combos * 3;  // 3 outputs per combo (filter, high, low)
        assert.strictEqual(result.rows, expectedRows, `Expected ${expectedRows} rows`);
        assert.strictEqual(result.cols, testData.close.length, 'Columns should match input length');
        
        // Verify values shape
        assert.strictEqual(result.values.length, expectedRows, 'Values should have correct number of rows');
        assert.strictEqual(result.values[0].length, testData.close.length, 'Each row should match input length');
        
        // Verify order field
        assert.deepStrictEqual(result.order, ['filter', 'high', 'low'], 
            'Order should indicate filter, high, low outputs');
        
        // Verify parameter arrays
        assert.strictEqual(result.fixed_periods.length, combos, 'Should have correct number of period values');
        assert.strictEqual(result.cycle_mults.length, combos, 'Should have correct number of cycle_mult values');
        assert.strictEqual(result.tr_mults.length, combos, 'Should have correct number of tr_mult values');
    });
    
    test.it('should test batch processing with parameter sweep', () => {
        const result = wasm.lpc_batch(
            testData.high.slice(0, 100),
            testData.low.slice(0, 100),
            testData.close.slice(0, 100),
            testData.close.slice(0, 100),
            {
                fixed_period_range: [10, 20, 10],  // 10, 20
                cycle_mult_range: [1.0, 2.0, 1.0],  // 1.0, 2.0
                tr_mult_range: [0.5, 1.0, 0.5],  // 0.5, 1.0
                cutoff_type: 'adaptive',
                max_cycle_limit: 60
            }
        );
        
        // 2 periods * 2 cycle_mults * 2 tr_mults = 8 combinations
        const expectedCombos = 8;
        const expectedRows = expectedCombos * 3;  // 3 outputs per combo
        
        assert.strictEqual(result.rows, expectedRows, 
            `Expected ${expectedRows} rows for ${expectedCombos} combos`);
        assert.strictEqual(result.fixed_periods.length, expectedCombos, 
            `Expected ${expectedCombos} period values`);
        assert.strictEqual(result.cycle_mults.length, expectedCombos, 
            `Expected ${expectedCombos} cycle_mult values`);
        assert.strictEqual(result.tr_mults.length, expectedCombos, 
            `Expected ${expectedCombos} tr_mult values`);
        
        // Verify all combinations are present
        const expectedPeriods = [10, 10, 10, 10, 20, 20, 20, 20];
        const expectedCycleMults = [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0];
        const expectedTrMults = [0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0];
        
        for (let i = 0; i < expectedCombos; i++) {
            assert.strictEqual(result.fixed_periods[i], expectedPeriods[i], 
                `Period mismatch at combo ${i}`);
            assert.strictEqual(result.cycle_mults[i], expectedCycleMults[i], 
                `Cycle mult mismatch at combo ${i}`);
            assert.strictEqual(result.tr_mults[i], expectedTrMults[i], 
                `TR mult mismatch at combo ${i}`);
        }
    });
    
    test.it.skip('should test batch processing with raw memory (lpc_batch_into)', () => {
        // Skip this test - lpc_batch_into expects raw pointers which JavaScript arrays can't provide directly
        // The high-level lpc_batch API is sufficient for JavaScript users
    });
});