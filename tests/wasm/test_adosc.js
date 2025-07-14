const { describe, it, before } = require('mocha');
const { expect } = require('chai');
const path = require('path');

// Load the WASM module
let wasm;
before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/my_project_bg.wasm');
    wasm = require('../../pkg/my_project.js');
    await wasm.default(wasmPath);
});

// Test data loader
const { loadTestData, assertClose, EXPECTED_OUTPUTS } = require('./test_utils');

describe('ADOSC WASM Bindings', () => {
    let testData;
    
    before(() => {
        testData = loadTestData();
    });

    describe('adosc_js', () => {
        it('should calculate ADOSC with default parameters', () => {
            const { high, low, close, volume } = testData;
            const result = wasm.adosc_js(high, low, close, volume, 3, 10);
            
            expect(result).to.be.an('array');
            expect(result).to.have.lengthOf(close.length);
            
            // All values should be finite
            result.forEach((val, i) => {
                expect(val, `Value at index ${i} should be finite`).to.be.finite;
            });
        });

        it('should match expected values from Rust tests', () => {
            const { high, low, close, volume } = testData;
            const expected = EXPECTED_OUTPUTS.adosc;
            
            const result = wasm.adosc_js(
                high, low, close, volume,
                expected.default_params.short_period,
                expected.default_params.long_period
            );
            
            // Check last 5 values match expected with tolerance
            const last5 = result.slice(-5);
            assertClose(last5, expected.last_5_values, 1e-1, 'ADOSC last 5 values mismatch');
        });

        it('should fail with zero period', () => {
            const high = [10.0, 10.0, 10.0];
            const low = [5.0, 5.0, 5.0];
            const close = [7.0, 7.0, 7.0];
            const volume = [1000.0, 1000.0, 1000.0];
            
            // Zero short period
            expect(() => wasm.adosc_js(high, low, close, volume, 0, 10))
                .to.throw(/Invalid period/);
            
            // Zero long period
            expect(() => wasm.adosc_js(high, low, close, volume, 3, 0))
                .to.throw(/Invalid period/);
        });

        it('should fail when short period >= long period', () => {
            const high = [10.0, 11.0, 12.0, 13.0, 14.0];
            const low = [5.0, 5.5, 6.0, 6.5, 7.0];
            const close = [7.0, 8.0, 9.0, 10.0, 11.0];
            const volume = [1000.0, 1000.0, 1000.0, 1000.0, 1000.0];
            
            // short = long
            expect(() => wasm.adosc_js(high, low, close, volume, 3, 3))
                .to.throw(/short_period must be less than long_period/);
            
            // short > long
            expect(() => wasm.adosc_js(high, low, close, volume, 5, 3))
                .to.throw(/short_period must be less than long_period/);
        });

        it('should fail when period exceeds data length', () => {
            const high = [10.0, 11.0, 12.0];
            const low = [5.0, 5.5, 6.0];
            const close = [7.0, 8.0, 9.0];
            const volume = [1000.0, 1000.0, 1000.0];
            
            expect(() => wasm.adosc_js(high, low, close, volume, 3, 10))
                .to.throw(/Invalid period/);
        });

        it('should fail with empty input', () => {
            const empty = [];
            
            expect(() => wasm.adosc_js(empty, empty, empty, empty, 3, 10))
                .to.throw(/empty/);
        });

        it('should handle zero volume correctly', () => {
            const high = [10.0, 11.0, 12.0, 13.0, 14.0];
            const low = [5.0, 5.5, 6.0, 6.5, 7.0];
            const close = [7.0, 8.0, 9.0, 10.0, 11.0];
            const volume = [0.0, 0.0, 0.0, 0.0, 0.0]; // All zero volume
            
            const result = wasm.adosc_js(high, low, close, volume, 2, 3);
            expect(result).to.have.lengthOf(close.length);
            
            // With zero volume, ADOSC should be 0
            result.forEach((val, i) => {
                expect(val, `Value at index ${i} should be 0`).to.equal(0);
            });
        });

        it('should handle constant price correctly', () => {
            const price = 10.0;
            const high = Array(10).fill(price);
            const low = Array(10).fill(price);
            const close = Array(10).fill(price);
            const volume = Array(10).fill(1000.0);
            
            const result = wasm.adosc_js(high, low, close, volume, 3, 5);
            expect(result).to.have.lengthOf(close.length);
            
            // With constant price (high = low), MFM is 0, so ADOSC should be 0
            result.forEach((val, i) => {
                expect(val, `Value at index ${i} should be 0`).to.equal(0);
            });
        });

        it('should calculate from the first value (no warmup period)', () => {
            const { high, low, close, volume } = testData;
            const result = wasm.adosc_js(high, low, close, volume, 3, 10);
            
            // First value should not be NaN (ADOSC calculates from the start)
            expect(result[0], 'First ADOSC value should not be NaN').to.not.be.NaN;
        });
    });

    describe('adosc_batch_js', () => {
        it('should calculate batch ADOSC with default parameters', () => {
            const { high, low, close, volume } = testData;
            
            const result = wasm.adosc_batch_js(
                high, low, close, volume,
                3, 3, 0,   // short_period range (single value)
                10, 10, 0  // long_period range (single value)
            );
            
            expect(result).to.be.an('array');
            expect(result).to.have.lengthOf(close.length); // 1 combination
        });

        it('should calculate batch ADOSC with parameter sweep', () => {
            const { high, low, close, volume } = testData;
            
            const result = wasm.adosc_batch_js(
                high, low, close, volume,
                2, 5, 1,   // short_period: 2, 3, 4, 5
                8, 12, 2   // long_period: 8, 10, 12
            );
            
            // Valid combinations where short < long
            const validCount = [2, 3, 4, 5].reduce((count, s) => {
                return count + [8, 10, 12].filter(l => s < l).length;
            }, 0);
            
            expect(result).to.be.an('array');
            expect(result).to.have.lengthOf(validCount * close.length);
        });

        it('should match single calculation for default parameters', () => {
            const { high, low, close, volume } = testData;
            
            const singleResult = wasm.adosc_js(high, low, close, volume, 3, 10);
            const batchResult = wasm.adosc_batch_js(
                high, low, close, volume,
                3, 3, 0,
                10, 10, 0
            );
            
            // Batch result should match single calculation
            assertClose(batchResult, singleResult, 1e-9, 'Batch vs single calculation mismatch');
        });
    });

    describe('adosc_batch_metadata_js', () => {
        it('should return correct metadata for single parameter', () => {
            const metadata = wasm.adosc_batch_metadata_js(3, 3, 0, 10, 10, 0);
            
            expect(metadata).to.be.an('array');
            expect(metadata).to.have.lengthOf(2); // 1 combination * 2 values (short, long)
            expect(metadata[0]).to.equal(3);  // short_period
            expect(metadata[1]).to.equal(10); // long_period
        });

        it('should return correct metadata for parameter sweep', () => {
            const metadata = wasm.adosc_batch_metadata_js(
                2, 5, 1,   // short_period: 2, 3, 4, 5
                8, 12, 2   // long_period: 8, 10, 12
            );
            
            expect(metadata).to.be.an('array');
            
            // Check all combinations are present and valid
            let idx = 0;
            for (let s = 2; s <= 5; s++) {
                for (let l = 8; l <= 12; l += 2) {
                    if (s < l) {
                        expect(metadata[idx], `Short period at index ${idx}`).to.equal(s);
                        expect(metadata[idx + 1], `Long period at index ${idx + 1}`).to.equal(l);
                        idx += 2;
                    }
                }
            }
        });
    });

    describe('adosc_batch (unified API)', () => {
        it('should calculate batch ADOSC with config object', () => {
            const { high, low, close, volume } = testData;
            
            const config = {
                short_period_range: [3, 3, 0],
                long_period_range: [10, 10, 0]
            };
            
            const result = wasm.adosc_batch(high, low, close, volume, config);
            
            expect(result).to.be.an('object');
            expect(result.values).to.be.an('array');
            expect(result.combos).to.be.an('array');
            expect(result.rows).to.equal(1);
            expect(result.cols).to.equal(close.length);
            
            // Check combo structure
            expect(result.combos[0]).to.have.property('short_period', 3);
            expect(result.combos[0]).to.have.property('long_period', 10);
        });

        it('should handle parameter sweep with unified API', () => {
            const { high, low, close, volume } = testData;
            
            const config = {
                short_period_range: [2, 5, 1],  // 2, 3, 4, 5
                long_period_range: [8, 12, 2]   // 8, 10, 12
            };
            
            const result = wasm.adosc_batch(high, low, close, volume, config);
            
            // Valid combinations where short < long
            const validCombos = result.combos.filter(c => 
                c.short_period < c.long_period
            );
            
            expect(result.rows).to.equal(validCombos.length);
            expect(result.values).to.have.lengthOf(result.rows * result.cols);
        });

        it('should reject invalid config', () => {
            const { high, low, close, volume } = testData;
            
            // Missing required fields
            const badConfig = { short_period_range: [3, 3, 0] };
            
            expect(() => wasm.adosc_batch(high, low, close, volume, badConfig))
                .to.throw(/Invalid config/);
        });
    });

    describe('Error handling', () => {
        it('should handle all error types from AdoscError enum', () => {
            const { high, low, close, volume } = testData;
            
            // EmptySlices
            expect(() => wasm.adosc_js([], [], [], [], 3, 10))
                .to.throw(/empty/);
            
            // InvalidPeriod
            expect(() => wasm.adosc_js(high, low, close, volume, 0, 10))
                .to.throw(/Invalid period/);
            
            // ShortPeriodGreaterThanLong
            expect(() => wasm.adosc_js(high, low, close, volume, 10, 3))
                .to.throw(/short_period must be less than long_period/);
            
            // NotEnoughValidData (period exceeds length)
            const shortData = high.slice(0, 5);
            expect(() => wasm.adosc_js(shortData, shortData, shortData, shortData, 3, 10))
                .to.throw(/Invalid period|Not enough valid data/);
        });
    });
});