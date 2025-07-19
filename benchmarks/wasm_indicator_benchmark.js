#!/usr/bin/env node
/**
 * Generic WASM Indicator Performance Benchmark
 * 
 * Designed to benchmark multiple indicators with consistent methodology.
 * Supports two main API patterns: Safe/Simple and Fast/Unsafe.
 * 
 * To add a new indicator:
 * 1. Add its configuration to the INDICATORS object below
 * 2. Ensure the WASM bindings follow the standard naming patterns:
 *    - Safe API: indicator_js(data, ...params) -> Vec<f64>
 *    - Fast API: indicator_alloc/free/into functions
 *    - Batch API (optional): indicator_batch(data, config) -> BatchResult
 * 3. Run: node --expose-gc wasm_indicator_benchmark.js indicator_name
 */

import { performance } from 'perf_hooks';
import { readFileSync, writeFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Benchmark configuration
const CONFIG = {
    warmupTargetMs: 150,    // 150ms warmup period
    sampleCount: 10,        // Number of samples to collect
    minIterations: 10,      // Minimum iterations per sample
    disableGC: true,        // Disable GC during measurement
};

// Data sizes to benchmark
const DATA_SIZES = {
    '10k': 10_000,
    '100k': 100_000,
    '1M': 1_000_000,
};

/**
 * Indicator configurations
 * Add new indicators here by following the pattern
 */
const INDICATORS = {
    frama: {
        name: 'FRAMA',
        // Safe API
        safe: {
            fn: 'frama_js',
            params: { window: 10, sc: 300, fc: 1 }
        },
        needsMultipleInputs: true,
        // Fast/Unsafe API
        fast: {
            allocFn: 'frama_alloc',
            freeFn: 'frama_free',
            computeFn: 'frama_into',
            params: { window: 10, sc: 300, fc: 1 },
            needsMultipleInputs: true
        },
        // Batch API
        batch: {
            fn: 'frama_batch',
            fastFn: 'frama_batch_into',
            config: {
                small: {
                    window_range: [8, 12, 2],      // 3 values
                    sc_range: [200, 300, 100],     // 2 values
                    fc_range: [1, 2, 1]            // 2 values = 12 combinations
                },
                medium: {
                    window_range: [6, 14, 2],      // 5 values
                    sc_range: [100, 400, 100],    // 4 values
                    fc_range: [1, 3, 1]            // 3 values = 60 combinations
                }
            }
        }
    },
    pwma: {
        name: 'PWMA',
        // Safe API
        safe: {
            fn: 'pwma_js',
            params: { period: 14 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'pwma_alloc',
            freeFn: 'pwma_free',
            computeFn: 'pwma_into',
            params: { period: 14 }
        },
        // Batch API
        batch: {
            fn: 'pwma_batch_js',
            fastFn: 'pwma_batch_into',
            config: {
                small: {
                    period_range: [5, 15, 2]       // 6 values
                },
                medium: {
                    period_range: [5, 25, 2]       // 11 values
                }
            }
        }
    },
    alma: {
        name: 'ALMA',
        // Safe API
        safe: {
            fn: 'alma_js',
            params: { period: 9, offset: 0.85, sigma: 6.0 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'alma_alloc',
            freeFn: 'alma_free',
            computeFn: 'alma_into',
            params: { period: 9, offset: 0.85, sigma: 6.0 }
        },
        // Batch API (optional)
        batch: {
            fn: 'alma_batch',
            config: {
                // Reduced combinations for faster execution
                small: {
                    period_range: [5, 15, 5],      // 3 values
                    offset_range: [0.7, 0.9, 0.1], // 3 values  
                    sigma_range: [4.0, 8.0, 2.0]   // 3 values
                    // Total: 27 combinations
                },
                medium: {
                    period_range: [5, 25, 4],      // 6 values
                    offset_range: [0.5, 0.9, 0.1], // 5 values
                    sigma_range: [3.0, 9.0, 3.0]   // 3 values
                    // Total: 90 combinations
                }
            },
            // Fast batch API (optional)
            fastFn: 'alma_batch_into'
        }
    },
    edcf: {
        name: 'EDCF',
        // Safe API
        safe: {
            fn: 'edcf_js',
            params: { period: 15 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'edcf_alloc',
            freeFn: 'edcf_free',
            computeFn: 'edcf_into',
            params: { period: 15 }
        },
        // Batch API
        batch: {
            fn: 'edcf_batch',
            config: {
                small: {
                    period_range: [10, 20, 5]  // 3 values: 10, 15, 20
                },
                medium: {
                    period_range: [15, 50, 5]  // 8 values: 15, 20, 25, ..., 50
                }
            }
        }
    },
    highpass: {
        name: 'HighPass',
        // Safe API
        safe: {
            fn: 'highpass_js',
            params: { period: 48 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'highpass_alloc',
            freeFn: 'highpass_free',
            computeFn: 'highpass_into',
            params: { period: 48 }
        },
        // Batch API
        batch: {
            fn: 'highpass_batch',
            config: {
                small: {
                    period_range: [30, 60, 10]  // 4 values: 30, 40, 50, 60
                },
                medium: {
                    period_range: [20, 80, 10]  // 7 values: 20, 30, 40, ..., 80
                }
            },
            // Fast batch API
            fastFn: 'highpass_batch_into'
        }
    },
    jsa: {
        name: 'JSA',
        // Safe API
        safe: {
            fn: 'jsa_js',
            params: { period: 30 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'jsa_alloc',
            freeFn: 'jsa_free',
            computeFn: 'jsa_fast',
            params: { period: 30 }
        },
        // Batch API
        batch: {
            fn: 'jsa_batch',
            fastFn: 'jsa_batch_into',
            config: {
                small: {
                    period_range: [10, 40, 10]  // 4 values: 10, 20, 30, 40
                },
                medium: {
                    period_range: [10, 50, 5]   // 9 values: 10, 15, 20, ..., 50
                }
            }
        }
    },
    maaq: {
        name: 'MAAQ',
        // Safe API
        safe: {
            fn: 'maaq_js',
            params: { period: 11, fast_period: 2, slow_period: 30 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'maaq_alloc',
            freeFn: 'maaq_free',
            computeFn: 'maaq_into',
            params: { period: 11, fast_period: 2, slow_period: 30 }
        },
        // Batch API
        batch: {
            fn: 'maaq_batch_js',
            fastFn: 'maaq_batch_into',
            config: {
                small: {
                    period_range: [10, 20, 5],       // 3 values: 10, 15, 20
                    fast_period_range: [2, 4, 1],    // 3 values: 2, 3, 4
                    slow_period_range: [20, 40, 10]  // 3 values: 20, 30, 40
                    // Total: 27 combinations
                },
                medium: {
                    period_range: [10, 30, 5],       // 5 values: 10, 15, 20, 25, 30
                    fast_period_range: [2, 6, 2],    // 3 values: 2, 4, 6
                    slow_period_range: [20, 50, 10]  // 4 values: 20, 30, 40, 50
                    // Total: 60 combinations
                }
            }
        }
    },
    smma: {
        name: 'SMMA',
        // Safe API
        safe: {
            fn: 'smma',
            params: { period: 7 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'smma_alloc',
            freeFn: 'smma_free',
            computeFn: 'smma_into',
            params: { period: 7 }
        },
        // Batch API
        batch: {
            fn: 'smma_batch_new',
            config: {
                small: {
                    period_range: [5, 15, 5]  // 3 values: 5, 10, 15
                },
                medium: {
                    period_range: [5, 25, 5]  // 5 values: 5, 10, 15, 20, 25
                }
            },
            // Fast batch API
            fastFn: 'smma_batch_into'
        }
    },
    supersmoother: {
        name: 'SuperSmoother',
        // Safe API
        safe: {
            fn: 'supersmoother_js',
            params: { period: 14 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'supersmoother_alloc',
            freeFn: 'supersmoother_free',
            computeFn: 'supersmoother_into',
            params: { period: 14 }
        },
        // Batch API
        batch: {
            fn: 'supersmoother_batch_js',  // Use the old API for benchmarking
            fastFn: 'supersmoother_batch_into',
            config: {
                small: {
                    period_range: [5, 15, 2]       // 6 values
                },
                medium: {
                    period_range: [5, 25, 2]       // 11 values
                }
            }
        }
    },
    // Example: Simple Moving Average (uncomment when SMA WASM bindings are added)
    /*
    sma: {
        name: 'SMA',
        safe: {
            fn: 'sma_js',
            params: { period: 20 }
        },
        fast: {
            allocFn: 'sma_alloc',
            freeFn: 'sma_free',
            computeFn: 'sma_into',
            params: { period: 20 }
        }
    },
    */
    // Example: Exponential Moving Average (uncomment when EMA WASM bindings are added)
    /*
    ema: {
        name: 'EMA',
        safe: {
            fn: 'ema_js',
            params: { period: 20 }
        },
        fast: {
            allocFn: 'ema_alloc',
            freeFn: 'ema_free',
            computeFn: 'ema_into',
            params: { period: 20 }
        }
    },
    */
    // Example: RSI with batch support (uncomment when RSI WASM bindings are added)
    /*
    rsi: {
        name: 'RSI',
        safe: {
            fn: 'rsi_js',
            params: { period: 14 }
        },
        fast: {
            allocFn: 'rsi_alloc',
            freeFn: 'rsi_free',
            computeFn: 'rsi_into',
            params: { period: 14 }
        },
        batch: {
            fn: 'rsi_batch',
            config: {
                small: {
                    period_range: [10, 20, 5]  // 3 values: 10, 15, 20
                },
                medium: {
                    period_range: [10, 30, 2]  // 11 values: 10, 12, 14, ..., 30
                }
            }
        }
    }
    */
};

class WasmIndicatorBenchmark {
    constructor() {
        this.wasm = null;
        this.data = {};
        this.results = {};
    }

    async initialize() {
        // Load WASM module
        console.log('Loading WASM module...');
        try {
            const { createRequire } = await import('module');
            const require = createRequire(import.meta.url);
            const wasmPath = join(__dirname, '../pkg/my_project.js');
            this.wasm = require(wasmPath);
            console.log('WASM module loaded successfully');
        } catch (error) {
            console.error('Failed to load WASM module:', error);
            console.error('Run "wasm-pack build --features wasm --target nodejs" first');
            process.exit(1);
        }

        // Load test data
        this.loadData();
    }

    loadData() {
        console.log('Loading test data...');
        
        const csvPath = join(__dirname, '../src/data/1MillionCandles.csv');
        const content = readFileSync(csvPath, 'utf8');
        const lines = content.trim().split('\n');
        
        // Skip header
        lines.shift();
        
        // Parse OHLC data
        const opens = [];
        const highs = [];
        const lows = [];
        const closes = [];
        
        for (const line of lines) {
            const parts = line.split(',');
            if (parts.length >= 5) {
                opens.push(parseFloat(parts[1]));
                highs.push(parseFloat(parts[2]));
                lows.push(parseFloat(parts[3]));
                closes.push(parseFloat(parts[4]));
            }
        }
        
        // Create different size datasets with single close array and OHLC data
        this.data['10k'] = new Float64Array(closes.slice(0, 10_000));
        this.data['100k'] = new Float64Array(closes.slice(0, 100_000));
        this.data['1M'] = new Float64Array(closes);
        
        // Also store OHLC data for indicators that need it
        this.ohlcData = {
            '10k': {
                open: new Float64Array(opens.slice(0, 10_000)),
                high: new Float64Array(highs.slice(0, 10_000)),
                low: new Float64Array(lows.slice(0, 10_000)),
                close: new Float64Array(closes.slice(0, 10_000))
            },
            '100k': {
                open: new Float64Array(opens.slice(0, 100_000)),
                high: new Float64Array(highs.slice(0, 100_000)),
                low: new Float64Array(lows.slice(0, 100_000)),
                close: new Float64Array(closes.slice(0, 100_000))
            },
            '1M': {
                open: new Float64Array(opens),
                high: new Float64Array(highs),
                low: new Float64Array(lows),
                close: new Float64Array(closes)
            }
        };
        
        console.log(`Loaded data sizes: ${Object.keys(this.data).join(', ')}`);
    }

    /**
     * Generic benchmark function
     */
    benchmarkFunction(fn, name, metadata = {}) {
        const gcWasEnabled = global.gc ? true : false;
        if (CONFIG.disableGC && global.gc) {
            global.gc();
        }

        try {
            // Warmup phase
            let warmupElapsed = 0;
            let warmupIterations = 0;
            const warmupStart = performance.now();
            
            while (warmupElapsed < CONFIG.warmupTargetMs) {
                fn();
                warmupIterations++;
                warmupElapsed = performance.now() - warmupStart;
            }

            // Sampling phase
            const samples = [];
            
            for (let i = 0; i < CONFIG.sampleCount; i++) {
                const iterations = Math.max(CONFIG.minIterations, Math.floor(warmupIterations / 10));
                
                const start = performance.now();
                for (let j = 0; j < iterations; j++) {
                    fn();
                }
                const end = performance.now();
                
                const timePerIteration = (end - start) / iterations;
                samples.push(timePerIteration);
            }

            // Calculate statistics
            samples.sort((a, b) => a - b);
            const median = samples[Math.floor(samples.length / 2)];
            const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
            const min = samples[0];
            const max = samples[samples.length - 1];
            
            // Calculate throughput if data size is known
            const dataSize = metadata.dataSize || 0;
            const throughput = dataSize > 0 ? dataSize / (median * 1000) / 1000 : 0;

            return {
                name,
                median,
                mean,
                min,
                max,
                samples: samples.length,
                warmupIterations,
                throughput,
                ...metadata
            };
        } finally {
            if (CONFIG.disableGC && gcWasEnabled && global.gc) {
                global.gc();
            }
        }
    }

    /**
     * Benchmark safe/simple API for an indicator
     */
    benchmarkSafeAPI(indicatorKey, indicatorConfig) {
        console.log(`\n--- ${indicatorConfig.name} Safe API ---`);
        
        const { fn, params } = indicatorConfig.safe;
        const wasmFn = this.wasm[fn];
        
        if (!wasmFn) {
            console.log(`  Function ${fn} not found, skipping...`);
            return;
        }

        for (const [sizeName, data] of Object.entries(this.data)) {
            const benchName = `${indicatorKey}_safe_${sizeName}`;
            
            const result = this.benchmarkFunction(() => {
                const paramArray = this.prepareParams(params, data, indicatorConfig, sizeName);
                wasmFn.apply(this.wasm, paramArray);
            }, benchName, {
                dataSize: data.length,
                api: 'safe',
                indicator: indicatorKey
            });

            this.results[benchName] = result;
            this.printResult(result);
        }
    }

    /**
     * Benchmark fast/unsafe API for an indicator
     */
    benchmarkFastAPI(indicatorKey, indicatorConfig) {
        console.log(`\n--- ${indicatorConfig.name} Fast/Unsafe API ---`);
        
        const { allocFn, freeFn, computeFn, params } = indicatorConfig.fast;
        
        if (!this.wasm[allocFn] || !this.wasm[freeFn] || !this.wasm[computeFn]) {
            console.log(`  Fast API functions not found, skipping...`);
            return;
        }

        for (const [sizeName, data] of Object.entries(this.data)) {
            const benchName = `${indicatorKey}_fast_${sizeName}`;
            const len = data.length;
            
            let inPtr, outPtr, highPtr, lowPtr, closePtr;
            
            try {
                // Handle multiple inputs if needed
                if (indicatorConfig.fast.needsMultipleInputs) {
                    const ohlc = this.ohlcData[sizeName];
                    
                    // Allocate buffers for high, low, close
                    highPtr = this.wasm[allocFn](len);
                    lowPtr = this.wasm[allocFn](len);
                    closePtr = this.wasm[allocFn](len);
                    outPtr = this.wasm[allocFn](len);
                    
                    // Copy data
                    const highView = new Float64Array(this.wasm.__wasm.memory.buffer, highPtr, len);
                    const lowView = new Float64Array(this.wasm.__wasm.memory.buffer, lowPtr, len);
                    const closeView = new Float64Array(this.wasm.__wasm.memory.buffer, closePtr, len);
                    
                    highView.set(ohlc.high);
                    lowView.set(ohlc.low);
                    closeView.set(ohlc.close);
                    
                    const result = this.benchmarkFunction(() => {
                        const paramArray = this.prepareFastParams(params, null, outPtr, len, indicatorConfig.fast, highPtr, lowPtr, closePtr);
                        this.wasm[computeFn].apply(this.wasm, paramArray);
                    }, benchName, {
                        dataSize: len,
                        api: 'fast',
                        indicator: indicatorKey
                    });

                    this.results[benchName] = result;
                    this.printResult(result);
                } else {
                    // Pre-allocate buffers outside of benchmark
                    inPtr = this.wasm[allocFn](len);
                    outPtr = this.wasm[allocFn](len);
                    
                    // Copy data once
                    const inView = new Float64Array(this.wasm.__wasm.memory.buffer, inPtr, len);
                    inView.set(data);
                    
                    const result = this.benchmarkFunction(() => {
                        const paramArray = this.prepareFastParams(params, inPtr, outPtr, len, indicatorConfig.fast);
                        this.wasm[computeFn].apply(this.wasm, paramArray);
                    }, benchName, {
                        dataSize: len,
                        api: 'fast',
                        indicator: indicatorKey
                    });

                    this.results[benchName] = result;
                    this.printResult(result);
                }
            } finally {
                // Clean up allocated memory
                if (indicatorConfig.fast.needsMultipleInputs) {
                    if (highPtr) this.wasm[freeFn](highPtr, len);
                    if (lowPtr) this.wasm[freeFn](lowPtr, len);
                    if (closePtr) this.wasm[freeFn](closePtr, len);
                    if (outPtr) this.wasm[freeFn](outPtr, len);
                } else {
                    if (inPtr) this.wasm[freeFn](inPtr, len);
                    if (outPtr) this.wasm[freeFn](outPtr, len);
                }
            }
        }
    }

    /**
     * Benchmark batch API if available
     */
    benchmarkBatchAPI(indicatorKey, indicatorConfig) {
        if (!indicatorConfig.batch) return;
        
        console.log(`\n--- ${indicatorConfig.name} Batch API ---`);
        
        const { fn, config } = indicatorConfig.batch;
        const wasmFn = this.wasm[fn];
        
        if (!wasmFn) {
            console.log(`  Batch function ${fn} not found, skipping...`);
            return;
        }

        // Only test with 10k data for batch operations
        const data = this.data['10k'];
        const sizeName = '10k';
        
        for (const [configName, batchConfig] of Object.entries(config)) {
            const benchName = `${indicatorKey}_batch_${configName}`;
            
            const result = this.benchmarkFunction(() => {
                if (indicatorConfig.needsMultipleInputs || indicatorConfig.fast?.needsMultipleInputs) {
                    const ohlc = this.ohlcData[sizeName];
                    wasmFn.call(this.wasm, ohlc.high, ohlc.low, ohlc.close, batchConfig);
                } else if ((indicatorKey === 'pwma' || indicatorKey === 'supersmoother') && batchConfig.period_range) {
                    // PWMA and SuperSmoother have special batch APIs that take individual parameters
                    const [start, end, step] = batchConfig.period_range;
                    wasmFn.call(this.wasm, data, start, end, step);
                } else {
                    wasmFn.call(this.wasm, data, batchConfig);
                }
            }, benchName, {
                dataSize: data.length,
                api: 'batch',
                indicator: indicatorKey,
                batchSize: configName
            });

            this.results[benchName] = result;
            this.printResult(result);
            
            // Calculate total combinations for batch
            if (batchConfig.period_range) {
                const periods = Math.floor((batchConfig.period_range[1] - batchConfig.period_range[0]) / batchConfig.period_range[2]) + 1;
                let total = periods;
                
                // Handle optional parameters
                if (batchConfig.offset_range) {
                    const offsets = Math.floor((batchConfig.offset_range[1] - batchConfig.offset_range[0]) / batchConfig.offset_range[2]) + 1;
                    total *= offsets;
                }
                if (batchConfig.sigma_range) {
                    const sigmas = Math.floor((batchConfig.sigma_range[1] - batchConfig.sigma_range[0]) / batchConfig.sigma_range[2]) + 1;
                    total *= sigmas;
                }
                
                console.log(`  Total combinations: ${total}`);
            }
        }
    }

    /**
     * Prepare parameters for safe API call
     */
    prepareParams(params, data, indicatorConfig, sizeName) {
        // Check if this indicator needs multiple inputs
        if (indicatorConfig.needsMultipleInputs) {
            const ohlc = this.ohlcData[sizeName];
            const result = [ohlc.high, ohlc.low, ohlc.close];
            
            // Add parameters in order
            for (const value of Object.values(params)) {
                result.push(value);
            }
            
            return result;
        }
        
        // Standard single data array
        const result = [data];
        
        // Add parameters in order (assumes params object maintains order)
        for (const value of Object.values(params)) {
            result.push(value);
        }
        
        return result;
    }

    /**
     * Prepare parameters for fast API call
     */
    prepareFastParams(params, inPtr, outPtr, len, indicatorConfig, highPtr, lowPtr, closePtr) {
        // Check if this indicator needs multiple inputs
        if (indicatorConfig.needsMultipleInputs) {
            // For FRAMA: high_ptr, low_ptr, close_ptr, out_ptr, len, ...params
            const result = [highPtr, lowPtr, closePtr, outPtr, len];
            
            // Add indicator parameters
            for (const value of Object.values(params)) {
                result.push(value);
            }
            
            return result;
        }
        
        // Standard fast API: in_ptr, out_ptr, len, ...params
        const result = [inPtr, outPtr, len];
        
        // Add indicator parameters
        for (const value of Object.values(params)) {
            result.push(value);
        }
        
        return result;
    }

    printResult(result) {
        console.log(`\n${result.name}:`);
        console.log(`  Median: ${result.median.toFixed(3)} ms`);
        console.log(`  Mean:   ${result.mean.toFixed(3)} ms`);
        console.log(`  Min:    ${result.min.toFixed(3)} ms`);
        console.log(`  Max:    ${result.max.toFixed(3)} ms`);
        if (result.throughput > 0) {
            console.log(`  Throughput: ${result.throughput.toFixed(1)} M elem/s`);
        }
        console.log(`  Samples: ${result.samples}, Warmup iterations: ${result.warmupIterations}`);
    }

    printSummary() {
        console.log('\n' + '='.repeat(80));
        console.log('SUMMARY');
        console.log('='.repeat(80));
        
        // Group results by indicator
        const byIndicator = {};
        for (const [name, result] of Object.entries(this.results)) {
            const indicator = result.indicator;
            if (!byIndicator[indicator]) {
                byIndicator[indicator] = [];
            }
            byIndicator[indicator].push(result);
        }

        // Print comparison table for each indicator
        for (const [indicator, results] of Object.entries(byIndicator)) {
            const config = INDICATORS[indicator];
            console.log(`\n${config.name} Performance Comparison:`);
            console.log(`${'Size'.padEnd(10)} ${'Safe API (ms)'.padStart(15)} ${'Fast API (ms)'.padStart(15)} ${'Speedup'.padStart(10)}`);
            console.log('-'.repeat(52));
            
            for (const size of ['10k', '100k', '1M']) {
                const safe = results.find(r => r.name === `${indicator}_safe_${size}`);
                const fast = results.find(r => r.name === `${indicator}_fast_${size}`);
                
                if (safe && fast) {
                    const speedup = safe.median / fast.median;
                    console.log(
                        `${size.padEnd(10)} ${safe.median.toFixed(3).padStart(15)} ${fast.median.toFixed(3).padStart(15)} ${speedup.toFixed(2).padStart(9)}x`
                    );
                }
            }
        }

        // Save results
        const outputPath = join(__dirname, 'wasm_indicator_benchmark_results.json');
        const jsonResults = {
            timestamp: new Date().toISOString(),
            config: CONFIG,
            results: this.results,
        };
        
        try {
            writeFileSync(outputPath, JSON.stringify(jsonResults, null, 2));
            console.log(`\nResults saved to: ${outputPath}`);
        } catch (error) {
            console.error('Failed to save results:', error);
        }
    }

    async runIndicator(indicatorKey, indicatorConfig) {
        console.log('\n' + '='.repeat(80));
        console.log(`Benchmarking ${indicatorConfig.name}`);
        console.log('='.repeat(80));

        // Benchmark safe API
        if (indicatorConfig.safe) {
            this.benchmarkSafeAPI(indicatorKey, indicatorConfig);
        }

        // Benchmark fast API
        if (indicatorConfig.fast) {
            this.benchmarkFastAPI(indicatorKey, indicatorConfig);
        }

        // Benchmark batch API (if available)
        if (indicatorConfig.batch) {
            this.benchmarkBatchAPI(indicatorKey, indicatorConfig);
        }
    }

    async run(options = {}) {
        await this.initialize();
        
        const { indicators = Object.keys(INDICATORS) } = options;
        
        console.log('\nWASM Indicator Performance Benchmark');
        console.log('='.repeat(80));
        console.log('Configuration:');
        console.log(`  Warmup: ${CONFIG.warmupTargetMs}ms`);
        console.log(`  Samples: ${CONFIG.sampleCount}`);
        console.log(`  Min iterations: ${CONFIG.minIterations}`);
        console.log(`  GC disabled: ${CONFIG.disableGC}`);
        console.log(`  Indicators: ${indicators.join(', ')}`);

        // Run benchmarks for each indicator
        for (const indicatorKey of indicators) {
            const config = INDICATORS[indicatorKey];
            if (config) {
                await this.runIndicator(indicatorKey, config);
            } else {
                console.log(`\nWarning: Unknown indicator '${indicatorKey}'`);
            }
        }

        // Print summary
        this.printSummary();
    }
}

// Command line interface
async function main() {
    const args = process.argv.slice(2);
    
    // Check if running with GC control
    if (!global.gc && CONFIG.disableGC) {
        console.warn('\nWarning: GC control not available. Run with: node --expose-gc wasm_indicator_benchmark.js\n');
    }
    
    // Parse command line arguments
    let indicators = [];
    
    for (let i = 0; i < args.length; i++) {
        if (args[i] === '--help' || args[i] === '-h') {
            console.log('\nUsage: node --expose-gc wasm_indicator_benchmark.js [indicators...]');
            console.log('\nAvailable indicators:');
            for (const [key, config] of Object.entries(INDICATORS)) {
                console.log(`  ${key.padEnd(10)} - ${config.name}`);
            }
            console.log('\nExamples:');
            console.log('  node --expose-gc wasm_indicator_benchmark.js          # Run all indicators');
            console.log('  node --expose-gc wasm_indicator_benchmark.js alma     # Run only ALMA');
            console.log('  node --expose-gc wasm_indicator_benchmark.js alma sma # Run ALMA and SMA');
            return;
        } else {
            indicators.push(args[i]);
        }
    }

    // Default to all indicators if none specified
    if (indicators.length === 0) {
        indicators = Object.keys(INDICATORS);
    }

    const benchmark = new WasmIndicatorBenchmark();
    await benchmark.run({ indicators });
}

// Run if called directly
if (import.meta.url.startsWith('file://')) {
    main().catch(console.error);
}

export { WasmIndicatorBenchmark, INDICATORS };