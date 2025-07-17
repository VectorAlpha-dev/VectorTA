#!/usr/bin/env node
/**
 * WASM Performance Benchmark
 * 
 * Implements Criterion-like methodology for accurate WASM performance measurement.
 * Matches the approach used in criterion_comparable_benchmark.py for consistency.
 */

console.log('Starting WASM benchmark script...');

import { performance } from 'perf_hooks';
import { readFileSync, writeFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { AlmaZeroCopy, AlmaContextWrapper, AlmaBenchmarkHelper } from './wasm_zero_copy_helpers.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Benchmark configuration matching Criterion
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

class WasmBenchmark {
    constructor() {
        this.wasm = null;
        this.data = {};
        this.results = {};
    }

    async initialize() {
        // Load WASM module
        console.log('Loading WASM module...');
        try {
            // Since the WASM module is CommonJS, we need to use createRequire
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
        
        // Load 1M candles CSV
        const csvPath = join(__dirname, '../src/data/1MillionCandles.csv');
        const content = readFileSync(csvPath, 'utf8');
        const lines = content.trim().split('\n');
        
        // Skip header
        lines.shift();
        
        // Parse data
        const closes = [];
        for (const line of lines) {
            const parts = line.split(',');
            if (parts.length >= 5) {
                closes.push(parseFloat(parts[4])); // Close price is column 4
            }
        }
        
        // Create different size datasets
        this.data['10k'] = new Float64Array(closes.slice(0, 10_000));
        this.data['100k'] = new Float64Array(closes.slice(0, 100_000));
        this.data['1M'] = new Float64Array(closes);
        
        console.log(`Loaded data sizes: ${Object.keys(this.data).join(', ')}`);
    }

    /**
     * Benchmark a function using Criterion-like methodology
     * @param {Function} fn - Function to benchmark
     * @param {string} name - Benchmark name
     * @returns {Object} Benchmark results
     */
    benchmarkFunction(fn, name) {
        // Disable GC if requested
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
            
            // Calculate throughput
            let dataSize = 0;
            if (name.includes('10k')) dataSize = 10_000;
            else if (name.includes('100k')) dataSize = 100_000;
            else if (name.includes('1M')) dataSize = 1_000_000;
            
            const throughput = dataSize > 0 ? dataSize / (median * 1000) / 1000 : 0; // Million elements/second

            return {
                name,
                median,
                mean,
                min,
                max,
                samples: samples.length,
                warmupIterations,
                throughput,
            };
        } finally {
            // Re-enable GC if it was enabled
            if (CONFIG.disableGC && gcWasEnabled && global.gc) {
                global.gc();
            }
        }
    }

    runAlmaBenchmarks() {
        console.log('\nRunning ALMA benchmarks...');
        console.log('='.repeat(80));

        const almaParams = {
            period: 9,
            offset: 0.85,
            sigma: 6.0,
        };

        // 1. Original API benchmarks
        console.log('\n--- Original API ---');
        for (const [sizeName, data] of Object.entries(this.data)) {
            const benchName = `alma_${sizeName}`;
            
            const result = this.benchmarkFunction(() => {
                this.wasm.alma_js(data, almaParams.period, almaParams.offset, almaParams.sigma);
            }, benchName);

            this.results[benchName] = result;
            this.printResult(result);
        }

        // 2. Zero-copy benchmarks
        console.log('\n--- Zero-Copy API ---');
        const zeroCopy = new AlmaZeroCopy(this.wasm);
        
        for (const [sizeName, data] of Object.entries(this.data)) {
            const benchName = `alma_zerocopy_${sizeName}`;
            
            const result = this.benchmarkFunction(() => {
                zeroCopy.run(data, almaParams);
            }, benchName);

            this.results[benchName] = result;
            this.printResult(result);
        }

        // 3. Context-based benchmarks (reusing weights)
        console.log('\n--- Context-Based API ---');
        const context = new AlmaContextWrapper(
            this.wasm, 
            almaParams.period, 
            almaParams.offset, 
            almaParams.sigma
        );
        
        for (const [sizeName, data] of Object.entries(this.data)) {
            const benchName = `alma_context_${sizeName}`;
            
            const result = this.benchmarkFunction(() => {
                context.process(data);
            }, benchName);

            this.results[benchName] = result;
            this.printResult(result);
        }

        // 4. Pre-allocated buffer benchmarks (minimal overhead)
        console.log('\n--- Pre-allocated Buffers ---');
        for (const [sizeName, data] of Object.entries(this.data)) {
            const benchName = `alma_preallocated_${sizeName}`;
            const helper = new AlmaBenchmarkHelper(this.wasm, data.length);
            
            const result = this.benchmarkFunction(() => {
                helper.run(data, almaParams);
            }, benchName);

            helper.free();
            this.results[benchName] = result;
            this.printResult(result);
        }
    }

    runAlmaBatchBenchmarks() {
        console.log('\nRunning ALMA batch benchmarks...');
        console.log('='.repeat(80));

        // Test different batch sizes to understand performance characteristics
        const batchConfigs = [
            {
                name: 'small_batch',
                data: this.data['10k'],
                params: {
                    period: { start: 5, end: 20, step: 5 },      // 4 values
                    offset: { start: 0.7, end: 0.9, step: 0.1 }, // 3 values
                    sigma: { start: 4.0, end: 8.0, step: 2.0 }   // 3 values
                },
                totalCombos: 36
            },
            {
                name: 'medium_batch',
                data: this.data['10k'],
                params: {
                    period: { start: 5, end: 25, step: 2 },      // 11 values
                    offset: { start: 0.5, end: 1.0, step: 0.1 }, // 6 values
                    sigma: { start: 3.0, end: 9.0, step: 3.0 }   // 3 values
                },
                totalCombos: 198
            },
            {
                name: 'large_batch',
                data: this.data['10k'],
                params: {
                    period: { start: 5, end: 50, step: 1 },      // 46 values
                    offset: { start: 0.5, end: 1.0, step: 0.05 },// 11 values
                    sigma: { start: 2.0, end: 10.0, step: 2.0 }  // 5 values
                },
                totalCombos: 2530
            }
        ];

        // 1. Test old batch API
        console.log('\n--- Old Batch API ---');
        for (const config of batchConfigs) {
            const benchName = `alma_batch_old_${config.name}`;
            
            const result = this.benchmarkFunction(() => {
                this.wasm.alma_batch_js(
                    config.data,
                    config.params.period.start, config.params.period.end, config.params.period.step,
                    config.params.offset.start, config.params.offset.end, config.params.offset.step,
                    config.params.sigma.start, config.params.sigma.end, config.params.sigma.step
                );
            }, benchName);

            this.results[benchName] = result;
            this.printResult(result);
            console.log(`  Total combinations: ${config.totalCombos}`);
        }

        // 2. Test new ergonomic API if available
        if (this.wasm.alma_batch) {
            console.log('\n--- New Batch API ---');
            for (const config of batchConfigs) {
                const benchName = `alma_batch_new_${config.name}`;
                
                const result = this.benchmarkFunction(() => {
                    this.wasm.alma_batch(config.data, {
                        period_range: [config.params.period.start, config.params.period.end, config.params.period.step],
                        offset_range: [config.params.offset.start, config.params.offset.end, config.params.offset.step],
                        sigma_range: [config.params.sigma.start, config.params.sigma.end, config.params.sigma.step],
                    });
                }, benchName);

                this.results[benchName] = result;
                this.printResult(result);
            }
        }

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
        console.log(`${'Benchmark'.padEnd(30)} ${'Median (ms)'.padStart(12)} ${'Throughput'.padStart(15)}`);
        console.log('-'.repeat(80));

        for (const [name, result] of Object.entries(this.results)) {
            const throughputStr = result.throughput > 0 
                ? `${result.throughput.toFixed(1)} M elem/s` 
                : 'N/A';
            console.log(
                `${name.padEnd(30)} ${result.median.toFixed(3).padStart(12)} ${throughputStr.padStart(15)}`
            );
        }

        // Save results to JSON
        const outputPath = join(__dirname, 'wasm_benchmark_results.json');
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

    async run(options = {}) {
        await this.initialize();
        
        const { runSingle = false, runBatch = false, runAll = true } = options;
        
        console.log('\nWASM Performance Benchmark');
        console.log('='.repeat(80));
        console.log('Configuration:');
        console.log(`  Warmup: ${CONFIG.warmupTargetMs}ms`);
        console.log(`  Samples: ${CONFIG.sampleCount}`);
        console.log(`  Min iterations: ${CONFIG.minIterations}`);
        console.log(`  GC disabled: ${CONFIG.disableGC}`);

        // Run benchmarks based on options
        if (runAll || runSingle) {
            this.runAlmaBenchmarks();
        }
        
        if (runAll || runBatch) {
            this.runAlmaBatchBenchmarks();
        }
        
        // Print summary
        this.printSummary();
    }
}

// Command line interface
async function main() {
    const args = process.argv.slice(2);
    
    // Check if running with --node-options
    if (!global.gc && CONFIG.disableGC) {
        console.warn('\nWarning: GC control not available. Run with: node --expose-gc wasm_benchmark.js\n');
    }
    
    // Parse command line arguments
    const runBatch = args.includes('--batch') || args.includes('batch');
    const runSingle = args.includes('--single') || args.includes('single');
    const runAll = !runBatch && !runSingle; // Default to all if nothing specified

    const benchmark = new WasmBenchmark();
    await benchmark.run({
        runSingle: runSingle,
        runBatch: runBatch,
        runAll: runAll
    });
    
    // Show usage if specific argument requested
    if (args.includes('--help') || args.includes('-h')) {
        console.log('\nUsage: node --expose-gc wasm_benchmark.js [options]');
        console.log('Options:');
        console.log('  --single    Run only single ALMA benchmarks');
        console.log('  --batch     Run only batch ALMA benchmarks');
        console.log('  (default)   Run all benchmarks');
    }
}

// Run if called directly
if (import.meta.url.startsWith('file://')) {
    main().catch(console.error);
}

export { WasmBenchmark };