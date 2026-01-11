
import { createRequire } from 'module';
import { AlmaParallelProcessor } from './wasm_parallel_processor.js';
import os from 'os';

const require = createRequire(import.meta.url);


const wasm = require('../pkg/vector_ta.js');


function generateTestData(size) {
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }
    return data;
}


function computeAlmaSerial(wasm, data, period, offset = 0.85, sigma = 6.0) {
    const ptr = wasm.alma_alloc(data.length);
    const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, data.length);
    memView.set(data);
    
    try {
        
        wasm.alma_into(ptr, ptr, data.length, period, offset, sigma);
        
        const output = new Float64Array(data.length);
        output.set(memView);
        wasm.alma_free(ptr, data.length);
        
        return output;
    } catch (error) {
        wasm.alma_free(ptr, data.length);
        throw new Error(`ALMA computation failed: ${error}`);
    }
}

async function runParallelBenchmark() {
    console.log('WASM Parallel Processing Benchmark');
    console.log('================================================================================');
    console.log(`CPU Cores: ${os.cpus().length}`);
    console.log(`CPU Model: ${os.cpus()[0].model}`);
    console.log();
    
    const processor = new AlmaParallelProcessor();
    
    try {
        await processor.initialize();
        console.log(`Initialized ${processor.numWorkers} workers`);
        console.log();
        
        
        const configs = [
            { size: 10000, batches: 100, period: 9 },
            { size: 100000, batches: 20, period: 9 },
            { size: 1000000, batches: 10, period: 9 },
        ];
        
        for (const config of configs) {
            console.log(`\nTest: ${config.batches} batches of ${config.size} elements`);
            console.log('--------------------------------------------------------------------------------');
            
            
            const dataBatches = [];
            for (let i = 0; i < config.batches; i++) {
                dataBatches.push(generateTestData(config.size));
            }
            
            
            console.log('Warming up...');
            for (let i = 0; i < 5; i++) {
                await processor.computeAlma(dataBatches[0], config.period);
                computeAlmaSerial(wasm, dataBatches[0], config.period);
            }
            
            
            const serialRuns = 5;
            const serialTimes = [];
            
            for (let run = 0; run < serialRuns; run++) {
                const start = performance.now();
                for (const data of dataBatches) {
                    computeAlmaSerial(wasm, data, config.period);
                }
                serialTimes.push(performance.now() - start);
            }
            
            const serialMedian = serialTimes.sort((a, b) => a - b)[Math.floor(serialRuns / 2)];
            const serialPerBatch = serialMedian / config.batches;
            
            
            const parallelRuns = 5;
            const parallelTimes = [];
            
            for (let run = 0; run < parallelRuns; run++) {
                const start = performance.now();
                await processor.computeAlmaBatch(dataBatches, config.period);
                parallelTimes.push(performance.now() - start);
            }
            
            const parallelMedian = parallelTimes.sort((a, b) => a - b)[Math.floor(parallelRuns / 2)];
            const parallelPerBatch = parallelMedian / config.batches;
            
            
            const speedup = serialMedian / parallelMedian;
            const efficiency = (speedup / processor.numWorkers) * 100;
            
            console.log(`Serial processing:`);
            console.log(`  Total time: ${serialMedian.toFixed(2)}ms`);
            console.log(`  Per batch: ${serialPerBatch.toFixed(2)}ms`);
            console.log(`  Throughput: ${(config.size * config.batches / serialMedian / 1000).toFixed(2)} M elem/s`);
            
            console.log(`\nParallel processing (${processor.numWorkers} workers):`);
            console.log(`  Total time: ${parallelMedian.toFixed(2)}ms`);
            console.log(`  Per batch: ${parallelPerBatch.toFixed(2)}ms`);
            console.log(`  Throughput: ${(config.size * config.batches / parallelMedian / 1000).toFixed(2)} M elem/s`);
            
            console.log(`\nPerformance improvement:`);
            console.log(`  Speedup: ${speedup.toFixed(2)}x`);
            console.log(`  Efficiency: ${efficiency.toFixed(1)}%`);
            
            
            if (config.size === 100000) {
                console.log('\n--- Parameter Sweep Test ---');
                const testData = dataBatches[0];
                const parameterSets = [];
                
                
                for (let period = 5; period <= 20; period += 5) {
                    for (let sigma = 4; sigma <= 8; sigma += 2) {
                        parameterSets.push({ period, offset: 0.85, sigma });
                    }
                }
                
                
                const serialSweepStart = performance.now();
                for (const params of parameterSets) {
                    computeAlmaSerial(wasm, testData, params.period, params.offset, params.sigma);
                }
                const serialSweepTime = performance.now() - serialSweepStart;
                
                
                const parallelSweepStart = performance.now();
                await processor.computeAlmaParameterSweep(testData, parameterSets);
                const parallelSweepTime = performance.now() - parallelSweepStart;
                
                console.log(`Parameter combinations: ${parameterSets.length}`);
                console.log(`Serial sweep: ${serialSweepTime.toFixed(2)}ms`);
                console.log(`Parallel sweep: ${parallelSweepTime.toFixed(2)}ms`);
                console.log(`Speedup: ${(serialSweepTime / parallelSweepTime).toFixed(2)}x`);
            }
        }
        
    } finally {
        await processor.terminate();
        console.log('\n\nBenchmark complete. Workers terminated.');
    }
}


runParallelBenchmark().catch(console.error);