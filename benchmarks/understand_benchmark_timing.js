#!/usr/bin/env node
import { performance } from 'perf_hooks';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load WASM
const wasmPath = join(__dirname, '../pkg/my_project.js');
const importPath = process.platform === 'win32' 
    ? 'file:///' + wasmPath.replace(/\\/g, '/')
    : wasmPath;
const wasm = await import(importPath);

// Replicate what the benchmark does
const CONFIG = {
    warmupTargetMs: 150,
    sampleCount: 10,
    minIterations: 10,
};

const size = 1_002_240;
const data = new Float64Array(size);
for (let i = 0; i < size; i++) {
    data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
}

console.log('Understanding Benchmark Timing');
console.log('==============================');

// Setup like benchmark
const ptr = wasm.jsa_alloc(size);
const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
memView.set(data);

// Warmup phase (exactly like benchmark)
console.log('\nWarmup phase:');
let warmupElapsed = 0;
let warmupIterations = 0;
const warmupStart = performance.now();

while (warmupElapsed < CONFIG.warmupTargetMs) {
    wasm.jsa_fast(ptr, ptr, size, 30);
    warmupIterations++;
    warmupElapsed = performance.now() - warmupStart;
}

console.log(`  Warmup iterations: ${warmupIterations}`);
console.log(`  Warmup time: ${warmupElapsed.toFixed(1)}ms`);
console.log(`  Average per iteration: ${(warmupElapsed / warmupIterations).toFixed(3)}ms`);

// Sampling phase (exactly like benchmark)
console.log('\nSampling phase:');
const samples = [];

for (let i = 0; i < CONFIG.sampleCount; i++) {
    const iterations = Math.max(CONFIG.minIterations, Math.floor(warmupIterations / 10));
    
    const start = performance.now();
    for (let j = 0; j < iterations; j++) {
        wasm.jsa_fast(ptr, ptr, size, 30);
    }
    const end = performance.now();
    
    const timePerIteration = (end - start) / iterations;
    samples.push(timePerIteration);
    
    console.log(`  Sample ${i + 1}: ${iterations} iterations, ${timePerIteration.toFixed(3)}ms per iteration`);
}

// Calculate median (like benchmark)
samples.sort((a, b) => a - b);
const median = samples[Math.floor(samples.length / 2)];

console.log('\nResults:');
console.log(`  Median: ${median.toFixed(3)} ms`);
console.log(`  Min: ${Math.min(...samples).toFixed(3)} ms`);
console.log(`  Max: ${Math.max(...samples).toFixed(3)} ms`);

wasm.jsa_free(ptr, size);

console.log('\nMystery solved?');
console.log('- The benchmark tool divides by iterations!');
console.log('- But wait, why is this ~0.7ms when our direct test showed ~0.7ms too?');