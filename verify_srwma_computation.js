#!/usr/bin/env node
import { createRequire } from 'module';
import { readFileSync } from 'fs';
import { join } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const require = createRequire(import.meta.url);

// Load WASM module
const wasmPath = join(__dirname, 'pkg/my_project.js');
const wasm = require(wasmPath);

// Load test data (same as WASM benchmark)
const csvPath = join(__dirname, 'src/data/1MillionCandles.csv');
const content = readFileSync(csvPath, 'utf8');
const lines = content.trim().split('\n');
lines.shift(); // Skip header

// Parse close prices
const closes = [];
for (const line of lines) {
    const parts = line.split(',');
    if (parts.length >= 5) {
        closes.push(parseFloat(parts[4]));
    }
}

// Test with different sizes
const sizes = [1000, 10000, 100000];
const period = 14;

console.log('Verifying SRWMA computation...\n');

for (const size of sizes) {
    const data = new Float64Array(closes.slice(0, size));
    console.log(`\nData size: ${size}`);
    console.log(`First 5 values: ${Array.from(data.slice(0, 5))}`);
    console.log(`Last 5 values: ${Array.from(data.slice(-5))}`);
    
    // Run SRWMA
    const start = performance.now();
    const result = wasm.srwma_js(data, period);
    const end = performance.now();
    
    console.log(`\nComputation time: ${(end - start).toFixed(3)} ms`);
    console.log(`Result length: ${result.length}`);
    
    // Count NaN values (warmup period)
    let nanCount = 0;
    let firstNonNanIndex = -1;
    for (let i = 0; i < result.length; i++) {
        if (isNaN(result[i])) {
            nanCount++;
        } else if (firstNonNanIndex === -1) {
            firstNonNanIndex = i;
        }
    }
    
    console.log(`NaN count (warmup): ${nanCount}`);
    console.log(`First non-NaN index: ${firstNonNanIndex}`);
    console.log(`Expected warmup: ${period + 1}`);
    
    // Show some results
    if (firstNonNanIndex >= 0) {
        const firstValues = Array.from(result.slice(firstNonNanIndex, firstNonNanIndex + 5));
        console.log(`First 5 non-NaN values: ${firstValues.map(v => v.toFixed(6))}`);
        
        const lastValues = Array.from(result.slice(-5));
        console.log(`Last 5 values: ${lastValues.map(v => v.toFixed(6))}`);
    }
}

// Test fast API
console.log('\n\nTesting Fast API...');
const testData = new Float64Array(10000);
for (let i = 0; i < testData.length; i++) {
    testData[i] = closes[i] || 0;
}

// Allocate buffers
const len = testData.length;
const inPtr = wasm.srwma_alloc(len);
const outPtr = wasm.srwma_alloc(len);

// Copy data to WASM memory
const wasmMemory = new Float64Array(wasm.memory.buffer);
wasmMemory.set(testData, inPtr / 8);

// Run computation
const startFast = performance.now();
wasm.srwma_into(inPtr, outPtr, len, period);
const endFast = performance.now();

// Read results
const results = new Float64Array(wasm.memory.buffer, outPtr, len);
const resultCopy = Array.from(results);

console.log(`Fast API time: ${(endFast - startFast).toFixed(3)} ms`);
console.log(`First 5 values after warmup: ${resultCopy.slice(15, 20).map(v => v.toFixed(6))}`);

// Clean up
wasm.srwma_free(inPtr, len);
wasm.srwma_free(outPtr, len);

console.log('\nDone!');