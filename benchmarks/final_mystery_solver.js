#!/usr/bin/env node
import { performance } from 'perf_hooks';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load WASM exactly like benchmark does
const { createRequire } = await import('module');
const require = createRequire(import.meta.url);
const wasmPath = join(__dirname, '../pkg/my_project.js');
const wasm = require(wasmPath);

// Load CSV data exactly like benchmark does
const csvPath = join(__dirname, '../src/data/1MillionCandles.csv');
const content = readFileSync(csvPath, 'utf8');
const lines = content.trim().split('\n');
lines.shift(); // Skip header

const closes = [];
for (const line of lines) {
    const parts = line.split(',');
    if (parts.length >= 5) {
        closes.push(parseFloat(parts[4]));
    }
}

// Create 1M dataset exactly like benchmark
const data = new Float64Array(closes);
console.log(`Data length from CSV: ${data.length}`);
console.log(`First few values: ${data.slice(0, 5).join(', ')}`);

// Test the fast API exactly like the benchmark would
const ptr = wasm.jsa_alloc(data.length);
const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, data.length);
memView.set(data);

console.log('\nTiming jsa_fast with exact benchmark data:');

// Single call
const start1 = performance.now();
wasm.jsa_fast(ptr, ptr, data.length, 30);
const end1 = performance.now();
console.log(`  Single call: ${(end1 - start1).toFixed(3)} ms`);

// Multiple calls to check consistency
const times = [];
for (let i = 0; i < 10; i++) {
    const start = performance.now();
    wasm.jsa_fast(ptr, ptr, data.length, 30);
    times.push(performance.now() - start);
}

console.log(`  10 calls average: ${(times.reduce((a,b) => a+b) / times.length).toFixed(3)} ms`);
console.log(`  Min: ${Math.min(...times).toFixed(3)} ms`);
console.log(`  Max: ${Math.max(...times).toFixed(3)} ms`);

// Check what prepareFastParams does
console.log('\nChecking prepareFastParams behavior:');
console.log('  JSA fast params: inPtr, outPtr, len, period');
console.log(`  Values: ${ptr}, ${ptr}, ${data.length}, 30`);

wasm.jsa_free(ptr, data.length);

// Final hypothesis check
console.log('\nHypothesis: Maybe the benchmark is measuring something else?');
console.log('Let me check if jsa_fast modifies data in place...');

// Test with fresh data
const testData = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
const testPtr = wasm.jsa_alloc(10);
const testView = new Float64Array(wasm.__wasm.memory.buffer, testPtr, 10);
testView.set(testData);

console.log(`Before: ${Array.from(testView).join(', ')}`);
wasm.jsa_fast(testPtr, testPtr, 10, 3);
console.log(`After:  ${Array.from(testView).join(', ')}`);

wasm.jsa_free(testPtr, 10);