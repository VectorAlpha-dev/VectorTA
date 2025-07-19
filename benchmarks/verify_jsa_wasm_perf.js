#!/usr/bin/env node
import { performance } from 'perf_hooks';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load WASM module
const wasmPath = join(__dirname, '../pkg/my_project.js');
const importPath = process.platform === 'win32' 
    ? 'file:///' + wasmPath.replace(/\\/g, '/')
    : wasmPath;
const wasm = await import(importPath);

// Test data
const size = 1_002_240; // Same as benchmark
const data = new Float64Array(size);
for (let i = 0; i < size; i++) {
    data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
}

console.log('JSA WASM Performance Verification');
console.log('=================================');
console.log(`Data size: ${size.toLocaleString()} elements`);

// Test 1: Safe API
console.log('\n1. Safe API (jsa_js):');
{
    // Warmup
    for (let i = 0; i < 100; i++) {
        wasm.jsa_js(data, 30);
    }
    
    const times = [];
    for (let i = 0; i < 100; i++) {
        const start = performance.now();
        wasm.jsa_js(data, 30);
        times.push(performance.now() - start);
    }
    
    times.sort((a, b) => a - b);
    console.log(`  Median: ${times[50].toFixed(3)} ms`);
    console.log(`  Min: ${times[0].toFixed(3)} ms`);
    console.log(`  Max: ${times[99].toFixed(3)} ms`);
}

// Test 2: Fast API
console.log('\n2. Fast API (jsa_fast):');
{
    const ptr = wasm.jsa_alloc(size);
    const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
    memView.set(data);
    
    // Warmup
    for (let i = 0; i < 100; i++) {
        wasm.jsa_fast(ptr, ptr, size, 30);
    }
    
    const times = [];
    for (let i = 0; i < 100; i++) {
        const start = performance.now();
        wasm.jsa_fast(ptr, ptr, size, 30);
        times.push(performance.now() - start);
    }
    
    wasm.jsa_free(ptr, size);
    
    times.sort((a, b) => a - b);
    console.log(`  Median: ${times[50].toFixed(3)} ms`);
    console.log(`  Min: ${times[0].toFixed(3)} ms`);
    console.log(`  Max: ${times[99].toFixed(3)} ms`);
}

// Test 3: Measure just the memcpy overhead
console.log('\n3. Memory copy overhead:');
{
    const src = new Float64Array(size);
    const dst = new Float64Array(size);
    
    const times = [];
    for (let i = 0; i < 100; i++) {
        const start = performance.now();
        dst.set(src);
        times.push(performance.now() - start);
    }
    
    times.sort((a, b) => a - b);
    console.log(`  Median: ${times[50].toFixed(3)} ms`);
}

console.log('\nExpected vs Actual:');
console.log('  Rust direct write: ~0.092 ms');
console.log('  Expected WASM fast: ~0.18-0.20 ms (2x overhead)');
console.log('  Actual WASM fast: see above');