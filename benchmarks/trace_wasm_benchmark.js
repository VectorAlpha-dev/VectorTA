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

// Create 1M element array
const size = 1_002_240;
const data = new Float64Array(size);
for (let i = 0; i < size; i++) {
    data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
}

console.log('Tracing WASM Benchmark Behavior');
console.log('================================');
console.log(`Data size: ${size.toLocaleString()} elements`);

// Test 1: Single call timing
console.log('\n1. Single jsa_fast call:');
{
    const ptr = wasm.jsa_alloc(size);
    const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
    memView.set(data);
    
    const start = performance.now();
    wasm.jsa_fast(ptr, ptr, size, 30);
    const end = performance.now();
    
    console.log(`  Time: ${(end - start).toFixed(3)} ms`);
    
    wasm.jsa_free(ptr, size);
}

// Test 2: Multiple iterations (like benchmark does)
console.log('\n2. Multiple iterations test:');
{
    const ptr = wasm.jsa_alloc(size);
    const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
    memView.set(data);
    
    // Test different iteration counts
    for (const iterations of [1, 2, 5, 10]) {
        const start = performance.now();
        for (let i = 0; i < iterations; i++) {
            wasm.jsa_fast(ptr, ptr, size, 30);
        }
        const end = performance.now();
        
        const total = end - start;
        const perIteration = total / iterations;
        
        console.log(`  ${iterations} iterations: ${total.toFixed(3)}ms total, ${perIteration.toFixed(3)}ms per iteration`);
    }
    
    wasm.jsa_free(ptr, size);
}

// Test 3: Check if results change on repeated calls
console.log('\n3. Checking result stability:');
{
    const ptr = wasm.jsa_alloc(size);
    const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
    memView.set(data);
    
    // Get initial result
    wasm.jsa_fast(ptr, ptr, size, 30);
    const firstResult = Array.from(memView.slice(0, 5));
    
    // Run again
    wasm.jsa_fast(ptr, ptr, size, 30);
    const secondResult = Array.from(memView.slice(0, 5));
    
    console.log(`  First run:  [${firstResult.map(x => x.toFixed(6)).join(', ')}]`);
    console.log(`  Second run: [${secondResult.map(x => x.toFixed(6)).join(', ')}]`);
    console.log(`  Results ${JSON.stringify(firstResult) === JSON.stringify(secondResult) ? 'are identical' : 'DIFFER!'}`);
    
    wasm.jsa_free(ptr, size);
}

console.log('\nConclusion:');
console.log('- If per-iteration time decreases with more iterations, there may be caching');
console.log('- If results differ on repeated calls, the operation may be cumulative');