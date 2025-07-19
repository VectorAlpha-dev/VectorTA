#!/usr/bin/env node
const Module = require('../pkg/my_project.js');

// Simple performance test
const size = 1000000;
const data = new Float64Array(size);
for (let i = 0; i < size; i++) {
    data[i] = Math.sin(i * 0.01);
}

console.log('JSA Performance Debug');
console.log('====================');

// Test 1: Measure overhead of calling WASM
console.log('\n1. WASM function call overhead:');
{
    // Test with a simple allocation/free cycle
    const times = [];
    for (let i = 0; i < 100; i++) {
        const start = performance.now();
        const ptr = Module.jsa_alloc(10);
        Module.jsa_free(ptr, 10);
        times.push(performance.now() - start);
    }
    console.log(`  Median: ${times.sort((a,b) => a-b)[50].toFixed(6)} ms`);
}

// Test 2: Safe API performance breakdown
console.log('\n2. Safe API (jsa_js) breakdown:');
{
    // Time just the WASM call
    const times = [];
    for (let i = 0; i < 50; i++) {
        const start = performance.now();
        const result = Module.jsa_js(data, 30);
        times.push(performance.now() - start);
    }
    console.log(`  Total time: ${times.sort((a,b) => a-b)[25].toFixed(3)} ms`);
}

// Test 3: Fast API with pre-allocated memory
console.log('\n3. Fast API (jsa_fast) with pre-allocated memory:');
{
    const ptr = Module.jsa_alloc(size);
    const memView = new Float64Array(Module.__wasm.memory.buffer, ptr, size);
    
    // Copy data once
    const copyStart = performance.now();
    memView.set(data);
    const copyTime = performance.now() - copyStart;
    console.log(`  Data copy time: ${copyTime.toFixed(3)} ms`);
    
    // Time just computation
    const times = [];
    for (let i = 0; i < 50; i++) {
        const start = performance.now();
        Module.jsa_fast(ptr, ptr, size, 30);
        times.push(performance.now() - start);
    }
    
    Module.jsa_free(ptr, size);
    console.log(`  Computation time: ${times.sort((a,b) => a-b)[25].toFixed(3)} ms`);
}

// Test 4: Check if we're using the right kernel
console.log('\n4. Kernel selection test:');
{
    // JSA uses Kernel::Auto which should resolve to Scalar in WASM
    // The computation is very simple: out[i] = (data[i] + data[i-period]) * 0.5
    // For 1M elements with period=30, we do ~1M additions and multiplications
    
    const opsPerElement = 2; // 1 add, 1 multiply
    const totalOps = size * opsPerElement;
    console.log(`  Total operations: ${(totalOps/1e6).toFixed(1)}M`);
    
    // Modern CPUs can do ~10-20 GFLOPS
    // WASM scalar should achieve maybe 1-2 GFLOPS
    const expectedGFLOPS = 1.0; // Conservative estimate for WASM
    const expectedTime = (totalOps / (expectedGFLOPS * 1e9)) * 1000; // ms
    console.log(`  Expected time at ${expectedGFLOPS} GFLOPS: ${expectedTime.toFixed(3)} ms`);
}

console.log('\nAnalysis:');
console.log('- Native Rust (direct write): ~0.092 ms');
console.log('- Expected WASM overhead: ~2x = ~0.18-0.20 ms');
console.log('- Actual WASM performance: see above');