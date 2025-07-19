// Compare WASM vs native performance more carefully
import { performance } from 'perf_hooks';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function measurePerformance() {
    // Load WASM module
    const wasmPath = join(__dirname, 'pkg/my_project.js');
    const importPath = process.platform === 'win32' 
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    const wasm = await import(importPath);
    
    // Test data sizes
    const sizes = [1000, 10000, 100000];
    
    for (const size of sizes) {
        console.log(`\n=== Testing with ${size} data points ===`);
        
        // Create test data
        const data = new Float64Array(size);
        for (let i = 0; i < size; i++) {
            data[i] = Math.sin(i * 0.1) * 100 + i * 0.01;
        }
        
        // Test Safe API
        console.log("\nSafe API (linreg_js):");
        let start = performance.now();
        for (let i = 0; i < 100; i++) {
            const result = wasm.linreg_js(data, 14);
        }
        let elapsed = performance.now() - start;
        console.log(`  100 iterations: ${elapsed.toFixed(2)} ms`);
        console.log(`  Per iteration: ${(elapsed / 100).toFixed(3)} ms`);
        console.log(`  Throughput: ${(size / (elapsed / 100) / 1000).toFixed(1)} K elem/ms`);
        
        // Test Fast API
        console.log("\nFast API (linreg_into):");
        const len = data.length;
        const inPtr = wasm.linreg_alloc(len);
        const outPtr = wasm.linreg_alloc(len);
        
        // Copy data to WASM memory
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inView.set(data);
        
        start = performance.now();
        for (let i = 0; i < 100; i++) {
            wasm.linreg_into(inPtr, outPtr, len, 14);
        }
        elapsed = performance.now() - start;
        console.log(`  100 iterations: ${elapsed.toFixed(2)} ms`);
        console.log(`  Per iteration: ${(elapsed / 100).toFixed(3)} ms`);
        console.log(`  Throughput: ${(size / (elapsed / 100) / 1000).toFixed(1)} K elem/ms`);
        
        // Cleanup
        wasm.linreg_free(inPtr, len);
        wasm.linreg_free(outPtr, len);
        
        // Also test ALMA for comparison
        console.log("\nALMA Fast API (for comparison):");
        const almaInPtr = wasm.alma_alloc(len);
        const almaOutPtr = wasm.alma_alloc(len);
        
        const almaInView = new Float64Array(wasm.__wasm.memory.buffer, almaInPtr, len);
        almaInView.set(data);
        
        start = performance.now();
        for (let i = 0; i < 100; i++) {
            wasm.alma_into(almaInPtr, almaOutPtr, len, 9, 0.85, 6.0);
        }
        elapsed = performance.now() - start;
        console.log(`  100 iterations: ${elapsed.toFixed(2)} ms`);
        console.log(`  Per iteration: ${(elapsed / 100).toFixed(3)} ms`);
        console.log(`  Throughput: ${(size / (elapsed / 100) / 1000).toFixed(1)} K elem/ms`);
        
        wasm.alma_free(almaInPtr, len);
        wasm.alma_free(almaOutPtr, len);
    }
}

measurePerformance().catch(console.error);