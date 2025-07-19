// Simple test script to debug linreg batch API
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function test() {
    try {
        // Load WASM module
        const wasmPath = join(__dirname, 'pkg/my_project.js');
        const importPath = process.platform === 'win32' 
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
        const wasm = await import(importPath);
        
        console.log("WASM module loaded successfully");
        
        // Create test data
        const data = new Float64Array(100);
        for (let i = 0; i < 100; i++) {
            data[i] = i + 1;
        }
        
        console.log("Testing linreg_js (safe API)...");
        const safeResult = wasm.linreg_js(data, 10);
        console.log("Safe API result length:", safeResult.length);
        console.log("Safe API first few values:", safeResult.slice(0, 15));
        
        console.log("\nTesting linreg_batch...");
        try {
            const batchResult = wasm.linreg_batch(data, {
                period_range: [10, 20, 10]  // Just 2 periods: 10, 20
            });
            console.log("Batch API succeeded!");
            console.log("Batch result:", batchResult);
        } catch (e) {
            console.error("Batch API failed:", e);
            console.error("Error stack:", e.stack);
        }
        
        // Also test with a simpler config
        console.log("\nTesting linreg_batch with single period...");
        try {
            const singleBatch = wasm.linreg_batch(data, {
                period_range: [10, 10, 0]  // Single period
            });
            console.log("Single batch succeeded!");
            console.log("Single batch result:", singleBatch);
        } catch (e) {
            console.error("Single batch failed:", e);
        }
        
    } catch (error) {
        console.error("Test failed:", error);
    }
}

test();