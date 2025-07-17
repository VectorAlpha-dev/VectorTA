/**
 * Zero-copy helper functions for WASM ALMA implementation
 * 
 * These helpers provide a convenient JavaScript API for the zero-copy
 * WASM functions, handling memory management and view creation.
 */

export class AlmaZeroCopy {
    constructor(wasm) {
        this.wasm = wasm;
        // Access memory through the __wasm export
        this.memory = wasm.__wasm ? wasm.__wasm.memory : wasm.memory;
    }

    /**
     * Create a Float64Array view into WASM memory
     * @param {number} ptr - Pointer to memory
     * @param {number} len - Length of array
     * @returns {Float64Array} View into WASM memory
     */
    createView(ptr, len) {
        return new Float64Array(this.memory.buffer, ptr, len);
    }

    /**
     * Allocate memory in WASM for input/output buffers
     * @param {number} len - Length of buffer
     * @returns {{ptr: number, view: Float64Array}} Pointer and view
     */
    allocBuffer(len) {
        const ptr = this.wasm.alma_alloc(len);
        const view = this.createView(ptr, len);
        return { ptr, view };
    }

    /**
     * Free allocated memory
     * @param {number} ptr - Pointer to memory
     * @param {number} len - Length of buffer
     */
    freeBuffer(ptr, len) {
        this.wasm.alma_free(ptr, len);
    }

    /**
     * Run ALMA with zero-copy using pre-allocated buffers
     * @param {Float64Array} inputData - Input data
     * @param {Object} params - ALMA parameters
     * @returns {Float64Array} Output array (view into WASM memory)
     */
    runWithPreallocated(inputData, params = {}) {
        const { period = 9, offset = 0.85, sigma = 6.0 } = params;
        const len = inputData.length;

        // Allocate input and output buffers
        const input = this.allocBuffer(len);
        const output = this.allocBuffer(len);

        try {
            // Copy data to WASM memory
            input.view.set(inputData);

            // Run computation
            const result = this.wasm.alma_into(
                input.ptr,
                output.ptr,
                len,
                period,
                offset,
                sigma
            );

            if (result !== undefined) {
                throw new Error('ALMA computation failed');
            }

            // Return view to output (caller must copy if needed)
            return output.view;
        } catch (error) {
            // Clean up on error
            this.freeBuffer(input.ptr, len);
            this.freeBuffer(output.ptr, len);
            throw error;
        }
    }

    /**
     * Run ALMA with zero-copy, returning a copy of the result
     * @param {Float64Array} inputData - Input data
     * @param {Object} params - ALMA parameters
     * @returns {Float64Array} Copy of output data
     */
    run(inputData, params = {}) {
        const { period = 9, offset = 0.85, sigma = 6.0 } = params;
        const len = inputData.length;

        // Allocate buffers
        const input = this.allocBuffer(len);
        const output = this.allocBuffer(len);

        try {
            // Copy data to WASM memory
            input.view.set(inputData);

            // Run computation
            const result = this.wasm.alma_into(
                input.ptr,
                output.ptr,
                len,
                period,
                offset,
                sigma
            );

            if (result !== undefined) {
                throw new Error('ALMA computation failed');
            }

            // Create a new view in case memory grew
            const currentOutputView = new Float64Array(
                this.wasm.__wasm.memory.buffer,
                output.ptr,
                len
            );
            
            // Copy result before freeing
            const resultCopy = new Float64Array(currentOutputView);
            
            // Free buffers
            this.freeBuffer(input.ptr, len);
            this.freeBuffer(output.ptr, len);
            
            return resultCopy;
        } catch (error) {
            // Clean up on error
            this.freeBuffer(input.ptr, len);
            this.freeBuffer(output.ptr, len);
            throw error;
        }
    }
}

/**
 * Context-based ALMA for repeated computations with same parameters
 */
export class AlmaContextWrapper {
    constructor(wasm, period, offset, sigma) {
        this.wasm = wasm;
        // Access memory through the __wasm export
        this.memory = wasm.__wasm ? wasm.__wasm.memory : wasm.memory;
        this.context = new wasm.AlmaContext(period, offset, sigma);
        this.warmupPeriod = this.context.get_warmup_period();
    }

    /**
     * Process data using the context
     * @param {Float64Array} inputData - Input data
     * @returns {Float64Array} Output data
     */
    process(inputData) {
        const len = inputData.length;
        
        // Allocate buffers
        const inputPtr = this.wasm.alma_alloc(len);
        const outputPtr = this.wasm.alma_alloc(len);
        
        try {
            // Create views
            const inputView = new Float64Array(this.memory.buffer, inputPtr, len);
            const outputView = new Float64Array(this.memory.buffer, outputPtr, len);
            
            // Copy input data
            inputView.set(inputData);
            
            // Process
            const result = this.context.update_into(inputPtr, outputPtr, len);
            if (result !== undefined) {
                throw new Error('ALMA context update failed');
            }
            
            // Copy result
            const output = new Float64Array(outputView);
            
            return output;
        } finally {
            // Always clean up
            this.wasm.alma_free(inputPtr, len);
            this.wasm.alma_free(outputPtr, len);
        }
    }

    /**
     * Process data in-place (modifies input array)
     * @param {Float64Array} data - Input/output data
     */
    processInPlace(data) {
        const len = data.length;
        
        // Allocate single buffer
        const ptr = this.wasm.alma_alloc(len);
        
        try {
            // Create view and copy data
            const view = new Float64Array(this.memory.buffer, ptr, len);
            view.set(data);
            
            // Process in-place
            const result = this.context.update_into(ptr, ptr, len);
            if (result !== undefined) {
                throw new Error('ALMA context update failed');
            }
            
            // Copy back
            data.set(view);
        } finally {
            this.wasm.alma_free(ptr, len);
        }
    }

    /**
     * Clean up the context
     */
    free() {
        // The context will be freed by WASM GC
        this.context = null;
    }
}

/**
 * Benchmark helper that reuses buffers
 */
export class AlmaBenchmarkHelper {
    constructor(wasm, dataSize) {
        this.wasm = wasm;
        // Access memory through the __wasm export
        this.memory = wasm.__wasm ? wasm.__wasm.memory : wasm.memory;
        this.dataSize = dataSize;
        
        // Pre-allocate buffers
        this.inputPtr = this.wasm.alma_alloc(dataSize);
        this.outputPtr = this.wasm.alma_alloc(dataSize);
        
        // Create views
        this.inputView = new Float64Array(this.memory.buffer, this.inputPtr, dataSize);
        this.outputView = new Float64Array(this.memory.buffer, this.outputPtr, dataSize);
    }

    /**
     * Run ALMA with minimal overhead
     * @param {Float64Array} data - Input data (must be same size as constructor)
     * @param {Object} params - ALMA parameters
     */
    run(data, params = {}) {
        const { period = 9, offset = 0.85, sigma = 6.0 } = params;
        
        // Copy data to pre-allocated buffer
        this.inputView.set(data);
        
        // Run computation
        const result = this.wasm.alma_into(
            this.inputPtr,
            this.outputPtr,
            this.dataSize,
            period,
            offset,
            sigma
        );
        
        if (result !== undefined) {
            throw new Error('ALMA computation failed');
        }
        
        // Return view (no copy)
        return this.outputView;
    }

    /**
     * Clean up allocated memory
     */
    free() {
        this.wasm.alma_free(this.inputPtr, this.dataSize);
        this.wasm.alma_free(this.outputPtr, this.dataSize);
    }
}