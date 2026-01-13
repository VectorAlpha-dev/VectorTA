
import { Worker } from 'worker_threads';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import os from 'os';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export class AlmaParallelProcessor {
    constructor(numWorkers = null) {
        this.numWorkers = numWorkers || os.cpus().length;
        this.workers = [];
        this.taskQueue = [];
        this.busyWorkers = new Set();
        this.taskCallbacks = new Map();
        this.nextTaskId = 0;
        this.initialized = false;
    }

    async initialize() {
        if (this.initialized) return;


        const workerPromises = [];

        for (let i = 0; i < this.numWorkers; i++) {
            const worker = new Worker(join(__dirname, 'wasm_parallel_worker.js'), {
                workerData: {}
            });
            this.workers.push(worker);


            worker.on('message', (message) => {
                this.handleWorkerMessage(worker, message);
            });

            worker.on('error', (error) => {
                console.error(`Worker ${i} error:`, error);
            });


            workerPromises.push(new Promise((resolve, reject) => {
                const timeout = setTimeout(() => {
                    reject(new Error(`Worker ${i} initialization timeout`));
                }, 10000);

                const messageHandler = (msg) => {
                    if (msg.type === 'ready') {
                        clearTimeout(timeout);
                        worker.off('message', messageHandler);
                        resolve();
                    } else if (msg.type === 'init_error') {
                        clearTimeout(timeout);
                        worker.off('message', messageHandler);
                        reject(new Error(msg.error));
                    }
                };

                worker.on('message', messageHandler);
            }));
        }

        await Promise.all(workerPromises);
        this.initialized = true;
    }

    handleWorkerMessage(worker, message) {
        const { type, taskId, result, error } = message;

        if (type === 'result' || type === 'error') {

            this.busyWorkers.delete(worker);


            const callback = this.taskCallbacks.get(taskId);
            if (callback) {
                this.taskCallbacks.delete(taskId);

                if (type === 'result') {
                    callback.resolve(result);
                } else {
                    callback.reject(new Error(error));
                }
            }


            this.processNextTask();
        }
    }

    processNextTask() {
        if (this.taskQueue.length === 0) return;


        const availableWorker = this.workers.find(w => !this.busyWorkers.has(w));
        if (!availableWorker) return;


        const task = this.taskQueue.shift();
        this.busyWorkers.add(availableWorker);


        availableWorker.postMessage(task);
    }

    async computeAlma(data, period, offset = 0.85, sigma = 6.0) {
        if (!this.initialized) {
            await this.initialize();
        }

        const taskId = this.nextTaskId++;

        return new Promise((resolve, reject) => {

            this.taskCallbacks.set(taskId, { resolve, reject });


            const dataArray = Array.from(data);


            this.taskQueue.push({
                type: 'compute',
                taskId,
                payload: {
                    data: dataArray,
                    length: data.length,
                    period,
                    offset,
                    sigma
                }
            });


            this.processNextTask();
        });
    }

    async computeAlmaBatch(dataArrays, period, offset = 0.85, sigma = 6.0) {
        if (!this.initialized) {
            await this.initialize();
        }


        const promises = dataArrays.map(data =>
            this.computeAlma(data, period, offset, sigma)
        );


        return Promise.all(promises);
    }


    async computeAlmaParameterSweep(data, parameterSets) {
        if (!this.initialized) {
            await this.initialize();
        }

        const promises = parameterSets.map(params =>
            this.computeAlma(data, params.period, params.offset, params.sigma)
        );

        return Promise.all(promises);
    }

    async terminate() {

        await Promise.all(this.workers.map(worker => worker.terminate()));
        this.workers = [];
        this.initialized = false;
        this.taskQueue = [];
        this.busyWorkers.clear();
        this.taskCallbacks.clear();
    }
}


export async function benchmarkParallelAlma() {
    const processor = new AlmaParallelProcessor();

    try {
        await processor.initialize();
        console.log(`Initialized ${processor.numWorkers} workers`);


        const sizes = [10000, 100000, 1000000];
        const numBatches = 10;

        for (const size of sizes) {

            const dataBatches = [];
            for (let i = 0; i < numBatches; i++) {
                const data = new Float64Array(size);
                for (let j = 0; j < size; j++) {
                    data[j] = Math.sin(j * 0.01) + Math.random() * 0.1;
                }
                dataBatches.push(data);
            }


            const serialStart = performance.now();
            for (const data of dataBatches) {
                await processor.computeAlma(data, 9);
            }
            const serialTime = performance.now() - serialStart;


            const parallelStart = performance.now();
            await processor.computeAlmaBatch(dataBatches, 9);
            const parallelTime = performance.now() - parallelStart;

            console.log(`\nSize: ${size}, Batches: ${numBatches}`);
            console.log(`Serial time: ${serialTime.toFixed(2)}ms`);
            console.log(`Parallel time: ${parallelTime.toFixed(2)}ms`);
            console.log(`Speedup: ${(serialTime / parallelTime).toFixed(2)}x`);
        }

    } finally {
        await processor.terminate();
    }
}