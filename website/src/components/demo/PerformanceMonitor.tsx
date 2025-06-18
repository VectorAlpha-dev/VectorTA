import { useState, useEffect } from 'react';
import { AnimatedCounter } from '../ui/AnimatedCounter';

interface PerformanceMetrics {
  executionTime: number;
  operationsPerSecond: number;
  dataPointsProcessed: number;
  memoryUsed?: number;
}

interface PerformanceMonitorProps {
  metrics: PerformanceMetrics | null;
  isCalculating: boolean;
  comparison?: {
    name: string;
    executionTime: number;
  };
}

export function PerformanceMonitor({ metrics, isCalculating, comparison }: PerformanceMonitorProps) {
  const [showDetails, setShowDetails] = useState(false);

  if (!metrics && !isCalculating) {
    return null;
  }

  const speedupFactor = comparison && metrics 
    ? (comparison.executionTime / metrics.executionTime).toFixed(1) 
    : null;

  return (
    <div className="bg-card rounded-lg border border-border p-6 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <svg className="w-5 h-5 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
              d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          Performance Metrics
        </h3>
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="text-sm text-muted-foreground hover:text-foreground transition-colors"
          aria-label={showDetails ? 'Hide details' : 'Show details'}
        >
          {showDetails ? 'Hide' : 'Details'}
        </button>
      </div>

      {isCalculating ? (
        <div className="flex items-center justify-center py-8">
          <div className="flex items-center gap-3">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary"></div>
            <span className="text-sm text-muted-foreground">Calculating...</span>
          </div>
        </div>
      ) : metrics && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-primary">
                <AnimatedCounter end={metrics.executionTime} suffix="ms" decimals={1} />
              </div>
              <div className="text-xs text-muted-foreground mt-1">Execution Time</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-secondary">
                <AnimatedCounter 
                  end={Math.floor(metrics.operationsPerSecond / 1000)} 
                  suffix="K" 
                  decimals={0} 
                />
              </div>
              <div className="text-xs text-muted-foreground mt-1">Ops/Second</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold">
                <AnimatedCounter 
                  end={metrics.dataPointsProcessed} 
                  separator="," 
                />
              </div>
              <div className="text-xs text-muted-foreground mt-1">Data Points</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-chart-up">
                {speedupFactor ? (
                  <span>{speedupFactor}x</span>
                ) : (
                  <span>—</span>
                )}
              </div>
              <div className="text-xs text-muted-foreground mt-1">Speedup</div>
            </div>
          </div>

          {showDetails && (
            <div className="pt-4 border-t border-border space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Processing Rate:</span>
                <span className="font-mono">
                  {(metrics.dataPointsProcessed / metrics.executionTime * 1000).toFixed(0)} points/sec
                </span>
              </div>
              
              {comparison && (
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">vs {comparison.name}:</span>
                  <span className="font-mono text-chart-up">
                    {((comparison.executionTime - metrics.executionTime) / comparison.executionTime * 100).toFixed(1)}% faster
                  </span>
                </div>
              )}
              
              {metrics.memoryUsed && (
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Memory Used:</span>
                  <span className="font-mono">{(metrics.memoryUsed / 1024).toFixed(1)} KB</span>
                </div>
              )}
              
              <div className="mt-3 p-3 bg-muted/50 rounded text-xs text-muted-foreground">
                <p className="font-medium mb-1">Performance Note:</p>
                <p>
                  VectorTA uses SIMD instructions and parallel processing to achieve near-native performance 
                  in the browser through WebAssembly.
                </p>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}