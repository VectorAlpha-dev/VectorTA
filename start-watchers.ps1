# start-watchers.ps1
$workDir = 'C:\Rust Projects\my_project'

$commands = @(
  cargo watch -x check
  cargo watch --features nightly-avx -x check
  cargo watch --features python -x check
  cargo watch --features wasm -x check
)

foreach ($cmd in $commands) {
  Start-Process -FilePath 'powershell.exe' `
    -WorkingDirectory $workDir `
    -ArgumentList '-NoExit','-NoLogo','-Command', $cmd
}


	
I have a guide for you to improve the accuracy of one of my existing cuda kernels. Evaluate critically to ensure that it will indeed improve accuracy without a significant reduction in performance. Proceed if the guide has merit. If you implement it then run unit tests and tighten unit test tolerance to confirm that it worked. Guide: "


  - You are a Codex CLI coding agent working in an existing Rust indicators repository. Your goal is to:          
      - Optimize the scalar implementation of a given indicator first.                                            
      - Implement SIMD for single-series if viable.                                                               
      - Implement row-specific optimized variants for the batch function if viable.                               
      - Keep changes minimal, focused, and in the style of src/indicators/moving_averages/alma.rs (gold standard  
  for API, docs, warmup, allocation, and optimization patterns).                                                  
                                                                                                                  
  Key Rules                                                                                                       
                                                                                                                  
  - Do not change unit test reference values; fix implementations instead.                                        
  - Each scalar optimization requires running unit tests and benchmarks for that indicator.                       
  - If the scalar indicator calls helpers, inline/loop-jam them where possible unless they use dynamic dispatch   
  (e.g., “MA” selector).                                                                                          
  - Maintain safety: optimize the scalar path as much as possible without converting it to unsafe if it’s         
  currently safe.                                                                                                 
  - Respect feature-gating and runtime selection patterns already used in the repo.                               
  - If SIMD or row-specific kernels already exist, review, validate, and iterate; on repeated failure or          
  underperformance, revert selection to the last known-good kernels and document why.                             
  - Ensure the indicator is registered in benches/indicator_benchmark.rs before doing SIMD work so benchmarking   
  is possible.                                                                                                    
                                                                                                                  
  Plan (use update_plan)                                                                                          
                                                                                                                  
  1. Baseline read + verify registration                                                                          
  2. Baseline test + bench                                                                                        
  3. Optimize scalar path                                                                                         
  4. Re-test + re-bench                                                                                           
  5. SIMD review/implement                                                                                        
  6. Re-test + re-bench (SIMD)                                                                                    
  7. Row-specific batch review/implement                                                                          
  8. Re-test + re-bench (batch)                                                                                   
  9. Document decisions                                                                                           
                                                                                                                  
  Keep exactly one step in progress; mark completed as you go. Adjust the plan if new work emerges.               
                                                                                                                  
  Baseline Tasks                                                                                                  
                                                                                                                  
  - Read AGENTS.md in scope; follow its conventions strictly.                                                     
  - Identify the target indicator files and their batch equivalents under src/indicators/**.                      
  - Confirm bench registration in benches/indicator_benchmark.rs. If missing, add minimal registration consistent 
  with existing patterns.                                                                                         
  - Establish a baseline:                                                                                         
      - Build stable: cargo build                                                                                 
      - Scalar tests for the indicator: cargo test --lib indicators::<module>::<indicator> -- --nocapture         
      - SIMD tests (if applicable): cargo +nightly test --features nightly-avx --lib                              
  indicators::<module>::<indicator> -- --nocapture                                                                
      - Discover benches: cargo bench --bench indicator_benchmark -- --list                                       
      - Run baseline benches for the indicator (single-series first):                                             
          - RUSTFLAGS="-C target-cpu=native" cargo bench --bench indicator_benchmark -- <ind>/<ind>_scalar/100k   
                                                                                                                  
  Phase 1: Scalar Optimization                                                                                    
                                                                                                                  
  - Use alma.rs as the implementation blueprint:                                                                  
      - API shape, warmup handling, zero-copy/uninitialized outputs, no O(N) temporaries for outputs.             
      - Prefer cache-aligned vectors and allocation helpers used elsewhere in the repo.                           
      - Hoist invariants, reduce branches, fuse loops/loop-jam across helper calls when safe and beneficial.      
      - Use assert_same_len!, once_cell instead of lazy_static, and document public items with ///.               
  - After each meaningful change:                                                                                 
      - Run indicator-specific tests; do not modify reference outputs.                                            
      - Re-run baseline benches; record deltas; keep changes only if net improvements are consistent.             
                                                                                                                  
  Phase 2: SIMD Implementation (if viable)                                                                        
                                                                                                                  
  - Guard with #[cfg(feature = "nightly-avx")] and #[cfg(target_arch = "x86_64")].                                
  - Keep scalar as the reference path; SIMD must match outputs within existing tolerances (bit-for-bit or within  
  permitted error).                                                                                               
  - Follow repo patterns:                                                                                         
      - Use detect_best_kernel()/detect_best_batch_kernel() for AVX512 → AVX2 → SSE2 → Scalar selection.          
      - Isolate intrinsics; minimize unsafe; keep hot loops tight; no unnecessary temporaries.                    
      - Maintain warmup prefix behavior and identical API semantics to the scalar function.                       
  - Validate:                                                                                                     
      - Stable build passes.                                                                                      
      - Nightly + nightly-avx tests for the indicator pass.                                                       
      - Benchmarks show >5% improvement vs scalar at realistic sizes (10k, 100k) using RUSTFLAGS="-C target-      
  cpu=native".                                                                                                    
  - If SIMD underperforms or is unstable after iterations:                                                        
      - Keep SIMD stubs delegating to scalar or short-circuit runtime selection to scalar.                        
      - Add a concise module-level note explaining why SIMD is disabled or de-prioritized.                        
                                                                                                                  
  Phase 3: Row-Specific Batch (if viable)                                                                         
                                                                                                                  
  - Attempt only if there’s clear shared precomputation across rows and a chance to reduce redundant work (e.g.,  
  reusable weights, norms).                                                                                       
  - Keep memory access contiguous/aligned; prefer SoA/time-major layouts or tiling consistent with the repo.      
  - Feature-gate under nightly-avx; wire via the batch kernel selector so AVX512 → AVX2 → SSE2 → Scalar paths are 
  chosen at runtime only when beneficial.                                                                         
  - Validation criteria mirror single-series SIMD:                                                                
      - Accuracy matches scalar batch outputs (tests unchanged).                                                  
      - Benchmarks show >5% improvement vs non row-specific batch for realistic sizes.                            
  - If underperformance persists, disable selection to row-specific SIMD batch, keep code for future, and document
  the rationale.                                                                                                  
                                                                                                                  
  Validation                                                                                                      
                                                                                                                  
  - Tests:                                                                                                        
      - Scalar-only suite: cargo test --lib indicators:: -- --nocapture                                           
      - Indicator-specific scalar tests: cargo test --lib indicators::<module>::<indicator> -- --nocapture        
      - Nightly SIMD tests: cargo +nightly test --features nightly-avx --lib indicators::<module>::<indicator>    
  -- --nocapture                                                                                                  
  - Benches:                                                                                                      
      - List: cargo bench --bench indicator_benchmark -- --list                                                   
      - Scalar: RUSTFLAGS="-C target-cpu=native" cargo bench --bench indicator_benchmark -- <ind>/                
  <ind>_scalar/100k                                                                                               
      - AVX2/AVX512: RUSTFLAGS="-C target-cpu=native" cargo bench --features nightly-avx --bench                  
  indicator_benchmark -- <ind>/<ind>_avx2/100k                                                                    
  - Treat batch benches as advisory unless row-specific optimized kernels exist and are wired in.                 
                                                                                                                  
  Reversion Policy                                                                                                
                                                                                                                  
  - If regressions (accuracy or performance) are unresolved after a few focused iterations:                       
      - Revert runtime selection to the last known-good (scalar or existing SIMD/batch).                          
      - Leave improved code paths present but disabled at runtime if they don’t meet criteria.                    
      - Add a short decision note at the top of the module describing status and rationale.                       
                                                                                                                  
  Deliverables                                                                                                    
                                                                                                                  
  - Focused patches to the indicator (and benches registration if missing).                                       
  - Brief module-level decision note(s) describing SIMD/batch status and rationale.                               
  - Short benchmark notes (commands + observed deltas) demonstrating improvements or reasons for disabling SIMD/  
  batch.                                                                                                          
  - No changes to unit test reference values.                                                                     
                                                                                                                  
  Reference Files                                                                                                 
                                                                                                                  
  - Gold standard: src/indicators/moving_averages/alma.rs                                                         
  - Bench registration: benches/indicator_benchmark.rs     
To start off, I have a possible optimized scalar kernel that is untested and not benchmarked. You may want to initially benchmark the indicator so that you can compare with the optimzied scalar variant. Initial optimized scalar variant: "