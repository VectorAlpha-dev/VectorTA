# Experimental CUDA Integration

This folder contains experimental CUDA support for the Rust-Backtester project. The implementation was functional but deemed too complex for general use due to build system requirements and platform-specific issues.

## What's Included

1. **CUDA Kernel** (`alma_kernel.cu`): GPU kernel for batch ALMA calculations
2. **Rust Wrapper** (`alma_wrapper.rs`): Rust interface using cudarc
3. **Build Configuration** (`build.rs.cuda`): CUDA compilation setup
4. **Integration Code**: Changes needed in main codebase (documented below)

## Key Features Implemented

- Batch ALMA calculations on GPU (single operations not supported due to memory transfer overhead)
- Manual opt-in only (never auto-selected due to FP32/FP64 precision concerns)
- Automatic fallback to CPU when CUDA unavailable
- Full test coverage with skip-if-unavailable logic
- Benchmark integration

## Build Requirements

- CUDA Toolkit (tested with 11.5, 12.1)
- Visual Studio C++ compiler on Windows
- cudarc crate with appropriate CUDA version feature

## Known Issues

1. **Windows**: Requires Visual Studio Developer Command Prompt
2. **WSL2**: CUDA detection issues with cudarc 0.12
3. **Build Complexity**: Multiple dependencies and platform-specific setup

## Integration Points

To re-enable CUDA support, you would need to:

1. **Cargo.toml**: Add `cuda` feature and cudarc dependency
2. **build.rs**: Replace with `build.rs.cuda`
3. **src/cuda/**: Copy this folder back to src/
4. **src/lib.rs**: Add `#[cfg(feature = "cuda")] pub mod cuda;`
5. **src/main.rs**: Add `#[cfg(feature = "cuda")] mod cuda;`
6. **src/utilities/enums.rs**: Add `CudaBatch` variant
7. **src/utilities/helpers.rs**: Add CUDA detection logic
8. **src/indicators/moving_averages/alma.rs**: Add CUDA kernel handling

## Performance Considerations

- CUDA showed promise for large batch operations
- Memory transfer overhead makes single operations inefficient
- Precision differences between CPU (f64) and GPU computation need validation

## Future Improvements

If revisiting this implementation:
1. Consider using pre-compiled PTX to avoid build complexity
2. Investigate newer cudarc versions for better WSL2 support
3. Add more indicators beyond ALMA
4. Create Python wheels with embedded CUDA support