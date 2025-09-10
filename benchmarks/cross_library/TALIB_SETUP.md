# TA-LIB Setup Guide for Windows

This guide will help you install and configure TA-LIB for use with the cross-library benchmarks.

## Prerequisites

- Windows 10/11 (64-bit)
- Visual Studio 2019+ or Build Tools for Visual Studio (for C++ compilation)
- Rust toolchain

## Installation Steps

### Step 1: Download TA-LIB

Choose one of the following options:

#### Option A: MSI Installer (Recommended)
1. Download [ta-lib-0.6.4-windows-x86_64.msi](https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-windows-x86_64.msi)
2. Double-click the `.msi` file
3. Follow the installation wizard
4. Default installation path: `C:\Program Files\TA-Lib`

#### Option B: ZIP Package
1. Download [ta-lib-0.6.4-windows-x86_64.zip](https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-windows-x86_64.zip)
2. Extract to a location like `C:\ta-lib`
3. Note the extraction path for the next step

### Step 2: Set Environment Variable

You need to set the `TALIB_PATH` environment variable to point to your TA-LIB installation.

#### Method 1: System Environment Variables (Permanent)
1. Press `Win + X` and select "System"
2. Click "Advanced system settings"
3. Click "Environment Variables"
4. Under "System variables", click "New"
5. Set:
   - Variable name: `TALIB_PATH`
   - Variable value: `C:\Program Files\TA-Lib` (or your installation path)
6. Click "OK" to save
7. Restart your terminal/IDE for changes to take effect

#### Method 2: PowerShell (Temporary - current session only)
```powershell
$env:TALIB_PATH = "C:\Program Files\TA-Lib"
```

#### Method 3: Command Prompt (Temporary - current session only)
```cmd
set TALIB_PATH=C:\Program Files\TA-Lib
```

### Step 3: Verify Installation

Check that the environment variable is set:
```powershell
echo $env:TALIB_PATH
```

Verify the TA-LIB files exist:
```powershell
dir "$env:TALIB_PATH\include"
dir "$env:TALIB_PATH\lib"
```

You should see:
- Header files (`.h`) in the `include` directory
- Library files (`.lib`) in the `lib` directory

## Building the Benchmarks with TA-LIB

Once TA-LIB is installed and configured:

```bash
cd benchmarks/cross_library
cargo build --release --features talib
```

If successful, you'll see TA-LIB being linked during compilation.

## Running TA-LIB Benchmarks

```bash
cargo bench --features talib
```

This will run benchmarks comparing:
- Rust native implementation
- Rust via FFI
- Tulip Indicators (C)
- TA-LIB (C)

## Troubleshooting

### "TA-Lib not found" Warning

If you see this warning during build:
```
warning: TA-Lib not found. Set TALIB_PATH environment variable.
```

**Solution**: Ensure `TALIB_PATH` is set correctly and points to the TA-LIB installation directory.

### Linking Errors

If you get linking errors like `LINK : fatal error LNK1181: cannot open input file 'ta_lib.lib'`:

**Solutions**:
1. Verify TA-LIB is installed for the correct architecture (64-bit for 64-bit Rust)
2. Check that `$TALIB_PATH/lib` contains the library files
3. Try using the full path without spaces (install to `C:\ta-lib` instead of Program Files)

### Bindgen Errors

If bindgen fails to generate bindings:

**Solutions**:
1. Install LLVM/Clang: Download from [LLVM releases](https://github.com/llvm/llvm-project/releases)
2. Add LLVM to PATH or set `LIBCLANG_PATH` environment variable
3. Ensure Visual Studio C++ tools are installed

### Version Mismatch

TA-LIB 0.6.x changed the library name from `ta_lib` to `ta-lib`. If you have an older version:

**Solution**: Update to TA-LIB 0.6.4 or newer using the links above.

## Testing TA-LIB Integration

Run the test example to verify everything works:

```bash
cargo run --example test_talib --features talib
```

If successful, you should see TA-LIB function calls working correctly.

## Additional Resources

- [TA-LIB Official Site](https://ta-lib.org/)
- [TA-LIB GitHub](https://github.com/TA-Lib/ta-lib)
- [TA-LIB C/C++ API Documentation](https://ta-lib.org/api/)

## Notes for Developers

- TA-LIB functions are prefixed with `TA_` (e.g., `TA_SMA`, `TA_RSI`)
- Input/output arrays use `double` (f64 in Rust)
- Most functions return `TA_RetCode` for error handling
- The library is thread-safe for different indicator calls