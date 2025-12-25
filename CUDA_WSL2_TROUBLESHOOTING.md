# CUDA in WSL2 (Troubleshooting)

This repo’s CUDA feature uses the CUDA **Driver API** (via `cust`) to load and launch embedded PTX. If the Driver API can’t initialize, CUDA tests will **skip** (they call `cuda_available()`).

## Symptom

- `cuda_available()` returns `false`
- CUDA tests print something like:
  - `cuda_available: cust::init failed: OperatingSystemError`
- Minimal repro (Driver API) fails with:
  - `CUDA_ERROR_OPERATING_SYSTEM (304)` / “OS call failed or operation not supported on this OS”

## Quick Repro (this repo)

```bash
CUDA_PROBE_DEBUG=1 CARGO_NET_OFFLINE=true \
  cargo test --features cuda --test zscore_cuda -- --nocapture --test-threads=1
```

If the output contains `skipped - no CUDA device`, the probe failed.

## Minimal Driver-API Repro

```bash
cat > /tmp/cuinit.c <<'EOF'
#include <cuda.h>
#include <stdio.h>

int main() {
  CUresult r = cuInit(0);
  if (r != CUDA_SUCCESS) {
    const char* name = NULL;
    const char* str = NULL;
    cuGetErrorName(r, &name);
    cuGetErrorString(r, &str);
    printf("cuInit failed: %d name=%s str=%s\n", (int)r, name, str);
    return 1;
  }
  int n = 0;
  if (cuDeviceGetCount(&n) != CUDA_SUCCESS) {
    printf("cuDeviceGetCount failed\n");
    return 2;
  }
  printf("cuInit ok; devices=%d\n", n);
  return 0;
}
EOF

gcc -I/usr/local/cuda/include \
  -L/usr/lib/wsl/lib -Wl,-rpath,/usr/lib/wsl/lib \
  /tmp/cuinit.c -lcuda -o /tmp/cuinit

/tmp/cuinit
```

Expected: `cuInit ok; devices=1` (or more).

## What To Check First

Inside WSL2:

```bash
ls -l /dev/dxg
nvidia-smi
ldconfig -p | rg "libcuda\\.so\\.1"
```

You should see `/usr/lib/wsl/lib/libcuda.so.1` available (often listed before any distro-provided `libcuda`).

## Common Fixes

### 1) Update Windows + WSL, then restart WSL

From **Windows PowerShell** (admin not usually required):

```powershell
wsl --update
wsl --shutdown
```

Then relaunch your distro.

### 2) Update / reinstall the NVIDIA Windows driver (WSL-enabled)

Install a current NVIDIA Game Ready or Studio driver that supports CUDA on WSL2, then reboot Windows.

If `cuInit` started failing after a driver update, a clean reinstall or a rollback to the last known-good driver is often the fastest confirmation.

### 3) Remove Linux NVIDIA driver packages inside WSL (avoid conflicts)

WSL2 uses the Windows driver; you generally **do not** want the Ubuntu `nvidia-driver-*` stack inside the distro.

In WSL, check for distro NVIDIA driver libraries:

```bash
dpkg -l | rg -i "nvidia-driver|nvidia-utils|libnvidia|libcuda1|libcuda\\.so"
```

If you see packages like `libnvidia-compute-*`, purge them (keep CUDA toolkit packages you actually need, like `cuda-toolkit-13-0`):

```bash
sudo apt purge 'nvidia-*' 'libnvidia-*' 'libcuda*'
sudo apt autoremove
sudo ldconfig
```

Then re-run `/tmp/cuinit`.

### 4) If you’re running inside Docker on WSL

CUDA inside containers requires the NVIDIA container runtime/toolkit and proper device passthrough. If you’re running builds/tests in Docker, validate CUDA on the host WSL distro first (via `/tmp/cuinit`) before debugging the container layer.

## When It’s Fixed

Re-run:

```bash
CARGO_NET_OFFLINE=true cargo test --features cuda --tests -- --nocapture --test-threads=1
```

CUDA tests should **run** (not skip) and compare GPU vs CPU outputs.

