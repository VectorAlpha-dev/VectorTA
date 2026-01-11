#!/usr/bin/env python3
"""
Indicator QA audit tool.

Given an indicator name (e.g., "alma", "bollinger_bands_width"), this script:
- Locates the Rust indicator source file under src/indicators/**.
- Locates the CUDA wrapper (if any) under src/cuda/**.
- Emits stripped code artifacts that keep only the top comment block and remove
  all subsequent comments (line and block) for easier final review.
- Optionally runs focused unit tests for just this indicator:
  - Rust unit tests in the indicator module (scalar path).
  - Rust CUDA integration test (if a wrapper and test file exist; skips gracefully when CUDA is unavailable).
  - Python binding tests for this indicator (CPU) and CUDA-binding tests (if present).

Outputs land in target/audit/<indicator>/.

Usage examples:
  python scripts/indicator_audit.py alma
  python scripts/indicator_audit.py bollinger_bands_width --python --rust --with-cuda
  python scripts/indicator_audit.py rsi --no-tests --emit-only

Notes:
- This script does not modify source files; it only writes artifacts and runs tests.
- CUDA builds may require nvcc. If nvcc is missing, use --cuda-stub to let the build
  generate placeholder PTX via nvcc_stub.sh. CUDA tests will then skip at runtime.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]


def die(msg: str, code: int = 2) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def run(cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None) -> int:
    print("$", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(cwd) if cwd else None, env=env)


def capture(cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None) -> Tuple[int, str]:
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def find_indicator_file(indicator: str) -> Path:
    candidates = list((REPO_ROOT / "src/indicators").rglob(f"{indicator}.rs"))
    if not candidates:
        die(f"could not find src/indicators/**/{indicator}.rs")
    
    candidates.sort(key=lambda p: len(str(p)))
    return candidates[0]


def find_cuda_wrapper(indicator: str) -> Optional[Path]:
    pats = [f"{indicator}_wrapper.rs", f"**/{indicator}_wrapper.rs"]
    base = REPO_ROOT / "src/cuda"
    for pat in pats:
        matches = list(base.rglob(pat))
        if matches:
            matches.sort(key=lambda p: len(str(p)))
            return matches[0]
    return None


def detect_python_tests(indicator: str) -> Tuple[List[Path], List[Path]]:
    """Identify Python CPU/CUDA tests that exercise the given indicator.

    Strategy:
    - Prefer exact filename matches: test_<indicator>.py and test_<indicator>_cuda.py
    - Otherwise, scan file content for calls like: my_project.<indicator>(...), ti.<indicator>(...)
      For CUDA: ....<indicator>_cuda_... occurrences.
    - Ignore docstring-only mentions by requiring a dotted call prefix.
    """
    py_dir = REPO_ROOT / "tests/python"
    cpu: List[Path] = []
    cuda: List[Path] = []

    exact_cpu = py_dir / f"test_{indicator}.py"
    exact_cuda = py_dir / f"test_{indicator}_cuda.py"
    if exact_cpu.exists():
        cpu.append(exact_cpu)
    if exact_cuda.exists():
        cuda.append(exact_cuda)

    
    call_cpu = re.compile(rf"\b(?:my_project|ta_indicators|ti)\.{re.escape(indicator)}\s*\(")
    call_cuda = re.compile(rf"\b(?:my_project|ta_indicators|ti)\.{re.escape(indicator)}_cuda")

    for p in py_dir.glob("test_*.py"):
        if p in cpu or p in cuda:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if p.name.endswith("_cuda.py"):
            if call_cuda.search(text):
                cuda.append(p)
        else:
            if call_cpu.search(text):
                cpu.append(p)
    return cpu, cuda


def detect_rust_cuda_test(indicator: str) -> Optional[Path]:
    tests_dir = REPO_ROOT / "tests"
    
    name_matches = list(tests_dir.glob(f"{indicator}_cuda.rs"))
    if name_matches:
        return name_matches[0]
    
    for p in tests_dir.glob("*_cuda.rs"):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if re.search(rf"\b{re.escape(indicator)}\b", text):
            return p
    return None


def compute_mod_path(indicator_file: Path) -> str:
    
    rel = indicator_file.relative_to(REPO_ROOT / "src/indicators")
    parts = list(rel.parts)
    assert parts[-1].endswith(".rs")
    parts[-1] = parts[-1][:-3]  
    mod_path = "indicators::" + "::".join([p for p in parts])
    return mod_path


def strip_comments_keep_header(src: str) -> str:
    
    lines = src.splitlines()
    header_end = 0
    in_block = False
    for i, line in enumerate(lines):
        s = line.strip()
        if in_block:
            header_end = i + 1
            if "*/" in s:
                in_block = False
            continue
        if not s:
            header_end = i + 1
            continue
        if s.startswith("/*"):
            in_block = True
            header_end = i + 1
            
            if s.endswith("*/") or "*/" in s:
                in_block = False
            continue
        if s.startswith("//"):
            header_end = i + 1
            continue
        
        break

    header = "\n".join(lines[:header_end])
    body = "\n".join(lines[header_end:])

    
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    
    body = re.sub(r"(^|\s)//.*?$", r"\1", body, flags=re.M)
    
    body = re.sub(r"\n{3,}", "\n\n", body)
    
    body = body.strip("\n")
    return (header + "\n\n" + body + "\n") if header else (body + "\n")


def write_artifacts(indicator: str, ind_file: Path, cuda_file: Optional[Path]) -> Path:
    out_dir = REPO_ROOT / "target/audit" / indicator
    out_dir.mkdir(parents=True, exist_ok=True)

    ind_text = ind_file.read_text(encoding="utf-8", errors="ignore")
    (out_dir / f"{indicator}.orig.rs").write_text(ind_text, encoding="utf-8")
    (out_dir / f"{indicator}.stripped.rs").write_text(strip_comments_keep_header(ind_text), encoding="utf-8")

    if cuda_file:
        cuda_text = cuda_file.read_text(encoding="utf-8", errors="ignore")
        (out_dir / f"{indicator}.cuda_wrapper.orig.rs").write_text(cuda_text, encoding="utf-8")
        (out_dir / f"{indicator}.cuda_wrapper.stripped.rs").write_text(strip_comments_keep_header(cuda_text), encoding="utf-8")

    return out_dir


def have_gpu() -> bool:
    rc, _ = capture(["bash", "-lc", "nvidia-smi >/dev/null 2>&1"])
    return rc == 0


def have_nvcc() -> bool:
    rc, _ = capture(["bash", "-lc", "command -v nvcc >/dev/null 2>&1"])
    return rc == 0


def ensure_python_env(auto_install: bool) -> bool:
    try:
        import importlib  
        import numpy  
        
        try:
            import maturin  
        except Exception:
            pass
        return True
    except Exception:
        if not auto_install:
            print("Python deps not present; run: pip install -r tests/python/requirements.txt")
            return False
        rc = run([sys.executable, "-m", "pip", "install", "-r", str(REPO_ROOT / "tests/python/requirements.txt")])
        return rc == 0


def build_python_module(with_cuda: bool, use_cuda_stub: bool) -> bool:
    env = os.environ.copy()
    features = ["python"]
    if with_cuda:
        features.append("cuda")
        if use_cuda_stub and not have_nvcc():
            env["NVCC"] = str(REPO_ROOT / "nvcc_stub.sh")
            env["CUDA_PLACEHOLDER_ON_FAIL"] = "1"
    cmd = ["bash", "-lc", f"maturin develop --features '{','.join(features)}' --release"]
    return run(cmd, cwd=REPO_ROOT, env=env) == 0


def run_rust_scalar_tests(mod_path: str) -> bool:
    cmd = [
        "bash",
        "-lc",
        f"cargo test --lib {mod_path} -- --nocapture",
    ]
    return run(cmd, cwd=REPO_ROOT) == 0


def run_rust_cuda_test(test_file: Path, use_cuda_stub: bool) -> bool:
    name = test_file.stem  
    env = os.environ.copy()
    if use_cuda_stub and not have_nvcc():
        env["NVCC"] = str(REPO_ROOT / "nvcc_stub.sh")
        env["CUDA_PLACEHOLDER_ON_FAIL"] = "1"
    cmd = [
        "bash",
        "-lc",
        f"cargo test --features cuda --test {name} -- --nocapture",
    ]
    return run(cmd, cwd=REPO_ROOT, env=env) == 0


def run_pytests(py_files: List[Path], label: str) -> bool:
    if not py_files:
        print(f"[pytest] no {label} tests found for this indicator")
        return True
    rels = [str(p.relative_to(REPO_ROOT)) for p in py_files]
    cmd = ["bash", "-lc", f"pytest -q {' '.join(rels)}"]
    return run(cmd, cwd=REPO_ROOT) == 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Indicator QA audit")
    ap.add_argument("indicator", help="indicator name, e.g. alma, bollinger_bands_width")
    ap.add_argument("--emit-only", action="store_true", help="only emit stripped artifacts; do not run tests")
    ap.add_argument("--rust", action="store_true", help="run Rust tests (scalar module tests)")
    ap.add_argument("--python", action="store_true", help="run Python binding tests")
    ap.add_argument("--with-cuda", action="store_true", help="include CUDA tests (Rust + Python) when possible")
    ap.add_argument("--cuda-stub", action="store_true", help="use nvcc_stub.sh to allow building with --features cuda when nvcc is unavailable")
    ap.add_argument("--auto-install-deps", action="store_true", help="pip install Python test deps automatically if missing")
    args = ap.parse_args()

    indicator = args.indicator.strip()
    ind_file = find_indicator_file(indicator)
    cuda_file = find_cuda_wrapper(indicator)
    out_dir = write_artifacts(indicator, ind_file, cuda_file)

    mod_path = compute_mod_path(ind_file)
    py_cpu, py_cuda = detect_python_tests(indicator)
    rust_cuda_test = detect_rust_cuda_test(indicator) if cuda_file else None

    print("\n=== Indicator Audit Summary ===")
    print(f"indicator:           {indicator}")
    print(f"rust file:           {ind_file}")
    print(f"cuda wrapper:        {cuda_file if cuda_file else '(none)'}")
    print(f"artifacts dir:       {out_dir}")
    print(f"module path:         {mod_path}")
    print(f"pytest (cpu):        {[p.name for p in py_cpu] if py_cpu else 'none'}")
    print(f"pytest (cuda):       {[p.name for p in py_cuda] if py_cuda else 'none'}")
    print(f"rust cuda test:      {rust_cuda_test.name if rust_cuda_test else 'none'}")
    print("==============================\n")

    if args.emit_only:
        return

    
    run_rust = args.rust or (not args.python)
    run_python = args.python or (not args.rust)

    ok = True

    if run_rust:
        print("[rust] running scalar module tests...")
        ok &= run_rust_scalar_tests(mod_path)

    

    if args.with_cuda and rust_cuda_test:
        print("\n[rust] running CUDA integration test...")
        ok &= run_rust_cuda_test(rust_cuda_test, use_cuda_stub=args.cuda_stub)
    elif args.with_cuda:
        print("\n[rust] no CUDA test file detected; skipping")

    if run_python:
        if not ensure_python_env(auto_install=args.auto_install_deps):
            die("python test environment missing; re-run with --auto-install-deps or install manually")

        
        want_cuda_build = args.with_cuda and (py_cuda or rust_cuda_test or cuda_file is not None)
        print("\n[python] building extension module (features: {} )...".format("python,cuda" if want_cuda_build else "python"))
        if not build_python_module(with_cuda=want_cuda_build, use_cuda_stub=args.cuda_stub):
            die("maturin build failed (see output above)")

        if py_cpu:
            print("\n[pytest] running CPU binding tests...")
            ok &= run_pytests(py_cpu, label="cpu")
        else:
            print("\n[pytest] no CPU binding tests detected for this indicator; skipping")

        if args.with_cuda and py_cuda:
            print("\n[pytest] running CUDA binding tests...")
            ok &= run_pytests(py_cuda, label="cuda")
        elif args.with_cuda:
            print("\n[pytest] no CUDA binding tests detected for this indicator; skipping")

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
