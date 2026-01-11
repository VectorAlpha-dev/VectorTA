#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


KEEP_PREFIXES = (b"///", b"//!")


@dataclass(frozen=True)
class StripStats:
    files_seen: int
    files_changed: int
    bytes_removed: int


def _run_git_ls_files(repo_root: Path) -> list[str]:
    out = subprocess.check_output(["git", "ls-files"], cwd=repo_root)
    return out.decode("utf-8", errors="replace").splitlines()


def _is_allowed_path(rel: str) -> bool:
    if rel.startswith("third_party/"):
        return False
    if rel.startswith(".git/"):
        return False
    return True


def _has_allowed_ext(path: str, exts: set[str]) -> bool:
    lower = path.lower()
    for ext in exts:
        if lower.endswith(ext):
            return True
    return False


def _is_ascii_ident_char(b: int) -> bool:
    return (ord("a") <= b <= ord("z")) or (ord("A") <= b <= ord("Z")) or (ord("0") <= b <= ord("9")) or b == ord("_")


def _strip_double_slash_comments_c_like(data: bytes) -> bytes:
    out = bytearray()
    i = 0
    n = len(data)

    in_line_comment = False
    in_block_comment = False
    in_single_quote = False
    in_double_quote = False
    in_backtick = False

    # Rust raw strings: b?r####" ... "####, tracked by hash count when active.
    rust_raw_hashes: Optional[int] = None

    # C++ raw strings: R"delim(... )delim"
    cpp_raw_delim: Optional[bytes] = None

    while i < n:
        b = data[i]

        if in_line_comment:
            if b == ord("\n"):
                in_line_comment = False
                out.append(b)
            else:
                # swallow comment bytes (including \r if CRLF)
                pass
            i += 1
            continue

        if in_block_comment:
            # We are not stripping block comments, just need to avoid starting line comments inside.
            out.append(b)
            if b == ord("*") and i + 1 < n and data[i + 1] == ord("/"):
                out.append(data[i + 1])
                i += 2
                in_block_comment = False
            else:
                i += 1
            continue

        if rust_raw_hashes is not None:
            out.append(b)
            if b == ord('"'):
                # Check for closing hashes.
                hashes = rust_raw_hashes
                if hashes == 0:
                    rust_raw_hashes = None
                else:
                    if i + hashes < n and data[i + 1 : i + 1 + hashes] == (b"#" * hashes):
                        out.extend(data[i + 1 : i + 1 + hashes])
                        i += 1 + hashes
                        rust_raw_hashes = None
                        continue
            i += 1
            continue

        if cpp_raw_delim is not None:
            out.append(b)
            # Look for )delim"
            if b == ord(")") and cpp_raw_delim is not None:
                delim = cpp_raw_delim
                tail = b")" + delim + b'"'
                if data[i : i + len(tail)] == tail:
                    # we already wrote ')', write rest and exit raw string
                    out.extend(data[i + 1 : i + len(tail)])
                    i += len(tail)
                    cpp_raw_delim = None
                    continue
            i += 1
            continue

        if in_single_quote:
            out.append(b)
            if b == ord("\\") and i + 1 < n:
                out.append(data[i + 1])
                i += 2
                continue
            if b == ord("'"):
                in_single_quote = False
            i += 1
            continue

        if in_double_quote:
            out.append(b)
            if b == ord("\\") and i + 1 < n:
                out.append(data[i + 1])
                i += 2
                continue
            if b == ord('"'):
                in_double_quote = False
            i += 1
            continue

        if in_backtick:
            out.append(b)
            if b == ord("\\") and i + 1 < n:
                out.append(data[i + 1])
                i += 2
                continue
            if b == ord("`"):
                in_backtick = False
            i += 1
            continue

        # NORMAL state
        if b == ord("/") and i + 1 < n:
            nxt = data[i + 1]
            if nxt == ord("*"):
                in_block_comment = True
                out.append(b)
                out.append(nxt)
                i += 2
                continue
            if nxt == ord("/"):
                # preserve doc-style prefixes (mostly Rust/TS directives)
                if i + 3 <= n and data[i : i + 3] == b"///":
                    out.extend(b"///")
                    i += 3
                    continue
                if i + 3 <= n and data[i : i + 3] == b"//!":
                    out.extend(b"//!")
                    i += 3
                    continue
                in_line_comment = True
                i += 2
                continue

        # Rust raw string start: optional b then r then hashes then "
        if b in (ord("b"), ord("r")):
            start_i = i
            j = i

            # Optional leading 'b'
            if data[j] == ord("b"):
                if j + 1 < n and data[j + 1] == ord("r"):
                    j += 2
                else:
                    j = start_i
            if j == start_i and data[j] == ord("r"):
                j += 1

            if j != start_i:
                hashes = 0
                while j < n and data[j] == ord("#"):
                    hashes += 1
                    j += 1
                if j < n and data[j] == ord('"'):
                    # Ensure we aren't in an identifier like `driver`
                    prev = data[start_i - 1] if start_i > 0 else None
                    if prev is None or not _is_ascii_ident_char(prev):
                        out.extend(data[start_i : j + 1])
                        i = j + 1
                        rust_raw_hashes = hashes
                        continue

        # C++ raw string start: R"delim(
        if b == ord("R") and i + 1 < n and data[i + 1] == ord('"'):
            # Parse delimiter until '('
            j = i + 2
            while j < n and data[j] != ord("(") and data[j] != ord("\n") and data[j] != ord("\r"):
                j += 1
            if j < n and data[j] == ord("("):
                delim = data[i + 2 : j]
                out.extend(data[i : j + 1])
                i = j + 1
                cpp_raw_delim = delim
                continue

        if b == ord("'"):
            in_single_quote = True
            out.append(b)
            i += 1
            continue
        if b == ord('"'):
            in_double_quote = True
            out.append(b)
            i += 1
            continue
        if b == ord("`"):
            in_backtick = True
            out.append(b)
            i += 1
            continue

        out.append(b)
        i += 1

    return bytes(out)


def _strip_file(path: Path) -> tuple[bool, int]:
    original = path.read_bytes()
    stripped = _strip_double_slash_comments_c_like(original)
    if stripped == original:
        return False, 0
    path.write_bytes(stripped)
    return True, max(0, len(original) - len(stripped))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Strip non-doc `//` comments from tracked code-like files.")
    ap.add_argument(
        "--ext",
        action="append",
        default=[],
        help="File extension to process (include dot), can be specified multiple times.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Show what would change without editing files.")
    args = ap.parse_args(argv)

    repo_root = Path.cwd()

    exts = set(args.ext) if args.ext else {
        ".rs",
        ".c",
        ".h",
        ".cc",
        ".cpp",
        ".cxx",
        ".hpp",
        ".cu",
        ".cuh",
        ".js",
        ".cjs",
        ".mjs",
        ".ts",
        ".tsx",
        ".jsx",
        ".ptx",
    }

    files = [p for p in _run_git_ls_files(repo_root) if _is_allowed_path(p) and _has_allowed_ext(p, exts)]

    files_seen = 0
    files_changed = 0
    bytes_removed = 0

    try:
        for rel in files:
            files_seen += 1
            path = repo_root / rel
            if not path.is_file():
                continue
            original = path.read_bytes()
            stripped = _strip_double_slash_comments_c_like(original)
            if stripped == original:
                continue

            files_changed += 1
            bytes_removed += max(0, len(original) - len(stripped))

            if args.dry_run:
                print(rel)
                continue

            path.write_bytes(stripped)
    except BrokenPipeError:
        return 0

    stats = StripStats(files_seen=files_seen, files_changed=files_changed, bytes_removed=bytes_removed)
    print(f"files_seen={stats.files_seen} files_changed={stats.files_changed} bytes_removed={stats.bytes_removed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
