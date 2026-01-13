#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


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
    if rel == "Cargo.lock":
        return False
    return True


def _is_ascii_ident_char(b: int) -> bool:
    return (ord("a") <= b <= ord("z")) or (ord("A") <= b <= ord("Z")) or (ord("0") <= b <= ord("9")) or b == ord("_")


def _is_ascii_hexdigit(b: int) -> bool:
    return (ord("0") <= b <= ord("9")) or (ord("a") <= b <= ord("f")) or (ord("A") <= b <= ord("F"))


def _try_parse_rust_char_literal(data: bytes, start: int) -> Optional[int]:
    n = len(data)
    if start >= n or data[start] != ord("'"):
        return None
    if start + 1 >= n:
        return None

    b1 = data[start + 1]
    if b1 in (ord("\n"), ord("\r")):
        return None

    if b1 == ord("\\"):
        if start + 2 >= n:
            return None
        esc = data[start + 2]
        if esc in (ord("n"), ord("r"), ord("t"), ord("0"), ord("\\"), ord("'"), ord('"')):
            end = start + 3
            if end < n and data[end] == ord("'"):
                return end + 1
            return None
        if esc == ord("x"):
            if start + 5 < n and _is_ascii_hexdigit(data[start + 3]) and _is_ascii_hexdigit(data[start + 4]) and data[start + 5] == ord("'"):
                return start + 6
            return None
        if esc == ord("u"):
            if start + 3 >= n or data[start + 3] != ord("{"):
                return None
            j = start + 4
            digits = 0
            while j < n and data[j] != ord("}"):
                if not _is_ascii_hexdigit(data[j]):
                    return None
                digits += 1
                if digits > 6:
                    return None
                j += 1
            if digits == 0:
                return None
            if j >= n or data[j] != ord("}"):
                return None
            if j + 1 < n and data[j + 1] == ord("'"):
                return j + 2
            return None
        return None

    if b1 < 0x80:
        if _is_ascii_ident_char(b1):
            end = start + 2
            if end < n and data[end] == ord("'"):
                return end + 1
            return None
        if b1 in (ord("'"), ord("\\")):
            return None
        end = start + 2
        if end < n and data[end] == ord("'"):
            return end + 1
        return None

    first = b1
    if 0xC2 <= first <= 0xDF:
        size = 2
    elif 0xE0 <= first <= 0xEF:
        size = 3
    elif 0xF0 <= first <= 0xF4:
        size = 4
    else:
        return None
    if start + 1 + size >= n:
        return None
    for k in range(1, size):
        if not (0x80 <= data[start + 1 + k] <= 0xBF):
            return None
    end = start + 1 + size
    if data[end] == ord("'"):
        return end + 1
    return None


def _strip_rust_comments(data: bytes) -> bytes:
    out = bytearray()
    i = 0
    n = len(data)

    in_line = False
    block_depth = 0
    in_double = False
    in_backtick = False
    rust_raw_hashes: Optional[int] = None

    while i < n:
        b = data[i]

        if in_line:
            if b == ord("\n"):
                in_line = False
                out.append(b)
            i += 1
            continue

        if block_depth > 0:
            if b == ord("/") and i + 1 < n and data[i + 1] == ord("*"):
                block_depth += 1
                i += 2
                continue
            if b == ord("*") and i + 1 < n and data[i + 1] == ord("/"):
                block_depth -= 1
                i += 2
                if block_depth == 0:
                    if out and out[-1] not in (ord(" "), ord("\t"), ord("\n"), ord("\r")):
                        out.append(ord(" "))
                continue
            i += 1
            continue

        if rust_raw_hashes is not None:
            out.append(b)
            if b == ord('"'):
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

        if in_double:
            out.append(b)
            if b == ord("\\") and i + 1 < n:
                out.append(data[i + 1])
                i += 2
                continue
            if b == ord('"'):
                in_double = False
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

        if b == ord("/") and i + 1 < n:
            nxt = data[i + 1]
            if nxt == ord("/"):
                in_line = True
                i += 2
                continue
            if nxt == ord("*"):
                block_depth = 1
                i += 2
                continue

        if b in (ord("b"), ord("r")):
            start_i = i
            j = i
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
                    prev = data[start_i - 1] if start_i > 0 else None
                    if prev is None or not _is_ascii_ident_char(prev):
                        out.extend(data[start_i : j + 1])
                        i = j + 1
                        rust_raw_hashes = hashes
                        continue

        if b == ord("'"):
            end = _try_parse_rust_char_literal(data, i)
            if end is not None:
                out.extend(data[i:end])
                i = end
                continue
            out.append(b)
            i += 1
            continue

        if b == ord('"'):
            in_double = True
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


def _strip_c_like_comments(data: bytes) -> bytes:
    out = bytearray()
    i = 0
    n = len(data)

    in_line = False
    in_block = False
    in_single = False
    in_double = False
    in_backtick = False

    cpp_raw_delim: Optional[bytes] = None

    while i < n:
        b = data[i]

        if in_line:
            if b == ord("\n"):
                in_line = False
                out.append(b)
            i += 1
            continue

        if in_block:
            if b == ord("*") and i + 1 < n and data[i + 1] == ord("/"):
                in_block = False
                i += 2
                if out and out[-1] not in (ord(" "), ord("\t"), ord("\n"), ord("\r")):
                    out.append(ord(" "))
                continue
            i += 1
            continue

        if cpp_raw_delim is not None:
            out.append(b)
            if b == ord(")"):
                delim = cpp_raw_delim
                tail = b")" + delim + b'"'
                if data[i : i + len(tail)] == tail:
                    out.extend(data[i + 1 : i + len(tail)])
                    i += len(tail)
                    cpp_raw_delim = None
                    continue
            i += 1
            continue

        if in_single:
            out.append(b)
            if b == ord("\\") and i + 1 < n:
                out.append(data[i + 1])
                i += 2
                continue
            if b == ord("'"):
                in_single = False
            i += 1
            continue

        if in_double:
            out.append(b)
            if b == ord("\\") and i + 1 < n:
                out.append(data[i + 1])
                i += 2
                continue
            if b == ord('"'):
                in_double = False
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

        if b == ord("R") and i + 1 < n and data[i + 1] == ord('"'):
            j = i + 2
            while j < n and data[j] not in (ord("("), ord("\n"), ord("\r")):
                j += 1
            if j < n and data[j] == ord("("):
                delim = data[i + 2 : j]
                out.extend(data[i : j + 1])
                i = j + 1
                cpp_raw_delim = delim
                continue

        if b == ord("/") and i + 1 < n:
            nxt = data[i + 1]
            if nxt == ord("/"):
                in_line = True
                i += 2
                continue
            if nxt == ord("*"):
                in_block = True
                i += 2
                continue

        if b == ord("'"):
            in_single = True
            out.append(b)
            i += 1
            continue
        if b == ord('"'):
            in_double = True
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


def _strip_hash_comments_generic(data: bytes, *, preserve_shebang: bool) -> bytes:
    out = bytearray()
    i = 0
    n = len(data)

    if preserve_shebang and data.startswith(b"#!"):
        while i < n and data[i] != ord("\n"):
            i += 1
        if i < n and data[i] == ord("\n"):
            i += 1
        out.extend(data[:i])

    in_single = False
    in_double = False

    while i < n:
        b = data[i]

        if in_single:
            out.append(b)
            if b == ord("\\") and i + 1 < n:
                out.append(data[i + 1])
                i += 2
                continue
            if b == ord("'"):
                in_single = False
            i += 1
            continue

        if in_double:
            out.append(b)
            if b == ord("\\") and i + 1 < n:
                out.append(data[i + 1])
                i += 2
                continue
            if b == ord('"'):
                in_double = False
            i += 1
            continue

        if b == ord("#"):
            while i < n and data[i] != ord("\n"):
                i += 1
            if i < n and data[i] == ord("\n"):
                out.append(ord("\n"))
                i += 1
            continue

        if b == ord("'"):
            in_single = True
            out.append(b)
            i += 1
            continue

        if b == ord('"'):
            in_double = True
            out.append(b)
            i += 1
            continue

        out.append(b)
        i += 1

    return bytes(out)


def _strip_batch_comments(data: bytes) -> bytes:
    out = bytearray()
    for line in data.splitlines(keepends=True):
        stripped = line.lstrip(b" \t")
        if stripped.startswith(b"@"):
            stripped = stripped[1:].lstrip(b" \t")

        lower = stripped.lower()
        if lower.startswith(b"rem") and (len(lower) == 3 or lower[3] in (ord(" "), ord("\t"), ord("\r"), ord("\n"))):
            continue
        if stripped.startswith(b"::"):
            continue

        out.extend(line)
    return bytes(out)


def _strip_toml_comments(data: bytes) -> bytes:
    out = bytearray()
    i = 0
    n = len(data)

    in_basic = False
    in_literal = False
    in_ml_basic = False
    in_ml_literal = False

    while i < n:
        b = data[i]

        if in_ml_basic:
            out.append(b)
            if b == ord("\\") and i + 1 < n:
                out.append(data[i + 1])
                i += 2
                continue
            if b == ord('"') and i + 2 < n and data[i + 1] == ord('"') and data[i + 2] == ord('"'):
                out.extend(b'""')
                i += 3
                in_ml_basic = False
                continue
            i += 1
            continue

        if in_ml_literal:
            out.append(b)
            if b == ord("'") and i + 2 < n and data[i + 1] == ord("'") and data[i + 2] == ord("'"):
                out.extend(b"''")
                i += 3
                in_ml_literal = False
                continue
            i += 1
            continue

        if in_basic:
            out.append(b)
            if b == ord("\\") and i + 1 < n:
                out.append(data[i + 1])
                i += 2
                continue
            if b == ord('"'):
                in_basic = False
            i += 1
            continue

        if in_literal:
            out.append(b)
            if b == ord("'"):
                in_literal = False
            i += 1
            continue

        if b == ord("#"):
            while i < n and data[i] != ord("\n"):
                i += 1
            if i < n and data[i] == ord("\n"):
                out.append(ord("\n"))
                i += 1
            continue

        if b == ord('"'):
            if i + 2 < n and data[i + 1] == ord('"') and data[i + 2] == ord('"'):
                out.extend(b'"""')
                i += 3
                in_ml_basic = True
                continue
            in_basic = True
            out.append(b)
            i += 1
            continue

        if b == ord("'"):
            if i + 2 < n and data[i + 1] == ord("'") and data[i + 2] == ord("'"):
                out.extend(b"'''")
                i += 3
                in_ml_literal = True
                continue
            in_literal = True
            out.append(b)
            i += 1
            continue

        out.append(b)
        i += 1

    return bytes(out)


def _strip_file(path: Path, kind: str) -> tuple[bool, int]:
    original = path.read_bytes()
    if kind == "rust":
        stripped = _strip_rust_comments(original)
    elif kind == "c_like":
        stripped = _strip_c_like_comments(original)
    elif kind == "toml":
        stripped = _strip_toml_comments(original)
    elif kind == "hash":
        stripped = _strip_hash_comments_generic(original, preserve_shebang=path.name.endswith(".sh") or path.suffix == ".py")
    elif kind == "batch":
        stripped = _strip_batch_comments(original)
    else:
        raise ValueError(kind)

    if stripped == original:
        return False, 0
    path.write_bytes(stripped)
    return True, max(0, len(original) - len(stripped))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Strip lexical comments from code/config files (excluding third_party/).")
    ap.add_argument("--dry-run", action="store_true", help="Print changed files without editing.")
    ap.add_argument("--include-third-party", action="store_true", help="Also process third_party/ (not recommended).")
    args = ap.parse_args(argv)

    repo_root = Path.cwd()
    files = _run_git_ls_files(repo_root)

    exts_rust = {".rs", ".cuda"}
    exts_c_like = {
        ".c",
        ".h",
        ".cc",
        ".cpp",
        ".cxx",
        ".hpp",
        ".cu",
        ".cuh",
        ".ptx",
        ".js",
        ".mjs",
        ".cjs",
        ".ts",
        ".tsx",
        ".jsx",
        ".css",
    }
    exts_toml = {".toml"}
    exts_hash = {".py", ".sh", ".ps1", ".yml", ".yaml"}
    exts_batch = {".bat", ".cmd"}

    files_seen = 0
    files_changed = 0
    bytes_removed = 0

    for rel in files:
        if not args.include_third_party and not _is_allowed_path(rel):
            continue

        p = repo_root / rel
        if not p.is_file():
            continue

        suffix = p.suffix.lower()
        kind: Optional[str] = None
        if suffix in exts_rust:
            kind = "rust"
        elif suffix in exts_c_like:
            kind = "c_like"
        elif suffix in exts_toml:
            kind = "toml"
        elif suffix in exts_hash:
            kind = "hash"
        elif suffix in exts_batch:
            kind = "batch"
        else:
            continue

        files_seen += 1

        original = p.read_bytes()
        if kind == "rust":
            stripped = _strip_rust_comments(original)
        elif kind == "c_like":
            stripped = _strip_c_like_comments(original)
        elif kind == "toml":
            stripped = _strip_toml_comments(original)
        elif kind == "hash":
            stripped = _strip_hash_comments_generic(original, preserve_shebang=p.name.endswith(".sh") or suffix == ".py")
        elif kind == "batch":
            stripped = _strip_batch_comments(original)
        else:
            continue

        if stripped == original:
            continue

        files_changed += 1
        bytes_removed += max(0, len(original) - len(stripped))

        if args.dry_run:
            print(rel)
            continue

        p.write_bytes(stripped)

    stats = StripStats(files_seen=files_seen, files_changed=files_changed, bytes_removed=bytes_removed)
    print(f"files_seen={stats.files_seen} files_changed={stats.files_changed} bytes_removed={stats.bytes_removed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
