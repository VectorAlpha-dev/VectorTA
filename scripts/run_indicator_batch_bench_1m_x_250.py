#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchEstimate:
    id: str
    mean_ns: int
    median_ns: int
    std_dev_ns: int

    @property
    def median_ms(self) -> float:
        return self.median_ns / 1_000_000.0


DEFAULT_INDICATORS: list[str] = [
    "adx",
    "bandpass",
    "cci",
    "chandelier_exit",
    "correl_hl",
    "dm",
    "dma",
    "dx",
    "ehlers_ecema",
    "ehlers_kama",
    "ehlers_pma",
    "emv",
    "eri",
    "frama",
    "halftrend",
    "hwma",
    "kaufmanstop",
    "kst",
    "kurtosis",
    "linearreg_angle",
    "linearreg_slope",
    "linreg",
    "mwdx",
    "nama",
    "natr",
    "reflex",
    "sama",
    "srsi",
    "stochf",
    "supersmoother_3_pole",
    "tradjema",
    "trendflex",
    "trix",
    "tsi",
    "uma",
    "vama",
    "vlma",
    "wad",
    "wto",
]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_estimate(target_dir: Path, bench_id: str) -> BenchEstimate:
    est_path = target_dir / "criterion" / bench_id / "new" / "estimates.json"
    if not est_path.exists():
        raise FileNotFoundError(str(est_path))
    data = _read_json(est_path)
    return BenchEstimate(
        id=bench_id,
        mean_ns=int(data["mean"]["point_estimate"]),
        median_ns=int(data["median"]["point_estimate"]),
        std_dev_ns=int(data.get("std_dev", {}).get("point_estimate", 0)),
    )


def _bench_ids(indicator: str) -> dict[str, str]:
    stem = f"{indicator}_batch"
    return {
        "scalar": f"{stem}/{stem}_scalarbatch/1M",
        "avx2": f"{stem}/{stem}_avx2batch/1M",
        "avx512": f"{stem}/{stem}_avx512batch/1M",
    }


def _run_one_indicator(indicator: str, *, measurement_ms: int, warmup_ms: int) -> None:
    env = dict(os.environ)
    env["INDICATOR_BENCH_BATCH_SIZES"] = "1m"
    env["INDICATOR_BENCH_ONLY_BATCH"] = "1"
    env["INDICATOR_BENCH_MEASUREMENT_MS"] = str(measurement_ms)
    env["INDICATOR_BENCH_WARMUP_MS"] = str(warmup_ms)

    subprocess.run(
        [
            "cargo",
            "+nightly",
            "bench",
            "--features",
            "nightly-avx",
            "--bench",
            "indicator_benchmark",
            "--",
            f"{indicator}_batch/",
        ],
        check=True,
        env=env,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run 1M batch benchmarks (scalar/avx2/avx512) for a fixed indicator list and record Criterion estimates."
    )
    ap.add_argument(
        "--indicator",
        action="append",
        default=[],
        help="Indicator name (without _batch). Can be specified multiple times.",
    )
    ap.add_argument("--measurement-ms", type=int, default=100)
    ap.add_argument("--warmup-ms", type=int, default=1)
    ap.add_argument("--target-dir", default=os.environ.get("CARGO_TARGET_DIR", "target"))
    ap.add_argument(
        "--out-json",
        default=str(Path("benchmarks") / "indicator_batch_1m_x_250_results.json"),
    )
    ap.add_argument(
        "--out-txt",
        default=str(Path("benchmarks") / "indicator_batch_1m_x_250_results.txt"),
    )
    args = ap.parse_args()

    indicators = args.indicator if args.indicator else list(DEFAULT_INDICATORS)
    target_dir = Path(args.target_dir)

    results: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "measurement_ms": args.measurement_ms,
        "warmup_ms": args.warmup_ms,
        "batch_rows_target": 250,
        "bench_size": "1M",
        "cargo_target_dir": str(target_dir),
        "indicators": {},
    }

    for ind in indicators:
        entry: dict[str, Any] = {}
        ids = _bench_ids(ind)

        try:
            _run_one_indicator(ind, measurement_ms=args.measurement_ms, warmup_ms=args.warmup_ms)
        except subprocess.CalledProcessError as e:
            entry["error"] = f"cargo bench failed (exit {e.returncode})"
            results["indicators"][ind] = entry
            continue

        for k in ("scalar", "avx2", "avx512"):
            bench_id = ids[k]
            try:
                est = _load_estimate(target_dir, bench_id)
                entry[k] = est.__dict__ | {"median_ms": est.median_ms}
            except Exception as e:
                entry[k] = {"id": bench_id, "error": str(e)}

        results["indicators"][ind] = entry

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    lines = []
    lines.append("indicator\tscalar_ms\tavx2_ms\tavx512_ms")
    for ind in indicators:
        row = results["indicators"][ind]
        if "scalar" not in row or "avx2" not in row or "avx512" not in row:
            lines.append(f"{ind}\tERR\tERR\tERR")
            continue
        if "error" in row["scalar"] or "error" in row["avx2"] or "error" in row["avx512"]:
            lines.append(f"{ind}\tERR\tERR\tERR")
            continue
        lines.append(f"{ind}\t{row['scalar']['median_ms']:.3f}\t{row['avx2']['median_ms']:.3f}\t{row['avx512']['median_ms']:.3f}")
    out_txt = Path(args.out_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {out_json} and {out_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
