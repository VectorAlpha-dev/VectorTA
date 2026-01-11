#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def _default_target_dir() -> str:
    env = os.environ.get("CARGO_TARGET_DIR")
    if env:
        return env
    temp = os.environ.get("TEMP") or os.environ.get("TMP") or "."
    return str(Path(temp) / "my_project_target_cuda")


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Aggregate Criterion 'cuda_bench' results into a single JSON file."
    )
    ap.add_argument("--target-dir", default=_default_target_dir())
    ap.add_argument("--match", default="cuda_batch_dev/1m_x_250/")
    ap.add_argument(
        "--out",
        default=str(Path("benchmarks") / "cuda_bench_1m_x_250_results.json"),
    )
    args = ap.parse_args()

    target_dir = Path(args.target_dir)
    criterion_dir = target_dir / "criterion"
    if not criterion_dir.exists():
        raise SystemExit(f"criterion dir not found: {criterion_dir}")

    match = args.match
    benchmarks = []

    for est_path in criterion_dir.glob("**/new/estimates.json"):
        bench_dir = est_path.parent.parent
        bench_id = bench_dir.relative_to(criterion_dir).as_posix()
        if match and match not in bench_id:
            continue

        try:
            data = _read_json(est_path)
            mean_ns = int(data["mean"]["point_estimate"])
            median_ns = int(data["median"]["point_estimate"])
            std_dev_ns = int(data.get("std_dev", {}).get("point_estimate", 0))
        except Exception as e:
            benchmarks.append(
                {
                    "id": bench_id,
                    "error": f"failed to parse {est_path.name}: {e}",
                }
            )
            continue

        benchmarks.append(
            {
                "id": bench_id,
                "mean_ns": mean_ns,
                "median_ns": median_ns,
                "std_dev_ns": std_dev_ns,
                "median_ms": median_ns / 1_000_000.0,
            }
        )

    benchmarks.sort(key=lambda b: b.get("id", ""))

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_dir": str(target_dir),
        "match": match,
        "count": len(benchmarks),
        "benchmarks": benchmarks,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} ({len(benchmarks)} benchmarks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

