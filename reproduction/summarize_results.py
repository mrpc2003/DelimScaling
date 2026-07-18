#!/usr/bin/env python3
"""Emit a compact, machine-readable-in-the-log MuirBench evaluation summary."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--scaling", required=True)
    parser.add_argument("--scale", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--attention", required=True)
    parser.add_argument("--limit", default="")
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    metrics = payload.get("results", {}).get("muirbench", {})
    matches = {
        key: value
        for key, value in metrics.items()
        if key.startswith("muirbench_score_overall,") and "_stderr," not in key
    }
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one MuirBench aggregate metric, found: {matches}")
    metric_name, score = next(iter(matches.items()))
    elapsed = payload.get("total_evaluation_time_seconds", "unknown")

    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        gpu = f"unavailable ({exc})"

    lines = [
        "# MuirBench reproduction result",
        "",
        f"- Variant: {args.variant}",
        f"- MuirBench aggregate ({metric_name}): {float(score) * 100:.2f}% ({float(score):.6f})",
        f"- Delimiter scaling: {args.scaling}; scale={args.scale}; selected layers={args.layers}",
        f"- Attention implementation: {args.attention}",
        "- Evaluation: "
        + (
            f"smoke limit {args.limit} from `MUIRBENCH/MUIRBENCH` test split"
            if args.limit
            else "full `MUIRBENCH/MUIRBENCH` test split"
        )
        + "; Qwen/Qwen2.5-VL-3B-Instruct; batch size 1/GPU; seed 1234.",
        f"- Elapsed evaluator time: {elapsed} s",
        f"- Python: {platform.python_version()}",
        "- GPUs: " + gpu.replace("\n", "; "),
        f"- Raw aggregate JSON: {args.results}",
    ]
    Path("EVAL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"REPRO_RESULT variant={args.variant} muirbench_percent={float(score) * 100:.6f} elapsed_seconds={elapsed}")


if __name__ == "__main__":
    main()
