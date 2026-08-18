"""Merge real and pseudo-label manifests into a single combined manifest.

Usage:
    python scripts/build_mixed_manifest.py \
        --real-manifest cache/real_manifest.json \
        --pseudo-manifest cache/pseudo_manifest.json \
        --output-manifest results/mixed_manifest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge real + pseudo manifests")
    parser.add_argument("--real-manifest", type=Path, required=True)
    parser.add_argument("--pseudo-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--pseudo-sample-ratio", type=float, default=1.0,
                        help="Fraction of pseudo rows to include (1.0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.real_manifest) as f:
        real_data = json.load(f)
    with open(args.pseudo_manifest) as f:
        pseudo_data = json.load(f)

    real_rows = real_data.get("rows", [])
    pseudo_rows = pseudo_data.get("rows", [])

    for row in real_rows:
        row["source"] = "real"
    for row in pseudo_rows:
        row["source"] = "pseudo"

    if args.pseudo_sample_ratio < 1.0:
        import random
        rng = random.Random(args.seed)
        k = max(1, int(len(pseudo_rows) * args.pseudo_sample_ratio))
        pseudo_rows = rng.sample(pseudo_rows, k)

    combined = real_rows + pseudo_rows

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_manifest, "w", encoding="utf-8") as f:
        json.dump({
            "stage": "mixed_selftraining",
            "num_rows": len(combined),
            "num_real": len(real_rows),
            "num_pseudo": len(pseudo_rows),
            "rows": combined,
        }, f, indent=2, ensure_ascii=False)

    print(f"Combined manifest: {len(real_rows)} real + {len(pseudo_rows)} pseudo = {len(combined)} total")
    print(f"Saved to: {args.output_manifest}")


if __name__ == "__main__":
    main()
