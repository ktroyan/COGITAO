"""Generate depth-scaling experiment datasets and record timing/success metrics.

Produces one HDF5 file per depth (1-8) and writes a JSON metrics file with:
- generation wall-clock time per depth
- number of samples successfully generated
- failure rate (retries / total attempts, estimated from timing)

Usage:
    python supplementary_material_experiments/run_depth_scaling.py \
        --num-workers 16 --n-samples 1000 --output-dir ./data/depth_scaling

    # Quick smoke test
    python supplementary_material_experiments/run_depth_scaling.py \
        --num-workers 4 --n-samples 50 --output-dir ./data/depth_scaling_test
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on the path so imports work when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supplementary_material_experiments.depth_scaling_config import (
    DEPTHS,
    depth_scaling_configs,
)
from generate_experiment_data_parallel import generate_balanced_parallel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run depth-scaling experiment generation.")
    parser.add_argument("--output-dir", type=str, default="./data/depth_scaling",
                        help="Root output directory (default: ./data/depth_scaling)")
    parser.add_argument("--num-workers", type=int, default=16,
                        help="Number of worker processes (default: 16)")
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="Number of samples per depth (default: 1000)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {}

    for entry in depth_scaling_configs:
        depth = entry.experiment
        depth_dir = output_dir / entry.subdir
        depth_dir.mkdir(parents=True, exist_ok=True)
        h5_path = depth_dir / "data.h5"
        db_name = f"depth_scaling_depth_{depth}"

        n_combos = len(entry.cfg.allowed_combinations)

        if h5_path.exists():
            logger.info(f"Skipping depth {depth} — {h5_path} already exists")
            continue

        logger.info(f"=== Depth {depth} ({n_combos} combos, {args.n_samples} samples) ===")

        start = time.time()
        n_saved = generate_balanced_parallel(
            dataset_cfg=entry.cfg,
            output_path=h5_path,
            n_tasks=args.n_samples,
            num_workers=args.num_workers,
            db_name=db_name,
            db_path=depth_dir,
        )
        elapsed = time.time() - start

        metrics[f"depth_{depth}"] = {
            "depth": depth,
            "n_combos": n_combos,
            "n_requested": args.n_samples,
            "n_generated": n_saved,
            "wall_time_seconds": round(elapsed, 2),
            "samples_per_second": round(n_saved / elapsed, 2) if elapsed > 0 else None,
        }

        logger.info(f"Depth {depth}: {n_saved}/{args.n_samples} samples in {elapsed:.1f}s "
                     f"({n_saved / elapsed:.1f} samples/s)")

    # Save metrics
    metrics_path = output_dir / "generation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
