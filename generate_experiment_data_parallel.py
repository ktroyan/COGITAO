"""Parallel balanced experiment data generation.

Generates HDF5 datasets with exact balance across transformation combinations,
deduplication via SQLite, and multiprocessing for speed.

Usage:
    python generate_experiment_data_parallel.py --study c0 --num-workers 16 --output-dir ./data
    python generate_experiment_data_parallel.py --study compositionality --n-train 1000000
    python generate_experiment_data_parallel.py --study all --num-workers 64
"""

import argparse
import hashlib
import logging
import multiprocessing as mp
import time
from pathlib import Path
from queue import Empty as QueueEmpty
from queue import Full as QueueFull

import numpy as np
from tqdm import tqdm

from arcworld.config import DatasetConfig
from arcworld.general_utils import generate_key
from arcworld.generator import Generator
from arcworld.utils.db_utils import access_db, close_db, store_task_in_db
from arcworld.wrapper.dataset import HDF5CogitaoStore
from experiment_configs.entry import ExperimentEntry

try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _balanced_worker(worker_id, work_queue, result_queue, dataset_cfg, shutdown_event):
    """Worker process: pulls (combo_idx, combo) items, generates tasks, pushes results.

    Caches Generator instances per combo to avoid re-loading the shape HDF5 file.
    """
    gen_cache: dict[tuple, Generator] = {}
    failures = 0
    max_consecutive_failures = 50

    while not shutdown_event.is_set():
        try:
            item = work_queue.get(timeout=1)
        except QueueEmpty:
            continue
        except KeyboardInterrupt:
            break

        if item is None:
            break  # Sentinel — exit

        combo_idx, combo = item
        combo_key = tuple(combo)

        if combo_key not in gen_cache:
            cfg_copy = dataset_cfg.model_copy()
            cfg_copy.allowed_combinations = [list(combo)]
            gen_cache[combo_key] = Generator(cfg_copy)

        gen = gen_cache[combo_key]
        try:
            task = gen.generate_single_task()
            if task is None:
                failures += 1
                if failures >= max_consecutive_failures:
                    logger.error(f"Worker {worker_id}: {max_consecutive_failures} consecutive failures for combo {combo}. Stopping.")
                    break
                # Re-enqueue so another attempt can be made
                work_queue.put(item)
                continue

            result_queue.put((task, combo_idx), timeout=5)
            failures = 0
        except QueueFull:
            # Put work item back so it isn't lost
            work_queue.put(item)
            continue
        except KeyboardInterrupt:
            break
        except Exception as e:
            failures += 1
            if failures >= max_consecutive_failures:
                logger.error(f"Worker {worker_id} hit {max_consecutive_failures} failures. Last: {e}")
                break
            work_queue.put(item)
            continue

    result_queue.cancel_join_thread()


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def hash_task_raw(task: dict) -> str:
    """Compute SHA256 dedup hash from a raw Generator task dict."""
    input_grid = task["pairs"][0]["input"]
    grid_bytes = np.asarray(input_grid, dtype=np.uint8).tobytes()
    ts_str = "|".join(task["transformation_suite"])
    return hashlib.sha256(grid_bytes + ts_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate_balanced_parallel(
    dataset_cfg: DatasetConfig,
    output_path: Path,
    n_tasks: int,
    num_workers: int,
    db_name: str,
    db_path: Path,
):
    """Generate a balanced HDF5 dataset with deduplication.

    Args:
        dataset_cfg: Full DatasetConfig (all combinations included).
        output_path: Path to the output .h5 file.
        n_tasks: Total number of tasks to generate.
        num_workers: Number of worker processes.
        db_name: SQLite database name (shared across splits for cross-split dedup).
        db_path: Directory for the SQLite database.
    """
    combos = dataset_cfg.allowed_combinations
    if combos is None or len(combos) == 0:
        raise ValueError("dataset_cfg.allowed_combinations must be a non-empty list")

    n_combos = len(combos)
    n_per_combo = n_tasks // n_combos
    remainder = n_tasks % n_combos

    # Targets per combo (first `remainder` combos get +1)
    combo_targets = {i: n_per_combo + (1 if i < remainder else 0) for i in range(n_combos)}

    logger.info(f"Generating {n_tasks} tasks ({n_combos} combos, ~{n_per_combo} each) → {output_path}")

    # Enqueue work items
    work_queue = mp.Queue()
    for combo_idx, combo in enumerate(combos):
        for _ in range(combo_targets[combo_idx]):
            work_queue.put((combo_idx, combo))

    # Sentinel values to stop workers
    for _ in range(num_workers):
        work_queue.put(None)

    result_queue = mp.Queue(maxsize=num_workers * 16)
    shutdown_event = mp.Event()

    # Create HDF5 store with full config
    output_path.parent.mkdir(parents=True, exist_ok=True)
    store = HDF5CogitaoStore(path=output_path, cfg=dataset_cfg)

    # Open dedup DB
    db_path.mkdir(parents=True, exist_ok=True)
    cursor, conn = access_db(db_name, str(db_path))

    # Start workers
    workers = []
    effective_workers = min(num_workers, n_tasks)
    for i in range(effective_workers):
        p = mp.Process(
            target=_balanced_worker,
            args=(i, work_queue, result_queue, dataset_cfg, shutdown_event),
        )
        p.start()
        workers.append(p)

    # Main loop: collect, dedup, write
    save_batch_size = max(effective_workers * 8, 1)
    batch = []
    total_saved = 0
    combo_saved = {i: 0 for i in range(n_combos)}

    try:
        with tqdm(total=n_tasks, desc=f"Generating {output_path.name}") as pbar:
            while total_saved < n_tasks:
                alive = sum(1 for w in workers if w.is_alive())
                if alive == 0 and result_queue.empty():
                    logger.error("All workers died and queue is empty.")
                    break

                try:
                    task, combo_idx = result_queue.get(timeout=10)
                except QueueEmpty:
                    continue
                except KeyboardInterrupt:
                    break

                # Dedup
                task_hash = hash_task_raw(task)
                task_key = generate_key()
                ts_str = str(task["transformation_suite"])

                if not store_task_in_db(cursor, conn, task_key, task_hash, ts_str):
                    # Duplicate — re-enqueue work for this combo
                    if combo_saved[combo_idx] < combo_targets[combo_idx]:
                        work_queue.put((combo_idx, combos[combo_idx]))
                    continue

                batch.append(task)
                combo_saved[combo_idx] += 1
                total_saved += 1
                pbar.update(1)

                if len(batch) >= save_batch_size:
                    store.save_batch(batch)
                    batch = []

            # Flush remaining
            if batch:
                store.save_batch(batch)
                batch = []

    finally:
        shutdown_event.set()
        time.sleep(1)
        for w in workers:
            if w.is_alive():
                w.terminate()
        time.sleep(2)
        for w in workers:
            w.join(timeout=0)

        work_queue.cancel_join_thread()
        work_queue.close()
        result_queue.cancel_join_thread()
        result_queue.close()

        del store
        close_db(conn)

    logger.info(f"Done: {total_saved} tasks saved to {output_path}")
    for i in range(n_combos):
        logger.info(f"  combo {combos[i]}: {combo_saved[i]}/{combo_targets[i]}")

    return total_saved


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def build_output_path(output_dir: str, study_name: str, entry: ExperimentEntry, split_name: str) -> Path:
    """Build the output file path for a given entry and split."""
    parts = [output_dir, study_name, f"exp_setting_{entry.setting}", f"experiment_{entry.experiment}"]
    if entry.subdir:
        parts.append(entry.subdir)
    parts.append(f"{split_name}.h5")
    return Path(*parts)


def build_db_name(study_name: str, entry: ExperimentEntry) -> str:
    """Build a unique DB name scoped to the experiment (shared across splits)."""
    base = f"{study_name}_setting_{entry.setting}_exp_{entry.experiment}"
    if entry.subdir:
        base += f"_{entry.subdir}"
    return base


# ---------------------------------------------------------------------------
# Split expansion
# ---------------------------------------------------------------------------

def run_study(
    study_name: str,
    entries: list[ExperimentEntry],
    output_dir: str,
    num_workers: int,
    n_train: int,
    n_val: int,
    n_test: int,
):
    """Run generation for all entries in a study, expanding splits."""
    time_log: dict[str, float] = {}

    for entry in entries:
        db_name = build_db_name(study_name, entry)
        db_path = build_output_path(output_dir, study_name, entry, "db").parent

        if entry.split == "train":
            # Generate train, val, test (in-distribution) — all sharing the same dedup DB
            for split_name, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
                path = build_output_path(output_dir, study_name, entry, split_name)
                if path.exists():
                    logger.info(f"Skipping {path} (already exists)")
                    continue
                start = time.time()
                generate_balanced_parallel(entry.cfg, path, n, num_workers, db_name, db_path)
                time_log[str(path)] = time.time() - start

        elif entry.split == "test":
            # Generate val_ood, test_ood
            for split_name, n in [("val_ood", n_val), ("test_ood", n_test)]:
                path = build_output_path(output_dir, study_name, entry, split_name)
                if path.exists():
                    logger.info(f"Skipping {path} (already exists)")
                    continue
                start = time.time()
                generate_balanced_parallel(entry.cfg, path, n, num_workers, db_name, db_path)
                time_log[str(path)] = time.time() - start

        else:
            logger.warning(f"Unknown split role '{entry.split}' for setting={entry.setting} exp={entry.experiment}")

    # Log timing
    if time_log:
        logger.info("Generation times:")
        for path, t in time_log.items():
            logger.info(f"  {path}: {t:.1f}s")


# ---------------------------------------------------------------------------
# Study registry
# ---------------------------------------------------------------------------

STUDIES: dict[str, list[ExperimentEntry]] = {}


def _load_studies():
    """Lazily import all experiment config modules."""
    if STUDIES:
        return

    from experiment_configs.c0 import c0_configs
    from experiment_configs.c4 import c4_configs
    from experiment_configs.compositionality import compositionality_configs
    from experiment_configs.compositionality_gridsize import compositionality_gridsize_config
    from experiment_configs.generalization import generalization_configs
    from experiment_configs.sample_efficiency import sample_efficiency_configs
    from experiment_configs.test_config_for_klim import test_config_for_klim

    STUDIES["c0"] = c0_configs
    STUDIES["compositionality"] = compositionality_configs
    STUDIES["generalization"] = generalization_configs
    STUDIES["sample_efficiency"] = sample_efficiency_configs
    STUDIES["compositionality_gridsize"] = compositionality_gridsize_config
    STUDIES["c4"] = c4_configs
    STUDIES["test_config_for_klim"] = test_config_for_klim


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate balanced experiment datasets in parallel.")
    parser.add_argument("--study", type=str, required=True,
                        help="Study name (c0, compositionality, generalization, sample_efficiency, compositionality_gridsize, c4) or 'all'.")
    parser.add_argument("--output-dir", type=str, default="./data",
                        help="Root output directory (default: ./data)")
    parser.add_argument("--num-workers", type=int, default=16,
                        help="Number of worker processes (default: 16)")
    parser.add_argument("--n-train", type=int, default=1_000_000,
                        help="Number of training samples per config (default: 1000000)")
    parser.add_argument("--n-val", type=int, default=1_000,
                        help="Number of validation samples per config (default: 1000)")
    parser.add_argument("--n-test", type=int, default=1_000,
                        help="Number of test samples per config (default: 1000)")
    args = parser.parse_args()

    _load_studies()

    if args.study == "all":
        studies_to_run = list(STUDIES.keys())
    else:
        if args.study not in STUDIES:
            parser.error(f"Unknown study '{args.study}'. Choose from: {', '.join(STUDIES.keys())} or 'all'.")
        studies_to_run = [args.study]

    for study_name in studies_to_run:
        logger.info(f"=== Running study: {study_name} ===")
        run_study(
            study_name=study_name,
            entries=STUDIES[study_name],
            output_dir=args.output_dir,
            num_workers=args.num_workers,
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
        )

    logger.info("All done.")


if __name__ == "__main__":
    main()
