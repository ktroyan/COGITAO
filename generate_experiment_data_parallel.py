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
import sqlite3
import time
from pathlib import Path
from queue import Empty as QueueEmpty
from queue import Full as QueueFull

import numpy as np
from tqdm import tqdm

from arcworld.config import DatasetConfig
from arcworld.general_utils import generate_key
from arcworld.generator import Generator
from arcworld.utils.db_utils import access_db, close_db
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

def _balanced_worker(worker_id, work_queue, result_queue, dataset_cfg, shutdown_event, done_event):
    """Worker process: pulls (combo_idx, combo) items, generates tasks, pushes results.

    Workers are pinned to a fixed subset of combos by the main process, so each worker
    only needs Generators for its assigned combos (typically 1-2). All Generators are
    built on first encounter and kept for the worker's lifetime — no eviction needed.
    """
    gen_cache: dict[tuple, Generator] = {}
    combo_failures: dict[tuple, int] = {}
    max_consecutive_failures = 50
    empty_streak = 0

    while not shutdown_event.is_set():
        if done_event.is_set():
            break

        try:
            item = work_queue.get(timeout=1)
        except QueueEmpty:
            empty_streak += 1
            if empty_streak >= 3 and done_event.is_set():
                break
            continue
        except KeyboardInterrupt:
            break

        empty_streak = 0

        if item is None:
            break

        combo_idx, combo = item
        combo_key = tuple(combo)

        if combo_key not in gen_cache:
            cfg_copy = dataset_cfg.model_copy()
            cfg_copy.allowed_combinations = [list(combo)]
            gen_cache[combo_key] = Generator(cfg_copy)

        if combo_key not in combo_failures:
            combo_failures[combo_key] = 0

        gen = gen_cache[combo_key]
        try:
            task = gen.generate_single_task()
            if task is None:
                combo_failures[combo_key] += 1
                if combo_failures[combo_key] >= max_consecutive_failures:
                    logger.error(f"Worker {worker_id}: {max_consecutive_failures} consecutive failures for combo {combo}. Dropping.")
                    continue
                work_queue.put(item)
                continue

            result_queue.put((task, combo_idx), timeout=5)
            combo_failures[combo_key] = 0
        except QueueFull:
            work_queue.put(item)
            continue
        except KeyboardInterrupt:
            break
        except Exception as e:
            combo_failures[combo_key] += 1
            if combo_failures[combo_key] >= max_consecutive_failures:
                logger.error(f"Worker {worker_id}: {max_consecutive_failures} failures for combo {combo}. Last: {e}. Dropping.")
                continue
            work_queue.put(item)
            continue

    # Cleanup: close all cached HDF5 handles
    for gen in gen_cache.values():
        if hasattr(gen, 'shape_file'):
            gen.shape_file.close()
    gen_cache.clear()
    result_queue.cancel_join_thread()


# ---------------------------------------------------------------------------
# Paired worker (shared-grid ID/OOD generation)
# ---------------------------------------------------------------------------

def _paired_worker(worker_id, work_queue, result_queue, dataset_cfg, shutdown_event, done_event):
    """Worker for paired generation: generates a grid then applies both ID and OOD transforms.

    Work items are (id_combo_idx, id_combo, ood_combo_idx, ood_combo).
    Results are (id_task, ood_task, id_combo_idx, ood_combo_idx).
    """
    import copy as _copy

    gen_cache: dict[tuple, Generator] = {}
    failures = 0
    max_consecutive_failures = 50
    empty_streak = 0

    # Build a single Generator for grid generation (uses dataset_cfg directly)
    grid_gen_key = ("__grid_gen__",)
    cfg_copy = dataset_cfg.model_copy()
    # Keep allowed_combinations from the ID side for config validity — we won't use sample_transform_suite
    gen_cache[grid_gen_key] = Generator(cfg_copy)

    while not shutdown_event.is_set():
        if done_event.is_set():
            break

        try:
            item = work_queue.get(timeout=1)
        except QueueEmpty:
            empty_streak += 1
            if empty_streak >= 3 and done_event.is_set():
                break
            continue
        except KeyboardInterrupt:
            break

        empty_streak = 0

        if item is None:
            break

        id_combo_idx, id_combo, ood_combo_idx, ood_combo = item

        grid_gen = gen_cache[grid_gen_key]

        # Generate a grid
        grid_result = grid_gen.generate_grid()
        if grid_result is None:
            failures += 1
            if failures >= max_consecutive_failures:
                logger.error(f"Worker {worker_id}: {max_consecutive_failures} consecutive failures. Dropping item.")
                continue
            try:
                work_queue.put(item)
            except QueueFull:
                pass
            continue

        input_grid, positioned_shapes = grid_result

        # Try ID transform
        id_task = grid_gen.generate_task_from_grid(input_grid, positioned_shapes, list(id_combo))
        if id_task is None:
            failures += 1
            if failures >= max_consecutive_failures:
                logger.error(f"Worker {worker_id}: {max_consecutive_failures} failures on ID combo {id_combo}. Dropping.")
                continue
            try:
                work_queue.put(item)
            except QueueFull:
                pass
            continue

        # Try OOD transform on the same grid
        ood_task = grid_gen.generate_task_from_grid(input_grid, positioned_shapes, list(ood_combo))
        if ood_task is None:
            failures += 1
            if failures >= max_consecutive_failures:
                logger.error(f"Worker {worker_id}: {max_consecutive_failures} failures on OOD combo {ood_combo}. Dropping.")
                continue
            try:
                work_queue.put(item)
            except QueueFull:
                pass
            continue

        failures = 0
        try:
            result_queue.put((id_task, ood_task, id_combo_idx, ood_combo_idx), timeout=5)
        except QueueFull:
            try:
                work_queue.put(item)
            except QueueFull:
                pass
            continue

    # Cleanup
    for gen in gen_cache.values():
        if hasattr(gen, 'shape_file'):
            gen.shape_file.close()
    gen_cache.clear()
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

    # Shared result queue (bounded to limit memory)
    result_queue = mp.Queue(maxsize=num_workers * 16)

    # Track how many work items we still need to enqueue per combo
    combo_remaining = {i: combo_targets[i] for i in range(n_combos)}
    shutdown_event = mp.Event()
    done_event = mp.Event()  # Signalled when all tasks collected — workers exit cleanly

    # Create HDF5 store with full config
    output_path.parent.mkdir(parents=True, exist_ok=True)
    store = HDF5CogitaoStore(path=output_path, cfg=dataset_cfg)

    # Open dedup DB
    db_path.mkdir(parents=True, exist_ok=True)
    cursor, conn = access_db(db_name, str(db_path))

    # Pin combos to workers: each worker gets a fixed subset of combo indices.
    # This ensures each worker only ever needs Generators for its assigned combos,
    # eliminating cache thrashing entirely.
    effective_workers = min(num_workers, n_tasks, n_combos)
    worker_queues: list[mp.Queue] = []
    combo_to_worker: dict[int, int] = {}
    for w in range(effective_workers):
        worker_queues.append(mp.Queue(maxsize=64))

    for combo_idx in range(n_combos):
        combo_to_worker[combo_idx] = combo_idx % effective_workers

    # Start workers — each with its own work queue
    workers = []
    for i in range(effective_workers):
        p = mp.Process(
            target=_balanced_worker,
            args=(i, worker_queues[i], result_queue, dataset_cfg, shutdown_event, done_event),
        )
        p.start()
        workers.append(p)

    # Main loop: collect, dedup, write
    save_batch_size = max(effective_workers * 8, 1)
    db_commit_interval = save_batch_size  # Commit DB every N successful inserts
    db_pending_commits = 0
    batch = []
    total_saved = 0
    combo_saved = {i: 0 for i in range(n_combos)}

    def feed_work_queues():
        """Feed work items into per-worker bounded queues (non-blocking, best effort)."""
        for combo_idx in range(n_combos):
            if combo_remaining[combo_idx] <= 0:
                continue
            wid = combo_to_worker[combo_idx]
            try:
                worker_queues[wid].put_nowait((combo_idx, combos[combo_idx]))
                combo_remaining[combo_idx] -= 1
            except QueueFull:
                pass  # Queue full, will retry next round

    def _try_insert_task(cursor, task_key, task_hash, ts_str):
        """Insert task into dedup DB without committing (batched commits)."""
        try:
            cursor.execute(
                "INSERT INTO tasks (task_key, task_hash, transformations) VALUES (?, ?, ?)",
                (task_key, task_hash, ts_str),
            )
            return True
        except sqlite3.IntegrityError:
            return False

    # Initial feed to get workers started
    feed_work_queues()

    try:
        with tqdm(total=n_tasks, desc=f"Generating {output_path.name}") as pbar:
            while total_saved < n_tasks:
                feed_work_queues()

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

                if not _try_insert_task(cursor, task_key, task_hash, ts_str):
                    if combo_saved[combo_idx] < combo_targets[combo_idx]:
                        combo_remaining[combo_idx] += 1
                    continue

                db_pending_commits += 1
                if db_pending_commits >= db_commit_interval:
                    conn.commit()
                    db_pending_commits = 0

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
            if db_pending_commits > 0:
                conn.commit()

    finally:
        done_event.set()
        shutdown_event.set()
        time.sleep(1)
        for w in workers:
            if w.is_alive():
                w.terminate()
        time.sleep(2)
        for w in workers:
            w.join(timeout=0)

        for wq in worker_queues:
            wq.cancel_join_thread()
            wq.close()
        result_queue.cancel_join_thread()
        result_queue.close()

        del store
        close_db(conn)

    logger.info(f"Done: {total_saved} tasks saved to {output_path}")
    for i in range(n_combos):
        logger.info(f"  combo {combos[i]}: {combo_saved[i]}/{combo_targets[i]}")

    return total_saved


# ---------------------------------------------------------------------------
# Paired generation (shared-grid ID/OOD)
# ---------------------------------------------------------------------------

def generate_paired_balanced_parallel(
    id_cfg: DatasetConfig,
    ood_cfg: DatasetConfig,
    id_output_path: Path,
    ood_output_path: Path,
    n_tasks: int,
    num_workers: int,
    db_name: str,
    db_path: Path,
):
    """Generate paired ID/OOD HDF5 datasets sharing the same input grids.

    For each sample, the same grid is used with an ID transform and an OOD
    transform. Both must succeed and pass dedup for the pair to be kept,
    ensuring strict 1-to-1 alignment between the two output files.

    Balance is maintained across transformation combos on both sides.
    """
    id_combos = id_cfg.allowed_combinations
    ood_combos = ood_cfg.allowed_combinations
    if not id_combos or not ood_combos:
        raise ValueError("Both id_cfg and ood_cfg must have non-empty allowed_combinations")

    n_id_combos = len(id_combos)
    n_ood_combos = len(ood_combos)
    n_per_id = n_tasks // n_id_combos
    n_per_ood = n_tasks // n_ood_combos
    id_remainder = n_tasks % n_id_combos
    ood_remainder = n_tasks % n_ood_combos

    id_targets = {i: n_per_id + (1 if i < id_remainder else 0) for i in range(n_id_combos)}
    ood_targets = {i: n_per_ood + (1 if i < ood_remainder else 0) for i in range(n_ood_combos)}

    logger.info(
        f"Paired generation: {n_tasks} tasks "
        f"({n_id_combos} ID combos, {n_ood_combos} OOD combos) "
        f"→ {id_output_path}, {ood_output_path}"
    )

    result_queue = mp.Queue(maxsize=num_workers * 16)
    shutdown_event = mp.Event()
    done_event = mp.Event()

    # Create HDF5 stores
    id_output_path.parent.mkdir(parents=True, exist_ok=True)
    ood_output_path.parent.mkdir(parents=True, exist_ok=True)
    id_store = HDF5CogitaoStore(path=id_output_path, cfg=id_cfg)
    ood_store = HDF5CogitaoStore(path=ood_output_path, cfg=ood_cfg)

    # Open dedup DB
    db_path.mkdir(parents=True, exist_ok=True)
    cursor, conn = access_db(db_name, str(db_path))

    # Create per-worker queues
    effective_workers = min(num_workers, n_tasks)
    worker_queues: list[mp.Queue] = [mp.Queue(maxsize=64) for _ in range(effective_workers)]

    # Start workers — all use the ID config for grid generation
    workers = []
    for i in range(effective_workers):
        p = mp.Process(
            target=_paired_worker,
            args=(i, worker_queues[i], result_queue, id_cfg, shutdown_event, done_event),
        )
        p.start()
        workers.append(p)

    # Main loop: collect, dedup, write
    save_batch_size = max(effective_workers * 8, 1)
    db_commit_interval = save_batch_size
    db_pending_commits = 0
    id_batch = []
    ood_batch = []
    total_saved = 0
    id_saved = {i: 0 for i in range(n_id_combos)}
    ood_saved = {i: 0 for i in range(n_ood_combos)}

    # Track remaining work to enqueue per combo pair
    id_remaining = {i: id_targets[i] for i in range(n_id_combos)}
    ood_remaining = {i: ood_targets[i] for i in range(n_ood_combos)}

    def _next_needed_combo(remaining_dict):
        """Return the combo index with the most remaining quota, or None if all done."""
        best = None
        best_rem = 0
        for idx, rem in remaining_dict.items():
            if rem > best_rem:
                best = idx
                best_rem = rem
        return best

    def feed_work_queues():
        """Feed paired work items into worker queues (round-robin across workers)."""
        id_idx = _next_needed_combo(id_remaining)
        ood_idx = _next_needed_combo(ood_remaining)
        if id_idx is None or ood_idx is None:
            return
        for wid in range(effective_workers):
            # Recompute each iteration since remaining changes
            id_idx = _next_needed_combo(id_remaining)
            ood_idx = _next_needed_combo(ood_remaining)
            if id_idx is None or ood_idx is None:
                return
            try:
                worker_queues[wid].put_nowait(
                    (id_idx, id_combos[id_idx], ood_idx, ood_combos[ood_idx])
                )
                id_remaining[id_idx] -= 1
                ood_remaining[ood_idx] -= 1
            except QueueFull:
                pass

    def _try_insert_task(cursor, task_key, task_hash, ts_str):
        try:
            cursor.execute(
                "INSERT INTO tasks (task_key, task_hash, transformations) VALUES (?, ?, ?)",
                (task_key, task_hash, ts_str),
            )
            return True
        except sqlite3.IntegrityError:
            return False

    # Initial feed
    feed_work_queues()

    try:
        with tqdm(total=n_tasks, desc=f"Paired gen {id_output_path.stem}+{ood_output_path.stem}") as pbar:
            while total_saved < n_tasks:
                feed_work_queues()

                alive = sum(1 for w in workers if w.is_alive())
                if alive == 0 and result_queue.empty():
                    logger.error("All workers died and queue is empty.")
                    break

                try:
                    id_task, ood_task, id_combo_idx, ood_combo_idx = result_queue.get(timeout=10)
                except QueueEmpty:
                    continue
                except KeyboardInterrupt:
                    break

                # Dedup both tasks — if either is a duplicate, discard the pair
                id_hash = hash_task_raw(id_task)
                id_key = generate_key()
                id_ts_str = str(id_task["transformation_suite"])

                ood_hash = hash_task_raw(ood_task)
                ood_key = generate_key()
                ood_ts_str = str(ood_task["transformation_suite"])

                if not _try_insert_task(cursor, id_key, id_hash, id_ts_str):
                    # Re-enqueue both combo slots
                    if id_saved[id_combo_idx] < id_targets[id_combo_idx]:
                        id_remaining[id_combo_idx] += 1
                    if ood_saved[ood_combo_idx] < ood_targets[ood_combo_idx]:
                        ood_remaining[ood_combo_idx] += 1
                    continue

                if not _try_insert_task(cursor, ood_key, ood_hash, ood_ts_str):
                    if id_saved[id_combo_idx] < id_targets[id_combo_idx]:
                        id_remaining[id_combo_idx] += 1
                    if ood_saved[ood_combo_idx] < ood_targets[ood_combo_idx]:
                        ood_remaining[ood_combo_idx] += 1
                    continue

                db_pending_commits += 1
                if db_pending_commits >= db_commit_interval:
                    conn.commit()
                    db_pending_commits = 0

                id_batch.append(id_task)
                ood_batch.append(ood_task)
                id_saved[id_combo_idx] += 1
                ood_saved[ood_combo_idx] += 1
                total_saved += 1
                pbar.update(1)

                if len(id_batch) >= save_batch_size:
                    id_store.save_batch(id_batch)
                    ood_store.save_batch(ood_batch)
                    id_batch = []
                    ood_batch = []

            # Flush remaining
            if id_batch:
                id_store.save_batch(id_batch)
                ood_store.save_batch(ood_batch)
                id_batch = []
                ood_batch = []
            if db_pending_commits > 0:
                conn.commit()

    finally:
        done_event.set()
        shutdown_event.set()
        time.sleep(1)
        for w in workers:
            if w.is_alive():
                w.terminate()
        time.sleep(2)
        for w in workers:
            w.join(timeout=0)

        for wq in worker_queues:
            wq.cancel_join_thread()
            wq.close()
        result_queue.cancel_join_thread()
        result_queue.close()

        del id_store
        del ood_store
        close_db(conn)

    logger.info(f"Done: {total_saved} paired tasks saved")
    for i in range(n_id_combos):
        logger.info(f"  ID combo {id_combos[i]}: {id_saved[i]}/{id_targets[i]}")
    for i in range(n_ood_combos):
        logger.info(f"  OOD combo {ood_combos[i]}: {ood_saved[i]}/{ood_targets[i]}")

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

def _find_ood_entry(entries: list[ExperimentEntry], train_entry: ExperimentEntry) -> ExperimentEntry | None:
    """Find the matching OOD (split='test') entry for a given train entry."""
    for e in entries:
        if (e.split == "test"
                and e.setting == train_entry.setting
                and e.experiment == train_entry.experiment
                and e.subdir == train_entry.subdir):
            return e
    return None


def run_study(
    study_name: str,
    entries: list[ExperimentEntry],
    output_dir: str,
    num_workers: int,
    n_train: int,
    n_val: int,
    n_test: int,
):
    """Run generation for all entries in a study, expanding splits.

    When a train entry has paired_splits=True, val/test ID and OOD splits
    are generated with shared input grids (paired mode). The matching OOD
    entry (same setting/experiment, split='test') is looked up automatically
    and does not need to be processed separately.
    """
    time_log: dict[str, float] = {}

    # Track which test entries are consumed by paired generation so we don't
    # process them independently as well.
    paired_ood_keys: set[tuple] = set()

    for entry in entries:
        if entry.split == "train" and entry.paired_splits:
            ood_entry = _find_ood_entry(entries, entry)
            if ood_entry is None:
                logger.error(
                    f"paired_splits=True but no matching test entry for "
                    f"setting={entry.setting} exp={entry.experiment}. "
                    f"Falling back to independent generation."
                )
                # Fall through to normal train handling below
            else:
                paired_ood_keys.add((ood_entry.setting, ood_entry.experiment, ood_entry.subdir))

                db_name = build_db_name(study_name, entry)
                db_path = build_output_path(output_dir, study_name, entry, "db").parent

                # 1. Generate train independently (as before)
                train_path = build_output_path(output_dir, study_name, entry, "train")
                if train_path.exists():
                    logger.info(f"Skipping {train_path} (already exists)")
                else:
                    start = time.time()
                    generate_balanced_parallel(entry.cfg, train_path, n_train, num_workers, db_name, db_path)
                    time_log[str(train_path)] = time.time() - start

                # 2. Generate val + val_ood paired
                val_id_path = build_output_path(output_dir, study_name, entry, "val")
                val_ood_path = build_output_path(output_dir, study_name, entry, "val_ood")
                if val_id_path.exists() and val_ood_path.exists():
                    logger.info(f"Skipping paired val (both exist)")
                else:
                    start = time.time()
                    generate_paired_balanced_parallel(
                        entry.cfg, ood_entry.cfg,
                        val_id_path, val_ood_path,
                        n_val, num_workers, db_name, db_path,
                    )
                    time_log[f"{val_id_path}+{val_ood_path}"] = time.time() - start

                # 3. Generate test + test_ood paired
                test_id_path = build_output_path(output_dir, study_name, entry, "test")
                test_ood_path = build_output_path(output_dir, study_name, entry, "test_ood")
                if test_id_path.exists() and test_ood_path.exists():
                    logger.info(f"Skipping paired test (both exist)")
                else:
                    start = time.time()
                    generate_paired_balanced_parallel(
                        entry.cfg, ood_entry.cfg,
                        test_id_path, test_ood_path,
                        n_test, num_workers, db_name, db_path,
                    )
                    time_log[f"{test_id_path}+{test_ood_path}"] = time.time() - start

                continue  # Skip normal train handling

    # Second pass: process entries not handled by paired generation
    for entry in entries:
        db_name = build_db_name(study_name, entry)
        db_path = build_output_path(output_dir, study_name, entry, "db").parent

        if entry.split == "train" and not entry.paired_splits:
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
            # Skip if already consumed by paired generation
            key = (entry.setting, entry.experiment, entry.subdir)
            if key in paired_ood_keys:
                logger.info(
                    f"Skipping test entry setting={entry.setting} exp={entry.experiment} "
                    f"(handled by paired generation)"
                )
                continue

            # Generate val_ood, test_ood independently
            for split_name, n in [("val_ood", n_val), ("test_ood", n_test)]:
                path = build_output_path(output_dir, study_name, entry, split_name)
                if path.exists():
                    logger.info(f"Skipping {path} (already exists)")
                    continue
                start = time.time()
                generate_balanced_parallel(entry.cfg, path, n, num_workers, db_name, db_path)
                time_log[str(path)] = time.time() - start

        elif entry.split != "train":
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
    from experiment_configs.compgen_ktroyan import compgen_ktroyan_experiments
    from experiment_configs.compgen_basics_ktroyan import compgen_basics_ktroyan_experiments

    STUDIES["c0"] = c0_configs
    STUDIES["compositionality"] = compositionality_configs
    STUDIES["generalization"] = generalization_configs
    STUDIES["sample_efficiency"] = sample_efficiency_configs
    STUDIES["compositionality_gridsize"] = compositionality_gridsize_config
    STUDIES["c4"] = c4_configs
    STUDIES["compgen_ktroyan_experiments"] = compgen_ktroyan_experiments
    STUDIES["compgen_basics_ktroyan_experiments"] = compgen_basics_ktroyan_experiments


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate balanced experiment datasets in parallel.")
    parser.add_argument("--study", type=str, required=True,
                        help="Study name (c0, compositionality, generalization, sample_efficiency, compositionality_gridsize, c4, compgen_ktroyan_experiments, compgen_basics_ktroyan_experiments) or 'all'.")
    parser.add_argument("--study_name_suffix", type=str, default="",
                        help="Optional suffix to append to study name right after the study config was loaded. This allows to use only one config file for several generations.")
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
            study_name=study_name+args.study_name_suffix,   # NOTE: I updated this
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
