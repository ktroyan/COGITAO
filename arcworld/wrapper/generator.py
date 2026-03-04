import multiprocessing as mp
import time
from logging import getLogger
from pathlib import Path
from queue import Empty as QueueEmpty
from queue import Full as QueueFull

from tqdm import tqdm

from ..config import GeneratorConfig
from ..generator import Generator
from .dataset import HDF5CogitaoStore

try:
    mp.set_start_method("spawn")
except RuntimeError:
    # Already set, ignore
    pass

_logger = getLogger(__name__)


def _sample_generation_worker(worker_id, sample_queue, dataset_cfg, shutdown_event):
    """Worker process that generates samples and puts them in a shared queue.

    This worker does NOT write to any files - it only generates samples and
    puts them in the queue. The main process handles all H5 file writes.

    Args:
        worker_id: Worker ID for debugging
        sample_queue: Shared queue to put generated tasks (all workers write here)
        dataset_cfg: DatasetConfig for root Generator
        shutdown_event: Multiprocessing Event to signal worker shutdown
    """
    gen = Generator(dataset_cfg)

    failures = 0
    max_consecutive_failures = 50

    while not shutdown_event.is_set():
        try:
            # Generate full task structure (images or grids natively managed by Generator now)
            task_dict = gen.generate_single_task()

            sample_queue.put(task_dict, timeout=1)
            failures = 0  # Reset on success
        except QueueFull:
            continue  # retry until shutdown or space available
        except KeyboardInterrupt:
            continue
        except Exception as e:
            failures += 1
            if failures >= max_consecutive_failures:
                _logger.error(
                    f"Worker {worker_id} failed {max_consecutive_failures} times. Last error: {e}"
                )
                import traceback

                traceback.print_exc()
                break  # Stop this worker
            continue

    sample_queue.cancel_join_thread()
    _logger.debug(f"Worker {worker_id} exiting gracefully (failures: {failures})")


class ParallelGenerator:
    """Parallel dataset generator using multiple worker processes.

    This class manages worker processes that generate datasets in parallel
    and saves them to an HDF5 dataset file efficiently.
    """

    def __init__(self, cfg: GeneratorConfig):
        """
        Initialize the parallel cache generator.

        Args:
            cfg: GeneratorConfig instance with all generation parameters
        """
        self.cfg = cfg

    def generate(
        self,
        num_samples: int | None = None,
        *,
        buffer_size: int | None = None,
        save_batch_size: int | None = None,
    ):
        """
        Generate samples in parallel and save to HDF5 cache file.

        Workers generate samples and put them in a shared queue.
        Only the main process writes to the H5 file to avoid contention.

        Args:
            num_samples(int | None = None): Number of samples to generate. If None, uses cfg.num_tasks

        Kwargs:
            buffer_size(int | None = None): Size of the sample buffer queue. If None, uses num_workers * 16
            save_batch_size(int | None = None): Batch size for saving to disk. If None, uses num_workers * 8

        Returns:
            Path to cache file
        """
        if num_samples is None:
            num_samples = self.cfg.dataset.n_examples

        # Default batch size based on workers
        if save_batch_size is None:
            save_batch_size = self.cfg.num_workers * 8

        # Default buffer size for the queue
        if buffer_size is None:
            buffer_size = self.cfg.num_workers * 16

        path = Path(self.cfg.output_dir) / self.cfg.output_file
        path.parent.mkdir(parents=True, exist_ok=True)

        _logger.info(f"Generating {num_samples} samples to store: {path.absolute()}")

        # Initialize store (only main process will write to it)
        store = HDF5CogitaoStore(path=path, cfg=self.cfg.dataset)

        # Create shared queue for all workers to put their samples
        sample_queue = mp.Queue(maxsize=buffer_size)

        # Create shutdown event for graceful worker termination
        shutdown_event = mp.Event()

        # Start worker processes - they only generate and queue samples
        workers = []
        _logger.info(f"Starting {self.cfg.num_workers} worker processes...")

        # Create a copy of the dataset config for the internal generator making only one item at once
        dataset_cfg = self.cfg.dataset.model_copy()
        dataset_cfg.n_examples = 1  # ? Could try using batch size here instead to optimize calling the generator

        for i in range(self.cfg.num_workers):
            p = mp.Process(
                target=_sample_generation_worker,
                args=(
                    i,
                    sample_queue,
                    dataset_cfg,
                    shutdown_event,
                ),
            )
            p.start()
            workers.append(p)

        _logger.info(f"All {len(workers)} workers started and generating samples")

        # Main process: collect samples from queue and write to H5 in batches
        batch = []
        total_saved = 0

        try:
            with tqdm(total=num_samples, desc="Saving to store") as pbar:
                while total_saved < num_samples:
                    # Check if workers are still alive
                    alive_workers = sum(1 for w in workers if w.is_alive())
                    if alive_workers == 0:
                        _logger.error("All workers have died!")
                        break

                    try:
                        sample = sample_queue.get(timeout=5.0)
                    except QueueEmpty:
                        _logger.info("Queue is empty, waiting for workers...")
                        continue
                    except KeyboardInterrupt:
                        break

                    # Skip empty / failed tasks
                    if not sample or "pairs" not in sample:
                        continue

                    # Drop any task whose pairs contain None inputs or outputs —
                    # these are produced when a shape goes off-grid after a transform.
                    if any(
                        p.get("input") is None or p.get("output") is None
                        for p in sample["pairs"]
                    ):
                        _logger.warning("Dropping task with None input/output pair")
                        continue

                    batch.append(sample)

                    if len(batch) >= min(save_batch_size, num_samples - total_saved):
                        try:
                            store.save_batch(batch)
                            total_saved += len(batch)
                            pbar.update(len(batch))
                        except Exception as e:
                            _logger.error("Error while saving batch: %s", e)
                            import traceback

                            traceback.print_exc()
                        finally:
                            batch = []  # Always clear — never retry a broken batch

                    if total_saved >= num_samples:
                        break

                # Main process writes remaining samples to H5
                if batch and total_saved < num_samples:
                    to_save = batch[: num_samples - total_saved]
                    store.save_batch(to_save)
                    total_saved += len(to_save)
                    pbar.update(len(to_save))

        finally:
            _logger.info("Stopping worker processes...")
            shutdown_event.set()

            # Give workers a moment to notice the shutdown event
            time.sleep(1)

            # First pass: SIGTERM
            for w in workers:
                if w.is_alive():
                    w.terminate()

            # Wait briefly for graceful exit
            for w in workers:
                w.join(timeout=0.1)

            # Detach main-process feeder thread before touching the queue
            sample_queue.cancel_join_thread()
            sample_queue.close()

            del store

        _logger.info(
            f"Dataset generation complete! Saved {total_saved} samples to {self.cfg.output_file}"
        )

        return self.cfg.output_file
