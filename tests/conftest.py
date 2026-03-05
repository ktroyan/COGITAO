from pathlib import Path

import pytest

from arcworld import DatasetConfig, Generator
from arcworld.wrapper.dataset import HDF5CogitaoStore

# ---------------------------------------------------------------------------
# Shared config fixtures — session-scoped so the Generator's HDF5 shape file
# is opened only once per process, avoiding HDF5 advisory-lock contention
# with spawned worker processes.
# ---------------------------------------------------------------------------

_BASE_CFG = DatasetConfig(
    min_n_shapes_per_grid=1,
    max_n_shapes_per_grid=3,
    min_grid_size=32,
    max_grid_size=64,
    n_examples=4,
    batch_size=2,
    shape_compulsory_conditionals=[],
    allowed_transformations=["change_shape_color"],
    min_transformation_depth=1,
    max_transformation_depth=1,
)


@pytest.fixture(scope="session")
def mock_dataset_cfg():
    """Grid-format dataset config (default)."""
    return _BASE_CFG.model_copy()


@pytest.fixture(scope="session")
def mock_dataset_cfg_image():
    """Image-format dataset config (fixed 64×64 output images)."""
    cp = _BASE_CFG.model_copy()
    cp.env_format = "image"
    cp.image_size = 64
    return cp


# ---------------------------------------------------------------------------
# Generator fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def task_generator(mock_dataset_cfg):
    """Generator that produces raw grid pairs."""
    return Generator(mock_dataset_cfg)


@pytest.fixture(scope="session")
def task_generator_image(mock_dataset_cfg_image):
    """Generator that produces CHW image pairs."""
    return Generator(mock_dataset_cfg_image)


# ---------------------------------------------------------------------------
# Path / store helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_h5_path(tmp_path):
    """Provide a temporary path for an HDF5 store file."""
    return str(tmp_path / "test_store.h5")


def generate_tasks(generator: Generator, n: int, max_tries: int = 200) -> list[dict]:
    """Generate *n* valid tasks, retrying up to *max_tries* attempts.

    Skips tasks that are empty or missing a ``pairs`` key (can happen with
    very constrained grid configs). Returns fewer than *n* items only when
    *max_tries* is exhausted.
    """
    tasks: list[dict] = []
    for _ in range(max_tries):
        if len(tasks) >= n:
            break
        task = generator.generate_single_task()
        if task and "pairs" in task:
            tasks.append(task)
    return tasks


@pytest.fixture
def dummy_dataset_path(tmp_path: Path, mock_dataset_cfg, task_generator):
    """Create a dummy HDF5 dataset for DataLoader testing.

    Returns a named tuple ``(path, num_rows)`` where ``num_rows`` is the
    total number of stored pairs (one row per pair, not per task).
    """
    import collections

    DummyDataset = collections.namedtuple("DummyDataset", ["path", "num_rows"])

    path = str(tmp_path / "dataloader_test.h5")
    store = HDF5CogitaoStore(path, cfg=mock_dataset_cfg)
    tasks = generate_tasks(task_generator, n=20)
    store.save_batch(tasks)
    num_rows = len(store)
    return DummyDataset(path=path, num_rows=num_rows)
