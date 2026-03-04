import os

import pytest
from .conftest import generate_tasks

from arcworld.wrapper.dataset import CogitaoDataset, HDF5CogitaoStore


def test_store_creation_and_loading(temp_h5_path, mock_dataset_cfg):
    """Test creating a store and reloading it."""
    store = HDF5CogitaoStore(temp_h5_path, cfg=mock_dataset_cfg)
    assert store.cfg.batch_size == 2
    assert os.path.exists(temp_h5_path)

    # Reload without providing cfg
    del store
    store_reloaded = HDF5CogitaoStore(temp_h5_path)
    assert store_reloaded.cfg.batch_size == 2


def test_dataset_basic_access(temp_h5_path, mock_dataset_cfg, task_generator):
    """Test accessing items from the dataset.

    Each generated task may produce several pairs; the dataset length equals
    the total number of pairs (not tasks).
    """
    torch = pytest.importorskip("torch")

    store = HDF5CogitaoStore(temp_h5_path, cfg=mock_dataset_cfg)
    tasks = generate_tasks(task_generator, n=5)
    if len(tasks) < 5:
        pytest.skip("Generator could not produce enough tasks on the test grid")

    store.save_batch(tasks)
    total_pairs = sum(len(t["pairs"]) for t in tasks)

    dataset = CogitaoDataset(temp_h5_path)
    assert len(dataset) == total_pairs

    item0 = dataset[0]
    assert "inputs" in item0
    assert "outputs" in item0

    # Each row is a single padded (H, W) grid
    assert isinstance(item0["inputs"], torch.Tensor)
    assert item0["inputs"].ndim == 2
    assert item0["inputs"].shape == (
        mock_dataset_cfg.max_grid_size,
        mock_dataset_cfg.max_grid_size,
    )


def test_dataset_batch_access(temp_h5_path, mock_dataset_cfg, task_generator):
    """Test retrieving multiple items at once via __getitems__."""
    pytest.importorskip("torch")

    store = HDF5CogitaoStore(temp_h5_path, cfg=mock_dataset_cfg)
    tasks = generate_tasks(task_generator, n=10)
    if len(tasks) < 10:
        pytest.skip("Generator could not produce enough tasks on the test grid")

    store.save_batch(tasks)

    dataset = CogitaoDataset(temp_h5_path)
    items = dataset.__getitems__([0, 1, 2])
    assert len(items) == 3


def test_dataset_iteration(temp_h5_path, mock_dataset_cfg, task_generator):
    """Test iterating over the full dataset."""
    pytest.importorskip("torch")

    store = HDF5CogitaoStore(temp_h5_path, cfg=mock_dataset_cfg)
    tasks = generate_tasks(task_generator, n=4)
    if len(tasks) < 4:
        pytest.skip("Generator could not produce enough tasks on the test grid")

    store.save_batch(tasks)
    total_pairs = sum(len(t["pairs"]) for t in tasks)

    dataset = CogitaoDataset(temp_h5_path)
    count = sum(1 for item in dataset if "inputs" in item)
    assert count == total_pairs
