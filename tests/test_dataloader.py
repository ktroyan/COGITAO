import pytest

from arcworld.wrapper.dataset import CogitaoDataset

pytest.importorskip("torch")
from torch.utils.data import DataLoader


def test_multiworker_loading(dummy_dataset_path, mock_dataset_cfg):
    """Test DataLoader with multiple workers.

    ``dummy_dataset_path`` generates 100 tasks; the actual row count equals
    the total number of pairs across those tasks (one row per pair).
    """
    dataset = CogitaoDataset(dummy_dataset_path.path)
    assert len(dataset) == dummy_dataset_path.num_rows

    max_size = mock_dataset_cfg.max_grid_size

    for num_workers in [0, 2]:
        loader = DataLoader(
            dataset,
            batch_size=4,
            num_workers=num_workers,
            shuffle=True,
            persistent_workers=num_workers > 0,
        )

        item_count = 0
        for batch in loader:
            imgs = batch["inputs"]
            # Each row is a single padded (H, W) grid
            assert imgs.shape[1:] == (max_size, max_size)
            item_count += imgs.shape[0]

        assert item_count == dummy_dataset_path.num_rows


def test_dataloader_shuffle(dummy_dataset_path):
    """Verify shuffle functionality works (order differs between passes)."""
    dataset = CogitaoDataset(dummy_dataset_path.path)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # Collect means from the first 5 batches across two passes
    run1 = [batch["inputs"].mean().item() for i, batch in enumerate(loader) if i < 5]
    run2 = [batch["inputs"].mean().item() for i, batch in enumerate(loader) if i < 5]

    # Statistically unlikely to be identical if shuffle is working
    assert run1 != run2
