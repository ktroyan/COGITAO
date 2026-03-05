import json
import os
from pathlib import Path
from typing import Dict, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np

from ..config import DatasetConfig


class HDF5CogitaoStore:
    """HDF5-based persistent storage for pre-generated training samples in optimized format.

    Uses a contiguous 'imgs' dataset [N, C, H, W] for fast loading.
    Supports incremental batch writing to build up the dataset.
    """

    LATEST_VERSION: int = 1

    def __init__(
        self,
        path: str | Path,
        cfg: DatasetConfig | None = None,
    ):
        """Initialize sample store.

        Args:
            path: Path to HDF5 store file
            cfg: DatasetConfig. Required to create a new store.
        """
        if isinstance(path, str):
            path = Path(path)

        self.path = path

        self._read_handle = None
        self._write_handle = None

        self._pid = None  # Track process ID to detect forks
        self._length = 0  # Track number of samples in store

        # Initialize or open existing store file
        if not self.path.exists():
            if cfg is None:
                raise ValueError("cfg is required when creating a new store file")
            self._create_h5(cfg)

        # Get dataset info from file
        handle = self._get_read_handle()
        if handle.attrs.get("version") != self.LATEST_VERSION:
            raise ValueError(
                f"File {path} uses old format. Regenerating dataset is recommended. Use older library version if necessary."
            )

        # Read back config from attr if loading existing
        stored_cfg_str = handle.attrs.get("dataset_config")
        if stored_cfg_str is None:
            raise ValueError("DatasetConfig not found in HDF5 file.")

        stored_cfg = DatasetConfig.model_validate_json(stored_cfg_str)

        if cfg is not None:
            self._validate_cfg(provided_cfg=cfg, stored_cfg=stored_cfg)

        self.cfg = stored_cfg
        self._length = handle["inputs"].shape[0]
        self.env_shape, self.env_dtype = self._get_shape_and_dtype(self.cfg)
        self._close_handle()

    def _validate_cfg(self, *, provided_cfg: DatasetConfig, stored_cfg: DatasetConfig):
        if provided_cfg.model_dump(exclude={"n_examples"}) != stored_cfg.model_dump(
            exclude={"n_examples"}
        ):
            raise ValueError(
                "Provided DatasetConfig does not match the config stored in this HDF5 file."
            )

    def _get_shape_and_dtype(self, cfg: DatasetConfig):
        env_format = cfg.env_format

        if env_format == "image":
            # For images we rely on the shape provided by the user config or default
            size = cfg.image_size
            if size is None:
                size = (cfg.max_grid_size, cfg.max_grid_size)
            if isinstance(size, int):
                size = (size, size)
            return (3, *size), "float32"
        elif env_format == "grid":
            return (cfg.max_grid_size, cfg.max_grid_size), "int8"
        else:
            raise ValueError(f"Unknown env_format: {env_format}")

    def _create_h5(self, cfg: DatasetConfig):
        """Initialize store file with resizable dataset for incremental writing."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

        chunk_size = cfg.batch_size

        shape, dtype = self._get_shape_and_dtype(cfg)

        with h5py.File(self.path, "w") as f:
            f.attrs["dataset_config"] = cfg.model_dump_json()
            f.attrs["version"] = self.LATEST_VERSION

            # Inputs / Outputs
            f.create_dataset(
                "inputs",
                shape=(0, *shape),
                maxshape=(None, *shape),
                chunks=(chunk_size, *shape),
                dtype=dtype,
                compression="lzf",
            )
            f.create_dataset(
                "outputs",
                shape=(0, *shape),
                maxshape=(None, *shape),
                chunks=(chunk_size, *shape),
                dtype=dtype,
                compression="lzf",
            )

            # Metadata
            f.create_dataset(
                "grid_sizes",
                shape=(0, 2),
                maxshape=(None, 2),
                chunks=(chunk_size, 2),
                dtype="int32",
                compression="lzf",
            )
            f.create_dataset(
                "n_shapes",
                shape=(0, 1),
                maxshape=(None, 1),
                chunks=(chunk_size, 1),
                dtype="int32",
                compression="lzf",
            )

            # Transformation Suites
            vlen_str = h5py.special_dtype(vlen=str)
            f.create_dataset(
                "transformation_suites",
                shape=(0,),
                maxshape=(None,),
                chunks=(chunk_size,),
                dtype=vlen_str,
            )

        print(f"Created dataset at {self.path}")

    def _get_read_handle(self):
        """Get persistent read handle.

        Opens a new handle if:
        - No handle exists yet
        - We're in a different process (after fork)

        Returns:
            h5py.File handle
        """
        current_pid = os.getpid()

        # Forked process protection
        if self._pid is not None and self._pid != current_pid:
            self._close_handle()

        if self._read_handle is None:
            # Close write handle first - can't have both open simultaneously
            if self._write_handle is not None:
                try:
                    self._write_handle.close()
                except Exception:
                    pass
                self._write_handle = None

            self._read_handle = h5py.File(self.path, "r")
            self._pid = current_pid

        return self._read_handle

    def _get_write_handle(self):
        """Get persistent write handle.

        Opens a new handle if:
        - No handle exists yet
        - We're in a different process (after fork)

        Returns:
            h5py.File handle opened in append mode
        """
        current_pid = os.getpid()

        # Forked process protection
        if self._pid is not None and self._pid != current_pid:
            self._close_handle()

        if self._write_handle is None:
            # Close read handle first - can't have both open simultaneously
            if self._read_handle is not None:
                try:
                    self._read_handle.close()
                except Exception:
                    pass
                self._read_handle = None

            self._write_handle = h5py.File(self.path, "a")
            self._pid = current_pid

        return self._write_handle

    def _close_handle(self):
        """Close persistent file handles."""
        # Close read handle
        if self._read_handle is not None:
            try:
                self._read_handle.close()
            except Exception:
                pass
            self._read_handle = None

        # Close write handle
        if self._write_handle is not None:
            try:
                self._write_handle.close()
            except Exception:
                pass
            self._write_handle = None

        self._pid = None

    def __del__(self):
        """Cleanup: close file handle on deletion."""
        self._close_handle()

    def __getstate__(self):
        """Prepare object for pickling - exclude file handles."""
        state = self.__dict__.copy()
        # Remove unpicklable h5py file handles
        state["_read_handle"] = None
        state["_write_handle"] = None
        state["_pid"] = None
        return state

    def __setstate__(self, state):
        """Restore object from pickle - reinitialize handles on demand."""
        self.__dict__.update(state)
        # Handles will be reopened on first access via _get_read_handle()/_get_write_handle()

    def __len__(self) -> int:
        """Get number of samples currently in store."""
        return self._length

    def __getitem__(self, idx: int) -> Optional[dict]:
        """Load a sample from store by index.

        Args:
            idx: Sample index

        Returns:
            Dictionary payload or None if not found
        """
        return self.load_batch([idx])[0]

    def __getitems__(self, idxs: list[int]) -> list[dict]:
        """Load multiple samples from store by index.

        Args:
            idxs: List of sample indices

        Returns:
            List of dictionary payloads
        """
        return self.load_batch(idxs)

    def load_batch(self, indices: list[int]) -> list[dict]:
        """Read a batch of samples using efficient fancy indexing.

        Indices MUST be valid and within bounds.

        Args:
           indices: List of indices to read

        Returns:
           List of dictionary payloads containing "inputs", "outputs", etc.
        """
        f = self._get_read_handle()
        if len(indices) == 0:
            return []

        # h5py requires sorted indices for best performance and compatibility.
        indices_arr = np.array(indices)
        unique_sorted, inverse_map = np.unique(indices_arr, return_inverse=True)

        if unique_sorted[0] < 0 or unique_sorted[-1] >= self._length:
            raise IndexError("Index range out of bounds")


        batch_inputs = f["inputs"][unique_sorted]
        batch_outputs = f["outputs"][unique_sorted]
        batch_grid_sizes = f["grid_sizes"][unique_sorted]
        batch_n_shapes = f["n_shapes"][unique_sorted]
        batch_transformation_suites = f["transformation_suites"][unique_sorted]

        def _batch_to_dict(mapped_idx: np.ndarray):
            ts_serial = batch_transformation_suites[mapped_idx]
            return {
                "inputs": batch_inputs[mapped_idx],
                "outputs": batch_outputs[mapped_idx],
                "grid_sizes": batch_grid_sizes[mapped_idx],
                "n_shapes": batch_n_shapes[mapped_idx],
                "transformation_suite": json.loads(ts_serial)
                if ts_serial and ts_serial != "null"
                else [],
            }

        results = list(map(_batch_to_dict, inverse_map))

        return results

    def save_batch(
        self, samples: list[dict], start_idx: Optional[int] = None
    ) -> list[int]:
        """Save multiple samples efficiently to the dataset.

        This method appends samples to the dataset by resizing it.
        For best performance, write larger batches.

        Args:
            samples: List of task dictionary payloads
            start_idx: Optional starting index. If None, appends to end.
                      If specified, must equal current length (no gaps allowed).

        Returns:
            List of indices where samples were saved
        """
        if not samples:
            return []

        f = self._get_write_handle()
        current_length = f["inputs"].shape[0]

        if start_idx is None:
            start_idx = current_length
        elif start_idx != current_length:
            raise ValueError(
                f"start_idx must equal current length {current_length} (no gaps allowed). "
                f"Got start_idx={start_idx}"
            )

        # Each pair in each task becomes one row — count total pairs
        total_pairs = sum(len(task_dict["pairs"]) for task_dict in samples)
        new_length = start_idx + total_pairs

        # Resize all datasets
        for dset_name in [
            "inputs",
            "outputs",
            "grid_sizes",
            "n_shapes",
            "transformation_suites",
        ]:
            f[dset_name].resize(new_length, axis=0)

        # Accumulators
        batch_inputs = []
        batch_outputs = []
        batch_grid_sizes = []
        batch_n_shapes = []
        batch_transformations = []

        env_format = self.cfg.env_format
        expected_shape = self.env_shape

        for task_dict in samples:
            ts_json = json.dumps(task_dict["transformation_suite"])

            for pair in task_dict["pairs"]:
                grid_sizes = (
                    pair["grid_sizes"] if "grid_sizes" in pair else pair["grid_size"]
                )
                batch_grid_sizes.append(np.array(grid_sizes))
                batch_n_shapes.append(np.array([pair["n_shapes"]]))

                inp = pair["input"]
                outp = pair["output"]

                # If we are grids, pad them out up to max_grid_size
                if env_format == "grid":
                    max_dim = expected_shape[-1]  # max_grid_size
                    pad_inp = np.full((max_dim, max_dim), -1, dtype=np.int8)
                    pad_outp = np.full((max_dim, max_dim), -1, dtype=np.int8)

                    h, w = inp.shape
                    pad_inp[:h, :w] = inp
                    pad_outp[:h, :w] = outp

                    batch_inputs.append(pad_inp)
                    batch_outputs.append(pad_outp)
                else:
                    batch_inputs.append(inp)
                    batch_outputs.append(outp)

                batch_transformations.append(ts_json)

        # Write batches
        f["inputs"][start_idx:new_length] = np.array(batch_inputs)
        f["outputs"][start_idx:new_length] = np.array(batch_outputs)
        f["grid_sizes"][start_idx:new_length] = np.array(batch_grid_sizes)
        f["n_shapes"][start_idx:new_length] = np.array(batch_n_shapes)
        f["transformation_suites"][start_idx:new_length] = np.array(
            batch_transformations, dtype=object
        )

        # Update internal length
        self._length = new_length

        # Flush to ensure data is written
        f.flush()

        return list(range(start_idx, new_length))

    def inspect(self):
        """Print information about the store."""
        if not self.path.exists():
            print(f"Store file not found: {self.path}")
            return

        f = self._get_read_handle()
        print(f"Store file: {self.path}")
        print(f"File size: {self.path.stat().st_size / (1024**2):.2f} MB")
        print("Format: Optimized (contiguous 'imgs' dataset)")
        print()

        # Print dataset info
        if "inputs" in f:
            dset = f["inputs"]
            print("Dataset 'inputs':")
            print(f"  Shape: {dset.shape}")
            print(f"  Dtype: {dset.dtype}")
            print(f"  Chunks: {dset.chunks}")
            print(f"  Compression: {dset.compression}")

            if self.cfg.batch_size is not None:
                print(f"  Batch size: {self.cfg.batch_size}")
            print()

            print(f"Total samples: {self._length}")
            if self._length > 0:
                print(f"Sample shape: {self.env_shape}")
                # Show value range of first sample
                first_sample = dset[0]
                print(
                    f"Sample range (first): [{first_sample.min():.3f}, {first_sample.max():.3f}]"
                )
        else:
            print("No 'inputs' dataset found!")

    def clear(self, confirm: bool = False):
        """Clear all samples from the store.

        Args:
            confirm: If True, skip confirmation prompt
        """
        if not self.path.exists():
            print(f"Store file not found: {self.path}")
            return

        if not confirm:
            response = input(f"Are you sure you want to clear {self.path}? (yes/no): ")
            if response.lower() != "yes":
                print("Cancelled.")
                return

        # Close persistent handle before deleting
        self._close_handle()

        # Delete the file
        self.path.unlink(missing_ok=True)
        self._length = 0
        print(f"Store cleared: {self.path}")

    def show_examples(self, num_examples: int = 5, output_dir: str = "."):
        """Show example samples from the store.

        Args:
            num_examples: Number of examples to show
            output_dir: Directory to save the visualization
        """
        if not self.path.exists():
            print(f"Store file not found: {self.path}")
            return

        f = self._get_read_handle()
        if "inputs" not in f:
            print("No 'inputs' dataset found in store.")
            return

        if self.cfg.env_format == "grid":
            print("Grid format not supported for show_examples.")
            return

        num_to_show = min(num_examples, self._length)

        if num_to_show == 0:
            print("No samples available to show.")
            return

        print(f"Showing {num_to_show} examples from store:")

        # Create subplots to show all images at once (inputs on top, outputs below)
        fig, axes = plt.subplots(2, num_to_show, figsize=(4 * num_to_show, 8))
        if num_to_show == 1:
            axes = axes[:, np.newaxis]

        for i in range(num_to_show):
            for row, key in enumerate(("inputs", "outputs")):
                sample = f[key][i]
                print(f"Sample {i} {key}: shape={sample.shape}, dtype={sample.dtype}")
                # Transpose from (C, H, W) to (H, W, C) for matplotlib
                if len(sample.shape) == 3 and sample.shape[0] in [1, 3, 4]:
                    sample = sample.transpose(1, 2, 0)
                axes[row, i].imshow(sample)
                axes[row, i].set_title(f"Sample {i} ({key})")
                axes[row, i].axis("off")

        plt.tight_layout()
        save_path = Path(output_dir) / f"dataset_examples_{num_to_show}.png"
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")


try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None

    class Dataset:  # Dummy class to avoid TypeError in class definition
        pass


class CogitaoDataset(Dataset):
    """PyTorch Dataset for optimized HDF5 format.

    Expects HDF5 structure:
    - imgs: [N, C, H, W] dataset with chunks=(1, C, H, W)

    Args:
        path (str | Path): Path to HDF5 store file

    Keyword Args:
        max_length (int, optional): Maximum number of samples the dataset exposes. WARNING: Setting it essentially tells dataset to ignore samples beyond that index, which implies that the data distribution can change. Use with caution.
    """

    def __init__(self, path: str | Path, *, max_length: int | None = None):
        """
        Initialize dataset from optimized store file.

        Args:
            path (str | Path): Path to HDF5 store file

        Keyword Args:
            max_length (int, optional): Maximum number of samples the dataset exposes. WARNING: Setting it essentially tells dataset to ignore samples beyond that index, which implies that the data distribution can change. Use with caution.
        """
        if torch is None:
            raise ImportError(
                "PyTorch is required to use CogitaoDataset. Please install it."
            )

        super().__init__()

        if isinstance(path, str):
            path = Path(path)

        self.path = path

        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        # Get dataset info and validate format
        self.h5df_store = HDF5CogitaoStore(path)
        if max_length is not None and (
            not isinstance(max_length, int) or max_length < 0
        ):
            raise ValueError("max_length must be a non-negative integer")

        if max_length is None:
            self._length = self.h5df_store._length
        else:
            self._length = min(self.h5df_store._length, max_length)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | list]:
        """Load a sample from dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary with parsed tensors
        """
        sample = self.h5df_store[idx]

        return {
            "inputs": torch.from_numpy(sample["inputs"]).float(),
            "outputs": torch.from_numpy(sample["outputs"]).float(),
            "grid_sizes": torch.from_numpy(sample["grid_sizes"]).long(),
            "n_shapes": torch.from_numpy(sample["n_shapes"]).long(),
            "transformation_suite": json.dumps(sample["transformation_suite"] or []),
        }

    def __getitems__(self, idxs: list[int]) -> list[Dict[str, torch.Tensor | list]]:
        """Load multiple samples from dataset efficiently.

        This method is called by DataLoader when batch_sampler is used or
        when fetching a batch of indices.

        Args:
            idxs: List of sample indices

        Returns:
            List of dictionaries with torch tensors
        """

        if len(idxs) <= 0:
            return []

        try:
            batch_arr = self.h5df_store.load_batch(idxs)

            results = []
            for sample in batch_arr:
                results.append(
                    {
                        "inputs": torch.from_numpy(sample["inputs"]).float(),
                        "outputs": torch.from_numpy(sample["outputs"]).float(),
                        "grid_sizes": torch.from_numpy(sample["grid_sizes"]).long(),
                        "n_shapes": torch.from_numpy(sample["n_shapes"]).long(),
                        "transformation_suite": json.dumps(
                            sample["transformation_suite"] or []
                        ),
                    }
                )
            return results

        except Exception as e:
            print(f"Error in batch reading: {e}")
            raise

    def __getstate__(self):
        """Prepare for pickling (DataLoader multiprocessing)."""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """Restore after unpickling."""
        self.__dict__.update(state)

    @property
    def batch_size(self) -> int:
        return self.h5df_store.cfg.batch_size

    @property
    def cfg(self) -> DatasetConfig:
        return self.h5df_store.cfg
