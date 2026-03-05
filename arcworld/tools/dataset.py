import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from arcworld.constants import COLORMAP, NORM

from ..wrapper.dataset import CogitaoDataset

_PAD_COLOR = "#1a1a2e"  # dark navy shown outside the active grid
_PAD_RGBA = np.array(mcolors.to_rgba(_PAD_COLOR), dtype=np.float32)


def _render_grid(ax, grid: np.ndarray, grid_size: tuple, title: str = ""):
    """Render the full padded grid: active cells in project colours, padding in dark navy."""
    h, w = int(grid_size[0]), int(grid_size[1])
    grid = grid.astype(np.int16)

    # Build RGBA for every cell
    rgba = np.empty((*grid.shape, 4), dtype=np.float32)
    rgba[:] = _PAD_RGBA  # start with padding colour everywhere
    for v in range(10):
        mask = grid == v
        if mask.any():
            rgba[mask] = COLORMAP(NORM(v))

    ax.imshow(rgba, interpolation="nearest", aspect="equal")
    ax.set_title(title, fontsize=8, pad=3)
    ax.axis("off")

    # White border around the active (non-padded) area
    rect = mpatches.FancyBboxPatch(
        (-0.5, -0.5),
        w,
        h,
        boxstyle="square,pad=0",
        linewidth=1.2,
        edgecolor="white",
        facecolor="none",
    )
    ax.add_patch(rect)


def _render_image(ax, img: np.ndarray, title: str = ""):
    """Render a CHW float32 image."""
    hwc = np.clip(np.transpose(img, (1, 2, 0)), 0.0, 1.0)
    ax.imshow(hwc, interpolation="nearest", aspect="equal")
    ax.set_title(title, fontsize=8, pad=3)
    ax.axis("off")


def plot_sequence_examples(
    dataset: str | Path | CogitaoDataset,
    n_samples: int = 3,
    output_path: str | Path | None = None,
):
    """
    Plot the full transformation sequence for ``n_samples`` samples from an
    HDF5 store.

    Each figure shows one sample as a row of frames::

        input  --[action_0]-->  step 1  --[action_1]-->  ...  --[action_N-1]-->  output

    Args:
        dataset:      Path to an ``.h5`` file or an :class:`HDF5CogitaoStore` instance.
        n_samples:    Number of samples to plot.
        output_path:  If given, figures are saved as
                      ``<stem>_s<idx><suffix>`` files instead of displayed.
                      The parent directory is created automatically.
    """
    if isinstance(dataset, (str, Path)):
        dataset = CogitaoDataset(dataset)
    elif isinstance(dataset, CogitaoDataset):
        dataset = dataset
    else:
        raise TypeError("dataset must be a path or CogitaoDataset instance")

    n_samples = min(n_samples, len(dataset))
    if n_samples == 0:
        print(f"Store is empty: {dataset.path}")
        return

    env_format = dataset.cfg.env_format
    print(f"Store: {dataset.path}  |  format: {env_format}  |  samples: {len(dataset)}")

    samples = dataset.__getitems__(list(range(n_samples)))

    for sample_idx, sample in enumerate(samples):
        seq_len = int(sample["seq_len"])
        sequences = sample["full_grid_sequence"].numpy()  # (max_seq_len, *env_shape)
        ts = json.loads(sample["transformation_suite"])  # list[str]
        grid_size = sample["grid_sizes"].numpy()  # (H, W)

        # Interleave frame cols and narrow arrow cols
        n_frames = seq_len
        n_cols = n_frames * 2 - 1
        width_ratios = [1.0 if i % 2 == 0 else 0.22 for i in range(n_cols)]

        fig, axes = plt.subplots(
            1,
            n_cols,
            figsize=(max(4, 2.2 * n_frames), 2.8),
            gridspec_kw={"width_ratios": width_ratios},
        )
        if n_cols == 1:
            axes = [axes]

        suite_str = " → ".join(ts) if ts else "(no transformation)"
        fig.suptitle(
            f"Sample {sample_idx}  |  suite: [{suite_str}]",
            fontsize=9,
            y=1.02,
        )

        frame_col = 0
        for k in range(n_frames):
            frame = sequences[k]
            if k == 0:
                label = "input"
            elif k == n_frames - 1:
                label = "output"
            else:
                label = f"step {k}"

            ax_frame = axes[frame_col]
            if env_format == "grid":
                _render_grid(ax_frame, frame, grid_size, title=label)
            else:
                _render_image(ax_frame, frame, title=label)

            if k < n_frames - 1:
                ax_arrow = axes[frame_col + 1]
                ax_arrow.axis("off")
                ax_arrow.set_facecolor("#1c1c1c")
                action_name = ts[k] if k < len(ts) else "?"
                ax_arrow.text(
                    0.5,
                    0.5,
                    f"→\n{action_name}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    transform=ax_arrow.transAxes,
                    color="white",
                )

            frame_col += 2

        fig.patch.set_facecolor("#1c1c1c")
        for ax in axes:
            ax.set_facecolor("#1c1c1c")

        plt.tight_layout()

        if output_path is not None:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            save_path = p.parent / f"{p.stem}_s{sample_idx}{p.suffix or '.png'}"
            plt.savefig(
                save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()
            )
            print(f"  Saved → {save_path}")
        else:
            plt.show()

        plt.close(fig)


def color_analysis(
    dataset: str | Path | CogitaoDataset,
    *,
    num_workers: int = 4,
    output_path: str | Path = "color_distribution.png",
):
    """
    Analyze the color distribution of the dataset.

    Args:
        dataset: Path to the dataset or CogitaoDataset instance
        num_workers: Number of workers for DataLoader
        output_path: Path to save the output plot
    """
    if isinstance(dataset, str | Path):
        dataset = CogitaoDataset(dataset)
    elif isinstance(dataset, CogitaoDataset):
        pass
    else:
        raise TypeError("dataset must be a string, Path, or CogitaoDataset instance")

    if dataset.cfg.env_format == "grid":
        print("Grid format not supported for color analysis.")
        return

    # Build color palette (10 colors)
    color_palette = np.zeros((10, 3), dtype=np.float64)
    for i in range(10):
        test_grid = np.array([[i]])
        color_palette[i] = np.array(COLORMAP(NORM(test_grid)))[:, :, :3][0, 0]

    color_palette_t = torch.tensor(color_palette, dtype=torch.float32)

    # Optimization: move palette to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    color_palette_t = color_palette_t.to(device)

    # Create DataLoader
    loader = DataLoader(
        dataset, batch_size=dataset.batch_size, num_workers=num_workers, shuffle=False
    )

    color_counts = np.zeros(10, dtype=np.int64)

    print("Analyzing dataset colors...")
    for batch in tqdm(loader, desc="Batches"):
        # batch["inputs"] shape: (B, 3, H, W) or (B, H, W)
        imgs = batch["inputs"].to(device)
        B, C, H, W = imgs.shape

        # Reshape to (B*H*W, 3)
        pixels = imgs.permute(0, 2, 3, 1).reshape(-1, 3)

        # Distances to color_palette
        # (N, 1, 3) - (1, 10, 3) -> (N, 10, 3)
        distances = torch.sum(
            (pixels.unsqueeze(1) - color_palette_t.unsqueeze(0)) ** 2, dim=2
        )
        closest_colors = torch.argmin(distances, dim=1)

        counts = torch.bincount(closest_colors, minlength=10)
        color_counts += counts.cpu().numpy()

    # Plot
    total_pixels = color_counts.sum()
    percentages = (color_counts / total_pixels) * 100

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        range(10), color_counts, color=color_palette, edgecolor="black", linewidth=1.5
    )
    plt.xlabel("Color Index")
    plt.ylabel("Pixel Count")
    plt.title("Color Distribution in Dataset")
    plt.xticks(range(10))

    # Add percentage labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{percentages[i]:.2f}%",
            ha="center",
            va="bottom",
            rotation=90,
        )

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Color analysis saved to {output_path}")


def plot_dataset_examples(
    dataset: str | Path | CogitaoDataset,
    output_path: str | Path,
    num_images: int = 64,
    shuffle: bool = True,
):
    """
    Load a dataset and save a grid of images from it.
    """
    if isinstance(dataset, str | Path):
        dataset = CogitaoDataset(dataset)
    elif isinstance(dataset, CogitaoDataset):
        pass
    else:
        raise TypeError("dataset must be a string, Path, or CogitaoDataset instance")

    if dataset.cfg.env_format == "grid":
        print("Grid format not supported for color analysis.")
        return

    print(f"Dataset '{dataset.path}' has {len(dataset)} items.")
    dataloader = DataLoader(dataset, num_workers=4, shuffle=shuffle)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Collect images
    images = []
    num_to_print = min(num_images, len(dataset))

    for batch in dataloader:
        if len(images) >= num_to_print:
            break
        img = batch["inputs"]
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img)

        # Ensure image is [C, H, W]
        if len(img.shape) == 4:
            # If multiple images per task, take the first one or flatten
            img = img[0]

        images.append(img)

    print(f"Collected {len(images)} images.")

    if len(images) > 0:
        # Stack images into a batch [B, C, H, W]
        img_batch = torch.stack(images)

        import math

        B, C, H, W = img_batch.shape
        nrow = math.ceil(math.sqrt(B))
        ncol = math.ceil(B / nrow)

        if B < nrow * ncol:
            padding = torch.zeros(
                (nrow * ncol - B, C, H, W),
                dtype=img_batch.dtype,
                device=img_batch.device,
            )
            img_batch = torch.cat((img_batch, padding), dim=0)

        # Create grid layout
        grid = img_batch.view(nrow, ncol, C, H, W)
        grid = grid.permute(2, 0, 3, 1, 4).contiguous().view(C, nrow * H, ncol * W)

        # Convert to numpy and shape [H_total, W_total, C]
        image_np = grid.permute(1, 2, 0).numpy()

        # PIL needs uint8 array type
        if np.issubdtype(image_np.dtype, np.floating):
            # Scale if maximum is <= 1.0 (some margins for float accuracy)
            if image_np.max() <= 1.001:
                image_np = image_np * 255.0
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)

        if image_np.shape[-1] == 1:
            image_np = image_np.squeeze(-1)  # For grayscale PIL saving

        # Save as a grid using PIL
        PIL.Image.fromarray(image_np).save(output_file)
        print(f"Saved image grid to {output_file}")
    else:
        print("No images found to process.")
