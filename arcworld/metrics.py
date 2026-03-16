import numpy as np

from .utils.img_transform import to_grid

# ---------------------------------------------------------------------------
# Grid-based metrics (primary implementations)
# ---------------------------------------------------------------------------


def non_white_pixel_accuracy_grid(
    targets: np.ndarray,
    preds: np.ndarray,
) -> float:
    """Accuracy of grid recreation for non-white (non-zero) pixels of the target.

    Args:
        targets: (B, H, W) or (H, W) target grids with integer values
        preds:   (B, H, W) or (H, W) predicted grids with integer values

    Returns:
        Fraction of non-zero target pixels that are correctly predicted.
    """
    mask = targets > 0
    if not np.any(mask):
        return 0.0
    return float((targets[mask] == preds[mask]).mean())


def per_pixel_accuracy_grid(
    targets: np.ndarray,
    preds: np.ndarray,
) -> float:
    """Per-pixel accuracy across the full grid (including background/zero pixels).

    Args:
        targets: (B, H, W) or (H, W) target grids
        preds:   (B, H, W) or (H, W) predicted grids

    Returns:
        Fraction of all pixels that match exactly.
    """
    return float((targets == preds).mean())


def object_location_accuracy_grid(
    targets: np.ndarray,
    preds: np.ndarray,
) -> float:
    """IOU of non-zero pixel masks between target and predicted grids.

    Args:
        targets: (B, H, W) or (H, W) target grids
        preds:   (B, H, W) or (H, W) predicted grids

    Returns:
        Intersection-over-union of non-zero masks.
    """
    targets_mask = targets > 0
    preds_mask = preds > 0
    union = (targets_mask | preds_mask).sum()
    if union == 0:
        return 0.0
    return float((targets_mask & preds_mask).sum() / union)


def _extract_objects(grid: np.ndarray) -> list[np.ndarray]:
    """Extract connected components of non-zero pixels as separate object grids.

    Each returned grid has the same shape as input, containing one object.

    Args:
        grid: (H, W) grid

    Returns:
        List of (H, W) grids, one per connected component.
    """
    grid_working = grid.copy()
    objects = []

    while np.any(grid_working > 0):
        # Find first coloured pixel
        y, x = np.argwhere(grid_working > 0)[0]
        color = grid_working[y, x]

        # BFS to find all connected pixels of the same colour
        object_mask = np.zeros_like(grid_working, dtype=bool)
        object_mask[y, x] = True
        queue = {(y, x)}
        while queue:
            cy, cx = queue.pop()
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ny, nx = cy + dy, cx + dx
                if (
                    0 <= ny < grid_working.shape[0]
                    and 0 <= nx < grid_working.shape[1]
                    and grid_working[ny, nx] == color
                    and not object_mask[ny, nx]
                ):
                    object_mask[ny, nx] = True
                    queue.add((ny, nx))

        obj_grid = np.zeros_like(grid_working)
        obj_grid[object_mask] = grid_working[object_mask]
        objects.append(obj_grid)
        grid_working[object_mask] = 0

    return objects


def number_of_perfectly_reconstructed_objects_grid(
    target: np.ndarray,
    preds: np.ndarray,
) -> tuple[int, int, int, int]:
    """Compare how many objects in *target* are perfectly reconstructed by *preds*.

    All-white (all-zero) predicted grids are ignored.

    Args:
        target: (H, W) target grid
        preds:  (H, W) predicted grid  OR  (N, H, W) array of N object grids
                (each slot being a full-canvas grid with one object)

    Returns:
        (found, duplicates, missed, extra)
    """
    target_objects = _extract_objects(target)

    if preds.ndim == 2:
        # Single (H, W) grid – extract objects from it
        found_object_grids = _extract_objects(preds)
    else:
        # (N, H, W) – each slice is already one object grid; drop all-white slots
        found_object_grids = [p for p in preds if np.any(p > 0)]

    target_objects_found = [False] * len(target_objects)
    found = 0
    duplicates = 0
    missed = 0

    for i, target_object in enumerate(target_objects):
        for found_object in found_object_grids:
            if np.array_equal(target_object, found_object):
                if target_objects_found[i]:
                    duplicates += 1
                else:
                    target_objects_found[i] = True
                    found += 1
        if not target_objects_found[i]:
            missed += 1

    extra = len(found_object_grids) - found - duplicates

    return found, duplicates, missed, extra


def number_of_perfectly_reconstructed_objects_batch_grid(
    targets: np.ndarray,
    preds: np.ndarray,
) -> tuple[int, int, int, int]:
    """Batch version of :func:`number_of_perfectly_reconstructed_objects_grid`.

    Args:
        targets: (B, H, W) target grids
        preds:   (B, H, W) predicted grids  OR  (B, N, H, W) object grids

    Returns:
        Summed (found, duplicates, missed, extra) over the whole batch.
    """
    found = duplicates = missed = extra = 0

    for i in range(targets.shape[0]):
        f, d, m, e = number_of_perfectly_reconstructed_objects_grid(
            targets[i], preds[i]
        )
        found += f
        duplicates += d
        missed += m
        extra += e

    return found, duplicates, missed, extra


def compare_reconstruction_grids(
    targets: np.ndarray,
    preds: np.ndarray,
) -> dict[str, float | dict[str, float]]:
    """Aggregate all metrics for a batch of output grids.

    Args:
        targets: (B, H, W) target grids
        preds:   (B, H, W) predicted grids  OR  (B, N, H, W) object grids

    Returns:
        Dictionary of metric name → value (or sub-dict for object counts).
    """
    found, duplicates, missed, extra = (
        number_of_perfectly_reconstructed_objects_batch_grid(targets, preds)
    )
    total = found + duplicates + missed + extra or 1  # avoid div-by-zero

    return {
        "per_pixel_accuracy": per_pixel_accuracy_grid(targets, preds),
        "non_white_pixel_accuracy": non_white_pixel_accuracy_grid(targets, preds),
        "object_location_accuracy": object_location_accuracy_grid(targets, preds),
        "number_of_perfectly_reconstructed_objects": {
            "found": found / total,
            "duplicates": duplicates / total,
            "missed": missed / total,
            "extra": extra / total,
        },
    }


# ---------------------------------------------------------------------------
# Image-based wrappers (convert images → grids, then call grid functions)
# ---------------------------------------------------------------------------


def non_white_pixel_accuracy(
    targets: np.ndarray,
    preds: np.ndarray,
    *,
    grid_size: int | None = None,
) -> float:
    """Non-white pixel accuracy for image inputs.

    Converts (B, C, H, W) images to grids, then delegates to
    :func:`non_white_pixel_accuracy_grid`.

    Args:
        targets:   (B, C, H, W) target images
        preds:     (B, C, H, W) predicted images
        grid_size: Optional downscale target for :func:`to_grid`
    """
    return non_white_pixel_accuracy_grid(
        to_grid(targets, grid_size=grid_size),
        to_grid(preds, grid_size=grid_size),
    )


def per_pixel_accuracy(
    targets: np.ndarray,
    preds: np.ndarray,
    *,
    grid_size: int | None = None,
) -> float:
    """Per-pixel accuracy for image inputs.

    Converts (B, C, H, W) images to grids (at native resolution unless
    *grid_size* is given), then delegates to :func:`per_pixel_accuracy_grid`.

    Args:
        targets:   (B, C, H, W) target images
        preds:     (B, C, H, W) predicted images
        grid_size: Optional downscale target for :func:`to_grid`
    """
    resolved = grid_size or targets.shape[-1]
    return per_pixel_accuracy_grid(
        to_grid(targets, grid_size=resolved),
        to_grid(preds, grid_size=resolved),
    )


def object_location_accuracy(
    targets: np.ndarray,
    preds: np.ndarray,
    *,
    grid_size: int | None = None,
) -> float:
    """IOU location accuracy where *targets* may already be grids or images.

    If *targets* has 2 dimensions (H, W) or 3 dimensions (B, H, W) it is
    treated as a grid directly.  If 4 dimensions (B, C, H, W) it is converted
    via :func:`to_grid` first.  *preds* is always converted.

    Args:
        targets:   (B, H, W) target grids  OR  (B, C, H, W) target images
        preds:     (B, C, H, W) predicted images
        grid_size: Optional downscale target when converting from images
    """
    if targets.ndim == 4:
        targets_grid = to_grid(targets, grid_size=grid_size)
    else:
        targets_grid = targets
    preds_grid = to_grid(preds, grid_size=grid_size)
    return object_location_accuracy_grid(targets_grid, preds_grid)


def object_location_accuracy_target_image(
    targets: np.ndarray,
    preds: np.ndarray,
    *,
    grid_size: int | None = None,
) -> float:
    """IOU location accuracy when *both* targets and preds are images.

    Converts both (B, C, H, W) arrays to grids, then delegates to
    :func:`object_location_accuracy_grid`.

    Args:
        targets:   (B, C, H, W) target images
        preds:     (B, C, H, W) predicted images
        grid_size: Optional downscale target for :func:`to_grid`
    """
    return object_location_accuracy_grid(
        to_grid(targets, grid_size=grid_size),
        to_grid(preds, grid_size=grid_size),
    )


def number_of_perfectly_reconstructed_objects(
    target: np.ndarray,
    preds: np.ndarray,
    *,
    grid_size: int | tuple[int, int] | None = None,
) -> tuple[int, int, int, int]:
    """Object reconstruction metric where *target* is a grid and *preds* are images.

    Args:
        target:    (H, W) target grid
        preds:     (C, H, W) single predicted image  OR
                   (N, C, H, W) images of N object slots

    Returns:
        (found, duplicates, missed, extra)
    """
    resolved_size = grid_size or target.shape
    preds_grid = to_grid(preds, grid_size=resolved_size)
    return number_of_perfectly_reconstructed_objects_grid(target, preds_grid)


def number_of_perfectly_reconstructed_objects_batch(
    targets: np.ndarray,
    preds: np.ndarray,
    *,
    grid_size: int | tuple[int, int] | None = None,
) -> tuple[int, int, int, int]:
    """Batch object reconstruction metric: *targets* are grids, *preds* are images.

    Args:
        targets:   (B, H, W) target grids
        preds:     (B, C, H, W) predicted images  OR  (B, N, C, H, W) object images

    Returns:
        Summed (found, duplicates, missed, extra) over the batch.
    """
    found = duplicates = missed = extra = 0

    for i in range(targets.shape[0]):
        f, d, m, e = number_of_perfectly_reconstructed_objects(
            targets[i], preds[i], grid_size=grid_size
        )
        found += f
        duplicates += d
        missed += m
        extra += e

    return found, duplicates, missed, extra


def compare_reconstruction_images(
    targets: np.ndarray,
    preds: np.ndarray,
    objects: np.ndarray | None = None,
    *,
    grid_size: int | None = None,
) -> dict[str, float | dict[str, float]]:
    """Aggregate all metrics for a batch of predicted images.

    Converts images to grids for the object-level metric; uses
    image-aware wrappers for pixel metrics.

    Args:
        targets:   (B, C, H, W) target images
        preds:     (B, C, H, W) predicted images
        objects:   (B, N, C, H, W) or (B, C, H, W) object images (optional).
                   When provided, used for the object reconstruction metric
                   instead of *preds*.
        grid_size: Optional downscale target for :func:`to_grid`

    Returns:
        Dictionary of metric name → value (or sub-dict for object counts).
    """
    target_grid = to_grid(targets, grid_size=grid_size)
    preds_grid = to_grid(preds, grid_size=grid_size)

    found, duplicates, missed, extra = number_of_perfectly_reconstructed_objects_batch(
        target_grid, objects if objects is not None else preds, grid_size=grid_size
    )

    return {
        "per_pixel_accuracy": per_pixel_accuracy_grid(target_grid, preds_grid),
        "non_white_pixel_accuracy": non_white_pixel_accuracy_grid(
            target_grid, preds_grid
        ),
        "object_location_accuracy": object_location_accuracy_grid(
            target_grid, preds_grid
        ),
        "number_of_perfectly_reconstructed_objects": {
            "found": found,
            "duplicates": duplicates,
            "missed": missed,
            "extra": extra,
        },
    }
