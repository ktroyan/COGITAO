import numpy as np
import pytest

from arcworld.metrics import (
    compare_reconstruction_images,
    non_white_pixel_accuracy,
    number_of_perfectly_reconstructed_objects,
    number_of_perfectly_reconstructed_objects_batch,
    object_location_accuracy,
    object_location_accuracy_target_image,
    per_pixel_accuracy,
)
from arcworld.utils.img_transform import to_image


def test_non_white_pixel_accuracy():
    batch_size = 2
    image_size = 10

    # Create target
    target_grids = np.zeros((batch_size, image_size, image_size), dtype=np.int32)
    target_grids[0, 2:5, 2:5] = 1  # blue square
    target_grids[1, 5:8, 5:8] = 2  # red square
    targets = np.stack([to_image(g) for g in target_grids])

    # Perfect match
    preds = targets.copy()
    acc = non_white_pixel_accuracy(targets, preds)
    assert acc == 1.0

    # Half mismatch: batch 0 is correct, batch 1 is totally wrong
    mismatch_grids = target_grids.copy()
    mismatch_grids[1, 5:8, 5:8] = 3  # wrong color
    preds_half = np.stack([to_image(g) for g in mismatch_grids])

    acc_half = non_white_pixel_accuracy(targets, preds_half)
    assert np.isclose(acc_half, 0.5)

    # Blank targets
    blank_targets = np.zeros((batch_size, image_size, image_size), dtype=np.int32)
    blank_targets_img = np.stack([to_image(g) for g in blank_targets])
    acc_blank = non_white_pixel_accuracy(blank_targets_img, preds)
    assert acc_blank == 0.0


def test_per_pixel_accuracy():
    batch_size = 4
    image_size = 32

    # Create valid target grids and preds
    target_grids = np.zeros((batch_size, image_size, image_size), dtype=np.int32)
    target_grids[:, :10, :10] = 1  # blue square
    targets = np.stack([to_image(g) for g in target_grids])

    preds = targets.copy()

    # Perfect match
    acc = per_pixel_accuracy(targets, preds)
    assert acc == 1.0

    # Complete mismatch
    mismatch_grids = np.zeros((batch_size, image_size, image_size), dtype=np.int32)
    mismatch_grids[:, :, :] = 2  # red image entirely
    preds_mismatch = np.stack([to_image(g) for g in mismatch_grids])

    acc_mismatch = per_pixel_accuracy(targets, preds_mismatch)
    assert acc_mismatch == 0.0


def test_object_location_accuracy_target_image_perfect_match():
    # Create a simple grid
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[2:5, 2:5] = 1  # Blue square

    # Convert to image
    img = to_image(grid)  # (3, 10, 10)

    # Create batch of 1
    preds = img[np.newaxis, ...]  # (1, 3, 10, 10)
    targets = img[np.newaxis, ...]

    acc = object_location_accuracy_target_image(targets, preds)
    assert acc == 1.0, f"Expected 1.0 accuracy for perfect match, got {acc}"


def test_object_location_accuracy_grid_perfect_match():
    # Create a simple grid
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[2:5, 2:5] = 1  # Blue square

    # Convert to image for preds
    img = to_image(grid)
    preds = img[np.newaxis, ...]

    # Keep target as grid
    targets = grid[np.newaxis, ...]  # (1, 10, 10)

    acc = object_location_accuracy(targets, preds)
    assert acc == 1.0, (
        f"Expected 1.0 accuracy for perfect match with grid target, got {acc}"
    )


def test_object_location_accuracy_target_image_mismatch():
    # Grid 1: Blue square
    grid1 = np.zeros((10, 10), dtype=np.int32)
    grid1[2:5, 2:5] = 1

    # Grid 2: Red square
    grid2 = np.zeros((10, 10), dtype=np.int32)
    grid2[2:5, 2:5] = 2

    img1 = to_image(grid1)
    img2 = to_image(grid2)

    preds = img1[np.newaxis, ...]
    targets = img2[np.newaxis, ...]

    acc = object_location_accuracy_target_image(targets, preds)
    # Since masks match perfectly (locations are same), accuracy should be 1.0
    assert acc == 1.0, f"Expected 1.0 accuracy for mask match, got {acc}"

    # Now different location
    grid3 = np.zeros((10, 10), dtype=np.int32)
    grid3[5:8, 5:8] = 1

    img3 = to_image(grid3)

    targets = img3[np.newaxis, ...]

    acc = object_location_accuracy_target_image(targets, preds)
    # expected = 0.82
    expected = 0.0
    assert np.isclose(acc, expected), f"Expected {expected}, got {acc}"


def test_object_location_accuracy_target_image_batch():
    B = 4
    grids = []
    for i in range(B):
        g = np.zeros((10, 10), dtype=np.int32)
        if i % 2 == 0:
            g[1, 1] = 1
        grids.append(g)

    imgs = [to_image(g) for g in grids]
    batch = np.stack(imgs)  # (B, 3, 10, 10)

    # Perfect match
    acc = object_location_accuracy_target_image(batch, batch)
    assert acc == 1.0

    # Mismatch in one element
    targets = batch.copy()

    # Construct new grid for modification
    g_mod = np.zeros((10, 10), dtype=np.int32)
    g_mod[2, 2] = 1
    targets[0] = to_image(g_mod)

    acc = object_location_accuracy_target_image(targets, batch)
    # expected = 0.995
    expected = 1.0 / 3.0
    assert np.isclose(acc, expected), f"Expected {expected}, got {acc}"


@pytest.mark.parametrize(
    "gen_fixture",
    ["task_generator", "task_generator_image"],
)
def test_number_of_perfectly_reconstructed_objects_generated(request, gen_fixture):
    """Test single-object metrics with both grid and image env_formats."""
    gen = request.getfixturevalue(gen_fixture)

    task = gen.generate_single_task()
    if not task or "pairs" not in task:
        pytest.skip("Generator failed to produce a valid task")

    pair = task["pairs"][0]

    if gen_fixture == "task_generator":
        # env_format="grid": pair["input"] is a (H, W) int/float grid
        target_grid = np.array(pair["input"]).astype(np.int32)
        reconstructed_img = to_image(target_grid)  # (3, H, W), native size
    else:
        # env_format="image": pair["input"] is already a (3, H, W) float image
        from arcworld.utils.img_transform import to_grid

        reconstructed_img = np.array(pair["input"])  # (3, H, W)
        target_grid = to_grid(reconstructed_img).astype(np.int32)  # (H, W)

    if not np.any(target_grid > 0):
        pytest.skip("Generated grid has no coloured objects (unlikely but possible)")

    # Case 1: Perfect reconstruction
    found, duplicates, missed, extra = number_of_perfectly_reconstructed_objects(
        target_grid.copy(), reconstructed_img
    )
    assert found > 0
    assert duplicates == 0
    assert missed == 0
    assert extra == 0

    # Case 2: Blank prediction
    blank_img = to_image(np.zeros_like(target_grid))
    found, duplicates, missed, extra = number_of_perfectly_reconstructed_objects(
        target_grid.copy(), blank_img
    )
    assert found == 0
    assert duplicates == 0
    assert missed > 0
    assert extra == 0


@pytest.mark.parametrize(
    "gen_fixture",
    ["task_generator", "task_generator_image"],
)
def test_number_of_perfectly_reconstructed_objects_batch_generated(
    request, gen_fixture
):
    """Batch metrics test with both env_formats.

    We duplicate a single task's grid into a (B=2) batch so np.stack always
    works regardless of the grid size the generator produced.
    """
    from arcworld.utils.img_transform import to_grid

    gen = request.getfixturevalue(gen_fixture)
    task = gen.generate_single_task()
    if not task or "pairs" not in task:
        pytest.skip("Generator failed to produce a valid task")

    pair = task["pairs"][0]
    if gen_fixture == "task_generator":
        target_grid = np.array(pair["input"]).astype(np.int32)
    else:
        target_grid = to_grid(np.array(pair["input"])).astype(np.int32)

    if not np.any(target_grid > 0):
        pytest.skip("Generated grid has no coloured objects")

    targets = np.stack([target_grid, target_grid])  # (2, H, W)
    preds = np.stack([to_image(t) for t in targets])  # (2, 3, H, W)

    found, duplicates, missed, extra = number_of_perfectly_reconstructed_objects_batch(
        targets, preds
    )
    assert found > 0
    assert duplicates == 0
    assert missed == 0
    assert extra == 0

    # Mixed batch: row 1 perfect, row 2 blank
    preds_mixed = preds.copy()
    preds_mixed[1] = to_image(np.zeros_like(targets[1]))

    found_m, _, missed_m, _ = number_of_perfectly_reconstructed_objects_batch(
        targets, preds_mixed
    )
    assert found_m < found
    assert found_m + missed_m == found


@pytest.mark.parametrize(
    "gen_fixture",
    ["task_generator", "task_generator_image"],
)
def test_image_metrics_smoke_generated(request, gen_fixture):
    """Smoke test for compare_reconstruction_images with both env_formats."""
    from arcworld.utils.img_transform import to_grid

    gen = request.getfixturevalue(gen_fixture)
    task = gen.generate_single_task()
    if not task or "pairs" not in task:
        pytest.skip("Generator failed to produce a valid task")

    pair = task["pairs"][0]
    if gen_fixture == "task_generator":
        target_grid = np.array(pair["input"]).astype(np.int32)
    else:
        target_grid = to_grid(np.array(pair["input"])).astype(np.int32)

    if not np.any(target_grid > 0):
        pytest.skip("Generated grid has no coloured objects")

    targets_grid = np.stack([target_grid, target_grid])  # (2, H, W)
    targets_img = np.stack([to_image(g) for g in targets_grid])  # (2, 3, H, W)
    preds = targets_img

    metrics = compare_reconstruction_images(targets_img, preds)
    assert metrics["per_pixel_accuracy"] == 1.0
    assert metrics["non_white_pixel_accuracy"] == 1.0
    assert metrics["object_location_accuracy"] == 1.0
    assert metrics["number_of_perfectly_reconstructed_objects"]["missed"] == 0


def test_number_of_perfectly_reconstructed_objects_batch_with_slots():
    # Test with explicitly provided slots (N=2 objects)
    B = 2
    N = 2
    C, H, W = 3, 10, 10

    # Batch of objects: (B, N, C, H, W)
    objects = np.zeros((B, N, C, H, W), dtype=np.float32)

    # Batch of targets: (B, H, W)
    targets = np.zeros((B, H, W), dtype=np.int32)

    # Ex 1: Perfect match
    t1 = np.zeros((H, W), dtype=np.int32)
    t1[1:3, 1:3] = 1  # obj 1
    t1[5:7, 5:7] = 2  # obj 2
    targets[0] = t1

    # Objects for Ex 1
    o1_1 = to_image(np.zeros((H, W), dtype=np.int32))
    # We need to manually construct the image for the object because to_image expects a full grid
    # But usually objects are full grids masked?
    # number_of_... compares `to_state(object)` with `target_object` (which is a full grid with 0s elsewhere)
    # So we should pass full grid images.

    g1_1 = np.zeros((H, W), dtype=np.int32)
    g1_1[1:3, 1:3] = 1
    o1_1 = to_image(g1_1)

    g1_2 = np.zeros((H, W), dtype=np.int32)
    g1_2[5:7, 5:7] = 2
    o1_2 = to_image(g1_2)

    objects[0, 0] = o1_1
    objects[0, 1] = o1_2

    # Ex 2: 1 match, 1 miss, 1 extra (but N=2, so we can't represent 3 things perfectly unless we pad or use N=3)
    # Let's say we have N=2 slots. Target has 2 objects. We output 1 correct, 1 wrong.
    # Result: Match=1, Miss=1, Extra=1.
    t2 = t1.copy()
    targets[1] = t2

    # Obj 1 correct
    objects[1, 0] = o1_1
    # Obj 2 wrong (random color 3)
    g2_2 = np.zeros((H, W), dtype=np.int32)
    g2_2[8:9, 8:9] = 3
    objects[1, 1] = to_image(g2_2)

    found, duplicates, missed, extra = number_of_perfectly_reconstructed_objects_batch(
        targets, objects
    )

    # Ex 1: F=2, M=0, E=0
    # Ex 2: F=1, M=1, E=1 (Using N=2 slots, we provided 2 objects. 1 matched. 1 didn't match anything -> extra. 1 target object not matched -> missed.)

    assert found == 3
    assert missed == 1
    assert extra == 1


def test_image_metrics_crash_with_image_input():
    # image_metrics takes preds as (B, C, H, W)
    # It calls number_of... which expects (B, N, C, H, W).
    # If we pass (B, C, H, W), it iterates B. objects[i] is (C, H, W).
    # It attempts to treat (C, H, W) as list of objects?

    B, C, H, W = 2, 3, 10, 10
    preds = np.random.rand(B, C, H, W).astype(np.float32)
    targets = np.random.rand(B, C, H, W).astype(np.float32)  # images

    try:
        metrics = compare_reconstruction_images(targets, preds)
        print("image_metrics computed:", metrics)
    except Exception as e:
        pytest.fail(f"image_metrics crashed with standard image input: {e}")
