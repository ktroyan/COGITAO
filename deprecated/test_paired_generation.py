"""Test script for paired (shared-grid) ID/OOD generation.

Tests both the serial and parallel paths, then plots examples to visually
verify that ID and OOD tasks share the same input grids but have different
outputs (different transformation suites applied).
"""

import copy
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
from matplotlib import pyplot as plt

from arcworld.config import DatasetConfig
from arcworld.constants import COLORMAP, NORM
from arcworld.general_utils import plot_grid
from arcworld.generator import Generator

# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

_BASE = dict(
    min_n_shapes_per_grid=2,
    max_n_shapes_per_grid=2,
    n_examples=1,
    min_grid_size=15,
    max_grid_size=15,
    shape_compulsory_conditionals=[
        "is_shape_less_than_6_rows",
        "is_shape_less_than_6_cols",
        "is_shape_fully_connected",
    ],
)

ID_COMBOS = [["translate_up"], ["rot90"], ["mirror_horizontal"]]
OOD_COMBOS = [["translate_up", "rot90"], ["rot90", "translate_up"]]

id_cfg = DatasetConfig(**_BASE, allowed_combinations=ID_COMBOS)
ood_cfg = DatasetConfig(**_BASE, allowed_combinations=OOD_COMBOS)

N_TASKS = 20  # small count for testing


# ---------------------------------------------------------------------------
# 1. Test Generator-level API (generate_grid + generate_task_from_grid)
# ---------------------------------------------------------------------------

def test_generator_api():
    print("=" * 60)
    print("TEST 1: Generator.generate_grid + generate_task_from_grid")
    print("=" * 60)

    gen = Generator(id_cfg)

    n_ok = 0
    n_attempts = 0
    target = 10

    while n_ok < target and n_attempts < 50:
        n_attempts += 1
        result = gen.generate_grid()
        assert result is not None, f"generate_grid returned None on attempt {n_attempts}"
        input_grid, positioned_shapes = result

        # Some grids may be incompatible with certain transforms (e.g. shapes
        # near the edge that go off-grid after translate_up+rot90). This is
        # expected — in production the paired generator retries with a new grid.
        id_task = gen.generate_task_from_grid(input_grid, positioned_shapes, ["rot90"])
        if id_task is None:
            continue

        ood_task = gen.generate_task_from_grid(input_grid, positioned_shapes, ["mirror_horizontal", "rot90"])
        if ood_task is None:
            continue

        # Verify same input
        assert np.array_equal(
            id_task["pairs"][0]["input"], ood_task["pairs"][0]["input"]
        ), f"Input grids differ at attempt {n_attempts}!"

        # Verify transforms are different (outputs may coincidentally match
        # for symmetric shapes, so we check the transform suites instead)
        assert id_task["transformation_suite"] != ood_task["transformation_suite"], \
            f"Transform suites unexpectedly identical at attempt {n_attempts}"

        n_ok += 1

    assert n_ok == target, f"Only got {n_ok}/{target} valid pairs in {n_attempts} attempts"
    print(f"  PASSED: {n_ok}/{target} pairs verified ({n_attempts} attempts, "
          f"{n_attempts - n_ok} skipped due to transform incompatibility)\n")
    return gen


# ---------------------------------------------------------------------------
# 2. Test parallel paired generation
# ---------------------------------------------------------------------------

def test_parallel_paired():
    from generate_experiment_data_parallel import generate_paired_balanced_parallel

    print("=" * 60)
    print("TEST 2: generate_paired_balanced_parallel")
    print("=" * 60)

    tmpdir = Path(tempfile.mkdtemp())
    id_path = tmpdir / "val.h5"
    ood_path = tmpdir / "val_ood.h5"

    try:
        n = generate_paired_balanced_parallel(
            id_cfg, ood_cfg,
            id_path, ood_path,
            n_tasks=N_TASKS, num_workers=2,
            db_name="test_paired_parallel", db_path=tmpdir / "db",
        )
        assert n == N_TASKS, f"Expected {N_TASKS}, got {n}"

        with h5py.File(id_path, "r") as f_id, h5py.File(ood_path, "r") as f_ood:
            n_id = f_id["inputs"].shape[0]
            n_ood = f_ood["inputs"].shape[0]
            assert n_id == n_ood == N_TASKS, f"Count mismatch: ID={n_id}, OOD={n_ood}"

            # Check 1-to-1 input grid matching
            mismatches = 0
            for i in range(n_id):
                if not np.array_equal(f_id["inputs"][i], f_ood["inputs"][i]):
                    mismatches += 1

            assert mismatches == 0, f"{mismatches}/{n_id} input grids differ!"
            print(f"  PASSED: {n_id} paired samples, all input grids match 1-to-1")

            # Check balance
            id_transforms = [s.decode() for s in f_id["transformation_suites"][:]]
            ood_transforms = [s.decode() for s in f_ood["transformation_suites"][:]]

            from collections import Counter
            id_counts = Counter(id_transforms)
            ood_counts = Counter(ood_transforms)
            print(f"  ID balance:  {dict(id_counts)}")
            print(f"  OOD balance: {dict(ood_counts)}")

        # Return paths for plotting
        return id_path, ood_path, tmpdir

    except Exception:
        shutil.rmtree(tmpdir)
        raise


# ---------------------------------------------------------------------------
# 3. Test serial paired generation
# ---------------------------------------------------------------------------

def test_serial_paired():
    # The serial script has pre-existing import issues (c0 module name mismatch).
    # Import just the functions we need by working around the top-level imports.
    # The serial script has a pre-existing broken import (c0 module name mismatch).
    # Patch it before importing the module.
    try:
        from generate_experiment_data import generate_paired_splits
    except ImportError:
        print("  (Working around pre-existing import issue in generate_experiment_data.py)")
        from experiment_configs.c0 import c0_configs
        import experiment_configs.c0 as c0_mod
        c0_mod.compositionality_configs = c0_configs
        from generate_experiment_data import generate_paired_splits

    print("\n" + "=" * 60)
    print("TEST 3: generate_paired_splits (serial)")
    print("=" * 60)

    tmpdir = Path(tempfile.mkdtemp())
    id_file = str(tmpdir / "val.json")
    ood_file = str(tmpdir / "val_ood.json")

    # The serial function expects dict configs with saving_path
    id_config_dict = {
        **_BASE,
        "allowed_combinations": ID_COMBOS,
        "saving_path": str(tmpdir / "val.json"),
    }
    ood_config_dict = {
        **_BASE,
        "allowed_combinations": OOD_COMBOS,
        "saving_path": str(tmpdir / "val_ood.json"),
    }

    try:
        generate_paired_splits(id_config_dict, ood_config_dict, N_TASKS, id_file, ood_file)

        import json
        with open(id_file) as f:
            id_tasks = json.load(f)
        with open(ood_file) as f:
            ood_tasks = json.load(f)

        assert len(id_tasks) == len(ood_tasks) == N_TASKS, \
            f"Count mismatch: ID={len(id_tasks)}, OOD={len(ood_tasks)}"

        mismatches = 0
        for i in range(N_TASKS):
            if id_tasks[i]["input"] != ood_tasks[i]["input"]:
                mismatches += 1

        assert mismatches == 0, f"{mismatches}/{N_TASKS} input grids differ!"
        print(f"  PASSED: {N_TASKS} paired samples, all input grids match 1-to-1")

        from collections import Counter
        id_counts = Counter(str(t["transformation_suite"]) for t in id_tasks)
        ood_counts = Counter(str(t["transformation_suite"]) for t in ood_tasks)
        print(f"  ID balance:  {dict(id_counts)}")
        print(f"  OOD balance: {dict(ood_counts)}")

        return id_tasks, ood_tasks, tmpdir

    except Exception:
        shutil.rmtree(tmpdir)
        raise


# ---------------------------------------------------------------------------
# 4. Plot examples
# ---------------------------------------------------------------------------

def plot_paired_examples_from_h5(id_path, ood_path, n_examples=5, save_path=None):
    """Plot paired ID/OOD examples from HDF5 files side by side."""
    print("\n" + "=" * 60)
    print("PLOTTING: Paired examples from parallel generation (HDF5)")
    print("=" * 60)

    with h5py.File(id_path, "r") as f_id, h5py.File(ood_path, "r") as f_ood:
        n_available = min(f_id["inputs"].shape[0], n_examples)

        fig, axes = plt.subplots(n_available, 4, figsize=(20, 5 * n_available))
        if n_available == 1:
            axes = axes[np.newaxis, :]

        fig.suptitle(
            "Paired Generation Verification (Parallel/HDF5)\n"
            "Columns: ID Input | ID Output | OOD Input | OOD Output\n"
            "ID and OOD inputs should be identical",
            fontsize=14, fontweight="bold", y=1.02,
        )

        for i in range(n_available):
            id_input = f_id["inputs"][i]
            id_output = f_id["outputs"][i]
            ood_input = f_ood["inputs"][i]
            ood_output = f_ood["outputs"][i]
            id_ts = f_id["transformation_suites"][i].decode()
            ood_ts = f_ood["transformation_suites"][i].decode()

            grids_match = np.array_equal(id_input, ood_input)
            match_label = "MATCH" if grids_match else "MISMATCH"

            for col, (grid, title) in enumerate([
                (id_input, f"ID Input\n{id_ts}"),
                (id_output, f"ID Output"),
                (ood_input, f"OOD Input [{match_label}]\n{ood_ts}"),
                (ood_output, f"OOD Output"),
            ]):
                ax = axes[i, col]
                ax.imshow(grid, cmap=COLORMAP, norm=NORM)
                ax.grid(True, which="both", color="lightgrey", linewidth=0.5)
                ax.set_yticks([x - 0.5 for x in range(1 + grid.shape[0])])
                ax.set_xticks([x - 0.5 for x in range(1 + grid.shape[1])])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_title(title, fontsize=10)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved to {save_path}")
        plt.show()


def plot_paired_examples_from_json(id_tasks, ood_tasks, n_examples=5, save_path=None):
    """Plot paired ID/OOD examples from JSON task lists side by side."""
    print("\n" + "=" * 60)
    print("PLOTTING: Paired examples from serial generation (JSON)")
    print("=" * 60)

    n_available = min(len(id_tasks), n_examples)

    fig, axes = plt.subplots(n_available, 4, figsize=(20, 5 * n_available))
    if n_available == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        "Paired Generation Verification (Serial/JSON)\n"
        "Columns: ID Input | ID Output | OOD Input | OOD Output\n"
        "ID and OOD inputs should be identical",
        fontsize=14, fontweight="bold", y=1.02,
    )

    for i in range(n_available):
        id_input = np.array(id_tasks[i]["input"])
        id_output = np.array(id_tasks[i]["output"])
        ood_input = np.array(ood_tasks[i]["input"])
        ood_output = np.array(ood_tasks[i]["output"])
        id_ts = str(id_tasks[i]["transformation_suite"])
        ood_ts = str(ood_tasks[i]["transformation_suite"])

        grids_match = np.array_equal(id_input, ood_input)
        match_label = "MATCH" if grids_match else "MISMATCH"

        for col, (grid, title) in enumerate([
            (id_input, f"ID Input\n{id_ts}"),
            (id_output, f"ID Output"),
            (ood_input, f"OOD Input [{match_label}]\n{ood_ts}"),
            (ood_output, f"OOD Output"),
        ]):
            ax = axes[i, col]
            ax.imshow(grid, cmap=COLORMAP, norm=NORM)
            ax.grid(True, which="both", color="lightgrey", linewidth=0.5)
            ax.set_yticks([x - 0.5 for x in range(1 + grid.shape[0])])
            ax.set_xticks([x - 0.5 for x in range(1 + grid.shape[1])])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_title(title, fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved to {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    output_dir = Path("test_paired_output")
    output_dir.mkdir(exist_ok=True)

    # Test 1: Generator API
    test_generator_api()

    # Test 2: Parallel paired generation
    id_h5_path, ood_h5_path, parallel_tmpdir = test_parallel_paired()
    plot_paired_examples_from_h5(
        id_h5_path, ood_h5_path,
        n_examples=5,
        save_path=str(output_dir / "paired_parallel_examples.png"),
    )

    # Test 3: Serial paired generation
    id_json_tasks, ood_json_tasks, serial_tmpdir = test_serial_paired()
    plot_paired_examples_from_json(
        id_json_tasks, ood_json_tasks,
        n_examples=5,
        save_path=str(output_dir / "paired_serial_examples.png"),
    )

    # Cleanup temp dirs
    shutil.rmtree(parallel_tmpdir)
    shutil.rmtree(serial_tmpdir)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print(f"Plots saved to {output_dir}/")
    print("=" * 60)
