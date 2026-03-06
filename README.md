## Usage as a package

### Installation
Using uv:
```shell
uv add git+https://github.com/yassinetb/COGITAO.git
```

Using pip:
```shell
pip install git+https://github.com/yassinetb/COGITAO.git
```

Using poetry:
```shell
poetry add git+https://github.com/yassinetb/COGITAO.git
```

If you plan to use CogitaoDataset, you should install torch as well: 
  pip: https://pytorch.org/get-started/locally/
  uv: https://docs.astral.sh/uv/guides/integration/pytorch/

## Installation from source

We use [poetry](https://python-poetry.org/) as our dependency manager and build
system framework. Please install it using:

```shell
$ pip install poetry
```
and install the dependencies by running:
```shell
$ poetry update
```

Install the library in editable mode so that changes in the codebase
can be tested easily.

```shell
$ pip install -e .
```

## Run code 

Use of our generator is demonstrated in the `demo.ipynb` jupyter notebook...

### Usage

To generate dataset:
```python
from arcworld.generator import GeneratorConfig, DatasetConfig, ParallelGenerator

cfg = GeneratorConfig(
    output_dir="./data",
    output_file="train.h5",
    num_workers=64,
    dataset=DatasetConfig(
        env_format="image",
        image_size=224,
        image_upscale_method="nearest",
        n_examples=10000,
        batch_size=16,
        min_grid_size=32,
        max_grid_size=32,
        min_n_shapes_per_grid=2,
        max_n_shapes_per_grid=3,
        allowed_transformations=[
            "rot90",
        ],
        min_transformation_depth=0,
        max_transformation_depth=0,
        shape_compulsory_conditionals=[
            "is_shape_less_than_11_rows",
            "is_shape_less_than_11_cols",
            "is_shape_more_than_2_cell",
            "is_shape_evenly_colored",
            "is_shape_fully_connected",
            "is_shape_not_hollow",
        ],
    ),
)

# Generate items
parallel_generator = ParallelGenerator(cfg)
parallel_generator.generate()
```

To use for pytorch dataloader:
```python
from arcworld.generator import CogitaoDataset
from torch.utils.data import DataLoader 
dataset = CogitaoDataset("dset_images.h5")

dataloader= DataLoader(dataset, batch_size=dataset.cfg.batch_size, ...)
```
<b>Note: </b>setting batch_size=dataset.cfg.batch_size is purely performance improvement. It is not strictly necessary

## Generating Experiment Datasets

COGITAO supports generating datasets with controlled transformation compositions, useful for studying compositionality, generalization, and sample efficiency. Generation is parallelized across multiple worker processes.

### Config structure

A `GeneratorConfig` wraps a `DatasetConfig` (which defines the task distribution) along with output and parallelization settings:

```python
from arcworld.generator import GeneratorConfig, DatasetConfig, ParallelGenerator

cfg = GeneratorConfig(
    output_dir="./data/compositionality/setting_1/experiment_1",
    output_file="train.h5",
    num_workers=64,
    dataset=DatasetConfig(
        n_examples=100000,
        batch_size=16,
        min_grid_size=20,
        max_grid_size=20,
        min_n_shapes_per_grid=2,
        max_n_shapes_per_grid=2,
        # --- Transformation control (pick ONE of the two options) ---
        # Option A: explicit transformation sequences (compositions)
        allowed_combinations=[
            ["translate_up"],
            ["rot90"],
            ["translate_up", "rot90"],
            ["rot90", "mirror_horizontal"],
        ],
        allowed_transformations=None,
        min_transformation_depth=None,
        max_transformation_depth=None,
        # Option B: let the generator randomly compose from a set
        # allowed_combinations=None,
        # allowed_transformations=["rot90", "mirror_horizontal", "translate_up"],
        # min_transformation_depth=1,
        # max_transformation_depth=3,
        # --- Shape constraints ---
        shape_compulsory_conditionals=[
            "is_shape_less_than_6_rows",
            "is_shape_less_than_6_cols",
            "is_shape_fully_connected",
        ],
        # --- Output format ---
        env_format="grid",          # "grid" for raw arrays, "image" for upscaled images
        # image_size=224,           # required when env_format="image"
        # image_upscale_method="nearest",
    ),
)
```

**Transformation control** — exactly one of these must be provided:
- `allowed_combinations`: a list of explicit transformation sequences (e.g. `[["rot90", "translate_up"], ["mirror_horizontal"]]`). Each inner list is an ordered composition applied to shapes. The generator samples uniformly across the provided combinations.
- `allowed_transformations` + `min/max_transformation_depth`: a list of individual transforms. The generator randomly composes them into sequences of length between `min_transformation_depth` and `max_transformation_depth`.

### Running generation

```python
gen = ParallelGenerator(cfg)
gen.generate()  # spawns num_workers processes, writes to output_dir/output_file
```

`ParallelGenerator` spawns `num_workers` processes that each run an independent `Generator` instance. Samples are placed in a shared queue and written to an HDF5 file by the main process. You can override the number of samples at call time:

```python
gen.generate(num_samples=5000)
```

### Experiment config pattern

Each config file in `experiment_configs/` exports a list of `ExperimentEntry` objects. An `ExperimentEntry` pairs a `DatasetConfig` (generation parameters) with experiment metadata:

```python
@dataclass
class ExperimentEntry:
    cfg: DatasetConfig       # transformations, grid size, shapes, conditionals, etc.
    setting: int             # experiment setting number
    experiment: int          # experiment number within setting
    split: str               # "train" or "test" (role — see split expansion below)
    subdir: str | None       # optional extra path segment (e.g. "grid_size_30")
```

Config files use a shared base dict and a `make_entry()` factory. For example (from `experiment_configs/compositionality.py`):

```python
from arcworld.config import DatasetConfig
from experiment_configs.entry import ExperimentEntry

_BASE = dict(
    min_n_shapes_per_grid=2,
    max_n_shapes_per_grid=2,
    n_examples=1,
    shape_compulsory_conditionals=[
        "is_shape_less_than_6_rows",
        "is_shape_less_than_6_cols",
        "is_shape_fully_connected",
    ],
)

def make_entry(combos, setting, exp_number, split, min_size=15, max_size=15):
    cfg = DatasetConfig(
        **_BASE,
        allowed_combinations=combos,
        min_grid_size=min_size,
        max_grid_size=max_size,
    )
    return ExperimentEntry(cfg=cfg, setting=setting, experiment=exp_number, split=split)

compositionality_configs: list[ExperimentEntry] = []

# Train: all single and pairwise compositions
compositionality_configs.append(make_entry(
    [["translate_up"], ["rot90"], ["translate_up", "rot90"], ["rot90", "translate_up"]],
    setting=1, exp_number=1, split="train",
))

# Test: held-out compositions (used for OOD evaluation)
compositionality_configs.append(make_entry(
    [["translate_up", "mirror_horizontal"], ["mirror_horizontal", "translate_up"]],
    setting=1, exp_number=1, split="test",
))
```

To add a new experiment, append more `make_entry()` calls to the list in an existing config file, or create a new file following the same pattern and register it in `generate_experiment_data_parallel.py` (see `_load_studies()`).

### Running experiment generation

`generate_experiment_data_parallel.py` generates balanced HDF5 datasets from experiment configs using multiprocessing. Key features:

- **Exact balance** — generates exactly `N / len(combinations)` samples per transformation combination
- **Deduplication** — SHA256 hash of each task's input grid + transformation suite, checked against a shared SQLite DB
- **Split expansion** — each `ExperimentEntry` with `split="train"` auto-expands into `train.h5`, `val.h5`, and `test.h5`; entries with `split="test"` expand into `val_ood.h5` and `test_ood.h5`. All splits within the same experiment share a dedup DB to prevent cross-split duplicates.
- **Parallelization** — spawns `num_workers` processes, each running an independent `Generator` instance. Workers pull specific combination assignments from a shared queue; the main process deduplicates and writes to HDF5.

```shell
# Generate one study
python generate_experiment_data_parallel.py \
    --study compositionality \
    --num-workers 64 \
    --output-dir ./data \
    --n-train 1000000 --n-val 1000 --n-test 1000

# Generate all studies
python generate_experiment_data_parallel.py --study all --num-workers 64

# Small test run
python generate_experiment_data_parallel.py --study c0 --num-workers 4 --n-train 100 --n-val 10 --n-test 10
```

Output structure:
```
data/
  compositionality/
    exp_setting_1/
      experiment_1/
        train.h5        # from "train" entry
        val.h5          # from "train" entry
        test.h5         # from "train" entry
        val_ood.h5      # from "test" entry
        test_ood.h5     # from "test" entry
```

Files already present on disk are skipped, so you can safely re-run after a partial failure.

### Existing experiment suites

| Config file | Study | CLI `--study` name |
|---|---|---|
| `experiment_configs/compositionality.py` | Compositionality across 3 settings (single ops, pairwise, depth-3) | `compositionality` |
| `experiment_configs/c0.py` | Translation-only compositionality baseline | `c0` |
| `experiment_configs/generalization.py` | Generalization along 5 axes: n-shapes, grid size, object dimension, object complexity, and all combined | `generalization` |
| `experiment_configs/sample_efficiency.py` | Sample efficiency with varying training set sizes | `sample_efficiency` |
| `experiment_configs/compositionality_gridsize.py` | Compositionality with varying grid sizes | `compositionality_gridsize` |
| `experiment_configs/c4.py` | Compositional scaling (progressive transform pool growth) | `c4` |

### Converting HDF5 to Parquet and uploading to Hugging Face

The generated HDF5 files can be converted to Parquet for uploading to Hugging Face Datasets. Grids are stored padded to `max_grid_size` in the HDF5, so they need to be cropped back using the stored `grid_sizes` before export.

**Converting a single HDF5 file to Parquet:**

```python
import h5py
import json
import numpy as np
import pandas as pd
from pathlib import Path


def h5_to_parquet(h5_path, parquet_path):
    """Convert a COGITAO HDF5 file to Parquet (compatible with HF Datasets)."""
    with h5py.File(h5_path, "r") as f:
        n = f["inputs"].shape[0]
        inputs = f["inputs"][:]
        outputs = f["outputs"][:]
        grid_sizes = f["grid_sizes"][:]
        transformation_suites = f["transformation_suites"][:]

    rows = []
    for i in range(n):
        h, w = int(grid_sizes[i][0]), int(grid_sizes[i][1])
        rows.append({
            "input": inputs[i, :h, :w].tolist(),
            "output": outputs[i, :h, :w].tolist(),
            "transformation_suite": json.loads(transformation_suites[i]),
        })

    df = pd.DataFrame(rows)
    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    print(f"Converted {h5_path} → {parquet_path} ({n} rows)")
```

**Batch-converting an entire output directory:**

```python
from pathlib import Path

data_dir = Path("./data/compositionality")
parquet_dir = Path("./data_parquet/compositionality")

for h5_file in sorted(data_dir.rglob("*.h5")):
    rel = h5_file.relative_to(data_dir)
    parquet_file = parquet_dir / rel.with_suffix(".parquet")
    h5_to_parquet(h5_file, parquet_file)
```

**Uploading to Hugging Face:**

```python
from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="./data_parquet/compositionality",
    path_in_repo="compositionality/",
    repo_id="your-username/your-dataset",
    repo_type="dataset",
    commit_message="Upload compositionality experiment data",
    ignore_patterns=["*.db"],
)
```

You can then load the data from Hugging Face:

```python
from datasets import load_dataset

dataset = load_dataset(
    "your-username/your-dataset",
    data_files={"data": "compositionality/exp_setting_1/experiment_1/train.parquet"},
)
print(dataset["data"][0].keys())
# dict_keys(['input', 'output', 'transformation_suite'])
```

<!--

### To mention: 

- [ ] What the point cloud and shape classes are for
- [ ] How does generating shapes work, and why it's the first step to using the generator
- [ ] How one should pre-compute conditionals for shapes 

# Installation 

# Demos 

# Algorithm 

# Shapes 

# Transforms 

# Conditionals 

# Before-ARC Dataset 

# Cite 

The generator relies on pre-created set of shapes stored in a dedicated .h5py file. Furthermore, the conditionals for the shapes are 
also pre-computed. This simplifies the generation process as generation shapes on the fly based on conditionals can be complex. 

For simplicity, you can use our lightweight precomputed set of shapes with precomputed conditionals "arcworld/datasets/shapes.h5py".

For extended use:

* If you'd like to extend the set of shapes, you can run the "shapes.py" file in the shapes folder, which in the published form maximizes 
diversity of generated shapes patterns. You can extend the number of shapes by modifying the "k_obj_per_config" variable in the "generate_shapes.py file. This will create N different shapes per configuration, as opposed to just one in the demo format. You can specify the file name on the hdf5_utils.py file. 

* If you'd like to add conditionals to the generator, you should simply re-run the "compute_conditions.py" script in the shapes folder. You can re compute the entirety of the conditionals, or simply compute the conditional for your new conditional of interest. 


# Instruction on configuration of the generator: 

## Number of Shapes: x  

Constraints for shapes should be considered in relation to the grid size, and the maximum shape size wanted by the user. For instance, if the maximum shape dimension is 6x6, and the grid size is 10x10, the generator will be forced to randomly sample shapes smaller than 6x6 if it wants them to fit more than one shape into the 10x10 grid. 

## Grid Size

As mentionned above, this should be considered in relation to the number of shapes, as well as to the number of transformations. We recommend having a minimum grid size of 10, as below this grid size, the capabilities of the generator starts to be come difficult to utilize (although it works). 

## Number of Examples

This is the number of examples to return for a given sample transform suite, and is the number of input-output pairs grids return from the `generate_single_task()` function. We always return a dict with two keys: `pairs` and `transformations`. `pairs` is the list of input-output pairs, and  `transformations` is the sampled transformations for this set of pairs. 

## Allowed Combinations, Allowed Transformations and Max Transformation Depth 

Either `allowed_combinations` OR `allowed_transformations` should be specified, not both. If `allowed_combinations` is specified, the generator will randomly sample from the provided list of possible combinations to sample from. If however `allowed_combinations` is set to `None`, and a list is provided for `allowed_transformations`, then the generator will randomly compose transformations, with a depth comprised between `min_transform_depth` and `max_transform_depth` (which must be provided as ints). 

Note: the `min_transformation_depth` and `max_transformation_depth` must also be considered in relation to the grid size and number of objects. 

All transformations (either within `allowed_combinations` or `allowed_transformations`) must be defined in the `transforms.py`. 

## Shape Compulsory Conditionals. 

List of "conditionals" that the shapes must satisfy. Shape constraints would probably have been a better name. Can be empty if all shapes can be sampled from the shape list. 
These must be as defined in `conditionals.py`. 

## Summary of Config Constraints. 

Must be passed to the generator as a dict, similarly to demonstrated in the demo.ipynb. 
The validity of the config is verified entirely in the `Config` class, as implemented in the `config_validation.py` script. 

* `min_n_shapes_per_grid` should be `int` >=1 
* `max_n_shapes_per_grid` should be `int` >= 1 and >= `min_n_shapes_per_grid`

* `min_n_transformations` should be `int` >=1 
* `max_n_transformations` should be `int` >= 1 and >= `min_n_transformations_per_grid`

* `min_grid_size` should be `int` >= 1
* `max_grid_size` should be `int` >= 1 and >= `min_grid_size`
  
* `n_examples` should be `int` and >= 1
  
* `allowed_combinations` should be `list of list` OR `None`. 
* `allowed_transformations` should be `list` or `None`.
  * If `allowed_transformations` is provided, the user must also set 
    * `min_transformation_depth` as `int` >= 1
    * `max_transformation_depth` as `int` >= 1 and >= `max_transformation_depth` 
  * Elif `allowed_transformations` is None - `min_transformations_depth` and `max_transformations` should also be set to None. 
  
* `shape_compulsory_conditionals` should be `list`. Could be empty list if no constraints are required. 

# To-Do 

- [ ] Update README
- [ ] Find sustainable way to not push the big Shapes file onto GH. 
- [x] Extend the experiments 
- [x] STILL TO CHECK: Check for duplicate bug
- [x] Fix bug with round numbers for Klim
- [ ] Add Metadata to all of the subfolders and upload to the README.
- [x] Debug Rotation and some other transformations which seem to be buggy. Especially on large objects. Check the paper_plot.ipynb (Plot Tasks section) to see what I'm talking about 
- [x] Debug Shape Emptying
- [x] Add transformations.
- [x] Change the names to match Klim's and generate in distribution test set. 
- [x] Upload to HF datasets.
- [x] Complete demo.ipynb
- [x] Fix code to not allow shapes to be neighbours! 
- [ ] Improve Figures with new transformations
- [ ] Add function (and experimental setting) where we vary the background of the grid. 
- [ ] Fix Bug for the image dimensions of the .parquet files 
- [ ] The compatible vs non compatible shape constraints in the beginning of `generate_single_task` function
- [ ] Verify that `shape_compulsory_conditionals`, `allowed_combinations` and `allowed_transformations` are within what the relevant files allow for.  
- [ ] Write a demo file to show how to play with the shapes  -->