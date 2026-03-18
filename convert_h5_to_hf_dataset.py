"""

"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import h5py
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def h5_to_parquet(h5_path: Path, parquet_path: Path) -> None:
    with h5py.File(h5_path, "r") as f:
        n = f["inputs"].shape[0]
        inputs = f["inputs"][:]
        outputs = f["outputs"][:]
        grid_sizes = f["grid_sizes"][:]
        transformation_suites = f["transformation_suites"][:]

    rows: list[dict] = []
    for i in range(n):
        h, w = int(grid_sizes[i][0]), int(grid_sizes[i][1])
        suite = transformation_suites[i]
        if isinstance(suite, bytes):
            suite = suite.decode("utf-8")
        rows.append(
            {
                "input": inputs[i, :h, :w].tolist(),
                "output": outputs[i, :h, :w].tolist(),
                "transformation_suite": json.loads(suite),
            }
        )

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    print(f"Converted {h5_path} -> {parquet_path} ({n} rows)")


def convert_folder_to_parquet(data_dir: Path, parquet_dir: Path) -> None:
    for h5_file in sorted(data_dir.glob("*.h5")):
        h5_to_parquet(h5_file, parquet_dir / h5_file.with_suffix(".parquet").name)


def upload_dataset_to_hf(parquet_dir: Path, repo_id: str, path_in_repo: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.upload_folder(
        folder_path=str(parquet_dir),
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        repo_type="dataset",
        commit_message="Upload compgen dataset (ID and OOD splits)",
        ignore_patterns=["*.db"],
    )


def main(data_folder: str, hf_repo_id: str, upload: bool) -> None:

    # Local paths
    suffix = "_" + data_folder.split("_")[-1] if len(data_folder.split("_")) > 3 else ""
    data_dir_path = Path(f"./datasets/{data_folder}")
    parquet_dir_path = Path(f"./datasets_parquet/{data_folder}")
    
    settings = [1, 2, 3, 4]
    experiments = [1]

    for setting in settings:
        for experiment in experiments:
            logger.info(f"Processing Setting {setting}, Experiment {experiment}...")
            data_dir_path_setting_exp = data_dir_path / f"exp_setting_{setting}" / f"experiment_{experiment}"
            parquet_dir_path_setting_exp = parquet_dir_path / f"exp_setting_{setting}" / f"experiment_{experiment}"
            convert_folder_to_parquet(data_dir_path_setting_exp, parquet_dir_path_setting_exp)

            if upload:
                hf_path_in_repo = f"compgen/exp_setting_{setting}/experiment_{experiment}{suffix}"

                upload_dataset_to_hf(
                    parquet_dir_path_setting_exp,
                    hf_repo_id,
                    hf_path_in_repo,
                )

def get_cli_args():
    argparser = argparse.ArgumentParser(description="Convert H5 datasets to Parquet and optionally upload to Hugging Face")
    argparser.add_argument("--data_folder", type=str, required=True, help="Name of the data folder (e.g., 'compgen_ktroyan', 'compgen_basics_ktroyan', etc.)")
    argparser.add_argument("--hf_repo_id", type=str, required=True, help="Hugging Face repository ID to upload to (e.g.: 'ktroyan/COGITAO')")
    argparser.add_argument("--upload", action="store_true", help="Whether to upload the converted dataset to Hugging Face")
    return argparser.parse_args()

if __name__ == "__main__":
    args = get_cli_args()

    main(args.data_folder, args.hf_repo_id, args.upload)

