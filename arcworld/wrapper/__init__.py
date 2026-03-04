import logging

if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

from .dataset import CogitaoDataset
from .generator import ParallelGenerator

__all__ = [
    "CogitaoDataset",
    "ParallelGenerator",
]
