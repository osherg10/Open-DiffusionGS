import math
import os
import json
import re
import shutil
import logging
from pathlib import Path
import cv2
from dataclasses import dataclass, field

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusionGS import register
from diffusionGS.utils.typing import *
from diffusionGS.utils.config import parse_structured
import requests

from .base_scene import BaseDataModuleConfig, BaseDataset


@dataclass
class Re10KDataModuleConfig(BaseDataModuleConfig):
    pass

class Re10KDataset(BaseDataset):
    pass


@register("Re10k-datamodule")
class Re10KDataModule(pl.LightningDataModule):
    cfg: Re10KDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(Re10KDataModuleConfig, cfg)
        self.repo_root = Path(__file__).resolve().parents[2]

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = Re10KDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = Re10KDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = Re10KDataset(self.cfg, "test")

    def prepare_data(self):
        if not self.cfg.auto_download:
            return

        updated_paths = {}
        for attr in ["local_dir", "local_eval_dir", "view_idx_file_path"]:
            current_path = getattr(self.cfg, attr)
            resolved_path = self._ensure_file_available(current_path)
            if resolved_path:
                updated_paths[attr] = resolved_path

        for attr, resolved in updated_paths.items():
            setattr(self.cfg, attr, resolved)

    def _ensure_file_available(self, path_str: str) -> str:
        if not path_str:
            return path_str

        target_path = Path(path_str)
        if target_path.exists():
            return str(target_path)

        mount_root = Path(self.cfg.colab_mount_path or "/content")
        mount_root.mkdir(parents=True, exist_ok=True)

        if target_path.is_absolute() and str(target_path).startswith(str(mount_root)):
            destination = target_path
        else:
            destination = mount_root / target_path.name

        source_candidates = []
        source_candidates.append(Path(path_str))
        if not target_path.is_absolute():
            source_candidates.append(self.repo_root / target_path)
        else:
            source_candidates.append(self.repo_root / target_path.name)

        for source in source_candidates:
            if source.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(source, destination)
                logging.info(f"Copied {source} to {destination}")
                return str(destination)

        if path_str.startswith("http://") or path_str.startswith("https://"):
            try:
                response = requests.get(path_str, timeout=30)
                response.raise_for_status()
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(response.content)
                logging.info(f"Downloaded {path_str} to {destination}")
                return str(destination)
            except requests.RequestException as exc:
                logging.warning(f"Failed to download {path_str}: {exc}")

        logging.warning(f"Unable to find or download file for path: {path_str}")
        return path_str

    def general_loader(self, dataset, batch_size, collate_fn=None, num_workers=0) -> DataLoader:
        return DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=self.train_dataset.collate,
            num_workers=self.cfg.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(self.val_dataset, batch_size=self.cfg.eval_batch_size, num_workers=self.cfg.num_workers_val)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=self.cfg.eval_batch_size, num_workers=self.cfg.num_workers_val)

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=self.cfg.eval_batch_size, num_workers=self.cfg.num_workers_val)