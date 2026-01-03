import math
import os
import json
import re
import cv2
from dataclasses import dataclass, field
from pathlib import Path
import zipfile
import urllib.request

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusionGS import register
from diffusionGS.utils.typing import *
from diffusionGS.utils.config import parse_structured

from .base import BaseDataModuleConfig, BaseDataset


@dataclass
class ObjaverseDataModuleConfig(BaseDataModuleConfig):
    auto_download: bool = False
    cache_dir: Optional[str] = None
    gdrive_folder_id: Optional[str] = None

class ObjaverseDataset(BaseDataset):
    pass


@register("Objaverse-datamodule")
class ObjaverseDataModule(pl.LightningDataModule):
    cfg: ObjaverseDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(ObjaverseDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = ObjaverseDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = ObjaverseDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = ObjaverseDataset(self.cfg, "test")

    def prepare_data(self):
        expected_local = self.cfg.local_dir
        expected_images = self.cfg.image_dir

        def _missing_path(path: Optional[str]):
            return (
                path is None
                or "DATAPATH" in str(path)
                or not os.path.exists(path)
            )

        local_missing = _missing_path(expected_local)
        image_missing = _missing_path(expected_images)
        if not (local_missing or image_missing):
            return

        if not self.cfg.auto_download:
            print(
                "G-Objaverse data not found and auto_download is disabled. "
                "Please download the dataset manually."
            )
            return

        in_colab = os.path.exists("/content")
        drive_mounted = os.path.exists("/content/drive")

        cache_root = (
            self.cfg.cache_dir
            or (
                "/content/drive/MyDrive/diffusiongs-cache" if drive_mounted else None
            )
            or ("/content/.cache/diffusiongs" if in_colab else None)
            or os.path.join(Path.home(), ".cache", "diffusiongs")
        )
        os.makedirs(cache_root, exist_ok=True)

        # Default locations inside cache when the user did not provide valid paths.
        if local_missing:
            expected_local = os.path.join(cache_root, "json")
            self.cfg.local_dir = expected_local
        if image_missing:
            expected_images = os.path.join(cache_root, "gobjaverse") + os.sep
            self.cfg.image_dir = expected_images

        os.makedirs(expected_local, exist_ok=True)
        os.makedirs(expected_images, exist_ok=True)

        def download_file(url: str, target_path: str):
            if os.path.exists(target_path):
                return target_path
            print(f"Downloading {url} -> {target_path}")
            with urllib.request.urlopen(url) as response, open(target_path, "wb") as out:
                out.write(response.read())
            return target_path

        def unzip_file(zip_path: str, target_dir: str):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(target_dir)

        # Prefer a shared Google Drive folder when explicitly provided.
        base_cache = Path(cache_root)
        json_zip = base_cache / "gobjaverse_json.zip"
        image_zip = base_cache / "gobjaverse_images.zip"

        if self.cfg.gdrive_folder_id:
            try:
                import gdown

                gdown.download_folder(
                    id=self.cfg.gdrive_folder_id,
                    output=str(base_cache),
                    quiet=False,
                )
            except Exception as exc:  # pragma: no cover - optional dependency
                print(
                    "Failed to download from Google Drive; falling back to Hugging Face.\n"
                    f"Reason: {exc}"
                )

        hf_base = "https://huggingface.co/datasets/CaiYuanhao/G-Objaverse/resolve/main"
        if not json_zip.exists():
            download_file(f"{hf_base}/json.zip", str(json_zip))
        if not image_zip.exists():
            download_file(f"{hf_base}/gobjaverse.zip", str(image_zip))

        if local_missing and not any(Path(expected_local).iterdir()):
            unzip_file(str(json_zip), expected_local)
        if image_missing and not any(Path(expected_images).iterdir()):
            unzip_file(str(image_zip), expected_images)

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
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)