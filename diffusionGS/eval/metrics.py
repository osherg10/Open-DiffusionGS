import json
import os
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from PIL import Image

try:  # torchmetrics is an optional dependency for evaluation
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
except ImportError as exc:  # pragma: no cover - guarded import for optional dependency
    raise ImportError(
        "torchmetrics is required for FID/KID computation. Please install it with `pip install torchmetrics`."
    ) from exc


def _list_image_files(directory: str) -> List[str]:
    supported_ext = {"png", "jpg", "jpeg", "bmp", "tiff"}
    files = [
        os.path.join(directory, name)
        for name in sorted(os.listdir(directory))
        if os.path.isfile(os.path.join(directory, name))
        and name.split(".")[-1].lower() in supported_ext
    ]
    if not files:
        raise FileNotFoundError(f"No image files found in {directory}.")
    return files


def _load_image(path: str, image_size: Optional[int]) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if image_size is not None:
        image = image.resize((image_size, image_size), Image.BICUBIC)
    array = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor


def _batched(iterable: List[str], batch_size: int) -> Iterable[List[str]]:
    for idx in range(0, len(iterable), batch_size):
        yield iterable[idx : idx + batch_size]


def compute_fid_kid(
    reference_dir: str,
    generated_dir: str,
    *,
    batch_size: int = 8,
    device: Optional[str] = None,
    image_size: int = 299,
    kid_subset_size: int = 100,
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute FID and KID scores between reference and generated image folders.

    Args:
        reference_dir: Directory containing reference images.
        generated_dir: Directory containing generated images.
        batch_size: Number of images to process per batch.
        device: Torch device string (e.g., "cuda" or "cpu"). Defaults to CUDA if available.
        image_size: Images are resized to this square resolution before evaluation.
        kid_subset_size: Number of samples per subset for KID estimation.
        save_path: Optional path to persist metrics as JSON.

    Returns:
        A dictionary with keys ``fid``, ``kid_mean``, and ``kid_std``.
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    real_paths = _list_image_files(reference_dir)
    fake_paths = _list_image_files(generated_dir)

    fid = FrechetInceptionDistance(normalize=True).to(device)
    kid = KernelInceptionDistance(subset_size=kid_subset_size, normalize=True).to(device)

    def update_metrics(paths: List[str], is_real: bool) -> None:
        for batch_paths in _batched(paths, batch_size):
            images = [_load_image(path, image_size) for path in batch_paths]
            batch = torch.stack(images, dim=0).to(device)
            fid.update(batch, real=is_real)
            kid.update(batch, real=is_real)

    update_metrics(real_paths, is_real=True)
    update_metrics(fake_paths, is_real=False)

    fid_score = float(fid.compute().item())
    kid_mean, kid_std = kid.compute()
    result = {
        "fid": fid_score,
        "kid_mean": float(kid_mean.item()),
        "kid_std": float(kid_std.item()),
    }

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return result


__all__ = ["compute_fid_kid"]
