from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class VesselSegmentationDataset(Dataset):
    """Folder dataset for binary vessel segmentation.

    Expected layout:

    data_root/
      train_images/
      train_masks/
      val_images/
      val_masks/
      test_images/
      test_masks/

    You can also pass direct image and mask directories.
    """

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        image_size: int = 512,
        augment: bool = False,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.images = sorted([p for p in self.image_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
        if not self.images:
            raise FileNotFoundError(f"No images found in {self.image_dir}")
        self.masks = [self._match_mask(p) for p in self.images]
        self.image_tf = T.Compose(
            [
                T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BILINEAR),
                T.ToTensor(),
            ]
        )
        self.mask_tf = T.Compose(
            [
                T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.NEAREST),
                T.ToTensor(),
            ]
        )

    @classmethod
    def from_root(cls, root: str | Path, split: str, image_size: int = 512, augment: bool = False):
        root = Path(root)
        aliases = {
            "train": ("train_images", "train_masks"),
            "val": ("val_images", "val_masks"),
            "valid": ("val_images", "val_masks"),
            "test": ("test_images", "test_masks"),
            "true_test": ("true_test_images", "true_test_masks"),
        }
        if split not in aliases:
            raise ValueError(f"Unknown split {split!r}; expected one of {sorted(aliases)}")
        img_name, mask_name = aliases[split]
        return cls(root / img_name, root / mask_name, image_size=image_size, augment=augment)

    def _match_mask(self, image_path: Path) -> Path:
        direct = self.mask_dir / image_path.name
        if direct.exists():
            return direct
        stem_matches = [p for p in self.mask_dir.iterdir() if p.stem == image_path.stem and p.suffix.lower() in IMG_EXTS]
        if stem_matches:
            return sorted(stem_matches)[0]
        raise FileNotFoundError(f"No mask matching {image_path.name} in {self.mask_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = Image.open(self.images[index]).convert("L")
        mask = Image.open(self.masks[index]).convert("L")
        x = self.image_tf(image)
        y = (self.mask_tf(mask) > 0.5).float()
        if self.augment:
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=[-1])
                y = torch.flip(y, dims=[-1])
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=[-2])
                y = torch.flip(y, dims=[-2])
            gain = 1.0 + (torch.rand(()) - 0.5).item() * 0.2
            x = torch.clamp(x * gain, 0.0, 1.0)
        return x, y
