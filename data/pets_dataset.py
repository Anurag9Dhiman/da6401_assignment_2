"""Dataset skeleton for Oxford-IIIT Pet.
"""

from __future__ import annotations
import tarfile
from torch.utils.data import Dataset
import urllib.request
import ssl

# Bypass SSL verification for macOS / restricted environments
ssl._create_default_https_context = ssl._create_unverified_context
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from packaging.version import Version as _V

_ALBU_V2 = _V(A.__version__) >= _V("2.0.0")   # size= API for crops
_ALBU_NEW_API = _V(A.__version__) >= _V("1.4.0")  # new CoarseDropout API

# Helpers downloading step

_IMAGES_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
_ANNOTS_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

def _is_kaggle() -> bool:
    return Path("/kaggle/input").exists()


# Candidate Kaggle dataset paths (checked in order)
_KAGGLE_CANDIDATES = [
    "/kaggle/input/the-oxfordiiit-pet-dataset",   # tanlikesmath dataset
    "/kaggle/input/oxford-iiit-pet",
    "/kaggle/input/oxfordiiit-pet-dataset",
]


def _resolve_root(root: str) -> Path:
    """Return correct data root; auto-detects Kaggle dataset paths."""
    p = Path(root)
    if p.exists():
        return p
    if _is_kaggle():
        for candidate in _KAGGLE_CANDIDATES:
            cp = Path(candidate)
            if cp.exists() and (cp / "images").exists():
                print(f"[dataset] Auto-detected Kaggle path: {cp}")
                return cp
    return p


def maybe_download(root: str = "data/pets"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for url in (_IMAGES_URL, _ANNOTS_URL):
        fname = root / url.split("/")[-1]
        folder = root / ("images" if "images" in url else "annotations")
        if not folder.exists():
            if not fname.exists():
                print(f"Downloading {url} ...")
                urllib.request.urlretrieve(url, fname)
            print(f"Extracting {fname} ...")
            with tarfile.open(fname) as tf:
                tf.extractall(root)


# ImageNet normalisation constants
_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)


def _coarse_dropout(p: float = 0.2) -> A.BasicTransform:
    """Return CoarseDropout compatible with albumentations 1.x and 2.x."""
    if _ALBU_NEW_API:
        return A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(16, 32),
            hole_width_range=(16, 32),
            fill=0, p=p,
        )
    return A.CoarseDropout(
        max_holes=4, max_height=32, max_width=32,
        min_holes=1, min_height=16, min_width=16,
        fill_value=0, p=p,
    )


def _random_resized_crop(image_size: int) -> A.BasicTransform:
    """Return RandomResizedCrop compatible with albumentations 1.x and 2.x."""
    kwargs = dict(scale=(0.7, 1.0), ratio=(0.75, 1.33), p=1.0)
    if _ALBU_V2:
        return A.RandomResizedCrop(size=(image_size, image_size), **kwargs)
    return A.RandomResizedCrop(height=image_size, width=image_size, **kwargs)


def get_transform(split: str, image_size: int = 224) -> A.Compose:
    if split == "train":
        return A.Compose(
            [
                _random_resized_crop(image_size),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.6
                ),
                A.Rotate(limit=15, p=0.4),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                _coarse_dropout(p=0.2),
                A.Normalize(mean=_MEAN, std=_STD),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["bbox_labels"]),
        )
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["bbox_labels"]),
    )


# The dataset class

class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader.

    Each sample returns a dict:
        'image': (3, H, W) float tensor - normalised
        'label' : int in [0, 36] - breed index
        'bbox'  : (4,) float tensor - [x_c, y_c, w, h] in [0, 1]
        'mask'  : (H, W) long tensor - 0=fg, 1=bg, 2=boundary
    """

    def __init__(
        self,
        root: str = "data/pets",
        split: str = "train",
        transform: A.Compose | None = None,
        image_size: int = 224,
        val_fraction: float = 0.15,
        seed: int = 42,
    ):
        self.root = _resolve_root(root)
        self.split = split
        self.img_dir = self.root / "images"
        self.annot_dir = self.root / "annotations"
        self.mask_dir = self.annot_dir / "trimaps"
        self.transform = transform or get_transform(split, image_size)

        self.class_names, self.samples = self._parse_list(val_fraction, seed)
        self._bbox_cache: dict[str, list[float]] = {}
        self._load_bboxes()

    def _parse_list(self, val_fraction: float, seed: int):
        list_path = self.annot_dir / "list.txt"
        name_to_classid: dict[str, int] = {}
        classid_to_breed: dict[int, str] = {}

        with open(list_path) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                name = parts[0]
                class_id = int(parts[1]) - 1  # 0-indexed (parts[1] is CLASS-ID 1..37)
                breed = "_".join(name.split("_")[:-1])
                name_to_classid[name] = class_id
                classid_to_breed[class_id] = breed

        num_classes = max(classid_to_breed.keys()) + 1
        class_names = [classid_to_breed.get(i, f"class_{i}") for i in range(num_classes)]

        # Use official split files
        trainval_path = self.annot_dir / "trainval.txt"
        test_path     = self.annot_dir / "test.txt"

        def _read_split_file(path):
            names = []
            with open(path) as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split()
                    if parts:
                        names.append(parts[0])
            return names

        trainval_names = _read_split_file(trainval_path)
        test_names     = _read_split_file(test_path)

        trainval = [(n, name_to_classid[n], 1) for n in trainval_names if n in name_to_classid]
        test_set = [(n, name_to_classid[n], 2) for n in test_names if n in name_to_classid]

        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(trainval))
        n_val = int(len(trainval) * val_fraction)

        if self.split == "train":
            samples = [trainval[i] for i in idx[n_val:]]
        elif self.split == "val":
            samples = [trainval[i] for i in idx[:n_val]]
        else:
            samples = test_set

        return class_names, samples

    def _load_bboxes(self):
        xml_dir = self.annot_dir / "xmls"
        if not xml_dir.exists():
            return
        for name, _, _ in self.samples:
            xml_path = xml_dir / f"{name}.xml"
            if not xml_path.exists():
                continue
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find("size")
            W, H = int(size.find("width").text), int(size.find("height").text)
            obj = root.find("object")
            if obj is None:
                continue
            bb = obj.find("bndbox")
            xmin = float(bb.find("xmin").text) / W
            ymin = float(bb.find("ymin").text) / H
            xmax = float(bb.find("xmax").text) / W
            ymax = float(bb.find("ymax").text) / H
            self._bbox_cache[name] = [
                (xmin + xmax) / 2,
                (ymin + ymax) / 2,
                xmax - xmin,
                ymax - ymin,
            ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        name, label, _ = self.samples[idx]

        image = np.array(Image.open(self.img_dir / f"{name}.jpg").convert("RGB"))
        mask_path = self.mask_dir / f"{name}.png"
        mask = np.array(Image.open(mask_path)) if mask_path.exists() \
                else np.zeros(image.shape[:2], dtype=np.uint8)

        bbox_raw = self._bbox_cache.get(name, [0.5, 0.5, 1.0, 1.0])

        out = self.transform(
            image=image,
            mask=mask,
            bboxes=[bbox_raw],
            bbox_labels=[label],
        )

        bbox_t = torch.tensor(
            out["bboxes"][0] if out["bboxes"] else bbox_raw,
            dtype=torch.float32,
        ).clamp(0.0, 1.0)

        # Oxford trimaps: 1=fg, 2=bg, 3=boundary → subtract 1 → 0,1,2
        # Clamp to [-1, 2]: -1 is ignored by CE loss; anything outside [1,3]
        # in the raw mask (e.g. 0 from padding) maps to -1 (ignored).
        mask_t = torch.as_tensor(out["mask"], dtype=torch.long) - 1
        mask_t = mask_t.clamp(-1, 2)

        return {
            "image": out["image"],
            "label": label,
            "bbox": bbox_t,
            "mask": mask_t,
        }


def get_dataloader(
    root: str = "data/pets",
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 2,
    val_fraction: float = 0.15,
    seed: int = 42,
    persistent_workers: bool = True,
) -> tuple[dict, list[str]]:
    datasets = {
        split: OxfordIIITPetDataset(root, split, image_size=image_size,
                                    val_fraction=val_fraction, seed=seed)
        for split in ("train", "val", "test")
    }
    # persistent_workers only makes sense when num_workers > 0
    _persistent = persistent_workers and num_workers > 0
    loaders = {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
            persistent_workers=_persistent,
        )
        for split, ds in datasets.items()
    }
    return loaders, datasets["train"].class_names
