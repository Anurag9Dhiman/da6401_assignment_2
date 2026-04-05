"""Dataset skeleton for Oxford-IIIT Pet.
"""

from __future__ import annotations
import tarfile
from torch.utils.data import Dataset
import urllib.request
import ssl

# Bypass SSL verification for macOS
ssl._create_default_https_context = ssl._create_unverified_context
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Helpers downloading step

_IMAGES_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
_ANNOTS_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

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

# To transform image data
def get_transform(split: str, image_size: int = 224) -> A.Compose:
    if split == "train":
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["bbox_labels"]),
        )
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["bbox_labels"]),
    )

# The dataset class

class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader skeleton."""

    """ Here, each sample will return a dict with corresponding keys:
        'image': (3, H, W) float tensor - normalised
        'label' : int in [0, 36] - breed index
        'bbox' : (4, ) float tensor - [x_c, y_c, w, h] in [0,1]
        'mask' : (H, W) long tensor - 0=fg, 1=bg, 2= boundary
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
        self.root = Path(root)
        self.split = split
        self.img_dir = self.root / "images"
        self.annot_dir = self.root / "annotations"
        self.mask_dir = self.annot_dir / "trimaps"
        self.transform = transform or get_transform(split, image_size)

        self.class_names, self.samples = self._parse_list(val_fraction, seed)
        self._bbox_cache: dict[str, list[float]] = {}
        self._load_bboxes()

    def _parse_list(self, val_fraction: float, seed: int):
        # here, we will parse annotation/list.txt first and then split into train/val/test.
        list_path = self.annot_dir / "list.txt"
        seen: dict[str, int] = {}
        class_names: list[str] = []
        all_samples = []

        with open(list_path) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                name, split_id = parts[0], int(parts[3])
                breeds = "_".join(name.split("_")[:-1])
                if breeds not in seen:
                    seen[breeds] = len(seen)
                    class_names.append(breeds)
                all_samples.append((name, seen[breeds], split_id))

        trainval = [s for s in all_samples if s[2] == 1]
        test_set = [s for s in all_samples if s[2] == 2]

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
        """To load the bound boxes from file annotations/xmls/."""
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

        bbox_t = torch.tensor(out["bboxes"][0] if out["bboxes"] else bbox_raw,
                              dtype=torch.float32)
        mask_t = torch.as_tensor(out["mask"], dtype=torch.long) - 1  # indexing is 0 indexed

        return {
            "image": out["image"],
            "label": label,
            "bbox": bbox_t,
            "mask": mask_t,
        }


# settings for DataLoader
def get_dataloader(
    root: str = "data/pets",
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[dict, list[str]]:
    datasets = {
        split: OxfordIIITPetDataset(root, split, image_size=image_size,
                                    val_fraction=val_fraction, seed=seed)
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )
        for split, ds in datasets.items()
    }
    return loaders, datasets["train"].class_names
