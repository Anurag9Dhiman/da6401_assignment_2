# DA6401 Assignment 2 — Visual Perception Pipeline on Oxford-IIIT Pet

## Links
- **GitHub Repo:** https://github.com/Anurag9Dhiman/da6401_assignment_2
- **W&B Report:** https://api.wandb.ai/links/anuragdhiman666-indian-institute-of-technology-madras/u3mxoxv7

## Tasks
1. **Classification** — VGG11-based breed classifier (37 classes)
2. **Localization** — Bounding box regression using IoU loss
3. **Segmentation** — U-Net style encoder-decoder for trimap segmentation
4. **Multi-task** — Unified model with shared VGG11 backbone for all 3 tasks

## Setup
```bash
pip install -r requirements.txt
```

## Training
```bash
python train.py --task cls
python train.py --task loc
python train.py --task seg
python train.py --task multi
```

## Inference
```bash
python inference.py --image <path_to_image> --checkpoint checkpoints/multitask.pth
```
