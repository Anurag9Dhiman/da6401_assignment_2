"""§2.7 Final Pipeline Showcase — run inference on novel pet images.

Usage:
    python showcase.py --images pet1.jpg pet2.jpg pet3.jpg
    python showcase.py --images pet1.jpg pet2.jpg pet3.jpg --wandb_project da6401-assignment2
"""

import argparse
import numpy as np
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.multitask import MultiTaskPerceptionModel
from data.pets_dataset import get_dataloader


_TRANSFORM = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def run(args):
    _, class_names = get_dataloader('data/pets', 1, 224, 0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiTaskPerceptionModel(num_breeds=37, seg_classes=3)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()

    descriptions = args.descriptions if args.descriptions else [''] * len(args.images)

    # collect results
    rows = []
    for path, desc in zip(args.images, descriptions):
        img = np.array(Image.open(path).convert('RGB'))
        tensor = _TRANSFORM(image=img)['image'].unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(tensor)

        probs = torch.softmax(out['classification'], dim=1)[0]
        top1_idx = probs.argmax().item()
        top1_conf = probs[top1_idx].item()
        bbox = out['localization'][0].tolist()
        collapsed = bbox[2] > 0.9 and bbox[3] > 0.9
        bbox_str = f'Degenerate [{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]' if collapsed else str([round(b,3) for b in bbox])

        rows.append({
            'image': path,
            'description': desc,
            'top_pred': class_names[top1_idx],
            'confidence': f'{top1_conf*100:.1f}%',
            'bbox': bbox_str,
        })

    # print table
    col_w = [10, 22, 22, 12, 26]
    headers = ['Image', 'Description', 'Top Prediction', 'Confidence', 'Bbox']
    sep = '+' + '+'.join('-' * (w + 2) for w in col_w) + '+'

    print('\n' + sep)
    print('| ' + ' | '.join(f'{h:<{w}}' for h, w in zip(headers, col_w)) + ' |')
    print(sep)
    for r in rows:
        vals = [r['image'], r['description'], r['top_pred'], r['confidence'], r['bbox']]
        print('| ' + ' | '.join(f'{v:<{w}}' for v, w in zip(vals, col_w)) + ' |')
    print(sep + '\n')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--images', nargs='+', required=True, help='Paths to input images')
    p.add_argument('--descriptions', nargs='+', default=None, help='Description for each image')
    p.add_argument('--checkpoint', default='checkpoints/multitask.pth')
    p.add_argument('--wandb_project', default=None)
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
