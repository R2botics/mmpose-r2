"""Convert COCO keypoint annotations to YOLO pose format."""
import json
import os
import shutil
from pathlib import Path

SRC_IMAGES   = Path('/home/siddharth/Deps/mmpose/data/RSC_Keypoints/images')
ANN_DIR      = Path('/home/siddharth/Deps/mmpose/data/RSC_Keypoints/annotations')
OUT_ROOT     = Path('/home/siddharth/Deps/mmpose/yolo_dataset')

NUM_KEYPOINTS = 8


def convert_split(split: str):
    ann_file = ANN_DIR / f'person_keypoints_{split}.json'
    data = json.load(open(ann_file))

    img_by_id = {img['id']: img for img in data['images']}
    img_dir = OUT_ROOT / 'images' / split
    lbl_dir = OUT_ROOT / 'labels' / split

    count = 0
    for ann in data['annotations']:
        img = img_by_id[ann['image_id']]
        w, h = img['width'], img['height']

        # Use full-image bbox (which we've been doing all along)
        cx, cy, bw, bh = 0.5, 0.5, 1.0, 1.0

        # Keypoints normalized to [0, 1]
        kps = ann['keypoints']
        kp_strs = []
        for i in range(NUM_KEYPOINTS):
            x, y, v = kps[i*3], kps[i*3+1], kps[i*3+2]
            # YOLO normalizes by image dims, clamps to [0, 1] for visible, 0 for invisible
            if v == 0:
                kp_strs.extend([f'0', f'0', f'0'])
            else:
                nx = max(0.0, min(1.0, x / w))
                ny = max(0.0, min(1.0, y / h))
                kp_strs.extend([f'{nx:.6f}', f'{ny:.6f}', f'{v}'])

        line = f'0 {cx} {cy} {bw} {bh} ' + ' '.join(kp_strs)

        # Write label
        stem = Path(img['file_name']).stem
        (lbl_dir / f'{stem}.txt').write_text(line + '\n')

        # Copy image
        src = SRC_IMAGES / img['file_name']
        dst = img_dir / img['file_name']
        if not dst.exists():
            shutil.copy(src, dst)
        count += 1

    print(f'  {split}: {count} annotations')


print('Converting splits:')
convert_split('train')
convert_split('val')
print('Done.')
