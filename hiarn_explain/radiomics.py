import logging
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PARAMS = {
    'imageType': {'Original': {}},
    'featureClass': {'firstorder': [], 'glcm': [], 'glszm': [], 'shape2D': []},
    'setting': {
        'binWidth': 25, 'normalize': True, 'normalizeScale': 100,
        'removeOutliers': 3.0, 'force2D': True, 'force2Ddimension': 0,
    },
}
CODE_MAP = {'gl': 0, 'me': 1, 'pi': 2}
NO_TUMOR_LABEL = 3
NO_TUMOR_LIMIT = 1000
CLASS_NAMES = {0: 'glioma', 1: 'meningioma', 2: 'pituitary', 3: 'no_tumor'}


def make_extractor():
    from radiomics import featureextractor
    logging.getLogger('radiomics').setLevel(logging.ERROR)
    warnings.filterwarnings('ignore')
    return featureextractor.RadiomicsFeatureExtractor(**PARAMS)


def pseudo_mask(img_arr):
    h, w = img_arr.shape[:2]
    msk = np.zeros((h, w), dtype=np.int32)
    msk[int(h * .4):int(h * .6), int(w * .4):int(w * .6)] = 1
    return msk


def extract_features(extractor, img_arr, mask_arr):
    import SimpleITK as sitk
    img_sitk = sitk.GetImageFromArray(img_arr.astype(np.float32))
    mask_sitk = sitk.GetImageFromArray(mask_arr.astype(np.int32))
    result = extractor.execute(img_sitk, mask_sitk)
    return {k: float(v) for k, v in result.items() if k.startswith('original_') and not isinstance(v, str)}


def parse_code(name):
    m = re.search(r'_(gl|me|pi)_', name, re.IGNORECASE)
    if m:
        return CODE_MAP.get(m.group(1).lower())
    return None


def find_brisc_root(base):
    candidates = [Path(base, 'brisc2025'), Path(base), Path(base, 'brisc2025', 'brisc2025')]
    for c in candidates:
        if (c / 'segmentation_task').exists():
            return c
    raise FileNotFoundError(f'segmentation_task not found under {base}')


def collect_brisc_tasks(root, no_tumor_limit=NO_TUMOR_LIMIT):
    tasks, n_skipped = [], 0
    for split in ('train', 'test'):
        img_dir = root / 'segmentation_task' / split / 'images'
        mask_dir = root / 'segmentation_task' / split / 'masks'
        if not img_dir.exists():
            continue
        for ip in sorted(img_dir.glob('*.jpg')):
            label = parse_code(ip.name)
            if label is None:
                n_skipped += 1
                continue
            mp = mask_dir / (ip.stem + '.png')
            if mp.exists():
                tasks.append((str(ip), str(mp), label, ip.name))
    n_no_tumor = 0
    for split in ('train', 'test'):
        nt_dir = root / 'classification_task' / split / 'no_tumor'
        if nt_dir.exists():
            for ip in sorted(nt_dir.glob('*.jpg')):
                if n_no_tumor >= no_tumor_limit:
                    break
                tasks.append((str(ip), None, NO_TUMOR_LABEL, ip.name))
                n_no_tumor += 1
    return tasks, n_skipped


def region_mask(img, mask_path):
    import cv2
    if mask_path is None:
        return pseudo_mask(img)
    msk_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, msk = cv2.threshold(msk_raw, 127, 1, cv2.THRESH_BINARY)
    if msk.sum() == 0:
        msk = pseudo_mask(img).astype(np.uint8)
    return msk.astype(np.int32)


def extract_brisc(root, progress=None):
    import cv2
    extractor = make_extractor()
    tasks, n_skipped = collect_brisc_tasks(root)
    rows, errors = [], []
    iterator = progress(tasks) if progress else tasks
    for img_path, mask_path, label, filename in iterator:
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError('imread returned None')
            feats = extract_features(extractor, img, region_mask(img, mask_path))
            feats['label'] = label
            feats['filename'] = filename
            rows.append(feats)
        except Exception as e:
            errors.append((filename, str(e)[:80]))
    df = pd.DataFrame(rows).fillna(0.0)
    summary = {
        'n_tasks': len(tasks),
        'n_skipped_names': n_skipped,
        'n_errors': len(errors),
        'class_counts': dict(Counter(int(x) for x in df['label'])),
        'n_features': len([c for c in df.columns if c.startswith('original_')]),
    }
    return df, summary, errors
