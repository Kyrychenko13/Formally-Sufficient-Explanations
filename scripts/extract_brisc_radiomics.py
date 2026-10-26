import argparse
import json
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from hiarn_explain.radiomics import extract_brisc, find_brisc_root


def main():
    parser = argparse.ArgumentParser(description='Extract PyRadiomics features from BRISC 2025.')
    parser.add_argument('--brisc', required=True, help='folder with the unpacked BRISC 2025 dataset')
    parser.add_argument('--out', default=os.path.join('data', 'brisc_features.csv'))
    args = parser.parse_args()
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    root = find_brisc_root(args.brisc)
    df, summary, errors = extract_brisc(root, progress=tqdm)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(json.dumps(summary, indent=1))
    for filename, message in errors[:10]:
        print('error:', filename, message)
    print('saved', args.out)


if __name__ == '__main__':
    main()
