import argparse
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from hiarn_explain.data import load_tabular_datasets, tabular_splits
from hiarn_explain.experiments import run_fold_tabular, summarize_tabular
from hiarn_explain.model import EPOCHS

ARCHS = ('H2', 'H3')
SMOKE = {'epochs': 20, 'n_compare': 5, 'n_repeats': 1}
FULL = {'epochs': EPOCHS, 'n_compare': 100, 'n_repeats': 3}


def run_job(job):
    key, args, settings = job
    arch, ds, X_tr, y_tr, X_te, y_te, seed, compare = args
    out = run_fold_tabular(arch, ds, X_tr, y_tr, X_te, y_te, seed, compare,
                           epochs=settings['epochs'], n_compare=settings['n_compare'])
    return key, out


def partial_path(folder, key):
    ds, arch, i = key
    return os.path.join(folder, f'{ds}_{arch}_{i}.json')


def main():
    parser = argparse.ArgumentParser(description='Explanations of HiARN-2 and HiARN-3 on the tabular datasets.')
    parser.add_argument('--out', default='tabular_out')
    parser.add_argument('--processes', type=int, default=min(10, os.cpu_count() or 1))
    parser.add_argument('--smoke', action='store_true', help='short run for testing the pipeline')
    args = parser.parse_args()
    settings = SMOKE if args.smoke else FULL
    partial = os.path.join(args.out, 'partial_smoke' if args.smoke else 'partial')
    os.makedirs(partial, exist_ok=True)
    t0 = time.time()
    data = load_tabular_datasets()
    jobs = {}
    n_folds = 5 * settings['n_repeats']
    for ds, (X, y) in data.items():
        for i, fold in enumerate(tabular_splits(X, y, True, n_repeats=settings['n_repeats'])):
            for arch in ARCHS:
                jobs[(ds, arch, i)] = (arch, ds, *fold, 1000 + i, i == 0)
    outs = {}
    for key in jobs:
        path = partial_path(partial, key)
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                outs[key] = json.load(f)
    todo = sorted((k for k in jobs if k not in outs), key=lambda k: (k[2] != 0, k))
    print(f'folds: {len(jobs)}, done: {len(outs)}, to run: {len(todo)}, processes: {args.processes}', flush=True)
    work = [(k, jobs[k], settings) for k in todo]
    if args.processes > 1:
        with mp.Pool(args.processes) as pool:
            results = pool.imap_unordered(run_job, work)
            for key, out in results:
                outs[key] = out
                with open(partial_path(partial, key), 'w', encoding='utf-8') as f:
                    json.dump(out, f)
                print(f'  {key}: n={out["n"]}, violations {out["ours_viol"]}', flush=True)
    else:
        for job in work:
            key, out = run_job(job)
            outs[key] = out
            with open(partial_path(partial, key), 'w', encoding='utf-8') as f:
                json.dump(out, f)
            print(f'  {key}: n={out["n"]}, violations {out["ours_viol"]}', flush=True)
    res = {'status': 'reproduction', 'smoke': args.smoke}
    res.update(summarize_tabular(outs, list(data), n_folds, ARCHS))
    res['runtime_s'] = round(time.time() - t0, 1)
    res['n_proc'] = args.processes
    path = os.path.join(args.out, 'tabular_smoke.json' if args.smoke else 'tabular_results.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print('verdict:', res['verdict'])
    print('saved', path)


if __name__ == '__main__':
    main()
