import argparse
import json
import multiprocessing as mp
import os
import sys
import time

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, ROOT)
from hiarn_explain.data import brisc_group_splits, load_brisc
from hiarn_explain.experiments import (BRISC_GRID, N_ANCHOR_BRISC, N_COMPARE, brisc_controls, config_key,
                                       evaluate_config, run_fold_brisc, select_brisc, summarize_brisc)
from hiarn_explain.model import EPOCHS

SMOKE = {'epochs': 20, 'n_compare': 5, 'n_anchor': 2, 'confirmation_seeds': [2026],
         'grid': [{'model': 'H2', 'rules': 16, 'lam': 1e-3}, {'model': 'H3', 'rules': 16, 'lam': 1e-3}]}
FULL = {'epochs': EPOCHS, 'n_compare': N_COMPARE, 'n_anchor': N_ANCHOR_BRISC, 'confirmation_seeds': [2026, 2027, 2028],
        'grid': BRISC_GRID}
SELECTION_SEEDS = [42]


def grid_job(job):
    index, cfg, fold, seed, epochs = job
    return index, evaluate_config(cfg, *fold, seed, epochs=epochs)


def fold_job(job):
    arch, i, cfg, fold, seed, compare, settings = job
    out = run_fold_brisc(arch, cfg, *fold, seed, compare, epochs=settings['epochs'],
                         n_compare=settings['n_compare'], n_anchor=settings['n_anchor'])
    return (arch, i), out


def parallel(fn, jobs, processes):
    if processes > 1:
        with mp.Pool(processes) as pool:
            return list(pool.imap_unordered(fn, jobs))
    return [fn(j) for j in jobs]


def main():
    parser = argparse.ArgumentParser(description='Selection, explanations and comparison on BRISC radiomics.')
    parser.add_argument('--features', default=os.path.join(ROOT, 'data', 'brisc_features.csv'))
    parser.add_argument('--groups', default=os.path.join(ROOT, 'data', 'brisc_groups.csv'))
    parser.add_argument('--out', default='brisc_out')
    parser.add_argument('--processes', type=int, default=min(10, os.cpu_count() or 1))
    parser.add_argument('--smoke', action='store_true', help='short run for testing the pipeline')
    args = parser.parse_args()
    settings = SMOKE if args.smoke else FULL
    partial = os.path.join(args.out, 'partial_smoke' if args.smoke else 'partial')
    os.makedirs(partial, exist_ok=True)
    t0 = time.time()
    X, Y, G, _, _, data_info = load_brisc(args.features, args.groups)
    print('BRISC:', data_info)
    selection_folds = brisc_group_splits(X, Y, G, SELECTION_SEEDS)
    confirmation_folds = brisc_group_splits(X, Y, G, settings['confirmation_seeds'])
    grid = settings['grid']

    selection_file = os.path.join(partial, 'selection.json')
    if os.path.exists(selection_file):
        with open(selection_file, encoding='utf-8') as f:
            selection = json.load(f)
    else:
        jobs = [(j, cfg, fold, 42 + i, settings['epochs'])
                for j, (cfg, (i, fold)) in enumerate((c, f) for c in grid for f in enumerate(selection_folds))]
        rows = [out for _, out in sorted(parallel(grid_job, jobs, args.processes), key=lambda t: t[0])]
        selection = select_brisc(rows, selection_folds, grid)
        with open(selection_file, 'w', encoding='utf-8') as f:
            json.dump(selection, f, indent=1, ensure_ascii=False)
    print('selected:', selection['selected'])
    by_key = {config_key(c): c for c in grid}
    selected = {m: by_key[selection['selected'][m]] for m in ('H2', 'H3')}

    outs = {}
    jobs = []
    for arch in ('H2', 'H3'):
        for i, fold in enumerate(confirmation_folds):
            path = os.path.join(partial, f'conf_{arch}_{i}.json')
            if os.path.exists(path):
                with open(path, encoding='utf-8') as f:
                    outs[(arch, i)] = json.load(f)
            else:
                jobs.append((arch, i, selected[arch], fold, 1000 + i, i == 0, settings))
    jobs.sort(key=lambda j: (j[1] != 0, j[0], j[1]))
    print(f'confirmation folds to run: {len(jobs)}', flush=True)
    for key, out in parallel(fold_job, jobs, args.processes):
        outs[key] = out
        with open(os.path.join(partial, f'conf_{key[0]}_{key[1]}.json'), 'w', encoding='utf-8') as f:
            json.dump(out, f)
    R, verdict = summarize_brisc(outs, len(confirmation_folds), selected)
    controls = brisc_controls(confirmation_folds, selection['selected']['CART'])
    res = {'status': 'reproduction', 'smoke': args.smoke, 'data': data_info, 'selection': selection,
           'controls': controls, 'datasets': {'BRISC': R}, 'verdict': verdict,
           'runtime_s': round(time.time() - t0, 1), 'n_proc': args.processes}
    path = os.path.join(args.out, 'brisc_smoke.json' if args.smoke else 'brisc_results.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print('controls:', controls)
    print('verdict:', verdict)
    print('saved', path)


if __name__ == '__main__':
    main()
