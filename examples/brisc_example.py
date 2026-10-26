import json
import os
import sys

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, ROOT)
from hiarn_explain import train, parts_of, explain, violations, sufficient_bounds

CFG_H2 = {'model': 'H2', 'rules': 16, 'lam': 0.01}
CLASS_NAMES = ['glioma', 'meningioma', 'pituitary', 'no_tumor']
FEATURES_CSV = os.environ.get('BRISC_FEATURES', os.path.join(ROOT, 'data', 'brisc_features.csv'))


def main():
    print('sklearn', sklearn.__version__, 'torch', torch.__version__)
    df = pd.read_csv(FEATURES_CSV)
    grp = pd.read_csv(os.path.join(ROOT, 'data', 'brisc_groups.csv'))
    grp = grp[grp.source == 'brisc'][['key', 'label', 'group', 'label_conflict']]
    df = df.merge(grp, left_on='filename', right_on='key', how='left', suffixes=('', '_g'))
    assert df.group.notna().all() and (df.label == df.label_g).all()
    feats = [c for c in df.columns if c.startswith('original_')]
    X = df[feats].values.astype(np.float32)
    Y = df.label.values.astype(np.int64)
    G = df.group.values.astype(np.int64)
    tr, te = next(iter(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026).split(X, Y, G)))
    sc = StandardScaler().fit(X[tr])
    X_tr, y_tr, X_te, y_te = sc.transform(X[tr]).astype(np.float32), Y[tr], sc.transform(X[te]).astype(np.float32), Y[te]
    n_cls = int(max(y_tr.max(), y_te.max()) + 1)
    seed = 1000
    m = train(CFG_H2, X_tr, y_tr, n_cls, seed)
    with torch.no_grad():
        lg, info = m.discrete(torch.tensor(X_te))
    p_hard = lg.argmax(1).numpy()
    P = parts_of(m)
    t = P[0]
    S_te, S_tr = (X_te > t).astype(np.float64), (X_tr > t).astype(np.float64)
    rng = np.random.RandomState(seed)
    lens, viols, expl = [], [], []
    for k in range(len(S_te)):
        c_hat, fx = explain(P, S_te[k], sufficient_bounds)
        lens.append(int(fx.sum()))
        viols.append(violations(P, S_te[k], fx, c_hat, S_tr, rng))
        expl.append((c_hat, fx))
    ref = json.load(open(os.path.join(ROOT, 'results', 'brisc_H2_fold0.json'), encoding='utf-8'))
    rf1 = float(f1_score(y_te, p_hard, average='macro', zero_division=0))
    used = np.abs(P[1][0]).sum(0) > 0
    repro = {'n': [len(S_te), ref['n']], 'n_rules': [info['n_rules'], ref['n_rules']], 'dl': [info['dl'], ref['dl']],
             'n_used_conditions': [int(used.sum()), ref['n_used_conditions']], 'rules_f1': [rf1, ref['rules_f1']],
             'ours_len_equal': lens == ref['ours_len'], 'ours_viol': [int(sum(viols)), ref['ours_viol']]}
    repro['identical'] = bool(repro['ours_len_equal'] and repro['n'][0] == repro['n'][1] and repro['n_rules'][0] == repro['n_rules'][1]
                              and repro['dl'][0] == repro['dl'][1] and abs(rf1 - ref['rules_f1']) < 1e-6)
    print('reproduction of the reported fold:', repro)
    files = df.filename.values[te]
    cases = []
    for c in range(4):
        k = next(k for k in range(len(te)) if y_te[k] == c and p_hard[k] == c)
        c_hat, fx = expl[k]
        conds = []
        for j in np.where(fx)[0]:
            conds.append({'feature': feats[j], 'op': '>' if S_te[k][j] == 1 else '<=', 't_std': float(t[j]),
                          't_raw': float(t[j] * sc.scale_[j] + sc.mean_[j]), 'x_raw': float(X[te][k][j]),
                          'importance': float(np.abs(P[1][0][:, j]).sum())})
        conds.sort(key=lambda d: -d['importance'])
        cases.append({'class': CLASS_NAMES[c], 'test_index': int(k), 'row': int(te[k]), 'filename': files[k],
                      'decision': CLASS_NAMES[c_hat], 'length': int(fx.sum()), 'violated': viols[k], 'conditions': conds})
        print(f'{CLASS_NAMES[c]}: test case {k}, file {files[k]}, length {int(fx.sum())}, violated {viols[k]}')
    out = {'status': 'ILLUSTRATION (not part of reported results)', 'env': {'sklearn': sklearn.__version__, 'torch': torch.__version__},
           'config': CFG_H2, 'fold': 'confirmation fold 1 (seed 2026, index 0)', 'train_seed': seed, 'reproduction': repro,
           'n_used_conditions': int(used.sum()), 'cases': cases}
    path = os.path.join(ROOT, 'examples', 'brisc_example_explanations.json')
    json.dump(out, open(path, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('saved', path)


if __name__ == '__main__':
    main()
