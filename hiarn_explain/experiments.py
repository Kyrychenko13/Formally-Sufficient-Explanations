import itertools
import time

import numpy as np
import torch
from sklearn.metrics import f1_score

from .anchors import anchor_features, anchor_violated, make_explainer, network_predictor
from .baselines import CART_DEPTHS, cart_run, trivial_run
from .explain import N_VERIFY, explain, predict, sufficient_bounds, sufficient_milp, violations
from .model import EPOCHS, TAUE, parts_of, train

N_COMPARE = 100
N_ANCHOR_BRISC = 20
F1_TOL = 0.01
LAMBDAS = [1e-4, 1e-3, 1e-2]
TABULAR_CONFIGS = {
    'H2': {
        'breast_cancer': {'model': 'H2', 'rules': 8, 'lam': 1e-2},
        'pima_diabetes': {'model': 'H2', 'rules': 32, 'lam': 1e-4},
        'compas': {'model': 'H2', 'rules': 8, 'lam': 1e-2},
    },
    'H3': {
        'breast_cancer': {'model': 'H3', 'rules': 16, 'lam': 1e-2},
        'pima_diabetes': {'model': 'H3', 'rules': 16, 'lam': 1e-4},
        'compas': {'model': 'H3', 'rules': 16, 'lam': 1e-2},
    },
}
BRISC_GRID = ([{'model': 'H2', 'rules': r, 'lam': lam} for r in (8, 16, 32) for lam in LAMBDAS]
              + [{'model': 'H3', 'rules': 16, 'lam': lam} for lam in LAMBDAS])
CMP_KEYS = ('cmp_ours_len', 'cmp_ours_ms', 'milp_len', 'milp_ms', 'anchor_len', 'anchor_ms')


def config_key(cfg):
    return f'{cfg["model"]}|r={cfg["rules"]}|lam={cfg["lam"]}'


def n_classes(y_tr, y_te):
    return int(max(y_tr.max(), y_te.max()) + 1)


def timed_explain(P, s, check):
    t0 = time.perf_counter()
    c_hat, fixed = explain(P, s, check)
    return c_hat, fixed, 1000 * (time.perf_counter() - t0)


def explain_all(P, S_te, S_tr, rng, out, n_verify=N_VERIFY):
    for i in range(len(S_te)):
        c_hat, fixed, ms = timed_explain(P, S_te[i], sufficient_bounds)
        out['ours_ms'].append(ms)
        out['ours_len'].append(int(fixed.sum()))
        out['ours_viol'] += violations(P, S_te[i], fixed, c_hat, S_tr, rng, n_verify)


def compare_methods(P, X_tr, X_te, S_te, S_tr, n_cls, rng, out, n_compare, n_anchor, n_verify=N_VERIFY):
    pred_fn = network_predictor(P)
    explainer = make_explainer(X_tr, n_cls)
    out.update({
        'cmp_ours_len': out['ours_len'][:n_compare],
        'cmp_ours_ms': out['ours_ms'][:n_compare],
        'milp_len': [], 'milp_ms': [], 'anchor_len': [], 'anchor_ms': [], 'anchor_viol': 0, 'milp_viol': 0,
    })
    for i in range(min(n_compare, len(S_te))):
        c_hat, fixed, ms = timed_explain(P, S_te[i], sufficient_milp)
        out['milp_ms'].append(ms)
        out['milp_len'].append(int(fixed.sum()))
        out['milp_viol'] += violations(P, S_te[i], fixed, c_hat, S_tr, rng, n_verify)
        if i >= n_anchor:
            continue
        t0 = time.perf_counter()
        feats = anchor_features(explainer, X_te[i], pred_fn)
        out['anchor_ms'].append(1000 * (time.perf_counter() - t0))
        out['anchor_len'].append(len(feats))
        out['anchor_viol'] += anchor_violated(P, X_te[i], S_te[i], feats, X_tr, rng, n_verify)


def run_fold_tabular(arch, dataset, X_tr, y_tr, X_te, y_te, seed, compare, epochs=EPOCHS, n_compare=N_COMPARE):
    n_cls = n_classes(y_tr, y_te)
    m = train(TABULAR_CONFIGS[arch][dataset], X_tr, y_tr, n_cls, seed, epochs=epochs)
    P = parts_of(m)
    t = P[0]
    S_te = (X_te > t).astype(np.float64)
    S_tr = (X_tr > t).astype(np.float64)
    rng = np.random.RandomState(seed)
    out = {'ours_len': [], 'ours_ms': [], 'ours_viol': 0, 'n': len(S_te)}
    explain_all(P, S_te, S_tr, rng, out)
    if compare:
        compare_methods(P, X_tr, X_te, S_te, S_tr, n_cls, rng, out, n_compare, n_compare)
    return out


def length_counts(lengths):
    return {str(int(v)): int(n) for v, n in zip(*np.unique(lengths, return_counts=True))}


def comparison_summary(c):
    return {k: (float(np.median(c[k])) if k.endswith('len') else float(np.mean(c[k]))) for k in CMP_KEYS}


def summarize_explanations(folds):
    lengths = np.concatenate([o['ours_len'] for o in folds])
    c = folds[0]
    return lengths, {
        'ours_median': float(np.median(lengths)),
        'ours_mean': float(lengths.mean()),
        'ours_p90': float(np.percentile(lengths, 90)),
        'ours_viol': int(sum(o['ours_viol'] for o in folds)),
        'n': int(sum(o['n'] for o in folds)),
        'ours_ms': float(np.mean(np.concatenate([o['ours_ms'] for o in folds]))),
        'cmp': comparison_summary(c),
        'cmp_anchor_viol_share': c['anchor_viol'] / len(c['anchor_len']),
        'cmp_milp_viol': c['milp_viol'],
    }


def criterion_k2(h):
    return bool(h['cmp']['milp_ms'] >= 10 * h['cmp']['cmp_ours_ms'] and h['cmp']['cmp_ours_len'] <= h['cmp']['milp_len'] + 1)


def criterion_k3(h):
    return bool(h['cmp_anchor_viol_share'] > 0 and h['ours_viol'] == 0)


def summarize_tabular(outs, datasets, n_folds, archs=('H2', 'H3')):
    res = {'datasets': {}}
    for ds in datasets:
        R = {}
        for arch in archs:
            folds = [outs[(ds, arch, i)] for i in range(n_folds)]
            lengths, r = summarize_explanations(folds)
            c = folds[0]
            r['cmp_n'] = len(c['anchor_len'])
            r['cmp_raw'] = {k: c[k] for k in CMP_KEYS}
            r['ours_len_counts'] = length_counts(lengths)
            R[arch] = r
        R['K2'] = criterion_k2(R['H2'])
        R['K3'] = criterion_k3(R['H2'])
        res['datasets'][ds] = R
    dd = res['datasets'].values()
    verdict = {
        'K1_H3_guarantee': all(d['H3']['ours_viol'] == 0 for d in dd),
        'K2_datasets': sum(d['K2'] for d in dd),
        'K3_datasets': sum(d['K3'] for d in dd),
    }
    verdict['all'] = verdict['K1_H3_guarantee'] and verdict['K2_datasets'] >= 2 and verdict['K3_datasets'] >= 2
    res['verdict'] = verdict
    return res


def evaluate_config(cfg, X_tr, y_tr, X_te, y_te, seed, epochs=EPOCHS):
    n_cls = n_classes(y_tr, y_te)
    m = train(cfg, X_tr, y_tr, n_cls, seed, epochs=epochs)
    Xe = torch.tensor(X_te)
    with torch.no_grad():
        p_soft = m(Xe, TAUE).argmax(1).numpy()
        logits, info = m.discrete(Xe)
    p = logits.argmax(1).numpy()
    info['feats'] = sorted(info['feats'])
    return {
        'rules_f1': float(f1_score(y_te, p, average='macro', zero_division=0)),
        'fidelity': float((p == p_soft).mean()),
        'degenerate': bool(len(np.unique(p)) == 1),
        **info,
    }


def mean_of(rows, k):
    return float(np.mean([r[k] for r in rows]))


def feature_jaccard(rows):
    sets = [set(r['feats']) for r in rows]
    values = [len(a & b) / len(a | b) if (a | b) else 1.0 for a, b in itertools.combinations(sets, 2)]
    return float(np.mean(values))


def summarize_grid(rows):
    out = {k: round(mean_of(rows, k), 4)
           for k in ('rules_f1', 'fidelity', 'dl', 'n_rules', 'items_per_rule', 'expanded_len', 'degenerate')}
    if 'n_concepts' in rows[0]:
        out['n_concepts'] = round(mean_of(rows, 'n_concepts'), 2)
    out['feature_jaccard'] = round(feature_jaccard(rows), 3)
    return out


def select_brisc(grid_rows, selection_folds, configs=BRISC_GRID):
    G = {config_key(c): grid_rows[j * len(selection_folds):(j + 1) * len(selection_folds)] for j, c in enumerate(configs)}
    selected = {}
    for model in ('H2', 'H3'):
        sub = {k: v for k, v in G.items() if k.startswith(model + '|')}
        best = max(mean_of(v, 'rules_f1') for v in sub.values())
        ok = {k: v for k, v in sub.items() if mean_of(v, 'rules_f1') >= best - F1_TOL}
        selected[model] = min(ok, key=lambda k: (mean_of(ok[k], 'dl'), -mean_of(ok[k], 'rules_f1')))
    cart = {d: [cart_run(d, f, 42 + i) for i, f in enumerate(selection_folds)] for d in CART_DEPTHS}
    best_cart = max(mean_of(v, 'rules_f1') for v in cart.values())
    selected['CART'] = min((d for d in CART_DEPTHS if mean_of(cart[d], 'rules_f1') >= best_cart - F1_TOL),
                           key=lambda d: mean_of(cart[d], 'dl'))
    return {
        'selected': selected,
        'grid': {k: summarize_grid(v) for k, v in G.items()},
        'cart_grid': {str(d): {k: round(mean_of(v, k), 4) for k in ('rules_f1', 'dl')} for d, v in cart.items()},
    }


def run_fold_brisc(arch, cfg, X_tr, y_tr, X_te, y_te, seed, compare, epochs=EPOCHS,
                   n_compare=N_COMPARE, n_anchor=N_ANCHOR_BRISC):
    n_cls = n_classes(y_tr, y_te)
    m = train(cfg, X_tr, y_tr, n_cls, seed, epochs=epochs)
    Xe = torch.tensor(X_te)
    with torch.no_grad():
        p_soft = m(Xe, TAUE).argmax(1).numpy()
        logits, info = m.discrete(Xe)
    p_hard = logits.argmax(1).numpy()
    P = parts_of(m)
    t = P[0]
    S_te = (X_te > t).astype(np.float64)
    S_tr = (X_tr > t).astype(np.float64)
    rng = np.random.RandomState(seed)
    out = {
        'rules_f1': float(f1_score(y_te, p_hard, average='macro', zero_division=0)),
        'model_f1': float(f1_score(y_te, p_soft, average='macro', zero_division=0)),
        'fidelity': float((p_hard == p_soft).mean()),
        'degenerate': bool(len(np.unique(p_hard)) == 1),
        'dl': info['dl'],
        'n_rules': info['n_rules'],
        'pred_mismatch': int((predict(P, S_te) != p_hard).sum()),
        'n_used_conditions': int((np.abs(P[1][0]).sum(0) > 0).sum()),
        'ours_len': [], 'ours_ms': [], 'ours_viol': 0, 'n': len(S_te),
    }
    explain_all(P, S_te, S_tr, rng, out)
    if compare:
        compare_methods(P, X_tr, X_te, S_te, S_tr, n_cls, rng, out, n_compare, n_anchor)
    return out


def summarize_brisc(outs, n_folds, selected_configs):
    R = {}
    for arch in ('H2', 'H3'):
        folds = [outs[(arch, i)] for i in range(n_folds)]
        lengths, r = summarize_explanations(folds)
        c = folds[0]
        R[arch] = {
            'config': selected_configs[arch],
            'rules_f1': [round(float(np.mean([o['rules_f1'] for o in folds])), 4),
                         round(float(np.std([o['rules_f1'] for o in folds])), 4)],
            'model_f1': round(float(np.mean([o['model_f1'] for o in folds])), 4),
            'fidelity': [round(float(np.mean([o['fidelity'] for o in folds])), 4),
                         round(float(np.min([o['fidelity'] for o in folds])), 4)],
            'dl': round(float(np.mean([o['dl'] for o in folds])), 2),
            'n_rules': round(float(np.mean([o['n_rules'] for o in folds])), 2),
            'n_used_conditions': round(float(np.mean([o['n_used_conditions'] for o in folds])), 2),
            'degenerate_folds': int(sum(o['degenerate'] for o in folds)),
            'pred_mismatch': int(sum(o['pred_mismatch'] for o in folds)),
            'empty_explanations': int((lengths == 0).sum()),
            'ours_median': r['ours_median'],
            'ours_mean': r['ours_mean'],
            'ours_p90': r['ours_p90'],
            'ours_viol': r['ours_viol'],
            'n': r['n'],
            'ours_ms': r['ours_ms'],
            'cmp': r['cmp'],
            'cmp_anchor_viol_share': r['cmp_anchor_viol_share'],
            'cmp_milp_viol': r['cmp_milp_viol'],
            'cmp_n': len(c['milp_len']),
            'cmp_n_anchor': len(c['anchor_len']),
            'cmp_raw': {k: c[k] for k in CMP_KEYS},
            'ours_len_counts': length_counts(lengths),
            'per_fold': [{k: o[k] for k in ('rules_f1', 'model_f1', 'fidelity', 'dl', 'n_rules', 'degenerate',
                                            'pred_mismatch', 'ours_viol', 'n')} for o in folds],
        }
    for arch in ('H2', 'H3'):
        R[f'K2_{arch}'] = criterion_k2(R[arch])
        R[f'K3_{arch}'] = criterion_k3(R[arch])
    verdict = {
        'K1_H2': R['H2']['ours_viol'] == 0,
        'K1_H3': R['H3']['ours_viol'] == 0,
        'K2_H2': R['K2_H2'],
        'K3_H2': R['K3_H2'],
        'K2_H3_ref': R['K2_H3'],
        'K3_H3_ref': R['K3_H3'],
    }
    verdict['all'] = bool(verdict['K1_H2'] and verdict['K1_H3'] and R['K2_H2'] and R['K3_H2'])
    return R, verdict


def brisc_controls(confirmation_folds, cart_depth):
    controls = {'trivial_f1': round(float(np.mean([trivial_run(f)['rules_f1'] for f in confirmation_folds])), 4)}
    cr = [cart_run(cart_depth, f, 1000 + i) for i, f in enumerate(confirmation_folds)]
    controls['cart'] = {
        'depth': cart_depth,
        'rules_f1': round(mean_of(cr, 'rules_f1'), 4),
        'dl': round(mean_of(cr, 'dl'), 2),
        'n_rules': round(mean_of(cr, 'n_rules'), 2),
    }
    return controls
