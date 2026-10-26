import numpy as np

from .explain import predict, N_VERIFY

ANCHOR_THRESHOLD = 0.95


def make_explainer(X_tr, n_cls):
    from anchor import anchor_tabular
    class_names = [str(c) for c in range(n_cls)]
    feature_names = [f'f{j}' for j in range(X_tr.shape[1])]
    return anchor_tabular.AnchorTabularExplainer(class_names, feature_names, X_tr)


def network_predictor(P):
    t = P[0]

    def pred_fn(Xr):
        return predict(P, (Xr > t).astype(np.float64))

    return pred_fn


def anchor_feature_index(name):
    parts = name.split(' ')
    if parts[0].startswith('f'):
        return int(parts[0][1:])
    return int(parts[2][1:])


def anchor_features(explainer, x, pred_fn, threshold=ANCHOR_THRESHOLD):
    e = explainer.explain_instance(x, pred_fn, threshold=threshold)
    names = e.names()
    if not names:
        return []
    return sorted({anchor_feature_index(nm) for nm in names})


def anchor_violated(P, x, s, feats, X_tr, rng, n_verify=N_VERIFY):
    keep = np.zeros(X_tr.shape[1], bool)
    keep[feats] = True
    Xr = np.repeat(x[None], n_verify, 0)
    Xr[:, ~keep] = X_tr[rng.randint(0, len(X_tr), n_verify)][:, ~keep]
    decision = int(predict(P, s[None])[0])
    return bool((network_predictor(P)(Xr) != decision).any())
