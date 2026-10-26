import numpy as np
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

CART_DEPTHS = [2, 3, 4, 5, 6]


def leaf_depths(tree):
    depth = np.zeros(tree.node_count, int)
    for i in range(tree.node_count):
        for child in (tree.children_left[i], tree.children_right[i]):
            if child != -1:
                depth[child] = depth[i] + 1
    leaves = np.where(tree.children_left == -1)[0]
    return depth, leaves


def cart_run(depth, fold, seed):
    X_tr, y_tr, X_te, y_te = fold
    clf = DecisionTreeClassifier(max_depth=depth, class_weight='balanced', random_state=seed).fit(X_tr, y_tr)
    d, leaves = leaf_depths(clf.tree_)
    return {
        'rules_f1': float(f1_score(y_te, clf.predict(X_te), average='macro')),
        'dl': int(d[leaves].sum()),
        'n_rules': int(len(leaves)),
    }


def trivial_run(fold):
    X_tr, y_tr, X_te, y_te = fold
    majority = np.full(len(y_te), np.bincount(y_tr).argmax())
    return {'rules_f1': float(f1_score(y_te, majority, average='macro'))}
