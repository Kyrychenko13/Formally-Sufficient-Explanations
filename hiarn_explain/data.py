import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

PIMA_URL = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv'
COMPAS_URL = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'
PIMA_FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age']
PIMA_ZERO_AS_MISSING = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
COMPAS_FEATURES = ['sex', 'age', 'race', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count',
                   'days_b_screening_arrest', 'c_charge_degree']
COMPAS_CATEGORICAL = ['sex', 'race', 'c_charge_degree']
BRISC_CLASSES = ['glioma', 'meningioma', 'pituitary', 'no_tumor']
CONFIRMATION_SEED = 2026
SELECTION_SEED = 42
N_SPLITS = 5


def load_breast_cancer_data():
    bc = load_breast_cancer()
    return bc.data.astype(np.float32), bc.target.astype(np.int64)


def load_pima(url=PIMA_URL):
    df = pd.read_csv(url, header=None)
    df.columns = PIMA_FEATURES + ['Outcome']
    for c in PIMA_ZERO_AS_MISSING:
        df[c] = df[c].replace(0, np.nan).fillna(df[c].median())
    return df[PIMA_FEATURES].values.astype(np.float32), df['Outcome'].values.astype(np.int64)


def load_compas(url=COMPAS_URL):
    raw = pd.read_csv(url)
    df = raw[COMPAS_FEATURES + ['two_year_recid']].copy().dropna()
    df = df[(df['days_b_screening_arrest'] <= 30) & (df['days_b_screening_arrest'] >= -30)]
    for c in COMPAS_CATEGORICAL:
        df[c] = LabelEncoder().fit_transform(df[c])
    return df[COMPAS_FEATURES].values.astype(np.float32), df['two_year_recid'].values.astype(np.int64)


def load_tabular_datasets():
    return {
        'breast_cancer': load_breast_cancer_data(),
        'pima_diabetes': load_pima(),
        'compas': load_compas(),
    }


def standardise(X_tr, y_tr, X_te, y_te):
    sc = StandardScaler().fit(X_tr)
    return sc.transform(X_tr).astype(np.float32), y_tr, sc.transform(X_te).astype(np.float32), y_te


def tabular_splits(X, y, confirm, n_repeats=3):
    if confirm:
        it = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=n_repeats, random_state=CONFIRMATION_SEED).split(X, y)
    else:
        it = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SELECTION_SEED).split(X, y)
    return [standardise(X[tr], y[tr], X[te], y[te]) for tr, te in it]


def load_brisc(features_csv, groups_csv):
    df = pd.read_csv(features_csv)
    grp = pd.read_csv(groups_csv)
    grp = grp[grp.source == 'brisc'][['key', 'label', 'group', 'label_conflict']]
    df = df.merge(grp, left_on='filename', right_on='key', how='left', suffixes=('', '_g'))
    if not (df.group.notna().all() and (df.label == df.label_g).all()):
        raise ValueError('groups do not match the feature file')
    features = [c for c in df.columns if c.startswith('original_')]
    X = df[features].values.astype(np.float32)
    Y = df.label.values.astype(np.int64)
    G = df.group.values.astype(np.int64)
    is_test = df.filename.str.contains('_test_').values
    info = {
        'n': int(len(df)),
        'n_features': len(features),
        'class_counts': np.bincount(Y).tolist(),
        'n_groups': int(df.group.nunique()),
        'groups_gt1': int((df.group.value_counts() > 1).sum()),
        'groups_cross_official_split': int(pd.Series(is_test).groupby(G).nunique().gt(1).sum()),
        'label_conflict_rows': int(df.label_conflict.sum()),
    }
    return X, Y, G, features, df, info


def brisc_group_splits(X, Y, G, seeds):
    out = []
    for seed in seeds:
        folds = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed).split(X, Y, G)
        for tr, te in folds:
            if set(G[tr]) & set(G[te]):
                raise ValueError('a group appears in both training and test parts')
            out.append(standardise(X[tr], Y[tr], X[te], Y[te]))
    return out
