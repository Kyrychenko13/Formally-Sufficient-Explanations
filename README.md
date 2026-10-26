# Formally Sufficient Explanations of Individual Decisions in Differentiable Threshold Rule Networks

The method explains a single decision of a trained threshold rule network (HiARN-2, HiARN-3) by greedy removal of conditions combined with interval bound propagation through the threshold layers. Every returned explanation is formally sufficient: whatever values the conditions outside the explanation take, the decision does not change. No solver is used.

## Contents

| Path                                            | What it is                                                                                                                       |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `hiarn_explain/model.py`                        | Threshold rule network (HiARN-2, HiARN-3), training with temperature annealing and proximal ℓ1 step, discretisation and pruning. |
| `hiarn_explain/explain.py`                      | Interval bound check, greedy explanation, exact MILP check, empirical check of an explanation.                                   |
| `hiarn_explain/anchors.py`                      | Anchors baseline and its empirical check.                                                                                        |
| `hiarn_explain/data.py`                         | Tabular datasets, BRISC features and groups, confirmation and selection splits.                                                  |
| `hiarn_explain/baselines.py`                    | Decision tree and majority-class controls.                                                                                       |
| `hiarn_explain/experiments.py`                  | Configuration selection, one confirmation fold with explanations and comparisons, aggregation and criteria K1–K3.                |
| `hiarn_explain/radiomics.py`                    | PyRadiomics feature extraction for BRISC.                                                                                        |
| `scripts/extract_brisc_radiomics.py`            | Builds `data/brisc_features.csv` from the unpacked BRISC 2025 dataset.                                                           |
| `scripts/run_tabular.py`                        | The tabular experiment without notebooks; writes a result file of the same structure as `results/tabular_results_run2.json`.     |
| `scripts/run_brisc.py`                          | The BRISC experiment without notebooks; writes a result file of the same structure as `results/brisc_results.json`.              |
| `notebooks/01_brisc_radiomics_extraction.ipynb` | Extraction of PyRadiomics features from the BRISC 2025 MRI dataset (`brisc_features.csv`).                                       |
| `notebooks/02_explanations_tabular.ipynb`       | Official run on Breast Cancer Wisconsin, Pima Indians Diabetes and COMPAS (executed copy, outputs included).                     |
| `notebooks/03_explanations_brisc.ipynb`         | Official run on BRISC radiomics (executed copy, outputs included).                                                               |
| `results/`                                      | Result files of the official runs (see below).                                                                                   |
| `data/brisc_groups.csv`                         | Groups of near-duplicate BRISC images used for group-aware splits.                                                               |
| `examples/toy_example.py`                       | The worked example of the paper (Figures 2–3), checked by enumeration.                                                           |
| `examples/brisc_example.py`                     | The illustrative BRISC explanations (Figure 10, Table A2).                                                                       |
| `tests/test_explain.py`                         | Checks against exhaustive enumeration on random networks.                                                                        |

## Results files

| File                                      | Content                                                                                                        | Used in the paper                         |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| `results/tabular_results_run2.json`       | Tabular datasets, HiARN-2 and HiARN-3, 15 confirmation folds; comparison with exact MILP and Anchors on fold 1 | Tables 3–5, Figures 5–9                   |
| `results/tabular_results_run1.json`       | First run of the same notebook (inflated times, identical lengths and violations)                              | Appendix B                                |
| `results/brisc_results.json`              | BRISC, both networks, 15 confirmation folds, configuration selection, controls                                 | Tables 2–5, Figures 5–9                   |
| `results/brisc_H2_fold0.json`             | Per-case output of HiARN-2 on BRISC, confirmation fold 1                                                       | reference for `examples/brisc_example.py` |
| `results/brisc_example_explanations.json` | Illustrative explanations of four BRISC cases (retrained network, not part of the reported results)            | Figure 10, Table A2                       |

## Using the method

```python
import numpy as np
from hiarn_explain import train, parts_of, explain, violations, sufficient_bounds, conditions

model = train({'model': 'H2', 'rules': 16, 'lam': 1e-2}, X_train, y_train, n_classes, seed=1000)
P = parts_of(model)
S_test = (X_test > P[0]).astype(np.float64)
decision, fixed = explain(P, S_test[0], sufficient_bounds)
print(decision, conditions(P, S_test[0], fixed))
```

`X_train` and `X_test` are standardised `float32` arrays, `y_train` is an `int64` array of class indices. `conditions` returns the explanation as `(feature index, '>' or '<=', threshold)` triples on the standardised scale.

## Running the experiments without notebooks

```
python scripts/run_tabular.py --out tabular_out
python scripts/extract_brisc_radiomics.py --brisc path/to/brisc2025 --out data/brisc_features.csv
python scripts/run_brisc.py --out brisc_out
```

`--smoke` gives a short run (fewer epochs, one repetition, few compared cases) for testing the pipeline, and
`--processes` sets the number of worker processes. Every finished fold is saved in the `partial` folder of the output directory, so an interrupted run continues from where it stopped. 

The code of `hiarn_explain` and `scripts` was checked against the notebooks: on the same folds and seeds it returns the same trained networks, explanations, MILP and Anchors results and empirical checks, and aggregating the per-fold outputs of the official runs with `summarize_tabular` and `summarize_brisc` reproduces the result files in `results/`.

## Reproducing the official runs

The official runs were executed in Google Colab; the notebooks write their outputs to Google Drive.

1. Tabular datasets: run `notebooks/02_explanations_tabular.ipynb`. The data are downloaded automatically
   (scikit-learn, the Pima Indians Diabetes CSV and the ProPublica COMPAS CSV).
2. BRISC: run `notebooks/01_brisc_radiomics_extraction.ipynb`. It downloads BRISC 2025 from Kaggle
   (`briscdataset/brisc2025`) with the user's own Kaggle API token and writes `MyDrive/HiARN/data/brisc_features.csv`.
   Copy `data/brisc_groups.csv` to `MyDrive/HiARN/E5D/groups.csv` and run `notebooks/03_explanations_brisc.ipynb`.
3. Outside Colab, notebook 02 writes to `$AXP_OUT`; notebook 03 reads `$E5E_DATA/data/brisc_features.csv` and
   `$E5E_DATA/E5D/groups.csv`.

Notes:

- PyRadiomics ignored the parameter dictionary passed in notebook 01 and used its default settings; the extracted file has 93 first-order and texture features, as reported in the paper.
- Training is deterministic for a given seed and environment, but results can differ slightly between CPU types and library versions. The retrained network of `examples/brisc_example.py` differs from the network of the official
  run (macro-F1 0.774 versus 0.770 on the fold), which is why the illustrative example is not part of the reported results.
- Notebook 02 also contains helper definitions from earlier diagnostics (other network variants and baselines) that are not called in this run. 
- The functions in `hiarn_explain` are the functions of notebooks 02 and 03 without the unused variants, written out in full; on the same data and seed they return identical results.

## Installation

```
pip install -r requirements.txt
python tests/test_explain.py
```

## Citation

If you use this code, please cite the paper (citation details will be added after publication).
