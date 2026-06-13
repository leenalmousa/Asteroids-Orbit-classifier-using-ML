# Asteroid Orbit Classifier

A supervised machine-learning pipeline that classifies asteroids into their **orbit type** (MBA, Atira, Apollo, Amor, Aten, etc.) from physical and orbital features. Built end-to-end: exploratory analysis, class-imbalance handling, outlier removal, feature scaling, PCA, model comparison via cross-validation, and ensemble methods.

[![Conference](https://img.shields.io/badge/Published-ICTCS%202025%20%C2%B7%20PSUT-2E8B57)](https://www.researchgate.net/publication/391933697_Asteroid_Orbit_Classification_with_Machine_Learning_A_Data-Driven_Approach)
[![Made with Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](Code.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Publication

This work was **published at [ICTCS 2025](https://www.researchgate.net/publication/391933697_Asteroid_Orbit_Classification_with_Machine_Learning_A_Data-Driven_Approach)** &mdash; *International Conference on Information Technology and Computer Science*, hosted at **Princess Sumaya University for Technology (PSUT)**, Jordan.

> **Asteroid Orbit Classification with Machine Learning: A Data-Driven Approach**
> &nbsp;&nbsp;&nbsp;Read on [ResearchGate](https://www.researchgate.net/publication/391933697_Asteroid_Orbit_Classification_with_Machine_Learning_A_Data-Driven_Approach) &middot; local copy: [`Asteroid_Orbit_Classifier_Report.pdf`](Asteroid_Orbit_Classifier_Report.pdf)

## Highlights

- **Real pipeline, not just `model.fit()`** — duplicate removal, class-aware outlier handling, log-transform of skewed features, scaling, PCA.
- **Imbalance handling** — undersamples the dominant **MBA** class, oversamples the rare **Atira** class on the training set only (no leakage).
- **Six classifiers compared** via k-fold cross-validation: Decision Tree, K-Nearest Neighbors, Multi-Layer Perceptron, Gaussian Naïve Bayes, Random Forest, and SVM.
- **Ensembles** — bagging on decision trees and a soft-voting classifier combining the strongest base models.
- **Top result:** Multi-Layer Perceptron — **100% test accuracy**.

## Dataset

Orbital and physical parameters for all known asteroids, with **`Orbit_type`** as the target label. The notebook expects `orbits.csv` in the repository root.

Source: NASA JPL Small-Body Database — also available as the [Asteroid Dataset on Kaggle](https://www.kaggle.com/datasets/sakhawat18/asteroid-dataset). Download `orbits.csv` (or rename the equivalent file) and place it next to the notebook before running.

## Pipeline

1. **EDA** — schema inspection, correlation matrix vs. one-hot labels, categorical/numerical split.
2. **Cleaning** — drop irrelevant columns and duplicates; convert mistyped numerical columns; impute remaining nulls (median for numeric, mode for categorical).
3. **Class balance** — undersample MBA; oversample Atira **on the training fold only**.
4. **Outliers** — per-class outlier removal so dispersion of rare classes is preserved.
5. **Transforms** — log transform on heavily-skewed features; Standard Scaling.
6. **Dimensionality reduction** — PCA (Truncated SVD also evaluated and rejected).
7. **Model selection** — k-fold cross-validation across the six classifiers.
8. **Ensembles** — `BaggingClassifier(DecisionTreeClassifier())` and a `VotingClassifier` over the strongest base learners.

## Results

| Model                          | Accuracy |
|--------------------------------|----------|
| Multi-Layer Perceptron (MLP)   | **100%** |
| Random Forest                  | high     |
| Decision Tree                  | high     |
| Gaussian Naïve Bayes           | medium   |
| K-Nearest Neighbors            | **83%** (lowest) |

> Full per-class precision / recall / F1 and confusion matrices are in [`Code.ipynb`](Code.ipynb).

## Repository structure

```
.
├── Code.ipynb     # full notebook — EDA, preprocessing, model comparison, ensembles
├── Asteroid_Orbit_Classifier_Report.pdf     # written report describing methodology and results
├── README.md
└── LICENSE
```

> `orbits.csv` is **not** committed (large + redistributable from JPL/Kaggle). Place it in the repo root before running.

## Quick start

```bash
git clone https://github.com/leenalmousa/Asteroids-Orbit-classifier-using-ML.git
cd Asteroids-Orbit-classifier-using-ML

# (optional) create a virtual env
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# place orbits.csv in this folder, then:
jupyter notebook Code.ipynb
```

## Citation

If you reference this work, please cite:

```bibtex
@inproceedings{almousa2025asteroid,
  title     = {Asteroid Orbit Classification with Machine Learning: A Data-Driven Approach},
  author    = {Almousa, Leen},
  booktitle = {Proceedings of the International Conference on Information Technology and Computer Science (ICTCS)},
  year      = {2025},
  address   = {Princess Sumaya University for Technology (PSUT), Jordan},
  url       = {https://www.researchgate.net/publication/391933697_Asteroid_Orbit_Classification_with_Machine_Learning_A_Data-Driven_Approach}
}
```

## Author

**Leen Almousa** &mdash; [github.com/leenalmousa](https://github.com/leenalmousa)

## License

Released under the [MIT License](LICENSE).
