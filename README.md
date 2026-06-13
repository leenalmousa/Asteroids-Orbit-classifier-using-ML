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
> &nbsp;&nbsp;&nbsp;Leen Almousa, Tala Hammouri, Lojain Hamadan and Dr.Heba Abdel-Nabi &mdash; PSUT
> &nbsp;&nbsp;&nbsp;[Read on ResearchGate](https://www.researchgate.net/publication/391933697_Asteroid_Orbit_Classification_with_Machine_Learning_A_Data-Driven_Approach) &middot; [Preprint PDF](Preprint_Asteroid_Orbit_Classifier_Report.pdf)

## Highlights

- **Real pipeline, not just `model.fit()`** — duplicate removal, class-aware outlier handling, log-transform of skewed features, scaling, PCA.
- **Imbalance handling** — undersamples the dominant **MBA** class, oversamples the rare **Atira** class on the training set only (no leakage).
- **Six classifiers compared** via k-fold cross-validation: Decision Tree, K-Nearest Neighbors, Multi-Layer Perceptron, Gaussian Naïve Bayes, Random Forest, and SVM.
- **Ensembles** — bagging on decision trees and a soft-voting classifier combining the strongest base models.
- **Top result:** Multi-Layer Perceptron — **100% test accuracy**.

## Dataset

Orbital parameters for every known asteroid catalogued by the IAU **Minor Planet Center (MPC)**, with **`Orbit_type`** as the target label. The notebook expects `orbits.csv` in the repository root.

**Source:** [Orbit Data for All Known Asteroids in MPC Database](https://www.kaggle.com/datasets/shivamshandilya/orbit-data-for-all-known-asteroids-in-mpc-database) on Kaggle. Download the CSV, rename it to `orbits.csv` (if needed), and place it next to the notebook before running.

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

Test-set accuracy across the five classifiers (Table 13 of the paper):

| Algorithm                      | Accuracy |
|--------------------------------|----------|
| **Multi-Layer Perceptron (MLP)** | **100%** |
| Random Forest                  | 97%      |
| Decision Tree                  | 96%      |
| Gaussian Naïve Bayes           | 95%      |
| K-Nearest Neighbors (KNN)      | 83%      |

**Comparison to prior work on the same dataset:**

| Source                            | Best model            | Accuracy |
|-----------------------------------|-----------------------|----------|
| **This paper**                    | Multi-Layer Perceptron| **100%** |
| Tibrewal & Dwivedi (prior work)   | SVM (RBF kernel)      | 97.9%    |

> Per-class precision / recall / F1 and confusion matrices for every model are in the [preprint PDF](Preprint_Asteroid_Orbit_Classifier_Report.pdf) (Tables 8&ndash;12) and reproduced in [`Code.ipynb`](Code.ipynb).

## Repository structure

```
.
├── Code.ipynb     # full notebook — EDA, preprocessing, model comparison, ensembles
├── Preprint_Asteroid_Orbit_Classifier_Report.pdf     # written report describing methodology and results
├── README.md
└── LICENSE
```

> `orbits.csv` is **not** committed (large + freely available from the Kaggle source above). Place it in the repo root before running.

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

## Author

**Leen Almousa** &mdash; [github.com/leenalmousa](https://github.com/leenalmousa)
&nbsp;&nbsp;Co-authored with Tala Hammouri , Lojain Hamadan and Dr.Heba Abdel-Nabi(PSUT).

## License

Released under the [MIT License](LICENSE).
