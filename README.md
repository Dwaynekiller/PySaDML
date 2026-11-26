# PySaDML

## Presentation

This repository contains the code for our project **PySaDML**, developed during our [Data Scientist training](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/).

The goal of this project is **to detect whether the sound emitted by a target machine is normal or abnormal**, on the [DCASE 2020 Task 2](https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds) dataset (6 machine types: ToyCar, ToyConveyor, fan, pump, slider, valve).

This project was developed by the following team :

- Laurent Noyelle ([LinkedIn](http://linkedin.com/in/lnoyelle))
- Steve Minlo ([LinkedIn](https://www.linkedin.com/in/steve-minlo-b3a844b9))
- Mike Guidy ([GitHub](https://github.com/Dwaynekiller) / [LinkedIn](https://www.linkedin.com/in/mike-eddie-g-04404144/))

## Results

The project was presented in 2022. It was revisited in 2026 to follow the official DCASE protocol: the anomaly detector is trained **only on the `train` folder** (normal sounds) and evaluated by **AUC / pAUC** per machine on the `test` folder (notebook [`PySaDML - BASELINE DCASE`](./notebooks/PySaDML%20-%20BASELINE%20DCASE.ipynb)).

| Machine type | AUC autoencoder (DCASE protocol) | pAUC | AUC supervised, random split (2022 protocol) | AUC supervised, unseen machine |
|---|---|---|---|---|
| ToyCar | 0.763 | 0.640 | 0.985 | 0.953 |
| ToyConveyor | 0.662 | 0.553 | 0.921 | 0.547 |
| fan | 0.634 | 0.526 | 0.907 | 0.538 |
| pump | 0.712 | 0.603 | 0.961 | 0.706 |
| slider | 0.787 | 0.610 | 0.982 | 0.588 |
| valve | 0.553 | 0.504 | 0.974 | 0.580 |
| **Mean** | **0.685** | **0.573** | 0.955 | 0.652 |

- The 2022 supervised scores (86-90 % accuracy) came mostly from the protocol: classifiers were trained and tested on a random split of the `test` folder, so the same machines were on both sides. On a machine never seen during training, the same approach drops to a mean AUC of 0.652.
- The autoencoder (one per machine type, DCASE 2020 baseline architecture, trained without any anomaly) reaches a mean AUC of 0.685, slightly below the official baseline (about 0.74). valve is the hardest machine type.
- Improvement experiments (section 6 of the notebook): scoring a file by the 99th percentile of its window errors instead of their mean raises valve from 0.553 to 0.705 AUC (slider: 0.787 to 0.853 with the 90th percentile), but lowers ToyCar and ToyConveyor, so the overall mean barely changes. One model per machine ID is worse (mean AUC 0.657). These choices were compared on the development `test` folder, so they would need confirmation on the DCASE evaluation set; the app keeps the baseline rule (mean).

## Repository content

| Path | Content |
|---|---|
| `notebooks/PySaDML - EDA.ipynb` | Exploratory data analysis |
| `notebooks/PySaDML - PRE_PROCESSING.ipynb` | MFCC / mel feature extraction (2022) |
| `notebooks/*classification*`, `CNN-MODEL-*`, `Auto encodeur FINAL` | Models presented in 2022. Kept as an archive: outputs are those of the 2022 run, and a review note at the top of each notebook lists its known issues |
| `notebooks/PySaDML - BASELINE DCASE.ipynb` | DCASE protocol: one autoencoder per machine type, AUC / pAUC, a fair evaluation of the 2022 supervised approach, and improvement experiments (file score, one model per machine ID) |
| `notebooks/include/common.py`, `models.py` | Shared functions of the 2022 notebooks |
| `notebooks/include/dcase_baseline.py` | DCASE features, model, training, evaluation and inference |
| `notebooks/include/legacy_2022.py` | Restores the 2022 models for the app (recomputed scalers, Keras 3 conversion) |
| `streamlit_app/` | Streamlit app |

## Installation

Tested with **Python 3.13** on Windows 11 (CPU only: TensorFlow does not use the GPU on native Windows).

```shell
python -m venv .venv
.venv\Scripts\activate          # Linux / macOS : source .venv/bin/activate
pip install -r requirements.txt
```

**Data and models are not in this repository.** Expected layout, at the root of the repository:

- `data/dev_data/` : the DCASE 2020 Task 2 development set (`fan/`, `pump/`, `slider/`, `ToyCar/`, `ToyConveyor/`, `valve/`, each with `train/` and `test/`)
- `data/dev_data.csv` : metadata file created by `common.load_metadata` (PRE_PROCESSING notebook), used by the EDA tab of the app
- `models/weight_train/` : the 2022 models (`Machine_L/model_svm`, `Deep_L/`, `Autoencoder/model_1/model_1`)

Then, from the `notebooks` folder:

```shell
# 1. DCASE autoencoders: trains and evaluates the 6 models per machine type, then the
#    23 models per machine ID (about 25 min each on CPU), results and models in models/dcase_baseline/
jupyter nbconvert --to notebook --execute --inplace "PySaDML - BASELINE DCASE.ipynb"

# 2. 2022 models for the app: recomputes their scalers and converts them to Keras 3
python -m include.legacy_2022
```

## Streamlit App

The app has three tabs: an introduction, an exploratory data analysis (listen to and compare normal / abnormal sounds, mel and MFCC spectrograms) and a prediction tab: pick a test file and a model. The **DCASE autoencoder** gives an anomaly score, the threshold of the machine type (chosen on validation data, never on the test set) and the reconstruction error over time. The 2022 models (SVM, CNN LeNet, autoencoder) are kept for reference.

To run the app (after the installation steps above):

```shell
cd streamlit_app
streamlit run app.py
```

The app should then be available at [localhost:8501](http://localhost:8501).

**Docker**

The image is built from the root of the repository, because the app uses `notebooks/include`:

```shell
docker build -f streamlit_app/Dockerfile -t pysadml-app .
docker run --name pysadml-app -p 8501:8501 -v "$(pwd)/data:/repo/data" -v "$(pwd)/models:/repo/models" pysadml-app
```

The `data/` and `models/` folders are mounted into the container because they are not copied into the image. The EDA tab reads `data/dev_data.csv`, whose paths are relative (`./../data/dev_data/...`): they are valid inside the container as long as the CSV was generated from the `notebooks` folder.
