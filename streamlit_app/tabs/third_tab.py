import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Racine du dépôt : les chemins ne dépendent plus du dossier depuis lequel on lance l'app
ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / 'data' / 'dev_data'
MODELS_ROOT = ROOT / 'models'
sys.path.insert(0, str(ROOT / 'notebooks'))
from include import dcase_baseline as dcase  # noqa: E402
from include import legacy_2022  # noqa: E402

title = "Prédiction des données"
sidebar_name = "Test prédiction"

#################################################################################
# Modèles DCASE (un autoencodeur par type de machine, entraîné sur train seulement)
#################################################################################
DCASE_DIR = MODELS_ROOT / 'dcase_baseline'

#################################################################################
# Modèles de la soutenance 2022 (remis en état par notebooks/include/legacy_2022.py)
#################################################################################
LEGACY_DIR = MODELS_ROOT / 'weight_train'
#################################################################################


@st.cache_data
def get_data():
    """Fichiers du dossier test (les seuls qui contiennent des anomalies)."""
    df = dcase.load_dataset_index(DATA_ROOT)
    df['filename'] = df.pathname.map(lambda p: Path(p).name)
    return df[df.data_split == 'test'].reset_index(drop=True)


@st.cache_resource
def load_dcase_model(machine_type):
    import keras
    model = keras.models.load_model(DCASE_DIR / f'{machine_type}.keras')
    info = json.loads((DCASE_DIR / f'{machine_type}.json').read_text(encoding='utf-8'))
    return model, info


@st.cache_data
def load_dcase_metrics():
    path = DCASE_DIR / 'metrics_by_machine_id.csv'
    return pd.read_csv(path) if path.exists() else None


@st.cache_resource
def load_legacy():
    import keras
    with open(LEGACY_DIR / 'Machine_L' / 'model_svm', 'rb') as f:
        svm = pickle.load(f)
    return {'svm': svm,
            'lenet': keras.models.load_model(LEGACY_DIR / legacy_2022.LENET_FILE),
            'ae': keras.models.load_model(LEGACY_DIR / legacy_2022.AE_FILE),
            'scalers': legacy_2022.load_scalers(LEGACY_DIR)}


def run_dcase(path, machine_type, machine_id):
    model, info = load_dcase_model(machine_type)
    with st.spinner('Calcul du score d\'anomalie...'):
        score, is_anomaly, logmel, errors = dcase.predict_file(path, model, info)

    col1, col2, col3 = st.columns(3)
    col1.metric("Score d'anomalie", f"{score:.2f}")
    col2.metric("Seuil (" + machine_type + ")", f"{info['threshold']:.2f}")
    metrics = load_dcase_metrics()
    if metrics is not None:
        row = metrics[(metrics.machine_type == machine_type) & (metrics.machine_id == machine_id)]
        if len(row):
            col3.metric("AUC du modèle sur " + machine_id, f"{row.AUC.iloc[0]:.2f}")
    if is_anomaly:
        st.error("Le son est anormal : score supérieur au seuil.")
    else:
        st.success("Le son est normal : score inférieur au seuil.")
    st.caption("Seuil : " + info['threshold_rule'] + ". Le jeu de test n'a pas servi à le choisir.")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    librosa.display.specshow(logmel, sr=dcase.SR, hop_length=dcase.HOP_LENGTH, x_axis='time', y_axis='mel',
                             cmap='viridis', ax=axes[0])
    axes[0].set(title='Log-mel spectrogramme', xlabel='')
    t = librosa.frames_to_time(np.arange(len(errors)) + dcase.N_FRAMES // 2, sr=dcase.SR, hop_length=dcase.HOP_LENGTH)
    axes[1].plot(t, errors, color='tab:red', lw=0.8)
    axes[1].axhline(info['threshold'], color='k', ls='--', lw=0.8, label='seuil (moyenne sur le fichier)')
    axes[1].set(ylabel='erreur', xlabel='temps (s)', title='Erreur de reconstruction au cours du temps')
    axes[1].legend(loc='upper right')
    fig.tight_layout()
    st.pyplot(fig)


def run_legacy(path, machine_type, model_choisi):
    st.warning("Modèles de la soutenance 2022, conservés pour mémoire : ils ont été évalués avec un protocole "
               "optimiste (voir le notebook « BASELINE DCASE »). Préférer l'autoencodeur DCASE.")
    if not (LEGACY_DIR / legacy_2022.SCALERS_FILE).exists():
        st.error("Modèles 2022 non préparés : exécuter « python -m include.legacy_2022 » depuis le dossier notebooks.")
        return
    legacy = load_legacy()
    with st.spinner('Prédiction...'):
        mel_db, mel2_db = legacy_2022.mel_features(path)
        if model_choisi == "Autoencodeur 2022":
            mse = legacy_2022.predict_autoencoder(legacy['ae'], mel_db, legacy['scalers'])
            st.write("Erreur de reconstruction : {:.3f} (seuil {})".format(mse, legacy_2022.THRESHOLD_AE))
            is_normal = mse < legacy_2022.THRESHOLD_AE
        else:
            if model_choisi == "SVM 2022":
                predicted = legacy_2022.predict_svm(legacy['svm'], mel_db, legacy['scalers'])
            else:
                predicted, _ = legacy_2022.predict_lenet(legacy['lenet'], mel2_db, legacy['scalers'])
            st.write("Type de machine reconnu : **" + predicted + "**")
            st.caption("Règle 2022 : si le type de machine est mal reconnu, le son est considéré comme anormal.")
            is_normal = predicted == machine_type
    if is_normal:
        st.success("Le son est normal")
    else:
        st.error("Le son est anormal")


def run():

    st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/3.gif")

    st.title(title)

    st.image(str(Path(__file__).resolve().parents[1] / "assets" / "PySaDML_process.PNG"))

    df = get_data()

    col1, col2, col3 = st.columns(3)
    machine_selected = col1.selectbox("Type de machine", sorted(df.machine_type.unique()))
    df_m = df[df.machine_type == machine_selected]
    id_selected = col2.selectbox("Identifiant de la machine", sorted(df_m.machine_id.unique()))
    condition_selected = col3.selectbox("Condition réelle", ['normal', 'anomaly'])
    files = df_m[(df_m.machine_id == id_selected) & (df_m.condition == condition_selected)]
    audio_selected = st.selectbox('Fichier à prédire', ['Aucun'] + files.filename.tolist())

    model_choisi = st.radio("Modèle", ["Autoencodeur DCASE (un par machine)", "Autoencodeur 2022", "SVM 2022", "CNN LeNet 2022"],
                            horizontal=True)

    if audio_selected == 'Aucun':
        return
    path = files.pathname[files.filename == audio_selected].iloc[0]
    st.audio(path)

    if model_choisi.startswith("Autoencodeur DCASE"):
        if not (DCASE_DIR / f'{machine_selected}.keras').exists():
            st.error("Modèle introuvable dans " + str(DCASE_DIR) + " : exécuter le notebook « PySaDML - BASELINE DCASE ».")
            return
        run_dcase(path, machine_selected, id_selected)
    else:
        run_legacy(path, machine_selected, model_choisi)
