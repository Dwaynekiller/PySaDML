"""
Détection d'anomalies sonores selon le protocole DCASE 2020 Task 2.

Ce module remplace l'approche de 2022 (classifieurs supervisés entraînés sur le
dossier `test`, autoencodeur unique pour toutes les machines) par la méthode de
référence du challenge :

- un autoencodeur par type de machine, entraîné **uniquement sur `train`**
  (qui ne contient que des sons normaux) ;
- en entrée, des fenêtres de 5 trames consécutives de log-mel à 128 bandes
  (5 x 128 = 640 valeurs), extraites du signal complet (10 ou 11 s) ;
- le score d'anomalie d'un fichier est l'erreur de reconstruction moyenne de
  ses fenêtres ;
- l'évaluation se fait par AUC et pAUC (FPR <= 0,1) par type de machine et par
  identifiant de machine, sur le dossier `test`. Aucun seuil n'est choisi en
  regardant le jeu de test.

Référence : Koizumi et al., « Description and Discussion on DCASE2020 Challenge
Task2: Unsupervised Anomalous Sound Detection for Machine Condition Monitoring »,
2020, et le code de référence du challenge.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

#---------------------------------------------------------------------------------------------
# Paramètres de la référence DCASE 2020
#---------------------------------------------------------------------------------------------
SR = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
N_FRAMES = 5           # trames consécutives par fenêtre
POWER = 2.0
DIMS = N_MELS * N_FRAMES

MAX_FPR = 0.1          # pAUC : AUC partielle sur FPR dans [0, 0.1]
THRESHOLD_PERCENTILE = 90  # seuil de décision : 90e centile des scores de validation (sons normaux)

MACHINE_TYPES = ['ToyCar', 'ToyConveyor', 'fan', 'pump', 'slider', 'valve']


#---------------------------------------------------------------------------------------------
# Caractéristiques
#---------------------------------------------------------------------------------------------
def file_to_logmel(file_path):
    """
    Log-mel spectrogramme (N_MELS, T) d'un fichier, en float32.

    Le signal n'est PAS normalisé par son pic : le niveau sonore peut justement
    trahir une anomalie.
    """
    y, sr = librosa.load(file_path, sr=None, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                         n_mels=N_MELS, power=POWER)
    logmel = 20.0 / POWER * np.log10(mel + np.finfo(np.float32).eps)
    return logmel.astype(np.float32)


def logmel_to_vectors(logmel):
    """
    Fenêtres glissantes de N_FRAMES trames : (T - N_FRAMES + 1, DIMS).
    Ordre des valeurs : trame par trame (comme le code de référence DCASE).
    """
    n_vectors = logmel.shape[1] - N_FRAMES + 1
    vectors = np.empty((n_vectors, DIMS), dtype=np.float32)
    for t in range(N_FRAMES):
        vectors[:, N_MELS * t: N_MELS * (t + 1)] = logmel[:, t: t + n_vectors].T
    return vectors


def file_to_vectors(file_path):
    return logmel_to_vectors(file_to_logmel(file_path))


def load_dataset_index(data_root):
    """
    Liste des fichiers .wav de dev_data sous forme de DataFrame (un fichier par ligne).
    Les chemins sont reconstruits à partir de data_root : le CSV de 2022 contenait des
    chemins relatifs avec des séparateurs Windows.
    """
    rows = []
    for path in sorted(Path(data_root).glob('*/*/*.wav')):
        condition, _, machine_id, sample_id = path.stem.split('_')
        rows.append({'pathname': str(path),
                     'machine_type': path.parent.parent.name,
                     'data_split': path.parent.name,
                     'condition': condition,
                     'machine_id': 'id_' + machine_id,
                     'sample_id': sample_id})
    return pd.DataFrame(rows)


def extract_logmels(df, cache_file, n_jobs=-1):
    """
    Log-mel de tous les fichiers de df, empilés en (N, N_MELS, T).
    Résultat mis en cache dans cache_file (.npy, float16 pour diviser la place par 2).
    """
    cache_file = Path(cache_file)
    if cache_file.exists():
        return np.load(cache_file).astype(np.float32)
    logmels = Parallel(n_jobs=n_jobs, batch_size=16)(
        delayed(file_to_logmel)(p) for p in df.pathname)
    t_min = min(l.shape[1] for l in logmels)
    stacked = np.stack([l[:, :t_min] for l in logmels])
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, stacked.astype(np.float16))
    return stacked


#---------------------------------------------------------------------------------------------
# Modèle
#---------------------------------------------------------------------------------------------
def build_autoencoder(input_dims=DIMS):
    """Autoencodeur dense de la référence DCASE 2020 (128 x4, goulot de 8, 128 x4)."""
    import keras
    from keras import layers

    inputs = keras.Input(shape=(input_dims,))
    h = inputs
    for units in [128, 128, 128, 128, 8, 128, 128, 128, 128]:
        h = layers.Dense(units)(h)
        h = layers.BatchNormalization()(h)
        h = layers.Activation('relu')(h)
    outputs = layers.Dense(input_dims)(h)   # sortie linéaire : le log-mel n'est pas borné
    model = keras.Model(inputs, outputs, name='dcase2020_autoencoder')
    # steps_per_execution : moins d'allers-retours Python par lot (~25 % plus rapide sur CPU)
    model.compile(optimizer='adam', loss='mean_squared_error', steps_per_execution=50)
    return model


class WindowSequence:
    """
    Générateur de lots de fenêtres tirées au hasard dans les log-mel (N, N_MELS, T),
    pour ne pas matérialiser toutes les fenêtres en mémoire (plusieurs Go par machine).
    """
    def __init__(self, logmels, batch_size=512, seed=0):
        self.logmels = logmels
        self.batch_size = batch_size
        self.n_files, _, t = logmels.shape
        self.n_starts = t - N_FRAMES + 1
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        """Nombre de lots pour voir en moyenne chaque fenêtre une fois."""
        return int(np.ceil(self.n_files * self.n_starts / self.batch_size))

    def batch(self):
        f = self.rng.integers(0, self.n_files, self.batch_size)
        t = self.rng.integers(0, self.n_starts, self.batch_size)
        x = np.concatenate([self.logmels[f, :, t + k] for k in range(N_FRAMES)], axis=1)
        return x

    def as_dataset(self):
        """
        Même tirage que batch(), mais en opérations TensorFlow : un générateur Python
        plafonnait à ~30 ms par lot, soit ~10 h pour entraîner les 6 machines.
        """
        import tensorflow as tf
        # (N, T, N_MELS) : une trame = un vecteur de N_MELS valeurs contigu
        frames = tf.constant(np.ascontiguousarray(self.logmels.transpose(0, 2, 1)))
        n_files, n_starts, batch_size = self.n_files, self.n_starts, self.batch_size

        def make_batch(_):
            f = tf.random.uniform([batch_size], 0, n_files, dtype=tf.int32)
            t = tf.random.uniform([batch_size], 0, n_starts, dtype=tf.int32)
            x = tf.concat([tf.gather_nd(frames, tf.stack([f, t + k], axis=1)) for k in range(N_FRAMES)], axis=1)
            return x, x

        return (tf.data.Dataset.from_tensors(0).repeat()
                .map(make_batch, num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))


def file_scores(model, logmels, batch_size=4096):
    """Score d'anomalie de chaque fichier : erreur quadratique moyenne de ses fenêtres."""
    scores = np.empty(len(logmels), dtype=np.float32)
    for i, logmel in enumerate(logmels):
        x = logmel_to_vectors(logmel)
        x_hat = model.predict_on_batch(x) if len(x) <= batch_size else model.predict(x, batch_size=batch_size, verbose=0)
        scores[i] = np.mean(np.square(x - np.asarray(x_hat)))
    return scores


def window_errors(model, logmels, files_per_batch=64):
    """
    Erreur de reconstruction de chaque fenêtre de chaque fichier : (N, n_fenêtres).
    Plusieurs fichiers sont prédits ensemble pour aller plus vite.
    """
    n_windows = logmels.shape[2] - N_FRAMES + 1
    errors = np.empty((len(logmels), n_windows), dtype=np.float32)
    for start in range(0, len(logmels), files_per_batch):
        chunk = logmels[start:start + files_per_batch]
        x = np.concatenate([logmel_to_vectors(l) for l in chunk])
        x_hat = np.asarray(model.predict(x, batch_size=8192, verbose=0))
        errors[start:start + len(chunk)] = np.mean(np.square(x - x_hat), axis=1).reshape(len(chunk), n_windows)
    return errors


# Façons de résumer les erreurs par fenêtre en un score par fichier.
# 'moyenne' est la règle de la référence DCASE ; les autres donnent plus de poids
# aux passages courts mais très mal reconstruits.
AGGREGATIONS = {
    'moyenne': lambda e: e.mean(axis=1),
    'centile 90': lambda e: np.percentile(e, 90, axis=1),
    'centile 99': lambda e: np.percentile(e, 99, axis=1),
    'maximum': lambda e: e.max(axis=1),
}


def train_machine(machine_type, df, feature_dir, model_dir, epochs=40, batch_size=512,
                  val_fraction=0.1, steps_fraction=0.2, seed=0, verbose=2, machine_id=None):
    """
    Entraîne (ou recharge) l'autoencodeur d'un type de machine sur ses fichiers `train`.
    Avec machine_id (ex. 'id_00'), le modèle n'est entraîné que sur cette machine.
    10 % des fichiers de train (tirés au hasard) servent de validation : arrêt précoce
    et calcul du seuil de décision. Le jeu `test` n'intervient pas ici.

    Budget de calcul : une « epoch » couvre steps_fraction (20 %) des fenêtres, soit au
    plus 40 x 0,2 = 8 passages complets, contre 100 dans la référence officielle. Sur un
    CPU (TensorFlow n'utilise pas le GPU sous Windows), un pas coûte ~18 ms : 100 passages
    complets demanderaient ~10 h pour les 6 machines, contre ~30 min ici.
    Renvoie (model, info) où info contient le seuil et l'historique.
    """
    import keras

    model_dir = Path(model_dir)
    name = machine_type if machine_id is None else f'{machine_type}_{machine_id}'
    model_path = model_dir / f'{name}.keras'
    info_path = model_dir / f'{name}.json'

    train_df = df[(df.machine_type == machine_type) & (df.data_split == 'train')].reset_index(drop=True)
    logmels = extract_logmels(train_df, Path(feature_dir) / f'{machine_type}_train.npy')
    if machine_id is not None:
        logmels = logmels[(train_df.machine_id == machine_id).values]

    if model_path.exists() and info_path.exists():
        return keras.models.load_model(model_path), json.loads(info_path.read_text())

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(logmels))
    n_val = int(len(logmels) * val_fraction)
    val_idx, fit_idx = perm[:n_val], perm[n_val:]

    keras.utils.set_random_seed(seed)
    model = build_autoencoder()
    train_seq = WindowSequence(logmels[fit_idx], batch_size=batch_size, seed=seed)
    val_seq = WindowSequence(logmels[val_idx], batch_size=batch_size, seed=seed + 1)
    callbacks = [keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                 keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)]
    steps = max(50, int(len(train_seq) * steps_fraction) // 50 * 50)
    val_steps = max(50, int(len(val_seq) * steps_fraction) // 50 * 50)
    history = model.fit(train_seq.as_dataset(), steps_per_epoch=steps,
                        validation_data=val_seq.as_dataset(), validation_steps=val_steps,
                        epochs=epochs, callbacks=callbacks, verbose=verbose)

    val_scores = file_scores(model, logmels[val_idx])
    info = {
        'machine_type': machine_type,
        'machine_id': machine_id,
        'threshold': float(np.percentile(val_scores, THRESHOLD_PERCENTILE)),
        'threshold_rule': f'{THRESHOLD_PERCENTILE}e centile des scores des fichiers de validation (train, sons normaux)',
        'n_train_files': int(len(fit_idx)),
        'n_val_files': int(n_val),
        'n_frames_per_file': int(logmels.shape[2]),
        'epochs_run': len(history.history['loss']),
        'steps_per_epoch': steps, 'batch_size': batch_size,
        'history': {k: [float(v) for v in vals] for k, vals in history.history.items()},
        'features': {'sr': SR, 'n_mels': N_MELS, 'n_fft': N_FFT, 'hop_length': HOP_LENGTH,
                     'n_frames': N_FRAMES, 'power': POWER},
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    info_path.write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding='utf-8')
    return model, info


#---------------------------------------------------------------------------------------------
# Évaluation
#---------------------------------------------------------------------------------------------
def evaluate_machine(machine_type, model, info, df, feature_dir):
    """
    Scores des fichiers `test` d'un type de machine, puis AUC / pAUC par identifiant.
    Renvoie (scores_df, metrics_df).
    """
    test_df = df[(df.machine_type == machine_type) & (df.data_split == 'test')].reset_index(drop=True)
    logmels = extract_logmels(test_df, Path(feature_dir) / f'{machine_type}_test.npy')
    test_df['score'] = file_scores(model, logmels)
    test_df['y_true'] = (test_df.condition == 'anomaly').astype(int)
    test_df['y_pred'] = (test_df.score > info['threshold']).astype(int)

    rows = []
    for machine_id, g in test_df.groupby('machine_id'):
        rows.append(_metrics_row(machine_type, machine_id, g))
    return test_df, pd.DataFrame(rows)


def _metrics_row(machine_type, machine_id, g):
    tp = int(((g.y_pred == 1) & (g.y_true == 1)).sum())
    fp = int(((g.y_pred == 1) & (g.y_true == 0)).sum())
    fn = int(((g.y_pred == 0) & (g.y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {'machine_type': machine_type, 'machine_id': machine_id,
            'AUC': roc_auc_score(g.y_true, g.score),
            'pAUC': roc_auc_score(g.y_true, g.score, max_fpr=MAX_FPR),
            'precision': precision, 'recall': recall,
            'F1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            'n_normal': int((g.y_true == 0).sum()), 'n_anomaly': int((g.y_true == 1).sum())}


def summarize(metrics_df):
    """Moyenne par type de machine (comme dans les résultats officiels DCASE) et moyenne générale."""
    by_type = metrics_df.groupby('machine_type')[['AUC', 'pAUC', 'precision', 'recall', 'F1']].mean()
    by_type.loc['Moyenne'] = by_type.mean()
    return by_type


#---------------------------------------------------------------------------------------------
# Inférence sur un fichier (utilisé par l'application Streamlit)
#---------------------------------------------------------------------------------------------
def predict_file(file_path, model, info):
    """Renvoie (score, est_anomalie, logmel, reconstruction_moyenne_par_trame)."""
    logmel = file_to_logmel(file_path)
    x = logmel_to_vectors(logmel)
    x_hat = np.asarray(model.predict(x, batch_size=4096, verbose=0))
    score = float(np.mean(np.square(x - x_hat)))
    # erreur par fenêtre, utile pour voir *quand* le son devient anormal
    errors = np.mean(np.square(x - x_hat), axis=1)
    return score, score > info['threshold'], logmel, errors
