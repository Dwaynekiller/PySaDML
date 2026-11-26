"""
Remise en état des modèles de la soutenance 2022 pour l'application Streamlit.

Trois problèmes empêchaient ces modèles de donner des prédictions fiables :

1. Les StandardScaler ajustés à l'entraînement n'avaient pas été sauvegardés.
   On les recalcule ici sur tous les fichiers `train` (en 2022 : 80 % tirés au
   hasard, les statistiques sont quasi identiques), avec exactement les mêmes
   caractéristiques qu'en 2022.
2. Les modèles Keras étaient au format SavedModel de TensorFlow 2.x, que Keras 3
   ne sait plus recharger. On reconstruit leur architecture (déduite des poids et
   des notebooks) et on y recopie les poids enregistrés.
3. Le CNN (LeNet) avait un ordre de classes propre au notebook, différent de
   l'ordre alphabétique supposé par l'application.

Modèles concernés :
- SVM (`Machine_L/model_svm`, notebook « Classification by type ML », cellule 17) :
  classifieur de type de machine, mel 64 bandes en dB, StandardScaler.
- CNN (`Deep_L`, notebook « Classification by type CNN Lenet ») : LeNet, classifieur
  de type de machine, entrée power_to_db(mel**2), StandardScaler.
- Autoencodeur (`Autoencoder/model_1`, notebook « Auto encodeur FINAL », 1er modèle) :
  4096 premières valeurs du mel en dB aplati, StandardScaler.

Usage (depuis le dossier notebooks) :
    python -m include.legacy_2022
"""
from pathlib import Path

import numpy as np
import librosa
from joblib import Parallel, delayed

from include import dcase_baseline as dcase

N_MELS, N_FFT, HOP_LENGTH, DURATION = 64, 2**13, 2**11, 10.0
AE_INPUT = 4096
# Ordre des classes du LeNet (notebook « Classification by type CNN Lenet », cellule 35)
LENET_CLASSES = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
THRESHOLD_AE = 0.7

SCALERS_FILE = 'scalers_2022.npz'
AE_FILE = 'autoencoder_2022.keras'
LENET_FILE = 'lenet_2022.keras'


def mel_features(path):
    """Renvoie (mel_db, mel2_db), chacun de forme (64, 79) pour 10 s."""
    y, sr = librosa.load(path, sr=None, duration=DURATION, res_type='kaiser_fast')
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)           # SVM et autoencodeur
    mel2_db = librosa.power_to_db(mel ** 2, ref=np.max)     # LeNet (choix du notebook)
    return mel_db.astype(np.float32), mel2_db.astype(np.float32)


def _std(X):
    mean, scale = X.mean(axis=0), X.std(axis=0)
    scale[scale == 0] = 1.0   # même convention que StandardScaler
    return mean, scale


def compute_scalers(data_root):
    df = dcase.load_dataset_index(data_root)
    paths = df.pathname[df.data_split == 'train'].tolist()
    feats = Parallel(n_jobs=-1, batch_size=32)(delayed(mel_features)(p) for p in paths)
    mel = np.stack([f[0].reshape(-1) for f in feats]).astype(np.float64)
    mel2 = np.stack([f[1].reshape(-1) for f in feats]).astype(np.float64)
    mel_mean, mel_scale = _std(mel)
    mel2_mean, mel2_scale = _std(mel2)
    return {'mel_mean': mel_mean, 'mel_scale': mel_scale,
            'mel2_mean': mel2_mean, 'mel2_scale': mel2_scale, 'n_files': len(paths)}


def _checkpoint_weights(savedmodel_dir):
    """Poids (kernel, bias) des couches, dans l'ordre, lus dans le checkpoint du SavedModel."""
    import tensorflow as tf
    ckpt = str(Path(savedmodel_dir) / 'variables' / 'variables')
    reader = tf.train.load_checkpoint(ckpt)
    weights, i = [], 0
    while True:
        key = f'layer_with_weights-{i}/kernel/.ATTRIBUTES/VARIABLE_VALUE'
        if not reader.has_tensor(key):
            return weights
        weights.append((reader.get_tensor(key),
                        reader.get_tensor(f'layer_with_weights-{i}/bias/.ATTRIBUTES/VARIABLE_VALUE')))
        i += 1


def rebuild_autoencoder(savedmodel_dir):
    import keras
    from keras import layers
    model = keras.Sequential([keras.Input(shape=(AE_INPUT,))] +
                             [layers.Dense(u, activation='relu') for u in [64, 32, 16, 8, 16, 32, 64]] +
                             [layers.Dense(AE_INPUT, activation='sigmoid')], name='autoencoder_2022')
    for layer, (k, b) in zip([l for l in model.layers if l.weights], _checkpoint_weights(savedmodel_dir)):
        layer.set_weights([k, b])
    return model


def rebuild_lenet(savedmodel_dir):
    import keras
    from keras import layers
    model = keras.Sequential([
        keras.Input(shape=(N_MELS, 79, 1)),
        layers.Conv2D(30, (5, 5), padding='valid', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (3, 3), padding='valid', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(LENET_CLASSES), activation='softmax'),
    ], name='lenet_2022')
    for layer, (k, b) in zip([l for l in model.layers if l.weights], _checkpoint_weights(savedmodel_dir)):
        layer.set_weights([k, b])
    return model


def main(data_root='../data/dev_data', legacy_dir='../models/weight_train'):
    legacy_dir = Path(legacy_dir)
    rebuild_autoencoder(legacy_dir / 'Autoencoder' / 'model_1' / 'model_1').save(legacy_dir / AE_FILE)
    rebuild_lenet(legacy_dir / 'Deep_L').save(legacy_dir / LENET_FILE)
    print('modèles reconstruits :', legacy_dir / AE_FILE, legacy_dir / LENET_FILE)
    if not (legacy_dir / SCALERS_FILE).exists():
        np.savez(legacy_dir / SCALERS_FILE, **compute_scalers(data_root))
    print('scalers :', legacy_dir / SCALERS_FILE)


#---------------------------------------------------------------------------------------------
# Inférence (application Streamlit)
#---------------------------------------------------------------------------------------------
def load_scalers(legacy_dir):
    return dict(np.load(Path(legacy_dir) / SCALERS_FILE))


def predict_svm(svm, mel_db, scalers):
    x = (mel_db.reshape(1, -1) - scalers['mel_mean']) / scalers['mel_scale']
    return str(svm.predict(x)[0])


def predict_lenet(model, mel2_db, scalers):
    x = (mel2_db.reshape(1, -1) - scalers['mel2_mean']) / scalers['mel2_scale']
    proba = np.asarray(model.predict(x.reshape(1, N_MELS, -1, 1), verbose=0))[0]
    return LENET_CLASSES[int(proba.argmax())], proba


def predict_autoencoder(model, mel_db, scalers):
    x = mel_db.reshape(1, -1)[:, :AE_INPUT]
    x = (x - scalers['mel_mean'][:AE_INPUT]) / scalers['mel_scale'][:AE_INPUT]
    x_hat = np.asarray(model.predict(x, verbose=0))
    return float(np.mean(np.square(x - x_hat)))


if __name__ == '__main__':
    main()
