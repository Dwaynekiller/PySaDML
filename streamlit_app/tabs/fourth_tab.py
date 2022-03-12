import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.style.use('fivethirtyeight')
import seaborn as sns
from PIL import Image
import altair as alt
# for sound analysis
import librosa
import librosa.display
import IPython.display as IP
from pathlib import Path
import tensorflow as tf
import pickle


title = "Détection d'un son normal ou anormal"
sidebar_name = "Prediction"
st.set_option('deprecation.showPyplotGlobalUse', False)

# ----------- Data -------------------
#################################################################################
# Les dossiers dans lesquels se trouvent les audios.
#################################################################################
TRAIN_SUBFOLDER = 'train'
TEST_SUBFOLDER = 'test'
#################################################################################
#################################################################################
# Paramètres des fonctions audio Librosa
#################################################################################
n_mels=64
n_fft=2**13
hop_length=2**11
duration = 10.0
#################################################################################

@st.cache
def get_data():
    """
    This function return a pandas DataFrame with the list of the wav files.
    """

    df = pd.read_csv('./../data/dev_data.csv', dtype={"machine_id": "str", "sample_id": "str"})
    df = df.drop(['sample_id','audio_format', 'machine_kind'], axis=1)
    df = df[(df.data_split == TEST_SUBFOLDER)] # & (df.condition == 'normal')]
    return df.sample(frac=1, random_state=1).reset_index(drop=True)

def load_sound_file(audio_path):
    try:
        signal, sr = librosa.load(audio_path, sr=None, res_type='kaiser_fast')
        # normalize signal amplitude
        signal_norm = librosa.util.normalize(S=signal, axis=0)
        audio = np.array(signal_norm)
        frames = audio.shape[0]
        sec = audio.shape[-1]/sr
    
        return audio, sr, sec
    except:
        st.write("Le fichier est corrompu ou n'existe pas!! : {}".format(audio_path))
        pass

def display_audio(audio_path):
    file_path = Path(audio_path)

    y, sr, _ = load_sound_file(file_path)
        
    del y, sr
    return st.audio(audio_path)

def run():

    st.title(title)

    st.markdown(
        """
        ### Histogramme des fichiers audio
        """
    )
    df = get_data()
    
    # st.write(df)
    model = pickle.load(open("./../models/weight_train/Machine_L/model_svm", 'rb'))
    with st.expander("Choisir les filtres pour afficher l'audio", expanded=True):
        machine_list = np.append('Aucun', df.machine_type.unique())
        machine_selected = st.selectbox(label = 'Sélectionner une machine', options = machine_list)
        audio_path_selected = ''
        if machine_selected != 'Aucun':
            machine_id_list = np.append('Aucun', df[df.machine_type == machine_selected].machine_id.unique())
            machine_id_selected = st.selectbox(label = "Sélectionner l'identifiant de la machine", options = machine_id_list)
            if machine_id_selected != 'Aucun':
                filename_list = np.append('Aucun', df[(df.machine_type == machine_selected) & (df.machine_id == machine_id_selected)].pathname.unique())
                filename_selected = st.selectbox(label = 'Sélectionner le son à prédire:', options = filename_list)
                if filename_selected != 'Aucun':
                    # st.write('./../data/dev_data/'+ machine_selected + '/test/' + filename_selected)
                    # st.audio(display_audio('./../data/dev_data/'+ machine_selected + '/test/' + filename_selected))
                    display_audio(filename_selected)

                    y, sr = librosa.load(filename_selected, sr=None, duration=duration, res_type='kaiser_fast')
                    vector = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
                    if st.button('Predict'):
                        #add warning for image not selected
                        vector = vector.reshape(1,-1)
                        pred = model.predict(vector)

                        import time
                        my_bar = st.progress(0)
                        with st.spinner('Predicting'):
                            time.sleep(2)

                        if pred[0] == machine_selected:
                            st.success("La machine est normale")
                            # return st.write("La machine est normale")
                        else:
                            st.error("La machine est anormale")
                            # return st.write("La machine est anormale")
                        # st.write(pred[0])


    # st.subheader('Please upload an image to identify the digit')
    # file_uploader = st.file_uploader("Select an image for Classification", type="wav")
    # print(file_uploader)
    # model = pickle.load(open("./../models/weight_train/Machine_L/model_svm", 'rb'))
    # if file_uploader:
        # y, sr = librosa.load(file_uploader, sr=None, duration=duration, res_type='kaiser_fast')
        # vector = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    # if st.button('Predict'):
        ##add warning for image not selected
        # vector = vector.reshape(1,-1)
        # pred = model.predict(vector)

        # import time
        # my_bar = st.progress(0)
        # with st.spinner('Predicting'):
            # time.sleep(2)

        # st.write(pred[0])