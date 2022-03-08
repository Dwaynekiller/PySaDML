import streamlit as st
import pandas as pd
import numpy as np

import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import time

title = "Prédiction des données"
sidebar_name = "Test prédiction"
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
    df = df[(df.data_split == TEST_SUBFOLDER)]
    return df

def load_model_ae():
    model_ae = tf.keras.models.load_model('./../models/weight_train/Autoencoder/model_1/model_1')
    return model_ae
    
@st.cache
def load_model_svm():
    model_svm = pickle.load(open("./../models/weight_train/Machine_L/model_svm", 'rb'))
    return model_svm
    
def load_model_cnn():
    model_cnn = tf.keras.models.load_model('./../models/weight_train/Deep_L')
    return model_cnn
    

def run():

    st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/3.gif")

    st.title(title)

    st.image("assets/PySaDML_process.png")

    df = get_data()
    
    machine_list = np.append('Aucun', df.machine_type.unique())
    condition_list = np.append('Aucun', df.condition.unique())
    
    machine_selected = st.radio("Choix du type de machine",machine_list)
    
    condition_selected = st.radio("Choix de la condition de la machine",condition_list)
    
    audio_path_selected = ''
    audio_selected = 'Aucun'
    model_choisi = 'Aucun'
    prediction = ''
    
    if (machine_selected != 'Aucun') and (condition_selected != 'Aucun'):
        audio_list = np.append('Aucun', df.filename.loc[(df.machine_type == machine_selected) & (df.condition == condition_selected)])
        audio_selected = st.selectbox(label = 'Sélection de la machine à prédire:', options = audio_list)
    else:    
        audio_selected = st.selectbox(label = 'Sélection de la machine à prédire:', options = ["Aucun"])

    if (audio_selected != 'Aucun'):
        model_choisi = st.selectbox(label = "Modèles de classification", options = ["Aucun", "Autoencoder", "Machine Learning", "Deep Learning"])
    else:
        model_choisi = st.selectbox(label = "Modèles de classification", options = ["Aucun"])
        
    if model_choisi != 'Aucun':
        audio_path_selected = df.pathname.loc[(df.filename == audio_selected) & (df.machine_type == machine_selected)].iloc[0]
        y, sr = librosa.load(audio_path_selected, sr=None, duration=duration, res_type='kaiser_fast')
        vector = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

        prediction = True #st.button('Predict')
        if prediction:
            if model_choisi == "Machine Learning":
                model = load_model_svm()
                vector_svm = vector.reshape(1,-1)
                pred = model.predict(vector_svm)
            if model_choisi == "Deep Learning":
                model = load_model_cnn()
                vector_cnn = vector.reshape(-1,vector.shape[0],vector.shape[1],1)
                pred = model.predict(vector_cnn)
            if model_choisi == "Autoencoder":
                model = load_model_ae()
                vector_ae = vector.reshape(1,-1)
                vector_ae = vector_ae[:,:4096]
                pred = model.predict(vector_ae)
                mse = np.mean(np.power(vector_ae - pred, 2), axis=1)

            st.progress(0)
            with st.spinner('Predicting'):
                time.sleep(2)
               
            if model_choisi == "Autoencoder":
                if mse < 0.7:
                    st.write("Le son est normale")
                else:
                     st.write("Le son est anormale")
            else :
                if pred[0] == machine_selected:
                    st.write("La machine est normale")
                else:
                     st.write("La machine est anormale")
            

    if prediction and model_choisi == "Autoencoder":
        col1, col2 = st.columns(2)
        with col1:
            vector_db = librosa.power_to_db(vector, ref=np.max)
            fig, ax = plt.subplots()
            librosa.display.specshow(vector_db, sr=sr, hop_length=hop_length, y_axis='log', x_axis='s', cmap='viridis', ax=ax)
            ax.set(title="Audio sélectionné")
            st.pyplot(fig)
        with col2:
            vector_db_pred = librosa.power_to_db(pred.reshape(vector.shape[0],vector.shape[0]), ref=np.max)
            fig, ax = plt.subplots()
            librosa.display.specshow(vector_db_pred, sr=sr, hop_length=hop_length, y_axis='log', x_axis='s', cmap='viridis', ax=ax)
            ax.set(title='Audio prédit')
            st.pyplot(fig)

