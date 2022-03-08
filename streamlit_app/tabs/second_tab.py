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


#################################################################################
# Paramètres des fonctions audio Librosa
#################################################################################
n_mfcc1=40
n_mels1=64
n_fft1=2048
hop_length1=512
#################################################################################
AUDIO_DEV_FOLDER = './../data/dev_data'
AUDIO_DEV_PATH = Path(AUDIO_DEV_FOLDER)
#################################################################################
# Les dossiers dans lesquels se trouvent les audios.
#################################################################################
TRAIN_SUBFOLDER = 'train'
TEST_SUBFOLDER = 'test'
#################################################################################

title = "Analyse Exploratoire des Données"
sidebar_name = "EDA"
st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache
def get_data():
    df = pd.read_csv('./../data/dev_data.csv', dtype={"machine_id": "str", "sample_id": "str"})
    df = df.drop(['sample_id','audio_format', 'machine_kind'], axis=1)
    return df.sample(frac=1, random_state=1).reset_index(drop=True)
    
@st.cache
def load_sound_file(audio_path, duration=None):
    try:
        y, sr = librosa.load(audio_path, sr=None, duration=duration, res_type='kaiser_fast')
        # normalize signal amplitude
        y = librosa.util.normalize(S=y, axis=0)
    
        return y, sr
    except:
        st.write("Le fichier est corrompu ou n'existe pas!! : {}".format(audio_path))
        pass

#@st.cache
def audio_properties(dataframe):
    df = dataframe[['durations']]
    serie = df['durations'].value_counts()
    
    fig, axe = plt.subplots()
    serie.plot(ax=axe, kind='bar')
    axe.set_xlabel('Durée(s)')
    axe.set_ylabel('Nombre de fichier audio')
    axe.set_title('Histogramme des durées audio')
    axe.grid(False)
    st.pyplot(fig)

    df_serie = pd.DataFrame(serie)
    df_serie.reset_index(inplace=True)
    df_serie = df_serie.rename(columns = {'index':'durée', 'durations':'Nombre'})
    tot = df_serie['Nombre'].sum()
    df_serie['Moyenne'] = df_serie['Nombre'].apply(lambda x: 100*(x/tot)).map('{:,.1f}%'.format)
    df_serie['durée'] = df_serie['durée'].map('{:,.1f}s'.format)
    
    st.dataframe(df_serie)
   
    del df, serie, df_serie

def display_audio(audio_path, machine_type, condition, duration=None):
    file_path = Path(audio_path)

    y, sr = load_sound_file(file_path, duration=duration)
        
    if condition == 'normal':
        color = 'blue'
    elif condition == 'anomaly':
        color = 'red'
    else:
        color = 'green'
        
    fig, axe = plt.subplots()
    fig = plt.figure(figsize=(10, 5))
    librosa.display.waveshow(y, alpha=0.5, sr=sr, x_axis='s', color=color, linewidth=0.5)
    plt.title(machine_type + " - " + file_path.name)
    st.pyplot(fig)
    
    # del y, sr
    return st.audio(audio_path)

@st.cache
def get_mel(file_path, hop_length=512, n_mels=128, n_fft=2048, duration=None):
    try:
        # Load audio file
        y, sr = load_sound_file(file_path, duration=duration)

        # Computing the mel spectrograms
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, n_mels=n_mels, n_fft=n_fft)        

        # Convert to db
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    except Exception as e:
        st.write("Erreur d'analyse du fichier audio: ", e)
        return None
        
    del y

    return mel_spectrogram, sr

@st.cache
def get_mfcc(file_path, hop_length=512, n_mels=128, n_mfcc=20, duration=None):
    try:

        # Load audio file
        y, sr = load_sound_file(file_path, duration=duration)

        # Calculate MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mels=n_mels, n_mfcc=n_mfcc)

    except Exception as e:
        st.write("Erreur d'analyse du fichier audio: ", e)
        return None
    
    del y
    
    return mfcc, sr


def run():

    st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/3.gif")
    
    st.title(title)

    st.markdown(
        """
        ### Histogramme des fichiers audio
        """
    )
    df = get_data()
    
    with st.expander('Afficher les caractéristiques audio'):
        col1, col2 = st.columns([2,1])
    
        with col1:
            st.subheader('Dataset initial')
            st.dataframe(df[['filename', 'machine_id','machine_type','data_split','condition','durations','samplingrate']])

        with col2:
            st.subheader('Statistiques')
            st.dataframe(df.describe().apply(lambda s: s.apply('{0:.2f}'.format)))
            
        st.markdown(
            """
            **On constate que la variable `samplingrate` à la même valeur sur l'ensemble des données de développement.**
            - mean(samplingrate) = 16,000 Hz.
            - max(samplingrate) = 16,000 Hz.
            - min(samplingrate) = 16,000 Hz.
            - std(samplingrate) = 0.

            **On constate que la variable `durations` contient deux valeurs:**

            - max(durations) = 10s.
            - min(durations) = 11s.
            
                """
        )
        st.subheader('Crosstab')
        st.dataframe(pd.crosstab(df['condition'], df['machine_type']))
    with st.expander('Afficher la distribution sur la durée audio'):
        st.write(audio_properties(df))
    
    col1, col2 = st.columns(2)
    with col1:
        fig = sns.catplot(x="data_split", hue="condition", data=df, kind="count")
        st.pyplot(fig)
    with col2:
        fig = sns.catplot(x="machine_type", hue="condition", data=df, kind="count")
        st.pyplot(fig)

    st.subheader('Spectogramme des fichiers audio')
    
         
    machine_list = np.append('Aucun', df.machine_type.unique())
    machine_selected = st.selectbox(label = "Sélectionner une machine pour écouter et afficher son amplitude audio", options = machine_list)
    audio_path_selected = ''
    sec = None
    
    if machine_selected != 'Aucun':
        if st.checkbox('Modifier les hyperparamètres des fonctions librosa'):

            st.markdown(
                """
                **à gauche des "slider" pour sélectionner les variables dans une liste pour afficher les spectrogrammes mel et mfcc.**
                - n_fft = [1024, 2048, 4096, 8192].
                - n_mfcc = [13, 20, 40, 64].
                - hop_length = [512, 1024, 2048, 4096].
                - n_mels = [32, 64, 128].
                """
            )
            if st.checkbox("Tronquer l'audio"):
                sec = st.number_input('Choisir la durée en seconde', min_value=1., max_value=10., step=1.0, value = 5.)
            else:
                sec = None
            n_fft = st.sidebar.select_slider(
                 'Choisir la valeur n_fft',
                 options=[1024, 2048, 4096, 8192], value=(n_fft1))
            n_mfcc = st.sidebar.select_slider(
                 'Choisir la valeur n_mfcc',
                 options=[13, 20, 40, 64], value=(n_mfcc1))

            hop_length = st.sidebar.select_slider(
                 'Choisir la valeur hop_length',
                 options=[512, 1024, 2048, 4096], value=(hop_length1))
            n_mels = st.sidebar.select_slider(
                 'Choisir la valeur n_mels',
                 options=[32, 64, 128], value=(n_mels1))
                 
            normal_audio_path = df.pathname.loc[(df.machine_type == machine_selected) & (df.condition == 'normal')].iloc[0]
            anormal_audio_path = df.pathname.loc[(df.machine_type == machine_selected) & (df.condition == 'anomaly')].iloc[0]
        else :
            n_fft = n_fft1
            n_mfcc = n_mfcc1
            hop_length = hop_length1
            n_mels = n_mels1
            sec = None
            normal_audio_path = df.pathname.loc[(df.machine_type == machine_selected) & (df.condition == 'normal')].sample(1).iloc[0]
            anormal_audio_path = df.pathname.loc[(df.machine_type == machine_selected) & (df.condition == 'anomaly')].sample(1).iloc[0]

        col1, col2 = st.columns(2)
        with col1:
            display_audio(normal_audio_path, machine_selected, 'normal', duration=sec)
        with col2:
            display_audio(anormal_audio_path, machine_selected, 'anomaly', duration=sec)
            
        if st.checkbox("Afficher le spectrogramme des mel d'un son normal et anormal de cette machine "):           
            col1, col2 = st.columns(2)
            with col1:
                vector1, sr1 = get_mel(normal_audio_path, hop_length=hop_length, n_mels=n_mels, n_fft=n_fft, duration=sec)
                fig, ax = plt.subplots()
                librosa.display.specshow(vector1, sr=sr1, hop_length=hop_length, y_axis='log', x_axis='s', cmap='viridis', ax=ax)
                ax.set(title=machine_selected + " - " + Path(normal_audio_path).name + ' ' + str(vector1.shape))
                st.pyplot(fig)
            with col2:
                vector2, sr2 = get_mel(anormal_audio_path, hop_length=hop_length, n_mels=n_mels, n_fft=n_fft, duration=sec)
                fig, ax = plt.subplots()
                librosa.display.specshow(vector2, sr=sr2, hop_length=hop_length, y_axis='log', x_axis='s', cmap='viridis', ax=ax)
                ax.set(title=machine_selected + " - " + Path(anormal_audio_path).name + ' ' + str(vector2.shape))
                st.pyplot(fig)
        
        if st.checkbox("Afficher le spectrogramme des mfcc d'un son normal et anormal de cette machine "):
            
            col1, col2 = st.columns(2)
            with col1:
                vector1, sr1 = get_mfcc(normal_audio_path, hop_length=hop_length, n_mels=n_mels, n_mfcc=n_mfcc, duration=sec)
                fig, ax = plt.subplots()
                librosa.display.specshow(vector1, sr=sr1, hop_length=hop_length, y_axis='log', x_axis='s', ax=ax)
                ax.set(title=machine_selected + " - " + Path(normal_audio_path).name + ' ' + str(vector1.shape))
                st.pyplot(fig)
            with col2:
                vector2, sr2 = get_mfcc(anormal_audio_path, hop_length=hop_length, n_mels=n_mels, n_mfcc=n_mfcc, duration=sec)
                fig, ax = plt.subplots()
                librosa.display.specshow(vector2, sr=sr2, hop_length=hop_length, y_axis='log', x_axis='s', ax=ax)
                ax.set(title=machine_selected + " - " + Path(anormal_audio_path).name + ' ' + str(vector2.shape))
                st.pyplot(fig)

