import streamlit as st
import pandas as pd
import numpy as np

import librosa
import librosa.display
import matplotlib.pyplot as plt


title = "PySaDML."
sidebar_name = "Introduction"

@st.cache
def get_audio_path(id):
    df = pd.read_csv('./../data/dev_data.csv', dtype={"machine_id": "str", "sample_id": "str"})
    audio_path = df.pathname[id]
    return audio_path
    

def run():

    # TODO: choose between one of these GIFs
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/1.gif")
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/2.gif")
    st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/3.gif")
            
    st.title(title)

    st.markdown("-------")

    st.markdown(
        """
        **Ce projet consiste à la détection de son anormal dans les pièces industrielles.**
        
        Dans la mesure où seules des données sonores de machines normales sont disponibles, il s'agit d'utiliser l'intelligence artificielle pour pouvoir classer une machine dans un état normal ou anormal.
        
        Une des méthodes qui pourraient permettre de détecter les machines défaillantes est l’analyse du son qu’elles émettent pour identifier s’il est normal ou anormal. 
        
        Le principal défi de cette tâche est de détecter des sons anormaux inconnus dans la mesure où seuls des échantillons de sons normaux ont été fournis comme données d'entraînement.
                
        Les données utilisées pour ce projet comprennent des sons de 6 machines différentes : 
        
        Les données sonores `ToyADMOS` issues de 2 jouets (train, voiture) et les données `MIMII` de 4 machines réelles (pompe, ventilo etc..)
        
            - Toy-car (ToyADMOS)
        
            - Toy-conveyor (ToyADMOS)
        
            - Valve (MIMII)
        
            - Pump (MIMII)
        
            - Fan (MIMII)
        
            - Slide rail (MIMII)


        **Nous avons étudiez 2 approches :**
        
        1 - L'approche supervisée de Machine Learning : classification par type de machine
        
        2 - L'approche non supervisée de Deep Learning : L'AutoEncoder (AE)
        
        """
    )

    st.header("Alors qu'est ce donc un AE ?")
    
    st.markdown(
        """
        un auto-encodeur est un type d’algorithme d’apprentissage non supervisé utilisé notamment dans le deep learning. 
        Il est donc constitué d’un réseau neuronal profond permettant de construire une nouvelle représentation de données.
        """
    )
    st.markdown(
        """
        **Comment ça fonctionne ?**\n
        L’architecture d’un auto-encodeur se compose de deux ensembles de couches de neurones. 
        Le premier ensemble forme ce qui est appelé l’encodeur qui traite les données d’entrées pour construire de nouvelles représentations code. 
        Le deuxième ensemble, dit décodeur, tente de reconstruire les données à partir de ce code.
        """
    )

    col1, col2, col3 = st.columns([1,2,1])

    with col1:
        st.write("")
    with col2:
        st.image('assets/autoencoder.jpg')
    with col3:
        st.write("")


    st.markdown(
        """
        De ce fait, la performance d’un auto-encodeur se mesure par les différences entre les données d’entrées et les données de sortie. Par ailleurs, pour entraîner l’algorithme, ses paramètres sont modifiés de manière à réduire l’erreur de reconstruction.

        **L’encodeur**
        
        Pour faire simple, l’encodeur sert à compresser le jeu de données d’entrée en une représentation plus petite. À cette fin, il extrait les caractéristiques (features) les plus importantes à partir de données initiales. Cela aboutit à une formation compacte appelée bottleneck aussi connu comme l’espace latent.

        **Le décodeur**
        
        À l’inverse de l’encodeur, le décodeur décompresse le bottleneck pour reconstituer les données. Son défi consiste à utiliser les caractéristiques contenues dans le vecteur condensé pour essayer de reconstruire le plus fidèlement possible le jeu de données.

        **L’espace latent**
        
        L’espace latent correspond donc aux données compressées, autrement dit à l’espace entre l’encodeur et le décodeur. Le but de la création de cet espace latent est de limiter le flux d’information entre les deux composants de l’auto-encodeur. Cette limitation se traduit par la suppression du bruit afin de ne laisser passer que les informations importantes.
        
        """
    )

    
    st.header("Qu'est ce donc un son ?")
    
    st.markdown(
        """
        Le **son** est une onde produite par la vibration mécanique d'un support fluide, solide ou gazeux et propagée grâce à l'élasticité du milieu environnant.
        Par anthropomorphisme, nous pouvons définir le son comme représentant la partie audible du spectre des vibrations acoustiques, tout comme la lumière est définie comme la partie visible du spectre des vibrations électromagnétiques.
        Il est représenté sous la forme d'un signal audio exprimé en fonction de l'amplitude et du temps.
        
        Or tout être vivant doté d'une ouïe ne peut percevoir qu'une partie du spectre sonore (=plage de fréquences), par exemple, l'oreille humaine moyenne ne perçoit les sons que dans une certaine plage de fréquences située environ entre 20 Hz et 20 kHz.
        
        Par conséquent l'analyse du son se fera dans le domaine des fréquences dans l'échelle des mel.
        
        **Schématiquement cela ressemble à ceci :**
        """
    )
    
    #################################################################################
    # Paramètres des fonctions audio Librosa
    #################################################################################
    n_mfcc=40
    n_mels=64
    n_fft=2048
    hop_length=512
    #################################################################################
    audio_path = get_audio_path(30000)
    y, sr = librosa.load(audio_path, sr=None, res_type='kaiser_fast')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig, ax = plt.subplots()
        librosa.display.waveshow(y=y, alpha=0.5, sr=sr, x_axis='s', linewidth=0.5, color='red')
        ax.set_ylabel('Amplitude')
        st.pyplot(fig)
    with col2:
        # fig, ax = plt.subplots()
        # S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        # plt.plot(S, color='red', alpha=0.5)
        # ax.set_xlabel('Fréquence (Hz)')
        # ax.set_ylabel('Amplitude')
        # st.pyplot(fig)
        
        import scipy as sp
        fig, ax = plt.subplots()
        #derive spectrum using FT
        ft = sp.fft.fft(y)
        magnitude = np.absolute(ft)
        frequency = np.linspace(0, sr, len(magnitude)) 
        #plot spectrum
        plt.plot(frequency[:len(magnitude)//2], magnitude[:len(magnitude)//2]) # magnitude spectrum
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        st.pyplot(fig)
        
    with col3:
        fig, ax = plt.subplots()
        S_scale = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        Y_scale = np.abs(S_scale) ** 2
        Y_log_scale = librosa.power_to_db(Y_scale)
        librosa.display.specshow(Y_log_scale, sr=sr, hop_length=hop_length, y_axis='linear', x_axis='s', cmap='viridis', ax=ax)
        st.pyplot(fig)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("")
    with col2:
        fig, ax = plt.subplots()
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, n_mels=n_mels, n_fft=n_fft)        
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)       
        librosa.display.specshow(mel_spectrogram, sr=sr, hop_length=hop_length, y_axis='log', x_axis='s', cmap='viridis', ax=ax)
        ax.set_ylabel('Fréquence (Hz)')
        st.pyplot(fig)
        
    with col3:
        st.write("")
