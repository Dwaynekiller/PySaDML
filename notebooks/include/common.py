#---------------------------------------------------------------------------------------------
# import python-library
#---------------------------------------------------------------------------------------------
# default
import scipy
import numpy as np           # linear algebra
import pandas as pd          # data processing, CSV file I/O (e.g. pd.read_csv)
import time, datetime
import itertools

from pathlib import Path

# for sound analysis
import librosa
import librosa.display
import IPython.display as IP

from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

# for plotting
import matplotlib.pyplot as plt
import seaborn as sns
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
# load_sound_file
#---------------------------------------------------------------------------------------------
def load_sound_file(audio_path, offset=0.0, duration=None):
    try:
        y, sr = librosa.load(audio_path, sr=None, offset=offset, duration=duration, res_type='kaiser_fast')
        y = librosa.util.normalize(S=y, axis=0)
        y = np.array(y)
    
        return y, sr
    except:
        print("Le fichier est corrompu ou n'existe pas!! : {}".format(audio_path))
        pass
#---------------------------------------------------------------------------------------------
 
#---------------------------------------------------------------------------------------------
# load_metadata
#---------------------------------------------------------------------------------------------
def load_metadata(data_path, ext):
    """ 
    This program creates the dataframe and saves it as a csv file
    Function that will create a dataframe with all the file paths
    :attrib df will contain the created dataframe
    This function returns the attrib df
    Loads a sound file
    
    PARAMS
    ======
        data_path : dataframe
        ext : float
            extension file (*.csv)
    
    RETURNS
    =======
        df : (dataframe)
    """
    df = []    
    metadata = data_path.name + ext
    metadata_path = Path(data_path.parent, metadata)
        
    if Path.exists(metadata_path):
        print("Chargement du fichier <{}> des m??tadonn??es audio...".format(metadata))
        with open(metadata_path, 'rb') as f:
            # Load dataset
            if ext == '.ftr':
                df = pd.read_feather(f, use_threads=True);
            if ext == '.csv':
                df = pd.read_csv(f, dtype={"machine_id": "str", "sample_id": "str"});
        print('Termin??.')      
    else:
        print("Cr??ation du fichier <{}> des m??tadonn??es audio...".format(metadata))
        start_time = time.time()
        
        df = pd.DataFrame()
        df['pathname'] = sorted(data_path.glob('*/*/*id*.wav'))
        df['filename'] = df.pathname.map(lambda f: f.name)
        df['name'] = df.pathname.map(lambda f: f.name.split('_'))
        df['machine_id'] = df.name.map(lambda f: 'id_'+f[2])
        df['sample_id'] = df.name.map(lambda f: f[3][:-4])
        df['audio_format'] = df.pathname.map(lambda f: f.suffix)
        df['machine_type'] = df.pathname.map(lambda f: f.parent.parent.name)
        df['machine_kind'] = df.pathname.map(lambda f: 'toys' if 'toy' in f.parent.parent.name else 'real_machine')
        df['data_split'] = df.pathname.map(lambda f: f.parent.name)
        df['condition'] = df.name.map(lambda f: f[0])
        df['infos'] = df.pathname.apply(lambda f: load_sound_file(f))
        df['durations'] = df.infos.map(lambda x: x[0].shape[-1]/x[1])
        df['samplingrate'] = df.infos.map(lambda x: x[1])
        del df['infos']
        del df['name']
        
        df = df.astype(dtype= {"sample_id":"object","samplingrate":"int16","durations":"float64"})
        df.sample(frac=1, random_state=100).reset_index(drop=True)
        
        with open(metadata_path, 'wb') as f:
            # Save dataset
            if ext == '.ftr':
                df.to_feather(f);
            else:
                df.to_csv(f, index=False);
        
        end_time = time.time()
        duree = int(end_time - start_time)
        print("Cr??ation du fichier csv termin?? avec succ??s.")
        print("Dur??e : {}s ".format(datetime.timedelta(seconds = duree)))

    return df.sample(frac=1, random_state=6472).reset_index(drop=True)
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
# audio_properties
#---------------------------------------------------------------------------------------------
def audio_properties(dataframe):
    df = dataframe[['durations']]
    serie = df['durations'].value_counts()
    
    serie.plot(kind='bar')
    plt.xlabel('Dur??e(s)')
    plt.ylabel('Nombre de fichier audio')
    plt.title('Histogramme des dur??es audio')
    plt.grid(False)
    plt.show()

    df_serie = pd.DataFrame(serie)
    df_serie.reset_index(inplace=True)
    df_serie = df_serie.rename(columns = {'index':'dur??e', 'durations':'Nombre'})
    tot = df_serie['Nombre'].sum()
    df_serie['Moyenne'] = df_serie['Nombre'].apply(lambda x: 100*(x/tot)).map('{:,.1f}%'.format)
    df_serie['dur??e'] = df_serie['dur??e'].map('{:,.1f}s'.format)
    
    display(df_serie)
   
    del df, serie, df_serie
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
# display_audio
#---------------------------------------------------------------------------------------------
def display_audio(audio_path, machine_type=None):
    if machine_type is not None:
        print("Machine type: ", machine_type)
        machine_type = machine_type + ' - '
    else:
        machine_type = ''

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(machine_type + "2D repr??sentation des formes d'onde", fontsize=16)

    file_path = Path(audio_path)
    y, sr = load_sound_file(file_path)
    
    condition = file_path.name.split('_')[0]
    
    if condition == 'normal':
        color = 'blue'
    elif condition == 'anomaly':
        color = 'red'
    else:
        color = 'green'
    
    print("Fichier: ", Path(file_path))
    print("Fr??quence d'??chantillonnage: {}".format(sr))
    print("Dur??e: {} seconds".format(y.shape[-1]/sr))
    
    librosa.display.waveshow(y, alpha=0.5, sr=sr, x_axis='s', color=color, linewidth=0.5)
    plt.title(file_path.name)
    
    del y, sr, sec
    # Sound preview
    return IP.Audio(file_path)
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
# display_both_audio
#---------------------------------------------------------------------------------------------
def display_both_audio(normal_path, anormal_path, machine_type=None):
    if machine_type is None:
        machine_type = ''

    fig = plt.figure(figsize=(24, 6))
    fig.suptitle(machine_type + " - 2D repr??sentation des formes d'onde\n\n", fontsize=16)

    plt.subplot(131)
    file_path = Path(normal_path)
    normal_signal, sr = load_sound_file(file_path)
    librosa.display.waveplot(normal_signal, sr=sr, alpha=0.5, color='blue', linewidth=0.5)
    plt.title('Signal normal\n' + file_path.name)

    plt.subplot(132)
    file_path = Path(anormal_path)
    abnormal_signal, sr_an = load_sound_file(file_path)
    librosa.display.waveplot(abnormal_signal, sr=sr_an, alpha=0.6, color='red', linewidth=0.5)
    plt.title('Signal anormal\n' + file_path.name)

    plt.subplot(133)
    librosa.display.waveplot(abnormal_signal, sr=sr_an, alpha=0.6, color='red', linewidth=0.5, label='Signal anormal')
    librosa.display.waveplot(normal_signal, sr=sr, alpha=0.5, color='blue', linewidth=0.5, label='Signal normal')
    plt.title('Les deux signaux')
    plt.legend();
    
    del normal_signal, sr, abnormal_signal, sr_an
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
# display_spectrum
#---------------------------------------------------------------------------------------------
def display_spectrum(anomaly, normal, size=(12, 8)):
    fig = plt.figure(figsize=size)
    fig.suptitle('Transform??e de Fourier', fontsize=16)

    plt.subplot(121)
    y, sr = load_sound_file(Path(anomaly))
    plt.plot(np.abs(librosa.stft(y)), color='red', alpha=0.6);
    plt.title(Path(normal).name + ' - Signal anormal')
    plt.xlabel('Fr??quence (Hz)')
    plt.ylabel('Amplitude')

    plt.subplot(122)
    y, sr = load_sound_file(Path(normal))
    plt.plot(np.abs(librosa.stft(y)), color='blue', alpha=0.6);
    plt.title(Path(normal).name + ' - Signal normal')
    plt.xlabel('Fr??quence (Hz)')
    plt.ylabel('Amplitude')

    del y, sr
#---------------------------------------------------------------------------------------------
   
#---------------------------------------------------------------------------------------------
# display_powerspectraldensity
#---------------------------------------------------------------------------------------------
def display_powerspectraldensity(anomaly, normal, size=(12, 8)):
    fig = plt.figure(figsize=size)
    fig.suptitle('Densit?? spectrale de puissance', fontsize=16)
  
    plt.subplot(221)
    y, _ = load_sound_file(Path(normal))
    f, Pxx_den = scipy.signal.welch(y, average='mean')
    f_med, Pxx_den_med = scipy.signal.welch(y, average='median')
    plt.semilogy(f, Pxx_den, label='mean')
    plt.semilogy(f_med, Pxx_den_med, label='median')
    plt.title(Path(normal).name + ' - Signal normal')
    plt.xlabel('Fr??quence (Hz)')
    plt.ylabel('Puissance')
    plt.legend();
    
    plt.subplot(222)
    plt.plot(f, Pxx_den, label='mean')
    plt.plot(f_med, Pxx_den_med, label='median')
    plt.title(Path(normal).name + ' - Signal normal')
    plt.xlabel('Fr??quence (Hz)')
    plt.ylabel('Puissance')
    plt.legend();

    plt.subplot(223)
    y, _ = load_sound_file(Path(anomaly))
    f, Pxx_den = scipy.signal.welch(y, average='mean')
    f_med, Pxx_den_med = scipy.signal.welch(y, average='median')
    plt.semilogy(f, Pxx_den, label='mean')
    plt.semilogy(f_med, Pxx_den_med, label='median')
    plt.title(Path(anomaly).name + ' - Signal anormal')
    plt.xlabel('Fr??quence (Hz)')
    plt.ylabel('Puissance')
    plt.legend();

    plt.subplot(224)
    plt.plot(f, Pxx_den, label='mean')
    plt.plot(f_med, Pxx_den_med, label='median')
    plt.title(Path(anomaly).name + ' - Signal anormal')
    plt.xlabel('Fr??quence (Hz)')
    plt.ylabel('Puissance')
    plt.legend();
    del y, f, f_med, Pxx_den, Pxx_den_med
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
# display_powerspectrogram
#---------------------------------------------------------------------------------------------
def display_powerspectrogram(anomaly, normal, y_axis='linear', size=(12, 8)):
    fig = plt.figure(figsize=size)
    fig.suptitle('Spectrogramme de puissance en fr??quence', fontsize=16)

    plt.subplot(121)
    y, sr = load_sound_file(Path(anomaly))
    amp_to_db = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
    librosa.display.specshow(amp_to_db, sr=sr, y_axis=y_axis,  x_axis='s');
    plt.title(str(amp_to_db.shape) + ' - Signal anormal')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(122)
    y, sr = load_sound_file(Path(normal))
    amp_to_db = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
    librosa.display.specshow(amp_to_db, sr=sr, y_axis=y_axis,  x_axis='s');
    plt.title(str(amp_to_db.shape) + ' - Signal normal')
    plt.colorbar(format='%+2.0f dB')
    del y, sr, amp_to_db
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
# display_melspectrogram
#---------------------------------------------------------------------------------------------
def display_melspectrogram(anomaly, normal, y_axis='mel', size=(12, 8)):
    fig = plt.figure(figsize=size)
    fig.suptitle('Spectrogramme Mel', fontsize=16)

    plt.subplot(121)
    y, sr = load_sound_file(Path(anomaly))
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spect = librosa.amplitude_to_db(np.abs(mel))
    librosa.display.specshow(mel_spect, sr=sr, y_axis=y_axis, x_axis='s', cmap='viridis');    
    plt.title(str(mel_spect.shape) + ' - Signal anormal')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(122)
    y, sr = load_sound_file(Path(normal))
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spect = librosa.amplitude_to_db(np.abs(mel))
    librosa.display.specshow(mel_spect, sr=sr, y_axis=y_axis, x_axis='s', cmap='viridis');
    plt.title(str(mel_spect.shape) + ' - Signal normal')
    plt.colorbar(format='%+2.0f dB')
    del y, sr
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
# display_comparisonmelspectrogram
#---------------------------------------------------------------------------------------------
def display_comparisonmelspectrogram(normal, size=(20, 20)):
    plt.figure(figsize=size)

    y, sr = load_sound_file(Path(normal))
    plt.subplot(321)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spect = librosa.amplitude_to_db(abs(mel))
    librosa.display.specshow(mel_spect, sr=sr, y_axis='log', x_axis='s', cmap='viridis');    
    plt.title('amplitude_to_db - signal normal')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(322)
    D = np.abs(librosa.stft(y))**2
    mel = librosa.feature.melspectrogram(S=D, sr=sr)
    mel_spect = librosa.power_to_db(mel, ref=np.max)
    librosa.display.specshow(mel_spect, sr=sr, y_axis='log', x_axis='s', cmap='viridis');    
    plt.title('power_to_db - signal normal')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(323)
    mel = librosa.feature.melspectrogram(y=librosa.util.normalize(y), sr=sr)
    mel_spect = librosa.amplitude_to_db(abs(mel))
    librosa.display.specshow(mel_spect, sr=sr, y_axis='log', x_axis='s', cmap='viridis');
    plt.title('amplitude_to_db - signal normal normalis??')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(324)
    D = np.abs(librosa.stft(y=librosa.util.normalize(y)))**2
    mel = librosa.feature.melspectrogram(S=D, sr=sr)
    mel_spect = librosa.power_to_db(mel, ref=np.max)
    librosa.display.specshow(mel_spect, sr=sr, y_axis='log', x_axis='s', cmap='viridis');
    plt.title('power_to_db - signal normal normalis??')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(325)
    mel = librosa.feature.melspectrogram(y=librosa.util.normalize(y), sr=sr)
    mel_spect = librosa.amplitude_to_db(abs(mel))
    mel_spect = librosa.util.normalize(mel_spect, axis=1)
    librosa.display.specshow(mel_spect, sr=sr, y_axis='log', x_axis='s', cmap='viridis');
    plt.title('amplitude_to_db - signal normal normalis?? + mel normalis??')
    plt.colorbar(format='%+2.1f dB')

    plt.subplot(326)
    D = np.abs(librosa.stft(y=librosa.util.normalize(y)))**2
    mel = librosa.feature.melspectrogram(S=D, sr=sr)
    mel_spect = librosa.power_to_db(mel, ref=np.max)
    mel_spect = librosa.util.normalize(mel_spect, axis=1)
    librosa.display.specshow(mel_spect, sr=sr, y_axis='log', x_axis='s', cmap='viridis');
    plt.title('power_to_db - signal normalis?? + mel normalis??')
    plt.colorbar(format='%+2.1f dB')
    del y, sr, mel, mel_spect, D 
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
# plot_catplot
#---------------------------------------------------------------------------------------------
def plot_catplot(dataframe, data_split=None, size=(20,10)):
    df = dataframe[['machine_id', 'machine_type', 'data_split', 'condition']]
    
    if data_split is not None:
        df = df[df.data_split == data_split]
    
    sns.catplot(x="condition", data=df, kind="count");
    
    if len(df.condition.unique()) > 1 :
        sns.catplot(x="data_split", hue="condition", data=df, kind="count");
    
    sns.catplot(x="machine_type", hue="condition", data=df, kind="count");

    sns.catplot(x="machine_type", hue="data_split", data=df, kind="count");
    
    if len(df.condition.unique()) > 1 :
        sns.catplot(x="machine_type", hue="data_split", col="condition", data=df, kind="count");

    sns.catplot(x="machine_type", hue="machine_id", col="condition", data=df, kind="count");
      
    plt.show();
    
    del df
#---------------------------------------------------------------------------------------------
#sns.displot(data=penguins, x="flipper_length_mm", hue="species", multiple="stack")
#---------------------------------------------------------------------------------------------
# get_mfcc : Generates/extracts MFCC coefficients with LibRosa 
#---------------------------------------------------------------------------------------------
def get_mfcc(file_path, hop_length=512, n_mels=64, n_mfcc=20, offset=0.0, duration=None, normalize=False):
    """ MFCC feature extractor.

    Extracts Mel-frequency cepstral coefficients (MFCCs).
    The MFCCS are calculated over the whole audio signal and then are
    separated in overlapped sequences (frames).

    Notes
    -----
    Based in `librosa.core.stft` and `librosa.filters.mel` functions.

    Parameters
    ----------
    file_path (str): a directory of audio files in .wav format
    n_mfcc : int, default=40
        Number of mel bands.
    """
    try:

        # Load audio file
        y, sr = load_sound_file(file_path, offset=offset, duration=duration)

        # Calculate MFCCs
        mfcc = librosa.feature.mfcc(y, sr=sr, hop_length=hop_length, n_mels=n_mels, n_mfcc=n_mfcc)

        if normalize:
            # Normalize MFCC between -1 and 1
            mfcc = librosa.util.normalize(mfcc, axis=1)

    except Exception as e:
        print("Erreur d'analyse du fichier audio: ", e)
        return None
    
    del y
    
    return mfcc, sr
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
# get_mel : Generates/extracts Log-MEL Spectrogram coefficients with LibRosa 
#---------------------------------------------------------------------------------------------
def get_mel(file_path, hop_length=512, n_mels=64, n_fft=2048, offset=0.0, duration=None, normalize=False):
    """ MelSpectrogram feature extractor.

    Extracts the log-scaled mel-spectrogram of the audio signals.
    The mel-spectrogram is calculated over the whole audio signal and then is
    separated in overlapped sequences (frames).

    Notes
    -----
    Based in `librosa.core.stft` and `librosa.filters.mel` functions.

    This function takes in a directory of audio files in .wav format, computes the
    mel spectrogram for each audio file, reshapes them so that they are all the 
    same size. 
    
    Parameters
    ----------
    file_path (str): a directory of audio files in .wav format
    n_mels : int, default=40
        Number of mel bands.
    
    Returns:
    mel_spectrogram (array): np.ndarray [shape=(n_mels, t)] of mel spectrogram data from the audio file in the given
    file_path
    """
    try:
        # Load audio file
        y, sr = load_sound_file(file_path, offset=offset, duration=duration)

        # Computing the mel spectrograms
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)        

        # Convert to db
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        if normalize:
            # Normalize between -1 and 1
            mel_spectrogram = librosa.util.normalize(mel_spectrogram, axis=1)

    except Exception as e:
        print("Erreur d'analyse du fichier audio: ", e)
        return None
        
    del y

    return mel_spectrogram, sr
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
# get_mfcc_mix : Generates/extracts MFCC coefficients with LibRosa 
#---------------------------------------------------------------------------------------------
def get_mfcc_mix(file_path, hop_length=512, n_mels=64, n_mfcc=20, offset=0.0, duration=None, normalize=False):
    try:
        # Load audio file
        y, sr = load_sound_file(file_path, offset=offset, duration=duration)
        
        # Compute MFCC coefficients
        mfcc = librosa.feature.mfcc(y, sr=sr, hop_length=hop_length, n_mels=n_mels, n_mfcc=n_mfcc)
        spectral_centroids = librosa.feature.spectral_centroid(y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y, sr=sr)
        rms = librosa.feature.rms(y)
        zcr1 = librosa.feature.zero_crossing_rate(y)
        chroma_stft = librosa.feature.chroma_stft(y, sr=sr)
        chroma_cqt = librosa.feature.chroma_cqt(y, sr=sr)
        chroma_cens = librosa.feature.chroma_cens(y, sr=sr)

        # Concate MFCC coefficients
        mfcc_mix = np.concatenate((mfcc, spectral_centroids, rolloff, rms, zcr1, chroma_stft, chroma_cqt, chroma_cens), axis=0)
        
        if normalize:
            # Normalize between -1 and 1
            mfcc_mix = librosa.util.normalize(mfcc_mix, axis=1)

    except Exception as e:
        print("Erreur d'analyse du fichier audio: ", e)
        return None
        
    del y, mfcc, spectral_centroids, rolloff, rms, zcr1, chroma_stft, chroma_cqt, chroma_cens
    
    return mfcc_mix, sr
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
# get_mel_mfcc : Generates/extracts MEL and MFCC coefficients with LibRosa  
#---------------------------------------------------------------------------------------------
def get_mel_mfcc(file_path, n_fft=2048, hop_length=512, n_mels=64, n_mfcc=20, offset=0.0, duration=None, normalize=False):
    try:
        # Load audio file
        y, sr = load_sound_file(file_path, offset=offset, duration=duration)
        
        # Calculate MFCCs
        mfcc = librosa.feature.mfcc(y, sr=sr, hop_length=hop_length, n_mels=n_mels, n_mfcc=n_mfcc)

        # Computing the mel spectrograms
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

        # Convert to db
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        if normalize:
            # Normalize MFCC between -1 and 1
            mfcc = librosa.util.normalize(mfcc, axis=1)
            # Normalize between -1 and 1
            mel_spectrogram = librosa.util.normalize(mel_spectrogram, axis=1)

        # Concate MFCC coefficients
        mel_mfcc = np.concatenate((mfcc, mel_spectrogram), axis=0)
        
            
    except Exception as e:
        print("Erreur d'analyse du fichier audio: ", e)
        return None 
        
    del y, mfcc, mel_spectrogram

    return mel_mfcc, sr
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
# generate_spectrograms : Generate spectrograms pictures from a list of WAV files  
#---------------------------------------------------------------------------------------------
def generate_spectrograms(list_files, output_dir, n_mels=64, n_fft=2048, hop_length=512):
    """
    Generate spectrograms pictures from a list of WAV files. Each sound
    file in WAV format is processed to generate a spectrogram that will 
    be saved as a PNG file.
    
    PARAMS
    ======
        list_files (list) - list of WAV files to process
        output_dir (string) - root directory to save the spectrogram to
        n_mels (integer) - number of Mel buckets (default: 64)
        n_fft (integer) - length of the windowed signal to compute the short Fourier transform on
        hop_length (integer) - window increment when computing STFT
        
    RETURNS
    =======
        files (list) - list of spectrogram files (PNG format)
    """
    files = []
    
    # Loops through all files:
    for index in tqdm(range(len(list_files)), desc=f'Building spectrograms for {output_dir}'):
        # Building file name for the spectrogram PNG picture:
        file = list_files[index]
        path_components = file.split('/')
        
        # machine_id = id_00, id_02...
        # sound_type = normal or abnormal
        # wav_file is the name of the original sound file without the .wav extension
        machine_id, sound_type = path_components[-3], path_components[-2]
        wav_file = path_components[-1].split('.')[0]
        filename = sound_type + '-' + machine_id + '-' + wav_file + '.png'
        
        # Example: train/normal/normal-id_02-00000259.png:
        filename = os.path.join(output_dir, sound_type, filename)

        if not os.path.exists(filename):
            # Loading sound file and generate Mel spectrogram:
            signal, sr = load_sound_file(file)
            mels = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
            mels = librosa.power_to_db(mels, ref=np.max)

            # Preprocess the image: min-max, putting 
            # low frequency at bottom and inverting to 
            # match higher energy with black pixels:
            img = scale_minmax(mels, 0, 255).astype(np.uint8)
            img = np.flip(img, axis=0)
            img = 255 - img
            img = Image.fromarray(img)

            # Saving the picture generated to disk:
            img.save(filename)

        files.append(filename)
        
    return files

#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
def mel_features_extractor(file_path, n_fft=2048, hop_length=512, n_mels=64, offset=0.0, duration=None):
    y, sr = load_sound_file(file_path, offset=offset, duration=duration)
        
    # Computing the mel spectrograms
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    # Convert to db
    mel_features = librosa.power_to_db(mel, ref=np.max)

    return mel_features
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
def mfcc_features_extractor(file_path, hop_length=512, n_mels=64, n_mfcc=20, offset=0.0, duration=None):
    y, sr = load_sound_file(file_path, offset=offset, duration=duration)
    
    mfcc_features = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mels=n_mels, n_mfcc=n_mfcc)
    
    return mfcc_features
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
def create_mel_data(DATASET_ROOT, data, file, ext, n_fft=2048, hop_length=512, n_mels=64, duration=None):
    file = file + ext
    mel_data_train_location = Path(DATASET_ROOT, file)
    
    if Path.exists(mel_data_train_location):
        print("Chargement du fichier <{}> des m??tadonn??es audio...".format(file))
        with open(mel_data_train_location, 'rb') as f:
            # Load dataset
            if ext == '.ftr':
                df = pd.read_feather(f, use_threads=True);
            else:
                df = pd.read_csv(f);
        print('Termin??.')
    else:
        start_time = time.time()
        print("Cr??ation du fichier <{}> des m??tadonn??es audio...".format(file))
        mel_data = data[['pathname','machine_type','machine_kind','data_split','condition']].copy()

        mel_data['feature'] = mel_data['pathname'].apply(lambda x : mel_features_extractor(x, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, duration=duration).flatten())
        
        # featuredf = []
        featuredf = pd.DataFrame(np.array(mel_data.feature.tolist()))
        # featuredf = featuredf[:,1:-1] #suppression d'une frame ?? gauche et ?? droite
        featuredf.rename(columns=lambda x : "mel"+str(x), inplace = True)

        df = mel_data.drop('feature', axis=1)
        df = pd.concat([df, featuredf], axis=1)
        # df = mel_data.copy()

        with open(mel_data_train_location, 'wb') as f:
            # Save dataset
            if ext == '.ftr':
                df.to_feather(f);
            else:
                df.to_csv(f, index=False);        
        print('Termin??.')
        
        del featuredf
        del mel_data
        
        end_time = time.time()
        duree = int(end_time - start_time)
        print("Cr??ation du fichier csv termin?? avec succ??s.")
        print("Dur??e : {}s ".format(datetime.timedelta(seconds = duree)))
        
    return df.sample(frac=1, random_state=6472).reset_index(drop=True)
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
def create_mfcc_data(DATASET_ROOT, data, file, ext, hop_length=512, n_mfcc=20, duration=None):
    file = file + ext
    mfcc_data_train_location = Path(DATASET_ROOT, file)
    
    if Path.exists(mfcc_data_train_location):
        print("Chargement du fichier <{}> des m??tadonn??es audio...".format(file))
        with open(mfcc_data_train_location, 'rb') as f:
            # Load dataset
            if ext == '.ftr':
                df = pd.read_feather(f, use_threads=True);
            else:
                df = pd.read_csv(f);
        print('Termin??.')                  
    else:
        start_time = time.time()
        print("Cr??ation du fichier <{}> des m??tadonn??es audio...".format(file))
        mfcc_data = data[['pathname','machine_type','machine_kind','data_split','condition']]

        mfcc_data['feature'] = mfcc_data['pathname'].apply(lambda x : mfcc_features_extractor(x, hop_length=hop_length, n_mfcc=n_mfcc, duration=duration).flatten())

        featuredf = []
#        featuredf = pd.DataFrame(np.array(mel_data.feature.tolist()))
#        featuredf = featuredf[:,1:-1] #suppression d'une frame ?? gauche et ?? droite
#        featuredf.rename(columns=lambda x : "mel"+str(x), inplace = True)
#
#        df = mel_data.drop('feature', axis=1)
#        df = pd.concat([df, featuredf], axis=1)

#        featuredf = pd.DataFrame(np.array(mfcc_data.feature.tolist()))
#        featuredf.rename(columns=lambda x : "mfcc"+str(x), inplace = True)

#        df = mfcc_data.drop('feature', axis=1)
#        df = pd.concat([df, featuredf], axis=1)
        df = mfcc_data.copy()

        with open(mfcc_data_train_location, 'wb') as f:
            # Save dataset
            if ext == '.ftr':
                df.to_feather(f);
            else:
                df.to_csv(f, index=False);        
        print('Termin??.')
        
        del featuredf
        del mfcc_data
        
        end_time = time.time()
        duree = int(end_time - start_time)
        print("Cr??ation du fichier csv termin?? avec succ??s.")
        print("Dur??e : {}s ".format(datetime.timedelta(seconds = duree)))
        
    return df.sample(frac=1, random_state=6472).reset_index(drop=True)
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
def mfcc_mean_features_extractor(file_path, hop_length=512, n_mels=64, n_mfcc=20, offset=0.0, duration=None):
    y, sr = load_sound_file(file, offset=offset, duration=duration)
    
    mfcc_features = librosa.feature.mfcc(y, sr=sr, hop_length=hop_length, n_mels=n_mels, n_mfcc=n_mfcc)
    
    return np.mean(mfcc_features,axis=1)

#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
def create_mfcc_mean_data(DATASET_ROOT, data, file, ext, hop_length=512, n_mfcc=20, duration=None):
    file = file + ext
    mfcc_data_train_location = Path(DATASET_ROOT, file)
    
    if Path.exists(mfcc_data_train_location):
        print("Chargement du fichier <{}> des m??tadonn??es audio...".format(file))
        with open(mfcc_data_train_location, 'rb') as f:
            # Load dataset
            if ext == '.ftr':
                df = pd.read_feather(f, use_threads=True);
            else:
                df = pd.read_csv(f, dtype={"machine_id": "str", "sample_id": "str"});
        print('Termin??.')                  
    else:
        start_time = time.time()
        print("Cr??ation du fichier <{}> des m??tadonn??es audio...".format(file))
        mfcc_data = data[['pathname','machine_type','machine_kind','condition']]

        mfcc_data['feature'] = mfcc_data['pathname'].apply(lambda x : mfcc_mean_features_extractor(x, hop_length=hop_length, n_mfcc=n_mfcc, duration=duration).T)

        featuredf = pd.DataFrame(np.array(mfcc_data.feature.tolist()))
        featuredf.rename(columns=lambda x : "mfcc"+str(x), inplace = True)

        df = mfcc_data.drop('feature', axis=1)
        df = pd.concat([df, featuredf], axis=1)

        with open(mfcc_data_train_location, 'wb') as f:
            # Save dataset
            if ext == '.ftr':
                df.to_feather(f);
            else:
                df.to_csv(f, index=False);        
        print('Termin??.')
        
        del featuredf
        del mfcc_data
        
        end_time = time.time()
        duree = int(end_time - start_time)
        print("Cr??ation du fichier csv termin?? avec succ??s.")
        print("Dur??e : {}s ".format(datetime.timedelta(seconds = duree)))
        
    return df.sample(frac=1, random_state=6472).reset_index(drop=True)

#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
def tsne_scatter(features, labels, dimensions=2):
    if dimensions not in (2, 3):
        raise ValueError("tsne_scatter ne peut tracer qu'en 2d ou 3d. Assurez-vous que l'argument 'dimensions' est dans (2, 3)")

    # t-SNE dimensionality reduction
    features_embedded = TSNE(n_components=dimensions, random_state=42).fit_transform(features)

    # initialising the plot
    fig, ax = plt.subplots(figsize=(10,10))
    
    # counting dimensions
    if dimensions == 3: ax = fig.add_subplot(111, projection='3d')

    # plotting data
    ax.scatter(
        *zip(*features_embedded[np.where(labels=='fan')]),
        marker='o',
        color="brown",
        s=3,
        alpha=0.7,
        label='Fan'
    )
    ax.scatter(
        *zip(*features_embedded[np.where(labels=='pump')]),
        marker='o',
        color="orange",
        s=3,
        alpha=0.9,
        label='Pump'
    )
    ax.scatter(
        *zip(*features_embedded[np.where(labels=='slider')]),
        marker='o',
        color="violet",
        s=3,
        alpha=0.9,
        label='Slider'
    )
    ax.scatter(
        *zip(*features_embedded[np.where(labels=='ToyCar')]),
        marker='o',
        color="blue",
        s=3,
        alpha=0.9,
        label='ToyCar'
    )
    ax.scatter(
        *zip(*features_embedded[np.where(labels=='ToyConveyor')]),
        marker='o',
        color="red",
        s=3,
        alpha=0.9,
        label='ToyConveyor'
    )
    ax.scatter(
        *zip(*features_embedded[np.where(labels=='valve')]),
        marker='o',
        color="green",
        s=3,
        alpha=0.9,
        label='Valve'
    )
    plt.title('t-SNE R??sultat: Audio', weight='bold').set_fontsize('14')
    plt.xlabel('Dimension 1', weight='bold').set_fontsize('10')
    plt.ylabel('Dimension 2', weight='bold').set_fontsize('10')
    # storing it to be displayed later
    plt.legend(loc='best')
    plt.show;
#---------------------------------------------------------------------------------------------
# plot_confusion_matrix
#---------------------------------------------------------------------------------------------    
def plot_confusion_matrix(y_true, y_pred, class_label=None, title=None, cmap=None, size=(8,6)):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting 'normalize=True'.
    """
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy
    
    plt.figure(figsize=size)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    if class_label is not None:
        tick_marks = np.arange(len(class_label))
        plt.xticks(tick_marks, class_label, rotation=45)
        plt.yticks(tick_marks, class_label)
        
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment='center',
                 color='white' if cm[i, j] > (cm.max() / 2.) else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\n\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
