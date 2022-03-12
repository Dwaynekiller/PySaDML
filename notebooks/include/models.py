#---------------------------------------------------------------------------------------------
# import keras-library
#---------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import random as rn
import itertools
from include import common
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

import librosa
import tensorflow as tf
import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, AvgPool2D, BatchNormalization, UpSampling2D
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt

# setting random seeds for libraries to ensure reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
rn.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
# autoencoder_model :   
#---------------------------------------------------------------------------------------------
def autoencoder_model(input_dims):
    """
    Defines a Keras model for performing the anomaly detection. 
    This model is based on a simple dense autoencoder.
    
    PARAMS
    ======
        inputs_dims (integer) - number of dimensions of the input features
        
    RETURN
    ======
        Model (tf.keras.models.Model) - the Keras model of our autoencoder
    """
    
    # Autoencoder definition:
    inputLayer = Input(shape=(input_dims,))
    h = Dense(64, activation="relu")(inputLayer)
    h = Dense(64, activation="relu")(h)
    h = Dense(8, activation="relu")(h)
    h = Dense(64, activation="relu")(h)
    h = Dense(64, activation="relu")(h)
    h = Dense(input_dims, activation=None)(h)

    return Model(inputs=inputLayer, outputs=h)

#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
# autoencoder :   
#---------------------------------------------------------------------------------------------
def autoencoder(shape=(224,224,1)):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    input_img = Input(shape = shape)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 256 (small and thick)
    conv4 = BatchNormalization()(conv4)
    
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    encoded = BatchNormalization()(conv4)

    #decoder
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded) #7 x 7 x 128
    conv5 = BatchNormalization()(conv5)
    
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5) #7 x 7 x 64
    conv6 = BatchNormalization()(conv6)
    
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)

    up1 = UpSampling2D((2,2))(conv6) #14 x 14 x 64
    
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 32
    conv7 = BatchNormalization()(conv7)
    
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)

    up2 = UpSampling2D((2,2))(conv7) # 28 x 28 x 32
    
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    
    model = Model(inputs = input_img, outputs = decoded)

    return model
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
# cnn_simple_model :   
#---------------------------------------------------------------------------------------------
def cnn_simple_model(shape=(224,224,1), num_labels=1):
    # Initializing a random seed for replication purposes
    np.random.seed(23456)
    K.clear_session()

    # Initialising the CNN
    model = Sequential(name='cnn_simple_model')

    # Convolution
    model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation='relu', input_shape=shape))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(BatchNormalization())

    # Adding a dropout layer for regularization
    model.add(Dropout(0.2))

    # Convolution
    model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation='relu'))
    model.add(AvgPool2D(pool_size=2, strides=2))
    model.add(BatchNormalization())

    # Flattening
    model.add(Flatten())

    # Full Connection
    model.add(Dense(units=128, activation='relu'))

    # Adding a dropout layer for regularization
    model.add(Dropout(0.25))

    if num_labels > 2:
        # Output Layer
        model.add(Dense(units=num_labels, activation='softmax'))
    else:
        # Output Layer
        model.add(Dense(units=num_labels, activation='sigmoid'))
    
    return model
#---------------------------------------------------------------------------------------------
# Insérez votre code ici 
def cnn_model(shape=(224,224,1), num_labels=1):
    # Initializing a random seed for replication purposes
    np.random.seed(23456)

    # Initialising the CNN
    model = Sequential(name='cnn_simple_model')

    # Convolution1
    model.add(Conv2D(filters=8, kernel_size=3, padding="valid", activation='relu', input_shape=shape))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Convolution2
    model.add(Conv2D(filters=16, kernel_size=3, padding="valid", activation='relu'))
    model.add(AvgPool2D(pool_size=2, strides=2))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Convolution3
    model.add(Conv2D(filters=32, kernel_size=3, padding="valid", activation='relu'))
    model.add(AvgPool2D(pool_size=2, strides=2))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Convolution4
    model.add(Conv2D(filters=64, kernel_size=3, padding="valid", activation='relu'))
    model.add(AvgPool2D(pool_size=2, strides=2))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Flattening
    model.add(Flatten())

    # Full Connection
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.2))

    if num_labels > 2:
        # Output Layer
        model.add(Dense(units=num_labels, activation='softmax'))
    else:
        # Output Layer
        model.add(Dense(units=num_labels, activation='sigmoid'))
    
    return model

#---------------------------------------------------------------------------------------------
# LeNet_model :   
#---------------------------------------------------------------------------------------------
def LeNet_model(shape=(224,224,1), num_labels=1):
    np.random.seed(23456)
    K.clear_session()
    # Initialising the CNN
    model = Sequential(name='LeNet_model')

    # Step 1 - Convolution
    model.add(Conv2D(filters=32, kernel_size=5, activation='relu', padding='valid', input_shape=shape))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    # Step 2 - Convolution
    model.add(Conv2D(filters=16, kernel_size=3, activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    # Step 3 - Flattening
    model.add(Flatten())

    # Step 4 - Adding a dropout layer for regularization
    model.add(Dropout(0.2))

    # Step 4 - Full Connection
    model.add(Dense(units=128, activation='relu'))

    if num_labels > 2:
        # Step 5 - Output Layer
        model.add(Dense(units=num_labels, activation='softmax'))
    else:
        # Step 5 - Output Layer
        model.add(Dense(units=num_labels, activation='sigmoid'))
    
    return model

#---------------------------------------------------------------------------------------------
# plot_model_history
#---------------------------------------------------------------------------------------------    
def plot_model_history(history, ylabel_metric='MSE', metrics='mean_absolute_error', size=(10,10)):
    plt.figure(figsize=size)

    hist_dict = history.history
    acc = hist_dict[metrics]
    val_acc = hist_dict['val_'+metrics]
    loss = hist_dict['loss']
    val_loss = hist_dict['val_loss']

    epochs = range(len(acc))

    # min loss / max accs
    min_loss = min(loss)
    min_val_loss = min(val_loss)
    max_accuracy = max(acc)
    max_val_accuracy = max(val_acc)

    # x pos for loss / acc min/max
    min_loss_x = loss.index(min_loss)
    min_val_loss_x = val_loss.index(min_val_loss)
    max_accuracy_x = acc.index(max_accuracy)
    max_val_accuracy_x = val_acc.index(max_val_accuracy)


    # summarize history for loss, display min
    plt.subplot(121)
    plt.plot(epochs, loss, 'bo', markersize=3, label='Training loss', color="#1f77b4")
    plt.plot(epochs, val_loss, 'b', label='Validation loss', color="#ff7f0e")
    plt.plot(min_loss_x, min_loss, marker='o', markersize=7, color="#1f77b4", label='Inline label')
    plt.plot(min_val_loss_x, min_val_loss, marker='o', markersize=7, color="#ff7f0e", label='Inline label')
    plt.title(f'Training vs validation loss by epoch', fontweight ="bold")
    plt.ylabel('Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.xticks(np.arange(0, len(loss), 5.0))
    plt.legend(['Train : loss', 'Test : val_loss', ('%.3f' % min_loss), ('%.3f' % min_val_loss)], loc='upper right', fancybox=True, framealpha=0.9, shadow=True, borderpad=1)
    plt.grid(color='grey', linestyle='--', linewidth=0.5, axis='x')

    # summarize history for accuracy, display max
    plt.subplot(122)
    plt.plot(epochs, acc, 'bo', markersize=3, label='Training accuracy', color="#1f77b4")
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy', color="#ff7f0e")
    plt.plot(max_accuracy_x, max_accuracy, marker='o', markersize=7, color="#1f77b4")
    plt.plot(max_val_accuracy_x, max_val_accuracy, marker='o', markersize=7, color="#ff7f0e")
    plt.title('Training vs validation ' + ylabel_metric + ' by epoch', fontweight ="bold")
    plt.ylabel(ylabel_metric, fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.xticks(np.arange(0, len(acc), 5.0))
    plt.legend(['Train : acc', 'Test : val_acc', ('%.2f' % max_accuracy), ('%.2f' % max_val_accuracy)], loc='lower right', fancybox=True, framealpha=0.9, shadow=True, borderpad=1)
    plt.grid(color='grey', linestyle='--', linewidth=0.5, axis='x')

    plt.show()
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
# model_metric_report
#---------------------------------------------------------------------------------------------
def model_metric_report(model, X_train=None, y_train=None, X_test=None, y_test=None,test_data=None, train_data=None):
    # Compute scores
    if (test_data is None) or (train_data is None):
        train_accuracy = model.evaluate(X_train, y_train, verbose=0)
        test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    else:
        train_accuracy = model.evaluate_generator(train_data, verbose=0)
        test_accuracy = model.evaluate_generator(test_data, verbose=0)

    # Calculate and report normalized error difference?
    max_err = max(train_accuracy[0], test_accuracy[0])
    min_err = min(train_accuracy[0], test_accuracy[0])
    error_rate = 1.0 - (min_err / max_err)

    # Pint Train vs Test report
    print('{:<10s}{:>14s}{:>14s}'.format("", "LOSS", "ACCURACY"))
    print('-' * 38)
    print('{:<10s}{:>14.4f}{:>14.2%}'.format( "Training:", train_accuracy[0], train_accuracy[1]))
    print('{:<10s}{:>14.4f}{:>14.2%}'.format( "Test:", test_accuracy[0], test_accuracy[1]))
    print('-' * 38)
    print()
    print('{:<10s}{:>13.2%}{:>1s}'.format("Error rate ", error_rate, ""))
#---------------------------------------------------------------------------------------------

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
#---------------------------------------------------------------------------------------------
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_true=Y_class, y_pred=Y_predict_class)
#
#def plot_confusion_matrix(cm,
#                          target_names,
#                          title='Confusion matrix',
#                          cmap=None,
#                          normalize=True):
#    """
#    given a sklearn confusion matrix (cm), make a nice plot
#
#    Arguments
#    ---------
#    cm:           confusion matrix from sklearn.metrics.confusion_matrix
#
#    target_names: given classification classes such as [0, 1, 2]
#                  the class names, for example: ['high', 'medium', 'low']
#
#    title:        the text to display at the top of the matrix
#
#    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
#                  see http://matplotlib.org/examples/color/colormaps_reference.html
#                  plt.get_cmap('jet') or plt.cm.Blues
#
#    normalize:    If False, plot the raw numbers
#                  If True, plot the proportions
#
#    Usage
#    -----
#    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
#                                                              # sklearn.metrics.confusion_matrix
#                          normalize    = True,                # show proportions
#                          target_names = y_labels_vals,       # list of names of the classes
#                          title        = best_estimator_name) # title of graph
#
#    Citiation
#    ---------
#    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
#
#    """
#    import matplotlib.pyplot as plt
#    import numpy as np
#    import itertools
#
#    accuracy = np.trace(cm) / np.sum(cm).astype('float')
#    misclass = 1 - accuracy
#
#    if cmap is None:
#        cmap = plt.get_cmap('Blues')
#
#    plt.figure(figsize=(8, 6))
#    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    plt.title(title)
#    plt.colorbar()
#
#    if target_names is not None:
#        tick_marks = np.arange(len(target_names))
#        plt.xticks(tick_marks, target_names, rotation=45)
#        plt.yticks(tick_marks, target_names)
#
#    if normalize:
#        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#
#
#    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
#    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#        if normalize:
#            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
#                     horizontalalignment="center",
#                     color="white" if cm[i, j] > thresh else "black")
#        else:
#            plt.text(j, i, "{:,}".format(cm[i, j]),
#                     horizontalalignment="center",
#                     color="white" if cm[i, j] > thresh else "black")
#
#
#    plt.tight_layout()
#    plt.ylabel('True label')
#    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
#    plt.show()
#---------------------------------------------------------------------------------------------
# target_metric_report
#---------------------------------------------------------------------------------------------    
def target_metric_report(y_true, y_pred, class_label):
    pd.options.display.float_format = '{:.2%}'.format
    accs = []
    cm = confusion_matrix(y_true, y_pred)
    for idx in range(0, cm.shape[0]):
        correct = cm[idx][idx].astype(int)
        total = cm[idx].sum().astype(int)
        acc = (correct / total)
        accs.append(acc)
    return pd.DataFrame({'CLASS': class_label,'ACCURACY': accs}).sort_values(by="ACCURACY", ascending=False)
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
# pred_and_plot_audio
#---------------------------------------------------------------------------------------------    
def pred_and_plot_audio(model, file_path, class_label, featuretoextract, n_mels=64, n_mfcc=20, sec_max=11.0, duration=None, normalize=False):
    """
    Imports an image located at filename, makes a prediction on it with
    a trained model and plots the image with the predicted class as the title.
    """
    # Import the target image and preprocess it
    if featuretoextract == 'mfcc':
        features, samplingrate = common.get_mfcc(file_path, hop_length=512, n_mels=64, n_mfcc=20, offset=0.0, duration=duration, normalize=normalize)
        
    if featuretoextract == 'mel':
        features, samplingrate = common.get_mel(filename, n_mels, normalize=normalize)
        
    if featuretoextract == 'mel-mfcc':
        features, samplingrate = common.get_mel_mfcc(filename, n_mfcc, n_mels, normalize=normalize)
        
    frames_max = int((sec_max * samplingrate) // 512 + 1)

    if (features.shape[0] < frames_max):
        features = librosa.util.fix_length(features, frames_max, axis=0)
    
    features = features.reshape(features.shape[0], features.shape[1], 1)
    
    # Make a prediction
    pred = model.predict(tf.expand_dims(features, axis=0))

    # Get the predicted class
    if len(pred[0]) > 1: # check for multi-class
        pred_class = class_label[pred.argmax()] # if more than one output, take the max
    else:
        pred_class = class_label[int(tf.round(pred)[0][0])] # if only one output, round

    # Plot the image and predicted class
    print(f"Prediction: {pred_class}")
    print(f"-------------------------")
    
    return common.display_audio(filename)
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
# load_and_prep_image
#---------------------------------------------------------------------------------------------    
# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_image(filename, img_shape=224, channels=3):
    """
    Reads an image from filename, turns it into a tensor
    and reshapes it to (img_shape, img_shape, colour_channel).
    """
    # Read in target file (an image)
    img = tf.io.read_file(filename)
    
    # Decode the read file into a tensor & ensure 3 colour channels 
    # (our model is trained on images with 3 colour channels and sometimes images have 4 colour channels)
    img = tf.image.decode_image(img, channels=channels)

    # Resize the image (to the same size our model was trained on)
    img = tf.image.resize(img, size = [img_shape, img_shape])

    # Rescale the image (get all values between 0 and 1)
    img = img/255.
    return img
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
# pred_and_plot_picture
#---------------------------------------------------------------------------------------------    
def pred_and_plot_picture(model, filename, class_names, img_shape=224, channels=3):
    """
    Imports an image located at filename, makes a prediction on it with
    a trained model and plots the image with the predicted class as the title.
    """
    # Import the target image and preprocess it
    img = load_and_prep_image(filename, img_shape=img_shape, channels=channels)

    # Make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))

    # Get the predicted class
    if len(pred[0]) > 1: # check for multi-class
        pred_class = class_names[pred.argmax()] # if more than one output, take the max
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])] # if only one output, round

    # Plot the image and predicted class
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False);
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
# tsne_scatter
#---------------------------------------------------------------------------------------------    
def tsne_scatter(features, labels, dimensions=2):
    if dimensions not in (2, 3):
        raise ValueError('tsne_scatter can only plot in 2d or 3d (What are you? An alien that can visualise >3d?). Make sure the "dimensions" argument is in (2, 3)')

    # t-SNE dimensionality reduction
    features_embedded = TSNE(n_components=dimensions, random_state=RANDOM_SEED).fit_transform(features)
    
    # initialising the plot
    fig, ax = plt.subplots(figsize=(10,10))
    
    # counting dimensions
    if dimensions == 3: ax = fig.add_subplot(111, projection='3d')

    # plotting data
    ax.scatter(
        *zip(*features_embedded[np.where(labels==0)]),
        marker='o',
        color='r',
        s=3,
        alpha=0.7,
        label='Anomaly'
    )
    ax.scatter(
        *zip(*features_embedded[np.where(labels==1)]),
        marker='o',
        color='g',
        s=3,
        alpha=0.9,
        label='Normal'
    )
    plt.title('t-SNE Résultat: Audio', weight='bold').set_fontsize('14')
    plt.xlabel('Dimension 1', weight='bold').set_fontsize('10')
    plt.ylabel('Dimension 2', weight='bold').set_fontsize('10')
    # storing it to be displayed later
    plt.legend(loc='best')
    plt.show;


#model = Sequential()
#
#model.add(Conv2D(32, (3, 3), padding='same',activation='relu', input_shape=INPUT_SHAPE))
#model.add(MaxPooling2D(pool_size=(2,2), padding='same')) # using pool_size (4,4) makes the layer 4x smaller in height and width
#
#model.add(Conv2D(16,(3, 3),activation='relu',  padding='same'))
#model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
#
#model.add(Conv2D(8,(3, 3),activation='relu',  padding='same'))
#model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
#
##-------------------------
#
#model.add(Conv2D(8,(3, 3),activation='relu',  padding='same'))
#model.add(UpSampling2D((2, 2)))
#
#model.add(Conv2D(16,(3, 3),activation='relu',  padding='same'))
#model.add(UpSampling2D((2, 2)))
#
#model.add(Conv2D(32,(3, 3),activation='relu',  padding='same'))
#model.add(UpSampling2D((2, 2)))
#
#model.add(Conv2D(1,(3, 3), activation='sigmoid', padding='same'))
##-------------------------
#
## Display model architecture summary 
#model.summary()