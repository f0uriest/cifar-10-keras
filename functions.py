import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


def unpickle(file):
    """Get data from archives into dictionary"""
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def convert_images(raw, num_channels, img_size):
    """
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """

    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float) / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    return images


def get_input_data(batches):
    """Unpack CIFAR-10 archives and format data into ndarrays ready for TF"""

    filenames = ['./cifar-10-batches-py/data_batch_1',
                 './cifar-10-batches-py/data_batch_2',
                 './cifar-10-batches-py/data_batch_3',
                 './cifar-10-batches-py/data_batch_4',
                 './cifar-10-batches-py/data_batch_5',
                 './cifar-10-batches-py/test_batch']

    num_channels = 3
    img_size = 32
    num_categories = 10
    data = np.empty((0, img_size, img_size, num_channels))
    labels = np.empty((0, num_categories))

    for fn in [filenames[i] for i in batches]:
        d = unpickle(fn)
        batch_data = convert_images(d[b'data'], num_channels, img_size)
        batch_labels = tf.keras.utils.to_categorical(d[b'labels'])
        data = np.concatenate((data, batch_data))
        labels = np.concatenate((labels, batch_labels))

    return data, labels

def plot_train_history(history):
    '''Plot training data vs epoch number and save to file'''
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'],'b-')
    plt.plot(history.history['val_acc'],'r-')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.tight_layout()
    plt.savefig('accuracy.png')
    plt.clf()

    # Plot training & validation loss values
    plt.plot(history.history['loss'],'b-')
    plt.plot(history.history['val_loss'],'r-')
    plt.title('Model Cross Entropy Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.tight_layout()
    plt.savefig('loss.png')
    plt.clf()

def save_imgs(ims,labels):
    '''Plot a 4x4 subset of images and their labels and save to file'''
    plt.figure(1)
    k = 0
    for i in range(0,4):
        for j in range(0,4):
            plt.subplot2grid((4,4),(i,j))
            plt.imshow(ims[k], interpolation='nearest')
            plt.title(labels[k])
            k = k+1
    plt.tight_layout()
    plt.savefig('samples.png')

def plot_cm(labels_true,labels_predict):
    '''Generate a confusion matrix from true and predicted labels and save to file'''
    cm = confusion_matrix(labels_true,labels_predict)
    cmap=plt.cm.Blues
    title='Confusion matrix'
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion.png')
