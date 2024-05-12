import keras
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D,	MaxPooling2D, Flatten
from keras.optimizers import Adam

import argparse
import sys
import numpy as np
import cv2
from tqdm import tqdm


def train(mag, features, label, epoch=10):
    # Train and Validation
    n_train = int(len(label) * 0.9)
    n_test = len(label) - n_train

    x_train = np.zeros([n_train,512,512,1],dtype='float32')
    x_test  = np.zeros([n_test,512,512,1],dtype='float32')
    y_train = np.zeros([n_train],dtype='uint')
    y_test  = np.zeros([n_test],dtype='uint')

    for ii in np.arange(n_train):
      x_train[ii,:,:,0] = mag[:,:,ii]

    for ii in np.arange(n_test):
      x_test[ii,:,:,0] = mag[:,:,n_train+ii]

    y_train = label[:n_train]
    y_test = label[n_train:]

    x_train.astype('float32')
    x_train.astype('float32')

    # Convert class vectors to binary class
    y_train = to_categorical(y_train, 2)
    y_test = to_categorical(y_test, 2)

    # Noramlization
    x_train = x_train / 3000
    x_test = x_test / 3000

    # HyperParameters
    batch_size = 16
    epochs = epoch

    # CNN Model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])


    # Learning and Score
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                        verbose=1, validation_data=(x_test, y_test))

    # Epoch vs Accuracy
    import matplotlib.pyplot as plt
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    predict_classes = np.argmax(model.predict(x_test), axis=-1)
    true_classes    = np.argmax(y_test,1)
    print(confusion_matrix(true_classes, predict_classes))

    tp = confusion_matrix(true_classes, predict_classes)[0][0]
    fn = confusion_matrix(true_classes, predict_classes)[0][1]
    fp = confusion_matrix(true_classes, predict_classes)[1][0]
    tn = confusion_matrix(true_classes, predict_classes)[1][1]
    print('tp,fn,fp,tn',tp,fn,fp,tn)

    tss = tp/(tp+fn) - fp/(fp+tn)
    return tss

def main(line_of_sight_mag_filepath, features_filepath, label_solar_flare_filepath):

    # Load Data
    mag = np.load(line_of_sight_mag_filepath)
    features = list(np.load(features_filepath, allow_pickle=True).item().values())
    label = np.load(label_solar_flare_filepath)

    features = [[features[j][i] for j in range(len(features))] for i in range(len(features[0]))]

    features = np.array(features)
    print(features)
    #print(list(features.values())[1])
    #print(list(features.values())[2])
    #print(list(features.values())[3])
    #print(list(features.values())[4])

    # Check Data Shape and type
    print(mag.shape)
    print(features.shape)
    print(label.shape)
    print(type(mag.shape))
    print(type(features.shape))
    print(type(label.shape))

    tss = train(mag, features, label)
    print('tss',tss)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--line-of-sight-mag-filepath', default='train_mag_0_499.npy', help='npy file path for line-of-sight magnetogram')
    parser.add_argument('--features-filepath', default='features.npy', help='npy file path for features')
    parser.add_argument('--label-solar-flare-filepath', default='train_label_0_499.npy', help='npy file path for label of solar flare')
    #parser.add_argument('--gauss-thresh', default=140, type=int, help='Threshold of Gauss')
    #parser.add_argument('--show-imgs', action='store_true',default=False)

    args = parser.parse_args()
    print('args',args)
    line_of_sight_mag_filepath = args.line_of_sight_mag_filepath
    features_filepath = args.features_filepath
    label_solar_flare_filepath = args.label_solar_flare_filepath
    #gauss_thresh = args.gauss_thresh
    #is_show_imgs = args.show_imgs
    sys.exit(main(line_of_sight_mag_filepath, features_filepath, label_solar_flare_filepath))
