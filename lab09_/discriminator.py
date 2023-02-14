import numpy as np

from keras import backend as K
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def cnn_model(num_classes, input_shape=(32, 32, 3)):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    return model


def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[:, :128], y_pred[:, 128:256], y_pred[:, 256:]
    positive_distance = K.mean(K.square(anchor - positive), axis=-1)
    negative_distance = K.mean(K.square(anchor - negative), axis=-1)
    loss = K.maximum(positive_distance - negative_distance + alpha, 0)
    return loss


def generate_triplets(x, y, batch_size=32):
    while True:
        indices = np.random.choice(x.shape[0], batch_size, replace=False)
        anchor_positive_pairs = x[indices]
        y_pairs = y[indices]
        negative_indices = np.random.choice(x.shape[0], batch_size, replace=False)
        negative_images = x[negative_indices]
        input_data = np.concatenate((anchor_positive_pairs, anchor_positive_pairs, negative_images), axis=-1)
        target_data = np.zeros((batch_size, 1))
        yield input_data, target_data


def classifier_model(num_classes, input_shape=(32, 32, 3)):
    model = cnn_model(num_classes=1, input_shape=input_shape)
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
discriminator_model = cnn_model(num_classes=1)
discriminator_model.compile(optimizer='adam', loss=triplet_loss)

discriminator_model.fit_generator(generate_triplets(x_train, y_train),
                                  steps_per_epoch=len(y_train) // 32, epochs=10)

discriminator_model.save_weights("discriminator_weights.h5")
