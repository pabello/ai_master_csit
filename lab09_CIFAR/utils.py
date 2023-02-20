import tensorflow as tf

from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten


def normalize_img(img, label):
    img = tf.cast(img, tf.float32) / 255.
    return (img, label)


def one_hot_encode(img, label):
    label = tf.one_hot(label, 10)
    return (img, label)


def initialize_discriminator():
    return Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        Flatten(),
    ])

# def initialize_discriminator():
#     return Sequential([
#         Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
#         MaxPooling2D(pool_size=(2, 2)),
#         Conv2D(64, (3, 3), padding='same', activation='relu'),
#         MaxPooling2D(pool_size=(2, 2)),
#         Conv2D(128, (3, 3), padding='same', activation='relu'),
#         MaxPooling2D(pool_size=(2, 2)),
#         Conv2D(256, (3, 3), padding='same', activation='relu'),
#         MaxPooling2D(pool_size=(2, 2)),
#         Conv2D(64, (3, 3), padding='same', activation='relu'),
#         Flatten(),
#     ])