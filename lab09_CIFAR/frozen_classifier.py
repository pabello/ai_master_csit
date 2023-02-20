import utils

import tensorflow_datasets as tfds

from keras.layers import Dense
from keras.losses import CategoricalCrossentropy


if __name__ == "__main__":
    train_dataset, test_dataset = tfds.load(name="cifar10", split=['train', 'test'], as_supervised=True)
    train_dataset, test_dataset = train_dataset.shuffle(1024).batch(32), test_dataset.shuffle(1024).batch(32)
    train_dataset, test_dataset = train_dataset.map(utils.normalize_img), test_dataset.map(utils.normalize_img)
    
    train_dataset = train_dataset.map(utils.one_hot_encode)
    test_dataset = test_dataset.map(utils.one_hot_encode)

    classifier = utils.initialize_discriminator()
    classifier.load_weights('discriminator_weights.h5')
    for layer in classifier.layers:
        layer.trainable = False
    classifier.add(Dense(32, activation='relu'))
    classifier.add(Dense(10, activation='softmax'))

    classifier.compile(optimizer='adam', loss=CategoricalCrossentropy(), metrics=["accuracy"])
    history = classifier.fit(train_dataset, epochs=5, validation_data=test_dataset)