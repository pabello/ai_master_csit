import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras


dataset_builder = tfds.builder("mnist")
dataset_builder.download_and_prepare()
ds = dataset_builder.as_dataset(split="train", shuffle_files=True, as_supervised=True)

model = keras.Sequential(
    [
        keras.layers.Dense(28, name="layer1"),
        keras.layers.Dense(28, name="layer2"),
        keras.layers.Dense(10, name="layer3"),
    ]
)

ones = np.ones((28, 28))

d1 = ds.take(1)
for data, label in d1:
    print(data.shape)
    print()
    print(label)
    print()
    result = model(data)
    print(dir(result))
    print()
    print(result.shape)