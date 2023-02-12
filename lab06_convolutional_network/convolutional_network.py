from tensorflow import keras, reshape

# Loading the dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalizing data and one-hot encoding labels
x_train = reshape(x_train.astype('float32') / 255, (-1, 28, 28, 1))
x_test = reshape(x_test.astype('float32') / 255, (-1, 28, 28, 1))
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = keras.models.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
