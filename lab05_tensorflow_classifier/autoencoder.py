from tensorflow import keras


encoding_dim = 32

# Loading dataset
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

# Preprocessing the data
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

# Build the model
model = keras.models.Sequential([
    keras.layers.Dense(encoding_dim, activation="relu", input_shape=(784,)),
    keras.layers.Dense(784, activation="sigmoid")
])

# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")

# Train the model
model.fit(x_train, x_train, epochs=10, batch_size=128, validation_data=(x_test, x_test))