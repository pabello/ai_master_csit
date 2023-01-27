from sklearn.neighbors import KNeighborsClassifier

# Define the model
model = KNeighborsClassifier(n_neighbors=5)  # TODO: use argument metric = (euclidean distance)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)