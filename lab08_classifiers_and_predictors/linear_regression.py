from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


diabetes = datasets.load_diabetes()

x_train, x_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2)

# Linear regression
model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = model.score(x_test, y_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Linear regression")
print(f"Accuracy: {accuracy:.4f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2: {r2:.4f}")