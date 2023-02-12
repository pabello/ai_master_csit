from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


iris = datasets.load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

# Logistic regression
model = LogisticRegression(max_iter=200)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = model.score(x_test, y_test)

confusion_matrix = confusion_matrix(y_test, y_pred)
precision = confusion_matrix[1,1] / (confusion_matrix[1,1] + confusion_matrix[0,1])
recall = confusion_matrix[1,1] / (confusion_matrix[1,1] + confusion_matrix[1,0])

print("Logistic regression")
print(f"Accuracy: {accuracy:.4f}")
# print(f"Precision: {precision:.2f}")
# print(f"Recall: {recall:.4f}")